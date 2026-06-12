"""
Diagnostic: Isolate the Newton step for xylem_Nm at the collar.

Run from the test/ directory:
    python debug_newton.py
"""
import os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600


def main():
    from openalea.fspm.utility.scenario import MakeScenarios as ms
    from openalea.metafspm.component_factory import Choregrapher
    from openalea.metafspm.utils import mtg_to_arraydict, ArrayDict
    from openalea.rootcynaps.soon_public_packages.mtg_structural_init import StaticRootGrowthModel
    from openalea.rootcynaps import RootAnatomy, RootWaterModel
    from openalea.rootcynaps.root_nitrogen_graph import RootNitrogenModelGraph

    scenarios = ms.from_table(file_path=SCENARIO_FILE, which=[SCENARIO_NAME])
    scenario = scenarios[SCENARIO_NAME]
    g = scenario["input_mtg"]["root_mtg_file"]
    root_params = list(scenario["parameters"]["root_cynaps"].values())[0]

    Choregrapher().add_simulation_time_step(TIME_STEP)

    growth = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
    anatomy = RootAnatomy(g, TIME_STEP, **root_params)
    water = RootWaterModel(g, TIME_STEP, **root_params)
    nitrogen = RootNitrogenModelGraph(g, TIME_STEP, **root_params)

    descriptors = anatomy.descriptor + water.descriptor + nitrogen.descriptor
    mtg_to_arraydict(g, ignore=descriptors)

    for m_inst in (anatomy, water, nitrogen):
        if not hasattr(m_inst, "pullable_inputs"):
            m_inst.pullable_inputs = {}

    water.collar_children = growth.collar_children
    water.collar_skip = growth.collar_skip
    nitrogen.collar_children = growth.collar_children
    nitrogen.collar_skip = growth.collar_skip

    props = g.properties()
    vertices = list(g.vertices(scale=g.max_scale()))
    props["total_living_struct_mass"][1] = float(sum(props["living_struct_mass"].values()))
    if "mstruct_axis_shoot" not in props:
        props["mstruct_axis_shoot"] = {1: 0.0}
    props["Cv_AA_phloem_collar"][1] = 0.1
    if "deficit_hexose_root" not in props or not hasattr(props["deficit_hexose_root"], "values_array"):
        props["deficit_hexose_root"] = ArrayDict({v: 0.0 for v in vertices}, dtype=float)

    # --- Phase 1: run anatomy + water ---
    anatomy()
    water()

    print(f"=== After anatomy+water: xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING')}")
    print(f"    axial_export_water_up_xylem[1] = {props['axial_export_water_up_xylem'].get(1, 'MISSING')}")

    # --- Phase 2: run _precompute_axial_N manually ---
    nitrogen._precompute_axial_N()

    gv = nitrogen._graph_view
    n = gv.n_nodes
    focus_vids_int = [int(v) for v in gv.node_ids]

    # Find the collar's local index
    root_local = None
    for i, vid in enumerate(focus_vids_int):
        if vid == 1:
            root_local = i
            break
    print(f"\n=== GraphView: n_nodes={n}, n_edges={gv.n_edges}")
    print(f"    collar local index = {root_local}")

    # Print collar's precomputed coefficients
    for key in ["xylem_Nm_bo", "xylem_Nm_kd", "xylem_Nm_r0", "xylem_Nm_cs", "xylem_Nm_Nold"]:
        val = props.get(key, {}).get(1, "MISSING")
        if isinstance(val, float):
            print(f"    {key}[1] = {val:.6e}")
        else:
            print(f"    {key}[1] = {val}")

    # Print key geometry at collar
    for key in ["xylem_Nm", "xylem_volume", "living_struct_mass", "axial_export_water_up_xylem"]:
        val = props.get(key, {}).get(1, "MISSING")
        if isinstance(val, float):
            print(f"    {key}[1] = {val:.6e}")
        else:
            print(f"    {key}[1] = {val}")

    # --- Phase 3: manually assemble the balance equation at the collar ---
    # Snapshot the required arrays (same way _invoke_graph_system does it)
    def snap(name):
        pdict = props.get(name, {})
        return np.asarray([float(pdict.get(v, 0.0)) for v in focus_vids_int], dtype=np.float64)

    xylem_Nm     = snap("xylem_Nm")
    xylem_volume = snap("xylem_volume")
    M            = snap("living_struct_mass")
    kd           = snap("xylem_Nm_kd")
    r0           = snap("xylem_Nm_r0")
    bo           = snap("xylem_Nm_bo")
    cs           = snap("xylem_Nm_cs")
    F_arr        = snap("axial_export_water_up_xylem")

    children = gv.tail  # local child indices per edge
    parents  = gv.head  # local parent indices per edge

    V = np.where(xylem_volume > 0, xylem_volume, 1e-30)
    C = xylem_Nm * M / V

    F    = F_arr[children]
    Fpos = np.maximum(F, 0.0)
    Fneg = np.minimum(F, 0.0)
    F_cp = (-Fneg) * C[parents] - Fpos * C[children]

    dCm = np.zeros(n, dtype=np.float64)
    np.add.at(dCm, children, F_cp)
    np.add.at(dCm, parents, -F_cp)
    dCm += r0 + kd * cs - kd * C - bo * C

    balance = -dCm / np.where(M > 0, M, 1e-30)  # = _balance return

    print(f"\n=== Manual _balance at collar (local={root_local}, vid=1) ===")
    print(f"    C[root]         = {C[root_local]:.6e} mol/m3")
    print(f"    adv_inflow      = {sum(Fpos[e]*C[children[e]] for e in range(len(children)) if parents[e]==root_local):.6e} mol/s")
    print(f"    adv_outflow_bfs = {bo[root_local]*C[root_local]:.6e} mol/s")
    print(f"    r0[root]        = {r0[root_local]:.6e} mol/s")
    print(f"    kd*cs[root]     = {kd[root_local]*cs[root_local]:.6e} mol/s")
    print(f"    kd*C[root]      = {kd[root_local]*C[root_local]:.6e} mol/s")
    print(f"    dCm[root]       = {dCm[root_local]:.6e} mol/s")
    print(f"    balance[root]   = {balance[root_local]:.6e} mol/(kg*s)")

    dt = float(TIME_STEP)
    u_old = xylem_Nm.copy()
    # Full residual at u_old: R = balance + (u_old - u_old)/dt = balance
    R_full_at_u_old = balance.copy()
    print(f"    R_full[root] at u_old = {R_full_at_u_old[root_local]:.6e}")

    # --- Phase 4: compute the FD Jacobian for just the collar row ---
    # We perturb xylem_Nm[root_local] by eps and compute the change in balance
    eps = 1e-8

    def balance_func(u):
        C_u = u * M / V
        F_cp_u = (-Fneg) * C_u[parents] - Fpos * C_u[children]
        dCm_u = np.zeros(n, dtype=np.float64)
        np.add.at(dCm_u, children, F_cp_u)
        np.add.at(dCm_u, parents, -F_cp_u)
        dCm_u += r0 + kd * cs - kd * C_u - bo * C_u
        return -dCm_u / np.where(M > 0, M, 1e-30)

    # Compute the collar row of the Jacobian analytically:
    # balance[root] = -(adv_in[root] + r0[root] + kd[root]*cs[root] - kd[root]*C[root] - bo[root]*C[root]) / M[root]
    # where adv_in = sum_e_where_parent=root Fpos[e]*C[child[e]]
    #       C[i] = u[i] * M[i] / V[i]
    # d(balance[root]) / d(u[j]):
    #   For j=root_local:
    #     d(C[root])/d(u[root]) = M[root]/V[root]
    #     d(balance[root])/d(u[root]) = -(- kd[root]*M[root]/V[root] - bo[root]*M[root]/V[root]) / M[root]
    #                                 = (kd[root]+bo[root]) * M[root] / (V[root] * M[root])
    #                                 = (kd[root]+bo[root]) / V[root]
    #   For j=child of root (edge e where parent=root):
    #     d(adv_in[root])/d(u[child[e]]) = Fpos[e] * M[child[e]] / V[child[e]]  (from upward advection)
    #     ... but wait, Fpos uses xylem_Nm[children] NOT xylem_Nm[parents].
    #     Actually for edge e where parent=root:
    #       F_cp[e] = -Fpos[e] * C[child[e]]  (when F>0, no Fneg term)
    #       dCm[root] += -F_cp[e] = Fpos[e] * C[child[e]]
    #     So d(dCm[root])/d(u[child[e]]) = Fpos[e] * M[child[e]] / V[child[e]]
    #     d(balance[root])/d(u[child[e]]) = -Fpos[e] * M[child[e]] / (V[child[e]] * M[root])

    # Let's compute this analytically:
    J_root_root_analytic = (kd[root_local] + bo[root_local]) * M[root_local] / (V[root_local] * M[root_local])
    # Wait, C[root] = u[root]*M[root]/V[root], d(-kd*C[root])/du[root] = -kd*M[root]/V[root]
    # In balance: balance[root] = -dCm[root]/M[root]
    # dCm[root] = ... - kd[root]*C[root] - bo[root]*C[root] + r0[root] + kd[root]*cs[root] + adv_in[root]
    # d(dCm[root])/d(u[root]) = (-kd[root] - bo[root]) * M[root]/V[root]
    # d(balance[root])/d(u[root]) = -(-kd[root]-bo[root])*M[root]/V[root] / M[root]
    #                              = (kd[root]+bo[root]) / V[root]
    J_root_root_analytic = (kd[root_local] + bo[root_local]) / V[root_local]

    print(f"\n=== Analytical Jacobian diagonal (collar) ===")
    print(f"    J[root,root] analytic = {J_root_root_analytic:.6e} (= (kd+bo)/V)")

    # FD estimate:
    u_p = u_old.copy(); u_p[root_local] += eps
    u_m = u_old.copy(); u_m[root_local] -= eps
    db_p = balance_func(u_p)[root_local]
    db_m = balance_func(u_m)[root_local]
    J_root_root_fd = (db_p - db_m) / (2 * eps)
    print(f"    J[root,root] FD       = {J_root_root_fd:.6e}")

    # Full Newton step for a decoupled 1x1 system at the collar (approximation):
    J_full_root = J_root_root_analytic + 1.0 / dt
    delta_root = -R_full_at_u_old[root_local] / J_full_root
    u_new_root_approx = u_old[root_local] + delta_root
    print(f"\n=== 1x1 Newton estimate (decoupled collar) ===")
    print(f"    J_full[root,root] = J_spatial + 1/dt = {J_root_root_analytic:.6e} + {1.0/dt:.6e} = {J_full_root:.6e}")
    print(f"    delta[root] = -R/J = -{R_full_at_u_old[root_local]:.6e} / {J_full_root:.6e} = {delta_root:.6e}")
    print(f"    u_new[root] approx = {u_new_root_approx:.6e}  (actual u_old = {u_old[root_local]:.6e})")

    # --- Phase 5: run the actual nitrogen solver and check result ---
    nitrogen()
    print(f"\n=== After nitrogen(): xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")
    print(f"    Expected (barely changed) ≈ {u_new_root_approx:.6e}")

    # Also check: is the residual at the true solution actually zero?
    u_new_true = snap("xylem_Nm")
    C_new = u_new_true * M / V
    F_cp_new = (-Fneg) * C_new[parents] - Fpos * C_new[children]
    dCm_new = np.zeros(n, dtype=np.float64)
    np.add.at(dCm_new, children, F_cp_new)
    np.add.at(dCm_new, parents, -F_cp_new)
    dCm_new += r0 + kd * cs - kd * C_new - bo * C_new
    balance_new = -dCm_new / np.where(M > 0, M, 1e-30)
    R_at_solution = balance_new + (u_new_true - u_old) / dt
    print(f"\n=== Residual check at reported solution ===")
    print(f"    |R|_inf at u_new = {np.linalg.norm(R_at_solution, ord=np.inf):.6e}")
    print(f"    R[root] at u_new = {R_at_solution[root_local]:.6e}")
    print(f"    u_new[root] = {u_new_true[root_local]:.6e}")
    print(f"    u_old[root] = {u_old[root_local]:.6e}")


if __name__ == "__main__":
    main()
