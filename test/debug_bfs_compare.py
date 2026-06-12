"""
Compare the BFS boundary_outflow computed by:
  (a) the graph model's _bfs_collar_outflow (correct)
  (b) the original model's inline BFS (potentially buggy ordering)

Run as:   python debug_bfs_compare.py
"""
import os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600


def build_tree_arrays(g, root_vid=1, vertex_index=None):
    """Return focus_vids, children, parents, adj, edge_parent, offsets."""
    props = g.properties()
    if vertex_index is None:
        vertex_index = props["vertex_index"]
    focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
    focus_glob_idx = vertex_index.indices_of(props["focus_elements"])
    n = focus_vids.size

    global2local = np.full(vertex_index.size, -1, dtype=np.int64)
    global2local[focus_glob_idx] = np.arange(n, dtype=np.int64)

    root_glob_idx = vertex_index.indices_of([root_vid])[0]
    root = int(global2local[root_glob_idx])

    parent_vid_global = props["parent_id"].values_array()
    parent_vid_focus  = parent_vid_global[focus_glob_idx]
    has_parent = parent_vid_focus >= 0

    parent_idx = np.full(n, -1, dtype=np.int64)
    parent_glob_idx = vertex_index.indices_of(parent_vid_focus[has_parent]).astype(np.int64)
    parent_loc = global2local[parent_glob_idx]
    child_loc  = np.flatnonzero(has_parent)
    valid = parent_loc >= 0
    parent_idx[child_loc[valid]] = parent_loc[valid]

    children = np.flatnonzero(parent_idx >= 0).astype(np.int64)
    parents  = parent_idx[children]

    counts  = np.bincount(parents, minlength=n).astype(np.int32)
    offsets = np.empty(n + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    order = np.argsort(parents, kind="stable")
    adj        = children[order]
    edge_parent = parents[order]

    return focus_vids, children, parents, adj, edge_parent, offsets, root, n


def bfs_graph_model(root, adj, edge_parent, offsets,
                    water_flux, conductive_element_volume, dt):
    """Exact copy of _bfs_collar_outflow from root_nitrogen_graph.py."""
    n = conductive_element_volume.size
    sgn_root = np.sign(water_flux[root]) if water_flux[root] != 0.0 else 1.0
    Q_aligned = np.where(sgn_root * water_flux > 0.0, np.abs(water_flux), 0.0)

    vol_budget = np.abs(water_flux[root]) * dt
    Q_edge = Q_aligned[adj]
    m_edges = Q_edge.size
    prefix = np.empty(m_edges + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(Q_edge, out=prefix[1:])
    sum_Q_parent = prefix[offsets[1:]] - prefix[offsets[:-1]]

    in_budget = np.zeros(n, dtype=np.float64)
    in_budget[root] = vol_budget
    adv_vol   = np.zeros(n, dtype=np.float64)
    time_rem  = np.zeros(n, dtype=np.float64)
    time_rem[root] = dt
    crossing_time = np.zeros(n, dtype=np.float64)
    parent_is_active = np.zeros(n, dtype=bool)

    while True:
        active = np.flatnonzero(in_budget > 0.0)
        if active.size == 0:
            break
        adv_here = np.minimum(in_budget[active], conductive_element_volume[active])
        adv_vol[active] += adv_here
        wf_safe = np.where(np.abs(water_flux[active]) > 0.0,
                           np.abs(water_flux[active]), 1.0)
        crossing_time[active] = np.maximum(
            time_rem[active] - 0.5 * adv_vol[active] / wf_safe, 0.0)
        out_here = in_budget[active] - adv_here

        out_full = np.zeros(n, dtype=np.float64)
        out_full[active] = out_here
        parent_is_active[:] = False
        parent_is_active[active] = True

        out_e   = out_full[edge_parent]
        sumQ_e  = sum_Q_parent[edge_parent]
        edge_ok = (parent_is_active[edge_parent]
                   & (Q_edge > 0.0) & (sumQ_e > 0.0) & (out_e > 0.0))

        child_recv = np.zeros(m_edges, dtype=np.float64)
        child_recv[edge_ok] = out_e[edge_ok] * Q_edge[edge_ok] / sumQ_e[edge_ok]

        next_in = np.zeros(n, dtype=np.float64)
        np.add.at(next_in, adj, child_recv)
        in_budget = next_in

        if edge_ok.any():
            ep_ok  = edge_parent[edge_ok]
            ch_ok  = adj[edge_ok]
            wf_ep  = np.where(np.abs(water_flux[ep_ok]) > 0.0,
                               np.abs(water_flux[ep_ok]), 1.0)
            time_rem[ch_ok] = np.maximum(
                time_rem[ep_ok] - adv_vol[ep_ok] / wf_ep, 0.0)

    denom = adv_vol.sum()
    if denom > 0.0:
        return np.maximum(0.0, water_flux) * np.clip(crossing_time / dt, 0.0, 1.0)
    out = np.zeros(n, dtype=np.float64)
    out[root] = water_flux[root]
    return out


def bfs_original_model(root, adj, edge_parent, offsets,
                       water_flux, conductive_element_volume, dt):
    """
    Replicate the original model's BFS from root_nitrogen.py lines 1330-1411.
    Key difference: time_remaining is updated using boolean index (potentially wrong order).
    """
    n = conductive_element_volume.size
    sgn_root = np.sign(water_flux[root]) if water_flux[root] != 0.0 else 1.0
    Q_down = np.where(sgn_root * water_flux > 0.0, np.abs(water_flux), 0.0)

    vol_budget0 = np.abs(water_flux[root]) * dt

    Q_child_edge = Q_down[adj]
    m_edges = Q_child_edge.size
    prefix = np.empty(m_edges + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(Q_child_edge, out=prefix[1:])
    sum_child_Q = prefix[offsets[1:]] - prefix[offsets[:-1]]

    in_budget  = np.zeros(n, dtype=np.float64)
    in_budget[root] = vol_budget0
    adv_vol    = np.zeros(n, dtype=np.float64)
    time_remaining_after_parent = np.zeros(n, dtype=np.float64)
    time_remaining_after_parent[root] = dt
    crossing_time = np.zeros(n, dtype=np.float64)

    ct = 0
    while True:
        active_parents = np.flatnonzero(in_budget > 0.0)
        if active_parents.size == 0:
            break

        adv_here = np.minimum(in_budget[active_parents], conductive_element_volume[active_parents])
        adv_vol[active_parents] += adv_here
        crossing_time[active_parents] = np.maximum(
            time_remaining_after_parent[active_parents]
            - (0.5 * adv_vol[active_parents] / np.abs(water_flux[active_parents])),
            0.)
        out_here = in_budget[active_parents] - adv_here

        out_full = np.zeros(n, dtype=np.float64)
        out_full[active_parents] = out_here
        parent_is_active = np.zeros(n, dtype=bool)
        parent_is_active[active_parents] = True

        out_edge  = out_full[edge_parent]
        sumQ_edge = sum_child_Q[edge_parent]
        edge_ok   = (parent_is_active[edge_parent]
                     & (Q_child_edge > 0.0) & (sumQ_edge > 0.0) & (out_edge > 0.0))

        child_in_edge = np.zeros_like(out_edge)
        child_in_edge[edge_ok] = out_edge[edge_ok] * (Q_child_edge[edge_ok] / sumQ_edge[edge_ok])

        next_in_budget = np.zeros(n, dtype=np.float64)
        np.add.at(next_in_budget, adj, child_in_edge)

        in_budget = next_in_budget

        # ORIGINAL MODEL: uses boolean mask on in_budget (wrong order vs edge_ok order)
        if edge_ok.any():
            time_remaining_after_parent[in_budget > 0.0] = np.maximum(
                time_remaining_after_parent[edge_parent[edge_ok]]
                - (adv_vol[edge_parent[edge_ok]] / np.abs(water_flux[edge_parent[edge_ok]])),
                0.)

        ct += 1

    denom = adv_vol.sum()
    if denom > 0.0:
        return np.maximum(0.0, water_flux) * np.clip(crossing_time / dt, 0.0, 1.0)
    out = np.zeros(n, dtype=np.float64)
    out[root] = water_flux[root]
    return out


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

    growth  = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
    anatomy = RootAnatomy(g, TIME_STEP, **root_params)
    water   = RootWaterModel(g, TIME_STEP, **root_params)
    nitrogen = RootNitrogenModelGraph(g, TIME_STEP, **root_params)

    descriptors = anatomy.descriptor + water.descriptor + nitrogen.descriptor
    mtg_to_arraydict(g, ignore=descriptors)

    for m in (anatomy, water, nitrogen):
        if not hasattr(m, "pullable_inputs"):
            m.pullable_inputs = {}

    water.collar_children    = growth.collar_children
    water.collar_skip        = growth.collar_skip
    nitrogen.collar_children = growth.collar_children
    nitrogen.collar_skip     = growth.collar_skip

    props = g.properties()
    vertices = list(g.vertices(scale=g.max_scale()))
    props["total_living_struct_mass"][1] = float(sum(props["living_struct_mass"].values()))
    if "mstruct_axis_shoot" not in props:
        props["mstruct_axis_shoot"] = {1: 0.0}
    props["Cv_AA_phloem_collar"][1] = 0.1
    if "deficit_hexose_root" not in props or not hasattr(props["deficit_hexose_root"], "values_array"):
        props["deficit_hexose_root"] = ArrayDict({v: 0.0 for v in vertices}, dtype=float)

    anatomy()
    water()

    # Build shared tree arrays
    vertex_index = props["vertex_index"]
    focus_vids, children, parents, adj, edge_parent, offsets, root, n = \
        build_tree_arrays(g, root_vid=1, vertex_index=vertex_index)

    focus_glob_idx = vertex_index.indices_of(props["focus_elements"])

    water_flux = props["axial_export_water_up_xylem"].values_array()[focus_glob_idx]
    xylem_vol  = props["xylem_volume"].values_array()[focus_glob_idx]

    dt = float(TIME_STEP)

    bo_graph = bfs_graph_model(root, adj, edge_parent, offsets,
                                water_flux, xylem_vol, dt)
    bo_orig  = bfs_original_model(root, adj, edge_parent, offsets,
                                   water_flux, xylem_vol, dt)

    diff = bo_graph - bo_orig
    nonzero_diff = np.flatnonzero(np.abs(diff) > 1e-30)

    print(f"=== BFS boundary_outflow comparison ===")
    print(f"  n_nodes = {n}")
    print(f"  bo_graph sum  = {bo_graph.sum():.6e}   bo_orig sum  = {bo_orig.sum():.6e}")
    print(f"  bo_graph[collar] = {bo_graph[root]:.6e}   bo_orig[collar] = {bo_orig[root]:.6e}")
    print(f"  |diff|_max = {np.abs(diff).max():.6e}")
    print(f"  nodes where diff≠0: {len(nonzero_diff)} of {n}")

    # Show first 10 differing nodes
    for i in nonzero_diff[:10]:
        vid = int(focus_vids[i])
        print(f"  node {i} (vid={vid}): bo_graph={bo_graph[i]:.4e}  bo_orig={bo_orig[i]:.4e}  diff={diff[i]:+.4e}")

    # Show children of collar
    collar_children_edges = np.where(parents == root)[0]
    print(f"\n=== Collar's {len(collar_children_edges)} direct children ===")
    for e in collar_children_edges:
        c = children[e]
        vid_c = int(focus_vids[c])
        print(f"  child (vid={vid_c}): F={water_flux[c]:.4e} V={xylem_vol[c]:.4e} "
              f"bo_graph={bo_graph[c]:.4e}  bo_orig={bo_orig[c]:.4e}")

    # Compute effective alpha for collar's children
    print(f"\n=== α_c for collar's direct children (should be >> 1/dt=2.78e-4) ===")
    kd_arr = np.asarray([float(props.get("xylem_Nm_kd", {}).get(int(focus_vids[i]), 0.0)) for i in range(n)])
    for e in collar_children_edges:
        c = children[e]
        vid_c = int(focus_vids[c])
        alpha_graph = (kd_arr[c] + bo_graph[c] + water_flux[c]) / xylem_vol[c]
        alpha_orig  = (kd_arr[c] + bo_orig[c]  + water_flux[c]) / xylem_vol[c]
        u_old_c = float(props.get("xylem_Nm", {}).get(int(focus_vids[c]), 1e-7))
        u_new_graph = u_old_c / (dt * alpha_graph + 1)
        u_new_orig  = u_old_c / (dt * alpha_orig + 1)
        print(f"  child (vid={vid_c}): α_graph={alpha_graph:.4e}  α_orig={alpha_orig:.4e}  "
              f"u_new_graph≈{u_new_graph:.4e}  u_new_orig≈{u_new_orig:.4e}")


if __name__ == "__main__":
    main()
