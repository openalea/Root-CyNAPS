"""
Compare source terms (r0, kd*cs) and alpha_c between original and graph models.
Run BEFORE nitrogen() to see what the precomputed coefficients are.
"""
import os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600

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
growth   = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
anatomy  = RootAnatomy(g, TIME_STEP, **root_params)
water    = RootWaterModel(g, TIME_STEP, **root_params)
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

# Check initial values of metabolic props
print("=== Initial metabolic prop values (before nitrogen()) ===")
for pname in ["export_Nm", "apoplastic_Nm_soil_xylem", "diffusion_Nm_xylem"]:
    if pname in props:
        vals = list(props[pname].values())
        arr = np.array([float(v) for v in vals if isinstance(v, (int, float))])
        if len(arr) > 0:
            print(f"  {pname}: min={arr.min():.4e}  max={arr.max():.4e}  sum={arr.sum():.4e}")
        else:
            print(f"  {pname}: present but no numeric values")
    else:
        print(f"  {pname}: NOT IN PROPS")

# Run precompute to get the r0 and kd values
nitrogen._precompute_axial_N()

vertex_index = props["vertex_index"]
focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
focus_glob_idx = vertex_index.indices_of(props["focus_elements"])
n = focus_vids.size

def snap(name):
    pdict = props.get(name, {})
    return np.asarray([float(pdict.get(int(v), 0.0)) for v in focus_vids], dtype=np.float64)

r0  = snap("xylem_Nm_r0")
kd  = snap("xylem_Nm_kd")
cs  = snap("xylem_Nm_cs")
bo  = snap("xylem_Nm_bo")
M   = snap("living_struct_mass")
V   = snap("xylem_volume")
F   = snap("axial_export_water_up_xylem")
u   = snap("xylem_Nm")

print(f"\n=== Precomputed coefficients summary ===")
print(f"  r0: min={r0.min():.4e}  max={r0.max():.4e}  nonzero={np.sum(r0 != 0)}/{n}")
print(f"  kd: min={kd.min():.4e}  max={kd.max():.4e}")
print(f"  cs: min={cs.min():.4e}  max={cs.max():.4e}")
print(f"  bo: min={bo.min():.4e}  max={bo.max():.4e}  nonzero={np.sum(bo > 0)}/{n}")
print(f"  kd*cs: min={(kd*cs).min():.4e}  max={(kd*cs).max():.4e}  sum={(kd*cs).sum():.4e}")
print(f"  r0+kd*cs: sum={(r0+kd*cs).sum():.4e}")

# Build tree
parent_vid_global = props["parent_id"].values_array()
parent_vid_focus  = parent_vid_global[focus_glob_idx]
has_parent = parent_vid_focus >= 0

global2local = np.full(vertex_index.size, -1, dtype=np.int64)
global2local[focus_glob_idx] = np.arange(n, dtype=np.int64)
root_glob_idx = vertex_index.indices_of([1])[0]
root = int(global2local[root_glob_idx])

parent_idx = np.full(n, -1, dtype=np.int64)
parent_glob_idx = vertex_index.indices_of(parent_vid_focus[has_parent]).astype(np.int64)
parent_loc = global2local[parent_glob_idx]
child_loc  = np.flatnonzero(has_parent)
valid = parent_loc >= 0
parent_idx[child_loc[valid]] = parent_loc[valid]
children_arr = np.flatnonzero(parent_idx >= 0).astype(np.int64)
parents_arr  = parent_idx[children_arr]

dt = float(TIME_STEP)

# For each collar child: compute 1D steady state and backward Euler solution
collar_children_edges = np.where(parents_arr == root)[0]
print(f"\n=== Collar's {len(collar_children_edges)} children: source vs sink ===")
for e in collar_children_edges:
    c = children_arr[e]
    vid_c = int(focus_vids[c])
    Fpos_c = max(F[c], 0.0)
    alpha_c = (kd[c] + bo[c] + Fpos_c) / max(V[c], 1e-30)
    source_c = r0[c] + kd[c] * cs[c]  # mol/s
    # steady state (dt→∞): u_ss = source / (M * alpha) = (source/M) / alpha
    # but source/M in mol/(kg*s) and alpha in 1/s... not directly comparable
    # In moles: n_ss = (source * V_c) / (kd+bo+Fpos) = source/alpha_c * V_c / V_c
    # Actually: at steady state, d(n)/dt = 0: source = (kd+bo+Fpos)*C_ss
    # → C_ss = source / (kd+bo+Fpos) → u_ss = C_ss * V_c / M_c
    C_ss = source_c / max(kd[c] + bo[c] + Fpos_c, 1e-30)  # mol/m3
    u_ss = C_ss * V[c] / max(M[c], 1e-30)  # mol/kg
    # Backward Euler (1D decoupled):
    # (u_new - u_old)/dt = (source - (kd+bo+Fpos)*C_new) / M
    # u_new * M / dt + (kd+bo+Fpos)*M/V * u_new = u_old*M/dt + source
    # u_new = (u_old/dt + source/M) / (1/dt + (kd+bo+Fpos)/V)
    u_new_c = (u[c]/dt + source_c/max(M[c], 1e-30)) / (1.0/dt + alpha_c)
    print(f"  vid={vid_c}: r0={r0[c]:.3e} kd*cs={kd[c]*cs[c]:.3e} "
          f"source={source_c:.3e} alpha_c={alpha_c:.3e} "
          f"u_ss={u_ss:.3e} u_new_1D={u_new_c:.3e}")

print(f"\n=== Collar (root={root}, vid=1): source vs sink ===")
source_root = r0[root] + kd[root] * cs[root]
alpha_root = (kd[root] + bo[root]) / max(V[root], 1e-30)
C_ss_root = source_root / max(kd[root] + bo[root], 1e-30)
u_ss_root = C_ss_root * V[root] / max(M[root], 1e-30)
u_new_root_1D = (u[root]/dt + source_root/max(M[root], 1e-30)) / (1.0/dt + alpha_root)
print(f"  r0={r0[root]:.3e} kd*cs={kd[root]*cs[root]:.3e} "
      f"source={source_root:.3e} alpha={alpha_root:.3e} "
      f"u_ss={u_ss_root:.3e} u_new_1D={u_new_root_1D:.3e}")

# Compare with original model: check what export_Nm and apoplastic look like
print(f"\n=== export_Nm and apoplastic_Nm at collar's children ===")
for e in collar_children_edges:
    c = children_arr[e]
    vid_c = int(focus_vids[c])
    exp_nm = float(props.get("export_Nm", {}).get(vid_c, "N/A") if "export_Nm" in props else "N/A")
    apo    = float(props.get("apoplastic_Nm_soil_xylem", {}).get(vid_c, "N/A") if "apoplastic_Nm_soil_xylem" in props else "N/A")
    diff   = float(props.get("diffusion_Nm_xylem", {}).get(vid_c, "N/A") if "diffusion_Nm_xylem" in props else "N/A")
    print(f"  vid={vid_c}: export_Nm={exp_nm:.3e}  apoplastic={apo:.3e}  diffusion_Nm_xylem={diff:.3e}")
