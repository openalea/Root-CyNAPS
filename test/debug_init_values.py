"""Check initial xylem_Nm distribution, alpha values, and solve manually."""
import os
import numpy as np
from scipy.sparse import identity, csc_matrix
from scipy.sparse.linalg import spsolve

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

# Build focus arrays
vertex_index = props["vertex_index"]
focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
focus_glob_idx = vertex_index.indices_of(props["focus_elements"])
n = focus_vids.size

def snap(name):
    pdict = props.get(name, {})
    return np.asarray([float(pdict.get(int(v), 0.0)) for v in focus_vids], dtype=np.float64)

u_init = snap("xylem_Nm")
M      = snap("living_struct_mass")
V      = snap("xylem_volume")
F_arr  = snap("axial_export_water_up_xylem")
Nm     = snap("Nm")        # symplasm concentration
V_sym  = snap("symplasmic_volume")

print(f"=== Initial xylem_Nm distribution (n={n}) ===")
print(f"  min={u_init.min():.4e}  max={u_init.max():.4e}  mean={u_init.mean():.4e}")
print(f"  First 10: {[f'{v:.3e}' for v in u_init[:10]]}")
print(f"\n=== Nm (symplasm) distribution ===")
print(f"  min={Nm.min():.4e}  max={Nm.max():.4e}  mean={Nm.mean():.4e}")
print(f"\n=== V / V_sym ===")
V_safe = np.where(V > 0, V, 1e-30)
C_init = u_init * M / V_safe
C_sym  = Nm * M / np.where(V_sym > 0, V_sym, 1e-30)
print(f"  C_xylem: min={C_init.min():.4e}  max={C_init.max():.4e}")
print(f"  C_sym:   min={C_sym.min():.4e}  max={C_sym.max():.4e}")

# Now run _precompute_axial_N to get coefficients
nitrogen._precompute_axial_N()

r0  = snap("xylem_Nm_r0")
kd  = snap("xylem_Nm_kd")
cs  = snap("xylem_Nm_cs")
bo  = snap("xylem_Nm_bo")
Nold= snap("xylem_Nm_Nold")

# Build tree topology
global2local = np.full(vertex_index.size, -1, dtype=np.int64)
global2local[focus_glob_idx] = np.arange(n, dtype=np.int64)
root_glob_idx = vertex_index.indices_of([1])[0]
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
children_arr = np.flatnonzero(parent_idx >= 0).astype(np.int64)
parents_arr  = parent_idx[children_arr]

dt = float(TIME_STEP)
m_edges = len(children_arr)
F_edge = F_arr[children_arr]
Fpos   = np.maximum(F_edge, 0.0)
Fneg   = np.minimum(F_edge, 0.0)

# Build LHS matrix (same as original model for xylem_Nm with D=0)
n_nz = n + 2 * m_edges
row_data = np.zeros(n_nz, dtype=np.float64)
col_data = np.zeros(n_nz, dtype=np.int64)
row_idx  = np.zeros(n_nz, dtype=np.int64)

# Diagonal
diag = np.zeros(n, dtype=np.float64)
diag += -kd - bo
np.add.at(diag, children_arr, -Fpos)
np.add.at(diag, parents_arr, Fneg)

# Column-scale: A_scaled[i,j] = A[i,j] / V[j]
# Diagonal: diag[i] / V[i]
diag_scaled = diag / V_safe

# Off-diagonal (child row, parent col): D - Fneg = 0 - Fneg = -Fneg
off_ip = -Fneg   # child gains from parent when F < 0 (downward)
off_ip_scaled = off_ip / V_safe[parents_arr]

# Off-diagonal (parent row, child col): D + Fpos = Fpos
off_pi = Fpos    # parent gains from child when F > 0 (upward)
off_pi_scaled = off_pi / V_safe[children_arr]

# Assemble as LHS = I - dt * A_scaled
LHS_rows = np.concatenate([np.arange(n, dtype=np.int64),
                             children_arr, parents_arr])
LHS_cols = np.concatenate([np.arange(n, dtype=np.int64),
                             parents_arr, children_arr])
LHS_vals = np.concatenate([np.ones(n) + (-dt)*diag_scaled,
                             (-dt)*off_ip_scaled, (-dt)*off_pi_scaled])

LHS = csc_matrix((LHS_vals, (LHS_rows, LHS_cols)), shape=(n, n))

# RHS: n_old + dt * R_total
n_old   = u_init * M
R_total = r0 + kd * cs   # mol/s
RHS     = n_old + dt * R_total

# Solve
n_new = spsolve(LHS, RHS)
u_new = n_new / M

print(f"\n=== Manual sparse solve (=original model logic) ===")
print(f"  u_new[root] (vid=1)  = {u_new[root]:.6e}")
print(f"  u_new distribution: min={u_new.min():.4e}  max={u_new.max():.4e}")
print(f"  First 10 u_new: {[f'{v:.3e}' for v in u_new[:10]]}")

# Residual check
C_new = u_new * M / V_safe
F_cp = (-Fneg) * C_new[parents_arr] - Fpos * C_new[children_arr]
dCm = np.zeros(n)
np.add.at(dCm, children_arr, F_cp)
np.add.at(dCm, parents_arr, -F_cp)
dCm += r0 + kd*cs - kd*C_new - bo*C_new
balance_new = -dCm / np.where(M > 0, M, 1e-30)
R_be = balance_new + (u_new - u_init) / dt
print(f"  |R_BE|_inf at solution = {np.linalg.norm(R_be, ord=np.inf):.6e}")
print(f"  R_BE[root] = {R_be[root]:.6e}")
