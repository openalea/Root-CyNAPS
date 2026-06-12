"""Run ORIGINAL model and print collar value after step 1."""
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
from openalea.rootcynaps import RootAnatomy, RootWaterModel, RootNitrogenModel

scenarios = ms.from_table(file_path=SCENARIO_FILE, which=[SCENARIO_NAME])
scenario = scenarios[SCENARIO_NAME]
g = scenario["input_mtg"]["root_mtg_file"]
root_params = list(scenario["parameters"]["root_cynaps"].values())[0]

Choregrapher().add_simulation_time_step(TIME_STEP)

growth   = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
anatomy  = RootAnatomy(g, TIME_STEP, **root_params)
water    = RootWaterModel(g, TIME_STEP, **root_params)
nitrogen = RootNitrogenModel(g, TIME_STEP, **root_params)

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

print(f"Before: xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")

anatomy()
water()
nitrogen()

print(f"After:  xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")

# Also check a child
focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
nm = props["xylem_Nm"]
vals = [f"{nm.get(int(v), 0):.3e}" for v in focus_vids[:5]]
print(f"xylem_Nm at first 5 focus nodes: {vals}")
