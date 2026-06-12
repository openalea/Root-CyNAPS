"""Compare per-step per-node errors between original and graph nitrogen models."""
import os, multiprocessing as mp
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600
N_STEPS = 3

def _run(queue, use_graph):
    from openalea.fspm.utility.scenario import MakeScenarios as ms
    from openalea.metafspm.component_factory import Choregrapher
    from openalea.metafspm.utils import mtg_to_arraydict, ArrayDict
    from openalea.rootcynaps.soon_public_packages.mtg_structural_init import StaticRootGrowthModel
    from openalea.rootcynaps import RootAnatomy, RootWaterModel

    scenarios = ms.from_table(file_path=SCENARIO_FILE, which=[SCENARIO_NAME])
    scenario  = scenarios[SCENARIO_NAME]
    g         = scenario["input_mtg"]["root_mtg_file"]
    root_params = list(scenario["parameters"]["root_cynaps"].values())[0]

    Choregrapher().add_simulation_time_step(TIME_STEP)

    growth   = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
    anatomy  = RootAnatomy(g, TIME_STEP, **root_params)
    water    = RootWaterModel(g, TIME_STEP, **root_params)

    if use_graph:
        from openalea.rootcynaps.root_nitrogen_graph import RootNitrogenModelGraph as NCls
    else:
        from openalea.rootcynaps import RootNitrogenModel as NCls

    nitrogen = NCls(g, TIME_STEP, **root_params)

    descriptors = anatomy.descriptor + water.descriptor + nitrogen.descriptor
    mtg_to_arraydict(g, ignore=descriptors)

    for m in (anatomy, water, nitrogen):
        if not hasattr(m, "pullable_inputs"):
            m.pullable_inputs = {}

    water.collar_children    = growth.collar_children
    water.collar_skip        = growth.collar_skip
    nitrogen.collar_children = growth.collar_children
    nitrogen.collar_skip     = growth.collar_skip

    props    = g.properties()
    vertices = list(g.vertices(scale=g.max_scale()))
    props["total_living_struct_mass"][1] = float(sum(props["living_struct_mass"].values()))
    if "mstruct_axis_shoot" not in props:
        props["mstruct_axis_shoot"] = {1: 0.0}
    props["Cv_AA_phloem_collar"][1] = 0.1
    if "deficit_hexose_root" not in props or not hasattr(props["deficit_hexose_root"], "values_array"):
        props["deficit_hexose_root"] = ArrayDict({v: 0.0 for v in vertices}, dtype=float)

    anatomy()

    filt_v = [v for v in g.vertices(scale=g.max_scale())
              if float(props.get("struct_mass", {}).get(v, 0)) > 0]

    steps = []
    for _ in range(N_STEPS):
        anatomy()
        water()
        nitrogen()
        steps.append({
            f: {v: float(props[f][v]) for v in filt_v}
            for f in ("xylem_Nm", "xylem_AA", "phloem_AA")
        })

    queue.put(("ok", steps))

import numpy as np

if __name__ == "__main__":
 ctx = mp.get_context("spawn")
 qo, qg = ctx.Queue(), ctx.Queue()
 po = ctx.Process(target=_run, args=(qo, False))
 pg = ctx.Process(target=_run, args=(qg, True))
 po.start(); pg.start()
 po.join(timeout=300); pg.join(timeout=300)

 _, orig_steps  = qo.get_nowait()
 _, graph_steps = qg.get_nowait()

 ABS_TOL = 1e-15
 for step_i, (orig, graph) in enumerate(zip(orig_steps, graph_steps)):
     print(f"\n=== Step {step_i+1} ===")
     for field in ("xylem_Nm", "xylem_AA", "phloem_AA"):
         errs = []
         worst_vid, worst_err, worst_o, worst_g = None, 0, 0, 0
         for vid in orig[field]:
             o, g2 = orig[field][vid], graph[field][vid]
             denom = max(abs(o), ABS_TOL)
             err = abs(o - g2) / denom
             errs.append(err)
             if err > worst_err:
                 worst_err, worst_vid, worst_o, worst_g = err, vid, o, g2
         arr = np.array(errs)
         print(f"  {field}: max={arr.max():.4f} (vid={worst_vid}, orig={worst_o:.4e}, graph={worst_g:.4e})"
               f"  p90={np.percentile(arr,90):.4f}  mean={arr.mean():.4f}  n_fail={np.sum(arr>0.10)}")
