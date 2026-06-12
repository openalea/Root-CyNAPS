"""Compare solver coefficients (r0, kd, bo, cs) at step 1 for vid=433 or worst-error node."""
import os, multiprocessing as mp, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

TARGET_VID = 433
SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600


def _setup(root_params, g, nitrogen_cls):
    from openalea.metafspm.component_factory import Choregrapher
    from openalea.metafspm.utils import mtg_to_arraydict, ArrayDict
    from openalea.rootcynaps.soon_public_packages.mtg_structural_init import StaticRootGrowthModel
    from openalea.rootcynaps import RootAnatomy, RootWaterModel

    Choregrapher().add_simulation_time_step(TIME_STEP)
    growth   = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
    anatomy  = RootAnatomy(g, TIME_STEP, **root_params)
    water    = RootWaterModel(g, TIME_STEP, **root_params)
    nitrogen = nitrogen_cls(g, TIME_STEP, **root_params)

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
    from openalea.metafspm.utils import ArrayDict
    if "deficit_hexose_root" not in props or not hasattr(props["deficit_hexose_root"], "values_array"):
        props["deficit_hexose_root"] = ArrayDict({v: 0.0 for v in vertices}, dtype=float)

    anatomy()
    return anatomy, water, nitrogen, props


def _run_orig(queue):
    try:
        from openalea.fspm.utility.scenario import MakeScenarios as ms
        scenarios = ms.from_table(file_path=SCENARIO_FILE, which=[SCENARIO_NAME])
        scenario  = scenarios[SCENARIO_NAME]
        g         = scenario["input_mtg"]["root_mtg_file"]
        root_params = list(scenario["parameters"]["root_cynaps"].values())[0]

        from openalea.rootcynaps import RootNitrogenModel as NCls
        anatomy, water, nitrogen, props = _setup(root_params, g, NCls)

        anatomy()
        water()

        # Patch nitrogen to capture coefficients mid-solve
        orig_fn = nitrogen.axial_transport_N_arrays

        captured = {}
        def patched_fn():
            import inspect as ins
            from openalea.metafspm.utils import ArrayDict
            # Replicate the first few lines to capture props
            dt = float(nitrogen.time_step)
            vertex_index = props["vertex_index"]
            focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
            focus_glob_idx = vertex_index.indices_of(props["focus_elements"])
            n = focus_vids.size
            vid_to_loc = {int(v): i for i, v in enumerate(focus_vids)}

            # Capture the key rate values BEFORE the solve
            target = TARGET_VID
            if target in vid_to_loc:
                loc = vid_to_loc[target]
                captured["export_Nm"]  = float(props["export_Nm"].values_array()[focus_glob_idx][loc])
                captured["apoplastic_Nm_soil_xylem"] = float(props["apoplastic_Nm_soil_xylem"].values_array()[focus_glob_idx][loc])
                captured["diffusion_Nm_xylem"] = float(props["diffusion_Nm_xylem"].values_array()[focus_glob_idx][loc])
                captured["xylem_Nm_init"]   = float(props["xylem_Nm"].values_array()[focus_glob_idx][loc])
                captured["Nm_init"]     = float(props["Nm"].values_array()[focus_glob_idx][loc])
                captured["water_flux"]  = float(props["axial_export_water_up_xylem"].values_array()[focus_glob_idx][loc])
                captured["xylem_volume"] = float(props["xylem_volume"].values_array()[focus_glob_idx][loc])
                captured["struct_mass"] = float(props["living_struct_mass"].values_array()[focus_glob_idx][loc])
            orig_fn()
            if target in vid_to_loc:
                loc = vid_to_loc[target]
                captured["xylem_Nm_after"] = float(props["xylem_Nm"].values_array()[focus_glob_idx][loc])
            return

        nitrogen.axial_transport_N_arrays = patched_fn
        nitrogen()

        queue.put(("ok", captured))
    except Exception:
        import traceback
        queue.put(("error", traceback.format_exc()))


def _run_graph(queue):
    try:
        from openalea.fspm.utility.scenario import MakeScenarios as ms
        scenarios = ms.from_table(file_path=SCENARIO_FILE, which=[SCENARIO_NAME])
        scenario  = scenarios[SCENARIO_NAME]
        g         = scenario["input_mtg"]["root_mtg_file"]
        root_params = list(scenario["parameters"]["root_cynaps"].values())[0]

        from openalea.rootcynaps.root_nitrogen_graph import RootNitrogenModelGraph as NCls
        anatomy, water, nitrogen, props = _setup(root_params, g, NCls)

        anatomy()
        water()
        nitrogen()  # full step — coefficients are now in props

        target = TARGET_VID
        vertex_index = props["vertex_index"]
        focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
        focus_glob_idx = vertex_index.indices_of(props["focus_elements"])
        vid_to_loc = {int(v): i for i, v in enumerate(focus_vids)}

        captured = {}
        if target in vid_to_loc:
            loc = vid_to_loc[target]
            captured["xylem_Nm_r0"] = float(props["xylem_Nm_r0"].get(target, float("nan")))
            captured["xylem_Nm_kd"] = float(props["xylem_Nm_kd"].get(target, float("nan")))
            captured["xylem_Nm_bo"] = float(props["xylem_Nm_bo"].get(target, float("nan")))
            captured["xylem_Nm_cs"] = float(props["xylem_Nm_cs"].get(target, float("nan")))
            captured["export_Nm"]   = float(props["export_Nm"].values_array()[focus_glob_idx][loc])
            captured["diffusion_Nm_xylem"] = float(props["diffusion_Nm_xylem"].values_array()[focus_glob_idx][loc])
            captured["xylem_Nm_after"] = float(props["xylem_Nm"].values_array()[focus_glob_idx][loc])
            captured["water_flux"]  = float(props["axial_export_water_up_xylem"].values_array()[focus_glob_idx][loc])

        queue.put(("ok", captured))
    except Exception:
        import traceback
        queue.put(("error", traceback.format_exc()))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    qo, qg = ctx.Queue(), ctx.Queue()
    po = ctx.Process(target=_run_orig,  args=(qo,))
    pg = ctx.Process(target=_run_graph, args=(qg,))
    po.start(); pg.start()
    po.join(timeout=120); pg.join(timeout=120)

    _, orig  = qo.get_nowait()
    _, graph = qg.get_nowait()

    print(f"\n=== vid={TARGET_VID} at step 1 ===")
    print(f"{'Key':<35} {'Original':>15} {'Graph':>15}")
    print("-" * 70)

    if isinstance(orig, str):
        print("ORIG ERROR:", orig)
    if isinstance(graph, str):
        print("GRAPH ERROR:", graph)
    else:
        all_keys = sorted(set(orig) | set(graph))
        for k in all_keys:
            ov = orig.get(k, float("nan"))
            gv = graph.get(k, float("nan"))
            print(f"  {k:<33} {ov:>15.4e} {gv:>15.4e}")
