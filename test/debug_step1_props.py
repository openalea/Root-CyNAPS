"""Capture props after step 1 for both models and compare key values."""
import os, multiprocessing as mp, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

TARGET_VID = 433
SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600


def _run(queue, use_graph):
    try:
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
        anatomy()
        water()

        # Capture pre-solve props (after @rate but before nitrogen solve)
        vertex_index = props["vertex_index"]
        focus_vids = np.asarray(props["focus_elements"], dtype=np.int64)
        focus_glob_idx = vertex_index.indices_of(props["focus_elements"])
        vid_to_loc = {int(v): i for i, v in enumerate(focus_vids)}

        captured = {"pre": {}, "post": {}}
        if TARGET_VID in vid_to_loc:
            loc = vid_to_loc[TARGET_VID]
            for key in ("export_Nm", "diffusion_Nm_xylem", "apoplastic_Nm_soil_xylem",
                        "xylem_Nm", "Nm", "axial_export_water_up_xylem",
                        "xylem_volume", "living_struct_mass", "symplasmic_volume"):
                try:
                    pdict = props.get(key, {})
                    if hasattr(pdict, "values_array"):
                        val = float(pdict.values_array()[focus_glob_idx[loc]])
                    else:
                        val = float(pdict.get(TARGET_VID, float("nan")))
                    captured["pre"][key] = val
                except Exception as e:
                    captured["pre"][key] = float("nan")

        nitrogen()

        if TARGET_VID in vid_to_loc:
            loc = vid_to_loc[TARGET_VID]
            for key in ("xylem_Nm", "Nm", "diffusion_Nm_xylem"):
                try:
                    pdict = props.get(key, {})
                    if hasattr(pdict, "values_array"):
                        val = float(pdict.values_array()[focus_glob_idx[loc]])
                    else:
                        val = float(pdict.get(TARGET_VID, float("nan")))
                    captured["post"][key] = val
                except Exception:
                    captured["post"][key] = float("nan")

            # Graph-specific: capture precomputed coefficients
            if use_graph:
                for key in ("xylem_Nm_r0", "xylem_Nm_kd", "xylem_Nm_bo", "xylem_Nm_cs"):
                    captured["post"][key] = float(props.get(key, {}).get(TARGET_VID, float("nan")))

        queue.put(("ok", captured))
    except Exception:
        import traceback
        queue.put(("error", traceback.format_exc()))


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    qo, qg = ctx.Queue(), ctx.Queue()
    po = ctx.Process(target=_run, args=(qo, False))
    pg = ctx.Process(target=_run, args=(qg, True))
    po.start(); pg.start()
    po.join(timeout=120); pg.join(timeout=120)

    so, co = qo.get_nowait()
    sg, cg = qg.get_nowait()

    if so == "error":
        print("ORIG ERROR:", co)
    if sg == "error":
        print("GRAPH ERROR:", cg)

    if so == "ok" and sg == "ok":
        print(f"\n=== vid={TARGET_VID} PRE-SOLVE (after @rate, before nitrogen solve) ===")
        print(f"{'Prop':<35} {'Original':>15} {'Graph':>15}")
        print("-" * 68)
        all_keys = sorted(set(co["pre"]) | set(cg["pre"]))
        for k in all_keys:
            ov = co["pre"].get(k, float("nan"))
            gv = cg["pre"].get(k, float("nan"))
            rel = abs(ov - gv) / max(abs(ov), 1e-30) if not (np.isnan(ov) or np.isnan(gv)) else float("nan")
            print(f"  {k:<33} {ov:>15.4e} {gv:>15.4e}  rel={rel:.4f}")

        print(f"\n=== vid={TARGET_VID} POST-SOLVE ===")
        print(f"{'Prop':<35} {'Original':>15} {'Graph':>15}")
        print("-" * 68)
        all_keys = sorted(set(co["post"]) | set(cg["post"]))
        for k in all_keys:
            ov = co["post"].get(k, float("nan"))
            gv = cg["post"].get(k, float("nan"))
            rel = abs(ov - gv) / max(abs(ov), 1e-30) if not (np.isnan(ov) or np.isnan(gv)) else float("nan")
            print(f"  {k:<33} {ov:>15.4e} {gv:>15.4e}  rel={rel:.4f}")
