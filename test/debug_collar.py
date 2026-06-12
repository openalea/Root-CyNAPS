"""
Debug script: print collar-node coefficients for both original and graph models.
Run in isolation (one model per invocation) to avoid Choregrapher contamination.

Usage:
    python debug_collar.py graph    # prints graph model collar values
    python debug_collar.py orig     # prints original model collar values
"""

import os, sys
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600


def load_scenario():
    from openalea.fspm.utility.scenario import MakeScenarios as ms
    scenarios = ms.from_table(file_path=SCENARIO_FILE, which=[SCENARIO_NAME])
    scenario  = scenarios[SCENARIO_NAME]
    g         = scenario["input_mtg"]["root_mtg_file"]
    root_params = list(scenario["parameters"]["root_cynaps"].values())[0]
    return g, root_params


def build_stack(g, root_params, nitrogen_cls):
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
    if "deficit_hexose_root" not in props or not hasattr(props["deficit_hexose_root"], "values_array"):
        props["deficit_hexose_root"] = ArrayDict({v: 0.0 for v in vertices}, dtype=float)

    anatomy()
    return anatomy, water, nitrogen


def run_graph():
    from openalea.rootcynaps.root_nitrogen_graph import RootNitrogenModelGraph
    g, root_params = load_scenario()
    anatomy, water, nitrogen = build_stack(g, root_params, RootNitrogenModelGraph)

    props = g.properties()

    # Patch _precompute_axial_N to print collar coefficients
    orig_precompute = RootNitrogenModelGraph._precompute_axial_N

    def patched_precompute(self):
        orig_precompute(self)
        root_vid = 1
        print(f"\n=== GRAPH MODEL collar (vid=1) coefficients (after _precompute_axial_N) ===")
        for key in ["xylem_Nm_kd", "xylem_Nm_r0", "xylem_Nm_bo", "xylem_Nm_cs", "xylem_Nm_Nold"]:
            val = self.props.get(key, {}).get(root_vid, "MISSING")
            print(f"  {key}[1] = {val:.6e}" if isinstance(val, float) else f"  {key}[1] = {val}")
        # Also print water flux at collar and its direct children
        collar_wf = self.props.get("axial_export_water_up_xylem", {}).get(root_vid, "MISSING")
        print(f"  axial_export_water_up_xylem[1] = {collar_wf:.6e}" if isinstance(collar_wf, float) else f"  ...[1] = {collar_wf}")
        # Print xylem_Nm at collar
        nm_collar = self.props.get("xylem_Nm", {}).get(root_vid, "MISSING")
        print(f"  xylem_Nm[1] = {nm_collar:.6e}" if isinstance(nm_collar, float) else f"  xylem_Nm[1] = {nm_collar}")
        # print graph_view info
        if hasattr(self, "_graph_view"):
            gv = self._graph_view
            print(f"  n_nodes={gv.n_nodes}, n_edges={gv.n_edges}")
            # Find root local index
            root_local = None
            for li, vid in enumerate(gv.node_ids):
                if int(vid) == root_vid:
                    root_local = li
                    break
            if root_local is not None:
                print(f"  root local index = {root_local}")
                # Find edges where root is parent (head)
                root_children_edges = np.where(gv.head == root_local)[0]
                print(f"  root's child edges: {root_children_edges[:5]} (total {len(root_children_edges)})")

    RootNitrogenModelGraph._precompute_axial_N = patched_precompute

    print(f"Initial xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")
    anatomy()
    water()
    nitrogen()
    print(f"\nAfter step 1 xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")


def run_orig():
    from openalea.rootcynaps import RootNitrogenModel
    g, root_params = load_scenario()
    anatomy, water, nitrogen = build_stack(g, root_params, RootNitrogenModel)

    props = g.properties()

    # Patch axial_transport_N_arrays to print collar coefficients
    orig_axial = RootNitrogenModel.axial_transport_N_arrays

    def patched_axial(self):
        import inspect as ins

        # Reproduce the preamble to get collar index
        p = self.props
        from openalea.metafspm.utils import ArrayDict
        focus_vids = np.asarray(p["focus_elements"], dtype=np.int64)
        vertex_index = p["vertex_index"]
        focus_glob_idx = vertex_index.indices_of(p["focus_elements"])
        n = focus_vids.size
        global2local = np.full(vertex_index.size, -1, dtype=np.int64)
        global2local[focus_glob_idx] = np.arange(n, dtype=np.int64)
        root_vid = 1
        root_glob_idx = vertex_index.indices_of([root_vid])[0]
        root = int(global2local[root_glob_idx])

        orig_axial(self)  # run the original

        # Print what was computed for the collar
        print(f"\n=== ORIGINAL MODEL collar (vid=1, local={root}) ===")
        print(f"  focus_vids[:3] = {focus_vids[:3]}")
        print(f"  root local index = {root}")

    RootNitrogenModel.axial_transport_N_arrays = patched_axial

    print(f"Initial xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")
    anatomy()
    water()
    nitrogen()
    print(f"\nAfter step 1 xylem_Nm[1] = {props['xylem_Nm'].get(1, 'MISSING'):.6e}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "graph"
    if mode == "graph":
        run_graph()
    else:
        run_orig()
