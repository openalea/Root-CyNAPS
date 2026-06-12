"""
Regression test: RootNitrogenModelGraph vs RootNitrogenModel.

Both model stacks are executed in isolated subprocesses so the Choregrapher
singleton (which is global per-process) cannot cause cross-contamination.
Per-node xylem/phloem concentrations and scalar shoot-export fluxes are
compared after several timesteps.

Run from the Root-CyNAPS/test/ directory:
    pytest test_graph_regression.py -v -m slow
"""

import os
import multiprocessing as mp
import numpy as np
import pytest

SCENARIO_FILE = "inputs/Scenarios_24_06.xlsx"
SCENARIO_NAME = "Reference_Fischer"
TIME_STEP = 3600
N_STEPS = 3      # keep short; mainly checks sign / order-of-magnitude agreement
REL_TOL = 0.10   # 10 % — generous: one-timestep lag and numerical path differ
ABS_TOL = 1e-15  # absolute floor avoids divide-by-zero on near-zero values


# ──────────────────────── subprocess workers ──────────────────────────────────


def _extra_setup(g, props, vertices):
    """Patch props that are expected by the axial-transport code but that may
    be absent or incorrectly initialised in a minimal (no-soil) test stack."""
    from openalea.metafspm.utils import ArrayDict

    # Structural mass at the plant scale (default = 0 from link_self_to_mtg;
    # overwrite with the actual sum so the phloem ramp is computed correctly).
    props["total_living_struct_mass"][1] = float(
        sum(props["living_struct_mass"].values())
    )

    # Shoot structural mass — the graph model accesses this unconditionally
    # to estimate a shoot phloem volume.  Zero is safe (gives 1e-20 volume).
    if "mstruct_axis_shoot" not in props:
        props["mstruct_axis_shoot"] = {1: 0.0}

    # Phloem collar concentration: both models have a code path that reads
    # props["AA_phloem_shoot"] and divides by shoot_phloem_volume when this
    # is None.  The original model has an UnboundLocalError in that path when
    # "C_sucrose_root" is not in solute_configs.  Setting a representative
    # value here bypasses both that bug and the AA_phloem_shoot lookup.
    props["Cv_AA_phloem_collar"][1] = 0.1  # mol m-3 — typical phloem AA

    # Hexose deficit at root level — read from props as ArrayDict by both models.
    # The RSML-based scenario already provides this via a previously saved
    # simulation state.  If absent for any reason, initialise to zero.
    if "deficit_hexose_root" not in props:
        props["deficit_hexose_root"] = ArrayDict(
            {v: 0.0 for v in vertices}, dtype=float
        )
    elif not hasattr(props["deficit_hexose_root"], "values_array"):
        props["deficit_hexose_root"] = ArrayDict(
            {v: 0.0 for v in vertices}, dtype=float
        )


def _build_stack(g, root_params, nitrogen_cls):
    """Initialise growth + anatomy + water + nitrogen on *g*.

    Returns (anatomy, water, nitrogen).  Leaves the Choregrapher configured
    for the given sub-time-step.
    """
    from openalea.metafspm.component_factory import Choregrapher
    from openalea.metafspm.utils import mtg_to_arraydict
    from openalea.rootcynaps.soon_public_packages.mtg_structural_init import (
        StaticRootGrowthModel,
    )
    from openalea.rootcynaps import RootAnatomy, RootWaterModel

    Choregrapher().add_simulation_time_step(TIME_STEP)

    growth   = StaticRootGrowthModel(g=g, time_step_in_seconds=TIME_STEP, **root_params)
    anatomy  = RootAnatomy(g, TIME_STEP, **root_params)
    water    = RootWaterModel(g, TIME_STEP, **root_params)
    nitrogen = nitrogen_cls(g, TIME_STEP, **root_params)

    # Convert plain-dict MTG props to ArrayDict *after* all models have called
    # link_self_to_mtg() so their state-variable defaults are recorded.
    descriptors = anatomy.descriptor + water.descriptor + nitrogen.descriptor
    mtg_to_arraydict(g, ignore=descriptors)

    # No YAML coupling: all models share g.properties() directly.
    for m in (anatomy, water, nitrogen):
        if not hasattr(m, "pullable_inputs"):
            m.pullable_inputs = {}

    # Collar topology comes from the growth model.
    water.collar_children    = growth.collar_children
    water.collar_skip        = growth.collar_skip
    nitrogen.collar_children = growth.collar_children
    nitrogen.collar_skip     = growth.collar_skip

    props    = g.properties()
    vertices = list(g.vertices(scale=g.max_scale()))
    _extra_setup(g, props, vertices)

    # Warm up anatomy once so xylem/phloem volumes and exchange surfaces are
    # populated before the first water or nitrogen call.
    anatomy()

    return anatomy, water, nitrogen


def _run_stack(queue, test_dir, scenario_name, n_steps, use_graph):
    """Subprocess entry point.

    Loads the scenario, builds the model stack, runs *n_steps* timesteps, and
    puts a ("ok", [outputs]) or ("error", traceback_str) message on *queue*.
    """
    try:
        os.chdir(test_dir)

        from openalea.fspm.utility.scenario import MakeScenarios as ms

        scenarios   = ms.from_table(file_path=SCENARIO_FILE, which=[scenario_name])
        scenario    = scenarios[scenario_name]
        g           = scenario["input_mtg"]["root_mtg_file"]
        # Parameters are keyed by scenario row identifier (may be nan for unnamed rows)
        root_params = list(scenario["parameters"]["root_cynaps"].values())[0]

        if use_graph:
            from openalea.rootcynaps.root_nitrogen_graph import (
                RootNitrogenModelGraph as NitrogenCls,
            )
        else:
            from openalea.rootcynaps import RootNitrogenModel as NitrogenCls

        anatomy, water, nitrogen = _build_stack(g, root_params, NitrogenCls)

        props    = g.properties()
        vertices = [
            v
            for v in g.vertices(scale=g.max_scale())
            if float(props.get("struct_mass", {}).get(v, 0)) > 0
        ]

        step_outputs = []
        for _ in range(n_steps):
            anatomy()
            water()
            nitrogen()

            step_outputs.append(
                {
                    "xylem_Nm":  {v: float(props["xylem_Nm"][v])  for v in vertices},
                    "xylem_AA":  {v: float(props["xylem_AA"][v])  for v in vertices},
                    "phloem_AA": {v: float(props["phloem_AA"][v]) for v in vertices},
                    "Nm_root_to_shoot_xylem":  float(
                        props.get("Nm_root_to_shoot_xylem",  {}).get(1, 0.0)
                    ),
                    "AA_root_to_shoot_xylem":  float(
                        props.get("AA_root_to_shoot_xylem",  {}).get(1, 0.0)
                    ),
                    "AA_root_to_shoot_phloem": float(
                        props.get("AA_root_to_shoot_phloem", {}).get(1, 0.0)
                    ),
                }
            )

        queue.put(("ok", step_outputs))

    except Exception:
        import traceback

        queue.put(("error", traceback.format_exc()))


# ─────────────────────────── comparison helpers ───────────────────────────────


def _assert_close(orig, graph, label, step):
    denom = max(abs(orig), ABS_TOL)
    err   = abs(orig - graph) / denom
    assert err <= REL_TOL, (
        f"step={step} | {label}: orig={orig:.4e}  graph={graph:.4e}  "
        f"rel_err={err:.3%} > tol={REL_TOL:.0%}"
    )


def _compare_step(orig, graph, step):
    for field in ("xylem_Nm", "xylem_AA", "phloem_AA"):
        for vid in orig[field]:
            _assert_close(orig[field][vid], graph[field][vid],
                          f"{field}[vid={vid}]", step)

    for key in (
        "Nm_root_to_shoot_xylem",
        "AA_root_to_shoot_xylem",
        "AA_root_to_shoot_phloem",
    ):
        _assert_close(orig[key], graph[key], key, step)


# ──────────────────────────────── test ────────────────────────────────────────


@pytest.mark.slow
def test_nitrogen_graph_regression():
    """Graph nitrogen model outputs agree with the original within tolerance.

    Both variants are run in isolated subprocesses to prevent Choregrapher
    singleton state from leaking between the two model classes.
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    scenario_file = os.path.join(test_dir, SCENARIO_FILE)

    if not os.path.exists(scenario_file):
        pytest.skip(f"Scenario file not found: {scenario_file}")

    # Use "spawn" so each child has a clean Choregrapher singleton.
    ctx = mp.get_context("spawn")

    q_orig  = ctx.Queue()
    q_graph = ctx.Queue()

    p_orig  = ctx.Process(
        target=_run_stack,
        args=(q_orig,  test_dir, SCENARIO_NAME, N_STEPS, False),
    )
    p_graph = ctx.Process(
        target=_run_stack,
        args=(q_graph, test_dir, SCENARIO_NAME, N_STEPS, True),
    )

    p_orig.start()
    p_graph.start()
    p_orig.join(timeout=600)
    p_graph.join(timeout=600)

    assert p_orig.exitcode == 0,  "Original nitrogen stack subprocess crashed"
    assert p_graph.exitcode == 0, "Graph nitrogen stack subprocess crashed"

    status_orig,  out_orig  = q_orig.get_nowait()
    status_graph, out_graph = q_graph.get_nowait()

    if status_orig == "error":
        pytest.fail(f"Original stack raised:\n{out_orig}")
    if status_graph == "error":
        pytest.fail(f"Graph stack raised:\n{out_graph}")

    assert len(out_orig)  == N_STEPS, "Original produced wrong number of steps"
    assert len(out_graph) == N_STEPS, "Graph produced wrong number of steps"

    for step_idx, (orig, graph) in enumerate(zip(out_orig, out_graph)):
        _compare_step(orig, graph, step=step_idx + 1)
