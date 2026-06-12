# root_water_graph.py — Implementation Plan

## What's been done

- `metafspm/test/test_graph_system_decorators.py` establishes the `@graph_system` decorator API through three use cases:
  - **UC1** `NitrogenAxialTransport` — single unknown per node, Newton, `@node_balance` + `@graph_jacobian`
  - **UC2** `WaterMunchTransport` — two coupled unknowns per node (`xylem_pressure`, `phloem_pressure`), linear system (Newton in 1 step), analytic Jacobian
  - **UC3** `MechaAnatomyHydraulics` — `BoundaryPort` Robin-penalty BCs, `@graph_output` post-solve callbacks

- `root_water.py::water_transport_munch_arrays()` (lines 653–898, decorated `@actual @rate`) is the fully working baseline. Note: the user referred to this as `axial_transport_much_array`; the actual method name is `water_transport_munch_arrays`.

- `root_water_graph.py` has been created as a **full standalone copy** of `root_water.py` with no dependency on the original. All modifications to implement the `@graph_system` API will be made exclusively in `root_water_graph.py`.

- `metafspm/src/openalea/metafspm/mpg.py` contains an `MPG(MTG)` subclass with a fixed 9-scale structure (`plant → … → segment → … → node → edge`), `remove_anchors()`, a `graph()` helper, and `array_at_scale()`. The `from_mtg` converter and `populate_node_edge_scales` methods are **not yet implemented** (proposed below).

---

## Current diagnosis

### Physics mapping to the decorator API

The Münch coupled water transport is a linear system of 2n equations (xylem pressure + phloem pressure at each of n active nodes). The existing `water_transport_munch_arrays` already assembles a `2n × 2n` sparse Jacobian `J` and solves in one step with `splu`. This maps cleanly to:

```
@graph_system(
    node_unknowns=["xylem_pressure_in", "phloem_pressure_in"],
    edge_unknowns=[],
    method="newton",   # converges in 1 step — equivalent to direct solve
    max_iter=2,
    schedule_as="axial"
)
class _transport_solve:
    @node_balance(field="xylem_pressure_in") → G_xylem residual
    @node_balance(field="phloem_pressure_in") → G_phloem residual
    @graph_jacobian                           → analytic 2n×2n J
    @graph_output                             → write 11 post-solve arrays back to props
```

The `@node_balance` residuals mirror the `G_xylem` / `G_phloem` expressions in the current code. The `@graph_jacobian` mirrors the sparse matrix already assembled there.

---

### Decisions

**D1 — `GraphView` construction via MPG** ✅ *Decided*

The long-term path is to build `GraphView` from an `MPG` that has populated node and edge scales. This requires two additions to `metafspm/mpg.py`:

**`MPG.from_mtg(mtg)` — promote an existing MTG to MPG:**
```python
@classmethod
def from_mtg(cls, mtg):
    """
    Promotes an existing MTG to an MPG in-place (no vertex copy).
    The original MTG's vertices are treated as living at the segment scale.
    MPG scale-anchor vertices are added on top of the existing structure.
    """
    obj = object.__new__(cls)
    obj.__dict__.update(mtg.__dict__)   # borrow MTG internal state
    obj.anchors = {}
    anchor = obj.root
    for scale_label, scale in cls.scales.items():
        anchor = obj.add_component(anchor, isanchor=True, label=scale_label)
        obj.anchors[scale] = anchor
    return obj
```

**`MPG.populate_node_edge_scales(focus_vids, skip_predicate=None)` — one-pass node/edge population:**
```python
def populate_node_edge_scales(self, focus_vids, skip_predicate=None):
    """
    Populates the node and edge scales from parent-child topology at segment scale.

    focus_vids      : iterable of segment-scale VIDs to include as graph nodes
    skip_predicate  : callable(vid) → bool; returns True for structural connector
                      segments (e.g. Support_for_seminal_root) that are excluded
                      from node/edge scales. Their children are re-parented upward
                      to the nearest non-skipped functional ancestor.

    After this call:
    - Each included vid has a vertex at node_scale with property 'vertex_id' = vid
    - Each parent→child connection between included vids has a vertex at edge_scale
      with properties 'n_id_a' (parent vid) and 'n_id_b' (child vid)

    Note: cell-scale use cases will populate axial connections differently
    (e.g. plasmodesmata between adjacent cells rather than parent-child MTG edges).
    """
    skip = skip_predicate or (lambda v: False)
    node_anchor = self.anchors[self.scales["node"]]
    edge_anchor = self.anchors[self.scales["edge"]]

    seg_to_node = {}
    for vid in focus_vids:
        if not skip(vid):
            nv = self.add_component(node_anchor, label="node")
            self.property("vertex_id")[nv] = vid
            seg_to_node[vid] = nv

    # is_collar: boolean prop on segment VIDs, used as a `types` filter by
    # @node_balance methods to partition bulk physics from collar BCs.
    # Collar = vid whose functional parent (after skip walk) is None.
    collar_prop = self.properties().setdefault("is_collar", {})

    for vid in seg_to_node:
        p = self.parent(vid)
        while p is not None and (skip(p) or p not in seg_to_node):
            p = self.parent(p)
        if p is not None:
            ev = self.add_component(edge_anchor, label="edge")
            self.property("n_id_a")[ev] = p
            self.property("n_id_b")[ev] = vid
            collar_prop[vid] = False
        else:
            collar_prop[vid] = True   # no functional parent → collar node
```

Once these two methods exist, `_rebuild_graph_view()` in `root_water_graph.py` calls `MPG.from_mtg(self.g).populate_node_edge_scales(focus_vids, skip_predicate)`, then passes the MPG to `GraphView.from_mtg_subset()`.

**D2 — `collar_children` topology skip** ✅ *Decided*

At node and edge scales, segments of type `Support_for_seminal_root` / `Support_for_adventitious_root` are ignored. The `skip_predicate` passed to `populate_node_edge_scales` encodes this: `skip_predicate = lambda vid: vid in self.collar_children`. Their children are automatically re-parented to the collar node (vid=1) by the parent-walk loop above.

**D3 — Dynamic growing architecture** ✅ *Decided: option (a)*

`focus_elements` grows each timestep. A `@stepinit` method `_update_graph_view` calls `_rebuild_graph_view()` each timestep, before the `axial` step. `post_coupling_init` calls it once at startup.

**D4 — Collar boundary conditions** ✅ *Decided*

The collar BC is expressed as a **separate `@node_balance` method** with a `types` filter, not as an if/else branch inside the bulk physics method. The `types` filter mechanism is already implemented in `graph_system_decorators.py` (lines 64–79 and 235–244): `types={"prop": ["value"]}` restricts a method to matching nodes and scatters its result additively back into the full residual vector. For **disjoint** type filters (collar vs non-collar), additive scatter is equivalent to partitioning — each node's residual comes from exactly one method.

```python
@node_balance(field="xylem_pressure_in", types={"is_collar": [False]})
def _xylem_bulk_balance(self, xylem_pressure_in, phloem_pressure_in, K_xylem, ...):
    B = self._graph_view.incidence
    return np.asarray(B @ diags(K_xylem) @ B.T @ xylem_pressure_in).reshape(-1) + ...  # radial terms

@node_balance(field="xylem_pressure_in", types={"is_collar": [True]})
def _xylem_collar_bc(self, xylem_pressure_in, collar_pressure_xylem):
    # Default: Dirichlet pressure BC.
    # To switch to flux BC, replace this method body with the flux equation.
    return xylem_pressure_in - collar_pressure_xylem
```

`is_collar` is a boolean node property (`True` only at vid=1) added to `props` during `populate_node_edge_scales` (see Step 0 below). It is snapshotted as part of the normal prop snapshot; `_type_mask` evaluates `v in {True}` per node.

`collar_pressure_xylem` (and its phloem counterpart) are node props pre-populated each timestep in `_update_graph_view` from the existing collar pressure logic — they are zero everywhere except vid=1, so the sub-array received by `_xylem_collar_bc` (after the mask) is a length-1 vector.

The `@boundary_condition(location, kind, field)` decorator that also exists in `graph_system_decorators.py` is a lower-level escape hatch for BCs that cannot be expressed as a filtered `@node_balance`; it is not needed here.

**D5 — `K_xylem` / `K_phloem` edge location** ✅ *Decided*

Declare with `location="edge"`. Since `edge_ids = child_vids` in the `GraphView`, and `props['K_xylem'][child_vid]` is already populated by the existing `_K_xylem` / `_K_phloem` rate methods, the snapshot aligns correctly today.

Longer-term: yes, progressively migrating segment-scale computations (conductances, lengths, transport parameters) to the MPG edge scale is the right direction. The `@state` decorator could accept `location="edge"` to write results directly to edge-scale vertices, making the mapping explicit. This is a future enhancement to `component_factory.py` — not in scope for this reimplementation but worth tracking.

**D6 — Osmotic terms as pre-computed node props** ✅ *Decided*

Pre-compute `osmotic_xylem_term = reflection_xylem * RT * (Cv_soil - Cv_xylem)` and `osmotic_phloem_term = reflection_phloem * RT * (Cv_phloem - Cv_xylem)` in a `@rate` method that runs before `axial`. They appear as snapshotted node props in `@node_balance` arguments. Dynamic solute-water coupling (if needed) will be a separate model.

---

## Next steps

Implementation order (each step verifiable in isolation):

### Step 0 — Extend `metafspm/mpg.py`
Add `MPG.from_mtg(mtg)` classmethod and `MPG.populate_node_edge_scales(focus_vids, skip_predicate)` method (signatures and docstrings above). No changes to existing MPG methods.

### Step 1 — Declare new fields in `RootWaterGraphModel` (`root_water_graph.py`)
- Add `osmotic_xylem_term`, `osmotic_phloem_term` as `state_variable`, `location="node"`
- Ensure `K_xylem`, `K_phloem` have `location="edge"` in `declare()`

### Step 2 — Implement `_rebuild_graph_view()`
- Identify `collar_children` (support segments) as `skip_predicate`
- Call `MPG.from_mtg(self.g).populate_node_edge_scales(focus_vids, skip_predicate)`
  - This populates `is_collar` on segment VIDs in `props` as a side-effect
- Pass the populated MPG to `GraphView.from_mtg_subset()` to get `self._graph_view`

### Step 3 — Add `post_coupling_init` and `@stepinit _update_graph_view`
- `post_coupling_init` calls `_rebuild_graph_view()` once at startup
- `@stepinit` `_update_graph_view` calls it each timestep (runs before `axial`)

### Step 4 — Add `@rate _osmotic_terms`
- Compute `osmotic_xylem_term` and `osmotic_phloem_term` per node
- Write to `self.props` entries, keyed by VID

### Step 5 — Add `@graph_system _transport_solve` block

Each field gets **two** `@node_balance` methods, partitioned by `types={"is_collar": [...]}`:

| Method | `types` filter | Physics |
|--------|---------------|---------|
| `_xylem_bulk_balance` | `{"is_collar": [False]}` | Laplacian + radial exchange + osmotic terms |
| `_xylem_collar_bc` | `{"is_collar": [True]}` | Default: Dirichlet `P_xy = collar_pressure_xylem` |
| `_phloem_bulk_balance` | `{"is_collar": [False]}` | Laplacian + symplasmic exchange + osmotic terms |
| `_phloem_collar_bc` | `{"is_collar": [True]}` | Default: Dirichlet `P_ph = collar_pressure_phloem` |

To change xylem or phloem collar BC type, rewrite only the corresponding `_*_collar_bc` method body — bulk physics is untouched. The framework's additive scatter on disjoint sets makes this safe.

Add also:
- `@graph_jacobian` → analytic 2n×2n Jacobian (same block structure as current sparse matrix)
- `@graph_output` → compute and write back all 11 post-solve arrays (axial fluxes, radial fluxes, total flows)

### Step 6 — Remove `water_transport_munch_arrays`
Replace the `@actual @rate` decorated monolithic method with the new `@graph_system` block.

### Step 7 — Equivalence verification
Run both `RootWaterModel` (from `root_water.py`) and `RootWaterGraphModel` (from `root_water_graph.py`) on the same MTG fixture for 24 timesteps. Assert max absolute difference < 1e-8 on all 11 output arrays.

---

### Future enhancements (out of scope here)
- `@state` decorator `location="edge"` to write conductance computations directly to edge scale
- Dynamic solute-water coupling in a separate model class
- `MPG.populate_node_edge_scales` variant for cell-scale use cases (explicit plasmodesmata / symplastic connections, not parent-child MTG edges)
