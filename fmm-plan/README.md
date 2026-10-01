# nd-fmm-plan

The kernel-agnostic plan of a distributed fast multipole method, on top of an
[`nd-octree`](../octree) octree. In development; the API is unstable.

`nd-fmm-plan` decides which boxes interact, in which order, and when data crosses MPI
ranks. It does no kernel arithmetic: translation operators plug in through a trait, and
`nd-fmm-exec` provides the Laplace operator.

- **Box index.** Every box a rank holds gets a dense, Morton-ordered `u32` index per
  level; no hot path looks a key up in a map.
- **Interaction lists.** The U-, V-, W- and X-lists and the parent/child relations as
  index arrays, held once per target (CSR) and once grouped by V-list offset or child
  octant.
- **Data stores.** Multipoles and locals as one buffer per level; source data, target
  input and target output as CSR leaf stores with a variable number of points per leaf.
- **Ghost exchange.** Sources in one exchange for all levels, multipoles per level, and
  a gather of the coarse blocks for the global upward pass.
- **Evaluator.** The distributed pass order, with one operator call per level and kind
  (`FmmOperator`), and a per-pair adapter (`PairOperator`, `PerPair`) for simple
  operators.
- **Index FMM.** A test FMM that propagates leaf indices, which checks the complete
  distributed compute graph exactly.

```rust,ignore
let plan = Plan::new(&octree)?; // octree built with ghost children; collective
let mut evaluator = Evaluator::new(&plan, comm, operator, &source_counts, &target_counts)?;
// fill local_sources_mut() and local_target_inputs_mut() per leaf, then
evaluator.evaluate(); // collective
let output = evaluator.target_output(leaf); // leaf index, as numbered by the plan
```

The crate documentation (`cargo doc -p nd-fmm-plan --open`) describes the compute graph,
the accumulation order and the collectives. The design is
[docs/design/fmm-plan-redesign.md](../docs/design/fmm-plan-redesign.md).

Licensed under MIT or Apache-2.0, at your option.
