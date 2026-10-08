# nd-fmm-plan

The root `CLAUDE.md` applies as well; it holds every workspace-wide rule. This file
adds only what is specific to this crate.

## Project

`nd-fmm-plan` (Cargo package `nd-fmm-plan`, Rust crate `nd_fmm_plan`, Rust 2024) plans the
**topology and data flow** of a fast multipole method on top of an
`nd_octree::Octree`: a Morton-ordered integer index of the boxes, the interaction lists
as index arrays, the data stores and ghost exchanges, and the distributed evaluation
order with a level-batched operator interface. It does **not** own point coordinates,
kernel evaluations, or translation operators (M2M / M2L / L2L / near-field arithmetic)
— analytical and kernel-independent FMMs are meant to share this layer. Keep
kernel-specific arithmetic out. The design is `docs/design/fmm-plan-redesign.md`
(signed off; the decisions are recorded in its §12).

Real FMM operators belong outside this crate (the Laplace operator in `nd-fmm-exec`)
and plug in through the `FmmOperator` trait, or `PairOperator` with the `PerPair`
adapter. The crate's own tests use test operators whose values do not matter: they
record and check the calls, the batches and the slices they are handed. The values of
the distributed passes are checked by the Laplace FMM in `nd-fmm-exec`.

- Library only, no binaries.
- Version `0.1.0-dev`; the public API is unstable. Redistribution of points to their
  owning ranks is designed (design §9) but not implemented (C5.1).

## Code map

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate doc (module overview, the compute graph, future extensions) and the `pub mod` lines. |
| `src/index.rs` | `BoxIndex`, `LeafNumbering`: Morton-ordered `u32` box indices per level for every held key (local, `Global`, ghost), and the leaf numbering (local leaves by (level, key), then the ghost leaves named by U or X, by key). |
| `src/interaction_manager.rs` | `V_LIST_DIRECTIONS` (the 316 V-list offsets, CONVENTIONS §3.12) and the private per-key list rule `key_lists`, with its helpers. See the section below. |
| `src/lists.rs` | `Csr`, `GroupedCsr` as `VList`/`Children`/`Parents`, `LevelLists` (the views of one level), `offset_index`, `NOFFSETS`, `NOCTANTS`. |
| `src/plan.rs` | `Plan::new` (collective), `Plan::from_key_types` (local), `PlanError`. |
| `src/store.rs` | `LevelBuffers`, `LeafStore` and their slice types: one buffer per level kind, CSR leaf data with variable counts. |
| `src/exchange.rs` | `SourceExchange`, `MultipoleExchange`, `CoarseExchange`, `ExchangeError`: variable-size source exchange into the ghost tail, per-level multipole exchange, coarse-block gather. |
| `src/operator.rs` | `FmmSizes`, the level-batched `FmmOperator` with its batch types `P2m` … `P2p` and `UpwardPass`, the per-pair `PairOperator` and its `PerPair` adapter. |
| `src/evaluator.rs` | `Evaluator`, `EvaluatorError`: the pass order on the plan, stores and exchanges, public stages, debug-checked stage order. |
| `src/interaction_manager_tests.rs` | 7 serial unit tests of the per-key rule on synthetic key maps (hand counts, W/X on a refined octant, brute-force oracle, invariants, adjacency, ghost classification, V-list offsets), through the test helper `ListMaps`, included via `#[path]`. No MPI. |
| `src/plan_tests.rs` | 12 serial unit tests of the index and the index-form lists against `ListMaps` on the trees of `interaction_manager_tests.rs`. No MPI. |
| `src/store_tests.rs`, `src/exchange_tests.rs` | 9 serial unit tests of the store layouts and 6 of the ghost bucketing and per-key chunk sizes (on the trees of `plan_tests.rs`). No MPI. |
| `src/operator_tests.rs`, `src/evaluator_tests.rs` | 3 serial unit tests of the `PerPair` adapter on hand-made batches, and 4 of the evaluator passes: a marker operator checks that every batch hands the views and slices of its level and leaves (variable counts, on the ghost-free trees of `plan_tests.rs`), reset and repeat, call order, validation errors. No MPI. |
| `tests/mpi_regressions.rs` | One `#[test]` that owns MPI init and runs 12 named scenarios sequentially. Each builds an `Octree` from the union of a source and a target point set (with the ghost-children layer) and computes brute-force U/V/W/X lists of every non-ghost key over the gathered global tree (`oracle_lists`). It then builds a `Plan` and checks the numbering, the leaves, the index-form lists against the oracle and the view invariants; runs the exchanges (sources with counts of one, `hash(key) % 5` and the real source points per leaf; multipoles with per-level sizes, exactly the ghosts of the oracle's V- and W-lists; the coarse gather); and runs the `Evaluator`: with `hash(key) % 5` counts a per-pair operator whose values do not matter (`Mixer`) must give bit-identical stores on a second evaluation and on an evaluator that owns its plan; and a recording operator checks the call order, the groupings of every batch and that every oracle pair is issued exactly once. On the graded and the dense-leaf scenarios rank 0 prints the exchange traffic (`--nocapture`). |
| `examples/plan_build_cost.rs` | Release-mode timing and heap use of `Plan::new`, on one rank (nothing asserted). |

Read `src/plan.rs`, `src/lists.rs`, `src/evaluator.rs` and `tests/mpi_regressions.rs`
for how the API actually behaves; prose is a summary, they are the contract.

### Navigating the code

Follow the root navigation rules (LSP first; `rlst` sources in the registry cache, not
`../rlst`). `nd-octree` is the workspace sibling at `../octree/src/…`. Within this
crate's small source modules (plus their `*_tests.rs` files), reading a whole
file beats either LSP or `grep`.

## The interaction lists

`Plan::from_key_types` applies the private per-key rule `interaction_manager::key_lists`
to every non-ghost key of `octree.all_keys()` and translates the lists to indices. The
rule is **local — no collectives**; `Plan::new` adds two all-reduces (validation agreed
on every rank). The module docs of `src/interaction_manager.rs` and `src/lists.rs` are
the contract; the points that matter most when changing them:

- **Rows for every box.** A box view of level l has one row per held box of the level,
  ghosts included (with empty rows); a leaf view has one row per local leaf of the
  level. Inapplicable lists (U and W for interior boxes, V and X at levels 0 and 1)
  are empty rows. Rows never contain the box itself, except the near list (U plus the
  leaf itself, design §12 decision 1a).
- **Adjacency** means the closed cubes share at least a vertex and neither box
  is an ancestor of the other. Boxes may be at different levels.
- **It relies on the octree's guarantees**: complete, 2:1 balanced across all
  26 directions, plus the one-cell same-level ghost halo. Those let the rule decide
  everything from a neighbour's leaf/interior classification and Morton
  arithmetic without looking children up. If `nd-octree` ever relaxes balance
  or shrinks the halo, the `.expect`/`debug_assert`s in the rule fire.
- **Every listed key is a held key**, with its own `KeyType` — and, for a ghost, its
  owner rank via `KeyType::ghost_rank`. This holds only when the octree is built with
  `OctreeOptions::with_ghost_children(true)`; `Plan::new` returns
  `PlanError::MissingGhostChildren` otherwise, and checks every entry, every child of a
  non-ghost interior box and every own coarse block. `Plan::from_key_types` is what
  `src/plan_tests.rs` drives with hand-built maps.
- **Listed boxes are routinely ghosts** — V and W entries are children of remote
  neighbours. `exchange` moves their data; the plan provides topology only.

## Dependencies

- All dependencies come from the root `[workspace.dependencies]`, and the package
  fields `edition`, `license`, `homepage` and `repository` from `[workspace.package]`.
  The crate adds the features it needs: `rlst` with `mpi`, `mpi` with `complex` (the
  workspace entry has `derive`).
- `nd-octree` is the workspace path dependency on `../octree`. Keep it that way: do not
  switch to a git or crates.io source, which can diverge from the workspace copy.
- `rlst` 0.9.0 and `mpi` 0.8.2 (rsmpi) come from crates.io. The crate uses rlst's
  `distributed_tools::{GhostCommunicator, GhostCommunicatorBuilder, ChunkSizes}` in
  `exchange.rs`, and `distributed_tools::array_tools::gather_to_all` in the tests. Tests and examples also use `rlst_dynamic_array` and `println_mpi`;
  `rand_chacha` is a dev-dependency.

## Crate checks

The root checks cover this crate. For this crate alone, add `-p nd-fmm-plan` (e.g.
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan`).

`cargo test -p nd-fmm-plan` gives 41 unit tests plus the integration test **on one rank
only**. It exercises no ghost exchange and no ghost layer, which is where the
interesting bugs are.

### Multi-rank runs

For any change touching the plan, the lists, the exchanges or the evaluator (whose
ghost-dependent and global-level paths only exist on more than one rank), build the
test binary and launch it under MPI, under an external timeout:

```sh
cargo test -p nd-fmm-plan --test mpi_regressions --no-run   # note the executable path it prints
RUST_MIN_STACK=8388608 timeout 600 mpiexec -n 2 target/debug/deps/mpi_regressions-<hash> --test-threads=1
RUST_MIN_STACK=8388608 timeout 600 mpiexec -n 4 target/debug/deps/mpi_regressions-<hash> --test-threads=1
```

Use the executable Cargo prints, not the sibling `.d` file, and note the hash
changes on rebuild.

On macOS, add the loopback flags from the root `CLAUDE.md`; with them, 2 and 4 ranks
pass:

```sh
mpiexec --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 4 <exe> --test-threads=1
```

`tests/mpi_regressions.rs` is the crate's multi-rank run. No example of this crate is
registered for the weekly `run-examples` job; to have one run there at 3 ranks,
register it with `[[example]]` and `[package.metadata.example.<name>.templated-examples]`
as the root `CLAUDE.md` describes.

## Octree input and MPI discipline

Octree fine keys must be valid Morton keys at level `DEEPEST_LEVEL` (16) —
`Octree::new` validates this and panics naming the offending key, so build them
with `points_to_morton(&points, DEEPEST_LEVEL as usize, &bbox)`. Point arrays
are `[3, npoints]`, one point per **column**; `PhysicalBox` is
`[xmin, ymin, zmin, xmax, ymax, zmax]`. Build the octree with
`OctreeOptions::with_ghost_children(true)`; `Plan::new` requires it.

**MPI discipline.** The root MPI rules apply; several test scenarios exist precisely
to cover empty and uneven ranks. `tests/mpi_regressions.rs` owns the one MPI
initialisation for `tests/`; do not add a second independently initialising test
there. Globally empty octree construction is not covered by any test and should not
be assumed to work.

## Conventions

- Match the existing module layout and rustdoc style. Cite the design as
  `design §x.y` (`docs/design/fmm-plan-redesign.md`) and conventions as
  `CONVENTIONS §3.x`.
- Keep topology tests deterministic and free of translation operators. Serial
  helper tests go next to the helpers (`src/*_tests.rs`, included with `#[path]`);
  anything needing MPI goes in `tests/` or `examples/`.
- Existing coverage: interaction lists — serial: hand-counted U/V sizes on a uniform
  level-2 tree, W/X on a one-octant-refined tree, a brute-force geometric oracle on
  adaptive trees, list invariants, the adjacency helper, ghost classification, and the
  index-form views against the per-key rule; multi-rank: every
  `tests/mpi_regressions.rs` scenario compares all four index-form lists of every local
  non-ghost box against a brute-force oracle over the gathered global tree. Most
  scenarios only reach leaf level 3; `graded corner blob` adds a dense corner at leaf
  level 6 against coarse remote neighbours, and `dense max-level leaf` puts 1,000
  duplicate points into one leaf at `max_level`, so one source chunk dwarfs the others.
  Every scenario also runs the exchanges, a repeated evaluation and the recording
  operator. Scenarios
  cover refinement caps, duplicate keys, empty populations, and uneven and empty ranks.
- Add new `tests/mpi_regressions.rs` scenarios to the existing `cases` array rather
  than as new `#[test]` functions, so MPI ownership stays with one test. The scenario
  set must not shrink.
- Keep the local `../octree` path dependency unless asked to change it.
- Phase 3 rebuilt this crate (docs/phase3/README.md, T1 and T4–T7): the old map-based
  API (`InteractionManager`, `FmmGhostCommunicator`, `FmmEvaluator`, `LevelData`, the
  per-pair `FmmOperator`) was removed in T7, after T6 showed that the new evaluator
  reproduces it on every scenario. Changes now go to the code as it is; there is no
  parallel reference implementation to keep.
- Phase 5 T2 removed the index FMM (a test FMM that propagated leaf indices) with its
  tests and examples. Do not add a replacement test FMM that propagates values exactly:
  the values of the passes are `nd-fmm-exec`'s to check, against the direct sum and, on
  several ranks, against one rank (docs/phase5/README.md).
