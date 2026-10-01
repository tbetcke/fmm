# nd-fmm-plan

The root `CLAUDE.md` applies as well; it holds every workspace-wide rule. This file
adds only what is specific to this crate.

## Project

`nd-fmm-plan` (Cargo package `nd-fmm-plan`, Rust crate `nd_fmm_plan`, Rust 2024) plans the
**topology and data flow** of a fast multipole method on top of an
`nd_octree::Octree`: the interaction lists, the ghost exchange, and the
distributed evaluation order. It does **not** own point
coordinates, kernel evaluations, or translation operators (M2M / M2L / L2L /
near-field arithmetic) — analytical and kernel-independent FMMs are meant to
share this layer. Keep kernel-specific arithmetic out.

The one exception is `IndexFmm` in `src/fmm/index_fmm.rs`. It is a test FMM,
not a real kernel: its operators propagate leaf indices so the distributed FMM
topology workflow (upward and downward passes, the global levels, ghost
exchanges, U/V/W/X lists) can be checked exactly. It is included only to test
those workflows. Real FMM operators belong outside this crate and plug in
through the `FmmOperator` trait.

- Library only, no binaries.
- Version `0.1.0-dev`; the public API is unstable and partly unimplemented.
- The `repository` field in `Cargo.toml` still points at the former standalone
  repository `nd-project/nd-plan`, and `homepage` has a typo (`codeberg.com.com`).

## Code map

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate doc plus `pub mod fmm;`, `pub mod ghost_communicator;` and `pub mod interaction_manager;`. |
| `src/interaction_manager.rs` | `InteractionManager` — U/V/W/X lists for every non-ghost key of an `Octree`, computed locally. See the section below. |
| `src/interaction_manager_tests.rs` | 9 serial unit tests on synthetic key maps (hand counts, brute-force oracle, adjacency, V-list directions), included via `#[path]`. No MPI. |
| `src/ghost_communicator.rs` | `FmmGhostCommunicator<T>` — one rlst `GhostCommunicator<MortonKey>` per level with owned host send/receive buffers, built from the Morton keys of required ghosts (local and `Global` keys skipped) and a `LevelChunkSizes` (uniform or per level). |
| `src/ghost_communicator_tests.rs` | 5 serial unit tests of the ghost bucketing and chunk-size lookup, included via `#[path]`. No MPI. |
| `src/fmm.rs`, `src/fmm/` | Distributed FMM evaluation: the `FmmOperator` trait (`operator.rs`), the generic driver `FmmEvaluator` and per-level store `LevelData` (`evaluator.rs`), and the index-propagating test FMM `IndexFmm` / `run_index_fmm` (`index_fmm.rs`). Serial tests in `evaluator_tests.rs` and `index_fmm_tests.rs`. |
| `tests/mpi_regressions.rs` | One `#[test]` that owns MPI init and runs 11 named scenarios sequentially. Each builds an `Octree` from the union of a source and a target point set (with the ghost-children layer) and checks the tree, the interaction lists against an oracle, and a forward/backward ghost exchange of every interaction-list ghost, and the index FMM (every leaf must receive every leaf index exactly once). |
| `examples/test_index_fmm.rs` | Seeded-random MPI run of the index FMM. |

Read `src/interaction_manager.rs`, `src/fmm/evaluator.rs` and `tests/mpi_regressions.rs` for
how the API actually behaves; prose is a summary, they are the contract.

### Navigating the code

Follow the root navigation rules (LSP first; `rlst` sources in the registry cache, not
`../rlst`). `nd-octree` is the workspace sibling at `../octree/src/…`. Within this
crate's seven small source modules (plus their `*_tests.rs` files), reading a whole
file beats either LSP or `grep`.

## The interaction manager

`InteractionManager::new(&octree)` derives the U-, V-, W- and X-lists from
`octree.all_keys()` alone. It is **local — no collectives** — so it may be
called on any subset of ranks. The module doc in `src/interaction_manager.rs`
is the contract; the points that matter most when changing it:

- **Entries exist for every non-ghost key** (`LocalLeaf`, `LocalInterior`,
  `Global`) in all four maps. Inapplicable lists (U and W for interior boxes,
  V and X at levels 0 and 1) are empty vectors, never missing entries. Ghost
  keys get no entry. Every list is sorted ascending, deduplicated, and never
  contains the box itself.
- **Adjacency** means the closed cubes share at least a vertex and neither box
  is an ancestor of the other. Boxes may be at different levels.
- **It relies on the octree's guarantees**: complete, 2:1 balanced across all
  26 directions, plus the one-cell same-level ghost halo. Those let it decide
  everything from a neighbour's leaf/interior classification and Morton
  arithmetic without looking children up. If `nd-octree` ever relaxes balance
  or shrinks the halo, the `.expect`/`debug_assert`s in this module fire.
- **Every listed key is a key of `all_keys`**, with its own `KeyType` — and,
  for a ghost, its owner rank via `KeyType::ghost_rank`; a `Global` entry has no
  single owner because it lives on every rank. This holds only when the
  octree is built with `OctreeOptions::with_ghost_children(true)`, as the tests
  and examples do; a tree built with default options does not give it. That
  layer replicates the children of the interior boxes bordering the rank, which
  is exactly what the V- and W-lists need. `InteractionManager::new` asserts it whenever
  `octree.options().ghost_children()` is set; the assertion lives in `new`, not
  in `from_key_types`, because `src/interaction_manager_tests.rs` drives
  `from_key_types` with hand-built maps that need not satisfy the guarantee.
- **Listed keys are routinely ghosts** — V and W entries are children of remote
  neighbours. The manager provides topology only; any exchange of multipole or
  particle data for those keys is the caller's job.

## Dependencies

- `nd-octree` is a path dependency on the workspace sibling `../octree`
  (`nd-octree = { path = "../octree" }`). Keep it that way: do not switch to a
  git or crates.io source, which can diverge from the workspace copy.
- `rlst` 0.8.0 (feature `mpi`) comes from crates.io, as does `mpi` 0.8.2 (rsmpi).
  The crate uses rlst's `distributed_tools::{GhostCommunicator,
  GhostCommunicatorBuilder, ChunkSizes}` in `ghost_communicator.rs`, and
  `distributed_tools::array_tools::gather_to_all` in `evaluator.rs`,
  `index_fmm.rs` and the tests. Tests and examples also use `rlst_dynamic_array`
  and `println_mpi`.
- All dependencies are declared directly in this crate's `Cargo.toml`. The root
  `[workspace.dependencies]` serves the new `nd-fmm-*` crates; migrating this crate
  to it is a separate decision.

## Crate checks

The root checks cover this crate. For this crate alone, add `-p nd-fmm-plan` (e.g.
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-plan`). Also useful locally:

```sh
cargo run -p nd-fmm-plan --example test_index_fmm
```

`cargo test -p nd-fmm-plan` gives 22 unit tests plus the integration test **on one rank
only**. It exercises no redistribution and no ghost layer, which is where the
interesting bugs are.

### Multi-rank runs

For any change touching the interaction manager, the ghost
communicator or the FMM evaluator (whose ghost-dependent and global-level
paths only exist on more than one rank), build the test binary and launch it under MPI:

```sh
cargo test -p nd-fmm-plan --test mpi_regressions --no-run   # note the executable path it prints
RUST_MIN_STACK=8388608 mpiexec -n 2 target/debug/deps/mpi_regressions-<hash> --test-threads=1
RUST_MIN_STACK=8388608 mpiexec -n 4 target/debug/deps/mpi_regressions-<hash> --test-threads=1
```

Use the executable Cargo prints, not the sibling `.d` file, and note the hash
changes on rebuild.

On macOS, add the loopback flags from the root `CLAUDE.md`; with them, 2 and 4 ranks
pass:

```sh
mpiexec --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 4 <exe> --test-threads=1
```

`Cargo.toml` has **no** `[[example]]` or `[package.metadata.example.*]` entries, so
the weekly `run-examples` job does nothing for this crate today; register examples as
the root `CLAUDE.md` describes if they should run there.

## Octree input and MPI discipline

Octree fine keys must be valid Morton keys at level `DEEPEST_LEVEL` (16) —
`Octree::new` validates this and panics naming the offending key, so build them
with `points_to_morton(&points, DEEPEST_LEVEL as usize, &bbox)`. Point arrays
are `[3, npoints]`, one point per **column**; `PhysicalBox` is
`[xmin, ymin, zmin, xmax, ymax, zmax]`. Build the octree with
`OctreeOptions::with_ghost_children(true)` whenever interaction lists or the FMM
evaluator are used.

**MPI discipline.** The root MPI rules apply; several test scenarios exist precisely
to cover empty and uneven ranks. `tests/mpi_regressions.rs` owns the one MPI
initialisation for `tests/`; do not add a second independently initialising test
there. Globally empty octree construction is not covered by any test and should not
be assumed to work.

## Conventions

- Match the existing module layout and rustdoc style.
- Keep topology tests deterministic and free of translation operators. Serial
  helper tests go next to the helpers (`src/*_tests.rs`, included with `#[path]`);
  anything needing redistribution goes in `tests/` or `examples/`.
- Existing coverage: interaction lists — serial: hand-counted U/V sizes on a uniform level-2 tree,
  W/X on a one-octant-refined tree, a brute-force geometric oracle on adaptive
  trees, list invariants, the adjacency helper, and ghost classification;
  multi-rank: every `tests/mpi_regressions.rs` scenario compares all four lists of every
  local non-ghost key against a brute-force oracle over the gathered global tree,
  and checks that every list entry is a key of `all_keys`. Most scenarios only
  reach leaf level 3; `graded corner blob` adds a dense corner at leaf level 6
  against coarse remote neighbours. Every scenario also runs a forward/backward
  ghost exchange and the index FMM. Scenarios cover refinement caps, duplicate
  keys, empty populations, and uneven and empty ranks.
- Add new `tests/mpi_regressions.rs` scenarios to the existing `cases` array rather
  than as new `#[test]` functions, so MPI ownership stays with one test.
- Preserve in-progress work and the local `../octree` path dependency unless
  asked to change them.
- Phase 3 rewrites this crate (docs/phase3/README.md, T1 and T4–T7; design in
  docs/design/fmm-plan-redesign.md once T1 lands). Until T7, add the new modules beside
  the old ones and keep the old API working as the reference.
