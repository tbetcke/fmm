# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Working agreement

Approval for one action is not approval for the next one of its kind. A request
to commit, push, delete, or publish covers the message it appears in and nothing
after it — ask again rather than carrying the permission forward.

Default flow: make the change, run the checks, report exactly which ones ran and
what they said, and leave the result uncommitted unless the current message asks
for a commit. Never report a check as passing that you did not run.

## Project

`fmm-plan` (Cargo package `fmm-plan`, Rust crate `fmm_plan`, Rust 2024) plans the
**topology and data layout** of a fast multipole method. It owns the tree, the
assignment of source/target records to leaves, and the permutations between the
caller's distributed layout and tree order. It does **not** own point
coordinates, kernel evaluations, or translation operators (M2M / M2L / L2L /
near-field arithmetic) — analytical and kernel-independent FMMs are meant to
share this layer. Keep kernel-specific arithmetic out.

- Single Cargo package, not a workspace. Library only, no binaries.
- MPI is a **required** dependency, including for one-rank runs.
- Version `0.1.0-dev`; the public API is unstable and partly unimplemented.
- Upstream <https://codeberg.org/nd-project/nd-plan>, CI via Forgejo Actions.
  (The `homepage` field in `Cargo.toml` has a typo: `codeberg.com.com`.)
- Licensed MIT / Apache-2.0.

## Code map

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate doc plus `pub mod fmm_tree;`, `pub mod ghost_communicator;` and `pub mod interaction_manager;`. |
| `src/fmm_tree.rs` | `FmmTree<'a, C>` plus the private helpers `get_ancestor_key` and `sort_by_leafs`. |
| `src/fmm_tree_tests.rs` | 7 serial unit tests of the private helpers, included via `#[path]` from `fmm_tree.rs`. No MPI. |
| `src/interaction_manager.rs` | `InteractionManager` — U/V/W/X lists for every non-ghost key of an `Octree`, computed locally. See the section below. |
| `src/interaction_manager_tests.rs` | 9 serial unit tests on synthetic key maps (hand counts, brute-force oracle, adjacency, V-list directions), included via `#[path]`. No MPI. |
| `src/ghost_communicator.rs` | `FmmGhostCommunicator<T>` — one rlst `GhostCommunicator<MortonKey>` per level with owned host send/receive buffers, built from the Morton keys of required ghosts (local and `Global` keys skipped) and a `LevelChunkSizes` (uniform or per level). |
| `src/ghost_communicator_tests.rs` | 5 serial unit tests of the ghost bucketing and chunk-size lookup, included via `#[path]`. No MPI. |
| `tests/fmm_tree.rs` | One `#[test]` that owns MPI init and runs 11 named scenarios sequentially, each checking the tree, the interaction lists against an oracle, and a forward/backward ghost exchange of every interaction-list ghost. |
| `examples/test_fmm_tree.rs` | Seeded-random MPI smoke run; asserts global counts, backwards-mapped leaf membership, and interaction-list invariants. |
| `.forgejo/workflows/run-tests.yml` | The authoritative CI commands (PRs to `main` only). |
| `.forgejo/workflows/run-examples.yml`, `run-dependency-checks.yml` | Weekly scheduled jobs. |

Read `src/fmm_tree.rs`, `src/interaction_manager.rs` and `tests/fmm_tree.rs` for
how the API actually behaves; prose is a summary, they are the contract.

### Navigating the code

**Prefer LSP tools — go-to-definition, find-references, hover, document and
workspace symbols — over `grep`/`find` whenever you are chasing a symbol.** They
resolve through the real dependency graph, so they land on the right definition
without you having to know where a crate keeps its sources, and a miss is a
genuine "not found" rather than a typo in a path.

This matters most across the two path dependencies. Their layouts do not match:
`nd-octree` is a plain crate (`../octree/src/…`), but rlst nests its crate one
level down (`../rlst/rlst/src/…`). A grep aimed at `../rlst/src/` silently
returns nothing and reads as "this symbol does not exist" — a real trap when
checking what `rlst` provides (`DistributedArray`, `IndexLayout`,
`DataPermutation`, `sort_to_bins`, `all_to_allv`) or what `nd_octree` guarantees.

`grep` is still the right tool for what LSP does not index: text in comments,
CI YAML, `Cargo.toml`, and quick "where is this string" sweeps. Within this
crate's six small source files, reading a whole file beats either.

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
  single owner because it lives on every rank. This holds because `FmmTree::new`
  builds the octree with `OctreeOptions::with_ghost_children(true)`; a tree built
  with default options does not give it. That layer replicates the children of
  the interior boxes bordering the rank, which is exactly what the V- and
  W-lists need. `InteractionManager::new` asserts it whenever
  `octree.options().ghost_children()` is set; the assertion lives in `new`, not
  in `from_key_types`, because `src/interaction_manager_tests.rs` drives
  `from_key_types` with hand-built maps that need not satisfy the guarantee.
- **Listed keys are routinely ghosts** — V and W entries are children of remote
  neighbours. The manager provides topology only; any exchange of multipole or
  particle data for those keys is the caller's job.

Clippy is clean under `-D warnings` (also with `--all-targets`). Keep it that
way: do not add blanket `#[allow]`s.

## Build environment

Stable Rust with Rust 2024, plus `rustfmt` and `clippy`.

- `nd-octree` resolves to the **sibling path `../octree`** (the git source is
  commented out in `Cargo.toml`). That checkout must exist and be readable. Do
  not silently switch to the git dependency; the two can diverge.
- `rlst` supplies `DistributedArray`, `IndexLayout`, `DataPermutation`,
  `sort_to_bins` and `all_to_allv`.
- Native prerequisites, as installed by CI: `libclang-dev cmake libfftw3-dev
  libopenblas-dev openmpi-bin libopenmpi-dev`. A dependency build failure is far
  more often a missing native library than a defect here.

## Checks

Run from the repository root. CI (`run-tests.yml`) runs exactly:

```sh
cargo fmt -- --check
cargo clippy -- -D warnings
cargo clippy --examples -- -D warnings
RUST_MIN_STACK=8388608 cargo test
cargo doc --no-deps
```

Also useful locally:

```sh
cargo run --example test_fmm_tree
```

`RUST_MIN_STACK=8388608` matters — keep it on test invocations.

Plain `cargo test` gives 21 unit tests plus the integration test **on one rank
only**. It exercises no redistribution and no ghost layer, which is where the
interesting bugs are.

### Multi-rank runs

CI never runs anything on more than one rank, so multi-rank coverage is entirely
on you. For any change touching `FmmTree::new`, `sort_by_leafs`, the
permutations, or the interaction manager (whose ghost-dependent paths only
exist on more than one rank), build the test binary and launch it under MPI:

```sh
cargo test --test fmm_tree --no-run   # note the executable path it prints
RUST_MIN_STACK=8388608 mpiexec -n 2 target/debug/deps/fmm_tree-<hash> --test-threads=1
RUST_MIN_STACK=8388608 mpiexec -n 4 target/debug/deps/fmm_tree-<hash> --test-threads=1
```

Use the executable Cargo prints, not the sibling `.d` file, and note the hash
changes on rebuild.

**On macOS here, plain `mpiexec -n 2 …` aborts** — Open MPI picks a non-loopback
interface, the TCP connect times out after 60s, and every rank dies on a NULL
communicator. Restrict it to loopback:

```sh
mpiexec --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0 -n 4 <exe> --test-threads=1
```

With that, 2 and 4 ranks pass. The failure is environmental; do not chase it as
a bug in this crate.

Always wrap multi-rank runs in an external timeout. An assertion on one rank
leaves the others blocked in a collective, and the job hangs rather than failing.

The scheduled `run-examples` workflow calls `cargo templated-examples`, but
`Cargo.toml` has **no** `[[example]]` or `[package.metadata.example.*]` entries,
so that job does nothing for this crate today. If you add examples that should
run in CI, add both sections, mirroring the `nd-octree` repository.

## The `FmmTree` contract

`FmmTree::new(source_keys, target_keys, options, comm)` takes two independently
distributed `u64` Morton-key arrays, one `nd_octree::OctreeOptions` carrying the
tree settings, and borrows the communicator for `'a`. It stores permutations, CSR-style pointers and the
octree — never coordinates or matrices.

**Input requirements.** Both local arrays must be contiguous (asserted). Every
key must be a valid Morton key at level `DEEPEST_LEVEL` (16) — `Octree::new`
validates this and panics naming the offending key, so build inputs with
`points_to_morton(&points, DEEPEST_LEVEL as usize, &bbox)`. Point arrays are
`[3, npoints]`, one point per **column**; `PhysicalBox` is
`[xmin, ymin, zmin, xmax, ymax, zmax]`.

**What construction does**, in order:

1. Concatenate local source and target keys and build one `nd_octree::Octree`
   from the union, so a single tree serves both populations. The caller's
   `options` are passed through with `.with_ghost_children(true)` forced on,
   which the interaction manager depends on.
2. Pair each population's keys with their **original global indices**
   (`layout.local2global`) and sort by key.
3. Bin against `octree.coarse_tree_bounds()` and redistribute keys and indices
   with `all_to_allv`.
4. Group each received population by its ancestor leaf (`sort_by_leafs`).
5. Build independent `DataPermutation`s and `indptr` arrays per population.

**Invariants to preserve.** For leaf `octree().leaf_keys()[i]`, that leaf's
records occupy `indptr[i]..indptr[i + 1]` in tree order. Each pointer array has
`leaf_count + 1` entries, starts at 0, is nondecreasing, and ends at the
**local redistributed** count — which generally differs from the local input
count, and differs between sources and targets. Empty ranges are normal;
source and target occupancy are unrelated. Global counts are conserved.

`forward_permute` maps original layout → tree order, `backward_permute` inverts
it. Since rlst 0.8 the chunk size (contiguous values per record) is fixed when
the `DataPermutation` is built; the tree's own permutations use chunk size 1. For
multi-value records build a permutation from `source_indices()` /
`target_indices()` (the original global indices in tree order) and the caller's
index layout — a collective call. Duplicate Morton keys must keep distinct records — the tests rely on
distinct global IDs to catch lost or duplicated entries where keys coincide.
Grouping is stable within a leaf, but fine keys are **not** globally sorted after
communication from several ranks; do not assume they are.

**`get_ancestor_key` is a `debug_assert` + `get_unchecked` fast path.** Its
preconditions: `arr` sorted ascending, and an ancestor of `key` present. A key
below `arr[0]` yields `binary_search` → `Err(0)` and then `index - 1`, which
underflows: a panic in debug, an out-of-bounds unchecked read in release. So
never write a test that feeds it an uncovered key and expects a panic — that is
undefined behaviour, not a testable contract. Guard at the call site instead if
you ever relax the precondition.

**MPI discipline.** Every rank must reach every collective in the same order,
including ranks with empty input — several test scenarios exist precisely to
cover empty and uneven ranks. Keep MPI initialization in a **single** test per
executable (MPI cannot be re-initialized after finalization in one process); do
not add a second independently initializing parallel test to `tests/`. Globally
empty octree construction is not covered by any test and should not be assumed
to work.

## Conventions

- Match the existing module layout and rustdoc style. Document new public items;
  CI builds docs. Keep changes targeted — no drive-by reformatting.
- Keep topology tests deterministic and free of translation operators. Serial
  helper tests go next to the helpers (`src/*_tests.rs`, included with `#[path]`);
  anything needing redistribution goes in `tests/` or `examples/`.
- Existing coverage: helpers — exact and mixed-depth ancestors, stable grouping,
  duplicates, empty leaf ranges, empty populations, and a linear oracle over 32
  input rotations. Public API — count conservation, leaf membership, pointer
  invariants, scalar and 2-value permutation round trips, unequal and identical
  populations, refinement caps, empty populations, uneven and empty ranks.
  Interaction lists — serial: hand-counted U/V sizes on a uniform level-2 tree,
  W/X on a one-octant-refined tree, a brute-force geometric oracle on adaptive
  trees, list invariants, the adjacency helper, and ghost classification;
  multi-rank: every `tests/fmm_tree.rs` scenario compares all four lists of every
  local non-ghost key against a brute-force oracle over the gathered global tree,
  and checks that every list entry is a key of `all_keys`. Most scenarios only
  reach leaf level 3; `graded corner blob` adds a dense corner at leaf level 6
  against coarse remote neighbours.
- Add new `tests/fmm_tree.rs` scenarios to the existing `cases` array rather
  than as new `#[test]` functions, so MPI ownership stays with one test.
- Preserve in-progress work and the local `../octree` path dependency unless
  asked to change them.
