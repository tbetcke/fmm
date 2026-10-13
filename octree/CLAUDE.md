# nd-octree

The root `CLAUDE.md` applies as well; it holds every workspace-wide rule. This file
adds only what is specific to this crate.

## Project

`nd-octree` (crate `nd-octree`, imported as `nd_octree`) is a Rust 2024 library
for adaptive, 2:1-balanced three-dimensional octrees, running on a single MPI
rank or distributed across a communicator. It is a **topology library**: it
stores Morton keys, tree structure, ownership, and neighbour relationships. It
does **not** store point arrays or application data — those, and any algorithms
on them, live outside this crate.

- `nd-fmm-plan` depends on this crate through the path `../octree`.
- The `repository` and `homepage` fields in `Cargo.toml` still point at the former
  standalone repository `nd-project/octree`.

## Code map

| Path | Contents |
| --- | --- |
| `src/lib.rs` | Crate-level guide (prerequisites, point layout, ownership, operation/communication table), module declarations, public re-exports, `#![warn(missing_docs)]`. |
| `src/morton.rs` | `MortonKey = u64`, encode/decode, ancestry, children/siblings/neighbours, and **serial** linearization, completion, and balancing. Most unit tests live here (~31). |
| `src/constants.rs` | `DEEPEST_LEVEL = 16`, `NLEVELS`, `LEVEL_SIZE`, bit masks/displacements, the 26 `DIRECTIONS`, and the X/Y/Z encode/decode lookup tables. |
| `src/geometry.rs` | `PhysicalBox` and physical/reference coordinate transforms. |
| `src/octree.rs` | Public `Octree`, `OctreeOptions`, `PartitionWeight`, `KeyType`, `LeafLocation`, `LookupError`, `LookupBatchError`, `compute_global_bounding_box`, `points_to_morton`, `is_complete_linear_and_balanced`; re-exports `PartitionMap`. |
| `src/octree/implementation.rs` | Private module with the distributed machinery: input validation, the key routing by weight that linearizes the input keys without a sort, the replicated coarse tree (capped at `max_level`), built by weight from the root, with the nearest-boundary cut and the partition bounds (ranges may be empty), redistribution, refinement, distributed balancing by routing the closure along the partition bounds, the blocks' leaf flags, the locally essential key set (the map's top-tree keys and the push), the second cut's helpers (the weights scaled to integers, gathered, and the leaves moved; Phase 5S T7), neighbour maps, and the leaf-lookup helpers. |
| `src/octree/partition_map.rs` | Public `PartitionMap` (Phase 5S T3): every coarse block, the cut and a leaf bit per block, replicated on every rank; the owner, sharers and kind of any key without communication; its unit tests against brute force. |
| `src/tools.rs` | `seeded_rng` (ChaCha8) and `generate_random_keys`. |
| `src/types.rs` | Placeholder module (doc comment only). |
| `src/vtk.rs` | Dependency-free ASCII VTK/PVTU writer: serial `write_vtu` and collective `write_pvtu`. |
| `tests/` | `mpi_edge_cases.rs` (one-rank construction regression), `vtk.rs` (serial VTK writer tests against the public `nd_octree::vtk` API). |
| `examples/test_mpi_*.rs` | Executable, assertion-based MPI integration tests. |
| `find_examples.py` | Legacy shell-script generator driven by `//?` comments. It is **not** the current example runner; ignore it unless asked to work on it. |

## Features

`strict` denies warnings and unused crate dependencies; `battleship` is an empty
placeholder and does **not** enable MPI.

## Crate checks

In addition to the root checks, run these locally; CI does not enforce them, but they
pass on the current code:

```sh
cargo clippy -p nd-octree --all-targets --features strict -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps -p nd-octree
cargo build -p nd-octree --examples
```

For a focused run, select the package and append a filter, e.g.
`RUST_MIN_STACK=8388608 cargo test -p nd-octree test_neighbours`.

`src/octree/implementation.rs` already spends the lib test binary's one MPI
initialisation (`test_ghost_children_option_is_a_noop_on_one_rank`); a further
MPI-using unit test has to reuse it, or move to `tests/` or `examples/`.

### MPI examples

`cargo test` does **not** run the example executables. CI runs them in debug at 2 and 4
ranks on every pull request (the `run-tests-mpi` job, Phase 5 T3) and in release weekly
at 3 ranks (all seven are registered with `templated-examples`), never on one rank. For
any change touching distributed code, run them by hand with one rank and with multiple
ranks (on macOS add the loopback flags from the root `CLAUDE.md`):

```sh
cargo build -p nd-octree --examples
mpirun -n 1 target/debug/examples/test_mpi_complete_tree
mpirun -n 3 target/debug/examples/test_mpi_complete_tree
mpirun -n 3 target/debug/examples/test_mpi_global_bounding_box
mpirun -n 1 target/debug/examples/test_mpi_construction_edge_cases
mpirun -n 3 target/debug/examples/test_mpi_construction_edge_cases
mpirun -n 3 target/debug/examples/test_mpi_leaf_lookup
mpirun -n 3 target/debug/examples/test_mpi_vtk target/vtk-example
mpirun -n 3 target/debug/examples/test_mpi_weighted_partition
mpirun -n 3 target/debug/examples/test_mpi_locally_essential
```

`test_mpi_weighted_partition` (Phase 5 T4) checks the partition by weight on four
workloads: the leaves against the one-rank tree, the weight bound, independence of the
input distribution, and a tree whose only block is the root; and (Phase 5S T7) the
second cut, `Octree::repartition`, by five weights per block (uneven, heavy, zero,
unit): the leaves the first cut's key for key, the nearest-boundary cut of the gathered
weights, ownership, the held keys' kinds, one rank unchanged, a repeat a fixed point, and
the minimum gain's decision against brute force.
Run it also at 2, 4 and 8 ranks for a change to the partition;
`test_mpi_construction_edge_cases` at 1–8, and `test_mpi_leaf_lookup` at 8, where rank 1 owns no leaves. `test_mpi_locally_essential`
(Phase 5S T3) checks the partition map and the held keys against brute force over the
gathered tree (H = N(S) ∪ C(N(S)) exactly, every key a list, the halo or a parent-child
step names held, a subset of the Phase 5 set, which `examples/phase5_keys/` rebuilds),
also on every tree cut a second time (Phase 5S T7); run it at 1–8 for a change to the
held keys.

## Invariants and domain rules

**Morton keys.** Levels 0–16. The low 15 bits hold the level, the next 48 the
interleaved position, and the highest bit marks an invalid key. Zero is the
**root** — a valid key, not a sentinel; use `morton::invalid_key()` and
`morton::is_valid` for invalidity. Use the constructors and helpers
(`from_index_and_level`, `from_physical_point`, `parent`, `children`, …) rather
than ad hoc bit arithmetic. Do not hand-edit the lookup tables in
`constants.rs` without adding encode/decode regression coverage.

**Tree properties.** Preserve Morton ordering, ancestor/descendant semantics,
completeness, linearity, and 2:1 balance when changing tree algorithms. Serial
helpers in `morton.rs` and distributed helpers in `octree/implementation.rs`
often share names but have entirely different communication requirements — do
not conflate them.

**Geometry.** Point arrays have shape `[3, npoints]`, one point per **column**.
`PhysicalBox` coordinates are `[xmin, ymin, zmin, xmax, ymax, zmax]`.
`points_to_morton` expects points inside the box — a point on or beyond a face
is clamped into the nearest cell rather than reported — and it clamps
`max_level` to `DEEPEST_LEVEL` and asserts `shape()[0] == 3`.
`compute_global_bounding_box` is collective, pads a cubic box, and panics on a
degenerate domain (no points anywhere, or every point coincident). Think about
boundary, empty, and degenerate inputs when changing any of this.

**MPI.** The root collective-ordering rule applies. In particular,
`lookup_leaves` is collective and must be entered even with `&[]` or only invalid
keys. Preserve communicator lifetimes (`Octree<'o, C>` borrows its communicator)
and collective ordering, including debug assertions that themselves communicate.

**Ownership and ghosts.** `KeyType` is `LocalLeaf`, `LocalInterior`, `Global`,
`GhostLeaf(rank)`, or `GhostInterior(rank)`; ghosts carry their owner's zero-based
rank (`KeyType::ghost_rank`). Ghosts are interface topology, not local ownership.
Since Phase 5S T3 (docs/design/scale-out.md §3) `all_keys` is the rank's locally
essential set H = N(S) ∪ C(N(S)): S its own keys and its shared ancestors (`Global`
now means an ancestor of an own block, nothing else), N(S) S with every tree key that
is a same-level neighbour of a key of S, C(N(S)) the children of N(S)'s interior keys,
only with `OctreeOptions::with_ghost_children`. So for every non-ghost key and every
same-level neighbour cell of it that the tree holds as interior, all eight children of
that neighbour are in `all_keys` too: one step of closure, not a fixed point. No rank
holds every block or every ancestor of one any more; `Octree::partition_map`
(`PartitionMap`) holds every block as a key with its owner and leaf bit (8 B per
block, a bit, 8 (P + 1) B for the cut), and gives the owner of any key (for a key above
the blocks, its first descendant block's rank), its sharers and the kind of any block
or ancestor of one locally. The top-tree keys of H come from the map; the keys below
other ranks' blocks come by one push (an all-to-all-v: every own key to the ranks with
a same-level neighbour cell of it in their S, with its children under the layer), and
the receiver keeps what lies in H. No key list is gathered to every rank.
`neighbour_map` has entries only for non-ghost keys, though its *values* may be
ghosts. Interior-key neighbours are at the same level; a leaf entry holds, per
neighbouring cell, either its same-level neighbour or that neighbour's parent,
one level coarser, with each key listed once.

**Construction contract.** `Octree::new(fine_keys, options, comm)` panics on keys
that are invalid or not at `DEEPEST_LEVEL`, and on nothing else. Every setting other
than the keys and the communicator lives on `OctreeOptions`: `with_max_level`
(default `DEEPEST_LEVEL`), `with_max_fine_keys` (default 1), `with_ghost_children`,
and, for several ranks, `with_partition_weight` (default `PartitionWeight::Keys`,
every key with its duplicates) and `with_block_refinement` (default 8). On several
ranks the coarse tree is built by weight from the root (docs/design/distributed-fmm.md
§3.2, O1): a block splits while it weighs more than W/(kP), holds more than
`max_fine_keys` distinct keys and lies above `max_level`, one all-reduce per round;
it is then 2:1 balanced serially. Its blocks are therefore nodes of the one-rank tree,
and the leaves equal the one-rank leaves on every rank count and input distribution;
keep it so. No key is sorted across the ranks (S4, Phase 5S T3a): each rank sorts its
input keys once; rounds of provisional blocks, split by weight alone (one all-reduce
each), cut at the nearest boundaries, route the keys deduplicated as (key,
multiplicity) pairs in one all-to-all-v, so every copy of a key lands on one rank and
O1 counts distinct keys and weights from the routed keys. The 2:1 balance routes each
rank's linearized closure by the partition bounds (a key by its first finest
descendant, so a key coarser than the blocks reaches the owner of its first
descendant block) and linearizes locally, which leaves every leaf on its owner. The
leaves, blocks and cut are the Phase 5 ones key for key (T3a checked them against a
private copy of the Phase 5 construction, removed with T3). The coarse tree is
replicated on every rank and cut into contiguous
ranges at the block boundaries nearest to W p/P (O3; ties to the later boundary).
`Octree::repartition(&self, weights, min_gain)` (Phase 5S T7, S3; docs/design/scale-out.md
§6.3) cuts the same blocks again by the caller's weights, one per own block (an all-gather-v,
the weights scaled to integers summing to about 2⁵² so that `partition_blocks` cuts
them exactly as it cuts keys), moves the leaves to their new owners (one all-to-all-v;
they arrive sorted) and rebuilds the partition map (the leaf bits do not depend on the
cut) and H by the push. The coarse tree, the leaves and every key stay; only ownership
changes, so the leaves stay the one-rank leaves. With `Some(min_gain)` the new cut is
taken only if it lowers the heaviest rank's (scaled) weight by at least that fraction,
decided alike on every rank from the gathered weights; otherwise, as when the new cut
equals the current one, it returns a copy without the move and the push. One rank is
unchanged. The crate never
interprets a weight (no kernel knowledge: the FMM's cost model lives in `nd-fmm-exec`).
A range may be empty (O4): such a rank owns no blocks or leaves, holds no key (its S is
empty), and still enters every collective. The partition bounds are
computed locally from the replicated tree; an empty rank takes the next non-empty
rank's bound, or `morton::invalid_key()` after the last one. When the root is the only
block (few distinct keys, `max_level` 0, or no keys at all, then on rank 0) it is a
local key of its owner, held by no other rank, never `Global` on several ranks (on one
rank the root is `Global` unless it is the only leaf, as in Phase 5). A rank may contribute
no keys. No leaf is deeper than `max_level` on any rank count. On one rank the coarse
tree is the root and none of this changes the tree.

**Lookup semantics.** `owner_rank` and `local_leaf` are communication-free; both
reject the invalid marker first, then non-finest-level keys with `LookupError`.
`local_leaf` returns `Ok(None)` for a valid key owned by another rank, even when
a matching ghost is stored locally. `owner_rank` never names a rank without blocks. `lookup_leaves` returns
`Result<Vec<Result<LeafLocation, LookupError>>, LookupBatchError>`: per-query
errors stay in the inner result, while the outer error signals an MPI
count/displacement overflow that prevents the exchange and is returned
consistently on every rank. Partition intervals are lower-inclusive and
upper-exclusive, with the final rank that has blocks extending through the last
finest key.

**Debug vs release.** Debug construction performs expensive distributed
invariant checks. Use release builds for performance measurement — never as a
way around a failing debug assertion.

## Conventions

- Follow the existing module structure and Rust naming.
- The crate sets `#![warn(missing_docs)]`. Follow the existing rustdoc shape:
  `# Parameters`, `# Returns`, `# Examples`, and `# Collective operation` where
  relevant, with a compiling (or `no_run`) example.
- Clippy `forbid`s wildcard imports (`[lints.clippy] wildcard_imports = "forbid"`).
- Add deterministic unit tests next to serial helpers, using
  `tools::seeded_rng` / `tools::generate_random_keys` where randomness helps.
  Exercise distributed behaviour in `examples/test_mpi_*.rs`, covering rank
  boundaries, empty ranks, and ghost/neighbour consistency.
- `_test_sphere.vtk` is an existing tracked file; other VTK output stays out of commits.
