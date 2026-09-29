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
| `src/octree.rs` | Public `Octree`, `OctreeOptions`, `KeyType`, `LeafLocation`, `LookupError`, `LookupBatchError`, `compute_global_bounding_box`, `points_to_morton`, `is_complete_linear_and_balanced`. |
| `src/octree/implementation.rs` | Private module with the distributed machinery: input validation, parallel sort/linearization, the replicated coarse tree (capped at `max_level`) with weighting and block partitioning, redistribution, refinement, distributed balancing, ghost discovery, neighbour maps, and the leaf-lookup helpers. |
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

`cargo test` does **not** run the example executables, and CI runs them only weekly at
3 ranks (all five are registered with `templated-examples`), never on one rank and
never on pull requests. For any change touching distributed code, run them by hand
with one rank and with multiple ranks (on macOS add the loopback flags from the root
`CLAUDE.md`):

```sh
cargo build -p nd-octree --examples
mpirun -n 1 target/debug/examples/test_mpi_complete_tree
mpirun -n 3 target/debug/examples/test_mpi_complete_tree
mpirun -n 3 target/debug/examples/test_mpi_global_bounding_box
mpirun -n 1 target/debug/examples/test_mpi_construction_edge_cases
mpirun -n 3 target/debug/examples/test_mpi_construction_edge_cases
mpirun -n 3 target/debug/examples/test_mpi_leaf_lookup
mpirun -n 3 target/debug/examples/test_mpi_vtk target/vtk-example
```

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
`GhostLeaf(rank)`, or `GhostInterior(rank)`; ghosts carry their originating
zero-based rank (`KeyType::ghost_rank`). Ghosts are interface topology, not
local ownership. The halo is a one-cell same-level halo unless
`OctreeOptions::with_ghost_children` is set; with it, construction also
replicates the children of the interior boxes bordering the rank, so that for
every non-ghost key and every same-level neighbour cell of it that the tree
holds as interior, all eight children of that neighbour are in `all_keys` too.
That is one step of closure, not a fixed point, and it travels inside the two
collectives the ghost exchange already performs — no extra round.
`neighbour_map` has entries only for non-ghost keys, though its *values* may be
ghosts. Interior-key neighbours are at the same level; a leaf entry holds, per
neighbouring cell, either its same-level neighbour or that neighbour's parent,
one level coarser, with each key listed once.

**Construction contract.** `Octree::new(fine_keys, options, comm)` panics on keys
that are invalid or not at `DEEPEST_LEVEL`. Every setting other than the keys and
the communicator lives on `OctreeOptions`: `with_max_level` (default
`DEEPEST_LEVEL`), `with_max_fine_keys` (default 1) and `with_ghost_children`. The
coarse tree is replicated on every rank, capped at `max_level`, and partitioned
so every rank owns at least one block; it panics with a message when there are
fewer blocks than ranks. A rank may contribute no keys, and a rank may own
blocks that contain no keys. No leaf is deeper than `max_level` on any rank
count.

**Lookup semantics.** `owner_rank` and `local_leaf` are communication-free; both
reject the invalid marker first, then non-finest-level keys with `LookupError`.
`local_leaf` returns `Ok(None)` for a valid key owned by another rank, even when
a matching ghost is stored locally. `lookup_leaves` returns
`Result<Vec<Result<LeafLocation, LookupError>>, LookupBatchError>`: per-query
errors stay in the inner result, while the outer error signals an MPI
count/displacement overflow that prevents the exchange and is returned
consistently on every rank. Partition intervals are lower-inclusive and
upper-exclusive, with the final rank extending through the last finest key.

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
