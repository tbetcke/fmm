# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Working agreement

Approval for one action is not approval for the next one of its kind. A request
to commit, push, delete, or publish covers the message it appears in and nothing
after it — so ask again rather than carrying the permission forward, even when
the later action looks like an obvious continuation of the same task.

In particular: make the changes, run the checks, report what happened, and leave
the result uncommitted unless the current message asks for a commit.

Always run `cargo fmt --all` after editing any Rust code, before running the
other checks or committing. CI rejects unformatted code, and mechanical edits
(renames, `sed` replacements) easily break import ordering.

## Project

`nd-octree` (crate `nd-octree`, imported as `nd_octree`) is a Rust 2024 library
for adaptive, 2:1-balanced three-dimensional octrees, running on a single MPI
rank or distributed across a communicator. It is a **topology library**: it
stores Morton keys, tree structure, ownership, and neighbour relationships. It
does **not** store point arrays or application data — those, and any algorithms
on them, live outside this crate.

- Single Cargo package, not a workspace.
- MPI is a **required** dependency (also for one-rank programs), not a feature.
- Upstream: <https://codeberg.org/nd-project/octree>, CI via Forgejo Actions.
- Licensed MIT / Apache-2.0.

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
| `.forgejo/workflows/` | Authoritative CI commands. |
| `find_examples.py` | Legacy shell-script generator driven by `//?` comments. It is **not** the current example runner; ignore it unless asked to work on it. |

When checking how the API actually behaves, read the implementation and the
examples rather than trusting prose.

## Build environment

Stable Rust with Rust 2024 support, plus `rustfmt` and `clippy`. Native
prerequisites (as installed in Linux CI): `libclang-dev cmake libfftw3-dev
libopenblas-dev openmpi-bin libopenmpi-dev`. The `mpi` and `rlst` crates build
against a real MPI installation and BLAS/LAPACK, so check that the MPI
compiler/runtime and native libraries are present before diagnosing a dependency
build failure as a defect in this crate.

Features: `strict` denies warnings and unused crate dependencies; `battleship`
is an empty placeholder and does **not** enable MPI.

## Checks

Run everything from the repository root. These are exactly what the
`run-tests` workflow does:

```sh
cargo fmt -- --check
cargo clippy --all-targets --features strict -- -D warnings
RUST_MIN_STACK=8388608 cargo test
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo build --examples
```

`RUST_MIN_STACK=8388608` is set as a CI environment variable; some tests need
it. For a focused run, append a filter, e.g.
`RUST_MIN_STACK=8388608 cargo test test_neighbours`.

`cargo test` also runs the doctests in the crate documentation. MPI doctests are
marked `no_run` because they must be launched by an MPI launcher; keep that
marker on any new example that initializes MPI.

**Keep MPI initialization to a single test per test executable.** MPI cannot be
re-initialized after it has been finalized inside one process, so a second
independently initializing test in the same binary breaks `cargo test`.
`src/octree/implementation.rs` already spends the lib binary's one
initialization (`test_ghost_children_option_is_a_noop_on_one_rank`); a further
MPI-using unit test has to reuse it — or move to `tests/` or `examples/` —
rather than call `mpi::initialize` again. For the same reason `cargo test` now
needs a working MPI *runtime*, not just a linkable `libmpi`.

### MPI examples

`cargo test` does **not** run the example executables. For any change touching
distributed code, build and run them explicitly with one rank and with multiple
ranks:

```sh
cargo build --examples
mpirun -n 1 target/debug/examples/test_mpi_complete_tree
mpirun -n 3 target/debug/examples/test_mpi_complete_tree
mpirun -n 3 target/debug/examples/test_mpi_global_bounding_box
mpirun -n 1 target/debug/examples/test_mpi_construction_edge_cases
mpirun -n 3 target/debug/examples/test_mpi_construction_edge_cases
mpirun -n 3 target/debug/examples/test_mpi_leaf_lookup
mpirun -n 3 target/debug/examples/test_mpi_vtk target/vtk-example
```

CI (both `run-tests` and the scheduled `run-examples`) only runs
`test_mpi_complete_tree`, `test_mpi_global_bounding_box`, and
`test_mpi_construction_edge_cases`, at 1 and 3 ranks.
`test_mpi_leaf_lookup` and `test_mpi_vtk` are registered in `Cargo.toml` but are
**not** in the CI loops — run them by hand when you touch lookup or VTK output.

Register every new example in `Cargo.toml` with both a `[[example]]` section and
a `[package.metadata.example.<name>.templated-examples]` entry using
`command = "mpirun -n {{NPROCESSES}}"`, matching the existing entries.

A scheduled `run-dependency-checks` workflow runs `cargo audit` and
`cargo upgrades` weekly.

Report which checks you actually ran, and say plainly when something could not
run (missing MPI, missing native libraries). Never claim an unrun check passed.

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

**MPI.** Collectives must be reached by all participating ranks in the same
order — including ranks with empty input. `lookup_leaves` is collective and must
be entered even with `&[]` or only invalid keys. Preserve communicator lifetimes
(`Octree<'o, C>` borrows its communicator) and collective ordering, including
debug assertions that themselves communicate. Do not put communication inside a
conditional that only some ranks take.

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

- Follow the existing module structure and Rust naming. Keep targeted changes
  targeted: no unrelated reformatting or refactors.
- Document every new public item — the crate sets `#![warn(missing_docs)]` and
  CI builds docs with `-D warnings`. Follow the existing rustdoc shape:
  `# Parameters`, `# Returns`, `# Examples`, and `# Collective operation` where
  relevant, with a compiling (or `no_run`) example.
- Clippy `forbid`s wildcard imports (`[lints.clippy] wildcard_imports = "forbid"`).
- Add deterministic unit tests next to serial helpers, using
  `tools::seeded_rng` / `tools::generate_random_keys` where randomness helps.
  Exercise distributed behaviour in `examples/test_mpi_*.rs`, covering rank
  boundaries, empty ranks, and ghost/neighbour consistency.
- Keep ignored artifacts out of commits: `target/`, VTK output, `*.csv`.
  `Cargo.lock` is gitignored (library crate) — do not add it.
