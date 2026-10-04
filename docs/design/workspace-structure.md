# FMM workspace structure: crates, boundaries and conventions

As of 2026-09-29; revised the same day after reading the `octree` and `fmm-plan` sources.
Revised at the end of Phase 3 (2026-10-02; Sections 1.1, 3, 3.1, 4 and 6): `nd-fmm-plan`
is rewritten around a batched operator interface, and `nd-fmm-exec` and the Phase 3
parts of `nd-fmm-validate` are built.
Revised again after Phase 3 (2026-10-02; Sections 2, 3, 3.1, 4, 5.1 and 6): Phase 3S
adds the crate `nd-fmm-simd`, hand-written SIMD P2P kernels for the host
([simd-p2p.md](simd-p2p.md)), and a narrow exception to the unsafe rule for it.
Revised at the end of Phase 3S (2026-10-03; Sections 3, 3.1 and 6): `nd-fmm-simd` is
built and used by `nd-fmm-exec`, and `nd-fmm-validate` gains the P2P benchmarks.
Revised at the end of Phase 4 (2026-10-04; Sections 2, 3, 3.1, 5 and 6): `nd-fmm-kernels`
is built on CubeCL 0.11.0-pre.4, a workspace member but not a default member, tested on
the CubeCL CPU runtime by its own CI job; `nd-fmm-exec` gains the device path behind the
feature `gpu`, and `nd-fmm-validate` the device benchmarks
([device-path.md](device-path.md)).

Add seven new crates to the existing workspace, created phase by phase rather than all at
once. The existing `nd-fmm-plan` crate is the integration layer. It already owns the
interaction lists, the pass order and the ghost exchange on top of `nd-octree`. The new
Laplace crates plug into it through its `FmmOperator` trait, so the planned octree adapter
(`nd-fmm-tree`) and most of the planned distribution crate (`nd-fmm-dist`) are no longer
needed. Only `fmm-math` (plus fixture tooling) is needed for Phase 0.

This document refines Section 5 of the design document
(`docs/design/laplace-fmm-plan.md`) into concrete crates for
[tbetcke/fmm](https://github.com/tbetcke/fmm).

## 1. Starting point

The repository is a Cargo workspace with two crates, both carrying their own
`CLAUDE.md`, and GitHub Actions CI at the root.

| Item | What it is | State |
| --- | --- | --- |
| `Cargo.toml` (root) | virtual workspace manifest: `members = ["fmm-plan", "octree"]`, `resolver = "2"`; no `[workspace.package]`, no `[workspace.dependencies]`, no `default-members` | in place, minimal |
| `Cargo.lock` (root) | committed | in place |
| `octree/` (package `nd-octree`, lib `nd_octree`, 0.1.1-dev) | distributed, adaptive, 2:1-balanced Morton octree; topology only | fairly mature |
| `fmm-plan/` (package `nd-fmm-plan`, lib `nd_fmm_plan`, 0.1.0-dev) | FMM topology and data flow: interaction lists, ghost exchange, distributed pass order | early; API unstable, partly unimplemented; renamed from `fmm-plan` on 2026-09-29 |
| `CLAUDE.md` per crate | agent instructions; both require `cargo fmt --all` after Rust edits | in place; partly stale since the merge into one workspace (Section 6) |
| root `CLAUDE.md` | workspace-wide rules | added with these documents |
| `.github/workflows/` | CI, see below | in place |
| `LICENSE-MIT`, `LICENSE-APACHE` | dual licence | in place |

Both existing crates declare their package fields and dependencies directly and point
`repository` at their former standalone repositories (`nd-project/octree` and
`nd-project/nd-plan`). `nd-fmm-plan` writes its licence as `"MIT / Apache-2.0"` (the
deprecated form) and has a `codeberg.com.com` typo in `homepage`.

**CI as it runs today** (`run-tests.yml`, on pull requests to `main`, from the root):

- Installs `libclang-dev cmake libfftw3-dev libopenblas-dev openmpi-bin libopenmpi-dev`.
- Runs `cargo fmt -- --check`, `cargo clippy -- -D warnings`,
  `cargo clippy --examples -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test` and
  `cargo doc --no-deps`.
- Consequences: clippy does not cover test targets (no `--all-targets`); docs are not
  built with `-D warnings`; nothing runs on more than one MPI rank.
- Weekly jobs: `run-examples` runs `cargo templated-examples NPROCESSES 3` (only
  `nd-octree` registers templated examples); `run-dependency-checks` runs `cargo upgrades`.

Two conventions follow from what is there and are adopted below: directory names without
prefix, package names with the `nd-` prefix (directory `octree`, package `nd-octree`;
directory `fmm-plan`, package `nd-fmm-plan`), and a `CLAUDE.md` in every crate.

### 1.1 What the existing crates provide

Checked against the sources: `nd-octree` on 2026-09-29, `nd-fmm-plan` as rewritten in
Phase 3 (2026-10-02).

**`nd-octree`** is a topology library. It stores keys, ownership and neighbour relations.
It stores no points and no application data.

- **Keys.** `MortonKey = u64`.
  - Bits 0–14 hold the level; the next 48 bits hold the interleaved position; bit 63 marks
    an invalid key. Root is the valid key `0`.
  - Levels 0–16 (`DEEPEST_LEVEL = 16`).
  - Child index is `4x + 2y + z` (`morton::child_index`).
  - Sorting raw keys gives Morton order, with ancestors before descendants.
  - There is no integer box index of any kind. Keys live in `Vec<MortonKey>` and
    `HashMap<MortonKey, _>`.
- **Tree.** Adaptive, complete, linear and 2:1 balanced across all 26 directions.
  - Refinement stops at `max_level` or at `max_fine_keys` keys per box, so a leaf at
    `max_level` can hold more.
  - `Octree::leaf_keys()` gives the owned leaves in Morton order.
  - `Octree::all_keys()` is a `HashMap<MortonKey, KeyType>` over leaves, interiors,
    replicated `Global` coarse keys and ghosts.
  - `KeyType` is `LocalLeaf`, `LocalInterior`, `Global`, `GhostLeaf(rank)` or
    `GhostInterior(rank)`.
  - Nodes are not grouped by level.
- **Relatives.** In `nd_octree::morton`: `parent`, `children`, `siblings`, same-level
  `neighbours` (26), `key_in_direction`, `ancestor_at_level`, `is_ancestor`.
  `Octree::neighbour_map()` gives neighbours of non-ghost keys. **No U, V, W or X lists.**
- **Geometry.**
  - `PhysicalBox` is `[xmin, ymin, zmin, xmax, ymax, zmax]`. It derives no `Clone` or
    `Copy`; use `coordinates()`.
  - `morton::physical_box(key, &domain)` gives a key's box.
  - There is no centre or half-width helper.
  - `compute_global_bounding_box` (collective) pads the domain to a cube, so every box on
    a level is a cube of the same size.
  - Points are rlst `[3, n]` arrays of `f64`, one point per column.
  - `points_to_morton` bins points into level-16 keys, but the octree never stores,
    sorts or permutes points.
- **Distribution.** Uses rsmpi (`mpi` 0.8.2) and rlst 0.9 distributed tools directly.
  - `Octree<'o, C: CommunicatorCollectives>` borrows its communicator. MPI is required,
    also on one rank.
  - Construction is collective. It computes the coarse tree, replicates it on every
    rank, and partitions it into contiguous Morton blocks.
  - The ghost layer exchanges **keys only**: a one-cell, same-level halo.
  - `OctreeOptions::with_ghost_children(true)` also replicates the children of bordering
    interior boxes, which the V- and W-lists need.

**`nd-fmm-plan`** plans the topology and data flow of an FMM on a concrete
`nd_octree::Octree`. Kernel arithmetic is deliberately kept out of it. Phase 3 rewrote
it (design: [fmm-plan-redesign.md](fmm-plan-redesign.md); T4–T7, PRs #30–#33). As of
the end of Phase 3 it provides:

- **`index::BoxIndex`.** On every level, the boxes a rank holds (local, `Global` and
  ghost) are numbered 0..n in Morton order, by their sorted keys: deterministic for a
  fixed tree and number of ranks. Index → key is O(1); key → index is a binary search,
  used only off hot paths. Leaves have a separate leaf numbering: the local leaves by
  (level, key), so each level's leaves are one contiguous range, then the ghost leaves
  that a U or X list names.
- **`lists`.** The U, V, W and X lists and the parent/child relations as index arrays:
  - target-centric CSR rows (`Csr`, `VList`, `Children`, `Parents`), whose rows line
    up with `chunks_mut(size)` of an output buffer;
  - V grouped per (level, offset) in the order of `V_LIST_DIRECTIONS` (CONVENTIONS
    §3.12), with each target at most once per batch; M2M and L2L grouped per (level,
    child octant), with local and global M2M views.

  The per-key list rule of the original `InteractionManager` is kept (private, in
  `interaction_manager`), and its brute-force oracle in `tests/mpi_regressions.rs` is
  the specification. The tree must be built with `with_ghost_children(true)`.
- **`plan::Plan::new(&octree)`** builds the index and the lists of every level
  without communication, then agrees on the level count and on validity (missing
  ghost-children layer, unheld list entries or children) with two all-reduces, so an
  error is returned on every rank.
- **`store`.** `LevelBuffers`: one buffer per kind (multipoles, locals), box i of level
  l at offset(l) + i · size(l), so a level is a column-major size × n matrix.
  `LeafStore`: CSR stores for source data (local and ghost leaves), target input and
  target output (local leaves only), with variable counts per leaf, zero allowed.
- **`exchange`.** `SourceExchange` (variable-size ghost chunks, one exchange for all
  levels), `MultipoleExchange` (one per level) and `CoarseExchange` (the gather of the
  coarse blocks for the global upward pass). Buffers are flat and contiguous per level.
- **`operator`.** `FmmSizes` and the level-batched `FmmOperator`: one `&mut self`
  call per level and kind (`p2m`, `m2m` per pass, `m2l`, `p2l`, `l2l`, `l2p`, `m2p`,
  `p2p`), each with the level, the `BoxIndex`, both views of its list, the shared
  inputs and an exclusive output. `PairOperator` and the `PerPair<P>` adapter serve
  simple per-pair operators.
- **`evaluator::Evaluator`** runs the unchanged distributed pass order in six public
  stages: source exchange, local upward pass, global upward pass, multipole exchange,
  downward pass, leaf evaluation. Every rank enters every collective, also with empty
  input. Every accumulation follows the order of the target's row.
- **`index_fmm`.** `IndexFmm` (per pair) and `BatchedIndexFmm` (walking the rows or the
  groupings) propagate leaf indices and check the whole distributed compute graph
  exactly, with any per-leaf counts.
- **Still not provided** ("Future extensions" in the crate docs): overlap of exchange
  and computation (C5.2) and device-resident data. The redistribution of points to
  their owning ranks is designed (redesign §9) and implemented in C5.1.

## 2. Workspace layout

The new crates form a strict layering with `nd-fmm-math` at the bottom. `nd-fmm-exec` is
the only new crate that sees `nd-fmm-plan`, the octree and MPI; CubeCL is reached only
through `nd-fmm-kernels`. Arrows mean "depends on"; shaded boxes are existing crates.

```mermaid
flowchart TB
  exec["nd-fmm-exec<br/>Phase 3–4 · Laplace FmmOperator, FMM object"]
  plan["nd-fmm-plan<br/>existing · lists, pass order, ghost exchange"]:::existing
  octree["nd-octree<br/>existing · mature"]:::existing
  tables["nd-fmm-tables<br/>Phase 2 · op. tables"]
  kernels["nd-fmm-kernels<br/>Phase 4 · CubeCL 0.11.0-pre.4; member, not default"]
  ref["nd-fmm-ref<br/>Phase 1 · f64 oracle"]
  math["nd-fmm-math<br/>Phase 0 · harmonics, rotation blocks, layout, RealScalar"]:::phase0
  validate["nd-fmm-validate<br/>Phase 1 · dev tooling; uses every crate"]
  simd["nd-fmm-simd<br/>Phase 3S · SIMD P2P: NEON, AVX2"]
  exec --> plan
  exec --> octree
  plan --> octree
  exec --> tables
  exec -. "feature gpu" .-> kernels
  exec --> simd
  simd --> math
  tables --> ref
  ref --> math
  kernels --> math
  classDef existing fill:#eeeeee,stroke:#888888
  classDef phase0 stroke:#2b6cb0,stroke-width:2px
```

`nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-simd` and `nd-fmm-kernels` stay
free of MPI and of the octree, so Phases 0 to 2 and the kernel tests of Phases 3S and 4
build and test without an MPI runtime. `nd-fmm-exec` inherits the MPI
requirement from `nd-fmm-plan` and `nd-octree`.

*As built in Phase 4:* `nd-fmm-exec` depends on `nd-fmm-kernels` only through its
optional feature `gpu` (the dashed arrow), and `nd-fmm-validate` the same way. Neither
feature is on by default, and `nd-fmm-kernels` is a workspace member but not a default
member (decision 7 of docs/phase4/README.md, signed off with T4). The default members and
the root CI job therefore never build CubeCL; the CI job `run-tests-kernels` builds and
tests `nd-fmm-kernels` alone on the CubeCL CPU runtime, without MPI.

## 3. Crate specifications

Each crate has one job and a public surface small enough to describe in a few lines. Only
`fmm-kernels` depends on CubeCL, and only `fmm-exec` depends on `nd-fmm-plan`.

| Directory | Package | Created in | Purpose | Depends on |
| --- | --- | --- | --- | --- |
| `fmm-math` | `nd-fmm-math` | Phase 0 (done) | real solid harmonics and gradients, rotation blocks, index layout, scalar trait | `num-traits` |
| `fmm-ref` | `nd-fmm-ref` | Phase 1 (done) | f64 reference operators (direct O(p⁴) and rotation O(p³)), P2P, direct-sum oracle | `nd-fmm-math`, `num-traits` |
| `fmm-tables` | `nd-fmm-tables` | Phase 2 (done) | M2M/L2L (8 octants each), M2L (316 offsets, symmetry classes), rotation tables, versioned cache, later SVD compression | `nd-fmm-math`, `nd-fmm-ref`, `num-traits`, `thiserror` (cache errors); `rlst` (without its `mpi` feature) only with SVD compression (C6.2); no serialiser, as the cache writes its own little-endian format |
| `fmm-exec` | `nd-fmm-exec` | Phase 3 (host, done), Phase 3S (SIMD P2P, done), Phase 4 (device, done) | the level-batched `FmmOperator` for Laplace (`LaplaceOperator`), box geometry from integer Morton keys, user-facing FMM object, M2L strategy selection, host threading, the choice of P2P kernel; behind `gpu` the device path (`DeviceOperator` with a host fallback per kind) and the autotune of its choices | as built in Phase 3: `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables`, `nd-fmm-plan`, `nd-octree`, `mpi` (feature `derive`), `rlst` (no features of its own; `nd-octree` enables `mpi`), `rayon`, `thiserror`; dev-dependency `proptest`. Phase 3S added `nd-fmm-simd`; Phase 4 added `nd-fmm-kernels` (optional, feature `gpu`; features `cpu`, `metal` and `cuda` enable `gpu` and the backend). No direct CubeCL dependency and no `unsafe` |
| `fmm-simd` | `nd-fmm-simd` | Phase 3S (done) | hand-written SIMD kernels for the host path, P2P first: `core::arch` intrinsics for aarch64 NEON and x86_64 AVX2 + FMA (AVX-512 deferred), a scalar fallback, runtime ISA dispatch ([simd-p2p.md](simd-p2p.md)) | `nd-fmm-math`, `thiserror`; dev-dependencies `nd-fmm-ref`, `proptest`. No MPI, no external SIMD crate |
| `fmm-kernels` | `nd-fmm-kernels` | Phase 4 (done) | every `#[cube]` kernel behind safe wrappers: backends and the f64 capability check, device buffers, data movement, the plan's views on the device, P2P, the leaf operators, the grouped translations (M2M, L2L, dense M2L), rotation M2L, timing windows; every kernel on every runtime, the runtime a run-time value | `cubecl` =0.11.0-pre.4, `cubek-matmul` and `cubek-std` =0.3.0-pre.4 (the pinned workspace entries), `nd-fmm-math`, `thiserror`; dev-dependencies `nd-fmm-ref`, `nd-fmm-tables`, `proptest`. No MPI, no `nd-fmm-plan`, no rayon. Features `cpu`, `metal` (wgpu with the MSL compiler), `cuda` (type-checked only), none by default. A workspace member, not a default member |
| `fmm-validate` | `nd-fmm-validate` (`publish = false`) | Phase 1 (done), grows with each phase | error norms, point distributions, accuracy sweeps, the FMM accuracy reports and the calibration of p, benchmarks | all of the above, as dev tooling; in Phase 1 `nd-fmm-math` and `nd-fmm-ref`, in Phase 2 also `nd-fmm-tables`, in Phase 3 also `nd-fmm-exec` and `mpi` (so building it needs MPI), in Phase 3S also `nd-fmm-simd`, in Phase 4 also `nd-fmm-kernels` (optional, feature `gpu`, with `cpu`, `metal` and `cuda` passed through). No library crate depends on it; the spike `spikes/p2p-simd` does (Phase 3S T7) |
| `fmm-plan` | `nd-fmm-plan` (existing; rewritten in Phase 3) | before this plan; rewritten in Phase 3 (T4–T7) | kernel-agnostic plan of a distributed FMM: box index, lists, stores, ghost exchange, batched operator interface, evaluator | as built: `nd-octree`, `rlst` (feature `mpi`), `mpi` (feature `complex`, kept for complex-valued operators downstream), all from `[workspace.dependencies]`; dev-dependency `rand_chacha`. No rayon: threading lives in `nd-fmm-exec` |

Dropped after scouting:

- **`fmm-tree`.** It was to implement a `TreeView` trait for the octree types. There is
  no such trait: `nd-fmm-plan` works on `nd_octree::Octree` directly.
- **`fmm-dist`.** Ghost exchange of sources and multipoles already exists in
  `nd-fmm-plan` (`FmmEvaluator` then, `Evaluator` since the Phase 3 rewrite). What
  remains of Phase 5 is the redistribution of points (C5.1), multi-rank validation of
  the Laplace operator, and overlap of communication and computation. The overlap
  belongs in `nd-fmm-plan`.

Outside the crates, Phase 0 also adds these folders:

- `docs/CONVENTIONS.md`;
- `tools/fixtures/`, the fixture generator (see `docs/phase0/`);
- `spikes/cubecl-gemm/`, excluded from the default workspace members so CI does not build
  CubeCL.

Phase 3S adds `spikes/p2p-simd/`, also outside the default members: the T2 spike and
the T7 example `compare`, the only code that uses green-kernels. Phase 4 adds
`spikes/device-arith/` (T3: device arithmetic per backend, and a CPU-shaped P2P on the
CPU runtime against `nd-fmm-simd`), likewise outside the default members, and ports
`spikes/cubecl-gemm/` to CubeCL 0.11.0-pre.4 (T2).

### 3.1 Public surface per crate

**`nd-fmm-math`**

- `trait RealScalar` is implemented for `f32` and `f64`. It is the only float bound used
  across the new crates. It must not require `mpi::Equivalence`, so that the math crate
  stays MPI-free; `nd-fmm-exec` adds that bound where it meets `FmmOperator::Value`.
- `struct Layout { p }` provides `len()`, `idx(n, m)`, `nm(i)` and iteration by degree.
- These functions write into caller-provided slices and do not allocate:
  - `harmonics::regular(p, x, out)` and `harmonics::irregular(p, x, out)`
  - `harmonics::regular_grad(p, x, value, [gx, gy, gz])` and
    `harmonics::irregular_grad(p, x, value, [gx, gy, gz])`, which also write the values
  - `rotation::blocks(p, &q, out)`: per-degree real blocks Dⁿ(Q) for any rotation Q,
    contiguous, row-major; `rotation::blocks_len(p)`, `rotation::block_range(n)`
  - `rotation::to_irregular(p, blocks)`: S Dⁿ S⁻¹ in place, for irregular harmonics
- `rotation::euler_zyz(alpha, beta, gamma)` returns the 3 × 3 rotation matrix.
- `Layout` is built with `Layout::new(p)` (p: `usize`) and also offers `degree(n)` and
  `degrees()`.
- As implemented in Phase 0, there is no `legendre` module (harmonics use the Cartesian
  recursions of CONVENTIONS §3.5 directly). There is no `wigner::d_blocks(p, beta, …)`
  either: `rotation::blocks` takes a full rotation instead of a polar angle.
- `CONVENTION_VERSION: u32` is bumped whenever a convention in `CONVENTIONS.md` changes;
  a test checks it against the file.

**`nd-fmm-ref`**

- `Frame<T> { centre, radius }` is the centre and scaling radius of an expansion
  (CONVENTIONS §3.7). `Frame::new(centre, radius)` panics unless radius > 0, and
  `Frame::scaled(x)` returns (x − c)/r. Operators take frames, never shift vectors.
- `Workspace<T>` holds every temporary. `Workspace::new(p)` allocates it for degree at
  most p, and `p()` returns that degree.
- Free functions over plain slices, generic over `T: RealScalar`:
  - `leaf::p2m` and `leaf::p2l` `(p, &frame, sources, charges, &mut ws, out)`
  - `leaf::l2p` and `leaf::m2p` `(p, &frame, coefficients, targets, &mut ws, potential,
    gradient: Option<&mut [[T; 3]]>)`
  - `direct::{m2m, l2l, m2l}` `(p, &from, &to, &mut ws, input, output)`: the O(p⁴) sums
    of CONVENTIONS §3.11. `rotation::{m2m, l2l, m2l}` have the same signatures and work
    in O(p³): rotate onto the z-axis, coaxial translation, rotate back, with the
    rotation blocks built per call.
  - `p2p::p2p(sources, charges, targets, potential, gradient)`.
- `p2p::direct_sum(...)` has the same arguments in f64 only and sums with Neumaier
  compensation. Both P2P functions skip exactly coincident pairs.
- Every operator accumulates (+=), applies no 1/(4π) and does not allocate. It panics
  on a slice of the wrong length or a workspace built for a smaller p.
- As implemented in Phase 1, the leaf operators live in `leaf`, shared by both
  translation methods, not in `direct` and `rotation`. There is no `rayon` dependency.

**`nd-fmm-tables`** (Phase 2; MPI-free)

- Every family is generic over `T: RealScalar`, built in f64 with `build(p)` at the
  canonical frames of CONVENTIONS §3.12 and rounded entry by entry to `T`.
- Every application accumulates (+=), applies no 1/(4π) and does not allocate; any
  temporaries come from caller-owned scratch.
- `MatrixSet<T>` holds `count` square matrices of order n, column-major and contiguous
  (§3.12, "Matrix layout"). It offers `matrix(i)`, `as_slice()`, `apply(i, x, y)`
  (y += Aᵢ x) and `cast()`.
- `geometry` restates the octant and offset order of §3.12 instead of importing it from
  `nd-octree` or `nd-fmm-plan` (C3.1 checks that they agree):
  - `octant_direction(o)` gives the sign vector s_o of child index o = 4x + 2y + z;
  - `m2l_offsets()` gives the 316 offsets d = index(target) − index(source) in
    lexicographic order, and `m2l_offset_index(d)` their closed-form table index;
  - `m2m_frames(o)`, `l2l_frames(o)` and `m2l_frames(index)` give the canonical frames.
- `M2mTables` and `L2lTables` hold 8 matrices each, applied with `apply(o, input,
  output)`.
- `M2lTables` holds the 316 dense matrices, applied with `apply(index, multipole,
  local)`. `m2l::build_matrices(p, indices)` builds chosen offsets only.
- `M2lClasses` is the 16-class form: the class matrices, T_M(P) and T_L(P) of the 48
  elements of O_h, and each offset's class and element. It offers `apply(index, m, l,
  &mut M2lScratch)` without expanding, and `expand()` to the dense tables.
  - `symmetry` provides `SignedPermutation`, `class_representatives()`, `class_of`
    and `CoefficientTransform`.
- `RotationTables` holds the rotation and coaxial tables of the 316 offsets and the 8
  octants, one `ShiftTables` per `rotation::Operator`. It offers `m2l(index, …)`,
  `m2m(o, …)` and `l2l(o, …)`, each with a `RotationScratch`, in O(p³) and building no
  block per call.
- `cache::TableCache::new(dir)` works in a directory the caller chooses, and offers
  `load::<F>(p)`, `store(&F)` and `load_or_build::<F>(p) -> (F, CacheOutcome)`.
  - F is any of the five families in f32 or f64 (`CachedTable`).
  - A file is keyed by `TableKind`, p, precision, `CONVENTION_VERSION` and
    `FORMAT_VERSION`; its header repeats the key and holds an FNV-1a checksum of the
    payload.
  - Writes are atomic. A stale, mismatched or corrupt file is rejected, and
    `load_or_build` then rebuilds it (`CacheOutcome::Rebuilt`).
- As implemented in Phase 2, the cache is generic over the family rather than taking
  `(p, precision)`. The offsets and octants are restated in `geometry`, not keyed by
  `nd-fmm-plan` types. There is no `rlst` and no serialiser.

**`nd-fmm-exec`** (Phase 3, host path; as built)

- `geometry`: the validated cubic `Domain` (`Domain::new` rejects a box whose sides
  differ by more than `SIDE_TOLERANCE` = 4 times ε (|corner| + side), the rounding of
  `compute_global_bounding_box`), `radius`,
  `centre`, `integer_centre`, `relative_frame::<T>(s, t)` (the exact dyadic frame of
  box s seen from box t, CONVENTIONS §3.13), `leaf_coordinates` and `contains`.
- `tables`: `enum M2lStrategy { Dense, Classes, Rotation, Auto }` (`Auto`: `Dense`
  for p ≤ 8, `Rotation` above; `resolve(p)`), and `Tables<T>`, which builds or loads
  (`load_or_build` with a `TableCache`) only the families the strategy uses and applies
  `m2m(o, …)`, `l2l(o, …)` and `m2l(index, …)` by octant and offset index with a
  `TableScratch`. `Compressed` comes with C6.2.
- `operator`: `LaplaceOperator<T>` implements `FmmSizes`, the batched `FmmOperator` and
  `PairOperator`. `new(tables, gradients, max_leaf_points)` sizes its scratch;
  `with_pool(Arc<ThreadPool>)` adds per-thread scratch and runs every level call in
  the pool; `set_serial` runs a threaded build serially. One public method per pair or
  leaf (`p2m_leaf`, `m2m_pair`, `m2l_pair`, …). A source point has 4 values
  (`SOURCE_POINT_SIZE`; a chunk of n points holds the n leaf-scaled coordinate
  triples, then the n charges), a target point 3 of input. No operator allocates or
  applies 1/(4π).
- `fmm`: `FmmBuilder<T>::new(p)` with `strategy`, `gradients`, `max_level` (default
  16), `max_points_per_leaf` (default `DEFAULT_MAX_POINTS_PER_LEAF`, 64), `domain`,
  `table_cache(dir)`, `threads(n)` (default 1) and, since Phase 3S, `p2p_kernel`
  (default `P2pChoice::Auto`); `build(sources, targets, comm)` (collective) gives
  `Fmm<'o, T, C>`. `Fmm::evaluate(charges) -> Result<Output<T>, FmmError>` returns φ
  and optional ∇φ in the caller's order with 1/(4π) applied once (CONVENTIONS §3.1),
  and `StageTimings`. Accessors: the octree, plan, operator, domain, counts,
  `list_sizes`, `cache_outcomes`, `build_timings`, `threading` and `p2p_kernel` (the
  kernel as it runs, never `Auto`). `FmmError` and
  `SettingsError` cover every input error, agreed on every rank; until C5.1, points
  in another rank's leaves give `PointsNotOwned` on every rank. `MAX_DEGREE` = 20.
- `threading`: `ThreadingReport` (rayon threads, MPI level provided, the five BLAS
  variables, `warnings()`), `REQUIRED_MPI_THREADING` = `Funneled`, `BLAS_VARIABLES`.
- Phase 3S (T6): `operator::P2pChoice { Auto, Reference, Isa(Isa) }` (default `Auto`,
  the widest ISA; `Reference` is `nd_fmm_ref::p2p`, the Phase 3 path), with
  `resolve()`, `Display` and `FromStr` (`auto`, `reference`, ISA names);
  `LaplaceOperator::with_p2p(choice)` and `p2p_kernel()`; `SettingsError::
  P2pIsaUnavailable`, agreed on every rank like every input error. `operator`
  re-exports `Isa`, `IsaUnavailable` and `SimdScalar`, and `LaplaceOperator<T>` has
  the bound `T: SimdScalar`. P2P stays one kernel call per source leaf; the module
  has no `unsafe`.
- As built in Phase 3 there was no backend parameter: the host path was the only one
  until Phase 4 added the device path behind the same operator interface.
- Phase 4 (T5–T12; feature `gpu`; [device-path.md](device-path.md) §3.3), with
  `Backend::Host` the default and the host path unchanged without the feature:
  - `fmm`: `Backend { Host, Cpu, Metal, Cuda }` (`FromStr`, `probe()`), `OperatorKind`,
    `Placement`; `FmmBuilder::{backend, host_fallback, device_p2p_layout,
    device_leaf_layout, device_gemm, device_scratch_budget, synchronous_stages,
    device_timestamps, tuning_cache, tuning_budget, tuning_hook}` with
    `DeviceP2pLayout`, `DeviceLeafLayout` and `DeviceGemm { Auto, Library, HandWritten }`;
    `SettingsError::{BackendNotCompiled, NoDevice, PrecisionUnsupported, DeviceMemory,
    DeviceNeedsOneRank}` and `FmmError::Device`; `Fmm::{backend, placement,
    device_report, device_counters, download_device_views, expansions}`;
    `BuildTimings::device`; `StageTimings::device` with `DeviceStage` and
    `DeviceStageTimings` (T11).
  - `device`: `DeviceOperator` (every kind on the device by default, a host fallback per
    kind through the wrapped `LaplaceOperator`), `DeviceReport` (device, placement,
    tables, layouts, the GEMM of every translation level call, memory, stage timing,
    tuning), `DeviceCounters` with `Counters` and the traffic by kind of data
    (`DataKind`, `Traffic`), `TranslationReport`, `RotationReport`, `StageTiming`,
    `RotationHostArrays`.
  - `tune` (T12): the strategy-level tuner and its cache (`TuningKey`, `TuningReport`,
    `Decision`, `Candidate`, `Source`, `Timing`, `CacheState`, `TuningCache`, `Tuner`,
    `TuningHook`), the static rule (`static_strategy`, `static_gemm`, `static_p2p`,
    `STATIC_F64_DENSE_MAX_P` = 11), `CANDIDATE_SET_VERSION`.
  - No direct CubeCL dependency and no `unsafe`: CubeCL only through `nd-fmm-kernels`.

**`nd-fmm-plan`** (existing; rewritten in Phase 3, see Section 1.1)

- `index::{BoxIndex, LeafNumbering, NONE}`; `lists::{Csr, GroupedCsr, VList, Children,
  Parents, LevelLists, offset_index, NOFFSETS, NOCTANTS}`; `plan::{Plan, PlanError}`.
- `store::{LevelBuffers, LeafStore}` and their slice types (`LevelSlice`,
  `LevelSliceMut`, `LeafSlice`, `LeafSliceMut` with `split_at_mut`).
- `exchange::{SourceExchange, MultipoleExchange, CoarseExchange, ExchangeError}`.
- `operator::{FmmSizes, FmmOperator, PairOperator, PerPair, UpwardPass}` and the batch
  types `P2m`, `M2m`, `M2l`, `P2l`, `L2l`, `L2p`, `M2p`, `P2p`.
- `evaluator::{Evaluator, EvaluatorError}`: `new(plan, comm, op, source_counts,
  target_counts)`, `reset`, `evaluate` and the six public stages. The `Evaluator` can
  own its plan (`P: Borrow<Plan>`), which lets `Fmm` hold both.
- `index_fmm::{IndexFmm, BatchedIndexFmm, Walk, GlobalLeaves, check_counts,
  run_index_fmm, IndexPath}`; `interaction_manager::V_LIST_DIRECTIONS`.
- Examples `test_index_fmm` (registered with templated-examples), `evaluator_stage_cost`
  and `plan_build_cost`.

**`nd-fmm-simd`** (Phase 3S, T3–T5; MPI-free; as built, [simd-p2p.md](simd-p2p.md) §5)

- `Isa { Scalar, Neon, Avx2 }`, `#[non_exhaustive]` so that AVX-512 can follow, with
  `detect()` (the widest available), `is_available()`, `all()`, `available()`,
  `lanes::<T>()` and `Display` (`scalar`, `neon`, `avx2`). Every variant exists on
  every target; those of another architecture are never available. `IsaUnavailable
  { isa }` is the error for one this machine cannot run.
- `SimdScalar`, sealed, for f32 and f64.
- `P2pKernel<T>` (`Copy`, `Send`, `Sync`): `new(isa) -> Result<_, IsaUnavailable>`,
  `detect()`, `isa()` and `evaluate(sources, charges, targets, potential, gradient)`,
  with the signature and semantics of `nd_fmm_ref::p2p::p2p`. It excludes a pair by
  r² = 0 (CONVENTIONS §3.13, "Fast kernels"), adds each target's sources in input
  order, and is bit-invariant under splitting the sources or moving the targets.
  Targets in lanes; NEON uses FSQRT and FDIV with K = 2 (f32) and 4 (f64) vectors per
  block, AVX2 `vrsqrtps` with a polynomial correction and K = 1; the scalar path is
  the reference's loop.
- `rsqrt::rsqrt_slice(isa, x, out)`: the kernel's inverse square root, within 4 u_T.
- Internals: `arch::{scalar, neon, avx2}` (the vector layer per ISA, `cfg`-gated),
  the generic kernel body `arch::p2p`, and `#[target_feature]` entry points
  `arch::{neon, avx2}::p2p_f{32,64}[_gradient]`; `unsafe` only there and in their
  dispatch. `tools/inner_loops.awk` counts the instructions of their inner loops.

**`nd-fmm-kernels`** (Phase 4, T4–T12; MPI-free; as built, [device-path.md](device-path.md) §3.1)

- The runtime is a run-time value, never an `R: Runtime` type parameter (CubeCL 0.11's
  `Client` is not generic; device-path.md §3.2). `BackendKind { Cpu, Metal, Cuda }`,
  `Device::open(kind)` (a Metal device that comes up without the MSL compiler is refused
  as `NoDevice`), `DeviceInfo` (backend, compiler, name, CubeCL version, plane size,
  precisions; one-line `Display`), `Precision`, the capability check `supports` /
  `require` (f64 on Metal: `KernelError::UnsupportedPrecision`), `Counters` (transfers,
  bytes, launches, syncs, timing windows), `limit_units` / `units_cap` (the CPU
  runtime's units per cube, default `CPU_MAX_UNITS`), `available_memory`,
  `open_window` / `close_window` / `times_on_device` and `WindowTime` (T11),
  `CUBECL_VERSION` and `VERSION` (keys of the tuning cache, T12).
- Buffers: `DeviceBuffer<E>` of f32, f64 or u32 (`DeviceElement`, `DeviceFloat`), with
  `DeviceSlice` / `DeviceSliceMut`; `IndexBuffer` with one bound over the whole buffer,
  checked before every unchecked launch. Round trips are bit for bit (−0, subnormals,
  extremes, NaN payloads).
- `movement`: `zero`, `gather_columns`, `gather_coefficients`, `scatter_add_columns`,
  `scatter_values`; no atomics, distinct scatter indices per launch (checked in debug).
- `view`: the plan's arrays on the device without an `nd-fmm-plan` dependency
  (`IndexView`, `GroupedView` with its row-to-batch map, `BoxCoordinates`,
  `LeafCoordinates`, `PointOffsets`), validated on the host at upload.
- `p2p`: `p2p` (one launch per level call), `P2pLayout { Cube, Plane, Cpu }`,
  `P2pInputs`, `near_frames`.
- `leaf`: `p2m`, `p2l`, `l2p`, `m2p` (one launch per level call, p comptime up to
  `MAX_DEGREE` = 20), `LeafLayout { Cube, Cpu }`, `SourceInputs`, `TargetInputs`,
  `harmonics`, `x_frames`, `w_frames`.
- `translate`: the grouped translation of device-path.md §6.4 (gather, one grouped GEMM,
  row-ordered reduction or scatter-add, in chunks within a scratch budget, default
  `DEFAULT_SCRATCH_BYTES` = 128 MB): `Tables`, `GroupedPlan` (`PlanSettings`, `PlanSize`,
  `TileSchedule`), `TranslationScratch`, `grouped`, `grouped_stage` (one stage, for
  profiling), `PerGroupPlan` / `per_group` (structure (A), the test reference), the
  hand-written GEMM `gemm` (`GemmLayout { Cube, Cpu }`) and the library GEMM `library`
  (`cubek-matmul`'s `SimpleCyclicCmma`, named explicitly; `GemmPolicy`, `Gemm`,
  `library_stride`), `Accumulate { Rows, Scatter }`, `Orientation` (box-major, the
  default; coefficient-major, tested, not used by `nd-fmm-exec`).
- `rotation`: `RotationTables` (the M2L family of the Phase 2 rotation tables, uploaded
  in their own storage from `RotationArrays`), `RotationPlan`, `m2l` (one launch per
  level), `RotationLayout { Cube, Cpu }`.
- `KernelError` for every refusal; every public function is safe, and `unsafe` is
  confined to CubeCL launches and buffer views, each block with `// SAFETY:`.
- Backend features `cpu`, `metal` and `cuda` (no `hip`, Vulkan or WebGPU build), none by
  default; `nd-fmm-exec` and `nd-fmm-validate` pass them through. Kernels are generic
  over f32 and f64, with p, tile sizes and layouts as comptime parameters, and every
  layout is chosen per backend from `DeviceInfo`.

**`nd-fmm-validate`** (Phase 1; MPI since Phase 3, and no library crate depends on it)

- `SplitMix64`: the seeded generator of the `nd-fmm-math` tests, with `new(seed)`,
  `next_u64()`, `uniform()` and `range(lo, hi)`.
- `points::{cube, ball, sphere}` draw uniform points, `points::plummer` the Plummer
  sphere (truncated at 10 a) and `points::gaussian_clusters` Gaussian clusters
  (truncated at 4 σ) (Phase 3, T11), and `points::charges` draws charges in [−1, 1).
- `metrics`:
  - `ErrorNorms { l2, max }`, `potential_errors` and `gradient_errors`, and
    `ErrorAccumulator` to pool several target sets;
  - `potential_magnitudes` and `gradient_magnitudes`, with
    `potential_error_relative_to_magnitude` and
    `gradient_error_relative_to_magnitude`;
  - `CoefficientKind::{Multipole, Local}`, `weight(kind, n, m)`, `degree_norms`,
    `degree_errors` and `max_relative_degree_error`, per degree in the §3.8 weighting.
- `accuracy`: `sweep::<T>(&config, p_max, translations) -> Vec<Row>` runs the five
  `Chain`s over the 316 offsets of `nd_fmm_tables::geometry::m2l_offsets()`, with the
  translations of `nd_fmm_ref::direct` or the dense tables of `nd-fmm-tables`
  (`Translations::{Reference, Tables}`, Phase 2); also `Config`, `Errors`, `Row`,
  `SOURCE_CENTRE` and `RADIUS`.
- `bench` (Phase 2): the helpers of the timing reports, `median_time_per_call`,
  `fitted_exponent`, `crossover` and the machine description.
- `fmm_accuracy` (Phase 3, T9 and T11): `Distribution` (cube, sphere, Plummer,
  clusters), `Config` (`Config::C32`, `Config::c33(d)`), `Problem`, `Oracle`, `Run`
  and `run::<T>(config, problem, charges, oracle, (p, threads), comm)`, the complete
  FMM of `nd-fmm-exec` against `direct_sum` over several charge vectors; `PREDICTION`.
- `calibration` (Phase 3, T12): `config(d, n)`, `Reference` (the problem and both
  oracles, once per distribution), `Precision`, `sweep`, `leaf_study`, `Measure`,
  `smallest_p`, `worst`, `floor` and `Reached`; `F64_DEGREES` (1..=20),
  `F32_DEGREES` (1..=8), `TARGET_EXPONENTS` (3..=12), `LEAF_SIZES`. Phase 3S (T7) adds
  `leaf_size_rule` and `LeafSizeChoice`, the rule that set the default leaf size.
  `fmm_accuracy::run`, `sweep` and `leaf_study` take an `Execution { threads, p2p }`
  since T6, with `backend` since Phase 4 T11.
- `p2p_kernels` (Phase 3S, T7; no MPI): the workloads W1 and W2 of simd-p2p.md §8.2
  (`Cell`, `Form`, `Set`, `w1`, `w2`; the inputs of the T2 spike), `Kernel` (the
  reference or `P2pKernel` on an ISA), `Oracle`, `Accuracy`, `accuracy`, `passes`
  (requirement 2 as decided in T5), `check`, `per_pair_equals_gathered`,
  `pairs_per_second`, `time_kernel` and `model` (design §4.6 as corrected by the
  spike).
- `bench` gains `performance_cores()` and `toolchain()` (Phase 3S).
- Examples `accuracy` (with `--tables` for the table path), `timing`, `tables`,
  `fmm_accuracy` (`--distribution`, `--threads`, `--p2p`) and `calibrate`
  (`--threads`, `--n`, `--p2p`) print Markdown reports; no timing is asserted. Phase 3S
  adds `p2p_kernels` (`--quick`, `--ghz`; no MPI), the kernels against the reference,
  and `p2p_fmm` (`--threads`, `--n`, `--part`), every P2P kernel in the FMM and the
  leaf-size study. `fmm_accuracy`, `calibrate` and `p2p_fmm` initialise MPI and run on
  one rank; they are not registered with templated-examples. The one MPI-initialising
  test is `tests/fmm_accuracy.rs`; `tests/p2p_kernels.rs` needs no MPI.
- Phase 4 (feature `gpu`, with a backend; every device example takes `--device` or
  `--backend` and runs on one rank, by hand, never in CI):
  - `p2p_device` (T6): the W1 and W2 workloads as device P2P calls, for `p2p_kernels
    --device` (and `--fmm`, the leaf stage inside the FMM);
  - `fmm_accuracy::{Execution::backend, Run::device, DeviceRun}` (T11), and since T13
    `fmm_accuracy::{builder, measure}`, which return the built `Fmm` too;
  - `device_fmm` (T13): `Settings`, `Measurement`, `measure`, `compare`, `error_ratios`,
    `bounds`: the device FMM against the host path, stage by stage, with its build,
    transfers, launches and syncs, and the device-against-host check of C4.8;
  - examples `device_fallback` (T5), `leaf_kernels` (T7), `translation_kernels` (T8),
    `m2l_kernels` (T9, T12 `--orientation`), `rotation_kernels` (T10), `fmm_accuracy
    --backend` (T11), `autotune` (T12) and `device_fmm` (T13: the device FMM against the
    host and the device leaf-size study); `p2p_kernels --device` (T6). The test
    `tests/device_fmm.rs` (feature `cpu`) is a smoke run of the `device_fmm` core on the
    CPU runtime, in its own MPI-initialising executable.

## 4. How `nd-fmm-plan` and the octree connect

`nd-fmm-plan` decides *what* to compute and *when*: interaction lists, pass order, ghost
exchange and the global coarse levels. The new crates decide *how*, for the Laplace
kernel. The seam is `nd-fmm-plan`'s level-batched `FmmOperator` (Section 1.1); no tree
trait is needed. Phase 3 rewrote `nd-fmm-plan` to provide it
([fmm-plan-redesign.md](fmm-plan-redesign.md)).

| Concern | Owner | Status |
| --- | --- | --- |
| Tree construction, partitioning, ghost keys | `nd-octree` | exists |
| Morton-ordered integer box index per level | `nd-fmm-plan` (`BoxIndex`) | done in Phase 3 (T4) |
| U, V, W, X lists as index arrays; groupings by V-list offset and child octant | `nd-fmm-plan` (`Plan`, `lists`) | done in Phase 3 (T4) |
| Pass order, level loop, which lists feed which operator | `nd-fmm-plan` (`Evaluator`) | done in Phase 3 (T6); order unchanged |
| Ghost exchange of sources and multipoles, global coarse levels | `nd-fmm-plan` (`exchange`, `Evaluator`) | done in Phase 3 (T5); host buffers, flat per level |
| Variable-size source and target data per leaf | `nd-fmm-plan` (`LeafStore`) | done in Phase 3 (T5, C3.0) |
| Batched operator hooks (per level, per octant, per V-list offset) | `nd-fmm-plan` (`FmmOperator`, batch types) | done in Phase 3 (T6, C4.0) |
| Redistribution of points to their owning ranks | `nd-fmm-plan` | designed (redesign §9); C5.1 |
| Overlap of exchange with local work; device-resident buffers | `nd-fmm-plan` | missing; Phase 4–5 |
| Box geometry from keys, M2L strategies, host threading, 1/(4π) | `nd-fmm-exec` | done in Phase 3 (host path) |
| SIMD P2P kernels on the host | `nd-fmm-simd`, called by `nd-fmm-exec` | Phase 3S |
| Device buffers and kernels | `nd-fmm-exec`, `nd-fmm-kernels` | Phase 4 |
| Operator math and tables | `nd-fmm-math`, `nd-fmm-ref`, `nd-fmm-tables` | done (Phases 0–2) |

How Phase 3 closed the two gaps of the original crate:

- **Fixed size per leaf.** `source_size()` was one number for every leaf, but particle
  counts per leaf vary and are unbounded at `max_level`. The rewrite holds sources,
  target input and target output in CSR stores with per-leaf counts, zero allowed, and
  exchanges variable-size ghost chunks. Sizes per point are fixed.
- **One pair at a time.** The per-pair interface with `HashMap` lookups ruled out
  batched GEMM. The rewrite makes one call per level and kind. Each call carries the
  target-centric rows, which the host path walks target by target and threads with
  `par_chunks_mut`, and the groupings by offset (M2L) and octant (M2M, L2L), from which
  the GEMM path of Phase 4 gathers its columns. A per-pair adapter (`PerPair`) keeps the
  per-pair form for `IndexFmm`, tests and reference paths.

Both are general, not Laplace-specific, and are checked with `IndexFmm` on 1, 2 and 4
ranks, with counts of one and with seeded variable counts.

Phases 0 to 2 depend on none of this, because they never touch a tree.

## 5. Cross-cutting conventions

Every new crate follows the same template, so Claude Code can create one from a single
instruction and every crate looks the same to a reviewer. The existing crates keep their
own manifests and rules; they are not migrated to the template as a side effect of
another task.

### 5.1 Rules

| Topic | Rule |
| --- | --- |
| Naming | directory `fmm-<name>`, package `nd-fmm-<name>`, library name `nd_fmm_<name>` |
| Edition and versions | Rust 2024; edition, licence and repository inherited from `[workspace.package]` (added in Phase 0 T1) |
| Dependencies | declared once in `[workspace.dependencies]`, used with `.workspace = true`; `cubecl` pinned to an exact version there; `mpi` and `rlst` at the versions the existing crates use (0.8.2, 0.9.0) when a new crate needs them |
| Precision | all numeric code generic over `T: RealScalar` (from `nd-fmm-math`); no `f64` hard-coding outside tests and table building. Does not apply to `nd-fmm-plan`, whose `FmmOperator::Value` is deliberately generic (its `IndexFmm` uses `u32`) |
| Allocation | kernels and hot loops write into caller-provided slices; allocation only in constructors and plan building |
| Errors | `thiserror` enums in public APIs; panics only for violated internal invariants (`debug_assert!`) |
| Unsafe | none outside `nd-fmm-kernels` (as built in Phase 4: CubeCL launches and buffer views only) and, from Phase 3S, the architecture modules and ISA dispatch of `nd-fmm-simd` (intrinsics, `#[target_feature]` calls after detection; [simd-p2p.md](simd-p2p.md) §5.4); spikes are exempt; each unsafe block carries a `// SAFETY:` comment, and every public function stays safe |
| Docs | every public item documented; operator functions cite the equation they implement in `docs/CONVENTIONS.md` |
| Tests | unit tests in-crate; property tests with `proptest`; fixtures under `<crate>/fixtures/`, small and committed |
| MPI tests | only in crates that depend on MPI (`nd-fmm-exec` and later): one MPI-initialising test per test executable, as in `nd-octree` and `nd-fmm-plan`; run tests with `RUST_MIN_STACK=8388608`; multi-rank runs by hand (on macOS with `--mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0`) |
| Formatting and lints | `cargo fmt` after every edit; `cargo clippy --workspace --all-targets -- -D warnings` locally (stricter than CI, which omits `--all-targets`) |
| CI | the root GitHub Actions workflow runs fmt, clippy, tests and docs for default members on CPU and one rank; since Phase 3S a job tests `nd-fmm-simd` alone on x86_64 and arm64, and since Phase 4 (T4) a job `run-tests-kernels` tests `nd-fmm-kernels` alone on the CubeCL CPU runtime, without MPI; GPU tests are run locally and never block CI |

### 5.2 Root `Cargo.toml` additions

Before Phase 0 the root manifest had only `members` and `resolver = "2"`. Phase 0 added
the block below. *As of Phase 4* the CubeCL entries are pinned to `cubecl =0.11.0-pre.4`
(features `std`, `stdlib`), `cubek-matmul =0.3.0-pre.4` (features `std`, `multi-level`)
and `cubek-std =0.3.0-pre.4`, all with `default-features = false`: a pre-release, chosen
on 2026-10-03 because 0.10.0 disabled f64 on CUDA (Phase 4 T2;
`spikes/cubecl-gemm/SPIKE_REPORT.md`, "CubeCL 0.11.0-pre.4"). Only `nd-fmm-kernels` and
spikes use them. With `metal`, CubeCL's own manifests turn on its `persistence` feature
(an empty store `target/environment/default.db` appears; accepted at the T4 sign-off).
Moving to the final 0.11.0 is a separate decision. The workspace also has entries for
every `nd-fmm-*` crate, `mpi`, `rlst`, `rand_chacha`, `rayon`, and for the spikes only
`green-kernels` and `pulp`; `nd-fmm-kernels` is in `members`, not `default-members`.

```toml
[workspace]
resolver = "2"   # keep; moving to "3" (MSRV-aware resolution) is a separate decision
members = ["octree", "fmm-plan", "fmm-math", "spikes/cubecl-gemm"]
# New: without default-members every member is built; the spike must be excluded
# so CI never builds CubeCL.
default-members = ["octree", "fmm-plan", "fmm-math"]

# New. Only new crates inherit from it; octree/ and fmm-plan/ keep their own fields.
[workspace.package]
edition = "2024"
license = "MIT OR Apache-2.0"
repository = "https://github.com/tbetcke/fmm"

[workspace.dependencies]
num-traits = "0.2"
thiserror = "2"
proptest = "1"
approx = "0.5"
serde_json = "1"
# Pinned by Phase 0 T6; only nd-fmm-kernels and spikes/ may use them:
cubecl = { version = "=0.10.0", default-features = false, features = ["std", "stdlib"] }
cubek-matmul = { version = "=0.2.0", default-features = false, features = ["std"] }
cubek-std = { version = "=0.2.0", default-features = false }
# Added when their phases start:
# rlst = "0.9.0"   # SVD compression (C6.2); match the existing crates, never with "mpi" here
# rayon = "1"
# mpi = { version = "0.8.2", features = ["derive"] }   # Phase 3, match existing crates
nd-fmm-math = { path = "fmm-math" }
```

### 5.3 Per-crate `CLAUDE.md` template

Workspace-wide rules (working agreement, formatting, CI and workspace checks, build
environment, MPI) live only in the root `CLAUDE.md`; a crate file holds only what is
specific to the crate, so shared rules cannot drift apart.

```markdown
# nd-fmm-<name>

Purpose: <one sentence from the structure document>.
Phase and components: <e.g. Phase 0, C0.2 and C0.3>.

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
- Generic over T: RealScalar; no allocation in hot paths.
- Before finishing: `cargo clippy -p nd-fmm-<name> --all-targets -- -D warnings`
  and `cargo test -p nd-fmm-<name>` must pass.

## Allowed dependencies
<list; anything else needs a note in the PR>

## Test oracle
<what this crate is checked against: fixtures, identities, nd-fmm-ref>
```

## 6. Order of creation and open questions

Create a crate only when its phase starts; an empty crate adds build time and review
noise without testing anything.

| Phase | Crates, folders and tasks in existing crates |
| --- | --- |
| 0 | `fmm-math`, `docs/CONVENTIONS.md`, `tools/fixtures/`, `spikes/cubecl-gemm/`, root `CLAUDE.md` |
| 1 | `fmm-ref`, `fmm-validate` |
| 2 | `fmm-tables` |
| 3 (done) | the rewrite of `nd-fmm-plan` (box index, lists, variable-size leaf data, batched operator interface; T1, T4–T7); `fmm-exec` (host path on the batched interface, threaded with rayon; T3, T8–T11); the calibration in `fmm-validate` (T12) |
| 3S (done) | `fmm-simd` (SIMD P2P on the host; T3–T5), its use in `fmm-exec` (T6), kernel and FMM benchmarks in `fmm-validate` (T7), `spikes/p2p-simd/` (the T2 spike and the T7 green-kernels comparison) |
| 4 (done) | the CubeCL pin moved to 0.11.0-pre.4 and `spikes/cubecl-gemm/` ported (T2); `spikes/device-arith/` (T3); `fmm-kernels` (T4, kernels in T6–T10; a member, not a default member, with the CI job `run-tests-kernels`); the device backend in `fmm-exec` on the batched interface that Phase 3 delivered, with no `nd-fmm-plan` change (T5–T11) and autotune (T12); device examples and benchmarks in `fmm-validate` (T5–T13) |
| 5 | multi-rank validation of `fmm-exec`; exchange/compute overlap in `nd-fmm-plan` (no new crate) |

### Answered by scouting

- [x] Does the octree expose U, V, W and X lists? No. It gives only same-level neighbours
      and parent/child helpers. `nd-fmm-plan`'s `InteractionManager` builds all four lists.
- [x] What box identifier is used? `MortonKey = u64` everywhere, with `HashMap` lookups
      and no integer box index. Per-level contiguous positions exist only inside
      `nd-fmm-plan`'s `LevelData`.
- [x] Does `nd-fmm-plan` own the pass order? Yes: `FmmEvaluator::evaluate`, generic over
      `FmmOperator`.
- [x] Does the distributed layer wrap MPI? Both crates use rsmpi and rlst directly.
      `nd-fmm-plan` already exchanges sources and multipoles, so no `nd-fmm-dist` is
      needed.

### Still open

- [x] Who designs variable-size leaf data and batched operator hooks in `nd-fmm-plan`, and
      does the per-pair `FmmOperator` stay as the reference path next to them?
      Answered in Phase 3: T1 designed a rewrite of the crate
      ([fmm-plan-redesign.md](fmm-plan-redesign.md), signed off), built in T4–T7.
      The batched `FmmOperator` replaced the per-pair trait. The per-pair form stays
      as `PairOperator` with the `PerPair` adapter, the path of `IndexFmm` and of
      reference operators; `LaplaceOperator` implements both, bit-identically.
- [x] Column order within a level's `LevelData` followed insertion (partly `HashMap`)
      order. Should it be Morton order, for reproducible floating-point sums and
      cache-friendly gathers? Answered in Phase 3: yes. `BoxIndex` numbers every
      level's boxes in Morton order, deterministically, and `LevelBuffers` store
      them in that order. Results are bit-identical from run to run and for any
      number of host threads.
- [x] Box geometry conventions used by the tables are now CONVENTIONS §3.12 (Phase 2
      T2, PR #17, signed off before T3), which §3.10 covers; `CONVENTION_VERSION`
      stays 1. §3.12 states:
  - child octant index 4x + 2y + z;
  - M2L offset sign (target − source, in units of the box width);
  - the shift vector c_target − c_source = 2 r_l · offset.

  It also states the offset order and table index, the canonical frames, the matrix
  layout and the cube symmetry group.
- [x] Stale crate `CLAUDE.md` files from the pre-workspace era were updated on
      2026-09-29. They now describe workspace membership, the root CI commands, the
      committed root `Cargo.lock`, and rlst as a crates.io dependency.
- [x] Root `CLAUDE.md` asks for `cargo test --workspace`. With MPI crates in the
      workspace, that needs a working MPI runtime and `RUST_MIN_STACK=8388608`, as in CI.
      Stated in root `CLAUDE.md` since Phase 0 T1.
- [ ] Check that the `nd-fmm-*` names are free on crates.io if you plan to publish.
- [x] Hosting: Codeberg's members voted in July 2026 for Terms of Use changes that
      discourage projects "written and maintained with heavy use of LLMs"
      ([Codeberg blog](https://blog.codeberg.org/protecting-our-floss-commons-from-llms.html)).
      The repository moved to [tbetcke/fmm](https://github.com/tbetcke/fmm) on
      2026-09-29, and CI now runs on GitHub Actions.
