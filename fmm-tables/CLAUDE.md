# nd-fmm-tables

Purpose: precomputed, level-independent operator tables, built in f64 from nd-fmm-ref,
and a versioned on-disk cache. The tables are M2M and L2L for the 8 child octants, M2L
for the 316 V-list offsets and their 16 symmetry classes, and the rotation and coaxial
tables of point-and-shoot translation.
Phase and components: Phase 2, C2.1–C2.4 (tasks T1 and T3–T7 in docs/phase2/).

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  - Geometry, table layout and symmetry follow §3.12.
  - The translations are those of §3.11, reached only through nd-fmm-ref.
- Tables are built in f64 by applying nd-fmm-ref operators at the canonical frames of
  §3.12. An f32 table is the f64 table rounded entry by entry. Never build in f32, and
  never re-derive an operator formula here.
- Matrices are column-major with output = A · input. Every application accumulates
  (+=), and none applies 1/(4π) (§3.1).
- Builds are serial and deterministic: the same inputs give bit-identical tables and
  cache files.
- Generic over T: RealScalar. Allocate only in builders and constructors, never in
  apply functions: their temporaries come from caller-owned scratch.
- The octant and offset order are restated here from §3.12, not imported from
  nd-octree or nd-fmm-plan. Phase 3 (C3.1) tests that they agree.
- Tests name their error measure (docs/phase1/README.md, "Error measures", and
  docs/phase2/README.md). A table is compared with nd-fmm-ref on other levels in a
  dyadic domain.
- Keep the debug-mode test run under a minute. Large-p checks go into `#[ignore]` tests.
- Before finishing: `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings` and
  `cargo test -p nd-fmm-tables` must pass. Tasks that add ignored tests must also pass
  `cargo test -p nd-fmm-tables --release -- --ignored`.

## Allowed dependencies
nd-fmm-math, nd-fmm-ref, num-traits; thiserror (from T7, for cache errors);
dev-dependencies: proptest.
Not allowed:
- MPI, nd-octree, nd-fmm-plan or CubeCL;
- rlst, which waits for SVD compression (C6.2). When it comes, it is used without
  its `mpi` feature, so `cargo test -p nd-fmm-tables` still needs no MPI;
- serde or another serialiser;
- rayon;
- nd-fmm-validate, not even as a dev-dependency.

Anything else needs a note in the PR.

## Test oracle
- `nd_fmm_ref::direct` for every table, and `nd_fmm_ref::rotation` for the rotation
  tables, at the canonical frames and on other levels.
- The Phase 1 physical chains (M2M after P2M, L2L exact on local polynomials).
- The symmetry identities of §3.12: the class form against the dense tables.
- A cold build for the cache.
