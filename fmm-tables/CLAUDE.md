# nd-fmm-tables

Purpose: precomputed, level-independent operator tables, built in f64 from nd-fmm-ref,
and a versioned on-disk cache. The tables are M2M and L2L for the 8 child octants, M2L
for the 316 V-list offsets and their 16 symmetry classes, and the rotation and coaxial
tables of point-and-shoot translation.
Phase and components: Phase 2, C2.1–C2.4 (tasks T1 and T3–T7 in docs/phase2/); Phase 6
C6.5 (task T13: `plane_wave`, the tables of plane-wave M2L, CONVENTIONS §3.14);
Helmholtz CH3.1a (`helmholtz`, the per-level RCR tables of the Helmholtz family,
CONVENTIONS §3.15).

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  - Geometry, table layout and symmetry follow §3.12.
  - The translations are those of §3.11, reached only through nd-fmm-ref.
- Tables are built in f64 by applying nd-fmm-ref operators at the canonical frames of
  §3.12. An f32 table is the f64 table rounded entry by entry. Never build in f32, and
  never re-derive an operator formula here.
- The exception, `plane_wave` (Phase 6 T13): the plane-wave M2L has no operator in
  nd-fmm-ref, so its tables follow the formulas of CONVENTIONS §3.14 (built in f64, rounded
  to T) and are checked against `nd_fmm_ref::direct::m2l` (levels 2, 9 and 16, within the
  quadrature's bound; the ignored `pair_m2l_every_degree` at every p ≤ 20). FMM3D's
  quadratures (`quadrature_data.rs`, NIST's notice kept) are regenerated only by
  `tools/fixtures/plane_wave_quadrature.sh`. The degree's quadrature
  (`nodes_for_degree`) is the study of the ignored `level_every_degree` (the spike's
  matched rule on level 3); a change to it changes the FMM's errors, so rerun the gates of
  nd-fmm-exec. The functions that a `nd-fmm-simd` `WaveKernel` loop restates
  (`waves_from_modes`, `sums_from_waves`, `shift_add`, `shift`, `add`) define its chains
  with `mul_add`; keep the two in step. The other steps round products and sums apart: an
  x86_64 build without `fma` turns `mul_add` into a library call. The plane-wave tables
  are not cached (milliseconds to build).
- The second exception, `helmholtz` (Helmholtz CH3.1a; its module docs): the coaxial
  tables come from `nd_fmm_ref::helmholtz::rcr::{m2m_table, l2l_table, m2l_table}`, but
  the stored form's factors folded into them and the three steps of the application
  restate `rcr`'s private `translate` in its order of operations, so that every
  application in f64 is `rcr`'s bit for bit at the same frames (`tests/helmholtz/`), but
  for the real shifts of real k, which keep the tables' real parts (the FMM-weighted
  error, within 1e-12). Keep the two in step. Its tables are keyed by the bits of the
  shifts' |t| and polar angles as `rcr` forms them (21 distances and 53 polar angles over
  the 316 offsets on the M6), not by exact classes. Its y-blocks are built in T, as `rcr`
  builds them per call, so that stored and per-call blocks agree; the M2L blocks are
  stored from the finest level up within a byte budget. Not cached (CH2.2).
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
