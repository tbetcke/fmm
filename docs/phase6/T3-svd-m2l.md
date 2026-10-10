# Phase 6 / T3 — SVD-compressed M2L on the host (C6.2, host)

Dense M2L tables grow as (p + 1)⁴: 16.6 MB for the 316 offsets at p = 8 in f64, about
400 MB at p = 19 (laplace-fmm-plan §9.1, "Dense M2L memory at high p"). Each 316-table
set has low numerical rank for the far offsets: a truncated SVD per offset, U Σ Vᵀ with k
≪ (p + 1)² columns, applies an M2L as two thin products and cuts both the memory and the
flops, at an error that can be held below the p-truncation error. This task builds the
compressed tables in `nd-fmm-tables` and their application on the host, as signed off in
`docs/design/optimisation.md` §3 (T1), on top of T2's batched layout.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md, fmm-tables/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase6/README.md (requirement 1, decision 3);
- docs/design/optimisation.md §3 (the factorisation, the truncation rule, the error
  budget) and §2 (the batched layout);
- docs/design/laplace-fmm-plan.md §2.3–§2.5 (translations, level independence), §3.6,
  §4, §9.1; the Phase 3 calibration of p (Section 7, Phase 3);
- docs/phase3/README.md "Threads and BLAS" (an SVD at table build may use BLAS threads,
  outside rayon);
- the code: `nd_fmm_tables::{m2l, symmetry, matrix_set, cache}` (`M2lTables`,
  `M2lClasses`, `MatrixSet`, `TableKind`, `CacheKey`, `TableCache`), T2's batched level
  call in `nd_fmm_exec::operator`, `nd_fmm_exec::tables::{Tables, M2lStrategy}`.

Do:
- **The compressed tables** in `nd-fmm-tables`: per offset (or per symmetry class, as T1
  decides) the factors of a truncated SVD of the dense table, computed in f64 at build
  through rlst's LAPACK (outside any rayon worker), truncated by the signed-off rule, and
  stored for f32 and f64. A new `TableKind` (and cache key fields: p, precision, the
  truncation parameter) so `TableCache` stores and loads them; the cache format version
  bumped if the header changes. Level independence holds as for the dense tables
  (CONVENTIONS §3.x, cite the section).
- **The host application**: a strategy value (T1 names it, for example
  `M2lStrategy::Compressed`), applied through T2's batched layout as two products per
  offset chunk (Vᵀ then U Σ, or as T1 orders them), with each target's local meeting its
  offsets in index order. Bit-identical for every thread count.
- **The error budget**: the added error measured against the dense path and against the
  direct sum at every p of the workloads, per truncation setting; the default setting
  proposed for T11 (it stays opt-in here).
- **Measurements** (release; M3 Max, locust, and one Kathleen node if Kathleen is usable): memory of the tables,
  build time (SVD) and cache load time, the M2L time and the evaluation time against
  dense (target by target and batched) and rotation, f64 p = 6, 8, 12, 18 and f32 p = 6,
  8, one rank and many ranks per node (the `scaling` harness's strategy sweep); the
  error against the direct sum. Report in the Phase 6 results file.

Tests that define done:
- Each compressed table reproduces its dense table within the truncation bound
  (Frobenius or operator norm, as T1 states) for every offset at p = 3, 8, 12.
- The compressed FMM against the dense FMM on every `tests/mpi_exec.rs` scenario at 1, 2
  and 4 ranks within the signed-off budget, and against the direct sum within the Phase 3
  bounds plus the budget.
- Bit-identical across thread counts, evaluations and builds; the cache round-trips the
  tables bit for bit.
- The C5.1 host gate passes with the new strategy at 2 and 4 ranks.

Must pass: `cargo fmt --all`, the root checks and the stricter workspace checks;
`cargo test -p nd-fmm-tables`; `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec`;
`.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks.

Do not:
- make the compressed path a default (T11 decides, with decision 3);
- run the SVD inside a rayon worker with BLAS threads, or add a dependency;
- change the dense tables or their cache entries.
