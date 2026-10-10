# Phase 6 / T1 — design of the optimisations (design for C6.1–C6.8)

Phase 6 is benchmark-driven (README). Before any code, this task measures where the time
goes on the three machines, designs the components that change the host operator (the
batched M2L, SVD compression, several right-hand sides) and the AVX-512 P2P, and orders
the rest by measured gain. It writes `docs/design/optimisation.md` for sign-off by hand
before T2. It is the Phase 6 counterpart of Phase 4 T1 and Phase 5 T1, and writes no Rust
apart from throwaway measurements in the scratch directory, which are not committed.

**Revision note.** Written on 2026-10-10, before Phase 5N and 5S. Phase 5S T9 revises
this brief; until then, every number it cites from those phases is a placeholder.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-tables/CLAUDE.md, fmm-simd/CLAUDE.md,
  fmm-kernels/CLAUDE.md, fmm-validate/CLAUDE.md, fmm-bench/CLAUDE.md;
- docs/phase6/README.md, closely ("Requirements", "Design decisions", "Decisions to sign
  off");
- docs/design/laplace-fmm-plan.md §3 (§3.3 plane waves, §3.4 FFT, §3.6 dense BLAS-3), §4,
  §6.4–§6.8, §7 (the Phase 4, 4S and 5 recommendations, the Phase 6 table), §8.3, §9;
- docs/phase3/README.md, "Threads and BLAS" (the rule for any GEMM inside rayon);
  docs/phase4/README.md, decisions 6 (host batched GEMM deferred) and 9 (FMM3D);
- docs/design/simd-p2p.md §3 (requirement 2: 8 / 16 u_T per pair), §4.3, §4.6, §4.7 (AVX-512, deferred), §5.2 (dispatch), §5.4 (`unsafe`);
- docs/design/device-path.md §6.4–§6.6 (the dense M2L launch structure the host batched
  M2L may mirror), §10 (the spike's model), §18.2 (CUDA layouts; the kernel work left);
- docs/design/node-m2l.md (Phase 5N: the strategy rule per node) and
  docs/design/scale-out.md (Phase 5S), once written;
- the reports: `fmm-validate/results/phase4-m3max.md`, `fmm-bench/results/phase4s-gh200.md`,
  `fmm-validate/results/phase5-m3max.md` and `phase5-gh200.md` (§1, "Stages against ranks"
  and "The M2L strategy at p = 8"), and Phase 5N's and 5S's Kathleen reports;
- the code:
  - `nd_fmm_exec::operator`: `LaplaceOperator`, its `m2l` level call and `m2l_target`
    (the target-by-target body over a V row), the per-thread `Scratch` sets, the
    `Execution` that splits a level with `par_chunks_mut`;
  - `nd_fmm_exec::tables`: `Tables`, `M2lStrategy` (`resolve`, `AUTO_DENSE_MAX_P`),
    `load_or_build`;
  - `nd_fmm_tables`: `MatrixSet::apply` (the hand-written product), `M2lTables`,
    `M2lClasses` (`expand`), `RotationTables`, the `cache` module (`TableKind`,
    `CacheKey`, `TableCache`);
  - `nd_fmm_kernels::translate`: `grouped`, `grouped_stage`, `Gemm`, `GemmLayout`,
    `Accumulate`, `TileSchedule`, `Tables`; `movement::gather_columns`;
  - `nd_fmm_simd`: `Isa` (`#[non_exhaustive]`), the dispatch in `isa.rs` and
    `kernel.rs`, `arch::{avx2, neon, scalar, p2p}`, `rsqrt`, `toolchain`;
  - `nd_fmm_plan::view::VList` and the V batches (`batch_targets`, `batch_sources`,
    `offset_indices`; fmm-plan-redesign §4.3).

Measure before you design where a decision rests on a number. On each machine in use (the
M3 Max; locust's host; one Kathleen node if Kathleen is usable, Phase 5N decision 0; no
Kathleen job above 4 nodes), release, the cube and the Plummer sphere at N = 10⁶,
f64 p = 3, 6, 8, 12 and f32 p = 3, 8, one rank and the machine's useful ranks per node
(12, 72, and 40 and 2 × 20 on Kathleen):
- the evaluation by stage and by operator kind (`StageTimings`, `KindTiming::Synchronous`);
- M2L under each strategy, and its achieved GFLOP/s against the core's peak (state the
  peak's source: datasheet or measured);
- the dense tables' size against the caches (L2, L3 per socket);
- a throwaway host GEMM of the V level's shape (columns of one offset gathered, one
  table): rlst's BLAS (Accelerate, OpenBLAS, MKL on Kathleen) single-threaded against a
  hand-written loop, at the column counts the levels give; and whether its bits change
  with the BLAS build or the thread count;
- P2P on Kathleen (AVX2 path, `p2p_kernels`, `p2p_fmm`) at FMM leaf sizes, as the AVX-512
  baseline; without Kathleen, say that C6.7 waits and skip this.

The design states, for every component, what it does on the path without Kathleen
(README, "Kathleen may turn out unusable").

Write `docs/design/optimisation.md` with these sections:

1. **Baseline.** Where an evaluation spends its time on each machine and ranks per node,
   from the measurements above, with each number's source.
2. **The host batched M2L (C6.6).** The layout: per level, offset by offset (or in
   chunks), gather the source multipoles of the offset's pairs into columns, one product
   with the offset's table, add the products into each target's local in row order. Show
   that the order of additions into each local is the target-by-target path's (offsets in
   index order), or state the new order and its bound. Scratch per thread as a formula,
   chunked like the device's scratch budget. Threading: which loop rayon splits, so that
   each target is written by one thread and the bits do not depend on the thread count.
   BLAS or a hand-written kernel (decision 2), with the measured comparison. Whether M2M
   and L2L follow. Expected gain as a *model* from the measured GEMM throughput, at one
   rank and at many ranks per node (Phase 5's table-pressure finding).
3. **SVD-compressed M2L (C6.2).** Per offset (or per symmetry class, `M2lClasses`) a
   truncated factorisation U Σ Vᵀ; the truncation rule; the error it adds against the
   p-truncation error (laplace-fmm-plan §2 and Phase 3's calibration); memory and flops
   per pair against dense and rotation per p; build cost (an SVD per table at build,
   cached through `TableCache` with new `TableKind`s); the device form. Propose the error
   budget (decision 3).
4. **Several right-hand sides (C6.3).** The interface (`Fmm::evaluate_many` or a charge
   matrix), the layouts through the stores, the exchanges and the redistribution, and
   what changes per operator; whether each vector's output stays bit for bit its
   single-vector evaluation.
5. **AVX-512 P2P (C6.7).** simd-p2p.md §4.7's plan on stable Rust: check which Rust release
   stabilised the AVX-512 target features and intrinsics in `core::arch` and that the
   toolchain of the root CLAUDE.md and CI includes it; `pulp`'s V4 still needs its
   `nightly` feature, so the path is hand-written in `arch::avx512`. The estimate and its
   Newton steps (`_mm512_rsqrt14_ps`/`_pd`), masks for tails and r² = 0, the 8 / 16 u_T
   per-pair contract, the dispatch rule (frequency licences on Cascade Lake: dispatch only where
   it measures faster inside the FMM?), and how CI covers it.
6. **The device items (C6.1).** The fused gather, and device-path §18.2's left-over work
   (a shared-memory-tiled GEMM, the per-chunk row walk of `Accumulate::Rows`, L2P at high
   p): which are in T7, with the expected gain from the Phase 4S measurements.
7. **Dipoles (C6.4).** The source-type interface (charges, dipoles, both), the operators
   that change (P2M, P2L, P2P, the direct sum), CONVENTIONS sections they need (a
   convention addition is proposed in the PR, never made in code).
8. **FMM3D (C6.8) and plane waves (C6.5).** The comparison's method (matched accuracy,
   the problems, the build); the plane-wave spike's question and stop rule.
9. **Order and gain.** The tasks ordered by measured or modelled gain per machine, with
   any brief that should change (propose, do not rewrite).
10. **Questions for sign-off.** README decisions 1–6, each with a recommendation.

Keep the document in the style of the other design documents: short sections, tables,
Rust signatures where they decide something, every model marked *model* and every
measurement with its machine, ranks × threads and command.

The PR description must contain a one-page summary, the sign-off questions with
recommendations, and the baseline table.

Must pass: nothing builds differently. Run `cargo fmt -- --check` and
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` once to confirm the starting point.
Kathleen runs go in Slurm jobs (tools/kathleen/README.md); locust runs with the load
checked and stated.

Do not:
- write or change Rust in the workspace, or a brief except to fix a factual error (say
  which);
- commit measurement code or output;
- propose a new external dependency without asking, or an FMM3D link into any crate.
