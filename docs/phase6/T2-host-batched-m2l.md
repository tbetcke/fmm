# Phase 6 / T2 — the host batched M2L (C6.6)

The host path applies M2L target by target: for each target box, `m2l_target` walks its V
row and multiplies each source multipole by the offset's dense table through
`MatrixSet::apply`. Every rank streams all 316 tables (16.6 MB at p = 8 in f64) once per
target. On one node that is what stops strong scaling at p = 8: the downward pass's total
over the ranks grows 2.4–4.4× by 8–12 ranks on the M3 Max and 5.8–6.9× by 72 on locust,
while Rotation, Classes and threads per rank, which hold or share small tables, scale
(`fmm-validate/results/phase5-m3max.md` §1, "Stages against ranks" and "The M2L strategy
at p = 8"; `phase5-gh200.md` §1, the same; distributed-fmm.md §15.2). Phase 5N chose the
strategy per node to avoid the worst of it; this task removes the cause.

The device already does it differently (device-path.md §6.4, structure (B)): per level and
per chunk of offsets it gathers the offset's source multipoles into columns, multiplies
them by the offset's table in one GEMM, and adds the products into each target's local in
row order. Each table is read once per level, and the product is BLAS-3. This task builds
the host counterpart, as signed off in `docs/design/optimisation.md` §2 (T1), and
measures it where Phase 5 found the problem: many ranks per node. It was deferred in
Phase 4 (docs/phase4/README.md, decision 6).

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it. T1's design
overrides any detail here that it settles differently.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-tables/CLAUDE.md;
- docs/phase6/README.md ("Requirements", "Design decisions": BLAS inside rayon);
- docs/design/optimisation.md §1, §2 and the signed-off answers to decision 2;
- docs/phase3/README.md, "Threads and BLAS"; docs/design/laplace-fmm-plan.md §3.6, §6.8;
- docs/design/device-path.md §6.4 (the dense M2L launch structure and its order);
- docs/design/fmm-plan-redesign.md §4.3 (the V grouping: `batch_targets`,
  `batch_sources`, `offset_indices`) and §7.5 (the accumulation order);
- docs/design/node-m2l.md (Phase 5N: the strategy rule per node);
- the code: `nd_fmm_exec::operator` (`LaplaceOperator::m2l`, `m2l_target`, `Execution`,
  the per-thread `Scratch`), `nd_fmm_exec::tables` (`Tables`, `M2lStrategy`),
  `nd_fmm_tables::{MatrixSet, M2lTables}`, and for comparison `nd_fmm_kernels::translate`
  (`grouped`, `Accumulate`, `TileSchedule`).

Do:
- **The batched M2L level call** on the host, as T1 designed it: a new way to apply the
  dense tables, chosen by a strategy value (for example `M2lStrategy::Batched`, or a mode
  of `Dense`; T1 names it). Per level:
  - offsets in index order, in chunks whose gathered columns and products fit a scratch
    budget per thread (T1's formula; default stated and measured);
  - for each offset of a chunk, the source multipoles of its pairs gathered into a
    column block, one product with the offset's table (BLAS through rlst, single-threaded
    inside rayon, or the hand-written kernel; decision 2), the products added into each
    target's local in the order the target-by-target path adds them;
  - rayon splits the work so that every target's local is written by one thread at a
    time and meets its offsets in index order, so the output is bit-identical for every
    thread count (requirement 2). If T1's design changes the order of the additions,
    implement its documented order and test its bound instead of bit identity against the
    target-by-target path; bit identity across thread counts still holds.
  - nothing allocated per pair or per call beyond the scratch sets, as today.
- **M2M and L2L** in the same form if T1 found it pays (decision 2); otherwise unchanged.
- **Classes** through the batched form if T1 recommends it (the class matrices with the
  coefficient transforms around the product), else unchanged.
- **The strategy rule**: `Auto` unchanged in this task. Report how the batched M2L
  compares to every strategy so that T11 (and Phase 5N's rule) can be revisited.
- **Accounting**: the per-kind timings (`KindTiming`) and the scaling harness's stage
  times cover the new path; `Fmm::strategy()` (or the build report) names it.
- **Measurements** (release, BLAS variables 1, the machine named), against the
  target-by-target `Dense`, `Classes` and `Rotation`:
  - one rank, threads 1 and the machine's cores, the cube and the Plummer sphere at
    N = 10⁶, f64 p = 3, 6, 8, 12, 18 and f32 p = 3, 6, 8, on the M3 Max, locust and one
    Kathleen node (if Kathleen is usable: Phase 5N decision 0);
  - **many ranks per node** with the `scaling` harness (`tools/scaling/run.sh`, its
    `strategy` sweep extended to the new value): the M3 Max at 1–12 ranks, locust at
    1–72, and if usable Kathleen at 1–40 ranks per node and 2 × 20 (one node; at most 4
    nodes in any job), N = 10⁶, f64 p = 8 (and p = 12); without Kathleen the M3 Max and
    locust carry the criterion;
  - the downward pass's total over the ranks against one rank, as in Phase 5 T10's
    "Stages against ranks" table: the criterion is that it no longer grows with the
    ranks per node the way Dense does;
  - achieved GFLOP/s of the products, and the scratch per thread.
  Report in the PR and as a section of `fmm-validate/results/` (T1 names the file, for
  example `phase6-m2l.md`).

Tests that define done:
- The batched M2L against the target-by-target `Dense` on every `tests/mpi_exec.rs`
  scenario at 1, 2 and 4 ranks: bit for bit if the order is kept, otherwise within T1's
  bound (and requirement 1 against the direct sum).
- Bit-identical across 1, 2, 4 and 8 threads, and across two evaluations and two builds.
- The multi-rank guarantees of Phase 5 (README requirement 3): the C5.1 host gate
  (`tests/multi_rank.rs`) with the new strategy at 2, 4 and 8 ranks.
- f32 and f64 at p = 3, 8 and 18 against the direct sum within the Phase 3 bounds.
- If BLAS is used: a test that runs the same level call with BLAS and with the
  hand-written fallback (or two BLAS thread settings in a subprocess) and compares bits,
  or states the bound.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec`, and `.github/scripts/run-mpi-tests.sh`
  at 2 and 4 ranks;
- the ignored release gates of `nd-fmm-exec` at 1 rank on the M3 Max, and the C5.1 host
  gate at 2, 4 and 8 ranks on the M3 Max and locust, and on 2 Kathleen nodes if Kathleen
  is usable.

Do not:
- change the default strategy or any default output (README requirement 6; T11 decides);
- call a multi-threaded BLAS inside a rayon worker, or set thread variables from the
  library;
- add a dependency (BLAS goes through rlst or a hand-written kernel);
- assert a timing.
