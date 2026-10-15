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

**Revision note.** Written on 2026-10-10; **revised by Phase 5S T9 on 2026-10-15** (decision
7 of docs/phase5s/README.md): the premise holds after Phase 5N and 5S. Phase 5N's rule picks
Rotation at f64 p = 8 (and at f32 p = 8 above a few ranks per node), which avoids the
dense tables' limit but not their cause; at 160 Kathleen ranks the downward pass (mostly M2L)
is 67–92 % of an evaluation at p = 8 (`fmm-validate/results/phase5s-kathleen.md`), so this
task is the largest gain left at high p. Kathleen is usable (Phase 5N decision 0): its runs
are not conditional, and 40 × 1 per node is the configuration to beat at f64 p = 8 (2–10 %
faster than 2 × 20 with Phase 5S's cut by work). T1's design overrides any detail here
that it settles differently.

**Revised after T1's sign-off (2026-10-15; docs/design/optimisation.md §2, §10).** Decided:
no host BLAS; the strategy `M2lStrategy::Batched`; the order-preserving kernel (offsets in
index order, products accumulated into each target's own local, no FMA), which is bit for
bit the target-by-target `Dense`, in `nd-fmm-simd` behind its run-time ISA dispatch; an
**opt-in FMA mode** of the same kernel; M2M, L2L and Classes unchanged. T1 measured the
kernel at 6.7–15× today's Dense per pair with one rank per core and faster than Rotation
at f64 p = 8 on every machine (optimisation.md §2.6), and modelled the evaluation at f64
p = 8 at 1.13× (locust) to 2.07× (Kathleen) today's best strategy (§2.10).

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-tables/CLAUDE.md, fmm-simd/CLAUDE.md;
- docs/phase6/README.md ("Requirements", "Design decisions": BLAS inside rayon);
- docs/design/optimisation.md §1, §2 (the design: §2.2 layout, §2.3 order, §2.4 threads,
  §2.5 scratch, §2.11 the kernel's place) and §10 (the sign-off);
- docs/design/simd-p2p.md §5.2–§5.4 (detection, dispatch, `unsafe`);
- docs/phase3/README.md, "Threads and BLAS"; docs/design/laplace-fmm-plan.md §3.6, §6.8;
- docs/design/device-path.md §6.4 (the dense M2L launch structure and its order);
- docs/design/fmm-plan-redesign.md §4.3 (the V grouping: `batch_targets`,
  `batch_sources`, `offset_indices`) and §7.5 (the accumulation order);
- docs/design/node-m2l.md (Phase 5N: the strategy rule per node);
- the code: `nd_fmm_exec::operator` (`LaplaceOperator::m2l`, `m2l_target`, `Execution`,
  the per-thread `Scratch`), `nd_fmm_exec::tables` (`Tables`, `M2lStrategy`),
  `nd_fmm_tables::{MatrixSet, M2lTables}` (`MatrixSet::apply`: the chain to keep),
  `nd_fmm_simd::{isa, kernel, arch}` (the dispatch the kernel joins), and for comparison
  `nd_fmm_kernels::translate` (`grouped`, `Accumulate`, `TileSchedule`).

Do:
- **The kernel** in `nd-fmm-simd` (optimisation.md §2.11): a safe public function that adds
  A · B[:, b_j] into C[:, c_j] for lists of column offsets, every element as
  `MatrixSet::apply` (k ascending, the product and the sum rounded apart), the NB source
  columns packed k-major, an MR × NB block of the targets in registers across k; bodies in
  `arch::{scalar, neon, avx2}` (the AVX-512 body comes with T6), `unsafe` only in `arch` and
  the dispatch, every block with `// SAFETY:`; and the **FMA mode** (`fma(a, b, acc)`, k
  ascending; `mul_add` on the scalar body, so every ISA gives the same bits).
- **The batched M2L level call** in `nd-fmm-exec`, `M2lStrategy::Batched` (the dense tables
  of `Dense`; on the device it runs as `Dense` and is reported so), per level:
  - rayon over blocks of B consecutive targets (fixed, not from the thread count; 64
    proposed, measured), `par_chunks_mut` on the level's locals;
  - per block, the offsets in index order, the block's pairs of each offset found in
    `batch(d)` (binary searches, or ranges per (block, offset) precomputed when the level is
    first seen), the kernel reading the multipoles and the locals in place (no gathered
    copy of the level);
  - nothing allocated per pair or per call: the packing buffer and the index lists are
    per-thread scratch made at construction (optimisation.md §2.5);
  - the ISA from the `Fmm`'s P2P choice (`P2pChoice`), the mode (no FMA by default, FMA
    opt-in) from the strategy or a builder setting that T2 names and documents.
- **M2M, L2L and Classes** unchanged (≤ 1.2 % of an evaluation; optimisation.md §2.7, §2.8).
- **The strategy rule**: `Auto` unchanged in this task (Phase 5N T4's rule:
  `M2lStrategy::resolve_on_node`, Dense while the dense tables of a rank and of a node are
  small, Rotation otherwise; docs/design/node-m2l.md). Report how the batched M2L compares
  to every strategy so that T11 (and Phase 5N's rule) can be revisited.
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
  - achieved GFLOP/s of the products, the scratch per thread, the block size B, and the
    FMA mode beside the default at the same cells;
  - Kathleen with the kernel's AVX2 body (T1's native-build figures are the model to
    check).
  Report in the PR and in `fmm-validate/results/phase6-m2l.md`.

Tests that define done:
- The batched M2L against the target-by-target `Dense` on every `tests/mpi_exec.rs`
  scenario at 1, 2 and 4 ranks, bit for bit; the FMA mode within its bound (relative
  ≲ n u per element, optimisation.md §2.3; requirement 1's 1e-12 / 1e-5 against `Dense`)
  and bit-identical across thread counts and rank counts.
- The kernel of every available ISA against its scalar body bit for bit, in both modes,
  at p = 0–20 and column counts 1–64 (`nd-fmm-simd` tests; the AVX2 body on CI's x86_64
  runner, NEON on the M3 Max and the arm64 runner).
- Bit-identical across 1, 2, 4 and 8 threads, and across two evaluations and two builds.
- The multi-rank guarantees of Phase 5 (README requirement 3): the C5.1 host gate
  (`tests/multi_rank.rs`) with the new strategy at 2, 4 and 8 ranks.
- f32 and f64 at p = 3, 8 and 18 against the direct sum within the Phase 3 bounds.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- the `nd-fmm-simd` checks of the root CLAUDE.md, including the other architecture's
  clippy (`--target x86_64-apple-darwin` on the M3 Max);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec`, and `.github/scripts/run-mpi-tests.sh`
  at 2 and 4 ranks;
- the ignored release gates of `nd-fmm-exec` at 1 rank on the M3 Max, and the C5.1 host
  gate at 2, 4 and 8 ranks on the M3 Max and locust, and on 2 Kathleen nodes if Kathleen
  is usable.

Do not:
- change the default strategy or any default output (README requirement 6; T11 decides);
- call BLAS on the host, or set thread variables from the library (decision 2);
- add a dependency, or `unsafe` outside `nd-fmm-simd`'s `arch` and dispatch;
- assert a timing.
