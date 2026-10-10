# Phase 6: optimisation and extensions

Phase 6 makes the operators faster and adds what the plan deferred. It is benchmark-
driven: every component is measured on the machines available, adopted when it wins, and
kept behind its slower trusted path. By the time it starts the workspace has:
- **the host path** (`nd-fmm-exec`'s `LaplaceOperator`): M2L by `Dense`, `Classes` or
  `Rotation` (`nd_fmm_exec::tables::M2lStrategy`), applied target by target by the
  hand-written loops of `nd-fmm-tables` (`MatrixSet::apply`), threaded with rayon and
  bit-identical for every thread count (C3.5); P2P through `nd-fmm-simd` (NEON, AVX2 +
  FMA, scalar; Phase 3S); no BLAS call in any compute path (Phase 3, "Threads and
  BLAS");
- **the device path** (`nd-fmm-kernels`, CubeCL 0.11.0-pre.4): every operator on Metal
  f32, CUDA f32/f64 and the CubeCL CPU runtime, dense M2L as gathered GEMMs per level
  (device-path.md §6.4), rotation M2L, a tuner (Phase 4, Phase 4S);
- **the distributed FMM** on any number of ranks (Phase 5), with the M2L strategy chosen
  per node (Phase 5N) and the top tree no longer replicated (Phase 5S).

What is missing, collected from the earlier phases:
- **The host M2L reads its tables once per target.** At p = 8 every rank holds its own
  316 dense 81 × 81 tables (16.6 MB in f64) and streams them target by target. On one
  node this stops strong scaling: the downward pass stops scaling beyond 4 ranks on the
  M3 Max and 32 on locust, while Rotation, Classes and threads per rank scale
  (`fmm-validate/results/phase5-m3max.md` §1, "The M2L strategy at p = 8";
  `phase5-gh200.md` §1, "The M2L strategy at p = 8"; laplace-fmm-plan §7, Phase 5,
  "Recommendation for Phase 6"). A host M2L as per-level GEMMs over gathered columns
  reads each table once per level. It was deferred in Phase 4 (docs/phase4/README.md,
  decision 6: "A host batched-GEMM path through BLAS ... deferred, a candidate for Phase
  6"). Phase 5N fixes the default configuration; this phase fixes the operator (C6.6).
- **Large tables at high p** (laplace-fmm-plan §9.1, "Dense M2L memory at high p",
  about 400 MB at p = 19): SVD compression (C6.2).
- **One charge vector per evaluation.** Several vectors as extra GEMM columns (C6.3).
- **AVX-512.** `nd-fmm-simd` runs the AVX2 code on AVX-512 machines (simd-p2p.md §4.7,
  "AVX-512, deferred": deferred until hardware to test and time it was available).
  Kathleen's Xeon Gold 6248 nodes have AVX-512F (C6.7).
- **The gather before each device GEMM** is a separate pass (C6.1), and the device
  kernel work left by Phase 4S (device-path.md §18.2; laplace-fmm-plan §7, Phase 4S,
  "Recommendation for Phase 5": a shared-memory-tiled GEMM, the per-chunk row walk of
  `Accumulate::Rows`, L2P at high p) is folded into C6.1 (T7).
- **Dipole sources** and the source-type interface (C6.4).
- **Plane-wave M2L**, optional (C6.5).
- **An external baseline.** FMM3D only, in a later phase (docs/phase4/README.md,
  decision 9) (C6.8).
- **Defaults that are still open**: the leaf size per p (laplace-fmm-plan §9.2, "the best
  leaf size depends on p ... still open for every backend"), and the strategy rule once
  C6.6 and C6.2 exist (T11).

**These briefs were written on 2026-10-10, before Phase 5N and 5S ran.** Phase 5S T9
revises them with its measurements, as Phase 4S T8 revised the Phase 5 briefs. Where a
brief names a number from 5N or 5S, it is a placeholder for that revision.

Hardware: three machines (docs/phase5n/README.md, "Machines"):
- the Apple M3 Max (12 performance and 4 efficiency cores, 64 GB, Metal f32);
- locust (GH200: 72 Neoverse-V2 cores in one NUMA node, one H100 with CUDA f32 and f64;
  shared, no scheduler; tools/gh200/);
- Kathleen (UCL cluster, Slurm; 190 nodes of 2 × Intel Xeon Gold 6248, 40 cores per
  node, two NUMA domains, AVX-512, 188 GB, Omni-Path; no GPUs; tools/kathleen/ from
  Phase 5N T1), **if it is usable** (below). **No Kathleen job is ever larger than 4
  nodes (160 cores)**, in this phase as in 5N and 5S.

**Kathleen may turn out unusable.** Phase 5N T1 probes its queue waits, and the user
decides whether it is used ("decision 0" of docs/phase5n/README.md). Phase 6 has a path
without it:
- **with Kathleen**: everything below, including the x86 measurements (C6.7's AVX-512
  P2P, and the Kathleen parts of T1, T2, T3, T5, T9 and T11), on at most 4 nodes per job;
- **without Kathleen**: C6.7 (T6) waits until an AVX-512 machine is available (as it
  waited in Phase 3S); T1, T2, T3, T5, T9 and T11 run on the M3 Max and locust only,
  "many ranks per node" means 1–12 ranks on the M3 Max and 1–72 on locust, and no
  multi-node figure is produced (the multi-rank checks run on locust at up to 72 ranks).
  Every report states which path it took. Nothing else in the phase depends on Kathleen.

The phase has four parts:
1. **Design** (T1): the baseline per machine, the design of the host batched M2L, SVD
   compression, several right-hand sides and AVX-512 P2P, and the order of the rest by
   measured gain.
2. **The host M2L** (T2–T5): the batched host M2L (C6.6), SVD-compressed M2L on the
   host and the device (C6.2), several right-hand sides (C6.3).
3. **Kernels and extensions** (T6–T10): AVX-512 P2P (C6.7), the fused gather and the
   device GEMM work (C6.1), dipoles (C6.4), the FMM3D comparison (C6.8), the plane-wave
   spike (C6.5, optional).
4. **Benchmarks and defaults** (T11): the final runs on the machines in use (Kathleen at
   up to 4 nodes, if usable), the defaults revisited, the design-document update.

Companion documents:
- [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md): §3 (translation
  families: §3.3 plane waves, §3.4 FFT, §3.6 dense BLAS-3), §4 (the strategy
  comparison), §6 (CubeCL, with §6.4–§6.7 and §6.8 threads and BLAS), §7 (the Phase 4,
  Phase 4S and Phase 5 recommendations, the Phase 6 table), §8.3 and §9;
- [docs/design/simd-p2p.md](../design/simd-p2p.md): §4.3 (inverse square root), §4.6
  (the model), §4.7 (AVX-512, deferred), §5 (the crate: §5.2 detection and dispatch, §5.4 `unsafe`);
- [docs/design/device-path.md](../design/device-path.md): §6 (the kernels), §10 (the
  spike's model), §13, §17 and §18 (CUDA layouts, the host part);
- [docs/design/distributed-fmm.md](../design/distributed-fmm.md) §15 (measured
  numbers), and the designs of Phase 5N and 5S once written:
  `docs/design/node-m2l.md` (the M2L strategy per node) and `docs/design/scale-out.md`;
- the reports `fmm-validate/results/phase4-m3max.md`,
  `fmm-bench/results/phase4s-gh200.md`, `fmm-validate/results/phase5-m3max.md`,
  `fmm-validate/results/phase5-gh200.md`, and Phase 5N's and 5S's Kathleen reports.

T1 adds `docs/design/optimisation.md`.

Prerequisite: Phase 5S is complete (T9 merged, these briefs revised by it). If Phase 5S is
deferred, Phase 6 may start after Phase 5N; T1 then says which briefs assume 5S.

## Scope

In scope:
- `docs/design/optimisation.md`, signed off; the design-document updates at the end.
- `nd-fmm-exec`: the host batched M2L (and M2M, L2L where it pays) as a fourth way to
  apply the tables (T2); compressed M2L on the host and the device (T3, T4); several
  charge vectors per evaluation (T5); dipole sources (T8); the strategy rule and the
  defaults revisited (T11).
- `nd-fmm-tables`: SVD-compressed M2L tables and their cache entries (T3); dipole
  leaf operators' tables if T1's design needs them (T8).
- `nd-fmm-simd`: an `arch::avx512` module, `Isa::Avx512`, its detection and tests (T6).
- `nd-fmm-kernels`: the fused gather (T7), the device kernel work of device-path §18.2
  (T7), compressed M2L (T4), several right-hand sides (T5), dipole leaf kernels (T8).
- `nd-fmm-ref`, `nd-fmm-math`: the dipole reference operators and fixtures (T8), under
  CONVENTIONS (a convention change is proposed in the PR, never made in code).
- `nd-fmm-validate`, `nd-fmm-bench`: the benchmarks and reports.
- A spike for plane-wave M2L in `spikes/` (T10).

Out of scope:
- Kernels other than Laplace (Helmholtz, Yukawa), periodic boundaries, p > 20.
- FP64 tensor cores, and f16, bf16 or tf32 arithmetic (Phase 4's rule stands).
- Changes to the distributed structure of Phase 5 and 5S (partition, top tree,
  exchanges), beyond what a new operator needs (for example the extra columns of C6.3
  through the exchanges).
- FMM3D as a dependency of any crate: it is an external program run beside ours (T9).
- New GPUs or a multi-GPU node: none is available. The device index from the local rank
  (distributed-fmm.md §7.4, §14.4 S6) stays a later question.

## Requirements

T1 designs against these, and T2–T11 are accepted against them. T1 may propose a change,
with reasons, for sign-off.

1. **Accuracy unchanged.** Every new path meets the bounds of the path it replaces:
   - against the existing host path of the same p: relative L2 within 1e-12 (f64) and
     1e-5 (f32), Phase 4's device-against-host bounds, unless T1 derives a tighter one;
   - against the direct sum: the errors within 0.1% (f64) and 1% (f32) of the existing
     path's at the C3.2 cube and the C3.3 Plummer sphere (Phase 5, "Accuracy
     measures");
   - a compressed or approximate operator (C6.2, C6.5) states its own error budget,
     signed off with T1, and is opt-in until T11 decides a default.
2. **Deterministic.** For fixed input and settings the output is bit-identical from
   evaluation to evaluation, from run to run, for every thread count (C3.5) and every
   overlap setting (Phase 5). A BLAS call inside the compute path is allowed only where
   its result does not depend on the thread count: single-threaded BLAS inside rayon
   workers (Phase 3, "Threads and BLAS"), with a fixed blocking, or a hand-written
   kernel.
3. **Multi-rank guarantees kept.** The host output on P ranks stays bit for bit the
   one-rank output over the union in rank order, and within 100 u_T between input
   distributions (Phase 5, decision 7). A new operator that changes the one-rank bits
   changes the multi-rank bits the same way, and the multi-rank tests keep passing.
4. **A slower trusted path for every fast path.** The existing target-by-target host
   operator, `nd-fmm-ref`'s direct operators and the direct sum stay the references; a
   new path is tested against them on every scenario of `tests/mpi_exec.rs` it touches.
5. **Measured, never asserted.** Timings in release builds, every BLAS thread variable at
   1 unless a run states otherwise, the machine and ranks × threads named; on Kathleen
   also the nodes, the binding and the job id. No timing is asserted in a test or taken in
   CI.
6. **Defaults change only by sign-off.** A new path is opt-in until T11 proposes it as a
   default with measurements on all three machines; the one-rank outputs then change,
   and the change is recorded.

## Design decisions for this phase

- **BLAS inside rayon.** The Phase 3 rule stands for every host GEMM (docs/phase3/README.md,
  "Threads and BLAS"): a matrix product called inside a rayon worker runs
  single-threaded; ranks × rayon threads × BLAS threads stay within the physical cores;
  large products outside rayon (the SVD of C6.2 at table build) may use BLAS threads; the
  library never sets thread environment variables. rlst 0.9.0 already links the system
  BLAS and LAPACK (CI installs `libopenblas-dev`; Accelerate on macOS through rlst's
  defaults, OpenBLAS from spack on locust, the `openblas/0.3.28` module or MKL on
  Kathleen). A hand-written host GEMM is the alternative T1 weighs; whichever runs, the
  bits must not depend on the thread count.
- **No new dependency without asking.** BLAS goes through rlst or a hand-written kernel.
  FMM3D is built and run outside the workspace (a script in `tools/`), never linked.
- **Every operator behind the `FmmOperator` interface.** No change to `nd-fmm-plan`'s
  interface unless T1 shows the batched views cannot carry it, and then as a sign-off
  question (requirement 5 of Phase 5: kernel-agnostic plan).
- **`unsafe`** only where the root CLAUDE.md allows it: AVX-512 intrinsics in
  `nd-fmm-simd`'s `arch` modules and its dispatch, kernels in `nd-fmm-kernels`, spikes.
- **Workloads.** The C3.2 cube and the C3.3 Plummer sphere at N = 10⁵ and 10⁶ (and 10⁷
  on locust and Kathleen), f64 p = 3, 6, 8, 12, 18 and f32 p = 3, 6, 8; eight charge
  vectors for accuracy; the Gaussian clusters where load or adaptivity matters; the
  `scaling` harness of Phase 5 T10 for multi-rank runs.
- **Machines and runs.** As in Phase 5 and 5N: the macOS loopback flags and at most 12
  ranks × threads on the M3 Max; the load checked and stated on locust; on Kathleen (if usable) jobs
  only (never builds or runs beyond a short test on a login node), at most 4 nodes per
  job, node-hours stated.

## Components

The IDs of laplace-fmm-plan §7, Phase 6, kept; C6.6–C6.8 are new.

| ID | Component | Acceptance criterion | Depends on | Task |
| --- | --- | --- | --- | --- |
| C6.1 | Fuse gather into GEMM loads (device), with the device kernel work of device-path §18.2 | faster than C4.5 at equal accuracy (bit for bit where the order is kept) on CUDA and Metal | C4.5 | T7 |
| C6.2 | SVD-compressed M2L | error within the budget T1 signs off; memory and time reported against dense and rotation, host and device | C2.2, C4.5, C6.6 | T3, T4 |
| C6.3 | Multiple right-hand sides (charge vectors as extra GEMM columns) | throughput per right-hand side improves with the batch size; each vector's output bit for bit its single-vector evaluation, or within T1's bound if the order changes | C4.5, C6.6 | T5 |
| C6.4 | Dipole sources and other source types | matches the direct sum within the Phase 3 bounds | C3.2 | T8 |
| C6.5 | Plane-wave M2L (optional) | beats C4.5/C4.6 (and C6.6) on some configuration, else dropped | C4.7, C6.6 | T10 |
| C6.6 | Host batched M2L (per-level GEMMs over gathered columns) | within requirement 1 of the target-by-target host path; bit-identical for every thread count; faster than every existing host strategy at p ≥ 6 on one rank, and scaling with ranks per node where Dense does not (Phase 5's tables) | C3.3, C4.5 | T2 |
| C6.7 | AVX-512 P2P | within simd-p2p.md's per-pair contract (8 u_T for φ, 16 u_T for ∇φ; §3, requirement 2); faster than AVX2 on Kathleen in f32 and f64 at FMM leaf sizes, or recorded as not worth it | C3S.3 | T6 |
| C6.8 | FMM3D comparison | both codes at matched accuracy on the same problems and machine, one node; the build documented | C3.3 | T9 |

## Exit gate
- Every acceptance test in the task briefs passes: in CI for the default members; in
  `run-tests-mpi` at 2 and 4 ranks; by hand at 1, 2, 4 and 8 ranks on the M3 Max and
  locust, and on 2 Kathleen nodes for anything that touches the multi-rank path (if
  Kathleen is usable; otherwise at up to 72 ranks on locust).
- `docs/design/optimisation.md` is signed off before T2 starts.
- Without Kathleen, C6.7 is recorded as waiting for AVX-512 hardware, not as failed;
  the gate is the rest.
- C6.6 (T2) and C6.2 (T3, T4) meet their criteria, or the PR reports why not and the
  sign-off records it; C6.3, C6.4, C6.7 and C6.8 meet theirs; C6.1 is adopted or dropped
  by measurement; C6.5 is adopted or dropped at its spike.
- T11's report on the three machines, with the defaults signed off and the design
  documents updated.

## Tasks

One pull request each.
- T1 first. T2 needs T1 signed off.
- T3 needs T2 (compression is applied through the batched layout); T4 needs T3.
- T5 needs T2 for the host part; its device part needs nothing new.
- T6 needs Kathleen (or another AVX-512 machine); without one it waits.
- T6, T7 and T8 need only T1 and can run beside T2–T5 (disjoint crates: `nd-fmm-simd`;
  `nd-fmm-kernels`; `nd-fmm-ref` and the leaf operators).
- T9 needs T2 (the comparison uses the fastest host path).
- T10 needs T2 and T3 (it competes with both).
- T11 needs every other task merged or closed.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-optimisation-design.md](T1-optimisation-design.md) | `docs/design/optimisation.md`: baseline per machine, the host batched M2L, SVD compression, several right-hand sides, AVX-512 P2P, order of the rest by measured gain, sign-off questions | design for C6.x | Phase 5S complete |
| T2 | [T2-host-batched-m2l.md](T2-host-batched-m2l.md) | the host batched M2L (and M2M/L2L where it pays): per-level GEMMs, deterministic, measured on the three machines and at many ranks per node | C6.6 | T1 |
| T3 | [T3-svd-m2l.md](T3-svd-m2l.md) | SVD-compressed M2L tables and their host application; error budget, memory and time against dense and rotation | C6.2 (host) | T1, T2 |
| T4 | [T4-svd-m2l-device.md](T4-svd-m2l-device.md) | compressed M2L on Metal, CUDA and the CPU runtime | C6.2 (device) | T3 |
| T5 | [T5-multiple-rhs.md](T5-multiple-rhs.md) | several charge vectors per evaluation, host and device, through the exchanges | C6.3 | T2 |
| T6 | [T6-avx512-p2p.md](T6-avx512-p2p.md) | `arch::avx512` in `nd-fmm-simd`, measured on Kathleen (waits without an AVX-512 machine) | C6.7 | T1; Kathleen usable |
| T7 | [T7-fused-gather.md](T7-fused-gather.md) | the gather fused into the device GEMM loads, and the device kernel work of device-path §18.2 | C6.1 | T1 |
| T8 | [T8-dipoles.md](T8-dipoles.md) | dipole sources and a source-type interface, reference to device | C6.4 | T1 |
| T9 | [T9-fmm3d-comparison.md](T9-fmm3d-comparison.md) | FMM3D at matched accuracy, one node, the build in `tools/` | C6.8 | T2 |
| T10 | [T10-plane-wave-spike.md](T10-plane-wave-spike.md) | a plane-wave M2L spike; adopt or drop | C6.5 | T2, T3 |
| T11 | [T11-benchmarks.md](T11-benchmarks.md) | final benchmarks on the three machines, defaults revisited (strategy, leaf size per p), design-document update | gate | all |

T2, T3, T5 and T8 all change `fmm-exec/src/operator.rs` and `tables.rs`: merge one, then
rebase the next. T4, T5, T7 and T8 change `nd-fmm-kernels`.

## Machines

| Machine | Ranks | Used for |
| --- | --- | --- |
| Apple M3 Max (development; 12 performance and 4 efficiency cores, 64 GB, Metal) | 1–12 | development; host timings (NEON, Accelerate); Metal f32 |
| locust (GH200; 72 Neoverse-V2 cores, one NUMA node, one H100; tools/gh200/) | 1–72 | host timings (NEON, OpenBLAS); CUDA f32 and f64; the load checked and stated |
| Kathleen, if usable (UCL cluster, Slurm; 40 Cascade Lake cores per node, AVX-512, two NUMA domains, Omni-Path; tools/kathleen/) | 1–40 per node; 2 nodes routinely, **at most 4 nodes (160 cores) in any job** | AVX-512 (T6); x86 host timings (AVX2/AVX-512, OpenBLAS or MKL); many ranks per node (T2); multi-node runs with the new operators at up to 4 nodes (T11) |
| GitHub Actions (`ubuntu-latest`, `ubuntu-24.04-arm`) | 1; 2 and 4 in `run-tests-mpi` | correctness; GitHub's x86_64 runners usually lack AVX-512, so its tests skip there |

## Decisions to sign off

Each is recorded in the exit checklist when made:
1. The design document `docs/design/optimisation.md`, with any change it proposes to the
   requirements above (T1; before T2).
2. BLAS on the host: rlst's BLAS (which library per machine, single-threaded inside
   rayon) or a hand-written host GEMM; and whether M2M and L2L move to the batched form
   too (T1).
3. The error budget of the compressed M2L (C6.2): the truncation rule (a relative
   singular-value cut, or a rank per p), and whether it may become a default (T1, T11).
4. AVX-512 in `nd-fmm-simd` (T1, T6): hand-written `core::arch` intrinsics in
   `arch::avx512` (stable Rust; check the release that stabilised the AVX-512 target
   features against the toolchain in use), the `Isa::Avx512` dispatch rule (for example,
   only where it measures faster than AVX2), and how CI covers it without AVX-512
   runners.
5. The FMM3D build route (T9): its Fortran build on each machine, outside the workspace,
   and which of its interfaces is compared.
6. Whether C6.5 proceeds past its spike (T10).
7. The defaults at the end (T11): the M2L strategy rule with C6.6 and C6.2, the leaf size
   per p, and any one-rank output change that follows.

## Risks

| Risk | Mitigation |
| --- | --- |
| A host GEMM through BLAS changes bits with the thread count or the BLAS build (blocking, threading inside BLAS, FMA use) | requirement 2: single-threaded BLAS inside rayon, the bits checked across thread counts and against the hand-written fallback; a hand-written kernel if a BLAS cannot be pinned (T1 decides) |
| The batched host M2L needs gathered columns and scratch per thread that grow with the level, and memory per rank grows | T1 states the scratch per thread as a formula (chunked like the device's scratch budget); T2 measures it at 40 ranks per Kathleen node |
| SVD compression trades accuracy silently | an error budget signed off (decision 3); opt-in until T11; tested against the direct sum at every p |
| AVX-512 clocks down Cascade Lake cores (frequency licences), so it is slower in an FMM than in a kernel benchmark | T6 measures inside the FMM at 40 ranks per node, not only the kernel; adopted only where faster end to end |
| Kathleen's queue makes it unusable, or a job larger than 4 nodes would be needed to show something | the without-Kathleen path above (decision 0 of Phase 5N); no job above 4 nodes, ever: multi-node questions beyond that are answered by model, marked as such |
| GitHub's x86_64 runners lack AVX-512, so CI never runs the new path | runtime dispatch with a scalar check; the AVX-512 tests run by hand on Kathleen and skip cleanly elsewhere; T1 checks whether any CI runner has it |
| FMM3D cannot be built on one of the machines, or compares unlike things (accuracy definitions, precomputation) | matched accuracy against the same direct sum; FMM3D's own setup time reported apart; one machine is enough for the gate |
| Plane-wave M2L is a large project for an uncertain gain | a spike with a stop rule (T10); dropped if it cannot beat C6.6 on any configuration |
| The briefs predate Phase 5N and 5S | Phase 5S T9 revises them; T1 re-checks every number it cites |

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase6/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [ ] Briefs revised by Phase 5S T9
- [ ] T1 merged: `docs/design/optimisation.md` drafted, sign-off questions listed
- [ ] Optimisation design signed off (decisions 1–5)
- [ ] T2 merged: host batched M2L, deterministic, measured on the three machines (C6.6)
- [ ] T3 merged: SVD-compressed M2L on the host (C6.2)
- [ ] T4 merged: SVD-compressed M2L on the device (C6.2)
- [ ] T5 merged: several right-hand sides (C6.3)
- [ ] T6 merged: AVX-512 P2P, measured on Kathleen (C6.7), or recorded as waiting for AVX-512 hardware (Kathleen unusable)
- [ ] T7 merged: fused gather and device GEMM work, adopted or dropped (C6.1)
- [ ] T8 merged: dipole sources (C6.4)
- [ ] T9 merged: FMM3D comparison (C6.8)
- [ ] T10 merged: plane-wave spike; C6.5 adopted or dropped (decision 6)
- [ ] T11 merged: final benchmarks; defaults signed off (decision 7)
- [ ] Design documents updated: laplace-fmm-plan §4, §6, §7 (Phase 6 status and numbers), §8.3, §9.1, §9.2; simd-p2p.md §4.7; device-path.md (outcome section); workspace-structure §3, §6
