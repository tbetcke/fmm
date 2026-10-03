# Phase 3S: SIMD P2P on the host

Phase 3S makes the near field of the host FMM fast. It sits between Phase 3 (the CPU
FMM on the rewritten `nd-fmm-plan`) and Phase 4 (CubeCL kernels). It is called 3S
rather than 4 so that the phase and component numbers that the code and documents
already cite (Phase 4, C4.x, C5.x) keep their meaning.

The Phase 3 calibration (T12) measured that the leaf stage, dominated by P2P, takes
72–81% of an evaluation at p = 3 and 20–51% at p = 8 on one thread. P2P is
`nd_fmm_ref::p2p`, a scalar loop with a square root and divisions per pair, which the
compiler cannot vectorise. This phase writes hand-written SIMD kernels for aarch64
NEON and x86_64 AVX2 + FMA, with no portable SIMD library. 1/r comes from the hardware
inverse-square-root estimate refined by Newton steps to full precision. The kernels are
used in `LaplaceOperator` and benchmarked against the Laplace kernels of
[green-kernels](https://github.com/bempp/green-kernels).

No x86_64 machine is available for timings (decided on 2026-10-02). The phase therefore
**times NEON only**, on the Apple M3 Max. The x86_64 path (AVX2 + FMA) is written,
tested for correctness and accuracy on real x86_64 hardware in CI, and judged for speed
by its instruction counts against the operation-count model of
[simd-p2p.md](../design/simd-p2p.md) §4.6, not by timings.

AVX-512 is deferred (decided on 2026-10-02) until hardware to test and time it is
available. The design keeps room for it: `Isa` is `#[non_exhaustive]`, the kernel body
is generic over the ISA layer, and simd-p2p.md §4.7 records what an AVX-512 path would
use.

The phase has three parts:
1. **Decisions.** The design document
   [docs/design/simd-p2p.md](../design/simd-p2p.md) (signed off by hand), an addition
   to CONVENTIONS §3.13 on how fast kernels exclude coincident pairs, and a spike that
   measures loop orders, inverse-square-root variants and green-kernels before any
   production code is written.
2. **The crate `nd-fmm-simd`.** A scaffold with ISA detection, dispatch and a scalar
   path; the per-ISA vector layer and inverse square root; the P2P kernel.
3. **Use and measurement.** `nd-fmm-exec` uses the kernel by default with the reference
   path still selectable, the Phase 3 accuracy gates are re-run, and a benchmark report
   compares the kernel with `nd_fmm_ref::p2p` and green-kernels, alone and inside the
   FMM.

Companion documents: [docs/design/simd-p2p.md](../design/simd-p2p.md) (all of it; it is
the design of this phase), [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md)
(Section 7, Phase 3 "Per-pair cost" and "Recommendation for Phase 4", and Phase 3S;
Sections 6.8, 8.3 and 9) and
[docs/design/workspace-structure.md](../design/workspace-structure.md) (Sections 2, 3,
3.1 and 5.1).

Prerequisite: Phase 3 is complete. T12 is merged (PR #39) and the design documents carry
the Phase 3 outcome. The briefs assume the workspace as it is after that merge.

## Scope

In scope:
- `docs/design/simd-p2p.md`, signed off; the design-document updates at the end of
  the phase.
- `docs/CONVENTIONS.md` §3.13: an addition on the coincident-pair rule of fast kernels
  and the domain of r², with `tools/fixtures/check_p2p_domain.py`.
- `spikes/p2p-simd/`: prototype kernels, the inverse-square-root study and the
  green-kernels baseline; later the green-kernels comparison of the production kernel.
- `nd-fmm-simd` (new crate, `fmm-simd/`): ISA detection and dispatch, a scalar path,
  the per-ISA vector layer and inverse square root, and the P2P kernel (potential and
  gradient, f32 and f64) for NEON and AVX2 + FMA.
- `nd-fmm-exec`: the P2P of `LaplaceOperator` through `nd-fmm-simd`, selectable;
  `FmmBuilder::p2p_kernel` and `Fmm::p2p_kernel`.
- `nd-fmm-validate`: a P2P kernel benchmark example, and the FMM timings and leaf-size
  study with the new kernel.
- `.github/workflows/`: a job for `nd-fmm-simd` on an x86_64 and an arm64 runner,
  with the release-mode accuracy tests (approved on 2026-10-02; T3).
- The default `max_points_per_leaf`: T7 measures it and adopts its own recommendation
  (approved in advance on 2026-10-02).

Out of scope:
- SIMD for P2M, L2P, P2L, M2P and the translations. At p ≥ 8 the far field dominates,
  and its host answer is the batched GEMM path of Phase 4.
- AVX-512, until hardware to test and time it is available (a later task adds
  `Isa::Avx512` behind the same interface).
- Portable SIMD libraries (`pulp`, `wide`, `std::simd`), C or assembly sources, and
  SVE/SVE2.
- Threading inside the kernel, and any change to the threading of Phase 3 (C3.5).
- Changes to the leaf layout of §3.13, to `nd-fmm-plan`, `nd-fmm-tables` or `nd-octree`,
  and to `nd-fmm-ref` beyond fixing a defect that a test exposes (stop and report
  first).
- Other kernels and source types (Helmholtz, dipoles C6.4, several charge vectors
  C6.3).
- x86_64 timings of any kind, including the green-kernels comparison on x86_64. The
  spike and the T7 harness compile and run on x86_64, so the comparison can be added by
  whoever later has a machine.

## Requirements

[docs/design/simd-p2p.md](../design/simd-p2p.md) §3 lists the ten requirements the
kernel is designed and accepted against: the same operator and signature as
`nd_fmm_ref::p2p`; stated accuracy per pair and per sum; exclusion by r² = 0; each
target's sources in input order, with chunk and target-position invariance; determinism;
no allocation, threads or MPI; four ISA paths with runtime dispatch; tests on every ISA
a machine offers; unsafe confined; timings measured, never asserted. T1 or T2 may
propose changing one, with reasons, for sign-off.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **A new crate.** `fmm-simd/`, package `nd-fmm-simd`, library `nd_fmm_simd`, a default
  member. It depends on `nd-fmm-math` and `thiserror` only (dev-dependencies
  `nd-fmm-ref`, `proptest`), so it builds and tests without MPI. It holds the
  hand-written host kernels; P2P is the first. `nd-fmm-exec` and `nd-fmm-validate`
  depend on it.
- **Hand-written intrinsics.** Only `core::arch` (`std::arch`) intrinsics: aarch64 NEON
  and x86_64 AVX2 + FMA. The crate's own thin internal layer over them (one
  module per ISA, one generic kernel body) is allowed; external SIMD crates are not.
  Stable Rust only.
- **Dispatch.** NEON is always available on aarch64. On x86_64,
  `is_x86_feature_detected!` picks AVX2 + FMA, else the scalar path, once at
  construction. An AVX-512 machine runs the AVX2 path. Every path is selectable
  explicitly (`Isa`), for tests and benchmarks. No environment variable selects an ISA.
- **Loop order** (provisional, design §4.1): targets in lanes, sources broadcast, each
  target's sources in input order. T2 confirms it by the decision rule of design §4.1.
  Changing it needs sign-off, because it would relax the C3.1 identity between the
  per-pair and the batched P2P. The rule is applied to NEON measurements only, and the
  chosen order is used on every ISA.
- **Inverse square root** (design §4.3). The hardware estimate, refined by Newton steps
  or an equivalent polynomial correction, to within 4 u_T relative (u_T = 2⁻²⁴, 2⁻⁵³),
  f32 checked exhaustively over [1, 4). T2 chooses the formulation per ISA and
  precision:
  - for NEON by measured error and throughput;
  - for AVX2 from the documented worst-case error bound of its estimate and the
    operation count, with a margin, since nothing can be timed there. The
    exhaustive tests of T4 confirm it on real x86_64 hardware in CI.

  `sqrt` and division stay the scalar path and the yardstick. A relaxed f64 level ships
  only if T2's sign-off says so (recommended: not).
- **Coincident pairs** (design §4.4). Fast kernels exclude a pair by r² = 0. T1 shows
  that on leaf-scaled data this is the exact-coincidence rule of §3.13 and states the
  domain of r² the kernels support.
- **Same operator, same layout.** `P2pKernel::evaluate` has the signature and semantics
  of `nd_fmm_ref::p2p::p2p` (design §5.1): it adds into the outputs, applies no 1/(4π)
  and reads the §3.13 chunks as stored. §3.13's layout does not change.
- **Order and identity.** Within one machine, ISA and build:
  - results are bit-identical from run to run;
  - they are bit-identical for every thread count of `Fmm` (C3.5);
  - they are bit-identical between `PerPair` and the batched path (C3.1);
  - chunk and target-position invariance hold bit for bit.

  Results differ between ISAs, and from `nd_fmm_ref::p2p`, within the accuracy of
  design §3, requirement 2.
- **Accuracy measures.** Per pair: relative to the reference term (potential) and to
  |q| / r² (gradient components). Sums: relative to the sum of term magnitudes, against
  `direct_sum`, as in Phase 1 T4. FMM: the Phase 3 measures (docs/phase3/README.md,
  "Error measures") and the reference-P2P run of the same `Fmm`.
- **Unsafe** (design §5.4). Only in `nd-fmm-simd`'s `arch` modules and its dispatch,
  each block with `// SAFETY:`, under `#![deny(unsafe_op_in_unsafe_fn)]` and
  `#![deny(clippy::undocumented_unsafe_blocks)]`. Every public function is safe.
  `nd-fmm-exec` stays free of `unsafe`. This is an exception to workspace-structure
  §5.1, recorded there.
- **Reference path kept.** `P2pChoice::Reference` runs `nd_fmm_ref::p2p` inside the
  FMM, as in Phase 3. It is the trusted slower path for the FMM tests and the baseline
  of the benchmarks. `nd-fmm-ref` itself does not change.
- **ISA coverage, reported honestly.**
  - CI, x86_64 runner: scalar and AVX2, in debug and with the release-mode accuracy
    tests (`--ignored`). This is the only x86_64
    hardware the phase has.
  - CI, arm64 runner: scalar and NEON, the same tests.
  - The development machine (Apple M3 Max): scalar and NEON, plus AVX2 under Rosetta 2
    for quick correctness checks before CI, if T3 confirms it works. Rosetta emulates
    the x86 estimate instructions, so their bits there need not match hardware.
    Accuracy contracts count only from real hardware.

  Every test run prints the ISAs it exercised. A task report lists which ran, and
  never reports a path that did not run as passing.
- **Benchmarks** (design §8): on the M3 Max (NEON) only; release, default target, one
  thread per kernel, median of 15 batches of at least 20 ms, seeded inputs, machine and
  ISA printed. Timings are reported, never asserted, and never taken in CI. Run with
  every BLAS thread variable set to 1 (design §6.8). For the x86_64 paths, the
  instruction counts of their inner loops stand in for timings.
- **green-kernels** only in `spikes/p2p-simd`. It is pinned to commit
  `7d757c579f8633d58163cd5b2d011a9468541f17` in `[workspace.dependencies]`, with a
  comment that only spikes may use it. No default member depends on it.
- **Dependencies.** No new external dependency in any default member. The spike may use
  green-kernels and what it brings. Anything else needs asking first.
- **Test time.** Debug-mode `cargo test -p nd-fmm-simd` runs in under 30 seconds. The
  exhaustive f32 checks and the large sweeps are `#[ignore]` tests run in release; a
  debug run checks a strided subset.

## Exit gate
- Every acceptance test in the task briefs passes in CI, and on the development machine
  for NEON. `nd-fmm-simd` is a default member.
- `docs/design/simd-p2p.md` is signed off before T4 starts; the §3.13 addition before
  T5; the spike's recommendation before T4.
- C3S.1: `check_p2p_domain.py` passes; the §3.13 addition is signed off.
- C3S.2: the spike report holds the NEON measurements, the green-kernels baseline on
  NEON, the x86_64 formulations chosen from documented bounds and operation counts, and
  a signed-off recommendation.
- C3S.3: on every ISA run, the kernel's inverse square root is within 4 u_T relative:
  f32 exhaustively over [1, 4), f64 on 10⁷ seeded samples, plus the domain ends and
  r² = 0. The disassembly of every entry point has no call in its inner loop.
- C3S.4: on every ISA run, in f32 and f64, with and without gradients:
  - pair terms within 8 u_T (potential) and 16 u_T (gradient) of `nd_fmm_ref::p2p`;
  - sums within 1e-14 (f64) and 1e-6 (f32) of `direct_sum`, relative to term
    magnitudes, for up to 4,096 sources, or within twice the error of
    `nd_fmm_ref::p2p` on the same inputs where that is larger (decided on 2026-10-03
    in T5: in-order f32 sums with gradients exceed 1e-6 on some FMM-shaped sets, for
    the reference too);
  - coincident, empty and accumulating inputs as the reference;
  - chunk and target-position invariance bit for bit;
  - on NEON, at least 90% of the spike prototype's throughput on the gathered
    FMM-shaped workload (reported, not asserted);
  - on AVX2, inner-loop instruction counts within 25% of the operation count of design
    §4.2 (reported).
- C3S.5: with every available ISA and with `Reference`:
  - P2P of `LaplaceOperator` within 1e-13 (terms) of `nd_fmm_ref::p2p` on levels 2, 9
    and 16;
  - the C3.2 and C3.3 gates still pass, with errors within 1% (f64) and 2% (f32) of the
    `Reference` run, and outputs within 1e-13 (f64) and 1e-6 (f32) relative L2 of it;
  - outputs bit-identical for 1, 2, 4 and 8 threads, and between per-pair and batched;
  - `Auto` is the default.
- C3S.6: the T7 report gives NEON throughput against `nd_fmm_ref::p2p` and
  green-kernels, with accuracy. The performance target: on NEON the default kernel
  reaches at least green-kernels' throughput, at equal or better accuracy, in every cell
  of the FMM-shaped (both forms) and all-pairs workloads. A cell below it is analysed
  (operation count, disassembly) and reported for a decision, not hidden. The report
  also gives the FMM speed-ups on the M3 Max, the leaf-size recommendation as adopted,
  and states that no x86_64 path was timed.

## Tasks

One pull request each.
- T1, T2 and T3 touch disjoint files and can start at once.
- T4 needs T3 and the T2 recommendation signed off. T5 needs T4 and T1 signed off. T6
  needs T5. T7 needs T6.
- T2 and T7 both work in `spikes/p2p-simd`: T7 extends what T2 merged.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-p2p-conventions.md](T1-p2p-conventions.md) | CONVENTIONS §3.13 addition (coincident pairs in fast kernels, domain of r²), `check_p2p_domain.py`; sign-off of simd-p2p.md recorded | C3S.1 | none |
| T2 | [T2-simd-spike.md](T2-simd-spike.md) | `spikes/p2p-simd`: loop orders, inverse-square-root variants, register blocking, green-kernels baseline; `SPIKE_REPORT.md` | C3S.2 | none |
| T3 | [T3-scaffold.md](T3-scaffold.md) | `fmm-simd` crate, workspace entries, `Isa` detection and dispatch, scalar kernel, crate `CLAUDE.md`, root `CLAUDE.md` points to Phase 3S, cross-target clippy, the `run-tests-simd` CI job (x86_64 and arm64) | part of C3S.3 | none |
| T4 | [T4-vector-rsqrt.md](T4-vector-rsqrt.md) | per-ISA vector layer and inverse square root, with its accuracy contract | C3S.3 | T2 (signed off), T3 |
| T5 | [T5-p2p-kernel.md](T5-p2p-kernel.md) | the SIMD P2P kernel on NEON and AVX2, potential and gradient, f32 and f64 | C3S.4 | T1 (signed off), T4 |
| T6 | [T6-exec-integration.md](T6-exec-integration.md) | `LaplaceOperator` and `FmmBuilder` on `nd-fmm-simd`, `P2pChoice`, the Phase 3 gates re-run | C3S.5 | T5 |
| T7 | [T7-benchmarks.md](T7-benchmarks.md) | kernel benchmarks against the reference and green-kernels, FMM timings and leaf-size study, design-document update | C3S.6 | T6 |

Review T1 and the T2 recommendation yourself before the tasks that build on them. T4
and T5 encode the spike's choices, and T5 and T6 encode the coincident-pair rule.

T2, T3 and T6 change the root `Cargo.toml` and `Cargo.lock`, and T3 also changes
`.github/workflows/run-tests.yml`. Rebase whichever merges later, and regenerate the
lock file with the manifest change (root `CLAUDE.md`, "Layout").

## Machines

| Machine | ISAs | Used for |
| --- | --- | --- |
| Apple M3 Max (development) | NEON; AVX2 + FMA under Rosetta 2, emulated, quick checks only | every task's tests; all timings (T2, T5, T6, T7) |
| GitHub Actions `ubuntu-latest` (x86_64) | AVX2 + FMA | x86_64 correctness and accuracy, debug and release `--ignored` (from T3); never timings |
| GitHub Actions arm64 runner (`macos-latest`) | NEON | NEON correctness and accuracy on a second aarch64 CPU (from T3); never timings |

No x86_64 machine is available for timings. The spike and the T7 examples print a
self-contained Markdown report and build on x86_64, so a later run there is a single
command. The job log of each CI run names the runner's CPU; a task that relies on the
x86_64 job reports which ISAs it ran (from the test output).

## Decisions to sign off

From [docs/design/simd-p2p.md](../design/simd-p2p.md) §9, each recorded in the exit
checklist when made:
1. The design document, including the unsafe exception (before T4).
2. The §3.13 addition (T1; before T5).
3. The spike's recommendation: loop order, inverse-square-root formulation and register
   blocking per ISA and precision, and whether to ship a relaxed f64 level (T2; before
   T4).
4. An arm64 CI job for `nd-fmm-simd`. **Decided on 2026-10-02: yes.** T3 adds a job
   for `nd-fmm-simd` on an arm64 and an x86_64 runner.
5. The x86_64 machines for x86_64 timings. **Decided on 2026-10-02:
   none is available.** x86_64 is correctness-only, through CI; every timing is NEON on
   the M3 Max.
6. Whether to change `max_points_per_leaf` after the T7 leaf-size study. **Decided on
   2026-10-02: T7 adopts its own recommendation**, in its PR. The recommendation rests
   on M3 Max timings.
7. AVX-512. **Decided on 2026-10-02: deferred** until hardware to test and time it is
   available; Phase 3S ships NEON, AVX2 + FMA and scalar.

## Risks

| Risk | Mitigation |
| --- | --- |
| An ISA path is never exercised | ISAs printed by every run and listed in every report; the `nd-fmm-simd` CI job on x86_64 (AVX2) and arm64 (NEON) runners, with the release accuracy tests |
| The x86_64 kernels are correct but slow, and nobody can measure it | inner-loop instruction counts against the operation-count model (C3S.4); the inlining check; the same generic kernel body as the measured NEON path; x86_64 throughput reported as "not measured" |
| Undefined behaviour in intrinsic code | unsafe confined and documented (design §5.4); bounds by construction; ISA checked by `P2pKernel::new`; every n_t mod W and empty inputs tested on each ISA |
| An estimate is less accurate on some CPU than documented, or differs between vendors | the 4 u_T contract is tested on every machine used, exhaustively in f32; vendor differences recorded (design §5.5) |
| Intrinsics fail to inline and the kernel is slow but correct | disassembly check (T4, T5); T5 compares throughput with the spike's prototype |
| The gain is smaller than modelled (lane waste in small leaves, memory traffic) | the spike measures loop orders and blocking at realistic leaf sizes on NEON before the production kernel exists; AVX2 f32 (8 lanes) wastes more lanes than NEON and cannot be timed, so T2 reports the lane utilisation at the W1 leaf sizes per ISA |
| green-kernels does not build beside rlst 0.9.0 in one workspace | it lives only in the spike; T2 checks the build first and has fallbacks (design §8.4) |
| The SIMD P2P changes FMM results beyond rounding | C3S.5 compares every scenario with the `Reference` run of the same `Fmm` and re-runs the Phase 3 gates |

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase3s/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [x] `docs/design/simd-p2p.md` signed off, including the unsafe exception
- [x] T1 merged: `check_p2p_domain.py` passes; §3.13 addition drafted
- [x] §3.13 addition reviewed and signed off
- [x] T2 merged: spike report with NEON measurements, the green-kernels baseline on NEON and the x86_64 formulations
- [x] Spike recommendation signed off (loop order, inverse square root, blocking, relaxed level)
- [x] x86_64 benchmark machines: none available; x86_64 is correctness-only in CI (decided 2026-10-02)
- [x] arm64 CI job: yes (decided 2026-10-02)
- [x] AVX-512: deferred until hardware is available (decided 2026-10-02)
- [x] T3 merged: `nd-fmm-simd` skeleton, CI green including the new x86_64 and arm64 `nd-fmm-simd` job, root `CLAUDE.md` points to Phase 3S
- [x] T4 merged: inverse square root within 4 u_T on every ISA run; no calls in the inner loops
- [ ] T5 merged: kernel accuracy and invariance on every ISA run; throughput against the spike reported
- [ ] T6 merged: Phase 3 gates pass with every ISA; bit-identity for threads and per-pair; `Auto` default
- [x] Leaf-size default: T7 adopts its own recommendation (decided 2026-10-02)
- [ ] T7 merged: NEON benchmark report against the reference and green-kernels; FMM speed-ups; leaf-size recommendation applied
- [ ] Design documents updated: laplace-fmm-plan §7 (Phase 3S status and numbers), §8.3, §9.1 and §9.2; simd-p2p.md (decisions and measurements); workspace-structure §3 and §3.1 match the result
