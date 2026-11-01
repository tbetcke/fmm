# CH3.1c The acceptance runs

Started 2026-11-01 on `tbetcke/h-ch3-1c`. Roadmap: "CH3.1c The acceptance runs" (H3; the
third sub-task of CH3.1, tenth on the priority list). Proposal: §10 CH3.1; §2
(constraints 2, 9), §9 (items 1 and 6), §9.1.

## Goal

Show the Helmholtz FMM of CH3.1a and CH3.1b accurate and deterministic on the reference
workloads at the sizes of the gate, and record it in a report.

## Acceptance (proposal §10, verbatim)

CH3.1: "FMM against the direct sum on W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500;
deterministic". Roadmap's CH3.1c: the same, plus "a report".

Read as (to be confirmed in the PR): for every row, the relative ℓ2 errors of φ and ∇φ
over the sampled targets (§9.1's norms) at most the row's ε, with the degrees of S7's
closed form at ε (`Degrees::Tolerance`); "deterministic" as constraint 2: the output's
checksum the same at every thread count and rank count run for that row.

## Read first

- Proposal §9.1 (workloads, charges, 1,000 sampled targets, the norms), §2 constraint 2.
- `nd_fmm_validate::{helmholtz_workloads, fmm_accuracy::Problem, metrics::ErrorAccumulator,
  rng::SplitMix64}`.
- `nd_fmm_exec::helmholtz::{HelmholtzFmmBuilder, HelmholtzFmm, Charges, Output, Degrees,
  Wavenumber}`; `nd_fmm_ref::helmholtz::p2p::direct_sum`.
- `fmm-exec/tests/helmholtz_fmm.rs` (the CH3.1b scenarios and errors).

## Changes

One crate, `nd-fmm-validate` (development tooling; no library crate changes):

1. `src/helmholtz_accuracy.rs` (new, MPI): a `Case` (workload, kD as a `Wavenumber`, ε or
   named degrees, real or complex charges, vectors, gradients, samples, threads); the
   problem (the workload's points; 1,000 distinct sampled targets by the partial
   Fisher–Yates shuffle of `fmm_accuracy::Problem`; charge vectors with parts uniform in
   [−1, 1), seeded per vector); each rank passes a contiguous block of the points; the
   oracle (`direct_sum` in f64 at the sampled targets over every point, the samples
   sharded over ranks and over `std::thread::scope` threads, one all-reduce); the errors
   (§9.1's relative ℓ2 of φ and of the 3-vector ∇φ, per vector and over all vectors); a
   rank-invariant checksum of the whole output (a wrapping sum over every value of a mix
   of its global point index, output slot and bits, summed over ranks); build and
   evaluation times (the slowest rank).
2. `examples/helmholtz_accuracy.rs` (new; a timing example, not registered for the weekly
   job): the cases from the command line (`--workload`, `--kd`, `--eps`, `--charges`,
   `--vectors`, `--threads`, `--samples`, `--no-gradients`), one Markdown row per case:
   N, levels, the leaf level, the degrees, times, errors, checksum.
3. `tests/helmholtz_accuracy.rs` (new, one MPI test): a smoke run on small cases (W1 at
   kD = 10, W4 at N = 2,000): the errors within ε, the checksum the same at 1 and 2
   threads, the sharded oracle the unsharded one bit for bit; and without MPI the checksum
   independent of the partition.
4. `fmm-validate/CLAUDE.md`: the module, the example and the report.
5. The report `fmm-validate/results/helmholtz-ch3-1c.md`.

No public API outside `nd-fmm-validate`, no table, layout, convention, dependency or
manifest change other than the new example's `[[example]]` entry if one is needed.

## Tests that prove it

- The acceptance itself is the runs (below), each against the sampled direct sum in f64
  (`nd_fmm_ref::helmholtz::p2p::direct_sum`, constraint 6).
- `tests/helmholtz_accuracy.rs`: the module's oracle, sampling, checksum and errors on
  small cases (debug, one rank in CI; by hand at 2 ranks).

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, the default and
workspace tests, `cargo doc`), with `cargo clippy -p nd-fmm-validate --all-targets -- -D
warnings` and `cargo test -p nd-fmm-validate`. No compiled change to Laplace's paths, so
no D9 or byte comparison.

## Machine runs

On the M6 (6 performance cores, 24 GB), release, BLAS threads 1, outside the sandbox:

| Workload | kD | N | ε |
| --- | --- | --- | --- |
| W1 | 10, 30, 100 | 796 to 79,578 | 10⁻³, 10⁻⁶, 10⁻⁸ |
| W1 | 500 | 1,989,437 | 10⁻³ (and 10⁻⁶ if it fits) |
| W3 resolved | 10, 30, 100 | 4,032 to 4,031,442 | 10⁻³, 10⁻⁶ |
| W4 cube, sphere, Plummer, clusters | 1, 10, 100 | 10⁵ | 10⁻³, 10⁻⁶ |

Complex charges, gradients, one charge vector (eight on W1 at kD = 100 to show the
per-vector spread); real charges on W1 at kD = 100; a complex k (100 + 10i) on W1. The
sizes are confirmed by a first timing; any row cut for time is named in the report.
Determinism: each row at 1 rank × 6 threads; the checksums of 1 × 1 and 3 × 2 (and 2 × 3)
compared on the rows that fit. Kathleen only if the M6 cannot hold W1 at kD = 500.

## Constraints and risks

- Constraint 2 (determinism and rank invariance): the checksum comparisons.
- Constraint 9: the degrees and the leaf level depend on k, D, ε and c_leaf only.
- Laplace: not touched (no compiled change outside `nd-fmm-validate`).
- Risk: time. The P2P is the reference kernel until CH3.4 and RCR runs on every level; W1
  at kD = 500 (2 M points, levels up to p ≈ 130) and W3 at kD = 100 (4 M points) may be
  hours. Then fewer ε, or the run on Kathleen.
- Risk: S7's closed form overshoots on low-frequency levels (CH3.1b), so errors far below
  ε; that is CH3.3's calibration, recorded here, not tuned.

## Results

Ran on the M6 on 2026-11-01 (report `fmm-validate/results/helmholtz-ch3-1c.md`): 61 cases,
81 runs with the repeats, every error of φ and ∇φ below ε (largest 3.7e-3 ε, ∇φ on W3 at
kD = 100, ε = 10⁻³); equal checksums at 1 and 6 threads on 13 cases and on several rank
counts on 4 (W1 at kD = 500 on 1 and 3 ranks). As built:

- the oracle scales `direct_sum` by 1/(4π): the reference leaves it out (its first run gave
  errors of 1 − 1/(4π) = 0.92 everywhere);
- ε = 10⁻⁸ on W3, W4 and W1 at kD = 500 ran in a second matrix after the self-review named
  them missing against D1; W3 at kD = 100 at 10⁻⁸ was not run (about 21 GB, beyond the M6);
- `tests/helmholtz_accuracy.rs` (26 s at one rank in debug) passed at 1, 2 and 3 ranks;
  on several ranks the errors equal the one-rank errors within 1e-12 relative (the sums
  reduce in another order), the checksums exactly.

## Pending runs

None.

## Open questions

None blocking. For sign-off in the PR: the reading of "against the direct sum" as errors
at most ε at S7's degrees, and W3 at kD = 100 at 10⁻⁸ left out.
