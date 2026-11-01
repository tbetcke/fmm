# CH3.6 Gate: RCR on every level against FMM3D

Started 2026-11-01 on `tbetcke/h-ch3-6`. Roadmap: "CH3.6 Gate: RCR on every level against
FMM3D" (H3; the last item of the priority list). Proposal: §10 CH3.6; §2 (constraints 2,
6, 9), §9 (items 1, 9), §9.1, §9.2 (H3 row).

## Goal

Run our Helmholtz FMM (RCR on every level, CH3.3's degree rule, CH3.4's vectorised P2P)
and FMM3D's `hfmm3d` on the same problems against the same sampled direct sum, report
their errors and times, and compare their outputs at every target.

## Acceptance (proposal §10, verbatim)

"both codes against the same sampled direct sum on W1, W3 and W4 for kD ≤ 100, and on W1
at kD = 500 with time reported; the full-target crosscheck". Roadmap: the same, "timed with
CH3.4's vectorised P2P"; *Machines:* Kathleen.

Read as (to be confirmed in the PR):
- for every case, our FMM and FMM3D at the same requested tolerance ε ∈ {10⁻³, 10⁻⁶,
  10⁻⁸} (decision D1), both measured with §9.1's relative ℓ2 norms of φ and ∇φ over the
  same 1,000 sampled targets, against the same f64 direct sum
  (`nd_fmm_ref::helmholtz::p2p::direct_sum` × 1/(4π), which CH1.4 showed equal to FMM3D's
  `h3ddirectcg` to round-off); our errors at most ε (CH3.1c's criterion, now on the
  default rule); FMM3D's reported, not gated;
- time reported for both on one Kathleen node: ours build and evaluation (the slowest
  rank), FMM3D's whole call (its setup included, as it keeps nothing between calls);
- the full-target crosscheck (§9.2): our output at every point against FMM3D at a tight
  eps (10⁻¹²), the relative ℓ2 of φ and ∇φ over all N targets, and the maximum error per
  box (the boxes of the FMM's deepest level, the error at a target divided by the RMS of
  the reference over all targets), so that an error confined to a few boxes shows; the
  same for FMM3D at ε against FMM3D at 10⁻¹².
- not in scope (proposal §9.2): performance at matched accuracy (after CH4.5 and CH7.6);
  the mapping of box errors to offset classes (needs the high-frequency levels, H4).

## Read first

- Proposal §9.1, §9.2; `fmm-validate/results/helmholtz-ch3-1c.md`, `helmholtz-ch3-3.md`,
  `helmholtz-ch1-4.md` (R12: FMM3D's real k passes to 10⁻¹², a 2× gradient miss at 10⁻¹²
  on a sphere at N = 10⁶).
- `nd_fmm_validate::helmholtz_accuracy` (`Case`, `Problem`, `Oracle`, `errors`, `run`).
- `tools/fmm3d/{driver.f90, run.sh, helmholtz.py, README.md}` (the `helmholtz` mode and
  its files `NDFMMHP1`, `NDFMMHO1`), `tools/kathleen/jobs/fmm3d.sbatch`.
- `fmm-validate/results/phase6-fmm3d.md` (T9, the Laplace comparison's method).

## Changes

`nd-fmm-validate` and `tools/` only (development tooling; no library crate changes):

1. `tools/fmm3d/driver.f90`: the `helmholtz` mode gains an optional 8th argument FULL; with
   FULL = 1 every FMM token also writes φ and ∇φ at every source to
   `PREFIX-<TOKEN>.full.out` (`'NDFMMHF1'`, int64 n, complex(16) pot(n), grad(3,n)).
2. `fmm-validate/src/helmholtz_fmm3d.rs` (new): the driver's files (write `NDFMMHP1` from a
   `helmholtz_accuracy::Problem`, one charge vector; read `NDFMMHO1` and `NDFMMHF1`); the
   driver's `helmholtz:` line (its seconds); FMM3D's errors at the samples against the
   `Oracle`; the crosscheck of a distributed `Output` (or of FMM3D's own output) against a
   full reference (sums of squares all-reduced, the per-box maxima gathered to rank 0);
   the boxes of a level of the FMM's `Domain`.
3. `fmm-validate/examples/helmholtz_fmm3d.rs` (new, not registered: it times):
   `problem` writes the problem file of a case; `compare` runs our FMM on the same case
   for each ε (any ranks), reads FMM3D's outputs and log, and prints one Markdown row per
   ε with both codes' errors and times and both crosschecks.
4. `tools/fmm3d/run.sh`: a step `helmholtz` (the cases of `H_CASES`, default the gate's:
   for each, `problem`, the driver at every ε and the reference eps with FULL = 1, each
   variant of `VARIANTS`, then `compare` under `mpirun` at `H_SPLIT`); `fmm3d.sbatch`
   passes it through; `tools/fmm3d/README.md` documents it.
5. `tests/helmholtz_accuracy.rs` (the MPI-owning test): the compare path on a small case
   with a synthetic FMM3D output (the direct sum at every point written in the driver's
   formats): our crosscheck error equals our FMM's error against the direct sum over all
   targets, FMM3D's sampled errors are zero to round-off.
6. `fmm-validate/CLAUDE.md`; the report `fmm-validate/results/helmholtz-ch3-6.md`.

No public API outside `nd-fmm-validate`; no table, layout, convention, dependency or
manifest change other than the example's `[[example]]` entry if needed.

## Tests that prove it

- The acceptance is the runs (below), each against the sampled f64 direct sum.
- Unit tests (no MPI): the problem file round trip; `NDFMMHO1` and `NDFMMHF1` read from
  files written in the driver's byte layout; the driver line parsed; the box of a point at
  a level (corners, the upper face); the crosscheck norms on a hand-made example.
- `tests/helmholtz_accuracy.rs` (debug; one rank in CI's `cargo test`, at 2 and 3 ranks by
  hand: it is not in `run-tests-mpi`'s list): the synthetic FMM3D case above, the
  full-target error equal on one rank and on several.
- The FMM3D driver's FULL output: on the M6, its full output at the sampled positions bit
  for bit its sampled output (checked by `compare`, which refuses a mismatch).

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, the default and
workspace tests, `cargo doc`, the MPI list at 2 and 4 ranks), with
`cargo clippy -p nd-fmm-validate --all-targets -- -D warnings` and
`cargo test -p nd-fmm-validate`; `tools/fmm3d/build.sh driver` on the M6 and Kathleen.
No Laplace file changes, so no D9.

## Machine runs

- M6 (smoke, both FMM3D variants): W1 at kD = 10 and 30, W4 cube at N = 10⁴, all ε.
- Kathleen, one node (`singlenode`), release, BLAS threads 1, the driver relinked: the
  gate's cases, each at ε = 10⁻³, 10⁻⁶, 10⁻⁸, ours at 4 × 10 ranks × threads (T9's split),
  FMM3D (`default` and `fastker`) on 40 OpenMP threads:

  | Workload | kD | N |
  | --- | --- | --- |
  | W1 | 10, 30, 100, 500 | 796 to 1,989,437 |
  | W3 resolved | 10, 30, 100 | 4,032 to 4,031,442 |
  | W4 cube, sphere, plummer, clusters | 1, 10, 100 | 10⁵ |

  Plus one thread for both codes on W1 at kD = 100 (all ε). Decides the acceptance.

## Constraints and risks

- Constraint 2 (determinism): unchanged code; the checksum printed as in CH3.1c.
- Constraint 6: every number against the f64 direct sum; FMM3D's tight output only as the
  crosscheck's reference, its own error at the samples printed beside it.
- Laplace: not touched.
- Risks: FMM3D at 10⁻¹² on W3 at kD = 100 (4 × 10⁶ points) may be slow (hours?): measured
  first on the M6 at smaller N; fallback the reference at 10⁻¹⁰. A job's 6-hour limit:
  split the cases over jobs.

## Pending runs

Kathleen, submitted 2026-11-01 from a6b3aa7 (`tools/kathleen/sync.sh`):
- 617649 `fmm3d.sbatch driver` (the driver relinked for FULL);
- 617650 (after 617649) `fmm3d.sbatch helmholtz`, OUT `logs/ch3-6-w1`: W1 at kD = 10, 30,
  100, 500, and W1 at kD = 100 on one thread;
- 617651 (after 617649) `fmm3d.sbatch helmholtz`, OUT `logs/ch3-6-w34`: W3 at kD = 10, 30,
  100 and W4 (4 distributions × kD = 1, 10, 100).
They decide the acceptance (both codes' errors and times, the crosscheck).

## Smoke runs on the M6

`tools/fmm3d/run.sh helmholtz`, 1 × 6 ours, FMM3D on 6 threads, both variants: W1 at
kD = 10, 30, 100 and the W4 cube at N = 10⁴, kD = 10. Our sampled and all-target errors
within ε everywhere; FMM3D at eps = 10⁻³ on W1 at kD = 100 at 1.2e-3 (above its eps).
The crosscheck finds our worst box 30–60× our own all-target ℓ2 error at 10⁻⁶ and 10⁻⁸
(FMM3D 13–18×), up to 22 ε pointwise (∇φ, W1 at kD = 100, 10⁻⁸, relative to the RMS).
On W1 the worst boxes are the six where the sphere touches its bounding cube; with a
margin of 0.03 or 0.1 around the domain (a throwaway experiment, not committed) they move
elsewhere and the ratio stays (20–70×): not a boundary effect.

## Open questions

1. The worst-box ratio above: the default rule meets ε in §9.1's ℓ2 norm, but its errors
   are more concentrated than FMM3D's. A question for the PR (a follow-up, not a change
   here).
