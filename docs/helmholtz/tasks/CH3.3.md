# CH3.3 Calibration of p_l over (k a_l, ε)

Started 2026-11-01 on `tbetcke/h-ch3-3`. Roadmap: "CH3.3 Calibration of p_l over (k a_l, ε).
*Done when:* S7's low-frequency tables confirmed in the full FMM. *Needs:* CH3.1, CH0.7."
Proposal: §10 CH3.3; §3.5 (truncation: "p_l is calibrated empirically, as in C3.4, as a
function of (k a_l, ε) and the operator … CH2.3, CH3.3 and CH4.3 confirm them in the full
FMM"); §9.1 (the norms and workloads); §2 constraints 2 and 9; S7's report
`spikes/truncation/SPIKE_REPORT.md` §3, §4, §7 and §8 (question 1, signed off 2026-10-29:
"CH3.3 confirms it in the full FMM"); CH3.1c's report §5 (S7's closed form leaves errors
3 × 10³ to 6 × 10⁶ below ε on low-frequency trees).

## Goal

Measure in the full FMM the smallest per-level degrees that meet ε on the reference
workloads, confirm S7's tables as upper bounds of them, and make a closed form fitted to
the full FMM the default rule of `Degrees::Tolerance`.

## Acceptance (proposal §10, verbatim)

S7's low-frequency tables confirmed in the full FMM.

Read here (signed off on the plan, 2026-11-01, "Decisions" below): "confirmed" means measured in the full FMM with
§9.1's norm. Where the full FMM needs fewer degrees than S7's per-pair worst case (CH3.1c:
it does, by orders of magnitude in the error), the calibrated degrees replace S7's closed
form as the default, as §3.5 intends ("calibration lowers the cost of the low-frequency
levels"); S7's closed form stays as the upper bound.

## Read first

- `fmm-exec/src/helmholtz/degree.rs` (the rule, `Degrees`), `fmm.rs` (the builder).
- `fmm-validate/src/helmholtz_accuracy.rs` and `examples/helmholtz_accuracy.rs` (CH3.1c's
  harness: problem, sharded oracle, errors, checksum), `helmholtz_workloads.rs`.
- `fmm-validate/results/helmholtz-ch3-1c.md` (the matrix and its degrees).
- S7: `spikes/truncation/SPIKE_REPORT.md`, `results/summary.md` (p₁, δ, p*).
- Laplace's C3.4: `docs/phase3/T12-calibration.md`, `fmm-validate/examples/calibrate.rs`.

## Changes

**The calibration** (`nd-fmm-validate`, new; no other crate depends on it):

- `nd_fmm_validate::helmholtz_calibration`: the search and the fit, independent of MPI and
  of the FMM so that it is unit-testable.
  - `smallest_degree(lo, hi, errors)`: the smallest p in lo..=hi from which on the error
    at p, p + 1 and p + 2 meets the target (as S7's search: errors are not monotone in p
    to the last digit), with `errors(p) -> [φ, ∇φ]` a closure;
  - the fit of the rule's two constants to the measured degrees (below).
- Example `helmholtz_calibrate` (release, any ranks, not registered for the weekly job):
  per workload, kD and ε, the problem and the oracle built once, then FMM runs at named
  per-level degrees (`Degrees::PerLevel`) through `helmholtz_accuracy::run`:
  - **plateau scan**: one common p on every level ≥ 2 (all levels low-frequency), from 1
    upward to S7's degree;
  - **level scan**: one level's p scanned, every other level at S7's closed form (whose
    errors CH3.1c measured 10²–10⁶ below ε), so that the level alone sets the error;
  - prints Markdown rows: case, level, k a_l, S7's degree, the smallest p meeting ε/2 and
    ε, the errors on the way, the evaluation time.

**The rule** (`nd-fmm-exec`, `helmholtz::degree` only; Helmholtz code, no Laplace path):

- `degree(k, a, ε)` becomes the calibrated closed form of the same shape as S7's,
  max(p_lo(ε), ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉) with x = Re(k) a, p_lo(ε) =
  ⌈α log₁₀(1/ε) + β⌉ and the constants α, β, C fitted (below); the module documentation
  gives the data and the range it was fitted on.
- `s7_degree(k, a, ε)` (new, public): S7's closed form unchanged, documented as the
  per-pair upper bound and used by the calibration's level scan and the "before" times.
- `Degrees` unchanged (`Tolerance` takes the new rule; `PerLevel` as before). No change
  to the builder, the operator, the tables, `nd-fmm-plan` or any Laplace crate.
- The `helmholtz_accuracy` example gets `--rule calibrated|s7` (default calibrated) for
  the before-and-after rows.
- Conventions: none.

**The fit.** p_lo(ε) is the largest plateau degree over the plateau cases at each ε, fitted
by α, β so that the formula is never below it at the measured ε and between them is
monotone; C is the largest (p − x)/((ln 1/ε)^{2/3} x^{1/3}) over the level scans with
x ≥ 8 (S7's crossing region starts at x ≈ 4–8). Each measured degree is the one meeting
**ε/2** (S7's margin of 2: a target sums several levels and V lists). If the plateau does
not fit α log₁₀(1/ε) + β to within one degree, a short per-ε table replaces the formula
(said in the report).

**Docs:** `fmm-exec/CLAUDE.md`, `fmm-validate/CLAUDE.md`; the report
`fmm-validate/results/helmholtz-ch3-3.md`.

## Tests that prove it

- `fmm-validate/src/helmholtz_calibration.rs` unit tests: `smallest_degree` on synthetic
  error curves (monotone; a non-monotone dip one degree below the threshold, which must
  not be taken; never reached → `None`); the fit reproduces constants from data made by
  a known rule, and is never below a data point.
- `fmm-exec/src/helmholtz/degree.rs` unit tests: the new rule never above `s7_degree` on
  a grid of x in [10⁻³, 300] and ε in [10⁻¹⁰, 10⁻³]; monotone non-increasing in ε and
  non-decreasing in x; the fitted values at named points; the doc examples updated.
- The acceptance, in the full FMM, against the sampled direct sum (CH3.1c's harness,
  relative ℓ2 of φ and ∇φ over 1,000 targets, f64, 1 rank × 6 threads on the M6):
  1. **S7 confirmed**: at every plateau and level scan, S7's degree meets ε/2 (so S7's
     tables are upper bounds in the full FMM), with the measured degree beside S7's p*
     at the same k a in the report's table.
  2. **The calibrated rule meets ε** on CH3.1c's whole matrix (W1 kD = 10, 30, 100, 500;
     W3 kD = 10, 30, 100; W4's four distributions at kD = 1, 10, 100; ε = 10⁻³, 10⁻⁶,
     10⁻⁸; real charges, 8 vectors, kD = 100 + 10i), and on held-out cases not used in
     the fit: W2 at kD = 30 and 100, W1 at kD = 50 and 150, W4 cube and Plummer at kD = 3
     and 30, ε = 10⁻⁴ and 10⁻⁷ on W1 kD = 100.
  3. Deterministic as before: checksums equal at 1 and 6 threads on 3 cases, and on 1 and
     3 ranks on one.
- `fmm-validate/tests/helmholtz_accuracy.rs` (the existing MPI test): its default-rule
  cases within ε, unchanged, plus a smoke run of the calibration core (a plateau scan at
  N ≈ 500, p ≤ 6) inside the same MPI-owning test.
- `fmm-exec`'s Helmholtz tests that use `Degrees::Tolerance` pass unchanged (they test
  against the direct sum within their bounds, or bit identities that hold for any degrees).

## Checks

`/verify` (format; Clippy default, examples, workspace all-targets; `cargo test` and
`--workspace`; `cargo doc`; the MPI list at 2 and 4 ranks, since `nd-fmm-exec` changes);
`cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`;
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`.

## Machine runs

- The M6 (release, one rank × 6 threads, BLAS threads 1, outside the sandbox): the scans,
  the fit, then the verification matrix and the held-out cases with the new rule, and
  before-and-after evaluation times on a few cases (indicative, the M6's ±25 % between
  launches; not a gate).
- W3 at kD = 100, ε = 10⁻⁸ (left out of CH3.1c for memory: about 21 GB at S7's p = 46):
  with the calibrated degrees on the M6 if it fits; else one Kathleen node in
  `singlenode` (CH3.1c sign-off 2's "or run on Kathleen in CH3.3"), under "Pending runs"
  if it has not finished.
- No locust run; no D9 run (Laplace is not touched).

## Constraints and risks

- Constraint 9: the rule stays a function of k, the domain and ε only.
- Constraint 2: the degrees are the same on every rank; determinism is re-checked.
- Constraint 1: no Laplace code or default changes, so no D9.
- Risk: a calibration on the reference workloads under-predicts on a geometry it has not
  seen (the corner clusters of S7 converge slowest). Mitigations: the margin ε/2, the
  held-out cases, and S7's rule kept public as the conservative choice.
- Risk: the scans cost time (W1 kD = 300 has 7.2 × 10⁵ points; kD = 500 is verification
  only, not scanned: its top level x = 108 lies beyond the scanned range, and the fit is
  extrapolated there, which the report says).
- Complex k keeps Re(k) a (provisional, as in CH3.1a); damping is CH5.2's, Yukawa CH2.3's.

## Pending runs

## Open questions

## Decisions

Taken 2026-11-01 on the plan, the user: "Calibrate, new default (Recommended)" (the
calibrated rule replaces S7's closed form as `Degrees::Tolerance`'s default, S7's kept
public as the upper bound); "Both branches (Recommended)" (the plateau p_lo(ε) and the
high-frequency constant C); "ε/2, verify ≤ ε (Recommended)" (each measured degree meets
ε/2; the rule meets ε on the matrix and the held-out cases).
