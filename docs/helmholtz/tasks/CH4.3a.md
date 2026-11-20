# CH4.3a The bound's constants in the full FMM

Started 2026-11-20 on `tbetcke/h-ch4-3a`. Roadmap: CH4.3a "The bound's constants in the full
FMM" (the first of CH4.3a–CH4.3e; CH4.3 split on 2026-11-20 on this branch by the size rule of
`/next-task`). Proposal: §10 CH4.3 ("C_T and θ from S6 confirmed in the full FMM on W1 and W3");
§3.6 (the bound E(l, c), θ, "Complex k"); §5.4 (the interpolated build, C_T = 0.23); §9.1 (W1,
W3, the norms); §2 (constraints 1, 2, 6, 9).

## Goal
Confirm S6's constants of the diagonal form's rule (C_T = 0.15 direct, 0.23 interpolated,
θ = 1/2) in the full FMM on W1 and W3 at 10⁻³ to 10⁻⁸, measure how much margin the rule leaves,
and measure the damped classes against the bound's damping factor, with the pair evidence on the
same levels, so that CH4.3b can turn the form on by default on constants that hold.

## Acceptance (the roadmap's CH4.3a; proposal §10 CH4.3, its first item)
"S6's C_T (0.15 direct, 0.23 interpolated, the default build) and θ = 1/2 confirmed in the full
FMM on W1 (kD = 100 to 1000) and W3 (N = 10⁶, kD = 200 and 500) at 10⁻³ to 10⁻⁸, under Rotation
so that the rule reaches every level: each case within 1.1× Rotation's errors, and the margin
measured (the rule at s θ ε for growing s, until a case leaves 1.1×); with the pair evidence on
the same (k a, p): each class's error against RCR over E, both builds. Also: verified on the
constant density (kD ≤ 100) and the plane wave's trace (kD = 500) as well as on §9.1's charges.
Also: the damped classes measured in the FMM-weighted norm of CH1.3c (relative to the input's
size), per pair and in the FMM on damped W1, against the per-pair bound's factor
e^{Im(k)(|t| + 2a)}."

Proposal §10's CH4.3 row: "C_T and θ from S6 confirmed in the full FMM on W1 and W3; the time
crossover per class; the fallback of a whole level to RCR when the scratch budget binds". The
second and third items are CH4.3b's.

"Within 1.1× Rotation's errors" is read as CH4.2b1–CH4.2c2 read it: the relative ℓ2 errors of φ
and ∇φ against the direct sum at §9.1's sampled targets, each over the same run under Rotation
without the diagonal form (`DIAGONAL_RATIO`). Under Rotation the rule considers every level with
V pairs (CH4.2b1), the low-frequency ones included, where the nearest classes fail it; under the
defaults it would see only the levels Auto leaves on RCR (2–4), where every class passes with a
wide margin (CH4.2b2), so Rotation is the stricter test.

The margin: `HelmholtzDiagonal::Tolerance(ε′)` runs the rule E ≤ θ ε′ in an FMM of tolerance ε,
so ε′ = s ε is the rule at θ′ = s θ (equivalently C_T′ = C_T / s, up to C₀). s = 1, 4, 16, 64,
256, 1024 (θ′ from 1/2 to 512): the smallest s at which some case leaves 1.1× is the measured
margin; whether s is limited by C_T (the slope) or the floor is read from the classes that enter.

## Read first
- `fmm-exec/src/helmholtz/operator_diagonal.rs` (the rule), `fmm-tables/src/helmholtz/diagonal/
  transfer.rs` (`class_bounds`, `bound_constants`), `nd_fmm_ref::helmholtz::diagonal::bound`.
- `spikes/diagonal/SPIKE_REPORT.md` §3, §4, question 7 (the damping factor).
- `fmm-tables/tests/helmholtz/diagonal_transfer.rs` (`against`, the floor reading).
- `fmm-validate/examples/helmholtz_accuracy.rs` (`--diagonal`, `--kim`, `--workload w3:<n>`),
  `fmm-validate/src/helmholtz_accuracy.rs` (`Density`, `Case::density`).
- CH4.2b2's and CH4.2c2's reports (the W1 baselines and the classes per level).

## Changes
- `nd-fmm-tables`, tests only: `tests/helmholtz/diagonal_transfer.rs`, an ignored study
  `the_bound_on_the_fmms_levels` (release): the 34 classes at the (k a, p) of the FMM's levels
  (k a from 1.7 to 108, p of the calibrated rule at 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁸, restated in the test
  from `nd_fmm_exec::helmholtz::DegreeRule::CALIBRATED`: tables cannot depend on exec), real k
  and η = Im k / Re k = 0.1 and 1, both builds: measured/E per class against RCR (`against`'s
  measure and floor reading), and for damped k also the input-relative measure (below).
- `nd-fmm-validate`: the `helmholtz_accuracy` example's `--density random|planewave|constant`
  (`Case::density`, already in the library since CH4.0b; complex charges for the plane wave).
- `tools/helmholtz/ch4-3a-runs.sh` (the Kathleen stages) and `ch4-3a-tables.py` (the report's
  tables); the report `fmm-validate/results/helmholtz-ch4-3a.md`.
- No library code, no default, no convention, no dependency, no `nd-fmm-plan` interface, no
  Laplace file.

The input-relative measure (CH1.3c's FMM weighting divides by the larger of the output's and the
input's weighted size; for a damped pair the output is e^{−Im(k) d} smaller than its input): the
pair's error at the target points over the field of the same charges with k replaced by Re k
(the direct sum, undamped), φ and ∇φ as `error` takes them. In the FMM a target's field is
dominated by its near boxes, of the undamped scale, so a damped pair's error matters on that
scale. Reported beside the per-pair relative measure, over E with and without the damping
factor.

## Tests that prove it
- Pair (`the_bound_on_the_fmms_levels`, ignored, release, one thread): for real k every class at
  every (k a, ε) within 2E against RCR (the CH4.2a floor reading at p ≥ 72: against ref's
  projection), both builds, asserted; the damped rows' ratios printed, the per-pair relative
  measure within 2E asserted (S6's fit); at η = 0 the undamped direct field against the RCR
  chain's within 10⁻⁶ (the units of the input-relative measure checked).
- FMM (Kathleen, below): every case at s = 1 within 1.1× Rotation's errors, φ and ∇φ; the
  checksum of the rule's run at 1 × 40 the same in both builds' launches where the classes agree.
- `cargo test -p nd-fmm-validate` (the example parses; `tests/helmholtz_accuracy.rs` unchanged).

## Checks
`/verify`: fmt; Clippy (default, examples, workspace all-targets); rustdoc `-D warnings`; the
default and workspace tests at opt-level 1; the tables' new ignored test in release. No MPI list
change (no exec or plan change); the MPI list at 2 and 4 ranks still run by `/verify`'s rule if it
asks. `cargo deny` not needed (no manifest).

## Machine runs
Kathleen, `singlenode`, 1 × 40 (one rank, every core; release `helmholtz_accuracy`, BLAS
threads 1, §9.1's complex charges and 1,000 samples, `--near-v cost --near-wx cost --levels`,
one evaluation), `--m2l rotation --diagonal off,rule,<s ε>…` with the default build
(interpolated) and, on W1 at kD = 500, `--diagonal-build direct,interpolated`:
- `w1`: W1 at kD = 100, 200, 500 at ε = 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁸, s = 1, 4, 16, 64, 256, 1024;
- `w1-large`: W1 at kD = 1000 at 10⁻⁶ and 10⁻⁸, s = 1, 16, 256;
- `w3`: W3 at N = 10⁶, kD = 200 and 500, the four ε, s as `w1`;
- `densities`: the constant density on W1 at kD = 100 and the plane wave's trace at kD = 500,
  the four ε, s = 1, 16, 256;
- `damped`: W1 at kD = 100 + 10i, 100 + 100i and 500 + 50i, the four ε (η = 1: 10⁻³, 10⁻⁶),
  s as `w1`.
What they decide: the acceptance (s = 1 within 1.1×); the margin per case and where it binds (the
levels and classes that enter as s grows); whether the damping factor's over-prediction shows in
the FMM (a damped case keeps 1.1× to a far larger s than a real one); the questions for sign-off
(keep θ = 1/2 and both C_T; whether CH4.3b or a later task should loosen the damped factor). No
assert on a time.

Wider scope (the `/next-task` rule): no bound, default or range moves here. The cases earlier
sign-offs left out that this task brings in: ε = 10⁻³ and 10⁻⁴ (CH4.2b2 and CH4.2c2 ran 10⁻⁶
and 10⁻⁸ only), kD = 100 and 200, W3, the smooth densities and damped W1 (none ran the diagonal
form); k a = 216 (W1 at kD = 1000's level 2, p = 245) in the FMM only, its pairs being CH4.3c's
(C₀ at large degrees, a double-double reference).

## Constraints and risks
- Constraint 1: no Laplace path (a tables test, a harness flag, scripts).
- Constraints 2 and 9: no library change; the runs at s > 1 are named tolerances of the rule.
- Constraint 6: trusted paths: RCR and ref's projection (pairs), Rotation's FMM and the direct
  sum (FMM).
- Risk: the low-frequency levels under Rotation at s ≥ 64 take classes whose E is far above ε;
  the errors may be large or overflow (a finding, not a failure: the margin's end).
- Risk: the Kathleen job's length (about 150 launches; the kD = 1000 cases about 2 min each):
  two jobs if one would exceed the `singlenode` limit.

## Pending runs
<filled in step 6>

## Open questions
<filled when they arise>
