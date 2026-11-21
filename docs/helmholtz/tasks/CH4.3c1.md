# CH4.3c1 C₀ at large degrees, against a double-double reference

Started 2026-11-21 on `tbetcke/h-ch4-3c1`. Roadmap: CH4.3c1 "C₀ at large degrees, against a
double-double reference" (the first of CH4.3c1–CH4.3c2; CH4.3c split on 2026-11-21 on this
branch by the size rule of `/next-task`: two independent studies, one on pairs on the M6, one a
calibration on Kathleen). Proposal: §10 CH4.3 ("C_T and θ from S6 confirmed …"); §3.6 (the
bound E(l, c), C₀ = 200, "below 176 u where E is below 10⁻¹³"); §5.4 (the interpolated build);
§2 (constraints 1, 2, 9). Review 2026-11-19, M12 (`docs/reviews/2026-11-19-H1.md` §1.2).

## Goal
Measure the diagonal form's rounding at its floor, where E = C₀ u, at the FMM's degrees p ≥ 72
against a reference that resolves it (the same chain in double-double), and confirm C₀ = 200 or
raise it, so that the floor of the bound is measured and not inferred from f64 references.

## Acceptance (the roadmap's CH4.3c1, from CH4.3c's *Done when*)
"C₀ checked at p ≥ 72 against a double-double reference, or raised, with the glibc margin (at
p = 131–245 no f64 reference resolves the floor: CH4.2a's record, the diagonal form within
2.3·10⁻¹³, 5.8 E, of RCR on (3, 3, 3) at k a = 216, p = 245, the projection rounding more than
both); pair evidence and FMM evidence, both."

M12 as the review wrote it: "At k a = 54 the diagonal form errs 1.8–2.9·10⁻¹⁴ against the
projection, with E ≈ 2.7·10⁻¹⁴. That is up to 1.07 E, about 261 u against S6's 'below 176 u'.
Every reference there is f64, so C₀ = 200 is neither confirmed nor refuted at p ≥ 72. The ratio
against RCR is 1.994 of the limit of 2, in an ignored test. Proposed: CH4.3 checks C₀ at
p ≥ 72 against a double-double reference, or raises it, with the glibc margin rule."
Checked against the tree on 2026-11-21: `at_the_floor_against_the_projection` and
`m2l_at_the_degrees_of_kd_1000` (`fmm-ref/tests/helmholtz/diagonal.rs`) and CH4.3a's
`the_bound_on_the_fmms_levels` (`fmm-tables/tests/helmholtz/diagonal_transfer.rs`: k a = 108 up
to 3.29 E against RCR, read against the projection) are still the only floor evidence; every
reference in them is f64.

What "confirmed" means: the rule takes a class where E ≤ θ ε, and S6 fitted the constants so
that the measured rounding stays within E/θ = 2E (`THETA`, `bound`'s documentation). At the
floor E = C₀ u + (Hankel term) ≈ C₀ u, so C₀ = 200 holds where every class's rounding against
the double-double reference is at most 2E, about 400 u, at p ≥ 72; S6's own statement ("below
176 u") is reported beside it. If some class exceeds 2E, C₀ is raised to the smallest multiple
of 50 that covers the measured floor at 2E, and the proposal's §3.6 amended for sign-off.

The glibc margin (memory of CH1.2a/CH1.2c): a test bound on a floating-point accuracy leaves
about 10× over the M6's measurement, or scales with the conditioning. The tests here assert
2E (the rule's own contract, which has the factor θ = 1/2 built in) and print the measured
floor in units of u, and the non-ignored test (small p, runs in CI) asserts the reference's
agreement at a bound 10× above what the M6 measures.

FMM evidence: C₀ u = 2.2·10⁻¹⁴ enters the rule only through E ≤ θ ε; in D1's range
(ε ≥ 10⁻⁸, θ ε ≥ 5·10⁻⁹) the floor is 2·10⁵ times below the threshold, so a class's decision
depends on C₀ only for C₀ ≳ 2·10⁷. If C₀ stays, the FMM evidence is CH4.3a's (every W1 and W3
case 1.000–1.007× Rotation's, W1 at kD = 1000 at 10⁻⁶ and 10⁻⁸ included, job 653621). If C₀ is
raised, a test shows the classes taken on every level of the FMM's trees (k a 1.35–216,
ε = 10⁻³ … 10⁻⁸, both builds) identical under the old and the new constant, so the FMM's output
is unchanged bit for bit; that is the FMM evidence.

## Read first
- `fmm-ref/src/helmholtz/diagonal/{mod,grid,transfer,dd,bound}.rs`: the chain, its stored form
  (sⁿ, e^β, 4π ik e^{2β − Im(k)|t|}), the double-double scalar and what the grid keeps in it.
- `fmm-ref/tests/helmholtz/diagonal.rs`: `at_the_floor_against_the_projection`,
  `m2l_at_the_degrees_of_kd_1000`, the helpers (`multipole`, `evaluate`, `error`).
- `fmm-tables/tests/helmholtz/diagonal_transfer.rs`: `the_bound_on_the_fmms_levels` and
  `study_row` (the production chain through `tables_local`, both builds).
- `docs/helmholtz/tasks/CH1.3d1.md` (Results: the floor at k a = 54) and CH4.2a's record (k a 216).
- `spikes/diagonal/SPIKE_REPORT.md` §1.2, §2, §3 (the floor, "below 176 u").

## Changes
- `nd-fmm-ref`, `helmholtz::diagonal`:
  - `transfer.rs`: the sampling split into the double-double samples and their rounding, the
    rounded samples bit for bit as before (same operations, same order).
  - `dd.rs`: what the chain needs beyond what is there (a `ldexp`-style exact scaling by powers
    of two, a dd normalised Legendre column by the three-term recursion with a binary exponent
    per column, as `nd_fmm_math::orthonormal::legendre` keeps one, since P̄ₘᵐ ~ sᵐ leaves f64's
    range at m of a few hundred near the poles).
  - a new module `extended.rs` (private, its function re-exported): `m2l_extended`, the chain
    S → F, T F and F → R of `m2l` in double-double from the f64 stored multipole, returning the
    stored local as two f64 vectors (hi, lo), with every grid quantity, the θ-factors, T, the
    factor 4π ik e^{2β − Im(k)|t|} and sⁿ in double-double from the f64 inputs. In exact
    arithmetic it is the truncated M2L (the module's identity), so it is the rounding-free
    reference of `m2l`, of `rcr::m2l` and of the tables' production chain. O(p³) double-double
    operations per pair.
  - module documentation: the tested range and the floor at p ≥ 72.
- `nd-fmm-ref` tests (`tests/helmholtz/diagonal.rs`): the reference's tests (below) and the
  floor study at large degrees (ignored).
- `nd-fmm-tables` tests (`tests/helmholtz/diagonal_transfer.rs`): the production chain (both
  builds) against the reference at the FMM's levels with p ≥ 72 (ignored study); the floor
  reading of `study_row` (against the projection at p ≥ 72) replaced by the reference where the
  study's numbers show it resolves the floor.
- If C₀ moves: `BoundConstants::{F64, F64_INTERPOLATED}` (and the f32 one only if measured), the
  proposal §3.6 (for sign-off, amended in this PR only after the user's answer), the rustdoc that
  quotes 200, and a test of unchanged decisions (FMM evidence above).
- Report `fmm-validate/results/helmholtz-ch4-3c1.md`.
- No public API change outside `nd_fmm_ref::helmholtz::diagonal` (one new function); no
  convention change; no dependency.

## Tests that prove it
1. **The reference is the chain** (CI, debug): `m2l_extended` against `diagonal::m2l::<f64>` and
   `rcr::m2l` at p = 7–24 on the 34 classes, k a 2.8–16, real k and η = 0.1: within the f64
   chains' rounding (each f64 chain against the reference within its own 2E, the reference
   within 10× the M6's measured difference to `rcr::m2l` where E is the floor).
2. **The reference resolves what f64 cannot** (CI): the S → F → R identity in double-double (a
   zero transfer step replaced by the identity: the reference with T ≡ 1 and the factor removed
   gives back the multipole) to 10⁻²⁶ relative at p ≤ 16; `m2l_extended` evaluated in two ways
   (hi + lo against hi alone) differs by ≤ u relative (the lo part is a correction of size u).
3. **The legendre column in double-double** (unit, CI): against `legendre::normalised` in f64
   within 10⁻¹⁴ (scaled by n) at p ≤ 300 on Gauss–Legendre nodes near the poles, and finite with
   correct (zero) underflow where P̄ₙᵐ < 10⁻³⁰⁰.
4. **Transfer bit for bit** (CI): the existing `the_samples_are_refs_transfer_functions`,
   `both_builds_in_parts_by_ring` and the transfer tests pass unchanged (the rounded samples
   equal to before).
5. **The floor study** (ignored, release, ref): `diagonal::m2l::<f64>` and `rcr::m2l` against the
   reference on the 34 classes at k a = 27, 43, 54, 108 and 216 (the FMM's levels; p from the
   calibrated rule at 10⁻³, 10⁻⁶, 10⁻⁸, so p = 36–251), two placements (uniform and corner
   sources, targets in the target box); printed per (k a, p, class) the errors in u and over E;
   asserted: the diagonal form within 2E on every class (the rule's contract).
6. **The production chain at the floor** (ignored, release, tables): the tables' chain, direct
   and interpolated builds, against the reference at the same (k a, p) and classes; asserted
   within 2E; printed the largest error in u at E's floor per (k a, p, build). This replaces the
   projection reading of `the_bound_on_the_fmms_levels` at p ≥ 72 (the study reads the floor
   classes against the reference).
7. If C₀ moves: the decisions test of the FMM evidence (CI): the classes taken per (level k a, ε,
   build) identical at the old and new C₀ over the FMM's levels of W1 at kD = 100–1000.

## Checks
`/verify` (the branch changes `nd-fmm-ref`, tests of `nd-fmm-tables`, docs): fmt, Clippy (default,
workspace all-targets), rustdoc `-D warnings`, the default and workspace tests (opt-level 1 as CI
for the default), the MPI list at 2 and 4 ranks only if the verify skill asks for it for these
crates. The ignored studies 5 and 6 in release on the M6 (accuracy only, no timing).

## Machine runs
None: the study is accuracy only and runs on the M6 in release (minutes; ask before any timing,
none here). Kathleen and locust are not needed. The FMM evidence is CH4.3a's job 653621 if C₀
stays, a decisions test if it moves.

## Constraints and risks
- Constraint 1 (Laplace): not touched (`nd-fmm-ref`'s Helmholtz diagonal module and tests only).
- Constraint 9 (degrees from k, the domain and ε only): unaffected.
- Risk: the double-double Legendre recursion near the poles at p ≈ 250 (P̄ₘᵐ ~ sᵐ with
  s ≈ 5·10⁻³ underflows f64's exponent range); handled by a per-column binary exponent, tested.
- Risk: the reference must not share the f64 chain's rounding: every input it rounds is an f64
  input of the f64 chain too (the multipole, k, the frames, s); everything the f64 chain derives
  (nodes, weights, azimuths, θ-factors, T, the factor, powers of s) the reference derives in
  double-double.
- Risk: the cost at p = 251 (O(p³) dd operations per pair, about 10⁸): seconds per pair in
  release; the study runs one case per thread.
- Risk: the production chain's level call (`tables_local`) uses the tables' conversions (CH4.1's
  fast transforms), not ref's; both are measured.

## Pending runs
None.

## Results
- C₀ = 200 confirmed (report §2): at E's floor the diagonal form errs 25–149 u over k a 27–216,
  p 36–251, all three f64 paths (tables direct and interpolated, ref's) within 0.94 E; RCR
  290–3,345 u. C₀ stays, so test 7 (decisions unchanged) is not needed and the FMM evidence is
  CH4.3a's.
- The reference (report §1): identity 3.6·10⁻³¹, θ-factors 8.4·10⁻¹⁶ against f64, f64 diagonal
  within 0.57 E and RCR within 5.5·10⁻¹² of it at p ≤ 36.
- The floor readings read again (report §3): every pair above 2E against RCR lies within 0.24 E
  of the reference; CH4.3a's k a 1.69 finding confirmed as the form's own.
- As built: three public types (`ExtendedM2l` with `transfer` and `apply`, `ExtendedTransfer`,
  `ExtendedLocal` with `difference`) in place of one function `m2l_extended`, so that a class's
  transfer function serves both placements; the scaled Legendre column and the power-of-two
  scaling live in `extended.rs`, not `dd.rs`. Test 5 became the "ref" column of the tables' study
  (one study, four f64 paths against one reference); the study's ε are 10⁻³ and 10⁻⁸ above
  k a 100 (p moves little there) and 10⁻³, 10⁻⁶, 10⁻⁸ below.
- The reference's own accuracy at the study's largest degree: the identity at p = 251 to
  1.0·10⁻²⁹ (ignored unit test). The transfer samples bit for bit as on `main`: an FNV hash of
  `Transfer::new`'s samples taken on `main` (78a6f65), asserted on the branch (self-review note:
  the earlier tests compared two paths through the new code).
- Test 2's "hi + lo against hi alone" was dropped: `ExtendedLocal::difference` forms the
  difference in double-double, which the tests use, so a separate check adds nothing.

## Open questions
- (For the PR) whether CH4.3c2 (the degrees refit) should be a *Need* of CH4.3b2c, so that the
  form's default is measured on the final high-frequency degrees. Signed off 2026-11-21 on #246
  with question 1 (C₀ = 200 kept), the user: "Both questions as recommended."; the *Need* added.
