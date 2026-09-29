# Phase 1 / T2 — translation conventions (prerequisite of C1.2 and C1.3)

The design document writes the translation formulas "up to signs and the direction of
the shift vector" (Section 2.3). This task fixes them in docs/CONVENTIONS.md and checks
them numerically before any Rust implements them. It is the Phase 1 counterpart of
Phase 0 T2 and T3 together.

Read first: docs/CONVENTIONS.md (all); docs/design/laplace-fmm-plan.md §2.2–§2.4 and
§3.2; tools/fixtures/README.md, gen_harmonics.py, crosscheck_scipy.py; the doc comment
of `nd_fmm_math::harmonics::irregular_grad` (derivation style to follow).

Do:
- Derive M2M, L2L and M2L in the complex form of §3.3, for the scaled coefficients of
  §3.7, from an input frame (c, r) to an output frame (c', r'). For each operator,
  state:
  - the shift vector explicitly (c' − c or c − c');
  - the index ranges, and where terms vanish (Rⱼⁱ = Iⱼⁱ = 0 for |i| > j);
  - the radius factors (powers of r/r');
  - the convergence condition (M2L: the output sphere lies outside the input sphere).

  The irregular addition theorem that M2L needs follows from the separation identity
  (§3.4), with y replaced by −b and the ladder of `irregular_grad`. Derive it; do not
  copy it from the literature, whose normalisations differ.
- State the real-storage form (§3.6): how each complex sum is evaluated from the
  stored m ≥ 0 values, including the (−1)ᵐ conjugate symmetry for negative orders.
- State the coaxial special case, shift along +z, where only order-0 harmonics of the
  shift are nonzero, so each translation keeps m fixed. Give Rₖ⁰ and Iₖ⁰ on the
  positive z-axis in closed form.
- State how coefficient vectors rotate. Coefficients are conjugates of basis values,
  so the expectation is K Dⁿ K for multipoles and K S Dⁿ S⁻¹ K for locals in real
  storage, where K = diag(+1 for m ≥ 0, −1 for m < 0) is conjugation. Verify this
  numerically; do not assume it.
- State L2P and M2P with gradients in scaled form, including the 1/r and 1/r² factors.
- Write all of this as a new section §3.11 "Translation operators" in
  docs/CONVENTIONS.md, citing §3.3–§3.8 for everything it builds on. Leave §3.10 in
  place.
- Propose a change to §3.9. M2L evaluates irregular harmonics up to degree 2p at
  scaled shifts |c' − c|/r between 4 and 6√3 ≈ 10.4 (the V-list offsets
  {−3..3}³ \ {−1..1}³ with shift 2r · offset). So f64 must be tested for irregular
  harmonics up to degree 40 (p ≤ 20) at |v| in [4, 11].
- tools/fixtures/check_translations.py (mpmath at 40 digits, PEP 723 inline
  dependencies like the other scripts; reuse gen_harmonics.py's harmonics by import,
  not by copy). For seeded random frames and charges it checks:
  - M2M: P2M at the child, then M2M, equals P2M at the parent, to 1e-30.
  - L2L: evaluating a random local expansion after L2L equals evaluating it before, at
    points inside the output sphere, to 1e-30.
  - M2L: with input degree 60 and output degree 8, M2L after P2M equals P2L at the
    target frame within the input truncation bound.
  - Coaxial forms equal the general ones for shifts along +z, to 1e-30.
  - The rotation rule for coefficients, at n ≤ 8.

  Print the worst error per check; exit nonzero on any failure. Document it in
  tools/fixtures/README.md.
- Fixture set C in gen_harmonics.py: irregular values only (no gradients), 4 points,
  p = 40, |v| in [4, 11], seed 20261001.
  - Bump GENERATOR_VERSION. The values in sets A and B must not change; show this in
    the PR, e.g. by a diff that ignores the header.
  - Raise MAX_TOTAL_BYTES to 3.5 MB, and record the new limit and the reason in
    tools/fixtures/README.md.
- fmm-math/tests/harmonics/fixtures.rs: test set C in f64, per degree, relative, in
  the orthonormal weighting Nₘ/Sₘ of §3.8, to 1e-13. Report the measured worst error.
  If it fails, stop and report; do not change the recursion in this task.

Must pass:
- `uv run tools/fixtures/gen_harmonics.py --check` (byte-identical, under the new
  limit), `uv run tools/fixtures/crosscheck_scipy.py`,
  `uv run tools/fixtures/check_translations.py`.
- `cargo test -p nd-fmm-math`, `cargo clippy -p nd-fmm-math --all-targets -- -D warnings`.

The PR description must contain:
- a summary of the derivation and the measured agreement of every check;
- the questions for sign-off:
  1. the wording of §3.11;
  2. whether §3.10's rule should cover §3.11, since cached tables will depend on it;
  3. whether adding §3.11 bumps `CONVENTION_VERSION`. Recommend no bump: no existing
     section changes and nothing depends on translations yet. The decision is the
     reviewer's.
  4. the §3.9 range change.

Do not: change §3.1–§3.8; write any translation code in Rust; touch fmm-ref/.
If a derived formula contradicts §3.3–§3.8, stop and report instead of adjusting either.
