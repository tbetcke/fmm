# CH1.2c Coaxial coefficients

Started 2026-10-30 on `tbetcke/h-ch1-2c`. Roadmap: "CH1.2c Coaxial coefficients" (H1; the
third part of CH1.2, fourth of the priority list). Proposal: §10 CH1.2; §3.1, §5.2, §5.3,
§6.1 (placement, D2: A), §8 R2; CONVENTIONS §3.15 ("Coaxial translations", "Precision and
range"); S2 (`spikes/coaxial/SPIKE_REPORT.md`) and S7 (`spikes/truncation/SPIKE_REPORT.md`
§2.4 and §8 question 3).

## Goal

Put the coaxial coefficients (R|R) = (S|S) and (S|R) of CONVENTIONS §3.15 into
`nd-fmm-math` as a module generic over `T: RealScalar`: S2's recursions in their triangle
and S2's projections, chosen by §3.15's rule, tested against S2's mpmath fixture and the
identities of §3.15.

## Acceptance (proposal §10 CH1.2, verbatim, for the whole of CH1.2)

> the identities of §3.15; the reality of the M2M and L2L coefficients for real k

CH1.2c's part (roadmap): against an mpmath fixture (S2's); the M2M and L2L coefficients
real for real k; i^{n−n′} (E|F) real for Yukawa; the symmetries; the rectangular
truncation a block of the square table. §3.15's generator rule: the recursion where
Re(k) a_l ≤ 40 and the projection above, for (S|R) and (S|S); (R|R) by the recursion
where it is stable, every table checked in the weights of both directions, an
inner-sphere projection tested for the rest (S7, signed off 2026-10-29).

## Read first

- CONVENTIONS §3.15 "Coaxial translations", "Precision and range", "Verification".
- `spikes/coaxial/`: `src/coaxial.rs` (recursions, the halved `Table`), `src/projection.rs`
  (the three projections), `src/metric.rs` (the FMM-weighted error), `src/config.rs`,
  `tests/accuracy.rs` (tests 1–6), `tools/reference.py` (the fixture), SPIKE_REPORT §2,
  §4–§6, §8.
- `spikes/truncation/src/chain.rs` (`shift_table`) and SPIKE_REPORT §2.3, §2.4: the L2L
  failure of the outer projection. S7 projected a square dense table and read L2L's
  entries (target degree low, source degree high) from the block computed in that
  orientation, i.e. projected high-degree waves on the outer sphere.
- `fmm-math/src/radial.rs` (start values), `fmm-math/src/orthonormal/legendre.rs`
  (`Columns`, the overflow-free P̄ₙᵐ for the projection), `fmm-math/CLAUDE.md`.

## Changes

`nd-fmm-math` only, plus the fixture generator:

- `coaxial` (new, public), generic over `T`, with `k: Complex<T>`. As built (the API was
  planned as a public `Coefficients` enum, a `Translation` enum and `for_translation`;
  named constructors replaced them during implementation):
  - `CoaxialTable<T>`: the halved table of §3.15, per order m the triangle n′ ≥ n with
    rows to max(p_src, p_tgt) and columns to min(p_src, p_tgt); `get(m, n_tgt, n_src)`
    reads either orientation through the index symmetry, so one stored table serves M2M
    (p_child → p_parent) and L2L (p_parent → p_child); `reversed` exchanges the degrees
    without recomputing. Stored normalisation of §3.15: (R|R) times e^{−Im z}, (S|R)
    times e^{Im z} s^{n+n′}, z = k t.
  - The recursion: `regular(z, p_src, p_tgt)` and `singular_regular(z, s, p_src, p_tgt)`
    (start column from `radial`, evaluated to the square table's length so that a
    rectangular table is a block of the square one bit for bit; both recursions over the
    triangle closed under them).
  - The projections: `projection_singular_regular` ((S|R) on ρ < t, least squares from
    value and derivative, S2), `projection_regular_outer` ((S|S) on ρ > t, hₙ onto hₙ,
    S2) and `projection_regular_inner` ((R|R) on any ρ, jₙ onto jₙ by least squares; S2's
    `regular_inner` extended to complex k with the exponential factors chosen so that no
    applied exponent is positive). Each projection fills the triangle from the half it
    computes: the outer one from target degree ≥ source degree (M2M's orientation), the
    inner ones from target ≤ source; the other half is read through the symmetry. The
    Legendre columns come from `orthonormal::legendre::Columns`.
  - The rule of §3.15: `uses_recursion(k, a)` with `RECURSION_LIMIT = 40` on Re(k) a, and
    `m2l(k, t, a, s, p_src, p_tgt)`, `m2m(k, t, p_child, p_parent)` and
    `l2l(k, t, p_parent, p_child)` (t = a_child, the parent's circumradius 2t), with the
    radii of S2 (M2L: ρ = a, the target box; M2M: ρ = 2 w_parent = 8t/√3; L2L: ρ = t, the
    child's circumradius) and N = max(p) + 40 nodes (+ ⌈|k| t⌉ for L2L). L2L above the
    threshold uses the inner projection, as signed off; test 8 found the M2M table read
    reversed accurate too, which is open question 1.
  - Module documentation: the definitions, the normalisation, the rule, the tested ranges
    (CONVENTIONS §3.15 asks `nd-fmm-math` to state them), and the range of the (S|R)
    start column (s^l hₗ at l = p_src + p_tgt leaves f64 near l ≈ 175 when s = |k| a < 1,
    S7 §2.3; the builder panics on a non-finite start value).
- `lib.rs` documentation, `fmm-math/CLAUDE.md`.
- Fixture: `tools/fixtures/gen_helmholtz_coaxial.py` (S2's `reference.py`, the same 120
  sets and entries value for value, with the fmm-math header (`convention_version`) and
  each set's k, t, a and kind as fields, so the tests need not parse names) writes
  `fmm-math/fixtures/helmholtz_coaxial.json`; `tools/fixtures/README.md`.
- Report: `fmm-validate/results/helmholtz-ch1-2c.md` (the L2L measurement, the threshold
  sweep, times per table), with the times from `fmm-math/examples/coaxial_time.rs`.

No Laplace module changes; no table format; no convention; no dependency.

Not in scope: Yukawa's real tables i^{n−n′} (E|F) as a stored type (the operators',
CH2.1; here the test shows they are real); applying a table to coefficients (CH1.3 and
the operators); a BLAS projection.

## Tests that prove it

In `fmm-math/tests/coaxial/`, f64 unless stated, one thread. "Weighted" is S2's
FMM-weighted error (SPIKE_REPORT §4.1), ported to the tests' support module: M2L weights
uₙ u_{n′}, uₙ = Jₙ(a) √(2n + 1)/sⁿ; shift weights Jₙ(a_child) √(2n + 1) on the child
degree and |hₙ′(k · 2w_parent)| √(2n′ + 1) on the parent degree, the same for M2M and
L2L (the table is read transposed; SPIKE_REPORT §4.1).

1. **Fixture** (S2's 120 sets, 21,809 entries): the start column against the fixture's
   radial samples, 1e-13 relative; every set's table by `for_translation` (the rule) and
   by the recursion where S2 found it stable, sampled weighted error normalised by the
   full table ≤ 1e-11; per-entry relative error ≤ 1e-13 for the recursion at k a ≤ 3.
   Sets above p = 100 in an ignored release test.
2. **The definition, pointwise** (independent of the fixture): Fₙᵐ(r + t ẑ) =
   Σ_{n′} Eᵐ_{n′n} G_{n′}ᵐ(r) for every slot ±m at n ≤ 6, n′ summed to convergence, at
   points with |r| < t ((S|R), (R|R)) and |r| > t ((S|S) = (R|R)), real, complex and
   imaginary k, from `radial` and `orthonormal::harmonics`; 1e-12 relative to the
   largest term. The reflection: F(r − t ẑ) = Σ (−1)^{n+n′} E G(r), same bound.
3. **Real k** (M2M and L2L coefficients real): the recursion's (R|R) has every imaginary
   part exactly 0; the outer projection's (S|S) imaginary parts ≤ 1e-11 in the shift
   weights at k a = 100 and 300; the inner projection's (R|R), the same in both
   orientations; Re (S|R) = (R|R) in the M2L weights by the recursion (k a ≤ 30, all 15
   distances), 1e-11.
4. **Yukawa**: i^{n−n′} Êᵐ_{n′n} has imaginary part exactly 0 for (R|R) and (S|R) by the
   recursion at κ a = 1, 30 and 300 (the rule uses the recursion for every Yukawa level),
   and ≤ 1e-30 relative in the fixture.
5. **Symmetries**: the triangle read in the other orientation equals (−1)^{n+n′} times the
   entry (structural, also against the fixture's upper-triangle samples in test 1); one
   coefficient for ±m (test 2); the reflection (test 2).
6. **Rectangular truncation**: p_src → p_tgt in both orientations equals the block of the
   square table at max(p_src, p_tgt), bit for bit, for the recursion and for each
   projection (same nodes).
7. **The threshold** (S2 test 6, ignored, release): at k a = 40, every kind of k, M2M and
   all 15 V-list distances, the recursion against the projection over whole tables,
   weighted ≤ 1e-11.
8. **L2L** (ignored, release; decides the L2L rule): the shift table read in L2L's
   orientation, over whole tables in the shift weights, at k a ∈ {40, 64, 100, 128, 200,
   300} (parent), real, Im k = 0.1 Re k, Im k = Re k and imaginary k: the recursion, the
   outer projection (M2M orientation) and the inner projection (L2L orientation) against
   each other. Accepted for a method where it agrees with an independent one within
   1e-11; where no two agree, the case is reported and becomes an open question.
9. **f32**: the recursion in f32 against f64 at k a ≤ 3, p ≤ 8: 1e-5 relative to the
   table's largest entry per degree.
10. **Panics**: s outside (0, 1], t ≤ 0, Im k < 0, k = 0, a projection radius on the
    wrong side of t, a non-finite start value.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
default and workspace tests), plus `cargo test -p nd-fmm-math --release -- --ignored`,
and the generator's `--check`. No MPI crate changes, no manifest change.

## Machine runs

None. The L2L sweep and times run on the M3 Max, release, one thread.

## Constraints and risks

- Constraint 1: no Laplace code path changes (a new module; `Columns` is reused as is).
- Constraint 5: the tables are built in f64 by the caller and rounded when stored; the
  module is generic so that f32 can be tested.
- Constraint 9: the rule uses k, the geometry and the degrees only.
- Risk: the L2L measurement finds no method accurate for weakly damped k at high k a in
  f64 (S7 needed double-double). Then the rule's L2L branch stays the inner projection
  as signed off, with the measured range stated, and the gap is an open question.
- Risk: the fixture's samples miss the recursion's instability (S2 §4.3); tests 7 and 8
  measure whole tables.

## Pending runs

None.

## Open questions

Both are in the report (`fmm-validate/results/helmholtz-ch1-2c.md` §5) and in the PR;
neither blocks the acceptance, and the code follows the signed-off rule.

1. One shift table for M2M and L2L above the threshold. Test 8 shows the outer
   projection computed in M2M's orientation and read reversed within 6.8e-13 of the
   inner projection in L2L's weights, at every k a from 40 to 300 and every kind of k.
   S7's failure came from projecting in L2L's orientation. Recommended: allow it (one
   build, half the shift memory), amending §3.15's generator text; the alternative is the
   signed-off rule, which the code implements.
2. §3.15 calls the (S|R) projection "on an outer sphere"; S2's is on the target box's
   sphere ρ = a < t. Recommended: correct the wording when §3.15 next changes; no version
   bump.
