# Phase 1 / T3 — nd-fmm-ref: P2M, P2L, L2P, M2P (C1.1)

Read first: docs/CONVENTIONS.md §3.1, §3.6, §3.7, §3.9; docs/design/laplace-fmm-plan.md
§2.2, §2.4, §2.5; docs/phase1/README.md ("Design decisions"); fmm-ref/CLAUDE.md; the
docs of `nd_fmm_math::harmonics`.

Do:
- `Workspace<T>`: owns every temporary the operators need. `Workspace::new(p)`
  allocates for degrees up to p. Operators never allocate, and panic if the workspace
  is too small for their p. T5 and T6 extend it.
- Module `leaf`, with this shape (argument order may change, but keep it the same for
  all four):

  ```rust
  pub fn p2m<T: RealScalar>(p: usize, frame: &Frame<T>, sources: &[[T; 3]], charges: &[T],
                            ws: &mut Workspace<T>, multipole: &mut [T]);
  pub fn p2l<T: RealScalar>(p: usize, frame: &Frame<T>, sources: &[[T; 3]], charges: &[T],
                            ws: &mut Workspace<T>, local: &mut [T]);
  pub fn l2p<T: RealScalar>(p: usize, frame: &Frame<T>, local: &[T], targets: &[[T; 3]],
                            ws: &mut Workspace<T>, potential: &mut [T],
                            gradient: Option<&mut [[T; 3]]>);
  pub fn m2p<T: RealScalar>(/* as l2p, with a multipole */);
  ```

  - P2M and P2L form M̃ and L̃ exactly as in §3.7, conjugation included, in real
    storage (§3.6).
  - L2P and M2P evaluate φ with the doubling rule of §3.6 and the 1/r factor of §3.7.
    Gradients carry 1/r². M2P's gradient uses `irregular_grad`, which forms degree
    p + 1 internally.
  - Everything accumulates (+=); there is no 1/(4π).
- Doc comments cite the equations used. Each states the range in which the operator
  is tested (§3.9: P2M sources at |u| ≤ √3, P2L sources at |u| ≥ 2) and that behaviour
  outside it is not guaranteed.

Tests that define done (f64 unless stated). Use frames with radius in
{1, 0.37, 2⁻¹⁶} and centres away from the origin, so that the scaling is exercised.
- Storage: P2M of one unit charge at y equals conj(R(u)) in real storage, i.e. R(u)
  with the −m slots negated, bit for bit. The same for P2L with I(u).
- Same-degree oracle: P2M then M2P, and P2L then L2P, reproduce the truncated
  Legendre series Σₙ≤ₚ q aⁿ / dⁿ⁺¹ Pₙ(cos γ), for p ≤ 30. Here a < d are the distances
  of the nearer and farther point from the centre. The series is computed in the test
  from the Legendre recursion. Tolerance 1e-13, relative to the sum of term magnitudes.
  Gradients are compared with the gradient of the same series (from Pₙ and Pₙ′) at
  1e-12.
- Truncation bound: sources within √3 r of the centre and targets at distance
  ≥ (4 − √3) r, the standard one-box separation of design §2.5. Then
  |φ − φₚ| ≤ Σ|qⱼ| (a/d)ᵖ⁺¹ / (d − a) for p = 0..=30, against the exact 1/|x − y|,
  with a rounding floor of 1e-14 times the potential magnitude. The same holds with
  the roles swapped for P2L and L2P. For gradients, derive a bound, document it in the
  test's doc comment and check it.
- Linearity: the operator applied to a set of sources equals the sum over subsets;
  outputs accumulate onto nonzero initial values.
- f32, p ≤ 8: each operator matches f64 to 1e-5, measured as in the corresponding f64
  test.
- proptest for random frames, points and charges, 64 cases per property, to keep CI
  time low.

Must pass: `cargo test -p nd-fmm-ref`,
`cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`, `cargo doc -p nd-fmm-ref
--no-deps` without warnings, plus the root checks. Report the measured worst errors.

Do not: add translations (T5, T6) or P2P (T4); depend on nd-fmm-validate.
