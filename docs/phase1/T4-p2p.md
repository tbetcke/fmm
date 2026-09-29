# Phase 1 / T4 — nd-fmm-ref: P2P and the direct-sum oracle (C1.4)

Can run in parallel with T3 once T1 is merged.

Read first: docs/CONVENTIONS.md §3.1; docs/design/laplace-fmm-plan.md §2.5 and §8;
fmm-ref/CLAUDE.md.

Two functions with different jobs:
- `p2p` is the near-field operator, generic over `T: RealScalar`. Phase 4's P2P
  kernel is checked against it.
- `direct_sum` is the f64 oracle that end-to-end tests in every later phase compare
  against.

Do:
- Module `p2p`:

  ```rust
  pub fn p2p<T: RealScalar>(sources: &[[T; 3]], charges: &[T], targets: &[[T; 3]],
                            potential: &mut [T], gradient: Option<&mut [[T; 3]]>);
  pub fn direct_sum(sources: &[[f64; 3]], charges: &[f64], targets: &[[f64; 3]],
                    potential: &mut [f64], gradient: Option<&mut [[f64; 3]]>);
  ```

  - Both accumulate Σⱼ qⱼ / |xᵢ − yⱼ| and its gradient with respect to xᵢ; there is
    no 1/(4π) (§3.1).
  - A pair with xᵢ == yⱼ exactly contributes nothing. This is how self-interaction is
    excluded when targets and sources coincide. Nearly coincident pairs are not
    skipped. The doc comment says both.
  - `p2p` sums over sources in input order, with no reordering or compensation.
  - `direct_sum` uses compensated (Neumaier) summation per target. It stays serial
    and simple; a rayon variant waits until a later phase asks for it.
- Doc comments give the formulas, the exclusion rule and the summation method.

Tests that define done:
- Brute force: on small sets (up to 64 points), `p2p` in f64 equals a naive double
  loop in the test bit for bit. `direct_sum` agrees with it to 1e-15, relative to
  Σⱼ |qⱼ| / |xᵢ − yⱼ|.
- Cancellation: charges summing to nearly zero. `direct_sum` stays within 1e-15 of a
  reference computed in the test with double-double or exact-rational accumulation
  (choose one and document it), relative to the sum of term magnitudes.
- Coincident points: targets equal to sources (the same slice), and duplicated
  sources. Results are finite and equal the sum over the non-coincident pairs.
- Gradient: equals the analytic −Σ qⱼ (xᵢ − yⱼ) / |xᵢ − yⱼ|³ to 1e-15 relative, and
  agrees with central differences of the potential to 1e-7.
- Symmetry: a unit charge at y seen at x equals one at x seen at y, bit for bit.
- Accumulation onto nonzero initial values; empty sources and empty targets.
- f32: `p2p` matches `direct_sum` to 1e-6 relative to the sum of term magnitudes.

Must pass: `cargo test -p nd-fmm-ref`,
`cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`, `cargo doc -p nd-fmm-ref
--no-deps` without warnings, plus the root checks.

Do not: add tiling, SIMD or parallelism; touch the leaf or translation operators.
