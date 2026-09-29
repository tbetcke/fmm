# Phase 1 / T5 — nd-fmm-ref: direct O(p⁴) M2M, L2L, M2L (C1.2)

Starts only after CONVENTIONS §3.11 (T2) is signed off.

Read first: docs/CONVENTIONS.md §3.6–§3.9 and §3.11; docs/design/laplace-fmm-plan.md
§2.3, §3.1 and §8.1; docs/phase1/README.md ("Design decisions"); the T3 and T4 code.

Do:
- Module `direct`:

  ```rust
  pub fn m2m<T: RealScalar>(p: usize, from: &Frame<T>, to: &Frame<T>, ws: &mut Workspace<T>,
                            multipole_in: &[T], multipole_out: &mut [T]);
  pub fn l2l<T: RealScalar>(/* same, local in, local out */);
  pub fn m2l<T: RealScalar>(p: usize, source: &Frame<T>, target: &Frame<T>, ws: &mut Workspace<T>,
                            multipole: &[T], local: &mut [T]);
  ```

  - Implement §3.11 literally, as O(p⁴) sums on real storage. A private two-field
    complex helper is fine; num-complex is not.
  - Evaluate the harmonics of the shift once per call into the workspace: R up to
    degree p for M2M and L2L, I up to degree 2p for M2L. Extend `Workspace`
    accordingly.
  - Accumulate (+=).
  - Doc comments cite the §3.11 equation, the shift vector used and the convergence
    condition. Do not assert separation; tests exercise near-limit cases.

Tests that define done. f64 unless stated. Coefficients are compared per degree,
relative, in the orthonormal weighting of §3.8: Nₘ for multipoles, Nₘ/Sₘ for locals.
- M2M exactness: P2M at a child frame, then M2M to the parent, equals P2M at the
  parent, for p ≤ 30, to 1e-13. Cover the 8 octant children (r' = 2r,
  c − c' = r · (±1, ±1, ±1)) and random frame pairs with other radius ratios.
- M2M composition: M2M(b → c) after M2M(a → b) equals M2M(a → c), to 1e-13.
- L2L exactness: for random local coefficients at a parent, L2P after L2L equals L2P
  of the original. Evaluate at points inside the child, relative to the sum of term
  magnitudes, 1e-13, p ≤ 30. (Translating a degree-p local polynomial is exact.) Also
  test composition, as for M2M.
- M2L against the direct sum: P2M, then M2L, then L2P, against `direct_sum` for p ≤ 20.
  - Geometry: same-level frames (r' = r) with shift 2r · d for all 316 offsets d in
    {−3..3}³ \ {−1..1}³; sources and targets uniform in their boxes.
  - Tolerance: an M2L truncation bound derived and documented in the test's doc
    comment.
  - The observed error must decay in p at least at the bound's rate.
- M2L structure:
  - M2L of a unit monopole (slot 0 only) equals P2L of a unit charge at the source
    centre, to 1e-13.
  - M2L at p is the leading block of M2L at p + 5 applied to the zero-padded input,
    to 1e-13.
- Special cases:
  - zero shift, where M2M and L2L reduce to the radius rescaling;
  - shifts along ±z and ±x;
  - p = 0 and p = 1 against values worked out by hand in the test's doc comment.
- Linearity; accumulation onto nonzero initial values.
- f32, p ≤ 8: matches f64 to 1e-5 in the same weighted measure.
- proptest for random frames and coefficients, 64 cases per property.

Must pass: `cargo test -p nd-fmm-ref` (keep its debug-mode run under a minute),
`cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`, `cargo doc -p nd-fmm-ref
--no-deps` without warnings, plus the root checks. Report the measured worst error of
every test; they go into design Section 7.

Do not: add rotation-based operators (T6), precomputed matrices, octant or offset
tables (Phase 2), or optimise the sums beyond clarity. If a test contradicts §3.11,
stop and report; do not adjust the formula to fit.
