# Phase 0 / T4 — nd-fmm-math: RealScalar, Layout, solid harmonics (C0.2)

Read first: docs/CONVENTIONS.md (all), fmm-math/CLAUDE.md, fixtures from T3.

Do:
- trait RealScalar: num_traits::Float + FloatConst + Copy + Send + Sync + 'static,
  with from_f64/to_f64; impls for f32 and f64.
- struct Layout { p }: len() = (p+1)^2, idx(n, m), nm(i), iteration by degree.
- harmonics::regular(p, x, out) and harmonics::irregular(p, x, out) using the Cartesian
  recursions of §3.5 on real and imaginary parts; no trig, no allocation.
- harmonics::regular_grad and harmonics::irregular_grad writing three arrays in the
  same layout; regular via the ladder of §3.4; derive the irregular gradient and
  document the derivation in the doc comment.

Tests that define done:
- f64 matches fixture set A to relative 1e-13 (p <= 30); f32 matches set B to 1e-5.
- Separation identity at 200 random pairs with |y|/|x| <= 0.5, truncated at p = 30: 1e-12.
- Addition theorem (§3.4) for p <= 12 at random a, b: relative 1e-13.
- Harmonicity: discrete Laplacian by 4th-order finite differences below 1e-6 (f64).
- proptest: Layout idx/nm round-trip for p <= 40.

Do not: add translation operators; they belong to Phase 1.
