# Phase 0 / T5 — nd-fmm-math: rotation blocks for real solid harmonics (C0.3)

Read first: docs/CONVENTIONS.md §3.6 and §3.8; the T4 code.

Do:
- rotation::blocks(p, q: &[[T; 3]; 3], out) filling per-degree (2n+1)x(2n+1) blocks
  D^n(Q) as defined in §3.8 (contiguous, degree by degree);
  rotation::euler_zyz(alpha, beta, gamma).
- Suggested method: a stable recursion for real orthonormal spherical harmonics
  (e.g. Ivanic and Ruedenberg, with its later erratum), then D = N^-1 W N with N from
  §3.8. Any sign or ordering mismatch with the library basis is resolved by the tests.
- Helper converting a block for use on irregular harmonics (S D S^-1, §3.8).

Tests that define done:
- R_n(Qx) = D^n(Q) R_n(x) at random x and Q: relative 1e-13 for p <= 20, 1e-11 for p <= 30.
- Same for irregular harmonics with the S-conjugated blocks.
- Homomorphism D(Q1 Q2) = D(Q1) D(Q2); identity maps to identity.
- N D N^-1 orthogonal to 1e-13 (p <= 20).
- Rotation about z by alpha: each (+m, -m) pair rotates by angle m*alpha, rest zero.

Do not: implement coaxial translation or point-and-shoot operators (Phase 1).
