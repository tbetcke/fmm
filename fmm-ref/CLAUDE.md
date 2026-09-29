# nd-fmm-ref

Purpose: f64 reference operators (leaf operators, direct O(p⁴) and rotation O(p³)
translations), P2P and the direct-sum oracle that every fast path in later phases is
tested against.
Phase and components: Phase 1, C1.1–C1.4 (tasks T1 and T3–T6 in docs/phase1/).

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  Translations implement §3.11; leaf operators implement §3.6 and §3.7.
- Coefficients are always the scaled ones of §3.7. Operators take the input and output
  `Frame` (centre, radius) explicitly, never a bare shift vector.
- Every operator accumulates (+=) into its output. No 1/(4π) anywhere (§3.1).
- Generic over T: RealScalar; no allocation in operators: temporaries come from a
  caller-owned `Workspace`.
- `direct` and `rotation` keep identical signatures.
- This crate is the oracle: clarity over speed. Optimisation belongs in nd-fmm-tables,
  nd-fmm-kernels and nd-fmm-exec.
- Tests name their error measure (docs/phase1/README.md, "Error measures").
- Before finishing: `cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`
  and `cargo test -p nd-fmm-ref` must pass.

## Allowed dependencies
nd-fmm-math, num-traits; dev-dependencies: proptest, approx.
No MPI, no octree, no CubeCL, and no dependency on nd-fmm-validate (not even as a
dev-dependency). Anything else needs a note in the PR.

## Test oracle
Exact 1/|x − y| and its gradient; the truncated Legendre series for same-degree checks;
exactness identities (M2M after P2M equals P2M at the parent, L2L exact on local
polynomials, composition); `direct` checks `rotation`; `direct_sum` checks every chain
within its truncation bound.
