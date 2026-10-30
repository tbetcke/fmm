# nd-fmm-ref

Purpose: f64 reference operators (leaf operators, direct O(p⁴) and rotation O(p³)
translations), P2P and the direct-sum oracle that every fast path in later phases is
tested against.
Phase and components: Phase 1, C1.1–C1.4 (tasks T1 and T3–T6 in docs/phase1/); Phase 6
T5 (C6.3): `leaf::{p2m, p2l, l2p, m2p}_many`, the leaf operators of several charge
vectors.
Phase 6 T8 (C6.4): the dipole operators `leaf::{p2m_dipoles, p2l_dipoles}`,
`p2p::{p2p_dipoles, direct_sum_dipoles}` (CONVENTIONS §3.7, "Dipole sources"), checked in
`tests/dipoles/` against `fixtures/dipoles.json` (tools/fixtures/gen_dipoles.py) and
central differences of the charge operators.
Helmholtz CH1.3a: `helmholtz::p2p::{p2p, direct_sum, p2p_yukawa, direct_sum_yukawa}`, the
Helmholtz-family P2P and direct-sum oracle (CONVENTIONS §3.15, e^{ikr}/r for complex k and
complex charges, Yukawa's e^{−κr}/r for real charges), checked in `tests/helmholtz/`
against `fixtures/helmholtz_fmm3d.json` (FMM3D's `h3ddirectcg`, Helmholtz CH1.4), Laplace's
`p2p::direct_sum` as k → 0, and the complex kernel at k = iκ (bit for bit).
Helmholtz CH1.3b: `helmholtz::{Level, Workspace}` and `helmholtz::leaf::{p2m, p2l, l2p, m2p}`
(complex k, complex charges, two real planes per expansion) and their `_yukawa` forms
(k = iκ, real charges, real expansions), in the stored form of §3.15 (the level's power
scale and damping), checked in `tests/helmholtz/leaf_*.rs` against the direct sum, closed
forms, Laplace's leaf operators under the diagonal map of proposal §3.4, and the complex
operators at k = iκ.
Helmholtz CH1.3c: `helmholtz::rcr::{m2m, l2l, m2l}` (rotation, coaxial translation with the
tables of `nd_fmm_math::coaxial`, rotation back; the tables from `rcr::{m2m_table,
l2l_table, m2l_table}`) and `helmholtz::direct::{m2m, l2l, m2l}` (projection of the leaf
evaluators' fields on a sphere), with rectangular degrees and `_yukawa` forms, checked in
`tests/helmholtz/translation_*.rs`: RCR against direct in the FMM-weighted measure, the
chains against the direct sum, the limit k → 0 against Laplace's `rotation`, and Yukawa
two ways. `direct` and `rcr` keep identical signatures apart from RCR's table.

## Rules
- Read docs/CONVENTIONS.md before changing any formula; never change a convention here.
  Translations implement §3.11; leaf operators implement §3.6 and §3.7. The Helmholtz
  family (`helmholtz`) implements §3.15.
- Coefficients are always the scaled ones of §3.7 (the stored ones of §3.15 for
  `helmholtz`, whose operators also take the `helmholtz::Level`). Operators take the input
  and output `Frame` (centre, radius) explicitly, never a bare shift vector.
- Every operator accumulates (+=) into its output. No 1/(4π) anywhere (§3.1).
- Generic over T: RealScalar; no allocation in operators: temporaries come from a
  caller-owned `Workspace`.
- `direct` and `rotation` keep identical signatures.
- The `_many` leaf operators (Phase 6 T5) form each point's harmonics once and give every
  vector exactly the single-vector operator's operations in its order; the single-vector
  operators are their m = 1 case with their own panic messages. Vector v of the result is
  the single-vector operator on vector v bit for bit (`tests/leaf/many.rs`).
- This crate is the oracle: clarity over speed. Optimisation belongs in nd-fmm-tables,
  nd-fmm-kernels, nd-fmm-simd and nd-fmm-exec.
- Tests name their error measure (docs/phase1/README.md, "Error measures").
- Before finishing: `cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`
  and `cargo test -p nd-fmm-ref` must pass.

## Allowed dependencies
nd-fmm-math (with its re-exported `Complex` for the Helmholtz family), num-traits;
dev-dependencies: proptest, approx, serde_json (the dipole fixture, Phase 6 T8, and the
Helmholtz fixture, CH1.3a).
No MPI, no octree, no CubeCL, and no dependency on nd-fmm-validate (not even as a
dev-dependency). Anything else needs a note in the PR.

## Test oracle
Exact 1/|x − y| and its gradient; the truncated Legendre series for same-degree checks;
exactness identities (M2M after P2M equals P2M at the parent, L2L exact on local
polynomials, composition); `direct` checks `rotation`; `direct_sum` checks every chain
within its truncation bound. Helmholtz: FMM3D's direct sum (the fixture, itself checked
at 30 digits), Laplace's operators in the limit k → 0 with the tolerances of proposal §9,
and the Yukawa real path against the complex one at k = iκ.
