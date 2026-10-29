# CH1.2a Orthonormal harmonics and rotation blocks

Started 2026-10-29 on `tbetcke/h-ch1-2a`. Roadmap: "CH1.2a Orthonormal harmonics and
rotation blocks" (H1; the first part of CH1.2, fourth of the priority list). Proposal:
§10 CH1.2; §3.2, §6.1 (placement, D2: A); CONVENTIONS §3.15 ("Angular basis",
"Rotations"); S1 (`spikes/rotation-blocks/SPIKE_REPORT.md`).

## Goal

Put S1's Legendre functions and Gumerov–Duraiswami rotation blocks, and the real
orthonormal harmonics of CONVENTIONS §3.15, into `nd-fmm-math` as a module beside the
Laplace ones, generic over `T: RealScalar`, tested against an mpmath fixture.

## Acceptance (proposal §10 CH1.2, verbatim, for the whole of CH1.2)

> the identities of §3.15; the reality of the M2M and L2L coefficients for real k

CH1.2a's part (roadmap): §3.15's angular identities (against the Racah harmonics through
N at p ≤ 30, the addition formula, orthonormality); the blocks against an mpmath fixture
to p = 1000, orthogonal, against N-conjugated `rotation::blocks` at p ≤ 30, and
Y(Qx̂) = Oⁿ Y(x̂); the π − β rule. The reality of M2M and L2L is CH1.2c's.

## Read first

- CONVENTIONS §3.15 "Angular basis", "Rotations", "Precision and range"; §3.3, §3.5,
  §3.6, §3.8.
- `spikes/rotation-blocks/`: `src/legendre.rs` (telescoped recursion, exponent per
  column), `src/gd.rs` (the wedge, rows 0 and 1, the real map, the compact layout),
  `tests/accuracy.rs`, `tools/reference.py`, SPIKE_REPORT §2–§6.
- `fmm-math/src/harmonics.rs`, `rotation.rs` (style, `blocks` layout, `euler_zyz`),
  `fmm-math/CLAUDE.md`.

## Changes

`nd-fmm-math` only, a new public module `orthonormal` (files `fmm-math/src/orthonormal/`):

- `legendre`: P̄ₙᵐ = √((n − m)!/(n + m)!) Pₙᵐ (no Condon–Shortley phase), S1's telescoped
  recursion on |cos β| with a binary exponent per column; also the column divided by
  sin θ (for the φ-derivative). Generic over `T`, the exponent span from `T`'s range.
- Harmonics: `harmonics(p, x, out)`, Yₙᵐ(x̂) in the layout idx(n, m) of §3.6 (cos in +m,
  sin in −m), cos mφ and sin mφ by the Cartesian recursion of (x + iy)/ρ (no
  trigonometric calls, §3.5); x = 0 and the z-axis take φ = 0. `harmonics_derivatives(p,
  x, value, d_theta, d_phi, scratch)`: ∂θ Y and (1/sin θ) ∂φ Y, regular at the poles,
  from which the gradient of f(kr) Y follows (documented).
- Rotation: `y_blocks(p, cos β, sin β, out, scratch)` in S1's compact layout
  (`compact_len`, `compact_offset`, `compact_blocks_len`); `supplement(p, blocks)` (π − β
  from β by the signs of §3.15); `blocks(p, α, β, γ, out, scratch)`, the full blocks of
  R_z(α) R_y(β) R_z(γ) in the layout of `rotation::blocks`. `Scratch` structs allocate
  once.
- `lib.rs` documentation and `fmm-math/CLAUDE.md` (the module, the new fixture).
- Fixture: `tools/fixtures/gen_helmholtz_angular.py` (from S1's `reference.py`, plus
  harmonics and their derivatives) writes `fmm-math/fixtures/helmholtz_angular.json`;
  `tools/fixtures/README.md`.

No public API of the Laplace modules changes; no table; no convention; no dependency.

## Tests that prove it

All in f64 unless stated, in `fmm-math/tests/orthonormal/`:

1. Harmonics against the Racah `harmonics::regular` through N (§3.15) at p ≤ 30, random
   points: 1e-13 relative to the degree's largest value.
2. Harmonics and both derivatives against the fixture (mpmath, 30 digits) at degrees to
   1000, directions including near-polar (β = 10⁻³) and on-axis: 1e-13 relative to the
   degree's largest |Y| (the S1 error grew to 5e-15 at n = 1000).
3. Addition formula Σₘ Yₙᵐ(a) Yₙᵐ(b) = (2n + 1)/(4π) Pₙ(a · b), n ≤ 300, against the
   fixture's Pₙ: 1e-12 relative to (2n + 1)/(4π).
4. Orthonormality by Gauss–Legendre × uniform-φ quadrature, n ≤ 30: 1e-13.
5. Derivatives against central differences of `harmonics`, n ≤ 20: 1e-7.
6. `y_blocks` against the fixture entries (S1's 54 sets, n ≤ 1000, six angles): 1e-13
   absolute (entries are of order one); n ≤ 300 by default, n = 500–1000 in an ignored
   release test.
7. Orthogonality ‖O Oᵀ − I‖_max ≤ 1e-13, n ≤ 300 (1000 ignored), the six angles.
8. `blocks(α, β, γ)` against N-conjugated `rotation::blocks(euler_zyz(α, β, γ))`, p ≤ 30:
   1e-12; Yₙ(Q x̂) = Oⁿ(Q) Yₙ(x̂) at p ≤ 50: 1e-12.
9. `supplement` against `y_blocks` at π − β: 1e-13.
10. Lower degrees independent of p (bit for bit), length panics, f32 against f64 at
    p ≤ 8 (1e-5).

## Checks

`/verify`; `cargo clippy -p nd-fmm-math --all-targets -- -D warnings`; `cargo test -p
nd-fmm-math` (and `--release -- --ignored` for the large degrees); the generator's
`--check`.

## Machine runs

None.

## Constraints and risks

- Constraint 1: Laplace's modules and their outputs untouched (new module only); the
  default and workspace tests show it.
- Generic over `T` (root rule), though the tables are built in f64 (constraint 5); f32 only
  smoke-tested (D3).
- Risk: derivative accuracy at the poles; the m-ladder for ∂θ and the column divided by
  sin θ for ∂φ avoid any division by sin θ.

## Pending runs

None.

## Open questions

None.
