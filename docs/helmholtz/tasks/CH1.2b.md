# CH1.2b Radial functions and scaling

Started 2026-10-29 on `tbetcke/h-ch1-2b`. Roadmap: "CH1.2b Radial functions and scaling"
(H1; the second part of CH1.2, fourth of the priority list). Proposal: §10 CH1.2; §3.4,
§5.1, §5.5, §6.1 (placement, D2: A); CONVENTIONS §3.15 ("Radial functions and scaling",
"Precision and range"); S3 (`spikes/radial/SPIKE_REPORT.md`).

The branch starts from `main` (fa43104), beside CH1.2a's open PR #136; it takes the
roadmap's split of CH1.2 from that branch unchanged (identical lines, so the two merge).

## Goal

Put S3's radial functions jₙ, hₙ (complex z), iₙ, kₙ (real x) and their derivatives, with
the scaling of CONVENTIONS §3.15, into `nd-fmm-math` as a module generic over
`T: RealScalar`, tested against an mpmath fixture on every operator's range.

## Acceptance (proposal §10 CH1.2, verbatim, for the whole of CH1.2)

> the identities of §3.15; the reality of the M2M and L2L coefficients for real k

CH1.2b's part (roadmap): against an mpmath fixture on every operator's range (S3's), the
Wronskians, the relations of §3.15 (k₀, jₙ(ix), hₙ(ix)). The reality of M2M and L2L is
CH1.2c's.

## Read first

- CONVENTIONS §3.15 "Radial functions and scaling", "Precision and range".
- `spikes/radial/`: `src/radial.rs` (the algorithms), `src/ranges.rs` (the order ranges
  of the sweeps), `tests/accuracy.rs` (tests 1–5), `tools/reference.py` (the fixture),
  SPIKE_REPORT §1–§4 and §8 question 3.
- `fmm-math/src/scalar.rs`, `fmm-math/CLAUDE.md`, the style of `harmonics.rs`.

## Changes

`nd-fmm-math` only:

- The complex type: planned as a minimal own `Complex<T: RealScalar>`; on the user's
  sign-off of 2026-10-30 (open question 1) `num_complex::Complex` instead, a new
  `[workspace.dependencies]` entry `num-complex = "0.4"` (0.4.6, already in the lock file
  through rlst), re-exported as `nd_fmm_math::Complex`. `radial` divides by Smith's
  algorithm in a private helper: num-complex's division forms |b|², and its extra
  rounding took the Wronskian at |z| = 800 to 1.0e-13.
  Proposal §2 item 3 keeps complex quantities in real planes in the operators' storage;
  this type is for the arithmetic of `nd-fmm-math` and the crates above it.
- `radial` (new, public): with s the power scale, 0 < s ≤ 1,
  - `bessel_j(z, s, values, derivatives)`: ĵₙ = e^{−Im z} jₙ(z)/sⁿ, Im z ≥ 0, by Miller's
    method in ratio form with FMM3D's start test (computed in f64), normalised by ĵ₀ or
    ĵ₁, a power series for |z| < 0.5;
  - `hankel_h(z, s, …)`: ĥₙ = e^{Im z} hₙ⁽¹⁾(z) sⁿ, z ≠ 0, upward;
  - `modified_i(x, s, …)`: îₙ = e^{−x} iₙ(x)/sⁿ, x ≥ 0, Miller;
  - `modified_k(x, s, …)`: k̂ₙ = eˣ kₙ(x) sⁿ, x > 0, k₀ = e^{−x}/x, upward;
  - derivatives (optional, same length) from the neighbouring orders with the same
    factors. Orders 0..values.len(); nothing allocates.
  - Module documentation: the definitions, the relations jₙ(ix) = iⁿ iₙ(x) and
    hₙ(ix) = −i^{−n} kₙ(x), the tested ranges (CONVENTIONS §3.15 asks `nd-fmm-math` to
    state them), and the tangential gradient's fₙ(z)/z: a plain division away from
    z = 0, and at z = 0 the limit ĵ₁/z = 1/(3s), ĵₙ/z = 0 for n ≥ 2 (S3 question 3; the
    operators that need it are CH1.3's and CH3.x's).
- `lib.rs` documentation, `fmm-math/CLAUDE.md`.
- Fixture: `tools/fixtures/gen_helmholtz_radial.py` (S3's `reference.py`, same sets and
  rows, plus the header of the CH1.2a fixture with `convention_version`) writes
  `fmm-math/fixtures/helmholtz_radial.json`; `tools/fixtures/README.md`.

No Laplace module changes; no table; no convention. One dependency: `num-complex`
(signed off).

Not in scope: a real-arithmetic jₙ for real k (S3 §7: an optimisation, measured when the
operators need it); a quotient output fₙ(z)/z (documented above instead).

## Tests that prove it

In `fmm-math/tests/radial/`, f64 unless stated, one thread:

1. Every fixture row (S3's 571 sets, 13,174 rows: each operator's range for real,
   complex and imaginary k, Yukawa, random points to |z| = 4550 and order 1481): value
   and derivative within 1e-13, relative for h, i, k and relative to the oscillation
   envelope for j; reference underflows require ours below 1e-280. Sets above order 300
   in an ignored release test if the debug run is slow.
2. The Wronskians ĵĥ′ − ĵ′ĥ = i/z², the cross product ĵₙĥₙ₊₁/s − sĵₙ₊₁ĥₙ = −i/z²,
   îk̂′ − î′k̂ = −1/x², on S3's dense grid (|z| 10⁻⁶ to 4550, seven arguments, every
   order an operator needs there): 1e-13 relative to the terms.
3. §3.15's relations: the real path against the complex one at z = ix (îₙ = i⁻ⁿ ĵₙ(ix),
   k̂ₙ = −iⁿ ĥₙ(ix) and the derivatives), on the same grid: 1e-13; k̂₀(x) = 1/x and
   ĵ₀, ĵ₁, ĵ₂, ĥ₀, ĥ₁, ĥ₂, î₀, î₁, k̂₁ against their closed forms: 1e-14.
4. Finite on every operator's range (Yukawa to κρ = 4550), and z = 0 exact (ĵ₀ = 1, the
   rest 0, ĵ₁′ = 1/(3s)).
5. Two scales differ by (s′/s)ⁿ: 1e-13.
6. Prefix consistency: ĥ and k̂ bit for bit independent of the length; ĵ and î within
   1e-15 relative (Miller's start order depends on it).
7. f32 against f64 at s = 1, |z| from 0.1 to 20, n ≤ 8: 1e-5 relative.
8. Panics: Im z < 0, ĥ at z = 0, k̂ at x = 0, s outside (0, 1], a derivative slice of the
   wrong length.
9. (Dropped with the own type: `Complex` unit tests.)

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
default and workspace tests), plus `cargo test -p nd-fmm-math --release -- --ignored`,
and the generator's `--check`. No MPI crate changes.

## Machine runs

None.

## Constraints and risks

- Constraint 1: no Laplace code path changes (a new module and a re-export).
- Constraint 9: no data-dependent decision; Miller's start order depends on |z| and the
  length only, so a value is a function of (z, s, length), deterministic.
- Risk: generic `T` arithmetic in the Miller start test would change the start order in
  f32; it is computed in f64 for both.

## Pending runs

None.

## Open questions

1. The complex type: a minimal `Complex<T>` in `nd-fmm-math` (as planned), or
   `num-complex` as a workspace dependency (a new direct dependency of `nd-fmm-math`).
   Signed off 2026-10-30: "num-complex".
