# CH6.1c The pair chain

Started 2026-11-03 on `tbetcke/h-ch6-1c`. Roadmap: "CH6.1c The pair chain" (H6; priority
list item 2). Proposal: §10 CH6.1; §5.3 (Exponential M2L), §8 R7; CONVENTIONS §3.14 (the
Laplace scheme this generalises) and §3.15; S4's report `spikes/pw-quadrature/SPIKE_REPORT.md`
§1, §2.3 and §6.

## Goal

Translate one V pair of a Helmholtz level by plane waves, multipole to waves, one shift,
waves to local, in the orthonormal basis and stored form of CONVENTIONS §3.15 and along
any of §3.14's six directions, and write the formulas down as a §3.15 subsection for
sign-off.

## Acceptance (proposal §10, verbatim)

CH6.1's: "errors within about 1.1× Rotation's, as T13's budget for Laplace". That needs
CH6.1d's FMM. CH6.1c's own condition (the roadmap's split):

> every V offset on several levels against RCR within the quadrature's error, with FMM3D's
> and with CH6.1b's generated quadratures, real and complex k, one and two multipole
> planes.

## Read first

- CONVENTIONS §3.14 (box widths, the quadrature, the conversions, shifts, directions,
  frames, merge and shift) and §3.15 (kernel and factors, angular basis, stored form,
  rotations, storage).
- `fmm-tables/src/plane_wave/mod.rs` and `geometry.rs` (Laplace's steps, `Direction`,
  `FRAMES`, the pair `m2l`); `fmm-tables/src/helmholtz.rs` (the level units: half-width
  1, k̂_l = k r_l, a box represents r_l φ̃; `HelmholtzTables::m2l`, the RCR trusted path);
  `helmholtz/classes.rs` (`CubeTransforms`, Oⁿ(P) of the cube elements);
  `helmholtz/plane_wave/quadrature.rs` (CH6.1a) and `nd_fmm_ref::helmholtz::plane_wave`
  (CH6.1b), the two sources of nodes.
- `nd_fmm_ref::helmholtz::leaf` (P2M, L2P in stored form) for the tests' charges and
  targets.

## The formulas (derived here; checked in mpmath at 30 digits before planning)

In box widths u (κ = k w = 2k̂_l), with ŝ = (t cos α, t sin α, iλ)/κ, ŝ·ŝ = 1, and
e^{iκŝ·u} = e^{−λz + it(x cos α + y sin α)}, the quadrature of CH6.1a and CH6.1b represents
e^{iκ|u|}/|u| = iκ h₀(κ|u|). Applying the solid harmonic 𝒴ₙᵐ(∇/(iκ)) to both sides
(Hobson's theorem, (z⁻¹ d/dz)ⁿ h₀ = (−1)ⁿ hₙ/zⁿ):

  hₙ(κ|u|) Yₙᵐ(û) = 1/(iⁿ⁺¹ κ) ∫_Γ dλ (1/2π) ∫ dα Yₙᵐ(ŝ) e^{iκŝ·u},   z > 0,

checked for n ≤ 4 (scratchpad script; it becomes `tools/fixtures/check_3_15_plane_wave.py`).
With the plane-wave expansion e^{iκŝ·u} = 4π Σₙ iⁿ jₙ(κ|u|) Σₘ Yₙᵐ(ŝ) Yₙᵐ(û) on the target
side and the stored form M = e^β sⁿ M̂, L̂ = e^β sⁿ L (§3.15; a box represents r_l φ̃,
r_l φ̃ = 4π ik̂ Σ M hₙ Y, so the constant is 4π ik̂/(iκ) = 2π):

- Yₙ^{±m}(ŝ) = κ⁻ⁿ 𝒴ₙ^{±m}(t cos α, t sin α, iλ) = κ⁻ⁿ cₙₘ 𝒫ₙᵐ(iλ, t) {cos mα, sin mα},
  with 𝒫ₙᵐ(z, ρ) the solid associated Legendre polynomial (r² = z² + ρ² = κ²), a
  polynomial in iλ, t and κ², finite as κ → 0; the per-node factor
  Pₖ(n, m) = i⁻ⁿ (s/κ)ⁿ cₙₘ 𝒫ₙᵐ(iλₖ, tₖ) (s/κ has modulus √3/2 on low-frequency levels).
- **Up waves** (target above) of a stored multipole, the stored waves X̂ = e^{−β} X:
  X̂(λₖ, αⱼ) = 2π (wₖ/Nₖ) Σ_{0≤m≤min(Fₖ, p)} [Aₖ(m) cos mαⱼ + Bₖ(m) sin mαⱼ],
  Aₖ(m) = Σₙ Pₖ(n, m) M̂ₙ^{+m}, Bₖ(m) = Σₙ Pₖ(n, m) M̂ₙ^{−m}; **down waves** with
  (−1)ⁿ⁻ᵐ Pₖ(n, m). All Nₖ nodes complex (no conjugate symmetry, unlike §3.14).
- **Shift** by d (box widths, the direction's frame): e^{∓λₖ d_z + itₖ(d_x cos αⱼ + d_y sin αⱼ)}.
- **Waves to local**: L̂ₙ^{+m} += 4π e^{2β} (−1)ⁿ Pₖ(n, m) Σⱼ V cos mαⱼ, slot −m with sin,
  m ≤ min(Fₖ, n); down with (−1)ⁿ⁻ᵐ more. (iⁿ = (−1)ⁿ i⁻ⁿ, so one table serves both.)
  e^{2β} with |e^{−λₖ d_z}| ≤ e^{−Im κ d_z}, d_z ≥ 2 (Re λₖ ≥ Im κ on both node maps) and
  2β = √3 Im κ keeps the product ≤ 1, as RCR's e^{2β − Im(k)|t|}.
- **Frames**: §3.14's three, the multipole into frame f by Oⁿ(Q) per degree and the local
  back by Oⁿ(Q)ᵀ (orthonormal, so no K or S as in §3.14); Oⁿ from `CubeTransforms`
  (R_x(π/2) and R_y(−π/2) are cube elements).

## Changes

`nd-fmm-tables` and `docs/CONVENTIONS.md` only (plus the check script and the records).
No Laplace code, no dependency, no `nd-fmm-plan` change.

- `fmm-tables/src/helmholtz/plane_wave/pair.rs` (new; re-exported from
  `helmholtz::plane_wave`): `PlaneWaveLevel<T>` (name to settle in the code), one level's
  tables built in f64 and rounded to T from k̂_l, p_l and the nodes of either source
  (`from_quadrature(&Quadrature)` for CH6.1a's, `from_nodes(κ, λ, t, w, N, F)` for any,
  CH6.1b's included): the factors Pₖ(n, m), cos and sin of every αⱼ, the frames' Oⁿ;
  steps `rotate_multipole`, `waves` (up and down), `shift`, `local_from_waves`,
  `add_local_from_frame`; and the pair `m2l(offset_index, direction, multipole,
  multipole_planes, local, scratch)` in the level units and slices of
  `HelmholtzTables::m2l` (one or two multipole planes, two local planes), plus
  `m2l_of_offset`. Directions, frames and offsets from `crate::plane_wave::geometry` and
  `crate::geometry` (read, not changed). Allocation only in the builder and `scratch()`.
- `fmm-tables/tests/helmholtz/plane_wave_pair.rs` (new, in the `helmholtz` test binary).
- `docs/CONVENTIONS.md`: a §3.15 subsection "Exponential M2L" (unversioned, as §3.14):
  box widths and κ in the level units, the representation of hₙ Yₙᵐ, the factors, the
  stored waves X̂, the shift, waves to local, the directions, frames and Oⁿ, the region
  and which quadratures (CH6.1a's tables and counts, CH6.1b's generator as the tests'
  path); merge and shift left to CH6.1d (by §3.14's geometry). **For sign-off.**
- `tools/fixtures/check_3_15_plane_wave.py` (new, mpmath): the representation of hₙ Yₙᵐ
  up and down, the plane-wave expansion at complex ŝ, and P2M → waves → shift → local →
  L2P in stored form against the direct kernel, for real, complex and imaginary κ, by a
  numerical λ integral (no quadrature table), for n ≤ 6.
- `fmm-tables/CLAUDE.md`: the module in the purpose list and its rule.

## Tests that prove it

In `fmm-tables/tests/helmholtz/plane_wave_pair.rs`, f64, one thread:

1. **The factors**: Pₖ(n, m) against `nd_fmm_math::orthonormal::harmonics` at real
   directions (λ = −iκ cos θ, t = κ sin θ gives a real ŝ), to 1e-13 relative.
2. **The pair against RCR** (the acceptance): per level the sources are point charges at
   the source box's corners, centre and seeded points with seeded complex charges (real
   for one plane), P2M in stored form (`nd_fmm_ref::helmholtz::leaf::p2m`), one multipole
   per offset; the chain's local and `HelmholtzTables::m2l`'s local of the same multipole,
   both by L2P at the 27 points {−½, 0, ½}³ (box widths) of the target box. Measure:
   max |Δ (w φ̃)|, absolute in box widths (the quadrature's norm, S4 §1) per unit Σ|q|,
   within 3× the stated error (FMM3D's tables; CH6.1a's chain bound) or 3ε (the
   generator), at every one of the 316 offsets along `Direction::of` and along the
   merge-and-shift direction of each child (`PlaneWaveTables::pair_direction`'s rule).
   Cases: real k and complex k (Re, Im > 0, a strongly damped one), levels with κ_l from
   about 0.3 to 40 (four or more levels, several domains), iprec 1–3 at degrees that
   meet them (S7/CH3.3 scale: about 6, 12, 18), one plane (real k, real charges) and two
   planes. Debug: a sample (each case on one or two levels); release, ignored: all.
   Generated quadratures: two κ in debug, the rest ignored.
3. **Composition**: shift(d₁) then shift(d₂) equals shift(d₁ + d₂) to rounding; a pair
   along a direction and its frame rotation round trip (Oⁿ orthogonal).
4. **Panics**: an offset not 2 or 3 boxes along the direction; wrong slice lengths.

The 3× bounds are set after the first measurements, which the PR reports; any bound
beyond what CH6.1a/CH6.1b measured is a question, not a quiet change.

## Checks

`/verify` (nd-fmm-tables changed: format, Clippy default/examples/workspace, tests,
`cargo doc`, `cargo test -p nd-fmm-tables`, its ignored release tests); the check script
by `uv run`.

## Machine runs

None (no timing, no MPI, no Laplace path).

## Constraints and risks

- Constraint 1: no Laplace code changes; `crate::plane_wave::geometry` is only read.
- Constraint 9: the tables are functions of k, w_l, p_l and the quadrature only.
- Risk: the factors (s/κ)ⁿ 𝒫ₙᵐ(iλ, t) grow like |λ|ⁿ; at p ≈ 18–25 and λ ≈ 30 the
  waves are sums of large terms that the shift's e^{−λ d_z} then suppresses (as Laplace's
  λⁿ). If round-off shows, the factors can be split between the two conversions.
- Risk: arbitrary multipoles (not those of in-box charges) need not be represented to
  the quadrature's error (Fₖ is set for a source's reach √2/2); the tests use P2M of
  in-box charges, as the FMM does.

## Pending runs

None yet.

## Open questions

- Sign-off: the §3.15 "Exponential M2L" subsection (the convention).
