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
- **Up waves** (target above) of a stored multipole (changed in the implementation from
  the X̂ = e^{−β} X first planned: the waves represent r_l φ̃ itself, with e^β on each
  conversion, so that no factor e^{2β}, up to e^{65}, leaves a later f32's range):
  X(λₖ, αⱼ) = 2π e^β (wₖ/Nₖ) Σ_{0≤m≤min(Fₖ, p)} [Aₖ(m) cos mαⱼ + Bₖ(m) sin mαⱼ],
  Aₖ(m) = Σₙ Pₖ(n, m) M̂ₙ^{+m}, Bₖ(m) = Σₙ Pₖ(n, m) M̂ₙ^{−m}; **down waves** with
  (−1)ⁿ⁻ᵐ Pₖ(n, m). All Nₖ nodes complex (no conjugate symmetry, unlike §3.14).
- **Shift** by d (box widths, the direction's frame): e^{∓λₖ d_z + itₖ(d_x cos αⱼ + d_y sin αⱼ)}.
- **Waves to local**: L̂ₙ^{+m} += 4π e^β (−1)ⁿ Pₖ(n, m) Σⱼ V cos mαⱼ, slot −m with sin,
  m ≤ min(Fₖ, n); down with (−1)ⁿ⁻ᵐ more. (iⁿ = (−1)ⁿ i⁻ⁿ, so one table serves both.)
  The two e^β with |e^{−λₖ d_z}| ≤ e^{−Im κ d_z}, d_z ≥ 2 (Re λₖ ≥ Im κ on both node maps)
  and 2β = √3 Im κ keep the product ≤ 1, as RCR's e^{2β − Im(k)|t|}.
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
  waves X (r_l φ̃ itself), the shift, waves to local, the directions, frames and Oⁿ, the region
  and which quadratures (CH6.1a's tables and counts, CH6.1b's generator as the tests'
  path); merge and shift left to CH6.1d (by §3.14's geometry). **For sign-off.**
- `tools/fixtures/check_3_15_plane_wave.py` (new, mpmath): the representation of hₙ Yₙᵐ
  up and down, the plane-wave expansion at complex ŝ, and P2M → waves → shift → local →
  L2P in stored form against the direct kernel, for real, complex and imaginary κ, by a
  numerical λ integral (no quadrature table), for n ≤ 5 (the representation) and n ≤ 40
  (the expansion and the one-wave local).
- `fmm-tables/CLAUDE.md`: the module in the purpose list and its rule.

## Tests that prove it

In `fmm-tables/tests/helmholtz/plane_wave_pair.rs`, f64, one thread:

1. **The factors**: Pₖ(n, m) against `nd_fmm_math::orthonormal::harmonics` at real
   directions (λ = −iκ cos θ, t = κ sin θ gives a real ŝ), to 1e-13 (absolute, at
   |x| ≤ √3/2).
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
   about 0.3 to 40 (four or more levels, several domains), iprec 1–3, one plane (real k,
   real charges) and two planes. Debug: a sample (each case on one or two levels);
   release, ignored: all. Generated quadratures: two κ in debug, the rest ignored.
   **Degrees** (settled in the implementation, "Results"): degrees that resolve the
   multipoles of the charges, base 20, 36, 60 for iprec 1–3 (ε 10⁻³, 10⁻⁶, 10⁻⁹ for the
   generator) plus ⌈0.9 |κ|⌉; below them the chain's difference from RCR also holds the
   quadrature's error on the truncated multipole's tail, which the ignored
   `degree_study` records against the direct sum.
3. **Composition**: shift(d₁) then shift(d₂) equals shift(d₁ + d₂) to rounding; a pair
   along a direction and its frame rotation round trip (Oⁿ orthogonal).
4. **Panics**: an offset not 2 or 3 boxes along the direction; a multipole or local of
   another degree; a quadrature of another level.
5. `rotation_blocks` against `CubeTransforms`' blocks, bit for bit (a unit test in
   `helmholtz::classes`).

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

## Results

Release, the M6, f64, one thread, 2026-11-03. Every V offset (316) along `Direction::of`
and every direction merge and shift gives it; the error measure of the test module (the
largest |Δ| at the 27 target points, box widths, per unit Σ|q|).

**Against RCR at the resolved degrees** (`against_rcr_with_fmm3ds_quadratures_every_level`,
`…_generated_quadratures_every_level`; 36 s together): k̂₀ = 80 (real), 80 + 12i and
20 + 40i, levels 2–9 (κ_l from 40, level 2 of k̂₀ = 80, down to 0.08 + 0.16i, level 9 of
20 + 40i), iprec 1–3, one and two planes, 96 cases: the worst 0.76× the stated error
(iprec 1, κ = 10 + 20i, p = 41), the median 0.14×, against the bound 3×. Generated at
ε = 10⁻³, 10⁻⁶, 10⁻⁹ on levels 2, 5 and 9: the worst 0.07 ε. Debug sample 11 s.

**The convention's check** (`uv run tools/fixtures/check_3_15_plane_wave.py`, mpmath at 30
digits, about eight minutes on the M6): all passed. The representation of hₙ Yₙᵐ up and down
(n ≤ 5, every m) for κ = 1.3, 0.9 + 0.4i and 1.1i within 2.8·10⁻²⁹ relative; the solid
recursion against the real harmonics 7.9·10⁻³¹; the plane-wave expansion at complex ŝ
3.5·10⁻³⁰; the waves of a stored multipole against its r_l φ̃, and one up and one down wave
to a stored local by L2P, within 2.4·10⁻³⁰ (a first run, with the λ integral cut at t = 40,
missed the representation at κ = 1.1i by 3.6·10⁻¹¹, the truncated tail; cut at 160 since).

**The degree** (`degree_study`; sources at the box's corners, centre and four seeded
points, every third offset along `Direction::of`, two planes). The chain is exact per
degree, but its waves of a degree-p multipole are a truncated series in λ; the
quadrature, accurate for the full series, errs on the truncation's tail, the more the
larger the λ its nodes reach (iprec 3 more than 2). Below the resolving degree the chain
therefore departs from RCR by more than the stated error (55× at κ = 1.25, iprec 2, p = 14
in the first debug run; 8,935× at iprec 3, p = 20), and against the direct sum it errs
above RCR at the degrees an FMM uses (the 55× is from the first debug run, at a degree the
study does not include). Chain − direct and RCR − direct, box widths, the study's rows
arranged by precision into columns:

| κ | p | RCR − direct | chain − direct, iprec 1 | iprec 2 | iprec 3 |
| --- | --- | --- | --- | --- | --- |
| 10.0000+0.0000i | 8 | 1.9e-1 | 1.9e-1 | 1.9e-1 | 1.9e-1 |
| 10.0000+0.0000i | 12 | 6.7e-3 | 6.7e-3 | 6.7e-3 | 6.7e-3 |
| 10.0000+0.0000i | 16 | 6.2e-5 | 6.1e-5 | 6.2e-5 | 6.2e-5 |
| 10.0000+0.0000i | 20 | 1.2e-6 | 2.8e-6 | 1.2e-6 | 1.2e-6 |
| 10.0000+0.0000i | 24 | 7.5e-8 | 2.8e-6 | 1.6e-7 | 3.0e-7 |
| 10.0000+0.0000i | 28 | 4.1e-9 | 2.8e-6 | 4.9e-9 | 5.5e-8 |
| 10.0000+0.0000i | 32 | 2.1e-10 | 2.8e-6 | 4.5e-9 | 2.2e-9 |
| 10.0000+0.0000i | 40 | 1.8e-12 | 2.8e-6 | 4.5e-9 | 5.9e-12 |
| 1.2500+0.0000i | 8 | 2.5e-4 | 2.6e-4 | 2.5e-4 | 2.5e-4 |
| 1.2500+0.0000i | 12 | 1.8e-5 | 2.4e-5 | 1.8e-5 | 1.8e-5 |
| 1.2500+0.0000i | 16 | 7.8e-7 | 1.6e-5 | 2.1e-6 | 7.9e-7 |
| 1.2500+0.0000i | 20 | 8.7e-8 | 1.6e-5 | 1.7e-7 | 5.1e-7 |
| 1.2500+0.0000i | 24 | 1.2e-8 | 1.6e-5 | 1.8e-8 | 1.7e-7 |
| 1.2500+0.0000i | 28 | 8.5e-10 | 1.6e-5 | 1.8e-8 | 1.2e-8 |
| 1.2500+0.0000i | 32 | 5.1e-11 | 1.6e-5 | 1.8e-8 | 3.2e-10 |
| 1.2500+0.0000i | 40 | 6.3e-13 | 1.6e-5 | 1.8e-8 | 1.5e-11 |
| 0.3125+0.0000i | 8 | 2.4e-4 | 2.4e-4 | 2.4e-4 | 2.4e-4 |
| 0.3125+0.0000i | 12 | 1.6e-5 | 3.0e-5 | 1.6e-5 | 1.6e-5 |
| 0.3125+0.0000i | 16 | 8.1e-7 | 3.0e-5 | 1.1e-6 | 8.1e-7 |
| 0.3125+0.0000i | 20 | 7.3e-8 | 3.0e-5 | 1.0e-7 | 6.6e-7 |
| 0.3125+0.0000i | 24 | 9.9e-9 | 3.0e-5 | 1.5e-8 | 2.5e-7 |
| 0.3125+0.0000i | 28 | 7.1e-10 | 3.0e-5 | 1.5e-8 | 1.8e-8 |
| 0.3125+0.0000i | 32 | 5.2e-11 | 3.0e-5 | 1.5e-8 | 5.3e-10 |
| 0.3125+0.0000i | 40 | 5.5e-13 | 3.0e-5 | 1.5e-8 | 9.2e-12 |
| 1.2500+0.1875i | 8 | 2.5e-4 | 2.6e-4 | 2.5e-4 | 2.5e-4 |
| 1.2500+0.1875i | 12 | 1.8e-5 | 2.7e-5 | 1.8e-5 | 1.8e-5 |
| 1.2500+0.1875i | 16 | 7.8e-7 | 2.2e-5 | 2.0e-6 | 7.8e-7 |
| 1.2500+0.1875i | 20 | 8.7e-8 | 2.2e-5 | 2.0e-7 | 5.3e-7 |
| 1.2500+0.1875i | 24 | 1.2e-8 | 2.2e-5 | 1.1e-8 | 1.8e-7 |
| 1.2500+0.1875i | 28 | 8.5e-10 | 2.2e-5 | 1.1e-8 | 1.3e-8 |
| 1.2500+0.1875i | 32 | 5.1e-11 | 2.2e-5 | 1.1e-8 | 4.0e-10 |
| 1.2500+0.1875i | 40 | 6.3e-13 | 2.2e-5 | 1.1e-8 | 7.1e-12 |
| 0.6250+1.2500i | 8 | 2.1e-4 | 2.1e-4 | 2.1e-4 | 2.1e-4 |
| 0.6250+1.2500i | 12 | 1.5e-5 | 1.4e-5 | 1.4e-5 | 1.5e-5 |
| 0.6250+1.2500i | 16 | 6.3e-7 | 4.8e-6 | 9.5e-7 | 6.3e-7 |
| 0.6250+1.2500i | 20 | 6.8e-8 | 4.9e-6 | 8.8e-8 | 6.2e-7 |
| 0.6250+1.2500i | 24 | 1.0e-8 | 4.9e-6 | 5.7e-9 | 2.2e-7 |
| 0.6250+1.2500i | 28 | 7.4e-10 | 4.9e-6 | 5.7e-9 | 1.4e-8 |
| 0.6250+1.2500i | 32 | 5.0e-11 | 4.9e-6 | 5.7e-9 | 5.3e-10 |
| 0.6250+1.2500i | 40 | 5.9e-13 | 4.9e-6 | 5.7e-9 | 5.6e-12 |

Stated errors (iprec 1, 2, 3): κ = 10.0000+0.0000i: 4.9e-5, 7.2e-8, 7.9e-11; κ = 1.2500+0.0000i: 5.0e-5, 9.4e-8, 6.1e-11; κ = 0.3125+0.0000i: 7.0e-5, 5.3e-8, 4.5e-11; κ = 1.2500+0.1875i: 7.7e-5, 7.4e-8, 5.9e-11; κ = 0.6250+1.2500i: 5.4e-5, 8.5e-8, 4.4e-11.

Reading: for each degree some precision keeps the chain within about 1–2.3× RCR against
the direct sum in D1's range of errors (iprec 3 at p ≤ 16, iprec 2 at p = 20–24), but
`hwts3e`'s rule, the precision from ε alone, pairs ε = 10⁻⁸ (iprec 3) with p ≈ 24 at
14–25× RCR's error. Above p ≈ 28 (RCR below 10⁻⁹, finer than D1's 10⁻⁸) neither precision
reaches RCR's error. The pairing of a level's quadrature with its degree is CH6.1d's to
choose by measurement against CH6.1's criterion (open question 3).

## Pending runs

None.

## Open questions

Signed off 2026-11-03 (the user: "as recommended"), after the recommendations in the
session: yes to all three, question 3 with the rule's shape named.

1. The §3.15 "Exponential M2L" subsection (the convention): signed off 2026-11-03: as
   recommended, accepted as written (unversioned).
2. The acceptance as tested, within the quadrature's error of RCR on every offset at
   degrees that resolve the multipole, the lower degrees recorded in the degree study:
   signed off 2026-11-03: as recommended, accepted as CH6.1c's criterion met; the FMM's
   errors at its own degrees are CH6.1d's (CH6.1's 1.1× Rotation).
3. CH6.1d pairs each level's quadrature with its degree by measurement: signed off
   2026-11-03: as recommended, with the rule's shape: per level, from p_l and ε, the
   coarsest precision whose chain error against the direct sum stays within a small factor
   of RCR's (about 1.1×, CH6.1's criterion), a function of k, w_l, p_l and ε (constraint
   9); a level where no precision reaches it keeps Rotation, and CH6.1d reports which
   levels those are. To watch: the degree study puts the factor nearer 1.3–2.3× at
   p = 20–24 on its worst-case charges; if CH3.1c's cases agree, H6's gain shrinks to the
   levels at p ≤ 16.
