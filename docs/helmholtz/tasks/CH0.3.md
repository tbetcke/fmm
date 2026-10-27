# CH0.3 S3: radial functions

Started 2026-10-27 on `tbetcke/h-ch0-3`. Roadmap: "CH0.3 S3: radial functions" (H0).
Proposal: §10 CH0.3; §3.1 (the expansion, Yukawa's powers of i), §3.4 (scaling), §3.5
(p), §3.7 (leaves), §4 item 3, §5.1, §5.2, §5.5, §5.6, §8 R4.

## Goal

Show that jₙ, hₙ = hₙ⁽¹⁾ (complex argument in the closed first quadrant) and iₙ, kₙ (real
argument), with their derivatives, can be computed in f64 to near round-off on the
argument and degree ranges of every operator of the Helmholtz family, and choose the
scaling (power scale and its radius, exponential factor, double factorials) for §3.15.

## Acceptance (proposal §10, verbatim)

> accuracy against arbitrary precision on the argument and degree ranges of every operator
> (the leaf ranges bounded by Section 3.7); the scaling chosen

## Read first

- Proposal §3.1, §3.4, §3.5, §3.7, §4 item 3, §5.1, §5.5, §5.6, §8 R4.
- `spikes/coaxial/` (CH0.2): its spike-grade `src/radial.rs`, its reference tool and the
  report's §1 (the scale's radius left to CH0.3) and §2.3.
- FMM3D 2.1.0 (`~/projects/fmm3d/src/FMM3D-2.1.0.tar.gz`): `src/Common/besseljs3d.f`
  (jₙ/sⁿ by Miller's method), `src/Helmholtz/h3dcommon.f` (`h3dall`, hₙ sⁿ upward),
  `src/Helmholtz/hfmm3d.f` lines 415–420: the scale is min(|k| w_l, 1) with w_l the
  **full box width**, not the half-width that proposal §3.4 cites.

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- `spikes/radial/` (new, package `nd-fmm-spike-radial`), a workspace member outside
  `default-members` like CH0.1's and CH0.2's spikes; dev-dependency `serde_json` (in
  `[workspace.dependencies]` already); no new dependency.
  - `src/complex.rs`: CH0.2's minimal complex f64 type (copied; spikes do not depend on
    each other).
  - `src/radial.rs`, the four families with a power scale s and the exponential factor:
    - ĵₙ = e^{−|Im z|} jₙ(z)/sⁿ and ĵ′ₙ = e^{−|Im z|} jₙ′(z)/sⁿ: Miller's method in ratio
      form (rₙ = jₙ/jₙ₋₁ by the backward recurrence from a start order found by the
      forward growth test, as FMM3D), normalised by the closed form of ĵ₀ or ĵ₁,
      whichever is larger; z = 0 exactly handled;
    - ĥₙ = e^{Im z} hₙ(z) sⁿ and ĥ′ₙ: upward recursion from the closed forms;
    - îₙ = e^{−x} iₙ(x)/sⁿ: Miller's ratios (all positive), normalised by
      î₀ = −expm1(−2x)/(2x); k̂ₙ = eˣ kₙ(x) sⁿ: upward (dominant, positive);
    - derivatives from the neighbouring orders (FMM3D's forms).
  - `src/ranges.rs`: the operators' argument and degree ranges (below), and the scale
    candidates.
  - `tools/reference.py` (PEP 723, `mpmath==1.4.1` as CH0.1 and CH0.2): values from
    mpmath's `besselj`, `bessely`, `besseli` and `besselk` at half-integer order, hₙ
    through `besselk` (H⁽¹⁾_ν(z) = (2/(πi)) e^{−iπν/2} K_ν(−iz), no cancellation for
    Im z ≥ 0), derivatives from the neighbouring orders, so it shares no algorithm with
    the spike; each value at two precisions 40 digits apart, agreeing to 10⁻³⁰. First, the
    closed forms at n ≤ 2 and the relations jₙ(ix) = iⁿ iₙ(x), hₙ(ix) = −i⁻ⁿ k̃ₙ(x) to
    10⁻⁴⁰. Writes `fixtures/reference.json` (committed, under 1 MB); `--check`
    regenerates it byte for byte.
  - `tests/accuracy.rs`, `examples/{ranges,time}.rs`, `SPIKE_REPORT.md`.
- Root `Cargo.toml` `members` gains `spikes/radial`; `Cargo.lock` gains its entry.

The normalisation of kₙ: kₙ here is k̃ₙ = (2/π) times DLMF's 𝗄ₙ, so k̃₀(x) = e⁻ˣ/x and
hₙ(iκρ) = −i⁻ⁿ k̃ₙ(κρ); with jₙ(iκρ) = iⁿ iₙ(κρ) the Yukawa series is
κ Σ iₙ(κρ<) k̃ₙ(κρ>) Σₘ Yₙᵐ Yₙᵐ (ik = −κ). A proposal for §3.15 item 3; CH1.1 decides.

**The ranges** (ε from 10⁻³ to 10⁻¹⁰, p from proposal §3.5's formula, k a with a the
circumradius as in §3.5's tables, c_leaf ≤ 2 from S8's range, kD from 10⁻² to 3000 as in
§3.5's last row, arg z from 0 to π/2):

| Operator | Family | Arguments | Orders |
| --- | --- | --- | --- |
| P2M, L2P | j, j′ (i, i′) | k ρ, 0 ≤ ρ ≤ a_leaf: \|z\| ≤ 5.44 c_leaf ≤ 10.9 | ≤ p_leaf + 1 ≈ 42 |
| M2P, P2L | h, h′ (k̃, k̃′) | k ρ, √3 a_b ≤ ρ ≲ 3.5 w_leaf: \|z\| ≤ 50 | ≤ p_leaf + 1 |
| M2M, L2L start | j (i) | k a/2, k a ≤ 650 | ≤ p_l + p_{l+1} ≤ 1460 |
| M2L start, T(ŝ) | h (k̃) | k t, 2w ≤ t ≤ 6a: \|z\| ≤ 3900 | ≤ 2P − 2 ≤ 1460 |
| Projection (k a > 40) | j, j′, h, h′ | k ρ, ρ ∈ {a, 4a/√3}: \|z\| ≤ 870 | ≤ p_l ≈ 730 |

Yukawa has no leaf rule (§3.7), so its P2M and L2P arguments are not bounded by c_leaf:
the sweep takes κρ to 3900 there too, where e^{κρ} overflows f64 without the exponential
factor. Small arguments: down to |z| = 10⁻¹² and z = 0 exactly (a point at a centre).

**Scale candidates** (s = min(|k| r, 1)): r = w (FMM3D 2.1.0), r = w/2 (proposal §3.4),
r = a = √3 w/2 (CH0.2); and the double factorials absorbed (a per-degree scale on
low-frequency levels). The exponential factor applied always (a no-op for real k) or
only above |Im z| ≈ 1.

## Tests that prove it

All in f64 against the reference, one thread,
`cargo test -p nd-fmm-spike-radial --release -- --include-ignored`.

1. **Reference on every operator's range**: every fixture entry (each operator's range
   at its ends and inside, every kind of k, plus random (z, n) over the whole range),
   the four families and their derivatives. Metric: relative error for h, h′, i, i′, k̃,
   k̃′ (no zeros in the closed upper half plane or for x > 0); for j and j′ the error
   relative to the oscillation envelope max(|jₙ|, |hₙ|) where n < |z| (real zeros),
   relative elsewhere. Bound 10⁻¹³ (CH0.2's start values reached 3.3·10⁻¹⁴). Reference
   values below the f64 range: ours below 10⁻²⁸⁰ (graceful underflow).
2. **Identities on a dense grid** (f64 only, no reference): the Wronskians
   jₙ hₙ′ − jₙ′ hₙ = i/z² and iₙ k̃ₙ′ − iₙ′ k̃ₙ = −1/x², and jₙ hₙ₊₁ − jₙ₊₁ hₙ = −i/z², in
   the scaled form (the scales cancel), over |z| ∈ [10⁻⁶, 4000], arg z ∈ [0, π/2], every
   order to the range's maximum; relative to the size of the terms.
3. **Yukawa two ways**: the real path against the complex path at z = iκρ,
   îₙ = i⁻ⁿ ĵₙ(iκρ) and k̃ₙ = −iⁿ ĥₙ(iκρ), to round-off.
4. **Finite everywhere**: every family on every operator's range, including Yukawa at
   κρ = 3900 and z = 0, returns finite values (no overflow, NaN or infinity); the exact
   values at z = 0 (ĵ₀ = 1, ĵₙ = 0, ĵ′₁ = 1/(3s)).
5. **Scale consistency**: the same z at two scales s, s′ differs by (s′/s)ⁿ to round-off.

## Checks

`/verify` (no default member changes: format, Clippy of the workspace with all targets,
the workspace tests, the spike's tests in release with `--include-ignored`);
`uv run spikes/radial/tools/reference.py --check` regenerates the fixture byte for byte.

## Machine runs

None. Times per value on the M3 Max (release, one thread), for §7 and CH1.2.

## Constraints and risks

- Constraint 1 (Laplace): untouched; only the root `Cargo.toml` (`members`) and
  `Cargo.lock` (the spike's entry) are shared files.
- §3.15 is not written (CH1.4, CH1.1); the spike proposes the scaling and k̃ₙ's
  normalisation for sign-off.
- Risk: the reference at orders to 1460 and |z| to 3900 (mpmath took up to 0.7 s per
  value at (1500, 3000 + 300i)); sampled entries only.
- Risk R4 (proposal §8): loss of accuracy for complex arguments or overflow for Yukawa;
  tests 1, 3 and 4 measure both.

## Pending runs

None.

## Open questions

Filled when they arise.
