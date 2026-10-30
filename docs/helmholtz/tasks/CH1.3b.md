# CH1.3b Leaf operators

Started 2026-10-30 on `tbetcke/h-ch1-3b`. Roadmap: "CH1.3b Leaf operators" (H1; the second
part of CH1.3, fifth of the priority list). Proposal: §10 CH1.3; §2, §3.4 (scaling and the
k → 0 map), §6.1 (placement, D2: A), §9 items 2–4; CONVENTIONS §3.15 ("Kernel and
factors", "Angular basis", "Radial functions and scaling", "Storage").

## Goal

P2M, L2P, P2L and M2P of the Helmholtz family in `nd-fmm-ref`, in §3.15's stored form (the
level's power scale and damping), with gradients, for complex expansions (any k, complex
charges) and Yukawa's real ones (real charges), tested against CH1.3a's direct sum.

## Acceptance (proposal §10 CH1.3, verbatim, for the whole of CH1.3)

> addition theorems; the k → 0 limit with the tolerances of Section 9; Yukawa two ways;
> the direct sum against the FMM3D fixtures of CH1.4 to round-off

CH1.3b's part (roadmap): the series (P2M → M2P, P2L → L2P) against CH1.3a's direct sum
within the truncation; the k → 0 limit against Laplace's leaf operators under the diagonal
map of §3.4 with the tolerances of §9; Yukawa two ways; every applied exponent
non-positive (finite at large Im(k) a).

## Read first

- CONVENTIONS §3.15 (all of "Kernel and factors" to "Storage"), §3.3 (Rₙᵐ, Iₙᵐ), §3.6,
  §3.7 (Laplace's scaled coefficients).
- `fmm-ref/src/leaf.rs` (Laplace's operators: structure, panics, real storage, `Frame`),
  `fmm-ref/src/helmholtz/p2p.rs` (the oracle), `fmm-ref/tests/helmholtz/` (helpers).
- `nd_fmm_math::orthonormal::{harmonics, harmonics_derivatives}` (the frame φ = 0 on the
  axis), `nd_fmm_math::radial` (the scaled functions; "The tangential gradient").

## Changes

`nd-fmm-ref` only, new modules under `helmholtz`; no Laplace function changes.

- `src/helmholtz/level.rs` (new; `helmholtz::Level`): a level's constants of §3.15 from k
  and the box half-width r: the circumradius a = √3 r, the scale s = min(|k| a, 1), the
  damping exponent β = Im(k) a, and the kind (Yukawa when Re k = 0). Panics unless k is
  finite, nonzero, Re k ≥ 0 and Im k ≥ 0, and r > 0.
- `src/helmholtz/workspace.rs` (new; `helmholtz::Workspace`): the operators' temporaries
  for degrees up to p (harmonics and their angular derivatives, radial values and
  derivatives, complex and real), allocated once.
- `src/helmholtz/leaf.rs` (new, public), generic over `T: RealScalar`, each taking the
  degree p, the `Level`, the box's `Frame` (centre and half-width; its radius must equal
  the level's), the points, the `Workspace` and the output, accumulating (+=):
  - complex path, any k: `p2m`, `p2l` (complex charges → one complex expansion, two real
    planes of (p + 1)², the real part then the imaginary part), `l2p`, `m2p` (complex
    potential and optionally gradient per target);
  - Yukawa's real path, Re k = 0: `p2m_yukawa`, `p2l_yukawa` (real charges → M̂^Y, L̂^Y,
    (p + 1)² reals), `l2p_yukawa`, `m2p_yukawa` (real potential and gradient);
  - the formulas of §3.15's stored form, with ρ = y − c in physical units, z = kρ:
    P2M ĵₙ(kρ) e^{Im(k)ρ − β}, L2P ĵₙ(k|v|) e^{Im(k)|v| − β}, P2L and M2P 4π ik ĥₙ e^{β − Im(k)ρ};
    Yukawa the same with î, k̂ and 4πκ. Gradients from the radial derivative and
    `harmonics_derivatives`: f′ k Yₙᵐ r̂ + (f/|v|)(∂θY θ̂ + (1/sin θ) ∂φY φ̂), with
    f/|v| = k f(z)/z and its limit at z = 0 for L2P at the centre (radial's "The tangential
    gradient").
  - The real-k, real-charge multipole of §3.15's storage table is the real plane of `p2m`
    (its imaginary plane is exactly zero); M2L and the translations are CH1.3c's.
  - Single charge vector only (the `_many` forms belong to the drivers' tasks).
- `helmholtz/mod.rs`, `lib.rs` docs, `fmm-ref/CLAUDE.md`.

## Tests that prove it

`fmm-ref/tests/helmholtz/` (the CH1.3a executable), new modules, f64 unless stated, single
thread. Kinds of k: real, complex (arg π/4 and Im k = Re k/10) and imaginary.

1. **Series** (`leaf_series.rs`): P2M → M2P and P2L → L2P against `direct_sum`, potential
   and gradient, relative ℓ2 (§9.1) over the targets, for |k| a ∈ {0.5, 5, 30} and every
   kind, complex charges. P2M: 60 sources uniform in the box plus its 8 corners (ρ ≤ a), 40
   targets at |v| = 2.5a in random directions and two on the ±z-axis. P2L: 60 sources at
   |ρ| = 2.5a, 40 targets in the box plus its corners, its centre (the z = 0 gradient) and
   two on the ±z-axis. Within the truncation: the error at p₀ (per configuration: 40, 48,
   76) ≤ 1e-12, at a lower pₜ (12, 12, 36) above 1e-7, falling from pₜ to (pₜ + p₀)/2 to p₀.
2. **Stored form** (`leaf_stored.rs`): a single charge on the z-axis, M̂ₙ⁰ and L̂ₙ⁰ for n ≤ 1 against
   the closed forms (j₀ = sin z/z, j₁, h₀ = −i e^{iz}/z, h₁) times e^{∓β} s^{∓n}, at s < 1 and
   s = 1 and Im(k) a up to 50, within 1e-12: pins the scale, the damping and 4π ik. Real k and real
   charges: the imaginary plane of `p2m` exactly zero (§3.15's real multipole).
3. **Large damping**: Im(k) a = 1000 (Yukawa and complex k), sources at the corners and
   the centre, P2L sources and M2P targets at 1.5a: every output finite (e^β overflows),
   the P2M coefficients below Σ|q|. Series (test 1) also at |k| a = 100, Im(k) a = 100
   and about 70, p₀ = 116 (the direct sum stays above underflow).
4. **k → 0** (`leaf_limit.rs`), on a fixed box and points, at |k| = 1e-2, 1e-4, 1e-6 for
   every kind and Yukawa's real path: the diagonal map of §3.4,
   M̂ₙᵐ ≈ k̂ⁿ 3^{−n/2} √((2n + 1)/(4π)) cₘ νₘ σₘ / (2n + 1)!! · M̃ₙᵐ and
   L̂ₙᵐ ≈ 4π (2n − 1)!! k̂^{−n} 3^{n/2} √((2n + 1)/(4π)) cₘ σₘ / (νₘ r) · L̃ₙᵐ (k̂ = k/|k|,
   cₘ = √2 for m ≠ 0, νₘ = √((n + m)!(n − m)!), σₘ = −1 in the slots m < 0; Laplace's M̃, L̃
   of §3.7), per degree in ℓ2 over the orders relative to the per-source magnitudes:
   - P2M, undamped (e^{β} M̂): within (|k| a)²/6; damped within |k| a;
   - P2L, undamped (e^{−β} L̂): degree 0 within |k| ρ_max, degrees ≥ 1 within (|k| ρ_max)²;
     real k degree 0's real part within (k ρ_max)²/2 (§9: (k a)² for real parts, k a for
     the imaginary part of degree 0);
   - M2P and L2P on Laplace's coefficients through the map against Laplace's `m2p`, `l2p`,
     potential and gradient, within the same orders;
   - each plus round-off; every error falls with the bound's power of k.
5. **Yukawa two ways** (`leaf_yukawa.rs`): the complex operators at k = iκ against the real
   ones for κ a ∈ {0.5, 5, 50, 300}: M̂ = iⁿ M̂^Y and L̂ = i^{−n} L̂^Y per degree, and the
   potentials and gradients, complex charges as two real vectors on the Yukawa path, within
   1e-13; for real charges the complex path's values real within 1e-13.
6. **f32**: P2M → M2P and P2L → L2P at p = 8, s = 1, against the f64 operators, within 1e-5.
7. **Panics** (`panics.rs`): lengths, the workspace's degree, a frame whose radius is not
   the level's, a Yukawa operator on a level with Re k ≠ 0, `Level::new` outside the range.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`, the
default and workspace tests); `cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`,
`cargo test -p nd-fmm-ref`. No MPI crate changes, no manifest changes.

## Machine runs

None.

## Constraints and risks

- Constraint 1 (Laplace): no Laplace code changes; `nd-fmm-ref`'s Laplace modules untouched.
- Constraint 6 (trusted paths): the series against CH1.3a's direct sum, the limit against
  Laplace's operators, the closed forms for the stored normalisation.
- Risk: the truncation tests' p₀ at |k| a = 30 needs p ≈ 60–80; the oracle is O(p²) per
  point, fine in debug.

## Pending runs

None.

## Open questions

None.
