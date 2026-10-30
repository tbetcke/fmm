# CH1.3c Translations

Started 2026-10-30 on `tbetcke/h-ch1-3c`. Roadmap: "CH1.3c Translations" (H1; the third
part of CH1.3, fifth of the priority list). Proposal: §10 CH1.3; §2, §3.4 (the k → 0 map),
§5.2 (M2M and L2L by rotation, coaxial translation, rotation back), §5.3 (Rotation M2L),
§5.5 (Yukawa's real form), §9 items 2–4; CONVENTIONS §3.15 ("Radial functions and
scaling", "Storage", "Rotations", "Coaxial translations").

## Goal

M2M, L2L and M2L of the Helmholtz family in `nd-fmm-ref`, two ways (direct, by projection
on a sphere from the leaf operators' fields; RCR, by the orthonormal rotations and the
coaxial tables), with rectangular degrees, for complex expansions and Yukawa's real ones,
in §3.15's stored form.

## Acceptance (proposal §10 CH1.3, verbatim, for the whole of CH1.3)

> addition theorems; the k → 0 limit with the tolerances of Section 9; Yukawa two ways;
> the direct sum against the FMM3D fixtures of CH1.4 to round-off

CH1.3c's part (roadmap): the addition theorems (P2M → M2M → M2P, P2L → L2L → L2P and
P2M → M2L → L2P against the direct sum within the truncation; RCR against direct to
round-off); the k → 0 limit against Laplace's translations with the tolerances of §9;
Yukawa two ways.

## Read first

- CONVENTIONS §3.15 ("Radial functions and scaling" to "Coaxial translations").
- `fmm-ref/src/helmholtz/leaf.rs` (the stored form, the evaluators), `fmm-ref/src/rotation.rs`
  (Laplace's RCR: structure, the axis cases), `fmm-ref/src/direct.rs`.
- `nd_fmm_math::orthonormal::rotation` (`y_blocks`, the compact layout, the z-rotation of
  `blocks`), `nd_fmm_math::coaxial` (`CoaxialTable::{m2m, l2l, m2l}`, `get`, the stored
  normalisation), `nd_fmm_math::radial`.
- `fmm-ref/tests/helmholtz/leaf_limit.rs` (the diagonal map of §3.4).

## Changes

`nd-fmm-ref` only, new modules under `helmholtz`; no Laplace function changes, no manifest
change, no convention change.

- `src/helmholtz/direct.rs` (new, public): `m2m`, `l2l`, `m2l` and their `_yukawa` forms.
  Each evaluates the input expansion's field (and, for a local output, its radial
  derivative) with the leaf evaluators (`m2p`, `l2p`) at the nodes of a product quadrature
  on the output's sphere ρ = a_out (Gauss–Legendre in cos θ, uniform in φ), projects onto
  the real orthonormal harmonics, and divides by the output's radial function in stored
  form: ĥₙ for a multipole (4π ik ĥₙ(kρ) e^{β′ − Im(k)ρ}), and for a local the least
  squares from the value and ρ times the radial derivative (jₙ has zeros); Yukawa divides by
  k̂ₙ and îₙ. The node count is fixed by the degrees and |k| times the geometry
  (Q = p_in + p_out + ⌈|k| ρ⌉ + 40 rings, 2Q azimuths). General frames:
  M2M and L2L need the input's sphere inside the parent's (|t| + a_child ≤ a_parent), M2L
  needs the spheres apart (|t| ≥ a_src + a_tgt); asserted.
- `src/helmholtz/rcr.rs` (new, public): `m2m`, `l2l`, `m2l` and `_yukawa` forms, by
  §5.2: rotate the input so that the shift lies on +z (the z-rotation by −φ, a 2 × 2 per
  order, then the transposed y-block of θ from `y_blocks`, cos and sin halves), apply the
  coaxial table per order to slots ±m (rectangular p_in → p_out), rotate back (the y-block,
  then the z-rotation by φ). Stored form (§3.15): M2M multiplies by s_c^n / s_P^{n′}, L2L
  by s_c^{n′} / s_P^n, M2L by 4π ik e^{2β − Im(k)|t|}; Yukawa reads i^{n−n′} E, i^{n′−n} E
  and −i^{n+n′} E (real) with 4πκ. Each takes a `&CoaxialTable` built by the helpers
  `m2m_table`, `l2l_table`, `m2l_table` (the rule of §3.15 through `CoaxialTable::{m2m,
  l2l, m2l}`, with the shift from the levels and frames), and checks its degrees. M2M and
  L2L need a parent–child pair (r_P = 2 r_c, each |Δᵢ| = r_c); M2L one level; asserted.
- Inputs and outputs: degrees `p_in`, `p_out` (rectangular), two real planes of (p + 1)²
  per complex expansion (real then imaginary), (p + 1)² reals on the Yukawa path;
  accumulate (+=); no 1/(4π).
- `src/helmholtz/workspace.rs`: the compact y-blocks and `rotation::Scratch`, two
  coefficient buffers, and the projection's accumulators, sized for p.
- `helmholtz/mod.rs`, `lib.rs` docs, `fmm-ref/CLAUDE.md`.

## Tests that prove it

`fmm-ref/tests/helmholtz/` (the existing executable), new `translation_*.rs` modules, f64
unless stated, single thread. Kinds of k as in CH1.3b (real, complex arg π/4, weak
Im k = Re k/10, imaginary), complex charges unless stated. A parent box of half-width 0.5
and its children of 0.25.

1. **Addition theorems** (`translation_series.rs`), RCR, against `p2p::direct_sum`,
   relative ℓ2 of potential and gradient (§9.1):
   - P2M (child) → M2M → M2P (parent): sources in a child box with its corners, targets at
     2.5 a_P;
   - P2L (parent) → L2L → L2P (child): sources at 2.5 a_P from the parent centre, targets in
     the child box with its corners and centre;
   - P2M → M2L → L2P between two boxes of one level at |t| = 5a in a general direction;
   - |k| a_P ∈ {0.5, 5, 30}, every kind: at p₀ ≤ 1e-12, at a lower pₜ above 1e-7, falling
     between (as CH1.3b's series).
2. **RCR against direct** (`translation_direct.rs`), FMM-weighted as CH1.2c: per degree the
   coefficient difference weighted by |fₙ(kR)| where the output is evaluated (a multipole
   at the nearest V-list points R = (4 − √3) r, a local on its sphere R = a), within 1e-12
   relative to the larger of the output's and the input's weighted size, and for real k
   relative to the output's alone. (Changed during the task from a comparison of fields:
   under strong damping an unconverged output's terms cancel, and a damped M2L's output is
   e^{−Im(k)(|t| − 2a)} times its input, so a field or output-relative measure reports the
   cancellation, not the operators; see "Open questions".)
   - M2M over the 8 children, L2L over the 8 children, M2L over offsets that include the
     nearest (2, 0, 0)w, the axis ±z (θ = 0, π), (3, 3, 3)w and four general ones; all 316
     offsets in a release-ignored test;
   - |k| a ∈ {0.5, 5, 30}, every kind; above the projection threshold (Re(k) a = 50, real and
     weakly damped k, release-ignored); Im(k) a = 100;
   - rectangular degrees (p_in < p_out and p_in > p_out).
3. **Rectangular blocks**: RCR with (p_in, p_out) equals the square RCR at max(p_in, p_out)
   on the zero-padded input, truncated, bit for bit.
4. **Real k**: RCR M2M and L2L of real-k expansions with an exactly zero imaginary plane keep
   it exactly zero below the threshold (the real coaxial step of §5.2).
5. **k → 0** (`translation_limit.rs`), RCR and direct, on CH1.3b's diagonal map of §3.4
   (multipoles k̂ⁿ 3^{−n/2} …, locals …/r per level): mapped Laplace coefficients through the
   Helmholtz translation against Laplace's `rotation::{m2m, l2l, m2l}`, per degree relative
   to the input's mapped magnitudes, at |k| a = 1e-2, 1e-4, 1e-6, every kind and Yukawa's
   real path: within |k| d for every path and (|k| d)² for real k's real parts (§9), d the
   translation's size (a_P, or |t| + a for the M2L), plus round-off (1e-14 RCR, 1e-11 for
   the projection, whose high-degree local coefficients carry amplified rounding at low
   frequency); every error falls with its power of k.
6. **Yukawa two ways** (`translation_yukawa.rs`): the complex RCR and direct at k = iκ
   against the real ones on real-charge inputs, κ a ∈ {0.5, 5, 50, 300}: M̂ = iⁿ M̂^Y,
   L̂ = i^{−n} L̂^Y, within 1e-13 in the FMM-weighted norm (bit for bit where the operations
   coincide, reported).
7. **Strong damping**: RCR at Im(k) a = 1000: every output finite.
8. **f32**: RCR at p = 8, s = 1 against f64 within 1e-5.
9. **Panics** (`panics.rs`): lengths, workspace degree, table degrees, a frame not of the
   level, non-parent-child frames for RCR M2M and L2L, M2L across levels, a Yukawa operator
   on a non-Yukawa level, the direct operators' geometry.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`, the
default and workspace tests); `cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`,
`cargo test -p nd-fmm-ref`, and `cargo test -p nd-fmm-ref --release -- --ignored`. No MPI
crate changes, no manifest changes.

## Machine runs

None.

## Constraints and risks

- Constraint 1 (Laplace): no Laplace code changes; at most a private helper of `nd-fmm-ref`
  becomes `pub(crate)`.
- Constraint 6 (trusted paths): the chains against CH1.3a's direct sum, RCR against the
  quadrature projection of the leaf operators' fields, the limit against Laplace's
  translations, Yukawa against the complex path.
- Risk: the direct projection costs O(Q² p²) per call (Q ≈ 2p + |k| a); the large cases
  (all 316 offsets, k a = 50) run in release only.
- Risk: the projection onto jₙ at low frequency amplifies rounding in the high-degree
  coefficients (jₙ(ka) ~ (ka)ⁿ/(2n + 1)!!); hence the comparison of fields, not
  coefficients, between direct and RCR.

## Pending runs

None.

## Open questions

None blocking. Findings for the PR:

- Under strong damping RCR's M2L output loses relative accuracy (about 1e-10 relative to
  itself at κ a = 30 and |t| = 3w, where e^{−Im(k)(|t| − 2a)} ≈ 1e-19): the coaxial sum
  Σₙ Ê_{n′n} M̂ₙ cancels. Both table generators (recursion and projection) agree with each
  other there, so the tables are not the cause. Relative to the input it stays at
  round-off, and §5.5's pruning removes such contributions; worth remembering for CH2.3's
  θ_D.
