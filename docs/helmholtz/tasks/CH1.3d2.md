# CH1.3d2 CONVENTIONS §3.15 revised

Started 2026-11-19 on `tbetcke/h-ch1-3d2`. Roadmap: "CH1.3d2 CONVENTIONS §3.15 revised" (H1;
the second part of CH1.3d, item 8 of the priority list). Proposal: §10 CH1.3; §2, §3.1, §3.3,
§3.4, §3.6, §4 item 7, §5.4; S6's report (`spikes/diagonal/SPIKE_REPORT.md`, "The proposal
for CONVENTIONS §3.15 'Signatures'", CH0.6's question 5); review 2026-11-12 M1–M4, M10
(`docs/reviews/2026-11-12-H6.md` §1.2, §3).

## Goal

CONVENTIONS §3.15 says what the code does: "Signatures" written from CH1.3d1's
`nd_fmm_ref::helmholtz::diagonal`, the stored expansions in level units as CH3.1a built them,
the versioning of its unversioned subsection and the Fourier counts made unambiguous, with
one `CONVENTION_VERSION` bump; and the proposal's §3.1 and §3.4 corrected to CONVENTIONS.

## Acceptance (roadmap *Done when*; proposal §10 has no row for it)

"The text proposed in the PR and signed off; the version in `nd-fmm-math` and the file
agree." With CH1.3d's text: M1, M3, M4, M10's §3.9; the proposal's §3.4 and §3.1 (M2), with
their Appendix A line; `tools/fixtures/check_3_15.py` extended to the signature identities.

## Read first

- `docs/CONVENTIONS.md` header, §3.9, §3.10, §3.13 ("Leaf-scaled coordinates"), §3.14,
  §3.15 (all of it).
- `fmm-ref/src/helmholtz/diagonal/{mod,grid,transfer,classes}.rs` (the module docs, "The
  chain", "Stored form"); `docs/helmholtz/tasks/CH1.3d1.md`.
- `fmm-tables/src/helmholtz.rs` ("Levels and units"), `fmm-exec/src/helmholtz/operator.rs`
  ("Levels and units"), `fmm-exec/src/helmholtz/densities.rs` ("Scaling");
  `docs/helmholtz/tasks/CH3.1a.md`, question 2.
- `fmm-tables/src/helmholtz/plane_wave/{quadrature,pair}.rs` (Fₖ: "orders |m| ≤ Fₖ"),
  `fmm-tables/src/plane_wave` (§3.14's Fₖ, a count).
- Proposal §3.1, §3.3, §3.4, §4, Appendix A.

## Findings checked against the tree (2026-11-19)

- **M1** holds: §3.15 states the absolute-unit forms only. As built, level l works in units
  of its half-width r_l (k̂_l = k r_l); a box holds r_l φ̃. The multipole coefficients
  Mₙᵐ = Σ q jₙ(kρ) Y are unit-free (kρ = k̂ρ̂), so they are the same numbers in any unit and
  the factor 4π ik̂ carries the change; a local scales with φ̃, so it holds r_l L. Hence L2L
  (parent's units to the child's) carries r_{l+1}/r_l = ½ and P2L, computed in the source
  leaf's units, scales its charges by r_t/r_s. s_l and β_l depend on k a only, the same in
  both units. Nothing else changes: §3.15's formulas hold verbatim in level units with k̂_l
  and a_l = √3.
- **M3** holds: the header and §3.10 version all of §3.15; "Exponential M2L" calls itself
  unversioned.
- **M4**: not a code defect but a notation clash. §3.14's Fₖ (FMM3D's Laplace tables) is a
  count, orders 0 ≤ m < Fₖ' = min(Fₖ, p + 1); §3.15's Fₖ (`Quadrature::fourier_counts`,
  FMM3D's Helmholtz `nfourier`, nexptot = Σ (2Fₖ + 1)) is the largest order, orders
  0 ≤ m ≤ Fₖ' = min(Fₖ, p). Both keep the orders the code keeps; the texts mean different
  things by one symbol.
- **M2** holds: proposal §3.4 has s_l = min(|k| r_l, 1) with the half-width (§3.15: the
  circumradius a_l), §3.1 says no coefficient carries ik (§3.15: 4π ik on M2L, M2P, P2L,
  and it is in Lₙᵐ).
- **M10's §3.9** holds: §3.9 does not say that Laplace's M2L above p = 20 is untested.

## Changes

Docs, one constant, the fixtures' headers, one check script. No code path, no public API.

1. `docs/CONVENTIONS.md` §3.15:
   - **"Signatures"** replaces "Reserved": the grid (2P − 1 Gauss–Legendre nodes ascending,
     N_φ = 4P − 3 rounded up to a multiple of 4, i.e. 4P; sample (i, j) at i N_φ + j; the
     direction; the weight wᵢ 2π/N_φ; exact to degree 4P − 4), the signature in stored form
     F̂(ŝ) = Σₙ (−i)ⁿ sⁿ Σₘ M̂ₙᵐ Yₙᵐ(ŝ) (4π F of charges = Σ q e^{−ik ŝ·(y − c)}), the
     transfer function T̂ = e^{Im z} Σ_{l ≤ 2p} (2l + 1) iˡ hₗ(z) Pₗ(ŝ · t̂) in extended
     precision rounded once, F → R (L̂ += sⁿ iⁿ Σ wᵢ (2π/N_φ) Yₙᵐ G), the M2L
     G = 4π ik e^{2β − Im(k)|t|} T̂ F̂, the identity behind it, the 34 classes and their
     sample permutations; the diagonal form square (one degree at both ends). The
     buffer layout (interleaved on the host, planes on the device, per-box alignment) is
     in-memory, CH4.1's, unversioned (question 3).
   - **"Level units"** (M1), a new subsection after "Radial functions and scaling": the
     forms hold in any unit of length with k → k r; multipoles unchanged, locals and φ̃
     scale by r; a box of level l stores them in units of r_l (k̂_l = k r_l), as §3.13's
     chunks; the consequences, L2L's ½ and P2L's r_t/r_s; the "Exponential M2L" bullet
     "Units" then cites it.
   - **Versioning** (M3) per question 1, in §3.15's opening, the file's header and §3.10.
   - **Fₖ** (M4) per question 2.
   - "Precision and range": f32 still reserved, D3's scope cited (signed off 2026-11-06,
     CH7.1 states the ranges); the tested ranges of `nd-fmm-math` (CH1.2), `nd-fmm-ref`'s
     translations (CH1.3c) and the diagonal form (CH1.3d1: k a 2.83–54, p 7–82, f64; f32 at
     k a ≈ 16).
   - "Verification": the new checks of `check_3_15.py`.
2. `docs/CONVENTIONS.md` §3.9 (M10): Laplace's M2L tested to p = 20 only (irregular
   harmonics to degree 40); M2L above it untested. §3.9 is unversioned.
3. `CONVENTION_VERSION` 3 → 4: the header and §3.10's line "Version 4 (Helmholtz CH1.3d2)";
   `fmm-math/src/lib.rs`; the fixtures regenerated by their generators, headers only
   (`harmonics_{A,B,C}.json`, `helmholtz_{angular,radial,coaxial}.json`, `dipoles.json`,
   `helmholtz_fmm3d.json`), each with `--check` after; `tools/fixtures/crosscheck_scipy.py`'s
   docstring if it names the version.
4. `tools/fixtures/check_3_15.py`, new checks in mpmath at 40 digits:
   - the plane-wave expansion e^{ik ŝ·r} = 4π Σ iⁿ jₙ(kr) Σₘ Yₙᵐ(r̂) Yₙᵐ(ŝ), real and
     complex k;
   - the signature of charges: 4π F(ŝ) against Σ q e^{−ik ŝ·(y − c)} within the truncation;
   - the identity e^{ik|x−y|}/|x−y| = (ik/4π) ∫ e^{ik ŝ·(x−c_t)} T(ŝ) e^{−ik ŝ·(y−c_s)} dŝ
     (the truncated T at large L) on a few pairs, and the diagonal M2L on the grid against
     the (S|R) M2L of "Coaxial translations" at low degree, in stored form;
   - the grid's quadrature exact to degree 4P − 4 (small P);
   - the sample permutations: for each symmetry, the permuted direction is S ŝ exactly;
   - the level units: L2L in level units equals ½ times the absolute-unit translation, P2L
     scaled by r_t/r_s, P2M → M2L → L2P in level units against the direct sum.
5. Proposal (M2): §3.4 to the circumradius a_l (amended at CH1.3d2), §3.1 to the factor
   4π ik on M2L, M2P and P2L, carried by Lₙᵐ (amended at CH1.3d2); §3.3/§4 item 7's
   signature storage sentence if question 3 changes it; Appendix A: "Amended at CH1.3d2".
6. Citations of the old "Signatures: Reserved" (`grep`), e.g. `fmm-ref`'s diagonal module
   docs ("No convention changed"), updated to cite §3.15 "Signatures"; `fmm-ref/CLAUDE.md`
   if it says the signatures are unfixed.

## Tests that prove it

- `check_3_15.py` (all old checks and the new ones above) passes; the run printed in the PR.
- `check_3_15_plane_wave.py` passes (it reads §3.15's notation).
- Every regenerated fixture's `--check` passes and its diff is the header line alone.
- `fmm-math/tests/convention_version.rs` (the file and the constant agree), every fixture
  test, `fmm-tables`' cache tests (version in the key).
- The workspace tests unchanged otherwise (no code path changed).

## Checks

`/verify` (fmt, Clippy default/examples/workspace, docs, default and workspace tests at
opt-level 1). The MPI list: not needed (no MPI code changed) unless `/verify` says so.
No manifest (no `cargo deny`).

## Machine runs

None.

## Constraints and risks

- Constraint 1: no Laplace code path changes; only the version constant. Cached Laplace
  tables and tuning files of version 3 are rejected once and rebuilt (as at CH1.1). The
  tables come out identical.
- CONVENTIONS changes: this task is the proposal; the PR asks for sign-off of the text.
- Risk: the fixtures' generators are slow (`gen_harmonics.py` about 5 minutes;
  `gen_helmholtz_fmm3d.py` needs FMM3D at `../fmm3d`, `tools/fmm3d/README.md`).

## Pending runs

None planned.

## Open questions

Signed off 2026-11-19 at the plan (the user, all four as recommended):

1. M3, the versioning of "Exponential M2L": "Exempt in place (Recommended)": it stays where
   it is, and the header, §3.10 and §3.15's opening version §3.15 except that subsection, as
   §3.14.
2. M4, Fₖ: "Keep, say so (Recommended)": §3.15's Fₖ stays the largest order kept (the code's
   `fourier_counts`), stated explicitly beside §3.14's count.
3. The signature buffers: "Leave layout to CH4.1 (Recommended)": the mathematics and the
   sample order versioned; the buffers' layout in memory fixed by CH4.1, unversioned.
4. The plan and the bump to version 4: "Go ahead (Recommended)".
