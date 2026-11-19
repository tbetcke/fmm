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
     scaled by r_t/r_s. (As built: P2M the same numbers and P2L r times them in units r,
     M2P and L2P giving r φ̃, the L2L chain with ½ and P2L from a coarser leaf against the
     direct sum. No separate M2L in level units: an M2L stays within one level's units,
     which check 9 and check 17 cover.)
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

Ran on the M6: `cargo fmt --all -- --check`, `cargo clippy -- -D warnings`, `cargo clippy
--examples -- -D warnings`, `cargo doc --no-deps`, `cargo clippy --workspace --all-targets --
-D warnings` (all pass, again after the self-review's doc edits); at opt-level 1
(`CARGO_TARGET_DIR=target/opt1`, `HWLOC_COMPONENTS=-darwin`) `cargo test` 1,302 passed and
`cargo test --workspace` 1,346 passed, none failed. Not run: the MPI list (no code path
changed; doc comments and the version constant only), `cargo deny` (no manifest).

## Machine runs

None.

## Constraints and risks

- Constraint 1: no Laplace code path changes; only the version constant. Cached Laplace
  tables and tuning files of version 3 are rejected once and rebuilt (as at CH1.1). The
  tables come out identical.
- CONVENTIONS changes: this task is the proposal; the PR asks for sign-off of the text.
- Risk: the fixtures' generators are slow (`gen_harmonics.py` about 5 minutes;
  `gen_helmholtz_fmm3d.py` needs FMM3D at `../fmm3d`, `tools/fmm3d/README.md`).

## Results

Run on the M6 on 2026-11-19 (load 3–5).

- `tools/fixtures/check_3_15.py`: all 168 checks pass (105 s). The new ones: the plane-wave
  expansion to 10⁻⁴⁰; the signature of charges to 10⁻³⁴ (k a 0.4 and 2, so s < 1 and s = 1);
  the grid's Gram matrix to 3·10⁻⁴⁰ (p = 3) and N_φ = 4P for p ≤ 8; the 34 classes and the 16
  symmetries' sample maps to 4·10⁻⁴¹ (p = 2); the diagonal M2L equal to check 9's (S|R) M2L on
  the axis to 1.4·10⁻²⁸ (p = 6, real, complex and imaginary k, real and complex charges), and
  off the axis within 1.5·10⁻¹⁰ of the direct sum (truncation); the level units: P2M and P2L to
  10⁻³⁹, M2P and L2P to 10⁻²², L2L with ½ to 2.6·10⁻¹⁴ (Yukawa; the truncation at degree 14),
  P2L from a coarser leaf to 10⁻²².
- `tools/fixtures/check_3_15_plane_wave.py`: all pass.
- The eight fixtures regenerated by their generators: each diff the header's
  `convention_version` alone (3 → 4; compared with the digit masked); every `--check` passes
  (angular, radial, coaxial byte for byte, dipoles, harmonics, FMM3D to 10⁻¹³ and
  byte-identical); `crosscheck_scipy.py` passes.

## Self-review

A subagent (the plan, §2, §10, the review findings, the branch): mathematics checked against
`diagonal/`, the tables' and operator's level units and the Fourier counts, all agree; checks
13–18 not vacuous. Taken: N_φ fixed at 4P contradicted proposal §6.5/§6.7 (N_φ ≥ 4P − 3 a
per-level choice for FFT sizes): now any multiple of 4 at least 4P − 3, 4P the reference's (a
question for sign-off); how T̂ is computed (recursions or interpolation, proposal §5.4) moved to
"Not fixed here"; "to 10⁻²⁸" → the check's 10⁻²⁵; S6's range 2.83–400; Yukawa added to checks
13, 14 and 18 and k a = 0.4 to check 14 (s < 1); the k̂_l formation detail dropped from the
versioned text; two long doc lines rewrapped; a verbless sentence. The roadmap's status line
reworded (H1's tasks done once the PR merges). Left: checks 15 and 16 at p = 3 and 2 only
(`nd-fmm-ref` tests the grid to p = 10 and the 316 offsets' permutations at p = 16).

## Pending runs

None.

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

The PR's question 1, the §3.15 text as written (N_φ a per-level multiple of 4, ≥ 4P − 3), signed
off 2026-11-19 by the user on #229: "All signed off. Merge when green".
