# CH0.7 S7: truncation under realistic geometry

Started 2026-10-28 on `tbetcke/h-ch0-7`. Roadmap: "CH0.7 S7: truncation under realistic
geometry" (H0; priority list item 2). Proposal: §10 CH0.7; §3.1, §3.4, §3.5, §3.7, §5.1,
§5.2, §5.3, §5.5, §5.6, §9 item 5, §8 R6.

## Goal

Find, under realistic geometry and with every rounding error out of the way, the smallest
degree p that meets a tolerance ε, as a function of (k a, ε) and the kind of operator,
and compare it with the formula of proposal §3.5.

## Acceptance (proposal §10, verbatim)

> the chain P2M → M2M → M2L → L2P over two levels, and M2M and L2L chains over three, on
> all 316 offsets, with points filling the boxes (uniform, corners, W1 patches), for real,
> complex and imaginary k, in extended precision with the S1 and S2 prototypes, so that
> only truncation is measured. Output: p as a function of (k a, ε, operator) on the low-
> and high-frequency ranges against the formula of Section 3.5, meeting ε with a stated
> margin on every placement; fed to S6, S8, Section 7 and CH2.3, CH3.3, CH4.3

(The roadmap's form: "P2M → M2M → M2L → L2L → L2P over two levels"; the chain has L2L as
well.)

## Read first

- Proposal §3.1 (the expansion, ik and 1/(4π)), §3.4 and the CH0.3 sign-offs (scale
  s_l = min(|k| a_l, 1), a_l the circumradius), §3.5 (the formula), §3.7 (leaves),
  §5.1–§5.3, §5.5 (Yukawa pruning, which bounds the useful Im k a), §9.1 (norms).
- `spikes/rotation-blocks/` (S1: `src/gd.rs`, `src/legendre.rs`; the basis and Y(Qx) =
  O Y(x)), `spikes/coaxial/` (S2: `src/coaxial.rs`, `src/projection.rs`, the basis, the
  FMM-weighted metric, the sign-off "recursion where Re(k) a ≤ 40, projection above"),
  `spikes/radial/` (S3), `spikes/census/src/model.rs` (`degree`, the formula p).
- Their fixtures, which the tests reuse as independent references.

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- `spikes/truncation/` (new, package `nd-fmm-spike-truncation`), a workspace member
  outside `default-members` like S1–S3 and S8; dev-dependency `serde_json` (already in
  `[workspace.dependencies]`); no new dependency.
  - `src/real.rs`: a `Real` trait (the arithmetic, `sqrt`, `exp`, `sin_cos`, f64
    conversion) for `f64` and for `Dd`, a double-double type written here (≈ 32 digits,
    error-free transformations with `mul_add`; `exp`, `sin_cos` by argument reduction and
    Taylor series). `src/complex.rs`: `Complex<T: Real>`.
  - Generic ports of the prototypes, so that the same code runs in f64 and in
    double-double: `src/radial.rs` (S2/S3's jₙ by Miller's method and hₙ upward, with the
    power scale; derivatives from neighbouring orders), `src/legendre.rs` and
    `src/rotation.rs` (S1's GD y-blocks with the telescoped Legendre functions; the z
    rotation; applying O(Q) = O(R_y(−β)) O(R_z(−α)) to a real or complex expansion),
    `src/coaxial.rs` (S2's recursion in its triangle, and S2's projection), and
    `src/harmonics.rs` (the real orthonormal Yₙᵐ of S2's basis with their gradients).
  - `src/chain.rs`: P2M, M2M, M2L and L2L by rotation, coaxial step, rotation back (RCR),
    L2P with gradient, P2L and M2P by direct evaluation, the direct sum; the expansions
    scaled by s_l per level (multipoles divided by sₙ, locals multiplied, the M2L table
    scaled s^{n+n′} as in S2). Constants: G = e^{ik r}/(4π r) = ik Σ jₙ hₙ Σₘ Y Y.
  - `src/geometry.rs`: the 316 V-list offsets, children and grandchildren, the three
    placements (uniform in the box; clusters in the eight corner cubes of side w_leaf/8 of
    every leaf box; W1 patches, points on a sphere of radius R ∈ {2w, 8w} through the box
    at a random orientation), the random complex charges of §9.1, seeded.
  - `src/calibrate.rs`: the error of a chain per offset and placement (relative ℓ2 over the
    targets of the target box, potential and gradient separately, §9.1), the search for
    the smallest p and the formula p of §3.5 (copied from S8's `degree`).
  - `tests/`, `examples/sweep.rs` (the calibration, writing `results/*.csv` and
    `results/*.md`), `examples/time.rs`, `SPIKE_REPORT.md`.
- Root `Cargo.toml` `members` gains `spikes/truncation`; `Cargo.lock` its entry.

### The chains

Box width w at the M2L level l (circumradius a = √3 w/2); k a is k a_l. The kinds of k:
real, complex k = κ(1 + iη) with η = 0.1 and η = 1, and imaginary k = iκ, k a meaning
κ a (S2's set). The expansion of §3.1 in the real orthonormal basis of S2, scaled per
level as above.

1. **C1, one level**: sources fill box S, P2M at p about its centre, M2L by RCR to box T at
   each of the 316 offsets, L2P at the targets of T. Gives p₁(k a, ε, kind), the base
   calibration of the M2L.
2. **C2, two levels (the acceptance's chain)**: sources fill S; P2M in each of its eight
   children at p_{l+1}, M2M to S at p_l, M2L to T, L2L to T's eight children, L2P. With
   p_l = p₁(k a_l) + δ on both levels; reports the smallest δ ≥ 0 that meets ε.
3. **C3m, M2M over three levels**: P2M in S's 64 grandchildren, M2M twice, then the
   truncated multipole of S evaluated directly at the targets of each offset box (M2P).
4. **C3l, L2L over three levels**: the truncated local of S formed directly from the
   sources of each offset box (P2L), L2L twice, L2P in S's 64 grandchildren.

C3m and C3l report δ as C2. Placements are generated relative to each chain's leaf boxes
(where P2M and L2P act). The direct sum and every operator run in double-double; f64 runs
the same code at the calibrated p for comparison (the f64 floor, an input to S6 and D3).

### The sweep

- k a on the half-octave grid 2^{j/2}, j = −12..16 (0.016 to 256), so that k a/2 and
  k a/4 lie on the grid. The top of the range covers W1's level 2 at kD = 1000 (k a = 216,
  S8).
- Real k and η = 0.1 over the whole grid; η = 1 and imaginary k to κ a = 32, beyond the
  pruning cut of §5.5 (e^{−Im(k) w} below 10⁻¹² there), where a damped level no longer
  carries a V list.
- ε = 10⁻³ to 10⁻¹⁰ by decades.
- **The calibrated p** of (k a, ε, kind) is the smallest p whose worst error, over the 316
  offsets, the three placements, potential and gradient, is at most ε/2: a margin of 2,
  for the sum over a target's V lists and levels that the full FMM adds and CH3.3
  confirms. The report also gives p for the potential alone and the margin reached.
- The search runs on the offsets with |t| ≤ √6 w (54 of them, where the error is
  largest), then every chosen p is checked on all 316 offsets and raised until it passes.
- Coaxial tables by S2's recursion where Re(k) a ≤ 128 in double-double (≤ 40 in f64, the
  signed-off rule), by projection above; the end-to-end test below bounds what either
  leaves.
- Output: tables of p₁ and δ per (kind, k a, ε) beside the formula; a fit of the form
  p = k a + C(ε) (k a)^{1/3} on the high-frequency range and the low-frequency floor, for
  S6, S8 (CH0.8b), §7 and CH2.3, CH3.3, CH4.3.

If the sweep at the top of the grid runs too long on the M3 Max (C2, C3m and C3l cost
about 16 RCR translations per offset), the chains are run there at fewer ε (10⁻³, 10⁻⁶,
10⁻⁸, 10⁻¹⁰) or the run moves to Kathleen; the report says which.

## Tests that prove it

`cargo test -p nd-fmm-spike-truncation --release -- --include-ignored`, one thread per
test unless stated.

1. **Double-double**: sin² + cos² = 1, exp(x) exp(−x) = 1, √x² = x, π, e and ln 2 against
   their decimal values, to ≤ 10⁻³⁰ relative; the f64 path of every function within
   10⁻¹⁵ of `std`.
2. **Radial functions** in double-double: the cross-product (Wronskian) identities to
   ≤ 10⁻²⁸ relative over real, complex and imaginary z with |z| from 10⁻³ to 2000 and
   orders to 700; Miller's start raised gives the same values to 10⁻³⁰; against S3's
   mpmath fixture (`spikes/radial/fixtures/reference.json`, f64 values) to ≤ 10⁻¹⁴.
3. **Harmonics**: orthonormality by quadrature and the addition theorem
   Σₘ Yₙᵐ(x̂) Yₙᵐ(ŷ) = (2n + 1)/(4π) Pₙ(x̂ · ŷ) to ≤ 10⁻²⁸; the gradients against
   central differences.
4. **Rotations**: S1's fixture (f64, ≤ 10⁻¹⁴); orthogonality to ≤ 10⁻²⁸ at n ≤ 300; the
   definition Y(Qx) = O(Q) Y(x) to ≤ 10⁻²⁸ for the 316 offset directions and the eight
   child shifts at n ≤ 20.
5. **Coaxial tables**: S2's fixture (FMM-weighted error ≤ 10⁻¹¹, S2's metric); the
   double-double recursion against the double-double projection at k a = 128 (real and
   η = 0.1) and at k a ≤ 32 (every kind), FMM-weighted ≤ 10⁻²⁴; rectangular truncation a
   block of the square table.
6. **Only truncation is measured** (the acceptance's "extended precision"): each chain
   (C1, C2, C3m, C3l) at a p far above the calibrated one against the double-double direct
   sum, relative error ≤ 10⁻²⁴ for potential and gradient, at k a = 0.1, 4, 32 for every
   kind and at k a = 128 and 256 for real k and η = 0.1, on the nearest and farthest
   offsets and every placement. So the rounding of the chain is at least 10⁻¹⁴ below every
   ε of the sweep.
7. **Geometry**: 316 offsets, 15 distances; placements inside their boxes; corner
   clusters inside each leaf; patch points on their sphere.
8. **Search**: on a synthetic error curve the search returns the smallest passing p;
   the formula p reproduces proposal §3.5's tables.

## Checks

`/verify` (no default member changes: format, Clippy of the workspace with all targets, the
default and workspace tests, `cargo doc`, the licence check after the manifest change);
the spike's Clippy and tests in release with `--include-ignored`.

## Machine runs

The sweep on the M3 Max (release, all cores; no timing is a result, only the p tables).
Kathleen only if the top of the grid needs it (above), as a Slurm job of one node.

## Constraints and risks

- Constraint 1 (Laplace): untouched; only the root `Cargo.toml` (`members`) and
  `Cargo.lock` (the spike's entry) are shared files.
- Constraint 9: the calibrated p depends on k a, ε and the kind only.
- §3.15 is not written (CH1.1); the spike uses S2's basis and the scale signed off in
  CH0.3, and says so.
- Risk: double-double speed at p ≈ 330 (k a = 256). Mitigation above.
- Risk: S2's recursion in double-double beyond k a ≈ 128 (its f64 instability grows like
  e^{0.2 k a}); the projection takes over and test 5 checks the hand-over.
- Interpretation: "operator" in "p as a function of (k a, ε, operator)" is read as the
  kind of k (Helmholtz real, complex, Yukawa), since §3.5 calibrates "as a function of
  (k a_l, ε) and the operator"; the translation operators' extra needs are δ of C2, C3m
  and C3l.

## Pending runs

None yet.

## Open questions

None yet.
