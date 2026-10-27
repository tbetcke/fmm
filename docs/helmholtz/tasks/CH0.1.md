# CH0.1 S1: rotation blocks at large degree

Started 2026-10-27 on `tbetcke/h-ch0-1`. Roadmap: "CH0.1 S1: rotation blocks at large
degree" (H0). Proposal: §10 CH0.1; §3.2, §5.2, §7 ("Rotation blocks"), §8 R1, §4 item 5.

## Goal

Show that the Gumerov–Duraiswami (2014) recursion gives orthogonal rotation blocks in the
orthonormal real basis that are accurate to degree 1000 in f64, how far the existing
Racah-basis blocks of `nd_fmm_math::rotation::blocks` conjugated by N stay accurate, how
the blocks of β and π − β are related, and what a block set costs.

## Acceptance (proposal §10, verbatim)

> orthogonality and agreement with an arbitrary-precision reference to p = 1000; agreement
> with N-conjugated `nd_fmm_math::rotation::blocks` at p ≤ 30; the degree up to which the
> conjugated blocks stay accurate; the relation between the blocks of β and π − β; time
> per block set

## Read first

- CONVENTIONS §3.3 (Racah harmonics, no Condon–Shortley phase), §3.6 (real storage),
  §3.8 (blocks, N, z-y-z), §3.11 "Rotation of coefficients".
- `fmm-math/src/rotation.rs` (`blocks`, `euler_zyz`; the Ivanic–Ruedenberg chain rule in
  the Racah basis), `fmm-tables/src/rotation.rs` (what a block set per polar angle is:
  y-rotations R_y(∓θ) per polar angle, z-rotations per azimuth).
- Gumerov and Duraiswami, *Recursive computation of spherical harmonic rotation
  coefficients of large degree* (2014), arXiv:1403.7698.

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- `spikes/rotation-blocks/` (new, package `nd-fmm-spike-rotation-blocks`): a member of
  the workspace outside `default-members`, like the other spikes (proposal §6.1), so its
  entry enters the root `Cargo.lock`. Dependencies: `nd-fmm-math` and `serde_json`, both
  already in `[workspace.dependencies]`; no new dependency.
  - `src/legendre.rs`: P̄ₙᵐ = √((n − m)!/(n + m)!) Pₙᵐ(cos β) for n ≤ p + 1 by the stable
    recursion in n, with the start values carried with a separate binary exponent so
    that sinᵐ β does not underflow at m ≈ 1000.
  - `src/gd.rs`: the y-rotation block of degree n in the orthonormal complex basis
    Ŷₙᵐ = √((n + |m|)!(n − |m|)!) Rₙᵐ (real for a y-rotation): row 0 from P̄ at degree n,
    row 1 from row 0 at degree n + 1 (the ladder of CONVENTIONS §3.4 applied in the
    rotated frame, no division by sin β), then rows 2 … n by the three-term relation that
    J_y commutes with the rotation (Gumerov–Duraiswami's step 3), rows m′ < 0 by symmetry.
    Then the real orthonormal block (the real storage of N Rₙ, the basis proposed for
    §3.15 item 2): a y-rotation splits it into a cos block of (n + 1)² and a sin block of
    n² entries. Full blocks of R_z(α) R_y(β) R_z(γ) by the z-rotations, for the
    comparison with `nd_fmm_math`.
  - `src/conjugated.rs`: N Dⁿ N⁻¹ from `nd_fmm_math::rotation::blocks`, N formed as
    ratios so no factorial overflows.
  - `tools/reference.py` (PEP 723, `mpmath==1.4.1` pinned as in `tools/fixtures/`): the
    reference entries from the explicit Wigner sum in arbitrary precision (digits about
    0.61 n + 40, each value recomputed with 50 more digits and required to agree to
    10⁻²⁵), mapped to the real orthonormal block; the mapping itself checked at n ≤ 6
    against the definition Yₙ(Qx) = O Yₙ(x) with harmonics evaluated in mpmath from
    CONVENTIONS §3.3. Writes `fixtures/reference.json` (committed, kept under 1 MB):
    sampled entries (rows and columns 0 and ±n, the diagonals, random entries) at
    n ∈ {10, 30, 50, 100, 200, 300, 500, 700, 1000} for polar angles of the octree
    (arccos(1/√3) of M2M, its π − β, atan(1/3) the smallest off-axis V-list angle, π/2)
    and two generic angles (1.234 and 10⁻³).
  - `tests/accuracy.rs`, `examples/{sweep,time}.rs`, `SPIKE_REPORT.md`.
- Root `Cargo.toml` `members` gains `spikes/rotation-blocks`.

## Tests that prove it

All in f64, one thread, `cargo test -p nd-fmm-spike-rotation-blocks --release -- --include-ignored`.

1. **Reference to p = 1000**: every sampled entry against `fixtures/reference.json`,
   absolute error ≤ 10⁻¹² (entries are of order one; the measured maximum is reported per
   degree and angle).
2. **Orthogonality**: ‖O Oᵀ − I‖_max ≤ 10⁻¹² for every degree n ≤ 100 and at
   n ∈ {150, 200, 300, 500, 700, 1000} for each angle; at every degree n ≤ 1000 the probe
   ‖Oᵀ(O v) − v‖_∞ for a random v.
3. **Against `nd_fmm_math` at p ≤ 30**: the GD full blocks of euler_zyz(α, β, γ) for
   several angle triples (and the pure y-rotations) against N Dⁿ N⁻¹ from
   `rotation::blocks`, absolute ≤ 10⁻¹³, which pins the basis, phase and orientation.
4. **Definition at low degree**: Yₙ(Qx) = O Yₙ(x) with Yₙ = N Rₙ from
   `nd_fmm_math::harmonics::regular` at random points, n ≤ 20, relative 10⁻¹³.
5. **β and π − β**: the identity found (expected a signed permutation of columns,
   Oⁿ(π − β) = Oⁿ(β) Pₙ with Pₙ diagonal ±1 composed with m ↦ −m on the sin block, from
   d^n_{m′m}(π − β) = (−1)^{n+m′} d^n_{m′,−m}(β)), checked bit for bit or to round-off on
   every degree ≤ 1000 for the octree's angles, and stated in the report.
6. **Group property** at n = 500: O(R_y(β₁)) O(R_y(β₂)) = O(R_y(β₁ + β₂)) to 10⁻¹².

## Checks

`/verify` (no default member changes: format, Clippy of the workspace with all targets,
the spike's own Clippy and tests); `uv run spikes/rotation-blocks/tools/reference.py
--check` regenerates the fixture byte for byte.

## Machine runs

None required. Times per block set on the M3 Max (release, one thread), for p ∈ {8, 20,
33, 50, 82, 144, 261, 500, 1000}, the GD recursion against `rotation::blocks` plus
conjugation where that is accurate; these feed §7's "generated per level call above a
degree" rule. locust and Kathleen times are not part of the acceptance.

## Constraints and risks

- Constraint 1 (Laplace): untouched; no default member, table or layout changes. The
  only shared file is the root `Cargo.toml` (`members`) and `Cargo.lock` (the spike's
  entry).
- §3.15 is not written yet (CH1.4, CH1.1): the spike fixes the basis as the real storage
  of N Rₙ (no Condon–Shortley phase, cos in +m, sin in −m), which §3.2 implies; the
  report says so, the choice stays CH1.1's.
- Risk: the in-degree recursion run over all columns may lose accuracy where GD restrict
  it to a triangle and fill by symmetry; the test decides, and the report names the form
  used.
- Risk: the reference's cost at n = 1000 (sums of up to 2001 terms at about 650 digits);
  sampled entries only.

## Pending runs

None.

## Open questions

None yet.
