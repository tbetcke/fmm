# CH3.2b Classes

Started 2026-11-02 on `tbetcke/h-ch3-2b`. Roadmap: "CH3.2b Classes" (H3; CH3.2 split on
2026-11-02 into a/b/c; named by the user). Proposal: §10 CH3.2; §5.3 (Classes), §6.2,
§6.4, §7; §2 (constraints 1, 2, 5, 6, 7, 9).

## Goal

A Classes M2L for the Helmholtz operator: per level the 16 class matrices of CONVENTIONS
§3.12 in CH3.2a's dense layout, and the cube-group coefficient transforms in the
orthonormal basis of §3.15, signed permutations for the 16 elements that fix the z-axis and
dense per-degree blocks for the 32 that move it, applied pair by pair on the unchanged
shared drivers.

## Acceptance (proposal §10, verbatim)

CH3.2: "measured against Rotation per p, including the cost of the cube-group transforms
that move the z-axis". The roadmap's *Done when* of CH3.2b: against Dense to rounding at
every offset; the FMM as CH3.2a's (against the direct sum and against Rotation's output,
the same at every thread count and on P ranks the one-rank FMM bit for bit); the cost per
pair against Rotation and Dense per p on the M6, the transforms that move the z-axis
included.

## Read first

- docs/CONVENTIONS.md §3.12 ("The cube symmetry group", "Coefficients under improper P",
  "Operator identities", "Symmetry classes", "The z-axis elements") and §3.15 ("Angular
  basis", "Rotations").
- `fmm-tables/src/symmetry.rs` (`SignedPermutation`, `class_of`, Laplace's
  `CoefficientTransform`), `fmm-tables/src/m2l.rs` (`M2lClasses::apply`, the Laplace
  chain), `fmm-tables/src/helmholtz/dense.rs` (CH3.2a), `nd_fmm_math::orthonormal::rotation`
  (`y_blocks`, the compact layout).
- `fmm-exec/src/helmholtz/operator.rs` (`Kernel::m2l`, `with_m2l`, `Scratch`),
  `fmm-exec/src/helmholtz/fmm.rs` (the builder), `fmm-exec/tests/helmholtz/dense.rs`,
  `fmm-exec/tests/helmholtz_fmm.rs` (`check_dense`).
- `fmm-validate/examples/helmholtz_m2l.rs`, `fmm-validate/src/helmholtz_accuracy.rs`.

## The transforms (no convention change)

The Helmholtz coefficients of degree n are those of **Y**_n times a radial function of
|ρ| (§3.15, "Radial functions and scaling"), so for sources y ↦ c + P(y − c) multipoles and
locals alike become Oⁿ(P) times the old ones, with **Y**_n(P x̂) = Oⁿ(P) **Y**_n(x̂). For a
rotation Oⁿ is §3.15's block; for improper P, parity **Y**_n(−x̂) = (−1)ⁿ **Y**_n(x̂) gives
Oⁿ(P) = (−1)ⁿ Oⁿ(−P), §3.12's rule for Dⁿ. It is N Dⁿ(P) N⁻¹ with §3.12's Dⁿ, so this
follows from §3.12 and §3.15 as they stand; T_M = T_L = Oⁿ, orthogonal, Oⁿ(Pᵀ) = Oⁿ(P)ᵀ.
The stored form's factors (s_l^n, e^{±β_l}, 4π ik and the M2L's e^{2β − Im(k)|t|}) are
per degree or depend on |t| only, so they commute with Oⁿ. Hence

M2L(d) = O(P) M2L(d₀) O(P)ᵀ, with d = P d₀ and P the element of §3.12's rule.

Construction, exact wherever the entries are 0 and ±1:

- A proper Q is factored as R_z(α) R_y(β) R_z(γ) with β ∈ {0, π/2, π} and α, γ quarter
  turns, found by matching the integer matrices (§3.8's z-y-z form). Oⁿ(R_z(α)) for a
  quarter turn is a signed permutation (cos mα, sin mα ∈ {0, ±1}, formed in integers);
  Oⁿ(R_y(π)) = (−1)ⁿ F (§3.15's π − β rule at β = 0); Oⁿ(R_y(π/2)) is `y_blocks` at
  x = 0, s = 1 exactly (the Gumerov–Duraiswami generator of §3.15). The product picks
  entries and signs, so it adds no rounding.
- The 16 elements with π(2) = 2 (β ∈ {0, π}) come out as signed permutations of the slots,
  with the structure of §3.12's "The z-axis elements"; they are stored as one (source slot,
  sign) per slot. The 32 others are dense per-degree blocks (as built: all 32 stored, an
  improper element's blocks those of −P with the degree's sign (−1)ⁿ folded in, exact; the
  plan had stored the 16 proper ones only and applied the sign per degree).
- One set of transforms at the largest degree of the class levels serves every level:
  block n does not depend on p (`block_range`).

## Changes

`nd-fmm-tables` (a submodule of `helmholtz`), `nd-fmm-exec` (the Helmholtz modules) and
`nd-fmm-validate` (the example and an option). No Laplace file (Laplace's `symmetry` is used,
not changed), no driver, no `nd-fmm-plan`, no convention, no manifest dependency.

- `nd_fmm_tables::helmholtz::classes` (new `fmm-tables/src/helmholtz/classes.rs`):
  - `CubeTransforms<T>`: the 48 Oⁿ(P), n ≤ p (above); `apply(element, x, out)` (out += O x)
    on one plane of (p_l + 1)² ≤ (p + 1)² values; per slot or per degree block row in
    increasing column order, each product and sum rounded apart; `is_permutation(element)`,
    `block(element, n)` (the dense block, for the tests), `bytes_for(p)`. As built, Oᵀ is
    applied as O(Pᵀ) (Oⁿ(Pᵀ) = Oⁿ(P)ᵀ to rounding; as Laplace's `M2lClasses` takes T(Pᵀ)), so
    there is no `apply_transpose`.
  - `ClassM2l<T>`: per level with Classes the 16 matrices of the representatives, each
    CH3.2a's 2n × n [A_r; A_i] of that offset (built by `DenseM2l`'s rule from
    `HelmholtzTables<f64>`, so bit for bit `DenseM2l`'s matrix of the representative), plus
    the transforms at the largest chosen degree. `level_bytes(p)` = 16 · 2 (p + 1)⁴ · size
    of T; `select(degrees, budget)` from the finest level up as `DenseM2l::select`, the
    transforms' bytes charged as the largest chosen degree grows (constraint 9: degrees and
    budget only). `build(k_root, degrees, levels)`, `is_class(level)`, `matrix(level,
    class)`, `transforms()`, `bytes()`, `degree(level)`.
  - `m2l(level, d, multipole, planes, local, vectors, scratch)`: per vector, the multipole's
    planes rotated by O(P)ᵀ into scratch, the class matrix applied by Dense's chain (one or
    two planes) into a zeroed scratch local, then O(P) of each plane added to the local.
    `ClassScratch<T>` (caller-owned, sized at the largest degree, 4 (p + 1)² values).
- `nd_fmm_exec::helmholtz`: `HelmholtzM2l::Classes`; under it `with_m2l` builds
  `ClassM2l` on the levels of `select` within the same budget (`DEFAULT_DENSE_BUDGET`,
  the docs saying it covers both), RCR elsewhere; `m2l_on`, a `class_bytes` (or
  `dense_bytes` covering both, whichever reads better); the thread scratch gains the class
  scratch. `HelmholtzFmmBuilder::m2l(Classes)` with `dense_budget`. Docs: the operator
  table, "M2L"; `fmm-exec/CLAUDE.md` and `fmm-tables/CLAUDE.md`.
- `nd-fmm-validate`: `helmholtz_m2l` gains Classes per pair (all 316 offsets, and split
  into the offsets whose element fixes the z-axis and those whose element moves it, and
  the transforms alone), the class tables' bytes and build time; `helmholtz_accuracy
  --m2l classes`. Report `fmm-validate/results/helmholtz-ch3-2b.md`.

## Tests that prove it

- `fmm-tables/tests/helmholtz/classes.rs`:
  - the transforms: **Y**_n(P x̂) = Oⁿ(P) **Y**_n(x̂) for all 48 P at seeded points, p = 12,
    against `nd_fmm_math::orthonormal::harmonics` (1e-14 absolute; values of order 1);
    orthogonality O Oᵀ = I and the homomorphism O(P₁P₂) = O(P₁) O(P₂) on all pairs at
    p = 8 (1e-14); the 16 z-axis elements signed permutations with entries exactly 0, ±1
    mapping each slot pair (±m) to itself, the elements g < 8 diagonal; O(−I) = (−1)ⁿ
    exactly; O(Pᵀ) after O(P) the identity (as built: a round trip through `transpose()`);
  - ClassM2l against DenseM2l on every offset of levels 2–4 for the K_ROOTS (real and
    complex k̂), one and two planes, in the FMM-weighted measure within `BOUND` times the
    conditioning (as CH3.2a's test against Rotation);
  - the class matrices bit for bit `DenseM2l::matrix` of the representatives; a
    representative's M2L into a zero local bit for bit Dense's (identity transforms);
  - each vector the one-vector application; one plane against two with Im = 0; f32 the f64
    tables rounded; `select` against a hand count; refusals (levels 0 and 1, a level without
    tables, lengths, a scratch of too small a degree).
- `fmm-exec/tests/helmholtz/` (`dense.rs` extended, or `classes.rs`): the operator under
  `Classes` pair by pair against `Rotation` and `Dense` on every offset of a level within
  1e-14; `m2l_on` against `select`; a zero budget bit for bit Rotation.
- `fmm-exec/tests/helmholtz_fmm.rs`: `check_dense` generalised to both strategies, on the
  same two scenarios: Classes against the direct sum and against Rotation within
  `DENSE_TOLERANCE`, 2 threads, pair by pair and the one-rank FMM bit for bit, the finest
  level alone. Keep the one-rank debug run's growth small (Classes builds 16 offsets per
  level, not 316) and report it.

## Checks

`/verify`: format; Clippy default, examples and workspace all-targets; `cargo test` and
`--workspace`; `cargo doc`; the MPI list at 2 and 4 ranks (`helmholtz_fmm` changes);
`cargo test -p nd-fmm-tables` and `-p nd-fmm-exec`. No licence check (no dependency change).

## Machine runs

The M6 only (release, BLAS threads 1, one thread, rebuilt right before, every launch behind
a quiet gate, two launches): `helmholtz_m2l` at p = 3–20 with Rotation, Dense and Classes
per pair (Classes over all offsets, over the z-axis offsets and over the others, and the
transforms alone); `helmholtz_accuracy --m2l classes` on CH3.2a's cases (W1 at kD = 30 and
100, the W4 cube at kD = 10, W3 at kD = 30; ε = 10⁻³ and 10⁻⁶) against Rotation, both
orders. They decide nothing for the defaults (Rotation stays the default); locust and
Kathleen are CH3.2c's.

As run (2026-11-02, at b8b8301; the report's §3): both examples rebuilt first, the launches
started after the user confirmed the machine quiet (another job of theirs had run in
parallel), every launch gated (other processes at most 87 % before and 81 % during; none
void). Report `fmm-validate/results/helmholtz-ch3-2b.md`.

## Constraints and risks

- Constraint 1: Laplace untouched (no Laplace file, no driver change; `batched_m2l` stays
  `None`). The per-level branch is in the Helmholtz kernel, above the pair arithmetic.
- Constraint 2: each pair's chain is fixed (transforms and class matrix per element), so
  threads and ranks give the same bits.
- Constraint 5: class matrices built in f64 by Rotation and rounded once; transforms in f64,
  rounded once (exact for the permutations). Not cached (CH2.2).
- Constraints 7, 9: budget per rank, the choice from the degrees and the budget.
- Cost: per pair two transform passes per plane, dense O(p³) for 32 of the 48 elements (by
  offset count: the elements of most of the 316 offsets move the z-axis), on top of Dense's
  2n² reads of a matrix that is now 20× less to hold. At small p the transforms dominate; at
  large p Classes may win where Dense is memory-bound. Measured, not assumed.

## Pending runs

None: the M6 runs are in the report; locust and Kathleen are CH3.2c's.

## Open questions

For sign-off, in the PR:
1. Classes stays opt-in (`HelmholtzM2l::Rotation` the default) with no Auto rule, as Dense:
   end to end 1.23–3.0× Rotation's evaluation and 1.07–1.42× Dense's; per pair faster than
   Rotation only to p = 4. The rule waits for CH3.2c.
2. Classes shares Dense's budget (`dense_budget`, `DEFAULT_DENSE_BUDGET` 1 GiB per rank),
   the transforms counted. Its small tables then cover coarse levels of high p (level 2 at
   p = 30–35 on W1 at kD = 100), where a table form is slowest per pair.
3. The transforms Oⁿ(P) (Oⁿ(P) = (−1)ⁿ Oⁿ(−P) for improper P, T_M = T_L = Oⁿ) follow from
   CONVENTIONS §3.12 and §3.15 as they stand, so no convention changed. Should §3.15 state
   them in a paragraph of its own at its next version bump?
4. CH3.2a's report says Dense per pair is bound by memory; this task's measurement (the
   class tables, in cache, take Dense's time within 5 % from p = 6) shows the arithmetic
   binds. The correction is in this report and the progress log; CH3.2a's report is left
   as it was merged.
