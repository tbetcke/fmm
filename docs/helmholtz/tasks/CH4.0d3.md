# CH4.0d3 The leaf size and c_leaf

Started 2026-11-13 on `tbetcke/h-ch4-0d3`. Roadmap: "CH4.0d3 The leaf size and c_leaf" (H4;
priority list item 4, after CH4.0d2). Proposal: no §10 row (CH4.0d was added by the roadmap
after the H6 gate); §2 (constraints 1, 2, 9); §3.7 (the leaf rule, c_leaf, D11); §5.7 (the P2P
and its phase range); §9.1 (the workloads, the tolerance). Sources: CH4.0c's report §3.2, the
reports and records of CH4.0d1, CH4.0d1b and CH4.0d2, Phase 6 T11's leaf rule for Laplace
(`fmm::default_max_points_per_leaf`, fmm-validate/results/phase6.md), CH3.6's matrix
(fmm-validate/results/helmholtz-ch3-6.md), review 2026-11-12 A2 (`docs/reviews/2026-11-12-H6.md`).

## Goal

A default points-per-leaf rule for `HelmholtzFmmBuilder` (by ε or p, the machine class and the
multipole planes), with the V rule's default (`near_v`) and c_leaf decided beside it, from the
FMM's build plus evaluation measured over the leaf size on Kathleen and locust; or the report
that 64 stays best, and why.

## Acceptance (the roadmap's *Done when*; no proposal §10 row)

"A default leaf size by ε (or p), the machine class and the multipole planes, proposed for
sign-off with its gain per case, as T11 did for Laplace. Inputs that do not depend on the data
(constraint 9). The errors within ε, at most CH3.4b's contract range (Re(k) r ≤ 22 c_leaf,
unchanged). Deterministic and rank-invariant at each leaf size. Laplace's defaults unchanged
(the Helmholtz builder only). Or the report shows that 64 stays best, and why `fastker`'s
leaves do not carry over." Also (CH4.0d2's question 1, signed off 2026-11-13: "I agree with
your recommendation"): the sweep with `near_v` off and with `VNear::CostModel`, on Kathleen and
locust, and the V rule's default proposed with the leaf size. And (review A2; #209's question
2, "I agree"): c_leaf swept from 1 to 2 on the cases where it binds, the errors within ε and
the contract range checked at each c_leaf; D11's c_leaf = 1 stays the default until this
task's sign-off says otherwise.

## What the record shows

- `HelmholtzFmmBuilder` refines to `DEFAULT_MAX_POINTS_PER_LEAF` = 64 (Laplace's base value,
  without T11's per-p rule: 128 from p = 12 on the host); the leaf rule (`min_leaf_level`,
  c_leaf = 1 wavelength) only refines further. FMM3D's `hfmm3d` allows 40–700 points per leaf
  (default) and 300–1000 (`fastker`).
- CH4.0d2: under Rotation nearly every V pair between leaves of 64 points paid as a near pair
  from p ≈ 11, a sign that the leaves are small for those degrees; under the default (Batched
  to p = 11, the waves above) the rule paid only at 10⁻⁸ on Kathleen.
- Where the leaf rule binds, the points alone cannot change the tree: W1 at kD = 500 has every
  leaf at level 7 with about 27 points; W4 at N = 10⁵ and kD = 100 has about 25 points per
  λ³. W3 at ten points per wavelength (1000 per λ³) never binds below 1000 points per leaf.
  c_leaf = 2 lets those leaves be one level coarser (8× the points).
- The P2P's phase: a U pair spans at most 2√3 leaf widths, so Re(k) r ≤ 2π · 2√3 c_leaf ≈ 22
  c_leaf (proposal §5.7); a converted V pair at most 4√3 widths, ≈ 44 c_leaf. Both far below
  the kernel's contract domain (Re(k) r ≤ 2¹⁰, `P2pKernel::evaluate_helmholtz`), whose
  per-term bound (32 + 12 |k| r) u grows with the phase.
- A shallower tree raises the finest levels' degrees (x = Re(k) a_l grows with the box), and
  the empty-box rule (CH4.0d1, on by default) already removes the empty leaves' work.

## Changes

### `nd-fmm-validate` (the harness)

- `helmholtz_accuracy`: `--leaf-size n,…` (`Case::max_points_per_leaf`, the library's when
  absent) and `--c-leaf c,…` (`Case::c_leaf`, new), each case run at each in turn (outermost
  loops: each changes the tree); the row's tree column names both. `Row` gains the leaf size
  and c_leaf as built, the leaves per level and the occupied leaves (summed over the ranks),
  so the report names where c_leaf binds (leaves on the leaf rule's level that the points alone
  would not have split), and the largest phase bound Re(k) × the farthest near pair's point
  distance (U and converted V, from the leaf widths), for the contract range.
- `tools/helmholtz/ch4-0d3-runs.sh` and `ch4-0d3-tables.py`.

### `nd-fmm-exec` (the Helmholtz builder only; after the measurements)

- `helmholtz::default_max_points_per_leaf(…)`: the measured rule, a function of the
  tolerance (or the degrees when named), `Isa` (the P2P kernel's, the machine class as Auto's
  constants) and the multipole planes; `HelmholtzFmmBuilder::max_points_per_leaf` stays,
  stored as an `Option` (named or the rule), `HelmholtzFmm::max_points_per_leaf` reports the
  target used. If the measurements say 64 stays best, no code change here.
- `near_v`'s default and `C_LEAF` changed only as the measurements propose, each a sign-off
  question; the PR builds the proposed defaults (as CH4.0d1 did), reverted if not signed off.
- (Open question 1.) `HelmholtzFmmBuilder::near_wx(WxNear)`: `Off`, `Crossovers { w, x }`
  (named, for the measurements) or `CostModel` (per level, N_W = α_M2P(p_s)/α_P2P in source
  points of the W leaf, N_X = α_P2L(p_t)/α_P2P in target points of the X target, `WxCosts` per
  machine class, agreed over the ranks), through `NearRule::new`'s `LeafPair` predicate in the
  one plan build, with the empty-box rule and the V rule. `HelmholtzOperator::kind_times`
  (P2M, M2M, L2L, L2P, P2L, M2P per level) and `HelmholtzFmm::far_interactions` (per level
  Σ_X n_s and Σ_W n_t) for the constants.
- No change to `FmmBuilder`, `DEFAULT_MAX_POINTS_PER_LEAF`, `default_max_points_per_leaf`
  (Laplace's), `nd-fmm-plan`, `nd-octree` or `nd-fmm-simd`.

Conventions: none. Dependencies: none. `nd-fmm-plan`'s interface: unchanged.

## Tests that prove it

- `fmm-exec/tests/helmholtz/` (no MPI): the rule's table, its inputs only (a function of ε or
  the degrees, the ISA and the planes; monotone where the measurements say so), the named
  size kept, zero refused as before.
- `fmm-exec/tests/helmholtz_fmm.rs` (1, 2, 4 ranks): at small and large leaf sizes (8 and
  512 on the sphere, as built; 16 planned), with c_leaf 1 and 2 where the rule binds), the FMM against
  the direct sum within the scenario's bound, 2 threads and the one-rank FMM bit for bit
  (rank invariance and determinism at each leaf size); the default builder reports the rule's
  size; with the V rule's proposed default the existing `check_near_v` bounds.
- `fmm-validate/tests/helmholtz_accuracy.rs`: the new options and `Row` fields (leaf size and
  c_leaf as named, leaves per level summing to the leaves).
- Laplace: every existing test unchanged (no Laplace code touched).

## Checks

`/verify` (fmt, Clippy default, examples, workspace; docs; the default and workspace tests at
opt-level 1; the MPI list at 2 and 4 ranks; `helmholtz_fmm` at 8 by hand). No manifest change.

## Machine runs

All release, BLAS threads 1, one rank × all cores unless named, `--m2l auto-pw` (the
default), `--near-v off,cost`, `--levels`, two evaluations, ABBA order of the leaf sizes:

- Kathleen (`singlenode`, 1 × 40), round 1 (the sweep):
  - `matrix`: CH3.6's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4
    distributions at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶, 10⁻⁸, leaf sizes 32, 64,
    128, 256, 512, 1024;
  - `large`: W1 at kD = 500 and W3 at kD = 100 (N = 4·10⁶) at 10⁻³, 10⁻⁶, 10⁻⁸, the same
    sizes (W3 at kD = 100 to 512);
  - `cleaf`: c_leaf 1, 1.5, 2 × leaf sizes 64, 128, 256, 512 on W1 at kD = 100 and 500 and
    W4's cube and Plummer sphere at kD = 100, 10⁻³, 10⁻⁶, 10⁻⁸ (the errors against ε, the
    phase bound);
  - `planes`: real charges (one multipole plane) on W1 at kD = 30, 100 and W4's cube at
    kD = 10, 100, the sizes of `matrix`.
- locust (1 × 72; NEON, the other machine class): `matrix`, `large` (W1 at kD = 500 only) and
  `planes`, round 1's sizes.
- Round 2 (both machines): the proposed rule against 64 (rule off, rule on; ABBA) on the
  matrix and the large cases, and `traffic`: 4 × 10 against 1 × 40 on Kathleen at the
  proposed default (checksums, rank-count invariance), W1 at kD = 500 and W4's Plummer sphere.
- The M6: none unless the user gives the go-ahead for a timing run.
- What it decides: the default leaf size by (ε, machine class, planes), the V rule's default
  and c_leaf; each proposed with its gain per case (build plus one evaluation, and the
  evaluation alone), as T11's rule: a size other than 64 only where the geometric mean over a
  class's cases is at least 5 % lower, at errors within ε.

## Constraints and risks

- Constraint 1: no Laplace code changes (the Helmholtz builder, the harness, docs). Laplace's
  `DEFAULT_MAX_POINTS_PER_LEAF` and `default_max_points_per_leaf` stay as they are.
- Constraint 2: the leaf size changes the tree, not the guarantees: the one-rank FMM bit for
  bit at each size (tests), 4 × 10 = 1 × 40 (round 2).
- Constraint 9: the rule reads ε or the degrees, the ISA and the planes, never the points.
- Risk: the gain per size may depend on the case more than on (ε, class); then the rule is a
  compromise (the geometric mean) and the report lists the losers, as T11's did.
- Risk: the evaluation's errors may move with the tree (the worst box, CH4.0b); the check is
  §9.1's ℓ2 tolerance on every run.
- Risk: scope. Three parts (harness, runs, rule). If round 2 needs another calibration, the
  rule's adoption is split off as CH4.0d3b, this PR the sweep and the proposal.

## Pending runs

None. Every round ran (report §1, §5): calibration Kathleen 634049 and locust; round 1 Kathleen
633910, 633911 and locust; round 2 Kathleen 634080, 634081 and locust; round 3 Kathleen 634114
(with `traffic`), 634115 and locust.

## Open questions

1. Round 1 (Kathleen 633910, 633911; locust, both at 23e993c) and the diagnostic runs
   (locust, ae7d541) found P2L and M2P, the W and X pairs, 40–67 % of the evaluation on
   adaptive trees (W4's Plummer sphere at kD = 10, 10⁻⁶: M2P 0.10–0.29 s of 0.20–0.74 s), and
   trees with leaves on two levels slow (W3 at kD = 100 with 128 points: the evaluation 3× 64's;
   W1 at kD = 500 with 128 points at c_leaf 1.5 and 2: 1.14–1.37×): the Helmholtz builder
   converts no W or X pair (Laplace's near rule of Phase 6 T12 does). Decided by the user on
   2026-11-14: "Add W/X rule here (Recommended)": `HelmholtzFmmBuilder::near_wx` with
   `nd_fmm_plan`'s existing `NearRule` (no plan change) and a cost model per machine class
   from the P2L and M2P timings, then round 2 sweeps the leaf size with it and proposes the
   leaf size, `near_v`, the W/X rule and c_leaf together.
2. c_leaf: round 1's c_leaf 2 with 256–512 points cut build plus evaluation to 0.35–0.67
   where the leaf rule binds, errors within ε, the phase within 22 c_leaf. Decided by the user
   on 2026-11-14: "Yes, sweep 1 and 2 (Recommended)": round 2 runs c_leaf 1 and 2 and the PR
   proposes c_leaf for sign-off (D11 amended only on that sign-off).
