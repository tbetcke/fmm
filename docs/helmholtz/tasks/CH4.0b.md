# CH4.0b The worst-box concentration

Started 2026-11-12 on `tbetcke/h-ch4-0b`. Roadmap: "CH4.0b The worst-box concentration" (H4;
priority list item 2), with its amendment from the review of 2026-11-12 (M5, A3: the smooth
density). Proposal: §10 CH4.0b; §9.1 (the box measure); §2 (constraints 1, 2, 6, 9); §3.5
(the degree rule). Sources: CH3.6's report §0 and §2 (`fmm-validate/results/helmholtz-ch3-6.md`),
its plan's results (`docs/helmholtz/tasks/CH3.6.md`, the margin experiment); the review
`docs/reviews/2026-11-12-H6.md` M5, A3 and B3.

## Goal

Find where CH3.6's worst boxes come from, propose whether a pointwise criterion joins §9.1's
ℓ2 norm, and measure the FMM on a smooth density (the review's M5 and A3) before CH4.3
calibrates on the degree rule.

## Acceptance (proposal §10, verbatim)

> the cause stated; whether a pointwise criterion joins Section 9.1's ℓ2 proposed

and from the roadmap's *Done when*: "the cause located (level, list, operator or placement)
with the box measure of proposal §9.1; whether a pointwise criterion joins §9.1's ℓ2
proposed. Also (review 2026-11-12, M5 and A3): … A smooth density on W1 (q = e^{ik d·x}, the
trace of a plane wave; a constant density at low kD), at 10⁻³, 10⁻⁶ and 10⁻⁸ under the
default and on all-RCR, in ℓ2 and per box against the sampled direct sum: within ε, or the
degrees (the plateau, p_w at 10⁻⁶ on the exponential M2L) that such a density needs, before
CH4.3 calibrates on this rule." Sign-off: the criterion; the charge model of the gates (a
check of this task only; a standing workload is decided on its result).

## What the record shows

- CH3.6 (Kathleen, 4 × 10, Rotation everywhere, before D8): the worst box of the deepest
  level is 8–25× the all-target ℓ2 error at 10⁻³, 15–51× at 10⁻⁶, 13–146× at 10⁻⁸ (FMM3D
  5–14×, 7–26×, 6–37×); pointwise up to 57 ε (∇φ, W1 at kD = 500, 10⁻⁸). The box measure is
  the largest error at one of a box's targets over the reference's RMS over all targets.
- On W1 at 10⁻⁶ and 10⁻⁸ most worst boxes are the six where the sphere touches its bounding
  cube (kD = 30, 100, 500). With a margin of 0.03 or 0.1 (throwaway, the M6, kD = 100) they
  moved elsewhere and the ratio stayed 20–70×. Where they moved was not recorded.
- The small cases show it: W1 at kD = 30 (N = 7,163) 4.0 ε and 16.7 ε at 10⁻⁶; W1 at kD = 100
  (N = 79,578) 9.6 ε and 21.9 ε at 10⁻⁸. A direct sum at every target fits on the M6 there.
- At the tangent points the domain's faces are box faces at every level: the points of those
  boxes lie on one face of their box at every level at once. Hypothesis, not tested.
- Every calibration and gate so far used random charges in [−1, 1]; the near field, computed
  exactly, dominates ‖φ‖ and ‖∇φ‖ (review M5).

## Read first

- `fmm-validate/src/helmholtz_accuracy.rs` (`Case`, `Problem`, `Oracle`, `run_output`) and
  `src/helmholtz_fmm3d.rs` (`Grid`, `Crosscheck`, `crosscheck`).
- `fmm-exec/src/helmholtz/fmm.rs` (`HelmholtzFmmBuilder::{domain, max_points_per_leaf}`),
  `src/helmholtz/degree.rs` (`Degrees::PerLevel`, `DegreeRule`).
- `fmm-ref/src/helmholtz/{leaf, rcr, p2p}.rs` (the single-level chain, if stage B runs).

## Changes

nd-fmm-validate only (tooling; no library change unless the cause is a defect, which would
then be asked first):

1. **The density** (`helmholtz_accuracy::Density`, `Case::density`): `Random` (§9.1's, the
   default, every earlier bit unchanged), `PlaneWave { direction }` (q_j = e^{ik d·x_j}, d a
   fixed unit vector off the axes, complex charges), `Constant` (q_j = 1). `Problem::new` draws
   by it; the oracle, the checksums and FMM3D's problem file follow.
2. **The box study** (`helmholtz_boxes` module, MPI-free parts tested; `helmholtz_boxes`
   example, not registered):
   - the reference at every target (the direct sum, threaded) or, above a size, at the
     §9.1 samples plus s targets per box of the deepest level (the "sampled direct sum");
   - per run: §9.1's ℓ2 at the samples, ℓ2 over the reference's targets, the box measure
     (`Crosscheck`), the worst boxes with their place (the box's centre, whether a face
     lies on the domain's face, each worst target's offset from its box centre at every
     level in units of r_l), and the leaf levels and points of those boxes;
   - variants of one case: the degree rule (`Degrees::Tolerance`), one level's degree raised
     by d (`Degrees::PerLevel`) for each level with M2L, all levels raised, a uniform tree
     (`max_points_per_leaf` above N: no W or X list), the M2L (Rotation, the default), the
     domain with a margin or an offset;
   - the decomposition by level: per target the error e(t) of the rule's run minus that of
     the run with level l raised, so e(t) ≈ Σ_l e_l(t) + rest; reported over all targets
     (each level's share of Σ|e|²) and at the worst boxes' worst targets.
3. **Stage B, only if stage A leaves the mechanism open** (decided in the plan's results):
   the single-level chain per V pair at a target (P2M of the source box's points at level l,
   `rcr::m2l`, L2P at the target, against the direct sum of that box), grouped by offset
   class, on a uniform tree, for the worst targets and random ones.
4. **Scripts**: `tools/helmholtz/ch4-0b-runs.sh` (the cause runs on the M6, the density runs on
   Kathleen), `ch4-0b-tables.py` if the tables need it.
5. **The report** `fmm-validate/results/helmholtz-ch4-0b.md`.

Public API: new items in nd-fmm-validate only. Conventions: none. Dependencies: none.

## Tests that prove it

- `tests/helmholtz_accuracy.rs` (the one MPI test): `Density::Random` keeps the default
  case's checksum (bit for bit the run before the field existed); the plane-wave and constant
  densities against the direct sum within the FMM's tolerance on W1 at kD = 10, 2 threads
  bit for bit 1.
- `helmholtz_boxes` unit tests (no MPI): the reference at every target equals
  `Oracle::of_samples` at the same targets bit for bit; the per-box sample takes s distinct
  targets in every box that holds s, all in a box with fewer, deterministic by seed; the
  level offsets of a hand-made point (on a face at every level, at a centre); the
  decomposition sums on hand-made vectors.
- A smoke run of the example's core on W1 at kD = 10 in the MPI test: the box study's
  worst box equal to `crosscheck`'s against the same reference.

## Checks

`/verify` (nd-fmm-validate: fmt, Clippy default, examples and workspace, its tests, the
workspace tests at opt-level 1, docs; no Laplace path, no MPI test list change).

## Machine runs

- **The M6** (accuracy only, no timing: no quiet gate needed; one rank, 6 threads): the
  cause study on W1 at kD = 30 and 100 at 10⁻⁶ and 10⁻⁸, on all-RCR (Rotation, as CH3.6)
  and under the default: the variants above, the reference at every target. Also W3 at
  kD = 30 and the W4 Plummer sphere (N = 10⁵) at 10⁻⁸, to see whether the cause is W1's.
- **Kathleen** (1 node, 1 × 40; `singlenode`): the density matrix. W1 at kD = 10, 30, 100
  and 500; densities random, plane wave and (at kD ≤ 30) constant; ε = 10⁻³, 10⁻⁶, 10⁻⁸;
  the default and Rotation. ℓ2 at the samples and the box measure: at every target to
  kD = 100, at the per-box sample at kD = 500. Where the smooth density misses ε, the level
  scan (stage A's raise per level) on that case, for the degrees it needs.

## Constraints and risks

- Constraint 1: Laplace untouched (nd-fmm-validate's Helmholtz modules and tools only).
- Constraint 9 bounds any remedy: a pointwise criterion or a degree change may read k, the
  tree's geometry and ε, never the charges. A density that needs higher degrees changes the
  rule for all densities.
- Pair and FMM evidence (H6 gate §7): if the task proposes a degree change, it needs both
  the per-level scan (FMM) and the per-pair chain (stage B) before it asks to adopt it; here
  it only measures and proposes.
- A wider scope widens the verification set: the task changes no bound or default, so no
  earlier exclusion returns; noted for any remedy that follows.
- Risk: the decomposition by level assumes the levels' errors add; the rest term is reported.
- Risk: the per-box sample at kD = 500 under-reads the box maximum; the full reference at
  kD ≤ 100 is the measure of record, kD = 500 a check.

## Results

The report is `fmm-validate/results/helmholtz-ch4-0b.md`; raw outputs in
`~/projects/fmm-archive/m6/logs/ch4-0b/` (cause, densities to kD = 100) and
`~/projects/fmm-archive/kathleen/logs/ch4-0b/` (job 633154, node-c11b-041, 1 × 40: W1 at
kD = 500, the pointwise runs).

- Stage A located the cause; stage B (the per-pair chain) was not needed and was not built.
  The levels' parts add up to the whole error (rest ≤ 0.002), and binning them by the target's
  distance from its box centre shows the corner effect on every level and workload.
- The cause: placement near a box corner, mostly on the coarsest plateau level of the degree
  rule; W1's tangent points are box vertices on every level; clipped boxes elsewhere. Not the
  lists (a tree of one leaf level), not the boundary (margins), not a defect (the levels'
  truncation accounts for all of it); which operator of a level was not separated.
- The smooth densities meet ε in ℓ2 at every point of the matrix; the constant density is
  the hardest, 0.66 ε. Below D1's range (10⁻⁹, 10⁻¹⁰) the constant density misses ε in ℓ2
  (1.5–3.4×), all from level 2's high-frequency degree: a sentence in CH4.0c's roadmap line.
- The self-review (a subagent on the plan, §10 and §2) found five numbers that did not match
  the outputs and claims stated beyond the data (the operator, the sources, the tangent
  points' sum, the smooth densities' worst boxes); fixed, with the extra runs in
  `~/projects/fmm-archive/m6/logs/ch4-0b/extra/` (the default's and the constant density's
  worst boxes placed, the constant density's pointwise scan and its levels at 10⁻¹⁰).
- A pointwise criterion: ε/10 at 10⁻³ and 10⁻⁶, ε/30–ε/100 at 10⁻⁸ (§4 of the report).
- The CH3.6 values re-run here agree to every printed digit with CH3.6's (FMM3D at 10⁻¹² as
  reference there, the direct sum here).
- Test cost: the one-rank debug run of `tests/helmholtz_accuracy.rs` 95–99 s against `main`'s
  83–87 s, back to back on a loaded M6 (load 19–21, another project's tests); a first version
  of the checks took 161 s against 76 s and was cut (the named domain at kD = 20, the constant
  density in one run).

## Pending runs

None.

## Open questions

For sign-off (the report's §5): 1. the criterion (recommended: ℓ2 stays, the box measure a
gate bound of 100 ε); 2. the charge model (recommended: §9.1's random charges stay; CH4.3 and
CH4.4 verify on the constant density and the plane wave's trace too).
