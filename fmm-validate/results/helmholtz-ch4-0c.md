# Helmholtz CH4.0c: the degree rule beyond its measured range

Task: `docs/helmholtz/tasks/CH4.0c.md`. Roadmap: H4, CH4.0c, with the review's amendment
(`docs/reviews/2026-11-12-H6.md`, M6 and M7: the rule's stated range) and CH4.0b's sign-off 2
(the constant density and the plane wave's trace). Proposal §3.5 (the calibrated rule), §9
item 8 (the under-resolved W3), §9.1 (workloads, norms, the box measure), §10 CH4.0c.

## 0. Summary

- **Within ε; the rule stands.** W1 at kD = 1000 (N = 7,957,748, leaves on level 8), where the
  rule's high-frequency branch runs at x = Re(k) a_l = 216.5 on level 2 and 108.3 on level 3,
  beyond the x ≤ 65 it was measured on and the x = 108 it was checked at, meets ε in §9.1's
  relative ℓ2 norm on every case: at 10⁻⁸, 10⁻⁷, 10⁻⁶ and 10⁻³, under Rotation (all-RCR) and
  the library's default, for the random charges, the plane wave's trace and the constant
  density. At 10⁻⁸ under Rotation: 0.20 ε and 0.30 ε (φ, ∇φ) at §9.1's samples, 0.83 ε for
  ∇φ over all 292,810 reference targets (the samples and one per level-8 box), the closest to
  ε of the task. No correction, and no S7 degrees named.
- **The extrapolated levels are near their limit.** With each level raised in turn to the
  rule's degree at 10⁻¹¹ (level 2 from 251 to 258, level 3 from 135 to 142), levels 2, 3 and 4
  carry 0.6 %, 1.0 % and 0.6 % of Σ|e|² (φ; ∇φ 0.6 %, 0.6 %, 0.4 %). The error is level 6's
  (x = 13.5, p = 27, 85 % of φ's and 86 % of ∇φ's), inside the measured range, where the
  high-frequency branch first rises above the plateau (24). The parts add up: the rest is
  0.000.
- **The under-resolved W3** (N = 10⁶): at kD = 200 within ε on every case, the three densities,
  10⁻⁸, 10⁻⁶ and 10⁻³, both M2Ls (at most 0.61 ε, ∇φ of the plane wave's trace at the samples,
  10⁻⁸, Rotation); at kD = 500 too (at most 0.72 ε, the plane wave's trace, the same case),
  after a defect it found. So §9 item 8 holds. Its tree there has 2.06 million leaves for 10⁶
  points, 61 % of them empty (the leaf rule fills level 7, and the octree is complete), and
  level 7's M2L, which runs every V pair of the full level, empty boxes included, takes 86 % of
  the evaluation under Rotation (§3.2), CH4.0d's.
- **A defect in the default M2L, fixed here.** On W3 at kD = 500 the default's build asked
  for 218 GB and aborted (all three densities): `HelmholtzFmmBuilder` sized the exponential
  M2L's waves buffer for the plan's boxes before it set the buffer's bound (CH6.2a's 8 GiB
  per node), so the buffer held every box of the largest level with waves until the bound
  came. A zeroed allocation is mapped lazily, so every earlier case passed unnoticed; here
  the one allocation exceeded the node's memory. The builder now sets the bound first;
  `HelmholtzOperator::plane_wave_buffer_peak_bytes` reports the most the buffer held, and the
  floor check of `tests/helmholtz_fmm.rs` asserts it is the bound's buffer (it failed before
  the fix). The output does not change (the buffer's final size was already the bound's).
- **The rule's range, as documented** (review M6): measured over 10⁻⁸ ≤ ε ≤ 10⁻³ (D1's range;
  the plateau to 10⁻¹⁰) and 0.14 ≤ x ≤ 65; checked in the FMM at x = 108 (CH3.3) and at
  x = 216 and 108 here. CH3.3's 10⁻¹⁸ ≤ ε ≤ 10⁻¹ was a comparison of two formulas (the rule
  at or below S7's closed form), not a range it was measured on; the module docs and
  fmm-exec/CLAUDE.md now say so, and the proposal's §3.5 change is question 1.
- **The box measure** (reported, CH4.4 gates it at 100 ε): on W1 at kD = 1000 and 10⁻⁸ the
  worst level-8 box is 35 ε (∇φ, Rotation; 23 ε under the default) with one target per box;
  on W3 at kD = 200, 20 ε (two targets per level-5 box); on W3 at kD = 500, 21 ε (the
  samples alone).

## 1. Method

- **The tool.** The `helmholtz_boxes` example (one rank, release) of CH4.0b,
  `tools/helmholtz/ch4-0c-runs.sh`: CH3.1c's harness (`helmholtz_accuracy`) on §9.1's
  workloads at D = 1, one complex charge vector, gradients on, the default degree rule
  (`Degrees::Tolerance(ε)`). Since this task it prints each run's degrees, errors and times
  to stderr as the run finishes (the tables print after the last run).
- **The reference.** The direct sum in f64 (`nd_fmm_ref::helmholtz::p2p::direct_sum`) at §9.1's
  1,000 samples, plus for W1's random charges one target drawn in each level-8 box with
  points (291,849 boxes, 292,810 targets with the samples; 4,641 s on the node) and for W3 at kD = 200 two per level-5 box
  (66,471 targets). "All" in the tables is the relative ℓ2 error over every reference
  target; the per-box sample weighs sparse boxes more than the points do (CH4.0b §1).
- **The densities** (CH4.0b §3): §9.1's random complex charges; the plane wave's trace
  e^{ik d·x} (d fixed); the constant density q = 1.
- **The levels** (CH4.0b §1): for each level with M2L, a run with p_l raised to the rule's
  degree at ε/1000 (at least p_l + 1), and one with every level raised; level l's part is the
  first run's error minus the raised run's, its share Σ|part|² over Σ|e|². Named degrees change
  the exponential M2L's precision rule, so this runs on Rotation.
- **x by level** on the unit cube: x = Re(k) √3 r_l, at kD = 1000 216.5, 108.3, 54.1, 27.1,
  13.5, 6.8, 3.4 (levels 2–8); at kD = 500 halved, at kD = 200 43.3, 21.7, 10.8, 5.4
  (levels 2–5).
- **Machines.** Kathleen, one node per job, one rank × 40 threads (Xeon Gold 6248, AVX-512
  P2P), BLAS threads 1: jobs 633320 (`w1-random`, node-c11a-078), 633321 (`w1-levels`,
  node-c11a-048), 633323 (`w1-smooth`, node-c11a-005), 633324 (`w3`, node-c11a-048), built
  from 2eb339e (633320, 633321) and c6b8e75 with this task's example changes (633323, 633324);
  the reruns of W3 at kD = 500, 633429–633431, from c6b8e75 with the fix and the example's
  changes (f6cc2d3's code). Raw outputs in
  `~/projects/fmm-archive/kathleen/logs/ch4-0c/`. Times are reported, never asserted.

## 2. W1 at kD = 1000

### 2.1 The errors

Relative ℓ2 errors over ε (φ, ∇φ), at §9.1's samples and over all the reference's targets;
the worst level-8 box over ε (§9.1's box measure; the random charges at one target per box,
the smooth densities at the samples alone); build and evaluation in seconds (the node, one
rank × 40 threads).

| density | ε | M2L | p by level 2–8 | samples / ε | all / ε | worst box / ε | build, evaluate s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 10⁻⁸ | Rotation | 251, 135, 76, 44, 27, 24, 24 | 0.20, 0.30 | 0.34, **0.83** | 15.1, 34.9 | 63, 518 |
| random | 10⁻⁸ | default | 251, 135, 76, 44, 32, 32, 32 | 0.17, 0.22 | 0.27, 0.59 | 12.0, 23.5 | 93, 300 |
| random | 10⁻⁷ | Rotation | 248, 133, 74, 43, 26, 20, 20 | 0.09, 0.15 | 0.15, 0.41 | 6.0, 14.7 | 62, 412 |
| random | 10⁻⁷ | default | 248, 133, 74, 43, 28, 28, 28 | 0.08, 0.12 | 0.11, 0.30 | 4.2, 13.2 | 90, 274 |
| random | 10⁻⁶ | Rotation | 245, 131, 72, 41, 25, 16, 15 | 0.12, 0.32 | 0.19, 0.68 | 5.2, 14.3 | 60, 285 |
| random | 10⁻⁶ | default | (the same) | 0.12, 0.32 | 0.19, 0.67 | 5.2, 14.3 | 62, 188 |
| random | 10⁻³ | Rotation | 234, 123, 66, 36, 21, 13, 8 | 0.03, 0.10 | 0.05, 0.16 | 0.7, 2.1 | 55, 186 |
| random | 10⁻³ | default | (the same) | 0.03, 0.10 | 0.05, 0.16 | 0.7, 2.1 | 79, 138 |
| plane wave | 10⁻⁸ | Rotation | as random | 0.11, 0.31 | | 2.3, 6.1 | 65, 583 |
| plane wave | 10⁻⁸ | default | as random | 0.09, 0.26 | | 2.1, 5.8 | 99, 297 |
| plane wave | 10⁻⁷ | Rotation | | 0.04, 0.14 | | 1.0, 2.9 | 65, 426 |
| plane wave | 10⁻⁷ | default | | 0.04, 0.12 | | 0.9, 2.7 | 96, 269 |
| plane wave | 10⁻⁶ | both | | 0.05, 0.30 | | 0.8, 7.2 | 63, 312 / 64, 188 |
| plane wave | 10⁻³ | both | | 0.02, 0.10 | | 0.3, 1.5 | 59, 210 / 84, 142 |
| constant | 10⁻⁸ | Rotation | | 0.37, 0.46 | | 3.2, 7.1 | 66, 565 |
| constant | 10⁻⁸ | default | | 0.33, 0.41 | | 3.7, 6.2 | 98, 297 |
| constant | 10⁻⁷ | Rotation | | 0.19, 0.20 | | 2.9, 2.8 | 65, 437 |
| constant | 10⁻⁷ | default | | 0.17, 0.18 | | 2.9, 2.7 | 96, 268 |
| constant | 10⁻⁶ | both | | 0.26, 0.46 | | 5.0, 8.3 | 64, 321 / 64, 189 |
| constant | 10⁻³ | both | | 0.07, 0.17 | | 0.6, 1.3 | 59, 203 / 84, 139 |

At 10⁻⁶ and 10⁻³ the default runs Rotation's degrees, levels 5–8 by another M2L (the same
errors to the printed digits); at 10⁻⁸ and 10⁻⁷ it raises levels 6–8 to 32 and 28, the
exponential M2L's resolving degree (CH6.1g, CH6.1h). The constant density, the closest to ε at
kD = 100 in CH4.0b (0.66 ε), is 0.46 ε here; the plane wave's trace is no harder than the
random charges.

The over-all-targets error of the random charges is 1.4–2.8× the samples' (most for ∇φ at
10⁻⁸ to 10⁻⁶, 0.83 against 0.30 ε at 10⁻⁸): the one-per-box sample weighs the boxes the sphere clips at a corner as
much as full ones, and those hold the worst targets (CH4.0b §2.3). It stays within ε.

### 2.2 The levels

W1 at kD = 1000, 10⁻⁸, random charges, Rotation, at the samples (999 level-8 boxes), each level
raised in turn to the rule's degree at 10⁻¹¹:

| level | x | p | raised to | V pairs | share φ | share ∇φ | samples φ, ∇φ / ε with it raised |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 216.5 | 251 | 258 | 3,096 | 0.006 | 0.006 | 0.20, 0.29 |
| 3 | 108.3 | 135 | 142 | 53,352 | 0.010 | 0.006 | 0.20, 0.29 |
| 4 | 54.1 | 76 | 81 | 563,040 | 0.006 | 0.004 | 0.20, 0.29 |
| 5 | 27.1 | 44 | 48 | 3,286,224 | 0.080 | 0.054 | 0.19, 0.29 |
| 6 | 13.5 | 27 | 42 | 13,585,734 | 0.851 | 0.862 | 0.07, 0.10 |
| 7 | 6.8 | 24 | 42 | 42,928,686 | 0.003 | 0.020 | 0.20, 0.29 |
| 8 | 3.4 | 24 | 42 | 58,010,646 | 0.000 | 0.017 | 0.20, 0.29 |
| rest | | | | | 0.000 | 0.000 | |

Every level raised: 5.2·10⁻¹², 5.9·10⁻¹² (φ, ∇φ). Raising level 2 by 7 degrees moves the error
by under 1 %, so at x = 216 the rule's degree is past the point where the level matters at 10⁻⁸;
S7's closed form (259) would add 8 degrees to level 2 and 7 to level 3 for nothing measurable.
Level 6 sits where the high-frequency branch, ⌈x + 0.80 (ln 1/ε)^{2/3} x^{1/3}⌉ = 27, first
exceeds the plateau (24): the rule's tightest level here, as level 3 (p = 24, the plateau) was
at kD = 100 (CH4.0b §2.1). That is inside the measured range and CH4.3's recalibration's.

### 2.3 Where the time goes

From the per-run lines of the smooth densities (job 633323; the random charges' jobs ran
before the lines existed): under Rotation at 10⁻⁸ the evaluation takes 565–583 s, M2L 539–557
s of it. The top levels at p = 251, 135 and 76 take 26–27, 54–55 and 60 s (3,096, 53,352 and
563,040 V pairs); the leaf levels 7 and 8 at p = 24 take 123–124 and 166–167 s (43 and 58
million pairs). The default takes levels 5–8 off RCR (121–122 s for the four against 399–417 s) and
runs levels 2–4 in Rotation's time. So the levels beyond x = 65 (2 and 3) cost 14 % of the evaluation under Rotation;
the per-run times of 10–11 min matched the plan's estimate.

## 3. The under-resolved W3

### 3.1 kD = 200

W3, N = 10⁶ uniform random points in the cube, kD = 200: leaves on level 5 (32,768 boxes,
about 30 points each), x = 43.3 on level 2. The reference at the samples and two targets per
level-5 box (66,471 targets).

| density | ε | M2L | p by level 2–5 | samples / ε | all / ε | worst box / ε | build, evaluate s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random | 10⁻⁸ | Rotation | 63, 38, 24, 24 | 0.22, 0.49 | 0.22, 0.43 | 9.0, 16.4 | 2.0, 21.0 |
| random | 10⁻⁸ | default | 63, 38, 32, 32 | 0.14, 0.27 | 0.15, 0.25 | 8.2, 11.6 | 3.1, 6.4 |
| random | 10⁻⁶ | Rotation | 60, 35, 22, 15 | 0.07, 0.18 | 0.09, 0.22 | 2.5, 6.4 | 1.8, 7.0 |
| random | 10⁻⁶ | default | (the same) | 0.07, 0.18 | 0.09, 0.22 | 2.4, 6.3 | 2.4, 2.5 |
| random | 10⁻³ | both | 54, 30, 18, 11 | 0.02, 0.04 | 0.02, 0.04 | 0.4, 0.8 | 1.8, 3.7 / 2.9, 2.0 |
| plane wave | 10⁻⁸ | Rotation | | 0.25, **0.61** | 0.12, 0.37 | 9.1, 19.6 | 2.0, 19.6 |
| plane wave | 10⁻⁸ | default | | 0.20, 0.49 | 0.11, 0.31 | 8.3, 17.1 | 3.1, 6.4 |
| plane wave | 10⁻⁶ | both | | 0.05, 0.10 | 0.04, 0.10 | 2.4, 5.1 | 1.9, 7.4 / 2.4, 2.5 |
| plane wave | 10⁻³ | both | | 0.01, 0.04 | 0.01, 0.04 | 0.5, 1.1 | 1.7, 3.7 / 3.0, 1.9 |
| constant | 10⁻⁸ | Rotation | | 0.21, 0.45 | 0.21, 0.43 | 8.6, 20.2 | 2.0, 19.4 |
| constant | 10⁻⁸ | default | | 0.15, 0.30 | 0.14, 0.24 | 4.8, 11.2 | 3.0, 6.4 |
| constant | 10⁻⁶ | both | | 0.07, 0.17 | 0.09, 0.22 | 2.6, 7.4 | 1.9, 7.5 / 2.4, 2.5 |
| constant | 10⁻³ | both | | 0.02, 0.04 | 0.02, 0.04 | 0.4, 0.8 | 1.8, 3.7 / 2.9, 1.9 |

### 3.2 kD = 500

W3, N = 10⁶, kD = 500: leaves on level 7 by the leaf rule (128³ cells, about 0.5 points per
cell), x = 108.3 on level 2. The reference at the samples alone (999–1,000 level-7 boxes).
Jobs 633429 (random, node-c11a-048), 633430 (plane wave, node-c11a-025) and 633431 (constant,
node-c11a-048), with the fix of §0 (the first attempt, job 633324, aborted in the default's
build).

| density | ε | M2L | p by level 2–7 | samples / ε | worst box / ε | build, evaluate s |
| --- | --- | --- | --- | --- | --- | --- |
| random | 10⁻⁸ | Rotation | 135, 76, 44, 27, 24, 24 | 0.29, 0.45 | 4.1, 7.1 | 91, 1,234 |
| random | 10⁻⁸ | default | 135, 76, 44, 32, 32, 32 | 0.27, 0.44 | 3.8, 7.6 | 96, 342 |
| random | 10⁻⁶ | Rotation | 131, 72, 41, 25, 16, 15 | 0.11, 0.27 | 1.7, 2.9 | 94, 326 |
| random | 10⁻⁶ | default | (the same) | 0.11, 0.27 | 1.7, 2.9 | 92, 112 |
| random | 10⁻³ | both | 123, 66, 36, 21, 13, 8 | 0.03, 0.10 | 0.3, 1.0 | 94, 102 / 94, 45 |
| plane wave | 10⁻⁸ | Rotation | | 0.40, **0.72** | 11.8, 20.5 | 90, 1,083 |
| plane wave | 10⁻⁸ | default | | 0.36, 0.64 | 10.9, 18.4 | 95, 338 |
| plane wave | 10⁻⁶ | both | | 0.11, 0.33 | 2.5, 7.3 | 94, 322 / 92, 109 |
| plane wave | 10⁻³ | both | | 0.02, 0.10 | 0.4, 1.2 | 94, 97 / 94, 44 |
| constant | 10⁻⁸ | Rotation | | 0.32, 0.51 | 5.7, 12.0 | 91, 1,232 |
| constant | 10⁻⁸ | default | | 0.28, 0.43 | 4.5, 9.3 | 95, 340 |
| constant | 10⁻⁶ | both | | 0.11, 0.27 | 0.8, 3.5 | 91, 330 / 91, 112 |
| constant | 10⁻³ | both | | 0.03, 0.10 | 0.3, 0.9 | 93, 98 / 94, 44 |

Within ε on every case; the closest, 0.72 ε, is ∇φ of the plane wave's trace at 10⁻⁸ under
Rotation. So §9 item 8 holds: the under-resolved inputs of W3 meet the tolerance at kD = 200
and 500.

**The cost of the under-resolved tree** (reported, not this task's to change). The leaf rule
puts every leaf that holds a point on level 7 (λ = 2π/500 ≈ 0.0126 D, and level 6's boxes are
0.0156 D wide), and nd-octree's tree is complete: a refined box has all eight children, and an
empty child is a leaf. With 10⁶ uniform points over 128³ = 2,097,152 cells (0.477 per cell)
nearly every level-6 box holds a point, so level 7 is nearly full. Poisson counts give 62 % of
the cells empty, and 262,144 · e^(−8 · 0.477) ≈ 5,780 level-6 boxes whose eight children are
all empty, which stay leaves: 2,097,152 − 7 · 5,780 ≈ 2,056,700 leaves, against 2,056,727
measured. About 1.26 million leaves (61 %) are empty, 0.62 million hold one point and
0.17 million more.

The plan treats an empty leaf as any other box. Its lists come from the keys alone
(`nd_fmm_plan::interaction_manager`), so it is a target and a source of V pairs; its P2M and
L2P are no-ops on an empty chunk; and the drivers skip a box with no V row, never one without
points. A full level 7 has 383,233,032 V pairs, level 6 46,298,376 (8.3×). At p = 24 on both
levels, level 7's M2L took 9.1× level 6's (1,056 against 116 s; 7.9× for the plane wave's
trace), so it ran every pair. About 38 % of those pairs have a source that holds a point and
14 % a source and a target that do; the rest translate a zero multipole or give a local that no
point reads. Under Rotation at 10⁻⁸ level 7's M2L takes 1,056 s of the 1,232 s evaluation
(level 6 116 s, levels 2–5 37 s); the default's waves bring it to 247 s of 340 s. W1 at
kD = 1000, with 8 million points, evaluates in 518 s.

Two parts are CH4.0d's (a sentence there): the leaves of about one point that the rule makes
here (even without the empty boxes, about 795 thousand boxes of 1.3 points each, each with an
M2L at p = 24), and the work on empty boxes.

## 4. Questions for sign-off

1. **Proposal §3.5, the check at x = 216.** §3.5 already states the rule as measured over
   0.14 ≤ x ≤ 65 and 10⁻⁸ ≤ ε ≤ 10⁻³ and checked at x = 108; CH3.3's 10⁻¹⁸ to 10⁻¹ appears only in
   CH3.3's report and the code's documentation, now corrected there (review M6). Recommended:
   add "and at x = 216 and 108 (W1 at kD = 1000, CH4.0c: within ε, at most 0.83 ε; raising
   either level moves the error by under 1 %)" to its sentence, "(amended at CH4.0c)" and an
   Appendix A line. Alternative: leave §3.5 as is; the check is recorded in the code's
   documentation and this report.

## 5. Reproduce

```sh
tools/kathleen/sync.sh
for stage in w1-random w1-levels w1-smooth w3; do
    tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch \
        'cargo build --release -p nd-fmm-validate --example helmholtz_boxes' \
        "tools/helmholtz/ch4-0c-runs.sh $stage <out>/$stage"
done
```

W3 at kD = 500 alone, one job per density: `ONLY=w3-500-<density>
tools/helmholtz/ch4-0c-runs.sh w3 <out>/w3-500`.
