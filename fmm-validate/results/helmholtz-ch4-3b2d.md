# Helmholtz CH4.3b2d: the waves' V crossover on raised levels, in one plan

Task: docs/helmholtz/tasks/CH4.3b2d.md (roadmap CH4.3b2d, added at CH4.3b2b, its question 3, signed
off 2026-11-21 on #245; proposal §10 CH4.3; §3.6, §3.7, §5.3). CH4.3b2b's report §5–§6 found that
on a level whose waves are raised (p_w above the degree given, ε < 10⁻⁶) the V near rule's
per-pair constant (`VCosts::plane_wave` b (p_w + 1)²) under-prices the waves, which cost per box,
and that a second plan priced by the waves' cost per box over the level's pairs converted 10–100 %
more of the level's leaf pairs at 0.09–0.75 s per case. This task prices those levels per box
before the plan, in one plan, against that second plan as the trusted measure.

## 0. Summary

1. **The change** (§2): `VRaisedPrice::PerBox` prices a V pair on a raised level at the higher of
   the per-pair constant and the waves' cost per box with a V row at p_w over the V entries per
   leaf that the per-pair price leaves, the leaves and entries counted from the octree's keys and
   the point counts before the plan (summed over the ranks), one plan. Where a raised level's
   boxes with a V row are leaves, this is exactly the crossover the trusted measure
   (`VAfterPlan::RaisedLevels`, the second plan on the raised levels alone) finds after a first
   plan.
2. **Pair evidence** (§4): on every raised level of W1 at kD = 100–500 and W3 at N = 10⁶, kD = 200
   and 500, at 10⁻⁷ and 10⁻⁸, the per-box crossover is the trusted measure's (W1: 20,529–83,064
   per level against the per-pair 3,882 and 5,026), and the per-box plan leaves the trusted plan's V
   pairs on every level the trusted measure raises. On W3's dense raised level 4 (256 points per
   leaf) neither converts a pair (crossovers 6,743–9,348 against n_s n_t up to 256²).
3. **FMM evidence** (§5), both run orders, under the defaults and under the time rule: per-box is
   the trusted measure's output bit for bit on 36 of 40 runs (the other 4: the time rule decides on
   the one plan, §6); the raised levels' M2L and P2P −0.12 to −0.98 s per case against the per-pair
   price on W1 (within −0.15 to +0.07 s of the trusted measure), the evaluation 0.53–0.98× the
   per-pair price's; the errors the trusted measure's to every digit, 0.58–1.00× the per-pair
   price's. On W3 the three plans are the same; its evaluation's changes (0.92–1.17×) are the
   spread of one plan.
4. **One plan**: the second plan costs 0.03–1.09 s of build on W1 (0.21 s on W3 at kD = 200); the
   count, after it ran on the pool, 0.003–0.025 s on W1 and 0.35–0.37 s on W3 at kD = 500 (5.3 s
   serially, job 653782), within the builds' spread there (16.9–17.0 s per box, 17.4 s per pair).
5. **The first count** (§3, job 653776, kept for the record): counting every V entry of a leaf,
   not those the per-pair price leaves, put the crossover at 0.64–0.68× the trusted measure's on W1
   at kD = 500's level 6 and left 11,160–39,839 pairs that the second plan converts (+0.25 to
   +0.42 s against it under the defaults). The count was changed to the trusted measure's shape.
6. **Recommended** (§7): the acceptance as read; `PerBox` the default (made so in this PR, for
   sign-off); `VAfterPlan` stays opt-in.

## 1. Setup

- **Runs.** `helmholtz_accuracy` (release, BLAS threads 1), one rank with 40 threads bound to the
  node's cores, launched by `tools/helmholtz/ch4-3b2d-runs.sh`; §9.1's complex charges and 1,000
  samples, gradients, the W and X rules of the defaults, the interpolated build, the defaults' M2L
  (`--m2l auto-pw`) with the cost model's V rule, two evaluations (the rows the second's). Each
  launch: `--diagonal off,auto` (the defaults; the time rule, CH4.3b2c's default next) ×
  `--v-raised pair,box` × `--v-after-plan off,raised`, so four runs per setting: `pair` (the
  per-pair price), `pair+raised` (the trusted measure), `box` (one plan), `box+raised` (what a
  second plan would still add).
  - `w1`: W1 at kD = 100, 200, 500 at ε 10⁻⁷ and 10⁻⁸; `w3`: W3 at N = 10⁶, kD = 200 and 500, the
    same ε; `reverse`: the same launches with every list reversed.
  - `counts`: the same cases under the defaults with `--v-raised pair,box` only (the count's build
    time after it ran on the pool).
- **Machine.** Kathleen, `singlenode`, 1 × 40 (Xeon Gold 6248):
  - job 653776 (node-c11b-025, 2026-11-21 09:50–10:15, at cdaf285, the first count): `smoke`,
    `w1`, `w3`, every launch exit 0; cancelled before `reverse` when §3's defect was seen;
  - job 653782 (node-c11b-008, 10:17–11:08, at 43f29ad): `smoke`, `w1`, `w3`, `reverse`, 21
    launches, every one exit 0;
  - job 653786 (node-c11b-032, 11:14–11:22, at 8c51ae0, the count on the pool): `counts`, 10
    launches, every one exit 0.
  653782's runs counted serially; 8c51ae0 runs the count in chunks of leaves on the pool (integer
  sums, the same counts and plans; `check_v_raised` holds two threads to one). Every plan and output
  of §4–§6 is 653782's; the count's time is 653786's.
- **Raw outputs**: `~/projects/fmm-archive/kathleen/runs/ch4-3b2d{,-reverse,-counts,-first}/`.
- **Tables**: `tools/helmholtz/ch4-3b2d-tables.py runs|levels|summary <dir>`.
- **Spread.** As CH4.3b2b's §1: runs of one plan in one launch differ by up to 30 % in the
  evaluation; §5 reads the raised levels' own seconds in both orders, and the plans (V pairs, near
  interactions, checksums) exactly.

## 2. The change

`HelmholtzFmmBuilder::v_raised_price(VRaisedPrice)` (`PerPair`, CH4.0d2's constant; `PerBox`, the
default since this PR), under `VNear::CostModel` only. Before the plan, on each level whose
predicted waves are raised (`wave_levels`, p_w = `plane_wave_degree(p_l, ε)` > p_l):

- `leaf_v_shapes` counts, per local leaf t, its V entries by the per-key rule of
  `nd_fmm_plan::interaction_manager` (the children of the interior same-level neighbours of t's
  parent not adjacent to t), less those the empty-box rule leaves out and those between two leaves
  that the V rule converts at the agreed per-pair crossover (`v_group`, per (parent, source), the
  plan's own decision): the entries the per-pair plan leaves in t's row; t counted only with one
  left. Every local leaf is held by one rank, so the sums over the ranks (one all-reduce) are the
  one-rank tree's (constraints 2 and 9). The leaves in chunks on the FMM's pool.
- `raised_crossover` = c_w(p_w) L / P over the P2P's per interaction: c_w the waves' cost per box
  with a V row (`DiagonalCosts::plane_wave`, a (p + 1)² + b (p + 1)³), L the leaves and P their
  entries; the level's crossover the higher of it and the per-pair one, agreed (all-reduce min).
- One plan, with the crossovers as before (`v_group` on levels with waves).

`VAfterPlan::RaisedLevels` (opt-in) is CH4.3b2b's `AllLevels` restricted to the levels that keep
their waves at a raised degree: the trusted measure here, with no unraised level changed (CH4.3b2b's
question 4) and no level the time rule took. `HelmholtzFmm::{v_raised_price, raised_v_shapes}`
report the setting and the counts; the harness's last column of "Near pairs" prints them.

Tests: `tests/helmholtz/v_crossover.rs` (the closed form, equal to the after-plan crossover of the
same boxes and pairs); `check_v_raised` in `tests/helmholtz_fmm.rs` (3,000 points on the sphere at
kD = 30, 10⁻⁷, NEON's costs named): the counts equal the per-pair plan's computed leaf rows, the
crossovers by `raised_crossover` on the raised levels only, more conversions, one plan, the errors
within 1.1×, 2 threads and on P ranks the one-rank FMM bit for bit, Rotation counting nothing, and
the trusted measure's crossover after the plan equal to the per-box one (10,890 both);
`check_v_after_plan` `RaisedLevels` and `PerBox` on an unraised tree (nothing raised or counted);
`check_defaults` the default; the harness test on W1 at kD = 30, 10⁻⁷ (4,226.39 per box, the
trusted measure's to every digit).

## 3. The first count (job 653776)

The plan's first reading counted every V entry of a raised level's leaves. The trusted measure
prices over the first plan's pairs, which lack those the per-pair crossover already converted, so
on a level where that crossover converts many the first count's mean per pair is lower:

| case | level | first count: leaves, entries | crossover: first count, trusted | V pairs left: first count, trusted | M2L + P2P s: first count − trusted (defaults; time rule) |
| --- | ---: | --- | --- | --- | --- |
| W1, kD = 500, 10⁻⁷ | 6 | 18,829, 880,872 | 20,579, 30,073 | 39,839, 7,466 | +0.425; −0.136 |
| W1, kD = 500, 10⁻⁸ | 6 | 18,829, 880,872 | 28,496, 45,180 | 11,160, 0 | +0.254; +0.234 |

(At 10⁻⁷ the per-pair plan leaves 567,612 of the 880,872 entries, 0.64 of them: 30,073 × 0.64 ≈ 19,400.)
Elsewhere the first count already matched the trusted plans (W1 at kD = 100 and 200: every raised
leaf level converts all its pairs under either). The count now leaves out what the per-pair price
converts (43f29ad); §4–§6 are the runs after it.

## 4. Pair evidence

Per case and raised level (job 653782, forward; the reverse order gives the same plans, crossovers
and counts): the counts before the plan, the crossovers (per pair, per box, the trusted measure's
after its first plan), the V pairs left, the near interactions and the level's M2L and P2P seconds
of each (`levels`). The time rule's rows are the same but where noted.

| case | level | p_w | counted leaves, entries | crossover: pair, box, trusted | V pairs left: pair, box, trusted | M2L + P2P s: pair, box, trusted |
| --- | ---: | ---: | --- | --- | --- | --- |
| W1, kD = 100, 10⁻⁷ | 3 | 28 | 96, 3,912 | 3,882, 23,626, 21,436 | 12,216, 12,216, 12,216 | 0.067, 0.077, 0.074 |
| | 4 | 28 | 944, 17,648 | 3,882, 51,498, 51,498 | 17,648, 0, 0 | 0.165, 0.040, 0.041 |
| W1, kD = 100, 10⁻⁸ | 4 | 32 | 935, 15,006 | 5,026, 83,064, 83,064 | 15,006, 0, 0 | 0.169, 0.039, 0.039 |
| W1, kD = 200, 10⁻⁷ | 4 | 28 | 552, 25,887 | 3,882, 20,529, 20,974 | 53,247, 53,016, 53,016 | 0.256, 0.277, 0.278 |
| | 5 | 28 | 3,479, 76,735 | 3,882, 43,649, 43,649 | 76,735, 0, 0 | 0.410, 0.138, 0.134 |
| W1, kD = 200, 10⁻⁸ | 5 | 32 | 3,479, 58,551 | 5,026, 79,211, 79,211 | 58,551, 0, 0 | 0.389, 0.138, 0.130 |
| W1, kD = 500, 10⁻⁷ | 6 | 28 | 17,730, 567,612 | 3,882, 30,073, 30,073 | 567,612, 7,466, 7,466 | 2.219, 1.579, 1.556 |
| W1, kD = 500, 10⁻⁸ | 6 | 32 | 17,538, 517,489 | 5,026, 45,180, 45,180 | 517,489, 0, 0 | 2.312, 1.431, 1.372 |
| W3, kD = 200, 10⁻⁷ | 4 | 28 | 3,226, 460,635 | 3,882, 6,743, 6,751 | 584,136 (all three) | 1.182, 1.367, 1.396 |
| W3, kD = 200, 10⁻⁸ | 4 | 32 | 3,226, 460,635 | 5,026, 9,336, 9,348 | 584,136 (all three) | 1.226, 1.380, 1.388 |

- **Where every pair of a raised level is between leaves** (W1's levels 4–6, the Also's level 5 at
  kD = 200, 10⁻⁸ among them) the per-box crossover is the trusted measure's to the unit, and both
  plans leave the same pairs: all 58,551 of the Also's level convert in one plan.
- **Where a raised level holds interior boxes beside leaves** (W1's level 3 at kD = 100, level 4
  at kD = 200; W3's level 4) the per-box crossover counts the leaves alone and differs from the
  trusted measure's level mean by 0.1–10 %; the plans differ by 0–231 pairs (W1 at kD = 200, level 4)
  and the seconds by the spread.
- **W3's dense raised level 4** (256 points per leaf, n_s n_t up to 256² against crossovers
  6,743–9,348): no plan converts a pair; W3 at kD = 500 raises no level with pairs left (its level 6
  converts all 44,278,020 under the per-pair price already). The three plans of W3 are the same, so
  its seconds' changes are spread (forward +0.03 to +0.19 s, reverse +0.005 to +0.13 s, one plan).

## 5. FMM evidence

Per case and setting, both orders (forward: job 653782's `w1`, `w3`; reverse: its `reverse`), the
raised levels' M2L and P2P summed under each run, per box against per pair and against the trusted
measure, and the evaluation (`summary`):

| case | M2L | M2L + P2P s: pair, box, trusted (forward / reverse) | box − pair s | box − trusted s | evaluate box ÷ pair | φ ÷ pair (box = trusted) |
| --- | --- | --- | --- | --- | --- | --- |
| W1, kD = 100, 10⁻⁷ | defaults | 0.232, 0.117, 0.115 / 0.238, 0.110, 0.095 | −0.115 / −0.128 | +0.002 / +0.015 | 0.666 / 0.627 | 0.991 |
| | time rule | 0.213, 0.095, 0.089 / 0.240, 0.103, 0.094 | −0.118 / −0.137 | +0.006 / +0.009 | 0.604 / 0.598 | 0.991 |
| W1, kD = 100, 10⁻⁸ | defaults | 0.240, 0.117, 0.111 / 0.252, 0.101, 0.103 | −0.123 / −0.152 | +0.007 / −0.003 | 0.721 / 0.614 | 1.000 |
| | time rule | 0.229, 0.102, 0.140 / 0.234, 0.105, 0.119 | −0.127 / −0.129 | −0.038 / −0.015 | 0.668 / 0.673 | 1.000 |
| W1, kD = 200, 10⁻⁷ | defaults | 0.666, 0.415, 0.411 / 0.864, 0.503, 0.519 | −0.251 / −0.361 | +0.003 / −0.016 | 0.857 / 0.767 | 0.983 |
| | time rule | 0.783, 0.445, 0.426 / 0.822, 0.321, 0.345 | −0.338 / −0.501 | +0.019 / −0.025 | 0.753 / 0.529 | 0.983 |
| W1, kD = 200, 10⁻⁸ | defaults | 0.633, 0.420, 0.416 / 0.914, 0.564, 0.493 | −0.213 / −0.350 | +0.004 / +0.071 | 0.902 / 0.843 | 1.000 |
| | time rule | 0.796, 0.499, 0.494 / 0.924, 0.431, 0.386 | −0.297 / −0.493 | +0.005 / +0.045 | 0.857 / 0.621 | 1.000 |
| W1, kD = 500, 10⁻⁷ | defaults | 2.219, 1.579, 1.556 / 2.185, 1.605, 1.580 | −0.639 / −0.580 | +0.024 / +0.024 | 0.973 / 0.952 | 0.756 |
| | time rule | 2.217, 1.527, 1.607 / 2.236, 1.485, 1.633 | −0.690 / −0.751 | −0.081 / −0.148 | 0.891 / 0.871 | 0.756 |
| W1, kD = 500, 10⁻⁸ | defaults | 2.312, 1.431, 1.372 / 2.316, 1.427, 1.389 | −0.881 / −0.889 | +0.059 / +0.039 | 0.982 / 0.911 | 1.000 |
| | time rule | 2.328, 1.439, 1.409 / 2.356, 1.380, 1.420 | −0.888 / −0.976 | +0.031 / −0.040 | 0.883 / 0.858 | 1.000 |
| W3, kD = 200, 10⁻⁷ | defaults | 1.182, 1.367, 1.396 / 1.378, 1.382, 1.369 | +0.185 / +0.005 | −0.029 / +0.013 | 1.167 / 1.020 | 1.000 |
| | time rule | 1.398, 1.423, 1.391 / 1.327, 1.456, 1.325 | +0.025 / +0.129 | +0.032 / +0.131 | 1.016 / 1.099 | 1.000 |
| W3, kD = 200, 10⁻⁸ | defaults | 1.226, 1.380, 1.388 / 1.342, 1.360, 1.361 | +0.155 / +0.019 | −0.008 / −0.000 | 1.138 / 1.009 | 1.000 |
| | time rule | 1.346, 1.389, 1.504 / 1.368, 1.413, 1.417 | +0.043 / +0.045 | −0.115 / −0.004 | 1.024 / 1.059 | 1.000 |
| W3, kD = 500, 10⁻⁷ | defaults | no raised level with pairs left | – | – | 0.992 / 1.007 | 1.000 |
| | time rule | | – | – | 0.989 / 1.019 | 1.000 |
| W3, kD = 500, 10⁻⁸ | defaults | | – | – | 1.009 / 1.002 | 1.000 |
| | time rule | | – | – | 0.922 / 1.037 | 1.000 |

- **W1**: per box against per pair, the raised levels gain 0.12–0.98 s per case in both orders, the
  evaluation 0.53–0.98× (forward 0.60–0.98×, reverse 0.53–0.95×); against the trusted measure
  −0.15 to +0.07 s, the sign changing with the order on several cases: the same within the
  spread.
- **W3**: the same plan under every run (§4), the checksums equal; the changes are the spread of one
  plan (CH4.3b2c's Also on W3 at kD = 200 sees the same).
- **Errors**: per box the trusted measure's on every case to the printed digits (bit for bit on 36
  of 40 runs, §6); against the per pair price 0.756 / 0.577 (φ / ∇φ) on W1 at kD = 500, 10⁻⁷, 0.983
  / 0.892 at kD = 200, 0.991 / 0.745 at kD = 100, 1.000 at 10⁻⁸ and on W3: within 1.1× everywhere
  (the converted pairs are exact).
- **Build**: the trusted measure's second plan costs 0.026 s (W1 at kD = 100), 0.12 s (200),
  1.07–1.09 s (500), 0.21 s (W3 at kD = 200). The count, after it ran on the pool (job 653786; the
  build part "counts", which also holds the point counts' exchange): 0.003–0.025 s per box against
  0.003–0.017 s per pair on W1, 0.011 against 0.003 s on W3 at kD = 200, 0.43–0.45 against 0.082 s
  at kD = 500 (5.3 s serially in 653782, where every pair is converted already and none is left to
  count); the builds within their spread (W3 at kD = 500: 16.9–17.0 s per box, 17.4 s per pair).
- **A second plan after the per-box one** (`box+raised`) converts at most 7,466 more pairs (W1 at
  kD = 500, 10⁻⁷, the defaults: the per-box plan's own remaining pairs give a higher mean; −0.27 s of
  evaluation, inside the spread) and elsewhere none.

## 6. The time rule on the one plan

Under the time rule at 10⁻⁷, W1 at kD = 200 and 500 (both orders), the per-box run is not the
trusted measure's output bit for bit:

- **kD = 500**: the time rule decides each level's M2L after the plan, from the plan's shape. The
  trusted measure decides on its first plan (567,612 V pairs on level 6) and keeps the waves; the
  per-box run decides on its one plan (7,466 V pairs left) and takes level 6 from the waves to RCR.
  Its level 6 runs 1.527 / 1.485 s against the trusted measure's 1.607 / 1.633 s (−0.08 / −0.15 s).
- **kD = 200**: the time rule takes level 4 (raised, waves predicted) for the form after the plan;
  the per-box price had already converted 231 of its pairs (53,016 left against 53,247), which the
  trusted measure, naming only the levels that keep their waves, does not. Level 4's M2L and P2P:
  0.270 s per box against 0.266 s for the trusted measure (forward), within the spread.

Both follow from one plan that the operator reads: the choices are made on the plan that runs.

## 7. Questions for sign-off

1. **The acceptance as read** (§4, §5): the per-box price before the plan converts the trusted
   measure's pairs in one plan on every raised level of W1 and W3 at 10⁻⁷ and 10⁻⁸ (W3's dense level
   4 none under either), its output the trusted measure's bit for bit on 36 of 40 runs and §6's
   decision on the other 4, the errors within 1.1× (0.58–1.00×) the per-pair price's, the count
   0.35 s at most. *Recommended: accept.*
2. **Adoption**: `VRaisedPrice::PerBox` the builder's default (made so in this PR: CH4.3b2c then
   measures the defaults with it). *Recommended: adopt.* *Alternative:* keep `PerPair` the default
   and adopt with CH4.3b2c.
3. **`VAfterPlan` stays opt-in** (with `RaisedLevels` added), a measurement tool: after the per-box
   plan a second plan converts at most 7,466 more pairs, within the spread. *Recommended: accept, no
   task.*

Signed off 2026-11-21 by the user on #247, "All sign off questions as recommended": 1, the
acceptance as read; 2, `VRaisedPrice::PerBox` the default; 3, `VAfterPlan` stays opt-in, no task.
