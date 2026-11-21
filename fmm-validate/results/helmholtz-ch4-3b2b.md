# Helmholtz CH4.3b2b: the V rule on the form's levels, and the defaults' errors at 10⁻⁶

Task: docs/helmholtz/tasks/CH4.3b2b.md (roadmap CH4.3b2b, the second of CH4.3b2a–c; proposal §10
CH4.3; §3.6, §3.7, §5.3). CH4.3b1's report §5 left two questions about the V near rule (CH4.0d2,
`VNear::CostModel`): why the defaults err up to 1.35× Rotation's at 10⁻⁶, and whether the rule's
crossover on a level the time rule takes from the waves, set before the plan for the waves' cost,
should be decided again after the plan. CH4.3b2a's report §6 added the level that keeps 58,551 V
pairs under the defaults' plan against 2,296 under Rotation's (W1 at kD = 200, level 5, 10⁻⁸).

## 0. Summary

1. **The defaults' errors are the V rule's plan, not the waves' M2L** (§2). With the V rule off
   the defaults err 0.966–1.024× Rotation's (φ) on W1 at kD = 100–1000 and W3 at kD = 200 and 500,
   at 10⁻³ and 10⁻⁶. With it on, 1.000–1.344×, the excess coming with the V pairs converted
   (qualitatively, at 10⁻⁶; at 10⁻³ the leaf pairs carry little of the error):
   RCR is dear per pair, so Rotation's plan converts 1.6–9.2× as many pairs between leaves into
   exact near pairs (W1 at kD = 200, 10⁻⁶: 63,426 of 136,740 against 19,131; 1.344×). Where both
   convert every pair (W3 at kD = 500) the ratio is 1.000. Nothing to change: each plan converts
   what pays for its own M2L, and the errors stay within the tolerance.
2. **The V rule after the plan** (`VAfterPlan`, opt-in, §3): after the operator's choice each level
   gets the crossover of the M2L it runs where that is higher, and the plan is built once more.
   Tested (2 threads, P ranks the one-rank FMM bit for bit, errors within 1.1×), measured on
   Kathleen in two run orders.
3. **On the levels the time rule took from the waves: justified as it is** (§4, §5). The form's
   crossover after the plan is 1.2–1.4× the waves' per-pair one at ε ≥ 10⁻⁶; the second plan adds
   few pairs (W1 at kD = 1000: 2,813,175 → 2,433,718 V pairs on level 7), and the measured change of
   the levels' M2L and P2P is −0.12 to +0.02 s per case (at most 3 % of the evaluation; its size
   varies up to 6× and on two of six cases its sign with the run order), against a second plan as dear
   as the first (0.02–4.6 s of build).
4. **On the raised levels that keep their waves (10⁻⁷, 10⁻⁸): the plan converts far too few pairs**
   (§5, §6). The waves' per-pair constant prices a pair at b (p_w + 1)², but a raised level's
   waves cost per box at p_w; on W1's sparse leaf levels (levels 4–6) the after-plan crossover is
   8–17× higher and converts every V pair between leaves (98.7 % on one level). Measured in both
   run orders: the levels' M2L and P2P fall by 0.09–0.75 s per case, linearly in the pairs and
   interactions converted at the levels' measured costs (within 2–32 %); the evaluation by up to 44 % (W1 at kD = 100 and 200), by 8–13 % under
   the time rule at kD = 500 (−0.2 to 7 % under the defaults there, within the spread); the errors
   at 10⁻⁷ to 0.58–0.99× (at 10⁻⁸ unchanged). The Also's level is one of them: all 58,551 pairs convert, its waves' 0.34–0.46 s
   become 0.05–0.07 s more P2P.
5. **On unraised levels with waves (ε ≥ 10⁻⁶) the after-plan crossover over-converts, by the
   levels' own seconds** (§5): on W1
   at kD = 500 and 1000 the levels' M2L and P2P grow by 0.02–0.45 s in both orders against a predicted fall,
   because converting part of a level's pairs leaves its boxes' waves (the mean cost per pair
   over-states what a pair saves).
6. **Recommended** (§7): keep the plan as it is on the levels taken from the waves; fix the waves'
   crossover on raised levels before the plan, in one plan, as a follow-up task before CH4.3b2c
   (a second plan costs as much as the first: 1.0–1.1 s on W1 at kD = 500, 4.6 s at 1000).

## 1. Setup

- **Runs.** `helmholtz_accuracy` (release, BLAS threads 1), one rank with 40 threads bound to the
  node's cores, launched by `tools/helmholtz/ch4-3b2b-runs.sh`; §9.1's complex charges and 1,000
  samples, gradients, the W and X rules of the defaults, the interpolated build, two evaluations
  (the rows the second's).
  - `errors`: W1 at kD = 100, 200, 500, 1000 and W3 at N = 10⁶, kD = 200 and 500, at 10⁻³ and
    10⁻⁶, `--m2l rotation,auto-pw --near-v off,cost`.
  - `w1`, `w3`: W1 at kD = 100, 200, 500 (10⁻³, 10⁻⁶, 10⁻⁷, 10⁻⁸) and 1000 (10⁻⁶), W3 at N = 10⁶,
    kD = 200 and 500 (the same ε): Rotation with the V rule (its own launch), then the defaults
    with `--diagonal off,auto --v-after-plan off,taken,all` (`taken` only with the time rule):
    five runs in this order.
  - `reverse`: the w1 cases and W3 at kD = 200 (10⁻⁶, 10⁻⁷) with `--diagonal auto,off
    --v-after-plan all,taken,off`, the same runs in the reverse order.
- **Machine.** Kathleen, `singlenode`, 1 × 40 (Xeon Gold 6248):
  - job 653717 (node-c11a-021, 2026-11-21 00:29–01:36, at 9c1f72e), `errors`, `w1`, `w3`, 54
    launches, every one exit 0;
  - job 653724 (node-c11a-006, 02:02–02:17, at 5184d30), `reverse`, 15 launches, every one exit 0.
  The second plan of 9c1f72e reused the first plan's index (the same on one rank; on several the
  fix of 3ecfa8a builds a fresh one, whose time 653724's "replan" includes). Both jobs ran before
  c315554, which raises no level whose first plan holds no V pair between leaves: their runs built
  a second plan for nothing on W3 at kD = 500 (14 s, the same plan) and where only levels without
  leaf pairs rose; the outputs are those of c315554 bit for bit. c315554 still builds a second,
  identical plan where a raised level's leaf pairs all lie above the new crossover (W3 at
  kD = 200, level 4, at 10⁻⁷ and 10⁻⁸: 0.21 s).
- **Raw outputs**: `~/projects/fmm-archive/kathleen/runs/ch4-3b2b/` and `ch4-3b2b-reverse/`.
- **Tables**: `tools/helmholtz/ch4-3b2b-tables.py errors|runs|levels|summary <dir>`.
- **Spread.** Two runs of the same plan in one launch (`taken` and `all` where no waves level rose)
  differ by up to 30 % in the evaluation and 50 % in a small level's seconds (W1 at kD = 200, 10⁻³:
  0.317 against 0.415 s; W3 at kD = 200, 10⁻⁷ and 10⁻⁸, an unchanged plan: +0.32 and +0.41 s of
  2.4 and 2.8 s), and a launch's later runs tend to be slower (CH4.3b2a's §5.2). §5 therefore
  reads the levels' own seconds (M2L and P2P of the raised levels whose plan changed, measured
  within each run) and both orders, and treats a change whose sign flips with the order as noise.

## 2. The defaults' errors at 10⁻⁶

The defaults' errors over Rotation's (relative ℓ2 at the samples), with the V rule off and with
the cost model's, and the V pairs between leaves each plan converts (of those offered):

| case | ÷ Rotation's, V rule off (φ, ∇φ) | ÷ Rotation's, V rule on (φ, ∇φ) | converted: Rotation, defaults | offered |
| --- | --- | --- | --- | ---: |
| W1, kD = 100, 10⁻³ | 1.000, 1.000 | 1.026, 1.036 | 11,934, 6,020 | 36,192 |
| W1, kD = 200, 10⁻³ | 1.000, 1.000 | 1.010, 1.008 | 42,918, 26,909 | 136,740 |
| W1, kD = 500, 10⁻³ | 1.000, 1.000 | 1.009, 1.010 | 296,984, 142,608 | 880,872 |
| W1, kD = 1000, 10⁻³ | 1.000, 1.000 | 1.009, 1.005 | 1,134,776, 567,207 | 3,513,338 |
| W3, kD = 200, 10⁻³ | 1.000, 1.000 | 0.991, 1.011 | 237,610, 25,836 | 605,214 |
| W3, kD = 500, 10⁻³ | 1.000, 1.000 | 1.000, 1.000 | 44,278,020, 44,278,020 | 44,278,020 |
| W1, kD = 100, 10⁻⁶ | 0.998, 0.983 | 1.143, 1.220 | 18,198, 6,682 | 36,192 |
| W1, kD = 200, 10⁻⁶ | 0.990, 0.989 | 1.344, 1.374 | 63,426, 19,131 | 136,740 |
| W1, kD = 500, 10⁻⁶ | 0.966, 0.969 | 1.153, 1.156 | 381,358, 177,703 | 880,872 |
| W1, kD = 1000, 10⁻⁶ | 1.009, 1.000 | 1.049, 1.088 | 1,456,760, 700,163 | 3,513,338 |
| W3, kD = 200, 10⁻⁶ | 1.024, 1.010 | 1.037, 1.047 | 241,986, 135,935 | 605,214 |
| W3, kD = 500, 10⁻⁶ | 1.009, 0.996 | 1.000, 1.005 | 44,278,020, 44,278,020 | 44,278,020 |

- With the V rule off the waves' M2L errs as Rotation's (0.966–1.024×; CH6.1e's 1.07× bound), so
  the excess is the plan's.
- The V rule converts a pair below the crossover of the level's cheapest M2L: RCR's at p (W1 at
  p = 15: 3,904 on x86) against the waves' (b (p + 1)², 1,181), per (parent, source) on a level
  with waves (`v_group`). A converted pair is exact, so the plan that converts more errs less:
  Rotation's own error falls from 1.02·10⁻⁷ to 7.01·10⁻⁸ on W1 at kD = 200, 10⁻⁶ with the rule.
- At 10⁻³ the M2L's truncation dominates less of the error (1.009–1.036×); at 10⁻⁶ the leaf-level
  pairs carry more of it (1.049–1.344×). CH4.3b1's first runs gave the levels up before the plan,
  so the rule ran at RCR's crossover there (50,902 converted on W1 at kD = 200): 1.034×.

The excess is relative to a plan with more exact pairs; the defaults stay within ε (9.4·10⁻⁸ at
10⁻⁶). Nothing changes here (question 1).

## 3. The change

`HelmholtzFmmBuilder::v_after_plan(VAfterPlan)` (opt-in; `Off` the default), under
`VNear::CostModel` only. After the operator's choice, `v_after_plan_crossover` prices a V pair of
each named level by the M2L it runs at its degree, over the P2P's per interaction:

- `TakenLevels`: a level the time rule took from the waves, the form's pair
  (`DiagonalCosts::pair`) or RCR's (`VCosts::rotation`) where no class takes the form;
- `AllLevels`: those, and a level that keeps its waves, the waves' cost per box with a V row
  (`DiagonalCosts::plane_wave`) times those boxes over the level's pairs (the mean per pair).

A named level whose first plan holds a V pair between two leaves (summed over the ranks; c315554)
takes the higher of the two crossovers; if any rose, the plan is built once more on a fresh index:
it converts the first plan's pairs and those the higher crossover adds (per (parent, source) on a
level that keeps its waves, per pair elsewhere), so its V lists lie within the first plan's, every
buffer the operator sized holds, and the operator's M2L per level stays. From agreed costs and
shapes, so every rank builds the second plan or none. Build part "replan"; `HelmholtzFmm::{
v_after_plan, v_crossovers, v_crossovers_after_plan, replanned}`. Tests: the closed form
(`tests/helmholtz/v_crossover.rs`); `check_v_after_plan` in `tests/helmholtz_fmm.rs` (the sphere at
kD = 30: the M2L per level unchanged, the levels raised exactly where named, V pairs per level at
most and conversions at least the setting off's, errors within 1.1×, 2 threads and the one-rank
FMM bit for bit, Rotation unchanged); the harness's columns at W1, kD = 30, 10⁻⁷.

## 4. Pair evidence

Per raised level, the run with the V rule after the plan against the same run without it (job
653717; `levels`): the crossovers, the V pairs, and the change of the level's M2L and P2P seconds
against the change predicted from the level's measured costs (the M2L's seconds per pair and the
P2P's per interaction of the run without it). This checks that a converted pair saves the level's
mean M2L cost, the after-plan rule's assumption; it does not check the cost model's constants
themselves (the crossovers come from `DiagonalCosts` and `VCosts`). A selection:

| case | run | level | p | M2L | crossover | V pairs | Δ M2L + P2P s | predicted s |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| W1, kD = 100, 10⁻⁷ | defaults + all | 4 | 28 | waves | 3,882 → 51,498 | 17,648 → 0 | −0.112 | −0.109 |
| W1, kD = 200, 10⁻⁷ | defaults + all | 5 | 28 | waves | 3,882 → 43,649 | 76,735 → 0 | −0.255 | −0.259 |
| W1, kD = 200, 10⁻⁸ | defaults + all | 5 | 32 | waves | 5,026 → 79,211 | 58,551 → 0 | −0.285 | −0.278 |
| W1, kD = 500, 10⁻⁷ | defaults + all | 6 | 28 | waves | 3,882 → 30,073 | 567,612 → 7,466 | −0.457 | −0.530 |
| W1, kD = 500, 10⁻⁸ | defaults + all | 6 | 32 | waves | 5,026 → 45,180 | 517,489 → 0 | −0.679 | −0.665 |
| W3, kD = 200, 10⁻⁶ | defaults + all | 5 | 15 | waves | 1,182 → 14,753 | 106,051 → 0 | −0.134 | −0.162 |
| W1, kD = 500, 10⁻³ | defaults + all | 6 | 13 | waves | 905 → 4,126 | 738,264 → 558,205 | +0.041 | −0.031 |
| W1, kD = 500, 10⁻⁶ | defaults + all | 6 | 16 | waves | 1,334 → 6,762 | 703,169 → 452,599 | +0.115 | −0.018 |
| W1, kD = 1000, 10⁻⁶ | defaults + all | 7 | 16 | waves | 1,334 → 6,752 | 2,813,175 → 1,788,564 | +0.399 | −0.077 |
| W1, kD = 1000, 10⁻⁶ | time + taken | 7 | 16 | form | 1,334 → 1,664 | 2,813,175 → 2,433,718 | −0.118 | −0.092 |
| W1, kD = 500, 10⁻⁶ | time + taken | 6 | 16 | form | 1,334 → 1,664 | 703,169 → 597,164 | −0.024 | −0.029 |
| W1, kD = 200, 10⁻⁶ | time + taken | 5 | 15 | form | 1,182 → 1,452 | 104,993 → 88,842 | −0.011 | −0.006 |

- **Where every pair of a level converts** (the raised sparse leaf levels, W3's level 5 at 10⁻⁶),
  the level's waves go with its pairs and the mean is the right price: measured within 2–25 % of
  the prediction here (2–32 % over every such level in both orders, §5).
- **Where part of a dense level converts** (unraised, ε ≥ 10⁻⁶), its boxes keep their waves: a
  converted pair saves about half the mean (W1 at kD = 1000: 0.42 s for 1.02·10⁶ pairs against
  the mean's 0.87 s) while its P2P costs what it predicts, so the level loses.
- **The form's levels** convert few more pairs (its crossover 1.2–1.4× the waves' at p_w = p_l,
  1.5× at p_w = 28);
  the changes are tens of milliseconds, near the prediction.

## 5. FMM evidence

Per case and run, the measured change of the raised levels whose plan changed, summed, the
prediction, the evaluation's
change (each the run with the V rule after the plan over the same run without it; `summary`), in
both orders (forward: job 653717, the setting after its reference; reverse: job 653724, before it):

| case | run | Δ M2L + P2P s, forward / reverse | predicted s | Δ evaluate s, forward / reverse | evaluate s (without) | replan s | φ, ∇φ ÷ |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| W1, kD = 100, 10⁻⁷ | defaults + all | −0.112 / −0.111 | −0.109 | −0.106 / −0.109 | 0.303 | 0.026 | 0.991, 0.745 |
| W1, kD = 100, 10⁻⁷ | time + all | −0.105 / −0.129 | −0.094 | −0.111 / −0.132 | 0.261 | 0.026 | 0.989, 0.745 |
| W1, kD = 100, 10⁻⁸ | defaults + all | −0.090 / −0.105 | −0.092 | −0.056 / −0.092 | 0.331 | 0.027 | 1.000, 1.000 |
| W1, kD = 100, 10⁻⁸ | time + all | −0.102 / −0.121 | −0.100 | −0.100 / −0.129 | 0.341 | 0.026 | 1.000, 1.000 |
| W1, kD = 200, 10⁻⁷ | defaults + all | −0.249 / −0.450 | −0.260 | −0.214 / −0.594 | 1.123 | 0.125 | 0.983, 0.892 |
| W1, kD = 200, 10⁻⁷ | time + all | −0.207 / −0.316 | −0.259 | −0.109 / −0.359 | 0.866 | 0.124 | 0.983, 0.891 |
| W1, kD = 200, 10⁻⁸ | defaults + all | −0.284 / −0.365 | −0.279 | −0.314 / −0.315 | 1.369 | 0.124 | 1.000, 1.000 |
| W1, kD = 200, 10⁻⁸ | time + all | −0.184 / −0.499 | −0.271 | −0.057 / −0.634 | 1.209 | 0.126 | 1.000, 1.000 |
| W1, kD = 500, 10⁻⁷ | defaults + all | −0.457 / −0.469 | −0.530 | +0.017 / −0.423 | 7.645 | 1.073 | 0.756, 0.577 |
| W1, kD = 500, 10⁻⁷ | time + all | −0.439 / −0.446 | −0.518 | −0.357 / −0.557 | 4.706 | 1.082 | 0.756, 0.577 |
| W1, kD = 500, 10⁻⁸ | defaults + all | −0.679 / −0.749 | −0.665 | −0.226 / −0.682 | 9.111 | 1.069 | 1.000, 1.000 |
| W1, kD = 500, 10⁻⁸ | time + all | −0.713 / −0.727 | −0.681 | −0.814 / −0.830 | 6.447 | 1.076 | 1.000, 1.000 |
| W3, kD = 200, 10⁻⁶ | defaults + all | −0.134 / −0.239 | −0.162 | +0.431 / −0.265 | 1.992 | 0.210 | 0.986, 0.967 |
| W1, kD = 100, 10⁻⁶ | defaults + all | −0.012 / −0.018 | −0.028 | −0.021 / −0.035 | 0.195 | 0.026 | 0.612, 0.643 |
| W1, kD = 200, 10⁻⁶ | defaults + all | −0.022 / +0.045 | −0.050 | −0.066 / +0.054 | 0.724 | 0.122 | 0.702, 0.695 |
| W1, kD = 500, 10⁻³ | defaults + all | +0.041 / +0.023 | −0.031 | +0.616 / +0.016 | 4.510 | 1.016 | 0.966, 0.967 |
| W1, kD = 500, 10⁻⁶ | defaults + all | +0.115 / +0.098 | −0.018 | +0.574 / −0.056 | 5.567 | 1.037 | 0.787, 0.808 |
| W1, kD = 1000, 10⁻⁶ | defaults + all | +0.399 / +0.447 | −0.077 | −0.664 / −2.583 | 37.654 | 4.631 | 0.926, 0.896 |
| W1, kD = 100, 10⁻⁶ | time + taken | −0.002 / −0.004 | −0.002 | −0.006 / −0.005 | 0.111 | 0.024 | 0.914, 0.866 |
| W1, kD = 200, 10⁻⁶ | time + taken | −0.014 / +0.004 | −0.009 | −0.036 / +0.014 | 0.446 | 0.118 | 0.770, 0.748 |
| W1, kD = 500, 10⁻³ | time + taken | −0.050 / −0.009 | −0.015 | −0.246 / −0.064 | 2.557 | 1.031 | 1.000, 0.990 |
| W1, kD = 500, 10⁻⁶ | time + taken | −0.024 / +0.024 | −0.027 | −0.093 / +0.136 | 3.057 | 1.025 | 0.965, 0.945 |
| W1, kD = 1000, 10⁻⁶ | time + taken | −0.118 / −0.028 | −0.092 | −0.601 / +0.561 | 14.273 | 4.585 | 0.981, 0.973 |
| W3, kD = 200, 10⁻⁶ | time + taken | −0.041 / −0.014 | −0.045 | −0.119 / +0.156 | 2.250 | 0.206 | 0.988, 0.970 |

- **Raised levels that keep their waves (10⁻⁷, 10⁻⁸; `all`)**: in both orders the levels gain
  0.09–0.75 s per case, as predicted, and the evaluation falls by 12–44 % on W1 at kD = 100 and 200
  (5 % on one run) and by 8–13 % under the time rule at kD = 500 (6.45 → 5.63 s forward, 6.47 →
  5.64 s reverse at 10⁻⁸); under the defaults at kD = 500 by −0.2 to 7 %, the spread of a 7–10 s
  evaluation. At 10⁻⁷ the errors fall to 0.58–0.99× (the converted pairs' truncation gone); at 10⁻⁸
  they are unchanged.
- **Unraised levels with waves (ε ≥ 10⁻⁶; `all`)**: small and mixed on W1 at kD = 100–200; a
  loss of 0.02–0.45 s of the levels' own seconds on the dense levels of W1 at kD = 500 and 1000 in
  both orders (§4), although the evaluation's changes there (−2.6 to +0.6 s at kD = 1000) are
  spread and fell in both orders at kD = 1000; W3 at kD = 200's level 5 gains 0.13–0.24 s (every
  pair converts).
- **Levels taken from the waves (`taken`)**: −0.12 to +0.02 s, the size varying up to 6× and on two
  of six cases the sign with the order; at most 3 % of the evaluation, below the spread.
- **The second plan** costs what the first does: 0.02 s (W1, kD = 100), 0.12 s (200), 1.0–1.1 s
  (500), 4.6 s (1000), 0.21 s (W3, kD = 200), and 14 s on W3 at kD = 500, where no more pairs
  could convert (c315554 now skips that). Against one evaluation the raised levels' gain pays for
  it at kD ≤ 200 and half pays at kD = 500; the levels taken from the waves never do.

## 6. The sparse level of the Also

W1 at kD = 200, level 5 (k a 5.41, p_l 22 raised to p_w = 32 at 10⁻⁸): the defaults' plan keeps
58,551 V pairs, Rotation's 2,296 (CH4.3b2a §6). The V rule priced the level for the waves at
b (p_w + 1)² = 5,026 per pair (x86), but the level's waves cost 3,700 (p_w + 1)² + 129 (p_w + 1)³
thread-ns per box with a V row, about 15× that over its pairs (79,211): the level is sparse (few
pairs per box) and raised, so its cost is per box. After the plan every one of its 58,551 V pairs
converts: its waves (0.335 s forward, 0.461 s reverse) go, its P2P grows by 0.051 and 0.069 s, the
case's evaluation falls by 0.31 s (1.37 → 1.06 s and 1.75 → 1.43 s). CH4.3b1's "31×" is this: the
V rule's waves crossover, not the waves' M2L. The same holds on every raised leaf level measured
(W1 levels 4–6 at 10⁻⁷ and 10⁻⁸).

## 7. Questions for sign-off

1. **The explanation as read** (§2): the defaults' excess over Rotation's errors at 10⁻⁶ is the V
   rule's plan (fewer exact near pairs under the waves' crossover), not the waves' M2L (0.966–1.024×
   with the rule off); nothing changes. *Recommended: accept.*
2. **The levels taken from the waves: justified as it is** (§4, §5): the after-plan crossover of
   the form adds few pairs, the measured change is within the spread (−0.12 to +0.02 s, ≤ 3 %), and a
   second plan costs as much as the first. The plan stays as built; `VAfterPlan` stays opt-in, a
   measurement tool and the trusted path of question 3's follow-up. *Recommended: accept.*
   *Alternative:* adopt `TakenLevels` with CH4.3b2c's default.
3. **The raised levels with waves** (§5, §6, the Also): the waves' V crossover under-prices a raised
   level (p_w > p_l, ε < 10⁻⁶), which then keeps 10–100 % of its leaf pairs on the waves at 0.09–0.75 s
   per case. *Recommended:* a follow-up task, CH4.3b2d, before CH4.3b2c on the priority list: the
   waves' V crossover on a raised level priced per box before the plan (one plan), with
   `VAfterPlan::AllLevels` on raised levels as its trusted measure, pair and FMM evidence on this
   report's cases (and W3's dense raised levels, where converting every pair may not pay: n_s n_t
   up to 256² per pair). *Alternative:* adopt `AllLevels` restricted to raised levels with CH4.3b2c's
   default, a second plan in the build (+1.0–4.6 s on W1 at kD = 500–1000).
4. **Unraised levels with waves**: by the levels' own seconds the after-plan mean over-converts
   dense levels (§4; the evaluation is within its spread there); not adopted,
   no task. *Recommended: accept.*

Signed off 2026-11-21 by the user on #245, "Q1-4 as recommended": 1, the explanation accepted,
nothing changes; 2, the levels taken from the waves justified as they are, `VAfterPlan` stays
opt-in; 3, the follow-up CH4.3b2d before CH4.3b2c, added to the roadmap; 4, not adopted, no task.
