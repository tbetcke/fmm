# Helmholtz CH4.3b2a: the diagonal form's time rule at matched accuracy

Task: docs/helmholtz/tasks/CH4.3b2a.md (roadmap CH4.3b2a, the first of CH4.3b2a–c; proposal §10
CH4.3, "the time crossover per class"; §3.6, §5.3, §5.4). CH4.3b1 built the time rule
(`HelmholtzDiagonal::Auto`, opt-in): on a level whose M2L would be Auto's exponential M2L, it took
the level from the waves where the form was cheaper and ran it at the level's own degree p_l. Below
10⁻⁶ the waves run at a raised degree p_w (CH6.1g, CH6.1h), which made the defaults 1.07–2.2× more
accurate than the time rule at 10⁻⁸ (its report §5, question 4, signed off 2026-11-20: "CH4.3b2
decides at matched accuracy"). This task keeps that degree.

## 0. Summary

1. **A level with waves keeps its resolving degree p_w** whatever M2L the time rule gives it: its
   stores, its M2L and its leaves' operators at p_w, its translations as `PlaneWaveRaise` gives the
   waves. The form's classes are the bound's at p_w. The degrees and translation degrees of every
   run with the time rule are the defaults', checked in the tests on every level.
2. **Matched accuracy holds** (§3): the defaults with the time rule err 0.990–1.037× the defaults'
   errors (φ and ∇φ) on every case on both machines, W1 at kD = 100–1000 and W3 at N = 10⁶, kD = 200
   and 500, at 10⁻³, 10⁻⁶, 10⁻⁷ and 10⁻⁸; at 10⁻⁸ 0.993–1.000×, where CH4.3b1 measured 1.07–2.2×.
3. **The order on a level with waves** (§4): the waves unless the form beats them at p_w, then the
   cheaper of the form and RCR. The plan's rule of three (RCR against the waves too) ran first: on
   the seven levels where it put RCR against the waves, RCR lost 0.47 s on three sparse W1 levels
   (1.07–1.52× the waves) and gained 0.11 s on four, RCR's constant (fitted at the degrees given) under-predicting RCR at p_w on sparse
   levels. Question 2.
4. **The choices** (final runs, §5): within 1.05× of the fastest option on 81 of 83 levels on
   Kathleen (0.54 s lost of 48.0 s at the best) and 82 of 92 on locust (0.28 s of 13.8 s). Every
   miss keeps the waves where the form was faster, all but two on W3's dense levels (CH4.3b1's misses,
   CH4.3b2c's Also). The score does not time RCR on a level with waves where no class passes (the
   rule never takes RCR there); on two such levels the first runs had RCR faster (0.09 s, §4).
5. **The evaluation** under the time rule: 0.38–1.12× the defaults' on Kathleen, 0.43–0.99× on
   locust; at 10⁻⁸ 0.71–1.03× and 0.71–0.98×. On Kathleen W3 at kD = 200 runs 1.03–1.12× although its
   choices are within 1.05×: the same waves level ran 1.18–1.37× slower in the time rule's run of the
   launch than in the defaults' (§5.2). Left to CH4.3b2c (question 4).
6. **The sparse level of CH4.3b1's Also** (31× RCR, W1 at kD = 200, level 5, 10⁻⁸; §6): at matched
   degree and the same plan RCR costs as much as the waves there (0.32 against 0.30 s on Kathleen).
   The 31× was the plan: with waves predicted on the level the V near rule converts few pairs (58,551
   V pairs against Rotation's plan's 2,296). That is CH4.3b2b's question, added to its line.
7. **At the raised degree the bound passes fewer classes at low k a** (§3.2): on the cube at k = 40,
   level 2 (k a 8.7) 16 classes at p_w = 28 (10⁻⁷) and 5 at p_w = 32 (10⁻⁸); the bound's rounding
   term grows with p. So the form takes fewer of the raised levels than CH4.3b1's runs at p_l did.

## 1. Setup

- **Runs.** `helmholtz_accuracy` (release, BLAS threads 1), one rank with every core, launched by
  `tools/helmholtz/ch4-3b2a-runs.sh`: W1 at kD = 100, 200, 500 (10⁻³, 10⁻⁶, 10⁻⁷, 10⁻⁸) and 1000
  (10⁻⁶); W3 at N = 10⁶, kD = 200 and 500 (the same ε). §9.1's complex charges and 1,000 samples,
  gradients, the near rules of the defaults, the interpolated build (the library's), two evaluations
  (the rows the second's). Each launch runs the library's defaults (`--m2l auto-pw`) with
  `--diagonal off,auto --diagonal-costs machine,form,rcr`: the defaults, the time rule at the
  machine's costs, and the time rule at costs that force the form (`FORM_COSTS`) and RCR
  (`RCR_COSTS`) on the levels it reaches with a class that passes, all at the levels' degrees as the
  defaults run them, so that each option is timed at p_w in the same launch.
- **Machines.**
  - First runs (the rule of three, at c02d2b3): Kathleen job 653691 (node-c11a-044, `singlenode`,
    1 × 40 bound to every core, 2026-11-20 22:05–22:34, every launch exit 0); locust (GH200,
    1 × 72, 22:05–22:29). On locust another user's processes raised the load to a 15-minute average
    of 316 during W1 at kD = 1000 (its defaults' evaluation 160 s): that launch was rerun alone at
    22:29–22:32 with the load sampled every 30 s (1-minute load at most 33) and replaces it.
  - Final runs (the order of §4, at a4bc1c2): Kathleen job 653702 (node-c11a-018, `singlenode`, 1 × 40,
    queued 17 min, ran 22:57–23:26, every launch exit 0); locust 1 × 72, 22:39–22:51, the load
    sampled every 30 s, at most 45 (the run's own threads), every launch exit 0.
- **Raw outputs**: `~/projects/fmm-archive/{kathleen,locust}/runs/ch4-3b2a/` (first runs; locust's
  W1 at kD = 1000 from `ch4-3b2a-rerun/`, the contaminated one in `ch4-3b2a-contaminated/`) and
  `ch4-3b2a-final/` (final runs, with locust's `load.log`).
- **Tables**: `tools/helmholtz/ch4-3b2a-tables.py errors|choices|summary <machine>=<dir>`. The
  evaluation is read from the per-level table's Σ row (0.1 ms), not the main table's 0.01 s.

## 2. The change

`select_tables` under `HelmholtzDiagonal::Auto` and `HelmholtzM2l::Auto` with a tolerance: on each
level with waves in the time rule's range (k a ≥ 2.8), p_w = `plane_wave_degree(p_l, ε)`; the
bound's classes at p_w where the form's scratch at p_w fits the budget (none else);
`wave_level_choice` (§4) on the agreed shape. A level taken from the waves keeps p_w in the
degrees the tables are built at, takes no batched tables, and `diagonal::select` runs the form or
RCR on it at p_w. `HelmholtzOperator::taken_from_waves` reports it. At ε ≥ 10⁻⁶ p_w = p_l, so the
change there is the order alone.

## 3. Matched accuracy

### 3.1 The errors

The final runs, the time rule at the machine's costs over the defaults (relative ℓ2 at §9.1's
samples), and the evaluation (the second's) and the build over the defaults':

| case | Kathleen φ | ∇φ | evaluation | build | locust φ | ∇φ | evaluation | build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, kD = 100, 10⁻³ | 1.000 | 1.000 | 0.888 | 0.625 | 1.000 | 1.000 | 0.780 | 1.000 |
| W1, kD = 100, 10⁻⁶ | 1.006 | 1.022 | 0.600 | 1.100 | 1.007 | 1.016 | 0.761 | 1.000 |
| W1, kD = 100, 10⁻⁷ | 1.000 | 1.000 | 0.999 | 1.200 | 1.000 | 1.000 | 0.954 | 1.000 |
| W1, kD = 100, 10⁻⁸ | 1.000 | 1.000 | 1.006 | 1.000 | 1.000 | 1.000 | 0.977 | 1.000 |
| W1, kD = 200, 10⁻³ | 1.000 | 1.000 | 0.684 | 1.147 | 1.000 | 1.000 | 0.723 | 1.000 |
| W1, kD = 200, 10⁻⁶ | 1.002 | 1.007 | 0.548 | 1.171 | 1.000 | 1.007 | 0.727 | 1.037 |
| W1, kD = 200, 10⁻⁷ | 1.000 | 1.000 | 1.060 | 1.243 | 1.000 | 1.000 | 0.944 | 1.037 |
| W1, kD = 200, 10⁻⁸ | 1.000 | 1.000 | 0.923 | 1.222 | 1.000 | 1.000 | 0.944 | 1.045 |
| W1, kD = 500, 10⁻³ | 1.000 | 1.000 | 0.530 | 1.289 | 1.000 | 1.000 | 0.533 | 1.038 |
| W1, kD = 500, 10⁻⁶ | 1.037 | 1.035 | 0.523 | 1.308 | 1.021 | 1.011 | 0.580 | 1.056 |
| W1, kD = 500, 10⁻⁷ | 1.000 | 1.000 | 0.611 | 1.349 | 1.000 | 1.000 | 0.719 | 1.026 |
| W1, kD = 500, 10⁻⁸ | 1.000 | 1.000 | 0.713 | 1.375 | 1.000 | 1.000 | 0.731 | 1.075 |
| W1, kD = 1000, 10⁻⁶ | 1.000 | 1.004 | 0.379 | 1.214 | 0.992 | 1.000 | 0.425 | 1.073 |
| W3, kD = 200, 10⁻³ | 1.000 | 1.000 | 1.123 | 1.093 | 1.000 | 1.000 | 0.990 | 0.992 |
| W3, kD = 200, 10⁻⁶ | 0.990 | 0.994 | 1.051 | 1.171 | 1.000 | 1.000 | 0.973 | 0.984 |
| W3, kD = 200, 10⁻⁷ | 1.000 | 0.995 | 1.066 | 1.169 | 1.000 | 0.995 | 0.978 | 1.008 |
| W3, kD = 200, 10⁻⁸ | 0.993 | 0.996 | 1.026 | 1.146 | 1.000 | 1.000 | 0.954 | 1.000 |
| W3, kD = 500, 10⁻³ | 1.000 | 1.000 | 0.660 | 1.038 | 1.000 | 1.000 | 0.637 | 0.989 |
| W3, kD = 500, 10⁻⁶ | 1.000 | 0.995 | 0.659 | 1.005 | 1.000 | 1.000 | 0.625 | 1.009 |
| W3, kD = 500, 10⁻⁷ | 1.000 | 1.000 | 0.660 | 1.079 | 1.000 | 1.000 | 0.696 | 0.993 |
| W3, kD = 500, 10⁻⁸ | 1.000 | 1.000 | 0.718 | 1.034 | 1.000 | 1.000 | 0.706 | 1.009 |

At 10⁻⁸ the time rule now gives the defaults' errors: W1 at kD = 200 1.24·10⁻⁹ and 1.71·10⁻⁹
under both on locust (CH4.3b1's time rule at p_l: 2.01·10⁻⁹ and 3.62·10⁻⁹ on Kathleen, 1.6× and
2.1×). The forced costs, every option at p_w, stay within 0.978–1.037× too (the first and the final
runs; `errors`). The largest, 1.037× on W1 at kD = 500, 10⁻⁶, on Kathleen, comes with the form at
the level's own degree (p_w = p_l there) and lies in CH4.3b1's range under the defaults at 10⁻³ and
10⁻⁶ (0.99–1.04×); under Rotation the form gave 1.000–1.007× (CH4.3a).

The build is up to 1.38× the defaults' on Kathleen (W1 at kD = 500: the transfer functions at p_w on
40 threads; CH4.2c2's interpolated build) and up to 1.08× on locust; a first build, measured, not
part of the rule.

### 3.2 The classes at the raised degree

The bound E(l, c) ≤ θ ε at p_w passes fewer classes than at p_l at low k a: its rounding term grows
with the degree. In `tests/helmholtz/diagonal.rs` on the cube at k = 40, level 2 (k a 8.7) takes
16 classes at p_w = 28 (10⁻⁷) and 5 at p_w = 32 (10⁻⁸). On W1 at kD = 200, 10⁻⁸, level 4 (k a 10.8)
takes 16 at p_w = 32 against 29 at p_l = 24 in CH4.3b1, and the time rule keeps the waves there
(Kathleen's level 4 is the waves' at 0.24 s against the form's 0.08 s at p_l in CH4.3b1, not at
matched accuracy). Levels at k a ≥ 21 keep all or nearly all 34 classes.

## 4. The order on a level with waves

The plan's rule took the cheapest of three, RCR included. The first runs timed it: where it put
RCR against the waves (no class passing, or RCR modelled below both), RCR was:

| machine | case | level | p_w | k a | waves s | RCR s | took | ÷ best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Kathleen | W1, kD = 100, 10⁻⁷ | 4 | 28 | 5.41 | 0.129 | 0.074 | RCR | 1.00 |
| Kathleen | W1, kD = 100, 10⁻⁸ | 4 | 32 | 5.41 | 0.127 | 0.095 | RCR | 1.00 |
| Kathleen | W1, kD = 200, 10⁻⁷ | 5 | 28 | 5.41 | 0.315 | 0.412 | RCR | 1.31 |
| Kathleen | W1, kD = 200, 10⁻⁸ | 5 | 32 | 5.41 | 0.298 | 0.318 | RCR | 1.07 |
| locust | W1, kD = 500, 10⁻⁷ | 7 | 28 | 3.38 | 0.664 | 1.012 | RCR | 1.52 |
| locust | W3, kD = 500, 10⁻³ | 6 | 13 | 6.77 | 0.018 | 0.002 | RCR | 1.00 |
| locust | W3, kD = 500, 10⁻⁶ | 6 | 16 | 6.77 | 0.005 | 0.001 | RCR | 1.00 |

0.47 s lost (1.07–1.52× on three levels) and 0.11 s gained (four levels) against keeping the waves: RCR's constant, fitted in CH4.3b1 at the
degrees given, under-predicts RCR at p_w on these sparse, low-k a levels. The rule as built keeps
the waves unless the form beats them (both at p_w), then runs the cheaper of the form and RCR
(`wave_level_choice`; a tie to the waves, then to RCR), so RCR is taken only where the model has it
below the form and the form below the waves (locust's W3 at kD = 500, level 6, in the final runs:
1.00×). In the first runs' summary the rule of three lost 0.71 s of 45.2 s (Kathleen, 79/83 within
1.05×) and 0.66 s of 13.6 s (locust, 82/92); the final runs §5. The order as built gives up the two
levels where RCR beat the waves with no class passing at p_w (Kathleen, W1 at kD = 100, level 4 at
10⁻⁷ and 10⁻⁸: 0.074 and 0.095 s against the waves' 0.129 and 0.127 s, 0.09 s in all). Question 2.

## 5. The choices

### 5.1 Per level

Per level with V pairs in the rule's range (k a ≥ 2.8) and not Batched's, the options timed in the
same launch at the level's degree as run: the waves (the defaults' run), the form (the `form` run,
where a class passes) and RCR (the defaults' run on an RCR level, the `rcr` run on a level with waves
where a class passes); the choice scored by the option it took as timed there (CH4.3b1 §4.3). RCR
is not timed on a level with waves where no class passes, so the score cannot see RCR's wins there
(two levels in the first runs, 0.09 s, §4).

| machine | ε | levels | within 1.05× | seconds lost | seconds at the best | taken from the waves: form, RCR |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Kathleen | 10⁻³ | 19 | 18 | 0.06 | 6.74 | 11, 0 |
| Kathleen | 10⁻⁶ | 26 | 26 | 0.00 | 13.42 | 17, 0 |
| Kathleen | 10⁻⁷ | 19 | 19 | 0.00 | 13.19 | 10, 0 |
| Kathleen | 10⁻⁸ | 19 | 18 | 0.47 | 14.64 | 7, 0 |
| Kathleen | all | 83 | 81 | 0.54 | 47.99 | 45, 0 |
| locust | 10⁻³ | 22 | 18 | 0.12 | 1.57 | 11, 1 |
| locust | 10⁻⁶ | 29 | 28 | 0.01 | 4.63 | 17, 1 |
| locust | 10⁻⁷ | 21 | 19 | 0.07 | 4.23 | 7, 0 |
| locust | 10⁻⁸ | 20 | 17 | 0.08 | 3.32 | 5, 0 |
| locust | all | 92 | 82 | 0.28 | 13.75 | 40, 2 |

The misses, every one the waves kept where the form was faster at p_w:

| machine | case | level | p | k a | waves s | form s | RCR s | ÷ best |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kathleen | W3, kD = 200, 10⁻³ | 4 | 18 | 10.83 | 0.302 | 0.238 | 0.715 | 1.27 |
| Kathleen | W3, kD = 500, 10⁻⁸ | 4 | 44 | 27.06 | 2.382 | 1.910 | 8.565 | 1.25 |
| locust | W1, kD = 200, 10⁻⁷ | 4 | 28 | 10.83 | 0.051 | 0.031 | 0.050 | 1.65 |
| locust | W1, kD = 500, 10⁻⁸ | 5 | 32 | 13.53 | 0.186 | 0.169 | 0.314 | 1.10 |
| locust | W3, kD = 200, 10⁻³ | 3–5 | 30–11 | 21.65–5.41 | | | | 1.38–2.10 |
| locust | W3, kD = 200, 10⁻⁶ | 4 | 22 | 10.83 | 0.076 | 0.067 | 0.265 | 1.14 |
| locust | W3, kD = 200, 10⁻⁸ | 3 | 38 | 21.65 | 0.059 | 0.031 | 0.131 | 1.91 |
| locust | W3, kD = 500, 10⁻³, 10⁻⁷, 10⁻⁸ | 4 | 36–44 | 27.06 | | | | 1.07–1.11 |

W3's dense levels are CH4.3b1's misses (the waves' constant per box overestimates their cost against
the form there), CH4.3b2c's Also. The two W1 levels are raised levels where the form passes 27
classes at p_w.

### 5.2 W3 at kD = 200 on Kathleen

Every choice within 1.05×, yet the evaluation 1.03–1.12× the defaults'. The M2L levels are faster
under the time rule (their sum 0.598 against 0.622 s at 10⁻³, 1.001 against 1.076 s at 10⁻⁷), but
the waves' level 4, the same computation in both runs, took 0.414 against 0.302 s (10⁻³) and 0.856
against 0.727 s (10⁻⁷) in the time rule's run of the same launch, and the rest of the evaluation
grew too. In the first runs the same cases ran 0.92–1.06×, and locust's 0.95–0.99×. Run-to-run
spread on a two-socket node, or a placement effect of the form's buffers on the levels that follow;
these runs do not separate the two. Question 4: CH4.3b2c's runs repeat it.

## 6. The sparse level of CH4.3b1's Also

CH4.3b1 measured RCR at 0.012 s against the raised waves' 0.363 s on W1 at kD = 200, level 5, 10⁻⁸
(Kathleen), 31×. Under matched conditions:

| run | plan | p | V pairs | M2L | s |
| --- | --- | ---: | ---: | --- | ---: |
| CH4.3b1, Rotation | Rotation's | 24 | 2,296 | RCR | 0.012 |
| CH4.3b1, the defaults | the defaults' | 32 | 58,551 | waves | 0.363 |
| here, the defaults | the defaults' | 32 | 58,551 | waves | 0.298 |
| here, forced RCR | the defaults' | 32 | 58,551 | RCR | 0.318 |

On the same plan and degree RCR costs what the waves do. The 25× in V pairs is the V near rule: the
builder reads the waves' levels before the plan, and its crossover there is the waves' cost, which
converts far fewer pairs between leaves than RCR's. That is CH4.3b2b's subject (the V rule's
crossover on a level the form takes from the waves); its roadmap line now names this level.

## 7. Questions for sign-off

1. **The acceptance as read.** Under the time rule a level with waves keeps its resolving degree
   (the degrees and translation degrees the defaults', tested); the defaults with the time rule err
   0.990–1.037× the defaults' on every case of W1 and W3 at 10⁻³–10⁻⁸ on both machines (§3.1); the
   choices within 1.05× on 81/83 (Kathleen) and 82/92 (locust) levels, every miss the waves kept
   (§5.1; RCR not timed on a level with waves where no class passes: §4's two levels, 0.09 s); bit
   for bit at every thread count, chunk and rank count (tests, a raised level taken from the waves
   included). *Recommended: accept.*
2. **The order on a level with waves** (§4): the waves unless the form beats them at p_w, then the
   cheaper of the form and RCR; RCR never against the waves alone. This departs from the plan's and
   the roadmap's "cheapest of three" as split on this branch. *Recommended: as built, the roadmap's
   CH4.3b2a line amended in this PR.* *Alternative:* the rule of three, with RCR's constant refitted
   at the raised degree (the first runs: 0.47 s lost, 0.11 s gained).
3. **CH4.3b1's sparse level** (§6): closed here as a plan effect, carried to CH4.3b2b (the V rule's
   crossover), whose line names it. *Recommended: accept.*
4. **W3 at kD = 200 on Kathleen** (§5.2): the evaluation 1.03–1.12× the defaults' with every choice
   within 1.05×. *Recommended:* an Also on CH4.3b2c (repeat and separate spread from placement before
   the default); no change here.
