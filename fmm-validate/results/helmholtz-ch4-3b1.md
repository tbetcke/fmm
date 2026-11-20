# Helmholtz CH4.3b1: the diagonal form's time rule

Task: docs/helmholtz/tasks/CH4.3b1.md (roadmap CH4.3b1, the first half of CH4.3b; proposal §10
CH4.3, "the time crossover per class; the fallback of a whole level to RCR when the scratch budget
binds"; §3.6, §5.3, §5.4). Until here the diagonal form took every offset class that passes the
bound E(l, c) ≤ θ ε on the levels whose M2L is RCR's (`HelmholtzDiagonal::Rule`, CH4.2b1–CH4.3a),
whatever it cost, and never a level with the exponential M2L. This task gives it a rule by time,
opt-in as `HelmholtzDiagonal::Auto`, so that CH4.3b2 can measure it as the default.

## 0. Summary

1. **The per-class crossover is never reached.** A diagonal pair costs 0.08–0.44× an RCR pair on
   Kathleen and 0.09–0.52× on locust at every degree from 8 to 245 (§2.2). The conversions are
   paid per box, not per class, so once a level pays them every class that passes the bound is
   faster diagonal. The decision is per level.
2. **The cost model** (§2): every unit a (p + 1)² + b (p + 1)³ thread-ns of a whole level call
   (wall × threads, idle threads included): S → F and F → R per box with a V row, the diagonal
   pair, RCR's pair, and the exponential M2L per box with a V row at its degree p_w. Fitted on
   Kathleen (1 × 40) and locust (1 × 72) from W1 at kD = 100–1000 and W3 at N = 10⁶, kD = 200
   and 500, at 10⁻³, 10⁻⁶ and 10⁻⁸ (16 launches per machine, each launch six settings). The
   level's counts are the plan's, agreed over the ranks (`DiagonalShape`: boxes with a V row,
   pairs per class).
3. **The waves go by the box, the form by the pair** (§3). Both passes of the exponential M2L
   cost about the same per box with a V row whatever the pairs per box (merge and shift
   900–3,800 (p_w + 1)² ns per box on locust, W1 at about 45 pairs per box and W3 at about 165
   alike), while the diagonal form pays per pair. So the order between the two turns on the
   level's pairs per box: the form wins on W1's surface, the waves on W3's dense levels. The
   plan's first design decided the order before the plan, on a uniform interior level (189 pairs
   per box) with the waves costed per pair; the first runs refuted it (W3's dense levels on the
   form at up to 1.9× the waves' time), and it was rebuilt to decide after the plan, on the
   level's own counts.
4. **The rule's choices** (§4): in the final runs, with the refitted constants built in, the
   time rule's choice per level is within 1.05× of the fastest option measured on 123 of 127
   levels on Kathleen (0.96 s lost of 64.8 s at the best, 1.5 %) and 132 of 140 on locust (0.14 s
   of 18.5 s, 0.8 %). The first model managed 74 of 127 (5.1 s lost) and 118 of 140 (1.5 s). The
   misses left are W3's small dense levels at kD = 200, where the waves were kept and the form was
   1.1–1.9× faster.
5. **The acceptance** (§5): under Rotation the time rule is 1.000–1.001× Rotation's errors on
   every case on both machines, and the evaluation 0.27–0.84× Rotation's on Kathleen, 0.38–1.00×
   on locust. Under the defaults it is 0.99–1.04× the defaults' errors at 10⁻³ and 10⁻⁶, and the
   evaluation 0.34–0.83× the defaults' on Kathleen, 0.43–1.00× on locust. At 10⁻⁸ the levels it
   takes from the waves return to Rotation's errors (at most 1.015× Rotation's), 1.07–2.2× the
   defaults' (φ): the waves' raised degree (CH6.1g) made the defaults more accurate there. That is
   the matched-accuracy question of CH4.3b2.
6. **Below k a = 2.8** (§6, CH4.3a's question 3): the bound takes classes on one level of the
   cases, locust's W1 at kD = 200, level 6 (k a = 2.71: 34 classes at 10⁻³, 9 at 10⁻⁶); the time
   rule took neither (RCR is cheaper on such small levels). The time rule now takes no level
   below 2.8 (`DIAGONAL_AUTO_MIN_KA`); `Rule` keeps reaching below it. Question 3.

## 1. Setup

- **Runs.** `helmholtz_accuracy` (release, BLAS threads 1), one rank with every core, launched
  by `tools/helmholtz/ch4-3b1-runs.sh`: W1 at kD = 100, 200, 500 (10⁻³, 10⁻⁶, 10⁻⁸) and 1000
  (10⁻⁶); W3 at N = 10⁶, kD = 200 and 500 (10⁻³, 10⁻⁶, 10⁻⁸). §9.1's complex charges and 1,000
  samples, gradients, the near rules of the defaults, the interpolated build (the library's),
  two evaluations (the rows the second's). Each launch runs `--m2l rotation,auto-pw` (RCR on
  every level; the library's defaults) × `--diagonal off,rule,auto`.
- **Machines.**
  - Kathleen job 653672 (node-c11a-080, `singlenode`, 1 × 40 bound to every core, 2026-11-20
    19:57–20:27), at 164ddd7 (the first model);
  - locust (GH200, 1 × 72; load 1.1 and the GPU idle before), 2026-11-20 19:47–19:59, at
    164ddd7;
  - the final runs, `auto` (`--diagonal off,auto` alone) at 14b2103 with the refitted constants
    and the decision after the plan: Kathleen job 653681 (node-c11a-080, 20:31–20:51), locust
    (20:30–20:51): §4.3. The budget's fallback before a level leaves its waves (the self-review,
    after 14b2103) changes nothing here: at one rank per node every level's scratch fits.
- **Raw outputs**: `~/projects/fmm-archive/{kathleen,locust}/runs/ch4-3b1/` (first runs) and
  `ch4-3b1-auto/` (final runs).
- **Tables**: `tools/helmholtz/ch4-3b1-fit.py fit` (the constants), `check` (the choices),
  `summary` (errors and evaluations). The harness's new table "Time rule" prints, per level, k a,
  the bound's classes, the M2L that ran and the model's seconds by each option.

## 2. The cost model

### 2.1 The constants

Each unit's cost a (p + 1)² + b (p + 1)³ thread-ns, fitted by least squares on the relative
error, each level weighted by its measured time (a level of a few boxes is mostly overhead, and
the rule's choices matter where the time is). The diagonal units come from the levels where every
class is diagonal ("Diagonal M2L"'s thread-seconds of S → F, the pairs and F → R); RCR's pair and
the waves per box from "M2L per level" (wall × threads over the level's pairs, or its boxes with a
V row).

| unit | Kathleen [a, b] | model ÷ measured (levels ≥ 0.05 s) | locust [a, b] | model ÷ measured |
| --- | --- | --- | --- | --- |
| S → F per box | [383, 2.30] | 0.80–1.10 (median 1.04) | [108, 1.58] | (no level ≥ 0.05 s; all 0.66–1.50) |
| F → R per box | [313, 5.85] | 0.73–1.23 (0.98) | [92, 3.26] | 0.85–0.86 |
| diagonal pair | [28.1, 0.548] | 0.59–1.58 (0.97) | [13.0, 0.224] | 0.59–1.52 (1.04) |
| RCR pair | [0, 8.40] | 0.61–1.16 (0.99) | [0, 3.20] | 0.38–1.21 (1.08) |
| waves per box (p_w) | [3700, 129] | 0.20–1.46 (0.79) | [1210, 39.2] | 0.46–1.79 (0.94) |

The spread per level is wide (the diagonal pair's cost depends on the level's size, its transfer
functions' fit in cache and the class mix; RCR's on whether the level runs by pairs, CH4.0i),
which is why the rule is judged by its choices (§4), not by the fit. `VCosts` (CH4.0d2) stays the
V near rule's; its exponential M2L per pair measured thread time on the leaf levels and is 4–7×
below a whole level call's.

### 2.2 The per-class crossover

The diagonal pair over RCR's pair, by the fitted constants:

| p | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 131 | 180 | 245 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kathleen | 0.44 | 0.32 | 0.26 | 0.20 | 0.17 | 0.13 | 0.12 | 0.10 | 0.09 | 0.08 | 0.08 |
| locust | 0.52 | 0.38 | 0.31 | 0.23 | 0.19 | 0.15 | 0.13 | 0.11 | 0.10 | 0.09 | 0.09 |

Below 1 at every degree: no class is cheaper on RCR once its level's conversions are paid. A
level's decision weighs the conversions (per box) against the pairs' savings: a level whose
passing classes hold few of its pairs (12 of 34 classes at 10⁻⁸ on W1's deepest V level, the near
classes there failing the bound) is near the break-even, as CH4.2b2 measured (0.95–1.02× RCR).

## 3. The waves by the box

The exponential M2L's two passes on locust, per box with a V row (thread-ns over (p_w + 1)²):

| case | level | p_w | pairs per box | waves per source | merge and shift per box | merge and shift per pair |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1, kD = 500, 10⁻⁶ | 5 | 25 | 48 | 817 | 1,382 | 28.8 |
| W1, kD = 1000, 10⁻⁶ | 7 | 16 | 46 | 713 | 1,123 | 24.5 |
| W3, kD = 200, 10⁻⁶ | 4 | 22 | 143 | 782 | 1,637 | 11.5 |
| W3, kD = 500, 10⁻⁶ | 5 | 25 | 165 | 782 | 1,530 | 9.3 |

Per box the two workloads cost the same; per pair W3 costs a third of W1. The diagonal form pays
per pair (the pair) and per box (the conversions), so on W3's dense levels the waves win where on
W1 the form does. The plan decided the order before the plan, on a uniform interior level with the
waves per pair (so that the builder's V rule and the operator read the same function, CHA.11); the
first runs put W3's dense levels on the form at 1.1–1.9× the waves' time on locust (levels 4 and
5 at kD = 500). The rule now decides after the plan, in the operator's `select_tables`, on the
level's agreed shape, with the waves per box:

- the builder's V rule reads `wave_levels` before the plan as before: a superset of the levels
  that keep their waves. Its grouping per (parent, source) (CH4.0d2) converts exact pairs under
  every M2L, so a level the form takes from the waves keeps an exact plan; only the V rule's
  crossover on such a level was set for the waves (a cost heuristic, never an accuracy one);
- a level taken from the waves runs at its own degree (not CH6.1g's raised one), takes no
  batched tables, and runs the cheaper of the form and RCR by the same model (on Kathleen at
  kD = 200, 10⁻⁸, level 5, the raised waves took 0.363 s, RCR 0.012 s and the form with its 3
  classes 0.020 s).

This departs from the roadmap's wording of CH4.3b1 as split on this branch ("the exponential M2L
before it, in `wave_levels`"); question 2.

## 4. The rule's choices

Per level with V pairs, the options measured in the same launch: RCR (Rotation without the
form), the form with the bound's classes (Rotation with the rule), and the setting's own M2L
(the defaults' waves; Batched levels left out, the rule does not consider them). "Within 1.05×"
is the choice's measured seconds within 1.05× of the fastest.

### 4.1 The first model (the runs at 164ddd7)

| machine | levels | within 1.05× | seconds lost | seconds at the best |
| --- | ---: | ---: | ---: | ---: |
| Kathleen | 127 | 74 | 5.11 | 64.76 |
| locust | 140 | 118 | 1.46 | 18.52 |

### 4.2 The refitted model, replayed from the measured counts

| machine | levels | within 1.05× | seconds lost | seconds at the best |
| --- | ---: | ---: | ---: | ---: |
| Kathleen | 127 | 122 | 1.04 | 64.76 |
| locust | 140 | 133 | 0.14 | 18.52 |

The misses left: W3 at kD = 200 (levels 3–5 under the defaults, where the form was 1.1–1.9×
faster than the waves the model kept: the waves' constant overestimates small dense levels), and
three W1 levels at 10⁻⁸ with 3 classes passing, where the form with them is 1.1–1.8× RCR.

### 4.3 The final runs (14b2103)

The `auto` stage (`--diagonal off,auto`) with the refitted constants and the decision after the
plan: Kathleen job 653681 (node-c11a-080, `singlenode`, 1 × 40, 2026-11-20 20:31–20:51, 16
launches, every one exit 0), locust 1 × 72 (20:30–20:51). Each choice is scored by the option it
took, timed in the first runs' launch that measured every option (the same tree and degrees): a
level's own seconds vary by 5–30 % from launch to launch on small levels, so the run's own
seconds against another launch's best would score noise (the check prints both).

| machine | levels | within 1.05× | seconds lost | seconds at the best |
| --- | ---: | ---: | ---: | ---: |
| Kathleen | 127 | 123 | 0.96 | 64.76 |
| locust | 140 | 132 | 0.14 | 18.52 |

The misses (above 1.05×):

| machine | case | setting | level | p | k a | RCR s | form s | own M2L s | took | ÷ best |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Kathleen | W1, kD = 500, 10⁻⁸ | Rotation | 6 | 24 | 6.77 | 0.940 | 0.855 | – | RCR | 1.10 |
| Kathleen | W3, kD = 200, 10⁻³ | defaults | 4 | 18 | 10.83 | 0.792 | 0.257 | waves 0.467 | waves | 1.82 |
| Kathleen | W3, kD = 200, 10⁻⁶ | defaults | 4 | 22 | 10.83 | 1.349 | 0.386 | waves 0.497 | waves | 1.29 |
| Kathleen | W3, kD = 500, 10⁻⁸ | defaults | 4 | 44 | 27.06 | 9.769 | 2.044 | waves 2.490 | waves | 1.22 |
| locust | W1, kD = 200, 10⁻³ | defaults | 6 | 7 | 2.71 | 0.0001 | 0.0002 | RCR 0.0001 | RCR | 1.09 |
| locust | W1, kD = 200, 10⁻⁶ | defaults | 6 | 15 | 2.71 | 0.0002 | 0.0005 | waves 0.0019 | waves | 11.8 |
| locust | W3, kD = 200, 10⁻³ | defaults | 3–5 | 30–11 | 21.65–5.41 | | | | waves | 1.31–1.89 |
| locust | W3, kD = 200, 10⁻⁸ | defaults | 3, 4 | 38, 32 | 21.65, 10.83 | | | | waves | 1.07–1.30 |
| locust | W3, kD = 500, 10⁻³ | defaults | 4 | 36 | 27.06 | 1.093 | 0.278 | waves 0.303 | waves | 1.09 |

The W3 levels are the model's: the waves' constant per box overestimates the form's cost against
them on small dense levels (the fit's spread, §2.1). Locust's W1 level 6 at kD = 200 lies below
k a 2.8, where the time rule leaves the level to the defaults (§6); a level of 2 ms.

## 5. The acceptance: errors and evaluations

The final runs (§4.3). Per case, the time rule's errors (φ/∇φ, relative ℓ2 at §9.1's samples)
over the same setting's without the form and over Rotation's without it, and the evaluation (the
second's) over the setting's without the form.

| machine | setting | ÷ the setting's errors (φ) | ÷ Rotation's errors (φ/∇φ, at most) | evaluation ÷ the setting's |
| --- | --- | --- | --- | --- |
| Kathleen | Rotation | 1.000–1.001 | 1.001/1.000 | 0.27–0.84 |
| Kathleen | defaults, 10⁻³ and 10⁻⁶ | 0.990–1.037 | 1.347/1.384 | 0.34–0.83 |
| Kathleen | defaults, 10⁻⁸ | 1.07–2.21 | 1.015/1.030 | 0.47–0.78 |
| locust | Rotation | 1.000–1.001 | 1.001/1.000 | 0.38–1.00 |
| locust | defaults, 10⁻³ and 10⁻⁶ | 0.992–1.021 | 1.279/1.260 | 0.43–1.00 |
| locust | defaults, 10⁻⁸ | 1.00–2.21 | 1.007/1.010 | 0.63–0.97 |

- **Under Rotation** (the reading of CH4.2b1–CH4.3a: the form against the same setting without
  it, under the setting that lets the rule reach every level) the time rule keeps Rotation's
  errors on every case, and the evaluation drops to 0.27–1.00×.
- **Under the defaults** at 10⁻³ and 10⁻⁶ it keeps the defaults' errors (0.99–1.04×). The
  defaults themselves err up to 1.35× Rotation's (W1 at kD = 200, 10⁻⁶: 1.344× without the form
  on Kathleen), which the form does not change. In the first runs, where the waves were given up
  before the plan (so the builder's V rule saw no waves on those levels), the same levels gave
  0.77–0.95× the defaults' errors at 10⁻⁶ (W1 at kD = 100–1000 on Kathleen): the defaults' excess
  over Rotation seems to come with the V rule's treatment of the levels with waves, not with the
  waves' M2L. Left to CH4.3b2 (its Also).
- **At 10⁻⁸** the levels taken from the waves return to Rotation's errors (at most 1.015×
  Rotation's), 1.07–2.21× the defaults' where the waves had run at the raised degree (W1; W3 at
  kD = 500 1.07×; on locust's W3 the rule kept the waves). Question 4.

## 6. Below k a = 2.8

CH4.3a's question 3 (signed off 2026-11-20): whether the time rule takes the diagonal form below
S6's calibrated range, where one damped class measured 2.9 E per pair (direct build, η = 0.1,
k a 1.69). On these cases the bound passes classes below 2.8 on one level, locust's W1 at kD = 200,
level 6 (k a = 2.71, NEON's 128-point leaves give one more level than Kathleen's 256): 34 classes
at 10⁻³ and 9 at 10⁻⁶, real k. The time rule took neither (RCR measured 2–2.5× faster than the
form there, a level of under 0.5 ms). On Kathleen no level with V pairs lies below k a = 5.41. The time rule now takes no
level below 2.8 (`DIAGONAL_AUTO_MIN_KA`), so the default of CH4.3b2 stays in S6's range; it
changes none of the runs above.

## 7. Questions for sign-off

1. **The acceptance as read.** The per-class crossover is never reached (§2.2), so the rule is
   per level; `HelmholtzDiagonal::Auto` takes a level's bound classes where the model's diagonal
   level beats RCR, or Auto's waves, on the level's agreed counts; under Rotation within 1.001×
   Rotation's errors on every case (under the defaults the defaults' own errors at 10⁻³ and 10⁻⁶,
   up to 1.35× Rotation's without the form; at 10⁻⁸ question 4); its choices within 1.05× of the fastest on 122/127 (Kathleen) and 133/140
   (locust) levels replayed, 123/127 and 132/140 in the final runs (§4.3). *Recommended: accept.*
2. **The order against the waves after the plan** (§3), not before it as the split on this
   branch worded CH4.3b1: the waves cost per box and the form per pair, so the order needs the
   level's counts; the builder's V rule keeps reading the waves before the plan (a superset,
   exact under every M2L). *Recommended: accept, the roadmap's CH4.3b1 line as amended in this PR.*
   *Alternative:* the uniform level before the plan (the first runs: 5.1 s lost of 64.8 s on
   Kathleen).
3. **No level below k a = 2.8 under the time rule** (§6). *Recommended: as built* (S6's range;
   no measured case changes). *Alternatives:* the bound for damped k only (the finding's case);
   or none (real k measured within 1.09 E per pair down to k a 1.35, CH4.3a).
4. **At 10⁻⁸ the time rule gives up accuracy for time** (§5): taking a level from the raised
   waves returns the errors to Rotation's, 1.07–2.2× the defaults'. *Recommended:* CH4.3b2
   decides at matched accuracy as its line says (the form at the raised degree on such a level,
   or the waves kept below 10⁻⁷ where they are the more accurate); no change here.
