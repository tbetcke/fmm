# Helmholtz CH4.0: the M2L rule's sparse-level term

Helmholtz task CH4.0 ([docs/helmholtz/tasks/CH4.0.md](../../docs/helmholtz/tasks/CH4.0.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§2, §5.3, §6.4 and §10 CH4.0; before it [helmholtz-ch3-2d.md](helmholtz-ch3-2d.md) (the rule
by degree, its miss at 1.106) and [helmholtz-ch6-2h.md](helmholtz-ch6-2h.md) §5–§6 (Batched's
few-box levels, and its losses at 72 threads). Measured on 2026-11-12 on the M6 (Apple M6,
12 cores; [tools/m6/machine.md](../../tools/m6/machine.md)), locust (Neoverse-V2, 72 cores;
[tools/gh200/machine.md](../../tools/gh200/machine.md)) and Kathleen (Xeon Gold 6248, 40 cores;
[tools/kathleen/machine.md](../../tools/kathleen/machine.md)). Release builds, rustc 1.99.0,
one rank, f64, BLAS threads 1. Times are indicative (§9); nothing here is timed for a gate.

## Summary

- **The term** (`LevelShape`, `auto_batched_min_columns`). `HelmholtzM2l::Auto` now reads each
  level's V pairs and boxes over the whole tree (each box counted by the rank that computes
  it, summed over the ranks at build: rank-invariant, constraint 9) and runs Batched on a level
  only where m̂, its estimate of Batched's columns per table read, is at least m*(p) of the
  machine class (a table agreed over the ranks, the largest, as the crossover is), besides
  p_l ≤ the crossover d_b as before. m̂ is exact on full levels and
  within 10 % on clustered ones, and errs low by up to 1.5× on surface levels (§5).
- **m*** comes from both measures (§1, §2): the pair evidence (`helmholtz_m2l` at 1–64 columns
  per offset, degrees 3–15) gives its shape in p, fitted per degree as Batched's cost a/c + b;
  the FMM's per-level times at one thread give the margin the level call needs over the pair,
  per class: 1.45 on AVX-512 and AVX2 with two planes, 1.2 with one, 1.15 on NEON.
- **The acceptance: CH3.2d's sweeps within 1.10 on every case of the three machines** (§3).
  At one thread, CH3.2d's nine cases, the rule with the term is the per-level best on every
  case of the M6 and locust (1.000), within 1.008 on Kathleen AVX-512 and within 1.049 on
  Kathleen AVX2 (36 case × machine rows). CH3.2d's miss, Kathleen AVX-512 W1 at kD = 30 and
  10⁻⁶, goes from 1.196 (without the term, this run; 1.106 in CH3.2d's) to 1.000: the
  sparse level of p = 15 at 15 columns per read runs Rotation.
- **The split** (`batched_block_targets`). Batched's level call narrows its blocks to
  ⌈rows / threads⌉ (8 to 64 targets; the floor of 8, `M2L_MIN_BLOCK_TARGETS`, is a choice,
  not measured) on a level of fewer blocks than threads, so that every thread has a block. The level call is the pair chain bit for bit at every width (tests), so
  the output does not change: an execution split (constraint 9 as amended at H6.G). At 72
  threads on locust it takes level 2 (64 boxes, p = 7) from 4.3 to 0.8 ms and level 3 (512
  boxes) from 7.1 to 2.0 ms; Rotation takes 0.2 and 1.6 ms there (§4).
- **Under D8's default** (§4), at the machines' threads (the M6 1 × 6, Kathleen 1 × 40, locust
  1 × 72) on CH6.2h's matrix and large cases, the evaluation takes 0.36–1.07× the previous
  default's (0.54–1.07× on the M6, 0.57–1.05× on Kathleen, 0.36–1.02× on locust; ratios
  above 1.00 within the runs' spread) and is within 1.09× of the per-level best on the M6,
  1.21× on Kathleen and 1.28× on locust, the excess on ms-scale cases (1–2 ms: level 2's
  Batched at p = 7 and D8's waves on level 2). **Under the default the term changes no level
  of these cases** (the same checksum with and without it in all 172 launches): Batched's
  levels there have p ≤ 11 and at least 5.9 columns per read, and the waves take the levels
  where the term would bind. The gain at many threads is the split's. The term decides where
  the waves do not run: CH3.2d's rule (`auto_plane_wave(false)`), f32, levels with
  Im κ_l ≥ 5π and ε below 10⁻⁸.
- **Many threads: the roadmap's item is met in part** (§4). The roadmap asks the term to
  catch the levels where Batched at p = 7–8 loses at 72 threads below about 3.5·10⁵ pairs
  (CH6.2h §5), within constraint 2. The term cannot: it may not read the thread count, and
  at one thread Batched wins those levels. The split recovers most of the loss (the
  evaluation at 0.36–0.99× the previous default's on locust at 10⁻³), but with it Batched
  on levels of 64–5,000 boxes at 40–72 threads is still 1.1–5× Rotation's level time
  (0.2–2 ms per level; at most 1.28× of the per-level best on an evaluation, on ms-scale
  cases). Narrower blocks read each table for fewer columns; a split across the matrix's rows
  would keep them, and needs a rectangular entry point in `nd-fmm-simd`'s column kernel, a
  Laplace path (constraint 1): proposed as CH4.0f (§8). Whether CH4.0 counts as done with this
  residual left to CH4.0f was question 2 of the PR: CH4.0 done, CH4.0f off the priority list
  (the user, 2026-11-12, on #211, "Q2 as recommended").
- **The node budget** (§6). Each rank's dense budget is min(`dense_budget`,
  `dense_node_budget` / ranks per node), `DEFAULT_DENSE_NODE_BUDGET` 8 GiB (the waves' node
  budget); one rank per node is unchanged (1 GiB). Signed off (the user, 2026-11-12, on #211, question 1): 8 GiB,
  with a few ranks per node and threads per rank the advice.
- **Accuracy** (§7): the term moves levels between Batched and Rotation, which agree to
  rounding; every error of the default is the previous default's to three digits, and the
  one-thread runs' are Rotation's (1.000).

## 1. The pair evidence

`helmholtz_m2l --ps 3,…,15 --k 1 --columns c --isa <isa>` at c = 1, 2, 4, 8, 16, 32, 64, twice
(ascending, then descending): Rotation's time per pair over fused Batched's, the call
sequence of the level call on a block of c pairs per offset (the appendix has the tables).
For each degree a least-squares fit of y = (p³/n²) / R against 1/c, y = a/c + b (Batched's
cost per pair a/c + b in units of Rotation's p³/n²), gives the column count at which
Batched reaches a margin μ over Rotation, m*(p) = a / (p³/(μ n²) − b), infinite where
p³/(μ n²) ≤ b. The table takes the running maximum in p (m* never falls as p grows) and 64,
a full block's columns that no level reaches, for p ≥ 16 and where m* is infinite:

| class | μ | m* at p = 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15 |
| --- | ---: | --- |
| AVX-512, 1 plane | 1.2 | 0.4, 0.8, 1.1, 1.6, 2.5, 3.1, 3.8, 4.6, 5.4, 9.1, 11.8 |
| AVX-512, 2 planes | 1.45 | 0.7, 1.2, 1.6, 2.4, 3.9, 4.6, 5.7, 7.0, 10.4, 15.9, 28.2 |
| AVX2, 1 plane | 1.2 | 0.3, 0.7, 1.2, 1.9, 3.1, 3.9, 10.6, 14.5, 16.3, 32.3, 41.3 |
| AVX2, 2 planes | 1.45 | 0.6, 1.0, 2.1, 3.1, 5.6, 7.5, 37.5, 64, 64, 64, 64 |
| NEON, 1 plane | 1.15 | 0.3, 0.8, 2.1, 3.2, 8.0, 13.7, 13.7, 64, 64, 64, 64 |
| NEON, 2 planes | 1.15 | 0.4, 1.2, 3.9, 7.5, 23.7, 64, 64, 64, 64, 64, 64 |

- NEON's is the larger of the M6's and locust's per degree (the M6 needs more columns near
  the crossover: m*(9) with two planes 23.7 against locust's 13.4), except m*(11) with one
  plane, the M6's 13.7: locust's pairs at p = 11 with one plane are not monotone in c (its
  fit is infinite) while on both machines the level of 24.4 columns per read at p = 11 runs
  Batched faster (1.13× on locust, 1.33× on the M6). Taking locust's 64 there put the M6's
  W1 at kD = 100 with real charges at 1.117 of the best (§3); found after the M6's runs and
  corrected in 294c024.
- The scalar kernel (the reference path only) takes NEON's tables, unmeasured.
- The crossover d_b (`auto_batched_max_degree`, CH3.2d) stays a condition of its own: the
  term never lets Batched above it, and a named crossover is not undone by it (m* is finite).
- locust's pair data is noisier than the others' (c = 64 sometimes reads slower than 32); the
  fit per degree smooths it.

## 2. The level evidence and the margins

The FMM's per-level M2L times at one thread (`ch3-2d`, §3) where both Rotation and Batched
ran a level give Rotation ÷ Batched at the level; over the pair model's prediction at the
level's m̂ this is the margin the level call needs. By class (levels at or below the
crossover, where the choice is made):

| class | levels | Rotation ÷ Batched measured | margin (predicted ÷ measured) |
| --- | ---: | --- | --- |
| AVX-512, 2 planes | 16 | p = 15: 0.62 at 9.8 columns, 0.80 at 15.0, 0.97 at 21.1, 1.02 at 24.5, 1.11 at 28.9; p = 7–12: 1.11–4.28 | 1.16–1.96, mostly 1.36–1.55 |
| AVX-512, 1 plane | 5 | p = 15: 0.83 at 9.8, 1.42 at 21.1, 1.70 at 28.9 | 1.05–1.29 |
| AVX2, 2 planes | 16 | p = 7–8: 1.54–2.49; p = 11: 1.04 at 24.5 | 1.22–1.80, mostly 1.28–1.50 |
| AVX2, 1 plane | 5 | p = 7: 3.69, p = 11: 1.50; p = 15: 0.58–1.06 | 1.05–1.31 |
| NEON (locust), 2 planes | 8 | p = 7–8: 1.21–1.74 | 1.07–1.27 |
| NEON (the M6), 2 planes | 7 | p = 7–8: 1.11–1.78 | 0.89–1.32 |

So the level runs at 0.65–0.95 of the per-pair ratio on x86 with two planes and close to it
on NEON; the margins of §1 sit at the measured break-even: on AVX-512 with two planes
m*(15) = 28.2 keeps the cube's level 4 (28.9 columns, Batched 1.11×) and gives its level 3
(21.1, 0.97×) and W1's level 3 at kD = 30 (15.0, 0.80×, CH3.2d's miss) to Rotation.

## 3. CH3.2d's sweeps (the acceptance)

CH3.2d's nine cases (W1 at kD = 30 and 100, the W4 cube of 10⁵ points at kD = 10, each at
10⁻³ and 10⁻⁶; W3 at kD = 30 at 10⁻³; the cube at 10⁻⁶ and W1 at kD = 100 at 10⁻³ with
real charges), one rank × one thread, `--m2l rotation,batched-fused,auto --term on,off` and
the reverse order (ABBA), `auto` CH3.2d's rule (the exponential M2L off). The per-level
times of Rotation and of Batched on every level that fits (the mean of the two launches)
give each level's best; the rule's time at the final constants is evaluated on them (a level
runs Batched where the rule says so, as CH3.2d evaluated its crossovers), the evaluation
being Rotation's with its M2L replaced. `rule ÷ best` is the acceptance's ratio; `without
the term` is CH3.2d's rule on the same times.

| machine | ISA | rows | rule ÷ best | without the term ÷ best |
| --- | --- | ---: | --- | --- |
| the M6 | NEON | 9 | 1.000 on all | 1.000 on all |
| locust | NEON | 9 | 1.000 on all | 1.000 on all |
| Kathleen | AVX-512 | 9 | 1.000 on 8; W1 kD = 100, 10⁻⁶: 1.008 | 1.000–1.006 on 7; W1 kD = 30, 10⁻⁶: 1.196; W1 kD = 100, 10⁻⁶: 1.054 |
| Kathleen | AVX2 | 9 | 1.000 on 7; W1 kD = 100, 10⁻³: 1.016; the cube, 10⁻⁶, real: 1.049 | 1.000 on 8; the cube, 10⁻⁶, real: 1.049 |

- The cube at 10⁻⁶ with real charges on AVX2 (1.049 either way): the best batches the cube's
  levels at p = 15 with one plane, above AVX2's one-plane crossover 12 (CH3.2d's; the
  term does not change it).
- W1 at kD = 100, 10⁻³ on AVX2 (1.016): level 4 (p = 11, 24.4 columns, two planes) runs
  Rotation (m*(11) = 37.5); Batched was 1.04× there. The fit is conservative at many columns
  here: the unweighted fit a/c + b follows the few-column points, while AVX2's pairs at p = 11
  reach 1.68 at c = 32 (above the margin 1.45).

**Confirmation** at the final constants (the rule itself, not its evaluation):
`ch3-2d` again with the final constants (Kathleen job 633071 at 37e194d, locust at 37e194d
and, for W1 with real charges, 294c024): the rule's own run against the per-level best of the
same launches (`ch4-0-tables.py fmm`; the evaluation ÷ the best's, in brackets the M2L's):

| machine | ISA | rule ÷ best | without the term ÷ best | rule ÷ without the term |
| --- | --- | --- | --- | --- |
| locust | NEON | 1.000–1.003 (M2L 1.000–1.006) | M2L 1.000–1.010 | 0.994–1.003 |
| Kathleen | AVX-512 | 1.000–1.010 (M2L 1.000–1.016) | M2L 1.000–1.170 | 0.870–1.014 |
| Kathleen | AVX2 | 1.000–1.035 (M2L 1.000–1.040) | M2L 1.000–1.154 | 0.935–1.022 |

On AVX-512 the miss case (W1 at kD = 30, 10⁻⁶) runs 0.870× CH3.2d's rule's evaluation, its M2L
at the best (1.170× without the term); AVX2's 1.035 is the cube with real charges above its
crossover (above). The ratios between the rule and CH3.2d's rule elsewhere (0.99–1.02) are
the runs' spread (up to 1.46 between two launches of one level at one thread on Kathleen;
the same levels run the same calls). The M6 was not run again: its one level that the final
constants choose differently (W1 at kD = 100 with real charges, level 4) is evaluated above
from its own times.

The full table (the runs of the term `ch3-2d` stage; `p by level` from level 2):

| machine | ISA | case | charges | p by level | term: Batched levels | rule s | best s | rule ÷ best | without the term: levels | ÷ best | Rotation s |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| m6 | neon | W1 kD = 100, ε = 1e-3 | real | 30, 18, 11, 7 | 4,5 | 1.717 | 1.717 | 1.000 | 4,5 | 1.000 | 2.035 |
| m6 | neon | W1 kD = 100, ε = 1e-3 | complex | 30, 18, 11, 7 | 5 | 2.357 | 2.357 | 1.000 | 5 | 1.000 | 2.453 |
| m6 | neon | W1 kD = 100, ε = 1e-6 | complex | 35, 22, 15, 15 | – | 6.234 | 6.234 | 1.000 | – | 1.000 | 6.234 |
| m6 | neon | W1 kD = 30, ε = 1e-3 | complex | 12, 8 | 3 | 0.064 | 0.064 | 1.000 | 3 | 1.000 | 0.073 |
| m6 | neon | W1 kD = 30, ε = 1e-6 | complex | 16, 15 | – | 0.243 | 0.243 | 1.000 | – | 1.000 | 0.243 |
| m6 | neon | W3 (N = 108849) kD = 30, ε = 1e-3 | complex | 12, 8, 7 | 3,4 | 0.738 | 0.738 | 1.000 | 3,4 | 1.000 | 1.006 |
| m6 | neon | W4 cube (N = 100000) kD = 10, ε = 1e-3 | complex | 7, 7, 7 | 2,3,4 | 0.656 | 0.656 | 1.000 | 2,3,4 | 1.000 | 0.923 |
| m6 | neon | W4 cube (N = 100000) kD = 10, ε = 1e-6 | real | 15, 15, 15 | – | 3.453 | 3.453 | 1.000 | – | 1.000 | 3.453 |
| m6 | neon | W4 cube (N = 100000) kD = 10, ε = 1e-6 | complex | 15, 15, 15 | – | 4.248 | 4.248 | 1.000 | – | 1.000 | 4.248 |
| locust | neon | W1 kD = 100, ε = 1e-3 | real | 30, 18, 11, 7 | 4,5 | 3.765 | 3.765 | 1.000 | 4,5 | 1.000 | 4.141 |
| locust | neon | W1 kD = 100, ε = 1e-3 | complex | 30, 18, 11, 7 | 5 | 4.781 | 4.781 | 1.000 | 5 | 1.000 | 4.940 |
| locust | neon | W1 kD = 100, ε = 1e-6 | complex | 35, 22, 15, 15 | – | 12.537 | 12.537 | 1.000 | – | 1.000 | 12.537 |
| locust | neon | W1 kD = 30, ε = 1e-3 | complex | 12, 8 | 3 | 0.135 | 0.135 | 1.000 | 3 | 1.000 | 0.149 |
| locust | neon | W1 kD = 30, ε = 1e-6 | complex | 16, 15 | – | 0.475 | 0.475 | 1.000 | – | 1.000 | 0.475 |
| locust | neon | W3 (N = 108849) kD = 30, ε = 1e-3 | complex | 12, 8, 7 | 3,4 | 1.761 | 1.761 | 1.000 | 3,4 | 1.000 | 2.221 |
| locust | neon | W4 cube (N = 100000) kD = 10, ε = 1e-3 | complex | 7, 7, 7 | 2,3,4 | 1.570 | 1.570 | 1.000 | 2,3,4 | 1.000 | 2.046 |
| locust | neon | W4 cube (N = 100000) kD = 10, ε = 1e-6 | real | 15, 15, 15 | – | 6.740 | 6.740 | 1.000 | – | 1.000 | 6.740 |
| locust | neon | W4 cube (N = 100000) kD = 10, ε = 1e-6 | complex | 15, 15, 15 | – | 8.358 | 8.358 | 1.000 | – | 1.000 | 8.358 |
| kathleen | avx512 | W1 kD = 100, ε = 1e-3 | real | 30, 18, 11, 7 | 4,5 | 4.683 | 4.683 | 1.000 | 4,5 | 1.000 | 7.079 |
| kathleen | avx512 | W1 kD = 100, ε = 1e-3 | complex | 30, 18, 11, 7 | 4,5 | 5.538 | 5.538 | 1.000 | 4,5 | 1.000 | 7.469 |
| kathleen | avx512 | W1 kD = 100, ε = 1e-6 | complex | 35, 22, 15, 15 | – | 19.501 | 19.355 | 1.008 | 4,5 | 1.054 | 19.501 |
| kathleen | avx512 | W1 kD = 30, ε = 1e-3 | complex | 12, 8 | 2,3 | 0.120 | 0.120 | 1.000 | 2,3 | 1.000 | 0.194 |
| kathleen | avx512 | W1 kD = 30, ε = 1e-6 | complex | 16, 15 | – | 0.719 | 0.719 | 1.000 | 3 | 1.196 | 0.719 |
| kathleen | avx512 | W3 (N = 108849) kD = 30, ε = 1e-3 | complex | 12, 8, 7 | 2,3,4 | 1.096 | 1.096 | 1.000 | 2,3,4 | 1.000 | 2.469 |
| kathleen | avx512 | W4 cube (N = 100000) kD = 10, ε = 1e-3 | complex | 7, 7, 7 | 2,3,4 | 1.013 | 1.013 | 1.000 | 2,3,4 | 1.000 | 2.328 |
| kathleen | avx512 | W4 cube (N = 100000) kD = 10, ε = 1e-6 | real | 15, 15, 15 | 3,4 | 7.531 | 7.531 | 1.000 | 2,3,4 | 1.001 | 11.820 |
| kathleen | avx512 | W4 cube (N = 100000) kD = 10, ε = 1e-6 | complex | 15, 15, 15 | 4 | 11.473 | 11.473 | 1.000 | 2,3,4 | 1.006 | 12.466 |
| kathleen | avx2 | W1 kD = 100, ε = 1e-3 | real | 30, 18, 11, 7 | 4,5 | 5.784 | 5.784 | 1.000 | 4,5 | 1.000 | 7.201 |
| kathleen | avx2 | W1 kD = 100, ε = 1e-3 | complex | 30, 18, 11, 7 | 5 | 7.218 | 7.106 | 1.016 | 4,5 | 1.000 | 7.600 |
| kathleen | avx2 | W1 kD = 100, ε = 1e-6 | complex | 35, 22, 15, 15 | – | 19.583 | 19.583 | 1.000 | – | 1.000 | 19.583 |
| kathleen | avx2 | W1 kD = 30, ε = 1e-3 | complex | 12, 8 | 3 | 0.150 | 0.150 | 1.000 | 3 | 1.000 | 0.206 |
| kathleen | avx2 | W1 kD = 30, ε = 1e-6 | complex | 16, 15 | – | 0.732 | 0.732 | 1.000 | – | 1.000 | 0.732 |
| kathleen | avx2 | W3 (N = 108849) kD = 30, ε = 1e-3 | complex | 12, 8, 7 | 3,4 | 1.692 | 1.692 | 1.000 | 3,4 | 1.000 | 2.757 |
| kathleen | avx2 | W4 cube (N = 100000) kD = 10, ε = 1e-3 | complex | 7, 7, 7 | 2,3,4 | 1.534 | 1.534 | 1.000 | 2,3,4 | 1.000 | 2.571 |
| kathleen | avx2 | W4 cube (N = 100000) kD = 10, ε = 1e-6 | real | 15, 15, 15 | – | 12.036 | 11.476 | 1.049 | – | 1.049 | 12.036 |
| kathleen | avx2 | W4 cube (N = 100000) kD = 10, ε = 1e-6 | complex | 15, 15, 15 | – | 12.699 | 12.699 | 1.000 | – | 1.000 | 12.699 |

## 4. Many threads: the split, and the default (D8) against the per-level best

CH6.2h's matrix (W1 at kD = 10, 30, 100, W3 at kD = 10, 30, the four W4 at N = 10⁵ and kD = 1,
100) and large cases (W1 at kD = 500, W3 at kD = 100; Kathleen and locust), each at 10⁻³ and
10⁻⁶, under the default (`auto-pw`: D8 on) with `--term on,off --block auto,64`, beside
Rotation and Batched on every level at both widths, ABBA, `--evaluations 2 --levels` (the
second evaluation's per-level times). The previous default is the term off with blocks of
64; the per-level best is each level's fastest of Rotation, Batched (either width) and the
waves where the rule ran them.

| machine | threads | stage | ε | cases | evaluation ÷ previous default | evaluation ÷ per-level best | M2L ÷ per-level best |
| --- | ---: | --- | --- | ---: | --- | --- | --- |
| the M6 | 6 | matrix | 10⁻³ | 13 | 0.54–1.07 | 1.01–1.09 | 1.01–1.15 |
| the M6 | 6 | matrix | 10⁻⁶ | 13 | 0.96–1.03 | 1.00–1.02 | 1.00–1.03 |
| locust | 72 | matrix | 10⁻³ | 13 | 0.36–0.99 | 1.00–1.28 | 1.00–3.32 |
| locust | 72 | matrix | 10⁻⁶ | 13 | 0.78–1.02 | 1.00–1.05 | 1.00–1.15 |
| locust | 72 | large | both | 4 | 1.00–1.01 | 1.00–1.01 | 1.00–1.01 |
| Kathleen | 40 | matrix | 10⁻³ | 13 | 0.57–1.03 | 1.01–1.21 | 1.02–2.21 |
| Kathleen | 40 | matrix | 10⁻⁶ | 13 | 0.73–1.03 | 1.00–1.04 | 1.01–1.23 |
| Kathleen | 40 | large | both | 4 | 1.00–1.05 | 1.00–1.01 | 1.00–1.01 |

- The ratios above 1.00 against the previous default lie within the runs' spread (the
  largest ratio between the two launches of one run on a level, per case: up to 1.53 on the
  M6, 2.36 on locust and 1.69 on Kathleen; the outputs are the same bits).
- The largest gains are the ms-scale cases at 10⁻³ where every level is Batched at p = 7:
  W1 at kD = 10 takes 0.36× the previous default's evaluation on locust, 0.54× on the M6 and
  0.57× on Kathleen.
- What remains above the per-level best (the levels above 1.10× their fastest, with the time
  lost per evaluation; the tables script's summary in the archive):
  - level 2 (64 boxes) at p = 7 under Batched: 2.2–5.0× Rotation's level time at 40 and 72
    threads, 1.1–1.5× at 6; 0.1–1.0 ms;
  - levels of 512–5,000 boxes at p = 7–8 at 40 and 72 threads: 1.1–2.1× Rotation's, up to
    2 ms (locust's cube level 3: Batched 2.0 ms, Rotation 1.6 ms);
  - D8's waves on level 2 (p = 12, 15, 16, 35): 1.1–2.7×, up to 3 ms (CH6.2h §6, not this
    task's);
  - Batched against another Batched run of the same level, 1.10–1.25×: the runs' spread.
- The split by itself, at 72 threads on locust, the W4 cube at kD = 1 and 10⁻³: level 2
  4.3 → 0.8 ms, level 3 7.1 → 2.0 ms, level 4 (4,096 boxes) 12.6 → 11.9 ms, the M2L
  23.9 → 14.7 ms and the evaluation 45.4 → 35.3 ms (blocks of 64 → the split, term on).

## 5. The estimate m̂ against the exact count

`--levels` prints each level's boxes, m̂ and Batched's exact (block, offset) visits at blocks
of 64 on one rank (`Row::block_visits`), so the exact columns per read:

| level | boxes | m̂ | exact |
| --- | ---: | ---: | ---: |
| any level 2 (64 boxes, 3,096 pairs) | 64 | 9.80 | 9.80 |
| the cube, level 3 / 4 | 512 / 4,096 | 21.10 / 28.88 | 21.10 / 28.88 |
| W1 kD = 30, level 3 | 448 | 14.58 | 15.02 |
| W1 kD = 100, level 4 / 5 | 3,200 / 4,864 | 24.39 / 10.20 | 24.51 / 15.29 |
| W4 Plummer kD = 1, levels 5–8 | 904–2,392 | 15.89–27.76 | 17.10–28.27 |
| W4 clusters kD = 100, levels 3–8 | 288–3,152 | 12.52–22.27 | 13.49–23.19 |
| W4 sphere kD = 1, level 4 / 5 | 3,200 / 5,552 | 24.39 / 11.84 | 24.51 / 16.16 |

m̂ assumes each block's boxes meet random offsets; a block of surface boxes in Morton order
shares more of them, so m̂ errs low there (up to 1.5×), towards Rotation. On the measured
surface levels (W1 level 5, the sphere's level 5) Batched at 15–16 columns lost or tied at
p = 15 and won at p = 7, as the rule chooses with m̂ ≈ 10–12. An exact count over the ranks
(blocks aligned in the global index, the boundary blocks merged) is possible but costs
communication and code; not needed for these cases.

## 6. The node budget

`HelmholtzFmmBuilder::dense_node_budget` (default `DEFAULT_DENSE_NODE_BUDGET`, 8 GiB) bounds
the dense, class and batched tables of a node: each rank takes min(`dense_budget`, node
budget / ranks per node), the ranks per node agreed at build as for the waves
(`HelmholtzFmm::dense_budget` reports it). One rank per node keeps 1 GiB. At 40 ranks per
node a rank holds 205 MB (one Batched level of p = 15 is 331 MB, of p = 10 74 MB); at 72,
114 MB. As for Laplace's node rule (Phase 5N T4), the levels a rank batches then depend on
the ranks per node, so the one-rank reference of a multi-rank run names the budget the run
resolved (`check_batched_term`: the default per rank, and a node budget of the finest level's
tables per rank, which batches the finest level alone, at 1, 2 and 4 ranks). No timing.

Signed off (the user, 2026-11-12, on #211, question 1): 8 GiB stays the default. The budget binds only above 8 ranks
per node, and a level it leaves out runs Rotation (no failure). Each rank holds its own copy of
the tables and its threads share it, so the advice is a few ranks per node with threads per
rank (4 × 10 on a Kathleen node keeps 1 GiB per rank; node-m2l.md §7 measured the same for
Laplace's Dense). One copy per node, through an MPI shared window, is CH8.0 on the roadmap:
Laplace has none either (Phase 5N's spike, node-m2l.md §2.6, and its decision 6).

## 7. Accuracy and determinism

- The term moves a level between Batched and Rotation, which agree to rounding (CH3.2c);
  the split changes no bit. The default's errors against the sampled direct sum are the
  previous default's in every launch (the same checksum in all 172 case launches of §4, where
  the term changed no level), and at one thread Rotation's to three digits.
- The shapes are integers summed over the ranks and m* a table agreed over them (the
  largest): `check_batched_term` checks the shapes the same on every rank and equal to the
  one-rank tree's, the agreed table, some level sent to Rotation by the term alone (the
  clusters' finest, 1.2 columns per read), the output with the term and the library's
  default (term and waves on) the one-rank FMM's bit for bit, at 1, 2 and 4 ranks; Batched
  at blocks of 1, 7 and 64 at 1 and 2 threads the default width's bits.

## 8. Follow-ups

- **CH4.0f, Batched's split across the matrix's rows** (§4): each thread a slice of the local's
  coefficients for the whole block, so a level of few blocks keeps its columns per read at any
  thread count; every element keeps its chain, so no bit changes. It needs a column kernel
  for a row slice of the n × n table (`nd_fmm_simd::MatrixKernel::accumulate_columns` takes
  square column-major tables), a change to `nd-fmm-simd`, which Laplace runs (constraint 1:
  bit for bit and D9). Gain: at most 1–2 ms per level at 40–72 threads on these cases. Added to
  the roadmap after CH4.0e, off the priority list (the user, 2026-11-12, on #211, question 2).
- **CH8.0, one copy of the tables per node** (§6; the user, 2026-11-12, on #211): added to the roadmap's H8.
- **m* for the scalar kernel and AVX2 with one plane's crossover**: unmeasured end to end;
  with the first need.

## 9. How it was run

`tools/helmholtz/ch4-0-runs.sh pairs <out> <isa…>` and `fmm <out> <stage…>`; tables by
`tools/helmholtz/ch4-0-tables.py pairs|fmm|rule`. Raw outputs in
`~/projects/fmm-archive/{m6,locust,kathleen}/logs/ch4-0/`.

- **The M6** (the user's go-ahead, 2026-11-12; gated: the other processes below 150 % before,
  a launch voided and repeated above 200 % during, twice at most): pairs at 6932a6b
  10:45–12:10 while the user's jobs ran; three launches whose last attempt was still
  contended (c = 1 ascending, 1,019 %; c = 64 both, 344 % and 1,056 %) repeated quietly at
  12:03–12:10 (others at most 199 %). The FMM stages at 5bed1ae's build (the provisional m*;
  §3 evaluates the final constants on their times) 12:10–12:36, others at most 130 % during
  the kept attempts.
- **locust**: pairs at 6932a6b 10:10–10:41 (another user process at about 1.3 cores of 72 at
  the start); the FMM stages at f2c45da 10:44–11:16 with `GATE=1` (others below 5 %); the
  confirmation at 37e194d and 294c024 (§3).
- **Kathleen**: job 633057 (pairs, AVX-512 and AVX2, node-c11b-017, 63 min), 633065 (`matrix`
  and `large` at 1 × 40, node-c11a-064, 43 min), 633066 (`ch3-2d` at one thread, AVX-512
  and AVX2, node-c11a-113, 39 min), 633071 (the confirmation, `ch3-2d` at 37e194d for both ISAs, node-c11b-017, 39 min).
- The FMM stages ran the provisional m* (2 to p = 8, 4 to 11, 8 above); at the final constants
  no level of `matrix` and `large` chooses differently on any machine, so those runs are the
  final rule's; of `ch3-2d` five levels on AVX-512, one on AVX2 and one on NEON do, which §3's
  evaluation and confirmation cover.
- The run script was edited while the M6's pair sweep ran it; bash read a fragment after the
  sweep's last launch (`entries[@]: unbound variable`, exit 1), after every launch had
  finished. Later runs used a copy.

## Appendix: the pair tables

The tables' last column m* is at a fixed margin of 1.1 (`ch4-0-tables.py pairs`), CH3.2d's
headroom, for comparison; the shipped m* are §1's, at each class's margin.

### m6, neon, 1 plane

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3.31 | 4.14 | 7.90 | 8.37 | 8.40 | 8.45 | 8.74 | 1 |
| 4 | 2.23 | 2.83 | 5.25 | 5.55 | 5.66 | 5.23 | 5.73 | 1 |
| 5 | 1.56 | 1.96 | 4.10 | 4.31 | 4.36 | 4.51 | 4.38 | 1 |
| 6 | 1.20 | 1.43 | 3.13 | 3.35 | 3.44 | 3.38 | 3.45 | 1 |
| 7 | 0.80 | 0.80 | 1.76 | 2.08 | 2.85 | 3.06 | 2.96 | 4 |
| 8 | 0.60 | 0.65 | 1.52 | 1.81 | 2.23 | 2.36 | 2.46 | 4 |
| 9 | 0.27 | 0.40 | 0.86 | 1.23 | 1.58 | 1.86 | 1.95 | 8 |
| 10 | 0.23 | 0.35 | 0.63 | 0.98 | 1.28 | 1.54 | 1.66 | 16 |
| 11 | 0.22 | 0.34 | 0.74 | 1.08 | 1.32 | 1.51 | 1.56 | 16 |
| 12 | 0.21 | 0.30 | 0.66 | 0.97 | 1.15 | 1.32 | 1.38 | 16 |
| 13 | 0.18 | 0.26 | 0.52 | 0.79 | 0.98 | 1.15 | 1.22 | 32 |
| 14 | 0.17 | 0.23 | 0.47 | 0.73 | 0.91 | 1.04 | 1.11 | 64 |
| 15 | 0.13 | 0.22 | 0.44 | 0.61 | 0.82 | 1.00 | 1.06 | – |

### m6, neon, 2 planes

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2.12 | 2.56 | 4.57 | 4.63 | 4.83 | 4.99 | 4.88 | 1 |
| 4 | 1.53 | 1.78 | 3.03 | 3.28 | 3.26 | 3.49 | 3.36 | 1 |
| 5 | 1.09 | 1.29 | 2.50 | 2.59 | 2.62 | 2.70 | 2.59 | 2 |
| 6 | 0.85 | 0.90 | 1.97 | 1.94 | 2.10 | 2.13 | 2.09 | 4 |
| 7 | 0.62 | 0.62 | 1.26 | 1.49 | 1.81 | 1.80 | 1.82 | 4 |
| 8 | 0.46 | 0.49 | 1.14 | 1.34 | 1.44 | 1.46 | 1.46 | 4 |
| 9 | 0.27 | 0.35 | 0.74 | 0.99 | 1.14 | 1.22 | 1.25 | 16 |
| 10 | 0.22 | 0.30 | 0.60 | 0.80 | 0.94 | 1.02 | 1.07 | – |
| 11 | 0.20 | 0.28 | 0.65 | 0.83 | 0.91 | 0.96 | 1.00 | – |
| 12 | 0.18 | 0.24 | 0.59 | 0.73 | 0.85 | 0.87 | 0.90 | – |
| 13 | 0.16 | 0.22 | 0.49 | 0.64 | 0.71 | 0.77 | 0.79 | – |
| 14 | 0.14 | 0.17 | 0.43 | 0.58 | 0.65 | 0.69 | 0.72 | – |
| 15 | 0.13 | 0.18 | 0.39 | 0.53 | 0.62 | 0.67 | 0.70 | – |

### locust, neon, 1 plane

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3.21 | 4.28 | 6.53 | 5.50 | 7.25 | 7.14 | 7.88 | 1 |
| 4 | 1.68 | 2.25 | 3.95 | 4.54 | 6.61 | 6.52 | 5.02 | 1 |
| 5 | 1.37 | 1.76 | 3.19 | 3.60 | 5.14 | 3.89 | 4.11 | 1 |
| 6 | 1.00 | 1.33 | 2.51 | 3.86 | 3.01 | 3.18 | 3.27 | 2 |
| 7 | 0.84 | 1.16 | 2.09 | 1.45 | 2.65 | 2.53 | 2.83 | 2 |
| 8 | 0.56 | 0.73 | 1.43 | 1.79 | 2.06 | 2.25 | 2.09 | 4 |
| 9 | 0.37 | 0.64 | 1.21 | 1.55 | 1.66 | 1.88 | 1.95 | 4 |
| 10 | 0.32 | 0.47 | 0.85 | 1.15 | 1.31 | 1.51 | 1.38 | 8 |
| 11 | 0.18 | 0.28 | 0.41 | 0.69 | 0.92 | 1.24 | 1.43 | 32 |
| 12 | 0.13 | 0.20 | 0.34 | 0.45 | 0.74 | 1.01 | 1.56 | 64 |
| 13 | 0.12 | 0.13 | 0.30 | 0.46 | 0.56 | 0.81 | 0.91 | – |
| 14 | 0.10 | 0.10 | 0.35 | 0.27 | 0.51 | 0.63 | 0.82 | – |
| 15 | 0.08 | 0.13 | 0.23 | 0.35 | 0.54 | 0.69 | 0.99 | – |

### locust, neon, 2 planes

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2.20 | 2.66 | 3.98 | 3.67 | 4.51 | 5.44 | 4.25 | 1 |
| 4 | 1.34 | 1.63 | 2.68 | 3.02 | 3.34 | 3.08 | 3.31 | 1 |
| 5 | 1.09 | 1.31 | 2.03 | 2.30 | 3.26 | 2.82 | 2.47 | 2 |
| 6 | 0.77 | 0.97 | 1.70 | 1.85 | 1.99 | 2.99 | 1.88 | 4 |
| 7 | 0.72 | 0.89 | 1.48 | 2.03 | 1.95 | 1.77 | 1.76 | 4 |
| 8 | 0.47 | 0.58 | 1.12 | 1.35 | 1.41 | 1.45 | 1.46 | 4 |
| 9 | 0.37 | 0.50 | 0.92 | 1.09 | 1.13 | 1.27 | 1.38 | 16 |
| 10 | 0.26 | 0.34 | 0.68 | 0.86 | 0.94 | 1.05 | 1.35 | 64 |
| 11 | 0.16 | 0.26 | 0.39 | 0.58 | 0.66 | 0.91 | 1.00 | – |
| 12 | 0.16 | 0.19 | 0.38 | 0.64 | 0.53 | 0.75 | 0.98 | – |
| 13 | 0.07 | 0.13 | 0.19 | 0.30 | 0.67 | 0.65 | 0.69 | – |
| 14 | 0.06 | 0.09 | 0.17 | 0.21 | 0.39 | 0.71 | 0.57 | – |
| 15 | 0.07 | 0.11 | 0.17 | 0.29 | 0.34 | 0.55 | 0.55 | – |

### kathleen, avx2, 1 plane

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 3.12 | 4.45 | 8.43 | 10.97 | 12.47 | 13.47 | 14.46 | 1 |
| 4 | 2.15 | 3.06 | 6.23 | 8.43 | 9.90 | 11.19 | 11.88 | 1 |
| 5 | 1.77 | 2.58 | 4.83 | 6.75 | 8.18 | 9.23 | 10.01 | 1 |
| 6 | 1.38 | 2.03 | 4.02 | 5.53 | 6.67 | 7.38 | 8.05 | 1 |
| 7 | 1.03 | 1.57 | 3.03 | 4.02 | 4.63 | 5.10 | 5.44 | 2 |
| 8 | 0.73 | 1.17 | 2.27 | 3.21 | 3.89 | 4.28 | 4.94 | 2 |
| 9 | 0.51 | 0.81 | 1.31 | 2.06 | 2.87 | 3.55 | 4.06 | 4 |
| 10 | 0.43 | 0.69 | 1.15 | 1.83 | 2.53 | 3.16 | 3.66 | 4 |
| 11 | 0.22 | 0.44 | 0.51 | 0.99 | 1.56 | 2.22 | 2.81 | 16 |
| 12 | 0.19 | 0.36 | 0.45 | 0.83 | 1.35 | 1.97 | 2.54 | 16 |
| 13 | 0.18 | 0.33 | 0.47 | 0.81 | 1.28 | 1.79 | 2.29 | 16 |
| 14 | 0.15 | 0.25 | 0.38 | 0.66 | 1.06 | 1.53 | 1.92 | 32 |
| 15 | 0.15 | 0.25 | 0.38 | 0.65 | 1.02 | 1.44 | 1.81 | 32 |

### kathleen, avx2, 2 planes

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2.24 | 2.99 | 5.57 | 6.84 | 7.14 | 7.57 | 8.17 | 1 |
| 4 | 1.74 | 2.24 | 4.57 | 5.61 | 6.19 | 6.60 | 6.82 | 1 |
| 5 | 1.50 | 1.95 | 3.76 | 4.70 | 5.26 | 5.66 | 5.83 | 1 |
| 6 | 1.18 | 1.55 | 3.13 | 3.85 | 4.24 | 4.62 | 4.79 | 1 |
| 7 | 0.91 | 1.23 | 2.41 | 2.77 | 3.06 | 3.26 | 3.30 | 2 |
| 8 | 0.69 | 0.95 | 1.90 | 2.39 | 2.68 | 2.88 | 3.06 | 4 |
| 9 | 0.50 | 0.72 | 1.24 | 1.72 | 2.13 | 2.43 | 2.63 | 4 |
| 10 | 0.42 | 0.61 | 1.10 | 1.54 | 1.91 | 2.18 | 2.36 | 8 |
| 11 | 0.26 | 0.42 | 0.58 | 0.95 | 1.34 | 1.68 | 1.94 | 16 |
| 12 | 0.21 | 0.35 | 0.50 | 0.82 | 1.18 | 1.53 | 1.79 | 16 |
| 13 | 0.17 | 0.30 | 0.40 | 0.76 | 1.09 | 1.37 | 1.56 | 32 |
| 14 | 0.15 | 0.25 | 0.38 | 0.62 | 0.90 | 1.13 | 1.30 | 32 |
| 15 | 0.15 | 0.23 | 0.35 | 0.56 | 0.79 | 1.01 | 1.20 | 64 |

### kathleen, avx512, 1 plane

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2.85 | 4.53 | 9.50 | 12.61 | 14.84 | 16.05 | 16.91 | 1 |
| 4 | 2.01 | 2.95 | 6.90 | 9.57 | 11.57 | 12.76 | 13.64 | 1 |
| 5 | 1.51 | 2.30 | 5.42 | 7.72 | 9.46 | 10.51 | 11.25 | 1 |
| 6 | 1.35 | 2.19 | 4.83 | 7.18 | 9.28 | 10.72 | 11.74 | 1 |
| 7 | 1.13 | 1.75 | 3.72 | 6.28 | 7.60 | 9.06 | 9.68 | 1 |
| 8 | 0.82 | 1.25 | 2.53 | 4.00 | 5.38 | 6.67 | 7.52 | 2 |
| 9 | 0.57 | 0.89 | 1.75 | 2.84 | 3.96 | 4.95 | 5.67 | 4 |
| 10 | 0.46 | 0.76 | 1.45 | 2.40 | 3.54 | 4.59 | 5.42 | 4 |
| 11 | 0.39 | 0.65 | 1.20 | 2.05 | 3.08 | 4.15 | 4.99 | 4 |
| 12 | 0.34 | 0.58 | 1.01 | 1.74 | 2.69 | 3.69 | 4.58 | 8 |
| 13 | 0.30 | 0.50 | 0.93 | 1.59 | 2.38 | 3.17 | 3.87 | 8 |
| 14 | 0.20 | 0.35 | 0.60 | 1.08 | 1.76 | 2.62 | 3.41 | 16 |
| 15 | 0.17 | 0.31 | 0.52 | 0.90 | 1.48 | 2.22 | 3.06 | 16 |

### kathleen, avx512, 2 planes

Rotation's time per pair over fused Batched's at c pairs per offset (the mean of 2 launches); m* the smallest c at which it is at least 1.1.

| p | c = 1 | c = 2 | c = 4 | c = 8 | c = 16 | c = 32 | c = 64 | m* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 2.08 | 2.99 | 6.37 | 7.59 | 8.29 | 8.81 | 8.98 | 1 |
| 4 | 1.62 | 2.15 | 5.12 | 6.30 | 7.11 | 7.53 | 7.78 | 1 |
| 5 | 1.31 | 1.77 | 4.25 | 5.35 | 6.03 | 6.40 | 6.48 | 1 |
| 6 | 1.23 | 1.79 | 4.01 | 5.30 | 6.09 | 6.73 | 7.07 | 1 |
| 7 | 1.05 | 1.46 | 3.24 | 4.54 | 5.48 | 5.77 | 5.94 | 2 |
| 8 | 0.76 | 1.04 | 2.41 | 3.17 | 3.88 | 4.40 | 4.71 | 4 |
| 9 | 0.54 | 0.75 | 1.68 | 2.37 | 2.94 | 3.36 | 3.64 | 4 |
| 10 | 0.46 | 0.68 | 1.44 | 2.13 | 2.77 | 3.27 | 3.60 | 4 |
| 11 | 0.40 | 0.58 | 1.23 | 1.86 | 2.52 | 3.01 | 3.36 | 4 |
| 12 | 0.35 | 0.53 | 1.04 | 1.63 | 2.25 | 2.74 | 3.11 | 8 |
| 13 | 0.27 | 0.43 | 0.79 | 1.40 | 1.92 | 2.31 | 2.54 | 8 |
| 14 | 0.20 | 0.34 | 0.62 | 1.05 | 1.59 | 2.02 | 2.36 | 16 |
| 15 | 0.18 | 0.31 | 0.51 | 0.82 | 1.24 | 1.69 | 2.08 | 16 |
