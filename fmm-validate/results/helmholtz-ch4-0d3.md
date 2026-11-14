# Helmholtz CH4.0d3: the leaf size, c_leaf, and W and X pairs as near pairs

Task: `docs/helmholtz/tasks/CH4.0d3.md`. Roadmap: H4, CH4.0d3 (priority list item 4). The W and
X rule and c_leaf 2 in scope by the user's decisions of 2026-11-14 ("Add W/X rule here
(Recommended)", "Yes, sweep 1 and 2 (Recommended)"; the task's open questions 1 and 2).
Proposal §2 (constraints 1, 2, 9), §3.7 (the leaf rule, D11), §5.7, §9.1. Builds on
`fmm-validate/results/helmholtz-ch4-0d2.md` (the V rule) and CH3.6's matrix
(`fmm-validate/results/helmholtz-ch3-6.md`).

## 0. Summary

- **Round 1 (the leaf size alone)** found 256 points per leaf faster than 64 on build plus one
  evaluation (Kathleen 0.72, 0.67, 0.61× at 10⁻³, 10⁻⁶, 10⁻⁸; locust 0.80, 0.84, 0.78×), but
  slower evaluations at 10⁻³ and on trees with leaves on two levels: there P2L and M2P, the W
  and X pairs, took 40–67 % of the evaluation (W4's Plummer sphere at kD = 10), and W3 at
  kD = 100 with 128 points evaluated 3× slower than with 64. The Helmholtz builder converted no
  W or X pair; Laplace's near rule (Phase 6 T12) does. (Round 1's launches are in the archive,
  `kat-r1a`, `kat-r1b`, `locust-r1`; round 2's rules-off rows, §3.1 and §3.2, repeat them
  within 2 %.)
- **The W and X rule (new; §2).** `HelmholtzFmmBuilder::near_wx(WxNear)` makes a W pair a near
  pair where the W leaf's sources are below N_W = α_M2P(p)/α_P2P, an X pair where the X leaf's
  targets are below N_X = α_P2L(p)/α_P2P, α = a + b (p + 1)² per point at the finer leaf's
  degree, fitted per machine class (`WX_COSTS_X86` from Kathleen, `WX_COSTS_NEON` from locust),
  agreed over the ranks; through `nd_fmm_plan`'s existing `NearRule::new` predicate in the one
  plan build (no plan change). Converted pairs are exact; the output stays deterministic and
  on P ranks the one-rank FMM's bit for bit.
- **Round 2 (§3)**, the leaf sizes 32–1024 × the V rule off and by its cost model × the W and X
  rule off and by its cost model × c_leaf 1 and 2, on CH3.6's matrix (14 cases), the large
  cases and real charges. Build plus one evaluation over the library's default before this task
  (64 points, no rule, c_leaf 1), the geometric mean over the matrix's cases; the evaluation
  alone in parentheses; both rules on:

  | Leaf size, c_leaf 1 | Kathleen 10⁻³ | 10⁻⁶ | 10⁻⁸ | locust 10⁻³ | 10⁻⁶ | 10⁻⁸ |
  | ---: | --- | --- | --- | --- | --- | --- |
  | 64 | 0.90 (0.68) | 0.84 (0.56) | 0.60 (0.36) | 1.00 (0.79) | 0.89 (0.58) | 0.79 (0.54) |
  | 128 | 0.79 (0.65) | 0.72 (0.51) | 0.52 (0.31) | **0.84 (0.89)** | **0.75 (0.56)** | 0.70 (0.47) |
  | 256 | **0.64 (0.71)** | **0.54 (0.41)** | **0.42 (0.27)** | 0.80 (1.10) | 0.73 (0.66) | **0.58 (0.44)** |
  | 512 | 0.65 (0.90) | 0.53 (0.44) | 0.40 (0.26) | 0.90 (1.62) | 0.85 (0.99) | 0.63 (0.56) |

  No run's φ or ∇φ error exceeded ε (§3.4; at most 0.16 ε and 0.45 ε).
- **Proposed defaults (built in this PR, each a sign-off question, §6):**
  1. the points per leaf `helmholtz::default_max_points_per_leaf(tolerance, isa)`: 256 on x86;
     on NEON 128 at ε ≥ 10⁻⁷ (or named degrees) and 256 below. 256 on NEON at 10⁻³ and 10⁻⁶
     would make the evaluation alone slower than 128's (1.10 and 0.66× against 0.89 and 0.56×)
     for 4 % and 3 % on build plus one evaluation. The multipole planes would change one
     choice, NEON at 10⁻⁶ (§3.5: 256 5.7 % faster on build plus one evaluation over two cases,
     the evaluation 1 % slower), so the rule does not read them;
  2. `WxNear::CostModel` (the largest single gain: alone, at 64 points, the evaluation
     0.61–0.71× on Kathleen and 0.58–0.79× on locust);
  3. `VNear::CostModel` (with the W and X rule: Kathleen 0.64 against 0.68 at 10⁻³, 256 points;
     locust 0.2–2.6 % slower on build plus one evaluation at its sizes, the evaluation alone
     within 1.3 %). It widens the P2P's phase range to Re(k) r ≤ 44 c_leaf (§4);
  4. c_leaf 2 (`DEFAULT_C_LEAF`, amending D11's builder default): where the leaf rule binds,
     W1 at kD = 500, build plus one evaluation 0.62, 0.71, 0.78× c_leaf 1's on Kathleen (256
     points) and 0.84, 0.84, 0.70× on locust (128, 128, 256), the errors within ε (§4); on the
     matrix, where it rarely binds, 0.95–1.01×. Against D11's reason: at c_leaf 2 the census
     found leaves on high-frequency levels (9 trees at 10⁻⁶); no high-frequency M2L runs yet
     (CH4.2, CH4.3).
- **Round 3 (§5)**: the defaults as built against the default before, build plus one evaluation
  (evaluation alone): Kathleen 0.64 (0.72), 0.53 (0.40), 0.41 (0.26) at 10⁻³, 10⁻⁶, 10⁻⁸; locust
  0.83 (0.87), 0.75 (0.56), 0.57 (0.43); no case of Kathleen's loses on build plus one evaluation
  (its evaluation alone at 10⁻³ on three cases, up to 1.43×), locust's cube and W3 at kD = 30
  lose 4–9 % at 10⁻³ and 10⁻⁶ (small cases, 0.08–0.34 s). 4 × 10 equals 1 × 40 bit for bit.

## 1. Method

`helmholtz_accuracy` (`--leaf-size`, `--c-leaf`, `--near-v`, `--near-wx`, `--levels`), through
`tools/helmholtz/ch4-0d3-runs.sh`; tables by `tools/helmholtz/ch4-0d3-tables.py`. Release, BLAS
threads 1, one rank × all cores (Kathleen 1 × 40, `singlenode`, exclusive nodes; locust 1 × 72,
checked idle before each launch), the library's M2L (`--m2l auto-pw`: Auto with the exponential
M2L, D8), the empty-box rule on, two evaluations per run (the second timed), the matrix in ABBA
order of the leaf sizes and both orders averaged.

- The matrix: CH3.6's (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; W4's cube, sphere, Plummer
  sphere and clusters at N = 10⁵ and kD = 1, 10, 100) at ε = 10⁻³, 10⁻⁶, 10⁻⁸; the summaries
  take the cases whose reference runs at least 0.3 s (14 at 10⁻³ and 10⁻⁶, 15 at 10⁻⁸).
- The large cases: W1 at kD = 500 (the leaf rule fixes every leaf at level 7, about 27 points)
  and W3 at kD = 100 (N = 4·10⁶; Kathleen only).
- Real charges (`planes`, one multipole plane): W1 at kD = 30, 100 and W4's cube at kD = 10, 100.
- The reference of every ratio: 64 points per leaf, the V and W/X rules off, c_leaf 1.
- Every run's φ and ∇φ against the direct sum on the samples (§9.1's ℓ2 measure).

Raw outputs: `fmm-archive/{kathleen,locust}/logs/ch4-0d3/` (round 1: Kathleen jobs 633910,
633911, locust at 23e993c; calibration: Kathleen 634049, locust at 5d82ea4; round 2: Kathleen
634080, 634081 and locust at df2b2f7; round 3: see §5).

## 2. The W and X rule's constants

`calibrate` (`--near-wx off --levels`, the table "Leaf operators"): thread-nanoseconds per P2L
and M2P point interaction and per P2P interaction per level, W4's Plummer sphere, clusters and
sphere at N = 10⁵ and kD = 1, 10, 100 and W1 at kD = 100, 10⁻³, 10⁻⁶, 10⁻⁸, 64 and 256 points;
the medians over the rows of at least 10⁵ interactions of the kind, and least-squares fits of
a + b (p + 1)² (`ch4-0d3-tables.py wx`):

| p | Kathleen P2P ns | P2L ns | M2P ns | locust P2P ns | P2L ns | M2P ns |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 6.8 | 801 | 2089 | 16.1 | 509 | 1267 |
| 11 | 9.7 | – | 2102 | 19.1 | – | 1672 |
| 15 | 6.7 | 2405 | 6540 | 16.4 | 1461 | 4607 |
| 32 | 6.9 | 8847 | 27448 | 17.1 | 6000 | 21763 |

| | P2L a + b (p + 1)² | M2P a + b (p + 1)² | P2P (V rule's `VCosts`) |
| --- | --- | --- | --- |
| x86 (`WX_COSTS_X86`) | 339 + 7.85 (p + 1)² ns | 274 + 26.4 (p + 1)² ns | 6.5 ns |
| NEON (`WX_COSTS_NEON`) | 129 + 5.67 (p + 1)² ns | 20.2 (p + 1)² ns (a fitted −87, set 0) | 14.0 ns |

So at p = 15 a W pair converts below about 1100 sources of its leaf on x86 and 370 on NEON, an
X pair below about 360 and 110 targets (`tests/helmholtz/v_crossover.rs` checks these). The
P2P time per interaction is the V rule's (`V_COSTS_*`, CH4.0d2), so both rules weigh the same
P2P; this calibration measured 6.9 and 16.6 ns, so the crossovers lie 6 % (x86) and 18 % (NEON)
above what its own P2P gives. The rows at p = 11 are few (4–6, W1 at kD = 100) and the fits
overestimate M2P there about 1.9× (Kathleen 2102 ns measured, 4076 fitted; locust 1672 and
2909), which converts more W pairs on such levels than the measured cost would. Rounds 2 and 3
measured the rule with these constants as built, so the gains above include both offsets; a
crossover only moves which pairs convert, never the errors' bound or the bits' guarantees.

## 3. Round 2: the leaf size and the rules

### 3.1 Kathleen (1 × 40), c_leaf 1

Build plus one evaluation over the reference (evaluation alone), the geometric mean over the
matrix's cases (§1):

| V rule | W, X rule | leaf size | 10⁻³ | 10⁻⁶ | 10⁻⁸ |
| --- | --- | ---: | --- | --- | --- |
| off | off | 64 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| off | off | 128 | 0.899 (0.996) | 0.888 (0.897) | 0.854 (0.786) |
| off | off | 256 | 0.724 (0.961) | 0.664 (0.649) | 0.612 (0.513) |
| off | off | 512 | 0.747 (1.151) | 0.697 (0.733) | 0.698 (0.678) |
| off | cost | 64 | 0.943 (0.707) | 0.843 (0.701) | 0.747 (0.612) |
| off | cost | 128 | 0.837 (0.695) | 0.710 (0.562) | 0.599 (0.442) |
| off | cost | 256 | 0.681 (0.748) | 0.535 (0.425) | 0.438 (0.299) |
| off | cost | 512 | 0.698 (0.930) | 0.528 (0.444) | 0.420 (0.283) |
| cost | off | 64 | 0.967 (0.974) | 1.001 (0.864) | 0.838 (0.691) |
| cost | off | 128 | 0.859 (0.969) | 0.902 (0.838) | 0.751 (0.593) |
| cost | off | 256 | 0.680 (0.936) | 0.670 (0.637) | 0.594 (0.475) |
| cost | off | 512 | 0.699 (1.113) | 0.698 (0.719) | 0.682 (0.635) |
| cost | cost | 64 | 0.904 (0.676) | 0.838 (0.560) | 0.601 (0.357) |
| cost | cost | 128 | 0.791 (0.653) | 0.721 (0.509) | 0.516 (0.308) |
| cost | cost | **256** | **0.636 (0.713)** | **0.539 (0.409)** | **0.419 (0.265)** |
| cost | cost | 512 | 0.652 (0.895) | 0.532 (0.435) | 0.404 (0.257) |

Both rules on, 32 points: 1.14, 1.17, 0.90 (0.76, 0.73, 0.63); 1024 points: 0.64, 0.55, 0.35
(1.12, 0.51, 0.27). At 10⁻⁸ 1024 is 0.83× 256's build plus one evaluation (0.93× without W3 at
N = 4032, which 1024 points make a two-level tree, 0.06 against 0.39×; the Plummer sphere 1.16×
at worst), with the evaluation alone equal (0.27 and 0.27×); 512 is 0.96×. The full table in
the archive (`r2-summary.md`).

### 3.2 locust (1 × 72), c_leaf 1

| V rule | W, X rule | leaf size | 10⁻³ | 10⁻⁶ | 10⁻⁸ |
| --- | --- | ---: | --- | --- | --- |
| off | off | 64 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| off | off | 128 | 0.843 (1.027) | 0.871 (0.931) | 0.938 (0.955) |
| off | off | 256 | 0.803 (1.182) | 0.832 (0.944) | 0.770 (0.786) |
| off | off | 512 | 0.892 (1.661) | 0.936 (1.204) | 0.874 (0.991) |
| off | cost | 64 | 0.965 (0.793) | 0.848 (0.583) | 0.782 (0.575) |
| off | cost | 128 | 0.817 (0.887) | 0.731 (0.561) | 0.691 (0.478) |
| off | cost | 256 | 0.788 (1.104) | 0.714 (0.673) | 0.583 (0.446) |
| off | cost | 512 | 0.887 (1.629) | 0.839 (0.997) | 0.626 (0.561) |
| cost | off | 64 | 1.039 (1.002) | 1.043 (0.998) | 1.012 (0.972) |
| cost | off | 128 | 0.864 (1.027) | 0.891 (0.924) | 0.956 (0.955) |
| cost | off | 256 | 0.818 (1.171) | 0.846 (0.943) | 0.775 (0.785) |
| cost | off | 512 | 0.905 (1.659) | 0.945 (1.196) | 0.878 (0.988) |
| cost | cost | 64 | 0.997 (0.787) | 0.887 (0.579) | 0.790 (0.538) |
| cost | cost | **128** | **0.835 (0.886)** | **0.750 (0.558)** | 0.703 (0.472) |
| cost | cost | **256** | 0.801 (1.104) | 0.727 (0.664) | **0.584 (0.440)** |
| cost | cost | 512 | 0.898 (1.624) | 0.845 (0.988) | 0.627 (0.556) |

### 3.3 What it shows

- The W and X rule is the largest gain on both machines: alone, at 64 points, the evaluation
  0.61–0.71× on Kathleen and 0.58–0.79× on locust, and with it the leaf size's optimum is
  flatter (no size above 1.0× on build plus one evaluation from 128 on).
- The leaf size: on Kathleen 256 and 512 are within 4 % of each other at every ε, and 256 has
  the faster evaluation at 10⁻³ (0.71 against 0.90×); 256 is the x86 default. On locust 256 is
  best on build plus one evaluation at every ε, but at 10⁻³ and 10⁻⁶ its evaluation alone is
  slower than 128's (1.10 and 0.66 against 0.89 and 0.56×; NEON's P2P costs 2.4× x86's per
  interaction, §2), so NEON takes 128 there and 256 below 10⁻⁷, where 256 is best on both
  measures. The boundary 10⁻⁷ lies between the measured 10⁻⁶ and 10⁻⁸.
- FMM3D's `hfmm3d` allows 40–700 points per leaf by default and 300–1000 with `fastker`; the
  measured optimum here, 128–512 with the near rules, lies at the low end of `fastker`'s range:
  the rules convert the pairs that make small leaves costly, and at 10⁻³ 1024 points evaluate
  1.12× (Kathleen) and 2.8× (locust) slower than 64 (geometric means, both rules on).
- On x86 at 10⁻⁸ 1024 points pass T11's 5 % criterion against 256 on build plus one
  evaluation (§3.1: 0.83×, 0.93× without the smallest case, the Plummer sphere 1.16×), not on
  the evaluation alone (equal). The proposal keeps 256 (one size per class from 10⁻⁷, the
  evaluation unchanged); 1024 below 10⁻⁷ on x86 is sign-off question 1's alternative.
- The V rule: with the W and X rule, on Kathleen 0.636 against 0.681 at 10⁻³ (256 points),
  0.539 against 0.535 at 10⁻⁶ and 0.419 against 0.438 at 10⁻⁸; on locust at NEON's sizes 2.2 %
  and 2.6 % slower on build plus one evaluation at 10⁻³ and 10⁻⁶ (128 points: 0.835 against
  0.817, 0.750 against 0.731) and 0.2 % at 10⁻⁸ (256), the evaluation alone within 1.3 % (up to
  1.3 % faster). One default for both classes, the cost model's, within T11's 5 % line on
  NEON; or `VNear::Off` on NEON (sign-off question 3).

### 3.4 The errors

No run of rounds 1 and 2 had a φ or a ∇φ error above ε (every launch of both machines, the
matrix, the large cases, `cleaf` and real charges): at most 0.16 ε for φ and 0.45 ε for ∇φ,
both on W4's Plummer sphere at kD = 100 and 10⁻⁶ with 32 points. The converted pairs are
exact, so the errors move only with the tree.

### 3.5 Real charges (one multipole plane)

Both rules on against both off, c_leaf 1, the geometric means over the real-charge cases:

| Leaf size | Kathleen 10⁻³ | 10⁻⁶ | 10⁻⁸ | locust 10⁻³ | 10⁻⁶ | 10⁻⁸ |
| ---: | --- | --- | --- | --- | --- | --- |
| 64, rules off | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) |
| 256, rules off | 0.965 (0.996) | 0.674 (0.553) | 0.604 (0.500) | 0.852 (0.979) | 0.836 (0.753) | 0.734 (0.681) |
| 128, rules on | 0.813 (0.920) | 0.440 (0.465) | 0.655 (0.393) | 0.866 (1.000) | 0.908 (0.745) | 0.912 (0.722) |
| 256, rules on | 0.717 (0.997) | 0.425 (0.436) | 0.534 (0.361) | 0.866 (0.983) | 0.859 (0.753) | 0.749 (0.683) |

On Kathleen the proposed 256 with the rules is best or within 1 % at every ε. On locust the
rules lose 1.6–2.8 % at 256 points with one plane (the W and X pairs' P2L and M2P cost half as
much, so fewer pairs pay; 0.866, 0.859, 0.749 against 0.852, 0.836, 0.734), and where
the rule takes 128 (ε ≥ 10⁻⁷) 256 is equal at 10⁻³ and 5.7 % faster at 10⁻⁶ on build plus one
evaluation (0.859 against 0.908) with the evaluation 1 % slower (0.753 against 0.745), over two
cases. The rule does not read the planes (one measured point above the 5 % line, on few
cases); NEON with one plane at 256 from 10⁻⁶ is sign-off question 1's other alternative. The
cases per ε: Kathleen 4, 3, 3; locust 2, 2, 3.

## 4. c_leaf 2

Where the leaf rule binds (W1 at kD = 500: every leaf at level 7, about 27 points, whatever
the leaf size), c_leaf 2 lets the leaves be one level coarser. It pays from 128 points (0.84–0.89×
c_leaf 1's at 128) and most from 256 (0.62–0.78× on Kathleen, 0.70× on locust at 10⁻⁸); at 64
points most leaves stay split by points (0.90–0.94×):

| W1, kD = 500, both rules | Kathleen c_leaf 1, 256 | c_leaf 2, 128 | c_leaf 2, 256 | locust c_leaf 1, 128 | c_leaf 2, 128 | c_leaf 2, 256 |
| --- | --- | --- | --- | --- | --- | --- |
| 10⁻³ | 1.010 (1.028) | 0.867 (0.966) | 0.629 (0.989) | 1.010 (0.994) | 0.846 (1.064) | 0.719 (1.176) |
| 10⁻⁶ | 0.971 (0.884) | 0.823 (0.829) | 0.692 (0.783) | 1.028 (1.005) | 0.865 (0.996) | 0.708 (0.967) |
| 10⁻⁸ | 0.823 (0.663) | 0.729 (0.638) | 0.642 (0.584) | 0.990 (0.938) | 0.872 (0.885) | 0.711 (0.725) |

W3 at kD = 100 (Kathleen; the leaf rule binds at level 4 with 64 points): 256 points or more
0.25–0.36×, with c_leaf 1 or 2 alike; 128 with both rules off 0.80–2.24× (an evaluation 3×
64's), with them 0.33–0.48×: the rules remove round 1's defect of trees with leaves on two
levels. On the matrix c_leaf 2 changes the geometric means by −5 to +1 % (Kathleen, 256:
0.641, 0.514, 0.404 against 0.636, 0.539, 0.419; locust, 128: 0.827, 0.742, 0.699 against
0.835, 0.750, 0.703).

Errors: φ ÷ ε at c_leaf 2 at most that of c_leaf 1 on every large case (0.01–0.16).

The phase: the bound Re(k) r ÷ c_leaf (`Row::phase_bound`) is 13.5 without converted V pairs
and 27.1 with them on W1 at kD = 500, against the proposal's 22 (2√3 leaf widths, §5.7) and 44
(4√3, a converted V pair, CH4.0d2). The task's acceptance quotes CH3.4b's 22 c_leaf
"unchanged": with `VNear::CostModel` the default the P2P's range is 44 c_leaf, 88 at c_leaf 2.
The kernel's contract domain is Re(k) r ≤ 2¹⁰ (`P2pKernel::evaluate_helmholtz`), whose per-term
bound (32 + 12 |k| r) u is about 1100 u at 88 (1.2·10⁻¹³, u = 2⁻⁵³): far below every tolerance measured,
and the errors above hold it. Sign-off questions 3 and 4 carry it.

High-frequency leaves: D11 chose c_leaf 1 as the largest with no leaf on a high-frequency
level over the census's 43 trees (`spikes/census/SPIKE_REPORT.md` §3, §4; its first-pass
classifier); at c_leaf 2 it found 9 trees at 10⁻⁶ with an H leaf level (W1 and W2 at
kD = 50–200 and the Plummer sphere at kD = 100, 2 λ leaves) and none at 10⁻⁸. Proposal §3.7
assumes every leaf, and so every M2P and P2L, on a low-frequency level (P2M and L2P arguments
k a_leaf ≤ 5.4 c_leaf). Today no level runs a high-frequency M2L (the diagonal form is CH4.2,
the per-level rule CH4.3, its damped form CH5.1), the leaf operators do not depend on the regime, and the degree rule
covers the leaf arguments (x = k a ≤ 10.9 at c_leaf 2, calibrated to x = 65): the measurements
here stand for the FMM as built. When CH4.3's rule lands, a leaf level it calls H would put W
and X lists on it, which §3.7 rules out; so c_leaf 2 is proposed with a roadmap line in CH4.3
to re-decide c_leaf there (sign-off question 4).

## 5. Round 3: the defaults as built

Kathleen jobs 634114 (matrix, traffic) and 634115 (large), locust at 6542ec9 (the defaults as
built; the later commits change docs and tests only): `--leaf-size auto,64 --c-leaf auto,1
--near-v off,cost --near-wx off,cost`, the matrix in ABBA order. The defaults (`auto`: the
rule's size, c_leaf 2, both rules by the cost model) against the default before (64, c_leaf 1,
the rules off), build plus one evaluation (evaluation alone), geometric means over the cases
whose reference runs at least 0.3 s, with their range:

| | Kathleen (256 points) | locust (128; 256 at 10⁻⁸) |
| --- | --- | --- |
| 10⁻³ | 0.638 (0.718), 0.42–0.77 (0.47–1.43) | 0.825 (0.873), 0.77–1.04 (0.78–1.04) |
| 10⁻⁶ | 0.525 (0.403), 0.45–0.64 (0.28–0.57) | 0.745 (0.555), 0.65–1.04 (0.44–0.98) |
| 10⁻⁸ | 0.414 (0.260), 0.34–0.54 (0.11–0.38) | 0.571 (0.432), 0.47–0.80 (0.26–0.87) |

Round 2's named settings gave the same within 2 % (Kathleen 0.641, 0.514, 0.404; locust 0.827,
0.742, 0.563). The large cases: W1 at kD = 500 0.62, 0.69, 0.63 on Kathleen (evaluation 1.00,
0.80, 0.59) and 0.83, 0.85, 0.71 on locust (1.07, 1.00, 0.72); W3 at kD = 100 (Kathleen) 0.36,
0.31, 0.26 (0.87, 0.40, 0.27).

The losers, every case of the matrix and the large ones (all, also those under 0.3 s):
- Kathleen: no case loses on build plus one evaluation (the worst 0.77×, W4's Plummer sphere at
  kD = 100, 10⁻³). The evaluation alone loses at 10⁻³ only: W3 at kD = 30 1.43×, W4's cube at
  kD = 1 and 10 1.20 and 1.18×, W1 at kD = 500 1.00×; from 10⁻⁶ on every case wins (at most
  0.80×).
- locust: on build plus one evaluation W4's cube at 10⁻³ and 10⁻⁶ (1.04–1.09×, every kD), W3 at
  kD = 30 (1.04, 1.06×) and W1 at kD = 30 at 10⁻⁶ (1.03×), small cases (0.08–0.34 s at the
  reference; only the cube at kD = 100 is in the means); the evaluation alone at 10⁻³ on W1 at
  kD = 10 (1.50×, 0.04 s), 30 and 100 (1.10, 1.04×), W1 at kD = 500 (1.07×), the sphere
  (1.03–1.06×) and the cube and W3 (1.01–1.03×), and at 10⁻⁶ the cube and W3 (1.02, 1.01×). From 10⁻⁸ every case wins on both measures (at most 0.80 and 0.87×).

The errors: every run within ε (φ ÷ ε at most 0.16, the defaults' worst W1 at kD = 500 at
10⁻⁸).

Rank-count invariance at the defaults (`traffic`, Kathleen, 4 ranks × 10 threads against one
rank × 40, W1 at kD = 500 and W4's Plummer sphere at kD = 10, 10⁻⁶, at the rule's 256 and at
64): every checksum of the 4-rank run equals the one-rank run's (`ch4-0d3-tables.py
checksums`: `892a1ee0805286d4`, `9e7a87efc1bbf055`, `6f542b0a00c3283a`, `3b5754d3bf64dbf4`).

## 6. Questions for sign-off

1. The points per leaf: `default_max_points_per_leaf` (256 on x86; NEON 128 at ε ≥ 10⁻⁷ and 256
   below), from build plus one evaluation with the evaluation alone as the tie-break on NEON.
   Alternatives: (a) by the evaluation alone (as Laplace's T11): 128 at 10⁻³ on both classes and
   NEON at 10⁻⁶, 256 elsewhere (x86 at 10⁻³: 0.79 against 0.64× on build plus one evaluation,
   0.65 against 0.71× on the evaluation); (b) x86 1024 below 10⁻⁷ (§3.3: build plus one
   evaluation 0.83× 256's, the evaluation equal, the Plummer sphere 1.16×); (c) NEON with one
   multipole plane 256 from 10⁻⁶ (§3.5, two cases).
2. `WxNear::CostModel` the default.
3. `VNear::CostModel` the default (CH4.0d2's question, decided with the leaf size), on both
   classes (NEON 0.2–2.6 % slower on build plus one evaluation, within 5 %), or `Off` on NEON;
   either way the P2P's phase range becomes 44 c_leaf against the acceptance's 22 (§4).
4. c_leaf 2 the builder's default (`DEFAULT_C_LEAF`; D11's `C_LEAF` = 1 stays the leaf rule's
   constant and the census's), amending D11, with a roadmap line in CH4.3 to re-decide c_leaf
   with the per-level high-frequency rule (at 2 the census's first pass put leaves on
   high-frequency levels on 9 trees at 10⁻⁶; §4); or keep 1 until CH4.3.

## 7. Reproduce

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch4-0d3-runs.sh <dir> calibrate                     # §2
python3 tools/helmholtz/ch4-0d3-tables.py wx <dir>/calibrate-*.md
NEAR_WX=off,cost C_LEAF=1,2 tools/helmholtz/ch4-0d3-runs.sh <dir> matrix planes large   # §3, §4
python3 tools/helmholtz/ch4-0d3-tables.py summary <dir>/matrix-*.md
python3 tools/helmholtz/ch4-0d3-tables.py cases <dir>/large-*.md
NEAR=off,cost NEAR_WX=off,cost C_LEAF=auto,1 SIZES=auto,64 LARGE_W3_SIZES=auto,64 \
    tools/helmholtz/ch4-0d3-runs.sh <dir> matrix large                 # §5
NEAR=cost NEAR_WX=cost C_LEAF=auto TRAFFIC_SIZES=auto,64 tools/helmholtz/ch4-0d3-runs.sh <dir> traffic
python3 tools/helmholtz/ch4-0d3-tables.py checksums <dir>/traffic-*.md
```

On locust `THREADS=72`, on Kathleen inside `run.sbatch` (`tools/kathleen/`).
