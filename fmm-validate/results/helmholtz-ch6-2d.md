# Helmholtz CH6.2d: beside FMM3D's exponential M2L

Helmholtz task CH6.2d ([docs/helmholtz/tasks/CH6.2d.md](../../docs/helmholtz/tasks/CH6.2d.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.2 (split on this branch: FMM3D per level here, the counts' corner in CH6.2e, D8 in
CH6.2f), §2 (constraints 1, 2, 6, 9), §5.3, §9.2 (the H6 row). Our side is CH6.2b's harness
([helmholtz-ch6-2b.md](helmholtz-ch6-2b.md)), the scan of ours CH6.1e's
([helmholtz-ch6-1e.md](helmholtz-ch6-1e.md) §2, the appendix). Measured on:

- **Kathleen** (two Xeon Gold 6248 sockets, 2 × 20 Cascade Lake cores, AVX-512,
  `tools/kathleen/`; exclusive nodes in `singlenode`): FMM3D in job 630125
  (node-c11a-109, 2026-11-07 00:08–00:45, source 3f683b5; the four builds of FMM3D v2.1.0
  with the gcc 12.3.0 module, `-O3 -march=native`), 40 OpenMP threads; ours in job 630164
  (node-c11a-111, 2026-11-07, source 9ff114a, the pass timers), one rank × 40 threads,
  `tools/helmholtz/ch6-2b-runs.sh matrix large` (each case forward and reversed, ABBA, the
  second evaluation's times). Open MPI 4.1.6, the P2P kernel `Auto` (AVX-512).
- **locust** (NVIDIA GH200, 72 Neoverse-V2 cores, `tools/gh200/`): pending, see §6. The runs
  made so far had another session's test on one core and are kept as records only.
- **the M6**: the builds, the bit-for-bit check and smoke runs of every stage (§5); nothing
  timed.

Release builds, rustc 1.99.0, BLAS threads 1, the default waves budget (8 GiB per node).
FMM3D's whole call is timed by the driver (its setup included, as it keeps nothing between
calls); ours is the build apart from the evaluation. Commands: §7.

## Summary

- **The tools.** `tools/fmm3d/hfmm3d-levels.patch`, applied by `tools/fmm3d/build.sh` to two
  new variants (`timed`, `fastker-timed`), times FMM3D's step 3 per level: the level's setup
  (quadrature, factors, rotation matrices, built in every call), the multipoles to waves, the
  parents' loop (shifts and waves to the children's locals) with its plane-wave and list-3
  thread-seconds apart, list 4, and on point-and-shoot levels its M2L; and a switch,
  `FMM3D_PW_LEVEL`, that keeps plane waves to one level (FMM3D's point-and-shoot M2L on the
  others). The patched builds write the unpatched builds' outputs byte for byte (both ε, two
  cases, on Kathleen, locust and the M6). Our exponential M2L's level call now times its two
  passes per level (`HelmholtzOperator::plane_wave_times`: the waves, and merge and shift;
  no value changed).
- **Per level, FMM3D's exponential M2L is faster than ours on most levels on Kathleen**
  (§2): FMM3D's M2L over ours, geometric mean over the levels where both ran plane waves,
  0.21 on coarse levels (at most 512 boxes), 0.45 on middle ones and 0.56 on full ones (from
  4,096 boxes) at ε = 10⁻³; 0.44, 0.63 and 1.03 (0.78–1.37) at 10⁻⁶. At 10⁻³ FMM3D runs its
  iprec 1, whose waves are fewer, and misses ε on three cases (below).
- **Where the time goes in ours** (§3): merge and shift is 87–88 % of our exponential M2L
  on the coarse levels and 65–69 % on the others (level 2 of W1 at kD = 100, 10⁻⁶: waves
  12.5 ms, merge and shift 149 ms at 1 × 40). Against FMM3D's parents' loop it is at par on
  the full levels (0.81–1.13×) and 2.4–3.8× slower on levels 2 and 3. Level 2 has 8
  parents and both codes split this step by parent, so both use at most 8 threads there
  (FMM3D's loop kept 7.7 of 40 busy, about 38 ms per parent for its six directions; ours
  about 149 ms per parent for its three frames; 2.4× on locust in a record run). Our waves
  (12.5 ms) match FMM3D's multipoles to waves (13.8 ms). Why a coarse level's parent costs
  ours so much more is open: our counts give 1.31–1.51× FMM3D's waves (CH6.1a; most waves
  on the coarse levels), merge and shift also converts each child's waves to a local, and
  our loops are scalar where FMM3D's are vectorised (`-march=native`, AVX-512 on
  Kathleen). The next measurements are CH6.2f's (CH6.2e measures the counts).
- **The whole call is faster in ours on the large cases** (§2): one evaluation of W1 at
  kD = 500 took 22.4 s at 10⁻⁶ against FMM3D's 61.6 s (`default`) and 43.2 s (`fastker`)
  per call, W3 at kD = 100 10.9 s against 31.9 and 22.6 s, our build apart. Our RCR on the
  high-frequency levels (W1 at kD = 500, levels 2–3) and FMM3D's per-call setup (4.0 s at
  10⁻⁶ there) weigh more than the M2L levels FMM3D wins.
- **Per-level accuracy** (§4): with every level on point-and-shoot, FMM3D's error is far
  below ε (1.6 × 10⁻⁸ to 1.5 × 10⁻⁶ at 10⁻³, 10⁻¹⁵ to 1.1 × 10⁻⁹ at 10⁻⁶): its degrees are
  generous and its error is set by its plane-wave quadratures. One level at κ ≈ 12.5 on
  plane waves alone gives FMM3D 2.0 × 10⁻³ on the W4 cube at kD = 100 and 1.2 × 10⁻³ on W1
  at kD = 100 at 10⁻³ (iprec 1), above ε; ours at the same level with the same quadrature's nodes (iprec 1, our counts)
  stays within 1.007–1.11× Rotation's error (2.1 × 10⁻⁵ and 4.2 × 10⁻⁵ in φ). The two codes
  differ there in their counts (ours resolve the region's corner, FMM3D's do not, S4 §3.2)
  and their degrees (FMM3D's are larger), so FMM3D's 30–100× larger error points at its
  counts: CH6.2e's corner question has a first answer from FMM3D's side.
- **End-to-end errors at the same ε** (§2): FMM3D exceeds ε at 10⁻³ on W1 at kD = 100
  (1.23 × 10⁻³), W3 at kD = 100 (2.03 × 10⁻³) and the W4 cube at kD = 100 (1.99 × 10⁻³),
  meets it elsewhere, and meets 10⁻⁶ everywhere (at most 7.9 × 10⁻⁷); ours meets ε on every
  case (at most 1.5 × 10⁻⁴ at 10⁻³, 4.1 × 10⁻⁷ at 10⁻⁶). The times above are therefore at the
  same requested ε, not at matched accuracy, where FMM3D's 10⁻³ times would rise.

## 1. Method

- **Problems**: CH6.2b's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4
  distributions at N = 10⁵ and kD = 1, 100) and large cases (W1 at kD = 500, W3 at kD = 100),
  the problem files of `helmholtz_fmm3d problem` (CH3.6), at ε = 10⁻³ and 10⁻⁶ for both codes.
- **FMM3D per level**: the patched driver's `hfmm3d-level:` lines (`tools/fmm3d/README.md`,
  "Per level (CH6.2d)"), REPEAT 2, the second call's lines, two launches (a, b), their mean.
  FMM3D's M2L on a plane-wave level is its multipoles to waves plus the parents' loop's wall
  time times the loop's plane-wave share of its thread-seconds (the rest is list 3); the
  setup and lists 3 and 4 are apart. List 4's waves are shifted inside the same loop and
  cannot be separated without changing FMM3D's code; `h3dlist4pw` bounds their share.
- **Ours per level**: CH6.2b's `m2l_seconds` of the exponential M2L (the rule) and of the
  other strategies, the slowest rank's, with this task's two passes.
- **Levels matched by κ**: our root box is the unit cube; our level is
  round(log2(kD / Re κ_FMM3D)). FMM3D's leaves hold more points, so its trees are shallower
  and its deepest level may hold fewer boxes than ours (W1 at kD = 500, level 7: 3,264 boxes
  in FMM3D's, 14.5 × 10⁶ V pairs in ours); levels are grouped by FMM3D's boxes.
- **The scan**: as CH6.1e's, on FMM3D's side: `FMM3D_PW_LEVEL=-1` (every level on FMM3D's
  point-and-shoot M2L, the reference), then each level FMM3D runs on plane waves alone, each
  run's errors at the 1,000 samples against the f64 direct sum (`helmholtz_fmm3d errors`),
  beside CH6.1e's scan of ours at the same precision (iprec 1 at 10⁻³, 2 at 10⁻⁶, FMM3D's
  `hwts3e` rule) against Rotation. FMM3D's reference is far more accurate than its plane
  waves, so its ratios are large; the absolute errors compare.

## 2. Kathleen: per level and per call

FMM3D's M2L over ours on the levels where both ran plane waves, by FMM3D's boxes on the level:

| machine | eps | coarse | middle | full |
| --- | --- | --- | --- | --- |
| kathleen | 1e-3 | 0.21 (0.07–0.51, 30 levels) | 0.45 (0.30–0.63, 18 levels) | 0.56 (0.36–0.75, 12 levels) |
| kathleen | 1e-6 | 0.44 (0.21–0.74, 28 levels) | 0.63 (0.06–1.19, 20 levels) | 1.03 (0.78–1.37, 4 levels) |

Reading it:

- **Coarse levels** (64 and 512 boxes, levels 2 and 3 of most cases, and the deepest levels
  of the W4 Plummer sphere and clusters): FMM3D 2.0–14× faster at 10⁻³ (4.8× in the
  geometric mean) and 1.4–4.8× at 10⁻⁶ (2.3×). Both codes split merge and shift by parent
  (§3); there are 8 parents on level 2, 64 on level 3.
- **Middle and full levels**: FMM3D still ahead at 10⁻³ (0.30–0.75), where it runs iprec 1
  with fewer waves than our iprec 2, and misses ε on three cases (§4). At 10⁻⁶ the two
  are within 0.78–1.37 on the full levels (W1 at kD = 500 levels 5–6: FMM3D 0.78–0.84×
  ours; W3 at kD = 100 levels 4–5: 1.28–1.37×, ours faster).
- Per level, per case: the appendix. The tables of every case with FMM3D's setup, list 3
  and list 4, our Rotation and the per-level best, are in `bench-results/ch6-2d/` (not
  committed), from `tools/helmholtz/ch6-2d-tables.py`.

Per call, at the same requested ε (FMM3D's call includes its setup; ours is the evaluation
under the exponential M2L's rule and under `Auto`, our build apart; FMM3D's errors and ours
against the same sampled direct sum):

| case | ε | FMM3D `default` call s | FMM3D `fastker` call s | ours evaluation s, exp. M2L | ours evaluation s, Auto | FMM3D φ, ∇φ | ours φ, ∇φ (exp. M2L) |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| w1-10 | 1e-3 | 0.008 | 0.003 | 0.014 | 0.006 | 1.54e-05, 3.95e-05 | 4.40e-05, 1.53e-04 |
| w1-10 | 1e-6 | 0.007 | 0.014 | 0.023 | 0.115 | 1.22e-15, 1.30e-15 | 5.65e-08, 4.09e-07 |
| w1-30 | 1e-3 | 0.053 | 0.026 | 0.050 | 0.052 | 7.50e-05, 7.42e-05 | 2.66e-05, 7.15e-05 |
| w1-30 | 1e-6 | 0.073 | 0.055 | 0.071 | 0.101 | 1.05e-08, 1.19e-08 | 1.12e-07, 2.97e-07 |
| w1-100 | 1e-3 | 0.907 | 0.413 | 0.476 | 0.253 | 1.23e-03, 1.20e-03 | 4.15e-05, 1.36e-04 |
| w1-100 | 1e-6 | 0.903 | 0.489 | 0.715 | 0.824 | 3.25e-07, 4.40e-07 | 9.56e-08, 2.88e-07 |
| w1-500 | 1e-3 | 48.944 | 33.120 | 16.812 | 22.008 | 1.87e-04, 1.75e-04 | 3.24e-05, 8.71e-05 |
| w1-500 | 1e-6 | 61.631 | 43.198 | 22.423 | 45.327 | 6.21e-08, 7.20e-08 | 1.45e-07, 3.73e-07 |
| w3-10 | 1e-3 | 0.021 | 0.011 | 0.036 | 0.021 | 1.73e-05, 2.27e-05 | 4.05e-05, 8.10e-05 |
| w3-10 | 1e-6 | 0.037 | 0.022 | 0.073 | 0.217 | 9.66e-09, 2.09e-08 | 4.45e-08, 1.61e-07 |
| w3-30 | 1e-3 | 0.354 | 0.209 | 0.160 | 0.102 | 1.01e-04, 8.82e-05 | 4.60e-05, 1.31e-04 |
| w3-30 | 1e-6 | 0.939 | 0.306 | 0.257 | 0.578 | 1.58e-08, 4.34e-08 | 1.42e-07, 4.02e-07 |
| w3-100 | 1e-3 | 22.243 | 7.276 | 6.065 | 3.147 | 2.03e-03, 1.31e-03 | 4.86e-05, 1.08e-04 |
| w3-100 | 1e-6 | 31.904 | 22.551 | 10.891 | 31.645 | 5.45e-07, 5.18e-07 | 1.02e-07, 2.46e-07 |
| w4-clusters-1 | 1e-3 | 0.891 | 0.450 | 0.321 | 0.212 | 1.13e-04, 1.60e-04 | 5.07e-05, 1.01e-04 |
| w4-clusters-1 | 1e-6 | 1.869 | 1.167 | 0.714 | 1.286 | 4.30e-08, 4.60e-08 | 9.16e-08, 3.69e-07 |
| w4-clusters-100 | 1e-3 | 1.144 | 0.623 | 0.407 | 0.268 | 6.49e-05, 7.92e-05 | 5.26e-05, 1.03e-04 |
| w4-clusters-100 | 1e-6 | 2.213 | 1.376 | 0.805 | 1.278 | 2.70e-08, 4.19e-08 | 8.81e-08, 3.71e-07 |
| w4-cube-1 | 1e-3 | 0.217 | 0.152 | 0.129 | 0.069 | 1.63e-04, 2.15e-04 | 4.07e-05, 8.96e-05 |
| w4-cube-1 | 1e-6 | 0.829 | 0.220 | 0.229 | 0.704 | 5.66e-08, 1.18e-07 | 6.29e-08, 2.46e-07 |
| w4-cube-100 | 1e-3 | 0.685 | 0.364 | 0.411 | 0.237 | 1.99e-03, 2.12e-03 | 1.85e-05, 4.04e-05 |
| w4-cube-100 | 1e-6 | 1.522 | 0.532 | 0.500 | 0.567 | 5.24e-07, 7.90e-07 | 9.66e-08, 3.01e-07 |
| w4-plummer-1 | 1e-3 | 0.702 | 0.396 | 0.300 | 0.205 | 1.50e-04, 1.63e-04 | 6.47e-05, 9.67e-05 |
| w4-plummer-1 | 1e-6 | 1.547 | 0.974 | 0.680 | 1.080 | 7.37e-08, 7.04e-08 | 1.18e-07, 3.13e-07 |
| w4-plummer-100 | 1e-3 | 1.180 | 0.749 | 0.585 | 0.359 | 2.87e-04, 1.70e-04 | 7.95e-05, 1.18e-04 |
| w4-plummer-100 | 1e-6 | 2.295 | 1.360 | 0.981 | 1.405 | 5.87e-08, 6.39e-08 | 1.55e-07, 3.42e-07 |
| w4-sphere-1 | 1e-3 | 0.359 | 0.140 | 0.216 | 0.112 | 7.15e-05, 1.05e-06 | 2.30e-05, 8.09e-07 |
| w4-sphere-1 | 1e-6 | 0.506 | 0.298 | 0.438 | 0.866 | 2.10e-08, 4.74e-10 | 4.18e-08, 2.23e-09 |
| w4-sphere-100 | 1e-3 | 0.972 | 0.428 | 0.491 | 0.259 | 9.08e-04, 1.83e-05 | 3.06e-05, 2.04e-06 |
| w4-sphere-100 | 1e-6 | 1.042 | 0.581 | 0.732 | 0.889 | 2.25e-07, 6.41e-09 | 6.26e-08, 3.97e-09 |

## 3. Our level call's two passes

`HelmholtzOperator::plane_wave_times` splits each level's exponential M2L into the waves of
each chunk's sources (box by box on the pool) and merge and shift (parent by parent on the
pool, including each child's conversion of its waves back to a local); the rest of the
level's M2L time is the chunking between them. Kathleen, 1 × 40, ε = 10⁻⁶, beside FMM3D's
multipoles to waves and its parents' loop (with the threads its loop kept busy, its
thread-seconds over its wall time; the loop includes list 3 where the level has any, on W1
at kD = 500 levels 6–7):

| case | level | boxes (FMM3D) | FMM3D multipoles to waves s | FMM3D parents' loop s (threads busy) | ours waves s | ours merge and shift s | merge and shift ÷ FMM3D's parents' loop |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| w1-100 | 2 | 64 | 0.0154 | 0.0395 (7.7) | 0.0125 | 0.1487 | 3.76 |
| w1-100 | 3 | 512 | 0.0152 | 0.0336 (27.2) | 0.0156 | 0.0794 | 2.36 |
| w1-100 | 4 | 1600 | 0.0257 | 0.0906 (18.7) | 0.0417 | 0.0898 | 0.99 |
| w3-100 | 2 | 64 | 0.0112 | 0.0396 (7.7) | 0.0133 | 0.1496 | 3.78 |
| w3-100 | 3 | 512 | 0.0199 | 0.0326 (30.5) | 0.0164 | 0.0793 | 2.43 |
| w3-100 | 4 | 4096 | 0.0927 | 0.1333 (37.3) | 0.0521 | 0.1241 | 0.93 |
| w3-100 | 5 | 32768 | 0.5229 | 0.7328 (38.2) | 0.3115 | 0.5972 | 0.81 |
| w1-500 | 4 | 3840 | 0.3872 | 0.9059 (25.4) | 0.5515 | 1.0249 | 1.13 |
| w1-500 | 5 | 16624 | 0.4341 | 1.2813 (20.2) | 0.7900 | 1.4033 | 1.10 |
| w1-500 | 6 | 34432 | 0.6110 | 2.4994 (16.5) | 1.0602 | 2.0705 | 0.83 |
| w1-500 | 7 | 3264 | 0.0629 | 2.0044 (1.7) | 1.7010 | 3.0409 | 1.52 |

Reading it:

- **Merge and shift is most of our exponential M2L**: 87–88 % of its time on the coarse
  levels, 65–69 % on the others (Kathleen, every case, both ε).
- **The waves** (ours) against FMM3D's multipoles to waves: 0.56–1.8×, ours faster on W3 at
  kD = 100, slower on W1 at kD = 500's levels 4–6.
- **Merge and shift against FMM3D's parents' loop**: at par on the full levels (0.81–0.93×
  on W3 at kD = 100, levels 4–5; 0.83–1.13× on W1 at kD = 500, levels 4–6), 2.4–3.8× slower
  on the coarse ones (levels 2 and 3, κ = 25 and 12.5).
- **Level 2 holds both codes to 8 threads.** It has 8 parents and both split this step by
  parent: FMM3D's loop kept 7.7 of 40 threads busy, about 38 ms per parent for its six
  directions; ours took 149 ms of wall time, about 149 ms per parent for its three frames
  (our level 2 takes 46 ms on the M6's 6 threads, 60 ms on locust's 72 and 161 ms on
  Kathleen's 40: the cores' speed, not their number). On level 3 (64 parents) FMM3D's loop
  kept 27–31 threads busy.
- **Open: why a coarse level's parent costs ours 2.4–3.8× FMM3D's while a full level's does
  not.** The candidates, not separated by these timers: our counts give 1.31–1.51× FMM3D's
  waves (CH6.1a), and the coarse levels have the most waves (5,568 of FMM3D's at κ = 25,
  982 at κ = 3.1); merge and shift also converts each child's waves back to a local at the
  level's degree; and our loops are scalar, FMM3D's vectorised, which pays most on the
  longest wave vectors. Splitting merge and shift into its shifts and its conversions, and
  the parents' pass below the parent (CH6.2f's coarse-level remedy), are the next
  measurements; CH6.2f's line records them, and CH6.2e measures the counts.

## 4. Per-level accuracy: the scan

Each level that FMM3D runs on plane waves alone, the others on its point-and-shoot M2L,
beside CH6.1e's scan of ours (that level alone with the exponential M2L at the same iprec,
the others Rotation). Kathleen, FMM3D `timed` on 40 threads (job 630125); ours CH6.1e's
M6 rows (accuracy does not depend on the machine). The errors at the 1,000 samples against
the f64 direct sum; each code's ratio over its own reference run:

| case | eps | our level | Re κ | FMM3D p | FMM3D level alone φ, ∇φ | ÷ FMM3D ps | ours iprec | our level alone φ, ∇φ | ÷ Rotation |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |
| w1@10 | 1e-3 | 2 | 2.5 | 13 | 1.54e-05, 3.95e-05 | 62.3, 27 | 1 | 4.41e-05, 1.53e-04 | 1.002, 1.000 |
| w1@100 | 1e-3 | 2 | 25 | 35 | 2.73e-04, 2.41e-04 | 1.1e+04, 3.87e+03 | 1 | 4.21e-05, 1.36e-04 | 1.014, 1.000 |
| w1@100 | 1e-3 | 3 | 12.5 | 23 | 1.19e-03, 1.17e-03 | 4.79e+04, 1.87e+04 | 1 | 4.18e-05, 1.36e-04 | 1.007, 1.000 |
| w1@100 | 1e-3 | 4 | 6.25 | 18 | 3.86e-05, 6.14e-05 | 1.55e+03, 985 | 1 | 4.23e-05, 1.36e-04 | 1.019, 1.000 |
| w1@100 | 1e-3 | 5 | 3.12 | 17 | 3.98e-05, 9.15e-05 | 1.6e+03, 1.47e+03 | 1 | 4.16e-05, 1.36e-04 | 1.002, 1.000 |
| w1@100 | 1e-6 | 2 | 25 | 42 | 7.85e-08, 8.66e-08 | 6.81e+03, 941 | 2 | 9.62e-08, 2.94e-07 | 1.001, 1.000 |
| w1@100 | 1e-6 | 3 | 12.5 | 32 | 3.10e-07, 4.26e-07 | 2.69e+04, 4.63e+03 | 2 | 9.66e-08, 2.94e-07 | 1.005, 1.000 |
| w1@100 | 1e-6 | 4 | 6.25 | 29 | 2.80e-08, 5.60e-08 | 2.43e+03, 608 | 2 | 9.50e-08, 2.90e-07 | 0.989, 0.986 |
| w1@30 | 1e-3 | 2 | 7.5 | 18 | 7.48e-05, 7.37e-05 | 2.59e+03, 852 | 1 | 2.82e-05, 7.18e-05 | 1.060, 1.004 |
| w1@30 | 1e-3 | 3 | 3.75 | 16 | 4.34e-06, 7.39e-06 | 150, 85.5 | 1 | 2.70e-05, 7.17e-05 | 1.015, 1.003 |
| w1@30 | 1e-6 | 2 | 7.5 | 28 | 1.05e-08, 1.19e-08 | 609, 197 | 2 | 1.12e-07, 2.97e-07 | 1.000, 1.000 |
| w3@10 | 1e-3 | 2 | 2.5 | 13 | 1.73e-05, 2.27e-05 | 104, 45.7 | 1 | 4.10e-05, 8.10e-05 | 1.012, 1.000 |
| w3@10 | 1e-6 | 2 | 2.5 | 24 | 9.66e-09, 2.09e-08 | 113, 40.7 | 2 | 4.44e-08, 1.65e-07 | 0.989, 0.971 |
| w3@30 | 1e-3 | 2 | 7.5 | 18 | 1.00e-04, 8.17e-05 | 1.45e+03, 188 | 1 | 4.66e-05, 1.31e-04 | 1.013, 1.000 |
| w3@30 | 1e-3 | 3 | 3.75 | 16 | 4.83e-06, 7.10e-06 | 69.7, 16.3 | 1 | 4.63e-05, 1.31e-04 | 1.007, 1.000 |
| w3@30 | 1e-3 | 4 | 1.87 | 14 | 8.80e-06, 3.27e-05 | 127, 75.3 | 1 | 4.61e-05, 1.31e-04 | 1.002, 1.000 |
| w3@30 | 1e-6 | 2 | 7.5 | 28 | 1.40e-08, 1.62e-08 | 134, 36.9 | 2 | 1.41e-07, 4.11e-07 | 0.966, 0.986 |
| w3@30 | 1e-6 | 3 | 3.75 | 27 | 3.58e-09, 8.10e-09 | 34.3, 18.5 | 2 | 1.46e-07, 4.15e-07 | 1.000, 0.995 |
| w3@30 | 1e-6 | 4 | 1.87 | 26 | 6.69e-09, 3.94e-08 | 64.3, 89.8 | 2 | 1.46e-07, 4.11e-07 | 1.000, 0.986 |
| w4-clusters@1 | 1e-3 | 3 | 0.157 | 12 | 3.84e-06, 1.41e-07 | 80.9, 1.58 | 1 | 5.07e-05, 1.01e-04 | 1.000, 1.000 |
| w4-clusters@1 | 1e-3 | 4 | 0.0784 | 13 | 6.93e-07, 1.37e-07 | 14.6, 1.53 | 1 | 5.07e-05, 1.01e-04 | 1.000, 1.000 |
| w4-clusters@1 | 1e-3 | 5 | 0.0392 | 14 | 5.00e-06, 1.03e-06 | 105, 11.4 | 1 | 5.08e-05, 1.01e-04 | 1.002, 1.000 |
| w4-clusters@1 | 1e-3 | 6 | 0.0196 | 15 | 2.31e-05, 8.72e-06 | 487, 97.1 | 1 | 5.20e-05, 1.01e-04 | 1.026, 1.000 |
| w4-clusters@1 | 1e-3 | 7 | 0.00979 | 16 | 6.56e-05, 4.25e-05 | 1.38e+03, 474 | 1 | 5.38e-05, 1.02e-04 | 1.061, 1.010 |
| w4-clusters@1 | 1e-3 | 8 | 0.0049 | 17 | 7.63e-05, 9.74e-05 | 1.61e+03, 1.08e+03 | 1 | 5.14e-05, 1.01e-04 | 1.014, 1.000 |
| w4-clusters@1 | 1e-3 | 9 | 0.00245 | 19 | 4.72e-05, 1.18e-04 | 994, 1.32e+03 | 1 | – | – |
| w4-clusters@1 | 1e-6 | 3 | 0.157 | 23 | 7.85e-10, 2.08e-10 | 14.4, 1 | 2 | 9.45e-08, 4.03e-07 | 1.000, 1.000 |
| w4-clusters@1 | 1e-6 | 4 | 0.0784 | 24 | 3.92e-10, 2.21e-10 | 7.2, 1.07 | 2 | 9.45e-08, 4.03e-07 | 1.000, 1.000 |
| w4-clusters@1 | 1e-6 | 5 | 0.0392 | 25 | 1.89e-09, 7.14e-10 | 34.8, 3.46 | 2 | 9.43e-08, 4.02e-07 | 0.998, 0.998 |
| w4-clusters@1 | 1e-6 | 6 | 0.0196 | 27 | 1.26e-08, 4.74e-09 | 231, 22.9 | 2 | 9.33e-08, 4.01e-07 | 0.987, 0.995 |
| w4-clusters@1 | 1e-6 | 7 | 0.00979 | 28 | 1.96e-08, 1.69e-08 | 360, 81.7 | 2 | 9.60e-08, 3.99e-07 | 1.016, 0.990 |
| w4-clusters@1 | 1e-6 | 8 | 0.0049 | 29 | 3.56e-08, 4.24e-08 | 653, 205 | 2 | 9.14e-08, 3.74e-07 | 0.967, 0.928 |
| w4-clusters@100 | 1e-3 | 3 | 15.7 | 25 | 1.70e-05, 1.66e-06 | 510, 19.8 | 1 | 5.26e-05, 1.03e-04 | 1.000, 1.000 |
| w4-clusters@100 | 1e-3 | 4 | 7.84 | 19 | 4.89e-07, 1.07e-07 | 14.7, 1.28 | 1 | 5.26e-05, 1.03e-04 | 1.000, 1.000 |
| w4-clusters@100 | 1e-3 | 5 | 3.92 | 17 | 9.24e-07, 2.40e-07 | 27.8, 2.86 | 1 | 5.26e-05, 1.03e-04 | 1.000, 1.000 |
| w4-clusters@100 | 1e-3 | 6 | 1.96 | 16 | 7.09e-06, 3.71e-06 | 213, 44.2 | 1 | 5.39e-05, 1.03e-04 | 1.025, 1.000 |
| w4-clusters@100 | 1e-3 | 7 | 0.979 | 16 | 4.99e-05, 2.76e-05 | 1.5e+03, 328 | 1 | 5.42e-05, 1.03e-04 | 1.030, 1.000 |
| w4-clusters@100 | 1e-3 | 8 | 0.49 | 17 | 2.62e-05, 3.72e-05 | 787, 443 | 1 | 5.32e-05, 1.03e-04 | 1.011, 1.000 |
| w4-clusters@100 | 1e-3 | 9 | 0.245 | 19 | 2.24e-05, 6.36e-05 | 673, 758 | 1 | – | – |
| w4-clusters@100 | 1e-6 | 3 | 15.7 | 33 | 7.14e-09, 9.43e-10 | 143, 4.56 | 2 | 9.08e-08, 4.03e-07 | 1.000, 1.000 |
| w4-clusters@100 | 1e-6 | 4 | 7.84 | 29 | 1.84e-10, 2.12e-10 | 3.66, 1.02 | 2 | 9.08e-08, 4.03e-07 | 1.000, 1.000 |
| w4-clusters@100 | 1e-6 | 5 | 3.92 | 28 | 5.54e-10, 3.15e-10 | 11.1, 1.52 | 2 | 9.04e-08, 4.03e-07 | 0.996, 1.000 |
| w4-clusters@100 | 1e-6 | 6 | 1.96 | 27 | 5.39e-09, 4.60e-09 | 108, 22.2 | 2 | 9.06e-08, 4.02e-07 | 0.998, 0.998 |
| w4-clusters@100 | 1e-6 | 7 | 0.979 | 28 | 1.96e-08, 2.00e-08 | 392, 96.5 | 2 | 9.13e-08, 3.98e-07 | 1.006, 0.988 |
| w4-clusters@100 | 1e-6 | 8 | 0.49 | 29 | 1.78e-08, 3.70e-08 | 356, 179 | 2 | 8.83e-08, 3.78e-07 | 0.972, 0.938 |
| w4-cube@1 | 1e-3 | 2 | 0.25 | 12 | 3.30e-05, 2.04e-05 | 79, 25.1 | 1 | 4.29e-05, 8.96e-05 | 1.054, 1.000 |
| w4-cube@1 | 1e-3 | 3 | 0.125 | 13 | 1.52e-04, 1.65e-04 | 364, 203 | 1 | 4.62e-05, 8.98e-05 | 1.135, 1.002 |
| w4-cube@1 | 1e-3 | 4 | 0.0625 | 14 | 5.86e-05, 1.39e-04 | 140, 170 | 1 | 4.47e-05, 9.01e-05 | 1.098, 1.006 |
| w4-cube@1 | 1e-6 | 2 | 0.25 | 23 | 1.53e-08, 1.74e-08 | 42.6, 16.2 | 2 | 6.38e-08, 2.64e-07 | 0.997, 0.996 |
| w4-cube@1 | 1e-6 | 3 | 0.125 | 24 | 5.16e-08, 1.00e-07 | 143, 93.1 | 2 | 6.26e-08, 2.61e-07 | 0.978, 0.985 |
| w4-cube@1 | 1e-6 | 4 | 0.0625 | 25 | 1.67e-08, 6.15e-08 | 46.2, 57.1 | 2 | 6.42e-08, 2.53e-07 | 1.003, 0.955 |
| w4-cube@100 | 1e-3 | 2 | 25 | 35 | 3.82e-04, 3.58e-04 | 1.78e+04, 8.31e+03 | 1 | 2.07e-05, 4.14e-05 | 1.119, 1.025 |
| w4-cube@100 | 1e-3 | 3 | 12.5 | 23 | 1.98e-03, 2.10e-03 | 9.22e+04, 4.87e+04 | 1 | 2.05e-05, 4.11e-05 | 1.108, 1.017 |
| w4-cube@100 | 1e-3 | 4 | 6.25 | 18 | 5.89e-05, 1.01e-04 | 2.75e+03, 2.34e+03 | 1 | 2.11e-05, 4.15e-05 | 1.141, 1.027 |
| w4-cube@100 | 1e-6 | 2 | 25 | 42 | 1.04e-07, 1.19e-07 | 1.58e+04, 5.1e+03 | 2 | 9.66e-08, 3.04e-07 | 1.003, 1.000 |
| w4-cube@100 | 1e-6 | 3 | 12.5 | 32 | 5.14e-07, 7.81e-07 | 7.78e+04, 3.35e+04 | 2 | 9.69e-08, 3.04e-07 | 1.006, 1.000 |
| w4-cube@100 | 1e-6 | 4 | 6.25 | 29 | 3.05e-08, 6.38e-08 | 4.62e+03, 2.74e+03 | 2 | 9.58e-08, 3.01e-07 | 0.995, 0.990 |
| w4-plummer@1 | 1e-3 | 2 | 0.246 | 12 | 4.52e-06, 4.18e-07 | 18.1, 2.82 | 1 | 6.48e-05, 9.67e-05 | 1.002, 1.000 |
| w4-plummer@1 | 1e-3 | 3 | 0.123 | 13 | 5.32e-05, 6.66e-06 | 213, 44.9 | 1 | 6.50e-05, 9.67e-05 | 1.005, 1.000 |
| w4-plummer@1 | 1e-3 | 4 | 0.0615 | 14 | 4.97e-05, 1.38e-05 | 199, 93.1 | 1 | 6.51e-05, 9.67e-05 | 1.006, 1.000 |
| w4-plummer@1 | 1e-3 | 5 | 0.0307 | 15 | 4.02e-05, 2.18e-05 | 161, 147 | 1 | 6.56e-05, 9.66e-05 | 1.014, 0.999 |
| w4-plummer@1 | 1e-3 | 6 | 0.0154 | 16 | 8.78e-05, 8.04e-05 | 352, 542 | 1 | 6.95e-05, 9.69e-05 | 1.074, 1.002 |
| w4-plummer@1 | 1e-3 | 7 | 0.00769 | 17 | 7.40e-05, 1.28e-04 | 297, 860 | 1 | 6.86e-05, 9.67e-05 | 1.060, 1.000 |
| w4-plummer@1 | 1e-3 | 8 | 0.00384 | 19 | 1.79e-05, 5.83e-05 | 71.6, 393 | 1 | 6.40e-05, 9.68e-05 | 0.989, 1.001 |
| w4-plummer@1 | 1e-6 | 2 | 0.246 | 23 | 2.02e-09, 6.93e-10 | 7.57, 1.13 | 2 | 1.10e-07, 3.33e-07 | 0.991, 1.000 |
| w4-plummer@1 | 1e-6 | 3 | 0.123 | 24 | 1.51e-08, 3.17e-09 | 56.4, 5.2 | 2 | 1.11e-07, 3.33e-07 | 1.000, 1.000 |
| w4-plummer@1 | 1e-6 | 4 | 0.0615 | 25 | 1.57e-08, 7.65e-09 | 58.6, 12.5 | 2 | 1.11e-07, 3.33e-07 | 1.000, 1.000 |
| w4-plummer@1 | 1e-6 | 5 | 0.0307 | 27 | 1.58e-08, 1.08e-08 | 59.2, 17.8 | 2 | 1.10e-07, 3.30e-07 | 0.991, 0.991 |
| w4-plummer@1 | 1e-6 | 6 | 0.0154 | 28 | 5.62e-08, 4.10e-08 | 210, 67.1 | 2 | 1.16e-07, 3.25e-07 | 1.045, 0.976 |
| w4-plummer@1 | 1e-6 | 7 | 0.00769 | 29 | 3.70e-08, 5.68e-08 | 138, 92.9 | 2 | 1.13e-07, 3.24e-07 | 1.018, 0.973 |
| w4-plummer@100 | 1e-3 | 2 | 24.6 | 34 | 1.43e-05, 3.70e-06 | 436, 36.1 | 1 | 7.95e-05, 1.18e-04 | 1.000, 1.000 |
| w4-plummer@100 | 1e-3 | 3 | 12.3 | 23 | 2.39e-04, 6.71e-05 | 7.3e+03, 655 | 1 | 7.95e-05, 1.18e-04 | 1.000, 1.000 |
| w4-plummer@100 | 1e-3 | 4 | 6.15 | 18 | 1.70e-05, 7.52e-06 | 518, 73.4 | 1 | 7.95e-05, 1.18e-04 | 1.000, 1.000 |
| w4-plummer@100 | 1e-3 | 5 | 3.07 | 17 | 4.15e-05, 2.91e-05 | 1.27e+03, 284 | 1 | 7.95e-05, 1.18e-04 | 1.000, 1.000 |
| w4-plummer@100 | 1e-3 | 6 | 1.54 | 16 | 1.46e-04, 1.30e-04 | 4.44e+03, 1.27e+03 | 1 | 8.05e-05, 1.18e-04 | 1.013, 1.000 |
| w4-plummer@100 | 1e-3 | 7 | 0.769 | 17 | 4.00e-05, 7.09e-05 | 1.22e+03, 692 | 1 | 8.07e-05, 1.18e-04 | 1.015, 1.000 |
| w4-plummer@100 | 1e-3 | 8 | 0.384 | 19 | 9.71e-06, 3.68e-05 | 296, 359 | 1 | 7.98e-05, 1.18e-04 | 1.004, 1.000 |
| w4-plummer@100 | 1e-6 | 2 | 24.6 | 41 | 3.34e-09, 1.36e-09 | 39.2, 2.62 | 2 | 1.53e-07, 3.58e-07 | 1.000, 1.000 |
| w4-plummer@100 | 1e-6 | 3 | 12.3 | 32 | 5.04e-08, 1.84e-08 | 591, 35.5 | 2 | 1.53e-07, 3.58e-07 | 1.000, 1.000 |
| w4-plummer@100 | 1e-6 | 4 | 6.15 | 29 | 7.35e-09, 3.82e-09 | 86.1, 7.37 | 2 | 1.53e-07, 3.58e-07 | 1.000, 1.000 |
| w4-plummer@100 | 1e-6 | 5 | 3.07 | 28 | 1.45e-08, 1.78e-08 | 169, 34.3 | 2 | 1.54e-07, 3.56e-07 | 1.007, 0.994 |
| w4-plummer@100 | 1e-6 | 6 | 1.54 | 28 | 1.93e-08, 3.41e-08 | 226, 65.8 | 2 | 1.54e-07, 3.51e-07 | 1.007, 0.980 |
| w4-plummer@100 | 1e-6 | 7 | 0.769 | 29 | 1.65e-08, 4.72e-08 | 193, 91.1 | 2 | 1.53e-07, 3.51e-07 | 1.000, 0.980 |
| w4-sphere@1 | 1e-3 | 2 | 0.25 | 12 | 1.47e-05, 6.94e-08 | 50.5, 15.4 | 1 | 2.36e-05, 8.09e-07 | 1.026, 1.000 |
| w4-sphere@1 | 1e-3 | 3 | 0.125 | 13 | 6.02e-05, 6.00e-07 | 207, 133 | 1 | 2.46e-05, 8.09e-07 | 1.070, 1.000 |
| w4-sphere@1 | 1e-3 | 4 | 0.0625 | 14 | 2.82e-05, 5.84e-07 | 97.3, 129 | 1 | 2.40e-05, 8.10e-07 | 1.043, 1.001 |
| w4-sphere@1 | 1e-3 | 5 | 0.0312 | 15 | 1.60e-05, 6.34e-07 | 55.3, 140 | 1 | 2.39e-05, 8.09e-07 | 1.039, 1.000 |
| w4-sphere@1 | 1e-6 | 2 | 0.25 | 23 | 6.84e-09, 6.51e-11 | 44.6, 12.4 | 2 | 4.24e-08, 2.36e-09 | 1.000, 1.000 |
| w4-sphere@1 | 1e-6 | 3 | 0.125 | 24 | 1.69e-08, 3.10e-10 | 110, 58.9 | 2 | 4.22e-08, 2.35e-09 | 0.995, 0.996 |
| w4-sphere@1 | 1e-6 | 4 | 0.0625 | 25 | 9.94e-09, 3.45e-10 | 64.7, 65.7 | 2 | 4.24e-08, 2.34e-09 | 1.000, 0.992 |
| w4-sphere@100 | 1e-3 | 2 | 25 | 35 | 1.87e-04, 3.47e-06 | 1.18e+04, 4.45e+03 | 1 | 3.09e-05, 2.04e-06 | 1.010, 1.000 |
| w4-sphere@100 | 1e-3 | 3 | 12.5 | 23 | 8.96e-04, 1.80e-05 | 5.66e+04, 2.31e+04 | 1 | 3.08e-05, 2.04e-06 | 1.007, 1.000 |
| w4-sphere@100 | 1e-3 | 4 | 6.25 | 18 | 2.58e-05, 8.18e-07 | 1.63e+03, 1.05e+03 | 1 | 3.11e-05, 2.05e-06 | 1.016, 1.005 |
| w4-sphere@100 | 1e-3 | 5 | 3.12 | 17 | 2.87e-05, 1.36e-06 | 1.82e+03, 1.74e+03 | 1 | 3.10e-05, 2.05e-06 | 1.013, 1.005 |
| w4-sphere@100 | 1e-6 | 2 | 25 | 42 | 5.49e-08, 1.24e-09 | 8.96e+03, 1.06e+03 | 2 | 6.27e-08, 4.07e-09 | 1.005, 1.002 |
| w4-sphere@100 | 1e-6 | 3 | 12.5 | 32 | 2.17e-07, 6.25e-09 | 3.54e+04, 5.35e+03 | 2 | 6.27e-08, 4.07e-09 | 1.005, 1.002 |
| w4-sphere@100 | 1e-6 | 4 | 6.25 | 29 | 1.79e-08, 7.32e-10 | 2.92e+03, 627 | 2 | 6.25e-08, 4.05e-09 | 1.002, 0.998 |

| case | eps | FMM3D point-and-shoot φ, ∇φ | FMM3D as it is φ, ∇φ | ours Rotation φ, ∇φ |
| --- | --- | --- | --- | --- |
| w1@10 | 1e-3 | 2.48e-07, 1.46e-06 | 1.54e-05, 3.95e-05 | 4.40e-05, 1.53e-04 |
| w1@10 | 1e-6 | 1.22e-15, 1.30e-15 | 1.22e-15, 1.30e-15 | 5.62e-08, 4.11e-07 |
| w1@100 | 1e-3 | 2.49e-08, 6.23e-08 | 1.23e-03, 1.20e-03 | 4.15e-05, 1.36e-04 |
| w1@100 | 1e-6 | 1.15e-11, 9.20e-11 | 3.25e-07, 4.40e-07 | 9.61e-08, 2.94e-07 |
| w1@30 | 1e-3 | 2.89e-08, 8.64e-08 | 7.50e-05, 7.42e-05 | 2.66e-05, 7.15e-05 |
| w1@30 | 1e-6 | 1.73e-11, 6.05e-11 | 1.05e-08, 1.19e-08 | 1.12e-07, 2.97e-07 |
| w3@10 | 1e-3 | 1.66e-07, 4.97e-07 | 1.73e-05, 2.27e-05 | 4.05e-05, 8.10e-05 |
| w3@10 | 1e-6 | 8.54e-11, 5.12e-10 | 9.66e-09, 2.09e-08 | 4.49e-08, 1.70e-07 |
| w3@30 | 1e-3 | 6.93e-08, 4.34e-07 | 1.01e-04, 8.82e-05 | 4.60e-05, 1.31e-04 |
| w3@30 | 1e-6 | 1.04e-10, 4.38e-10 | 1.58e-08, 4.34e-08 | 1.46e-07, 4.17e-07 |
| w4-clusters@1 | 1e-3 | 4.74e-08, 8.98e-08 | 1.13e-04, 1.60e-04 | 5.07e-05, 1.01e-04 |
| w4-clusters@1 | 1e-6 | 5.45e-11, 2.07e-10 | 4.30e-08, 4.60e-08 | 9.45e-08, 4.03e-07 |
| w4-clusters@100 | 1e-3 | 3.33e-08, 8.39e-08 | 6.49e-05, 7.92e-05 | 5.26e-05, 1.03e-04 |
| w4-clusters@100 | 1e-6 | 5.01e-11, 2.07e-10 | 2.70e-08, 4.19e-08 | 9.08e-08, 4.03e-07 |
| w4-cube@1 | 1e-3 | 4.18e-07, 8.13e-07 | 1.63e-04, 2.15e-04 | 4.07e-05, 8.96e-05 |
| w4-cube@1 | 1e-6 | 3.60e-10, 1.08e-09 | 5.66e-08, 1.18e-07 | 6.40e-08, 2.65e-07 |
| w4-cube@100 | 1e-3 | 2.15e-08, 4.30e-08 | 1.99e-03, 2.12e-03 | 1.85e-05, 4.04e-05 |
| w4-cube@100 | 1e-6 | 6.60e-12, 2.33e-11 | 5.24e-07, 7.90e-07 | 9.63e-08, 3.04e-07 |
| w4-plummer@1 | 1e-3 | 2.49e-07, 1.48e-07 | 1.50e-04, 1.63e-04 | 6.47e-05, 9.67e-05 |
| w4-plummer@1 | 1e-6 | 2.67e-10, 6.11e-10 | 7.37e-08, 7.04e-08 | 1.11e-07, 3.33e-07 |
| w4-plummer@100 | 1e-3 | 3.28e-08, 1.02e-07 | 2.87e-04, 1.70e-04 | 7.95e-05, 1.18e-04 |
| w4-plummer@100 | 1e-6 | 8.53e-11, 5.18e-10 | 5.87e-08, 6.39e-08 | 1.53e-07, 3.58e-07 |
| w4-sphere@1 | 1e-3 | 2.90e-07, 4.52e-09 | 7.15e-05, 1.05e-06 | 2.30e-05, 8.09e-07 |
| w4-sphere@1 | 1e-6 | 1.54e-10, 5.26e-12 | 2.10e-08, 4.74e-10 | 4.24e-08, 2.36e-09 |
| w4-sphere@100 | 1e-3 | 1.58e-08, 7.81e-10 | 9.08e-04, 1.83e-05 | 3.06e-05, 2.04e-06 |
| w4-sphere@100 | 1e-6 | 6.13e-12, 1.17e-12 | 2.25e-07, 6.41e-09 | 6.24e-08, 4.06e-09 |

Reading it:

- **FMM3D's point-and-shoot reference is far more accurate than its plane waves** (1.6 × 10⁻⁸
  to 1.5 × 10⁻⁶ at 10⁻³, 10⁻¹⁵ to 1.1 × 10⁻⁹ at 10⁻⁶): its degrees are chosen generously
  (`h3dterms`; FMM3D's p is 13–42 where ours is 7–35 on the same levels), so its error is
  its quadratures', and its ratios run to 10⁴–10⁵ where ours stay at 0.93–1.14. The ratios
  do not compare; the absolute errors do.
- **At 10⁻³ (iprec 1 for both)** FMM3D's plane waves on one level give up to 2.0 × 10⁻³ (the
  W4 cube at kD = 100, level 3, κ = 12.5), 1.2 × 10⁻³ (W1 at kD = 100, level 3) and
  9.0 × 10⁻⁴ (the W4 sphere at kD = 100, level 3), at or above ε; ours on the same levels
  with the same quadrature's nodes and our counts 2.1 × 10⁻⁵, 4.2 × 10⁻⁵ and 3.1 × 10⁻⁵
  (1.007–1.11× Rotation's). FMM3D's end-to-end misses of ε at 10⁻³ (§2) come from these
  levels. Our counts resolve the region's corner ρ = 4√2, FMM3D's about ρ = 4 (CH6.1a; S4
  §3.2: FMM3D's counts miss the stated error at the corner by up to 83× at iprec 1); FMM3D's
  degrees are larger than ours there (23 against 18), so the 30–100× points at its counts.
  It is evidence for CH6.2e that FMM3D's counts are too few in the FMM at ε = 10⁻³ with
  iprec 1 on these levels. Our rule runs iprec 2 at every ε ≥ 10⁻⁶ (CH6.1e), so whether fewer
  waves than ours would hold at iprec 2 is still CH6.2e's to measure.
- **At 10⁻⁶ (iprec 2)** FMM3D's worst level gives 7.8 × 10⁻⁷ (the W4 cube at kD = 100,
  level 3), within ε; ours 9.7 × 10⁻⁸ to 3.0 × 10⁻⁷ there (1.006×).
- Rows with FMM3D's ratio 1 (W1 at kD = 30, level 3 at 10⁻⁶; the deepest levels of the W4
  cases): that level's plane waves change nothing in FMM3D's tree there (its shallower trees
  leave it little or no plane-wave work).

## 5. Tests and checks

- **The patch changes no value**: `tools/helmholtz/ch6-2d-runs.sh check`, on W1 at kD = 30
  and the W4 cube at kD = 100 at 10⁻³ and 10⁻⁶ with FULL = 1, `timed` against `default` and
  `fastker-timed` against `fastker`: every full output byte for byte the same, on Kathleen
  (job 630125), locust and the M6.
- **The switch**: `FMM3D_PW_LEVEL=-1` gives `mode=2` on every level that was `mode=1` and its
  errors are far below ε (§4); a level l alone gives `mode=1` on l only (the M6's smoke run,
  W1 at kD = 30: levels 2 and 3 each alone).
- **The timers**: the per-level parts sum to at most 0.999 of the call's step-3 time
  (`timeinfo(3)`) over the 110 calls of Kathleen's `times` launches a; each patched call of
  the check took 1.00–1.09× its unpatched call's time (single calls; the 1.09 a 60 ms call).
- **Our passes**: `fmm-exec/tests/helmholtz_fmm.rs`, `check_plane_wave_times` in
  `check_plane_wave` (both scenarios; one entry per level, both passes positive and within
  the level's M2L time on every level with waves and V pairs, zero elsewhere, on every rank
  or none), the existing bit-for-bit checks unchanged; `fmm-validate/tests/helmholtz_accuracy.rs`,
  `check_levels` on the exponential-M2L rows too.
- **The tables**: `tools/helmholtz/ch6-2b-tables.py` and `ch6-2d-tables.py` reproduce
  CH6.2b's tables byte for byte from its outputs (the new columns are read past the first
  twelve).

## 6. Runs, records and defects

- **Kathleen**: job 630125 (node-c11a-109; the four builds, `check`, `times` on 15 cases ×
  2 variants × 2 launches, `scan` on the 13 matrix cases; 147 launches, every one exit 0,
  2,302 s); job 630164 (node-c11a-111; ours, `matrix large` at 1 × 40, 16 launches, exit 0,
  43 min). The a/b spread of FMM3D's calls: 1.00–1.09.
- **locust, pending** (the user, 2026-11-07: "Wait for gridcl"): another session's
  `gridcl_geometry` tests held one of the 72 cores from 00:12 on, and a new one started as
  soon as the last ended. FMM3D's OpenMP loops wait for their slowest thread, so a shared
  core can slow a whole parallel loop. The runs made beside it are records only:
  FMM3D's `times` of 00:07–00:22 (source 3f683b5; the large cases after 00:12), its repeat
  and our runs with the pass timers of 00:43–01:10. A chain on locust repeats both once no
  gridcl test has run for 10 minutes and the load is below 2
  (`/data/ucahtbe/logs/ch6-2d-chain2.sh`); until then no locust number is used here. The
  records agree with Kathleen in kind: our level 2 at 1 × 72 took 61 ms (waves 2 ms, merge
  and shift 59 ms; CH6.2b's uncontended run 61 ms), FMM3D's parents' loop 26 ms with 7.9
  threads busy.
- **The M6**: the four builds, `check` (byte for byte), and smoke runs of `scan` (W1 at
  kD = 10, the W4 cube at kD = 100) and `times` (W1 at kD = 30) at 4 threads; nothing timed
  for the report (the user runs other jobs there).
- **Defect found and fixed**: the patch prints a call's `hfmm3d-steps:` line before its
  `hfmm3d-level:` lines, which the first version of the tables script read the other way:
  the scan (one call per ε) took each ε's levels from the call before, and `times` the
  first of the two repeats. Fixed (6d5f863) before any table here was made.

## 7. Reproducing

```sh
# FMM3D's four builds, on each machine (Kathleen: as a job)
FMM3D_ROOT=... VARIANTS="default fastker timed fastker-timed" tools/fmm3d/build.sh build
cargo build --release -p nd-fmm-validate --example helmholtz_fmm3d --example helmholtz_accuracy

# Kathleen
tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch \
    'VARIANTS="default fastker timed fastker-timed" tools/fmm3d/build.sh build' \
    'cargo build --release -p nd-fmm-validate --example helmholtz_fmm3d' \
    'THREADS=40 tools/helmholtz/ch6-2d-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2d check times scan'
tools/kathleen/submit.sh --time=02:00:00 -- run.sbatch \
    'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
    'THREADS=40 tools/helmholtz/ch6-2b-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2d-ours matrix large'

# locust (quiet: no other job)
THREADS=72 tools/helmholtz/ch6-2d-runs.sh /data/ucahtbe/logs/ch6-2d-r3 check times
THREADS=72 SPLITS=1x72 tools/helmholtz/ch6-2b-runs.sh /data/ucahtbe/logs/ch6-2d-ours-r3 matrix large

# The tables (the outputs fetched to bench-results/ch6-2d/, CH6.1e's to bench-results/ch6-1e/)
tools/helmholtz/ch6-2d-tables.py times kathleen=bench-results/ch6-2d/kathleen,bench-results/ch6-2d/kathleen-ours
tools/helmholtz/ch6-2d-tables.py scan bench-results/ch6-2d/kathleen bench-results/ch6-1e
```

The raw outputs are archived in `~/projects/fmm-archive/kathleen/logs/ch6-2d/` and
`~/projects/fmm-archive/locust/logs/ch6-2d-records/` (the records of §6; the Mac), not committed.

## Appendix: per level, both on plane waves (Kathleen)

FMM3D `timed` at 40 threads; ours one rank × 40 threads, the exponential M2L's rule, with its
two passes. "FMM3D ÷ ours" is FMM3D's M2L over ours.

| machine | eps | case | our level | Re κ | FMM3D boxes | FMM3D M2L s | ours s | ours waves + merge and shift s | FMM3D ÷ ours |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| kathleen | 1e-3 | w1-10 | 2 | 2.5 | 64 | 0.0014 | 0.0113 | 0.0011 + 0.0102 | 0.12 |
| kathleen | 1e-3 | w1-30 | 2 | 7.5 | 64 | 0.0048 | 0.0232 | 0.0019 + 0.0213 | 0.21 |
| kathleen | 1e-3 | w1-30 | 3 | 3.75 | 384 | 0.0073 | 0.0206 | 0.0033 + 0.0172 | 0.35 |
| kathleen | 1e-3 | w1-100 | 2 | 25 | 64 | 0.0316 | 0.1371 | 0.0075 + 0.1296 | 0.23 |
| kathleen | 1e-3 | w1-100 | 3 | 12.5 | 512 | 0.0222 | 0.0780 | 0.0121 + 0.0658 | 0.29 |
| kathleen | 1e-3 | w1-100 | 4 | 6.25 | 3200 | 0.0423 | 0.1017 | 0.0288 + 0.0724 | 0.42 |
| kathleen | 1e-3 | w1-100 | 5 | 3.12 | 4824 | 0.0563 | 0.0746 | 0.0232 + 0.0511 | 0.75 |
| kathleen | 1e-3 | w1-500 | 4 | 31.2 | 3840 | 0.7975 | 1.4104 | 0.4770 + 0.9325 | 0.57 |
| kathleen | 1e-3 | w1-500 | 5 | 15.6 | 19968 | 0.8115 | 1.8874 | 0.6488 + 1.2333 | 0.43 |
| kathleen | 1e-3 | w1-500 | 6 | 7.81 | 65920 | 1.4536 | 2.6307 | 0.8681 + 1.7524 | 0.55 |
| kathleen | 1e-3 | w1-500 | 7 | 3.91 | 94368 | 1.9634 | 2.8187 | 0.9335 + 1.8658 | 0.70 |
| kathleen | 1e-3 | w3-10 | 2 | 2.5 | 64 | 0.0014 | 0.0114 | 0.0012 + 0.0101 | 0.12 |
| kathleen | 1e-3 | w3-30 | 2 | 7.5 | 64 | 0.0056 | 0.0233 | 0.0019 + 0.0214 | 0.24 |
| kathleen | 1e-3 | w3-30 | 3 | 3.75 | 512 | 0.0127 | 0.0250 | 0.0038 + 0.0211 | 0.51 |
| kathleen | 1e-3 | w3-30 | 4 | 1.87 | 4096 | 0.0470 | 0.0739 | 0.0182 + 0.0550 | 0.64 |
| kathleen | 1e-3 | w3-100 | 2 | 25 | 64 | 0.0329 | 0.1372 | 0.0072 + 0.1299 | 0.24 |
| kathleen | 1e-3 | w3-100 | 3 | 12.5 | 512 | 0.0316 | 0.0799 | 0.0129 + 0.0669 | 0.40 |
| kathleen | 1e-3 | w3-100 | 4 | 6.25 | 4096 | 0.0696 | 0.1373 | 0.0397 + 0.0968 | 0.51 |
| kathleen | 1e-3 | w3-100 | 5 | 3.12 | 32768 | 0.3306 | 0.5237 | 0.1542 + 0.3640 | 0.63 |
| kathleen | 1e-3 | w3-100 | 6 | 1.56 | 262136 | 1.9782 | 3.8891 | 1.1613 + 2.6807 | 0.51 |
| kathleen | 1e-3 | w4-clusters-1 | 3 | 0.157 | 64 | 0.0008 | 0.0113 | 0.0019 + 0.0094 | 0.07 |
| kathleen | 1e-3 | w4-clusters-1 | 4 | 0.0784 | 288 | 0.0019 | 0.0127 | 0.0030 + 0.0096 | 0.15 |
| kathleen | 1e-3 | w4-clusters-1 | 5 | 0.0392 | 600 | 0.0056 | 0.0187 | 0.0050 + 0.0136 | 0.30 |
| kathleen | 1e-3 | w4-clusters-1 | 6 | 0.0196 | 1056 | 0.0134 | 0.0335 | 0.0085 + 0.0247 | 0.40 |
| kathleen | 1e-3 | w4-clusters-1 | 7 | 0.00979 | 1928 | 0.0240 | 0.0393 | 0.0102 + 0.0287 | 0.61 |
| kathleen | 1e-3 | w4-clusters-1 | 8 | 0.0049 | 2328 | 0.0239 | 0.0475 | 0.0123 + 0.0348 | 0.50 |
| kathleen | 1e-3 | w4-clusters-100 | 3 | 15.7 | 64 | 0.0110 | 0.0281 | 0.0040 + 0.0240 | 0.39 |
| kathleen | 1e-3 | w4-clusters-100 | 4 | 7.84 | 288 | 0.0090 | 0.0194 | 0.0043 + 0.0151 | 0.46 |
| kathleen | 1e-3 | w4-clusters-100 | 5 | 3.92 | 600 | 0.0138 | 0.0217 | 0.0057 + 0.0159 | 0.63 |
| kathleen | 1e-3 | w4-clusters-100 | 6 | 1.96 | 1056 | 0.0201 | 0.0360 | 0.0091 + 0.0266 | 0.56 |
| kathleen | 1e-3 | w4-clusters-100 | 7 | 0.979 | 1928 | 0.0256 | 0.0429 | 0.0111 + 0.0314 | 0.60 |
| kathleen | 1e-3 | w4-clusters-100 | 8 | 0.49 | 2328 | 0.0238 | 0.0487 | 0.0123 + 0.0359 | 0.49 |
| kathleen | 1e-3 | w4-cube-1 | 2 | 0.25 | 64 | 0.0010 | 0.0104 | 0.0014 + 0.0090 | 0.10 |
| kathleen | 1e-3 | w4-cube-1 | 3 | 0.125 | 512 | 0.0033 | 0.0168 | 0.0025 + 0.0142 | 0.20 |
| kathleen | 1e-3 | w4-cube-1 | 4 | 0.0625 | 4096 | 0.0238 | 0.0660 | 0.0163 + 0.0491 | 0.36 |
| kathleen | 1e-3 | w4-cube-100 | 2 | 25 | 64 | 0.0329 | 0.1361 | 0.0072 + 0.1289 | 0.24 |
| kathleen | 1e-3 | w4-cube-100 | 3 | 12.5 | 512 | 0.0277 | 0.0817 | 0.0129 + 0.0687 | 0.34 |
| kathleen | 1e-3 | w4-cube-100 | 4 | 6.25 | 4096 | 0.0697 | 0.1379 | 0.0393 + 0.0980 | 0.51 |
| kathleen | 1e-3 | w4-plummer-1 | 2 | 0.246 | 64 | 0.0009 | 0.0105 | 0.0014 + 0.0091 | 0.09 |
| kathleen | 1e-3 | w4-plummer-1 | 3 | 0.123 | 512 | 0.0031 | 0.0169 | 0.0025 + 0.0143 | 0.18 |
| kathleen | 1e-3 | w4-plummer-1 | 4 | 0.0615 | 512 | 0.0049 | 0.0169 | 0.0025 + 0.0143 | 0.29 |
| kathleen | 1e-3 | w4-plummer-1 | 5 | 0.0307 | 744 | 0.0065 | 0.0211 | 0.0038 + 0.0172 | 0.31 |
| kathleen | 1e-3 | w4-plummer-1 | 6 | 0.0154 | 1368 | 0.0114 | 0.0265 | 0.0063 + 0.0199 | 0.43 |
| kathleen | 1e-3 | w4-plummer-1 | 7 | 0.00769 | 1840 | 0.0199 | 0.0398 | 0.0097 + 0.0298 | 0.50 |
| kathleen | 1e-3 | w4-plummer-1 | 8 | 0.00384 | 96 | 0.0026 | 0.0160 | 0.0022 + 0.0138 | 0.16 |
| kathleen | 1e-3 | w4-plummer-100 | 2 | 24.6 | 64 | 0.0340 | 0.1340 | 0.0071 + 0.1269 | 0.25 |
| kathleen | 1e-3 | w4-plummer-100 | 3 | 12.3 | 512 | 0.0284 | 0.0778 | 0.0124 + 0.0653 | 0.36 |
| kathleen | 1e-3 | w4-plummer-100 | 4 | 6.15 | 512 | 0.0142 | 0.0912 | 0.0258 + 0.0650 | 0.16 |
| kathleen | 1e-3 | w4-plummer-100 | 5 | 3.07 | 744 | 0.0096 | 0.0256 | 0.0048 + 0.0207 | 0.37 |
| kathleen | 1e-3 | w4-plummer-100 | 6 | 1.54 | 1368 | 0.0126 | 0.0304 | 0.0070 + 0.0231 | 0.41 |
| kathleen | 1e-3 | w4-plummer-100 | 7 | 0.769 | 1840 | 0.0213 | 0.0427 | 0.0100 + 0.0322 | 0.50 |
| kathleen | 1e-3 | w4-plummer-100 | 8 | 0.384 | 96 | 0.0026 | 0.0171 | 0.0024 + 0.0145 | 0.15 |
| kathleen | 1e-3 | w4-sphere-1 | 2 | 0.25 | 64 | 0.0009 | 0.0102 | 0.0012 + 0.0089 | 0.09 |
| kathleen | 1e-3 | w4-sphere-1 | 3 | 0.125 | 512 | 0.0025 | 0.0169 | 0.0026 + 0.0143 | 0.15 |
| kathleen | 1e-3 | w4-sphere-1 | 4 | 0.0625 | 3200 | 0.0152 | 0.0503 | 0.0135 + 0.0365 | 0.30 |
| kathleen | 1e-3 | w4-sphere-1 | 5 | 0.0312 | 5016 | 0.0379 | 0.0674 | 0.0209 + 0.0459 | 0.56 |
| kathleen | 1e-3 | w4-sphere-100 | 2 | 25 | 64 | 0.0333 | 0.1355 | 0.0073 + 0.1282 | 0.25 |
| kathleen | 1e-3 | w4-sphere-100 | 3 | 12.5 | 512 | 0.0223 | 0.0810 | 0.0144 + 0.0665 | 0.28 |
| kathleen | 1e-3 | w4-sphere-100 | 4 | 6.25 | 3200 | 0.0437 | 0.1027 | 0.0299 + 0.0722 | 0.43 |
| kathleen | 1e-3 | w4-sphere-100 | 5 | 3.12 | 5016 | 0.0580 | 0.0843 | 0.0257 + 0.0581 | 0.69 |
| kathleen | 1e-6 | w1-30 | 2 | 7.5 | 64 | 0.0117 | 0.0302 | 0.0022 + 0.0279 | 0.39 |
| kathleen | 1e-6 | w1-100 | 2 | 25 | 64 | 0.0549 | 0.1613 | 0.0125 + 0.1487 | 0.34 |
| kathleen | 1e-6 | w1-100 | 3 | 12.5 | 512 | 0.0488 | 0.0951 | 0.0156 + 0.0794 | 0.51 |
| kathleen | 1e-6 | w1-100 | 4 | 6.25 | 1600 | 0.0928 | 0.1320 | 0.0417 + 0.0898 | 0.70 |
| kathleen | 1e-6 | w1-500 | 4 | 31.2 | 3840 | 1.2931 | 1.5773 | 0.5515 + 1.0249 | 0.82 |
| kathleen | 1e-6 | w1-500 | 5 | 15.6 | 16624 | 1.7153 | 2.1971 | 0.7900 + 1.4033 | 0.78 |
| kathleen | 1e-6 | w1-500 | 6 | 7.81 | 34432 | 2.6374 | 3.1422 | 1.0602 + 2.0705 | 0.84 |
| kathleen | 1e-6 | w1-500 | 7 | 3.91 | 3264 | 0.2859 | 4.7607 | 1.7010 + 3.0409 | 0.06 |
| kathleen | 1e-6 | w3-10 | 2 | 2.5 | 64 | 0.0061 | 0.0201 | 0.0022 + 0.0180 | 0.30 |
| kathleen | 1e-6 | w3-30 | 2 | 7.5 | 64 | 0.0134 | 0.0297 | 0.0025 + 0.0271 | 0.45 |
| kathleen | 1e-6 | w3-30 | 3 | 3.75 | 512 | 0.0291 | 0.0400 | 0.0062 + 0.0336 | 0.73 |
| kathleen | 1e-6 | w3-30 | 4 | 1.87 | 3224 | 0.1162 | 0.1218 | 0.0363 + 0.0848 | 0.95 |
| kathleen | 1e-6 | w3-100 | 2 | 25 | 64 | 0.0508 | 0.1632 | 0.0133 + 0.1496 | 0.31 |
| kathleen | 1e-6 | w3-100 | 3 | 12.5 | 512 | 0.0525 | 0.0958 | 0.0164 + 0.0793 | 0.55 |
| kathleen | 1e-6 | w3-100 | 4 | 6.25 | 4096 | 0.2260 | 0.1770 | 0.0521 + 0.1241 | 1.28 |
| kathleen | 1e-6 | w3-100 | 5 | 3.12 | 32768 | 1.2557 | 0.9178 | 0.3115 + 0.5972 | 1.37 |
| kathleen | 1e-6 | w4-clusters-1 | 3 | 0.157 | 64 | 0.0043 | 0.0207 | 0.0036 + 0.0170 | 0.21 |
| kathleen | 1e-6 | w4-clusters-1 | 4 | 0.0784 | 288 | 0.0092 | 0.0227 | 0.0054 + 0.0172 | 0.41 |
| kathleen | 1e-6 | w4-clusters-1 | 5 | 0.0392 | 600 | 0.0249 | 0.0343 | 0.0102 + 0.0240 | 0.73 |
| kathleen | 1e-6 | w4-clusters-1 | 6 | 0.0196 | 1024 | 0.0563 | 0.0562 | 0.0167 + 0.0391 | 1.00 |
| kathleen | 1e-6 | w4-clusters-1 | 7 | 0.00979 | 1328 | 0.0796 | 0.0691 | 0.0210 + 0.0477 | 1.15 |
| kathleen | 1e-6 | w4-clusters-1 | 8 | 0.0049 | 1296 | 0.0533 | 0.0822 | 0.0242 + 0.0575 | 0.65 |
| kathleen | 1e-6 | w4-clusters-100 | 3 | 15.7 | 64 | 0.0215 | 0.0329 | 0.0048 + 0.0280 | 0.65 |
| kathleen | 1e-6 | w4-clusters-100 | 4 | 7.84 | 288 | 0.0226 | 0.0316 | 0.0074 + 0.0241 | 0.72 |
| kathleen | 1e-6 | w4-clusters-100 | 5 | 3.92 | 600 | 0.0419 | 0.0376 | 0.0109 + 0.0266 | 1.11 |
| kathleen | 1e-6 | w4-clusters-100 | 6 | 1.96 | 1024 | 0.0682 | 0.0608 | 0.0182 + 0.0423 | 1.12 |
| kathleen | 1e-6 | w4-clusters-100 | 7 | 0.979 | 1328 | 0.0886 | 0.0748 | 0.0226 + 0.0518 | 1.19 |
| kathleen | 1e-6 | w4-clusters-100 | 8 | 0.49 | 1296 | 0.0575 | 0.0839 | 0.0249 + 0.0587 | 0.68 |
| kathleen | 1e-6 | w4-cube-1 | 2 | 0.25 | 64 | 0.0048 | 0.0190 | 0.0015 + 0.0175 | 0.25 |
| kathleen | 1e-6 | w4-cube-1 | 3 | 0.125 | 512 | 0.0170 | 0.0311 | 0.0046 + 0.0265 | 0.55 |
| kathleen | 1e-6 | w4-cube-1 | 4 | 0.0625 | 1496 | 0.0435 | 0.1180 | 0.0342 + 0.0832 | 0.37 |
| kathleen | 1e-6 | w4-cube-100 | 2 | 25 | 64 | 0.0511 | 0.1567 | 0.0081 + 0.1485 | 0.33 |
| kathleen | 1e-6 | w4-cube-100 | 3 | 12.5 | 512 | 0.0551 | 0.0937 | 0.0153 + 0.0783 | 0.59 |
| kathleen | 1e-6 | w4-cube-100 | 4 | 6.25 | 1496 | 0.0794 | 0.1729 | 0.0522 + 0.1201 | 0.46 |
| kathleen | 1e-6 | w4-plummer-1 | 2 | 0.246 | 64 | 0.0071 | 0.0174 | 0.0015 + 0.0159 | 0.41 |
| kathleen | 1e-6 | w4-plummer-1 | 3 | 0.123 | 504 | 0.0152 | 0.0299 | 0.0048 + 0.0250 | 0.51 |
| kathleen | 1e-6 | w4-plummer-1 | 4 | 0.0615 | 504 | 0.0223 | 0.0301 | 0.0048 + 0.0253 | 0.74 |
| kathleen | 1e-6 | w4-plummer-1 | 5 | 0.0307 | 504 | 0.0232 | 0.0370 | 0.0077 + 0.0291 | 0.63 |
| kathleen | 1e-6 | w4-plummer-1 | 6 | 0.0154 | 688 | 0.0279 | 0.0462 | 0.0127 + 0.0333 | 0.60 |
| kathleen | 1e-6 | w4-plummer-1 | 7 | 0.00769 | 600 | 0.0307 | 0.0682 | 0.0186 + 0.0494 | 0.45 |
| kathleen | 1e-6 | w4-plummer-100 | 2 | 24.6 | 64 | 0.0508 | 0.1577 | 0.0106 + 0.1471 | 0.32 |
| kathleen | 1e-6 | w4-plummer-100 | 3 | 12.3 | 504 | 0.0525 | 0.0902 | 0.0149 + 0.0752 | 0.58 |
| kathleen | 1e-6 | w4-plummer-100 | 4 | 6.15 | 504 | 0.0427 | 0.1173 | 0.0357 + 0.0811 | 0.36 |
| kathleen | 1e-6 | w4-plummer-100 | 5 | 3.07 | 504 | 0.0310 | 0.0458 | 0.0097 + 0.0359 | 0.68 |
| kathleen | 1e-6 | w4-plummer-100 | 6 | 1.54 | 688 | 0.0309 | 0.0510 | 0.0135 + 0.0371 | 0.61 |
| kathleen | 1e-6 | w4-plummer-100 | 7 | 0.769 | 600 | 0.0331 | 0.0747 | 0.0200 + 0.0541 | 0.44 |
| kathleen | 1e-6 | w4-sphere-1 | 2 | 0.25 | 64 | 0.0051 | 0.0175 | 0.0017 + 0.0158 | 0.29 |
| kathleen | 1e-6 | w4-sphere-1 | 3 | 0.125 | 512 | 0.0138 | 0.0300 | 0.0047 + 0.0253 | 0.46 |
| kathleen | 1e-6 | w4-sphere-1 | 4 | 0.0625 | 1600 | 0.0481 | 0.0873 | 0.0269 + 0.0602 | 0.55 |
| kathleen | 1e-6 | w4-sphere-100 | 2 | 25 | 64 | 0.0500 | 0.1571 | 0.0081 + 0.1489 | 0.32 |
| kathleen | 1e-6 | w4-sphere-100 | 3 | 12.5 | 512 | 0.0486 | 0.0932 | 0.0154 + 0.0778 | 0.52 |
| kathleen | 1e-6 | w4-sphere-100 | 4 | 6.25 | 1600 | 0.0906 | 0.1312 | 0.0405 + 0.0903 | 0.69 |
