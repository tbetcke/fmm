# Helmholtz CH6.1e: the quadrature per level, measured

Helmholtz task CH6.1e ([docs/helmholtz/tasks/CH6.1e.md](../../docs/helmholtz/tasks/CH6.1e.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.1 ("errors within about 1.1× Rotation's"), §5.3, §2 (constraint 9); CH6.1c's
degree study ([docs/helmholtz/tasks/CH6.1c.md](../../docs/helmholtz/tasks/CH6.1c.md),
"Results"). Measured on 2026-11-03 on the M6 (Mac mini, Apple M6, 24 GB;
[tools/m6/machine.md](../../tools/m6/machine.md)) from branch `tbetcke/h-ch6-1e` (the scan
on the harness of d443ac9, the verification on the rule of c926149), release build, rustc
1.99.0, outside the sandbox, BLAS threads 1, `HWLOC_COMPONENTS=-darwin`, one rank × 6
threads unless stated, the P2P kernel `Auto` (NEON). The machine ran other jobs throughout
(load average 7 to 27), so the times are indicative only; the errors are deterministic and
nothing here is timed for a gate.

## Summary

- **The rule.** Under `HelmholtzM2l::PlaneWave` a level whose κ_l = k w_l lies in the
  quadrature's tables takes **iprec 2 at the FMM's tolerance ε ≥ 10⁻⁶ and Rotation below**
  (`nd_fmm_tables::helmholtz::plane_wave::merge::precision_for_tolerance`,
  `PLANE_WAVE_MIN_TOLERANCE`). With named degrees and no tolerance it takes iprec 2 to
  p = 15, the calibrated plateau at 10⁻⁶ (`precision_for_degree`, `MAX_PLANE_WAVE_DEGREE`),
  and Rotation above. The rule reads ε, k and the level's width; not its degree where a
  tolerance is given, and never the charges (constraint 9).
- **How it was set.** On CH3.1c's matrix at the calibrated degrees (CH3.3), plus 27 cases at
  ε = 10⁻⁴, 10⁻⁵ and 10⁻⁷, each level with a quadrature was forced to iprec 1, 2 and 3 in
  turn, the other levels Rotation: 347 level × ε points, 1,040 forced runs. The FMM's relative ℓ2
  errors of φ and ∇φ against the sampled direct sum were compared with Rotation's on the
  same case. The precision a level needs follows ε, not its degree or κ_l:
  - iprec 1 misses 1.1× already at ε = 10⁻³ (up to 1.145× on one level alone, the W4 cube
    at kD = 100), and is far off below (3.6× at 10⁻⁴);
  - iprec 2 stays within 1.044× on every level from 10⁻³ to 10⁻⁶ (the worst the Plummer
    sphere's level 6 at kD = 1, κ ≈ 0.016) and misses from 10⁻⁷ (up to 5.4×, 25 of 32
    levels above 1.1×; 77× at 10⁻⁸);
  - iprec 3 holds on all but three levels at 10⁻⁷ and 10⁻⁸, but not on the deep levels of
    the clustered trees: 1.116× at 10⁻⁷ (the clusters' level 7 at kD = 1, κ ≈ 0.008), 1.30×
    and 1.76× at 10⁻⁸ (the Plummer sphere's level 7 at kD = 1 and the clusters' level 7 at
    kD = 10, κ ≈ 0.01 and 0.08), each on one level alone. At p = 20 and 24 iprec 3's waves
    of a degree-p multipole err on its truncated tail (CH6.1c's degree study: the
    resolving degree of iprec 3 is about 60). On the same trees other levels move the
    error by up to 30 % in the other direction (0.68×), so a guard by κ would be fitted to
    noise (§3).
- **The acceptance holds.** Under the rule the FMM is within **1.070×** Rotation's error
  on every case of CH3.1c's matrix measured and on the 27 in-between cases: 87 cases in
  all, the worst φ on the Plummer sphere at kD = 1 and ε = 10⁻⁶ (§4). The two cases left
  out are W3 at kD = 100 and W1 at kD = 500 at ε = 10⁻⁸, where the rule takes Rotation
  on every level (bit for bit Rotation's level calls). The tests assert 1.1× on their
  scenarios (§6).
- **The levels left to Rotation** (§5): every level of every case at ε < 10⁻⁶ (in D1's
  range 10⁻⁷ and 10⁻⁸), and at any ε the levels whose κ_l lies outside the tables
  (Re κ_l ≥ 16π: W1 at kD = 500, levels 2 and 3). At ε ≥ 10⁻⁶ no level with a quadrature
  is left out. H6's gain is therefore confined to ε ≥ 10⁻⁶, which the H3 gate's target,
  the low-frequency trees at tight ε (H3 gate report §3.3), does not reach at ε = 10⁻⁸. As
  the roadmap's risk said: CH6.1c's study put the limit at p ≤ 16, and the measured one is
  p = 15 at 10⁻⁶.
- **Faster, for orientation.** At ε = 10⁻⁶ the FMM under the rule evaluated in 0.25 to
  0.65 of Rotation's time on the trees of 10⁵ points and more (W3 at kD = 100: 21.9 s
  against 88.3 s; W1 at kD = 500: 35.9 s against 88.7 s), and in 0.5 to 1.2 of it at 10⁻³;
  slower on the smallest trees. The timings are CH6.2's, on a quiet machine.
- **Memory: a finding for CH6.2.** The level call's waves buffer holds one frame of every
  box of the largest level with waves (4W reals per box, as Laplace's), which at Helmholtz's
  W is large: under the rule 11.8 GB on W3 at kD = 100 and 9.0 GB on W1 at kD = 500, and
  22.9 GB for W3's level 6 forced to iprec 3, which swapped on the M6 (334 s against 16 s)
  and ran the machine out of memory in the next run (§7). The follow-up is in the
  roadmap's CH6.2 line.
- **Deterministic.** The rule's output has the same checksum at 1 × 6, 2 × 3 and 3 × 2
  ranks × threads (W1 at kD = 30 and 100, ε = 10⁻³ and 10⁻⁶, §6).

## 1. Method

The harness is CH3.1c's `nd_fmm_validate::helmholtz_accuracy` (W1, W3 and W4 of proposal
§9.1, complex charges with parts uniform in [−1, 1), 1,000 sampled targets, the oracle
`nd_fmm_ref::helmholtz::p2p::direct_sum` in f64 times 1/(4π), §9.1's relative ℓ2 norms of φ
and of the ∇φ field), with three new `--m2l` entries in its example:

- `planewave-scan`: Rotation, then each level l ≥ 2 whose κ_l = k w_l lies in the tables
  at iprec 1, 2 and 3 alone (`PlaneWavePrecision::PerLevel` with that level named), the
  other levels Rotation;
- `planewave`: the rule (`PlaneWavePrecision::Rule`, the builder's tolerance);
- `planewave:<1|2|3>`: one precision on every level with a quadrature.

Each row's last column is its error of φ and of ∇φ over the case's Rotation row; a level's
**ratio** below is the larger of the two. The degrees are the calibrated rule's at ε
(`Degrees::Tolerance`), so the levels' degrees are those of CH3.3's verification. κ_l ≈
kD 2⁻ˡ in the tables (the FMM's domain is the points' bounding cube, within a few per cent
of the unit cube). `tools/helmholtz/ch6-1e-runs.sh` runs the stages (§8).

The cases: CH3.1c's matrix (W1 at kD = 10, 30, 100 and W3 at kD = 10, 30, each at
ε = 10⁻³, 10⁻⁶, 10⁻⁸; the four W4 distributions at N = 10⁵, kD = 1, 10, 100, the same ε; W1
at kD = 100 with real charges and with k = 100 + 10i at 10⁻³ and 10⁻⁶; eight vectors at
10⁻⁶ in the verification), W3 at kD = 100 and W1 at kD = 500 at 10⁻³ and 10⁻⁶ (the user's
choice for the M6: their 10⁻⁸ rows run Rotation under the rule; the scan of W3 at kD = 100
lacks level 6 at iprec 3 and ε = 10⁻⁶, and W1 at kD = 500 was verified but not scanned,
both for memory, §7), and between them ε =
10⁻⁴, 10⁻⁵ and 10⁻⁷ on W1 at kD = 10, 30, 100, W3 at kD = 10, 30, and the W4 cube and
clusters at kD = 1 and 100.

## 2. The scan

Per ε and precision, over every level with a quadrature of every scanned case: the
ratio's median and worst, where the worst is, and how many levels exceed 1.1. The full
table per level is the appendix.

| ε | precision | levels | median | worst | worst at | above 1.1 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 1e-3 | iprec 1 | 88 | 1.012 | 1.145 | W4 cube (N = 100000) kD 100, level 4 (κ ≈ 6.2) | 4 |
| 1e-3 | iprec 2 | 88 | 1.000 | 1.000 | W4 sphere (N = 100000) kD 100, level 5 (κ ≈ 3.1) | 0 |
| 1e-3 | iprec 3 | 88 | 1.000 | 1.000 | W4 sphere (N = 100000) kD 100, level 5 (κ ≈ 3.1) | 0 |
| 1e-4 | iprec 1 | 32 | 1.515 | 3.565 | W4 cube (N = 100000) kD 1, level 3 (κ ≈ 0.12) | 23 |
| 1e-4 | iprec 2 | 32 | 1.000 | 1.000 | W4 cube (N = 100000) kD 100, level 4 (κ ≈ 6.2) | 0 |
| 1e-4 | iprec 3 | 32 | 1.000 | 1.000 | W4 cube (N = 100000) kD 100, level 4 (κ ≈ 6.2) | 0 |
| 1e-5 | iprec 1 | 32 | 9.051 | 38.764 | W4 cube (N = 100000) kD 1, level 3 (κ ≈ 0.12) | 30 |
| 1e-5 | iprec 2 | 32 | 1.000 | 1.002 | W3 (N = 4032) kD 10, level 2 (κ ≈ 2.5) | 0 |
| 1e-5 | iprec 3 | 32 | 1.000 | 1.000 | W4 cube (N = 100000) kD 100, level 4 (κ ≈ 6.2) | 0 |
| 1e-6 | iprec 1 | 88 | 77.151 | 357.495 | W4 cube (N = 100000) kD 1, level 3 (κ ≈ 0.12) | 88 |
| 1e-6 | iprec 2 | 88 | 1.000 | 1.044 | W4 plummer (N = 100000) kD 1, level 6 (κ ≈ 0.016) | 0 |
| 1e-6 | iprec 3 | 87 | 1.000 | 1.002 | W4 plummer (N = 100000) kD 100, level 7 (κ ≈ 0.78) | 0 |
| 1e-7 | iprec 1 | 32 | 1063.592 | 10339.140 | W4 cube (N = 100000) kD 1, level 3 (κ ≈ 0.12) | 32 |
| 1e-7 | iprec 2 | 32 | 1.257 | 5.403 | W3 (N = 4032) kD 10, level 2 (κ ≈ 2.5) | 25 |
| 1e-7 | iprec 3 | 32 | 1.000 | 1.116 | W4 clusters (N = 100000) kD 1, level 7 (κ ≈ 0.0078) | 1 |
| 1e-8 | iprec 1 | 75 | 9311.942 | 88879.403 | W4 cube (N = 100000) kD 1, level 3 (κ ≈ 0.12) | 75 |
| 1e-8 | iprec 2 | 75 | 10.587 | 76.920 | W1 kD 10, level 2 (κ ≈ 2.5) | 68 |
| 1e-8 | iprec 3 | 75 | 1.000 | 1.760 | W4 clusters (N = 100000) kD 10, level 7 (κ ≈ 0.078) | 2 |

Reading the table by precision:

- **iprec 1** is the cheapest (the fewest waves) but misses 1.1× on 4 of 88 levels at
  10⁻³, all on the W4 cube (kD = 1 and 100: 1.108–1.145×), and on most levels below. Its stated error (about 5 × 10⁻⁵ relative to the kernel) is not far enough
  below the FMM's own error at 10⁻³ (0.02–0.15 ε). The rule never takes it.
- **iprec 2** keeps every level within 0.967–1.044× from 10⁻³ to 10⁻⁶, the median 1.000.
  Every level that moves by more than 1 % is at 10⁻⁶, most of them deep levels of the
  Plummer sphere and the clusters (1.044× and 1.020× on the Plummer sphere's levels 6 and 7
  at kD = 1, κ ≈ 0.02 and 0.01; 0.967× on the clusters' level 8). At 10⁻⁷ its stated error
  (about 10⁻⁷) is the FMM's own, and it misses on 25 of 32 levels.
- **iprec 3** is within 0.999–1.002× of Rotation from 10⁻³ to 10⁻⁶, a finer precision than
  those levels need. At 10⁻⁷ (p = 20) and 10⁻⁸ (p = 24) it holds on most levels, but it misses
  on the deepest levels of the trees with dense clusters of boxes (κ_l ≤ 0.08):

  | ε | case | level | κ_l | p | ratio | its neighbours |
  | --- | --- | ---: | ---: | ---: | ---: | --- |
  | 10⁻⁷ | W4 clusters, kD = 1 | 7 | 0.008 | 20 | 1.116 | 1.000 (6), 0.994 (8) |
  | 10⁻⁸ | W4 Plummer, kD = 1 | 7 | 0.008 | 24 | 1.303 | 1.043 (6), 1.000 (8) |
  | 10⁻⁸ | W4 clusters, kD = 10 | 7 | 0.078 | 24 | 1.760 | 0.998 (6), 0.817 (8) |
  | 10⁻⁸ | W4 sphere, kD = 1 | 5 | 0.031 | 24 | 1.086 | 0.995 (4) |
  | 10⁻⁸ | W4 clusters, kD = 100 | 8 | 0.39 | 24 | 0.680 | 0.983 (7) |

  These are single levels with the others Rotation. Several of them together add (§3).

## 3. The rule

Per CH6.1c's sign-off (question 3: per level the coarsest precision within about 1.1×
Rotation's, Rotation where none is), with a share for the levels together: a tree has up
to seven levels with waves, and their extra errors add. The coarsest precision within 1.1×
on every scanned level is:

| ε | coarsest within 1.1× on every level | worst single level |
| --- | --- | ---: |
| 10⁻³, 10⁻⁴, 10⁻⁵, 10⁻⁶ | iprec 2 | 1.044 |
| 10⁻⁷ | none (iprec 3: 1.116 on one level) | – |
| 10⁻⁸ | none (iprec 3: 1.760 on one level) | – |

so `precision_for_tolerance(ε)` is iprec 2 for ε ≥ 10⁻⁶ and none below; between 10⁻⁷
and 10⁻⁶ nothing was measured, and the rule takes the measured side. Neither the degree
nor κ_l is needed: at ε ≥ 10⁻⁶ iprec 2 holds at κ_l from 0.004 to 25 and p from 7 to 35.
Above ε = 10⁻³, outside D1, the rule takes iprec 2 unmeasured (iprec 1 might do there).

An FMM with named degrees (`Degrees::PerLevel`) has no tolerance. `precision_for_degree`
reads the tolerance a degree p meets on the calibrated plateau ⌈0.30 L² + 0.25 L + 2.60⌉:
p ≤ 15 is the plateau at 10⁻⁶, so iprec 2 there, none above. A high-frequency level of a
tree at 10⁻⁶ has p > 15 and so keeps Rotation without the tolerance (W1 at kD = 100: p = 35
and 22 on levels 2 and 3), where with it the rule gives iprec 2. The builder always passes
`Degrees::Tolerance`'s ε (`HelmholtzOperator::with_plane_wave_tolerance`), so only named
degrees take this path.

**Not taken: iprec 3 with a guard on κ_l.** The failures of iprec 3 lie at κ_l ≤ 0.08, and
every level at κ_l ≥ 0.16 kept it within 1.013× at 10⁻⁷ and 10⁻⁸. A rule "iprec 3 for
10⁻⁸ ≤ ε < 10⁻⁶ where κ_l ≥ 0.15, else Rotation" would fit the data. But the deviations on
the same trees are large in both directions (0.68× on the clusters' level 8 at kD = 100,
κ ≈ 0.39; 0.82× at κ ≈ 0.04), which says iprec 3's error at p = 20–24 is of the size of the
FMM's own there, and its sign depends on the charges. A guard set from five failing levels
would be fitted to them. It is question 2 of the PR.

## 4. Verification against Rotation

The rule (`--m2l rotation,planewave`) on every case: the errors under Rotation and under
the rule, their ratios, the levels with waves (`l:i2`; "none" where the rule takes none),
and the evaluation times (indicative). Where the rule takes none, the output is
Rotation's bit for bit (each level runs Rotation's level call).

| Workload | kD | ε | variant | p by level | rule's levels | Rotation φ | ∇φ | PlaneWave φ | ∇φ | ÷ Rotation φ / ∇φ | Rotation s | PlaneWave s |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| W1 | 10 | 1e-3 | – | 0×2, 7 | 2:i2 | 4.40e-5 | 1.53e-4 | 4.40e-5 | 1.53e-4 | 1.000/1.000 | 0.03 | 0.17 |
| W1 | 10 | 1e-6 | – | 0×2, 15 | 2:i2 | 5.62e-8 | 4.11e-7 | 5.65e-8 | 4.09e-7 | 1.005/0.997 | 0.06 | 0.15 |
| W1 | 10 | 1e-8 | – | 0×2, 24 | none | 7.81e-11 | 9.22e-10 | 7.81e-11 | 9.22e-10 | 1.000/1.000 | 0.06 | 0.10 |
| W1 | 30 | 1e-3 | – | 0×2, 12, 8 | 2:i2,3:i2 | 2.66e-5 | 7.15e-5 | 2.66e-5 | 7.15e-5 | 1.000/1.000 | 0.14 | 0.30 |
| W1 | 30 | 1e-6 | – | 0×2, 16, 15 | 2:i2,3:i2 | 1.12e-7 | 2.97e-7 | 1.12e-7 | 2.97e-7 | 1.001/0.998 | 0.18 | 0.19 |
| W1 | 30 | 1e-8 | – | 0×2, 24×2 | none | 2.62e-10 | 8.32e-10 | 2.62e-10 | 8.32e-10 | 1.000/1.000 | 0.39 | 0.39 |
| W1 | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7 | 2:i2,3:i2,4:i2,5:i2 | 4.15e-5 | 1.36e-4 | 4.15e-5 | 1.36e-4 | 1.000/1.000 | 1.30 | 1.29 |
| W1 | 100 | 1e-6 | – | 0×2, 35, 22, 15×2 | 2:i2,3:i2,4:i2,5:i2 | 9.61e-8 | 2.94e-7 | 9.56e-8 | 2.88e-7 | 0.995/0.979 | 2.67 | 1.74 |
| W1 | 100 | 1e-8 | – | 0×2, 38, 24×3 | none | 1.39e-9 | 3.40e-9 | 1.39e-9 | 3.40e-9 | 1.000/1.000 | 6.89 | 7.58 |
| W3 (N = 4032) | 10 | 1e-3 | – | 0×2, 7×2 | 2:i2,3:i2 | 4.05e-5 | 8.10e-5 | 4.05e-5 | 8.10e-5 | 1.000/1.000 | 0.08 | 0.24 |
| W3 (N = 4032) | 10 | 1e-6 | – | 0×2, 15×2 | 2:i2,3:i2 | 4.49e-8 | 1.70e-7 | 4.45e-8 | 1.61e-7 | 0.991/0.951 | 0.10 | 0.24 |
| W3 (N = 4032) | 10 | 1e-8 | – | 0×2, 24×2 | none | 9.83e-11 | 8.92e-10 | 9.83e-11 | 8.92e-10 | 1.000/1.000 | 0.27 | 0.22 |
| W3 (N = 108849) | 30 | 1e-3 | – | 0×2, 12, 8, 7 | 2:i2,3:i2,4:i2 | 4.60e-5 | 1.31e-4 | 4.60e-5 | 1.31e-4 | 1.000/1.000 | 0.45 | 0.52 |
| W3 (N = 108849) | 30 | 1e-6 | – | 0×2, 16, 15×2 | 2:i2,3:i2,4:i2 | 1.46e-7 | 4.17e-7 | 1.42e-7 | 4.02e-7 | 0.973/0.965 | 1.94 | 0.63 |
| W3 (N = 108849) | 30 | 1e-8 | – | 0×2, 24×3 | none | 7.34e-10 | 2.96e-9 | 7.34e-10 | 2.96e-9 | 1.000/1.000 | 5.63 | 5.81 |
| W4 cube (N = 100000) | 1 | 1e-3 | – | 0×2, 7×3 | 2:i2,3:i2,4:i2 | 4.07e-5 | 8.96e-5 | 4.07e-5 | 8.96e-5 | 1.000/1.000 | 0.54 | 0.47 |
| W4 cube (N = 100000) | 1 | 1e-6 | – | 0×2, 15×3 | 2:i2,3:i2,4:i2 | 6.40e-8 | 2.65e-7 | 6.29e-8 | 2.46e-7 | 0.982/0.927 | 1.61 | 0.59 |
| W4 cube (N = 100000) | 1 | 1e-8 | – | 0×2, 24×3 | none | 2.57e-10 | 1.73e-9 | 2.57e-10 | 1.73e-9 | 1.000/1.000 | 7.00 | 5.41 |
| W4 cube (N = 100000) | 10 | 1e-3 | – | 0×2, 7×3 | 2:i2,3:i2,4:i2 | 5.00e-5 | 9.95e-5 | 5.00e-5 | 9.95e-5 | 1.000/1.000 | 0.53 | 0.63 |
| W4 cube (N = 100000) | 10 | 1e-6 | – | 0×2, 15×3 | 2:i2,3:i2,4:i2 | 6.08e-8 | 2.69e-7 | 6.10e-8 | 2.50e-7 | 1.004/0.929 | 1.87 | 0.76 |
| W4 cube (N = 100000) | 10 | 1e-8 | – | 0×2, 24×3 | none | 2.29e-10 | 1.71e-9 | 2.29e-10 | 1.71e-9 | 1.000/1.000 | 5.69 | 5.47 |
| W4 cube (N = 100000) | 100 | 1e-3 | – | 0×2, 30, 18, 11 | 2:i2,3:i2,4:i2 | 1.85e-5 | 4.04e-5 | 1.85e-5 | 4.04e-5 | 1.000/1.000 | 1.24 | 0.85 |
| W4 cube (N = 100000) | 100 | 1e-6 | – | 0×2, 35, 22, 15 | 2:i2,3:i2,4:i2 | 9.63e-8 | 3.04e-7 | 9.66e-8 | 3.01e-7 | 1.003/0.989 | 2.46 | 1.04 |
| W4 cube (N = 100000) | 100 | 1e-8 | – | 0×2, 38, 24×2 | none | 2.11e-9 | 4.52e-9 | 2.11e-9 | 4.52e-9 | 1.000/1.000 | 5.46 | 7.51 |
| W4 sphere (N = 100000) | 1 | 1e-3 | – | 0×2, 7×4 | 2:i2,3:i2,4:i2,5:i2 | 2.30e-5 | 8.09e-7 | 2.30e-5 | 8.09e-7 | 1.000/1.000 | 0.77 | 0.75 |
| W4 sphere (N = 100000) | 1 | 1e-6 | – | 0×2, 15×4 | 2:i2,3:i2,4:i2,5:i2 | 4.24e-8 | 2.36e-9 | 4.18e-8 | 2.23e-9 | 0.986/0.946 | 2.85 | 1.30 |
| W4 sphere (N = 100000) | 1 | 1e-8 | – | 0×2, 24×4 | none | 1.31e-10 | 1.20e-11 | 1.31e-10 | 1.20e-11 | 1.000/1.000 | 7.29 | 7.56 |
| W4 sphere (N = 100000) | 10 | 1e-3 | – | 0×2, 7×4 | 2:i2,3:i2,4:i2,5:i2 | 2.85e-5 | 8.42e-7 | 2.85e-5 | 8.42e-7 | 1.000/1.000 | 0.70 | 0.64 |
| W4 sphere (N = 100000) | 10 | 1e-6 | – | 0×2, 15×4 | 2:i2,3:i2,4:i2,5:i2 | 4.36e-8 | 2.40e-9 | 4.34e-8 | 2.27e-9 | 0.994/0.946 | 2.47 | 1.31 |
| W4 sphere (N = 100000) | 10 | 1e-8 | – | 0×2, 24×4 | none | 1.27e-10 | 1.21e-11 | 1.27e-10 | 1.21e-11 | 1.000/1.000 | 7.91 | 9.07 |
| W4 sphere (N = 100000) | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7 | 2:i2,3:i2,4:i2,5:i2 | 3.06e-5 | 2.04e-6 | 3.06e-5 | 2.04e-6 | 1.000/1.000 | 1.24 | 1.14 |
| W4 sphere (N = 100000) | 100 | 1e-6 | – | 0×2, 35, 22, 15×2 | 2:i2,3:i2,4:i2,5:i2 | 6.24e-8 | 4.06e-9 | 6.26e-8 | 3.97e-9 | 1.002/0.978 | 3.00 | 1.73 |
| W4 sphere (N = 100000) | 100 | 1e-8 | – | 0×2, 38, 24×3 | none | 8.86e-10 | 4.86e-11 | 8.86e-10 | 4.86e-11 | 1.000/1.000 | 7.57 | 7.75 |
| W4 plummer (N = 100000) | 1 | 1e-3 | – | 0×2, 7×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 6.47e-5 | 9.67e-5 | 6.47e-5 | 9.67e-5 | 1.000/1.000 | 1.15 | 1.06 |
| W4 plummer (N = 100000) | 1 | 1e-6 | – | 0×2, 15×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 1.11e-7 | 3.33e-7 | 1.18e-7 | 3.13e-7 | 1.070/0.940 | 3.36 | 2.00 |
| W4 plummer (N = 100000) | 1 | 1e-8 | – | 0×2, 24×7 | none | 5.90e-10 | 2.41e-9 | 5.90e-10 | 2.41e-9 | 1.000/1.000 | 9.39 | 9.46 |
| W4 plummer (N = 100000) | 10 | 1e-3 | – | 0×2, 7×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 5.01e-5 | 9.69e-5 | 5.01e-5 | 9.69e-5 | 1.000/1.000 | 1.08 | 1.10 |
| W4 plummer (N = 100000) | 10 | 1e-6 | – | 0×2, 15×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 8.31e-8 | 3.33e-7 | 8.62e-8 | 3.15e-7 | 1.038/0.945 | 3.57 | 1.96 |
| W4 plummer (N = 100000) | 10 | 1e-8 | – | 0×2, 24×7 | none | 4.34e-10 | 2.41e-9 | 4.34e-10 | 2.41e-9 | 1.000/1.000 | 9.98 | 9.41 |
| W4 plummer (N = 100000) | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7×4 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 7.95e-5 | 1.18e-4 | 7.95e-5 | 1.18e-4 | 1.000/1.000 | 1.74 | 1.37 |
| W4 plummer (N = 100000) | 100 | 1e-6 | – | 0×2, 35, 21, 15×5 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 1.53e-7 | 3.58e-7 | 1.55e-7 | 3.42e-7 | 1.015/0.955 | 4.04 | 1.40 |
| W4 plummer (N = 100000) | 100 | 1e-8 | – | 0×2, 37, 24×6 | none | 1.27e-9 | 2.59e-9 | 1.27e-9 | 2.59e-9 | 1.000/1.000 | 10.83 | 8.50 |
| W4 clusters (N = 100000) | 1 | 1e-3 | – | 0×2, 7×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 5.07e-5 | 1.01e-4 | 5.07e-5 | 1.01e-4 | 1.000/1.000 | 0.61 | 0.49 |
| W4 clusters (N = 100000) | 1 | 1e-6 | – | 0×2, 15×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 9.45e-8 | 4.03e-7 | 9.16e-8 | 3.69e-7 | 0.970/0.916 | 2.68 | 1.18 |
| W4 clusters (N = 100000) | 1 | 1e-8 | – | 0×2, 24×7 | none | 5.37e-10 | 5.00e-9 | 5.37e-10 | 5.00e-9 | 1.000/1.000 | 8.18 | 9.60 |
| W4 clusters (N = 100000) | 10 | 1e-3 | – | 0×2, 7×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 5.05e-5 | 1.01e-4 | 5.05e-5 | 1.01e-4 | 1.000/1.000 | 0.76 | 0.70 |
| W4 clusters (N = 100000) | 10 | 1e-6 | – | 0×2, 15×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 9.42e-8 | 4.03e-7 | 8.95e-8 | 3.67e-7 | 0.951/0.910 | 3.28 | 1.66 |
| W4 clusters (N = 100000) | 10 | 1e-8 | – | 0×2, 24×7 | none | 5.35e-10 | 5.00e-9 | 5.35e-10 | 5.00e-9 | 1.000/1.000 | 10.05 | 10.45 |
| W4 clusters (N = 100000) | 100 | 1e-3 | – | 0×2, 21, 13, 8, 7×4 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 5.26e-5 | 1.03e-4 | 5.26e-5 | 1.03e-4 | 1.000/1.000 | 0.90 | 0.84 |
| W4 clusters (N = 100000) | 100 | 1e-6 | – | 0×2, 25, 16, 15×5 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 9.08e-8 | 4.03e-7 | 8.81e-8 | 3.71e-7 | 0.970/0.919 | 3.55 | 1.86 |
| W4 clusters (N = 100000) | 100 | 1e-8 | – | 0×2, 27, 24×6 | none | 5.01e-10 | 4.98e-9 | 5.01e-10 | 4.98e-9 | 1.000/1.000 | 9.82 | 8.75 |
| W1 | 100 | 1e-3 | real charges | 0×2, 30, 18, 11, 7 | 2:i2,3:i2,4:i2,5:i2 | 4.17e-5 | 1.34e-4 | 4.17e-5 | 1.34e-4 | 1.000/1.000 | 0.76 | 0.89 |
| W1 | 100 | 1e-6 | real charges | 0×2, 35, 22, 15×2 | 2:i2,3:i2,4:i2,5:i2 | 9.92e-8 | 3.00e-7 | 9.79e-8 | 2.93e-7 | 0.987/0.977 | 1.92 | 1.25 |
| W1 | 100 + 10i | 1e-3 | complex k | 0×2, 30, 18, 11, 7 | 2:i2,3:i2,4:i2,5:i2 | 5.58e-5 | 1.54e-4 | 5.58e-5 | 1.54e-4 | 1.000/1.000 | 0.96 | 0.76 |
| W1 | 100 + 10i | 1e-6 | complex k | 0×2, 35, 22, 15×2 | 2:i2,3:i2,4:i2,5:i2 | 1.53e-7 | 3.87e-7 | 1.53e-7 | 3.88e-7 | 1.003/1.002 | 2.54 | 1.21 |
| W1 | 100 | 1e-6 | 8 vectors | 0×2, 35, 22, 15×2 | 2:i2,3:i2,4:i2,5:i2 | 9.52e-8 | 2.96e-7 | 9.44e-8 | 2.90e-7 | 0.992/0.980 | 14.51 | 7.29 |
| W3 (N = 4031442) | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7×2 | 2:i2,3:i2,4:i2,5:i2,6:i2 | 4.86e-5 | 1.08e-4 | 4.86e-5 | 1.08e-4 | 1.000/1.000 | 13.86 | 12.30 |
| W3 (N = 4031442) | 100 | 1e-6 | – | 0×2, 35, 22, 15×3 | 2:i2,3:i2,4:i2,5:i2,6:i2 | 1.01e-7 | 2.50e-7 | 1.02e-7 | 2.46e-7 | 1.009/0.983 | 88.34 | 21.94 |
| W1 | 500 | 1e-3 | – | 0×2, 123, 66, 36, 21, 13, 8 | 4:i2,5:i2,6:i2,7:i2 | 3.24e-5 | 8.71e-5 | 3.24e-5 | 8.71e-5 | 1.000/1.000 | 51.40 | 27.92 |
| W1 | 500 | 1e-6 | – | 0×2, 131, 72, 41, 25, 16, 15 | 4:i2,5:i2,6:i2,7:i2 | 1.49e-7 | 3.79e-7 | 1.45e-7 | 3.73e-7 | 0.974/0.986 | 88.71 | 35.93 |
| W1 | 10 | 1e-4 | – | 0×2, 9 | 2:i2 | 6.64e-6 | 3.00e-5 | 6.64e-6 | 3.00e-5 | 1.000/1.000 | 0.01 | 0.00 |
| W1 | 10 | 1e-5 | – | 0×2, 12 | 2:i2 | 5.74e-7 | 3.28e-6 | 5.75e-7 | 3.28e-6 | 1.002/1.001 | 0.00 | 0.01 |
| W1 | 10 | 1e-7 | – | 0×2, 20 | none | 1.42e-9 | 1.22e-8 | 1.42e-9 | 1.22e-8 | 1.000/1.000 | 0.02 | 0.01 |
| W1 | 30 | 1e-4 | – | 0×2, 14, 9 | 2:i2,3:i2 | 7.47e-6 | 2.44e-5 | 7.47e-6 | 2.44e-5 | 1.000/1.000 | 0.04 | 0.05 |
| W1 | 30 | 1e-5 | – | 0×2, 15, 12 | 2:i2,3:i2 | 5.16e-7 | 1.74e-6 | 5.16e-7 | 1.75e-6 | 1.001/1.000 | 0.06 | 0.07 |
| W1 | 30 | 1e-7 | – | 0×2, 20×2 | none | 4.33e-9 | 1.21e-8 | 4.33e-9 | 1.21e-8 | 1.000/1.000 | 0.17 | 0.15 |
| W1 | 100 | 1e-4 | – | 0×2, 32, 19, 12, 9 | 2:i2,3:i2,4:i2,5:i2 | 4.58e-6 | 1.70e-5 | 4.59e-6 | 1.70e-5 | 1.000/1.000 | 0.98 | 0.73 |
| W1 | 100 | 1e-5 | – | 0×2, 34, 20, 13, 12 | 2:i2,3:i2,4:i2,5:i2 | 7.85e-7 | 2.29e-6 | 7.85e-7 | 2.29e-6 | 0.999/0.999 | 1.62 | 1.02 |
| W1 | 100 | 1e-7 | – | 0×2, 36, 23, 20×2 | none | 9.12e-9 | 1.78e-8 | 9.12e-9 | 1.78e-8 | 1.000/1.000 | 4.00 | 3.36 |
| W3 (N = 4032) | 10 | 1e-4 | – | 0×2, 9×2 | 2:i2,3:i2 | 5.01e-6 | 1.32e-5 | 5.01e-6 | 1.32e-5 | 1.000/1.000 | 0.02 | 0.03 |
| W3 (N = 4032) | 10 | 1e-5 | – | 0×2, 12×2 | 2:i2,3:i2 | 4.07e-7 | 1.40e-6 | 4.08e-7 | 1.40e-6 | 1.002/0.997 | 0.04 | 0.04 |
| W3 (N = 4032) | 10 | 1e-7 | – | 0×2, 20×2 | none | 1.19e-9 | 7.68e-9 | 1.19e-9 | 7.68e-9 | 1.000/1.000 | 0.10 | 0.09 |
| W3 (N = 108849) | 30 | 1e-4 | – | 0×2, 14, 9×2 | 2:i2,3:i2,4:i2 | 9.50e-6 | 2.93e-5 | 9.50e-6 | 2.93e-5 | 1.000/1.000 | 0.46 | 0.27 |
| W3 (N = 108849) | 30 | 1e-5 | – | 0×2, 15, 12×2 | 2:i2,3:i2,4:i2 | 7.36e-7 | 2.59e-6 | 7.36e-7 | 2.59e-6 | 1.001/0.997 | 0.84 | 0.33 |
| W3 (N = 108849) | 30 | 1e-7 | – | 0×2, 20×3 | none | 5.84e-9 | 2.18e-8 | 5.84e-9 | 2.18e-8 | 1.000/1.000 | 2.64 | 2.63 |
| W4 cube (N = 100000) | 1 | 1e-4 | – | 0×2, 9×3 | 2:i2,3:i2,4:i2 | 6.69e-6 | 1.84e-5 | 6.69e-6 | 1.84e-5 | 1.000/1.000 | 0.43 | 0.24 |
| W4 cube (N = 100000) | 1 | 1e-5 | – | 0×2, 12×3 | 2:i2,3:i2,4:i2 | 5.91e-7 | 1.86e-6 | 5.89e-7 | 1.85e-6 | 0.998/0.998 | 0.79 | 0.29 |
| W4 cube (N = 100000) | 1 | 1e-7 | – | 0×2, 20×3 | none | 2.21e-9 | 1.37e-8 | 2.21e-9 | 1.37e-8 | 1.000/1.000 | 2.63 | 2.64 |
| W4 cube (N = 100000) | 100 | 1e-4 | – | 0×2, 32, 19, 12 | 2:i2,3:i2,4:i2 | 3.79e-6 | 8.74e-6 | 3.79e-6 | 8.74e-6 | 1.000/1.000 | 0.95 | 0.50 |
| W4 cube (N = 100000) | 100 | 1e-5 | – | 0×2, 34, 20, 13 | 2:i2,3:i2,4:i2 | 9.56e-7 | 2.53e-6 | 9.57e-7 | 2.53e-6 | 1.001/1.001 | 1.14 | 0.53 |
| W4 cube (N = 100000) | 100 | 1e-7 | – | 0×2, 36, 23, 20 | none | 9.37e-9 | 1.71e-8 | 9.37e-9 | 1.71e-8 | 1.000/1.000 | 2.81 | 2.79 |
| W4 clusters (N = 100000) | 1 | 1e-4 | – | 0×2, 9×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 7.92e-6 | 1.85e-5 | 7.92e-6 | 1.85e-5 | 1.000/1.000 | 0.92 | 0.61 |
| W4 clusters (N = 100000) | 1 | 1e-5 | – | 0×2, 12×7 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 7.98e-7 | 2.52e-6 | 7.90e-7 | 2.49e-6 | 0.990/0.991 | 1.77 | 0.85 |
| W4 clusters (N = 100000) | 1 | 1e-7 | – | 0×2, 20×7 | none | 4.22e-9 | 2.87e-8 | 4.22e-9 | 2.87e-8 | 1.000/1.000 | 5.20 | 5.33 |
| W4 clusters (N = 100000) | 100 | 1e-4 | – | 0×2, 22, 14, 9×5 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 8.07e-6 | 1.88e-5 | 8.07e-6 | 1.88e-5 | 1.000/1.000 | 1.02 | 0.66 |
| W4 clusters (N = 100000) | 100 | 1e-5 | – | 0×2, 24, 15, 12×5 | 2:i2,3:i2,4:i2,5:i2,6:i2,7:i2,8:i2 | 7.87e-7 | 2.53e-6 | 7.83e-7 | 2.51e-6 | 0.994/0.992 | 1.66 | 0.92 |
| W4 clusters (N = 100000) | 100 | 1e-7 | – | 0×2, 26, 20×6 | none | 3.99e-9 | 2.87e-8 | 3.99e-9 | 2.87e-8 | 1.000/1.000 | 5.44 | 5.58 |

The worst ratio is 1.070 (φ on the Plummer sphere at kD = 1, ε = 10⁻⁶, seven levels with
waves); the worst ∇φ 1.002 (W1 at k = 100 + 10i, 10⁻⁶). Many ratios at 10⁻⁶ lie below 1 (to
0.910 on the clusters' ∇φ), which is iprec 2's error of the size of a few per cent of the
FMM's, in either direction. With several levels together the ratio is not the product of
the single levels' (Plummer at kD = 1: single levels up to 1.044, together 1.070).

## 5. The levels left to Rotation

Under the rule, on CH3.1c's cases:

- **Every level at ε < 10⁻⁶**: in D1's range ε = 10⁻⁷ and 10⁻⁸, on every workload. These are
  the rows "none" of §4. CH3.1c's matrix has 19 such cases (its 10⁻⁸ rows, 17 of them run
  here), and the in-between cases 9 more at 10⁻⁷.
- **The levels whose κ_l lies outside the tables** (Re κ_l ≥ 16π, Im κ_l ≥ 12π), at every
  ε: W1 at kD = 500, levels 2 and 3 (κ ≈ 125 and 62). On kD ≤ 100 every level ≥ 2 lies in
  the tables.
- Levels 0 and 1, which have no V list.

At ε ≥ 10⁻⁶ the rule gives every other level iprec 2, and no case needs a level left out:
the 1.1× holds with all of them (§4).

## 6. Tests and determinism

- `nd-fmm-tables` (`tests/helmholtz/plane_wave_merge.rs`): `precision_for_tolerance` at
  and around `PLANE_WAVE_MIN_TOLERANCE`, `precision_for_degree` to p = 40, the degree the
  calibrated plateau at the tolerance, `select` by tolerance and by degree, outside the
  tables and for invalid k.
- `nd-fmm-exec` (`tests/helmholtz/plane_wave.rs`): the operator's precisions by the degree,
  by the tolerance (before and after `with_m2l`, named precisions ignoring it), the rule's
  degree the `DegreeRule::CALIBRATED` plateau; `tests/helmholtz_fmm.rs`, `check_plane_wave`
  on its two scenarios (complex k at ε = 10⁻³ with the tolerance; real k at named p = 8 by
  the degree): the errors **within 1.1× Rotation's, asserted** (1.000× and 1.000× in
  debug on the M6), with CH6.1d's checks (the direct sum, the pair path to 2.0e-16, 2
  threads, the one-rank FMM, two vectors).
- `nd-fmm-validate` (`tests/helmholtz_accuracy.rs`, `check_plane_wave`): W1 at kD = 10 at
  ε = 10⁻³ and 10⁻⁶ and W4's clusters at N = 2,000, complex k, real charges, ε = 10⁻³,
  under the rule **within 1.1× Rotation's, asserted**, the row's precisions the rule's, at
  10⁻⁶ also the errors within ε at 1 and 2 threads and on one rank; at 10⁻⁸ no level with
  waves and Rotation's checksum; a named precision as named.
- Determinism under the rule (`ranks` stage): the same checksum at 1 × 6, 2 × 3 and 3 × 2
  ranks × threads on W1 at kD = 30 and 100, ε = 10⁻³ and 10⁻⁶ (four cases, every level with
  waves):

  | case | checksum (all three splits) |
  | --- | --- |
  | W1, kD = 30, 10⁻³ | e5b2189c0f68d9a3 |
  | W1, kD = 30, 10⁻⁶ | 967fe2b9fa69b1fe |
  | W1, kD = 100, 10⁻³ | 564b21950593560d |
  | W1, kD = 100, 10⁻⁶ | 8cb364bf7af356d7 |

## 7. Observations for CH6.2

- At ε ≥ 10⁻⁶ the exponential M2L with iprec 2 evaluated in 0.25 to 0.65 of Rotation's
  time on the trees of 10⁵ points and more at 10⁻⁶ and 0.5 to 1.2 at 10⁻³ (§4, a loaded
  machine), and slower on the smallest trees (W1 and W3 at kD = 10, under 0.3 s). CH6.2
  times it against
  Rotation and the table-based strategies, and against FMM3D.
- **Memory.** `plane_wave_bytes` (the tables, the waves buffer of one frame of every box of
  the largest level with waves, and the scratch) under the rule: 9.0 GB on W1 at kD = 500
  (levels 4–7), 11.8 GB on W3 at kD = 100 (levels 2–6), against Rotation's tables of a few
  hundred MB; per level of W3 at kD = 100 forced alone, at iprec 1, 2, 3: level 5 0.7, 1.7,
  3.2 GB; level 6 (about 2.6 × 10⁵ boxes) 4.7, 11.7, 22.9 GB. The last ran on the M6's 24 GB
  in swap (334 s against 16 s at iprec 2), and the run after it, the same level at 10⁻⁶,
  was stopped when the machine ran out of memory (a debug `cargo test --workspace` ran
  beside it). Laplace's level call (T13) holds the same buffer, but its W is small. A
  buffer per chunk of parents instead of per level would bound it, and the proposal's node
  budget (§2, constraint 7) has to count it. CH6.2's roadmap line takes it.
- Its scalar loops have no SIMD kernel yet (CH6.1d). The rule's precision is a cost too:
  iprec 2 has more waves than iprec 1, which the rule never takes.
- Tight ε needs another quadrature: FMM3D's iprec 3 errs on the truncated multipole's tail
  at the FMM's degrees (CH6.1c's study, this scan). A quadrature matched to the degree, as
  §3.14's `nodes_for_degree` is for Laplace, generated by CH6.1b's `generate` at the
  FMM's ε and p, is the follow-up CH6.1f (question 3 of the PR).

## 8. Reproducing

```sh
export HWLOC_COMPONENTS=-darwin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch6-1e-runs.sh <out> scan between      # the scan, about 60 min
tools/helmholtz/ch6-1e-runs.sh <out> verify ranks      # the rule, about 8 min
tools/helmholtz/ch6-1e-runs.sh <out> verify-large               # about 15 min, 12 GB of waves
```

`scan-large` needs more memory than the M6 has (W3's level 6 at iprec 3: 22.9 GB of waves
alone). It ran to that point on W3 at kD = 100 and not on W1 at kD = 500.

The scan ran before the rule changed (the scan names every precision per level, so the
rule does not enter it); the verification after.

## Appendix: the scan per level

Each level with a quadrature forced to each precision alone, the other levels Rotation:
the ratio of the FMM's errors to Rotation's (the larger of φ and ∇φ). κ ≈ kD 2⁻ˡ (Re κ).

| workload | kD | charges | ε | level | p | κ≈ | iprec 1 | 2 | 3 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| W1 | 10 | complex | ε = 1e-3 | 2 | 7 | 2.50 | 1.001 | 1.000 | 1.000 |
| W1 | 10 | complex | ε = 1e-4 | 2 | 9 | 2.50 | 1.217 | 1.000 | 1.000 |
| W1 | 10 | complex | ε = 1e-5 | 2 | 12 | 2.50 | 8.580 | 1.002 | 1.000 |
| W1 | 10 | complex | ε = 1e-6 | 2 | 15 | 2.50 | 86.964 | 1.005 | 1.000 |
| W1 | 10 | complex | ε = 1e-7 | 2 | 20 | 2.50 | 3449.949 | 4.310 | 1.001 |
| W1 | 10 | complex | ε = 1e-8 | 2 | 24 | 2.50 | 62540.804 | 76.920 | 0.996 |
| W1 | 30 | complex | ε = 1e-3 | 2 | 12 | 7.50 | 1.060 | 1.000 | 1.000 |
| W1 | 30 | complex | ε = 1e-3 | 3 | 8 | 3.75 | 1.016 | 1.000 | 1.000 |
| W1 | 30 | complex | ε = 1e-4 | 2 | 14 | 7.50 | 1.519 | 1.000 | 1.000 |
| W1 | 30 | complex | ε = 1e-4 | 3 | 9 | 3.75 | 1.101 | 1.000 | 1.000 |
| W1 | 30 | complex | ε = 1e-5 | 2 | 15 | 7.50 | 17.061 | 1.000 | 1.000 |
| W1 | 30 | complex | ε = 1e-5 | 3 | 12 | 3.75 | 6.572 | 1.001 | 1.000 |
| W1 | 30 | complex | ε = 1e-6 | 2 | 16 | 7.50 | 78.580 | 1.000 | 1.000 |
| W1 | 30 | complex | ε = 1e-6 | 3 | 15 | 3.75 | 30.127 | 1.001 | 1.000 |
| W1 | 30 | complex | ε = 1e-7 | 2 | 20 | 7.50 | 2026.582 | 2.055 | 1.001 |
| W1 | 30 | complex | ε = 1e-7 | 3 | 20 | 3.75 | 776.300 | 1.142 | 1.000 |
| W1 | 30 | complex | ε = 1e-8 | 2 | 24 | 7.50 | 33492.598 | 29.953 | 0.995 |
| W1 | 30 | complex | ε = 1e-8 | 3 | 24 | 3.75 | 12829.621 | 9.700 | 1.002 |
| W1 | 100 | complex | ε = 1e-3 | 2 | 30 | 25.00 | 1.013 | 1.000 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-3 | 2 | 30 | 25.00 | 1.000 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-3 | 3 | 18 | 12.50 | 1.007 | 1.000 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-3 | 3 | 18 | 12.50 | 1.001 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-3 | 4 | 11 | 6.25 | 1.018 | 1.000 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-3 | 4 | 11 | 6.25 | 1.003 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-3 | 5 | 7 | 3.12 | 1.003 | 1.000 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-3 | 5 | 7 | 3.12 | 1.003 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-4 | 2 | 32 | 25.00 | 1.811 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-4 | 3 | 19 | 12.50 | 1.598 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-4 | 4 | 12 | 6.25 | 1.950 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-4 | 5 | 9 | 3.12 | 1.511 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-5 | 2 | 34 | 25.00 | 8.824 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-5 | 3 | 20 | 12.50 | 7.271 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-5 | 4 | 13 | 6.25 | 9.871 | 0.999 | 1.000 |
| W1 | 100 | complex | ε = 1e-5 | 5 | 12 | 3.12 | 6.617 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-6 | 2 | 35 | 25.00 | 71.472 | 1.001 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-6 | 2 | 35 | 25.00 | 1.784 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-6 | 3 | 22 | 12.50 | 58.752 | 1.005 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-6 | 3 | 22 | 12.50 | 11.134 | 1.000 | 1.000 |
| W1 | 100 | complex | ε = 1e-6 | 4 | 15 | 6.25 | 80.244 | 0.988 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-6 | 4 | 15 | 6.25 | 18.855 | 1.005 | 1.000 |
| W1 | 100 | complex | ε = 1e-6 | 5 | 15 | 3.12 | 53.442 | 1.001 | 1.000 |
| W1 | 100 + 10i | complex | ε = 1e-6 | 5 | 15 | 3.12 | 36.103 | 0.999 | 1.000 |
| W1 | 100 | complex | ε = 1e-7 | 2 | 36 | 25.00 | 753.233 | 1.133 | 1.000 |
| W1 | 100 | complex | ε = 1e-7 | 3 | 23 | 12.50 | 619.802 | 1.260 | 1.002 |
| W1 | 100 | complex | ε = 1e-7 | 4 | 20 | 6.25 | 846.149 | 1.139 | 0.999 |
| W1 | 100 | complex | ε = 1e-7 | 5 | 20 | 3.12 | 563.353 | 1.205 | 1.000 |
| W1 | 100 | complex | ε = 1e-8 | 2 | 38 | 25.00 | 4926.408 | 3.540 | 1.000 |
| W1 | 100 | complex | ε = 1e-8 | 3 | 24 | 12.50 | 4053.735 | 5.067 | 1.005 |
| W1 | 100 | complex | ε = 1e-8 | 4 | 24 | 6.25 | 5533.849 | 3.894 | 1.000 |
| W1 | 100 | complex | ε = 1e-8 | 5 | 24 | 3.12 | 3684.252 | 4.351 | 1.000 |
| W1 | 100 | real | ε = 1e-3 | 2 | 30 | 25.00 | 1.010 | 1.000 | 1.000 |
| W1 | 100 | real | ε = 1e-3 | 3 | 18 | 12.50 | 1.007 | 1.000 | 1.000 |
| W1 | 100 | real | ε = 1e-3 | 4 | 11 | 6.25 | 1.017 | 1.000 | 1.000 |
| W1 | 100 | real | ε = 1e-3 | 5 | 7 | 3.12 | 1.010 | 1.000 | 1.000 |
| W1 | 100 | real | ε = 1e-6 | 2 | 35 | 25.00 | 70.380 | 1.001 | 1.000 |
| W1 | 100 | real | ε = 1e-6 | 3 | 22 | 12.50 | 59.172 | 1.002 | 1.000 |
| W1 | 100 | real | ε = 1e-6 | 4 | 15 | 6.25 | 79.001 | 0.984 | 1.000 |
| W1 | 100 | real | ε = 1e-6 | 5 | 15 | 3.12 | 53.844 | 1.001 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-3 | 2 | 12 | 7.50 | 1.012 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-3 | 3 | 8 | 3.75 | 1.007 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-3 | 4 | 7 | 1.88 | 1.001 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-4 | 2 | 14 | 7.50 | 1.298 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-4 | 3 | 9 | 3.75 | 1.073 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-4 | 4 | 9 | 1.88 | 1.058 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-5 | 2 | 15 | 7.50 | 10.652 | 1.002 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-5 | 3 | 12 | 3.75 | 5.209 | 0.999 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-5 | 4 | 12 | 1.88 | 4.783 | 1.000 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-6 | 2 | 16 | 7.50 | 53.568 | 0.986 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-6 | 3 | 15 | 3.75 | 26.008 | 1.003 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-6 | 4 | 15 | 1.88 | 23.487 | 1.001 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-7 | 2 | 20 | 7.50 | 1336.779 | 1.925 | 0.978 |
| W3 (N = 108849) | 30 | complex | ε = 1e-7 | 3 | 20 | 3.75 | 647.791 | 1.116 | 0.996 |
| W3 (N = 108849) | 30 | complex | ε = 1e-7 | 4 | 20 | 1.88 | 586.083 | 1.191 | 1.000 |
| W3 (N = 108849) | 30 | complex | ε = 1e-8 | 2 | 24 | 7.50 | 10648.685 | 13.821 | 0.779 |
| W3 (N = 108849) | 30 | complex | ε = 1e-8 | 3 | 24 | 3.75 | 5160.033 | 4.315 | 0.979 |
| W3 (N = 108849) | 30 | complex | ε = 1e-8 | 4 | 24 | 1.88 | 4668.495 | 5.273 | 0.998 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-3 | 2 | 30 | 25.00 | 1.021 | 1.000 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-3 | 3 | 18 | 12.50 | 1.014 | 1.000 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-3 | 4 | 11 | 6.25 | 1.021 | 1.000 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-3 | 5 | 7 | 3.12 | 1.007 | 1.000 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-3 | 6 | 7 | 1.56 | 1.013 | 1.000 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-6 | 2 | 35 | 25.00 | 98.920 | 1.002 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-6 | 3 | 22 | 12.50 | 83.030 | 1.004 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-6 | 4 | 15 | 6.25 | 102.461 | 1.000 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-6 | 5 | 15 | 3.12 | 67.402 | 1.005 | 1.000 |
| W3 (N = 4031442) | 100 | complex | ε = 1e-6 | 6 | 15 | 1.56 | 63.854 | 1.000 | – |
| W3 (N = 4032) | 10 | complex | ε = 1e-3 | 2 | 7 | 2.50 | 1.012 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-3 | 3 | 7 | 1.25 | 1.003 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-4 | 2 | 9 | 2.50 | 1.587 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-4 | 3 | 9 | 1.25 | 1.245 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-5 | 2 | 12 | 2.50 | 15.171 | 1.002 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-5 | 3 | 12 | 1.25 | 9.065 | 1.000 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-6 | 2 | 15 | 2.50 | 137.477 | 0.989 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-6 | 3 | 15 | 1.25 | 81.598 | 1.002 | 1.000 |
| W3 (N = 4032) | 10 | complex | ε = 1e-7 | 2 | 20 | 2.50 | 5200.777 | 5.403 | 0.982 |
| W3 (N = 4032) | 10 | complex | ε = 1e-7 | 3 | 20 | 1.25 | 3086.553 | 3.781 | 0.997 |
| W3 (N = 4032) | 10 | complex | ε = 1e-8 | 2 | 24 | 2.50 | 62806.328 | 64.139 | 0.905 |
| W3 (N = 4032) | 10 | complex | ε = 1e-8 | 3 | 24 | 1.25 | 37274.240 | 44.095 | 0.908 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 2 | 7 | 0.25 | 1.001 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 3 | 7 | 0.12 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 4 | 7 | 0.06 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 5 | 7 | 0.03 | 1.002 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 6 | 7 | 0.02 | 1.027 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 7 | 7 | 0.01 | 1.061 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-3 | 8 | 7 | 0.00 | 1.015 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 2 | 9 | 0.25 | 1.034 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 3 | 9 | 0.12 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 4 | 9 | 0.06 | 1.005 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 5 | 9 | 0.03 | 1.149 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 6 | 9 | 0.02 | 1.577 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 7 | 9 | 0.01 | 2.348 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-4 | 8 | 9 | 0.00 | 1.759 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 2 | 12 | 0.25 | 2.648 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 3 | 12 | 0.12 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 4 | 12 | 0.06 | 1.385 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 5 | 12 | 0.03 | 5.610 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 6 | 12 | 0.02 | 12.308 | 0.999 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 7 | 12 | 0.01 | 21.439 | 0.998 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-5 | 8 | 12 | 0.00 | 14.517 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 2 | 15 | 0.25 | 20.681 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 3 | 15 | 0.12 | 1.223 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 4 | 15 | 0.06 | 8.038 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 5 | 15 | 0.03 | 46.663 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 6 | 15 | 0.02 | 103.625 | 0.996 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 7 | 15 | 0.01 | 180.771 | 1.016 | 0.999 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-6 | 8 | 15 | 0.00 | 122.116 | 0.967 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 2 | 20 | 0.25 | 462.283 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 3 | 20 | 0.12 | 14.681 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 4 | 20 | 0.06 | 178.326 | 1.009 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 5 | 20 | 0.03 | 1043.714 | 1.212 | 1.004 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 6 | 20 | 0.02 | 2318.007 | 1.881 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 7 | 20 | 0.01 | 4045.194 | 2.677 | 1.116 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-7 | 8 | 20 | 0.00 | 2732.347 | 1.803 | 0.994 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 2 | 24 | 0.25 | 3637.634 | 1.077 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 3 | 24 | 0.12 | 115.422 | 1.004 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 4 | 24 | 0.06 | 1403.327 | 1.525 | 1.001 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 5 | 24 | 0.03 | 8212.984 | 5.741 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 6 | 24 | 0.02 | 18240.451 | 12.850 | 1.000 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 7 | 24 | 0.01 | 31831.609 | 20.027 | 1.071 |
| W4 clusters (N = 100000) | 1 | complex | ε = 1e-8 | 8 | 24 | 0.00 | 21501.014 | 12.411 | 0.834 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 2 | 7 | 2.50 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 3 | 7 | 1.25 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 4 | 7 | 0.62 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 5 | 7 | 0.31 | 1.006 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 6 | 7 | 0.16 | 1.023 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 7 | 7 | 0.08 | 1.049 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-3 | 8 | 7 | 0.04 | 1.016 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 2 | 15 | 2.50 | 9.894 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 3 | 15 | 1.25 | 1.230 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 4 | 15 | 0.62 | 8.436 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 5 | 15 | 0.31 | 37.977 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 6 | 15 | 0.16 | 106.183 | 0.997 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 7 | 15 | 0.08 | 167.018 | 0.998 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-6 | 8 | 15 | 0.04 | 115.270 | 0.968 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 2 | 24 | 2.50 | 1739.996 | 2.673 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 3 | 24 | 1.25 | 115.939 | 1.006 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 4 | 24 | 0.62 | 1486.364 | 1.394 | 1.000 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 5 | 24 | 0.31 | 6690.000 | 3.144 | 0.999 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 6 | 24 | 0.16 | 18703.526 | 14.488 | 0.998 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 7 | 24 | 0.08 | 29409.890 | 19.125 | 1.760 |
| W4 clusters (N = 100000) | 10 | complex | ε = 1e-8 | 8 | 24 | 0.04 | 20300.071 | 14.405 | 0.817 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 2 | 21 | 25.00 | 1.001 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 3 | 13 | 12.50 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 4 | 8 | 6.25 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 5 | 7 | 3.12 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 6 | 7 | 1.56 | 1.025 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 7 | 7 | 0.78 | 1.031 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-3 | 8 | 7 | 0.39 | 1.012 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 2 | 22 | 25.00 | 1.007 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 3 | 14 | 12.50 | 1.000 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 4 | 9 | 6.25 | 1.002 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 5 | 9 | 3.12 | 1.005 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 6 | 9 | 1.56 | 1.743 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 7 | 9 | 0.78 | 1.597 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-4 | 8 | 9 | 0.39 | 1.313 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 2 | 24 | 25.00 | 1.530 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 3 | 15 | 12.50 | 1.014 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 4 | 12 | 6.25 | 1.186 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 5 | 12 | 3.12 | 1.516 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 6 | 12 | 1.56 | 14.675 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 7 | 12 | 0.78 | 12.312 | 0.998 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-5 | 8 | 12 | 0.39 | 9.120 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 2 | 25 | 25.00 | 10.218 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 3 | 16 | 12.50 | 1.219 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 4 | 15 | 6.25 | 5.830 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 5 | 15 | 3.12 | 10.118 | 1.000 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 6 | 15 | 1.56 | 126.940 | 0.997 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 7 | 15 | 0.78 | 107.128 | 1.005 | 1.002 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-6 | 8 | 15 | 0.39 | 78.663 | 0.972 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 2 | 26 | 25.00 | 231.496 | 1.029 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 3 | 20 | 12.50 | 17.559 | 1.001 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 4 | 20 | 6.25 | 131.727 | 1.005 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 5 | 20 | 3.12 | 229.427 | 1.043 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 6 | 20 | 1.56 | 2889.631 | 3.136 | 0.998 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 7 | 20 | 0.78 | 2436.548 | 3.882 | 0.987 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-7 | 8 | 20 | 0.39 | 1790.407 | 1.587 | 0.962 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 2 | 27 | 25.00 | 1842.943 | 2.156 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 3 | 24 | 12.50 | 139.464 | 1.022 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 4 | 24 | 6.25 | 1048.501 | 1.218 | 1.000 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 5 | 24 | 3.12 | 1826.980 | 2.988 | 0.999 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 6 | 24 | 1.56 | 23008.886 | 23.723 | 0.997 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 7 | 24 | 0.78 | 19401.472 | 29.996 | 0.983 |
| W4 clusters (N = 100000) | 100 | complex | ε = 1e-8 | 8 | 24 | 0.39 | 14256.367 | 10.587 | 0.680 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-3 | 2 | 7 | 0.25 | 1.054 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-3 | 3 | 7 | 0.12 | 1.135 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-3 | 4 | 7 | 0.06 | 1.099 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-4 | 2 | 9 | 0.25 | 2.255 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-4 | 3 | 9 | 0.12 | 3.565 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-4 | 4 | 9 | 0.06 | 2.932 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-5 | 2 | 12 | 0.25 | 22.453 | 0.999 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-5 | 3 | 12 | 0.12 | 38.764 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-5 | 4 | 12 | 0.06 | 30.573 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-6 | 2 | 15 | 0.25 | 207.226 | 0.997 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-6 | 3 | 15 | 0.12 | 357.495 | 0.982 | 1.001 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-6 | 4 | 15 | 0.06 | 282.118 | 1.003 | 1.000 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-7 | 2 | 20 | 0.25 | 5993.575 | 4.086 | 0.995 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-7 | 3 | 20 | 0.12 | 10339.140 | 3.127 | 0.994 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-7 | 4 | 20 | 0.06 | 8158.522 | 4.984 | 0.999 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-8 | 2 | 24 | 0.25 | 51522.951 | 34.401 | 0.964 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-8 | 3 | 24 | 0.12 | 88879.403 | 25.718 | 0.992 |
| W4 cube (N = 100000) | 1 | complex | ε = 1e-8 | 4 | 24 | 0.06 | 70133.699 | 42.158 | 0.887 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-3 | 2 | 7 | 2.50 | 1.016 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-3 | 3 | 7 | 1.25 | 1.019 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-3 | 4 | 7 | 0.62 | 1.029 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-6 | 2 | 15 | 2.50 | 127.816 | 0.994 | 1.000 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-6 | 3 | 15 | 1.25 | 161.070 | 1.003 | 1.000 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-6 | 4 | 15 | 0.62 | 155.165 | 1.011 | 1.000 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-8 | 2 | 24 | 2.50 | 33960.903 | 33.556 | 0.961 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-8 | 3 | 24 | 1.25 | 42799.405 | 50.507 | 0.989 |
| W4 cube (N = 100000) | 10 | complex | ε = 1e-8 | 4 | 24 | 0.62 | 41230.717 | 64.324 | 0.847 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-3 | 2 | 30 | 25.00 | 1.119 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-3 | 3 | 18 | 12.50 | 1.108 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-3 | 4 | 11 | 6.25 | 1.145 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-4 | 2 | 32 | 25.00 | 2.859 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-4 | 3 | 19 | 12.50 | 2.473 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-4 | 4 | 12 | 6.25 | 2.909 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-5 | 2 | 34 | 25.00 | 10.687 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-5 | 3 | 20 | 12.50 | 9.037 | 1.001 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-5 | 4 | 13 | 6.25 | 11.025 | 1.000 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-6 | 2 | 35 | 25.00 | 105.383 | 1.003 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-6 | 3 | 22 | 12.50 | 89.515 | 1.006 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-6 | 4 | 15 | 6.25 | 108.727 | 0.994 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-7 | 2 | 36 | 25.00 | 1083.471 | 1.254 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-7 | 3 | 23 | 12.50 | 920.468 | 1.536 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-7 | 4 | 20 | 6.25 | 1118.101 | 1.241 | 0.999 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-8 | 2 | 38 | 25.00 | 4809.524 | 3.471 | 1.000 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-8 | 3 | 24 | 12.50 | 4086.045 | 5.348 | 1.013 |
| W4 cube (N = 100000) | 100 | complex | ε = 1e-8 | 4 | 24 | 6.25 | 4963.354 | 3.429 | 0.998 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 2 | 7 | 0.25 | 1.001 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 3 | 7 | 0.12 | 1.005 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 4 | 7 | 0.06 | 1.007 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 5 | 7 | 0.03 | 1.014 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 6 | 7 | 0.02 | 1.075 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 7 | 7 | 0.01 | 1.060 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-3 | 8 | 7 | 0.00 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 2 | 15 | 0.25 | 4.820 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 3 | 15 | 0.12 | 24.385 | 1.002 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 4 | 15 | 0.06 | 49.614 | 1.002 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 5 | 15 | 0.03 | 99.396 | 0.997 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 6 | 15 | 0.02 | 195.076 | 1.044 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 7 | 15 | 0.01 | 182.547 | 1.020 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-6 | 8 | 15 | 0.00 | 44.930 | 1.002 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 2 | 24 | 0.25 | 885.385 | 1.033 | 1.002 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 3 | 24 | 0.12 | 4570.886 | 1.362 | 1.000 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 4 | 24 | 0.06 | 9311.942 | 11.051 | 1.008 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 5 | 24 | 0.03 | 18641.595 | 19.976 | 0.971 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 6 | 24 | 0.02 | 36586.246 | 38.370 | 1.043 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 7 | 24 | 0.01 | 34240.950 | 24.155 | 1.303 |
| W4 plummer (N = 100000) | 1 | complex | ε = 1e-8 | 8 | 24 | 0.00 | 8417.230 | 5.721 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 2 | 7 | 2.50 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 3 | 7 | 1.25 | 1.002 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 4 | 7 | 0.62 | 1.021 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 5 | 7 | 0.31 | 1.009 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 6 | 7 | 0.16 | 1.082 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 7 | 7 | 0.08 | 1.059 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-3 | 8 | 7 | 0.04 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 2 | 15 | 2.50 | 2.321 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 3 | 15 | 1.25 | 12.832 | 1.003 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 4 | 15 | 0.62 | 64.831 | 1.010 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 5 | 15 | 0.31 | 103.969 | 0.997 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 6 | 15 | 0.16 | 226.656 | 1.018 | 1.001 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 7 | 15 | 0.08 | 180.274 | 1.008 | 0.999 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-6 | 8 | 15 | 0.04 | 44.933 | 1.002 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 2 | 24 | 2.50 | 417.311 | 1.090 | 1.001 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 3 | 24 | 1.25 | 2447.148 | 1.855 | 1.000 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 4 | 24 | 0.62 | 12398.993 | 18.121 | 0.991 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 5 | 24 | 0.31 | 19881.778 | 11.791 | 0.950 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 6 | 24 | 0.16 | 43363.886 | 17.556 | 0.942 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 7 | 24 | 0.08 | 34467.408 | 22.721 | 1.004 |
| W4 plummer (N = 100000) | 10 | complex | ε = 1e-8 | 8 | 24 | 0.04 | 8585.509 | 8.257 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 2 | 30 | 25.00 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 3 | 18 | 12.50 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 4 | 11 | 6.25 | 1.001 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 5 | 7 | 3.12 | 1.000 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 6 | 7 | 1.56 | 1.012 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 7 | 7 | 0.78 | 1.015 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-3 | 8 | 7 | 0.39 | 1.005 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 2 | 35 | 25.00 | 1.761 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 3 | 21 | 12.50 | 6.421 | 1.001 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 4 | 15 | 6.25 | 26.392 | 1.000 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 5 | 15 | 3.12 | 46.173 | 1.008 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 6 | 15 | 1.56 | 90.453 | 1.007 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 7 | 15 | 0.78 | 75.722 | 0.998 | 1.002 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-6 | 8 | 15 | 0.39 | 25.937 | 1.001 | 1.000 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 2 | 37 | 25.00 | 179.133 | 1.003 | 1.001 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 3 | 24 | 12.50 | 764.319 | 1.348 | 0.989 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 4 | 24 | 6.25 | 3181.775 | 2.853 | 0.994 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 5 | 24 | 3.12 | 5566.065 | 6.505 | 0.964 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 6 | 24 | 1.56 | 10910.075 | 10.969 | 0.995 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 7 | 24 | 0.78 | 9133.444 | 12.058 | 0.999 |
| W4 plummer (N = 100000) | 100 | complex | ε = 1e-8 | 8 | 24 | 0.39 | 3127.159 | 2.317 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-3 | 2 | 7 | 0.25 | 1.024 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-3 | 3 | 7 | 0.12 | 1.071 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-3 | 4 | 7 | 0.06 | 1.046 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-3 | 5 | 7 | 0.03 | 1.039 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-6 | 2 | 15 | 0.25 | 150.929 | 1.001 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-6 | 3 | 15 | 0.12 | 210.954 | 0.997 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-6 | 4 | 15 | 0.06 | 170.662 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-6 | 5 | 15 | 0.03 | 165.021 | 0.990 | 1.000 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-8 | 2 | 24 | 0.25 | 48911.943 | 28.389 | 1.001 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-8 | 3 | 24 | 0.12 | 68383.317 | 18.418 | 0.973 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-8 | 4 | 24 | 0.06 | 55308.447 | 34.726 | 0.995 |
| W4 sphere (N = 100000) | 1 | complex | ε = 1e-8 | 5 | 24 | 0.03 | 53492.648 | 44.785 | 1.086 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-3 | 2 | 7 | 2.50 | 1.016 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-3 | 3 | 7 | 1.25 | 1.018 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-3 | 4 | 7 | 0.62 | 1.014 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-3 | 5 | 7 | 0.31 | 1.020 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-6 | 2 | 15 | 2.50 | 80.560 | 0.999 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-6 | 3 | 15 | 1.25 | 99.799 | 1.003 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-6 | 4 | 15 | 0.62 | 111.708 | 1.007 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-6 | 5 | 15 | 0.31 | 118.940 | 0.988 | 1.001 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-8 | 2 | 24 | 2.50 | 27693.389 | 27.885 | 1.000 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-8 | 3 | 24 | 1.25 | 34308.676 | 39.952 | 0.972 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-8 | 4 | 24 | 0.62 | 38409.015 | 52.650 | 0.995 |
| W4 sphere (N = 100000) | 10 | complex | ε = 1e-8 | 5 | 24 | 0.31 | 40889.817 | 30.040 | 0.901 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-3 | 2 | 30 | 25.00 | 1.011 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-3 | 3 | 18 | 12.50 | 1.007 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-3 | 4 | 11 | 6.25 | 1.017 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-3 | 5 | 7 | 3.12 | 1.014 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-6 | 2 | 35 | 25.00 | 80.852 | 1.004 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-6 | 3 | 22 | 12.50 | 65.143 | 1.005 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-6 | 4 | 15 | 6.25 | 88.427 | 1.000 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-6 | 5 | 15 | 3.12 | 65.687 | 0.994 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-8 | 2 | 38 | 25.00 | 5700.725 | 4.103 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-8 | 3 | 24 | 12.50 | 4590.148 | 5.838 | 0.998 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-8 | 4 | 24 | 6.25 | 6231.500 | 4.539 | 1.000 |
| W4 sphere (N = 100000) | 100 | complex | ε = 1e-8 | 5 | 24 | 3.12 | 4630.056 | 5.113 | 1.000 |
