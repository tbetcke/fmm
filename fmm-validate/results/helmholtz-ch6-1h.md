# Helmholtz CH6.1h: the resolving degree on the M2L alone

Helmholtz task CH6.1h ([docs/helmholtz/tasks/CH6.1h.md](../../docs/helmholtz/tasks/CH6.1h.md)),
from CH6.1g's sign-off 3; proposal
[docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md) §10 CH6.1
("errors within about 1.1× Rotation's"), §5.3, §2 (constraints 6, 7 and 9), §12 D8; CH6.1g's
report ([helmholtz-ch6-1g.md](helmholtz-ch6-1g.md) §3, §5) and CH6.1f's
([helmholtz-ch6-1f.md](helmholtz-ch6-1f.md) §3, §4). Errors and times measured on 2026-11-11 on
Kathleen, one node (2 × 20 Cascade Lake cores, exclusive; node-c11a-031), jobs 632717 (CH6.1g's
26 cases, each in both orders) and 632725 (eight more W4 cases at 10⁻⁷, forward only, for the
errors), one rank × 40 threads, release build, f64, AVX-512 P2P, BLAS threads 1, from branch
`tbetcke/h-ch6-1h` (7710b1a; 632725 with the run script's stage `e7x` added). A probe on the M6
(release, 4 threads, errors only) in §2.

## Summary

- **Three forms of the raised level** (`PlaneWaveRaise`, §1). In each the level's stores, its
  M2L, the M2M from its children and the L2L to them are at the resolving degree p_w.
  **`Level`** (CH6.1g's, the default): every other operator of the level at p_w too.
  **`Leaves`**: the M2M to the parent and the L2L from it on the first p_l degrees.
  **`M2l`**, the M2L alone: those and the leaves' P2M, L2P, P2L and M2P on the first p_l degrees.
- **`Leaves` keeps `Level`'s accuracy and gains little.** Its errors are within a few per cent of
  `Level`'s (at most 0.81× Rotation's), its evaluation 0.93–1.01× `Level`'s at 10⁻⁷ and
  0.94–1.03× at 10⁻⁸: the leaves' P2M and L2P at p_w are most of the raised rest.
- **`M2l` is fast but close to the limit.** Its rest (evaluation − M2L) takes 1.06–1.35×
  Rotation's (`Level`: 1.36–1.88×), its evaluation 0.66–0.96× `Level`'s. At 10⁻⁷ its errors are
  **0.11–1.00× Rotation's on all 17 cases** (CH6.1g's 9 and 8 more); at 10⁻⁸ 0.06–1.00× on 16 of
  17, but **1.53× (φ) on the clusters at kD = 10**, which fails the acceptance. A leaf's
  multipole and local at p_l are iprec 3's chain at p_l, which CH6.1f's study (§4) and CH6.1g
  §1 found at about Rotation's error; the interior boxes, at p_w, pull the FMM's error below it,
  by how much depending on the tree.
- **Recommended (question 1):** the form by ε, a function of ε only (constraint 9): `M2l` at
  10⁻⁷ ≤ ε < 10⁻⁶ and `Leaves` at 10⁻⁸ ≤ ε < 10⁻⁷. Then at 10⁻⁷ the evaluation takes **0.43–0.83×
  Rotation's** on every case above 0.1 s (0.95–1.13× on three ms cases), against 0.48–0.95× under
  `Level`; at 10⁻⁸ 0.31–0.71× above 0.1 s, as `Level` (W3 at N = 4,032, 0.10 s, 1.43×, as under
  `Level`).
- **D8 (question 2):** with `M2l` at 10⁻⁷ the exponential M2L beats Rotation there on every case
  above 0.1 s; recommended: `Auto` takes it at 10⁻⁷ ≤ ε < 10⁻⁶ too.

## 1. The forms

CH6.1f (§4) reads the chain of a degree-p exponential M2L as the quadrature of
S_p(q; s) e^{iq·d} T_p(q; y), the truncated plane-wave series of the sources about the
multipole's centre and of the targets about the local's; at complex q a truncated series
converges worse the larger |s| or |y|. Its study held at p_w with the multipole formed by M2M
from the children at p (half the radius) and the local taken to the children by L2L at p: what
works at the children's radius needs no raise. `HelmholtzTables::build_with_m2l_degrees` gives a
level two degrees, its expansions' and M2L's q_l = p_w and its translations' to and from the
parent p_l; the operator's `PlaneWaveRaise` chooses which operators of a raised level use which:

| operator on a raised level l | `Level` (CH6.1g) | `Leaves` | `M2l` |
| --- | --- | --- | --- |
| stores, ghost exchange, M2L | p_w | p_w | p_w |
| M2M from the children into l, L2L from l to them | children's p → p_w, p_w → children's p | same | same |
| M2M from l to its parent, L2L from the parent into l | p_w | **p_l** (first degrees) | **p_l** |
| P2M, L2P on l's leaves; P2L into, M2P from l's boxes | p_w | p_w | **p_l** (first degrees, through a scratch expansion) |

(Under `Level` the children's p is their p_w where they are raised too.) `Level` is CH6.1g's
output bit for bit (`build` is `build_with_m2l_degrees` at q = p).

## 2. The probe: leaves at p_l

By the reading above a leaf at p_l should fail: its multipole at the box's own radius is
S_{p_l}. The plan built `Leaves` alone. A temporary change before the Kathleen runs (the leaves'
P2M and L2P at p_l, not committed; the M6, release, 4 threads, errors only) held 1.1× on four
large cases, φ/∇φ over Rotation's: the cube at kD = 1, 10⁻⁸, 0.41/0.59; the clusters at kD = 1,
10⁻⁷, 0.90/0.96; W1 at kD = 100, 10⁻⁷, 0.82/0.65; the Plummer sphere at kD = 1, 10⁻⁸, 0.25/0.75.
So `M2l` was built and joined the runs (§3).

## 3. Errors

CH6.1g's 26 cases (CH6.1e's at 10⁻⁷ and 10⁻⁸; `tools/helmholtz/ch6-1h-runs.sh` stages `e7`,
`e8`) and the eight W4 cases of the 10⁻⁸ matrix that the 10⁻⁷ list lacks (stage `e7x`: the
sphere and the Plummer sphere at kD = 1, 10, 100, the cube and the clusters at kD = 10). Each
launch runs Rotation, then PlaneWave by the rule in the three forms; the checksums of a run's two
launches agree in every case. "p by level" in the harness's notation (`0×2, 28×3`: degree 0 on
two levels, 28 on three); the forms share PlaneWave's degrees.

| ε | workload | kD | p by level, Rotation → PlaneWave | φ Rot. | ∇φ Rot. | Level φ | ∇φ | Leaves φ | ∇φ | M2l φ | ∇φ |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-7 | W1 | 10 | 0×2, 20 → 0×2, 28 | 1.42e-09 | 1.22e-08 | 0.004 | 0.006 | 0.004 | 0.006 | 1.000 | 1.000 |
| 1e-7 | W1 | 30 | 0×2, 20×2 → 0×2, 28×2 | 4.33e-09 | 1.21e-08 | 0.004 | 0.005 | 0.004 | 0.005 | 0.183 | 0.317 |
| 1e-7 | W1 | 100 | 0×2, 36, 23, 20×2 → 0×2, 36, 28×3 | 9.12e-09 | 1.78e-08 | 0.803 | 0.544 | 0.808 | 0.573 | 0.816 | 0.646 |
| 1e-7 | W3 (N = 108849) | 30 | 0×2, 20×3 → 0×2, 28×3 | 5.84e-09 | 2.18e-08 | 0.005 | 0.005 | 0.005 | 0.005 | 0.112 | 0.404 |
| 1e-7 | W3 (N = 4032) | 10 | 0×2, 20×2 → 0×2, 28×2 | 1.19e-09 | 7.68e-09 | 0.007 | 0.004 | 0.007 | 0.004 | 0.815 | 0.878 |
| 1e-7 | W4 clusters (N = 100000) | 1 | 0×2, 20×7 → 0×2, 28×7 | 4.22e-09 | 2.87e-08 | 0.019 | 0.019 | 0.019 | 0.019 | 0.900 | 0.962 |
| 1e-7 | W4 clusters (N = 100000) | 10 | 0×2, 20×7 → 0×2, 28×7 | 4.21e-09 | 2.87e-08 | 0.007 | 0.012 | 0.007 | 0.012 | 0.634 | 0.878 |
| 1e-7 | W4 clusters (N = 100000) | 100 | 0×2, 26, 20×6 → 0×2, 28×7 | 3.99e-09 | 2.87e-08 | 0.006 | 0.006 | 0.006 | 0.006 | 0.566 | 0.812 |
| 1e-7 | W4 cube (N = 100000) | 1 | 0×2, 20×3 → 0×2, 28×3 | 2.21e-09 | 1.37e-08 | 0.005 | 0.006 | 0.005 | 0.006 | 0.440 | 0.818 |
| 1e-7 | W4 cube (N = 100000) | 10 | 0×2, 20×3 → 0×2, 28×3 | 2.02e-09 | 1.36e-08 | 0.009 | 0.004 | 0.009 | 0.004 | 0.381 | 0.757 |
| 1e-7 | W4 cube (N = 100000) | 100 | 0×2, 36, 23, 20 → 0×2, 36, 28×2 | 9.37e-09 | 1.71e-08 | 0.644 | 0.468 | 0.721 | 0.552 | 0.756 | 0.766 |
| 1e-7 | W4 plummer (N = 100000) | 1 | 0×2, 20×7 → 0×2, 28×7 | 4.22e-09 | 1.43e-08 | 0.008 | 0.019 | 0.008 | 0.018 | 0.367 | 0.790 |
| 1e-7 | W4 plummer (N = 100000) | 10 | 0×2, 20×7 → 0×2, 28×7 | 3.12e-09 | 1.43e-08 | 0.006 | 0.011 | 0.006 | 0.011 | 0.340 | 0.713 |
| 1e-7 | W4 plummer (N = 100000) | 100 | 0×2, 36, 22, 20×5 → 0×2, 36, 28×6 | 9.68e-09 | 1.55e-08 | 0.112 | 0.030 | 0.183 | 0.057 | 0.212 | 0.572 |
| 1e-7 | W4 sphere (N = 100000) | 1 | 0×2, 20×4 → 0×2, 28×4 | 1.47e-09 | 1.27e-10 | 0.005 | 0.019 | 0.005 | 0.019 | 0.385 | 0.913 |
| 1e-7 | W4 sphere (N = 100000) | 10 | 0×2, 20×4 → 0×2, 28×4 | 1.48e-09 | 1.28e-10 | 0.006 | 0.018 | 0.006 | 0.018 | 0.333 | 0.867 |
| 1e-7 | W4 sphere (N = 100000) | 100 | 0×2, 36, 23, 20×2 → 0×2, 36, 28×3 | 6.41e-09 | 2.48e-10 | 0.724 | 0.544 | 0.761 | 0.569 | 0.771 | 0.774 |
| 1e-8 | W1 | 10 | 0×2, 24 → 0×2, 32 | 7.81e-11 | 9.22e-10 | 0.045 | 0.007 | 0.045 | 0.007 | 0.997 | 0.992 |
| 1e-8 | W1 | 30 | 0×2, 24×2 → 0×2, 32×2 | 2.62e-10 | 8.32e-10 | 0.022 | 0.008 | 0.022 | 0.008 | 0.137 | 0.300 |
| 1e-8 | W1 | 100 | 0×2, 38, 24×3 → 0×2, 38, 32×3 | 1.39e-09 | 3.40e-09 | 0.396 | 0.194 | 0.452 | 0.257 | 0.453 | 0.271 |
| 1e-8 | W3 (N = 108849) | 30 | 0×2, 24×3 → 0×2, 32×3 | 7.34e-10 | 2.96e-09 | 0.008 | 0.003 | 0.008 | 0.003 | 0.061 | 0.187 |
| 1e-8 | W3 (N = 4032) | 10 | 0×2, 24×2 → 0×2, 32×2 | 9.83e-11 | 8.92e-10 | 0.076 | 0.004 | 0.076 | 0.004 | 0.638 | 0.520 |
| 1e-8 | W4 clusters (N = 100000) | 1 | 0×2, 24×7 → 0×2, 32×7 | 5.37e-10 | 5.00e-09 | 0.005 | 0.004 | 0.005 | 0.004 | 0.831 | 0.872 |
| 1e-8 | W4 clusters (N = 100000) | 10 | 0×2, 24×7 → 0×2, 32×7 | 5.35e-10 | 5.00e-09 | 0.005 | 0.003 | 0.005 | 0.003 | 1.525 | 1.048 |
| 1e-8 | W4 clusters (N = 100000) | 100 | 0×2, 27, 24×6 → 0×2, 32×7 | 5.01e-10 | 4.98e-09 | 0.009 | 0.002 | 0.009 | 0.002 | 0.485 | 0.566 |
| 1e-8 | W4 cube (N = 100000) | 1 | 0×2, 24×3 → 0×2, 32×3 | 2.57e-10 | 1.73e-09 | 0.008 | 0.005 | 0.008 | 0.005 | 0.409 | 0.590 |
| 1e-8 | W4 cube (N = 100000) | 10 | 0×2, 24×3 → 0×2, 32×3 | 2.29e-10 | 1.71e-09 | 0.071 | 0.005 | 0.071 | 0.005 | 0.281 | 0.413 |
| 1e-8 | W4 cube (N = 100000) | 100 | 0×2, 38, 24×2 → 0×2, 38, 32×2 | 2.11e-09 | 4.52e-09 | 0.263 | 0.169 | 0.399 | 0.281 | 0.404 | 0.305 |
| 1e-8 | W4 plummer (N = 100000) | 1 | 0×2, 24×7 → 0×2, 32×7 | 5.90e-10 | 2.41e-09 | 0.003 | 0.005 | 0.003 | 0.005 | 0.247 | 0.743 |
| 1e-8 | W4 plummer (N = 100000) | 10 | 0×2, 24×7 → 0×2, 32×7 | 4.34e-10 | 2.41e-09 | 0.006 | 0.006 | 0.006 | 0.006 | 0.332 | 0.722 |
| 1e-8 | W4 plummer (N = 100000) | 100 | 0×2, 37, 24×6 → 0×2, 37, 32×6 | 1.27e-09 | 2.59e-09 | 0.228 | 0.049 | 0.240 | 0.054 | 0.244 | 0.398 |
| 1e-8 | W4 sphere (N = 100000) | 1 | 0×2, 24×4 → 0×2, 32×4 | 1.31e-10 | 1.20e-11 | 0.008 | 0.187 | 0.008 | 0.187 | 0.504 | 1.000 |
| 1e-8 | W4 sphere (N = 100000) | 10 | 0×2, 24×4 → 0×2, 32×4 | 1.27e-10 | 1.21e-11 | 0.054 | 0.186 | 0.054 | 0.186 | 0.222 | 0.519 |
| 1e-8 | W4 sphere (N = 100000) | 100 | 0×2, 38, 24×3 → 0×2, 38, 32×3 | 8.86e-10 | 4.86e-11 | 0.483 | 0.317 | 0.492 | 0.309 | 0.492 | 0.337 |

Ranges of each form's errors over Rotation's (φ and ∇φ):

| ε | `Level` | `Leaves` | `M2l` |
| --- | --- | --- | --- |
| 10⁻⁷ (17 cases) | 0.004–0.80, 0.004–0.54 | 0.004–0.81, 0.004–0.57 | 0.11–1.00, 0.32–1.00 |
| 10⁻⁸ (17 cases) | 0.003–0.48, 0.002–0.32 | 0.003–0.49, 0.002–0.31 | 0.06–**1.53**, 0.19–1.05 |

- **`Leaves`** stays at `Level`'s errors: the translations to and from the parent at p_l
  truncate what L2L and M2M carry between levels, as Rotation at p_l does.
- **`M2l`** sits at 1.00× where a raised level is all leaves (W1 at kD = 10: level 2 alone),
  the chain of iprec 3 at p_l; elsewhere the interior boxes at p_w bring it to 0.06–0.96×. At
  10⁻⁸ the clusters at kD = 10 reach 1.53× (φ; ∇φ 1.05×), while the same clusters at kD = 1 and
  100 give 0.83× and 0.49× (why kD = 10 alone is not analysed here). At 10⁻⁷ no case of the 17 exceeds 1.00×; the test trees' 0.98–0.99×
  (`check_m2l_raise`) agree.

## 4. Times

Times of the second evaluation of each launch, the smaller of the two launches (one launch for
the `e7x` rows, marked *); "rest" is the evaluation less the M2L summed over the levels.

| ε | workload | kD | evaluate Rot. | Level | Leaves | M2l | rest Rot. | Level | Leaves | M2l | M2L Rot. | Level | Leaves | M2l | build Rot. | Level | Leaves | M2l |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-7 | W1 | 10 | 0.010 | 1.12 | 1.04 | 1.04 | 0.003 | 1.36 | 1.20 | 1.08 | 0.007 | 0.96 | 0.98 | 1.00 | 0.010 | 9.00 | 9.00 | 9.00 |
| 1e-7 | W1 | 30 | 0.076 | 0.99 | 0.94 | 0.95 | 0.011 | 1.62 | 1.47 | 1.21 | 0.065 | 0.89 | 0.86 | 0.91 | 0.030 | 6.33 | 6.33 | 6.33 |
| 1e-7 | W1 | 100 | 1.591 | 0.95 | 0.96 | 0.83 | 0.323 | 1.79 | 1.83 | 1.14 | 1.262 | 0.74 | 0.74 | 0.75 | 0.220 | 2.95 | 2.95 | 2.91 |
| 1e-7 | W3 (N = 108849) | 30 | 1.204 | 0.52 | 0.49 | 0.48 | 0.093 | 1.75 | 1.56 | 1.27 | 1.102 | 0.42 | 0.41 | 0.40 | 0.160 | 2.50 | 2.44 | 2.44 |
| 1e-7 | W3 (N = 4032) | 10 | 0.068 | 1.70 | 1.69 | 1.13 | 0.049 | 1.84 | 1.82 | 1.06 | 0.019 | 1.33 | 1.32 | 1.32 | 0.020 | 8.00 | 8.00 | 8.50 |
| 1e-7 | W4 clusters (N = 100000) | 1 | 2.482 | 0.90 | 0.89 | 0.69 | 0.693 | 1.88 | 1.85 | 1.14 | 1.789 | 0.52 | 0.52 | 0.52 | 0.270 | 2.63 | 2.56 | 2.52 |
| 1e-7 | W4 clusters (N = 100000) | 10 * | 2.503 | 0.87 | 0.87 | 0.67 | 0.694 | 1.86 | 1.84 | 1.10 | 1.809 | 0.49 | 0.50 | 0.50 | 0.280 | 2.61 | 2.50 | 2.54 |
| 1e-7 | W4 clusters (N = 100000) | 100 | 2.510 | 0.87 | 0.88 | 0.67 | 0.703 | 1.84 | 1.84 | 1.10 | 1.801 | 0.50 | 0.51 | 0.49 | 0.260 | 3.00 | 3.04 | 2.96 |
| 1e-7 | W4 cube (N = 100000) | 1 | 1.199 | 0.48 | 0.46 | 0.43 | 0.094 | 1.66 | 1.48 | 1.20 | 1.105 | 0.37 | 0.37 | 0.37 | 0.150 | 2.27 | 2.33 | 2.27 |
| 1e-7 | W4 cube (N = 100000) | 10 * | 1.196 | 0.48 | 0.46 | 0.43 | 0.091 | 1.88 | 1.51 | 1.23 | 1.105 | 0.36 | 0.37 | 0.36 | 0.170 | 2.12 | 2.12 | 2.12 |
| 1e-7 | W4 cube (N = 100000) | 100 | 1.282 | 0.61 | 0.59 | 0.58 | 0.097 | 1.68 | 1.45 | 1.35 | 1.185 | 0.52 | 0.52 | 0.51 | 0.180 | 3.06 | 3.11 | 3.00 |
| 1e-7 | W4 plummer (N = 100000) | 1 * | 2.085 | 0.91 | 0.91 | 0.66 | 0.682 | 1.83 | 1.83 | 1.06 | 1.403 | 0.46 | 0.47 | 0.46 | 0.220 | 3.05 | 2.86 | 2.82 |
| 1e-7 | W4 plummer (N = 100000) | 10 * | 2.069 | 0.91 | 0.91 | 0.66 | 0.678 | 1.86 | 1.89 | 1.08 | 1.391 | 0.44 | 0.44 | 0.45 | 0.200 | 3.30 | 3.20 | 3.30 |
| 1e-7 | W4 plummer (N = 100000) | 100 * | 2.681 | 0.85 | 0.83 | 0.64 | 0.656 | 1.92 | 1.86 | 1.13 | 2.025 | 0.51 | 0.49 | 0.48 | 0.320 | 2.81 | 2.88 | 2.84 |
| 1e-7 | W4 sphere (N = 100000) | 1 * | 1.637 | 0.84 | 0.82 | 0.71 | 0.311 | 1.94 | 1.82 | 1.22 | 1.325 | 0.58 | 0.58 | 0.58 | 0.220 | 2.14 | 2.09 | 2.09 |
| 1e-7 | W4 sphere (N = 100000) | 10 * | 1.621 | 0.84 | 0.82 | 0.71 | 0.301 | 1.99 | 1.90 | 1.27 | 1.319 | 0.57 | 0.58 | 0.58 | 0.210 | 2.24 | 2.29 | 2.29 |
| 1e-7 | W4 sphere (N = 100000) | 100 * | 1.713 | 0.92 | 0.91 | 0.79 | 0.313 | 1.86 | 1.87 | 1.13 | 1.400 | 0.71 | 0.70 | 0.72 | 0.250 | 2.68 | 2.68 | 2.72 |
| 1e-8 | W1 | 10 | 0.015 | 0.84 | 0.80 | 0.75 | 0.003 | 1.57 | 1.25 | 1.10 | 0.012 | 0.65 | 0.65 | 0.65 | 0.020 | 5.50 | 5.50 | 5.50 |
| 1e-8 | W1 | 30 | 0.124 | 0.71 | 0.67 | 0.65 | 0.014 | 1.85 | 1.50 | 1.29 | 0.110 | 0.57 | 0.55 | 0.57 | 0.030 | 7.00 | 7.00 | 7.00 |
| 1e-8 | W1 | 100 | 2.510 | 0.71 | 0.69 | 0.61 | 0.464 | 1.68 | 1.56 | 1.10 | 2.032 | 0.49 | 0.49 | 0.50 | 0.230 | 2.96 | 3.00 | 2.96 |
| 1e-8 | W3 (N = 108849) | 30 | 1.998 | 0.35 | 0.34 | 0.33 | 0.133 | 1.55 | 1.57 | 1.14 | 1.861 | 0.26 | 0.26 | 0.27 | 0.160 | 2.62 | 2.62 | 2.62 |
| 1e-8 | W3 (N = 4032) | 10 | 0.100 | 1.39 | 1.43 | 0.99 | 0.070 | 1.63 | 1.65 | 1.03 | 0.030 | 0.83 | 0.90 | 0.89 | 0.030 | 6.00 | 6.00 | 6.33 |
| 1e-8 | W4 clusters (N = 100000) | 1 | 3.994 | 0.68 | 0.68 | 0.52 | 0.982 | 1.72 | 1.68 | 1.08 | 3.012 | 0.34 | 0.33 | 0.33 | 0.280 | 2.79 | 2.79 | 2.82 |
| 1e-8 | W4 clusters (N = 100000) | 10 | 3.969 | 0.67 | 0.65 | 0.51 | 0.932 | 1.81 | 1.76 | 1.11 | 2.995 | 0.32 | 0.32 | 0.32 | 0.280 | 2.71 | 2.71 | 2.71 |
| 1e-8 | W4 clusters (N = 100000) | 100 | 4.008 | 0.68 | 0.66 | 0.51 | 1.001 | 1.71 | 1.67 | 1.04 | 3.008 | 0.33 | 0.33 | 0.33 | 0.280 | 3.07 | 3.07 | 3.04 |
| 1e-8 | W4 cube (N = 100000) | 1 | 1.982 | 0.35 | 0.33 | 0.30 | 0.126 | 1.77 | 1.58 | 1.20 | 1.856 | 0.25 | 0.24 | 0.24 | 0.180 | 2.06 | 2.06 | 2.06 |
| 1e-8 | W4 cube (N = 100000) | 10 | 1.993 | 0.32 | 0.31 | 0.30 | 0.122 | 1.65 | 1.46 | 1.25 | 1.866 | 0.24 | 0.24 | 0.24 | 0.160 | 2.44 | 2.38 | 2.38 |
| 1e-8 | W4 cube (N = 100000) | 100 | 2.021 | 0.43 | 0.41 | 0.40 | 0.125 | 1.64 | 1.44 | 1.19 | 1.886 | 0.35 | 0.35 | 0.35 | 0.190 | 2.95 | 2.95 | 3.00 |
| 1e-8 | W4 plummer (N = 100000) | 1 | 3.295 | 0.71 | 0.71 | 0.52 | 0.953 | 1.70 | 1.71 | 1.06 | 2.342 | 0.30 | 0.30 | 0.30 | 0.220 | 3.14 | 3.23 | 3.23 |
| 1e-8 | W4 plummer (N = 100000) | 10 | 3.275 | 0.72 | 0.69 | 0.51 | 0.937 | 1.79 | 1.69 | 1.08 | 2.334 | 0.29 | 0.29 | 0.28 | 0.230 | 3.09 | 3.13 | 3.13 |
| 1e-8 | W4 plummer (N = 100000) | 100 | 4.231 | 0.62 | 0.61 | 0.48 | 0.929 | 1.70 | 1.64 | 1.06 | 3.298 | 0.32 | 0.32 | 0.32 | 0.300 | 3.17 | 3.13 | 3.13 |
| 1e-8 | W4 sphere (N = 100000) | 1 | 2.680 | 0.60 | 0.58 | 0.51 | 0.440 | 1.69 | 1.65 | 1.17 | 2.229 | 0.38 | 0.37 | 0.38 | 0.220 | 2.27 | 2.23 | 2.27 |
| 1e-8 | W4 sphere (N = 100000) | 10 | 2.673 | 0.61 | 0.58 | 0.49 | 0.454 | 1.69 | 1.53 | 1.04 | 2.219 | 0.38 | 0.37 | 0.38 | 0.220 | 2.32 | 2.32 | 2.32 |
| 1e-8 | W4 sphere (N = 100000) | 100 | 2.686 | 0.69 | 0.69 | 0.59 | 0.433 | 1.79 | 1.73 | 1.13 | 2.252 | 0.47 | 0.49 | 0.48 | 0.250 | 2.88 | 2.80 | 2.80 |

Ranges, each form's over Rotation's (or `Level`'s), the cases run in both orders:

| ε | measure | form | ÷ Rotation (or Level), smallest | largest |
| --- | --- | --- | ---: | ---: |
| 1e-7 | M2L | Leaves | 0.370 | 1.323 |
| 1e-7 | M2L | Level | 0.368 | 1.332 |
| 1e-7 | M2L | M2l | 0.365 | 1.318 |
| 1e-7 | build | Leaves | 2.333 | 9.000 |
| 1e-7 | build | Level | 2.267 | 9.000 |
| 1e-7 | build | M2l | 2.267 | 9.000 |
| 1e-7 | evaluate | Leaves | 0.458 | 1.691 |
| 1e-7 | evaluate | Level | 0.475 | 1.701 |
| 1e-7 | evaluate | M2l | 0.430 | 1.129 |
| 1e-7 | evaluate ÷ Level's | Leaves | 0.929 | 1.010 |
| 1e-7 | evaluate ÷ Level's | M2l | 0.663 | 0.957 |
| 1e-7 | rest | Leaves | 1.198 | 1.849 |
| 1e-7 | rest | Level | 1.359 | 1.884 |
| 1e-7 | rest | M2l | 1.056 | 1.350 |
| 1e-7 | φ | Leaves | 0.004 | 0.808 |
| 1e-7 | φ | Level | 0.004 | 0.803 |
| 1e-7 | φ | M2l | 0.112 | 1.000 |
| 1e-7 | ∇φ | Leaves | 0.004 | 0.573 |
| 1e-7 | ∇φ | Level | 0.004 | 0.544 |
| 1e-7 | ∇φ | M2l | 0.317 | 1.000 |
| 1e-8 | M2L | Leaves | 0.238 | 0.895 |
| 1e-8 | M2L | Level | 0.236 | 0.833 |
| 1e-8 | M2L | M2l | 0.235 | 0.885 |
| 1e-8 | build | Leaves | 2.056 | 7.000 |
| 1e-8 | build | Level | 2.056 | 7.000 |
| 1e-8 | build | M2l | 2.056 | 7.000 |
| 1e-8 | evaluate | Leaves | 0.312 | 1.428 |
| 1e-8 | evaluate | Level | 0.322 | 1.391 |
| 1e-8 | evaluate | M2l | 0.297 | 0.988 |
| 1e-8 | evaluate ÷ Level's | Leaves | 0.941 | 1.027 |
| 1e-8 | evaluate ÷ Level's | M2l | 0.710 | 0.943 |
| 1e-8 | rest | Leaves | 1.251 | 1.756 |
| 1e-8 | rest | Level | 1.550 | 1.845 |
| 1e-8 | rest | M2l | 1.027 | 1.295 |
| 1e-8 | φ | Leaves | 0.003 | 0.492 |
| 1e-8 | φ | Level | 0.003 | 0.483 |
| 1e-8 | φ | M2l | 0.061 | 1.525 |
| 1e-8 | ∇φ | Leaves | 0.002 | 0.309 |
| 1e-8 | ∇φ | Level | 0.002 | 0.317 |
| 1e-8 | ∇φ | M2l | 0.187 | 1.048 |

- **The rest** (leaves, P2P, M2M, L2L): `Level` 1.36–1.88× Rotation's at 10⁻⁷ and 1.55–1.85×
  at 10⁻⁸, `Leaves` 1.20–1.85× and 1.25–1.76×, `M2l` 1.06–1.35× and 1.03–1.30×. The leaves'
  operators at p_w are most of the raise; the translations to and from the parent, which
  `Leaves` saves, a small part of it.
- **The evaluation, `M2l`**: at 10⁻⁷ on the cases above 0.1 s 0.43–0.83× Rotation's (W1 at
  kD = 100 0.83, the clusters 0.67–0.69, the cube 0.43–0.58, W3 0.48; the `e7x` cases 0.43–0.79,
  one launch), on the ms cases 0.95–1.13× (W1 at kD = 10 and 30, W3 at N = 4,032); at 10⁻⁸
  0.30–0.65× above 0.1 s. `Level`'s at 10⁻⁷: 0.48–0.95× above 0.1 s in this job (CH6.1g's job
  632472 had 0.54–1.05×: the clusters 1.00 there, 0.87–0.90 here; between jobs the same case
  moves by up to about 10 %, within a job the forms' order is reversed, A B C D D C B A).
- **The M2L** is the same in all three forms (0.37–1.32× Rotation's at 10⁻⁷, 0.24–0.89× at
  10⁻⁸), as are the build (2.1–9× Rotation's: the RCR and plane-wave tables at p_w; under 1 s)
  and the stores. Peak resident memory at most 1.4 GB.

## 5. Questions for sign-off

1. **The form by ε (recommended):** `PlaneWaveRaise` by the rule, a function of ε only:
   **`M2l` at 10⁻⁷ ≤ ε < 10⁻⁶, `Leaves` at 10⁻⁸ ≤ ε < 10⁻⁷**, built as the default (a new
   variant, the rule's; the named forms stay for the measurements). At 10⁻⁷ it holds on all 17
   cases (0.11–1.00×) and takes 0.43–0.83× Rotation's evaluation above 0.1 s; its margin is
   thin where a raised level is all leaves (1.00×), so a tree not measured could exceed 1.1×. At
   10⁻⁸ `M2l` fails once (1.53×) and `Leaves` holds as `Level` does, 0.94–1.03× its time.
   Alternatives: `Leaves` in both bands (`Level`'s margin, at most 0.81×; 0.93–1.03× `Level`'s
   time); `Level` in both (CH6.1g as merged).
2. **D8's bound at 10⁻⁷ ≤ ε < 10⁻⁶ (recommended: lift it):** with question 1's form, `Auto`
   takes the exponential M2L at 10⁻⁷ ≤ ε < 10⁻⁶ too (the evaluation 0.43–0.83× Rotation's above
   0.1 s; three ms cases 0.95–1.13×). Alternative: keep Rotation there (D8 as amended
   2026-11-11). If question 1 keeps `Level` or takes `Leaves` there, the bound's case is CH6.1g's
   (above 0.1 s 0.48–0.95× under `Level` and 0.46–0.96× under `Leaves` here, up to 1.05× in
   CH6.1g's job) and keeping it is recommended.

## 6. Tests

- nd-fmm-tables `tests/helmholtz/main.rs`: `m2m_and_l2l_on_the_first_degrees_of_raised_levels`
  (M2M and L2L between every pair of levels on the first degrees against `rcr`, the degrees
  above NaN or untouched, five wavenumbers; M2L on every offset of levels 2–4 bit for bit the
  tables built at the raised degrees), `build_is_the_m2l_degrees_at_the_degrees`,
  `m2l_degrees_below_the_degrees_are_refused`.
- nd-fmm-exec `tests/helmholtz/plane_wave.rs`,
  `the_raised_levels_translations_keep_the_degrees_given`: per form the degrees, translation
  degrees, store sizes and tables per level, the M2L pair body `Level`'s bits, a leaf's P2M and
  L2P (`Level`'s under `Leaves`; Rotation's at p_l on the first degrees under `M2l`, the others
  zero), the order of the calls, Rotation afterwards, Auto, nothing raised at 10⁻⁶.
- nd-fmm-exec `tests/helmholtz_fmm.rs`, `check_m2l_raise` in `check_resolving_degree`: both forms
  on the clusters at kD = 1 and the cube at kD = 20 + 2i at 10⁻⁷ and 10⁻⁸, within
  `PLANE_WAVE_RATIO` of Rotation's (`Leaves` 0.003–0.011×, `M2l` 0.56–1.04×), 2 threads and on P
  ranks the one-rank FMM bit for bit; `Leaves` bit for bit `Level`'s where only level 2 is
  raised, both differing from it elsewhere.
- nd-fmm-validate `tests/helmholtz_accuracy.rs`: both forms on W1 at kD = 10, 10⁻⁸, within 1.1×
  Rotation's, the row's form and translation degrees.

## 7. Reproducing

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch6-1h-runs.sh <out-dir> e7 e8            # on Kathleen through run.sbatch
ONLY=-forward tools/helmholtz/ch6-1h-runs.sh <out-dir> e7x
python3 tools/helmholtz/ch6-1h-tables.py <out-dir>
```

Raw outputs: `~/projects/fmm-archive/kathleen/logs/ch6-1h/` (28 launches, `runs.log`, the jobs'
`fmm-run-632717.out` and `fmm-run-632725.out`).
