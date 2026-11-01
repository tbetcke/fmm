# Helmholtz CH3.3: the calibration of p_l over (k a_l, ε)

Helmholtz task CH3.3 ([docs/helmholtz/tasks/CH3.3.md](../../docs/helmholtz/tasks/CH3.3.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§3.5, §9.1 and §10 CH3.3; spike S7
([spikes/truncation/SPIKE_REPORT.md](../../spikes/truncation/SPIKE_REPORT.md)). Measured on
2026-11-01 on the M6 (Mac mini, Apple M6, 24 GB; [tools/m6/machine.md](../../tools/m6/machine.md))
from branch `tbetcke/h-ch3-3`, release build, rustc 1.99.0, outside the sandbox, BLAS
threads 1, `HWLOC_COMPONENTS=-darwin`, one rank × 6 threads (the Super and Performance
cores) unless stated, the P2P kernel `Auto` (NEON). The errors are deterministic; the
times are indicative only (the M6's launches vary by up to 25 %, CH3.4c), and nothing here
is timed for a gate.

## Summary

- **S7's tables are confirmed as upper bounds in the full FMM.** At each of the 176
  measured (level, ε) points (plateau and level scans on W1, W3 and W4, 0.14 ≤ x ≤ 65,
  10⁻⁸ ≤ ε ≤ 10⁻³, the plateau to 10⁻¹⁰), the smallest degree that meets ε/2 is at most
  S7's: 9 to 32 degrees below S7's closed form on the low-frequency levels (x < 4) and 4
  to 30 below it on the others, and below S7's tabled p* wherever its grid of k a brackets
  the point (at k a = 1, p* is 12, 17, 23, 29, 35, 42 at 10⁻³ … 10⁻⁸; the plateau 6 to 24;
  at k a = 64, 75 … 90; x = 65 needs 69 … 86). The full FMM at S7's degrees themselves is
  CH3.1c's matrix (errors 10³ to 10⁶ below ε) and §5's `--rule s7` rows. That is the
  acceptance, "S7's low-frequency tables confirmed in the full FMM".
- **The full FMM needs about half S7's low-frequency degree.** The measured plateau (every
  point with x ≤ 4) is 7, 9, 12, 15, 20 and 24 at ε = 10⁻³ … 10⁻⁸ (30 and 36 at 10⁻⁹ and
  10⁻¹⁰), against S7's 16 … 46; the plateau scans alone give 6 at 10⁻³, W1's level 5 at
  kD = 100 (x = 2.7) needs 7. It is
  convex in L = log₁₀(1/ε) (3 degrees per decade at 10⁻³, 6 at 10⁻¹⁰), so the fitted
  plateau is quadratic in L. The Gaussian clusters set it at small ε, the cube, the
  Plummer sphere and W1's deeper levels at large ε. On the high-frequency levels the
  excess over x is 0.2 to 0.8 of S7's (ln 1/ε)^{2/3} x^{1/3}.
- **The calibrated rule is the new default** (`nd_fmm_exec::helmholtz::degree`,
  `DegreeRule::CALIBRATED`; the user's decision of 2026-11-01):

  ```text
  p(x, ε) = max(⌈0.30 L² + 0.25 L + 2.60⌉, ⌈x + 0.80 (ln 1/ε)^{2/3} x^{1/3}⌉),
  x = Re(k) a_l, L = log₁₀(1/ε).
  ```

  It is at least every measured degree (0 to 8 above it in D1's range) and at most S7's
  closed form for 10⁻¹⁸ ≤ ε ≤ 10⁻¹. S7's closed form stays public as `s7_degree`.
  Above the measured x = 65 the high-frequency branch is an extrapolation, checked only on
  W1 at kD = 500 (x = 108), the tightest case of the verification (0.63 ε at 10⁻⁸); there
  it falls below S7's per-pair p* at small ε (§6, question 2 of the PR).
- **Verified against ε in the full FMM**: CH3.1c's whole matrix (62 cases: its 61 and W3 at
  kD = 100 and ε = 10⁻⁸, which now fits in memory) and 26 held-out cases (W2, W1 at kD = 50 and
  150, the W4 cube and Plummer sphere at kD = 3 and 30, W1 at ε = 10⁻⁴ and 10⁻⁷): every
  error of φ and ∇φ within ε, the largest 0.63 ε (∇φ on W1 at kD = 500, ε = 10⁻⁸).
  Deterministic: equal checksums at 1 and 6 threads on 15 cases, and on 1, 2 and 3 ranks on
  two.
- **Faster**: 2.7 times at ε = 10⁻³ and 6 to 8 times at 10⁻⁶ and 10⁻⁸ (W1 at kD = 100,
  W3 at kD = 30, the Plummer sphere at kD = 10), and 4.9 times on W1 at kD = 500 and 10⁻⁶
  (98 s against 478 s). The errors are now 0.1 to 0.6 ε, where S7's degrees left them
  10⁻³ to 10⁻⁶ ε.

## 1. Method

The calibration is `nd_fmm_validate::helmholtz_calibration` (the search and the fit) driven
by the `helmholtz_calibrate` example, on CH3.1c's harness (`helmholtz_accuracy`: the
workloads of `helmholtz_workloads` with D = 1, §9.1's complex charges, the FMM against the
direct sum in f64 at 1,000 sampled targets, the relative ℓ2 errors of φ and ∇φ).

- **The measured degree.** Per scan the FMM runs at consecutive degrees, all else fixed,
  and the degree at a target is the smallest p whose larger error (φ or ∇φ) meets the
  target at p, p + 1 and p + 2 (S7's rule against a lucky degree). The target is **ε/2**
  (decision of 2026-11-01, "ε/2, verify ≤ ε"): a target's error sums every level.
- **Plateau scans** (stage 1): one degree on every level ≥ 2 of trees whose levels all
  lie at x = Re(k) a_l ≤ 2.2 (a_l = √3 r_l the circumradius in the FMM's domain, the
  points' bounding cube): W4's four distributions at N = 10⁵ and kD = 1 and 10, W1 and W3
  (ten points per wavelength) at kD = 10; ε = 10⁻³ … 10⁻¹⁰.
- **Level scans** (stage 2): one level's degree, every other level held at the stage-1
  plateau at ε = 10⁻¹⁰ with S7's high-frequency branch (γ = 0.29, α = 0.45, β = 1.65,
  C = 1 at 10⁻¹⁰: degree 36 on low levels), whose errors lie about 100 times below the
  smallest target, so the scanned level alone sets the error. W1 at kD = 30 and 100 (every
  level), W3 at kD = 30, W4's cube at kD = 100 (every level) and clusters at kD = 100
  (x ≥ 4), and the top levels of W1 at kD = 200 (x = 43) and 300 (x = 65); ε = 10⁻³ …
  10⁻⁸ (D1's range). Each scan starts at ⌊x⌋ − 2 (or a named degree near the threshold
  at kD ≥ 200).
- **The fit** (`fit`): a rule of S7's shape with a quadratic plateau,
  p(x, ε) = max(⌈γ L² + α L + β⌉, ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉), L = log₁₀(1/ε);
  γ, α, β on grids (0.01, 0.05, 0.05) with the least total excess over every point with
  x ≤ 4 (plateau and level scans), then C the smallest multiple of 0.05 that covers every
  other point the plateau misses. The rule is at least every measured degree.
- **Verification** (stage 3): the rule as `Degrees::Tolerance`'s default, against ε
  (not ε/2) on CH3.1c's whole matrix and on held-out cases that the fit never saw.

## 2. The plateau

One degree on every level ≥ 2 (stage 1). Each cell is the measured degree at ε/2, then
(the calibrated rule / S7's closed form) at the tree's largest x.

| workload | kD | level | x | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | 10⁻⁸ | 10⁻⁹ | 10⁻¹⁰ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W4 clusters | 1 | 2 | 0.14 | 6 (7 / 16) | 8 (9 / 22) | 12 (12 / 28) | 15 (15 / 34) | 20 (20 / 40) | 24 (24 / 46) | 30 (30 / 52) | 36 (36 / 58) |
| W4 plummer | 1 | 2 | 0.21 | 6 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 15 (15 / 34) | 19 (20 / 40) | 23 (24 / 46) | 29 (30 / 52) | 34 (36 / 58) |
| W4 sphere | 1 | 2 | 0.22 | 4 (7 / 16) | 7 (9 / 22) | 9 (12 / 28) | 12 (15 / 34) | 15 (20 / 40) | 19 (24 / 46) | 22 (30 / 52) | 26 (36 / 58) |
| W4 cube | 1 | 2 | 0.22 | 5 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 15 (15 / 34) | 18 (20 / 40) | 23 (24 / 46) | 27 (30 / 52) | 33 (36 / 58) |
| W4 clusters | 10 | 2 | 1.36 | 6 (7 / 16) | 8 (9 / 22) | 12 (12 / 28) | 15 (15 / 34) | 20 (20 / 40) | 24 (24 / 46) | 30 (30 / 52) | 36 (36 / 58) |
| W4 plummer | 10 | 2 | 2.13 | 6 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 15 (15 / 34) | 19 (20 / 40) | 23 (24 / 46) | 29 (30 / 52) | 34 (36 / 58) |
| W1 | 10 | 2 | 2.16 | 6 (7 / 16) | 9 (9 / 22) | 12 (12 / 28) | 15 (15 / 34) | 19 (20 / 40) | 22 (24 / 46) | 25 (30 / 52) | 29 (36 / 58) |
| W3 (N = 4032) | 10 | 2 | 2.16 | 6 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 14 (15 / 34) | 17 (20 / 40) | 21 (24 / 46) | 26 (30 / 52) | 30 (36 / 58) |
| W4 sphere | 10 | 2 | 2.17 | 5 (7 / 16) | 7 (9 / 22) | 10 (12 / 28) | 12 (15 / 34) | 15 (20 / 40) | 19 (24 / 46) | 22 (30 / 52) | 26 (36 / 58) |
| W4 cube | 10 | 2 | 2.17 | 6 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 15 (15 / 34) | 18 (20 / 40) | 23 (24 / 46) | 27 (30 / 52) | 33 (36 / 58) |

The ten trees agree within 2 degrees at ε ≥ 10⁻⁶ and spread to 10 at 10⁻¹⁰. The sphere's
surface needs the fewest, the clusters the most. The clusters' errors still fall by a
factor of 1.5 per degree at p = 38, so this is their convergence rate, not a rounding
floor. Below x ≈ 2 the plateau does not depend on k: kD = 1 and 10 give the same degrees
on every distribution.

## 3. The level scans

One level's degree, the others held (§1). Each cell is as in §2, at the level's x. The
plateau scans and the levels with x ≤ 4 set γ, α and β; the rest set C.

| workload | kD | level | x | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ | 10⁻⁷ | 10⁻⁸ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W3 (N = 108849) | 30 | 4 | 1.62 | 6 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 14 (15 / 34) | 18 (20 / 40) | 22 (24 / 46) |
| W1 | 100 | 5 | 2.71 | 7 (7 / 16) | 8 (9 / 22) | 11 (12 / 28) | 14 (15 / 34) | 17 (20 / 40) | 20 (24 / 46) |
| W1 | 30 | 3 | 3.25 | 7 (8 / 16) | 9 (9 / 22) | 11 (12 / 28) | 14 (15 / 34) | 17 (20 / 40) | 20 (24 / 46) |
| W3 (N = 108849) | 30 | 3 | 3.25 | 7 (8 / 16) | 9 (9 / 22) | 11 (12 / 28) | 14 (15 / 34) | 17 (20 / 40) | 22 (24 / 46) |
| W1 | 100 | 4 | 5.41 | 9 (11 / 16) | 11 (12 / 22) | 13 (13 / 28) | 15 (15 / 34) | 18 (20 / 40) | 21 (24 / 46) |
| W4 cube | 100 | 4 | 5.41 | 10 (11 / 16) | 11 (12 / 22) | 13 (13 / 28) | 15 (15 / 34) | 18 (20 / 40) | 21 (24 / 46) |
| W1 | 30 | 2 | 6.49 | 10 (12 / 16) | 12 (14 / 22) | 14 (15 / 28) | 16 (16 / 34) | 19 (20 / 40) | 22 (24 / 46) |
| W3 (N = 108849) | 30 | 2 | 6.50 | 10 (12 / 16) | 12 (14 / 22) | 14 (15 / 28) | 16 (16 / 34) | 19 (20 / 40) | 22 (24 / 46) |
| W4 clusters | 100 | 3 | 6.79 | 8 (13 / 16) | 10 (14 / 22) | 11 (15 / 28) | 13 (16 / 34) | 14 (20 / 40) | 16 (24 / 46) |
| W1 | 100 | 3 | 10.83 | 15 (18 / 19) | 17 (19 / 22) | 19 (20 / 28) | 20 (22 / 34) | 22 (23 / 40) | 24 (24 / 46) |
| W4 cube | 100 | 3 | 10.83 | 15 (18 / 19) | 17 (19 / 22) | 19 (20 / 28) | 21 (22 / 34) | 23 (23 / 40) | 24 (24 / 46) |
| W4 clusters | 100 | 2 | 13.57 | 13 (21 / 23) | 16 (22 / 25) | 18 (24 / 28) | 20 (25 / 34) | 22 (26 / 40) | 24 (27 / 46) |
| W1 | 100 | 2 | 21.65 | 25 (30 / 32) | 28 (32 / 34) | 31 (34 / 36) | 33 (35 / 38) | 35 (36 / 40) | 37 (38 / 46) |
| W4 cube | 100 | 2 | 21.65 | 26 (30 / 32) | 28 (32 / 34) | 31 (34 / 36) | 33 (35 / 38) | 35 (36 / 40) | 37 (38 / 46) |
| W1 | 200 | 2 | 43.30 | 47 (54 / 57) | 51 (56 / 59) | 54 (58 / 62) | 57 (60 / 64) | 60 (62 / 66) | 62 (63 / 68) |
| W1 | 300 | 2 | 64.95 | 69 (77 / 80) | 73 (80 / 83) | 77 (82 / 86) | 80 (84 / 89) | 83 (86 / 91) | 86 (88 / 93) |

W1 at kD = 300, level 3 (x = 32.5), was stopped after two degrees to save time: p = 36 met
5 × 10⁻⁴, and the rule gives 42 at 10⁻³. That x lies between the measured 21.7 and 43.3.
The levels at x ≈ 5–7 need the high-frequency branch above the plateau at ε ≥ 10⁻⁵ (10
against 7 at 10⁻³): this is S7's crossing region. The clusters' level 3 (x = 6.8) needs
less than their own plateau, because their boxes on level 3 are sparse.

On the top levels the excess p − x grows about linearly in L (about 3 degrees per decade
at x = 22 to 65), faster than (ln 1/ε)^{2/3}. One C for D1's range therefore
over-predicts at 10⁻³ by up to 8 degrees (x = 65: 77 against 69) and fits at 10⁻⁸ (88
against 86). A branch x + (a L + b) x^{1/3} would fit better (§6).

## 4. The rule

`helmholtz_calibrate fit` over the 176 points (104 of them with x ≤ 4) gives γ = 0.30,
α = 0.25, β = 2.60 and C = 0.80. Its excess over the measured degree, by x:

| x | points | rule − measured | S7 − measured |
| --- | ---: | --- | --- |
| < 4 | 104 | 0 to 10 (10 only at 10⁻¹⁰, on the sphere) | 9 to 32 |
| 4 to 8 | 30 | 0 to 8 | 6 to 30 |
| 8 to 30 | 30 | 0 to 8 | 4 to 22 |
| 30 to 65 | 12 | 1 to 8 | 6 to 11 |

The plateau moved by at most one degree between the first fit (the plateau scans alone:
γ = 0.29, α = 0.45, β = 1.65) and the final one. W1's level 5 at kD = 100 (x = 2.7)
raised the plateau at 10⁻³ from 6 to 7. Between the decades of ε and between the measured
x the rule is an interpolation. Above x = 65 it is an extrapolation (W1's level 2 at
kD = 500 has x = 108), which §5 checks at kD = 500.

## 5. Verification against ε

The rule as `Degrees::Tolerance(ε)` in CH3.1c's `helmholtz_accuracy` example, the errors
measured against ε itself, 1 rank × 6 threads. The variants are those of CH3.1c's §3.

### CH3.1c's matrix

| Workload | N | kD | ε | variant | p by level | evaluate s | φ | ∇φ | max / ε |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| W1 | 796 | 10 | 1e-3 | – | 0×2, 7 | 0.00 | 4.40e-5 | 1.53e-4 | 0.15 |
| W1 | 796 | 10 | 1e-6 | – | 0×2, 15 | 0.01 | 5.62e-8 | 4.11e-7 | 0.41 |
| W1 | 796 | 10 | 1e-8 | – | 0×2, 24 | 0.02 | 7.81e-11 | 9.22e-10 | 0.09 |
| W1 | 7162 | 30 | 1e-3 | – | 0×2, 12, 8 | 0.02 | 2.66e-5 | 7.15e-5 | 0.07 |
| W1 | 7162 | 30 | 1e-6 | – | 0×2, 16, 15 | 0.06 | 1.12e-7 | 2.97e-7 | 0.30 |
| W1 | 7162 | 30 | 1e-8 | – | 0×2, 24×2 | 0.20 | 2.62e-10 | 8.32e-10 | 0.08 |
| W1 | 79578 | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7 | 0.59 | 4.15e-5 | 1.36e-4 | 0.14 |
| W1 | 79578 | 100 | 1e-6 | – | 0×2, 35, 22, 15×2 | 1.47 | 9.61e-8 | 2.94e-7 | 0.29 |
| W1 | 79578 | 100 | 1e-8 | – | 0×2, 38, 24×3 | 4.11 | 1.39e-9 | 3.40e-9 | 0.34 |
| W3 (N = 4032) | 4032 | 10 | 1e-3 | – | 0×2, 7×2 | 0.01 | 4.05e-5 | 8.10e-5 | 0.08 |
| W3 (N = 4032) | 4032 | 10 | 1e-6 | – | 0×2, 15×2 | 0.04 | 4.49e-8 | 1.70e-7 | 0.17 |
| W3 (N = 4032) | 4032 | 10 | 1e-8 | – | 0×2, 24×2 | 0.11 | 9.83e-11 | 8.92e-10 | 0.09 |
| W3 (N = 108849) | 108849 | 30 | 1e-3 | – | 0×2, 12, 8, 7 | 0.26 | 4.60e-5 | 1.31e-4 | 0.13 |
| W3 (N = 108849) | 108849 | 30 | 1e-6 | – | 0×2, 16, 15×2 | 1.05 | 1.46e-7 | 4.17e-7 | 0.42 |
| W3 (N = 108849) | 108849 | 30 | 1e-8 | – | 0×2, 24×3 | 3.39 | 7.34e-10 | 2.96e-9 | 0.30 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7×2 | 16.17 | 4.86e-5 | 1.08e-4 | 0.11 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-6 | – | 0×2, 35, 22, 15×3 | 89.00 | 1.01e-7 | 2.50e-7 | 0.25 |
| W3 (N = 4031442) | 4031442 | 100 | 1e-8 | – | 0×2, 38, 24×4 | 313.56 | 2.60e-9 | 4.10e-9 | 0.41 |
| W4 cube | 100000 | 1 | 1e-3 | – | 0×2, 7×3 | 0.33 | 4.07e-5 | 8.96e-5 | 0.09 |
| W4 cube | 100000 | 1 | 1e-6 | – | 0×2, 15×3 | 1.20 | 6.40e-8 | 2.65e-7 | 0.27 |
| W4 cube | 100000 | 1 | 1e-8 | – | 0×2, 24×3 | 3.88 | 2.57e-10 | 1.73e-9 | 0.17 |
| W4 cube | 100000 | 10 | 1e-3 | – | 0×2, 7×3 | 0.25 | 5.00e-5 | 9.95e-5 | 0.10 |
| W4 cube | 100000 | 10 | 1e-6 | – | 0×2, 15×3 | 1.17 | 6.08e-8 | 2.69e-7 | 0.27 |
| W4 cube | 100000 | 10 | 1e-8 | – | 0×2, 24×3 | 3.88 | 2.29e-10 | 1.71e-9 | 0.17 |
| W4 cube | 100000 | 100 | 1e-3 | – | 0×2, 30, 18, 11 | 0.75 | 1.85e-5 | 4.04e-5 | 0.04 |
| W4 cube | 100000 | 100 | 1e-6 | – | 0×2, 35, 22, 15 | 1.38 | 9.63e-8 | 3.04e-7 | 0.30 |
| W4 cube | 100000 | 100 | 1e-8 | – | 0×2, 38, 24×2 | 4.08 | 2.11e-9 | 4.52e-9 | 0.45 |
| W4 sphere | 100000 | 1 | 1e-3 | – | 0×2, 7×4 | 0.32 | 2.30e-5 | 8.09e-7 | 0.02 |
| W4 sphere | 100000 | 1 | 1e-6 | – | 0×2, 15×4 | 1.53 | 4.24e-8 | 2.36e-9 | 0.04 |
| W4 sphere | 100000 | 1 | 1e-8 | – | 0×2, 24×4 | 5.13 | 1.31e-10 | 1.20e-11 | 0.01 |
| W4 sphere | 100000 | 10 | 1e-3 | – | 0×2, 7×4 | 0.36 | 2.85e-5 | 8.42e-7 | 0.03 |
| W4 sphere | 100000 | 10 | 1e-6 | – | 0×2, 15×4 | 1.58 | 4.36e-8 | 2.40e-9 | 0.04 |
| W4 sphere | 100000 | 10 | 1e-8 | – | 0×2, 24×4 | 5.23 | 1.27e-10 | 1.21e-11 | 0.01 |
| W4 sphere | 100000 | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7 | 0.83 | 3.06e-5 | 2.04e-6 | 0.03 |
| W4 sphere | 100000 | 100 | 1e-6 | – | 0×2, 35, 22, 15×2 | 1.79 | 6.24e-8 | 4.06e-9 | 0.06 |
| W4 sphere | 100000 | 100 | 1e-8 | – | 0×2, 38, 24×3 | 5.21 | 8.86e-10 | 4.86e-11 | 0.09 |
| W4 plummer | 100000 | 1 | 1e-3 | – | 0×2, 7×7 | 0.48 | 6.47e-5 | 9.67e-5 | 0.10 |
| W4 plummer | 100000 | 1 | 1e-6 | – | 0×2, 15×7 | 2.18 | 1.11e-7 | 3.33e-7 | 0.33 |
| W4 plummer | 100000 | 1 | 1e-8 | – | 0×2, 24×7 | 6.40 | 5.90e-10 | 2.41e-9 | 0.24 |
| W4 plummer | 100000 | 10 | 1e-3 | – | 0×2, 7×7 | 0.50 | 5.01e-5 | 9.69e-5 | 0.10 |
| W4 plummer | 100000 | 10 | 1e-6 | – | 0×2, 15×7 | 2.00 | 8.31e-8 | 3.33e-7 | 0.33 |
| W4 plummer | 100000 | 10 | 1e-8 | – | 0×2, 24×7 | 6.05 | 4.34e-10 | 2.41e-9 | 0.24 |
| W4 plummer | 100000 | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7×4 | 0.95 | 7.95e-5 | 1.18e-4 | 0.12 |
| W4 plummer | 100000 | 100 | 1e-6 | – | 0×2, 35, 21, 15×5 | 2.71 | 1.53e-7 | 3.58e-7 | 0.36 |
| W4 plummer | 100000 | 100 | 1e-8 | – | 0×2, 37, 24×6 | 8.18 | 1.27e-9 | 2.59e-9 | 0.26 |
| W4 clusters | 100000 | 1 | 1e-3 | – | 0×2, 7×7 | 0.57 | 5.07e-5 | 1.01e-4 | 0.10 |
| W4 clusters | 100000 | 1 | 1e-6 | – | 0×2, 15×7 | 2.65 | 9.45e-8 | 4.03e-7 | 0.40 |
| W4 clusters | 100000 | 1 | 1e-8 | – | 0×2, 24×7 | 7.58 | 5.37e-10 | 5.00e-9 | 0.50 |
| W4 clusters | 100000 | 10 | 1e-3 | – | 0×2, 7×7 | 0.56 | 5.05e-5 | 1.01e-4 | 0.10 |
| W4 clusters | 100000 | 10 | 1e-6 | – | 0×2, 15×7 | 2.76 | 9.42e-8 | 4.03e-7 | 0.40 |
| W4 clusters | 100000 | 10 | 1e-8 | – | 0×2, 24×7 | 7.99 | 5.35e-10 | 5.00e-9 | 0.50 |
| W4 clusters | 100000 | 100 | 1e-3 | – | 0×2, 21, 13, 8, 7×4 | 0.63 | 5.26e-5 | 1.03e-4 | 0.10 |
| W4 clusters | 100000 | 100 | 1e-6 | – | 0×2, 25, 16, 15×5 | 2.50 | 9.08e-8 | 4.03e-7 | 0.40 |
| W4 clusters | 100000 | 100 | 1e-8 | – | 0×2, 27, 24×6 | 7.87 | 5.01e-10 | 4.98e-9 | 0.50 |
| W1 | 79578 | 100 | 1e-3 | real charges | 0×2, 30, 18, 11, 7 | 0.60 | 4.17e-5 | 1.34e-4 | 0.13 |
| W1 | 79578 | 100 | 1e-6 | real charges | 0×2, 35, 22, 15×2 | 1.44 | 9.92e-8 | 3.00e-7 | 0.30 |
| W1 | 79578 | 100 | 1e-6 | 8 vectors | 0×2, 35, 22, 15×2 | 13.22 | 9.52e-8 | 2.96e-7 | 0.30 |
| W1 | 79578 | 100 + 10i | 1e-3 | complex k | 0×2, 30, 18, 11, 7 | 0.75 | 5.58e-5 | 1.54e-4 | 0.15 |
| W1 | 79578 | 100 + 10i | 1e-6 | complex k | 0×2, 35, 22, 15×2 | 1.69 | 1.53e-7 | 3.87e-7 | 0.39 |
| W1 | 1989437 | 500 | 1e-3 | – | 0×2, 123, 66, 36, 21, 13, 8 | 52.14 | 3.24e-5 | 8.71e-5 | 0.09 |
| W1 | 1989437 | 500 | 1e-6 | – | 0×2, 131, 72, 41, 25, 16, 15 | 97.63 | 1.49e-7 | 3.79e-7 | 0.38 |
| W1 | 1989437 | 500 | 1e-8 | – | 0×2, 135, 76, 44, 27, 24×2 | 201.06 | 2.72e-9 | 6.33e-9 | 0.63 |

### Held out

| Workload | N | kD | ε | variant | p by level | evaluate s | φ | ∇φ | max / ε |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| W2 | 11480 | 30 | 1e-3 | – | 0×2, 12, 8, 7×2 | 0.04 | 4.14e-5 | 1.10e-4 | 0.11 |
| W2 | 11480 | 30 | 1e-6 | – | 0×2, 16, 15×3 | 0.15 | 9.48e-8 | 2.45e-7 | 0.24 |
| W2 | 11480 | 30 | 1e-8 | – | 0×2, 24×4 | 0.40 | 2.09e-10 | 9.37e-10 | 0.09 |
| W2 | 127554 | 100 | 1e-3 | – | 0×2, 30, 18, 11, 7×3 | 0.94 | 4.91e-5 | 1.16e-4 | 0.12 |
| W2 | 127554 | 100 | 1e-6 | – | 0×2, 35, 22, 15×4 | 2.90 | 1.09e-7 | 3.05e-7 | 0.30 |
| W2 | 127554 | 100 | 1e-8 | – | 0×2, 38, 24×5 | 8.65 | 1.42e-9 | 1.58e-9 | 0.16 |
| W1 | 19895 | 50 | 1e-3 | – | 0×2, 18, 11, 7 | 0.10 | 4.78e-5 | 1.60e-4 | 0.16 |
| W1 | 19895 | 50 | 1e-6 | – | 0×2, 22, 15×2 | 0.33 | 9.22e-8 | 3.25e-7 | 0.33 |
| W1 | 19895 | 50 | 1e-8 | – | 0×2, 24×3 | 0.95 | 2.48e-9 | 4.95e-9 | 0.49 |
| W1 | 179050 | 150 | 1e-3 | – | 0×2, 42, 24, 14, 9, 7 | 2.32 | 2.89e-5 | 6.85e-5 | 0.07 |
| W1 | 179050 | 150 | 1e-6 | – | 0×2, 48, 28, 18, 15×2 | 5.19 | 7.31e-8 | 1.84e-7 | 0.18 |
| W1 | 179050 | 150 | 1e-8 | – | 0×2, 51, 31, 24×3 | 14.52 | 1.01e-9 | 1.63e-9 | 0.16 |
| W4 cube | 100000 | 3 | 1e-3 | – | 0×2, 7×3 | 0.26 | 3.81e-5 | 9.01e-5 | 0.09 |
| W4 cube | 100000 | 3 | 1e-6 | – | 0×2, 15×3 | 1.20 | 5.91e-8 | 2.66e-7 | 0.27 |
| W4 cube | 100000 | 3 | 1e-8 | – | 0×2, 24×3 | 3.89 | 2.36e-10 | 1.73e-9 | 0.17 |
| W4 cube | 100000 | 30 | 1e-3 | – | 0×2, 12, 8, 7 | 0.26 | 4.41e-5 | 1.06e-4 | 0.11 |
| W4 cube | 100000 | 30 | 1e-6 | – | 0×2, 16, 15×2 | 1.18 | 1.30e-7 | 3.22e-7 | 0.32 |
| W4 cube | 100000 | 30 | 1e-8 | – | 0×2, 24×3 | 3.93 | 3.23e-10 | 1.62e-9 | 0.16 |
| W4 plummer | 100000 | 3 | 1e-3 | – | 0×2, 7×7 | 0.48 | 6.01e-5 | 9.67e-5 | 0.10 |
| W4 plummer | 100000 | 3 | 1e-6 | – | 0×2, 15×7 | 1.98 | 1.03e-7 | 3.33e-7 | 0.33 |
| W4 plummer | 100000 | 3 | 1e-8 | – | 0×2, 24×7 | 6.05 | 5.46e-10 | 2.41e-9 | 0.24 |
| W4 plummer | 100000 | 30 | 1e-3 | – | 0×2, 12, 8, 7×5 | 0.52 | 3.84e-5 | 9.79e-5 | 0.10 |
| W4 plummer | 100000 | 30 | 1e-6 | – | 0×2, 15×7 | 2.17 | 1.17e-7 | 3.35e-7 | 0.34 |
| W4 plummer | 100000 | 30 | 1e-8 | – | 0×2, 24×7 | 6.03 | 3.26e-10 | 2.40e-9 | 0.24 |
| W1 | 79578 | 100 | 1e-4 | – | 0×2, 32, 19, 12, 9 | 0.88 | 4.58e-6 | 1.70e-5 | 0.17 |
| W1 | 79578 | 100 | 1e-7 | – | 0×2, 36, 23, 20×2 | 3.10 | 9.12e-9 | 1.78e-8 | 0.18 |

All 88 cases are within ε. The tightest are the clusters at 10⁻⁸ (0.50 ε) and W1 at
kD = 500 at 10⁻⁸ (0.63 ε), where the top level (x = 108) is extrapolated. W3 at kD = 100
and 10⁻⁸, left out of CH3.1c for memory, now evaluates in 314 s.

**Determinism.** The checksum of the whole output is the same at 1 and 6 threads on 15
cases (W1 at kD = 10, 30 and 100 and W3 at kD = 10 and 30, at each ε). It is also the same
on W1 at kD = 100, 10⁻⁶ at 1 × 1, 1 × 6 and 3 × 2 ranks × threads (`f2a48c10166612a0`),
and on W3 at kD = 30, 10⁻⁶ at 1 × 1, 1 × 6 and 2 × 3 (`07751832c80b5a9f`).

**Times** (1 × 6). `--rule s7` names S7's degrees per level, found through one run at
degree 1. The calibrated rule (A) and S7's (B) ran in the order A B B A; each time is the
mean of the two runs.

| Case | ε | p by level, calibrated | p by level, S7 | calibrated s | S7 s | ratio |
| --- | --- | --- | --- | ---: | ---: | ---: |
| W1, kD = 100 | 10⁻³ | 0×2, 30, 18, 11, 7 | 0×2, 32, 19, 16×2 | 0.70 | 1.86 | 2.7 |
| W1, kD = 100 | 10⁻⁶ | 0×2, 35, 22, 15×2 | 0×2, 38, 34×3 | 1.68 | 12.68 | 7.5 |
| W1, kD = 100 | 10⁻⁸ | 0×2, 38, 24×3 | 0×2, 46×4 | 4.65 | 29.51 | 6.3 |
| W3, kD = 30 | 10⁻⁶ | 0×2, 16, 15×2 | 0×2, 34×3 | 1.22 | 10.20 | 8.4 |
| W4 Plummer, kD = 10 | 10⁻⁶ | 0×2, 15×7 | 0×2, 34×7 | 2.01 | 16.24 | 8.1 |
| W1, kD = 500 | 10⁻⁶ | 0×2, 131, 72, 41, 25, 16, 15 | 0×2, 136, 76, 45, 34×3 | 97.63 | 477.88 | 4.9 |

W1 at kD = 500 ran once for each rule: the calibrated one in the matrix, S7's after the
timing runs. S7's rows reproduce CH3.1c's errors to the printed digits (W1 at kD = 100,
10⁻⁶: 5.52e-10 and 6.60e-10), so `--rule s7` is the old default.

## 6. Observations for later tasks

- **CH3.6.** The comparison with FMM3D runs on this rule: W1 at kD = 500 and 10⁻⁶ in 98 s
  (CH3.1c: 508 s, with the reference P2P and S7's degrees).
- **Beyond kD ≈ 500.** The rule's high-frequency branch is measured to x = 65 and
  checked at x = 108. Beyond, at ε ≤ 10⁻⁷, it falls increasingly below S7's per-pair p*
  (157 against 161 at x = 128, 292 against 297 at x = 256, at 10⁻⁸), and §3's linear growth
  of p − x in L says a constant C under-predicts more as x and L grow. A kD ≈ 1000 run at
  10⁻⁸ (W1, about 8 million points, a Kathleen node) would check it; until then the module
  docs say to check the errors there or name S7's degrees.
- **CH4.3.** On the top levels the high-frequency branch over-predicts at ε = 10⁻³ by up
  to 8 degrees (§3), because the excess p − x grows about linearly in L. CH4.3
  recalibrates the high-frequency levels for the diagonal form; a branch
  x + (a L + b) x^{1/3} is the candidate.
- **CH2.3, CH5.2.** Yukawa and damped k are not calibrated here. Complex k takes Re(k) a,
  as before (kD = 100 + 10i is within 0.39 ε).
- **Laplace's C3.4** chose degree 12 for 10⁻⁶ in φ alone. This plateau's 15 at 10⁻⁶ (φ and
  ∇φ, margin 2) is consistent with it: the gradient sets the degree here.

## 7. Reproducing

```sh
export HWLOC_COMPONENTS=-darwin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cargo build --release -p nd-fmm-validate --examples
C=target/release/examples/helmholtz_calibrate
E=1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10
for d in cube sphere plummer clusters; do
  $C --workload w4-$d:100000 --kd 1,10 --eps $E --scan plateau --threads 6 > p-w4-$d.md
done
$C --workload w1 --kd 10 --eps $E --scan plateau --threads 6 > p-w1.md
$C --workload w3 --kd 10 --eps $E --scan plateau --threads 6 > p-w3.md
L=(--eps 1e-3,1e-4,1e-5,1e-6,1e-7,1e-8 --scan levels --hold 0.29,0.45,1.65,1 \
   --hold-eps 1e-10 --threads 6)   # zsh array
$C --workload w1 --kd 30 $L > l-w1-30.md
$C --workload w3 --kd 30 $L > l-w3-30.md
$C --workload w4-cube:100000 --kd 100 $L > l-w4-cube-100.md
$C --workload w4-clusters:100000 --kd 100 --min-x 4 $L > l-w4-clusters-100.md
$C --workload w1 --kd 100 $L > l-w1-100.md
$C --workload w1 --kd 200 --level 2 --first 45 $L > l-w1-200-2.md
$C --workload w1 --kd 300 --level 2 --first 68 $L > l-w1-300-2.md
$C fit p-*.md l-*.md
# Verification: CH3.1c's commands (its §6) on this branch, W3 at kD = 100 also at 1e-8,
# plus the held-out cases and the times before:
X=target/release/examples/helmholtz_accuracy
$X --workload w2 --kd 30,100 --eps 1e-3,1e-6,1e-8 --threads 6
$X --workload w1 --kd 50,150 --eps 1e-3,1e-6,1e-8 --threads 6
$X --workload w4-cube:100000 --kd 3,30 --eps 1e-3,1e-6,1e-8 --threads 6   # and w4-plummer
$X --workload w1 --kd 100 --eps 1e-4,1e-7 --threads 6
$X --workload w1 --kd 100 --eps 1e-3,1e-6,1e-8 --threads 6 --rule s7
```

The scans took about 4.5 hours: stage 1 about 35 minutes, the level scans at kD ≤ 100
about 50 minutes, and the top levels at kD = 200 and 300 about 3 hours, at 75 and 205 s
per run. The verification took about 70 minutes. The kD = 300 scan ran in two parts: to
degree 80, then from 79 with `--max-degree 100`. The default limit is now S7's degree
plus 8.
