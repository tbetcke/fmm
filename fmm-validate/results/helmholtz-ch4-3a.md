# Helmholtz CH4.3a: the bound's constants in the full FMM

Task: docs/helmholtz/tasks/CH4.3a.md (roadmap CH4.3a, the first of CH4.3's five parts; proposal
§10 CH4.3, "C_T and θ from S6 confirmed in the full FMM on W1 and W3"; §3.6). The diagonal form's
rule sends an offset class c of level l to the diagonal form where

E(l, c) = C_T u |k||t_c| max_{j ≤ 2p} (2j + 1)|h_j(k|t_c|)| e^{Im(k)(|t_c| + 2a_l)} + C₀ u ≤ θ ε,

with S6's constants C_T = 0.15 (transfer functions built directly) or 0.23 (interpolated, the
default build since CH4.2c2), C₀ = 200 and θ = 1/2. This report confirms them in the full FMM, on
pairs over the FMM's levels, and measures how much margin the rule leaves.

## 0. Summary

1. **The acceptance holds.** With the rule at the FMM's tolerance every case stays at
   1.000–1.007× Rotation's errors (φ and ∇φ against the direct sum at §9.1's 1,000 samples): W1
   at kD = 100, 200, 500 and 1000, W3 at N = 10⁶ and kD = 200 and 500, ε = 10⁻³ to 10⁻⁸, §9.1's
   random charges, the constant density (kD = 100) and the plane wave's trace (kD = 500), and
   damped W1 at kD = 100 + 10i, 100 + 100i and 500 + 50i; both builds on W1 at kD = 500 (§2).
2. **The margin is 16× for real k, and binds only at 10⁻⁸.** At ε ≥ 10⁻⁶ every class of every
   level with V pairs already takes the diagonal form at θ = 1/2 on these trees (c_leaf 2: the
   deepest V level has k a ≥ 5.4), so a looser θ adds nothing. At 10⁻⁸ the rule at 4θ and 16θ
   adds 2–6 classes at 1.000–1.005×; at 64θ the nearest classes of the low-frequency levels enter
   and the errors reach 1.19–1.52× (§3). θ = 1/2 is safe with a factor of about 16 to spare and
   gains nothing by loosening.
3. **On pairs the bound holds at the FMM's degrees** (§4): for real k every class at every
   (k a, ε) of the FMM's levels within 2E per pair against RCR (at most 1.09 E for k a ≤ 27; from
   k a ≈ 43 up at most 1.99 E, and at k a = 108 E's floor C₀ u, where f64 RCR rounds more and the
   measurement reads against ref's projection, up to 3.3 E against RCR: CH4.2a's known reading;
   C₀ there is CH4.3c's), both builds. One finding below
   S6's calibrated range (k a ≥ 2.8): the direct build's class (0, 0, 3) at k a = 1.69,
   η = 0.1, 10⁻⁴ measures 2.9 E (E = 7.7·10⁻⁹ ≪ ε); question 3.
4. **The damping factor is conservative in the FMM, but cannot simply go** (§5): on damped W1 the
   errors stay at 1.000–1.006× up to 1024θ while the rule takes more classes (up to 1.8× at
   kD = 500 + 50i, 10⁻⁸; 3 → 27 at kD = 100 + 100i, 10⁻⁶); on pairs, in the input-relative measure
   (CH1.3c's weighting), the factor over-predicts by up to 10¹⁴ (η = 0.1) and 10²⁵ (η = 1), yet
   without it E under-predicts by up to 2.4·10⁶ (η = 0.1) and 3.4·10⁶ (η = 1) and would take
   classes whose error exceeds ε. A fitted factor needs its own study (question 2).

## 1. Setup

- FMM: Kathleen job 653621, node-c11a-013, `singlenode`, 2026-11-20 15:21–16:00 (38 min, 40 launches, every one exit 0); one rank bound
  to every core (1 × 40, `mpirun --map-by slot:PE=40 --bind-to core`), release
  `helmholtz_accuracy` from `ffb6518` with the study test uncommitted (no library change), BLAS
  threads 1. `tools/helmholtz/ch4-3a-runs.sh w1 w3 damped densities w1-large`: §9.1's complex
  charges (or `--density`), 1,000 samples, one evaluation, `--near-v cost --near-wx cost`
  (the defaults), the builder's c_leaf 2 and points per leaf, `--m2l rotation` and
  `--diagonal off,rule,<s ε>…` with `--diagonal-build interpolated` (both builds on W1 at
  kD = 500). Raw outputs in `~/projects/fmm-archive/kathleen/runs/ch4-3a/`; the tables below
  from `tools/helmholtz/ch4-3a-tables.py`.
- The margin: `HelmholtzDiagonal::Tolerance(s ε)` in an FMM of tolerance ε runs the rule
  E ≤ θ s ε, so s is θ′/θ (and, away from the floor, C_T/C_T′). s = 1, 4, 16, 64, 256, 1024; on
  the large cases 1, 16, 256.
- Pairs: `nd-fmm-tables`' ignored `the_bound_on_the_fmms_levels` (release, the M6, 9 min): the 34
  classes at k a = kD √3 / 2^{l+1} of the levels l ≥ 2 of kD = 100, 200, 500 and 1000 down to
  k a 1.35 (216 is CH4.3c's), the calibrated p at the four ε, real k, η = Im k / Re k = 0.1
  (every k a) and 1 (k a ≤ 21.65, kD = 100 + 100i's levels), both builds; P2M of 48 complex
  charges in the source box (uniform and its corners), L2P at 32 targets of the target box.
  Measures: per pair, the larger of the relative ℓ2 errors of φ̃ and ∇φ̃ against RCR (CH1.3d1's,
  CH4.2a's), and input-relative, the same differences over the field of the same charges with k
  replaced by Re k (the direct sum), CH1.3c's weighting by the input's size.

## 2. The acceptance: the rule at θ = 1/2

Errors over Rotation's (φ / ∇φ), and the classes per level that took the diagonal form (of 34).

| case | ε | p by level (2, 3, …) | φ ÷ Rot. | ∇φ ÷ Rot. | classes per level |
| --- | --- | --- | ---: | ---: | --- |
| W1, kD = 100 | 10⁻³ | 30, 18, 11 | 1.000 | 1.000 | 2:34, 3:34, 4:34 |
| | 10⁻⁴ | 32, 19, 12 | 1.000 | 1.000 | 2:34, 3:34, 4:34 |
| | 10⁻⁶ | 35, 22, 15 | 1.001 | 1.000 | 2:34, 3:34, 4:34 |
| | 10⁻⁸ | 38, 24, 24 | 1.000 | 1.000 | 2:32, 3:29, 4:3 |
| W1, kD = 200 | 10⁻³ | 54, 30, 18, 11 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁴ | 56, 32, 19, 12 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁶ | 60, 35, 22, 15 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁸ | 63, 38, 24, 24 | 1.000 | 1.000 | 2:34, 3:32, 4:29, 5:3 |
| W1, kD = 500, both builds | 10⁻³ | 123, 66, 36, 21, 13 | 1.000 | 1.000 | 34 on 2–6 |
| | 10⁻⁴ | 126, 68, 38, 22, 14 | 1.000 | 1.000 | 34 on 2–6 |
| | 10⁻⁶ | 131, 72, 41, 25, 16 | 1.000 | 1.000 | 34 on 2–6 |
| | 10⁻⁸ | 135, 76, 44, 27, 24 | 1.000 | 1.000 | 2:34, 3:34, 4:32, 5:32, 6:12 |
| W1, kD = 1000 | 10⁻⁶ | 245, 131, 72, 41, 25, 16 | 1.000 | 1.000 | 34 on 2–7 |
| | 10⁻⁸ | 251, 135, 76, 44, 27, 24 | 1.000 | 1.000 | 2:34, 3:34, 4:34, 5:32, 6:32, 7:12 |
| W3, N = 10⁶, kD = 200 | 10⁻³ | 54, 30, 18, 11 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁴ | 56, 32, 19, 12 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁶ | 60, 35, 22, 15 | 1.000 | 1.000 | 34 on 2–4 (level 5 no V pair) |
| | 10⁻⁸ | 63, 38, 24, 24 | 1.000 | 1.000 | 2:34, 3:32, 4:29 |
| W3, N = 10⁶, kD = 500 | 10⁻³ | 123, 66, 36, 21, 13 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁴ | 126, 68, 38, 22, 14 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁶ | 131, 72, 41, 25, 16 | 1.000 | 1.000 | 34 on 2–5 |
| | 10⁻⁸ | 135, 76, 44, 27, 24 | 1.000 | 1.000 | 2:34, 3:34, 4:32, 5:32 |
| W1 constant density, kD = 100 | 10⁻³ | 30, 18, 11 | 1.000 | 1.000 | 34 on 2–4 |
| | 10⁻⁴ | 32, 19, 12 | 1.000 | 1.000 | 34 on 2–4 |
| | 10⁻⁶ | 35, 22, 15 | 1.007 | 1.001 | 34 on 2–4 |
| | 10⁻⁸ | 38, 24, 24 | 1.000 | 1.000 | 2:32, 3:29, 4:3 |
| W1 plane wave's trace, kD = 500 | 10⁻³ | 123, 66, 36, 21, 13 | 1.000 | 1.000 | 34 on 2–6 |
| | 10⁻⁴ | 126, 68, 38, 22, 14 | 1.000 | 1.000 | 34 on 2–6 |
| | 10⁻⁶ | 131, 72, 41, 25, 16 | 1.000 | 1.000 | 34 on 2–6 |
| | 10⁻⁸ | 135, 76, 44, 27, 24 | 1.000 | 1.000 | 2:34, 3:34, 4:32, 5:32, 6:12 |
| W1, kD = 100 + 10i | 10⁻³ | 30, 18, 11 | 1.000 | 1.000 | 34 on 2–4 |
| | 10⁻⁴ | 32, 19, 12 | 1.000 | 1.000 | 34 on 2–4 |
| | 10⁻⁶ | 35, 22, 15 | 1.000 | 1.000 | 32 on 2–4 |
| | 10⁻⁸ | 38, 24, 24 | 1.000 | 1.000 | 2:27, 3:27, 4:1 |
| W1, kD = 100 + 100i | 10⁻³ | 30, 18, 11 | 1.000 | 1.000 | 4:34 |
| | 10⁻⁶ | 35, 22, 15 | 1.000 | 1.000 | 4:3 |
| W1, kD = 500 + 50i | 10⁻³ | 123, 66, 36, 21, 13 | 1.000 | 1.000 | 3–6: 34 (2: 0) |
| | 10⁻⁴ | 126, 68, 38, 22, 14 | 1.000 | 1.000 | 3:32, 4–6: 34 |
| | 10⁻⁶ | 131, 72, 41, 25, 16 | 1.000 | 1.000 | 3:25, 4:32, 5:32, 6:34 |
| | 10⁻⁸ | 135, 76, 44, 27, 24 | 1.000 | 1.000 | 4:27, 5:27, 6:7 |

The direct and the interpolated build took the same classes on every W1 case at kD = 500, as in
CH4.2c2. The classes left on RCR are the nearest ones of the levels where p sits on the plateau
(24 at 10⁻⁸, the low-frequency levels), and with damping the coarse levels, where the factor
e^{Im(k)(|t| + 2a)} is largest.

## 3. The margin: the rule at s θ

The largest of the two ratios (φ, ∇φ) over Rotation's at each s, the largest s up to which every
run stays within 1.1×, and the classes taken (summed over the levels) at each s. Cases where no
class is left on RCR at s = 1 are summarised in the last row.

| case | s = 1 | 4 | 16 | 64 | 256 | 1024 | margin | classes at s = 1, 4, 16, 64, 256, 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| W1, kD = 100, 10⁻⁸ | 1.000 | 1.003 | 1.003 | 1.424 | 1.429 | 9.924 | 16 | 64, 67, 69, 73, 75, 77 |
| W1, kD = 200, 10⁻⁸ | 1.000 | 1.003 | 1.003 | 1.217 | 1.220 | 5.200 | 16 | 98, 101, 103, 107, 109, 111 |
| W1, kD = 500, 10⁻⁸, direct | 1.000 | 1.001 | 1.002 | 1.321 | 1.321 | 2.123 | 16 | 144, 149, 150, 154, 154, 157 |
| W1, kD = 500, 10⁻⁸, interpolated | 1.000 | 1.000 | 1.002 | 1.370 | 1.394 | 2.497 | 16 | 144, 147, 150, 152, 154, 157 |
| W1, kD = 1000, 10⁻⁸ | 1.000 | – | 1.005 | – | 1.518 | – | 16 | 178, –, 184, –, 188, – |
| W3, kD = 200, 10⁻⁸ | 1.000 | 1.004 | 1.004 | 1.435 | 1.435 | 7.154 | 16 | 95, 98, 98, 100, 100, 102 |
| W3, kD = 500, 10⁻⁸ | 1.000 | 1.000 | 1.000 | 1.242 | 1.242 | 1.242 | 16 | 132, 132, 134, 136, 136, 136 |
| constant, kD = 100, 10⁻⁸ | 1.000 | – | 1.004 | – | 1.453 | – | 16 | 64, –, 69, –, 75, – |
| plane wave, kD = 500, 10⁻⁸ | 1.000 | – | 1.004 | – | 1.188 | – | 16 | 144, –, 150, –, 154, – |
| every real case at ε ≥ 10⁻⁶ | 1.000–1.007 at every s | | | | | | ≥ 1024 (≥ 256) | every class at s = 1 |

- At 10⁻⁸ the classes added by 4θ and 16θ cost at most 0.5 %; at 64θ (256θ on the large cases)
  the nearest classes of the plateau levels (p = 24, k a 5.4–6.8) enter and the errors rise
  1.19–1.52×; at 1024θ up to 9.9×. θ = 1/2 leaves a factor of 16 before any case leaves 1.1×.
- At ε ≥ 10⁻⁶ the margin is not measured from below: every class already passes at s = 1, so the
  margin only says that the rule does not cost classes there. The bound binds the nearest classes
  at k a ≲ 5 (the pairs' table, §4), which on these trees carry no V pairs (c_leaf 2).
- A looser θ would gain little: at 10⁻⁸ 2–6 classes of 64–170, all on low-frequency levels whose
  M2L CH4.3b's time rule decides (Auto runs the exponential M2L there by default).

## 4. Pairs: the bound on the FMM's levels

`the_bound_on_the_fmms_levels`, real k: the largest measured error over E over the 34 classes,
per (k a, ε), per build (direct / interpolated), and the classes the rule takes.

| k a | p at 10⁻³ … 10⁻⁸ | measured ÷ E, direct | measured ÷ E, interpolated | classes taken at 10⁻³, 10⁻⁴, 10⁻⁶, 10⁻⁸ |
| ---: | --- | --- | --- | --- |
| 108.25 | 123, 126, 131, 135 | 3.29, 3.20, 2.98, 3.26 (floor) | 3.22, 3.02, 2.64, 2.70 (floor) | 34, 34, 34, 34 |
| 54.13 | 66, 68, 72, 76 | 1.82, 1.99, 1.74, 1.91 | 1.62, 1.81, 1.76, 1.74 | 34, 34, 34, 34 |
| 43.30 | 54, 56, 60, 63 | 1.69, 1.41, 1.68, 1.35 | 1.46, 1.37, 1.34, 1.23 | 34, 34, 34, 34 |
| 27.06 | 36, 38, 41, 44 | 1.02, 1.02, 0.97, 0.99 | 1.01, 0.94, 0.91, 0.93 | 34, 34, 34, 32 |
| 21.65 | 30, 32, 35, 38 | 1.04, 1.02, 1.04, 0.97 | 0.98, 0.92, 0.93, 0.95 | 34, 34, 34, 32 |
| 13.53 | 21, 22, 25, 27 | 0.52, 0.54, 0.52, 0.51 | 0.51, 0.53, 0.51, 0.50 | 34, 34, 34, 32 |
| 10.83 | 18, 19, 22, 24 | 0.53, 0.53, 0.50, 0.51 | 0.52, 0.52, 0.49, 0.51 | 34, 34, 34, 29 |
| 6.77 | 13, 14, 16, 24 | 0.33, 0.27, 0.43, 0.12 | 0.28, 0.30, 0.30, 0.30 | 34, 34, 34, 12 |
| 5.41 | 11, 12, 15, 24 | 0.36, 0.47, 0.57, 0.18 | 0.27, 0.27, 0.28, 0.23 | 34, 34, 34, 3 |
| 3.38 | 8, 9, 15, 24 | 0.46, 0.63, 0.33, 0.34 | 0.24, 0.38, 0.39, 0.19 | 34, 34, 21, 0 |
| 2.71 | 7, 9, 15, 24 | 1.09, 0.59, 0.36, 0.43 | 0.45, 0.37, 0.48, 0.20 | 34, 34, 9, 0 |
| 1.69 | 7, 9, 15, 24 | 0.39, 0.51, 0.35, 0.22 | 0.79, 1.09, 0.24, 0.18 | 34, 34, 0, 0 |
| 1.35 | 7, 9, 15, 24 | 1.02, 0.50, 0.36, 0.29 | 0.55, 0.85, 0.51, 0.59 | 34, 32, 0, 0 |

- Within 2E everywhere S6 fitted (k a ≥ 2.8), the interpolated build's C_T = 0.23 as tight as the
  direct build's 0.15. Above 2E only at k a = 108 (k a 43 and 54 reach at most 1.99 E), where E
  is its floor C₀ u = 2.2·10⁻¹⁴ and the difference is f64 RCR's own rounding: against ref's projection the diagonal form lies within
  max(2E, RCR's error) (CH4.2a's reading; CH4.3c checks C₀ there against a double-double
  reference).
- The interpolated build takes the same classes as the direct one at every (k a, ε) of the table.
- The RCR chain against the direct sum (the units of the input-relative measure) at 10⁻⁸:
  1.7·10⁻⁷ to 4.1·10⁻⁵, the truncation of a single pair at the box corners (the test's bound is
  10⁻⁴; the plan's 10⁻⁶ was below this truncation).
- **Below S6's range.** Damped, η = 0.1, the direct build's class (0, 0, 3) at k a = 1.69, p = 9,
  10⁻⁴ measured 2.23·10⁻⁸ against 2E = 1.55·10⁻⁸ (2.9 E; RCR's own error 2.6·10⁻¹⁵): the bound
  under-predicts by 1.44×. E there is 7.7·10⁻⁹, far below θ ε, so the FMM is not affected at
  this ε. It is the only class above 2E at k a < 2.8 (real k at most 1.09 E there). Question 3.

## 5. The damped classes

Pairs (`the_bound_on_the_fmms_levels`, interpolated; the direct build alike): per (k a, η, ε) the
largest input-relative error over E with the factor and over E without it (E′ = (E − C₀u)
e^{−Im(k)(|t| + 2a)} + C₀u), with the classes each bound takes and the classes whose
input-relative error is within ε.

| η | k a | ε | input ÷ E | input ÷ E′ | rule (E) | rule (E′) | input ≤ ε |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 108.25 | 10⁻⁶ | 7.8·10⁻¹⁵ | 3.0·10⁵ | 0 | 34 | 34 |
| 0.1 | 108.25 | 10⁻⁸ | 9.9·10⁻¹⁵ | 1.6·10⁶ | 0 | 34 | 33 |
| 0.1 | 54.13 | 10⁻⁶ | 3.4·10⁻⁹ | 45 | 25 | 34 | 34 |
| 0.1 | 54.13 | 10⁻⁸ | 2.1·10⁻⁹ | 29 | 0 | 34 | 33 |
| 0.1 | 27.06 | 10⁻⁸ | 4.2·10⁻⁶ | 0.48 | 27 | 32 | 34 |
| 0.1 | 13.53 | 10⁻⁸ | 1.2·10⁻³ | 0.51 | 27 | 32 | 32 |
| 0.1 | 5.41 | 10⁻⁸ | 1.1·10⁻² | 0.13 | 1 | 3 | 7 |
| 1 | 21.65 | 10⁻⁶ | 2.6·10⁻²⁵ | 5.8·10⁴ | 0 | 34 | 34 |
| 1 | 21.65 | 10⁻⁸ | 7.0·10⁻²⁶ | 2.0·10⁶ | 0 | 34 | 33 |
| 1 | 10.83 | 10⁻⁶ | 7.4·10⁻¹⁵ | 2.3·10³ | 0 | 34 | 34 |
| 1 | 6.77 | 10⁻⁸ | 2.1·10⁻¹⁰ | 1.9·10³ | 0 | 29 | 27 |
| 1 | 5.41 | 10⁻⁶ | 1.4·10⁻⁸ | 1.8·10² | 3 | 34 | 34 |
| 1 | 5.41 | 10⁻⁸ | 1.1·10⁻⁸ | 1.5·10² | 0 | 25 | 21 |
| 1 | 2.71 | 10⁻⁶ | 2.1·10⁻⁵ | 3.3 | 0 | 27 | 27 |

- With the factor the bound over-predicts the input-relative error by up to 10¹⁴ (η = 0.1,
  k a 108) and 10²⁵ (η = 1, k a 21.65) on damped levels, and sends classes to RCR whose error lies far within ε (k a 108, η = 0.1: 0 of 34 taken,
  34 within ε). Without it the bound under-predicts by up to 1.6·10⁶ interpolated and 2.4·10⁶
  direct (k a 108, η = 0.1, 10⁻⁸), and 2.0·10⁶ and 3.4·10⁶ (η = 1, k a 21.65, 10⁻⁸), and takes
  classes whose input-relative error exceeds ε (η = 1 at 10⁻⁸: 29 against 27 within ε at k a 6.77,
  25 against 21 at 5.41; η = 0.1 at 10⁻⁸: 34 against 33 at k a 54 and 108): neither form fits
  the input-relative measure; one in between (a factor of part of the distance) would have to be
  fitted.
- In the FMM (§2, §3): every damped case at 1.000× at s = 1 and at most 1.006× up to s = 1024,
  the rule taking more classes (kD = 500 + 50i at 10⁻⁸: 61 → 107; at kD = 100 + 100i: 34 → 63
  at 10⁻³, 3 → 27 at 10⁻⁶). The FMM measure is the errors over Rotation's (output-relative over
  all sources), not the per-pair input-relative one. On kD = 500 + 50i's coarse levels (2 and 3)
  the factor sends most classes to RCR; what that costs in time is CH4.3b's (one evaluation per
  run here: 9.01 s under Rotation, 9.75 s with the form at s = 1, 8.46 s at s = 1024, within the
  run-to-run spread of single evaluations).

## 6. Questions for sign-off

Signed off 2026-11-20 on #242: every question as recommended.

1. **The acceptance as read, and the constants unchanged.** *Recommended*: C_T = 0.15 (direct)
   and 0.23 (interpolated) and θ = 1/2 confirmed in the full FMM on W1 and W3 at 10⁻³ to 10⁻⁸
   (every case 1.000–1.007× Rotation's; W1 at kD = 1000 at 10⁻⁶ and 10⁻⁸ only), with a margin of
   16 at 10⁻⁸ and nothing to gain from a looser θ at ε ≥ 10⁻⁶; keep them. C₀ = 200 is not
   confirmed here: at k a = 108 the floor is read against the projection, and C₀ at p ≥ 72 against
   a double-double reference is CH4.3c's. Alternative: θ = 2 (s = 4, 1.000–1.004× at 10⁻⁸), for
   0–5 classes more on low-frequency levels.
2. **The damping factor.** *Recommended*: keep e^{Im(k)(|t| + 2a)} (safe: every damped case at
   1.000×), and fit a looser factor in the input-relative measure, confirmed in the FMM on damped
   W1 (both measures), as an Also on CH5.1 (the classifier and pruning for Im k > 0, which owns
   the damped regimes and needs CH4.3). Alternative: a task of its own before CH4.3b, so that
   CH4.3b's times on damped trees use the looser factor.
3. **Below S6's range.** *Recommended*: an Also on CH4.3b: whether its time rule takes the
   diagonal form at k a < 2.8 at all (p ≤ 15 there, where S6 measured RCR 5× the diagonal pair at
   p = 24); if it does, the rule there either bounded to k a ≥ 2.8 or C_T raised to cover the
   2.9 E found (direct build, η = 0.1, k a 1.69). Alternative: bound the rule to k a ≥ 2.8 now.
4. **The split of CH4.3** into CH4.3a (this), CH4.3b (the time rule and the default), CH4.3c
   (the degrees refit and C₀ at large p), CH4.3d (the transfer function's length) and CH4.3e
   (c_leaf), in that order, every Also of CH4.3 on one of them. The split adds three items the
   old line did not name: CH4.3b's diagonal form on by default for sign-off (CH4.2b's "opt-in, the
   default unchanged until CH4.3"), CH4.3c's excess p − x refitted in L (proposal §3.5, "CH4.3's
   to refit") and CH4.3e's sign-off as D11's (D11's amendment: "CH4.3 re-decides").
   *Recommended*: as split.
