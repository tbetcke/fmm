# Helmholtz CH4.3c1: C₀ at large degrees, against a double-double reference

Helmholtz task CH4.3c1 ([docs/helmholtz/tasks/CH4.3c1.md](../../docs/helmholtz/tasks/CH4.3c1.md));
proposal [§3.6](../../docs/design/helmholtz-fmm-proposal.md) (the bound E, C₀ = 200) and §10 CH4.3;
review 2026-11-19, M12 ([docs/reviews/2026-11-19-H1.md](../../docs/reviews/2026-11-19-H1.md) §1.2).
Measured on 2026-11-21 on the M6 (Mac mini, Apple M6, 24 GB; [tools/m6/machine.md](../../tools/m6/machine.md))
from branch `tbetcke/h-ch4-3c1`, release builds. Accuracy only: nothing here is timed.

## Summary

1. **C₀ = 200 is confirmed at p ≥ 72; nothing changes.** Against the diagonal chain computed in
   double-double, the diagonal form's rounding at E's floor is 25–149 u over k a = 27–216 and
   p = 36–251 (u = 2⁻⁵³), below C₀ u = 200 u. Every class of every row lies within 0.94 E, inside
   the rule's contract of 2E, for the tables' production chain (direct and interpolated builds) and
   for ref's chain. S6's "below 176 u" holds to p = 251.
2. **M12's excess was the references' rounding.** The f64 references round more than the diagonal
   form at the floor: RCR 290–3,345 u (1.3–9.4 E), growing with p; the projection of CH1.3d1 about as
   much. The 261 u that M12 read at k a = 54 against the projection is the projection's rounding;
   against the double-double chain the diagonal form errs at most 44 u there. CH4.2a's 5.84 E
   against RCR at k a = 216 is RCR's 3,345 u.
3. **The floor grows slowly with p**: about 40 u at p = 72, 100 u at p = 123 and 145 u at
   p = 234–251, about 0.6 u per degree. At that rate it reaches C₀ u near p ≈ 350 (level 2 at
   kD ≈ 1,400) and 2E near p ≈ 700, beyond D1 and CH4.4's kD = 1000.
4. **The FMM evidence is CH4.3a's.** C₀ does not move, so no class decision changes: the FMM's
   output is unchanged bit for bit (the library change adds a reference and splits the transfer
   function's sampling without changing a bit, tested). C₀ u = 2.2·10⁻¹⁴ lies 2·10⁵ times below
   θ ε at ε = 10⁻⁸, so in D1's range it could move a decision only beyond C₀ ≈ 2·10⁷. CH4.3a
   measured the rule's classes in the full FMM on W1 at kD = 100–1000 and W3, every case 1.000–1.007×
   Rotation's errors (job 653621, W1 at kD = 1000 at 10⁻⁶ and 10⁻⁸ included).

## 1. The reference

`nd_fmm_ref::helmholtz::diagonal::ExtendedM2l` runs the chain of `diagonal::m2l` (S → F, the
diagonal step, F → R; CONVENTIONS §3.15 "Signatures") in double-double, from the same f64 inputs
(the stored multipole, k, s, β, the frames). Everything the f64 chain derives it derives in
double-double: the grid's nodes, weights and azimuths, the θ-factors (a three-term recursion with
a binary exponent per order, since P̄ₘᵐ ~ sinᵐ θ leaves f64's range near the poles at p ≈ 250),
the transfer function (the series `Transfer` rounds, now `sample_dd`), the factor
4π ik e^{2β − Im(k)|t|} and sⁿ. In exact arithmetic the chain is the truncated M2L; its own rounding
is about 10⁻¹⁶ of the f64 chain's. The local is returned as hi + lo, and an f64 local's error is
the field of their difference (formed in double-double) over the reference's field, the larger of
φ̃'s and ∇φ̃'s relative ℓ2 norms at the targets: the measure of CH1.3d1 and CH4.3a, with the
evaluation's rounding relative to the difference.

Tests (`nd-fmm-ref`, CI):

- `s2f_then_f2r_is_the_identity_in_double_double`: S → F → R without the diagonal step gives back
  s²ⁿ M̂ₙᵐ to 3.6·10⁻³¹ (s = 1) and 2.2·10⁻³² (s < 1); bound 10⁻²⁶.
- `the_theta_factors_against_the_f64_table`: the θ-factors against the f64 grid's at p = 300 on
  the rings nearest the poles, 8.4·10⁻¹⁶; bound 10⁻¹⁴.
- `the_extended_chain_against_the_f64_chains`: on the 34 classes at k a = 2.83 (p = 7), 8 (p = 27,
  real and η = 0.1) and 16 (p = 36), the f64 diagonal form within 0.05–0.57 E of it, and RCR within
  5.5·10⁻¹² (bound 5·10⁻¹¹). On the nearest classes at k a = 8 the f64 diagonal form errs by up to
  10⁴ (E up to 2.8·10⁵) while RCR stays within 5.5·10⁻¹² of the reference: two algorithms meet it,
  one of them where the f64 chain the reference mirrors fails by sixteen orders.
- `the_identity_in_double_double_at_p_251` (ignored, release): the identity at p = 251, the
  study's largest degree, to 1.0·10⁻²⁹, so the reference's own rounding is far below the floor
  it reads there.
- The transfer functions bit for bit as on `main`: `the_transfer_samples_as_before_the_split`
  asserts an FNV hash of `Transfer::new`'s samples (real and damped k) taken on `main` at
  78a6f65; the ring and parts tests pass unchanged.

## 2. The floor at the FMM's degrees

`nd-fmm-tables`' ignored `the_floor_against_the_extended_chain` (release, 303 s on six threads):
k a = kD √3 / 2^{l+1} of W1's levels (216.5 and 108.3: levels 2 and 3 at kD = 1000; 54.1 and 27.1:
levels 4 and 5; 43.3: level 2 at kD = 200), p of the calibrated rule at ε = 10⁻³ and 10⁻⁸ (and
10⁻⁶ below k a 100), real k; the 34 classes, two placements each (P2M of 48 complex charges in the
source box, uniform and its corners; 32 targets in the target box, uniform and its corners). Four
f64 paths: the tables' level call (CH4.1's conversions) with the direct build (E at C_T = 0.15) and
the interpolated build (C_T = 0.23), ref's `diagonal::m2l`, and RCR (read against the direct E).
The floor classes are those with E < 2 C₀ u (the Hankel term below C₀ u).

| k a | ε | p | floor classes | largest error / E: tables direct, interpolated, ref, RCR | at the floor, in u | every class, in u |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 216.51 | 10⁻³ | 234 | 32 | 0.933, 0.723, 0.936, 6.163 | 145, 144, 149, 2130 | 408, 408, 410, 2130 |
| 216.51 | 10⁻⁸ | 251 | 29 | 0.437, 0.316, 0.441, 9.370 | 139, 139, 146, 3345 | 334, 336, 336, 3345 |
| 108.25 | 10⁻³ | 123 | 34 | 0.266, 0.222, 0.269, 4.346 | 99, 99, 100, 1203 | 99, 99, 100, 1203 |
| 108.25 | 10⁻⁸ | 135 | 32 | 0.270, 0.203, 0.269, 3.723 | 71, 71, 71, 1062 | 8607, 8641, 8590, 1062 |
| 54.13 | 10⁻³ | 66 | 32 | 0.164, 0.122, 0.169, 2.093 | 31, 30, 32, 507 | 98, 99, 101, 507 |
| 54.13 | 10⁻⁶ | 72 | 29 | 0.222, 0.147, 0.223, 1.997 | 41, 40, 43, 491 | 17243, 17456, 17332, 491 |
| 54.13 | 10⁻⁸ | 76 | 29 | 0.170, 0.114, 0.173, 2.190 | 42, 42, 44, 545 | 1572977, 1610970, 1596981, 545 |
| 43.30 | 10⁻³ | 54 | 32 | 0.301, 0.216, 0.300, 1.458 | 30, 30, 32, 341 | 222, 221, 221, 341 |
| 43.30 | 10⁻⁶ | 60 | 29 | 0.341, 0.224, 0.332, 1.538 | 33, 34, 33, 367 | 87847, 88495, 85653, 367 |
| 43.30 | 10⁻⁸ | 63 | 29 | 0.201, 0.149, 0.206, 1.377 | 39, 39, 38, 331 | 2848862, 2806019, 2925567, 331 |
| 27.06 | 10⁻³ | 36 | 32 | 0.260, 0.183, 0.258, 1.299 | 25, 25, 27, 290 | 370, 381, 368, 290 |
| 27.06 | 10⁻⁶ | 41 | 29 | 0.176, 0.120, 0.180, 1.278 | 26, 27, 27, 290 | 172701, 180027, 176365, 290 |
| 27.06 | 10⁻⁸ | 44 | 27 | 0.136, 0.101, 0.159, 1.266 | 27, 27, 28, 290 | 19597695, 20070070, 22884969, 290 |

- The diagonal paths stay within 0.94 E everywhere (the rule needs 2E), the largest on a nearest
  class at k a = 216, p = 234, where E is the Hankel term and not the floor; at the floor they are
  within 0.75 C₀ u. The interpolated build rounds no more than the direct one (its E is larger by
  C_T 0.23/0.15, so its ratios are smaller).
- "Every class, in u" is large where the nearest classes' E is large (k a ≤ 108 at ε ≤ 10⁻⁶): the
  rule sends those classes to RCR, or E stays below θ ε there; the ratio column is the contract.
- RCR's error over the direct E is above 2 wherever p ≥ 66: against f64 RCR no class at the floor
  can be shown within 2E, which is what CH1.3d1, CH4.2a and CH4.3a met.

## 3. The earlier floor readings, read again

The tests that read the floor against the projection or a regression limit now read it against
the double-double chain, within 2E (stricter: the old reading allowed max(2E, RCR's error) against
the projection, and at k a = 216 a regression limit of 5·10⁻¹³):

- `nd-fmm-ref`'s `m2l_at_the_degrees_of_kd_1000` (k a 108 at p = 131, 216 at p = 245; three
  classes, 16 s): passes. The classes above 2E against RCR: (3, 3, 3) at k a 108, 2.86 E against
  RCR, is 3.9·10⁻¹⁵ (0.12 E) from the double-double chain; at k a 216, 5.84 E against RCR, it is
  9.5·10⁻¹⁵ (0.24 E), while RCR lies 2.3·10⁻¹³ from it.
- `nd-fmm-tables`' `against_rcr_at_large_degrees` and `interpolated_at_large_degrees`: pass. Every
  class above 2E against RCR (up to 3.7 E at k a 108, p = 131; 2.4 E at k a 54, p = 82) lies within
  2.3–5.0·10⁻¹⁵ of the double-double chain, RCR 6.5·10⁻¹⁴–1.2·10⁻¹³ from it.
- CH4.3a's `the_bound_on_the_fmms_levels` (the FMM's levels, both builds, real and damped k):
  passes (35 s). At k a = 108 the largest per pair is now 1.85–1.99 E (both builds, every ε),
  against CH4.3a's 2.64–3.29 E: the classes above 2E against RCR are read at their error against the
  double-double chain, and the column's remainder is RCR's own rounding on classes just below 2E.
  The finding below S6's range that CH4.3a recorded (the direct build's (0, 0, 3) at k a = 1.69,
  η = 0.1, 10⁻⁴, where E = 7.7·10⁻⁹ ≪ ε) is confirmed against the double-double chain: 2.2·10⁻⁸,
  2.9 E, with RCR 9.7·10⁻¹⁶ from it, so it is the diagonal form's own error, not RCR's; it stays a
  printed finding, as CH4.3a's question 3 left it (E there is far below every tolerance).

`at_the_floor_against_the_projection` (k a 54, p = 72) is kept as it was: it records the
projection's reading that M12 quoted.

## 4. Questions for sign-off

1. **C₀ = 200 kept** (recommended): confirmed at p = 36–251 against the double-double chain, the
   floor at most 149 u; no change to the bound, the rule or the proposal (§3.6 already says C₀ = 200
   "below 176 u", which holds). Alternative: none indicated by the data.
2. **The order of CH4.3c2 and CH4.3b2c** (recommended: add CH4.3c2 to CH4.3b2c's *Needs*). CH4.3b2c
   turns the diagonal form on by default and measures the order of waves, form and RCR per level;
   CH4.3c2 may lower the high-frequency degrees at x ≥ 54, which changes those levels' costs. With
   the *Need*, CH4.3b2c measures on the final degrees. Alternative: keep the order as it is and let
   CH4.3c2 re-check CH4.3b2c's choices.

## 5. Reproduce

```sh
cargo test -p nd-fmm-ref --test helmholtz the_extended_chain           # CI test
cargo test -p nd-fmm-ref --lib extended                                # unit tests
cargo test -p nd-fmm-tables --release --test helmholtz the_floor_against_the_extended_chain \
  -- --ignored --nocapture                                             # §2 (FLOOR_DETAIL=1 per class)
cargo test -p nd-fmm-ref --release --test helmholtz m2l_at_the_degrees_of_kd_1000 -- --ignored --nocapture
cargo test -p nd-fmm-tables --release --test helmholtz -- --ignored --nocapture \
  against_rcr_at_large_degrees interpolated_at_large_degrees the_bound_on_the_fmms_levels
```
