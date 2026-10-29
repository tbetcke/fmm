# Truncation under realistic geometry: spike report (Helmholtz CH0.7, S7)

Helmholtz task CH0.7 ([docs/helmholtz/tasks/CH0.7.md](../../docs/helmholtz/tasks/CH0.7.md)).
The question is proposal §10 CH0.7, with §3.5, §5.1–§5.3, §5.5, §5.6 and §9 item 5. Written
on 2026-10-29 from branch `tbetcke/h-ch0-7` (main at 88024d6 plus this spike), on the M3
Max (16 cores, 64 GB), release builds, rustc 1.99.0. The measurements are in double-double
precision, with f64 for comparison.

## Summary

1. **Only truncation is measured** (§1.3). Every operator and the direct sum run in
   double-double precision: S1's rotation blocks, S2's coaxial tables, and S2's and S3's
   radial functions, ported to a generic scalar. At far higher p every chain leaves at most
   10⁻²⁰, 5·10⁹ below the smallest ε/2; where Im(k) a ≥ 16 (η = 1, Yukawa, and η = 0.1 at
   k a = 256) at most 6·10⁻¹⁷. The
   building blocks meet S1's, S2's and S3's mpmath fixtures to their f64 rounding.
2. **p for real k has two regimes** (§3). Below k a ≈ 8 it is nearly flat in k a (the
   Laplace limit, a few degrees lower near k a = 4–8), and **above** the formula of §3.5
   for k a ≤ 2: 30 against 22 at 10⁻⁶, 55 against 35 at 10⁻¹⁰. Corner clusters and the
   gradient set it. The two cross between k a = 4 and 23. From k a = 32 p is **below**
   the formula by 6–19 degrees (5–15 %): 288 against 303 at k a = 256 and 10⁻⁶. The
   closed form
   p = max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉) never under-predicts the
   per-level degree at the grid points it was fitted to (0.016 ≤ k a ≤ 256,
   10⁻¹⁰ ≤ ε ≤ 10⁻³) and over-predicts by at most 9 degrees.
3. **Over two and three levels** (§4). M2M and L2L chains need no extra degree for real k:
   margins of 9–57 at the top. The two-level chain needs up to 3 more at low frequency,
   where both levels contribute equally, at most 1 from k a = 8 and none from 64. Damped k
   needs up to 4 (η = 0.1 to k a = 128; 7 at 181 and more than 8 at 256, a level pruned at
   every ε) and 8 (η = 1, Yukawa).
4. **Placements and norms** (§3). Corners and the gradient set p at low frequency: up to
   19 degrees above uniform points and 14 above the potential alone. At high frequency the
   placements differ by at most 16 degrees (about 5 %) and the gradient by 5. The worst
   offset is a nearest one on an axis at low frequency, and from k a ≈ 32 (real k and
   η = 0.1) mostly the farthest diagonal (±3, ±3, ±3) or |t|² = 6.
5. **Strong damping and Yukawa need 2–2.5 times the formula** at κ a = 16–32 in the
   per-pair relative norm (§6). That is a property of the norm, since the damped pair field
   is decades below its terms. The full FMM decides (question 2). η = 0.1 stays within 10
   degrees of real k.
6. **f64 meets every calibrated p for real k** (§5), on S2's signed-off tables and all 316
   offsets, but not for strong damping from κ a = 16, nor for η = 0.1 at k a ≥ 128 on this
   spike's f64 path.
7. **S2's projection does not serve L2L with complex k** (§2.4). The L2L tables come from
   the recursion; question 3 amends S2's sign-off for CH1.2.

## 1. Method

### 1.1 The chains (`src/calibrate.rs`)

Level 0 is the M2L level, with boxes of width w = 1 and circumradius a = √3/2; level 1 holds
their children and level 2 their grandchildren. k a means κ a at level 0, and the kinds are
real k = κ, complex k = κ(1 + iη) with η = 0.1 and 1, and imaginary k = iκ (Yukawa), the
set of S2. The source box S sits at the origin. The target box T sits at each of the 316
V-list offsets:

- **C1, one level**: P2M in S at p₀ → M2L to T → L2P in T. This gives the base calibration
  p₁(k a, ε, kind).
- **C2, two levels** (the acceptance's chain): P2M in S's eight children at p₁ → M2M to S at p₀
  → M2L to T → L2L to T's children → L2P.
- **C3m, M2M over three levels**: P2M in S's 64 grandchildren → M2M → M2M → S's multipole
  evaluated at the targets of T (M2P).
- **C3l, L2L over three levels**: the local of S formed from the sources of T (P2L) → L2L →
  L2L → L2P in S's grandchildren.

Every M2M, M2L and L2L is a rotation, a coaxial step and the rotation back (RCR). Each
chain is compared with the direct sum of the same sources at the same targets. The error is
the relative ℓ2 error over the targets, of the potential and, separately, of the gradient
(proposal §9.1). Each box holds 128 sources and 64 targets, with random complex charges whose
real and imaginary parts are uniform in [−1, 1]. Placements and charges are seeded, so
every run repeats bit for bit.

### 1.2 Placements (`src/geometry.rs`)

The points fill the boxes in three ways. Each is generated relative to the chain's leaf
boxes, the boxes where P2M and L2P act:

- **uniform**: uniform in the box;
- **corners**: each point in a corner cube, side w_leaf/8, of a random leaf box, a random
  one of its eight corners. These are the points farthest from the leaf centres, where the
  expansions converge slowest. With 128 points over 64 leaves (C3m) not every leaf is
  occupied;
- **patch**: on a sphere of radius 2w or 8w (W1's surface at levels 2 and 4) through a
  random point near the box centre, at a random orientation.

### 1.3 Precision: only truncation is measured

Every operator runs in double-double arithmetic (`src/real.rs`): about 32 digits, built
from the error-free transformations of Dekker and Knuth with a fused multiply-add, as in
the QD library. That covers the radial functions, the harmonics, S1's rotation blocks, S2's
coaxial tables and the direct sum. The code is generic over a `Real` trait, so the same
chains also run in f64. The prototypes are ported as follows:

- the radial functions: S2's (Miller's method for jₙ, upward recursion for hₙ), which S3
  confirmed;
- the rotations: S1's Gumerov–Duraiswami y-blocks;
- the coaxial tables: S2's recursion in its triangle, or S2's projection (§2.3, §2.4).

Test 6 (`tests/chains.rs`) runs every chain at a degree far above the calibrated one, so
that what is left is rounding plus a negligible truncation. It does this at k a = 0.1, 4 and
32 for every kind, at 128 for real k and η = 0.1, and at 256 for both; on the nearest
offsets (on the axis, β = 0 and π, and off it) and the farthest; and in every placement.
The error left is at most 10⁻²⁰, which puts rounding at least 5·10⁹ below every ε/2 of the
sweep. The exceptions have Im(k) a ≥ 16. There the damped pair field is tens of decades
below its own terms, and the chain levels off between 2·10⁻¹⁷ and 6·10⁻¹⁷, still 10⁶ below
the smallest ε/2.

| Chain | real, k a = 4 | real, 256 | η = 0.1, 128 | η = 0.1, 256 | imaginary, κa = 32 |
| --- | --- | --- | --- | --- | --- |
| C1 | 2.3·10⁻²⁴ | 1.2·10⁻²⁵ | 1.0·10⁻²³ | 5.8·10⁻²¹ | 3.6·10⁻¹⁸ |
| C2 | 2.1·10⁻²³ | 2.9·10⁻²⁵ | 2.9·10⁻²² | 8.1·10⁻¹⁸ | 3.0·10⁻¹⁷ |
| C3m | 6.8·10⁻³¹ | 2.5·10⁻²⁶ | 1.6·10⁻²⁵ | 6.3·10⁻¹⁷ | 1.9·10⁻²⁴ |
| C3l | 3.0·10⁻³¹ | 2.0·10⁻²⁹ | 1.9·10⁻²⁵ | 2.3·10⁻¹⁷ | 1.3·10⁻²⁸ |

(The larger of the potential and gradient errors; `cargo test -p nd-fmm-spike-truncation
--release -- --include-ignored`.)

### 1.4 The calibration (`examples/sweep.rs`)

- **The grid**: k a on the half-octave grid 2^{j/2}, j = −12 to 16 (0.016 to 256), so that
  k a/2 and k a/4 lie on it. The top covers W1's level 2 at kD = 1000 (k a = 216, S8).
  Real k and η = 0.1 cover the whole grid; η = 1 and imaginary k go to κ a = 32. That
  passes §5.5's pruning cut, where e^{−Im(k) w} = e^{−37} and a damped level carries no V
  list.
- **The tolerances**: ε = 10⁻³ to 10⁻¹⁰ by decades.
- **The calibrated p₁(k a, ε, kind)** is the smallest p from which on the worst error over
  the offsets, the three placements, potential and gradient, is at most **ε/2**. The margin
  of 2 stands for the sum over a target's V lists and levels that the full FMM adds. The
  search runs on the 54 offsets with |t|² ≤ 6, for cost: at low frequency they hold the
  worst case, at high frequency the farthest offsets often do (§3). It scans every fourth
  p and then refines below each threshold's first pass (the first 40 of the 104 (kind, k a)
  sets scanned every p, an equivalent search; `results/c1_curves.csv` holds both).
  Each p found is then checked on all 316 offsets at p, p + 1 and p + 2 in one run, and
  raised further until it passes (capped at p + 8, and the multi-level δ at 8;
  `results/summary.md` lists any row below the margin: none for C1, six multi-level rows
  for η = 0.1 at k a = 256, §4). The near search found the final p in 771 of the
  832 (kind, k a, ε) cases and was at most 5 short; the 61 others are high-frequency real
  k and η = 0.1, where farther offsets bind.
  The margins reached run from 2.0 to 6.5.
- **The multi-level chains** use p_l = p₁(k a_l) + δ on every level, with the smallest δ ≥ 0
  that meets ε/2, searched on the 54 offsets and checked on all 316 in the same way.
- **f64** is C1 at the calibrated p in f64 with S2's signed-off rule for the tables
  (recursion where Re(k) a ≤ 40, projection above), on all 316 offsets
  (`results/f64_all.csv`, `examples/sweep.rs f64all`).
- **The per-placement and potential-only degrees** come from the search's curves, on the 54
  nearest offsets; the 316-offset check applies to p itself.
- **Output**: `results/c1.csv` (p₁ with its errors, per placement, for the potential alone,
  and in f64 on the 54 nearest offsets), `results/c1_curves.csv` (the scanned error
  curves), `results/multi.csv` (δ), `results/f64_all.csv` (f64 on all 316 offsets) and
  `results/summary.md` (the tables of §3 to §5).

## 2. Building blocks

### 2.1 Tests 1–5 (`tests/blocks.rs`, `tests/coaxial.rs`)

- **Double-double**: π, e and ln 2 agree with their decimal values to 10⁻³¹, and
  sin² + cos² = 1 and e^x e^{−x} = 1 hold to 10⁻³⁰. The f64 path agrees with `std`.
- **Radial functions**: the cross products jₙ yₙ₋₁ − jₙ₋₁ yₙ = z⁻² hold to 10⁻²⁸ relative
  for real, complex and imaginary z with |z| from 10⁻³ to 2000. Raising Miller's start
  changes nothing beyond 10⁻³⁰ of the envelope. Against S3's mpmath fixture the worst
  error is below 10⁻¹⁴, the fixture's f64 rounding.
- **Harmonics**: orthonormal by quadrature, and the addition theorem holds, to 10⁻²⁹. The
  gradients agree with central differences.
- **Rotations**: orthogonal to 10⁻²⁸ at degree 300, and S1's fixture holds to 10⁻¹⁴. The
  definition Y(Q x) = O(Q) Y(x) holds to 10⁻²⁸ for all 316 offset directions and the eight
  child shifts, with Q taking the shift onto +z.
- **Coaxial tables**: the recursion and the projection meet S2's mpmath fixture in S2's
  FMM-weighted metric to 6·10⁻²⁹ over the 132 sets in the spike's range: the double-double
  tables round to the same f64 values. The recursion agrees with the projection over whole
  tables to 3·10⁻³⁰ at k a ≤ 32 (every kind) and to 2·10⁻²⁵ at k a = 90.5. Rectangular
  truncation is a block of the square table, bit for bit.

### 2.2 The recursion in double-double

S2's recursion loses accuracy like e^{0.2 k a}. In double-double it still agrees with the
projection to 2·10⁻²⁵ at k a = 90.5, but only to 6·10⁻²² at 128. The spike therefore builds
its M2L tables by the recursion where Re(k) a ≤ 100 and by projection above. In f64 it keeps
S2's signed-off threshold of 40.

### 2.3 The scale

The spike scales level l by s_l = min(|k| a_l/8, 1), not by CH0.3's min(|k| a_l, 1). The
(S|R) table needs h_{2p}(k t) at the nearest distance. Under min(|k| a, 1) that overflows the
exponent range from p ≈ 90 at k a = 1, a degree that only test 6 needs. The truncated
chain does not depend on the scale: C1 at k a = 1 and p = 60 gives 9.512·10⁻¹⁴ under both
scales. The calibrated degrees stay below 2p ≈ 125 at low frequency, where CH0.3's scale is
still in range, so the sign-off is unaffected.

The scale factors of M2M and L2L are applied per entry, as (s_src/s_tgt)ⁿ s_tgt^{n−n′} for
M2M and (s_tgt/s_src)^{n′} s_src^{n′−n} for L2L (`ShiftScale`). The
two powers applied one after the other form the unscaled local coefficients, which leave the
range at low frequency (10³⁵⁸ at k a = 0.1, p = 135).

### 2.4 The L2L tables: S2's projection serves M2M, not L2L with damping

S2 builds (R|R) = (S|S) by projection on an outer sphere and checked it in M2M's weights.
An L2L reads the same table the other way: from a parent local of high degree to a child
local of low degree. In this spike that failed:

- C3l with η = 0.1 at k a = 128 leaves 1.4·10⁻¹⁰ with projection tables at the degrees of
  test 6 ([217, 160, 160]), where the recursion gives 1.9·10⁻²⁵: above ε/2 at ε = 10⁻¹⁰.
  With p = 220 on every level it grows to 6·10¹⁹ (3·10²⁰ on the probe's offsets);
- at k a = 256 with η = 0.1 the projection reaches only 2·10⁻¹²;
- real k at k a = 256 blew up when the child degrees were far above the child's k a
  (p = 345 on every level), but gave 2·10⁻²⁹ at the degrees the chains use;
- more quadrature nodes (p + 100 + |k| t) did not change it.

The ignored test `l2l_by_projection_fails_with_damping` (`tests/chains.rs`) records the
first case at both degree sets, with the shift tables forced to projection and to
recursion; its output is in `results/chains.log`.

The spike therefore builds the shift tables by the recursion for complex k and for real k
to Re(k) a = 128 (the shift's argument is k a/2), and by projection only for real k above.
The recursion levels off at 2·10⁻¹⁷ for η = 0.1 at k a = 256 (§1.3). This bears on S2's
signed-off rule (projection above Re(k) a = 40) for CH1.2: question 3 of §8.

## 3. p₁: the one-level calibration

Full tables: `results/summary.md`. Selected values below, calibrated p₁ / the formula
of proposal §3.5:

| k a | real, 10⁻³ | 10⁻⁶ | 10⁻⁸ | 10⁻¹⁰ | η = 0.1, 10⁻⁶ | 10⁻¹⁰ |
| --- | --- | --- | --- | --- | --- | --- |
| 0.016 | 13 / 12 | 30 / 22 | 42 / 29 | 55 / 35 | 30 / 22 | 55 / 35 |
| 1 | 12 / 12 | 29 / 22 | 42 / 29 | 53 / 35 | 29 / 22 | 53 / 35 |
| 4 | 11 / 14 | 26 / 23 | 37 / 30 | 49 / 36 | 27 / 23 | 52 / 36 |
| 8 | 15 / 18 | 26 / 27 | 37 / 33 | 49 / 39 | 29 / 27 | 53 / 39 |
| 16 | 24 / 28 | 30 / 36 | 38 / 41 | 49 / 47 | 33 / 36 | 56 / 47 |
| 32 | 41 / 47 | 49 / 56 | 53 / 61 | 57 / 67 | 50 / 56 | 67 / 67 |
| 64 | 75 / 83 | 85 / 94 | 90 / 100 | 95 / 106 | 85 / 94 | 96 / 106 |
| 128 | 141 / 151 | 153 / 166 | 161 / 174 | 167 / 181 | 149 / 166 | 164 / 181 |
| 181 | 195 / 207 | 209 / 223 | 217 / 232 | 224 / 241 | 202 / 223 | 217 / 241 |
| 256 | 272 / 286 | 288 / 303 | 297 / 314 | 304 / 323 | 280 / 303 | 294 / 323 |

**Two regimes, crossing between k a = 4 and 23:**

- **Low frequency (k a ≲ 4).** p₁ hardly depends on k a; it is Laplace's limit. For
  k a ≤ 2 it is above the formula at every ε below 10⁻³, by 7–8 at 10⁻⁶ and 18–20 at
  10⁻¹⁰; at k a = 2.8 and 4 the two meet at 10⁻⁴ to 10⁻⁵. The formula's
  floor p_lo assumes a convergence ratio of 1/2 per degree. The corner clusters at the
  nearest offset converge more slowly.
- **High frequency (k a ≥ 32).** p₁ is below the formula at every ε, by 6 to 19 degrees
  (5–15 %). At k a = 16 and 23 it is 1–7 below at ε ≥ 10⁻⁹ and up to 2 above at 10⁻¹⁰.
  The excess over k a grows like (k a)^{1/3}, as Gumerov and Duraiswami's
  formula says, but with a smaller constant: p₁ − k a = C(ε) (k a)^{1/3}, where C runs from
  3.5 at 10⁻³ to 7.9 at 10⁻¹⁰. That is just below (ln 1/ε)^{2/3} (3.6 to 8.1), against the
  formula's (1.5 ln 1/ε)^{2/3} (4.8 to 10.6).

**The closed form for real k**, fitted to the per-level degree p* of §4 (p₁ plus the
multi-level δ):

```math
p(ka, \varepsilon) = \max\left(\lceil 6\log_{10}(1/\varepsilon)\rceil - 2,\;
\left\lceil ka + (\ln(1/\varepsilon))^{2/3} (ka)^{1/3}\right\rceil\right)
```

It never under-predicts p* at the real-k grid points (0.016 to 256, ε from 10⁻³ to
10⁻¹⁰), the data it was fitted to, and over-predicts by at most 9 degrees. Between grid
points it is an interpolation. It does not hold for damped k (§6).

**Placements and the norm** (the per-placement and potential-only degrees on the 54
nearest offsets, "all" on the 316):

| k a | ε | all | uniform | corners | patch | potential alone |
| --- | --- | --- | --- | --- | --- | --- |
| 0.016 | 10⁻⁶ | 30 | 21 | 30 | 19 | 19 |
| 0.016 | 10⁻¹⁰ | 55 | 39 | 55 | 35 | 42 |
| 4 | 10⁻¹⁰ | 49 | 32 | 49 | 31 | 42 |
| 64 | 10⁻¹⁰ | 95 | 92 | 94 | 91 | 93 |
| 256 | 10⁻⁶ | 288 | 279 | 284 | 273 | 284 |
| 256 | 10⁻¹⁰ | 304 | 295 | 301 | 289 | 300 |

At low frequency the corner clusters set p, up to 19 degrees above uniform points
(k a = 1.4, 10⁻¹⁰), and so does the gradient, up to 14 above the potential alone. At high
frequency the placements differ by at most 16 degrees, about 5 %, and the gradient by 5.

**The worst offset** depends on the regime. At k a ≤ 8 it is a nearest offset on a
coordinate axis, (0, 0, ±2), (±2, 0, 0) or (0, ±2, 0), in 604 of 608 rows (every kind),
where the corner clusters of the two boxes come closest. From k a ≈ 32, for real k and
η = 0.1, it is mostly the farthest diagonal (±3, ±3, ±3) or an offset with |t|² = 6 or 12.
There the M2L's argument k|t| is largest, and the 316-offset check raised the near
search's p by 1 to 5 in 61 rows.

For scale: Laplace's own calibration (Phase 3 T12) chose degree 12 for 10⁻⁶ in the
potential of the whole FMM, where the near field dominates the norm. The per-pair worst
case here asks 19 for the potential and 30 for potential and gradient at k a → 0. The
per-pair norm is the conservative side, and CH3.3 measures the whole FMM.

## 4. The multi-level chains

The largest δ over ε, by band of k a. The chains use p_l = p₁(k a_l) + δ on every level,
and δ is the smallest that meets ε/2 on all 316 offsets and every placement:

| Kind | Chain | k a ≤ 1 | 1–8 | 8–64 | 64–256 |
| --- | --- | --- | --- | --- | --- |
| real | C2 | 3 | 2 | 1 | 0 |
| real | C3m, C3l | 0 | 0 | 0 | 0 |
| η = 0.1 | C2 | 3 | 3 | 4 | 4 to k a = 128, 7 at 181, > 8 at 256 † |
| η = 0.1 | C3m | 0 | 0 | 1 | 2 to 128, 6 at 181, > 8 at 256 † |
| η = 0.1 | C3l | 0 | 0 | 0 | 2 to 128, 3 at 181, 4 at 256 |
| η = 1 | C2 | 4 | 5 | 8 | – |
| imaginary | C2 | 5 | 4 | 8 | – |
| η = 1, imaginary | C3m, C3l | 0 | 0 | 0 | – |

† The cap of the δ search (8) was reached without meeting ε/2 in six rows, all η = 0.1 at
k a = 256: C2 at ε = 10⁻³ to 10⁻⁶ (margins 0.31 to 1.34) and C3m at 10⁻³ and 10⁻⁴ (0.24,
1.22). There δ is above 8, unresolved. That level is far past §5.5's pruning cut
(e^{−Im(k) w} = 1.5·10⁻¹³), so the FMM prunes it at every ε of the sweep.

- **M2M and L2L over three levels need nothing extra** for real, η = 1 and imaginary k. At
  each level's own p₁, C3m and C3l meet ε/2 with margins of 9 to 57 at the top of the grid.
  For η = 0.1 they need at most 2 more to k a = 128, then 3 to 6 at 181 and up to 8 or
  more at 256.
- **C2 adds up to 3 degrees for real k**, all at low frequency. There p₁ does not depend on
  k a, so the child and parent levels contribute about equally and the error roughly
  doubles. From k a = 8 the two-level chain needs at most 1, from k a = 64 nothing.
- **Damped k needs more**: up to 4 for η = 0.1 to k a = 128, growing with Im(k) a beyond
  (7 at k a = 181, more than 8 at 256, where Im(k) a = 18 and 26), and up to 8 for η = 1
  and imaginary k from κ a = 8, on top of their already large p₁ (§6). So the extra
  degrees of damping appear in the multi-level chains even where C1's p₁ sits below
  real k's.
- **The per-level degree** p* = p₁(k a) plus the largest δ of every chain that uses the
  level (C2 at k a and 2 k a, C3m and C3l at k a, 2 k a and 4 k a) is what the FMM would
  use, since a level is the parent of one chain and the child of the next. It is
  tabulated in `results/summary.md`, with the closed form of §3 beside it.

## 5. f64

C1 at the calibrated p in f64, with S2's signed-off tables (recursion where Re(k) a ≤ 40,
projection above), on all 316 offsets and the three placements (`results/f64_all.csv`):

- **Real k**: meets ε/2 in all 232 (k a, ε) cases, up to k a = 256, including the far
  offsets that bind at high frequency. f64 does not limit the calibrated degrees for
  real k.
- **η = 0.1**: fails in 16 cases, all at k a ≥ 128, levelling off at 1.6·10⁻⁹
  (k a = 128), 5.2·10⁻⁶ (181) and 0.61 (256). The spike's f64 path lacks S2's
  exponential normalisation of the projection, so this is an upper bound on f64's own
  floor, not a property of f64.
- **η = 1 and imaginary k**: fail in 20 and 21 cases, from κ a = 16 (imaginary k: one
  case at κ a = 11.3 and 10⁻¹⁰), levelling off at 1.2–1.4·10⁻⁷ (κ a = 16), 4·10⁻³ (22.6)
  and 10³–10⁴ (32). Here the damped pair field lies tens of decades below the terms
  that sum to it, so the cancellation is intrinsic to the per-pair relative norm. The
  levels concerned are at or beyond §5.5's pruning cut (e^{−Im(k) w} = e^{−18.5} at
  κ a = 16).

On the 54 nearest offsets alone the damped floors are lower (for example 3·10⁻⁴ for
η = 0.1 at k a = 256), so the far offsets set f64's limit as well.

These are inputs to S6 and D3, which take up precision on the high-frequency path.

## 6. Damping and Yukawa: the norm

p₁ for strong damping, calibrated / formula:

| κ a | η = 1, 10⁻⁶ | η = 1, 10⁻¹⁰ | imaginary, 10⁻⁶ | imaginary, 10⁻¹⁰ |
| --- | --- | --- | --- | --- |
| 0.016 | 30 / 22 | 53 / 35 | 30 / 22 | 53 / 35 |
| 1 | 32 / 22 | 57 / 35 | 32 / 22 | 57 / 35 |
| 4 | 39 / 23 | 65 / 36 | 40 / 23 | 65 / 36 |
| 8 | 52 / 27 | 76 / 39 | 52 / 27 | 78 / 39 |
| 16 | 76 / 36 | 102 / 47 | 76 / 36 | 102 / 47 |
| 32 | 128 / 56 | 155 / 67 | 125 / 56 | 152 / 67 |

With strong damping, p₁ grows about linearly in κ a: 2–2.5 times the formula at κ a = 16
to 32, and p* up to 105 degrees above the closed form of §3. The reason is the norm. Across a
V-list offset the damped pair field e^{−κ|x − y|}/|x − y| is many decades below the terms
of its expansion, so a relative error of ε in that pair field asks for an absolute accuracy
far beyond what the undamped sizes of the coefficients need. In the full FMM the far field
of a damped kernel is small next to the near field, which makes the per-pair relative norm
a large over-estimate there. That is why §5.5 prunes such levels outright once the decay
across a box falls below θ_D ε.

η = 0.1 sits between the two. Its p₁ is within 4 degrees of real k at k a ≤ 8, up to 10
above it at k a 11 to 45 (where Im(k) a is 1 to 4.5), equal at 64, and 2 to 10 below it
from k a = 90.5. Its multi-level δ is larger than real k's and grows with Im(k) a (§4).

The calibration of decaying kernels therefore depends on the norm: question 2 of §8.

## 7. Consequences

- **S6 (CH0.6)** takes P = p* + 1 from `results/summary.md`, or the closed form for real k,
  instead of the formula. From k a = 32 that is 5–15 % shorter, which
  shortens the transfer functions and widens the classes on the diagonal form (§3.6).
- **S8, final pass (CH0.8b)**: `examples/census.rs -- sweep` uses `model::degree`, the
  formula. It takes p* the same way. The low-frequency levels then cost more than §7
  assumed: p ≈ 30 instead of 22 at 10⁻⁶, and 55 instead of 35 at 10⁻¹⁰, in the per-pair
  worst case. The high-frequency levels cost less.
- **§7**, the initial cost model, inherits both corrections through CH0.8b.
- **CH2.3, CH3.3, CH4.3** confirm p in the full FMM against the direct sum, with the global
  norm of §9.1. The per-pair worst case of this spike is an upper bound: Laplace's own
  calibration chose degree 12 at 10⁻⁶ in the potential, where this spike's per-pair
  figure is 19 (§3).
- **CH1.2** builds the L2L tables: §2.4 and question 3.
- **CH1.1** writes the scale of §3.15 item 3 from CH0.3's sign-off. This spike's scale (§2.3)
  is a range device for test degrees, not a proposal.

## 8. Questions for sign-off

Signed off on 2026-10-29: "1–4 as recommended".

1. **The calibrated degree.** Take p* of `results/summary.md` (C1's p₁ plus the largest δ
   of the multi-level chains that use the level, the margin ε/2, the per-pair relative ℓ2
   norm of potential and gradient, the worst of the 316 offsets and three placements) as
   S7's output for S6, CH0.8b and §7. For real k, use the closed form
   max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈k a + (ln 1/ε)^{2/3} (k a)^{1/3}⌉), which never under-predicts
   it at the grid points. CH3.3 confirms it in the full FMM. *Recommended.*
   Alternatives: the uniform placement alone, or the potential alone, which are up to 19
   and 14 degrees lower at low frequency; or the formula of §3.5, which under-predicts the
   low-frequency levels by up to 20 degrees.
2. **The norm for decaying kernels** (η = 1, Yukawa). The per-pair relative norm asks 2–2.5
   times the formula at κ a = 16 to 32 (§6). *Recommended*: keep the measured tables as
   upper bounds, and let CH2.3 calibrate Yukawa's p in the full FMM with the global norm,
   together with pruning and θ_D. Until then S6 and CH0.8b use them only below the pruning
   cut. Alternative: a follow-up spike measuring the chains with an error relative to the
   undamped scale (for example the kernel at the near-field distance), which would give a
   table close to real k's.
3. **The L2L tables (S2's sign-off, question 1).** The outer-sphere projection that S2
   signed off for Re(k) a > 40 is accurate in M2M's weights, but not for L2L with complex
   k (§2.4). *Recommended*: CH1.2 builds the L2L direction by the recursion where it is
   stable (the shift's argument is only k a/2), checks every table in both directions'
   weights, and tests an inner-sphere projection (j onto j by least squares, S2's
   `projection::regular_inner`, so far real k only) for the remaining range. The signed-off rule then holds
   for M2L and M2M only. Alternative: projection on an inner sphere for both directions.
4. **The proposal's next draft** replaces §3.5's formula and its example tables by p* and
   the closed form. It also says that p is set by the low-frequency plateau below k a ≈ 8
   and by k a + (ln 1/ε)^{2/3} (k a)^{1/3} above, and records §6's finding for Yukawa and
   strong damping. *Recommended.*

## Files

- `src/real.rs`, `src/complex.rs`: the `Real` trait, f64 and the double-double type; complex
  numbers.
- `src/radial.rs`, `src/harmonics.rs`, `src/rotation.rs`, `src/coaxial.rs`: generic ports
  of S2's radial functions, S2's basis, S1's rotation blocks and S2's coaxial tables.
- `src/chain.rs`: the operators, the scales, the direct sum and the norms.
- `src/geometry.rs`: offsets, children, the placements, the charges, a seeded generator.
- `src/calibrate.rs`: the four chains over offsets, placements and degrees, in parallel.
- `tests/blocks.rs`, `tests/coaxial.rs`, `tests/chains.rs`: tests 1–8 of the plan.
- `examples/sweep.rs`: the calibration (`c1`, `multi`, `f64all`, `summary`).
- `results/`: `c1.csv`, `c1_curves.csv`, `multi.csv`, `f64_all.csv`, `summary.md`, and
  `chains.log`, the output of test 6 and of the L2L record.
