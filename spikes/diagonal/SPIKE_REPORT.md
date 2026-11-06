# Stability of the diagonal form: spike report (Helmholtz CH0.6, S6)

Helmholtz task CH0.6 ([docs/helmholtz/tasks/CH0.6.md](../../docs/helmholtz/tasks/CH0.6.md)).
The question is proposal §10 CH0.6, with §3.5, §3.6, §5.4, §5.6, §6.5 and Appendix B.
Written on 2026-11-06 from branch `tbetcke/h-ch0-6`. The sweep ran on locust (72 Grace
cores, release build, rustc 1.99.0); the tests and the seed comparison on the M6.

## Summary

The acceptance is met with the corrected bound of question 2 (a smaller C_T and a damping
factor), pending its sign-off: the 2× criterion holds by construction of C_T, and without the
damping factor the damped cases exceed the bound by up to 10⁴⁴. Seven questions go to
sign-off (§9). Raw rows: 23,188
(unit, class) runs, 682 units, 188.7 core-hours on locust; the rows (68 MB) are archived in
`~/projects/fmm-archive/locust/runs/ch0-6/`, not committed; `results/summary.md` is their
summary (`examples/summary.rs`), `results/times.md` the prototype times.

1. **The bound holds with a smaller constant and a damping factor.** With T computed in
   double-double and rounded to f64 (as §6.2 stores tables), the measured rounding of the
   diagonal form against double-double RCR follows
   E = C_T u |k||t| maxⱼ≤₂P₋₂ (2j + 1)|hⱼ(k|t|)| · e^{Im(k)(|t| + 2a)} + 176 u with
   **C_T = 0.153** and **θ = 1/2**: the largest ratio measured/E is exactly 2 by construction,
   at real k a 2.83, p 7, |t|² = 4 (§3). The proposal's C_T ≈ 2 is the seed's, whose model
   builds T in f64; in 3D with T from extended precision C_T is 13× smaller (§3, §6). For Im k > 0 the per-pair relative error grows like e^{Im(k) d} with d the largest
   source–target distance of the pair; e^{Im(k)|t|} alone under-predicts by up to 610 at
   the corner placement and 1.6·10⁵ at the W1 patches (§3).
2. **T must come from extended precision.** T built in f64 (the seed's model) exceeds that
   bound by up to 47×; interpolating T in ŝ · t̂ from a double-double table keeps the bound's
   form with C_T = 0.23 (1.5×) and builds 11× faster at p = 256 (§5, §8).
3. **Gumerov and Duraiswami's reading holds where S6 can test it** (a the circumradius,
   bandwidth 2P − 2, grid (2P − 1)(4P − 3), one-box separation, as their 2007 report's text
   gives it): the spike's chain with T built in f64 gives, in 3D at §3.6's points, 0.2–0.5×
   the 1-D seed's errors on every level, so §3.5 and §5.4 need no correction of the scheme;
   the formula's p_l are 144, 82, 50, 33 as in §3.5. Open: their eq. (25) as extracted reads
   ½(3 ln 1/ε)^{2/3} where §3.5 has (3/2 ln 1/ε)^{2/3} (question 4), and §3.6's level-switch
   table (their switch at P − p_hi ≤ 2) is not redone (§6).
4. **Classes on RCR** (§4): with the bound, at kD = 500 and 10⁻⁶ with the formula's p, level
   2 none, levels 3 and 4 the class |t|² = 4, level 5 the classes 4, 5, 6 (0 / 3 / 3 / 29 %
   of an interior box's V pairs; the proposal had 0 / 3 / 16 / 29 %); with the calibrated p
   no class at all on levels 2–5. The measurement agrees with the bound or names one class
   fewer.
5. **f32 (D3)**: for real k f32 meets 10⁻³ on all 34 classes at the calibrated p from k a = 8
   up, and 10⁻⁴ from k a = 45 (33 or 34 of 34 from 16); with η = 0.1 most classes, with
   η = 1 none at k a ≤ 5.7, above which those levels are pruned at 10⁻³ and 10⁻⁴ (§5). The bound in u = 2⁻²⁴ holds with C_T = 0.18.
6. **The bandwidth is 2P − 2.** P − 1 truncates to O(1); ⌈3(P − 1)/2⌉ is not the truncated
   M2L and fails at high k a, though its rounding is far smaller at small k a (§6).
7. **Times** (§8, prototypes, one Grace core): RCR per pair costs 5× the diagonal pair at
   p = 24 and 53× at p = 256.

## 1. Method

### 1.1 The chain (`src/signature.rs`, `src/transfer.rs`, `src/run.rs`)

Level 0 of S7's chains (`spikes/truncation`): boxes of width w = 1, the source box S at
the origin, the target box at an offset t. The kernel, its expansions and normalisation
are S7's: G = e^{ik|x−y|}/(4π|x−y|) = ik Σ jₙ(k r_<) hₙ(k r_>) Σₘ Yₙᵐ Yₙᵐ in S7's real
orthonormal basis, multipoles Mₙᵐ = Σ q jₙ(k|y − c_s|) Yₙᵐ, locals evaluated as
ik Σ Lₙᵐ jₙ Yₙᵐ, unscaled (k a ≥ 2.8). The diagonal M2L of proposal §5.4 for P = p + 1:

- **S → F**: F(ŝ) = Σₙ (−i)ⁿ Σₘ Mₙᵐ Yₙᵐ(ŝ), the far-field signature
  (Σ q e^{−ikŝ·(y − c_s)} = 4π F), on the grid of 2P − 1 Gauss–Legendre nodes in cos θ and
  N_φ = 4P − 3 azimuths padded to a multiple of 4 (so N_φ = 4P), by a Legendre step per
  order and a DFT per ring written as a product;
- **the diagonal step**: G(ŝ) = T(ŝ) F(ŝ) with
  T(ŝ) = Σ_{l ≤ L} (2l + 1) iˡ hₗ(k|t|) Pₗ(ŝ · t̂);
- **F → R**: Lₙᵐ = iⁿ ∫ Yₙᵐ G dŝ by the grid's quadrature, exact for degree 4P − 4.

This follows from G = (ik/16π²) ∫ e^{ikŝ·(x − c_t)} T(ŝ) e^{−ikŝ·(y − c_s)} dŝ, the
identity of the seed (Appendix B.1). With L = 2P − 2 the chain is the truncated M2L that
RCR applies: in double-double it agrees with S7's RCR to 10⁻²⁰ (test
`diagonal_is_rcr_in_double_double`).

### 1.2 What is measured

Per placement (S7's three: uniform, corners, W1 patches; S7's C1 clouds with its seeds,
128 sources and 64 targets per box), the multipole of S by S7's P2M in double-double. Per
class representative t:

- the reference: S7's RCR (rotation, (S|R), rotation back) in double-double at degree p,
  and the direct sum in double-double;
- the diagonal chain in a working precision: the multipole and T rounded to it, S → F,
  T F and F → R in it, the local widened back; every local evaluated by S7's L2P in
  double-double, so the errors are the M2L's;
- the errors: S7's relative ℓ2 norms over the targets of φ and ∇φ (§9.1); the report
  takes the worst of the two and the worst placement. **round** is the diagonal against
  RCR (its rounding), **total** the diagonal against the direct sum, **trunc** RCR against
  the direct sum.

### 1.3 The variants

- f64 and f32 with T rounded from double-double at every sample (the acceptance's), at
  the bandwidths L = 2P − 2 (the proposal's), ⌈3(P − 1)/2⌉ and P − 1;
- f64 with **T built in f64** (`TransferF64`): the terms (2l + 1) iˡ hₗ rounded to f64,
  Pₗ(μ) by the recurrence and the sum in f64, μ from f64 directions: what a host table build
  in working precision gives, and the seed's model;
- f64 with **T interpolated** in ŝ · t̂ (`Interp`): a table of T on a uniform grid in
  ψ = arccos(ŝ · t̂), oversampled 3×, in double-double rounded to f64, read by
  Gaussian-regularised sinc interpolation (36 terms each side) at a position computed in
  double-double; the O(p²)-per-class build of §5.4 and R10.

### 1.4 The sweep (`examples/sweep.rs`)

Kinds real k, k = κ(1 + 0.1i) and k = κ(1 + i), each up to the pruning cut per ε
(Im(k) w ≤ ln(1/ε), §5.5's test at its conservative end); Yukawa has no high-frequency
regime. k a = 2^{j/2}, j = 3..17 (2.83 to 362), and 400; ε = 10⁻³ … 10⁻¹⁰; P = p + 1 from
proposal §3.5's formula, from S7's p* (`spikes/truncation/results/summary.md`; its
closed form above k a = 256) and from CH3.3's `DegreeRule::CALIBRATED` (the FMM's rule,
an addition to the acceptance's two). Every distinct (kind, k a, p) is a unit: 612
units, each with the 34 classes (|t| = 3w on an axis, the control, among them), three
placements and eight variants.

### 1.5 The bound

E(l, c) = C_T u |k||t| max_{j ≤ L} (2j + 1)|hⱼ(k|t|)| (§3.6), its term computed in f64
by the upward recursion with rescaling (`log10_term`, against S7's double-double Hankel
functions in a test), so it stays finite beyond the exponent range.

Tests: `cargo test -p nd-fmm-spike-diagonal --release -- --include-ignored`, 18 tests
(`tests/chain.rs`, `tests/parts.rs`): the plan's tests 1–11. Notably the chain in double-double
is S7's RCR to 10⁻²⁰ (real k and η = 0.1, k a up to 16, where the plan said 32: at 32 the
double-double test takes minutes, and the sweep compares the two at every k a), the identity
S → F → R to 10⁻²⁸, `l2p_many` bit for bit S7's L2P, a unit's rows bit for bit identical at
1 and 3 threads, and the seed's geometry reproduced (test 8, §6).

## 2. What the sweep measured

612 units (kind, k a, p) of the plan's sweep plus 70 at §3.6's four levels (`extra.csv`),
each with the 34 classes, three placements and eight variants. Every number below is the
worst over the placements of max(potential, gradient), the relative ℓ2 norm over the targets.
**round** is the diagonal form against double-double RCR at the same p (rounding only; the
truncation is RCR's), **total** against the direct sum. Points whose bound lies below 10⁻¹³
in f64 form the floor (the chain's own rounding, about u times the conditioning of S → F
and F → R); above it the ratio measured/E is fitted. The per-class rounding and total of every
(kind, k a, p, rule's p, placement, variant), potential and gradient apart (output 1), are the
archived rows; `results/summary.md` reduces them to the worst placement and norm.

## 3. C_T and θ (output 2)

`results/summary.md` §1–§2 and §7. The ratio measured / (u 10^term), C_T = 1, for f64 with
T rounded from double-double at L = 2P − 2:

| kind | no damping factor: median / max | with e^{Im(k)(\|t\| + 2a)}: median / max |
| --- | --- | --- |
| real | 1.1·10⁻² / 0.31 | 1.1·10⁻² / 0.31 |
| η = 0.1 | 0.17 / 2.8·10²⁰ | 2.8·10⁻⁴ / 0.16 |
| η = 1 | 4.7·10⁵ / 3.9·10⁴⁴ | 4.7·10⁻⁷ / 5.7·10⁻⁴ |

- **Real k.** The ratio lies in [3.5·10⁻⁴, 0.31] over 7,184 points; the largest at k a 2.83,
  p 7, the nearest class. C_T = 0.153 (half the largest) and θ = 1/2: E ≤ θ ε then implies a
  measured rounding ≤ ε, and E never under-predicts by more than 2×. Below the floor the
  rounding stays below 176 u. The bound is not tight: at C_T = 0.153 its median
  over-prediction is 14× (the 3D quadratures average the rounding of T over the sphere, the
  1-D seed does not).
- **Damped k.** Without a factor the bound fails by up to 10⁴⁴. The local's field decays
  like e^{−Im(k) d} over the pair's source–target distances d, while the rounding keeps the
  size of the undamped terms. With e^{Im(k)|t|} the largest ratio is 14 on the uniform
  placement (η = 1, k a 16), 610 at the corners (η = 0.1, k a 181) and 1.6·10⁵ on the W1
  patches, whose points sit up to 2a = √3 w beyond |t| (counted from the archived rows per
  placement; the summary takes the worst placement); e^{Im(k)(|t| + 2a)}, the largest distance a pair can have, fits all three
  (largest ratio 0.16 with η = 0.1, below real k's 0.31), so one C_T serves every kind. It
  over-predicts the damped cases (median 10⁻⁴ to 10⁻⁷ at C_T = 1), so damped levels send
  more classes to RCR than needed (§4).
- **Variants.** The same fit for the other variants at 2P − 2 (largest ratio over kinds; C_T
  half of it): T interpolated 0.45 → C_T = 0.23; f32 with T rounded from double-double 0.35
  → 0.18, its floor 44 u (u = 2⁻²⁴); T built in f64 7.2 → 3.6.

The correction of §3.6 (question 2): E(l, c) = C_T u k|t_c| maxⱼ≤₂P_l₋₂ (2j + 1)|hⱼ(k|t_c|)|
e^{Im(k)(|t_c| + 2a_l)} + C₀ u, with C_T = 0.15 for T rounded from extended precision (0.23
interpolated, 0.18 in f32), C₀ = 200 (50 in f32) and θ = 1/2. For real k the factor is 1. It still
depends only on k, w_l, P_l and the class (constraint 9).

## 4. Classes on RCR per (k a, ε) (output 3)

`results/summary.md` §3 has, per kind and rule (the formula of §3.5, S7's p*, CH3.3's
calibrated rule), the classes whose diagonal form fails by the bound E ≤ θ ε and by the
measurement (round > ε), and the share of an interior box's 189 V pairs on RCR by the bound.
For real k the bound names the measured classes, or one class more; it never misses one.

§3.6's table at kD = 500, ε = 10⁻⁶, redone (`results/summary.md` §3, rows k a 108.253 …
13.532; classes as |t|²/w², share of an interior box's V pairs):

| Level | k a | formula p: by the bound / measured (share) | calibrated p: by the bound / measured | proposal §3.6 |
| --- | --- | --- | --- | --- |
| 2 | 108 | [144] none / none (0 %) | [131] none / none | none (0 %) |
| 3 | 54 | [82] 4 / none (3 %) | [72] none / none | 4 (3 %) |
| 4 | 27 | [50] 4 / 4 (3 %) | [41] none / none | 4, 5 (16 %) |
| 5 | 13.5 | [33] 4, 5, 6 / 4, 5 (29 %) | [25] none / none | 4, 5, 6 (29 %) |

At 10⁻⁸ with the formula: level 2 class 4, level 3 classes 4 and 5, level 4 classes 4 to 6,
as §3.6 says, and level 5 18 classes (62 %), where §3.6 says eight; with the calibrated p: levels 2 and 3 none, levels 4
and 5 class 4 (3 %). The W1 sphere-surface shares of §3.6 are not redone (the sweep has no
tree); CH0.8b recounts them with these constants.

Below k a ≈ 8 the diagonal form fails on most classes at ε ≤ 10⁻⁶ whatever the rule, which is
where the regime classifier puts low-frequency levels anyway. With η = 1 the bound sends
every class to RCR from k a = 8 at the calibrated p, while the measurement fails only a few
(the over-prediction of §3): question 7.

## 5. f32 (output 4, D3)

f32 at L = 2P − 2 with T rounded from double-double, the classes meeting ε in both the total
and the rounding (worst placement), at the p of each rule:

| kind | ε = 10⁻³, calibrated p | ε = 10⁻⁴, calibrated p |
| --- | --- | --- |
| real | 21–30 of 34 at k a ≤ 5.7; 34 of 34 from k a = 8 to 400 | 12–32 below k a = 16; 33–34 from 16 to 32; 34 from 45 to 400 |
| η = 0.1 | 30–34 of 34 from k a = 8 to 45 (beyond: pruned) | 24–29 of 34 from 8 to 45, 12 at 64 |
| η = 1 | none at k a ≤ 5.7 (beyond: pruned) | none at k a ≤ 5.7 (beyond: pruned) |

(`results/summary.md` §4, with the formula's p beside.) The formula's larger p passes fewer
classes at small k a, and all from k a ≈ 45 (10⁻³) or 108 (10⁻⁴).
The bound in u = 2⁻²⁴ holds with C_T = 0.18 (largest ratio 0.35). So the diagonal form in f32
is stable for real k at ε ≥ 10⁻⁴ on every class of every high-frequency level that the
calibrated rule produces from k a ≈ 45 (at 10⁻³ from k a ≈ 8), and the per-class bound
decides the rest. The RCR fallback in f32 and the low-frequency levels are CH7.1's.

## 6. The reading of Gumerov and Duraiswami (output 6)

**Their paper.** Their 2007 technical report in the University of Maryland's repository
(https://drum.lib.umd.edu/items/719d51cc-d3da-44cd-b155-b09e3b922fad) confirms the proposal's reading: the expansion domain of a box on level l is a sphere of
radius a_l concentric with the box (the circumradius); the transfer function sums to
2p_l − 2; the signature grid is (2p_l − 1) × (4p_l − 3); the switch to signatures is taken
at the level where p − p_high ≤ 2, and p = p₀ + p_high when p − p_high > 5. Their prescribed
accuracies were loose: 10⁻⁵ for the FMM tests, about 1 % in the BEM.

Their eq. (25), as the PDF's text layer gives it, reads
p_high = ka + ½(3 ln 1/ε)^{2/3}(ka)^{1/3}; §3.5 (citing the 2009 paper, eq. 18) has
(3/2 ln 1/ε)^{2/3}, a transition term 1.26× larger. The extraction may have garbled it; if
not, the two papers differ. It changes only §3.5's formula, which the calibrated rule
(CH3.3) has replaced for the FMM: question 4.

**The seed in 3D.** §3.6's points (kD = 500, 10⁻⁶, the formula's p, the nearest axis offset
|t| = 2w and the axis control 3w; `results/summary.md` §3b), worst placement:

| Level | k a | p | 1-D seed, 2w | T built in f64, 2w | T from double-double, 2w | E (§3), 2w | double-double, 3w |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 108.3 | 144 | 1·10⁻⁷ | 2.9·10⁻⁸ | 4.0·10⁻¹⁰ | 1.8·10⁻⁸ | 2.4·10⁻¹⁵ |
| 3 | 54.1 | 82 | 8·10⁻⁵ | 1.7·10⁻⁵ | 2.6·10⁻⁷ | 5.4·10⁻⁶ | 1.9·10⁻¹⁵ |
| 4 | 27.1 | 50 | 1.5·10⁻² | 7.4·10⁻³ | 1.6·10⁻⁴ | 1.9·10⁻³ | 5.9·10⁻¹⁵ |
| 5 | 13.5 | 33 | ≈ 10 | 4.2 | 0.13 | 0.64 | 7.9·10⁻¹¹ |

With the seed's model (T built in f64) the 3D errors are 0.2–0.5× the seed's on every level:
the reading holds in 3D, and §3.5 and §5.4 need no correction of the scheme. Not redone:
§3.6's level-switch table (E at their switch P − p_hi ≤ 2 per ε) and the check that a
prescribed 10⁻⁴ gives about 10⁻⁶ in their runs; both concern their FMM's switch, which the
per-class rule replaces. The 3D sums
over the sphere average the rounding the 1-D model accumulates on the axis, and building T
in extended precision removes most of the rest. Test 8 (`seed_reproduced_on_the_axis`)
runs the seed's own geometry (one source and one target on the axis) in 3D: 0.11–1.2× the
seed's errors at 2w, never far above.

**The bandwidth** (`results/summary.md` §6, real k, the formula's p): L = P − 1 truncates T
to an error of order 1 from k a ≈ 32; ⌈3(P − 1)/2⌉ is no longer the truncated M2L, so its
error grows with k a (0.08–0.46 at k a ≥ 256); 2P − 2 matches RCR wherever it is stable.
At k a 5.7–45 and ε ≤ 10⁻⁶, though, ⌈3(P − 1)/2⌉ stays within a few times ε where 2P − 2
loses orders of magnitude to rounding (k a 16, 10⁻⁶: 2.2·10⁻⁶ against 1.1·10⁻²), and it
meets 10⁻⁶ at k a 22.6–45: a shorter T trades a truncation for stability. Using it
would need its own truncation analysis; this spike only records it (§9, follow-ups).

## 7. Interpolated T (output 5)

A table of T in ψ = arccos(ŝ · t̂), oversampled 3× (M = ⌈3L⌉ + 1 points on [0, π]), in
double-double rounded to f64, read by Gaussian-regularised sinc interpolation with 36 terms
on each side at a ψ computed in double-double (`results/summary.md` §5). Its rounding is
0.3–6× that of T rounded per sample (median 1.2), largest ratio to the bound 0.45 against
0.31: the bound holds with C_T = 0.23. A table of M values costs O(L) Legendre sums instead
of O(L²) samples, so the per-class build drops from 5.3 s to 0.49 s at p = 256 (§8).
Question 3.

## 8. Prototype times (output 7)

`results/times.md`: locust, one Grace core (`taskset -c 8`), release, f64, medians of 7, two
runs within 10–20 % of each other; the class (0, 2, 0), k a set so that p is the formula's
degree at 10⁻⁶. These are the spike's prototypes: S → F and F → R are a Legendre step per
ring and order plus a DFT written as a product, O(p³) per box, not CH4.x's transforms.

| p | N_l | diagonal pair (µs) | RCR pair (µs) | RCR / diagonal | S → F + F → R per box (ms) | T per class: double-double / interpolated / RCR tables (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | 4,900 | 4.0 | 21 | 5.3 | 0.31 | 5 / 5 / 0 |
| 48 | 19,012 | 19 | 161–183 | 8–10 | 2.7 | 36 / 18 / 0 |
| 96 | 74,884 | 100–104 | 1,291–1,303 | 13 | 22 | 285 / 70 / 45 |
| 128 | 132,612 | 119–138 | 3,022 | 22–25 | 53 | 673 / 124 / 120 |
| 256 | 527,364 | 548–557 | 29,034–29,209 | 52–53 | 513–526 | 5,341 / 491 / 1,509 |

Per box with 189 V pairs at p = 256 RCR takes 5.5 s and the diagonal form 0.1 s plus 0.5 s for
the two conversions (prototype). The sweep itself took 188.7 core-hours.

## 9. Questions for sign-off

1. **D3, f32 on high-frequency levels.** *Recommended*: high-frequency levels may run the
   diagonal form in f32 at ε ≥ 10⁻⁴ for real k and weak damping, each class by the bound in
   u = 2⁻²⁴ with C_T = 0.18 and θ = 1/2, failing classes on RCR; f64 below 10⁻⁴ and for
   η ≈ 1. CH7.1 settles the f32 RCR and the low-frequency levels. Alternative: keep f64 on
   every high-frequency level (the proposal's default until S6).
2. **The bound of §3.6.** *Recommended*: E = C_T u k|t| maxⱼ≤₂P₋₂ (2j + 1)|hⱼ(k|t|)|
   e^{Im(k)(|t| + 2a)} + C₀ u with C_T = 0.15, C₀ = 200 and θ = 1/2 (T rounded from
   extended precision), replacing C_T ≈ 2 and θ = 1; §3.6's example tables and the
   "Complex k" paragraph rewritten from §4 and §3 of this report. Alternative: keep C_T = 2
   (at least 13× conservative for real k).
3. **Building T.** *Recommended*: T always from extended precision, either per sample or by
   interpolation in ŝ · t̂ from a double-double table with C_T = 0.23 when interpolated
   (R10); never summed in the working precision (47× above the bound).
4. **§3.5's p_high.** Note the factor ½(3 ln 1/ε)^{2/3} of the 2007 report's eq. (25)
   beside the 2009 paper's (3/2 ln 1/ε)^{2/3} in §3.5, without changing the formula (only
   the uncalibrated rule uses it). Alternative: check the 2009 paper's typesetting first.
5. **CONVENTIONS §3.15 "Signatures"** (below). *Recommended*: add it as proposed, in CH1.3d
   (the extended-precision diagonal form in nd-fmm-ref) or CH4.1, whichever lands first.
6. **The PR's size.** About 2,900 lines in one spike crate, like S7 (#134): one PR, not split.
7. **The damping factor.** The per-pair relative norm makes the damped bound 10⁴–10⁷×
   conservative at η = 1. *Recommended*: keep it (it is safe, and §5.5's pruning already
   drops the most damped pairs), and let CH4.3 measure the classes with the FMM-weighted
   norm of CH1.3c; a looser factor e^{Im(k)|t|} would under-predict at the W1 patches.

### The proposal for CONVENTIONS §3.15 "Signatures"

For a box of degree p (P = p + 1), in the angular basis of §3.15 (S7's):

- **Grid.** 2P − 1 Gauss–Legendre nodes uᵢ = cos θᵢ in ascending order (ring 0 nearest the
  south pole), weights wᵢ; N_φ = 4P − 3 rounded up to a multiple of 4 (so N_φ = 4P),
  φⱼ = 2πj/N_φ; sample (i, j) at index i N_φ + j, direction ŝ = (sin θᵢ cos φⱼ,
  sin θᵢ sin φⱼ, cos θᵢ); quadrature weight wᵢ · 2π/N_φ. Exact for degree 4P − 4.
- **Signature.** F(ŝ) = Σₙ (−i)ⁿ Σₘ Mₙᵐ Yₙᵐ(ŝ), so that a multipole of charges q at y has
  4π F(ŝ) = Σ q e^{−ik ŝ·(y − c)} up to the truncation.
- **Transfer function.** T(ŝ) = Σ_{l ≤ 2P − 2} (2l + 1) iˡ hₗ(k|t|) Pₗ(ŝ · t̂), t = c_t − c_s,
  computed in extended precision and rounded once.
- **Back to a local.** Lₙᵐ = iⁿ Σᵢⱼ wᵢ (2π/N_φ) Yₙᵐ(ŝᵢⱼ) T(ŝᵢⱼ) F(ŝᵢⱼ): the truncated
  M2L of M in S7's normalisation (a local field ik Σ Lₙᵐ jₙ Yₙᵐ), by the identity
  G(x − y) = (ik/16π²) ∫ e^{ik ŝ·(x − c_t)} T(ŝ) e^{−ik ŝ·(y − c_s)} dŝ. How this maps onto
  §3.15's factors and scaled coefficients (s_l) is for CH1.3d to write down.
- **Classes.** An offset o = (a, b, c) belongs to the class of its representative
  (min(|a|, |b|), max(|a|, |b|), |c|) (34 classes for the 316 offsets); its T is the
  representative's read through a sample permutation: T_o at sample (i, j) is T_rep at the
  sample of S ŝ, where S flips the negative components and then swaps x and y if |a| > |b|: x → −x maps j to
  (N_φ/2 − j) mod N_φ, y → −y j to −j mod N_φ, z → −z ring i to 2P − 2 − i, x ↔ y j to
  (N_φ/4 − j) mod N_φ, applied in that order.

## Files

- `src/grid.rs`: the signature grid in double-double and its tables.
- `src/signature.rs`: S → F and F → R per ring.
- `src/transfer.rs`: T in double-double (forward Legendre recurrence with snapshots), T built
  in f64, the bound's Hankel term (`log10_term`), interpolation in ψ (`Interp`).
- `src/classes.rs`: the 34 classes and their sample permutations.
- `src/rules.rs`: the three degree rules (formula, S7's p* from its `results/summary.md`,
  CH3.3's calibrated rule).
- `src/run.rs`: the variants, a unit's run (RCR and direct in double-double, the diagonal
  variants, the shared L2P basis).
- `tests/chain.rs`, `tests/parts.rs`: tests 1–11 of the plan; `seed.py`: Appendix B.1.
- `examples/sweep.rs`, `examples/summary.rs`, `examples/times.rs`: the sweep, its summary and
  the prototype times.
- `results/`: `summary.md`, `times.md`, `sweep.log`. The rows: archived (above).
