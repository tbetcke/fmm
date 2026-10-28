# Workload census, first pass: spike report (Helmholtz CH0.8a, S8)

Helmholtz task CH0.8a ([docs/helmholtz/tasks/CH0.8a.md](../../docs/helmholtz/tasks/CH0.8a.md)).
The question is proposal §10 CH0.8 (first pass: the formula p and the bound of §3.6 with
C_T = 2 and θ = 1), with §3.5–§3.7, §5.2–§5.4, §6.6, §6.7, §7, §9.1 and decisions D11 and
D12. Written on 2026-10-28 from branch `tbetcke/h-ch0-8a` (main at b345e26 plus this spike), on
the M3 Max (16 cores, 64 GB), release build, one rank, one thread. No FMM is run.

## Summary

1. **The census reproduces §7 and §6.6 from real trees** (§2). W1 at kD = 500 by
   `nd-octree` and `nd-fmm-plan` gives 56, 272, 1,160, 4,760, 18,828 and 73,022 occupied
   boxes on levels 2–7 (§7, from a sampled sphere: 56, 272, 1,160, 4,760, 18,748, 74,337),
   the same regimes and V pairs on RCR (0, 6, 23, 37 %), and M2L counts within 4 %.
   At kD = 1000: H on levels 2–6 with 200·10⁹ against 196, levels 7–8 488·10⁹ against 477,
   peak signature scratch 5.35 GiB against 5.0, and multipoles 4.08 GiB against 4.0. The
   replicated top is 93 and 268 MiB per rank for W1 and 237 MiB for the full cube at
   kD = 500, §6.6's numbers exactly. The first-pass model stands. CH0.8b replaces p and the
   constants, not the counting.
2. **D11's rule as written is degenerate; c_leaf = 1 is proposed by a revised rule** (§3).
   The P2P and leaf costs always fall as c_leaf shrinks, so the rule ("the largest c_leaf
   within 10 % of the best P2P and leaf cost") picks the smallest candidate, 0.25 (0.5 on
   one row). That multiplies the total modelled work 3–15× at kD ≥ 20 (90× on W1 at
   kD = 10), because the far field moves down a level per halving. The leaf rule exists to keep leaves low frequency. The
   revised rule is therefore the largest c_leaf with no high-frequency leaf on any workload
   at ε = 10⁻⁶ and 10⁻⁸, which gives **c_leaf = 1**. Its total modelled cost is within 2 %
   of the best (no limit) on every resolved workload to kD = 200, 22–25 % above it on W1 at
   kD = 500 and 1000, where 1.24 λ leaves become 0.62 λ, and 10.4× above it on the
   under-resolved W3 at kD = 500.
3. **No leaf is high frequency at c_leaf = 1 for ε ≤ 10⁻⁶** (§4), over all 43 trees of the
   sweep. Without a limit, 9 trees have H leaf levels at 10⁻⁶ and one at 10⁻⁸ (W3 at
   kD = 500, 2.5 λ leaves); with c_leaf = 1.5 one tree does at 10⁻⁶. **At ε = 10⁻⁴ the
   first-pass rule calls levels H down to 0.5–1 λ boxes.** The far classes (|t|² up to 27
   w²) pass the bound there, and the hybrid counts fewer operations than RCR, so 14 trees
   at c_leaf = 1 have an H leaf level. Only c_leaf = 0.25 avoids that, at 3–15× the work.
   Since the leaf operators do not depend on the regime (§3.6), this is a question for S6's
   constants and for D1's loosest tolerance, not a correctness issue (question 3).
4. **The under-resolved volume is the memory case of R17** (§5). W3 with 10⁶ points at
   kD = 500 (2.0 points per λ³) at c_leaf = 1 is refined to 0.62 λ leaves holding 1.3
   points each. It carries 10.2 GiB of multipoles, and level 6 (1.24 λ, 256,367 boxes) is H
   with **42 GiB of signature scratch** at 10⁻⁶ (56 GiB at 10⁻⁴). By §3.6's rule that level
   runs RCR under any realistic node budget; the census applies the operation count only.
   W1 needs at most 5.35 GiB (kD = 1000, 10⁻⁶).
5. **Most of the far field is low-frequency work** (§5): 122 of 160·10⁹ M2L at kD = 500
   and 488 of 688·10⁹ at kD = 1000 on W1 (10⁻⁶). At 10⁻⁸ the share is 88 % and 84 %,
   because tighter tolerances push the H levels up. This confirms §7 and D6 (H6 before
   H4).
6. **Batch sizes for S5 and S9** (§6): at 10⁻⁶ on W1 to kD = 1000 the H levels convert 56 to
   19,132 boxes per level call (256,359 on the under-resolved W3), with p from 25 to 261.
   Over every ε, host N_φ ranges from 72 to 1,120 and power-of-two N_φ from 128 to 2,048; the
   transforms of one level call are 2P − 1 rings per box, up to 1.3 million (W1, kD = 1000,
   level 6) and 13 million (W3 under-resolved, level 6).
7. **The workloads (D12)** (§7): W1–W4 as §9.1 states them are buildable on one node. W1
   at kD = 1000 (8.0 million points) builds and counts in 13 s at c_leaf = 1, and the whole
   sweep of 378 trees took 22 minutes and 40 GB at most. W2's high-frequency levels equal
   W1's; it adds W and X lists on the low-frequency levels (1.7·10⁹ M2P and P2L at
   kD = 500). The proposal is to keep the set, with W3 at 10⁶ and kD = 500 named as the
   case for the node-budget fallback.

## 1. Method

### 1.1 The workloads (proposal §9.1)

All in the root cube [0, 1]³, so the wavenumber k is kD with D the root's side
(`src/workloads.rs`):

| ID | Points | kD |
| --- | --- | --- |
| W1 | Fibonacci lattice on the sphere of diameter D (less 2·10⁻⁹), N = ⌈100 (kD)²/(4π)⌉ ≈ 7.96 (kD)², ten per wavelength | 10, 20, 50, 100, 200, 500, 1000 |
| W2 | W1 outside the polar cap of half-angle 30° about +z, and inside it the cap points of the lattice of 10 N: ten times the density, 1.60 N in all | 10, 20, 50, 100, 200, 500 |
| W3 | uniform random (SplitMix64, fixed seed), 1000 per λ³ (N = ⌈1000 (kD/2π)³⌉: 4,030 at kD = 10 to 4.03 million at 100); 10⁶ at kD = 200 and 500 (under-resolved: 2.0 per λ³ at 500) | 10, 20, 50, 100; 200, 500 |
| W4 | the four distributions of the Laplace gates (`nd_fmm_validate::fmm_accuracy::Distribution`: cube, sphere, Plummer, clusters), the same generators and parameters (this spike's seed, not the gates'), mapped from [−1, 1]³ to the root cube; N = 10⁵ and 10⁶ | 1, 10, 100 |

### 1.2 The trees: the prototype maximum leaf width (proposal §3.7)

nd-octree has no maximum leaf width (CH3.0 adds one behind its Laplace check). The spike
refines in Morton keys by nd-octree's own rule, a box with more than n distinct level-16
keys is refined (`create_local_tree`), plus the width rule, a box that holds a point and is
wider than c_leaf λ is refined; nothing below level 16. The leaves of that complete tree go
to `Octree::new` as one level-16 key each with `max_fine_keys = 1`: every refined box then
holds at least eight keys and every leaf one, so nd-octree rebuilds the same tree and 2:1
balances it exactly as it balances the Laplace trees (ghost children on, as `Fmm` builds
them). Test 1 shows the construction equal, leaf for leaf, to `Octree::new` on the points
when there is no width limit, on W1, W2, W3 and the four W4 distributions; test 2 that
with the limit every occupied leaf is at most c_leaf λ wide, no leaf above level 16 holds
more than n keys, and the tree refines the tree without the limit.

n = 128, Laplace's host default from p = 12 (`default_max_points_per_leaf`); n = 64 as a
sensitivity row in the c_leaf study. c_leaf = 0.25, 0.5, 0.75, 1, 1.5, 2 and none. The
width rule sends every occupied leaf to the shallowest level whose width is at most
c_leaf λ, so c_leaf acts in octaves: at kD = 500 (λ = 0.0126) the values 0.25, 0.5, 0.75, 1,
1.5 and 2 give leaf levels 9, 8, 7, 7, 6 and 6 at the shallowest (0.16, 0.31, 0.62, 0.62,
1.24 and 1.24 λ).

### 1.3 The counts (`src/census.rs`)

`Plan::new` of the tree gives the lists; the census counts per level: the boxes the plan
holds and those with a point (occupied); the leaves and the occupied leaves, their points;
the U entries between occupied leaves and the P2P point pairs (n_t Σ n_s over the near list,
the leaf itself included); the V pairs between occupied boxes per distance class
|t|²/w² (15 classes) and per reflection class (34, under x → −x, y → −y, z → −z and
x ↔ y, proposal §5.4); for each occupied box the mask of distance classes it is a V source
(and target) of; the W and X entries between occupied boxes with their point counts. The
plan holds empty boxes in its rows (about half the boxes of a surface workload are empty);
the counts here are those of an FMM that skips them, beside the plan's own entry counts.
Test 3 recomputes every count (boxes, leaves, the entries of every list with and without
empty boxes, the V pairs per distance and per reflection class, the masks, the points of
P2P, W and X) by brute force from the list definitions of
`nd_fmm_plan::interaction_manager` (closed-cube adjacency over every pair of boxes, no use
of the plan) on four small adaptive trees with W and X lists.

### 1.4 The model (`src/model.rs`)

- **p_l** by the formula of §3.5 at k a_l, a_l the circumradius √3 w_l/2 (test 5
  reproduces §3.5's tables). No S7 calibration (CH0.8b).
- **The bound** E(l, c) = C_T k|t_c| u max_{j ≤ 2P−2} (2j + 1)|h_j(k|t_c|)|, C_T = 2,
  θ = 1, u = 2⁻⁵³, |h_j| by the upward recurrence (test 6: within 3.5·10⁻¹⁵ of spike S3's
  arbitrary-precision reference on its 36 real-argument sets). Test 7 reproduces §3.6's
  classes on RCR at kD = 500 (ε = 10⁻⁶ and 10⁻⁸) and its widths from which the nearest
  class passes: E is not monotone in w (p moves in integers), so the first width that passes
  and the width from which every wider box passes bracket §3.6's two-digit values: 1.54 and
  1.90 λ around 1.8 λ at 10⁻⁴, 15.9 and 17.7 λ around 17 λ at 10⁻⁶, 42.3 and 44.0 λ around
  42 λ at 10⁻⁸.
- **The regime** of a level ≥ 2: H if some distance class present passes E ≤ θ ε and the
  hybrid M2L (conversions, diagonal pairs, RCR for the rest) counts fewer operations than
  RCR on every pair; else L (Rotation on every pair). Every passing class present runs
  diagonal: §3.6's per-class test ("faster than RCR for that class") is not applied
  separately, since a diagonal pair, (2P − 1) N_φ ≈ 8P², costs less than an RCR pair, 2P³,
  for every P here once the conversions are paid. Constraint 9 holds: the passing classes
  depend on k, w_l, p_l and ε only, the regime also on the global tree's counts (which
  classes occur, how many pairs and boxes), never on charges or a rank's share. Levels 0
  and 1 carry no V list.
- **Grids**: N_φ on the host the smallest multiple of 4 at least 4P − 3 with no prime factor
  above 7 (§6.7: 588 at p = 144, 1080 at 261, 216 at 50), and as a power of two.
- **Operations**, complex multiply-adds with §7's conventions: an RCR translation 2P³ (from
  p₁ to p₂: P₁³ + P₂³), so M2M and L2L count one translation per occupied box on levels
  ≥ 3; a diagonal pair (2P − 1) N_φ; a conversion S → F or F → R 2P³ + (2P − 1)² N_φ; P2M,
  L2P, M2P and P2L (p + 1)² per point and expansion; P2P in point pairs. Test 8 reproduces
  §7's level 2 of W1 at kD = 500 from §7's own box count (6.5 against 6.6·10⁹; §7 rounds
  the φ step to 16P³).
- **Memory**, f64: multipoles and locals (p + 1)² complex per occupied box (real multipoles
  half); signature scratch (2P − 1) N_φ complex per converted box, sources and targets; the
  transfer functions of the passing reflection classes present; the (S|R) tables of the RCR
  distances present, Σ_{m ≤ p} (p + 1 − m)² complex each; the rotation block sets, one for
  M2M and L2L and 47 more where RCR M2L runs, (p + 1)(2p + 1)(2p + 3)/3 reals each (test 8:
  0.09 MiB at p = 20, 31 MiB at 144, 183 MiB at 261, about 3 MiB per (S|R) table at p = 82).
  The replicated top per rank: the multipoles of levels 2 to the deepest level whose occupied
  boxes from level 2 down stay within Phase 5S's cap of 4,096 boxes, and proposal §6.6's
  levels 2–4; per node at 40 ranks (Kathleen) and 72 (locust).

## 2. Agreement with §7 and §6.6

W1 at kD = 500, ε = 10⁻⁶, c_leaf = 1 (leaves on level 7, as §7 assumed), against §7's table
(from `results/w1-levels.md`):

| Level | Box | p | Occupied (§7) | Mean V (§7) | Regime | V on RCR (§7) | Signatures GiB (§7) | M2L 10⁹ (§7) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 19.9 λ | 144 | 56 (56) | 44.7 (44.7) | H | 0 % (0 %) | 0.28 (0.28) | 6.61 (6.6) |
| 3 | 9.9 λ | 82 | 272 (272) | 44.9 (44.9) | H | 6 % (6 %) | 0.45 (0.44) | 7.01 (7.0) |
| 4 | 5.0 λ | 50 | 1,160 (1,160) | 46.6 (46.6) | H | 23 % (23 %) | 0.75 (0.70) | 9.87 (9.6) |
| 5 | 2.5 λ | 33 | 4,760 (4,760) | 47.9 (47.9) | H | 37 % (37 %) | 1.33 (1.26) | 14.72 (14.7) |
| 6 | 1.2 λ | 25 | 18,828 (18,748) | 46.8 (46.7) | L | all | – | 30.96 (30.8) |
| 7 | 0.6 λ | 23 | 73,022 (74,337) | 45.2 (46.2) | L | all | – | 91.25 (95.0) |

The signature scratch is larger here by the host grid's padding (N_φ = 588, 336, 216, 140
against 4P − 3 = 577, 329, 201, 133). The plan holds 1.1 to 4× more boxes than are
occupied on the levels of W1 (empty boxes inside the sphere and the 2:1 balance).
The counts skip them.

## 3. c_leaf (D11)

`results/c-leaf.md` lists every W1 and W3 tree at n = 128 and 64 with its leaf levels,
occupied leaves, P2P pairs, leaf cost, far field and total (ε = 10⁻⁶, w = 4). The rows
that decide, n = 128:

| Workload | kD | c_leaf | leaf levels | occupied leaves | P2P pairs | leaf cost rel. best | total rel. best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | 100 | 1 | 4–5 | 1,300 | 7.8·10⁷ | 5.28 | 1.02 |
| W1 | 100 | none | 3–5 | 1,298 | 7.9·10⁷ | 5.59 | 1.00 |
| W1 | 500 | 0.25 | 9 | 949,119 | 5.0·10⁷ | 1.00 | 9.51 |
| W1 | 500 | 1 | 7 | 73,022 | 8.0·10⁸ | 2.39 | 1.25 |
| W1 | 500 | none | 6–7 | 58,685 | 1.1·10⁹ | 9.45 | 1.00 |
| W1 | 1000 | 1 | 8 | 291,855 | 3.2·10⁹ | 2.39 | 1.22 |
| W1 | 1000 | none | 7–8 | 236,282 | 4.5·10⁹ | 9.43 | 1.00 |
| W3 (10⁶) | 500 | 1 | 7 | 795,927 | 1.4·10⁷ | 1.14 | 10.42 |
| W3 (10⁶) | 500 | none | 5 | 32,768 | 7.7·10⁸ | 5.09 | 1.00 |
| W3 (4.0·10⁶) | 100 | any ≥ 0.5 | 5–6 | 103,027 | 7.4·10⁹ | 14.55 | 1.00 |

- **The rule as written.** The leaf cost (w · P2P pairs + P2M + L2P + M2P + P2L) never
  grows as c_leaf shrinks, on any W1 or W3 row, pair weight (2, 4, 8) or leaf size (64,
  128). At n = 128 the largest c_leaf within 10 % of the best is 0.25 on every row but the
  under-resolved W3 at kD = 500 (0.5), so the rule gives 0.25. At n = 64 the resolved W3 at
  kD = 50 and 100 give 2, because the width rule does not bind there. The proposal expected
  0.5 to 1, presumably from a minimum in the leaf cost that does not exist with the formula
  p: P2P falls with the leaf, and P2M and L2P do not grow, because p at a leaf is the
  formula's floor (22–24 at 10⁻⁶) for every c_leaf ≤ 1. What the rule leaves out is the far
  field: each halving of c_leaf adds a level of boxes with M2L at that floor, and the total
  is 3–15× the best at 0.25 for kD ≥ 20 (90× on W1 at kD = 10, where c_leaf ≥ 1 leaves
  two levels and no far field at all).
- **The revised rule.** The leaf rule's purpose (§3.7) is that no leaf is high frequency, and
  the total cost is best for the largest c_leaf. Hence the largest c_leaf with no H leaf
  level on any workload at 10⁻⁶ and 10⁻⁸ (§4): **c_leaf = 1**. Since the width rule acts in
  octaves, 0.75 to 1 give the same trees here; 1 is the round value. The P2P phase k r stays
  below about 22 radians (§5.7's 22 c_leaf).
- **Its cost.** Within 2 % of the best total on every resolved workload to kD = 200 (W1,
  W3 resolved: c_leaf = 1 changes nothing or one level); 22–25 % above it on W1 at
  kD = 500 and 1000 (11 % at n = 64), because 1.24 λ leaves are halved; 10.4× on the
  under-resolved W3 at kD = 500, the price §3.7 accepts for "more, sparser leaves instead of
  high-frequency ones", which S8 was to measure.
- **Sensitivity.** n = 64 (not run for W1 at kD = 1000 and the under-resolved W3 at
  kD = 500, the two most expensive sets) changes no conclusion. The ranks of c_leaf are the same at
  w = 2, 4 and 8.

## 4. High-frequency leaves

`results/leaves.md` gives, for every tree, the widest occupied leaf and the leaf levels that
are H at each ε. Trees with an H leaf level, out of 43 per c_leaf (n = 128, every
workload):

| c_leaf | 10⁻⁴ | 10⁻⁶ | 10⁻⁸ |
| --- | --- | --- | --- |
| 0.25 | 0 | 0 | 0 |
| 0.5 | 1 | 0 | 0 |
| 0.75 | 2 | 0 | 0 |
| 1 | 14 | 0 | 0 |
| 1.5 | 17 | 1 | 0 |
| 2 | 17 | 9 | 0 |
| none | 17 | 9 | 1 |

At 10⁻⁶ the H leaves are 2 λ wide (W1 and W2 at kD = 50–200 and the Plummer sphere at
kD = 100, at c_leaf = 2 or without a limit), 1.24 λ (W3 at kD = 500, c_leaf = 1.5 and 2)
or 2.49 λ (W3 at kD = 500 without a limit, also H at 10⁻⁸). At 10⁻⁴ the first-pass bound
passes the far classes on boxes of 0.5–1 λ: at p from the 10⁻⁴ formula (16–18 there) max (2j + 1)|h_j(k|t|)| at
|t| = 3√3 w is small enough, and with full V lists the conversions cost less than RCR on
those pairs. Whether that survives S6's constants (C_T, θ) and S7's calibrated p is CH0.8b's;
the leaf operators, M2P and P2L do not depend on the regime, so an H leaf level at 10⁻⁴
changes the M2L of that level, not the leaf path.

## 5. The numbers that replace §7

At c_leaf = 1, n = 128, formula p. Every (workload, kD, ε) is in `results/summary.md`; the
per-level tables at 10⁻⁶ in `results/levels.md` (W1, W2, W3 at every kD; W4 at kD = 100), and
W1 at kD = 500 and 1000 at every ε in `results/w1-levels.md`. The headline rows (10⁹
complex multiply-adds; P2P in 10⁹ point pairs):

| Workload | kD | ε | N | H levels | P2P | M2M+L2L | M2L (L) | M2L (H) | multipoles GiB | max signatures GiB | top per rank MiB | top per node GiB (40; 72) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | 500 | 10⁻⁴ | 2.0 M | 2–6 | 0.80 | 6.9 | 32 | 29 | 0.57 | 1.75 | 76 | 3.0; 5.3 |
| W1 | 500 | 10⁻⁶ | 2.0 M | 2–5 | 0.80 | 12.0 | 122 | 38 | 1.00 | 1.33 | 93 | 3.6; 6.5 |
| W1 | 500 | 10⁻⁸ | 2.0 M | 2–4 | 0.80 | 19.4 | 269 | 35 | 1.51 | 0.86 | 108 | 4.2; 7.6 |
| W1 | 1000 | 10⁻⁴ | 8.0 M | 2–7 | 3.2 | 38 | 130 | 155 | 2.34 | 7.00 | 233 | 9.1; 16.4 |
| W1 | 1000 | 10⁻⁶ | 8.0 M | 2–6 | 3.2 | 60 | 488 | 200 | 4.08 | 5.35 | 268 | 10.5; 18.8 |
| W1 | 1000 | 10⁻⁸ | 8.0 M | 2–5 | 3.2 | 91 | 1,073 | 204 | 6.13 | 3.53 | 299 | 11.7; 21.0 |
| W3 | 100 | 10⁻⁶ | 4.0 M | 2–3 | 7.4 | 6.6 | 273 | 2.4 | 0.96 | 0.12 | 9 | 0.4; 0.7 |
| W3 (10⁶) | 200 | 10⁻⁶ | 1.0 M | 2–4 | 0.77 | 4.4 | 169 | 25 | 0.39 | 0.95 | 20 | 0.8; 1.4 |
| W3 (10⁶) | 500 | 10⁻⁶ | 1.0 M | 2–6 | 0.01 | 100 | 1,526 | 1,652 | 10.2 | 42.1 | 74 | 2.9; 5.2 |

"Top per rank" is the multipoles of levels 2 to the deepest level within Phase 5S's
4,096-box cap (levels 2–4 for the W1 rows, 2–3 for the W3 rows; §6.6 counted levels
2–4); W4 and the small kD in
`summary.md`. The resolved W3 at kD = 100 also does 123·10⁹ of M2P and P2L (its leaves sit
on two levels). The largest table set of one level is the 47 rotation block sets of an
RCR level at p ≥ 144: 1.78 GiB at kD = 500, 10 GiB at kD = 1000 and 10⁻⁸ (p = 271),
which §7 already marked as generated per level call.

What follows:

- **Low-frequency M2L dominates the far field** below kD ≈ 10³ on surfaces (76 % of the M2L
  at kD = 500, 71 % at 1000, 10⁻⁶) and more at tighter ε, as §7 found: H6 before H4 (D6).
- **The node budget binds first on under-resolved volumes**: 42 GiB of signature scratch on
  one level of W3 at kD = 500. CH4.1's chunked target accumulators and the RCR fallback of
  §3.6 are needed there, not on W1.
- **The replicated top grows with the H levels, not with N**: 268 MiB per rank on W1 at
  kD = 1000 (10.5 GiB per Kathleen node at 40 ranks, 3.6 GiB at kD = 500). Whether that
  calls for CH8.1's byte cap depends on the node budget beside the tables, which this
  census does not fix.

## 6. Batch sizes for S5 and S9

`results/batches.md` lists every H level. At 10⁻⁶:

| Workload | kD | Level | p | rings 2P − 1 | N_φ host | N_φ pow2 | boxes converted (each way) | transforms per level call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W1 | 500 | 2 | 144 | 289 | 588 | 1024 | 56 | 16,184 |
| W1 | 500 | 5 | 33 | 67 | 140 | 256 | 4,760 | 318,920 |
| W1 | 1000 | 2 | 261 | 523 | 1080 | 2048 | 56 | 29,288 |
| W1 | 1000 | 6 | 33 | 67 | 140 | 256 | 19,132 | 1,281,844 |
| W3 (10⁶) | 500 | 6 | 25 | 51 | 108 | 128 | 256,359 | 13,074,309 |

Over the sweep host N_φ takes 72 to 1,120 and the power of two 128 to 2,048; S5's range
(256 to 2048) covers the top levels, and the lowest H levels (p 25–33) need N_φ = 108–140
on the host, 128–256 as a power of two. A level converts the same boxes in both directions
here (every converted box is both source and target). S9's per-box plans are one plan per
(level, N_φ) with 2P − 1 = 35 to 543 transforms each over every ε (51 to 523 at 10⁻⁶).

## 7. The workloads (D12)

All four as §9.1 states them build on one node, and the census suggests no change of their
definitions:

- **W1** (to kD = 1000, 8.0 million points): 13 s to build and count at c_leaf = 1.
- **W2** (to kD = 500, 3.2 million): the same H levels as W1; its cap adds a level of
  leaves and W and X lists (1.7·10⁹ M2P and P2L at kD = 500).
- **W3** resolved to kD = 100 (4.0 million, all levels L except 2–3), under-resolved at 10⁶
  for kD = 200 and 500; the latter is the node-budget case (§5) and the 10× cost of the leaf
  rule (§3), so it should be named as such in the gates.
- **W4** (N = 10⁵ and 10⁶, kD = 1 to 100): H levels only at kD = 100.

## 8. Limits of this first pass

- p is the formula's, not S7's calibration; the bound's constants are C_T = 2, θ = 1, not
  S6's. Calibration lowers p most on the low-frequency levels, which carry most of the work,
  so the absolute counts fall in CH0.8b. The c_leaf comparison may shift with them.
- The operation counts are §7's conventions (§1.4), not measured costs; a P2P pair is
  weighted as 4 complex multiply-adds (2 and 8 checked).
- The regime rule here uses the operation count only. The node budget (§3.6's third
  condition) is reported, not applied, and neither is the per-class speed test (§1.4).
- P2M and L2P are counted on every leaf, also where a tree of one or two levels has no far
  field (W1 at kD = 10 and 20 with c_leaf ≥ 1); the effect on those small trees is
  negligible in absolute terms.
- The counts skip empty boxes. The Laplace plan holds them in its rows; whether the
  Helmholtz drivers skip them is CH2/CH3's.
- Build times are of the spike's one-thread census (tree, plan and counting), not of a tuned
  builder. The largest tree, W1 at kD = 1000 and c_leaf = 0.25, took 205 s and 24 GB.

## 9. Questions for sign-off

Signed off by the user on 2026-10-28: "1–4 as recommended". D11 and D12 are recorded in
`docs/helmholtz/roadmap.md`'s decision table.

1. **D11: c_leaf = 1**, by the revised rule (the largest c_leaf with no high-frequency leaf
   on any workload at 10⁻⁶ and 10⁻⁸), replacing the rule of §12 ("the largest c_leaf within
   10 % of the best P2P and leaf cost"), which is degenerate (§3). Alternatives: the rule as
   written (c_leaf = 0.25, 3–15× the total work at kD ≥ 20); 1.5 (best total on W1 at kD = 500 and
   1000, but an H leaf level on the under-resolved W3 at 10⁻⁶); 0.5 (no H leaf at 10⁻⁴ except
   one tree, 2.4–3.2× the total work of c_leaf = 1 on W1 at kD ≥ 50).
2. **D12: the workloads of §9.1 unchanged**, with W3 at 10⁶ and kD = 500 named as the
   node-budget and leaf-rule-cost case.
3. **H leaf levels at 10⁻⁴**: accept that the first-pass rule makes levels of 0.5–1 λ high
   frequency at the loosest tolerance (the leaf operators are unaffected), and leave the
   threshold to CH0.8b with S6's constants and S7's p; or require no H leaf at every ε,
   which forces c_leaf = 0.25.
4. **Proposal corrections for the next draft**: §12 D11's rule as in question 1; §7's
   numbers by `results/summary.md` once CH0.8b confirms them; §7's "full cube at kD = 500"
   signature scratch (8.7 GiB on level 5) is 9.2 GiB on level 5 and 42 GiB on level 6 of
   the under-resolved W3 with the leaf rule.

## Files

- `src/workloads.rs`, `src/tree.rs`, `src/census.rs`, `src/model.rs`: the spike.
- `tests/census.rs`: the eight tests of the plan (in six test functions; `trees` owns MPI).
- `examples/census.rs`: the sweep (`-- sweep`, 22 minutes, 40 GB peak) and single trees
  (`-- probe W1 1000 1 128`).
- `results/c-leaf.md`, `leaves.md`, `summary.md`, `levels.md`, `w1-levels.md`,
  `batches.md`: the sweep's output, committed as generated (one cosmetic fix by hand: a
  "-0" from an empty f64 sum in three W1 rows of `summary.md`, since fixed in the code).
