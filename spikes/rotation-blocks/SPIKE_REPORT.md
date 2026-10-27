# Rotation blocks of large degree: spike report (Helmholtz CH0.1, S1)

Helmholtz task CH0.1 ([docs/helmholtz/tasks/CH0.1.md](../../docs/helmholtz/tasks/CH0.1.md));
the question is proposal §10 CH0.1, with §3.2, §5.2, §7 and risk R1. Written on
2026-10-27 from branch `tbetcke/h-ch0-1` (main at 9625ac2 plus this spike), on the M3
Max, release builds, rustc 1.99.0, f64, one thread.

## Summary

1. **The Gumerov–Duraiswami recursion gives accurate, orthogonal blocks to degree 1000**
   (§4). In the orthonormal real basis (§1), every sampled entry of the y-rotation blocks
   agrees with an arbitrary-precision reference to at most **5.0·10⁻¹⁵** at n = 1000, and
   to at most 3.4·10⁻¹⁵ at the octree's polar angles. ‖O Oᵀ − I‖_max stays at or below
   **4.4·10⁻¹⁴** at every degree checked to 1000, for six angles from 10⁻³ to
   arccos(−1/√3). The error grows slowly, from about 10⁻¹⁵ at n ≤ 100 to
   3–5·10⁻¹⁵ at n = 1000.
2. **Two details decide stability** (§2.3). The recursion of GD's step 3 must run only in
   the wedge k ≥ |m|, with the rest filled by symmetry; marched over all columns it
   reached 4·10¹³³ at n = 500. The Legendre functions of row 0 must be computed in a
   telescoped form in u = 1 − |cos β|. The textbook recursion in cos β cost
   2.75·10⁻¹² at n = 1000 and β = 10⁻³: Pₙ moves by about n²/2 per unit of cos β, and
   its coefficients nearly cancel near cos β = 1.
3. **Against N-conjugated `nd_fmm_math::rotation::blocks` at p ≤ 30** (§4.3), the full
   z-y-z blocks agree to **1.8·10⁻¹⁴**. This pins the basis, its phase and the
   orientation. The definition Yₙ(Qx) = Oⁿ Yₙ(x) holds to 10⁻¹³ relative at n ≤ 20.
4. **The conjugated blocks stay accurate only to a low degree** (§5). Measured against
   the GD blocks, 10⁻¹⁴ is lost from n = 28–43, 10⁻¹² from n = 52–78 and 10⁻⁸ from
   n = 94–140, depending on the angle. The blocks become non-finite at n = 515, as §3.2
   predicted. At n = 100 the error is 3·10⁻⁸ for M2M's polar angle.
5. **β and π − β are related by signs** (§6): Oⁿ(π − β) = (−1)ⁿ E Oⁿ(β) E F, with
   E = diag((−1)^|m|) and F = +1 on the cos slots and −1 on the sin slots. In the real
   basis no permutation remains. Computed independently, the two agree to 7·10⁻¹⁵, but
   not bit for bit. A table that derives π − β from β by these signs is exact.
6. **Time per block set** (§7): 1.8 ns per stored value from p = 20 upwards. That is
   0.75 ms at p = 82, 3.8 ms at p = 144, 22 ms at p = 261 and 1.2 s at p = 1000, for one
   polar angle. This is about 4.5× faster than `rotation::blocks` plus the conjugation at
   20 ≤ p ≤ 82, which compute twice the values for a general rotation. The compact
   y-rotation set holds half the values of `blocks_len(p)`: 15.5 MiB at p = 144 and
   91 MiB at p = 261.
7. **Recommendation for CH1.2** (§8): put the GD recursion in `nd-fmm-math` as the
   generator of the Helmholtz blocks at every degree, not only above a threshold. Keep
   the conjugated `rotation::blocks` as the test reference at p ≤ 30. Then the degree
   above which §5.2 and §7 generate blocks per level call instead of storing them is a
   memory question, not an accuracy one. Questions 1 and 2 of §8 are for sign-off.

## 1. The basis and the blocks

The basis is the real storage (CONVENTIONS §3.6) of N Rₙ. Here Rₙᵐ are the Racah
regular harmonics of §3.3, without the Condon–Shortley phase, and
N = diag(√((n + |m|)!(n − |m|)!) cₘ) with c₀ = 1 and cₘ = √2 otherwise (§3.8). So slot +m
holds √2 νₘ Re Rₙᵐ, slot −m holds √2 νₘ Im Rₙᵐ and slot 0 holds ν₀ Rₙ⁰, with
νₘ = √((n + |m|)!(n − |m|)!). These are the real orthonormal harmonics up to a factor
per degree, which no rotation block sees. Proposal §3.2 implies this basis, but §3.15
(CH1.4, CH1.1) fixes it. The block Oⁿ(Q) satisfies Yₙ(Qx) = Oⁿ(Q) Yₙ(x) and equals
N Dⁿ(Q) N⁻¹.

A y-rotation does not mix the cos slots (m ≥ 0) with the sin slots (m < 0). Its block
therefore splits into a cos half of (n + 1)² values and a sin half of n². The spike
stores the two halves per degree (`gd::y_blocks`, the "compact" layout):
(p + 1)(2p² + 4p + 3)/3 values for degrees 0 to p, against
`blocks_len(p)` = (p + 1)(2p + 1)(2p + 3)/3 for full blocks. A z-rotation is a 2 × 2
rotation per order, so a general block is Oⁿ(R_z(α)) Oⁿ(R_y(β)) Oⁿ(R_z(γ))
(`gd::euler_blocks`), the z-y-z factorisation of §3.8.

## 2. Method

### 2.1 The complex blocks

In the complex orthonormal basis Ŷₙᵐ = νₘ Rₙᵐ, the rotation R_y(β) has real entries oₘₖ.
In Wigner's standard phase they are oₘₖ = dⁿₖₘ(β) = (−1)^{m+k} dⁿₘₖ(β), which the
reference checks against the definition (§3). The spike computes:

1. **Row 0** from the addition theorem: o₀ₖ = (−1)ᵏ P̄ₙᵏ(cos β) for k ≥ 0 and
   P̄ₙ^|k|(cos β) for k < 0, with P̄ₙᵐ = √((n − m)!/(n + m)!) Pₙᵐ.
2. **Row 1 of degree n from row 0 of degree n + 1** (GD's step 2). The ladder
   D₊Rₙ₊₁⁰ = −Rₙ¹ of §3.4, applied in the rotated frame and expressed in the frame of x,
   gives 𝒟ⁿ₁ₗ = ½(1 − cos β) 𝒟ⁿ⁺¹₀,ₗ₊₁ + ½(1 + cos β) 𝒟ⁿ⁺¹₀,ₗ₋₁ − sin β 𝒟ⁿ⁺¹₀ₗ in the
   Racah basis, with no division by sin β. The smaller of ½(1 ± cos β) is formed as
   sin²β / (2(1 ± cos β)).
3. **The other rows** (GD's steps 3 and 4) come from the rotation commuting with J_y. With
   cⱼ = √((n − j)(n + j + 1)), c_{m′} o_{m′+1,m} = c_{m′−1} o_{m′−1,m} − cₘ o_{m′,m+1} +
   c_{m−1} o_{m′,m−1}. This is solved upwards for rows 2 to n and downwards for rows −1
   to −n, each only over the wedge m ≥ |row|. The relation holds in any basis whose J±
   elements are ±cⱼ, so the phase of the basis does not enter.
4. **The rest of the block by symmetry**: o₋ₘ,₋ₖ = (−1)^{m+k} oₘₖ and
   oₖₘ = (−1)^{m+k} oₘₖ.

The real block follows from O[+m,+k] = oₘₖ + (−1)ᵏ oₘ,₋ₖ and
O[−m,−k] = oₘₖ − (−1)ᵏ oₘ,₋ₖ for m, k > 0, O[+m,0] = √2 oₘ₀ and
O[0,+k] = (o₀ₖ + (−1)ᵏ o₀,₋ₖ)/√2 (`src/gd.rs`).

### 2.2 The Legendre functions

Column m runs the recursion in n from P̄ₘᵐ = Πᵢ √((2i − 1)/(2i)) sᵐ (`src/legendre.rs`).
Two measures make it accurate to degree 1001 at any β:

- **No underflow of the start values.** sᵐ leaves the f64 range near m ≈ 100 at
  β = 10⁻³, while P̄ₙᵐ at n ≫ m can be far larger. Each column therefore carries a binary
  exponent of its own, an X-number in Fukushima's sense, and is scaled back when stored.
- **No n² ε loss near cos β = ±1.** The recursion runs on |x| with u = 1 − |x| =
  s²/(1 + |x|), in the telescoped form Eₙ = cₙ(P̄ₙ − P̄ₙ₋₁),
  Eₙ = Eₙ₋₁ + (rₙ − (2n − 1)u) P̄ₙ₋₁, with rₙ = m²/(n + cₙ) + m²/(n − 1 + cₙ₋₁) =
  2n − 1 − cₙ − cₙ₋₁ formed without cancellation. x < 0 follows from
  P̄ₙᵐ(−x) = (−1)ⁿ⁺ᵐ P̄ₙᵐ(x).

### 2.3 What the first versions got wrong

| version | change | worst error at n = 1000 (reference) | ‖O Oᵀ − I‖ at β = 10⁻³ |
| --- | --- | ---: | ---: |
| 1 | step 3 over all columns k = −n..n | 4·10¹³³ (group property at n = 500) | – |
| 2 | step 3 in the wedge only | 3.05·10⁻¹² (β = 10⁻³), 2·10⁻¹⁴ (octree angles) | 6.4·10⁻¹² |
| 3 | Legendre in u = 1 − x | 3.8·10⁻¹³ (β = 10⁻³) | 1.3·10⁻¹² |
| 4 | Legendre telescoped (final) | 5.0·10⁻¹⁵ (β = 10⁻³), 3.4·10⁻¹⁵ (octree) | 8.9·10⁻¹⁵ |

The cancellation-free ½(1 − cos β) of step 2 made no measurable difference. In versions
2 and 3 the error sat on the diagonal entries near 1, carried there from P̄ₙ⁰ of row 0.
This affects any implementation in CH1.2: the stability of GD's recursion holds only if
its input, row 0, is itself accurate to round-off at large n.

## 3. The reference

`tools/reference.py` (PEP 723, `mpmath==1.4.1` pinned as in `tools/fixtures/`) evaluates
Wigner's explicit sum for dⁿₘ′ₘ(β) with 0.61 n + 40 digits. That covers the cancellation
of its terms, whose absolute sum is at most 4ⁿ. Every value is recomputed with 50 more
digits and must agree to 10⁻²⁵. The script maps the values to the real block by the
formulas of §2.1. It first checks the map against the definition Yₙ(Qx) = O Yₙ(x) at
n ≤ 6, with Rₙᵐ from the Cartesian recursion of CONVENTIONS §3.5 in mpmath (worst
4·10⁻⁶⁰ at 60 digits). The angles are evaluated at exactly the f64 value the spike uses.

`fixtures/reference.json` (390 kB) holds 12,914 entries in 54 sets: degrees 10, 30, 50,
100, 200, 300, 500, 700 and 1000, each at six angles:

- arccos(1/√3), M2M's child shift;
- π − arccos(1/√3);
- atan(1/3), the smallest off-axis polar angle of the V list;
- π/2;
- 1.234, a generic angle;
- 10⁻³, a small one.

Per set, the script samples rows and columns 0 and ±n, the diagonal and antidiagonal of
each half (at most 41 points per line) and 60 random entries. The values are rounded to
f64. `uv run spikes/rotation-blocks/tools/reference.py --check` regenerates the file
byte for byte in about 50 s, which was checked.

## 4. Accuracy of the GD blocks

All in `tests/accuracy.rs`, run with
`cargo test -p nd-fmm-spike-rotation-blocks --release -- --include-ignored`. All six tests pass.

### 4.1 Against the reference (test 1, bound 10⁻¹²)

Maximum absolute error over the sampled entries:

| n | arccos(1/√3) | π − arccos(1/√3) | atan(1/3) | π/2 | 1.234 | 10⁻³ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 7.5e-16 | 3.9e-16 | 2.2e-16 | 3.8e-16 | 9.6e-16 | 3.3e-16 |
| 30 | 6.4e-16 | 9.7e-16 | 3.3e-16 | 8.6e-16 | 6.7e-16 | 2.2e-16 |
| 50 | 8.3e-16 | 8.7e-16 | 8.3e-16 | 9.7e-16 | 7.6e-16 | 8.9e-16 |
| 100 | 1.2e-15 | 1.0e-15 | 6.0e-16 | 1.2e-15 | 1.2e-15 | 1.3e-15 |
| 200 | 1.9e-15 | 1.4e-15 | 8.9e-16 | 1.5e-15 | 1.6e-15 | 7.8e-16 |
| 300 | 2.0e-15 | 1.9e-15 | 1.3e-15 | 1.3e-15 | 1.3e-15 | 1.7e-15 |
| 500 | 2.7e-15 | 2.4e-15 | 1.2e-15 | 2.6e-15 | 1.7e-15 | 2.4e-15 |
| 700 | 2.7e-15 | 3.0e-15 | 1.4e-15 | 2.6e-15 | 1.7e-15 | 1.1e-15 |
| 1000 | 3.3e-15 | 3.4e-15 | 1.4e-15 | 3.1e-15 | 2.2e-15 | 5.0e-15 |

### 4.2 Orthogonality (test 2, bound 10⁻¹²)

The test computes ‖O Oᵀ − I‖_max at every degree n ≤ 100 and at 150, 200, 300, 500, 700
and 1000. It also computes the probe ‖Oᵀ(O v) − v‖_∞ for a random v at every degree to
1000. The worst over those degrees:

| β | ‖O Oᵀ − I‖_max | probe |
| --- | ---: | ---: |
| arccos(1/√3) | 4.3e-14 | 2.8e-14 |
| π − arccos(1/√3) | 4.4e-14 | 3.0e-14 |
| atan(1/3) | 6.7e-15 | 7.3e-15 |
| π/2 | 7.1e-15 | 2.0e-14 |
| 1.234 | 1.8e-14 | 2.1e-14 |
| 10⁻³ | 8.9e-15 | 7.3e-15 |

### 4.3 Low degree and the definition (tests 3, 4 and 6)

- The full z-y-z blocks, p ≤ 30, against N Dⁿ N⁻¹ from `rotation::blocks`: worst
  1.8·10⁻¹⁴ (bound 10⁻¹³). The triples tested are the six y-rotations above and
  (0.3, 1.1, −0.4), (−1.2, 0.8, 2.5), (2.9, 2.7, 0.6), (0, 0, 1) and (1, π, −2).
- Yₙ(Qx) = Oⁿ(Q) Yₙ(x), with Yₙ = N Rₙ from `harmonics::regular`, holds at three points
  and three rotations for n ≤ 20, relative to the largest |Yₙ(Qx)| of the degree, 10⁻¹³.
- The group property Oⁿ(R_y(0.7)) Oⁿ(R_y(1.1)) = Oⁿ(R_y(1.8)) at n = 500: 1.2·10⁻¹⁴.

## 5. How far the conjugated blocks stay accurate

`examples/sweep.rs` compares N Dⁿ N⁻¹ from `rotation::blocks` with the GD blocks of §4,
which are accurate to a few 10⁻¹⁵, degree by degree to 600. The table gives the degree
from which the maximum entry error stays above each threshold:

| rotation | 10⁻¹⁴ | 10⁻¹² | 10⁻¹⁰ | 10⁻⁸ | 10⁻⁶ | 10⁻² | non-finite | error at n = 20, 30, 40, 60, 80, 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| y, β = arccos(1/√3) (M2M) | 28 | 52 | 72 | 94 | 116 | 158 | 515 | 2.6e-15, 1.8e-14, 1.1e-13, 7.3e-12, 5.3e-10, 3.0e-8 |
| y, β = atan(1/3) (V list) | 43 | 78 | 108 | 140 | 175 | 236 | 515 | 7.8e-16, 1.9e-15, 7.1e-15, 1.2e-13, 1.4e-12, 3.4e-11 |
| y, β = π/2 | 37 | 53 | 73 | 96 | 118 | 160 | 515 | 7.8e-16, 2.6e-15, 4.0e-14, 5.0e-12, 4.0e-10, 2.9e-8 |
| z-y-z (0.3, 1.1, −0.4) | 30 | 52 | 73 | 95 | 116 | 160 | 515 | 2.0e-15, 1.1e-14, 8.5e-14, 6.0e-12, 4.7e-10, 2.8e-8 |
| z-y-z (2.9, 2.7, 0.6) | 37 | 61 | 90 | 116 | 140 | 193 | 515 | 1.8e-15, 3.7e-15, 2.1e-14, 8.3e-13, 1.5e-11, 6.6e-10 |

The error grows by about a factor of ten every eight to ten degrees, which is the 4ⁿ span
of the Racah entries (proposal §3.2) showing up through the rounding of the recursion.
At the 10⁻¹⁴ that §3.9 asks of f64 (p ≤ 30), conjugation is good to n ≈ 28. That is
just enough for the p ≤ 30 check of the acceptance, and not enough for any
high-frequency level.

## 6. The blocks of β and π − β

From dⁿₘ′ₘ(π − β) = (−1)ⁿ⁺ᵐ′ dⁿₘ′,₋ₘ(β), in the complex basis
oₘₖ(π − β) = (−1)ⁿ⁺ᵐ oₘ,₋ₖ(β): a signed permutation k ↦ −k of the columns. In the real
basis that permutation is absorbed by the cos and sin combinations, and only signs
remain:

- on the cos half, O[+m,+k](π − β) = (−1)ⁿ⁺ᵐ⁺ᵏ O[+m,+k](β);
- on the sin half, O[−m,−k](π − β) = −(−1)ⁿ⁺ᵐ⁺ᵏ O[−m,−k](β);

that is, Oⁿ(π − β) = (−1)ⁿ E Oⁿ(β) E F with E = diag((−1)^|m|) and F = diag(+1 on the
cos slots, −1 on the sin slots). Test 5 checks this at every degree to 1000 for the six
angles, with (cos, sin) = (−cos β, sin β) as the second input. The worst difference is
6.7·10⁻¹⁵, but 22–97 % of the 6.7·10⁸ entries differ in their last bits. The recursion is not
bit-symmetric under x ↦ −x, because steps 2 and 3 add their terms in a different order.
A table therefore stores the blocks of β ≤ π/2 and applies the signs. The M2M of a level
then needs one block set for both polar angles arccos(±1/√3), as §5.2 assumed, and the
two are exactly consistent.

## 7. Time per block set

`examples/time.rs`, median of seven rounds of at least 50 ms each, one thread, the M3 Max
(load average 2.5 at the time, 16 cores), f64. The GD column is the compact y-rotation
set of degrees 0 to p for β = arccos(1/√3). The Racah columns are `rotation::blocks` for
the same rotation (full blocks of a general rotation), alone and with the conjugation of
§5, for p ≤ 82 only, where they are still accurate:

| p | GD set (ms) | ns per stored value | compact set (MiB) | Racah `blocks` (ms) | `blocks` + N (ms) | full set (MiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.005 | 10.95 | 0.00 | 0.006 | 0.006 | 0.01 |
| 20 | 0.014 | 2.20 | 0.05 | 0.052 | 0.060 | 0.09 |
| 33 | 0.055 | 2.11 | 0.20 | 0.211 | 0.244 | 0.40 |
| 50 | 0.180 | 2.03 | 0.67 | 0.698 | 0.802 | 1.35 |
| 82 | 0.747 | 1.96 | 2.91 | 2.974 | 3.389 | 5.82 |
| 144 | 3.767 | 1.85 | 15.51 | – | – | 31.01 |
| 261 | 21.540 | 1.80 | 91.48 | – | – | 182.95 |
| 500 | 147.6 | 1.76 | 639.61 | – | – | 1279.21 |
| 1000 | 1225.0 | 1.83 | 5101.54 | – | – | 10203.07 |

The cost is O(p³) at about 1.8 ns per stored value; at p = 8 the Legendre table and the
loop overheads dominate. The GD code is a straightforward scalar loop, not optimised.

For scale: applying a compact set to one complex expansion takes about two
multiply-adds per stored value and per plane, so 8 per value. Generating a set
therefore costs about as much as applying it to a few boxes. At p = 144 that is
3.8 ms to generate against roughly 2.0·10⁶ × 8 ≈ 16 Mflop per box. Generating the
blocks of a level per level call is cheap next to the M2M and L2L of that level once
it has more than a handful of boxes. That supports §5.2's plan to generate rather than
store above some degree.

## 8. Consequences and questions for sign-off

For CH1.2 (`nd-fmm-math`), §3.15 item 5 (CH1.1) and §7:

1. **Question: generate the Helmholtz blocks by GD at every degree?** The proposal
   (§3.2) keeps `rotation::blocks` conjugated by N as the generator "up to the degree
   where the conjugation stays accurate". S1 finds that degree is about 28 at 10⁻¹⁴.
   GD is accurate at every degree tested, and about 4.5× faster at 20 ≤ p ≤ 82 for the
   y-rotations the tables store. Recommendation: GD at every degree for the Helmholtz
   tables, with the conjugated Racah blocks only as the test reference at p ≤ 30.
   Laplace keeps its own blocks unchanged (constraint 1).
2. **Question: the threshold of §5.2 and §7 is memory, not accuracy.** "Above a degree
   measured in S1 the blocks are generated per level call ... instead of stored." S1
   finds no accuracy limit to p = 1000. Generation costs about 1.8 ns per stored value,
   so the threshold should come from the node budget of constraint 7, set in CH0.8 or
   CH2.2, not from S1. The compact y-rotation layout halves the memory of §7's figures:
   15.5 MiB rather than 31 MiB at p = 144, 91 MiB rather than 183 MiB at p = 261, per
   polar angle.
3. **For §3.15 item 5**: the relation of β and π − β is the sign rule of §6, with no
   permutation in the real basis, and tables store β ≤ π/2. The basis of §1 (N Rₙ in
   the real storage of §3.6, no Condon–Shortley phase) is the one the spike verified.
   CH1.4 and CH1.1 fix it.
4. **For CH1.2's implementation**: the wedge-only recursion and the telescoped,
   exponent-carrying Legendre functions of §2.2 are needed, not optional (§2.3). The
   spike is f64 only and not generic over `RealScalar`. f32 blocks (D3) are out of
   scope here; a table would be generated in f64 and rounded (constraint 5).

## 9. Files

- `src/legendre.rs`: P̄ₙᵐ to any degree (§2.2).
- `src/gd.rs`: the GD y-rotation blocks, the compact layout, z-rotations and full blocks.
- `src/conjugated.rs`: N Dⁿ N⁻¹ from `nd_fmm_math::rotation::blocks`.
- `tests/accuracy.rs`: tests 1–6 of the plan.
- `examples/sweep.rs`: §5. `examples/time.rs`: §7.
- `tools/reference.py`, `fixtures/reference.json`: §3.
