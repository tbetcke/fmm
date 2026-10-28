# Radial functions: spike report (Helmholtz CH0.3, S3)

Helmholtz task CH0.3 ([docs/helmholtz/tasks/CH0.3.md](../../docs/helmholtz/tasks/CH0.3.md)).
The question is proposal §10 CH0.3, with §3.4, §4 item 3, §5.1, §5.5, §5.6 and risk R4.
Written on 2026-10-27 and 2026-10-28 from branch `tbetcke/h-ch0-3` (main at 0367855 plus this spike), on
the M3 Max, release builds, rustc 1.99.0, f64, one thread.

## Summary

1. **Every family is accurate to near round-off on every operator's range** (§3).
   jₙ, hₙ = hₙ⁽¹⁾ (complex z, 0 ≤ arg z ≤ π/2), iₙ and kₙ (real x) and their
   derivatives, against an arbitrary-precision reference (mpmath's J and I, and the
   exact finite sums of hₙ and kₙ): **6.5·10⁻¹⁴ at worst** over 13,174 entries in 571
   sets, at most 3.6·10⁻¹⁵ on the leaf and low-frequency ranges; the error grows only
   with the largest |z| and orders (hₙ to n = 1481 at |z| = 1501).
   The Wronskians hold to 5.4·10⁻¹⁴ relative to their terms over |z| from 10⁻⁶ to
   4550, seven values of arg z, every order an operator needs there (§4). Every value is
   finite on those ranges, including Yukawa at κρ = 4550, and z = 0 gives the exact
   values. R4 does not materialise in f64 with the algorithms below.
2. **The algorithms** (§2): j and i by Miller's method in ratio form (the backward
   recursion of rₙ = fₙ/fₙ₋₁ from a start order found by FMM3D's forward growth test),
   normalised by the closed form of ĵ₀ or ĵ₁, whichever is larger (a power series for
   |z| < 0.5); h and k by the upward recursion from their closed forms; derivatives from
   the neighbouring orders. No allocation; one pass each.
3. **The exponential factor is needed, always** (§6). Yukawa has no leaf rule
   (proposal §3.7), so a leaf can be many κ⁻¹ wide. Without the factor kₙ at the nearest
   distance underflows f64 from κa ≈ 405 (f32 from 48) and iₙ(κa) overflows f64 at
   κa ≈ 717. Damped complex k has the same growth e^{Im z}. Removing e^{∓|Im z|}
   costs nothing for real k (it is 1) and needs no threshold, so the recommendation is
   to remove it on every level and every kind of k, not only above κr ≈ 1 as §3.4
   proposed.
4. **The power scale: s = min(|k| a, 1), a the circumradius** (§5). In f64 every radius
   works: the scaled values a level carries stay within 10⁻⁴³ to 10³⁹ (the
   half-width) or 10⁻⁵⁴ to 10⁴⁹ (the width), so the choice is a convention. The circumradius is the variable
   p_l is calibrated in (§3.5's k a_l), the radius inside which every source of a box
   lies, and the scale CH0.2's tables already use. FMM3D 2.1.0 uses the **full box
   width**, min(|k| w, 1) (`hfmm3d.f`, `scales(ilev)`), not the half-width proposal §3.4
   attributes to it.
5. **The double factorials cannot be absorbed into the basis** (§5): ĵₙ (2n + 1)!! leaves
   f64 above k a ≈ 100 (10^±300 at k a = 100, 10^±2000 at 650). Not needed in f64, where
   the power scale's range is harmless. **For f32 (D3)** the power scale fails on every
   low-frequency level (k a ≤ 1.7: 10⁻⁵¹ to 10⁴⁷), and a per-degree envelope scale
   σₙ = max_{x ≤ |k| a} |jₙ(x)| keeps every stored value within 10⁻²⁰ to 10⁶ on every
   level. That is a storage conversion an f32 path can apply; the f64 convention need
   not change for it.
6. **kₙ normalised as k₀(x) = e⁻ˣ/x** (§1): hₙ(iκρ) = −i⁻ⁿ kₙ(κρ) and
   jₙ(iκρ) = iⁿ iₙ(κρ), so the Yukawa series is κ Σ iₙ(κρ<) kₙ(κρ>) Σₘ Yₙᵐ Yₙᵐ, with κ in
   place of ik and every factor real. The real path agrees with the complex path at
   z = iκρ bit for bit.
7. **Yukawa's degree grows like √(κ a)** (§6), contrary to the assumption that the
   decaying kernel needs no more terms than Laplace: at the nearest M2L distance, for
   10⁻¹⁰ relative to the series' own size, 27 orders at κa → 0 (Laplace's), 43 at
   κa = 10, 105 at κa = 100. Pruning removes most such levels (§5.5), but CH2.3's p_l and
   θ_D need the relative and the absolute view both.
8. **Cost** (§7): 19–31 ns per value for j by Miller's method (complex z), 5–7 ns for h,
   7–10 ns for i, 2–4 ns for k; 0.8–1.3 µs per particle for P2M's j to order 40.

## 1. Definitions and the scaling

For complex z with Im z ≥ 0 (the branch Im k ≥ 0 of proposal §4 item 1), real x ≥ 0 and
a power scale 0 < s ≤ 1, the spike computes

```text
ĵₙ = e^{−|Im z|} jₙ(z) / sⁿ        ĥₙ = e^{Im z} hₙ(z) sⁿ,   hₙ = hₙ⁽¹⁾ = jₙ + i yₙ
îₙ = e^{−x} iₙ(x) / sⁿ            k̂ₙ = eˣ kₙ(x) sⁿ
```

and the derivatives with the same factors (ĵ′ₙ = e^{−|Im z|} jₙ′(z)/sⁿ and so on: the
factors multiply the derivative, they are not differentiated). iₙ is DLMF's
√(π/(2x)) I_{n+½}(x); kₙ is √(2/(πx)) K_{n+½}(x), which is 2/π times DLMF's 𝗄ₙ, so that

```text
k₀(x) = e^{−x}/x,    hₙ(ix) = −i⁻ⁿ kₙ(x),    jₙ(ix) = iⁿ iₙ(x).
```

With proposal §3.1's expansion and k = iκ (so ik = −κ),

```text
e^{−κ|x−y|} / (4π|x−y|) = κ Σₙ iₙ(κ r<) kₙ(κ r>) Σₘ Yₙᵐ(ŷ) Yₙᵐ(x̂),
```

every factor real. This is the absorption of the powers of i of proposal §3.1, stated
as the radial functions §3.15 item 3 would name.

The factors pair up: a regular function (j, i) is divided by sⁿ and has its growth
e^{|Im z|} removed, a singular one (h, k) is multiplied by sⁿ and has its decay e^{−Im z}
removed, so every product of a regular and a singular value of the same order (P2M
against M2P, the addition theorem's jₙ hₙ) is the unscaled product times
e^{Im z₂ − |Im z₁|}, a factor common to a box pair. Where the operators apply the
factors (per point, per box, per table) is CH1.2's; the spike only fixes the functions.

## 2. Method

**j (and i).** Miller's method in ratio form. The ratios rₙ = jₙ/jₙ₋₁ satisfy
rₙ = z/((2n + 1) − z rₙ₊₁); started from r = 0 at an order N well above the highest
order L wanted and above |z|, the backward recursion converges to the minimal
solution. N is FMM3D's: the forward recursion from (0, 1) at L until it has grown by
10²⁰ (`besseljs3d.f` uses the same test with the squared modulus against 10⁴⁰). Then
ĵ₀ (or ĵ₁, whichever is larger in modulus) from its closed form, and
ĵₙ = ĵₙ₋₁ rₙ/s upward. The ratio form never overflows (FMM3D rescales the linear form
by 10⁻⁴⁰ on the way down instead) and needs no second pass; values below the f64 range
underflow gradually, where they are negligible. ĵ₀ = e^{−|Im z|} sin z / z is formed
from e^{±iz} e^{−|Im z|}, which do not overflow, and from its power series for
|z| < 0.5, where the closed form cancels; ĵ₁'s closed form is used only where
|j₁| > |j₀|, far from its own cancellation. z = 0 gives r = 0, ĵ₀ = 1. iₙ is the same
with rₙ = x/((2n + 1) + x rₙ₊₁), all terms positive, and î₀ = −expm1(−2x)/(2x).

**h (and k).** The upward recursion hₙ₊₁ = (2n + 1)/z hₙ − hₙ₋₁, scaled, from
h₀ = −i e^{iz}/z and h₁ = −e^{iz}(z + i)/z² (with e^{iz} e^{Im z} = e^{i Re z}). It is
the stable direction in the closed upper half plane; for kₙ every term is positive.

**Derivatives** from the neighbouring orders, FMM3D's forms:
jₙ′ = (n jₙ₋₁ − (n + 1) jₙ₊₁)/(2n + 1), hₙ′ = hₙ₋₁ − (n + 1)/z hₙ,
iₙ′ = (n iₙ₋₁ + (n + 1) iₙ₊₁)/(2n + 1), kₙ′ = −kₙ₋₁ − (n + 1)/x kₙ, and j₀′ = −j₁,
h₀′ = −h₁, i₀′ = i₁, k₀′ = −k₁. Each routine computes one order beyond L for them.

## 3. Accuracy against the reference

**The reference** (`tools/reference.py`, mpmath 1.4.1): jₙ and iₙ from mpmath's
`besselj` and `besseli` at half-integer order; hₙ and kₙ from their finite sums of
half-integer order, hₙ(z) = e^{iz} Σₖ₌₀ⁿ i^{k−n−1} aₖ/z^{k+1} and
kₙ(x) = e^{−x} Σₖ₌₀ⁿ aₖ/x^{k+1} with aₖ = (n + k)!/(2ᵏ k! (n − k)!) (DLMF 10.49.6 and
10.49.12), summed with digits added until the largest term is covered (the terms of hₙ
cancel by up to 295 digits, at the largest orders and |z| ≈ 1500). The derivatives from the
neighbouring orders. Nothing is shared with the spike's recursions. Each value is
computed at 40 and 80 digits and must agree to 10⁻³⁰. Before anything is written, the
tool checks the closed forms at n ≤ 2, jₙ(ix) = iⁿ iₙ(x) and hₙ(ix) = −i⁻ⁿ kₙ(x), and the
Wronskians, which tie the sums to `besselj` and `besseli` (worst 1.0·10⁻⁵⁷ against the
bound 10⁻⁴⁰). mpmath's own singular functions were not usable at these orders:
`besselk` does not converge at large argument and moderate order (order 370.5 at 1800,
on the real axis), and `hankel1` = J + iY loses about 0.87 Im z digits to cancellation
and takes up to 20 s per value at the largest arguments.

**The ranges** (plan, "The ranges"): ε = 10⁻¹⁰, p from proposal §3.5's formula with
k a, a the circumradius; c_leaf ≤ 2; kD up to 3000 (k a = 650 at level 2):

| Operator | Family | Arguments | Orders |
| --- | --- | --- | --- |
| P2M, L2P | j, j′ (i, i′) | k ρ, 0 ≤ ρ ≤ a_leaf: Re(k) a_leaf ≤ 5.44 c_leaf ≤ 10.9 | ≤ p_leaf + 1 ≈ 42 |
| M2P, P2L | h, h′ (k, k′) | k ρ, ρ ≥ √3 a_b (box b the smaller) | ≤ p_b + 1 |
| M2M, L2L start values | j (i) | k a/2, k a ≤ 650 | ≤ p_l + p_{l+1} ≤ 1138 |
| M2L start values, T(ŝ) | h (k) | k t, 2w ≤ t ≤ 6a | ≤ 2p ≤ 1482 |
| Projection (k a > 40, CH0.2) | j, j′ at k a; h, h′ at 4 k a/√3; h at k R, R ≤ t + a = 7a | \|z\| ≤ 4550 | ≤ p + 1 |

The leaf rule bounds Re(k) a only, so for damped k |k| a_leaf = 10.9 √(1 + η²) grows
without bound towards the imaginary axis; Yukawa's leaf arguments are not bounded at
all (§6). Each kind of k is sampled: real, k = κ(1 + iη) with η = 0.1, 1 and 10
(arg k = 84°, leaves to |k| a = 110), and imaginary (the complex path at z = iκρ). M2P
and P2L are sampled at ρ = √3 a_b and 8 a_b; larger ratios ρ/a_b (W and X lists across
several levels) only make ĥₙ smaller, and the dense sweeps of §4 take the scale down to
|z|/8. Yukawa is sampled to Helmholtz's orders at the same κ a, which exceed its own
(§6), and at both M2L distances. The fixture (1.3 MB) holds 13,174 rows in 571 sets:
each operator's range at its ends and inside (all orders up to 40, else the ends, the
transition near |z|, 18 spread and 4 random orders; leaves at ρ/a from 10⁻⁶, so
|z| down to 10⁻¹²), and 160 random points with |z| log-uniform from 10⁻⁶ to 4550, arg z
uniform in [0, π/2] and a random order up to the operator's maximum. The largest value
stored is k̂ₙ ≈ 1.6·10²⁹⁴ (Yukawa M2L at κ a = 650, x = 1501, n = 1482), 14 decades
below overflow; Yukawa's own degrees there are far lower.

**The metric.** Relative error for h, h′, i, i′, k, k′, which have no zeros in the
closed upper half plane or for x > 0. jₙ and jₙ′ have real zeros where n < |z|, so
there the error is relative to the oscillation envelope max(|jₙ|, |hₙ|) (|hₙ| is the
modulus √(jₙ² + yₙ²) for real z), relative elsewhere. Reference values below the f64
range (stored as 0) require the spike's value below 10⁻²⁸⁰.

**Result** (test 1, bound 10⁻¹³):

| operator | family | entries | worst value | worst derivative | at |
| --- | --- | --- | --- | --- | --- |
| P2M, L2P | j | 4125 | 1.6e-15 | 1.5e-15 | p2m-complex10-ka0.001-rho1.0 n=36 |
| M2P, P2L | h | 2498 | 3.4e-15 | 3.6e-15 | m2p-complex10-ka1.0-rho8.000 n=36 |
| M2M start | j | 1002 | 3.8e-15 | 3.8e-15 | m2m-complex0.1-ka300.0 n=577 |
| M2L start | h | 1948 | 6.4e-14 | 6.5e-14 | m2l-complex0.1-ka650.0-d2 n=1481 |
| projection | j | 606 | 5.5e-15 | 5.2e-15 | proj-complex0.1-ka650.0-j n=611 |
| projection | h | 1093 | 1.3e-14 | 1.3e-14 | proj-complex0.1-ka650.0-h n=742 |
| Yukawa P2M | i | 825 | 3.5e-15 | 3.7e-15 | yp2m-ka300.0-rho0.5 n=263 |
| Yukawa M2P | k | 271 | 1.3e-15 | 1.3e-15 | ym2p-ka650.0 n=480 |
| Yukawa M2M start | i | 218 | 3.5e-15 | 3.3e-15 | ym2m-ka300.0 n=272 |
| Yukawa M2L start | k | 428 | 1.8e-15 | 1.8e-15 | ym2l-ka1.0-d27 n=69 |
| random | j, h, i, k | 40 each | 4.4e-15, 1.7e-14, 2.2e-15, 1.5e-15 | 4.6e-15, 1.7e-14, 2.3e-15, 1.7e-15 | |

Every one of the 13,174 entries (26,348 values) is within the bound; the worst is
**6.5·10⁻¹⁴**, hₙ at n = 1481, |z| = 1501. The error grows with |z| and n as the
recursions' rounding accumulates, and stays at or below 3.6·10⁻¹⁵ on the leaf and
low-frequency ranges, strongly damped k included. Part of it at large |z| is intrinsic: e^{iz} has condition number
|z|, so a rounded argument z = k ρ alone moves hₙ by up to |z| ulp (3·10⁻¹³ at
|z| = 1500) in any implementation; the fixture holds z exactly, so the test measures
the functions only.

Two defects of the first test runs were the tooling's, not the spike's, and are fixed:
Python's `1j ** k` is exact only to |k| = 100 (CPython then switches to the polar form,
10⁻¹⁶ off), which the cancelling terms of the hₙ sum amplified to O(1) above order 100;
and serde_json's default float parsing can be one ulp off, which at |z| = 1500 moved
e^{iz} by 2·10⁻¹³. The fixture now holds z and s as decimal strings, parsed with
`str::parse`, and computes powers of i exactly.

## 4. Identities, and Yukawa two ways

Without the reference, on a dense grid: |z| at 65 points log-uniform from 10⁻⁶ to
4550, arg z ∈ {0, 0.05, π/8, π/4, 3π/8, π/2 − 10⁻³, π/2}, s = min(|z|, 1), every order an
operator needs at that |z| (`src/ranges.rs`: for the regular functions the M2M start
values, p(2|z|) + p(|z|); for the singular ones the M2L start values, 2p(√3|z|/4)).

| Identity (scaled; the factors cancel) | worst, relative to the terms |
| --- | --- |
| ĵₙ ĥ′ₙ − ĵ′ₙ ĥₙ = i/z² | 5.4·10⁻¹⁴ |
| ĵₙ ĥₙ₊₁/s − s ĵₙ₊₁ ĥₙ = −i/z² | 4.5·10⁻¹⁴ |
| îₙ k̂′ₙ − î′ₙ k̂ₙ = −1/x² | 5.8·10⁻¹⁵ |

The real Yukawa path against the complex path at z = ix (îₙ = i⁻ⁿ ĵₙ(ix),
î′ₙ = i¹⁻ⁿ ĵ′ₙ(ix), k̂ₙ = −iⁿ ĥₙ(ix), k̂′ₙ = −iⁿ⁺¹ ĥ′ₙ(ix)) agrees **bit for bit** on the
same grid: on the imaginary axis the complex recursions perform the real path's
arithmetic with exact zeros. The test checks the relations and their powers of i; the
independent check of i and k is the reference.

A first version of the sweep took the orders of the singular functions from p(|z|)
instead of p(√3|z|/4); at κρ = 348 that asked for orders to 846, and k̂ₙ(348) passes
10³⁰⁸ near n = 790 even with the exponential factor (it grows like e^{n²/(2x)}). No operator needs such
orders: the M2L distance is at least 2.3 times the box's circumradius.

## 5. The power scale

On a level with |k| a = ka and degree p, the expansions carry ĵₙ(k ρ), ρ ≤ a, and
ĥₙ(k ρ) at the nearest M2P or M2L distance ρ ≥ √3 a. `examples/ranges.rs` records the
range of log₁₀ of the values carried, ĵₙ(k a) and ĥₙ(√3 k a) for n ≤ p (real k,
ε = 10⁻¹⁰), under five candidates: the power scale s = min(|k| r, 1) with r the width w
(FMM3D 2.1.0), the half-width w/2 (proposal §3.4) or the circumradius a = √3 w/2 (CH0.2);
the circumradius with the double factorial absorbed (σₙ = sⁿ/(2n + 1)!!); and the
envelope σₙ = max_{0 ≤ x ≤ |k| a} |jₙ(x)| (regular values divided by σₙ, singular ones
multiplied). The M2L start values to order 2p are left out: they are an intermediate of
the table build, whose entries carry s^{n+n′} (CH0.2).

| k a | p | power, w | power, w/2 | power, a | power, a, (2n+1)!! | envelope |
| --- | --- | --- | --- | --- | --- | --- |
| 0.000001 | 35 | -54 to 49 (f32 out) | -43 to 39 (f32 out) | -51 to 47 (f32 out) | -4 to 6 | -4 to 6 |
| 0.001 | 35 | -54 to 46 (f32 out) | -43 to 36 (f32 out) | -51 to 44 (f32 out) | -7 to 3 | -7 to 3 |
| 0.1 | 35 | -54 to 44 (f32 out) | -43 to 34 (f32 out) | -51 to 42 (f32 out) | -9 to 1 | -9 to 1 |
| 0.5 | 35 | -54 to 43 (f32 out) | -43 to 33 (f32 out) | -51 to 41 (f32 out) | -10 to 0 | -10 to 0 |
| 1 | 35 | -51 to 41 (f32 out) | -43 to 33 (f32 out) | -51 to 41 (f32 out) | -10 to -0 | -10 to 0 |
| 1.7 | 35 | -43 to 33 (f32 out) | -43 to 32 (f32 out) | -43 to 33 (f32 out) | -19 to 8 | -11 to 0 |
| 3 | 36 | -36 to 25 | -36 to 25 | -36 to 25 | -28 to 17 | -11 to 0 |
| 10 | 40 | -21 to 9 | -21 to 9 | -21 to 9 | -52 to 40 (f32 out) | -12 to 0 |
| 40 | 77 | -16 to -1 | -16 to -1 | -16 to -1 | -138 to 121 (f32 out) | -17 to 0 |
| 100 | 149 | -16 to -2 | -16 to -2 | -16 to -2 | -309 to 290 (f32 out) | -18 to 0 |
| 300 | 371 | -17 to -2 | -17 to -2 | -17 to -2 | -909 to 890 (f32 out) | -19 to 0 |
| 650 | 741 | -17 to -2 | -17 to -2 | -17 to -2 | -2034 to 2014 (f32 out) | -20 to 0 |

- **f64.** Every power-scale radius works, with 250 decades to spare. The half-width
  has the narrowest range on low-frequency levels (10⁻⁴³ to 10³⁹ against 10⁻⁵¹ to 10⁴⁷
  for the circumradius); above k a ≈ 2 all three coincide (s = 1). The difference is a
  constant factor per degree, so the choice is a convention.
- **The double factorials** keep low-frequency levels within 10^±10 but leave f64 at
  high frequency (k a = 300: 10⁸⁹⁰), because there s = 1 and jₙ(k a) is of order 1/k a,
  not 1/(2n + 1)!!. They cannot be part of a frequency-independent basis, and absorbing
  them only where s < 1 makes the scale jump by up to 10⁵⁰ at |k| a = 1.
- **f32 (D3).** Every power scale leaves f32 on low-frequency levels (k a ≤ 1.7) and is
  within it from k a = 3 on. The envelope is within 10⁻²⁰ to 10⁶ everywhere; the double
  factorial only where s < 1. So f32 on low-frequency levels, if D3 allows it, needs a
  per-degree scale such as the envelope (p + 1 numbers per level, a function of k and
  the widths only, constraint 9). It can be a conversion at the f32 boundary: the f64
  convention does not have to carry it.

FMM3D's scale, read from FMM3D 2.1.0: `hfmm3d.f` lines 415–420 set
`scales(ilev) = boxsize(ilev)*abs(zkfmm)`, capped at 1, with `boxsize` the full width
of a box on the level. `h3dall` (`h3dcommon.f`) returns hₙ(z) sⁿ and `besseljs3d`
(`Common/besseljs3d.f`) jₙ(z)/sⁿ, the forms used here; neither removes an exponential
factor.

## 6. The exponential factor, and Yukawa

`examples/ranges.rs`, log₁₀ of the unscaled values at the box radius and the nearest
distance:

| κ a | p | log₁₀ i₀ | log₁₀ i_p | log₁₀ k₀ | log₁₀ k_p | f64 without the factor |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35 | 0 | -51 | -1 | 41 | in range |
| 10 | 35 | 3 | -16 | -9 | 4 | in range |
| 88 | 35 | 36 | 33 | -68 | -67 | in range |
| 100 | 35 | 41 | 38 | -77 | -76 | in range |
| 300 | 35 | 128 | 127 | -228 | -228 | in range |
| 709 | 35 | 305 | 304 | -536 | -536 | out of range |
| 1000 | 35 | 431 | 431 | -755 | -755 | out of range |
| 3900 | 35 | 1690 | 1690 | -2937 | -2937 | out of range |

Without the factor, kₙ(√3 κ a) underflows f64 from κ a ≈ 405 (k₀ = e^{−√3κa}/(√3κa); f32
from κ a ≈ 48; at the farthest M2L distance 6a already from κ a ≈ 117), and iₙ(κ a)
overflows f64 at κ a ≈ 717 (f32 near 93); the table's "out of range" at 709 is k's.
Yukawa's leaves are not bounded in κ a, because there is no wavelength (proposal §3.7). Pruning does not
help: it drops far-field lists on coarse levels, and the leaf level's own P2M and L2P
are always formed. With the factor, every Yukawa value in test 4 is finite to κρ = 4550.
The same growth e^{|Im z|} affects jₙ for damped complex k. Hence the recommendation:
remove the factor always, for every kind of k and every level (a no-op for real k),
which needs no threshold and no regime switch.

**Yukawa's degree.** The terms (2n + 1) iₙ(κa) kₙ(κt) of the Yukawa series at the
nearest M2L distance t = 4a/√3, relative to the n = 0 term:

| κ a | ε = 10⁻³ | ε = 10⁻⁶ | ε = 10⁻¹⁰ |
| --- | --- | --- | --- |
| Laplace | 8 | 16 | 27 |
| 0.01 | 8 | 16 | 27 |
| 1 | 10 | 19 | 30 |
| 10 | 22 | 31 | 43 |
| 100 | 65 | 84 | 105 |
| 1000 | 213 | 266 | 324 |

At κ a → 0 the series is Laplace's ((a/t)ⁿ). For larger κ a the terms decay like a
Gaussian of width about √(κ a) before the geometric factor takes over, so the degree for
a relative ε grows like √(κ a ln(1/ε)): it is not bounded by Laplace's. The absolute
size of the far field falls like e^{−κ t}, which pruning (proposal §5.5) exploits; which
of the two governs p_l on unpruned levels is CH2.3's calibration. The fixture and the
sweeps use Helmholtz's larger orders for Yukawa (p(κa) + 1 at the leaves, 2p(κa) for the
M2L), so they cover either.

## 7. Time

`examples/time.rs`, ns per call and per value, median of 15 runs:

| use | family | z | orders | derivative | ns per call | ns per value |
| --- | --- | --- | --- | --- | --- | --- |
| P2M, L2P | j | 5 +0 i | 0..=40 | no | 1267 | 30.9 |
| P2M, L2P | j | 5 +0 i | 0..=40 | yes | 1063 | 25.9 |
| P2M, L2P | j | 5 +5 i | 0..=40 | no | 1059 | 25.8 |
| P2M, L2P | j | 5 +5 i | 0..=40 | yes | 1143 | 27.9 |
| P2M, L2P | j | 0.01 +0 i | 0..=40 | no | 834 | 20.3 |
| P2M, L2P | j | 0.01 +0 i | 0..=40 | yes | 903 | 22.0 |
| M2P, P2L | h | 15 +0 i | 0..=40 | no | 189 | 4.6 |
| M2P, P2L | h | 15 +0 i | 0..=40 | yes | 271 | 6.6 |
| M2P, P2L | h | 15 +15 i | 0..=40 | no | 189 | 4.6 |
| M2P, P2L | h | 15 +15 i | 0..=40 | yes | 272 | 6.6 |
| Yukawa P2M | i | 5 +0 i | 0..=40 | no | 273 | 6.7 |
| Yukawa P2M | i | 5 +0 i | 0..=40 | yes | 404 | 9.9 |
| Yukawa M2P | k | 15 +0 i | 0..=40 | no | 75 | 1.8 |
| Yukawa M2P | k | 15 +0 i | 0..=40 | yes | 98 | 2.4 |
| M2M start | j | 325 +0 i | 0..=1050 | no | 19690 | 18.7 |
| M2M start | j | 325 +0 i | 0..=1050 | yes | 20554 | 19.6 |
| M2L start | h | 1500 +0 i | 0..=1460 | no | 8638 | 5.9 |
| M2L start | h | 1500 +0 i | 0..=1460 | yes | 9712 | 6.6 |
| Yukawa M2L start | k | 1500 +0 i | 0..=72 | no | 174 | 2.4 |
| Yukawa M2L start | k | 1500 +0 i | 0..=72 | yes | 273 | 3.7 |

j costs 3–5 times h: Miller's method makes three passes (the forward start test, the
backward ratios, the upward products) with a complex division per ratio, where h makes
one pass of multiplications. For real k a real-arithmetic j would cost a fraction of
the complex one; that is CH1.2's. At P2M's 41 orders a particle's radial values
cost about 0.8–1.3 µs. (A second run varied by about 10 %.)

## 8. Consequences and questions for sign-off

Questions 1 and 2 were put to the user before the pull request and signed off on
2026-10-27 as recommended ("Circumradius, exp always"; "k₀ = e^{−x}/x"). Questions 3 to
5 need no decision now; they are for CH1.2, D3 and the proposal's next draft.

1. **The scaling of §3.15 item 3.** Recommended: ĵₙ = e^{−|Im z|} jₙ(z)/sⁿ,
   ĥₙ = e^{Im z} hₙ(z) sⁿ, îₙ = e^{−x} iₙ(x)/sⁿ, k̂ₙ = eˣ kₙ(x) sⁿ, and the same on the
   derivatives, with s_l = min(|k| a_l, 1), a_l the circumradius of the level's boxes.
   The exponential factor on every level and every kind of k (not only Yukawa above
   κr ≈ 1, proposal §3.4). Double factorials not absorbed. Alternatives: FMM3D's radius,
   the full width w (its exact convention; the range is the same within 10⁵); the
   half-width w/2 (proposal §3.4's text; the narrowest range, by 10⁸ at low frequency).
2. **kₙ's normalisation.** Recommended: kₙ(x) = √(2/(πx)) K_{n+½}(x), k₀ = e⁻ˣ/x, so
   hₙ(iκρ) = −i⁻ⁿ kₙ(κρ) and the Yukawa series is κ Σ iₙ kₙ Σₘ Yₙᵐ Yₙᵐ. Alternative:
   DLMF's 𝗄ₙ = (π/2) kₙ, which puts 2κ/π in front of the series.
3. **The algorithms for CH1.2.** Recommended: §2 (Miller's ratios with FMM3D's start
   test for j and i, upward for h and k, derivatives from neighbouring orders), with a
   real-arithmetic j for real k as an optimisation measured against this spike. Not
   covered here: the tangential gradient of L2P and M2P divides jₙ and hₙ by z, which
   needs its own limit as z → 0 (jₙ(z)/z for n ≥ 1); CH1.2 forms it from the
   neighbouring orders, (jₙ₋₁ + jₙ₊₁)/(2n + 1), as for the derivative.
4. **f32 (D3, for CH0.6 and CH7.1).** The power scale is out of f32's range on
   low-frequency levels (k a ≤ 1.7); a per-degree envelope scale is the route if D3 puts
   those levels in f32, as a conversion at the f32 boundary. No action now.
5. **Corrections for the proposal's next draft.** §3.4: FMM3D's scale is
   min(|k| w, 1) with the full width, and the exponential factor applies to every kind
   of k, not only Yukawa above κr ≈ 1. §5.5 and CH2.3: Yukawa's degree for a relative
   tolerance grows like √(κ a) (§6); the calibration of p_l and θ_D should take that,
   not Laplace's degree, as its starting point.

## 9. Files

- `Cargo.toml`: package `nd-fmm-spike-radial`, a workspace member outside the default
  members; dev-dependency `serde_json` (workspace).
- `src/radial.rs` (the four families), `src/ranges.rs` (degrees, order ranges, scale
  candidates), `src/complex.rs` (CH0.2's complex type, copied).
- `tests/accuracy.rs`: five tests, numbered as the plan (1 reference, 2 identities,
  3 Yukawa two ways, 4 finite everywhere, 5 scale consistency). Run them with
  `cargo test -p nd-fmm-spike-radial --release -- --include-ignored`.
- `examples/ranges.rs` (§5, §6) and `examples/time.rs` (§7).
- `tools/reference.py`: writes `fixtures/reference.json`; `--check` regenerates it
  byte for byte.
