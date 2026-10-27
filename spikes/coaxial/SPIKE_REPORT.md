# Coaxial translation coefficients: spike report (Helmholtz CH0.2, S2)

Helmholtz task CH0.2 ([docs/helmholtz/tasks/CH0.2.md](../../docs/helmholtz/tasks/CH0.2.md)).
The question is proposal §10 CH0.2, with §3.1, §5.2, §5.3 and risk R2. Written on
2026-10-27 from branch `tbetcke/h-ch0-2` (main at 8e7d25b plus this spike), on the M3
Max, release builds, rustc 1.99.0, f64, one thread.

## Summary

1. **The Gumerov–Duraiswami recursions are unstable at high frequency** (§4.3), which
   confirms R2. For real k the recursion meets 10⁻¹² in the FMM-weighted error of §4.1
   up to k a = 50 (p = 89 at ε = 10⁻¹⁰). Above that the error grows: 6·10⁻¹¹ at k a = 70,
   3·10⁻⁸ at 100, 10⁻³ at 150, and O(1) from 200. The growth depends on k a, not on ε or
   p: at ε = 10⁻³ the errors are the same at each k a. Weak damping (Im k = 0.1 Re k)
   delays it by about a factor 2 in k a. Strong damping (Im k = Re k) and imaginary k
   (Yukawa) stay at 10⁻¹³ or better to k a = 300. Two independent full-table checks agree
   to within a factor 2 at every point: the projection of item 2, and the same recursion
   in arbitrary precision (`tools/full_table.py`).
2. **Projection is stable everywhere** (§4.4). It evaluates the translated wave on a
   sphere about the new centre and projects it by Gauss–Legendre quadrature. This is
   FMM3D's stabilised "point and shoot", with each degree solved by least squares from
   the value and the radial derivative. Against the arbitrary-precision reference it
   meets **1.2·10⁻¹²** in the weighted error on all 120 configurations: p ≤ 371,
   k a ≤ 300, real, complex and imaginary k. It is 10–60 times slower to build: one
   p = 371 M2L table takes 5.1 s instead of 85 ms (§7).
3. **Per-entry relative error is neither reachable nor needed** (§4.2). At large p a
   table spans up to 10²⁰⁰ and more, because hₗ grows and jₗ decays beyond l ≈ |k t|.
   Its smallest entries cancel to nothing in f64 by any method. Those entries only ever
   multiply coefficients that the geometry makes equally small. The recursion is
   accurate entry by entry (≤ 4·10⁻¹⁴) only on tables of small dynamic range (k a ≤ 10 at
   the near distance). §4.1 defines the metric that matters, the error of the field the
   table produces. The acceptance item "relative error" is reported both ways, and the
   change of metric is question 2 of §8.
4. **(S|S) = (R|R), real for real k** (§5). For real k the recursion for (R|R) closes
   over the reals: every imaginary part is exactly zero. Re (S|R) = (R|R) holds to
   10⁻¹⁴ in the M2L weights for the recursion (k a ≤ 30, all 15 distances) and to
   2·10⁻¹⁵ for the projection (k a = 100 and 300). In the reference the addition
   theorems hold to 5·10⁻⁴⁹ with (S|S) = (R|R), for real, complex and imaginary k. For
   imaginary k, i^{n−n′} (E|F)ᵐ_{n′n} is real for (R|R) and for (S|R), to round-off in
   the reference and exactly in f64.
5. **The index symmetry halves the tables** (§6). The relation is
   (E|F)ᵐ_{nn′} = (−1)^{n+n′} (E|F)ᵐ_{n′n}, and the coefficient depends on |m| only. In
   FMM3D's symmetric scaling s^{n+n′} of the M2L, the symmetry carries over to the scaled
   table. The reflection (E|F)(−t) = (−1)^{n+n′} (E|F)(t) gives the β and π − β shifts
   of M2M from one table. The recursion must use the symmetry, not only exploit it: the
   triangle n′ ≥ n is closed under the recursions and stable, while the upper triangle
   marched directly is wrong by O(1) even at k a = 0.1. Rectangular truncation
   p_src → p_tgt is a block of the square table, bit for bit.
6. **Recommendation** (§8): build the coaxial tables by the recursion in its triangle
   where Re(k) a ≤ 40, and by projection above. The rule uses only k and the geometry
   (constraint 9). It keeps every low-frequency level, and the high-frequency levels to
   about 6 wavelengths per box, on the fast path. Replace R2's fallback "through
   signatures" by the projection: the signature route is ill-conditioned at exactly the
   near classes that RCR serves on high-frequency levels.

## 1. Basis, normalisation and scaling

The basis is the real orthonormal Yₙᵐ: CONVENTIONS §3.3's Legendre functions without the
Condon–Shortley phase, cos in slot +m and sin in slot −m (§3.6), normalised so that
Σₘ Yₙᵐ(x̂) Yₙᵐ(ŷ) = (2n + 1)/(4π) Pₙ(x̂ · ŷ), as proposal §3.1's expansion needs. It is
CH0.1's basis (the real storage of N Rₙ) times √((2n + 1)/(4π)) per degree. A rotation
does not see that factor; a translation does, and the index symmetry of §6 needs it.
§3.15 is not written yet (CH1.4, CH1.1), so this basis is the spike's choice. Proposal
§3.1 implies it.

With Fₙᵐ(r) = fₙ(k r) Yₙᵐ(r̂), f = j (regular) or h = h⁽¹⁾ (singular), a translation by
t = t ẑ is Fₙᵐ(r + t) = Σ_{n′} (E|F)ᵐ_{n′n}(t) G_{n′}ᵐ(r), with one coefficient for the
slots +m and −m. This gives (R|R) at any r, (S|R) for |r| < t, and (S|S) for |r| > t.

Normalisation of the stored values: (R|R) times e^{−|Im z|}, and (S|R) times e^{Im z}
s^{n+n′}, with z = k t. The exponential factor is common to a table, and a linear
recursion carries it unchanged. It removes the growth or decay of jₗ and hₗ with Im z,
which is what makes imaginary κ t = 1800 representable. The power scale s = min(|k| a, 1)
is FMM3D's (proposal §3.4), applied at both ends of an M2L, which keeps the scaled table
symmetric. (R|R) is stored unscaled: at the configurations here the scaling is not
needed for range. Whether and how §3.15 scales is S3's (CH0.3) and CH1.1's decision.

## 2. Method

### 2.1 The recursions (`src/coaxial.rs`)

Three identities give every coefficient from the start column
(E|F)⁰_{n′0} = (−1)^{n′} √(2n′ + 1) f_{n′}(k t), which follows from Gegenbauer's addition
theorem:

- **in n**, from (1/k) ∂_z Fₙᵐ = aᵐₙ₋₁ Fᵐₙ₋₁ − aᵐₙ Fᵐₙ₊₁ commuting with the translation:
  aᵐₙ Eᵐ_{n′,n+1} = aᵐₙ₋₁ Eᵐ_{n′,n−1} − aᵐ_{n′} Eᵐ_{n′+1,n} + aᵐ_{n′−1} Eᵐ_{n′−1,n}, with
  aᵐₙ = √((n + 1 + m)(n + 1 − m)/((2n + 1)(2n + 3)));
- **in m**, from (1/k)(∂_x + i∂_y) Fₙᵐ = −βₙᵐ Fᵐ⁺¹ₙ₋₁ − γₙᵐ Fᵐ⁺¹ₙ₊₁ at n = m:
  γᵐₘ Eᵐ⁺¹_{n′,m+1} = βᵐ_{n′+1} Eᵐ_{n′+1,m} + γᵐ_{n′−1} Eᵐ_{n′−1,m}, with
  βₙᵐ = √((n − m − 1)(n − m)/((2n − 1)(2n + 1))) and
  γₙᵐ = √((n + m + 1)(n + m + 2)/((2n + 1)(2n + 3)));
- **the index symmetry** Eᵐ_{nn′} = (−1)^{n+n′} Eᵐ_{n′n}.

Both signs of the m-ladder are positive without the Condon–Shortley phase. They were
fitted numerically in mpmath before use; the first guess, by analogy with ∂_z, was wrong.
Each recursion was checked against the Gaunt sum of §3 to 10⁻³⁵ at p = 8 for real,
complex and imaginary k.

Two variants: **Triangle** computes the triangle n′ ≥ n of every order. It is closed
under both recursions, because every entry it reads is in it, and the rest comes from
the symmetry. **Full** marches every row n′ ≥ m of every column, as the recursions are
usually written. A table is computed order by order, in O(p³) operations and O(p)
working memory; the start column needs orders to p_src + p_tgt.

### 2.2 Projection (`src/projection.rs`)

Take points r = ρ(sin α, 0, cos α) on a sphere about the new centre, at Gauss–Legendre
nodes in cos α. The translated wave fₙ(k R) p̄ₙᵐ(cos Θ) about the old centre equals
Σ_{n′} Eᵐ_{n′n} g_{n′}(k ρ) p̄_{n′}ᵐ(cos α), where p̄ are the orthonormal associated
Legendre functions and the φ factor is common to both sides. Projecting onto p̄_{n′}ᵐ
gives Eᵐ_{n′n} g_{n′}(kρ):

- **(S|R)**: f = h, g = j, with ρ = a (the target box) < t. Since jₙ(kρ) has zeros for
  n < |k|ρ, each degree is solved by least squares from the value and the radial
  derivative: E = (A conj ĵ + B conj ĵ′)/(|ĵ|² + |ĵ′|²). The division is by a small
  number only where evaluation at radius ρ also makes the coefficient small.
- **(R|R) = (S|S)**: f = g = h, on a sphere ρ > t, where hₙ(kρ) never vanishes. The
  spike takes ρ = 4a/√3, the nearest M2L distance of the parent, the radius at which
  §4.1 weighs M2M's output.

The cost is O(N p³) per table with N nodes. N = max(p_src, p_tgt) + 40 reaches the
rounding floor on every configuration. Tested with N = p + 10, 20, 40, 60 against
N = 2p + 200: N = p + 20 leaves up to 10⁻⁸ at the near distance for damped k, and N = p
+ 40 leaves ≤ 7·10⁻¹³.

### 2.3 Radial functions (`src/radial.rs`)

The radial functions are spike grade, because S3 (CH0.3) does them properly. hₗ comes by
upward recursion; jₗ by Miller's downward recursion from 1.2 max(L, |z|) + 60, rescaled
on the way down and normalised by ĵ₀ or ĵ₁. Against the reference the start values agree
to **3.3·10⁻¹⁴** relative at every sampled order to 742, for |z| from 0.005 to 1800,
real, complex and imaginary.

## 3. The reference

`tools/reference.py` (PEP 723, `mpmath==1.4.1` as in `tools/fixtures/`) computes each
coefficient from the Gaunt sum

(E|F)ᵐ_{n′n} = (−1)ᵐ √((2n + 1)(2n′ + 1)) Σₗ i^{n′−n+l} (2l + 1) (n n′ l; 0 0 0)
(n n′ l; m −m 0) fₗ(k t),

an independent formula. The 3j symbols come from their three-term recursion in l
downward from the closed form at l = n + n′. Each value is computed at 40 + (p_src +
p_tgt)/2 digits and again with 50 more, and the two must agree to 10⁻²⁵. Before the
fixture is written, the script checks three things:

- the radial recurrences against mpmath's `besselj` and `hankel1` (1.5·10⁻⁴⁰), and its
  Legendre recursion against `legenp` (2.8·10⁻⁴⁰);
- the formula against the definition: the addition theorems evaluated pointwise at
  n ≤ 6, every slot, three points each, for (R|R), (S|R), (S|S) with the (R|R)
  coefficients, and the reflection t → −t, for k = 1.3, 1.3 + 0.4i and 1.3i. Worst
  **5.0·10⁻⁴⁹** at 60 digits.

`fixtures/reference.json` (1.2 MB) holds 21,809 entries in 120 configurations:

- k a ∈ {0.01, 0.1, 1, 3, 10, 30, 100, 300}, with a the circumradius of a box;
- k real, complex (κ(1 + 0.1i) and κ(1 + i)) and imaginary (iκ);
- p = p(k a, ε) of proposal §3.5 at ε = 10⁻¹⁰, and for real k also at 10⁻³ (p from 12
  to 371);
- per configuration, M2M at the child shift (p_child → p_parent) and M2L at d = 2 and
  d = √27 box widths.

Per configuration it samples orders 0, 1, the middle and the largest; rows and columns
at both ends; the diagonal; and random entries. z and s are the f64 values the spike
uses. Regeneration is byte for byte (`--check`, §9).

## 4. Accuracy

### 4.1 The metric

A coaxial table only ever multiplies bounded coefficients:

- **M2L, (S|R)**, between boxes of circumradius a: a multipole of charges within a has
  |Mₙ| ≤ |jₙ(k a)| √(2n + 1) per unit charge, and a local coefficient of degree n′
  reaches its targets through |j_{n′}(k a)| √(2n′ + 1). The weight is uₙ u_{n′}.
- **M2M and L2L, (R|R) = (S|S)**: the child's multipole is bounded by
  |jₙ(k a/2)| √(2n + 1). The parent's multipole of degree n′ is evaluated no closer than
  the nearest M2L distance 4a/√3, through |h_{n′}(k 4a/√3)| √(2n′ + 1). L2L has the same
  weight with the indices exchanged.

The **weighted error** max |ΔÊ| w / max |Ê| w is then the relative error of the field a
table produces, up to a factor of order p from summing. `src/metric.rs` implements it.

### 4.2 Per-entry error

Per-entry relative error at the sampled entries (test 1), ε = 10⁻¹⁰, real k:

| k a | M2M recursion | M2L d = 2 recursion | M2L d = √27 recursion | projection, any |
| --- | --- | --- | --- | --- |
| 0.01 | 1.4e-15 | 2.6e-15 | 1.8e-15 | up to 3e225 |
| 1 | 1.6e-15 | 2.4e-15 | 1.8e-15 | up to 3e85 |
| 10 | 4.4e-14 | 1.2e-14 | 1.9e-6 | up to 3e20 |
| 30 | 4.3e-12 | 3.8e-3 | 2.8e19 | up to 6e18 |
| 100 | 1.0e-4 | 2.5e30 | 1.3e92 | up to 2e39 |
| 300 | 1.4e14 | 1.1e132 | 1.1e190 | up to 1e90 |

The large values are in entries that are tiny relative to the rest of their table: the
smallest sampled entries reach 10⁻²⁹⁰. Rounding of the large entries swamps them by any
method, and their weights make them as small in the field. The projection has the large
per-entry errors even where it is exact in the weighted sense, because it computes each
entry to an absolute accuracy set by its weight.

### 4.3 The recursion

Weighted error of the recursion (triangle variant) over the whole table, against the
projection. The M2L column is the worst of the 15 V-list distances (`examples/sweep.rs`,
d² in brackets). ε = 10⁻¹⁰:

| k a | p_child → p | real: M2M | real: M2L | Im k = 0.1 Re k: M2M | M2L | Im k = Re k: M2M | M2L | imaginary: M2M | M2L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 35 → 35 | 1.2e-14 | 5.7e-16 | 1.3e-14 | 6.6e-16 | 1.2e-14 | 5.3e-16 | 1.2e-14 | 6.0e-16 |
| 1 | 35 → 35 | 1.2e-14 | 6.7e-16 | 1.2e-14 | 6.5e-16 | 1.3e-14 | 8.8e-16 | 1.4e-14 | 1.3e-15 |
| 10 | 37 → 40 | 3.3e-15 | 7.5e-15 | 4.1e-15 | 8.5e-15 | 6.8e-15 | 8.3e-15 | 5.8e-15 | 6.7e-15 |
| 30 | 46 → 64 | 8.1e-14 | 4.1e-14 | 1.9e-13 | 1.6e-13 | 2.9e-13 | 3.2e-13 | 2.3e-13 | 2.1e-13 |
| 40 | 51 → 77 | 4.4e-14 | 9.7e-14 | 6.9e-14 | 7.8e-14 | 4.1e-14 | 8.6e-14 | 3.2e-14 | 4.7e-14 |
| 50 | 58 → 89 | 4.8e-13 | 7.2e-13 (4) | 3.4e-13 | 4.1e-13 | 4.3e-13 | 5.7e-13 | 3.4e-13 | 3.7e-13 |
| 70 | 70 → 113 | 1.7e-11 | 6.2e-11 (4) | 1.7e-12 | 1.1e-12 | 1.3e-13 | 1.4e-13 | 1.2e-13 | 1.0e-13 |
| 100 | 89 → 149 | 5.5e-9 | 2.7e-8 (4) | 1.2e-10 | 2.6e-11 | 4.5e-13 | 6.6e-13 | 3.6e-13 | 4.1e-13 |
| 150 | 119 → 206 | 9.5e-5 | 1.1e-3 (4) | 5.5e-8 | 5.5e-9 | 3.7e-13 | 4.5e-13 | 2.9e-13 | 3.1e-13 |
| 200 | 149 → 262 | 2.0 | 52 (4) | 3.8e-5 | 1.5e-6 | 2.4e-13 | 5.0e-13 | 2.2e-13 | 3.1e-13 |
| 300 | 206 → 371 | 7.9e8 | 1.3e16 (5) | 7.7 | 76 | 7.1e-13 | 1.2e-12 | 5.4e-13 | 7.9e-13 |

Below 10⁻¹² the numbers are the projection's own floor (§4.4), not the recursion's error.
At ε = 10⁻³ (p from 12 to 331) the errors at each k a are the same to two digits, so the
instability depends on k a, not on p. The near distance d = 2 is the worst at every
unstable point.

**Independent check.** `tools/full_table.py` runs the same recursion in f64 and at 120
digits (the exact run agreeing with the Gaunt sum to 10⁻¹⁰⁵ or better) and compares
every entry in the same metric:

| | f64 weighted error | the sweep, against the projection |
| --- | --- | --- |
| M2L k a = 30, d = 2 | 1.5e-14 | ≤ 4.1e-14 |
| M2M k a = 30 | 1.1e-14 | 8.1e-14 |
| M2L k a = 50, d = 2 | 7.6e-13 | 7.2e-13 |
| M2L k a = 100, d = 2 | 2.9e-8 | 2.7e-8 |
| M2L k a = 100, d = √27 | 5.0e-13 | (5.4e-13 by debug run) |
| M2M k a = 100 | 4.8e-9 | 5.5e-9 |

A first version of this tool rounded the recursion coefficients to f64 in the "exact"
run. Its truth then disagreed with the Gaunt sum by up to 10⁶, because the recursion
amplifies a perturbation of its coefficients as much as one of its values.

**The fixture's samples miss the instability.** At k a = 100 the sampled weighted error
of the recursion is 2.5·10⁻¹⁵ for M2L d = 2, against 2.7·10⁻⁸ over the full table. Test
1 therefore asserts the recursion only where the full-table sweep finds it stable, and
the claims of this section rest on the full tables.

**Where the error comes from** (the diagnosis of `tools/full_table.py`'s predecessor at
k a = 100, d² = 6, m = 34):

- The recursion in m on the sectorial column is absolutely stable: its absolute error
  stays at 10⁻¹⁷ for every m, while the values fall from 0.3 to 10⁻¹⁸.
- The recursion in n is not. Its error appears in rows near the top of the computed
  region (n + n′ above |k t|, where jₗ is the decaying solution). It grows about 2.7× per
  column, from 3·10⁻¹⁷ at n = 34 to 5·10⁻² at n = 146, and spreads to lower rows. The
  absolute error in (S|R) equals that in (R|R) to two digits, so it is the j part of h
  that carries it.
- A mixed recursion that raises m and n together (from the ∂_x + i∂_y identity at
  general n) was tried as a replacement and is worse: weighted error 2·10³ at m = 70 for
  k a = 100.

For Im k = Re k and for imaginary k the instability is absent to k a = 300; why damping
removes it was not investigated.

### 4.4 The projection

Against the reference (test 1), the weighted error over the sampled entries,
normalised by the full table, is at most **1.2·10⁻¹²** over all 120 configurations
(worst: Im k = 0.1 Re k, k a = 300, M2L d = 2). Typical values:

- ≤ 10⁻¹⁴ at k a ≤ 10;
- 2–6·10⁻¹⁴ at k a = 30 to 100 for real k;
- 1–2.4·10⁻¹³ at k a = 300 for real k;
- 4–12·10⁻¹³ for damped and imaginary k at k a = 300.

Against the full recursion table where that is stable, it agrees to ≤ 10⁻¹³ (the table
of §4.3). The floor grows slowly with p and with damping.

## 5. Reality of (S|S) = (R|R), and Yukawa

- **(S|S) = (R|R)** holds as functions of t. The two share the start column and the
  recursions, and the definition check confirms it to 5·10⁻⁴⁹ for real, complex and
  imaginary k (§3).
- **Real k (test 2).** The f64 (R|R) recursion has every imaginary part exactly 0.0 at
  every table entry: its start values jₗ(z) are real and the recursion closes over the
  reals. The reference's (R|R) is real by its formula (n′ − n + l is even). The
  non-trivial check is Re (S|R) = (R|R), since hₗ = jₗ + i yₗ and the Gaunt phases are
  ±1. In the M2L weights it holds to **1.0·10⁻¹⁴** by the recursion (k a ∈ {0.01, 1, 10,
  30}, all 15 distances). By the projection at k a = 100 and 300 (d = 2) it holds to
  **1.9·10⁻¹⁵**, and the projected (R|R) has imaginary parts exactly zero there too.
- **Imaginary k (test 3).** With k = iκ: jₗ(iκt) = iˡ iₗ(κt), and hₗ(iκt) = −i⁻ˡ times a
  positive multiple of kₗ(κt). So i^{n−n′} (E|F)ᵐ_{n′n} is real for both (R|R) and (S|R).
  In f64 the imaginary part of i^{n−n′} Ê is exactly zero at k a = 1 and 30. In the
  reference it is below 10⁻³⁰ relative at every sampled entry. This is proposal §3.1's
  "powers of i absorbed per degree": the Yukawa tables are real once the basis carries
  iⁿ. Proposal §4 item 6 says (S|S) = (R|R) is "real for Yukawa"; it is real only after
  that absorption, which §3.15 has to state.

## 6. The index symmetry and the table layout

- **Symmetry.** Eᵐ_{nn′} = (−1)^{n+n′} Eᵐ_{n′n} follows from the Gaunt formula (the 3j
  products are symmetric, and i^{n′−n+l} and i^{n−n′+l} differ by (−1)^{n+n′}). The
  definition check confirms it at n ≤ 6. The halved table (`Table`, the triangle n′ ≥ n
  per order) reproduces every sampled reference entry in both triangles. The scaled
  M2L table Ê = E s^{n+n′} keeps the symmetry, because s is the same at both ends.
- **|m| only.** The slots +m and −m share a coefficient. The y-reflection maps the
  slots into each other and fixes t, and the definition check confirms it for every
  slot.
- **Reflection.** (E|F)(−t) = (−1)^{n+n′} (E|F)(t), checked in the definition at
  t = ±0.9. The eight child shifts of M2M therefore need one table: the four with polar
  angle π − β read it with signs, as CH0.1 found for the rotation blocks.
- **Size.** A square table of degree p stores Σₘ (p − m + 1)(p − m + 2)/2 ≈ p³/6
  complex values: 8.8 MiB at p = 149 and 132 MiB at p = 371 (complex f64). M2L needs one
  per V-list distance, 15 per level: 2 GiB at p = 371 even halved. For real k, (R|R) is
  real, which halves M2M's table again. §8 question 3 takes this to the memory model.
- **The symmetry is needed for stability, not only for memory.** The full variant
  computes the triangle n′ ≥ n bit for bit like the triangle variant (test 4: the
  triangle is closed under the recursions). Its upper triangle, marched away from the
  diagonal, differs from the symmetric values by **1.8** (k a = 3) and **1.4**
  (k a = 0.1) relative for (R|R). It is fine for (S|R) (1.6·10⁻¹⁵). That is the
  analogue of CH0.1's finding that GD's rotation recursion must stay in its wedge.
- **Rectangular truncation** (test 5). The table p_src → p_tgt (M2M's p_child →
  p_parent, and L2L's reverse) equals the corresponding block of the square table at
  max(p_src, p_tgt) bit for bit, from the same start values.

## 7. Time and memory per table

One thread, M3 Max, real k, ε = 10⁻¹⁰, best of 2 to 5 runs (`examples/time.rs`). The
recursion includes the start values; the projection uses N = p + 40 and plain loops
(no BLAS):

| k a | p_child → p | M2M recursion | M2M projection | M2L recursion | M2L projection | M2L table (halved) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 35 → 35 | 0.08 ms | 0.85 ms | 0.08 ms | 1.4 ms | 0.1 MiB |
| 10 | 37 → 40 | 0.11 ms | 1.1 ms | 0.12 ms | 1.9 ms | 0.2 MiB |
| 30 | 46 → 64 | 0.28 ms | 3.0 ms | 0.45 ms | 8.3 ms | 0.7 MiB |
| 50 | 58 → 89 | 0.64 ms | 7.4 ms | 1.2 ms | 28 ms | 1.9 MiB |
| 100 | 89 → 149 | 2.7 ms | 41 ms | 5.9 ms | 179 ms | 8.8 MiB |
| 200 | 149 → 262 | 14 ms | 305 ms | 31 ms | 1.4 s | 47 MiB |
| 300 | 206 → 371 | 38 ms | 1.05 s | 85 ms | 5.1 s | 132 MiB |

The recursion is O(p³) and the projection O(N p³). The projection's inner loop is a
complex (p × N)·(N × p) product per order, which BLAS would run several times faster.
The recommended split (§8) puts it only where k a > 40. There, a level's 15 M2L tables
by projection take about 3 s at k a = 100 and 75 s at k a = 300 on one thread, plus one
M2M table. Load average during the runs was about 2.5.

## 8. Consequences and questions for sign-off

1. **How the coaxial tables are built.** Recommended: the recursion, in its triangle
   with the index symmetry, where Re(k) a ≤ 40 (a the circumradius of the level's boxes;
   the weighted error there is ≤ 3.2·10⁻¹³ for every kind of k). Projection above that,
   with N = p + 40. The threshold is a margin below the measured 10⁻¹² at k a = 50; it
   depends only on k and the geometry (constraint 9). Alternative: projection
   everywhere, which is simpler but 10–60× slower to build, and slower at low
   frequency where most of the tables are. A per-kind threshold (strong damping is
   stable everywhere) is possible but not worth a second rule.
2. **The acceptance metric of CH0.2.** Per-entry relative error at p = 300 is not
   achievable in f64 by any method and not needed (§4.2). Proposed: record CH0.2's
   acceptance as met in the FMM-weighted error of §4.1, bound 10⁻¹¹: projection
   ≤ 1.2·10⁻¹² everywhere, recursion ≤ 3.2·10⁻¹³ where the rule of question 1 uses it.
   The proposal's §9 item 2 tests should use the same weights for coaxial tables.
3. **R2's fallback, and the memory model.** Proposed: replace "compute the coaxial step
   through signatures at the top levels" (proposal §8 R2) by "projection (FMM3D's
   stabilised point and shoot) above Re(k) a = 40". The signature route needs the
   diagonal form's transfer function, which is ill-conditioned for the near classes
   (§3.6). Those are exactly the classes for which RCR remains on high-frequency levels.
   Also for §7 and CH0.8: halved M2L coaxial tables reach 132 MiB each at p = 371, or 2
   GiB for a level's 15 distances. Projection can also be applied without a table, at
   O(N p²) per order and translation (two products against N × p matrices of the
   distance), which the memory rule may prefer at the top levels.
4. **§3.15 item 6** should state: the coefficients depend on |m| only; the index
   symmetry and the reflection of §6; (S|S) = (R|R), real for real k; for k = iκ,
   i^{n−n′} (E|F) real for (R|R) and (S|R) once the basis absorbs the powers of i; the
   start column; and the recursions with their signs (§2.1) where they are used.

## 9. Files

- `Cargo.toml`: package `nd-fmm-spike-coaxial`, a workspace member outside the default
  members; dev-dependency `serde_json` (workspace).
- `src/lib.rs` (start values, `Op`), `src/coaxial.rs` (recursions, variants, the halved
  `Table`), `src/projection.rs` (Gauss–Legendre, projection, `Dense`),
  `src/metric.rs` (weights, weighted comparison), `src/radial.rs`, `src/config.rs`
  (degrees and shifts), `src/complex.rs`.
- `tests/accuracy.rs`: the six tests of the plan. Run them with
  `cargo test -p nd-fmm-spike-coaxial --release -- --include-ignored` (56 s).
- `examples/sweep.rs` (§4.3, 9 min for ε = 10⁻¹⁰ and 10⁻³) and `examples/time.rs` (§7).
- `tools/reference.py`: writes `fixtures/reference.json`. Regenerating with `--check`
  matches byte for byte (about 4 min).
- `tools/full_table.py`: the arbitrary-precision full-table check of §4.3.
