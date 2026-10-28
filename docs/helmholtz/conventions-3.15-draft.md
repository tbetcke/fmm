# Draft: CONVENTIONS §3.15, the Helmholtz family

Status: **draft** (Helmholtz CH1.4, 2026-10-28). It is not part of `docs/CONVENTIONS.md`.
No code may cite it as `CONVENTIONS §3.15` yet. CH1.1 turns it into that section, with a
`CONVENTION_VERSION` bump, once the user has signed it off. Sources: proposal draft 0.8 §1.1,
§3.1–§3.6 and §4 (the nine items below); the S1–S3 spike reports and their sign-offs
(`spikes/rotation-blocks/`, `spikes/coaxial/`, `spikes/radial/`, docs/helmholtz/progress.md);
CH1.4's two-point check against FMM3D 2.1.0 (`fmm-validate/results/helmholtz-ch1-4.md`).

Each item is marked:

- **fixed by CH1.4**: checked against FMM3D on two points, or against the S1–S3
  references;
- **from S1–S3**: taken from a signed-off spike result;
- **proposed**: CH1.1 asks for sign-off;
- **open**: a later task decides it, and that task is named.

Notation as in CONVENTIONS §3.2–§3.8: P_n^m without the Condon–Shortley phase (§3.3),
idx(n, m) = n² + n + m (§3.6), the z-y-z Euler angles (§3.8).

## 1. Kernel and factors (fixed by CH1.4)

```math
G_k(\mathbf{x}, \mathbf{y}) = \frac{e^{ik|\mathbf{x}-\mathbf{y}|}}{4\pi|\mathbf{x}-\mathbf{y}|},
\qquad k \ne 0,\quad \operatorname{Re} k \ge 0,\quad \operatorname{Im} k \ge 0
```

- **Three kinds of k.** Yukawa (modified Helmholtz) has k = iκ with κ > 0, so
  G = e^{−κr}/(4πr). Real k has k > 0. Complex k has Re k > 0 and Im k > 0. k = 0 is
  the Laplace FMM of §3.1 and is not a Helmholtz case. k with Im k < 0 (growing kernels)
  or Re k < 0 (incoming waves) is rejected by the builder.
- **Branch.** The kernel is the outgoing wave for the time factor e^{−iωt}. The expansions
  use spherical Hankel functions of the first kind, h_n = h_n^{(1)} = j_n + i y_n
  (item 3). For Im k > 0 the kernel decays like e^{−Im k · r}.
- **Potential and gradient.** For charges q_j at y_j, with coincident pairs excluded as in
  §3.13 ("Coincident pairs"):

```math
\phi(\mathbf{x}) = \sum_j q_j\, G_k(\mathbf{x}, \mathbf{y}_j), \qquad
\nabla\phi(\mathbf{x}) = \sum_j q_j\, \frac{(ik r_j - 1)\, e^{ik r_j}}{4\pi r_j^3}\, (\mathbf{x} - \mathbf{y}_j),
\quad r_j = |\mathbf{x} - \mathbf{y}_j|
```

  The gradient is taken with respect to the target x. As k → 0 both reduce to §3.1 and
  §3.13: φ = q/(4πr) and ∇φ = −q (x − y)/(4πr³).
- **Where the factors go** (proposed). The expansion of proposal §3.1 is

```math
\frac{e^{ik|\mathbf{x}-\mathbf{y}|}}{4\pi|\mathbf{x}-\mathbf{y}|}
= ik \sum_{n} j_n(k r_<)\, h_n(k r_>) \sum_{m=-n}^{n} Y_n^m(\hat{\mathbf{y}})\, Y_n^m(\hat{\mathbf{x}})
```

  As for Laplace (§3.1), 1/(4π) is applied once, by `nd-fmm-exec`, when it produces the
  output. Every operator accumulates the potential of e^{ikr}/r into a target's output,
  P2P and the expansions alike. The factor ik is therefore part of the far field. The
  proposal is that the three operators where a far-field contribution crosses from a
  singular to a regular representation carry it: M2L (in its (S|R) table), M2P and P2L.
  Each far-field contribution passes through exactly one of the three. The alternative is
  that P2P computes e^{ikr}/(ik r) and `nd-fmm-exec` applies ik/(4π) once. That costs no
  table factor, but it makes the near field singular as k → 0. CH1.1 chooses between the
  two, and CH1.2 implements the choice.

FMM3D 2.1.0 computes exactly this φ and ∇φ: `hfmm3d_s_c_g`, `hfmm3d_st_c_g` and
`h3ddirectcg`, with k = `zk`. The two-point identification is in the CH1.4 report. FMM3D
drops a pair when |x − y| < thresh. Its FMM uses thresh = 2⁻⁵¹ times the root box width,
and its direct routines take thresh from the caller. §3.13 drops exact coincidences only.

## 2. Angular basis (from S1–S2; proposed)

Real orthonormal spherical harmonics in the slots of §3.6: the cos part in slot +m, the
sin part in slot −m, no Condon–Shortley phase.

```math
Y_n^0 = \sqrt{\tfrac{2n+1}{4\pi}}\, P_n(\cos\theta), \qquad
Y_n^{\pm m} = \sqrt{2\,\tfrac{2n+1}{4\pi}\,\tfrac{(n-m)!}{(n+m)!}}\; P_n^m(\cos\theta)
\begin{cases}\cos m\phi \\ \sin m\phi\end{cases} \quad (m > 0)
```

They satisfy ∫ Y_n^m Y_{n′}^{m′} dΩ = δ_{nn′} δ_{mm′} and
Σ_m Y_n^m(x̂) Y_n^m(ŷ) = (2n + 1)/(4π) P_n(x̂ · ŷ). Against the Racah harmonics of
§3.3, r^n Y_n^m is √((2n + 1)/(4π)) times the slot ±m entry of N R_n(x) (N as §3.8):
slot +m holds √2 ν_m Re R_n^m, slot −m holds √2 ν_m Im R_n^m, and slot 0 holds ν₀ R_n^0,
with ν_m = √((n + |m|)!(n − |m|)!). They are evaluated with the Cartesian recursions of §3.5
(no trigonometric calls), in the overflow-free form of S1 §2.2 at large degree.

## 3. Radial functions and scaling (from S3, signed off 2026-10-27)

- j_n is the spherical Bessel function and h_n = h_n^{(1)}. For Yukawa,
  i_n(x) = √(π/(2x)) I_{n+½}(x) and k_n(x) = √(2/(πx)) K_{n+½}(x), so that
  k₀(x) = e^{−x}/x, j_n(ix) = iⁿ i_n(x) and h_n(ix) = −i^{−n} k_n(x).
- **The scale** of level l: s_l = min(|k| a_l, 1), with a_l = (√3/2) w_l the circumradius
  of a box (not the half-width of proposal §3.4, which S3 replaced).
- **Scaled radial functions**, for z = kρ with Im z ≥ 0 and x = κρ:

```text
ĵₙ = e^{−|Im z|} jₙ(z) / sⁿ        ĥₙ = e^{Im z} hₙ(z) sⁿ
îₙ = e^{−x} iₙ(x) / sⁿ            k̂ₙ = eˣ kₙ(x) sⁿ
```

  The exponential factor is used on every level and for every kind of k. The derivatives
  carry the same factors, which are not differentiated. No double factorial is absorbed.
- **Scaled coefficients** (proposed): the stored multipole is M̂ = M / s_lⁿ and the stored
  local is L̂ = L s_lⁿ. The product of a regular and a singular value of the same order is
  then the unscaled product times e^{Im z₂ − |Im z₁|}, a factor common to a box pair.
  Which operator applies the exponential factors (per point, per box or per table) is
  open for CH1.2.
- **Yukawa's powers of i.** With k = iκ the expansion is

```math
\frac{e^{-\kappa|\mathbf{x}-\mathbf{y}|}}{4\pi|\mathbf{x}-\mathbf{y}|}
= \kappa \sum_n i_n(\kappa r_<)\, k_n(\kappa r_>) \sum_m Y_n^m(\hat{\mathbf{y}})\, Y_n^m(\hat{\mathbf{x}})
```

  Every factor is real. The Yukawa path stores M^Y = i^{−n} M and L^Y = −iⁿ L. Here M and
  L are the coefficients of Σ M h_n Y and Σ L j_n Y at k = iκ, without the factor ik.
  These relations are proposed, and §9 item 4 ("Yukawa two ways", CH1.3) tests them.

## 4. Storage (proposed, from proposal §3.3)

- **A complex expansion** is two real planes per box, the real part then the imaginary
  part, each in the layout idx(n, m) of §3.6: 2(p_l + 1)² reals per charge vector.
- **Real expansions** have (p_l + 1)² reals. They are used for Yukawa with real charges,
  multipoles and locals alike, and for real k with real charges, multipoles only.
- **Sizes per path**, with m charge vectors:

| Path | multipole_size | local_size |
| --- | --- | --- |
| Yukawa, real charges | (p_l + 1)² m | (p_l + 1)² m |
| real k, real charges | (p_l + 1)² m | 2 (p_l + 1)² m |
| every other case | 2 (p_l + 1)² m | 2 (p_l + 1)² m |

- **Complex charges.** One complex charge vector makes **one** complex expansion (P2M
  forms it from both input planes). On the Yukawa path it is two real vectors instead.
- **Several vectors** are stored as in §3.13: m expansions, one after the other.

## 5. Rotations (from S1, signed off 2026-10-27)

- The degree-n block Oⁿ(Q) satisfies Y_n(Qx̂) = Oⁿ(Q) Y_n(x̂) and equals N Dⁿ(Q) N⁻¹
  (§3.8); the per-degree factor of item 2 commutes with it. Oⁿ is orthogonal.
- The z-y-z factorisation Oⁿ(R_z(α)) Oⁿ(R_y(β)) Oⁿ(R_z(γ)). A z-rotation is a 2 × 2
  rotation per order. A y-block splits into a cos half ((n + 1)² values) and a sin half
  (n² values).
- **β and π − β**: Oⁿ(π − β) = (−1)ⁿ E Oⁿ(β) E F, with E = diag((−1)^|m|) and
  F = diag(+1 on the cos slots, −1 on the sin slots). Tables store the blocks of
  β ≤ π/2 and apply the signs.
- **The generator** is the Gumerov–Duraiswami recursion at every degree. The N-conjugated
  blocks of §3.8 are the test reference at p ≤ 30 only.

## 6. Coaxial translations (from S2, signed off 2026-10-27)

- With F_n^m(r) = f_n(kr) Y_n^m(r̂) (f = j regular, f = h singular) and t = t ẑ:
  F_n^m(r + t) = Σ_{n′} (E|F)^m_{n′n}(t) G_{n′}^m(r). This gives (R|R) at any r, (S|R) for
  |r| < t and (S|S) for |r| > t. The slots +m and −m share one coefficient (|m| only).
- (S|S) = (R|R) as functions of t. They are real for real k. For Yukawa,
  i^{n−n′} (E|F) is real. (S|R) is complex for real and complex k.
- **Symmetries.** The index symmetry is E^m_{nn′} = (−1)^{n+n′} E^m_{n′n}: a table stores
  the triangle n′ ≥ n. The reflection rule is (E|F)(−t) = (−1)^{n+n′} (E|F)(t).
  Rectangular truncation p_src → p_tgt gives the corresponding block of the square
  table of degree max(p_src, p_tgt).
- **Stored normalisation**, with z = kt: (R|R) × e^{−|Im z|}, and (S|R) × e^{Im z} s^{n+n′}
  with s the scale of item 3 at both ends.
- **The generator** is the recursion (triangle, index symmetry) where Re(k) a ≤ 40, and
  the projection above that. The acceptance measure is the FMM-weighted error.

## 7. Signatures (open: CH0.6, CH0.9, CH4.1)

Proposed by proposal §5.4 and §6.7, and not yet tested:

- the (θ, φ) grid of a high-frequency level: Gauss–Legendre in cos θ and uniform in φ,
  N_l = (2P − 1)(4P − 3) samples with P = p_l + 1, and the φ count padded to a multiple of
  4 for the reflection classes;
- interleaved complex samples on the host, split planes on the device;
- the index permutations of the 34 classes;
- the transforms S → F and F → R;
- the diagonal transfer function with bandwidth 2P − 2.

S6 (CH0.6) decides the classes and the bound. S9 (CH0.9) decides the ring count
(2P − 1 or 2P). CH4.1 fixes the rest.

## 8. Precision and range (from S1–S3; open for D3)

f64 only until D3 is taken. The ranges tested so far:

- rotation blocks to n = 1000 (S1);
- coaxial coefficients to p = 371 and k a = 300, for real, complex and imaginary k (S2);
- radial functions to order 1482 and |z| from 10⁻¹² to 4550 (S3);
- the two-point check and the fixtures of CH1.4 (item 9).

CH1.2 restates these ranges for the crate's own tests, in the spirit of §3.9.

## 9. Inputs and outputs (proposed; the values fixed by CH1.4)

- **k** is given as (Re k, Im k) in f64. The builder classifies it as Yukawa, real or
  complex (item 1).
- **Charges** are given per vector as two planes (Re q, Im q), or Re q alone with the
  real-charge flag.
- **Outputs** are φ and ∇φ of item 1 (the factor 1/(4π) included) as split planes, per
  vector:
  - Re φ over the targets, then Im φ;
  - with gradients, the Re ∇φ triples (x, y, z) point-major, then the Im ∇φ triples.
- **Leaf chunks** (in-memory, as §3.13) are fixed by CH3.0 and are not versioned.
- **Fixtures.** `fmm-ref/fixtures/helmholtz_fmm3d.json` holds FMM3D's values of this φ
  and ∇φ (`h3ddirectcg`, FMM3D 2.1.0) for real, complex and imaginary k and for real
  and complex charges. Each value was checked against this item's kernel at 30 digits
  (`tools/fixtures/gen_helmholtz_fmm3d.py`).

## Versioning

CH1.1 adds this section to §3.10's list of sections that bump `CONVENTION_VERSION`, as
version 3. The leaf chunks of item 9 are excluded, as §3.13 is.
