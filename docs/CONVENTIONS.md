# Conventions

`CONVENTION_VERSION = 1` .

This file is the single source of truth for basis functions, phases, storage and
scaling in the `nd-fmm-*` crates. Code cites it as `CONVENTIONS §3.x`. Any change to
§3.1–§3.8 or §3.11 bumps `CONVENTION_VERSION`, which invalidates committed fixtures and
cached operator tables.

Summary of the choices: Legendre functions without the Condon–Shortley phase for
m ≥ 0, negative orders defined by a (−1)^m conjugate symmetry, Racah-type factorial
normalisation, real storage of m ≥ 0 coefficients, and scaling by the box half-width
so every operator table is level-independent.

While drafting, every identity and recursion below was checked numerically in double
precision against SciPy's Legendre functions (with its Condon–Shortley phase removed):
recursions to p = 12, the addition theorem to n = 6, the gradient ladder by finite
differences, and the rotation rescaling in §3.8 at n = 3 and 8. Errors were 1e-14 or
smaller. Phase 0 tasks T3 and T4 re-verify against independent high-precision fixtures.

## 3.1 Kernel

All operators expand 1/|x − y|. The factor 1/(4π) is applied once, by `nd-fmm-exec`
when producing output, never inside operators or tables.

## 3.2 Coordinates

x = r sinθ cosφ, y = r sinθ sinφ, z = r cosθ, with θ measured from +z and φ from +x
towards +y. Implementations should use the Cartesian recursions of §3.5 and never call
trigonometric functions.

## 3.3 Basis functions

Associated Legendre functions for m ≥ 0, **without** the Condon–Shortley phase:

```math
P_n^m(t) = (1-t^2)^{m/2}\, \frac{d^m}{dt^m} P_n(t), \qquad 0 \le m \le n
```

Complex regular and irregular solid harmonics for m ≥ 0:

```math
R_n^m(\mathbf{x}) = \frac{r^n}{(n+m)!}\, P_n^m(\cos\theta)\, e^{i m \phi}, \qquad
I_n^m(\mathbf{x}) = \frac{(n-m)!}{r^{n+1}}\, P_n^m(\cos\theta)\, e^{i m \phi}
```

Negative orders, for both families:

```math
R_n^{-m} = (-1)^m\, \overline{R_n^m}, \qquad I_n^{-m} = (-1)^m\, \overline{I_n^m}
```

Examples: R₁⁰ = z, R₁¹ = (x + iy)/2, R₁⁻¹ = −(x − iy)/2, R₂⁰ = (2z² − x² − y²)/4.

## 3.4 Identities the tests must confirm

Separation identity, for |y| < |x|:

```math
\frac{1}{|\mathbf{x}-\mathbf{y}|} = \sum_{n=0}^{\infty}\sum_{m=-n}^{n} \overline{R_n^m(\mathbf{y})}\, I_n^m(\mathbf{x})
```

Addition theorem for regular harmonics (the (−1)^m rule in §3.3 is what makes it hold;
without it the sign of cross terms flips):

```math
R_n^m(\mathbf{a}+\mathbf{b}) = \sum_{k=0}^{n} \sum_{l=-k}^{k} R_k^l(\mathbf{a})\, R_{n-k}^{m-l}(\mathbf{b}),
\qquad R_j^i := 0 \text{ for } |i| > j
```

Gradient ladder for regular harmonics:

```math
\partial_z R_n^m = R_{n-1}^m, \qquad
(\partial_x - i\partial_y) R_n^m = R_{n-1}^{m-1}, \qquad
(\partial_x + i\partial_y) R_n^m = -R_{n-1}^{m+1}
```

Harmonicity: ΔRₙᵐ = 0 and ΔIₙᵐ = 0 away from the origin. Gradients of Iₙᵐ are derived
in T4 and checked against the fixtures, not stated here.

## 3.5 Evaluation by Cartesian recursion

With ρ = x + iy and r² = x² + y² + z²:

```math
R_0^0 = 1,\quad R_m^m = \frac{\rho}{2m} R_{m-1}^{m-1},\quad R_{m+1}^m = z\, R_m^m,\quad
R_n^m = \frac{(2n-1)\, z\, R_{n-1}^m - r^2 R_{n-2}^m}{(n+m)(n-m)}
```

```math
I_0^0 = \frac{1}{r},\quad I_m^m = \frac{(2m-1)\,\rho}{r^2} I_{m-1}^{m-1},\quad
I_{m+1}^m = \frac{(2m+1)\, z}{r^2} I_m^m,\quad
I_n^m = \frac{(2n-1)\, z\, I_{n-1}^m - \big((n-1)^2 - m^2\big) I_{n-2}^m}{r^2}
```

Run the recursions on real and imaginary parts separately; no complex type is needed.

## 3.6 Real storage and index layout

- Length (p+1)²; index idx(n, m) = n² + n + m for −n ≤ m ≤ n.
- Slot m = 0 holds the real value; slot +m (m > 0) holds the real part of the order-m
  quantity; slot −m holds its imaginary part.
- The same layout stores basis values (R, I and their gradients) and expansion
  coefficients.
- For real charges, coefficients satisfy Mₙ⁻ᵐ = (−1)^m conj(Mₙᵐ), so the m ≥ 0 values
  carry everything.
- Evaluation from real storage doubles the m > 0 terms:

```math
\sum_{m=-n}^{n} M_n^m I_n^m = M_n^0 I_n^0 + 2\sum_{m=1}^{n}
\big(\operatorname{Re}M_n^m \operatorname{Re}I_n^m - \operatorname{Im}M_n^m \operatorname{Im}I_n^m\big)
```

## 3.7 Expansions and scaling

For a box with centre c and half-width r, and scaled coordinates u = (y − c)/r and
v = (x − c)/r, the stored (scaled) coefficients are:

```math
\tilde M_n^m = \sum_j q_j\, \overline{R_n^m(\mathbf{u}_j)}, \qquad
\tilde L_n^m = \sum_j q_j\, \overline{I_n^m(\mathbf{u}_j)}
```

```math
\phi(\mathbf{x}) \approx \frac{1}{r}\sum_{n\le p}\sum_m \tilde M_n^m\, I_n^m(\mathbf{v}) \ \text{(multipole)},
\qquad
\phi(\mathbf{x}) \approx \frac{1}{r}\sum_{n\le p}\sum_m \tilde L_n^m\, R_n^m(\mathbf{v}) \ \text{(local)}
```

Equivalently M̃ = M / rⁿ and L̃ = L · rⁿ⁺¹ in terms of unscaled coefficients. With this
choice, M2M, L2L and M2L tables are identical on every level with no extra factor.
(This refines Section 2.4 of docs/design/laplace-fmm-plan.md, which used L̃ = L · rⁿ and a 1/r
factor on M2L.)

## 3.8 Rotations

For a proper rotation Q (3 × 3, orthogonal, determinant +1), the degree-n block Dⁿ(Q)
acts on basis vectors in real storage:

```math
\mathbf{R}_n(Q\mathbf{x}) = D^n(Q)\, \mathbf{R}_n(\mathbf{x}), \qquad
D^n(Q_1 Q_2) = D^n(Q_1)\, D^n(Q_2)
```

- Irregular harmonics use S Dⁿ S⁻¹ with S = diag((n − |m|)! (n + |m|)!), because
  Iₙᵐ = (n − m)!(n + m)! Rₙᵐ / r²ⁿ⁺¹.
- With N = diag(√((n + |m|)!(n − |m|)!) · c_m), where c₀ = 1 and c_m = √2 otherwise,
  N Dⁿ N⁻¹ is orthogonal. T5 confirms this and uses it as a test.
- Euler angles follow the z-y-z convention: Q = R_z(α) R_y(β) R_z(γ).

## 3.9 Precision and range

- f64 is tested for p ≤ 30; f32 for p ≤ 8.
- Regular harmonics are only evaluated at |u| ≤ √3 (inside the box's sphere); irregular
  ones at |v| ≥ 2 in operator use. Tests cover these ranges plus a margin; behaviour
  outside them is not guaranteed.
- M2L (§3.11) evaluates irregular harmonics up to degree 2p at the scaled shift
  b = (c' − c)/r. On a uniform level, with the V-list offsets {−3..3}³ \ {−1..1}³ and
  shift 2r · offset, 4 ≤ |b| ≤ 6√3 ≈ 10.4. f64 is therefore also tested for irregular
  harmonics up to degree 40 (M2L up to p = 20) at |v| in [4, 11].

## 3.10 Versioning

`CONVENTION_VERSION = 1`. Any change to §3.1–§3.8 or §3.11 bumps it, which invalidates
committed fixtures and cached operator tables.

## 3.11 Translation operators

Every translation maps a scaled expansion (§3.7) in an input frame (c, r) to one in an
output frame (c', r'): a frame is a centre and the scaling radius r of §3.7. Harmonics
are those of §3.3, with negative orders from the (−1)ᵐ rule and Rⱼⁱ = Iⱼⁱ = 0 for
|i| > j; conj denotes complex conjugation. Each operator states its shift vector inside
its formula:

| Operator | Shift | Radius factor | Harmonics of the shift | Truncation |
| --- | --- | --- | --- | --- |
| M2M | b = (c − c')/r' | ρᵏ, ρ = r/r' | conj Rₙ, n ≤ p | exact |
| L2L | t = (c' − c)/r | σʲ⁺¹, σ = r'/r | Rₙ, n ≤ p | exact |
| M2L | b = (c' − c)/r | σʲ⁺¹, σ = r'/r | Iₙ, n ≤ 2p | bound below |

On a uniform octree level the radii are fixed: r' = 2r for M2M (child to parent),
r' = r/2 for L2L (parent to child) and r' = r for M2L, so every table depends only on
the offset (§3.7).

Two facts are used throughout. Homogeneity: Rₙᵐ(λx) = λⁿ Rₙᵐ(x) and
Iₙᵐ(λx) = λ⁻⁽ⁿ⁺¹⁾ Iₙᵐ(x) for λ > 0. Parity: Rₙᵐ(−x) = (−1)ⁿ Rₙᵐ(x) and
Iₙᵐ(−x) = (−1)ⁿ Iₙᵐ(x), since Rₙᵐ is a homogeneous polynomial of degree n and
Iₙᵐ = (n − m)!(n + m)! Rₙᵐ / r²ⁿ⁺¹ (§3.8).

### Irregular addition theorem

For |a| < |x|:

```math
I_n^m(\mathbf{x}+\mathbf{a}) = \sum_{k=0}^{\infty}\sum_{l=-k}^{k} (-1)^k\, \overline{R_k^l(\mathbf{a})}\, I_{n+k}^{m+l}(\mathbf{x})
= \sum_{j=0}^{\infty}\sum_{i=-j}^{j} (-1)^{j+i}\, R_j^i(\mathbf{a})\, I_{n+j}^{m-i}(\mathbf{x})
```

*Derivation.* The separation identity of §3.4 with y = −a, and parity, give

```math
\frac{1}{|\mathbf{x}+\mathbf{a}|} = \sum_{k,l} (-1)^k\, \overline{R_k^l(\mathbf{a})}\, I_k^l(\mathbf{x}), \qquad |\mathbf{a}| < |\mathbf{x}|
```

The irregular gradient ladder (derived in the doc comment of
`nd_fmm_math::harmonics::irregular_grad` from the separation identity and the ladder of
§3.4) reads −∂z Iₖˡ = Iₖ₊₁ˡ, (∂x − i∂y) Iₖˡ = Iₖ₊₁ˡ⁻¹ and −(∂x + i∂y) Iₖˡ = Iₖ₊₁ˡ⁺¹,
for all |l| ≤ k. So the constant-coefficient operator
Oₙᵐ = (−(∂x + i∂y))ᵐ (−∂z)ⁿ⁻ᵐ for m ≥ 0, and (∂x − i∂y)^|m| (−∂z)^(n−|m|) for m < 0,
maps every Iₖˡ to Iₖ₊ₙˡ⁺ᵐ; in particular Oₙᵐ I₀⁰ = Iₙᵐ, with I₀⁰(x) = 1/|x|. Apply Oₙᵐ
in x to both sides. On the left it commutes with the translation by a and gives
Iₙᵐ(x + a). On the right the series converges absolutely and uniformly, with all
derivatives, on |a| ≤ c|x| for c < 1, so it may be differentiated term by term; this
gives the first form. The second follows from conj(Rₖˡ) = (−1)ˡ Rₖ⁻ˡ (§3.3) and the
relabelling i = −l.

### M2M

Input M̃ in (c, r), output M̃' in (c', r'). With b = (c − c')/r' and ρ = r/r':

```math
\tilde M'^{\,i}_j = \sum_{k=0}^{j}\sum_{l=-k}^{k} \rho^k\, \tilde M_k^l\, \overline{R_{j-k}^{\,i-l}(\mathbf{b})},
\qquad 0 \le j \le p,\ |i| \le j
```

Terms with |i − l| > j − k vanish, so l runs from max(−k, i − j + k) to
min(k, i + j − k).

*Derivation.* For a source y, (y − c')/r' = ρu + b with u = (y − c)/r. The addition
theorem of §3.4 with homogeneity gives Rⱼⁱ(ρu + b) = Σₖ Σₗ ρᵏ Rₖˡ(u) Rⱼ₋ₖⁱ⁻ˡ(b);
conjugate it and sum qⱼ over the sources (§3.7).

Truncation and convergence: output degree j uses input degrees k ≤ j only, so the M2M
of a degree-p expansion is exact to degree p for any two frames. It is an identity of
polynomials and needs no convergence condition. The result is a multipole expansion
about c' of the same sources, valid outside a sphere about c' that contains them.

### L2L

Input L̃ in (c, r), output L̃' in (c', r'). With t = (c' − c)/r and σ = r'/r:

```math
\tilde L'^{\,i}_j = \sigma^{j+1} \sum_{n=j}^{p}\sum_{m=-n}^{n} \tilde L_n^m\, R_{n-j}^{\,m-i}(\mathbf{t}),
\qquad 0 \le j \le p,\ |i| \le j
```

Terms with |m − i| > n − j vanish.

*Derivation.* For a target x, v = (x − c)/r = σv' + t with v' = (x − c')/r'. The
addition theorem of §3.4 with homogeneity gives
Rₙᵐ(σv' + t) = Σⱼ Σᵢ σʲ Rⱼⁱ(v') Rₙ₋ⱼᵐ⁻ⁱ(t). Collect the coefficient of Rⱼⁱ(v') in
(1/r) Σ L̃ₙᵐ Rₙᵐ(v) and write 1/r = σ/r' (§3.7).

Truncation and convergence: the local expansion of degree p is a polynomial of degree p
in x, and L2L re-expands it exactly in the output frame, with no terms above degree p.
No convergence condition applies; in use the output sphere lies inside the input sphere.

### M2L

Input M̃ of degree p_in in (c, r), output L̃' of degree p_out in (c', r'). With
b = (c' − c)/r and σ = r'/r:

```math
\tilde L'^{\,i}_j = (-1)^{j+i}\, \sigma^{j+1} \sum_{n=0}^{p_\text{in}}\sum_{m=-n}^{n} \tilde M_n^m\, I_{n+j}^{\,m-i}(\mathbf{b}),
\qquad 0 \le j \le p_\text{out},\ |i| \le j
```

No term vanishes (|m − i| ≤ n + j always), and the irregular harmonics of b are needed
up to degree p_in + p_out, which is 2p for p_in = p_out = p (§3.9).

*Derivation.* For a target x, (x − c)/r = b + σv' with v' = (x − c')/r'. The irregular
addition theorem above, with x = b and a = σv', and homogeneity give
Iₙᵐ(b + σv') = Σⱼ Σᵢ (−1)ʲ⁺ⁱ σʲ Rⱼⁱ(v') Iₙ₊ⱼᵐ⁻ⁱ(b) for σ|v'| < |b|. Collect the
coefficient of Rⱼⁱ(v') in (1/r) Σ M̃ₙᵐ Iₙᵐ((x − c)/r) and write 1/r = σ/r'.

Convergence: with sources in the input sphere |y − c| ≤ √3 r and targets in the output
sphere |x − c'| ≤ √3 r', the double expansion converges when the output sphere lies
outside the input sphere, √3 (r + r') < |c' − c|, that is |b| > √3 (1 + σ). On a
uniform level (σ = 1, V list) 4 ≤ |b| ≤ 6√3, and 4 > 2√3 ≈ 3.46.

Truncation: for a unit charge at scaled position u = (y − c)/r with |u| < |b|, the
input coefficients M̃ₙᵐ = conj Rₙᵐ(u) are homogeneous of degree n in u, and the full
sum (p_in = ∞) equals the P2L coefficient in the output frame,
σʲ⁺¹ conj Iⱼⁱ(u − b), because (y − c')/r' = (u − b)/σ. So the terms of degree n are the
degree-n part of the Taylor series in u of a function harmonic for |u| < |b|. By the
Poisson integral over a sphere |w| = ρ, |u| < ρ < |b|, and |Pₙ| ≤ 1, that part is at
most (2n + 1)(|u|/ρ)ⁿ times the maximum of the function on the sphere, and
|Iⱼⁱ(w)| ≤ √((j − |i|)! (j + |i|)!) / |w|ʲ⁺¹, because
|Pₙᵐ(cos θ)| ≤ √((n + m)!/(n − m)!) (Unsöld's theorem for the orthonormal spherical
harmonics). Hence the error of truncating at p_in is at most

```math
\sigma^{j+1}\, \frac{\sqrt{(j-|i|)!\,(j+|i|)!}}{(|\mathbf{b}|-\rho)^{j+1}}
\sum_{n>p_\text{in}} (2n+1) \left(\frac{|\mathbf{u}|}{\rho}\right)^{n}
```

per unit charge and coefficient, for any ρ in (|u|, |b|); sum |qⱼ| times this over the
sources.

### Real storage

Every sum above is over products of two factors that each satisfy
X⁻ᵐ = (−1)ᵐ conj(Xᵐ): scaled coefficients (§3.6) and R, conj R and I of the shift
(§3.3; conj R inherits the rule from R). From real storage (§3.6):

- Only outputs with 0 ≤ i ≤ j are formed: Re into slot +i (slot 0 for i = 0), Im into
  slot −i. The outputs with i < 0 follow as (−1)ⁱ conj; each operator preserves the
  symmetry (substitute l → −l, or m → −m, in its sum).
- A factor of order m ≥ 0 is read as (slot m, slot −m), with imaginary part 0 for
  m = 0. A factor of order −m < 0 is read as (−1)ᵐ (slot m, −slot −m). Orders beyond
  the degree are 0.
- The complex products are formed in real arithmetic: Re = Re·Re − Im·Im,
  Im = Re·Im + Im·Re.

Written with the stored orders only, the order sums are

```math
\sum_{l=-k}^{k} A^l B^{\,i-l} = A^0 B^{\,i} + \sum_{l=1}^{k} \Big( A^l B^{\,i-l} + (-1)^l\, \overline{A^l}\, B^{\,i+l} \Big)
\quad \text{(M2M)}, \qquad
\sum_{m=-n}^{n} A^m B^{\,m-i} = A^0 B^{-i} + \sum_{m=1}^{n} \Big( A^m B^{\,m-i} + (-1)^m\, \overline{A^m}\, B^{-m-i} \Big)
\quad \text{(L2L, M2L)}
```

where A are the input coefficients of one degree (k for M2M, n for L2L and M2L) and
B^q the harmonics of the shift of the degree that the sum pairs with A:
conj Rⱼ₋ₖ^q(b) for M2M, Rₙ₋ⱼ^q(t) for L2L and Iₙ₊ⱼ^q(b) for M2L. B is read with the
rule above for any order q, negative ones included, and is 0 when |q| exceeds its
degree; so the second identity covers both L2L, where Rₙ₋ⱼ^q vanishes for
|q| > n − j, and M2L, where no term vanishes.

### Coaxial translations

For a shift along the z-axis, c' − c = d e_z (along +z for d > 0), every shift vector
above lies on the z-axis. There Pₖˡ(±1) = 0 for l ≠ 0 and Pₖ(±1) = (±1)ᵏ, so only
order 0 survives:

```math
R_k^0(s\,\mathbf{e}_z) = \frac{s^k}{k!}\ \ (s \in \mathbb{R}), \qquad
I_k^0(s\,\mathbf{e}_z) = \frac{k!}{s^{k+1}},\quad I_k^0(-s\,\mathbf{e}_z) = (-1)^k \frac{k!}{s^{k+1}}\ \ (s > 0), \qquad
R_k^l = I_k^l = 0 \text{ on the axis for } l \ne 0
```

Each translation then keeps the order fixed, and its factors are real:

```math
\text{M2M:}\ \ \tilde M'^{\,i}_j = \sum_{k=|i|}^{j} \rho^k\, \tilde M_k^i\, \frac{(-d/r')^{j-k}}{(j-k)!}, \qquad
\text{L2L:}\ \ \tilde L'^{\,i}_j = \sigma^{j+1} \sum_{n=j}^{p} \tilde L_n^i\, \frac{(d/r)^{n-j}}{(n-j)!}
```

```math
\text{M2L:}\ \ \tilde L'^{\,i}_j = (-1)^{j+i}\, \sigma^{j+1} \sum_{n=|i|}^{p_\text{in}} \tilde M_n^i\, (n+j)!\, (\operatorname{sgn} d)^{n+j} \Big(\frac{r}{|d|}\Big)^{n+j+1}
```

All three hold for either sign of d (for M2L, |d| > √3 (r + r')). The M2L factor is
Iₙ₊ⱼ⁰((d/r) e_z) by the axis values above; for d > 0 it is (n + j)! (r/d)ⁿ⁺ʲ⁺¹. In
real storage slots +i and −i are transformed by the same real factors.

### Rotation of coefficients

In real storage conjugation is K = diag(+1 for m ≥ 0, −1 for m < 0) (§3.6). Multipole
coefficients are sums of qⱼ K Rₙ(uⱼ), and local coefficients of qⱼ K Iₙ(uⱼ). With
§3.8, for a proper rotation Q about the frame centre (sources y ↦ c + Q(y − c)) the
coefficients of degree n become

```math
\tilde{\mathbf{M}}_n \mapsto K\, D^n(Q)\, K\, \tilde{\mathbf{M}}_n, \qquad
\tilde{\mathbf{L}}_n \mapsto K\, S\, D^n(Q)\, S^{-1} K\, \tilde{\mathbf{L}}_n
```

and the rotated expansion at c + Q(x − c) equals the original one at x. K and S are
diagonal and commute.

Rotation-based translation: choose Q with Q(c' − c) = |c' − c| e_z, for example
Q = R_y(−θ) R_z(−φ), the z-y-z Euler angles (0, −θ, −φ) of §3.8, with θ and φ the angles
of c' − c (§3.2). Rotate the input with Q, apply the coaxial form with d = |c' − c|, and
rotate the output back with Qᵀ (Dⁿ(Qᵀ) = Dⁿ(Q)⁻¹), each with the rule of its kind:
multipole to multipole for M2M, local to local for L2L, multipole in and local out for
M2L. This equals the general form.

### L2P and M2P

With v = (x − c)/r:

```math
\phi(\mathbf{x}) = \frac{1}{r}\sum_{n\le p}\sum_m \tilde L_n^m R_n^m(\mathbf{v}), \qquad
\nabla\phi(\mathbf{x}) = \frac{1}{r^2}\sum_{n\le p}\sum_m \tilde L_n^m\, (\nabla R_n^m)(\mathbf{v})
```

```math
\phi(\mathbf{x}) = \frac{1}{r}\sum_{n\le p}\sum_m \tilde M_n^m I_n^m(\mathbf{v}), \qquad
\nabla\phi(\mathbf{x}) = \frac{1}{r^2}\sum_{n\le p}\sum_m \tilde M_n^m\, (\nabla I_n^m)(\mathbf{v})
```

The 1/r is the scaling of §3.7; the gradient's further 1/r is the chain rule for
v = (x − c)/r. By the ladders (§3.4 and `irregular_grad`):

```math
\partial_z R_n^m = R_{n-1}^m,\quad \partial_x R_n^m = \tfrac12\big(R_{n-1}^{m-1} - R_{n-1}^{m+1}\big),\quad
\partial_y R_n^m = \tfrac{i}{2}\big(R_{n-1}^{m-1} + R_{n-1}^{m+1}\big)
```

```math
\partial_z I_n^m = -I_{n+1}^m,\quad \partial_x I_n^m = \tfrac12\big(I_{n+1}^{m-1} - I_{n+1}^{m+1}\big),\quad
\partial_y I_n^m = \tfrac{i}{2}\big(I_{n+1}^{m-1} + I_{n+1}^{m+1}\big)
```

So the L2P gradient of degree p needs only degrees below p, and the M2P gradient needs
Iₚ₊₁. Each Cartesian component of ∇Xₙᵐ satisfies the (−1)ᵐ rule, so each sum over m is
evaluated from real storage by the doubling rule of §3.6. No operator applies 1/(4π)
(§3.1).

### Verification

`tools/fixtures/check_translations.py` checks, in mpmath at 40 digits, for seeded random
frames and charges:

- the complex forms: M2M after P2M against P2M at the output frame; L2L against the
  potential of the input expansion inside the output sphere; M2L with p_in = 60,
  p_out = 8 against P2L within the truncation bound;
- the real-storage forms: M2M, L2L and M2L evaluated from real storage through the two
  order-sum identities (stored orders m ≥ 0 of the input only, shift harmonics read by
  the rule) against the complex forms;
- the coaxial forms against the general ones, M2M, L2L and M2L each for d > 0 and
  d < 0;
- the rotation rule, with Dⁿ fitted from its definition in §3.8, the invariance of the
  potential under it, and rotate–coaxial–rotate back against the general forms;
- the L2P and M2P gradients against central differences of the potential.

The irregular addition theorem and the truncation bound are checked only through M2L;
the axis values Rₖ⁰(s e_z), Iₖ⁰(±s e_z) only through the coaxial forms.
