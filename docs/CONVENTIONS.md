# Conventions

`CONVENTION_VERSION = 1` .

This file is the single source of truth for basis functions, phases, storage and
scaling in the `nd-fmm-*` crates. Code cites it as `CONVENTIONS §3.x`. Any change to
§3.1–§3.8, §3.11 or §3.12 bumps `CONVENTION_VERSION`, which invalidates committed
fixtures and cached operator tables. §3.13 describes in-memory data only; a change to
it does not bump the version (§3.10).

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

`CONVENTION_VERSION = 1`. Any change to §3.1–§3.8, §3.11 or §3.12 bumps it, which
invalidates committed fixtures and cached operator tables. §3.12 is included because
cached tables depend on its geometry, layout and class rule.

§3.13 is not included, and a change to it does not bump `CONVENTION_VERSION`: it
describes in-memory data only (chunk layouts, leaf-scaled values, relative frames), and
no committed fixture and no cached table depends on it. A change to §3.13 changes the
code that implements it in the same pull request.

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

## 3.12 Box geometry and operator tables

The precomputed tables of `nd-fmm-tables` encode the geometry of an octree level and
are stored in a fixed layout. This section states both, and the action of the cube
symmetry group on coefficients that the symmetry-reduced M2L tables use. It builds on
real storage (§3.6), the scaling (§3.7), rotations (§3.8) and the translations of §3.11.
The octant and offset orders restate those of `nd_octree::morton` and
`nd_fmm_plan::interaction_manager`, which crates without MPI cannot import; code that
uses both must agree with them.

### Domain and levels

- The domain is a cube with lower corner a and side w. Level l, 0 ≤ l ≤ 16, has 2^l
  boxes per axis.
- The box on level l with index (i, j, k) from `morton::decode`, each in 0..2^l, has
  centre and half-width

```math
\mathbf{c} = \mathbf{a} + \big((i, j, k) + \tfrac12\big)\, \frac{w}{2^l}, \qquad
r_l = \frac{w}{2^{l+1}}
```

- r_l is the scaling radius of §3.7 on level l. c and r_l are the midpoint and the
  half-side of `morton::physical_box`, which spans [a + i w / 2^l, a + (i + 1) w / 2^l]
  along x, and likewise along y and z.

### Child octants

- The child index is o = 4x + 2y + z with x, y, z ∈ {0, 1}, as `morton::child_index`
  returns it and `morton::children` orders the children. x, y and z are the lowest bits
  of the child's index: the children of box (i, j, k) have indices
  (2i + x, 2j + y, 2k + z).
- With the sign vector s_o = (2x − 1, 2y − 1, 2z − 1), the child centre is

```math
\mathbf{c}_\text{child} = \mathbf{c}_\text{parent} + r_\text{child}\, \mathbf{s}_o, \qquad
r_\text{child} = r_\text{parent} / 2
```

- So o = 0 is the child at the lower corner, s₀ = (−1, −1, −1), and o = 4 has
  s₄ = (+1, −1, −1).

### V-list offsets

- The offset of a V-list pair is d = index(target) − index(source), in the index units
  of their common level, as `InteractionManager::v_list_by_direction` computes it. It
  lies in 𝒟 = {−3..3}³ \ {−1..1}³, which has 7³ − 3³ = 316 elements.
- The offsets are ordered lexicographically in (d_x, d_y, d_z), as `V_LIST_DIRECTIONS`
  is. The position of d in this order is its table index:

```math
\operatorname{index}(\mathbf{d}) = 49(d_x + 3) + 7(d_y + 3) + (d_z + 3) - 9\,\kappa(d_x)
- [\,|d_x| \le 1\,]\,\big(3\,\kappa(d_y) + [\,|d_y| \le 1\,]\,\kappa(d_z)\big)
```

  with κ(t) = min(max(t + 1, 0), 3), the number of t′ ∈ {−1, 0, 1} with t′ < t, and
  [·] = 1 if the condition holds and 0 otherwise. The first three terms are the position
  in the lexicographic order of {−3..3}³; the others subtract the points of {−1..1}³
  that precede d. So index(−3, −3, −3) = 0 and index(3, 3, 3) = 315.
- The shift is c_target − c_source = 2 r_l d. In the terms of §3.11, the M2L from
  (c, r_l) to (c + 2 r_l d, r_l) has b = 2d and σ = 1, and 4 ≤ |b| ≤ 6√3 (§3.9).

### Canonical frames and level independence

Each table is built at these frames, for child index o and offset d:

| Table | Input frame | Output frame | §3.11 parameters |
| --- | --- | --- | --- |
| M2M(o) | child (½ s_o, ½) | parent (0, 1) | b = ½ s_o, ρ = ½ |
| L2L(o) | parent (0, 1) | child (½ s_o, ½) | t = ½ s_o, σ = ½ |
| M2L(d) | source (0, 1) | target (2d, 1) | b = 2d, σ = 1 |

The operators of §3.11 depend on their frames only through these parameters. On level
l of any cubic domain the child (c + r_(l+1) s_o, r_(l+1)) and its parent (c, r_l), and
the V-list pair (c, r_l) and (c + 2 r_l d, r_l), give the same parameters. So a table
built at the canonical frames is the operator on every level of every cubic domain,
with no further factor (§3.7).

A centre difference c′ − c formed in floating point carries a relative error of about
ε |c| / r_l, which grows with the level; the tables are exact in their geometry. Code
therefore looks tables up by integer offset d and child index o, never by a
floating-point shift.

### Matrix layout

- A table has one degree p for input and output. Each operator is a real
  (p + 1)² × (p + 1)² matrix A with output = A · input. Rows index output slots and
  columns input slots, both in the real storage of §3.6 (index n² + n + m).
- Storage is column-major: entry (i, k) sits at i + k (p + 1)².
- The matrices of one family are contiguous, matrix t at offset t (p + 1)⁴: M2M and L2L
  with t = o in child-index order, M2L with t = index(d).
- Applying a table accumulates: output += A · input, as every operator does.
- The entries do not depend on p, so the table at p is the leading
  (p + 1)² × (p + 1)² block of the one at any larger p (not a contiguous range in
  column-major storage).
- Tables are computed in f64. An f32 table is the f64 table rounded entry by entry.

### The cube symmetry group

- O_h consists of the 48 signed permutation matrices P, (P v)_a = s_a v_π(a), for a
  permutation π of the axes (0, 1, 2) = (x, y, z) and signs s = (s_x, s_y, s_z) ∈ {±1}³.
  det P = sgn(π) s_x s_y s_z; 24 elements are proper (the rotations of the cube) and 24
  improper, among them −I.
- Enumeration order: element g, 0 ≤ g < 48, is

```math
g = 8k + 4\,[s_x < 0] + 2\,[s_y < 0] + [s_z < 0]
```

  where k is the position of (π(0), π(1), π(2)) in the lexicographic list
  (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0). So g = 0 is I, g = 7
  is −I, and for g < 8 the element is diagonal.
- Offsets: P permutes and negates the components of d, so P maps 𝒟 onto itself.
- Octants: P s_o = s_o′ for a child index o′, so P permutes the eight octants. They form
  one orbit, and element o, P = diag(1 − 2x, 1 − 2y, 1 − 2z) = −diag(s_o), is the first
  element in the enumeration order with P s₀ = s_o.

### Coefficients under improper P

For proper P, Dⁿ(P) is the block of §3.8. For improper P, −P is proper, and parity
(§3.11) gives Rₙ(Px) = (−1)ⁿ Rₙ(−Px) = (−1)ⁿ Dⁿ(−P) Rₙ(x). So define

```math
D^n(P) = (-1)^n\, D^n(-P) \qquad \text{for improper } P
```

Then, for all P in O_h,

- Rₙ(Px) = Dⁿ(P) Rₙ(x), and Dⁿ(P₁ P₂) = Dⁿ(P₁) Dⁿ(P₂): the values Rₙ(x) at all x span
  ℝ²ⁿ⁺¹, so the relation determines Dⁿ(P), and Rₙ(P₁ P₂ x) = Dⁿ(P₁) Dⁿ(P₂) Rₙ(x).
  Hence Dⁿ(Pᵀ) = Dⁿ(P)⁻¹ and Dⁿ(−I) = (−1)ⁿ I;
- Iₙ(Px) = S Dⁿ(P) S⁻¹ Iₙ(x), with S of §3.8, because |Px| = |x|.

Per degree n let

```math
T_M(P) = K\, D^n(P)\, K, \qquad T_L(P) = K\, S\, D^n(P)\, S^{-1} K
```

with K of §3.11. The rule of §3.11 ("Rotation of coefficients") uses only
Rₙ(Qx) = Dⁿ Rₙ(x), so it holds for every P in O_h: for sources y ↦ c + P(y − c) the
coefficients of degree n become T_M(P) M̃ₙ and T_L(P) L̃ₙ. T_M and T_L are
homomorphisms, and T(P)⁻¹ = T(Pᵀ) for both. Applied to a whole expansion, T(P) acts
degree by degree. Coefficients and values depend on the frame (c, r) only through the
scaled points u = (y − c)/r and v = (x − c)/r (§3.7), so the rule holds equally for
sources Py in the frame (Pc, r). Likewise, the expansion with coefficients T(P) C in the
frame (Pc, r), evaluated at Px, equals the one with coefficients C in (c, r) at x, for
either kind: for a unit source the degree-n terms are those of the separation identity
(§3.4), which depend on u and v only through |u|, |v| and u · v, and the coefficients
of point sources span every degree.

### Operator identities

With the tables at the canonical frames, M2L(d) for an offset d and M2M(s), L2L(s) for
the octant with sign vector s = s_o, for every P in O_h:

```math
\mathrm{M2L}(P\mathbf{d}) = T_L(P)\, \mathrm{M2L}(\mathbf{d})\, T_M(P)^{-1}, \qquad
\mathrm{M2M}(P\mathbf{s}) = T_M(P)\, \mathrm{M2M}(\mathbf{s})\, T_M(P)^{-1}, \qquad
\mathrm{L2L}(P\mathbf{s}) = T_L(P)\, \mathrm{L2L}(\mathbf{s})\, T_L(P)^{-1}
```

with T(P)⁻¹ = T(Pᵀ). For inversion, P = −I, T_M(−I) = T_L(−I) = diag((−1)ⁿ), so

```math
\mathrm{M2L}(-\mathbf{d}) = \operatorname{diag}\big((-1)^j\big)\, \mathrm{M2L}(\mathbf{d})\, \operatorname{diag}\big((-1)^n\big)
```

with j the output and n the input degree; this is also the parity
Iₙ₊ⱼ(−b) = (−1)ⁿ⁺ʲ Iₙ₊ⱼ(b) in the formula of §3.11.

*Derivation.* M2M: for sources y, the child coefficients M̃ = Σ q K Rₙ(2y − s) map to
the parent coefficients Σ q K Rₙ(y) exactly (§3.11). Replace every y by Py: the child
of sign vector Ps then has coefficients Σ q K Rₙ(P(2y − s)) = T_M(P) M̃, and the parent
T_M(P) times the parent coefficients. So M2M(Ps) T_M(P) = T_M(P) M2M(s) on all
coefficient vectors of point sources, which span every degree. L2L: the child expansion
L2L(s) L̃ is the parent expansion re-expanded exactly (§3.11). Transform both with P as
in the previous section: T_L(P) L̃ at the parent and T_L(P) L2L(s) L̃ at the child
(½ Ps, ½) equal, at Px, the parent and child expansions at x. They therefore agree with
each other, so T_L(P) L2L(s) L̃ = L2L(Ps) T_L(P) L̃. M2L: for a unit source at u,
|u| ≤ √3, the M2L of its full multipole series equals its P2L coefficients
K Iⱼ(u − b) (§3.11, truncation), and the terms of input degree n are the part of
degree n in u. Replacing u by Pu and b by Pb maps K Iⱼ(u − b) to T_L(P) K Iⱼ(u − b)
and the degree-n multipole coefficients to T_M(P) K Rₙ(u), so
M2L(Pd)ⱼₙ T_M(P) = T_L(P) M2L(d)ⱼₙ for every block of output degree j and input
degree n. The identity is therefore exact for the truncated tables of any degree p.

### Symmetry classes

- The 316 offsets form 16 orbits under O_h, the classes. Each class contains exactly
  one offset with 0 ≤ d_x ≤ d_y ≤ d_z, its representative (the sorted absolute values
  of any member).
- Classes are numbered by the lexicographic order of their representatives:

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Representative | (0,0,2) | (0,0,3) | (0,1,2) | (0,1,3) | (0,2,2) | (0,2,3) | (0,3,3) | (1,1,2) |
| Size | 6 | 6 | 24 | 24 | 12 | 24 | 12 | 24 |

| Class | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Representative | (1,1,3) | (1,2,2) | (1,2,3) | (1,3,3) | (2,2,2) | (2,2,3) | (2,3,3) | (3,3,3) |
| Size | 24 | 24 | 48 | 24 | 8 | 24 | 24 | 8 |

- The group element of an offset d is the first element P in the enumeration order
  above with P · representative = d. A representative with a zero or repeated component
  has a non-trivial stabiliser, so several elements qualify; the rule picks one, which
  makes the class form, and the cached class tables, reproducible. The class form is
  then M2L(d) = T_L(P) M2L(representative) T_M(Pᵀ).
- Under the 16 elements that map the z-axis to itself or to its negative (π(2) = 2,
  that is g ∈ {0, …, 7} ∪ {16, …, 23}) the offsets form 34 orbits.

### The z-axis elements

For the 16 elements with P e_z = ±e_z, per degree:

- T_L(P) = T_M(P), and both are signed permutation matrices that map slot 0 to ± itself
  and each slot pair (+m, −m) to itself, with entries 0 and ±1.
- For g ∈ {0, …, 7}, P = diag(±1, ±1, ±1), they are diagonal.
- For g ∈ {16, …, 23}, which exchange x and y, they exchange the two slots of every pair
  with odd m and are diagonal on the pairs with even m.

The reason: on ρ = x + iy these elements act as ρ ↦ ±ρ or ±conj ρ (g < 8) and as
ρ ↦ ±iρ or ±i conj ρ (16 ≤ g ≤ 23), and Rₙᵐ is ρᵐ times a real polynomial in z and
|ρ|² (§3.3), which z ↦ −z multiplies by (−1)ⁿ⁺ᵐ. S is constant on each slot pair, so it
commutes with Dⁿ(P). So these transforms need no arithmetic beyond sign changes and slot
exchanges.

### Verification

`tools/fixtures/check_symmetry.py` checks, in mpmath at 40 digits, for n ≤ 8 and seeded
random points and coefficients:

- the 316 offsets, their order and the closed-form index; the box centres, the child
  index from the bits of a Morton key, the child centres and the shift 2 r_l d, in exact
  rationals;
- the enumeration of O_h, the 16 classes (34 under the z-axis elements), their
  representatives and the group-element rule, and the action on the octants;
- Rₙ(Px) = Dⁿ(P) Rₙ(x) and the irregular counterpart for all 48 P, with Dⁿ(P) fitted
  for proper P and taken as (−1)ⁿ Dⁿ(−P) for improper P; the homomorphism on sampled
  pairs; T(P) T(Pᵀ) = I;
- the operator identities through the general forms of §3.11: M2L for all 316 offsets
  from their representatives, M2M and L2L for all 48 P from octant 0, and inversion
  for all 316 offsets;
- the structure of T_M(P) and T_L(P) for the z-axis elements.

## 3.13 Leaf data and relative geometry

The per-leaf chunks of `nd-fmm-plan` hold source data, target input and target output
(docs/phase3/README.md, requirement 4). This section fixes their layout for the Laplace
kernel and how one box sees the geometry of another. It builds on the kernel (§3.1),
real storage (§3.6), the scaling (§3.7), the L2P and M2P forms of §3.11 and the domain,
levels and tables of §3.12. It is arranged so that no operator ever forms a
floating-point shift: the domain enters only when points are loaded and when output
is produced, and every frame an operator uses is an exact dyadic rational computed
from integer key indices.

Notation: ε₆₄ = 2⁻⁵³ is the unit roundoff of f64 and ε_T that of the storage type T
(2⁻²⁴ for f32); fl(·) is one correctly rounded f64 operation.

### Integer centres

- For a key on level l with index i from `morton::decode` and a reference level L,
  l ≤ L ≤ 16, the integer centre is, componentwise,

```math
\mathbf{C}_L = (2\mathbf{i} + 1)\, 2^{L-l}, \qquad
\mathbf{c} = \mathbf{a} + \mathbf{C}_L\, \frac{w}{2^{L+1}}
```

- Each component of C_L is an odd multiple of 2^(L−l) in [1, 2^(L+1) − 1].
- Since (2i + 1) 2^(L−l) / 2^(L+1) = (i + ½) / 2^l, c is the centre of §3.12, the
  midpoint of `morton::physical_box`, for every L ≥ l. C_L is that centre in units of
  the half-width r_L, measured from a.

### Relative frames

For boxes s and t on levels l_s and l_t, with L = max(l_s, l_t), the frame of s seen
from t is

```math
\hat{\mathbf{c}}(s|t) = \frac{\mathbf{c}_s - \mathbf{c}_t}{r_t} = \big(\mathbf{C}_L(s) - \mathbf{C}_L(t)\big)\, 2^{\,l_t - L},
\qquad
\hat r(s|t) = \frac{r_s}{r_t} = 2^{\,l_t - l_s}
```

because c_s − c_t = (C_L(s) − C_L(t)) w / 2^(L+1) and r_t = w / 2^(l_t+1). Neither a
nor w remains: relative frames do not depend on the domain.

- **Exact.** N = C_L(s) − C_L(t) is an integer with |N| ≤ 2^(L+1) − 2 ≤ 2¹⁷ − 2 = 131070
  for keys on levels 0–16; boxes in opposite corners of level 16 attain it. So
  ĉ(s|t) = N · 2^(l_t − L) with 0 ≤ L − l_t ≤ 16, and r̂(s|t) = 2^k with |k| ≤ 16.
  Both have at most 17 significant bits and exponents far inside the normal range, so
  they are exact in f32 (24-bit significand) and f64. Code forms N in integer
  arithmetic, converts it to T (exact, |N| < 2²⁴) and multiplies by the power of two
  (exact); no step rounds.
- **Reversal.** ĉ(t|s) = −ĉ(s|t) / r̂(s|t) and r̂(t|s) = 1 / r̂(s|t), both exact;
  ĉ(t|t) = 0 and r̂(t|t) = 1.
- **The parameters of §3.11.** For an input expansion in box s and an output expansion
  in box t: M2M has b = ĉ(s|t) and ρ = r̂(s|t); L2L has the shift ĉ(t|s) (the t of
  §3.11) and σ = r̂(t|s); M2L has b = ĉ(t|s) and σ = r̂(t|s). A child of octant o seen from its parent has
  (ĉ, r̂) = (½ s_o, ½), and the target of a V-list pair with offset d, seen from its
  source, has (2d, 1). These are the canonical frames of §3.12, so each table is its
  operator at the relative frames, exactly, on every level.

### Leaf-scaled coordinates

A point x in leaf b, with level l and index i, of the domain with lower corner a and
side w (§3.12) is stored, per component, as

```math
\mathbf{u} = (\mathbf{x} - \mathbf{a})\, \frac{2^{l+1}}{w} - (2\mathbf{i} + 1)
```

- **Evaluation.** In f64, in this order: d = fl(x − a); then the scaling
  fl(d · 2^(l+1) / w), which rounds once because the product with 2^(l+1) is exact
  (fl(d / r_l), with r_l = w · 2^−(l+1) exact, is the same double); then
  fl(· − (2i + 1)), where 2i + 1 < 2¹⁷ is exact; then rounding to T. Multiplying by a
  precomputed fl(2^(l+1) / w) rounds once more and is not this formula.
- **Exact value.** In exact arithmetic u = (x − c_b) / r_b, the scaled coordinates of
  §3.7 in the frame of b.
- a is the lower corner of the `PhysicalBox` the tree was built with, and w its side
  (one value for all three axes).

**Error bound.** Let u be the exact value of the formula for the given doubles x, a and
w, and ũ the stored one. Per component, to first order in ε₆₄ and ε_T,

```math
|\tilde u - u| \le
\underbrace{\varepsilon_{64} \frac{|x - a|}{r_l}}_{x - a}
+ \underbrace{\varepsilon_{64} \frac{|x - a|}{r_l}}_{\text{scaling}}
+ \underbrace{\varepsilon_{64}\, |u|}_{-\,(2i+1)}
+ \underbrace{\varepsilon_T\, |u|}_{\text{cast}}
```

- *The rounding of x − a* is at most ε₆₄ |x − a| / r_l ≤ ε₆₄ (|x| + |a|) / r_l, and zero
  when x − a is exact, as for a domain far from the origin (Sterbenz). It stays within
  the precision of the input itself: x and a are doubles, which locate a point only to
  about ε₆₄ (|x| + |a|), that is ε₆₄ (|x| + |a|) / r_l in units of the leaf. No layout of
  the leaf data recovers more. Interleaved absolute coordinates stored in T lose
  ε_T |x| / r_l instead: for |x| ≈ w, 2¹⁷ ε₃₂ ≈ 7.8e-3 at level 16 in f32.
- *The scaling* is at most ε₆₄ |x − a| / r_l. For points of the domain |x − a| ≤ w up
  to rounding, so the two terms together are at most 2 · 2^(l+1) ε₆₄ = 2^(l−51),
  2.9e-11 at level 16, wherever the domain lies.
- *The subtraction of 2i + 1* is at most ε₆₄ |u|, and exact (Sterbenz) in every
  component with i ≥ 1 or u ≥ −½.
- *The cast* is at most ε_T |u|, and absent for T = f64.

So in f64 the stored u is within 2^(l−51) + ε₆₄ of the exact one, and in f32 the cast
dominates on every level.

**Containment.** Let w_k = fl(max_k − min_k) be the side of the `PhysicalBox` along axis
k, which `points_to_morton` divides by, and a_k = min_k. For every point that
`points_to_morton` puts into leaf b (from its key on the leaf's level or any deeper
one), component k of the exact u lies in [−1 − β_k, 1 + β_k], and that of the stored
ũ in [−1 − β_k − E, 1 + β_k + E], with E the error bound above and, to first order,

```math
\beta_k = \frac{|w_k - w| + 2\varepsilon_{64}\, w}{r_l}
```

*Derivation.* `points_to_morton` computes ρ = fl(fl(x − a_k) / w_k). Rounding is
monotone and a_k < x < max_k, so 0 ≤ ρ ≤ 1. The index on level l is
i = min(⌊2^l ρ⌋, 2^l − 1) (a deeper key's index shifted right gives the same), so
i ≤ 2^l ρ ≤ i + 1. Exactly, 2^l (x − a) / w = 2^l ρ (w_k / w)(1 + δ) with
|δ| ≤ 2ε₆₄ to first order, which differs from 2^l ρ by at most
2^l (|w_k − w| / w + 2ε₆₄). And u = 2 (2^l (x − a) / w − i) − 1.

- For w_k = w, β = 2^(l−51), 2.9e-11 at level 16.
- `compute_global_bounding_box` forms each side as the difference of two rounded corner
  coordinates, so its sides agree only to |w_k − w| = O(ε₆₄ (|a| + w)) (docs/phase3/README.md,
  "Domain"). Then β = O(ε₆₄ (|a| + w) / r_l): the order of the input precision above,
  and far inside the tested range |u| ≤ √3 of §3.9.

### Source chunks

A leaf with n ≥ 0 source points holds 4n values of T, 4 per point:

```text
u_0 u_1 … u_(n−1) q_0 q_1 … q_(n−1)        u_j = (x, y, z) of point j
```

- The first 3n values are the leaf-scaled coordinate triples, point-major, x before y
  before z, so they read as n values of `[T; 3]` without copying.
- The last n values are the charges in the same point order, rounded to T and not
  scaled: the kernel's scale enters only through r_t at output.
- These chunks are exchanged for ghost leaves (U and X lists). u is relative to the
  leaf's own key, so a received chunk needs no transformation.

### Target input and output

A leaf with n ≥ 0 target points has two chunks, in the same point order:

- **Target input**, 3n values: the leaf-scaled positions u₀, …, uₙ₋₁, point-major,
  from the same formula as sources. Local only, never exchanged.
- **Target output**, n values (potentials only) or 4n values (with gradients): first
  φ̂₀, …, φ̂ₙ₋₁, then, with gradients, the triples ĝ₀, …, ĝₙ₋₁, point-major. Which of
  the two is fixed for an FMM. Local only.

For a target x in leaf t with half-width r_t,

```math
\hat\phi(\mathbf{x}) = r_t \sum_j \frac{q_j}{|\mathbf{x} - \mathbf{y}_j|}, \qquad
\hat{\mathbf{g}}(\mathbf{x}) = r_t^2\, \nabla_{\mathbf{x}} \sum_j \frac{q_j}{|\mathbf{x} - \mathbf{y}_j|}
```

the potential and gradient of the 1/|x − y| kernel of §3.1 in units of the target
leaf. Every operator below accumulates (+=) into them with no further factor.

### Operators in scaled coordinates

A `nd_fmm_ref::Frame` (C, R) maps a point p to (p − C) / R (`Frame::scaled`); `leaf::l2p`
and `leaf::m2p` multiply by 1/R and, for the gradient, 1/R² (§3.11, "L2P and M2P").
Each operator passes:

| Operator | Call | Frame | Applied to | Result, with no factor |
| --- | --- | --- | --- | --- |
| P2M at leaf s | `leaf::p2m` | ((0, 0, 0), 1) | u_s | M̃ of s in (c_s, r_s) |
| P2L from leaf s into box t | `leaf::p2l` | (ĉ(t\|s), r̂(t\|s)) | u_s | L̃ of t in (c_t, r_t) |
| L2P at leaf t | `leaf::l2p` | ((0, 0, 0), 1) | u_t | φ̂, ĝ |
| M2P from box s at leaf t | `leaf::m2p` | (ĉ(s\|t), r̂(s\|t)) | u_t | φ̂, ĝ |
| P2P from leaf s to leaf t | `p2p::p2p` | none | ŷ = ĉ(s\|t) + r̂(s\|t) u_s and u_t | φ̂, ĝ |
| M2M, L2L, M2L | tables (§3.12) | by child index o and offset index (d) | coefficients | coefficients in the output frame |

*Derivations.* With y = c_s + r_s u_s for a source in leaf s and x = c_t + r_t u_t for a
target in leaf t:

- **P2M.** §3.7 defines the multipole of leaf s in its frame (c_s, r_s) as
  M̃ = Σ q conj Rₙᵐ((y − c_s) / r_s) = Σ q conj Rₙᵐ(u_s). The unit frame maps u_s to
  itself exactly, so `p2m` returns M̃.
- **P2L.** §3.7 defines the local expansion of box t in its frame as
  L̃ = Σ q conj Iₙᵐ((y − c_t) / r_t). Here (y − c_t) / r_t = ĉ(s|t) + r̂(s|t) u_s
  = (u_s − ĉ(t|s)) / r̂(t|s) by the reversal rule, which the frame (ĉ(t|s), r̂(t|s))
  computes. `p2l` uses its frame only through `Frame::scaled`, so it returns L̃ with no
  factor.
- **L2P.** By §3.11, φ(x) = (1/r_t) Σ L̃ₙᵐ Rₙᵐ(v) and ∇φ(x) = (1/r_t²) Σ L̃ₙᵐ (∇Rₙᵐ)(v)
  with v = (x − c_t) / r_t = u_t. With the unit frame, `l2p` returns
  Σ L̃ₙᵐ Rₙᵐ(u_t) = r_t φ = φ̂ and Σ L̃ₙᵐ (∇Rₙᵐ)(u_t) = r_t² ∇φ = ĝ.
- **M2P.** By §3.11, in the frame (c_s, r_s) of the multipole,
  φ(x) = (1/r_s) Σ M̃ₙᵐ Iₙᵐ(v) and ∇φ(x) = (1/r_s²) Σ M̃ₙᵐ (∇Iₙᵐ)(v) with
  v = (x − c_s) / r_s = (u_t − ĉ(s|t)) / r̂(s|t), which the frame (ĉ(s|t), r̂(s|t))
  computes. `m2p` returns (1/r̂) Σ M̃ₙᵐ Iₙᵐ(v) = (r_t / r_s) Σ M̃ₙᵐ Iₙᵐ(v) = r_t φ = φ̂, and
  (1/r̂²) Σ M̃ₙᵐ (∇Iₙᵐ)(v) = (r_t² / r_s²) Σ M̃ₙᵐ (∇Iₙᵐ)(v) = r_t² ∇φ = ĝ. The 1/r̂ and
  1/r̂² that `m2p` applies are exactly the factors that convert to units of t.
- **P2P.** ŷ = (y − c_t) / r_t = ĉ(s|t) + r̂(s|t) u_s and x − y = r_t (u_t − ŷ). So
  Σ q / |u_t − ŷ| = r_t Σ q / |x − y| = φ̂, and
  −Σ q (u_t − ŷ) / |u_t − ŷ|³ = r_t² · (−Σ q (x − y) / |x − y|³) = ĝ, which `p2p`
  computes from ŷ and u_t. For s = t, ŷ = u_s: the stored chunk is passed unchanged.
  For s ≠ t, ŷ is formed in scratch; r̂ u_s is exact (barring underflow) and the
  addition rounds once.
- **M2M, L2L and M2L.** The coefficients are scaled (§3.7) and the tables of §3.12 are
  the operators at the relative frames (see "Relative frames"). They are looked up by
  child index o and offset index (d); no geometry is evaluated.

Since ĉ and r̂ are exact in T, `Frame::scaled` rounds once in P2L and M2P (the
subtraction; division by a power of two is exact), relative to |u − ĉ|, on every level
and wherever the domain lies, in place of the ε |c| / r_l of a floating-point shift
(§3.12).

### Coincident pairs

A source and a target at the same point x lie in the same leaf, because
`points_to_morton` maps a point to its key by its coordinates alone. They get the same
u bit for bit, and for s = t P2P passes the stored u_s unchanged, so the exact-coincidence
rule of `nd_fmm_ref::p2p` still excludes the pair.

This assumes that sources and targets in one leaf are loaded with the same formula from
the same f64 coordinates: the same a, w and key, the same operation order and the same
T. Two consequences, both below the resolution of T in the leaf:

- distinct points whose stored u agree in T (closer than about ε_T r_l per component)
  are excluded as coincident;
- for s ≠ t, a rounded ŷ that equals u_t is excluded likewise.

### Output

```math
\phi(\mathbf{x}) = \frac{\hat\phi}{4\pi r_t}, \qquad
\nabla\phi(\mathbf{x}) = \frac{\hat{\mathbf{g}}}{4\pi r_t^2}
```

with r_t = w / 2^(l_t+1), exact in f64. `nd-fmm-exec` applies both once, when producing
output (§3.1). No operator, table or chunk contains 1/(4π) or r_t.

### Verification

`tools/fixtures/check_leaf_geometry.py` checks, with seeded random keys and points, in
exact rationals, in IEEE doubles that reproduce `nd_octree` operation by operation, and
in mpmath at 40 digits:

- the integer centres against the midpoint of `morton::physical_box` and the centre of
  §3.12, on every level 0–16 and every L ≥ l, for a dyadic and a generic domain;
- the relative frames against (c_s − c_t) / r_t and r_s / r_t, for random key pairs on
  all level combinations and three domains, their exact round trip through f32 and f64,
  the largest numerator (2¹⁷ − 2), the reversal rule, and the canonical frames of
  §3.12;
- the f64 evaluation of u, and its cast to f32, against the error bound, at the deepest
  level and on random levels, for a domain far from the origin and one around it;
- containment in [−1, 1]³ up to β_k, for the domain of `compute_global_bounding_box`
  and points near the leaf faces;
- the frame maps of P2L, M2P and P2P, exactly;
- the scaling identities: P2P in leaf-scaled coordinates against the absolute potential
  and gradient, L2P and M2P at the frames above against r_t and r_t² times their
  absolute values, and P2M and P2L against the coefficients of the absolute frames.

The expansion operators themselves are checked in Rust by Phase 3 T8.
