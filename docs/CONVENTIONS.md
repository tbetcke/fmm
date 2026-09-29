# Conventions

`CONVENTION_VERSION = 1` (draft until signed off at the end of Phase 0 task T2).

This file is the single source of truth for basis functions, phases, storage and
scaling in the `nd-fmm-*` crates. Code cites it as `CONVENTIONS §3.x`. Any change to
§3.1–§3.8 bumps `CONVENTION_VERSION`, which invalidates committed fixtures and cached
operator tables.

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

## 3.10 Versioning

`CONVENTION_VERSION = 1`. Any change to §3.1–§3.8 bumps it, which invalidates committed
fixtures and cached operator tables.
