# Fixture tools

High-precision reference values for `nd-fmm-math`, generated here and committed in
`fmm-math/fixtures/`. Fixtures are never edited by hand; regenerate them with the
scripts below.

| Script | Purpose | Needs |
| --- | --- | --- |
| `gen_harmonics.py` | writes `fmm-math/fixtures/harmonics_{A,B,C}.json` | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `gen_dipoles.py` | writes `fmm-ref/fixtures/dipoles.json`: the dipole P2M and P2L coefficients and the dipole direct sum of CONVENTIONS §3.7, "Dipole sources" (Phase 6 T8) | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `crosscheck_scipy.py` | independent double-precision second opinion on the conventions and the fixtures | Python ≥ 3.9, `numpy`, `scipy` |
| `check_translations.py` | high-precision check of the translation operators of CONVENTIONS §3.11; writes nothing | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `check_symmetry.py` | high-precision check of the box geometry and cube symmetry rules of CONVENTIONS §3.12; writes nothing | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `check_leaf_geometry.py` | exact and high-precision check of the leaf data and relative box geometry of CONVENTIONS §3.13; writes nothing | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `plane_wave_quadrature.sh` | writes `fmm-tables/src/plane_wave/quadrature_data.rs`: FMM3D's plane-wave quadratures for separation 1 (NIST's `lwtsexp_sep1.f` of FMM3D v2.1.0, with its notice; Phase 6 T13, from the spike of T10) | POSIX `sh`, `awk`, `sed`; FMM3D's source (`tools/fmm3d/build.sh fetch`) |
| `gen_helmholtz_fmm3d.py` | writes `fmm-ref/fixtures/helmholtz_fmm3d.json`: FMM3D 2.1.0's Helmholtz direct sum (`h3ddirectcg`) for real, complex and imaginary k and real and complex charges, every value checked against the kernel of CONVENTIONS §3.15 in mpmath, with FMM3D's two-point check (Helmholtz CH1.4) | Python ≥ 3.9, `mpmath==1.4.1` (pinned); FMM3D built by `tools/fmm3d/build.sh` (`FMM3D_ROOT`) |
| `gen_helmholtz_angular.py` | writes `fmm-math/fixtures/helmholtz_angular.json`: the real orthonormal harmonics with ∂θ and (1/sin θ) ∂φ, and sampled entries of the y-rotation blocks, to degree 1000 (CONVENTIONS §3.15; Helmholtz CH1.2a) | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `check_3_15.py` | high-precision check of the identities of the Helmholtz family, CONVENTIONS §3.15 (Helmholtz CH1.1); writes nothing | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `check_p2p_domain.py` | exact check of the coincident-pair rule and the domain of r² of fast P2P kernels, CONVENTIONS §3.13 "Fast kernels"; writes nothing | Python ≥ 3.9, `mpmath==1.4.1` (pinned, through `check_leaf_geometry.py`) |

All scripts declare their dependencies inline (PEP 723), so with
[uv](https://docs.astral.sh/uv/) no environment needs to be set up. Run them from the
repository root.

## Regenerate

```sh
uv run tools/fixtures/gen_harmonics.py
```

or, in an environment with `pip install mpmath==1.4.1`,
`python3 tools/fixtures/gen_harmonics.py`. It takes about five minutes. The script
refuses to run with any other mpmath version.

Regeneration is byte-identical: `--check` regenerates in memory and fails unless the
committed files match exactly and stay under the size limit (3.5 MB in total; it was
3 MB until fixture set C, about 0.17 MB, took the total to 2.98 MB, and was raised to
leave room for regenerations):

```sh
uv run tools/fixtures/gen_harmonics.py --check
```

Then run the cross-check, which must pass (agreement to about 1e-13):

```sh
uv run tools/fixtures/crosscheck_scipy.py
```

Regenerate when `CONVENTION_VERSION` changes (the generator reads it from
`docs/CONVENTIONS.md` and writes it into every header) or when the generator itself
changes; in the latter case bump `GENERATOR_VERSION` in `gen_harmonics.py`. Commit the
regenerated fixtures together with the change that caused them.

The dipole fixture regenerates the same way, in about a second, byte-identically:

```sh
uv run tools/fixtures/gen_dipoles.py
uv run tools/fixtures/gen_dipoles.py --check
```

It holds, for p = 16 (seed 20261017), three P2M and three P2L cases (five dipoles each, in
the frames ((0, 0, 0), 1), ((0.3125, −0.21875, 0.6875), 0.37) and ((1, 1, 1), 2⁻¹⁶),
sources at |u| ≤ √3 and 2 ≤ |u| ≤ 8), with the coefficients Σ (d/r) · ∇ conj Xₙᵐ(u) in
real storage, and the potential and gradient of 20 dipoles at 10 targets. The
coefficients are directional derivatives of the definitions of §3.3 by central
differences at 120 digits, verified by a second step size; the direct sum is the closed
form, its gradient verified against differences of the potential. Regenerate it when
`CONVENTION_VERSION` changes (it is written into the header) or the generator changes
(bump its `GENERATOR_VERSION`). `nd-fmm-ref`'s `tests/dipoles/fixtures.rs` reads it.

The Helmholtz FMM3D fixture (CH1.4) comes from FMM3D's own code, so it needs FMM3D
built outside the repository (tools/fmm3d/README.md). It takes about 25 seconds:

```sh
FMM3D_ROOT=... uv run tools/fixtures/gen_helmholtz_fmm3d.py
FMM3D_ROOT=... uv run tools/fixtures/gen_helmholtz_fmm3d.py --check
FMM3D_ROOT=... python3 tools/fixtures/gen_helmholtz_fmm3d.py --check --no-reference   # no mpmath
```

It holds 200 sources in [0, 1]³ and 40 targets in [−0.5, 1.5]³ (seed 20261028), real and
complex charges, and k = 1, 25, 1 + 0.5i, 25 + 2.5i, i and 25i. For each case it stores the
potential and gradient of e^{ik|x−y|}/(4π|x−y|) at the sources (the self term dropped)
and at the targets. The header records the release, commit, routine, `thresh`, compiler
and machine, and how closely each case agrees with the kernel of CONVENTIONS §3.15 summed
in mpmath at 30 digits; generation fails above 10⁻¹³. It also has the raw two-point values and the
convention they identify. FMM3D's sums differ in their last bits between compilers, so
`--check` compares the values to 10⁻¹³ relative, not the bytes. It also says whether the
file is byte-identical, which it is on the machine and compiler that wrote it.
`nd-fmm-ref` reads it from CH1.3 on.

The Helmholtz angular fixture (CH1.2a) regenerates byte-identically, in about a minute
and a half:

```sh
uv run tools/fixtures/gen_helmholtz_angular.py
uv run tools/fixtures/gen_helmholtz_angular.py --check
```

It holds two parts. "harmonics": Yₙᵐ, ∂θ Yₙᵐ and (1/sin θ) ∂φ Yₙᵐ at seven directions
(generic, both poles, near both poles at θ ≈ 10⁻³, the equator, below it), degrees 0 to
1000, the orders sampled; P̄ₙᵐ by the plain three-term recursion at 60 and 100 digits,
which must agree to 10⁻³⁰, the ∂θ ladder checked against mpmath's derivative first.
"rotation": spike S1's 54 sets of y-rotation block entries (degrees 10 to 1000, six
polar angles), from Wigner's sum, value for value those of
`spikes/rotation-blocks/fixtures/reference.json`. Regenerate it when `CONVENTION_VERSION`
changes or the generator changes (bump its `GENERATOR_VERSION`). `nd-fmm-math`'s
`tests/orthonormal/fixtures.rs` reads it.

## What the fixtures contain

| File | Points per family | p | Seed |
| --- | --- | --- | --- |
| `harmonics_A.json` | 10 | 30 | 20260929 |
| `harmonics_B.json` | 50 | 8 | 20260930 |
| `harmonics_C.json` | 4 (irregular only, values only) | 40 | 20261001 |

Each file has a `header` (set, seed, p, `convention_version`, `generator_version`,
mpmath version, verified digits, layout and format notes) and two lists of point
records: `regular`, with R_n^m at |u| in [0, √3], and `irregular`, with I_n^m at
|v| in [2, 8] (CONVENTIONS §3.3, §3.9). Set C covers the M2L shifts of CONVENTIONS
§3.9 and §3.11 instead: irregular harmonics up to degree 40 = 2p for p = 20, at |v| in
[4, 11]; its `regular` list is empty and its records have no `grad`. A record is

```json
{"x": [x, y, z], "value": [...], "grad": [[d/dx ...], [d/dy ...], [d/dz ...]]}
```

where `value` and each gradient component hold (p + 1)² reals in the real storage
order of CONVENTIONS §3.6: index n² + n + m; slot m = 0 is the real value, slot +m the
real part and slot −m the imaginary part of the order-m harmonic (or of its
derivative).

Every number is a decimal string with 17 significant digits of the double nearest to
the exact value, so parsing it (for example with Rust's `str::parse::<f64>`) gives that
double exactly. The coordinates in `x` are doubles, and the harmonics are those of
exactly these doubles.

## How the values are computed

The generator uses the definitions of CONVENTIONS §3.3, not the recursions of §3.5 that
the Rust code implements:

- P_n^m(t) = (1 − t²)^{m/2} dᵐ/dtᵐ P_n(t) from the exact rational coefficients of the
  Legendre polynomial, so the Condon–Shortley phase cannot slip in. `mpmath.legenp`,
  which includes that phase (its P₁¹(t) is −√(1 − t²); the script asserts this),
  serves as a sign-corrected cross-check at a few points per set.
- Spherical coordinates and e^{imφ} with mpmath's functions.
- Gradients by central differences at 100 digits of working precision.

Each value is verified to 40 significant digits, relative to the largest magnitude in
its degree, before rounding: values by recomputing at a higher precision, gradients by
repeating the differences with a second step size, and regular gradients against the
ladder of CONVENTIONS §3.4. Gradient components that vanish identically (such as
∂z Rₙⁿ) are written as exact zeros; their central differences are rounding noise,
detected because it changes with the step size.

Points are drawn with Python's `random.Random(seed)` and only correctly rounded IEEE
operations, so they are identical on every platform. Sets A and B draw their regular
points first, then the irregular ones; set C draws only irregular points.

`crosscheck_scipy.py` compares the fixtures with SciPy's Legendre functions (with the
Condon–Shortley phase removed) in double precision: values from the definitions,
regular gradients from the ladder of §3.4, and irregular gradients from the analogous
ladder ∂z Iₙᵐ = −Iₙ₊₁ᵐ, (∂x − i∂y) Iₙᵐ = Iₙ₊₁ᵐ⁻¹, (∂x + i∂y) Iₙᵐ = −Iₙ₊₁ᵐ⁺¹. Errors
are measured relative to the largest reference value of the same degree, because single
components can be arbitrarily close to zero.

## Translation check

```sh
uv run tools/fixtures/check_translations.py
```

checks the formulas of CONVENTIONS §3.11 in mpmath at 40 significant digits (the
harmonics, imported from `gen_harmonics.py`, are evaluated with guard digits because
its monomial Legendre polynomials cancel at high degree). For seeded random frames and
charges it checks, and prints the worst error of:

| Check | Tolerance |
| --- | --- |
| M2M after P2M equals P2M in the output frame (p = 20) | 1e-30 |
| L2L: the local expansion evaluated before and after, inside the output sphere (p = 20) | 1e-30 |
| M2L after P2M equals P2L, input degree 60, output degree 8 | the truncation bound of §3.11 per coefficient, plus 1e-30 relative |
| coaxial forms equal the general ones for shifts along the z-axis, d > 0 and d < 0 (p = 16) | 1e-30 |
| rotation rule K Dⁿ K, K S Dⁿ S⁻¹ K for coefficients, n ≤ 8 | 1e-30 |
| rotate, coaxial, rotate back equals the general form (p = 8) | 1e-30 |
| L2P and M2P gradients against central differences (p = 8) | 1e-30 |
| M2M, L2L, M2L from real storage through the order-sum identities of §3.11 equal the complex forms (p = 12) | 1e-30 |

Coefficients are compared per degree in the orthonormal weighting of CONVENTIONS §3.8
(Nₘ for multipoles, Nₘ/Sₘ for locals), potentials relative to the sum of term
magnitudes. Dⁿ is fitted from its definition Rₙ(Qx) = Dⁿ Rₙ(x) at random points, not
taken from `nd-fmm-math`. The script exits non-zero on any failure and takes about
half a minute.

## Symmetry check

```sh
uv run tools/fixtures/check_symmetry.py
```

checks the box geometry and the cube symmetry rules of CONVENTIONS §3.12 in mpmath at 40
significant digits, for degrees n ≤ 8 and seeded random points and coefficients. It
imports the harmonics from `gen_harmonics.py` and the translation operators, the fit of
Dⁿ and the error measures from `check_translations.py`. It prints the worst error of:

| Check | Tolerance |
| --- | --- |
| the 316 V-list offsets: lexicographic order, none in {−1..1}³, the closed-form table index round-trips | exact |
| box centres and half-widths as `morton::physical_box`, the child index o = 4x + 2y + z from the bits of a Morton key, child centres c + r_child s_o, the shift 2 r_l d (random domains, levels 0–15, exact rationals) | exact |
| O_h: 48 distinct signed permutations in the enumeration order of §3.12, 24 proper, closed under composition, Pᵀ the inverse, 16 z-axis elements | exact |
| classes: 16 orbits (34 under the z-axis elements), one representative 0 ≤ d_x ≤ d_y ≤ d_z each, P · representative = d under the group-element rule | exact |
| octants: every P permutes the eight s_o; element o is the first with P s₀ = s_o | exact |
| Rₙ(Px) = Dⁿ(P) Rₙ(x) and Iₙ(Px) = S Dⁿ(P) S⁻¹ Iₙ(x) for all 48 P, Dⁿ fitted for the 24 proper P and (−1)ⁿ Dⁿ(−P) for the improper ones | 1e-30 |
| Dⁿ(P₁P₂) = Dⁿ(P₁) Dⁿ(P₂) on 96 sampled pairs, T(P) T(Pᵀ) = I for T_M and T_L and all 48 P, Dⁿ(−I) = (−1)ⁿ | 1e-30 |
| M2L(P d) = T_L(P) M2L(d) T_M(Pᵀ) for all 316 offsets from their representatives (p = 8) | 1e-30 |
| M2M(P s) = T_M(P) M2M(s) T_M(Pᵀ) and L2L(P s) = T_L(P) L2L(s) T_L(Pᵀ) for all 48 P from octant 0 (p = 8) | 1e-30 |
| M2L(−d) = diag((−1)ʲ) M2L(d) diag((−1)ⁿ) for all 316 offsets (p = 8) | 1e-30 |
| z-axis elements: T_L = T_M, signed permutations of the (+m, −m) slot pairs, diagonal unless P exchanges x and y, then exchanging the pairs of odd m; the pattern is printed | 1e-30 |

The operators are the general forms of §3.11 from `check_translations.py`, applied to
random coefficient vectors at the canonical frames of §3.12. Coefficient vectors are
compared per degree in the orthonormal weighting of CONVENTIONS §3.8, as in the
translation check. The script exits non-zero on any failure and takes a little over
a minute.

## Leaf geometry check

```sh
uv run tools/fixtures/check_leaf_geometry.py
```

checks the leaf data and relative box geometry of CONVENTIONS §3.13 with seeded random
keys and points: the geometry in exact rationals (`fractions.Fraction`), the f64
evaluation of leaf-scaled coordinates in Python floats (IEEE doubles, reproducing
`compute_global_bounding_box`, `points_to_morton` and the formula of §3.13 operation by
operation), and the scaling identities in mpmath at 40 significant digits. It imports
the leaf sums, the gradient ladder and the error measures from `check_translations.py`.
It prints the worst error of:

| Check | Tolerance |
| --- | --- |
| integer centres a + C_L w / 2^(L+1) equal the midpoint of `morton::physical_box` and the §3.12 centre; C_L odd times 2^(L − l) (levels 0–16, every L ≥ l, a dyadic and a generic domain) | exact |
| relative frames ĉ(s\|t), r̂(s\|t) equal (c_s − c_t) / r_t and r_s / r_t, round-trip through f32 and f64 (`struct.pack`), reversal rule; largest numerator reported (random key pairs on all 17 × 17 level combinations, three domains) | exact |
| the relative frames of child and parent, and of V-list pairs, are the canonical frames (½ s_o, ½) and (2d, 1) of §3.12 | exact |
| u evaluated in f64 in the order of §3.13, and cast to f32, against the exact u (level 16 and random levels, a domain far from the origin and one around it) | the error bound of §3.13 (ratio ≤ 1) |
| u of points that `points_to_morton` puts into the leaf, including points within a few ulps of the leaf and domain faces, lies in [−1, 1]³ up to β_k (exact u) and β_k plus the error bound (f64, f32) | ratio ≤ 1 |
| frame maps: (u_t − ĉ(s\|t)) / r̂(s\|t) = (x − c_s) / r_s, (u_s − ĉ(t\|s)) / r̂(t\|s) = (y − c_t) / r_t, ĉ(s\|t) + r̂(s\|t) u_s = (y − c_t) / r_t | exact |
| P2P in leaf-scaled coordinates times 1 / r_t and 1 / r_t² equals the absolute potential and gradient | 1e-30 |
| L2P at the unit frame and M2P at (ĉ(s\|t), r̂(s\|t)) give r_t φ and r_t² ∇φ of the absolute frames (p = 8) | 1e-30 |
| P2M at the unit frame and P2L at (ĉ(t\|s), r̂(t\|s)) give the coefficients of the absolute frames (p = 8) | 1e-30 |

The domain of the containment check comes from the emulated
`compute_global_bounding_box`, and u uses its x side as w, so the y and z sides differ
from w by rounding, as in docs/phase3/README.md ("Domain"). The bounds are the
rigorous forms (with the ε² terms) of the first-order bounds in §3.13. The expansion
operators themselves are checked in Rust by Phase 3 T8. The script exits non-zero on
any failure and takes a few seconds.

## P2P domain check

```sh
uv run tools/fixtures/check_p2p_domain.py
```

checks the claims of CONVENTIONS §3.13, "Fast kernels": that a fast P2P kernel, which
skips a pair when r² = 0, skips exactly the pairs that the exact-coincidence rule of
`nd_fmm_ref::p2p` skips on leaf-scaled data, and that every nonzero r² lies in the kernel
domain [2⁻¹⁰⁸, 2⁷]. The stored values come from `check_leaf_geometry.py`'s emulation of
`points_to_morton` and of the f64 formula of §3.13, cast to f32. ŷ, dₖ and r² are formed
operation by operation in T: an f32 operation as the f32 rounding of the f64 result
(correctly rounded, since 53 ≥ 2 · 24 + 2), an fma rounded once from its exact value
(`fractions.Fraction`). The claims are checked in exact rationals. It prints, and fails
on any violation of:

| Check | Claim |
| --- | --- |
| the f32 and f64 emulation against exact rounding of rationals, including f32 subnormals | exact |
| u = fl(v − (2i + 1)) and its f32 cast for every double v near 2i + 1, (2i + 1) ± ½ and (2i + 1) ± 1, and tiny v; smallest nonzero \|u\| reported | u ∈ G₅₃ |
| adversarial pairs: levels 0–16, the self pair and every U-list neighbour of 2:1 balance (every ĉ component), points within a few spacings of leaf centres, leaf faces (as `points_to_morton` draws them), source centres and cancellations ĉ + r̂ u_s ≈ 0, four domains | ŷ, dₖ ∈ G₅₄ and G₅₃; dₖ = 0 ⇔ u_t,k == ŷₖ; r² = 0 ⇔ the reference rule skips; nonzero r² ≥ 2⁻¹⁰⁶ |
| seeded random pairs on levels 0–16, three domains, duplicated points in the self pairs | the same |
| the far corners of corner neighbours on all three levels | r² ≤ 3 (6 + 3β′)² (1 + ε_T)⁷ < 2⁷ |
| f32 gradient intermediates r² · r (reference) and (q ρ) ρ² (fast kernels) for r² in [2⁻⁸⁴, 2⁷] | normal and finite |

r² is formed as the reference forms it, ((d₀² + d₁²) + d₂²), and with fma in two
orders. The last lines give, per precision, the smallest nonzero r² found (2⁻¹⁰⁶, the
sharp bound) with its margin to the domain's lower end 2⁻¹⁰⁸ and to the smallest normal
number, and the largest r² (108). It takes about half a minute.

## Helmholtz check

```sh
uv run tools/fixtures/check_3_15.py
```

checks the identities of CONVENTIONS §3.15 (Helmholtz CH1.1) in mpmath at 40 significant
digits, from the section's own definitions (its harmonics, radial functions and stored
coefficients are written out in the script, not imported), on seeded random points, for
real, complex and imaginary k and real and complex charges. It prints the worst error of:

| Check | Tolerance |
| --- | --- |
| the gradient of the kernel against a numerical derivative; k → 0 against §3.13 | 1e-30; 1e-18 |
| h_n = j_n + i y_n against its finite Hankel sum; k₀(x) = e^{−x}/x; j_n(ix) = iⁿ i_n(x); h_n(ix) = −i^{−n} k_n(x) | 1e-30 |
| Σ_m Y_n^m(a) Y_n^m(b) = (2n + 1)/(4π) P_n(a · b), and r^n Y_n^m against the Racah R_n^m of §3.3 through ν_m and √2 (n ≤ 15) | 1e-32 |
| e^{ikr}/r = 4π ik Σ j_n h_n Σ Y Y (p = 60) | 1e-28 |
| P2M → M2P and P2L → L2P in stored form (scale, per-box damping factor, 4π ik), |k| a below and above 1, against the direct sum (p = 50); every applied exponent ≤ 0 | 1e-24; exact |
| Yukawa: M^Y = i^{−n} M and L^Y = iⁿ L real for real charges, their real forms, and e^{−κr}/r = 4πκ Σ i_n k_n Σ Y Y | 1e-30; 1e-28 |
| Oⁿ(R_y(β)) orthogonal and Oⁿ(R_y(π − β)) = (−1)ⁿ E Oⁿ(R_y(β)) E F, Oⁿ fitted from Y(Qx̂) = Oⁿ Y(x̂) (n ≤ 6) | 1e-30 |
| coaxial (R\|R), (S\|S), (S\|R) by projection (n, n′ ≤ 5, m ≤ 2): one coefficient per ±m, (S\|S) = (R\|R), real for real k, i^{n−n′} (E\|F) real for Yukawa, the index symmetry and the reflection of (S\|R) and (R\|R) (relative) | 1e-20 |
| P2M → M2L → L2P on the z-axis in stored form: the stored (S\|R) table (by projection), 4π ik and e^{2β − Im(k)\|t\|} (p = 10), against the direct sum; every applied exponent ≤ 0 | 1e-10 (truncation about 3e-13); exact |
| P2M → M2M → M2P and P2L → L2L → L2P on the z-axis in stored form, child and parent, the stored (S\|S) and (R\|R) with the power scales alone (the damping factors cancel) (p = 14), against the direct sum | 1e-14 (truncation: Yukawa 1e-11 at p = 10, 1e-17 at 14) |
| the start column and the recursions in n and m of §3.15 against the projected (R\|R) and (S\|R) (n, n′ ≤ 6, m ≤ 3), relative to the table's largest entry | 1e-20 |
| Yukawa's stored M̂^Y and L̂^Y from î and k̂ against i^{−n} M̂ and iⁿ L̂ (p = 30) | 1e-28 |

The radial functions come from mpmath's Bessel functions; h_n from its finite Hankel sum,
since j_n + i y_n cancels by e^{2 Im z} (the two are compared at moderate arguments). The
script exits non-zero on any failure and takes about three minutes.
