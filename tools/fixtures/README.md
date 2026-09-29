# Fixture tools

High-precision reference values for `nd-fmm-math`, generated here and committed in
`fmm-math/fixtures/`. Fixtures are never edited by hand; regenerate them with the
scripts below.

| Script | Purpose | Needs |
| --- | --- | --- |
| `gen_harmonics.py` | writes `fmm-math/fixtures/harmonics_{A,B,C}.json` | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `crosscheck_scipy.py` | independent double-precision second opinion on the conventions and the fixtures | Python ≥ 3.9, `numpy`, `scipy` |
| `check_translations.py` | high-precision check of the translation operators of CONVENTIONS §3.11; writes nothing | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |

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
