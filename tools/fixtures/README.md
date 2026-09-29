# Fixture tools

High-precision reference values for `nd-fmm-math`, generated here and committed in
`fmm-math/fixtures/`. Fixtures are never edited by hand; regenerate them with the
scripts below.

| Script | Purpose | Needs |
| --- | --- | --- |
| `gen_harmonics.py` | writes `fmm-math/fixtures/harmonics_{A,B}.json` | Python ≥ 3.9, `mpmath==1.4.1` (pinned) |
| `crosscheck_scipy.py` | independent double-precision second opinion on the conventions and the fixtures | Python ≥ 3.9, `numpy`, `scipy` |

Both scripts declare their dependencies inline (PEP 723), so with
[uv](https://docs.astral.sh/uv/) no environment needs to be set up. Run them from the
repository root.

## Regenerate

```sh
uv run tools/fixtures/gen_harmonics.py
```

or, in an environment with `pip install mpmath==1.4.1`,
`python3 tools/fixtures/gen_harmonics.py`. It takes about two minutes. The script
refuses to run with any other mpmath version.

Regeneration is byte-identical: `--check` regenerates in memory and fails unless the
committed files match exactly and stay under the size limit (3 MB in total):

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

Each file has a `header` (set, seed, p, `convention_version`, `generator_version`,
mpmath version, verified digits, layout and format notes) and two lists of point
records: `regular`, with R_n^m at |u| in [0, √3], and `irregular`, with I_n^m at
|v| in [2, 8] (CONVENTIONS §3.3, §3.9). A record is

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
operations, so they are identical on every platform.

`crosscheck_scipy.py` compares the fixtures with SciPy's Legendre functions (with the
Condon–Shortley phase removed) in double precision: values from the definitions,
regular gradients from the ladder of §3.4, and irregular gradients from the analogous
ladder ∂z Iₙᵐ = −Iₙ₊₁ᵐ, (∂x − i∂y) Iₙᵐ = Iₙ₊₁ᵐ⁻¹, (∂x + i∂y) Iₙᵐ = −Iₙ₊₁ᵐ⁺¹. Errors
are measured relative to the largest reference value of the same degree, because single
components can be arbitrarily close to zero.
