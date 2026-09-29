#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""High-precision fixtures for the solid harmonics of docs/CONVENTIONS.md.

Writes fmm-math/fixtures/harmonics_{A,B,C}.json: regular harmonics R_n^m and irregular
harmonics I_n^m (CONVENTIONS §3.3) and their Cartesian gradients, in the real storage
order of §3.6, at seeded random points in the ranges of §3.9. Set C holds irregular
values only (no gradients), up to the degree 2p that M2L needs (§3.9, §3.11).

Everything is computed from the definitions in §3.3, not from the recursions of §3.5
that the Rust code implements:
  - P_n^m(t) = (1 - t^2)^(m/2) d^m/dt^m P_n(t) with exact rational coefficients of the
    Legendre polynomial, so no Condon-Shortley phase can slip in; mpmath.legenp (which
    has that phase) serves as a sign-corrected cross-check;
  - spherical coordinates and e^{i m phi} evaluated with mpmath's own functions;
  - gradients by central differences at elevated working precision.
Every value is checked to at least DIGITS significant digits (relative to the largest
value of its degree) by recomputing it at a higher precision or with a second step
size, then rounded to the nearest double and written with 17 significant digits, so
that parsing the string gives exactly that double.

Run from the repository root:  uv run tools/fixtures/gen_harmonics.py
(or: python3 tools/fixtures/gen_harmonics.py with mpmath==1.4.1 installed).
Regeneration is byte-identical. See tools/fixtures/README.md.
"""
import argparse
import json
import math
import random
import re
import sys
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

import mpmath
from mpmath import mp, mpf, mpc

# Bump when the generator changes its output (format, points or algorithm).
GENERATOR_VERSION = 2
MPMATH_VERSION = "1.4.1"
# Significant digits every value is verified to before rounding to double.
DIGITS = 40
# Limit on the total size of the committed fixtures: 3 MB in docs/phase0/T3-fixtures.md,
# raised to 3.5 MB for set C (docs/phase1/T2-translation-conventions.md).
MAX_TOTAL_BYTES = 3_500_000

ROOT = Path(__file__).resolve().parents[2]
CONVENTIONS = ROOT / "docs" / "CONVENTIONS.md"
OUT_DIR = ROOT / "fmm-math" / "fixtures"

# Ranges of CONVENTIONS §3.9: regular harmonics inside the box's sphere, irregular ones
# well separated; for set C the scaled M2L shifts |c' - c| / r in [4, 6 sqrt(3)].
REGULAR_RADII = (0.0, math.sqrt(3.0))
IRREGULAR_RADII = (2.0, 8.0)
M2L_RADII = (4.0, 11.0)

# name, seed, number of points per family, maximum degree p. Sets A and B: both
# families with gradients; their points are drawn regular first, then irregular.
SETS = [("A", 20260929, 10, 30), ("B", 20260930, 50, 8)]
# name, seed, number of points, maximum degree p. Set C: irregular values only.
IRREGULAR_VALUE_SETS = [("C", 20261001, 4, 40)]

# Working precision (decimal digits) for values and its verification, and for the
# central differences, with their step sizes.
VALUE_DPS, VALUE_CHECK_DPS = 60, 80
GRAD_DPS = 100
GRAD_STEP, GRAD_CHECK_STEP = mpf(10) ** -30, mpf(10) ** -27
# Relative disagreement between the two step sizes above which a gradient component is
# rounding noise of an exactly vanishing value (see flush_noise).
NOISE_AGREEMENT = mpf(10) ** -10
# Points per family at which P_n^m is cross-checked against mpmath.legenp.
LEGENP_POINTS = 2


def convention_version():
    """CONVENTION_VERSION as stated in docs/CONVENTIONS.md (every statement must agree)."""
    found = set(re.findall(r"`CONVENTION_VERSION = (\d+)`", CONVENTIONS.read_text()))
    if len(found) != 1:
        sys.exit(f"expected one CONVENTION_VERSION in {CONVENTIONS}, found {sorted(found)}")
    return int(found.pop())


# ---------------------------------------------------------------------------------------
# Points. Only correctly rounded IEEE operations (+, -, *, /, sqrt) and Python's
# Mersenne Twister are used, so the points are identical on every platform.


def random_points(rng, count, rmin, rmax):
    points = []
    while len(points) < count:
        v = [2.0 * rng.random() - 1.0 for _ in range(3)]
        s = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
        # Uniform directions by rejection from the cube; the lower bound keeps the
        # normalisation well conditioned.
        if not 1e-2 < s <= 1.0:
            continue
        norm = math.sqrt(s)
        r = rmin + (rmax - rmin) * rng.random()
        points.append([r * (c / norm) for c in v])
    return points


# ---------------------------------------------------------------------------------------
# Basis functions from the definitions of CONVENTIONS §3.3.


def legendre_derivative_coefficients(p):
    """coef[n][m]: exact coefficients (ascending powers) of d^m/dt^m P_n(t)."""
    coef = []
    for n in range(p + 1):
        # P_n(t) = 2^-n sum_k (-1)^k C(n, k) C(2n - 2k, n) t^(n - 2k).
        c = [Fraction(0)] * (n + 1)
        for k in range(n // 2 + 1):
            c[n - 2 * k] = Fraction((-1) ** k * comb(n, k) * comb(2 * n - 2 * k, n), 2**n)
        derivs = [c]
        for _ in range(n):
            c = [j * c[j] for j in range(1, len(c))]
            derivs.append(c)
        coef.append(derivs)
    return coef


def polyval(c, t):
    acc = mpf(0)
    for a in reversed(c):
        acc = acc * t + mpf(a.numerator) / a.denominator
    return acc


def legendre_table(coef, p, t):
    """P[n][m] = P_n^m(t) without the Condon-Shortley phase, 0 <= m <= n <= p."""
    s = mpmath.sqrt(1 - t * t)
    return [[s**m * polyval(coef[n][m], t) for m in range(n + 1)] for n in range(p + 1)]


def harmonics(coef, p, x, regular):
    """Complex X_n^m(x) for 0 <= m <= n <= p, X = R (regular) or I (irregular), §3.3."""
    x, y, z = (mpf(c) for c in x)
    r = mpmath.sqrt(x * x + y * y + z * z)
    theta, phi = mpmath.acos(z / r), mpmath.atan2(y, x)
    P = legendre_table(coef, p, mpmath.cos(theta))
    out = []
    for n in range(p + 1):
        for m in range(n + 1):
            if regular:
                radial = r**n / factorial(n + m)
            else:
                radial = factorial(n - m) / r ** (n + 1)
            out.append(radial * P[n][m] * mpmath.expj(m * phi))
    return out


def tri(n, m):
    """Position of (n, m), m >= 0, in the list returned by harmonics()."""
    return n * (n + 1) // 2 + m


# ---------------------------------------------------------------------------------------
# Verification helpers. Errors are measured against the largest magnitude in the same
# degree, because single components can be arbitrarily close to zero.


def degree_scales(p, vectors):
    scales = []
    for n in range(p + 1):
        s = max(abs(v[tri(n, m)]) for v in vectors for m in range(n + 1))
        scales.append(s if s > 0 else mpf(1))
    return scales


def check_close(p, a, b, what, digits=DIGITS):
    scales = degree_scales(p, [a])
    tol = mpf(10) ** -digits
    for n in range(p + 1):
        for m in range(n + 1):
            k = tri(n, m)
            err = abs(a[k] - b[k]) / scales[n]
            if err > tol:
                sys.exit(f"{what}: (n, m) = ({n}, {m}) agrees only to {mpmath.nstr(err, 3)}")


def check_legendre_sign():
    # mpmath.legenp includes the Condon-Shortley phase; CONVENTIONS §3.3 does not.
    t = mpf("0.3")
    if not abs(mpmath.legenp(1, 1, t) + mpmath.sqrt(1 - t * t)) < mpf(10) ** -30:
        sys.exit("mpmath.legenp no longer has the Condon-Shortley phase; revisit the correction")


def cross_check_legendre(coef, p, t):
    ours = legendre_table(coef, p, t)
    theirs = [[(-1) ** m * mpmath.legenp(n, m, t, type=2) for m in range(n + 1)] for n in range(p + 1)]
    flat_ours = [v for row in ours for v in row]
    flat_theirs = [v for row in theirs for v in row]
    check_close(p, flat_ours, flat_theirs, f"P_n^m({mpmath.nstr(t, 5)}) vs mpmath.legenp")


def check_ladder(p, value, grad, what):
    """Gradient ladder of CONVENTIONS §3.4 for regular harmonics (an internal check)."""

    def R(n, m):
        if abs(m) > n or n < 0:
            return mpc(0)
        v = value[tri(n, abs(m))]
        return v if m >= 0 else (-1) ** m * mpmath.conj(v)

    gx, gy, gz = grad
    scales = degree_scales(p, [gx, gy, gz])
    tol = mpf(10) ** -DIGITS
    for n in range(p + 1):
        for m in range(n + 1):
            k = tri(n, m)
            errs = (
                gz[k] - R(n - 1, m),
                gx[k] - 1j * gy[k] - R(n - 1, m - 1),
                gx[k] + 1j * gy[k] + R(n - 1, m + 1),
            )
            if max(abs(e) for e in errs) / scales[n] > tol:
                sys.exit(f"{what}: gradient ladder fails at (n, m) = ({n}, {m})")


# ---------------------------------------------------------------------------------------


def values_and_gradients(coef, p, x, regular):
    what = f"{'R' if regular else 'I'} at {x}"
    with mp.workdps(VALUE_DPS):
        value = harmonics(coef, p, x, regular)
    with mp.workdps(VALUE_CHECK_DPS):
        check_close(p, harmonics(coef, p, x, regular), value, what)

    def central(h):
        grad = []
        with mp.workdps(GRAD_DPS):
            for axis in range(3):
                plus = [mpf(c) + (h if i == axis else 0) for i, c in enumerate(x)]
                minus = [mpf(c) - (h if i == axis else 0) for i, c in enumerate(x)]
                fp = harmonics(coef, p, plus, regular)
                fm = harmonics(coef, p, minus, regular)
                grad.append([(a - b) / (2 * h) for a, b in zip(fp, fm)])
        return grad

    grad, grad_check = central(GRAD_STEP), central(GRAD_CHECK_STEP)
    with mp.workdps(GRAD_DPS):
        for axis, g, g_check in zip("xyz", grad, grad_check):
            check_close(p, g_check, g, f"d/d{axis} {what}")
        if regular:
            check_ladder(p, value, grad, what)
    return value, grad, grad_check


def to_double(v):
    """Nearest double to an mpf (round half to even), via exact rational arithmetic."""
    sign, man, exp, _ = v._mpf_
    if man == 0:
        return 0.0
    return float(Fraction((-1) ** sign * man) * Fraction(2) ** exp)


def fmt(v):
    """17 significant digits of the double nearest to v; parses back to that double."""
    return f"{to_double(mpf(v)):.16e}"


def real_storage(p, cvalues):
    """Real storage of CONVENTIONS §3.6: idx(n, m) = n^2 + n + m."""
    out = [None] * (p + 1) ** 2
    for n in range(p + 1):
        out[n * n + n] = cvalues[tri(n, 0)].real
        for m in range(1, n + 1):
            c = cvalues[tri(n, m)]
            out[n * n + n + m] = c.real
            out[n * n + n - m] = c.imag
    return out


def flush_noise(p, grad, grad_check, what):
    """Real-storage gradients with rounding noise of exactly vanishing components zeroed.

    Some components vanish identically (d/dx of R_1^0 = z, say); their central
    differences are pure rounding noise that changes with the step size. Genuine values,
    however small, agree between the two step sizes to many digits.
    """
    scales = degree_scales(p, grad)
    out = []
    for g, g_check in zip(grad, grad_check):
        a, b = real_storage(p, g), real_storage(p, g_check)
        for n in range(p + 1):
            for i in range(n * n, (n + 1) ** 2):
                if abs(a[i] - b[i]) > NOISE_AGREEMENT * abs(a[i]):
                    if abs(a[i]) > mpf(10) ** -DIGITS * scales[n]:
                        sys.exit(f"{what}: gradient component {i} is neither verified nor noise")
                    a[i] = mpf(0)
        out.append(a)
    return out


def point_record(coef, p, x, regular):
    value, grad, grad_check = values_and_gradients(coef, p, x, regular)
    with mp.workdps(GRAD_DPS):
        grad = flush_noise(p, grad, grad_check, f"{'R' if regular else 'I'} at {x}")
    return {
        "x": [fmt(c) for c in x],
        "value": [fmt(v) for v in real_storage(p, value)],
        "grad": [[fmt(v) for v in g] for g in grad],
    }


def value_record(coef, p, x):
    """Irregular values only, verified as in values_and_gradients (set C)."""
    what = f"I at {x}"
    with mp.workdps(VALUE_DPS):
        value = harmonics(coef, p, x, False)
    with mp.workdps(VALUE_CHECK_DPS):
        check_close(p, harmonics(coef, p, x, False), value, what)
    return {"x": [fmt(c) for c in x], "value": [fmt(v) for v in real_storage(p, value)]}


def generate_irregular_values(name, seed, count, p, version):
    rng = random.Random(seed)
    points = random_points(rng, count, *M2L_RADII)
    coef = legendre_derivative_coefficients(p)
    with mp.workdps(VALUE_DPS):
        for x in points[:LEGENP_POINTS]:
            x, y, z = (mpf(c) for c in x)
            cross_check_legendre(coef, p, z / mpmath.sqrt(x * x + y * y + z * z))
    header = {
        "description": (
            "Real irregular solid harmonics I_n^m, values only "
            "(docs/CONVENTIONS.md §3.3, §3.6), at seeded random points in the range "
            "of the M2L shifts (§3.9, §3.11)"
        ),
        "set": name,
        "convention_version": version,
        "generator": "tools/fixtures/gen_harmonics.py",
        "generator_version": GENERATOR_VERSION,
        "mpmath_version": MPMATH_VERSION,
        "verified_digits": DIGITS,
        "seed": seed,
        "p": p,
        "layout": "real storage, index n*n + n + m for -n <= m <= n (§3.6)",
        "format": (
            "decimal strings, 17 significant digits of the nearest double; "
            "no gradients, and no regular records"
        ),
        "irregular_radius": [fmt(r) for r in M2L_RADII],
    }
    return {
        "header": header,
        "regular": [],
        "irregular": [value_record(coef, p, x) for x in points],
    }


def generate(name, seed, count, p, version):
    rng = random.Random(seed)
    regular_points = random_points(rng, count, *REGULAR_RADII)
    irregular_points = random_points(rng, count, *IRREGULAR_RADII)
    coef = legendre_derivative_coefficients(p)
    with mp.workdps(VALUE_DPS):
        # mpmath.legenp is slow; a few points per family suffice to confirm the sign
        # and normalisation of the exact polynomials.
        for x in regular_points[:LEGENP_POINTS] + irregular_points[:LEGENP_POINTS]:
            x, y, z = (mpf(c) for c in x)
            cross_check_legendre(coef, p, z / mpmath.sqrt(x * x + y * y + z * z))
    header = {
        "description": (
            "Real solid harmonics R_n^m, I_n^m and their Cartesian gradients "
            "(docs/CONVENTIONS.md §3.3, §3.6) at seeded random points"
        ),
        "set": name,
        "convention_version": version,
        "generator": "tools/fixtures/gen_harmonics.py",
        "generator_version": GENERATOR_VERSION,
        "mpmath_version": MPMATH_VERSION,
        "verified_digits": DIGITS,
        "seed": seed,
        "p": p,
        "layout": "real storage, index n*n + n + m for -n <= m <= n (§3.6)",
        "format": (
            "decimal strings, 17 significant digits of the nearest double; "
            "grad is [d/dx, d/dy, d/dz], each in the same layout as value"
        ),
        "regular_radius": [fmt(r) for r in REGULAR_RADII],
        "irregular_radius": [fmt(r) for r in IRREGULAR_RADII],
    }
    return {
        "header": header,
        "regular": [point_record(coef, p, x, True) for x in regular_points],
        "irregular": [point_record(coef, p, x, False) for x in irregular_points],
    }


def to_json(value, indent=""):
    """JSON with one line per array of scalars: compact, but diffs stay readable."""
    inner = indent + " "
    if isinstance(value, dict):
        items = [f"{inner}{json.dumps(k)}: {to_json(v, inner)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [inner + to_json(v, inner) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + indent + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail unless the committed files are byte-identical",
    )
    args = parser.parse_args()
    if mpmath.__version__ != MPMATH_VERSION:
        sys.exit(f"needs mpmath {MPMATH_VERSION}, found {mpmath.__version__}")
    check_legendre_sign()
    version = convention_version()
    total, stale = 0, []
    jobs = [(name, seed, count, p, generate) for name, seed, count, p in SETS]
    jobs += [(name, seed, count, p, generate_irregular_values) for name, seed, count, p in IRREGULAR_VALUE_SETS]
    for name, seed, count, p, make in jobs:
        data = make(name, seed, count, p, version)
        path = OUT_DIR / f"harmonics_{name}.json"
        text = (to_json(data) + "\n").encode("utf-8")
        total += len(text)
        if args.check:
            if not path.exists() or path.read_bytes() != text:
                stale.append(str(path.relative_to(ROOT)))
            print(f"checked {path.relative_to(ROOT)} ({len(text)} bytes)")
        else:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text)} bytes)")
    print(f"total {total} bytes (limit {MAX_TOTAL_BYTES})")
    if stale:
        sys.exit(f"not byte-identical to a fresh regeneration: {', '.join(stale)}")
    if total > MAX_TOTAL_BYTES:
        sys.exit("fixtures exceed the size limit")


if __name__ == "__main__":
    main()
