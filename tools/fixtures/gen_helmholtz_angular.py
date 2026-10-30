#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""High-precision fixtures for the orthonormal harmonics and rotation blocks of
docs/CONVENTIONS.md §3.15 ("Angular basis", "Rotations"), Helmholtz CH1.2a.

Writes fmm-math/fixtures/helmholtz_angular.json with two parts:

  - "rotation": sampled entries of the real y-rotation blocks O^n(beta) of degrees 10 to
    1000 at six polar angles. The method is spike S1's (spikes/rotation-blocks/tools/
    reference.py, whose sets these are, value for value): Wigner's explicit sum for
    d^n_{m'm}(beta) at 0.61 n + 40 digits, each value recomputed with 50 more digits
    (agreement to 1e-25), mapped to the real block; the map is first checked at n <= 6
    against the definition Y_n(Qx) = O Y_n(x) (1e-40 at 60 digits).
  - "harmonics": Y_n^m(x), d/dtheta Y_n^m and (1/sin theta) d/dphi Y_n^m at seven
    directions (generic, on both poles, near both poles, on the equator, below it) for
    degrees 0 to 1000, the orders sampled as for the blocks. The normalised P-bar_n^m
    come from the plain three-term recursion in n at high precision (not the telescoped
    form of the Rust code); P-bar_n^m / sin theta at the poles from the limit of d/dx P_n;
    d/dtheta from the ladder of the normalised functions, which the script first checks
    against mpmath's numerical derivative at a generic angle. Every value is computed at
    two precisions (DPS and DPS + 40) that must agree to 1e-30 relative to the degree's
    largest value.

The inputs (angles, directions) are doubles; the values are those of exactly these
doubles, rounded to the nearest double and written as shortest round-trip strings.

Run from the repository root:  uv run tools/fixtures/gen_helmholtz_angular.py
(or python3 with mpmath==1.4.1). Regeneration is byte-identical; --check compares with
the committed file. See tools/fixtures/README.md.
"""
import argparse
import json
import math
import random
import re
import sys
from math import factorial
from pathlib import Path

import mpmath
from mpmath import mp, mpf

if mpmath.__version__ != "1.4.1":
    sys.exit(f"needs mpmath 1.4.1, found {mpmath.__version__}")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fmm-math" / "fixtures" / "helmholtz_angular.json"
CONVENTIONS = ROOT / "docs" / "CONVENTIONS.md"
GENERATOR_VERSION = 1
SEED = 20261027
DEGREES = [10, 30, 50, 100, 200, 300, 500, 700, 1000]
# Name, f64 value. The octree's: M2M's child shift and its supplement, the smallest
# off-axis polar angle of the V list (offset (1, 0, 3)), the equator; two generic ones.
ANGLES = [
    ("m2m", math.acos(1.0 / math.sqrt(3.0))),
    ("m2m_supplement", math.pi - math.acos(1.0 / math.sqrt(3.0))),
    ("v_smallest", math.atan(1.0 / 3.0)),
    ("equator", math.pi / 2.0),
    ("generic", 1.234),
    ("small", 1e-3),
]
LINE_SAMPLES = 41
RANDOM_SAMPLES = 60
AGREE = mpf(10) ** -25

H_DEGREES = [0, 1, 2, 3, 5, 8, 13, 20, 30, 50, 100, 300, 1000]
# Name, the direction as a double triple (not normalised).
DIRECTIONS = [
    ("generic", [0.31, -0.72, 0.55]),
    ("north", [0.0, 0.0, 2.5]),
    ("south", [0.0, 0.0, -0.3]),
    ("near_north", [1e-3, 2e-4, 1.0]),
    ("near_south", [-3e-4, 7e-4, -1.0]),
    ("equator", [0.6, -0.8, 0.0]),
    ("below", [-0.5, 0.2, -0.8]),
]
DPS = 60
H_AGREE = mpf(10) ** -30


def convention_version():
    found = set(re.findall(r"`CONVENTION_VERSION = (\d+)`", CONVENTIONS.read_text()))
    if len(found) != 1:
        sys.exit(f"expected one CONVENTION_VERSION in {CONVENTIONS}, found {sorted(found)}")
    return int(found.pop())


# ---------------------------------------------------------------- Wigner's sum


def wigner_d(n, mp_, m, beta):
    """d^n_{mp_, m}(beta), standard phase, at the current precision."""
    half = beta / 2
    c, s = mpmath.cos(half), mpmath.sin(half)
    s_lo = max(0, m - mp_)
    s_hi = min(n + m, n - mp_)
    if s_lo > s_hi:
        return mpf(0)
    # First term exactly in integers, then the term ratios.
    num = factorial(n + mp_) * factorial(n - mp_) * factorial(n + m) * factorial(n - m)
    den = (
        factorial(n + m - s_lo)
        * factorial(s_lo)
        * factorial(mp_ - m + s_lo)
        * factorial(n - mp_ - s_lo)
    )
    sign = -1 if (mp_ - m + s_lo) % 2 else 1
    term = sign * mpmath.sqrt(mpf(num)) / mpf(den)
    term *= c ** (2 * n + m - mp_ - 2 * s_lo) * s ** (mp_ - m + 2 * s_lo)
    t2 = (s / c) ** 2 if c != 0 else None
    total = term
    for k in range(s_lo, s_hi):
        if t2 is None:
            # cos(beta/2) = 0: only the term with the largest power of sin survives.
            raise ValueError("beta = pi is not sampled")
        ratio = mpf(-(n + m - k) * (n - mp_ - k)) / ((k + 1) * (mp_ - m + k + 1))
        term *= ratio * t2
        total += term
    return total


def d_checked(n, mp_, m, beta_f64):
    """d^n_{mp_, m} at two precisions; returns the value or exits on disagreement."""
    dps = int(0.61 * n) + 40
    values = []
    for extra in (0, 50):
        with mp.workdps(dps + extra):
            values.append(wigner_d(n, mp_, m, mpf(beta_f64)))
    with mp.workdps(dps + 50):
        if abs(values[0] - values[1]) > AGREE:
            sys.exit(f"precision check failed: n={n} m'={mp_} m={m} beta={beta_f64!r}")
    return values[1]


def real_entry(n, i, j, d):
    """Entry (i, j) of the real block, slots -n..n, from d(m, k) = o_mk, the complex
    orthonormal entry (module docstring)."""
    if (i >= 0) != (j >= 0):
        return mpf(0)
    sqrt2 = mpmath.sqrt(2)
    if i == 0 and j == 0:
        return d(0, 0)
    if j == 0:
        return sqrt2 * d(i, 0)
    if i == 0:
        return (d(0, j) + (-1) ** j * d(0, -j)) / sqrt2
    m, k = abs(i), abs(j)
    if i > 0:
        return d(m, k) + (-1) ** k * d(m, -k)
    return d(m, k) - (-1) ** k * d(m, -k)


# ------------------------------------------------- the map checked at low degree


def racah_regular(p, x):
    """R_n^m, 0 <= m <= n <= p, as complex mpf pairs by the recursion of §3.5."""
    X, Y, Z = x
    r2 = X * X + Y * Y + Z * Z
    R = {(0, 0): (mpf(1), mpf(0))}
    for m in range(1, p + 1):
        a, b = R[(m - 1, m - 1)]
        R[(m, m)] = ((X * a - Y * b) / (2 * m), (X * b + Y * a) / (2 * m))
    for m in range(0, p):
        a, b = R[(m, m)]
        R[(m + 1, m)] = (Z * a, Z * b)
        for n in range(m + 2, p + 1):
            a1, b1 = R[(n - 1, m)]
            a2, b2 = R[(n - 2, m)]
            k = (n + m) * (n - m)
            R[(n, m)] = (
                ((2 * n - 1) * Z * a1 - r2 * a2) / k,
                ((2 * n - 1) * Z * b1 - r2 * b2) / k,
            )
    return R


def real_orthonormal(n, x):
    """Y_n(x), slots -n..n."""
    R = racah_regular(n, x)
    out = []
    for slot in range(-n, n + 1):
        m = abs(slot)
        w = mpmath.sqrt(mpf(factorial(n + m) * factorial(n - m)))
        re, im = R[(n, m)]
        if slot == 0:
            out.append(w * re)
        elif slot > 0:
            out.append(mpmath.sqrt(2) * w * re)
        else:
            out.append(mpmath.sqrt(2) * w * im)
    return out


def check_map():
    rng = random.Random(SEED)
    worst = mpf(0)
    with mp.workdps(60):
        for beta in [mpf("0.7"), mpf("2.3"), mpf("-1.1")]:
            c, s = mpmath.cos(beta), mpmath.sin(beta)
            for n in range(0, 7):

                def d(a, b, n=n, beta=beta):
                    return wigner_d(n, b, a, beta)

                block = [[real_entry(n, i, j, d) for j in range(-n, n + 1)] for i in range(-n, n + 1)]
                for _ in range(3):
                    x = [mpf(rng.uniform(-1, 1)) for _ in range(3)]
                    qx = [c * x[0] + s * x[2], x[1], -s * x[0] + c * x[2]]
                    y, yq = real_orthonormal(n, x), real_orthonormal(n, qx)
                    for row, expected in zip(block, yq):
                        value = mpmath.fsum(a * b for a, b in zip(row, y))
                        worst = max(worst, abs(value - expected))
        if worst > mpf(10) ** -40:
            sys.exit(f"map check failed: worst {mpmath.nstr(worst, 3)}")
    print(f"map checked at n <= 6: worst {mpmath.nstr(worst, 3)}", file=sys.stderr)


# --------------------------------------------------------------------- sampling


def line(n):
    """At most LINE_SAMPLES indices in -n..n, both ends and 0 included."""
    if 2 * n + 1 <= LINE_SAMPLES:
        return list(range(-n, n + 1))
    step = (2 * n) / (LINE_SAMPLES - 1)
    picks = {-n, 0, n} | {round(-n + t * step) for t in range(LINE_SAMPLES)}
    return sorted(picks)


def samples(n, rng):
    """Nonzero (same-half) entries: rows 0, n, -n; columns 0, n, -n; the diagonal and
    the antidiagonal of each half; random entries."""
    picks = set()
    for t in line(n):
        for fixed in (0, n, -n):
            picks.add((fixed, t))
            picks.add((t, fixed))
        picks.add((t, t))
        if t >= 0:
            picks.add((t, n - t))
        else:
            picks.add((t, -(n + 1) - t))
    for _ in range(RANDOM_SAMPLES):
        i = rng.randint(-n, n)
        j = rng.randint(0, n) if i >= 0 else rng.randint(-n, -1)
        picks.add((i, j))
    return sorted((i, j) for i, j in picks if (i >= 0) == (j >= 0))


# ----------------------------------------------------------------- harmonics


_COLUMNS = {}


def pbar_column(m, c, s, top):
    """P-bar_n^m(c), n = m..top, by the plain three-term recursion in n at the current
    precision (independent of the telescoped form the Rust code uses):
    P-bar_m^m = prod sqrt((2i - 1)/(2i)) s^m, P-bar_{m+1}^m = sqrt(2m + 1) c P-bar_m^m,
    P-bar_n^m = ((2n - 1) c P-bar_{n-1}^m - sqrt((n-1-m)(n-1+m)) P-bar_{n-2}^m) / sqrt((n-m)(n+m))."""
    key = (mp.prec, c, s, m)
    column = _COLUMNS.get(key)
    if column is not None and len(column) > top - m:
        return column
    diag = mpf(1)
    for i in range(1, m + 1):
        diag *= mpmath.sqrt(mpf(2 * i - 1) / (2 * i)) * s
    column = [diag]
    if top > m:
        column.append(mpmath.sqrt(2 * m + 1) * c * diag)
    for n in range(m + 2, top + 1):
        a = (2 * n - 1) * c * column[-1]
        b = mpmath.sqrt(mpf((n - 1 - m) * (n - 1 + m))) * column[-2]
        column.append((a - b) / mpmath.sqrt(mpf((n - m) * (n + m))))
    _COLUMNS[key] = column
    return column


def angles_of(point):
    """(cos theta, sin theta, phi) of a double triple; phi = 0 on the z-axis."""
    x, y, z = (mpf(c) for c in point)
    rho = mpmath.sqrt(x * x + y * y)
    r = mpmath.sqrt(rho * rho + z * z)
    phi = mpmath.atan2(y, x) if rho != 0 else mpf(0)
    return z / r, rho / r, phi


def pbar(n, m, c, s):
    """P-bar_n^m(cos theta) = sqrt((n-m)!/(n+m)!) P_n^m, no Condon-Shortley phase."""
    if m < 0 or m > n:
        return mpf(0)
    top = max(H_DEGREES) + 1 if n > 10 else 11
    return pbar_column(m, c, s, top)[n - m]


def pbar_over_sin(n, m, c, s):
    """P-bar_n^m / sin theta for m >= 1, with its limit on the axis."""
    if m < 1 or m > n:
        return mpf(0)
    if s != 0:
        return pbar(n, m, c, s) / s
    if m > 1:
        return mpf(0)
    # P_n^1 / s -> d/dx P_n(+-1) = (+-1)^(n+1) n(n+1)/2.
    norm = mpmath.sqrt(mpf(factorial(n - 1)) / mpf(factorial(n + 1)))
    return norm * c ** (n + 1) * mpf(n * (n + 1)) / 2


def dtheta_pbar(n, m, c, s):
    """d/dtheta P-bar_n^m by the ladder: -sqrt(n(n+1)) P-bar^1 for m = 0, else
    (sqrt((n+m)(n-m+1)) P-bar^{m-1} - sqrt((n-m)(n+m+1)) P-bar^{m+1}) / 2."""
    if m == 0:
        return -mpmath.sqrt(mpf(n * (n + 1))) * pbar(n, 1, c, s)
    up = mpmath.sqrt(mpf((n + m) * (n - m + 1))) * pbar(n, m - 1, c, s)
    down = mpmath.sqrt(mpf((n - m) * (n + m + 1))) * pbar(n, m + 1, c, s)
    return (up - down) / 2


def ylm_parts(n, slot, c, s, phi):
    """(Y, d/dtheta Y, (1/sin theta) d/dphi Y) of slot -n..n."""
    m = abs(slot)
    norm = mpmath.sqrt(mpf(2 * n + 1) / (4 * mpmath.pi)) * (mpmath.sqrt(2) if m else 1)
    if slot >= 0:
        ang, dang = mpmath.cos(m * phi), -m * mpmath.sin(m * phi)
    else:
        ang, dang = mpmath.sin(m * phi), m * mpmath.cos(m * phi)
    value = norm * pbar(n, m, c, s) * ang
    dtheta = norm * dtheta_pbar(n, m, c, s) * ang
    dphi = norm * pbar_over_sin(n, m, c, s) * dang if m else mpf(0)
    return value, dtheta, dphi


def check_ladder():
    """The ladder for d/dtheta against mpmath's numerical derivative, n <= 8."""
    worst = mpf(0)
    with mp.workdps(50):
        theta0 = mpf("0.83")
        for n in range(0, 9):
            for m in range(0, n + 1):
                def f(t, n=n, m=m):
                    return pbar(n, m, mpmath.cos(t), mpmath.sin(t))
                d = mpmath.diff(f, theta0)
                ladder = dtheta_pbar(n, m, mpmath.cos(theta0), mpmath.sin(theta0))
                worst = max(worst, abs(d - ladder))
    if worst > mpf(10) ** -35:
        sys.exit(f"ladder check failed: {mpmath.nstr(worst, 3)}")
    print(f"ladder checked at n <= 8: worst {mpmath.nstr(worst, 3)}", file=sys.stderr)


def harmonic_sets():
    check_ladder()
    sets = []
    for name, point in DIRECTIONS:
        for n in H_DEGREES:
            slots = line(n)
            parts = []
            for extra in (0, 40):
                with mp.workdps(DPS + extra):
                    c, s, phi = angles_of(point)
                    parts.append([ylm_parts(n, slot, c, s, phi) for slot in slots])
            with mp.workdps(DPS + 40):
                scale = max(max(abs(v) for v in triple) for triple in parts[1]) or mpf(1)
                for a, b in zip(parts[0], parts[1]):
                    for u, v in zip(a, b):
                        if abs(u - v) > H_AGREE * scale:
                            sys.exit(f"precision check failed: {name} n={n}")
            sets.append(
                {
                    "direction": name,
                    "x": [repr(float(c)) for c in point],
                    "n": n,
                    "m": slots,
                    "value": [repr(float(t[0])) for t in parts[1]],
                    "dtheta": [repr(float(t[1])) for t in parts[1]],
                    "dphi": [repr(float(t[2])) for t in parts[1]],
                }
            )
            print(f"harmonics {name} n={n}: {len(slots)} slots", file=sys.stderr)
    return sets


# ---------------------------------------------------------------- rotation blocks


def rotation_sets():
    check_map()
    rng = random.Random(SEED)
    sets = []
    for n in DEGREES:
        for name, beta in ANGLES:
            cache = {}

            def d(a, b, n=n, beta=beta, cache=cache):
                if (a, b) not in cache:
                    cache[(a, b)] = d_checked(n, b, a, beta)
                return cache[(a, b)]

            entries = samples(n, rng)
            values = []
            for i, j in entries:
                with mp.workdps(int(0.61 * n) + 90):
                    values.append(float(real_entry(n, i, j, d)))
            sets.append(
                {
                    "n": n,
                    "angle": name,
                    "beta": repr(beta),
                    "i": [i for i, _ in entries],
                    "j": [j for _, j in entries],
                    "value": [repr(v) for v in values],
                }
            )
            print(f"rotation n={n} {name}: {len(entries)} entries", file=sys.stderr)
    return sets


def generate():
    doc = {
        "header": {
            "description": (
                "Real orthonormal harmonics Y_n^m, d/dtheta Y_n^m and (1/sin theta) "
                "d/dphi Y_n^m, and entries of the real y-rotation blocks O^n(beta) "
                "(CONVENTIONS §3.15, Angular basis and Rotations; Helmholtz CH1.2a)"
            ),
            "convention_version": convention_version(),
            "generator": "tools/fixtures/gen_helmholtz_angular.py",
            "generator_version": GENERATOR_VERSION,
            "mpmath": mpmath.__version__,
            "seed": SEED,
            "format": (
                "decimal strings, shortest round-trip doubles rounded from arbitrary "
                "precision; harmonics per direction (x, a double triple) and degree n at "
                "the slots m (-n..n, cos in +m, sin in -m); rotation entries (i, j), "
                "slots -n..n, of the block of degree n at the polar angle beta"
            ),
        },
        "harmonics": harmonic_sets(),
        "rotation": rotation_sets(),
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).replace("},{", "},\n{") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare with the committed file")
    args = parser.parse_args()
    text = generate()
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} differs from a regeneration")
        print(f"checked {OUT.relative_to(ROOT)} ({len(text)} bytes): byte-identical", file=sys.stderr)
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
