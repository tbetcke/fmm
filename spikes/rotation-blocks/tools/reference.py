#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Arbitrary-precision reference entries of the orthonormal real y-rotation blocks
(Helmholtz CH0.1, spike S1; spikes/rotation-blocks/SPIKE_REPORT.md).

The block O^n(beta) of degree n maps Y_n(x) to Y_n(R_y(beta) x), with Y_n the real
storage (CONVENTIONS §3.6) of N R_n: slot 0 sqrt(n! n!) R_n^0, slot +m
sqrt(2 (n+m)! (n-m)!) Re R_n^m, slot -m the same with Im, R_n^m the Racah regular
harmonics of §3.3 (no Condon-Shortley phase) and R_y(beta) = [c 0 s; 0 1 0; -s 0 c] as in
nd_fmm_math::rotation::euler_zyz.

Method:
  - Wigner's explicit sum for d^n_{m'm}(beta) (standard phase), with
    0.61 n + 40 digits, which covers the cancellation of its terms (their absolute sum is
    at most 4^n); every value is recomputed with 50 more digits and must agree to 1e-25;
  - the map to the real block: in the complex basis sqrt((n+|m|)!(n-|m|)!) R_n^m the
    y-rotation has the entries o_mk = d^n_{km}(beta) = (-1)^(m+k) d^n_{mk}(beta), and,
    writing d_mk for o_mk, for m, k > 0
      O[+m,+k] = d_mk + (-1)^k d_m,-k,   O[-m,-k] = d_mk - (-1)^k d_m,-k,
      O[+m,0] = sqrt(2) d_m0,  O[0,+k] = (d_0k + (-1)^k d_0,-k) / sqrt(2),  O[0,0] = d_00,
    the entries between the cos (slots >= 0) and sin (slots < 0) halves zero;
  - the map is checked first at n <= 6 against the definition Y_n(Qx) = O Y_n(x), with
    R_n^m evaluated in mpmath by the Cartesian recursion of §3.5, at random points
    (1e-40 at 60 digits).

The angles are the f64 values the spike uses (the reference is computed at exactly that
double), so that only the recursion's own error is measured.

Writes spikes/rotation-blocks/fixtures/reference.json; `--check` regenerates in memory
and fails unless the committed file matches byte for byte.

Run from the repository root:  uv run spikes/rotation-blocks/tools/reference.py [--check]
"""
import json
import math
import random
import sys
from math import factorial
from pathlib import Path

import mpmath
from mpmath import mp, mpf

if mpmath.__version__ != "1.4.1":
    sys.exit(f"needs mpmath 1.4.1, found {mpmath.__version__}")

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "reference.json"
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


def generate():
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
            print(f"n={n} {name}: {len(entries)} entries", file=sys.stderr)
    doc = {
        "about": "Orthonormal real y-rotation block entries O^n(beta)[i][j], slots -n..n; "
        "f64 values as shortest round-trip decimal strings, rounded from arbitrary precision. "
        "Generated by spikes/rotation-blocks/tools/reference.py (mpmath 1.4.1).",
        "sets": sets,
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).replace("},{", "},\n{") + "\n"


def main():
    text = generate()
    if "--check" in sys.argv[1:]:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} differs from a regeneration")
        print("reference.json regenerates byte for byte", file=sys.stderr)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
