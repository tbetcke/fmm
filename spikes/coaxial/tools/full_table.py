#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""The whole recursion table in f64 against the same recursion in arbitrary precision,
in the FMM-weighted metric (Helmholtz CH0.2, spike S2; SPIKE_REPORT.md §4).

The fixture samples about 180 entries per configuration, too few to catch where the
recursion goes unstable; this compares every entry of one table, real k, a = 1. The
f64 side repeats src/coaxial.rs (triangle variant, unscaled) with Python floats; the
exact side runs the identical recursion with the coefficients and start values at
`dps` digits, checked against the Gaunt sum of tools/reference.py at three entries.

  uv run spikes/coaxial/tools/full_table.py SR <ka> <d^2> [dps]   # M2L at distance d w
  uv run spikes/coaxial/tools/full_table.py RR <ka> 0 [dps]       # M2M, child shift a/2
"""
import importlib.util
import math
import sys
from pathlib import Path

import mpmath
from mpmath import mp, mpc, mpf

spec = importlib.util.spec_from_file_location("reference", Path(__file__).with_name("reference.py"))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def coefficients(sqrt, num):
    """a, beta, gamma of src/coaxial.rs with `sqrt` and `num` (float or mpf)."""

    def a(n, m):
        return 0 if n + 1 <= m else sqrt(num((n + 1 + m) * (n + 1 - m)) / ((2 * n + 1) * (2 * n + 3)))

    def beta(n, m):
        return 0 if n < m + 2 else sqrt(num((n - m - 1) * (n - m)) / ((2 * n - 1) * (2 * n + 1)))

    def gamma(n, m):
        return sqrt(num((n + m + 1) * (n + m + 2)) / ((2 * n + 1) * (2 * n + 3)))

    return a, beta, gamma


def triangle(start, ps, pt, sqrt, num):
    """{(m, n', n): E} for n' >= n, n' <= max(ps, pt), n <= min(ps, pt)."""
    a, beta, gamma = coefficients(sqrt, num)
    L, R, C = ps + pt, max(ps, pt), min(ps, pt)
    out = {}
    sect = [(-1) ** n1 * sqrt(num(2 * n1 + 1)) * start[n1] for n1 in range(L + 1)] + [0, 0]
    for m in range(C + 1):
        if m > 0:
            new = [0] * (L + 3)
            for n1 in range(m, L - m + 1):
                new[n1] = (sect[n1 + 1] * beta(n1 + 1, m - 1) + sect[n1 - 1] * gamma(n1 - 1, m - 1)) / gamma(m - 1, m - 1)
            sect = new
        prev, cur = [0] * (L + 3), list(sect)
        for n1 in range(m, R + 1):
            out[(m, n1, m)] = cur[n1]
        for n in range(m, C):
            nxt = [0] * (L + 3)
            for n1 in range(n + 1, L - n):
                lower = a(n1 - 1, m) if n1 >= 1 else 0
                nxt[n1] = (prev[n1] * (a(n - 1, m) if n >= 1 else 0) - cur[n1 + 1] * a(n1, m) + cur[n1 - 1] * lower) / a(n, m)
            for n1 in range(n + 1, R + 1):
                out[(m, n1, n + 1)] = nxt[n1]
            prev, cur = cur, nxt
    return out


def get(table, m, n1, n):
    if n1 >= n:
        return table[(m, n1, n)]
    v = table[(m, n, n1)]
    return -v if (n + n1) % 2 else v


def main():
    op, ka, d2 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
    mp.dps = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    if op == "SR":
        kt = ka * 2 * math.sqrt(d2) / math.sqrt(3.0)
        ps = pt = ref.degree(ka, 1e-10)
    else:
        kt = ka / 2.0
        ps, pt = ref.degree(ka / 2.0, 1e-10), ref.degree(ka, 1e-10)
    f = ref.radial(op, mpc(kt, 0), ps + pt)
    exact = triangle(f, ps, pt, mpmath.sqrt, mpf)
    approx = triangle([complex(v) for v in f], ps, pt, math.sqrt, float)
    gaunt = max(
        abs(get(exact, m, n1, n) - ref.gaunt(n1, n, m, f)) / abs(ref.gaunt(n1, n, m, f))
        for m, n1, n in [(0, pt, ps), (min(ps, pt) // 2, pt, ps), (min(ps, pt) // 3, pt - 3, ps - 5)]
    )

    def bessel_weight(x, n):
        return float(abs(mpmath.sqrt(mpmath.pi / (2 * x)) * mpmath.besselj(n + 0.5, x))) * math.sqrt(2 * n + 1)

    def hankel_weight(x, n):
        return float(abs(mpmath.sqrt(mpmath.pi / (2 * x)) * mpmath.hankel1(n + 0.5, x))) * math.sqrt(2 * n + 1)

    if op == "SR":
        w = [bessel_weight(mpf(ka), n) for n in range(ps + 1)]
        weight = lambda n1, n: w[n] * w[n1]  # noqa: E731
    else:
        wc = [bessel_weight(mpf(ka) / 2, n) for n in range(ps + 1)]
        wp = [hankel_weight(mpf(ka) * 4 / mpmath.sqrt(3), n) for n in range(pt + 1)]
        weight = lambda n1, n: wc[n] * wp[n1]  # noqa: E731
    num = den = 0.0
    for m in range(min(ps, pt) + 1):
        for n1 in range(m, pt + 1):
            for n in range(m, ps + 1):
                e = complex(get(exact, m, n1, n))
                w_ = weight(n1, n)
                num = max(num, abs(e - get(approx, m, n1, n)) * w_)
                den = max(den, abs(e) * w_)
    print(f"{op} ka={ka} kt={kt:.1f} p {ps} -> {pt}: exact recursion against the Gaunt sum "
          f"{float(gaunt):.1e}; f64 weighted error {num / den:.2e}")


if __name__ == "__main__":
    main()
