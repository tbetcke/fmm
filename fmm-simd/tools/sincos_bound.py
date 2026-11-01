#!/usr/bin/env python3
"""The first-order error bound of `arch::sincos` (Helmholtz CH3.4a;
docs/design/simd-p2p.md §4.11), absolute, in units of u_T, for f32 and f64.

For t on a grid of [0, π/2 + 10⁻⁴] it sums, with the weights of the design's derivation:

- the reduction: ½ ulp(t) (plus |n| |π − π₁ − π₂|, 0.012 u in f32), times |cos t| for the
  sine and |sin t| for the cosine;
- the sine: Horner's roundings in P, u Σⱼ |pⱼ| sʲ, times |t s|; the roundings of s and t s,
  2 u |t s P(s)|; the coefficients' actual rounding errors; the final ½ ulp(sin t);
- the cosine: Horner's roundings in Q, u Σⱼ |qⱼ| sʲ (the final one included); s's rounding,
  |t sin t| / 2; the coefficients' actual rounding errors; the truncation.

It prints the largest bound of each. Run: python3 fmm-simd/tools/sincos_bound.py
"""
import math
import struct
from fractions import Fraction
from math import factorial


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def half_ulp(x, p):
    """½ ulp(|x|) in units of u = 2^-p."""
    return 0.0 if x == 0 else 2.0 ** math.floor(math.log2(abs(x)))


def horner_partials(c, s):
    """|q_j| of Horner over c (constant term first) at s, from the top."""
    q = c[-1]
    parts = [abs(q)]
    for ci in reversed(c[:-1]):
        q = q * s + ci
        parts.append(abs(q))
    return q, parts


def bound(n_s, n_c, rnd, p, pi_error):
    u = 2.0**-p
    sin_c = [(-1) ** j / factorial(2 * j + 1) for j in range(1, n_s + 1)]
    cos_c = [(-1) ** j / factorial(2 * j) for j in range(0, n_c + 1)]
    worst_sin = worst_cos = 0.0
    for i in range(1, 4001):
        t = i / 4000 * (math.pi / 2 + 1e-4)
        s = t * t
        reduction = half_ulp(t, p) + pi_error
        q, parts = horner_partials(sin_c, s)
        horner = sum(pj * s ** (len(parts) - 1 - k) for k, pj in enumerate(parts))
        coefficients = sum(
            abs(float(Fraction(rnd(sin_c[j])) - Fraction((-1) ** (j + 1), factorial(2 * j + 3))))
            / u
            * t ** (2 * j + 3)
            for j in range(n_s)
        )
        e_sin = (
            reduction * abs(math.cos(t))
            + abs(t * s) * horner
            + 2 * abs(t * s * q)
            + coefficients
            + half_ulp(math.sin(t), p)
        )
        _, parts = horner_partials(cos_c, s)
        horner = sum(pj * s ** (len(parts) - 1 - k) for k, pj in enumerate(parts))
        coefficients = sum(
            abs(float(Fraction(rnd(cos_c[j])) - Fraction((-1) ** j, factorial(2 * j)))) / u * t ** (2 * j)
            for j in range(n_c + 1)
        )
        truncation = t ** (2 * n_c + 2) / factorial(2 * n_c + 2) / u
        e_cos = (
            reduction * abs(math.sin(t))
            + horner
            + abs(t * math.sin(t)) / 2
            + coefficients
            + truncation
        )
        worst_sin = max(worst_sin, e_sin)
        worst_cos = max(worst_cos, e_cos)
    return worst_sin, worst_cos


for name, args in (("f64", (10, 10, float, 53, 0.0)), ("f32", (6, 6, f32, 24, 0.012))):
    s, c = bound(*args)
    print(f"{name}: sin at most {s:.3f} u, cos at most {c:.3f} u")
