#!/usr/bin/env python3
"""Seed for spike S6 (helmholtz-fmm-proposal.md, Section 3.6 and Appendix B.1).

Axisymmetric check of the diagonal M2L as the proposal specifies it: signatures of
bandwidth P - 1, transfer functions of bandwidth 2P - 2, Gauss-Legendre quadrature with
2P - 1 nodes in cos(theta), one-box separation (nearest offset |t| = 2w), a = (sqrt 3 / 2) w.
Sources and targets sit on the translation axis at radius a on the far sides, so the
truncation error is benign and what remains is the rounding of the transfer function.

This is a 1-D model of a 3-D question. S6 replaces it with the full chain on the 34 offset
classes, realistic point placements, f32 as well as f64, and complex k.

Requires numpy, scipy, mpmath. Runs in about a minute.
"""
import math

import mpmath as mp
import numpy as np
from scipy.special import eval_legendre, spherical_jn

mp.mp.dps = 40
U_F64 = 1.1e-16


def truncation(ka, eps, delta=2.0):
    """Gumerov-Duraiswami (2009, eq. 18): returns (P, p_hi); P counts degrees 0..P-1."""
    p_hi = ka + (1.5 * math.log(1 / eps)) ** (2 / 3) * ka ** (1 / 3)
    p_lo = 1 - math.log(eps * (1 - 1 / delta) ** 1.5) / math.log(delta)
    return (p_lo**4 + p_hi**4) ** 0.25, p_hi


def hankel(l, x):
    """Spherical Hankel function of the first kind, in extended precision."""
    f = mp.sqrt(mp.pi / (2 * x))
    return f * (mp.besselj(l + 0.5, x) + 1j * mp.bessely(l + 0.5, x))


def diagonal_error(kw, eps, t_in_w=2.0):
    """Relative error of the f64 diagonal M2L against the exact kernel (units k = 1)."""
    a = math.sqrt(3) / 2 * kw
    P = math.ceil(truncation(a, eps)[0])
    L = 2 * P - 2
    kt = t_in_w * kw
    mu, wq = np.polynomial.legendre.leggauss(2 * P - 1)
    # transfer function: terms computed in extended precision, rounded to f64, summed in f64
    h = np.array([complex(hankel(l, kt)) for l in range(L + 1)])
    Pl = np.array([eval_legendre(l, mu) for l in range(L + 1)])
    T = ((2 * np.arange(L + 1) + 1) * (1j) ** np.arange(L + 1) * h) @ Pl
    n = np.arange(P)

    def plane_wave(z):  # e^{i mu z} band-limited to degree P - 1
        return ((2 * n + 1) * (1j) ** n * spherical_jn(n, abs(z)) * np.sign(z) ** n) @ Pl[:P]

    u = (1j / (16 * math.pi**2)) * 2 * math.pi * np.sum(wq * T * plane_wave(a) * plane_wave(a))
    r = kt + 2 * a
    exact = np.exp(1j * r) / (4 * math.pi * r)
    return P, abs(u - exact) / abs(exact)


def bound(ka, eps, t_in_w=2.0, P=None, u=U_F64, c_t=2.0):
    """E(l, c) of Section 3.6: C_T * k|t| * u * max_j (2j + 1) |h_j(k|t|)|, j <= 2P - 2."""
    P = P or math.ceil(truncation(ka, eps)[0])
    L = 2 * P - 2
    kt = t_in_w * 2 * ka / math.sqrt(3)
    log_max = max(math.log(2 * j + 1) + float(mp.log(abs(hankel(j, kt))))
                  for j in range(max(0, L - 40), L + 1))
    return c_t * kt * u * math.exp(log_max)


def draft06_switch(eps):
    """k a at which P - p_hi <= 2 (the level switch of draft 0.6)."""
    lo, hi = 0.01, 500.0
    for _ in range(100):
        mid = (lo + hi) / 2
        P, p_hi = truncation(mid, eps)
        lo, hi = (lo, mid) if P - p_hi <= 2 else (mid, hi)
    return hi


def stable_from(eps, eps_formula=None, t_in_w=2.0):
    """Smallest k a from which the bound at offset |t| meets eps."""
    eps_formula = eps_formula or eps
    lo, hi = 1.0, 3000.0
    for _ in range(40):
        mid = math.sqrt(lo * hi)
        P = math.ceil(truncation(mid, eps_formula)[0])
        lo, hi = (lo, mid) if bound(mid, eps, t_in_w, P) <= eps else (mid, hi)
    return hi


def wavelengths(ka):
    return 2 * ka / math.sqrt(3) / (2 * math.pi)


if __name__ == "__main__":
    print("Direct test, kD = 500, eps = 1e-6 (relative error at |t| = 2w and 3w)")
    for level in range(2, 6):
        kw = 500 / 2**level
        P, e2 = diagonal_error(kw, 1e-6, 2.0)
        _, e3 = diagonal_error(kw, 1e-6, 3.0)
        print(f"  level {level}: box {kw / (2 * math.pi):5.1f} lambda, P = {P:3d}: "
              f"{e2:.1e} at 2w, {e3:.1e} at 3w, bound {bound(math.sqrt(3) / 2 * kw, 1e-6):.1e}")
    print("\nSwitch of draft 0.6 against the bound (nearest class)")
    for eps in (1e-4, 1e-6, 1e-8):
        s = draft06_switch(eps)
        print(f"  eps {eps:.0e}: switch {wavelengths(s):4.1f} lambda, bound there "
              f"{bound(s, eps):.1e}, stable from {wavelengths(stable_from(eps)):5.1f} lambda")
    print("\nCalibrated P (formula at a looser eps)")
    for eps, eps_f in ((1e-6, 1e-4), (1e-8, 1e-6)):
        print(f"  eps {eps:.0e} with the {eps_f:.0e} formula: stable from "
              f"{wavelengths(stable_from(eps, eps_f)):5.1f} lambda")
