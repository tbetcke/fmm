#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Numerical check of docs/CONVENTIONS.md §3.15, "Exponential M2L" (Helmholtz CH6.1c).

Every check works from the definitions written there, in mpmath at DPS digits, on seeded
points, for real, complex and imaginary kappa = k w (box widths), and fails (exit 1) above
its tolerance. No quadrature table is used: the lambda integral is taken numerically on the
contour of real t (lambda = -i sqrt(kappa^2 - t^2), Im >= 0; d lambda = t dt / lambda), the
alpha integral in closed form, (1/2pi) int cos(m a) e^{i t rho cos(a - phi)} da =
i^m J_m(t rho) cos(m phi), and sin likewise. It checks:

  1. the representation h_n(kappa|u|) Y_n^m(u^) = 1/(i^{n+1} kappa) int d lambda (1/2pi)
     int d alpha Y_n^m(s^) e^{i kappa s^.u}, up (u_z > 0) and down (u_z < 0, with
     s^_z = -i lambda/kappa), n <= 5, every m;
  2. Y_n^{+-m}(s^) = kappa^{-n} c_nm Pbar_n^m(i lambda, t; kappa^2) {cos, sin}(m alpha) by
     the solid recursion, against the real harmonics at real directions;
  3. the plane-wave expansion e^{i kappa s^.u} = 4 pi sum_n i^n j_n(kappa|u|) sum_m
     Y_n^m(s^) Y_n^m(u^) at complex s^;
  4. the stored forms: the up waves of a stored multipole M^ (P2M of seeded charges) give
     its r phi~ at points above the box, and the local of one wave by the factors
     4 pi e^beta (-1)^n P_k(n, m) evaluates, by L2P in stored form, to the wave itself,
     both in the level's units (half-width 1, k^ = kappa/2, s = min(|k^| sqrt3, 1),
     beta = Im(k^) sqrt3).

Run from the repository root:  uv run tools/fixtures/check_3_15_plane_wave.py
(or python3 with mpmath==1.4.1). It takes about ten minutes.
"""
import random
import sys

import mpmath as mp

DPS = 30
mp.mp.dps = DPS
I = mp.mpc(0, 1)
FAILURES = []


def check(name, err, tol):
    status = "ok" if err <= tol else "FAIL"
    print(f"  {status:4} {name}: {mp.nstr(err, 3)} (tolerance {mp.nstr(tol, 2)})")
    if err > tol:
        FAILURES.append(name)


def solid(p, z, rho, r2):
    """c_nm Pbar_n^m(z, rho; r2), 0 <= m <= n <= p, as a dict (n, m) -> value."""
    out = {}
    start = mp.mpf(1)
    for m in range(p + 1):
        if m > 0:
            start = start * rho * mp.sqrt(mp.mpf(2 * m - 1) / (2 * m))
        prev, cur = mp.mpf(0), start
        for n in range(m, p + 1):
            if n == m + 1:
                prev, cur = cur, z * cur * mp.sqrt(2 * m + 1)
            elif n > m + 1:
                c = lambda q: mp.sqrt((q - m) * (q + m))
                prev, cur = cur, (z * cur * (2 * n - 1) - r2 * prev * c(n - 1)) / c(n)
            norm = 1 if m == 0 else 2
            out[(n, m)] = cur * mp.sqrt(norm * (2 * n + 1) / (4 * mp.pi))
    return out


def real_y(n, m, x):
    """Y_n^m of §3.15 at the real direction of x, from the associated Legendre function."""
    r = mp.sqrt(sum(c * c for c in x))
    ct = x[2] / r
    phi = mp.atan2(x[1], x[0])
    am = abs(m)
    # mpmath's legenp has the Condon-Shortley phase for integer order: remove it.
    pnm = mp.legenp(n, am, ct) * (-1) ** am
    c = mp.sqrt((1 if am == 0 else 2) * (2 * n + 1) / (4 * mp.pi)
                * mp.factorial(n - am) / mp.factorial(n + am))
    trig = mp.cos(am * phi) if m >= 0 else mp.sin(am * phi)
    return c * pnm * trig


def hn(n, z):
    return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z) + \
        I * mp.sqrt(mp.pi / (2 * z)) * mp.bessely(n + mp.mpf(1) / 2, z)


def jn(n, z):
    return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mp.mpf(1) / 2, z)


def lam_integral(kappa, f):
    """int over Gamma of f(lambda, t) d lambda, on the contour of real t."""
    def g(t):
        q = mp.sqrt(kappa * kappa - t * t)
        if mp.im(q) < 0:
            q = -q
        lam = -I * q
        return f(lam, t) * t / lam
    a = mp.re(kappa)
    if mp.im(kappa) == 0:
        # Real kappa: t = kappa sin(th), lambda = -i kappa cos(th), t dt / lambda =
        # i kappa sin(th) d th; then t = kappa cosh(ph), lambda = kappa sinh(ph),
        # t dt / lambda = kappa cosh(ph) d ph (no 1/sqrt at t = kappa).
        part1 = mp.quad(lambda th: f(-I * a * mp.cos(th), a * mp.sin(th)) * I * a * mp.sin(th),
                        [0, mp.pi / 4, mp.pi / 2])
        part2 = mp.quad(lambda ph: f(a * mp.sinh(ph), a * mp.cosh(ph)) * a * mp.cosh(ph),
                        [0, 1, 3, 6, 9])
        return part1 + part2
    if a > 0:
        return mp.quad(g, [0, a / 2, a, 2 * a, 6 * a + 4, 40, 160])
    return mp.quad(g, [0, 1, 4, 12, 40, 160])


def representation(kappa, n, m, u, up):
    """The right-hand side of item 1 at u (box widths)."""
    x, y, z = u
    rho = mp.sqrt(x * x + y * y)
    phi = mp.atan2(y, x)
    am = abs(m)
    trig = mp.cos(am * phi) if m >= 0 else mp.sin(am * phi)
    sign = 1 if up else -1

    def f(lam, t):
        fac = solid(n, sign * I * lam / kappa, t / kappa, 1)[(n, am)]
        return mp.exp(-sign * lam * z) * fac * I ** am * mp.besselj(am, t * rho) * trig
    return lam_integral(kappa, f) / (I ** (n + 1) * kappa)


KAPPAS = [mp.mpf("1.3"), mp.mpc("0.9", "0.4"), mp.mpc(0, "1.1")]


def check_representation(rng):
    print("1. the representation of h_n Y_n^m, up and down")
    for kappa in KAPPAS:
        worst = mp.mpf(0)
        for up in (True, False):
            u = [mp.mpf(rng.uniform(-1.5, 1.5)), mp.mpf(rng.uniform(-1.5, 1.5)),
                 mp.mpf(rng.uniform(1.0, 3.0)) * (1 if up else -1)]
            r = mp.sqrt(sum(c * c for c in u))
            for n in range(6):
                for m in range(-n, n + 1):
                    lhs = hn(n, kappa * r) * real_y(n, m, u)
                    rhs = representation(kappa, n, m, u, up)
                    worst = max(worst, abs(lhs - rhs) / max(abs(lhs), mp.mpf(10) ** -12))
        check(f"kappa = {mp.nstr(kappa, 3)}", worst, mp.mpf(10) ** -15)


def check_factors(rng):
    print("2. the solid recursion against the real harmonics")
    worst = mp.mpf(0)
    for _ in range(10):
        x = [mp.mpf(rng.uniform(-1, 1)) for _ in range(3)]
        r = mp.sqrt(sum(c * c for c in x))
        rho = mp.sqrt(x[0] ** 2 + x[1] ** 2)
        phi = mp.atan2(x[1], x[0])
        f = solid(8, x[2], rho, r * r)
        for n in range(9):
            for m in range(n + 1):
                v = f[(n, m)] / r ** n
                worst = max(worst, abs(v * mp.cos(m * phi) - real_y(n, m, x)))
                if m > 0:
                    worst = max(worst, abs(v * mp.sin(m * phi) - real_y(n, -m, x)))
    check("n <= 8", worst, mp.mpf(10) ** -25)


def check_expansion(rng):
    print("3. the plane-wave expansion at complex s^")
    for kappa in KAPPAS:
        lam = mp.mpc(rng.uniform(0.5, 3), rng.uniform(-1, 1))
        t = mp.sqrt(lam * lam + kappa * kappa)
        alpha = mp.mpf(rng.uniform(0, 6))
        s = [t * mp.cos(alpha) / kappa, t * mp.sin(alpha) / kappa, I * lam / kappa]
        u = [mp.mpf(rng.uniform(-0.4, 0.4)) for _ in range(3)]
        r = mp.sqrt(sum(c * c for c in u))
        exact = mp.exp(I * kappa * sum(a * b for a, b in zip(s, u)))
        f = solid(40, I * lam, t, kappa * kappa)
        total = mp.mpc(0)
        for n in range(41):
            for m in range(-n, n + 1):
                am = abs(m)
                trig = mp.cos(am * alpha) if m >= 0 else mp.sin(am * alpha)
                ys = f[(n, am)] / kappa ** n * trig
                total += 4 * mp.pi * I ** n * jn(n, kappa * r) * ys * real_y(n, m, u)
        check(f"kappa = {mp.nstr(kappa, 3)}", abs(total - exact) / abs(exact), mp.mpf(10) ** -20)


def level_constants(kappa):
    k_hat = kappa / 2
    a = mp.sqrt(3)
    s = min(abs(k_hat) * a, mp.mpf(1))
    beta = mp.im(k_hat) * a
    return k_hat, s, beta


def p_factor(n, m, lam, t, kappa, s, sign):
    """P_k(n, m) of §3.15 for the up (sign 1) or down (sign -1) waves."""
    f = solid(n, sign * s / kappa * I * lam, s / kappa * t, s * s)[(n, m)]
    return I ** (-n) * f


def check_stored(rng):
    print("4. the stored forms: multipole to waves, one wave to a local")
    p = 12
    for kappa in KAPPAS:
        k_hat, s, beta = level_constants(kappa)
        # P2M in stored form (level units): M^ = e^{-beta} sum q j_n(k^ rho) Y / s^n.
        src = [[mp.mpf(rng.uniform(-0.5, 0.5)) for _ in range(3)] for _ in range(3)]
        q = [mp.mpc(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in src]
        mhat = {}
        for n in range(p + 1):
            for m in range(-n, n + 1):
                mhat[(n, m)] = sum(qq * jn(n, k_hat * mp.sqrt(sum(c * c for c in y)))
                                   * real_y(n, m, y) for qq, y in zip(q, src)) \
                    * mp.exp(-beta) / s ** n
        # r phi~ of the multipole at a point u (box widths) above, by the waves' integral.
        u = [mp.mpf("0.4"), mp.mpf("-0.7"), mp.mpf("2.2")]
        rho = mp.sqrt(u[0] ** 2 + u[1] ** 2)
        phi = mp.atan2(u[1], u[0])

        def waves(lam, t):
            total = mp.mpc(0)
            for m in range(p + 1):
                a = sum(p_factor(n, m, lam, t, kappa, s, 1) * mhat[(n, m)] for n in range(m, p + 1))
                b = sum(p_factor(n, m, lam, t, kappa, s, 1) * mhat[(n, -m)]
                        for n in range(m, p + 1)) if m > 0 else 0
                bes = I ** m * mp.besselj(m, t * rho)
                total += a * bes * mp.cos(m * phi) + b * bes * mp.sin(m * phi)
            return 2 * mp.pi * mp.exp(beta) * total * mp.exp(-lam * u[2])
        got = lam_integral(kappa, waves)
        v = [2 * c for c in u]
        r = mp.sqrt(sum(c * c for c in v))
        expected = 4 * mp.pi * I * k_hat * sum(
            mhat[(n, m)] * mp.exp(beta) * s ** n * hn(n, k_hat * r) * real_y(n, m, v)
            for n in range(p + 1) for m in range(-n, n + 1))
        check(f"waves of a multipole, kappa = {mp.nstr(kappa, 3)}",
              abs(got - expected) / abs(expected), mp.mpf(10) ** -15)

        # One wave (lambda, alpha), down and up, to a local; L2P in stored form at v.
        for sign in (1, -1):
            lam = mp.mpc(rng.uniform(0.5, 2), rng.uniform(-0.5, 0.5))
            t = mp.sqrt(lam * lam + kappa * kappa)
            alpha = mp.mpf(rng.uniform(0, 6))
            v = [mp.mpf(rng.uniform(-0.6, 0.6)) for _ in range(3)]
            uu = [c / 2 for c in v]
            wave = mp.exp(-sign * lam * uu[2] + I * t * (uu[0] * mp.cos(alpha) + uu[1] * mp.sin(alpha)))
            r = mp.sqrt(sum(c * c for c in v))
            total = mp.mpc(0)
            for n in range(41):
                for m in range(-n, n + 1):
                    am = abs(m)
                    trig = mp.cos(am * alpha) if m >= 0 else mp.sin(am * alpha)
                    lhat = 4 * mp.pi * mp.exp(beta) * (-1) ** n \
                        * p_factor(n, am, lam, t, kappa, s, sign) * trig
                    # L2P: phi~ = sum L^ j_n(k^|v|) Y e^{-beta} / s^n.
                    total += lhat * mp.exp(-beta) / s ** n * jn(n, k_hat * r) * real_y(n, m, v)
            check(f"one {'up' if sign > 0 else 'down'} wave to a local, kappa = {mp.nstr(kappa, 3)}",
                  abs(total - wave) / abs(wave), mp.mpf(10) ** -15)


def main():
    rng = random.Random(20261103)
    check_factors(rng)
    check_expansion(rng)
    check_stored(rng)
    check_representation(rng)
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
