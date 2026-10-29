#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Numerical check of the identities of docs/CONVENTIONS.md §3.15 (Helmholtz CH1.1).

Every check works from the definitions written in §3.15, in mpmath at DPS digits, on
seeded random points, for real, complex and imaginary k, and fails (exit 1) above its
tolerance. It checks:

  1. the kernel's gradient (item 1) against a numerical derivative, and its k -> 0 limit;
  2. the radial relations of item 3: h_n against its finite Hankel sum, k_0 = e^{-x}/x,
     j_n(ix) = i^n i_n(x), h_n(ix) = -i^{-n} k_n(x);
  3. the real orthonormal harmonics of item 2: the addition formula
     sum_m Y Y = (2n+1)/(4 pi) P_n(cos gamma), and r^n Y against the Racah R_n^m of
     §3.3 (slot +m sqrt(2) nu_m Re R, slot -m sqrt(2) nu_m Im R, slot 0 nu_0 R);
  4. the series e^{ikr}/r = 4 pi ik sum_n j_n h_n sum_m Y Y (item 1);
  5. the leaf operators in stored form (items 1 and 3): P2M -> M2P and P2L -> L2P with
     the scale s_l, the per-box factor e^{-/+ Im(k) a_l}, the scaled radial functions and
     the factor 4 pi ik on the crossing operators, against the direct sum; every
     exponent an operator applies is <= 0;
  6. Yukawa (item 3): M^Y = i^{-n} M and L^Y = i^n L real for real charges, and the
     series kappa sum i_n k_n with 4 pi kappa on the crossing operators;
  7. the rotation blocks of item 5: orthogonal, and O(pi - b) = (-1)^n E O(b) E F;
  8. the coaxial coefficients of item 6 by projection: one coefficient for slots +m and
     -m, (S|S) = (R|R), real for real k, i^{n-n'} (E|F) real for Yukawa, the index
     symmetry and the reflection rule;
  9. P2M -> M2L -> L2P on the z-axis in stored form: the table in its stored
     normalisation, the factor 4 pi ik and the scalar e^{2 beta_l - Im(k)|t|} <= 1 on the
     (S|R) step, against the direct sum;
 10. P2M -> M2M -> M2P and P2L -> L2L -> L2P on the z-axis in stored form, child and
     parent (|t| = a_child), with the stored (S|S) and (R|R) and the power scales alone;
 11. the recursions of item 6 (start column, in n, in m) against the projected tables;
 12. Yukawa's stored forms with i^ and k^.

Run from the repository root:  uv run tools/fixtures/check_3_15.py
(or python3 with mpmath==1.4.1). It takes about three minutes.
"""
import random
import sys

import mpmath as mp
from mpmath import mpf, mpc

DPS = 40
mp.mp.dps = DPS
I = mpc(0, 1)
RNG = random.Random(20261029)

# Wavenumbers: real, complex (Re k > 0, Im k > 0), imaginary (Yukawa, k = i kappa).
KS = {"real": mpf("1.7"), "complex": mpc("1.7", "0.6"), "yukawa": mpc(0, "2.3")}

failures = []


def check(name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {mp.nstr(err, 3)} (tol {mp.nstr(tol, 1)})")
    if not ok:
        failures.append(name)


# --- radial functions (item 3) -------------------------------------------------------


def sph_j(n, z):
    if z == 0:
        return mpf(1) if n == 0 else mpf(0)
    return mp.sqrt(mp.pi / (2 * z)) * mp.besselj(n + mpf(1) / 2, z)


def sph_y(n, z):
    return mp.sqrt(mp.pi / (2 * z)) * mp.bessely(n + mpf(1) / 2, z)


def sph_h(n, z):
    """h_n = h_n^(1) by its finite Hankel sum: j + i y cancels by e^{2 Im z} for Im z > 0
    (check 2 compares the two)."""
    return sph_h_sum(n, z)


def sph_h_sum(n, z):
    """h_n^(1)(z) = (-i)^{n+1} e^{iz}/z sum_k i^k (n+k)! / (k! (n-k)! (2z)^k)."""
    s = sum(I**k * mp.factorial(n + k) / (mp.factorial(k) * mp.factorial(n - k) * (2 * z) ** k)
            for k in range(n + 1))
    return (-I) ** (n + 1) * mp.exp(I * z) / z * s


def sph_i(n, x):
    return mp.sqrt(mp.pi / (2 * x)) * mp.besseli(n + mpf(1) / 2, x)


def sph_k(n, x):
    return mp.sqrt(2 / (mp.pi * x)) * mp.besselk(n + mpf(1) / 2, x)


def scale(k, a):
    """s_l = min(|k| a_l, 1)."""
    return min(abs(k) * a, mpf(1))


def j_hat(n, z, s):
    """Scaled regular radial function: e^{-|Im z|} j_n(z) / s^n."""
    return mp.exp(-abs(mp.im(z))) * sph_j(n, z) / s**n


def h_hat(n, z, s):
    """Scaled singular radial function: e^{Im z} h_n(z) s^n."""
    return mp.exp(mp.im(z)) * sph_h(n, z) * s**n


# --- angular basis (item 2) ----------------------------------------------------------


def legendre(n, m, t):
    """P_n^m(t) without the Condon-Shortley phase (§3.3), by recursion in n."""
    pmm = mp.fac2(2 * m - 1) * (1 - t * t) ** (mpf(m) / 2) if m > 0 else mpf(1)
    if n == m:
        return pmm
    p1 = t * (2 * m + 1) * pmm
    if n == m + 1:
        return p1
    p0 = pmm
    for nn in range(m + 2, n + 1):
        p0, p1 = p1, (t * (2 * nn - 1) * p1 - (nn + m - 1) * p0) / (nn - m)
    return p1


def angles(x):
    r = mp.sqrt(sum(c * c for c in x))
    return r, x[2] / r, mp.atan2(x[1], x[0])


def ylm(n, m, x):
    """Real orthonormal Y_n^m at the direction of x: cos part in slot +m, sin in -m."""
    _, t, phi = angles(x)
    am = abs(m)
    p = legendre(n, am, t)
    if m == 0:
        return mp.sqrt((2 * n + 1) / (4 * mp.pi)) * p
    c = mp.sqrt(2 * (2 * n + 1) / (4 * mp.pi) * mp.factorial(n - am) / mp.factorial(n + am))
    return c * p * (mp.cos(am * phi) if m > 0 else mp.sin(am * phi))


def y_vec(n, x):
    return [ylm(n, m, x) for m in range(-n, n + 1)]


def racah_r(n, m, x):
    """R_n^m of §3.3 for m >= 0."""
    r, t, phi = angles(x)
    return r**n / mp.factorial(n + m) * legendre(n, m, t) * mp.exp(I * m * phi)


def rand_unit():
    while True:
        v = [mpf(RNG.uniform(-1, 1)) for _ in range(3)]
        r2 = sum(c * c for c in v)
        if mpf("0.01") < r2 <= 1:
            r = mp.sqrt(r2)
            return [c / r for c in v]


def rand_point(rmin, rmax, centre=(0, 0, 0)):
    u = rand_unit()
    r = mpf(RNG.uniform(float(rmin), float(rmax)))
    return [centre[i] + r * u[i] for i in range(3)]


def sub(x, y):
    return [x[i] - y[i] for i in range(3)]


def norm(x):
    return mp.sqrt(sum(c * c for c in x))


# --- 1. kernel and gradient ----------------------------------------------------------


def kernel(k, x, y):
    r = norm(sub(x, y))
    return mp.exp(I * k * r) / (4 * mp.pi * r)


def kernel_grad(k, x, y):
    d = sub(x, y)
    r = norm(d)
    f = (I * k * r - 1) * mp.exp(I * k * r) / (4 * mp.pi * r**3)
    return [f * c for c in d]


def check_kernel():
    print("1. kernel gradient (item 1) and k -> 0")
    for label, k in KS.items():
        x, y = rand_point(0.5, 1), rand_point(0, 0.3)
        g = kernel_grad(k, x, y)
        err = mpf(0)
        for i in range(3):
            def f(t, i=i):
                xx = list(x)
                xx[i] = t
                return kernel(k, xx, y)
            err = max(err, abs(mp.diff(f, x[i]) - g[i]) / abs(g[i]))
        check(f"gradient, {label} k", err, mpf(10) ** (-30))
    x, y = rand_point(0.5, 1), rand_point(0, 0.3)
    k = mpf(10) ** -20
    r = norm(sub(x, y))
    lap = 1 / (4 * mp.pi * r)
    lap_grad = [-c / (4 * mp.pi * r**3) for c in sub(x, y)]
    err = max([abs(kernel(k, x, y) - lap) / lap]
              + [abs(a - b) / abs(b) for a, b in zip(kernel_grad(k, x, y), lap_grad)])
    check("k -> 0 gives the Laplace potential and gradient of §3.13", err, mpf(10) ** -18)


# --- 2. radial relations ----------------------------------------------------------------


def check_radial():
    print("2. radial functions (item 3)")
    err = mpf(0)
    for n in range(0, 12):
        for z in [mpf("0.7"), mpf("13.2"), mpc("2.1", "0.9"), mpc(0, "3.4"), mpc("0.05", "7")]:
            jy = sph_j(n, z) + I * sph_y(n, z)
            err = max(err, abs(jy - sph_h_sum(n, z)) / abs(sph_h_sum(n, z)))
    check("h_n = j_n + i y_n against the Hankel sum", err, mpf(10) ** -30)
    x = mpf("2.7")
    check("k_0(x) = e^{-x}/x", abs(sph_k(0, x) - mp.exp(-x) / x) / (mp.exp(-x) / x),
          mpf(10) ** -35)
    e1 = e2 = mpf(0)
    for n in range(0, 15):
        for x in [mpf("0.3"), mpf("4.1"), mpf("19")]:
            e1 = max(e1, abs(sph_j(n, I * x) - I**n * sph_i(n, x)) / abs(sph_i(n, x)))
            e2 = max(e2, abs(sph_h(n, I * x) + I ** (-n) * sph_k(n, x)) / abs(sph_k(n, x)))
    check("j_n(ix) = i^n i_n(x)", e1, mpf(10) ** -30)
    check("h_n(ix) = -i^{-n} k_n(x)", e2, mpf(10) ** -30)


# --- 3. harmonics ------------------------------------------------------------------------


def check_harmonics():
    print("3. real orthonormal harmonics (item 2)")
    err = mpf(0)
    for _ in range(5):
        a, b = rand_unit(), rand_unit()
        cg = sum(a[i] * b[i] for i in range(3))
        for n in range(0, 16):
            lhs = sum(p * q for p, q in zip(y_vec(n, a), y_vec(n, b)))
            rhs = (2 * n + 1) / (4 * mp.pi) * mp.legendre(n, cg)
            err = max(err, abs(lhs - rhs))
    check("sum_m Y_n^m(a) Y_n^m(b) = (2n+1)/(4 pi) P_n(a.b)", err, mpf(10) ** -32)
    err = mpf(0)
    for _ in range(5):
        x = rand_point(0.2, 1.7)
        r = norm(x)
        for n in range(0, 16):
            c = mp.sqrt((2 * n + 1) / (4 * mp.pi))
            for m in range(-n, n + 1):
                am = abs(m)
                nu = mp.sqrt(mp.factorial(n + am) * mp.factorial(n - am))
                rr = racah_r(n, am, x)
                if m == 0:
                    want = c * nu * mp.re(rr)
                elif m > 0:
                    want = c * mp.sqrt(2) * nu * mp.re(rr)
                else:
                    want = c * mp.sqrt(2) * nu * mp.im(rr)
                err = max(err, abs(r**n * ylm(n, m, x) - want) / max(r**n, 1))
    check("r^n Y against the Racah R_n^m through nu_m and sqrt 2", err, mpf(10) ** -32)


# --- 4. the series -----------------------------------------------------------------------


def series(k, x, y, p):
    """4 pi ik sum_{n<=p} j_n(k|y|) h_n(k|x|) sum_m Y(y) Y(x), |y| < |x|."""
    rx, ry = norm(x), norm(y)
    s = mpf(0)
    for n in range(p + 1):
        s += sph_j(n, k * ry) * sph_h(n, k * rx) * sum(
            a * b for a, b in zip(y_vec(n, y), y_vec(n, x)))
    return 4 * mp.pi * I * k * s


def check_series():
    print("4. e^{ikr}/r = 4 pi ik sum_n j_n h_n sum_m Y Y (item 1)")
    for label, k in KS.items():
        err = mpf(0)
        for _ in range(3):
            x, y = rand_point(2, 2.5), rand_point(0.1, 0.4)
            want = 4 * mp.pi * kernel(k, x, y)
            err = max(err, abs(series(k, x, y, 60) - want) / abs(want))
        check(f"series, {label} k", err, mpf(10) ** -28)


# --- 5. leaf operators in stored form ------------------------------------------------------


def stored_p2m(k, a, centre, sources, p, exps):
    """M^_n^m = e^{-Im(k) a} M / s^n = sum_j q_j j^_n(k rho) e^{Im(k) rho - Im(k) a} Y."""
    s, beta = scale(k, a), mp.im(k) * a
    coeffs = {}
    for y, q in sources:
        rho_v = sub(y, centre)
        rho = norm(rho_v)
        e = mp.im(k) * rho - beta
        exps.append(e)
        for n in range(p + 1):
            rad = j_hat(n, k * rho, s) * mp.exp(e)
            for m, yv in zip(range(-n, n + 1), y_vec(n, rho_v)):
                coeffs[(n, m)] = coeffs.get((n, m), 0) + q * rad * yv
    return coeffs


def stored_m2p(k, a, centre, coeffs, x, p, exps):
    """phi~(x) = 4 pi ik sum M^_n^m h^_n(k|v|) e^{Im(k) a - Im(k)|v|} Y(v)."""
    s, beta = scale(k, a), mp.im(k) * a
    v = sub(x, centre)
    rv = norm(v)
    e = beta - mp.im(k) * rv
    exps.append(e)
    out = mpf(0)
    for n in range(p + 1):
        rad = h_hat(n, k * rv, s) * mp.exp(e)
        out += rad * sum(coeffs[(n, m)] * yv for m, yv in zip(range(-n, n + 1), y_vec(n, v)))
    return 4 * mp.pi * I * k * out


def stored_p2l(k, a, centre, sources, p, exps):
    """L^_n^m = e^{Im(k) a} L s^n = 4 pi ik sum_j q_j h^_n(k rho) e^{Im(k) a - Im(k) rho} Y."""
    s, beta = scale(k, a), mp.im(k) * a
    coeffs = {}
    for y, q in sources:
        rho_v = sub(y, centre)
        rho = norm(rho_v)
        e = beta - mp.im(k) * rho
        exps.append(e)
        for n in range(p + 1):
            rad = h_hat(n, k * rho, s) * mp.exp(e)
            for m, yv in zip(range(-n, n + 1), y_vec(n, rho_v)):
                coeffs[(n, m)] = coeffs.get((n, m), 0) + 4 * mp.pi * I * k * q * rad * yv
    return coeffs


def stored_l2p(k, a, centre, coeffs, x, p, exps):
    """phi~(x) = sum L^_n^m j^_n(k|v|) e^{Im(k)|v| - Im(k) a} Y(v)."""
    s, beta = scale(k, a), mp.im(k) * a
    v = sub(x, centre)
    rv = norm(v)
    e = mp.im(k) * rv - beta
    exps.append(e)
    out = mpf(0)
    for n in range(p + 1):
        rad = j_hat(n, k * rv, s) * mp.exp(e)
        out += rad * sum(coeffs[(n, m)] * yv for m, yv in zip(range(-n, n + 1), y_vec(n, v)))
    return out


def charges(nq, cplx):
    if cplx:
        return [mpc(RNG.uniform(-1, 1), RNG.uniform(-1, 1)) for _ in range(nq)]
    return [mpf(RNG.uniform(-1, 1)) for _ in range(nq)]


def check_leaf_operators():
    print("5. leaf operators in stored form (items 1 and 3)")
    centre = [mpf("0.1"), mpf("-0.2"), mpf("0.05")]
    p = 50
    # |k| a below and above 1, so that s_l < 1 and s_l = 1 are both covered.
    for a in [mpf("0.15"), mpf("1.2")]:
        for label, k in KS.items():
            for cplx in [False, True]:
                exps = []
                q = charges(4, cplx)
                inside = [(rand_point(0, a, centre), qq) for qq in q]
                outside = [(rand_point(4 * a, 5 * a, centre), qq) for qq in q]
                m_hat = stored_p2m(k, a, centre, inside, p, exps)
                l_hat = stored_p2l(k, a, centre, outside, p, exps)
                err = mpf(0)
                for _ in range(2):
                    x_far = rand_point(4 * a, 5 * a, centre)
                    want = sum(qq * 4 * mp.pi * kernel(k, x_far, y) for y, qq in inside)
                    got = stored_m2p(k, a, centre, m_hat, x_far, p, exps)
                    err = max(err, abs(got - want) / abs(want))
                    x_near = rand_point(0, a, centre)
                    want = sum(qq * 4 * mp.pi * kernel(k, x_near, y) for y, qq in outside)
                    got = stored_l2p(k, a, centre, l_hat, x_near, p, exps)
                    err = max(err, abs(got - want) / abs(want))
                tag = f"{label} k, |k|a = {mp.nstr(abs(k) * a, 2)}, {'complex' if cplx else 'real'} q"
                check(f"P2M->M2P and P2L->L2P, {tag}", err, mpf(10) ** -24)
                check(f"no positive exponent, {tag}", max(max(exps), mpf(0)), mpf(0))


# --- 6. Yukawa ---------------------------------------------------------------------------


def check_yukawa():
    print("6. Yukawa's real form (item 3)")
    kappa = mpf("2.3")
    k = I * kappa
    q = charges(4, False)
    inside = [(rand_point(0, 0.3), qq) for qq in q]
    outside = [(rand_point(1.5, 2), qq) for qq in q]
    p = 40
    err_real = err_m = err_l = mpf(0)
    for n in range(p + 1):
        jin = [sph_j(n, k * norm(y)) for y, _ in inside]
        hout = [sph_h(n, k * norm(y)) for y, _ in outside]
        iin = [sph_i(n, kappa * norm(y)) for y, _ in inside]
        kout = [sph_k(n, kappa * norm(y)) for y, _ in outside]
        for m in range(-n, n + 1):
            # M = sum q j_n(k rho) Y and L = 4 pi ik sum q h_n(k rho) Y (no scaling).
            mm = sum(qq * jv * ylm(n, m, y) for (y, qq), jv in zip(inside, jin))
            ll = sum(4 * mp.pi * I * k * qq * hv * ylm(n, m, y)
                     for (y, qq), hv in zip(outside, hout))
            my = I ** (-n) * mm
            ly = I**n * ll
            my_want = sum(qq * iv * ylm(n, m, y) for (y, qq), iv in zip(inside, iin))
            ly_want = sum(4 * mp.pi * kappa * qq * kv * ylm(n, m, y)
                          for (y, qq), kv in zip(outside, kout))
            err_real = max(err_real, abs(mp.im(my)) / (abs(my) + mpf(10) ** -300),
                           abs(mp.im(ly)) / (abs(ly) + mpf(10) ** -300))
            err_m = max(err_m, abs(my - my_want) / max(abs(my_want), mpf(10) ** -300))
            err_l = max(err_l, abs(ly - ly_want) / max(abs(ly_want), mpf(10) ** -300))
    check("M^Y = i^{-n} M and L^Y = i^n L real for real charges", err_real, mpf(10) ** -30)
    check("M^Y = sum q i_n(kappa rho) Y", err_m, mpf(10) ** -30)
    check("L^Y = 4 pi kappa sum q k_n(kappa rho) Y", err_l, mpf(10) ** -30)
    # The real series: 4 pi kappa sum i_n k_n sum Y Y = e^{-kappa r}/r.
    err = mpf(0)
    for _ in range(3):
        x, y = rand_point(2, 2.5), rand_point(0.1, 0.4)
        s = sum(sph_i(n, kappa * norm(y)) * sph_k(n, kappa * norm(x))
                * sum(a * b for a, b in zip(y_vec(n, y), y_vec(n, x))) for n in range(61))
        want = mp.exp(-kappa * norm(sub(x, y))) / norm(sub(x, y))
        err = max(err, abs(4 * mp.pi * kappa * s - want) / want)
    check("e^{-kappa r}/r = 4 pi kappa sum i_n k_n sum Y Y", err, mpf(10) ** -28)


# --- 7. rotation blocks --------------------------------------------------------------------


def rot_y(b):
    c, s = mp.cos(b), mp.sin(b)
    return mp.matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def apply(qm, x):
    return [sum(qm[i, j] * x[j] for j in range(3)) for i in range(3)]


def block(n, qm):
    """O^n(Q) with Y_n(Q x) = O^n(Q) Y_n(x), by least squares on 4n + 8 directions."""
    xs = [rand_unit() for _ in range(4 * n + 8)]
    a = mp.matrix([y_vec(n, x) for x in xs]).T
    b = mp.matrix([y_vec(n, apply(qm, x)) for x in xs]).T
    return b * a.T * mp.inverse(a * a.T)


def check_rotations():
    print("7. rotation blocks (item 5)")
    beta = mpf("0.73")
    e_orth = e_rel = mpf(0)
    for n in range(0, 7):
        o1 = block(n, rot_y(beta))
        o2 = block(n, rot_y(mp.pi - beta))
        dim = 2 * n + 1
        e_orth = max(e_orth, mp.mnorm(o1 * o1.T - mp.eye(dim), 1))
        ms = list(range(-n, n + 1))
        e = mp.diag([(-1) ** abs(m) for m in ms])
        f = mp.diag([1 if m >= 0 else -1 for m in ms])
        e_rel = max(e_rel, mp.mnorm(o2 - (-1) ** n * e * o1 * e * f, 1))
    check("O^n orthogonal, n <= 6", e_orth, mpf(10) ** -30)
    check("O^n(pi - b) = (-1)^n E O^n(b) E F, n <= 6", e_rel, mpf(10) ** -30)


# --- 8. coaxial coefficients ------------------------------------------------------------


GL_N = 30
GL = None


def gauss_legendre():
    global GL
    if GL is None:
        GL = list(zip(*mp.gauss_quadrature(GL_N, "legendre")))
    return GL


def coaxial(kind, k, t, m, nmax, r):
    """(E|F)^m_{n'n}(t z) for n, n' <= nmax by projection on the sphere |r|: the expansion
    F_n^m(r + t) = sum_n' (E|F)^m_{n'n}(t) G_n'^m(r), with F = j (R) or h (S) and
    G = j (R) or h (S) as kind = "RR", "SS", "SR". On the axis the integrand is
    cos(m phi) or sin(m phi) times a function of theta, so the phi integral is taken at
    the two points phi = 0 (slot +m) and phi = pi/(2m) (slot -m). Returns
    {(n, n', slot): value} for the slots +|m| and -|m|."""
    f_fun = sph_j if kind[0] == "R" else sph_h
    g_fun = sph_j if kind[1] == "R" else sph_h
    out = {}
    slots = [0] if m == 0 else [m, -m]
    for slot in slots:
        phi = mpf(0) if slot >= 0 else mp.pi / (2 * m)
        acc = {}
        for ct, w in gauss_legendre():
            st = mp.sqrt(1 - ct * ct)
            rv = [r * st * mp.cos(phi), r * st * mp.sin(phi), r * ct]
            shifted = [rv[0], rv[1], rv[2] + t]
            rs = norm(shifted)
            fvals = {n: f_fun(n, k * rs) * ylm(n, slot, shifted) for n in range(m, nmax + 1)}
            gvals = {nn: ylm(nn, slot, rv) for nn in range(m, nmax + 1)}
            for n in range(m, nmax + 1):
                for nn in range(m, nmax + 1):
                    acc[(n, nn)] = acc.get((n, nn), 0) + w * fvals[n] * gvals[nn]
        # The phi integral of cos^2 or sin^2 (m > 0) is pi, of 1 (m = 0) is 2 pi; the
        # value at the chosen phi of the angular factor is 1.
        for (n, nn), v in acc.items():
            # Y(slot) at phi carries the factor cos(m phi) or sin(m phi) = 1 there.
            ang = (2 * mp.pi) if m == 0 else mp.pi
            out[(n, nn, slot)] = v * ang / g_fun(nn, k * r)
    return out


def check_coaxial():
    print("8. coaxial coefficients (item 6), by projection")
    nmax, t = 5, mpf("1.1")
    tol = mpf(10) ** -20
    for label, k in [("real", mpf("1.7")), ("complex", mpc("1.7", "0.6")),
                     ("yukawa", mpc(0, "2.3"))]:
        e_slot = e_ss = e_real = e_sym = e_refl = mpf(0)
        for m in [0, 1, 2]:
            rr = coaxial("RR", k, t, m, nmax, mpf("0.4"))
            ss = coaxial("SS", k, t, m, nmax, mpf("4.4"))
            # (S|R) grows with n and n'; a smaller sphere keeps the projection free of
            # aliasing (0.4 aliases at 1e-17 by n = 4).
            sr = coaxial("SR", k, t, m, nmax, mpf("0.25"))
            sr_neg = coaxial("SR", k, -t, m, nmax, mpf("0.25"))
            rr_neg = coaxial("RR", k, -t, m, nmax, mpf("0.4"))
            for n in range(m, nmax + 1):
                for nn in range(m, nmax + 1):
                    if m > 0:
                        for tab in (rr, ss, sr):
                            e_slot = max(e_slot, abs(tab[(n, nn, m)] - tab[(n, nn, -m)]))
                    e_ss = max(e_ss, abs(ss[(n, nn, m)] - rr[(n, nn, m)]))
                    for tab in (rr, sr):
                        e_sym = max(e_sym, abs(tab[(n, nn, m)]
                                               - (-1) ** (n + nn) * tab[(nn, n, m)])
                                    / abs(tab[(n, nn, m)]))
                    for neg, tab in ((sr_neg, sr), (rr_neg, rr)):
                        e_refl = max(e_refl, abs(neg[(n, nn, m)]
                                                 - (-1) ** (n + nn) * tab[(n, nn, m)])
                                     / abs(tab[(n, nn, m)]))
                    if label == "real":
                        e_real = max(e_real, abs(mp.im(rr[(n, nn, m)])))
                    if label == "yukawa":
                        e_real = max(e_real, abs(mp.im(I ** (n - nn) * rr[(n, nn, m)])),
                                     abs(mp.im(I ** (n - nn) * sr[(n, nn, m)])))
        check(f"one coefficient for slots +m and -m, {label} k", e_slot, tol)
        check(f"(S|S) = (R|R), {label} k", e_ss, tol)
        check(f"index symmetry E_nn' = (-1)^(n+n') E_n'n (relative), {label} k", e_sym, tol)
        check(f"reflection (E|F)(-t) = (-1)^(n+n') (E|F)(t), (S|R) and (R|R) (relative), {label} k", e_refl, tol)
        if label == "real":
            check("(R|R) real for real k", e_real, tol)
        if label == "yukawa":
            check("i^(n-n') (E|F) real for Yukawa", e_real, tol)


def check_m2l_chain():
    print("9. P2M -> M2L -> L2P on the z-axis in stored form (items 1, 3 and 6)")
    p, a, t = 10, mpf("0.04"), mpf("1.1")
    src_c = [mpf(0)] * 3
    tgt_c = [mpf(0), mpf(0), t]
    for label, k in KS.items():
        for cplx in [False, True]:
            exps = []
            q = charges(4, cplx)
            inside = [(rand_point(0, a, src_c), qq) for qq in q]
            m_hat = stored_p2m(k, a, src_c, inside, p, exps)
            s_l, beta = scale(k, a), mp.im(k) * a
            # The table as stored, e^{Im z} s^{n+n'} (S|R) with z = kt, and the scalar
            # e^{2 beta - Im(k) t} <= 1 besides 4 pi ik: L^_n' = 4 pi ik e^{2 beta - Im(k) t}
            # sum_n S^_{n'n} M^_n.
            scalar = mp.exp(2 * beta - mp.im(k) * t)
            exps.append(2 * beta - mp.im(k) * t)
            l_hat = {}
            for m in range(0, p + 1):
                sr = coaxial("SR", k, t, m, p, mpf("0.25"))
                for slot in ([0] if m == 0 else [m, -m]):
                    for nn in range(m, p + 1):
                        acc = sum(mp.exp(mp.im(k * t)) * s_l ** (n + nn) * sr[(n, nn, slot)]
                                  * m_hat[(n, slot)] for n in range(m, p + 1))
                        l_hat[(nn, slot)] = 4 * mp.pi * I * k * scalar * acc
            err = mpf(0)
            for _ in range(2):
                x = rand_point(0, a, tgt_c)
                want = sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in inside)
                got = stored_l2p(k, a, tgt_c, l_hat, x, p, exps)
                err = max(err, abs(got - want) / abs(want))
            tag = f"{label} k, {'complex' if cplx else 'real'} q"
            # Truncation at p = 10 with (2a/t)^11 about 3e-13.
            check(f"M2L chain, {tag}", err, mpf(10) ** -10)
            check(f"no positive exponent, {tag}", max(max(exps), mpf(0)), mpf(0))


def check_m2m_l2l_chain(p=14, ks=None):
    print("10. P2M -> M2M -> M2P and P2L -> L2L -> L2P on the z-axis in stored form")
    a_c = mpf("0.3")
    a_p = 2 * a_c
    parent = [mpf(0)] * 3
    child = [mpf(0), mpf(0), -a_c]  # t = P - c = a_c z for M2M; c - P = -a_c z for L2L
    for label, k in (ks or KS).items():
        for cplx in [False, True]:
            exps = []
            q = charges(4, cplx)
            s_c, s_p = scale(k, a_c), scale(k, a_p)
            # M2M: M^P_n' = s_p^{-n'} sum_n S^_{n'n} s_c^n M^c_n, S^ = e^{-Im z} (S|S),
            # z = k a_c; the damping factors cancel (beta_P = 2 beta_c, |t| = a_c).
            inside = [(rand_point(0, a_c / 2, child), qq) for qq in q]
            m_c = stored_p2m(k, a_c, child, inside, p, exps)
            m_p = {}
            # L2L: L^c_n' = s_c^{n'} sum_n R^_{n'n} s_p^{-n} L^P_n, R^ = e^{-Im z} (R|R).
            far = [(rand_point(14, 16, parent), qq) for qq in q]
            l_p = stored_p2l(k, a_p, parent, far, p, exps)
            l_c = {}
            for m in range(0, p + 1):
                ss = coaxial("SS", k, a_c, m, p, 4 * a_c)
                rr = coaxial("RR", k, -a_c, m, p, a_c / 2)
                damp = mp.exp(-mp.im(k * a_c))
                for slot in ([0] if m == 0 else [m, -m]):
                    for nn in range(m, p + 1):
                        m_p[(nn, slot)] = s_p ** (-nn) * sum(
                            damp * ss[(n, nn, slot)] * s_c**n * m_c[(n, slot)]
                            for n in range(m, p + 1))
                        l_c[(nn, slot)] = s_c**nn * sum(
                            damp * rr[(n, nn, slot)] * s_p ** (-n) * l_p[(n, slot)]
                            for n in range(m, p + 1))
            err = mpf(0)
            for _ in range(2):
                x = rand_point(14, 16, parent)
                want = sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in inside)
                got = stored_m2p(k, a_p, parent, m_p, x, p, exps)
                err = max(err, abs(got - want) / abs(want))
                x = rand_point(0, a_c / 2, child)
                want = sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in far)
                got = stored_l2p(k, a_c, child, l_c, x, p, exps)
                err = max(err, abs(got - want) / abs(want))
            tag = f"{label} k, {'complex' if cplx else 'real'} q"
            # Truncation at p = 14, measured: 1e-11 at p = 10 for Yukawa (whose field is
            # e^{-34} at the targets), 1e-17 at 14, 1e-22 at 18.
            check(f"M2M and L2L chains, {tag}", err, mpf(10) ** -14)
            check(f"no positive exponent, {tag}", max(max(exps), mpf(0)), mpf(0))


def check_recursions():
    print("11. coaxial recursions (item 6) against the projected tables")
    nmax, t = 6, mpf("1.1")

    def a_(n, m):
        return mp.sqrt(mpf((n + 1 + m) * (n + 1 - m)) / ((2 * n + 1) * (2 * n + 3)))

    def b_(n, m):
        return mp.sqrt(mpf((n - m - 1) * (n - m)) / ((2 * n - 1) * (2 * n + 1)))

    def g_(n, m):
        return mp.sqrt(mpf((n + m + 1) * (n + m + 2)) / ((2 * n + 1) * (2 * n + 3)))

    for label, k in KS.items():
        for kind, r in (("RR", mpf("0.4")), ("SR", mpf("0.25"))):
            f_fun = sph_j if kind[0] == "R" else sph_h
            tabs = {m: coaxial(kind, k, t, m, nmax, r) for m in range(0, 4)}

            def e(m, nn, n):
                """E^m_{n'n}, zero outside n, n' >= m."""
                if n < m or nn < m:
                    return mpf(0)
                return tabs[m][(n, nn, m)]

            scale_ = max(abs(v) for tab in tabs.values() for v in tab.values())
            e_start = max(abs(e(0, nn, 0) - (-1) ** nn * mp.sqrt(2 * nn + 1) * f_fun(nn, k * t))
                          for nn in range(0, nmax + 1)) / scale_
            e_n = e_m = mpf(0)
            for m in range(0, 4):
                for n in range(m, nmax - 1):
                    for nn in range(m, nmax):
                        lhs = a_(n, m) * e(m, nn, n + 1)
                        rhs = (a_(n - 1, m) * e(m, nn, n - 1) - a_(nn, m) * e(m, nn + 1, n)
                               + a_(nn - 1, m) * e(m, nn - 1, n))
                        e_n = max(e_n, abs(lhs - rhs) / scale_)
                if m < 3:
                    for nn in range(m + 1, nmax):
                        lhs = g_(m, m) * e(m + 1, nn, m + 1)
                        rhs = b_(nn + 1, m) * e(m, nn + 1, m) + g_(nn - 1, m) * e(m, nn - 1, m)
                        e_m = max(e_m, abs(lhs - rhs) / scale_)
            tag = f"({kind[0]}|{kind[1]}), {label} k"
            check(f"start column, {tag}", e_start, mpf(10) ** -20)
            check(f"recursion in n, {tag}", e_n, mpf(10) ** -20)
            check(f"recursion in m, {tag}", e_m, mpf(10) ** -20)


def check_yukawa_scaled():
    print("12. Yukawa's stored forms with i^ and k^ (item 3)")
    kappa = mpf("2.3")
    k = I * kappa
    p = 30
    for a in [mpf("0.15"), mpf("1.2")]:
        s, beta = scale(k, a), kappa * a
        centre = [mpf(0)] * 3
        q = charges(4, False)
        inside = [(rand_point(0, a, centre), qq) for qq in q]
        outside = [(rand_point(4 * a, 5 * a, centre), qq) for qq in q]
        m_hat = stored_p2m(k, a, centre, inside, p, [])
        l_hat = stored_p2l(k, a, centre, outside, p, [])
        err = mpf(0)
        for n in range(p + 1):
            for m in range(-n, n + 1):
                my = sum(qq * mp.exp(-kappa * norm(y)) * sph_i(n, kappa * norm(y)) / s**n
                         * mp.exp(kappa * norm(y) - beta) * ylm(n, m, y) for y, qq in inside)
                ly = sum(4 * mp.pi * kappa * qq * mp.exp(kappa * norm(y))
                         * sph_k(n, kappa * norm(y)) * s**n * mp.exp(beta - kappa * norm(y))
                         * ylm(n, m, y) for y, qq in outside)
                err = max(err, abs(I ** (-n) * m_hat[(n, m)] - my) / max(abs(my), mpf(10) ** -300),
                          abs(I**n * l_hat[(n, m)] - ly) / max(abs(ly), mpf(10) ** -300))
        check(f"M^Y = i^(-n) M^ and L^Y = i^n L^ from i^ and k^, kappa a = {mp.nstr(beta, 2)}",
              err, mpf(10) ** -28)


def main():
    check_kernel()
    check_radial()
    check_harmonics()
    check_series()
    check_leaf_operators()
    check_yukawa()
    check_rotations()
    check_coaxial()
    check_m2l_chain()
    check_m2m_l2l_chain()
    check_recursions()
    check_yukawa_scaled()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
