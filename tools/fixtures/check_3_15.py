#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Numerical check of the identities of docs/CONVENTIONS.md §3.15 (Helmholtz CH1.1, CH1.3d2).

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
 12. Yukawa's stored forms with i^ and k^;
 13. the plane-wave expansion e^{ik s.r} = 4 pi sum_n i^n j_n(kr) sum_m Y(r^) Y(s)
     ("Signatures"; Helmholtz CH1.3d2);
 14. the signature of charges in stored form: 4 pi e^beta F^(s) = sum q e^{-ik s.(y - c)};
 15. the signature grid: the reference's N_phi = 4P, the symmetric rings, the quadrature exact for the
     products of harmonics to degree 4P - 4;
 16. the 34 classes of the 316 offsets and their sample permutations: the sample of S s
     is S s for every symmetry of an offset;
 17. the diagonal M2L in stored form on the grid (G = 4 pi ik e^{2 beta - Im(k)|t|} T^ F^,
     L^ = s^n i^n sum w Y G): equal to check 9's (S|R) M2L on the axis, and against the
     direct sum off it;
 18. the level units ("Level units"): in units of r with k^ = k r, P2M gives the same
     numbers and P2L r times them, M2P and L2P give r phi~; L2L from the parent's units
     to the child's with the factor 1/2, and P2L from a coarser source leaf with its
     charges times r_t / r_s.

Run from the repository root:  uv run tools/fixtures/check_3_15.py
(or python3 with mpmath==1.4.1). It takes about five minutes.
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


# --- 13-17. signatures and the diagonal form ("Signatures") -------------------------------


def legendre_column(m, nmax, t):
    """P_n^m(t), n = m..nmax, without the Condon-Shortley phase (§3.3), by one recursion."""
    pmm = mp.fac2(2 * m - 1) * (1 - t * t) ** (mpf(m) / 2) if m > 0 else mpf(1)
    out = [pmm]
    if nmax > m:
        out.append(t * (2 * m + 1) * pmm)
    for nn in range(m + 2, nmax + 1):
        out.append((t * (2 * nn - 1) * out[-1] - (nn + m - 1) * out[-2]) / (nn - m))
    return out


def y_all(p, x):
    """{(n, m): Y_n^m(x^)} for n <= p, the harmonics of ylm by one recursion per order."""
    _, t, phi = angles(x)
    out = {}
    for am in range(0, p + 1):
        col = legendre_column(am, p, t)
        for n in range(am, p + 1):
            c = mp.sqrt((2 - (am == 0)) * (2 * n + 1) / (4 * mp.pi)
                        * mp.factorial(n - am) / mp.factorial(n + am))
            if am == 0:
                out[(n, 0)] = c * col[n]
            else:
                out[(n, am)] = c * col[n - am] * mp.cos(am * phi)
                out[(n, -am)] = c * col[n - am] * mp.sin(am * phi)
    return out


def azimuth_count(p):
    """N_phi = 4P - 3 rounded up to a multiple of 4, P = p + 1."""
    n = 4 * (p + 1) - 3
    return n + (-n) % 4


def grid(p):
    """The signature grid of degree p: 2P - 1 Gauss-Legendre rings in cos(theta),
    ascending, N_phi azimuths 2 pi j / N_phi; sample (i, j) at i N_phi + j. Returns the
    rings (u_i, w_i), N_phi and the list of (direction, weight w_i 2 pi / N_phi)."""
    rings = sorted(zip(*mp.gauss_quadrature(2 * p + 1, "legendre")), key=lambda r: r[0])
    nphi = azimuth_count(p)
    samples = []
    for u, w in rings:
        st = mp.sqrt(1 - u * u)
        for j in range(nphi):
            ph = 2 * mp.pi * j / nphi
            samples.append(([st * mp.cos(ph), st * mp.sin(ph), u], w * 2 * mp.pi / nphi))
    return rings, nphi, samples


def check_plane_wave_expansion():
    print("13. e^{ik s.r} = 4 pi sum_n i^n j_n(kr) sum_m Y_n^m(r^) Y_n^m(s) (Signatures)")
    for label, k in KS.items():
        err = mpf(0)
        for _ in range(3):
            r, s = rand_point(0.2, 1.5), rand_unit()
            yr, ys = y_all(45, r), y_all(45, s)
            got = 4 * mp.pi * sum(I**n * sph_j(n, k * norm(r))
                                  * sum(yr[(n, m)] * ys[(n, m)] for m in range(-n, n + 1))
                                  for n in range(46))
            want = mp.exp(I * k * sum(s[i] * r[i] for i in range(3)))
            err = max(err, abs(got - want) / abs(want))
        check(f"plane-wave expansion, {label} k", err, mpf(10) ** -28)


def signature(p, s_l, m_hat, direction):
    """F^(s) = sum_n (-i)^n s^n sum_m M^_n^m Y_n^m(s), the stored signature."""
    ys = y_all(p, direction)
    return sum((-I) ** n * s_l**n * sum(m_hat[(n, m)] * ys[(n, m)] for m in range(-n, n + 1))
               for n in range(p + 1))


def check_signature_of_charges():
    print("14. the signature of charges: 4 pi e^beta F^(s) = sum q e^{-ik s.(y - c)}")
    p = 30
    centre = [mpf("0.2"), mpf("-0.1"), mpf("0.3")]
    # k a = 2 (s = 1) and 0.4 (s = 0.4 < 1, so the s^n of F^ is exercised).
    cases = [(mpf("0.5"), "real", mpf("4")), (mpf("0.5"), "complex", mpc("4", "0.4")),
             (mpf("0.5"), "yukawa", mpc(0, "4")), (mpf("0.1"), "real", mpf("4")),
             (mpf("0.1"), "complex", mpc("4", "0.4"))]
    for a, label, k in cases:
        for cplx in [False, True]:
            q = charges(4, cplx)
            inside = [(rand_point(0, a, centre), qq) for qq in q]
            m_hat = stored_p2m(k, a, centre, inside, p, [])
            s_l, beta = scale(k, a), mp.im(k) * a
            err = mpf(0)
            for _ in range(3):
                d = rand_unit()
                got = 4 * mp.pi * mp.exp(beta) * signature(p, s_l, m_hat, d)
                want = sum(qq * mp.exp(-I * k * sum(d[i] * (y[i] - centre[i]) for i in range(3)))
                           for y, qq in inside)
                err = max(err, abs(got - want) / max(abs(qq) for _, qq in inside))
            tag = f"{label} k, k a = {mp.nstr(abs(k) * a, 2)}, {'complex' if cplx else 'real'} q"
            # Truncation at p = 30 with |k| a <= 2: j_31(2)-sized, below 1e-30.
            check(f"signature of charges, {tag}", err, mpf(10) ** -25)


def check_grid():
    print("15. the signature grid: N_phi, the ring symmetry, exact to degree 4P - 4")
    for p in range(0, 9):
        n = azimuth_count(p)
        assert n % 4 == 0 and n == 4 * (p + 1), (p, n)
    check("N_phi = 4P - 3 rounded up to a multiple of 4 = 4P, p <= 8", mpf(0), mpf(0))
    p = 3
    rings, nphi, samples = grid(p)
    e_sym = max(abs(rings[i][0] + rings[2 * p - i][0]) for i in range(2 * p + 1))
    check("nodes symmetric, u_{2P-2-i} = -u_i (z -> -z maps ring i to 2p - i)", e_sym,
          mpf(10) ** -35)
    ys = [(y_all(2 * p, d), w) for d, w in samples]
    keys = [(n, m) for n in range(2 * p + 1) for m in range(-n, n + 1)]
    err = mpf(0)
    for a_ in keys:
        for b_ in keys:
            g = sum(w * y[a_] * y[b_] for y, w in ys)
            err = max(err, abs(g - (1 if a_ == b_ else 0)))
    check(f"Y_n^m orthonormal on the grid for n, n' <= 2p (degree 4P - 4), p = {p}", err,
          mpf(10) ** -35)


def offsets():
    return [(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)
            if max(abs(i), abs(j), abs(k)) >= 2]


def symmetry_of(o):
    """S of an offset: flip its negative components, then swap x and y if |o_x| > |o_y|."""
    return [o[0] < 0, o[1] < 0, o[2] < 0], abs(o[0]) > abs(o[1])


def apply_symmetry(sym, x):
    flip, swap = sym
    y = [-x[i] if flip[i] else x[i] for i in range(3)]
    return [y[1], y[0], y[2]] if swap else y


def sample_of(sym, rings, nphi, i, j):
    """The sample of S s for sample (i, j): x -> -x maps j to (N/2 - j) mod N, y -> -y to
    -j mod N, z -> -z ring i to rings - 1 - i, x <-> y j to (N/4 - j) mod N, in that order."""
    flip, swap = sym
    if flip[0]:
        j = (nphi // 2 - j) % nphi
    if flip[1]:
        j = (-j) % nphi
    if flip[2]:
        i = rings - 1 - i
    if swap:
        j = (nphi // 4 - j) % nphi
    return i, j


def check_classes():
    print("16. the 34 classes of the 316 offsets and their sample permutations")
    reps = {}
    e_rep = 0
    for o in offsets():
        sym = symmetry_of(o)
        rep = tuple(apply_symmetry(sym, list(o)))
        want = (min(abs(o[0]), abs(o[1])), max(abs(o[0]), abs(o[1])), abs(o[2]))
        e_rep = max(e_rep, int(rep != want))
        reps[rep] = reps.get(rep, 0) + 1
    check("S o = (min(|a|,|b|), max(|a|,|b|), |c|) for every offset", mpf(e_rep), mpf(0))
    check("34 classes of 316 offsets", mpf(abs(len(reps) - 34) + abs(sum(reps.values()) - 316)),
          mpf(0))
    p = 2
    rings, nphi, samples = grid(p)
    nr = len(rings)
    err = mpf(0)
    syms = {(tuple(f), s) for f, s in (symmetry_of(o) for o in offsets())}
    for flip, swap in syms:
        sym = (list(flip), swap)
        for i in range(nr):
            for j in range(nphi):
                ii, jj = sample_of(sym, nr, nphi, i, j)
                want = apply_symmetry(sym, samples[i * nphi + j][0])
                got = samples[ii * nphi + jj][0]
                err = max(err, max(abs(got[c] - want[c]) for c in range(3)))
    check(f"the sample of S s is S s, every S of the offsets, p = {p} "
          f"({len(syms)} symmetries)", err, mpf(10) ** -35)


def transfer_coefficients(p, k, t):
    """e^{Im z} (2l + 1) i^l h_l(z), l <= 2p, z = k|t|: T^(s) = sum_l c_l P_l(s . t^)."""
    z = k * norm(t)
    return [mp.exp(mp.im(z)) * (2 * l + 1) * I**l * sph_h(l, z) for l in range(2 * p + 1)]


def transfer_stored(coeffs, t, direction):
    """T^(s) = e^{Im z} sum_{l <= 2p} (2l + 1) i^l h_l(z) P_l(s . t^), z = k|t|."""
    c = sum(direction[i] * t[i] for i in range(3)) / norm(t)
    return sum(cl * mp.legendre(l, c) for l, cl in enumerate(coeffs))


def diagonal_m2l(p, k, a, t, m_hat):
    """The diagonal M2L in stored form on the grid of degree p:
    G = 4 pi ik e^{2 beta - Im(k)|t|} T^ F^, L^_n^m = s^n i^n sum w Y_n^m G."""
    s_l, beta = scale(k, a), mp.im(k) * a
    _, _, samples = grid(p)
    factor = 4 * mp.pi * I * k * mp.exp(2 * beta - mp.im(k) * norm(t))
    coeffs = transfer_coefficients(p, k, t)
    l_hat = {(n, m): mpf(0) for n in range(p + 1) for m in range(-n, n + 1)}
    for d, w in samples:
        ys = y_all(p, d)
        f = sum((-I) ** n * s_l**n * sum(m_hat[(n, m)] * ys[(n, m)] for m in range(-n, n + 1))
                for n in range(p + 1))
        g = factor * transfer_stored(coeffs, t, d) * f
        for (n, m), y in ys.items():
            l_hat[(n, m)] += s_l**n * I**n * w * y * g
    return l_hat


def coaxial_m2l(p, k, a, t, m_hat):
    """Check 9's M2L on the z-axis: the stored (S|R) table, 4 pi ik and e^{2 beta - Im(k) t}."""
    s_l, beta = scale(k, a), mp.im(k) * a
    scalar = mp.exp(2 * beta - mp.im(k) * t)
    l_hat = {}
    for m in range(0, p + 1):
        sr = coaxial("SR", k, t, m, p, mpf("0.25"))
        for slot in ([0] if m == 0 else [m, -m]):
            for nn in range(m, p + 1):
                acc = sum(mp.exp(mp.im(k * t)) * s_l ** (n + nn) * sr[(n, nn, slot)]
                          * m_hat[(n, slot)] for n in range(m, p + 1))
                l_hat[(nn, slot)] = 4 * mp.pi * I * k * scalar * acc
    return l_hat


def check_diagonal_m2l():
    print("17. the diagonal M2L in stored form (Signatures)")
    p = 6
    for label, k in KS.items():
        for cplx in [False, True]:
            q = charges(3, cplx)
            tag = f"{label} k, {'complex' if cplx else 'real'} q"
            # On the axis it is the truncated (S|R) M2L of check 9, exactly.
            a, t = mpf("0.3"), mpf("1.1")
            src = [mpf(0)] * 3
            inside = [(rand_point(0, a, src), qq) for qq in q]
            m_hat = stored_p2m(k, a, src, inside, p, [])
            got = diagonal_m2l(p, k, a, [mpf(0), mpf(0), t], m_hat)
            want = coaxial_m2l(p, k, a, t, m_hat)
            scale_ = max(abs(v) for v in want.values())
            err = max(abs(got[key] - want[key]) for key in want) / scale_
            check(f"diagonal = (S|R) M2L on the axis, p = {p}, {tag}", err, mpf(10) ** -25)
            # Off the axis, a V offset (2, 1, -1) of boxes of half-width 1/4: against the
            # direct sum, with charges and targets within 0.05 of the centres.
            r = mpf(1) / 4
            a = mp.sqrt(3) * r
            t = [2 * r * c for c in (2, 1, -1)]
            inside = [(rand_point(0, mpf("0.05"), src), qq) for qq in q]
            m_hat = stored_p2m(k, a, src, inside, p, [])
            l_hat = diagonal_m2l(p, k, a, t, m_hat)
            err = mpf(0)
            for _ in range(2):
                x = rand_point(0, mpf("0.05"), t)
                want = sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in inside)
                got = stored_l2p(k, a, t, l_hat, x, p, [])
                err = max(err, abs(got - want) / max(abs(want), mpf(10) ** -30))
            # Truncation (0.1/|t|)^7 with |t| = 1.22: about 2e-8.
            check(f"diagonal M2L off the axis against the direct sum, {tag}", err,
                  mpf(10) ** -6)


# --- 18. level units ("Level units") ----------------------------------------------------


def check_level_units():
    print("18. level units: the forms in units of r, k^ = k r (Level units)")
    p = 30
    for label, k in KS.items():
        for cplx in [False, True]:
            tag = f"{label} k, {'complex' if cplx else 'real'} q"
            q = charges(3, cplx)
            # (a) P2M: the same stored numbers in units r; M2P there gives r phi~.
            r = mpf("0.375")
            a = mp.sqrt(3) * r
            c = [mpf("0.1"), mpf("0.2"), mpf("-0.3")]
            unit = lambda x: [v / r for v in x]  # noqa: E731
            inside = [(rand_point(0, a, c), qq) for qq in q]
            m_abs = stored_p2m(k, a, c, inside, p, [])
            m_lev = stored_p2m(k * r, a / r, unit(c), [(unit(y), qq) for y, qq in inside], p, [])
            e_m = max(abs(m_abs[key] - m_lev[key]) for key in m_abs) / max(
                abs(v) for v in m_abs.values())
            x = rand_point(4 * a, 5 * a, c)
            want = r * sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in inside)
            got = stored_m2p(k * r, a / r, unit(c), m_lev, unit(x), p, [])
            e_mp = abs(got - want) / abs(want)
            # (b) P2L: r times the stored numbers; L2P there gives r phi~.
            outside = [(rand_point(4 * a, 5 * a, c), qq) for qq in q]
            l_abs = stored_p2l(k, a, c, outside, p, [])
            l_lev = stored_p2l(k * r, a / r, unit(c), [(unit(y), qq) for y, qq in outside], p, [])
            e_l = max(abs(r * l_abs[key] - l_lev[key]) for key in l_abs) / max(
                abs(r * v) for v in l_abs.values())
            x = rand_point(0, a, c)
            want = r * sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in outside)
            got = stored_l2p(k * r, a / r, unit(c), l_lev, unit(x), p, [])
            e_lp = abs(got - want) / abs(want)
            check(f"P2M the same numbers in units r, {tag}", e_m, mpf(10) ** -30)
            check(f"M2P in units r gives r phi~, {tag}", e_mp, mpf(10) ** -14)
            check(f"P2L r times the numbers in units r, {tag}", e_l, mpf(10) ** -30)
            check(f"L2P in units r gives r phi~, {tag}", e_lp, mpf(10) ** -14)
            # (c) L2L from the parent's units to the child's carries 1/2: on the z-axis,
            # parent at 0 with half-width r_p = 2 r_c, child at -r_c z (a corner direction
            # is not needed: the factor is the units', not the shift's).
            r_c = mpf("0.25")
            r_p = 2 * r_c
            a_c, a_p = mp.sqrt(3) * r_c, mp.sqrt(3) * r_p
            parent, child = [mpf(0)] * 3, [mpf(0), mpf(0), -a_c]
            far = [(rand_point(14, 16, parent), qq) for qq in q]
            kp, kc = k * r_p, k * r_c
            l_p = stored_p2l(kp, a_p / r_p, parent, [([v / r_p for v in y], qq) for y, qq in far],
                             p, [])
            s_c, s_p = scale(kc, a_c / r_c), scale(kp, a_p / r_p)
            l_c = {}
            pc = 14  # the projected table's degree (its 30-point rule resolves n + n' <= 28)
            t_child = -a_c / r_c  # the shift c - P in the child's units
            for m in range(0, pc + 1):
                rr = coaxial("RR", kc, t_child, m, pc, a_c / r_c / 2)
                damp = mp.exp(-mp.im(kc * a_c / r_c))
                for slot in ([0] if m == 0 else [m, -m]):
                    for nn in range(m, pc + 1):
                        l_c[(nn, slot)] = mpf(1) / 2 * s_c**nn * sum(
                            damp * rr[(n, nn, slot)] * s_p ** (-n) * l_p[(n, slot)]
                            for n in range(m, pc + 1))
            err = mpf(0)
            for _ in range(2):
                x = rand_point(0, a_c / 2, child)
                want = r_c * sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in far)
                got = stored_l2p(kc, a_c / r_c, [v / r_c for v in child], l_c,
                                 [v / r_c for v in x], pc, [])
                err = max(err, abs(got - want) / abs(want))
            # Truncation at degree 14, measured below 1e-16 but 2.6e-14 for Yukawa (whose
            # field at the targets is e^{-35}); a wrong factor of the units errs by O(1).
            check(f"L2L in level units with the factor 1/2 gives r_c phi~, {tag}", err,
                  mpf(10) ** -12)
            # (d) P2L from a coarser source leaf (r_s = 2 r_t) in the source's units, its
            # charges scaled by r_t / r_s, gives the target box's r_t phi~.
            r_t, r_s = mpf("0.25"), mpf("0.5")
            a_t = mp.sqrt(3) * r_t
            c_t = [mpf("0.1"), mpf("0"), mpf("0.2")]
            srcs = [(rand_point(4 * a_t, 5 * a_t, c_t), qq) for qq in q]
            l_t = stored_p2l(k * r_s, a_t / r_s, [v / r_s for v in c_t],
                             [([v / r_s for v in y], qq * r_t / r_s) for y, qq in srcs], p, [])
            x = rand_point(0, a_t, c_t)
            want = r_t * sum(qq * 4 * mp.pi * kernel(k, x, y) for y, qq in srcs)
            got = stored_l2p(k * r_t, a_t / r_t, [v / r_t for v in c_t], l_t,
                             [v / r_t for v in x], p, [])
            check(f"P2L in the source leaf's units with charges r_t/r_s gives r_t phi~, {tag}",
                  abs(got - want) / abs(want), mpf(10) ** -14)


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
    check_plane_wave_expansion()
    check_signature_of_charges()
    check_grid()
    check_classes()
    check_diagonal_m2l()
    check_level_units()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
