#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""High-precision check of the translation operators of docs/CONVENTIONS.md §3.11.

Checks the formulas of §3.11 in mpmath at 40 significant digits, for seeded random
frames (centre c, scaling radius r) and charges, with the solid harmonics of
gen_harmonics.py (the definitions of §3.3, imported, not copied):
  - M2M: P2M at the child, then M2M, equals P2M at the parent (exact, 1e-30);
  - L2L: a random local expansion evaluated after L2L equals the same expansion
    evaluated before, at points inside the output sphere (exact, 1e-30);
  - M2L: with input degree 60 and output degree 8, M2L after P2M equals P2L at the
    target frame within the truncation bound of §3.11, coefficient by coefficient
    (plus 1e-30 relative to the sum of term magnitudes, where the bound falls below
    the working precision);
  - coaxial forms equal the general ones for shifts along the z-axis, d > 0 and d < 0
    (1e-30);
  - the rotation rule for coefficients, K D^n K for multipoles and K S D^n S^-1 K for
    locals (n <= 8, 1e-30), with D^n fitted from the definition R_n(Qx) = D^n R_n(x)
    of §3.8, and rotate-coaxial-rotate back equals the general form (1e-30);
  - L2P and M2P gradients from the ladder of §3.4 and §3.11 equal central differences
    of the potential (1e-30);
  - M2M, L2L and M2L evaluated from real storage through the order-sum identities of
    §3.11 (stored orders m >= 0 of the input only) equal the complex forms (1e-30).

Coefficient vectors are compared per degree in the orthonormal basis of §3.8 (slot m
weighted by N_m for multipoles, N_m / S_m for locals), relative to the weighted norm of
the reference in the same degree; potentials relative to the sum of term magnitudes.

The harmonics are evaluated with guard digits (HARMONIC_DPS), because the monomial
Legendre polynomials of gen_harmonics.py cancel at high degree; everything else runs at
40 digits.

Run from the repository root:  uv run tools/fixtures/check_translations.py
(or: python3 tools/fixtures/check_translations.py with mpmath==1.4.1 installed).
Prints the worst error per check and exits non-zero on any failure.
"""
import random
import sys
from math import factorial
from pathlib import Path

# Import gen_harmonics.py without leaving a __pycache__ in the repository.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_harmonics as gh  # noqa: E402  (the definitions of §3.3)
import mpmath  # noqa: E402
from mpmath import mp, mpc, mpf  # noqa: E402

DPS = 40
HARMONIC_DPS = 100
TOL = mpf(10) ** -30
SEED = 20261002
SQRT3 = None  # set in main() at working precision

_coef_cache = {}


def harmonics(p, x, regular):
    """Complex X_n^m(x), 0 <= m <= n <= p, in gen_harmonics' order (index gh.tri(n, m))."""
    if p not in _coef_cache:
        _coef_cache[p] = gh.legendre_derivative_coefficients(p)
    with mp.workdps(HARMONIC_DPS):
        values = gh.harmonics(_coef_cache[p], p, x, regular)
    return [+v for v in values]  # round to the working precision


def at(v, n, m):
    """Order-m value (any integer m) of degree n from a list of m >= 0 values, with
    X_n^-m = (-1)^m conj(X_n^m) (§3.3, §3.6) and X_j^i = 0 for |i| > j."""
    if n < 0 or abs(m) > n:
        return mpc(0)
    z = v[gh.tri(n, abs(m))]
    return z if m >= 0 else (-1) ** m * mpmath.conj(z)


def zeros(p):
    return [mpc(0)] * gh.tri(p + 1, 0)


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def scaled(x, frame):
    c, r = frame
    return [(xi - ci) / r for xi, ci in zip(x, c)]


# ---------------------------------------------------------------------------------------
# Leaf operators (§3.7) and evaluation (§3.6).


def p2m(p, frame, sources, charges):
    out = zeros(p)
    for y, q in zip(sources, charges):
        R = harmonics(p, scaled(y, frame), True)
        out = [o + q * mpmath.conj(v) for o, v in zip(out, R)]
    return out


def p2l(p, frame, sources, charges):
    out = zeros(p)
    for y, q in zip(sources, charges):
        I = harmonics(p, scaled(y, frame), False)
        out = [o + q * mpmath.conj(v) for o, v in zip(out, I)]
    return out


def evaluate(p, frame, coeffs, x, regular):
    """(1/r) sum_n sum_m C_n^m X_n^m(v) and the sum of term magnitudes."""
    X = harmonics(p, scaled(x, frame), regular)
    total, size = mpc(0), mpf(0)
    for n in range(p + 1):
        for m in range(-n, n + 1):
            t = at(coeffs, n, m) * at(X, n, m)
            total += t
            size += abs(t)
    r = frame[1]
    return total.real / r, size / r


# ---------------------------------------------------------------------------------------
# Translations of §3.11, general form. Input frame (c, r), output frame (c', r').


def m2m(p, fin, fout, M):
    (c, r), (c2, r2) = fin, fout
    rho = r / r2
    b = [(ci - c2i) / r2 for ci, c2i in zip(c, c2)]
    Rb = harmonics(p, b, True)
    out = zeros(p)
    for j in range(p + 1):
        for i in range(j + 1):
            s = mpc(0)
            for k in range(j + 1):
                for l in range(-k, k + 1):
                    s += rho**k * at(M, k, l) * mpmath.conj(at(Rb, j - k, i - l))
            out[gh.tri(j, i)] = s
    return out


def l2l(p, fin, fout, L):
    (c, r), (c2, r2) = fin, fout
    sigma = r2 / r
    t = [(c2i - ci) / r for ci, c2i in zip(c, c2)]
    Rt = harmonics(p, t, True)
    out = zeros(p)
    for j in range(p + 1):
        for i in range(j + 1):
            s = mpc(0)
            for n in range(j, p + 1):
                for m in range(-n, n + 1):
                    s += at(L, n, m) * at(Rt, n - j, m - i)
            out[gh.tri(j, i)] = sigma ** (j + 1) * s
    return out


def m2l(p_in, p_out, fin, fout, M):
    (c, r), (c2, r2) = fin, fout
    sigma = r2 / r
    b = [(c2i - ci) / r for ci, c2i in zip(c, c2)]
    Ib = harmonics(p_in + p_out, b, False)
    out = zeros(p_out)
    for j in range(p_out + 1):
        for i in range(j + 1):
            s = mpc(0)
            for n in range(p_in + 1):
                for m in range(-n, n + 1):
                    s += at(M, n, m) * at(Ib, n + j, m - i)
            out[gh.tri(j, i)] = (-1) ** (j + i) * sigma ** (j + 1) * s
    return out


# Coaxial forms: c' - c = d e_z. M2M and L2L hold for either sign of d, M2L for d > 0.


def m2m_coaxial(p, r, r2, d, M):
    rho, beta = r / r2, d / r2
    out = zeros(p)
    for j in range(p + 1):
        for i in range(j + 1):
            out[gh.tri(j, i)] = sum(
                (rho**k * M[gh.tri(k, i)] * (-beta) ** (j - k) / factorial(j - k) for k in range(i, j + 1)),
                mpc(0),
            )
    return out


def l2l_coaxial(p, r, r2, d, L):
    sigma, beta = r2 / r, d / r
    out = zeros(p)
    for j in range(p + 1):
        for i in range(j + 1):
            s = sum((L[gh.tri(n, i)] * beta ** (n - j) / factorial(n - j) for n in range(j, p + 1)), mpc(0))
            out[gh.tri(j, i)] = sigma ** (j + 1) * s
    return out


def m2l_coaxial(p_in, p_out, r, r2, d, M):
    """Coaxial M2L for either sign of d: (sgn d)^(n + j) (r / |d|)^(n + j + 1) (§3.11)."""
    sigma, beta, sign = r2 / r, abs(d) / r, 1 if d > 0 else -1
    out = zeros(p_out)
    for j in range(p_out + 1):
        for i in range(j + 1):
            s = sum(
                (
                    M[gh.tri(n, i)] * sign ** (n + j) * factorial(n + j) / beta ** (n + j + 1)
                    for n in range(i, p_in + 1)
                ),
                mpc(0),
            )
            out[gh.tri(j, i)] = (-1) ** (j + i) * sigma ** (j + 1) * s
    return out


# Real-storage forms (§3.6, §3.11): inputs and outputs are (p + 1)^2 reals at index
# n^2 + n + m; the order sums use the stored orders m >= 0 of the input only, through the
# two identities of §3.11, and read the shift's harmonics with the (-1)^m rule.


def real_vector(p, v):
    return gh.real_storage(p, v)


def complex_vector(p, x):
    out = zeros(p)
    for n in range(p + 1):
        out[gh.tri(n, 0)] = mpc(x[n * n + n])
        for m in range(1, n + 1):
            out[gh.tri(n, m)] = mpc(x[n * n + n + m], x[n * n + n - m])
    return out


def stored(x, n, m):
    """Order m >= 0 of degree n from real storage: (slot m, slot -m), imaginary 0 at m = 0."""
    return mpc(x[n * n + n], 0) if m == 0 else mpc(x[n * n + n + m], x[n * n + n - m])


def read(x, n, m):
    """Any order m, by the rule of §3.11: order -m < 0 is (-1)^m (slot m, -slot -m);
    orders beyond the degree are 0."""
    if n < 0 or abs(m) > n:
        return mpc(0)
    if m >= 0:
        return stored(x, n, m)
    k = -m
    return (-1) ** k * mpc(x[n * n + n + k], -x[n * n + n - k])


def convolution(x, k, i, B):
    """sum_{l=-k}^{k} A^l B^(i-l), A the degree-k input: first identity of §3.11."""
    s = stored(x, k, 0) * B(i)
    for l in range(1, k + 1):
        a = stored(x, k, l)
        s += a * B(i - l) + (-1) ** l * mpmath.conj(a) * B(i + l)
    return s


def correlation(x, n, i, B):
    """sum_{m=-n}^{n} A^m B^(m-i), A the degree-n input: second identity of §3.11."""
    s = stored(x, n, 0) * B(-i)
    for m in range(1, n + 1):
        a = stored(x, n, m)
        s += a * B(m - i) + (-1) ** m * mpmath.conj(a) * B(-m - i)
    return s


def store(out, j, i, z):
    out[j * j + j + i] = z.real
    if i > 0:
        out[j * j + j - i] = z.imag


def m2m_real(p, fin, fout, x):
    (c, r), (c2, r2) = fin, fout
    rho = r / r2
    b = [(ci - c2i) / r2 for ci, c2i in zip(c, c2)]
    # conj R of the shift, in real storage; it satisfies the (-1)^m rule too.
    B = real_vector(p, [mpmath.conj(v) for v in harmonics(p, b, True)])
    out = [mpf(0)] * (p + 1) ** 2
    for j in range(p + 1):
        for i in range(j + 1):
            z = sum((rho**k * convolution(x, k, i, lambda q: read(B, j - k, q)) for k in range(j + 1)), mpc(0))
            store(out, j, i, z)
    return out


def l2l_real(p, fin, fout, x):
    (c, r), (c2, r2) = fin, fout
    sigma = r2 / r
    t = [(c2i - ci) / r for ci, c2i in zip(c, c2)]
    B = real_vector(p, harmonics(p, t, True))
    out = [mpf(0)] * (p + 1) ** 2
    for j in range(p + 1):
        for i in range(j + 1):
            z = sum((correlation(x, n, i, lambda q: read(B, n - j, q)) for n in range(j, p + 1)), mpc(0))
            store(out, j, i, sigma ** (j + 1) * z)
    return out


def m2l_real(p_in, p_out, fin, fout, x):
    (c, r), (c2, r2) = fin, fout
    sigma = r2 / r
    b = [(c2i - ci) / r for ci, c2i in zip(c, c2)]
    B = real_vector(p_in + p_out, harmonics(p_in + p_out, b, False))
    out = [mpf(0)] * (p_out + 1) ** 2
    for j in range(p_out + 1):
        for i in range(j + 1):
            z = sum((correlation(x, n, i, lambda q: read(B, n + j, q)) for n in range(p_in + 1)), mpc(0))
            store(out, j, i, (-1) ** (j + i) * sigma ** (j + 1) * z)
    return out


# ---------------------------------------------------------------------------------------
# Error measures.


def weight(n, m, local):
    """N_m, or N_m / S_m for locals (§3.8), for the complex order-m entry; the factor
    c_m^2 = 2 for m > 0 accounts for its two real slots."""
    f = mpmath.sqrt(factorial(n + m) * factorial(n - m))
    return f if not local else 1 / f


def degree_norms(p, v, local):
    out = []
    for n in range(p + 1):
        s = mpf(0)
        for m in range(n + 1):
            c2 = 1 if m == 0 else 2
            s += c2 * (weight(n, m, local) * abs(v[gh.tri(n, m)])) ** 2
        out.append(mpmath.sqrt(s))
    return out


def degree_error(p, got, want, local):
    """max_n ||W (got - want)_n||_2 / ||W want_n||_2 (degrees with want_n = 0 must match)."""
    errs = degree_norms(p, sub(got, want), local)
    refs = degree_norms(p, want, local)
    worst = mpf(0)
    for e, s in zip(errs, refs):
        worst = max(worst, e / s if s > 0 else (mpf(0) if e == 0 else mpf("inf")))
    return worst


# ---------------------------------------------------------------------------------------
# Random data. Only Python's random.Random with a fixed seed, converted to mpf.


def unit_vector(rng):
    while True:
        v = [mpf(2 * rng.random() - 1) for _ in range(3)]
        s = mpmath.sqrt(sum(c * c for c in v))
        if mpf("0.1") < s <= 1:
            return [c / s for c in v]


def point_in_ball(rng, centre, radius):
    u = unit_vector(rng)
    t = radius * mpf(rng.random())
    return [c + t * e for c, e in zip(centre, u)]


def random_frame(rng):
    c = [mpf(4 * rng.random() - 2) for _ in range(3)]
    return (c, mpf("0.25") + 2 * mpf(rng.random()))


def charges(rng, count):
    return [mpf(2 * rng.random() - 1) for _ in range(count)]


def random_coefficients(rng, p):
    """Random complex coefficients for m >= 0; order 0 is real (§3.6)."""
    out = zeros(p)
    for n in range(p + 1):
        out[gh.tri(n, 0)] = mpc(2 * rng.random() - 1)
        for m in range(1, n + 1):
            out[gh.tri(n, m)] = mpc(2 * rng.random() - 1, 2 * rng.random() - 1)
    return out


# ---------------------------------------------------------------------------------------
# Checks. Each returns (worst error, tolerance, detail).


def check_m2m(rng):
    p, worst = 20, mpf(0)
    for trial in range(4):
        child = random_frame(rng)
        if trial == 0:
            # Octree geometry: parent radius 2r, child centre c' + r (+-1, +-1, +-1).
            parent = ([ci - child[1] for ci in child[0]], 2 * child[1])
        else:
            parent = random_frame(rng)
        sources = [point_in_ball(rng, child[0], SQRT3 * child[1]) for _ in range(8)]
        q = charges(rng, len(sources))
        got = m2m(p, child, parent, p2m(p, child, sources, q))
        worst = max(worst, degree_error(p, got, p2m(p, parent, sources, q), local=False))
    return worst, TOL, f"p = {p}, 4 frame pairs"


def check_l2l(rng):
    p, worst = 20, mpf(0)
    for trial in range(4):
        parent = random_frame(rng)
        if trial == 0:
            child = ([ci + parent[1] / 2 for ci in parent[0]], parent[1] / 2)
        else:
            child = random_frame(rng)
        L = random_coefficients(rng, p)
        L2 = l2l(p, parent, child, L)
        for _ in range(4):
            x = point_in_ball(rng, child[0], SQRT3 * child[1])
            before, size = evaluate(p, parent, L, x, True)
            after, _ = evaluate(p, child, L2, x, True)
            worst = max(worst, abs(after - before) / size)
    return worst, TOL, f"p = {p}, 4 frame pairs x 4 points, relative to the sum of |terms|"


def truncation_bound(p_in, p_out, fin, fout, sources, q):
    """Bound of §3.11 on |M2L(P2M) - P2L| per output coefficient (j, i), i >= 0."""
    (c, r), (c2, r2) = fin, fout
    sigma = r2 / r
    b = [(c2i - ci) / r for ci, c2i in zip(c, c2)]
    nb = mpmath.sqrt(sum(x * x for x in b))
    N = p_in + 1
    bound = [mpf(0)] * gh.tri(p_out + 1, 0)
    for y, qj in zip(sources, q):
        nu = mpmath.sqrt(sum(x * x for x in scaled(y, fin)))
        rhos = [nu + (nb - nu) * k / 64 for k in range(1, 64)]
        for j in range(p_out + 1):
            for i in range(j + 1):
                g = mpmath.sqrt(factorial(j - i) * factorial(j + i))
                best = mpf("inf")
                for rho in rhos:
                    x = nu / rho
                    tail = x**N * ((2 * N + 1) / (1 - x) + 2 * x / (1 - x) ** 2)
                    best = min(best, tail * g / (nb - rho) ** (j + 1))
                bound[gh.tri(j, i)] += abs(qj) * sigma ** (j + 1) * best
    return bound


def check_m2l(rng):
    p_in, p_out = 60, 8
    worst_ratio, worst_err = mpf(0), mpf(0)
    cases = []
    # V-list geometry: r' = r, c' - c = 2 r offset (CONVENTIONS §3.9, §3.11).
    for offset in ((2, 0, 0), (3, 3, 3), (-2, 1, 3)):
        fin = random_frame(rng)
        cases.append((fin, ([ci + 2 * fin[1] * o for ci, o in zip(fin[0], offset)], fin[1])))
    # Random frames: r'/r in [1/2, 2], |c' - c| / r in [sqrt3 (1 + r'/r) + 1/2, 10.4].
    for _ in range(2):
        fin = random_frame(rng)
        ratio = mpf("0.5") + mpf("1.5") * mpf(rng.random())
        lo = SQRT3 * (1 + ratio) + mpf("0.5")
        dist = (lo + (mpf("10.4") - lo) * mpf(rng.random())) * fin[1]
        e = unit_vector(rng)
        cases.append((fin, ([ci + dist * ei for ci, ei in zip(fin[0], e)], ratio * fin[1])))
    for fin, fout in cases:
        sources = [point_in_ball(rng, fin[0], SQRT3 * fin[1]) for _ in range(6)]
        q = charges(rng, len(sources))
        got = m2l(p_in, p_out, fin, fout, p2m(p_in, fin, sources, q))
        want = p2l(p_out, fout, sources, q)
        bound = truncation_bound(p_in, p_out, fin, fout, sources, q)
        # Where the bound falls below the working precision, rounding dominates: allow
        # TOL relative to the sum of term magnitudes of P2L, sum_j |q_j| |I_j^i(u'_j)|.
        size = [mpf(0)] * len(want)
        for y, qj in zip(sources, q):
            I = harmonics(p_out, scaled(y, fout), False)
            size = [s + abs(qj) * abs(v) for s, v in zip(size, I)]
        for g, w, bd, sz in zip(got, want, bound, size):
            worst_ratio = max(worst_ratio, abs(g - w) / (bd + TOL * sz))
        worst_err = max(worst_err, degree_error(p_out, got, want, local=True))
    detail = (
        f"p_in = {p_in}, p_out = {p_out}, {len(cases)} frame pairs; worst |error| / (bound + 1e-30 size), "
        f"worst weighted per-degree error {mpmath.nstr(worst_err, 3)}"
    )
    return worst_ratio, mpf(1), detail


def check_coaxial(rng):
    p, worst = 16, mpf(0)
    for _ in range(3):
        r, r2 = mpf("0.25") + 2 * mpf(rng.random()), mpf("0.25") + 2 * mpf(rng.random())
        c = [mpf(4 * rng.random() - 2) for _ in range(3)]
        for sign in (1, -1):
            d = sign * (mpf("0.1") + 3 * mpf(rng.random()))
            fin, fout = (c, r), ([c[0], c[1], c[2] + d], r2)
            M, L = random_coefficients(rng, p), random_coefficients(rng, p)
            worst = max(worst, degree_error(p, m2m_coaxial(p, r, r2, d, M), m2m(p, fin, fout, M), False))
            worst = max(worst, degree_error(p, l2l_coaxial(p, r, r2, d, L), l2l(p, fin, fout, L), True))
        for sign in (1, -1):
            d = sign * (SQRT3 * (1 + r2 / r) + mpf("0.5") + 5 * mpf(rng.random())) * r
            fin, fout = (c, r), ([c[0], c[1], c[2] + d], r2)
            M = random_coefficients(rng, p)
            worst = max(worst, degree_error(p, m2l_coaxial(p, p, r, r2, d, M), m2l(p, p, fin, fout, M), True))
    return worst, TOL, f"p = {p}, M2M, L2L and M2L, each for d > 0 and d < 0"


def check_real_storage(rng):
    """M2M, L2L, M2L from real storage through the order-sum identities vs complex forms."""
    p, worst = 12, mpf(0)
    for _ in range(3):
        fin, fout = random_frame(rng), random_frame(rng)
        v = random_coefficients(rng, p)
        got = complex_vector(p, m2m_real(p, fin, fout, real_vector(p, v)))
        worst = max(worst, degree_error(p, got, m2m(p, fin, fout, v), False))
        got = complex_vector(p, l2l_real(p, fin, fout, real_vector(p, v)))
        worst = max(worst, degree_error(p, got, l2l(p, fin, fout, v), True))
        ratio = mpf("0.5") + mpf("1.5") * mpf(rng.random())
        dist = (SQRT3 * (1 + ratio) + mpf("0.5") + 5 * mpf(rng.random())) * fin[1]
        e = unit_vector(rng)
        fout = ([ci + dist * ei for ci, ei in zip(fin[0], e)], ratio * fin[1])
        got = complex_vector(p, m2l_real(p, p, fin, fout, real_vector(p, v)))
        worst = max(worst, degree_error(p, got, m2l(p, p, fin, fout, v), True))
    return worst, TOL, f"p = {p}, 3 frame pairs per operator, against the complex forms"


# Rotation. Real storage of one degree: slots m = -n..n (§3.6), as mpmath matrices.


def real_block(v, n):
    col = mpmath.matrix(2 * n + 1, 1)
    for m in range(-n, n + 1):
        z = v[gh.tri(n, abs(m))]
        col[n + m] = z.real if m >= 0 else z.imag
    return col


def from_real(p, cols):
    out = zeros(p)
    for n in range(p + 1):
        col = cols[n]
        out[gh.tri(n, 0)] = mpc(col[n])
        for m in range(1, n + 1):
            out[gh.tri(n, m)] = mpc(col[n + m], col[n - m])
    return out


def matvec(Q, x):
    return [sum(Q[a][k] * x[k] for k in range(3)) for a in range(3)]


def transpose(Q):
    return [[Q[k][a] for k in range(3)] for a in range(3)]


def rotation_about(axis, angle):
    """Proper rotation by `angle` about the unit vector `axis` (Rodrigues)."""
    x, y, z = axis
    c, s = mpmath.cos(angle), mpmath.sin(angle)
    C = 1 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def fitted_blocks(rng, p, Q):
    """D^n(Q), n <= p, from R_n(Q x) = D^n R_n(x) at 2n + 1 random points (§3.8),
    validated at three further points."""
    points = [unit_vector(rng) for _ in range(2 * p + 4)]
    Rx = [harmonics(p, x, True) for x in points]
    RQx = [harmonics(p, matvec(Q, x), True) for x in points]
    blocks, worst = [], mpf(0)
    for n in range(p + 1):
        k = 2 * n + 1
        A, B = mpmath.matrix(k, k), mpmath.matrix(k, k)
        for col in range(k):
            a, bb = real_block(Rx[col], n), real_block(RQx[col], n)
            for row in range(k):
                A[row, col], B[row, col] = a[row], bb[row]
        D = B * mpmath.inverse(A)
        for x_r, xq_r in zip(Rx[-3:], RQx[-3:]):
            diff = D * real_block(x_r, n) - real_block(xq_r, n)
            worst = max(worst, mpmath.mnorm(diff, 1) / mpmath.mnorm(real_block(xq_r, n), 1))
        blocks.append(D)
    return blocks, worst


def K(n):
    return mpmath.diag([1 if m >= 0 else -1 for m in range(-n, n + 1)])


def S(n):
    return mpmath.diag([factorial(n - abs(m)) * factorial(n + abs(m)) for m in range(-n, n + 1)])


def rotate(p, v, blocks, local):
    """K D^n K v (multipoles) or K S D^n S^-1 K v (locals), degree by degree (§3.11)."""
    cols = []
    for n in range(p + 1):
        D = blocks[n]
        if local:
            D = S(n) * D * mpmath.inverse(S(n))
        cols.append(K(n) * D * K(n) * real_block(v, n))
    return from_real(p, cols)


def check_rotation(rng):
    p = 8
    worst_fit = worst_rule = worst_potential = mpf(0)
    for _ in range(2):
        Q = rotation_about(unit_vector(rng), 2 * mpmath.pi * mpf(rng.random()))
        blocks, fit = fitted_blocks(rng, p, Q)
        worst_fit = max(worst_fit, fit)
        frame = random_frame(rng)
        c, r = frame
        # Rotation about the frame centre: y -> c + Q (y - c).
        rot = lambda y: [ci + e for ci, e in zip(c, matvec(Q, [yi - ci for yi, ci in zip(y, c)]))]
        near = [point_in_ball(rng, c, SQRT3 * r) for _ in range(5)]
        far = [[ci + (2 + 6 * mpf(rng.random())) * r * e for ci, e in zip(c, unit_vector(rng))] for _ in range(5)]
        q = charges(rng, 5)
        M, L = p2m(p, frame, near, q), p2l(p, frame, far, q)
        M_rot, L_rot = p2m(p, frame, [rot(y) for y in near], q), p2l(p, frame, [rot(y) for y in far], q)
        worst_rule = max(worst_rule, degree_error(p, rotate(p, M, blocks, False), M_rot, False))
        worst_rule = max(worst_rule, degree_error(p, rotate(p, L, blocks, True), L_rot, True))
        # The rotated expansion at the rotated point equals the original at the point.
        for _ in range(3):
            x_far = [ci + (3 + 4 * mpf(rng.random())) * r * e for ci, e in zip(c, unit_vector(rng))]
            x_near = point_in_ball(rng, c, SQRT3 * r)
            for coeffs, x, regular in ((M, x_far, False), (L, x_near, True)):
                before, size = evaluate(p, frame, coeffs, x, regular)
                after, _ = evaluate(p, frame, rotate(p, coeffs, blocks, regular), rot(x), regular)
                worst_potential = max(worst_potential, abs(after - before) / size)
    worst = max(worst_fit, worst_rule, worst_potential)
    detail = (
        f"n <= {p}: fit of D^n {mpmath.nstr(worst_fit, 3)}, coefficient rule "
        f"{mpmath.nstr(worst_rule, 3)}, potential invariance {mpmath.nstr(worst_potential, 3)}"
    )
    return worst, TOL, detail


def aligning_rotation(t):
    """Q = R_y(-theta) R_z(-phi), so that Q t = |t| e_z (z-y-z Euler angles 0, -theta, -phi)."""
    nt = mpmath.sqrt(sum(x * x for x in t))
    theta, phi = mpmath.acos(t[2] / nt), mpmath.atan2(t[1], t[0])
    cz, sz = mpmath.cos(-phi), mpmath.sin(-phi)
    cy, sy = mpmath.cos(-theta), mpmath.sin(-theta)
    Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    return [[sum(Ry[a][k] * Rz[k][b] for k in range(3)) for b in range(3)] for a in range(3)], nt


def check_rotation_translation(rng):
    p, worst = 8, mpf(0)
    for op in ("m2m", "l2l", "m2l"):
        fin = random_frame(rng)
        ratio = mpf("0.5") + mpf("1.5") * mpf(rng.random())
        dist = (SQRT3 * (1 + ratio) + 1 + 4 * mpf(rng.random())) * fin[1] if op == "m2l" else fin[1]
        e = unit_vector(rng)
        fout = ([ci + dist * ei for ci, ei in zip(fin[0], e)], ratio * fin[1])
        t = [a - b for a, b in zip(fout[0], fin[0])]
        Q, d = aligning_rotation(t)
        aligned = matvec(Q, t)
        assert abs(aligned[0]) + abs(aligned[1]) < TOL and abs(aligned[2] - d) < TOL
        forward, _ = fitted_blocks(rng, p, Q)
        backward, _ = fitted_blocks(rng, p, transpose(Q))
        v = random_coefficients(rng, p)
        r, r2 = fin[1], fout[1]
        if op == "m2m":
            got = rotate(p, m2m_coaxial(p, r, r2, d, rotate(p, v, forward, False)), backward, False)
            want, local = m2m(p, fin, fout, v), False
        elif op == "l2l":
            got = rotate(p, l2l_coaxial(p, r, r2, d, rotate(p, v, forward, True)), backward, True)
            want, local = l2l(p, fin, fout, v), True
        else:
            got = rotate(p, m2l_coaxial(p, p, r, r2, d, rotate(p, v, forward, False)), backward, True)
            want, local = m2l(p, p, fin, fout, v), True
        worst = max(worst, degree_error(p, got, want, local))
    return worst, TOL, f"p = {p}, M2M, L2L, M2L via Q with Q (c' - c) = |c' - c| e_z"


def ladder_gradient(p, coeffs, v, regular):
    """sum_n sum_m C_n^m grad X_n^m(v) from the ladders of §3.4 and §3.11."""
    X = harmonics(p + 1, v, regular)
    g = [mpc(0)] * 3
    for n in range(p + 1):
        for m in range(-n, n + 1):
            C = at(coeffs, n, m)
            if regular:
                dm, dp, dz = at(X, n - 1, m - 1), -at(X, n - 1, m + 1), at(X, n - 1, m)
            else:
                dm, dp, dz = at(X, n + 1, m - 1), -at(X, n + 1, m + 1), -at(X, n + 1, m)
            # dm = (dx - i dy) X, dp = (dx + i dy) X.
            g[0] += C * (dm + dp) / 2
            g[1] += C * 1j * (dm - dp) / 2
            g[2] += C * dz
    return [z.real for z in g]


def check_gradients(rng):
    p, worst = 8, mpf(0)
    for regular in (True, False):
        for _ in range(3):
            frame = random_frame(rng)
            c, r = frame
            coeffs = random_coefficients(rng, p)
            if regular:
                x = point_in_ball(rng, c, SQRT3 * r)
            else:
                x = [ci + (2 + 6 * mpf(rng.random())) * r * e for ci, e in zip(c, unit_vector(rng))]
            g = [z / r**2 for z in ladder_gradient(p, coeffs, scaled(x, frame), regular)]
            h = mpf(10) ** -25
            with mp.workdps(HARMONIC_DPS):
                fd = []
                for axis in range(3):
                    xp = [xi + (h if k == axis else 0) for k, xi in enumerate(x)]
                    xm = [xi - (h if k == axis else 0) for k, xi in enumerate(x)]
                    fd.append((evaluate(p, frame, coeffs, xp, regular)[0] - evaluate(p, frame, coeffs, xm, regular)[0]) / (2 * h))
            scale = sum(abs(z) for z in fd)
            worst = max(worst, max(abs(a - b) for a, b in zip(g, fd)) / scale)
    return worst, TOL, f"p = {p}, L2P and M2P, 3 frames each, relative to |grad|_1"


def main():
    global SQRT3
    if mpmath.__version__ != gh.MPMATH_VERSION:
        sys.exit(f"needs mpmath {gh.MPMATH_VERSION}, found {mpmath.__version__}")
    mp.dps = DPS
    SQRT3 = mpmath.sqrt(3)
    rng = random.Random(SEED)
    checks = [
        ("M2M after P2M = P2M at parent", check_m2m),
        ("L2L preserves the local expansion", check_l2l),
        ("M2L after P2M = P2L, within bound", check_m2l),
        ("coaxial = general, shift along z", check_coaxial),
        ("rotation rule for coefficients", check_rotation),
        ("rotate, coaxial, rotate back = general", check_rotation_translation),
        ("L2P/M2P gradients = central diff.", check_gradients),
        ("real-storage order sums = complex forms", check_real_storage),
    ]
    ok = True
    for name, check in checks:
        err, tol, detail = check(rng)
        passed = err <= tol
        ok &= passed
        print(f"{'ok' if passed else 'FAIL':4}  {name:40} {mpmath.nstr(err, 3):>10}  (tol {mpmath.nstr(tol, 1)})  {detail}")
        sys.stdout.flush()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
