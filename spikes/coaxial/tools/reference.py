#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Arbitrary-precision reference entries of the coaxial translation coefficients
(Helmholtz CH0.2, spike S2; spikes/coaxial/SPIKE_REPORT.md).

Basis: real orthonormal harmonics Y_n^m (CONVENTIONS §3.3's Legendre functions without the
Condon-Shortley phase; cos in slot +m, sin in slot -m, as §3.6; normalised so that
sum_m Y_n^m(x) Y_n^m(y) = (2n + 1)/(4 pi) P_n(x.y), proposal §3.1) and
F_n^m(r) = f_n(k r) Y_n^m(r^), f = j (regular) or h = h^(1) (singular). The coaxial
translation by t = t z^:

    F_n^m(r + t) = sum_n' (E|F)^m_{n'n}(t) G_n'^m(r),

the same coefficient for the slots +m and -m. (R|R): f = g = j, any r; (S|R): f = h,
g = j, |r| < t; (S|S): f = g = h, |r| > t, and (S|S) = (R|R).

Method: the Gaunt sum

    (E|F)^m_{n'n} = (-1)^m sqrt((2n + 1)(2n' + 1)) sum_l i^(n' - n + l) (2l + 1)
                    (n n' l; 0 0 0) (n n' l; m -m 0) f_l(k t),

over l = |n - n'| .. n + n' with n + n' + l even, f = j for (R|R) and h for (S|R); the 3j
symbols by their three-term recursion in l downward from the closed form at l = n + n';
f_l from the closed forms of f_0, f_1 by h upward and j by Miller's downward recursion,
normalised by j_0 or j_1. Every value is computed at two precisions 50 digits apart and
must agree to 1e-25 relative. Before anything is written:

  - the formula against the definition: the addition theorems evaluated pointwise at
    n <= 6, every slot, for (R|R), (S|R), (S|S) = (R|R) and the reflection
    (E|F)(-t) = (-1)^(n + n') (E|F)(t), for real, complex and imaginary k (1e-40);
  - the radial functions against mpmath's besselj and hankel1 at sampled orders, and
    the Legendre recursion of the harmonics against mpmath's legenp.

Stored values carry the spike's normalisation (SPIKE_REPORT.md §1): (R|R) times
e^(-|Im z|), (S|R) times e^(Im z) g^(n + n'), with z = k t and g the FMM3D scale
min(|k| a, 1); z and g are the f64 values the spike uses.

Writes spikes/coaxial/fixtures/reference.json; `--check` regenerates in memory and fails
unless the committed file matches byte for byte.

Run from the repository root:  uv run spikes/coaxial/tools/reference.py [--check]
"""
import json
import math
import random
import sys
from math import factorial
from pathlib import Path

import mpmath
from mpmath import mp, mpc, mpf

if mpmath.__version__ != "1.4.1":
    sys.exit(f"needs mpmath 1.4.1, found {mpmath.__version__}")

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "reference.json"
SEED = 20261027
AGREE = mpf(10) ** -25
# k a (a the circumradius of a box), and the kinds of k: real, complex k = kappa (1 + i eta),
# imaginary k = i kappa; k a means kappa a.
KA = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
KINDS = [("real", 1.0, 0.0), ("complex0.1", 1.0, 0.1), ("complex1", 1.0, 1.0), ("imaginary", 0.0, 1.0)]
EPS = [1e-10, 1e-3]
# V-list distances in box widths sampled by the reference (the spike sweeps all 15).
DISTANCES = [("d2", 4), ("d27", 27)]
LINE = 11
RANDOM = 12


# ------------------------------------------------------------------ configurations


def degree(ka, eps):
    """p of proposal §3.5 (Gumerov-Duraiswami 2009 eq. 18, delta = 2): p = ceil(P) - 1."""
    p_hi = ka + (1.5 * math.log(1.0 / eps)) ** (2.0 / 3.0) * ka ** (1.0 / 3.0)
    p_lo = 1.0 - math.log(eps * (1.0 - 0.5) ** 1.5) / math.log(2.0)
    return math.ceil((p_lo**4 + p_hi**4) ** 0.25) - 1


def configurations():
    """(name, op, z = (re, im), g, p_src, p_tgt), with the f64 arithmetic the spike's
    sweep repeats (src/config.rs)."""
    out = []
    for kind, re, im in KINDS:
        for ka in KA:
            for eps in EPS:
                if eps != EPS[0] and kind != "real":
                    continue
                p_parent, p_child = degree(ka, eps), degree(ka / 2.0, eps)
                # M2M: (S|S) = (R|R) at the child shift |t| = a/2, p_child -> p_parent.
                kt = ka / 2.0
                out.append((f"{kind}-ka{ka}-eps{eps}-m2m", "RR", (re * kt, im * kt), 1.0, p_child, p_parent))
                # M2L: (S|R) at |t| = d w = d 2a/sqrt(3), p -> p, both ends scaled by g.
                modulus = ka * math.hypot(re, im)
                g = min(modulus, 1.0)
                for dname, d2 in DISTANCES:
                    kt = ka * 2.0 * math.sqrt(d2) / math.sqrt(3.0)
                    out.append(
                        (f"{kind}-ka{ka}-eps{eps}-m2l-{dname}", "SR", (re * kt, im * kt), g, p_parent, p_parent)
                    )
    return out


# ---------------------------------------------------------------- radial functions


def radial(op, z, L):
    """f_l(z), l = 0..L, f = j for "RR" and h for "SR", at the current precision."""
    if op == "SR":
        e = mpmath.expj(z)
        h = [-1j * e / z, -e * (z + 1j) / (z * z)]
        for l in range(1, L):
            h.append((2 * l + 1) / z * h[l] - h[l - 1])
        return h[: L + 1]
    j0 = mpmath.sin(z) / z
    j1 = mpmath.sin(z) / (z * z) - mpmath.cos(z) / z
    start = int(max(L, abs(z)) * 1.2) + 60
    u = [mpc(0)] * (start + 2)
    u[start] = mpc(1)
    for l in range(start, 0, -1):
        u[l - 1] = (2 * l + 1) / z * u[l] - u[l + 1]
    c = j0 / u[0] if abs(j0) >= abs(j1) else j1 / u[1]
    return [c * v for v in u[: L + 1]]


def check_radial():
    """The recurrences against mpmath's Bessel functions at sampled orders."""
    worst = mpf(0)
    with mp.workdps(40):
        cases = [(z, 200, (0, 1, 7, 60, 200)) for z in (mpf("0.005"), mpf("3.7"), mpf("150"), mpc("20", "2"), mpc("5", "5"), mpc(0, "40"))]
        # The largest arguments and orders of the fixture.
        cases += [(z, 742, (0, 371, 742)) for z in (mpf("1800"), mpc("1800", "180"), mpc(0, "1800"))]
        for z, top, orders in cases:
            for op, fn in (("RR", mpmath.besselj), ("SR", mpmath.hankel1)):
                values = radial(op, z, top)
                for l in orders:
                    exact = mpmath.sqrt(mpmath.pi / (2 * z)) * fn(l + mpf(1) / 2, z)
                    worst = max(worst, abs(values[l] - exact) / abs(exact))
    if worst > mpf(10) ** -30:
        sys.exit(f"radial check failed: worst {mpmath.nstr(worst, 3)}")
    print(f"radial functions checked: worst {mpmath.nstr(worst, 3)}", file=sys.stderr)


# ------------------------------------------------------------------- the 3j symbols


def threej(n1, n2, m):
    """(n1 n2 l; m -m 0) for l = |n1 - n2| .. n1 + n2, as a list indexed by l - |n1 - n2|,
    by the recursion of Schulten and Gordon downward from the closed form at l = n1 + n2."""
    J, lo = n1 + n2, abs(n1 - n2)
    if m > min(n1, n2):
        return [mpf(0)] * (J - lo + 1)
    num = factorial(2 * n1) * factorial(2 * n2) * factorial(J) ** 2
    den = factorial(2 * J + 1) * factorial(n1 + m) * factorial(n1 - m) * factorial(n2 + m) * factorial(n2 - m)
    out = {J: (-1) ** (n1 - n2) * mpmath.sqrt(mpf(num) / den)}

    def e(j):
        return j * mpmath.sqrt(mpf((j * j - (n1 - n2) ** 2) * ((J + 1) ** 2 - j * j)))

    # j e(j+1) f(j+1) + f_coef(j) f(j) + (j+1) e(j) f(j-1) = 0, with m1 = m, m2 = -m, m3 = 0.
    for j in range(J, lo, -1):
        f_coef = -2 * m * (2 * j + 1) * j * (j + 1)
        out[j - 1] = -(f_coef * out[j] + j * e(j + 1) * out.get(j + 1, 0)) / ((j + 1) * e(j))
    return [out[l] for l in range(lo, J + 1)]


def gaunt(n1, n, m, f):
    """(E|F)^m_{n1 n} from f_l = f[l]."""
    t0, tm = threej(n, n1, 0), threej(n, n1, m)
    lo = abs(n - n1)
    total = mpc(0)
    for l in range(lo, n + n1 + 1, 2):
        phase = (n1 - n + l) // 2  # i^(n1 - n + l) with an even exponent
        term = (2 * l + 1) * t0[l - lo] * tm[l - lo] * f[l]
        total += -term if phase % 2 else term
    sign = -1 if m % 2 else 1
    return sign * mpmath.sqrt(mpf((2 * n + 1) * (2 * n1 + 1))) * total


# -------------------------------------------------- the formula against the definition


def legendre_all(n_max, u):
    """pbar[n][m] = sqrt((2n + 1)/(4 pi) (n - m)!/(n + m)!) P_n^m(u), no Condon-Shortley
    phase, 0 <= m <= n <= n_max, by the standard recursions."""
    s = mpmath.sqrt(1 - u * u)
    pbar = [[mpf(0)] * (n_max + 1) for _ in range(n_max + 1)]
    pbar[0][0] = 1 / mpmath.sqrt(4 * mpmath.pi)
    for m in range(1, n_max + 1):
        pbar[m][m] = pbar[m - 1][m - 1] * s * mpmath.sqrt(mpf(2 * m + 1) / (2 * m))
    for m in range(0, n_max):
        pbar[m + 1][m] = u * mpmath.sqrt(2 * m + 3) * pbar[m][m]
        for n in range(m + 2, n_max + 1):
            a1 = mpmath.sqrt(mpf((n - m) * (n + m)) / ((2 * n - 1) * (2 * n + 1)))
            a0 = mpmath.sqrt(mpf((n - 1 - m) * (n - 1 + m)) / ((2 * n - 3) * (2 * n - 1)))
            pbar[n][m] = (u * pbar[n - 1][m] - a0 * pbar[n - 2][m]) / a1
    return pbar


def real_harmonics(n_max, x):
    """Y[n][slot + n] for 0 <= n <= n_max at the point x (module docstring)."""
    r = mpmath.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    phi = mpmath.atan2(x[1], x[0])
    pbar = legendre_all(n_max, x[2] / r)
    out = []
    for n in range(n_max + 1):
        row = []
        for slot in range(-n, n + 1):
            m = abs(slot)
            if slot == 0:
                row.append(pbar[n][0])
            else:
                trig = mpmath.cos(m * phi) if slot > 0 else mpmath.sin(m * phi)
                row.append(mpmath.sqrt(2) * pbar[n][m] * trig)
        out.append(row)
    return out


def check_legendre():
    """legendre_all against mpmath's legenp (with its Condon-Shortley phase removed)."""
    worst = mpf(0)
    with mp.workdps(40):
        for u in (mpf("0.3"), mpf("-0.77")):
            pbar = legendre_all(30, u)
            for n, m in ((0, 0), (5, 3), (17, 0), (17, 17), (30, 11)):
                c = mpmath.sqrt((2 * n + 1) / (4 * mpmath.pi) * mpf(factorial(n - m)) / factorial(n + m))
                exact = c * (-1) ** m * mpmath.legenp(n, m, u)
                worst = max(worst, abs(pbar[n][m] - exact) / abs(exact))
    if worst > mpf(10) ** -30:
        sys.exit(f"Legendre check failed: worst {mpmath.nstr(worst, 3)}")
    print(f"Legendre functions checked: worst {mpmath.nstr(worst, 3)}", file=sys.stderr)


def waves(f, k, n_max, x):
    """F_n^slot(x) = f_n(k |x|) Y_n^slot(x) for every n <= n_max, f = "j" or "h", with the
    radial recurrences checked in check_radial."""
    r = mpmath.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    radii = radial("RR" if f == "j" else "SR", k * r, n_max)
    harmonics = real_harmonics(n_max, x)
    return [[radii[n] * y for y in harmonics[n]] for n in range(n_max + 1)]


def check_definition():
    check_legendre()
    rng = random.Random(SEED)
    worst = mpf(0)
    terms = 70
    with mp.workdps(60):
        for k in [mpf("1.3"), mpc("1.3", "0.4"), mpc(0, "1.3")]:
            for sign in (1, -1):
                t = sign * mpf("0.9")
                # name, f of F, f of G, start f, |r| relative to |t|
                for name, f_out, f_in, start, radius in (
                    ("RR", "j", "j", "RR", mpf("0.7")),
                    ("SR", "h", "j", "SR", mpf("0.12")),
                    ("SS", "h", "h", "RR", mpf("8")),
                ):
                    f = radial(start, k * abs(t), 2 * terms + 12)
                    coef = {}
                    for _ in range(3):
                        direction = [mpf(rng.uniform(-1, 1)) for _ in range(3)]
                        scale = radius * abs(t) / mpmath.sqrt(sum(v * v for v in direction))
                        x = [v * scale for v in direction]
                        lhs = waves(f_out, k, 6, [x[0], x[1], x[2] + t])
                        rhs_basis = waves(f_in, k, 6 + terms, x)
                        for n in range(0, 7):
                            for slot in range(-n, n + 1):
                                m = abs(slot)
                                rhs = mpc(0)
                                for n1 in range(m, m + terms):
                                    if (n1, n, m) not in coef:
                                        coef[(n1, n, m)] = gaunt(n1, n, m, f)
                                    c = coef[(n1, n, m)]
                                    if sign < 0 and (n + n1) % 2:
                                        c = -c
                                    rhs += c * rhs_basis[n1][slot + n1]
                                value = lhs[n][slot + n]
                                worst = max(worst, abs(value - rhs) / abs(value))
    if worst > mpf(10) ** -40:
        sys.exit(f"definition check failed: worst {mpmath.nstr(worst, 3)}")
    print(f"addition theorems checked at n <= 6: worst {mpmath.nstr(worst, 3)}", file=sys.stderr)


# ------------------------------------------------------------------------- sampling


def line(lo, hi):
    """At most LINE indices in lo..hi, both ends included."""
    if hi - lo + 1 <= LINE:
        return list(range(lo, hi + 1))
    step = (hi - lo) / (LINE - 1)
    return sorted({lo, hi} | {round(lo + i * step) for i in range(LINE)})


def samples(p_src, p_tgt, rng):
    """(m, n', n): orders 0, 1, the middle and the largest; rows n' = m and p_tgt; columns
    n = m and p_src; the diagonal; random entries."""
    top = min(p_src, p_tgt)
    picks = set()
    for m in sorted({0, min(1, top), top // 2, top}):
        for t in line(m, p_src):
            picks.add((m, m, t))
            picks.add((m, p_tgt, t))
        for t in line(m, p_tgt):
            picks.add((m, t, m))
            picks.add((m, t, p_src))
        for t in line(m, top):
            picks.add((m, t, t))
        for _ in range(RANDOM):
            picks.add((m, rng.randint(m, p_tgt), rng.randint(m, p_src)))
    return sorted(picks)


def normalised(op, z, g, n1, n, value):
    """The spike's normalisation of a coefficient (module docstring)."""
    if op == "RR":
        return value * mpmath.exp(-abs(z.imag))
    return value * mpmath.exp(z.imag) * mpf(g) ** (n + n1)


def entries_checked(op, z, g, p_src, p_tgt, picks):
    """The stored values of the picks (m, n', n) at two precisions; exits on disagreement."""
    L = p_src + p_tgt
    dps = 40 + L // 2
    runs = []
    for extra in (0, 50):
        with mp.workdps(dps + extra):
            zz = mpc(mpf(z[0]), mpf(z[1]))
            f = radial(op, zz, L)
            runs.append([normalised(op, zz, g, n1, n, gaunt(n1, n, m, f)) for m, n1, n in picks])
    with mp.workdps(dps + 50):
        for (m, n1, n), a, b in zip(picks, runs[0], runs[1]):
            if abs(a - b) > AGREE * abs(b):
                sys.exit(f"precision check failed: {op} z={z} m={m} n'={n1} n={n}")
    return runs[1]


def radial_samples(op, z, g, L):
    """The spike's normalised start values at sampled orders: f_l times e^(-|Im z|) (RR)
    or e^(Im z) g^l (SR)."""
    with mp.workdps(60 + L):
        zz = mpc(mpf(z[0]), mpf(z[1]))
        f = radial(op, zz, L)
        out = []
        for l in line(0, L):
            v = f[l] * (mpmath.exp(-abs(zz.imag)) if op == "RR" else mpmath.exp(zz.imag) * mpf(g) ** l)
            out.append((l, v))
        return out


def f64_pair(v):
    return [repr(float(v.real)), repr(float(v.imag))]


def generate(only=None):
    check_radial()
    if only is None:
        check_definition()
    rng = random.Random(SEED)
    sets = []
    for name, op, z, g, p_src, p_tgt in configurations():
        if only is not None and only not in name:
            continue
        picks = samples(p_src, p_tgt, rng)
        values = entries_checked(op, z, g, p_src, p_tgt, picks)
        radial_values = radial_samples(op, z, g, p_src + p_tgt)
        sets.append(
            {
                "name": name,
                "op": op,
                "z": [repr(z[0]), repr(z[1])],
                "g": repr(g),
                "p_src": p_src,
                "p_tgt": p_tgt,
                "m": [s[0] for s in picks],
                "row": [s[1] for s in picks],
                "col": [s[2] for s in picks],
                "value": [f64_pair(v) for v in values],
                "radial_l": [l for l, _ in radial_values],
                "radial": [f64_pair(v) for _, v in radial_values],
            }
        )
        print(f"{name}: p {p_src} -> {p_tgt}, {len(picks)} entries", file=sys.stderr)
    doc = {
        "about": "Coaxial translation coefficients (E|F)^m_{n'n} in the real orthonormal basis, "
        "sampled as (m, row n', col n), with the spike's normalisation (tools/reference.py); f64 "
        "pairs (re, im) as shortest round-trip strings, rounded from arbitrary precision. "
        "Generated by spikes/coaxial/tools/reference.py (mpmath 1.4.1).",
        "sets": sets,
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).replace("},{", "},\n{") + "\n"


def main():
    if "--only" in sys.argv[1:]:
        # Development aid: one configuration, timed, nothing written.
        import time

        start = time.time()
        text = generate(only=sys.argv[sys.argv.index("--only") + 1])
        print(f"{len(text)} bytes in {time.time() - start:.1f} s", file=sys.stderr)
        return
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
