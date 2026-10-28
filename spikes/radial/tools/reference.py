#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Arbitrary-precision reference values of the radial functions of the Helmholtz family
(Helmholtz CH0.3, spike S3; spikes/radial/SPIKE_REPORT.md).

The four families and their derivatives, with the spike's scaling (SPIKE_REPORT.md §1):

    j^_n  = e^(-|Im z|) j_n(z) / s^n          h^_n = e^(Im z) h_n(z) s^n,   h = h^(1)
    i^_n  = e^(-x) i_n(x) / s^n               k^_n = e^x k_n(x) s^n

for complex z with Im z >= 0 and real x > 0, and the same factors on the derivatives
(j^'_n = e^(-|Im z|) j_n'(z) / s^n, ...). k_n is (2/pi) times DLMF's k_n, so
k_0(x) = e^(-x)/x and h_n(i x) = -i^(-n) k_n(x); j_n(i x) = i^n i_n(x).

Method: mpmath's Bessel functions at half-integer order for the regular functions, and
the finite sums of half-integer order for the singular ones, so nothing is shared with
the spike's recursions:

    j_n(z) = sqrt(pi/(2z)) J_{n+1/2}(z),     i_n(x) = sqrt(pi/(2x)) I_{n+1/2}(x)
    h_n(z) = e^(iz) sum_{k=0}^n i^(k-n-1) a_k / z^(k+1)          (DLMF 10.49.6)
    k_n(x) = e^(-x) sum_{k=0}^n a_k / x^(k+1)                    (DLMF 10.49.12, times 2/pi)
    a_k = (n + k)! / (2^k k! (n - k)!)

The sums take digits until their largest term is covered. (mpmath's besselk does not
converge at large argument and moderate order, e.g. order 370.5 at 1800, and
hankel1 = J + iY cancels by about 0.87 Im z digits.)

and the derivatives from the neighbouring orders (f_n' = f_{n-1} - (n+1)/z f_n for j, h,
i; k_n' = -k_{n-1} - (n+1)/x k_n; at n = 0, j_0' = -j_1, h_0' = -h_1, i_0' = i_1,
k_0' = -k_1). Every value at two precisions 40 digits apart, required to agree to 1e-30
relative. Before anything is written, the definitions are checked: the closed forms at
n <= 2, the relations to the complex path at z = i x, and the Wronskians (1e-40).

For j and j' each row also carries the oscillation envelope the tests measure against:
the same scaling times max(|j_n|, |h_n|) where n < |z| (j_n has real zeros there), and
|j_n| elsewhere; likewise for j' with h'.

Writes spikes/radial/fixtures/reference.json; `--check` regenerates in memory and fails
unless the committed file matches byte for byte.

Run from the repository root:  uv run spikes/radial/tools/reference.py [--check]
"""
import json
import math
import random
import sys
from pathlib import Path

import mpmath
from mpmath import mp, mpc, mpf

if mpmath.__version__ != "1.4.1":
    sys.exit(f"needs mpmath 1.4.1, found {mpmath.__version__}")

OUT = Path(__file__).resolve().parent.parent / "fixtures" / "reference.json"
SEED = 20261027
DPS = (40, 80)
AGREE = mpf(10) ** -30
# Values the f64 range cannot hold are stored as 0 (underflow); the tests then require
# the spike's value below 1e-280. Nothing on the ranges may exceed OVER.
UNDER = 1e-300
OVER = 1e300

# The kinds of k, by arg k: real, complex k = kappa (1 + i eta) for eta = 0.1, 1 and 10,
# and imaginary (the complex path at z = i x, which Yukawa's real path is checked against).
KINDS = [
    ("real", 0.0),
    ("complex0.1", math.atan(0.1)),
    ("complex1", math.pi / 4),
    ("complex10", math.atan(10.0)),
    ("imaginary", math.pi / 2),
]
EPS = 1e-10
# The circumradius of a leaf of width c_leaf wavelengths: k a = 2 pi c sqrt(3)/2.
C_LEAF = [0.25, 1.0, 2.0]
LOW_KA = [1e-6, 1e-3, 0.1]
# |k| a of the levels whose translations the sweep covers (k a = 650 is level 2 at
# kD = 3000, proposal §3.5's last row).
LEVEL_KA = [0.01, 1.0, 10.0, 40.0, 100.0, 300.0, 650.0]
# Yukawa: kappa a of a leaf has no wavelength bound (proposal §3.7).
YUKAWA_KA = [1e-6, 1e-3, 0.1, 1.0, 5.0, 30.0, 300.0, 650.0]
RANDOM = 160
# The largest |z| of the random points: the projection's h_n(k R), R up to t + a = 7a, at
# k a = 650.
Z_MAX = 7 * 650.0
SAMPLES = 18


# ------------------------------------------------------------------ configurations


def degree(ka, eps):
    """p of proposal §3.5 (Gumerov-Duraiswami 2009 eq. 18, delta = 2): p = ceil(P) - 1.
    The same f64 arithmetic as src/ranges.rs."""
    p_hi = ka + (1.5 * math.log(1.0 / eps)) ** (2.0 / 3.0) * ka ** (1.0 / 3.0)
    p_lo = 1.0 - math.log(eps * (1.0 - 0.5) ** 1.5) / math.log(2.0)
    return math.ceil((p_lo**4 + p_hi**4) ** 0.25) - 1


def orders(top, z_abs, rng):
    """The sampled orders of a set: all of them up to 40, else the ends, the transition
    near |z| and spread samples."""
    if top <= 40:
        return list(range(top + 1))
    picked = set(range(0, 6)) | {top - 1, top}
    for c in (z_abs - 2, z_abs, z_abs + 2):
        if 0 <= c <= top:
            picked.add(int(c))
    for i in range(SAMPLES):
        picked.add(round(i * top / (SAMPLES - 1)))
    for _ in range(4):
        picked.add(rng.randrange(top + 1))
    return sorted(picked)


def polar(r, theta):
    if theta == math.pi / 2:
        return (0.0, r)
    return (r * math.cos(theta), r * math.sin(theta))


def configurations(rng):
    """(name, family, z = (re, im), s, orders): family "j" (with j'), "h" (with h'), "i"
    (with i') or "k" (with k'); for "i" and "k" z is real. s = min(|k| a, 1), a the
    circumradius of the box the scale belongs to (SPIKE_REPORT.md §1)."""
    out = []

    def add(name, fam, z, s, top):
        out.append((name, fam, z, s, orders(top, math.hypot(*z), rng)))

    # P2M and L2P: j at k rho, 0 < rho <= a_leaf.
    for kind, theta in KINDS:
        leaf = [(f"c{c}", 2 * math.pi * c * math.sqrt(3) / 2) for c in C_LEAF]
        leaf += [(f"ka{ka}", ka) for ka in LOW_KA]
        for tag, re_ka in leaf:
            # The leaf rule bounds Re k a; |k| a follows from the kind (imaginary: kappa a).
            ka = re_ka if kind == "imaginary" else re_ka / math.cos(theta)
            p = degree(min(ka, 11.0), EPS)
            for frac in (1e-6, 1e-3, 0.5, 1.0):
                add(f"p2m-{kind}-{tag}-rho{frac}", "j", polar(ka * frac, theta), min(ka, 1.0), p + 1)
    # M2P and P2L: h at k rho, sqrt(3) a_b <= rho <= 8 a_b.
    for kind, theta in KINDS:
        for ka in LOW_KA + [1.0, 2.7, 5.4, 10.9]:
            p = degree(ka, EPS)
            for frac in (math.sqrt(3), 8.0):
                add(f"m2p-{kind}-ka{ka}-rho{frac:.3f}", "h", polar(ka * frac, theta), min(ka, 1.0), p + 1)
    # Translations: the start values j_l(k a/2) of M2M and L2L, h_l(k t) of the M2L at
    # t = 2w and 6a (= sqrt(27) w), and the projection's j and h at rho = a and 4a/sqrt(3),
    # and its translated h_n(k R) at the far end R = t + a = 7a of the farthest M2L.
    for kind, theta in KINDS:
        for ka in LEVEL_KA:
            p, pc = degree(ka, EPS), degree(ka / 2, EPS)
            s = min(ka, 1.0)
            add(f"m2m-{kind}-ka{ka}", "j", polar(ka / 2, theta), s, p + pc)
            for tname, t in (("d2", 4 / math.sqrt(3)), ("d27", 6.0)):
                add(f"m2l-{kind}-ka{ka}-{tname}", "h", polar(ka * t, theta), s, 2 * p)
            if ka >= 40:
                add(f"proj-{kind}-ka{ka}-j", "j", polar(ka, theta), s, p + 1)
                add(f"proj-{kind}-ka{ka}-h", "h", polar(ka * 4 / math.sqrt(3), theta), s, p + 1)
                add(f"proj-{kind}-ka{ka}-h7", "h", polar(ka * 7, theta), s, p + 1)
    # Yukawa, the real path: i at kappa rho (P2M, L2P, M2M), k at kappa rho and kappa t, to
    # Helmholtz's orders at the same kappa a (larger than Yukawa's own, SPIKE_REPORT.md §6).
    for ka in YUKAWA_KA:
        p = degree(min(ka, 650.0), EPS)
        s = min(ka, 1.0)
        for frac in (1e-3, 0.5, 1.0):
            add(f"yp2m-ka{ka}-rho{frac}", "i", (ka * frac, 0.0), s, p + 1)
        add(f"ym2p-ka{ka}", "k", (ka * math.sqrt(3), 0.0), s, p + 1)
        add(f"ym2m-ka{ka}", "i", (ka / 2, 0.0), s, p + degree(ka / 2, EPS))
        for tname, t in (("d2", 4 / math.sqrt(3)), ("d27", 6.0)):
            add(f"ym2l-ka{ka}-{tname}", "k", (ka * t, 0.0), s, 2 * p)
    # Random points over the whole range: |z| log-uniform in [1e-6, Z_MAX], arg in [0, pi/2],
    # the scale of a box with |k| a between |z|/8 and |z|/sqrt(3) (h) or |z| and 4 |z| (j).
    for i in range(RANDOM):
        fam = "jhik"[i % 4]
        r = 10 ** rng.uniform(-6, math.log10(Z_MAX))
        theta = rng.uniform(0, math.pi / 2) if fam in "jh" else 0.0
        ka = r * (rng.uniform(1, 4) if fam in "ji" else rng.uniform(1 / 8, 1 / math.sqrt(3)))
        top = min(1460, 2 * degree(min(ka, 650.0), EPS))
        n = rng.randrange(top + 1)
        z = polar(r, theta)
        out.append((f"random{i}-{fam}", fam, z, min(ka, 1.0), [n]))
    return out


# ---------------------------------------------------------------- radial functions


def half(n):
    return n + mpf(1) / 2


def jn(n, z):
    return mpmath.sqrt(mpmath.pi / (2 * z)) * mpmath.besselj(half(n), z)


# The most digits a sum of hankel_sum lost to cancellation, reported at the end.
CANCELLED = [0]


def i_pow(k):
    """i^k exactly. (Python's 1j ** k leaves exact repeated multiplication above |k| = 100
    and errs by 1e-16, which the cancelling terms of h_n amplify to O(1).)"""
    return (mpc(1), mpc(0, 1), mpc(-1), mpc(0, -1))[k % 4]


def hankel_sum(n, z, imaginary):
    """sum_{k=0}^n c^k a_k(n + 1/2) / z^(k+1) with c = i if `imaginary`, else 1, a_k = (n + k)! / (2^k k! (n - k)!)
    (DLMF 10.49.1; the sums of 10.49.6 and 10.49.12), with digits added until the largest term is covered: the terms of h_n
    alternate in phase and cancel by up to hundreds of digits near |z| ~ n."""
    extra = 20
    while True:
        with mp.extradps(extra):
            a = mpf(1)
            term_scale = 1 / z
            terms = []
            for k in range(n + 1):
                terms.append((i_pow(k) if imaginary else 1) * a * term_scale)
                a = a * (n + k + 1) * (n - k) / (2 * (k + 1))
                term_scale = term_scale / z
            total = mpmath.fsum(terms)
            largest = max(abs(t) for t in terms)
            lost = int(mpmath.log10(largest / abs(total))) if total != 0 else 10**6
            if lost + 15 <= extra:
                CANCELLED[0] = max(CANCELLED[0], lost)
                return +total
            extra = lost + 30


def hn(n, z):
    """h_n(z) = e^(iz) i^(-n-1) sum_k i^k a_k / z^(k+1) (DLMF 10.49.6)."""
    return mpmath.expj(z) * i_pow(-n - 1) * hankel_sum(n, z, True)


def i_n(n, x):
    return mpmath.sqrt(mpmath.pi / (2 * x)) * mpmath.besseli(half(n), x)


def kn(n, x):
    """k_n(x) = e^(-x) sum_k a_k / x^(k+1) (DLMF 10.49.12 times 2/pi), all terms positive."""
    return mpmath.exp(-x) * hankel_sum(n, x, False)


def derivative(fam, f, n, z):
    if n == 0:
        if fam == "i":
            return f(1, z)
        return -f(1, z)
    if fam == "k":
        return -f(n - 1, z) - (n + 1) / z * f(n, z)
    return f(n - 1, z) - (n + 1) / z * f(n, z)


CACHE = {}


def cached(f):
    """f(n, z) memoised per precision (the derivatives reuse the neighbouring orders)."""

    def g(n, z):
        key = (f.__name__, n, str(z), mp.dps)
        if key not in CACHE:
            CACHE[key] = f(n, z)
        return CACHE[key]

    g.__name__ = f.__name__
    return g


jn, hn, i_n, kn = cached(jn), cached(hn), cached(i_n), cached(kn)
FUNCTIONS = {"j": jn, "h": hn, "i": i_n, "k": kn}


def row(fam, z, s, n):
    """(value, derivative, envelope of value, envelope of derivative) with the scaling, at
    the current precision; envelopes only for j."""
    f = FUNCTIONS[fam]
    if fam in "jh":
        z = mpc(*z)
    else:
        z = mpf(z[0])
    s = mpf(s)
    if fam == "j":
        factor = mpmath.exp(-abs(mpmath.im(z))) / s**n
    elif fam == "h":
        factor = mpmath.exp(mpmath.im(z)) * s**n
    elif fam == "i":
        factor = mpmath.exp(-z) / s**n
    else:
        factor = mpmath.exp(z) * s**n
    v, d = f(n, z), derivative(fam, f, n, z)
    env_v = env_d = None
    if fam == "j":
        env_v, env_d = abs(v), abs(d)
        if n < abs(z):
            env_v = max(env_v, abs(hn(n, z)))
            env_d = max(env_d, abs(derivative("h", hn, n, z)))
        env_v, env_d = env_v * factor, env_d * factor
    return v * factor, d * factor, env_v, env_d


def agree(a, b):
    if a == 0 and b == 0:
        return True
    return abs(a - b) <= AGREE * max(abs(a), abs(b))


def reference(fam, z, s, n):
    """row() at the two precisions, required to agree."""
    results = []
    for dps in DPS:
        with mp.workdps(dps):
            results.append(row(fam, z, s, n))
    lo, hi = results
    for a, b in zip(lo, hi):
        if a is not None and not agree(a, b):
            sys.exit(f"precision disagreement: {fam} z={z} s={s} n={n}: {a} vs {b}")
    return hi


def to_f64(v):
    """A scaled value as f64, 0 below the f64 range."""
    x = float(abs(v))
    if x > OVER:
        sys.exit(f"value beyond the f64 range: {v}")
    if x < UNDER:
        return 0.0, 0.0
    c = mpc(v)
    return float(c.real), float(c.imag)


# ------------------------------------------------------------------- definition check


def check_definitions():
    """Closed forms at n <= 2, the relations at z = i x, the Wronskians."""
    worst = mpf(0)

    def rel(a, b):
        nonlocal worst
        e = abs(a - b) / max(abs(b), mpf(10) ** -300)
        worst = max(worst, e)
        if e > mpf(10) ** -40:
            sys.exit(f"definition check failed: {a} vs {b}")

    with mp.workdps(60):
        for z in (mpc("0.3", "0"), mpc("7.5", "0.4"), mpc("2", "9"), mpc("0", "4"), mpc("40", "0")):
            e = mpmath.expj(z)
            rel(jn(0, z), mpmath.sin(z) / z)
            rel(jn(1, z), mpmath.sin(z) / z**2 - mpmath.cos(z) / z)
            rel(jn(2, z), (3 / z**2 - 1) * mpmath.sin(z) / z - 3 * mpmath.cos(z) / z**2)
            rel(hn(0, z), -1j * e / z)
            rel(hn(1, z), -e * (z + 1j) / z**2)
            rel(hn(2, z), 1j * e / z * (1 + 3j / z - 3 / z**2))
            for n in (0, 1, 5, 30):
                # Wronskian j h' - j' h = i / z^2 and the cross product j_n h_{n+1} - j_{n+1} h_n.
                w = jn(n, z) * derivative("h", hn, n, z) - derivative("j", jn, n, z) * hn(n, z)
                rel(w, 1j / z**2)
                rel(jn(n, z) * hn(n + 1, z) - jn(n + 1, z) * hn(n, z), -1j / z**2)
        for x in (mpf("0.01"), mpf("1.5"), mpf("12"), mpf("200")):
            rel(i_n(0, x), mpmath.sinh(x) / x)
            rel(i_n(1, x), mpmath.cosh(x) / x - mpmath.sinh(x) / x**2)
            rel(kn(0, x), mpmath.exp(-x) / x)
            rel(kn(1, x), mpmath.exp(-x) * (1 + x) / x**2)
            for n in (0, 1, 4, 25):
                rel(jn(n, 1j * x), i_pow(n) * i_n(n, x))
                rel(hn(n, 1j * x), -i_pow(-n) * kn(n, x))
                w = i_n(n, x) * derivative("k", kn, n, x) - derivative("i", i_n, n, x) * kn(n, x)
                rel(w, -1 / x**2)
    return worst


# ------------------------------------------------------------------------------ main


def build():
    worst = check_definitions()
    print(f"definitions: worst {mpmath.nstr(worst, 3)}", file=sys.stderr)
    rng = random.Random(SEED)
    sets = []
    for name, fam, z, s, ns in configurations(rng):
        CACHE.clear()
        rows = []
        for n in ns:
            v, d, ev, ed = reference(fam, z, s, n)
            entry = [n, *to_f64(v), *to_f64(d)]
            if fam == "j":
                entry += [float(ev), float(ed)]
            rows.append(entry)
        # z and s as decimal strings: the tests parse them with Rust's correctly rounded
        # str::parse, since serde_json's default float parsing can be one ulp off, which
        # moves e^(iz) by |z| ulp (2e-13 at |z| = 1500).
        sets.append({"name": name, "family": fam, "z": [repr(z[0]), repr(z[1])], "s": repr(s), "rows": rows})
        print(f"{name}: {len(rows)} rows", file=sys.stderr)
    print(f"largest cancellation in the sums: {CANCELLED[0]} digits", file=sys.stderr)
    doc = {
        "about": "Helmholtz CH0.3 (S3) reference radial functions; spikes/radial/tools/reference.py",
        "mpmath": mpmath.__version__,
        "dps": list(DPS),
        "eps": EPS,
        "rows": "[n, value re, value im, derivative re, derivative im (, envelope, envelope of derivative for j)]",
        "sets": sets,
    }
    return json.dumps(doc, separators=(",", ":")) + "\n"


def main():
    text = build()
    if "--check" in sys.argv[1:]:
        if OUT.read_text() != text:
            sys.exit(f"{OUT} differs from a fresh run")
        print("fixture matches", file=sys.stderr)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
