#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Check of the leaf data and relative box geometry of CONVENTIONS §3.13.

Checks, with seeded random keys and points, the geometry in exact rationals
(fractions.Fraction), the f64 evaluation of leaf-scaled coordinates in IEEE doubles
(Python floats, emulating nd_octree in the same operation order), and the scaling
identities in mpmath at 40 significant digits:
  - integer centres: a + C_L w / 2^(L+1) with C_L = (2i + 1) 2^(L - l) is the midpoint
    of the box of nd_octree::morton::physical_box (restated in exact rationals) and the
    centre of §3.12, on every level 0-16 and every reference level L >= l, for a dyadic
    and a generic domain;
  - relative frames: c^(s|t) = (C_L(s) - C_L(t)) 2^(l_t - L) and r^(s|t) = 2^(l_t - l_s)
    equal (c_s - c_t) / r_t and r_s / r_t exactly, for random key pairs on all 17 x 17
    level combinations and three domains; every value round-trips exactly through IEEE
    f32 and f64 (struct.pack); the largest numerator is reported;
  - canonical frames: the relative frames of a child and its parent, and of a V-list
    pair, are the canonical frames of the §3.12 tables;
  - leaf-scaled coordinates: u = (x - a) 2^(l+1) / w - (2i + 1), evaluated in f64 in that
    order and rounded to f32, stays within the error bound of §3.13 of the exact u, at
    the deepest level and on random levels, for a domain far from the origin and one
    around it;
  - containment: for points that points_to_morton puts into a leaf (with the domain of
    compute_global_bounding_box, whose sides agree only to rounding), the exact and the
    stored u lie in [-1, 1]^3 up to the bound of §3.13, also at the leaf faces;
  - frame maps, in exact rationals: the M2P frame (c^(s|t), r^(s|t)) maps u_t to
    (x - c_s) / r_s, the P2L frame (c^(t|s), r^(t|s)) maps u_s to (y - c_t) / r_t, and
    the P2P map c^(s|t) + r^(s|t) u_s gives (y - c_t) / r_t;
  - scaling identities, at 40 digits: P2P in leaf-scaled coordinates times 1 / r_t and
    1 / r_t^2 is the absolute potential and gradient; L2P at the unit frame and M2P at
    the relative frame give r_t and r_t^2 times the absolute potential and gradient;
    P2M at the unit frame and P2L at the relative frame give the coefficients of the
    absolute frames, with no factor.

The expansion operators themselves are checked in Rust by Phase 3 T8; the L2P, M2P,
P2M and P2L sums here are those of check_translations.py, used only to confirm the
frames and the factors.

Run from the repository root:  uv run tools/fixtures/check_leaf_geometry.py
(or: python3 tools/fixtures/check_leaf_geometry.py with mpmath==1.4.1 installed).
Prints the worst error per check and exits non-zero on any failure.
"""
import itertools
import math
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

# Import the sibling scripts without leaving a __pycache__ in the repository.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_translations as ct  # noqa: E402  (the leaf sums of §3.7 and §3.11)
import gen_harmonics as gh  # noqa: E402  (the pinned mpmath version)
import mpmath  # noqa: E402
from mpmath import mp, mpf  # noqa: E402

DPS = 40
TOL = mpf(10) ** -30
SEED = 20261004
P = 8
DEEPEST_LEVEL = 16
EPS64 = Fraction(1, 2**53)  # unit roundoff of f64
EPS32 = Fraction(1, 2**24)  # unit roundoff of f32


# ---------------------------------------------------------------------------------------
# Exact geometry (§3.12, §3.13). A box is (level, index) with index from morton::decode.


def integer_centre(index, level, ref_level):
    """C_L = (2i + 1) 2^(L - l), componentwise."""
    return [(2 * i + 1) * 2 ** (ref_level - level) for i in index]


def centre(a, w, index, level):
    """§3.12: c = a + (i + 1/2) w / 2^l."""
    return [a[k] + (index[k] + Fraction(1, 2)) * w / 2**level for k in range(3)]


def half_width(w, level):
    """§3.12: r_l = w / 2^(l+1)."""
    return w / 2 ** (level + 1)


def physical_box(a, w, index, level):
    """morton::physical_box restated exactly: [a + i w / 2^l, a + (i + 1) w / 2^l]."""
    size = w / 2**level
    return [a[k] + index[k] * size for k in range(3)], [a[k] + (index[k] + 1) * size for k in range(3)]


def relative_frame(s, t):
    """(c^(s|t), r^(s|t), numerators) from the integer centres on L = max(l_s, l_t)."""
    (ls, i_s), (lt, i_t) = s, t
    ref = max(ls, lt)
    numerators = [x - y for x, y in zip(integer_centre(i_s, ls, ref), integer_centre(i_t, lt, ref))]
    return [n * Fraction(2) ** (lt - ref) for n in numerators], Fraction(2) ** (lt - ls), numerators


def random_box(rng, level):
    return level, [rng.randrange(2**level) for _ in range(3)]


def random_domain(rng):
    """A generic cubic domain: rational corner and side, not dyadic."""
    a = [Fraction(rng.randint(-10**7, 10**7), rng.randint(1, 10**6)) for _ in range(3)]
    return a, Fraction(rng.randint(5 * 10**5, 4 * 10**6), rng.randint(10**6 - 10**3, 10**6))


DYADIC_DOMAIN = ([Fraction(-1), Fraction(1, 2), Fraction(3, 4)], Fraction(4))


def round64(x):
    return struct.unpack("<d", struct.pack("<d", float(x)))[0]


def round32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def exact_in(x, rounder):
    """True if the rational x is exactly representable (rounding it changes nothing)."""
    return Fraction(rounder(x)) == x


def significant_bits(x):
    """Bits of the odd part of the numerator of a dyadic rational in lowest terms."""
    n = abs(Fraction(x).numerator)
    while n and n % 2 == 0:
        n //= 2
    return n.bit_length()


def check_integer_centres(rng):
    failures = count = 0
    domains = [DYADIC_DOMAIN, random_domain(rng)]
    for a, w in domains:
        for level in range(DEEPEST_LEVEL + 1):
            n = 2**level
            indices = [[0, 0, 0], [n - 1] * 3] + [[rng.randrange(n) for _ in range(3)] for _ in range(12)]
            for index in indices:
                lo, hi = physical_box(a, w, index, level)
                mid = [(x + y) / 2 for x, y in zip(lo, hi)]
                failures += any(y - x != 2 * half_width(w, level) for x, y in zip(lo, hi))
                for ref in range(level, DEEPEST_LEVEL + 1):
                    C = integer_centre(index, level, ref)
                    c = [a[k] + C[k] * w / 2 ** (ref + 1) for k in range(3)]
                    failures += c != mid or c != centre(a, w, index, level)
                    # C_L is odd times 2^(L - l), in [1, 2^(L+1) - 1].
                    step = 2 ** (ref - level)
                    failures += any(Ck % step != 0 or (Ck // step) % 2 != 1 for Ck in C)
                    failures += any(not 1 <= Ck <= 2 ** (ref + 1) - 1 for Ck in C)
                    count += 1
    detail = f"{count} (box, L) pairs, levels 0-16, all L >= l, dyadic and generic domain, exact"
    return mpf(failures), mpf(0), detail


def check_relative_frames(rng):
    failures = pairs = 0
    max_num, max_reduced, max_bits = 0, 0, 0
    domains = [DYADIC_DOMAIN, random_domain(rng), random_domain(rng)]
    for ls, lt in itertools.product(range(DEEPEST_LEVEL + 1), repeat=2):
        last_s, last_t = 2**ls - 1, 2**lt - 1
        boxes = [((ls, [0, 0, 0]), (lt, [last_t] * 3)), ((ls, [last_s] * 3), (lt, [0, 0, 0]))]
        boxes += [(random_box(rng, ls), random_box(rng, lt)) for _ in range(4)]
        for s, t in boxes:
            c_hat, r_hat, numerators = relative_frame(s, t)
            for a, w in domains:
                cs, ct_ = centre(a, w, s[1], ls), centre(a, w, t[1], lt)
                rs, rt = half_width(w, ls), half_width(w, lt)
                failures += [(cs[k] - ct_[k]) / rt for k in range(3)] != c_hat
                failures += rs / rt != r_hat
            # The reverse frame, as P2L uses it.
            c_rev, r_rev, _ = relative_frame(t, s)
            failures += c_rev != [-c / r_hat for c in c_hat] or r_rev != 1 / r_hat
            for v in c_hat + [r_hat]:
                failures += not exact_in(v, round32) or not exact_in(v, round64)
                max_reduced = max(max_reduced, abs(v.numerator))
                max_bits = max(max_bits, significant_bits(v))
            max_num = max(max_num, max(abs(n) for n in numerators))
            pairs += 1
    failures += max_num > 2 ** (DEEPEST_LEVEL + 1) - 2
    detail = (
        f"{pairs} key pairs, levels 0-16 x 0-16, 3 domains; largest |C_L(s) - C_L(t)| = {max_num}"
        f" (2^17 - 2 = {2**17 - 2}), largest reduced numerator {max_reduced},"
        f" at most {max_bits} significant bits; all exact in f32 and f64"
    )
    return mpf(failures), mpf(0), detail


def octant_sign(o):
    return [(o >> 2 & 1) * 2 - 1, (o >> 1 & 1) * 2 - 1, (o & 1) * 2 - 1]


def check_canonical_frames(rng):
    failures = count = 0
    offsets = [d for d in itertools.product(range(-3, 4), repeat=3) if max(abs(c) for c in d) >= 2]
    for level in range(DEEPEST_LEVEL):
        for _ in range(4):
            parent = random_box(rng, level)
            for o in range(8):
                bits = [o >> 2 & 1, o >> 1 & 1, o & 1]
                child = (level + 1, [2 * parent[1][k] + bits[k] for k in range(3)])
                c_hat, r_hat, _ = relative_frame(child, parent)
                # M2M(o): b = c^(child|parent), rho = r^(child|parent); L2L(o): t and sigma alike.
                failures += c_hat != [Fraction(s, 2) for s in octant_sign(o)] or r_hat != Fraction(1, 2)
                count += 1
    for level in range(2, DEEPEST_LEVEL + 1):
        for _ in range(40):
            source = random_box(rng, level)
            d = rng.choice(offsets)
            target = [source[1][k] + d[k] for k in range(3)]
            if all(0 <= x < 2**level for x in target):
                # M2L(d): b = c^(target|source), sigma = r^(target|source).
                c_hat, r_hat, _ = relative_frame((level, target), source)
                failures += c_hat != [2 * x for x in d] or r_hat != 1
                count += 1
    detail = f"{count} child-parent and V-list pairs: (s_o / 2, 1/2) and (2d, 1), exact"
    return mpf(failures), mpf(0), detail


# ---------------------------------------------------------------------------------------
# Leaf-scaled coordinates in IEEE doubles. Python floats are IEEE binary64 with correctly
# rounded +, -, *, /, so these functions reproduce nd_octree and the formula of §3.13
# operation by operation.


def bounding_box(points):
    """nd_octree::octree::compute_global_bounding_box on one rank, in f64."""
    lo = [min(x[k] for x in points) for k in range(3)]
    hi = [max(x[k] for x in points) for k in range(3)]
    diam = [hi[k] - lo[k] for k in range(3)]
    mean = [lo[k] + 0.5 * diam[k] for k in range(3)]
    max_diam = max(diam) * (1.0 + 1.0 / 2**DEEPEST_LEVEL)
    return [mean[k] - 0.5 * max_diam for k in range(3)], [mean[k] + 0.5 * max_diam for k in range(3)]


def leaf_index(x, lo, hi, level):
    """points_to_morton (morton::from_physical_point at level 16), then the ancestor
    on `level`: index >> (16 - level)."""
    index = []
    for k in range(3):
        reference = (x[k] - lo[k]) / (hi[k] - lo[k])
        deepest = min(int(reference * 2.0**DEEPEST_LEVEL), 2**DEEPEST_LEVEL - 1)
        index.append(deepest >> (DEEPEST_LEVEL - level))
    return index


def leaf_scaled_f64(x, a, w, index, level):
    """§3.13: u = (x - a) 2^(l+1) / w - (2i + 1), in f64, left to right."""
    return [(x[k] - a[k]) * 2.0 ** (level + 1) / w - (2 * index[k] + 1) for k in range(3)]


def leaf_scaled_exact(x, a, w, index, level):
    return [(Fraction(x[k]) - Fraction(a[k])) * 2 ** (level + 1) / Fraction(w) - (2 * index[k] + 1) for k in range(3)]


def error_bound_f64(x_minus_a, r, u_exact):
    """Rigorous form of the f64 bound of §3.13 for one component:
    (2e + e^2) |x - a| / r for the rounding of x - a and the scaling, plus e |u| for the
    subtraction of 2i + 1, with e the f64 unit roundoff."""
    e = EPS64
    scaling = (2 * e + e * e) * abs(x_minus_a) / r
    return scaling + e * (abs(u_exact) + scaling)


def error_bound_f32(bound64, u_exact):
    """The f64 bound plus the cast to f32: e32 |u|."""
    return bound64 + EPS32 * (abs(u_exact) + bound64)


def containment_bound(w_axis, w, level):
    """beta of §3.13 in rigorous form: 2^(l+1) g with g the largest relative deviation of
    2^l (x - a) / w from 2^l fl(fl(x - a) / w_k), for 0 <= fl(fl(x - a) / w_k) <= 1."""
    e = EPS64
    ratio = Fraction(w_axis) / Fraction(w)
    g = max(abs(ratio / (1 - e) ** 2 - 1), abs(ratio / (1 + e) ** 2 - 1))
    return 2 ** (level + 1) * g


def domain_cloud(rng, centre_, spread, count):
    return [[centre_[k] + spread * (2 * rng.random() - 1) for k in range(3)] for _ in range(count)]


def face_probes(rng, lo, hi, level, count):
    """Points within a few ulps of the faces of random leaves on `level`, and of the
    domain's own faces, strictly inside the domain."""
    probes = []
    for _ in range(count):
        axis = rng.randrange(3)
        base = [lo[k] + (hi[k] - lo[k]) * rng.random() for k in range(3)]
        side = hi[axis] - lo[axis]
        face = lo[axis] + rng.randrange(1, 2**level) * (side / 2**level)
        for target in (face, lo[axis], hi[axis]):
            x = target
            for _ in range(4):
                x = math.nextafter(x, -math.inf)
            for _ in range(9):
                point = list(base)
                point[axis] = x
                if lo[axis] < x < hi[axis]:
                    probes.append(point)
                x = math.nextafter(x, math.inf)
    return probes


def leaf_scaled_samples(rng):
    """Domains, points and levels shared by the error and the containment checks."""
    domains = [
        ("far from the origin", [3.7e5 + 0.3, -1.2e6 - 0.7, 8.1e5 + 0.1], 37.25),
        ("around the origin", [0.2, -0.35, 0.05], 1.3),
    ]
    samples = []
    for name, centre_, spread in domains:
        cloud = domain_cloud(rng, centre_, spread, 1500)
        lo, hi = bounding_box(cloud)
        # The side of u: the x side of the box, so the y and z sides differ by rounding.
        w = hi[0] - lo[0]
        points = [(x, DEEPEST_LEVEL) for x in cloud]
        points += [(x, rng.randint(0, DEEPEST_LEVEL)) for x in cloud[:500]]
        points += [(x, DEEPEST_LEVEL) for x in face_probes(rng, lo, hi, DEEPEST_LEVEL, 60)]
        points += [(x, level) for level in (3, 9) for x in face_probes(rng, lo, hi, level, 20)]
        samples.append((name, lo, hi, w, points))
    return samples


def check_leaf_scaled(samples):
    worst64 = worst32 = Fraction(0)
    max_err64 = max_err32 = Fraction(0)
    count = 0
    for _, lo, hi, w, points in samples:
        a = lo
        for x, level in points:
            index = leaf_index(x, lo, hi, level)
            u64 = leaf_scaled_f64(x, a, w, index, level)
            u_exact = leaf_scaled_exact(x, a, w, index, level)
            r = Fraction(w) / 2 ** (level + 1)
            for k in range(3):
                b64 = error_bound_f64(Fraction(x[k]) - Fraction(a[k]), r, u_exact[k])
                b32 = error_bound_f32(b64, u_exact[k])
                e64 = abs(Fraction(u64[k]) - u_exact[k])
                e32 = abs(Fraction(round32(u64[k])) - u_exact[k])
                worst64, worst32 = max(worst64, e64 / b64), max(worst32, e32 / b32)
                if level == DEEPEST_LEVEL:
                    max_err64, max_err32 = max(max_err64, e64), max(max_err32, e32)
            count += 1
    detail = (
        f"{count} points, 2 domains; worst error / bound f64 {float(worst64):.3f}, f32 {float(worst32):.3f};"
        f" largest error at level 16: f64 {float(max_err64):.2e}, f32 {float(max_err32):.2e}"
    )
    return mpf(float(max(worst64, worst32))), mpf(1), detail


def check_containment(samples):
    worst = Fraction(0)
    excess64 = excess32 = excess_exact = Fraction(0)
    beta_max = Fraction(0)
    mismatch = 0
    count = 0
    for _, lo, hi, w, points in samples:
        a = lo
        sides = [hi[k] - lo[k] for k in range(3)]
        mismatch = max(mismatch, max(abs(Fraction(s) - Fraction(w)) / (EPS64 * Fraction(w)) for s in sides))
        for x, level in points:
            index = leaf_index(x, lo, hi, level)
            u64 = leaf_scaled_f64(x, a, w, index, level)
            u_exact = leaf_scaled_exact(x, a, w, index, level)
            r = Fraction(w) / 2 ** (level + 1)
            for k in range(3):
                beta = containment_bound(sides[k], w, level)
                b64 = error_bound_f64(Fraction(x[k]) - Fraction(a[k]), r, u_exact[k])
                b32 = error_bound_f32(b64, u_exact[k])
                over_exact = abs(u_exact[k]) - 1
                over64 = abs(Fraction(u64[k])) - 1
                over32 = abs(Fraction(round32(u64[k]))) - 1
                worst = max(worst, over_exact / beta, over64 / (beta + b64), over32 / (beta + b32))
                if level == DEEPEST_LEVEL:
                    beta_max = max(beta_max, beta)
                excess_exact, excess64, excess32 = max(excess_exact, over_exact), max(excess64, over64), max(excess32, over32)
            count += 1
    detail = (
        f"{count} points incl. leaf-face probes; sides differ by up to {float(mismatch):.1f} w e64;"
        f" largest |u| - 1: exact {float(excess_exact):.2e}, f64 {float(excess64):.2e},"
        f" f32 {float(excess32):.2e}; beta at level 16 up to {float(beta_max):.2e}"
    )
    return mpf(float(worst)), mpf(1), detail


# ---------------------------------------------------------------------------------------
# Frame maps (exact) and scaling identities (mpmath).


def to_mp(x):
    return mpf(x.numerator) / x.denominator


def random_scaled_point(rng):
    """A rational point of [-1, 1]^3, in leaf-scaled coordinates."""
    return [Fraction(rng.randint(-10**9, 10**9), 10**9) for _ in range(3)]


def frame_maps(s, t, u_s, u_t, a, w):
    """Exact x, y and the three maps of §3.13 for a source box s and a target box t."""
    cs, ct_ = centre(a, w, s[1], s[0]), centre(a, w, t[1], t[0])
    rs, rt = half_width(w, s[0]), half_width(w, t[0])
    x = [ct_[k] + rt * u_t[k] for k in range(3)]
    y = [cs[k] + rs * u_s[k] for k in range(3)]
    c_st, r_st, _ = relative_frame(s, t)
    c_ts, r_ts, _ = relative_frame(t, s)
    m2p = [(u_t[k] - c_st[k]) / r_st for k in range(3)]
    p2l = [(u_s[k] - c_ts[k]) / r_ts for k in range(3)]
    p2p = [c_st[k] + r_st * u_s[k] for k in range(3)]
    return x, y, (cs, rs), (ct_, rt), m2p, p2l, p2p


def check_frame_maps(rng):
    failures = count = 0
    for _ in range(400):
        a, w = random_domain(rng)
        s, t = random_box(rng, rng.randint(0, DEEPEST_LEVEL)), random_box(rng, rng.randint(0, DEEPEST_LEVEL))
        u_s, u_t = random_scaled_point(rng), random_scaled_point(rng)
        x, y, (cs, rs), (ct_, rt), m2p, p2l, p2p = frame_maps(s, t, u_s, u_t, a, w)
        failures += m2p != [(x[k] - cs[k]) / rs for k in range(3)]
        failures += p2l != [(y[k] - ct_[k]) / rt for k in range(3)]
        failures += p2p != [(y[k] - ct_[k]) / rt for k in range(3)]
        # |x - y| = r_t |u_t - y^|, compared squared.
        failures += sum((x[k] - y[k]) ** 2 for k in range(3)) != rt**2 * sum((u_t[k] - p2p[k]) ** 2 for k in range(3))
        count += 1
    return mpf(failures), mpf(0), f"{count} random pairs (s, t) on levels 0-16, generic domains, exact"


def moderate_domain(rng):
    """A generic domain with |a| <= 10 and w in [1/2, 4], so that the absolute frames lose
    at most about 6 of the 40 digits to cancellation at level 16."""
    a = [Fraction(rng.randint(-10**7, 10**7), 10**6) for _ in range(3)]
    return a, Fraction(rng.randint(5 * 10**5, 4 * 10**6), 10**6 - rng.randint(1, 10**3))


def p2p_mp(sources, charges, target):
    """Potential, gradient and the term magnitudes of sum q / |x - y| at one target."""
    phi, grad, size, gsize = mpf(0), [mpf(0)] * 3, mpf(0), mpf(0)
    for y, q in zip(sources, charges):
        d = [target[k] - y[k] for k in range(3)]
        r2 = sum(c * c for c in d)
        rr = mpmath.sqrt(r2)
        phi += q / rr
        grad = [grad[k] - q * d[k] / (r2 * rr) for k in range(3)]
        size += abs(q) / rr
        gsize += abs(q) / r2
    return phi, grad, size, gsize


def check_p2p(rng):
    worst = mpf(0)
    count = 0
    for case in range(60):
        a, w = moderate_domain(rng)
        t = random_box(rng, rng.randint(0, DEEPEST_LEVEL))
        s = t if case % 4 == 0 else random_box(rng, rng.randint(0, DEEPEST_LEVEL))
        u_t = random_scaled_point(rng)
        sources_u = [random_scaled_point(rng) for _ in range(5)]
        q = ct.charges(rng, 5)
        ct_, rt = centre(a, w, t[1], t[0]), half_width(w, t[0])
        x = [ct_[k] + rt * u_t[k] for k in range(3)]
        absolute, scaled = [], []
        for u_s in sources_u:
            _, y, _, _, _, _, _ = frame_maps(s, t, u_s, u_t, a, w)
            absolute.append([to_mp(c) for c in y])
            if s == t:
                # y^ = u_s for s = t: no arithmetic at all.
                scaled.append([to_mp(c) for c in u_s])
            else:
                # y^ = c^(s|t) + r^(s|t) u_s, formed in floating point at 40 digits.
                c_hat, r_hat, _ = relative_frame(s, t)
                scaled.append([to_mp(c_hat[k]) + to_mp(r_hat) * to_mp(u_s[k]) for k in range(3)])
        phi, grad, size, gsize = p2p_mp(absolute, q, [to_mp(c) for c in x])
        phi_hat, grad_hat, _, _ = p2p_mp(scaled, q, [to_mp(c) for c in u_t])
        r = to_mp(rt)
        worst = max(worst, abs(phi_hat / r - phi) / size)
        worst = max(worst, max(abs(grad_hat[k] / r**2 - grad[k]) for k in range(3)) / gsize)
        count += 1
    detail = f"{count} leaf pairs on levels 0-16 (every 4th s = t), 5 sources each, relative to the terms"
    return worst, TOL, detail


def separated_pair(rng, a, w, far_from):
    """Boxes s, t and scaled points with the evaluation point at scaled distance >= 2 from
    the expansion centre: far_from = 's' for M2P (x from c_s), 't' for P2L (y from c_t)."""
    while True:
        s, t = random_box(rng, rng.randint(0, DEEPEST_LEVEL)), random_box(rng, rng.randint(0, DEEPEST_LEVEL))
        u_s, u_t = random_scaled_point(rng), random_scaled_point(rng)
        maps = frame_maps(s, t, u_s, u_t, a, w)
        v = maps[4] if far_from == "s" else maps[5]
        if sum(c * c for c in v) >= 4:
            return s, t, u_s, u_t, maps


def gradient(p, coeffs, v, regular, r):
    return [g / r**2 for g in ct.ladder_gradient(p, coeffs, v, regular)]


def check_l2p_m2p(rng):
    worst = mpf(0)
    unit = ([mpf(0)] * 3, mpf(1))
    for _ in range(12):
        a, w = moderate_domain(rng)
        # L2P at leaf t: Frame((0, 0, 0), 1) on u_t, against the absolute frame (c_t, r_t).
        t = random_box(rng, rng.randint(0, DEEPEST_LEVEL))
        u_t = random_scaled_point(rng)
        ct_, rt = centre(a, w, t[1], t[0]), half_width(w, t[0])
        x = [to_mp(ct_[k] + rt * u_t[k]) for k in range(3)]
        absolute = ([to_mp(c) for c in ct_], to_mp(rt))
        L = ct.random_coefficients(rng, P)
        phi, size = ct.evaluate(P, absolute, L, x, True)
        phi_hat, _ = ct.evaluate(P, unit, L, [to_mp(c) for c in u_t], True)
        r = to_mp(rt)
        worst = max(worst, abs(phi_hat - r * phi) / (r * size))
        g = gradient(P, L, ct.scaled(x, absolute), True, absolute[1])
        g_hat = gradient(P, L, [to_mp(c) for c in u_t], True, mpf(1))
        worst = max(worst, max(abs(g_hat[k] - r**2 * g[k]) for k in range(3)) / sum(abs(r**2 * c) for c in g))

        # M2P from box s at leaf t: Frame(c^(s|t), r^(s|t)) on u_t, against (c_s, r_s).
        s, t, _, u_t, (x, _, (cs, rs), (ct_, rt), _, _, _) = separated_pair(rng, a, w, "s")
        c_hat, r_hat, _ = relative_frame(s, t)
        relative = ([to_mp(c) for c in c_hat], to_mp(r_hat))
        absolute = ([to_mp(c) for c in cs], to_mp(rs))
        x, u_t = [to_mp(c) for c in x], [to_mp(c) for c in u_t]
        M = ct.random_coefficients(rng, P)
        phi, size = ct.evaluate(P, absolute, M, x, False)
        phi_hat, _ = ct.evaluate(P, relative, M, u_t, False)
        r = to_mp(rt)
        worst = max(worst, abs(phi_hat - r * phi) / (r * size))
        g = gradient(P, M, ct.scaled(x, absolute), False, absolute[1])
        g_hat = gradient(P, M, ct.scaled(u_t, relative), False, relative[1])
        worst = max(worst, max(abs(g_hat[k] - r**2 * g[k]) for k in range(3)) / sum(abs(r**2 * c) for c in g))
    return worst, TOL, f"p = {P}, 12 leaves (L2P) and 12 separated pairs (M2P), levels 0-16"


def check_p2m_p2l(rng):
    worst = mpf(0)
    unit = ([mpf(0)] * 3, mpf(1))
    for _ in range(12):
        a, w = moderate_domain(rng)
        # P2M at leaf s: Frame((0, 0, 0), 1) on u_s, against (c_s, r_s).
        s = random_box(rng, rng.randint(0, DEEPEST_LEVEL))
        sources_u = [random_scaled_point(rng) for _ in range(4)]
        q = ct.charges(rng, 4)
        cs, rs = centre(a, w, s[1], s[0]), half_width(w, s[0])
        y = [[to_mp(cs[k] + rs * u[k]) for k in range(3)] for u in sources_u]
        absolute = ([to_mp(c) for c in cs], to_mp(rs))
        want = ct.p2m(P, absolute, y, q)
        got = ct.p2m(P, unit, [[to_mp(c) for c in u] for u in sources_u], q)
        worst = max(worst, ct.degree_error(P, got, want, False))

        # P2L from leaf s into box t: Frame(c^(t|s), r^(t|s)) on u_s, against (c_t, r_t).
        s, t, u_s, _, (_, y, _, (ct_, rt), _, _, _) = separated_pair(rng, a, w, "t")
        c_hat, r_hat, _ = relative_frame(t, s)
        relative = ([to_mp(c) for c in c_hat], to_mp(r_hat))
        absolute = ([to_mp(c) for c in ct_], to_mp(rt))
        q = ct.charges(rng, 1)
        want = ct.p2l(P, absolute, [[to_mp(c) for c in y]], q)
        got = ct.p2l(P, relative, [[to_mp(c) for c in u_s]], q)
        worst = max(worst, ct.degree_error(P, got, want, True))
    return worst, TOL, f"p = {P}, 12 leaves (P2M) and 12 separated pairs (P2L), per degree, §3.8 weighting"


def main():
    if mpmath.__version__ != gh.MPMATH_VERSION:
        sys.exit(f"needs mpmath {gh.MPMATH_VERSION}, found {mpmath.__version__}")
    mp.dps = DPS
    rng = random.Random(SEED)
    samples = leaf_scaled_samples(rng)
    checks = [
        ("integer centres = physical_box midpoint", check_integer_centres),
        ("relative frames exact, f32 and f64", check_relative_frames),
        ("relative frames = canonical frames", check_canonical_frames),
        ("leaf-scaled u within the f64/f32 bound", lambda _: check_leaf_scaled(samples)),
        ("u in [-1, 1]^3 up to the bound", lambda _: check_containment(samples)),
        ("M2P, P2L and P2P frame maps", check_frame_maps),
        ("P2P scaled / r_t, / r_t^2 = absolute", check_p2p),
        ("L2P, M2P give r_t phi, r_t^2 grad phi", check_l2p_m2p),
        ("P2M, P2L coefficients, no factor", check_p2m_p2l),
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
