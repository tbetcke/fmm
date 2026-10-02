#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""Check of the coincident-pair rule and the domain of r^2 of fast P2P kernels,
CONVENTIONS §3.13 ("Fast kernels").

A fast kernel computes, per pair and component, d = fl(u_t - y^) in T and
r^2 = d_0^2 + d_1^2 + d_2^2 in T (any order, with or without fma), and skips the pair
when r^2 = 0. §3.13 derives, for T = f32 and f64 and the leaf-scaled data of the FMM,
with G_k = 2^-k Z:
  - every stored coordinate lies in G_53, every mapped U-list source
    y^ = fl(c^(s|t) + r^(s|t) u_s) in G_54, hence every d in G_54 (the short
    argument); so r^2 = 0 exactly when u_t == y^ in all three components (the rule of
    nd_fmm_ref::p2p on the same inputs), and otherwise r^2 >= 2^-108, the lower end of
    the kernel domain, a normal number in f32 (2^-126) with a margin of 2^18;
  - in fact y^ and d lie in G_53 (the finer argument, which uses |u| <= 1 + 1/8): a
    nonzero |d| is at least 2^-53 and a nonzero r^2 at least 2^-106, both attained;
  - r^2 <= 3 (6 + 3 b)^2 (1 + e_T)^7 < 2^7 for |u| <= 1 + b, b <= 1/8;
  - in f32 with gradients, the intermediates r^2 r (reference) and (q rho) rho^2 (fast
    kernels) stay normal and finite for r^2 >= 2^-84, |q| <= 1, and leave that range
    below.

This script checks those claims, with the stored values computed as the code computes
them: u in IEEE doubles in the order of §3.13, by check_leaf_geometry.py (Python
floats), cast to f32; y^, d and r^2 operation by operation in T. An f32 operation on
f32 operands is the f32 rounding of the f64 result, which is correctly rounded because
53 >= 2 * 24 + 2; an fma is rounded once from its exact value (fractions.Fraction). The
claims themselves are checked in exact rationals. Checks:
  - the f32 and f64 rounding emulation against exact rounding of rationals;
  - the formula: for every double v near the subtrahends 2i + 1 (and near +-1/2 from
    them, and tiny v), u = fl(v - (2i + 1)) and its f32 cast lie in G_53; the smallest
    nonzero |u| is reported;
  - adversarial pairs: target leaves on levels 0-16 at the first, last, middle and
    random indices, the leaf itself and every U-list neighbour that 2:1 balance allows (coarser, same
    level, finer; every c^ component), with points within a few ulps of the leaf
    centres, the leaf faces, the source centres and the cancellations
    c^ + r^ u_s ~ 0, in four domains (dyadic at the origin, dyadic, generic, far from
    the origin): the grids, the lower bounds, the equivalence r^2 = 0 <=> u_t == y^,
    with r^2 as the reference forms it and with fma in two orders;
  - seeded random leaf-scaled pairs on levels 0-16 (dyadic, generic, far from the
    origin), with duplicated points in the self pairs: the same claims;
  - the upper bound, at the far corners of corner neighbours on all three levels;
  - the f32 range of the gradient intermediates.

Run from the repository root:  uv run tools/fixtures/check_p2p_domain.py
(or: python3 tools/fixtures/check_p2p_domain.py with mpmath==1.4.1 installed, which
check_leaf_geometry.py imports). Prints the extremes per check and exits non-zero on
any failure. It writes nothing.
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

import check_leaf_geometry as lg  # noqa: E402  (the emulation of nd_octree and §3.13)
import gen_harmonics as gh  # noqa: E402  (the pinned mpmath version)
import mpmath  # noqa: E402

SEED = 20261006
DEEPEST_LEVEL = 16
WINDOW = 4  # ulps on either side of every adversarial point

# The derived bounds of §3.13, "Fast kernels".
STORED_GRID = 53  # u in G_53
MAPPED_GRID = 54  # y^ and d in G_54 (U lists, 2:1 balance): the short argument
SHARP_GRID = 53  # y^ and d in G_53 (U lists, |u| <= 1 + 1/8): the finer argument
D_MIN = Fraction(1, 2**53)  # smallest nonzero |d| (from G_53)
D_MIN_FLOAT = 2.0**-53
R2_MIN = Fraction(1, 2**108)  # the lower end of the kernel domain (short argument)
R2_SHARP = Fraction(1, 2**106)  # the smallest nonzero r^2 (finer argument), attained
R2_MAX = Fraction(2**7)
R2_SHARP_FLOAT, R2_MAX_FLOAT = float(R2_SHARP), float(R2_MAX)
B_MAX = Fraction(1, 8)  # largest |u| - 1 for which R2_MAX holds
GRADIENT_R2_MIN = Fraction(1, 2**84)  # f32 gradient intermediates normal and finite


class Format:
    """An IEEE binary format: significand bits p (with the hidden bit), exponent range."""

    def __init__(self, name, p, emin, emax):
        self.name, self.p, self.emin, self.emax = name, p, emin, emax
        self.u = Fraction(1, 2**p)  # unit roundoff
        self.min_normal = Fraction(2) ** emin
        self.min_normal_float = float(self.min_normal)
        self.max = (2 - Fraction(2) ** (1 - p)) * Fraction(2) ** emax


F32 = Format("f32", 24, -126, 127)
F64 = Format("f64", 53, -1022, 1023)
FORMATS = (F32, F64)


def round_exact(x, fmt):
    """x (a rational) rounded to fmt, to nearest with ties to even, with gradual
    underflow; OverflowError above the largest finite value."""
    x = Fraction(x)
    if x == 0:
        return Fraction(0)
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** e:
        e -= 1
    quantum = Fraction(2) ** (max(e, fmt.emin) - (fmt.p - 1))
    scaled = a / quantum
    m = scaled.numerator // scaled.denominator
    rest = scaled - m
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and m % 2 == 1):
        m += 1
    result = m * quantum
    if result > fmt.max:
        raise OverflowError(f"{float(x)} overflows {fmt.name}")
    return result if x > 0 else -result


def round32(x):
    """The f32 nearest to the double x (struct.pack rounds to nearest, ties to even)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def to_t(x, fmt):
    """The double x rounded to fmt (a no-op for f64)."""
    return x if fmt is F64 else round32(x)


def fma(a, b, c, fmt):
    """fl(a b + c), rounded once."""
    return float(round_exact(Fraction(a) * Fraction(b) + Fraction(c), fmt))


def in_grid(x, k):
    """x in G_k = 2^-k Z (x a double; scaling by 2^k is exact)."""
    return (x * 2.0**k).is_integer()


def grid_exponent(x):
    """The smallest k with x in G_k (x a nonzero double): log2 of its denominator."""
    return x.as_integer_ratio()[1].bit_length() - 1


def log2(x):
    return math.log2(Fraction(x)) if x else -math.inf


def p2(x):
    """A positive power-of-two-ish value as '2^k' (k rounded to 2 decimals)."""
    k = log2(x)
    return f"2^{k:.0f}" if k == int(k) else f"2^{k:.2f}"


# ---------------------------------------------------------------------------------------
# Domains and keys.


class Domain:
    """A cubic domain as nd_fmm_exec::geometry::Domain sees it: the box lo, hi that
    points_to_morton uses (one side per axis) and the side w = max_k fl(hi_k - lo_k)."""

    def __init__(self, name, lo, hi):
        self.name, self.lo, self.hi = name, lo, hi
        self.w = max(hi[k] - lo[k] for k in range(3))


def domains(rng):
    around = lg.bounding_box(lg.domain_cloud(rng, [0.2, -0.35, 0.05], 1.3, 200))
    far = lg.bounding_box(lg.domain_cloud(rng, [3.7e5 + 0.3, -1.2e6 - 0.7, 8.1e5 + 0.1], 37.25, 200))
    a, w = lg.DYADIC_DOMAIN
    return {
        "dyadic at the origin": Domain("dyadic at the origin", [0.0] * 3, [1.0] * 3),
        "dyadic": Domain("dyadic", [float(c) for c in a], [float(c + w) for c in a]),
        "generic": Domain("generic", *around),
        "far from the origin": Domain("far from the origin", *far),
    }


def adjacent(a, b):
    """interaction_manager::is_adjacent: closed cubes touch, neither contains the other."""
    (la, ia), (lb, ib) = a, b
    level = max(la, lb)
    sa, sb = level - la, level - lb
    if la <= lb and all(ib[k] >> (lb - la) == ia[k] for k in range(3)):
        return False
    if lb <= la and all(ia[k] >> (la - lb) == ib[k] for k in range(3)):
        return False
    for k in range(3):
        a_min, b_min = ia[k] << sa, ib[k] << sb
        a_max, b_max = a_min + (1 << sa) - 1, b_min + (1 << sb) - 1
        if a_min > b_max + 1 or b_min > a_max + 1:
            return False
    return True


def u_list(t):
    """Every key on levels l_t - 1, l_t, l_t + 1 adjacent to t: the U-list entries that
    a 2:1-balanced tree can give t (each in some tree)."""
    lt, it = t
    out = []
    for ls in (lt - 1, lt, lt + 1):
        if not 0 <= ls <= DEEPEST_LEVEL:
            continue
        n = 2**ls
        ranges = []
        for k in range(3):
            if ls < lt:
                centre_ = it[k] >> 1
            elif ls > lt:
                centre_ = 2 * it[k]
            else:
                centre_ = it[k]
            ranges.append([j for j in range(centre_ - 2, centre_ + 4) if 0 <= j < n])
        for index in itertools.product(*ranges):
            s = (ls, list(index))
            if adjacent(s, t):
                out.append(s)
    return out


def index_patterns(rng, level):
    n = 2**level
    return sorted({0, 1 % n, n - 1, max(n - 2, 0), n // 2, max(n // 2 - 1, 0), rng.randrange(n)})


def targets(rng, level):
    patterns = index_patterns(rng, level)
    keys = [(level, [i, i, i]) for i in patterns]
    keys.append((level, [rng.choice(patterns) for _ in range(3)]))
    return keys


# ---------------------------------------------------------------------------------------
# One component of one pair: the stored values as the code computes them.


def axis_index(x, dom, k, level):
    """The index along axis k of the leaf on `level` that points_to_morton gives x."""
    point = [x, x, x]
    return lg.leaf_index(point, dom.lo, dom.hi, level)[k]


def stored_u(x, dom, k, level, index):
    """§3.13 in f64, component k: fl(fl(fl(x - a) 2^(l+1) / w) - (2i + 1))."""
    point = [x, x, x]
    index3 = [index] * 3
    return lg.leaf_scaled_f64(point, dom.lo, dom.w, index3, level)[k]


def ordered(x):
    """The position of the double x in the order of the doubles (an integer)."""
    bits = struct.unpack("<q", struct.pack("<d", x))[0]
    return bits if bits >= 0 else -(bits & 0x7FFF_FFFF_FFFF_FFFF)


def from_ordered(n):
    bits = n if n >= 0 else -n | 1 << 63
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def coarse_ulp(x, dom, k):
    """The spacing of fl(x - a) near x: the ulp of the largest of |x|, |a_k|, |hi_k|."""
    return math.ulp(max(abs(x), abs(dom.lo[k]), abs(dom.hi[k])))


def face(x0, dom, k, level, below):
    """The smallest double near x0 that points_to_morton puts above index `below` on
    `level` along axis k: the face of the leaves as the octree draws it, which differs
    from a + j w / 2^l by the rounding of fl(x - a) / w_k and by w_k != w."""
    span = 64 * coarse_ulp(x0, dom, k)
    lo, hi = max(x0 - span, math.nextafter(dom.lo[k], math.inf)), min(x0 + span, math.nextafter(dom.hi[k], -math.inf))
    if not axis_index(lo, dom, k, level) <= below < axis_index(hi, dom, k, level):
        return None
    a, b = ordered(lo), ordered(hi)
    while b - a > 1:
        mid = (a + b) // 2
        if axis_index(from_ordered(mid), dom, k, level) > below:
            b = mid
        else:
            a = mid
    return from_ordered(b)


def window(x, dom, k):
    """x, its WINDOW neighbours on either side, and WINDOW steps of the spacing of
    fl(x - a) on either side, where the stored u changes."""
    xs = {x}
    lo = hi = x
    for _ in range(WINDOW):
        lo, hi = math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)
        xs.update((lo, hi))
    step = coarse_ulp(x, dom, k)
    xs.update(x + j * step for j in range(-WINDOW, WINDOW + 1))
    return xs


def centre_exact(dom, k, level, index):
    return Fraction(dom.lo[k]) + (2 * index + 1) * Fraction(dom.w) / 2 ** (level + 1)


def axis_frame(k, s, t):
    c_hat, r_hat, _ = lg.relative_frame(s, t)
    return c_hat[k], r_hat


def leaf_values(dom, k, key, positions):
    """(x, u in f64) for the doubles near the scaled positions p of the leaf `key`
    (x = c + p r, and at p = +-1 the face as points_to_morton draws it), that
    points_to_morton puts into the leaf along axis k."""
    level, index = key[0], key[1][k]
    c, r = centre_exact(dom, k, level, index), Fraction(dom.w) / 2 ** (level + 1)
    centres = set()
    for p in positions:
        x = float(c + p * r)
        centres.add(x)
        if abs(p) == 1:
            x_face = face(x, dom, k, level, index if p == 1 else index - 1)
            if x_face is not None:
                centres.add(x_face)
    seen, out = set(), []
    for x0 in sorted(centres):
        for x in sorted(window(x0, dom, k)):
            if x in seen or not dom.lo[k] < x < dom.hi[k]:
                continue
            seen.add(x)
            if axis_index(x, dom, k, level) == index:
                out.append((x, stored_u(x, dom, k, level, index)))
    return out


def stored_values(values, fmt):
    """The distinct stored u in T, each with the set of points x that give it."""
    out = {}
    for x, u in values:
        out.setdefault(to_t(u, fmt), set()).add(x)
    return out


def mapped(c_hat, r_hat, u, fmt):
    """y^ = fl(c^ + r^ u) in T; r^ u must be exact."""
    c_hat, r_hat = float(c_hat), float(r_hat)
    product = to_t(r_hat * u, fmt)
    assert product == r_hat * u, "r^ u_s is not exact"
    return to_t(c_hat + product, fmt)


class Stats:
    """What the checks of the lower bound record."""

    def __init__(self):
        self.failures = []  # the first few messages
        self.failure_count = 0
        self.pairs = 0
        self.zero_r2 = 0
        self.zero_distinct = 0  # r^2 = 0 although the points differ
        self.min_d = {f.name: [None, None] for f in FORMATS}  # [s != t, s = t]
        self.min_u = {f.name: None for f in FORMATS}
        self.min_y = {f.name: None for f in FORMATS}  # smallest nonzero |y^|, s != t
        self.finest = {f.name: 0 for f in FORMATS}  # largest grid exponent of y^ and d
        self.min_r2 = {f.name: None for f in FORMATS}
        self.max_r2 = {f.name: None for f in FORMATS}
        self.c_hats = {-1: set(), 0: set(), 1: set()}

    def fail(self, message):
        self.failure_count += 1
        if len(self.failures) < 10:
            self.failures.append(message)

    @staticmethod
    def lower(slot, key, value):
        if value and (slot[key] is None or abs(value) < slot[key]):
            slot[key] = abs(value)

    @staticmethod
    def upper(slot, key, value):
        if slot[key] is None or abs(value) > slot[key]:
            slot[key] = abs(value)


def check_grid(stats, label, fmt, name, value):
    """value in G_54 (the short argument) and in G_53 (the finer one)."""
    if value:
        k = grid_exponent(value)
        if k > SHARP_GRID:
            which = "G_53" if k <= MAPPED_GRID else "G_54"
            stats.fail(f"{label} {fmt.name}: {name} = {value!r} not in {which}")
        stats.finest[fmt.name] = max(stats.finest[fmt.name], k)


def check_component(stats, label, fmt, ut, y, self_pair):
    """The claims on one component, given u_t and y^ already checked: d = fl(u_t - y^)
    in G_54 and G_53, d = 0 exactly when u_t == y^, and |d| >= 2^-53 otherwise."""
    d = to_t(ut - y, fmt)
    if (d == 0) != (ut == y):
        stats.fail(f"{label} {fmt.name}: d = {d!r} but u_t == y^ is {ut == y}")
    if d:
        check_grid(stats, label, fmt, "d", d)
        if abs(d) < D_MIN_FLOAT:
            stats.fail(f"{label} {fmt.name}: |d| = {d!r} below 2^-53")
        Stats.lower(stats.min_d[fmt.name], int(self_pair), d)
    return d


def axis_case(cache, stats, dom, k, s, t, positions_t, positions_s, fmt):
    """The d of one component of the pair (s, t), for every distinct target and source
    value in the windows, in T: a list of (|d|, d, u_t, y^, whether more than one point
    gives these values), by |d|. Checks the grids and the per-component bounds."""
    key = (dom.name, k, s[0], s[1][k], t[0], t[1][k], tuple(positions_t), tuple(positions_s), fmt.name)
    if key in cache:
        return cache[key]
    self_pair = s == t
    c_hat, r_hat = axis_frame(k, s, t)
    stats.c_hats[s[0] - t[0]].add(c_hat)
    tv = stored_values(leaf_values(dom, k, t, positions_t), fmt)
    sv = tv if self_pair else stored_values(leaf_values(dom, k, s, positions_s), fmt)
    out = []
    label = f"{dom.name}, axis {k}, s = {s}, t = {t}, c^ = {c_hat}"
    for u in itertools.chain(tv, sv):
        if not in_grid(u, STORED_GRID):
            stats.fail(f"{label} {fmt.name}: stored u = {u!r} not in G_53")
        Stats.lower(stats.min_u, fmt.name, u)
    for us, xs in sv.items():
        # For s = t the stored chunk is passed unchanged: y^ = u_s, no arithmetic.
        y = us if self_pair else mapped(c_hat, r_hat, us, fmt)
        check_grid(stats, label, fmt, "y^", y)
        if not self_pair:
            Stats.lower(stats.min_y, fmt.name, y)
        for ut, xt in tv.items():
            d = check_component(stats, label, fmt, ut, y, self_pair)
            # Whether a coincidence here can come from distinct points.
            out.append((abs(d), d, ut, y, len(xt | xs) > 1))
    out.sort(key=lambda e: e[0])
    cache[key] = out
    return out


def r2_variants(d, fmt):
    """r^2 in T as the reference forms it, ((d0 d0 + d1 d1) + d2 d2), and as a kernel
    may: d0 d0 then two fmas, in either order of the components."""
    sq = [to_t(c * c, fmt) for c in d]
    plain = to_t(to_t(sq[0] + sq[1], fmt) + sq[2], fmt)
    forward = fma(d[2], d[2], fma(d[1], d[1], sq[0], fmt), fmt)
    backward = fma(d[0], d[0], fma(d[1], d[1], sq[2], fmt), fmt)
    return plain, forward, backward


def check_r2(stats, label, d, ut, y, fmt, r2_cache):
    """The claims on r^2 for one triple of components."""
    key = (fmt.name, d)
    if key not in r2_cache:
        r2_cache[key] = r2_variants(d, fmt)
    variants = r2_cache[key]
    reference_skips = all(ut[k] == y[k] for k in range(3))
    for r2 in variants:
        if (r2 == 0) != reference_skips:
            stats.fail(f"{label} {fmt.name}: r^2 = {r2!r} but the reference rule says {reference_skips}")
        if r2:
            if r2 < R2_SHARP_FLOAT or r2 < fmt.min_normal_float:
                stats.fail(f"{label} {fmt.name}: r^2 = {r2!r} below 2^-106")
            if r2 > R2_MAX_FLOAT:
                stats.fail(f"{label} {fmt.name}: r^2 = {r2!r} above 2^7")
            Stats.lower(stats.min_r2, fmt.name, r2)
            Stats.upper(stats.max_r2, fmt.name, r2)
    stats.zero_r2 += variants[0] == 0
    stats.pairs += 1


def locus(c_hat, r_hat):
    """Scaled positions, in the frame of t, where the closed leaves s and t meet along
    one axis: both ends, the midpoint, and t's centre (the cancellation c^ + r^ u ~ 0)
    and s's centre (u_s ~ 0) where they lie inside."""
    lo, hi = max(Fraction(-1), c_hat - r_hat), min(Fraction(1), c_hat + r_hat)
    points = {lo, hi, (lo + hi) / 2}
    for p in (Fraction(0), c_hat, Fraction(1, 2), Fraction(-1, 2)):
        if lo <= p <= hi:
            points.add(p)
    return sorted(points)


# ---------------------------------------------------------------------------------------
# Checks.


def check_rounding(rng):
    """The emulation of f32 and f64 operations against exact rounding of rationals."""
    failures = count = 0
    for _ in range(20000):
        a = round32(rng.uniform(-2, 2) * 2.0 ** rng.randint(-60, 4))
        b = round32(rng.uniform(-2, 2) * 2.0 ** rng.randint(-60, 4))
        for exact, emulated in ((Fraction(a) - Fraction(b), round32(a - b)), (Fraction(a) * Fraction(b), round32(a * b)), (Fraction(a) + Fraction(b), round32(a + b))):
            failures += round_exact(exact, F32) != Fraction(emulated)
            count += 1
        x, y = rng.uniform(-2, 2) * 2.0 ** rng.randint(-60, 4), rng.uniform(-2, 2) * 2.0 ** rng.randint(-60, 4)
        failures += round_exact(Fraction(x) - Fraction(y), F64) != Fraction(x - y)
        count += 1
    # Subnormal and boundary values of f32 against struct.pack.
    for k in range(-152, -120):
        for v in (2.0**k, 1.5 * 2.0**k, 2.0**k * (1 + 2.0**-30)):
            failures += round_exact(Fraction(v), F32) != Fraction(round32(v))
            count += 1
    return failures == 0, f"{count} operations and roundings, f32 (struct) and f64 (floats) against exact rounding"


def check_formula(rng):
    """u = fl(v - (2i + 1)) and its f32 cast lie in G_53 for every double v near the
    subtrahend, near it +- 1/2 and +- 1, and for tiny v."""
    failures = count = 0
    smallest = {f.name: None for f in FORMATS}
    odd = [1, 3, 5, 7, 2**16 - 1, 2**16 + 1, 2**17 - 1] + [2 * rng.randrange(2**16) + 1 for _ in range(20)]
    tiny = [0.0, 5e-324, 2.0**-1022, 2.0**-60, 2.0**-54, 0.25, math.nextafter(0.5, 0), 0.5]
    for m in odd:
        vs = set(tiny)
        for base in (m, m - 0.5, m + 0.5, m - 1, m + 1):
            v_lo = v_hi = float(base)
            vs.add(v_lo)
            for _ in range(64):
                v_lo, v_hi = math.nextafter(v_lo, -math.inf), math.nextafter(v_hi, math.inf)
                vs.update((v_lo, v_hi))
        for v in vs:
            u64 = v - m
            for fmt in FORMATS:
                u = to_t(u64, fmt)
                failures += not in_grid(u, STORED_GRID)
                if u:
                    failures += abs(Fraction(u)) < D_MIN
                    Stats.lower(smallest, fmt.name, u)
                count += 1
    detail = (
        f"{count} (v, 2i + 1, T) cases; all in G_53; smallest nonzero |u|:"
        f" f32 {p2(smallest['f32'])}, f64 {p2(smallest['f64'])} (bound 2^-53)"
    )
    return failures == 0, detail


def adversarial_stats(rng, doms):
    stats = Stats()
    cache, r2_cache = {}, {}
    skipped = 0
    self_positions = [Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3)]
    for dom in doms.values():
        for level in range(DEEPEST_LEVEL + 1):
            for t in targets(rng, level):
                for s in [t] + u_list(t):
                    for fmt in FORMATS:
                        per_axis = []
                        for k in range(3):
                            if s == t:
                                pos_t = pos_s = self_positions
                            else:
                                c_hat, r_hat = axis_frame(k, s, t)
                                pos_t = locus(c_hat, r_hat)
                                pos_s = [(p - c_hat) / r_hat for p in pos_t]
                            per_axis.append(axis_case(cache, stats, dom, k, s, t, pos_t, pos_s, fmt))
                        if not all(per_axis):
                            skipped += 1
                            continue
                        # Per component: a zero d if there is one, the smallest nonzero
                        # |d| and the largest.
                        choices = []
                        for entries in per_axis:
                            pick = [e for e in entries if e[0] == 0][:1]
                            nonzero = [e for e in entries if e[0] != 0]
                            pick += nonzero[:1] + nonzero[-1:]
                            choices.append(pick)
                        for combo in itertools.product(*choices):
                            d = tuple(e[1] for e in combo)
                            ut = [e[2] for e in combo]
                            y = [e[3] for e in combo]
                            check_r2(stats, dom.name, d, ut, y, fmt, r2_cache)
                            if d == (0.0, 0.0, 0.0):
                                stats.zero_distinct += any(e[4] for e in combo)
    return stats, skipped


def report_lower(stats, extra):
    ok = stats.failure_count == 0
    lines = list(stats.failures)
    if stats.failure_count > len(lines):
        lines.append(f"... {stats.failure_count - len(lines)} more failures")
    detail = (
        f"{stats.pairs} component triples; smallest nonzero |d|"
        f" s != t: f32 {p2(stats.min_d['f32'][0])}, f64 {p2(stats.min_d['f64'][0])};"
        f" s = t: f32 {p2(stats.min_d['f32'][1])}, f64 {p2(stats.min_d['f64'][1])}"
        f" (bound 2^-53); every u_t, y^ and d in G_{max(stats.finest.values())};"
        f" smallest nonzero r^2: f32 {p2(stats.min_r2['f32'])}, f64 {p2(stats.min_r2['f64'])}"
        f" (bound 2^-106); r^2 = 0 in {stats.zero_r2} triples, {stats.zero_distinct} of them"
        f" from distinct points; {extra}"
    )
    return ok, detail, lines


def check_adversarial(rng, doms):
    stats, skipped = adversarial_stats(rng, doms)
    c_hats = "; ".join(
        f"{name} {{{', '.join(str(c) for c in sorted(stats.c_hats[rel]))}}}"
        for rel, name in ((-1, "coarser"), (0, "same"), (1, "finer"))
    )
    extra = (
        f"smallest nonzero |y^| (cancellation) f32 {p2(stats.min_y['f32'])}, f64 {p2(stats.min_y['f64'])};"
        f" smallest nonzero |u| f32 {p2(stats.min_u['f32'])}, f64 {p2(stats.min_u['f64'])};"
        f" c^ components: {c_hats}; {skipped} pair cases without points on one axis"
    )
    ok, detail, lines = report_lower(stats, extra)
    expected = {-1: {-3, -1, 1, 3}, 0: {-2, 0, 2}, 1: {Fraction(-3, 2), Fraction(-1, 2), Fraction(1, 2), Fraction(3, 2)}}
    if any(stats.c_hats[rel] != set(map(Fraction, want)) for rel, want in expected.items()):
        ok = False
        lines.append(f"c^ components {stats.c_hats} differ from the 2:1 set")
    return ok, detail, lines, stats


def random_point(rng, dom, key):
    """A double uniform in the leaf `key` that points_to_morton puts into it."""
    level, index = key
    while True:
        x = [float(centre_exact(dom, k, level, index[k]) + Fraction(rng.uniform(-1, 1)) * Fraction(dom.w) / 2 ** (level + 1)) for k in range(3)]
        if all(dom.lo[k] < x[k] < dom.hi[k] for k in range(3)) and lg.leaf_index(x, dom.lo, dom.hi, level) == index:
            return x


def check_random(rng, doms):
    stats = Stats()
    r2_cache = {}
    for name in ("dyadic", "generic", "far from the origin"):
        dom = doms[name]
        for _ in range(500):
            level = rng.randint(0, DEEPEST_LEVEL)
            t = (level, [rng.randrange(2**level) for _ in range(3)])
            neighbours = u_list(t)
            s = t if not neighbours or rng.random() < 0.25 else rng.choice(neighbours)
            xt = [random_point(rng, dom, t) for _ in range(6)]
            xs = [random_point(rng, dom, s) for _ in range(6)]
            if s == t:
                xs[:2] = xt[:2]  # duplicated points: coincident pairs
            c_hat, r_hat, _ = lg.relative_frame(s, t)
            for fmt in FORMATS:
                ut = [[to_t(u, fmt) for u in lg.leaf_scaled_f64(x, dom.lo, dom.w, t[1], t[0])] for x in xt]
                us = [[to_t(u, fmt) for u in lg.leaf_scaled_f64(x, dom.lo, dom.w, s[1], s[0])] for x in xs]
                ys = us if s == t else [[mapped(c_hat[k], r_hat, u[k], fmt) for k in range(3)] for u in us]
                label = f"{name}, s = {s}, t = {t}"
                for c in itertools.chain(*ut, *ys):
                    check_grid(stats, label, fmt, "u_t or y^", c)
                for (u, x_t), (y, x_s) in itertools.product(zip(ut, xt), zip(ys, xs)):
                    d = tuple(check_component(stats, label, fmt, u[k], y[k], s == t) for k in range(3))
                    check_r2(stats, name, d, u, y, fmt, r2_cache)
                    if d == (0.0, 0.0, 0.0):
                        stats.zero_distinct += x_t != x_s
    ok, detail, lines = report_lower(stats, f"largest r^2: f32 {float(stats.max_r2['f32']):.3f}, f64 {float(stats.max_r2['f64']):.3f}")
    # The duplicated points of the self pairs must have occurred, and been skipped.
    if stats.zero_r2 == 0:
        ok = False
        lines.append("no coincident pair was generated")
    return ok, detail, lines


def check_upper(rng, doms):
    """Points within a few ulps of the far faces of corner neighbours (coarser, same,
    finer), on every level and in every domain: the largest |d| per axis and r^2."""
    failures = []
    largest = {f.name: Fraction(0) for f in FORMATS}
    largest_b = Fraction(0)
    count = 0
    for dom in doms.values():
        for level in range(1, DEEPEST_LEVEL + 1):
            for t in targets(rng, level):
                for s in u_list(t):
                    c_hat, r_hat, _ = lg.relative_frame(s, t)
                    if not all(abs(c) == 1 + r_hat for c in c_hat):
                        continue  # not a corner neighbour
                    for fmt in FORMATS:
                        d, b = [], Fraction(0)
                        for k in range(3):
                            tv = [to_t(u, fmt) for _, u in leaf_values(dom, k, t, [Fraction(1), Fraction(-1)])]
                            sv = [to_t(u, fmt) for _, u in leaf_values(dom, k, s, [Fraction(1), Fraction(-1)])]
                            b = max([b] + [abs(Fraction(u)) - 1 for u in tv + sv])
                            ys = [mapped(c_hat[k], r_hat, u, fmt) for u in sv]
                            d.append(max((to_t(u - y, fmt) for u in tv for y in ys), key=abs))
                        largest_b = max(largest_b, b)
                        bound = 3 * (6 + 3 * b) ** 2 * (1 + fmt.u) ** 7
                        for r2 in r2_variants(d, fmt):
                            if Fraction(r2) > bound or Fraction(r2) > R2_MAX or b > B_MAX:
                                failures.append(f"{dom.name} {fmt.name}: r^2 = {r2!r}, bound {float(bound)}, b = {float(b)}")
                            largest[fmt.name] = max(largest[fmt.name], Fraction(r2))
                        count += 1
    ok = not failures
    detail = (
        f"{count} corner pairs, levels 1-16, 4 domains; largest r^2: f32 {float(largest['f32']):.7f},"
        f" f64 {float(largest['f64']):.7f} (108 for |u| = 1; bound 3 (6 + 3b)^2 (1 + u)^7 with"
        f" b = |u| - 1 up to {float(largest_b):.2e}; 2^7 = 128)"
    )
    return ok, detail, failures[:10], largest


def f32_gradient_intermediates(r2, q=1.0):
    """For an f32 r^2: the reference's fl(r^2 fl(sqrt r^2)), and a fast kernel's
    fl(fl(q rho) fl(rho rho)) with rho the correctly rounded 1/sqrt r^2 (inf on
    overflow)."""
    r = round32(math.sqrt(r2))  # correctly rounded: 53 >= 2 * 24 + 2
    r3 = round32(r2 * r)
    rho = float(round_exact(1 / Fraction(r), F32))
    try:
        fast = float(round_exact(Fraction(round32(q * rho)) * Fraction(round32(rho * rho)), F32))
    except OverflowError:
        fast = math.inf
    return r3, fast


def check_gradient_range(rng, stats):
    """f32 with gradients: for r^2 in [2^-84, 2^7] and |q| <= 1 the reference's r^2 r is
    normal and a fast kernel's (q rho) rho^2 finite; at the smallest r^2 found, neither."""
    failures = []
    low, high = log2(GRADIENT_R2_MIN), log2(R2_MAX)
    samples = [2.0**k for k in range(int(low), int(high) + 1)]
    samples += [round32(2.0 ** rng.uniform(low, high)) for _ in range(2000)]
    for r2 in samples:
        r3, fast = f32_gradient_intermediates(max(r2, float(GRADIENT_R2_MIN)))
        if Fraction(r3) < F32.min_normal or not math.isfinite(fast):
            failures.append(f"r^2 = {r2!r}: r^2 r = {r3!r}, (q rho) rho^2 = {fast!r}")
    smallest = float(stats.min_r2["f32"])
    r3_min, fast_min = f32_gradient_intermediates(smallest)
    below_r3, below_fast = f32_gradient_intermediates(2.0**-86)
    detail = (
        f"{len(samples)} r^2 in [2^-84, 2^7]: r^2 r normal, (q rho) rho^2 finite;"
        f" at 2^-86: r^2 r = {p2(below_r3)} (subnormal), (q rho) rho^2 = {below_fast};"
        f" at the smallest r^2 found, {p2(smallest)}: r^2 r = {r3_min}, (q rho) rho^2 = {fast_min}"
    )
    ok = not failures and Fraction(below_r3) < F32.min_normal and math.isinf(below_fast)
    return ok, detail, failures[:10]


def main():
    if mpmath.__version__ != gh.MPMATH_VERSION:
        sys.exit(f"needs mpmath {gh.MPMATH_VERSION}, found {mpmath.__version__}")
    rng = random.Random(SEED)
    doms = domains(rng)
    all_ok = True

    def show(ok, name, detail, lines=()):
        nonlocal all_ok
        all_ok &= ok
        print(f"{'ok' if ok else 'FAIL':4}  {name:44} {detail}")
        for line in lines:
            print(f"      {line}")
        sys.stdout.flush()

    ok, detail = check_rounding(rng)
    show(ok, "f32 and f64 emulation = exact rounding", detail)
    ok, detail = check_formula(rng)
    show(ok, "stored u in G_53 for every double v", detail)
    ok, detail, lines, adversarial = check_adversarial(rng, doms)
    show(ok, "adversarial pairs: d, r^2 bounds, r^2 = 0 rule", detail, lines)
    ok, detail, lines = check_random(rng, doms)
    show(ok, "random leaf pairs: d, r^2 bounds, r^2 = 0 rule", detail, lines)
    ok, detail, lines, largest = check_upper(rng, doms)
    show(ok, "upper bound r^2 < 2^7 at corner neighbours", detail, lines)
    ok, detail, lines = check_gradient_range(rng, adversarial)
    show(ok, "f32 gradient intermediates for r^2 >= 2^-84", detail, lines)

    print()
    for fmt in FORMATS:
        smallest = adversarial.min_r2[fmt.name]
        print(
            f"{fmt.name}: smallest nonzero r^2 found {p2(smallest)} (sharp bound 2^-106); domain lower"
            f" end 2^-108 (margin {p2(Fraction(smallest) / R2_MIN)}), smallest normal {p2(fmt.min_normal)}"
            f" (margin {p2(R2_MIN / fmt.min_normal)}); largest r^2 {float(largest[fmt.name]):.7f},"
            f" domain upper end 2^7 (margin {float(R2_MAX / largest[fmt.name]):.3f})"
        )
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
