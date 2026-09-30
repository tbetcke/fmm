#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""High-precision check of the box geometry and cube symmetry rules of CONVENTIONS §3.12.

Checks, in mpmath at 40 significant digits, with seeded random points and coefficients
and degrees n <= 8, using the harmonics of gen_harmonics.py and the translation operators
of check_translations.py (the §3.11 forms, imported, not copied):
  - offsets: the 316 V-list offsets in lexicographic (x, y, z) order, none in {-1..1}^3,
    and the closed-form table index of §3.12 round-trips;
  - geometry: box centres and half-widths from the Morton index (as
    nd_octree::morton::physical_box computes them), the child octant index
    o = 4x + 2y + z from the bits of a Morton key (as morton::child_index reads them),
    the child centre c + r_child s_o and the M2L shift 2 r_l d, all in exact rationals;
  - group: the 48 signed permutations in the enumeration order of §3.12, 24 proper,
    closed under composition, with the transpose as inverse;
  - classes: 16 orbits of the offsets under O_h (34 under the z-axis elements), each
    with exactly one representative 0 <= d_x <= d_y <= d_z, and P . representative = d
    for the element P chosen by the rule of §3.12;
  - octants: every P permutes the eight s_o, and element o maps s_0 to s_o;
  - improper rule: R_n(Px) = D^n(P) R_n(x) for all 48 P, with D^n(P) = (-1)^n D^n(-P)
    and D^n(-P) fitted by check_translations.fitted_blocks; the irregular counterpart
    with S D^n S^-1; the homomorphism of §3.8 on sampled pairs; T(P) T(P^T) = I;
  - operator identities, all through the general forms of check_translations.py:
    M2L(P d) = T_L(P) M2L(d) T_M(P^T) for all 316 offsets from their representatives;
    M2M(P s) = T_M(P) M2M(s) T_M(P^T) and L2L(P s) = T_L(P) L2L(s) T_L(P^T) for all 48 P
    from octant 0; inversion M2L(-d) = diag((-1)^j) M2L(d) diag((-1)^n) for all 316 d;
  - z-axis elements: T_M(P) and T_L(P) for the 16 elements that map the z-axis to
    itself or its negative are equal, signed permutations of the (+m, -m) slot pairs,
    diagonal when P fixes the x- and y-axes up to sign; the structure is printed.

Coefficient vectors are compared per degree in the orthonormal weighting of §3.8
(N_m for multipoles and regular harmonics, N_m / S_m for locals and irregular ones),
relative to the weighted norm of the reference in the same degree.

Run from the repository root:  uv run tools/fixtures/check_symmetry.py
(or: python3 tools/fixtures/check_symmetry.py with mpmath==1.4.1 installed).
Prints the worst error per check and exits non-zero on any failure.
"""
import itertools
import random
import sys
from fractions import Fraction
from math import factorial
from pathlib import Path

# Import the sibling scripts without leaving a __pycache__ in the repository.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_translations as ct  # noqa: E402  (the operators of §3.11)
import gen_harmonics as gh  # noqa: E402  (the definitions of §3.3)
import mpmath  # noqa: E402
from mpmath import mp, mpf  # noqa: E402

DPS = 40
TOL = mpf(10) ** -30
SEED = 20261003
P_MAX = 8
DEEPEST_LEVEL = 16


# ---------------------------------------------------------------------------------------
# Offsets (§3.12): d = index(target) - index(source), d in {-3..3}^3 \ {-1..1}^3.


def v_list_offsets():
    return [d for d in itertools.product(range(-3, 4), repeat=3) if max(abs(c) for c in d) >= 2]


def kappa(t):
    """Number of t' in {-1, 0, 1} with t' < t."""
    return min(max(t + 1, 0), 3)


def offset_index(d):
    """Closed-form position of d in the lexicographic order of the 316 offsets (§3.12)."""
    x, y, z = d
    index = 49 * (x + 3) + 7 * (y + 3) + (z + 3) - 9 * kappa(x)
    if abs(x) <= 1:
        index -= 3 * kappa(y)
        if abs(y) <= 1:
            index -= kappa(z)
    return index


def check_offsets(rng):
    offsets = v_list_offsets()
    ok = len(offsets) == 316
    ok &= offsets == sorted(offsets) and len(set(offsets)) == 316
    ok &= all(max(abs(c) for c in d) >= 2 for d in offsets)
    ok &= all(offset_index(d) == k for k, d in enumerate(offsets))
    # Every vector of {-3..3}^3 that is not an offset lies in {-1..1}^3.
    others = [d for d in itertools.product(range(-3, 4), repeat=3) if d not in set(offsets)]
    ok &= len(others) == 27 and all(max(abs(c) for c in d) <= 1 for d in others)
    return (mpf(0) if ok else mpf(1)), mpf(0), "316 offsets, lexicographic, closed-form index round-trips"


# ---------------------------------------------------------------------------------------
# Box geometry (§3.12) in exact rationals, with the Morton bit layout of nd_octree:
# per level a triple of bits (x << 2) | (y << 1) | z, the root level most significant.


def morton_bits(index, level):
    """Interleaved index bits of a key on `level`, without the level field."""
    key = 0
    for bit in range(level):
        triple = (((index[0] >> bit) & 1) << 2) | (((index[1] >> bit) & 1) << 1) | ((index[2] >> bit) & 1)
        key |= triple << (3 * (bit + DEEPEST_LEVEL - level))
    return key


def child_index(index, level):
    """morton::child_index: the bit triple of the key's own level."""
    return (morton_bits(index, level) >> (3 * (DEEPEST_LEVEL - level))) % 8


def physical_box(a, w, index, level):
    """morton::physical_box: [a + i w / 2^l, a + (i + 1) w / 2^l] per axis."""
    size = w / 2**level
    return [a[k] + index[k] * size for k in range(3)], [a[k] + (index[k] + 1) * size for k in range(3)]


def centre(a, w, index, level):
    return [a[k] + (index[k] + Fraction(1, 2)) * w / 2**level for k in range(3)]


def half_width(w, level):
    return w / 2 ** (level + 1)


def octant_sign(o):
    """s_o = (2x - 1, 2y - 1, 2z - 1) for o = 4x + 2y + z."""
    return ((o >> 2 & 1) * 2 - 1, (o >> 1 & 1) * 2 - 1, (o & 1) * 2 - 1)


def check_geometry(rng):
    failures = 0
    for _ in range(300):
        a = [Fraction(rng.randint(-10**6, 10**6), rng.randint(1, 10**3)) for _ in range(3)]
        w = Fraction(rng.randint(1, 10**6), rng.randint(1, 10**3))
        level = rng.randint(0, DEEPEST_LEVEL - 1)
        index = [rng.randrange(2**level) for _ in range(3)]
        lo, hi = physical_box(a, w, index, level)
        c, r = centre(a, w, index, level), half_width(w, level)
        failures += any(c[k] != (lo[k] + hi[k]) / 2 or hi[k] - lo[k] != 2 * r for k in range(3))
        # Children: index 2 idx + (x, y, z), child index o = 4x + 2y + z from the key bits,
        # centre c + r_child s_o.
        for bits in itertools.product((0, 1), repeat=3):
            child = [2 * index[k] + bits[k] for k in range(3)]
            o = child_index(child, level + 1)
            failures += o != 4 * bits[0] + 2 * bits[1] + bits[2]
            failures += [child[k] & 1 for k in range(3)] != list(bits)
            s, rc = octant_sign(o), half_width(w, level + 1)
            failures += rc != r / 2
            failures += centre(a, w, child, level + 1) != [c[k] + rc * s[k] for k in range(3)]
        # V-list shift: c_target - c_source = 2 r_l d with d = index(target) - index(source).
        if level >= 2:
            d = rng.choice(v_list_offsets())
            target = [index[k] + d[k] for k in range(3)]
            if all(0 <= t < 2**level for t in target):
                ct_ = centre(a, w, target, level)
                failures += [ct_[k] - c[k] for k in range(3)] != [2 * r * d[k] for k in range(3)]
    return mpf(failures), mpf(0), "300 random boxes on levels 0-15 of random domains, exact rationals"


# ---------------------------------------------------------------------------------------
# The cube group O_h (§3.12). Element g = 8 k + 4 [s_x < 0] + 2 [s_y < 0] + [s_z < 0],
# with k the position of the permutation pi in the lexicographic order of the six
# permutations of (0, 1, 2); (P v)_a = s_a v_(pi(a)).

PERMUTATIONS = list(itertools.permutations(range(3)))


def element(g):
    pi = PERMUTATIONS[g // 8]
    signs = tuple(-1 if (g >> (2 - a)) & 1 else 1 for a in range(3))
    return pi, signs


def matrix(g):
    pi, signs = element(g)
    return [[signs[a] if b == pi[a] else 0 for b in range(3)] for a in range(3)]


def apply(g, v):
    pi, signs = element(g)
    return tuple(signs[a] * v[pi[a]] for a in range(3))


def index_of(P):
    return next(g for g in range(48) if matrix(g) == P)


def matmul3(A, B):
    return [[sum(A[a][k] * B[k][b] for k in range(3)) for b in range(3)] for a in range(3)]


def det3(A):
    return (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )


def compose_table():
    return [[index_of(matmul3(matrix(g), matrix(h))) for h in range(48)] for g in range(48)]


def transpose_index(g):
    return index_of(ct.transpose(matrix(g)))


def negate_index(g):
    return index_of([[-e for e in row] for row in matrix(g)])


Z_AXIS = [g for g in range(48) if element(g)[0][2] == 2]


def check_group(rng):
    mats = [matrix(g) for g in range(48)]
    ok = len({tuple(map(tuple, M)) for M in mats}) == 48 and matrix(0) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ok &= sum(det3(M) == 1 for M in mats) == 24
    table = compose_table()  # index_of raises if a product left the group
    ok &= all(table[g][transpose_index(g)] == 0 for g in range(48))
    ok &= all(apply(g, apply(h, v)) == apply(table[g][h], v) for g in range(48) for h in range(48) for v in [(1, 2, 3)])
    ok &= len(Z_AXIS) == 16 and all(abs(apply(g, (0, 0, 1))[2]) == 1 for g in Z_AXIS)
    return (mpf(0) if ok else mpf(1)), mpf(0), "48 elements, 24 proper, closed, P^-1 = P^T, 16 z-axis elements"


# ---------------------------------------------------------------------------------------
# Symmetry classes (§3.12).


def is_representative(d):
    return 0 <= d[0] <= d[1] <= d[2]


def representative(d):
    return tuple(sorted(abs(c) for c in d))


def class_element(d):
    """The first g in the enumeration order with P_g . representative(d) = d."""
    rep = representative(d)
    return next(g for g in range(48) if apply(g, rep) == d)


def orbits(group, points):
    seen, out = set(), []
    for d in points:
        if d not in seen:
            orbit = {apply(g, d) for g in group}
            seen |= orbit
            out.append(orbit)
    return out


def check_classes(rng):
    offsets = v_list_offsets()
    full, zaxis = orbits(range(48), offsets), orbits(Z_AXIS, offsets)
    ok = len(full) == 16 and len(zaxis) == 34
    ok &= sum(len(o) for o in full) == 316 and all(o <= set(offsets) for o in full)
    ok &= all(sum(is_representative(d) for d in o) == 1 for o in full)
    ok &= all(apply(class_element(d), representative(d)) == d for d in offsets)
    reps = sorted({representative(d) for d in offsets})
    sizes = [len(next(o for o in full if r in o)) for r in reps]
    detail = f"16 classes (sizes {sizes}), 34 under the z-axis elements"
    return (mpf(0) if ok else mpf(1)), mpf(0), detail


def check_octants(rng):
    signs = [octant_sign(o) for o in range(8)]
    ok = all(sorted(apply(g, s) for s in signs) == sorted(signs) for g in range(48))
    # Element o is the first element mapping s_0 to s_o (it is diag(-s_o)).
    ok &= all(next(g for g in range(48) if apply(g, signs[0]) == signs[o]) == o for o in range(8))
    return (mpf(0) if ok else mpf(1)), mpf(0), "every P permutes the s_o; element o maps s_0 to s_o"


# ---------------------------------------------------------------------------------------
# D^n(P) for all of O_h: fitted for the 24 proper elements, (-1)^n D^n(-P) for the
# improper ones (§3.12).

BLOCKS = {}


def build_blocks(rng):
    fit = mpf(0)
    for g in range(48):
        if det3(matrix(g)) == 1:
            BLOCKS[g], err = ct.fitted_blocks(rng, P_MAX, [[mpf(e) for e in row] for row in matrix(g)])
            fit = max(fit, err)
    for g in range(48):
        if det3(matrix(g)) == -1:
            BLOCKS[g] = [(-1) ** n * D for n, D in enumerate(BLOCKS[negate_index(g)])]
    return fit


def W(n, local):
    """Orthonormal weighting of §3.8 per real slot of degree n: N_m (multipoles and R_n) or
    N_m / S_m (locals and I_n), with N_m = sqrt((n - |m|)! (n + |m|)!) c_m."""
    diag = []
    for m in range(-n, n + 1):
        f = mpmath.sqrt(factorial(n - abs(m)) * factorial(n + abs(m)))
        c = 1 if m == 0 else mpmath.sqrt(2)
        diag.append(c * f if not local else c / f)
    return mpmath.diag(diag)


def block_error(got, want, n, local):
    Wn = W(n, local)
    s = mpmath.mnorm(Wn * want, "f")
    e = mpmath.mnorm(Wn * (got - want), "f")
    return e / s if s > 0 else (mpf(0) if e == 0 else mpf("inf"))


def TM(g, n):
    return ct.K(n) * BLOCKS[g][n] * ct.K(n)


def TL(g, n):
    return ct.K(n) * ct.S(n) * BLOCKS[g][n] * mpmath.inverse(ct.S(n)) * ct.K(n)


def check_improper(rng):
    fit = build_blocks(rng)
    worst_r = worst_i = mpf(0)
    for g in range(48):
        P = [[mpf(e) for e in row] for row in matrix(g)]
        for _ in range(3):
            x = ct.unit_vector(rng)
            x = [c * (mpf("0.2") + mpf("1.5") * mpf(rng.random())) for c in x]
            R, RP = ct.harmonics(P_MAX, x, True), ct.harmonics(P_MAX, ct.matvec(P, x), True)
            y = [c * (2 + 6 * mpf(rng.random())) for c in ct.unit_vector(rng)]
            I, IP = ct.harmonics(P_MAX, y, False), ct.harmonics(P_MAX, ct.matvec(P, y), False)
            for n in range(P_MAX + 1):
                D = BLOCKS[g][n]
                worst_r = max(worst_r, block_error(D * ct.real_block(R, n), ct.real_block(RP, n), n, False))
                SDS = ct.S(n) * D * mpmath.inverse(ct.S(n))
                worst_i = max(worst_i, block_error(SDS * ct.real_block(I, n), ct.real_block(IP, n), n, True))
    worst = max(worst_r, worst_i)
    detail = (
        f"n <= {P_MAX}, 48 P x 3 points: fit {mpmath.nstr(fit, 3)}, regular {mpmath.nstr(worst_r, 3)}, "
        f"irregular {mpmath.nstr(worst_i, 3)}"
    )
    return worst, TOL, detail


def matrix_error(A, B):
    """max |A - B| / max(1, max |B|), entrywise."""
    big = max([mpf(1)] + [abs(B[a, b]) for a in range(B.rows) for b in range(B.cols)])
    return max(abs(A[a, b] - B[a, b]) for a in range(A.rows) for b in range(A.cols)) / big


def check_homomorphism(rng):
    table = compose_table()
    worst_hom = worst_inv = mpf(0)
    pairs = [(g, rng.randrange(48)) for g in range(48) for _ in range(2)]
    for g, h in pairs:
        for n in range(P_MAX + 1):
            worst_hom = max(worst_hom, matrix_error(BLOCKS[table[g][h]][n], BLOCKS[g][n] * BLOCKS[h][n]))
    for g in range(48):
        gt = transpose_index(g)
        for n in range(P_MAX + 1):
            eye = mpmath.eye(2 * n + 1)
            worst_inv = max(worst_inv, matrix_error(TM(g, n) * TM(gt, n), eye))
            worst_inv = max(worst_inv, matrix_error(TL(g, n) * TL(gt, n), eye))
    # Inversion: D^n(-I) = (-1)^n I.
    minus_i = negate_index(0)
    for n in range(P_MAX + 1):
        worst_inv = max(worst_inv, matrix_error(BLOCKS[minus_i][n], (-1) ** n * mpmath.eye(2 * n + 1)))
    worst = max(worst_hom, worst_inv)
    detail = (
        f"n <= {P_MAX}: D(PQ) = D(P) D(Q) on {len(pairs)} pairs {mpmath.nstr(worst_hom, 3)}; "
        f"T(P) T(P^T) = I for all 48 and D(-I) = (-1)^n {mpmath.nstr(worst_inv, 3)}"
    )
    return worst, TOL, detail


# ---------------------------------------------------------------------------------------
# Operator identities (§3.12), through the general forms of check_translations.py at the
# canonical frames of §3.12.

ORIGIN = ([mpf(0)] * 3, mpf(1))


def transform(g, v, local, p=P_MAX):
    """T_M(P_g) v or T_L(P_g) v for a coefficient vector v (m >= 0 complex list)."""
    return ct.rotate(p, v, BLOCKS[g], local)


def m2l_frame(d):
    return ([mpf(2 * c) for c in d], mpf(1))


def child_frame(o):
    return ([mpf(c) / 2 for c in octant_sign(o)], mpf(1) / 2)


def m2l(d, v):
    return ct.m2l(P_MAX, P_MAX, ORIGIN, m2l_frame(d), v)


def check_m2l(rng):
    worst = mpf(0)
    for d in v_list_offsets():
        g, rep = class_element(d), representative(d)
        x = ct.random_coefficients(rng, P_MAX)
        got = transform(g, m2l(rep, transform(transpose_index(g), x, False)), True)
        worst = max(worst, ct.degree_error(P_MAX, got, m2l(d, x), local=True))
    return worst, TOL, f"p = {P_MAX}, all 316 offsets from their representatives"


def parity(v, p=P_MAX):
    return [(-1) ** n * v[gh.tri(n, m)] for n in range(p + 1) for m in range(n + 1)]


def check_inversion(rng):
    worst = mpf(0)
    for d in v_list_offsets():
        x = ct.random_coefficients(rng, P_MAX)
        got = parity(m2l(d, parity(x)))
        worst = max(worst, ct.degree_error(P_MAX, got, m2l(tuple(-c for c in d), x), local=True))
    return worst, TOL, f"p = {P_MAX}, M2L(-d) = diag((-1)^j) M2L(d) diag((-1)^n), all 316 offsets"


def check_octant_operators(rng):
    worst_m = worst_l = mpf(0)
    signs = [octant_sign(o) for o in range(8)]
    for g in range(48):
        o = signs.index(apply(g, signs[0]))
        gt = transpose_index(g)
        x = ct.random_coefficients(rng, P_MAX)
        got = transform(g, ct.m2m(P_MAX, child_frame(0), ORIGIN, transform(gt, x, False)), False)
        worst_m = max(worst_m, ct.degree_error(P_MAX, got, ct.m2m(P_MAX, child_frame(o), ORIGIN, x), False))
        got = transform(g, ct.l2l(P_MAX, ORIGIN, child_frame(0), transform(gt, x, True)), True)
        worst_l = max(worst_l, ct.degree_error(P_MAX, got, ct.l2l(P_MAX, ORIGIN, child_frame(o), x), True))
    detail = f"p = {P_MAX}, all 48 P from octant 0: M2M {mpmath.nstr(worst_m, 3)}, L2L {mpmath.nstr(worst_l, 3)}"
    return max(worst_m, worst_l), TOL, detail


# ---------------------------------------------------------------------------------------
# The z-axis elements (§3.12).


def pair_structure(T, n):
    """'diag' or 'swap' for each pair (m, -m), m >= 1, if T is a signed permutation that
    maps every slot pair {m, -m} (and slot 0) to itself, with entries in {0, +-1} to TOL;
    None otherwise."""
    size = 2 * n + 1
    for a in range(size):
        for b in range(size):
            e = T[a, b]
            if min(abs(e), abs(abs(e) - 1)) > TOL:
                return None
            if abs(e) > TOL and abs(a - n) != abs(b - n):
                return None
    if abs(abs(T[n, n]) - 1) > TOL:
        return None
    out = []
    for m in range(1, n + 1):
        p_, q_ = n + m, n - m
        if abs(T[p_, q_]) < TOL and abs(T[q_, p_]) < TOL:
            out.append("diag")
        elif abs(T[p_, p_]) < TOL and abs(T[q_, q_]) < TOL:
            out.append("swap")
        else:
            return None
    return out


def check_z_axis(rng):
    ok, worst, lines = True, mpf(0), []
    for g in Z_AXIS:
        pi = element(g)[0]
        swaps_xy = pi[:2] == (1, 0)
        pattern = []
        for n in range(P_MAX + 1):
            tm, tl = TM(g, n), TL(g, n)
            worst = max(worst, matrix_error(tl, tm))
            s = pair_structure(tm, n)
            if s is None:
                ok = False
                pattern.append("?")
                continue
            # Expected: diagonal unless P exchanges x and y; then odd m swap, even m diagonal.
            ok &= s == ["swap" if swaps_xy and m % 2 else "diag" for m in range(1, n + 1)]
            pattern.append("".join("s" if e == "swap" else "d" for e in s) or "-")
        lines.append(f"      g = {g:2}  P = {matrix(g)}  pairs m = 1..n per degree: {' '.join(pattern)}")
    detail = f"16 elements, n <= {P_MAX}: T_L = T_M ({mpmath.nstr(worst, 3)}), signed permutations of slot pairs"
    return (worst if ok else mpf(1)), TOL, detail + "\n" + "\n".join(lines)


def main():
    if mpmath.__version__ != gh.MPMATH_VERSION:
        sys.exit(f"needs mpmath {gh.MPMATH_VERSION}, found {mpmath.__version__}")
    mp.dps = DPS
    rng = random.Random(SEED)
    checks = [
        ("offsets: order, range, closed-form index", check_offsets),
        ("geometry: centres, octants, shift", check_geometry),
        ("group: enumeration, closure, inverse", check_group),
        ("classes: orbits, representatives, rule", check_classes),
        ("octants: permuted by O_h", check_octants),
        ("improper rule for R_n and I_n", check_improper),
        ("homomorphism and T(P) T(P^T) = I", check_homomorphism),
        ("M2L(P d) = T_L M2L(d) T_M^-1", check_m2l),
        ("M2M, L2L under O_h, all octants", check_octant_operators),
        ("inversion M2L(-d)", check_inversion),
        ("z-axis elements: structure of T_M, T_L", check_z_axis),
    ]
    ok = True
    for name, check in checks:
        err, tol, detail = check(rng)
        passed = err <= tol
        ok &= passed
        print(f"{'ok' if passed else 'FAIL':4}  {name:42} {mpmath.nstr(err, 3):>10}  (tol {mpmath.nstr(tol, 1)})  {detail}")
        sys.stdout.flush()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
