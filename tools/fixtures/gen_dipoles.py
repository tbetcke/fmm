#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""High-precision fixtures for the dipole operators of docs/CONVENTIONS.md §3.7.

Writes fmm-ref/fixtures/dipoles.json (Phase 6 T8, C6.4): for point dipoles of moment d
at y, whose potential is d . grad_y 1/|x - y| = d . (x - y)/|x - y|^3 (§3.7, "Dipole
sources"),
  - "p2m": the scaled multipole coefficients sum_j (d_j / r) . grad conj R_n^m(u_j),
    u_j = (y_j - c)/r, in three frames, sources inside the box's sphere |u| <= sqrt(3);
  - "p2l": the scaled local coefficients sum_j (d_j / r) . grad conj I_n^m(u_j), the
    same frames, sources at 2 <= |u| <= 8 (§3.9);
  - "direct": the potential and gradient of the dipoles at targets in and near the
    cube of the sources, against which nd_fmm_ref::p2p::direct_sum_dipoles and the
    dipole P2P are tested.

The coefficients come from the definitions of §3.3 (gen_harmonics.harmonics), with the
directional derivative along d_j / r taken by central differences at elevated precision,
not from the gradient ladder that the Rust code uses; the direct sum from the closed
form, its gradient checked against central differences of the potential. Every value is
verified to DIGITS significant digits (relative to the largest value of its degree, or of
the target's sum of term magnitudes) by a second step size or a second precision, then
rounded to the nearest double and written with 17 significant digits. The inputs (frame,
points, moments) are doubles, and the values are those of exactly these doubles.

Run from the repository root:  uv run tools/fixtures/gen_dipoles.py
(or: python3 tools/fixtures/gen_dipoles.py with mpmath==1.4.1 installed).
Regeneration is byte-identical; --check compares with the committed file. See
tools/fixtures/README.md.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gen_harmonics as gh  # noqa: E402  (the definitions of §3.3, the pinned mpmath)
import mpmath  # noqa: E402
from mpmath import mp, mpf  # noqa: E402

# Bump when the generator changes its output (format, points or algorithm).
GENERATOR_VERSION = 1
DIGITS = 40
SEED = 20261017
P = 16
SOURCES = 5
DIRECT_SOURCES = 20
DIRECT_TARGETS = 10

OUT = gh.ROOT / "fmm-ref" / "fixtures" / "dipoles.json"

# Frames (centre, radius): the unit frame, a generic one, and a deep leaf far from the
# origin, as in the leaf tests of nd-fmm-ref. Every input is a double.
FRAMES = [
    ([0.0, 0.0, 0.0], 1.0),
    ([0.3125, -0.21875, 0.6875], 0.37),
    ([1.0, 1.0, 1.0], 2.0**-16),
]

VALUE_DPS = 60
DIFF_DPS = 120
STEP, CHECK_STEP = mpf(10) ** -35, mpf(10) ** -32


def directional(p, u, e, regular, h):
    """(X(u + h e) - X(u - h e)) / (2h), m >= 0, complex: e . grad X_n^m at u."""
    with mp.workdps(DIFF_DPS):
        plus = [ui + h * ei for ui, ei in zip(u, e)]
        minus = [ui - h * ei for ui, ei in zip(u, e)]
        coef = coefficients(p)
        a = gh.harmonics(coef, p, plus, regular)
        b = gh.harmonics(coef, p, minus, regular)
        return [(x - y) / (2 * h) for x, y in zip(a, b)]


_coef = {}


def coefficients(p):
    if p not in _coef:
        _coef[p] = gh.legendre_derivative_coefficients(p)
    return _coef[p]


def check_degrees(p, a, b, what):
    """a and b (complex, m >= 0) agree to DIGITS per degree."""
    gh.check_close(p, a, b, what, DIGITS)


def expansion(p, frame, sources, dipoles, regular):
    """sum_j conj(d_j / r . grad X(u_j)) in real storage, verified by two step sizes."""
    c, r = [mpf(v) for v in frame[0]], mpf(frame[1])
    total = [mpf(0)] * gh.tri(p + 1, 0)
    total_check = list(total)
    with mp.workdps(DIFF_DPS):
        for y, d in zip(sources, dipoles):
            u = [(mpf(yi) - ci) / r for yi, ci in zip(y, c)]
            e = [mpf(di) / r for di in d]
            g = directional(p, u, e, regular, STEP)
            g_check = directional(p, u, e, regular, CHECK_STEP)
            total = [t + mpmath.conj(v) for t, v in zip(total, g)]
            total_check = [t + mpmath.conj(v) for t, v in zip(total_check, g_check)]
        what = f"{'P2M' if regular else 'P2L'} in frame {frame}"
        check_degrees(p, total_check, total, what)
        # Degree 0 vanishes identically (grad R_0^0 = 0); its central difference is
        # rounding noise at this precision.
        if regular:
            if abs(total[0]) > mpf(10) ** -(DIFF_DPS - 40):
                sys.exit(f"{what}: degree 0 does not vanish")
            total[0] = mpf(0)
        return gh.real_storage(p, total)


def direct(sources, dipoles, targets):
    """Potential and gradient of the dipoles at each target, with term magnitudes."""
    out = []
    with mp.workdps(VALUE_DPS):
        for x in targets:
            x = [mpf(v) for v in x]

            def potential(z):
                s = mpf(0)
                for y, d in zip(sources, dipoles):
                    w = [zi - mpf(yi) for zi, yi in zip(z, y)]
                    r = mpmath.sqrt(sum(t * t for t in w))
                    s += sum(mpf(di) * wi for di, wi in zip(d, w)) / r**3
                return s

            phi = potential(x)
            grad = [mpf(0)] * 3
            for y, d in zip(sources, dipoles):
                w = [xi - mpf(yi) for xi, yi in zip(x, y)]
                r2 = sum(t * t for t in w)
                r = mpmath.sqrt(r2)
                s = sum(mpf(di) * wi for di, wi in zip(d, w))
                for k in range(3):
                    grad[k] += mpf(d[k]) / r**3 - 3 * s * w[k] / r**5
            # The closed-form gradient against central differences of the potential.
            with mp.workdps(DIFF_DPS):
                h = mpf(10) ** -40
                for k in range(3):
                    xp = [xi + (h if i == k else 0) for i, xi in enumerate(x)]
                    xm = [xi - (h if i == k else 0) for i, xi in enumerate(x)]
                    fd = (potential(xp) - potential(xm)) / (2 * h)
                    scale = max(abs(g) for g in grad)
                    if abs(fd - grad[k]) > mpf(10) ** -DIGITS * scale:
                        sys.exit(f"direct sum at {x}: gradient {k} disagrees with differences")
            out.append(
                {
                    "x": [gh.fmt(v) for v in x],
                    "potential": gh.fmt(phi),
                    "gradient": [gh.fmt(g) for g in grad],
                }
            )
    return out


def in_ball(rng, centre, radius, rmin, rmax):
    """A double point c + r u with |u| in [rmin, rmax], uniform direction."""
    (u,) = gh.random_points(rng, 1, rmin, rmax)
    return [ci + radius * ui for ci, ui in zip(centre, u)]


def moment(rng):
    return [2.0 * rng.random() - 1.0 for _ in range(3)]


def generate(version):
    rng = random.Random(SEED)
    p2m, p2l = [], []
    for centre, radius in FRAMES:
        for regular, radii, out in ((True, gh.REGULAR_RADII, p2m), (False, gh.IRREGULAR_RADII, p2l)):
            sources = [in_ball(rng, centre, radius, *radii) for _ in range(SOURCES)]
            dipoles = [moment(rng) for _ in range(SOURCES)]
            coeffs = expansion(P, (centre, radius), sources, dipoles, regular)
            out.append(
                {
                    "centre": [gh.fmt(v) for v in centre],
                    "radius": gh.fmt(radius),
                    "sources": [[gh.fmt(v) for v in y] for y in sources],
                    "dipoles": [[gh.fmt(v) for v in d] for d in dipoles],
                    "coefficients": [gh.fmt(v) for v in coeffs],
                }
            )
    sources = [[rng.random() for _ in range(3)] for _ in range(DIRECT_SOURCES)]
    dipoles = [moment(rng) for _ in range(DIRECT_SOURCES)]
    targets = [[1.5 * rng.random() - 0.25 for _ in range(3)] for _ in range(DIRECT_TARGETS)]
    header = {
        "description": (
            "Dipole P2M and P2L coefficients and the dipole direct sum "
            "(docs/CONVENTIONS.md §3.7, 'Dipole sources') at seeded random points"
        ),
        "convention_version": version,
        "generator": "tools/fixtures/gen_dipoles.py",
        "generator_version": GENERATOR_VERSION,
        "mpmath_version": gh.MPMATH_VERSION,
        "verified_digits": DIGITS,
        "seed": SEED,
        "p": P,
        "layout": "real storage, index n*n + n + m for -n <= m <= n (§3.6)",
        "format": (
            "decimal strings, 17 significant digits of the nearest double; moments in "
            "the units of the points (the operators divide them by the frame radius); "
            "potential and gradient without 1/(4 pi)"
        ),
        "regular_radius": [gh.fmt(r) for r in gh.REGULAR_RADII],
        "irregular_radius": [gh.fmt(r) for r in gh.IRREGULAR_RADII],
    }
    return {
        "header": header,
        "p2m": p2m,
        "p2l": p2l,
        "direct": {
            "sources": [[gh.fmt(v) for v in y] for y in sources],
            "dipoles": [[gh.fmt(v) for v in d] for d in dipoles],
            "targets": direct(sources, dipoles, targets),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail unless the committed file is byte-identical",
    )
    args = parser.parse_args()
    if mpmath.__version__ != gh.MPMATH_VERSION:
        sys.exit(f"needs mpmath {gh.MPMATH_VERSION}, found {mpmath.__version__}")
    gh.check_legendre_sign()
    text = (gh.to_json(generate(gh.convention_version())) + "\n").encode("utf-8")
    rel = OUT.relative_to(gh.ROOT)
    if args.check:
        if not OUT.exists() or OUT.read_bytes() != text:
            sys.exit(f"not byte-identical to a fresh regeneration: {rel}")
        print(f"checked {rel} ({len(text)} bytes)")
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(text)
        print(f"wrote {rel} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
