#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["mpmath==1.4.1"]
# ///
"""FMM3D's Helmholtz direct sum as fixtures for nd-fmm-ref (Helmholtz CH1.4).

Writes fmm-ref/fixtures/helmholtz_fmm3d.json: the potential and gradient of
G_k(x, y) = e^{ik|x - y|}/(4 pi |x - y|) (CONVENTIONS §3.15, "Kernel and factors";
drafted in CH1.4, adopted in CH1.1) as FMM3D 2.1.0's direct routine `h3ddirectcg`
computes them, through tools/fmm3d/driver.f90's `helmholtz` mode, for
  - 200 sources in [0, 1]^3 and 40 further targets in [-0.5, 1.5]^3 (seed SEED): the
    values at the 200 sources (the self term dropped) and at the 40 targets;
  - real charges and complex charges (each part uniform in [-1, 1]);
  - k = 1, 25 (real), 1 + 0.5i, 25 + 2.5i (complex), i, 25i (imaginary: Yukawa).
Before writing, it runs the driver's two-point check (which stops unless FMM3D computes
the draft's kernel), identifies FMM3D's conventions on the two points among the
candidates of tools/fmm3d/helmholtz.py (exactly one must match each case), and checks
every value against the draft's kernel summed in mpmath at DPS digits: relative l2 per
case and quantity, and the largest error per target relative to its sum of term
magnitudes, both at most TOLERANCE. The values are FMM3D's doubles, written with 17
significant digits.

Run from the repository root, with FMM3D built (tools/fmm3d/README.md):
    FMM3D_ROOT=... uv run tools/fixtures/gen_helmholtz_fmm3d.py
--check regenerates in memory and compares with the committed file: every value to
TOLERANCE relative (per case and quantity, relative l2), the header's FMM3D release, routine
and inputs exactly; it says whether the file is also byte-identical (it is on the machine
and compiler that wrote it; FMM3D's sums differ in the last bits between compilers).
--no-reference skips the mpmath check, for a machine without mpmath (Kathleen's Python 3.9:
    python3 tools/fixtures/gen_helmholtz_fmm3d.py --check --no-reference).
See tools/fixtures/README.md.
"""
import argparse
import json
import os
import platform
import random
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "fmm3d"))
sys.dont_write_bytecode = True  # no __pycache__ in tools/fmm3d

import helmholtz as hz  # noqa: E402  (the driver's formats, the check, the candidates)

# Bump when the generator changes its output (format, points or algorithm).
GENERATOR_VERSION = 1
MPMATH_VERSION = "1.4.1"
SEED = 20261028
SOURCES = 200
TARGETS = 40
DPS = 30
TOLERANCE = 1e-13
THRESH = "1e-300"
# (label, kind, k)
WAVENUMBERS = [
    ("real k = 1", "real", complex(1, 0)),
    ("real k = 25", "real", complex(25, 0)),
    ("complex k = 1 + 0.5i", "complex", complex(1, 0.5)),
    ("complex k = 25 + 2.5i", "complex", complex(25, 2.5)),
    ("imaginary k = i", "imaginary", complex(0, 1)),
    ("imaginary k = 25i", "imaginary", complex(0, 25)),
]
FMM3D_RELEASE = "v2.1.0"
OUT = ROOT / "fmm-ref" / "fixtures" / "helmholtz_fmm3d.json"


def fmt(x):
    """17 significant digits of a double; parses back to the same double."""
    return f"{x:.16e}"


def cfmt(z):
    return [fmt(z.real), fmt(z.imag)]


def to_json(value, indent=""):
    """JSON with one line per array of scalars (as tools/fixtures/gen_harmonics.py)."""
    inner = indent + " "
    if isinstance(value, dict):
        items = [f"{inner}{json.dumps(k)}: {to_json(v, inner)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + indent + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        items = [inner + to_json(v, inner) for v in value]
        return "[\n" + ",\n".join(items) + "\n" + indent + "]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def convention_version():
    text = (ROOT / "docs" / "CONVENTIONS.md").read_text()
    found = set(re.findall(r"`CONVENTION_VERSION = (\d+)`", text))
    if len(found) != 1:
        sys.exit(f"expected one CONVENTION_VERSION in docs/CONVENTIONS.md, found {found}")
    return int(found.pop())


def build_info(driver):
    """FMM3D's release, commit and compilers from build.sh's build-info.txt."""
    info = Path(driver).resolve().parents[1] / "build-info.txt"
    lines = info.read_text().splitlines() if info.exists() else []
    head = lines[0] if lines else ""
    if not head.startswith(f"FMM3D {FMM3D_RELEASE},"):
        sys.exit(f"needs FMM3D {FMM3D_RELEASE} (tools/fmm3d/build.sh); {info}: {head!r}")
    fortran = next((ln for ln in lines if ln.startswith("FC ")), "")
    return {
        "fmm3d": head,
        "variant": next((ln for ln in lines if ln.startswith("variant ")), ""),
        "fortran": re.sub(r"^FC\s+\S+:\s*", "", fortran),
        "machine": f"{platform.system()} {platform.machine()}",
    }


def inputs():
    rng = random.Random(SEED)
    sources = [(rng.random(), rng.random(), rng.random()) for _ in range(SOURCES)]
    targets = [tuple(2 * rng.random() - 0.5 for _ in range(3)) for _ in range(TARGETS)]
    real = [complex(2 * rng.random() - 1, 0) for _ in range(SOURCES)]
    cplx = [complex(2 * rng.random() - 1, 2 * rng.random() - 1) for _ in range(SOURCES)]
    return sources, targets, {"real": real, "complex": cplx}


def reference(k, sources, points, charges):
    """The draft's potential and gradient at points (exact coincidences dropped), summed
    in mpmath at DPS digits, and per point the sum of the term magnitudes."""
    import mpmath
    from mpmath import mp, mpc, mpf

    if mpmath.__version__ != MPMATH_VERSION:
        sys.exit(f"needs mpmath {MPMATH_VERSION}, found {mpmath.__version__}")
    with mp.workdps(DPS):
        kk = mpc(k.real, k.imag)
        four_pi = 4 * mp.pi
        ys = [[mpf(c) for c in y] for y in sources]
        qs = {name: [mpc(q.real, q.imag) for q in qv] for name, qv in charges.items()}
        out = {name: [] for name in charges}
        for x in points:
            xm = [mpf(c) for c in x]
            pot = {name: mpc(0) for name in charges}
            grad = {name: [mpc(0)] * 3 for name in charges}
            size = {name: mpf(0) for name in charges}
            for j, y in enumerate(ys):
                d = [a - b for a, b in zip(xm, y)]
                r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
                if r2 == 0:
                    continue
                r = mp.sqrt(r2)
                g = mp.exp(1j * kk * r) / (four_pi * r)
                dg = (1j * kk * r - 1) * g / r2
                for name in charges:
                    q = qs[name][j]
                    pot[name] += q * g
                    grad[name] = [grad[name][c] + q * dg * d[c] for c in range(3)]
                    size[name] += abs(q * g) + abs(q * dg) * mp.sqrt(r2)
            for name in charges:
                out[name].append((complex(pot[name]), [complex(v) for v in grad[name]],
                                  float(size[name])))
        return out


def agreement(values, ref):
    """Relative l2 of potential and gradient, and the largest error per point relative
    to the point's sum of term magnitudes."""
    pots, grads = values
    rp = hz.rel_l2(pots, [r[0] for r in ref])
    rg = hz.rel_l2(grads, [r[1] for r in ref])
    worst = 0.0
    for p, g, r in zip(pots, grads, ref):
        err = max([abs(p - r[0])] + [abs(a - b) for a, b in zip(g, r[1])])
        worst = max(worst, err / r[2])
    return {"rel_l2_potential": rp, "rel_l2_gradient": rg, "max_error_by_term_sum": worst}


def generate(driver, with_reference):
    records, passed = hz.run_check(driver)
    if not passed:
        sys.exit("the two-point check failed: FMM3D does not compute the draft's kernel")
    two_point = []
    for r in records:
        names = hz.identify(r)
        if not r["record"] and len(names) != 1:
            sys.exit(f"the two-point case zk={r['zk']} {r['routine']} matches {names}")
        two_point.append({
            "k": cfmt(r["zk"]),
            "routine": r["routine"],
            "case": "outside the draft's branch, for the record" if r["record"] else "checked",
            "potential_at_x1": cfmt(r["pot1"]),
            "gradient_at_x1": [cfmt(v) for v in r["grad1"]],
            "potential_at_x2": cfmt(r["pot2"]),
            "gradient_at_x2": [cfmt(v) for v in r["grad2"]],
            "identified": names,
        })

    sources, targets, charges = inputs()
    points = list(sources) + list(targets)
    work = Path(tempfile.mkdtemp(prefix="helmholtz-fixtures-"))
    cases = []
    for label, kind, k in WAVENUMBERS:
        ref = reference(k, sources, points, charges) if with_reference else None
        for name, q in charges.items():
            problem = work / f"p-{len(cases)}.bin"
            hz.write_problem(problem, sources, q, list(range(SOURCES)), targets, SEED)
            _, (pot, grad, pt, gt) = hz.run_driver(driver, problem, work / f"o-{len(cases)}",
                                                   k, ["direct"])["direct"]
            pots, grads = pot + pt, grad + gt
            case = {"label": f"{label}, {name} charges", "kind": kind, "k": cfmt(k),
                    "charges": name}
            if ref is not None:
                a = agreement((pots, grads), ref[name])
                bad = {key: v for key, v in a.items() if not v <= TOLERANCE}
                if bad:
                    sys.exit(f"{case['label']}: FMM3D against the draft's kernel: {bad}")
                case["agreement_with_the_draft"] = {key: f"{v:.2e}" for key, v in a.items()}
            case["potential"] = [cfmt(v) for v in pots]
            case["gradient"] = [[s for v in g for s in cfmt(v)] for g in grads]
            cases.append(case)
            print(f"{case['label']}: {case.get('agreement_with_the_draft', 'not checked')}",
                  file=sys.stderr)
    for path in work.iterdir():
        path.unlink()
    work.rmdir()

    header = {
        "description": (
            "FMM3D's Helmholtz direct sum, the potential and gradient of "
            "e^{ik|x-y|}/(4 pi |x-y|) (CONVENTIONS §3.15, Kernel and factors), for real, "
            "complex and imaginary k and real and complex charges (Helmholtz CH1.4)"
        ),
        "convention_version": convention_version(),
        "convention": "CONVENTIONS §3.15 (drafted in Helmholtz CH1.4, adopted in CH1.1)",
        "generator": "tools/fixtures/gen_helmholtz_fmm3d.py",
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "fmm3d_release": FMM3D_RELEASE,
        "routine": "h3ddirectcg (FMM3D's direct sum: charges, potential and gradient)",
        "eps": "none: a direct evaluation (hfmm3d_s_c_g at eps 1e-12 only in two_point)",
        "thresh": THRESH + " (only exact coincidences dropped, as CONVENTIONS §3.13)",
        "build": None,  # filled by main()
        "reference": (
            f"every value checked against the kernel of §3.15 summed in mpmath "
            f"{MPMATH_VERSION} at {DPS} digits, to {TOLERANCE:g} (relative l2 per case and "
            "quantity, and per point relative to its sum of term magnitudes)"
        ),
        "format": (
            "decimal strings, 17 significant digits of FMM3D's doubles; complex values as "
            "[re, im]; a gradient as [re x, im x, re y, im y, re z, im z]; the values at "
            "the sources (self term dropped) in source order, then at the targets; the "
            "factor 1/(4 pi) included"
        ),
        "two_point_geometry": "charges 0.3 - 0.7i at x1 = (0, 0, 0) and 1 at x2 = (1, 0, 0)",
    }
    return {
        "header": header,
        "two_point": two_point,
        "sources": [[fmt(c) for c in y] for y in sources],
        "targets": [[fmt(c) for c in x] for x in targets],
        "charges": {name: [cfmt(v) for v in q] for name, q in charges.items()},
        "cases": cases,
    }


def compare(fresh, committed):
    """Problems of the fresh regeneration against the committed file, per --check."""
    problems = []
    for key in ("convention_version", "generator_version", "seed", "fmm3d_release",
                "routine", "thresh"):
        if fresh["header"][key] != committed["header"][key]:
            problems.append(f"header {key}: {committed['header'][key]!r} -> "
                            f"{fresh['header'][key]!r}")
    for key in ("sources", "targets", "charges"):
        if fresh[key] != committed[key]:
            problems.append(f"{key} differ")
    if len(fresh["cases"]) != len(committed["cases"]):
        return problems + ["the number of cases differs"]

    def cx(pairs):
        return [complex(float(a), float(b)) for a, b in pairs]

    for a, b in zip(fresh["cases"], committed["cases"]):
        if (a["k"], a["charges"]) != (b["k"], b["charges"]):
            problems.append(f"case {b['label']}: k or charges differ")
            continue
        ep = hz.rel_l2(cx(a["potential"]), cx(b["potential"]))
        eg = hz.rel_l2([cx(zip(g[::2], g[1::2])) for g in a["gradient"]],
                       [cx(zip(g[::2], g[1::2])) for g in b["gradient"]])
        print(f"{b['label']}: against the committed values {ep:.2e} (potential), "
              f"{eg:.2e} (gradient)", file=sys.stderr)
        if not (ep <= TOLERANCE and eg <= TOLERANCE):
            problems.append(f"case {b['label']}: {ep:.2e}, {eg:.2e} > {TOLERANCE:g}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--driver", help="default $FMM3D_ROOT/default/bin/fmm3d_driver")
    parser.add_argument("--check", action="store_true",
                        help="compare a fresh regeneration with the committed file")
    parser.add_argument("--no-reference", action="store_true",
                        help="skip the mpmath check of every value")
    args = parser.parse_args()
    driver = args.driver or hz.default_driver()
    if not os.access(driver, os.X_OK):
        sys.exit(f"no driver at {driver} (tools/fmm3d/build.sh build, or driver)")
    data = generate(driver, not args.no_reference)
    data["header"]["build"] = build_info(driver)
    text = (to_json(data) + "\n").encode("utf-8")
    rel = OUT.relative_to(ROOT)
    if args.check:
        if not OUT.exists():
            sys.exit(f"no committed {rel}")
        committed = json.loads(OUT.read_text())
        problems = compare(data, committed)
        if problems:
            sys.exit(f"{rel} does not match a fresh regeneration:\n  " + "\n  ".join(problems))
        if args.no_reference:
            same = "the bytes not compared (no reference values without mpmath)"
        elif OUT.read_bytes() == text:
            same = "byte-identical"
        else:
            same = "not byte-identical (another build or compiler)"
        print(f"checked {rel} ({len(text)} bytes): every value to {TOLERANCE:g}; {same}")
    else:
        if args.no_reference:
            sys.exit("--no-reference only with --check: committed values are always checked")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(text)
        print(f"wrote {rel} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
