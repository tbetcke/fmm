#!/usr/bin/env python3
"""tools/helmholtz/ch4-3b1-fit.py: the cost model of the diagonal form's time rule, Helmholtz
CH4.3b1 (docs/helmholtz/tasks/CH4.3b1.md), fitted from `helmholtz_accuracy --levels` outputs.

    python3 tools/helmholtz/ch4-3b1-fit.py fit <machine>=<file-or-dir> …
    python3 tools/helmholtz/ch4-3b1-fit.py check <machine>=<file-or-dir> …

`fit` prints, per machine, the constants [a, b] of a (p + 1)² + b (p + 1)³ thread-ns for
each unit, by least squares on the relative error over every level of every run given:

- S → F per source signature and F → R per accumulator converted back, from the table
  "Diagonal M2L"'s thread-seconds and counts (runs with the diagonal form, every class
  diagonal: the levels of 34 classes alone, so that no RCR pair is in the parts);
- the diagonal pair, from the same table's pairs' thread-seconds over the diagonal pairs;
- RCR's pair, from "M2L per level" under Rotation without the form: the level's seconds ×
  the threads over its V pairs (wall × threads, idle threads included, as the rule compares
  whole level calls);
- (one-rank runs only: on several ranks the seconds are the slowest rank's and the counts
  the sum's)
- the exponential M2L's pair at its degree p_w, from "M2L per level" rows that ran
  `planewave`, the same measure (p_w read from the p column: the degrees run).

With the table each unit's fitted values against the measured per level, the ratio, and the
diagonal pair over RCR's at p = 8 … 245 (the per-class crossover). `check` prints, per run
with the form and level, the model's diagonal level against the measured one. No timing is
asserted; the numbers go into fmm-validate/results/helmholtz-ch4-3b1.md.
"""

import pathlib
import sys


def tables(path):
    """Every Markdown table of a launch's output, by the heading above it ("" for the
    first): a list of dicts keyed by the header."""
    out, heading, header = {}, "", None
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("## "):
            heading, header = line[3:].strip(), None
            continue
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            out.setdefault(heading, [])
        elif set(cells[0]) <= set("-: "):
            continue
        elif len(cells) == len(header):
            out[heading].append(dict(zip(header, cells)))
    return out


def files(arg):
    machine, where = arg.split("=", 1)
    where = pathlib.Path(where)
    paths = sorted(where.glob("*.md")) if where.is_dir() else [where]
    return machine, paths


def threads(row):
    ranks, per = row["ranks × threads"].split("×")
    return int(ranks) * int(per)


def one_rank(row):
    """Whether the run is on one rank: on several, "Diagonal M2L"'s seconds are the slowest
    rank's and its counts summed over the ranks."""
    return row["ranks × threads"].split("×")[0].strip() == "1"


def samples(machine, paths):
    """Per unit a list of (p, thread-ns per unit, weight) from every run."""
    units = {"s2f": [], "f2r": [], "pair": [], "rotation": [], "plane_wave": []}
    for path in paths:
        t = tables(path)
        for r in t.get("Diagonal M2L", []):
            if not one_rank(r):
                continue
            if r["classes"] != "34" or int(r["RCR pairs"]) != 0:
                continue
            p = int(r["p"])
            s2f, f2r = int(r["S → F"]), int(r["F → R"])
            pairs = int(r["diagonal pairs"])
            if s2f:
                units["s2f"].append((p, float(r["S → F thread-s"]) * 1e9 / s2f, s2f))
            if f2r:
                units["f2r"].append((p, float(r["F → R thread-s"]) * 1e9 / f2r, f2r))
            if pairs:
                units["pair"].append((p, float(r["pairs thread-s"]) * 1e9 / pairs, pairs))
        for r in t.get("M2L per level", []):
            if not one_rank(r) or r["level"] == "Σ" or r["V pairs"] in ("0", "–"):
                continue
            pairs = int(r["V pairs"])
            ns = float(r["M2L s"]) * threads(r) * 1e9 / pairs
            ran = r["ran"]
            if ran.startswith("rotation"):
                units["rotation"].append((int(r["p"]), ns, pairs))
            elif ran.startswith("planewave"):
                units["plane_wave"].append((int(r["p"]), ns, pairs))
    return units


def fit(points):
    """[a, b] of a q² + b q³, q = p + 1, minimising Σ ((model − y) / y)², and, if b < 0,
    the one-term fits instead (the smaller residual)."""
    if not points:
        return None
    rows = [((p + 1) ** 2 / y, (p + 1) ** 3 / y) for p, y, _ in points]
    s11 = sum(x * x for x, _ in rows)
    s12 = sum(x * z for x, z in rows)
    s22 = sum(z * z for _, z in rows)
    t1 = sum(x for x, _ in rows)
    t2 = sum(z for _, z in rows)
    det = s11 * s22 - s12 * s12
    if det > 0:
        a = (t1 * s22 - t2 * s12) / det
        b = (s11 * t2 - s12 * t1) / det
        if a >= 0 and b >= 0:
            return [a, b]
    # One term alone.
    best = None
    for coef, col in ((0, 0), (1, 1)):
        num = sum(r[col] for r in rows)
        den = sum(r[col] ** 2 for r in rows)
        c = num / den
        ab = [c, 0.0] if coef == 0 else [0.0, c]
        res = sum((model(ab, p) / y - 1) ** 2 for p, y, _ in points)
        if best is None or res < best[0]:
            best = (res, ab)
    return best[1]


def model(ab, p):
    q = p + 1
    return ab[0] * q * q + ab[1] * q * q * q


def run_fit(args):
    for arg in args:
        machine, paths = files(arg)
        units = samples(machine, paths)
        consts = {u: fit(pts) for u, pts in units.items()}
        print(f"### {machine}\n")
        print("| unit | a (ns) | b (ns) |")
        print("| --- | ---: | ---: |")
        for u, ab in consts.items():
            if ab:
                print(f"| {u} | {ab[0]:.4g} | {ab[1]:.4g} |")
        print()
        print("| unit | p | measured ns | model ns | model / measured |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for u, pts in units.items():
            ab = consts[u]
            for p, y, _ in sorted(set(pts)):
                print(f"| {u} | {p} | {y:.4g} | {model(ab, p):.4g} | {model(ab, p) / y:.3f} |")
        print()
        if consts["pair"] and consts["rotation"]:
            print("| p | diagonal pair / RCR pair |")
            print("| ---: | ---: |")
            for p in (8, 12, 16, 24, 32, 48, 64, 96, 131, 180, 245):
                print(f"| {p} | {model(consts['pair'], p) / model(consts['rotation'], p):.3f} |")
            print()
        print("Rust:")
        for u, ab in consts.items():
            if ab:
                print(f"    {u}: [{ab[0]:.3g}, {ab[1]:.3g}],")
        print()


def main(argv):
    if len(argv) < 3 or argv[1] not in ("fit",):
        print(__doc__)
        return 2
    run_fit(argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
