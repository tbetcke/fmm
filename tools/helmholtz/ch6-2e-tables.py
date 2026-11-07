#!/usr/bin/env python3
"""The tables of Helmholtz CH6.2e's report (fmm-validate/results/helmholtz-ch6-2e.md) from
the outputs of tools/helmholtz/ch6-2e-runs.sh:

    tools/helmholtz/ch6-2e-tables.py <out-dir>[,<out-dir>…] verify|scan|times

Several output directories (comma-separated) are read as one: a later directory's rows of the
same case and reach replace an earlier one's (the report's 4.25 launch beside the first, its
two scans side by side).

`verify`: per case (workload, kD, ε, charges) the larger of the φ and ∇φ ratios to
Rotation's error at each reach, then per reach the largest ratio, the cases above 1.1 and
the tables' bytes over the corner's. `scan`: per case and level the larger ratio of the
level alone at iprec 2, at the corner and at each other reach, and per reach the largest.
`times`: per case and level with waves the waves per direction and the M2L seconds at the
corner and at the other reach (each the mean of the two launches, ABBA), their ratios, and
per case the M2L's seconds over every level and the evaluation's (its ratio only from 1 s:
the example prints it to 10 ms).
"""

import glob
import os
import re
import sys
from collections import defaultdict

CORNER = "corner"


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def reach_of(m2l):
    m = re.search(r" ρ ([0-9.]+)$", m2l)
    return m.group(1) if m else CORNER


def reach_key(r):
    return float("inf") if r == CORNER else float(r)


def main_rows(path):
    """The rows of the first table: dicts with the case, the setting, the reach, the
    errors, the ratio and the M2L column."""
    rows = []
    for line in open(path):
        if not line.startswith("| W"):
            continue
        c = cells(line)
        if len(c) != 20 or c[0] == "Workload":
            continue
        rows.append(
            {
                "case": (c[0], c[1], c[2], c[3], c[4]),
                "m2l": c[7],
                "reach": reach_of(c[7]),
                "evaluate": float(c[15]),
                "errors": (float(c[16]), float(c[17])),
                "ratio": c[19],
            }
        )
    return rows


def level_rows(path):
    """The rows of the second table (`--levels`)."""
    rows = []
    seen = False
    for line in open(path):
        if line.startswith("## M2L per level"):
            seen = True
            continue
        if not seen or not line.startswith("| W"):
            continue
        c = cells(line)
        if len(c) != 15 or c[0] == "Workload":
            continue
        rows.append(
            {
                "case": (c[0], c[1], c[2], c[3]),
                "setting": c[4],
                "reach": reach_of(c[4]),
                "level": c[6],
                "p": c[7],
                "ran": c[8],
                "pairs": c[9],
                "seconds": float(c[10]),
                "waves": c[14],
            }
        )
    return rows


def ratio(text):
    if text in ("–", ""):
        return None
    return max(float(x) for x in text.split("/"))


def megabytes(m2l):
    m = re.search(r"\(([0-9]+) MB;", m2l)
    return float(m.group(1)) if m else None


def case_name(case):
    workload, n, kd, degrees, charges = case
    extra = "" if charges == "complex" else f", {charges}"
    return f"{workload} | {kd} | {degrees.replace('ε = ', '')}{extra}"


def files_of(out, pattern):
    """The files matching `pattern` in each comma-separated directory of `out`, in order."""
    return [
        f for d in out.split(",") for f in sorted(glob.glob(os.path.join(d, pattern)))
    ]


def verify(out):
    files = files_of(out, "verify-*.md")
    table = defaultdict(dict)
    sizes = defaultdict(dict)
    reaches = set()
    for path in files:
        for row in main_rows(path):
            if row["m2l"] == "rotation":
                continue
            r = ratio(row["ratio"])
            if r is None:
                continue
            table[row["case"]][row["reach"]] = r
            sizes[row["case"]][row["reach"]] = megabytes(row["m2l"])
            reaches.add(row["reach"])
    order = sorted(reaches, key=reach_key, reverse=True)
    print("| workload | kD | ε | " + " | ".join(f"ρ {r}" for r in order) + " |")
    print("| --- | --- | --- | " + " | ".join("---:" for _ in order) + " |")
    for case in table:
        line = " | ".join(
            f"{table[case][r]:.3f}" if r in table[case] else "–" for r in order
        )
        print(f"| {case_name(case)} | {line} |")
    print()
    print(
        "| reach | cases | largest ratio | above 1.1 | tables and buffers over the corner's "
        "| the cases above 1.1 |"
    )
    print("| --- | ---: | ---: | ---: | --- | --- |")
    for r in order:
        values = [table[c][r] for c in table if r in table[c]]
        above = [case_name(c) for c in table if table[c].get(r, 0) > 1.1]
        fractions = [
            sizes[c][r] / sizes[c][CORNER]
            for c in sizes
            if r in sizes[c] and sizes[c].get(CORNER)
        ]
        span = f"{min(fractions):.2f}–{max(fractions):.2f}" if fractions else "–"
        names = "; ".join(n.replace(" | ", ", ") for n in above) or "–"
        print(
            f"| {r} | {len(values)} | {max(values):.3f} | {len(above)} | {span} | {names} |"
        )


def scan(out):
    files = files_of(out, "scan-*.md")
    table = defaultdict(dict)
    reaches = set()
    for path in files:
        for row in main_rows(path):
            m = re.match(r"planewave ([0-9]+):i2 ", row["m2l"])
            r = ratio(row["ratio"])
            if not m or r is None:
                continue
            table[(row["case"], int(m.group(1)))][row["reach"]] = r
            reaches.add(row["reach"])
    order = sorted(reaches, key=reach_key, reverse=True)
    print("| workload | kD | ε | level | " + " | ".join(f"ρ {r}" for r in order) + " |")
    print("| --- | --- | --- | ---: | " + " | ".join("---:" for _ in order) + " |")
    for (case, level), values in table.items():
        line = " | ".join(f"{values[r]:.3f}" if r in values else "–" for r in order)
        print(f"| {case_name(case)} | {level} | {line} |")
    print()
    for r in order:
        values = [v[r] for v in table.values() if r in v]
        above = sum(1 for v in values if v > 1.1)
        print(f"ρ {r}: {len(values)} levels, largest {max(values):.3f}, {above} above 1.1")


def times(out):
    files = files_of(out, "times-*.md")
    seconds = defaultdict(list)
    waves = {}
    evaluate = defaultdict(list)
    m2l_total = defaultdict(list)
    meta = {}
    reaches = set()
    for path in files:
        for row in level_rows(path):
            if row["level"] == "Σ":
                m2l_total[(row["case"], row["reach"])].append(row["seconds"])
                continue
            key = (row["case"], int(row["level"]))
            seconds[(key, row["reach"])].append(row["seconds"])
            waves[(key, row["reach"])] = row["waves"]
            meta[key] = (row["p"], row["pairs"])
            reaches.add(row["reach"])
        for row in main_rows(path):
            evaluate[(row["case"][:4], row["reach"])].append(row["evaluate"])
    other = [r for r in reaches if r != CORNER]
    assert len(other) == 1, other
    other = other[0]
    print(
        f"| workload | kD | ε | level | p | V pairs | waves corner | waves ρ {other} | ratio "
        f"| M2L s corner | M2L s ρ {other} | ratio |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    total = defaultdict(float)
    for key in sorted(meta, key=lambda k: (k[0], k[1])):
        a, b = waves.get((key, CORNER)), waves.get((key, other))
        if a in (None, "–"):
            continue
        sa = sum(seconds[(key, CORNER)]) / len(seconds[(key, CORNER)])
        sb = sum(seconds[(key, other)]) / len(seconds[(key, other)])
        case, level = key
        p, pairs = meta[key]
        workload, n, kd, degrees = case
        print(
            f"| {workload} | {kd} | {degrees.replace('ε = ', '')} | {level} | {p} | {pairs} "
            f"| {a} | {b} | {int(b) / int(a):.3f} | {sa:.4f} | {sb:.4f} | {sb / sa:.3f} |"
        )
    print()
    print(
        f"| workload | kD | ε | M2L s corner | M2L s ρ {other} | ratio | evaluate s corner "
        f"| evaluate s ρ {other} | ratio |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    mean = lambda v: sum(v) / len(v)
    for case in sorted({k[0] for k in evaluate}):
        ea, eb = mean(evaluate[(case, CORNER)]), mean(evaluate[(case, other)])
        ma, mb = mean(m2l_total[(case, CORNER)]), mean(m2l_total[(case, other)])
        workload, n, kd, degrees = case
        evaluation = (
            f"{ea:.2f} | {eb:.2f} | {eb / ea:.3f}" if ea >= 1.0 else f"{ea:.2f} | {eb:.2f} | –"
        )
        print(
            f"| {workload} | {kd} | {degrees.replace('ε = ', '')} | {ma:.4f} | {mb:.4f} "
            f"| {mb / ma:.3f} | {evaluation} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("verify", "scan", "times"):
        sys.exit(__doc__)
    {"verify": verify, "scan": scan, "times": times}[sys.argv[2]](sys.argv[1])
