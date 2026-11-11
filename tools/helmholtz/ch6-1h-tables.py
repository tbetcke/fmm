#!/usr/bin/env python3
"""tools/helmholtz/ch6-1h-tables.py: the tables of Helmholtz CH6.1h's report from the
outputs of tools/helmholtz/ch6-1h-runs.sh (docs/helmholtz/tasks/CH6.1h.md).

    python3 tools/helmholtz/ch6-1h-tables.py <out-dir>

Per case (workload, kD, ε), Rotation and PlaneWave by the rule in each `PlaneWaveRaise` form
(`level`, `leaves`, `m2l`): the errors of φ and ∇φ, each form's over Rotation's; and the
times of the two launches (forward and reverse order; the smaller of each): the evaluation,
the rest of the evaluation (evaluation − M2L summed over the levels), the M2L and the
build, each form's over Rotation's. The checksums of a run's two launches must agree (the
output is deterministic). Then the ranges per ε, measure and form; the times' ranges over
the cases run in both orders only (a stage run forward alone, as `e7x`, is marked *).
"""

import glob
import os
import sys

FORMS = ["planewave", "planewave/leaves", "planewave/m2l"]
NAMES = {"planewave": "Level", "planewave/leaves": "Leaves", "planewave/m2l": "M2l"}


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def run_name(m2l):
    """`rotation`, or `planewave` with the form's suffix of the example's column "M2L"."""
    first = m2l.split()[0]
    for suffix in ("/leaves", "/m2l"):
        if m2l.endswith(suffix) and not first.endswith(suffix):
            return first + suffix
    return first


def parse(path):
    """Rows of the main table and Σ rows of the level table, keyed by (case, run)."""
    main, sums = {}, {}
    header = None
    with open(path) as f:
        for line in f:
            if not line.startswith("|") or line.startswith("| ---"):
                continue
            c = cells(line)
            if c[0] == "Workload":
                header = c
                continue
            row = dict(zip(header, c))
            case = (row["Workload"], row["kD"], row["degrees"])
            run = run_name(row["M2L"])
            if "level" in row:
                if row["level"] == "Σ":
                    evaluate = float(row["ran"].split()[1])
                    sums[(case, run)] = (float(row["M2L s"]), evaluate)
            else:
                main[(case, run)] = row
    return main, sums


def main():
    out = sys.argv[1]
    rows = {}
    for path in sorted(glob.glob(os.path.join(out, "e*-*.md"))):
        main_rows, sums = parse(path)
        for key, row in main_rows.items():
            m2l_s, evaluate = sums[key]
            rows.setdefault(key, []).append((row, m2l_s, evaluate))
    cases = []
    for (case, _) in rows:
        if case not in cases:
            cases.append(case)
    order = {"1e-7": 0, "1e-8": 1}
    cases.sort(key=lambda c: (order[c[2].split("= ")[1]], c[0], float(c[1].split()[0])))

    def ratio(a, b):
        return b / a if a > 0 else float("nan")

    measures = {}
    for case in cases:
        runs = {r: rows[(case, r)] for r in ["rotation"] + FORMS}
        for r, launches in runs.items():
            sums = {x[0]["checksum"] for x in launches}
            assert len(sums) == 1, (case, r, sums)
        first = {r: launches[0][0] for r, launches in runs.items()}
        m = {}
        for r, launches in runs.items():
            m[r] = {
                "φ": float(first[r]["φ"]),
                "∇φ": float(first[r]["∇φ"]),
                "M2L": min(x[1] for x in launches),
                "evaluate": min(x[2] for x in launches),
                "rest": min(x[2] - x[1] for x in launches),
                "build": min(float(x[0]["build s"]) for x in launches),
            }
        measures[case] = (first, m, len(runs["rotation"]))

    print("Errors, each form's over Rotation's:")
    print()
    print("| ε | workload | kD | p by level, Rotation → PlaneWave | φ Rot. | ∇φ Rot. | Level φ | ∇φ "
          "| Leaves φ | ∇φ | M2l φ | ∇φ |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        first, m, _ = measures[case]
        rot = m["rotation"]
        eps = case[2].split("= ")[1]
        line = (f"| {eps} | {case[0]} | {case[1]} | {first['rotation']['p by level']} → "
                f"{first['planewave']['p by level']} | {rot['φ']:.2e} | {rot['∇φ']:.2e} |")
        for form in FORMS:
            line += f" {ratio(rot['φ'], m[form]['φ']):.3f} | {ratio(rot['∇φ'], m[form]['∇φ']):.3f} |"
        print(line)
    print()
    print("Times (s; the smaller of the two launches, or of one where marked *), each form's "
          "over Rotation's:")
    print()
    print("| ε | workload | kD | evaluate Rot. | Level | Leaves | M2l | rest Rot. | Level | Leaves "
          "| M2l | M2L Rot. | Level | Leaves | M2l | build Rot. | Level | Leaves | M2l |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
          "| ---: | ---: | ---: | ---: | ---: | ---: |")
    worst = {}
    for case in cases:
        _, m, launches = measures[case]
        rot = m["rotation"]
        eps = case[2].split("= ")[1]
        mark = "" if launches > 1 else " *"
        line = f"| {eps} | {case[0]} | {case[1]}{mark} |"
        for name in ("evaluate", "rest", "M2L", "build"):
            line += f" {rot[name]:.3f} |"
            for form in FORMS:
                line += f" {ratio(rot[name], m[form][name]):.2f} |"
        print(line)
        for form in FORMS:
            for name in ("φ", "∇φ", "M2L", "evaluate", "rest", "build"):
                x = ratio(rot[name], m[form][name])
                if x != x:  # a build below the column's resolution
                    continue
                if launches < 2 and name not in ("φ", "∇φ"):
                    continue  # times of one launch: in the table, not the ranges
                key = (eps, name, NAMES[form])
                lo, hi = worst.get(key, (float("inf"), 0.0))
                worst[key] = (min(lo, x), max(hi, x))
            # M2l and Leaves over Level, the evaluation.
            if form != "planewave" and launches > 1:
                x = ratio(m["planewave"]["evaluate"], m[form]["evaluate"])
                key = (eps, "evaluate ÷ Level's", NAMES[form])
                lo, hi = worst.get(key, (float("inf"), 0.0))
                worst[key] = (min(lo, x), max(hi, x))
    print()
    print("| ε | measure | form | ÷ Rotation (or Level), smallest | largest |")
    print("| --- | --- | --- | ---: | ---: |")
    for (eps, name, form), (lo, hi) in sorted(worst.items()):
        print(f"| {eps} | {name} | {form} | {lo:.3f} | {hi:.3f} |")


if __name__ == "__main__":
    main()
