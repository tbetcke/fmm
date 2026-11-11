#!/usr/bin/env python3
"""tools/helmholtz/ch6-1g-tables.py: the tables of Helmholtz CH6.1g's report from the
outputs of tools/helmholtz/ch6-1g-runs.sh (docs/helmholtz/tasks/CH6.1g.md).

    python3 tools/helmholtz/ch6-1g-tables.py <out-dir>

Per case (workload, kD, ε): the degrees by level of Rotation and of PlaneWave by the rule,
the errors of φ and ∇φ and PlaneWave's over Rotation's, and the times of the two launches
(forward and reverse order; the smaller of each): the M2L summed over the levels, the
evaluation, the rest of the evaluation (evaluation − M2L) and the build, PlaneWave's over
Rotation's. The checksums of the two launches must agree (the output is deterministic).
"""

import glob
import os
import sys


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(path):
    """Rows of the main table and Σ rows of the level table, keyed by (case, m2l)."""
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
            m2l = row["M2L"].split()[0]
            if "level" in row:
                if row["level"] == "Σ":
                    evaluate = float(row["ran"].split()[1])
                    sums[(case, m2l)] = (float(row["M2L s"]), evaluate)
            else:
                main[(case, m2l)] = row
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
    for (case, m2l) in rows:
        if case not in cases:
            cases.append(case)
    order = {"1e-7": 0, "1e-8": 1}
    cases.sort(key=lambda c: (order[c[2].split("= ")[1]], c[0], float(c[1].split()[0])))

    print("| ε | workload | kD | p by level, Rotation → PlaneWave | φ Rot. | φ PW | ÷ | ∇φ Rot. | "
          "∇φ PW | ÷ | M2L s Rot. | PW | ÷ | evaluate s Rot. | PW | ÷ | rest s Rot. | PW | ÷ | "
          "build ÷ |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
          "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    worst = {}
    for case in cases:
        rot, pw = rows[(case, "rotation")], rows[(case, "planewave")]
        for runs in (rot, pw):
            sums = {r[0]["checksum"] for r in runs}
            assert len(sums) == 1, (case, sums)

        def best(runs, i):
            return min(r[i] for r in runs)

        def build(runs):
            return min(float(r[0]["build s"]) for r in runs)

        r0, p0 = rot[0][0], pw[0][0]
        phi = (float(r0["φ"]), float(p0["φ"]))
        grad = (float(r0["∇φ"]), float(p0["∇φ"]))
        m2l = (best(rot, 1), best(pw, 1))
        ev = (best(rot, 2), best(pw, 2))
        rest = (min(r[2] - r[1] for r in rot), min(r[2] - r[1] for r in pw))
        bl = (build(rot), build(pw))

        def ratio(x):
            return x[1] / x[0] if x[0] > 0 else float("nan")

        eps = case[2].split("= ")[1]
        for name, x in (("φ", phi), ("∇φ", grad), ("M2L", m2l), ("evaluate", ev), ("rest", rest)):
            lo, hi = worst.get((eps, name), (float("inf"), 0.0))
            worst[(eps, name)] = (min(lo, ratio(x)), max(hi, ratio(x)))
        print(f"| {eps} | {case[0]} | {case[1]} | {r0['p by level']} → {p0['p by level']} | "
              f"{phi[0]:.2e} | {phi[1]:.2e} | {ratio(phi):.3f} | {grad[0]:.2e} | {grad[1]:.2e} | "
              f"{ratio(grad):.3f} | {m2l[0]:.3f} | {m2l[1]:.3f} | {ratio(m2l):.2f} | {ev[0]:.3f} | "
              f"{ev[1]:.3f} | {ratio(ev):.2f} | {rest[0]:.3f} | {rest[1]:.3f} | {ratio(rest):.2f} | "
              f"{ratio(bl):.2f} |")
    print()
    print("| ε | measure | PlaneWave ÷ Rotation, smallest | largest |")
    print("| --- | --- | ---: | ---: |")
    for (eps, name), (lo, hi) in sorted(worst.items()):
        print(f"| {eps} | {name} | {lo:.3f} | {hi:.3f} |")


if __name__ == "__main__":
    main()
