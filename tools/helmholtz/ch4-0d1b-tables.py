#!/usr/bin/env python3
"""tools/helmholtz/ch4-0d1b-tables.py: the build table of Helmholtz CH4.0d1b's report from
the launches of tools/helmholtz/ch4-0d1b-runs.sh and those of CH4.0d1 on the same cases
(docs/helmholtz/tasks/CH4.0d1b.md).

    python3 tools/helmholtz/ch4-0d1b-tables.py <dir of CH4.0d1b> <dir of CH4.0d1>...

Per case (workload, N, kD, ε) and M2L setting (Rotation, the default): the build with the
empty-box rule on and off, the mean over the launches (both orders), on this branch and in
CH4.0d1's launches; on over off on each; the evaluation with the rule on; and whether every
checksum of the case equals CH4.0d1's (rule on and rule off). Only the cases both
directories hold. Markdown on stdout.
"""

import glob
import os
import sys
from collections import defaultdict


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def rows(directory):
    """(case, setting, on/off) -> lists of (build, evaluate, checksum) from the main tables."""
    found = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*.md"))):
        header = None
        with open(path) as f:
            for line in f:
                if line.startswith("## "):
                    break
                if not line.startswith("|"):
                    continue
                c = cells(line)
                if c[0] == "Workload":
                    header = c
                    continue
                if header is None or c[0].startswith("---"):
                    continue
                row = dict(zip(header, c))
                m2l = row["M2L"]
                off = m2l.endswith("/full")
                setting = "Rotation" if m2l.startswith("rotation") else "default"
                case = (row["Workload"], row["kD"], row["degrees"], row["ranks × threads"])
                found[(case, setting, "off" if off else "on")].append(
                    (float(row["build s"]), float(row["evaluate s"]), row["checksum"])
                )
    return found


def mean(values):
    return sum(values) / len(values)


def main(branch_dir, base_dirs):
    branch = rows(branch_dir)
    base = defaultdict(list)
    for directory in base_dirs:
        for key, values in rows(directory).items():
            base[key].extend(values)
    print(
        "| case | ε | M2L | build on s | build off s | on/off | CH4.0d1 on s | CH4.0d1 off s "
        "| CH4.0d1 on/off | evaluate on s | checksums |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    keys = sorted({(case, setting) for (case, setting, _) in branch})
    for case, setting in keys:
        on, off = branch.get((case, setting, "on")), branch.get((case, setting, "off"))
        base_on, base_off = base.get((case, setting, "on")), base.get((case, setting, "off"))
        if not (on and off and base_on and base_off):
            continue
        b_on, b_off = mean([r[0] for r in on]), mean([r[0] for r in off])
        a_on, a_off = mean([r[0] for r in base_on]), mean([r[0] for r in base_off])
        sums = {r[2] for r in on} | {r[2] for r in off}
        base_sums = {r[2] for r in base_on} | {r[2] for r in base_off}
        same = "equal" if sums == base_sums else f"differ: {sorted(sums)} / {sorted(base_sums)}"
        workload, kd, eps, ranks = case
        print(
            f"| {workload}, kD = {kd}, {ranks} | {eps.replace('ε = ', '')} | {setting} "
            f"| {b_on:.2f} | {b_off:.2f} | {b_on / b_off:.2f} | {a_on:.2f} | {a_off:.2f} "
            f"| {a_on / a_off:.2f} | {mean([r[1] for r in on]):.2f} | {same} |"
        )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
