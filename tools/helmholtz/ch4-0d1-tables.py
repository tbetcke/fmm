#!/usr/bin/env python3
"""tools/helmholtz/ch4-0d1-tables.py: the tables of Helmholtz CH4.0d1's report from the
launches of tools/helmholtz/ch4-0d1-runs.sh (docs/helmholtz/tasks/CH4.0d1.md).

    python3 tools/helmholtz/ch4-0d1-tables.py <launch.md>...

Per case (workload, N, kD, ε) and M2L setting (Rotation, the default): the build and the
evaluation with the empty-box rule on and off (the mean over the launches, which run both
orders), on over off, whether the checksums and the errors agree, the V pairs on and off and
the levels whose M2L differs between on and off (from `--levels`' table). Markdown on stdout.
"""

import re
import sys
from collections import defaultdict


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def setting(m2l):
    """`rotation` or `default` (Auto, with or without waves), and whether the rule is off."""
    off = m2l.endswith("/full")
    return ("rotation" if m2l.startswith("rotation") else "default"), off


def main(paths):
    runs = defaultdict(lambda: {"on": [], "off": []})
    pairs = {}
    levels = defaultdict(dict)
    for path in paths:
        section = "main"
        with open(path) as f:
            for line in f:
                if line.startswith("## M2L per level"):
                    section = "levels"
                    continue
                if line.startswith("## Empty boxes"):
                    section = "empty"
                    continue
                if not line.startswith("| ") or line.startswith("| Workload") or line.startswith("| ---"):
                    continue
                c = cells(line)
                if section == "main" and len(c) >= 20:
                    name, rule_off = setting(c[7])
                    key = (c[0], c[1], c[2], c[3], name)
                    runs[key]["off" if rule_off else "on"].append(
                        {
                            "build": float(c[14]),
                            "evaluate": float(c[15]),
                            "errors": (c[16], c[17]),
                            "checksum": c[18],
                            "split": c[8],
                        }
                    )
                elif section == "empty" and len(c) >= 17:
                    name = "rotation" if c[4].startswith("rotation") else "default"
                    key = (c[0], c[1], c[2], c[3], name)
                    pairs[(key, c[6])] = int(c[15])
                elif section == "levels" and len(c) >= 12 and c[6] != "Σ":
                    name, rule_off = setting(c[4])
                    key = (c[0], c[1], c[2], c[3], name)
                    levels[(key, "off" if rule_off else "on")][int(c[6])] = (c[8], float(c[10]))
    print(
        "| Workload | N | kD | ε | M2L | ranks × threads | build on / off s | evaluate on / off s "
        "| on ÷ off (evaluate) | on ÷ off (build + evaluate) | same checksum | same errors "
        "| V pairs on / off | levels whose M2L differs |"
    )
    print("| --- | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for key in sorted(runs, key=lambda k: (k[0], int(k[1]), float(k[2].split()[0]), k[3], k[4])):
        on, off = runs[key]["on"], runs[key]["off"]
        if not on or not off:
            continue
        mean = lambda rows, f: sum(r[f] for r in rows) / len(rows)
        b_on, b_off = mean(on, "build"), mean(off, "build")
        e_on, e_off = mean(on, "evaluate"), mean(off, "evaluate")
        same_sum = {r["checksum"] for r in on} == {r["checksum"] for r in off}
        same_err = {r["errors"] for r in on} == {r["errors"] for r in off}
        lv_on, lv_off = levels.get((key, "on"), {}), levels.get((key, "off"), {})
        differ = [str(l) for l in sorted(set(lv_on) | set(lv_off)) if lv_on.get(l, ("–",))[0] != lv_off.get(l, ("–",))[0]]
        ratio = lambda a, b: f"{a / b:.3f}" if b > 0 else "–"
        degrees = key[3].replace("ε = ", "")
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {degrees} | {key[4]} | {on[0]['split']} "
            f"| {b_on:.2f} / {b_off:.2f} | {e_on:.2f} / {e_off:.2f} | {ratio(e_on, e_off)} "
            f"| {ratio(b_on + e_on, b_off + e_off)} | {'yes' if same_sum else 'no'} "
            f"| {'yes' if same_err else 'no'} "
            f"| {pairs.get((key, 'on'), '–')} / {pairs.get((key, 'off'), '–')} "
            f"| {', '.join(differ) or '–'} |"
        )
    # Per level, the M2L seconds on and off, where a level's M2L takes at least 10 % of the
    # rule off's evaluation.
    print()
    print("| Workload | N | kD | ε | M2L | level | ran on / off | M2L s on / off |")
    print("| --- | ---: | --- | --- | --- | ---: | --- | --- |")
    for key in sorted(runs, key=lambda k: (k[0], int(k[1]), float(k[2].split()[0]), k[3], k[4])):
        off_rows = runs[key]["off"]
        if not off_rows:
            continue
        e_off = sum(r["evaluate"] for r in off_rows) / len(off_rows)
        lv_on, lv_off = levels.get((key, "on"), {}), levels.get((key, "off"), {})
        for l in sorted(lv_off):
            if lv_off[l][1] >= 0.1 * e_off and e_off >= 0.5:
                ran_on, s_on = lv_on.get(l, ("–", 0.0))
                print(
                    f"| {key[0]} | {key[1]} | {key[2]} | {key[3].replace('ε = ', '')} | {key[4]} | {l} "
                    f"| {ran_on} / {lv_off[l][0]} | {s_on:.3f} / {lv_off[l][1]:.3f} |"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
