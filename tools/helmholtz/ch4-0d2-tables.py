#!/usr/bin/env python3
"""tools/helmholtz/ch4-0d2-tables.py: the tables of Helmholtz CH4.0d2's report from the
launches of tools/helmholtz/ch4-0d2-runs.sh (docs/helmholtz/tasks/CH4.0d2.md).

    python3 tools/helmholtz/ch4-0d2-tables.py cases <launch.md>...
    python3 tools/helmholtz/ch4-0d2-tables.py levels <launch.md>...

`cases`: per case (workload, N, kD, ε), M2L setting (Rotation, the default) and V crossover
(`off`, n, `inf`): the V pairs converted, the build and the evaluation (the mean over the
launches, which run both orders where the stage does), each over the rule off's, the errors
over the rule off's, the source points received and the multipoles read from other ranks.

`levels`: per (M2L that ran, degree) over every case and level where the crossover removed V
pairs: the level's M2L thread-microseconds saved per V pair removed, and the P2P's
thread-nanoseconds per near interaction added (the level's P2P seconds × threads over its
interactions, on and off), the medians and ranges over the rows; the cost model's constants.
Markdown on stdout.
"""

import re
import statistics
import sys
from collections import defaultdict


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def setting(m2l):
    """(`rotation` or `default`, the crossover: `off`, a number or `inf`)."""
    name = "rotation" if m2l.startswith("rotation") else "default"
    m = re.search(r"/v(\S+)$", m2l)
    return name, (m.group(1) if m else "off")


def threads(split):
    ranks, t = (int(x) for x in split.split("×"))
    return ranks, t


def parse(paths):
    """Rows by (case, setting, crossover): lists of dicts (one per launch)."""
    main = defaultdict(list)
    levels = defaultdict(list)
    near = defaultdict(list)
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
                if line.startswith("## Near pairs"):
                    section = "near"
                    continue
                if not line.startswith("| ") or line.startswith("| Workload") or line.startswith("| ---"):
                    continue
                c = cells(line)
                if section == "main" and len(c) >= 20:
                    name, v = setting(c[7])
                    key = ((c[0], c[1], c[2], c[3]), name, v)
                    main[key].append(
                        {
                            "split": c[8],
                            "build": float(c[14]),
                            "evaluate": float(c[15]),
                            "errors": (float(c[16]), float(c[17]) if c[17] != "–" else 0.0),
                            "checksum": c[18],
                        }
                    )
                elif section == "levels" and len(c) >= 12 and c[6] != "Σ":
                    name, v = setting(c[4])
                    key = ((c[0], c[1], c[2], c[3]), name, v)
                    levels[key].append(
                        (int(c[6]), int(c[7]), c[8], int(c[9]), float(c[10]), c[5])
                    )
                elif section == "near" and len(c) >= 18:
                    name, v = setting(c[4])
                    key = ((c[0], c[1], c[2], c[3]), name, v)
                    if c[6] == "Σ":
                        near[key].append(
                            {
                                "offered": int(c[7]),
                                "converted": int(c[8]),
                                "remote": int(c[10]),
                                "received": int(c[12]),
                                "split": c[5],
                            }
                        )
                    else:
                        near[(key, "level")].append(
                            (int(c[6]), float(c[13]), int(c[14]), c[5])
                        )
    return main, levels, near


def mean(values):
    return sum(values) / len(values)


def order(v):
    return {"off": -2.0, "cost": -1.0}.get(v) or float(v)


def case_order(case):
    workload, n, kd, eps = case
    return (workload, int(n), float(kd.split()[0]), eps)


def cases_table(main, near):
    print(
        "| Workload | N | kD | ε | M2L | ranks × threads | V crossover | V pairs converted "
        "| build s | evaluate s | build ÷ off | evaluate ÷ off | (build + evaluate) ÷ off "
        "| φ ÷ off | ∇φ ÷ off | points received | remote multipoles |"
    )
    print("| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    groups = defaultdict(dict)
    for (case, name, v), rows in main.items():
        groups[(case, name, rows[0]["split"])][v] = rows
    for (case, name, split) in sorted(groups, key=lambda g: (case_order(g[0]), g[1], g[2])):
        by_v = groups[(case, name, split)]
        if "off" not in by_v:
            continue
        off = by_v["off"]
        b0, e0 = mean([r["build"] for r in off]), mean([r["evaluate"] for r in off])
        f0, g0 = off[0]["errors"]
        for v in sorted(by_v, key=order):
            rows = by_v[v]
            b, e = mean([r["build"] for r in rows]), mean([r["evaluate"] for r in rows])
            f, g = rows[0]["errors"]
            n = near.get((case, name, v), [{}])[0]
            converted = (
                f"{n['converted']} of {n['offered']}" if n and n.get("offered") else "–"
            )
            ratio = lambda a, z: f"{a / z:.3f}" if z > 0 else "–"
            print(
                f"| {case[0]} | {case[1]} | {case[2]} | {case[3]} | {name} | {split} | {v} "
                f"| {converted} | {b:.2f} | {e:.3f} | {ratio(b, b0)} | {ratio(e, e0)} "
                f"| {ratio(b + e, b0 + e0)} | {ratio(f, f0)} | {ratio(g, g0)} "
                f"| {n.get('received', '–')} | {n.get('remote', '–')} |"
            )


def levels_table(levels, near):
    m2l = defaultdict(list)
    p2p = defaultdict(list)
    for (case, name, v), rows in levels.items():
        if v == "off" or (case, name, "off") not in levels:
            continue
        off = {(l, s): (p, ran, pairs, sec) for (l, p, ran, pairs, sec, s) in levels[(case, name, "off")]}
        on_near = {(l, s): (sec, n) for (l, sec, n, s) in near.get(((case, name, v), "level"), [])}
        off_near = {(l, s): (sec, n) for (l, sec, n, s) in near.get(((case, name, "off"), "level"), [])}
        for (l, p, ran, pairs, sec, s) in rows:
            if (l, s) not in off:
                continue
            p0, ran0, pairs0, sec0 = off[(l, s)]
            ranks, t = threads(s)
            if ran0 != ran or pairs0 - pairs < max(100, pairs0 // 20):
                continue
            m2l[(ran, p)].append((sec0 - sec) * t * 1e6 / (pairs0 - pairs))
            if (l, s) in on_near and (l, s) in off_near:
                (ps, n), (ps0, n0) = on_near[(l, s)], off_near[(l, s)]
                if n - n0 > 10**6:
                    p2p[(ran, p)].append((ps - ps0) * t * 1e9 / (n - n0))
    print(
        "| M2L that ran | p | rows | M2L µs saved per pair removed (median, range) "
        "| P2P ns per interaction added (median, range) | crossover n_s n_t (medians) |"
    )
    print("| --- | ---: | ---: | --- | --- | ---: |")
    for key in sorted(m2l, key=lambda k: (k[0], k[1])):
        a = m2l[key]
        b = p2p.get(key, [])
        ma = statistics.median(a)
        mb = statistics.median(b) if b else None
        cross = f"{ma * 1e3 / mb:.0f}" if mb else "–"
        brange = f"{mb:.2f} ({min(b):.2f}–{max(b):.2f})" if b else "–"
        print(
            f"| {key[0]} | {key[1]} | {len(a)} | {ma:.2f} ({min(a):.2f}–{max(a):.2f}) "
            f"| {brange} | {cross} |"
        )


def main(argv):
    if len(argv) < 2 or argv[0] not in ("cases", "levels"):
        print(__doc__, file=sys.stderr)
        return 2
    main_rows, levels, near = parse(argv[1:])
    if argv[0] == "cases":
        cases_table(main_rows, near)
    else:
        levels_table(levels, near)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
