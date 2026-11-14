#!/usr/bin/env python3
"""tools/helmholtz/ch4-0d3-tables.py: the tables of Helmholtz CH4.0d3's report from the
launches of tools/helmholtz/ch4-0d3-runs.sh (docs/helmholtz/tasks/CH4.0d3.md).

    python3 tools/helmholtz/ch4-0d3-tables.py cases <launch.md>...
    python3 tools/helmholtz/ch4-0d3-tables.py summary <launch.md>...
    python3 tools/helmholtz/ch4-0d3-tables.py checksums <launch.md>...
    python3 tools/helmholtz/ch4-0d3-tables.py wx <launch.md>...

Every number comes from the launches' table "Leaves" (one row per run) and, for
`checksums`, the first table. A run is keyed by its case (workload, N, kD, ε), the V rule
(`off` or the crossover of `/v…`), the W and X rule (`off`, `cost` or the crossover of
`/wx…`), c_leaf and the leaf size; the times are the means over
the launches that ran it (both orders where the stage runs them), the errors the first's.
The reference of a case is the library's default before this task: 64 points per leaf, the
V rule and the W and X rule off, c_leaf 1.

`cases`: per case, rules and c_leaf a row per leaf size: the levels, the leaf rule's level,
the points per occupied leaf, the largest degree, the P2P's share of the evaluation, the
build, the evaluation, (build + evaluation) and the evaluation over the reference's, the
errors over ε, the phase bound over c_leaf; the best size of the row group marked.

`summary`: per ε, rules and c_leaf, for each leaf size the geometric mean over the cases
(that ran the reference and take at least 0.3 s there) of (build + evaluation) and of the
evaluation over the reference's, their range, and the runs whose φ error exceeds ε.

`checksums`: per case and run (the first table's M2L column, the leaf size in it) the
checksums of every launch, and whether they agree (4 × T/4 against 1 × T, two orders).

`wx`: from the per-level rows of the table "Leaf operators" (`--levels`), per degree p the
thread-nanoseconds per P2L and per M2P point interaction and per P2P interaction (the
medians over the rows with at least 10⁵ interactions of the kind), and the least-squares
fits a + b (p + 1)² of P2L and M2P over the rows, the constants of `WxCosts`.
Markdown on stdout.
"""

import math
import re
import statistics
import sys
from collections import defaultdict


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def rule(m2l):
    """The V rule of an M2L column: `off` or the crossover after `/v`."""
    m = re.search(r"/v([^/\s]+)", m2l)
    return m.group(1) if m else "off"


def wx_rule(m2l):
    """The W and X rule of an M2L column: `off`, `cost` or the crossover after `/wx`."""
    m = re.search(r"/wx([^/\s]+)", m2l)
    return m.group(1) if m else "off"


def epsilon(degrees):
    m = re.match(r"ε = (\S+)", degrees)
    return float(m.group(1)) if m else None


def parse(paths):
    """Runs by (case, V rule, W and X rule, c_leaf, size): lists of dicts, one per launch."""
    runs = defaultdict(list)
    sums = defaultdict(list)
    for path in paths:
        section = "main"
        with open(path) as f:
            for line in f:
                if line.startswith("## "):
                    section = line[3:].strip()
                    continue
                if (
                    not line.startswith("| ")
                    or line.startswith("| Workload")
                    or line.startswith("| ---")
                ):
                    continue
                c = cells(line)
                if section == "main" and len(c) >= 20:
                    # Rank 0's waves buffer and chunks (in parentheses) depend on the ranks.
                    m2l = re.sub(r" \([^)]*\)", "", c[7])
                    sums[((c[0], c[1], c[2], c[3]), m2l, c[8])].append((path, c[18]))
                elif section == "Leaves" and len(c) >= 24:
                    case = (c[0], c[1], c[2], c[3])
                    key = (case, rule(c[4]), wx_rule(c[4]), float(c[7]), int(c[6]))
                    # Round 1's table had no P2L, M2P, other kinds or X and W columns.
                    t = 5 if len(c) >= 29 else 0
                    run = {
                        "split": c[5],
                        "levels": int(c[8]),
                        "leaf_level": int(c[9]),
                        "per_leaf": float(c[13]),
                        "p": int(c[14]),
                        "p2p": float(c[15]),
                        "share": float(c[16]),
                        "m2l": float(c[17]),
                        "phase": float(c[19 + t]),
                        "build": float(c[20 + t]),
                        "evaluate": float(c[21 + t]),
                        "errors": (float(c[22 + t]), float(c[23 + t])),
                    }
                    if t:
                        run.update(
                            p2l=float(c[18]),
                            m2p=float(c[19]),
                            other=float(c[20]),
                            x=int(c[21]),
                            w=int(c[22]),
                        )
                    runs[key].append(run)
    return runs, sums


def mean(values):
    return sum(values) / len(values)


def case_order(case):
    workload, n, kd, eps = case
    return (workload, int(n), float(kd.split()[0]), -(epsilon(eps) or 0.0))


def rule_order(v):
    return {"off": -2.0, "cost": -1.0}.get(v, None) or float(v)


def reference(runs, case):
    rows = runs.get((case, "off", "off", 1.0, 64))
    if not rows:
        return None
    return mean([r["build"] for r in rows]), mean([r["evaluate"] for r in rows])


def cases_table(runs):
    print(
        "| Workload | N | kD | ε | V rule | W, X rule | c_leaf | leaf size | levels | leaf level "
        "| points per leaf | largest p | P2P share | build s | evaluate s "
        "| (build + evaluate) ÷ ref | evaluate ÷ ref | φ ÷ ε | ∇φ ÷ ε | Re(k) r ÷ c_leaf |"
    )
    print(
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    groups = defaultdict(dict)
    for (case, v, wx, c_leaf, size), rows in runs.items():
        groups[(case, v, wx, c_leaf)][size] = rows
    for case, v, wx, c_leaf in sorted(
        groups, key=lambda g: (case_order(g[0]), rule_order(g[1]), rule_order(g[2]), g[3])
    ):
        ref = reference(runs, case)
        eps = epsilon(case[3]) or 1.0
        by_size = groups[(case, v, wx, c_leaf)]
        total = {
            s: mean([r["build"] + r["evaluate"] for r in rows]) for s, rows in by_size.items()
        }
        best = min(total, key=total.get)
        for size in sorted(by_size):
            rows = by_size[size]
            r0 = rows[0]
            b, e = mean([r["build"] for r in rows]), mean([r["evaluate"] for r in rows])
            ratio = lambda a, z: f"{a / z:.3f}" if ref and z > 0 else "–"
            mark = " **best**" if size == best and len(by_size) > 1 else ""
            print(
                f"| {case[0]} | {case[1]} | {case[2]} | {case[3]} | {v} | {wx} | {c_leaf:g} "
                f"| {size}{mark} | {r0['levels']} | {r0['leaf_level']} | {r0['per_leaf']:.1f} "
                f"| {r0['p']} | {r0['share']:.2f} | {b:.3f} | {e:.3f} "
                f"| {ratio(b + e, sum(ref) if ref else 0)} | {ratio(e, ref[1] if ref else 0)} "
                f"| {r0['errors'][0] / eps:.2f} | {r0['errors'][1] / eps:.2f} "
                f"| {r0['phase']:.1f} |"
            )


def geomean(values):
    return math.exp(sum(math.log(x) for x in values) / len(values))


def summary_table(runs):
    print(
        "| ε | V rule | W, X rule | c_leaf | leaf size | cases | (build + evaluate) ÷ ref, geometric mean "
        "(range) | evaluate ÷ ref (range) | φ above ε |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: |")
    groups = defaultdict(lambda: defaultdict(list))
    over = defaultdict(int)
    for (case, v, wx, c_leaf, size), rows in runs.items():
        ref = reference(runs, case)
        eps = epsilon(case[3])
        if ref is None or eps is None or sum(ref) < 0.3:
            continue
        b, e = mean([r["build"] for r in rows]), mean([r["evaluate"] for r in rows])
        groups[(eps, v, wx, c_leaf)][size].append(((b + e) / sum(ref), e / ref[1]))
        if rows[0]["errors"][0] > eps:
            over[(eps, v, wx, c_leaf, size)] += 1
    for eps, v, wx, c_leaf in sorted(
        groups, key=lambda g: (-g[0], rule_order(g[1]), rule_order(g[2]), g[3])
    ):
        for size in sorted(groups[(eps, v, wx, c_leaf)]):
            values = groups[(eps, v, wx, c_leaf)][size]
            total = [t for t, _ in values]
            evaluate = [x for _, x in values]
            print(
                f"| {eps:g} | {v} | {wx} | {c_leaf:g} | {size} | {len(values)} "
                f"| {geomean(total):.3f} ({min(total):.2f}–{max(total):.2f}) "
                f"| {geomean(evaluate):.3f} ({min(evaluate):.2f}–{max(evaluate):.2f}) "
                f"| {over[(eps, v, wx, c_leaf, size)]} |"
            )


def checksums_table(sums):
    print("| Workload | N | kD | degrees | M2L | launches | checksums | agree |")
    print("| --- | ---: | --- | --- | --- | ---: | --- | --- |")
    groups = defaultdict(list)
    for (case, m2l, split), entries in sums.items():
        groups[(case, m2l)].extend((split, path, value) for path, value in entries)
    for case, m2l in sorted(groups, key=lambda g: (case_order(g[0]), g[1])):
        entries = groups[(case, m2l)]
        values = sorted({value for _, _, value in entries})
        splits = ", ".join(sorted({split for split, _, _ in entries}))
        print(
            f"| {case[0]} | {case[1]} | {case[2]} | {case[3]} | {m2l} | {len(entries)} ({splits}) "
            f"| {' '.join(values)} | {'yes' if len(values) == 1 else '**no**'} |"
        )


def wx_table(paths):
    rows = defaultdict(lambda: {"p2p": [], "p2l": [], "m2p": []})
    for path in paths:
        section = None
        with open(path) as f:
            for line in f:
                if line.startswith("## "):
                    section = line[3:].strip()
                    continue
                if section != "Leaf operators" or not line.startswith("| ") or "---" in line:
                    continue
                c = cells(line)
                if len(c) < 22 or c[6] in ("level", "Σ"):
                    continue
                p = int(c[7])
                for kind, n, ns in (("p2p", 9, 10), ("p2l", 12, 13), ("m2p", 15, 16)):
                    if c[ns] != "–" and int(c[n]) >= 10**5:
                        rows[p][kind].append(float(c[ns]))
    print("| p | P2P ns (rows) | P2L ns (rows) | M2P ns (rows) |")
    print("| ---: | ---: | ---: | ---: |")
    fit = {"p2l": [], "m2p": []}
    for p in sorted(rows):
        r = rows[p]
        cell = lambda v: f"{statistics.median(v):.1f} ({len(v)})" if v else "–"
        print(f"| {p} | {cell(r['p2p'])} | {cell(r['p2l'])} | {cell(r['m2p'])} |")
        for kind in fit:
            fit[kind].extend(((p + 1) ** 2, y) for y in r[kind])
    for kind, points in fit.items():
        if len(points) < 2:
            continue
        n = len(points)
        sx = sum(x for x, _ in points)
        sy = sum(y for _, y in points)
        sxx = sum(x * x for x, _ in points)
        sxy = sum(x * y for x, y in points)
        b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
        a = (sy - b * sx) / n
        print(f"\n{kind.upper()}: a = {a:.1f} ns, b = {b:.3f} ns per (p + 1)², over {n} rows")
    p2p = [y for r in rows.values() for y in r["p2p"]]
    if p2p:
        print(f"P2P: {statistics.median(p2p):.2f} ns per interaction (median of {len(p2p)} rows)")


def main(argv):
    if len(argv) < 2 or argv[0] not in ("cases", "summary", "checksums", "wx"):
        print(__doc__, file=sys.stderr)
        return 2
    runs, sums = parse(argv[1:])
    {"cases": lambda: cases_table(runs), "summary": lambda: summary_table(runs),
     "checksums": lambda: checksums_table(sums), "wx": lambda: wx_table(argv[1:])}[argv[0]]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
