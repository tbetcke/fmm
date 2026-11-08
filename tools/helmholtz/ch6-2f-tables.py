#!/usr/bin/env python3
"""tools/helmholtz/ch6-2f-tables.py: the tables of Helmholtz CH6.2f's report
(fmm-validate/results/helmholtz-ch6-2f.md) from the outputs of tools/helmholtz/ch6-2f-runs.sh.

    tools/helmholtz/ch6-2f-tables.py <machine>=<out-dir> …

Reads every `coarse-*.md` of each directory (the `--levels` table of `helmholtz_accuracy`),
and prints per machine and split (ranks x threads):

1. per case, ε and level with waves: Rotation's M2L seconds, the exponential M2L's under the
   split by parents, by child and Auto, with the threads busy, and the ratios;
2. per case and ε: the levels' M2L summed and the evaluation, the same four runs;
3. per case, ε and level with waves, the split by parents' parts: busy thread-seconds and
   the shares of the shifts, the children's sums and the conversions, and the split by
   child's busy over the parents' (the repeated shifts).

Each figure is the mean of the launch's two orders (forward and reverse), the spread
(max - min over mean) printed where it exceeds 10 %.
"""

import collections
import glob
import os
import re
import sys

SETTINGS = ["rotation", "planewave/parents", "planewave/children", "planewave"]
NAMES = {"rotation": "Rotation", "planewave/parents": "parents",
         "planewave/children": "child", "planewave": "Auto"}


def read(directory):
    """{(case, split): {(eps, setting, level): [rows]}}, a row the level line's columns."""
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(glob.glob(os.path.join(directory, "coarse-*.md"))):
        match = re.match(r"coarse-(.+)-(\d+x\d+)-(forward|reverse)\.md", os.path.basename(path))
        if not match:
            continue
        case, split = match.group(1), match.group(2)
        in_levels = False
        for line in open(path):
            if line.startswith("## M2L per level"):
                in_levels = True
                continue
            if not in_levels or not line.startswith("| W") or line.startswith("| Workload"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            eps = cells[3].replace("ε = ", "")
            setting, level = cells[4], cells[6]
            runs[(case, split)][(eps, setting, level)].append(cells)
    return runs


def number(cell):
    try:
        return float(cell)
    except ValueError:
        return None


def mean(rows, column):
    values = [number(r[column]) for r in rows]
    values = [v for v in values if v is not None]
    if not values:
        return None, 0.0
    m = sum(values) / len(values)
    spread = (max(values) - min(values)) / m if m > 0 else 0.0
    return m, spread


def fmt(value, spread=0.0, digits=4):
    if value is None:
        return "–"
    text = f"{value:.{digits}f}"
    return text + (f" (±{100 * spread / 2:.0f} %)" if spread > 0.10 else "")


def ratio(a, b):
    return "–" if a is None or b is None or b == 0 else f"{a / b:.2f}"


def evaluate_seconds(rows):
    values = []
    for r in rows:
        match = re.search(r"evaluate ([0-9.]+) s", r[8])
        if match:
            values.append(float(match.group(1)))
    if not values:
        return None, 0.0
    m = sum(values) / len(values)
    return m, (max(values) - min(values)) / m


def tables(machine, runs):
    for (case, split) in sorted(runs):
        data = runs[(case, split)]
        epsilons = sorted({k[0] for k in data}, key=float, reverse=True)
        print(f"\n### {machine}, {case}, {split.replace('x', ' × ')}\n")
        print("| ε | level | p | V pairs | Rotation s | parents s (busy) | child s (busy) "
              "| Auto s (busy) | child ÷ parents | Auto ÷ Rotation | Auto ÷ parents |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for eps in epsilons:
            levels = sorted({k[2] for k in data if k[0] == eps and k[2] != "Σ"
                             and k[1] == "planewave/parents"
                             and data[k][0][8].startswith("planewave")}, key=int)
            for level in levels:
                rot = data.get((eps, "rotation", level), [])
                m = {s: mean(data.get((eps, s, level), []), 10) for s in SETTINGS}
                busy = {s: mean(data.get((eps, s, level), []), 19)[0] for s in SETTINGS}
                first = data[(eps, "planewave/parents", level)][0]
                cells = [eps, level, first[7], first[9], fmt(*m["rotation"])]
                for s in SETTINGS[1:]:
                    b = "–" if busy[s] is None else f"{busy[s]:.1f}"
                    cells.append(f"{fmt(*m[s])} ({b})")
                cells += [ratio(m["planewave/children"][0], m["planewave/parents"][0]),
                          ratio(m["planewave"][0], m["rotation"][0]),
                          ratio(m["planewave"][0], m["planewave/parents"][0])]
                del rot
                print("| " + " | ".join(cells) + " |")
        print("\n| ε | run | M2L Σ s | evaluate s |")
        print("| --- | --- | ---: | ---: |")
        for eps in epsilons:
            for s in SETTINGS:
                rows = data.get((eps, s, "Σ"), [])
                if rows:
                    print(f"| {eps} | {NAMES[s]} | {fmt(*mean(rows, 10))} "
                          f"| {fmt(*evaluate_seconds(rows))} |")
        print("\n| ε | level | parents: busy s | shifts | children | conversions "
              "| child: busy s | child busy ÷ parents' | child's shifts ÷ parents' |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for eps in epsilons:
            levels = sorted({k[2] for k in data if k[0] == eps and k[2] != "Σ"
                             and k[1] == "planewave/parents"
                             and data[k][0][8].startswith("planewave")}, key=int)
            for level in levels:
                parents = data[(eps, "planewave/parents", level)]
                child = data.get((eps, "planewave/children", level), [])
                busy, _ = mean(parents, 15)
                parts = [mean(parents, c)[0] for c in (16, 17, 18)]
                cbusy, _ = mean(child, 15)
                cshifts, _ = mean(child, 16)
                shares = ["–" if busy in (None, 0) or p is None else f"{100 * p / busy:.0f} %"
                          for p in parts]
                print(f"| {eps} | {level} | {fmt(busy)} | " + " | ".join(shares)
                      + f" | {fmt(cbusy)} | {ratio(cbusy, busy)} | {ratio(cshifts, parts[0])} |")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    for arg in sys.argv[1:]:
        machine, directory = arg.split("=", 1)
        tables(machine, read(directory))


if __name__ == "__main__":
    main()
