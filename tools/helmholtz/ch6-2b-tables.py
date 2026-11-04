#!/usr/bin/env python3
"""tools/helmholtz/ch6-2b-tables.py: the tables of Helmholtz CH6.2b's report
(docs/helmholtz/tasks/CH6.2b.md) from the outputs of tools/helmholtz/ch6-2b-runs.sh.

    tools/helmholtz/ch6-2b-tables.py <machine>=<out-dir> [<machine>=<out-dir> ...]

Reads every `<stage>-<case>-<split>-<order>.md` of each directory: the "M2L per level"
table (`helmholtz_accuracy --levels`) and the main table (the errors and checksums). Per
machine, case (workload, kD, ε) and split, each strategy's M2L seconds per level and its
evaluation are the mean of the forward and the reverse launch (ABBA), the spread the two
launches' larger over smaller. Prints Markdown on stdout:

- per case a table of the levels: p, V pairs, the seconds of rotation, batched-fused,
  auto and planewave (in brackets the method a run ran there, where not its own), the
  fastest method the level ran (each method timed by the first strategy that ran it), and
  the exponential M2L over the fastest other method;
- a summary: per machine, case and split the M2L summed over the levels and the
  evaluation per strategy, the per-level best (each level's fastest method, summed) and
  each strategy over it, the largest ABBA spread, and the errors of planewave over
  Rotation's.

Python 3.9 standard library only.
"""

import os
import re
import sys
from collections import defaultdict

STRATEGIES = ["rotation", "batched-fused", "auto", "planewave"]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def strategy_of(setting):
    if setting.startswith("auto"):
        return "auto"
    return setting


def read(path):
    """The per-level rows and the main rows of one launch."""
    levels, main, in_levels = [], [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## M2L per level"):
                in_levels = True
                continue
            if not line.startswith("| ") or line.startswith("| Workload") or "---" in line:
                continue
            c = cells(line)
            if in_levels:
                levels.append(c)
            else:
                main.append(c)
    return levels, main


def main():
    # data[machine][(workload, n, kd, eps, split)][strategy][level] = [seconds…]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    info = defaultdict(dict)  # (machine, key) -> level -> (p, pairs, ran per strategy)
    errors = defaultdict(lambda: defaultdict(dict))  # (machine, key) -> strategy -> φ, ∇φ
    for arg in sys.argv[1:]:
        machine, directory = arg.split("=", 1)
        for name in sorted(os.listdir(directory)):
            m = re.match(r"(matrix|large)-(.+)-(\d+x\d+)-(forward|reverse)\.md$", name)
            if not m:
                continue
            split = m.group(3)
            levels, rows = read(os.path.join(directory, name))
            for c in levels:
                workload, n, kd, eps, setting, rt, level, p, ran, pairs, seconds, _ = c
                key = (workload, n, kd, eps, split)
                s = strategy_of(setting)
                if level == "Σ":
                    evaluate = float(re.match(r"evaluate ([0-9.]+) s", ran).group(1))
                    data[machine][key][s]["evaluate"].append(evaluate)
                    continue
                data[machine][key][s][int(level)].append(float(seconds))
                entry = info[(machine, key)].setdefault(int(level), [p, pairs, {}])
                entry[2][s] = ran
            # The main rows: the strategies in the launch's order, per (kD, ε).
            order = STRATEGIES if m.group(4) == "forward" else STRATEGIES[::-1]
            groups = defaultdict(list)
            for c in rows:
                groups[(c[0], c[1], c[2], c[3])].append(c)
            for (workload, n, kd, eps), group in groups.items():
                for s, c in zip(order, group):
                    key = (workload, n, kd, eps, split)
                    errors[(machine, key)][s] = (float(c[16]), float(c[17]), c[18])

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    def spread(values):
        return max(values) / min(values) if values and min(values) > 0 else float("nan")

    summary = []
    for machine in data:
        print(f"## {machine}\n")
        for key in sorted(data[machine], key=lambda k: (k[0], float(k[2]), -float(k[3].split("= ")[1]), k[4])):
            workload, n, kd, eps, split = key
            per = data[machine][key]
            lv = sorted(info[(machine, key)])
            print(f"### {workload} (N = {n}), kD = {kd}, {eps}, {split.replace('x', ' × ')}\n")
            print("| level | p | V pairs | " + " | ".join(STRATEGIES) + " | fastest | planewave ÷ fastest other |")
            print("| ---: | ---: | ---: | " + " | ".join(["---:"] * len(STRATEGIES)) + " | --- | ---: |")
            best_total, worst_spread = 0.0, 1.0
            for level in lv:
                p, pairs, ran = info[(machine, key)][level]
                times = {s: mean(per[s][level]) for s in STRATEGIES if per[s][level]}
                for s in STRATEGIES:
                    if per[s][level]:
                        worst_spread = max(worst_spread, spread(per[s][level]))
                # The methods the level ran (rotation, batched-fused, planewave iq), each
                # timed by the first strategy that ran it, so that two runs of one method
                # never compete on noise.
                methods = {}
                for s in STRATEGIES:
                    if s in times:
                        methods.setdefault(ran.get(s, s), times[s])
                fastest = min(methods, key=methods.get)
                best_total += methods[fastest]
                pw = [t for m_, t in methods.items() if m_.startswith("planewave")]
                others = [t for m_, t in methods.items() if not m_.startswith("planewave")]
                ratio = pw[0] / min(others) if pw and others else float("nan")
                shown = []
                for s in STRATEGIES:
                    if s not in times:
                        shown.append("–")
                        continue
                    tag = "" if ran.get(s) == s or s == "rotation" else f" ({ran.get(s)})"
                    if s in ("auto", "planewave", "batched-fused") and ran.get(s) == "rotation":
                        tag = " (rot.)"
                    shown.append(f"{times[s]:.4f}{tag}")
                print(f"| {level} | {p} | {pairs} | " + " | ".join(shown) + f" | {fastest} | {ratio:.2f} |")
            sums = {s: sum(mean(per[s][l]) for l in lv if per[s][l]) for s in STRATEGIES}
            evals = {s: mean(per[s]["evaluate"]) for s in STRATEGIES}
            print("| Σ | | | " + " | ".join(f"{sums[s]:.4f}" for s in STRATEGIES) + f" | best {best_total:.4f} | |")
            print("| evaluate | | | " + " | ".join(f"{evals[s]:.3f}" for s in STRATEGIES) + " | | |\n")
            e = errors.get((machine, key), {})
            pw = e.get("planewave"), e.get("rotation")
            ratio = (f"{pw[0][0] / pw[1][0]:.3f}/{pw[0][1] / pw[1][1]:.3f}"
                     if pw[0] and pw[1] else "–")
            summary.append((machine, key, sums, evals, best_total, worst_spread, ratio))
    print("## Summary\n")
    print("| machine | workload | kD | ε | split | M2L Σ rot. | b-f | auto | pw | best per level "
          "| rot./best | b-f/best | auto/best | pw/best | evaluate rot. | b-f | auto | pw "
          "| ABBA max | pw ÷ rot. error φ/∇φ |")
    print("| --- | --- | --- | --- | --- | " + " | ".join(["---:"] * 13) + " | --- |")
    for machine, key, sums, evals, best, worst, ratio in summary:
        workload, n, kd, eps, split = key
        print(f"| {machine} | {workload} | {kd} | {eps.split('= ')[1]} | {split.replace('x', '×')} | "
              + " | ".join(f"{sums[s]:.3f}" for s in STRATEGIES)
              + f" | {best:.3f} | "
              + " | ".join(f"{sums[s] / best:.2f}" for s in STRATEGIES)
              + " | " + " | ".join(f"{evals[s]:.3f}" for s in STRATEGIES)
              + f" | {worst:.2f} | {ratio} |")


if __name__ == "__main__":
    main()
