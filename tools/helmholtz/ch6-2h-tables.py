#!/usr/bin/env python3
"""tools/helmholtz/ch6-2h-tables.py: the tables of Helmholtz CH6.2h's report
(docs/helmholtz/tasks/CH6.2h.md) from the outputs of tools/helmholtz/ch6-2h-runs.sh.

    tools/helmholtz/ch6-2h-tables.py [--extra auto-pw:16] <machine>=<out-dir> [...]

Reads every `<stage>-<case>-<split>-<order>.md` of each directory. The strategies are the
launch's `--m2l` entries in order, `rotation,auto,auto-pw,planewave` and the `--extra`
ones (a directory without them has none); a launch's rows and level rows come in that order
(reversed in the reverse launch), so each setting is told apart by its position. Per
machine, case (workload, kD, ε) and split, each strategy's M2L seconds per level and its
evaluation are the mean of the forward and the reverse launch (ABBA). Prints Markdown:

- per case a table of the levels: p, V pairs, each strategy's seconds (in brackets the
  method it ran there, where not its own), the fastest method the level ran (each method
  timed by the first strategy that ran it), and auto-pw (the rule) over the fastest;
- a summary per machine, case and split: the M2L summed over the levels per strategy, the
  per-level best and auto-pw over it and over auto, the evaluation per strategy and
  auto-pw's over auto's, the largest ABBA spread, auto-pw's errors over Rotation's and
  whether its checksum is one of the other strategies';
- the levels where auto-pw's method is more than 1.10× the level's fastest, with the time
  lost and its share of the case's per-level best.

Python 3.9 standard library only.
"""

import os
import re
import sys
from collections import defaultdict

BASE = ["rotation", "auto", "auto-pw", "planewave"]


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


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
            (levels if in_levels else main).append(cells(line))
    return levels, main


def short(ran):
    return {"rotation": "rot.", "batched-fused": "b-f"}.get(ran, ran.replace("planewave ", "pw "))


def main():
    args = sys.argv[1:]
    extra = []
    if args and args[0] == "--extra":
        extra = args[1].split(",")
        args = args[2:]
    # data[machine][key][strategy][level] = [seconds…]; key = (workload, n, kd, eps, split)
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    info = defaultdict(dict)  # (machine, key) -> level -> [p, pairs, {strategy: ran}]
    rows = defaultdict(dict)  # (machine, key) -> strategy -> (φ, ∇φ, checksum, label)
    strategies_of = {}
    for arg in args:
        machine, directory = arg.split("=", 1)
        for name in sorted(os.listdir(directory)):
            m = re.match(r"(matrix|large)-(.+)-(\d+x\d+)-(forward|reverse)\.md$", name)
            if not m:
                continue
            split = m.group(3)
            levels, main_rows = read(os.path.join(directory, name))
            groups = defaultdict(list)
            for c in main_rows:
                groups[(c[0], c[1], c[2], c[3])].append(c)
            n_runs = len(next(iter(groups.values()))) if groups else 0
            strategies = BASE + (extra if n_runs > len(BASE) else [])
            strategies_of[machine] = strategies
            order = strategies if m.group(4) == "forward" else strategies[::-1]
            for (workload, n, kd, eps), group in groups.items():
                if len(group) != len(order):
                    sys.exit(f"{name}: {len(group)} rows for {len(order)} strategies")
                for s, c in zip(order, group):
                    rows[(machine, (workload, n, kd, eps, split))][s] = (
                        float(c[16]), float(c[17]), c[18], c[7])
            # The level rows: one block per run, each ending with its Σ row.
            run_index = defaultdict(int)
            for c in levels:
                workload, n, kd, eps, setting, rt, level, p, ran, pairs, seconds, per_pair = c[:12]
                key = (workload, n, kd, eps, split)
                s = order[run_index[key]]
                if level == "Σ":
                    evaluate = float(re.match(r"evaluate ([0-9.]+) s", ran).group(1))
                    data[machine][key][s]["evaluate"].append(evaluate)
                    run_index[key] += 1
                    continue
                data[machine][key][s][int(level)].append(int(pairs) * float(per_pair) * 1e-9)
                entry = info[(machine, key)].setdefault(int(level), [p, pairs, {}])
                entry[2][s] = ran

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    def ratio_of(a, b):
        return a / b if b > 0 else float("nan")

    def spread(values):
        return max(values) / min(values) if values and min(values) > 0 else float("nan")

    summary, losses = [], []
    for machine in data:
        strategies = strategies_of[machine]
        # The first strategy to time a method: Rotation, Auto (Batched), PlaneWave, then the rule.
        timing_order = ["rotation", "auto", "planewave", "auto-pw"] + [
            s for s in strategies if s not in BASE]
        print(f"## {machine}\n")
        keys = sorted(data[machine],
                      key=lambda k: (k[0], float(k[2]), -float(k[3].split("= ")[1]), k[4]))
        for key in keys:
            workload, n, kd, eps, split = key
            per = data[machine][key]
            lv = sorted(info[(machine, key)])
            title = workload if "N =" in workload else f"{workload} (N = {n})"
            for s in strategies:
                if len(per[s]["evaluate"]) != 2:
                    print(f"warning: {machine} {title} kD = {kd} {eps} {split}: {s} in "
                          f"{len(per[s]['evaluate'])} launches, not 2", file=sys.stderr)
            print(f"### {title}, kD = {kd}, {eps}, {split.replace('x', ' × ')}\n")
            print("| level | p | V pairs | " + " | ".join(strategies)
                  + " | fastest | auto-pw ÷ fastest |")
            print("| ---: | ---: | ---: | " + " | ".join(["---:"] * len(strategies)) + " | --- | ---: |")
            best_total, worst = 0.0, 1.0
            level_rows = []
            for level in lv:
                p, pairs, ran = info[(machine, key)][level]
                times = {s: mean(per[s][level]) for s in strategies if per[s][level]}
                for s in strategies:
                    if per[s][level]:
                        worst = max(worst, spread(per[s][level]))
                methods = {}
                for s in timing_order:
                    if s in times:
                        methods.setdefault(ran.get(s, s), times[s])
                fastest = min(methods, key=methods.get)
                best_total += methods[fastest]
                rule_method = ran.get("auto-pw")
                rule_time = methods.get(rule_method, float("nan"))
                r = ratio_of(rule_time, methods[fastest])
                level_rows.append((level, p, pairs, rule_method, fastest, r,
                                   rule_time - methods[fastest]))
                shown = []
                for s in strategies:
                    if s not in times:
                        shown.append("–")
                        continue
                    own = {"rotation": "rotation", "planewave": "planewave i2"}.get(s)
                    tag = "" if ran.get(s) == own else f" ({short(ran.get(s))})"
                    shown.append(f"{times[s]:.4f}{tag}")
                print(f"| {level} | {p} | {pairs} | " + " | ".join(shown)
                      + f" | {short(fastest)} | {r:.2f} |")
            sums = {s: sum(mean(per[s][l]) for l in lv if per[s][l]) for s in strategies}
            evals = {s: mean(per[s]["evaluate"]) for s in strategies}
            print("| Σ | | | " + " | ".join(f"{sums[s]:.4f}" for s in strategies)
                  + f" | best {best_total:.4f} | |")
            print("| evaluate | | | " + " | ".join(f"{evals[s]:.3f}" for s in strategies)
                  + " | | |\n")
            for level, p, pairs, rule_method, fastest, r, lost in level_rows:
                if r > 1.10:
                    losses.append((machine, key, level, p, pairs, rule_method, fastest, r, lost,
                                   ratio_of(lost, best_total)))
            e = rows.get((machine, key), {})
            rule, rot = e.get("auto-pw"), e.get("rotation")
            err = (f"{ratio_of(rule[0], rot[0]):.3f}/{ratio_of(rule[1], rot[1]):.3f}"
                   if rule and rot else "–")
            same = ([s for s in strategies if s != "auto-pw" and e.get(s) and rule
                     and e[s][2] == rule[2]] if rule else [])
            summary.append((machine, key, sums, evals, best_total, worst, err,
                            ",".join(same) or "–"))
    print("## Summary\n")
    all_strategies = max(strategies_of.values(), key=len)
    print("| machine | workload | kD | ε | split | M2L Σ " + " | ".join(all_strategies)
          + " | best per level | auto-pw/best | auto-pw/auto | evaluate "
          + " | ".join(all_strategies) + " | auto-pw/auto | ABBA max | auto-pw ÷ rot. error φ/∇φ "
          "| auto-pw's checksum also |")
    print("| --- | --- | --- | --- | --- | " + " | ".join(["---:"] * (2 * len(all_strategies) + 5))
          + " | --- | --- |")
    for machine, key, sums, evals, best, worst, err, same in summary:
        workload, n, kd, eps, split = key
        cell = lambda d, s, f: f.format(d[s]) if s in d else "–"
        print(f"| {machine} | {workload} | {kd} | {eps.split('= ')[1]} | {split.replace('x', '×')} | "
              + " | ".join(cell(sums, s, "{:.3f}") for s in all_strategies)
              + f" | {best:.3f} | {ratio_of(sums['auto-pw'], best):.2f}"
              + f" | {ratio_of(sums['auto-pw'], sums['auto']):.2f} | "
              + " | ".join(cell(evals, s, "{:.3f}") for s in all_strategies)
              + f" | {ratio_of(evals['auto-pw'], evals['auto']):.2f} | {worst:.2f} | {err} | {same} |")
    print("\n## The rule's levels above 1.10× the fastest\n")
    print("| machine | workload | kD | ε | split | level | p | V pairs | auto-pw ran | fastest "
          "| ratio | lost s | lost ÷ case best |")
    print("| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |")
    for machine, key, level, p, pairs, rule_method, fastest, r, lost, share in losses:
        workload, n, kd, eps, split = key
        print(f"| {machine} | {workload} | {kd} | {eps.split('= ')[1]} | {split.replace('x', '×')} "
              f"| {level} | {p} | {pairs} | {short(rule_method or '–')} | {short(fastest)} "
              f"| {r:.2f} | {lost:.4f} | {share:.3f} |")


if __name__ == "__main__":
    main()
