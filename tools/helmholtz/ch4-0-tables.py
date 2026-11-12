#!/usr/bin/env python3
"""tools/helmholtz/ch4-0-tables.py: the tables of Helmholtz CH4.0's report
(docs/helmholtz/tasks/CH4.0.md) from the outputs of tools/helmholtz/ch4-0-runs.sh.

    tools/helmholtz/ch4-0-tables.py pairs <machine>=<out-dir> [...]
    tools/helmholtz/ch4-0-tables.py fmm <machine>=<out-dir> [...]
    tools/helmholtz/ch4-0-tables.py rule <machine>:<isa>=<out-dir> [...]

`pairs` reads every `pairs-<isa>-c<c>-<a|b>.md` (helmholtz_m2l at c pairs per offset) and
prints, per machine, ISA and planes, Rotation's time per pair over fused Batched's at each
column count (the mean of the two launches' ratios), and m*, the smallest count at which
it is at least 1.1 (`–` where none is).

`fmm` reads every `<stage>-<case>-<forward|reverse>.md` (helmholtz_accuracy with `--levels`)
and prints, per machine, case (workload, kD, ε, charges) and run (the M2L column's setting,
`/no-term` and `/wN` included), each run's M2L seconds per level and its evaluation as the
mean of the two launches; then per case the per-level best (each level's fastest of
Rotation, Batched at either width and the waves where the rule ran them), the default's
(the term on, the library's widths) and the old rule's (the term off, blocks of 64; on
`ch3-2d` the term off) M2L sums and evaluations over it, the largest spread between the two
launches, and the levels where the default's method is more than 1.10× the level's
fastest, with m̂ and the exact columns per read.

`rule` reads the `ch3-2d` launches (one thread) of each directory and evaluates, per case, the
rule with the term at the constants of `MIN_COLUMNS` (which mirror
`nd_fmm_exec::helmholtz::auto_batched_min_columns`) and without it (CH3.2d's rule) on the
measured per-level times: a level runs Batched where the rule says so and Batched ran there
(its tables fit the budget), else Rotation, each at its time in the Rotation and the
Batched run (the mean of the two launches); the evaluation is Rotation's with its M2L
replaced. Against the per-level best of the two, as CH3.2d evaluated its crossovers. Python
3.9 standard library only.
"""

import os
import re
import sys
from collections import defaultdict

MARGIN = 1.1


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def pairs(machines):
    for machine, directory in machines:
        # (isa, planes, p) -> {c: [ratio, ...]}
        ratios = defaultdict(lambda: defaultdict(list))
        for name in sorted(os.listdir(directory)):
            m = re.fullmatch(r"pairs-(\w+)-c(\d+)-[ab]\.md", name)
            if not m:
                continue
            isa, c = m.group(1), int(m.group(2))
            with open(os.path.join(directory, name), encoding="utf-8") as f:
                for line in f:
                    if not re.match(r"\| \d+ \|", line):
                        continue
                    row = cells(line)
                    p, planes = int(row[0]), int(row[2])
                    rotation, fused = float(row[3]), float(row[18])
                    ratios[(isa, planes, p)][c].append(rotation / fused)
        isas = sorted({key[0] for key in ratios})
        for isa in isas:
            for planes in (1, 2):
                keys = sorted(k for k in ratios if k[0] == isa and k[1] == planes)
                if not keys:
                    continue
                columns = sorted({c for k in keys for c in ratios[k]})
                print(f"### {machine}, {isa}, {planes} plane{'s' if planes == 2 else ''}")
                print()
                print("Rotation's time per pair over fused Batched's at c pairs per offset "
                      f"(the mean of {max(len(v) for k in keys for v in ratios[k].values())} "
                      "launches); m* the smallest c at which it is at least 1.1.")
                print()
                print("| p | " + " | ".join(f"c = {c}" for c in columns) + " | m* |")
                print("| ---: |" + " ---: |" * (len(columns) + 1))
                for key in keys:
                    means = {c: sum(v) / len(v) for c, v in ratios[key].items()}
                    need = next((c for c in columns if means.get(c, 0) >= MARGIN), None)
                    print(f"| {key[2]} | "
                          + " | ".join(f"{means[c]:.2f}" if c in means else "–" for c in columns)
                          + f" | {need if need is not None else '–'} |")
                print()


def read(path):
    """The main rows and the level rows of one launch."""
    levels, main, in_levels = [], [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("## M2L per level"):
                in_levels = True
                continue
            if not line.startswith("| ") or line.startswith("| Workload") or "---" in line:
                continue
            (levels if in_levels else main).append(cells(line))
    return main, levels


def setting(m2l):
    """A run's name from the main table's M2L column: what precedes `: ` (Auto's name with
    its suffixes), else the column with the levels dropped."""
    if ": " in m2l:
        return m2l.split(": ")[0]
    head = m2l.split(" ")[0]
    suffix = "".join(re.findall(r"/w\d+", m2l))
    return head + suffix


def method(ran):
    return "pw" if ran.startswith("planewave") else ran


def fmm(machines):
    summary = []
    for machine, directory in machines:
        # (stage, case key) -> run -> {"levels": {l: [s, ...]}, "eval": [...], ...}
        data = defaultdict(lambda: defaultdict(lambda: {
            "levels": defaultdict(list), "eval": [], "p": {}, "ran": {}, "pairs": {},
            "mhat": {}, "exact": {}, "errors": None}))
        for name in sorted(os.listdir(directory)):
            m = re.fullmatch(r"(ch3-2d|matrix|large)-(.+)-(forward|reverse)\.md", name)
            if not m:
                continue
            stage = m.group(1)
            charges = "real" if m.group(2).endswith("real") else "complex"
            main, levels = read(os.path.join(directory, name))
            # The level rows carry the setting by name; the main rows give the errors.
            for row in main:
                key = (stage, row[0], row[2], row[3], charges)
                run = data[key][setting(row[7])]
                run["errors"] = (row[16], row[17])
            for row in levels:
                if len(row) < 24:
                    continue
                key = (stage, row[0], row[2], row[3], charges)
                run = data[key][row[4]]
                if row[6] == "Σ":
                    m = re.search(r"evaluate ([0-9.]+) s", row[8])
                    run["eval"].append(float(m.group(1)))
                    continue
                level = int(row[6])
                run["levels"][level].append(float(row[10]))
                run["p"][level] = int(row[7])
                run["ran"][level] = method(row[8])
                run["pairs"][level] = int(row[9])
                run["mhat"][level] = float(row[21])
                run["exact"][level] = float(row[23])
        print(f"## {machine}")
        print()
        for key in sorted(data):
            runs = data[key]
            stage = key[0]
            mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
            spread = max((max(v) / min(v) for r in runs.values() for v in r["levels"].values()
                          if len(v) > 1 and min(v) > 0), default=1.0)
            if stage == "ch3-2d":
                new = next((r for r in runs if r.startswith("auto") and "/no-term" not in r), None)
                old = next((r for r in runs if r.startswith("auto") and "/no-term" in r), None)
            else:
                new = next((r for r in runs if r.startswith("auto") and "/no-term" not in r
                            and "/w" not in r), None)
                old = next((r for r in runs if r.startswith("auto") and "/no-term" in r
                            and "/w64" in r), None)
            if new is None:
                continue
            levels = sorted(runs[new]["levels"])
            # Each level's fastest method: Rotation, Batched (any width) and the waves, each
            # timed by every run that ran it there.
            best, best_of = {}, {}
            for level in levels:
                for name, run in runs.items():
                    if level not in run["levels"]:
                        continue
                    t = mean(run["levels"][level])
                    if t < best.get(level, float("inf")):
                        best[level], best_of[level] = t, f"{run['ran'][level]} ({name})"
            m2l = {name: sum(mean(r["levels"][l]) for l in levels if l in r["levels"])
                   for name, r in runs.items()}
            best_sum = sum(best.values())
            evaluation = {name: mean(r["eval"]) for name, r in runs.items()}
            # The per-level best's evaluation: the default's with its M2L replaced.
            best_eval = evaluation[new] - m2l[new] + best_sum
            row = {
                "machine": machine, "stage": stage, "case": key[1:], "new": new, "old": old,
                "m2l_new": m2l[new], "m2l_old": m2l.get(old), "m2l_best": best_sum,
                "eval_new": evaluation[new], "eval_old": evaluation.get(old),
                "eval_best": best_eval, "eval_rotation": evaluation.get("rotation"),
                "spread": spread, "errors": runs[new]["errors"],
                "losses": [],
            }
            for level in levels:
                t = mean(runs[new]["levels"][level])
                if best[level] > 0 and t / best[level] > MARGIN:
                    row["losses"].append((level, runs[new]["p"][level], runs[new]["ran"][level],
                                          runs[new]["pairs"][level], runs[new]["mhat"][level],
                                          runs[new]["exact"][level], t / best[level],
                                          t - best[level], best_of[level]))
            summary.append(row)
            print(f"### {machine}, {stage}: {key[1]} kD = {key[2]}, {key[3]}, {key[4]} charges")
            print()
            names = sorted(runs)
            print("| level | p | pairs | m̂ | exact | " + " | ".join(names) + " | best |")
            print("| ---: | ---: | ---: | ---: | ---: |" + " ---: |" * (len(names) + 1))
            for level in levels:
                r = runs[new]
                print(f"| {level} | {r['p'][level]} | {r['pairs'][level]} | "
                      f"{r['mhat'][level]:.1f} | {r['exact'][level]:.1f} | "
                      + " | ".join(
                          f"{mean(runs[n]['levels'][level]):.4f} ({runs[n]['ran'][level]})"
                          if level in runs[n]["levels"] else "–" for n in names)
                      + f" | {best_of[level]} |")
            print(f"| Σ M2L | | | | | " + " | ".join(f"{m2l[n]:.4f}" for n in names)
                  + f" | {best_sum:.4f} |")
            print(f"| evaluate | | | | | " + " | ".join(f"{evaluation[n]:.4f}" for n in names)
                  + f" | {best_eval:.4f} |")
            print()
    print("## Summary")
    print()
    print("| machine | stage | case | M2L new ÷ best | M2L old ÷ best | evaluate new ÷ best "
          "| evaluate new ÷ old | spread | levels above 1.10× (l, p, ran, pairs, m̂, exact, "
          "ratio, s lost; fastest) |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for r in summary:
        old_m2l = f"{r['m2l_old'] / r['m2l_best']:.3f}" if r["m2l_old"] else "–"
        old_eval = f"{r['eval_new'] / r['eval_old']:.3f}" if r["eval_old"] else "–"
        losses = "; ".join(
            f"{l}, {p}, {ran}, {pairs}, {mhat:.1f}, {exact:.1f}, {ratio:.2f}, {lost:.4f}; {fast}"
            for (l, p, ran, pairs, mhat, exact, ratio, lost, fast) in r["losses"]) or "–"
        case = " ".join(r["case"])
        print(f"| {r['machine']} | {r['stage']} | {case} | {r['m2l_new'] / r['m2l_best']:.3f} | "
              f"{old_m2l} | {r['eval_new'] / r['eval_best']:.3f} | {old_eval} | "
              f"{r['spread']:.2f} | {losses} |")


# `auto_batched_max_degree` and `auto_batched_min_columns` of nd-fmm-exec, by (ISA, planes).
MAX_DEGREE = {("avx512", 1): 15, ("avx512", 2): 15, ("avx2", 1): 12, ("avx2", 2): 11,
              ("neon", 1): 12, ("neon", 2): 10}
MIN_COLUMNS = {
    ("avx512", 1): [0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.9, 1.1, 1.6, 2.5, 3.1, 3.8, 4.6, 5.4, 9.1, 11.8],
    ("avx512", 2): [0.7, 0.7, 0.7, 0.7, 0.9, 1.2, 1.3, 1.6, 2.4, 3.9, 4.6, 5.7, 7.0, 10.4, 15.9,
                    28.2],
    ("avx2", 1): [0.3, 0.3, 0.3, 0.3, 0.5, 0.7, 0.9, 1.2, 1.9, 3.1, 3.9, 10.6, 14.5, 16.3, 32.3,
                  41.3],
    ("avx2", 2): [0.6, 0.6, 0.6, 0.6, 0.8, 1.0, 1.4, 2.1, 3.1, 5.6, 7.5, 37.5] + [64.0] * 4,
    ("neon", 1): [0.3, 0.3, 0.3, 0.3, 0.6, 0.8, 1.3, 2.1, 3.2, 8.0, 13.7] + [64.0] * 5,
    ("neon", 2): [0.4, 0.4, 0.4, 0.4, 0.8, 1.2, 2.1, 3.9, 7.5, 23.7] + [64.0] * 6,
}


def min_columns(isa, planes, p):
    table = MIN_COLUMNS[(isa, planes)]
    return table[p] if p < len(table) else 64.0


def batched_columns(pairs, targets):
    """`LevelShape::batched_columns`."""
    if pairs == 0 or targets == 0:
        return 0.0
    blocks = -(-targets // 64)
    per_block = targets / blocks
    row = min(pairs / targets, 316.0)
    visited = 316.0 * (1.0 - (1.0 - row / 316.0) ** per_block)
    return pairs / (blocks * visited)


def rule(machines):
    print("| machine | ISA | case | charges | p by level | term: Batched levels | rule s "
          "| best s | rule ÷ best | without the term: levels | ÷ best | Rotation s |")
    print("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |")
    for spec, directory in machines:
        machine, isa = spec.split(":")
        data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        info, evaluate = {}, defaultdict(list)
        for name in sorted(os.listdir(directory)):
            m = re.fullmatch(r"ch3-2d-(.+)-(forward|reverse)\.md", name)
            if not m:
                continue
            planes = 1 if m.group(1).endswith("real") else 2
            _, levels = read(os.path.join(directory, name))
            for row in levels:
                if len(row) < 24:
                    continue
                key = (row[0], row[2], row[3], planes)
                if row[6] == "Σ":
                    if row[4] == "rotation":
                        evaluate[key].append(float(re.search(r"evaluate ([0-9.]+) s",
                                                             row[8]).group(1)))
                    continue
                level = int(row[6])
                if row[4] in ("rotation", "batched-fused"):
                    method = "batched" if row[8].startswith("batched") else "rotation"
                    data[key][level][(row[4], method)].append(float(row[10]))
                info[(key, level)] = (int(row[7]), int(row[9]), int(row[20]))
        for key in sorted(data):
            mean = lambda xs: sum(xs) / len(xs)
            planes = key[3]
            rows = []
            for level in sorted(data[key]):
                times = data[key][level]
                p, pairs, targets = info[(key, level)]
                rotation = mean(times[("rotation", "rotation")])
                batched = times.get(("batched-fused", "batched"))
                rows.append((level, p, pairs, targets, rotation,
                             mean(batched) if batched else None))
            def chosen(term):
                levels, total = [], 0.0
                for level, p, pairs, targets, rotation, batched in rows:
                    on = (batched is not None and p <= MAX_DEGREE[(isa, planes)]
                          and (not term or batched_columns(pairs, targets)
                               >= min_columns(isa, planes, p)))
                    total += batched if on else rotation
                    if on:
                        levels.append(str(level))
                return levels, total
            best = sum(min(r[4], r[5]) if r[5] is not None else r[4] for r in rows)
            rotation_m2l = sum(r[4] for r in rows)
            base = mean(evaluate[key]) - rotation_m2l
            new_levels, new = chosen(True)
            old_levels, old = chosen(False)
            ps = ", ".join(str(r[1]) for r in rows)
            print(f"| {machine} | {isa} | {key[0]} kD = {key[1]}, {key[2]} | "
                  f"{'real' if planes == 1 else 'complex'} | {ps} | {','.join(new_levels) or '–'} | "
                  f"{base + new:.3f} | {base + best:.3f} | {(base + new) / (base + best):.3f} | "
                  f"{','.join(old_levels) or '–'} | {(base + old) / (base + best):.3f} | "
                  f"{base + rotation_m2l:.3f} |")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("pairs", "fmm", "rule"):
        sys.exit(__doc__)
    machines = [tuple(a.split("=", 1)) for a in sys.argv[2:]]
    {"pairs": pairs, "fmm": fmm, "rule": rule}[sys.argv[1]](machines)


if __name__ == "__main__":
    main()
