#!/usr/bin/env python3
"""tools/helmholtz/ch4-0i-tables.py: the report's tables of Helmholtz CH4.0i from the launches
of tools/helmholtz/ch4-0i-runs.sh, beside CH4.0h's launches of the same machine
(docs/helmholtz/tasks/CH4.0i.md).

    ch4-0i-tables.py levels <ch4-0i-dir> <ch4-0h-dir>   W1 at kD = 500: levels 2 and 3 by split
    ch4-0i-tables.py stored <ch4-0i-dir> <ch4-0h-dir>   by pairs, blocks stored against not
    ch4-0i-tables.py matrix <ch4-0i-dir> <ch4-0h-dir>   CH3.6's matrix: boxes against auto
    ch4-0i-tables.py pairs  <ch4-0i-dir> <ch4-0h-dir>   CH3.6's matrix: boxes against pairs
    ch4-0i-tables.py degrees <ch4-0i-dir> -             every Rotation level of the stages
                                                         rotation, pairs, levels (order a)
                                                         and kd1000
                                                         by degree: boxes against pairs

Every row's checksum is compared with CH4.0h's launches of the same case and ε (all equal
there across splits and launches): a difference is printed as a line "CHECKSUM DIFFERS". Times
are means over the launches (orders a and b), in seconds; "linear" is the one-core level time
over (threads × the level time), against the one-core time by boxes (CH4.0e's measure) and by
pairs (the best one core).
"""

import re
import statistics
import sys
from pathlib import Path


def tables(path):
    """The Markdown tables of a launch's output: {heading: [row dicts]}; the first table
    (before any heading) under ""."""
    out, heading, header = {}, "", None
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            heading, header = line[3:].strip(), None
            continue
        if not line.startswith("|"):
            header = None if not line.strip() else header
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set(cells[0]) <= set("-: "):
            continue
        out.setdefault(heading, []).append(dict(zip(header, cells)))
    return out


def eps_of(row):
    m = re.search(r"ε = (\S+)", row["degrees"])
    return m.group(1) if m else row["degrees"]


def split_of(row):
    """The row's RCR split from its M2L column: boxes, pairs or auto."""
    m = re.search(r"/rcr-(boxes|pairs)", row["M2L"])
    return m.group(1) if m else "auto"


def threads_of(row):
    r, t = row["ranks × threads"].split("×")
    return int(r), int(t)


def checksums_before(before, prefix):
    """{(workload, kD, ε): set of checksums} over CH4.0h's launches named prefix-*.md."""
    sums = {}
    for path in Path(before).glob(f"{prefix}-*.md"):
        for r in tables(path).get("", []):
            sums.setdefault((r["Workload"], r["kD"], eps_of(r)), set()).add(r["checksum"])
    return sums


def check(rows, sums, what):
    for r in rows:
        key = (r["Workload"], r["kD"], eps_of(r))
        want = sums.get(key)
        if want is None:
            print(f"no CH4.0h checksum for {key} ({what})")
        elif r["checksum"] not in want or len(want) != 1:
            print(f"CHECKSUM DIFFERS: {what} {key}: {r['checksum']} against {sorted(want)}")


def level_times(paths, levels=(2, 3)):
    """{(ε, split, (ranks, threads), level): [M2L s, …]}, {(ε, split, rt): [evaluate s, …]}."""
    m2l, evaluate = {}, {}
    for path in paths:
        t = tables(path)
        for r in t.get("M2L per level", []):
            if not r["level"].isdigit():
                continue
            level = int(r["level"])
            if level in levels:
                key = (eps_of(r), split_of(r), threads_of(r), level)
                m2l.setdefault(key, []).append(float(r["M2L s"]))
        for r in t.get("Build", []):
            key = (eps_of(r), split_of(r), threads_of(r))
            evaluate.setdefault(key, []).append(float(r["evaluate s"]))
    return m2l, evaluate


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def levels(here, before, prefix="levels"):
    paths = sorted(Path(here).glob(f"{prefix}-*.md"))
    sums = checksums_before(before, "build")
    for path in paths:
        check(tables(path).get("", []), sums, path.name)
    m2l, evaluate = level_times(paths)
    eps = sorted({k[0] for k in m2l}, key=float, reverse=True)
    splits = sorted({k[2] for k in m2l if k[2] != (1, 1)})
    for level in (2, 3):
        print(f"\nLevel {level}:\n")
        print("| ε | ranks × threads | boxes s | auto s | pairs s | pairs ÷ boxes "
              "| linear (boxes, one core by boxes) | linear (pairs, one core by boxes) "
              "| linear (pairs, one core by pairs) |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for e in eps:
            one_boxes = mean(m2l.get((e, "boxes", (1, 1), level), []))
            one_pairs = mean(m2l.get((e, "pairs", (1, 1), level), []))
            print(f"| {e} | 1 × 1 | {one_boxes:.3f} | – | {one_pairs:.3f} | "
                  f"{one_pairs / one_boxes:.2f} | – | – | – |")
            for rt in splits:
                n = rt[0] * rt[1]
                b = mean(m2l.get((e, "boxes", rt, level), []))
                a = mean(m2l.get((e, "auto", rt, level), []))
                p = mean(m2l.get((e, "pairs", rt, level), []))
                print(f"| {e} | {rt[0]} × {rt[1]} | {b:.3f} | {a:.3f} | {p:.3f} | {p / b:.2f} "
                      f"| {one_boxes / (n * b):.2f} | {one_boxes / (n * p):.2f} "
                      f"| {one_pairs / (n * p):.2f} |")
    print("\nThe evaluation (s):\n")
    print("| ε | ranks × threads | boxes | auto (first rule) | pairs (the rule) | auto ÷ boxes "
          "| pairs ÷ boxes |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for e in eps:
        for rt in [(1, 1)] + splits:
            b = mean(evaluate.get((e, "boxes", rt), []))
            a = mean(evaluate.get((e, "auto", rt), []))
            p = mean(evaluate.get((e, "pairs", rt), []))
            print(f"| {e} | {rt[0]} × {rt[1]} | {b:.3f} | {a:.3f} | {p:.3f} | {a / b:.2f} "
                  f"| {p / b:.2f} |")


def stored(here, before):
    paths = sorted(Path(here).glob("stored-*.md"))
    sums = checksums_before(before, "build")
    rows = {}
    for path in paths:
        check(tables(path).get("", []), sums, path.name)
        budget = int(path.name.split("-")[2])
        t = tables(path)
        for r in t.get("M2L per level", []):
            if r["level"] == "2":
                rows.setdefault((eps_of(r), threads_of(r), budget, "m2l"), []).append(
                    float(r["M2L s"]))
        for r in t.get("Build", []):
            key = (eps_of(r), threads_of(r), budget)
            rows.setdefault(key + ("build",), []).append(float(r["build s"]))
            rows.setdefault(key + ("rcr",), []).append(float(r["RCR tables s"]))
            rows.setdefault(key + ("evaluate",), []).append(float(r["evaluate s"]))
    eps = sorted({k[0] for k in rows}, key=float, reverse=True)
    splits = sorted({k[1] for k in rows})
    print("| ε | ranks × threads | level 2 s: 512 MiB, 2 GiB | linear (one core by pairs, "
          "same budget) | RCR tables s | build s | evaluate s | build + evaluate ÷ |")
    print("| --- | --- | --- | --- | --- | --- | --- | ---: |")
    for e in eps:
        for rt in splits:
            n = rt[0] * rt[1]
            v = {k: [mean(rows.get((e, rt, b, k), [])) for b in (512, 2048)]
                 for k in ("m2l", "rcr", "build", "evaluate")}
            one = [mean(rows.get((e, (1, 1), b, "m2l"), [])) for b in (512, 2048)]
            lin = [one[i] / (n * v["m2l"][i]) for i in range(2)]
            total = [v["build"][i] + v["evaluate"][i] for i in range(2)]
            print(f"| {e} | {rt[0]} × {rt[1]} | {v['m2l'][0]:.3f}, {v['m2l'][1]:.3f} "
                  f"| {lin[0]:.2f}, {lin[1]:.2f} | {v['rcr'][0]:.3f}, {v['rcr'][1]:.3f} "
                  f"| {v['build'][0]:.3f}, {v['build'][1]:.3f} "
                  f"| {v['evaluate'][0]:.3f}, {v['evaluate'][1]:.3f} | {total[1] / total[0]:.2f} |")


def matrix(here, before, prefix="matrix", other="auto"):
    sums = checksums_before(before, "matrix")
    print(f"| case | ε | ranks × threads | evaluate s: boxes, {other} | {other} ÷ boxes "
          "| levels by pairs |")
    print("| --- | --- | --- | --- | ---: | --- |")
    ratios = []
    for path in sorted(Path(here).glob(f"{prefix}-*.md")):
        t = tables(path)
        check(t.get("", []), sums, path.name)
        ev = {}
        for r in t.get("Build", []):
            ev.setdefault((eps_of(r), threads_of(r)), {})[split_of(r)] = float(r["evaluate s"])
        pairs = {}
        for r in t.get("M2L per level", []):
            if "by pairs" in r["ran"]:
                pairs.setdefault(eps_of(r), []).append(r["level"])
        case = path.name[len(prefix) + 1:-len(".md")]
        for (e, rt), v in sorted(ev.items(), key=lambda kv: -float(kv[0][0])):
            if "boxes" in v and other in v:
                ratio = v[other] / v["boxes"]
                ratios.append(ratio)
                print(f"| {case} | {e} | {rt[0]} × {rt[1]} | {v['boxes']:.3f}, {v[other]:.3f} "
                      f"| {ratio:.2f} | {', '.join(pairs.get(e, [])) or '–'} |")
    if ratios:
        print(f"\n{other} ÷ boxes: geometric mean "
              f"{statistics.geometric_mean(ratios):.3f}, range {min(ratios):.2f}–{max(ratios):.2f}"
              f" over {len(ratios)} rows")


def degrees(here, _before):
    """Every Rotation level with both splits in a launch, by degree."""
    rows = {}
    paths = [p for prefix in ("rotation", "pairs", "levels", "kd1000") for p in
             sorted(Path(here).glob(f"{prefix}-*.md")) if not p.name.endswith("-b.md")]
    for path in paths:
        for r in tables(path).get("M2L per level", []):
            if not r["level"].isdigit() or not r["ran"].startswith("rotation"):
                continue
            case = path.name[:-len(".md")]
            key = (case, eps_of(r), r["ranks × threads"], int(r["level"]), int(r["p"]),
                   int(r["boxes"]), int(r["V pairs"]))
            split = split_of(r)
            if split != "auto":
                rows.setdefault(key, {})[split] = float(r["M2L s"])
    print("| launch | ε | ranks × threads | level | p | boxes | V pairs | boxes s | pairs s "
          "| pairs ÷ boxes |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, v in sorted(rows.items(), key=lambda kv: (kv[0][4], kv[0][5], kv[0][0])):
        if "boxes" in v and "pairs" in v and v["boxes"] >= 0.001:
            case, e, rt, level, p, boxes, pairs = key
            print(f"| {case} | {e} | {rt} | {level} | {p} | {boxes} | {pairs} "
                  f"| {v['boxes']:.4f} | {v['pairs']:.4f} | {v['pairs'] / v['boxes']:.2f} |")


if __name__ == "__main__":
    modes = {
        "levels": levels,
        "stored": stored,
        "matrix": matrix,
        "pairs": lambda here, before: matrix(here, before, "pairs", "pairs"),
        "degrees": degrees,
    }
    if len(sys.argv) != 4 or sys.argv[1] not in modes:
        sys.exit(__doc__)
    modes[sys.argv[1]](sys.argv[2], sys.argv[3])
