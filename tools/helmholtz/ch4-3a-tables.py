#!/usr/bin/env python3
"""tools/helmholtz/ch4-3a-tables.py: the tables of Helmholtz CH4.3a's report from the launches
of tools/helmholtz/ch4-3a-runs.sh (docs/helmholtz/tasks/CH4.3a.md).

    python3 tools/helmholtz/ch4-3a-tables.py <out-dir>

Reads every <out-dir>/*.md, the main table of `helmholtz_accuracy` (a row per run), and prints
Markdown:

- "Acceptance": per case (launch and build) the rule at the FMM's tolerance (s = 1), its errors
  over Rotation's ("÷ Rotation", φ and ∇φ) and the classes it took per level;
- "Margin": per case the ratio max(φ, ∇φ) over Rotation's at each s, and the largest s whose run
  and every smaller s's stay within 1.1×, with the classes the rule takes at each s summed over the
  levels (the levels that changed named in the last column).

s is the rule's tolerance over the FMM's (`--diagonal <s ε>`, θ′ = s θ); "/diag" alone is s = 1.
"""

import re
import sys
from pathlib import Path

RATIO = 1.1


def rows(path):
    """The main table's rows of one launch, with its density."""
    density = "random"
    out = []
    for line in path.read_text().splitlines():
        if line.startswith("Density: "):
            density = line.split(": ", 1)[1].strip()
        if line.startswith("## "):
            break  # the main table ends at the first of the other tables
        if not line.startswith("| W"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 20 or cells[0] == "Workload":
            continue
        out.append({
            "workload": cells[0],
            "n": cells[1],
            "kd": cells[2],
            "eps": float(cells[3].split("=")[1]),
            "m2l": cells[7],
            "p": cells[11],
            "phi": float(cells[16]),
            "grad": float(cells[17]),
            "ratio": cells[19],
            "density": density,
        })
    return out


def parse_m2l(m2l, eps):
    """(s, build, {level: classes}) of a row; s None without the diagonal form."""
    classes = {}
    match = re.search(r"diagonal ([0-9:,]+)", m2l)
    if match:
        for part in match.group(1).split(","):
            level, count = part.split(":")
            classes[int(level)] = int(count)
    tag = re.search(r"/diag([0-9.e+-]*)", m2l)
    if not tag:
        return None, None, classes
    s = float(tag.group(1)) / eps if tag.group(1) else 1.0
    build = "interpolated" if "/interp" in m2l else "direct"
    return s, build, classes


def ratio(cell):
    if cell in ("–", ""):
        return None
    phi, grad = (float(x) for x in cell.split("/"))
    return phi, grad


def main():
    out = Path(sys.argv[1])
    cases = {}
    for path in sorted(out.glob("*.md")):
        for row in rows(path):
            s, build, classes = parse_m2l(row["m2l"], row["eps"])
            if s is None:
                continue
            key = (path.stem, build)
            cases.setdefault(key, []).append((s, row, classes))
    print("## Acceptance (s = 1)\n")
    print("| launch | workload | kD | density | ε | build | p by level | φ ÷ Rotation | ∇φ ÷ "
          "Rotation | classes per level | within 1.1× |")
    print("| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- |")
    for (stem, build), runs in cases.items():
        for s, row, classes in runs:
            if abs(s - 1.0) > 1e-9:
                continue
            r = ratio(row["ratio"])
            ok = r is not None and max(r) <= RATIO
            print(f"| {stem} | {row['workload']} | {row['kd']} | {row['density']} | "
                  f"{row['eps']:.0e} | {build} | {row['p']} | {r[0]:.3f} | {r[1]:.3f} | "
                  f"{','.join(f'{l}:{c}' for l, c in sorted(classes.items()))} | "
                  f"{'yes' if ok else '**no**'} |")
    print("\n## Margin\n")
    scales = sorted({s for runs in cases.values() for s, _, _ in runs})
    head = " | ".join(f"s = {s:g}" for s in scales)
    print(f"| launch | build | {head} | margin | classes taken (Σ over levels) by s |")
    print("| --- | --- | " + "---: | " * len(scales) + "---: | --- |")
    for (stem, build), runs in cases.items():
        by_s = {s: (ratio(row["ratio"]), classes) for s, row, classes in runs}
        cells = []
        margin = None
        holding = True
        for s in scales:
            if s not in by_s or by_s[s][0] is None:
                cells.append("–")
                continue
            worst = max(by_s[s][0])
            cells.append(f"{worst:.3f}" if worst < 1e3 else f"{worst:.1e}")
            if holding and worst <= RATIO:
                margin = s
            else:
                holding = False
        taken = ", ".join(f"{s:g}: {sum(c.values())}" for s, (_, c) in sorted(by_s.items()))
        print(f"| {stem} | {build} | {' | '.join(cells)} | "
              f"{'–' if margin is None else f'{margin:g}'} | {taken} |")


if __name__ == "__main__":
    main()
