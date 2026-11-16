#!/usr/bin/env python3
"""tools/helmholtz/ch4-0j-tables.py: the report's tables of Helmholtz CH4.0j from the launches
of tools/helmholtz/ch4-0j-runs.sh, beside CH4.0h's launches of the same machine
(docs/helmholtz/tasks/CH4.0j.md).

    ch4-0j-tables.py matrix <ch4-0j-dir> <ch4-0h-dir>   CH3.6's matrix at one rank per node

Per case and ε: the build of this branch and of `main` in the same session (means of the
launches matrix-<case>-<split>-<branch|main>-<1|2>), their plane-wave tables, and CH4.0h's build
of the same case and split (matrix-<case>-<split>.md); the ratio branch ÷ CH4.0h (the
acceptance's) and branch ÷ main. A case is "under 1 s" where CH4.0h's build plus twice its
evaluation is under 1 s. Every launch's checksum is compared with CH4.0h's of the same case
and ε; a difference is printed as a line "CHECKSUM DIFFERS". Times in seconds.
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


def build_rows(path):
    """{ε: row of the table "Build"}."""
    return {eps_of(r): r for r in tables(path).get("Build", [])}


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def matrix(after, before):
    after, before = Path(after), Path(before)
    runs = {}
    for f in sorted(after.glob("matrix-*.md")):
        m = re.match(r"matrix-(.+)-(\d+x\d+)-(branch|main)-(\d)\.md", f.name)
        if not m:
            continue
        case, split, who = m.group(1), m.group(2), m.group(3)
        runs.setdefault((case, split), {}).setdefault(who, []).append(build_rows(f))
    print(
        "| case | ε | split | CH4.0h build s | + 2 evaluations | branch build s | main build s "
        "| branch ÷ CH4.0h | branch ÷ main | plane-wave tables s (branch, main, CH4.0h) "
        "| rest s (branch) | checksum |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |")
    summary = {}
    for (case, split), by in sorted(runs.items()):
        old_file = before / f"matrix-{case}-{split}.md"
        old = build_rows(old_file) if old_file.exists() else {}
        for e in ["1e-3", "1e-6", "1e-8"]:
            branch = [r[e] for r in by.get("branch", []) if e in r]
            main = [r[e] for r in by.get("main", []) if e in r]
            if not branch:
                continue
            o = old.get(e)
            sums = {r["checksum"] for r in branch + main}
            ok = o is not None and sums == {o["checksum"]}
            if not ok:
                print(f"CHECKSUM DIFFERS: {case} {split} {e} {sums} {o and o['checksum']}")
            b = mean([float(r["build s"]) for r in branch])
            a = mean([float(r["build s"]) for r in main])
            pw = lambda rows: mean([float(r["plane-wave tables s"]) for r in rows])
            old_build = float(o["build s"]) if o else float("nan")
            total = old_build + 2 * float(o["evaluate s"]) if o else float("nan")
            under = total < 1.0
            ratio = b / old_build if old_build > 0 else float("nan")
            if under:
                summary.setdefault((split, e), []).append((ratio, case))
            print(
                f"| {case} | {e} | {split} | {old_build:.3f} | {total:.3f}{' *' if under else ''} "
                f"| {b:.3f} | {a:.3f} | {ratio:.2f} | {b / a:.2f} "
                f"| {pw(branch):.3f}, {pw(main):.3f}, "
                f"{float(o['plane-wave tables s']) if o else float('nan'):.3f} "
                f"| {b - pw(branch):.3f} | {'CH4.0h' if ok else 'DIFFERS'} |"
            )
    print()
    print("\\* under 1 s (CH4.0h's build plus twice its evaluation).")
    print()
    print("| split | ε | cases under 1 s | ≤ 0.5 | branch ÷ CH4.0h: geometric mean | min | max | above 0.5 |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for (split, e), rs in sorted(summary.items()):
        ratios = [r for r, _ in rs if r == r]
        above = ", ".join(f"{c} {r:.2f}" for r, c in rs if r > 0.5)
        print(
            f"| {split} | {e} | {len(rs)} | {sum(r <= 0.5 for r in ratios)} | "
            f"{statistics.geometric_mean(ratios):.2f} | {min(ratios):.2f} | {max(ratios):.2f} "
            f"| {above or '–'} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "matrix":
        sys.exit(__doc__)
    matrix(sys.argv[2], sys.argv[3])
