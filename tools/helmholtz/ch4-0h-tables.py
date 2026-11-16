#!/usr/bin/env python3
"""tools/helmholtz/ch4-0h-tables.py: the report's tables of Helmholtz CH4.0h from the launches
of tools/helmholtz/ch4-0h-runs.sh, beside CH4.0e's launches of the same machine
(docs/helmholtz/tasks/CH4.0h.md).

    ch4-0h-tables.py build  <ch4-0h-dir> <ch4-0e-dir>   W1 at kD = 500: the build by split
    ch4-0h-tables.py matrix <ch4-0h-dir> <ch4-0e-dir>   CH3.6's matrix: builds and checksums

Every launch's checksum is compared with CH4.0e's launches of the same case and ε (all of
them equal there across splits and launches); a difference is printed as a line
"CHECKSUM DIFFERS". Times are means over the launches named, in seconds.
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


def main_rows(path):
    """{ε: (build s, checksum)} of a launch's first table."""
    return {
        eps_of(r): (float(r["build s"]), r["checksum"]) for r in tables(path).get("", [])
    }


def build_rows(path):
    """{ε: row of the table "Build"}."""
    return {eps_of(r): r for r in tables(path).get("Build", [])}


def checksums_before(before, case):
    """{ε: set of checksums} over every CH4.0e launch of `case` (ours-<case>-*.md)."""
    sums = {}
    for f in sorted(Path(before).glob(f"ours-{case}-*.md")):
        for e, (_, c) in main_rows(f).items():
            sums.setdefault(e, set()).add(c)
    return sums


def mean(values):
    return statistics.fmean(values) if values else float("nan")


def build(after, before):
    after, before = Path(after), Path(before)
    launches = sorted(after.glob("build-*-[ab].md"))
    splits = []
    for f in launches:
        s = f.name.split("-")[1]
        if s not in splits:
            splits.append(s)
    sums = checksums_before(before, "w1-500")
    parts = ["octree", "routes", "counts", "plan", "operator", "RCR tables", "scratch"]
    print(
        "| ε | split | build s (a, b) | ÷ one core | CH4.0e build s | "
        + " | ".join(f"{p} s" for p in parts)
        + " | checksum |"
    )
    print("| --- | --- | ---: | ---: | ---: | " + "---: | " * len(parts) + "--- |")
    rows = {s: [build_rows(after / f"build-{s}-{o}.md") for o in "ab"
                if (after / f"build-{s}-{o}.md").exists()] for s in splits}
    for e in ["1e-3", "1e-6", "1e-8"]:
        one = mean([float(r[e]["build s"]) for r in rows.get("1x1", []) if e in r])
        for s in splits:
            got = [r[e] for r in rows[s] if e in r]
            if not got:
                continue
            b = [float(r["build s"]) for r in got]
            # CH4.0e's one-core launch is `t1-ours-w1-500` (10⁻⁶ only).
            pattern = "t1-ours-w1-500.md" if s == "1x1" else f"ours-w1-500-{s}-[ab].md"
            old = [
                main_rows(f)[e][0]
                for f in sorted(before.glob(pattern))
                if e in main_rows(f)
            ]
            ok = all(r["checksum"] in sums.get(e, set()) for r in got)
            for r in got:
                if r["checksum"] not in sums.get(e, set()):
                    print(f"CHECKSUM DIFFERS: build {s} {e} {r['checksum']} {sums.get(e)}")
            cells = [f"{mean([float(r[p + ' s']) for r in got]):.3f}" for p in parts]
            print(
                f"| {e} | {s} | {mean(b):.2f} ({', '.join(f'{x:.2f}' for x in b)}) | "
                f"{mean(b) / one:.2f} | {mean(old):.2f} | " + " | ".join(cells)
                + f" | {'CH4.0e' if ok else 'DIFFERS'} |"
            )


def matrix(after, before):
    after, before = Path(after), Path(before)
    print("| case | ε | split | build s | CH4.0e build s | ÷ | operator s | evaluate s | checksum |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    ratios = {}
    for f in sorted(after.glob("matrix-*.md")):
        m = re.match(r"matrix-(.+)-(\d+x\d+)\.md", f.name)
        case, s = m.group(1), m.group(2)
        sums = checksums_before(before, case)
        olds = [main_rows(g) for g in sorted(before.glob(f"ours-{case}-{s}-[ab].md"))]
        for e, r in build_rows(f).items():
            old = mean([o[e][0] for o in olds if e in o])
            new = float(r["build s"])
            ok = r["checksum"] in sums.get(e, set())
            if not ok:
                print(f"CHECKSUM DIFFERS: {case} {s} {e} {r['checksum']} {sums.get(e)}")
            # CH4.0e printed its builds to 0.01 s: no ratio below 0.05 s.
            ratio = f"{new / old:.2f}" if old >= 0.05 else "–"
            if old >= 0.05:
                ratios.setdefault((s, e), []).append(new / old)
            print(
                f"| {case} | {e} | {s} | {new:.3f} | {old:.2f} | {ratio} | "
                f"{float(r['operator s']):.3f} | {float(r['evaluate s']):.3f} | "
                f"{'CH4.0e' if ok else 'DIFFERS'} |"
            )
    print()
    print("| split | ε | cases | build ÷ CH4.0e: geometric mean | min | max |")
    print("| --- | --- | ---: | ---: | ---: | ---: |")
    for (s, e), rs in sorted(ratios.items()):
        print(
            f"| {s} | {e} | {len(rs)} | {statistics.geometric_mean(rs):.2f} | "
            f"{min(rs):.2f} | {max(rs):.2f} |"
        )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("build", "matrix"):
        sys.exit(__doc__)
    {"build": build, "matrix": matrix}[sys.argv[1]](sys.argv[2], sys.argv[3])
