#!/usr/bin/env python3
"""tools/helmholtz/ch4-2c2-tables.py: the report's tables of Helmholtz CH4.2c2 from the
launches of `ch4-2c2-runs.sh` (docs/helmholtz/tasks/CH4.2c2.md).

    python3 tools/helmholtz/ch4-2c2-tables.py <machine>=<out-dir> …

Prints Markdown: (1) per launch the runs (Rotation, the diagonal form built directly and
interpolated): the diagonal levels and classes, errors, "÷ Rotation", evaluate, build, the
diagonal tables' build and the checksum; (2) per launch and diagonal level both builds side by
side: the classes, the transfer functions' build seconds and their ratio, the level's M2L
seconds. Reads the example's tables by their headers.
"""

import pathlib
import sys


def tables(path):
    """Every Markdown table of a launch's output, by the heading above it ("" for the
    first): a list of dicts keyed by the header."""
    out, heading, header = {}, "", None
    for line in pathlib.Path(path).read_text().splitlines():
        if line.startswith("## "):
            heading, header = line[3:].strip(), None
            continue
        if not line.startswith("|"):
            header = None
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
            out.setdefault(heading, [])
        elif set(cells[0]) <= set("-: "):
            continue
        elif len(cells) == len(header):
            out[heading].append(dict(zip(header, cells)))
    return out


def build_of(m2l):
    """The run of a column "M2L": "Rotation", "direct" or "interpolated"."""
    if "/diag" not in m2l:
        return "Rotation"
    return "interpolated" if "/interp" in m2l else "direct"


def ratio(a, b):
    try:
        return f"{float(a) / float(b):.2f}"
    except (ValueError, ZeroDivisionError):
        return "–"


def main(args):
    for arg in args:
        machine, directory = arg.split("=", 1)
        for path in sorted(pathlib.Path(directory).glob("*.md")):
            t = tables(path)
            runs = t.get("", [])
            if not runs:
                continue
            print(f"### {machine}: {path.stem}\n")
            build = {build_of(r["M2L"]): r for r in t.get("Build", [])}
            print("| run | diagonal levels:classes | ranks × threads | φ | ∇φ | ÷ Rotation "
                  "| evaluate s | build s | diagonal tables s | checksum |")
            print("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |")
            for r in runs:
                b = build_of(r["M2L"])
                levels = (r["M2L"].split("; diagonal ")[1].split("/")[0]
                          if b != "Rotation" else "–")
                print(f"| {b} | {levels} | {r['ranks × threads']} | {r['φ']} | {r['∇φ']} "
                      f"| {r['÷ Rotation']} | {r['evaluate s']} | {r['build s']} "
                      f"| {build.get(b, {}).get('diagonal tables s', '–')} | {r['checksum']} |")
            print()
            per = {}
            for r in t.get("Diagonal M2L", []):
                per.setdefault(int(r["level"]), {})[build_of(r["M2L"])] = r
            if per:
                print("| level | p | classes direct | interpolated | transfer build s direct "
                      "| interpolated | direct ÷ interpolated | M2L s direct | interpolated |")
                print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
                for level, rows in sorted(per.items()):
                    d, i = rows.get("direct", {}), rows.get("interpolated", {})
                    p = (d or i).get("p", "–")
                    print(f"| {level} | {p} | {d.get('classes', '0')} | {i.get('classes', '0')} "
                          f"| {d.get('transfer build s', '–')} "
                          f"| {i.get('transfer build s', '–')} "
                          f"| {ratio(d.get('transfer build s'), i.get('transfer build s'))} "
                          f"| {d.get('M2L s', '–')} | {i.get('M2L s', '–')} |")
                print()


if __name__ == "__main__":
    main(sys.argv[1:])
