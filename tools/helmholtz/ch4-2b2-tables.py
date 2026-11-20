#!/usr/bin/env python3
"""tools/helmholtz/ch4-2b2-tables.py: the report's tables of Helmholtz CH4.2b2 from the
launches of `ch4-2b2-runs.sh` (docs/helmholtz/tasks/CH4.2b2.md).

    python3 tools/helmholtz/ch4-2b2-tables.py <machine>=<out-dir> …

Prints Markdown: per machine and launch, (1) the runs: errors, "÷ Rotation", the evaluation
and the build with its diagonal tables; (2) per level the M2L seconds of each setting without
and with the diagonal form, and their ratio; (3) the diagonal level call by part, with the
conversions' rate and the tables' build. Reads the example's tables by their headers.
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


def setting(m2l):
    """The setting of a run from the column "M2L" of the main table."""
    base = "auto-pw" if m2l.startswith("auto") else "rotation"
    return base, "/diag" in m2l


def main(args):
    for arg in args:
        machine, directory = arg.split("=", 1)
        for path in sorted(pathlib.Path(directory).glob("*.md")):
            t = tables(path)
            runs = t.get("", [])
            if not runs:
                continue
            print(f"### {machine}: {path.stem}\n")
            build = {r["M2L"]: r for r in t.get("Build", [])}
            print("| M2L | diagonal | ranks × threads | φ | ∇φ | ÷ Rotation | evaluate s "
                  "| build s | diagonal tables s | checksum |")
            print("| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |")
            for r in runs:
                base, diag = setting(r["M2L"])
                levels = r["M2L"].split("; diagonal ")[1].split("/")[0] if diag else "–"
                b = next((v for k, v in build.items() if k.startswith(base[:4])
                          and ("/diag" in k) == diag), {})
                print(f"| {base} | {levels} | {r['ranks × threads']} | {r['φ']} | {r['∇φ']} "
                      f"| {r['÷ Rotation']} | {r['evaluate s']} | {r['build s']} "
                      f"| {b.get('diagonal tables s', '–')} | {r['checksum']} |")
            print()
            # Per level: each setting off and on.
            per = {}
            for r in t.get("M2L per level", []):
                if r["level"] == "Σ":
                    continue
                base = "auto-pw" if r["M2L"].startswith("auto") else "rotation"
                diag = "/diag" in r["M2L"]
                per.setdefault((base, int(r["level"])), {})[diag] = r
            if per:
                print("| setting | level | p | without: ran | M2L s | with: ran | M2L s "
                      "| with ÷ without |")
                print("| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |")
                for (base, level), rows in sorted(per.items()):
                    off, on = rows.get(False), rows.get(True)
                    if off is None or on is None:
                        continue
                    ratio = float(on["M2L s"]) / max(float(off["M2L s"]), 1e-12)
                    print(f"| {base} | {level} | {off['p']} | {off['ran']} | {off['M2L s']} "
                          f"| {on['ran']} | {on['M2L s']} | {ratio:.3f} |")
                print()
            diagonal = t.get("Diagonal M2L", [])
            if diagonal:
                print("| setting | level | p | classes | diagonal pairs | RCR pairs | M2L s "
                      "| S → F pass s | targets pass s | S → F / pairs / F → R / RCR thread-s "
                      "| S → F, F → R GF/s | transfer build s | conversions build s |")
                print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- "
                      "| --- | ---: | ---: |")
                for r in diagonal:
                    base = "auto-pw" if r["M2L"].startswith("auto") else "rotation"
                    parts = " / ".join(r[k] for k in ("S → F thread-s", "pairs thread-s",
                                                      "F → R thread-s", "RCR thread-s"))
                    print(f"| {base} | {r['level']} | {r['p']} | {r['classes']} "
                          f"| {r['diagonal pairs']} | {r['RCR pairs']} | {r['M2L s']} "
                          f"| {r['S → F pass s']} | {r['targets pass s']} | {parts} "
                          f"| {r['S → F GF/s']}, {r['F → R GF/s']} | {r['transfer build s']} "
                          f"| {r['conversions build s']} |")
                print()


if __name__ == "__main__":
    main(sys.argv[1:])
