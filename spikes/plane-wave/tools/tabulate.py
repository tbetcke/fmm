#!/usr/bin/env python3
"""Tabulates the output of tools/sweep.sh as the Markdown table of SPIKE_REPORT.md.

    python3 tools/tabulate.py MACHINE-NAME sweep-output.txt

One row per p and number of processes: the M2L time per V pair (µs; with K processes the
median over the processes) of Batched, Batched fused, Rotation and the plane waves with
the matched quadrature and with the model's (β = 0.1), and the bar (the faster of Batched
and Rotation) over the plane waves.
"""
import re
import sys

name, path = sys.argv[1], sys.argv[2]
rows = {}
copies = 1
for line in open(path):
    m = re.match(r"# p=(\d+) level=\d+ copies=(\d+)", line)
    if m:
        copies = int(m.group(2))
        continue
    fields = dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)
    if not line.strip() or "p" not in fields:
        continue
    variant = line.split()[0]
    p, s = int(fields["p"]), int(fields["s"])
    t = float(fields.get("us_per_pair", fields.get("us_per_pair_median", "nan")))
    row = rows.setdefault((p, copies), {})
    if variant == "merge":
        row.setdefault("merge", []).append((s, t))
    else:
        row[variant] = t

print(f"| machine | p | processes | Batched | fused | Rotation | plane waves, matched s | plane waves, model's s | bar ÷ plane waves (matched) | fused ÷ plane waves |")
print("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |")
for (p, k), r in sorted(rows.items()):
    merges = r.get("merge", [])
    if len(merges) < 2 or "batched" not in r:
        continue
    (sm, tm), (sb, tb) = merges[0], merges[1]
    bar = min(r["batched"], r["rotation"])
    which = "B" if r["batched"] <= r["rotation"] else "R"
    print(
        f"| {name} | {p} | {k} | {r['batched']:.4f} | {r['fused']:.4f} | {r['rotation']:.4f} | "
        f"{tm:.4f} (s = {sm}) | {tb:.4f} (s = {sb}) | **{bar / tm:.2f}** ({which}) | {r['fused'] / tm:.2f} |"
    )
