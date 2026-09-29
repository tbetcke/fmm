#!/usr/bin/env python3
"""Summarise a spike results table and compute the roofline prediction for SPIKE_REPORT.md.

Usage: python3 spikes/cubecl-gemm/analyse.py spikes/cubecl-gemm/results-m3max.md

Rows after the "## Metal repeat" heading are ignored.

Reads the Markdown table printed by the spike binary and prints Markdown sections:
the per-case summary with fractions of peak, and the data-centre roofline tables.
Every peak below is a stated assumption; see SPIKE_REPORT.md for the sources.
"""

import sys
from collections import defaultdict

# Measured-device peaks (GFLOP/s) and bandwidth (GB/s). Derivations in SPIKE_REPORT.md.
PEAK = {
    ("metal", "f32"): 14_300.0,  # 5120 ALUs x 2 flops x 1.398 GHz
    ("cpu", "f64"): 778.0,  # 12 P-cores x 4 FMA pipes x 2 lanes x 2 flops x 4.05 GHz
    ("cpu", "f32"): 1_555.0,  # same, 4 lanes
}
BW = {"metal": 400.0, "cpu": 400.0}  # unified memory, Apple spec for M3 Max (40-core GPU)

# Data-centre cards (vendor datasheets): f64 peak without / with FP64 tensor cores, f32
# peak, memory bandwidth, all in GFLOP/s and GB/s.
CARDS = {
    "A100 SXM 80GB": dict(f64=9_700.0, f64_tc=19_500.0, f32=19_500.0, bw=2_039.0),
    "H100 SXM": dict(f64=34_000.0, f64_tc=67_000.0, f32=67_000.0, bw=3_350.0),
}
PS = [4, 8, 12, 16]
BS = [1_000, 10_000, 100_000]


def parse(path):
    rows = []
    for line in open(path):
        if line.startswith("## Metal repeat"):
            break  # repeat runs are for variance only, not for the summary
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 11 or cells[0] in ("backend", "---"):
            continue
        backend, prec, p, _nc, b, impl, _ms, _reps, gf, err, check = cells
        if gf in ("–", "-"):
            continue
        rows.append(
            dict(backend=backend, prec=prec, p=int(p), b=int(b), impl=impl,
                 gflops=float(gf), err=float(err), check=check)
        )
    return rows


def family(impl):
    """lib-cmma: plane-level matrix units; lib-unit: scalar units; lib-auto picks one."""
    if impl == "lib:matmul_auto":
        return "lib-auto"
    if impl.startswith("lib:") and "cmma" in impl:
        return "lib-cmma"
    if impl.startswith("lib:"):
        return "lib-unit"
    return "tiled"


def intensity(nc, b, s):
    """Flops per byte of C = A X, moving A once and X, C once each (s bytes per element)."""
    return 2.0 * nc * nc * b / (s * (nc * nc + 2 * nc * b))


def summary(rows):
    best = defaultdict(dict)
    for r in rows:
        key = (r["backend"], r["prec"], r["p"], r["b"])
        fam = family(r["impl"])
        if fam not in best[key] or r["gflops"] > best[key][fam]["gflops"]:
            best[key][fam] = r
    print("| backend | precision | p | B | best library (GFLOP/s, % peak) "
          "| best tiled (GFLOP/s, % peak) | DRAM roofline GFLOP/s | max rel. error |")
    print("|---|---|---|---|---|---|---|---|")
    for key in sorted(best, key=lambda k: (k[0] != "metal", k[1], k[2], k[3])):
        backend, prec, p, b = key
        peak = PEAK[(backend, prec)]
        nc = (p + 1) ** 2
        s = 8 if prec == "f64" else 4
        bound = min(peak, intensity(nc, b, s) * BW[backend])
        cells = []
        for fams in (("lib-auto", "lib-cmma", "lib-unit"), ("tiled",)):
            cand = [best[key][f] for f in fams if f in best[key]]
            if not cand:
                cells.append("–")
                continue
            r = max(cand, key=lambda r: r["gflops"])
            name = r["impl"].removeprefix("lib:matmul_").split("(")[0]
            cells.append(f"{r['gflops']:.0f} ({100 * r['gflops'] / peak:.1f}%) {name}")
        err = max(r["err"] for r in best[key].values())
        print(f"| {backend} | {prec} | {p} | {b} | {cells[0]} | {cells[1]} | {bound:.0f} | {err:.1e} |")
    return best


def roof(peak, bw, nc, b, s):
    """Roofline bound in GFLOP/s for C = A X."""
    return min(peak, intensity(nc, b, s) * bw)


def efficiency(best, backend, prec, fams, b=100_000):
    """Fraction of the DRAM roofline bound reached by the best of `fams`, per p, capped at 1
    (shapes whose working set fits the system-level cache can exceed the DRAM bound)."""
    s = 8 if prec == "f64" else 4
    out = {}
    for p in PS:
        cand = [best[(backend, prec, p, b)][f] for f in fams
                if f in best.get((backend, prec, p, b), {})]
        if cand:
            bound = roof(PEAK[(backend, prec)], BW[backend], (p + 1) ** 2, b, s)
            out[p] = min(1.0, max(r["gflops"] for r in cand) / bound)
    return out


def rotation_flops(p):
    """Rotation M2L flops per pair: two rotations of (4/3)(p+1)^3 and a coaxial shift of
    (2/3)(p+1)^3 multiply-adds, i.e. (10/3)(p+1)^3 multiply-adds (design doc Section 4)."""
    return 2 * (10 / 3) * (p + 1) ** 3


def dense_flops(p):
    """Dense M2L flops per pair: one Nc x Nc matrix-vector product, Nc = (p+1)^2."""
    return 2 * (p + 1) ** 4


def roofline(best):
    effs = {
        "tiled (Metal f32, no matrix units)": efficiency(best, "metal", "f32", ["tiled"]),
        "library unit path (Metal f32)": efficiency(best, "metal", "f32", ["lib-unit"]),
        "library CMMA path (Metal f32, simdgroup matrices)":
            efficiency(best, "metal", "f32", ["lib-cmma", "lib-auto"]),
        "tiled (CPU runtime f64)": efficiency(best, "cpu", "f64", ["tiled"]),
        "library (CPU runtime f64)": efficiency(best, "cpu", "f64", ["lib-unit", "lib-auto"]),
    }
    print("\nEfficiency = measured / DRAM roofline bound, at B = 1e5 (capped at 100%):\n")
    print("| implementation | " + " | ".join(f"p = {p}" for p in PS) + " |")
    print("|---|" + "---|" * len(PS))
    for name, e in effs.items():
        print(f"| {name} | " + " | ".join(
            f"{100 * e[p]:.0f}%" if p in e else "–" for p in PS) + " |")

    print("\nArithmetic intensity (flop/byte) at B = 1e5 and ridge points (flop/byte):\n")
    print("| p | Nc | f64 | f32 |")
    print("|---|---|---|---|")
    for p in PS:
        nc = (p + 1) ** 2
        print(f"| {p} | {nc} | {intensity(nc, 100_000, 8):.1f} "
              f"| {intensity(nc, 100_000, 4):.1f} |")
    print()
    print("| device | f64 ridge | f64 ridge with FP64 tensor cores | f32 ridge |")
    print("|---|---|---|---|")
    for card, c in CARDS.items():
        print(f"| {card} | {c['f64'] / c['bw']:.1f} | {c['f64_tc'] / c['bw']:.1f} "
              f"| {c['f32'] / c['bw']:.1f} |")
    print(f"| M3 Max GPU (measured device) | – | – | {PEAK[('metal', 'f32')] / BW['metal']:.1f} |")

    central = effs["tiled (Metal f32, no matrix units)"]
    low = effs["library unit path (Metal f32)"]
    tc = effs["library CMMA path (Metal f32, simdgroup matrices)"]
    print("\nPredicted f64 GEMM throughput at B = 1e5 (GFLOP/s) = card roofline x Metal "
          "efficiency. Central: tiled kernel; low: library unit path; FP64-TC: hypothetical "
          "tensor-core path at the Metal CMMA efficiency (not available in CubeCL 0.10):\n")
    print("| card | p | roofline (no TC) | predicted low | predicted central "
          "| roofline with FP64 TC | FP64-TC hypothetical |")
    print("|---|---|---|---|---|---|---|")
    pred = {}
    for card, c in CARDS.items():
        for p in PS:
            nc = (p + 1) ** 2
            r = roof(c["f64"], c["bw"], nc, 100_000, 8)
            r_tc = roof(c["f64_tc"], c["bw"], nc, 100_000, 8)
            pred[(card, p)] = (r * low[p], r * central[p])
            print(f"| {card} | {p} | {r:.0f} | {r * low[p]:.0f} | {r * central[p]:.0f} "
                  f"| {r_tc:.0f} | {r_tc * tc[p]:.0f} |")

    print("\nDense vs rotation M2L in f64. Flops per pair: dense 2(p+1)^4, rotation "
          "(20/3)(p+1)^3; ratio 0.3(p+1). Break-even rotation efficiency: the fraction of "
          "f64 peak rotation must reach to match the predicted dense time; dense wins if "
          "rotation stays below it:\n")
    print("| card | p | dense ns/pair (central) | break-even rotation eff. (central) "
          "| break-even rotation eff. (low) |")
    print("|---|---|---|---|---|")
    for card, c in CARDS.items():
        for p in PS:
            lo, ce = pred[(card, p)]
            t_ce = dense_flops(p) / ce  # ns, since GFLOP/s = flop/ns
            t_lo = dense_flops(p) / lo
            e_ce = rotation_flops(p) / (c["f64"] * t_ce)
            e_lo = rotation_flops(p) / (c["f64"] * t_lo)
            print(f"| {card} | {p} | {t_ce:.2f} | {100 * e_ce:.1f}% | {100 * e_lo:.1f}% |")

    print("\nSame comparison for f32 on the measured M3 Max GPU (best measured dense):\n")
    print("| p | best dense GFLOP/s at B = 1e5 | ns/pair | break-even rotation eff. |")
    print("|---|---|---|---|")
    for p in PS:
        g = max(r["gflops"] for r in best[("metal", "f32", p, 100_000)].values())
        t = dense_flops(p) / g
        e = rotation_flops(p) / (PEAK[("metal", "f32")] * t)
        print(f"| {p} | {g:.0f} | {t:.3f} | {100 * e:.1f}% |")


def main():
    rows = parse(sys.argv[1])
    failed = [r for r in rows if r["check"] != "pass"]
    print(f"{len(rows)} measured rows, {len(failed)} failed checks\n")
    best = summary(rows)
    roofline(best)


if __name__ == "__main__":
    main()
