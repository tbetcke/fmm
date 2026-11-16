#!/usr/bin/env python3
"""tools/helmholtz/ch4-0e-tables.py: the tables of Helmholtz CH4.0e's report
(docs/helmholtz/tasks/CH4.0e.md) from the output directories of tools/helmholtz/ch4-0e-runs.sh.

    python3 tools/helmholtz/ch4-0e-tables.py compare <dir> [<dir> ...]
    python3 tools/helmholtz/ch4-0e-tables.py splits <dir> [<dir> ...]
    python3 tools/helmholtz/ch4-0e-tables.py levels <dir> <case> [<split>]
    python3 tools/helmholtz/ch4-0e-tables.py estimate <dir> [<dir> ...]
    python3 tools/helmholtz/ch4-0e-tables.py t1 <dir> [<dir> ...]
    python3 tools/helmholtz/ch4-0e-tables.py machines <dir A> <dir B> <case> <eps>
    python3 tools/helmholtz/ch4-0e-tables.py blocks <dir> [<dir> ...]

Several directories of one machine are read as one (the jobs of one machine). `compare`: per
case and ε ours at its best split (build plus one evaluation, the smaller of launches a and
b; the evaluation is the second of the build's two) beside FMM3D's whole call (the best of
two calls, the smaller of launches a and b) at the same requested ε and at matched accuracy
(the fastest grid eps whose φ and ∇φ errors are both at most ours; marked `>` where no grid
eps is, the most accurate one then). `splits`: build plus one evaluation per split, and the
checksums' agreement over splits and launches. `levels`: one case's M2L per level at its best
split. `estimate`: H4's bounded gain (the task's plan, "The estimate"). `t1`: the one-core
runs. `machines`: one case's per-level times on two machines at their best splits and at one
core. `blocks`: W1 at kD = 500 with every level's rotation blocks stored against the
defaults, per split (the `blocks` stage). Python 3.9 standard library only.
"""

import glob
import math
import os
import re
import sys
from collections import defaultdict

EPS = ["1e-3", "1e-6", "1e-8"]

# S6's prototype times (spikes/diagonal/results/times.md, run 1; one Grace core, µs):
# p -> (diagonal pair, S -> F + F -> R per box, RCR pair).
S6 = {
    24: (4.0, 139 + 172, 21.3),
    32: (7.0, 340 + 404, 47.5),
    48: (19.6, 1231 + 1420, 161.4),
    64: (32.6, 3046 + 3356, 411.2),
    96: (99.6, 10718 + 11455, 1290.6),
    128: (138.3, 26651 + 26758, 3021.8),
    192: (273.1, 111569 + 89866, 10923.8),
    256: (556.5, 297306 + 215673, 29208.9),
}

# S6's share of an interior box's V pairs left on RCR by the bound, real k, the calibrated p
# (spikes/diagonal/results/summary.md §3 "real, calibrated"), per k a row and ε.
S6_RCR_SHARE = [
    # k a, 1e-3, 1e-6, 1e-8
    (2.828, 0.0, 0.72, 1.0),
    (4.000, 0.0, 0.29, 1.0),
    (5.657, 0.0, 0.0, 0.96),
    (8.000, 0.0, 0.0, 0.49),
    (11.314, 0.0, 0.0, 0.03),
    (13.532, 0.0, 0.0, 0.03),
    (16.000, 0.0, 0.0, 0.03),
    (22.627, 0.0, 0.0, 0.03),
    (27.063, 0.0, 0.0, 0.03),
    (32.000, 0.0, 0.0, 0.03),
    (45.255, 0.0, 0.0, 0.0),
]

# The smallest k a of a level H4 counts as high-frequency (S6 §4: below k a ≈ 8 the
# diagonal form fails on most classes at ε ≤ 10⁻⁶ whatever the rule).
KA_MIN = 8.0


def cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def tables(path):
    """The Markdown tables of a file: {section heading: [row dicts]} (the first table
    under each `#`/`##` heading; the file's title for the first)."""
    out = {}
    section = None
    header = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                section = line.lstrip("#").strip()
                header = None
                continue
            if not line.startswith("|"):
                header = None if not line.strip() else header
                continue
            row = cells(line)
            if header is None:
                header = row
                out.setdefault(section, [])
                continue
            if set("".join(row)) <= set("-: "):
                continue
            out[section].append(dict(zip(header, row)))
    return out


def eps_of(degrees):
    m = re.match(r"ε = (\S+)", degrees)
    return m.group(1) if m else degrees


def num(s):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def case_of(name):
    """`ours-w4-cube-10-1x40-a` -> (`w4-cube-10`, `1x40`, `a`)."""
    m = re.match(r"(?:t1-)?(?:ours|blocks)-(.+)-(\d+x\d+)-([ab])$", name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def kd_of(case):
    return float(case.rsplit("-", 1)[1])


def ours(dirs, prefix="ours"):
    """{(case, eps): {split: [run dicts of launches a and b]}}."""
    runs = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, prefix + "-*.md"))):
            key = case_of(os.path.basename(path)[:-3])
            if key is None:
                continue
            case, split, launch = key
            t = tables(path)
            main = t.get("Helmholtz FMM against the sampled direct sum", [])
            leaves = {eps_of(r["degrees"]): r for r in t.get("Leaves", [])}
            levels = defaultdict(list)
            for r in t.get("M2L per level", []):
                levels[eps_of(r["degrees"])].append(r)
            for r in main:
                e = eps_of(r["degrees"])
                lv = leaves.get(e, {})
                total = [x for x in levels[e] if x["level"] == "Σ"]
                evaluate = num(r["evaluate s"])
                if total:
                    m = re.search(r"evaluate ([0-9.]+) s", total[0]["ran"])
                    if m:
                        evaluate = float(m.group(1))
                build = num(lv.get("build s", r["build s"]))
                runs[(case, e)][split].append(
                    {
                        "launch": launch,
                        "build": build,
                        "evaluate": evaluate,
                        "total": build + evaluate,
                        "phi": num(r["φ"]),
                        "grad": num(r["∇φ"]),
                        "checksum": r["checksum"],
                        "p": r["p by level"],
                        "leaf": lv.get("leaf size", "?"),
                        "levels": [x for x in levels[e] if x["level"] != "Σ"],
                        "leaves": lv,
                    }
                )
    return runs


def best(per_split):
    """(split, run) of the smallest total over splits and launches."""
    choice = None
    for split, rs in per_split.items():
        for r in rs:
            if choice is None or r["total"] < choice[1]["total"]:
                choice = (split, r)
    return choice


def fmm3d(dirs, prefix="fmm3d"):
    """{(case, variant): {eps: {"s": best seconds over launches, "phi", "grad"}}}."""
    out = defaultdict(dict)
    for d in dirs:
        for path in glob.glob(os.path.join(d, "fmm3d", prefix + "-*.log")):
            base = os.path.basename(path)[:-4]
            m = re.match(re.escape(prefix) + r"-(.+)-(default|fastker)-([ab])$", base)
            if not m:
                continue
            case, variant = m.group(1), m.group(2)
            with open(path, encoding="utf-8") as f:
                for line in f:
                    t = re.match(r"helmholtz: token=(\S+) .*seconds=\s*(\S+)", line)
                    if t:
                        e, s = t.group(1), float(t.group(2))
                        cell = out[(case, variant)].setdefault(e, {"s": s})
                        cell["s"] = min(cell["s"], s)
        for path in glob.glob(os.path.join(d, prefix + "-*.md")):
            case = os.path.basename(path)[len(prefix) + 1 : -3]
            t = tables(path)
            for rows in t.values():
                for r in rows:
                    if "run" not in r:
                        continue
                    cell = out[(case, r["run"])].setdefault(r["eps"], {"s": num(r["s"])})
                    cell["phi"], cell["grad"] = num(r["φ"]), num(r["∇φ"])
    return out


def matched(grid, phi, grad):
    """(eps, cell, exact): the fastest grid eps at least as accurate as (phi, grad) in both."""
    ok = [(c["s"], e, c) for e, c in grid.items() if "phi" in c and c["phi"] <= phi and c["grad"] <= grad]
    if ok:
        s, e, c = min(ok)
        return e, c, True
    measured = [(max(c["phi"] / phi, c["grad"] / grad), e, c) for e, c in grid.items() if "phi" in c]
    if not measured:
        return None, None, False
    _, e, c = min(measured)
    return e, c, False


def cases_in_order(keys):
    order = {"w1": 0, "w3": 1, "w4-cube": 2, "w4-sphere": 3, "w4-plummer": 4, "w4-clusters": 5}
    def k(case):
        w, kd = case.rsplit("-", 1)
        return (order.get(w, 9), float(kd))
    return sorted({c for c, _ in keys}, key=k)


def f2(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def e2(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2e}"


def compare(dirs):
    runs = ours(dirs)
    fm = fmm3d(dirs)
    print("| Case | ε | split | leaf | build s | evaluate s | ours s | φ, ∇φ | default s | ÷ | fastker s | ÷ | default matched: eps, s, ÷ | fastker matched: eps, s, ÷ |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for case in cases_in_order(runs.keys()):
        for e in EPS:
            if (case, e) not in runs:
                continue
            split, r = best(runs[(case, e)])
            row = [case, e, split, r["leaf"], f"{r['build']:.3f}", f"{r['evaluate']:.3f}", f"{r['total']:.3f}",
                   f"{e2(r['phi'])}, {e2(r['grad'])}"]
            for v in ("default", "fastker"):
                c = fm.get((case, v), {}).get(e)
                row += [f"{c['s']:.3f}", f"{r['total'] / c['s']:.2f}"] if c else ["–", "–"]
            for v in ("default", "fastker"):
                grid = fm.get((case, v), {})
                me, c, exact = matched(grid, r["phi"], r["grad"])
                if c is None:
                    row.append("–")
                else:
                    row.append(f"{'' if exact else '>'}{me}, {c['s']:.3f}, {r['total'] / c['s']:.2f}")
            print("| " + " | ".join(row) + " |")


def splits(dirs):
    runs = ours(dirs)
    names = sorted({s for v in runs.values() for s in v}, key=lambda s: (int(s.split("x")[0]), s))
    print("| Case | ε | " + " | ".join(names) + " | best | checksums |")
    print("| --- | --- | " + " | ".join("---:" for _ in names) + " | --- | --- |")
    for case in cases_in_order(runs.keys()):
        for e in EPS:
            per = runs.get((case, e))
            if not per:
                continue
            cellsv = []
            for s in names:
                rs = per.get(s, [])
                cellsv.append(f"{min(x['total'] for x in rs):.3f}" if rs else "–")
            sums = {x["checksum"] for rs in per.values() for x in rs}
            launches = sum(len(rs) for rs in per.values())
            agree = "equal" if len(sums) == 1 else f"{len(sums)} DIFFERENT"
            print(f"| {case} | {e} | " + " | ".join(cellsv) + f" | {best(per)[0]} | {agree} ({launches}) |")


def ka(kd, level):
    """k a_l with a_l the circumradius √3 w_l / 2 of a box of width D / 2^l (D = 1)."""
    return kd * math.sqrt(3.0) / 2 ** (level + 1)


def rcr_share(k_a, e):
    """S6's share of V pairs left on RCR at (k a, ε): the row of the largest k a ≤ k_a."""
    col = {"1e-3": 1, "1e-6": 2, "1e-8": 3}[e]
    rows = [r for r in S6_RCR_SHARE if r[0] <= k_a + 1e-9]
    return rows[-1][col] if rows else 1.0


def s6_at(p):
    """S6's prototype times at degree p, interpolated in log p (clamped at the ends)."""
    ps = sorted(S6)
    if p <= ps[0]:
        lo = hi = ps[0]
    elif p >= ps[-1]:
        lo = hi = ps[-1]
    else:
        lo = max(q for q in ps if q <= p)
        hi = min(q for q in ps if q >= p)
    if lo == hi:
        return S6[lo]
    t = (math.log(p) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return tuple(math.exp((1 - t) * math.log(a) + t * math.log(b)) for a, b in zip(S6[lo], S6[hi]))


def is_rcr(ran):
    return ran.startswith("rotation") or ran.startswith("batched") or ran.startswith("dense") or ran.startswith("classes")


def level_estimate(r, kd, e):
    """(M2L on RCR at k a ≥ 8, all M2L at k a ≥ 8, the prototype estimate's remaining time,
    the levels counted) for one run."""
    rcr = every = remain = 0.0
    counted = []
    for x in r["levels"]:
        level, p = int(x["level"]), int(x["p"])
        t, pairs, boxes = num(x["M2L s"]), num(x["V pairs"]), num(x["boxes"])
        if ka(kd, level) < KA_MIN or math.isnan(t):
            continue
        every += t
        if not is_rcr(x["ran"]):
            continue
        rcr += t
        share = 1.0 - rcr_share(ka(kd, level), e)
        diag, conv, rcr_pair = s6_at(p)
        scale = t / (pairs * rcr_pair) if pairs > 0 else 0.0
        remain += t * (1 - share) + share * scale * (pairs * diag + boxes * conv)
        counted.append(f"{level}:{p}")
    return rcr, every, remain, counted


def estimate(dirs):
    runs = ours(dirs)
    fm = fmm3d(dirs)
    print("| Case | ε | ours s | levels on RCR at k a ≥ 8 (level:p) | their M2L s | share | after H4, bound: s | after H4, prototype: s | ceiling (all M2L at k a ≥ 8) s | fastker s | after H4 ÷ fastker (bound, prototype) |")
    print("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for case in cases_in_order(runs.keys()):
        kd = kd_of(case)
        for e in EPS:
            if (case, e) not in runs:
                continue
            _, r = best(runs[(case, e)])
            rcr, every, remain, counted = level_estimate(r, kd, e)
            bound = r["total"] - rcr
            proto = r["total"] - rcr + remain
            ceiling = r["total"] - every
            c = fm.get((case, "fastker"), {}).get(e)
            ratio = f"{bound / c['s']:.2f}, {proto / c['s']:.2f}" if c else "–"
            print(f"| {case} | {e} | {r['total']:.3f} | {', '.join(counted) or 'none'} | {rcr:.3f} | "
                  f"{rcr / r['total']:.0%} | {bound:.3f} | {proto:.3f} | {ceiling:.3f} | "
                  f"{c['s']:.3f} | {ratio} |" if c else
                  f"| {case} | {e} | {r['total']:.3f} | {', '.join(counted) or 'none'} | {rcr:.3f} | "
                  f"{rcr / r['total']:.0%} | {bound:.3f} | {proto:.3f} | {ceiling:.3f} | – | – |")


def levels(dirs, case, split=None):
    runs = ours(dirs)
    kd = kd_of(case)
    print("| ε | split | level | p | k a | ran | V pairs | boxes | M2L s | ns per pair |")
    print("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for e in EPS:
        per = runs.get((case, e))
        if not per:
            continue
        s, r = (split, min(per[split], key=lambda x: x["total"])) if split else best(per)
        for x in r["levels"]:
            lv = int(x["level"])
            print(f"| {e} | {s} | {lv} | {x['p']} | {ka(kd, lv):.1f} | {x['ran']} | {x['V pairs']} | "
                  f"{x['boxes']} | {num(x['M2L s']):.4f} | {x['ns per pair']} |")


def t1(dirs):
    rows = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "t1-ours-*.md")):
            case = os.path.basename(path)[len("t1-ours-") : -3]
            t = tables(path)
            for r in t.get("Leaves", []):
                rows[(case, eps_of(r["degrees"]))] = r
    fm = fmm3d(dirs, "t1")
    print("| Case | ε | ours build s | ours evaluate s | ours s | φ, ∇φ | default s | fastker s | default matched: eps, s | fastker matched: eps, s |")
    print("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |")
    for (case, e), r in sorted(rows.items()):
        b, ev = num(r["build s"]), num(r["evaluate s"])
        phi, grad = num(r["φ"]), num(r["∇φ"])
        row = [case, e, f"{b:.3f}", f"{ev:.3f}", f"{b + ev:.3f}", f"{e2(phi)}, {e2(grad)}"]
        for v in ("default", "fastker"):
            c = fm.get((case, v), {}).get(e)
            row.append(f"{c['s']:.3f}" if c else "–")
        for v in ("default", "fastker"):
            me, c, exact = matched(fm.get((case, v), {}), phi, grad)
            row.append(f"{'' if exact else '>'}{me}, {c['s']:.3f}" if c else "–")
        print("| " + " | ".join(row) + " |")


def machines(dir_a, dir_b, case, e):
    a, b = ours([dir_a]), ours([dir_b])
    ra, rb = best(a[(case, e)]), best(b[(case, e)])
    print(f"Best splits: {os.path.basename(dir_a.rstrip('/'))} {ra[0]}, "
          f"{os.path.basename(dir_b.rstrip('/'))} {rb[0]}.")
    print()
    print("| level | p | ran | A M2L s | B M2L s | A ÷ B |")
    print("| ---: | ---: | --- | ---: | ---: | ---: |")
    lb = {x["level"]: x for x in rb[1]["levels"]}
    for x in ra[1]["levels"]:
        y = lb.get(x["level"])
        if y is None:
            continue
        ta, tb = num(x["M2L s"]), num(y["M2L s"])
        print(f"| {x['level']} | {x['p']} | {x['ran']} / {y['ran']} | {ta:.4f} | {tb:.4f} | {ta / tb:.2f} |")
    for key in ("P2P s", "M2L s", "P2L s", "M2P s", "P2M, M2M, L2L, L2P s", "build s", "evaluate s"):
        ta, tb = num(ra[1]["leaves"].get(key, "nan")), num(rb[1]["leaves"].get(key, "nan"))
        print(f"| {key} | – | – | {ta:.4f} | {tb:.4f} | {ta / tb if tb else float('nan'):.2f} |")


def blocks(dirs):
    base, stored = ours(dirs), ours(dirs, "blocks")
    print("| ε | split | build s: default, stored | evaluate s: default, stored | level 2 M2L s: default, stored | level 3 M2L s: default, stored | total ÷ | checksums |")
    print("| --- | --- | --- | --- | --- | --- | ---: | --- |")
    for e in EPS:
        a, b = base.get(("w1-500", e), {}), stored.get(("w1-500", e), {})
        for split in sorted(b, key=lambda s: int(s.split("x")[0])):
            if split not in a:
                continue
            x = min(a[split], key=lambda r: r["total"])
            y = min(b[split], key=lambda r: r["total"])
            def lv(r, level):
                hit = [num(v["M2L s"]) for v in r["levels"] if v["level"] == str(level)]
                tag = [v.get("blocks", "?") for v in r["levels"] if v["level"] == str(level)]
                return (hit[0] if hit else float("nan")), (tag[0] if tag else "?")
            sums = {r["checksum"] for r in a[split] + b[split]}
            (x2, _), (y2, t2) = lv(x, 2), lv(y, 2)
            (x3, _), (y3, _) = lv(x, 3), lv(y, 3)
            print(f"| {e} | {split} | {x['build']:.3f}, {y['build']:.3f} | {x['evaluate']:.3f}, "
                  f"{y['evaluate']:.3f} | {x2:.3f}, {y2:.3f} ({t2}) | {x3:.3f}, {y3:.3f} | "
                  f"{y['total'] / x['total']:.2f} | {'equal' if len(sums) == 1 else 'DIFFERENT'} |")


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    step, rest = argv[1], argv[2:]
    if step == "compare":
        compare(rest)
    elif step == "splits":
        splits(rest)
    elif step == "levels":
        levels([rest[0]], rest[1], rest[2] if len(rest) > 2 else None)
    elif step == "estimate":
        estimate(rest)
    elif step == "t1":
        t1(rest)
    elif step == "blocks":
        blocks(rest)
    elif step == "machines":
        machines(rest[0], rest[1], rest[2], rest[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
