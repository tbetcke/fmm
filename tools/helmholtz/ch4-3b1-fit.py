#!/usr/bin/env python3
"""tools/helmholtz/ch4-3b1-fit.py: the cost model of the diagonal form's time rule, Helmholtz
CH4.3b1 (docs/helmholtz/tasks/CH4.3b1.md), fitted from `helmholtz_accuracy --levels` outputs.

    python3 tools/helmholtz/ch4-3b1-fit.py fit <machine>=<file-or-dir> …
    python3 tools/helmholtz/ch4-3b1-fit.py check <machine>=<file-or-dir> …
    python3 tools/helmholtz/ch4-3b1-fit.py summary <machine>=<file-or-dir> …

`fit` prints, per machine, the constants [a, b] of a (p + 1)² + b (p + 1)³ thread-ns for
each unit, by least squares on the relative error over every level of every run given, each
level weighted by its measured time:

- S → F per source signature and F → R per accumulator converted back, from the table
  "Diagonal M2L"'s thread-seconds and counts (runs with the diagonal form, every class
  diagonal: the levels of 34 classes alone, so that no RCR pair is in the parts);
- the diagonal pair, from the same table's pairs' thread-seconds over the diagonal pairs;
- RCR's pair, from "M2L per level" under Rotation without the form: the level's seconds ×
  the threads over its V pairs (wall × threads, idle threads included, as the rule compares
  whole level calls);
- (one-rank runs only: on several ranks the seconds are the slowest rank's and the counts
  the sum's)
- the exponential M2L per box with a V row at its degree p_w, from "M2L per level" rows that
  ran `planewave`, the same measure, over the boxes of "Diagonal-form scratch" (p_w read from
  the p column: the degrees run): both of its passes go by the box, whatever the pairs per
  box.

With the table each unit's fitted values against the measured per level, the ratio, and the
diagonal pair over RCR's at p = 8 … 245 (the per-class crossover). `check` prints, per run
with the form and level, the options measured, the rule's choice in the run and the refit
model's from the measured counts; `summary` per launch the errors and evaluation of each
setting over the same setting's without the form. No timing is
asserted; the numbers go into fmm-validate/results/helmholtz-ch4-3b1.md.
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


def files(arg):
    machine, where = arg.split("=", 1)
    where = pathlib.Path(where)
    paths = sorted(where.glob("*.md")) if where.is_dir() else [where]
    return machine, paths


def threads(row):
    ranks, per = row["ranks × threads"].split("×")
    return int(ranks) * int(per)


def one_rank(row):
    """Whether the run is on one rank: on several, "Diagonal M2L"'s seconds are the slowest
    rank's and its counts summed over the ranks."""
    return row["ranks × threads"].split("×")[0].strip() == "1"


def samples(machine, paths):
    """Per unit a list of (p, thread-ns per unit, weight) from every run."""
    units = {"s2f": [], "f2r": [], "pair": [], "rotation": [], "plane_wave": []}
    for path in paths:
        t = tables(path)
        for r in t.get("Diagonal M2L", []):
            if not one_rank(r):
                continue
            if r["classes"] != "34" or int(r["RCR pairs"]) != 0:
                continue
            p = int(r["p"])
            s2f, f2r = int(r["S → F"]), int(r["F → R"])
            pairs = int(r["diagonal pairs"])
            if s2f:
                units["s2f"].append((p, float(r["S → F thread-s"]) * 1e9 / s2f, s2f))
            if f2r:
                units["f2r"].append((p, float(r["F → R thread-s"]) * 1e9 / f2r, f2r))
            if pairs:
                units["pair"].append((p, float(r["pairs thread-s"]) * 1e9 / pairs, pairs))
        # The boxes with a V row per (setting, level), from "Diagonal-form scratch".
        boxes = {}
        for r in t.get("Diagonal-form scratch", []):
            boxes[(r["M2L"].split("/")[0], r["level"])] = int(r["targets"])
        for r in t.get("M2L per level", []):
            if not one_rank(r) or r["level"] == "Σ" or r["V pairs"] in ("0", "–"):
                continue
            pairs = int(r["V pairs"])
            seconds = float(r["M2L s"]) * threads(r) * 1e9
            ran = r["ran"]
            if ran.startswith("rotation"):
                units["rotation"].append((int(r["p"]), seconds / pairs, pairs))
            elif ran.startswith("planewave"):
                n = boxes.get((r["M2L"].split("/")[0], r["level"]), 0)
                if n:
                    units["plane_wave"].append((int(r["p"]), seconds / n, n))
    return units


def fit(points):
    """[a, b] of a q² + b q³, q = p + 1, minimising Σ w ((model − y) / y)², each level weighted
    by its measured time w = y n (the rule's choices matter where the time is; a level of a
    few boxes is mostly overhead), and, if a or b < 0, the better of the one-term fits."""
    if not points:
        return None
    rows = [((p + 1) ** 2 / y, (p + 1) ** 3 / y, y * n) for p, y, n in points]
    s11 = sum(w * x * x for x, _, w in rows)
    s12 = sum(w * x * z for x, z, w in rows)
    s22 = sum(w * z * z for _, z, w in rows)
    t1 = sum(w * x for x, _, w in rows)
    t2 = sum(w * z for _, z, w in rows)
    det = s11 * s22 - s12 * s12
    if det > 0:
        a = (t1 * s22 - t2 * s12) / det
        b = (s11 * t2 - s12 * t1) / det
        if a >= 0 and b >= 0:
            return [a, b]
    best = None
    for col in (0, 1):
        num = sum(w * r[col] for *r, w in [(x, z, w) for x, z, w in rows])
        den = sum(w * r[col] ** 2 for *r, w in [(x, z, w) for x, z, w in rows])
        c = num / den
        ab = [c, 0.0] if col == 0 else [0.0, c]
        res = sum(y * n * (model(ab, p) / y - 1) ** 2 for p, y, n in points)
        if best is None or res < best[0]:
            best = (res, ab)
    return best[1]


def model(ab, p):
    q = p + 1
    return ab[0] * q * q + ab[1] * q * q * q


def run_fit(args):
    for arg in args:
        machine, paths = files(arg)
        units = samples(machine, paths)
        consts = {u: fit(pts) for u, pts in units.items()}
        print(f"### {machine}\n")
        print("| unit | a (ns) | b (ns) |")
        print("| --- | ---: | ---: |")
        for u, ab in consts.items():
            if ab:
                print(f"| {u} | {ab[0]:.4g} | {ab[1]:.4g} |")
        print()
        print("| unit | p | measured ns | model ns | model / measured |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for u, pts in units.items():
            ab = consts[u]
            for p, y, _ in sorted(set(pts)):
                print(f"| {u} | {p} | {y:.4g} | {model(ab, p):.4g} | {model(ab, p) / y:.3f} |")
        print()
        if consts["pair"] and consts["rotation"]:
            print("| p | diagonal pair / RCR pair |")
            print("| ---: | ---: |")
            for p in (8, 12, 16, 24, 32, 48, 64, 96, 131, 180, 245):
                print(f"| {p} | {model(consts['pair'], p) / model(consts['rotation'], p):.3f} |")
            print()
        print("Rust:")
        for u, ab in consts.items():
            if ab:
                print(f"    {u}: [{ab[0]:.3g}, {ab[1]:.3g}],")
        print()


def base_setting(m2l):
    """(base, diagonal) of a "M2L per level" or "Time rule" row's setting: base "rotation" or
    "defaults", diagonal "off", "rule" or "auto"."""
    base = "defaults" if m2l.startswith("auto") else "rotation"
    if "/diagauto" in m2l:
        return base, "auto"
    if "/diag" in m2l:
        return base, "rule"
    return base, "off"


def run_check(args):
    """Per launch and level with V pairs: the measured M2L seconds of each option the setting
    has (RCR: Rotation without the form; the diagonal form with the bound's classes: Rotation
    with the rule; the defaults' own M2L: the defaults without the form), the option the time
    rule took in the run (Rotation or the defaults with `auto`), and the choice the model makes
    with the constants fitted from the same launches (`fit`), from the level's measured counts:
    its boxes with a V row ("Diagonal-form scratch"), its pairs on each form under the rule
    ("Diagonal M2L"), the degrees run. Each choice's measured seconds over the best's."""
    for arg in args:
        machine, paths = files(arg)
        units = samples(machine, paths)
        c = {u: fit(pts) for u, pts in units.items()}
        total = {"levels": 0, "ran": 0, "model": 0, "lost_ran": 0.0, "lost_model": 0.0,
                 "best": 0.0}
        print(f"### {machine}\n")
        print("| launch | setting | level | p | k a | classes | RCR s | diagonal s "
              "| own M2L | its s | best | the run took | ÷ best | the refit model takes "
              "| ÷ best |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- "
              "| ---: | --- | ---: |")
        for path in paths:
            t = tables(path)
            m2l = {}
            for r in t.get("M2L per level", []):
                if r["level"] != "Σ":
                    m2l[(base_setting(r["M2L"]), int(r["level"]))] = r
            model = {}
            for r in t.get("Time rule", []):
                model[(base_setting(r["M2L"]), int(r["level"]))] = r
            boxes = {}
            for r in t.get("Diagonal-form scratch", []):
                boxes[(r["M2L"].split("/")[0], int(r["level"]))] = int(r["targets"])
            dpairs = {}
            for r in t.get("Diagonal M2L", []):
                if base_setting(r["M2L"]) == ("rotation", "rule"):
                    dpairs[int(r["level"])] = (int(r["diagonal pairs"]), int(r["RCR pairs"]))
            levels = sorted({l for (_, l) in m2l})
            for base in ("rotation", "defaults"):
                for l in levels:
                    took = m2l.get(((base, "auto"), l))
                    rcr = m2l.get((("rotation", "off"), l))
                    diag = m2l.get((("rotation", "rule"), l))
                    own = m2l.get(((base, "off"), l))
                    mod = model.get(((base, "auto"), l))
                    if not (took and rcr and diag and own and mod):
                        continue
                    if int(rcr["V pairs"]) == 0 or own["ran"].startswith(("batched", "dense")):
                        continue
                    options = {"rcr": float(rcr["M2L s"]), "own": float(own["M2L s"])}
                    if diag["ran"].startswith("diagonal"):
                        options["diagonal"] = float(diag["M2L s"])
                    best = min(options.values())
                    ran = float(took["M2L s"])
                    # The refit model's choice from the counts.
                    n = boxes.get(("rotation", l), 0)
                    p = int(diag["p"])
                    choice = "own"
                    if "diagonal" in options and n and l in dpairs:
                        d, r_ = dpairs[l]
                        t_diag = (n * (model_unit(c["s2f"], p) + model_unit(c["f2r"], p))
                                  + d * model_unit(c["pair"], p) + r_ * model_unit(c["rotation"], p))
                        t_rcr = (d + r_) * model_unit(c["rotation"], p)
                        if own["ran"].startswith("planewave"):
                            # Taken from the waves where the form beats them, then the cheaper
                            # of the form and RCR.
                            t_waves = n * model_unit(c["plane_wave"], int(own["p"]))
                            if t_diag < t_waves:
                                choice = "diagonal" if t_diag < t_rcr else "rcr"
                        elif t_diag < t_rcr:
                            choice = "diagonal"
                    chosen = options[choice]
                    total["levels"] += 1
                    total["ran"] += int(ran <= 1.05 * best)
                    total["model"] += int(chosen <= 1.05 * best)
                    total["lost_ran"] += max(0.0, ran - best)
                    total["lost_model"] += max(0.0, chosen - best)
                    total["best"] += best
                    print(f"| {path.stem} | {base} | {l} | {mod['p']} | {mod['k a']} "
                          f"| {mod['bound classes']} | {options['rcr']:.4f} "
                          f"| {options.get('diagonal', float('nan')):.4f} | {own['ran']} "
                          f"| {options['own']:.4f} | {min(options, key=options.get)} "
                          f"| {took['ran']} | {ran / best:.2f} | {choice} | {chosen / best:.2f} |")
        print()
        print(f"Levels: {total['levels']}; the run's choice within 1.05× the best: {total['ran']} "
              f"(seconds lost {total['lost_ran']:.3f}); the refit model's: {total['model']} "
              f"(lost {total['lost_model']:.3f}); of {total['best']:.3f} s at the best.\n")


def model_unit(ab, p):
    return model(ab, p)


def run_summary(args):
    """Per launch, Rotation and the defaults each without the form, with the rule and with
    the time rule: the errors (φ, ∇φ) over the same setting's without the form and over
    Rotation's without it (the acceptance's measure), the evaluation
    seconds (the second evaluation's) and their ratio, and the levels the form took."""
    print("| machine | launch | setting | φ | ∇φ | ÷ without | ÷ Rotation | evaluate s "
          "| ÷ without | diagonal levels |")
    print("| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |")
    for arg in args:
        machine, paths = files(arg)
        for path in paths:
            runs = tables(path).get("", [])
            plain = {}
            for r in runs:
                base, diag = base_setting(r["M2L"])
                if diag == "off":
                    plain[base] = r
            for r in runs:
                base, diag = base_setting(r["M2L"])
                ref = plain.get(base)
                if not ref:
                    continue
                e = (float(r["φ"]), float(r["∇φ"]))
                e0 = (float(ref["φ"]), float(ref["∇φ"]))
                t, t0 = float(r["evaluate s"]), float(ref["evaluate s"])
                levels = "–"
                if "; diagonal " in r["M2L"]:
                    levels = r["M2L"].split("; diagonal ")[1].split("/")[0]
                rot = plain.get("rotation", ref)
                er = (float(rot["φ"]), float(rot["∇φ"]))
                print(f"| {machine} | {path.stem} | {base} {diag} | {e[0]:.2e} | {e[1]:.2e} "
                      f"| {e[0] / e0[0]:.3f}/{e[1] / e0[1]:.3f} "
                      f"| {e[0] / er[0]:.3f}/{e[1] / er[1]:.3f} | {t:.2f} | {t / t0:.2f} "
                      f"| {levels} |")


def main(argv):
    if len(argv) < 3 or argv[1] not in ("fit", "check", "summary"):
        print(__doc__)
        return 2
    {"fit": run_fit, "check": run_check, "summary": run_summary}[argv[1]](argv[2:])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
