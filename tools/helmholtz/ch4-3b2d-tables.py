#!/usr/bin/env python3
"""tools/helmholtz/ch4-3b2d-tables.py: the tables of Helmholtz CH4.3b2d's report (the V near rule's
price per box before the plan on the levels whose waves are raised; docs/helmholtz/tasks/CH4.3b2d.md),
from the launches of `tools/helmholtz/ch4-3b2d-runs.sh`.

    python3 tools/helmholtz/ch4-3b2d-tables.py runs <dir>
    python3 tools/helmholtz/ch4-3b2d-tables.py levels <dir>
    python3 tools/helmholtz/ch4-3b2d-tables.py summary <dir> [<dir> …]

Each launch holds, under the defaults and under the time rule, four runs: `pair` (the per-pair
price, the library's default before CH4.3b2d), `pair+raised` (the trusted measure: the second
plan on the raised levels, `--v-after-plan raised`), `box` (the per-box price, one plan) and
`box+raised` (what a second plan would still add to it). `runs` prints per case and run the
evaluation and its ratio to `pair`'s, the build with its parts "counts", "plan" and "replan",
the errors over `pair`'s, the V pairs converted and left, and whether the plan was built again.
`levels` prints per case, M2L setting and raised level (a level the per-box run counted) the
crossovers of `pair`, `box` and the trusted measure, the V pairs left and the near interactions
of each, and the level's M2L and P2P seconds of each. `summary` prints per case and M2L setting
(each directory's launches, `reverse-` ones in the reverse order) the raised levels' M2L and P2P
seconds summed under each run, `box` against `pair` and against the trusted measure, the
evaluation of each, the build's plan parts and the errors. No timing is asserted; the numbers go
into fmm-validate/results/helmholtz-ch4-3b2d.md.
"""

import pathlib
import sys


def tables(path):
    """Every Markdown table of a launch's output, by the heading above it ("" for the first):
    a list of dicts keyed by the header."""
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


def label(m2l):
    """A run's (setting, run) from its column "M2L"."""
    setting = "time rule" if "/diagauto" in m2l else "defaults"
    run = "box" if "/vbox" in m2l else "pair"
    if "/vafterraised" in m2l:
        run += "+raised"
    return setting, run


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse(path):
    """Per (setting, run): the main row, the build row, the Near pairs Σ row and per level
    {level: {"pairs", "m2l", "p", "ran", "p2p", "interactions", "before", "after", "counted"}}."""
    out = {}
    t = tables(path)
    for row in t.get("", []):
        out.setdefault(label(row["M2L"]), {})["main"] = row
    for row in t.get("Build", []):
        out.setdefault(label(row["M2L"]), {})["build"] = row
    for row in t.get("M2L per level", []):
        if row["level"] == "Σ":
            continue
        lv = out.setdefault(label(row["M2L"]), {}).setdefault("levels", {})
        lv.setdefault(int(row["level"]), {}).update(
            pairs=int(row["V pairs"]), m2l=float(row["M2L s"]), p=int(row["p"]), ran=row["ran"]
        )
    for row in t.get("Near pairs", []):
        run = out.setdefault(label(row["M2L"]), {})
        if row["level"] == "Σ":
            run["near"] = row
            continue
        counted = row.get("leaves, V entries counted", "–")
        run.setdefault("levels", {}).setdefault(int(row["level"]), {}).update(
            p2p=float(row["P2P s"]),
            interactions=int(row["near interactions"]),
            before=number(row["crossover"]),
            after=number(row["after the plan"]),
            counted=None if counted == "–" else tuple(int(x) for x in counted.split(", ")),
        )
    return out


def launches(where):
    return sorted(pathlib.Path(where).glob("*.md"))


RUNS = ["pair", "pair+raised", "box", "box+raised"]
SETTINGS = ["defaults", "time rule"]


def evaluate(run):
    return float(run["near"]["evaluate s"])


def all_runs(where):
    print(
        "| case | M2L | run | evaluate s | ÷ pair | build s | counts s | plan s | replan s | "
        "φ ÷ pair | ∇φ ÷ pair | V converted | V pairs | replanned |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for path in launches(where):
        r = parse(path)
        for setting in SETTINGS:
            base = r.get((setting, "pair"))
            for name in RUNS:
                run = r.get((setting, name))
                if not run or "near" not in run or not base or "near" not in base:
                    continue
                main, build, near = run["main"], run.get("build", {}), run["near"]
                ratio = lambda k: f"{number(main[k]) / number(base['main'][k]):.3f}"
                print(
                    f"| {path.stem} | {setting} | {name} | {evaluate(run):.4f} | "
                    f"{evaluate(run) / evaluate(base):.3f} | {build.get('build s', '–')} | "
                    f"{build.get('counts s', '–')} | {build.get('plan s', '–')} | "
                    f"{build.get('replan s', '–')} | {ratio('φ')} | {ratio('∇φ')} | "
                    f"{near['converted']} | {near['V pairs']} | "
                    f"{'yes' if near.get('after the plan') == 'replanned' else '–'} |"
                )


def raised_levels(r, setting):
    """The levels the per-box run counted (raised, with V entries of leaves)."""
    box = r.get((setting, "box"), {})
    return sorted(l for l, e in box.get("levels", {}).items() if e.get("counted"))


def levels(where):
    print(
        "| case | M2L | level | p | ran (pair) | counted leaves, entries | crossover: pair, box, "
        "trusted | V pairs: pair, box, trusted | near interactions: pair, box, trusted | "
        "M2L + P2P s: pair, box, trusted |"
    )
    print("| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- |")
    for path in launches(where):
        r = parse(path)
        for setting in SETTINGS:
            runs = [r.get((setting, n), {}).get("levels", {}) for n in ("pair", "box", "pair+raised")]
            if not all(runs):
                continue
            for level in raised_levels(r, setting):
                e = [run.get(level, {}) for run in runs]
                seconds = [x.get("m2l", 0.0) + x.get("p2p", 0.0) for x in e]
                counted = e[1].get("counted")
                print(
                    f"| {path.stem} | {setting} | {level} | {e[1].get('p', '–')} | "
                    f"{e[0].get('ran', '–')} | {counted[0]}, {counted[1]} | "
                    f"{e[0].get('before') or 0:.0f}, {e[1].get('before') or 0:.0f}, "
                    f"{e[2].get('after') or 0:.0f} | "
                    f"{', '.join(str(x.get('pairs', 0)) for x in e)} | "
                    f"{', '.join(str(x.get('interactions', 0)) for x in e)} | "
                    f"{', '.join(f'{s:.4f}' for s in seconds)} |"
                )


def summary(wheres):
    print(
        "| case | M2L | raised levels | M2L + P2P s: pair, box, trusted | box − pair s | "
        "box − trusted s | evaluate s: pair, box, trusted | plan + counts s: pair, box | "
        "replan s (trusted) | φ ÷ pair: box, trusted |"
    )
    print("| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | --- |")
    for where in wheres:
        for path in launches(where):
            r = parse(path)
            for setting in SETTINGS:
                names = ("pair", "box", "pair+raised")
                runs = [r.get((setting, n)) for n in names]
                if not all(run and "near" in run for run in runs):
                    continue
                levels = raised_levels(r, setting)
                seconds = [
                    sum(
                        run.get("levels", {}).get(l, {}).get("m2l", 0.0)
                        + run.get("levels", {}).get(l, {}).get("p2p", 0.0)
                        for l in levels
                    )
                    for run in runs
                ]
                plan = [
                    float(run["build"]["plan s"]) + float(run["build"]["counts s"])
                    for run in runs[:2]
                ]
                phi = [number(run["main"]["φ"]) / number(runs[0]["main"]["φ"]) for run in runs[1:]]
                print(
                    f"| {path.stem} | {setting} | {','.join(map(str, levels)) or '–'} | "
                    f"{', '.join(f'{s:.4f}' for s in seconds)} | {seconds[1] - seconds[0]:+.4f} | "
                    f"{seconds[1] - seconds[2]:+.4f} | "
                    f"{', '.join(f'{evaluate(run):.4f}' for run in runs)} | "
                    f"{plan[0]:.3f}, {plan[1]:.3f} | {runs[2]['build'].get('replan s', '–')} | "
                    f"{phi[0]:.3f}, {phi[1]:.3f} |"
                )


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("runs", "levels", "summary"):
        sys.exit(__doc__)
    if sys.argv[1] == "summary":
        summary(sys.argv[2:])
        return
    {"runs": all_runs, "levels": levels}[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
