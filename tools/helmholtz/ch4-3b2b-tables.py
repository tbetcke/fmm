#!/usr/bin/env python3
"""tools/helmholtz/ch4-3b2b-tables.py: the tables of Helmholtz CH4.3b2b's report (the V near rule
on the levels the time rule decides after the plan, and the defaults' errors at 10⁻⁶;
docs/helmholtz/tasks/CH4.3b2b.md), from the launches of `tools/helmholtz/ch4-3b2b-runs.sh`.

    python3 tools/helmholtz/ch4-3b2b-tables.py errors <dir>
    python3 tools/helmholtz/ch4-3b2b-tables.py runs <dir>
    python3 tools/helmholtz/ch4-3b2b-tables.py levels <dir>
    python3 tools/helmholtz/ch4-3b2b-tables.py summary <dir> [<dir> …]

`errors` reads the `errors-*` launches (Rotation and the defaults, each with the V rule off and of
the cost model): per case the errors (φ, ∇φ) of each, the defaults' over Rotation's with the rule
off and with it on, and the V pairs each converted. `runs` reads the other launches (a case's
`<name>-rotation.md` and `<name>.md`): per case and run the evaluation, the build with its parts
"plan" and "replan", the errors over the defaults', the V pairs converted, whether the plan was
built again. `levels` prints per level whose crossover rose after the plan, each run with the V
rule after the plan against the same run without it: the crossovers, the V pairs, the near
interactions, the M2L's and the P2P's seconds before and after, their change, and the change the
level's own mean costs predict (the M2L's seconds per pair and the P2P's per interaction of the
run without it: the after-plan rule's assumption). `summary` prints per case and run with the V
rule after the plan (each directory's launches, `reverse-` ones in the reverse order) the raised
levels' measured change summed, the predicted change, the evaluation's change, the second plan's
seconds and the errors over the run without it. No timing is asserted; the numbers go into
fmm-validate/results/helmholtz-ch4-3b2b.md.
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
    """A run's name from its column "M2L"."""
    name = "rotation" if m2l.startswith("rotation") else "defaults"
    if "/diagauto" in m2l:
        name += "+time"
    if "/vafterall" in m2l:
        name += "+all"
    elif "/vafter" in m2l:
        name += "+taken"
    if "/vcost" not in m2l:
        name += " (V off)"
    return name


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def runs(paths):
    """Per run label: the main row, the build row, the Near pairs Σ row and per level
    {level: {"pairs", "m2l", "p2p", "interactions", "before", "after", "p", "ran"}}."""
    out = {}
    for path in paths:
        t = tables(path)
        for row in t.get("", []):
            out.setdefault(label(row["M2L"]), {})["main"] = row
        for row in t.get("Build", []):
            out.setdefault(label(row["M2L"]), {})["build"] = row
        for row in t.get("M2L per level", []):
            if row["level"] == "Σ":
                continue
            lv = out.setdefault(label(row["M2L"]), {}).setdefault("levels", {})
            entry = lv.setdefault(int(row["level"]), {})
            entry.update(
                pairs=int(row["V pairs"]),
                m2l=float(row["M2L s"]),
                p=int(row["p"]),
                ran=row["ran"],
            )
        for row in t.get("Near pairs", []):
            run = out.setdefault(label(row["M2L"]), {})
            if row["level"] == "Σ":
                run["near"] = row
                continue
            entry = run.setdefault("levels", {}).setdefault(int(row["level"]), {})
            entry.update(
                p2p=float(row["P2P s"]),
                interactions=int(row["near interactions"]),
                before=number(row["crossover"]),
                after=number(row["after the plan"]),
            )
    return out


def cases(where, prefix):
    """The launches by case name: the files `<name>.md` and `<name>-rotation.md`."""
    found = {}
    for path in sorted(pathlib.Path(where).glob("*.md")):
        name = path.stem
        if name.startswith("errors-") != (prefix == "errors-"):
            continue
        case = name.removesuffix("-rotation")
        found.setdefault(case, []).append(path)
    return found


def errors(where):
    print(
        "| case | Rotation φ (V off) | defaults φ (V off) | ÷ (V off) φ, ∇φ | Rotation φ | "
        "defaults φ | ÷ (V cost) φ, ∇φ | converted: Rotation, defaults | offered |"
    )
    print("| --- | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |")
    for case, paths in cases(where, "errors-").items():
        r = runs(paths)

        def err(name):
            row = r.get(name, {}).get("main")
            return (number(row["φ"]), number(row["∇φ"])) if row else (None, None)

        def conv(name):
            near = r.get(name, {}).get("near")
            return near["converted"] if near else "–"

        ro, do, rc, dc = (
            err("rotation (V off)"),
            err("defaults (V off)"),
            err("rotation"),
            err("defaults"),
        )
        if None in (ro[0], do[0], rc[0], dc[0]):
            continue  # an unfinished launch
        offered = r.get("rotation", {}).get("near", {}).get("V pairs offered", "–")
        ratio = lambda a, b: f"{a[0] / b[0]:.3f}, {a[1] / b[1]:.3f}" if a[0] and b[0] else "–"
        print(
            f"| {case.removeprefix('errors-')} | {ro[0]:.3g} | {do[0]:.3g} | {ratio(do, ro)} | "
            f"{rc[0]:.3g} | {dc[0]:.3g} | {ratio(dc, rc)} | {conv('rotation')}, "
            f"{conv('defaults')} | {offered} |"
        )


ORDER = [
    "rotation",
    "defaults",
    "defaults+all",
    "defaults+time",
    "defaults+time+taken",
    "defaults+time+all",
]


def all_runs(where):
    print(
        "| case | run | evaluate s | ÷ defaults | build s | plan s | replan s | φ ÷ defaults | "
        "∇φ ÷ defaults | V converted | V pairs | replanned |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for case, paths in cases(where, "").items():
        r = runs(paths)
        base = r.get("defaults", {}).get("main")
        for name in ORDER:
            run = r.get(name)
            if not run or "main" not in run:
                continue
            main, build, near = run["main"], run.get("build", {}), run.get("near", {})
            ev = float(near.get("evaluate s") or main["evaluate s"])
            ev0 = float(r["defaults"]["near"]["evaluate s"]) if base else None
            ratio = lambda k: (
                f"{number(main[k]) / number(base[k]):.3f}"
                if base and number(base[k])
                else "–"
            )
            print(
                f"| {case} | {name} | {ev:.4f} | {ev / ev0:.3f} | {build.get('build s', '–')} | "
                f"{build.get('plan s', '–')} | {build.get('replan s', '–')} | {ratio('φ')} | "
                f"{ratio('∇φ')} | {near.get('converted', '–')} | {near.get('V pairs', '–')} | "
                f"{'yes' if near.get('after the plan') == 'replanned' else '–'} |"
            )


def level_rows(where):
    """Per case, run and raised level: (case, name, level, b, a, predicted, Δ evaluate,
    on, off)."""
    for case, paths in cases(where, "").items():
        r = runs(paths)
        for name in ORDER:
            if not (name.endswith("+all") or name.endswith("+taken")) or name not in r:
                continue
            off = r.get(name.rsplit("+", 1)[0])
            if not off:
                continue
            on = r[name]
            ev = float(on["near"]["evaluate s"]) - float(off["near"]["evaluate s"])
            p2p_rate = sum(e.get("p2p", 0.0) for e in off["levels"].values()) / max(
                1, sum(e.get("interactions", 0) for e in off["levels"].values())
            )
            rows = []
            for level, a in sorted(on.get("levels", {}).items()):
                if not (a.get("after") and a.get("before") is not None) or a["after"] <= a["before"]:
                    continue
                b = off["levels"].get(level, {})
                per_pair = b.get("m2l", 0.0) / b["pairs"] if b.get("pairs") else 0.0
                predicted = (a.get("pairs", 0) - b.get("pairs", 0)) * per_pair + (
                    a.get("interactions", 0) - b.get("interactions", 0)
                ) * p2p_rate
                rows.append((level, b, a, predicted))
            yield case, name, rows, ev, on, off


def summary(wheres):
    print(
        "| case | run | raised levels | Δ M2L + P2P s | predicted s | Δ evaluate s | evaluate s "
        "(without) | replan s | φ ÷ without | ∇φ ÷ without |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for where in wheres:
        for case, name, rows, ev, on, off in level_rows(where):
            measured = sum(
                a.get("m2l", 0.0) + a.get("p2p", 0.0) - b.get("m2l", 0.0) - b.get("p2p", 0.0)
                for _, b, a, _ in rows
            )
            predicted = sum(p for *_, p in rows)
            ratio = lambda k: number(on["main"][k]) / number(off["main"][k]) if number(off["main"][k]) else float("nan")
            print(
                f"| {case} | {name} | {','.join(str(l) for l, *_ in rows) or '–'} | "
                f"{measured:+.4f} | {predicted:+.4f} | {ev:+.4f} | "
                f"{float(off['near']['evaluate s']):.4f} | {on.get('build', {}).get('replan s', '–')} | "
                f"{ratio('φ'):.3f} | {ratio('∇φ'):.3f} |"
            )


def levels(where):
    print(
        "| case | run | level | p | ran (without) | crossover | V pairs | near interactions | M2L s | "
        "P2P s | Δ M2L + P2P s | predicted s | Δ evaluate s |"
    )
    print(
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"
    )
    for case, paths in cases(where, "").items():
        r = runs(paths)
        for name in ORDER:
            if not (name.endswith("+all") or name.endswith("+taken")) or name not in r:
                continue
            base = r.get(name.rsplit("+", 1)[0])
            if not base:
                continue
            on, off = r[name], base
            ev = float(on["near"]["evaluate s"]) - float(off["near"]["evaluate s"])
            p2p_rate = sum(e.get("p2p", 0.0) for e in off["levels"].values()) / max(
                1, sum(e.get("interactions", 0) for e in off["levels"].values())
            )
            for level, a in sorted(on.get("levels", {}).items()):
                if not (a.get("after") and a.get("before") is not None) or a["after"] <= a["before"]:
                    continue
                b = off["levels"].get(level, {})
                pairs0, pairs1 = b.get("pairs", 0), a.get("pairs", 0)
                i0, i1 = b.get("interactions", 0), a.get("interactions", 0)
                m0, m1 = b.get("m2l", 0.0), a.get("m2l", 0.0)
                p0, p1 = b.get("p2p", 0.0), a.get("p2p", 0.0)
                per_pair = m0 / pairs0 if pairs0 else 0.0
                predicted = (pairs1 - pairs0) * per_pair + (i1 - i0) * p2p_rate
                print(
                    f"| {case} | {name} | {level} | {b.get('p', '–')} | {b.get('ran', '–')} | "
                    f"{a['before']:.0f} → {a['after']:.0f} | {pairs0} → {pairs1} | {i0} → {i1} | "
                    f"{m0:.4f} → {m1:.4f} | {p0:.4f} → {p1:.4f} | {m1 + p1 - m0 - p0:+.4f} | "
                    f"{predicted:+.4f} | {ev:+.4f} |"
                )


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("errors", "runs", "levels", "summary"):
        sys.exit(__doc__)
    if sys.argv[1] == "summary":
        summary(sys.argv[2:])
        return
    {"errors": errors, "runs": all_runs, "levels": levels}[sys.argv[1]](sys.argv[2])


if __name__ == "__main__":
    main()
