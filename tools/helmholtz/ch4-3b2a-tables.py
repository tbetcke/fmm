#!/usr/bin/env python3
"""tools/helmholtz/ch4-3b2a-tables.py: the tables of Helmholtz CH4.3b2a's report (the diagonal
form's time rule at matched accuracy; docs/helmholtz/tasks/CH4.3b2a.md), from the launches of
`tools/helmholtz/ch4-3b2a-runs.sh`.

    python3 tools/helmholtz/ch4-3b2a-tables.py errors <machine>=<dir> …
    python3 tools/helmholtz/ch4-3b2a-tables.py choices <machine>=<dir> …
    python3 tools/helmholtz/ch4-3b2a-tables.py summary <machine>=<dir> …

Each launch runs four settings under the library's defaults: the defaults (the waves), the time
rule at the machine's costs (`/diagauto`), at costs that force the form where a class passes and
RCR elsewhere (`/diagauto:form`) and at costs that force RCR (`/diagauto:rcr`), on every level
the rule reaches (k a ≥ 2.8), each level at its resolving degree.

`errors` prints per launch each time-rule setting's errors (φ, ∇φ) and evaluation over the
defaults'. `choices` prints per level with V pairs in the rule's range the options measured (the
waves from the defaults' run on a level with waves, the form from the `form` run where it ran the
form, RCR from the `rcr` run, or from the defaults' run on a level RCR runs there), the machine
rule's choice, scored by the option it took as timed in the forced run (a level's own seconds
vary from launch to launch on small levels; CH4.3b1 §4.3), and whether it is within 1.05× of the
fastest. `summary` prints per machine the levels, those within 1.05×, the seconds lost and at the
best, and per ε. No timing is asserted; the numbers go into
fmm-validate/results/helmholtz-ch4-3b2a.md.
"""

import pathlib
import re
import sys

MIN_KA = 2.8
WITHIN = 1.05


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


def setting(m2l):
    """The setting of a row from its column "M2L"."""
    if "diagauto:form" in m2l:
        return "form"
    if "diagauto:rcr" in m2l:
        return "rcr"
    if "diagauto" in m2l:
        return "machine"
    return "defaults"


def launches(arg):
    machine, where = arg.split("=", 1)
    return machine, sorted(pathlib.Path(where).glob("*.md"))


def number(x):
    try:
        return float(x)
    except ValueError:
        return None


def runs(path):
    """Per setting: the main row and, per level, (p, ran, seconds) of "M2L per level", and
    k a per level from "Time rule"."""
    t = tables(path)
    main = {setting(r["M2L"]): r for r in t.get("", [])}
    levels = {}
    for r in t.get("M2L per level", []):
        if r["level"] == "Σ":
            # "evaluate 0.8569 s (of 2; first 0.8801 s)": the evaluation to 0.1 ms.
            m = re.match(r"evaluate ([0-9.]+) s", r["ran"])
            name = setting(r["M2L"])
            if m and name in main:
                main[name]["evaluate s"] = m.group(1)
            continue
        levels.setdefault(setting(r["M2L"]), {})[int(r["level"])] = (
            int(r["p"]),
            r["ran"],
            float(r["M2L s"]),
        )
    ka = {}
    for r in t.get("Time rule", []):
        ka[int(r["level"])] = float(r["k a"])
    return main, levels, ka


def kind(ran):
    if ran.startswith("planewave"):
        return "waves"
    if ran.startswith("diagonal"):
        return "form"
    if ran.startswith("rotation"):
        return "rcr"
    return "other"


def choices(path):
    """Per level in range with V pairs: (level, p, k a, options {name: seconds}, the defaults'
    M2L, the machine run's choice, its own seconds, the forced runs' classes)."""
    main, levels, ka = runs(path)
    if not {"defaults", "machine", "form", "rcr"} <= set(levels):
        return []
    out = []
    for l, (p, ran, seconds) in sorted(levels["defaults"].items()):
        if l not in ka or ka[l] < MIN_KA:
            continue
        own = kind(ran)
        if own == "other":
            continue
        options = {}
        if own == "waves":
            options["waves"] = seconds
        else:
            options["rcr"] = seconds
        p_f, ran_f, s_f = levels["form"][l]
        if kind(ran_f) == "form":
            options["form"] = s_f
        p_r, ran_r, s_r = levels["rcr"][l]
        if kind(ran_r) == "rcr" and "rcr" not in options:
            options["rcr"] = s_r
        p_m, ran_m, s_m = levels["machine"][l]
        took = kind(ran_m)
        out.append((l, p_m, ka[l], options, own, took, s_m, ran_f))
    return out


def eps_of(path):
    m = re.search(r"-(1e-\d)$", path.stem)
    return m.group(1) if m else "?"


def errors_table(args):
    print("| machine | launch | setting | φ ÷ defaults | ∇φ ÷ defaults | evaluation ÷ defaults "
          "| build ÷ defaults | levels taken from the waves |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for arg in args:
        machine, paths = launches(arg)
        for path in paths:
            main, levels, _ = runs(path)
            if "defaults" not in main:
                continue
            d = main["defaults"]
            for name in ("machine", "form", "rcr"):
                if name not in main:
                    continue
                r = main[name]
                ratio = lambda key: float(r[key]) / float(d[key]) if float(d[key]) > 0 else 0.0
                taken = []
                for l, (_, ran, _) in sorted(levels.get(name, {}).items()):
                    own = kind(levels["defaults"][l][1])
                    if own == "waves" and kind(ran) != "waves":
                        taken.append(f"{l}:{kind(ran)}")
                print(f"| {machine} | {path.stem} | {name} | {ratio('φ'):.3f} | {ratio('∇φ'):.3f} "
                      f"| {ratio('evaluate s'):.3f} | {ratio('build s'):.3f} "
                      f"| {', '.join(taken) or '–'} |")


def choices_table(args):
    print("| machine | launch | level | p | k a | defaults | waves s | form s | RCR s | took "
          "| its own s | ÷ best |")
    print("| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |")
    fmt = lambda x: f"{x:.4f}" if x is not None else "–"
    for arg in args:
        machine, paths = launches(arg)
        for path in paths:
            for l, p, ka, options, own, took, s_m, _ in choices(path):
                best = min(options.values())
                scored = options.get(took, s_m)
                print(f"| {machine} | {path.stem} | {l} | {p} | {ka:.2f} | {own} "
                      f"| {fmt(options.get('waves'))} | {fmt(options.get('form'))} "
                      f"| {fmt(options.get('rcr'))} | {took} | {s_m:.4f} "
                      f"| {scored / best:.2f} |")


def summary_table(args):
    print("| machine | ε | levels | within 1.05× | seconds lost | seconds at the best "
          "| taken from the waves (form, RCR) |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for arg in args:
        machine, paths = launches(arg)
        groups = {}
        for path in paths:
            for l, p, ka, options, own, took, s_m, _ in choices(path):
                for key in (eps_of(path), "all"):
                    g = groups.setdefault(key, [0, 0, 0.0, 0.0, 0, 0])
                    best = min(options.values())
                    scored = options.get(took, s_m)
                    g[0] += 1
                    g[1] += scored <= WITHIN * best
                    g[2] += scored - best
                    g[3] += best
                    if own == "waves" and took == "form":
                        g[4] += 1
                    if own == "waves" and took == "rcr":
                        g[5] += 1
        for key in sorted(groups, key=lambda k: (k == "all", k)):
            n, ok, lost, at_best, form, rcr = groups[key]
            print(f"| {machine} | {key} | {n} | {ok} | {lost:.2f} | {at_best:.2f} "
                  f"| {form}, {rcr} |")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    {"errors": errors_table, "choices": choices_table, "summary": summary_table}[sys.argv[1]](
        sys.argv[2:]
    )


if __name__ == "__main__":
    main()
