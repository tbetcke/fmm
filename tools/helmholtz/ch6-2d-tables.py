#!/usr/bin/env python3
"""tools/helmholtz/ch6-2d-tables.py: the tables of Helmholtz CH6.2d's report
(docs/helmholtz/tasks/CH6.2d.md): FMM3D's exponential M2L per level beside ours.

    tools/helmholtz/ch6-2d-tables.py times <machine>=<ch6-2d-dir>,<ch6-2b-dir> [...]
    tools/helmholtz/ch6-2d-tables.py scan <ch6-2d-dir> <ch6-1e-dir>

`times` reads, per machine, tools/helmholtz/ch6-2d-runs.sh's `times-<case>-<variant>-<a|b>.log`
(the patched driver's `hfmm3d-level:` lines; per launch and eps the last of its REPEAT
calls, the two launches' mean, the spread the larger over the smaller) and its
`errors-<case>.md`, and CH6.2b's outputs of the same machine at one rank × all threads
(`ch6-2b-tables.py`'s reader: per level the strategies' M2L seconds, the mean of the forward
and the reverse launch, and the main rows' errors). FMM3D's levels are matched to ours by
κ: our level is round(log2(kD / Re κ_FMM3D)), our root box being the unit cube (D = 1).
FMM3D's M2L on a plane-wave level is its multipoles to waves plus the parents' loop's wall
time times the plane-wave share of its thread-seconds (the rest of that loop is list 3);
its setup (the quadrature and the factors, built in every call) and list 3 and list 4 are
apart. Where CH6.2b's outputs come from CH6.2d's harness, our exponential M2L's cell adds
its two passes in brackets, [waves + merge and shift]. Prints per machine and case a table
of the levels and a summary.

`scan` reads CH6.2d's `scan-<case>.md` (FMM3D at each level alone on plane waves, against
its point-and-shoot run) and CH6.1e's `scan-*.md` (ours at each level alone with the
exponential M2L, against Rotation) and prints, per case, eps and level, both codes' errors
of φ and ∇φ as measured (absolute) and each over its own reference run's, at FMM3D's
precision of that eps (iprec 1 at 1e-3, 2 at 1e-6: hwts3e's rule; ours at the same
precision in CH6.1e's scan), then the reference runs.

Python 3.9 standard library only.
"""

import importlib.util
import math
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ch6_2b", os.path.join(HERE, "ch6-2b-tables.py"))
ch6_2b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ch6_2b)

FIELDS = ("ilev", "zkre", "zkim", "nterms", "mode", "nlams", "nwaves", "nboxes", "setup",
          "list4", "m2e", "gather", "pw_ts", "list3_ts", "ps", "ps_list3", "hf_list4")
MODES = {0: "–", 1: "pw", 2: "ps"}


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def spread(values):
    return max(values) / min(values) if values and min(values) > 0 else float("nan")


def ratio(a, b):
    return a / b if b and b > 0 and not math.isnan(b) else float("nan")


def case_of(workload):
    """The run script's case base of a harness workload name: `W4 cube (N = …)` -> w4-cube."""
    words = workload.split()
    base = words[0].lower()
    if base == "w4":
        return f"w4-{words[1]}"
    return base


def keyvalues(line):
    tokens = line.split()
    values = {}
    for i, t in enumerate(tokens[:-1]):
        if t.endswith("="):
            values[t[:-1]] = tokens[i + 1]
    return values


def read_fmm3d_log(path):
    """{eps token: (levels of the last call, the call's seconds)} of one driver log."""
    result, calls, on = {}, [], False
    current = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("helmholtz: problem"):
                on, calls, current = True, [], []
            elif not on:
                continue
            elif line.startswith("hfmm3d-level:"):
                kv = keyvalues(line)
                level = {k: float(kv[k]) for k in FIELDS}
                level["ilev"] = int(level["ilev"])
                level["mode"] = int(level["mode"])
                current.append(level)
            elif line.startswith("hfmm3d-steps:"):
                calls.append(current)
                current = []
            elif line.startswith("helmholtz: token="):
                kv = keyvalues(line.replace("helmholtz: ", ""))
                token = re.search(r"token=(\S+)", line).group(1)
                result[token] = (calls[-1], float(kv["seconds"]))
                calls = []
    return result


def fmm3d_m2l(level):
    """FMM3D's M2L on a level: (M2L, list 3) seconds; plane waves split the parents' loop."""
    if level["mode"] == 1:
        busy = level["pw_ts"] + level["list3_ts"]
        share = level["pw_ts"] / busy if busy > 0 else 1.0
        return level["m2e"] + level["gather"] * share, level["gather"] * (1 - share)
    if level["mode"] == 2:
        return level["ps"], level["ps_list3"]
    return 0.0, 0.0


def read_errors(path):
    """{(eps, run label): (φ, ∇φ)} of a `helmholtz_fmm3d errors` table."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.startswith("| W") or line.startswith("| Workload"):
                continue
            c = ch6_2b.cells(line)
            out[(c[3], c[4])] = (float(c[7]), float(c[8]))
    return out


def read_ours(directory):
    """CH6.2b's outputs at one rank: {(case, kd, eps): {"levels": {level: {strategy: [s],
    "p", "pairs", "ran"}}, "evaluate": {strategy: [s]}, "errors": {strategy: (φ, ∇φ)}}}."""
    ours = defaultdict(lambda: {"levels": defaultdict(lambda: defaultdict(list)),
                                "info": {}, "evaluate": defaultdict(list), "errors": {}})
    best_split = {}
    files = []
    for name in sorted(os.listdir(directory)):
        m = re.match(r"(matrix|large)-(.+)-(1x\d+)-(forward|reverse)\.md$", name)
        if m:
            files.append((name, m))
            best_split[m.group(2)] = max(best_split.get(m.group(2), 0), int(m.group(3)[2:]))
    for name, m in files:
        if int(m.group(3)[2:]) != best_split[m.group(2)]:
            continue
        levels, rows = ch6_2b.read(os.path.join(directory, name))
        for c in levels:
            workload, n, kd, eps, setting, rt, level, p, ran, pairs, seconds, per_pair = c[:12]
            key = (case_of(workload), kd, eps.split("= ")[1])
            s = ch6_2b.strategy_of(setting)
            if level == "Σ":
                ours[key]["evaluate"][s].append(float(re.match(r"evaluate ([0-9.]+) s", ran).group(1)))
                continue
            ours[key]["levels"][int(level)][s].append(int(pairs) * float(per_pair) * 1e-9)
            # The passes of merge and shift (CH6.2d's harness; absent in CH6.2b's outputs).
            if s == "planewave" and len(c) >= 14 and c[12] != "–":
                ours[key]["levels"][int(level)]["waves"].append(float(c[12]))
                ours[key]["levels"][int(level)]["merge"].append(float(c[13]))
            info = ours[key]["info"].setdefault(int(level), {"p": p, "pairs": pairs, "ran": {}})
            info["ran"][s] = ran
        order = ch6_2b.STRATEGIES if m.group(4) == "forward" else ch6_2b.STRATEGIES[::-1]
        groups = defaultdict(list)
        for c in rows:
            groups[(c[0], c[2], c[3])].append(c)
        for (workload, kd, eps), group in groups.items():
            for s, c in zip(order, group):
                ours[(case_of(workload), kd, eps.split("= ")[1])]["errors"][s] = (float(c[16]), float(c[17]))
    return ours


def times(args):
    summary = []
    for arg in args:
        machine, dirs = arg.split("=", 1)
        mine, theirs = dirs.split(",", 1)
        ours = read_ours(theirs)
        # fmm3d[(case, variant)][eps] = [(levels, seconds) per launch]
        fmm3d = defaultdict(lambda: defaultdict(list))
        for name in sorted(os.listdir(mine)):
            m = re.match(r"times-(.+?)-(fastker-timed|timed)-([ab])\.log$", name)
            if not m:
                continue
            for eps, value in read_fmm3d_log(os.path.join(mine, name)).items():
                fmm3d[(m.group(1), m.group(2))][eps].append(value)
        print(f"## {machine}\n")
        for (case, variant) in sorted(fmm3d, key=lambda k: (k[0].rsplit("-", 1)[0], float(k[0].rsplit("-", 1)[1]), k[1])):
            base, kd = case.rsplit("-", 1)
            errs = read_errors(os.path.join(mine, f"errors-{case}.md"))
            for eps in sorted(fmm3d[(case, variant)], key=lambda e: -float(e)):
                launches = fmm3d[(case, variant)][eps]
                if len(launches) != 2:
                    print(f"warning: {machine} {case} {variant} {eps}: {len(launches)} launches",
                          file=sys.stderr)
                o = ours.get((base, kd, eps))
                print(f"### {case}, eps = {eps}, FMM3D {variant}\n")
                print("| our level | FMM3D level | Re κ | FMM3D p | regime | waves Σ Nₖ | boxes "
                      "| FMM3D M2L s | spread | setup s | list 3 s | list 4 s | our p | V pairs "
                      "| ours exp. M2L s | ours Rotation s | ours fastest | FMM3D ÷ ours exp. |")
                print("| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: "
                      "| ---: | ---: | ---: | ---: | ---: | --- | ---: |")
                tot = defaultdict(float)
                # FMM3D's levels by our level (κ), and ours, in one list.
                theirs = {}
                for i, level in enumerate(launches[0][0]):
                    ours_level = (round(math.log2(float(kd) / level["zkre"]))
                                  if level["zkre"] > 0 else level["ilev"])
                    theirs[ours_level] = (level, [l[0][i] for l in launches])
                rows = sorted(set(theirs) | set(o["info"] if o else ()))
                for ours_level in rows:
                    level, per = theirs.get(ours_level, (None, None))
                    info = o["info"].get(ours_level) if o else None
                    if (level is None or level["mode"] == 0) and not info:
                        continue
                    cells = [str(ours_level)]
                    m2l = []
                    if level is None:
                        cells += ["–"] * 11
                    else:
                        m2l = [fmm3d_m2l(l)[0] for l in per]
                        l3 = mean([fmm3d_m2l(l)[1] for l in per])
                        l4 = mean([l["list4"] + l["hf_list4"] for l in per])
                        setup = mean([l["setup"] for l in per])
                        cells += [str(level["ilev"]), f"{level['zkre']:.3g}",
                                  str(int(level["nterms"])), MODES[level["mode"]],
                                  str(int(level["nwaves"])) if level["mode"] == 1 else "–",
                                  str(int(level["nboxes"]))]
                        if level["mode"]:
                            cells += [f"{mean(m2l):.4f}", f"{spread(m2l):.2f}", f"{setup:.4f}",
                                      f"{l3:.4f}", f"{l4:.4f}"]
                            tot["fmm3d"] += mean(m2l)
                            tot["setup"] += setup
                            tot["list3"] += l3
                        else:
                            m2l = []
                            cells += ["–"] * 5
                    if info:
                        lv = o["levels"][ours_level]
                        pw = mean(lv["planewave"])
                        rot = mean(lv["rotation"])
                        methods = {}
                        for s in ch6_2b.STRATEGIES:
                            if lv[s]:
                                methods.setdefault(info["ran"].get(s, s), mean(lv[s]))
                        fastest = min(methods, key=methods.get)
                        ran = info["ran"].get("planewave", "planewave")
                        tag = "" if ran.startswith("planewave") else f" ({ran})"
                        if lv["waves"]:
                            tag += f" [{mean(lv['waves']):.4f} + {mean(lv['merge']):.4f}]"
                        cells += [info["p"], info["pairs"], f"{pw:.4f}{tag}", f"{rot:.4f}",
                                  f"{fastest} {methods[fastest]:.4f}",
                                  f"{ratio(mean(m2l), pw):.2f}" if m2l else "–"]
                        tot["ours_pw"] += pw
                        tot["ours_rot"] += rot
                        tot["ours_best"] += methods[fastest]
                    else:
                        cells += ["–"] * 6
                    print("| " + " | ".join(cells) + " |")
                calls = [l[1] for l in launches]
                e_f = errs.get((eps, variant))
                e_o = o["errors"].get("planewave") if o else None
                ev = mean(o["evaluate"]["planewave"]) if o else float("nan")
                ev_auto = mean(o["evaluate"]["auto"]) if o else float("nan")
                print(f"| Σ | | | | | | | {tot['fmm3d']:.4f} | | {tot['setup']:.4f} | "
                      f"{tot['list3']:.4f} | | | | {tot['ours_pw']:.4f} | {tot['ours_rot']:.4f} "
                      f"| best {tot['ours_best']:.4f} | {ratio(tot['fmm3d'], tot['ours_pw']):.2f} |\n")
                summary.append((machine, case, eps, variant, tot, mean(calls), spread(calls),
                                ev, ev_auto, e_f, e_o))
    print("## Summary\n")
    print("| machine | case | eps | FMM3D | FMM3D M2L Σ | its setup Σ | ours exp. M2L Σ "
          "| ours best per level | FMM3D ÷ ours exp. | FMM3D call s | spread | ours eval. s exp. "
          "| ours eval. s Auto | FMM3D φ, ∇φ | ours exp. φ, ∇φ |")
    print("| --- | --- | --- | --- | " + " | ".join(["---:"] * 9) + " | --- | --- |")
    for machine, case, eps, variant, tot, call, sp, ev, ev_auto, e_f, e_o in summary:
        fe = f"{e_f[0]:.2e}, {e_f[1]:.2e}" if e_f else "–"
        oe = f"{e_o[0]:.2e}, {e_o[1]:.2e}" if e_o else "–"
        print(f"| {machine} | {case} | {eps} | {variant} | {tot['fmm3d']:.3f} | {tot['setup']:.3f} "
              f"| {tot['ours_pw']:.3f} | {tot['ours_best']:.3f} | {ratio(tot['fmm3d'], tot['ours_pw']):.2f} "
              f"| {call:.3f} | {sp:.2f} | {ev:.3f} | {ev_auto:.3f} | {fe} | {oe} |")


def read_ours_scan(directory):
    """CH6.1e's scan: {(case, kd, eps): {"rotation": (φ, ∇φ), (level, iprec): (φ, ∇φ)}}."""
    out = defaultdict(dict)
    for name in sorted(os.listdir(directory)):
        if not re.match(r"scan-.*\.md$", name):
            continue
        with open(os.path.join(directory, name), encoding="utf-8") as f:
            for line in f:
                if not line.startswith("| W"):
                    continue
                c = ch6_2b.cells(line)
                if c[4] != "complex" or c[2].find("i") >= 0:
                    continue
                key = (case_of(c[0]), c[2], c[3].split("= ")[1])
                errors = (float(c[16]), float(c[17]))
                m = re.match(r"planewave (\d+):i(\d)", c[7])
                if c[7] == "rotation":
                    out[key]["rotation"] = errors
                elif m:
                    out[key][(int(m.group(1)), int(m.group(2)))] = errors
    return out


def scan(mine, theirs):
    ours = read_ours_scan(theirs)
    print("| case | eps | our level | Re κ | FMM3D p | FMM3D level alone φ, ∇φ | ÷ FMM3D ps | ours iprec "
          "| our level alone φ, ∇φ | ÷ Rotation |")
    print("| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- |")
    refs = []
    for name in sorted(os.listdir(mine)):
        m = re.match(r"scan-(.+)-([0-9.]+)\.md$", name)
        if not m:
            continue
        base, kd = m.group(1), m.group(2)
        errs = read_errors(os.path.join(mine, name))
        log = read_fmm3d_log(os.path.join(mine, "fmm3d", f"scan-{base}-{kd}-fmm3d.log"))
        for eps in sorted({e for e, _ in errs}, key=lambda e: -float(e)):
            iprec = 1 if float(eps) >= 5e-4 else 2 if float(eps) >= 5e-7 else 3
            levels = {l["ilev"]: l for l in log[eps][0]}
            ps = errs[(eps, "ps")]
            o = ours.get((base, kd, eps), {})
            refs.append((base, kd, eps, ps, errs.get((eps, "fmm3d")), o.get("rotation")))
            for (e, label), values in sorted(errs.items()):
                if e != eps or not label.startswith("pw"):
                    continue
                ilev = int(label[2:])
                level = levels.get(ilev)
                if not level or level["mode"] != 1:
                    continue
                ours_level = round(math.log2(float(kd) / level["zkre"]))
                mine_ = o.get((ours_level, iprec))
                rot = o.get("rotation")
                cells = [f"{base}@{kd}", eps, str(ours_level), f"{level['zkre']:.3g}",
                         str(int(level["nterms"])), f"{values[0]:.2e}, {values[1]:.2e}",
                         f"{ratio(values[0], ps[0]):.3g}, {ratio(values[1], ps[1]):.3g}",
                         str(iprec)]
                if mine_ and rot:
                    cells += [f"{mine_[0]:.2e}, {mine_[1]:.2e}",
                              f"{ratio(mine_[0], rot[0]):.3f}, {ratio(mine_[1], rot[1]):.3f}"]
                else:
                    cells += ["–", "–"]
                print("| " + " | ".join(cells) + " |")
    print("\n| case | eps | FMM3D point-and-shoot φ, ∇φ | FMM3D as it is φ, ∇φ | ours Rotation φ, ∇φ |")
    print("| --- | --- | --- | --- | --- |")
    for base, kd, eps, ps, f, rot in refs:
        fmt = lambda v: f"{v[0]:.2e}, {v[1]:.2e}" if v else "–"
        print(f"| {base}@{kd} | {eps} | {fmt(ps)} | {fmt(f)} | {fmt(rot)} |")


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "times":
        times(sys.argv[2:])
    elif len(sys.argv) == 4 and sys.argv[1] == "scan":
        scan(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
