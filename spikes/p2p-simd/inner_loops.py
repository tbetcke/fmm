#!/usr/bin/env python3
"""Inner-loop instruction counts of the spike's prototype entry points.

Usage (from the repository root):

    cargo build --release -p nd-fmm-spike-p2p-simd [--target x86_64-apple-darwin]
    python3 spikes/p2p-simd/inner_loops.py target/release/nd-fmm-spike-p2p-simd
    python3 spikes/p2p-simd/inner_loops.py \
        target/x86_64-apple-darwin/release/nd-fmm-spike-p2p-simd

It disassembles the binary with `objdump` (llvm-objdump on macOS), finds every
`kernels::{neon,avx2}::<prec>_<pot|grad>::<entry>` function, takes the innermost loop
(backward branch) that contains the inverse-square-root estimate (FRSQRTE /
vrsqrtps; FSQRT / vsqrtp for sqrt+div; the integer shift for the bit trick), and
counts its instructions by class. One iteration of a targets-in-lanes loop handles
one source against K target vectors, of a sources-in-lanes loop one source vector
against T targets, so FP ops per pair-vector are FP / K or FP / T. The model is
design simd-p2p.md §4.2 with the prototype's inverse square root: 9 (potential) or
15 (with gradient) plus the operations of the inverse square root, per pair and
lane. Prints Markdown.
"""

import re
import subprocess
import sys

# Vector FP operations of each inverse square root, including the estimate
# (crate::simd, `Rsqrt::OPS`), by (ISA, precision, rsqrt).
RSQRT_OPS = {
    # NEON: estimate 1; Newton N: 1 + 1 + 3N; Steps N: 1 + 3N; Poly M: 1 + M + 3.
    ("neon", "f32", "N2"): 8, ("neon", "f32", "N3"): 11,
    ("neon", "f32", "S2"): 7, ("neon", "f32", "S3"): 10,
    ("neon", "f32", "P2"): 6, ("neon", "f32", "P3"): 7, ("neon", "f32", "P4"): 8,
    ("neon", "f32", "SD"): 2,
    ("neon", "f64", "N3"): 11, ("neon", "f64", "N4"): 14,
    ("neon", "f64", "S2"): 7, ("neon", "f64", "S3"): 10, ("neon", "f64", "S4"): 13,
    ("neon", "f64", "P4"): 8, ("neon", "f64", "P6"): 10, ("neon", "f64", "P7"): 11,
    ("neon", "f64", "P8"): 12, ("neon", "f64", "N1P2"): 10, ("neon", "f64", "N1P3"): 11,
    ("neon", "f64", "SD"): 2,
    # AVX2: f32 estimate 1, f64 estimate 3 (two conversions); step form + 1 for ½x.
    ("avx2", "f32", "P2"): 6, ("avx2", "f32", "S1"): 5,
    ("avx2", "f64", "P5"): 11, ("avx2", "f64", "S2"): 10,
}

# The formulation of each entry-point flavour, by (ISA, precision); keep in step
# with `kernels::{neon, avx2}` (BestF32, BestF64 and the gk types).
FLAVOUR = {
    ("neon", "f32", "best"): "SD", ("neon", "f32", "gk"): "S2",
    ("neon", "f64", "best"): "SD", ("neon", "f64", "gk"): "S3",
    ("avx2", "f32", "best"): "P2", ("avx2", "f32", "gk"): "S1",
    ("avx2", "f64", "best"): "P5", ("avx2", "f64", "gk"): "S2",
}

NAME = re.compile(r"kernels::(neon|avx2)::(f32|f64)_(pot|grad)::(\w+?)(::h[0-9a-f]{16})?>?:?$")

ARM_FP = {"fmla", "fmls", "fmul", "fadd", "fsub", "frsqrte", "frsqrts", "fcmeq",
          "bic", "fsqrt", "fdiv", "fneg", "fmadd", "fmsub", "fnmsub", "fnmadd"}
ARM_LOAD = {"ld1r", "ld3r", "ld1", "ld3", "ldr", "ldp", "ldur", "ld2", "ld4"}
X86_FP_PREFIX = ("vfmadd", "vfnmadd", "vfmsub", "vfnmsub", "vmulp", "vaddp", "vsubp",
                 "vrsqrtps", "vcvtpd2ps", "vcvtps2pd", "vcmp", "vandnp", "vandp",
                 "vsqrtp", "vdivp", "vpsrlq", "vpsubq")


def disassemble(binary):
    out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "--demangle", binary],
                         capture_output=True, text=True, check=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            m = NAME.search(line.rstrip())
            cur = None
            if m is not None:
                cur = m.group(0)
                funcs[cur] = (m.groups()[:4], [])
            continue
        if cur is None:
            continue
        m = re.match(r"\s*([0-9a-f]+):\s+(\S+)\s*(.*)$", line)
        if m:
            funcs[cur][1].append((int(m.group(1), 16), m.group(2), m.group(3)))
    return funcs


def branch_target(mnem, ops, arm):
    if arm:
        if not (mnem == "b" or mnem.startswith("b.") or mnem in ("cbz", "cbnz", "tbz", "tbnz")):
            return None
    elif not mnem.startswith("j"):
        return None
    m = re.search(r"(?:0x)?([0-9a-f]{6,})\b", ops)
    return int(m.group(1), 16) if m else None


def classify(mnem, ops, arm):
    """fp (also with a folded memory operand), load (sources and broadcasts), spill
    (other stack traffic), move (register to register), shuffle, call or other
    (integer and loop control)."""
    if arm:
        base = mnem.split(".")[0]
        if base in ("bl", "blr"):
            return "call"
        if base in ARM_FP:
            return "fp"
        if "[sp" in ops:
            return "spill"
        if base in ARM_LOAD or base.startswith(("ld", "st")):
            return "load"
        if base in ("mov", "orr", "dup", "fmov"):
            return "move"
        if base in ("ext", "zip1", "zip2", "uzp1", "uzp2", "trn1", "trn2", "ins"):
            return "shuffle"
        return "other"
    if mnem.startswith("call"):
        return "call"
    mem = "(" in ops
    if mnem.startswith(X86_FP_PREFIX):
        return "fp"
    if mem and ("%rsp" in ops or "%rbp" in ops):
        return "spill"
    if mnem.startswith("vbroadcast") or (mem and mnem.startswith("v")):
        return "load"
    if mnem.startswith(("vmov", "mov")):
        return "move"
    if mnem.startswith(("vinsert", "vunpck", "vperm", "vshuf", "vblend", "vextract")):
        return "shuffle"
    return "other"


def inner_loop(insns, arm):
    key = ("frsqrte", "fsqrt") if arm else ("vrsqrtps", "vsqrtp", "vpsrlq")
    marks = [a for a, m, _ in insns if m.split(".")[0].startswith(key)]
    best = None
    for addr, mnem, ops in insns:
        t = branch_target(mnem, ops, arm)
        if t is not None and t <= addr and any(t <= k <= addr for k in marks):
            if best is None or addr - t < best[1] - best[0]:
                best = (t, addr)
    if best is None:
        return None
    return [(a, m, o) for a, m, o in insns if best[0] <= a <= best[1]]


def main():
    binary = sys.argv[1]
    funcs = disassemble(binary)
    rows = []
    for _, ((isa, prec, out, entry), insns) in sorted(funcs.items(), key=lambda kv: kv[1][0]):
        if entry == "list":
            continue
        arm = isa == "neon"
        loop = inner_loop(insns, arm)
        if loop is None:
            rows.append((isa, prec, out, entry, None))
            continue
        counts = {k: 0 for k in ("fp", "load", "spill", "move", "shuffle", "call", "other")}
        for _, m, o in loop:
            counts[classify(m, o, arm)] += 1
        rows.append((isa, prec, out, entry, (len(loop), counts)))
    print("| ISA | precision | output | entry | loop insns | FP | loads | spills | moves | shuffles | other | call | FP per pair-vector | model | FP / model |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for isa, prec, out, entry, res in rows:
        if res is None:
            print(f"| {isa} | {prec} | {out} | {entry} | no loop found | | | | | | | | | | |")
            continue
        n, c = res
        m = re.match(r"(til|sil)_[kt](\d)_(\w+)", entry)
        per, model, ratio = "", "", ""
        if m:
            k = int(m.group(2))
            tag = m.group(3)
            if tag in ("best", "gk"):
                r = FLAVOUR.get((isa, prec, tag))
            else:
                r = tag.split("_")[-1].upper()
            ops = RSQRT_OPS.get((isa, prec, r)) if r else None
            fp = c["fp"] / k
            per = f"{fp:.1f}"
            if ops is not None:
                mod = (15 if out == "grad" else 9) + ops
                model = f"{mod} ({r})"
                ratio = f"{fp / mod:.2f}"
        print(f"| {isa} | {prec} | {out} | {entry} | {n} | {c['fp']} | {c['load']} | {c['spill']} | "
              f"{c['move']} | {c['shuffle']} | {c['other']} | {'yes' if c['call'] else 'no'} | "
              f"{per} | {model} | {ratio} |")
    print()
    print("Entries missing from the table were merged by LLVM with an identical function "
          "(for example a `gk` entry with the `cand` entry of the same formulation and K); "
          "the surviving name is listed.")


if __name__ == "__main__":
    main()
