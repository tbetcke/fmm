#!/usr/bin/env python3
"""FMM3D's Helmholtz routines through tools/fmm3d/driver.f90 (Helmholtz CH1.4).

The Python side of the driver's `helmholtz-check` and `helmholtz` modes: writes the
problem files (`NDFMMHP1`), runs the driver, reads its outputs (`NDFMMHO1`) and parses
its lines; the error norms of proposal §9.1. Standard library only (Python 3.9, as on
Kathleen), so that tools/fixtures/gen_helmholtz_fmm3d.py and the R12 sweep below share it.

    python3 tools/fmm3d/helmholtz.py check [--driver PATH]
    python3 tools/fmm3d/helmholtz.py r12 --n N [--workload cube,sphere] [--driver PATH]
        [--work DIR] [--repeat R] [--kinds imaginary,real,complex] [--eps 1e-3,...]

`check` runs the two-point check and prints the identification of FMM3D's conventions
(`identify` below). `r12` is the sweep of docs/helmholtz/tasks/CH1.4.md: `hfmm3d` against
`h3ddirectcg` at 1,000 sampled sources, relative l2 of the potential and of the gradient
(§9.1), for purely imaginary k (risk R12) with real and complex k as controls; it prints a
Markdown table. The threads are OpenMP's (`OMP_NUM_THREADS`). The driver defaults to
$FMM3D_ROOT/default/bin/fmm3d_driver.
"""
import argparse
import array
import cmath
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

if sys.byteorder != "little":
    sys.exit("helmholtz.py: the driver's files are little-endian, this machine is not")

PROBLEM_MAGIC = b"NDFMMHP1"
OUTPUT_MAGIC = b"NDFMMHO1"


def default_driver():
    root = os.environ.get("FMM3D_ROOT")
    if not root:
        kathleen = os.environ.get("FMM_KATHLEEN_ROOT")
        if kathleen:
            root = os.path.join(kathleen, "fmm3d")
        elif os.path.isdir("/data/ucahtbe"):
            root = "/data/ucahtbe/fmm3d"
    if not root:
        sys.exit("helmholtz.py: set FMM3D_ROOT or pass --driver (tools/fmm3d/README.md)")
    return os.path.join(root, "default", "bin", "fmm3d_driver")


def write_problem(path, sources, charges, sampled, targets=(), seed=0):
    """A problem file of the driver's `helmholtz` mode (driver.f90's header)."""
    out = bytearray(struct.pack("<8s4q", PROBLEM_MAGIC, len(sources), len(sampled),
                                len(targets), seed))
    out += array.array("d", [c for p in sources for c in p]).tobytes()
    out += array.array("d", [v for q in charges for v in (q.real, q.imag)]).tobytes()
    out += array.array("q", sampled).tobytes()
    out += array.array("d", [c for p in targets for c in p]).tobytes()
    Path(path).write_bytes(bytes(out))


def _complex(values, start, count):
    return [complex(values[start + 2 * i], values[start + 2 * i + 1]) for i in range(count)]


def read_output(path):
    """(pot, grad, pottarg, gradtarg) of an output file; grad as [x, y, z] per point."""
    data = Path(path).read_bytes()
    magic, s, t = struct.unpack_from("<8s2q", data, 0)
    if magic != OUTPUT_MAGIC:
        sys.exit(f"helmholtz.py: not a Helmholtz output file: {path}")
    values = array.array("d")
    values.frombytes(data[24:])
    if len(values) != 8 * (s + t):
        sys.exit(f"helmholtz.py: {path} has {len(values)} values, expected {8 * (s + t)}")
    pot = _complex(values, 0, s)
    flat = _complex(values, 2 * s, 3 * s)
    grad = [flat[3 * i:3 * i + 3] for i in range(s)]
    pottarg = _complex(values, 8 * s, t)
    flat = _complex(values, 8 * s + 2 * t, 3 * t)
    gradtarg = [flat[3 * i:3 * i + 3] for i in range(t)]
    return pot, grad, pottarg, gradtarg


def _fields(line, prefix):
    """The key=value fields of a driver line (values may hold blanks, as `zk= a b`)."""
    body = line[len(prefix):].strip()
    out = {}
    key = None
    for word in body.split():
        if "=" in word:
            key, value = word.split("=", 1)
            out[key] = [value] if value else []
        elif key is not None:
            out[key].append(word)
    return out


def _numbers(words):
    return [float(w.replace("D", "E")) for w in words]


def run_check(driver):
    """The driver's two-point check: (records, passed). Each record has zk, routine, the
    four values pot1, grad1 (3), pot2, grad2 (3) as complex, and whether it is a record
    case (outside the draft's branch) and matched the draft."""
    proc = subprocess.run([driver, "helmholtz-check"], capture_output=True, text=True)
    records = []
    for line in proc.stdout.splitlines():
        if not line.startswith("helmholtz-check: zk="):
            continue
        f = _fields(line, "helmholtz-check:")
        zk = _numbers(f["zk"])

        def cx(key, count):
            v = _numbers(f[key])
            return [complex(v[2 * i], v[2 * i + 1]) for i in range(count)]

        records.append({
            "zk": complex(zk[0], zk[1]),
            "routine": f["routine"][0],
            "pot1": cx("pot1", 1)[0],
            "grad1": cx("grad1", 3),
            "pot2": cx("pot2", 1)[0],
            "grad2": cx("grad2", 3),
            "rel": float(f["rel"][0]),
            "record": "record" in f,
            "draft": (f.get("draft") or ["F"])[0] == "T",
        })
    if proc.returncode not in (0, 1) or not records:
        sys.exit(f"helmholtz.py: the check failed to run:\n{proc.stdout}\n{proc.stderr}")
    return records, proc.returncode == 0


# The two points of the check (driver.f90): q1 at (0,0,0), q2 at (1,0,0).
Q1, Q2 = complex(0.3, -0.7), complex(1.0, 0.0)


def candidates():
    """Conventions FMM3D might use: (name, factor, sign of k, gradient variable, charge)."""
    out = []
    for factor_name, factor in (("1/(4π)", 1 / (4 * math.pi)), ("1", 1.0)):
        for sign in (1, -1):
            for grad in ("target", "source"):
                for charge in ("q", "conj q"):
                    name = (f"factor {factor_name}, e^{{{'+' if sign > 0 else '−'}ikr}}, "
                            f"gradient in the {grad}, charge {charge}")
                    out.append((name, factor, sign, grad, charge))
    return out


def predict(zk, factor, sign, grad, charge):
    """pot1, grad1_x, pot2, grad2_x of the check under one candidate."""
    q1, q2 = (Q1, Q2) if charge == "q" else (Q1.conjugate(), Q2.conjugate())
    kk = sign * zk
    e = factor * cmath.exp(1j * kk)
    dg = (1j * kk - 1) * e  # d/dr of e^{ikr}/r at r = 1, times r: (ikr - 1) e^{ikr}/r^2
    g = 1.0 if grad == "target" else -1.0
    # The gradient in the target x of G(x, y) is dg (x - y)/r; x1 - x2 = (-1, 0, 0).
    return [q2 * e, -g * q2 * dg, q1 * e, g * q1 * dg]


def identify(record, tol=1e-12):
    """The names of the candidates that reproduce a check record to tol."""
    got = [record["pot1"], record["grad1"][0], record["pot2"], record["grad2"][0]]
    side = max(abs(v) for v in record["grad1"][1:] + record["grad2"][1:])
    names = []
    for name, factor, sign, grad, charge in candidates():
        want = predict(record["zk"], factor, sign, grad, charge)
        rel = max(abs(a - b) / abs(b) for a, b in zip(got, want))
        if rel <= tol and side <= tol * abs(want[0]):
            names.append(name)
    return names


def run_driver(driver, problem, prefix, zk, tokens, repeat=1):
    """Runs the `helmholtz` mode; {token: (seconds, outputs)}."""
    cmd = [driver, "helmholtz", str(problem), str(prefix), repr(zk.real), repr(zk.imag),
           ",".join(tokens), str(repeat)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"helmholtz.py: {' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}")
    out = {}
    for line in proc.stdout.splitlines():
        if line.startswith("helmholtz: token="):
            f = _fields(line, "helmholtz:")
            out[f["token"][0]] = (float(f["seconds"][0]), read_output(f["out"][0]))
    missing = [t for t in tokens if t not in out]
    if missing:
        sys.exit(f"helmholtz.py: no output for {missing}:\n{proc.stdout}")
    return out


def rel_l2(values, reference):
    """sqrt(sum |v - r|^2 / sum |r|^2) over complex values or lists of them (§9.1)."""
    num = den = 0.0
    for v, r in zip(values, reference):
        vs = v if isinstance(v, list) else [v]
        rs = r if isinstance(r, list) else [r]
        for a, b in zip(vs, rs):
            num += abs(a - b) ** 2
            den += abs(b) ** 2
    return math.sqrt(num / den) if den > 0 else math.sqrt(num)


def workload(name, n, rng):
    """Points of the R12 sweep: the unit cube (D = 1), or the surface of the sphere of
    diameter 1 centred in it."""
    if name == "cube":
        return [(rng.random(), rng.random(), rng.random()) for _ in range(n)]
    if name == "sphere":
        out = []
        for _ in range(n):
            x, y, z = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)
            r = 2 * math.sqrt(x * x + y * y + z * z)
            out.append((0.5 + x / r, 0.5 + y / r, 0.5 + z / r))
        return out
    sys.exit(f"helmholtz.py: unknown workload {name} (cube or sphere)")


def complex_charges(n, rng):
    """Charges with real and imaginary parts uniform in [-1, 1] (§9.1)."""
    return [complex(2 * rng.random() - 1, 2 * rng.random() - 1) for _ in range(n)]


# The k of the R12 sweep, as kD with D = 1 (docs/helmholtz/tasks/CH1.4.md).
R12_K = {
    "imaginary": [1j * v for v in (0.01, 0.1, 1, 10, 30, 50, 70, 100, 150, 200, 300, 1000)],
    "real": [complex(v) for v in (1, 10, 30, 100)],
    "complex": [complex(10, 1), complex(100, 10)],
}
R12_EPS = ["1e-3", "1e-6", "1e-9", "1e-12"]


def r12(args):
    driver = args.driver or default_driver()
    kinds = args.kinds.split(",")
    tokens = args.eps.split(",")
    work = Path(args.work or tempfile.mkdtemp(prefix="r12-"))
    work.mkdir(parents=True, exist_ok=True)
    threads = os.environ.get("OMP_NUM_THREADS", "OpenMP's default")
    print(f"R12 sweep: N = {args.n}, threads {threads}, driver {driver}, seed {args.seed}, "
          f"1,000 sampled sources, repeat {args.repeat} (fastest call, setup included)\n")
    print("| workload | N | kind | kD | eps | rel l2 φ | rel l2 ∇φ | max err / eps | s | "
          "s / real k |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in args.workload.split(","):
        rng = random.Random(args.seed)
        sources = workload(name, args.n, rng)
        charges = complex_charges(args.n, rng)
        sampled = sorted(rng.sample(range(args.n), min(1000, args.n)))
        problem = work / f"{name}-n{args.n}.bin"
        write_problem(problem, sources, charges, sampled, seed=args.seed)
        real_seconds = {}
        for kind in kinds:
            for zk in R12_K[kind]:
                prefix = work / f"{name}-n{args.n}-{kind}-{abs(zk):g}"
                out = run_driver(driver, problem, prefix, zk, ["direct"] + tokens,
                                 args.repeat)
                ref = out["direct"][1]
                for token in tokens:
                    seconds, (pot, grad, _, _) = out[token]
                    ep = rel_l2(pot, ref[0])
                    eg = rel_l2(grad, ref[1])
                    ratio = max(ep, eg) / float(token)
                    if kind == "real":
                        real_seconds[(abs(zk), token)] = seconds
                    base = real_seconds.get((abs(zk), token))
                    vs = f"{seconds / base:.2f}" if base and kind != "real" else "–"
                    kd = f"{zk.imag:g}i" if kind == "imaginary" else (
                        f"{zk.real:g}" if kind == "real" else f"{zk.real:g}+{zk.imag:g}i")
                    print(f"| {name} | {args.n} | {kind} | {kd} | {token} | {ep:.2e} | "
                          f"{eg:.2e} | {ratio:.2f} | {seconds:.3f} | {vs} |", flush=True)
                for path in work.glob(prefix.name + "-*.out"):
                    path.unlink()
        problem.unlink()


def check(args):
    driver = args.driver or default_driver()
    records, passed = run_check(driver)
    for r in records:
        names = identify(r)
        case = "record" if r["record"] else "check"
        print(f"{case} zk={r['zk']} {r['routine']}: rel {r['rel']:.2e}, matches "
              f"{len(names)} candidate(s): {'; '.join(names) or 'none'}")
    print("the draft's kernel on every checked case" if passed else
          "FMM3D does NOT compute the draft's kernel")
    sys.exit(0 if passed else 1)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="the two-point check and the identification")
    p.add_argument("--driver")
    p.set_defaults(run=check)
    p = sub.add_parser("r12", help="hfmm3d against h3ddirectcg over kD and eps")
    p.add_argument("--driver")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--workload", default="cube,sphere")
    p.add_argument("--kinds", default="real,imaginary,complex")
    p.add_argument("--eps", default=",".join(R12_EPS))
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--work", help="directory for the problem and output files")
    p.set_defaults(run=r12)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
