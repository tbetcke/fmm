# Helmholtz CH3.2a: the Dense M2L

Helmholtz task CH3.2a ([docs/helmholtz/tasks/CH3.2a.md](../../docs/helmholtz/tasks/CH3.2a.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§5.3 (Dense) and §10 CH3.2. Measured on 2026-11-02 on the M6 (Mac mini, Apple M6, 24 GB;
[tools/m6/machine.md](../../tools/m6/machine.md)) from branch `tbetcke/h-ch3-2a`, release
build, rustc 1.99.0, BLAS threads 1, one thread, f64, every launch behind a quiet gate
(§3: the other processes at most 1.5 cores before a launch and 2 cores during it; none
used more than 1.1). The times are indicative only (the M6's launches vary by a few per
cent at one thread, and by up to 25 % at six, CH3.4c); nothing here is timed for a gate,
and nothing decides a default.

## Summary

- **Dense agrees with Rotation to rounding.** Every one of the 316 offsets on levels 2–4,
  for five k̂ (real and complex, |k̂| a from 0.005 to 17), one and two multipole planes:
  at most 3.7e-14 per unit of conditioning in the FMM-weighted measure (`nd-fmm-tables`'s
  `tests/helmholtz/dense.rs`; the bound 2e-13 is Rotation's against `rcr`), 3.4e-15
  against `rcr` directly. Pair by pair in the operator at most 7.2e-16 relative L2
  (`nd-fmm-exec`'s `tests/helmholtz/dense.rs`, within 1e-14); in the FMM at most 7.9e-16
  (φ) and 4.6e-16 (∇φ) of Rotation's output (within 1e-14), with the same errors against the direct sum to the
  printed digits, the same bits at 1 and 2 threads, pair by pair, and on P ranks the
  one-rank FMM's (`tests/helmholtz_fmm.rs`, at 1, 2 and 4 ranks).
- **Per pair, Dense is faster than Rotation only at small p.** On one level with the 316
  offsets per pass: 2.5–2.7× faster at p = 3, 1.3–1.7× at p = 4–5, level at p = 6–7, then
  slower, 0.42× Rotation's speed at p = 12 and 0.22× at p = 20 (§1). Dense reads a
  2(p + 1)⁴-value table per pair (160 KB at p = 9) and is bound by memory; Rotation's
  O(p³) work runs from the cache. The calibrated degrees (CH3.3) are at least 7 on the
  low-frequency levels at ε ≤ 10⁻³ (15 at 10⁻⁶), so Dense applied pair by pair pays only
  above ε = 10⁻³.
- **End to end, Dense is level with Rotation at p = 7 and slower above** (§2): 0.97× the
  evaluation time of Rotation on the W4 cube at ε = 10⁻³ (p = 7 on every level) and 1.03×
  on W3 at kD = 30 (p = 7–12), 1.5–1.9× on W1 at 10⁻³ and 2.0–2.6× at 10⁻⁶, with the same
  errors to the printed digits; the build adds up to 1.2 s of tables. So Dense stays
  opt-in, with Rotation the default, and gets no rule of its own here.
- **Tables**: 316 · 2 (p + 1)⁴ values per level, 21 MB at p = 7, 331 MB at p = 15,
  983 MB at p = 20 in f64; built in 0.02, 0.4 and 1.4 s per level (n RCR applications
  per offset, O(p⁵)).
- **What this means for CH3.2b and CH3.2c**: the dense tables pay where they are read
  once for many pairs, which is what the batched M2L does (CH3.2c); Classes (CH3.2b)
  reads 16 class tables instead of 316, about 20 times less, plus the transforms.

## 1. The M2L per pair

`helmholtz_m2l` (`--ps 3,4,5,6,7,8,9,10,12,15,18,20`): one level (level 2 of three,
k̂ = 1 and 1 + 0.1i on the level), each pass the 316 offsets in turn, each with its own
multipole and local, the median over 15 batches; Dense's locals of one pass first compared
with Rotation's (the largest relative L2 of a local, last column). Per real k the real
multipoles of real charges (one plane) and complex multipoles (two planes); complex k
always two. ns per pair, the mean of two launches, which differ by at most 1.7 % from
p = 4 (5 %, 3 ns, at p = 3); build seconds the larger and errors the worse of the two:

| p | k̂ | planes | Rotation ns/pair | Dense ns/pair | Rotation / Dense | Dense MB | Dense build s | Dense against Rotation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 1 | 160 | 62 | 2.55 | 1 | 0.00 | 4.8e-16 |
| 3 | 1 | 2 | 195 | 72 | 2.71 | 1 | 0.00 | 6.5e-16 |
| 3 | 1 + 0.1i | 2 | 197 | 72 | 2.74 | 1 | 0.00 | 6.1e-16 |
| 4 | 1 | 1 | 254 | 146 | 1.73 | 3 | 0.00 | 4.3e-16 |
| 4 | 1 | 2 | 311 | 190 | 1.63 | 3 | 0.00 | 5.1e-16 |
| 4 | 1 + 0.1i | 2 | 312 | 190 | 1.64 | 3 | 0.00 | 5.1e-16 |
| 5 | 1 | 1 | 384 | 294 | 1.30 | 7 | 0.00 | 5.6e-16 |
| 5 | 1 | 2 | 472 | 312 | 1.51 | 7 | 0.00 | 5.0e-16 |
| 5 | 1 + 0.1i | 2 | 468 | 312 | 1.50 | 7 | 0.00 | 6.0e-16 |
| 6 | 1 | 1 | 554 | 542 | 1.02 | 12 | 0.01 | 5.8e-16 |
| 6 | 1 | 2 | 680 | 700 | 0.97 | 12 | 0.01 | 5.0e-16 |
| 6 | 1 + 0.1i | 2 | 686 | 700 | 0.98 | 12 | 0.01 | 5.1e-16 |
| 7 | 1 | 1 | 774 | 865 | 0.90 | 21 | 0.02 | 5.7e-16 |
| 7 | 1 | 2 | 954 | 962 | 0.99 | 21 | 0.02 | 5.4e-16 |
| 7 | 1 + 0.1i | 2 | 960 | 962 | 1.00 | 21 | 0.02 | 5.3e-16 |
| 8 | 1 | 1 | 1016 | 1399 | 0.73 | 33 | 0.03 | 5.8e-16 |
| 8 | 1 | 2 | 1266 | 1788 | 0.71 | 33 | 0.03 | 5.6e-16 |
| 8 | 1 + 0.1i | 2 | 1266 | 1788 | 0.71 | 33 | 0.03 | 5.6e-16 |
| 9 | 1 | 1 | 1314 | 2144 | 0.61 | 51 | 0.05 | 9.2e-16 |
| 9 | 1 | 2 | 1635 | 2318 | 0.71 | 51 | 0.05 | 5.3e-16 |
| 9 | 1 + 0.1i | 2 | 1640 | 2319 | 0.71 | 51 | 0.04 | 5.8e-16 |
| 10 | 1 | 1 | 1665 | 3173 | 0.52 | 74 | 0.07 | 6.4e-16 |
| 10 | 1 | 2 | 2089 | 3966 | 0.53 | 74 | 0.07 | 7.1e-16 |
| 10 | 1 + 0.1i | 2 | 2090 | 3964 | 0.53 | 74 | 0.07 | 6.3e-16 |
| 12 | 1 | 1 | 2620 | 6276 | 0.42 | 144 | 0.15 | 1.1e-15 |
| 12 | 1 | 2 | 3280 | 7712 | 0.43 | 144 | 0.15 | 5.7e-16 |
| 12 | 1 + 0.1i | 2 | 3289 | 7706 | 0.43 | 144 | 0.14 | 6.5e-16 |
| 15 | 1 | 1 | 4634 | 14312 | 0.32 | 331 | 0.39 | 8.5e-16 |
| 15 | 1 | 2 | 5830 | 16044 | 0.36 | 331 | 0.39 | 6.2e-16 |
| 15 | 1 + 0.1i | 2 | 5826 | 16060 | 0.36 | 331 | 0.38 | 6.8e-16 |
| 18 | 1 | 1 | 7323 | 29070 | 0.25 | 659 | 0.87 | 7.7e-16 |
| 18 | 1 | 2 | 9292 | 35848 | 0.26 | 659 | 0.87 | 6.6e-16 |
| 18 | 1 + 0.1i | 2 | 9270 | 35836 | 0.26 | 659 | 0.85 | 7.2e-16 |
| 20 | 1 | 1 | 9605 | 43462 | 0.22 | 983 | 1.40 | 7.4e-16 |
| 20 | 1 | 2 | 12220 | 53204 | 0.23 | 983 | 1.40 | 6.3e-16 |
| 20 | 1 + 0.1i | 2 | 12212 | 53170 | 0.23 | 983 | 1.36 | 7.0e-16 |

The cost per pair does not depend on k (the tables' sizes and the operations are p's),
which the two k̂ rows show. Over the measured range Rotation's time grows about 60-fold
(195 ns at p = 3 to 12.2 µs at p = 20 with two planes), Dense's about 700-fold (72 ns to
53 µs), close to the 2 (p + 1)⁴ values of its table: its two-plane rows read the table once (one pass, the four
real products combined per k) and cost 1.1–1.3 times the one-plane rows.

## 2. End to end

`helmholtz_accuracy` (one rank, one thread, the P2P kernel NEON, the default degree rule,
the default dense budget of 1 GiB; §3): each case with `--m2l rotation,dense`, then again
with `--m2l dense,rotation`; the evaluation's wall time, the mean of the two launches (they
differ by at most 0.12 s, 1 %). Dense ran on the levels named, from the finest up within the
budget; the levels above them ran Rotation.

| Case | ε | p by level (2, 3, …) | Dense levels (MB) | build s, Rotation / Dense | evaluate s, Rotation | evaluate s, Dense | Dense / Rotation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| W1, kD = 30 (7,162 points) | 10⁻³ | 12, 8 | 2, 3 (178) | 0.00 / 0.18 | 0.07 | 0.11 | 1.5 |
| W1, kD = 30 | 10⁻⁶ | 16, 15 | 2, 3 (754) | 0.00 / 0.93 | 0.24 | 0.62 | 2.6 |
| W1, kD = 100 (79,578 points) | 10⁻³ | 30, 18, 11, 7 | 3, 4, 5 (784) | 0.05 / 1.04 | 2.45 | 4.72 | 1.9 |
| W1, kD = 100 | 10⁻⁶ | 35, 22, 15, 15 | 4, 5 (663) | 0.06 / 0.84 | 6.14 | 12.49 | 2.0 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻³ | 7, 7, 7 | 2, 3, 4 (62) | 0.04 / 0.09 | 0.94 | 0.92 | 0.97 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻⁶ | 15, 15, 15 | 2, 3, 4 (994) | 0.04 / 1.23 | 4.24 | 10.75 | 2.5 |
| W3, kD = 30 (108,849 points) | 10⁻³ | 12, 8, 7 | 2, 3, 4 (198) | 0.04 / 0.24 | 1.01 | 1.04 | 1.03 |

- The errors against the sampled direct sum are Rotation's to the printed three digits in
  every case (for instance 4.15e-5 and 1.36e-4 on W1 at kD = 100, ε = 10⁻³; 6.08e-8 and
  2.69e-7 on the cube at 10⁻⁶). The checksums differ between the two strategies (Dense is
  Rotation to rounding, not bit for bit) and are each the same in both launches.
- The ratios follow §1: at p = 7 on every level (the cube at 10⁻³), where §1 has the two
  level per pair, the evaluations take the same time (0.97×); W3, with p = 7 on its finest
  Dense level and 8 and 12 above it, 1.03×. The cases with Dense levels of higher p (W1 at
  10⁻³: p = 8–18; at 10⁻⁶ and the cube at 10⁻⁶: p = 15–16) cost 1.5–2.6× Rotation's
  evaluation, less than §1's per-pair factors at those p (Dense at 0.26–0.72× Rotation's
  speed) because the M2L is only part of the evaluation.
- The build adds the tables: up to 1.23 s for 994 MB at p = 15.

## 3. How it was run

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    target/release/examples/helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20
HWLOC_COMPONENTS=-darwin OPENBLAS_NUM_THREADS=1 … target/release/examples/helmholtz_accuracy \
    --workload w1 --kd 30,100 --eps 1e-3,1e-6 --threads 1 --m2l rotation,dense
```

(`helmholtz_m2l` twice; `helmholtz_accuracy` also with `--m2l dense,rotation`, and for the
W4 cube at N = 10⁵, kD = 10 and W3 at kD = 30.) Both examples were built at 5fedcd8
immediately before and run outside the sandbox, one launch at a time, behind a quiet gate
modelled on `tools/scaling/run.sh`'s for macOS but stricter before a launch: the other
processes' CPU (`ps`) at most 150 % before each launch (they were at 31–99 %, Microsoft
Defender's daemons the largest), sampled every 5 s during it, and a launch during which
they rose above 200 % void and repeated. No launch was void: the other processes used at
most 102 and 109 % during the two `helmholtz_m2l` launches and 36–93 % during the
`helmholtz_accuracy` ones; the one-minute load average (which counts the run itself) was
at most 4.2. macOS binds no thread, so a launch may run on a Super or a Performance core.

An earlier version of this report (79b3fdd) ran without a gate, and its §2 on a stale
`helmholtz_accuracy` binary built before the one-pass chain (df11179): it timed Dense's
former two-pass chain, 1.4–4.6× Rotation's evaluation. Rotation's times and checksums were
the same as here, and so were §1's per-pair times (within 2 %; that example had been
rebuilt).
