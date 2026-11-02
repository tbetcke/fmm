# Helmholtz CH3.2a: the Dense M2L

Helmholtz task CH3.2a ([docs/helmholtz/tasks/CH3.2a.md](../../docs/helmholtz/tasks/CH3.2a.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§5.3 (Dense) and §10 CH3.2. Measured on 2026-11-02 on the M6 (Mac mini, Apple M6, 24 GB;
[tools/m6/machine.md](../../tools/m6/machine.md)) from branch `tbetcke/h-ch3-2a`, release
build, rustc 1.99.0, BLAS threads 1, one thread, f64. The times are indicative only (the
M6's launches vary by a few per cent at one thread, and by up to 25 % at six, CH3.4c);
nothing here is timed for a gate, and nothing decides a default.

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
  offsets per pass: 2.5–2.8× faster at p = 3, 1.3–1.7× at p = 4–5, level at p = 6–7, then
  slower, 0.42× Rotation's speed at p = 12 and 0.22× at p = 20 (§1). Dense reads a
  2(p + 1)⁴-value table per pair (160 KB at p = 9) and is bound by memory; Rotation's
  O(p³) work runs from the cache. The calibrated degrees (CH3.3) are at least 7 on the
  low-frequency levels at ε ≤ 10⁻³ (15 at 10⁻⁶), so Dense applied pair by pair pays only
  above ε = 10⁻³.
- **End to end, Dense is slower in every case measured** (§2): 1.4–1.5× the evaluation
  time of Rotation on the W4 cube at ε = 10⁻³ (p = 7) and on W3 at kD = 30, 2.3–3.3× on W1
  at 10⁻³, 3.4–4.6× at 10⁻⁶, with the same errors to the printed digits. So Dense stays
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
always two. ns per pair:

| p | k̂ | planes | Rotation ns/pair | Dense ns/pair | Rotation / Dense | Dense MB | Dense build s | Dense against Rotation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 1 | 1 | 161 | 64 | 2.52 | 1 | 0.00 | 4.8e-16 |
| 3 | 1 | 2 | 197 | 72 | 2.73 | 1 | 0.00 | 6.5e-16 |
| 3 | 1 + 0.1i | 2 | 197 | 72 | 2.76 | 1 | 0.00 | 6.1e-16 |
| 4 | 1 | 1 | 256 | 147 | 1.73 | 3 | 0.00 | 4.3e-16 |
| 4 | 1 | 2 | 311 | 191 | 1.63 | 3 | 0.00 | 5.1e-16 |
| 4 | 1 + 0.1i | 2 | 312 | 191 | 1.63 | 3 | 0.00 | 5.1e-16 |
| 5 | 1 | 1 | 386 | 298 | 1.30 | 7 | 0.00 | 5.6e-16 |
| 5 | 1 | 2 | 473 | 318 | 1.49 | 7 | 0.00 | 5.0e-16 |
| 5 | 1 + 0.1i | 2 | 473 | 313 | 1.51 | 7 | 0.00 | 6.0e-16 |
| 6 | 1 | 1 | 561 | 546 | 1.03 | 12 | 0.01 | 5.8e-16 |
| 6 | 1 | 2 | 690 | 701 | 0.98 | 12 | 0.01 | 5.0e-16 |
| 6 | 1 + 0.1i | 2 | 688 | 701 | 0.98 | 12 | 0.01 | 5.1e-16 |
| 7 | 1 | 1 | 785 | 875 | 0.90 | 21 | 0.02 | 5.7e-16 |
| 7 | 1 | 2 | 966 | 968 | 1.00 | 21 | 0.02 | 5.4e-16 |
| 7 | 1 + 0.1i | 2 | 967 | 964 | 1.00 | 21 | 0.02 | 5.3e-16 |
| 8 | 1 | 1 | 1023 | 1413 | 0.72 | 33 | 0.03 | 5.8e-16 |
| 8 | 1 | 2 | 1285 | 1792 | 0.72 | 33 | 0.03 | 5.6e-16 |
| 8 | 1 + 0.1i | 2 | 1288 | 1829 | 0.70 | 33 | 0.03 | 5.6e-16 |
| 9 | 1 | 1 | 1332 | 2165 | 0.62 | 51 | 0.05 | 9.2e-16 |
| 9 | 1 | 2 | 1661 | 2334 | 0.71 | 51 | 0.05 | 5.3e-16 |
| 9 | 1 + 0.1i | 2 | 1675 | 2328 | 0.72 | 51 | 0.04 | 5.8e-16 |
| 10 | 1 | 1 | 1688 | 3220 | 0.52 | 74 | 0.07 | 6.4e-16 |
| 10 | 1 | 2 | 2108 | 3977 | 0.53 | 74 | 0.07 | 7.1e-16 |
| 10 | 1 + 0.1i | 2 | 2110 | 4081 | 0.52 | 74 | 0.07 | 6.3e-16 |
| 12 | 1 | 1 | 2638 | 6341 | 0.42 | 144 | 0.15 | 1.1e-15 |
| 12 | 1 | 2 | 3311 | 7733 | 0.43 | 144 | 0.15 | 5.7e-16 |
| 12 | 1 + 0.1i | 2 | 3312 | 7727 | 0.43 | 144 | 0.14 | 6.5e-16 |
| 15 | 1 | 1 | 4703 | 14470 | 0.33 | 331 | 0.40 | 8.5e-16 |
| 15 | 1 | 2 | 5886 | 16085 | 0.37 | 331 | 0.40 | 6.2e-16 |
| 15 | 1 + 0.1i | 2 | 5900 | 16132 | 0.37 | 331 | 0.39 | 6.8e-16 |
| 18 | 1 | 1 | 7392 | 29513 | 0.25 | 659 | 0.88 | 7.7e-16 |
| 18 | 1 | 2 | 9335 | 35895 | 0.26 | 659 | 0.88 | 6.6e-16 |
| 18 | 1 + 0.1i | 2 | 9328 | 36016 | 0.26 | 659 | 0.85 | 7.2e-16 |
| 20 | 1 | 1 | 9623 | 44419 | 0.22 | 983 | 1.40 | 7.4e-16 |
| 20 | 1 | 2 | 12265 | 53241 | 0.23 | 983 | 1.40 | 6.3e-16 |
| 20 | 1 + 0.1i | 2 | 12291 | 53573 | 0.23 | 983 | 1.40 | 7.0e-16 |

The cost per pair does not depend on k (the tables' sizes and the operations are p's),
which the two k̂ rows show. Over the measured range Rotation's time grows about 60-fold
(197 ns at p = 3 to 12.3 µs at p = 20 with two planes), Dense's about 700-fold (72 ns to
53 µs), close to the 2 (p + 1)⁴ values of its table: its two-plane rows read the table once (one pass, the four
real products combined per k) and cost 1.1–1.3 times the one-plane rows.

## 2. End to end

`helmholtz_accuracy` (one rank, one thread, the P2P kernel NEON, the default degree rule,
the default dense budget of 1 GiB; §3): each case with `--m2l rotation,dense`, then again
with `--m2l dense,rotation`; the evaluation's wall time, the mean of the two launches (they
differ by at most 0.09 s). Dense ran on the levels named, from the finest up within the
budget; the levels above them ran Rotation.

| Case | ε | p by level (2, 3, …) | Dense levels (MB) | build s, Rotation / Dense | evaluate s, Rotation | evaluate s, Dense | Dense / Rotation |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| W1, kD = 30 (7,162 points) | 10⁻³ | 12, 8 | 2, 3 (178) | 0.00 / 0.18 | 0.07 | 0.16 | 2.3 |
| W1, kD = 30 | 10⁻⁶ | 16, 15 | 2, 3 (754) | 0.00 / 0.92 | 0.24 | 1.10 | 4.6 |
| W1, kD = 100 (79,578 points) | 10⁻³ | 30, 18, 11, 7 | 3, 4, 5 (784) | 0.06 / 1.05 | 2.45 | 8.04 | 3.3 |
| W1, kD = 100 | 10⁻⁶ | 35, 22, 15, 15 | 4, 5 (663) | 0.06 / 0.83 | 6.17 | 20.75 | 3.4 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻³ | 7, 7, 7 | 2, 3, 4 (62) | 0.04 / 0.09 | 0.93 | 1.30 | 1.4 |
| W4 cube, N = 10⁵, kD = 10 | 10⁻⁶ | 15, 15, 15 | 2, 3, 4 (994) | 0.04 / 1.24 | 4.27 | 19.10 | 4.5 |
| W3, kD = 30 (108,849 points) | 10⁻³ | 12, 8, 7 | 2, 3, 4 (198) | 0.04 / 0.24 | 1.02 | 1.49 | 1.5 |

- The errors against the sampled direct sum are Rotation's to the printed three digits in
  every case (for instance 4.15e-5 and 1.36e-4 on W1 at kD = 100, ε = 10⁻³; 6.08e-8 and
  2.69e-7 on the cube at 10⁻⁶). The checksums differ between the two strategies (Dense is
  Rotation to rounding, not bit for bit) and are each the same in both launches.
- Dense is slower in every case: 1.4–1.5× on the cube and W3 at 10⁻³ (Dense levels of
  p = 7 to 12), 2.3–3.3× on W1 at 10⁻³ and 3.4–4.6× at 10⁻⁶. Even at p = 7 on every level (the cube at 10⁻³), where §1 has the two at
  parity per pair, the evaluation takes 1.4× as long: in the FMM a target reads up to 189
  of the level's 316 tables (21 MB at p = 7), each once, so the tables stream from memory
  as in §1, and the per-target order of the offsets is less regular than §1's pass.
- The build adds the tables: up to 1.24 s for 994 MB at p = 15.

## 3. How it was run

```text
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    target/release/examples/helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20
HWLOC_COMPONENTS=-darwin OPENBLAS_NUM_THREADS=1 … target/release/examples/helmholtz_accuracy \
    --workload w1 --kd 30,100 --eps 1e-3,1e-6 --threads 1 --m2l rotation,dense
```

(and the same with `--m2l dense,rotation`, and for the W4 cube at N = 10⁵, kD = 10 and W3
at kD = 30). The `helmholtz_m2l` run's machine line reads "unknown": it ran inside Claude
Code's sandbox, which denies `sysctl`; it is the M6.
