# Helmholtz CH7.5c: the device's plane-wave M2L, measured

Helmholtz task CH7.5c ([docs/helmholtz/tasks/CH7.5c.md](../../docs/helmholtz/tasks/CH7.5c.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH7.5a–c ("adopted where it wins at p ≥ 12 (T13's gate), else dropped"), §2
(constraint 1). It measures CH7.5b's merge and shift on the device
(`FmmBuilder::device_plane_wave(DevicePlaneWave::Kernels)`) against the device's dense M2L
(Phase 6 T7's, `M2lStrategy::Dense` under the static rule), with the kernels' layouts and
the chunk budget. Measured on:

- **locust** (NVIDIA GH200: the H100, 96 GB HBM3, CUDA 12.6; 72 Neoverse-V2 cores;
  `tools/gh200/`): on 2026-11-05, 20:27–22:49 UTC (§2–§4) and 22:49–23:36 (§5), source
  8f57fd2 and 798ce48 (the same plane-wave code; 798ce48 adds the 512- and 1,024-unit
  rows), load average about 1 (our process) and no other GPU process at the start or end
  of any launch (each file's first and last line);
- **the M6** (Apple M6, 24 GB, Metal; `tools/m6/machine.md`): on 2026-11-05, 21:12–21:37
  UTC, source 798ce48, outside the sandbox, with this task's own verify tests paused
  (SIGSTOP) for the run.

Release builds, rustc 1.99.0, CubeCL 0.11.0, every BLAS thread variable at 1, one rank. The
tool is `nd-fmm-validate`'s `layout_sweep --part planewave` (this task; its module docs and
§7). Nothing here is asserted by a test; the accuracy gate of the device's `PlaneWave` is
CH7.5b's (`tests/device_fmm.rs`: within 1.011× the device `Dense`'s L2 errors), and the
bits do not depend on the layout or the budget (CH7.5a's kernel tests, CH7.5b's and this
task's `device_common::check_plane_waves`).

## Summary

1. **Merge and shift wins everywhere it was measured**, not only at p ≥ 12. On the H100
   the M2L stage takes 0.30–0.38× the dense M2L's time at p = 8 (f32 and f64), 0.19–0.20×
   at f64 p = 12 and 0.32–0.35× at f64 p = 18 with the default layout (0.19–0.21× with
   cubes of 256 units, §2), on the cube and the Plummer sphere at N = 10⁶ and 10⁷. The
   evaluation takes 0.79–0.90× at p = 8, 0.52–0.68× at p = 12 and 0.47–0.61× at p = 18
   (0.36–0.45× with 256 units). On the M6's Metal (f32 p = 8) M2L takes 0.47–0.60× and the
   evaluation 0.65–0.69×.
2. **T13's gate is met** (a measured device kernel showing the gain at p ≥ 12): f64
   p = 12 and 18 on the H100, both problems, both sizes. The accuracy side of T13's
   requirement 1 was met in CH7.5b (0.941–1.011× the device `Dense`'s errors).
3. **Layouts.** `GPU_UNITS` = 128 is the best or tied on Metal (f32 p = 8) and on CUDA to
   p = 12; on CUDA from p = 14 cubes of 256 units are faster (1.4–1.8× at p = 18 and 20),
   and 512 units at most 3.6 % faster than 256 (§5).
4. **Chunk budget.** On CUDA 4 GB is 1–9 % faster in M2L than the default 2 GB (the
   translation budget) at p ≥ 12; 128 MB is 2.0–2.8× slower. On Metal the default 128 MB
   makes up to 88 chunks per level call, and 1 GB is 14–17 % faster in M2L (§3).
5. **Proposed** (question 1 of the PR): adopt. `DevicePlaneWave::Kernels` becomes the
   default of `PlaneWave` on every device backend (`Dense` stays the named opt-out), on
   CUDA cubes of 256 units from p = 13 (128 below and on Metal; §5), and on Metal a
   plane-wave budget of 1 GB. The device
   `Auto` is unchanged (it never resolves to `PlaneWave`); question 2 asks whether it
   should, as a separate task.

## 1. Method

Per problem, precision and p, `layout_sweep --part planewave` builds the FMM once per
variant (N points, sources equal to targets, `max_level` 16, 64 points per leaf, gradients
on, one charge vector, every kind on the device, the static rule otherwise):

- **Dense (static rule)**: `M2lStrategy::Dense`, the device's dense M2L of Phase 6 T7 (on
  CUDA the hand-written GEMM in its CUDA layout and gather; on Metal the shared-memory
  GEMM with the fused gather, T7's defaults). The reference row of every table.
- **PlaneWave**: `M2lStrategy::PlaneWave` with `DevicePlaneWave::Kernels`, at the default
  layout and budget, then each cube layout (`device_plane_wave_layout`), then each other
  chunk budget (`device_plane_wave_budget`).

For each variant: the M2L kind time (`KindTiming::Device`, CUDA events per level call, a
breakdown on CUDA; `Synchronous` on Metal, a sync after every level call), the median of
10 evaluations after two warm-ups; the evaluation's wall time (`Fmm::evaluate`, charges up
and output down included) from a second build without kind timings, the median of 10 after
one warm-up. Then a third pair of builds times M2L per level for Dense and the default
PlaneWave, with each plane-wave level call's chunks and items (`DeviceReport::plane_waves`).
The problems: the C3.3 cube and Plummer sphere (`fmm_accuracy::Config::c33`) at N = 10⁶ and
10⁷. Ratios are PlaneWave over Dense.

Both builds take the library's near rule, whose W and X crossovers come from each
strategy's calibrated cost model (`CostModel::calibrated`), so the two may convert
different W and X pairs to P2P: the evaluation ratio is the end-to-end effect of naming
the strategy, not of M2L alone (M2L's V lists are the same). No error column: the accuracy
is CH7.5b's gate.

The evaluation's wall time includes the host's part of a device evaluation (the charge
load and the output pass on the host; 317–500 ms at N = 10⁷ on locust, T7 §4), the
same for both strategies, so its ratio is closer to 1 than M2L's.

## 2. CUDA (the H100)

Default layout (cube of 128 units) and budget (2 GB); in brackets the cube of 256 units.
Times in ms.

| problem | N | precision, p | Dense M2L | PlaneWave M2L | ratio | Dense evaluation | PlaneWave evaluation | ratio |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10⁶ | f64, 8 | 14.7 | 4.57 | 0.31 | 48.7 | 42.3 | 0.87 |
| cube | 10⁶ | f64, 12 | 50.4 | 9.90 | 0.20 | 99.5 | 55.1 | 0.55 |
| cube | 10⁶ | f64, 18 | 197.3 | 63.5 [38.8] | 0.32 [0.20] | 265.0 | 127.0 [102.3] | 0.48 [0.39] |
| cube | 10⁶ | f32, 8 | 9.19 | 3.30 | 0.36 | 36.2 | 28.6 | 0.79 |
| Plummer | 10⁶ | f64, 8 | 23.0 | 7.75 | 0.34 | 77.9 | 61.8 | 0.79 |
| Plummer | 10⁶ | f64, 12 | 78.6 | 16.0 | 0.20 | 138.9 | 72.8 | 0.52 |
| Plummer | 10⁶ | f64, 18 | 308.2 | 104.5 [61.3] | 0.34 [0.20] | 389.9 | 186.6 [139.7] | 0.48 [0.36] |
| Plummer | 10⁶ | f32, 8 | 14.5 | 5.46 | 0.38 | 44.5 | 36.2 | 0.81 |
| cube | 10⁷ | f64, 8 | 132.1 | 39.2 | 0.30 | 789.6 | 672.9 | 0.85 |
| cube | 10⁷ | f64, 12 | 448.1 | 83.6 | 0.19 | 1146.4 | 784.7 | 0.68 |
| cube | 10⁷ | f64, 18 | 1755.6 | 620.7 [340.8] | 0.35 [0.19] | 2596.1 | 1589.9 [1180.0] | 0.61 [0.45] |
| cube | 10⁷ | f32, 8 | 80.4 | 26.7 | 0.33 | 503.6 | 455.0 | 0.90 |
| Plummer | 10⁷ | f64, 8 | 258.2 | 81.1 | 0.31 | 932.0 | 751.5 | 0.81 |
| Plummer | 10⁷ | f64, 12 | 889.0 | 173.9 | 0.20 | 1649.3 | 935.6 | 0.57 |
| Plummer | 10⁷ | f64, 18 | 3471.7 | 1139.4 [716.5] | 0.33 [0.21] | 4438.2 | 2101.3 [1675.9] | 0.47 [0.38] |
| Plummer | 10⁷ | f32, 8 | 157.3 | 55.2 | 0.35 | 593.1 | 501.8 | 0.85 |

Per level (M2L ms, Dense / PlaneWave default, f64 p = 12, the Plummer sphere at 10⁷): the
ratio is 0.19–0.21 on every level from 5 down, 0.24 on level 4, 0.38 on level 3; on level
2 (64 targets, 3,096 pairs) merge and shift is 1.4× slower (0.18 against 0.13 ms; at
p = 18 2.4×, 0.58 against 0.24 ms): too few items to fill the device. At p = 18 the ratio
is 0.31–0.36 below level 3 with the default 128 units. The level calls run in 1–40 chunks
at the 2 GB budget (the widest chunk's scratch 2.0–2.4 GB).

The run-to-run spread at f64 p = 18 is visible: the default row and the identical "cube
(128 units)" row of the cube at 10⁷ differ by 15 % (620.7 and 541.8 ms), against at most
0.2 % elsewhere. The adoption rests on factors of 3 to 5 in M2L, which it does not touch; the layout and
budget choices rest on 3–17 % differences and are less certain (§4, §5).

## 3. Metal (the M6, f32 p = 8)

Default layout (cube of 128 units) and budget (128 MB); in brackets 1 GB. Times in ms.

| problem | N | Dense M2L | PlaneWave M2L | ratio | Dense evaluation | PlaneWave evaluation | ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cube | 10⁶ | 63.8 | 36.0 [31.0] | 0.56 [0.49] | 87.8 | 59.5 [55.0] | 0.68 [0.63] |
| Plummer | 10⁶ | 98.4 | 58.8 [50.4] | 0.60 [0.51] | 129.9 | 89.9 [81.0] | 0.69 [0.62] |
| cube | 10⁷ | 670.5 | 314.8 [270.1] | 0.47 [0.40] | 1051.2 | 694.3 [650.3] | 0.66 [0.62] |
| Plummer | 10⁷ | 1268.9 | 677.4 [562.3] | 0.53 [0.44] | 1680.7 | 1096.7 [998.5] | 0.65 [0.59] |

Layouts: 64, 128 and 256 units within 2 % of each other, 32 units 5–6 %, 512 units 6–7 %
and 1,024 units 32–33 % slower than 128. Budgets: 512 MB, 1 GB and 2 GB within 2.3 % of
each other, 4 GB 0–6 % slower than 1 GB; the default 128 MB makes 10 chunks of level 5 of
the cube at 10⁶ and 88 of level 6 at 10⁷. The
cube at 10⁶ ran first, while the one-minute load average still fell from the paused tests
(16 at its start, 8.7 at its end); its ratios agree with the quiet runs at 10⁷.

Per level (the Plummer sphere at 10⁷): 0.50–0.59 on levels 4–10, 0.77–0.81 on levels 2
and 3; merge and shift is faster on every level on Metal.

## 4. Budgets (CUDA)

M2L ratio to Dense by budget, default layout (the smallest and the largest over the four
problems of each row):

| precision, p | 128 MB | 512 MB | 1 GB | 2 GB (default) | 4 GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| f64, 8 | 0.51–0.56 | 0.36–0.39 | 0.32–0.35 | 0.30–0.34 | 0.29–0.34 |
| f64, 12 | 0.38–0.43 | 0.22–0.25 | 0.20–0.22 | 0.19–0.20 | 0.18–0.20 |
| f64, 18 | 0.79–0.93 | 0.37–0.42 | 0.34–0.38 | 0.32–0.35 | 0.30–0.32 |
| f32, 8 | 0.51–0.57 | 0.38–0.41 | 0.35–0.38 | 0.33–0.38 | 0.32–0.38 |

The cost of a small budget is the waves of a chunk's sources that the next chunk computes
again (on level 5 of the cube at 10⁷, f64 p = 18, five chunks of the 2 GB budget hold
47,216 sources for 32,768 local items), and the launches per chunk. 4 GB gains 1–9 % over
2 GB at p ≥ 12 (against the 128-unit row where the 2 GB row is the noisy one, §2); the device's 2 GB default stays (`default_scratch_bytes`, the scratch budget of
every device translation on CUDA, at most an eighth of the device's memory).

## 5. Units by p (CUDA)

The cube at N = 10⁶, f64, the default budget (2 GB), every cube layout; M2L in ms (CUDA
events, the median of 10) and its ratio to Dense, the fastest layout in bold. p = 8 from §2.

| p | Dense | 32 | 64 | 128 | 256 | 512 | 1,024 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 14.7 | 5.47 (0.37) | 4.92 (0.33) | **4.57 (0.31)** | 4.96 (0.34) | – | – |
| 10 | 27.4 | 12.7 (0.46) | 9.07 (0.33) | **7.91 (0.29)** | 8.67 (0.32) | 9.22 (0.34) | 13.6 (0.49) |
| 12 | 50.4 | 15.9 (0.32) | 17.8 (0.35) | **9.90 (0.20)** | 10.4 (0.21) | 11.3 (0.22) | 15.0 (0.30) |
| 14 | 73.2 | 36.0 (0.49) | 29.4 (0.40) | 20.5 (0.28) | **17.5 (0.24)** | 19.8 (0.27) | 23.1 (0.32) |
| 16 | 154.9 | 44.9 (0.29) | 34.0 (0.22) | 26.1 (0.17) | 23.4 (0.15) | **23.3 (0.15)** | 27.9 (0.18) |
| 18 | 197.2 | 72.6 (0.37) | 59.0 (0.30) | 63.5 (0.32) | 38.8 (0.20) | **38.1 (0.19)** | 43.6 (0.22) |
| 20 | 262.0 | 167.8 (0.64) | 108.9 (0.42) | 77.8 (0.30) | 53.7 (0.20) | **51.8 (0.20)** | 58.0 (0.22) |

The evaluation with the default layout over Dense's: 0.77 (p = 10), 0.58 (12), 0.61 (14),
0.40 (16), and with 256 units 0.39 (18, §2's quiet run) and 0.39 (20).

**Rule** (proposed): on CUDA a cube of 128 units to p = 12 and of 256 units from p = 13.
Where 256 is not the fastest (p = 16, 18, 20: 512 units) it is within 0.4 %, 1.9 % and 3.6 %
of it; 128 units are 17 % slower at p = 14 and 45–64 % slower at p = 18 and 20. The
crossover lies between p = 12 (128 faster by 4.7 %) and p = 14 (256 faster by 15 %); p = 13
was not measured. Metal keeps 128 units (measured at f32 p = 8 only, §3; Metal runs f32,
where p > 8 "lies beyond the useful range", `FmmBuilder`'s docs).

The launch (22:49–23:02 UTC) overlapped another user's job from about 22:55 (the load
average 10.4 at 23:02 against 0.7 at the start; no other GPU process at either end): p = 18
agrees with §2's quiet run within 0.6 % in M2L (the evaluations 3–5 % higher), but p = 20's
256-, 512- and 1,024-unit rows were 2–20 % slower than a rerun on the idle machine
(23:32–23:36, load average 0.00), which the table gives
(`layouts-cube-1e6-f64-p20-rerun.md`). The rule is the same with either run.

## 6. Proposal

For sign-off (question 1 of the PR; the roadmap's *Sign-off: adoption on the device*):

- **Adopt.** `DevicePlaneWave::Kernels` becomes the default of `M2lStrategy::PlaneWave` on
  every device backend; `DevicePlaneWave::Dense` stays as the named opt-out ("PlaneWave, run
  as dense on the device", `Dense`'s bits). It wins at every measured p on CUDA and on
  Metal, at p ≥ 12 by a factor of 3–5 in M2L.
- **Layout.** `DevicePlaneWaveLayout::Auto` resolves by backend and p: on CUDA cubes of 128
  units to p = 12 and of 256 from p = 13 (§5); on Metal 128 units (§3); the CPU layout on
  the CPU runtime.
- **Budget.** On Metal the plane-wave budget's default becomes 1 GB (at most an eighth of
  the device's memory); CUDA and the CPU runtime keep the translation budget.
- **The kernels stay** (CH7.5d builds on them).
- **The device `Auto`** stays as it is (`Dense`, `Rotation` by the static rule): it never
  resolves to `PlaneWave`, so no unnamed Laplace evaluation changes. Since `PlaneWave` on a
  device is now 2–5× faster in M2L than the static rule's `Dense` at p ≥ 12, question 2
  asks whether to add a task that makes the device `Auto` resolve to it (as the host's has
  since Phase 6 T11), with its own measurements and sign-off.

## 7. Reproducing

```sh
# locust (outside the sandbox), after tools/gh200/sync.sh:
tools/gh200/remote.sh 'cargo build --release -p nd-fmm-validate --features cuda --example layout_sweep'
tools/gh200/remote.sh 'target/release/examples/layout_sweep --device cuda --precision f64 \
    --degrees 8,12,18 --n 1e7 --part planewave --dist plummer --repeats 10'
# the M6, Metal (build in the sandbox, run outside it):
cargo build --release -p nd-fmm-validate --features metal --example layout_sweep
target/release/examples/layout_sweep --device metal --precision f32 --degrees 8 --n 1e6 \
    --part planewave --dist cube --repeats 10
```

The launches of §2 and §5 (`--degrees 10,12,14,16,18,20 --dist cube --n 1e6`) were wrapped in a script that records the load and the GPU's
processes before and after each (`/data/ucahtbe/runs/ch7-5c/sweep.sh` on locust).
