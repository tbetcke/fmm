# CH7.5c Measured, and adopted or dropped

Started 2026-11-05 on `tbetcke/h-ch7-5c`. Roadmap: "CH7.5c Measured, and adopted or
dropped" (priority list item 1, CH7.5a's split: "a/b/c, Helmholtz → d", the user,
2026-11-05). Proposal: §10 CH7.5a–c; §2 constraints 1 and 2; §8 (the device path).
Continues docs/helmholtz/tasks/CH7.5b.md and docs/phase6/T13-plane-wave-m2l.md,
"Afterwards" (a device plane-wave M2L "gated on … a measured device kernel showing the gain
at p ≥ 12").

## Goal
Measure the device's `PlaneWave` on merge and shift (CH7.5b's `DevicePlaneWave::Kernels`)
against the device's dense M2L, with its layouts and chunk budget, and adopt it as the
device's `PlaneWave` where it wins at p ≥ 12, else keep `PlaneWave` dense on the device and
drop the kernels.

## Acceptance (proposal §10, CH7.5a–c, verbatim)
T13's requirement 1 on the device (within 1.1× Dense's errors against the direct sum), bit
for bit run to run and on P ranks against one rank; adopted where it wins at p ≥ 12 (T13's
gate), else dropped.

The roadmap's line for CH7.5c: the device M2L stage and the evaluation under CH7.5b's
setting against T7's dense M2L at f64 p = 8, 12, 18 on locust's H100 (f32 p = 8 on the M6's
Metal), with the kernels' layouts measured (`GPU_UNITS` = 128 a first choice; the merge's
irregular lists the spike's open question) and the chunk budget
(`device_plane_wave_budget`, default the translation scratch budget); Laplace's device
gates and benchmarks within noise (D9). *Done when:* the report gives both; adopted as the
device's `PlaneWave` where it wins at p ≥ 12 (the setting becomes the default of
`PlaneWave` on the device), else `PlaneWave` stays `Dense` on the device and the kernels
are dropped; the device `Auto` unchanged (a change to it is a sign-off). *Sign-off:*
adoption on the device.

Requirement 1 and the bits (run to run, P ranks) were met and tested in CH7.5b
(`tests/device_fmm.rs`'s gate, `device_common::check_plane_waves`); this task keeps those
tests passing under whatever default it ends with.

## Read first
- docs/helmholtz/tasks/CH7.5a.md, CH7.5b.md; fmm-exec/CLAUDE.md (the device path, "CUDA
  (Phase 4S)"), fmm-kernels/CLAUDE.md ("Plane-wave M2L"), fmm-validate/CLAUDE.md.
- `fmm-exec/src/device.rs` ("Plane-wave M2L on the device", `PlaneWaveLayout::default_for`
  at the `Translations`), `fmm-exec/src/fmm.rs` (`DevicePlaneWave`, the builder table).
- `fmm-kernels/src/plane_wave.rs` (`GPU_UNITS`, `PlaneWaveLayout::check`).
- `fmm-validate/examples/layout_sweep.rs` (part **kinds**: one build per variant,
  `KindTiming::Device` on CUDA, `Synchronous` on Metal, the wall time from a build without
  kind timings); fmm-validate/results/phase6-device.md (T7's dense M2L on the H100 and
  Metal); fmm-validate/results/phase6-planewave.md (T13's host numbers).

## Changes
1. `nd-fmm-exec` (feature `gpu`), additive: `fmm::DevicePlaneWaveLayout` (`Auto`, the
   default: `PlaneWaveLayout::default_for`, as now; `Cube { units }`) and
   `FmmBuilder::device_plane_wave_layout`, as `device_leaf_layout` is for the leaf
   operators; checked at build by `PlaneWaveLayout::check` (a refused layout is a
   `SettingsError`/device error as the other layouts are); ignored without
   `DevicePlaneWave::Kernels`. Reported already in `DeviceReport::plane_wave_layout`.
2. `nd-fmm-validate`, `layout_sweep`: a part **planewave** and `--dist cube|plummer`
   (default the C3.3 cube, as now): per (precision, p) the FMM built with
   `M2lStrategy::Dense` (the device's dense M2L under the static rule: T7's) and with
   `PlaneWave` + `Kernels` at each layout (`Cube` with 32, 64, 128, 256 units; those the
   device refuses become a "refused" row) and each chunk budget (the default, then 128 MB,
   512 MB, 1 GB, 2 GB, 4 GB, those below the device's memory), the M2L kind time per
   evaluation (CUDA `Device`, Metal `Synchronous`; median of `--repeats`), the evaluation's
   wall time from a build without kind timings (median of five after a warm-up), the M2L
   level calls' chunks and launches from `DeviceReport::plane_waves`, the scratch bytes,
   and the φ and ∇φ errors of both against the direct sum at 1,000 targets (a sanity
   column; the gate is CH7.5b's test). Nothing asserted.
3. The report `fmm-validate/results/helmholtz-ch7-5c.md`: per machine, problem and p the
   M2L stage and the evaluation of `Dense` and of the best plane-wave layout and budget,
   the ratios, the layout and budget sweeps; the decision.
4. On the sign-off (question 1 of the PR), one of:
   - **adopt**: `DevicePlaneWave`'s default becomes `Kernels` (where it wins: for every p,
     or from a threshold of p by backend if it wins only at high p), `GPU_UNITS` and the
     budget's default set by the measurements; `DevicePlaneWave::Dense` kept as the named
     opt-out; docs, the builder table, fmm-exec/CLAUDE.md, the tests that assert the
     default (`check_plane_wave`'s "without the setting run as dense") turned round;
   - **drop**: `DevicePlaneWave::Kernels`, the setting, `nd_fmm_kernels::plane_wave` and
     their tests removed; CH7.5d's line rewritten to rebuild on a later measurement (a
     sign-off question itself, since CH7.5d needs the kernels).
   The device `Auto` is not changed; if the measurements argue for it, that is a question
   in the PR and a roadmap line, not code here.
No change to the host path, `nd-fmm-tables`, `nd-fmm-plan`, a convention or a dependency.

## Tests that prove it
- The new builder setting: `device_common::check_plane_waves` builds the default and each
  layout the device accepts (`Cube` 32 and 256 units on the GPUs; on the CPU runtime a
  `Cube` layout too) bit for bit the default layout's output (each item's chain does not
  depend on the units: CH7.5a's kernel tests compare every layout bit for bit); a refused
  layout (0 units; more units than the device allows) is an error on every rank.
- `layout_sweep --part planewave` on the CPU runtime in its smoke form (N = 10⁴, p = 3) by
  hand, since the example has no test.
- Under the adopted default: `check_plane_waves` without the setting runs merge and shift;
  `DevicePlaneWave::Dense` gives `Dense`'s bits; the gate of `tests/device_fmm.rs` and
  every device executable on the CPU runtime, Metal and CUDA pass.

## Checks
`/verify`; the device path's own (fmm-exec/CLAUDE.md, "Before finishing"): Clippy with
`cpu,metal`, `cargo check --features cuda`, `cargo test -p nd-fmm-exec --features cpu
--release` (and at 2 and 4 ranks), `cargo doc --features cpu`; Clippy of `nd-fmm-validate`
with `cpu,metal` and `cuda` checked.

## Machine runs
- locust, H100 (idle, checked before each run): `layout_sweep --part planewave --device cuda`
  on the cube and the Plummer sphere, N = 10⁶ and 10⁷, f64 p = 8, 12, 18 and f32 p = 8.
  Decides the adoption (p ≥ 12), `GPU_UNITS` and the budget on CUDA.
- The M6, Metal f32 p = 8 (outside the sandbox; a timing run, so asked first): the same at
  N = 10⁶ (10⁷ if it fits the 24 GB). Decides Metal's layout and whether Metal adopts too.
- Correctness after the default changes: `tests/device_cuda.rs` and the gate of
  `tests/device_fmm.rs` on locust; `tests/device_metal.rs` and the gate on the M6.
- D9: only `PlaneWave` on the device changes; every other device strategy and the host
  keep their code and bits (the tests above). D9 timing runs: `nd-fmm-bench --backend cuda`
  at the defaults (f32 and f64, p = 8, N = 10⁶), ABBA five runs per side against `main`, on
  locust; Metal the same at f32 p = 8 on the M6 if you agree to the timing runs there.

## Constraints and risks
- Constraint 1: the adoption changes a Laplace default (the device's `PlaneWave`, named
  only; `Auto` never resolves to it on a device). Agreed for this plan by the user before
  any code (below, "Open questions").
- Constraint 2: rank invariance of merge and shift was shown in CH7.5b; the layout does
  not change the bits.
- Risk: the H100 is shared; the sweep is gated on `nvidia-smi` and `uptime`, repeated where
  another process appears.
- Risk: the plane-wave M2L wins on the M2L stage but the evaluation hardly moves at p = 8
  (M2L 10 of 50 ms in T10's model); the gate is p ≥ 12, where M2L is most of the
  evaluation.

## As built
As planned, except: `layout_sweep --part planewave` has no error column (the accuracy is
CH7.5b's gate, which the layout and the budget cannot change: the same bits); the cube
layouts measured go to 1,024 units and, after the first CUDA results, by p at f64 p = 10–20
(report §5); `check_plane_waves` refuses both a cube of no unit and one above the device's
units. The review (a subagent) found nothing blocking; its points were taken (the report's
budget range and claims, the near rule's effect on the evaluation column, the ratio
column's header, the over-cap refusal, docs).

The adoption (after the sign-off, 2026-11-06), as question 1 stated: the default of
`DevicePlaneWave` flipped to `Kernels`; `DevicePlaneWaveLayout::resolve` takes p, with
`device::CUDA_PLANE_WAVE_WIDE_UNITS` (256, at most the device's units per cube) on CUDA
from `CUDA_PLANE_WAVE_WIDE_FROM` (13); `device::default_plane_wave_budget` gives Metal
`METAL_PLANE_WAVE_BUDGET` (1 GB, at most an eighth of the memory the device reports and
never below 128 MB, as CUDA's translation budget is bounded) and the translation budget
elsewhere. All of it lives in `nd-fmm-exec`; `nd_fmm_kernels::plane_wave::GPU_UNITS` and
`PlaneWaveLayout::default_for` are unchanged. The gate's printed evaluation times at
N = 10⁵ put merge and shift behind `Dense` on the cube (CUDA and Metal), ahead on the
Plummer sphere: left to CH7.5e, which measures the device `Auto` by N as well.

## Pending runs
None. Ran after the adoption (2026-11-06): on the M6's Metal `tests/device_metal.rs` and the
gate of `tests/device_fmm.rs` (φ L2 1.011× and 0.999× `Dense`'s); on locust
`tests/device_cuda.rs` (the 32- and 256-unit cubes of `check_plane_waves` on the GPU), the
gate (every L2 ratio ≤ 1.0114) and Clippy with `cpu,cuda`; D9 (below). All passed.

## D9
locust, 2026-11-06 06:18–06:24 UTC, the H100 idle before every run;
`nd-fmm-bench --backend cuda --precision f32,f64 --degree 8 --n 1000000` (`Auto`, 10 timed
evaluations after 2 warm-ups, the BLAS thread variables at 1); A = `main` b69e031, B = the
branch (959df26's tree); order A B B A A B B A A B; the spread max − min of `main`'s five
run medians.

| Row | `main` median of medians | branch | difference | `main`'s spread | D9 |
| --- | ---: | ---: | ---: | ---: | --- |
| f32 p = 8, Auto → Dense | 30.80 ms | 30.57 ms | −0.24 ms (−0.8 %) | 0.27 ms | pass |
| f64 p = 8, Auto → Dense | 58.28 ms | 54.80 ms | −3.48 ms (−6.0 %) | 4.16 ms | pass |

The errors (φ 1.36e-5, ∇φ 4.59e-5) are the same in every run, and every run's repeats are
bit-identical. In f64 `main`'s runs 4, 5, 8 and 9 took about 58 ms against 54.5 ms in its
run 1 and 54.7–55.3 ms in the branch's; the code the device `Auto` runs is the same on both
sides, so this is not the change's (B/A − 1 of adjacent pairs: +0.6, −6.5, −6.0, −5.2,
−5.9 %). Build times 2.24 / 2.24 s (f32 and f64 together, `main` / branch medians of the
f32 row). Reports on locust in /data/ucahtbe/runs/ch7-5c-d9.

## Open questions
Answered before any code (2026-11-05, the user): the plan "Go ahead (Recommended)": build
the layout setting and the planewave part, run on locust, the adoption question 1 of the
PR; the M6 timings "Ask again first": prepare everything and run locust, then ask before
any timing on the M6.

Answered in the PR (2026-11-06, the user: "sign off all three questions as recommended"):
1. adopt: `DevicePlaneWave::Kernels` the default of `PlaneWave` on every device
   (`DevicePlaneWave::Dense` the named opt-out); `DevicePlaneWaveLayout::Auto` cubes of 128
   units to p = 12 and 256 from p = 13 on CUDA, 128 on Metal, the CPU layout on the CPU
   runtime; the plane-wave budget's default 1 GB on Metal (the translation budget
   elsewhere); the kernels kept;
2. a new roadmap task for the device `Auto` to resolve to `PlaneWave` (CH7.5e, off the
   priority list);
3. D9: the device executables and gates on the CPU runtime, Metal and CUDA as the
   evidence, plus one ABBA run of `nd-fmm-bench --backend cuda` at the defaults (f32 and
   f64, p = 8, N = 10⁶) against `main` on locust; no Metal D9 timing.
