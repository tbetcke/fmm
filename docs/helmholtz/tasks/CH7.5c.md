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

## Pending runs
(filled in step 6)

## Open questions
Answered before any code (2026-11-05, the user): the plan "Go ahead (Recommended)": build
the layout setting and the planewave part, run on locust, the adoption question 1 of the
PR; the M6 timings "Ask again first": prepare everything and run locust, then ask before
any timing on the M6.
