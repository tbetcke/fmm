# CH7.5a Laplace's plane-wave M2L on the device

Started 2026-11-05 on `tbetcke/h-ch7-5a`. Roadmap: "CH7.5a Laplace's plane-wave M2L on the
device" (priority list item 1, the user, 2026-11-05). Proposal: §10 CH7.5 (split by the user
on 2026-11-05 into CH7.5a and CH7.5b); §2 constraint 1; §8 (the device path). Task brief it
continues: docs/phase6/T13-plane-wave-m2l.md, "Afterwards".

## Goal
Merge and shift of CONVENTIONS §3.14 as `nd-fmm-kernels` kernels, run by the device's
`PlaneWave`, measured against T7's dense M2L, and adopted on the device where it wins at
p ≥ 12 (else dropped).

## Acceptance of CH7.5 on Laplace (before the split, verbatim)
`PlaneWave` on the device (CUDA f32 and f64, Metal f32, the CubeCL CPU runtime in CI)
within 1.1× `Dense`'s errors against the direct sum at the C3.2 cube and the C3.3 Plummer
sphere (T13's requirement 1, f64 p = 8, 12, 18 and f32 p = 8), device locals against the
host `PlaneWave`'s within the stated bound, bit for bit from run to run and on P ranks
against the one-rank device run; the device M2L stage and the evaluation against T7's
dense M2L at f64 p = 8, 12, 18 on locust's H100 (f32 on the M6's Metal); adopted as the
device's `PlaneWave` where it wins at p ≥ 12 (T13's gate), else `PlaneWave` stays `Dense`
on the device and the kernels are dropped. Constraint 1: only the opt-in device
`PlaneWave` changes its output; every other strategy and Laplace's defaults bit for bit,
the device gates and Laplace's benchmarks within noise (D9); the device `Auto` unchanged
(a change to it is a sign-off).

## Size: the split (agreed 2026-11-05)
The task is a new operator family on the device (four to five kernels in
`nd-fmm-kernels`), its level call in `nd-fmm-exec`'s device operator, and measurements on
two machines: well over one reviewable PR. Split in the roadmap on 2026-11-05 (the user:
"a/b/c, Helmholtz → d"); this task is the first part:

- **CH7.5a The kernels** (`nd-fmm-kernels` only). *Done when:* the kernels of the steps
  of `nd_fmm_tables::plane_wave` (below) on the CubeCL CPU runtime (CI, f32 and f64), Metal
  (f32) and CUDA (f32, f64) equal a host replica of their order bit for bit and the
  tables' own functions (`waves_from_modes`, `shift_add`, `sums_from_waves`, …) within a
  stated bound; a whole level of merge and shift against `PlaneWaveTables::m2l` per pair
  and against `nd_fmm_ref::direct::m2l` on levels 2, 9 and 16 within T13's quadrature
  bound (p ≤ 20 f64, ignored sweep); deterministic run to run, independent of the layout.
  No `nd-fmm-exec` change. *Needs:* Phase 6 T13. *Machines:* locust (CUDA), the M6 (Metal).
- **CH7.5b The device's `PlaneWave`** (`nd-fmm-exec`'s device path). *Done when:* the
  roadmap's accuracy, determinism and rank items above, and constraint 1's bit-for-bit
  items, on the CPU runtime in CI, Metal by hand and CUDA on locust; opt-in behind a
  builder setting (the device's `PlaneWave` keeps running as `Dense` unless it is named), so
  no default and no other strategy changes. *Needs:* CH7.5a.
- **CH7.5c Measured, and adoption** (`nd-fmm-validate`/`nd-fmm-bench`, report). *Done
  when:* the roadmap's timing items on locust's H100 (f64 p = 8, 12, 18) and the M6's
  Metal (f32 p = 8), D9 on the device gates; *Sign-off:* adoption (the setting becomes the
  device's `PlaneWave`, or the kernels are dropped). *Needs:* CH7.5b. *Machines:* locust,
  the M6.
- Helmholtz's, today CH7.5b, renamed **CH7.5d** (its *Needs* CH7.5a → CH7.5c).

## Read first
- fmm-kernels/CLAUDE.md (rules: no atomics, owner-computes, explicit `fma`, layouts per
  backend, CPU unit cap); `fmm-kernels/src/rotation.rs` (the model: one launch per level,
  tables in their own storage uploaded once, a host `mul_add` replica tested bit for bit).
- `nd_fmm_tables::plane_wave` (module docs: the steps table, `LambdaBlock`, the frames,
  `geometry`: directions, slots, accumulators), CONVENTIONS §3.14.
- `nd-fmm-exec/src/operator_plane_wave.rs` (the host order: per vector and frame the waves
  of every box, then per parent: slots from the computed children's V rows, every
  accumulator over its slots in ascending order, each child's accumulators in order with
  its child shift, the sums, the local, the rotation back).
- `nd_fmm_simd::waves` (the `WaveKernel` loops and their `mul_add` chains).
- `spikes/plane-wave/SPIKE_REPORT.md` §3 (the device model: 2.4–20×, shifts
  bandwidth-bound, the irregular merge lists the open question).

## Changes (CH7.5a)
`nd-fmm-kernels`, a new module `plane_wave` (feature-free like `rotation`):
- `PlaneWaveTables` on the device: the host's tables in their own storage (outgoing and
  incoming powers, cos/sin rows, the frame rotations, the source shifts and child shifts,
  the λ blocks as `u32`), uploaded once from plain slices (no `nd-fmm-tables` dependency:
  `nd-fmm-exec` passes the arrays, as for `RotationArrays`).
- Kernels, one launch per step and level call:
  1. **outgoing**: per (box, frame) the multipole rotated into the frame, its weighted
     modes and its up and down waves (`rotate_multipole`, `outgoing_modes`,
     `waves_from_modes`), into a waves buffer of 6 × 2H values per box;
  2. **merge**: per (parent, direction, accumulator) the sum over its slots in ascending
     order of shift × wave (`shift_add`), from a merge list built on the host per level
     call (CSR: per parent and direction, each accumulator's (wave index, shift index)
     pairs; empty slots and uncomputed children dropped by the same geometric masks as
     the host);
  3. **gather**: per (child, direction) the sum of its accumulators in order times its
     child shift (`add`, `shift`), into the incoming waves;
  4. **local**: per (child, frame) the sums Σⱼ cos/sin, the local from the sums and the
     rotation back, added into the local (`sums_from_waves`, `local_from_sums`,
     `add_local_from_frame`).
  Each output value has one owner; every chain is an explicit `fma` sequence in the order of
  the table function it restates; no atomics.
- The merge list's type (`MergeView`, validated at upload like the other views), the
  layouts (`PlaneWaveLayout`, cube per parent on GPUs, the CPU layout honouring the unit
  cap), `bytes()` formulas.
- Tests in `fmm-kernels/tests/kernels/plane_wave.rs`.

Public API: new items only. Conventions: none (§3.14 as it stands). Dependencies: none.
`nd-fmm-plan`: untouched.

## Tests that prove it (CH7.5a)
- Each kernel against a host replica of its `fma` order, bit for bit, f32 and f64, p = 3,
  8, 12, 18 (20 in the ignored sweep), every layout, on the CPU runtime (CI), Metal (f32,
  by hand) and CUDA (locust, by hand).
- Each kernel against the `nd_fmm_tables::plane_wave` function it restates within a
  stated bound (relative, measured and given ~10× margin, glibc rule).
- A level call (outgoing, merge, gather, local) over a synthetic level of full parents
  and of an adaptive level (missing children, empty slots) against
  `PlaneWaveTables::m2l` per pair (to rounding) and against `nd_fmm_ref::direct::m2l` on
  levels 2, 9 and 16 within T13's quadrature bound.
- Two launches bit for bit; the result the same on every layout.

## Checks
`/verify` (format, clippy default and workspace, tests, docs) plus fmm-kernels' own:
`cargo clippy -p nd-fmm-kernels --all-targets --features cpu,metal -- -D warnings`,
`cargo check -p nd-fmm-kernels --features cuda`,
`cargo test -p nd-fmm-kernels --features cpu --release -- --show-output`; Metal ignored
tests on the M6 outside the sandbox; on locust `cargo test -p nd-fmm-kernels --release
--features cpu,cuda -- --include-ignored --show-output` and its clippy.

## Machine runs
CH7.5a: locust (CUDA correctness, no timing), the M6 (Metal correctness, no timing).
Timings are CH7.5c's.

## Constraints and risks
- Constraint 1: CH7.5a adds kernels and touches no existing one; no Laplace output can
  change. CH7.5b changes the output of the device's `PlaneWave` only when named; CH7.5c
  asks for adoption.
- `nd-fmm-kernels`' rules: no atomics, determinism by ownership, explicit `fma` (cubecl's
  `InstCombinePass` fuses lone a · b ± c), f64 refused where the device has none (Metal).
- Risk: the waves buffer (6 × 2H values per box per vector) on the device; CH7.5b bounds it
  by chunks of parents, as CH6.2a did on the host.
- Risk: the irregular merge lists (the spike's open question); CH7.5a times nothing but
  keeps the merge per (parent, direction) so that CH7.5c can measure it.

## Pending runs
(none yet)

## Open questions
Answered 2026-11-05 before the branch: the split as above ("a/b/c, Helmholtz → d"), and
constraint 1 ("Go ahead": new kernels in CH7.5a, the device's `PlaneWave` opt-in in CH7.5b
until CH7.5c's sign-off).
