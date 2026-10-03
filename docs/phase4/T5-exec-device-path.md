# Phase 4 / T5 — nd-fmm-exec: the device path with host fallback (C4.1)

Give `nd-fmm-exec` a device path: a backend setting on `FmmBuilder` and a device
operator that implements `FmmOperator`. Data stays on the device as the signed-off
design specifies, and plan views and tables are uploaded once per `Fmm`. In this task
**every operator kind runs on the host fallback**: the device operator moves data
exactly as the final device path will, but computes each kind with the
`LaplaceOperator` kernels on the host. The FMM therefore runs end to end on the device
path from the start, and must equal the host path bit for bit. T6–T10 then replace one
kind at a time with a device kernel.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-kernels/CLAUDE.md;
- docs/phase4/README.md ("Requirements", "Design decisions", "Exit gate" C4.1);
- docs/design/device-path.md, signed off: sections 3, 4, 7, 8, 11, 12 and 15;
- fmm-exec/src/operator.rs (the module docs; `Kernels` and the `*_target` bodies),
  fmm-exec/src/fmm.rs (`FmmBuilder`, `build`, `SettingsError`, `Fmm::evaluate`,
  `StageTimings`, the accessors), fmm-exec/src/tables.rs;
- the evaluator stages and `reset` in fmm-plan/src/evaluator.rs;
- fmm-exec/tests/mpi_exec.rs (`cases`, `evaluate_threaded`, `evaluate_every_kernel`);
- the `nd-fmm-kernels` public API.

Do:
- Cargo:
  - `nd-fmm-kernels` as an optional dependency of `nd-fmm-exec` (`workspace = true`),
    behind a feature `gpu`;
  - the backend features `metal`, `cpu` and `cuda`, each enabling `gpu` and the
    matching `nd-fmm-kernels` feature;
  - none of them on by default;
  - the same features on `nd-fmm-validate`, passed through.

  Update fmm-exec/CLAUDE.md: the allowed dependencies (`nd-fmm-kernels` from Phase 4,
  feature `gpu`; still no direct `cubecl`), the device rules (requirements 3–9), and the
  test commands with features.
- **Settings** (names as the design fixes them):
  - the backend setting on `FmmBuilder`, default `Host`;
  - at `build`, a device that cannot be opened, an unsupported precision (f64 on Metal)
    or a configuration that does not fit in device memory gives a `SettingsError`;
  - agree that error by `build`'s existing all-reduce of input errors, with no new
    collective;
  - a device backend on more than one rank: `build` runs steps 1–5 as today, so that
    `PointsNotOwned` still wins where it applies, then every rank returns
    `SettingsError::DeviceNeedsOneRank { ranks }`. The rank count is the same on every
    rank, so no collective is needed (device-path.md §2, §4.4);
  - the threads rule of device-path.md §11: with the CPU backend no rayon pool is
    built, `threads(n)` caps the units per cube of the CPU layouts, and host-fallback
    kinds run serially; with Metal or CUDA the pool of `threads(n)` serves the
    host-fallback kinds only;
  - `Fmm` reports the backend and device, and per operator kind whether it runs on the
    device or the host;
  - `Backend::probe`, a thin wrapper over `Device::open` that returns the reason a
    device is unavailable, since `SettingsError::NoDevice` is `Copy` and cannot carry
    it (device-path.md §3.3, §12).
- **The device operator**, implementing `FmmSizes` and `FmmOperator`:
  - at construction: upload the plan views, frames and tables it needs (once per
    `Fmm`), and allocate the device level buffers and leaf stores as the design
    specifies;
  - at each evaluation: upload charges, keep every intermediate on the device, and
    download the target output once. The points are uploaded once per build by
    `DeviceOperator::load_points`, from copies of the source store and target input
    that `build` takes after step 8, since the store accessors and `operator_mut()`
    cannot borrow the evaluator at once (device-path.md §4.3);
  - keep the device copies consistent with every host-side write of the `Evaluator`
    listed in the design's section 4, without an `nd-fmm-plan` change (device-path.md
    §4.3, option (a)): `Fmm::evaluate` calls the operator's `begin_evaluation` after
    `reset` (zero kernels, the charge upload and scatter) and `read_output` after the
    stages (one download, one sync). `DeviceOperator::new` checks that the plan has no
    ghost leaf and no ghost box;
  - a host fallback per kind: download the call's inputs, run the `LaplaceOperator`
    body (the same `Kernels::*_target` code, serially or on the `Fmm`'s pool), and
    upload the output. Selectable per kind; in this task every kind uses it;
  - **transfer accounting**: count the bytes and calls of every upload and download
    per evaluation, and every launch and sync, exposed for tests and reports.
- The `view` types of `nd-fmm-kernels` (`IndexView`, `GroupedView`, `BoxCoordinates`,
  `LeafCoordinates`; device-path.md §3.1) and their upload at build, the index arrays
  validated on the host.
- `StageTimings` on the device path as the design's section 8 specifies. Without a
  sync per stage they measure enqueue time, and the docs say so;
  `FmmBuilder::synchronous_stages(true)` syncs after every stage so that they time each
  stage, for reports only, off by default (device-path.md §8.3). With the host
  fallback the stages are dominated by transfers; say so in the docs.
- Crate docs: the backends, the residency (what lives where, the transfers per
  evaluation), the fallback, determinism (requirement 6), errors, and the threads rule.
  The module docs of `operator` gain a device section, or a new module carries it.

Tests that define done (on the CPU runtime in f32 and f64, every test printing the
backends it ran; on Metal in f32, `#[ignore]`d and run by hand):
- Bit identity with every kind on the host fallback. For every scenario of
  `tests/mpi_exec.rs` (uniform and adaptive trees, gradients on and off, empty leaves,
  coincident points), the device path's output equals the host path's bit for bit, at
  1 thread. Add a helper beside `evaluate_threaded` that runs a scenario on every
  backend compiled in. Keep the debug run within its budget: run the device repetition
  on the small scenarios only if needed.
- Transfers:
  - per evaluation, equal to the design's formula for the scenario (device-path.md
    §4.1, §7.2), counted by the accounting;
  - plan views and tables uploaded only at build: zero bytes of them in a second
    evaluation;
  - two evaluations with different charges both equal the host path.
- Settings:
  - f64 with the Metal backend is refused with `SettingsError` at build (by hand, on
    the M3 Max);
  - a backend not compiled in is refused;
  - a device backend on 2 ranks returns `DeviceNeedsOneRank` on both, or
    `PointsNotOwned` where that applies (in the 2-rank run below);
  - the PR shows that the check goes through `build`'s existing agreement of input
    errors;
  - `Host` stays the default, and with it nothing about the host path changes: the
    existing tests pass unchanged without the `gpu` feature.
- The threads rule of device-path.md §11: no rayon pool with the CPU backend.
- The `view` types against the plan's views they are uploaded from.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features, unchanged and
  within its one-minute debug budget;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release`, and with
  `--features metal --release -- --ignored` by hand, outside the sandbox;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- `cargo clippy -p nd-fmm-exec -p nd-fmm-validate --all-targets -- -D warnings`, without
  features and with `--features cpu,metal`;
- `cargo check -p nd-fmm-exec --features cuda`;
- `cargo doc -p nd-fmm-exec --no-deps` without warnings, also with `--features cpu`;
- `tests/mpi_exec.rs` with `--features cpu` on 2 ranks, by hand, under an external
  timeout, as in Phase 3S T6. It must neither hang nor diverge, and the device setting
  errors must be agreed on both ranks: each device scenario stops with
  `PointsNotOwned` or `DeviceNeedsOneRank` (device-path.md §4.4);
- the CPU-runtime CI job, if T4 kept it. Extend it to `nd-fmm-exec --features cpu` only
  if its budget allows and the job then installs MPI. Say which you did;
- the root checks and the stricter workspace checks.

Report:
- the backends run;
- the transfers per evaluation for the C3.2 problem (bytes and calls, by kind of data);
- evaluation times of the device path with full fallback against the host path, as a
  baseline of the transfer cost (not asserted).

Do not:
- write operator kernels (T6–T10);
- add `unsafe` to nd-fmm-exec, or a direct `cubecl` dependency;
- change the host path, its defaults or its results;
- change nd-fmm-plan (no T4b in Phase 4, device-path.md §4.3);
- use atomics.
