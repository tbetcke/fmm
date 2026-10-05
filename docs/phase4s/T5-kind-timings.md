# Phase 4S / T5 — timings per operator kind, on the host and on the device (C4S.5)

`Fmm::evaluate` times stages, not kernels. `StageTimings` has `upward_local`,
`downward`, `evaluate_leaves` and so on. Upward is P2M and M2M, downward is L2L, M2L and
P2L, and leaves is L2P, M2P and P2P (device-path.md §8.3). M2L and P2P alone are only in
the kernel harnesses (`m2l_kernels`, `p2p_kernels`), on synthetic batches, never inside
an evaluation. The Phase 4S benchmark (T6) must report each kind separately: P2M, M2M,
M2L, L2L, P2L, M2P, L2P and P2P. This task adds that to `nd-fmm-exec`, for the host path
and every device backend. It is opt-in, and it leaves the output unchanged bit for bit.

It does not depend on CUDA. Develop and test it on the M3 Max (host, CPU runtime,
Metal). T4 and T7 then run it on locust.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-kernels/CLAUDE.md;
- docs/phase4s/README.md ("Timing", requirement 6);
- docs/design/device-path.md §8 (launches and syncs, the one sync, `StageTimings` without
  a sync per stage), §11 (threads), §17.1 (stage timing as built: `synchronous_stages`,
  `device_timestamps` and their overlap on Metal);
- `nd_fmm_exec::fmm`: `StageTimings`, `DeviceStageTimings`, `FmmBuilder::{
  synchronous_stages, device_timestamps}`, `ExecOperator` and its `delegate!` macro;
- `nd_fmm_kernels::Device::{open_window, close_window, times_on_device, sync}` and
  `WindowTime`;
- `nd_fmm_plan::operator` ("Accumulation rule"): the order of the level calls.

Do:
- **The setting.** `FmmBuilder::kind_timings(mode: KindTiming)`, with
  `KindTiming::{Off, Synchronous, Device}` and `Off` the default:
  - `Off`: nothing changes, and no timer or sync is added (the default evaluation is
    the Phase 4 one);
  - `Synchronous`: every operator call is timed by the wall clock on the calling thread.
    On the host path the call is already synchronous (the rayon pool joins before it
    returns). On a device backend `Fmm` syncs the device after each call, so the time is
    the device work of that call. That adds one sync per call with device work, about
    40–130 per evaluation (device-path.md §17.2). Report-only, never the default;
  - `Device`: on a backend that times on the device (`Device::times_on_device`, CUDA and
    Metal with timestamp queries) one timing window per call, with no added sync,
    resolved after the download. Elsewhere `build` refuses it with a `SettingsError`
    (agreed by step 1's all-reduce, no new collective). Check that the windows of
    neighbouring calls do not overlap on the backend you test (they did between stages on
    Metal, T11). If they do, document it: the per-kind sums then overstate the total.
    T4 checks it on CUDA.

  The names are suggestions. Keep them consistent with `synchronous_stages` and
  `device_timestamps`, and say whether `kind_timings(Synchronous)` implies
  `synchronous_stages` or composes with it.
- **The result.** `StageTimings` gains `kinds: Option<KindTimings>`: per
  `OperatorKind`, the total time, the number of calls, and per level the time. It is
  `None` when the mode is `Off`. A method `KindTimings::total()` and the remainder
  `StageTimings::total() − kinds.total()` (loading, exchanges, the download, the
  scaling) make the gap visible in reports. Host-fallback kinds on a device backend are
  timed like any host call, and the result says where each kind ran.
- **Where.** The `delegate!` macro of `ExecOperator` is the one place every level call
  passes through. Wrap it there, behind the mode, so `LaplaceOperator`, the device
  driver and the kernels do not change. The device sync and window calls go through
  `DeviceDriver`, which needs one more method (or two). `nd-fmm-kernels` needs no change
  beyond, at most, a public wrapper it already has.
- **Docs.** The docs of `StageTimings`, `KindTimings` and the setting state:
  - what each mode measures;
  - that `Synchronous` on a device adds one sync per call, and so inflates the sum over
    kinds by the sync cost times the calls;
  - that `Device` measures device time only (no launch or enqueue time);
  - the per-call cost on the host: one `Instant::now` pair per call, about 40–1,000
    calls per evaluation.

  Update fmm-exec/CLAUDE.md and the "Timing" bullets of device-path.md §8.3 (one
  paragraph, "Added in Phase 4S T5").
- **The `fmm_accuracy` and `device_fmm` examples** of `nd-fmm-validate` do not change.
  T6's benchmark is the first user.

Tests that define done (new scenarios go into the existing MPI-owning executables:
`tests/mpi_exec.rs` for the host and the CPU runtime, `tests/device_metal.rs` for Metal):
- **No change to the output.** For each mode, the output is bit-identical to `Off`'s on
  every `tests/mpi_exec.rs` scenario, on the host at 1 and 4 threads and on the CPU
  runtime with every kind on the device. On Metal, by hand, the same at f32 p = 4 and 8.
- **Counts.** The calls per kind equal the level calls the evaluator makes (count them
  with the recording operator of `nd-fmm-plan`'s tests, or from the plan's non-empty
  views). Kinds whose views are all empty show zero calls and zero time.
- **Syncs.** With `Synchronous` on a device, the syncs per evaluation equal the
  Phase 4 formula plus one per device call. With `Device`, they equal the Phase 4
  formula (one per evaluation); `Fmm::device_counters` shows it.
- **Plausibility.** On the host at one thread, `kinds.total()` is at most
  `evaluate_leaves + downward + upward_local + upward_global` (the stages that call
  operators), and within 10% of it on the C3.2 cube at N = 10⁵, p = 6 (reported, and
  asserted only as "at most", never as a timing bound).
- **Refusal.** `KindTiming::Device` on the host and on the CPU runtime is refused with
  the new `SettingsError` at build, on every rank.

Must pass:
- `cargo fmt --all`, then the root checks and the stricter workspace checks;
- the device-path checks of fmm-exec/CLAUDE.md: clippy with `--features cpu,metal`,
  `cargo check -p nd-fmm-exec --features cuda`,
  `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release`, and the
  Metal run by hand outside the sandbox;
- `cargo doc -p nd-fmm-exec --no-deps --features cpu`.

Do not:
- change the default evaluation: with `Off`, no timer, sync or window is added;
- change the order of any call or launch, or move a kind between host and device;
- time inside `nd-fmm-plan` or `nd-fmm-kernels`, or add a dependency;
- assert any timing.
