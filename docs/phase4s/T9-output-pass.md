# Phase 4S / T9 — the host part of an evaluation: the output pass, the charge load and reusable outputs (C4S.8)

T8 found that, on the device, the largest cost of an evaluation at large N lies outside
every level call (fmm-bench/results/phase4s-gh200.md, Sections 1 and 7; device-path.md
§18.3). On locust's H100:
- this host part is 8.4 ms (f32) and 13 ms (f64) per evaluation at N = 10⁶, and
  183–189 ms and 298–303 ms at N = 10⁷;
- it is the same at every p, and 58–83% of an evaluation at N = 10⁷ for p ≤ 8;
- it caps the device's lead over the 72-core Grace host at 1.5–4.2× at N = 10⁷, against
  2.2–6.2× at N ≤ 10⁶;
- the host path pays it too: 10–21 ms at N = 10⁶ on 72 threads.

An `nsys` trace at N = 10⁷ puts `scaled_output` first (at least 29% of the CPU samples
in f32 and 32% in f64; half of the call stacks were broken). That is the single-threaded
pass that turns the leaf-ordered, leaf-scaled output into φ and ∇φ in the caller's
order. Then come the download path (3–4%) and a pinned host buffer per evaluation
(median 2.3–2.5 ms). The copies themselves are short: the 160 MB download takes 0.54 ms
on the GPU. The charge load at the start of an evaluation is the same kind of serial
per-point pass.

This task makes that host part fast **without changing a single output bit**:
- it times the parts separately;
- it parallelises the output pass and the charge load;
- it stops allocating and zeroing the outputs in every evaluation;
- it offers a way to evaluate into reused output buffers;
- on backends that can, it moves the output pass to the device.

It develops on the M3 Max (host, CPU runtime, Metal) and measures on locust (host,
CUDA).

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-bench/CLAUDE.md;
- docs/phase4s/README.md (requirements 2, 3 and 6, "Timing", decisions 11 and 12);
- fmm-bench/results/phase4s-gh200.md, Sections 1, 2 and 7, and device-path.md §18.2 (the
  findings "reported, not built") and §18.3;
- docs/design/device-path.md §4.1 (transfers per evaluation), §7.2, §8 (launches and
  syncs), §11 (threads and the rayon pool), §17.1;
- CONVENTIONS §3.1 (1/(4π) applied once by `nd-fmm-exec`) and §3.13 (leaf-scaled data);
- `nd_fmm_exec::fmm`: `Fmm::evaluate`, `scaled_output`, `LeafOrder`, `StageTimings`
  (`load`, `output`), `Output`, the `Fmm`'s rayon pool (`FmmBuilder::threads`);
  `nd_fmm_exec::device`: `DeviceDriver::{begin_evaluation, read_output}`;
- `nd_fmm_kernels::movement` (`gather_columns`, `scatter_values`) and the device index
  rule (index arrays are u32, device-path.md §15, T4).

Do:
1. **Time the parts.** `StageTimings` already records `load` (the charge gather and, on
   a device, the upload) and `output` (the download and `scaled_output`).
   - `nd-fmm-bench` reports them: two columns of the kind table, split out of "other",
     and the same in each combination's section. Update the golden Markdown string.
   - If `output` cannot separate the download from the pass on a device, add one timing
     inside `Fmm::evaluate` (host clock, no sync beyond the existing one) and document
     it in `StageTimings`.
   - Take the "before" numbers with this alone (Measurements, below).
2. **The output pass as a parallel gather.** At build, store for every target, in the
   caller's order:
   - its position in the leaf-ordered output store;
   - its leaf (or the leaf's two scales `4π r` and `4π r²`, computed exactly as now).

   The pass then writes `potential[i]` and `gradient[i]` from those, in order:
   - one parallel loop over i (`par_iter_mut().enumerate()` or chunks of it), on the
     `Fmm`'s rayon pool when it has one, serially otherwise;
   - writes in sequence and reads gathered, with no `unsafe` (nd-fmm-exec stays free of
     it).

   Each value is computed exactly as `scaled_output` computes it now, `T::from_f64(
   to_f64(x) / scale)`, so the output is bit-identical. On a device backend the pool is
   idle at that point (device-path.md §11), so using it costs no device work. State the
   memory the per-target index costs (u32 positions, 4 N bytes plus the scales), and
   keep it within the build's existing memory report.
3. **No fresh zeroed outputs.** Build `potential` and `gradient` by collecting the
   parallel iterator of item 2 into its `Vec`s (rayon's `collect`, safe and without a
   zero fill). No `vec![zero; n]` followed by an overwrite: every page is touched once,
   in parallel. Safe code only.
4. **The charge load in parallel.** The gather of the caller's charges into leaf order
   (the evaluator's source chunks and, on a device, the leaf-ordered upload buffer)
   writes disjoint per-leaf chunks: run it over the leaves on the pool, bit-identical.
   Measure whether the host source store needs filling at all when every kind runs on
   the device (it feeds the host fallback only). If it does not, skip it, documented and
   tested; otherwise say why it is needed.
5. **Reusable outputs.** Add
   `Fmm::evaluate_into(&mut self, charges: &[T], output: &mut Output<T>) -> Result<(),
   FmmError>`, which:
   - writes into the caller's buffers, resizing them only when their length is wrong;
   - leaves `Fmm::evaluate` as a thin wrapper that evaluates into a fresh `Output`.

   It is a new public API, so it needs **decision 11** (README). Propose the exact
   signature there, and say what happens to `gradient` when the setting changes and to
   `timings`.
   - `nd-fmm-bench` gains `--reuse-output`: the timed evaluations then go through
     `evaluate_into` with one `Output` reused. The default stays `evaluate`, so old
     reports stay comparable, and the header records which was used.
6. **The output pass on the device, where it is exact.** A kernel in `nd-fmm-kernels`
   (`movement`, with the other gathers) makes the caller-ordered φ and ∇φ on the device:
   - it gathers by the per-target positions of item 2, uploaded once per `Fmm` as a u32
     index view, and divides by the per-leaf scales uploaded once in f64;
   - one buffer of N + 3N values (φ, then ∇φ in the `[T; 3]` order of `Output`)
     replaces the leaf-ordered output download, so it is still **one download and one
     sync** per evaluation, and the host keeps only a copy into the `Output`.

   Bit identity forces the arithmetic: the host divides in f64 and rounds once to T. So
   the kernel converts to f64, divides by the f64 scale and rounds to T.
   - That needs f64 arithmetic on the device: CUDA and the CPU runtime have it (F27;
     correctly rounded division, F33). Metal has no f64 and keeps the host pass
     (items 2–4). The choice is made from `DeviceInfo` (Phase 4S requirement 2), not
     from the backend's name.
   - It is not a formulation change: the same 1/(4π) and leaf scaling, applied once by
     `nd-fmm-exec` (CONVENTIONS §3.1), now in a kernel it launches. Check this against
     §3.1 and §3.13 and say so in the PR.
   - Tests must be able to compare the device pass with the host pass on the same
     build, so the pass needs a control. Propose it as part of **decision 12**: a
     builder setting, or a placement through the existing host-fallback mechanism. The
     default is the device pass where the device has f64.
   - The transfer formula of device-path.md §4.1 and its tests change only in the
     download's size, if at all. Say which, and update both.
7. **Docs.**
   - fmm-exec/CLAUDE.md and fmm-kernels/CLAUDE.md: the new pass, the kernel, its
     capability rule.
   - fmm-bench/CLAUDE.md: the new columns and `--reuse-output`.
   - device-path.md: a §18.4 "The host part of an evaluation (T9)" with the before and
     after numbers, and §4.1 if the transfers change.
   - laplace-fmm-plan.md §7, Phase 4S: a C4S.8 row and one sentence in the
     "Recommendation for Phase 5", which names this work.
   - The Phase 4S README: the exit checklist.

Tests that define done (new scenarios go into the existing MPI-owning executables:
`tests/mpi_exec.rs` for the host and the CPU runtime, `tests/device_metal.rs` and
`tests/device_cuda.rs` for the GPUs):
- **Bit identity, the core of the task.** Keep today's `scaled_output` as a private test
  oracle (`#[cfg(test)]`, or a test-only helper). On every `tests/mpi_exec.rs` scenario
  the new pass equals it bit for bit:
  - on the host at 1 and 4 threads, in f32 and f64, with and without gradients;
  - on the CPU runtime with every kind on the device, both with the device pass of
    item 6 and with the host pass.

  On Metal by hand (host pass only) and on CUDA by hand (both passes), the same at f32
  p = 4 and 8 and (CUDA) f64 p = 6.
- **Whole outputs unchanged.** For the C3.2 cube at N = 10⁵, p = 6, the output of the
  default build equals the output from before this task bit for bit:
  - on the host and the CPU runtime: an `#[ignore]`d release test in an existing gate
    executable (`tests/accuracy.rs` for the host, `tests/device_fmm.rs` for the CPU
    runtime), run with `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release --
    --ignored` and with `--features cpu`, like the other C3.2 gates. It asserts against
    the output of the old pass (the test oracle above) on the same build, or against a
    hash recorded from the code before this task. It is not part of the debug run:
    `tests/mpi_exec.rs` keeps its one-minute budget, and its small scenarios carry the
    bit-identity checks above in CI;
  - on Metal and CUDA (by hand, in the ignored blocks of the same gates).

  This is Phase 4S requirement 3 for Metal and the CPU runtime.
- **`evaluate_into`.** Three evaluations into one reused `Output` equal three calls of
  `evaluate` bit for bit. A wrongly sized `Output` (wrong length, gradient present or
  absent) is resized, not refused. Charges of the wrong length give the same
  `FmmError::ChargesLength`, agreed on every rank, as `evaluate`.
- **The charge load.** Bit-identical leaf-ordered charges, host store and upload buffer
  at 1 and 4 threads. If the host store is skipped (item 4), every kind on the host
  fallback still equals the host path bit for bit (the existing C4.1 scenario).
- **Transfers and syncs.** One upload, one download and one sync per evaluation on every
  device backend, with the bytes of the (possibly changed) formula, asserted as today.
- **The kernel** (nd-fmm-kernels' suite, CPU runtime in CI, CUDA by hand): the
  caller-ordered gather and scaling against a host loop bit for bit:
  - in f32 and f64, sizes 0, 1, 7 and 10⁵, with an empty leaf;
  - the u32 index bound checked before the unchecked launch;
  - refused with `KernelError::UnsupportedPrecision`-style capability errors on a device
    without f64 (Metal).
- **The bench.** The smoke test covers the new columns and `--reuse-output`.

Measurements (reported, never asserted; release builds; every BLAS thread variable 1;
on locust with the load checks of the README, "Timing"):
- `nd-fmm-bench` before and after, the unit cube with gradients:
  - CUDA at N = 10⁶ and 10⁷: f32 p = 3 and 8, f64 p = 3 and 8;
  - the host at 72 threads, the same points;
  - Metal f32 at N = 10⁶ (p = 3, 8) on the M3 Max;
  - each with `load`, `output` and the rest of "other", and with and without
    `--reuse-output`.
- The device pass against the host pass on CUDA at N = 10⁷: `output` alone, and the
  evaluation.
- One `nsys` trace at N = 10⁷, f32 p = 3, after the change: what is left of the host
  part, by function. Rebuild `nd-fmm-bench` with `--features cuda` before running it
  directly (tools/gh200/README.md).
- *Model, for orientation only:* the pass moves about 0.3–0.6 GB at N = 10⁷, a few
  milliseconds over Grace's cores. T8's samples put it at 55–95 ms or more today.

Must pass:
- `cargo fmt --all`, then the root checks and the stricter workspace checks;
- the device-path checks of fmm-exec/CLAUDE.md:
  - clippy with `--features cpu,metal`;
  - `cargo check -p nd-fmm-exec --features cuda`;
  - `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release`, and the
    same with `-- --ignored`;
  - `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored` (the host's
    gates, the whole-output check included);
  - `cargo doc -p nd-fmm-exec --no-deps --features cpu`;
  - the Metal run by hand, outside the sandbox;
- the nd-fmm-kernels checks of fmm-kernels/CLAUDE.md (CPU runtime; Metal by hand) and the
  CPU-runtime CI job's commands;
- the nd-fmm-bench checks of fmm-bench/CLAUDE.md;
- on locust: the CUDA commands of the root CLAUDE.md ("Checks", on locust), and
  `RUST_MIN_STACK=8388608 cargo test --workspace`.

Do not:
- change any output bit: not the value of an element, its order, or the rounding of
  the division (no reciprocal multiply, no fused scaling, no f32 division on the device);
- change the order or number of level calls, the accumulation order or any kernel's
  arithmetic;
- add a sync, a transfer or a collective to an evaluation beyond what decisions 11 and
  12 accept;
- use `unsafe` in `nd-fmm-exec`, or uninitialised memory anywhere;
- change `Fmm::evaluate`'s signature or results, or the default of `nd-fmm-bench`;
- add a dependency;
- assert a timing.
