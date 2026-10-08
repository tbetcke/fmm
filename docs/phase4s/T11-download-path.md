# Phase 4S / T11 — the download and upload paths: measure, then a zero-copy download and a leaner upload (C4S.10)

T9 made the output pass and the charge load parallel and moved the output pass to the
device (fmm-bench/results/phase4s-t9-host-part.md, device-path.md §18.4). What is left
of the host part of a device evaluation is mostly the **download**:
- on CUDA at N = 10⁷ the download takes 12.7 ms (f32, 160 MB) and 24.6 ms (f64), while
  the copy itself takes 0.54 ms and 1.08 ms on the GPU (T8's `nsys`); at N = 10⁶ it is
  1.25 / 2.25 ms of 6.6 / 10.2 ms evaluations;
- on Metal at N = 10⁶ (f32) it is about 0.7 ms of a 14 ms evaluation (p = 3);
- the load, now mostly the charge upload, the zeroing and their sync, is 6.6 / 13.6 ms
  on CUDA at N = 10⁷ (f32 / f64) and 1.7 ms on Metal at N = 10⁶.

Reading the source of CubeCL 0.11.0-pre.4 shows where the download's host time can go
(a reading, not a measurement; this task measures it):
- `ComputeClient::read_one` returns a `cubecl::bytes::Bytes` that wraps CubeCL's own
  host memory: pinned memory from a pool on CUDA (`cubecl-cuda`, `PinnedMemoryStorage`,
  `cuMemAllocHost_v2` when the pool grows; T9's trace counted 6 calls in its whole run,
  median 2.5 ms, at most 54 ms), a mapped staging buffer from a pool on wgpu/Metal
  (`cubecl-wgpu`, `reserve_staging`, `map_async`);
- `nd_fmm_kernels::Device::download` then copies those bytes on one thread into the
  caller's slice (`out.copy_from_slice(E::from_bytes(&bytes))`): for the output, into
  the device operator's host buffer, from which `nd-fmm-exec` copies again into the
  `Output` (in parallel since T9). The output crosses host memory twice after the GPU;
- `Device::write` copies its data into a fresh `Vec` (`data.to_vec()`) before
  `ComputeClient::write`: the charges, N_s values, once more per evaluation.

This task first **measures** the steps of both paths on Metal and CUDA, then removes the
copies that are in this repository's code, and decides from the numbers whether CubeCL's
pools need anything. Every change is a copy or the removal of one: **no output bit
changes**, and an evaluation keeps its one upload, one download and one sync.

It develops on the M3 Max (the CPU runtime, Metal) and measures on locust (CUDA). The
host path does not download and is unaffected.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-bench/CLAUDE.md;
- docs/phase4s/README.md (requirements 2, 3 and 8, "Timing", decisions 14 and 15);
- device-path.md §4.1 (transfers per evaluation), §8 (syncs), §12 (errors: a launch error
  surfaces at the download), §18.3 and §18.4; fmm-bench/results/phase4s-t9-host-part.md
  ("What is left", the `nsys` section);
- `nd_fmm_kernels::device` (`Device::download`, `Device::write`, `Counters`) and
  `nd_fmm_exec::device` (`DeviceOperator::{begin_evaluation, read_output}`,
  `DeviceOutput`), `nd_fmm_exec::fmm` (`copy_output`, `gather_output`, `gather_charges`);
- CubeCL 0.11.0-pre.4 (`~/.cargo/registry/src/*/`): `cubecl-runtime` `client.rs`
  (`read_one`, `read_async`, `write`, `memory_persistent_allocation`),
  `cubecl-environment` `bytes/base.rs` (`Bytes`, its `Deref`, `from_elems`),
  `cubecl-cuda` `compute/server.rs` and `compute/storage/cpu.rs` (the pinned pool),
  `cubecl-wgpu` `compute/stream.rs` and `compute/mem_manager.rs` (the staging pool).

Do:
1. **Measure the paths** (a spike, `spikes/download-path/`, excluded from CI like the
   other spikes; it may use CubeCL directly and `unsafe` under the root rules). On
   Metal (M3 Max, f32) and CUDA (locust, f32 and f64), for 16, 160 and 320 MB buffers
   (N = 10⁶ and 10⁷ outputs with gradients):
   - the download split into: the wait for queued work (a `sync` first), `read_one`
     (the GPU copy, the pool, the mapping), and the copy into a host slice
     (`copy_from_slice`, one thread), and the same copy in parallel for comparison;
   - whether the pool allocates again in every download or only on growth: ten
     downloads in a row, and on CUDA one `nsys` run counting `cuMemAllocHost` per
     download; the same with `ComputeClient::memory_persistent_allocation` around the
     reads, if it covers the host pools (read the source first and say);
   - the upload split into `to_vec`, `ComputeClient::write` (to its queueing) and the
     transfer (to a following sync); and the zeroing kernels of `begin_evaluation` by
     timing windows (CUDA events, Metal timestamps).

   Report in `spikes/download-path/REPORT.md` with every number labelled "measured
   (machine, backend)", and the locust load checks of the README, "Timing". **Stop and
   bring the report for decisions 14 and 15 before item 2.**
2. **A zero-copy download** (decision 14). In `nd-fmm-kernels`, a way to read a
   downloaded range without copying it into a caller's slice: for example a guard
   `HostValues<'_, E>` that holds CubeCL's `Bytes` and derefs to `&[E]`, returned by
   `Device::download_view(slice)`, with the same counters (one download, one sync), the
   same checks (owner, length) and the same errors as `download`. Propose the exact API
   in the decision; keep every public function safe, with no `unsafe` beyond what the
   crate's rules allow (a cast of CubeCL's bytes as `download` does now). In
   `nd-fmm-exec`, `read_output` hands the view to `Fmm`, which copies it straight into
   the `Output`:
   - with the output pass on the device (CUDA, the CPU runtime): `copy_output` reads the
     view; the operator's host output store is no longer a download target for it;
   - with the host pass (Metal, `OutputPass::Host`): the view holds the leaf-ordered
     target output, which `gather_output` reads as it reads a `LeafStore` now (a view of
     the store's layout, or the pass generalised over a slice and the point offsets);
   - host-fallback calls keep `download` into their mirrors (their regions are small and
     are uploaded again);
   - `DeviceOutput`'s variants carry the view instead of a borrowed host buffer; the
     operator stays usable for the next evaluation once the view is dropped.
   Remove the device operator's host output buffer if nothing else needs it, and say what
   it saves (o N_t s bytes of host memory).
3. **A leaner upload** (if item 1 shows the copy matters). `Device::write` without the
   extra copy where the caller can give up its buffer (for example `write_owned(slice,
   Vec<E>)`, which hands the `Vec` to `Bytes::from_elems`), and `Fmm` gathering the
   charges into such a buffer by T9's parallel gather, without a zero fill. Keep `write`
   for the other callers.
4. **CubeCL's pools** (decision 15, only if item 1 shows a pool allocating again in
   steady state). Through CubeCL's public API at the pinned version only (for example
   `memory_persistent_allocation`, or reserving the pinned or staging size at build).
   If the public API cannot do it, write the upstream question with the measurements
   (`spikes/download-path/REPORT.md`, "Upstream") and change nothing; moving off
   `=0.11.0-pre.4` stays a separate decision.
5. **Docs.**
   - fmm-kernels/CLAUDE.md: the new download (and upload) API and its rules.
   - fmm-exec/CLAUDE.md: the output's path from the device to the `Output`.
   - device-path.md: §4.1 if a host buffer goes away; a §18.5 "The download and upload
     paths (T11)" with the spike's numbers and the before and after.
   - laplace-fmm-plan.md §7, Phase 4S: a C4S.10 row.
   - The Phase 4S README: the exit checklist.

Tests that define done:
- **The kernel API** (nd-fmm-kernels' suite, the CPU runtime in CI, Metal and CUDA by
  hand): the view equals `download` bit for bit, f32, f64 and u32, sizes 0, 1, 7 and
  10⁵, ranges at odd offsets inside larger buffers; the counters (one download, one
  sync, the bytes); a wrong device refused as `download` refuses it; a launch error
  surfaces at the view as at `download` (device-path.md §12). With item 3: the upload
  without the copy equals `write` bit for bit, with the same counters.
- **Whole outputs unchanged**: every existing scenario of `tests/mpi_exec.rs`,
  `tests/device_common` (the CPU runtime in CI, Metal and CUDA by hand) and the ignored
  gates, unchanged and passing, bit for bit `Fmm::reference_output` with both output
  passes; the C3.2 hashes of `tests/output_common` equal T9's
  (fmm-bench/results/phase4s-t9-host-part.md) on every backend.
- **Transfers and syncs**: the formula of device-path.md §4.1 and its tests unchanged (or
  changed only where a host buffer goes away, said so): one upload, one download, one
  sync per evaluation.
- **Memory**: if the host output buffer goes away, a test that the device operator's host
  memory no longer holds it (or the report says what replaced it).
- **The bench**: the smoke test unchanged and passing; no new column.

Measurements (reported, never asserted; release builds; every BLAS thread variable 1;
on locust with the load checks of the README, "Timing"):
- the spike of item 1 (before any change);
- `nd-fmm-bench` before (T9's code, its numbers if the machine and settings are the same,
  otherwise rerun) and after, the unit cube with gradients: CUDA at N = 10⁶ and 10⁷, f32
  and f64, p = 3 and 8; Metal f32 at N = 10⁶ (p = 3, 8) and, if it fits, 10⁷ (p = 3);
  each with `load`, `output` and the download's part, with and without `--reuse-output`;
- one `nsys` trace at N = 10⁷, f32, p = 3, after: `cuMemAllocHost` calls and the host
  time left in the download.

Must pass:
- `cargo fmt --all`, then the root checks and the stricter workspace checks;
- the nd-fmm-kernels checks of fmm-kernels/CLAUDE.md (CPU runtime; Metal by hand) and the
  CPU-runtime CI job's commands;
- the device-path checks of fmm-exec/CLAUDE.md, the ignored gates with `--features cpu`,
  and the Metal run by hand outside the sandbox;
- the nd-fmm-bench checks of fmm-bench/CLAUDE.md;
- on locust: the CUDA commands of the root CLAUDE.md ("Checks", on locust), and
  `RUST_MIN_STACK=8388608 cargo test --workspace`.

Do not:
- change any output bit, the order or number of launches, transfers or syncs of an
  evaluation, or any kernel's arithmetic;
- add `unsafe` to `nd-fmm-exec`, or `unsafe` in `nd-fmm-kernels` beyond its rules (each
  block with a `// SAFETY:` comment, every public function safe);
- hold a CubeCL host buffer across evaluations in a way that changes what the next
  evaluation computes or allocates on the device;
- patch, fork or vendor CubeCL, or move off `=0.11.0-pre.4`;
- add a dependency (rayon stays out of `nd-fmm-kernels`);
- change `Fmm::evaluate`'s or `Fmm::evaluate_into`'s signature or results, or the default
  of `nd-fmm-bench`;
- assert a timing.
