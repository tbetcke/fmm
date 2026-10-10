# Phase 6 / T7 — the fused gather and the device GEMM work (C6.1)

The device's dense M2L, M2M and L2L run per chunk as a gather of the source columns into
scratch, a grouped GEMM, and a reduction into the targets (device-path.md §6.4,
`nd_fmm_kernels::translate::grouped`, `movement::gather_columns`). The gather writes and
re-reads the scratch through device memory. C6.1 fuses it into the GEMM's loads: the GEMM
reads each source column straight from the multipole store through the batch's source
indices (laplace-fmm-plan §7, C6.1: "faster than C4.5 at equal accuracy").

Phase 4S left three kernel items for later (device-path.md §18.2; laplace-fmm-plan §7,
Phase 4S, "Recommendation for Phase 5", last bullet): a shared-memory-tiled GEMM for the
rest of the gap to the peak (the hand-written f64 GEMM at 13–24% of the H100's f64 peak);
the per-chunk row walk of `Accumulate::Rows`, which grows with the number of chunks (the
M2L efficiency drops from 23% to 16% between N = 10⁶ and 10⁷ at f64 p = 18); and L2P at
high p. They belong to the same kernels and are folded into this task, each adopted only
by measurement. `docs/design/optimisation.md` §6 (T1) states which are in scope and their
expected gains.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/design/optimisation.md §6;
- docs/design/device-path.md §6.4–§6.6, §10 (the spike's model and break-even
  efficiencies), §13, §17, §18.1–§18.2 (CUDA facts F25–F36, layouts, the chunk budget and
  `Accumulate::Rows`), §18.5;
- `fmm-bench/results/phase4s-gh200.md` and `fmm-validate/results/phase4-m3max.md` (the
  kernel efficiencies per level);
- spikes/cubecl-gemm/SPIKE_REPORT.md;
- the code: `nd_fmm_kernels::translate` (`grouped`, `grouped_stage`, `Gemm`,
  `GemmLayout`, `Accumulate`, `TileSchedule`), `movement`, `leaf` (L2P), `view`;
  `nd_fmm_exec::tune` (candidates, `CANDIDATE_SET_VERSION`) and `device`;
  `nd-fmm-validate`'s `translation_kernels`, `m2l_kernels`, `leaf_kernels` and
  `layout_sweep` examples.

Do:
- **The fused gather**: a GEMM variant that loads its input columns through the batch's
  source index list, for the hand-written kernel (and the library path where CubeCL
  allows it), with the same accumulation order, so the output is bit for bit the unfused
  path's. Kept beside the unfused path, chosen per (backend, precision, p) by
  measurement and offered to the tuner.
- **A shared-memory-tiled GEMM** for the hand-written kernel, if T1 keeps it: tiles of the
  table and the columns in shared memory, the accumulation order fixed so that the bits
  do not depend on the tile size, or the new order documented with its bound.
- **`Accumulate::Rows`**: a reduction whose cost does not grow with the number of chunks
  (for example, rows split by chunk at schedule time), bit for bit the current order.
- **L2P at high p**, if T1 keeps it.
- **Measurements**: per level and end to end on CUDA f32/f64 (locust) and Metal f32 (the
  M3 Max), N = 10⁶ and 10⁷, p = 3–18, against the Phase 4S numbers; the efficiency against
  the peaks of `nd_fmm_validate::peaks`. Report in the Phase 6 results file.

Tests that define done:
- Each new kernel against its existing counterpart on the CPU runtime, f32 and f64, bit
  for bit (or the documented bound), at the kernel tests' shapes and p = 3–20.
- The device FMM on every `tests/mpi_exec.rs` scenario as today (device against host
  within Phase 4's bounds), with each variant forced on.
- Metal and CUDA by hand, as in Phase 4 and 4S.

Must pass: the root checks, the stricter workspace checks, the `nd-fmm-kernels` checks of
the root CLAUDE.md; on locust the CUDA checks; Metal outside the sandbox.

Do not: change a default layout without the measurement that justifies it; use the
library matmul with an input precision below T (Phase 4's guard); assert a timing.
