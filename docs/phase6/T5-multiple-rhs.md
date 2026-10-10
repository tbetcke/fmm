# Phase 6 / T5 — several right-hand sides (C6.3)

`Fmm::evaluate` takes one charge vector. Applications (and our own accuracy runs, which
evaluate eight vectors one after another) often need several with the same points. The
tree, the lists, the tables and the exchanges' structure are the same for all of them, so
carrying m vectors at once turns every matrix-vector product of the far field into a
matrix-matrix product with m columns (laplace-fmm-plan §7, C6.3: "charge vectors as extra
GEMM columns"), and P2P reads each source's position once for m charges. This task adds
it to the host path (through T2's batched layout) and the device path, as
`docs/design/optimisation.md` §4 designs it.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-plan/CLAUDE.md, fmm-kernels/CLAUDE.md,
  fmm-simd/CLAUDE.md;
- docs/design/optimisation.md §4; docs/design/laplace-fmm-plan.md §5.3 (data layout);
- docs/design/fmm-plan-redesign.md §5 (stores) and §7 (the exchanges);
  docs/design/distributed-fmm.md §4 (the redistribution's forward and backward) and §6
  (the hook);
- the code: `nd_fmm_exec::fmm` (`Fmm::evaluate`, `evaluate_into`, `Output`, the forward
  of the charges and the backward of the output), `nd_fmm_exec::operator`, T2's batched
  level call, `nd_fmm_plan::{store, exchange, redistribute}`, `nd_fmm_simd` P2P kernels,
  `nd_fmm_exec::device`.

Do:
- **The interface**: as T1 signs it off, for example `Fmm::evaluate_many(&[&[T]])` or a
  column-major charge matrix, with an `_into` variant; the single-vector `evaluate`
  unchanged and its bits unchanged.
- **Through the pass**: multipole and local stores with m coefficient columns per box (or
  m stores, as T1 decides); the source exchange, the multipole exchange (and Phase 5S's
  top-tree reduction) carrying m columns in one message per neighbour; the redistribution
  forwarding m charges per point and returning m outputs.
- **The operators**: the host batched M2L (T2) and M2M/L2L with m columns; the leaf
  operators and P2P over m charges per source (in `nd-fmm-simd` if T1 finds the kernel
  should take several charges; its `unsafe` stays in `arch`); the device level calls with
  m columns (the GEMM's column count grows by m).
- **Determinism**: each vector's output equals its single-vector evaluation bit for bit
  if T1's design keeps the per-vector order (expected for every product whose columns are
  independent); otherwise within T1's bound.
- **Measurements**: throughput per right-hand side against m = 1, 2, 4, 8, 16, 32 on the
  M3 Max (host and Metal), locust (host and CUDA) and, if Kathleen is usable, one
  Kathleen node, N = 10⁶, f64 p = 3, 8 and f32 p = 3, 8; and at 2 Kathleen nodes (at most
  4) for the exchanges' message sizes, else the message sizes counted at 72 ranks on
  locust. Report
  in the Phase 6 results file.

Tests that define done:
- For m = 1, 3 and 8: each vector's output against its single-vector evaluation (bit for
  bit, or T1's bound), host and CPU runtime, on every `tests/mpi_exec.rs` scenario at 1,
  2 and 4 ranks, with overlap off and on.
- The C5.1 host gate with m = 8 at 2 and 4 ranks.
- Transfers per evaluation on the device as the extended formula of device-path.md §14
  (the counters).

Must pass: the root checks, the stricter workspace checks, `.github/scripts/run-mpi-tests.sh`
at 2 and 4 ranks, the `nd-fmm-kernels` and `nd-fmm-simd` checks of the root CLAUDE.md
(the other architecture's clippy for `nd-fmm-simd`), Metal and CUDA by hand.

Do not: change the single-vector path's output; assert a timing.
