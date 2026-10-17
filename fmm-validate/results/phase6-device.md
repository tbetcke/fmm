# Phase 6 T7: the fused gather and the device GEMM work (C6.1)

Phase 6, task T7 ([docs/phase6/T7-fused-gather.md](../../docs/phase6/T7-fused-gather.md));
design [docs/design/optimisation.md](../../docs/design/optimisation.md) §6, signed off on
2026-10-15 (§6.1 records what was built). Measured on 2026-10-16 from branch
`tbetcke/phase6_t7` (main at 6cd3425 plus this task's changes), release builds, rustc
1.99.0, CubeCL 0.11.0, every BLAS thread variable at 1. The M3 Max ran Metal f32
(unsandboxed); locust's H100 ran CUDA f32 and f64 (§4). No Kathleen job: the task has no
x86 or multi-node part.

## Summary

1. **Bit for bit.** The fused gather, the reduction by chunk and the shared-memory GEMM
   each keep Phase 4's order of additions: every kernel test compares bits on every layout
   and chunking (CPU runtime, Metal, CUDA), every level call measured (604) gave Phase 4S's
   bits, and in every device scenario each variant forced on gives the hand-written device
   evaluation's bits (720–840 evaluations per run on the CPU runtime at 1, 2 and 4 ranks,
   88 on Metal, 192 on CUDA) (§2).
2. **Metal f32** (§3): per level call 0.51–0.56 of Phase 4S's time at N = 10⁶, 0.18–0.26 at
   10⁷; end to end **1.13–1.65× at N = 10⁶ and 1.29–3.97× at 10⁷** (p = 3–8). The largest
   part is the reduction by chunk wherever the 128 MB budget makes many chunks (at 10⁷,
   p = 8, the accumulation stage 1,865 → 92 ms); then the fused gather (0.62–0.90 of the
   separate gather) and the shared-memory GEMM (0.78–0.91 of the default layout to p = 9).
3. **CUDA f32 and f64** (§4, beside an idle co-tenant under a sampling gate, the Phase 4S
   baseline reproduced within 0.6 %): per level call 0.60–0.77 at N = 10⁶ and 0.49–0.66 at
   10⁷, up to 33–34 % of the f64 peak at p = 18; the best variant alternates with p. End to
   end 1.0–1.17× in f32 and up to **1.65× in f64** (p = 18, N = 10⁷); the host's part of an
   evaluation (15–34 ms at N = 10⁶, 317–500 ms at 10⁷) now dominates there, also in the
   Phase 4S build, and is reported for T11.
4. **Defaults** (§5; signed off on 2026-10-17, docs/phase6/README.md decision 8; no output
   changes): the reduction by chunk everywhere; on Metal the fused gather and the
   shared-memory GEMM to p = 9; on CUDA Phase 4S's gather and layout until a sweep of every
   p (T11), the tuner offering both.
5. **Several charge vectors** (§6; T5's revision): with the reduction by chunk T5's
   per-vector chunk budget is no longer needed against the m² cost. Metal caps it at 8
   vectors (3.7 % per vector slower than T5's at m = 32, a quarter of the scratch), CUDA uses
   one vector's 2 GB (at most 3 % slower; T5's 10–12 GB failed to allocate beside the
   co-tenant, T5's own failure).
6. **Not done**: L2P at high p (T1: only if simple; it is not); the CUDA sweep of every p
   and the f64 several-vector runs (the H100 busy again from 13:12); the host part on CUDA.
   A regression found and fixed on the way: a multi-unit shared-memory cube on the CPU
   runtime spun in `sync_cube` with several ranks per node (`mpi_exec` 140× slower at 4
   ranks); the CPU runtime's default is now one unit (§7).

## 1. What T7 adds

Three settings of `nd_fmm_kernels::translate::PlanSettings`, each a change of the
hand-written path that keeps one accumulator per output, `fma` with k ascending and the
reduction's row order, so **each gives the bits of Phase 4's path** (the module docs of
`translate`, "The fused gather", "The reduction by chunk", "The shared-memory GEMM"):

- **The fused gather** (`Gather::Fused`; device-path.md §6.4, C6.1): the GEMM reads chunk
  column c straight from the multipole (or local) store at the batch's source index,
  `batch_sources[start + c]`, instead of from the gathered scratch X; no gather launch, two
  launches per chunk instead of three, and no X: the chunk budget holds the products Y
  alone, so a chunk has twice the columns, and the device's scratch holds X only for plans
  that gather. Each unit looks its columns' sources up once per block. The library GEMM
  reads strided tensors and cannot follow an index list, so a library plan keeps the
  separate gather; the coefficient-major layout (gathered by blocks) refuses it.
- **The reduction by chunk** (`Reduction::ByChunk`, the default; device-path.md §18.2,
  "findings reported, not built"): Phase 4's reduction walked every entry of every row in
  every chunk and added those in the chunk, so its cost was the entries times the chunks.
  Along a row the batch positions ascend, so a row's entries in a chunk follow one another;
  the plan lists at build, per chunk, the rows it touches with the first entry of each in
  it (two `u32` per (row, chunk), at most the entries and at most the rows times the
  chunks), and the launch walks each listed row from there while the entries lie in the
  chunk. Every entry is visited once per level call whatever the chunks. Phase 4's walk
  stays selectable (`Reduction::Walk`), as the counterpart and for measurement.
- **The shared-memory GEMM** (`GemmLayout::Shared { rows, columns, per_unit, depth }`;
  device-path.md §18.2, optimisation.md §6, "as a candidate"): per step of `depth` k the
  cube stages the tile's input values and the table's rows (padded to `rows · ⌈n /
  rows⌉`) in shared memory, syncs, and each unit adds the step's products into its
  accumulators in k order. `GemmLayout::shared_for` gives a default per device (256 units,
  rows the power of two at or above n / 8 between 16 and 32, 4 columns per unit, the
  deepest step of 16 or fewer that fits the shared memory); on the CPU runtime one unit,
  for correctness only (its `sync_cube` spins: §7).

In `nd-fmm-exec`: `FmmBuilder::device_gather` (`DeviceGather::{Auto, Separate, Fused}`),
`device_reduction` (`DeviceReduction::{ByChunk, Walk}`) and `DeviceGemm::Shared`;
`TranslationReport::{gather, reduction, launches}`; the static rule's layout and gather by
backend, precision and p (`tune::static_layout`, `tune::static_gather`, §5); the tuner's
candidates gain the shared-memory layout and the other gather (`CANDIDATE_SET_VERSION` 3;
the cache's `FORMAT_VERSION` 2 names both). In `nd-fmm-validate`: the `gemm_variants`
example (every level call under every variant, with the bits compared) and `scaling`'s
`--device-scratch`, `--device-gemm`, `--device-gather` and `--device-reduction`; in
`tools/scaling/run.sh` the sweep `t7-device`. In `nd-fmm-bench`: `--device-gemm`,
`--gather`, `--reduction` and `--scratch`, recorded in the header's "device translations".

**L2P at high p** (T1: "only if simple", at most 5 % of an evaluation) was not built: its
cost at p ≥ 12 is each unit's (p + 1)² harmonics array held per target (device-path.md
§18.2), and removing it means restructuring the leaf kernel's recursion and its order of
accumulation, which is not a simple change; it stays a finding for a later task.

## 2. Correctness

Every new path keeps the order of additions, so every test compares bits:

- **The kernels** (`nd-fmm-kernels`, `tests/kernels/translate.rs` and `translate/m2l.rs`):
  the GEMM at p ∈ {0, 1, 3, 8, 12, 20} and k ∈ {0, 1, 7, 64} on every layout (the
  shared-memory layout added: on the CPU runtime 4 × 2 units with a step of 5 k, so that
  most orders end on a partial step; on a GPU `shared_for`'s default and 16 × 2 units with a
  step of 5) bit for bit the host `mul_add` loop, and the fused gather (`gemm_gathered`,
  the columns at permuted sources of a wider input) bit for bit the gathered GEMM; every
  level call (M2M- and L2L-shaped views at p ∈ {0, 3, 8} on every layout and at p = 12 and
  20 in the default one, the 316-offset M2L at p ∈ {0, 3, 8}) on every layout, box-major and coefficient-major (not for the shared layout), with
  the fused gather (box-major) and with both reductions, under budgets of one chunk, chunks
  that cut batches and chunks of one column, bit for bit the host rows, with the launches
  per chunk (three, two fused); structure (B) against (A) on the shared layout too; the
  plan's facts (its gather, its columns of X (none fused), its reduction, one (row, chunk)
  pair per row with an entry and at most one per entry, one per entry with one-column
  chunks, the uploads). Unit tests:
  `chunk_rows` on a hand-made view at three chunkings, the shared layout's shared-memory
  fit and `PlanSettings::check`'s refusals. **Ran**: the CPU runtime in f32 and f64 on the
  M3 Max (75 default tests in 38 s; the ignored 1000-column sweep too), Metal in f32 (55
  tests, unsandboxed, 24.5 s) and CUDA in f32 and f64 on locust (every CUDA test and the
  CPU runtime's on Grace, `--include-ignored`).
- **The device FMM** (`nd-fmm-exec`, `tests/device_common::check_variants`, in every device
  scenario of `tests/mpi_exec.rs` (the CPU runtime), `tests/device_metal.rs` and
  `tests/device_cuda.rs`): the hand-written GEMM with the separate gather and the reduction
  by chunk is the reference; four builds (the fused gather; the walking reduction; the
  shared-memory GEMM; the shared-memory GEMM with the fused gather and the walking
  reduction) each evaluate twice and give its bits, report their gather, reduction and
  layout on every level call, and move, launch and sync as the formula says (two launches
  per chunk with the fused gather). **Ran**: on the CPU runtime 95 scenarios with 720
  variant evaluations at one rank (release; 62 s), 840 at 2 and at 4 ranks, on Metal 11 scenarios with 88, on CUDA
  24 scenarios with 192 (f32 and f64), all bit for bit; every scenario's default device
  placement within the Phase 4 bounds of the host as before (largest relative L2 of φ
  6.2e-7 in f32 and 1.4e-15 in f64 on the CPU runtime). The default placement on Metal
  now runs the static rule's shared-memory GEMM with the fused gather (§5), so
  `tests/device_metal.rs` checked it, and the C4.8 gate on Metal (`tests/device_fmm.rs`)
  passed with it.
- **The tuner** (`tune` unit tests): the static rule by backend, precision and p (the
  shared layout to p = 9 and the fused gather on Metal), the candidates (the shared layout
  and both gathers for the default and the shared layouts, the library's gather separate),
  and the cache encodings round trip with the new fields.
- **Several charge vectors**: `check_three_vectors` in every device scenario (each vector
  bit for bit its one-vector device evaluation) runs under the new defaults; the t7-device
  sweep's runs (§6) evaluate m vectors under both reductions and both budgets.

## 3. Metal (the M3 Max, f32)

### 3.1 Per level call

`gemm_variants --device metal`, the uniform cube at N = 10⁶ and 10⁷ and the Plummer sphere
at 10⁶ (the C3.3 tree settings: `max_level` 16, 64 points per leaf), every M2L, M2M and
L2L level call with a pair, the device's chunk budget (128 MB), the hand-written GEMM in
every column (on Metal, M2M and L2L take the library at p ≥ 8 under `DeviceGemm::Auto`;
this table compares the hand-written paths). The columns: `phase4s` (the default layout,
the separate gather, the walking reduction: Phase 4S's code), `by-chunk`, `fused` (with
the reduction by chunk; its chunks twice as wide, no gathered scratch), `shared` and
`shared-fused` (`GemmLayout::shared_for`: 16 × 16 units, 4 columns per unit, depth 16 at
p = 3–8). Every level call of every variant gave the bits of `phase4s` (203 calls compared,
p = 3–12). Summed over the level calls, ms (the best in bold):

| problem | N | p | kinds | `phase4s` | `by-chunk` | `fused` | `shared` | `shared-fused` | best / `phase4s` | the largest V level: chunks, `phase4s` → the best (µs), GFLOP/s of the best (% of 14,300) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 10⁶ | 3 | all | 5.49 | 4.91 | 3.55 | 4.67 | **3.06** | 0.56 | level 5, 6 chunks: 4,784 → 2,669, 1,036 (7.2 %) |
| cube | 10⁶ | 6 | all | 32.47 | 25.38 | 20.83 | 21.39 | **16.42** | 0.51 | level 5, 16 chunks: 29,275 → 14,334, 1,809 (12.6 %) |
| cube | 10⁶ | 8 | all | 72.38 | 50.01 | 43.69 | 46.73 | **37.83** | 0.52 | level 5, 27 chunks: 66,302 → 33,194, 2,134 (14.9 %) |
| cube | 10⁶ | 9 | M2L | 99.99 | 67.48 | 59.03 | 64.03 | **51.25** | 0.51 | level 5, 33 chunks: 92,680 → 45,810, 2,357 (16.5 %) |
| cube | 10⁶ | 10 | M2L | 130.02 | 83.26 | **72.30** | 95.57 | 80.77 | 0.56 | level 5, 39 chunks: 120,971 → 64,559 (`fused`), 2,449 (17.1 %) |
| cube | 10⁶ | 11 | M2L | 185.90 | 123.64 | 111.13 | 117.02 | **100.83** | 0.54 | level 5, 47 chunks: 173,101 → 90,163, 2,483 (17.4 %) |
| cube | 10⁶ | 12 | all | 256.56 | 160.79 | **142.30** | 193.61 | 171.70 | 0.55 | level 5, 55 chunks: 236,344 → 124,606 (`fused`), 2,475 (17.3 %) |
| cube | 10⁷ | 3 | all | 108.36 | 51.00 | 31.67 | 45.53 | **28.63** | 0.26 | level 6, 45 chunks: 102,078 → 25,292, 937 (6.6 %) |
| cube | 10⁷ | 6 | all | 869.27 | 275.30 | 201.71 | 242.09 | **171.66** | 0.20 | level 6, 136 chunks: 834,011 → 153,312, 1,450 (10.1 %) |
| cube | 10⁷ | 8 | all | 2,276.28 | 571.18 | 445.87 | 525.93 | **400.90** | 0.18 | level 6, 224 chunks: 2,198,510 → 358,725, 1,694 (11.8 %) |
| Plummer | 10⁶ | 3 | all | 8.25 | 7.82 | 5.69 | 7.43 | **4.82** | 0.58 | level 8, 4 chunks: 3,217 → 1,866, 1,042 (7.3 %) |
| Plummer | 10⁶ | 6 | all | 48.11 | 44.62 | 36.21 | 43.70 | **28.24** | 0.59 | level 8, 12 chunks: 20,671 → 11,415, 1,598 (11.2 %) |
| Plummer | 10⁶ | 8 | all | 99.02 | 97.44 | 82.53 | 90.51 | **65.24** | 0.66 | level 8, 19 chunks: 42,968 → 25,066, 1,988 (13.9 %) |

(2026-10-16 23:58 to 2026-10-17 01:25, each table started on a quiet machine: the CPU 88–95 %
idle over 3 s, no other process above half a core, the GPU idle; `.state` files in §8. A
first run of the same tables, before the fused gather's chunks were widened, was disturbed
between 23:33 and 23:44 by background work on the GPU, which that gate did not check; it is
not used. At p = 9–11 only the M2L level calls ran.)

What each item contributes, read off the stages (`grouped_stage`, summed over the level
calls; the walking reduction under `phase4s`, the reduction by chunk under `fused`):

- **The reduction by chunk** is the largest gain wherever a level call has many chunks
  (under Metal's 128 MB budget the cube's largest V level has 6–55 chunks at N = 10⁶ and
  45–224 at 10⁷): the accumulation stage falls from 28.2 to 5.0 ms at N = 10⁶, p = 8, and
  from 1,865 to 92 ms at 10⁷, p = 8 (20×); with few chunks it changes little (1.6 →
  1.1 ms at N = 10⁶, p = 3). `by-chunk` alone takes 0.63–0.98 of `phase4s` at N = 10⁶ (the
  Plummer sphere's largest levels have fewer chunks) and 0.25–0.47 at 10⁷.
- **The fused gather** removes the gather stage (1.1, 3.4, 5.2 and 11.1 ms at N = 10⁶,
  p = 3, 6, 8, 12) without making the GEMM slower: its GEMM stage, which reads the input
  through the source index, takes 0.83–1.03 of the separate GEMM's. With no gathered
  scratch its chunks are twice as wide, which also halves the reduction's per-chunk work.
  Summed over the calls it takes 0.62–0.90 of the separate gather's time in the default
  layout and 0.63–0.89 in the shared-memory one, and no level call of any problem was
  slower with it.
- **The shared-memory GEMM** with the fused gather takes 0.78–0.91 of the default layout's
  time with it at p = 3–9 and 11 on every problem, and 1.12 and 1.21 at p = 10 and 12
  (where `shared_for` gives 8 and 6 rows per unit, and the default cube layout's register
  blocks fit better). Without the fused gather it gains less (0.84–0.98 of `by-chunk` at
  p ≤ 9). On small level calls (≤ 4,096 pairs) its 256-unit cubes are up to 0.18 ms slower
  than the default layout; summed over the level calls they are a few per cent.

The largest level call's efficiency (useful flops over the whole call, gather and reduction
included) rises from 4.0–9.1 % of the 14.3 TFLOP/s f32 peak under `phase4s` to 7.2–17.4 %
at N = 10⁶, and from 1.6–1.9 % to 6.6–11.8 % at 10⁷, where Phase 4S's reduction
dominated.

### 3.2 End to end

`nd-fmm-bench --backend metal --precision f32 --degree 3,6,8 --n 1e6,1e7` (the uniform
cube, gradients, 2 warm-ups and 10 timed evaluations, the medians; the kind times from a
second build with synchronous kind timings), five builds of one binary with the layout named
(`DeviceGemm::Auto` takes the shared-memory GEMM on Metal now, §5):

| column | flags | the translations |
| --- | --- | --- |
| Phase 4S | `--device-gemm hand --gather separate --reduction walk` | Phase 4S's kernels (at p = 8 M2M and L2L on the hand-written kernel, where Phase 4S's `Auto` took the library: 4.2 against 4.5 ms per evaluation, below) |
| by chunk | `--device-gemm hand --gather separate` | + the reduction by chunk |
| + fused | `--device-gemm hand --gather fused` | + the fused gather |
| default | none | the new defaults: the shared-memory GEMM to p = 9 and the fused gather for M2L, the library for M2M and L2L at p = 8 |
| shared everywhere | `--device-gemm shared --gather fused` | the shared-memory GEMM for M2M and L2L too |

| N | p | Phase 4S ms | by chunk | + fused | **default** | shared everywhere | speed-up of the default | M2L (kind) ms: Phase 4S → default | M2M + L2L ms: Phase 4S → default | error φ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| 10⁶ | 3 | 18.4 | 18.2 | 17.0 | **16.2** | 16.2 | 1.13 | 6.7 → 4.3 | 3.48 → 3.46 | 2.00e-3 |
| 10⁶ | 6 | 45.6 | 38.7 | 34.3 | **30.3** | 30.4 | 1.50 | 32.7 → 17.4 | 3.89 → 3.85 | 8.53e-5 |
| 10⁶ | 8 | 85.6 | 63.7 | 57.4 | **51.9** | 52.0 | 1.65 | 71.9 → 38.0 | 4.22 → 4.45 | 1.36e-5 |
| 10⁷ | 3 | 348.2 | 289.2 | 272.4 | **270.9** | 270.8 | 1.29 | 115.4 → 35.4 | 6.74 → 6.15 | 1.87e-3 |
| 10⁷ | 6 | 1,115.1 | 519.3 | 438.9 | **414.7** | 418.5 | 2.69 | 869.4 → 172.0 | 12.00 → 8.03 | 7.90e-5 |
| 10⁷ | 8 | 2,528.5 | 790.6 | 664.9 | **637.5** | 638.0 | 3.97 | 2,271.6 → 383.4 | 13.60 → 11.10 | 1.21e-5 |

(2026-10-17 02:39–02:55, each build started on a quiet machine as in §3.1.) The errors are
the same in every build, as the bits are (§2). The shared-memory GEMM for M2M and L2L
changes nothing measurable, so the library stays their default at p = 8. At N = 10⁷ the
rest of an evaluation is now the larger part: P2P 57 ms and the host's part outside the
level calls ("other" of the kind table) 159–163 ms, which T7 does not touch (device-path.md
§18.2, the per-evaluation host work).

## 4. CUDA (locust's H100, f32 and f64)

Measured on locust's H100 (GH200, driver 565.57.01, CUDA 12.6, CubeCL 0.11.0 through LLVM
NVPTX) on 2026-10-17 from 09:44. Another user's vLLM server held 79.6 GB of the GPU
throughout. It had kept the GPU busy (100 %) from the evening before until about 03:00 and
was idle (0 %) from then on, and with the user's agreement (2026-10-17) the runs went ahead
beside it under a gate (§8, `cuda_measure.sh`): each run started after ten minutes in which
no foreign process used the SMs or the memory bandwidth and the GPU stayed below 5 % (one
sample and 30 s after a clean run), and a sampler read every process's SM and memory use
(`nvidia-smi pmon`) every 2 s during the run; a run in which a foreign process worked, or
changed its memory (a new process included), was void and repeated. Every run reported here
had no such sample. **Sanity check** first: Phase 4S's settings (`--device-gemm hand
--gather separate --reduction walk`) reproduced Phase 4S's M2L kind time (f32, N = 10⁶,
p = 8, dense, synchronous kinds: 9.43 ms against 9.37 ms in phase4s-gh200.md, +0.6 %). The
host load average was below 1.5 throughout.

### 4.1 Per level call

`gemm_variants --device cuda` as in §3.1, the CUDA defaults (the GEMM with the rows the
power of two at or above n / 4, 64 units, 8 columns per unit; `shared_for`: 256 units, 16 to
32 rows, 4 columns per unit, depth 16), the 2 GB chunk budget. Every level call of every
variant gave the bits of `phase4s` (401 calls compared). Summed over the level calls, ms
(the best in bold):

**f32** (peak 67 TFLOP/s):

| problem | N | p | kinds | `phase4s` | `by-chunk` | `fused` | `shared` | `shared-fused` | best / `phase4s` | the largest V level: chunks, `phase4s` → the best (µs), GFLOP/s of the best (% of 67,000) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 10⁶ | 3 | f32 | 1.34 | 1.31 | **0.97** | 1.61 | 1.33 | 0.72 | level 5, 1 chunks: 897 → 648 (`fused`), 4,269 (6.4 %) |
| cube | 10⁶ | 6 | f32 | 4.29 | 4.28 | 5.89 | 4.04 | **3.23** | 0.75 | level 5, 1 chunks: 3,471 → 2,640, 9,822 (14.7 %) |
| cube | 10⁶ | 8 | f32 | 9.68 | 9.56 | 9.03 | 7.44 | **5.80** | 0.60 | level 5, 2 chunks: 8,159 → 4,854, 14,596 (21.8 %) |
| Plummer | 10⁶ | 3 | f32 | 2.32 | 2.25 | **1.69** | 2.73 | 2.23 | 0.73 | level 8, 1 chunks: 690 → 502 (`fused`), 3,876 (5.8 %) |
| Plummer | 10⁶ | 6 | f32 | 7.04 | 7.00 | 9.52 | 6.55 | **5.29** | 0.75 | level 8, 1 chunks: 2,458 → 1,886, 9,673 (14.4 %) |
| Plummer | 10⁶ | 8 | f32 | 15.29 | 15.14 | 14.95 | 11.70 | **9.30** | 0.61 | level 8, 2 chunks: 5,788 → 3,443, 14,477 (21.6 %) |
| cube | 10⁷ | 3 | f32 | 9.92 | 9.13 | **6.55** | 11.76 | 9.91 | 0.66 | level 6, 3 chunks: 8,410 → 5,478 (`fused`), 4,328 (6.5 %) |
| cube | 10⁷ | 6 | f32 | 42.51 | 36.22 | 39.46 | 34.92 | **26.28** | 0.62 | level 6, 9 chunks: 37,668 → 22,683, 9,801 (14.6 %) |
| cube | 10⁷ | 8 | f32 | 97.80 | 82.10 | 71.02 | 65.29 | **48.89** | 0.50 | level 6, 14 chunks: 87,014 → 42,458, 14,309 (21.4 %) |

**f64** (peak 34 TFLOP/s):

| problem | N | p | kinds | `phase4s` | `by-chunk` | `fused` | `shared` | `shared-fused` | best / `phase4s` | the largest V level: chunks, `phase4s` → the best (µs), GFLOP/s of the best (% of 34,000) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cube | 10⁶ | 3 | f64 | 1.74 | 1.71 | **1.24** | 2.68 | 2.36 | 0.71 | level 5, 1 chunks: 1,239 → 862 (`fused`), 3,209 (9.4 %) |
| cube | 10⁶ | 6 | f64 | 6.42 | 6.36 | 6.38 | 6.10 | **4.95** | 0.77 | level 5, 2 chunks: 5,380 → 4,143, 6,258 (18.4 %) |
| cube | 10⁶ | 8 | f64 | 15.72 | 15.22 | 10.53 | 12.14 | **10.27** | 0.65 | level 5, 4 chunks: 13,479 → 8,751, 8,096 (23.8 %) |
| cube | 10⁶ | 12 | f64 | 53.11 | 51.72 | **40.07** | 46.55 | 44.20 | 0.75 | level 5, 7 chunks: 46,266 → 34,966 (`fused`), 8,820 (25.9 %) |
| cube | 10⁶ | 18 | f64 | 208.67 | 200.72 | 165.61 | 147.22 | **139.31** | 0.67 | level 5, 15 chunks: 183,815 → 122,818, 11,457 (33.7 %) |
| Plummer | 10⁶ | 3 | f64 | 2.99 | 2.93 | **2.13** | 4.43 | 3.86 | 0.71 | level 8, 1 chunks: 930 → 666 (`fused`), 2,919 (8.6 %) |
| Plummer | 10⁶ | 8 | f64 | 24.01 | 23.80 | 16.89 | 18.85 | **16.00** | 0.67 | level 8, 3 chunks: 9,270 → 6,132, 8,128 (23.9 %) |
| Plummer | 10⁶ | 12 | f64 | 82.07 | 80.74 | **60.52** | 72.36 | 66.35 | 0.74 | level 8, 5 chunks: 32,166 → 24,305 (`fused`), 8,927 (26.3 %) |
| cube | 10⁷ | 3 | f64 | 14.46 | 12.81 | **8.64** | 21.81 | 19.69 | 0.60 | level 6, 6 chunks: 12,492 → 7,246 (`fused`), 3,271 (9.6 %) |
| cube | 10⁷ | 8 | f64 | 166.66 | 134.26 | 96.34 | 109.60 | **90.05** | 0.54 | level 6, 28 chunks: 149,397 → 78,711, 7,718 (22.7 %) |
| cube | 10⁷ | 12 | f64 | 603.61 | 454.22 | **355.18** | 412.69 | 392.29 | 0.59 | level 6, 59 chunks: 545,547 → 311,354 (`fused`), 8,494 (25.0 %) |
| cube | 10⁷ | 18 | f64 | 2,492.64 | 1,776.39 | 1,460.29 | 1,318.64 | **1,228.29** | 0.49 | level 6, 125 chunks: 2,265,805 → 1,076,076, 11,214 (33.0 %) |

What differs from Metal:

- **The reduction by chunk** matters less at N = 10⁶: the 2 GB budget gives the largest
  level 1–4 chunks at p ≤ 8 (Phase 4S T7 chose it so), so `by-chunk` alone takes
  0.96–1.00 of `phase4s` there; at 10⁷ (3–125 chunks) 0.71–0.92, and the accumulation stage
  at f64 p = 18 falls from 820 to 78 ms (10×).
- **The fused gather in the default cube layout** depends on p and the precision: its GEMM
  stage (reading each unit's 8 columns from scattered sources) takes 0.77–0.95 of the
  separate GEMM's in f64 at p = 3, 8, 12 and 18 and 1.01–1.19 in f32 at p = 3 and 8, but
  1.29 (f64) and 1.59–1.94 (f32) at p = 6 (n = 49). Summed over the calls it is the best
  variant at p = 3 and 12, and at p = 6 1.36–1.38× *slower* than the separate gather in
  f32 at N = 10⁶ (1.09× at 10⁷) and level with it in f64.
- **The shared-memory GEMM with the fused gather** is the best at p = 6, 8 and 18 (its
  staged loads read each column's k contiguously): in f64 at p = 18 0.67 and 0.49 of
  `phase4s` at N = 10⁶ and 10⁷, 33.0–33.7 % of the f64 peak on the largest level against
  27.7–28.2 % with the fused cube layout and 15.7–22.5 % under `phase4s`; at p = 3 and 12
  it is slower than the fused cube layout.

The winner thus changes with p; §5 sets the rule from a sweep of every p.

### 4.2 End to end

`nd-fmm-bench --backend cuda` (the uniform cube, gradients, 2 warm-ups and 10 timed
evaluations, the medians; synchronous kind timings in a second build), five builds with the
layout named: Phase 4S's (`--device-gemm hand --gather separate --reduction walk`; on CUDA
Phase 4S's `Auto` also took the hand-written kernel), by chunk, + fused (cube layout), the
shared-memory GEMM with the separate gather, and with the fused gather. ms:

| precision | N | p | Phase 4S | by chunk | + fused (cube) | shared-memory | shared-memory, fused | best / Phase 4S | M2L (kind): Phase 4S → best | other (Phase 4S build) | error φ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| f32 | 10⁶ | 3 | 21.3 | 21.1 | 21.3 | **21.1** | 22.1 | 0.99 | 1.3 → 1.6 | 16.1 | 2.00e-3 |
| f32 | 10⁶ | 6 | 24.1 | 25.0 | 26.4 | 24.0 | **23.3** | 0.97 | 4.2 → 3.2 | 15.2 | 8.53e-5 |
| f32 | 10⁶ | 8 | 30.8 | 29.9 | 29.5 | 28.5 | **26.3** | 0.85 | 9.4 → 5.7 | 15.5 | 1.36e-5 |
| f32 | 10⁷ | 3 | 368.1 | 361.5 | 375.8 | 377.1 | **360.4** | 0.98 | 9.7 → 9.8 | 320.6 | 1.87e-3 |
| f32 | 10⁷ | 6 | 407.0 | **384.0** | 404.5 | 390.7 | 387.7 | 0.94 | 42.0 → 35.7 | 324.3 | 7.90e-5 |
| f32 | 10⁷ | 8 | 453.9 | 436.2 | 434.5 | 430.0 | **413.2** | 0.91 | 96.6 → 48.3 | 317.2 | 1.21e-5 |
| f64 | 10⁶ | 3 | 43.7 | **40.7** | 44.4 | 40.7 | 43.9 | 0.93 | 1.7 → 1.7 | 34.2 | 2.00e-3 |
| f64 | 10⁶ | 6 | 45.9 | 45.3 | 45.6 | 48.7 | **44.8** | 0.98 | 6.3 → 4.8 | 30.8 | 8.53e-5 |
| f64 | 10⁶ | 8 | 55.8 | 54.6 | 51.0 | 55.6 | **50.0** | 0.90 | 15.4 → 10.0 | 32.1 | 1.36e-5 |
| f64 | 10⁶ | 12 | 98.9 | 97.0 | **85.8** | 95.2 | 89.7 | 0.87 | 52.1 → 39.4 | 31.4 | 4.35e-7 |
| f64 | 10⁶ | 18 | 267.0 | 258.7 | 223.9 | 208.9 | **195.7** | 0.73 | 205.7 → 136.6 | 32.5 | 6.32e-9 |
| f64 | 10⁷ | 3 | 562.9 | **561.7** | 571.6 | 572.8 | 566.9 | 1.00 | 14.4 → 12.7 | 496.2 | 1.87e-3 |
| f64 | 10⁷ | 6 | 622.4 | 615.6 | 609.6 | 608.6 | **595.1** | 0.96 | 66.9 → 40.9 | 492.9 | 7.90e-5 |
| f64 | 10⁷ | 8 | 728.1 | 689.9 | 666.8 | 674.6 | **650.0** | 0.89 | 165.8 → 88.9 | 497.4 | 1.21e-5 |
| f64 | 10⁷ | 12 | 1,218.7 | 1,075.3 | **981.6** | 1,031.8 | 1,008.7 | 0.81 | 599.2 → 348.2 | 500.1 | 4.40e-7 |
| f64 | 10⁷ | 18 | 3,233.7 | 2,515.1 | 2,191.8 | 2,055.9 | **1,959.9** | 0.61 | 2,473.2 → 1,205.2 | 498.1 | 6.50e-9 |

The errors are the same in every build. The gains end to end are small at low p because
most of an evaluation lies outside the level calls: "other" of the kind table (the host's
part: the charge load, the output, the stages' remainder) is 15–16 ms in f32 and 31–34 ms in
f64 at N = 10⁶, and 317–324 ms and 493–500 ms at 10⁷, against 8.4 and 13.3 ms in Phase 4S
T7 at N = 10⁶ (device-path.md §18.2). It is there in the Phase 4S build of this binary too,
so it is not T7's; it is reported for T11. At f64 p = 18 the M2L kind falls from 206 to
137 ms (N = 10⁶) and from 2,473 to 1,205 ms (10⁷).


## 5. The defaults

What changes under the defaults (`DeviceGemm::Auto`, `DeviceGather::Auto`,
`DeviceReduction::ByChunk`, no named budget), and why. **None of it changes an output bit**
(§2: each variant forced on in every device scenario gives the hand-written device
evaluation's bits), so README requirement 6, whose concern is a changed one-rank output,
does not arise; they are kernel choices of the kind Phase 4S T7 set by measurement, and the
T7 brief asks for them "chosen per (backend, precision, p) by measurement". They were
signed off by the user on 2026-10-17 (docs/phase6/README.md, decision 8), with CUDA's
gather and layout left to T11, which revisits them with the other defaults.

| default | backends | confidence | the measurement behind it |
| --- | --- | --- | --- |
| the reduction by chunk | every backend | high | the same additions in the same order; never more work than the walk (it lists only the rows a chunk touches); no level call measured slower (§3.1); on CUDA not timed yet, where it can only remove work |
| the fused gather | Metal | high | 0.62–0.90 of the separate gather's time summed over the level calls, no level call slower (§3.1); no gathered scratch |
| the several-vector budget capped at 8 vectors | Metal | high | 3.7 % per vector slower than T5's budget at m = 32, f32 p = 8, with 1 GB of scratch instead of 4 GB (§6) |
| the several-vector budget at one vector's (2 GB) | CUDA | high | at most 3 % per vector slower than T5's budget at f32 p = 3 and 8, m ≤ 32, with 2 GB of scratch instead of up to 12 GB (§6); f64 not measured (the H100 was busy again) |
| the shared-memory GEMM to p = 9 | Metal | medium | 0.78–0.90 of the default layout's time at p = 3–9 with the fused gather, on every problem; the crossover is irregular (0.91 at p = 11, 1.12 at p = 10, 1.21 at p = 12), and on small level calls it is slower; T11 should look again |

**CUDA** keeps Phase 4S's gather and layout: the best variant alternates with p (§4.1: the
fused cube layout at p = 3 and 12, the shared-memory GEMM with the fused gather at p = 6, 8
and 18, and the fused cube layout 1.36× slower than the separate gather at f32 p = 6), and
a rule set from five values of p could be slower than today's at the others. A sweep of
every p (f32 2–10, f64 2–20, `cuda_sweep.sh`) was queued but did not run before the H100
was busy again; the rule is left to it and T11. The tuner already offers the fused gather
and the shared-memory layout there, so a build with a tuning cache takes them per level
call. The CPU runtime, the correctness backend, keeps its
layout and the separate gather (the variants run there in every test).

## 6. Several charge vectors: the chunk budget revisited

T5 found the device's M2L quadratic in the charge vectors m: an `Fmm` of m vectors has m
times the entries per level call and, under a fixed chunk budget, m times the chunks, and
Phase 4's reduction walked every entry once per chunk (phase6-rhs.md §3.5). T5 made the
default budget per vector (`device::vector_settings`: m times the bytes, at most an eighth
of the memory), at the price of m times the scratch (4 GB on the M3 Max at m = 32, about
12 GB on the H100). With the reduction by chunk every entry is visited once per level call,
so the brief asks whether the per-vector budget is still needed.

`tools/scaling/run.sh t7-device` (one rank, the N = 10⁶ cube, gradients, `auto` (Dense),
three timed evaluations of `Fmm::evaluate_many`, the wall time over m), each problem under
the reduction by chunk and the walking one, each with T5's per-vector budget (the default)
and with one vector's (`--device-scratch` at the device's default). Milliseconds per
vector, with the scratch the device allocated (`DeviceReport::scratch_bytes`) in MB:

**Metal, f32** (the M3 Max, 2026-10-17 01:39–02:38, every launch on a quiet machine as in
§3.1; the defaults of §5: the shared-memory GEMM to p = 9 and the fused gather for M2L, the
library for M2M and L2L at p = 8, and in the per-vector columns the budget capped at 8
vectors' worth). Milliseconds per vector, with the scratch the device allocated in MB (it
holds the products of the fused plans and the gathered columns of the library's, so it can
exceed the budget):

| precision | p | m | by chunk, per vector | by chunk, one vector | walk, per vector | walk, one vector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 3 | 1 | 16.41 (128) | 17.32 (128) | 16.23 (128) | 16.31 (128) |
| f32 | 3 | 2 | 15.13 (256) | 15.47 (128) | 15.40 (256) | 15.72 (128) |
| f32 | 3 | 4 | 15.01 (512) | 14.82 (128) | 14.87 (512) | 16.36 (128) |
| f32 | 3 | 8 | 17.64 (1,024) | 17.17 (128) | 17.43 (1,024) | 21.10 (128) |
| f32 | 3 | 16 | 16.89 (1,024) | 17.73 (128) | 17.65 (1,024) | 24.25 (128) |
| f32 | 3 | 32 | 16.85 (1,024) | 17.67 (128) | 17.88 (1,024) | 31.47 (128) |
| f32 | 8 | 1 | 51.21 (138) | 51.09 (138) | 59.95 (138) | 61.37 (138) |
| f32 | 8 | 2 | 50.00 (276) | 51.39 (148) | 59.01 (276) | 71.88 (148) |
| f32 | 8 | 4 | 51.27 (552) | 53.20 (168) | 59.82 (552) | 97.61 (168) |
| f32 | 8 | 8 | 54.42 (1,104) | 59.36 (188) | 62.62 (1,104) | 151.78 (188) |
| f32 | 8 | 16 | 54.89 (1,185) | 63.44 (188) | 75.06 (1,185) | 249.97 (188) |
| f32 | 8 | 32 | 55.94 (1,347) | 66.28 (168) | 99.50 (1,347) | 448.83 (168) |

The walking reduction under one vector's budget reproduces T5's finding (8.8× per vector
at m = 32, p = 8; with the walk, the per-vector budget is needed). Under the reduction by
chunk the one-vector budget costs 1.18× per vector at m = 32, p = 8 against the capped
per-vector budget (1.05× at p = 3), with an eighth of its scratch. The budget at f32 p = 8,
by hand under the same gate (ms per vector):

| m | 128 MB | 256 MB | 512 MB | 1,024 MB (the cap) | 4,096 MB (T5's at m = 32) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 59.65 | 57.07 | 54.77 | 54.37 | 53.52 |
| 32 | 65.28 | 62.48 | 58.92 | 55.98 | 53.98 |

So on Metal the per-vector budget is capped at **8 vectors' worth** (1 GB at 128 MB;
`device::METAL_BUDGET_VECTORS`): T5's budget to m = 8, and at m = 32 3.7 % slower per
vector than T5's 4 GB with a quarter of its scratch.

**CUDA, f32** (the H100 under the gate of §4, 2026-10-17 12:05–13:11; the CUDA defaults,
Dense with the default layout and the separate gather; the per-vector budget is T5's, m ×
2 GB up to an eighth of the 96 GB; "–": not run, the H100 busy again from 13:12):

| precision | p | m | by chunk, per vector | by chunk, one vector | walk, per vector | walk, one vector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32 | 3 | 1 | 21.81 (659) | 21.74 (659) | 23.39 (659) | 22.57 (659) |
| f32 | 3 | 2 | 23.19 (1,318) | 22.93 (1,318) | 23.74 (1,318) | 22.42 (1,318) |
| f32 | 3 | 4 | 15.66 (2,636) | 15.63 (2,048) | 15.70 (2,636) | 17.81 (2,048) |
| f32 | 3 | 8 | 13.82 (5,272) | 13.44 (2,048) | 15.02 (5,272) | 14.97 (2,048) |
| f32 | 3 | 16 | failed (10.5 GB) | 14.46 (2,048) | 14.28 (10,544) | 14.40 (2,048) |
| f32 | 3 | 32 | failed (12.2 GB) | 12.89 (2,048) | 12.98 (12,160) | 13.34 (2,048) |
| f32 | 8 | 1 | 30.35 (2,047) | 29.36 (2,047) | 29.88 (2,047) | 30.62 (2,047) |
| f32 | 8 | 2 | 31.03 (4,095) | 30.36 (2,047) | 31.84 (4,095) | 37.51 (2,047) |
| f32 | 8 | 4 | 24.19 (8,191) | 24.78 (2,047) | 24.40 (8,191) | 30.40 (2,047) |
| f32 | 8 | 8 | 23.29 (12,159) | 23.80 (2,047) | 23.59 (12,159) | 25.32 (2,047) |
| f32 | 8 | 16 | 22.77 (12,159) | 22.98 (2,047) | 23.28 (12,159) | – |
| f32 | 8 | 32 | 21.41 (12,159) | 22.02 (2,047) | 22.50 (12,159) | – |

Under T5's per-vector budget the scratch reaches 10.5–12.2 GB at m = 16 and 32, and at f32
p = 3 it **failed to allocate** beside the other user's 80 GB (the device error "bytes were
never written", T5's failure on locust); under one vector's 2 GB every run completed. With
the reduction by chunk one vector's budget costs at most 3 % per vector against T5's (p = 8,
m = 32: 22.02 against 21.41 ms; at p = 3, m ≤ 8, within run-to-run variation), so on CUDA
the several-vector budget is now **one vector's** (`device::vector_settings`). The walking
reduction under one vector's budget grows to 1.24× per vector at p = 8, m = 2 (37.5 against
30.4 ms) and stays within 25 % here, less than on Metal: CUDA's 2 GB gives few chunks. f64
was not reached (the H100 busy again); the mechanism is the same.

## 7. The checks

On the M3 Max (sandboxed unless said):

- the root CI set: `cargo fmt -- --check`, `cargo clippy -- -D warnings`, `cargo clippy
  --examples -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test`, `cargo doc --no-deps`;
  the stricter `cargo clippy --workspace --all-targets -- -D warnings` and
  `RUST_MIN_STACK=8388608 cargo test --workspace`: all passed;
- `nd-fmm-kernels`: clippy with `cpu,metal`, `cargo check --features cuda`, `cargo test`,
  `cargo test --features cpu --release` (75 tests, 38 s), `cargo doc`: passed; Metal
  unsandboxed, `--ignored`: 55 tests passed;
- `nd-fmm-exec`: clippy with `cpu,metal`, `cargo check --features cuda`,
  `RUST_MIN_STACK=8388608 cargo test --features cpu --release` (`mpi_exec` 62 s with the
  variant checks, `device_tune` 123 s), `cargo doc --features cpu`: passed; Metal
  unsandboxed, `--release -- --ignored` (`device_metal`, `device_fmm`, `accuracy`,
  `adaptive`, `device_ranks` at one rank, `multi_rank` at one rank, the operator tests):
  passed;
- `nd-fmm-validate` and `nd-fmm-bench`: clippy (also with `cpu,metal`), `cargo check
  --features cuda` (bench), `cargo test` (bench also with `--features cpu --release`):
  passed.

On locust (by hand, `tools/gh200/remote.sh` environment): `cargo test -p nd-fmm-kernels
--release --features cpu,cuda -- --include-ignored` (143 tests, the CUDA ones and the CPU
runtime's on Grace), clippy of `nd-fmm-kernels` and `nd-fmm-exec` with `cpu,cuda`,
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cuda --release` and the same
with `-- --ignored`, on the final tree: passed, with the 24 CUDA device scenarios of
`mpi_exec`'s cases × 8 variant evaluations bit for bit. In the ignored run one test,
`device_m2l_equals_direct_on_cuda`, failed once allocating 470 MiB (another user's vLLM
server held about 90 GB of the H100 at the time); run alone right after, it passed.

Multi-rank, by hand on the M3 Max with the loopback flags and an external timeout:
`tests/mpi_exec.rs` with the CPU runtime (`--features cpu --release`) at 1, 2 and 4 ranks
(62, 103 and 113 s; 720, 840 and 840 variant evaluations bit for bit) and the ignored
`tests/device_ranks.rs` at 2 and 4 ranks (38 and 37 s): passed. This found a regression,
fixed before these runs: with a shared-memory cube of 8 units on the CPU runtime, whose
`sync_cube` is a yielding spin barrier, `mpi_exec` at 4 ranks took 6,420 s (the same run on
main: 46 s; the four ranks' worker pools on 16 cores spun on descheduled units, seen in a
stack sample); `GemmLayout::shared_for` now gives the CPU runtime a one-unit cube, and the
kernel tests keep the multi-unit cubes covered.

Not run: 8 ranks (T7 changes no exchange, evaluator or `Fmm` stage); Kathleen (no x86 or
multi-node part).

## 8. The raw data

Kept under `bench-results/t7/` of the branch (ignored by git): `metal-*.md` (the
`gemm_variants` tables and their stderr), `bench-metal-*.md` (the four `nd-fmm-bench`
reports), `t7-device-metal/` (the 48 launches of the sweep, each with the load before and
after) and `t7-budgets-metal/`; from locust `cuda/` (copied from
`/data/ucahtbe/logs/t7/measure`, each run with its `.state` file: `nvidia-smi` and `uptime`
before and after, and `.foreign`: every other GPU process seen during the run, empty for
every run reported; runs another process touched are in `void/`).
