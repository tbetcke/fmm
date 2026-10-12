# rlst 0.10.0 and CubeCL 0.11.0 (Phase 5S T1s)

The two dependency moves of docs/phase5s/T1s-dependency-upgrade.md, one commit each: rlst
0.9.0 → 0.10.0 (rlst #40: `parsort` by histogram sort with sampling, every rank receives
⌊N/P⌋ or ⌈N/P⌉ records), and CubeCL `=0.11.0-pre.4` → `=0.11.0` with `cubek-matmul` and
`cubek-std` `=0.3.0-pre.4` → `=0.3.0`. Run on 2026-10-11 and 2026-10-12 on the M3 Max,
locust (GH200) and Kathleen. No algorithm, default, tolerance or test bound changed; one
safety check was added (the second part of the input-precision guard, "CUDA: the
library GEMM at TF32" below), decided with the user on 2026-10-12.

**Measured** means run here and read from the output; timings are reported, never
asserted. locust is a shared node: the load before each run is in the drivers' logs (0.0–0.1
before the timed runs, GPU 0 %). Kathleen nodes are exclusive; every job is named.

## Summary

- **Host outputs bit for bit.** rlst 0.10.0 changes only `parallel_sort.rs` among the
  modules the workspace uses, and its signature not at all. The octree on the random share
  is key for key rlst 0.9.0's at 1–8 ranks, and the C3.2 output hashes are unchanged.
- **The octree build at scale** (the first `parsort` of `Octree::new`, exactly balanced
  now): at 160 Kathleen ranks `Octree::new` takes 69 / 89 ms (cube / Plummer) at N = 10⁷
  against T1's 2,607 / 2,655 ms with 0.9.0, and 677 / 853 ms at N = 10⁸ against
  9,656 / 10,093 ms; the heaviest rank's peak during the sort 195 MB against 1,935 MB.
- **Device outputs bit for bit**, once the guard is extended: CPU runtime (f64 p = 8, 12,
  18), Metal (f32 p = 3, 8) and CUDA (f32 p = 3, 8; f64 p = 8, 12, 18) on the C4.8 gate's
  three problems all equal 0.11.0-pre.4's output value for value. No device value changed
  in its last bits, so there is no largest change to report.
- **CUDA: the library GEMM at TF32.** On 0.11.0 `cubek-matmul`'s CMMA launches for f32 on
  CUDA and computes at TF32 precision while reporting F32 types; the input-precision guard
  let it through, and the CUDA f32 FMM at p = 8 lost accuracy (C3.2 gate on CUDA failed,
  device error 17.9× the host's). Fixed in `nd-fmm-kernels` by requiring a registered CMMA
  form with T operands; recorded as docs/design/cubecl-upstream.md entry 4.
- **Compile times and evaluation time** within ±10 % of pre.4 (locust: +0.5 % in kernel
  compilation, −0.6 % to +2.8 % in the device evaluation).

## 1. rlst 0.10.0

**The diff.** `rlst` 0.9.0 → 0.10.0 changes, in `src/`, `distributed_tools/parallel_sort.rs`
(the new splitter search; `OVERSAMPLING` 4 → 16, now candidates per unresolved splitter
per round), `interface/burn.rs` (Burn 0.22; the workspace does not enable `burn`) and the
`doc` modules. `array_tools`, `ghost_communicator`, `all_to_allv`, `sort_to_bins`,
`threading`, the array types, the macros and the `simd` module are unchanged, so the
green-kernels patch moves to 0.10.0 as well. The manifest changes `burn` 0.21 → 0.22
(optional) and the repository links; `mpi` 0.8.2 and `mpi-sys` 0.2.4 stay. The workspace
names `OVERSAMPLING` nowhere; `Octree::new` calls `parsort` once (in `linearize`); no port
was needed.

**`Cargo.lock`:** `rlst` and `rlst-proc-macro` 0.9.0 → 0.10.0, nothing else. (A full
re-resolve also moves pliron 0.18's `spin = "0"` from 0.10.1 to 0.12.3; the lock was kept at
0.10.1, and `cargo metadata --locked` accepts it.)

**Key for key** (an MPI probe, run by hand and not committed: the `scaling` workload on the
random share, the keys of sources and targets as `Fmm` passes them, `max_level` 16, 64
points per leaf, the ghost-children layer; every rank writes its blocks, partition bounds,
leaves, keys with their kind and neighbour lists): the cube and the Plummer sphere at
N = 10⁶ on 1, 2, 3, 4, 5, 6, 7 and 8 ranks (M3 Max, Open MPI over loopback), rlst 0.9.0
against 0.10.0: **every file identical** (108k–242k lines per run). With 0.9.0 the first
sort gave the heaviest rank up to 2.0 times the mean of keys and some ranks none; with
0.10.0, 1.000 and 1.000 at every rank count.

**Host outputs:** the C3.2 gate (`tests/accuracy.rs`, ignored, release) prints the same
four output hashes with both versions (`0x94e1f03747ea1f20` without and `0xaf5fddbe12a92f52`
with gradients, at 1 and 4 threads); `tests/multi_rank.rs` (the one-rank `Fmm` bit for
bit) passes at 1, 2, 4 and 8 ranks in release; `.github/scripts/run-mpi-tests.sh`
(`mpi_regressions`, `mpi_exec`, `mpi_threading` and the six nd-octree examples) passes at
every rank count from 1 to 8 (M3 Max).

### 1.1 `Octree::new` before and after

The probe above at N = 10⁷ and 10⁸, two rounds, the second reported: the first sort
(`parsort` on the same keys with the same seed as `Octree::new`'s, a barrier before;
fastest–slowest rank), its keys received per rank (max / mean), its peak resident size
(`VmHWM` reset through `/proc/self/clear_refs` before the step; max / mean over ranks),
and `Octree::new` (fastest–slowest rank). The leaves per rank are the same with both
versions in every row.

locust (one node, Open MPI 5.0.10, `--map-by core --bind-to core`; rlst 0.9.0 and 0.10.0
built from the same tree, run one after the other, 2026-10-11/12):

| N | dist | ranks | first sort, 0.9.0 → 0.10.0 (ms) | keys max / mean | sort peak max / mean (MB) | `Octree::new`, 0.9.0 → 0.10.0 (ms) |
| ---: | --- | ---: | --- | --- | --- | --- |
| 10⁷ | cube | 8 | 165–238 → 181–181 | 2.00 → 1.00 | 421 / 319 → 309 / 306 | 396–399 → 327–330 |
| 10⁷ | cube | 72 | 50–181 → 80–83 | 17.86 → 1.00 | 248 / 75 → 86 / 81 | 236–237 → 91–92 |
| 10⁷ | Plummer | 8 | 167–240 → 178–179 | 2.00 → 1.00 | 426 / 325 → 322 / 314 | 522–526 → 437–444 |
| 10⁷ | Plummer | 72 | 51–181 → 61–62 | 17.88 → 1.00 | 251 / 77 → 87 / 81 | 253–259 → 101–106 |
| 10⁸ | cube | 8 | 2,004–2,751 → 2,148–2,160 | 2.00 → 1.00 | 3,332 / 2,607 → 2,469 / 2,453 | 4,836–4,856 → 4,027–4,049 |
| 10⁸ | cube | 72 | 674–2,025 → 850–860 | 17.96 → 1.00 | 1,980 / 356 → 355 / 337 | 2,684–2,697 → 1,098–1,112 |
| 10⁸ | Plummer | 8 | 1,993–2,746 → 2,148–2,158 | 2.00 → 1.00 | 3,364 / 2,647 → 2,554 / 2,512 | 7,824–7,898 → 7,129–7,209 |
| 10⁸ | Plummer | 72 | 664–1,998 → 869–880 | 17.97 → 1.00 | 2,008 / 367 → 372 / 346 | 2,938–3,072 → 1,512–1,649 |

The 0.9.0 rows reproduce T1's (docs/design/scale-out.md §7.1: first sort 237–238 ms at 8
and 186–188 at 72 ranks, 17.9 times the share, N = 10⁷).

Kathleen (rlst 0.10.0; 40 ranks per node, `--map-by ppr:20:socket:PE=1 --bind-to core`,
Open MPI 4.1.6 over PSM2 with `PSM2_TID=0`; 40 ranks in job 295044 on node-c11a-016, 160
ranks in job 295538 on node-c11a-[020,061,079,132]); the 0.9.0 column is T1's (job 252220,
scale-out.md §7.1, the same quantities from T1's replay of `Octree::new`):

| N | dist | ranks | first sort (ms), 0.10.0 | 0.9.0 (T1) | keys max / mean | sort peak max / mean (MB) | `Octree::new` (ms), 0.10.0 | 0.9.0 (T1) |
| ---: | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 10⁷ | cube | 40 | 136–137 | 413–418 | 1.00 | 99 / 92 | 197–199 | 552–634 |
| 10⁷ | Plummer | 40 | 134–136 | | 1.00 | 99 / 91 | 242–254 | |
| 10⁷ | cube | 160 | 45–50 | 2,486–2,512 | 1.00 | 68 / 64 | 69 | 2,607 |
| 10⁷ | Plummer | 160 | 44–46 | | 1.00 | 69 / 64 | 87–92 | 2,655 |
| 10⁸ | cube | 40 | 1,921–1,935 | 4,750–4,763 | 1.00 | 534 / 531 | 2,672–2,700 | 6,140 |
| 10⁸ | Plummer | 40 | 1,915–1,929 | | 1.00 | 572 / 556 | 3,342–3,628 | 7,317 |
| 10⁸ | cube | 160 | 480–499 | 8,526–8,592 | 1.00 | 195 / 184 | 675–679 | 9,656 |
| 10⁸ | Plummer | 160 | 486–500 | | 1.00 | 194 / 184 | 820–886 | 10,093 |

(T1's first-sort ranges span its three distributions; where T1 gives one `Octree::new`
range over them, it stands in the cube's row.) The figures agree with T1's measurement of
rlst #40's branch (scale-out.md §7.1: 59 ms and 73 / 92 ms at 160 ranks and N = 10⁷;
522–524 ms and 689 / 900 ms at N = 10⁸; peak 192 / 188 MB). At 160 ranks `Octree::new` is
11–38 times faster than with 0.9.0, and T0's 2 GB ranks are gone. At 8 ranks on locust
the sort itself is about as fast as before (its slowest rank 2.75 → 2.16 s at N = 10⁸): the
imbalance cost little at 8 ranks, where the heavy ranks received twice their share.

## 2. CubeCL 0.11.0

**The manifests.** `cubek-matmul` 0.3.0 and `cubek-std` 0.3.0 require `cubecl = "0.11.0"`
(a caret requirement; the pre-releases pinned `=0.11.0-pre.4`), so the workspace's exact
`=0.11.0` selects 0.11.0. The workspace entries keep `default-features = false` and their
features (`std`, `stdlib`; `std`, `multi-level`); `persistence` is on only with `metal`, as
before (`cargo tree -e features`).

**`Cargo.lock`:** gained `cubecl-monitoring` 0.11.0, `nvml-wrapper` 0.13.0,
`nvml-wrapper-sys` 0.10.0 and `wrapcenum-derive` 0.4.1 (through `cubecl-runtime` 0.11.0),
with a second `thiserror` (1.0.69) and `darling` (0.20.11) for them and a second
`hi_sparse_bitset` (0.9.0, `cubecl-opt`); lost none (`spin` 0.10.1 went with pliron 0.18);
changed every `cubecl-*` crate 0.11.0-pre.4 → 0.11.0, `cubek-matmul`, `cubek-std`,
`cubek-tile` 0.3.0-pre.4 → 0.3.0, `pliron`, `pliron-derive`, `pliron-llvm` 0.18.0 → 0.19.0
and the nine `turso*` crates 0.8.1 → 0.8.2, each required by the 0.11.0 manifests (pliron
"0.19", turso "0.8.2"). 41 changed in all. The CPU runtime's LLVM bundle is unchanged:
`tracel-llvm-bundler` 23.1.0-3 (`TRACEL_LLVM_VERSION` 23.1.0, release 3) in both versions.

**The port** needed one change: `Client::memory_usage()` is gone. `Device::available_memory`
now sums the device pools of every stream from `Client::memory_report(MemoryScope::Device)`,
leaving out the runtime's auxiliary memories (staging, uniforms), which pre.4's
`memory_usage` did not count either. `CUBECL_VERSION` and the version strings follow the
pin. `nd-fmm-exec` and the three spikes compile unchanged (`spikes/device-arith` prints the
new version). `cubecl-metal` with `metal-native` now compiles (`cargo check` of `cubecl`
0.11.0 with that feature, M3 Max); the spike still does not use it.

### 2.1 CUDA: the library GEMM at TF32

On locust, 0.11.0 failed two kernel tests that pass on pre.4 (`library_level_call`: the
library's error 1.48e-3 against the bound 6.57e-4; `m2l_library`: `Library` chosen where
`HandWritten` is expected), and the C3.2 gate on CUDA failed (f32 p = 8: device error 17.9
times the host's, output 6.0e-4 from the host's). The CUDA f32 p = 8 FMM output on the C4.8
problems changed in 399,458–399,558 of 400,000 values, by up to 2.5e-5 / 5.2e-5 / 1.3e-4
of the largest |value| (clusters / cube / Plummer). A diagnosis binary (not committed)
calling `launch_ref` as `library_gemm` does showed why:

- `SimpleCyclicCmma` on f32 now launches on CUDA (pre.4: "No tile size is available");
- the `MatmulElems` it hands back keep F32 for every stage and register type, so the guard
  passed;
- its error relative to Σ|x b| is 1.5e-4 (G = 4, k = 40, n = 81), 2.1e-4 (k = 2000) and
  1.5e-4 (G = 8, k = 64, n = 100): TF32 accuracy (f32 would be about 1e-7);
- the device registers CMMA forms with F16, BF16 and TF32 (16 × 16 × 8) operands only;
  `type_usage(tf32)` = `Conversion`.

With the user's agreement (2026-10-12) the guard gained a second part: the device must
register a CMMA form whose operands are T (`registers_cmma`, `fmm-kernels/src/translate.rs`,
a unit test beside the first part's). On CUDA f32 the library is then rejected again and
the hand-written kernel runs, as on pre.4; Metal registers an f32 form and keeps the
library (its f32 p = 8 output bit for bit pre.4's). Upstream: docs/design/cubecl-upstream.md
entry 4.

### 2.2 Device results against 0.11.0-pre.4

The device FMM with every kind on the device on the C4.8 gate's problems
(`fmm_accuracy::Config::c33`: the cube, the Plummer sphere and the clusters, N = 10⁵,
gradients on, charge vector 0, default settings), its output written as bits by a probe (not
committed) built against each version, compared value by value (400,000 values each):

| backend | precision, p | values differing from pre.4 |
| --- | --- | ---: |
| CPU runtime (M3 Max) | f64, 8 / 12 / 18 | 0 in all 9 runs |
| Metal (M3 Max) | f32, 3 / 8 | 0 in all 6 runs |
| CUDA (locust), with the guard | f32, 3 / 8; f64, 8 / 12 / 18 | 0 in all 15 runs |
| CUDA (locust), without it | f32, 8 | 399,458–399,558 (above) |

The code generator's changes left every value's bits as they were, so no bound comes near
being tested by the move.

### 2.3 Compile times and the device evaluation

Kernel compilation (`fmm-kernels/tools/compile_times.awk` on CubeCL's profiling log; the
first launch of each variant includes its compilation):

| where | run | pre.4 | 0.11.0 |
| --- | --- | --- | --- |
| locust | `cargo test -p nd-fmm-kernels --release --features cpu,cuda -- --include-ignored` | 862 variants, 72.46 s, slowest 5.77 s | 863 variants, 72.85 s (+0.5 %), slowest 6.08 s |
| CI, `run-tests-kernels` | the job's CPU-runtime test run | 340 variants, 185.82 s, slowest 94.14 s (main, run 38176953355) | 340 variants, 176.17 s (−5.2 %), slowest 77.10 s (PR #99, run 38208426942) |
| M3 Max | `cargo test -p nd-fmm-kernels --features cpu --release` | not measured | 406 variants, 21.86 s, slowest 4.21 s |

The device FMM on locust's H100 (`tools/bench/run.sh --backend host,cuda --precision
f32,f64`, one run each, the defaults: the cube at N = 10⁶, p = 6, 72 threads; median of 10
evaluations, ms):

| backend | precision | pre.4 (2026-10-11 21:59 UTC) | 0.11.0 (2026-10-12 04:58 UTC) | change |
| --- | --- | ---: | ---: | ---: |
| host | f32 | 76.75 | 78.64 | +2.5 % |
| host | f64 | 137.88 | 138.15 | +0.2 % |
| cuda | f32 | 25.26 | 25.97 | +2.8 % |
| cuda | f64 | 45.75 | 45.48 | −0.6 % |

The host does not use CubeCL, so its +2.5 % is the run-to-run noise; the CUDA changes are
within it and within ±10 %. Errors and the bit-identity of the evaluations are unchanged.

## 3. Checks

M3 Max (sandboxed unless said otherwise):
- after the rlst commit: the root checks (`cargo fmt -- --check`, both clippies, `cargo
  test`, `cargo doc`), the strict workspace checks, `run-mpi-tests.sh` at 1–8 ranks,
  `multi_rank` at 1, 2, 4, 8 ranks, the C3.2 hashes;
- after the CubeCL commit: the same root and strict checks; `nd-fmm-kernels`: clippy with
  `cpu,metal`, the CUDA type-check, the CPU-runtime tests (33 + 73, 2 ignored, the new
  guard test among them),
  `cargo doc`; Metal's ignored kernel tests outside the sandbox (53 + 1 passed);
  `nd-fmm-exec` with `cpu,metal` clippy, the CUDA type-check, the CPU-runtime tests, its
  ignored gates on the CPU runtime and on Metal (outside the sandbox), `cargo doc
  --features cpu`; the spikes' clippy and CUDA type-checks.

locust (Open MPI 5.0.10, CUDA 12.6, driver 565.57.01; 2026-10-12, with the guard): clippy
of `nd-fmm-kernels` and `nd-fmm-exec` with `cpu,cuda`; `cargo test -p nd-fmm-kernels
--release --features cpu,cuda -- --include-ignored` (139 + 34 passed, the two guard tests
among them); `cargo test -p nd-fmm-exec --features cuda --release` and the same with
`--no-fail-fast -- --ignored` (every CUDA test, the C3.2, C3.3 and C4.8 gates on CUDA
included: all passed; `device_fmm.rs` 694 s). Without the guard the ignored run stopped at
the C3.2 gate (§2.1). The exec runs used the guard's first form, which took the device
instead of its CMMA set (the same check); the kernel suite was re-run on the final form.

M3 Max, also: `mpi_exec` with `--features cpu` (every scenario repeated on the CPU
runtime) and the ignored `device_ranks` at 2 and 4 ranks: passed.

Kathleen (2026-10-12, 0.54 node-hours of the 5):

| Job | Nodes | What | Waited | Ran | Node-hours | Result |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 295044 | 1 (node-c11a-016) | fetch, release build, probe build, probe at 40 ranks | 1 s | 2:48 | 0.05 | passed |
| 295538 | 4 (node-c11a-[020,061,079,132]) | probe at 160 ranks | 4 s | 0:18 | 0.02 | passed |
| 295633 | 1 (node-c11a-016) | `check.sbatch root strict` | 0 s | 13:41 | 0.23 | root passed; strict: clippy passed, `cargo test --workspace` failed only in the two known `spikes/p2p-simd` tests (tools/kathleen/README.md, T1; also in jobs 238839 and 241834 with rlst 0.9.0) |
| 298133 | 2 (node-c11b-[005,027]) | `mpi.sbatch 80` (2 × 40 ranks, `test`) | 1 s | 7:08 | 0.24 | every executable and example passed |

Kathleen ran the tree before the guard of §2.1, which changes nothing on a backend without
CMMA (Kathleen builds the CPU runtime only).

CI (PR #99, run 38208426942, 2026-10-12): `run-tests`, `run-tests-simd` (x86_64, arm64),
`run-tests-kernels` and `run-tests-mpi` passed. The caches key on `Cargo.lock`, so the
cargo caches were restored only from the prefix fallback (main's pre.4 entries) and the
CubeCL crates were rebuilt: `run-tests-kernels` took 8:12 against 4:21 on main (run
38176953355; its release test build 2 min 9 s against 13 s), `run-tests` 6:52 against 6:18,
`run-tests-mpi` 5:07 against 7:46; `run-tests-kernels` saved its cache under the new key.
