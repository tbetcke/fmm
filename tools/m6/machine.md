# M6: machine facts

The development machine since 2026-10-31, which drives locust and Kathleen; it replaced
the M3 Max (`tools/m3max/machine.md`, kept as the old machine's record). This file starts
with the facts of the move; other tasks add theirs.

| Item | Value | Command |
| --- | --- | --- |
| Machine | Mac mini (Mac18,5), Apple M6, 12 cores (2 Super, 4 Performance, 6 Efficiency), 24 GB | `system_profiler SPHardwareDataType`, `sysctl hw.perflevel<i>.name hw.perflevel<i>.physicalcpu` |
| Cores used by the sweeps | 6: Super and Performance (`tools/scaling/run.sh`, `tools/fmm3d/run.sh`; `nd_fmm_validate::bench::performance_cores`) | the user, 2026-10-31 |
| GPU | Apple M6, 12 cores, Metal 4 | `system_profiler SPDisplaysDataType` |
| OS | macOS 27.2 (26B123), Darwin 27.2.0, arm64 | `sw_vers`, `uname -srm` |
| SIMD | NEON (also SME, SME2, FP16, BF16; unused) | `sysctl hw.optional.neon hw.optional.arm.FEAT_SME …` (denied inside Claude Code's sandbox) |
| Toolchain | rustc 1.99.0 (b940084d7 2026-09-28), as locust and Kathleen | `rustc -V` |
| MPI | Homebrew's Open MPI 5.0.11, PMIx 6.1.0, hwloc 2.15.0 | `mpirun --version`, `brew list --versions` |

## hwloc and MPI

Homebrew's hwloc 2.15.0 does not finish its `darwin` topology discovery on the M6 (the
three core levels; `hwloc__darwin_build_perflevel_cache_level`): `lstopo` and every
`MPI_Init` hang. Every MPI run and every `cargo test` of an MPI crate runs with
`HWLOC_COMPONENTS=-darwin`. hwloc then reports no cores and Open MPI no slots, so launches
map by slot (root `CLAUDE.md`, "MPI"):

- correctness, the CI list (`.github/scripts/run-mpi-tests.sh`):
  `PRTE_MCA_rmaps_default_mapping_policy=slot:OVERSUBSCRIBE`;
- timings: `mpirun --host localhost:6 --map-by slot --bind-to none` (`tools/scaling/run.sh`).

Measured on 2026-10-31, the cube at N = 10⁶, f64 p = 8, 4 and 6 ranks, in the order A B B
A A B: the two settings' wall times agree within the run-order drift (about 5 % over six
launches, both settings alike); at 6 ranks the oversubscribed exchanges took about 0.5 ms
longer (0.15 % of an evaluation).

## BUDGET

Phase 5N's node budget of `tools/scaling/run.sh` (`BUDGET`, MB) stays at the M3 Max's 56:
the user, 2026-10-31, after the `crossover` sweep here (`CROSSOVER="2 4 6"`, one launch per
cell under the quiet gate, `--host localhost:6 --map-by slot`, `HWLOC_COMPONENTS=-darwin`).
Wall time per evaluation, the median of 10, in ms; Dense / Rotation below 1 means Dense is
faster:

| configuration | ranks | dense MB per node | Dense | Rotation | Dense / Rotation |
| --- | ---: | ---: | ---: | ---: | ---: |
| f32 p = 8, cube | 2 | 17.4 | 1972 | 2020 | 0.98 |
| | 4 | 34.9 | 1163 | 1275 | 0.91 |
| | 6 | 52.3 | 994 | 1055 | 0.94 |
| f32 p = 8, Plummer | 2 | 17.4 | 3084 | 3151 | 0.98 |
| | 4 | 34.9 | 1855 | 1987 | 0.93 |
| | 6 | 52.3 | 1513 | 1617 | 0.94 |
| f64 p = 6, cube | 2 | 12.8 | 1500 | 1411 | 1.06 |
| | 4 | 25.5 | 895 | 883 | 1.01 |
| | 6 | 38.3 | 738 | 732 | 1.01 |
| f64 p = 6, Plummer | 2 | 12.8 | 2289 | 2102 | 1.09 |
| | 4 | 25.5 | 1361 | 1326 | 1.03 |
| | 6 | 38.3 | 1080 | 1053 | 1.03 |

Dense wins f32 p = 8 at every count up to 52 MB per node, as 56 decides. It loses f64
p = 6 at every count, most at 2 ranks (the smallest tables): a per-rank difference, not the
node-level cache effect the budget models, so no budget matches it (below 12.8 MB it would
also give f32 p = 8 to Rotation, which loses there by 2–9 %). The differences of 1–3 % at 4
and 6 ranks are within the run-order drift of this machine (about 5 % over six launches).
Since Phase 6 T11 the library's `auto` does not read the budget.

## FFTW (Helmholtz CHR.4)

Measured on 2026-10-31 with `tools/fftw/probe.sh` (tools/fftw/README.md), outside the
sandbox.

| Item | Value |
| --- | --- |
| FFTW | Homebrew's `fftw` 3.3.11 (bottled), `/opt/homebrew/Cellar/fftw/3.3.11/lib` |
| `pkg-config --modversion fftw3 fftw3f` | 3.3.11, 3.3.11 (Homebrew's pkg-config) |
| version strings (`fftw_version`, `fftwf_version`) | `fftw-3.3.11`, `fftw-3.3.11`: both precisions, **no SIMD** (no suffix: built without SIMD codelets), as on the M3 Max |
| `rlst::fftw::version()` | `fftw-3.3.11` |
| loaded at run time | `/opt/homebrew/opt/fftw/lib/libfftw3.3.dylib` and `libfftw3f.3.dylib` (`otool -L`) |
| RLST 0.12.0's `cargo test --locked --features fftw_system` | passed: 789 tests, 0 failed; `fftw_check_allocations` 10 checks; `fftw_determinism` 10 scenarios, 0 failed |

FFTW here runs scalar codelets (proposal R22), as on the M3 Max: correct and
deterministic, but its timings say nothing about a SIMD build. D7 decides per machine.

## The C3S.4 reference

`spikes/p2p-simd/results-m6.md`: the Phase 3S prototype on this machine (one thread, a
quiet machine, outside the sandbox), the reference of C3S.4's 90 % throughput gate here;
`results-m3max.md` remains the M3 Max's.
