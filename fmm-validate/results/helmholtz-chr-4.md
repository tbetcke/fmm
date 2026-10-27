# Helmholtz CHR.4: FFTW on the machines

Helmholtz task CHR.4 ([docs/helmholtz/tasks/CHR.4.md](../../docs/helmholtz/tasks/CHR.4.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CHR.4, §6.7, R15 and R22. Measured on 2026-10-27 from branch `tbetcke/h-chr-4` (the probe of `d4bb153`) with
`tools/fftw/probe.sh` ([tools/fftw/README.md](../../tools/fftw/README.md)), against RLST
0.12.0, the workspace's pin.

## Summary

Every machine of record, CI included, has FFTW in both precisions, at least 3.3.10, and
RLST 0.12's own `cargo test --features fftw_system` passes on each, `fftw_determinism`
included. locust, Kathleen and CI have SIMD codelets (CI only SSE2 and AVX); **the M3
Max's Homebrew FFTW has none** (R22), so its FFTW timings in S9 are those of scalar code.
No machine needs the fallback.

| Machine | FFTW | `pkg-config` fftw3, fftw3f | `rlst::fftw::version()` (SIMD) | single precision | RLST's tests | Record |
| --- | --- | --- | --- | --- | --- | --- |
| M3 Max (Apple M3 Max, NEON) | Homebrew `fftw` 3.3.11 (bottle) | 3.3.11, 3.3.11 | `fftw-3.3.11` (none) | `fftw-3.3.11` | 789 passed; determinism 10/10 | `tools/m3max/machine.md` |
| locust (Neoverse-V2: ASIMD, SVE, SVE2) | spack fftw 3.3.11 (`env.sh`) | 3.3.11, 3.3.11 | `fftw-3.3.11-neon` | `fftw-3.3.11-neon` | 789 passed; determinism 10/10 | `tools/gh200/machine.md` |
| Kathleen (Xeon Gold 6248: AVX2, AVX-512, job 608854) | module `fftw/3.3.10/gcc-12.3.0` | 3.3.10, 3.3.10 | `fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512` | the same | 789 passed; determinism 10/10 | `tools/kathleen/machine.md` |
| CI (`ubuntu-latest`, Ubuntu 26.04.1; AMD EPYC 7763: AVX2) | Ubuntu's `libfftw3-dev` 3.3.10 | 3.3.10, 3.3.10 | `fftw-3.3.10-sse2-avx` | the same | 789 passed; determinism 10/10 | `tools/ci/machine.md` |

"RLST's tests": the unit tests, the integration tests and the doctests of
`cargo test --locked --features fftw_system` (789 passed, 0 failed in 7 harness runs),
`fftw_check_allocations` (10 checks, no allocation per execution) and `fftw_determinism`
(10 scenarios, 0 failed: plans and output bits equal across fresh processes, planning
history, fixed-batch units, wisdom).

## 1. The probe

`tools/fftw/probe.sh`, run inside each machine's environment, prints the CPU and its SIMD
flags, pkg-config's versions and `libdir`s, both libraries' version strings from a small
C program built with pkg-config's flags (`rlst::fftw::version()` reads only the double
one), then copies the registry's RLST of the workspace pin to `target/fftw-probe/` and
runs its own tests there as a package of its own (`[workspace]` appended, RLST's packaged
`Cargo.lock`, `--locked`), with an added example that prints `rlst::fftw::version()`.
Last, it lists the FFTW libraries the binaries load. Nothing enters the workspace, its
lock file or the default build; `deny.toml`'s check is unaffected.

## 2. Per machine

**M3 Max** (outside the sandbox, which denies `sysctl` for the CPU line). Homebrew's
fftw 3.3.11 bottle; both libraries report a bare `fftw-3.3.11`: built without SIMD
codelets. RLST finds it through pkg-config (its build script's Homebrew fallback is not
needed). The binaries load `/opt/homebrew/opt/fftw/lib/libfftw3{,f}.3.dylib`.

**locust** (`tools/gh200/remote.sh`, load 0.05). spack's fftw 3.3.11 (`spack.lock`:
`precision=double,float`, `+mpi`, `~openmp`), with NEON in both precisions. The
binaries load it from the spack view through `env.sh`'s `LD_LIBRARY_PATH`, not the RPM's
3.3.8 in `/usr/lib64`. Without `env.sh` (`env -i PATH=/usr/bin:/bin pkg-config
--modversion fftw3 fftw3f`) the system pkg-config reports 3.3.8 for both: a build outside
the environment would link the old library.

**Kathleen** (job 608854, `run.sbatch`, one node, node-c11a-002, 51 s; job 608850 before
the self-review's changes to the probe gave the same results). The module's
fftw 3.3.10 (UCL's spack 0.23 for `cascadelake`), SSE2, AVX, AVX2 and AVX-512 codelets in
both precisions. Every compute node has the same CPU, so all ranks of a job run one FFTW
build with one SIMD set (R15).

**CI** (the `fftw-probe` workflow on this pull request, run 40481567890, job
122112891738). `ubuntu-latest` was image `ubuntu-26.04` 20261021.618 on an AMD EPYC 7763
(AVX2, no AVX-512). Ubuntu's FFTW 3.3.10 (`3.3.10-2fakesync1build3`) has SSE2 and AVX
codelets in both precisions, no AVX2, from `/usr/lib/x86_64-linux-gnu`. The runner's CPU
varies between runs; CI checks correctness only, so this matters only in that its FFTW
bits are not comparable with another machine's.

## 3. What it means for S9 and D7

- S9 (CH0.9) can run FFTW on all three machines of record; its FFTW timings on the M3
  Max are scalar and do not stand for a SIMD build. A SIMD FFTW on the M3 Max would be a
  build of our own (not Homebrew's bottle), a change to that machine's environment, not
  proposed here.
- D7's per-machine choice has its inputs: FFTW with SIMD on locust and Kathleen; on the
  M3 Max the product or rustfft may win, which S9 measures.
- R15's "one FFTW build per machine" holds on each machine of record: one library per
  machine, the same string in both precisions, and on Kathleen one CPU type. On CI the
  library is fixed by the image, the CPU is not; nothing there compares bits across runs.

## 4. Checks

| check | where | result |
| --- | --- | --- |
| `cargo fmt --all -- --check`, `cargo clippy -- -D warnings`, `cargo clippy --examples -- -D warnings`, `cargo doc --no-deps` | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test` | M3 Max | passed (823) |
| `cargo clippy --workspace --all-targets -- -D warnings` | M3 Max | passed |
| `RUST_MIN_STACK=8388608 cargo test --workspace` | M3 Max | passed (878) |
| `shellcheck -x tools/fftw/probe.sh`, `dash -n tools/fftw/probe.sh` | M3 Max | clean |
| `tools/fftw/probe.sh` | M3 Max, locust, Kathleen 608854, CI run 40481567890 | all steps passed on each |
| PR #127's CI | GitHub | `run-tests`, `run-tests-simd` (both), `run-tests-kernels`, `run-licence-check`, `fftw-probe` passed; `run-tests-mpi` see the PR |

No crate, manifest or lock file changed, so the MPI list, the crate-specific checks and
the Laplace protection of constraint 1 do not apply; Laplace is not touched.
