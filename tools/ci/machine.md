# CI: machine facts

GitHub's hosted runners (`.github/workflows/`). The image and the CPU vary between runs,
so each fact names the run it comes from. This file starts with the facts Helmholtz CHR.4
needed; other tasks add theirs.

## FFTW (Helmholtz CHR.4)

Measured on 2026-10-27 by the `fftw-probe` workflow (`tools/fftw/probe.sh`;
tools/fftw/README.md) on pull request #127, run 40481567890, job 122112891738.

| Item | Value |
| --- | --- |
| Runner | `ubuntu-latest`: image `ubuntu-26.04` 20261021.618, Ubuntu 26.04.1 LTS, kernel 7.0.0-1017-azure, x86_64 |
| CPU | AMD EPYC 7763; SIMD `sse2 avx avx2 fma` (no AVX-512) |
| FFTW | Ubuntu's `libfftw3-dev` 3.3.10-2fakesync1build3 (with `libfftw3-double3`, `libfftw3-single3`), `/usr/lib/x86_64-linux-gnu` |
| `pkg-config --modversion fftw3 fftw3f` | 3.3.10, 3.3.10 |
| version strings (`fftw_version`, `fftwf_version`) | `fftw-3.3.10-sse2-avx`, `fftw-3.3.10-sse2-avx`: both precisions with SSE2 and AVX codelets, **no AVX2** (Ubuntu's build enables these two only) |
| `rlst::fftw::version()` | `fftw-3.3.10-sse2-avx` |
| loaded at run time | `/usr/lib/x86_64-linux-gnu/libfftw3.so.3` and `libfftw3f.so.3` (`ldd`) |
| RLST 0.12.0's `cargo test --locked --features fftw_system` | passed: 789 tests, 0 failed; `fftw_check_allocations` 10 checks; `fftw_determinism` 10 scenarios, 0 failed |

CI checks correctness only, never timings, so the missing AVX2 codelets matter only for
the bits: a CI run's FFTW output is not comparable bit for bit with Kathleen's or another
runner CPU's (proposal R15; RLST's determinism test compares within one machine).
