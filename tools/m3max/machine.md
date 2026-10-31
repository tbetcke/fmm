# M3 Max: machine facts

The development machine until 2026-10-31, which drove locust and Kathleen; the M6
replaced it (`tools/m6/machine.md`). This file starts with the facts
Helmholtz CHR.4 needed; other tasks add theirs.

| Item | Value | Command |
| --- | --- | --- |
| Machine | Apple M3 Max, 16 cores (12 performance, 4 efficiency), 64 GB | `system_profiler SPHardwareDataType` |
| OS | macOS 27.0.1 (26A434), Darwin 27.0.0, arm64 | `sw_vers`, `uname -srm` |
| SIMD | NEON | `sysctl hw.optional.neon` (denied inside Claude Code's sandbox) |

## FFTW (Helmholtz CHR.4)

Measured on 2026-10-27 with `tools/fftw/probe.sh` (tools/fftw/README.md), outside the
sandbox.

| Item | Value |
| --- | --- |
| FFTW | Homebrew's `fftw` 3.3.11 (bottled), `/opt/homebrew/Cellar/fftw/3.3.11/lib` |
| `pkg-config --modversion fftw3 fftw3f` | 3.3.11, 3.3.11 (Homebrew's pkg-config) |
| version strings (`fftw_version`, `fftwf_version`) | `fftw-3.3.11`, `fftw-3.3.11`: both precisions, **no SIMD** (no suffix: built without SIMD codelets) |
| `rlst::fftw::version()` | `fftw-3.3.11` |
| loaded at run time | `/opt/homebrew/opt/fftw/lib/libfftw3.3.dylib` and `libfftw3f.3.dylib` (`otool -L`) |
| RLST 0.12.0's `cargo test --locked --features fftw_system` | passed: 789 tests, 0 failed; `fftw_check_allocations` 10 checks; `fftw_determinism` 10 scenarios, 0 failed |

FFTW here runs scalar codelets (proposal R22): correct and deterministic, but its timings
say nothing about a SIMD build. S9 (CH0.9) times it as it is; D7 decides per machine.
