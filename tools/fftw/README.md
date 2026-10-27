# tools/fftw: the FFTW of each machine

`probe.sh` (Helmholtz CHR.4) shows which FFTW RLST's `fftw_system` feature finds on a
machine and whether RLST's own FFTW tests pass against it. nd-fmm's optional `fftw`
feature (CHR.3, CH4.6) links that FFTW; the default build links none (proposal §6.7).
Each machine's result is recorded in its `machine.md`.

| Machine | Command | Record |
| --- | --- | --- |
| M3 Max | `tools/fftw/probe.sh` (outside the sandbox for the CPU line: it denies `sysctl`) | `tools/m3max/machine.md` |
| locust | `tools/gh200/sync.sh`, then `tools/gh200/remote.sh 'tools/fftw/probe.sh'` | `tools/gh200/machine.md` |
| Kathleen | `tools/kathleen/sync.sh`, then `tools/kathleen/submit.sh run.sbatch 'tools/fftw/probe.sh'` | `tools/kathleen/machine.md` |
| CI | the `fftw-probe` workflow (`.github/workflows/fftw-probe.yml`): on pull requests that change this directory, and by hand | `tools/ci/machine.md` |

The probe prints, as Markdown:

1. the host, OS, CPU and its SIMD flags;
2. `pkg-config --modversion fftw3 fftw3f` and each library's `libdir`;
3. both libraries' version strings, from a small C program built with pkg-config's
   flags. FFTW appends the SIMD extensions it was built with (`fftw-3.3.11-neon`,
   `fftw-3.3.10-sse2-avx-avx2-avx2_128-avx512`); a bare version means scalar code only;
4. RLST's own `cargo test --locked --features fftw_system`, `fftw_determinism` (plans and
   bits across fresh processes) and `fftw_check_allocations` included. The registry's copy
   of the workspace's RLST is copied to `target/fftw-probe/` and built there as a package
   of its own, with RLST's packaged `Cargo.lock`, so nothing enters the workspace;
5. `rlst::fftw::version()`, from an example added to that copy, and the FFTW libraries
   that example loads (`ldd`, `otool -L`), as the C program's in step 3: a build against
   one FFTW that loads another at run time shows here. On Linux the example lists only
   `libfftw3`, since the linker drops the single-precision library it does not call; the
   C program lists both;
6. a summary; the exit status is non-zero if a step failed.

The output of steps 2 to 5 is also kept in `target/fftw-probe/<step>.log` (removed at the
start of each run).
