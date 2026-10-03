# Phase 4 / T2 — CubeCL 0.11.0-pre.4: the pin, the spike ported, the notes (prerequisite of C4.1)

The workspace pins `cubecl =0.10.0`, which cannot run f64 on CUDA. 0.11.0-pre.4 can,
and brings a frontend refactor. Decided on 2026-10-03: Phase 4 moves to 0.11.0-pre.4
now, so every Phase 4 kernel is written against it once (docs/phase4/README.md,
"Decisions to sign off", question 5). This task moves the pin, ports the only CubeCL
code in the workspace (`spikes/cubecl-gemm`), re-measures it, and records what changed,
so that later tasks start from working 0.11 code.

The development machine is an Apple M3 Max: Metal (f32) and the CubeCL CPU runtime.
No CUDA card is available, so CUDA is type-checked only.

Read first: root CLAUDE.md; docs/phase4/README.md; docs/design/laplace-fmm-plan.md §6.1,
§6.2 and §9.2; spikes/cubecl-gemm/ in full (`Cargo.toml`, `src/*.rs`, `SPIKE_REPORT.md`,
`results-m3max.md`, `analyse.py`); the CubeCL release notes for 0.11.0-pre.1 to pre.4
and the `cubek-matmul` changes since 0.2.0; the 0.11.0-pre.4 sources of `cubecl`,
`cubecl-cpu`, `cubecl-wgpu`, `cubecl-cpp` (CUDA), `cubek-matmul` and `cubek-std` where
the port needs them.

Do:
- **The pin.** In `[workspace.dependencies]`:
  - `cubecl = "=0.11.0-pre.4"`, `cubek-matmul = "=0.3.0-pre.4"` and
    `cubek-std = "=0.3.0-pre.4"`, or whichever `cubek-*` pre-release is published
    against that `cubecl` (check its manifest and say which);
  - keep `default-features = false` and the features the spike needs, renamed if 0.11
    renamed them;
  - update the comment above the entries: the version, why 0.11 (f64 on CUDA), that it
    is a pre-release, and that only `nd-fmm-kernels` and spikes may use it.

  Regenerate `Cargo.lock` and report which crates it gained, lost and changed. No
  default member may build CubeCL: check that `cargo tree` for the default members
  shows none.
- **Port `spikes/cubecl-gemm`** to 0.11:
  - the hand-written kernels (`tiled-smem`, `tiled-reg-cols`, `tiled-reg-rows`), the
    library matmul calls and strategies, the benchmark loop and the backend features
    (`metal`, `cpu`, `cuda`);
  - keep the kernels' algorithms. Change only what the API requires, and keep the
    0.10.0 code readable in git history (no unrelated reformatting).
- **Re-run** on the M3 Max, outside the sandbox (Metal needs GPU access; the CPU
  runtime's build downloads the `tracel-llvm` bundle):
  - `cargo test -p nd-fmm-spike-cubecl-gemm --release`;
  - the full sweep on Metal, and the CPU runtime in f64 (and f32 with `--no-lib`, as
    before), with every BLAS thread variable set to 1;
  - keep the output as `results-m3max-0.11.md` beside the 0.10.0 file.
- **Compare with 0.10.0**, in a new section of SPIKE_REPORT.md ("CubeCL 0.11.0-pre.4"):
  - GFLOP/s per (backend, precision, p, B, implementation), old and new side by side,
    and their ratio. Flag changes beyond the ±25% Metal variance, and repeat a flagged
    cell three times before calling it a regression;
  - whether `Strategy::Auto` still picks CMMA for p ≥ 8 and the scalar path at p = 4
    in f32, and the names of the 0.11 strategies;
  - the f64 capability per backend as 0.11 reports it: Metal and the CPU runtime
    measured; CUDA from the `cubecl-cpp` source (does 0.11.0-pre.4 register f64 for
    CUDA?), with the file and line;
  - CPU-runtime facts: the LLVM bundle version, cold build time, kernel compile times
    (library f32 included, with a time limit), and whether `CUBECL_CPU_STACK_MB` is
    still needed;
  - **the CPU runtime's compiler changed.** 0.10.0 JIT-compiled kernels through MLIR at
    LLVM optimisation level 0 (`cubecl-cpu` 0.10.0, `compiler/module.rs`:
    `ExecutionEngine::new(&module, 0, …)`). 0.11.0-pre.4 compiles them through the new
    `cubecl-llvm` crate with the `default<O3>` pipeline (`cpu/jit/engine.rs`), and does
    not contract fma on the CPU target (`shared/to_llvm/math.rs`). Confirm both from
    the sources you build against. State in the report that the 0.10.0 CPU-runtime
    figures of Phase 0 are **superseded** by the 0.11 measurements, not carried over:
    design §6.1's "about 1% of CPU peak" for the library and "4–26%" for the
    hand-written kernels were measured at level 0. Also check whether the JIT targets
    the host CPU and its features (for example NEON and the host's `target-cpu`), and
    say how you established it;
  - how the CPU runtime maps a launch, confirmed from the 0.11 sources: one task per
    unit of a cube, each looping over every cube in turn, on a pool of one worker per
    core (a dedicated worker per unit for kernels with `sync_cube` or shared memory);
    a plane of one unit; `sync_cube` as a spin barrier; SIMD only from
    `Vector<T, N>` vectors and LLVM's vectorisation inside one unit's code. T3 builds
    on this.
- **Migration notes** in the same section, for the later tasks. List every API change
  the port met, with the 0.10.0 and the 0.11 form side by side:
  - kernel and launch macros, comptime parameters, `Vector` and vectorisation;
  - shared memory and plane operations;
  - the client and buffer API, and the matmul entry points and strategies;
  - device properties and the capability query;
  - compilation options (fast math, if exposed);
  - anything deprecated or behind a feature.

  Mark what you inferred rather than compiled.
- **CUDA type-check**: `cargo check -p nd-fmm-spike-cubecl-gemm --release
  --no-default-features --features cuda` succeeds on the Mac. Update the "CUDA run"
  section with the 0.11 command, and whether `--force-f64` is still needed.
- Root CLAUDE.md:
  - the CubeCL rule: pinned to `=0.11.0-pre.4` (matmul: `cubek-matmul` at its matching
    pre-release), a pre-release chosen on 2026-10-03 for f64 on CUDA; drop "0.10.0 has
    no f64 on CUDA";
  - "Current phase and task briefs" points to docs/phase4/README.md (Phase 4, CubeCL
    kernels; design docs/design/device-path.md once T1 has merged). Keep the older
    phases listed as before.

Tests that define done:
- `cargo test -p nd-fmm-spike-cubecl-gemm --release` passes on 0.11: both hand-written
  kernels against the reference on the CPU runtime, f32 and f64, every guard shape.
- Every measured result in the new sweep passes the spike's correctness check (1e-12
  for f64, 1e-5 for f32).

Must pass:
- `cargo build` and the root checks for the default members, which are unchanged apart
  from the lock file;
- `cargo clippy -p nd-fmm-spike-cubecl-gemm --all-targets -- -D warnings`, and the
  `cuda` type-check above;
- the stricter workspace checks. If the spike's CPU-runtime build cannot run in the
  sandbox, run them outside it and say so.

Do not:
- add `nd-fmm-kernels` or any production code (T4);
- add the spike to the default members or to CI;
- change the spike's algorithms or tolerances to make a result pass. If a kernel fails
  on 0.11, report it and stop;
- assert timings, or commit output beyond `results-m3max-0.11.md`;
- update the design documents beyond the spike report. T13 carries the pin into
  laplace-fmm-plan §6.1 and §9.2.
