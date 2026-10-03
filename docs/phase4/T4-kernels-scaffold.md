# Phase 4 / T4 — nd-fmm-kernels scaffold: backends, capability, buffers, data movement (part of C4.1)

Create the crate that holds every `#[cube]` kernel. Give it backend selection, the f64
capability check, device buffers, and the primitives every later kernel needs: gather,
scatter-add and scatter by an index array, and zeroing. Measure whether its
CPU-runtime tests fit in CI. T5 then builds the device path of `nd-fmm-exec` on this
crate, and T6–T10 add kernels behind an interface that already works and is tested.

Read first:
- root CLAUDE.md; docs/phase4/README.md ("Requirements", "Design decisions", "Machines",
  "Decisions to sign off" question 7);
- docs/design/device-path.md, signed off: sections 3, 4, 5, 12 and 13, and section 15
  for this task;
- docs/design/workspace-structure.md §3, §5.1 and §5.3 (crate template);
- spikes/cubecl-gemm/ as ported in T2, with the migration notes in its report;
- the Phase 3 T3 and Phase 3S T3 briefs (how a crate was scaffolded before);
- fmm-simd/CLAUDE.md (the unsafe rules, which this crate mirrors);
- `.github/workflows/run-tests.yml`.

Do:
- `fmm-kernels/` (package `nd-fmm-kernels`, library `nd_fmm_kernels`), version
  `0.1.0-dev`, inheriting `[workspace.package]`:
  - a workspace member with a path entry in `[workspace.dependencies]`. It is not a
    default member yet; the CI decision below settles that;
  - dependencies: `cubecl`, `cubek-matmul`, `cubek-std`, `nd-fmm-math` and `thiserror`;
    dev-dependencies: `nd-fmm-ref`, `nd-fmm-tables` and `proptest`; all through
    `[workspace.dependencies]`;
  - features `cpu`, `metal` and `cuda`, each enabling the matching CubeCL runtime, with
    none on by default. Without a feature the crate still builds: kernel definitions
    and host-side types, no runtime;
  - `fmm-kernels/CLAUDE.md` from the template of workspace-structure §5.3. Its rules
    cover:
    - the unsafe policy: CubeCL launches and buffer views only, each block with
      `// SAFETY:`, every public function safe, `#![deny(unsafe_op_in_unsafe_fn)]` and
      `#![deny(clippy::undocumented_unsafe_blocks)]`;
    - the backend coverage rule: print and report which backends ran, and never
      report a backend that did not run as passing;
    - the accumulation and determinism rules of README requirements 4–6;
    - the arithmetic rules signed off with T3 (once they exist; until then a pointer);
    - the sandbox notes for Metal and the LLVM bundle.

    Its oracle is `nd-fmm-ref`, `nd-fmm-tables` and host loops;
  - crate docs: purpose, backends and features, the capability rule, the data
    movement primitives, how to run the tests on each backend.
- **Backend and device**, as the design's section 3 specifies (names may be refined;
  the semantics may not):
  - selecting a backend and device: `Device::open` uses CubeCL 0.11's fallible
    `cubecl::Device` constructors (`cpu`, `metal_msl`, `cuda(0)`), and refuses a Metal
    device that does not come up with the MSL compiler (a silent WGSL fallback) with
    the no-device error. Detect it from the features only the MSL path registers
    (`Plane::Sync` and the CMMA combinations), not from `client.name()`, which is
    `wgpu<msl>` for every Metal device once `msl` is compiled in (device-path.md §3.1,
    F4);
  - the capability query for f32 and f64;
  - a typed error when a backend is not compiled in, no adapter is found, or a
    precision is unsupported;
  - a `Display` that names the backend, device and CubeCL version, for reports.
- **Device buffers**: allocate, upload from a host slice, download into a host slice,
  for f32, f64 and `u32`. Index arrays are `u32` only on the device: the `u16` offset
  indices and `u8` octants of `GroupedCsr` are widened at upload, so there are no `u16`
  or `u8` buffers (device-path.md §3.1). Sizes 0 and 1 must work. No
  allocation per kernel launch beyond what the design allows.
- **Data movement kernels** (fmm-plan-redesign §10 and the GEMM check of §6.4), each a
  safe launch wrapper over a `#[cube]` kernel, generic over f32/f64 (the runtime is a
  run-time value, device-path.md §3.2):
  - zero a buffer, or a range of it;
  - gather columns: Y[:, j] = X[:, idx[j]] for column size n (comptime) and an index
    array;
  - scatter-add columns: X[:, idx[j]] += Y[:, j], with the documented precondition that
    the indices of one launch are distinct, and a debug-mode host check of it;
  - scatter values: x[idx[j]] = y[j] (assign by index), for the charges into their
    slots of the source store (`scatter_values`, device-path.md §3.1).

  No gather or scatter by point offsets (CSR) is needed.
- Check with `cargo tree -e features` (with `cpu`, `metal` and `cuda`) that CubeCL's
  `persistence` feature is off, so that no CubeCL database is ever written, and say so
  in the PR (device-path.md §3.4, §10.1).
- Root CLAUDE.md, "Checks": the `nd-fmm-kernels` commands. These are
  `cargo test -p nd-fmm-kernels --features cpu --release` (and with `metal`, by hand,
  outside the sandbox), and the CUDA type-check
  `cargo check -p nd-fmm-kernels --features cuda`. Also the clippy forms.
- **The CPU-runtime CI job, measured** (README, decision 7, approved on 2026-10-03 as
  "measure, then decide"):
  - add a job `run-tests-kernels` to `.github/workflows/run-tests.yml`, beside the
    existing jobs and on the same trigger: `ubuntu-latest`; stable Rust with clippy; no
    MPI;
  - steps: clippy (`-p nd-fmm-kernels --all-targets --features cpu`), then the
    CPU-runtime tests in release;
  - cache the cargo registry, the target directory and the `tracel-llvm` bundle with
    `actions/cache`, keyed on `Cargo.lock`;
  - run it on the PR, cold and then warm, and report for each: the bundle download, the
    build, the kernel compilation inside the tests, and the total;
  - recommend: keep (and whether `nd-fmm-kernels` should then become a default member),
    change, or drop. If dropped, remove the job in this PR, and add the by-hand commands
    to fmm-kernels/CLAUDE.md instead. A workflow that has not run is not reported as
    passing.
- Fix the CPU-runtime test budget from what you measured (README, "Test time"; target
  under five minutes warm on the M3 Max), and write it into fmm-kernels/CLAUDE.md.

Tests that define done (on the CPU runtime in f32 and f64, and on Metal in f32 by hand;
each test prints the backends it ran):
- Capability:
  - the CPU runtime reports f32 and f64;
  - Metal reports f32 and refuses f64 with the typed error, not a panic;
  - an uncompiled backend gives its error.
- Round trip, bit for bit, f32 and f64: sizes 0, 1, 7, 1000 and 10⁶ (the last
  `#[ignore]` on the CPU runtime if slow), with −0, the smallest normal, a subnormal
  (documenting whether the backend flushes it, from T3), the extremes and
  ±∞; NaN round-trips as NaN. Also `u32`.
- Gather, scatter-add, scatter and zero against plain host loops, bit for bit (no
  arithmetic except the scatter-add, whose single add per element is exact to
  compare). Column sizes (p + 1)² for p ∈ {0, 3, 8, 20}; index arrays empty, of one
  element, permuted and repeated (gather only); accumulation onto nonzero data.
- Property tests (`proptest`) over random sizes and index arrays, on the CPU runtime,
  for gather and scatter-add against the host loops.

Must pass:
- `cargo test -p nd-fmm-kernels`, without features (seconds);
- `cargo test -p nd-fmm-kernels --features cpu --release`, within the budget;
- `cargo test -p nd-fmm-kernels --features metal --release -- --ignored`, by hand on
  the M3 Max;
- `cargo clippy -p nd-fmm-kernels --all-targets --features cpu,metal -- -D warnings`,
  and the same without features;
- `cargo check -p nd-fmm-kernels --features cuda`;
- `cargo doc -p nd-fmm-kernels --no-deps` without warnings;
- the new CI job (if kept), the existing CI jobs, the root checks and the stricter
  workspace checks. Say which runs needed to leave the sandbox.

Do not:
- write operator kernels (T6–T10) or touch nd-fmm-exec (T5);
- add a dependency beyond those listed;
- read environment variables to select a backend or device (CubeCL's own runtime
  variables, such as `CUBECL_CPU_STACK_MB`, are documented, not set by the crate);
- use atomics.
