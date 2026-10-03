# Phase 4 / T3 — device arithmetic, the §3.13 addition and the device P2P contract (prerequisite of C4.2, C4.3)

CONVENTIONS §3.13 "Fast kernels" proves that on leaf-scaled data r² = 0 exactly for
coincident points, and that every other r² lies in [2⁻¹⁰⁸, 2⁷]. The proof assumes one
correctly rounded subtraction per dₖ, and squares that are rounded or fused into an fma.
The host SIMD kernels meet that by construction. A GPU compiler need not:
- it may use fast math, reassociating d = u_t − (ĉ + r̂ u_s);
- it may contract or split operations;
- its `rsqrt`, `sqrt` and division may not be correctly rounded;
- it may flush subnormals to zero.

This task measures what each backend actually does, decides what device kernels may
assume, writes it into §3.13 for sign-off, and proposes the accuracy contract of the
device P2P. It also measures whether a CPU-shaped kernel on the CubeCL CPU runtime comes
close to the host SIMD P2P of Phase 3S, which decides whether the CPU runtime gets a
performance target in this phase or stays a correctness backend. It is the Phase 4
counterpart of Phase 3S T1 and T2. It writes spike code and conventions, and no
production Rust.

Backends: Metal (f32) and the CubeCL CPU runtime (f32, f64) on the M3 Max. CUDA cannot
be run. For CUDA, record what the documentation and the 0.11.0-pre.4 code generator say
(for example the `nvcc`/NVRTC flags CubeCL passes), marked as not measured.

Read first:
- root CLAUDE.md; docs/phase4/README.md ("Design decisions", "Accuracy measures");
- CONVENTIONS §3.1, §3.10 and §3.13 (all of it; "Relative frames", "Operators in scaled
  coordinates", "Coincident pairs" and "Fast kernels" closely);
- docs/design/simd-p2p.md §3, §4.2–§4.4 and §5.5;
- tools/fixtures/check_p2p_domain.py, and docs/phase3s/T1-p2p-conventions.md and
  T2-simd-spike.md (how the host case was argued and measured);
- `nd_fmm_ref::p2p`; the `nd-fmm-simd` tests of the 8 / 16 u_T term contract;
- for the CPU-runtime comparison: docs/design/simd-p2p.md §4.1, §4.5 and §8;
  spikes/p2p-simd/SPIKE_REPORT.md (loop order, K, the corrected throughput model);
  `nd_fmm_simd::P2pKernel`; `nd_fmm_validate::p2p_kernels` (workloads and timing
  conventions); and the 0.11.0-pre.4 CPU runtime: `cubecl-cpu`'s
  `compute/threadpool/` (one task per unit on a pool of one worker per core) and
  `cubecl-llvm`'s `cpu/entrypoint.rs` (the loop over cubes) and `cpu/jit/engine.rs`
  (the O3 pipeline);
- spikes/cubecl-gemm/ as ported in T2, with its migration notes;
- the CubeCL 0.11.0-pre.4 sources for the float intrinsics (`sqrt`, `inverse_sqrt`,
  `recip`, division, `fma`, `powf`) and how each backend lowers them:
  - the MSL compiler of `cubecl-wgpu` (what it emits for each, and whether it requests
    fast math from Metal);
  - `cubecl-cpu` (MLIR/LLVM: fast-math flags, fma contraction);
  - `cubecl-cpp` for CUDA (emitted functions and compiler flags);
  - any per-kernel or per-client fast-math or precision option 0.11 exposes;
- the T1 draft, if it is available, section 9.

Do:
- `spikes/device-arith/` (package `nd-fmm-spike-device-arith`, `publish = false`), in
  workspace members but NOT default members, using the pinned CubeCL entries,
  `nd-fmm-ref`, and for the CPU-runtime comparison `nd-fmm-simd` and `nd-fmm-validate`
  (its P2P workloads, as `spikes/p2p-simd` uses them). `unsafe` is allowed in the
  spike; comment each block anyway.
- **Primitive accuracy**, per backend and precision, as kernels that apply each
  operation to inputs uploaded from the host and compare with a correctly rounded host
  reference:
  - `sqrt`, division, `recip`, `inverse_sqrt`, and `inverse_sqrt` followed by one Newton
    step. f32 exhaustively over [1, 4) and at powers of two across the §3.13 domain; f64
    on 10⁷ seeded log-uniform samples over [2⁻¹⁰⁸, 2⁷]. Report the maximum error in
    u_T;
  - the edge cases: 0, −0, the smallest normal, a subnormal, ∞, and NaN in each.
- **Compiler behaviour**, per backend, with inputs constructed so that each effect
  shows in the bits:
  - fma contraction of `a * b + c` (fused or not; does it depend on how the expression
    is written, or on an option?);
  - reassociation (`(a + b) − a`, `x − (y + z)` against `(x − y) − z`);
  - flush to zero of subnormal inputs and outputs;
  - `x == 0.0` for −0, and `select` or masking on a comparison;
  - whether an explicit `fma` call is honoured;
  - whether an available option turns fast math off, and what that costs in a P2P
    loop.
- **The §3.13 argument on the device.** Construct the adversarial leaf-scaled pairs
  that `check_p2p_domain.py` uses (in Rust, from its description; generate no fixture
  files) and the seeded random pairs on levels 0–16. Then, per backend and precision:
  - compute ŷ, dₖ and r² in a device kernel exactly as a P2P kernel would (the
    formulations of the P2P candidates below);
  - check that r² = 0 occurs exactly for coincident stored points, and that every
    nonzero r² is at least 2⁻¹⁰⁶;
  - if a backend breaks this (reassociation, flush to zero), find the formulation or
    option that restores it, or report it and stop.
- **P2P candidates**, potential and with gradients, f32 and f64, in a small device
  kernel over FMM-shaped leaf sets (a target leaf and its 26 neighbours, mapped as
  `LaplaceOperator` maps them, n_t ∈ {8, 32, 64, 128}):
  - `sqrt` and division;
  - `inverse_sqrt` alone;
  - `inverse_sqrt` with one Newton step;
  - each with and without explicit fma.

  For each candidate, report:
  - the pair-term error against `nd_fmm_ref::p2p` in u_T (potential, relative; each
    gradient component, relative to |q| / r²), worst over the domain;
  - the sum error against `direct_sum`, relative to the term magnitudes;
  - the Metal throughput in pairs per second, a quick timing only, to rank the
    candidates. T6 times the production kernel.
- **Draft the §3.13 addition** ("Device kernels", after "Fast kernels"):
  - the coincident-pair rule for device kernels: r² = 0, valid under the conditions you
    found (for example "dₖ by one subtraction of ŷ formed as fl(ĉ + r̂ u_s), no
    reassociation across them", "no flush to zero on the domain", or a stated compiler
    option);
  - that the kernel domain and the f32 gradient range of "Fast kernels" carry over, or
    how they change;
  - the arithmetic every device kernel may assume per backend (correctly rounded or
    not, contraction, flush to zero), stated as the minimum over the backends
    measured, with CUDA marked as from documentation;
  - that §3.10 still holds: §3.13 is in-memory only, and `CONVENTION_VERSION` stays 1.
- **Propose the device P2P contract** for sign-off:
  - the pair-term and sum bounds per backend and precision; the provisional contract
    is that of C3S.4 (8 / 16 u_T per term), so say whether each backend meets it or
    what it meets instead;
  - the recommended formulation per backend and precision;
  - whether any compiler option is required, and the reproducibility statement (what
    differs between backends, as simd-p2p.md §5.5 does for ISAs).
- **A CPU-shaped P2P on the CPU runtime, against `nd-fmm-simd`** (approved on
  2026-10-03). On the CPU runtime a unit is a task on a pool of one worker per core,
  each unit loops over every cube, and SIMD comes only from `Vector<T, N>` (T2's
  notes). So a GPU-shaped P2P (one unit per target, tiles in shared memory,
  `sync_cube`) says nothing about what the runtime can reach on a CPU. From 0.11 on it
  compiles with LLVM's O3 pipeline, so the Phase 0 figures no longer apply. Measure
  what a CPU-shaped kernel reaches:
  - a P2P written for the CPU runtime, in the spike, mirroring the Phase 3S kernel
    (docs/design/simd-p2p.md §4.1–§4.2, and spikes/p2p-simd/SPIKE_REPORT.md):
    - targets in `Vector<T, N>` lanes, with N the width of the host's vectors (NEON:
      4 in f32, 2 in f64), and K vector blocks per unit as the Phase 3S recommendation
      chose;
    - sources broadcast, in input order; no shared memory, no `sync_cube`;
    - explicit `fma` for r² and the terms (cubecl-opt's `InstCombinePass` fuses every
      lone `a * b ± c` on every backend anyway; T2's notes);
    - `sqrt` and division, as the NEON kernel uses;
    - potential only and with gradients, f32 and f64;
  - check it against `direct_sum` before timing it (the T5 sum bounds of Phase 3S), and
    report its pair-term error in u_T;
  - time it on the Phase 3S W1 workload, gathered form (n_t ∈ {8, 16, 24, 32, 64,
    128}, a pool of 64 sets; `nd_fmm_validate::p2p_kernels`' workloads), against
    `nd_fmm_simd::P2pKernel` on NEON, on the M3 Max:
    - **one thread:** a launch with one unit, against `P2pKernel` called on one thread;
    - **all performance cores:** a launch with 12 units, one per core, each taking a
      contiguous share of the target leaves, against `P2pKernel` on 12 rayon threads
      over the same leaves;
    - per-launch overhead measured separately (an empty launch, and one target leaf),
      and excluded from neither figure: the FMM pays it;
    - kernel compilation excluded; median of 15 batches of at least 20 ms; every BLAS
      thread variable set to 1;
  - inspect the JIT output (an object or assembly dump, if the 0.11 runtime offers one;
    otherwise say so) for the inner loop: vector width, fma, `sqrt` and `fdiv` inline
    or called, spills. Compare its instruction count with the NEON kernel's;
  - apply the **decision rule**, fixed now. Take the geometric mean, over the W1 cells,
    of (CPU-runtime time) / (`nd-fmm-simd` time) at one thread, for f32 and f64,
    potential and gradient (24 cells):
    - **at most 1.5:** recommend a CPU-runtime performance target for Phase 4. The CPU
      runtime's P2P in the FMM is to be within 1.5× of the host SIMD P2P per pair at one
      thread, through a CPU layout of the device P2P kernel (a per-backend layout,
      design §6.5) added to T6. State the all-cores ratio too, and what the runtime's
      own thread pool means for `threads(n)` and MPI ranks per node (design §6.8);
    - **above 1.5:** the CPU runtime stays a correctness backend, as design §6.1 says.
      Report where the time goes (loop instruction count, calls, launch overhead) so
      that a later CubeCL version can be re-checked with the same harness;
    - either way the outcome is a sign-off decision (docs/phase4/README.md, decision
      10). Do not change any brief or production code on it.
- **Leaf-operator and GEMM implications**, short: whether contraction and
  reassociation change the harmonics recursion (C4.3) or the GEMM (C4.4, C4.5) beyond
  their tolerances, from a small test of each: one degree-20 regular and irregular
  harmonic evaluation per backend against `nd-fmm-math`, and one GEMM against the host
  `MatrixSet::apply`.
- `spikes/device-arith/REPORT.md` with every table, the machine and backends, the
  CubeCL version, and the recommendation. Keep the raw output as `results-m3max.md`.

Tests that define done:
- A smoke test in `cargo test -p nd-fmm-spike-device-arith --features cpu`: the
  primitive checks and the r² = 0 check on the CPU runtime, at a reduced size.
- The full runs on Metal and the CPU runtime, outside the sandbox, with the output in
  `results-m3max.md`.
- The CPU-shaped P2P passes its `direct_sum` check on every timed cell (a failed check
  aborts that row and is reported). A smoke run of it, one W1 cell per precision,
  belongs to the CPU-runtime smoke test above.

Must pass:
- `cargo clippy -p nd-fmm-spike-device-arith --all-targets --features cpu,metal -- -D
  warnings`;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-math`, which runs the
  `CONVENTION_VERSION` check, unchanged;
- the root checks for the default members, which this task leaves unchanged apart from
  the lock file.

Record in the PR, for sign-off:
- the §3.13 addition;
- the device P2P contract;
- every compiler option or formulation rule the later kernel tasks must follow;
- the CPU-runtime P2P against `nd-fmm-simd`: the table, the geometric-mean ratio, the
  rule's outcome, and the recommendation (a CPU-runtime target and a CPU layout in T6,
  or correctness-only).

Do not:
- change any convention other than the §3.13 addition, or bump `CONVENTION_VERSION`;
- write production code in `fmm-kernels/` or `fmm-exec/`;
- weaken the reference's exact-coincidence rule or the host fast-kernel rule;
- assert timings.
