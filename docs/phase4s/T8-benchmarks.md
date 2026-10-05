# Phase 4S / T8 — the GH200 benchmark report and the design-document update (gate: benchmarks published)

Phase 4 measured only Metal f32 on the M3 Max, and left "the CUDA run" as a documented
command (fmm-validate/results/phase4-m3max.md §4). This task makes that run, and the
`nd-fmm-bench` sweep, on locust's GH200. It reports the device FMM in f32 and f64 against
the Grace host path and against the M3 Max. It records what the numbers settle and what
they leave open, and carries the outcome into the design documents. It ends the phase.

Read first:
- root CLAUDE.md, fmm-bench/CLAUDE.md, fmm-validate/CLAUDE.md, tools/gh200/README.md;
- docs/phase4s/README.md (every section; "Exit gate", "Decisions to sign off");
- the T1–T7 reports, and tools/gh200/machine.md (T1);
- fmm-validate/results/phase4-m3max.md (all of it; §4 "The CUDA run");
- docs/design/device-path.md §10.5, §13.4, §17 and §18 (T3, T7);
- docs/design/laplace-fmm-plan.md §6, §7 (Phase 4), §8.3, §9.

Do:
- **The `nd-fmm-bench` sweep on locust**, in release builds through `tools/bench/run.sh`,
  with the GPU otherwise idle (check `nvidia-smi` before and after, and state it):
  - CUDA: N ∈ {10⁵, 10⁶, 10⁷}; f32 at p ∈ {3, 6, 8}; f64 at p ∈ {3, 6, 8, 12, 18};
    the tuned strategy with a fresh tuning cache, and at N = 10⁶ every fixed strategy;
  - the host on Grace: N ∈ {10⁵, 10⁶}, the same (precision, p) points, at 1 thread and
    at 72 threads (all cores). Use 10⁷ only if a 72-thread evaluation stays under about
    a minute;
  - if 10⁷ does not fit or is refused (`SettingsError::DeviceMemory`), report the
    largest N that runs, in steps of 2×.

  Keep the generated Markdown files as the raw output.
- **The Phase 4 CUDA run** (phase4-m3max.md §4), as documented there, adapted only where
  T1–T7 changed a command: `device_fmm` and `autotune` in f32 and f64, `m2l_kernels`,
  and the GEMM spike. Report what it settles, number by number:
  - the f64 GEMM efficiency of the hand-written kernel against the spike's model of
    19–36% of the roofline;
  - the f64 dense/rotation crossover per p, from `autotune` and the fixed-strategy rows;
  - so whether the f64 static rule (dense to p = 11, rotation from 12; decision 13 of
    Phase 4, provisional) holds on Hopper. If T7 already moved it, confirm or correct
    it;
  - the f32 GEMM with the input-precision guard (the library expected off on CUDA);
  - the device FMM against an aarch64 server host (Grace, 72 cores) rather than a
    laptop.
- **The report**, `fmm-bench/results/phase4s-gh200.md`, in the shape of phase4-m3max.md:
  - **Setup:** locust's hardware and software from tools/gh200/machine.md, the spack
    environment's lock hash, rustc, CubeCL, the CUDA driver and toolkit, the peaks used
    (datasheet values for the GH200's H100, labelled as such), and the timing method;
  - **Results:**
    1. the `nd-fmm-bench` summary and kind tables;
    2. the device against the host (Grace at 1 and 72 threads) per (N, precision, p),
       as speed-ups;
    3. GH200 against M3 Max, f32 only, the same problems, from phase4-m3max.md, labelled
       as two different runs on two machines;
    4. kernel efficiency on CUDA (P2P Gpairs/s and the GEMM's GFLOP/s against the
       peaks; rotation against dense per p; T7's numbers where it measured them);
    5. the autotune decisions per (precision, p);
    6. f64 on a GPU: the first measurements, against the models of device-path.md §10.5
       and §13.4;
    7. what was not measured, stated plainly (multi-GPU, multi-rank device runs, an
       x86_64 host, FMM3D).
  - Every number is labelled "measured (locust, backend)" or "model".
- **Leaf size on CUDA**: the Phase 4 leaf-size rule (T13, decision 8) on CUDA f32 and
  f64 at N = 10⁶, p = 3 and 8: `device_fmm --part leaf`, or `nd-fmm-bench` with
  `--leaf-size 16,32,64,128,256` if T6 accepts a list there. Apply the rule and report
  it. If it picks a size other than 64 on CUDA, do not change any default: decision 7 of
  this phase decides whether a per-backend leaf default is added.
- **Design documents**, with the Phase 4S outcome:
  - laplace-fmm-plan.md:
    - the revision note;
    - §6.1 and §6.2 (CUDA as run: compiler path, f64, TF32 guard);
    - §7, a new "Phase 4S" block (status per component C4S.1–C4S.7 with the measured
      numbers, and a "Recommendation for Phase 5" that says what Phase 4S learned for
      it);
    - §8.3 (benchmarking: `nd-fmm-bench`);
    - §9.1 and §9.2 (the risks and questions answered: f64 on a GPU, the f64 rule);
  - device-path.md §17 (a Phase 4S note) and §18 (T3 and T7's CUDA section, completed
    with the measured numbers);
  - workspace-structure.md §2, §3 and §3.1 (`nd-fmm-bench`; the CUDA test files), §6
    (Phase 4S done);
  - docs/phase4s/README.md: tick the exit checklist;
  - **the Phase 5 briefs, for decision 8** (decided on 2026-10-05: Phase 5 tests on
    both the M3 Max and locust). Use one commit of its own, so that it can be reviewed
    apart. Use what Phase 4S measured (MPI on locust, rank and core counts, the GPU's
    behaviour), and leave the briefs' design questions to Phase 5 T1:
    - docs/phase5/README.md:
      - the hardware paragraph and decision 2, revised and dated;
      - "Machines": locust with its ranks (up to 72 on the Grace cores, one rank per
        core) and its backends (host; CUDA for device ranks, all ranks sharing the one
        H100);
      - requirement 9, "tested on what can run": which multi-rank tests run by hand on
        locust and at which rank counts, under `timeout` and without the macOS loopback
        flags;
      - requirement 10 and "Ranks on the M3 Max": where timings come from now, and the
        BLAS and thread rules per machine;
      - "Device ranks": CUDA ranks on locust, correctness only, unless decision 3 of
        Phase 5 is revisited;
      - the risks that one node over shared memory implies (locust is also one node);
      - the exit gate and checklist lines that name the M3 Max alone;
    - T1 (the design measures and models on both machines), T3 (the CI job unchanged;
      by-hand multi-rank runs on locust added), T8 (device ranks on CUDA as well as the
      CPU runtime and Metal) and T10 (strong and weak scaling on locust to 64 or 72
      ranks beside the M3 Max's 1–12; the inter-node command stays documented):
      wherever they assume "the M3 Max only";
    - tools/gh200/README.md: how to run an MPI test at n ranks on locust.

    Do not change the Phase 5 requirements' substance, its other decisions or its
    tolerances. List every changed paragraph in the pull request.

Tests that define done:
- The T1–T7 test suites still pass on the M3 Max (host, CPU runtime, Metal by hand) and
  on locust (CUDA), with the backends run listed.
- `nd-fmm-bench`'s smoke test passes on both machines.

Must pass:
- the root checks and the stricter workspace checks on the M3 Max;
- on locust: `RUST_MIN_STACK=8388608 cargo test --workspace` (MPI from the spack
  environment), and the CUDA test commands of the README's "Checks on locust".

Do not:
- change kernels, layouts, defaults, autotune or the static rule. A defect found here is
  reported, and fixed in its own commit with its test, after asking;
- assert timings, or commit raw output beyond the report;
- edit docs/phase5/ beyond the decision 8 update above;
- present a model or a datasheet peak as a measurement.
