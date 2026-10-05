# Phase 4S / T3 — device arithmetic on CUDA: the §3.13 rules checked on the H100 (C4S.3)

Phase 4 T3 measured, for Metal and the CubeCL CPU runtime, what device kernels may assume
about arithmetic. Its findings became the "Device kernels" addition to CONVENTIONS §3.13
and the formulation rules every kernel follows (fmm-kernels/CLAUDE.md, "Arithmetic"). It
read CUDA's lowering from the sources and never ran it. Two CUDA facts matter most, both
read and never measured:
- On the default LLVM NVPTX path, fadd, fsub and fmul carry LLVM's `contract` flag. So
  besides cubecl-opt's own fusion, the backend may fuse any multiply–add (F17).
- `inverse_sqrt` lowers to the polyfill `1 / sqrt(x)` there (accurate). The NVRTC path
  would use CUDA's approximate `rsqrt` instead.

f64 has never run on any GPU in this project. This task runs the Phase 4 spike on CUDA,
checks every §3.13 assumption on the H100 in f32 and f64, and either confirms the rules
for CUDA or proposes the smallest change, for sign-off before T4 relies on the f64
results.

It can run in parallel with T2: the spike does not use `nd-fmm-kernels`.

Read first:
- root CLAUDE.md; docs/CONVENTIONS.md §3.13 (all of it, "Device kernels" in
  particular);
- docs/phase4s/README.md ("Accuracy measures", decisions 4 and 10);
- spikes/device-arith/REPORT.md (all of it) and its sources;
- docs/design/device-path.md §5.3, §9.2, §9.3, F16 and F17;
- the CubeCL sources for the NVPTX path: `cubecl-llvm-0.11.0-pre.4/src/nvptx/`,
  `src/shared/to_llvm/math.rs` (`fma_contraction`, the polyfills), and
  `cubecl-cuda-0.11.0-pre.4/src/compiler.rs`.

Do:
- **Run the spike on CUDA**, every section (primitives, compiler, domain, p2p, leafops;
  `cpu-p2p` is CPU-only), f32 and f64:
  `cargo run --release -p nd-fmm-spike-device-arith --no-default-features --features cuda
  -- --backends cuda > spikes/device-arith/results-gh200.md`.
  - Extend `backend.rs` only as far as CUDA needs: open, name, report.
  - Run the CPU runtime on locust as well, as the control.
  - Where the spike has Metal-only branches, add the CUDA case.
- **Answer, per precision, with measurements:**
  1. **Correct rounding.** Are `+ − ×` correctly rounded? Does the compiler keep a lone
     `a · b + c` unfused, fuse it, or fuse it only sometimes? Answer for cubecl-opt's
     pass and for LLVM's `contract` separately, if the spike's compiler section can tell
     them apart. Dump the PTX where it settles a question: CubeCL's debug options, or
     `CUBECL_DEBUG_LOG`.
  2. **`sqrt`, division and `inverse_sqrt`.** Their error in units of u_T on the spike's
     samples, and whether they stay within §3.13's 2.5 u_T. The NVPTX lowering of f64
     division and square root (IEEE `div.rn.f64` and `sqrt.rn.f64`, or approximations)
     must be read from the PTX, not assumed.
  3. **Subnormals.** Flushed in arithmetic (f32 `ftz`), in copies, or neither? Does the
     §3.13 charge condition (q = 0 or |q| ≥ 2⁻¹⁰⁰) still cover it?
  4. **The r² = 0 argument** of §3.13 on CUDA: the domain section, f32 and f64.
  5. **The P2P candidates** and the leaf operators against the host, per the spike's
     measures. Does the signed-off formulation (ŷ = `fma(r̂, u_s, ĉ)`, `inverse_sqrt`
     with no Newton step, masking by compare and select) meet the C3S.4 contract (8 u_T
     potential, 16 u_T gradient per term) in **f64** on CUDA?
  6. **Bit identity.** Do the bit-identity cases of §3.13 rule 6 hold on CUDA? These are
     copies, scatters, frames, and GEMMs and rotations written as host `mul_add`
     replicas. If LLVM's `contract` fuses a product that the replica rounds, it does not.
- **Recommendation**, in a new REPORT.md section "CUDA on GH200 (Phase 4S)":
  - **(a) The rules hold on CUDA as written.** Then record that and change nothing.
  - **(b) A rule needs a CUDA note.** For example: "on CUDA every multiply–add may be
    fused", or "f64 division is …". Draft the §3.13 text, keeping `CONVENTION_VERSION`
    at 1 unless the sign-off decides otherwise (root CLAUDE.md: conventions change only
    by sign-off, proposed in the PR).
  - **(c) A formulation change is needed for CUDA to meet the contract.** Propose it with
    measurements on every backend: a Newton step on CUDA f64, an explicit `fma`, or the
    LLVM `contract` flag turned off. A switch of CubeCL's (none is known, F16, F17) or a
    change to CubeCL itself is an upstream question; list it, do not patch a pinned
    crate.

  State the sign-off questions as a numbered list at the end of the report.
- **Compare with NVRTC only if (b) or (c) arises.** Build the spike once with
  `cubecl/cuda-cpp` (spike only, behind a spike feature) and say whether that path would
  avoid the problem. Switching `nd-fmm-kernels` to NVRTC is decision 4. Do not do it
  here.

Tests that define done:
- `cargo test -p nd-fmm-spike-device-arith --features cpu` passes on the M3 Max and on
  locust.
- The CUDA run completes every section in f32 and f64. results-gh200.md is committed;
  results-m3max.md does not change.
- The report answers questions 1–6 for f32 and f64 with numbers, says how each was
  established (measured, PTX read, source read), and ends with the recommendation and
  the sign-off questions.

Must pass:
- `cargo fmt --all`, then the root checks (the spike is not a default member);
- `cargo clippy -p nd-fmm-spike-device-arith --all-targets --features cpu -- -D
  warnings`, and with `--no-default-features --features cuda`;
- on locust, the CUDA run above.

Do not:
- change `nd-fmm-kernels`, CONVENTIONS.md or `CONVENTION_VERSION` in this task. A §3.13
  change is drafted in the report, and lands after sign-off (in T4, or in a follow-up
  commit to this PR if the sign-off says so);
- change the P2P contract or the tolerances;
- patch CubeCL or enable `cuda-cpp` outside the spike.
