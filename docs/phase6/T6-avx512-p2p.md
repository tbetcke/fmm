# Phase 6 / T6 — AVX-512 P2P (C6.7)

`nd-fmm-simd` ships NEON, AVX2 + FMA and scalar P2P kernels. AVX-512 was deferred until
hardware to test and time it was available (simd-p2p.md §4.7, decided on 2026-10-02):
`pulp` selects its V4 level only with its `nightly` feature, so on stable Rust an
AVX-512 machine runs the AVX2 code (simd-p2p.md §2.2, "Dispatch", on pulp; §4.7). Kathleen's Xeon Gold 6248
(Cascade Lake) nodes have AVX-512F, and `core::arch`'s AVX-512 intrinsics and target
features are stable in current Rust (T1 confirms the release that stabilised them against
the toolchain of the root CLAUDE.md and CI). This task adds the path simd-p2p.md §4.7
describes, by hand in the crate's `arch` modules, and measures it on Kathleen inside the
FMM, as `docs/design/optimisation.md` §5 designs it.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

**This task needs Kathleen** (or another AVX-512 machine). If Phase 5N decision 0 found
Kathleen unusable, T6 waits, as AVX-512 waited in Phase 3S, and the exit checklist
records C6.7 as waiting, not failed. No Kathleen job in this task uses more than one node
(and none ever more than 4).

Read first:
- root CLAUDE.md ("`unsafe` only in nd-fmm-simd's `arch` modules and its ISA dispatch";
  "For a change to nd-fmm-simd, also clippy the other architecture's code"),
  fmm-simd/CLAUDE.md;
- docs/design/simd-p2p.md: §4.1 (lane waste at FMM leaf sizes), §4.3 (inverse square
  root: `_mm512_rsqrt14_ps`/`_pd` below 2⁻¹⁴, one Newton step for f32, two for f64),
  §4.6 (the model: about 2× AVX2 in f32 and 2.4× in f64 per core, less where 512-bit
  operations split), §4.7, §3 (requirement 2: 8 u_T per φ term, 16 u_T per ∇φ term), §5.2
  (detection and dispatch), §5.4 (`unsafe` and its SAFETY comments), §5.5
  (reproducibility), §7 (testing: the exhaustive f32 test, instruction counts), §8
  (benchmarks);
- docs/design/optimisation.md §5 and decision 4's answer;
- tools/kathleen/README.md (jobs, the login-node rule);
- the code: `nd_fmm_simd::{isa, kernel, rsqrt, toolchain}`, `arch::{mod, avx2, p2p,
  scalar, tests}`; `nd_fmm_exec`'s `P2pChoice` and `Fmm::p2p_kernel`; the examples
  `p2p_kernels` and `p2p_fmm` of `nd-fmm-validate`.

Do:
- **`arch::avx512`**: the vector layer (16 f32 / 8 f64 lanes, masks for tails and for the
  r² = 0 rule) and the inverse square root with the Newton steps of simd-p2p.md §4.3,
  behind `#[target_feature(enable = "avx512f")]` (and only the extensions T1 justifies),
  every `unsafe` block with a `// SAFETY:` comment and every public function safe.
- **`Isa::Avx512`** (the enum is `#[non_exhaustive]`), detected at run time
  (`is_x86_feature_detected!("avx512f")`), and the dispatch rule decision 4 fixes (for
  example: chosen by `Auto` only where T6 measures it faster inside the FMM;
  `P2pChoice::Isa(Isa::Avx512)` always selectable where the CPU has it).
- **Accuracy**: the per-pair contract of simd-p2p.md §3 (8 / 16 u_T), with the exhaustive
  f32 test of §7 extended to the new path; bit identity across thread counts as for every kernel.
- **Measurements on Kathleen** (jobs; state node, binding, job id): pairs per second per
  ISA (scalar, AVX2, AVX-512) and precision on FMM-shaped leaf workloads and all-pairs
  sets (`p2p_kernels`); the leaf stage and the evaluation inside the FMM at 1 rank and 40
  ranks per node (`p2p_fmm`, the `scaling` harness) at p = 3 and 8, f32 and f64 — the
  frequency-licence effect of 512-bit code shows only under full-node load. Report in
  the Phase 6 results file, with the model of simd-p2p.md §4.6 beside it.
- **CI**: GitHub's x86_64 runners usually lack AVX-512, so the new tests skip cleanly
  there and run by hand on Kathleen; check whether any runner has it and say so.
  Instruction counts of the compiled loops (as Phase 3S T5 did for AVX2) where hardware
  is missing.

Tests that define done:
- On Kathleen: the AVX-512 kernels against `nd_fmm_ref::p2p` within the contract, f32
  exhaustively as for AVX2; the FMM with `P2pChoice::Isa(Isa::Avx512)` against the
  reference kernel within requirement 1 on every `tests/mpi_exec.rs` scenario.
- Everywhere else: the tests skip with a message; dispatch never selects AVX-512 on a CPU
  without it (a unit test of the dispatch logic with a forced feature set, if the design
  allows one).

Must pass:
- the root checks, and the nd-fmm-simd checks of the root CLAUDE.md:
  `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`, `cargo test -p nd-fmm-simd`,
  the release ignored tests, and clippy for the other architecture
  (`--target x86_64-apple-darwin` on the M3 Max);
- on Kathleen, in a job: `cargo test -p nd-fmm-simd --release -- --ignored --show-output`
  showing the AVX-512 rows.

Do not:
- put `unsafe` outside `arch` and the dispatch; use `pulp`'s nightly feature or a nightly
  toolchain;
- make AVX-512 the `Auto` choice without the in-FMM measurement on a fully loaded node.
