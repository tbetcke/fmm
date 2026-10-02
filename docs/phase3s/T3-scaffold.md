# Phase 3S / T3 — nd-fmm-simd scaffold: crate, ISA detection, dispatch, scalar path (part of C3S.3)

Create the crate that will hold the hand-written host kernels. Give it its public
surface, ISA detection, dispatch and a scalar P2P path, so that T4 and T5 only add vector
code behind an interface that already works and is tested. This task needs neither the
spike's decisions nor the §3.13 addition: the scalar path is plain Rust.

Read first: root CLAUDE.md; docs/design/workspace-structure.md §3, §5.1 and §5.3 (crate
template); docs/design/simd-p2p.md §3, §5 and §7; docs/phase3s/README.md ("Design
decisions", "Machines"); `nd_fmm_ref::p2p` (the semantics to match) and fmm-ref/CLAUDE.md;
fmm-math/src/scalar.rs (`RealScalar`); the `std::arch` documentation of
`is_x86_feature_detected!` and `#[target_feature]`, and the Rust 1.86 and 1.87
release notes (target_feature 1.1, safe intrinsics in matching contexts); the Phase 1
T1 and Phase 3 T3 briefs (how a crate was scaffolded before).

Do:
- `fmm-simd/` (package `nd-fmm-simd`, library `nd_fmm_simd`), version `0.1.0-dev`,
  inheriting `[workspace.package]`:
  - a workspace member and a default member, with a path entry in
    [workspace.dependencies];
  - dependencies: `nd-fmm-math` and `thiserror`; dev-dependencies: `nd-fmm-ref`,
    `proptest`; all through [workspace.dependencies];
  - `fmm-simd/CLAUDE.md` from the template of workspace-structure §5.3. Its rules
    include the unsafe policy of design §5.4, the ISA coverage rule (print and report
    which ISAs ran) and the inlining check. Its oracle is `nd_fmm_ref::p2p` and
    `direct_sum`;
  - crate docs: purpose, the ISAs, the dispatch rule, the semantics (the same as
    `nd_fmm_ref::p2p`, exclusion by r² = 0 with a pointer to §3.13), and how to run
    the tests on each architecture.
- Public surface, as in design §5.1 (names may be refined; the semantics may not):
  - `Isa { Scalar, Neon, Avx2 }`, `#[non_exhaustive]` so that AVX-512 can be added
    later without a breaking change, with `detect`, `is_available`, `available`,
    `lanes::<T>` and `Display`;
  - `SimdScalar`, sealed, for f32 and f64;
  - `P2pKernel<T>` with `new(isa) -> Result<_, IsaUnavailable>`, `detect`, `isa` and
    `evaluate` (the signature of `nd_fmm_ref::p2p::p2p`);
  - `IsaUnavailable`, a `thiserror` error naming the ISA (workspace-structure §5.1,
    "Errors").
- Detection and dispatch, as in design §5.2. In this task every ISA dispatches to the
  scalar path. Every `Isa` variant exists on every target, so that settings and command
  lines parse the same everywhere. `is_available` is false for the ISAs of another
  architecture, and their code is compiled only for theirs (`cfg`).
- The scalar path: a clear loop over targets and then sources in input order, with
  `sqrt` and division, excluding r² = 0. The r² = 0 rule is what T1 drafts. Until T1 is
  signed off, document it as pending, and test it only on inputs where it equals exact
  coincidence.
- Module skeleton for T4 and T5: `arch::{scalar, neon, avx2}` behind `cfg`,
  with the crate-level lints `#![deny(unsafe_op_in_unsafe_fn)]` and
  `#![deny(clippy::undocumented_unsafe_blocks)]`. No unsafe yet.
- Confirm on the pinned toolchain which of the Rust 1.86/1.87 features the design
  relies on are stable. Record the result in the crate docs, with a minimal compiled
  example per feature in a test module.
- Cross-target checks. Install the other architecture's target with rustup
  (`x86_64-apple-darwin` on the Mac, or `aarch64-unknown-linux-gnu` on Linux), and make
  `cargo clippy -p nd-fmm-simd --all-targets --target <it> -- -D warnings` pass. On the
  Mac, try `cargo test -p nd-fmm-simd --target x86_64-apple-darwin` under Rosetta 2, and
  report whether it runs and whether `Isa::Avx2` is detected there.
- Root CLAUDE.md:
  - "Current phase and task briefs" points to docs/phase3s/README.md;
  - the unsafe rule: only in `nd-fmm-simd`'s `arch` modules and dispatch, in
    `nd-fmm-kernels`, and in spikes;
  - the cross-target clippy command for `nd-fmm-simd`.

  In fmm-ref/CLAUDE.md, add `nd-fmm-simd` to "Optimisation belongs in …".
- CI (approved on 2026-10-02; docs/phase3s/README.md, "Decisions to sign off",
  question 4). Add a job `run-tests-simd` to `.github/workflows/run-tests.yml`, beside
  the existing job and on the same trigger (pull requests to `main`):
  - a matrix over an x86_64 runner (`ubuntu-latest`) and an arm64 runner
    (`macos-latest`; `ubuntu-24.04-arm` instead if the repository qualifies for it);
  - stable Rust with clippy, as the existing job installs it, and no MPI or other native
    libraries. The crate needs none: check that `cargo test -p nd-fmm-simd` does not
    build `mpi` or `rlst`;
  - steps: print the CPU (`lscpu` on Linux, `sysctl -n machdep.cpu.brand_string` on
    macOS); `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`;
    `cargo test -p nd-fmm-simd`; `cargo test -p nd-fmm-simd --release -- --ignored`
    (the exhaustive and sampled accuracy tests from T4 on; in this task the ignored set
    may be empty);
  - the x86_64 leg is the phase's only x86_64 hardware (no x86_64 timing machine is
    available), so later tasks rely on its log for the AVX2 results. Make the tests
    print the ISAs they ran so the log shows it;
  - keep it under about 10 minutes per leg. Say in the PR how long both legs took.

  Run the new job on the PR and report its result. A workflow that has not run is
  not reported as passing.

Tests that define done:
- `Isa`: `detect()` is available; `Scalar` is always available; `available()` contains
  `detect()` and `Scalar`; `P2pKernel::new` fails exactly for unavailable ISAs; lanes
  per ISA and precision as in design §3, requirement 7.
- Scalar `evaluate` against `nd_fmm_ref::p2p`, f32 and f64, potential and gradient:
  - bit for bit on inputs without coincident pairs, since it is the same formula in the
    same order. If it is not, explain the difference (for example fma contraction) and
    test to 2 u_T per term instead;
  - coincident points (targets equal to sources, duplicated sources) give the same
    result as the reference;
  - empty sources and targets; accumulation onto nonzero output; length mismatches
    panic as in the reference.
- Chunk invariance and target-position invariance, bit for bit (design §3,
  requirement 4).
- Each ISA test prints the ISAs it ran.

Must pass: the new CI job on both legs, and the existing CI job; `cargo test -p
nd-fmm-simd` (no MPI needed; debug run under 30 seconds);
`cargo clippy -p nd-fmm-simd --all-targets -- -D warnings`; the cross-target clippy
above; `cargo doc -p nd-fmm-simd --no-deps` without warnings; the root checks and the
stricter workspace checks.

Do not:
- write vector code or `unsafe` (T4);
- change nd-fmm-exec, nd-fmm-ref's code or any other crate's code;
- add a dependency beyond those listed;
- read environment variables to select an ISA.
