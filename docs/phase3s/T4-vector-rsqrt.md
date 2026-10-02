# Phase 3S / T4 — nd-fmm-simd: per-ISA vector layer and inverse square root (C3S.3)

The P2P kernel (T5) is written once, generic over a small internal vector layer, and
instantiated per ISA. This task writes that layer for NEON and AVX2 + FMA, and
the inverse square root of each ISA in the formulation the spike chose. It proves the
4 u_T accuracy contract on every ISA the machines offer. It is the first task with
`unsafe`.

Read first: root CLAUDE.md, fmm-simd/CLAUDE.md; docs/design/simd-p2p.md §4.3, §4.4,
§5.3, §5.4 and §7; spikes/p2p-simd/SPIKE_REPORT.md and its signed-off recommendation (the
formulation and the number of steps per ISA and precision); docs/phase3s/README.md
("Design decisions", "Machines"); the T3 code; the Intel Intrinsics Guide and the Arm
intrinsics reference for every intrinsic you use.

Do:
- In `arch::{neon, avx2}`, a crate-private trait over each ISA's f32 and f64
  vectors (design §5.3), with `#[inline(always)]` methods:
  - splat, load and store from a bounds-checked slice, add, sub, mul, fma, fnma,
    compare-equal giving a mask, and-not with a mask;
  - the inverse-square-root estimate and the refined inverse square root;
  - the block helpers the kernel needs: load W interleaved triples into three vectors
    and store them back (NEON `vld3q`/`vst3q`; on x86 a shuffle sequence or scalar
    inserts), and broadcast one source (3 coordinates and a charge).

  The scalar path implements the same trait with W = 1, so T5's generic body also runs
  on it as a check.
- The inverse square root per ISA and precision, exactly as signed off from T2:
  estimate, steps or polynomial, and the order of operations. Document each with its
  error bound and the measured error. If the spike's formulation misses the contract on
  a machine here, report the measured error and stop: do not add steps silently, since
  that changes the signed-off cost.
- Entry points per ISA and precision with `#[target_feature(enable = "neon")]` or
  `"avx2,fma"`, which the dispatch calls after `P2pKernel::new` has
  checked the ISA. For this task they are `rsqrt::rsqrt_slice(isa, x, out)` (design
  §5.1), which applies the kernel's inverse square root, with the r² = 0 mask, to a
  slice. Tails are handled by a padded stack copy, the same vector code.
- Unsafe, as in design §5.4: only in `arch` and the dispatch, each block with a
  `// SAFETY:` comment saying why the feature is present and why the access is in
  bounds. Use the safe forms that the toolchain check of T3 found stable wherever they
  remove a block.
- Inlining check: build the test binary in release and inspect the disassembly of each
  `rsqrt_slice` entry point. Its loop must contain no `call`/`bl` and only the expected
  instructions. Put the commands and the per-ISA instruction counts of the loop into
  the PR, and the commands into fmm-simd/CLAUDE.md.

Tests that define done (each prints the ISAs it ran):
- Contract, per available ISA and precision (design §4.3):
  - f32: every float in [1, 4) against `1/√x` evaluated in f64 and rounded, maximum
    relative error ≤ 4 u₃₂; an `#[ignore]` release test for the full set, and a
    strided subset (every 97th value) in debug;
  - f64: 10⁷ seeded log-uniform samples over the domain of §3.13 (as signed off in T1),
    against a double-double reference, maximum relative error ≤ 4 u₆₄; 10⁵ in debug;
  - powers of two and their neighbours, the domain ends, and r² = 0 (result 0);
  - the measured maximum error, in u_T, printed per ISA and precision.
- Scale invariance: rsqrt(4ᵏ x) = 2⁻ᵏ rsqrt(x) bit for bit over the domain, for sampled
  x. This is what lets the exhaustive f32 test cover one period only.
- Lane independence: a value's result does not depend on its lane or on the slice
  length (every length 0..=3W), bit for bit.
- Every vector-layer method against the scalar implementation, on random vectors,
  including the interleaved block load and store and the masks.
- Under Rosetta 2 (if T3 found that it works): the AVX2 tests also pass on the Mac
  with `--target x86_64-apple-darwin`. This is a quick check before CI only. Rosetta
  emulates the estimates, so the AVX2 contract counts only from the x86_64 CI leg.

Must pass: `cargo test -p nd-fmm-simd` (debug under 30 seconds) and
`cargo test -p nd-fmm-simd --release -- --ignored`:
- on the M3 Max;
- in the `run-tests-simd` CI job on both legs (T3). The x86_64 leg is the only x86_64
  hardware, so the AVX2 contract is proven there. Quote the measured maximum errors and
  the ISAs from its log in the PR.

Also `cargo clippy -p nd-fmm-simd --all-targets -- -D warnings` and the cross-target
clippy; `cargo doc -p nd-fmm-simd --no-deps` without warnings; the root checks. Report
which ISAs ran where.

Do not:
- write the P2P kernel (T5), or change `P2pKernel::evaluate`, which still runs the
  scalar path;
- use `unsafe` outside `arch` and the dispatch, or expose an unsafe public function;
- use a portable SIMD crate, or enable target features for the whole crate (no
  `-C target-feature` or `target-cpu` in a config file); features are per function;
- relax the contract to make a path pass.
