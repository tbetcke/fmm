# Phase 4 / T6 — the device P2P kernel (C4.2)

The near field on the device: the Laplace P2P with potential and gradient, f32 and f64,
as a `#[cube]` kernel in `nd-fmm-kernels`, used by the device operator for the P2P call.
After Phase 3S, the leaf stage, mostly P2P, is still 49–69% of a host evaluation at
p = 3 (design §7, "Recommendation for Phase 4"). The GPU kernel is timed against the
host SIMD kernel from the start.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("Requirements", "Accuracy measures", "Exit gate" C4.2);
- docs/design/device-path.md, signed off: the P2P part of section 6, sections 9 and 13;
- CONVENTIONS §3.1 and §3.13, with the signed-off T3 addition ("Device kernels");
- decision 10 (docs/phase4/README.md), signed off on 2026-10-03: the CPU runtime has a
  performance target, and this task adds the CPU layout below. Its prototype is T3's
  CPU-shaped P2P (`spikes/device-arith/src/cpu_p2p.rs`), with its measurements;
- spikes/device-arith/REPORT.md, "Recommendation": the device P2P contract and the
  formulation rules, signed off on 2026-10-03 (decision 3). No compiler option is
  required, and none exists;
- docs/design/simd-p2p.md §3 (the requirements of a fast P2P, which this kernel meets in
  its device form) and §8 (workloads);
- `nd_fmm_ref::p2p`, `nd_fmm_simd::P2pKernel` and their tests;
- `LaplaceOperator`'s P2P (`Kernels::p2p`, `p2p_target`: the near row, the mapping of
  s ≠ t sources);
- `nd_fmm_validate::p2p_kernels` and the `p2p_kernels` example (T7 of Phase 3S);
- the T4 and T5 code.

Do:
- **The kernel**, in `nd-fmm-kernels`, generic over f32/f64 (the runtime is a run-time
  value, device-path.md §3.2), with the structure the design fixes. By default:
  - one cube per target leaf (or block of targets), one unit per target point;
  - each source leaf of the target's near row in row order, staged through shared
    memory in tiles (tile size comptime and per backend), and each staged source in
    point order;
  - for s ≠ t, ŷ = ĉ(s|t) + r̂(s|t) u_s with the exact frames of §3.13, formed by an
    explicit `fma(r̂, u_s, ĉ)` and never written inline inside the subtraction; for
    s = t, u_s itself; then dₖ = u_t,k − ŷₖ (§3.13, "Device kernels");
  - r² = `fma(d₂, d₂, fma(d₁, d₁, d₀ · d₀))`, and pairs excluded by r² = 0: ρ =
    `inverse_sqrt(r²)`, then `select(r² == 0, 0, ρ)`, never a multiplication by a mask;
  - no Newton step; φ += q ρ and ∇φ −= (q ρ)(ρ ρ) d, written as `fma` calls;
  - two variants: potential only, and with gradients, the latter writing the §3.13
    output layout (φ̂ for every point, then the triples ĝ).

  Each target thus adds its sources in the order of its near row and, within a leaf,
  in point order: the order of the host path (requirement 4). Leaves larger than one
  tile, leaves with no points, and the last partial tile all run through the same
  code.
- **Plane-per-leaf layout**, a second GPU layout for small leaves (the Plummer sphere
  averages 17.6 points per leaf): one plane per target leaf (32 units on Metal) and
  several leaves per cube, each plane staging its own tile and syncing with
  `sync_plane`, with the same order and semantics. Both GPU layouts are candidates,
  for the timing below and for T12 (device-path.md §6.2).
- A safe launch wrapper for one level's P2P call: the near CSR, the source and target
  point offsets, the frames (or the keys they come from), and the device leaf stores.
  It adds into the device target output and allocates nothing beyond what the design
  allows.
- **In the device operator**: the P2P kind runs the kernel by default on every backend;
  the host fallback stays selectable.
- **CPU layout** (decision 10 set a CPU-runtime target, signed off on 2026-10-03). A
  second layout of the same kernel for the CPU runtime, selected by backend at
  construction (design §6.5), productionised from T3's CPU-shaped prototype:
  - targets in `Vector<T, N>` lanes with the host's vector width, and K vector blocks
    per unit as in T3: N = 4 and K = 2 in f32, N = 2 and K = 4 in f64 on NEON. One unit
    per core, each taking a contiguous range of target leaves. Sources broadcast in
    near-row order and mapped as above (the prototype took W1's absolute sources and
    did not map). No shared memory and no `sync_cube`;
  - the formulation above. On the CPU runtime `inverse_sqrt` is fl(1 / fl(√r²)), the
    same bits as the prototype's `sqrt` and division;
  - the same semantics as the GPU layout: each target adds its sources in near-row and
    point order, the r² = 0 rule, the signed-off contract; the tests below run on both
    layouts;
  - the threads rule of the design (section 11) for the runtime's own pool next to
    `threads(n)`;
  - timed on the M3 Max against `nd_fmm_simd::P2pKernel` (NEON) on the W1 workload,
    gathered form, f32 and f64, potential and gradient, at one thread (one unit) and on
    the 12 performance cores, in `p2p_kernels` (behind `gpu` and `cpu`). The target:
    the geometric mean of the per-pair time ratio at one thread is at most 1.5. T3's
    prototype measured 1.004. If the production layout misses the target, analyse the
    JIT output's inner loop (vector width, fma, inline `sqrt`/`fdiv`, spills) against
    the NEON kernel's, as T3 did (REPORT.md, "The JIT output"), and report; do not
    change the target;
  - the all-cores ratio is reported, not targeted. T3 measured 2.73×, dominated by the
    runtime's scheduling (190–230 µs per queued 12-unit launch, workers on the
    efficiency cores). Measure it with level-sized launches, not only T3's 768 leaves;
  - also time the leaf stage inside the FMM (CPU runtime against the host SIMD path,
    C3.2 cube at p = 3) and report it.
- **Timing**, on Metal f32 only, by an `nd-fmm-validate` row in `p2p_kernels` (behind
  `gpu` and `metal`), on the workloads of simd-p2p.md §8.2:
  - W1, FMM-shaped (n_t ∈ {8, …, 128}, both forms), and W2, all-pairs (N ∈ {10³, 10⁴},
    and 10⁵ on the GPU), potential and with gradients;
  - on the GPU, the W1 row launches at least 4,096 target leaves per launch (a
    level-sized launch, as on the C3.2 cube), not the 64-set pool of Phase 3S, which
    fills only 64 cubes (device-path.md §13.4);
  - pairs per second; the speed-up over host NEON (`P2pKernel::detect()`) at 1 thread
    and at 12 threads; the fraction of the C4.2 peak model of device-path.md §13.4
    (720 Gpairs/s φ and 480 φ and ∇φ, from 10 and 15 operations per pair);
  - the C4.2 target (device-path.md §13.4): at least 25% of the model on W2 at
    N = 10⁵ (180 and 120 Gpairs/s), and at least 10% on W1 at n_t = 64 (72 and 48
    Gpairs/s);
  - the accuracy of every row against `direct_sum`;
  - and the leaf-stage time inside the FMM (device operator, the C3.2 cube and the
    Plummer sphere at p = 3, f32), against the host leaf stage at 1 and 12 threads.

  Reported, never asserted. Below the design's target: analyse occupancy, shared-memory
  traffic, the inverse square root's share and the tile size, and report.

Tests that define done (CPU runtime f32 and f64; Metal f32 by hand, `#[ignore]`; each
test prints the backends it ran):
- Terms: one source and one target over seeded separations across the §3.13 domain,
  within the signed-off pair-term bounds against `nd_fmm_ref::p2p`: 8 u_T for the
  potential and 16 u_T per gradient component relative to |q| / r² (C3S.4, unchanged).
  On Metal, which flushes subnormals, the bounds apply to q = 0 or |q| ≥ 2⁻¹⁰⁰ (§3.13,
  "Device kernels"). T3 measured at most 4.6 and 10.3 u_T.
- Sums: seeded FMM-shaped leaf sets (a target leaf and its 26 neighbours, mapped as
  `LaplaceOperator` maps them, with neighbours on coarser and finer levels), n_t from 0
  to 3 tiles + 1, n_s up to 4,096, within the signed-off sum bounds against
  `direct_sum` (1e-14 and 1e-6 relative to the term magnitudes, or twice the
  reference's error), plus a cancelling set.
- Coincident points: targets equal to sources, duplicated sources, a mapped neighbour
  source that rounds onto a target, and the adversarial pairs of T3. Results are finite
  and equal the sum over the non-coincident pairs within tolerance.
- Frames: the device frames (ĉ, r̂) against `geometry::relative_frame`, bit for bit
  (device-path.md §6.1).
- Domain ends: pairs at 2⁻¹⁰⁸ and near 2⁷ (the domain carries over unchanged to device
  kernels), finite and within tolerance; the f32 gradient range as §3.13 states.
- Accumulation onto nonzero output; empty rows, empty leaves and an empty level are
  no-ops.
- Invariance and determinism, bit for bit:
  - a near row split into calls at every leaf boundary gives the same bits (chunk
    invariance per leaf);
  - permuting the target leaves of a level does not change any leaf's bits;
  - repeated launches give the same bits.
- Operator check, as T8 of Phase 3: the device operator's P2P on levels 2, 9 and 16
  against `nd_fmm_ref::p2p` at the absolute frames, within 1e-13 relative to the term
  magnitudes (f64, CPU runtime) and at the f32 bound.
- FMM: with P2P on the device and every other kind on the host fallback, every
  `tests/mpi_exec.rs` scenario on each backend run:
  - within the README's FMM bounds of the host output;
  - bit-identical across two evaluations;
  - the transfers per evaluation unchanged from T5 except as the design says.

  The ignored release gates of `tests/accuracy.rs` (C3.2) and `tests/adaptive.rs`
  (C3.3) pass on the device path (CPU runtime f64; Metal f32 by hand), with errors
  within 0.1% (f64) and 5% (f32) of the host run's.

Must pass:
- `cargo test -p nd-fmm-kernels --features cpu --release` (within the T4 budget) and
  `--features metal --release -- --ignored` by hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release`, also with
  `-- --ignored` for the gates, and on Metal by hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- clippy on the three crates, without features and with `--features cpu,metal`;
- `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report: the backends run; the maximum measured errors per backend and precision (terms
in u_T, sums); the timing tables; the fraction of the peak model against the C4.2
target; and the CPU layout's ratio to `nd-fmm-simd` at one thread, against the 1.5×
target, and on all cores. State that f64 was not timed on a GPU and that CUDA
was type-checked only.

Do not:
- change the formulation, tile structure or compiler options from what was signed off
  without asking first, even if a variant measures faster. Report it instead;
- use atomics, or change the accumulation order of any target;
- change §3.13, `nd_fmm_ref`, `nd-fmm-simd` or the host path;
- relax a tolerance to make a backend pass. If one fails, find whether the arithmetic,
  the mapping or the order is at fault, report it and stop.
