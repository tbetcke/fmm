# Optimisation of the operators (design for Phase 6)

Phase 6, task T1 ([docs/phase6/T1-optimisation-design.md](../phase6/T1-optimisation-design.md)).
Drafted on 2026-10-15 for sign-off by hand before T2. **Signed off by the user on
2026-10-15** (§10, "Outcome"): the document and its scope changes; no host BLAS, the
order-preserving kernel with an opt-in FMA mode; **SVD-compressed M2L (C6.2) dropped**
for now (§3 stays as the record of the measurement); C6.9's near rule in `nd-fmm-plan`
and its task T12. It measures where an evaluation
spends its time on the three machines (§1), designs the components that change the host
operator (the batched M2L §2, compressed M2L §3, several right-hand sides §4) and the
AVX-512 P2P (§5), states the device items (§6), dipoles (§7), the FMM3D comparison and the
plane-wave spike (§8), M2P and P2L only where they pay (§8a), orders the tasks by measured
or modelled gain (§9) and lists the questions for sign-off (§10). It is the Phase 6
counterpart of [device-path.md](device-path.md) (Phase 4) and
[distributed-fmm.md](distributed-fmm.md) (Phase 5), and answers "Requirements" and
"Decisions to sign off" 1–6 of [docs/phase6/README.md](../phase6/README.md).

Companion documents, cited by section: **plan**
([laplace-fmm-plan.md](laplace-fmm-plan.md)), **device-path**, **simd-p2p**
([simd-p2p.md](simd-p2p.md)), **node-m2l** ([node-m2l.md](node-m2l.md)), **scale-out**
([scale-out.md](scale-out.md)), **redesign** ([fmm-plan-redesign.md](fmm-plan-redesign.md)).

Every number is *measured*, with its machine, ranks × threads and command (§1.1,
Appendix A), or marked *model*. Release builds, every BLAS thread variable at 1 unless a
row says otherwise; raw output stays outside the repository. The path taken: **with
Kathleen** (README, "Kathleen is usable"): one Kathleen node (1 rank, 40 × 1, 2 × 20) is
part of every baseline; no job used more than one node.

## Summary

Measured on 2026-10-15 on the M3 Max (1 and 12 ranks), locust (1 and 72) and one Kathleen
node (1, 40 × 1 and 2 × 20; jobs 604311–604316), N = 10⁶, every M2L strategy, per-kind
timings (§1, Appendix A):

1. **At p ≥ 6 the far field is the cost, and today's dense M2L runs at 3.0–12 GFLOP/s
   per core on one rank in f64 (4–20 % of the f64 peak) and at 1.1–3.8 with one rank per
   core**, where the dense tables no longer fit the shared caches: Dense then takes
   2.2–6.9× the best strategy's time at f64 p = 8 (§1.2).
2. **The batched host M2L (C6.6, T2) keeps today's bits.** Offsets in index order,
   accumulated into each target's own local, no FMA: each local receives exactly
   `MatrixSet::apply`'s chain, at every thread count, rank count and ISA (§2.3; 0 values
   differing in every measured cell). Measured on a level, it is 6.7–15× faster than
   today's Dense at one rank per core and faster than Rotation in every cell but the M3
   Max at f64 p = 12; *modelled* end to end at f64 p = 8 with one rank per core 1.13×
   (locust), 1.20–1.30× (M3 Max) and 1.87–2.07× (Kathleen) against today's best
   strategy, and 1.5–3.0× at f32 p = 8 (§2.10).
3. **No BLAS** (decision 2): BLAS is 1.07–3.8× faster again at full load, but its bits
   change with the number of columns per call (so with the rank count), the CPU's kernel
   and the library; OpenBLAS called from 72 threads of one process collapses on locust;
   and no provider is linked today (the README said otherwise; corrected). The kernel
   lives in `nd-fmm-simd` with run-time ISA dispatch, which also gives the far field the
   AVX2 code Phase 5N asked for (2.1× the SSE2 build on Kathleen, §2.6, §2.11).
4. **M2P and P2L are 28–74 % of an evaluation on the Plummer sphere and 71–73 % on the
   cube at 2 × 10⁶** on every machine (§1.2, §8a): C6.9 is the general case of adaptive
   trees. A leaf W source below N_W = 46–963 points or a leaf X target below N_X = 15–460
   (the measured crossovers by machine and p) is cheaper by P2P, and a leaf holds at most
   64: *modelled* 1.36–1.6× on the Plummer sphere and 2.2× on that cube at f64 p = 8. It needs the plan to take a
   near rule (an `nd-fmm-plan` interface change) and a task of its own, **T12** (§8a).
5. **SVD compression (C6.2) pays only at high p** (dropped at sign-off): per offset, at a tolerance an order
   below the FMM's error, 1.10× the dense flops at p = 8, 0.53× at p = 12 and 0.31× at
   p = 18, with the tables 3.2× smaller at p = 18; shared bases do not compress at all
   (§3.1). Opt-in, budget 1.1× the dense errors, a deterministic Jacobi SVD (§3.2, §3.5).
6. **Several right-hand sides need no plan change** (sizes only), each vector bit for bit
   its single-vector evaluation; a CONVENTIONS §3.13 addition for the leaf layout (§4).
   **AVX-512** is stable since Rust 1.89 (1.99.0 in use) and joins as `Isa::Avx512`, not
   `Auto` until measured in the FMM at 40 ranks per node (§5). **Dipoles** change only the
   source side (§7). The **device** items: the fused gather ≤ 6 % of a CUDA evaluation, the
   row walk up to −30 % of the M2L at N = 10⁷ (§6).
7. **Order** (§9): T2, then T12 beside it, T6 after T2, T7 and T8 beside; T5 and T9
   after T2; T10 as a model first; T11 (T3 and T4 dropped with C6.2).

The questions for sign-off are §10.

## 1. Baseline

### 1.1 The machines, their caches and peaks

| machine | core | f64 peak per core (*model*: pipes × lanes × 2 × clock) | best measured `dgemm`, one core (Appendix B) | caches |
| --- | --- | --- | --- | --- |
| M3 Max | P-core, NEON, 4 × 128-bit FMA pipes (simd-p2p §4.6) | 16 flop/cycle × 4.05 GHz (simd-p2p §4.6, measured clock) = 64.8 GFLOP/s | 356 GFLOP/s at f64 p = 18, 64–1,024 columns (Accelerate, on Apple's matrix unit: above the NEON peak) | L1d 128 kB; L2 16 MB per 6-core cluster (32 MB for the 12 P-cores); system cache not reported (`sysctl hw`) |
| locust | Neoverse-V2, 4 × 128-bit FP pipes | 16 flop/cycle × 3.47 GHz (lscpu's maximum, tools/gh200/machine.md) = 55.6 GFLOP/s | 40 GFLOP/s at f64 p = 18, 1,024 columns (OpenBLAS 0.3.33, `neoversev2` kernels) | L2 1 MB per core; L3 114 MB shared |
| Kathleen | Xeon Gold 6248 (Cascade Lake), 2 × 512-bit FMA units | 32 flop/cycle × 2.5 GHz base = 80 GFLOP/s (lower under the AVX-512 licence, higher with one core busy) | 104 GFLOP/s at f64 p = 18, 1,024 columns (OpenBLAS 0.3.28, SkylakeX kernels, one core busy) | L2 1 MB per core; L3 27.5 MB per socket |

The default build is the target's baseline (the x86-64 baseline is SSE2: the far field's
loops on Kathleen run 128-bit vectors without FMA; `nd-fmm-simd`'s P2P alone dispatches
AVX2 at run time, phase5n-kathleen.md "Setup"). Dense tables per rank, 316 (p + 1)⁴ values
(the octant tables add 16 more): 0.65, 6.1, 16.6, 72 and 329 MB in f64 at p = 3, 6, 8, 12
and 18. With one rank per core: 199 MB on the M3 Max against 32 MB of L2, 1.2 GB on locust
against 114 MB of L3, 332 MB per Kathleen socket against 27.5 MB of L3 (f64 p = 8).

### 1.2 Where an evaluation spends its time

N = 10⁶, every strategy at one rank and at the node's ranks (Appendix A); the strategy
named is the fastest of Dense, Classes and Rotation; "Dense / best" is today's dense
tables against it; the shares are each stage's or kind's slowest rank over the wall time
(blocking path, `KindTiming::Synchronous`), so on several ranks they may add up to more
than 100 %. M2P is part of the leaves, P2L of the downward pass.

| machine | ranks × threads | problem | best (ms) | Dense / best | downward | leaves | M2L | P2P | M2P + P2L |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 Max | 1×1 | cube f64 p = 3 | Dense 1,155 | 1.00 | 31 % | 66 % | 30 % | 60 % | 0 % |
| M3 Max | 1×1 | cube f64 p = 8 | Rotation 6,537 | 1.25 | 82 % | 15 % | 81 % | 11 % | 0 % |
| M3 Max | 1×1 | cube f64 p = 12 | Rotation 15,827 | 1.95 | 89 % | 9 % | 88 % | 4 % | 0 % |
| M3 Max | 1×1 | cube f32 p = 8 | Dense 4,779 | 1.00 | 82 % | 14 % | 82 % | 7 % | 0 % |
| M3 Max | 1×1 | Plummer f64 p = 3 | Dense 2,450 | 1.00 | 33 % | 65 % | 23 % | 26 % | 47 % |
| M3 Max | 1×1 | Plummer f64 p = 8 | Rotation 14,649 | 1.18 | 67 % | 32 % | 57 % | 4 % | 35 % |
| M3 Max | 1×1 | Plummer f32 p = 8 | Dense 12,057 | 1.00 | 63 % | 36 % | 51 % | 3 % | 41 % |
| M3 Max | 12×1 | cube f64 p = 3 | Dense 123 | 1.00 | 28 % | 58 % | 28 % | 53 % | 0 % |
| M3 Max | 12×1 | cube f64 p = 8 | Rotation 610 | 4.55 | 79 % | 15 % | 79 % | 11 % | 0 % |
| M3 Max | 12×1 | cube f64 p = 12 | Rotation 1,453 | 8.14 | 88 % | 9 % | 87 % | 4 % | 0 % |
| M3 Max | 12×1 | cube f32 p = 8 | Rotation 528 | 2.44 | 83 % | 11 % | 83 % | 6 % | 0 % |
| M3 Max | 12×1 | Plummer f64 p = 3 | Dense 245 | 1.00 | 33 % | 61 % | 24 % | 24 % | 46 % |
| M3 Max | 12×1 | Plummer f64 p = 8 | Rotation 1,405 | 3.31 | 65 % | 33 % | 57 % | 4 % | 37 % |
| M3 Max | 12×1 | Plummer f32 p = 8 | Rotation 1,291 | 1.73 | 67 % | 33 % | 57 % | 3 % | 39 % |
| locust | 1×1 | cube f64 p = 3 | Dense 2,494 | 1.00 | 15 % | 82 % | 15 % | 79 % | 0 % |
| locust | 1×1 | cube f64 p = 8 | Rotation 9,333 | 1.05 | 72 % | 25 % | 72 % | 21 % | 0 % |
| locust | 1×1 | cube f64 p = 12 | Rotation 22,353 | 2.21 | 85 % | 13 % | 85 % | 9 % | 0 % |
| locust | 1×1 | cube f32 p = 8 | Dense 5,338 | 1.00 | 76 % | 20 % | 76 % | 13 % | 0 % |
| locust | 1×1 | Plummer f64 p = 3 | Dense 4,060 | 1.00 | 25 % | 74 % | 15 % | 43 % | 38 % |
| locust | 1×1 | Plummer f64 p = 8 | Rotation 19,627 | 1.02 | 64 % | 34 % | 55 % | 9 % | 32 % |
| locust | 1×1 | Plummer f32 p = 8 | Dense 14,067 | 1.00 | 58 % | 40 % | 45 % | 5 % | 45 % |
| locust | 72×1 | cube f64 p = 3 | Dense 38 | 1.00 | 17 % | 76 % | 17 % | 73 % | 0 % |
| locust | 72×1 | cube f64 p = 8 | Rotation 139 | 6.90 | 72 % | 27 % | 72 % | 21 % | 0 % |
| locust | 72×1 | cube f64 p = 12 | Rotation 400 | 11.15 | 86 % | 11 % | 86 % | 8 % | 0 % |
| locust | 72×1 | cube f32 p = 8 | Rotation 111 | 3.40 | 82 % | 15 % | 81 % | 9 % | 0 % |
| locust | 72×1 | Plummer f64 p = 3 | Dense 74 | 1.00 | 28 % | 72 % | 20 % | 62 % | 40 % |
| locust | 72×1 | Plummer f64 p = 8 | Rotation 295 | 5.23 | 81 % | 44 % | 75 % | 13 % | 47 % |
| locust | 72×1 | Plummer f32 p = 8 | Rotation 265 | 2.43 | 87 % | 42 % | 80 % | 5 % | 52 % |
| Kathleen | 1×1 | cube f64 p = 3 | Dense 2,501 | 1.00 | 21 % | 73 % | 21 % | 66 % | 0 % |
| Kathleen | 1×1 | cube f64 p = 8 | Rotation 16,758 | 1.23 | 82 % | 14 % | 82 % | 10 % | 0 % |
| Kathleen | 1×1 | cube f64 p = 12 | Rotation 44,016 | 2.70 | 90 % | 7 % | 89 % | 4 % | 0 % |
| Kathleen | 1×1 | cube f32 p = 8 | Dense 10,781 | 1.00 | 82 % | 14 % | 81 % | 6 % | 0 % |
| Kathleen | 1×1 | Plummer f64 p = 3 | Dense 5,811 | 1.00 | 27 % | 70 % | 14 % | 26 % | 54 % |
| Kathleen | 1×1 | Plummer f64 p = 8 | Rotation 37,765 | 1.17 | 69 % | 30 % | 57 % | 4 % | 35 % |
| Kathleen | 1×1 | Plummer f32 p = 8 | Dense 29,216 | 1.00 | 61 % | 37 % | 47 % | 2 % | 46 % |
| Kathleen | 2×20 | cube f64 p = 3 | Dense 122 | 1.00 | 21 % | 47 % | 21 % | 42 % | 0 % |
| Kathleen | 2×20 | cube f64 p = 8 | Rotation 601 | 1.18 | 77 % | 13 % | 77 % | 9 % | 0 % |
| Kathleen | 2×20 | cube f64 p = 12 | Rotation 1,550 | 2.83 | 87 % | 7 % | 86 % | 3 % | 0 % |
| Kathleen | 2×20 | cube f32 p = 8 | Dense 371 | 1.00 | 76 % | 13 % | 75 % | 6 % | 0 % |
| Kathleen | 2×20 | Plummer f64 p = 3 | Dense 266 | 1.00 | 30 % | 55 % | 16 % | 18 % | 52 % |
| Kathleen | 2×20 | Plummer f64 p = 8 | Rotation 1,451 | 1.11 | 67 % | 29 % | 51 % | 3 % | 40 % |
| Kathleen | 2×20 | Plummer f32 p = 8 | Dense 1,137 | 1.00 | 62 % | 35 % | 41 % | 2 % | 51 % |
| Kathleen | 40×1 | cube f64 p = 3 | Dense 95 | 1.00 | 24 % | 69 % | 23 % | 63 % | 0 % |
| Kathleen | 40×1 | cube f64 p = 8 | Rotation 616 | 2.69 | 82 % | 14 % | 81 % | 10 % | 0 % |
| Kathleen | 40×1 | cube f64 p = 12 | Rotation 1,695 | 4.15 | 90 % | 7 % | 89 % | 4 % | 0 % |
| Kathleen | 40×1 | cube f32 p = 8 | Rotation 528 | 1.46 | 85 % | 10 % | 85 % | 5 % | 0 % |
| Kathleen | 40×1 | Plummer f64 p = 3 | Dense 211 | 1.00 | 35 % | 70 % | 24 % | 38 % | 62 % |
| Kathleen | 40×1 | Plummer f64 p = 8 | Rotation 1,363 | 2.19 | 79 % | 38 % | 72 % | 5 % | 48 % |
| Kathleen | 40×1 | Plummer f32 p = 8 | Rotation 1,266 | 1.26 | 80 % | 40 % | 72 % | 2 % | 52 % |

At 160 Kathleen ranks (4 nodes, 40 × 1, Phase 5S T9's job 604159, the same shares):
the leaves take 42–79 % of an evaluation at p = 3 and 5–43 % at p = 8, the downward pass
67–92 % at p = 8 (N = 10⁸ strong and 2.5 × 10⁴ points per rank weak, the cube, the Plummer
sphere and the clusters; the README's figures, which phase5s-kathleen.md does not print).

Four findings:

1. **At p ≥ 6 the far field is the cost**: the downward pass is 54–90 % of an evaluation
   on the cube at p = 6–12 on every machine and split, and M2L alone 54–89 %.
2. **The dense tables do not scale on a node.** With one rank per core Dense takes
   2.2–6.9× the best strategy's time at f64 p = 8 and 3.5–11× at p = 12 (its M2L per V pair
   grows 3.3–8.9× from one rank to the node's at f64 p = 8, Appendix C), while on one rank
   it is within 1.0–1.25× of the best up to p = 8 (1.7–2.7× at p = 12); Rotation, the rule's
   choice there, is the best at p ≥ 6 with one rank per core. This is node-m2l's finding, unchanged after Phase 5S.
3. **Today's dense M2L runs at 3.0–12 GFLOP/s per core on one rank in f64 at p ≥ 6** (2 (p +
   1)⁴ flops per pair over the M2L time per pair: 11.5, 11.1 and 4.5 GFLOP/s at f64 p = 8 on
   the M3 Max, locust and Kathleen), 4–20 % of the f64 peaks of §1.1, and at 1.1–3.8 with one
   rank per core: 1.4–6 %.
4. **On the Plummer sphere, M2P and P2L take 28–74 % of an evaluation** (M2P 18–57 %, P2L
   8–22 %; Appendix C), more than P2P (1–62 %) in every cell at p ≥ 6, and more than M2L at
   p = 3. They are the W and X lists of the adaptive tree, and at f64 p = 8 one M2P target
   point costs 179–421 P2P pairs and one P2L source point 70–196, by machine (§8a.2).
   Phase 5S T9 found this on the half-refined cube at 2 × 10⁶ points (C6.9); the Plummer sphere shows it is the
   general case of an adaptive tree, not a corner case.

### 1.3 P2P on Kathleen (the AVX-512 baseline)

`p2p_kernels --ghz 2.5` on one core (job 604312, the geometric means over the gathered W1
cells of 8–128 targets): AVX2 at 2.95 / 1.74 Gpairs/s in f32 (φ / φ and ∇φ; 111 % and 91 %
of simd-p2p §4.6's AVX2 model at the base clock) and 0.82 / 0.56 in f64 (82 %, 72 %), every
accuracy check passed. Inside the FMM (`p2p_fmm --part kernels`, N = 10⁵, one thread) AVX2
makes the leaves 1.94–3.76× faster than the reference kernel on the cube and 1.14–1.56× on
the Plummer sphere (where M2P is part of the leaves). The fitted P2P constant at 40 ranks
per node is 3.09 ns (f64) and 1.35 ns (f32) per pair (`nd_fmm_exec::cost`, job 603049).

## 2. The host batched M2L (C6.6)

### 2.1 What it replaces

The host applies M2L target by target (`LaplaceOperator::m2l` → `Execution::boxes` →
`Kernels::m2l_target`): for target t, for each (source s, offset d) of its V row in
ascending d, `MatrixSet::apply(d, M_s, L_t)`, which adds A_d M_s into L_t column by
column, `L_t[r] ← L_t[r] + M_s[k] · A_d[r, k]` for k = 0 … n − 1, every product and sum
rounded separately (no FMA), n = (p + 1)². Each pair streams one n × n table (52 kB at
f64 p = 8) for 2n² flops: 0.25 flop per byte. The tables of a rank (316 offsets, 16.6 MB
at f64 p = 8) are read once per *target*; at many ranks per node they no longer fit the
shared cache and the downward pass stops scaling (node-m2l §2.3; §1.2 below).

### 2.2 The layout

Per level call (one `FmmOperator::m2l(M2l)` batch), with the plan's grouped view
(`VList`: `batch(d)` gives the pairs of offset d as (targets, sources), targets strictly
ascending, each at most once; redesign §4.3–§4.4):

```text
for each block B of consecutive targets (fixed size, below), in parallel:
    for d = 0 … 315:                                  (offsets in index order)
        pairs = the entries of batch(d) whose target lies in B   (two binary searches)
        for the pairs, NB at a time:
            C[:, j] ← C[:, j] + A_d · M[:, s_j]        (C[:, j] is L_{t_j}, in place)
```

The product is a GEMM with A_d as the left operand, the gathered multipoles as columns
and **the targets' own locals as the accumulator** (β = 1 on L, not a product added
afterwards). Two ways to feed it, both measured in §2.6:

- **direct** (recommended): the kernel reads M[:, s_j] and L[:, t_j] in place through the
  batch's index lists; it packs NB source columns k-major into a small buffer (n·NB
  values) and holds an MR × NB block of the locals in registers across k. No gathered
  copy of the level, no scatter.
- **gathered**: as the device (device-path §6.4), copy the chunk's source columns into X
  and the target locals into Y, multiply, copy Y back. Measured slower (the copies), kept
  only as the shape BLAS needs.

### 2.3 The order of additions, and why the bits do not change

Every element L_t[r] receives, in the target-by-target path, one chain of separately
rounded operations: for each offset d of t's row in ascending order, for k = 0 … n − 1,
`L_t[r] ← L_t[r] + M_s[k] · A_d[r, k]`. The batched layout:

1. visits the offsets in index order, and each target occurs at most once per offset, so
   every target meets its pairs in the order of its row (redesign §4.4);
2. accumulates **into L_t itself**, k ascending, one rounded multiply and one rounded add
   per step (no FMA): the kernel's register block is loaded from L_t before the k loop
   and stored after it;
3. computes each element of C from its own column of M and row of A_d only, so the
   result does not depend on which other columns share the product, on NB, MR, the block
   size or the thread count.

So each element's chain is exactly `MatrixSet::apply`'s: **the batched M2L is bit for bit
the target-by-target `Dense`**, at every thread count, chunking and rank count, and on
every ISA (IEEE-754 multiply and add are correctly rounded, so the SIMD and scalar forms
of the same chain agree). Measured: 0 values differing against the matvec loop in every
cell of §2.6 on the three machines. Requirement 3
(P ranks bit for bit one rank) holds because the plan already gives the same rows on P
ranks as on one (Phase 5 T6), and nothing here depends on the rank's set of targets.

An **FMA variant** (`acc ← fma(A_d[r, k], M_s[k], acc)`, the device's form) is
deterministic too, and bit-identical across thread counts and ISAs (`mul_add` is
correctly rounded everywhere), but not bit for bit `Dense`: each element differs by at
most n rounding errors of its partial sums (relative ≲ n u ‖A_d‖‖M_s‖/|L|); measured on
the level benchmark, max |Δ| / max |L| ≤ 1.6e-15 in f64 and 7.3e-7 in f32 (§2.6). Its gain is measured in §2.6; §10 question 2 asks whether T2 offers it.

### 2.4 Threading

Rayon splits each level call into **blocks of B consecutive targets** (B fixed, not
derived from the thread count; proposed 64) with
`par_chunks_mut(B · n)` on the level's locals, as `Execution::boxes` splits by one target
today. Each block runs the 316 offsets in order over its own targets, so every target is
written by one thread, meets its offsets in order, and the output is bit-identical for
every thread count (requirement 2) by §2.3 point 3 (blocks only decide which thread runs a
chain, never the chain). No `unsafe`, no atomics, no allocation per call: per-thread
scratch (below) is made at construction, as today.

The 2 × 316 binary searches per block (over each offset's `batch_targets`, a few thousand
targets on a level) cost about 1–2 ns per pair at B = 64 (*model*: about 140 pairs per
target on the benchmark's level), negligible at p ≥ 6 (0.5–5 µs per pair) and up to about
7 % at p = 3 (25 ns); T2 can precompute the ranges per (block, offset) when the level is
first seen (2 × 316 `u32` per block) and remove it. A block holds about 189 pairs per target on a
uniform interior, so each table is read once per block instead of once per target: a
factor of up to B fewer table reads per pair at full occupancy of the offsets (*model*;
the measured effect is §2.6).

### 2.5 Scratch per thread

| form | per thread, values | f64 p = 8 | f64 p = 18 |
| --- | --- | --- | --- |
| direct | n·NB (packed columns) + 2·B indices (`u32`) | 2.6 kB + 0.5 kB | 11.6 kB + 0.5 kB |
| gathered, C columns per product | 2·n·C (X and Y) + C indices | 332 kB at C = 256 | 1.5 MB at C = 256 |

With NB = 4 and B = 64. The direct form needs no budget; the gathered form (BLAS only)
caps C by a scratch budget per thread, C = ⌊S / (2 n · size_of(T))⌋, as the device's
`PlanSettings::max_columns` does for its chunks (`fmm-kernels/src/translate.rs`), with
S = 256 kB proposed (L2-resident on every machine: 1 MB per core on locust and
Kathleen, 16 MB per 6-core cluster on the M3 Max).

### 2.6 BLAS or a hand-written kernel (decision 2), measured

The throwaway benchmark of Appendix B runs §2.2 on a uniform level 4 (4,096 boxes, 584,136
V pairs, the 316 offsets) with random tables, the level call of today's host path as its
first variant. That variant reproduces the FMM's own M2L per pair under the same load
(Kathleen 40 processes f64 p = 8: 9.60 µs against the FMM's 9.56 µs at 40 × 1; p = 12:
43.6 against 42.8; the M3 Max 12 processes: 4.92 against 5.16; locust 72: 9.07 against
10.50), so the level benchmark is a fair model of the level call.

**M2L per V pair, µs** (the median over the processes; "node" is one process per core,
each with its own tables, as ranks; no-FMA is the order-preserving kernel of §2.3, "8x4"
its register block; Kathleen's hand-written rows from a `-C target-cpu=native` build,
standing for the AVX2 kernel the dispatch of §2.11 gives, with today's SSE2 build in
brackets):

| machine | processes | precision, p | target by target (today) | batched, no FMA (bit for bit) | batched, FMA | batched, BLAS (gathered 256) | Rotation in the FMM (§1.2) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| M3 Max | 1 | f64 8 | 1.14 | 0.52 | 0.38 | 0.145 | 0.88 |
| M3 Max | 1 | f64 12 | 4.75 | 2.11 | 1.56 | 0.354 | 2.32 |
| M3 Max | 1 | f32 8 | 0.65 | 0.29 | 0.25 | 0.063 | 0.81 |
| M3 Max | 12 | f64 8 | 4.92 | 0.66 | – | 0.25 (max 0.59) | 0.94 |
| M3 Max | 12 | f64 12 | 22.1 | 2.59 | – | 0.88 | 2.47 |
| M3 Max | 12 | f32 8 | 2.21 | 0.33 | – | 0.087 (max 0.29) | 0.85 |
| locust | 1 | f64 8 | 1.18 | 0.68 | 0.52 | 0.42 | 1.11 |
| locust | 1 | f64 12 | 7.34 | 2.78 | 2.03 | 1.66 | 3.14 |
| locust | 1 | f32 8 | 0.69 | 0.38 | 0.27 | 0.21 | 1.06 |
| locust | 72 | f64 8 | 9.07 | 0.97 | – | 0.91 | 1.15 |
| locust | 72 | f64 12 | 49.9 | 3.27 | – | 2.18 | 3.98 |
| locust | 72 | f32 8 | 3.96 | 0.42 | – | 0.26 | 1.04 |
| Kathleen | 1 | f64 8 | 2.96 | 0.71 (1.49) | 0.52 | 0.25 | 2.27 |
| Kathleen | 1 | f64 12 | 18.3 | 2.75 (6.35) | 2.06 | 0.94 | 6.52 |
| Kathleen | 1 | f32 8 | 1.45 | 0.40 (0.65) | 0.35 | 0.13 | 2.03 |
| Kathleen | 40 | f64 8 | 9.6 | 1.13 (2.16) | 0.86 | 0.56 | 3.11 |
| Kathleen | 40 | f64 12 | 43.6 | 4.25 (9.0) | 3.26 | 2.11 | 9.46 |
| Kathleen | 40 | f32 8 | 4.4 | 0.58 (0.93) | 0.52 | 0.25 | 2.78 |

(The FMA rows at the node's processes on the M3 Max and locust were not run; Appendix B
has the p = 3 and 6 rows.)

What the measurements say:

1. **The batched layout removes the table pressure.** With one process per core the
   order-preserving kernel loses only 1.2–1.6× against one process (the M3 Max 0.52 → 0.66,
   locust 0.68 → 0.97, Kathleen 0.71 → 1.13 at f64 p = 8), where target by target loses
   3.2–7.7×. It is 6.7–15× faster than today's Dense at the node's ranks, and faster than
   Rotation everywhere except the M3 Max at f64 p = 12 (2.59 against 2.47 µs).
2. **The bits are kept.** The no-FMA kernel gave 0 values differing from target by target
   in every cell on the three machines, directly and gathered, at 1 and at 12–72 threads.
3. **BLAS is faster, by 1.07–3.8× at the node's ranks** (Kathleen 2.0–2.3×, locust
   1.07–1.6×, the M3 Max 2.7–3.8× in the median), but:
   - **its bits depend on the shape of the call**: the product of 4,096 columns at once
     against the same product in chunks of 1, 3, 7, 64 or 333 columns differs in up to
     all values for Accelerate (every p), OpenBLAS on locust (chunks of 1–7) and OpenBLAS
     on Kathleen (p ≥ 8, chunks of 1–333); a rank's columns per offset depend on which
     targets it owns, so a BLAS M2L breaks requirement 3 (P ranks bit for bit one rank)
     and, unless the calls are cut independently of the threads, requirement 2;
   - **on the CPU's kernel**: forcing OpenBLAS's Haswell kernels on Kathleen changes the
     hash of the same product at p = 8 and 12 (bits differ between machines with the same
     library build);
   - **not cured by a reproducibility mode**: MKL's `MKL_CBWR` (COMPATIBLE, AVX2) keeps
     each call reproducible but the chunking still changes the bits (Appendix B);
   - **on the build in one process**: spack's OpenBLAS 0.3.33 (`threads=none`, which
     locks around each call) collapses when 72 threads of one process call it (at f64
     p = 3, 39 ns per pair against 2.3 for the hand-written kernel; at p = 8, 40 against
     14 ns), the case of BLAS inside rayon workers;
   - **Accelerate's speed is Apple's matrix unit**, one per cluster: with 12 processes the
     slowest takes 2.4× the median (0.59 against 0.25 µs);
   - **no BLAS is linked today**: rlst calls the Fortran symbols through the `blas` crate
     and leaves the provider to the final binary (its `blas-src`/`lapack-src` are
     dev-dependencies; `Cargo.lock` has no `*-src` crate; tools/kathleen/README.md "The
     workspace links no BLAS"), so BLAS means a new dependency or a link line in a build
     script (README: ask first). The README's "rlst 0.9.0 already links the system BLAS"
     is out of date (corrected in this PR, Appendix D).
4. **FMA buys 1.1–1.4× on the kernel** (one process) and changes the bits against Dense by
   at most 1.6e-15 relative (f64) and 7.3e-7 (f32) on the level (max |Δ| over max |L|);
   it is as deterministic as the no-FMA kernel (each element one `fma` chain, k
   ascending; `mul_add` is correctly rounded on every ISA, so the scalar fallback gives
   the same bits) and keeps requirement 3.
5. **The far field is built for the baseline today.** On Kathleen the default build (SSE2)
   runs the kernel at half the speed of an AVX2 build (1.49 against 0.71 µs, f64 p = 8,
   one process), Phase 5N's "far-field build for the CPU" question (node-m2l §9 q. 7).

**Recommendation (decision 2): the hand-written order-preserving kernel**, in
`nd-fmm-simd` behind its run-time ISA dispatch (§2.11), no BLAS. It keeps every bit of
today's Dense (so the tests are bit for bit, requirement 3 holds without a new bound), it
needs no dependency, it beats Rotation at f64 p = 8 on every machine at full load, and it
answers the far-field build question without a global `target-cpu`. The FMA variant is a
second, opt-in mode of the same kernel (§10 q. 2c); BLAS stays recorded as the faster,
non-reproducible option (2.0× on Kathleen at the node's ranks) for a later, relaxed mode.

### 2.7 M2M and L2L

Under the dense octant tables (Dense and Classes) M2M and L2L together take 0.14–1.15 %
of an evaluation on every machine, problem, p and split of §1.2: their 8 tables per family
(0.4 MB at f64 p = 8) stay in cache and every pair reuses them. The batched form (octants
in place of offsets, the same kernel; bit for bit, since a parent meets its children in
octant order and a child has one parent) can save at most that 1 %. **Recommended: not
in T2** (decision 2's second half); T2 may reuse the kernel for them if it costs no new
code path to test, and reports the kind times either way.

### 2.8 Classes and Rotation

`Classes` applies T_L(P) A_c T_M(Pᵀ) per pair with two O(p³) coefficient transforms
around a dense product into fresh scratch (`M2lClasses::apply`). Its dense part could run
through the same kernel (columns T_M m_s gathered, the class matrix, T_L added per target
in row order: bit for bit `Classes`), but the transforms stay per pair (2 · 969
multiply–adds against 6,561 at p = 8), and its purpose, memory, is served better by §3.
**Not in T2.** `Rotation` is O(p³) per pair with small tables (0.83 MB at f64 p = 8) and
has no GEMM form; it is unchanged and stays the comparison.

### 2.9 Naming and the strategy rule

A fifth value **`M2lStrategy::Batched`**: the dense tables of `Dense`, applied per level
by §2.2–§2.4 on the host. On the device it runs as `Dense` (the device's dense M2L is
already the gathered GEMM of device-path §6.4) and is reported so. `Dense` stays the
target-by-target trusted path (requirement 4) and `Auto` is unchanged in T2 (Phase 5N's
rule B); T11 revisits it with §2.10's model as the hypothesis: Batched in place of Dense
wherever rule B picks Dense, and in place of Rotation where its tables fit the rank's
memory. `Fmm::strategy()` and the threading report name it; `KindTiming` and the cost
model's M2L constant cover it unchanged (one level call per kind, as today).

### 2.10 Expected gain (*model*)

*Model*: each evaluation of §1.2 with the best strategy's M2L replaced by the batched
kernel's time per pair of §2.6 (one process, or the node's processes, the same p and
precision), every other stage as measured; the pairs from the M2L time and the time per
pair of the best strategy. It assumes every level behaves like the benchmark's level 4
(the top levels have fewer columns per offset; they are a few per cent of the pairs).
Kathleen with the AVX2 dispatch (the native build's numbers; at p = 3 at 40 processes
only the SSE2 build was measured). The evaluation against today's best strategy:

| machine | ranks × threads | problem | best today (ms) | Batched, bit for bit (ms, ×) | FMA (ms, ×) | BLAS (ms, ×) |
| --- | --- | --- | --- | --- | --- | --- |
| M3 Max | 1 × 1 | cube f64 p = 8 | Rotation 6,537 | 4,346 (1.50) | 3,537 (1.85) | 2,106 (3.10) |
| M3 Max | 1 × 1 | cube f32 p = 8 | Dense 4,779 | 2,595 (1.84) | 2,371 (2.02) | 1,242 (3.85) |
| M3 Max | 12 × 1 | cube f64 p = 8 | Rotation 610 | 468 (1.30) | – | 255 (2.39) |
| M3 Max | 12 × 1 | cube f64 p = 12 | Rotation 1,453 | 1,516 (0.96) | – | 639 (2.27) |
| M3 Max | 12 × 1 | cube f32 p = 8 | Rotation 528 | 259 (2.04) | – | 135 (3.90) |
| M3 Max | 12 × 1 | Plummer f64 p = 8 | Rotation 1,405 | 1,174 (1.20) | – | 821 (1.71) |
| locust | 1 × 1 | cube f64 p = 8 | Rotation 9,333 | 6,743 (1.38) | 5,746 (1.62) | 5,196 (1.80) |
| locust | 72 × 1 | cube f64 p = 8 | Rotation 139 | 123 (1.13) | – | 118 (1.18) |
| locust | 72 × 1 | cube f64 p = 12 | Rotation 400 | 338 (1.18) | – | 244 (1.64) |
| locust | 72 × 1 | cube f32 p = 8 | Rotation 111 | 57 (1.95) | – | 43 (2.57) |
| locust | 72 × 1 | Plummer f64 p = 8 | Rotation 295 | 261 (1.13) | – | 250 (1.18) |
| Kathleen | 1 × 1 | cube f64 p = 8 | Rotation 16,758 | 7,352 (2.28) | 6,199 (2.70) | 4,550 (3.68) |
| Kathleen | 40 × 1 | cube f64 p = 8 | Rotation 616 | 298 (2.07) | 255 (2.42) | 207 (2.98) |
| Kathleen | 40 × 1 | cube f64 p = 12 | Rotation 1,695 | 864 (1.96) | 706 (2.40) | 522 (3.25) |
| Kathleen | 40 × 1 | cube f32 p = 8 | Rotation 528 | 175 (3.02) | 165 (3.21) | 123 (4.31) |
| Kathleen | 40 × 1 | Plummer f64 p = 8 | Rotation 1,363 | 730 (1.87) | 646 (2.11) | 552 (2.47) |
| any | any | f64 p = 3 | Dense | 1.07–1.21× | 1.09–1.25× | 1.08–1.24× |

(Appendix C has every cell.) At f64 p = 8 with one rank per core, the bit-for-bit
batched M2L is *modelled* at 1.13× (locust) to 2.07× (Kathleen, where today's far field
runs SSE2) against today's best strategy, and 1.95–3.0× at f32 p = 8; at p = 3 M2L is
14–30 % of an evaluation and the gain is small. The one place it does not win is the M3
Max at f64 p = 12 with 12 ranks (0.96×; Rotation's O(p³) catches up). C6.6's criterion
("faster than every existing host strategy at p ≥ 6 on one rank, and scaling with ranks
per node where Dense does not") is met by the model everywhere but that cell; T2 measures
it (measured in T2: fmm-validate/results/phase6-m2l.md, its §7 against this model). At 160 Kathleen ranks, where M2L is most of the 67–92 % downward share at p = 8 (§1.2),
the same per-pair gain applies per rank.

### 2.11 Where the kernel lives

`nd-fmm-exec` is free of `unsafe` (root CLAUDE.md), and calling a
`#[target_feature]` function from code without the feature is `unsafe`, so the run-time
ISA dispatch lives where it already does for P2P: a new module of `nd-fmm-simd`,

```rust
/// C[:, c_j] += A · B[:, b_j] for each j, A n × n column-major, columns of length n,
/// each element as `MatrixSet::apply` (k ascending, multiply and add rounded apart).
pub fn accumulate_columns<T: SimdScalar>(
    isa: AvailableIsa, n: usize, a: &[T], b: &[T], b_offsets: &[usize],
    c: &mut [T], c_offsets: &[usize], pack: &mut [T],
);
```

with `arch::{scalar, neon, avx2}` bodies (and `avx512` with T6), `unsafe` only in `arch`
and the dispatch, safe public functions, and a test of every ISA against the scalar body
bit for bit (the same chain on every ISA). `nd-fmm-exec` calls it from the level call with
the `Fmm`'s P2P ISA choice (`P2pChoice`, which already selects the widest ISA). This
widens T2's crates by `nd-fmm-simd` (a sign-off point: README "Scope" lists only T6 there)
and touches the files T6 changes (`isa.rs`, `arch/mod.rs`): merge one, then rebase the
other.

## 3. SVD-compressed M2L (C6.2)

**Dropped at sign-off (2026-10-15)**: the gain (none at p ≤ 8, 0.53× and 0.31× of the dense
flops at p = 12 and 18) does not justify the task now; C6.2 is not built in Phase 6 and T3
and T4 are dropped. This section records the measurement and the design for a later phase.

### 3.1 How compressible the tables are, measured

The 316 dense f64 tables (`M2lTables::build`, dumped by a throwaway program) and their
singular values (numpy, f64), at relative 2-norm tolerances ε (σ_{k+1} ≤ ε σ₁ of each
table); Appendix B has every ε. Two forms:

- **per offset**, A_d ≈ U_d Σ_d V_dᵀ with its own rank k_d: an M2L costs two thin products,
  2 n k_d multiply–adds against n²;
- **shared bases** (Fong–Darve's form, laplace-fmm-plan §3.6): one U and one V for all 316
  offsets, A_d ≈ U C_d Vᵀ with k × k cores, k² per pair plus a basis change per box.

| p | n | per offset: ε | mean k / n (max) | flops per pair vs dense | memory vs dense | shared bases: k / n at the same ε |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 8 | 81 | 1e-5 | 0.43 (0.74) | 0.86 | 0.86 | 1.00 |
| 8 | 81 | 1e-6 | 0.55 (0.83) | 1.10 | 1.10 | 1.00 |
| 12 | 169 | 1e-6 | 0.20 (0.47) | 0.41 | 0.41 | 1.00 |
| 12 | 169 | 1e-7 | 0.27 (0.57) | 0.53 | 0.53 | 1.00 |
| 18 | 361 | 1e-8 | 0.13 (0.29) | 0.25 | 0.25 | 1.00 |
| 18 | 361 | 1e-9 | 0.15 (0.38) | 0.31 | 0.31 | 1.00 |

Shared bases do not compress at any tolerance that matters (k = n at p ≤ 8 for every ε
down to 1e-3; at p = 18 k/n = 0.55 only at ε = 1e-3): the near offsets (|d| = 2) need the
full rank, and one basis serves them all. Per offset the far tables compress better than the near ones
(k max at |d| = 2 is 1.2–1.7× k max at the farthest offsets at the tolerances below, up to
6× at ε = 1e-4).

### 3.2 The error budget (decision 3)

The FMM's own truncation error at p (laplace-fmm-plan §4, Phase 3's calibration, the
worst of the four distributions): about 1e-4 at p = 7, 1e-5 at p = 9, 1e-6 at p = 12,
1e-8 at p = 19 in f64 (φ, relative L2); on the cube 2.6e-3, 1.8e-5 and 1.0e-8 at p = 3, 8
and 18. A truncation error of ε relative to ‖A_d‖₂ on every table adds at most about ε
‖M‖ to each local's contribution, so ε an order below the FMM's error at p keeps the
compressed FMM's error within about 10 % of the dense one's.

**Proposed budget**: the compressed FMM's errors against the direct sum (φ and ∇φ,
relative L2, the C3.2 cube and C3.3 Plummer sphere, 8 vectors) at most **1.1×** the dense
FMM's at the same p, and against the dense FMM at most **0.1 × the dense FMM's own
error**; opt-in until T11 (README requirement 1, "states its own error budget").

**Proposed truncation rule**: a relative singular-value cut per offset, ε given by the
user (`M2lStrategy::Compressed { tolerance }`, or a builder setting), with a default
ε(p) table that T3 fits to the budget on its measurements (the *model* above suggests
ε ≈ 0.1 e_p: 1e-6 at p = 8, 1e-7 at p = 12, 1e-9 at p = 18). A rank per p instead of a
tolerance is rejected: the ranks vary about 2× between offsets (mean against maximum).

### 3.3 What it buys, and where

Flops per pair (f64; rotation (p + 1)(10p² + 32p + 9)/3 multiply–adds, `fmm-tables`'s
rotation module; dense 2n²; compressed 4 n k̄ at the ε above):

| p | dense | compressed | rotation | dense tables per rank | compressed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 13,122 | 14,400 (ε 1e-6) | 5,430 | 16.6 MB | 18 MB |
| 12 | 57,122 | 30,300 (ε 1e-7) | 15,886 | 72 MB | 38 MB |
| 18 | 260,642 | 80,800 (ε 1e-9) | 48,450 | 329 MB | 102 MB |

So compression **does not pay at p ≤ 8** (no flops saved at a tolerance below the FMM's
error) and is a **high-p and memory option**: 0.53× the dense flops at p = 12, 0.31× at
p = 18, and the tables of p = 18 at 102 MB instead of 329 MB per rank (laplace-fmm-plan
§9.1's "400 MB at p = 19"). Rotation has fewer flops at every p but runs at 6–7 GFLOP/s per
core (§1.2: 2.32 µs per pair at f64 p = 12 on the M3 Max); the compressed products run at
the batched kernel's rate if the thin shapes keep it (*model*: at p = 12 on the M3 Max
0.53 × 2.11 = 1.1 µs per pair against Rotation's 2.32; at p = 18 about 3 µs against about
7, from the kernel's 27 GFLOP/s at p = 18 and Rotation scaled by its flops), which T3 measures. The device has the same trade with its own rates (device-path
§18.2: dense levels at 14–23 % of the H100's f64 peak, rotation 1.5–3.4 %), where the
memory matters more (`SettingsError::DeviceMemory` at high p).

### 3.4 The application

Per offset in the batched layout of §2.2: the k_d × cols intermediate T = V_dᵀ M (fresh,
from zero, the order-preserving kernel), then L += (U_d Σ_d) T (accumulated into the
locals, as §2.3). Each target still meets its offsets in index order; the result is
bit-identical for every thread count and rank count (§2.3's argument: each element's chain
depends only on its own column), not bit for bit Dense (a different operator). The
factors are stored as U_d Σ_d (n × k_d) and V_dᵀ (k_d × n), contiguous per offset, f32
rounded from the f64 factors. Device (T4): per chunk the gather as today, two grouped
GEMMs with per-group inner dimension k_d (the tile schedule carries it; or k padded with
zero columns to the chunk's largest k_d, which adds exact zeros and keeps the bits), then
the reduction in row order.

### 3.5 The build and the cache

Per offset an SVD of an n × n matrix (n ≤ 441 at p = 20), in f64, at table build, outside
any rayon worker, then cached:

- `TableKind::M2lCompressed` (code 6) and a **tolerance in the key**: `CacheKey` today is
  (kind, p, precision), with no field for it, so the key and the file name gain the
  tolerance (its f64 bits) and `FORMAT_VERSION` goes from 1 to 2 (old files are rejected
  and rebuilt, as for any version change);
- **the SVD itself** (decision 3b): rlst's `svd` calls LAPACK's `?gesdd`, which, like BLAS
  (§2.6), is not linked today, so LAPACK needs a provider (a new dependency, or a link
  line), and its bits depend on the library; or a hand-written one-sided Jacobi SVD in
  `nd-fmm-tables` (no dependency; deterministic by construction, like the dense tables;
  about 3n³ flops per sweep, *model* a few minutes at p = 18 single-threaded, once per node
  with `table_cache`). **Recommended: the Jacobi SVD**, keeping `nd-fmm-tables` free of a
  native library and its tables the same on every machine; LAPACK if the build time at
  p ≥ 18 turns out to matter (T3 measures).

Level independence holds as for the dense tables (CONVENTIONS §3.12: the tables act on
scaled coefficients).

## 4. Several right-hand sides (C6.3)

### 4.1 The plan carries it unchanged

Every size of the plan is the operator's (`FmmSizes`): the stores are column-major
`size(l) × len(l)` matrices (`LevelBuffers`), the leaf store is CSR with
`count · point_size` values per leaf, the source exchange moves `count · point_size` per
leaf, the reduction, the replicated top and the multipole exchange `size(l)` per box, and
`Redistribution::{forward, backward}` take any width per item. With m vectors the
operator reports

| size | one vector | m vectors |
| --- | --- | --- |
| `multipole_size(l)`, `local_size(l)` | n = (p + 1)² | m n: the box's n × m block, vector after vector |
| `source_point_size()` | 4 (u, then q) | 3 + m |
| `target_output_point_size()` | 1 or 4 | m or 4m |

so **no `nd-fmm-plan` change** (README, "Every operator behind the `FmmOperator`
interface"): one message per neighbour and exchange as today, m times the values; the
replicated top's all-gather-v grows in its bandwidth term (0.62 and 2.06 µs per rank at
0.26 and 2.65 MB on Kathleen, phase5s-kathleen §1.2), reported per vector by T5.

The leaf layout inside a chunk is CONVENTIONS §3.13's ("Source chunks": 4n values, the
coordinates then the charges; "Target input and output": n or 4n). m vectors need an
addition there, **proposed in T5's PR, never made in code**: the charges point-major
(the m charges of point i adjacent, `q[i·m + v]`), so P2P broadcasts one source's m
charges and P2M adds one harmonic vector into m columns; the output likewise point-major
inside the leaf store, transposed to vector after vector by the output pass.

### 4.2 The interface

```rust
impl<T> FmmBuilder<T> {
    /// The charge vectors of every evaluation (default 1); sizes the stores at build.
    pub fn vectors(self, m: usize) -> Self;
}
impl<T> Fmm<T> {
    /// m vectors of N charges each (the caller's points, as `evaluate`).
    pub fn evaluate_many(&mut self, charges: &[&[T]]) -> Result<OutputMany<T>, FmmError>;
    pub fn evaluate_many_into(&mut self, charges: &[&[T]], output: &mut OutputMany<T>)
        -> Result<(), FmmError>;
}
/// Vector v's potential is `potential[v N .. (v + 1) N]`, its gradient likewise.
pub struct OutputMany<T> { pub vectors: usize, pub potential: Vec<T>,
                           pub gradient: Option<Vec<[T; 3]>>, pub timings: StageTimings }
```

`evaluate` stays as it is, bits included; on an `Fmm` built with m > 1 it is refused
(`FmmError::VectorCount`, agreed in the charge-length all-reduce that every evaluation
already makes). `evaluate_many` with m = 1 is `evaluate`. The charges are packed
point-major once, in the forward's input buffer (one O(mN) copy on the pool; the
forward already copies). The device path uploads m charges per point and downloads m or
4m outputs (device-path §14's formula: one upload, one download and one sync per
evaluation, m times the bytes).

### 4.3 What changes per operator, and the bits

Each vector's output is **bit for bit its single-vector evaluation** if every operator
keeps the single-vector chain per vector and shares only what does not depend on the
charge. That holds operator by operator:

| operator | shared across the m vectors | per vector, unchanged chain |
| --- | --- | --- |
| P2M, P2L | the harmonics R̄(u), Ī(u) of each point (`leaf::p2m` computes them, then `add_conjugate(q, …)`) | `add_conjugate` with q_v into column v |
| M2M, M2L, L2L | the table, read once for m columns | §2.3's chain per column (batched host, device GEMM, or target by target) |
| L2P, M2P | the harmonics at each target | the dot product with column v |
| P2P | d, r², ρ per (target, source) | φ_v = fma(q_v, ρ, φ_v); t = q_v ρ, … as `arch::p2p::block` |
| output | the scale per leaf | the division per value |

P2P needs a kernel that takes m charges per source: in `nd-fmm-simd`
(`P2pKernel::evaluate_many(sources, charges /* m per source */, m, targets, …)`), the
source loop of `arch::p2p::block` with an inner loop over v, `unsafe` only in `arch`.
The device kernels likewise. T5 tests the bit identity per vector (README C6.3 criterion).

### 4.4 Expected gain (*model*)

Per vector, an operator whose cost is a shared part plus a per-vector part costs
(shared + m · per vector)/m. From the operator shapes: M2L batched reads each table once
for m·(pairs) columns, so its per-vector cost falls to the GEMM's compute bound (§2.6's
kernel rate) once m ≥ NB; P2P on AVX2 counts 15 / 20 operations per pair and
lane for φ in f32 / f64 and 21 / 26 with ∇φ (simd-p2p §4.6, matched by the compiled
loops), of which one (φ) or about six (φ, ∇φ: t = q ρ, the add into φ, t ρ², three
fnma) depend on the charge, so per vector up to 15–20× (φ) and 3.5–4× (φ, ∇φ) at large m; P2M and L2P
share the harmonic recursion (O(n) per point) and keep O(n) per vector, about 2×. With
§1's stage split (leaves 42–79 % at p = 3, downward 67–92 % at p = 8, 160 Kathleen ranks)
the evaluation per vector at m = 8 is *modelled* at 0.3–0.5 of one vector's at p = 3 and
0.6–0.8 at p = 8 with Batched (the M2L already near its compute bound). T5 measures it at
m = 1, 2, 4, 8, 16, 32.

### 4.5 As built in T5

The interface of §4.2 (`FmmBuilder::vectors`, `Fmm::evaluate_many`, `evaluate_many_into`,
`OutputMany`, `FmmError::VectorCount`), the plan unchanged (§4.1), and every vector bit for
bit its single-vector evaluation (§4.3) on the host, the CPU runtime and CUDA. What
differs from the proposal above:

- **The target output is vector after vector, not point-major** (CONVENTIONS §3.13,
  "Several charge vectors", proposed in the PR): a leaf's chunk holds the m vectors' φ̂ one
  after the other, then their ĝ. P2P keeps the targets in vector lanes (simd-p2p §4.1), so
  it loads and stores a vector's potentials of W consecutive targets directly; point-major
  outputs would need a strided gather per block. Each vector's values are then the
  single-vector chunk's, which the reference P2P (vector by vector) and the device's output
  pass read unchanged. The charges are point-major as proposed (`q[i m + v]`): the forward
  moves them in one piece per leaf, and P2P and P2M read a source's m charges together.
- **P2P** (`P2pKernel::evaluate_many`, `nd-fmm-simd`): the vectors in register groups,
  ρ formed once per source and target vector per group (NEON: blocks of 2 target vectors
  with groups of 8 vectors without gradients, 1 and 4 with; AVX2: 1 and 8, 1 and 4, each
  chosen by timing), each vector then the single-vector kernel's steps. Its gain is
  capped by the registers: per vector at m ≥ 8, on the M3 Max 4.6× (f32) and 4.1× (f64)
  without gradients and 1.8–1.9× with, on Kathleen (AVX2) 3.9× and 4.4×, 1.5× and 1.9×,
  against the model's 15–20× and 3.5–4×, whose bounds assume every vector's accumulators
  in registers.
- **The leaf operators** share the harmonics through `nd_fmm_ref::leaf::{p2m, p2l, l2p,
  m2p}_many` (the oracle crate: the single-vector operators are their m = 1 case).
- **M2M, L2L and M2L target by target** apply the table to each vector's column in turn
  (the table from cache); **Batched** gives each pair m columns of the column kernel, so a
  table is read once per block for m times the columns.
- **The device**: the grouped GEMMs (M2M, L2L, dense M2L) and rotation M2L run on views
  with m columns per box (box b's vector v is column b m + v; rows and batches expanded,
  each column meeting its sources in the plan's row order), with no kernel change; P2M,
  P2L, L2P, M2P and P2P take the vectors along the grid's z (one launch per level call as
  before, the cube at z = v the one-vector kernel on vector v); the GEMM decisions are
  keyed by the plan's pairs, so a tuning cache gives FMMs of any m the same choices, and a
  choice's chunk budget is per vector unless the builder names one (m times its bytes, at
  most an eighth of the device's memory: the row reduction walks every entry once per
  chunk, so m times the chunks cost m², measured 10× per vector at m = 32 on Metal; the
  root fix, `Accumulate::Rows` by chunk, is T7's, which then revisits this budget). The
  transfers are device-path §14's with m times the bytes (one upload, one download, one
  sync on one rank).
- **Not changed**: the cost model weighs one vector's work (only the partition depends on
  it; T11 refits the constants), and the strategy rule (`Auto`) is m's.

**Measured** (`fmm-validate/results/phase6-rhs.md`; the N = 10⁶ cube, φ and ∇φ, one rank
per core): the evaluation per vector at m = 8 against m = 1 is 0.61–0.63 at f64 p = 3 and
0.69–0.74 at f64 p = 8 under `Batched` on Kathleen and locust (f32 0.61–0.71 and
0.70–0.76; the M3 Max 0.62 and 0.83, f32 0.64 and 0.83, its batched kernel already near
the core's compute bound), against §4.4's model of 0.3–0.5 and 0.6–0.8; the gain saturates
at m = 4–8.
The model's p = 3 figure assumed P2P's operation count (3.5–4× with ∇φ); the kernel gives
1.7–1.8× in f64, its accumulators bounded by the registers (§4.5 above), and P2P is most
of the evaluation at p = 3. Under `Rotation`, which `Auto` resolves at p = 8 with one rank
per core, M2L shares nothing across the vectors and the evaluation gains only 6–11 %
(6–10 % on one node, 6–11 % at 80 Kathleen ranks). On
the device the first sweep found M2L quadratic in m (the chunking above; 10× per vector at
m = 32 on Metal, 2.0× for M2L on CUDA); with the budget per vector the evaluation per
vector at m = 32 is 0.40–0.65 on CUDA (mostly the host's forward and backward, which
fall per vector) and flat on Metal, dense M2L costing the same per vector at every m on
both. The exchanges keep their
messages and grow in bytes as §4.1 says; the replicated top's all-gather per vector falls
from 4.8 to 3.0 ms (N = 10⁶) and 17.2 to 8.2 ms (10⁷) at 80 Kathleen ranks.

## 5. AVX-512 P2P (C6.7)

### 5.1 Stable Rust has it

Rust 1.89.0 (2025-08-07) stabilised the AVX-512 target features and the
`core::arch::x86_64` AVX-512 intrinsics ([release notes](https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/)).
The toolchain in use is 1.99.0 everywhere: the M3 Max (`rustc --version`), the pins of
`tools/kathleen/env.sh` and `tools/gh200/env.sh`, and CI's floating `stable` (its logs
show 1.99.0). `nd-fmm-simd`'s `toolchain` module already compiles a safe
`#[target_feature(enable = "avx512f")]` function and prints whether `avx512f` is
detected. No nightly toolchain and no `pulp` (fmm-simd/CLAUDE.md: hand-written
`core::arch` only). (pulp 0.22.3 now gates V4 behind its `x86-v4` feature, not `nightly`;
simd-p2p §2.2's wording is out of date but the crate does not use pulp.)

### 5.2 The kernel

simd-p2p §4.7's plan, unchanged: `arch::avx512` with the vector layer of `arch::Simd`
for f32 (16 lanes) and f64 (8 lanes), behind `#[target_feature(enable = "avx512f")]`
only (every operation it needs is AVX-512F: the 512-bit FMA, `_mm512_rsqrt14_ps/_pd`,
compares into `__mmask16`/`__mmask8`, masked moves):

- **inverse square root**: the estimate `rsqrt14` (relative error < 2⁻¹⁴, Intel documents
  a bit-exact reference), one Newton step in f32 (to about 2⁻²⁷ < 4 u_T) and two in f64
  (to about 2⁻⁵⁴), so no f32 detour as on AVX2 (simd-p2p §4.3); the C3S.3 contract (4
  u_T on the domain, 0 after the mask) checked exhaustively in f32 on [1, 4) and on 10⁷
  samples in f64, as for AVX2;
- **the r² = 0 rule** with a native mask: `Mask = __mmask16 / __mmask8`, `eq` a compare
  into a mask, `and_not(m, x)` = `_mm512_maskz_mov(!m, x)`; the `Simd` trait's associated
  `Mask` type allows it without a trait change;
- **tails**: the padded copy of `arch::p2p::p2p_body` as for the other ISAs (the same
  bits whatever the ISA's tail handling), with `MAX_LANES` raised from 8 to 16;
- **blocking**: 32 registers allow K = 2 vectors of targets per source in f32 and f64
  (AVX2 has K = 1 with 16 registers); K chosen by the instruction-count check of
  simd-p2p §7 and timing;
- **the contract**: 8 u_T per φ term and 16 u_T per ∇φ term (simd-p2p §3, requirement 2),
  bit identity across thread counts (§5.5 there);
- `unsafe` only in `arch::avx512` and the dispatch, each block with `// SAFETY:` (simd-p2p
  §5.4; root CLAUDE.md).

### 5.3 The dispatch rule

`Isa::Avx512` (non-breaking: `Isa` is `#[non_exhaustive]`), available iff
`is_x86_feature_detected!("avx512f")`. Recommended rule (decision 4):

- `P2pChoice::Isa(Isa::Avx512)` is always selectable where the CPU has it;
- `Isa::detect()` (and so `P2pChoice::Auto`) **keeps AVX2 until T11** signs it off: the
  default output changes with the ISA (a different rsqrt estimate), so requirement 6
  applies, and Cascade Lake's frequency licences (the 512-bit heavy licence lowers the
  clock of a core while it runs 512-bit FMAs) may make it slower inside a fully loaded
  FMM than in a kernel benchmark. T6 measures it inside the FMM at 40 ranks per node; T11
  makes it `Auto` only where that measurement wins (README risk table).

The baseline T6 must beat (§1.3): the AVX2 kernels on Kathleen at 2.95 / 1.74 Gpairs/s
(f32 φ / φ, ∇φ) and 0.82 / 0.56 (f64) on one core (job 604312, `p2p_kernels --ghz 2.5`;
phase5n-kathleen §10 had 2.84 / 1.61 and 0.79 / 0.55), and P2P's fitted constant inside the FMM at 40 ranks per node, 3.09 ns (f64)
and 1.35 ns (f32) per pair (`nd_fmm_exec::cost`, job 603049). The model (simd-p2p §4.6):
1.60 / 0.70 pairs per cycle (f32 / f64 with ∇φ) against AVX2's 0.80 / 0.29, 2× and 2.4×,
before the licence and before lane waste at FMM leaf sizes (simd-p2p §4.1: 16 f32 lanes
put 24 targets into 32 lanes).

### 5.4 How CI covers it

GitHub's `ubuntu-latest` x86_64 runners are drawn from a mixed pool. Over the last 40
`run-tests-simd` x86_64 jobs (2026-10-11 to 2026-10-15; the job prints `lscpu` and the
`toolchain` test prints `avx512f detected`): AMD EPYC 7763 (Zen 3, no AVX-512) 24 times,
EPYC 9V74 (Zen 4) 6 (AVX-512F exposed in 4), EPYC 9V45 (Zen 5) 3, Intel Xeon Platinum
8370C (Ice Lake) 3 and 8573C (Emerald Rapids) 4: AVX-512F in 14 of 40, 2 of the latest 12.
So the AVX-512 tests run **opportunistically** in CI and must skip cleanly with a message
elsewhere ("not run: no avx512f"), never fail; the timing and the exhaustive f32 check
run by hand on Kathleen (T6); the other-architecture clippy (`--target
x86_64-apple-darwin` on the M3 Max) type-checks the module on every change; and a unit
test of the dispatch with a forced feature set (an `AvailableIsa` built from a given set,
crate-private) checks that `detect` never returns `Avx512` without the feature. The
tests also print which ISAs ran, so a CI log shows whether AVX-512 was covered.

### 5.5 As built in T6

`Isa::Avx512` and `arch::avx512` as §5.2 planned them (`avx512f` only, `vrsqrt14ps`/`pd`
with one Newton step in f32 and two in f64, mask registers, tails on the padded copies with
`MAX_LANES` 16), for every kernel of `nd-fmm-simd`: P2P, P2P of several charge vectors and
the batched M2L's column kernel (bit for bit the scalar layer, so `Dense`, on AVX-512 too).
Available where the CPU has AVX-512F, AVX2 and FMA; `Isa::detect` and `P2pChoice::Auto`
keep AVX2 (§5.3), the rule a function of the feature set with a unit test over every
forced set (§5.4). What the measurements fixed or found
(`fmm-validate/results/phase6-avx512.md`, Kathleen jobs 604543–604551, one node each):

- **Blocking**: K = 2 target vectors per P2P block (K = 1, 2, 4 within 5 %), R = 4 row
  vectors in the column kernel (R = 2 was 14 % slower at f64 p = 8). The several-vector
  groups keep AVX2's.
- **Accuracy** on Cascade Lake: the inverse square root within 1.498 u₃₂ (exhaustive) and
  1.488 u₆₄ (10⁷ samples), P2P terms within 4.18 u (φ) and 9.60 u (∇φ); every
  `tests/mpi_exec.rs` scenario with AVX-512 within C3S.5's tolerance of the reference, at
  1–8 ranks and threads.
- **The kernel**, one core: 1.85–2.03× AVX2 in f32 and 2.98–3.08× in f64 on the FMM-shaped
  cells (the model said 2× and 2.4×; AVX2's f64 kernel reaches only 71–80 % of its model).
  Every core at once: AVX-512 keeps 0.64–0.66 of its one-core rate against AVX2's
  0.73–0.76 (the licence), and still leads by 1.37–1.47× and 2.6×. f32 leaves of fewer
  than 16 points lose lanes (0.55–0.71× AVX2 per pair at 8 targets).
- **Inside the FMM at 40 ranks per node** (the risk of the README's table): the P2P kind
  1.40–1.72× (f32) and 2.03–2.33× (f64) faster, the evaluation 1.49–1.63× at f64 p = 3 on
  the cube, 1.17–1.30× at f32 p = 3, 1.00–1.07× at p = 8, 0.99–1.11× on the Plummer
  sphere; the stages without AVX-512 code unchanged within run-to-run variation (repeated).
  The batched M2L's kernel 1.26–1.42× faster at 40 ranks.
- **For T11**: AVX-512 as `Auto`'s ISA on AVX-512F machines (measured on Cascade Lake
  only), and the cost model's P2P constant refitted (2.78–3.08 → 1.19–1.50 ns per pair in
  f64, 1.09–1.37 → 0.65–0.96 ns in f32, at 40 ranks per node).

## 6. The device items (C6.1)

The device's dense M2L, M2M and L2L run per chunk as gather, grouped GEMM and reduction
(`nd_fmm_kernels::translate::level_call`; device-path §6.4). Where the time goes, measured
in Phase 4 and 4S:

| item | measured | source |
| --- | --- | --- |
| gather, GEMM, reduction in the M2L stage (Metal f32, cube) | p = 3: 18 / 43 / 29 %; p = 8: 13 / 73 / 12 %; p = 12: 8 / 78 / 12 % | phase4-m3max.md (M2L stage split) |
| the same, largest level call (CUDA f32 p = 8, N = 10⁶) | gather 1.2 ms, GEMM 5.8 ms, reduction 1.1 ms | device-path §18.2 |
| GPU time per evaluation (CUDA f32 p = 8, N = 10⁶, nsys) | GEMM 55 %, P2P 16 %, gather 11 %, reduction 11 %, L2P 3 % | device-path §18.2 |
| hand-written GEMM alone, f64 (H100) | 13–24 % of the f64 peak at p = 3–12 (4.3–8.0 TFLOP/s), limited by loads and issue (no shared-memory staging) | device-path §18.2 |
| a shared-memory-tiled GEMM (the spike) | 24.1 / 22.8 / 29.2 % at p = 8 / 12 / 16 (f64, GH200) | spikes/cubecl-gemm/SPIKE_REPORT.md |
| M2L efficiency against N (CUDA f64 p = 18) | 23.1 % at N = 10⁶ → 16.0 % at 10⁷ (2,503 ms), the per-chunk row walk of `Accumulate::Rows` | phase4s-gh200.md §4; device-path §18.2 |
| L2P at high p (CUDA f64) | 4.8 ms at p = 12 (0.3 % of peak), 12.5 of 244 ms at p = 18 (N = 10⁶), 129 ms at N = 10⁷ | phase4s-gh200.md (per-kind table) |

The items, each adopted only by measurement (T7 brief), with the gain *modelled* from those
shares:

| item | in T7? | expected gain (*model*) |
| --- | --- | --- |
| **fused gather** (the GEMM loads its columns through the batch's source indices; the same accumulation order, bit for bit) | yes | at most the gather's share: 8–18 % of the M2L stage on Metal, 11 % of the GPU time of a CUDA evaluation at f32 p = 8 (the GPU busy 12.4 of 21.4 ms, so ≤ 6 % of the wall time there); more at low p, where the gather is a larger share |
| **`Accumulate::Rows` by chunk** (walk only the chunk's sub-range of each row, known at schedule time; bit for bit) | yes | the N = 10⁷ M2L back to N = 10⁶'s efficiency: 2,503 → about 1,740 ms at f64 p = 18 (−30 % of the M2L, about −25 % of the evaluation at that point); small at N = 10⁶ |
| **shared-memory-tiled GEMM** | yes, as a candidate | 0–20 % of the GEMM at p ≥ 8 (the spike's 22.8–29.2 % against the hand-written 20–24 %); the order must keep one accumulator per output with k ascending (bit for bit) or state its bound |
| **L2P at high p** | only if simple | ≤ 5 % of an evaluation (12.5 of 244 ms at f64 p = 18) |

The host batched M2L (§2) does not change the device: its order-preserving β = 1 form was
rejected for the device (device-path §6.4, "a gathered copy of L"), and the device keeps
its accumulate-from-zero GEMM with the row-order reduction (device and host agree within
Phase 4's bounds, not bit for bit, as today).

## 7. Dipoles (C6.4)

### 7.1 The source-type interface

The source type is fixed at build, like the gradients, because it sets the source chunk's
width (`source_point_size`) and with it the exchanges:

```rust
pub enum SourceType { Charges, Dipoles, ChargesAndDipoles }   // #[non_exhaustive]
impl<T> FmmBuilder<T> { pub fn sources(self, kind: SourceType) -> Self; }  // default Charges
/// One evaluation's source strengths, the caller's point order.
pub struct Sources<'a, T> { pub charges: Option<&'a [T]>, pub dipoles: Option<&'a [[T; 3]]> }
impl<T> Fmm<T> {
    pub fn evaluate_sources(&mut self, sources: &Sources<'_, T>) -> Result<Output<T>, FmmError>;
    pub fn evaluate_sources_into(&mut self, sources: &Sources<'_, T>, output: &mut Output<T>)
        -> Result<(), FmmError>;
}
```

`evaluate(charges)` stays the `Charges` path, bits unchanged; a `Sources` that does not
match the built type is refused (`FmmError::SourceType`, agreed with the length check).
The forward carries 1, 3 or 4 values per point (`Redistribution::forward` takes the
width), and `source_point_size` is 4, 6 or 7. Several right-hand sides (§4) and dipoles
compose (m sets of strengths), but T8 builds one set.

### 7.2 The operators that change

Only the source side. With φ(x) = d · ∇_y (1/|x − y|) = d · (x − y)/|x − y|³ (sign and
1/(4π) as §3.1 fixes for charges; the convention is T8's proposal):

| operator | charges (today) | dipoles |
| --- | --- | --- |
| P2M | Σ q R̄(u) | Σ d̂ · ∇_u R̄(u): `harmonics::regular_grad` exists (CONVENTIONS §3.4's ladder) |
| P2L | Σ q Ī(u) | Σ d̂ · ∇_u Ī(u): `harmonics::irregular_grad` exists |
| P2P | q / r, ∇ | d̂ · (x − y)/r³ and its gradient d̂/r³ − 3 (d̂ · (x − y))(x − y)/r⁵: new SIMD kernels |
| direct sum | `nd_fmm_ref::p2p::direct_sum` | the dipole sum, f64, in `nd-fmm-ref` |
| M2M, M2L, L2L, L2P, M2P, lists, exchanges of expansions | unchanged | unchanged |

d̂ is the moment in the source leaf's scaled units (d / r_s: a dipole carries one power of
length, so d · ∇_y = d̂ · ∇_u); the scale between two leaves in P2P is a power of two,
exact. The rsqrt and Newton steps of simd-p2p §4.3 serve r⁻³ and r⁻⁵ (ρ³, ρ⁵), so the
dipole P2P kernel shares the charge kernel's estimate and contract structure; its
contract (u_T per term) is T8's to derive and state.

### 7.3 CONVENTIONS

CONVENTIONS.md does not mention dipoles. T8 proposes in its PR (never in code; root
CLAUDE.md): a subsection of §3.7 "Dipole sources" (the potential's sign, the expansion
coefficients as the directional derivative of the charge coefficients with respect to the
source position, in §3.6's real storage), and in §3.13 "Source chunks" the layouts with
d̂ (3n values after the coordinates, or after the charges for `ChargesAndDipoles`), with
`CONVENTION_VERSION` following the file. Code that depends on it waits for that
sign-off (T8 brief).

### 7.4 The device

Recommended: the dipole P2M, P2L and P2P on the **host fallback** first (`host_fallback`:
every kind can run on the host with explicit transfers, bit for bit the host path;
requirement 8 of Phase 4), and device kernels only if T8 measures the fallback above 20 %
of a device evaluation at the workloads. The leaf kernels of `nd_fmm_kernels::leaf`
evaluate the harmonics per point; a dipole form needs the gradient ladder on the device
(new code); the far field on the device is unchanged.

## 8. FMM3D (C6.8) and plane waves (C6.5)

### 8.1 The FMM3D comparison (T9)

**Method.** Both codes on the same points and charges, at matched accuracy, against the
same reference, on one node, precomputation reported apart:

- *problems*: the C3.2 cube and the C3.3 Plummer sphere at N = 10⁵, 10⁶, 10⁷ (Phase 5
  T10's `Workload`, every point generated by index from the seed), written once to a
  binary file by an `nd-fmm-bench --write-problem` option (the one code change of T9);
  sources = targets, as our `Fmm` evaluates at its points;
- *reference*: our f64 direct sum (`nd_fmm_ref::p2p::direct_sum`) at 1,000 sampled targets
  for 8 charge vectors, the errors of laplace-fmm-plan §8.2 (relative L2 of φ and ∇φ);
  FMM3D's Laplace kernel is 1/r without our 1/(4π) (CONVENTIONS §3.1), so its output is
  divided by 4π before the comparison (T9 checks the normalisation on a two-point
  problem first);
- *matched accuracy*: for each of our p ∈ {3, 6, 8, 12} at f64 (errors 2.6e-3 … at p = 3,
  1.8e-5 at p = 8 on the cube, laplace-fmm-plan §4), FMM3D's `eps` whose measured error
  is closest; both errors reported;
- *configurations*: one thread, and all cores (ours as Phase 5N's rule and §9's best
  host configuration after T2, FMM3D's OpenMP threads, `OMP_NUM_THREADS` stated);
  FMM3D's own setup (tree, lists) and our table build and plan reported apart from the
  evaluation; FMM3D with its defaults otherwise;
- *several vectors*: FMM3D evaluates nd densities in one call (its vectorised
  interfaces); the 8-vector timing is reported beside ours (8 evaluations, and after T5
  `evaluate_many`).

**Build route (decision 5).** `tools/fmm3d/build.sh` fetches FMM3D at a pinned release
tag (the latest at T9, recorded) and builds its Fortran library with OpenMP outside the
repository: the M3 Max with Homebrew's `gfortran` (installed: `/opt/homebrew/bin/gfortran`,
GCC 16) and FMM3D's macOS `make.inc`; Kathleen with the `compilers/gcc/12.3.0` module's
gfortran, under `$FMM_KATHLEEN_ROOT/fmm3d`, built in a job; locust with the system or a
spack gfortran under `/data/ucahtbe/fmm3d`. The driver is a small Fortran program
(FMM3D's own language, no binding layer) that reads the problem file, calls the
sources-only interface with charges and gradients (`lfmm3d_s_c_g` and its vectorised
form, names checked against the pinned release's documentation), times with
`omp_get_wtime` and writes φ and ∇φ for the error check. Nothing is linked into a crate.
**Recommended gate machine: Kathleen** (x86, where FMM3D's compiled kernels are mature and
our AVX2/AVX-512 P2P is measured), with the M3 Max as the second machine.

### 8.2 Plane-wave M2L (T10): the question and the stop rule

*The question.* At which p, precision and machine would plane-wave M2L (laplace-fmm-plan
§3.3: multipole → exponentials in six directions, diagonal translation with
merge-and-shift, exponentials → local) beat T2's batched dense M2L (and Rotation) on the
uniform cube at N = 10⁶, counting the conversions? laplace-fmm-plan
§3.3 and §4 put it "within about 25 % of point-and-shoot" at "about 2.5× the memory", and
"at most tens of percent on CPUs in careful comparisons"; the batched M2L (§2) raises the bar
further (C6.2, compression, was dropped at sign-off).

*The model first.* Flops and memory traffic per box: 6 × 2 conversions per box (each
O(p³) after rotation to the direction's axis), per V pair one diagonal product of s(p)
complex exponentials (s(p) from the quadrature at the target accuracy; Cheng, Greengard
and Rokhlin 1999 give the node counts; verify before citing), reduced by merge-and-shift
to a few translations per box and direction; at the measured rates of T2 (and Rotation) on each
machine.

*The stop rule (proposed margin).* Build the spike only where the model shows the plane
waves at least **1.25×** faster than the faster of Batched and Rotation on the M2L
stage at some p ≤ 18 and some machine of the workloads; otherwise write the spike report
with the model and recommend dropping C6.5 (decision 6). If built, it must measure that
margin on one level against T2 and Rotation, else C6.5 is dropped.

## 8a. M2P and P2L only where they pay (C6.9)

### 8a.1 The pairs in question

For a leaf t, W(t) holds the children of t's refined same-level neighbours that do not
touch t (leaves or interior boxes), and X(b) of a box b the leaves at the level of b's
parent that do not touch b; s ∈ W(t) iff t ∈ X(s) (`nd-fmm-plan`'s `interaction_manager`
module docs). So between a leaf t and a finer **leaf** s ∈ W(t) the FMM makes two
expansions: M2P of s's multipole at t's n_t target points (cost α_M2P n_t in the cost
model's units, `nd_fmm_exec::cost`) and P2L of t's n_t sources into s's local (α_P2L n_t).
Evaluated directly, both are P2P: s's n_s sources at t's targets and t's sources at s's
targets, α_P2P n_t n_s each, and exact. So, per pair of leaves (t coarse, s fine):

- **W**: P2P instead of M2P iff n_s < N_W = α_M2P / α_P2P;
- **X**: P2P instead of P2L iff n_s < N_X = α_P2L / α_P2P (s is the X pair's target box).

When s is interior (W can name one), its points lie in leaves the source exchange does
not deliver; such pairs keep M2P and P2L (their source box holds more points anyway).

### 8a.2 The crossover, measured

The cost model's constants fitted to the kind times of §1.2's runs (`scaling --kinds`,
Σ time over Σ count per kind, `PathTimes::fit`), the Plummer sphere at N = 10⁶ under
Dense, φ and ∇φ; one rank unless named:

| machine | f64 p = 3 | p = 6 | p = 8 | p = 12 | f32 p = 3 | f32 p = 8 |
| --- | --- | --- | --- | --- | --- | --- |
| N_W = α_M2P / α_P2P: M3 Max | 98 | 257 | 406 | 810 | 177 | 745 |
| locust | 46 | 116 | 179 | 352 | 129 | 498 |
| Kathleen (p = 12 and f32 at 40 × 1) | 107 | 262 | 406 | 805 | 231 | 963 |
| N_X = α_P2L / α_P2P: M3 Max | 27 | 87 | 151 | 302 | 48 | 268 |
| locust | 15 | 43 | 70 | 151 | 38 | 194 |
| Kathleen (p = 12 and f32 at 40 × 1) | 33 | 102 | 186 | 419 | 77 | 460 |

A leaf holds at most 64 points below the deepest level (`FmmBuilder::max_points_per_leaf`,
`DEFAULT_MAX_POINTS_PER_LEAF`), so at p ≥ 6 every W pair with
a leaf source is cheaper as P2P on every machine (N_W ≥ 116), and at p ≥ 8 every X pair
with a leaf target (N_X ≥ 70); at p = 3 only the smaller leaves convert (N_X = 15–33,
N_W = 46–107). The constants are the same at one rank and at the node's ranks within a
few per cent (Kathleen p = 8: 406 and 421).

**How much is at stake** (one rank on the M3 Max, M2P + P2L of an evaluation; §1.2 has the
others): the cube at 2 × 10⁶ (Phase 5S T9's case) 71 % at f64 p = 8 and 73 % at p = 3; the
Gaussian clusters at 10⁶ 35 % and 50 %; the Plummer sphere at 10⁶ 35 % and 47 %; the
uniform cube at 10⁶ and 4 × 10⁶ none (no W or X pairs). On the cube at 2 × 10⁶ the W list
has 1.02 × 10⁶ pairs (and as many X pairs) with 5.8 × 10⁷ target points under them
(`work: M2P count`).

*Model* of the gain, if every W box is a leaf at the 64-point maximum (a converted pair
then costs 64/N_W of its M2P and 64/N_X of its P2L; fewer points cost less, interior W
boxes keep M2P): the Plummer sphere at f64 p = 8 1.36× faster (M3 Max one rank: M2P 25 %,
P2L 9 % of the evaluation) and about 1.6× at 40 Kathleen ranks per node; the cube at
2 × 10⁶ 2.2×; at p = 3 1.15× (only W converts). T12 measures the share of W boxes that are
leaves and their point counts.

### 8a.3 The rule and where it lives

The plan makes the lists; the operator knows the costs. Recommended: the plan's per-key
rule gains an optional, kernel-agnostic **near rule** that the operator supplies, a
predicate over two leaves and their point counts (source count of the fine leaf for W,
target count for X), deciding for each direction separately whether the W entry s of leaf t
becomes a near entry of t (t's near row gains s; s's M2P at t is dropped) and whether the
X entry t of s becomes a near entry of s (s's near row gains t; t's P2L into s is dropped). Everything
downstream follows without new code: the near rows are already ordered by (level, key)
(redesign §7.5), the leaf numbering then names s as a ghost leaf where it is one (as U
does), so the source exchange delivers its points, and P2P runs on host and device as for
any near pair. The counts are what Phase 5S T7's cut already gathers at build
(`cost::leaf_point_counts`, every local leaf's source and target counts and the ghost
leaves' source counts), identical on every rank, so every rank makes the same lists
(requirement 3). `nd_fmm_exec` passes the rule built from its `CostModel` (the same
constants as the cut, scaled by p), opt-in by a builder setting (`FmmBuilder::near_rule`
or similar, off by default until T11), and the cut's counts move the converted pairs from
M2P/P2L to P2P.

This is an **`nd-fmm-plan` interface change** (`Plan::new` takes the rule and the counts;
`from_key_types` likewise), so it is a sign-off question (README, "Every operator behind
the `FmmOperator` interface"; §10 question 7).

### 8a.4 Accuracy and order

Every converted pair replaces a truncated expansion by the exact sum, so the error
against the direct sum can only fall (requirement 1's direct-sum bounds hold); the
output is *not* within 1e-12 of today's host path (the truncation error of those pairs
leaves it), so, as for a compressed operator, C6.9 states its own criterion: the
direct-sum errors at most the existing path's on the C3.2 cube and the C3.3 Plummer sphere,
and the cube at 2 × 10⁶ on 80 ranks. Order: a target's output receives its near pairs in
the near row's (level, key) order, then M2P, then L2P, as today; the multi-rank and
thread-count guarantees are unchanged because the rows are.

### 8a.5 Which task builds it

A new task, **T12 "M2P and P2L only where they pay"**, after T1 (needs nothing from
T2–T5): `nd-fmm-plan` (the near rule in `key_lists`, `Plan::new`), `nd-fmm-exec` (the rule
from the cost model, the builder setting, the counts), the `scaling` harness (the counts
by kind, unchanged in form). It shares `fmm-exec/src/fmm.rs` with T5 and T8: merge one,
then rebase the next (README "Tasks"). Its brief is written after sign-off; its
criterion is the README's (C6.9).

## 9. Order and gain

The components by measured share and modelled gain (§1, §2.10, §3.3, §4.4, §5.3, §6,
§8a.2); "×" is the evaluation's speed-up against today's best strategy, *model* unless
marked:

| task | component | where it pays | gain (*model*) | crates |
| --- | --- | --- | --- | --- |
| T2 | C6.6 batched host M2L | every tree at p ≥ 6, most at many ranks per node | f64 p = 8 at the node's ranks 1.13× (locust), 1.20–1.30× (M3 Max), 1.87–2.07× (Kathleen, with the AVX2 kernel); f32 p = 8 1.5–3.0×; one rank 1.3–2.3×; p = 3 1.07–1.21× | nd-fmm-exec, nd-fmm-simd |
| T12 (new) | C6.9 M2P and P2L only where they pay | adaptive trees (Plummer, clusters, graded cubes), every p | Plummer f64 p = 8 1.36× (one rank) to about 1.6× (40 per node); the cube at 2 × 10⁶ 2.2×; uniform trees none | nd-fmm-plan, nd-fmm-exec |
| T6 | C6.7 AVX-512 P2P | Kathleen, low p (P2P 63–66 % of the cube at f64 p = 3) | P2P up to 2× (f32) and 2.4× (f64) per core before the licence (simd-p2p §4.6): the cube at p = 3 up to 1.3–1.5×, p = 8 ≤ 1.1× (more after T2 shrinks M2L) | nd-fmm-simd |
| T5 | C6.3 several right-hand sides | m vectors on the same points | per vector 0.3–0.5 of one evaluation at p = 3, 0.6–0.8 at p = 8 (m = 8) | nd-fmm-exec, nd-fmm-kernels, nd-fmm-simd |
| ~~T3, T4~~ | C6.2 compressed M2L, **dropped at sign-off** | p ≥ 12, memory | flops 0.53× (p = 12), 0.31× (p = 18) of dense; tables 3.2× smaller at p = 18; none at p ≤ 8 | – |
| T7 | C6.1 device work | CUDA and Metal | fused gather ≤ 6 % of a CUDA evaluation (f32 p = 8); `Accumulate::Rows` by chunk −30 % of the M2L at N = 10⁷, f64 p = 18 | nd-fmm-kernels, nd-fmm-exec |
| T8 | C6.4 dipoles | a feature | – | nd-fmm-ref, nd-fmm-math, nd-fmm-exec, nd-fmm-simd |
| T9 | C6.8 FMM3D | a comparison | – | tools/, nd-fmm-bench |
| T10 | C6.5 plane waves | model first | must show 1.25× over Batched and Rotation | spikes/ |

**Order** (signed off): T2 first (the largest gain on uniform trees at p ≥ 6, and T5, T9
and T10 build on it); then **T12** beside it (the largest on adaptive trees, in other
crates: `nd-fmm-plan`, and `fmm.rs`/`cost.rs` of `nd-fmm-exec`); T6 after T2 (both change
`nd-fmm-simd`'s `isa.rs` and `arch/mod.rs`; T6 adds the AVX-512 body of T2's kernel with its
P2P); T7 and T8 beside them (disjoint crates, as the README has it); T5 after T2;
T9 after T2; T10 its model only, unless the model shows the margin; T11 last.

**Brief changes** (signed off with decision 1, and made in this PR after the sign-off,
beside the factual corrections of Appendix D):

- **README**: C6.9's task is T12 (Components table, Tasks table, exit checklist); "Scope"
  adds `nd-fmm-simd` (the M2L kernel) to T2 and `nd-fmm-plan` (the near rule) to T12;
  "Design decisions", BLAS inside rayon: no host BLAS in Phase 6 (decision 2).
- **T2**: the strategy `M2lStrategy::Batched`, the order-preserving kernel in
  `nd-fmm-simd` (§2.11) instead of "BLAS through rlst", M2M and L2L unchanged (§2.7),
  Classes unchanged (§2.8); its tests bit for bit `Dense`; the FMA mode if signed off
  (question 2c); the results file `fmm-validate/results/phase6-m2l.md`.
- **T3, T4**: dropped with C6.2 (decision 3); the briefs stay as the record, marked so.
- **T5**: the interface of §4.2 (`FmmBuilder::vectors`, `evaluate_many`), point-major
  charges, the CONVENTIONS §3.13 addition proposed in its PR.
- **T6**: `Auto` keeps AVX2 until T11; the AVX-512 body of T2's M2L kernel joins it.
- **T9**: the Fortran driver, Kathleen as the gate machine.
- **T10**: the 1.25× margin over Batched and Rotation; it needs T2 only.
- **T11**: no compressed strategy; C6.9's default and the FMA mode's default among the
  defaults it revisits.
- **T12** (new brief, written after sign-off): §8a.

## 10. Questions for sign-off

1. **Decision 1, this document**, with the changes it proposes to the requirements and
   scope: (a) C6.9 states its own accuracy criterion (exact pairs, errors against the
   direct sum at most today's; not within 1e-12 of today's output), like a compressed
   operator; (b) a new task T12 for C6.9 and its `nd-fmm-plan` interface change (question
   7); (c) `nd-fmm-simd` in T2's scope; (d) the brief changes of §9. *Recommended: yes.*
2. **Decision 2, BLAS on the host.**
   a. *Recommended: no BLAS; the hand-written order-preserving kernel* (§2.3, §2.6), in
      `nd-fmm-simd` with run-time ISA dispatch (§2.11), bit for bit today's `Dense`.
      BLAS measured 1.07–3.8× faster than it at the node's ranks, but its bits change with
      the column count (so with the rank count), the CPU kernel and the library, its
      many-thread behaviour depends on the build (locust), and no provider is linked today.
   b. *M2M and L2L batched: no* (≤ 1.15 % of an evaluation, §2.7); Classes unchanged (§2.8).
   c. *The FMA mode*: offer it (opt-in, a second mode of `Batched`, measured 1.1–1.4× on
      the kernel, within 1.6e-15 (f64) of Dense, deterministic and rank-count invariant)?
      Recommended: yes, opt-in in T2, its bound tested; T11 decides a default.
   d. Naming: `M2lStrategy::Batched` (dense tables, per-level products on the host;
      `Dense` on the device). *Recommended.*
3. **Decision 3, compressed M2L.**
   a. *Per-offset truncated SVD, a relative singular-value cut ε per offset, default
      ε(p) ≈ 0.1 × the FMM's error at p, fitted by T3; budget: errors against the direct
      sum ≤ 1.1× dense's, against dense ≤ 0.1× dense's own error; opt-in until T11*
      (§3.2). Shared bases rejected (no compression, §3.1).
   b. The SVD: *a hand-written one-sided Jacobi in `nd-fmm-tables` (recommended:
      deterministic, no native library)*, or rlst's LAPACK with a linked provider (a new
      dependency: needs your approval).
   c. Whether it may become a default: only above the p where T3 measures it faster than
      Batched and Rotation (the *model* says p ≥ 12), decided in T11.
4. **Decision 4, AVX-512.** *Hand-written `core::arch` in `arch::avx512`, `avx512f` only,
   stable Rust (1.89+; 1.99.0 in use); `Isa::Avx512` selectable, `Auto` keeps AVX2 until
   T6's in-FMM measurement at 40 ranks per node and T11's sign-off; CI covers it
   opportunistically (AVX-512F on 14 of the last 40 `ubuntu-latest` jobs), skipping
   cleanly elsewhere, plus the other-architecture clippy and a dispatch unit test* (§5).
5. **Decision 5, FMM3D.** *A pinned release built with gfortran and OpenMP outside the
   repository (`tools/fmm3d/build.sh`), a Fortran driver calling its sources-only
   charge-and-gradient interface, Kathleen as the gate machine and the M3 Max as the
   second* (§8.1).
6. **Decision 6, plane waves.** *T10 models first; the spike only if the model shows at
   least 1.25× over the faster of Batched and Compressed on some configuration; otherwise
   C6.5 is dropped* (§8.2).
7. **C6.9's rule and its plan change.** *An optional near rule in `nd-fmm-plan`'s list
   construction, supplied by the operator from the cost model and the leaf point counts
   the cut already gathers: a W pair with a leaf source below N_W and an X pair with a
   leaf target below N_X become near pairs; opt-in until T11* (§8a). Approve the
   `Plan::new` interface change?
8. **Several right-hand sides.** *`FmmBuilder::vectors(m)` and `Fmm::evaluate_many`
   (§4.2); the plan unchanged; point-major charges; each vector bit for bit its
   single-vector evaluation; the CONVENTIONS §3.13 addition proposed in T5's PR.*
9. **Dipoles.** *`SourceType` fixed at build, `Fmm::evaluate_sources`, the device kinds on
   the host fallback first; CONVENTIONS text proposed in T8's PR before code* (§7).
10. **The order of the tasks** (§9): T2, then T12 beside it, T6 after T2, T7 and T8 beside;
    T3, T5, T9 after T2; T10 model only; T11.

### Outcome (signed off by the user on 2026-10-15)

- **Question 1 (decision 1)**: approved: the document, C6.9's own accuracy criterion, the
  new task T12, `nd-fmm-simd` in T2's scope and the brief changes of §9 (made in this PR).
  The recommendations of questions 4–6 and 8–10 are part of the approved document.
- **Question 2 (decision 2)**: approved as recommended, including **c, the opt-in FMA mode
  in T2**: no host BLAS; the order-preserving kernel in `nd-fmm-simd`, `M2lStrategy::Batched`,
  bit for bit `Dense`; M2M, L2L and Classes unchanged.
- **Question 3 (decision 3)**: **SVD-compressed M2L dropped for now** ("barely worth it"):
  C6.2 is not built in Phase 6, T3 and T4 are dropped, and T10 competes with T2 and
  Rotation only. §3 stays as the measurement and design for a later phase.
- **Question 7**: approved: the near rule in `nd-fmm-plan`'s list construction (the
  `Plan::new` interface change) for C6.9, built in T12.

## Appendix A. Runs and commands

All runs on 2026-10-15 from main at 8457395 (Phase 5S T9 merged), release builds,
rustc 1.99.0, every BLAS thread variable at 1. The drivers are throwaway copies under the
git-ignored `bench-results/t1/` and were not committed (T1 brief).

**The FMM baseline (§1).** A copy of `tools/scaling/run.sh` with one more sweep, `t1`:
for the cube and the Plummer sphere at N = 10⁶, f64 p = 3, 6, 8, 12 and f32 p = 3, 8,
each of Dense, Classes and Rotation, at one rank and at the machine's ranks per node, one
`scaling` launch with `--kinds` (`KindTiming::Synchronous`), overlap off, three timed
evaluations after a warm-up, the node budget of the machine class (56 MB M3 Max, 96 MB
otherwise; it does not matter for a named strategy), for example

```sh
mpirun --report-bindings --map-by slot:PE=1 --bind-to core -n 72 \
  target/release/examples/scaling --threads 1 --evals 3 --budget 96 --dist cube \
  --n 1000000 --precision f64 --p 8 --strategy dense --kinds --overlap off
```

| machine | ranks × threads | where, when | load |
| --- | --- | --- | --- |
| M3 Max | 1 × 1, 12 × 1 | unsandboxed, `run.sh`'s quiet gate (≤ 200 % CPU of other processes), 18:24–19:11 | other processes at 42–198 % CPU when launched (median 152 %: Spotlight indexing), recorded per launch |
| locust | 1 × 1, 72 × 1 (`--bind-to core`) | 18:27–19:10; the GEMM benchmark 19:10–19:33 | idle: no other user's process above 5 % CPU in any load record, the GPU at 0 % |
| Kathleen | 1 × 1 (job 604311, node-c11a-025); 40 × 1 and 2 × 20 (`--map-by ppr:<k>:socket:PE=<t> --bind-to core`; job 604312, node-c11a-026) | `singlenode`, exclusive, `--exclude=node-c11a-011` | exclusive |

**The host GEMM benchmark (§2.6, Appendix B).** `t1-gemm`, a standalone crate without
dependencies that links the system BLAS by a build script (Accelerate on the M3 Max,
spack's OpenBLAS 0.3.33 `threads=none` on locust, the `openblas/0.3.28` module and
`intel-oneapi-mkl/2023.2.0` with `MKL_THREADING_LAYER=SEQUENTIAL` on Kathleen), one
process pinned to core 0 (`taskset` on Linux), and for the concurrent rows one process
per core, each pinned and with its own tables. The parts: `gemm` (one table times m
columns), `bits` (BLAS with all columns against the same product in chunks of 1, 3, 7,
64 and 333 columns, and a hash compared across thread settings and BLAS builds), `level`
(a uniform level 4, 4,096 boxes and the 316 offsets, the V list of a uniform tree in
Morton order). Kathleen jobs 604313 (OpenBLAS, default and native builds, node-c11a-026),
604314 (the native build at 40 processes) and 604316 (MKL, `MKL_THREADING_LAYER=SEQUENTIAL`;
the module needs Intel's Open MPI, so ours was unloaded for it); the M3 Max 19:11–19:25,
right after the sweep (load average 3.5 at its start; no other process above 20 % CPU).

**Counts (§8a).** The M3 Max, one rank, 19:29–19:39: `scaling --kinds --evals 1` on the
cube at 2 × 10⁶ and 4 × 10⁶ points and the Gaussian clusters at 10⁶, f64 p = 8 Rotation
and p = 3 Dense (the work counts per kind and the kind times).

**The table SVDs (§3.1).** A throwaway program over `nd_fmm_tables::M2lTables::<f64>::build`
(p = 3, 6, 8, 12, 18; 17 s at p = 18) wrote the tables, numpy computed every table's singular
values and the shared bases (`svd` of the 316 tables side by side and stacked).

**Kathleen jobs and node-hours** (`singlenode`, each one node, exclusive): 604311 (1 h
24 min), 604312 (20 min), 604313 (38 min; its MKL step failed on a module name),
604314 (6 min), 604315 (1 min; the MKL module conflicted with our Open MPI), 604316
(12 min): 2.68 node-hours.

**The 160-rank stage shares (§1.2)** are from Phase 5S T9's raw output, job 604159 (4
nodes, 40 × 1, `t9-strong` at N = 10⁸ and `t9-weak` at 2.5 × 10⁴ points per rank, the
blocking path): each stage's slowest rank over the evaluation's wall time.

## Appendix B. The host GEMM benchmark

**The variants** (`t1-gemm`, Appendix A): *target by target*, a copy of
`MatrixSet::apply` called per pair in row order (today's level call); *direct, hand MR×NB*,
§2.2's kernel reading the columns in place (NB source columns packed k-major, an MR × NB
block of the locals in registers across k, no FMA: `acc = acc + b · a`); *gathered C, hand*,
the same kernel on copies of C source and target columns; *… fma*, `acc = a.mul_add(b,
acc)`; *gathered 256, blas*, `dgemm_`/`sgemm_` with β = 1 on gathered copies of 256
columns. Random tables (timing does not depend on the values), the multipoles random, the
locals zero; "bits" counts the values that differ from target by target.

**One level, one process, the rows §2.6 leaves out** (ns per pair; uniform level 4):

| machine | precision, p | target by target | no FMA (8×4) | FMA (8×4) | BLAS |
| --- | --- | ---: | ---: | ---: | ---: |
| M3 Max | f64 3 / 6 | 58.5 / 429 | 24.5 / 197 | 19.5 / 154 | 20.5 / 81 |
| M3 Max | f32 3 / 6 | 51.4 / 251 | 15.2 / 116 | 11.9 / 98 | 13.9 / 40 |
| locust | f64 3 / 6 | 61.6 / 460 | 32.8 / 263 | 27.5 / 204 | 30.0 / 177 |
| locust | f32 3 / 6 | 45.8 / 295 | 19.9 / 155 | 16.2 / 114 | 16.8 / 93 |
| Kathleen, default build | f64 3 / 6 | 77.3 / 1,065 | 65.4 / 539 | – | 24.2 / 114 |
| Kathleen, native build | f64 3 / 6 | 49.5 / 877 | 36.3 / 274 | 30.3 / 204 | 24.8 / 114 |
| Kathleen, default build | f32 3 / 6 | 44.0 / 522 | 33.7 / 259 | – | 16.4 / 64 |
| Kathleen, native build | f32 3 / 6 | 54.5 / 467 | 22.1 / 164 | 20.7 / 147 | 16.9 / 65 |

**One table times m columns** (GFLOP/s at m = 256; f64 p = 8 / 12 / 18):

| machine | matvec (`apply` per column) | no FMA, best block | FMA, best block | BLAS |
| --- | --- | --- | --- | --- |
| M3 Max | 14.7 / 11.7 / 12.1 | 26 / 27 / 27 | 35 / 37 / 37 | 188 / 259 / 356 |
| locust | 12.9 / 11.4 / 10.4 | 20.5 / 21.0 / 14.2 | 26.8 / 28.3 / 23.5 | 38.7 / 39.1 / 39.6 |
| Kathleen, default build | 7.4 / 7.5 / 6.3 | 10.4 / 10.8 / 6.5 | – | 84.8 / 87.7 / 94.7 |
| Kathleen, native build | 11.1 / 11.6 / 8.8 | 22.1 / 22.7 / 11.9 | 30.0 / 28.6 / 15.0 | 84.6 / 87.8 / 93.8 |

**BLAS bits.** The product of one table with 4,096 columns at once against the same
product in chunks of 1, 3, 7, 64 and 333 columns (values differing out of 4,096 n):

| BLAS | f64 p = 3 | p = 8 | p = 12 |
| --- | --- | --- | --- |
| Accelerate (M3 Max) | 1: 42,805; 3: 45,398; 7: 46,052; 64, 333: 0 | 1: 254,310; 3, 7: 68; 64, 333: 0 | 1: 533,223; 3, 7: 136; 64, 333: 0 |
| OpenBLAS 0.3.33 (locust) | 1: 46,256; 3, 7: 12; 64, 333: 0 | 1: 285,649; 3, 7: 70; 64, 333: 0 | 1: 621,328; 3, 7: 152; 64, 333: 0 |
| OpenBLAS 0.3.28, SkylakeX (Kathleen) | 0 in every chunking | 1–64: 2,854–2,877; 333: 168 | 1–7: 3,118–3,157; 64: 207; 333: 92 |
| OpenBLAS 0.3.28, `OPENBLAS_CORETYPE=Haswell` | 1: 38,851; 3: 12,988; 7: 5,613; 333: 116 | 1: 258,267 … 333: 759 | 1: 587,176 … 333: 1,717 |

With 4 BLAS threads the hashes of the whole product are unchanged on the M3 Max
(Accelerate, `VECLIB_MAXIMUM_THREADS=4`); the OpenBLAS builds on locust and Kathleen are
single-threaded, so the setting is a no-op there. Across builds the same product's hash
differs between Accelerate and OpenBLAS at every p, between OpenBLAS's SkylakeX and Haswell
kernels at p = 8 and 12 (equal at p = 3), and agrees between locust's `neoversev2` and
Kathleen's Haswell kernels at p = 3 and 8 only. MKL 2023.2.0 (job 604316, `MKL_THREADING_LAYER=SEQUENTIAL`):
77–98 GFLOP/s at f64 p = 8–18 and m = 256–1,024 (as OpenBLAS), and the same dependence on
the chunking (f64 p = 8: chunks of 1, 3 and 7 columns change 259,599, 71 and 71 values; 64
and 333 none), also under its reproducibility modes (`MKL_CBWR=COMPATIBLE`: chunks of 1–7
change 111,429–260,341 values at p = 8; `MKL_CBWR=AVX2` 37,526–259,896): CNR makes a call
reproducible, not a column independent of the call's shape. The three settings give three
different hashes of the same product (the default's equals OpenBLAS's Haswell kernels' at
p = 8 and 12).

## Appendix C. Every cell

**C.1 The stages and kinds of every cell** (as §1.2; N = 10⁶):

| machine | ranks × threads | problem | best (ms) | Dense / best | downward | leaves | M2L | P2P | M2P + P2L |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| machine | ranks × threads | problem | best (ms) | Dense / best | downward | leaves | M2L | P2P | M2P + P2L |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 Max | 1×1 | cube f64 p = 3 | Dense 1,155 | 1.00 | 31 % | 66 % | 30 % | 60 % | 0 % |
| M3 Max | 1×1 | cube f64 p = 6 | Dense 3,617 | 1.00 | 72 % | 25 % | 72 % | 19 % | 0 % |
| M3 Max | 1×1 | cube f64 p = 8 | Rotation 6,537 | 1.25 | 82 % | 15 % | 81 % | 11 % | 0 % |
| M3 Max | 1×1 | cube f64 p = 12 | Rotation 15,827 | 1.95 | 89 % | 9 % | 88 % | 4 % | 0 % |
| M3 Max | 1×1 | cube f32 p = 3 | Dense 765 | 1.00 | 41 % | 55 % | 40 % | 46 % | 0 % |
| M3 Max | 1×1 | cube f32 p = 8 | Dense 4,779 | 1.00 | 82 % | 14 % | 82 % | 7 % | 0 % |
| M3 Max | 1×1 | Plummer f64 p = 3 | Dense 2,450 | 1.00 | 33 % | 65 % | 23 % | 26 % | 47 % |
| M3 Max | 1×1 | Plummer f64 p = 6 | Dense 8,206 | 1.00 | 60 % | 39 % | 50 % | 8 % | 38 % |
| M3 Max | 1×1 | Plummer f64 p = 8 | Rotation 14,649 | 1.18 | 67 % | 32 % | 57 % | 4 % | 35 % |
| M3 Max | 1×1 | Plummer f64 p = 12 | Rotation 34,034 | 1.69 | 73 % | 26 % | 65 % | 2 % | 30 % |
| M3 Max | 1×1 | Plummer f32 p = 3 | Dense 2,045 | 1.00 | 36 % | 63 % | 24 % | 17 % | 54 % |
| M3 Max | 1×1 | Plummer f32 p = 8 | Dense 12,057 | 1.00 | 63 % | 36 % | 51 % | 3 % | 41 % |
| M3 Max | 12×1 | cube f64 p = 3 | Dense 123 | 1.00 | 28 % | 58 % | 28 % | 53 % | 0 % |
| M3 Max | 12×1 | cube f64 p = 6 | Rotation 371 | 2.56 | 72 % | 22 % | 71 % | 17 % | 0 % |
| M3 Max | 12×1 | cube f64 p = 8 | Rotation 610 | 4.55 | 79 % | 15 % | 79 % | 11 % | 0 % |
| M3 Max | 12×1 | cube f64 p = 12 | Rotation 1,453 | 8.14 | 88 % | 9 % | 87 % | 4 % | 0 % |
| M3 Max | 12×1 | cube f32 p = 3 | Dense 79 | 1.00 | 36 % | 48 % | 35 % | 40 % | 0 % |
| M3 Max | 12×1 | cube f32 p = 8 | Rotation 528 | 2.44 | 83 % | 11 % | 83 % | 6 % | 0 % |
| M3 Max | 12×1 | Plummer f64 p = 3 | Dense 245 | 1.00 | 33 % | 61 % | 24 % | 24 % | 46 % |
| M3 Max | 12×1 | Plummer f64 p = 6 | Rotation 844 | 1.71 | 61 % | 37 % | 52 % | 7 % | 38 % |
| M3 Max | 12×1 | Plummer f64 p = 8 | Rotation 1,405 | 3.31 | 65 % | 33 % | 57 % | 4 % | 37 % |
| M3 Max | 12×1 | Plummer f64 p = 12 | Rotation 3,190 | 5.92 | 73 % | 27 % | 65 % | 2 % | 32 % |
| M3 Max | 12×1 | Plummer f32 p = 3 | Dense 199 | 1.00 | 35 % | 59 % | 24 % | 16 % | 54 % |
| M3 Max | 12×1 | Plummer f32 p = 8 | Rotation 1,291 | 1.73 | 67 % | 33 % | 57 % | 3 % | 39 % |
| locust | 1×1 | cube f64 p = 3 | Dense 2,494 | 1.00 | 15 % | 82 % | 15 % | 79 % | 0 % |
| locust | 1×1 | cube f64 p = 6 | Dense 5,127 | 1.00 | 54 % | 43 % | 54 % | 38 % | 0 % |
| locust | 1×1 | cube f64 p = 8 | Rotation 9,333 | 1.05 | 72 % | 25 % | 72 % | 21 % | 0 % |
| locust | 1×1 | cube f64 p = 12 | Rotation 22,353 | 2.21 | 85 % | 13 % | 85 % | 9 % | 0 % |
| locust | 1×1 | cube f32 p = 3 | Dense 1,092 | 1.00 | 26 % | 69 % | 25 % | 61 % | 0 % |
| locust | 1×1 | cube f32 p = 8 | Dense 5,338 | 1.00 | 76 % | 20 % | 76 % | 13 % | 0 % |
| locust | 1×1 | Plummer f64 p = 3 | Dense 4,060 | 1.00 | 25 % | 74 % | 15 % | 43 % | 38 % |
| locust | 1×1 | Plummer f64 p = 6 | Dense 10,590 | 1.00 | 52 % | 47 % | 41 % | 17 % | 38 % |
| locust | 1×1 | Plummer f64 p = 8 | Rotation 19,627 | 1.02 | 64 % | 34 % | 55 % | 9 % | 32 % |
| locust | 1×1 | Plummer f64 p = 12 | Rotation 46,180 | 1.88 | 74 % | 25 % | 65 % | 4 % | 28 % |
| locust | 1×1 | Plummer f32 p = 3 | Dense 2,745 | 1.00 | 29 % | 69 % | 16 % | 23 % | 56 % |
| locust | 1×1 | Plummer f32 p = 8 | Dense 14,067 | 1.00 | 58 % | 40 % | 45 % | 5 % | 45 % |
| locust | 72×1 | cube f64 p = 3 | Dense 38 | 1.00 | 17 % | 76 % | 17 % | 73 % | 0 % |
| locust | 72×1 | cube f64 p = 6 | Rotation 87 | 3.01 | 58 % | 39 % | 58 % | 33 % | 0 % |
| locust | 72×1 | cube f64 p = 8 | Rotation 139 | 6.90 | 72 % | 27 % | 72 % | 21 % | 0 % |
| locust | 72×1 | cube f64 p = 12 | Rotation 400 | 11.15 | 86 % | 11 % | 86 % | 8 % | 0 % |
| locust | 72×1 | cube f32 p = 3 | Dense 17 | 1.00 | 26 % | 65 % | 26 % | 57 % | 0 % |
| locust | 72×1 | cube f32 p = 8 | Rotation 111 | 3.40 | 82 % | 15 % | 81 % | 9 % | 0 % |
| locust | 72×1 | Plummer f64 p = 3 | Dense 74 | 1.00 | 28 % | 72 % | 20 % | 62 % | 40 % |
| locust | 72×1 | Plummer f64 p = 6 | Rotation 181 | 2.42 | 66 % | 49 % | 60 % | 21 % | 48 % |
| locust | 72×1 | Plummer f64 p = 8 | Rotation 295 | 5.23 | 81 % | 44 % | 75 % | 13 % | 47 % |
| locust | 72×1 | Plummer f64 p = 12 | Rotation 800 | 9.01 | 90 % | 30 % | 86 % | 5 % | 37 % |
| locust | 72×1 | Plummer f32 p = 3 | Dense 43 | 1.00 | 36 % | 72 % | 24 % | 38 % | 69 % |
| locust | 72×1 | Plummer f32 p = 8 | Rotation 265 | 2.43 | 87 % | 42 % | 80 % | 5 % | 52 % |
| Kathleen | 1×1 | cube f64 p = 3 | Dense 2,501 | 1.00 | 21 % | 73 % | 21 % | 66 % | 0 % |
| Kathleen | 1×1 | cube f64 p = 6 | Dense 8,943 | 1.00 | 72 % | 24 % | 72 % | 18 % | 0 % |
| Kathleen | 1×1 | cube f64 p = 8 | Rotation 16,758 | 1.23 | 82 % | 14 % | 82 % | 10 % | 0 % |
| Kathleen | 1×1 | cube f64 p = 12 | Rotation 44,016 | 2.70 | 90 % | 7 % | 89 % | 4 % | 0 % |
| Kathleen | 1×1 | cube f32 p = 3 | Dense 1,258 | 1.00 | 23 % | 68 % | 23 % | 53 % | 0 % |
| Kathleen | 1×1 | cube f32 p = 8 | Dense 10,781 | 1.00 | 82 % | 14 % | 81 % | 6 % | 0 % |
| Kathleen | 1×1 | Plummer f64 p = 3 | Dense 5,811 | 1.00 | 27 % | 70 % | 14 % | 26 % | 54 % |
| Kathleen | 1×1 | Plummer f64 p = 6 | Dense 20,617 | 1.00 | 60 % | 38 % | 49 % | 7 % | 39 % |
| Kathleen | 1×1 | Plummer f64 p = 8 | Rotation 37,765 | 1.17 | 69 % | 30 % | 57 % | 4 % | 35 % |
| Kathleen | 1×1 | Plummer f64 p = 12 | Rotation 93,040 | 2.26 | 77 % | 22 % | 67 % | 2 % | 28 % |
| Kathleen | 1×1 | Plummer f32 p = 3 | Dense 4,529 | 1.00 | 27 % | 70 % | 10 % | 15 % | 68 % |
| Kathleen | 1×1 | Plummer f32 p = 8 | Dense 29,216 | 1.00 | 61 % | 37 % | 47 % | 2 % | 46 % |
| Kathleen | 2×20 | cube f64 p = 3 | Dense 122 | 1.00 | 21 % | 47 % | 21 % | 42 % | 0 % |
| Kathleen | 2×20 | cube f64 p = 6 | Dense 325 | 1.00 | 64 % | 21 % | 63 % | 16 % | 0 % |
| Kathleen | 2×20 | cube f64 p = 8 | Rotation 601 | 1.18 | 77 % | 13 % | 77 % | 9 % | 0 % |
| Kathleen | 2×20 | cube f64 p = 12 | Rotation 1,550 | 2.83 | 87 % | 7 % | 86 % | 3 % | 0 % |
| Kathleen | 2×20 | cube f32 p = 3 | Dense 71 | 1.00 | 22 % | 42 % | 21 % | 31 % | 0 % |
| Kathleen | 2×20 | cube f32 p = 8 | Dense 371 | 1.00 | 76 % | 13 % | 75 % | 6 % | 0 % |
| Kathleen | 2×20 | Plummer f64 p = 3 | Dense 266 | 1.00 | 30 % | 55 % | 16 % | 18 % | 52 % |
| Kathleen | 2×20 | Plummer f64 p = 6 | Dense 805 | 1.00 | 57 % | 35 % | 41 % | 6 % | 44 % |
| Kathleen | 2×20 | Plummer f64 p = 8 | Rotation 1,451 | 1.11 | 67 % | 29 % | 51 % | 3 % | 40 % |
| Kathleen | 2×20 | Plummer f64 p = 12 | Rotation 3,515 | 2.26 | 76 % | 21 % | 61 % | 1 % | 33 % |
| Kathleen | 2×20 | Plummer f32 p = 3 | Dense 210 | 1.00 | 31 % | 57 % | 9 % | 11 % | 65 % |
| Kathleen | 2×20 | Plummer f32 p = 8 | Dense 1,137 | 1.00 | 62 % | 35 % | 41 % | 2 % | 51 % |
| Kathleen | 40×1 | cube f64 p = 3 | Dense 95 | 1.00 | 24 % | 69 % | 23 % | 63 % | 0 % |
| Kathleen | 40×1 | cube f64 p = 6 | Rotation 344 | 1.66 | 75 % | 22 % | 74 % | 17 % | 0 % |
| Kathleen | 40×1 | cube f64 p = 8 | Rotation 616 | 2.69 | 82 % | 14 % | 81 % | 10 % | 0 % |
| Kathleen | 40×1 | cube f64 p = 12 | Rotation 1,695 | 4.15 | 90 % | 7 % | 89 % | 4 % | 0 % |
| Kathleen | 40×1 | cube f32 p = 3 | Dense 47 | 1.00 | 25 % | 67 % | 24 % | 52 % | 0 % |
| Kathleen | 40×1 | cube f32 p = 8 | Rotation 528 | 1.46 | 85 % | 10 % | 85 % | 5 % | 0 % |
| Kathleen | 40×1 | Plummer f64 p = 3 | Dense 211 | 1.00 | 35 % | 70 % | 24 % | 38 % | 62 % |
| Kathleen | 40×1 | Plummer f64 p = 6 | Rotation 785 | 1.46 | 74 % | 44 % | 68 % | 9 % | 50 % |
| Kathleen | 40×1 | Plummer f64 p = 8 | Rotation 1,363 | 2.19 | 79 % | 38 % | 72 % | 5 % | 48 % |
| Kathleen | 40×1 | Plummer f64 p = 12 | Rotation 3,446 | 3.50 | 87 % | 30 % | 82 % | 2 % | 41 % |
| Kathleen | 40×1 | Plummer f32 p = 3 | Dense 170 | 1.00 | 28 % | 72 % | 16 % | 21 % | 74 % |
| Kathleen | 40×1 | Plummer f32 p = 8 | Rotation 1,266 | 1.26 | 80 % | 40 % | 72 % | 2 % | 52 % |

**C.2 Wall time per strategy and M2L per V pair** (ms; µs per pair of the slowest rank in
M2L; Dense's GFLOP/s per core from 2 (p + 1)⁴ flops per pair, per thread at 2 × 20):

| machine | ranks × threads | problem | Dense | Classes | Rotation | M2L µs per pair D / C / R | Dense GFLOP/s per core |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| M3 Max | 1 × 1 | cube f64 p = 3 | 1155.4 | 1733.3 | 1793.9 | 0.06 / 0.15 / 0.16 | 8.8 |
| M3 Max | 12 × 1 | cube f64 p = 3 | 122.6 | 169.1 | 177.4 | 0.06 / 0.16 / 0.18 | 8.0 |
| M3 Max | 1 × 1 | cube f64 p = 6 | 3617.4 | 4734.5 | 3942.6 | 0.43 / 0.62 / 0.49 | 11.2 |
| M3 Max | 12 × 1 | cube f64 p = 6 | 949.5 | 457.8 | 370.8 | 1.58 / 0.68 / 0.51 | 3.0 |
| M3 Max | 1 × 1 | cube f64 p = 8 | 8173.7 | 9980.9 | 6536.7 | 1.14 / 1.45 / 0.88 | 11.5 |
| M3 Max | 12 × 1 | cube f64 p = 8 | 2775.7 | 1011.1 | 609.9 | 5.16 / 1.71 / 0.94 | 2.5 |
| M3 Max | 1 × 1 | cube f64 p = 12 | 30865.8 | 34697.8 | 15827.4 | 4.77 / 5.41 / 2.32 | 12.0 |
| M3 Max | 12 × 1 | cube f64 p = 12 | 11831.2 | 4707.1 | 1452.8 | 22.33 / 8.78 / 2.47 | 2.6 |
| M3 Max | 1 × 1 | cube f32 p = 3 | 764.8 | 1235.7 | 1327.5 | 0.05 / 0.13 / 0.14 | 10.0 |
| M3 Max | 12 × 1 | cube f32 p = 3 | 78.7 | 121.5 | 128.8 | 0.05 / 0.14 / 0.15 | 9.5 |
| M3 Max | 1 × 1 | cube f32 p = 8 | 4779.2 | 6921.1 | 5741.6 | 0.65 / 1.00 / 0.81 | 20.2 |
| M3 Max | 12 × 1 | cube f32 p = 8 | 1289.0 | 660.5 | 527.9 | 2.33 / 1.11 / 0.85 | 5.6 |
| M3 Max | 1 × 1 | Plummer f64 p = 3 | 2450.2 | 3340.9 | 3444.8 | 0.06 / 0.15 / 0.16 | 8.8 |
| M3 Max | 12 × 1 | Plummer f64 p = 3 | 244.6 | 327.3 | 339.0 | 0.07 / 0.16 / 0.17 | 7.8 |
| M3 Max | 1 × 1 | Plummer f64 p = 6 | 8206.1 | 9990.3 | 8717.1 | 0.43 / 0.62 / 0.48 | 11.2 |
| M3 Max | 12 × 1 | Plummer f64 p = 6 | 1446.1 | 1002.5 | 844.4 | 1.27 / 0.70 / 0.52 | 3.8 |
| M3 Max | 1 × 1 | Plummer f64 p = 8 | 17253.9 | 19971.7 | 14648.5 | 1.15 / 1.44 / 0.88 | 11.5 |
| M3 Max | 12 × 1 | Plummer f64 p = 8 | 4652.6 | 2040.7 | 1405.4 | 4.77 / 1.67 / 0.94 | 2.7 |
| M3 Max | 1 × 1 | Plummer f64 p = 12 | 57632.5 | 63753.7 | 34034.2 | 4.76 / 5.40 / 2.31 | 12.0 |
| M3 Max | 12 × 1 | Plummer f64 p = 12 | 18867.2 | 8235.3 | 3189.6 | 20.70 / 8.17 / 2.46 | 2.8 |
| M3 Max | 1 × 1 | Plummer f32 p = 3 | 2045.0 | 2789.7 | 2936.7 | 0.05 / 0.13 / 0.14 | 10.0 |
| M3 Max | 12 × 1 | Plummer f32 p = 3 | 199.0 | 272.1 | 287.3 | 0.05 / 0.14 / 0.15 | 9.5 |
| M3 Max | 1 × 1 | Plummer f32 p = 8 | 12057.3 | 15494.8 | 13552.3 | 0.65 / 1.01 / 0.81 | 20.2 |
| M3 Max | 12 × 1 | Plummer f32 p = 8 | 2228.8 | 1527.9 | 1291.0 | 2.02 / 1.12 / 0.85 | 6.5 |
| locust | 1 × 1 | cube f64 p = 3 | 2493.6 | 3003.4 | 3110.5 | 0.06 / 0.15 / 0.16 | 8.3 |
| locust | 72 × 1 | cube f64 p = 3 | 38.0 | 44.8 | 46.1 | 0.07 / 0.15 / 0.17 | 7.3 |
| locust | 1 × 1 | cube f64 p = 6 | 5126.9 | 6868.6 | 5768.0 | 0.46 / 0.74 / 0.56 | 10.5 |
| locust | 72 × 1 | cube f64 p = 6 | 260.8 | 105.6 | 86.5 | 2.55 / 0.83 / 0.58 | 1.9 |
| locust | 1 × 1 | cube f64 p = 8 | 9766.2 | 14515.5 | 9332.8 | 1.18 / 1.96 / 1.11 | 11.1 |
| locust | 72 × 1 | cube f64 p = 8 | 959.1 | 235.1 | 138.9 | 10.50 / 2.28 / 1.15 | 1.3 |
| locust | 1 × 1 | cube f64 p = 12 | 49312.7 | 45491.6 | 22352.6 | 7.57 / 6.94 / 3.14 | 7.5 |
| locust | 72 × 1 | cube f64 p = 12 | 4459.9 | 1933.6 | 399.8 | 50.57 / 22.20 / 3.98 | 1.1 |
| locust | 1 × 1 | cube f32 p = 3 | 1092.1 | 1739.1 | 1928.9 | 0.05 / 0.15 / 0.18 | 11.2 |
| locust | 72 × 1 | cube f32 p = 3 | 16.7 | 26.2 | 28.2 | 0.05 / 0.16 / 0.18 | 10.6 |
| locust | 1 × 1 | cube f32 p = 8 | 5338.5 | 8763.4 | 7735.6 | 0.67 / 1.23 / 1.06 | 19.6 |
| locust | 72 × 1 | cube f32 p = 8 | 376.7 | 131.4 | 110.9 | 4.12 / 1.30 / 1.04 | 3.2 |
| locust | 1 × 1 | Plummer f64 p = 3 | 4060.2 | 4848.7 | 5024.5 | 0.06 / 0.15 / 0.16 | 7.9 |
| locust | 72 × 1 | Plummer f64 p = 3 | 74.4 | 90.3 | 81.5 | 0.06 / 0.15 / 0.16 | 7.9 |
| locust | 1 × 1 | Plummer f64 p = 6 | 10590.2 | 13521.8 | 11609.1 | 0.46 / 0.77 / 0.56 | 10.5 |
| locust | 72 × 1 | Plummer f64 p = 6 | 437.7 | 240.1 | 180.9 | 1.71 / 0.79 / 0.56 | 2.8 |
| locust | 1 × 1 | Plummer f64 p = 8 | 20042.8 | 27574.2 | 19627.5 | 1.18 / 1.97 / 1.14 | 11.1 |
| locust | 72 × 1 | Plummer f64 p = 8 | 1542.1 | 496.3 | 295.0 | 9.00 / 2.14 / 1.14 | 1.5 |
| locust | 1 × 1 | Plummer f64 p = 12 | 86687.5 | 81833.8 | 46179.8 | 7.38 / 6.87 / 3.15 | 7.7 |
| locust | 72 × 1 | Plummer f64 p = 12 | 7204.3 | 3226.0 | 799.7 | 45.00 / 18.26 / 3.75 | 1.3 |
| locust | 1 × 1 | Plummer f32 p = 3 | 2745.3 | 3758.3 | 4065.5 | 0.05 / 0.15 / 0.18 | 11.1 |
| locust | 72 × 1 | Plummer f32 p = 3 | 42.6 | 62.3 | 65.0 | 0.05 / 0.15 / 0.18 | 11.1 |
| locust | 1 × 1 | Plummer f32 p = 8 | 14066.8 | 19209.6 | 17668.1 | 0.67 / 1.21 / 1.05 | 19.5 |
| locust | 72 × 1 | Plummer f32 p = 8 | 642.3 | 298.1 | 264.8 | 3.08 / 1.24 / 1.03 | 4.3 |
| Kathleen | 1 × 1 | cube f64 p = 3 | 2501.4 | 3532.6 | 4066.6 | 0.09 / 0.26 / 0.34 | 5.9 |
| Kathleen | 2 × 20 | cube f64 p = 3 | 121.8 | 171.4 | 171.0 | 0.01 / 0.02 / 0.02 | 3.0 |
| Kathleen | 40 × 1 | cube f64 p = 3 | 95.3 | 135.3 | 148.8 | 0.13 / 0.36 / 0.47 | 4.0 |
| Kathleen | 1 × 1 | cube f64 p = 6 | 8943.4 | 10304.6 | 9597.0 | 1.07 / 1.29 / 1.17 | 4.5 |
| Kathleen | 2 × 20 | cube f64 p = 6 | 324.6 | 393.3 | 358.8 | 0.07 / 0.09 / 0.08 | 3.5 |
| Kathleen | 40 × 1 | cube f64 p = 6 | 569.4 | 379.7 | 343.7 | 3.05 / 1.79 / 1.60 | 1.6 |
| Kathleen | 1 × 1 | cube f64 p = 8 | 20659.7 | 22246.7 | 16758.5 | 2.91 / 3.18 / 2.27 | 4.5 |
| Kathleen | 2 × 20 | cube f64 p = 8 | 708.1 | 808.6 | 601.3 | 0.19 / 0.22 / 0.15 | 3.5 |
| Kathleen | 40 × 1 | cube f64 p = 8 | 1655.7 | 936.5 | 615.7 | 9.56 / 4.65 / 3.11 | 1.4 |
| Kathleen | 1 × 1 | cube f64 p = 12 | 118700.7 | 92128.5 | 44016.0 | 18.80 / 14.40 / 6.52 | 3.0 |
| Kathleen | 2 × 20 | cube f64 p = 12 | 4389.1 | 3184.9 | 1549.7 | 1.38 / 0.98 / 0.44 | 2.1 |
| Kathleen | 40 × 1 | cube f64 p = 12 | 7041.8 | 4856.0 | 1695.5 | 42.78 / 29.78 / 9.46 | 1.3 |
| Kathleen | 1 × 1 | cube f32 p = 3 | 1257.6 | 2495.8 | 3018.2 | 0.05 / 0.25 / 0.34 | 10.7 |
| Kathleen | 2 × 20 | cube f32 p = 3 | 71.4 | 110.2 | 130.9 | 0.01 / 0.02 / 0.02 | 5.0 |
| Kathleen | 40 × 1 | cube f32 p = 3 | 47.2 | 93.2 | 110.1 | 0.07 / 0.35 / 0.47 | 7.3 |
| Kathleen | 1 × 1 | cube f32 p = 8 | 10780.6 | 14518.3 | 14374.2 | 1.45 / 2.07 / 2.03 | 9.0 |
| Kathleen | 2 × 20 | cube f32 p = 8 | 370.6 | 518.8 | 502.7 | 0.09 / 0.14 / 0.13 | 7.1 |
| Kathleen | 40 × 1 | cube f32 p = 8 | 771.8 | 551.5 | 528.1 | 4.35 / 2.98 / 2.78 | 3.0 |
| Kathleen | 1 × 1 | Plummer f64 p = 3 | 5811.4 | 7431.9 | 8272.0 | 0.09 / 0.26 / 0.34 | 5.9 |
| Kathleen | 2 × 20 | Plummer f64 p = 3 | 266.3 | 323.9 | 354.0 | 0.01 / 0.02 / 0.03 | 2.9 |
| Kathleen | 40 × 1 | Plummer f64 p = 3 | 210.7 | 307.8 | 295.5 | 0.12 / 0.36 / 0.47 | 4.1 |
| Kathleen | 1 × 1 | Plummer f64 p = 6 | 20617.3 | 22725.2 | 21677.5 | 1.07 / 1.29 / 1.17 | 4.5 |
| Kathleen | 2 × 20 | Plummer f64 p = 6 | 804.7 | 905.2 | 865.8 | 0.07 / 0.09 / 0.08 | 3.5 |
| Kathleen | 40 × 1 | Plummer f64 p = 6 | 1142.4 | 959.5 | 784.6 | 2.31 / 1.83 / 1.63 | 2.1 |
| Kathleen | 1 × 1 | Plummer f64 p = 8 | 44052.3 | 46546.1 | 37765.2 | 2.94 / 3.20 / 2.28 | 4.5 |
| Kathleen | 2 × 20 | Plummer f64 p = 8 | 1614.9 | 1798.3 | 1451.2 | 0.19 / 0.23 / 0.15 | 3.5 |
| Kathleen | 40 × 1 | Plummer f64 p = 8 | 2986.9 | 1871.4 | 1363.2 | 7.89 / 4.60 / 3.14 | 1.7 |
| Kathleen | 1 × 1 | Plummer f64 p = 12 | 210661.5 | 168142.7 | 93039.9 | 18.79 / 14.32 / 6.51 | 3.0 |
| Kathleen | 2 × 20 | Plummer f64 p = 12 | 7942.3 | 5997.4 | 3515.4 | 1.36 / 0.96 / 0.46 | 2.1 |
| Kathleen | 40 × 1 | Plummer f64 p = 12 | 12062.2 | 8675.7 | 3445.7 | 40.01 / 27.19 / 9.35 | 1.4 |
| Kathleen | 1 × 1 | Plummer f32 p = 3 | 4529.0 | 6488.3 | 7302.2 | 0.05 / 0.25 / 0.34 | 10.6 |
| Kathleen | 2 × 20 | Plummer f32 p = 3 | 210.3 | 278.8 | 315.9 | 0.00 / 0.02 / 0.03 | 6.7 |
| Kathleen | 40 × 1 | Plummer f32 p = 3 | 170.1 | 240.6 | 272.6 | 0.07 / 0.35 / 0.46 | 7.8 |
| Kathleen | 1 × 1 | Plummer f32 p = 8 | 29216.1 | 35132.9 | 35149.9 | 1.45 / 2.07 / 2.06 | 9.0 |
| Kathleen | 2 × 20 | Plummer f32 p = 8 | 1137.4 | 1364.9 | 1341.0 | 0.10 / 0.14 / 0.13 | 6.6 |
| Kathleen | 40 × 1 | Plummer f32 p = 8 | 1593.4 | 1346.1 | 1265.7 | 3.77 / 2.95 / 2.80 | 3.5 |

**C.3 The model of §2.10, every cell** (ms and ×; "–" where the FMA variant was not run
at the node's processes):

| machine | ranks × threads | problem | best today (ms) | Batched, bit for bit (*model*, ms) | × | with FMA | × | with BLAS | × |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 Max | 1×1 | cube f64 p = 8 | Rotation 6,537 | 4,346 | 1.50 | 3,537 | 1.85 | 2,106 | 3.10 |
| M3 Max | 1×1 | cube f64 p = 12 | Rotation 15,827 | 14,577 | 1.09 | 11,255 | 1.41 | 3,971 | 3.99 |
| M3 Max | 1×1 | cube f32 p = 8 | Dense 4,779 | 2,595 | 1.84 | 2,371 | 2.02 | 1,242 | 3.85 |
| M3 Max | 1×1 | cube f64 p = 3 | Dense 1,155 | 953 | 1.21 | 923 | 1.25 | 929 | 1.24 |
| M3 Max | 1×1 | plummer f64 p = 8 | Rotation 14,649 | 11,173 | 1.31 | 9,899 | 1.48 | 7,644 | 1.92 |
| M3 Max | 1×1 | plummer f64 p = 12 | Rotation 34,034 | 32,143 | 1.06 | 26,911 | 1.26 | 15,437 | 2.20 |
| M3 Max | 1×1 | plummer f32 p = 8 | Dense 12,057 | 8,607 | 1.40 | 8,255 | 1.46 | 6,476 | 1.86 |
| M3 Max | 1×1 | plummer f64 p = 3 | Dense 2,450 | 2,132 | 1.15 | 2,084 | 1.18 | 2,093 | 1.17 |
| M3 Max | 12×1 | cube f64 p = 8 | Rotation 610 | 468 | 1.30 | – | – | 255 | 2.39 |
| M3 Max | 12×1 | cube f64 p = 12 | Rotation 1,453 | 1,516 | 0.96 | – | – | 639 | 2.27 |
| M3 Max | 12×1 | cube f32 p = 8 | Rotation 528 | 259 | 2.04 | – | – | 135 | 3.90 |
| M3 Max | 12×1 | cube f64 p = 3 | Dense 123 | 104 | 1.17 | – | – | 102 | 1.20 |
| M3 Max | 12×1 | plummer f64 p = 8 | Rotation 1,405 | 1,174 | 1.20 | – | – | 821 | 1.71 |
| M3 Max | 12×1 | plummer f64 p = 12 | Rotation 3,190 | 3,301 | 0.97 | – | – | 1,849 | 1.72 |
| M3 Max | 12×1 | plummer f32 p = 8 | Rotation 1,291 | 841 | 1.54 | – | – | 633 | 2.04 |
| M3 Max | 12×1 | plummer f64 p = 3 | Dense 245 | 213 | 1.15 | – | – | 210 | 1.17 |
| locust | 1×1 | cube f64 p = 8 | Rotation 9,333 | 6,743 | 1.38 | 5,746 | 1.62 | 5,196 | 1.80 |
| locust | 1×1 | cube f64 p = 12 | Rotation 22,353 | 20,190 | 1.11 | 15,660 | 1.43 | 13,426 | 1.66 |
| locust | 1×1 | cube f32 p = 8 | Dense 5,338 | 3,614 | 1.48 | 2,932 | 1.82 | 2,569 | 2.08 |
| locust | 1×1 | cube f64 p = 3 | Dense 2,494 | 2,318 | 1.08 | 2,286 | 1.09 | 2,301 | 1.08 |
| locust | 1×1 | plummer f64 p = 8 | Rotation 19,627 | 15,297 | 1.28 | 13,728 | 1.43 | 12,862 | 1.53 |
| locust | 1×1 | plummer f64 p = 12 | Rotation 46,180 | 42,656 | 1.08 | 35,521 | 1.30 | 32,001 | 1.44 |
| locust | 1×1 | plummer f32 p = 8 | Dense 14,067 | 11,324 | 1.24 | 10,249 | 1.37 | 9,678 | 1.45 |
| locust | 1×1 | plummer f64 p = 3 | Dense 4,060 | 3,758 | 1.08 | 3,708 | 1.09 | 3,732 | 1.09 |
| locust | 72×1 | cube f64 p = 8 | Rotation 139 | 123 | 1.13 | – | – | 118 | 1.18 |
| locust | 72×1 | cube f64 p = 12 | Rotation 400 | 338 | 1.18 | – | – | 244 | 1.64 |
| locust | 72×1 | cube f32 p = 8 | Rotation 111 | 57 | 1.95 | – | – | 43 | 2.57 |
| locust | 72×1 | cube f64 p = 3 | Dense 38 | 35 | 1.09 | – | – | 35 | 1.10 |
| locust | 72×1 | plummer f64 p = 8 | Rotation 295 | 261 | 1.13 | – | – | 250 | 1.18 |
| locust | 72×1 | plummer f64 p = 12 | Rotation 800 | 711 | 1.12 | – | – | 511 | 1.57 |
| locust | 72×1 | plummer f32 p = 8 | Rotation 265 | 139 | 1.91 | – | – | 106 | 2.49 |
| locust | 72×1 | plummer f64 p = 3 | Dense 74 | 67 | 1.10 | – | – | 67 | 1.11 |
| Kathleen | 1×1 | cube f64 p = 8 | Rotation 16,758 | 7,352 | 2.28 | 6,199 | 2.70 | 4,550 | 3.68 |
| Kathleen | 1×1 | cube f64 p = 12 | Rotation 44,016 | 21,257 | 2.07 | 17,090 | 2.58 | 10,338 | 4.26 |
| Kathleen | 1×1 | cube f32 p = 8 | Dense 10,781 | 4,418 | 2.44 | 4,147 | 2.60 | 2,812 | 3.83 |
| Kathleen | 1×1 | cube f64 p = 3 | Dense 2,501 | 2,197 | 1.14 | 2,161 | 1.16 | 2,128 | 1.18 |
| Kathleen | 1×1 | plummer f64 p = 8 | Rotation 37,765 | 22,844 | 1.65 | 21,027 | 1.80 | 18,429 | 2.05 |
| Kathleen | 1×1 | plummer f64 p = 12 | Rotation 93,040 | 57,251 | 1.63 | 50,686 | 1.84 | 40,051 | 2.32 |
| Kathleen | 1×1 | plummer f32 p = 8 | Dense 29,216 | 19,187 | 1.52 | 18,759 | 1.56 | 16,657 | 1.75 |
| Kathleen | 1×1 | plummer f64 p = 3 | Dense 5,811 | 5,329 | 1.09 | 5,272 | 1.10 | 5,220 | 1.11 |
| Kathleen | 40×1 | cube f64 p = 8 | Rotation 616 | 298 | 2.07 | 255 | 2.42 | 207 | 2.98 |
| Kathleen | 40×1 | cube f64 p = 12 | Rotation 1,695 | 864 | 1.96 | 706 | 2.40 | 522 | 3.25 |
| Kathleen | 40×1 | cube f32 p = 8 | Rotation 528 | 175 | 3.02 | 165 | 3.21 | 123 | 4.31 |
| Kathleen | 40×1 | cube f64 p = 3 | Dense 95 | 89 | 1.07 | – | – | 80 | 1.19 |
| Kathleen | 40×1 | plummer f64 p = 8 | Rotation 1,363 | 730 | 1.87 | 646 | 2.11 | 552 | 2.47 |
| Kathleen | 40×1 | plummer f64 p = 12 | Rotation 3,446 | 1,912 | 1.80 | 1,614 | 2.13 | 1,268 | 2.72 |
| Kathleen | 40×1 | plummer f32 p = 8 | Rotation 1,266 | 539 | 2.35 | 518 | 2.44 | 433 | 2.93 |
| Kathleen | 40×1 | plummer f64 p = 3 | Dense 211 | 197 | 1.07 | – | – | 176 | 1.19 |

## Appendix D. Corrections to the briefs (factual errors, made in this PR)

- docs/phase6/README.md, "Design decisions", BLAS inside rayon: "rlst 0.9.0 already links
  the system BLAS and LAPACK" is out of date. The workspace uses rlst 0.10.2, which calls
  the Fortran symbols through the `blas` and `lapack` crates; its `blas-src` and
  `lapack-src` are dev-dependencies, `Cargo.lock` has no `*-src` crate, and
  tools/kathleen/README.md and tools/gh200/spack.yaml record that no workspace crate links
  a BLAS. Corrected to say so.
- docs/phase6/T1-optimisation-design.md, revision note: "45–52 µs on the Plummer sphere and
  the clusters": phase5s-kathleen.md §10 gives 45–52 µs for the Plummer sphere only (the
  cube 21–23 µs, the clusters not stated). Corrected.

Checked and found right, with a better source: the README's and T1's "leaves 42–79 % at
p = 3, downward 67–92 % at p = 8" at 160 ranks are not printed in phase5s-kathleen.md but
are those of job 604159's raw output (§1.2). The README's C6.9 figure "12× those of the
uniform trees at 10⁶ and 4 × 10⁶ points" is 12.2× against 10⁶ and 15.5× against 4 × 10⁶
(f64 p = 8; phase5s-kathleen.md §11's 1,083 against 89 and 70 ms); left as it is ("12×"
is the smaller of the two).
