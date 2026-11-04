# Helmholtz CH6.2a: the waves buffer bounded

Helmholtz task CH6.2a ([docs/helmholtz/tasks/CH6.2a.md](../../docs/helmholtz/tasks/CH6.2a.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§10 CH6.2, §2 (constraints 2, 7, 9); CH6.1e's report §7 and its question 4 (signed off
2026-11-03: the waves buffer bounded and counted in the node budget before CH6.2 takes
times). Measured on Kathleen (one exclusive node of 2 × 20 Cascade Lake cores, Intel Xeon
Gold 6248, 192 GB; `tools/kathleen/`), job 628518 on node-c11a-083 (source 7a51393, whose
code is the branch's; only docs differed), and job 628529 on node-c11b-023 (the budgets at
40 ranks, §4).
Release build, rustc 1.99.0, Open MPI 4.1.6, BLAS threads 1, the P2P kernel `Auto`
(AVX-512). Commands: `tools/helmholtz/ch6-2a-runs.sh <out> budgets ranks` (§6).

## Summary

- **The level call by chunks.** Under `HelmholtzM2l::PlaneWave` the level call now cuts the
  level, in index order, into chunks of whole parents whose distinct V sources' waves fit
  the buffer, and per chunk, vector and frame computes those sources' waves, then merges
  and shifts the chunk's parents. A box's waves depend on its multipole alone and every
  accumulator keeps its order, so **the output is the same bit for bit at every bound**:
  one checksum per case at the whole level, 4 GiB, 1 GiB, 256 MB and the floor, at
  1 rank × 40 threads and at 40 ranks × 1 thread (§2, §4).
- **The bound.** `HelmholtzFmmBuilder::plane_wave_budget` is bytes per node (default
  `DEFAULT_PLANE_WAVE_NODE_BUDGET`, 4 GiB in this branch, for sign-off), divided by the
  ranks per node that `build` agrees from one `split_shared` (proposal §2, constraint 7);
  the operator's floor is one parent's 216 slots per level. `plane_wave_bytes` counts the
  buffer, a position per box and the chunk's source list with the tables and the scratch.
- **The memory falls with the bound** (§2). At one rank the whole level's waves took 8.9 GB
  (W1, kD = 500) and 11.7 GB (W3, kD = 100), peak resident 14.6 and 17.1 GB; at 4 GiB the
  peak is 10.0 and 9.7 GB, at 1 GiB 6.8 and 6.5 GB, against Rotation's 5.4 and 5.3 GB.
- **The cost is the recomputation of a chunk's halo** (§3). At one rank × 40 threads the
  4 GiB default took 1.03× and 1.04× the whole level's evaluation (1.05× and 1.16× the
  waves), 1 GiB 1.05× and 1.11×, 256 MB 1.34× and 1.43×; the floor 13× and 28× (10× and
  22× the waves in 13,572 and 32,121 chunks): a correctness floor, not a setting.
- **At 40 ranks per node the default is tight** (§4): 4 GiB is 107 MB per rank, which took
  1.16× (W1) and 1.28× (W3) the whole level's evaluation (1.84× and 1.76× the waves on rank
  0); 8 GiB 1.04× and 1.11×, 16 GiB 1.01× and 1.06×. The default is question 1 of the PR.

## 1. Method

The harness is CH3.1c's `nd_fmm_validate::helmholtz_accuracy` with this task's
`Case::plane_wave_budget` (the example's `--plane-wave-budget MB|whole`) and `Row`'s
`plane_wave_buffer` and `plane_wave_counts` (`HelmholtzOperator::plane_wave_counts`: the
level calls, the chunks and the waves computed, boxes × frames × vectors, this rank's). The
cases are CH6.1e's large ones at ε = 10⁻⁶ under the rule (iprec 2 on every level with a
quadrature): W1 at kD = 500 (N = 1,989,437; waves on levels 4–7) and W3 at kD = 100
(N = 4,031,442; levels 2–6), complex charges, gradients, 1,000 sampled targets. One
evaluation per launch; the whole level and the default ran twice, first and last, which
gives the run-to-run spread. Rank 0's peak resident size is VmHWM at the end of the launch,
the oracle's memory included. Bounds in MB are MiB (`--plane-wave-budget 1024` is 1 GiB).

## 2. Bit for bit, bytes and memory (one rank × 40 threads)

| Case | bound | `plane_wave_bytes` MB | waves MB | chunks / calls | peak MB | checksum |
| --- | --- | ---: | ---: | --- | ---: | --- |
| W1, kD = 500 | Rotation | – | – | – | 5,406 | afecb3102e373417 |
| | whole | 9,220 | 8,937 | 4 / 4 | 14,643 | 30d1687b75dca4d0 |
| | 4 GiB (default) | 4,578 | 4,295 | 7 / 4 | 10,046 | 30d1687b75dca4d0 |
| | 1 GiB | 1,357 | 1,074 | 22 / 4 | 6,793 | 30d1687b75dca4d0 |
| | 256 MiB | 551 | 268 | 112 / 4 | 5,948 | 30d1687b75dca4d0 |
| | floor (0) | 348 | 65 | 13,572 / 4 | 5,790 | 30d1687b75dca4d0 |
| W3, kD = 100 | Rotation | – | – | – | 5,282 | 1418dae5eeeeb43f |
| | whole | 11,945 | 11,719 | 5 / 5 | 17,147 | 4041e7524c66aaac |
| | 4 GiB (default) | 4,520 | 4,295 | 8 / 5 | 9,720 | 4041e7524c66aaac |
| | 1 GiB | 1,299 | 1,074 | 21 / 5 | 6,495 | 4041e7524c66aaac |
| | 256 MiB | 494 | 268 | 110 / 5 | 5,685 | 4041e7524c66aaac |
| | floor (0) | 248 | 23 | 32,121 / 5 | 5,448 | 4041e7524c66aaac |

The errors are the same at every bound (they are the same output): W1 φ 1.45e-7, ∇φ
3.73e-7 (Rotation 1.49e-7, 3.79e-7); W3 φ 1.02e-7, ∇φ 2.46e-7 (Rotation 1.01e-7,
2.50e-7). The whole level's 8.9 and 11.7 GB of waves are CH6.1e's 9.0 and 11.8 GB of
`plane_wave_bytes` (§7 there); the waves are now computed for V sources only, so a level
call needs no waves of boxes no pair reads. The bound holds the waves within it except at
the floor, whose 216 boxes per level the bound cannot go below; the positions (4 bytes per
box of the largest level with waves) lie outside it.

## 3. Time and recomputation (one rank × 40 threads)

Evaluation seconds, the waves computed against the whole level's, and the time against the
mean of the whole level's two runs:

| Case | bound | evaluate s | waves | ÷ whole (waves) | ÷ whole (time) |
| --- | --- | --- | ---: | ---: | ---: |
| W1, kD = 500 | Rotation | 41.69 | – | – | 1.96 |
| | whole | 21.12, 21.41 | 745,320 | 1 | 1 |
| | 4 GiB (default) | 22.12, 21.46 | 784,296 | 1.052 | 1.025 |
| | 1 GiB | 22.43 | 895,536 | 1.202 | 1.055 |
| | 256 MiB | 28.49 | 1,181,496 | 1.585 | 1.340 |
| | floor | 286.67 | 7,717,128 | 10.35 | 13.5 |
| W3, kD = 100 | Rotation | 37.17 | – | – | 3.64 |
| | whole | 10.28, 10.16 | 898,752 | 1 | 1 |
| | 4 GiB (default) | 10.61, 10.61 | 1,038,768 | 1.156 | 1.038 |
| | 1 GiB | 11.38 | 1,265,088 | 1.408 | 1.114 |
| | 256 MiB | 14.62 | 1,873,200 | 2.084 | 1.431 |
| | floor | 285.97 | 19,994,304 | 22.25 | 28.0 |

The run-to-run spread of the whole level and the default is 1.4 % and 3.1 % (W1), 1.2 %
and 0 % (W3). A chunk of Morton-consecutive parents reads the children of their
neighbours, so a small chunk recomputes a halo of sources that its neighbours' chunks
compute again; the chunk's pool regions add a fixed cost per chunk, which dominates at the
floor (216 sources over 40 threads, three frames, 13,572 chunks). The recomputation is a
cost of the outgoing step alone; merge and shift does the same work at every bound.

## 4. Several ranks per node (40 ranks × 1 thread)

Each rank's bound is the budget over the 40 ranks per node: 4 GiB is 107 MB per rank,
8 GiB 215 MB, 16 GiB 429 MB. The whole level is each rank's own and ghost boxes; the
counts and the peak are rank 0's, the time the evaluation's (the slowest rank's). Job
628518 ran the whole level and the default once (node-c11a-083); job 628529
(node-c11b-023) ran the whole level, 4, 8 and 16 GiB in that order twice:

| Case | budget per node | waves MB (rank 0) | chunks / calls | waves (rank 0) | evaluate s (628529) | ÷ whole (waves) | ÷ whole (time) | peak MB (rank 0) |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| W1, kD = 500 | whole | 345 | 4 / 4 | 34,968 | 40.47, 40.71 | 1 | 1 | 1,510, 1,445 |
| | 16 GiB | 345 | 4 / 4 | 34,968 | 41.40, 40.29 | 1 | 1.006 | 1,509, 1,419 |
| | 8 GiB | 215 | 7 / 4 | 45,936 | 41.99, 42.26 | 1.314 | 1.038 | 1,315, 1,316 |
| | 4 GiB (default) | 107 | 19 / 4 | 64,488 | 46.82, 47.45 | 1.844 | 1.161 | 1,244, 1,221 |
| W3, kD = 100 | whole | 418 | 5 / 5 | 34,224 | 12.65, 12.57 | 1 | 1 | 934, 936 |
| | 16 GiB | 418 | 5 / 5 | 34,224 | 13.33, 13.29 | 1 | 1.056 | 936, 938 |
| | 8 GiB | 215 | 7 / 5 | 45,600 | 13.97, 13.94 | 1.332 | 1.107 | 732, 733 |
| | 4 GiB (default) | 107 | 12 / 5 | 60,336 | 16.39, 15.93 | 1.763 | 1.282 | 624, 625 |

Job 628518 gave the whole level 39.70 and 12.56 s and the default 47.05 and 16.00 s (1.19×
and 1.27×), the same within 3 %. The checksums are the one-rank runs' at every budget, in
both jobs (W1 `30d1687b75dca4d0`, W3 `4041e7524c66aaac`): rank invariance at 40 ranks. The
repeats agree within 0.2–2.9 %.

- **4 GiB costs 16–28 %**, the halo of 12–19 small chunks per rank (1.76–1.84× rank 0's
  waves); it saves rank 0 about 250 and 310 MB of peak (W1, W3), about 10–12 GB per node.
- **8 GiB costs 4 % and 11 %** (1.31–1.33× the waves) and saves about 160 and 200 MB per
  rank, about 6–8 GB per node.
- **16 GiB holds rank 0's whole level** in both cases. W1 costs nothing; W3's 5.6 % in both
  repeats, with rank 0's work and memory identical to the whole level's, is most likely a
  rank whose levels exceed 429 MB and chunk (the counts are rank 0's alone).

The whole level holds at most 345–418 MB of waves per rank on rank 0 here, about 14–17 GB
per node; Kathleen's nodes have 192 GB.

## 5. Tests

- `fmm-exec/tests/helmholtz_fmm.rs`, `check_plane_wave_buffer` (from `check_plane_wave`; in
  debug builds on the clusters scenario, whose finest level with waves has more sources
  than the floor holds, in release on both): the floor, twice the floor's buffer and (in
  release) no bound bit for bit the default budget's output; the floor at 2 threads; on
  several ranks the floor's one-rank FMM bit for bit; the counts (the floor's chunks and
  waves at least the default's, twice the floor's chunks between the two), agreed over the
  ranks; and on one rank that the floor chunked a level call (8 chunks against 4 on the
  clusters). The node budget's arithmetic (`plane_wave_buffer` = budget / ranks per node).
- `fmm-exec/tests/helmholtz/plane_wave.rs`, `the_waves_buffer_is_bounded`: the buffer's
  size and `plane_wave_bytes` at eight bounds from 0 to `usize::MAX` against the formula
  (each level's capacity clamped to [216, its boxes]), in either order of the calls, nothing
  under Rotation.
- One-rank debug time of `helmholtz_fmm` on the M6: 195 s (188 s after CH6.1e).

## 6. Reproducing

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/helmholtz/ch6-2a-runs.sh <out> budgets ranks
RANK_BUDGETS="whole default 8192 16384 whole default 8192 16384" \
    tools/helmholtz/ch6-2a-runs.sh <out> ranks
```

On Kathleen inside a one-node job (`tools/kathleen/submit.sh --time=04:00:00 --
run.sbatch '<build>' '<script>'`; jobs 628518 and 628529), `THREADS=40`, `RANKS=40`.
