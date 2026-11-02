# Helmholtz CH3.2d: the M2L rule per level

Helmholtz task CH3.2d ([docs/helmholtz/tasks/CH3.2d.md](../../docs/helmholtz/tasks/CH3.2d.md));
proposal [docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md)
§5.3, §6.4 and §10 CH3.2; the per-pair and end-to-end data before it in
[helmholtz-ch3-2c.md](helmholtz-ch3-2c.md). Measured on 2026-11-02 on three machines: the M6
(Mac mini, Apple M6, 24 GB; [tools/m6/machine.md](../../tools/m6/machine.md)), locust (one
Neoverse-V2 core of the GH200; [tools/gh200/machine.md](../../tools/gh200/machine.md)) and
Kathleen (one Cascade Lake core, Xeon Gold 6248, node-c11a-089, job 618430;
[tools/kathleen/machine.md](../../tools/kathleen/machine.md)). Release builds, rustc 1.99.0,
one rank, one thread, f64, BLAS threads 1, at 718306c (the rule's constants were set from
these runs afterwards, 16e8aa4; §1). The times are indicative (§4); nothing here is timed
for a gate.

## Summary

- **The rule.** `HelmholtzM2l::Auto` runs the fused Batched M2L on each level whose degree
  p_l is at most a crossover and whose tables fit the dense budget (from the finest level
  up), and Rotation on the others. The crossover is a function of the machine class only
  (constraint 9): the column kernel's ISA and the multipole's planes (one for real k with
  real charges, two otherwise), `auto_batched_max_degree`:

  | ISA | two planes | one plane | from |
  | --- | ---: | ---: | --- |
  | NEON | 10 | 12 | the end-to-end sweeps on the M6 and locust (§2) |
  | AVX-512 | 15 | 15 | the end-to-end sweeps on Kathleen (§2) |
  | AVX2 | 11 | 12 | the end-to-end sweeps on Kathleen with `--p2p avx2` (§2) |
  | scalar | 6 | 8 | the cost per pair on the M6 and Kathleen (§3); no end-to-end run |

  Auto's column kernel runs fused (one fma per step) unless a mode is named, except the
  scalar kernel in a build without a hardware fma (x86_64 without the `fma` target
  feature, Kathleen's build), which runs separate: there `mul_add` is a library call, and
  the fused scalar kernel takes 2.4–50× Rotation's time per pair at every degree (§3),
  `auto_m2l_mul_add`. Found by §3's Kathleen run after the end-to-end runs; it changes
  nothing on the vector ISAs.

  `HelmholtzFmmBuilder::build` agrees the crossover over the ranks (the smallest), so
  every rank chooses the same levels; each level then runs Batched's or Rotation's level
  call unchanged, so the output is deterministic and rank-invariant as theirs (the tests,
  the PR).
- **Against the best choice (§2).** For each case the sweep ran Auto at every crossover
  that chooses differently, from Batched on every level with tables to Rotation on every
  level: the best of these is the best per-level choice of the family "Batched on the
  finest levels", both uniform settings included. The rule's evaluation time is the best
  one on 34 of the 36 case × machine rows and within 1.001 on one more (Kathleen AVX2, the
  cube at 10⁻⁶ with real charges, where Batched and Rotation tie). The one miss: **Kathleen
  AVX-512, W1 at kD = 30 and
  ε = 10⁻⁶, 1.106**, past the stated margin of 1.10 (run spread 2 %): the rule batches a
  sparse W1 level at p = 15, where Rotation is 10 % faster, while the W4 cube at p = 15
  needs Batched (1.17 slower with Rotation), so no crossover by degree alone meets both.
  Accepted by the user during the work (2026-11-02, "Accept, follow-up"); the sparse-level
  term is CH4.0 in the roadmap.
- **What it gains.** Against Rotation, the default: 0.71–0.95× of Rotation's evaluation
  time on the M6 where the rule batches (Rotation's own choice elsewhere), 0.78–0.97× on
  locust, 0.40–0.99× on Kathleen with AVX-512 (1.10 on the miss) and 0.54–0.89× with AVX2.
  Against Batched on every level that fits: down to 0.38× (locust, W1 at kD = 30 and
  10⁻⁶), since the rule leaves the high-degree levels to Rotation.
- **Accuracy.** Every choice of a case printed the same errors of φ and ∇φ against the
  sampled direct sum (three significant digits; the script of §4 compares them on every row
  of a case, on every machine): the rule changes time only. The FMM's output under each
  choice is Batched's and Rotation's of CH3.2c level by level (the tests).
- **f64 only.** The crossovers were measured in f64 (D3: f64 until H7); `Auto` in f32 takes
  the same constants, unmeasured.
- The constants were fitted to the same nine cases they are checked on (no held-out
  workload); within the sweeps several crossovers give the same choice, and the table's
  values are the ones the data decide where they do, the per-pair crossovers of CH3.2c and
  §3 where they do not (for example NEON's 10 with two planes: 9 and 10 choose alike here).

## 1. The constants

The runs ran the first constants (NEON and AVX2 10 and 12, AVX-512 15 and 18, scalar 6 and
8, from CH3.2c's cost per pair). The sweeps measure every choice, so a crossover's time on a
case is the time of the choice it makes (computed from the degrees and the budget as
`BatchedM2l::select_within` does), and two constants were changed from the data:

- AVX-512 with one plane, 18 → 15: on W1 at kD = 100 and 10⁻³ with real charges, 18
  batches the level of p = 18 too (4.74 s against 4.62 s, 1.026); 15 gives the best choice
  there and on the W4 cube at 10⁻⁶ with real charges (p = 15).
- AVX2 with two planes, 10 → 11: on W1 at kD = 100 and 10⁻³ the level of p = 11 gains
  (7.30 s against 7.65 s, 1.048 with 10); 12 would batch W1's level of p = 12 at kD = 30
  and 10⁻³, slower (0.17 s against 0.15 s).

## 2. End to end

`helmholtz_accuracy --m2l rotation,auto,sweep` and again in the order `sweep,auto,rotation`
on CH3.2a's cases (W1 at kD = 30 and 100, the W4 cube of 10⁵ points at kD = 10, each at
ε = 10⁻³ and 10⁻⁶, W3 at kD = 30 at 10⁻³; complex charges) and two with real charges (one
plane: the cube at 10⁻⁶, W1 at kD = 100 at 10⁻³); the default P2P kernel of each machine,
the default degree rule and dense budget (1 GiB). The evaluation's wall time per choice is
the mean of all its runs (a choice that two crossovers make, or Rotation and Auto below
every degree, is one choice), in seconds. "rule" is the choice of the final constants (§1);
"best" the fastest choice; "all Batched" Batched on every level that fits; "spread" the
largest difference between the runs of one choice of the case.

### M6 (NEON)

| case | ε | charges | p by level | rule (crossover: levels) | rule s | best | best s | Rotation s | all Batched s | rule / best | spread |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| W4 cube (N = 100000) kD = 10 | ε = 1e-3 | complex | 0×2, 7×3 | ≤ 10: batched-fused 2,3,4 | 0.65 | batched-fused 2,3,4 | 0.65 | 0.91 | 0.65 | 1.000 | 2% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | complex | 0×2, 15×3 | ≤ 10: rotation | 4.22 | rotation | 4.22 | 4.22 | 6.34 | 1.000 | 1% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | real | 0×2, 15×3 | ≤ 12: rotation | 3.42 | rotation | 3.42 | 3.42 | 3.72 | 1.000 | 1% |
| W1 kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8 | ≤ 10: batched-fused 3 | 0.06 | batched-fused 3 | 0.06 | 0.07 | 0.07 | 1.000 | 0% |
| W1 kD = 30 | ε = 1e-6 | complex | 0×2, 16, 15 | ≤ 10: rotation | 0.24 | rotation | 0.24 | 0.24 | 0.41 | 1.000 | 5% |
| W1 kD = 100 | ε = 1e-3 | complex | 0×2, 30, 18, 11, 7 | ≤ 10: batched-fused 5 | 2.31 | batched-fused 5 | 2.31 | 2.42 | 3.06 | 1.000 | 2% |
| W1 kD = 100 | ε = 1e-6 | complex | 0×2, 35, 22, 15×2 | ≤ 10: rotation | 6.12 | rotation | 6.12 | 6.12 | 8.78 | 1.000 | 1% |
| W1 kD = 100 | ε = 1e-3 | real | 0×2, 30, 18, 11, 7 | ≤ 12: batched-fused 4,5 | 1.71 | batched-fused 4,5 | 1.71 | 2.03 | 2.00 | 1.000 | 1% |
| W3 (N = 108849) kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8, 7 | ≤ 10: batched-fused 3,4 | 0.72 | batched-fused 3,4 | 0.72 | 0.99 | 0.72 | 1.000 | 1% |

### locust (NEON)

| case | ε | charges | p by level | rule (crossover: levels) | rule s | best | best s | Rotation s | all Batched s | rule / best | spread |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| W4 cube (N = 100000) kD = 10 | ε = 1e-3 | complex | 0×2, 7×3 | ≤ 10: batched-fused 2,3,4 | 1.58 | batched-fused 2,3,4 | 1.58 | 2.03 | 1.58 | 1.000 | 1% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | complex | 0×2, 15×3 | ≤ 10: rotation | 8.31 | rotation | 8.31 | 8.31 | 16.27 | 1.000 | 1% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | real | 0×2, 15×3 | ≤ 12: rotation | 6.69 | rotation | 6.69 | 6.69 | 10.04 | 1.000 | 1% |
| W1 kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8 | ≤ 10: batched-fused 3 | 0.14 | batched-fused 3 | 0.14 | 0.15 | 0.17 | 1.000 | 0% |
| W1 kD = 30 | ε = 1e-6 | complex | 0×2, 16, 15 | ≤ 10: rotation | 0.47 | rotation | 0.47 | 0.47 | 1.25 | 1.000 | 5% |
| W1 kD = 100 | ε = 1e-3 | complex | 0×2, 30, 18, 11, 7 | ≤ 10: batched-fused 5 | 4.77 | batched-fused 5 | 4.77 | 4.93 | 8.43 | 1.000 | 3% |
| W1 kD = 100 | ε = 1e-6 | complex | 0×2, 35, 22, 15×2 | ≤ 10: rotation | 12.42 | rotation | 12.42 | 12.42 | 21.98 | 1.000 | 6% |
| W1 kD = 100 | ε = 1e-3 | real | 0×2, 30, 18, 11, 7 | ≤ 12: batched-fused 4,5 | 3.88 | batched-fused 4,5 | 3.88 | 4.10 | 5.61 | 1.000 | 2% |
| W3 (N = 108849) kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8, 7 | ≤ 10: batched-fused 3,4 | 1.77 | batched-fused 3,4 | 1.77 | 2.21 | 1.81 | 1.000 | 1% |

### Kathleen (AVX-512)

| case | ε | charges | p by level | rule (crossover: levels) | rule s | best | best s | Rotation s | all Batched s | rule / best | spread |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| W4 cube (N = 100000) kD = 10 | ε = 1e-3 | complex | 0×2, 7×3 | ≤ 15: batched-fused 2,3,4 | 0.99 | batched-fused 2,3,4 | 0.99 | 2.49 | 0.99 | 1.000 | 8% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | complex | 0×2, 15×3 | ≤ 15: batched-fused 2,3,4 | 11.52 | batched-fused 2,3,4 | 11.52 | 13.51 | 11.52 | 1.000 | 1% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | real | 0×2, 15×3 | ≤ 15: batched-fused 2,3,4 | 7.65 | batched-fused 2,3,4 | 7.65 | 11.38 | 7.65 | 1.000 | 1% |
| W1 kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8 | ≤ 15: batched-fused 2,3 | 0.12 | batched-fused 2,3 | 0.12 | 0.21 | 0.12 | 1.000 | 0% |
| W1 kD = 30 | ε = 1e-6 | complex | 0×2, 16, 15 | ≤ 15: batched-fused 3 | 0.86 | rotation | 0.78 | 0.78 | 0.92 | 1.106 | 2% |
| W1 kD = 100 | ε = 1e-3 | complex | 0×2, 30, 18, 11, 7 | ≤ 15: batched-fused 4,5 | 5.74 | batched-fused 4,5 | 5.74 | 8.00 | 6.58 | 1.000 | 1% |
| W1 kD = 100 | ε = 1e-6 | complex | 0×2, 35, 22, 15×2 | ≤ 15: batched-fused 4,5 | 20.68 | batched-fused 4,5 | 20.68 | 20.80 | 20.68 | 1.000 | 1% |
| W1 kD = 100 | ε = 1e-3 | real | 0×2, 30, 18, 11, 7 | ≤ 15: batched-fused 4,5 | 4.62 | batched-fused 4,5 | 4.62 | 6.88 | 4.74 | 1.000 | 1% |
| W3 (N = 108849) kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8, 7 | ≤ 15: batched-fused 2,3,4 | 1.13 | batched-fused 2,3,4 | 1.13 | 2.64 | 1.13 | 1.000 | 4% |

### Kathleen (AVX2, `--p2p avx2`)

| case | ε | charges | p by level | rule (crossover: levels) | rule s | best | best s | Rotation s | all Batched s | rule / best | spread |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| W4 cube (N = 100000) kD = 10 | ε = 1e-3 | complex | 0×2, 7×3 | ≤ 11: batched-fused 2,3,4 | 1.48 | batched-fused 2,3,4 | 1.48 | 2.75 | 1.48 | 1.000 | 21% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | complex | 0×2, 15×3 | ≤ 11: rotation | 13.77 | rotation | 13.77 | 13.77 | 19.24 | 1.000 | 1% |
| W4 cube (N = 100000) kD = 10 | ε = 1e-6 | real | 0×2, 15×3 | ≤ 12: rotation | 11.63 | batched-fused 2,3,4 | 11.62 | 11.63 | 11.62 | 1.001 | 4% |
| W1 kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8 | ≤ 11: batched-fused 3 | 0.15 | batched-fused 3 | 0.15 | 0.22 | 0.17 | 1.000 | 12% |
| W1 kD = 30 | ε = 1e-6 | complex | 0×2, 16, 15 | ≤ 11: rotation | 0.79 | rotation | 0.79 | 0.79 | 1.43 | 1.000 | 1% |
| W1 kD = 100 | ε = 1e-3 | complex | 0×2, 30, 18, 11, 7 | ≤ 11: batched-fused 4,5 | 7.30 | batched-fused 4,5 | 7.30 | 8.16 | 10.10 | 1.000 | 1% |
| W1 kD = 100 | ε = 1e-6 | complex | 0×2, 35, 22, 15×2 | ≤ 11: rotation | 20.92 | rotation | 20.92 | 20.92 | 28.83 | 1.000 | 1% |
| W1 kD = 100 | ε = 1e-3 | real | 0×2, 30, 18, 11, 7 | ≤ 12: batched-fused 4,5 | 5.70 | batched-fused 4,5 | 5.70 | 6.96 | 6.87 | 1.000 | 2% |
| W3 (N = 108849) kD = 30 | ε = 1e-3 | complex | 0×2, 12, 8, 7 | ≤ 11: batched-fused 3,4 | 1.64 | batched-fused 3,4 | 1.64 | 2.93 | 1.72 | 1.000 | 13% |

- The rule's choice and Rotation's are the same object below every degree: their outputs
  are the same bit for bit (the checksums of `auto ≤ d: rotation` equal Rotation's in every
  file).
- W1's levels are sparse: a block of 64 targets meets fewer pairs per offset, so each table
  is read for fewer columns. That is why W1 at kD = 30 and 10⁻⁶ prefers Rotation at p = 15
  on Kathleen where the cube prefers Batched, and why W1 at kD = 100 and 10⁻³ prefers
  Rotation at p = 11 on the NEON machines (on locust 5.25 s with that level batched against
  4.77 s without, the means of two runs).
- The script's reverse launch reverses the order of `rotation`, `auto` and `sweep`, not
  the order inside the sweep (∞ first, then descending, in both): drift between the
  sweep's own runs does not cancel. The spreads stayed at 0–8 % but for the AVX2 runs of
  the short cases (12–21 %: the cube at 10⁻³, W1 at kD = 30 and 10⁻³, W3), where the next
  choice can lie within the spread of the rule's (W3: 1.64 s against 1.72 s with Batched on
  every level; W1 at kD = 30: 0.15 s against 0.17 s); Rotation is far slower on all three.

## 3. The cost per pair of the AVX2 and scalar column kernels

`helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20 --isa <isa>`, twice (CH3.2c's NEON and
AVX-512 rows are in its report): Batched's speed relative to Rotation's per pair
(Rotation's ns per pair over Batched's, a block of 32 pairs per offset, k̂ = 1) in the
separate and the fused mode, the mean of the two launches.

TABLE_PAIRS

- AVX2 on Kathleen: fused Batched beats Rotation per pair to p = 12 with two planes and to
  p = 15 with one; the end-to-end sweeps (§2) set 11 and 12.
- The scalar kernel: on the M6 (aarch64, a hardware fma) the fused mode is the faster, on
  Kathleen (x86_64 built without `fma`, so `mul_add` is a call into the C library) the fused
  mode is 2.4–50× slower than Rotation at every degree and the separate one is the faster.
  In each machine's faster mode the largest p at which Batched is at least 1.2× Rotation's
  speed per pair, the headroom the end-to-end sweeps showed the vector kernels need on
  sparse levels, is 6 with two planes and 8 with one, on both. The scalar kernel runs only
  for `P2pChoice::Reference` (the trusted path), so no end-to-end sweep was run with it.

## 4. How it was run

`tools/helmholtz/ch3-2d-runs.sh <out> <p2p,…> <isa…>` (one launch per case and order, then
`helmholtz_m2l` twice per ISA), after building both examples in release on each machine:

- **M6**: `tools/helmholtz/ch3-2d-runs.sh <out> auto scalar`, outside the sandbox, after
  the user confirmed the machine free; every launch gated (the other processes at most 150 %
  CPU before, sampled every 5 s during, void and repeated above 200 %): none void, 9–115 %
  during.
- **locust**: `tools/gh200/sync.sh`, then the script through `tools/gh200/remote.sh` with
  `auto`; a root `find` used one core at the start (load average about 1 of 72), the GPU
  idle.
- **Kathleen**: job 618430 in `singlenode` on node-c11a-089 (`tools/kathleen/submit.sh
  --time=06:00:00 -- run.sbatch '<build>' 'tools/helmholtz/ch3-2d-runs.sh <out> auto,avx2
  avx2 scalar'`), the build and the runs in the batch steps.
- The tables above come from the runs' Markdown with a script (not committed) that groups
  the rows of a case by the choice (the levels that ran Batched), checks that every row of
  a case printed the same errors, and evaluates the final constants on the degrees and the
  budget (§1).
