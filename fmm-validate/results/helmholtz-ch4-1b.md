# Helmholtz CH4.1b: the signature buffers and the level's scratch

Task: `docs/helmholtz/tasks/CH4.1b.md`. Roadmap: H4, CH4.1b (priority list item 10), the second
half of CH4.1. Proposal §10 CH4.1: "round trips to the bandwidth of the grid; bits independent of
the chunk size; scratch within the budget on W1 at kD = 1000". CH4.1a took the first item and a box's
bits independent of its batch. This task takes the other two, plus the roadmap's Alsos: the RCR
FMM's resident memory (H3 gate §5), the buffers' layout, and review 2026-11-19's C5 and C7.

## 0. Summary

- **Built** (§1), in `nd_fmm_tables::helmholtz::diagonal::level`:
  - `Signatures`: a level's signatures box after box, each box on its own 64-byte line;
  - the level driver `m2l_level`: the source signatures kept for the level call, the targets'
    accumulators in chunks, F → R per chunk;
  - `scratch_shape`: the scratch under a byte budget per rank (the default
    `DEFAULT_SIGNATURE_NODE_BUDGET`, 8 GiB per node over the node's ranks).

  Every local is the same bits at every chunk size (tested, f64 and a padded f32 stride), and the
  driver equals the sum of `nd_fmm_ref`'s `diagonal::m2l` over each target's pairs to rounding.
- **C5**: ref's 316 offsets are the tables' set, and each of ref's 34 classes lies inside one of
  the tables' 16 (`signed_permutation`). **C7**: ref's `diagonal::Workspace` is renamed
  `SignatureWorkspace`.
- **The baseline** (§2): W1 at kD = 500 (N = 1,989,437) on one Kathleen node, 1 rank × 40
  threads. Rank 0's peak resident size:
  - RCR on every level: **4.4 GB at 10⁻⁶** and 5.1 GB at 10⁻⁸;
  - the library's defaults (RCR on levels 2–3, the exponential M2L below): **5.8 GB** and
    7.9 GB.

  If every V pair of a level ran the diagonal form, that level's scratch would be 0.38–0.45 GB on
  levels 2 and 3, the levels that run RCR under the defaults. It would be 0.5–1.3 GB on levels
  4–5, but those run the exponential M2L. The scratch is a transient of one level call, so the
  peak grows by at most the largest level's scratch: about 8 % over the defaults at 10⁻⁶.
- **The census at kD = 1000** (§3): W1, N = 7,957,748, the defaults, at 10⁻⁶ and 10⁻⁸.
  - **At 1 rank × 40 threads every level fits 8 GiB, in one chunk.** Levels 2–4 (p = 245–251,
    131–135 and 72–76, RCR under the defaults) need 1.3–1.7 GiB, the levels below at most
    5.0 GiB. Only level 7 at 10⁻⁸ chunks (25 chunks of 2,800 targets), and it runs the
    exponential M2L anyway.
  - **At 40 ranks × 1 thread** each rank has 205 MiB. Level 2 fits on no rank, because every
    rank needs all 56 source signatures (413–433 MiB at p = 245–251). Level 3 fails on 15 of the
    40 ranks. Levels 4–6 fit.
  - This is the risk the plan named. The proposal's answer for a level that does not fit is RCR
    for that whole level, agreed over the ranks, which is CH4.3's rule (question 1).

## 1. What was built

`nd_fmm_tables::helmholtz::diagonal` (the module of CH4.1a, now `diagonal/mod.rs`) gains the
module `level` and the function `signed_permutation`:

- **`Signatures<T>`** is the layout of the buffers (the Also from CH1.3d2). It is fixed in the
  module's documentation, not in CONVENTIONS (§3.15, "Not fixed here").
  - Each signature is §3.15's (2p + 1) N_φ samples in the order i N_φ + j, stored as
    interleaved complex values (`Complex<T>`).
  - Box b starts at b · stride. The stride is the samples rounded up to whole 64-byte lines
    (`SIGNATURE_ALIGN`), so each box starts on a line and no line holds two boxes' samples.
  - Since N_φ = 4(p + 1), an f64 signature is always whole lines and is never padded. f32 is
    padded at even p.
  - Alignment comes from the allocation's address in safe Rust (`is_aligned`), and a clone
    aligns on its own allocation.
  - The store is allocated once at its capacity and holds any number of boxes up to it
    (`set_boxes`).
- **`Conversions::{s2f_into, f2r_from}`** convert into and out of a store; each box gets the
  bits of `s2f` and `f2r`, which now run through a stride internally.
- **`m2l_level`** is the serial level driver.
  1. S → F of every source box once, a block of gathered multipoles at a time.
  2. The targets in chunks of `LevelBuffers::chunk`. For each target: the accumulator is
     zeroed, then its pairs are applied in its row's order by the caller's closure (CH4.2
     supplies the transfer functions).
  3. F → R of the chunk by blocks. The locals are gathered, added into and scattered back as
     copies.

  A target's bits depend on its own row alone. A threaded caller (nd-fmm-exec, CH4.2) may split
  the source boxes and a chunk's targets over threads without changing a bit.
- **`scratch_shape`** sizes one rank's scratch:
  - the tables (`Conversions::bytes_for`; the Legendre tables hold (p + 1)²(p + 2) values, the
    Also from CH4.1a's review);
  - per thread, a `ConversionScratch` and a gather buffer;
  - the source signatures, kept whole;
  - the largest chunk of accumulators that fits.

  It returns `None` when not even `BLOCK_BOXES` targets fit. It does not count CH4.2's transfer
  functions (proposal §7: 0.4 GiB at kD = 1000 for the 34 classes), which are tables.
- **`signed_permutation(Symmetry)`** gives the element of O_h that acts on offsets like ref's
  flips-then-swap (C5).

Tests (`fmm-tables/tests/helmholtz/diagonal_level.rs`, 12 tests, one of them ignored):
- alignment for f64 and f32, padded and not, and a clone;
- the store conversions bit for bit against the plain ones (f64 at p = 3, 12, 17; f32 at p = 2,
  padded, and p = 3);
- the level call against the sum of `nd_fmm_ref::helmholtz::diagonal::m2l` over each target's
  pairs, within 256 u, at p = 6 and 12 (and p = 17, ignored, in release). The test level is a
  6 × 6 × 2 block of unit boxes with six targets, one of them without pairs.
- chunks of 1, 2, 3, 4, 5, 7 (a buffer larger than the level) and all, each bit for bit the
  whole level, in f64 and through a padded f32 stride;
- the shapes by hand: the largest chunk, one byte under, the floor, and `None` below it;
- the sizes without building;
- C5's sets and classes.

A deliberate break (the accumulators not zeroed between chunks) fails the chunk test. The
harness `helmholtz_accuracy` reports the census below (`Row::signature_scratch`, the table
"Diagonal-form scratch" of `--levels`). For each level, rank and run, it records:
- the boxes with a V row the rank holds;
- the distinct sources of those rows;
- `scratch_shape` at the level's degree and the run's threads;
- the largest of each over the ranks, and the ranks over the budget.

It assumes every V pair of the level runs the diagonal form, an upper bound: CH4.3's rule takes
fewer.

## 2. The memory baseline: W1 at kD = 500

Kathleen job 651903, node-c11a-079 (`singlenode`, 2 × 20 Cascade Lake cores), one process per
run, 1 rank × 40 threads, `helmholtz_accuracy --workload w1 --kd 500 --levels`, the near rules
of the defaults (`--near-v cost --near-wx cost`). The peak is the process's VmHWM. It includes
the problem's points and charges and the 1,000-sample oracle, both small next to the FMM.

| ε | M2L | degrees, levels 2–6 | peak resident | build s | evaluate s | φ error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 10⁻⁶ | Rotation (RCR on every level) | 131, 72, 41, 25, 16 | 4,391 MB | 2.56 | 6.93 | 1.2·10⁻⁷ |
| 10⁻⁶ | defaults (RCR on 2–3, waves on 4–6) | 131, 72, 41, 25, 16 | 5,822 MB | 2.56 | 6.58 | 1.4·10⁻⁷ |
| 10⁻⁸ | Rotation | 135, 76, 44, 27, 24 | 5,086 MB | 2.54 | 8.83 | 2.7·10⁻⁹ |
| 10⁻⁸ | defaults | 135, 76, 44, 32, 32 | 7,899 MB | 2.58 | 10.17 | 1.6·10⁻⁹ |

The defaults' extra memory is the exponential M2L's waves buffer (2,825 MB at 10⁻⁶ and 4,919 MB
at 10⁻⁸ as built).

The diagonal form's scratch per level, as if every V pair ran it, from the same runs (1 × 40;
every level in one chunk):

| ε | level | p | targets = sources | source signatures MiB | scratch MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10⁻⁶ | 2 | 131 | 56 | 119 | 382 |
| 10⁻⁶ | 3 | 72 | 272 | 176 | 404 |
| 10⁻⁶ | 4 | 41 | 1,160 | 247 | 517 |
| 10⁻⁶ | 5 | 25 | 4,760 | 385 | 783 |
| 10⁻⁸ | 2 | 135 | 56 | 126 | 405 |
| 10⁻⁸ | 3 | 76 | 272 | 196 | 448 |
| 10⁻⁸ | 4 | 44 | 1,160 | 284 | 592 |
| 10⁻⁸ | 5 | 32 | 4,760 | 623 | 1,263 |

"Scratch" counts:
- the sources' signatures;
- the accumulators of one chunk (here every target);
- the conversion tables;
- 40 threads' conversion scratch and gather buffers. At p = 131 that is 40 × 2.5 MB, a quarter
  of level 2's scratch.

H4 adds the diagonal form on levels that run RCR today (CH4.3 decides which classes). Under the
defaults those are levels 2–3. One level call holds its scratch at a time, so the peak grows by
at most about 0.4–0.45 GB: **about 8 % of the defaults' peak and 9–10 % of Rotation's**. This is
far inside the node, and below the waves buffer the defaults already hold.

## 3. The census: W1 at kD = 1000

The same job, the library's defaults, at 10⁻⁶ and 10⁻⁸, at 1 rank × 40 threads and at 40 ranks ×
1 thread (job 653493 reran the 40 × 1 launches with the self-review's fix: every V row a rank
holds is counted, including the replicated top's, which every holding rank runs; job 651903
had counted only the boxes each rank computes). The budget is 8 GiB per node over the ranks per
node: 8,192 MiB at one rank and 205 MiB at 40.

**1 rank × 40 threads** (peak resident 21.7 GB at 10⁻⁶ and 29.9 GB at 10⁻⁸; build 13.1 and
13.3 s, evaluation 40.7 and 58.1 s; φ errors 1.1·10⁻⁷ and 1.7·10⁻⁹):

| ε | level | p | M2L under the defaults | targets | sources | source MiB | scratch MiB | chunk | chunks |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10⁻⁶ | 2 | 245 | RCR | 56 | 56 | 413 | 1,314 | 56 | 1 |
| 10⁻⁶ | 3 | 131 | RCR | 272 | 272 | 576 | 1,298 | 272 | 1 |
| 10⁻⁶ | 4 | 72 | RCR | 1,160 | 1,160 | 749 | 1,551 | 1,160 | 1 |
| 10⁻⁶ | 5 | 41 | waves | 4,760 | 4,760 | 1,013 | 2,048 | 4,760 | 1 |
| 10⁻⁶ | 6 | 25 | waves | 19,138 | 19,138 | 1,549 | 3,110 | 19,138 | 1 |
| 10⁻⁶ | 7 | 16 | waves | 72,493 | 68,302 | 2,339 | 4,828 | 72,493 | 1 |
| 10⁻⁸ | 2 | 251 | RCR | 56 | 56 | 433 | 1,380 | 56 | 1 |
| 10⁻⁸ | 3 | 135 | RCR | 272 | 272 | 612 | 1,377 | 272 | 1 |
| 10⁻⁸ | 4 | 76 | RCR | 1,160 | 1,160 | 834 | 1,725 | 1,160 | 1 |
| 10⁻⁸ | 5 | 44 | waves | 4,760 | 4,760 | 1,164 | 2,352 | 4,760 | 1 |
| 10⁻⁸ | 6 | 32 | waves | 19,138 | 19,138 | 2,506 | 5,027 | 19,138 | 1 |
| 10⁻⁸ | 7 | 32 | waves | 68,257 | 59,647 | 7,809 | 8,192 | 2,800 | 25 |

Every level fits. The levels H4 targets (2–4, RCR under the defaults; proposal §7 counts levels
2–6 as high frequency at kD = 1000) need 1.3–1.7 GiB in one chunk. Proposal §7 estimated a
5.0 GiB peak on level 6, sources and targets unchunked, at the formula's degrees; at the
calibrated degrees level 6 needs 3.0–4.9 GiB, and it runs the exponential M2L.

**40 ranks × 1 thread** (job 653493; peak resident of rank 0 3.7 GB at 10⁻⁶ and 4.1 GB at
10⁻⁸; the φ errors and checksums as at 1 × 40): the largest of each column over the ranks, and
the ranks over their 205 MiB.

| ε | level | p | M2L under the defaults | targets | sources | source MiB | scratch MiB | chunk | chunks | ranks over |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10⁻⁶ | 2 | 245 | RCR | 4 | 56 | 413 | – | – | – | **40** |
| 10⁻⁶ | 3 | 131 | RCR | 10 | 109 | 231 | 189 | 6 | 1 | **15** |
| 10⁻⁶ | 4 | 72 | RCR | 32 | 181 | 117 | 141 | 26 | 1 | 0 |
| 10⁻⁶ | 5 | 41 | waves | 128 | 416 | 89 | 115 | 108 | 1 | 0 |
| 10⁻⁶ | 6 | 25 | waves | 518 | 1,113 | 90 | 129 | 434 | 1 | 0 |
| 10⁻⁶ | 7 | 16 | waves | 1,926 | 2,694 | 92 | 154 | 1,663 | 1 | 0 |
| 10⁻⁸ | 2 | 251 | RCR | 4 | 56 | 433 | – | – | – | **40** |
| 10⁻⁸ | 3 | 135 | RCR | 10 | 109 | 245 | 201 | 6 | 1 | **15** |
| 10⁻⁸ | 4 | 76 | RCR | 32 | 181 | 130 | 157 | 26 | 1 | 0 |
| 10⁻⁸ | 5 | 44 | waves | 128 | 416 | 102 | 132 | 108 | 1 | 0 |
| 10⁻⁸ | 6 | 32 | waves | 518 | 1,113 | 146 | 205 | 434 | 2 | 0 |
| 10⁻⁸ | 7 | 32 | waves | 1,796 | 2,295 | 300 | – | – | – | **40** |

The scratch, chunk and chunks columns are over the ranks that fit, so a level where no rank fits
has none. The two jobs' checksums are the same, and so are the 1 × 40 runs' (the output does not
depend on the split). Level 7 at 10⁻⁸ (p = 32, raised for the exponential M2L) also exceeds
205 MiB, but it runs the exponential M2L and is no candidate for the diagonal form.

**What it shows.** On a coarse level, every rank's rows need nearly every source of the level:
- the replicated top's boxes have V rows on every rank that holds them;
- the 56 boxes of level 2 are each other's V sources.

So the source signatures do not divide over the ranks. Each rank's share of the node budget
falls with the ranks per node, but each rank's need on level 2 does not. A per-rank budget then
sends levels 2–3 to RCR at many ranks per node, which are the levels the diagonal form helps
most (CH4.1a: level 2 modelled at 0.06–0.10 of RCR's time). Two remedies, neither of them this
task's:
- a few ranks per node with threads, as CH4.0e and CH4.0i found best for the coarse levels
  anyway;
- the source signatures formed once per node and shared, as CH8.0 proposes for the tables.

## 4. The acceptance

| Item | Evidence | Result |
| --- | --- | --- |
| Bits independent of the chunk size | `the_level_call_against_refs_m2l_at_every_chunk_size` (f64, p = 6, 12; p = 17 ignored), `the_level_call_through_a_padded_stride` (f32) | met: chunks 1–5, 7 and all bit for bit |
| Scratch within the budget on W1 at kD = 1000 | §3, Kathleen 651903 and 653493 | met at one rank per node (every level, one chunk but level 7 at 10⁻⁸); at 40 ranks per node not on levels 2–3 (question 1) |
| The RCR FMM's resident memory at kD = 500 (H3 gate §5) | §2 | 4.4 GB at 10⁻⁶ (RCR everywhere), 5.8 GB under the defaults |
| Buffers' layout in the crate's documentation | `diagonal::level`'s module docs, `fmm-tables/CLAUDE.md` | done (question 2) |
| C5: ref's offsets and classes on the tables' | `the_offsets_are_one_set_and_the_classes_nest` | met |
| C7: `diagonal::Workspace` renamed | `SignatureWorkspace` | done |
| Legendre tables counted in the budget | `Conversions::bytes_for` in `scratch_shape`, `the_sizes_without_building` | done |
| The column kernel's rate on x86_64 | after CH4.2 (the roadmap's Also) | kept on CH4.1b's line for CH4.2 |

## 5. Questions for sign-off

Signed off 2026-11-19 by the user on #236, all three as recommended: "All questions as
recommended."

1. **The acceptance as read.** "Scratch within the budget on W1 at kD = 1000" is met at one rank
   per node: every level fits 8 GiB, levels 2–4 in 1.3–1.7 GiB. At 40 ranks per node, level 2
   fits on no rank and level 3 on 25 of 40, because coarse levels need nearly every source
   signature on every rank.
   - *Recommended:* accept it as met at one rank per node.
   - Add an Also to CH4.3: the whole-level fallback to RCR is decided from the largest need
     over the ranks (one all-reduce), so every rank takes the same M2L on a level
     (constraints 2 and 9).
   - Add an Also to CH8.0: a level's source signatures shared per node.
   - *Alternatives:* chunk the source signatures too (every chunk of targets would form its
     sources again: more S → F work, the same bits), or a separate budget for the coarse
     levels.
2. **The buffers' layout.** Each box's signature starts on its own 64-byte line, the stride
   padded to whole lines, interleaved complex in §3.15's sample order, box after box. It is
   fixed in `nd-fmm-tables`' documentation, not in CONVENTIONS.
   - *Recommended:* as built.
   - *Alternative:* split planes (real, then imaginary), which a SIMD kernel of the pair step
     might prefer. This is for CH4.2 to measure; changing it changes no bit of the output.
3. **The budget's default.** `DEFAULT_SIGNATURE_NODE_BUDGET` is 8 GiB per node over the node's
   ranks, as the dense and plane-wave budgets. Each is a transient of different level calls or
   tables, so their peaks do not add up.
   - *Recommended:* as built; nd-fmm-exec takes it in CH4.2.
   - *Alternative:* one budget shared by every M2L's scratch.

## 6. Reproduce

```sh
cargo build --release -p nd-fmm-validate --example helmholtz_accuracy
tools/kathleen/submit.sh --time=03:00:00 -- run.sbatch \
    'cargo build --release -p nd-fmm-validate --example helmholtz_accuracy' \
    'tools/helmholtz/ch4-1b-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch4-1b memory census'
```

The raw outputs are in `~/projects/fmm-archive/kathleen/runs/ch4-1b/`.
