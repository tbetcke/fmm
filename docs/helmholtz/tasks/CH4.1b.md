# CH4.1b The signature buffers and the level's scratch

Started 2026-11-19 on `tbetcke/h-ch4-1b`. Roadmap: CH4.1b "Per-box aligned signature buffers;
source signatures kept per level call, target accumulators chunked under the node budget",
with its Alsos (the H3 gate's memory baseline, the buffers' layout, review 2026-11-19 C5 and
C7, CH4.1a's column kernel rate and Legendre tables). Proposal: §10 CH4.1; §5.4 ("Scratch and
chunking", "Reflection classes"), §7 (the signature scratch), §2 (constraints 2, 7, 9).

## Goal
The level's signature storage and scratch of the diagonal form: per-box aligned signature
buffers, a level driver that keeps the source signatures for the level call and walks the
targets' accumulators in chunks sized from a byte budget, the bits independent of the chunk
size, and the measured memory it is added to.

## Acceptance (proposal §10 CH4.1, verbatim, and the roadmap's Alsos)
§10: "round trips to the bandwidth of the grid; bits independent of the chunk size; scratch
within the budget on W1 at kD = 1000". CH4.1a took the first and a box's bits independent of
its batch; this task takes "bits independent of the chunk size" and "scratch within the budget
on W1 at kD = 1000". Roadmap Alsos:
- the resident memory of the RCR FMM on W1 at kD = 500 at the calibrated degrees (H3 gate §5);
- the signature buffers' layout fixed in the crate's documentation, not in CONVENTIONS;
- C5: ref's `diagonal::Symmetry` mapped onto the tables' `SignedPermutation`, with a test that
  ref's `offsets()` and the tables' V offsets are one set; C7: `diagonal::Workspace` renamed;
- the column kernel's rate: measured after CH4.2 (nothing here but the line kept);
- the Legendre tables' memory counted in the budget.

## Read first
- `fmm-tables/src/helmholtz/diagonal.rs` (CH4.1a: `Conversions`, `ConversionScratch`,
  `s2f`/`f2r` over contiguous box-major signatures).
- `fmm-ref/src/helmholtz/diagonal/{mod,classes,transfer}.rs` (`s2f`, `f2r`, `m2l`, `Transfer`,
  `Symmetry`, `class_of`, `offsets`, `Workspace`).
- `fmm-tables/src/symmetry.rs` (`SignedPermutation`, CONVENTIONS §3.12), `geometry.rs`
  (`m2l_offsets`).
- `fmm-exec/src/helmholtz/fmm.rs` (`LevelShape` counting, the node budgets:
  `DEFAULT_PLANE_WAVE_NODE_BUDGET` 8 GiB, `DEFAULT_DENSE_NODE_BUDGET` 8 GiB, divided over the
  ranks per node).
- `fmm-validate/examples/helmholtz_accuracy.rs` (`--levels`, the peak resident size).
- `fmm-validate/results/helmholtz-ch4-0c.md` §2 (W1 at kD = 1000: degrees, times on Kathleen).

## Changes
- `nd-fmm-ref`: `helmholtz::diagonal::Workspace` renamed `SignatureWorkspace` (C7; its only
  users are the module, its tests and `nd-fmm-tables`' doc example and tests). No value changes.
- `nd-fmm-tables`, `helmholtz::diagonal` (moved to `diagonal/mod.rs`, with submodules):
  - `Signatures<T>`: a level's signatures, box after box, each box's first sample on a
    64-byte boundary (`SIGNATURE_ALIGN`), the stride the samples rounded up to whole 64-byte
    lines, interleaved complex (CONVENTIONS §3.15's sample order i N_φ + j). Allocated once,
    zeroed; alignment by `pointer::align_offset` on an over-allocated `Vec` (safe Rust); a
    box count up to its capacity (`set_boxes`), so chunks reuse it; `bytes_for`.
  - `Conversions::{s2f_into, f2r_from}`: S → F into a `Signatures` and F → R from one, each
    box's bits those of `s2f`/`f2r` (the blocks read and write through the stride).
  - `ScratchShape` and `scratch_shape(conversions_bytes, samples, sources, targets, threads,
    budget)`: the level's scratch under a byte budget per rank: the Legendre and φ tables, a
    `ConversionScratch` per thread, the sources' signatures for the whole level call, and the
    largest chunk of target accumulators that fits (at least `BLOCK_BOXES` or all targets), or
    `None` when not even that fits (CH4.3's whole-level fallback to RCR). A function of its
    arguments only (constraint 9 holds when the caller passes the agreed counts).
  - `m2l_level(…)`: the level driver, serial, generic over the pair step: S → F of every
    source into `Signatures`, then the targets in chunks of the shape's size in a fixed order:
    each accumulator zeroed, its pairs applied in its row's order by a caller's closure
    (CH4.2 supplies the transfer functions), F → R of the chunk into the locals. A target's
    bits depend on its row alone, never on the chunk size. Threads are nd-fmm-exec's (CH4.2):
    the driver's pieces are per box and per chunk, so the pool can split the S → F boxes and
    a chunk's targets without changing a bit.
  - `signed_permutation(Symmetry) -> SignedPermutation` (C5): the element of O_h that acts on
    offsets as ref's flips-then-swap.
  - `DEFAULT_SIGNATURE_NODE_BUDGET` = 8 GiB per node (as the dense and plane-wave budgets),
    divided over a node's ranks by the caller.
  - Docs: the buffers' layout in the module documentation (the Also from CH1.3d2).
- `nd-fmm-validate`: `helmholtz_accuracy --levels` gains, per level, the targets with a V row
  and the distinct V sources of rank 0's computed targets (from the plan), and the diagonal
  form's scratch on rank 0 under `DEFAULT_SIGNATURE_NODE_BUDGET` over the ranks per node, as
  if every V pair ran the diagonal form (an upper bound: CH4.3's rule takes fewer).
- Docs: `fmm-tables/CLAUDE.md`, `fmm-ref/CLAUDE.md`, `fmm-validate/CLAUDE.md`, the report
  `fmm-validate/results/helmholtz-ch4-1b.md`.
- No convention, no dependency, no Laplace file, no `nd-fmm-plan` interface, no nd-fmm-exec
  change (the operator wires it in CH4.2).

## Tests that prove it
`fmm-tables/tests/helmholtz/diagonal.rs` (and unit tests), f64 unless said:
- `Signatures`: every box's first sample 64-byte aligned for f64 and f32, at p = 0…7 and 12
  (strides with and without padding), `set_boxes` within capacity, the bytes.
- `s2f_into`/`f2r_from`: bit for bit `s2f`/`f2r` on the same boxes (p = 3, 12, 17; f32 too).
- The level driver against the trusted path: with ref's `Transfer` of each offset (real k,
  the high-frequency level's s = 1) as the pair step, the level's locals against the sum over
  each target's pairs of `nd_fmm_ref::helmholtz::diagonal::m2l`, to rounding (256 u of the
  largest value), on a small level (a 4 × 4 × 4 block of boxes, p = 6 and 12).
- Bits independent of the chunk size: chunks of 1, 2, 3, `BLOCK_BOXES`, 7 and every target,
  each local bit for bit the whole level's; also with an empty row and a target in no chunk
  boundary.
- `scratch_shape`: the sizes by hand, the chunk the largest that fits, `None` below the floor,
  all targets in one chunk where the budget allows.
- C5: ref's `offsets()` and the tables' `m2l_offsets()` one set (316); for every offset
  `signed_permutation(s).apply(o)` is ref's representative; each of ref's 34 classes lies in
  one of the tables' 16 classes (the subgroup's refinement).

## Checks
`/verify` (fmt, Clippy default and workspace all-targets, docs with `-D warnings`, the default
and workspace tests at opt-level 1); `cargo clippy -p nd-fmm-tables --all-targets`,
`cargo test -p nd-fmm-tables`, `cargo test -p nd-fmm-ref`. The MPI list: no MPI crate's
library code changes (an example's output only); run it if `/verify` says so.

## Machine runs
Kathleen, `singlenode` jobs, one node (40 cores), release, BLAS threads 1:
- the memory baseline: `helmholtz_accuracy --workload w1 --kd 500` at 10⁻⁶ and 10⁻⁸, `--m2l
  rotation` and the default, one process each at 1 rank × 40 threads, `--levels`: the peak
  resident size per run (the RCR FMM's baseline, H3 gate §5);
- the scratch census: W1 at kD = 1000 at 10⁻⁶ and 10⁻⁸, the default, 1 × 40 and 40 × 1,
  `--levels`: per level the sources, targets and the diagonal form's scratch per rank under
  8 GiB per node. Decides "scratch within the budget on W1 at kD = 1000" and which levels a
  node budget pushes to RCR at many ranks per node.
No M6 timing (none needed: memory and counts only).

## Constraints and risks
- Constraint 1: Laplace untouched (Helmholtz modules of ref and tables, an example).
- Constraint 2: every target's local the same bits at every chunk size (tested); the driver's
  order fixed.
- Constraint 7: the scratch counts against a node budget per rank, with the conversion tables.
- Constraint 9: the shape is a function of the counts, the degree and the budget; the caller
  must pass rank-independent inputs or agree the result (CH4.2's wiring).
- Risk: at many ranks per node every rank of a coarse level needs nearly every source's
  signature (ghosts), so the per-rank share of a node budget may not hold them; the census
  states it, CH4.3's fallback and CH8.0 (per-node tables) are where it is answered.

## Pending runs
None. Kathleen job 651903 (node-c11a-079, `singlenode`, 9 min 38 s; the memory baseline and
the census) and job 653493 (the 40 × 1 census again after the self-review's fix: every V row
a rank holds counted, the replicated top's included). Raw outputs in
`~/projects/fmm-archive/kathleen/runs/ch4-1b/`.

## Open questions
For sign-off on the PR (the report's §5): 1, the acceptance as read (within the budget at one
rank per node; at 40 ranks per node levels 2 and 3 do not fit); 2, the buffers' layout; 3, the
budget's default.

## Results
- Built as planned, with these deviations: `scratch_shape` takes p (it computes the tables'
  and samples' sizes itself) where the plan listed `conversions_bytes` and `samples`; the test
  level is a 6 × 6 × 2 block of boxes (six targets, one without pairs) where the plan said
  4 × 4 × 4; the chunks tested are 1, 2, 3, 4, 5, 7 (a buffer larger than the targets) and all.
- The self-review (a subagent, on the plan, the proposal's §2 and §10 row and the branch)
  found nothing blocking; fixed in the branch: the census counted only the targets a rank
  computes, not the replicated top's rows every rank holds and runs (the 40 × 1 census
  rerun); no f64 stride is ever padded (N_φ = 4(p + 1)), so F → R and the level call are now
  tested through a padded f32 stride; a clone of `Signatures` kept the original's offset (now
  aligned on its own allocation); the source signatures' bytes reported for ranks over the
  budget too; `rows[0] = 0` asserted.
