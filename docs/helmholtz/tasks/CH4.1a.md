# CH4.1a The product conversions, measured

Started 2026-11-19 on `tbetcke/h-ch4-1a`. Roadmap: CH4.1 "S → F and F → R by the transform as
a product", split on 2026-11-19 into CH4.1a (this: the conversions and the measurement that
review 2026-11-19, A5, puts first, with the stop to ask) and CH4.1b (the signature buffers, the
level's scratch under the budget, the memory baseline, C5 and C7). Proposal: §10 CH4.1; §5.4
(S → F, F → R), §6.7 (the φ step as a product, the grid's sizes), §7 (the cost model), §3.6.

## Goal
Write S → F and F → R as the level-batched product of proposal §5.4 (the Legendre step halved by
the rings' parity, the φ step by the grid's symmetries), deterministic per box, and measure them
per box against the RCR pair, so the user can decide whether H4 goes on with the product or the
FFT track moves ahead.

## Acceptance (proposal §10 CH4.1, verbatim, and the roadmap's A5 part)
§10: "round trips to the bandwidth of the grid; bits independent of the chunk size; scratch
within the budget on W1 at kD = 1000". This task takes the first and the per-box half of the
second (a box's bits independent of the batch it is converted in); the chunked accumulators and
the budget are CH4.1b's.

Roadmap (review 2026-11-19, A5; the user on #230: "CHA.11 + measure first"): "the product
conversion per box (S → F and F → R, with the ×4 the φ grid's symmetries give, N_φ a multiple of
4) against the RCR pair at the high-frequency levels' degrees, and a high-frequency level's time
with the product conversions against RCR's on W1; then a stop to ask before CH4.2 whether H4 goes
on as listed or the FFT track (CH0.9, CHR.3, CH4.6) moves ahead. With it the H4 gain stated for
the product and for an FFT apart."

## Read first
- `docs/reviews/2026-11-19-H1.md` A5 (checked on 2026-11-19 against 3662658: the counts are
  proposal §6.7's 2P³ + 16P³ complex multiply-adds per conversion and §7's 2P³ per RCR pair).
- `fmm-ref/src/helmholtz/diagonal/{mod,grid}.rs`: `s2f`, `f2r` (the reference, per ring a
  Legendre sum per order, then the DFT as a product), `Grid` (rings, weights, azimuths, the
  θ-factors per ring).
- `fmm-validate/examples/helmholtz_rcr_pair.rs` (CH4.0i): the RCR pair on one core, the method
  to repeat beside the conversions.
- `fmm-validate/results/helmholtz-ch4-0e.md` §3 (W1 at kD = 500: level 2 64 boxes and 2,504 V
  pairs at p = 123–135, level 3 512 boxes and 12,216 pairs at p = 66–76; the levels' M2L
  seconds) and `helmholtz-ch4-0i.md` (the RCR level by pairs).

## Changes
- `nd-fmm-ref`: `diagonal::Grid::theta` public (ring i's θ-factors, today `pub(super)` `ring`),
  so that tables are built from the reference's values, not re-derived. Nothing else.
- `nd-fmm-tables`: a module `helmholtz::diagonal` (proposal §5.4; CONVENTIONS §3.15
  "Signatures"):
  - `Conversions<T>`, built in f64 from `nd_fmm_ref::helmholtz::diagonal::Grid` and the level's
    power scale s and rounded once to T: per order μ the θ-factors times sⁿ of rings 0..=p
    (rows), split by the parity of n + μ, and the same times the ring's weight for F → R; per
    residue r = μ mod 4 the cos μφⱼ and sin μφⱼ at the base azimuths 0 ≤ j ≤ N_φ/8; the sign
    map of the other azimuths.
  - `s2f(&self, multipoles, signatures, &mut ConversionScratch)` and `f2r(…)` over a batch of
    boxes (box-major inputs and outputs, a signature §3.15's i N_φ + j, interleaved complex), in
    blocks of a fixed number of boxes: gather (with (−i)ⁿ exactly), Legendre by parity (rings i
    and 2p − i from one sum: E ± O), the φ step per residue as one product, the azimuths from
    the base ones by signs; F → R the transpose, adding into the locals (iⁿ exactly).
  - One deterministic column kernel (`C = A · X`, every element the chain k ascending from 0,
    multiply and add rounded apart), safe Rust; every element depends on its own row and
    column only, so a box's bits do not depend on the batch, its position or the block.
  - `ConversionScratch<T>`, sized at construction; no allocation in `s2f`/`f2r`.
  - Counts: `Conversions::{legendre_flops, phi_flops}` per box, for the report.
- `nd-fmm-validate`: example `helmholtz_conversions` (MPI-free, not registered; it times): per
  `p@k̂` case on one core, S → F and F → R per box at a batch (the Legendre and φ steps apart),
  the RCR pair as `helmholtz_rcr_pair` times it (by pairs in class order, blocks per call and
  stored), the diagonal product per pair (a complex multiply-add of a signature with a permuted
  transfer function, CH4.2's step, here a stand-in), and W1's level 2 and 3 modelled from these.
- Docs: `fmm-tables/CLAUDE.md` (the module, an exception like `helmholtz`'s), `fmm-ref/CLAUDE.md`
  one line, `fmm-validate/CLAUDE.md` (the example), the report.
- No convention (§3.15's grid and sample order as signed off at CH1.3d2; the buffers' layout is
  CH4.1b's). No dependency. No Laplace file. No `nd-fmm-plan` interface.

## Tests that prove it
`fmm-tables/tests/helmholtz/diagonal.rs` (or the crate's test layout), f64 unless said:
- against `nd_fmm_ref::helmholtz::diagonal::s2f` and `f2r` on random multipoles and signatures,
  p ∈ {0, 1, 2, 3, 7, 12, 24} (N_φ = 4p + 4: every residue class of N_φ/8's rounding), s = 1 and
  s < 1: within 64 u (p + 1) times the largest |value| (the debug run); p = 72 and 131 in an
  ignored release test;
- round trip: f2r(s2f(M̂)) = s²ⁿ M̂ to 10⁻¹³ relative (the grid's bandwidth p), p ≤ 24; and a
  signature of bandwidth above p (a sample set from the transfer function's product) shows
  F → R keeps degrees ≤ p only, against the reference;
- per-box bits: a batch of 1, 5, the block size and 2 blocks + 3 boxes, a box at every position,
  equal bit for bit to the box converted alone, S → F and F → R;
- f32 against the f64 reference within 64 u₃₂ (p + 1);
- the column kernel against a plain triple loop bit for bit (every row and column count up to
  the blocks plus tails).

## Checks
`/verify` (fmt, Clippy default and workspace all-targets, docs with `-D warnings`, the default
and workspace tests at opt-level 1; `cargo clippy -p nd-fmm-tables --all-targets`, `cargo test -p
nd-fmm-tables`, the ignored release tests of the new module). The MPI list: no MPI crate's code
changes (nd-fmm-validate gains an example only); run it if `/verify` says so.

## Machine runs
One core each, release, BLAS threads 1, the machine's load named (`uptime`):
- locust (Grace): `helmholtz_conversions --cases 66@31.25,72@31.25,76@31.25,123@62.5,131@62.5,
  135@62.5` and kD = 1000's high-frequency degrees (levels 2–4 at 10⁻⁶ and 10⁻⁸, from
  `DegreeRule`), plus `helmholtz_rcr_pair` on the same cases for the record;
- Kathleen (Cascade Lake, AVX-512), the same, as a `singlenode` job.
They decide the sign-off question (product or FFT first). No M6 timing (the user's rule: ask
first; locust and Kathleen are the machines of record of CH4.0e).

## Constraints and risks
- Constraint 1: Laplace untouched (new Helmholtz module, a public accessor in `nd-fmm-ref`'s
  Helmholtz module, an example).
- Constraint 2: a box's bits do not depend on the batch or the thread that converts it (tested
  bit for bit); the azimuths combine the residues in a fixed order.
- The column kernel is safe Rust (no `unsafe` in `nd-fmm-tables`): its rate is what LLVM's
  autovectorisation gives. The report states the achieved flop rate against the RCR pair's on
  the same core, so the comparison is between this code and ours, not against a peak.
- Risk: the level model uses one core and the per-pair diagonal step as a stand-in; the real
  level call is CH4.1b/CH4.2's. The report says so.

## Pending runs
(filled in step 6)

## Open questions
(filled when they arise)
