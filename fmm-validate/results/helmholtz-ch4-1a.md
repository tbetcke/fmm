# Helmholtz CH4.1a: the product conversions, measured

Task: `docs/helmholtz/tasks/CH4.1a.md`. Roadmap: H4, CH4.1a (priority list item 10), the first
half of CH4.1, split on 2026-11-19. Proposal §10 CH4.1 ("round trips to the bandwidth of the
grid; bits independent of the chunk size; …"), with the measurement that review 2026-11-19, A5,
put first (the user on #230: "CHA.11 + measure first"): the product conversion per box against
the RCR pair at the high-frequency degrees, a high-frequency level's time with the product
conversions against RCR's on W1, H4's gain for the product and for an FFT apart, and a stop to
ask before CH4.2.

## 0. Summary

- **What was built** (§1): `nd_fmm_tables::helmholtz::diagonal::Conversions`, S → F and F → R
  of a batch of boxes as products: the Legendre step halved by the rings' parity, the φ step
  quartered by the azimuths' symmetries (N_φ a multiple of 4: every azimuth is ±φⱼ + qπ/2 of a
  base azimuth 0 ≤ j ≤ N_φ/8, so per residue μ mod 4 four sums at the base azimuths give all of
  them by signs). Per box and conversion that is 2P³ real multiply-adds for the Legendre step
  and 8P³ for the φ step, 2.5P³ complex ones, against the 18P³ of proposal §6.7's count, which
  A5 used. Against `nd_fmm_ref`'s reference to rounding (≤ 3.6·10⁻¹⁵ of the largest value at
  p = 245); the round trip to the grid's bandwidth; a box's bits the same in every batch.
- **Per box against the RCR pair** (§2), one core: S → F plus F → R cost 1.5–2.4 RCR pairs
  (two planes) on locust and 1.6–2.8 on Kathleen at p = 66–135, about one at p = 245. A5
  expected about 18.
- **A high-frequency level** (§2), modelled on one core from these times (boxes × the two
  conversions + pairs × a stand-in of the diagonal pair, every pair diagonal as S6 allows for
  real k): level 2 of W1 at kD = 500 (64 boxes, 2,504 pairs) at **0.063–0.072 of RCR's time**
  on locust and 0.082–0.097 on Kathleen; level 3 (512 boxes, 12,216 pairs) at 0.156–0.162 and
  0.195–0.203. A5 expected about 0.43. At kD = 1000, levels 2–4 at 0.041, 0.089, 0.224
  (locust) and 0.039, 0.109, 0.281 (Kathleen).
- **An FFT** (§3): its gain is bounded by the φ step, 80 % of the conversions' multiply-adds.
  With the φ step free, those levels drop to 0.032–0.035 and 0.079–0.084 of RCR (locust),
  0.049–0.058 and 0.106–0.110 (Kathleen): the FFT's whole gain is at most about half of what
  remains after the product, i.e. at most 0.03–0.09 of the RCR level.
- **H4's gain on W1 at kD = 500** (§3), scaling CH4.0e's measured level times at the best split by
  these one-core ratios: build plus one evaluation 0.72–0.74× of today at 10⁻³ and 10⁻⁶ and
  0.81× at 10⁻⁸ on Kathleen with the product (0.73, 0.73, 0.88× on locust); with the φ step free,
  0.71–0.72× and 0.80× (0.72, 0.72, 0.87×). The FFT adds at most 1–2 % of the run. At kD = 1000
  (outside D1) 0.62× with the product and 0.60× with φ free on Kathleen at 10⁻⁶ (locust 0.65×,
  0.63×).
- **Recommendation** (§5, question 1): H4 goes on as listed with the product (CH4.1b, CH4.2, …);
  the FFT track stays item 11.

## 1. The transforms

`nd_fmm_tables::helmholtz::diagonal` (module documentation for the derivation):

- **Tables**, built in f64 from `nd_fmm_ref::helmholtz::diagonal::Grid` (its θ-factors, weights
  and azimuth tables, made public: `Grid::theta`, `Grid::azimuth_tables`) and the level's power
  scale s, rounded once to T: per order μ the θ-factors times sⁿ on the rings 0 ≤ i ≤ p split by
  the parity of n + μ (S → F), the same times the ring's weight transposed (F → R), and per
  residue r the cos μφⱼ and sin μφⱼ at the N_φ/8 + 1 base azimuths.
- **S → F** of a block of 4 boxes (`BLOCK_BOXES`), rings in blocks of 16 (`RING_BLOCK`): the four
  real planes of the coefficients ±μ gathered with (−i)ⁿ exactly; per order the even and odd
  sums E and O on the block's rings (rings i and 2p − i are E ± O); per residue one product
  [cos; sin] × (orders × ring slots) at the base azimuths; every azimuth from them by signs and
  one add, the residues added in order 0–3.
- **F → R**, the transpose: the samples folded onto the base azimuths with the same signs, per
  residue the transposed product, rings i and 2p − i folded into E and O, the transposed Legendre
  product (weights in the table) continuing each degree's chain across the ring blocks; iⁿ
  exactly on the way into the locals (+=).
- **Determinism**: every product is one column kernel, each element the chain over the inner
  index ascending from its own row and column, product and sum rounded apart; the folds are per
  element in a fixed order. So a box's bits do not depend on the batch, its position, the block
  sizes or the thread.
- **Tests** (`fmm-tables/tests/helmholtz/diagonal.rs`, the unit tests of the module):
  - S → F and F → R against the reference at p = 0–5, 7, 12, 24, s = 1 and s ≈ 0.61, within
    64 u (p + 1) of the largest value (p = 41, 72, 131 in an ignored release test);
  - the round trip f2r(s2f(M̂)) = s²ⁿ M̂ to 10⁻¹³ at p ≤ 24;
  - a box in batches of 1, 2, 5, 4, 5 and 11 at three positions bit for bit the box alone, both
    ways;
  - f32 within 64 u₃₂ (p + 1) of the f64 reference;
  - the column kernel bit for bit a triple loop over tiles and tails;
  - the azimuth map exactly in integers.
  Changing one sign of the azimuth combination, in S → F or in F → R, fails three of them.
- The example `helmholtz_conversions` checks a box against the reference before timing (the
  column "rel. err."): at most 2.1·10⁻¹⁵ (S → F) and 3.6·10⁻¹⁵ (F → R) at p = 245.

## 2. The measurements

`helmholtz_conversions` (fmm-validate/CLAUDE.md), release, one thread pinned to one core, BLAS
threads 1, median of three passes after one untimed:

- S → F and F → R per box over a batch of min(boxes, 64);
- the RCR pair as the FMM's RCR level call runs a level by pairs on one thread (CH4.0i's pair
  split: `m2l_contribution` in class order, then the adds; the blocks within the default 512 MiB
  budget, so stored at p ≤ 76 and formed per pair from p = 123), with two multipole planes
  (§9.1's complex charges, the harness's default) and one;
- the diagonal pair: a **stand-in** for CH4.2's step, G_t += T̂_c[π(s)] F_s over every sample
  through a permutation (ring reversal and azimuth reflection), 34 transfer functions, 16 source
  signatures and 8 targets cycled;
- the level modelled: RCR = pairs × the RCR pair (two planes); diagonal = boxes × (S → F + F → R)
  + pairs × the diagonal pair, every pair diagonal (S6's calibrated rule sends no class of real k
  to RCR at k a ≥ 45 for ε ≥ 10⁻⁸: `spikes/diagonal/results/summary.md` §3, "real, calibrated");
  "φ free" removes the φ step's share of the conversions (its share of their multiply-adds,
  0.80, both steps running through the same kernel).

The cases: W1 at kD = 500, level 3 (k̂ = 31.25, k a = 54.1) at p = 66, 72, 76 and level 2
(k̂ = 62.5, k a = 108.3) at p = 123, 131, 135, the degrees of 10⁻³, 10⁻⁶ and 10⁻⁸, with the
boxes and pairs of CH4.0e §3.1; W1 at kD = 1000 at 10⁻⁶, levels 4, 3, 2 (p = 72, 131, 245),
with kD = 500's tree's counts one level down (the same surface; an approximation).

locust (GH200, Neoverse V2, core 5; load 1.5–2.6 of 72 before and after; 2026-11-19 20:13–20:18):

| p | k̂ | N_φ | S → F ms | F → R ms | GF/s | rel. err. S → F, F → R | RCR pair ms (2 planes) | RCR pair ms (1 plane) | diagonal pair ms | S → F + F → R ÷ RCR pair | φ share | boxes | pairs | RCR level s | diagonal level s | diagonal ÷ RCR | φ free s | φ free ÷ RCR |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 31.25 | 268 | 0.830 | 0.933 | 6.93 | 1.2e-15, 1.8e-15 | 0.811 | 0.587 | 0.054 | 2.17 | 0.80 | 512 | 12216 | 9.903 | 1.557 | 0.157 | 0.834 | 0.084 |
| 72 | 31.25 | 292 | 1.061 | 1.452 | 6.28 | 1.1e-15, 1.6e-15 | 1.063 | 0.781 | 0.067 | 2.36 | 0.80 | 512 | 12216 | 12.987 | 2.104 | 0.162 | 1.074 | 0.083 |
| 76 | 31.25 | 308 | 1.213 | 1.640 | 6.48 | 9.0e-16, 2.1e-15 | 1.246 | 0.932 | 0.075 | 2.29 | 0.80 | 512 | 12216 | 15.221 | 2.372 | 0.156 | 1.204 | 0.079 |
| 123 | 62.5 | 496 | 4.054 | 5.920 | 7.76 | 1.4e-15, 2.3e-15 | 5.422 | 4.125 | 0.137 | 1.84 | 0.80 | 64 | 2504 | 13.576 | 0.982 | 0.072 | 0.471 | 0.035 |
| 131 | 62.5 | 528 | 4.854 | 5.912 | 8.66 | 1.4e-15, 2.6e-15 | 6.980 | 5.518 | 0.167 | 1.54 | 0.80 | 64 | 2504 | 17.477 | 1.106 | 0.063 | 0.554 | 0.032 |
| 135 | 62.5 | 544 | 5.228 | 7.061 | 8.30 | 1.8e-15, 3.0e-15 | 7.679 | 6.032 | 0.183 | 1.60 | 0.80 | 64 | 2504 | 19.227 | 1.245 | 0.065 | 0.615 | 0.032 |
| 72 | 31.25 | 292 | 1.054 | 1.444 | 6.32 | 1.1e-15, 1.6e-15 | 1.053 | 0.783 | 0.058 | 2.37 | 0.80 | 3840 | 54000 | 56.851 | 12.716 | 0.224 | 5.044 | 0.089 |
| 131 | 62.5 | 528 | 4.880 | 5.928 | 8.63 | 1.4e-15, 2.6e-15 | 6.962 | 5.497 | 0.166 | 1.55 | 0.80 | 512 | 12216 | 85.049 | 7.564 | 0.089 | 3.130 | 0.037 |
| 245 | 125 | 984 | 25.513 | 33.997 | 10.08 | 2.1e-15, 3.6e-15 | 59.316 | 44.511 | 0.884 | 1.00 | 0.80 | 64 | 2504 | 148.527 | 6.023 | 0.041 | 2.973 | 0.020 |

Kathleen (Xeon Gold 6248, job 644326, `singlenode`, node-c11a-079, core 3, exclusive node;
2026-11-19 20:14–20:24):

| p | k̂ | N_φ | S → F ms | F → R ms | GF/s | rel. err. S → F, F → R | RCR pair ms (2 planes) | RCR pair ms (1 plane) | diagonal pair ms | S → F + F → R ÷ RCR pair | φ share | boxes | pairs | RCR level s | diagonal level s | diagonal ÷ RCR | φ free s | φ free ÷ RCR |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 66 | 31.25 | 268 | 1.763 | 2.253 | 3.04 | 1.2e-15, 1.8e-15 | 1.454 | 1.099 | 0.126 | 2.76 | 0.80 | 512 | 12216 | 17.767 | 3.600 | 0.203 | 1.955 | 0.110 |
| 72 | 31.25 | 292 | 2.188 | 2.994 | 3.04 | 1.1e-15, 1.6e-15 | 1.889 | 1.430 | 0.163 | 2.74 | 0.80 | 512 | 12216 | 23.075 | 4.650 | 0.202 | 2.527 | 0.110 |
| 76 | 31.25 | 308 | 2.488 | 3.376 | 3.15 | 9.0e-16, 2.1e-15 | 2.223 | 1.690 | 0.187 | 2.64 | 0.80 | 512 | 12216 | 27.158 | 5.286 | 0.195 | 2.885 | 0.106 |
| 123 | 62.5 | 496 | 7.947 | 11.424 | 3.99 | 1.4e-15, 2.3e-15 | 10.304 | 7.888 | 0.500 | 1.88 | 0.80 | 64 | 2504 | 25.801 | 2.491 | 0.097 | 1.497 | 0.058 |
| 131 | 62.5 | 528 | 9.235 | 11.940 | 4.40 | 1.4e-15, 2.6e-15 | 13.363 | 10.500 | 0.572 | 1.58 | 0.80 | 64 | 2504 | 33.461 | 2.788 | 0.083 | 1.702 | 0.051 |
| 135 | 62.5 | 544 | 9.960 | 13.128 | 4.42 | 1.8e-15, 3.0e-15 | 14.578 | 11.601 | 0.599 | 1.58 | 0.80 | 64 | 2504 | 36.504 | 2.978 | 0.082 | 1.794 | 0.049 |
| 72 | 31.25 | 292 | 2.234 | 2.927 | 3.06 | 1.1e-15, 1.6e-15 | 1.894 | 1.435 | 0.165 | 2.72 | 0.80 | 3840 | 54000 | 102.299 | 28.716 | 0.281 | 12.861 | 0.126 |
| 131 | 62.5 | 528 | 9.257 | 11.901 | 4.41 | 1.4e-15, 2.6e-15 | 13.345 | 10.515 | 0.570 | 1.59 | 0.80 | 512 | 12216 | 163.020 | 17.798 | 0.109 | 9.118 | 0.056 |
| 245 | 125 | 984 | 50.426 | 69.207 | 5.01 | 2.1e-15, 3.6e-15 | 126.485 | 87.294 | 1.881 | 0.95 | 0.80 | 64 | 2504 | 316.718 | 12.368 | 0.039 | 6.237 | 0.020 |
| 1 | passed | 69.635 |
| 2 | passed | 507.550 |

Reading:

- **Per box.** Two conversions cost 1.5–2.8 RCR pairs at p = 66–135 and 0.95–1.0 at p = 245:
  the conversions grow like P³ at the kernel's rate, the RCR pair like P³ at a falling rate (its
  blocks formed per pair from p = 123, its working set beyond the caches).
- **Rate.** The conversions run at 6.3–10.1 GF/s on a Grace core and 3.0–5.0 on a Cascade Lake
  core: safe Rust, the column kernel autovectorised, no fma. On x86_64 the workspace builds for
  the baseline ISA (SSE2), so this is 128-bit code on an AVX-512 machine; the RCR pair is built
  the same way, so the comparison is between the two as built. A dispatched kernel (AVX-512, fma)
  could cut the conversions' time again; recorded as an "Also" of CH4.1b, not needed for the
  decision.
- **The diagonal pair** stand-in costs 0.05–0.18 ms at p = 66–135 on locust, 0.13–0.60 on
  Kathleen: 2–9 % of an RCR pair. It is memory-bound (a 0.6–2.4 MB transfer function read
  through a gather per pair); CH4.2 measures the real one.
- **Pairs per box decide the level.** Level 2 has 39 pairs per box, level 3 24, kD = 1000's
  level 4 14; the fewer the pairs, the larger the conversions' share and the smaller the gain
  (0.22–0.28 of RCR at 14 pairs per box).

## 3. H4's gain, with the product and with an FFT

The FFT replaces the φ step only (proposal §6.7); the Legendre step and the diagonal pairs stay.
Its gain is at most the φ step's time, which "φ free" removes. Per level that is about half of
the diagonal level's time at 24–39 pairs per box: 0.03–0.09 of the RCR level.

On the whole run, W1 at kD = 500 at the best split, CH4.0e's §3.1 levels 2 and 3 M2L seconds
scaled by this report's one-core ratios of the same degree (an estimate: it assumes the diagonal
level threads as well as RCR's does on the node; the conversions are per box and independent,
the pairs per target, as RCR's):

Kathleen:

| ε | levels 2, 3 M2L s (CH4.0e) | ours s | with the product: levels s | ours s | ÷ | φ free: levels s | ours s | ÷ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 2.972 | 9.851 | 0.371 | 7.250 | 0.74 | 0.213 | 7.092 | 0.72 |
| 1e-6 | 3.676 | 11.714 | 0.428 | 8.466 | 0.72 | 0.248 | 8.287 | 0.71 |
| 1e-8 | 4.240 | 19.886 | 0.492 | 16.138 | 0.81 | 0.281 | 15.927 | 0.80 |

locust:

| ε | levels 2, 3 M2L s (CH4.0e) | ours s | with the product: levels s | ours s | ÷ | φ free: levels s | ours s | ÷ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-3 | 1.303 | 4.479 | 0.111 | 3.288 | 0.73 | 0.056 | 3.232 | 0.72 |
| 1e-6 | 1.599 | 5.548 | 0.127 | 4.076 | 0.73 | 0.065 | 4.014 | 0.72 |
| 1e-8 | 1.128 | 8.337 | 0.102 | 7.311 | 0.88 | 0.051 | 7.260 | 0.87 |

At kD = 1000 (outside D1; CH4.0e §3.3, levels 2–4 summed, their ratio here the one-core
model's over the three levels, weighted by its RCR times):

| machine | ε | levels 2–4 M2L s | ours s | with the product: ours s | ÷ | φ free: ours s | ÷ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kathleen | 1e-6 | 36.180 | 85.345 | 52.82 | 0.62 | 50.92 | 0.60 |
| locust | 1e-6 | 15.307 | 39.550 | 25.63 | 0.65 | 24.83 | 0.63 |

None of these includes the transfer functions' build (CH4.0e §3.1: about 4.4 s for level 2 and
1.4 s for level 3 serially by S6's interpolation, about 0.2 s over a node's threads; CH4.2's)
or the signatures' scratch traffic (CH4.1b).

**Against A5.** A5's 2.3× per level assumed 18P³ complex multiply-adds per conversion (the DFT as
a product without the grid's symmetries, the Legendre step without parity, complex arithmetic
throughout). The transforms here need 2.5P³ (the θ-factors and the cos and sin are real, the
parity halves the Legendre step, the symmetries quarter the φ step), and run at a rate close to
the RCR pair's; so the per-box cost is 1–3 RCR pairs, not 18, and a level of W1 at kD = 500 is
5–16× faster on one core (3.6–26× over every case here), not 2.3×.

## 4. Checks

`/verify` on the M6 (branch `tbetcke/h-ch4-1a`):

| command | result |
| --- | --- |
| `cargo fmt --all -- --check` | passed |
| `cargo clippy -- -D warnings`; `--examples`; `--workspace --all-targets` | passed |
| `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps` | passed |
| `cargo test` at opt-level 1 (`target/opt1`) | 1,317 passed (a first run failed `Grid::theta`'s new doctest, a wrong expected value; fixed) |
| `cargo test --workspace` at opt-level 1 | 1,361 passed |
| `cargo test -p nd-fmm-tables --release --test helmholtz diagonal -- --include-ignored` | 9 passed (p = 41, 72, 131 included) |
| `cargo test -p nd-fmm-tables --test helmholtz` at opt-level 0, the executable the task adds to | 94 passed in 38.4 s at load 4.0 (the diagonal tests 0.14 s of it) |
| the MPI list | not run: no MPI crate's code changed (`nd-fmm-validate` gains an MPI-free example) |
| Laplace's fixtures and benchmarks | the fixtures pass in the default run; no benchmark run: no Laplace path changed (a new module of `nd-fmm-tables`, two accessors of `nd-fmm-ref`'s Helmholtz module made public) |
| `cargo deny` | not run: no manifest changed |

## 5. Questions for sign-off

1. **The order (A5's stop).** H4 goes on as listed with the product (CH4.1b, CH4.2, CH4.3,
   CH4.4, CH4.5), and the FFT track (CH0.9, CHR.3, CH4.6) stays item 11 after them
   (**recommended**: an FFT could save at most 1–2 % of W1 at kD = 500's run inside D1, about 2 %
   at kD = 1000). Alternatives: the FFT track moves ahead of CH4.2; or it leaves the priority list
   (to be taken by name).
2. **The acceptance as read.** CH4.1a's *Done when* met by: the round trip to the grid's bandwidth,
   the reference to round-off and a box's bits in every batch (tested), and the measurement (§2–§3)
   with the level modelled from measured per-box and per-pair times and a stand-in for CH4.2's
   diagonal pair, not timed in a level call (CH4.1b and CH4.2 build that); the chunked
   accumulators' bits and the budget are CH4.1b's (**recommended**: accept).
3. **Proposal §6.7's count.** Its "about 16P³ as a product" (and the table's "DFT as a product"
   column) counts the φ step without the grid's symmetries in complex arithmetic. Amend §6.7 with
   the form measured here (2P³ + 8P³ real multiply-adds per conversion, 2.5P³ complex; per box
   1–3 RCR pairs on locust and Kathleen) and the FFT's bound from §3, with a line in Appendix A
   (**recommended**); or leave the proposal and keep the record here.
