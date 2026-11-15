# CH4.0e The comparison under the default, and H4's expected gain

Started 2026-11-15 on `tbetcke/h-ch4-0e`. Roadmap: "CH4.0e The comparison under the
default, and H4's expected gain" (H4; priority list item 5). Proposal: no §10 row (added by
the review's triage, `docs/reviews/2026-11-12-H6.md` A1, B2, B5; question 1 of #209); §9.1,
§9.2, §3.6, §5.4, §7; S6's report (`spikes/diagonal/SPIKE_REPORT.md` §4, §8).

## Goal

Measure our Helmholtz FMM under the defaults as they stand after CH4.0d3 beside FMM3D's
`hfmm3d` on CH3.6's matrix, on Kathleen and locust, and from the per-level times bound what
H4 (the diagonal form on high-frequency levels) can still gain, so that the user can confirm
or change H4's go-ahead and order.

## Acceptance (roadmap *Done when*; no proposal §10 row)

The tables on Kathleen and locust; the estimate; the order of H4's tasks confirmed or
changed. In detail (the roadmap's three items):

1. CH3.6's matrix again under the defaults as they stand after CH4.0d3, beside `hfmm3d`
   (`default` and `fastker`), each code on the whole node at its best split, repeated; the
   build and the evaluation apart for ours, the whole call beside FMM3D's, at the same
   requested ε and at matched accuracy.
2. From the measured per-level times and S6's ratios (RCR over the diagonal pair), a bounded
   estimate of H4's gain per kD and ε, and what remains to FMM3D after it.
3. Kathleen against locust per level on W1 at kD = 500 (3.3–5.4× at a core ratio of 1.8),
   explained in a sentence or measured.

*Sign-off:* H4's go-ahead and order.

## Read first

- `docs/reviews/2026-11-12-H6.md` §1.3 A1, §1.4 B1–B5 (checked against the tree on
  2026-11-15: CH3.6 is still the only matrix with both codes; `helmholtz_fmm3d compare`
  still builds `Case::new`, that is Rotation with the near rules off, not the defaults).
- `fmm-validate/results/helmholtz-ch3-6.md` (the matrix, the method), `helmholtz-ch6-2d.md`
  (FMM3D per level), `helmholtz-ch4-0d3.md` (the defaults, round 3), `docs/helmholtz/gates/H6.md`
  §3, §8.
- `spikes/diagonal/SPIKE_REPORT.md` §4 (classes on RCR per (k a, ε); with the calibrated p
  none on levels 2–5 at kD = 500, 10⁻⁶) and §8, `results/times.md` (prototype pair and
  conversion times, one Grace core).
- `tools/fmm3d/README.md`, `run.sh`, `driver.f90` (the `helmholtz` mode: tokens, REPEAT is
  best of, FULL), `tools/helmholtz/ch6-2d-runs.sh` (driver and `helmholtz_fmm3d errors`),
  `tools/helmholtz/ch4-0d3-runs.sh` (our side at the defaults).

## Changes

No crate changes. The library's defaults are reached in the existing harness: the
`helmholtz_accuracy` example with `--m2l auto-pw --near-v cost --near-wx cost` and the leaf
size and c_leaf left to the library (CH4.0d3's round 3 ran the defaults so;
`nd-fmm-exec`'s `check_defaults` holds the named settings bit for bit to the builder's
defaults). Its case is `helmholtz_fmm3d problem`'s (same workload, seed, complex charges and
samples), so FMM3D's outputs are measured against the same oracle by `helmholtz_fmm3d
errors`.

- `tools/helmholtz/ch4-0e-runs.sh` (new): the stages
  - `ours`: CH3.6's matrix (W1 at kD = 10, 30, 100, 500; W3 at kD = 10, 30, 100; W4's
    cube, sphere, Plummer sphere and clusters at N = 10⁵ and kD = 1, 10, 100) at 10⁻³, 10⁻⁶
    and 10⁻⁸ at the defaults, `--levels`, two evaluations, at each split of SPLITS (ranks ×
    threads; Kathleen 1×40, 2×20, 4×10, 8×5 bound per socket; locust 1×72, 2×36, 4×18, 8×9),
    in the order of SPLITS and again reversed (ABBA);
  - `fmm3d`: per case and variant (`default`, `fastker`) the driver on every core over the
    eps grid 1e-2, 3e-3, 1e-3, …, 3e-9, 1e-9 (1-3 per decade; it includes the requested ε),
    REPEAT 2 (the best of two calls, setup included), launches a and b; `helmholtz_fmm3d
    errors` on launch a's outputs;
  - `t1`: one core: ours (1 × 1) and both FMM3D variants on W1 at kD = 100 (CH3.6's
    one-thread case) at the three ε and FMM3D's grid; ours on W1 at kD = 500 at 10⁻⁶ with
    `--levels` (the per-core part of item 3).
- `tools/helmholtz/ch4-0e-tables.py` (new): reads the launches; per machine and case the
  best split (the smallest build plus one evaluation, the minimum over the two launches),
  ours build, evaluation and both beside FMM3D's whole call at the same ε; FMM3D at matched
  accuracy: per case, ε and variant the largest grid eps whose φ and ∇φ errors are both at
  most ours (FMM3D at least as accurate in both fields; the tightest grid eps if none),
  and its time; the per-level tables of item 2 and 3; the estimate (below).
- `fmm-validate/results/helmholtz-ch4-0e.md` (new): the report.
- `docs/helmholtz/roadmap.md`, `docs/helmholtz/progress.md`, this file; any follow-up the
  sign-off creates as a roadmap line.

Public API: none. Conventions: none. Tables: none.

### The estimate (item 2)

Per machine, case and ε, from `--levels` at the best split: the levels whose M2L runs RCR
(Rotation or Batched, "M2L that ran") with k a_l ≥ 8 (below it S6 sends most classes to RCR
whatever the rule, and those levels are low-frequency); per level the share of V pairs whose
class the diagonal form admits (S6 §4's table for the calibrated p at the level's (k a, ε);
on W1 and W3 at kD ≤ 500 essentially all), and:

- the **upper bound** of the gain: that share of the level's M2L time removed entirely
  (diagonal pairs and conversions free);
- the **prototype estimate**: the share's time divided by S6's RCR/diagonal pair ratio at
  the level's p (5.3 at p = 24 to 53 at p = 256, interpolated in p), plus S → F and F → R
  per box at S6's prototype cost scaled by our measured RCR ns per pair over S6's RCR pair
  (the prototypes are scalar; ours are vectorised), on the level's boxes with V rows;
- what remains to FMM3D: our build plus one evaluation minus each, beside FMM3D's whole
  call at the same ε and at matched accuracy.

Bounded, not predicted: H4 adds scratch (proposal §7) and the class-permuted transfer
functions, which the prototype times do not include; the report says so.

## Tests that prove it

No code under test: the scripts are measurement tooling. Their correctness is checked by:

- a DRY_RUN listing of every launch on each machine before the jobs;
- the defaults: each launch's checksum at a split equal across the two launches (ABBA) and
  across the splits of one machine (rank- and thread-count invariance at the defaults; the
  harness's checksum is rank-invariant);
- our errors within ε on every case (§9.1's ℓ2 at 1,000 samples against the direct sum);
- FMM3D's errors from `helmholtz_fmm3d errors` against the same oracle; at the requested ε
  they reproduce CH3.6's to the digits shown where FMM3D's version and build are the same
  (a check that the problem files and the oracle are the same);
- the tables script run on the raw files twice gives the same report tables (deterministic),
  and its matched-eps choice checked by hand on three rows.

## Checks

`/verify` (docs and tools only: fmt and the Rust checks are unaffected; `shellcheck` and
`python3 -m py_compile` on the new scripts; the default checks still run as `/verify` decides).

## Machine runs

- Kathleen (`singlenode`, 1 node, ≤ 6 h each): job A `ours` (all splits, ABBA); job B
  `fmm3d` (both variants, launches a and b); job C `t1`. Each names its job id and node.
- locust (72 cores; `nvidia-smi` and `uptime` before, the load recorded): the same three
  stages at locust's splits, run detached in the synced tree, outputs fetched before any
  re-sync.
- The M6: none (no timing without the user's go-ahead).
- What it decides: nothing by itself; it is the evidence for the sign-off of H4's go-ahead
  and order.

## Constraints and risks

- Constraint 1: no Laplace code, no library code at all; Laplace is not touched.
- Constraint 2 (determinism): checked by the checksums across splits and launches.
- Risk: FMM3D's default variant at tight eps on W3 at kD = 100 (N = 4·10⁶) takes 30–40 s a
  call; the grid's 15 eps × 2 calls × 2 launches is about 1 h per variant on Kathleen: job B
  may need splitting by case (ONLY).
- Risk: matched accuracy moves in FMM3D's steps (its nterms tables), so a match is only as
  close as the grid; the report gives both errors.
- Risk: locust is shared; a contended launch is repeated, the load named.

## Pending runs

(filled in step 6)

## Open questions

(none yet)
