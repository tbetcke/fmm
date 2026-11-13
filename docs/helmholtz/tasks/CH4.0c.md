# CH4.0c The degree rule beyond its measured range

Started 2026-11-12 on `tbetcke/h-ch4-0c`. Roadmap: "CH4.0c The degree rule beyond its measured
range" (H4; priority list item 3), with its amendments from the review of 2026-11-12 (M6, M7:
the rule's stated range) and from CH4.0b's sign-off 2 (the constant density). Proposal: §10
CH4.0c; §3.5 (the calibrated rule); §9 item 8 (the under-resolved W3); §9.1 (workloads, norms,
the box measure); §2 (constraints 1, 6, 9). Sources: CH3.3's report §3–§6
(`fmm-validate/results/helmholtz-ch3-3.md`), the H3 gate §3 and question 4b
(`docs/helmholtz/gates/H3.md`), CH4.0b's report §4 (`fmm-validate/results/helmholtz-ch4-0b.md`),
the review `docs/reviews/2026-11-12-H6.md` M6 and M7.

## Goal

Check the calibrated degree rule's high-frequency branch where it is an extrapolation (W1 at
kD = 1000, x = 216 and 108 on its top levels) and on the under-resolved W3, against the
sampled direct sum, and state the range the rule is measured on where it is documented.

## Acceptance (proposal §10, verbatim)

> within ε, or the rule corrected or S7's degrees named per level

for "W1 at kD ≈ 1000 and 10⁻⁸, all-RCR, against the direct sum; the under-resolved W3
(N = 10⁶, kD = 200 and 500)". The roadmap's *Done when* adds: "the range of ε the rule is
measured on (D1's 10⁻³ to 10⁻⁸) stated where the rule is documented, in place of CH3.3's
10⁻¹⁸ to 10⁻¹", and (CH4.0b's sign-off 2) the runs include the constant density; CH4.0b's
sign-off 2 also names the plane wave's trace.

## What the record shows

- The rule: p(x, ε) = max(⌈0.30 L² + 0.25 L + 2.60⌉, ⌈x + 0.80 (ln 1/ε)^{2/3} x^{1/3}⌉),
  x = Re(k) a_l. Measured at ε/2 over 0.14 ≤ x ≤ 65, 10⁻⁸ ≤ ε ≤ 10⁻³ (plateau to 10⁻¹⁰);
  checked in the FMM at x = 108 (W1 at kD = 500: 0.63 ε at 10⁻⁸, the tightest of CH3.3).
- The excess p − x grows about linearly in L (CH3.3 §3), faster than (ln 1/ε)^{2/3}: the one C
  over-predicts by up to 8 degrees at 10⁻³ and leaves 2 at 10⁻⁸ (88 against 86 at x = 65).
- Pair evidence: below S7's per-pair p* from x ≈ 128 at 10⁻⁸ (157 against 161 at x = 128, 292
  against 297 at x = 256; CH3.3 §6). The FMM has never run there.
- W1 at kD = 1000 (N ≈ 7.96·10⁶, leaves on level 8 by c_leaf = 1) on the unit cube: x by level
  2–8 = 216.5, 108.3, 54.1, 27.1, 13.5, 6.8, 3.4; the rule at 10⁻⁸: 251, 135, 76, 44, 27, 24,
  24 (S7's closed form 259, 142, 81, 49, 46, 46, 46). Levels 2 and 3 are beyond x = 65, level 2
  beyond the one check.
- The under-resolved W3 (N = 10⁶): at kD = 200 leaves on level 5 (about 30 points), x ≤ 43; at
  kD = 500 leaves on level 7 by c_leaf (about 0.5 points per box, most boxes empty), x to 108.
  Never run in the FMM (H3 gate question 4b).
- CH4.0b §4: on W1 at kD = 100 the constant density misses ε in ℓ2 below D1 (1.5× at 10⁻⁹,
  3.4× at 10⁻¹⁰), all from level 2 on the high-frequency branch (x = 21.7, p = 40); within ε at
  10⁻⁸ (0.66 ε, ∇φ). The random charges do not show it.
- The operators at kD = 1000: the rotation blocks are accurate to n = 1000 (S1, the
  Gumerov–Duraiswami recursion in `nd_fmm_math::orthonormal::rotation`); the coaxial recursion
  was checked to p and k a of 300 (S2); the stored y-blocks are bounded by the block budget.

## Read first

- `fmm-exec/src/helmholtz/degree.rs` (module docs: the range, S7's comparison, the extrapolation
  note); `fmm-exec/CLAUDE.md` (the degree rule's bullet).
- `fmm-validate/examples/helmholtz_boxes.rs` and `fmm-validate/src/helmholtz_boxes.rs` (the
  reference at the samples plus s per box, the densities, `--study levels`).
- `tools/helmholtz/ch4-0b-runs.sh` (the launch pattern on Kathleen).
- Proposal §3.5 ("Calibrated (H3, CH3.3; amended at H3.G)").

## Changes

- `tools/helmholtz/ch4-0c-runs.sh`: the task's stages (below), one `helmholtz_boxes` launch per
  case, as CH4.0b's script.
- `fmm-validate/examples/helmholtz_boxes.rs`: a line on stderr per run as it finishes (the
  degrees, the build, the evaluation and its M2L by level, the error bookkeeping) and one for
  the problem's setup. The tables print only after the last run, so a long launch showed no
  progress (the user, 2026-11-12, during jobs 633320 and 633321).
- `fmm-exec/src/helmholtz/degree.rs`, module docs only: the measured range stated as D1's
  10⁻³ ≤ ε ≤ 10⁻⁸ (the plateau to 10⁻¹⁰), with S7's comparison over 10⁻¹⁸ ≤ ε ≤ 10⁻¹ named as an
  algebraic bound, not a measured range (review M6); the high-frequency branch checked at
  x = 216 and 108 with this task's result (or the correction); CH4.0b's below-D1 miss of the
  constant density named. `fmm-exec/CLAUDE.md`'s bullet the same.
- Proposal §3.5: the calibrated rule's paragraph states the range and this task's check, as an
  amendment for sign-off ("(amended at CH4.0c)", Appendix A line).
- The report `fmm-validate/results/helmholtz-ch4-0c.md`.
- No library code change unless the runs miss ε. If they do: the correction (stage D) changes
  `DegreeRule`'s high-frequency branch, or names S7's per-pair degrees above a stated x, as a
  proposal for sign-off; the plan is then amended with its tests (below) before the code.
- A defect the runs found (W3 at kD = 500, the default M2L, every density): `HelmholtzFmmBuilder`
  sized the exponential M2L's waves buffer for the plan's boxes before it set the buffer's
  bound, so it held every box of the largest level with waves until then; 218 GB there, refused
  by the node. The builder now sets the bound first (`fmm-exec/src/helmholtz/fmm.rs`); the test
  is `HelmholtzOperator::plane_wave_buffer_peak_bytes` (new, the most the buffer held) asserted
  equal to the bound's buffer in `check_plane_wave_buffer` (`tests/helmholtz_fmm.rs`), red
  before the fix and green after. No output changes. `helmholtz_boxes` also prints each run's
  errors as it finishes (job 633324 lost the Rotation run's errors to the abort).
- Public API: `HelmholtzOperator::plane_wave_buffer_peak_bytes` (added). Conventions: none.

## Tests that prove it

The acceptance is a measurement against the trusted path, the direct sum in f64
(`nd_fmm_ref::helmholtz::p2p::direct_sum` through `helmholtz_boxes::Reference`), so the machine
runs are the evidence:

- W1 at kD = 1000, ε = 10⁻⁸ (and 10⁻⁷, 10⁻⁶, 10⁻³), all-RCR (`--m2l rotation`) and the
  library's default: §9.1's relative ℓ2 errors of φ and ∇φ at the 1,000 samples ≤ ε; the
  random charges (§9.1), the plane wave's trace and the constant density.
- W1 at kD = 1000, 10⁻⁸, all-RCR, the levels' parts of the error (`--study levels`): how much
  of it levels 2 and 3 (x = 216, 108) carry, and whether the parts add up (the rest term).
- W3 at N = 10⁶, kD = 200 and 500, ε = 10⁻³, 10⁻⁶, 10⁻⁸, all-RCR and the default, the three
  densities: ℓ2 ≤ ε.
- Reported, not gated here: the box measure (§9.1's 100 ε bound is CH4.4's gate): at one target
  per box of level 8 for W1's random charges, at two per box of level 5 for W3 at kD = 200, at
  the samples alone elsewhere (named in each table).
- If the rule is corrected (stage D): `tests/helmholtz/degree.rs` (`MEASURED` and the rule's
  values at the new points), the doc test, and the pair and FMM evidence both (H6 gate §7) before
  it is proposed.
- No change to the FMM's code, so no new bit-identity or rank test; determinism and rank
  invariance are CH3.6's and CI's.

## Checks

`/verify` (docs and a script: fmt, Clippy, docs, the tests of nd-fmm-exec's doc tests for
`degree`; the rest as `/verify` picks). On Kathleen, the build inside each run job.

## Machine runs

Kathleen, one node per job (`singlenode`, 1 rank × 40 threads, release, AVX-512 P2P, BLAS
threads 1), `tools/kathleen/submit.sh --time=06:00:00 -- run.sbatch 'cargo build --release -p
nd-fmm-validate --example helmholtz_boxes' 'tools/helmholtz/ch4-0c-runs.sh <stage> logs/ch4-0c/<stage>'`:

- **A `w1-random`**: W1, kD = 1000, random charges, ε = 10⁻⁸ (first; it fixes the grid),
  10⁻⁷, 10⁻⁶, 10⁻³, Rotation and default; the reference at the samples and one target per
  level-8 box (about 2.9·10⁵ targets; about 80 min at CH4.0b's 5·10⁸ pairs per second).
  Decides the acceptance on W1.
- **B `w1-smooth`**: the same with the plane wave's trace and the constant density, the
  reference at the samples. Decides CH4.0b's sign-off 2 at kD = 1000.
- **C `w3`**: W3, N = 10⁶, kD = 200 and 500, the three densities, ε = 10⁻⁸, 10⁻⁶, 10⁻³,
  Rotation and default. Decides §9 item 8.
- **D `w1-levels`**: W1, kD = 1000, random, 10⁻⁸, Rotation, `--study levels` (the levels raised
  to the rule at 10⁻¹¹ one at a time), at the samples. Says how close the extrapolated levels
  are to their limit, whatever A finds.
- **E (only on a miss)**: `helmholtz_calibrate`'s level scan of the missing level(s) at the
  missing ε, the others held, for the measured degree, beside S7's per-pair p* (pair evidence);
  then the corrected rule on A–C's cases.

Accuracy runs: times reported from Kathleen's exclusive nodes, never asserted. Raw outputs to
`~/projects/fmm-archive/kathleen/logs/ch4-0c/`.

## Constraints and risks

- Constraint 1: Laplace untouched (Helmholtz docs, a run script, a report).
- Constraint 9: a correction may read k, the tree's geometry and ε, never the charges; a
  density that needs more raises the rule for all.
- Pair and FMM evidence (H6 gate §7): a rule change needs both the FMM's measured degree
  (stage E) and the worst pair (S7's p*) before it is proposed.
- A wider scope widens the verification set: this task widens no bound or default unless it
  corrects the rule; a correction above x = 65 changes the degrees of W1 at kD = 500 (x = 108)
  too, so its verification reruns CH3.3's kD = 500 rows and H6 §8's large cases.
- Risk: run time. Rotation at kD = 500, 10⁻⁸ took 88 s per evaluation (CH4.0b, 1 × 40); at
  kD = 1000 the top levels' degrees double (about 8× per pair) and N quadruples: an estimate of
  10–15 min per Rotation run at 10⁻⁸. Each job has a 6 h limit and each launch a timeout.
- Risk: memory. H6 §8 peaked at 21 GB on W1 at kD = 500 with both M2Ls; kD = 1000 an estimate
  of 4–6× that, within the node's 192 GB.
- D1's target range is kD ≤ 500; kD = 1000 is §9.1's and CH4.4's, beyond D1.

## Pending runs

Done: jobs 633320 (`w1-random`), 633321 (`w1-levels`), 633323 (`w1-smooth`), 633324 (`w3`;
kD = 200 complete, kD = 500 aborted in the default's build, the defect above). Outputs in
`~/projects/fmm-archive/kathleen/logs/ch4-0c/`; results in the report.

W3 at kD = 500 with the fix, one job per density: 633429 (random), 633430 (plane wave),
633431 (constant), all completed (within ε, at most 0.72 ε). None pending.

## Open questions

(filled when they arise)
