# CH6.2h D8: the exponential M2L's level rule

Started 2026-11-08 on `tbetcke/h-ch6-2h`. Roadmap: "CH6.2h D8: the exponential M2L's level
rule" (H6; priority list item 4, the last part of CH6.2; *Sign-off:* D8). Proposal: §10
CH6.2; §2 (constraints 1, 2, 7, 9), §5.3, §6.4, §12 D8; CH6.2b's report §7, CH6.2c's
(CH6.2b's report §4–§5), CH6.2d's report §3, CH6.2e's, CH6.2g's report §3, CH6.2f's report
§3–§6 and §9.

## Goal

Propose D8: the level rule by which `HelmholtzM2l::Auto` takes the exponential M2L, built and
measured behind a setting, made the default only on D8's sign-off.

## Acceptance (proposal §10, verbatim)

CH6.2: "per-level accuracy and time beside FMM3D's on the same problems; adopted as a strategy
where it wins, else kept opt-in or dropped (D8)". This sub-task's, from the roadmap:

> *Done when:* D8 proposed with the rule's inputs (degree, ε, the level's boxes), on CH6.2f's
> level call (or with Rotation on levels of few boxes, if CH6.2f's split does not bring them
> within the rule).

## What the data before this task shows

CH6.2b's per-level tables (its appendix, the three machines; levels 2–3 then under the split
by parents) by degree, where the exponential M2L ("PW") and Batched both ran:

| machine (ISA, Batched to) | p = 7 PW ÷ Batched | p = 8 | p = 11 | p = 13 | p = 15 | p = 16–18 |
| --- | --- | --- | --- | --- | --- | --- |
| the M6 (NEON, 10) | 0.70–2.77 (median 1.13) | 1.07–1.24 | 0.38–0.47 | 0.38 | 0.09–0.24 | 0.12–0.14 |
| locust (NEON, 10) | 0.57–3.15 (0.85) | 0.53–1.91 | 0.33–1.16 | 0.15–0.47 | 0.05–0.36 | 0.05–0.19 |
| Kathleen (AVX-512, 15) | 2.12–9.02 (3.25) | 2.34–4.37 | 1.14–2.47 | 0.61–1.30 | 0.16–0.84 | 0.19–0.63 |

So on NEON Batched to its crossover (p ≤ 10) and PW above it; on AVX-512 Batched to p = 11
and PW from p = 12 or 13 (p = 13: PW faster on three of four levels; p = 12 occurs only on
coarse levels, where CH6.2b timed the old split). Above the crossover PW beats Rotation on
every level of many boxes; on levels 2–3 it lost (to 35×) under the old split, which CH6.2f
fixed (0.38–0.88× Rotation's at one rank on locust and Kathleen, 0.44–0.69× level 2 on the
M6). Whether a boxes term is still needed is this task's measurement on CH6.2f's level call.

## Read first

- `fmm-exec/src/helmholtz/operator.rs` (`HelmholtzM2l`, `select_tables`, `m2l_on`,
  `auto_batched_max_degree`), `fmm-exec/src/helmholtz/fmm.rs` (the builder's agreed
  crossover), `nd_fmm_tables::helmholtz::plane_wave::merge::select`,
  `nd_fmm_tables::helmholtz::batched::BatchedM2l::select_within`.
- `fmm-exec/tests/helmholtz_fmm.rs` (`check_auto`, `check_plane_wave`),
  `fmm-exec/tests/helmholtz/dense.rs` (the Auto rule's tests), `tests/helmholtz/plane_wave.rs`.
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case::m2l*`), `examples/helmholtz_accuracy.rs`,
  `tools/helmholtz/ch6-2b-runs.sh`, `ch6-2b-tables.py`; CH6.2b's, CH6.2f's and CH6.2g's
  reports.

## Changes

`nd-fmm-exec` (the Helmholtz operator and builder), `nd-fmm-validate` (the harness),
`tools/helmholtz/`. No convention, table, dependency, `nd-fmm-plan` or Laplace change.

1. **The rule under `Auto`** (`HelmholtzOperator::with_auto_plane_wave(bool)`,
   `HelmholtzFmmBuilder::auto_plane_wave(bool)`, off until D8's sign-off). Per level l ≥ 2 with
   degree p_l and κ_l = k w_l:
   1. the exponential M2L where p_l ≥ d_pw, `select` gives the level a quadrature (κ_l in the
      tables; ε ≥ 10⁻⁶, iprec 2, CH6.1e's rule) and Im κ_l < 5π (the domains CH6.2g's guard
      test covers; above it the chains err 10⁻⁹–10⁻⁵ absolute, CH6.2g report §3);
   2. else Batched where p_l ≤ d_b (Auto's crossover) and its tables fit the dense budget,
      from the finest level up, the levels of 1. left out;
   3. else Rotation.
   d_pw is `auto_plane_wave_min_degree(isa, planes)`, a function of the machine class like
   d_b: d_b + 1 (PW above Batched's crossover) except AVX-512 with two planes, 12 (from the
   table above; the measurement below checks it against d_b + 1 = 16). Named for the
   measurements by `with_m2l_plane_wave_min_degree(Option<usize>)` and the builder's
   `m2l_plane_wave_min_degree`; the builder agrees it over the ranks (all-reduce min, as d_b)
   and reports it (`HelmholtzFmm::m2l_plane_wave_min_degree`). Every input is k, the level's
   width and degree, ε and the machine class: constraint 9. Each level runs Batched's,
   PlaneWave's or Rotation's level call unchanged (CH6.2f's split included), so the output is
   deterministic and rank-invariant as theirs (constraint 2); the waves buffer is the
   builder's existing node budget (constraint 7).
2. **The harness**: `--m2l auto-pw` and `auto-pw:<d>` (Auto with the rule, the rule's d_pw or
   d named), the column "M2L" saying so; `Case::{auto_plane_wave, m2l_plane_wave_min_degree}`,
   `Row::m2l_plane_wave_min_degree`.
3. **Runs and the report** `fmm-validate/results/helmholtz-ch6-2h.md`: per level and case the
   rule's choice and time against Rotation, Auto (today's default), PlaneWave and the per-level
   best, the evaluation, the errors; `tools/helmholtz/ch6-2h-runs.sh`, `ch6-2h-tables.py`.
4. **On D8's sign-off** (if the answer adopts it): `auto_plane_wave` on by default in the
   builder, the tests and docs restated, in this PR.

## Tests that prove it

- `tests/helmholtz/dense.rs` (operator, no MPI): under `Auto` with the rule each level's
  `m2l_on` is the rule's (PW where p ≥ d_pw with a quadrature and Im κ < 5π, Batched at or
  below d_b, Rotation else) for named d_b and d_pw, at ε = 10⁻³, 10⁻⁶ (PW) and 10⁻⁸ (none),
  a strongly damped k (no PW from Im κ_l ≥ 5π); off, the old rule bit for bit (the tables
  chosen); `auto_plane_wave_min_degree` of every ISA and plane count.
- `tests/helmholtz_fmm.rs` (`check_auto`, one rank and P ranks): Auto with the rule against
  the direct sum within `PLANE_WAVE_RATIO` (1.1) of Rotation's error; with d_b named 0 and
  d_pw 0, bit for bit `PlaneWave` (the rule's precisions) where every PW level is the same;
  with d_pw named above every degree, bit for bit today's `Auto`; 2 threads bit for bit; on
  P ranks the one-rank FMM bit for bit; the agreed d_pw reported.
- `tests/helmholtz_accuracy.rs`: `auto-pw` parsed, its row's method per level and errors
  within 1.1× Rotation's on a small case.

## Checks

`/verify` (format, Clippy default/examples/workspace all-targets, `cargo doc`, `cargo test`,
`cargo test --workspace`, the MPI list at 2 and 4 ranks); `cargo test -p nd-fmm-validate`.

## Machine runs

CH6.2b's matrix (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the four W4 at N = 10⁵, kD = 1,
100) and large cases (W1 at kD = 500, W3 at kD = 100) at ε = 10⁻³ and 10⁻⁶, each launch with
`--m2l rotation,auto,auto-pw,planewave` and reversed (ABBA), `--evaluations 2 --levels`:

- locust 1 × 72 (quiet node; check for other sessions' jobs first);
- Kathleen 1 × 40 (one `singlenode` job), with `auto-pw:16` added (d_pw = d_b + 1 on
  AVX-512), and the large cases at 4 × 10;
- the M6 1 × 6, matrix only, with the user's go-ahead.

They decide d_pw per machine class, whether a boxes term is needed, and the gain of the rule
over today's `Auto`, for D8.

## Constraints and risks

- Constraint 1: no Laplace path touched (the Helmholtz operator and builder only;
  `driver.rs` unchanged). `mpi_exec`, `mpi_threading`, `invariance` run in the MPI list.
- Constraint 2: each level runs an existing deterministic level call; d_pw agreed over ranks.
- Constraint 9: the rule reads p_l, κ_l, ε and the machine class only (CH6.2f's split reads
  the threads but decides no bit).
- Risk: at 10⁻³ PW runs iprec 2 (CH6.1e's rule), more waves than the tolerance needs; the
  rule takes it as measured. Risk: on small trees PW's fixed per-level cost (the waves of
  every frame) may lose to Rotation on a level of few boxes; the runs decide a boxes term.
- Accuracy: PW levels under CH6.1e's rule within 1.070× Rotation's on CH3.1c's cases;
  Batched levels Rotation's to rounding.

## Pending runs

- locust, one rank × 72, the matrix and the large cases behind a strict gate (others below
  30 % of a core for five minutes, another user's long Python process left out; a sample
  every 7 s during): `tools/gh200/remote.sh 'bash /data/ucahtbe/logs/ch6-2h-gated.sh'`, output
  `/data/ucahtbe/logs/ch6-2h-b` (`gate.log`, `during.log`). Started 2026-11-08 20:48, waiting
  for another session's GPU sweep. It decides locust's column of the report (§5) and whether
  locust's first pass, where Batched at p = 7 lost to Rotation on dense levels at 72 threads,
  holds on a quiet node. The first pass (beside a CUDA compile) is a record only.
- Done: the M6 1 × 6 (21:39–21:45, the user's go-ahead); Kathleen jobs 630761 (matrix,
  1 × 40) and 630762 (large, 1 × 40 and 4 × 10), both with `auto-pw:16`.

## Open questions

For sign-off (the report §8):
1. D8: `Auto` takes the rule by default (the builder's `auto_plane_wave` on), the defaults,
   tests and docs restated on this branch after the sign-off.
2. d_pw on AVX-512 with two planes: 12 (measured; d_b + 1 = 16 is 1.6–35× slower at 10⁻⁶).
3. The Im κ_l < 5π bound (CH6.2g's guarded domains).
4. No boxes term for the exponential M2L; Batched's few-box levels to a follow-up (CH4.0's
   sentence).
5. Proposal §5.3 and §6.4 ("opt-in until CH6.2") amended to D8's rule.
