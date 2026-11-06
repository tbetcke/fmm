# CH6.2b The M2L timed per level

Started 2026-11-04 on `tbetcke/h-ch6-2b`. Roadmap: "CH6.2b The M2L timed per level" (H6;
priority list item 3, split on this branch from the measurements of CH6.2 into this task and
CH6.2c, FMM3D with D8). Proposal: §10 CH6.2; §2 (constraints 1, 2, 7, 9), §5.3
(Exponential M2L), §6.4, §7, §9.1; CH6.1e's report §4 and §7, CH6.2a's report §3–§4.

## Goal

Time the M2L of every level under Rotation, Batched, Auto and the exponential M2L on the
three machines, so that CH6.2c can set them beside FMM3D's and propose D8.

## Acceptance (proposal §10, verbatim)

CH6.2: "per-level accuracy and time beside FMM3D's on the same problems; adopted as a
strategy where it wins, else kept opt-in or dropped (D8)". FMM3D and D8 are CH6.2c's. This
sub-task's, from the roadmap's CH6.2b:

> the per-level M2L times of Rotation, Batched (fused), Auto and PlaneWave (the rule) on
> CH6.1e's matrix at ε = 10⁻³ and 10⁻⁶ (W1 at kD = 10, 30, 100; W3 at kD = 10, 30; the
> four W4 distributions at kD = 1 and 100), on the M6, locust and Kathleen, and on W1 at
> kD = 500 and W3 at kD = 100 on locust and Kathleen, each launch twice in reversed order;
> the fastest strategy per level and machine, and what a per-level choice would gain over
> each strategy alone. Also: at the default waves budget (8 GiB per node), and the ranks ×
> threads split on the large cases (one rank per node, CH3.6's split, one rank per core).

Amended on 2026-11-04 (the user: "yes, go ahead with that change"): Kathleen was down (its
login node refused every connection from 10:10), so the acceptance is the M6 and locust,
and Kathleen's runs move to CH6.2c's line, which runs on Kathleen anyway.

The per-level accuracy of the exponential M2L against Rotation is CH6.1e's scan (its report
§2, each level's waves alone); this task records the end-to-end errors of every strategy
beside the times, and CH6.2c the per-level accuracy against FMM3D's.

## Read first

- `fmm-exec/src/helmholtz/operator.rs` (`FmmOperator::m2l`, `select_tables`,
  `plane_wave_counts`), `operator_plane_wave.rs` (`PlaneBuffer::counts`).
- `fmm-validate/src/helmholtz_accuracy.rs` (`Row`, `run_output`),
  `examples/helmholtz_accuracy.rs` (the table, `--m2l` lists), `tools/helmholtz/ch6-2a-runs.sh`,
  `tools/helmholtz/ch3-2d-runs.sh` (the macOS quiet gate).
- `fmm-plan/src/lists.rs` (`VList`, `GroupedCsr::len`).
- CH6.1e's report §4 (the matrix, the times on a loaded M6), CH6.2a's report §3–§4.

## Changes

No Laplace file, convention, dependency or `nd-fmm-plan` change. Crates: `nd-fmm-exec`
(the Helmholtz operator only) and `nd-fmm-validate` (the harness, the example, the run
script).

1. **The M2L time per level** (`helmholtz/operator.rs`): `FmmOperator::m2l` reads the
   clock around its dispatch (Batched's block, the exponential M2L's level call or the
   driver's target-by-target call) and adds the elapsed time to the level's entry;
   `HelmholtzOperator::m2l_times() -> &[Duration]` (one entry per level, summed over the
   level calls since the tables were last chosen, as `plane_wave_counts`). Two clock reads
   per level call, no allocation (the entries sized with the tables, reset in
   `select_tables`),
   no change to any value: the output stays bit for bit. Always on, as the counts are (a
   level call takes milliseconds; Laplace's opt-in `KindTiming` lives in `ExecOperator`,
   which the Helmholtz FMM does not use).
2. **The harness** (`nd-fmm-validate`): `Row::m2l_seconds` (per level, the slowest rank's,
   of the last evaluation) and `Row::v_pairs` (per level, the V pairs of the targets each
   rank owns, `Plan::computes`, summed over the ranks: a replicated top box has rows on
   every rank), `Case::evaluations` and `Row::first_evaluate` (several evaluations of one
   build; the first touches the buffers, seconds on a large waves buffer), the example's
   `--levels` (after the table, one row per case, strategy and level: degree, the M2L that
   ran, V pairs, seconds and ns per pair, and Σ; the existing table unchanged, so the
   earlier reports' commands repeat) and `--evaluations n`.
3. **The run script** `tools/helmholtz/ch6-2b-runs.sh`: stages `matrix` (the cases at one
   rank × the machine's cores, `--m2l rotation,batched-fused,auto,planewave` then reversed,
   `--evaluations 2 --levels`) and `large` (W1 at kD = 500 and W3 at kD = 100 at 10⁻³ and
   10⁻⁶, the same strategies, at each split of `SPLITS`; several ranks as `mpirun -n R
   --map-by slot:PE=T --bind-to core`), each launch under a timeout, the macOS quiet gate
   of CH3.2d's script; and `tools/helmholtz/ch6-2b-tables.py`, the report's tables (the
   ABBA means per level, the fastest method per level, the per-level best).
4. Docs: the operator's module docs ("M2L"), `fmm-exec/CLAUDE.md` and
   `fmm-validate/CLAUDE.md` entries, the report `fmm-validate/results/helmholtz-ch6-2b.md`.

## Tests that prove it

- `fmm-exec/tests/helmholtz_fmm.rs`: in `check_plane_wave` and `check_auto` (one build
  each, already there), `m2l_times()` has one entry per level, zero on level 0, positive
  on every level with V pairs on this rank (the evaluator's downward pass calls M2L on
  every level from 1, empty ones included, so those get a small time: "zero on levels
  without V pairs" cannot hold); the output unchanged (the existing bit-for-bit checks
  stay).
- `fmm-validate/tests/helmholtz_accuracy.rs`: `Row::v_pairs` summed over the ranks the same
  at every rank count (against the one-rank tree built on `self_comm`), none on levels 0
  and 1, `m2l_seconds` zero on level 0 and positive where `v_pairs` is; two evaluations of
  one build the one-evaluation checksum and errors. The example's `--levels` table is
  exercised by the machine runs, not by a test (examples are not tested in this crate).

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
`cargo test` and `--workspace`, the MPI list at 2 and 4 ranks), `cargo test -p
nd-fmm-validate --release --test helmholtz_accuracy`. D9 not needed (no Laplace path).

## Machine runs

Release, BLAS threads 1, the default P2P kernel, the default waves budget (8 GiB per
node), two evaluations per build (the second's times), every launch of a case twice with
the strategies in reversed order (ABBA):

- **Kathleen**, one node (`singlenode`): `matrix` at 1 × 40; `large` at 1 × 40, 4 × 10
  (CH3.6's split; `--map-by slot:PE=10`, two ranks per socket as CH3.6's
  `ppr:2:socket:PE=10`) and 40 × 1. The job and node in the report.
- **locust**: `matrix` at 1 × 72; `large` at 1 × 72, 8 × 9 and 72 × 1. Load checked first
  (`uptime`), stated in the report.
- **The M6** (24 GB; the user's go-ahead first, they run other jobs on it): `matrix` at
  1 × 6 under the quiet gate; no large case (14.6–17.1 GB peak with the whole level, CH6.2a).

Decides: the fastest strategy per level and machine, and the gain of a per-level choice
over each strategy alone, for CH6.2c's D8 proposal. Not a gate: no quiet-machine rule
beyond the gate on the M6 and Kathleen's exclusive node.

## Constraints and risks

- Constraint 1: no Laplace file changes; the timer is in the Helmholtz operator only.
- Constraint 2: the timer reads the clock and changes no value; the existing bit-for-bit
  checks (threads, ranks, budgets) keep holding.
- Constraint 7: the default waves budget of CH6.2a holds; the large cases are not run on
  the M6.
- Constraint 9: nothing chosen from a time.
- Risk: the cases at kD = 10 evaluate in under 0.3 s, so their per-level times are a few
  ms and noisy; reported, not used for a rule. On the M6 the times move ±25 % at 6 threads
  (CH3.4c), hence the ABBA order and the gate.

## Pending runs

- Done: the M6, 2026-11-04, `THREADS=6 tools/helmholtz/ch6-2b-runs.sh <out> matrix`
  (binary of cd67415; 9 min, one launch repeated for contention). The report's §2–§3.
- Done: locust, 2026-11-04, `THREADS=72 SPLITS="1x72 8x9 72x1" tools/helmholtz/ch6-2b-runs.sh
  /data/ucahtbe/logs/ch6-2b matrix large` (source 1e22995), and the W3 large cases at
  8 × 9 and 72 × 1 again after the stdin fix (8fb9f84). The report's §2–§4.
Moved to CH6.2c (the amendment above), done there on 2026-11-06 (jobs 628611 and 629953;
the report's §4 and §5):

- Kathleen job 628611 (node-c11a-142, 43 min, completed, every launch exit 0): the build,
  `THREADS=40 tools/helmholtz/ch6-2b-runs.sh /scratch/scratch/ucahtbe/fmm/logs/ch6-2b
  matrix large` (1 × 40). Pending: fetch `/scratch/scratch/ucahtbe/fmm/logs/ch6-2b/` and
  add `kathleen=<dir>` to the tables.
- Kathleen, not submitted: after `tools/kathleen/sync.sh`, `tools/kathleen/submit.sh
  --time=06:00:00 -- run.sbatch 'cargo build --release -p nd-fmm-validate --example
  helmholtz_accuracy' 'THREADS=40 SPLITS="4x10 40x1" tools/helmholtz/ch6-2b-runs.sh
  /scratch/scratch/ucahtbe/fmm/logs/ch6-2b large'` (the script's stdin fix needs the sync).
  Decides Kathleen's ranks × threads split. The login node has closed every connection
  through UCL's gateway since 2026-11-04 10:10.

## Open questions

None.
