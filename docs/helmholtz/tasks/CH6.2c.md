# CH6.2c CH6.2b's Kathleen runs

Started 2026-11-06 on `tbetcke/h-ch6-2c`. Roadmap: "CH6.2c CH6.2b's Kathleen runs" (H6;
priority list item 4; the former CH6.2c, FMM3D with D8, split on this branch into this
task, CH6.2d beside FMM3D and CH6.2e D8, by `/next-task`'s size rule: Kathleen's runs, a
patched FMM3D build with per-level timers and accuracy, and a level rule with a remedy for
the coarse levels do not fit one reviewable PR). Proposal: §10 CH6.2; §2 (constraints 2, 7,
9); CH6.2b's report and plan (`docs/helmholtz/tasks/CH6.2b.md`, "Machine runs", "Pending
runs").

## Goal

Add Kathleen (the x86 AVX-512 class, a two-socket node) to CH6.2b's per-level M2L times, so
that CH6.2e proposes D8 from all three machines.

## Acceptance (proposal §10, verbatim)

CH6.2: "Per-level accuracy and time beside FMM3D's on the same problems; D8 proposed." This
task's share (the roadmap line): Kathleen's per-level tables, summary rows and ranks ×
threads table in CH6.2b's report (§2, §4, §5, the appendix), with the fastest strategy per
level on Kathleen and what a per-level choice gains there, and §7's observations checked
against them.

## Read first

- `fmm-validate/results/helmholtz-ch6-2b.md` (§1 method, §2 summary, §4 splits, §5, §7, §8).
- `tools/helmholtz/ch6-2b-runs.sh`, `tools/helmholtz/ch6-2b-tables.py`.
- `tools/kathleen/README.md`, `tools/kathleen/jobs/run.sbatch`.

## Changes

- `fmm-validate/results/helmholtz-ch6-2b.md`: Kathleen's rows in §2, a Kathleen table in §4,
  §5 written, the appendix's Kathleen part, §7 checked; the header's machine list.
- `tools/helmholtz/ch6-2b-tables.py` only if Kathleen's outputs need it.
- No code, no convention, no dependency; Laplace not touched.

## Tests that prove it

None new (no code). Every evaluation's checksum is its strategy's in both launches of an
ABBA pair (the tables script's check), and the exponential M2L's errors within CH6.1e's
1.1× of Rotation's on every Kathleen case.

## Checks

`/verify` reduces to the docs: no Rust or manifest changes. `cargo fmt --all -- --check`
for form; the tables script run on all outputs.

## Machine runs

- Kathleen job 628611 (2026-11-04, node-c11a-142): `matrix` and `large` at 1 × 40,
  completed. Fetched to the scratchpad and archived in
  `~/projects/fmm-archive/kathleen/logs/ch6-2b/`.
- Kathleen job 629953 (2026-11-06, `singlenode`, 6 h limit): the build, then
  `THREADS=40 SPLITS="4x10 40x1" tools/helmholtz/ch6-2b-runs.sh
  /scratch/scratch/ucahtbe/fmm/logs/ch6-2b large`. Decides Kathleen's ranks × threads
  split.

## Constraints and risks

Constraint 2 (no rule by threads): the report states the splits, never a rule from them.
Kathleen's nodes are exclusive, so no quiet gate; the job and node named.

## Pending runs

- Kathleen job 629953 (submitted 2026-11-06 18:36, running).

## Open questions
