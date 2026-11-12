# Brief: the review of the record

The brief of the first review pass of `/milestone-gate` (its step 3), given to one subagent
as written, after the header the gate fills in (the milestone, `main`'s commit, the earlier
review files).

You review the repository in the working directory (nd-project/fmm: fast multipole methods
in Rust; the root CLAUDE.md is the map). Do the work yourself; start no subagents.

## Limits

- Read only. Change, create or delete no file. No git command that changes state;
  `git log`, `git show` and `git diff` are fine.
- Compile and run nothing: no `cargo`, no `mpirun`, no script of `tools/`, no `ssh`.
- `gh pr view <n>` and `gh pr list --state merged` may be tried for pull request texts; if
  `gh` fails, say so and use the documents and `git log`.

## Scope

The mathematical formulations, the algorithmic decisions and the benchmark results, from the
written record. Not the code: open a source file only to check one stated constant, sign or
normalisation, and rarely. A second reviewer covers the architecture.

## Read

Start with what the earlier reviews list as not read (their "Coverage"), then what changed
since the last one (`git log <its commit>..HEAD -- docs fmm-validate/results spikes`).

- `docs/CONVENTIONS.md`: every basis function, phase, layout and scaling.
- `docs/design/*.md`, the proposal `helmholtz-fmm-proposal.md` first.
- `docs/helmholtz/roadmap.md`, `progress.md`, `tasks/*.md`, `gates/*.md`.
- The reports: `fmm-validate/results/*.md`, `spikes/*/SPIKE_REPORT.md`, the phase briefs
  under `docs/phase*/`.

## Check

1. **Formulations.** Definitions, normalisations and phases; the translation formulas;
   quadratures; scalings (overflow, underflow, loss of accuracy); the degree and accuracy
   rules; error estimates. Inconsistencies between documents or with CONVENTIONS; formulas
   wrong, or right only in a regime not stated; claims without support.
2. **Decisions.** Are they founded on the recorded evidence? Rejected alternatives for
   sound reasons? Any decision made early, contradicted by later data, or resting on one
   noisy measurement? A standard technique of the literature missing or dismissed without a
   reason?
3. **Benchmarks.** Plausible and consistent (scaling in N, cost and accuracy against the
   degree, efficiencies, comparisons with FMM3D)? Sound method (a quiet machine,
   repetitions, what a time includes, the same accuracy, threads and leaf size on both
   sides)? Numbers that contradict each other across documents; conclusions the data do not
   carry; measurements a conclusion would need. Check the arithmetic of tables where cheap.
4. **Risks** for the milestones still open.
5. **The earlier findings.** For each finding of the earlier reviews in this pass's scope:
   resolved (where), open (which task has it), or no longer applies.

## Calibrate

A dozen findings with their evidence beat fifty guesses. Re-derive or cross-check whatever
you call wrong before you report it; what you could not verify is a *suspicion*, and labelled
so. Say briefly what you checked and found sound.

## Report

Your last message is the report, for the session that drives the gate. Markdown, with
sections Mathematics, Algorithms, Benchmarks, Risks, The earlier findings. Per finding: a
severity (high, medium, low), the file with its line or section, what is stated, why it is a
problem, a proposed resolution. End with what you read and what you did not reach.
