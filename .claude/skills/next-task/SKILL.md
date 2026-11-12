---
name: next-task
description: The default development loop of the Helmholtz FMM work. Syncs the open task PRs, picks the next runnable task from docs/helmholtz/roadmap.md, plans it, implements it tests first, verifies it, logs it and pushes it as a PR. Use whenever the user says "continue", "next" or "keep going", names a roadmap ID such as CHR.2 or CH3.1, or asks what to do next.
---

# Next task

One pass takes one roadmap task as far as it can go here and leaves it as a pull request:
ready for the user's merge, or a draft that says what it waits for. The user's part is to
answer the questions this loop asks and to merge.

## Permissions

Invoking `/next-task` is the user's request to create the task's branch, commit to it, push
it, and open or update its PR (root `CLAUDE.md`, "Working agreement"). Merging, pushing to
`main`, force-pushing, rewriting history, deleting a branch and closing a PR each need a yes
in the current message.

## 0. Orient

- Read root `CLAUDE.md`, the last three entries of `docs/helmholtz/progress.md` and
  `docs/helmholtz/roadmap.md` (its rules, the decisions, the tasks). Keep
  `docs/design/helmholtz-fmm-proposal.md` at hand: §2 (constraints) always, §10 for the
  task's acceptance, and the sections the task cites.
- `git fetch origin`. If the working tree has changes: on a `tbetcke/h-*` branch this is an
  unfinished task, so resume it where its plan file (`docs/helmholtz/tasks/<ID>.md`) and the
  commits show it stopped; on any other branch stop and ask (never stash or discard work).
- Otherwise `git switch main` and `git pull --ff-only`.

## 1. Sync the open task PRs

First `main`'s CI: `gh run list --branch main --limit 5`. Report every job that failed or
was cancelled since the last pass (`gh run view <id> --json jobs`). A job cancelled by its
`timeout-minutes` is a defect of its own: say so and propose a separate small fix before
the next task is taken. A cancellation on `main` after a merge is seen by no other step.

Then `gh pr list --state open --author "@me" --json number,title,headRefName,isDraft,url`,
the branches `tbetcke/h-*`. For each, in roadmap order:

1. **CI failing** (`gh pr checks <n>`): fixing it is this run's task. Switch to its branch,
   fix, `/verify`, push, update the PR, and go to step 10.
2. **Review comments from the user not yet answered** (`gh pr view <n> --comments` and
   `gh api repos/{owner}/{repo}/pulls/<n>/comments`): addressing them is this run's task, in
   the same way.
3. **Draft waiting for machine runs** (its plan's "Pending runs"): check each job. Kathleen:
   `tools/kathleen/remote.sh 'sacct -j <id> --format=JobID,State,Elapsed'`; locust: the log
   the plan names. Finished: fetch the output, write the results into the report, check the
   acceptance again, update the plan and the progress entry, push, and if nothing else is
   pending `gh pr ready <n>`. Still queued or running: note it and go on.
4. **Draft waiting for a sign-off**: if the roadmap's decision table or the user's message
   now has the answer, record it (step 2's wording) on that branch, push and mark it ready.
5. **Ready and green**: it waits for the user's merge. Compare each job's time on its
   latest run with the job's `timeout-minutes` (`gh run view <id> --json jobs`: `startedAt`
   and `completedAt`); above about two thirds of the limit, note it in the PR and propose
   raising the limit.

Merged PRs need no bookkeeping: their roadmap tick and progress entry were part of them. If
item 1 or 2 applied, that was this run's task: stop at step 10 after it.

## 2. Pick

- If the user named a task, take it; if its *Needs* are not met, say which and ask.
- Otherwise take the first task in roadmap order that is runnable (the roadmap's "How
  `/next-task` reads this file"): open, no branch or open PR, every task in *Needs* ticked on
  `main`, every decision in *Needs* signed off, and D0 signed off.
- If the first task in order waits only for decisions, ask the user for them now: one
  question per decision, the proposal's recommendation first, then the alternatives §12
  names. Record each answer in the decision table as
  `signed off <YYYY-MM-DD>: <the answer as given>`, never stronger than what the user said;
  commit it on the task's branch (step 4). Then take that task.
- If nothing is runnable, say which PRs to merge, which decisions to take or which runs to
  wait for, and stop.
- **Size.** A task that will not fit one reviewable PR (as a rule of thumb more than about
  1,000 changed lines, more than two crates, or a new operator plus measurements on several
  machines) is split in the roadmap into lettered sub-tasks, each with *Done when* and
  *Needs*; commit the split on the branch and take the first.

Say in one line which task you take and why it is next.

## 3. Plan

Write `docs/helmholtz/tasks/<ID>.md`:

```
# <ID> <title>

Started <YYYY-MM-DD> on `tbetcke/h-<id>`. Roadmap: <the line>. Proposal: §10 <ID>; §…

## Goal
<one sentence>

## Acceptance (proposal §10, verbatim)
<the criterion>

## Read first
<files, sections, code>

## Changes
<crates and files; public API; tables; conventions (none, or a proposal for sign-off)>

## Tests that prove it
<each against its trusted path (fixtures, identities, nd-fmm-ref, the direct sum, FMM3D
fixtures, the one-rank run), with the tolerance or bit identity, threads and ranks>

## Checks
<`/verify`, and this task's own commands>

## Machine runs
<machine, command, what it decides; or none>

## Constraints and risks
<the constraints of proposal §2 that bind; the effect on Laplace>

## Pending runs
<filled in step 6>

## Open questions
<filled when they arise>
```

Two rules for "Tests that prove it" and "Machine runs" (from the H6 gate,
`docs/helmholtz/gates/H6.md` §7):
- **A wider scope widens the verification set.** If the task lifts a bound, widens a default
  or moves a rule's range (a tolerance, a degree, a κ limit, a machine class), list the cases
  that earlier sign-offs or reports left out for the old scope ("left to Rotation", "not
  run", "excluded"; search the progress entries and reports of the tasks it builds on), and
  put them in the plan's runs, or record why each is not needed. CH6.1e's sign-off left the
  large cases out at 10⁻⁸ because Rotation ran there; CH6.1g and CH6.1h put waves there and
  did not bring them back, and the gate had to.
- **Pair evidence and FMM evidence, both.** A numerical rule (a quadrature, a precision, a
  degree, a count or a reach) changes only on both measures: the worst case per pair or per
  domain over everything the rule covers, and the FMM's errors on the cases. A study on one
  measure may recommend; the other confirms before the PR asks to adopt it. CH6.1f's pair
  chain recommended iprec 2 at 10⁻⁷ and the FMM refuted it (CH6.1g); CH6.2e's FMM held at
  reach 4.5 and the worst pair refuted it (CH6.2g).

Stop and ask, with the plan, before going on if the plan:
- changes code on Laplace's paths (constraint 1 of proposal §2: the host drivers, tables and
  layouts, `nd-fmm-simd`, `nd-fmm-kernels`' Laplace kernels, `nd-octree`, `nd-fmm-plan`) or a
  Laplace default;
- changes `docs/CONVENTIONS.md` (only CH1.1 does, and only as a proposal for sign-off);
- adds or moves a dependency or a pin;
- changes `nd-fmm-plan`'s interface;
- departs from the proposal's acceptance criterion or from a signed-off decision;
- needs a larger Kathleen job than `tools/kathleen/README.md` allows.

Otherwise go on without asking.

## 4. Branch

`git switch -c tbetcke/h-<id>` from `main`, the ID lower-cased with dots as dashes
(`tbetcke/h-chr-2`, `tbetcke/h-ch3-1a`). Commit the plan, and any decision or split, as
`Helmholtz <ID>: plan`.

## 5. Implement

- Tests first: the plan's tests, failing for the right reason; then the code until they
  pass. Never weaken, skip or delete a test to get green; if one seems wrong, say why and ask.
- Root `CLAUDE.md`'s working rules: `cargo fmt --all` after every Rust edit; numeric code
  generic over `T: RealScalar`; no allocation in hot loops; `unsafe` only where allowed;
  every public item documented; crate `CLAUDE.md` tables updated.
- Small steps that build, each committed as `Helmholtz <ID>: <what>`.
- Spikes (H0) go in `spikes/<name>/` with a `SPIKE_REPORT.md`; S9 has a `[workspace]` table
  of its own so nothing enters the root lock file.
- Measurements: release builds, the BLAS thread variables at 1, the machine and ranks ×
  threads named (on Kathleen also the nodes, binding and job id), no timing asserted in a
  test; the report at the path the plan names (`fmm-validate/results/helmholtz-<id>.md`
  unless the task says otherwise).

## 6. Verify

Run `/verify` with this task's plan. Machine runs go through `tools/`: locust
`tools/gh200/sync.sh`, then `tools/gh200/remote.sh '<command>'`; Kathleen
`tools/kathleen/sync.sh`, then `tools/kathleen/submit.sh <job>` (never on the login node);
Metal on the M6 outside the sandbox. Ask the user before any timing run on the M6: they
run other jobs on it. Run long jobs in the background and keep working.
A run that has not finished when the rest is done goes under "Pending runs" with machine,
command, job id and what it decides. Never report an unrun check as passing.

## 7. Self-review

Give a subagent the plan file, the proposal's §10 row and §2, and the branch name, not your
summary. It checks that every acceptance item has a test or a recorded run that shows it,
that the change stays in scope, that constraint 1 holds where Laplace is touched, and the
rules of root `CLAUDE.md`. Fix what blocks; list the rest in the PR text.

## 8. Record

On the branch:
- tick the task in `docs/helmholtz/roadmap.md`, adding the report's path, and cut its line to
  two or three lines: the result, the report or record, the sign-offs in the user's words.
  The plan's detail lives in the record, the history in the progress entry; a ticked line
  is not a log (the H6 gate condensed H3's and H6's);
- rewrite the roadmap's status line to the current state, one short paragraph: the date,
  the milestones closed, the task just done, what comes next. Rewrite it, do not append;
- give every follow-up that a sign-off or the work creates ("a follow-up before H4", "left
  to CH4.3") a roadmap line in the same PR: a new task with *Done when* and *Needs*, or a
  sentence in the task it belongs to. The progress entry's follow-ups name that line;
- append to `docs/helmholtz/progress.md`, using its template;
- where the work shows the proposal wrong, do not edit it silently: put the correction in
  the PR as a question for sign-off;
- when a signed-off answer amends the proposal, edit it in the same PR with an inline
  "(amended at <ID>)" and a line in its Appendix A: "Amended at <ID>, <YYYY-MM-DD>", the
  report, the PR and the user's words, and the sections changed. CH0.4, CH6.2h, CH6.1g and
  CH6.1h amended it inline only, and the H6 gate added their lines.

Commit as `Helmholtz <ID>: roadmap and progress`.

## 9. Push and the PR

`git push -u origin <branch>`. Write `target/pr-<id>.md` (ignored by git): the title
`Helmholtz <ID>: <title>`; a summary; the acceptance items, each with the test or run behind
it, ticked; the checks with their commands and results; the pending runs; the questions for
sign-off, numbered; the effect on Laplace; the files and crates touched; and a last line
`Part of #<n>`, the milestone's issue (`python3 tools/roadmap/sync_issues.py --issue-of
<ID>`; tools/roadmap/README.md), which ticks the task when the PR is merged.
`gh pr create --base main --title "…" --body-file target/pr-<id>.md`, with `--draft` while a
run is pending or a sign-off or open question blocks. An existing PR is updated with
`gh pr edit <n> --body-file …`.

## 10. Report and next

A few lines: the task and its PR (ready, or draft and why); checks run and pending;
questions waiting for the user; the open task PRs and their states; the next runnable task.
If every task of a milestone is ticked on `main` and its gate line is open, suggest
`/milestone-gate <milestone>`.
