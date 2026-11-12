---
name: milestone-gate
description: Checks the gate of a Helmholtz milestone (HR, H0 to H9) against docs/helmholtz/roadmap.md and the proposal, runs the two project review passes (subagents on Fable at medium effort), writes the gate report, and closes the milestone on the user's sign-off. Use when /next-task says a milestone is complete, or when the user asks to close one.
---

# Milestone gate

## Permissions

As `/next-task`: invoking this skill is the request to create the branch
`tbetcke/h-<milestone>-gate`, commit to it, push it and open its PR. Merging and pushing to
`main` need a yes in the current message.

## 1. Which milestone

The one named, else the first whose tasks are all ticked on `main` while its gate line
(`<milestone>.G`) is open. If a task is still open, list it and stop.

## 2. Evidence

- Per task: its merged PR (`gh pr list --state merged --search "Helmholtz <ID>"`), its
  report, its progress entry, and no pending run left.
- `main`'s latest CI runs green: `gh run list --branch main --limit 5`.
- The milestone's gate in the roadmap and the matching gate components of proposal §10
  (CH2.5, CH3.6, CH4.4 and CH4.5, CH5.3, CH7.6) met, with numbers.
- The decisions the milestone prepares put to the user, and signed off or open.
- Proposal §9's validation items that apply; constraint 1 kept wherever Laplace was touched
  (the "Laplace" line of each progress entry).
- Measured numbers set against the proposal's expectations (for example §7's memory model,
  §3.5's p tables), and the corrections they imply.
- The defaults as they stand at the gate, on the cases they now cover: for every case a
  sign-off or report left out under an earlier scope ("left to Rotation", "not run"), check
  that a later task ran it once a default reached it. If none did, run it on the gate branch
  before closing, or ask (the H6 gate's §2 and §8: Kathleen job 633000).
- Every inline "amended at <ID>" of the proposal has its Appendix A line.
- `main`'s CI jobs over the milestone: failures, cancellations (the cause from the job's
  log) and each job's time against its `timeout-minutes`.

## 3. The two review passes

Every gate repeats the project review of `docs/reviews/` (the first:
`docs/reviews/2026-11-12-H6.md`; the user, 2026-11-12: "revisited at every milestone gate").
The passes are independent reads by subagents; the session that drives the gate and the
implementation does not make them itself, it checks and triages what they return.

- **Start** two subagents in one message, so they run side by side, and no more than two:
  one with `review-record.md` (the mathematics, the algorithmic decisions, the benchmarks),
  one with `review-architecture.md` (the code's structure), both files beside this one.
  Each on the model **Fable** at **medium** effort (the user, 2026-11-12), whatever model
  drives the gate: the Agent tool with `model: "fable"` and `effort: "medium"`, or the
  harness's delegation tool where it asks for one for a model or effort other than the
  driver's. If Fable is not available, say so and ask; do not substitute a model.
- **The prompt** is the brief as written, under a header with: the milestone and what it
  closed; `main`'s commit; the earlier review files, and from the latest one its "Coverage"
  (what it did not read comes first now) and the IDs of its findings in the pass's scope.
  Give neither pass your own conclusions or the gate's evidence of step 2.
- **Read only.** The passes compile nothing, run nothing and change nothing; the driver
  writes every file. Wait for both; do not make the same checks yourself meanwhile.
- **Check.** A subagent's report is a claim, not a result. Check every finding of high or
  medium-high severity, and every one a correction or a new task will rest on, against the
  tree or the document it names, and mark it ✔; a finding that does not hold is dropped,
  with a line saying so. Leave the passes' *suspicions* labelled.
- **Record** `docs/reviews/<YYYY-MM-DD>-<milestone>.md` in the form of the first: the
  commit; per pass what was judged sound and the findings, each with an ID (the letters of
  the first review; cite as "review <date>, <ID>"), a severity and the files; the status of
  every earlier finding (resolved by which PR, open in which task, declined); the triage;
  the coverage of both passes.
- **Triage** each new finding to one of: an "Also" on a task that exists; a new task of a
  milestone; a task of the roadmap's section "Architecture and cleanup (HA)" (CHA.<n>,
  bound to constraint 1); a correction to the proposal or CONVENTIONS; declined, with the
  reason. The triage is a proposal: its choices join the gate's questions (step 5), and the
  roadmap changes only on the user's answers.

## 4. The report

Write `docs/helmholtz/gates/<milestone>.md`: the date; each gate condition with its
evidence (PRs, reports, CI runs); what failed or remains; the measured numbers against the
proposal; proposed corrections to the proposal, as questions for sign-off rather than edits;
the review (step 3: its file, the findings of high and medium-high severity, the earlier
findings still open, the proposed triage); recommendations for the next milestones; and a
short retrospective on the loop itself (what to change in `/next-task`, `/verify` or the
roadmap).

## 5. Close

Ask the user to sign off the gate, the proposed corrections and the review's triage. On
their answer, on the branch `tbetcke/h-<milestone>-gate`: tick the gate line; record the
sign-offs in the decision table in the user's words; apply the signed-off corrections to
the proposal with a line in its Appendix A; apply the signed-off triage to the roadmap
(the "Also" lines, the new tasks, the HA tasks, their place on or off the priority list);
rewrite the roadmap's status line to the current state; condense the
milestone's ticked lines to two or three lines each (the result, the report or record, the
sign-offs), with a note naming the commit that holds them in full (`git show
<sha>:docs/helmholtz/roadmap.md`), and check that `tools/roadmap/sync_issues.py` still parses
the roadmap; if the milestone finishes the priority list, propose the next one; append a
progress entry. Commit as
`Helmholtz <milestone>: gate`, push, and open the PR, its text ending with `Part of #<n>`,
the milestone's issue (`python3 tools/roadmap/sync_issues.py --issue-of <milestone>`); the
issue closes when the PR is merged (tools/roadmap/README.md).
