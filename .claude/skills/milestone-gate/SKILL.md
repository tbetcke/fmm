---
name: milestone-gate
description: Checks the gate of a Helmholtz milestone (HR, H0 to H8) against docs/helmholtz/roadmap.md and the proposal, writes the gate report, and closes the milestone on the user's sign-off. Use when /next-task says a milestone is complete, or when the user asks to close one.
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

## 3. The report

Write `docs/helmholtz/gates/<milestone>.md`: the date; each gate condition with its
evidence (PRs, reports, CI runs); what failed or remains; the measured numbers against the
proposal; proposed corrections to the proposal, as questions for sign-off rather than edits;
recommendations for the next milestones; and a short retrospective on the loop itself (what
to change in `/next-task`, `/verify` or the roadmap).

## 4. Close

Ask the user to sign off the gate and the proposed corrections. On their answer, on the
branch `tbetcke/h-<milestone>-gate`: tick the gate line; record the sign-offs in the
decision table in the user's words; apply the signed-off corrections to the proposal with a
line in its Appendix A; update the roadmap's status line; append a progress entry. Commit as
`Helmholtz <milestone>: gate`, push, and open the PR, its text ending with `Part of #<n>`,
the milestone's issue (`python3 tools/roadmap/sync_issues.py --issue-of <milestone>`); the
issue closes when the PR is merged (tools/roadmap/README.md).
