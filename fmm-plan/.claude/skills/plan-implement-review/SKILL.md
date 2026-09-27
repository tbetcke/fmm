---
name: plan-implement-review
description: Plans a non-trivial feature with deep reasoning, then hands off implementation to Opus and correctness/fidelity review to dedicated reviewer subagents. Use for changes complex enough to warrant a separate plan-implement-review pipeline rather than direct edits.
argument-hint: [task description]
disable-model-invocation: true
allowed-tools: Write, Read, Grep, Glob
---

Task: $ARGUMENTS

## Plan

Research what's needed for the task above and propose a plan directly in
this conversation. Do not write or edit any code yet.

Wait for the user's explicit approval before continuing to the next
section.

## Once approved

1. Write the full plan to `PLAN.md` before doing anything else. The
   subagents below start with fresh context and won't see this
   conversation — `PLAN.md` is the only thing they'll have.

2. Delegate implementation to `@implementer`. Do not implement this
   yourself in this conversation, even though you're capable of it.

3. Once `@implementer` reports back, delegate to `@adversarial-reviewer`
   for correctness review. Point it at `PLAN.md` and the changed files.
   Do not use any other reviewer for this step.

4. If `@adversarial-reviewer` finds critical issues, send them back to
   `@implementer` to fix, then re-run `@adversarial-reviewer`.

5. Once correctness review passes, delegate to `@plan-fidelity-reviewer`
   to confirm the implementation matches the plan's intent, not just
   its letter.

Report back to the user only once all of that is done, with a summary
of what each stage found — not a play-by-play of each subagent's output.

