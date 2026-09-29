# Phase 0 / T2 — conventions (C0.1)

docs/CONVENTIONS.md is already committed as a draft. This task wires it into the code
and prepares it for human sign-off.

Read first: docs/CONVENTIONS.md (all).

Do:
- In nd-fmm-math add `pub const CONVENTION_VERSION: u32 = 1;` with a doc comment
  linking to docs/CONVENTIONS.md.
- Add a crate-level doc comment summarising the conventions and pointing to the file.
- Read the file critically and list, in the PR description, anything ambiguous or
  inconsistent (for example an index, a sign, a range). Do not fix formulas yourself.

Must pass: cargo test -p nd-fmm-math; cargo doc -p nd-fmm-math without warnings.

Do not: change any formula or convention. If something looks wrong, stop and report it.
