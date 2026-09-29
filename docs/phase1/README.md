# Phase 1: CPU reference operators

Phase 1 builds `nd-fmm-ref`, the f64 oracle that every later fast path is checked
against. It contains the leaf operators (P2M, P2L, L2P, M2P), direct O(p⁴) and
rotation-based O(p³) translations (M2M, L2L, M2L), and P2P with a direct-sum oracle.
The translation formulas are derived, checked numerically and added to
`docs/CONVENTIONS.md` before any Rust implements them. That retires the last open sign
risk of the design document (Section 9.1) on paper first. The phase ends when the
rotation operators agree with the direct ones to 1e-13 and every operator chain agrees
with the direct sum within its truncation bound.

Companion documents: [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md)
(Sections 2.2–2.5, 3.1–3.2, 8.1; components C1.1–C1.4 in Section 7) and
[docs/design/workspace-structure.md](../design/workspace-structure.md) (Section 3,
`nd-fmm-ref` and `nd-fmm-validate`; Section 5, crate template).

## Scope

In scope:
- `docs/CONVENTIONS.md`: a new section §3.11 with the translation operators, and a
  wider tested range in §3.9 (M2L needs irregular harmonics up to degree 2p).
- `tools/fixtures/`: a numerical check of the §3.11 formulas and an irregular-harmonics
  fixture set C up to degree 40.
- `nd-fmm-ref`: expansion frames, a caller-owned workspace, leaf operators, direct and
  rotation-based translations, P2P and the direct sum. Generic over `T: RealScalar`;
  f64 is the reference, f32 is tested for comparison.
- `nd-fmm-validate`: error metrics, point distributions, a single-translation accuracy
  report and a direct-versus-rotation timing report.

Out of scope:
- Operator matrices, symmetry classes and caching (Phase 2).
- Box geometry from Morton keys, and the box-geometry conventions (child index
  4x + 2y + z, V-list offset sign). These are proposed for `docs/CONVENTIONS.md`
  before Phase 2 (workspace-structure §6, "Still open").
- Anything touching `nd-octree`, `nd-fmm-plan` or MPI; CubeCL; parallelism (rayon)
  and performance work beyond the O(p³) cost of the rotation operators.

## Design decisions for this phase

These hold for every task, so that no task decides them on its own:

- **Frames, not shift vectors.** Every operator takes the frame of its input and of its
  output expansion: a `Frame { centre, radius }`, where the radius is the scaling
  radius r of CONVENTIONS §3.7. No shift-sign convention leaks into the API, and
  §3.11 states the shift inside each formula. Phase 2 obtains the level-independent
  tables by fixing the radii (r' = 2r for M2M and L2L, r' = r for M2L) and the offsets.
- **Scaled coefficients only** (§3.7). There is no unscaled variant.
- **Accumulate.** Every operator adds (+=) into its output, as `nd-fmm-plan`'s
  `FmmOperator` does. No operator applies 1/(4π) (§3.1).
- **No allocation per call.** Temporaries come from a caller-owned `Workspace` built for
  a maximum p. Operators do not allocate.
- **One module per method.** Leaf operators live in `leaf` and are shared. Translations
  live in `direct` and `rotation` with identical signatures, so that tests and later
  crates can swap them.
- **Error measures.** These are Phase 0's lessons (design Sections 7 and 8.1), and every
  test names the one it uses:
  - Coefficient vectors are compared per degree, in the orthonormal basis of §3.8.
    Slot m is weighted by Nₘ for multipole-type data (conjugates of regular harmonics)
    and by Nₘ/Sₘ for local-type data (conjugates of irregular harmonics).
  - Potentials are compared relative to the sum of term magnitudes wherever terms can
    cancel.
  - A truncated result is compared with a reference truncated at the same degree. The
    truncation error is checked separately, against its bound.

## Exit gate
- Every acceptance test in the task briefs passes in CI. `nd-fmm-ref` and
  `nd-fmm-validate` are default members and MPI-free.
- CONVENTIONS §3.11 and the §3.9 change are reviewed and signed off by hand before T5
  starts.
- Rotation M2M, L2L and M2L agree with the direct operators to 1e-13 (per degree,
  weighted as above) for p ≤ 20 in f64. This is the design document's Phase 1 gate.
- M2M after P2M equals P2M at the parent, and L2L translates local polynomials exactly,
  both to 1e-13. M2L chains stay within their truncation bound against the direct sum.
- Measured errors, the single-translation accuracy table and the timing report are in
  the PRs and copied into the design document.

## Tasks

One pull request each. T2 does not need `fmm-ref` (it touches only the docs,
`tools/fixtures/` and one `nd-fmm-math` test), so it can start at once, alongside T1.
T3 and T4 need T1 and can run in parallel. T5 needs T2, T3 and T4; T6 needs T5; T7
needs T6.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-scaffold.md](T1-scaffold.md) | `fmm-ref` crate skeleton, `Frame`, workspace entries, phase pointer in root `CLAUDE.md` | — | none |
| T2 | [T2-translation-conventions.md](T2-translation-conventions.md) | CONVENTIONS §3.11 draft, §3.9 range, `check_translations.py`, fixture set C | prerequisite of C1.2, C1.3 | none |
| T3 | [T3-leaf-operators.md](T3-leaf-operators.md) | `Workspace`, P2M, P2L, L2P, M2P with gradients | C1.1 | T1 |
| T4 | [T4-p2p.md](T4-p2p.md) | `p2p` and `direct_sum` with gradients | C1.4 | T1 |
| T5 | [T5-direct-translations.md](T5-direct-translations.md) | `direct::{m2m, l2l, m2l}`, O(p⁴) | C1.2 | T2 (signed off), T3, T4 |
| T6 | [T6-rotation-translations.md](T6-rotation-translations.md) | `rotation::{m2m, l2l, m2l}`, O(p³) | C1.3 | T5 |
| T7 | [T7-validate.md](T7-validate.md) | `fmm-validate` crate, accuracy and timing reports | supports C1.1–C1.3; prediction for C3.2 | T6 |

Review T2 yourself before T5 starts. T5 and T6 encode §3.11 in tests, so a formula
changed afterwards means revisiting both.

T3 and T4 both add a module to `fmm-ref/src/lib.rs`. Merge one, then rebase the other;
the conflict is one line.

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase1/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [ ] T1 merged: `nd-fmm-ref` skeleton, CI green, root `CLAUDE.md` points to Phase 1
- [ ] T2 merged: `check_translations.py` passes, fixture set C committed, fixtures regenerate byte-identically
- [ ] CONVENTIONS §3.11 and the §3.9 change reviewed and signed off; `CONVENTION_VERSION` decision recorded
- [ ] T3 merged: leaf operators match the same-degree Legendre series and stay within the truncation bound, p ≤ 30
- [ ] T4 merged: P2P and direct sum exact against brute force, coincident points excluded
- [ ] T5 merged: M2M and L2L exactness and composition to 1e-13, p ≤ 30; M2L within its bound, p ≤ 20
- [ ] T6 merged: rotation operators agree with direct to 1e-13, p ≤ 20
- [ ] T7 merged: accuracy table and timing report in the PR
- [ ] Design document updated: Section 2.3 cites §3.11; Section 7 Phase 1 status and measured errors; Section 9.1 translation risk retired; workspace-structure §3.1 matches the built `nd-fmm-ref` surface
