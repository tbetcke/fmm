# Phase 0: conventions and math core

Phase 0 fixes every mathematical convention in one reviewed file
(`docs/CONVENTIONS.md`) and builds `nd-fmm-math` against high-precision fixtures, so
that no later phase ever has to guess a sign. It ends when the identities hold to about
1e-14 in f64 and a one-day CubeCL spike has measured GEMM throughput: f32 on the
Metal GPU and f64 on the CubeCL CPU runtime (Metal has no f64).

Companion documents: [docs/design/laplace-fmm-plan.md](../design/laplace-fmm-plan.md)
(components C0.1–C0.3) and [docs/design/workspace-structure.md](../design/workspace-structure.md)
(crate layout and naming).

## Scope

In scope:
- `docs/CONVENTIONS.md`: single source of truth for basis functions, phases, storage and scaling.
- `nd-fmm-math`: scalar trait, index layout, regular and irregular solid harmonics with
  gradients, rotation matrices.
- `tools/fixtures/`: high-precision fixture generator whose output is committed.
- `spikes/cubecl-gemm/`: throwaway benchmark that decides whether dense GEMM M2L is viable in f64.
- Workspace scaffolding: root `CLAUDE.md`, `[workspace.dependencies]`, CI still green.

Out of scope: translation operators (Phase 1), tables (Phase 2), anything touching the
octree or `nd-fmm-plan`.

## Exit gate
- Every acceptance test in the task briefs passes in CI.
- `docs/CONVENTIONS.md` reviewed and signed off by hand; `CONVENTION_VERSION = 1` frozen.
- The spike report states measured f32 (Metal) and f64 (CPU runtime) GEMM throughput,
  recommends the default M2L strategy in f32, and gives a provisional f64
  recommendation for data-centre CUDA cards from a roofline model.

## Tasks

One pull request each. T1–T5 run in order; T6 depends only on T1.

| Task | Brief | Delivers | Component | Depends on |
| --- | --- | --- | --- | --- |
| T1 | [T1-scaffold.md](T1-scaffold.md) | workspace dependencies, `fmm-math` crate skeleton, CI green | — | none |
| T2 | [T2-conventions.md](T2-conventions.md) | review of `docs/CONVENTIONS.md`, `CONVENTION_VERSION` constant | C0.1 | T1 |
| T3 | [T3-fixtures.md](T3-fixtures.md) | `tools/fixtures/` generator (Python, mpmath) and committed fixtures | supports C0.2, C0.3 | T2 |
| T4 | [T4-harmonics.md](T4-harmonics.md) | `RealScalar`, `Layout`, solid harmonics and gradients | C0.2 | T3 |
| T5 | [T5-rotations.md](T5-rotations.md) | per-degree rotation blocks | C0.3 | T4 |
| T6 | [T6-cubecl-gemm-spike.md](T6-cubecl-gemm-spike.md) | `spikes/cubecl-gemm/` and `SPIKE_REPORT.md` | de-risks Phase 4 | T1 |

Review T2 yourself before T3 starts: every later test encodes its choices, so a
convention changed afterwards means regenerating fixtures and revisiting code.

## How to run a task with Claude Code

In the repository root, start `claude` and say:
"Read docs/phase0/T<k>-<name>.md and do that task." Review and merge before the next.

## Exit checklist
- [x] T1 merged: workspace dependencies, `fmm-math` skeleton, CI green
- [x] T2 merged: `CONVENTION_VERSION = 1` in `nd-fmm-math`
- [x] `docs/CONVENTIONS.md` reviewed and signed off
- [ ] T3 merged: fixtures regenerate byte-identically, under 3 MB
- [ ] T4 merged: fixture, separation, addition-theorem and harmonicity tests pass in f64 and f32
- [ ] T5 merged: rotation, homomorphism, orthogonality and z-rotation tests pass up to p = 30
- [ ] T6 report written: f32 GEMM throughput on Metal, f64 on the CPU runtime, default M2L strategy recommended (f64 provisional)
- [ ] Design document updated where Phase 0 refined it (local-coefficient scaling, Section 2.4; spike outcome, Section 4)
