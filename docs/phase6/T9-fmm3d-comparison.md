# Phase 6 / T9 — the FMM3D comparison (C6.8)

The plan compares against one external code at matched accuracy: FMM3D (analytic
Laplace), decided on 2026-10-04 as the only baseline, in a later phase
(docs/phase4/README.md, decision 9; laplace-fmm-plan §8.3). ExaFMM-t and kifmm-rs are
dropped. This task builds FMM3D outside the workspace, runs both codes on the same
problems at matched accuracy on one node, and reports, as
`docs/design/optimisation.md` §8 designs the method.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md ("New dependencies ... only through [workspace.dependencies]": FMM3D is
  not one), fmm-bench/CLAUDE.md, fmm-validate/CLAUDE.md;
- docs/design/optimisation.md §8 and decision 5's answer;
- docs/design/laplace-fmm-plan.md §8.2 (error metrics), §8.3;
- FMM3D's documentation (its `lfmm3d` interfaces, the precision parameter `eps`, the
  normalisation of its kernel: compare its 1/r against our 1/(4π r), CONVENTIONS §3.1);
- `nd-fmm-bench` (the one-command benchmark: `tools/bench/run.sh`), the Phase 5 T10
  `scaling` harness.

Do:
- **The build**: a script `tools/fmm3d/build.sh` (and a README) that fetches FMM3D at a
  pinned release, builds it with the machine's Fortran compiler (gfortran from spack on
  locust, the `compilers/gcc` module on Kathleen if it is usable, Homebrew's on the M3
  Max) with OpenMP,
  and installs it under a directory outside the repository (on locust under
  `/data/ucahtbe`; on Kathleen under the layout of tools/kathleen/). Nothing is
  committed beyond the scripts.
- **The driver**: a small program outside the workspace (Fortran, C or Python through
  FMM3D's own Python bindings; not a crate) that reads the same points and charges as
  `nd-fmm-bench` (written to a binary file by a `--write-problem` option, or regenerated
  from the seed by index: Phase 5 T10's `Workload`), evaluates φ and ∇φ, and writes the
  output for the error check.
- **Matched accuracy**: for each of our p (3, 6, 8, 12), the FMM3D `eps` whose error
  against the same direct sum (8 vectors, 1,000 sampled targets, f64) is closest to ours;
  report both errors. Precomputation (our table build, FMM3D's setup) reported apart from
  the evaluation.
- **Runs**: one node per machine where FMM3D builds, f64, N = 10⁵, 10⁶ and 10⁷, the cube
  and the Plummer sphere, one thread and all cores (our ranks × threads chosen as Phase
  5N's rule recommends; FMM3D's OpenMP threads), the best host configuration of ours
  (after T2). The GPU path of ours reported beside it, labelled.
- **The report**: `fmm-validate/results/phase6-fmm3d.md` (or a section of the Phase 6
  results file), stating versions, compilers, flags, threads, and that FMM3D's numbers
  are measured with its defaults otherwise.

Tests that define done:
- The comparison reproducible from the scripts on at least one machine (gate), with the
  errors of both codes against the same direct sum.

Must pass: the root checks (only `tools/` and docs change, plus an `nd-fmm-bench` option
if added, with its checks).

Do not: link FMM3D into a crate; commit its sources, builds or outputs; compare at
unmatched accuracy.
