# Phase 6 / T8 — dipole sources (C6.4)

The FMM evaluates the field of point charges (monopoles); dipoles were planned as a later
extension (laplace-fmm-plan §1: "charges (monopoles) first, dipoles as a later extension";
§7, C6.4: "Dipole sources and other source types, matches direct sum"). Boundary integral
and molecular applications need double-layer (dipole) sources, alone or with charges.
Only the source-side operators change: P2M and P2L (a dipole's multipole and local
expansions), P2P (the dipole kernel and its gradient), the direct sum, and the
redistribution of a dipole vector per source. The far field (M2M, M2L, L2L), the targets'
L2P and M2P, the lists and the exchanges of multipoles do not change. This task adds a
source-type interface and dipoles through the whole path, as `docs/design/optimisation.md`
§7 designs it.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md ("CONVENTIONS.md wins"; convention changes are proposed in the PR, never
  made in code), docs/CONVENTIONS.md (the solid harmonics, the expansions, scaling: §3.x),
  fmm-math/CLAUDE.md, fmm-ref/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-simd/CLAUDE.md,
  fmm-kernels/CLAUDE.md;
- docs/design/optimisation.md §7;
- docs/design/laplace-fmm-plan.md §2.1–§2.2 (the separation identity, the expansions);
- the code: `nd_fmm_ref::{direct, leaf, p2p}` (the reference operators and the direct
  sum), `nd_fmm_math` (the harmonics and their gradients), `nd_fmm_exec::operator` (P2M,
  P2L, P2P), `nd_fmm_simd`'s kernels, `nd_fmm_kernels::{leaf, p2p}`, `nd_fmm_exec::fmm`
  (`build`, `evaluate`, the source store and the forward of the charges).

Do:
- **Conventions**: the dipole's expansion coefficients (the gradient of the charge
  expansion with respect to the source position) in the conventions' normalisation; if
  CONVENTIONS.md lacks a section, propose its text in the PR and stop for sign-off before
  code that depends on it (`CONVENTION_VERSION` follows the file).
- **Reference**: `nd-fmm-ref` gains the dipole direct sum (φ and ∇φ) and the dipole P2M,
  P2L and P2P reference operators, checked against finite differences of the charge
  operators and against mpmath fixtures (tools/fixtures/, regenerated only there).
- **The source-type interface** in `nd-fmm-exec`: as T1 signs it off (for example, a
  `Sources` value that holds charges, dipole moments, or both per point, passed to
  `evaluate`); the charge-only path and its bits unchanged.
- **Host operators**: dipole P2M, P2L and P2P (scalar reference first; `nd-fmm-simd`
  kernels if T1 finds them worth it, `unsafe` only in `arch`); the redistribution
  forwards three (or four) values per source.
- **Device operators**: dipole P2M, P2L and P2P kernels in `nd-fmm-kernels`, or the host
  fallback for those kinds on the device (`host_fallback`), as T1 decides.
- **Measurements**: accuracy against the direct sum at the workloads and p = 3, 8, 12;
  the cost of a dipole evaluation against a charge evaluation. Report in the Phase 6
  results file.

Tests that define done:
- The reference dipole operators against finite differences and fixtures (C1-style
  tolerances, stated).
- The dipole FMM, and the mixed charges-and-dipoles FMM, against the dipole direct sum
  within the Phase 3 error bounds at the C3.2 cube and the C3.3 Plummer sphere, f32 and
  f64, at 1, 2 and 4 ranks, and with every kind on the host fallback bit for bit the host
  path.
- The charge-only path bit for bit unchanged.

Must pass: the root checks, the stricter workspace checks, `cargo test -p nd-fmm-ref` and
`-p nd-fmm-math`, `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks, and the
`nd-fmm-simd`/`nd-fmm-kernels` checks for whichever crates change.

Do not: change a convention in code; add other kernels (Helmholtz, Yukawa); assert a
timing.
