# Draft: CONVENTIONS §3.15, the Helmholtz family

Superseded by `docs/CONVENTIONS.md` §3.15 (Helmholtz CH1.1, 2026-10-29,
`CONVENTION_VERSION = 3`). Cite the section, never this file. The draft was written
in CH1.4 (2026-10-28) and checked there against FMM3D 2.1.0 on two points
(`fmm-validate/results/helmholtz-ch1-4.md`); its full text is in the history of this file
(commit 0c30b72).

What §3.15 decides that the draft left open, or states differently:

- **Where 4π ik goes** (draft item 1): on the crossing operators M2L (in its (S|R) step),
  M2P and P2L; P2P computes e^{ikr}/r (CH1.1, question 1).
- **The stored coefficients** (draft item 3): besides the power scale s_lⁿ they carry a
  per-box damping factor, M̂ = e^{−β_l} M / s_lⁿ and L̂ = e^{β_l} L s_lⁿ with
  β_l = Im(k) a_l, so every leaf operator applies only non-positive exponents; the M2M
  and L2L tables need no damping factor, and M2L a scalar e^{2β_l − Im(k)|t|} ≤ 1 per
  offset (CH1.1, question 2).
- **Yukawa's locals**: L carries the factor 4π ik, so L^Y = iⁿ L (the draft's −iⁿ L was
  for L without that factor; the two agree).
- **Coaxial generators**: (R|R) by the recursion where stable, the outer-sphere
  projection for (S|R) and (S|S) only (S7's sign-off, 2026-10-29).
- **Signatures** (draft item 7): reserved, fixed by their tasks with a version bump.
- **Dipoles**: not in the first release (D5, signed off 2026-10-29).
