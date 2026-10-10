# Phase 6 / T10 — plane-wave M2L, a spike (C6.5, optional)

Plane-wave (exponential) M2L diagonalises the translation: a multipole is converted to
plane waves in a few directions, translated by diagonal multiplication and converted back
(laplace-fmm-plan §3.3), at O(p³) or better per box pair with merge-and-shift across the
interaction list. It is what the fastest analytic Laplace codes use at higher precision.
C6.5 is optional: "beats C4.5/C4.6 on some configuration, else dropped" (laplace-fmm-plan
§7). After T2 (the host batched M2L) and T3 (compressed M2L) the bar is higher. This task
is a spike in `spikes/`, with a stop rule, as `docs/design/optimisation.md` §8 frames it.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md ("spikes/ is excluded from default members and never built in CI");
- docs/design/laplace-fmm-plan.md §3.3, §4 (why it was not chosen first);
- docs/design/optimisation.md §8 (the question and the stop rule);
- the results of T2 and T3 (the M2L times to beat, per p and machine);
- the literature laplace-fmm-plan §3.3 cites for the exponential expansion
  (Greengard and Rokhlin 1997; Cheng, Greengard and Rokhlin 1999); verify the details
  before citing.

Do:
- **The question**: at which p, precision and machine (host per core; one GPU) would a
  plane-wave M2L beat T2's batched dense M2L and T3's compressed M2L on the uniform cube
  at N = 10⁶, counting the conversions and the six directional lists (up, down, north,
  south, east, west) with merge-and-shift? Answer first as a *model* (flops and memory
  traffic per box, with the measured throughputs of T2/T3).
- **If the model shows a configuration where it can win by at least the margin T1 sets**:
  a spike in `spikes/plane-wave/` (its own crate, excluded from default members) that
  implements the directional conversions and translations for one level on the host, f64,
  checked against `nd_fmm_ref`'s direct M2L to the plane-wave quadrature's accuracy, and
  times it against T2 and T3 on the same level.
- **The stop rule**: if neither the model nor the spike shows a win on any configuration
  of the workloads, write the spike report and recommend dropping C6.5 (decision 6).
- **SPIKE_REPORT.md** in the spike's directory: the model, the measurements, the
  recommendation.

Tests that define done:
- The spike's level M2L against the direct M2L within its stated quadrature error (if
  the spike is built); the report with a recommendation either way.

Must pass: the root checks (the spike is not a default member; run its own tests by hand
and report them).

Do not: add the plane-wave M2L to `nd-fmm-exec` or `nd-fmm-kernels` in this task (an
adoption would be its own task after decision 6); add a dependency to the workspace.
