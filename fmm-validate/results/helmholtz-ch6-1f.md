# Helmholtz CH6.1f: the degree-matched quadrature on the pair chain

Helmholtz task CH6.1f ([docs/helmholtz/tasks/CH6.1f.md](../../docs/helmholtz/tasks/CH6.1f.md)),
split on 2026-11-09 from the roadmap's former CH6.1f ("Quadratures for ε below 10⁻⁶") into this
study and the FMM (CH6.1g); proposal
[docs/design/helmholtz-fmm-proposal.md](../../docs/design/helmholtz-fmm-proposal.md) §10 CH6.1
("errors within about 1.1× Rotation's"), §5.3, §2 (constraints 6 and 9); CH6.1c's degree study
([docs/helmholtz/tasks/CH6.1c.md](../../docs/helmholtz/tasks/CH6.1c.md), "Results") and
CH6.1e's report ([helmholtz-ch6-1e.md](helmholtz-ch6-1e.md) §2, §3, §5). Measured on
2026-11-09 on the M6 (Mac mini, Apple M6, 24 GB; [tools/m6/machine.md](../../tools/m6/machine.md))
from branch `tbetcke/h-ch6-1f` (be75f52 with the fit's face centres, §4, committed with this
report), release build, rustc 1.99.0, f64, one thread, in the sandbox. Errors and counts only: deterministic, nothing timed but the generator's own run
time, for orientation.

## Summary

- **The problem is the plateau's degree, not the tables.** Below 10⁻⁶ the FMM's low-frequency
  levels run at the calibrated plateau, p = 20 at 10⁻⁷ and p = 24 at 10⁻⁸. There FMM3D's
  quadratures, fitted to the kernel, err on the truncated multipole's tail: on the pair chain
  against the direct sum iprec 2 is 1.08–4.1× RCR's error at 10⁻⁷ and 0.67–6.3× at 10⁻⁸,
  iprec 3 up to 15× and 53×. On the levels whose degree comes from the high-frequency branch
  (κ ≥ 12.5 at 10⁻⁷, κ ≥ 25 at 10⁻⁸) both tables already hold at the FMM's own degree
  (0.93–1.00×): that degree resolves their tails.
- **Candidate 4, the resolving degree, holds everywhere with the tables' own waves.** Each
  precision has a degree from which its waves of a degree-p multipole resolve the tail: about
  24 for iprec 2 and 32 for iprec 3. Running a level's M2L at p_w = max(p_l, 24) with iprec 2
  at 10⁻⁷, and at p_w = max(p_l, 32) with iprec 3 at 10⁻⁸, keeps the chain at **0.10–0.81×**
  RCR's error at 10⁻⁷ and **0.03–0.17×** at 10⁻⁸ on the plateau, and 0.93–1.00× above it (§3).
  This holds with the multipole formed as in the FMM, by M2M from children at p, and the local
  taken back to p by L2L. The cost is the degree, not the waves.
- **Candidate 3, the degree-matched quadrature, holds without raising the degree, at 1.6–2.9×
  the waves.** `nd_fmm_ref::helmholtz::plane_wave::generate_for_degree` (new) fits the λ rule to
  the truncated chain itself against RCR at degree p. Generated at ε/20 it keeps the chain at
  **1.000–1.010×** RCR's error on all 26 points. It needs 1.6–2.9× iprec 3's waves, every order
  to p at every node (1.7–3.3× the modes), a contour to Re λ ≈ 37 at p = 24, and about a minute
  to generate per level (§4).
- **Candidate 2, the plain generator at a chosen ε, is erratic** (0.35–5.8× on the plateau,
  non-monotone in ε), as CH6.1c's table already suggested: no ε per degree holds 1.1×.
- **Recommendation for CH6.1g (question 1):** candidate 4 as the resolving degree, a function
  of ε and p_l only (constraint 9), with FMM3D's tables. Whether its raised degree pays off in
  the FMM against Rotation at p_l is CH6.1g's measurement. Candidate 3 stays in nd-fmm-ref as a
  trusted path and the fallback if the raised degree costs more than the waves save.

## 1. Method

**Measure** (CH6.1c's, [`degree_study`'s](../../fmm-tables/tests/helmholtz/plane_wave_pair.rs)):
the source box's corners, centre and four seeded points with seeded complex charges, Σ|q| = 1;
every third V offset (106 of 316) on level 2 of k̂₀ = 2κ (so κ_2 = κ); at the 27 points
{−½, 0, ½}³ of the target box, the largest |Δ| in box widths, twice the level units' r_l φ̃.
Each candidate's *chain − direct* is compared with *RCR − direct* at the FMM's degree p on the
same path, and the ratio is reported. As in the FMM (`against_direct_children`): P2M into the
eight children at p and M2M to the level at the chain's degree; the chain's (or RCR's) local
L2L to the target's child at p and L2P there. A pilot run with the multipole by P2M at the
chain's degree directly gave the same ratios to three digits.

**Levels.** κ = k w_l ∈ {0.0078, 0.031, 0.078, 0.31, 1.25, 2.5, 6.25, 12.5, 25, 40} and
1.25 + 0.1875i, 0.625 + 1.25i, 10 + 5i: CH6.1e's levels at 10⁻⁷ and 10⁻⁸ (the clustered trees'
deep levels at κ ≤ 0.08, the failures of CH6.1e's scan), up to level 2 of kD = 100 and to 40
(the tables end at Re κ = 16π). p is `DegreeRule::CALIBRATED`'s at ε = 10⁻⁷ and 10⁻⁸: the
plateau 20 and 24 up to κ = 6.25 (and κ = 12.5 at 10⁻⁸), the high branch above (23, 36, 52 at
10⁻⁷; 38, 53 at 10⁻⁸).

**Candidates** (the plan's):

1. FMM3D's iprec 2 and 3 at p (CH6.1c's rows).
2. `generate(κ, ε_g)` at ε_g = 10⁻⁶, 10⁻⁷, 10⁻⁸.
3. `generate_for_degree(κ, ε/20, p)` (new, §4).
4. FMM3D's iprec 2 and 3 at p + 4 and p + 8: the level's M2L at a raised degree between
   children at p.

**Error measure of the fitting.** `generate_for_degree`'s ε is absolute, per unit charge, in box
widths (its module documentation); the study's measure is twice that per Σ|q| = 1. ε/20 puts
the quadrature's part below a tenth of RCR's error on the study's measure.

## 2. Every candidate, summarised

The ratio chain − direct over RCR − direct, its range and median over the levels of each
regime (plateau: p = 20 or 24; high: the high-frequency branch's p), and the waves Σₖ Nₖ over
iprec 3's at the same κ. The full table is the appendix.

| ε | regime | candidate | levels | ratio | median | waves ÷ iprec 3 |
| --- | --- | --- | ---: | --- | ---: | --- |
| 10⁻⁷ | plateau | iprec 2 at p | 10 | 1.079–4.070 | 2.335 | 0.45–0.58 |
| 10⁻⁷ | plateau | iprec 3 at p | 10 | 1.000–14.689 | 9.524 | 1 |
| 10⁻⁷ | plateau | iprec 2 at p + 4 | 10 | **0.105–0.810** | 0.267 | 0.45–0.58 |
| 10⁻⁷ | plateau | iprec 2 at p + 8 | 10 | 0.023–0.628 | 0.218 | 0.45–0.58 |
| 10⁻⁷ | plateau | iprec 3 at p + 4 | 10 | 0.593–8.326 | 3.236 | 1 |
| 10⁻⁷ | plateau | iprec 3 at p + 8 | 10 | 0.145–0.925 | 0.218 | 1 |
| 10⁻⁷ | plateau | generated at 10⁻⁶ | 10 | 0.812–2.560 | 1.120 | 0.60–0.75 |
| 10⁻⁷ | plateau | generated at 10⁻⁷ | 10 | 1.184–2.933 | 1.440 | 0.76–0.94 |
| 10⁻⁷ | plateau | generated at 10⁻⁸ | 10 | 1.881–5.054 | 4.547 | 1.03–1.18 |
| 10⁻⁷ | plateau | **matched at ε/20** | 10 | **1.000–1.005** | 1.001 | 1.60–2.24 |
| 10⁻⁷ | high | iprec 2 at p | 3 | 0.975–1.003 | 0.992 | 0.66–0.80 |
| 10⁻⁷ | high | iprec 3 at p | 3 | 0.957–0.999 | 0.998 | 1 |
| 10⁻⁷ | high | iprec 2 at p + 4 | 3 | 0.010–0.033 | 0.033 | 0.66–0.80 |
| 10⁻⁷ | high | matched at ε/20 | 3 | 1.000–1.000 | 1.000 | 1.67–1.82 |
| 10⁻⁸ | plateau | iprec 2 at p | 11 | 0.665–6.309 | 1.608 | 0.45–0.66 |
| 10⁻⁸ | plateau | iprec 3 at p | 11 | 0.999–53.113 | 19.765 | 1 |
| 10⁻⁸ | plateau | iprec 2 at p + 4 | 11 | 0.051–4.006 | 1.182 | 0.45–0.66 |
| 10⁻⁸ | plateau | iprec 2 at p + 8 | 11 | 0.006–4.006 | 1.182 | 0.45–0.66 |
| 10⁻⁸ | plateau | iprec 3 at p + 4 | 11 | 0.216–5.980 | 1.373 | 1 |
| 10⁻⁸ | plateau | iprec 3 at p + 8 | 11 | **0.010–0.166** | 0.036 | 1 |
| 10⁻⁸ | plateau | generated at 10⁻⁶ | 11 | 1.071–4.463 | 1.960 | 0.60–0.92 |
| 10⁻⁸ | plateau | generated at 10⁻⁷ | 11 | 0.351–3.539 | 0.421 | 0.76–1.07 |
| 10⁻⁸ | plateau | generated at 10⁻⁸ | 11 | 0.763–5.755 | 1.009 | 1.03–1.25 |
| 10⁻⁸ | plateau | **matched at ε/20** | 11 | **1.000–1.010** | 1.003 | 2.03–2.89 |
| 10⁻⁸ | high | iprec 2 at p | 2 | 0.957–0.974 | 0.966 | 0.77–0.80 |
| 10⁻⁸ | high | iprec 3 at p | 2 | 0.925–0.998 | 0.962 | 1 |
| 10⁻⁸ | high | iprec 3 at p + 8 | 2 | 0.000–0.000 | 0.000 | 1 |
| 10⁻⁸ | high | matched at ε/20 | 2 | 1.000–1.000 | 1.000 | 1.84–1.96 |

Reading it:

- At the FMM's degree the coarser table is the better one on the plateau, and neither holds:
  iprec 3 reaches larger λ, where the truncated waves depart further from the kernel's.
- Raising the degree helps each precision only from its own resolving degree: iprec 2 is
  resolved by p + 4 = 24 at 10⁻⁷ and stays there (p + 8 gives little more), but at 10⁻⁸ its
  stated error (about 10⁻⁸) is the FMM's own and it misses (up to 4.0×) at any degree;
  iprec 3 needs p + 8 = 32 and then holds at both ε.
- The plain generator at a fixed ε moves the λ range, not the tail's fit, so its ratio jumps
  with ε (at κ = 1.25, p = 24: 1.98, 0.35, 0.83 at 10⁻⁶, 10⁻⁷, 10⁻⁸) and no ε per degree holds.
- The pair chain's corner sources make this measure harsher than the FMM's relative ℓ2 error:
  CH6.1e saw iprec 3 at p hold within 1.1× on most levels where this table shows 10–50×. The
  ranking of the candidates is what transfers; the FMM's ratios are CH6.1g's.

## 3. Candidate 4: the resolving degree

The rule recommended for CH6.1g: a level with a quadrature runs its exponential M2L at
p_w = max(p_l, d) with

| FMM tolerance ε | precision | d, the resolving degree |
| --- | --- | ---: |
| 10⁻⁶ ≤ ε | iprec 2 (CH6.1e's rule, unchanged) | p_l (no raise) |
| 10⁻⁷ ≤ ε < 10⁻⁶ | iprec 2 | 24 |
| 10⁻⁸ ≤ ε < 10⁻⁷ | iprec 3 | 32 |
| ε < 10⁻⁸ (outside D1) | none: Rotation | – |

On the study's levels (the rule's column; "matched" for comparison):

| κ | ε | p | RCR − direct | iprec 2 at p | iprec 3 at p | rule: p_w, precision | rule ratio | rule waves | matched ratio | matched waves |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0.0078 + 0i | 1e-7 | 20 | 5.77e-8 | 1.079 | 12.214 | 24, iprec 2 | 0.315 | 1264 | 1.001 | 6012 |
| 0.0078 + 0i | 1e-8 | 24 | 9.05e-9 | 2.007 | 43.523 | 32, iprec 3 | 0.166 | 2839 | 1.004 | 7554 |
| 0.031 + 0i | 1e-7 | 20 | 5.77e-8 | 1.125 | 13.262 | 24, iprec 2 | 0.628 | 1260 | 1.000 | 6023 |
| 0.031 + 0i | 1e-8 | 24 | 9.05e-9 | 4.006 | 53.113 | 32, iprec 3 | 0.110 | 2704 | 1.010 | 7411 |
| 0.078 + 0i | 1e-7 | 20 | 5.77e-8 | 2.896 | 14.689 | 24, iprec 2 | 0.290 | 1361 | 1.005 | 5529 |
| 0.078 + 0i | 1e-8 | 24 | 9.06e-9 | 1.848 | 33.753 | 32, iprec 3 | 0.066 | 2663 | 1.007 | 7402 |
| 0.31 + 0i | 1e-7 | 20 | 5.79e-8 | 1.567 | 10.577 | 24, iprec 2 | 0.252 | 1330 | 1.004 | 5667 |
| 0.31 + 0i | 1e-8 | 24 | 9.07e-9 | 1.608 | 22.496 | 32, iprec 3 | 0.036 | 2671 | 1.004 | 7470 |
| 1.25 + 0i | 1e-7 | 20 | 5.97e-8 | 2.029 | 8.002 | 24, iprec 2 | 0.282 | 1375 | 1.005 | 5819 |
| 1.25 + 0i | 1e-8 | 24 | 9.31e-9 | 1.812 | 18.138 | 32, iprec 3 | 0.029 | 2709 | 1.007 | 7648 |
| 2.5 + 0i | 1e-7 | 20 | 6.62e-8 | 2.668 | 5.647 | 24, iprec 2 | 0.113 | 1546 | 1.001 | 6098 |
| 2.5 + 0i | 1e-8 | 24 | 1.01e-8 | 0.739 | 12.967 | 32, iprec 3 | 0.025 | 2945 | 1.001 | 8340 |
| 6.25 + 0i | 1e-7 | 20 | 1.40e-7 | 2.713 | 2.015 | 24, iprec 2 | 0.116 | 2136 | 1.000 | 6826 |
| 6.25 + 0i | 1e-8 | 24 | 1.86e-8 | 0.866 | 11.513 | 32, iprec 3 | 0.031 | 3680 | 1.000 | 8435 |
| 12.5 + 0i | 1e-7 | 23 | 7.43e-7 | 1.003 | 0.998 | 24, iprec 2 | 0.033–1.003* | 3270 | 1.000 | 8621 |
| 12.5 + 0i | 1e-8 | 24 | 2.26e-7 | 1.189 | 0.999 | 32, iprec 3 | 0.010 | 4933 | 1.000 | 10003 |
| 25 + 0i | 1e-7 | 36 | 5.51e-7 | 0.992 | 0.957 | 36, iprec 2 | 0.992 | 6954 | 1.000 | 16457 |
| 25 + 0i | 1e-8 | 38 | 6.80e-8 | 0.957 | 0.925 | 38, iprec 3 | 0.925 | 9051 | 1.000 | 17772 |
| 40 + 0i | 1e-7 | 52 | 4.19e-7 | 0.975 | 0.999 | 52, iprec 2 | 0.975 | 13433 | 1.000 | 28126 |
| 40 + 0i | 1e-8 | 53 | 1.86e-7 | 0.974 | 0.998 | 53, iprec 3 | 0.998 | 16833 | 1.000 | 30903 |
| 1.25 + 0.1875i | 1e-7 | 20 | 5.97e-8 | 2.640 | 9.055 | 24, iprec 2 | 0.184 | 1402 | 1.005 | 5831 |
| 1.25 + 0.1875i | 1e-8 | 24 | 9.31e-9 | 1.182 | 19.765 | 32, iprec 3 | 0.033 | 2741 | 1.002 | 7810 |
| 0.625 + 1.25i | 1e-7 | 20 | 5.63e-8 | 1.406 | 9.994 | 24, iprec 2 | 0.105 | 1293 | 1.000 | 5973 |
| 0.625 + 1.25i | 1e-8 | 24 | 8.87e-9 | 0.665 | 21.547 | 32, iprec 3 | 0.037 | 2671 | 1.003 | 7728 |
| 10 + 5i | 1e-7 | 20 | 2.49e-7 | 4.070 | 1.000 | 24, iprec 2 | 0.810 | 2459 | 1.000 | 6895 |
| 10 + 5i | 1e-8 | 24 | 3.20e-8 | 6.309 | 4.617 | 32, iprec 3 | 0.135 | 4296 | 1.000 | 8864 |

\* κ = 12.5 at 10⁻⁷ has p = 23 (the high branch), so the rule's p_w = 24 is p + 1, which the
study did not run; it measured 1.003 at p and 0.033 at p + 4.

Worst on the plateau 0.810 (10 + 5i at 10⁻⁷, iprec 2 at 24; at 28 it is 0.023) and 0.166 at
10⁻⁸; above the plateau no level is raised and the tables hold at 0.925–1.003. Where the high
branch's p_l already exceeds d the rule changes nothing.

**What it costs**, per box on a raised level (counts, not times):

| ε | p_l → p_w | coefficients (p + 1)² | RCR M2M, L2L ∝ (p + 1)³ | waves | multipole → waves ∝ modes × (p + 1) |
| --- | --- | --- | --- | --- | --- |
| 10⁻⁷ | 20 → 24 | 1.42× | 1.69× | iprec 2's (0.45–0.58× iprec 3's) | 1.19× |
| 10⁻⁸ | 24 → 32 | 1.74× | 2.30× | iprec 3's | 1.32× |

In the FMM the degree is raised either on the level itself (its multipoles and locals, M2M,
L2L and, on leaves, P2M and L2P at p_w), or on the M2L alone (a second multipole at p_w by M2M
from the children at p and the local at p_w L2L'd to the children at p, which is what this
study's path measured). Which, and whether either beats Rotation at p_l, is CH6.1g's timing
(question 2). For orientation: at 10⁻⁶ the exponential M2L took 0.14–0.70× `Auto`'s M2L time on
the M6 and 0.03–0.55× on Kathleen (CH6.2h).

## 4. Candidate 3: the degree-matched quadrature

`nd_fmm_ref::helmholtz::plane_wave::generate_for_degree(κ, ε, p)` (its module documentation,
`fmm-ref/src/helmholtz/plane_wave/degree.rs`):

- **The integrand.** With q = κŝ = (t cos α, t sin α, iλ) the chain at degree p is the quadrature
  of S_p(q; s) e^{i q·d} T_p(q; y), S_p and T_p the truncations at degree p of e^{−i q·s} and
  e^{i q·y} by Gegenbauer's series Σ (2n + 1)(∓i)ⁿ jₙ(κ|x|) Pₙ(q·x/(κ|x|)), a polynomial
  identity at complex q, evaluated in a scaled form finite as κ → 0.
- **The fit.** Samples: sources and targets at their boxes' corners, face centres and centres,
  the 98 centre offsets of the region (d_z ∈ {2, 3}, |d_x|, |d_y| ≤ 3); a check set of four seeded sources
  and targets. The trusted values are nd-fmm-ref's RCR at degree p (P2M, `rcr::m2l`, L2P). The
  generator's panels on the contour of real t, lengthened from the kernel's Re λ = ln(4/ε) + 4
  by p/4, p/2, p until the full rule meets RCR to ε/16; then its column-pivoted Gram–Schmidt with
  weights refitted to RCR's values; Fₖ = p; Nₖ the fewest angles whose trapezoid is within
  ε/(4s|wₖ|) of the node's resolved one (the chain is exactly that trapezoid).
- **The face centres.** A first fit on the corners and centres alone met ε on its samples but
  not between them: at single unit charges at seeded points the chain at p = 24, ε = 5·10⁻¹⁰
  erred 2.68ε (κ = 0.31) and 7.91ε (κ = 2.5) against RCR, and the study's ratio reached 1.114
  (κ = 0.625 + 1.25i, 10⁻⁸) and 1.191 (κ = 40, 10⁻⁸). With the six face centres added the same
  points give 0.06ε and 0.02ε, every study ratio is 1.000–1.010 and the waves change by −13 % to
  +8 % (the appendix's matched rows are the face centres' run).
- **What it gives** (§2 and the table of §3): 1.000–1.010× RCR's error on all 26 points. At
  p = 24, κ = 1.25, ε = 5·10⁻¹⁰: 80 nodes, the contour to Re λ = 37.3 (the full rule's search
  starts at ln(4/ε) + 4 = 26.8), 7,648 waves (iprec 3: 2,709), 3,920 modes (iprec 3: 1,175),
  55.4 s to generate. The whole study (260 rows, 26 matched quadratures from p = 20 to 53) took
  34.8 minutes.
- **As a candidate for the FMM** it would need the quadrature per level from a table cache
  (constraint 5: a minute of generation per FMM build is not acceptable), and 1.6–2.9× the waves
  and 1.7–3.3× the modes of iprec 3 at the level's own degree. It
  avoids raising the degree, so it wins only if the raised degree of §3 costs more than the
  extra waves; CH6.1g's times can say so.
- **As a trusted path** it is the second quadrature the exponential M2L can be checked against
  at any degree, with RCR as its own oracle inside nd-fmm-ref.

## 5. Questions for sign-off

1. **CH6.1g takes candidate 4, the resolving degree** (§3): below 10⁻⁶ a level with a quadrature
   runs its exponential M2L at p_w = max(p_l, 24) with iprec 2 for 10⁻⁷ ≤ ε < 10⁻⁶ and at
   p_w = max(p_l, 32) with iprec 3 for 10⁻⁸ ≤ ε < 10⁻⁷; Rotation below 10⁻⁸. A function of ε and
   p_l only (constraint 9). Recommended. Alternatives: candidate 3 in the FMM with a table cache
   (§4), or Rotation below 10⁻⁶ as now.
2. **Where CH6.1g raises the degree:** (a) on the level (`Degrees` per level: no new level call;
   the leaves' P2M and L2P pay too), or (b) on the M2L alone (a second multipole and local at
   p_w on that level, the path this study measured). Recommended: CH6.1g builds (a) first,
   since it needs only the degree rule, and times (b) only if (a)'s leaf and translation costs
   eat the M2L's gain.
3. **Candidate 3 stays in nd-fmm-ref** as a trusted path, not used by the FMM now. Recommended.

## 6. Tests

- nd-fmm-ref, `degree.rs`'s unit tests: the scaled truncated expansions against e^{∓i q·x} at
  p = 60 (κ = 0.004, 1.25, 2 + i; on and off the contour) within 10⁻¹² relative; RCR's values
  against the kernel at p = 40 within 10⁻¹⁰ (the units and normalisation).
- nd-fmm-ref, `tests/helmholtz/plane_wave.rs`: the chain of `generate_for_degree`, summed term
  by term in the test from the unscaled Gegenbauer series, against RCR at seeded points the
  generator did not fit, within 2ε (κ = 1.25, p = 6, ε = 10⁻⁵ in debug; the ignored
  `the_degree_matched_quadrature_meets_rcr_at_the_fmms_degrees` at κ = 0.0078, 0.31, 2.5,
  1.25 + 0.1875i, p = 20 and 24 at ε/20: 0.00–0.06 ε measured, 407 s); deterministic; no
  shortfall; `degree()`; κ = 0 refused.
- nd-fmm-tables, `tests/helmholtz/plane_wave_pair.rs`: `against_rcr_with_degree_matched_quadratures`
  (through `PlaneWaveLevel`, the FMM's own conversions and shifts, against `HelmholtzTables::m2l`
  at the same degree within 3ε, every direction of every seventh offset; κ = 1.25 + 0.25i, p = 5,
  10⁻⁴; 0.01 ε measured), the ignored `…_at_the_fmms_degrees` (κ = 0.078, 0.31, 2.5, p = 20 and
  24: 0.01–0.16 ε measured, 300 s), and the ignored `degree_matched_study`, this report's tables.

## 7. Reproducing

```sh
cargo test -p nd-fmm-tables --release --test helmholtz -- --ignored degree_matched_study --nocapture
cargo test -p nd-fmm-ref --release --test helmholtz -- --ignored degree_matched --nocapture
cargo test -p nd-fmm-tables --release --test helmholtz -- --ignored degree_matched_quadratures --nocapture
```

## Appendix: the study

`degree_matched_study`'s output: per κ and ε, the FMM's degree p, each candidate's waves and
modes, chain − direct, RCR − direct and their ratio.

| κ | ε | p | candidate | waves | modes | chain − direct | RCR − direct | ratio |
| --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: |
| 0.0078 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 1264 | 541 | 6.23e-8 | 5.77e-8 | 1.079 |
| 0.0078 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 1264 | 541 | 1.82e-8 | 5.77e-8 | 0.315 |
| 0.0078 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 1264 | 541 | 1.82e-8 | 5.77e-8 | 0.315 |
| 0.0078 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 2839 | 1201 | 7.05e-7 | 5.77e-8 | 12.214 |
| 0.0078 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 2839 | 1201 | 3.94e-7 | 5.77e-8 | 6.823 |
| 0.0078 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 2839 | 1201 | 5.34e-8 | 5.77e-8 | 0.925 |
| 0.0078 + 0i | 1e-7 | 20 | generated at 1e-6 | 1780 | 766 | 6.49e-8 | 5.77e-8 | 1.123 |
| 0.0078 + 0i | 1e-7 | 20 | generated at 1e-7 | 2416 | 1038 | 8.11e-8 | 5.77e-8 | 1.405 |
| 0.0078 + 0i | 1e-7 | 20 | generated at 1e-8 | 3059 | 1303 | 2.12e-7 | 5.77e-8 | 3.664 |
| 0.0078 + 0i | 1e-7 | 20 | matched at 5e-9 | 6012 | 2870 | 5.78e-8 | 5.77e-8 | 1.001 |
| 0.0078 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 1264 | 541 | 1.82e-8 | 9.05e-9 | 2.007 |
| 0.0078 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 1264 | 541 | 1.82e-8 | 9.05e-9 | 2.007 |
| 0.0078 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 1264 | 541 | 1.82e-8 | 9.05e-9 | 2.007 |
| 0.0078 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 2839 | 1201 | 3.94e-7 | 9.05e-9 | 43.523 |
| 0.0078 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 2839 | 1201 | 5.41e-8 | 9.05e-9 | 5.980 |
| 0.0078 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 2839 | 1201 | 1.50e-9 | 9.05e-9 | 0.166 |
| 0.0078 + 0i | 1e-8 | 24 | generated at 1e-6 | 1780 | 766 | 1.75e-8 | 9.05e-9 | 1.931 |
| 0.0078 + 0i | 1e-8 | 24 | generated at 1e-7 | 2416 | 1038 | 3.41e-9 | 9.05e-9 | 0.377 |
| 0.0078 + 0i | 1e-8 | 24 | generated at 1e-8 | 3059 | 1303 | 1.56e-8 | 9.05e-9 | 1.723 |
| 0.0078 + 0i | 1e-8 | 24 | matched at 5e-10 | 7554 | 3822 | 9.09e-9 | 9.05e-9 | 1.004 |
| 0.031 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 1260 | 543 | 6.50e-8 | 5.77e-8 | 1.125 |
| 0.031 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 1260 | 543 | 3.63e-8 | 5.77e-8 | 0.628 |
| 0.031 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 1260 | 543 | 3.63e-8 | 5.77e-8 | 0.628 |
| 0.031 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 2704 | 1150 | 7.66e-7 | 5.77e-8 | 13.262 |
| 0.031 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 2704 | 1150 | 4.81e-7 | 5.77e-8 | 8.326 |
| 0.031 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 2704 | 1150 | 5.01e-8 | 5.77e-8 | 0.868 |
| 0.031 + 0i | 1e-7 | 20 | generated at 1e-6 | 1632 | 720 | 6.98e-8 | 5.77e-8 | 1.209 |
| 0.031 + 0i | 1e-7 | 20 | generated at 1e-7 | 2379 | 1024 | 7.20e-8 | 5.77e-8 | 1.248 |
| 0.031 + 0i | 1e-7 | 20 | generated at 1e-8 | 3076 | 1317 | 2.75e-7 | 5.77e-8 | 4.756 |
| 0.031 + 0i | 1e-7 | 20 | matched at 5e-9 | 6023 | 2870 | 5.77e-8 | 5.77e-8 | 1.000 |
| 0.031 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 1260 | 543 | 3.63e-8 | 9.05e-9 | 4.006 |
| 0.031 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 1260 | 543 | 3.63e-8 | 9.05e-9 | 4.006 |
| 0.031 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 1260 | 543 | 3.63e-8 | 9.05e-9 | 4.006 |
| 0.031 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 2704 | 1150 | 4.81e-7 | 9.05e-9 | 53.113 |
| 0.031 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 2704 | 1150 | 5.07e-8 | 9.05e-9 | 5.602 |
| 0.031 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 2704 | 1150 | 1.00e-9 | 9.05e-9 | 0.110 |
| 0.031 + 0i | 1e-8 | 24 | generated at 1e-6 | 1632 | 720 | 2.86e-8 | 9.05e-9 | 3.163 |
| 0.031 + 0i | 1e-8 | 24 | generated at 1e-7 | 2379 | 1024 | 3.18e-9 | 9.05e-9 | 0.351 |
| 0.031 + 0i | 1e-8 | 24 | generated at 1e-8 | 3076 | 1317 | 7.87e-9 | 9.05e-9 | 0.870 |
| 0.031 + 0i | 1e-8 | 24 | matched at 5e-10 | 7411 | 3773 | 9.14e-9 | 9.05e-9 | 1.010 |
| 0.078 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 1361 | 586 | 1.67e-7 | 5.77e-8 | 2.896 |
| 0.078 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 1361 | 586 | 1.67e-8 | 5.77e-8 | 0.290 |
| 0.078 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 1361 | 586 | 1.67e-8 | 5.77e-8 | 0.290 |
| 0.078 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 2663 | 1138 | 8.48e-7 | 5.77e-8 | 14.689 |
| 0.078 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 2663 | 1138 | 3.06e-7 | 5.77e-8 | 5.292 |
| 0.078 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 2663 | 1138 | 2.46e-8 | 5.77e-8 | 0.426 |
| 0.078 + 0i | 1e-7 | 20 | generated at 1e-6 | 1690 | 743 | 7.54e-8 | 5.77e-8 | 1.306 |
| 0.078 + 0i | 1e-7 | 20 | generated at 1e-7 | 2208 | 972 | 6.84e-8 | 5.77e-8 | 1.184 |
| 0.078 + 0i | 1e-7 | 20 | generated at 1e-8 | 3066 | 1315 | 2.65e-7 | 5.77e-8 | 4.592 |
| 0.078 + 0i | 1e-7 | 20 | matched at 5e-9 | 5529 | 2747 | 5.80e-8 | 5.77e-8 | 1.005 |
| 0.078 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 1361 | 586 | 1.67e-8 | 9.06e-9 | 1.848 |
| 0.078 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 1361 | 586 | 1.67e-8 | 9.06e-9 | 1.848 |
| 0.078 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 1361 | 586 | 1.67e-8 | 9.06e-9 | 1.848 |
| 0.078 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 2663 | 1138 | 3.06e-7 | 9.06e-9 | 33.753 |
| 0.078 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 2663 | 1138 | 2.50e-8 | 9.06e-9 | 2.764 |
| 0.078 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 2663 | 1138 | 5.96e-10 | 9.06e-9 | 0.066 |
| 0.078 + 0i | 1e-8 | 24 | generated at 1e-6 | 1690 | 743 | 4.04e-8 | 9.06e-9 | 4.463 |
| 0.078 + 0i | 1e-8 | 24 | generated at 1e-7 | 2208 | 972 | 3.23e-9 | 9.06e-9 | 0.356 |
| 0.078 + 0i | 1e-8 | 24 | generated at 1e-8 | 3066 | 1315 | 8.00e-9 | 9.06e-9 | 0.884 |
| 0.078 + 0i | 1e-8 | 24 | matched at 5e-10 | 7402 | 3773 | 9.12e-9 | 9.06e-9 | 1.007 |
| 0.31 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 1330 | 574 | 9.07e-8 | 5.79e-8 | 1.567 |
| 0.31 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 1330 | 574 | 1.46e-8 | 5.79e-8 | 0.252 |
| 0.31 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 1330 | 574 | 1.46e-8 | 5.79e-8 | 0.252 |
| 0.31 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 2671 | 1148 | 6.12e-7 | 5.79e-8 | 10.577 |
| 0.31 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 2671 | 1148 | 2.04e-7 | 5.79e-8 | 3.526 |
| 0.31 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 2671 | 1148 | 1.31e-8 | 5.79e-8 | 0.226 |
| 0.31 + 0i | 1e-7 | 20 | generated at 1e-6 | 1843 | 795 | 6.46e-8 | 5.79e-8 | 1.117 |
| 0.31 + 0i | 1e-7 | 20 | generated at 1e-7 | 2405 | 1044 | 8.53e-8 | 5.79e-8 | 1.475 |
| 0.31 + 0i | 1e-7 | 20 | generated at 1e-8 | 3102 | 1342 | 2.80e-7 | 5.79e-8 | 4.844 |
| 0.31 + 0i | 1e-7 | 20 | matched at 5e-9 | 5667 | 2788 | 5.81e-8 | 5.79e-8 | 1.004 |
| 0.31 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 1330 | 574 | 1.46e-8 | 9.07e-9 | 1.608 |
| 0.31 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 1330 | 574 | 1.46e-8 | 9.07e-9 | 1.608 |
| 0.31 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 1330 | 574 | 1.46e-8 | 9.07e-9 | 1.608 |
| 0.31 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 2671 | 1148 | 2.04e-7 | 9.07e-9 | 22.496 |
| 0.31 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 2671 | 1148 | 1.34e-8 | 9.07e-9 | 1.476 |
| 0.31 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 2671 | 1148 | 3.30e-10 | 9.07e-9 | 0.036 |
| 0.31 + 0i | 1e-8 | 24 | generated at 1e-6 | 1843 | 795 | 1.75e-8 | 9.07e-9 | 1.924 |
| 0.31 + 0i | 1e-8 | 24 | generated at 1e-7 | 2405 | 1044 | 3.49e-9 | 9.07e-9 | 0.384 |
| 0.31 + 0i | 1e-8 | 24 | generated at 1e-8 | 3102 | 1342 | 1.89e-8 | 9.07e-9 | 2.083 |
| 0.31 + 0i | 1e-8 | 24 | matched at 5e-10 | 7470 | 3773 | 9.11e-9 | 9.07e-9 | 1.004 |
| 1.25 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 1375 | 591 | 1.21e-7 | 5.97e-8 | 2.029 |
| 1.25 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 1375 | 591 | 1.69e-8 | 5.97e-8 | 0.282 |
| 1.25 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 1375 | 591 | 1.69e-8 | 5.97e-8 | 0.282 |
| 1.25 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 2709 | 1175 | 4.78e-7 | 5.97e-8 | 8.002 |
| 1.25 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 2709 | 1175 | 1.69e-7 | 5.97e-8 | 2.827 |
| 1.25 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 2709 | 1175 | 1.18e-8 | 5.97e-8 | 0.198 |
| 1.25 + 0i | 1e-7 | 20 | generated at 1e-6 | 1797 | 807 | 6.66e-8 | 5.97e-8 | 1.115 |
| 1.25 + 0i | 1e-7 | 20 | generated at 1e-7 | 2482 | 1085 | 7.07e-8 | 5.97e-8 | 1.184 |
| 1.25 + 0i | 1e-7 | 20 | generated at 1e-8 | 3152 | 1381 | 2.75e-7 | 5.97e-8 | 4.600 |
| 1.25 + 0i | 1e-7 | 20 | matched at 5e-9 | 5819 | 2870 | 6.00e-8 | 5.97e-8 | 1.005 |
| 1.25 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 1375 | 591 | 1.69e-8 | 9.31e-9 | 1.812 |
| 1.25 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 1375 | 591 | 1.69e-8 | 9.31e-9 | 1.812 |
| 1.25 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 1375 | 591 | 1.69e-8 | 9.31e-9 | 1.812 |
| 1.25 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 2709 | 1175 | 1.69e-7 | 9.31e-9 | 18.138 |
| 1.25 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 2709 | 1175 | 1.21e-8 | 9.31e-9 | 1.297 |
| 1.25 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 2709 | 1175 | 2.68e-10 | 9.31e-9 | 0.029 |
| 1.25 + 0i | 1e-8 | 24 | generated at 1e-6 | 1797 | 807 | 1.84e-8 | 9.31e-9 | 1.976 |
| 1.25 + 0i | 1e-8 | 24 | generated at 1e-7 | 2482 | 1085 | 3.30e-9 | 9.31e-9 | 0.354 |
| 1.25 + 0i | 1e-8 | 24 | generated at 1e-8 | 3152 | 1381 | 7.71e-9 | 9.31e-9 | 0.828 |
| 1.25 + 0i | 1e-8 | 24 | matched at 5e-10 | 7648 | 3920 | 9.38e-9 | 9.31e-9 | 1.007 |
| 2.5 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 1546 | 677 | 1.77e-7 | 6.62e-8 | 2.668 |
| 2.5 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 1546 | 677 | 7.49e-9 | 6.62e-8 | 0.113 |
| 2.5 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 1546 | 677 | 5.89e-9 | 6.62e-8 | 0.089 |
| 2.5 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 2945 | 1281 | 3.74e-7 | 6.62e-8 | 5.647 |
| 2.5 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 2945 | 1281 | 1.31e-7 | 6.62e-8 | 1.986 |
| 2.5 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 2945 | 1281 | 9.61e-9 | 6.62e-8 | 0.145 |
| 2.5 + 0i | 1e-7 | 20 | generated at 1e-6 | 2068 | 921 | 6.80e-8 | 6.62e-8 | 1.028 |
| 2.5 + 0i | 1e-7 | 20 | generated at 1e-7 | 2633 | 1168 | 1.54e-7 | 6.62e-8 | 2.324 |
| 2.5 + 0i | 1e-7 | 20 | generated at 1e-8 | 3356 | 1486 | 2.67e-7 | 6.62e-8 | 4.035 |
| 2.5 + 0i | 1e-7 | 20 | matched at 5e-9 | 6098 | 2993 | 6.62e-8 | 6.62e-8 | 1.001 |
| 2.5 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 1546 | 677 | 7.49e-9 | 1.01e-8 | 0.739 |
| 2.5 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 1546 | 677 | 5.89e-9 | 1.01e-8 | 0.581 |
| 2.5 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 1546 | 677 | 5.89e-9 | 1.01e-8 | 0.581 |
| 2.5 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 2945 | 1281 | 1.31e-7 | 1.01e-8 | 12.967 |
| 2.5 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 2945 | 1281 | 9.74e-9 | 1.01e-8 | 0.961 |
| 2.5 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 2945 | 1281 | 2.59e-10 | 1.01e-8 | 0.025 |
| 2.5 + 0i | 1e-8 | 24 | generated at 1e-6 | 2068 | 921 | 2.22e-8 | 1.01e-8 | 2.195 |
| 2.5 + 0i | 1e-8 | 24 | generated at 1e-7 | 2633 | 1168 | 4.35e-9 | 1.01e-8 | 0.429 |
| 2.5 + 0i | 1e-8 | 24 | generated at 1e-8 | 3356 | 1486 | 8.23e-9 | 1.01e-8 | 0.812 |
| 2.5 + 0i | 1e-8 | 24 | matched at 5e-10 | 8340 | 4165 | 1.01e-8 | 1.01e-8 | 1.001 |
| 6.25 + 0i | 1e-7 | 20 | iprec 2 at p + 0 | 2136 | 921 | 3.79e-7 | 1.40e-7 | 2.713 |
| 6.25 + 0i | 1e-7 | 20 | iprec 2 at p + 4 | 2136 | 921 | 1.61e-8 | 1.40e-7 | 0.116 |
| 6.25 + 0i | 1e-7 | 20 | iprec 2 at p + 8 | 2136 | 921 | 9.55e-9 | 1.40e-7 | 0.068 |
| 6.25 + 0i | 1e-7 | 20 | iprec 3 at p + 0 | 3680 | 1587 | 2.81e-7 | 1.40e-7 | 2.015 |
| 6.25 + 0i | 1e-7 | 20 | iprec 3 at p + 4 | 3680 | 1587 | 2.15e-7 | 1.40e-7 | 1.538 |
| 6.25 + 0i | 1e-7 | 20 | iprec 3 at p + 8 | 3680 | 1587 | 2.53e-8 | 1.40e-7 | 0.182 |
| 6.25 + 0i | 1e-7 | 20 | generated at 1e-6 | 2752 | 1217 | 1.13e-7 | 1.40e-7 | 0.812 |
| 6.25 + 0i | 1e-7 | 20 | generated at 1e-7 | 3438 | 1529 | 3.14e-7 | 1.40e-7 | 2.254 |
| 6.25 + 0i | 1e-7 | 20 | generated at 1e-8 | 4146 | 1840 | 4.92e-7 | 1.40e-7 | 3.524 |
| 6.25 + 0i | 1e-7 | 20 | matched at 5e-9 | 6826 | 3116 | 1.40e-7 | 1.40e-7 | 1.000 |
| 6.25 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 2136 | 921 | 1.62e-8 | 1.86e-8 | 0.866 |
| 6.25 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 2136 | 921 | 9.55e-9 | 1.86e-8 | 0.512 |
| 6.25 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 2136 | 921 | 9.55e-9 | 1.86e-8 | 0.512 |
| 6.25 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 3680 | 1587 | 2.15e-7 | 1.86e-8 | 11.513 |
| 6.25 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 3680 | 1587 | 2.56e-8 | 1.86e-8 | 1.373 |
| 6.25 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 3680 | 1587 | 5.76e-10 | 1.86e-8 | 0.031 |
| 6.25 + 0i | 1e-8 | 24 | generated at 1e-6 | 2752 | 1217 | 3.45e-8 | 1.86e-8 | 1.852 |
| 6.25 + 0i | 1e-8 | 24 | generated at 1e-7 | 3438 | 1529 | 1.26e-8 | 1.86e-8 | 0.677 |
| 6.25 + 0i | 1e-8 | 24 | generated at 1e-8 | 4146 | 1840 | 2.89e-8 | 1.86e-8 | 1.551 |
| 6.25 + 0i | 1e-8 | 24 | matched at 5e-10 | 8435 | 4116 | 1.86e-8 | 1.86e-8 | 1.000 |
| 12.5 + 0i | 1e-7 | 23 | iprec 2 at p + 0 | 3270 | 1365 | 7.45e-7 | 7.43e-7 | 1.003 |
| 12.5 + 0i | 1e-7 | 23 | iprec 2 at p + 4 | 3270 | 1365 | 2.45e-8 | 7.43e-7 | 0.033 |
| 12.5 + 0i | 1e-7 | 23 | iprec 2 at p + 8 | 3270 | 1365 | 1.14e-8 | 7.43e-7 | 0.015 |
| 12.5 + 0i | 1e-7 | 23 | iprec 3 at p + 0 | 4933 | 2096 | 7.41e-7 | 7.43e-7 | 0.998 |
| 12.5 + 0i | 1e-7 | 23 | iprec 3 at p + 4 | 4933 | 2096 | 8.21e-8 | 7.43e-7 | 0.110 |
| 12.5 + 0i | 1e-7 | 23 | iprec 3 at p + 8 | 4933 | 2096 | 1.28e-8 | 7.43e-7 | 0.017 |
| 12.5 + 0i | 1e-7 | 23 | generated at 1e-6 | 4538 | 1925 | 8.40e-7 | 7.43e-7 | 1.131 |
| 12.5 + 0i | 1e-7 | 23 | generated at 1e-7 | 5298 | 2270 | 7.30e-7 | 7.43e-7 | 0.983 |
| 12.5 + 0i | 1e-7 | 23 | generated at 1e-8 | 6160 | 2657 | 8.16e-7 | 7.43e-7 | 1.098 |
| 12.5 + 0i | 1e-7 | 23 | matched at 5e-9 | 8621 | 3901 | 7.43e-7 | 7.43e-7 | 1.000 |
| 12.5 + 0i | 1e-8 | 24 | iprec 2 at p + 0 | 3270 | 1365 | 2.69e-7 | 2.26e-7 | 1.189 |
| 12.5 + 0i | 1e-8 | 24 | iprec 2 at p + 4 | 3270 | 1365 | 1.15e-8 | 2.26e-7 | 0.051 |
| 12.5 + 0i | 1e-8 | 24 | iprec 2 at p + 8 | 3270 | 1365 | 1.14e-8 | 2.26e-7 | 0.050 |
| 12.5 + 0i | 1e-8 | 24 | iprec 3 at p + 0 | 4933 | 2096 | 2.26e-7 | 2.26e-7 | 0.999 |
| 12.5 + 0i | 1e-8 | 24 | iprec 3 at p + 4 | 4933 | 2096 | 4.88e-8 | 2.26e-7 | 0.216 |
| 12.5 + 0i | 1e-8 | 24 | iprec 3 at p + 8 | 4933 | 2096 | 2.17e-9 | 2.26e-7 | 0.010 |
| 12.5 + 0i | 1e-8 | 24 | generated at 1e-6 | 4538 | 1925 | 2.99e-7 | 2.26e-7 | 1.326 |
| 12.5 + 0i | 1e-8 | 24 | generated at 1e-7 | 5298 | 2270 | 3.16e-7 | 2.26e-7 | 1.399 |
| 12.5 + 0i | 1e-8 | 24 | generated at 1e-8 | 6160 | 2657 | 2.28e-7 | 2.26e-7 | 1.009 |
| 12.5 + 0i | 1e-8 | 24 | matched at 5e-10 | 10003 | 4410 | 2.26e-7 | 2.26e-7 | 1.000 |
| 25 + 0i | 1e-7 | 36 | iprec 2 at p + 0 | 6954 | 2740 | 5.47e-7 | 5.51e-7 | 0.992 |
| 25 + 0i | 1e-7 | 36 | iprec 2 at p + 4 | 6954 | 2740 | 5.36e-9 | 5.51e-7 | 0.010 |
| 25 + 0i | 1e-7 | 36 | iprec 2 at p + 8 | 6954 | 2740 | 5.42e-9 | 5.51e-7 | 0.010 |
| 25 + 0i | 1e-7 | 36 | iprec 3 at p + 0 | 9051 | 3668 | 5.28e-7 | 5.51e-7 | 0.957 |
| 25 + 0i | 1e-7 | 36 | iprec 3 at p + 4 | 9051 | 3668 | 7.74e-9 | 5.51e-7 | 0.014 |
| 25 + 0i | 1e-7 | 36 | iprec 3 at p + 8 | 9051 | 3668 | 5.03e-11 | 5.51e-7 | 0.000 |
| 25 + 0i | 1e-7 | 36 | generated at 1e-6 | 10225 | 4034 | 5.47e-7 | 5.51e-7 | 0.993 |
| 25 + 0i | 1e-7 | 36 | generated at 1e-7 | 11427 | 4573 | 5.52e-7 | 5.51e-7 | 1.001 |
| 25 + 0i | 1e-7 | 36 | generated at 1e-8 | 12608 | 5088 | 5.52e-7 | 5.51e-7 | 1.002 |
| 25 + 0i | 1e-7 | 36 | matched at 5e-9 | 16457 | 8249 | 5.51e-7 | 5.51e-7 | 1.000 |
| 25 + 0i | 1e-8 | 38 | iprec 2 at p + 0 | 6954 | 2740 | 6.50e-8 | 6.80e-8 | 0.957 |
| 25 + 0i | 1e-8 | 38 | iprec 2 at p + 4 | 6954 | 2740 | 5.41e-9 | 6.80e-8 | 0.080 |
| 25 + 0i | 1e-8 | 38 | iprec 2 at p + 8 | 6954 | 2740 | 5.42e-9 | 6.80e-8 | 0.080 |
| 25 + 0i | 1e-8 | 38 | iprec 3 at p + 0 | 9051 | 3668 | 6.29e-8 | 6.80e-8 | 0.925 |
| 25 + 0i | 1e-8 | 38 | iprec 3 at p + 4 | 9051 | 3668 | 4.33e-10 | 6.80e-8 | 0.006 |
| 25 + 0i | 1e-8 | 38 | iprec 3 at p + 8 | 9051 | 3668 | 1.08e-11 | 6.80e-8 | 0.000 |
| 25 + 0i | 1e-8 | 38 | generated at 1e-6 | 10225 | 4034 | 6.97e-8 | 6.80e-8 | 1.026 |
| 25 + 0i | 1e-8 | 38 | generated at 1e-7 | 11427 | 4573 | 6.85e-8 | 6.80e-8 | 1.008 |
| 25 + 0i | 1e-8 | 38 | generated at 1e-8 | 12608 | 5088 | 6.93e-8 | 6.80e-8 | 1.020 |
| 25 + 0i | 1e-8 | 38 | matched at 5e-10 | 17772 | 9009 | 6.80e-8 | 6.80e-8 | 1.000 |
| 40 + 0i | 1e-7 | 52 | iprec 2 at p + 0 | 13433 | 4978 | 4.09e-7 | 4.19e-7 | 0.975 |
| 40 + 0i | 1e-7 | 52 | iprec 2 at p + 4 | 13433 | 4978 | 1.39e-8 | 4.19e-7 | 0.033 |
| 40 + 0i | 1e-7 | 52 | iprec 2 at p + 8 | 13433 | 4978 | 1.38e-8 | 4.19e-7 | 0.033 |
| 40 + 0i | 1e-7 | 52 | iprec 3 at p + 0 | 16833 | 6503 | 4.19e-7 | 4.19e-7 | 0.999 |
| 40 + 0i | 1e-7 | 52 | iprec 3 at p + 4 | 16833 | 6503 | 7.54e-9 | 4.19e-7 | 0.018 |
| 40 + 0i | 1e-7 | 52 | iprec 3 at p + 8 | 16833 | 6503 | 8.02e-11 | 4.19e-7 | 0.000 |
| 40 + 0i | 1e-7 | 52 | generated at 1e-6 | 20552 | 7611 | 3.86e-7 | 4.19e-7 | 0.921 |
| 40 + 0i | 1e-7 | 52 | generated at 1e-7 | 22291 | 8396 | 4.19e-7 | 4.19e-7 | 0.999 |
| 40 + 0i | 1e-7 | 52 | generated at 1e-8 | 23354 | 8924 | 4.19e-7 | 4.19e-7 | 1.000 |
| 40 + 0i | 1e-7 | 52 | matched at 5e-9 | 28126 | 15225 | 4.19e-7 | 4.19e-7 | 1.000 |
| 40 + 0i | 1e-8 | 53 | iprec 2 at p + 0 | 13433 | 4978 | 1.81e-7 | 1.86e-7 | 0.974 |
| 40 + 0i | 1e-8 | 53 | iprec 2 at p + 4 | 13433 | 4978 | 1.40e-8 | 1.86e-7 | 0.075 |
| 40 + 0i | 1e-8 | 53 | iprec 2 at p + 8 | 13433 | 4978 | 1.38e-8 | 1.86e-7 | 0.074 |
| 40 + 0i | 1e-8 | 53 | iprec 3 at p + 0 | 16833 | 6503 | 1.86e-7 | 1.86e-7 | 0.998 |
| 40 + 0i | 1e-8 | 53 | iprec 3 at p + 4 | 16833 | 6503 | 2.92e-9 | 1.86e-7 | 0.016 |
| 40 + 0i | 1e-8 | 53 | iprec 3 at p + 8 | 16833 | 6503 | 2.63e-11 | 1.86e-7 | 0.000 |
| 40 + 0i | 1e-8 | 53 | generated at 1e-6 | 20552 | 7611 | 1.83e-7 | 1.86e-7 | 0.983 |
| 40 + 0i | 1e-8 | 53 | generated at 1e-7 | 22291 | 8396 | 1.86e-7 | 1.86e-7 | 0.998 |
| 40 + 0i | 1e-8 | 53 | generated at 1e-8 | 23354 | 8924 | 1.86e-7 | 1.86e-7 | 1.001 |
| 40 + 0i | 1e-8 | 53 | matched at 5e-10 | 30903 | 16371 | 1.86e-7 | 1.86e-7 | 1.000 |
| 1.25 + 0.1875i | 1e-7 | 20 | iprec 2 at p + 0 | 1402 | 601 | 1.58e-7 | 5.97e-8 | 2.640 |
| 1.25 + 0.1875i | 1e-7 | 20 | iprec 2 at p + 4 | 1402 | 601 | 1.10e-8 | 5.97e-8 | 0.184 |
| 1.25 + 0.1875i | 1e-7 | 20 | iprec 2 at p + 8 | 1402 | 601 | 1.10e-8 | 5.97e-8 | 0.184 |
| 1.25 + 0.1875i | 1e-7 | 20 | iprec 3 at p + 0 | 2741 | 1181 | 5.41e-7 | 5.97e-8 | 9.055 |
| 1.25 + 0.1875i | 1e-7 | 20 | iprec 3 at p + 4 | 2741 | 1181 | 1.84e-7 | 5.97e-8 | 3.081 |
| 1.25 + 0.1875i | 1e-7 | 20 | iprec 3 at p + 8 | 2741 | 1181 | 1.17e-8 | 5.97e-8 | 0.196 |
| 1.25 + 0.1875i | 1e-7 | 20 | generated at 1e-6 | 1798 | 798 | 6.63e-8 | 5.97e-8 | 1.111 |
| 1.25 + 0.1875i | 1e-7 | 20 | generated at 1e-7 | 2570 | 1099 | 1.42e-7 | 5.97e-8 | 2.371 |
| 1.25 + 0.1875i | 1e-7 | 20 | generated at 1e-8 | 3247 | 1386 | 2.69e-7 | 5.97e-8 | 4.503 |
| 1.25 + 0.1875i | 1e-7 | 20 | matched at 5e-9 | 5831 | 2829 | 6.00e-8 | 5.97e-8 | 1.005 |
| 1.25 + 0.1875i | 1e-8 | 24 | iprec 2 at p + 0 | 1402 | 601 | 1.10e-8 | 9.31e-9 | 1.182 |
| 1.25 + 0.1875i | 1e-8 | 24 | iprec 2 at p + 4 | 1402 | 601 | 1.10e-8 | 9.31e-9 | 1.182 |
| 1.25 + 0.1875i | 1e-8 | 24 | iprec 2 at p + 8 | 1402 | 601 | 1.10e-8 | 9.31e-9 | 1.182 |
| 1.25 + 0.1875i | 1e-8 | 24 | iprec 3 at p + 0 | 2741 | 1181 | 1.84e-7 | 9.31e-9 | 19.765 |
| 1.25 + 0.1875i | 1e-8 | 24 | iprec 3 at p + 4 | 2741 | 1181 | 1.20e-8 | 9.31e-9 | 1.289 |
| 1.25 + 0.1875i | 1e-8 | 24 | iprec 3 at p + 8 | 2741 | 1181 | 3.10e-10 | 9.31e-9 | 0.033 |
| 1.25 + 0.1875i | 1e-8 | 24 | generated at 1e-6 | 1798 | 798 | 1.82e-8 | 9.31e-9 | 1.960 |
| 1.25 + 0.1875i | 1e-8 | 24 | generated at 1e-7 | 2570 | 1099 | 3.92e-9 | 9.31e-9 | 0.421 |
| 1.25 + 0.1875i | 1e-8 | 24 | generated at 1e-8 | 3247 | 1386 | 1.83e-8 | 9.31e-9 | 1.963 |
| 1.25 + 0.1875i | 1e-8 | 24 | matched at 5e-10 | 7810 | 3871 | 9.32e-9 | 9.31e-9 | 1.002 |
| 0.625 + 1.25i | 1e-7 | 20 | iprec 2 at p + 0 | 1293 | 547 | 7.92e-8 | 5.63e-8 | 1.406 |
| 0.625 + 1.25i | 1e-7 | 20 | iprec 2 at p + 4 | 1293 | 547 | 5.90e-9 | 5.63e-8 | 0.105 |
| 0.625 + 1.25i | 1e-7 | 20 | iprec 2 at p + 8 | 1293 | 547 | 5.90e-9 | 5.63e-8 | 0.105 |
| 0.625 + 1.25i | 1e-7 | 20 | iprec 3 at p + 0 | 2671 | 1133 | 5.63e-7 | 5.63e-8 | 9.994 |
| 0.625 + 1.25i | 1e-7 | 20 | iprec 3 at p + 4 | 2671 | 1133 | 1.91e-7 | 5.63e-8 | 3.392 |
| 0.625 + 1.25i | 1e-7 | 20 | iprec 3 at p + 8 | 2671 | 1133 | 1.19e-8 | 5.63e-8 | 0.211 |
| 0.625 + 1.25i | 1e-7 | 20 | generated at 1e-6 | 1624 | 693 | 6.94e-8 | 5.63e-8 | 1.233 |
| 0.625 + 1.25i | 1e-7 | 20 | generated at 1e-7 | 2254 | 955 | 7.77e-8 | 5.63e-8 | 1.380 |
| 0.625 + 1.25i | 1e-7 | 20 | generated at 1e-8 | 2880 | 1218 | 2.85e-7 | 5.63e-8 | 5.054 |
| 0.625 + 1.25i | 1e-7 | 20 | matched at 5e-9 | 5973 | 2747 | 5.63e-8 | 5.63e-8 | 1.000 |
| 0.625 + 1.25i | 1e-8 | 24 | iprec 2 at p + 0 | 1293 | 547 | 5.90e-9 | 8.87e-9 | 0.665 |
| 0.625 + 1.25i | 1e-8 | 24 | iprec 2 at p + 4 | 1293 | 547 | 5.90e-9 | 8.87e-9 | 0.665 |
| 0.625 + 1.25i | 1e-8 | 24 | iprec 2 at p + 8 | 1293 | 547 | 5.90e-9 | 8.87e-9 | 0.665 |
| 0.625 + 1.25i | 1e-8 | 24 | iprec 3 at p + 0 | 2671 | 1133 | 1.91e-7 | 8.87e-9 | 21.547 |
| 0.625 + 1.25i | 1e-8 | 24 | iprec 3 at p + 4 | 2671 | 1133 | 1.22e-8 | 8.87e-9 | 1.372 |
| 0.625 + 1.25i | 1e-8 | 24 | iprec 3 at p + 8 | 2671 | 1133 | 3.24e-10 | 8.87e-9 | 0.037 |
| 0.625 + 1.25i | 1e-8 | 24 | generated at 1e-6 | 1624 | 693 | 2.92e-8 | 8.87e-9 | 3.298 |
| 0.625 + 1.25i | 1e-8 | 24 | generated at 1e-7 | 2254 | 955 | 4.32e-9 | 8.87e-9 | 0.488 |
| 0.625 + 1.25i | 1e-8 | 24 | generated at 1e-8 | 2880 | 1218 | 6.76e-9 | 8.87e-9 | 0.763 |
| 0.625 + 1.25i | 1e-8 | 24 | matched at 5e-10 | 7728 | 3724 | 8.89e-9 | 8.87e-9 | 1.003 |
| 10 + 5i | 1e-7 | 20 | iprec 2 at p + 0 | 2459 | 924 | 1.01e-6 | 2.49e-7 | 4.070 |
| 10 + 5i | 1e-7 | 20 | iprec 2 at p + 4 | 2459 | 924 | 2.02e-7 | 2.49e-7 | 0.810 |
| 10 + 5i | 1e-7 | 20 | iprec 2 at p + 8 | 2459 | 924 | 5.61e-9 | 2.49e-7 | 0.023 |
| 10 + 5i | 1e-7 | 20 | iprec 3 at p + 0 | 4296 | 1709 | 2.49e-7 | 2.49e-7 | 1.000 |
| 10 + 5i | 1e-7 | 20 | iprec 3 at p + 4 | 4296 | 1709 | 1.48e-7 | 2.49e-7 | 0.593 |
| 10 + 5i | 1e-7 | 20 | iprec 3 at p + 8 | 4296 | 1709 | 6.13e-8 | 2.49e-7 | 0.246 |
| 10 + 5i | 1e-7 | 20 | generated at 1e-6 | 2707 | 1025 | 6.38e-7 | 2.49e-7 | 2.560 |
| 10 + 5i | 1e-7 | 20 | generated at 1e-7 | 3250 | 1259 | 7.31e-7 | 2.49e-7 | 2.933 |
| 10 + 5i | 1e-7 | 20 | generated at 1e-8 | 4441 | 1701 | 4.69e-7 | 2.49e-7 | 1.881 |
| 10 + 5i | 1e-7 | 20 | matched at 5e-9 | 6895 | 2829 | 2.49e-7 | 2.49e-7 | 1.000 |
| 10 + 5i | 1e-8 | 24 | iprec 2 at p + 0 | 2459 | 924 | 2.02e-7 | 3.20e-8 | 6.309 |
| 10 + 5i | 1e-8 | 24 | iprec 2 at p + 4 | 2459 | 924 | 5.63e-9 | 3.20e-8 | 0.176 |
| 10 + 5i | 1e-8 | 24 | iprec 2 at p + 8 | 2459 | 924 | 2.06e-10 | 3.20e-8 | 0.006 |
| 10 + 5i | 1e-8 | 24 | iprec 3 at p + 0 | 4296 | 1709 | 1.48e-7 | 3.20e-8 | 4.617 |
| 10 + 5i | 1e-8 | 24 | iprec 3 at p + 4 | 4296 | 1709 | 6.18e-8 | 3.20e-8 | 1.930 |
| 10 + 5i | 1e-8 | 24 | iprec 3 at p + 8 | 4296 | 1709 | 4.32e-9 | 3.20e-8 | 0.135 |
| 10 + 5i | 1e-8 | 24 | generated at 1e-6 | 2707 | 1025 | 3.43e-8 | 3.20e-8 | 1.071 |
| 10 + 5i | 1e-8 | 24 | generated at 1e-7 | 3250 | 1259 | 1.13e-7 | 3.20e-8 | 3.539 |
| 10 + 5i | 1e-8 | 24 | generated at 1e-8 | 4441 | 1701 | 1.84e-7 | 3.20e-8 | 5.755 |
| 10 + 5i | 1e-8 | 24 | matched at 5e-10 | 8864 | 3822 | 3.20e-8 | 3.20e-8 | 1.000 |
