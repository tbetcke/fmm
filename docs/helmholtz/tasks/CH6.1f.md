# CH6.1f Quadratures for ε below 10⁻⁶: the degree-matched quadrature on the pair chain

Started 2026-11-09 on `tbetcke/h-ch6-1f`. Roadmap: "CH6.1f Quadratures for ε below 10⁻⁶"
(H6; priority list item 6), split here by `/next-task`'s size rule into the quadrature and
its study on the pair chain (CH6.1f, this task) and the FMM with the acceptance (CH6.1g).
Proposal: §10 CH6.1 ("errors within about 1.1× Rotation's"); §5.3 (Exponential M2L), §2
(constraints 6 and 9); CONVENTIONS §3.15 "Exponential M2L" ("The degree": the pairing of a
level's degree and quadrature is not fixed there); CH6.1c's degree study
(docs/helmholtz/tasks/CH6.1c.md, "Results"); CH6.1e's report §2, §3, §5, §7
(fmm-validate/results/helmholtz-ch6-1e.md).

## Goal

Find, on the pair chain, a quadrature of the exponential M2L that keeps the chain at the
FMM's degrees of ε = 10⁻⁷ and 10⁻⁸ (p = 20 and 24 on the low-frequency plateau) within
about 1.1× RCR's error against the direct sum, at every κ_l of CH6.1e's levels, and say
what it costs in waves; or show that none does within a budget.

## Why the split

The roadmap's CH6.1f is a quadrature, its rule in the FMM and the FMM's errors on
CH6.1e's cases (87 cases, an hour or more on the M6). Which quadrature is an open numerical
question (below), and its answer decides the FMM's change: a generated quadrature per
level, FMM3D's iprec 3 at a raised degree, or none. A study and a new generator variant
(nd-fmm-ref, nd-fmm-tables' tests) and then the FMM's rule, operator and measurement
(nd-fmm-tables, nd-fmm-exec, nd-fmm-validate) do not fit one reviewable PR. Split:

- **CH6.1f** (this task): the candidates on the pair chain, a degree-matched generator in
  `nd_fmm_ref::helmholtz::plane_wave`, the study and its report, a recommendation for
  sign-off.
- **CH6.1g**: the recommended quadrature in `PlaneWaveM2l`'s rule below 10⁻⁶, the FMM
  within about 1.1× Rotation's at 10⁻⁷ and 10⁻⁸ on CH6.1e's cases (the roadmap's *Done
  when* of CH6.1f, moved), times beside Rotation; D8's bound ε ≥ 10⁻⁶ revisited as a
  question. *Needs:* CH6.1f and its sign-off.

## Acceptance

Proposal §10 CH6.1, verbatim: "errors within about 1.1× Rotation's, as T13's budget for
Laplace". The roadmap's CH6.1f: "the FMM within about 1.1× Rotation's error at ε = 10⁻⁷
and 10⁻⁸ on CH6.1e's cases, the clustered trees' deep levels included, with the rule still
a function of k, w_l, p_l and ε (constraint 9); or the report shows no such quadrature
fits the tables' budget." That is CH6.1g's after the split. CH6.1f's *Done when*:

> On the pair chain (CH6.1c's measure: sources at the box's corners, centre and seeded
> points, every V offset along its directions, L2P at 27 targets, box widths), at the FMM's
> degrees p = 20 and 24 and at κ from 0.008 to 25 (CH6.1e's levels at 10⁻⁷ and 10⁻⁸,
> the clustered trees' deep levels at κ ≤ 0.08 included), real and complex k: each
> candidate's chain − direct over RCR − direct at the same degree, its waves Σₖ Nₖ and
> modes against iprec 2's and 3's; a candidate within about 1.1× at every point
> recommended for CH6.1g with its rule as a function of k, w_l, p_l and ε, for sign-off;
> or the report shows that none is, and what it would cost.

## The problem (CH6.1c, CH6.1e)

The chain is exact per degree: with the exact integral in place of the quadrature it is
RCR at degree p. The quadrature is fitted to the kernel e^{iκ|u|}/|u| for 1 ≤ u_z ≤ 4. The
waves of a degree-p multipole are a truncated series in λ (the expansion of e^{−iκŝ·s},
whose terms grow like (λ |s|)ⁿ at complex ŝ): for λ |s| beyond about p they are not the
kernel's, and the quadrature, compressed for the kernel, errs on them. FMM3D's iprec 3
reaches λ ≈ 25 (its stated error 6·10⁻¹¹ at z = 1), iprec 2 less; so at p = 20–24 iprec 3
errs on the tail by up to 25× RCR's error on the pair chain (CH6.1c's study, κ = 0.31,
p = 24) and iprec 2 by 1.4–1.5×. In the FMM (CH6.1e) iprec 3 held on most levels but not
on the clustered trees' deep levels (κ_l ≤ 0.08: 1.12× at 10⁻⁷, 1.30× and 1.76× at 10⁻⁸),
and iprec 2 missed on 25 of 32 levels at 10⁻⁷.

## Candidates

1. **FMM3D's tables** (baseline): iprec 2 and 3 at the level's degree, CH6.1c's rows.
2. **The plain generator at a chosen ε** (`generate(κ, ε_g)`): the best ε_g per (κ, p) by
   a scan from 10⁻⁶ to 10⁻¹². If one ε_g per p holds 1.1×, the rule is a table of ε_g.
3. **A degree-matched quadrature** (new, `generate_for_degree(κ, ε, p)` in nd-fmm-ref):
   the λ rule fitted not to the kernel but to the truncated chain itself, so that the
   chain reproduces RCR at degree p (the exact integral of the truncated integrand) to a
   fraction of RCR's own error. Per node (λₖ, tₖ) the α-integrated product of the truncated
   expansions, by Gegenbauer's addition theorem at complex ŝ (ŝ·ŝ = 1):
   S_p(ŝ; s) = Σ_{n≤p} (2n + 1)(−i)ⁿ jₙ(κ|s|) Pₙ(ŝ·ŝ_s), T_p(ŝ; y) the same with iⁿ, and
   the shift e^{iκŝ·d}; the columns are their α means at sample (source, target, offset)
   triples; the panels extended in λ until the full rule meets RCR's values at degree p
   (`nd_fmm_ref::helmholtz::rcr` with `leaf::{p2m, l2p}`, the trusted path inside
   nd-fmm-ref), then compressed by the generator's column-pivoted Gram–Schmidt with
   refitted weights; Fourier counts min(p, …) and α counts until each kept node's α mean
   converges. A function of κ, ε and p only (constraint 9). Its cost is the open part: the
   class reaches larger λ than the kernel's (the tail lives near λ ≈ p/|s|), so larger tₖ
   and α counts.
4. **iprec 3 at a raised degree**: the level's degree p + Δ (Δ = 4, 8) with FMM3D's iprec
   3, compared with RCR at p (the FMM's errors are set by its other levels' p). Costs the
   level's M2M, L2L and leaf operators at the larger degree, not more waves.

The study reports all four at the same points; the recommendation weighs the error ratio
against the waves (merge and shift's cost) and, for 4, the degree's cost.

## Read first

- `fmm-ref/src/helmholtz/plane_wave.rs`, `plane_wave/rule.rs` (the generator, its λ rule,
  compression and counts), `plane_wave/special.rs` (jₙ, Legendre).
- `fmm-ref/src/helmholtz/{rcr,leaf}.rs` (RCR at degree p, P2M, L2P).
- `fmm-tables/src/helmholtz/plane_wave/pair.rs` (`Nodes`, `PlaneWaveLevel::from_nodes`, the
  factors Pₖ(n, m)); `tests/helmholtz/plane_wave_pair.rs` (`pair_error`, `degree_study`,
  `study`).
- CH6.1c's record ("Results", the degree table), CH6.1e's report §2 (the scan at 10⁻⁷ and
  10⁻⁸), §3 (why no κ guard for iprec 3).

## Changes

- `nd-fmm-ref`, `helmholtz::plane_wave`: `generate_for_degree(kappa, eps, p) -> Quadrature`
  (candidate 3) beside `generate`, whose arithmetic stays the spike's bit for bit
  (fmm-ref/CLAUDE.md: `generate` unchanged, so `examples/generated.rs` need not rerun);
  the new code in a module of its own (`plane_wave/degree.rs`), sharing the panels,
  `compress` and `tail_count` of `rule.rs` without changing them. `Quadrature` gains the
  degree it was matched to (`degree() -> Option<usize>`, `None` for `generate`'s) and its
  `shortfall` keeps its meaning (the target against RCR at degree p, named in `Bound`'s
  docs if a new bound is needed). Public items documented; fmm-ref/CLAUDE.md updated.
- `nd-fmm-tables`, tests only: `tests/helmholtz/plane_wave_pair.rs` gains the study of the
  four candidates (`#[ignore]`, release, prints the report's tables) and a debug check of
  candidate 3 against RCR on one level (`check_degree_matched`), with `pair_error`'s
  measure. No library change: `PlaneWaveLevel::from_nodes` already takes any `Nodes`.
- The report `fmm-validate/results/helmholtz-ch6-1f.md`.
- Roadmap: the split (CH6.1f renamed as above, CH6.1g added with *Done when* and *Needs*);
  the priority list's item 6 names both.
- No convention change: §3.15 admits a generated quadrature and leaves the pairing with the
  degree open. No dependency, no Laplace path, no nd-fmm-plan interface.

## Tests that prove it

- nd-fmm-ref (`tests/helmholtz/plane_wave.rs`), against RCR at degree p (nd-fmm-ref's own
  `rcr::m2l` between `leaf::p2m` and `leaf::l2p`): `generate_for_degree` at
  κ ∈ {0.0078, 0.31, 2.5, 1.25 + 0.19i}, p ∈ {20, 24}, ε ∈ {10⁻⁷, 10⁻⁸}: the chain's
  α-resolved sum at the fit's samples and at seeded samples off them within its target of
  RCR's values, absolute in box widths (debug: two points; release, ignored: all); no
  shortfall; the counts and `degree()` as stated; a function of (κ, ε, p) only (two calls
  bit for bit).
- nd-fmm-tables (`tests/helmholtz/plane_wave_pair.rs`), through `PlaneWaveLevel` (the
  FMM's own conversions and shifts) against `HelmholtzTables::m2l` and the direct sum:
  `check_degree_matched`, one level at κ ≈ 0.31, p = 20, every third offset, the chain
  within 0.1× RCR − direct of RCR (debug); the ignored `degree_matched_study` with the four
  candidates at every κ and p of the *Done when*, printing the report's tables.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
`cargo test`, `cargo test --workspace`, the MPI list at 2 and 4 ranks; Laplace untouched);
`cargo test -p nd-fmm-ref --release -- --ignored` and `cargo test -p nd-fmm-tables
--release -- --ignored` for the study (tables' ignored tests include CH6.1a's 6- and
8-minute runs; run the new ones by name, the rest only if `pair.rs` or `quadrature.rs`
change, which they do not).

## Machine runs

None beyond the M6, and no timing: the study measures errors and counts only
(deterministic). Release runs on the M6 in the sandbox (no MPI, no timing gate).

## Constraints and risks

- Constraint 6: the new quadrature is checked against RCR (nd-fmm-ref's) and the direct
  sum; the study is the tables' chain against `HelmholtzTables::m2l`.
- Constraint 9: the quadrature is a function of κ, ε and p only; the samples are fixed
  (no charges of the problem).
- Constraint 1: Laplace untouched (nd-fmm-ref's Helmholtz module and nd-fmm-tables' tests).
- Risk: candidate 3's λ range and α counts make it more waves than Rotation can beat per
  pair; then the report says so, and candidate 4 or "Rotation below 10⁻⁶" is the
  recommendation (the roadmap's "or the report shows no such quadrature fits").
- Risk: the generator's cost at tight ε and p = 24 (seconds per κ) is fine for a trusted
  path; CH6.1g decides whether the FMM generates per level or tabulates.

## Pending runs

None.

## Open questions

None yet.
