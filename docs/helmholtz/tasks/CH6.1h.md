# CH6.1h The resolving degree on the M2L alone

Started 2026-11-11 on `tbetcke/h-ch6-1h`. Roadmap: "CH6.1h The resolving degree on the M2L
alone" (H6; priority list item 6, the last part of CH6.1f "Quadratures for ε below 10⁻⁶";
added by CH6.1g's sign-off 3). Proposal: §10 CH6.1 ("errors within about 1.1× Rotation's");
§5.3 (Exponential M2L); §2 (constraints 1, 6, 7, 9); §12 D8. CH6.1g's report
(fmm-validate/results/helmholtz-ch6-1g.md §3, §5), CH6.1f's (helmholtz-ch6-1f.md §3, §4: the
chain as a quadrature of S_p(q; s) e^{iq·d} T_p(q; y)).

## Goal

Raise only what the exponential M2L needs to the resolving degree p_w on a level with waves
below 10⁻⁶, keep the level's other translations at p_l, and time it beside CH6.1g's
whole-level raise and Rotation on CH6.1e's cases, so that the faster form becomes the rule's
and D8's bound at 10⁻⁷ ≤ ε < 10⁻⁶ can be asked again.

## Acceptance (roadmap, verbatim; proposal §10 CH6.1: "errors within about 1.1× Rotation's")

> *Done when:* the FMM within about 1.1× Rotation's error at 10⁻⁷ and 10⁻⁸ on CH6.1e's cases
> (asserted on small cases), deterministic and rank-invariant; the evaluation timed beside
> CH6.1g's and Rotation's on the same cases on Kathleen; the faster form made the rule's, and
> D8's bound at 10⁻⁷ ≤ ε < 10⁻⁶ asked again. *Needs:* CH6.1g. *Machines:* Kathleen.
> *Sign-off:* the form and D8's bound.

## The form

CH6.1f (§3, §4) reads the chain of a degree-p M2L as the quadrature of
S_p(q; s) e^{iq·d} T_p(q; y): the truncated plane-wave series of the sources about the
multipole's centre and of the targets about the local's. At complex q a truncated series
converges worse the larger |s| or |y|, so iprec 3 fails at p_l on a box's own radius and
holds at p_w; the study's chain held with the multipole formed by M2M from children at p
(half the radius) and the local taken back to the children by L2L at p. So on a level l with
waves the M2L needs its multipole and local at p_w; what reads or forms them at the box's
own radius needs p_w too, what works at the children's radius does not:

| operator on a raised level l | CH6.1g (`Level`) | `Leaves` | `M2l` (the M2L alone) |
| --- | --- | --- | --- |
| stores (multipole, local), ghost exchange | p_w | p_w | p_w |
| M2L (merge and shift) | p_w | p_w | p_w |
| M2M from the children into l | p_{l+1} or p_w → p_w | p_{l+1} → p_w | p_{l+1} → p_w |
| M2M from l to its parent | p_w → p_{l−1} | **p_l** → p_{l−1} | **p_l** → p_{l−1} |
| L2L from the parent into l | p_{l−1} → p_w | p_{l−1} → **p_l** | p_{l−1} → **p_l** |
| L2L from l to the children | p_w → p_{l+1} or p_w | p_w → p_{l+1} | p_w → p_{l+1} |
| P2M, L2P on l's leaves; P2L into, M2P from its boxes | p_w | p_w | **p_l** (the first degrees) |

(p_{l+1} under `Leaves` and `M2l` is the child level's translation degree, its base.) The
plan first built `Leaves` alone: by the reading above a leaf's P2M at p_l gives S_{p_l} at
the box's own radius, the failing chain of iprec 3 at p_l (CH6.1g §1: 0.56–1.07× Rotation's
on the test trees; CH6.1e: up to 1.76× on the clustered trees' deep levels). A temporary
probe (leaves at p_l, not committed; "Open questions" 1) held 0.25–0.96× Rotation's on four
of CH6.1e's large cases, so `M2l`, the roadmap's "M2L alone", is built and measured too.

## Read first

- `fmm-tables/src/helmholtz.rs` (`HelmholtzTables::build`, `translate`, `Shift`, the module
  docs' table of degrees).
- `fmm-exec/src/helmholtz/operator.rs` (`select_tables`, `Kernel::p`, the bodies, `FmmSizes`),
  `fmm-exec/src/helmholtz/operator_plane_wave.rs` (`tables.p()`), `fmm-exec/src/helmholtz/fmm.rs`
  (the builder's plane-wave settings, `degrees`).
- `fmm-exec/tests/helmholtz/plane_wave.rs` (`the_rule_raises_the_degrees_…`, Auto),
  `fmm-exec/tests/helmholtz_fmm.rs` (`check_resolving_degree`),
  `fmm-tables/tests/helmholtz/` (the M2M and L2L tests against `rcr`).
- `fmm-validate/src/helmholtz_accuracy.rs` and the example (`--split`, as the pattern of a
  setting per exponential-M2L run), `tools/helmholtz/ch6-1g-runs.sh`, `ch6-1g-tables.py`.

## Changes

No convention, dependency, `nd-fmm-plan` or Laplace change (the Helmholtz module of
nd-fmm-tables and the Helmholtz operator of nd-fmm-exec only).

1. **nd-fmm-tables, `helmholtz`**: `HelmholtzTables::build_with_m2l_degrees(k̂₀, degrees,
   m2l_degrees, budget)`: level l's expansions (blocks) and M2L at q_l = m2l_degrees[l] ≥ p_l;
   M2M into l from p_{l+1} of the child's block to q_l, L2L from q_l to p_{l+1} of the child's
   block; `build` is it with q = p, bit for bit. `translate` reads and writes the first
   degrees of each plane of a larger block (the block's degree a parameter; the copy into
   `work` already exists). `m2l_degree(level)`; the module docs' table of degrees.
2. **nd-fmm-exec, `HelmholtzOperator`**: `PlaneWaveRaise { Level, Leaves, M2l }`
   (re-exported from `helmholtz`), `with_plane_wave_raise`, `plane_wave_raise()`;
   `select_tables` builds the tables with degrees = q = p_w under `Level` (CH6.1g's, bit for
   bit) and degrees = the base, q = p_w under `Leaves` and `M2l`. The stores and the M2L at
   q_l (`Kernel::p` reads `m2l_degree`), the leaves' operators at q_l or, under `M2l`, at
   p_l on the first degrees through a scratch expansion (`Kernel::leaf_degree`);
   `degree(level)` stays the degree of the stores and the M2L (q_l),
   `translation_degree(level)` new (p_l of the M2M out and the L2L in).
   `HelmholtzFmmBuilder::plane_wave_raise`; the default `Level` until the sign-off (then
   the form signed off).
3. **nd-fmm-validate**: `helmholtz_accuracy::Case::plane_wave_raise`, `Row::plane_wave_raise`,
   the example's `--raise level|leaves|m2l,…` (each exponential-M2L run at each form, the
   column "M2L" ending `/leaves` or `/m2l` where named), as `--split`.
4. **tools**: `tools/helmholtz/ch6-1h-runs.sh` (CH6.1g's 26 cases, `--m2l rotation,planewave
   --raise level,leaves,m2l`, the order reversed in the second launch), `ch6-1h-tables.py`.
5. Docs: fmm-tables/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-validate/CLAUDE.md, the module docs;
   the report `fmm-validate/results/helmholtz-ch6-1h.md`.

## Tests that prove it

- nd-fmm-tables `tests/helmholtz/` (debug): `build_with_m2l_degrees` with q = p equals `build`
  bit for bit on M2M, L2L and M2L of every level; with q_l > p_l on two levels, M2M (p_{l+1}
  of a q_{l+1} block → q_l) and L2L (q_l → p_{l+1} of a q_{l+1} block, the rest of the block
  untouched) against `rcr` on the truncated input, within the existing FMM-weighted bound,
  and M2L at q_l bit for bit the tables built at q.
- nd-fmm-exec `tests/helmholtz/plane_wave.rs` (no MPI): under `M2l` at 10⁻⁷ and 10⁻⁸ the
  degrees and store sizes p_w on the raised levels, the translation degrees the base, the
  plane-wave tables as under `Level`; `Level` named bit for bit the default; Rotation
  afterwards back at the base.
- nd-fmm-exec `tests/helmholtz_fmm.rs`, `check_resolving_degree`: also under `M2l` on its four
  trees (the clusters at kD = 1, the cube at kD = 20 + 2i, 10⁻⁷ and 10⁻⁸): the errors against
  the direct sum within `PLANE_WAVE_RATIO` (1.1) of Rotation's, 2 threads and on P ranks the
  one-rank FMM bit for bit.
- nd-fmm-validate `tests/helmholtz_accuracy.rs`: the `--raise` setting reaches the FMM (the
  row's form; W1 at kD = 10, 10⁻⁸ under `M2l` within 1.1× Rotation's).
- The report's runs: the 26 cases under both forms, every one within 1.1× Rotation's (φ, ∇φ).

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`,
`cargo test`, `cargo test --workspace`, the MPI list at 2 and 4 ranks; Laplace untouched);
`cargo clippy -p nd-fmm-tables --all-targets`, `cargo test -p nd-fmm-tables`;
`cargo test -p nd-fmm-validate --test helmholtz_accuracy`.

## Machine runs

Kathleen, one node 1 × 40 (exclusive), `tools/helmholtz/ch6-1h-runs.sh` through `run.sbatch`:
CH6.1e's 26 cases at 10⁻⁷ and 10⁻⁸, Rotation, PlaneWave under `Level` and under `M2l`, each
case forward and reversed; the errors, the evaluation, the M2L per level and the rest.
Decides the acceptance on the large cases, the form (the faster where both hold) and the
data for D8's question. No M6 timing.

## Constraints and risks

- Constraint 1: Laplace untouched (nd-fmm-tables' `helmholtz` module and the Helmholtz
  operator; `build` bit for bit as before, so `Level` is CH6.1g's output bit for bit).
- Constraint 9: q_l and p_l functions of ε, p_l, k and w_l as before; the form a setting.
- Constraint 7: the stores as under `Level` (p_w); the tables slightly smaller.
- Risk: the leaves at p_w dominate the rest (P2M and L2P ∝ (p + 1)² per point), so the form
  saves only the M2M out and the L2L in of each raised level (RCR ∝ rotations at both degrees
  plus the coaxial step: about 0.7× the whole-level translation at 24/32); the gain may be
  small on uniform trees, larger on the deep ones. The report says so either way, and D8's
  question then rests on the numbers.
- Risk: the debug cost of `tests/helmholtz_fmm.rs` (one more FMM per resolving scenario, about
  1,000 points each).

## Results

The report `fmm-validate/results/helmholtz-ch6-1h.md`. Kathleen jobs 632717 (CH6.1g's 26 cases
in both orders, `e7` and `e8`, 706 s of runs) and 632725 (`e7x`, eight more W4 cases at 10⁻⁷,
forward only, 125 s), one node 1 × 40 (node-c11a-031), 28 launches, all exit 0; raw outputs in
`~/projects/fmm-archive/kathleen/logs/ch6-1h/`. `Leaves`: `Level`'s errors (at most 0.81×),
0.93–1.03× its evaluation. `M2l`: the rest 1.03–1.35× Rotation's (`Level` 1.36–1.88×; the cases run in both orders), the
evaluation 0.66–0.96× `Level`'s; errors 0.11–1.00× at 10⁻⁷ on all 17 cases, at 10⁻⁸ 1.53× (φ)
on the clusters at kD = 10 (fails), 0.06–1.00× on the other 16. Recommended (report §5): `M2l`
at 10⁻⁷ ≤ ε < 10⁻⁶ and `Leaves` at 10⁻⁸ ≤ ε < 10⁻⁷, and D8's bound lifted at 10⁻⁷.

## Pending runs

None.

## Open questions

1. **The leaves at p_l** (2026-11-11, before the Kathleen runs; not a question for the user).
   A temporary change (P2M and L2P of a raised level's leaves at p_l on the first degrees,
   not committed), the M6, release, 4 threads, errors only, φ/∇φ over Rotation's:

   | case | `Level` | `Leaves` | leaves at p_l |
   | --- | --- | --- | --- |
   | W4 cube N = 10⁵, kD = 1, 10⁻⁸ | 0.008/0.005 | 0.008/0.005 | 0.407/0.592 |
   | W4 clusters N = 10⁵, kD = 1, 10⁻⁷ | 0.019/0.019 | 0.019/0.019 | 0.899/0.961 |
   | W1, kD = 100, 10⁻⁷ | 0.803/0.545 | 0.809/0.574 | 0.816/0.645 |
   | W4 Plummer N = 10⁵, kD = 1, 10⁻⁸ | 0.003/0.005 | 0.003/0.005 | 0.247/0.745 |

   Within 1.1× on all four, against the reading of CH6.1f §4, so `PlaneWaveRaise::M2l` was
   built and joins the runs. Its margin is thin (the test trees: 0.98–0.99× at 10⁻⁷, 0.56–1.04× at 10⁻⁸).
2. **The form by ε** (report §5, question 1): **signed off 2026-11-11, "M2l at 1e-7, Leaves at
   1e-8 (Recommended)"**. Built: `PlaneWaveRaise::Rule`, the default (`resolved`: `M2l` for
   ε ≥ 10⁻⁷, `Leaves` below).
3. **D8's bound at 10⁻⁷ ≤ ε < 10⁻⁶** (question 2): **signed off 2026-11-11, "Lift it
   (Recommended)"**. Built: Auto's kept band removed; `check_resolving_degree` checks Auto
   bit for bit PlaneWave's output at 10⁻⁷ and 10⁻⁸; proposal §5.3 and the D8 row amended.
