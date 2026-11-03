# CH6.1b The generator

Started 2026-11-03 on `tbetcke/h-ch6-1b`. Roadmap: "CH6.1b The generator" (H6; priority
list item 2; CH6.1b split on this branch into the generator, CH6.1b, and the pair chain,
CH6.1c, with merge and shift renamed CH6.1d). Proposal: §10 CH6.1; §5.3 (Exponential M2L),
§8 R7; S4's report `spikes/pw-quadrature/SPIKE_REPORT.md` §4 and §6 and its sign-off 1
(CH0.4, 2026-11-03: FMM3D's tables, the generator as the tests' trusted path).

## Goal

Give `nd-fmm-ref` S4's quadrature generator, so that CH6.1c can test its pair chain with a
second, independently made quadrature for any κ = k w, and say when the generator stops
short of ε.

## Acceptance (proposal §10, verbatim)

CH6.1's: "errors within about 1.1× Rotation's, as T13's budget for Laplace". CH6.1b
carries no part of it alone; its own condition (the roadmap's split):

> the spike's tests hold in the crate: ε on the region's grid, within 2ε at seeded points
> off it and in the point-to-point chain, every tₖ real, the panels converged, at the
> spike's κ and ε (debug: its five; release, ignored: its full set), and the generator's
> quadratures equal the spike's bit for bit at those κ; the result says when it stopped at
> a bound short of ε.

## Read first

- `spikes/pw-quadrature/src/generator.rs`, `bessel.rs`, `gauss.rs`, `quadrature.rs`
  (`evaluate`, `kernel_errors_with`, `lambda_errors`, `chain_sums`), `region.rs`; its
  `tests/generator.rs`; the report's §1 and §4.
- `fmm-ref/CLAUDE.md`, `fmm-ref/src/helmholtz/mod.rs` (the module's layout and docs).
- `nd_fmm_tables::helmholtz::plane_wave` (CH6.1a: the representation, the count rule for
  complex tₖ, which this generator's rule is the real-t case of).

## Changes

`nd-fmm-ref` only, plus the spike's comparison example. No Laplace path, no convention, no
dependency (nd-fmm-math's `Complex`), no `nd-fmm-plan` change.

- `fmm-ref/src/helmholtz/plane_wave.rs` (new; `pub mod plane_wave` in `helmholtz/mod.rs`),
  f64 only (a builder of quadratures, as `nd-fmm-tables` builds its tables in f64):
  - `Quadrature` (generated): κ, per node λₖ, the real tₖ, wₖ, Nₖ and Fₖ; `nodes`,
    `wave_count` (Σ Nₖ), `mode_count` (Σ (2Fₖ + 1)); the bound it stopped at, if any
    (`Shortfall`, below);
  - `generate(κ, ε) -> Quadrature`: S4's four steps in the spike's order of operations,
    so its values are the spike's bit for bit;
  - `Quadrature::evaluate(u)`: the quadrature at one offset term by term, with its α
    counts; `Quadrature::kernel_error()`: the largest |Q − G| on the region's grid (the
    spike's `kernel_errors_with`, the absolute norm);
  - `kernel(κ, u)`, e^{iκ|u|}/|u|; the constants `RHO_MAX`, `SOURCE_REACH`, `ORDER`,
    `FLOOR`, `MAX_PANELS`;
  - private: the panel rule, the samples, the compression, the tail count, Jₙ of real
    argument by Miller's recurrence (the spike's `bessel::j_all`) and the Gauss–Legendre
    rule (the spike's `gauss::legendre`, kept apart from `direct`'s `gauss_legendre`,
    whose stopping rule differs in the last bits, so that the port stays bit for bit).
  - **The shortfall.** Where the spike returns its best quadrature at a bound without
    saying so (S4 §4), `generate` records the kernel's error it reached and the bounds
    its final attempt hit, `Shortfall { error, bounds: Vec<Bound> }` with
    `Bound::{Floor, Panels, Compression, Tightening}` (ε below 10⁻¹³ raised to it; the
    panel cap; the compression's tolerance floor; the last tightening of the λ rule's
    target or the counts' share); `Quadrature::shortfall()` is `None` when the requested
    ε was met on the grid. The bookkeeping reads values the spike computes anyway and
    never feeds back.
- `fmm-ref/tests/helmholtz/plane_wave.rs` (new, in the `helmholtz` test binary): the tests
  below, with the chain as a test helper (the spike's `chain_sums`).
- `spikes/pw-quadrature/examples/generated.rs` (new; the spike gains a dependency on
  `nd-fmm-ref`): at the spike's κ and ε, `nd_fmm_ref`'s quadrature against the spike's
  `generate`, every λₖ, tₖ, wₖ, Nₖ, Fₖ bit for bit; output in the results below.
- `fmm-ref/CLAUDE.md`: the module in the purpose list; its rule (f64 only, the spike's
  order of operations).

## Tests that prove it

In `fmm-ref/tests/helmholtz/plane_wave.rs`, f64, one thread, the kernel e^{iκ|u|}/|u| (the
integrand's own closed form) as the trusted path, errors absolute in box widths (FMM3D's
norm, S4 §1):

1. **ε met** (the spike's `meets_eps`, debug): at κ = 0 (10⁻⁶), 5 (10⁻³), 5i (10⁻⁶),
   10 + 3i (10⁻³) and the regression κ 17.529 + 0.876i (1.4·10⁻⁴): on the grid
   z ∈ {1, 1.25, …, 4}, x, y ∈ {−4, −3.5, …, 4} within ε; at 200 seeded points off it
   within 2ε; the point-to-point chain (sources at the box's centre, corners and two
   seeded points, waves truncated to |m| ≤ Fₖ, the 15 × 15 × 5 targets of the up direction)
   within 2ε; every tₖ real to 10⁻¹⁰ relative; no shortfall.
2. **The full set** (ignored, release): κ ∈ {0, 5, 20, 16π, 5i, 30i, 10 + 3i, 40 + 20i,
   10 + 10⁻³i} × ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹}, and the three regression cases of the spike's
   `meets_eps_everywhere`, by the same checks.
3. **The panels converge** (the spike's `panels_converge`): at κ = 10, ε = 10⁻⁶ the λ rule
   with twice the panels differs by at most ε/4 at nine (z, ρ).
4. **The shortfall is reported** (unit tests in the module, with a private panel cap):
   one panel per segment at κ = 2 and ε = 10⁻¹⁵ gives a shortfall with `Floor`, `Panels`
   and `Tightening`, its error the grid's above ε; a λ rule capped at one panel says so
   (κ = 20, 10⁻⁹); a quadrature that meets ε (κ = 5, 10⁻³) has none. `Compression` has no
   test of its own: no cheap case hits the compression's floor with its error above
   target.
5. **Bessel and Gauss** (unit tests in the module, the spike's): Jₙ against the trapezoid
   to 10⁻¹⁴; Gauss–Legendre exact on polynomials to degree 2n − 1.
6. **Bit for bit the spike** (by hand, `cargo run --release -p nd-fmm-spike-pw-quadrature
   --example generated`): every node at the κ and ε of tests 1 and 2.
7. **Determinism**: two calls give the same bits (λₖ, tₖ, wₖ, the counts).

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`, `cargo
test` and `--workspace`); `cargo test -p nd-fmm-ref --release --test helmholtz plane_wave
-- --ignored`; the spike's Clippy and its example. Not needed: the MPI list and D9 (no MPI
crate, no Laplace path); the licence check (no dependency of a default member changes;
run anyway, it is cheap).

## Machine runs

None: accuracy only, on the M6 (no timing).

## Constraints and risks

- Constraint 1: no Laplace file changes; `nd-fmm-ref`'s Laplace modules untouched.
- Constraint 6: this is the trusted path of CH6.1c's quadratures, independent of FMM3D's
  tables and of CH6.1a's counts (which share S4's rule, for complex tₖ there).
- Constraint 9: a function of κ and ε only.
- Debug time: the spike's five cases took 13 s in debug on the M6; the crate's debug run
  stays well under a minute.
- Bit identity with the spike needs the same order of operations, including the
  Gauss–Legendre rule; any deliberate change (the shortfall's bookkeeping) must not touch
  the arithmetic.

## Results

Release builds on the M6, f64, one thread, 2026-11-03; accuracy only, nothing timed but
the runs' wall times.

**Bit for bit the spike** (`cargo run --release -p nd-fmm-spike-pw-quadrature --example
generated`, 14 s, run again on the final branch with the same output): at every κ and ε of
the spike's tests (`meets_eps` and `meets_eps_everywhere`; the first five rows are the
debug test's), every λₖ, tₖ, wₖ, Nₖ and Fₖ of `nd_fmm_ref` equals the spike's `generate` bit
for bit, and none has a shortfall. The kernel error is the largest |Q − G| on the region's
grid in box widths:

| κ | ε | nodes | Σ Nₖ | Σ (2Fₖ + 1) | kernel error | bit for bit |
| --- | --- | --- | --- | --- | --- | --- |
| 0+0i | 1.0e-6 | 35 | 1825 | 775 | 8.76e-8 | yes |
| 5+0i | 1.0e-3 | 26 | 1027 | 458 | 2.30e-4 | yes |
| 0+5i | 1.0e-6 | 34 | 1938 | 734 | 8.43e-9 | yes |
| 10+3i | 1.0e-3 | 25 | 1146 | 433 | 2.97e-5 | yes |
| 17.529+0.876i | 1.4e-4 | 48 | 3962 | 1548 | 1.15e-5 | yes |
| 0+0i | 1.0e-3 | 19 | 530 | 235 | 1.48e-4 | yes |
| 0+0i | 1.0e-6 | 35 | 1825 | 775 | 8.76e-8 | yes |
| 0+0i | 1.0e-9 | 47 | 3775 | 1569 | 1.11e-10 | yes |
| 5+0i | 1.0e-3 | 26 | 1027 | 458 | 2.30e-4 | yes |
| 5+0i | 1.0e-6 | 41 | 2456 | 1081 | 1.37e-7 | yes |
| 5+0i | 1.0e-9 | 58 | 4620 | 2046 | 1.13e-10 | yes |
| 20+0i | 1.0e-3 | 56 | 5352 | 2022 | 1.27e-4 | yes |
| 20+0i | 1.0e-6 | 68 | 7717 | 3136 | 1.15e-7 | yes |
| 20+0i | 1.0e-9 | 80 | 10489 | 4398 | 1.91e-10 | yes |
| 50.26548245743669+0i | 1.0e-3 | 126 | 25273 | 8458 | 7.90e-5 | yes |
| 50.26548245743669+0i | 1.0e-6 | 136 | 30306 | 10836 | 2.33e-8 | yes |
| 50.26548245743669+0i | 1.0e-9 | 147 | 35514 | 13297 | 2.94e-11 | yes |
| 0+5i | 1.0e-3 | 20 | 520 | 202 | 1.85e-5 | yes |
| 0+5i | 1.0e-6 | 34 | 1938 | 734 | 8.43e-9 | yes |
| 0+5i | 1.0e-9 | 51 | 4323 | 1655 | 4.36e-12 | yes |
| 0+30i | 1.0e-3 | 24 | 24 | 24 | 9.35e-14 | yes |
| 0+30i | 1.0e-6 | 33 | 33 | 33 | 9.36e-14 | yes |
| 0+30i | 1.0e-9 | 38 | 38 | 38 | 9.36e-14 | yes |
| 10+3i | 1.0e-3 | 25 | 1146 | 433 | 2.97e-5 | yes |
| 10+3i | 1.0e-6 | 40 | 2805 | 1126 | 2.41e-8 | yes |
| 10+3i | 1.0e-9 | 54 | 5056 | 2076 | 4.94e-11 | yes |
| 40+20i | 1.0e-3 | 22 | 22 | 22 | 2.49e-9 | yes |
| 40+20i | 1.0e-6 | 35 | 35 | 35 | 2.49e-9 | yes |
| 40+20i | 1.0e-9 | 86 | 4520 | 1488 | 1.35e-11 | yes |
| 10+0.001i | 1.0e-3 | 34 | 2013 | 834 | 1.83e-4 | yes |
| 10+0.001i | 1.0e-6 | 48 | 3674 | 1600 | 1.99e-7 | yes |
| 10+0.001i | 1.0e-9 | 64 | 6150 | 2676 | 1.18e-10 | yes |
| 28.524+14.262i | 1.5e-10 | 76 | 7691 | 2688 | 1.79e-12 | yes |
| 37.949+1.897i | 1.1e-3 | 79 | 10485 | 3517 | 4.50e-5 | yes |
| 6.533+0i | 1.9e-7 | 47 | 3234 | 1429 | 2.31e-8 | yes |

The rows of strongly damped κ (30i; 40 + 20i at 10⁻³ and 10⁻⁶) have Nₖ = 1 per node and
an error near round-off: there |G| ≤ e^{−Im κ} is below ε on the whole region, and the
counts' bound gives each node a single α point.

**The acceptance** (`cargo test -p nd-fmm-ref --release --test helmholtz plane_wave --
--ignored`): `the_generator_meets_eps_everywhere` passed in 23.5 s; with the debug run's
`the_generator_meets_eps` (17 s in debug), S4's checks hold at every κ and ε above. (A first
version had an ignored test of ε below the floor at full panels: 6 minutes, the generator
near 10⁻¹³. It became the unit test with one panel per segment.)

**Debug test time** (`cargo test -p nd-fmm-ref`, the M6): the helmholtz test binary 22.2 s
(21.2 s with the new tests skipped), the lib's unit tests 24.7 s (0.01 s without them; the
forced-bound test runs the generator's give-up path, ten grid sums with converged α
counts).

## Pending runs

None.

## Open questions

None yet.
