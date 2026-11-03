# CH6.1a The quadratures

Started 2026-11-03 on `tbetcke/h-ch6-1a`. Roadmap: "CH6.1a The quadratures" (H6; priority
list item 2, CH6.1 split into a/b/c on this branch). Proposal: §10 CH6.1; §5.3
(Exponential M2L), §8 R7; S4's report `spikes/pw-quadrature/SPIKE_REPORT.md` §6 and its
sign-offs (CH0.4, 2026-11-03): FMM3D's tables, the α counts by measurement, the generator
as the tests' trusted path.

## Goal

Give `nd-fmm-tables` the quadratures of the Helmholtz exponential M2L per level: FMM3D's
tables for iprec 1 to 3, the lookup by κ = k w_l and ε, and α and Fourier counts that keep
the stated error on the whole region.

## Acceptance (proposal §10, verbatim)

CH6.1's: "errors within about 1.1× Rotation's, as T13's budget for Laplace". CH6.1a
carries no part of it alone; its own condition (the roadmap's split):

> every domain's rectangle and node count as FMM3D's; the lookup on every rectangle and
> outside the tables; with the counts, the kernel e^{iκ|u|}/|u| on the region within 2× the
> stated error and the point-to-point chain within 3× it, the kernel at nine points of
> every domain and precision and the chain at every domain's centre (release; sampled in
> debug); the wave count Σ Nₖ against FMM3D's recorded.

(Amended on this branch from "at nine points of every domain" for both: the chain costs
about 10⁸ operations per quadrature, so nine points of every domain would take hours on
one thread, and this crate has no rayon. S4 measured its chain at the centres too.)

## Read first

- `spikes/pw-quadrature/`: `src/fmm3d.rs` (the parser, `domain_of`, `precision_of`, the
  node maps of types 1 and 2), `src/quadrature.rs` (the kernel and chain evaluation),
  `src/region.rs` (the grid and chain geometry), `src/generator.rs` (`tail_count`, the
  count rule), `src/bessel.rs`; the report's §2.1, §3 and §6.
- CONVENTIONS §3.14 (the region, the Laplace counts chosen by measurement) and
  `nd_fmm_tables::plane_wave::quadrature` (the Laplace counterpart, its data file and
  `tools/fixtures/plane_wave_quadrature.sh`).
- FMM3D v2.1.0 `src/Helmholtz/hwts3e.f` and `hwts_3d_iprec{1,2,3}.txt`.

## Changes

`nd-fmm-tables` only (plus the extraction script and the spike's comparison example). No
Laplace path, no convention, no dependency, no `nd-fmm-plan` change.

- `fmm-tables/src/helmholtz/plane_wave.rs` (new; `pub mod plane_wave` in
  `helmholtz.rs`): the module of the Helmholtz exponential M2L, for now its
  documentation and `quadrature`.
- `fmm-tables/src/helmholtz/plane_wave/quadrature.rs` (new):
  - `Precision` (iprec 1, 2, 3) and `Precision::for_tolerance(ε)`, `hwts3e`'s rule
    (ε ≥ 5·10⁻³ takes iprec 1, the coarsest shipped; ε < 5·10⁻¹⁰ none);
  - `domain_of(κ)` and `rectangle(domain)`, `hwts3dgetd`'s 23 × 19 rectangles;
  - `Quadrature::new(κ, precision) -> Option<Quadrature>`: the domain's nodes mapped to
    complex λₖ and wₖ at κ (types 1 and 2), tₖ = √(λₖ² + κ²), and per node the counts
    (below); its stated error; Σ Nₖ and Σ (2Fₖ + 1);
  - the counts, the generator's rule for complex tₖ: Fₖ the fewest orders with
    2|wₖ| e^{−Re λₖ} max_{m > Fₖ, ρ ≤ r_s} |J_m(tₖ ρ)| ≤ e/(4s), and Nₖ the fewest with the
    same bound for m ≥ Nₖ over ρ ≤ ρ_max, at least the count for ρ_max − r_s plus Fₖ
    (e the stated error, s the nodes, r_s = √2/2, ρ_max = 4√2); |J_m| of complex argument
    by Miller's backward recurrence, normalised by e^{iz} = J₀ + 2 Σ iⁿ Jₙ(z) at the
    conjugate with Im z ≤ 0 (no cancellation). Not FMM3D's `hnumphys`, which misses the
    stated error at the region's corner (S4 §3.2). The share may be tightened by
    measurement if the acceptance misses, and the rule says so.
- `fmm-tables/src/helmholtz/plane_wave/hwts_iprec{1,2,3}.txt` (new, generated): per
  domain its index, type, rectangle and stated error, then its nodes and weights, each
  value FMM3D's literal with the exponent letter D written as e; NIST's notice and a dated
  change note as the header. Read once through `include_str!` and parsed on first use
  (`LazyLock`). 1.68 MB.
- `tools/fixtures/helmholtz_pw_quadrature.sh` (new; POSIX `sh` and `awk`, as
  `plane_wave_quadrature.sh`): writes the three files from FMM3D's `src/Helmholtz/` (the tarball `tools/fmm3d/build.sh`
  pins, unpacked by `spikes/pw-quadrature/tools/fetch.sh`); `tools/fixtures/README.md`'s
  table row.
- `spikes/pw-quadrature/examples/counts.rs` (new; the spike gains a path dependency on
  `nd-fmm-tables`): per domain and precision Σ Nₖ and Σ (2Fₖ + 1) of `nd-fmm-tables`
  against FMM3D's `hnumphys` and `hnumfour`, and the nodes and weights bit for bit the
  spike's parse of FMM3D's own files. Output in the record below.
- `fmm-tables/CLAUDE.md`: the module's rule (the third exception, like `plane_wave`).

## Tests that prove it

All in `fmm-tables/tests/helmholtz/plane_wave_quadrature.rs` (the crate's `tests/helmholtz/`
binary), f64, no threads:

1. **The data.** Every precision has all 437 domains; each domain's stated rectangle
   equals `rectangle(domain)` to 10⁻¹⁵ relative, its type is 1 for Im κ₀ ≥ π/2 and 2
   below; the node totals per precision and, as FMM3D's files have them, the node counts
   and the first and last node and weight of domains 1, 11, 139, 172, 236 (the first at
   iprec 2 and 3, the last at iprec 1–3). Every value against FMM3D's own files, bit for
   bit: the spike's `examples/counts.rs` (by hand; it needs FMM3D's sources).
2. **The lookup.** `domain_of` at the centre, corners (pulled in by 10⁻⁹ relative) and
   edge midpoints of every rectangle gives that rectangle; Re κ ≥ 16π or Im κ ≥ 12π gives
   none; `Precision::for_tolerance` at the thresholds.
3. **The node maps.** Against the closed forms (type 1: tₖ = xₖ real; type 2:
   λₖ(u = 0) = −iκ), and the λ sum with α resolved to round-off against the kernel on the
   region within 1.5× the stated error (S4 §3.1) at the sampled domains.
4. **J_m of complex argument** against the trapezoidal integral (exact to round-off) for
   |z| to 400 and |Im z| to 40, every order to 2|z| + 80, relative to e^{|Im z|}.
5. **The counts** (the acceptance): on the grid of the region (z ∈ {1, 1.25, …, 4},
   x, y ∈ {−4, −3.5, …, 4}) the kernel with the counts within 2× the stated error, and the
   point-to-point chain (S4's sources and targets, the source waves truncated to |m| ≤ Fₖ)
   within 3× it. Debug: the kernel at the centres of the five fixture domains at iprec 1–3
   and the chain at iprec 2 (about a second each in debug). Ignored (release): the kernel
   at nine points of every domain and precision and the chain at every centre
   (`the_counts_on_every_domain`); the reference levels of S4 §5, kernel and chain
   (`the_counts_on_the_reference_levels`).
6. **Determinism**: two builds of the same (κ, precision) bit for bit; the result a
   function of κ and the precision alone.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo doc`, `cargo
test` and `--workspace`); `cargo test -p nd-fmm-tables --release -- --ignored`; the spike's
`cargo clippy -p nd-fmm-spike-pw-quadrature --all-targets` and its example run on the
directory `spikes/pw-quadrature/tools/fetch.sh` prints. Not needed: the MPI list and D9 (no MPI crate, no Laplace path);
the licence check (no dependency of a default member changes; run anyway, it is cheap).

## Machine runs

None: accuracy only, on the M6 (no timing).

## Constraints and risks

- Constraint 1: no Laplace file changes (`nd_fmm_tables::plane_wave` untouched).
- Constraint 9: the quadrature is a function of κ = k w_l and the precision from ε.
- Constraint 5: built in f64; the module holds f64 only (CH6.1b rounds to T).
- Risk: the bound's counts cost more waves than FMM3D's (the generator's were 1.5–1.9×
  FMM3D's Σ Nₖ for its own nodes); recorded, CH6.2 times it. The bound for complex tₖ
  may be loose where |Im tₖ| ρ is large (type 2 near Re κ = 16π).
- Data size: about 1.5 MB of text in the crate (NIST's, public domain in the US with the
  notice kept, as `plane_wave::quadrature_data`).

## Results

Release builds on the M6, f64, 2026-11-03; accuracy only, nothing timed.

**The counts against FMM3D's** (`spikes/pw-quadrature/examples/counts.rs`, at the centre,
the lower corner, the upper corner pulled in by 10⁻⁹ relative and the two mixed corners of
every domain, 2,185 points per precision; the nodes and weights are bit for bit the spike's
parse of FMM3D's files, the node maps the same to 10⁻¹⁵). The kernel's error on the grid of
the region relative to the stated error:

| iprec | kernel / stated, ours, worst | kernel / stated, FMM3D's `hnumphys`, worst | Σ Nₖ ours / FMM3D's, Im κ < π/2 (means per row) | Σ Nₖ ours / FMM3D's, all points, mean (range) |
| --- | --- | --- | --- | --- |
| 1 | 1.08 | 82.8 | 1.40–1.51 | 12.35 (0.99–119) |
| 2 | 1.19 | 94.3 | 1.35–1.39 | 10.17 (1.00–87) |
| 3 | 1.50 | 119.0 | 1.31–1.32 | 7.99 (1.00–87) |

The worst of ours is the λ sum's own error (S4 §3.1: 1.08, 1.19 and 1.50 at iprec 1–3), so
the counts add nothing visible on the grid. On the rows of D1's levels (Im κ small: real
and weakly damped k) the bound costs 1.31–1.51× FMM3D's waves: it resolves J₀ to the
region's corner ρ = 4√2, where FMM3D's counts resolve to about ρ = 4 (S4 §3.2), and adds Fₖ
for the chain's aliasing. The ratio grows with Im κ (1.42–1.86 for Im κ in [π/2, 2π)) and is
21–28 on the strongly damped rows (Im κ ≳ 3π at iprec 1, ≳ 5π at iprec 2, ≳ 7π at
iprec 3), where FMM3D's tables hold one node with one α node because |G| ≤ e^{−Im κ} is
near the precision there, and ours give that node 21–30; these rows set the means and the
largest ratios (119 against 1 wave at iprec 1, domain 299), small in absolute terms.
A looser share does not pay: e/(2s) and e/s per node cut the real-axis ratio only from
1.44 to 1.41 and 1.38 at iprec 1 (1.31 to 1.30 and 1.28 at iprec 3), since the Bessel
tails fall faster than exponentially; the rule keeps e/(4s).

**The acceptance on every domain** (`cargo test -p nd-fmm-tables --release --test
helmholtz plane_wave -- --ignored`, 380 s on the M6 beside other work): both ignored tests
passed. The worst over every domain, relative to its stated error:

| iprec | kernel, nine points of every domain | chain, every domain's centre |
| --- | --- | --- |
| 1 | 1.079 | 0.984 |
| 2 | 1.188 | 0.716 |
| 3 | 1.495 | 1.175 |

within the bounds 2 and 3; and `the_counts_on_the_reference_levels` (S4 §5's levels of real
kD 10, 100, 500, Yukawa κD 10 and 100, damped 100 + 5i and 500 + 25i, levels 2–8 where they
fall in the tables, at ε = 10⁻³, 10⁻⁶, 10⁻⁸: kernel and chain) passed.

## Pending runs

None.

## Open questions

None yet.
