# CH0.4 S4: plane-wave quadratures per k w_l

Started 2026-11-02 on `tbetcke/h-ch0-4`. Roadmap: "CH0.4 S4: plane-wave quadratures per
k w_l" (H0; H6 priority list item 1). Proposal: §10 CH0.4; §5.3 (Exponential M2L), §5.5
(pruning), §5.6, §8 R7; CONVENTIONS §3.14 (the Laplace scheme this generalises).

## Goal

Find out how FMM3D obtains the quadratures of its Helmholtz exponential M2L, and give this
library a table or a generator of them whose accuracy is measured over the k w_l of the
levels that would use it.

## Acceptance (proposal §10, verbatim)

> how FMM3D obtains them, and a generator or table for this library with its accuracy over
> the k w_l of low-frequency levels

## Read first

- CONVENTIONS §3.14 and `spikes/plane-wave/` (Phase 6 T10: FMM3D's Laplace tables
  `lwtsexp_sep1.f` extracted with NIST's notice, `tools/extract_quadrature.sh`; the region
  1 ≤ z ≤ 4, x² + y² ≤ 32 in box widths).
- FMM3D v2.1.0 (the release `tools/fmm3d/build.sh` pins, tarball under
  `$FMM3D_ROOT/src`): `src/Helmholtz/hwts3e.f` (`hwts3e`, `hwts3dgetd`, the node maps of
  `iquadtype` 1, 2 and 5), `hwts_3d_iprec{0..4}.txt` (the tables), `hnumfour.f` and
  `hnumphys.f` (Fourier and α counts per node), `hfmm3d.f` lines 750–770 and 1050–1130
  (which levels use plane waves: Re(k w) ≤ 16π, Im(k w) < 12π, finer than the levels
  where e^{−Im k w}/w < ε), `hpwrouts.f` (`hmkexps`: t = √(λ² + k²) per node, α_j = 2πj/N;
  `hrlscini`: the multipole-to-wave factors).
- Proposal §5.5's pruning test, which bounds the Im(k w) a level can reach.

## Changes

A spike only; no workspace crate, convention, table or Laplace path changes.

- `spikes/pw-quadrature/` (new, package `nd-fmm-spike-pw-quadrature`), a workspace member
  outside `default-members` like the other spikes; dependencies `num-complex` and `rayon`,
  both already in `[workspace.dependencies]`; no new dependency.
  - `src/fmm3d.rs`: a parser of FMM3D's `hwts_3d_iprec*.txt`, `hnumfour.f` and
    `hnumphys.f` (per precision and domain: the domain's rectangle in k w, the quadrature
    type, the stated error, the raw nodes and weights, the Fourier and α counts), the
    domain lookup of `hwts3dgetd`, and the node maps of `hwts3e` to (λ_k, w_k), with
    t_k = √(λ_k² + (k w)²).
  - `src/quadrature.rs`: a quadrature as (λ_k, t_k, w_k, N_k, F_k); its evaluation of
    e^{ik w|u|}/|u| (u in box widths) on the target region, and the point-to-point chain:
    a point source s in the source box, its waves per node truncated to the Fourier
    orders |m| ≤ F_k and sampled at N_k α nodes, evaluated at a target t in a target box
    of the up direction (offset 2 or 3 along z, up to 3 across).
  - `src/generator.rs`: an independent generator for any complex k w: the Sommerfeld
    integral on the contour of real t (λ = √(t² − (k w)²), Re λ ≥ 0), composite
    Gauss–Legendre panels graded towards t = Re(k w), truncated where the evanescent tail
    is below ε, and per node the fewest α nodes (and Fourier orders) that meet ε on the
    region. No compression (a generalised Gaussian quadrature) in this spike: its counts
    are reported against FMM3D's as the cost of a table-free path.
  - `src/region.rs`: the sample points (deterministic corners and edges of the region, and
    seeded uniform samples) and the error norms.
  - `examples/sweep.rs`: the sweeps below, writing Markdown tables to `results/`.
  - `fixtures/`: a few FMM3D domains (iprec 2 and 3, a real, an imaginary and a complex
    rectangle) extracted with NIST's notice by `tools/extract_fixture.sh`, so the parser
    and the evaluator are tested without FMM3D's sources.
  - `tools/fetch.sh`: unpacks the pinned tarball from `$FMM3D_ROOT/src` (SHA-256 checked)
    for the full sweeps and the ignored tests. As built: into `$FMM3D_ROOT/build/s4`, not
    `target/`, since `tools/fmm3d/build.sh` keeps FMM3D's sources outside the repository.
  - `SPIKE_REPORT.md`.
- Root `Cargo.toml`: the spike in `members`; `Cargo.lock` gains the spike's package entry.

## Tests that prove it

- Parser (fixture): the fixture domains' rectangles, counts and stated errors; the domain
  lookup of `hwts3dgetd` on rectangle corners and interiors; with FMM3D's sources
  (`#[ignore]`d, `tools/fetch.sh`): 437 domains per precision, every node count matching
  its `nfour`/`nphys` lists, the iprec thresholds of `hwts3e`.
- Normalisation and maps: at k w → 0 FMM3D's Helmholtz quadrature (domain 1) against
  1/|u| and against the Laplace §3.14 evaluation of `spikes/plane-wave` on the same points
  (each within its stated error); the node maps of type 1 and type 2 reproduce
  e^{ik w|u|}/|u| at one real and one complex k w within the stated error.
- Generator: at k w ∈ {0, 5, 20, 16π} real, i·{5, 30}, 10 + 3i and a damped 40 + 20i,
  e^{ik w|u|}/|u| within the requested ε ∈ {10⁻³, 10⁻⁶, 10⁻⁹} over the region (the
  direct kernel as the trusted path); its λ quadrature against a converged one (doubled
  panels) on the same points.
- Chain: with F_k = ∞ (all orders) the chain equals the kernel evaluation to round-off;
  with FMM3D's F_k and N_k it meets the table's precision on the sample.
- Error norm: absolute, relative to the largest |G| on the region (the norm FMM3D's
  tables are checked in, to be confirmed against their stated errors); for real k also
  the pointwise relative error. All in f64, one rank, threads from rayon (the norms are
  maxima, so no result depends on the thread count).

## Checks

`/verify` (the spike is outside `default-members`: format; workspace Clippy all-targets;
`cargo test -p nd-fmm-spike-pw-quadrature` and `--workspace`); the ignored tests with
FMM3D's sources by hand; `cargo deny --locked check licenses` (the lock file changes).

## Machine runs

None besides the M6, and no timing: the sweeps measure accuracy only.

- `cargo run --release -p nd-fmm-spike-pw-quadrature --example sweep`: (1) every FMM3D
  domain × precision at 9 points of its rectangle (corners, edge midpoints, centre),
  kernel and chain errors against the stated error and against `hwts3e`'s `errmax`;
  (2) fine sweeps along real k w ∈ [0, 16π], imaginary k w ∈ i[0, 12π] and two damped
  rays, per precision; (3) the k w_l of the reference workloads' levels (W1–W4, D1's
  kD ≤ 500, real, Yukawa to the pruning cut of §5.5, complex), which domain serves each
  and with what error, and which fall outside FMM3D's domain; (4) the generator at the
  same points, its node and wave counts (s, Σ N_k, Σ (2F_k + 1)) beside FMM3D's.

## Constraints and risks

- Constraint 1: no Laplace path is touched (a new spike crate; the root manifest's
  `members` only). Constraint 5 (tables in f64 on the host) and 9 (decisions from k, the
  geometry and ε only) bind CH6.1's use of the result; the spike records how a level's
  quadrature is chosen from k w_l and ε alone.
- Licence: FMM3D is Apache-2.0; the table files carry NIST's notice, which every extract
  keeps (the precedent of §3.14's `quadrature_data`). The spike commits only a few domains.
- Risk: for complex k with Im(k w) large the waves grow with Im t, so the norm relative to
  the region's largest |G| hides a pointwise loss; reported as such.
- Risk: the generator's uncompressed counts may be several times FMM3D's; that is the
  measured cost of a table-free path, not a failure of the spike.

## As built

- The ray sweep runs the generator at FMM3D's measured error (at least 1 % of the
  precision's `errmax` and 10⁻¹³) and only at iprec 1–3: at iprec 4, near 10⁻¹³, it took
  minutes per κ, and D1's ε needs iprec ≤ 3.
- `examples/alpha.rs` added: FMM3D's α counts scaled at the worst domains (report §3.2).
- The generator's Fourier and α counts take the maximum over all larger orders: the
  first version stopped at a zero of J_{F+1} and its chain missed ε by up to 120× on the
  rays; the four worst κ are regression cases.

## Pending runs

None.

## Open questions

For sign-off in the PR (report §6):

1. CH6.1's quadratures: FMM3D's tables (iprec 1–3) extracted with NIST's notice, as
   §3.14's Laplace tables; or the generator (1.5–1.9× the wave values on real κ, more on
   Yukawa).
2. The α counts by measurement, not from `hnumphys` (FMM3D's miss the stated error at the
   region's corner by up to 119×, and ε on 11 of 138 reference rows by up to 1.8×).
3. The proposal's correction below.

- The proposal says FMM3D uses plane waves "up to boxes of 32 wavelengths" (§5.3, §9.2,
  §11, §13), after FMM3D's documentation (`docs/index.rst`). Its code (`hfmm3d.f`,
  `zkrupbound = 16*pi` on `zk*boxsize(ilev)`, the box width) stops at Re(k w) ≤ 16π, boxes
  of 8 wavelengths. A correction for sign-off in the PR.
