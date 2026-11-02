# CH3.2d The M2L rule per level

Started 2026-11-02 on `tbetcke/h-ch3-2d`. Roadmap: "CH3.2d The M2L rule per level" (H3;
added on 2026-11-02 at CH3.2c's sign-off, question 1: "a follow-up before H3.G"; taken
ahead of CH0.6 by the user, 2026-11-02, "CH3.2d"). Proposal: §10 CH3.2; §5.3, §6.4
(strategy selection); §2 (constraints 1, 2, 7, 9).

## Goal

An `Auto` setting of `HelmholtzM2l` that runs the fused Batched M2L on the levels whose
degree is at most a crossover degree of the machine class and whose tables fit the budget,
and Rotation on the others, with the crossovers measured on the M6, locust and Kathleen.

## Acceptance

Proposal §10 CH3.2, verbatim: "measured against Rotation per p, including the cost of the
cube-group transforms that move the z-axis" (CH3.2a–c met it). The roadmap's *Done when*
of CH3.2d: "the rule's choice against the best of Rotation and fused Batched end to end on
CH3.2a's cases on the three machines, within a stated margin; deterministic and
rank-invariant as CH3.2c's FMM; whether it becomes the default is a question for sign-off."

The stated margin (proposed here): on every case and machine, the rule's evaluation time
at most 1.10 × the best of the per-level choices the rule could make (every threshold
below), the times the mean of two launches in opposite orders; where the two orders differ
by more than 10 % on a case, the margin is that spread, stated.

"The best of Rotation and fused Batched" is taken per level, not only as the two uniform
settings: for each case the sweep runs every distinct threshold d (Batched on the levels
with tables and p_l ≤ d, Rotation above), from Rotation everywhere (d below every p_l) to
Batched on every level that fits (d at or above every p_l). Since p_l falls with the level,
these are the assignments "Batched on the finest j levels" that the budget rule makes, and
they include both uniform settings.

## Read first

- `fmm-exec/src/helmholtz/operator.rs`: `HelmholtzM2l`, `with_m2l`, `with_p2p`,
  `with_m2l_mul_add`, `m2l_on`, `Scratch::new`; `fmm-exec/src/helmholtz/batched.rs`.
- `fmm-exec/src/helmholtz/fmm.rs`: the builder's `m2l`, `dense_budget`, `m2l_mul_add` and
  `build` (the operator's construction and the agreements).
- `fmm-tables/src/helmholtz/batched.rs` and `dense.rs`: `select`, `level_bytes`.
- `fmm-validate/src/helmholtz_accuracy.rs` (`Case`, `Row`), examples `helmholtz_accuracy`
  and `helmholtz_m2l`.
- Reports `fmm-validate/results/helmholtz-ch3-2c.md` (the per-pair crossovers, the end-to-end
  cases); records `docs/helmholtz/tasks/CH3.2c.md`.
- Laplace's rule for comparison: `M2lStrategy::auto_plane_wave_min_p`,
  `fmm::default_m2l_mul_add` (the `fmm-exec/CLAUDE.md` section "The defaults of Phase 6 T11").

## Changes

`nd-fmm-tables` (the `helmholtz::batched` module), `nd-fmm-exec` (the Helmholtz modules
only), `nd-fmm-validate` (the two examples, the harness, a report). No Laplace file, no
`driver.rs`, `nd-fmm-simd` or `nd-fmm-plan` change, no convention, no dependency.

- `nd_fmm_tables::helmholtz::batched::BatchedM2l::select_within(degrees, budget,
  max_degree)`: `select` over the levels with p_l ≤ `max_degree` only (from the finest level
  up, the others never chosen); `select` is `select_within(.., usize::MAX)`.
- `nd_fmm_exec::helmholtz`:
  - `HelmholtzM2l::Auto`: per level, Batched where `select_within` chooses the level for the
    crossover degree, Rotation elsewhere; `m2l_on(level)` reports `Batched` or `Rotation`,
    `batched_bytes()` the tables.
  - The crossover, a function of the machine class only (constraint 9): the column
    kernel's ISA (the P2P choice's, `Isa::Scalar` for `Reference`) and the multipole planes
    (one for real k with real charges, two otherwise; a function of `Densities`):
    `pub fn auto_batched_max_degree(isa: Isa, planes: usize) -> usize`, with the values of
    the measurements (below). CH3.2c's per-pair crossovers suggest about 10 (two planes) and
    12 (one) on NEON, 15 and 18 on AVX-512; AVX2 and the scalar kernel are measured here.
  - `HelmholtzOperator::with_m2l_max_degree(Option<usize>)` and the builder's
    `m2l_max_degree(usize)`: name the crossover (for the measurements and the tests; `None`
    the rule's). `m2l_max_degree()` reports the one in use.
  - The multiply–add mode under Auto: fused (CH3.2c: 1.2–1.8× the separate mode, Dense to
    rounding either way) unless named. The builder keeps `m2l_mul_add(MulAdd)` and stores an
    `Option`: unnamed, `Fused` under Auto, `Separate` otherwise (as before). The operator's
    `with_m2l(Auto, ..)` sets the fused mode; a later `with_m2l_mul_add` overrides it.
  - `with_p2p` after `with_m2l(Auto, ..)` selects again when the ISA's crossover differs
    (the tables of the new selection rebuilt), so the order of the calls does not matter.
    The builder applies the P2P choice first.
  - Agreement (constraints 2 and 9): the builder agrees the crossover over the ranks
    (all-reduce min) before the operator selects, so every rank chooses the same levels
    even where `Isa::detect` differs between ranks.
  - Docs: the operator module's "M2L" section, the builder, `fmm-exec/CLAUDE.md`.
- `nd-fmm-validate`:
  - `helmholtz_accuracy`: `auto` in `--m2l`, and `sweep` (for each case, after its
    Rotation run, Auto at every distinct threshold: each distinct p_l of the levels ≥ 2, and
    one below them all); `Case::m2l_max_degree`, `Row::{m2l_max_degree, m2l_levels}` (the
    levels that ran Batched). The example prints one row per run with the threshold.
  - `helmholtz_m2l --isa <name>`: the column kernel and P2P choice of a named ISA, for the
    AVX2 and scalar crossovers.
  - Report `fmm-validate/results/helmholtz-ch3-2d.md`.

## Tests that prove it

- `fmm-tables/tests/helmholtz/batched.rs`: `select_within` against a brute-force choice
  over degrees and budgets (levels above the degree never chosen, from the finest up within
  the budget), `usize::MAX` equal to `select`, a degree below every p_l chooses nothing.
- `fmm-exec/tests/helmholtz/dense.rs` (pair by pair, every offset of levels 2–4):
  - Auto with a named crossover that splits the levels: each level's `m2l_on` as the rule
    says; on Batched levels bit for bit the operator under `Batched` in the fused mode, on the
    others bit for bit `Rotation`; real and complex k, real and complex charges, two vectors.
  - Auto with a crossover above every degree: `Batched` fused bit for bit; below every
    degree: `Rotation` bit for bit, no tables.
  - The mode: fused by default under Auto, `Separate` when named; `with_p2p` after
    `with_m2l(Auto, ..)` gives the selection of the other order.
  - `auto_batched_max_degree` monotone in the planes (one plane at least two planes').
- `fmm-exec/tests/helmholtz_fmm.rs` (CI's MPI list): `check_tables` with Auto at a crossover
  that splits the scenario's levels: against the direct sum, Rotation's output within
  `DENSE_TOLERANCE`, 2 threads, pair by pair and the one-rank FMM bit for bit; the builder's
  default mode under Auto fused, named separate kept; the agreed crossover reported alike on
  every rank.
- `fmm-validate/tests/helmholtz_accuracy.rs`: a small case with `auto` and the sweep's
  thresholds (the harness).
- f64 (D3).

## Checks

`/verify`: format; Clippy default, examples and workspace all-targets; `cargo test` and
`--workspace`; `cargo doc`; the MPI list at 2 and 4 ranks (`helmholtz_fmm` changes);
`cargo test -p nd-fmm-tables`, `-p nd-fmm-exec`, `-p nd-fmm-validate`. No licence check
(no dependency change). No D9 (Laplace untouched).

## Machine runs

Release, one rank, one thread, BLAS threads 1; both examples rebuilt right before.

- `helmholtz_m2l --ps 3,4,5,6,7,8,9,10,12,15,18,20 --isa <isa>`: Kathleen `avx2` and
  `scalar`, the M6 `scalar` (NEON and AVX-512 are CH3.2c's). Decides the AVX2 and scalar
  crossovers' starting values.
- `helmholtz_accuracy --m2l rotation,auto,sweep` and the reverse order on CH3.2a's cases
  (W1 at kD = 30 and 100, the W4 cube at kD = 10, each at ε = 10⁻³ and 10⁻⁶; W3 at kD = 30
  at 10⁻³), plus one-plane cases (`--charges real`: the W4 cube at kD = 10 at 10⁻⁶, W1 at
  kD = 100 at 10⁻³), on the M6 (after the user's go-ahead, behind CH3.2a's quiet gate),
  locust (`tools/gh200/sync.sh`, `remote.sh`; `uptime` first) and Kathleen (one
  `singlenode` job through `run.sbatch`). Decides the crossovers (the threshold whose
  evaluation is best, per machine class and plane count) and checks the margin.

If a case misses the margin because a sparse level (W1) gains less than its degree says,
the rule gains a term from the level's global geometry (its V pairs per target, agreed
over the ranks; constraint 9 allows the global tree geometry), and the change and its data
go into the PR.

## Constraints and risks

- Constraint 1: Laplace untouched (no Laplace file, no driver, `nd-fmm-simd` or
  `nd-fmm-plan` change).
- Constraint 2: Auto's levels are the Batched and Rotation level calls as they are, chosen
  identically on every rank (the agreed crossover, the degrees and the budget), so CH3.2c's
  determinism and rank invariance carry over; tested above.
- Constraints 7, 9: the budget as Dense's; the choice from the degrees, the budget and the
  machine class (the ISA and the planes) only, never the charges or a rank's boxes.
- Risk: the machine class as the ISA lumps machines of one ISA together (an AVX2 Zen with
  Kathleen's AVX2); the override names another crossover, and the report says what was
  measured where.
- Risk: the timings move with the M6's background load (CH3.2a's gate, the user's
  go-ahead first) and with locust's other users (`uptime`).

## Pending runs

None yet.

## Open questions

- Whether Auto becomes the default of `HelmholtzFmmBuilder` (for sign-off, in the PR).
