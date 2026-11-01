# CH3.1b The Helmholtz FMM

Started 2026-11-01 on `tbetcke/h-ch3-1b`. Roadmap: "CH3.1b The Helmholtz FMM" (H3; the
second sub-task of CH3.1, tenth on the priority list). Proposal: §10 CH3.1; §2
(constraints 1, 2, 3, 9), §3.7, §6.1–§6.3.

The user decided on 2026-11-01 ("Yes, both as recommended", CH3.1a's plan): CH3.1b
extracts the operator-independent point pipeline of `fmm.rs` into crate-private code that
`Fmm` and the Helmholtz FMM share, with Laplace's outputs bit for bit and "minimal timing
runs to check the impact" for D9.

## Goal

A Helmholtz FMM for real and complex k with real and complex charges on any number of
ranks, built from CH3.1a's operator and the point pipeline it shares with `Fmm`.

## Acceptance (proposal §10, verbatim)

CH3.1: "FMM against the direct sum on W1, W3 and W4 for kD ≤ 100 and on W1 at kD = 500;
deterministic". That is CH3.1c's run. CH3.1b's own *Done when* (roadmap): against the
direct sum on small trees; bit for bit at every thread count and on P ranks the one-rank
FMM; the level calls the operator's `PerPair` bit for bit; Laplace's outputs bit for bit,
and minimal D9 timing runs to check the extraction's impact.

## Read first

- `fmm-exec/src/fmm.rs`: `FmmBuilder::build` (steps 1–8), `Fmm::run`, the helpers
  `agree`, `check_finite`, `check_inside`, `global_extent`, `LeafRanges`, `OutputOrder`,
  `load_source_charges` (and its split and strength helpers), `pack_charges`,
  `gather_output`, `unpack_output_many`; the module docs "Build", "Evaluation", "Several
  ranks".
- `fmm-exec/src/helmholtz/{mod, densities, degree, operator}.rs`.
- `nd_fmm_plan::{evaluator::Evaluator, operator::PerPair, redistribute::Redistribution}`.
- `nd_fmm_ref::helmholtz::p2p::direct_sum`; `fmm-exec/tests/helmholtz/common.rs`.

## Changes

One crate, `nd-fmm-exec`, plus the CI script's MPI list.

1. **`src/pipeline.rs` (new, crate-private): the point pipeline.** Moved from `fmm.rs`
   verbatim where they are already free functions: `agree`, `check_finite`,
   `check_inside`, `global_extent`, `LeafRanges`, `LOAD_GRAIN`, `OutputOrder`,
   `load_source_charges` with `ChargeInput`, `StrengthLayout`, `write_source_charges`,
   `write_leaf_strengths`, `split_source_charges`, `pack_charges`, `gather_output`,
   `unpack_output_many`. As built: the whole contiguous block of `fmm.rs` from `agree` to
   `unpack_output_many` moved, so the dipole and device helpers in it (`scale_moments`,
   `pack_sources`, `interleave_output`, `unpack_output`) moved too; their unit tests stayed
   in `fmm.rs`'s `mod tests`, which reaches them through its imports. `unpack_output_many`
   takes the output's two buffers instead of an `OutputMany` (same body). Extracted from
   `build` as
   functions of the same statements:
   - `domain` (step 3): the coordinates, the supplied or global domain, every point
     strictly inside it (agreed);
   - `octree` (step 4's first part): the Morton keys and `Octree::new` with the caller's
     `OctreeOptions` (Laplace's as now; Helmholtz adds `with_min_leaf_level`);
   - `Points::route` (steps 5 and 6): the two `Redistribution`s, the coordinate forward,
     the leaf ranges, the radii and the output order, with the overflow check of the
     values per item;
   - `Points::load_coordinates` (step 8): the leaf-scaled coordinates into an
     `Evaluator<'_, C, Op, Plan>` for any `Op: FmmOperator`.
   `Fmm` holds a `Points` in place of its six fields (`sources`, `targets`,
   `source_leaves`, `target_leaves`, `outputs`, `radii`). What stays in `fmm.rs`: settings,
   pool, device, the node split, step 2's all-reduce, the near rule and the work cut, the
   strategy, tables and operator, the device attachment, `run`'s stages and device
   branches, dipoles (`scale_moments` stays: device only). No statement changes its order
   or arithmetic; no public item of `fmm` changes.
2. **`src/helmholtz/fmm.rs` (new): `HelmholtzFmmBuilder` and `HelmholtzFmm`.**
   - Builder: `new(k: Wavenumber, degrees: Degrees)`, then `charges(ChargeKind)` (default
     complex), `vectors(m)`, `gradients`, `threads`, `domain`, `max_points_per_leaf`
     (default `fmm::DEFAULT_MAX_POINTS_PER_LEAF`), `max_level`, `block_budget`, `c_leaf`
     (default `C_LEAF`); `build(sources, targets, comm)`. Yukawa refused
     (`OperatorError::Yukawa`, CH2.1's path), agreed before any other collective.
   - Build: check (agreed); points or `NoPoints`; `pipeline::domain`; the leaf rule
     `min_leaf_level(k, domain, c_leaf)` into `OctreeOptions`; `pipeline::octree`;
     `Plan::new` (no near rule, no work cut: their cost model is Laplace's); `Points::route`;
     the agreed largest leaf; `Degrees::resolve(k, domain, plan.nlevels())`;
     `HelmholtzOperator::with_block_budget` (with the pool); `Evaluator::new`;
     `Points::load_coordinates`.
   - `evaluate(&[Charges<T>]) -> Result<Output<T>, HelmholtzFmmError>` and
     `evaluate_into`: `Densities::check_charges` (agreed); the c m columns packed point-major
     by `pack_charges` (Re q⁰, Im q⁰, …, the column slices); forward; `load_source_charges`
     with c m strengths; the six blocking stages; `gather_output` with o_m = the output
     columns; backward; `unpack_output_many` into `Output`'s split planes (`potential[o N
     + i]` is `OutputMany`'s layout). CH3.0c made every chunk a §3.13 chunk of c m source
     and o_m output vectors, so this is Laplace's charge load and output pass value for
     value, with Laplace's scales 4π r and 4π r².
   - Accessors: `octree`, `plan`, `operator`, `domain`, `degrees`, `nlevels`, `nleaves`,
     `min_leaf_level`, `max_leaf_points`. Doc-hidden for the tests: `pair_calls(true)` on
     the builder (the evaluator runs `PerPair` of the same operator, through a crate-private
     two-variant wrapper like `ExecOperator`'s engine) and `reference_output` (the output
     pass by `Densities::write_leaf_output` on one rank, the trusted path of CH3.0c).
   - `HelmholtzFmmError`: `Build(FmmError)` for the shared steps' errors, `Degree`,
     `Operator`, `Charges(DensityError)`, `OtherRank`. As built: `LeafWidth` for a c_leaf
     that is not positive and finite, and no `Wavenumber` variant (`Wavenumber::new`
     validates k before the builder sees it).
   - As built: the operator builds its tables serially and then takes the pool
     (`with_pool`), as CH3.1a's operator does; only the level calls run on it.
   - No overlap, device, near rule, work cut, dipoles or table cache (later tasks).
3. **`.github/scripts/run-mpi-tests.sh`**: the new executable `helmholtz_fmm` in the MPI
   list (2 and 4 ranks in CI); root `CLAUDE.md` and `fmm-exec/CLAUDE.md` name it.

No `nd-fmm-plan`, `nd-octree`, `nd-fmm-tables`, `nd-fmm-simd`, manifest, dependency or
convention change.

## Tests that prove it

`tests/helmholtz_fmm.rs` (new executable, its own MPI test at `Funneled`, any rank count;
each rank passes every P-th point, one scenario none on rank 0; domain [0, 1]³, at most 32
points per leaf). As built, six scenarios:

| Scenario | Points | Degrees | Bound | Builds |
| --- | --- | --- | --- | --- |
| cube, kD = 20 + 2i, complex charges, ∇φ, rank 0 empty | 1,000 | ε = 10⁻³ | 10 ε | debug |
| cube, kD = 1, complex charges, ∇φ | 1,500 | ε = 10⁻³ | 10 ε | debug |
| clusters, kD = 20, real charges, separate targets | 1,000 / 700 | p = 8 | 1e-3 | debug |
| cube, kD = 20 + 2i, complex charges, ∇φ | 1,500 | p = 8 | 1e-3 | debug |
| cube, kD = 20, complex charges, ∇φ | 2,000 | ε = 10⁻⁶ | 10 ε | release |
| clusters, kD = 100, complex charges | 3,000 | ε = 10⁻⁶ | 10 ε | release |

(p = 8 on every level where truncation dominates: the bound about 5–13× the measured
7.5e-5 and 2.2e-4. S7's closed form at ε = 10⁻³ gives p = 16 on low-frequency levels,
too slow in debug for the kD = 20 trees.) Per scenario:
- against the direct sum (`nd_fmm_ref::helmholtz::p2p::direct_sum`, f64, 1/(4π), over
  every rank's sources), relative L2 of φ and ∇φ over every rank's targets;
- the output pass bit for bit `reference_output` (`Densities::write_leaf_output`);
- 4 threads (2 and 4 in release) bit for bit one;
- `pair_calls(true)` (`PerPair`) bit for bit the level calls;
- on P ranks the one-rank FMM over the union in rank order (`self_comm`, built on every
  rank) bit for bit.
Then: two vectors of complex and of real charges, each bit for bit its one-vector FMM, the
second against the direct sum, and `evaluate_into` replacing an output of another path and
reusing it; real charges against complex ones with Im q = 0 (0 difference measured, bound
1e-13); the errors on every rank alike: Yukawa, ε = 2, c_leaf = 0, no points, a point that
is not finite on rank 0 only (`OtherRank` elsewhere), no vector, the wrong kind, a short
vector on rank 0 only, and an evaluation after them. About 36 s at one rank in debug on the
M6; 24 s, 17 s and 17 s at 2, 4 and 8 ranks.

Laplace bit for bit (constraint 1, condition 3):
- every existing `nd-fmm-exec` test, unchanged: `tests/mpi_exec.rs`, `invariance.rs`,
  `mpi_threading.rs` at 1, 2 and 4 ranks (8 by hand), the unit tests of the moved helpers,
  `tests/operator/`;
- the full outputs of `main` and the branch byte for byte on the M6 (CH3.0a's 36: the cube
  and the Plummer sphere, N = 10⁵, 2 vectors, f64, 4 threads, p = 3, 6, 8, Dense, Classes,
  Rotation, Batched, PlaneWave, Auto; `nd-fmm-bench --write-problem`, `--write-output`,
  `cmp`), `main` in a detached worktree with its own target directory.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, the default and
workspace tests, `cargo doc`, the MPI list at 2 and 4 ranks, the Laplace protection), plus
`nd-fmm-exec`'s own for a change to `Fmm`: `cargo clippy -p nd-fmm-exec --all-targets
--features cpu,metal -- -D warnings`, `cargo check -p nd-fmm-exec --features cuda` (the
device path reads the moved fields), `tests/mpi_exec.rs` at 8 ranks by hand under a
timeout. Not run unless asked: the ignored `multi_rank.rs` at 2, 4 and 8 ranks in release
(about 35 min each on the M3 Max; the change is a move of the build's statements, which
`mpi_exec` and `invariance` cover on every rank count).

## Machine runs

Minimal D9 runs (the user, 2026-11-01), on the M6 only, host only, outside the sandbox on a
quiet machine: D9's procedure (five runs per side, A B B A A B B A A B, `main` in the
detached worktree, `tools/bench/run.sh` at N = 10⁶, 10 timed evaluations, 2 warm-ups;
spread = max − min of `main`'s run medians; one repeat of a failing row allowed) on three
rows: `--strategy auto` f64 p = 6 (Batched) and p = 8 (PlaneWave), f32 p = 3. The
extraction moves build statements and the charge load and output pass, so the rows'
`load` and `output` columns are read beside the total. Not on locust or Kathleen.

## Constraints and risks

- Constraint 1 binds `fmm.rs`: (1) no dispatch below the level call (none touched; the
  operator, drivers and tables are not changed); (2) layouts and tables unchanged; (3)
  outputs bit for bit (pure moves; tested above); (4) benchmarks: the minimal D9 runs.
- Constraint 2: the Helmholtz FMM's determinism and rank invariance rest on the shared
  pipeline and drivers; tested above.
- Constraint 9: degrees, the leaf level and the tables from k, the domain and ε only, the
  same on every rank.
- Risk: code placement moves host times by ±5–13 % without any change of work (CH2.0);
  a failing row is checked with the `-align-loops=64` builds before it is called a
  regression.
- Risk: debug time of the reference P2P and RCR at ε = 10⁻⁶ (p ≈ 34); named low degrees
  for the bit-for-bit checks, ε = 10⁻⁶ only on the smallest tree, kD = 100 in release.
- Risk: `plan.nlevels()` differs by rank: the degrees depend on the level only, so the
  sizes agree; the evaluator's validation would catch a disagreement.

## Results

Ran on the M6 on 2026-11-01.

- `tests/helmholtz_fmm.rs`: debug at 1, 2, 4 and 8 ranks (36, 24, 17 and 17 s), release at
  1 and 4 ranks (with the ε = 10⁻⁶ scenarios), all passed. Errors against the direct sum,
  the same on every rank count:

  | Scenario | Degrees (levels 0–) | φ | ∇φ |
  | --- | --- | ---: | ---: |
  | cube, kD = 20 + 2i, ε = 10⁻³, rank 0 empty | 0, 0, 16 | 6.0e-8 | 1.5e-7 |
  | cube, kD = 1, ε = 10⁻³ | 0, 0, 16, 16 | 2.0e-8 | 1.0e-7 |
  | clusters, kD = 20, p = 8 | 8 on each of 6 | 7.5e-5 | – |
  | cube, kD = 20 + 2i, p = 8 | 8 on each of 4 | 1.5e-4 | 2.2e-4 |
  | cube, kD = 20, ε = 10⁻⁶ (release) | 0, 0, 34, 34 | 5.3e-13 | 4.0e-12 |
  | clusters, kD = 100, ε = 10⁻⁶ (release) | 0, 0, 38, 34, 34, 34 | 2.0e-10 | – |

  S7's closed form is far from tight on these low-frequency levels (its floor
  ⌈6 log₁₀(1/ε)⌉ − 2 gives p = 16 at 10⁻³ and 34 at 10⁻⁶): a point for CH3.3's calibration.
- Laplace's outputs: `nd-fmm-bench --write-output` from `main` (540f134, a detached
  worktree with its own target directory) and from the branch, the cube and the Plummer
  sphere at N = 10⁵, 2 vectors, f64, 4 threads, p = 3, 6, 8, Dense, Classes, Rotation,
  Batched, PlaneWave and Auto: 36 of 36 identical byte for byte.
- D9, the minimal runs (the M6, host, every core, N = 10⁶, 10 timed evaluations after 2
  warm-ups, `nd-fmm-bench` with the BLAS thread variables at 1; A = `main` 540f134,
  B = the branch e28eed1; order A B B A A B B A A B; started once Photos' media analysis,
  at about 115 % CPU earlier, had stopped):

  | Row | `main` median of medians | branch | difference | `main`'s spread | D9 |
  | --- | ---: | ---: | ---: | ---: | --- |
  | f32 p = 3, Auto → Batched | 71.05 ms | 71.55 ms | +0.50 ms (+0.7 %) | 4.02 ms | pass |
  | f64 p = 6, Auto → Batched | 209.54 ms | 212.36 ms | +2.81 ms (+1.3 %) | 14.14 ms | pass |
  | f64 p = 8, Auto → PlaneWave | 221.68 ms | 222.63 ms | +0.95 ms (+0.4 %) | 20.03 ms | pass |

  Every run's medians rose through the session, `main`'s and the branch's alike (f64
  p = 6: 200.7 ms in run 1 to 219.2 ms in run 10), most likely the M6 warming under
  sustained load. That widens `main`'s spread. The alternating order cancels the drift:
  B/A − 1 of adjacent pairs (runs 1–2, 4–3, 5–6, 8–7, 9–10) is −1.3 to +2.5 % (f32 p = 3,
  median −0.1 %), −1.4 to +2.0 % (f64 p = 6, median −0.1 %) and −2.1 to +1.0 % (f64 p = 8,
  median +0.4 %). Build times: 0.728 / 0.716 s, 0.745 / 0.750 s, 0.703 / 0.704 s (`main` /
  branch medians). Every run's repeats bit-identical. Reports in the scratch directory
  `../fmm-d9/runs/` (not committed).

## Pending runs

None.

## Open questions

Answered 2026-11-01 by the user, each as recommended: the extraction as planned
("Proceed as planned"); the minimal D9 runs on the M6, three rows ("M6, 3 rows"); the
ignored `multi_rank.rs` not run ("Skip it").
