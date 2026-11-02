# CH3.2c Batched, and the measurements per p

Started 2026-11-02 on `tbetcke/h-ch3-2c`. Roadmap: "CH3.2c Batched, and the measurements
per p" (H3; CH3.2 split on 2026-11-02 into a/b/c; off the priority list, named by the user).
Proposal: §10 CH3.2; §5.3 (Batched), §6.2, §6.4, §6.5 (the split planes); §2 (constraints
1, 2, 5, 6, 7, 9).

## Goal

A batched M2L for the Helmholtz operator on CH3.2a's dense tables, run block by block with
the column kernel of `nd-fmm-simd`, and Rotation, Dense, Classes and Batched measured per p
on the M6, locust and Kathleen.

## Acceptance (proposal §10, verbatim)

CH3.2: "measured against Rotation per p, including the cost of the cube-group transforms
that move the z-axis". The roadmap's *Done when* of CH3.2c: against Dense (bit for bit, or
within rounding where an epilogue reorders the sums, stated); the FMM as CH3.2a's
(against the direct sum and against Rotation's output, the same at every thread count and
on P ranks the one-rank FMM bit for bit); Rotation, Dense, Classes and Batched per p on the
M6, locust and Kathleen.

## Decisions taken before the work (the user, 2026-11-02)

1. **Helmholtz's own block** ("Own Helmholtz block", as recommended), the roadmap's
   alternative to the two-plane case of `driver::m2l_block`. The Helmholtz operator's M2L
   level call runs its own block on the crate-private `Execution::blocks` and
   `M2L_BLOCK_TARGETS`, with `nd_fmm_simd::MatrixKernel` unchanged. `driver.rs`,
   `nd-fmm-simd` and every Laplace file stay untouched, so constraint 1 holds trivially
   and needs no D9 runs. The shared block would need the plane count and two to four kernel
   calls per offset at other target offsets inside Laplace's driver, for no gain to
   Laplace, and then D9 on three machines.
2. **Split planes, 2n²** ("Split planes, 2n²", as recommended): per offset A_r and A_i as two
   square n × n matrices (Dense's entries bit for bit, re-laid; Dense's bytes). The real
   multipole of real k and real charges takes two calls (L_r += A_r M, L_i += A_i M), bit
   for bit Dense. Complex multipoles take four (below), Dense to rounding only: the
   existing kernel sums a column's k in order, so the real plane's products come before the
   imaginary plane's, where Dense interleaves them per k (CH3.2a's sign-off 2 allows it,
   stated).

## Read first

- `fmm-exec/src/driver.rs`: `m2l`, `m2l_block`, `Execution::blocks`, `Columns`.
- `fmm-exec/src/operator.rs`: "The batched M2L"; `M2L_BLOCK_TARGETS`.
- `fmm-simd/src/matrix.rs`: `MatrixKernel::accumulate_columns` (order, shapes, `pack_len`).
- `fmm-tables/src/helmholtz/dense.rs`: layout, `matrices`, `apply`, `select`.
- `fmm-exec/src/helmholtz/operator.rs`: `Kernel::m2l`, `with_m2l`, `with_p2p`, `Scratch`,
  the `FmmOperator` impl; `fmm-exec/src/helmholtz/fmm.rs` (the builder's `m2l`).
- Tests: `fmm-tables/tests/helmholtz/dense.rs`, `fmm-exec/tests/helmholtz/dense.rs`,
  `fmm-exec/tests/helmholtz_fmm.rs` (`check_tables`).
- `fmm-validate/examples/helmholtz_m2l.rs`, `helmholtz_accuracy.rs`; reports
  `fmm-validate/results/helmholtz-ch3-2a.md`, `helmholtz-ch3-2b.md`.

## Changes

`nd-fmm-tables` (the `helmholtz` module), `nd-fmm-exec` (the Helmholtz modules only) and
`nd-fmm-validate` (examples, a report). No Laplace file, no `driver.rs`, no `nd-fmm-simd`,
no `nd-fmm-plan`, no convention, no dependency.

- `nd_fmm_tables::helmholtz::batched::BatchedM2l<T>` (new file
  `fmm-tables/src/helmholtz/batched.rs`): per chosen level and offset A_r then A_i, each
  n × n column-major, Dense's matrices re-laid (built level by level through
  `dense::matrices`, one offset re-laid at a time). `select` and `level_bytes` are Dense's
  (the same bytes and budget rule); `planes(level, d) -> (&[T], &[T])`, `is_batched`,
  `degree`, `nlevels`, `bytes`.
- `nd_fmm_exec::helmholtz`:
  - `HelmholtzM2l::Batched`, on the levels of `DenseM2l::select` within the dense budget,
    Rotation on the others; `HelmholtzOperator::batched_bytes()`; `m2l_on` reports it.
  - The chain (new crate-private module `fmm-exec/src/helmholtz/batched.rs`), per offset
    of a target, per vector, with `MulAdd::Separate` (`MatrixSet::apply`'s steps):
    - one plane: L_r += A_r M; L_i += A_i M;
    - two planes: L_r += A_r M_r; L_r −= A_i M_i (L_r negated, A_i M_i added, negated back:
      exact, the same bits as adding −A_i M_i up to the sign of an exact zero);
      L_i += A_i M_r; L_i += A_r M_i.
    The L_i columns and the M_i columns are the same offset lists on the slices shifted by
    n, so a block needs only the driver's `Columns` lists (reused as a type, not changed).
  - The level call: `FmmOperator::m2l` runs the block where the level has batched tables
    (blocks of `M2L_BLOCK_TARGETS` targets on `Execution::blocks`; per offset d in order the
    pairs of `batch(d)` in the block, found by two binary searches), `driver::m2l`
    otherwise. The pair body (`Kernel::m2l`, used by the target-by-target driver on no
    level, by `PairOperator` and so `PerPair`) runs the same chain with one column per
    vector, so the level call is the pair-by-pair operator bit for bit.
  - The column kernel on the P2P choice's ISA (`Isa::Scalar` for `Reference`), as Laplace's,
    in `MulAdd::Separate` by default. The scratch's `Columns` sized at construction
    (`MatrixKernel::pack_len` of the largest n, m `M2L_BLOCK_TARGETS` offsets).
  - As built, beyond the first plan: the fused mode as an opt-in,
    `HelmholtzOperator::{with_m2l_mul_add, m2l_mul_add}` and
    `HelmholtzFmmBuilder::m2l_mul_add` (as Laplace's `FmmBuilder::m2l_mul_add`), after the
    first smoke run of `helmholtz_m2l` showed the separate form only level with Dense with
    complex multipoles (Dense's one-pass loop vectorises as well; both are bound by the
    separate multiply and add) and the fused form about 1.8× faster. Dense to rounding in
    both cases; deterministic, ISA- and thread-independent as the separate mode.
  - `HelmholtzFmmBuilder::m2l(Batched)` with `dense_budget`; docs (operator module "M2L",
    "Execution"; `fmm-exec/CLAUDE.md`'s Helmholtz rules).
- `nd-fmm-validate`:
  - `helmholtz_m2l`: Batched per p, the same chain on a block (32 targets per offset, each
    with its own source, the operator's call sequence, checked against Dense first), ns per
    pair, separate and fused; its build seconds (its bytes are Dense's).
  - `helmholtz_accuracy --m2l batched,batched-fused` (`Case::m2l_mul_add`,
    `Row::m2l_mul_add`, the bytes).
  - Report `fmm-validate/results/helmholtz-ch3-2c.md`: per p on the three machines, end to
    end on CH3.2a's cases on the three machines.

## Tests that prove it

- `fmm-tables/tests/helmholtz/batched.rs` (new module): `planes` are Dense's matrices'
  halves bit for bit on every offset of levels 2–4 (real and complex k̂); `select` and
  `level_bytes` Dense's; f32 the f64 tables rounded; refusals (level < 2, a level without
  tables).
- `fmm-exec/tests/helmholtz/dense.rs`: the operator under `Batched` pair by pair on every
  offset of levels 2–4, real and complex k, real and complex charges, two vectors: against
  Dense bit for bit with one plane (real k, real charges), within `DENSE_BOUND` of Dense and
  of Rotation with two; the same bits on every available ISA (`with_p2p`); `m2l_on` and the
  budget; a zero budget Rotation bit for bit.
- `fmm-exec/tests/helmholtz_fmm.rs` (CI's MPI list): `check_tables` with `Batched` on the
  two scenarios (the direct sum, Rotation's output within `DENSE_TOLERANCE`, 2 threads, pair
  by pair and the one-rank FMM bit for bit, the finest level alone), and against Dense's
  output: bit for bit on the real-charge real-k scenario (one plane), within
  `DENSE_TOLERANCE` on the complex one; the fused mode within `DENSE_TOLERANCE` of Rotation
  and Dense, 2 threads and the one-rank FMM bit for bit.
- `fmm-exec/tests/helmholtz/dense.rs` also: the fused mode against Dense and Rotation within
  `DENSE_BOUND` (not Dense's bits), and the same bits on every ISA in both modes.
- f64 for the FMM (D3); f32 only for the tables' rounding.

## Checks

`/verify`: format; Clippy default, examples and workspace all-targets; `cargo test` and
`--workspace`; `cargo doc`; the MPI list at 2 and 4 ranks (`helmholtz_fmm` changes);
`cargo test -p nd-fmm-tables` and `-p nd-fmm-exec`. No licence check (no dependency
change). No D9 (Laplace untouched).

## Machine runs

Release, one rank, one thread, BLAS threads 1:
- the M6 (behind the quiet gate, after the user's go-ahead): `helmholtz_m2l --ps
  3,4,5,6,7,8,9,10,12,15,18,20`, two launches; `helmholtz_accuracy --m2l
  rotation,dense,classes,batched,batched-fused` and the reverse order on CH3.2a's cases (W1 at kD = 30
  and 100, the W4 cube at kD = 10 at ε = 10⁻³ and 10⁻⁶, W3 at kD = 30 at 10⁻³);
- locust (Grace, NEON; `tools/gh200/sync.sh`, `remote.sh`; check `uptime` first): the same;
- Kathleen (Cascade Lake, AVX-512; one `singlenode` job through `run.sbatch`): the same.

They decide nothing on their own: the defaults stay as they are, and an Auto rule (which
M2L per level by p and the machine) is a question for sign-off with the data.

## Constraints and risks

- Constraint 1: Laplace untouched (no Laplace file, no `driver.rs` or `nd-fmm-simd`
  change; `HostKernel::batched_m2l` stays `None` for Helmholtz). The branch on the level's
  tables is in the Helmholtz operator's level call, above the pair arithmetic.
- Constraint 2: every element depends on its own column only and meets the offsets in
  index order, so blocks, threads and ranks give the pair-by-pair bits.
- Constraints 5, 7, 9: Dense's tables (f64, rounded once), its budget and its choice.
- Risk: the four calls of two planes each pack their source columns (four packs where one
  packed 2n column would do); measured, and a 2n-column form is the alternative if it
  matters (the 2n × 2n table, twice the bytes, declined above).

As run (2026-11-02; the report's §3): the M6 at a27a78e after the user's go-ahead, behind
CH3.2a's gate (one void launch, repeated); locust at 64b737e (another user's one-core job
throughout, GPU idle); Kathleen job 618018 on node-c11a-011 at 64b737e. The commits between
64b737e and a27a78e change tests and documentation only.

## Pending runs

None: the M6, locust and Kathleen runs are in the report.

## Open questions

For sign-off, in the PR:
1. Batched stays opt-in, Rotation the default and the separate mode Batched's default, with
   no Auto rule in this task. The data argue for a per-level rule by p (the fused Batched
   M2L on the levels with tables below a machine's crossover degree, about 10 on the M6 and
   locust and 15 on Kathleen, Rotation above), built and measured as a follow-up task
   before H3.G, not here.
2. The fused mode (`with_m2l_mul_add`, `HelmholtzFmmBuilder::m2l_mul_add`), added beyond the
   first plan as an opt-in after the first measurement (Dense to rounding in both cases).
3. The one-rank debug run of `tests/helmholtz_fmm.rs` grows from 69 to 122 s on the M6
   (Batched's checks 53 s, each run rebuilding the dense tables; about 7× cheaper at CI's
   opt-level 1; the MPI list at 2 and 4 ranks 197 and 148 s locally at opt-level 0, within
   CI's 900 s): kept, or trimmed (for example the two-vector check on one scenario).
4. The proposal's §6.2 ("The order", item 2) and §10 row for CH3.2 stay as written; the
   roadmap and this record carry Helmholtz's own block in place of the two-plane case of
   the shared driver (decided before the work, decision 1 above), as CH3.5's record did.

Signed off 2026-11-02, the user: "i agree with the recommendations". So Batched stays
opt-in, with Rotation the default and the separate mode Batched's default, and the per-level
rule by p is a follow-up task before H3.G (1); the fused mode stays (2); the debug cost is
kept (3); the proposal's §6.2 and §10 stay as written, the roadmap and this record carrying
the own block (4).
