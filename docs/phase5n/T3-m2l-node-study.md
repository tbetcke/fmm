# Phase 5N / T3 — M2L on a node: measurements and the strategy rule (design for C5N.3)

`M2lStrategy::Auto` resolves to Dense for p ≤ 8 and to Rotation above
(`fmm-exec/src/tables.rs`, `M2lStrategy::resolve`, `AUTO_DENSE_MAX_P = 8`). The rule comes
from Phase 2 T8 (laplace-fmm-plan §7, Phase 2): single-threaded, one rank, one product or
one rotation per pair. Phase 5 T10 measured it on several ranks of one node and found it
wrong there:
- at p = 8 every rank holds its own 316 dense M2L tables, 16.6 MB in f64 and 8.3 MB in f32
  (0.65 MB at p = 3; computed from the table shapes), and the downward pass stops scaling:
  its total over the ranks grows 2.4–4.4× by 8–12 ranks on the M3 Max and 5.8–6.9× by 72
  on locust (phase5-m3max.md §1, "Stages against ranks"; phase5-gh200.md §1);
- Rotation keeps an efficiency of 0.75–0.79 at 12 and 72 ranks; at 12 ranks on the M3 Max
  it takes 688 ms against Dense's 2,752 on the cube (4.0×), at 72 on locust 164 against
  940 ms (5.7×) and 356 against 1,534 on the Plummer sphere (4.3×) (both reports, "The M2L
  strategy at p = 8");
- Classes, Dense's products plus two O(p³) transforms from 16 class matrices (0.84 MB in
  f64), is 1.18–1.51× slower than Dense on one rank, yet faster than Dense from 8 ranks on:
  more arithmetic from small tables scales, the same products from large tables do not;
- Rotation is faster than Dense even on one rank: 0.79× (cube) and 0.85× (Plummer)
  Dense's time on the M3 Max, 0.95× and 0.96× on locust;
- threads share one copy: 1 × 12 is 3.0× faster than 12 × 1 on the M3 Max, and 4 × 18 is
  4.1× faster than 72 × 1 on locust (both reports, §4).

No cache counters were read, so the mechanism (per-rank read-only tables competing for
the shared caches and memory bandwidth) is inferred, not measured. This task measures M2L
on a node on every machine available after decision 0: the M3 Max, locust and, if
Kathleen is usable, one Kathleen node (with T2's numbers). Without Kathleen it runs on the
M3 Max and locust only, and `node-m2l.md` marks the Kathleen columns "not measured" and
lists what a later Kathleen run must confirm. It writes a short design,
`docs/design/node-m2l.md`, that proposes the rule `Auto` should follow, how the build
learns the node context, how tables are loaded at scale, and what to do about one copy
per rank. It is signed off by hand before T4. It writes no library code, apart from
throwaway measurements in the scratch directory and additions to `nd-fmm-validate`'s
examples where a measurement needs them (state which).

Read first:
- docs/phase5n/README.md (requirements 3, 4 and 7; decisions 5–8);
- `fmm-exec/src/tables.rs` (`M2lStrategy`, `resolve`, `Tables::load_or_build`, the
  families), `fmm-exec/src/fmm.rs` (`FmmBuilder::strategy`, `table_cache`, `threads`;
  `build`'s steps; where the strategy is resolved; the device path's `split_shared` and
  `RankPlacement` at `build`), `fmm-exec/src/threading.rs` (the rules and
  `ThreadingReport`), `fmm-exec/src/operator.rs` (how the host level calls apply the
  tables, target by target);
- `fmm-tables/src/m2l.rs` (`M2lTables`, `M2lClasses`), `rotation.rs` (`RotationTables`),
  `cache.rs` (`TableCache::store`: a temporary file created with `create_new`, synced, then
  renamed; `load`; `load_or_build`);
- `fmm-validate/examples/tables.rs` (build time, memory per family, cache cold and warm)
  and `scaling.rs` (`--strategy`, `--threads`);
- the T10 reports and the Kathleen baseline (`fmm-validate/results/phase5n-kathleen.md`,
  T2);
- distributed-fmm.md §14.4 (S6: one rank per socket or NUMA domain with threads) and §15
  (Phase 5's measured numbers);
- laplace-fmm-plan §7, Phase 2 (T8, the rule's source), Phase 5 ("Recommendation for
  Phase 6"), and §6.8 (threads and BLAS).

Measure, with how each number was obtained (machine, ranks per node × threads per rank,
build, command):
- **The grid.** On each machine, N = 10⁶, the cube and the Plummer sphere:
  - at f64 p = 8: Dense, Classes and Rotation at every ranks-per-node count the machine
    runs one thread each (M3 Max 1, 2, 4, 8, 12; locust 1, 8, 16, 32, 64, 72; Kathleen 1,
    2, 10, 20, 40 on one node, both sockets filled evenly) and at the splits with threads
    (M3 Max 1 × 12, 2 × 6; locust 1 × 72, 4 × 18, 8 × 9; Kathleen 1 × 40, 2 × 20, 4 × 10);
  - at f32 p = 8, and at f64 p = 3, 6, 10 and 12: the three strategies at one rank, at the
    largest one-thread count, and at the best split of the f64 p = 8 grid;
  - per run the downward stage's time over the busiest rank's V pairs (µs per pair), so
    that runs with different trees compare, and the M2M and L2L parts separately where
    the kind timings (`KindTiming::Synchronous`) give them;
  - the errors of each strategy against the direct sum (the harness's `--errors 8`): the
    strategies apply the same truncated translations and should differ only in rounding;
    report the differences between strategies at fixed settings, relative L2 of φ and ∇φ.
- **Table sizes against the caches.** Per strategy, p and precision, the bytes a rank
  reads in the M2L level calls (dense: 316 (p+1)⁴ values; classes: 16 (p+1)⁴ plus the
  transforms; rotation: `RotationTables` and its per-level data), beside each machine's
  cache sizes per core and per shared level (`lscpu -C` on Linux, `sysctl hw` on the M3
  Max), and the ranks that share each level. A model of when per-rank tables exceed a
  shared level, marked *model*, checked against the grid.
- **Isolating the cause** (one experiment, enough to confirm or refute the inference): on
  one machine, the per-pair downward cost of one rank running Dense alone against the same
  rank while the other cores run (a) the same Dense evaluation, (b) a Rotation evaluation,
  (c) a memory-bandwidth load without tables (for example a large array copy). If a
  hardware counter tool is available without new dependencies (`perf stat` on Kathleen or
  locust), report last-level-cache misses for (a) and (b); otherwise say it was not.
- **Tables at build and at scale.**
  - The time to build each strategy's tables at p = 3, 6, 8, 10 and 12 (the `tables`
    example, one thread), and the share of `build` they take at the largest rank count on
    each machine (T10: 216 ms at p = 8 on locust, 63% of the build at 8 ranks, 83% at 72;
    distributed-fmm.md §15.2).
  - The time to load each from `TableCache` cold and warm: on the M3 Max's local disk, on
    locust's `/data` (1, 8 and 72 ranks loading at once), and, with Kathleen, on its
    Lustre with 1, 40 (one node) and 80 (2 nodes, the largest job allowed for now) ranks loading
    at once.
  - Concurrent stores into one cache directory: two processes on one node and two on
    different nodes storing the same key at once (Lustre). `TableCache::store` is designed
    to make both succeed (`create_new` temporary names from the process id and a counter,
    then a rename); confirm it on Lustre, where two nodes may share a process id.
- **The node context.** The cost of a `split_shared` (and of agreeing its result) inside
  `build` on the host path, at 1, 12 and 72 ranks on the M3 Max and locust, and at 40 and
  80 ranks on Kathleen if it is usable; today only the device path
  splits (`build`'s `RankPlacement`).

Write `docs/design/node-m2l.md`, short, in the style of the other design documents, with
these sections:
1. **Starting point.** The rule and its source; T10's evidence; T2's Kathleen numbers (if
   any); what
   one copy per rank costs, as measured.
2. **Measurements.** The grid, the table sizes against the caches, the isolating
   experiment, the build and load times, the node-context cost, each with its source.
3. **The model.** When per-rank tables lose: bytes per rank × ranks sharing a cache level
   against its size, and the measured crossover per machine; marked *model* where it is
   one.
4. **The rule.** The inputs it may use and how every rank agrees them: p, precision, the
   ranks per node (`split_shared`), threads per rank (`FmmBuilder::threads`); whether a
   machine property (cache size) may enter, and how it would be agreed (a rule that reads
   the machine is harder to reproduce; prefer one that does not, unless the grid shows it
   must). Then the options, each with its measured cost against the best strategy on every
   machine and ranks-per-node count:
   - A. `Auto` resolves to Rotation (or Classes) at every rank count where the grid shows
     it faster, one rank included: the simplest rule, and it changes one-rank output bits
     at p ≤ 8;
   - B. `Auto` sees the node: Dense while ranks per node × table bytes fit the threshold
     the model gives, Rotation (or Classes) above it; one-rank results unchanged where
     Dense stays;
   - C. `Auto` unchanged; the configuration is documented instead (threads per rank, an
     explicit strategy for many ranks per node), and the harness defaults follow it;
   - anything better the measurements suggest.

   Recommend one, with the margin it keeps from the best strategy (decision 8). State what
   it does to Phase 5's guarantee (bit for bit the one-rank `Fmm` of the same *resolved*
   strategy over the union in rank order) and to what is reported (`Fmm::strategy()`
   returns the resolved strategy today; say what the node context adds to it or to
   `Fmm::threading()`'s `ThreadingReport`).
5. **Tables at build and at scale.** Every rank building (as today), `TableCache` warmed by
   one job, or one rank per node building and storing while the others wait at a barrier
   and then load (one collective more); recommend one (decision 7), with the measured
   times and, with Kathleen, the I/O load on Lustre at 80 ranks.
6. **One copy per node.** The options and their verdicts:
   - MPI shared memory (`MPI_Win_allocate_shared`): rsmpi 0.8.2 offers no RMA windows
     (checked in the registry source), so it needs an rsmpi addition behind a safe API,
     asked for upstream, or raw FFI, which root `CLAUDE.md` forbids outside the allowed
     crates;
   - memory-mapped cache files shared through the page cache: needs `unsafe` (`mmap`) or a
     new dependency; excluded by the rules;
   - threads per rank (one copy per rank, few ranks per node): available now;
   - the batched host M2L of Phase 6 (C6.6), which reads each table once per level for
     many targets and makes the copy per rank cheap;

   and recommend one for decision 6.
7. **Ranks × threads.** What the `threading` module and the harness should recommend for
   many cores per node (distributed-fmm.md §14.4, S6: one rank per socket or NUMA domain
   with threads), with the measured splits.
8. **Task check.** What T4 builds: the rule in `resolve` (or a new resolution step with the
   node context), the build plumbing, the reported fields, the table loading, the
   documentation, the tests, the harness knobs; and what stays for Phase 6.
9. **Questions for sign-off.** At least decisions 5–8 of the README, each with a
   recommendation.

Keep the document short: tables where they help, Rust signatures only where they decide
something, every model marked, every measured number with its source.

The PR description must contain a one-page summary, the sign-off questions with a
recommendation each, and the grid's headline table (per machine and ranks per node: the
best strategy, Dense's time over the best, `Auto`'s time over the best).

Must pass: nothing builds differently in the library crates. If examples changed:
`cargo fmt --all`, `cargo clippy -p nd-fmm-validate --all-targets -- -D warnings`,
`RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`. Every multi-rank measurement under
an external timeout; on locust the load checked before and after and stated; on Kathleen
in jobs, within the node-hours of decision 3 (proposed: 50), reported.

Do not:
- change `M2lStrategy`, `resolve`, any default, or any library crate (T4 builds what is
  signed off);
- commit measurement code or raw output (examples' additions excepted);
- propose a rule without the measurements behind it on every machine available after
  decision 0, or a new external dependency or `unsafe` outside the allowed crates;
- submit any Kathleen job larger than 2 nodes (80 cores).
