# Phase 5S / T8r — sparse exchanges in rlst, released as 0.10.1 (S5 upstream)

**Added on 2026-10-13 (T8, at the user's request; the user maintains rlst).** T8's step 1
(`fmm-validate/results/phase5s-t8.md` §3, §4; `docs/design/scale-out.md` §7.3) found
that the O(P) collectives S5 targets live almost all in rlst. This task makes the change
in rlst, merges it into rlst's `main`, and publishes the release. T8 then continues
with the new release (its step 3; the T8 brief's note).

What T8 measured:
- **Build.** Each `Fmm::build` runs 21–32 `MPI_Alltoall`s of P counts. 29–30 of them are
  rlst's:
  - 12–23 in `GhostCommunicatorBuilder::build`: one per source, top and second-cut
    exchange, and one per level of the multipole and reduction exchanges. Each build
    also creates 2 graph communicators and runs 3 validity all-reduces.
  - 7 in `array_tools::all_to_allv`: the octree's routing, key move, balance and push,
    `leaf_point_counts`, and the second cut's leaf move and push. Each call also runs 4
    validity all-reduces.

  The other 2 are in `Redistribution::new` (`nd-fmm-plan`).
  - At 160 Kathleen ranks they cost 3–7 ms of a 121–373 ms build (2.5 × 10⁴ points per
    rank).
  - *Model* at 10⁴ ranks: 1.1–3.8 ms each, 32–113 ms per build, 6–27 % of a modelled
    build. The trigger is a quarter.
  - With every count 0, an all-to-all-v costs Open MPI 4.1.6 0.3, 167 and 1.2 µs at 40,
    80 and 160 ranks. Its `tuned` component picks pairwise (P − 1 steps) at 80 and basic
    linear at 40 and 160.
- **Evaluation.** Every evaluation runs two `MPI_Alltoallv` with P-long arrays (the
  redistribution's forward and backward). *Model* at 10⁴ ranks: 0 to 42 ms per
  evaluation, depending on that algorithm choice. The threshold is 1 % of an evaluation,
  0.22–5.7 ms.

The remedy is a **sparse dynamic exchange**, non-blocking consensus (NBX, Hoefler,
Siebert and Lumsdaine 2010): each rank sends to the ranks it has data for with
synchronous sends, receives whatever arrives through a matched probe, and posts a
non-blocking barrier once its own sends complete. Every rank leaves when the barrier
completes. No count array of length P is exchanged, and the cost is the actual messages
plus O(log P) for the barrier. rsmpi 0.8.2 has everything it needs:
- `Destination::immediate_synchronous_send`;
- `Source::immediate_matched_probe` with `any_process()`, and
  `Message::matched_receive_into`;
- `CommunicatorCollectives::immediate_barrier`;
- `request::scope`.

No rsmpi change and no `unsafe` beyond rlst's existing FFI conventions.

*Model* of the gain at 10⁴ ranks: the build's all-to-alls of counts go from 32–113 ms to a
few ms, from 6–27 % of a modelled build to about 1 %. The evaluation's pair goes from 0–42 ms
(depending on the library) to the cost of the actual messages. At 160 Kathleen ranks today
the gain is 1–3 % of a build. Measured, not assumed: the "after" numbers come from T8's
re-run.

Read first:
- rlst's `CLAUDE.md` (in `../rlst`, the rlst repository: the MPI rules, the validation
  commands, `scripts/run_mpi_tests.py`, the changelog rule, "do not modify `Cargo.lock`");
  `rlst/src/doc/distributed_computations.rs`;
- `rlst/src/distributed_tools/ghost_communicator.rs` (`GhostCommunicatorBuilder::build`),
  `array_tools.rs` (`all_to_allv`, `collective_valid`), and their tests in
  `tests/mpi_empty_ranks.rs`;
- rsmpi 0.8.2: `src/point_to_point.rs` (`immediate_synchronous_send`, the matched probe,
  `matched_receive_into`), `src/collective.rs` (`immediate_barrier`), `src/request.rs`
  (`scope`, `WaitGuard`), `src/topology/mod.rs` (`any_process`);
- this workspace's callers: `octree/src/octree/implementation.rs` (the `all_to_allv`
  calls), `fmm-exec/src/cost.rs` (`leaf_point_counts`), `fmm-plan/src/exchange.rs` (the
  ghost-communicator builds), `fmm-plan/src/redistribute.rs` (`Redistribution::new`,
  `forward_into`, `backward_into`);
- T8's report §3–§4 and `docs/design/scale-out.md` §7.2–§7.3;
- how rlst 0.10.0 was released: the commits of rlst PRs #40 and #41 (`git log` in
  `../rlst`), the `v0.10.0` entry of `rlst/CHANGELOG.md`.

Do (in `../rlst`, on a branch from its `main`; rlst's `CLAUDE.md` applies there, the root
CLAUDE.md of this workspace here):
1. **`sparse_all_to_allv`** in `distributed_tools::array_tools`, re-exported beside
   `all_to_allv`.
   - **Same arguments and same result as `all_to_allv`:** the received counts per rank
     (P long, for the caller) and the received data grouped by source rank in rank order.
     So it is a drop-in replacement wherever the counts are sparse. The difference is the
     communication: one synchronous message per non-zero count, then the non-blocking
     barrier.
   - **Validation folded.** The checks of `all_to_allv`'s four validity all-reduces go
     into at most one all-reduce before the exchange and one after it (the received
     total), so that every rank panics alike on bad metadata. Never panic between them
     with a message pending.
   - **Own messages.** Use a communicator the call duplicates (cache it if the cost
     matters, and state which) or a reserved tag. A test shows that a caller's pending
     message on the same communicator is neither consumed nor disturbed.
   - **Empty buffers.** With Open MPI, an empty `Vec` of a 1-byte type has the address
     `0x1`, which is `MPI_IN_PLACE`; T8's probe hit this in an `MPI_Alltoallv`. Never pass
     an empty slice's pointer to MPI, and test `u8` payloads with empty ranks.
2. **`GhostCommunicatorBuilder::build` by NBX.**
   - Every rank knows the owners of its ghosts. It sends each owner the indices it needs,
     in its existing order, and the owners receive them by NBX. That one round replaces
     the all-to-all of counts and the first `neighbor_alltoallv` (the indices to the
     owners).
   - The out-neighbours and `send_indices` follow from what arrived, ordered by source
     rank as now. Then the two graph communicators and the chunk-size exchange, as now.
   - The three `collective_assert`s fold into as few all-reduces as the agreement allows
     (at most two). The P-long `receive_counts`/`send_counts` go.
   - **Every public result is identical to 0.10.0's:** `out_ranks`, `in_ranks`,
     `send_indices`, `receive_indices`, counts, offsets and chunk sizes, element for
     element, on every rank. Keep 0.10.0's algorithm as a private test oracle.
3. **A fixed-route exchange** for patterns built once and used often, as `Redistribution`
   does on every evaluation. Name it as rlst's API style suggests, for example
   `SparseRoutes` in `distributed_tools`.
   - **Built from each rank's send counts per destination:** one NBX round tells every
     rank its sources and receive counts.
   - **Then `exchange` and `exchange_back`:** values per item (`per_item`) sent with
     immediate sends to the ranks with a non-zero count, received with immediate
     receives from the known sources. The receive buffer is laid out by source rank, as
     `all_to_allv` lays it out, so the bytes and their order equal the dense exchange's.
     No collective per exchange, no array of length P.
   - **Usable as the blocking pair `Redistribution::{forward_into, backward_into}` needs,**
     with caller-provided buffers and checked lengths. The scope API leaves room for an
     overlapped use later.
4. **Tests** (rlst's conventions: unit tests in the module, the multi-rank cases in
   `tests/mpi_empty_ranks.rs` or a new harness-free executable registered like it, run by
   `scripts/run_mpi_tests.py` at 1, 2 and 4 ranks):
   - `sparse_all_to_allv` against `all_to_allv`: counts and data equal on every rank for
     these patterns: all counts zero, one peer, ring, random sparse, dense, empty ranks,
     every datum from one rank, and `u8`, `u64` and a derived `Equivalence` type;
   - the NBX `build` against 0.10.0's: every public result equal, with uniform and
     per-index chunk sizes, ghosts owned by self, repeated indices, empty ranks and
     invalid owners (rejected on every rank);
   - the fixed-route exchange against `all_to_allv` on the same routes, both directions,
     several `per_item`, and repeated exchanges on one route set;
   - by hand, under a timeout: the same at 8 ranks (M3 Max) and at 64 and 512
     oversubscribed (locust).
5. **Documentation:** rustdoc for every new public item with its collective contract
   (who must call it, in which order, what is agreed). Update
   `doc/distributed_computations.rs` to cover when to use the sparse functions.
6. **Before any release, this workspace on the branch.** In a scratch copy of this
   workspace, not committed, add `[patch.crates-io] rlst = { path = "<../rlst>/rlst" }`
   with no call changed:
   - `cargo test -p nd-octree`, `-p nd-fmm-plan`, `-p nd-fmm-exec`;
   - `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks;
   - `tests/invariance.rs` and `tests/mpi_exec.rs` at 8 ranks;
   - invariance at 512 against 64 on locust.

   Every check must pass bit for bit. Only the ghost builds change underneath; the
   octree, the plan and the outputs must not. The calls switch in T8, not here.
7. **The release, 0.10.1.**
   - **Version.** Every change is additive or internal, so this is 0.10.1. If any public
     signature had to change, it would be 0.11.0 with migration notes; stop and ask
     first.
   - **Release commit** in rlst's style (`chore(release): prepare workspace for
     0.10.1`):
     - `rlst`, `rlst-proc-macro` and `rlst-suitesparse` at 0.10.1, as their versions
       move together;
     - `rlst`'s dependency on `rlst-proc-macro` at the same version;
     - the `v0.10.1` entry of `rlst/CHANGELOG.md`, "changes since v0.10.0": the three
       features, the unchanged results, the gain with its numbers marked as T8's, and the
       Open MPI empty-buffer note;
     - guide version references where they name a patch version.
   - **rlst's checks:** every command of rlst's `CLAUDE.md` "Validation commands" (the six
     Clippy runs, `cargo fmt -- --check`, the feature tests, `cargo test -p
     rlst-suitesparse`, `scripts/run_mpi_tests.py`, `cargo doc`). Report any that cannot
     run (missing native libraries) as not run, never as passing.
   - **Merge into rlst `main` on Codeberg** (codeberg.org/rlst/rlst), as #40 and #41 were:
     the feature branch and the release commit in pull requests, merged into `main`. If
     no Forgejo CLI or token is available in the session, the user merges in the web UI.
     Then tag `v0.10.1` on the merge commit and push the tag.
   - **Publish:** `cargo publish --dry-run`, then `cargo publish`, for `rlst-proc-macro`,
     then `rlst`, then `rlst-suitesparse`. Each is published from the tagged commit, and
     `rlst-suitesparse` needs SuiteSparse, so publish it where that exists. The crates.io
     token lies outside the sandbox, so publishing runs outside it.
   - Confirm that 0.10.1 resolves from crates.io (`cargo search rlst`, or a fresh lock in
     a scratch crate).

   **Approval per action** (root CLAUDE.md, "Working agreement"): the user asked for the
   merge and the publication when scheduling this task (2026-10-13). The session still
   asks before each push, each merge, the tag and each `cargo publish`, because a
   published version cannot be withdrawn, only yanked.
8. **Hand back to T8.** Update this workspace's docs/phase5s/README.md checklist. Write a
   short note in T8's report (§6, the S5 questions) with the release, its commit and tag,
   and what T8's step 3 now does:
   - pin `rlst = "0.10.1"` in `[workspace.dependencies]` and in `octree/Cargo.toml`, one
     commit with `Cargo.lock`;
   - the octree's sparse moves (key move, balance, push, the second cut's leaf move and
     push) on `sparse_all_to_allv`;
   - `Redistribution::new`'s counts and `forward`/`backward` on the fixed-route exchange
     (S5 for the evaluation);
   - every result bit for bit;
   - T8's probe re-run before and after on locust and on Kathleen at 160 ranks timed and
     640 counted.

Tests that define done:
- the rlst tests of item 4 pass at 1, 2 and 4 ranks (`scripts/run_mpi_tests.py`), at 8 by
  hand and at 512 oversubscribed on locust;
- this workspace on the branch passes the checks of item 6 bit for bit;
- rlst 0.10.1 is on crates.io, tagged `v0.10.1` on rlst `main`, with its changelog entry.

Must pass:
- in rlst, every validation command of its `CLAUDE.md`;
- in this workspace (scratch, patched): the root checks, `run-mpi-tests.sh` at 2 and 4,
  `invariance` at 8 and at 512 against 64.

Do not:
- change any public signature or result of rlst 0.10.0;
- commit a `[patch]` of rlst into this workspace, or change this workspace's pin in this
  task (T8 does);
- publish before item 6 passes, or without asking at each push, merge, tag and publish;
- fork rlst or rsmpi, or add a dependency to rlst;
- change S8 here: it is T8s, after T8 (signed off on 2026-10-13).
