# Phase 5S / T8q — rotated send order in rlst's sparse exchanges, released as 0.10.2

**Added on 2026-10-14 (T8, at the user's request; the user maintains rlst).** T8's step 3
found a transport stall on 4 Kathleen nodes (`fmm-validate/results/phase5s-t8.md` §9.3).
At 160 ranks, a dense burst, with every rank sending a few KB to every other rank at once,
loses packets in about half the phases. PSM2 resends each lost packet only after its
error-check timeout, 160 ms by default. The timeout is a Kathleen setting, and `env.sh`
now shortens it.

The burst itself is generic, though, and rlst 0.10.1's sparse exchanges make it worse
than necessary. `SparseRoutes`' `exchange` and `exchange_back` and the non-blocking
consensus round behind `sparse_all_to_allv`, `SparseRoutes::new` and
`GhostCommunicatorBuilder::build` all post their messages in ascending rank order. So
every rank sends to rank 0 first, then to rank 1, and so on: at each moment all senders
converge on one receiver, the incast that loses packets. MPI libraries avoid this in
their all-to-all algorithms by having rank r start at r + 1 and wrap around. This task
does the same in rlst.

Read first: the T8r brief (`T8r-rlst-sparse-exchange.md`), rlst's `CLAUDE.md`, and
`rlst/src/distributed_tools/array_tools.rs` (`nbx`) and `sparse_routes.rs` (`route`).

Do (in `../rlst`, on a branch from its `main`):
1. **Rotate the posting order** in `nbx` and in `SparseRoutes::route`. A rank r posts to
   the destinations in the order r + 1, r + 2, …, P − 1, 0, …, r − 1, and its receives
   from the sources in the same rotated order.
   - Every buffer keeps its layout: the blocks are still cut by ascending rank, and only
     the order in which requests are posted changes.
   - Every public result, the received bytes and their order included, stays identical
     to 0.10.1's.
   - Unit-test the order (a pure function of rank and size).
2. **Tests:** rlst's `scripts/run_mpi_tests.py` at 1, 2 and 4 ranks, all cases at 8 by
   hand, and `sparse-all-to-allv`, `ghost-build-oracle` and `sparse-routes` at 64 and 512
   oversubscribed on locust.
3. **Measure before releasing.** Run T8's stall micro-benchmark (job 604021's: 160 ranks
   on 4 Kathleen nodes, 20 repetitions, twice) against the branch with PSM2's default
   timeout, `PSM2_ERRCHK_TIMEOUT=160:640:2` set explicitly. Compare the stalls of the
   `SparseRoutes` and consensus sequences with 0.10.1's. Report the result whichever way
   it goes: it tests whether converging senders trigger the losses.
4. **This workspace on the branch** (a scratch copy with `[patch.crates-io]`, as in T8r):
   `cargo test --workspace`, `run-mpi-tests.sh` at 2 and 4, and `invariance` at 8, bit
   for bit.
5. **Release 0.10.2** as T8r released 0.10.1:
   - all three crates at 0.10.2 and the `v0.10.2` CHANGELOG entry;
   - the feature PR and a release PR on Codeberg, merged;
   - tag `v0.10.2` with a Codeberg release;
   - `cargo publish` of proc-macro, then rlst, then suitesparse.

   Ask before each push, merge, tag and publish.
6. **Hand back:**
   - pin `rlst = "0.10.2"` here, one commit with `Cargo.lock`;
   - re-run invariance at 512 against 64 on locust;
   - note the release and the measurement in T8's report §9.3.

Do not: change a public signature or result of rlst 0.10.1, add a dependency, cap the
number of messages in flight or fall back to dense collectives. Those are for a measured
need on a larger system, not this task.
