# Phase 5S / T4 — the plan over the locally essential set (C5S.2; S1, part 2)

T3 gives every rank its locally essential key set and the partition map as keys. The plan
still assumes the Phase 5 set: `Plan::new` (fmm-plan/src/plan.rs) builds the box index,
the lists and the exchanges over every held key and checks that every coarse block is held
(fmm-plan-redesign §3.5, decision 6). `CoarseExchange` and `upward_global` still assume
every block on every rank; T5 replaces those.

This task builds the `nd-fmm-plan` half of S1 as `docs/design/scale-out.md` §3 specifies:
- `Plan::new` over the locally essential set;
- the check that every key a list names is held, replacing "every coarse block held";
- the source and multipole exchanges built from it.

Every output bit stays the same at every rank count, and the held boxes per rank stop
growing with P at fixed N/P.

Read first:
- fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md, root CLAUDE.md ("MPI");
- docs/phase5s/README.md (requirements 2–4, 7–9) and `docs/design/scale-out.md` §3, §8, §9;
- fmm-plan-redesign.md §3.3–§3.5 and §7.3; distributed-fmm.md §3.6 (P2, whose rows must
  survive), §9.2, §14.4 (S1);
- T3's merged API in `nd-octree` (the partition map, the key set, `KeyType`);
- the code: `Plan::new`, `Plan::coarse_blocks`, the P2 logic in plan.rs's module docs and
  `from_key_types`, `index.rs`, `lists.rs`, `exchange.rs` (`SourceExchange::new`,
  `MultipoleExchange::new`, `CoarseExchange::new`), `evaluator.rs`;
  `fmm-plan/tests/mpi_regressions.rs` (the scenarios, the list oracle, the exchange
  checks, the recording operator); `fmm-exec/tests/mpi_exec.rs`, `multi_rank.rs`;
  `nd_fmm_validate::scaling::held_boxes`.

Do:
- **`Plan::new` over the set.** The box index and the lists over the locally essential
  keys; every list entry the same key as in Phase 5 (the rows in P1's order); P2's rows
  unchanged. The held-key check: every key a list, a parent-child step or an exchange
  names is held, else a `PlanError` that the caller agrees on every rank (as plan errors
  are today). Remove or adapt the coarse-block check as the design says.
- **The exchanges.** `SourceExchange` and `MultipoleExchange` built from the new set; their
  neighbours and sizes reported as today (`exchange_traffic`, `exchange_sizes`).
  `CoarseExchange` keeps working until T5 (it may still read the partition map's blocks;
  say how it gets every block's multipole without the replicated blocks, or keep the
  blocks' keys replicated until T5, as the design orders).
- **`nd-fmm-exec`** compiles and runs unchanged, apart from what the new plan API requires;
  `scaling`'s `held_boxes` reports the new split (own, halo, shared ancestors, other
  ghosts).
- **Tests.** `mpi_regressions`: the list oracle and the exchange checks on the new set; the
  recording operator still sees every pair once, the same calls and groupings, on every
  scenario at 1, 2 and 4 ranks; a check that the held set is minimal enough (no key held
  that no list, halo or parent-child step names, apart from what the design allows). The
  invariance test of T2 and the one-rank comparisons of `mpi_exec` pass unchanged.
- **Measured** (release; reported, not asserted):
  - held boxes per rank (max, mean) against P at fixed N/P: locust 2–72 timed runs and
    up to 512 oversubscribed; (Kathleen) 40, 80 and 160 ranks (40 × 1 per node) and 320,
    640 and 1,280 oversubscribed on the 4 nodes, on the cube, the Plummer sphere and the
    clusters at 2.5 × 10⁴ points per rank where memory allows (else 10⁴);
  - `Plan::new`'s memory per rank at those P, and its time to 72 (locust) and 160
    (Kathleen) ranks, against Phase 5;
  - the evaluation time at 72 ranks on locust and at 80 and 160 on Kathleen against Phase
    5 (it should not change much until T5 removes the gather).

Tests that define done:
- `mpi_regressions`, `mpi_exec`, `mpi_threading` pass at 1, 2, 4 and 8 ranks on the M3 Max
  and on locust and in `run-tests-mpi`; `multi_rank` (release, ignored) passes at 2, 4
  and 8 with 0 values differing from the one-rank `Fmm` in rank order; the device gate
  (`tests/device_ranks.rs`) passes at 2 and 4 ranks on the CPU runtime.
- T2's invariance check: 0 differing values at 1, 2, 4 ranks (CI), at 512 against 64 on
  locust and (Kathleen) at 160 against 20.
- Held boxes per rank bounded at fixed N/P within T1's bound, from 2 to 512 ranks on
  locust and (Kathleen) from 40 to 1,280 ranks (measured table), against the Phase 5
  growth; at 512 oversubscribed ranks far below the 37,449 of Phase 5.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks; `cargo doc
  --no-deps` (nd-fmm-plan's docs updated);
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks on the M3 Max and on locust;
- (Kathleen) the correctness job of T2 at 2 nodes.

Do not:
- change any list entry, row order or value (bit for bit is the acceptance);
- change the coarse gather or the global pass beyond what the new set forces (T5);
- add a collective that some ranks skip;
- change `nd-octree` beyond fixing a defect found here (in its own commit, with its test).
