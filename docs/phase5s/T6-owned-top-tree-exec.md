# Phase 5S / T6 — the owned top tree in `nd-fmm-exec` (C5S.3; S2, part 2)

T5 replaces the coarse gather and the replicated global pass with the owned top tree in
`nd-fmm-plan`, with new `HostData` events for the reduction and the replicated top. The
host operator ignores events, so the host path already runs on it. What remains:
- the device operator must mirror the new movements, so that the device on several ranks
  stays within Phase 4's bounds and its transfers stay a formula (device-path.md §14);
- `Fmm`, `StageTimings`, `ExchangeTraffic`/`exchange_sizes` and the overlap option must
  report the new stages and traffic;
- the `scaling` harness must measure them per rank.

This task does that as `docs/design/scale-out.md` §4 (T1, signed off) specifies.

Read first:
- fmm-exec/CLAUDE.md, fmm-validate/CLAUDE.md, root CLAUDE.md;
- docs/phase5s/README.md (requirements 2, 5, 8, 9) and `docs/design/scale-out.md` §4, §9;
- device-path.md §14 (the device on several ranks: events, packed buffers, two syncs,
  `tests/device_common::expected_evaluation`);
- distributed-fmm.md §7 (the device on several ranks) and §8.6 (the C5.2 measure);
- T5's merged API in `nd-fmm-plan`;
- the code: `nd_fmm_exec::fmm` (`build`, `evaluate`, `StageTimings`, `OverlapTimes`,
  `exchange_traffic`, `exchange_sizes`), `device.rs` (`ExchangeLists`, the handling of
  `SendMultipoles`, `ReceivedCoarse`, `ReceivedMultipoles`, `DataKind`,
  `DeviceCounters::received_levels`), `tests/device_common`, `tests/device_ranks.rs`,
  `tests/mpi_exec.rs` (the shadow scenario, the device scenarios), `tests/multi_rank.rs`;
  `nd_fmm_validate::scaling` (`STAGES`, `RankTraffic`, `MemoryModel`, `Report`).

Do:
- **The device operator.** At each new event: gather the multipoles to send into a packed
  buffer on the device and download it; upload what is received, packed, and scatter it
  on the device (`movement::scatter_columns`, an assignment, as today); the replicated
  top likewise. Extend `ExchangeLists`, `DataKind` and the counters, and the formula of
  `expected_evaluation` (transfers and syncs per evaluation; say whether the sync count
  changes from Phase 5's two, and why). Device memory added per rank reported.
- **The shadow check.** The shadow operator of `tests/mpi_exec.rs` learns of the new
  movements only through the events and equals the plain operator bit for bit on 1, 2
  and 4 ranks; with the new events disabled it differs on 2 ranks.
- **`Fmm` and timings.** `StageTimings` (and `OverlapTimes`) report the reduction and the
  replicated top in place of the coarse gather; `exchange_traffic` and `exchange_sizes`
  report their messages and bytes per level; `BuildTimings` reports any new build part.
  The overlap option covers the reduction (bit for bit).
- **The harness.** `nd_fmm_validate::scaling`: the new stages and traffic in `STAGES`,
  `RankTraffic` and the report; the coarse-gather columns retired or kept as zero with a
  note; `MemoryModel` updated to the new held set and buffers (still labelled *model*);
  `tools/scaling/run.sh` and its Kathleen counterpart unchanged in their interface.
- **Measured** (release; reported, not asserted): T10's sweeps at 8 and 72 ranks on locust
  and (Kathleen) at 40, 80 and 160 ranks, Phase 5 against Phase 5S: evaluation time stage
  by stage, traffic and memory per rank; traffic and memory at 512 (locust) and 1,280
  (Kathleen) oversubscribed ranks, not timed; the device gate's transfer table at 2 and 4
  ranks.

Tests that define done:
- `mpi_exec` (every scenario, the shadow check, the device scenarios on the CPU runtime),
  `mpi_threading` and `multi_rank` pass at 1, 2, 4 and 8 ranks on the M3 Max and on
  locust, with 0 values differing from the one-rank `Fmm` in rank order;
- the device gate `tests/device_ranks.rs` passes at 1, 2 and 4 ranks on the CPU runtime
  (M3 Max), on CUDA (locust) and on Metal at 2 ranks (by hand), with every transfer equal
  to the extended formula;
- overlap bit for bit the blocking path on every scenario; the C5.2 measure reported;
- T2's invariance check passes at 1, 2, 4 ranks (CI), at 512 against 64 on locust and
  (Kathleen) at 160 against 20;
- `fmm-validate/tests/scaling.rs` passes at 1 and 2 ranks with the new fields.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
  `cargo clippy -p nd-fmm-exec --all-targets --features cpu -- -D warnings`;
  `cargo check -p nd-fmm-exec --features cuda`;
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks on the M3 Max and on locust;
- on locust (by hand): `cargo clippy -p nd-fmm-exec --all-targets --features cpu,cuda --
  -D warnings` and the CUDA device gate;
- (Kathleen) the correctness job of T2 at 2 nodes.

Do not:
- change `nd-fmm-plan` beyond fixing a defect found here (in its own commit, with its
  test);
- time the device on several ranks (Phase 5 decision 3 stands);
- change any output bit of the host path, or the device's results beyond Phase 4's bounds.
