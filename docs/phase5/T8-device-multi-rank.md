# Phase 5 / T8 — the device path on several ranks (C5.1 device; C5.2 device-resident ghost buffers)

The device FMM of Phase 4 runs on one rank and refuses more with `DeviceNeedsOneRank`.
With the hook of T7, the device operator learns of every host-side data movement of the
evaluator. It can then keep its stores on the device on any rank count:
- download exactly what an exchange sends, packed on the device;
- upload exactly what it receives, and scatter it on the device.

This task does that, opens a device per rank, agrees device errors across ranks, and
removes `DeviceNeedsOneRank`.

The M3 Max has one GPU, so this is a **correctness task** (docs/phase5/README.md,
decision 3):
- every rank opens the CubeCL CPU runtime, or the ranks share Metal;
- transfers and syncs are counted, not timed as scaling figures.

Read first:
- root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-kernels/CLAUDE.md;
- docs/phase5/README.md ("Requirements" 3, 4, 7, 9; "Design decisions": "Device ranks",
  "Threads and BLAS", "Errors"; "Exit gate" C5.1 device and C5.2);
- docs/design/distributed-fmm.md, signed off: §7 (the device on several ranks), §9, §10,
  and §12 for T8;
- docs/design/device-path.md §4 (residency, the one-rank argument), §8 (syncs), §11
  (threads), §12 (errors), §14;
- fmm-exec/src/device.rs (`DeviceOperator::new` and its ghost-free assertion,
  `load_points`, `begin_evaluation`, `read_output`, the transfer accounting, the launch
  formula of `tests/device_common`), fmm-exec/src/fmm.rs (`DeviceNeedsOneRank`, device
  opening at step 1);
- `nd_fmm_kernels::movement` (gather, scatter, scatter-add, zero) and `Device::open`;
- the T7 hook and its shadow test.

Do:
- **`DeviceOperator::host_data`**, as the design §7 specifies. The expected shape, which
  the design confirms or corrects:
  - the source exchange: the host source chunks already hold the coordinates and this
    evaluation's charges, so nothing is downloaded before it. After it, the received
    ghost tail is uploaded in one call;
  - the coarse gather and the multipole exchange:
    - before: a gather kernel packs the values to send into a device buffer, and one
      download moves it into the host store at the send indices;
    - after: one upload of the received values, packed, and a scatter kernel writes
      them into the device level buffers;
  - `reset`: as in Phase 4 (`begin_evaluation`'s zero kernels).

  Every packed buffer is allocated at build and sized from the exchanges' index lists.
  An evaluation allocates nothing on the device.
- Remove the ghost-free assertion from `DeviceOperator::new`, and `DeviceNeedsOneRank`
  from `build` and `SettingsError`. Upload the ghost slots' index lists with the other
  views at build.
- **A device per rank**: the index from the rank's position on its node (`split_shared`
  of the communicator, built once in `build`), modulo the device count, as the design
  fixes. `Fmm::device_report` names the device and the local rank.
  - On the CPU runtime: ranks × units per cube stay within the cores (design §7; the
    `threads(n)` cap per rank). A layout that would exceed it is capped, and the report
    says so.
  - With Metal, every rank opens the one GPU.
- **Errors** (requirement 4, decision 12): a device error at a mid-evaluation sync is
  agreed as the design fixes, so that no rank blocks in an exchange while another has
  failed. The `Fmm` returns the error on every rank from that `evaluate` on, as in
  Phase 4. Build-time errors keep going through step 1's agreement.
- **Transfer accounting**: the counters gain the exchange transfers by data kind (ghost
  sources, sent and received multipoles per level, coarse blocks), and the syncs. The
  formula of `tests/device_common` is extended to several ranks from the exchanges'
  index lists, as the design §7 states it.
- **Tuning on several ranks** (C4.7): every rank must resolve the same choices, or the
  choices must not affect the cross-rank bits. Follow the design: for example, rank 0
  tunes and broadcasts at build, or the cache is shared, or each rank tunes for its own
  device with the result agreed. Say what you did.
- Any new `nd-fmm-kernels` launch the design names (for example a gather or scatter for
  packed buffers that `movement` lacks), with its kernel test on the CPU runtime, in
  that crate's style. Nothing else in `nd-fmm-kernels` changes.
- Docs: the device section of the crate docs and fmm-exec/CLAUDE.md (the device runs on
  any rank count; residency on several ranks; transfers and syncs per evaluation; a
  device per rank; errors). The Phase 4 sentences "a device backend runs on one rank"
  and "until C5.1" go.

Tests that define done (every test prints the backends and the ranks it ran):
- **Every kind on the host fallback** (requirement 8 of Phase 4, now on several ranks):
  bit for bit the host path on the same ranks, on every `tests/mpi_exec.rs` scenario,
  at 1, 2 and 4 ranks, on the CPU runtime (`--features cpu --release`).
- **Every kind on the device**, on the same scenarios and ranks:
  - within the Phase 4 FMM bounds of the host on the same ranks (relative L2 1e-12 in f64,
    1e-5 in f32, φ and ∇φ);
  - two evaluations and two builds bit-identical;
  - the transfers and syncs per evaluation equal the extended formula.
- **The C5.1 device gate** (ignored, release, in `tests/device_fmm.rs` or a new
  executable with one MPI test): the cube and the Plummer sphere at N = 10⁴ (CPU runtime,
  f64 p = 8; f32 p = 3) on 2 and 4 ranks, against the host on the same ranks and against
  the one-rank host run (requirement 2's tolerance). Metal f32 p = 3 and 8 at N = 10⁵ on
  2 ranks sharing the GPU, by hand, outside the sandbox.
- **Errors**: a device error injected on one rank at a mid-evaluation sync, through a
  test hook, gives the agreed error on every rank within the external timeout. With no
  test hook available, say how else it was checked.
- One rank is unchanged: the Phase 4 transfer formula (charges up, output down, one
  sync) still holds there. If the design found that the hook's events add syncs on one
  rank, the formula and the report say so.

Must pass:
- `cargo fmt --all`, the root checks and the stricter workspace checks;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged), and
  `--features cpu --release` at 1 rank and, by hand, at 2 and 4 ranks under an external
  timeout;
- `cargo clippy -p nd-fmm-exec --all-targets --features cpu,metal -- -D warnings`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc -p nd-fmm-exec --no-deps
  --features cpu`;
- `cargo test -p nd-fmm-kernels --features cpu --release` (and the `run-tests-kernels`
  CI job) if a kernel was added;
- by hand on the M3 Max, outside the sandbox: `--features metal --release -- --ignored`
  at 1 rank, and the Metal gate at 2 ranks;
- the multi-rank CI job, if kept (it builds without device features; nothing changes
  there).

Report:
- the transfers and syncs per evaluation at 1, 2 and 4 ranks for the cube at N = 10⁵
  (p = 8, f32), by data kind, against the formula;
- the device-versus-host and multi-rank-versus-one-rank differences per point;
- what tuning did on several ranks;
- the device memory per rank added by the packed buffers and ghost slots.

Do not:
- time the multi-rank device path as a scaling figure, or claim speed-ups from ranks
  sharing one GPU;
- add overlap of device work and exchanges (T9 decides whether there is a device part);
- change kernels beyond the design's new launches, or the one-rank residency;
- add `unsafe` to `nd-fmm-exec` or a direct `cubecl` dependency.
