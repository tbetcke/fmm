# Phase 5S / T1s — rlst 0.10.0 and CubeCL 0.11.0 (prerequisite of T2)

Added on 2026-10-11 at the user's request. Two dependencies of the workspace have new
releases:

- **rlst 0.10.0** (crates.io, 2026-10-11; codeberg rlst/rlst). It carries PR #40: `parsort`
  finds its splitters by histogram sort with sampling, and rank r of P receives exactly the
  records at global positions ⌊rN/P⌋..⌊(r+1)N/P⌋. rlst 0.9.0's four local quantiles per
  rank sent a quarter of all keys to each of four ranks on a random share. That was 95% of
  `Octree::new` at 160 Kathleen ranks and T0's 2 GB ranks (`docs/design/scale-out.md`
  §1.4, §7.1). Measured in T1 with the PR's branch, the octree unchanged key for key:
  `Octree::new` at 160 ranks went from 2,607 to 73 ms at N = 10⁷ and from 9,656 to 689 ms
  at N = 10⁸, and the heaviest rank's peak from 1.94 GB to 0.19 GB.
  Other changes in 0.10.0:
  - `OVERSAMPLING` changes meaning;
  - Burn 0.22 for the `burn` feature, which the workspace does not use;
  - repository links now point to codeberg.org/rlst.

  It still depends on `mpi` 0.8.2 and `mpi-sys` 0.2.4.
- **CubeCL 0.11.0** with **`cubek-matmul` 0.3.0** and **`cubek-std` 0.3.0** (crates.io,
  2026-10-06): the releases after the pre-release `=0.11.0-pre.4` / `=0.3.0-pre.4` that
  Phase 4 T2 chose on 2026-10-03 for f64 on CUDA (spikes/cubecl-gemm/SPIKE_REPORT.md).

This task moves both pins, ports what the releases require, and shows that nothing else
moved. Host outputs must stay bit for bit; device results stay within Phase 4's bounds.
It changes no algorithm. It comes before T2 so that every Phase 5S measurement uses the
new releases, and T3a's "before" is rlst 0.10.0.

Read first:
- root CLAUDE.md (the dependency rules, "Checks", the Kathleen and locust sections),
  octree/CLAUDE.md, fmm-plan/CLAUDE.md, fmm-exec/CLAUDE.md, fmm-kernels/CLAUDE.md;
- docs/phase4/T2-cubecl-011.md (how the last CubeCL move was done and checked) and
  `docs/design/cubecl-upstream.md` (the observations to re-check);
- rlst's `CHANGELOG.md` for 0.10.0, and the diff of `rlst` 0.9.0 → 0.10.0 in the modules
  the workspace uses: `distributed_tools` (`parallel_sort`, `array_tools`,
  `ghost_communicator`, `all_to_allv`, `sort_to_bins`), `threading`, the array types and
  macros;
- CubeCL's release notes from 0.11.0-pre.4 to 0.11.0, and the `cubek-matmul` /
  `cubek-std` 0.3.0 changes, with their manifests (which `cubecl` they pin).

Do:
1. **rlst, one commit with its `Cargo.lock` change:**
   - `[workspace.dependencies]` `rlst = "0.10.0"`;
   - octree/Cargo.toml's own entry (`rlst = { version = "0.10.0", features = ["mpi"] }`;
     octree declares its own until migrated, root CLAUDE.md);
   - the `[patch."https://github.com/linalg-rs/rlst.git"]` entry for green-kernels
     (spikes/p2p-simd): to 0.10.0 if its `simd` module still serves green-kernels, else
     keep 0.9.0 for that spike alone and say why (both may then be in `Cargo.lock`; only
     the spike may use 0.9.0);
   - port whatever the API requires. `Octree::new` keeps calling `parsort`; its output
     is now exactly balanced, and the octree's leaves, blocks, partition and keys do not
     depend on the sort's split, because O1 counts globally;
   - update the comment above the patch, and every live mention of rlst 0.9.0: root
     CLAUDE.md ("Navigating the code": the registry path `rlst-0.10.0`),
     fmm-plan/CLAUDE.md ("Dependencies"), docs/phase5s/README.md ("Design decisions",
     "Dependencies"), docs/design/workspace-structure.md (the dependency rows and the
     rule). Historical briefs and results keep the version they ran with.
2. **CubeCL, a second commit with its `Cargo.lock` change:**
   - `cubecl = "=0.11.0"`, `cubek-matmul = "=0.3.0"`, `cubek-std = "=0.3.0"`. Keep exact
     pins, `default-features = false` and the features (`std`, `stdlib`;
     `std`, `multi-level`). Confirm from the manifests that the `cubek-*` releases pin
     `cubecl =0.11.0`, and say so in the comment above the entries (a release now, not a
     pre-release);
   - port `nd-fmm-kernels`, `nd-fmm-exec`'s device path and the spikes
     (`spikes/cubecl-gemm`, `spikes/device-arith`, `spikes/download-path`). Change only
     what the API requires;
   - report which crates `Cargo.lock` gained, lost and changed, and the `tracel-llvm`
     bundle version the CPU runtime now downloads. CI's caches key on `Cargo.lock`, so
     the first run is cold; note its time;
   - re-check every entry of `docs/design/cubecl-upstream.md`, and the 0.11.0-pre.4
     notes in fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md and `spikes/device-arith/Cargo.toml`
     (`cubecl-metal` with `metal-native` did not compile). Mark each still present,
     fixed, or changed, with the 0.11.0 file and line; nothing is removed, only annotated;
   - update every live mention of `0.11.0-pre.4` / `0.3.0-pre.4`: root CLAUDE.md
     ("Working rules"), fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md,
     docs/design/workspace-structure.md, the `DeviceInfo` string fmm-kernels/CLAUDE.md
     quotes. Historical briefs, results and spike reports keep the version they ran with.
3. **Show that nothing else moved:**
   - host: every `nd-fmm-exec` output test, `mpi_exec`, `multi_rank` and the
     `nd-fmm-plan` suites unchanged, bit for bit where they check bits;
   - the octree: on the random share, the leaves, blocks, partition and keys equal rlst
     0.9.0's key for key (T1's probe did it at 8–512 ranks; repeat with the release at
     1–8 ranks in an MPI example run by hand, not committed);
   - device: every device test within Phase 4's bounds on the CPU runtime (CI), Metal (M3
     Max, by hand) and CUDA (locust, by hand). Where device values change in their last
     bits (the code generator changed), report the largest change against 0.11.0-pre.4
     on the C4.x gate scenarios; no bound may be loosened;
   - the kernels' compile times (`fmm-kernels/tools/compile_times.awk` on CI's log and on
     locust) and the device FMM's evaluation time on locust's H100 (the bench: `tools/bench/run.sh
     --backend host,cuda --precision f32,f64`, one run, labelled) against 0.11.0-pre.4.
     A regression beyond ±10% is investigated or reported upstream through
     `cubecl-upstream.md`, not hidden.
4. **Measured** (release; reported, not asserted), rlst 0.9.0 against 0.10.0:
   `Octree::new` and its first sort at 8 and 72 ranks on locust and (Kathleen) at 40 and
   160 ranks, N = 10⁷ and 10⁸, the cube and the Plummer sphere, random share: time, keys
   received per rank (max/mean), peak resident size. A short `tools/scaling/run.sh` run is
   enough; T1's probe numbers are the reference.

Tests that define done:
- The root checks and the stricter workspace checks pass on the M3 Max (spikes
  included, since `--workspace` builds them); `run-tests-simd`, `run-tests-kernels` and
  `run-tests-mpi` pass in CI.
- The `nd-fmm-kernels` checks of root CLAUDE.md pass: clippy with `cpu,metal`, the CUDA
  type-check, the CPU-runtime tests; Metal's ignored tests by hand on the M3 Max; on
  locust, the CUDA tests and clippy of root CLAUDE.md ("On locust"), the device gates.
- `.github/scripts/run-mpi-tests.sh` at 2 and 4 ranks; the `nd-octree` MPI examples at
  1–8 ranks by hand.
- (Kathleen) `check.sbatch root strict` and `mpi.sbatch` at 80 ranks on 2 nodes (`test`).
  T2's correctness job does not exist yet.
- Every host output bit for bit as before (the suites above); the octree key for key
  (step 3).

Must pass: `cargo fmt --all`; everything under "Tests that define done".

Do not:
- change any algorithm, default, tolerance or test bound;
- move another dependency (mpi stays 0.8.2), or add a dependency;
- use a git or path source for rlst or CubeCL, or a fork;
- mix the two upgrades in one commit (a regression must bisect to one of them);
- edit historical briefs, results or spike reports beyond an annotation.

Budget: Kathleen at most 5 node-hours; locust by hand.
