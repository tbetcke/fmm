# Spikes

Throwaway studies, each with its report (`SPIKE_REPORT.md` or `REPORT.md`) and raw results.
They are finished: what each decided lives in the product crates, the design documents and
`docs/helmholtz/`. They are kept because tests, fixture tools, rustdoc and reports cite their
sources, and some are rerun as references.

## The spikes workspace

Since Helmholtz CHA.1 (2026-11-19; review 2026-11-12, K3 and U1; the user: every spike kept,
"One spikes workspace") the spikes are a workspace of their own, `spikes/Cargo.toml`, outside
the root's (`exclude = ["spikes"]` there):

- its lock file `spikes/Cargo.lock`, committed; seeded from the root's at CHA.1, so the crates
  both use resolve to the same versions;
- its `[workspace.dependencies]`: the product crates by path, the root's CubeCL pins repeated
  (move them together), green-kernels (pinned by commit) and pulp, for `p2p-simd` alone, and
  green-kernels' `[patch]` of rlst to 0.10.2 (`docs/design/simd-p2p.md` §8.4);
- its target directory `target/spikes` at the root (`spikes/.cargo/config.toml`).

No root check builds a spike, and CI never does. Run every spike command from this directory;
the commands in the reports (`cargo run -p nd-fmm-spike-<name> …`) work as written from here.

```sh
cd spikes
cargo check --all-targets                 # every spike, with each one's default features
cargo test -p nd-fmm-spike-<name>         # one spike's tests (most have ignored long ones)
```

When a task reruns a spike, or changes a product crate that a spike uses in a way the spike
cites, rebuild the spikes here first. A spike that no longer builds is fixed in that task or
recorded as broken in this file, never left silently broken.

## The spikes

| Spike | Study | Uses | Cited by (outside `spikes/`) | Build and run |
| --- | --- | --- | --- | --- |
| `cubecl-gemm` | Phase 0 T6, Phase 4 T2: CubeCL GEMM throughput on dense M2L shapes | CubeCL, cubek-matmul | `fmm-validate/src/peaks.rs` and the `*_kernels` examples (its results as the reference shapes), root CLAUDE.md (the CubeCL pin) | `cargo run -p nd-fmm-spike-cubecl-gemm --release` (Metal and CPU by default; `--no-default-features --features cuda` on locust) |
| `p2p-simd` | Phase 3S T2, T7; Helmholtz CH2.4b, CH3.4b: SIMD P2P, inverse square roots, against green-kernels | nd-fmm-math, -ref, -simd, -validate, green-kernels, pulp | **C3S.4's reference of D9** (`results-m6.md`, `tools/m6/machine.md`), `fmm-simd`'s rustdoc, the reports of CH2.4b, CH3.4b, CHR.2 and Phase 5N–6 | `cargo run --release -p nd-fmm-spike-p2p-simd --example compare` (also `yukawa`, `helmholtz`); `cargo test -p nd-fmm-spike-p2p-simd` runs the three examples' smoke tests |
| `device-arith` | Phase 4 T3: device arithmetic per backend, the r² = 0 argument, P2P candidates | CubeCL, nd-fmm-kernels and the host crates | `docs/CONVENTIONS.md`, `fmm-kernels`' rustdoc and CLAUDE.md, `fmm-exec/tests/operator/device_leaf.rs` | `cargo run --release -p nd-fmm-spike-device-arith --features cpu,metal -- …` (`REPORT.md`); `cargo test -p nd-fmm-spike-device-arith --features cpu` |
| `download-path` | Phase 4S T11: the device download and upload paths | CubeCL, nd-fmm-kernels, -validate | `fmm-kernels`' rustdoc and CLAUDE.md, `fmm-bench/results/phase4s-t11-download-path.md` | `cargo run --release -p nd-fmm-spike-download-path --features metal -- --backends metal --precisions f32` |
| `plane-wave` | Phase 6 T10: the plane-wave M2L with merge and shift on one level | nd-fmm-math, -ref, -tables, -simd | `fmm-tables/src/plane_wave/` rustdoc, `tools/fixtures/plane_wave_quadrature.sh` (moved from here) | `cargo test -p nd-fmm-spike-plane-wave --release -- --ignored --nocapture`; `cargo run --release -p nd-fmm-spike-plane-wave --example model` |
| `rotation-blocks` | Helmholtz CH0.1, S1: rotation blocks of large degree | nd-fmm-math | `fmm-math`'s rustdoc, `tools/fixtures/gen_helmholtz_angular.py` (its method and `fixtures/reference.json`), `docs/CONVENTIONS.md` | `cargo test -p nd-fmm-spike-rotation-blocks --release -- --include-ignored` |
| `coaxial` | Helmholtz CH0.2, S2: the coaxial translation coefficients | – | `fmm-math/src/coaxial/`, `tools/fixtures/gen_helmholtz_coaxial.py` (`tools/reference.py`, `fixtures/reference.json`), `docs/CONVENTIONS.md` | `cargo test -p nd-fmm-spike-coaxial --release -- --include-ignored` |
| `radial` | Helmholtz CH0.3, S3: the radial functions | – | `fmm-math/src/radial.rs`, `tools/fixtures/gen_helmholtz_radial.py` (`tools/reference.py`, `fixtures/reference.json`) | `cargo test -p nd-fmm-spike-radial --release -- --include-ignored` |
| `pw-quadrature` | Helmholtz CH0.4, S4: the quadratures of the exponential M2L | nd-fmm-ref, -tables, num-complex, rayon (dev: `plane-wave`) | `fmm-ref`'s generator (bit for bit against `examples/generated.rs`, by hand; `fmm-ref/CLAUDE.md`), `tools/fixtures/helmholtz_pw_quadrature.sh` (FMM3D's source from `tools/fetch.sh`), `fmm-tables`' rustdoc | `cargo test -p nd-fmm-spike-pw-quadrature`; `cargo run --release -p nd-fmm-spike-pw-quadrature --example generated` |
| `diagonal` | Helmholtz CH0.6, S6: the diagonal M2L's stability; rerun at CH4.0e and CH4.0i as the prototype | `truncation` | the proposal, the reports of CH4.0e and CH4.0i, `tools/helmholtz/ch4-0e-tables.py` | `cargo test -p nd-fmm-spike-diagonal --release -- --include-ignored`; examples `sweep`, `times`, `summary` |
| `truncation` | Helmholtz CH0.7, S7: the truncation degree in double-double | – | `fmm-exec/src/helmholtz/degree.rs` and its test, `fmm-validate`'s calibration, `fmm-math/src/coaxial/` | `cargo test -p nd-fmm-spike-truncation`; `--example sweep` |
| `census` | Helmholtz CH0.8a, S8: the workload census, the leaf rule | nd-octree, nd-fmm-plan, -validate, mpi | `fmm-validate`'s `helmholtz_workloads` (ported from here; **`tests/ported_workloads.rs` checks it**), `fmm-exec/tests/helmholtz/leaf_rule.rs`, `fmm-exec`'s rustdoc | `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-spike-census`; `--example census` (MPI; on the M6 `HWLOC_COMPONENTS=-darwin`) |
