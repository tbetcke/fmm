# Phase 3S / T6 — nd-fmm-exec: P2P through nd-fmm-simd (C3S.5)

Use the SIMD kernel in the Laplace FMM. `LaplaceOperator`'s P2P calls
`P2pKernel::evaluate` where it called `nd_fmm_ref::p2p::p2p`. The Phase 3 path stays
selectable. Every Phase 3 accuracy gate and identity is re-run with every ISA.

Read first: root CLAUDE.md, fmm-exec/CLAUDE.md, fmm-simd/CLAUDE.md;
docs/design/simd-p2p.md §5.5 and §6; docs/phase3s/README.md ("Design decisions", "Exit
gate" C3S.5); docs/phase3/README.md ("Exit gate" C3.1, C3.2, C3.3 and C3.5, "Error
measures"); CONVENTIONS §3.13 with the T1 addition; fmm-exec/src/operator.rs (`Kernels::p2p`,
`p2p_target`, `p2p_pair`, the module docs on order, threads and scratch),
fmm-exec/src/fmm.rs (`FmmBuilder`, `SettingsError`, `Fmm` accessors), the T8, T9, T10
and T11 tests in fmm-exec/tests/; the `nd-fmm-simd` public API.

Do:
- Add `nd-fmm-simd` to nd-fmm-exec's dependencies (`workspace = true`). Update the
  "Allowed dependencies" of fmm-exec/CLAUDE.md. Also update its rule that formulas are
  reached only through nd-fmm-tables and nd-fmm-ref, to include nd-fmm-simd for P2P.
- `P2pChoice { Auto, Reference, Isa(Isa) }` (design §6; the final name is yours):
  - `LaplaceOperator::with_p2p(choice)`; `new` uses `Auto`;
  - `FmmBuilder::p2p_kernel(choice)`, default `Auto`; an unavailable ISA gives a
    `SettingsError` on every rank. Agree it by the same all-reduce that `build` already
    uses for input errors: on a cluster with mixed CPUs one rank may lack the ISA. Add
    no new collective;
  - `Fmm::p2p_kernel()` returns the resolved choice (for `Auto`, the ISA picked) for
    reports.
- `Kernels::p2p` calls the kernel for `Auto` and `Isa`, and `nd_fmm_ref::p2p::p2p` for
  `Reference`. The mapping of s ≠ t sources into scratch, the order of the near list
  and the per-leaf calls stay as they are. Add the bound `T: SimdScalar` where it is
  needed.
- Gathered form (design §6): measure, on the T9 uniform problem at p = 3 in f64 and
  f32, whether one call per target leaf over its whole gathered near field beats one
  call per source leaf. Adopt it only if it is more than 5% faster on the leaf stage.
  Its scratch capacity must be fixed at construction, flushing in order when full.
  Report the measurement either way. Thanks to chunk invariance the output must be
  bit-identical; test that if you adopt it.
- Update the crate docs: the operator table's P2P row, the "Scratch" and "Threads"
  sections if anything changed, the `P2pChoice` choice and its default, and the
  reproducibility notes of design §5.5. The examples in `nd-fmm-validate`
  (`fmm_accuracy`, `calibrate`) print `Fmm::p2p_kernel()` next to the
  `ThreadingReport` and accept `--p2p auto|reference|<isa>`.

Tests that define done (for `Reference` and for every ISA the machine offers; each
prints the ISAs it ran):
- Operator, the T8 check: P2P of `LaplaceOperator` on levels 2, 9 and 16 against
  `nd_fmm_ref::p2p` at the absolute frames, within 1e-13 relative to the term
  magnitudes (f64), with and without gradients. The f32 check of T8 at its tolerance.
- Per-pair = batched: `PerPair<LaplaceOperator>` equals the batched path bit for bit
  for every P2P choice (chunk invariance makes this hold).
- FMM, every scenario of tests/mpi_exec.rs (T8, T9, T10, T11), through
  `evaluate_threaded`:
  - outputs bit-identical at 1, 2, 4 and 8 threads for every choice;
  - for every ISA, the output within 1e-13 (f64) and 1e-6 (f32) relative L2 of the
    `Reference` output of the same `Fmm` settings, potential and gradient.
- The ignored release gates of T9 (tests/accuracy.rs) and T11 (tests/adaptive.rs)
  pass with `Auto`, and their errors are within 1% (f64) and 2% (f32) of the
  `Reference` run.
- Settings: an unavailable `Isa` (`Neon` on x86_64, `Avx2` on aarch64) is rejected
  with `SettingsError`; the PR shows that the check goes through `build`'s existing
  agreement of input errors; `Auto` is the default of both `new` and `FmmBuilder`.
- No allocation per pair: the scratch-capacity checks of T8 and T10 still hold.

Must pass:
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug run under a minute; keep
  the ISA repetitions to the small scenarios if needed);
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored`, on the M3
  Max. No x86_64 machine runs it. On x86_64 the debug scenarios of tests/mpi_exec.rs in
  the existing CI job cover AVX2 inside the FMM; say so in the report;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate`;
- `cargo run --release -p nd-fmm-validate --example fmm_accuracy` with `--p2p auto`
  and `--p2p reference`, on the M3 Max;
- tests/mpi_exec.rs on 2 ranks with 2 threads per rank, by hand, under an external
  timeout;
- `cargo clippy -p nd-fmm-exec -p nd-fmm-validate --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-exec --no-deps` without warnings;
- the root checks and the stricter workspace checks.

Report: the ISAs run on each machine, the T9 and T11 errors with `Reference` and `Auto`
side by side, the gathered-form measurement, and the leaf-stage time at p = 3 and 8 with
both choices (not asserted).

Do not:
- change the accumulation order of any target, or any operator other than P2P;
- add `unsafe` to nd-fmm-exec;
- change the default `max_points_per_leaf` or any other default besides adding
  `P2pChoice::Auto`;
- remove the `Reference` path;
- touch nd-fmm-plan, nd-fmm-tables, nd-fmm-ref or the kernel itself. If a test exposes a
  kernel defect, report it and stop.
