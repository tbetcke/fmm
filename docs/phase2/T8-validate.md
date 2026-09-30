# Phase 2 / T8 — nd-fmm-validate: tables report and table-path accuracy

Read first: root CLAUDE.md, fmm-validate/CLAUDE.md, the nd-fmm-validate code
(`accuracy`, `metrics`, examples `accuracy` and `timing`), the nd-fmm-tables API,
docs/design/laplace-fmm-plan.md §3.6, §4 and §7 (Phase 1 timing, C3.1),
docs/phase2/README.md.

This task measures what the tables cost and what they buy. It does not test them
further. It gives Phase 3 its CPU default for M2L per p, and it records the build,
memory and cache figures for the design document.

Do:
- Add nd-fmm-tables (`workspace = true`) to nd-fmm-validate. In fmm-validate/CLAUDE.md,
  add "Phase 2: nd-fmm-tables" to the allowed dependencies. No other dependency.
- Replace `accuracy::v_list_offsets()` with `nd_fmm_tables::geometry::m2l_offsets()`,
  or check in a test that the two agree in content and order. Choose one and say which
  in the PR.
- Example `tables`, in release mode, f64 unless stated, printing Markdown tables and
  the machine (CPU model, core count):
  - Build time per family (M2M with L2L, dense M2L, class M2L, rotation) for
    p ∈ {4, 8, 12, 16, 20}, plus p = 30 for the octant tables.
  - Memory per family and form, in f64 and f32.
  - Time per application, as the median of repeated batches timed with
    `std::time::Instant`, as in the `timing` example:
    - M2L: dense table, class form, table-driven rotation, `nd_fmm_ref::rotation` and
      `nd_fmm_ref::direct`;
    - M2M and L2L: dense octant table, table-driven rotation and both nd-fmm-ref
      methods;
    - for p ∈ {2, 4, 6, 8, 10, 12, 16, 20};
    - a fitted exponent for p ≥ 8, and the p above which table-driven rotation beats
      the dense table.
  - Cache: `load_or_build` cold against warm, at p = 8 and p = 16, in a directory under
    `target/` that the example removes afterwards.
  - A recommendation for the Phase 3 per-pair CPU M2L (dense table or table-driven
    rotation) as a function of p. State that it holds for single-threaded per-pair
    application only, and that the batched GEMM path of Phase 4 changes the comparison.
- Extend the `accuracy` example and `sweep` with a table path, for example a `--tables`
  flag. Chains that use the tables must reproduce the Phase 1 table (design §7,
  "Single-translation accuracy") to all printed digits. If a chain's geometry is not an
  octant or a V-list offset, keep it on nd-fmm-ref and say so in the output.
- Tests:
  - a smoke run of the table path at p ≤ 4 with few points, whose errors equal the
    reference path's to relative 1e-12;
  - the offset agreement, if you chose that option.

Must pass: the root CI commands and the stricter workspace checks;
`cargo test -p nd-fmm-validate`; `cargo run --release -p nd-fmm-validate --example
tables` and `--example accuracy -- --tables`, each within a few minutes. Paste both
outputs into the PR.

Do not:
- assert timings in tests;
- commit CSV, cache files or other output;
- add criterion or any other dependency without asking;
- make any library crate depend on nd-fmm-validate;
- change nd-fmm-tables. If a measurement shows a defect, report it.
