# Phase 4 / T8 — M2M and L2L as batched GEMM (C4.4)

The first translations on the device, and the GEMM machinery that dense M2L (T9)
reuses. Per level, over the (level, octant) batches of the `Children` and `Parents`
groupings, one grouped translation (device-path.md §6.4):
- gather the input columns in batch order;
- multiply each octant's columns by its dense table in one grouped GEMM, into a
  temporary;
- add the products into the outputs: a reduction per target in row order for M2M, a
  scatter-add for L2L.

No atomics are needed: the GEMM writes distinct columns of the temporary, and only the
reduction, which owns each target, or the scatter-add, whose targets are distinct over
the level, writes the output. The upward pass runs both the local and the global M2M.

Read first:
- root CLAUDE.md, fmm-kernels/CLAUDE.md, fmm-exec/CLAUDE.md;
- docs/phase4/README.md ("Design decisions": M2M and L2L always dense on the device;
  "Exit gate" C4.4);
- docs/design/device-path.md, signed off: the M2M/L2L and GEMM parts of section 6,
  sections 9 and 13;
- docs/design/fmm-plan-redesign.md §4.3 (the groupings), §6.4 (the GEMM check) and §7.5;
- CONVENTIONS §3.12 ("Child octants", "Matrix layout": column-major, matrix o at
  o (p + 1)⁴);
- spikes/cubecl-gemm (the hand-written `tiled-smem`, `tiled-reg-cols` and
  `tiled-reg-rows` kernels, the library calls, the per-backend layout finding) and its
  report, with the 0.11 notes;
- spikes/device-arith/REPORT.md (the GEMM check);
- `nd_fmm_tables::{M2mTables, L2lTables, MatrixSet}`; `nd_fmm_plan::lists::{Children,
  Parents, GroupedCsr}` (the batch accessors and raw arrays);
- `LaplaceOperator`'s M2M and L2L, and how `Tables` builds the octant tables per
  strategy;
- the T4 gather and scatter kernels, and the T5 device operator.

Do:
- **The hand-written GEMM** in `nd-fmm-kernels`: C = A X into a temporary (β = 0),
  with A n × n (n = (p + 1)², comptime), X n × k, column-major as the level buffers
  and the tables are.
  - Production-quality from the spike's kernels.
  - The cube layout per backend, as the design fixes it (design §6.5: row blocks on
    Metal, column strips on the CPU runtime). Guard shapes for any k, including 0, 1
    and k not a multiple of the tile.
  - It sums over k in ascending order into one accumulator per output, starting from
    zero; the reduction or scatter-add then adds the result into the output once
    (device-path.md §6.4). Accumulating in the GEMM from a gathered copy of the output
    (β = 1) is rejected. The summation order inside each output is fixed and
    documented.
- **The library path**, where the design uses it for these shapes: `cubek-matmul` with
  an explicitly chosen strategy, never the per-call `Auto` (requirement 6).
- **The grouped translation** (`translate::grouped`, device-path.md §6.4), per level:
  - gather the input columns of every octant batch, in batch order;
  - one grouped GEMM over the level's octants, with a tile schedule (group and first
    column of every tile) built at build, in chunks of groups whose temporaries fit in
    the scratch budget;
  - M2M: a reduction per parent that loads its output, adds Y[:, pos(e)] for each
    entry e of its row in row order and stores it once, with the row-to-batch map
    pos(e) built on the host at build and uploaded once. L2L: a scatter-add
    (`movement::scatter_add_columns`), since a child has one parent;
  - M2M: children on l + 1 → parents on l. L2L: parents on l − 1 → children on l.
    Both M2M passes use the same code;
  - three launches per level and chunk: gather, grouped GEMM, reduction or scatter-add
    (device-path.md §8.1);
  - each target gets its octants in order, as the accumulation rule requires (M2M rows
    are by octant; L2L has one parent per child).
- **Tables on the device**: the dense M2M and L2L tables for every strategy (README,
  "Design decisions"). For `Rotation`, the device path builds them in addition to the
  host's rotation tables, through `table_cache` when given. Uploaded once per `Fmm`.
- The device operator runs M2M (both passes) and L2L on the device by default, the host
  fallback selectable.
- GEMM timing on Metal f32, reported only: GFLOP/s and % of peak for the octant shapes
  of the C3.2 cube and the Plummer sphere at p = 3, 8 and 16 (f32), against the spike's
  numbers at the nearest (p, B). Report the gather and scatter time separately.

Tests that define done (CPU runtime f32 and f64; Metal f32 by hand, `#[ignore]`; each
test prints the backends it ran):
- The GEMM against `MatrixSet::apply` column by column, for p ∈ {0, 1, 3, 8, 12, 20}
  and k ∈ {0, 1, 7, 64, 1000} (larger k `#[ignore]`d on the CPU runtime), relative to
  the term magnitudes: within (p + 1)² u_T in f32 and f64. If the internal order
  happens to equal the host loop's, say so and test bit for bit.
- Operator check, as C2.1 and T8 of Phase 3. On parent levels 1, 2, 9 and 15 of a
  dyadic domain, for every octant, the device M2M and L2L against `nd_fmm_ref::direct`
  at the canonical frames:
  - f64, CPU runtime, p ≤ 20: within 1e-14 relative to the term magnitudes, per degree
    in the §3.8 weighting, or within twice the host operator's measured error on the
    same cell, whichever is larger (device-path.md §2);
  - f32, p ≤ 8: within 1e-5 against f64.
- Batches: the result of a level call equals applying every (target, source, octant)
  triple of the rows in row order on the host, within the GEMM tolerance. Each target
  gets every octant once, and ghost and leaf rows stay untouched (bit for bit).
- Grouped against per-octant: a grouped level call equals one gather, GEMM and
  scatter-add per octant in octant order, bit for bit with the same GEMM
  (device-path.md §6.4).
- The global pass: on the coarse levels, the device M2M with the `m2m_global` view
  equals the host's within tolerance.
- Determinism: repeated level calls bit-identical.
- FMM: with M2M and L2L on the device (the other kinds as merged so far, or on the
  host), every `tests/mpi_exec.rs` scenario within the README's FMM bounds of the host
  output, bit-identical across two evaluations. This holds for `Dense` and `Rotation`
  alike, since the device M2M and L2L use the dense octant tables under both. The FMM
  error against the direct sum is within the README's bounds of the host run's at the
  C3.2 points.

Must pass:
- `cargo test -p nd-fmm-kernels --features cpu --release` (within the T4 budget) and
  `--features metal --release -- --ignored` by hand;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` without features (unchanged), and
  `--features cpu --release` (with `-- --ignored` for the gates), and on Metal by hand;
- clippy on both crates, without features and with `--features cpu,metal`;
  `cargo check -p nd-fmm-exec --features cuda`; `cargo doc` without warnings;
- the CPU-runtime CI job, if kept; the root checks and the stricter workspace checks.

Report: the backends run; the maximum errors per backend, precision and level; the GEMM
throughput table against the spike; the gather and scatter share.

Do not:
- use atomics, or a matmul strategy chosen per call by the library;
- change `nd-fmm-tables`, the octant order or the host path;
- start dense M2L (T9). Leave the GEMM API general enough for it, as the design
  specifies.
