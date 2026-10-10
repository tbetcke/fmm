# Phase 6 / T11 — benchmarks, defaults and the design-document update (gate)

Phase 6 ends as Phase 4 T13 and Phase 5 T10 did: the components measured together on the
three machines, the defaults revisited with those numbers, and the design documents
updated. The defaults in question:
- **the M2L strategy rule**: `Auto` (`nd_fmm_exec::tables::M2lStrategy`) picks Dense up to
  p = 8 and Rotation above on the host; Phase 5N made it see the ranks per node
  (docs/design/node-m2l.md). With the batched host M2L (C6.6) and compressed M2L (C6.2)
  the rule changes again;
- **the leaf size per p**: open since Phase 3S T7 ("the best leaf size depends on p ...
  64 at p = 3, 256 at p = 8"), confirmed on Metal (Phase 4 T13) and CUDA (Phase 4S T8),
  "still open for every backend" (laplace-fmm-plan §9.2);
- **the P2P ISA** on AVX-512 machines (C6.7, decision 4);
- **the device strategies and layouts** after C6.1;
- whether compressed M2L (decision 3) or several right-hand sides change any default.

**Revision note.** Written on 2026-10-10; Phase 5S T9 may revise it.

Read first:
- root CLAUDE.md; docs/phase6/README.md ("Requirements" 5 and 6, "Decisions to sign off"
  7, "Exit gate", "Exit checklist");
- docs/design/optimisation.md (signed off) and every Phase 6 PR's results section;
- `fmm-validate/results/phase4-m3max.md` (the format of a results report),
  `fmm-bench/results/phase4s-gh200.md`, the Phase 5, 5N and 5S reports;
- docs/design/laplace-fmm-plan.md §7 (Phase 6 table), §8.3, §9; the leaf-size rule of
  Phase 3S T7 and Phase 4 T13 (laplace-fmm-plan §7, Phase 3S and Phase 4, "Leaf size").

Do:
- **Benchmarks** (release; BLAS variables as each run states; machine, ranks × threads
  and backend named):
  - one node per machine: the M3 Max (host, Metal f32), locust (host, CUDA f32 and f64,
    the load checked and stated), and if Kathleen is usable one Kathleen node (host,
    AVX2 and AVX-512), with
    `nd-fmm-bench` and the `scaling` harness: the C3.2 cube and the C3.3 Plummer sphere,
    N = 10⁶ and 10⁷, f64 p = 3, 6, 8, 12, 18 and f32 p = 3, 6, 8; every M2L strategy
    including the batched and compressed ones; ranks per node and threads per rank as
    Phase 5N's rule recommends and at its extremes;
  - Kathleen across nodes, if usable: the strong and weak scaling runs of Phase 5S T9
    repeated with the new defaults at 1, 2 and 4 nodes — **never more than 4 nodes (160
    cores) in a job** — to show
    the operators' gain carries over;
  - before and after: every figure beside the Phase 5/5N/5S figure it replaces.
- **Defaults** (decision 7): the M2L rule (strategy by p, precision, machine class, ranks
  per node, threads), the leaf size per p by the T7 rule of Phase 3S applied on each
  backend (or 64 kept, with the reason), the P2P ISA rule; each proposed with the
  measurements, implemented after sign-off in the same PR (or a follow-up, as the
  sign-off says), with the one-rank output changes recorded.
- **The report**: `fmm-validate/results/phase6.md` (one per machine if that reads better,
  in the format of phase4-m3max.md), every figure labelled measured (machine, ranks ×
  threads, build) or model.
- **Design documents**:
  - laplace-fmm-plan.md: the revision note; §4 (the strategy comparison with C6.2, C6.5,
    C6.6); §6 (CubeCL facts that changed); §7 (Phase 6 status per component, the
    benchmark tables, a "Recommendation" for what follows); §8.3; §9.1 and §9.2;
  - simd-p2p.md §4.7 (AVX-512 as built and measured);
  - device-path.md: an outcome section for the Phase 6 kernel work;
  - workspace-structure.md §3, §3.1, §6;
  - docs/phase6/README.md: tick the exit checklist.

Tests that define done:
- Any default changed: the tests that pin the defaults updated, with every
  `tests/mpi_exec.rs` scenario, the C5.1 gates and the accuracy tests passing at the new
  defaults on 1, 2, 4 and 8 ranks (M3 Max, locust) and 2 Kathleen nodes (if usable;
  otherwise up to 72 ranks on locust).

Without Kathleen (Phase 5N decision 0): the benchmarks and defaults are from the M3 Max
and locust only; the P2P ISA rule for AVX-512 stays open with C6.7; the report says so.

Must pass: `cargo fmt --all`, the root checks, the stricter workspace checks, the
`nd-fmm-kernels` and `nd-fmm-simd` checks of the root CLAUDE.md, `run-mpi-tests.sh` at 2
and 4 ranks; Metal and CUDA by hand.

Do not: assert timings or commit raw output (the report only); change a default without
its sign-off; present a model as a measurement, or a one-node figure as multi-node
scaling.
