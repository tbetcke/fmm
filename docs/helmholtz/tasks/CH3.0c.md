# CH3.0c Complex densities

Started 2026-11-01 on `tbetcke/h-ch3-0c`. Roadmap: "CH3.0c Complex densities" (H3; the
last third of CH3.0, eighth on the priority list). Proposal: §10 CH3.0; §2, §3.3, §4 item
9, §6.3; CONVENTIONS §3.13 ("Source chunks", "Target input and output", "Several charge
vectors", "Output"), §3.15 ("Kernel and factors", "Storage", "Inputs and outputs").

## Goal

Fix the Helmholtz leaf chunks and the sizes of every path, take complex charges in as split
planes and give potentials and gradients out as split planes (§3.15, "Inputs and
outputs"), and form one complex expansion per complex charge vector, as library code in
`nd_fmm_exec::helmholtz` that CH3.1's operator builds on.

## Acceptance (proposal §10, verbatim)

> no leaf wider than c_leaf λ on W1–W4; if `nd-octree` gains the maximum leaf width,
> Laplace's trees and outputs bit for bit unchanged (constraint 1); complex charges in
> and potentials and gradients out as split planes (§3.15 item 9), one complex expansion
> per complex charge vector

CH3.0c's share is the last clause (CH3.0a and CH3.0b did the others). Its *Done when*
(roadmap): complex charges in and split potentials and gradients out (§3.15 "Inputs and
outputs"), the leaf chunks fixed, one complex expansion per complex charge vector.

**Reading.** There is no Helmholtz FMM before CH3.1, so "in" and "out" are the input and
output types, the chunk layouts and the two leaf operators that read the source chunk
(P2M) and write the target output chunk (L2P), with the output pass, tested end to end
against the direct sum through a leaf's own frame. The operators that need a second
box's frame (P2L, M2P, P2P) and the FMM itself are CH3.1's.

## Read first

- CONVENTIONS §3.13 ("Source chunks", "Target input and output", "Several charge vectors",
  "Output") and §3.15 ("Storage", "Inputs and outputs").
- `nd_fmm_ref::helmholtz::{Level, Workspace, leaf}` (CH1.3b): the stored forms, the
  complex path and Yukawa's real path.
- `fmm-exec/src/fmm.rs`: `write_leaf_strengths`, `gather_output`, `OutputMany` and the
  output scales 4π r, 4π r² (read, not changed).
- `fmm-exec/src/geometry.rs`: `leaf_coordinates`, `radius`.

## Changes

One crate, `nd-fmm-exec`; no Laplace code, no convention, no dependency, no
`nd-fmm-plan` change.

- `src/helmholtz.rs` becomes `src/helmholtz/mod.rs` (moved unchanged) with a new
  submodule `src/helmholtz/densities.rs`, re-exported from `helmholtz`:
  - `ChargeKind { Real, Complex }`: the real-charge flag of §3.15.
  - `Densities { kind: WavenumberKind, charges: ChargeKind, vectors: m }`, the path, with
    `columns()` (m real charges, 2m complex), `output_columns()` (m for Yukawa with real
    charges, else 2m), `source_point_size()` 3 + columns, `target_output_point_size(gradients)`
    output_columns or 4 × output_columns, `multipole_size(p)` and `local_size(p)` of
    §3.15's table, `complex_multipoles()`, `complex_locals()`, and the chunk indices.
  - **The leaf chunks** (module docs; §3.15 says CH3.0 fixes them, not versioned): a
    complex vector is two adjacent real columns, Re then Im, of §3.13's several-vector
    layout with `columns` vectors:
    - source chunk (3 + c m) n: u triples, then point j's strengths at j c m + c v + part
      (c = 1 real charges, 2 complex; part 0 Re, 1 Im);
    - target output chunk: the output columns' φ̂ one after the other (column 2v + part, or
      v on Yukawa's real path), then their ĝ triples; so §3.15's output order, Re φ then
      Im φ per vector and Re ∇φ triples then Im ∇φ triples, is the order of the columns;
    - expansions: m per box one after the other, each of `multipole_size(p)/m` or
      `local_size(p)/m` values: one complex expansion (Re plane, Im plane) per complex
      charge vector; Yukawa's complex vector two real expansions (Re q's, Im q's).
    - Consequence, tested: Yukawa with complex charges has, value for value, the chunks,
      expansions and output of Yukawa with the 2m real vectors (Re q₀, Im q₀, …), and
      every Helmholtz chunk is a §3.13 several-vector chunk of `columns` (sources) and
      `output_columns` (outputs) vectors, so CH3.1 can reuse the Laplace charge load and
      output pass with m → columns.
  - `Charges<'a, T>` (one vector: `Real(&[T])` or `Complex { re, im }`) and
    `check_charges(densities, &[Charges], n) -> Result<(), DensityError>` (vector count,
    kind, lengths); `write_leaf_charges(densities, vectors, points, values)` fills a leaf
    chunk's strengths from the caller's per-vector planes.
  - `LeafScratch<T>`: the reference `Workspace` and the real-multipole scratch, sized at
    construction (no allocation in the operators).
  - `p2m(densities, k, r, p, chunk, n, scratch, multipoles)` and
    `l2p(densities, k, r, p, locals, targets, scratch, output, gradients)`: in the leaf's
    own frame (centre 0, radius 1) at the level `Level::new(k̂, 1)` with k̂ = k r formed in
    f64 and rounded to T (the stored form's s and β depend on k a only, so they are the
    level's), through `nd_fmm_ref::helmholtz::leaf` (complex path, or `_yukawa` on
    Yukawa's real path and per column for Yukawa's complex vectors), each point one call,
    accumulating (+=). P2M of real charges for real k: the real plane (the reference
    leaves the imaginary one zero; the sum is formed in scratch, then added).
  - `Output<T>` (`helmholtz::Output`): m, whether complex, `potential` (column-major as
    §3.15: vector v's Re φ at 2v N.., Im φ at (2v + 1) N..; Yukawa real at v N..),
    `gradient` likewise as `[T; 3]`; `potential_of(v)`/`gradient_of(v)` give a `Split`
    (`re`, `im: Option`). `write_leaf_output(densities, chunk, n, r, gradients, targets,
    output)`: the output pass of one leaf, each value `T::from_f64(x.to_f64() / scale)` with
    Laplace's scales 4π r and 4π r² (§3.13 "Output"; the kernel's scaling in leaf units is
    Laplace's: φ̃ = φ̂/r, ∇φ̃ = ĝ/r²).
- `fmm-exec/CLAUDE.md`: the module's paragraph extended.

## Tests that prove it

`fmm-exec/tests/helmholtz/densities.rs` (no MPI), f64 unless noted:

- **Sizes**: `multipole_size`, `local_size`, `source_point_size` and
  `target_output_point_size` against §3.15's table for the three kinds × two charge kinds
  × m = 1, 3 at p = 0, 5; one complex vector costs 2(p + 1)² per expansion, never 4.
- **Chunk indices** equal §3.13's several-vector indices with `columns` vectors (closed
  form in the test).
- **Charges**: `write_leaf_charges` places Re/Im where the indices say; `check_charges`
  errors for a wrong count, kind or length.
- **P2M bit for bit** against `nd_fmm_ref::helmholtz::leaf::p2m` with `Complex` charges
  from the planes (complex vectors; real charges for complex k), its real plane for real
  k with real charges (and the reference's imaginary plane exactly zero), `p2m_yukawa`
  for Yukawa; Yukawa complex = the 2m real vectors' chunk and P2M bit for bit; vector v of
  m = 3 equals the one-vector P2M bit for bit.
- **L2P bit for bit** likewise (`l2p`, `l2p_yukawa`), into the split planes.
- **End to end against the direct sum** (`nd_fmm_ref::helmholtz::p2p::direct_sum` and
  `direct_sum_yukawa` in absolute coordinates, 1/(4π) applied): a leaf on level 3 of a
  domain [−0.3, 1.7]³, 40 points, complex charges, k = 7, 7 + 2i, 7i, and kD up to
  about 100 at the leaf (k̂ near 1 and 10):
  - P2M from the chunk, then the reference M2P at targets beyond 2.5 a in leaf units,
    scaled by `write_leaf_output`'s rule, against the direct sum: potential and gradient,
    relative L2 over the targets ≤ 1e-10 at p = 20;
  - far sources → reference P2L in leaf units → `l2p` into the target chunk →
    `write_leaf_output` → `Output`: against the direct sum at the leaf's points, the same
    bound; the real and imaginary planes at their §3.15 places;
  - Yukawa with real charges gives one plane; real k with real charges a real multipole
    and a complex output.

## Checks

`/verify` (format, Clippy default, examples and workspace all-targets, `cargo test` and
`--workspace`, `cargo doc`; the MPI list at 2 and 4 ranks); `cargo test -p nd-fmm-exec
--test helmholtz`. No manifest changes, so no licence check.

## Machine runs

None: no timing, no Laplace change.

## Constraints and risks

- Constraint 1: no Laplace path changes (no change to `fmm.rs`, `operator.rs`,
  `driver.rs`, the tables, `nd-octree`, `nd-fmm-plan` or `nd-fmm-simd`); the move of
  `helmholtz.rs` to `helmholtz/mod.rs` is Helmholtz-only.
- Constraint 3: `Value` stays real; complex quantities are real columns.
- Constraint 9: the layouts depend on k's kind, the charge kind and m only.
- Risk: CH3.1 may want a different P2M/L2P interface for its level calls; these keep to
  the reference operators per point and can be wrapped.

## Pending runs

None.

## Open questions

1. (For sign-off.) The reading above: CH3.0c's "in" and "out" are the types, chunks, P2M,
   L2P and the leaf output pass; the FMM's in and out are CH3.1's.
2. (For sign-off.) The leaf chunks are fixed in `nd_fmm_exec::helmholtz`'s module docs, not
   in CONVENTIONS (§3.15: "fixed by CH3.0 and are not versioned"); a later CONVENTIONS
   edit may point there.
