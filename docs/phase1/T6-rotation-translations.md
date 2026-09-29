# Phase 1 / T6 — nd-fmm-ref: rotation-based O(p³) M2M, L2L, M2L (C1.3)

Read first: docs/CONVENTIONS.md §3.8 and §3.11 (coaxial forms, rotation of coefficient
vectors); docs/design/laplace-fmm-plan.md §3.2; the docs of `nd_fmm_math::rotation`;
the T5 code and tests.

Do:
- Module `rotation`, with `m2m`, `l2l` and `m2l` taking exactly the arguments of their
  `direct` counterparts.
- Point-and-shoot, in three steps:
  1. Rotate the input coefficients into a frame whose shift lies along +z.
  2. Apply the coaxial translation of §3.11, which keeps m fixed and costs O(p³).
  3. Rotate back.
  - Coefficient vectors rotate as §3.11 states (K Dⁿ K for multipoles,
    K S Dⁿ S⁻¹ K for locals). Build Q with Q t̂ = ẑ from `rotation::euler_zyz`.
  - In raw storage the blocks are not orthogonal (§3.8). Take the inverse rotation
    from `blocks(Qᵀ)`, not by transposing blocks.
- Special cases, handled explicitly and tested:
  - zero shift: no rotation;
  - shift along +z: no rotation;
  - shift along −z: avoid the Euler-angle singularity, e.g. by a rotation by π about y.
- Extend `Workspace` with the block and rotated-coefficient buffers. Building the
  blocks per call is fine: it is O(p³) too. Phase 2 (C2.3) precomputes them.
- Doc comments give the multiply-add count per operator (compare design §3.2: about
  (10/3)(p + 1)³ for M2L).

Tests that define done. f64 unless stated; the weighted per-degree measure of T5.
- Agreement with `direct` for M2M, L2L and M2L, p ≤ 20, to 1e-13:
  - random frames (proptest, 64 cases);
  - the 8 octant children;
  - all 316 same-level offsets at p = 10 and p = 20;
  - the special cases above.
- M2M and L2L for 20 < p ≤ 30, to 1e-11 (the rotation blocks' tolerance at n ≤ 30,
  Phase 0 T5). M2L stays at p ≤ 20, the range of fixture set C.
- Coaxial step alone: for shifts along +z it equals `direct` to 1e-14, since no
  rotation is involved.
- f32, p ≤ 8: matches f64 to 1e-5.

Must pass: `cargo test -p nd-fmm-ref` (debug-mode run under a minute),
`cargo clippy -p nd-fmm-ref --all-targets -- -D warnings`, `cargo doc -p nd-fmm-ref
--no-deps` without warnings, plus the root checks. Report the measured worst errors.
Do not assert the O(p³) cost in tests; T7 measures it.

Do not: precompute rotation or coaxial tables (Phase 2, C2.3); benchmark; optimise
beyond the O(p³) structure.
