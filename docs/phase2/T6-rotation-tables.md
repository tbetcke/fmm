# Phase 2 / T6 — nd-fmm-tables: rotation and coaxial tables (C2.3)

`nd_fmm_ref::rotation` builds its rotation blocks on every call. On a uniform level the
shifts are fixed: the 316 V-list offsets and the 8 octant directions. So every
rotation block and every coaxial factor can be precomputed. This task builds those
tables and table-driven rotation M2M, L2L and M2L. They are the CPU O(p³) path of
Phase 3, and the data that the Phase 4 rotation kernel (C4.6) uploads.

Read first: docs/CONVENTIONS.md §3.8, §3.11 ("Coaxial translations", "Rotation of
coefficients") and §3.12; design §3.2; the module docs and code of
`nd_fmm_ref::rotation` (the angles, the rotation rule, the on-axis cases); the docs of
`nd_fmm_math::rotation`; the T3 code.

Do:
- Module `rotation`, with `RotationTables<T>`. For every M2L offset and every octant it
  holds what point-and-shoot needs:
  - the blocks of the forward rotation Q and of the inverse Qᵀ, with the rule of their
    kind (§3.11: K Dⁿ K for multipoles, K S Dⁿ S⁻¹ K for locals);
  - the coaxial factors for the offset's distance (§3.11, "Coaxial translations").

  Use the angles of `nd_fmm_ref::rotation`: Q = R_y(−θ) R_z(−φ), and no rotation for
  shifts on the z-axis.
- Choose the storage and record the choice in the module docs:
  - Suggested: factor Q as nd-fmm-ref does. Store y-rotation blocks once per distinct
    polar angle and z-rotations as cos mφ and sin mφ once per distinct azimuth (O(p)
    data, applied in O(p²)), and give each offset indices into them. Over the 316
    offsets there are 15 distinct distances, 32 distinct azimuths off the z-axis and
    49 distinct polar angles including the axis; the tests confirm these counts.
  - Alternative: full blocks per offset, about 62 MB in f64 at p = 20, against about
    10 MB factorised.
  - If you factor, verify the z-rotation action in real storage for coefficient
    vectors. Phase 0 T5 checked it only for basis vectors ("each (+m, −m) pair rotates
    by mα").
- Build in f64 from `nd_fmm_math::rotation` (`blocks`, `to_irregular`, `euler_zyz`),
  then cast to T.
- Operators `m2l(index, multipole, local, scratch)`, `m2m(o, input, output, scratch)`
  and `l2l(o, input, output, scratch)`:
  - rotate, apply the coaxial step, rotate back;
  - accumulate, allocate nothing, and take a caller-owned scratch type;
  - build no block per call.
- Doc comments give the multiply-add count per operator. It now excludes block
  construction entirely; compare it with design §3.2 and the `nd_fmm_ref::rotation`
  docs. They also give the storage formula.

Tests that define done. f64 unless stated; the "terms" measure of T3.
- Against `nd_fmm_ref::rotation` at the canonical frames: M2L for all 316 offsets and
  M2M and L2L for all 8 octants, to 1e-14 (terms). p ≤ 8 in the debug run; p = 12 and
  p = 20 in an `#[ignore]` release test.
- Against `direct`, the C2.3 criterion as the Phase 1 gate measured it:
  - to 1e-13 (terms), for p ≤ 20;
  - through the dense tables of T3 and T4 for p ≤ 8, and in the ignored test up to
    p = 20.
- M2M and L2L for 20 < p ≤ 30, to 1e-11, as in Phase 1 T6. M2L stays at p ≤ 20.
- Levels: on levels 2, 9 and 16 in the dyadic domain of T3, the table operators equal
  `nd_fmm_ref::rotation` at the actual frames, to 1e-14 (terms).
- On-axis offsets (0, 0, ±2) and (0, 0, ±3) use no rotation and equal the coaxial form,
  as in nd-fmm-ref.
- The distinct-angle and distinct-distance counts are as the module docs state.
- f32, p ≤ 8: the cast tables applied in f32 match f64 to 1e-5 (terms).
- Determinism: building twice gives bit-identical tables.
- proptest for random offsets, octants and coefficients, 64 cases per property.

Must pass:
- `cargo test -p nd-fmm-tables` (debug run under a minute);
- `cargo test -p nd-fmm-tables --release -- --ignored`;
- `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-tables --no-deps` without warnings;
- the root checks.

Report the measured worst errors, the memory and release build time at p = 8, 16 and
20, and the chosen storage. Do not assert timings; T8 measures them.

Do not: change `nd_fmm_ref::rotation`; add tables for non-uniform shifts; add the cache
(T7); optimise beyond the O(p³) structure and the precomputation.
