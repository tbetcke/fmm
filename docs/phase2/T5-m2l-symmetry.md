# Phase 2 / T5 — nd-fmm-tables: cube symmetry and 16-class M2L tables (C2.2)

The dense tables of T4 need 492 MB in f64 at p = 20. The 316 offsets fall into 16
classes under the 48-element cube group O_h, so 16 dense matrices and cheap
per-offset coefficient transforms carry the same operator in about 25 MB at p = 20
(design §3.6). This task builds that form. Checking it against T4 also checks the
§3.12 symmetry rules independently.

Read first: docs/CONVENTIONS.md §3.8, §3.11 ("Rotation of coefficients") and §3.12
(symmetry group, improper elements, operator identities, class rule);
tools/fixtures/check_symmetry.py; the docs of `nd_fmm_math::rotation`; the T4 code.

Do:
- Module `symmetry`:
  - `SignedPermutation`: one element of O_h, as a permutation and three signs. Provide:
    - `all() -> [SignedPermutation; 48]` in the enumeration order of §3.12;
    - `apply(d)` on integer vectors, `matrix() -> [[T; 3]; 3]`, `det()`, `transpose()`
      and `compose()`.
  - `M2L_CLASS_COUNT = 16`, `class_representatives()`, and
    `class_of(index) -> (class, SignedPermutation)` under the rule of §3.12.
  - `CoefficientTransform<T>`: the per-degree blocks of T_M(P) or T_L(P) for degree
    ≤ p.
    - Build them from `rotation::blocks` of P if P is proper, and of −P with the
      factor (−1)ⁿ if it is improper. Use `to_irregular` for T_L, and apply K on both
      sides.
    - `apply(x, out)` adds T x to out and does not allocate.
- `M2lClasses<T>` (in `m2l` or its own module):
  - It holds the 16 representative matrices (a `MatrixSet`, built as in T4) and, for
    each of the 316 offsets, its class and group element.
  - `build(p)`.
  - `expand() -> M2lTables<T>`: the dense tables, T_L(P) A T_M(P)⁻¹ per offset.
  - `apply(index, multipole, local, scratch)`: applies T_M(P)⁻¹, the class matrix and
    T_L(P) in turn, without expanding. It costs O(p⁴) plus two O(p³) transforms and
    uses a caller-owned scratch type.
- Doc comments cite §3.12. They give the storage, 16 (p + 1)⁴ reals plus the
  transforms, and the cost of `apply`.

Tests that define done. f64 unless stated; the "terms" measure of T3, where the terms
of the class form are those of its three products.
- Group:
  - 48 distinct elements, 24 of them proper, closed under composition, with
    transpose as the inverse.
  - The orbits of the 316 offsets are the 16 classes, and their sizes sum to 316.
  - For every offset, `class_of` gives P with P · representative = d, and every
    representative satisfies 0 ≤ d_x ≤ d_y ≤ d_z.
- Transforms:
  - Rₙ(Px) = Dⁿ(P) Rₙ(x) for all 48 P, n ≤ 20, at random x, to relative 1e-13 as in
    Phase 0 T5.
  - The irregular counterpart holds in the orthonormal weighting Nₘ/Sₘ.
  - T(P) T(Pᵀ) is the identity to 1e-14.
  - −I gives diag((−1)ⁿ) exactly.
- The z-axis elements: if §3.12 states that their transforms are diagonal or signed
  permutations, assert it to 1e-15. Otherwise report their structure.
- Expansion, the C2.2 symmetry criterion: `expand()` reproduces all 316 dense T4
  matrices to 1e-13 (terms, per degree). p ≤ 8 in the debug run; p = 16 and p = 20 in
  an `#[ignore]` release test.
- `apply` without expanding equals the dense application to 1e-13 (terms): all offsets
  at p ≤ 8, and the 16 representatives with three group elements each at p = 20.
- Octants: the same transforms map M2M and L2L of octant 0 to those of every octant
  (§3.12), against T3, to 1e-13. This is a test only; the octant tables stay dense.
- f32, p ≤ 8: the cast class form, expanded in f32, matches f64 to 1e-5 (terms).
- Determinism: building twice gives bit-identical classes and transforms.

Must pass:
- `cargo test -p nd-fmm-tables` (debug run under a minute);
- `cargo test -p nd-fmm-tables --release -- --ignored`;
- `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-tables --no-deps` without warnings;
- the root checks.

Report the measured worst errors; the memory of the dense and class forms at p = 8, 16
and 20; the build time of the class form against T4; and the time of `expand()`.

Do not: change the dense tables of T4 or the class rule of §3.12; add a z-axis-only
(34-class) form; add the cache (T7). If an identity of §3.12 fails, stop and report.
