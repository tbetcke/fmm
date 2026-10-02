# Phase 3S / T1 — coincident pairs and the domain of fast P2P kernels (C3S.1)

`nd_fmm_ref::p2p` skips a pair when x == y in all three components (§3.13, "Coincident
pairs"). The SIMD kernels skip it when r² = 0 instead, one compare per pair rather than
three (docs/design/simd-p2p.md §4.4). The two rules differ only when x ≠ y and every
dₖ² underflows, and the inverse-square-root estimates misbehave on subnormal input. This
task shows that neither happens on the leaf-scaled data of §3.13, states the domain of r²
that fast kernels support, and writes both into §3.13 for sign-off. It is the Phase 3S
counterpart of Phase 3 T2, and it writes no Rust.

Read first:
- docs/CONVENTIONS.md §3.1, §3.10 and §3.13 (all of it; "Leaf-scaled coordinates",
  "Relative frames", "Operators in scaled coordinates" and "Coincident pairs" closely);
- docs/design/simd-p2p.md §3 (requirement 3), §4.3 and §4.4;
- docs/phase3s/README.md ("Design decisions");
- the docs and code of `nd_fmm_ref::p2p` (the exclusion rule and its tests);
- `nd_fmm_exec::geometry::{leaf_coordinates, relative_frame}` and
  `LaplaceOperator`'s P2P mapping (`Kernels::p2p` in fmm-exec/src/operator.rs);
- the 2:1 balance of `nd-octree` (which neighbour levels a U list can hold);
- tools/fixtures/README.md and check_leaf_geometry.py (script conventions).

Do:
- Derive, for T = f32 and f64:
  - the smallest nonzero |dₖ| between a stored target coordinate u_t and a source
    coordinate that P2P uses: u_s itself for s = t, and the mapped ŷ = fl(ĉ(s|t) +
    r̂(s|t) u_s) for s ≠ t, with ĉ and r̂ as in §3.13 "Relative frames" and the neighbour
    levels that 2:1 balance allows;
  - hence the smallest nonzero r² = fl(fl(d₀²) + …), with the order of operations that
    a kernel uses (fma allowed) stated;
  - the largest r² in the FMM (|u| ≤ 1 + β, the largest |ŷ| a U-list neighbour has);
  - the consequence: on leaf-scaled data r² = 0 if and only if the stored points
    coincide, and every nonzero r² is a normal number in T with a stated margin.

  If a construction breaks the claim (for example a cancellation in ŷ that leaves a
  difference below the bound), report it and stop. Do not adjust the bound to fit
  without saying so.
- Draft the addition to §3.13 "Coincident pairs" (a new paragraph or subsection,
  "Fast kernels"):
  - fast kernels exclude a pair by r² = 0, after computing r² in T;
  - on leaf-scaled data this equals the exact-coincidence rule, with the derivation
    above (or a short form of it and a pointer to the script);
  - the **kernel domain**: r² = 0, or r² in [r²_min, r²_max] with the bounds you
    derived, rounded outwards to powers of two. design §4.4 provisionally uses
    [2⁻¹²⁰, 2¹²⁰]. Pairs outside it are outside the fast kernels' contract;
  - that §3.10 still holds: §3.13 is in-memory only, and `CONVENTION_VERSION` stays 1.
- `tools/fixtures/check_p2p_domain.py`, in the style of check_leaf_geometry.py
  (exact rationals for the claims, IEEE doubles and `numpy.float32` emulation operation
  by operation for the stored values), checking:
  - the derived lower bound against adversarial constructions: points at and next to
    leaf centres and faces, u values of the smallest magnitudes the formula can store,
    mapped neighbours on coarser and finer levels with every ĉ component that 2:1
    balance allows, including the cancellations ĉ + r̂ u ≈ 0;
  - seeded random leaf-scaled pairs on levels 0–16, for a dyadic domain, a generic
    domain and one far from the origin;
  - that r² = 0 occurs exactly for coincident stored points in all of these;
  - the upper bound.

  Use only the script dependencies tools/fixtures/ already pins. It generates no
  fixture files.
- Record in the PR, for sign-off:
  - the §3.13 addition, with the bounds;
  - whether the design document's provisional domain needs changing;
  - that `docs/design/simd-p2p.md` was reviewed alongside, and any change to it this
    derivation needs.

Tests that define done:
- `check_p2p_domain.py` passes and prints the measured extremes: the smallest nonzero
  r² found, the margin to the derived bound, and the largest r².
- The derivation and the script agree, for f32 and f64.

Must pass: the script; `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-math` (the
`CONVENTION_VERSION` check, unchanged); the root checks.

Do not:
- change any convention other than the §3.13 addition, or bump `CONVENTION_VERSION`;
- touch Rust code;
- weaken the reference's rule in `nd_fmm_ref::p2p`. The reference keeps exact
  coincidence; only fast kernels use r² = 0.
