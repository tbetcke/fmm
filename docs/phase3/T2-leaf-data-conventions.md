# Phase 3 / T2 — leaf data and relative box geometry (prerequisite of C3.1)

`LaplaceOperator` (T8) must know how a leaf's points and outputs are laid out in the
per-leaf chunks that the new `nd-fmm-plan` stores and exchanges (source data, target
input, target output: requirement 4 of docs/phase3/README.md), and how one box sees the geometry
of another. The design document (§5.3) interleaved absolute (x, y, z, q). Phase 2
showed that a centre difference formed in floating point carries a relative error of
about ε |c| / r_l (§3.12, "Canonical frames"), which reaches 6.5e4 ε at level 16 and
costs four digits in f32. This task writes the layout and the geometry as a new §3.13,
built so that no operator ever forms a floating-point shift. It is the Phase 3
counterpart of Phase 1 T2 and Phase 2 T2.

Read first:
- docs/CONVENTIONS.md (all; §3.1, §3.6, §3.7, §3.10 and §3.12 closely);
- docs/phase3/README.md ("Design decisions for this phase", "Leaf-scaled data");
- the docs of `nd_fmm_ref::{Frame, leaf, p2p}` (frames, scaling of L2P and M2P,
  coincident pairs);
- the docs and code of `nd_octree::morton::{decode, physical_box}`,
  `nd_octree::octree::compute_global_bounding_box`;
- docs/design/laplace-fmm-plan.md §2.4, §5.2 (geometry from the key) and §5.3;
- tools/fixtures/README.md and check_symmetry.py (script conventions).

Do:
- Draft §3.13 "Leaf data and relative geometry", citing §3.1, §3.6, §3.7 and §3.12 for
  everything it builds on. It covers:
  - **Integer centres.** For a key on level l with index i (a vector, from
    `morton::decode`) and a reference level L ≥ l, the integer vector
    C_L = (2i + 1) · 2^(L − l) gives c = a + C_L · w / 2^(L+1). State that this
    matches §3.12 and `morton::physical_box`.
  - **Relative frames.** For boxes s and t on levels l_s and l_t, with
    L = max(l_s, l_t):
    - the centre of s in the scaled coordinates of t,
      ĉ(s|t) = (c_s − c_t) / r_t = (C_L(s) − C_L(t)) · 2^(l_t − L);
    - the radius ratio r̂(s|t) = r_s / r_t = 2^(l_t − l_s).

    Both are dyadic rationals. State the bound on their numerators for keys on levels
    0–16 and that they are therefore exact in f32 and f64. They do not depend on the
    domain.
  - **Leaf-scaled coordinates.** A point x in leaf b (level l, index i) is stored as
    u = (x − a) · 2^(l+1)/w − (2i + 1), evaluated in f64 in that order, then rounded to
    the storage precision. State the resulting error bound of u: the rounding of
    x − a, of order ε (|x| + |a|) / r_l, which limits every layout because it is the
    precision of the input itself; then the scaling, the subtraction and the cast.
    State that u lies in [−1, 1]³ up to that bound for points that `points_to_morton`
    puts into b, also when the domain's sides agree only to rounding (docs/phase3/README.md,
    "Domain").
  - **Source chunks.** A leaf with n source points holds 4n values: the n coordinate
    triples u₀, …, uₙ₋₁ (point-major, x before y before z), then the n charges, in
    the same point order: 4 values per point. These chunks are exchanged for ghost
    leaves.
  - **Target input and output.** A leaf with n target points has a target-input chunk
    of 3n values, the leaf-scaled positions u₀, …, uₙ₋₁ (point-major), never
    exchanged. Its target-output chunk holds n values (the leaf-scaled potentials φ̂ᵢ)
    or 4n values (then the leaf-scaled gradients ĝᵢ, point-major), in the same point
    order. Define φ̂ = r_t Σ q/|x − y| and ĝ = r_t² ∇ₓ Σ q/|x − y|, the potential and
    gradient of the 1/|x − y| kernel in units of the target leaf.
  - **Operators in scaled coordinates.** For each of the eight operators, the frames
    it passes to nd-fmm-ref, and why the result needs no further factor:
    - P2M at leaf s: Frame((0, 0, 0), 1) on u_s;
    - P2L from leaf s into box t: the frame (ĉ(t|s), r̂(t|s)) on u_s, which gives
      (y − c_t)/r_t;
    - L2P at leaf t: Frame((0, 0, 0), 1) on u_t, which gives φ̂ and ĝ directly;
    - M2P from box s at leaf t: the frame (ĉ(s|t), r̂(s|t)) on u_t, whose 1/r̂ and
      1/r̂² factors make the result φ̂ and ĝ in units of t;
    - P2P from leaf s to leaf t: sources mapped into t's coordinates,
      ŷ = ĉ(s|t) + r̂(s|t) u_s, then 1/|u_t − ŷ| summed; ŷ = u_s for s = t;
    - M2M, L2L and M2L: the tables of §3.12, by child index and offset; no geometry.

    Derive each scaling from §3.7 and the L2P and M2P section of §3.11; do not assume
    it.
  - **Coincident pairs.** A source and a target at the same point lie in the same leaf
    and get the same u, so the exact-coincidence rule of `nd_fmm_ref::p2p` still
    excludes them. State the assumption: sources and targets in one leaf are loaded
    with the same formula from the same f64 coordinates.
  - **Output.** φ(x) = φ̂ / (4π r_t) and ∇φ(x) = ĝ / (4π r_t²), applied once by
    nd-fmm-exec when producing output (§3.1).
- Propose the versioning rule for §3.13 in §3.10, and the matching sentence at the top
  of the file. §3.13 describes in-memory data only; no fixture and no cached table
  depends on it. Recommend that a change to §3.13 does not bump `CONVENTION_VERSION`,
  and that the rule says so explicitly, so the next reader does not have to guess.
  The decision is the reviewer's.
- tools/fixtures/check_leaf_geometry.py, with inline dependencies (PEP 723) like the
  other scripts. Use exact rational arithmetic (`fractions.Fraction`) for the geometry
  and mpmath at 40 digits for the scaling identities. With seeded random keys and
  points, it checks:
  - integer centres: C_L reproduces the midpoint of the box of `physical_box`
    (restated in exact rationals) on every level 0–16, for a dyadic and a generic
    domain;
  - relative frames: ĉ(s|t) and r̂(s|t) equal (c_s − c_t)/r_t and r_s/r_t exactly, for
    random key pairs on all level combinations; every value round-trips exactly
    through IEEE f32 and f64 (`struct.pack`); report the largest numerator;
  - leaf-scaled coordinates: the f64 evaluation order above stays within the stated
    bound of the exact u, at the deepest level, for a domain far from the origin;
  - scaling identities, at 40 digits: P2P in scaled coordinates times 1/r_t and 1/r_t²
    equals the absolute potential and gradient; the M2P and P2L frame maps reproduce
    (x − c_s)/r_s and (y − c_t)/r_t. The expansion operators themselves are checked in
    Rust by T8; this script checks only the geometry and the factors.

  Print the worst error per check and exit nonzero on any failure. Document the script
  in tools/fixtures/README.md.

Must pass:
- `uv run tools/fixtures/check_leaf_geometry.py`;
- `uv run tools/fixtures/check_symmetry.py`, `uv run tools/fixtures/check_translations.py`
  and `uv run tools/fixtures/gen_harmonics.py --check`, all unchanged.

The PR description must contain:
- a summary of the derivations and the measured result of every check;
- the questions for sign-off:
  1. the wording of §3.13;
  2. the versioning rule for §3.13, and whether adding it bumps `CONVENTION_VERSION`.
     Recommend no bump and no extension of the bump rule: nothing persisted depends on
     §3.13;
  3. the source-chunk order (coordinates, then charges) against interleaved (x, y, z, q).
     Recommend the former: the coordinates are then a `[[T; 3]]` view without copying,
     and a later GPU kernel reads them as one contiguous block;
  4. whether target chunks should hold leaf-scaled values (recommended; every operator
     accumulates without a factor) or physical ones.

Do not: change §3.1–§3.12; write any Rust; touch any crate. If nd-octree contradicts
the integer-centre formula, or a scaling identity fails, stop and report. Do not adjust
a formula to fit.
