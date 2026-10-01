# Phase 3 / T3 — nd-fmm-exec: scaffold, box geometry and leaf-scaled coordinates (part of C3.1)

This task creates `nd-fmm-exec` and its `geometry` module: the cubic domain, box centres
and radii from Morton keys, the exact relative frames of §3.13, and leaf-scaled
coordinates. It also checks the octant and offset order of `nd-fmm-tables` against
`nd-octree` and the `V_LIST_DIRECTIONS` constant. It uses nothing of `nd-fmm-plan`
beyond that constant, so the plan rewrite (T4–T7) can run at the same time.

The scaffold part can start at once. The `geometry` module starts only after
CONVENTIONS §3.13 (T2) is signed off.

Read first: root CLAUDE.md, fmm-exec/CLAUDE.md (already committed with this brief),
root Cargo.toml, fmm-tables/Cargo.toml and fmm-tables/src/lib.rs (the template to
follow), docs/phase3/README.md ("Design decisions", "Leaf-scaled data", "Domain",
"Dependencies"), docs/design/workspace-structure.md §3 and §5, docs/CONVENTIONS.md §3.7,
§3.12 and §3.13, tools/fixtures/check_leaf_geometry.py, the docs of
`nd_fmm_tables::geometry`, `nd_octree::morton::{children, child_index, decode,
physical_box, from_physical_point}`, `nd_octree::{PhysicalBox,
compute_global_bounding_box}` and `nd_fmm_ref::Frame`.

Starting point: with Phase 2 T8 merged, `members` lists "fmm-plan", "octree",
"fmm-math", "fmm-ref", "fmm-validate", "fmm-tables" and "spikes/cubecl-gemm", and
`default-members` lists the same without the spike. The directory fmm-exec/ contains
only its CLAUDE.md. If fmm-tables is not a member, Phase 2 is not merged: stop and
report.

Do:
- Create the crate in fmm-exec/: package nd-fmm-exec, lib nd_fmm_exec, version
  0.1.0-dev, with edition, licence and repository inherited from [workspace.package].
  Add a one-line README and keep fmm-exec/CLAUDE.md.
- In [workspace.dependencies], add:
  - `mpi = { version = "0.8.2", features = ["derive"] }`;
  - `rlst = "0.8.0"`, without features. Keep the comment that it is never given the
    `mpi` feature there;
  - `nd-octree = { path = "octree" }`, `nd-fmm-plan = { path = "fmm-plan" }` and
    `nd-fmm-exec = { path = "fmm-exec" }`.

  Do not migrate octree/ or fmm-plan/ to these entries; T7 decides that for fmm-plan.
- Dependencies of nd-fmm-exec, each `workspace = true`, added when first used: in this
  task nd-fmm-math, nd-fmm-ref, nd-fmm-tables, nd-octree and thiserror, and nd-fmm-plan
  for `V_LIST_DIRECTIONS` only. Dev-dependencies: proptest. mpi and rlst come with T8
  and T9.
- Add fmm-exec to `members` and `default-members`.
- Crate-level doc comment:
  - the purpose: the Laplace kernel on `nd-fmm-plan`, host path;
  - the conventions the crate relies on (§3.1, §3.7, §3.12, §3.13);
  - the rules of docs/phase3/README.md: tables looked up by integer octant and offset,
    geometry from integer keys, 1/(4π) applied once on output, MPI discipline.
- Module `geometry` (§3.12, §3.13), MPI-free:
  - `Domain`: a validated cubic domain with lower corner a and side w, built with
    `Domain::new(&PhysicalBox) -> Result<Domain, GeometryError>`.
    - Reject non-finite coordinates, a non-positive side and unequal sides. Sides are
      equal when they differ by at most a small multiple of ε (max |corner| + w)
      (docs/phase3/README.md, "Domain"). Fix the multiple from the rounding of
      `compute_global_bounding_box`, and document why the tolerance exists and why it
      is not a fixed relative one. The side is the largest of the three.
    - `GeometryError` uses `thiserror` and names the sides found.
  - `centre(key, &domain) -> [f64; 3]` and `radius(level, &domain) -> f64`, by §3.12.
  - `integer_centre(key, level) -> [i64; 3]`: C_L of §3.13.
  - `relative_frame::<T>(s, t) -> Frame<T>`: the frame (ĉ(s|t), r̂(s|t)) of box s in
    the scaled coordinates of box t, from the keys alone. It is exact in T (§3.13) and
    needs no domain.
  - `leaf_coordinates::<T>(x, leaf, &domain) -> [T; 3]`: u of §3.13, in the f64
    evaluation order it states, then rounded to T.
  - `contains(leaf, x, &domain)`: whether `points_to_morton` would put x into a
    descendant of `leaf`, using `morton::from_physical_point` rather than a second
    formula.
- Doc comments cite §3.12 and §3.13, give the exactness bound of `relative_frame` and
  the error bound of `leaf_coordinates`, and state that every operator takes its
  geometry from here.
- In root CLAUDE.md, change "Current phase and task briefs: docs/phase2/README.md" to
  docs/phase3/README.md. Change nothing else there.

Tests that define done (all MPI-free):
- Table order:
  - `nd_fmm_tables::geometry::m2l_offsets()` equals `V_LIST_DIRECTIONS` in content and
    order, and `m2l_offset_index(d)` is the position of d in `V_LIST_DIRECTIONS`;
  - for parent keys on levels 0, 1, 7 and 15, `morton::children(parent)[o]` has
    `morton::child_index` o, and its index is 2 · index(parent) plus the bits of o,
    with `octant_direction(o)` = 2 · bits − 1.

  T4 keeps `V_LIST_DIRECTIONS` in the new nd-fmm-plan, in the same order, and T7 keeps
  its path; T8 repeats this check against the plan's batches.
- Domain:
  - the output of `compute_global_bounding_box`-style padding (reproduce its formula
    without MPI) for 100 seeded point clouds is accepted, including clouds whose centre
    is 10⁹ times their extent from the origin and clouds with very unequal extents;
  - a box near the origin with sides differing by a relative 1e-9 is rejected with the
    sides named; non-finite and zero-side boxes are rejected.
- Centres: `centre` and `radius` equal the midpoint and half-side of
  `morton::physical_box`, on levels 0–16:
  - exactly in a dyadic domain, a = (−1.25, 0.5, 2) and w = 3 (as in Phase 2);
  - to a documented few ulps of |a| + w in a generic domain, a = (0.1, −2.3, 7.9),
    w = 0.37.
- `integer_centre` reproduces `centre` exactly in the dyadic domain, on every level and
  every reference level L ≥ l.
- Relative frames:
  - in f64 and in f32, `relative_frame(s, t)` equals (c_s − c_t)/r_t and r_s/r_t,
    computed in the dyadic domain in f64 (exact there), bit for bit;
  - for every pair that the U, V, W and X lists can produce (same level at offsets in
    {−3..3}³, parent and child, leaves one level apart), and for arbitrary pairs on
    levels 0–16;
  - `relative_frame(t, t)` is ((0, 0, 0), 1);
  - the composition ĉ(s|t) = ĉ(s|b) r̂(b|t) + ĉ(b|t) holds exactly.
- Leaf-scaled coordinates:
  - for seeded points in leaves on levels 0, 4, 10 and 16 of the generic domain, u
    matches the exact u (a compensated f64 reference, or the Python check's numbers as
    a small fixture) within the §3.13 bound;
  - u lies in [−1, 1]³ up to that bound for every point that `contains` assigns to the
    leaf, including points on box faces;
  - f32: the cast of the f64 u, never a computation in f32.
- The geometry lesson, documented in the test's doc comment and reported: at level 12
  of the generic domain, P2P between neighbouring leaves via `relative_frame` and
  leaf-scaled f32 coordinates, against the same pair via absolute f32 coordinates,
  both against `direct_sum` in f64. Report both errors; assert only that the
  leaf-scaled one is within 1e-6 (relative to Σ|q|/|x − y|).
- proptest for random keys, pairs and points, 64 cases per property.

Must pass, and report which ran:
- the CI commands of root CLAUDE.md, and the stricter
  `cargo clippy --workspace --all-targets -- -D warnings` and
  `RUST_MIN_STACK=8388608 cargo test --workspace` (needs MPI). If the spike fails in
  the sandbox, exclude nd-fmm-spike-cubecl-gemm and say so;
- `RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec` (debug run under a minute);
- `cargo clippy -p nd-fmm-exec --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-exec --no-deps` without warnings;
- `cargo test -p nd-fmm-math -p nd-fmm-ref -p nd-fmm-tables` still builds and runs
  without MPI.

Report the measured worst error of every test, and the largest numerator of
`relative_frame` over the list-producible pairs.

Do not: touch octree/, fmm-plan/, fmm-math/, fmm-ref/, fmm-tables/ or fmm-validate/;
add operators (T8) or point binning into an octree (T9); depend on the domain anywhere
in `relative_frame`; form a centre difference in floating point outside the tests that
measure it; add rayon, CubeCL or nd-fmm-validate. If the tables' order disagrees with
nd-octree or `V_LIST_DIRECTIONS`, or an exactness claim of §3.13 fails, stop and report.
