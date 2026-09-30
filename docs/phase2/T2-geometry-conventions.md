# Phase 2 / T2 — box geometry and operator-table conventions (prerequisite of C2.1–C2.3)

The tables encode box-geometry conventions that `docs/CONVENTIONS.md` does not yet
state: the child octant index, the sign and order of the V-list offsets, and the shift
2r · d (workspace-structure §6, "Still open"; design §9.2). This task writes them as a
new §3.12. The section also gives the table layout and the action of the cube symmetry
group on coefficients, which T5 relies on, and the task checks that action numerically.
It is the Phase 2 counterpart of Phase 1 T2.

Read first:
- docs/CONVENTIONS.md (all; §3.7, §3.8, §3.10 and §3.11 closely);
- docs/phase2/README.md ("Design decisions for this phase");
- the docs and code of `nd_octree::morton::{decode, child_index, children,
  physical_box}` and of `nd_fmm_plan::interaction_manager::{V_LIST_DIRECTIONS,
  InteractionManager::v_list_by_direction}`;
- docs/design/laplace-fmm-plan.md §2.4, §2.5, §3.6, §5.2 (geometry from the key) and
  §5.3;
- tools/fixtures/check_translations.py (`fitted_blocks`, `rotate`, `m2m`, `l2l`,
  `m2l`) and tools/fixtures/README.md.

Do:
- Draft §3.12 "Box geometry and operator tables", citing §3.6–§3.11 for everything it
  builds on. It covers:
  - **Domain and levels.** A cubic domain with lower corner a and side w. The box on
    level l (0 ≤ l ≤ 16) with index (i, j, k) from `morton::decode` has centre
    c = a + ((i, j, k) + ½) w / 2^l and half-width r_l = w / 2^(l+1), which is the
    scaling radius of §3.7. State that this matches `morton::physical_box`.
  - **Child octants.** Child index o = 4x + 2y + z (`morton::child_index`), where x, y
    and z ∈ {0, 1} are the lowest bits of the child's index. The child centre is the
    parent centre plus r_child · s_o, with s_o = (2x − 1, 2y − 1, 2z − 1) and
    r_child = r_parent / 2.
  - **V-list offsets.**
    - d = index(target) − index(source) in the index units of the level, as
      `v_list_by_direction` computes it; d ∈ {−3..3}³ \ {−1..1}³.
    - The 316 offsets are ordered lexicographically in (x, y, z), as
      `V_LIST_DIRECTIONS` is. The position of d in that order is its table index; give
      it in closed form.
    - The shift is c_target − c_source = 2 r_l d, so in §3.11 terms b = 2d and σ = 1.
  - **Canonical frames and level independence.** Give the canonical frames of
    docs/phase2/README.md, and state (from §3.7 and §3.11) that a table built there
    serves every level and every cubic domain.
  - **Matrix layout.** Rows index output slots and columns input slots, stored
    column-major; families are contiguous in octant or offset order; application
    accumulates. As in docs/phase2/README.md.
  - **The cube symmetry group.** O_h consists of the 48 signed permutation matrices P,
    24 of them proper.
    - Give a fixed enumeration order of the 48 elements.
    - State the action on offsets (P d is again an offset) and on octants
      (P s_o = s_o′).
  - **Coefficients under improper P.** Parity (§3.11) gives Rₙ(Px) = (−1)ⁿ Rₙ(−Px),
    and −P is proper. So define Dⁿ(P) = (−1)ⁿ Dⁿ(−P), and state that
    Rₙ(Px) = Dⁿ(P) Rₙ(x) and the homomorphism of §3.8 then hold for all of O_h. State
    the same for irregular harmonics with S Dⁿ S⁻¹.
  - **Operator identities.** Per degree, let T_M(P) = K Dⁿ(P) K and
    T_L(P) = K S Dⁿ(P) S⁻¹ K (§3.11, "Rotation of coefficients"). The expected
    identities, to be derived and then verified numerically (do not assume the
    direction of any factor), are:
    - M2L(P d) = T_L(P) M2L(d) T_M(P)⁻¹;
    - M2M(P s) = T_M(P) M2M(s) T_M(P)⁻¹;
    - L2L(P s) = T_L(P) L2L(s) T_L(P)⁻¹.

    Also give T(P)⁻¹ = T(Pᵀ). Inversion P = −I is the special case
    M2L(−d) = diag((−1)ʲ) M2L(d) diag((−1)ⁿ).
  - **Symmetry classes.**
    - The 316 offsets form 16 orbits under O_h. Each has the representative with
      0 ≤ d_x ≤ d_y ≤ d_z.
    - For each offset d, the group element P with P · representative = d is the first
      such element in the enumeration order above. This makes the class form of T5,
      and so the cached class tables, reproducible.
    - State the counts: 16 classes under O_h, and 34 under the 16 elements that map
      the z-axis to itself or to its negative.
  - **The z-axis elements.** For those 16 elements, check whether T_M(P) and T_L(P) are
    diagonal or signed permutations of the (+m, −m) slot pairs. Phase 4 cares, because
    it could then apply them without arithmetic. State the result in §3.12 only if the
    check confirms it.
- In §3.10, propose extending the rule to §3.12: "Any change to §3.1–§3.8, §3.11 or
  §3.12 bumps it". Cached tables depend on the layout and on the class rule. Update
  the sentence at the top of the file to match.
- tools/fixtures/check_symmetry.py: mpmath at 40 digits, with inline dependencies
  (PEP 723) like the other scripts. Reuse check_translations.py and gen_harmonics.py by
  import, not by copy. With seeded random points and coefficients, at n ≤ 8, it checks:
  - offsets: 316 of them, in lexicographic order, the closed-form index round-trips,
    and none lies in {−1..1}³;
  - classes: 16 orbits (34 under the z-axis subgroup), representatives as stated, and
    P · representative = d for every offset under the stated rule;
  - octants: every P permutes the eight s_o, and the child-centre formula agrees with
    the bits of o;
  - improper rule: Rₙ(Px) = Dⁿ(P) Rₙ(x) for all 48 P, with Dⁿ(P) = (−1)ⁿ Dⁿ(−P) and
    Dⁿ(−P) from `fitted_blocks`; the irregular counterpart with S; to 1e-30;
  - operator identities:
    - M2L for all 316 offsets from their representatives;
    - M2M and L2L for all 8 octants from octant 0 (they form one orbit);
    - inversion;
    - all through the §3.11 forms of check_translations.py, to 1e-30;
  - z-axis elements: the structure of T_M(P) and T_L(P), printed.

  Print the worst error per check and exit nonzero on any failure. Document the script
  in tools/fixtures/README.md.

Must pass:
- `uv run tools/fixtures/check_symmetry.py`;
- `uv run tools/fixtures/check_translations.py` and
  `uv run tools/fixtures/gen_harmonics.py --check`, both unchanged.

The PR description must contain:
- a summary of the derivations and the measured agreement of every check;
- the questions for sign-off:
  1. the wording of §3.12;
  2. extending §3.10 to cover §3.12;
  3. whether adding §3.12 bumps `CONVENTION_VERSION`. Recommend no bump: no existing
     convention changes, and no table has been cached yet. The decision is the
     reviewer's.
  4. the representative and group-element rule of the symmetry classes;
  5. whether the matrix layout belongs in CONVENTIONS or only in the docs of
     nd-fmm-tables. Recommend CONVENTIONS: the cache, `nd-fmm-exec` and the Phase 4
     GEMM kernels all depend on it.

Do not: change §3.1–§3.11; write any Rust; touch any crate. If nd-octree or
nd-fmm-plan contradicts the geometry of design §5.2 (for example the offset sign or the
child bit order), or a symmetry identity fails, stop and report. Do not adjust a formula
to fit.
