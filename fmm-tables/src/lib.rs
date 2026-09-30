//! # `nd-fmm-tables`
//!
//! Precomputed, level-independent operator tables of the nd-project Laplace FMM, and
//! their versioned on-disk cache: M2M and L2L for the 8 child octants, M2L for the 316
//! V-list offsets and their symmetry classes, and the rotation and coaxial tables of
//! point-and-shoot translation. The FMM applies the same tables on every level. The
//! crate is generic over [`RealScalar`]; tables are built in f64 and stored in f64 or
//! f32.
//!
//! ## Conventions
//!
//! All basis functions, phases, storage and scaling are defined in
//! [`docs/CONVENTIONS.md`][conventions], the single source of truth; code cites it as
//! `CONVENTIONS §3.x`. This crate relies on:
//!
//! - Storage (§3.6): coefficients of degree ≤ p occupy (p + 1)² reals at index
//!   n² + n + m, as described by [`nd_fmm_math::Layout`].
//! - Scaling (§3.7): coefficients are stored as M̃ = M / rⁿ and L̃ = L · rⁿ⁺¹ with r
//!   the box half-width, so a translation depends only on the octant or offset, not on
//!   the level.
//! - Translations (§3.11): M2M, L2L and M2L, reached only through
//!   [`nd_fmm_ref::direct`] and [`nd_fmm_ref::rotation`].
//! - Box geometry and operator tables (§3.12): the child index o = 4x + 2y + z and its
//!   sign vector s_o, the order and closed-form index of the 316 V-list offsets, the
//!   canonical frames, and the matrix layout below. [`geometry`] restates them.
//!   [`symmetry`] implements the cube symmetry group, its action on coefficients and
//!   the 16 classes of the offsets.
//!
//! ## Rules for every table
//!
//! - **Built from the oracle, in f64.** Column k of a translation matrix is the
//!   `nd-fmm-ref` operator applied to the k-th unit vector, into a zeroed output. No
//!   operator formula is re-derived here. An f32 table is the f64 table rounded entry
//!   by entry ([`MatrixSet::cast`]), never a table built in f32.
//! - **Canonical frames.** Each table is built at fixed frames that depend only on its
//!   octant or offset: with s_o = (2x − 1, 2y − 1, 2z − 1) for child index
//!   o = 4x + 2y + z,
//!   - M2L from the source frame (0, 1) to the target frame (2d, 1) for offset d;
//!   - M2M from the child (½ s_o, ½) to the parent (0, 1);
//!   - L2L from the parent (0, 1) to the child (½ s_o, ½).
//! - **Column-major layout.** Each operator is a real (p + 1)² × (p + 1)² matrix A with
//!   output = A · input; rows index output slots and columns input slots, both in the
//!   storage of §3.6. Entry (i, k) sits at i + k (p + 1)², and the matrices of one
//!   family are contiguous ([`MatrixSet`]).
//! - **Every application accumulates.** Results are added (+=) into the output, as
//!   every `nd-fmm-ref` operator does; callers zero the output themselves.
//! - **No 1/(4π).** No table contains the factor 1/(4π) of §3.1; it is applied once,
//!   by the caller that evaluates the FMM.
//! - **Serial and deterministic.** Building the same table twice gives bit-identical
//!   results.
//! - **No allocation in applications.** Only builders and constructors allocate.
//!
//! ## Error measures
//!
//! Tests name the measure they use (docs/phase1/README.md and docs/phase2/README.md,
//! "Error measures"):
//!
//! - Coefficient vectors are compared per degree, in the orthonormal basis of §3.8,
//!   relative to the term magnitudes; for a table application the terms are |Aᵢₖ xₖ|.
//!   The result's own degree norm is printed but not asserted.
//! - A table is exact in its geometry, but a direct operator on level l takes its shift
//!   from a centre difference, which carries a relative error of about ε |c| / r.
//!   Comparisons with `nd-fmm-ref` on other levels therefore use a dyadic domain, whose
//!   centres and centre differences are exact, for the tight tolerance. A generic
//!   domain is used only with a documented tolerance that accounts for that rounding.
//!
//! ## Contents
//!
//! - [`MatrixSet`]: a family of square matrices, stored column-major and contiguously,
//!   with an accumulating matrix–vector product.
//! - [`geometry`]: child octants, V-list offsets and the canonical frames (§3.12).
//! - [`octant`]: the M2M and L2L tables of the 8 child octants, [`M2mTables`] and
//!   [`L2lTables`] (C2.1).
//! - [`m2l`]: the dense M2L tables of the 316 V-list offsets, [`M2lTables`], and their
//!   16-class form, [`M2lClasses`] (C2.2).
//! - [`rotation`]: the rotation and coaxial tables of point-and-shoot translation for
//!   the 316 offsets and the 8 octants, [`RotationTables`], and the table-driven
//!   rotation M2L, M2M and L2L, O(p³) per application (C2.3).
//! - [`symmetry`]: the cube group O_h, the symmetry classes of the offsets and the
//!   coefficient transforms T_M(P) and T_L(P) (§3.12).
//!
//! [conventions]: https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md
//! [`RealScalar`]: nd_fmm_math::RealScalar

pub mod geometry;
pub mod m2l;
mod matrix_set;
pub mod octant;
pub mod rotation;
pub mod symmetry;

pub use m2l::{M2lClasses, M2lScratch, M2lTables};
pub use matrix_set::MatrixSet;
pub use octant::{L2lTables, M2mTables};
pub use rotation::{RotationScratch, RotationTables};
