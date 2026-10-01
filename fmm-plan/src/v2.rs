//! The redesigned plan, built beside the old API (Phase 3, T4–T7).
//!
//! The design is `docs/design/fmm-plan-redesign.md`. Every box a rank holds gets a
//! dense, Morton-ordered `u32` index per level ([`index`]), and the U-, V-, W- and
//! X-lists become index arrays, held once per target and once grouped by V-list offset
//! or child octant ([`lists`]). [`plan::Plan`] builds both from an octree.
//!
//! The modules live under `v2` until the old API ([`interaction_manager`](crate::interaction_manager),
//! [`ghost_communicator`](crate::ghost_communicator), [`fmm`](crate::fmm)) is removed in
//! T7; they then move to the crate root. Until then the old code is the reference the
//! new code is tested against.

pub mod index;
pub mod lists;
pub mod plan;
