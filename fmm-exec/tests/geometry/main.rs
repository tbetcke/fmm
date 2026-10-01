//! Acceptance tests of the `geometry` module and the table order (Phase 3 / T3, part of
//! C3.1): the octant and offset order of `nd-fmm-tables` against `nd-octree` and
//! `V_LIST_DIRECTIONS`, the cubic domain, box centres and radii (CONVENTIONS §3.12),
//! integer centres, exact relative frames and leaf-scaled coordinates (§3.13), the
//! level-12 P2P lesson and random properties.
//!
//! None of these tests initialises MPI. Every test names its error measure and prints
//! its worst error with `--nocapture`.

mod centres;
mod common;
mod domain;
mod frames;
mod leaf;
mod lesson;
mod properties;
mod table_order;
