//! A library to generate FMM plans
//!
//! [`interaction_manager`] derives the U-, V-, W- and X-lists of the boxes of
//! an octree, and [`ghost_communicator`] exchanges the data of ghost boxes between
//! ranks. [`fmm`] runs a distributed FMM on top of them. [`v2`] holds the
//! redesigned, index-based plan that replaces them during Phase 3.

pub mod fmm;
pub mod ghost_communicator;
pub mod interaction_manager;
pub mod v2;
