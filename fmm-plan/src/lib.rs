//! A library to generate FMM plans
//!
//! [`fmm_tree`] plans the tree and the data layout of sources and targets,
//! [`interaction_manager`] derives the U-, V-, W- and X-lists of the resulting
//! boxes, and [`ghost_communicator`] exchanges the data of ghost boxes between
//! ranks. [`fmm`] runs a distributed FMM on top of them.

pub mod fmm;
pub mod fmm_tree;
pub mod ghost_communicator;
pub mod interaction_manager;
