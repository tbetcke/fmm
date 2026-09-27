//! A library to generate FMM plans
//!
//! [`fmm_tree`] plans the tree and the data layout of sources and targets, and
//! [`interaction_manager`] derives the U-, V-, W- and X-lists of the resulting
//! boxes.

pub mod fmm_tree;
pub mod interaction_manager;
