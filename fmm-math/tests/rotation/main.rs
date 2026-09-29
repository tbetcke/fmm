//! Acceptance tests for the rotation blocks (Phase 0 / T5): the action on regular and
//! irregular harmonics, the group properties and orthogonality of CONVENTIONS §3.8, and
//! rotations about z.

// The helpers of the harmonics tests, shared; only `complex_at` is unused here.
#[expect(dead_code, reason = "complex_at is used by tests/harmonics only")]
#[path = "../harmonics/common.rs"]
mod common;

mod action;
mod axis;
mod group;
mod support;
