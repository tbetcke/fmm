//! Acceptance tests of the Laplace operator (Phase 3 / T8, C3.1): the translations,
//! leaf operators and P2P of `LaplaceOperator` against `nd-fmm-ref`, chains against the
//! direct sum, the strategies and their tables, the generic domain, f32, panics,
//! allocation and random properties; the choice of the P2P kernel (Phase 3S / T6,
//! C3S.5); with the `gpu` feature, the device P2P against the reference (Phase 4 / T6,
//! C4.2), the device leaf operators against the reference and the host operator
//! (Phase 4 / T7, C4.3), the device M2M and L2L against the reference and the host
//! operator's error (Phase 4 / T8, C4.4), and the device dense M2L against the reference
//! and the host operator's error (Phase 4 / T9, C4.5).
//!
//! None of these tests initialises MPI (tests/mpi_exec.rs does). Every test names its
//! error measure and prints its worst error with `--nocapture`.

mod allocation;
mod chains;
mod common;
#[cfg(any(feature = "cpu", feature = "metal"))]
mod device_leaf;
#[cfg(any(feature = "cpu", feature = "metal"))]
mod device_m2l;
#[cfg(any(feature = "cpu", feature = "metal"))]
mod device_p2p;
#[cfg(any(feature = "cpu", feature = "metal"))]
mod device_translate;
mod generic;
mod leaf;
mod p2p_choice;
mod panics;
mod precision;
mod properties;
mod strategy;
mod tables;
mod translations;
