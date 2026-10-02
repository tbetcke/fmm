//! The per-ISA P2P entry points (docs/design/simd-p2p.md §5.3).
//!
//! Each module holds the code of one [`Isa`](crate::Isa), compiled only for its
//! architecture. [`P2pKernel::evaluate`](crate::P2pKernel::evaluate) checks the slice
//! lengths and then calls exactly one entry point, so the entry points take checked
//! slices.
//!
//! In this version every entry point runs [`scalar::p2p`]. The per-ISA vector layer
//! and inverse square root (Phase 3S T4) and the vector kernel (T5) replace the bodies
//! of [`neon`] and [`avx2`]; `unsafe` is allowed only in these modules and in the
//! dispatch (crate documentation, "Unsafe").

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod scalar;
