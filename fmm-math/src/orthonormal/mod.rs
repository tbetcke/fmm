//! Real orthonormal spherical harmonics and their rotation blocks at any degree, for the
//! Helmholtz family (CONVENTIONS §3.15, "Angular basis" and "Rotations").
//!
//! The Laplace modules ([`crate::harmonics`], [`crate::rotation`]) use the
//! Racah-normalised solid harmonics of §3.3, whose scale factors span about 4ⁿ within a
//! degree and leave the f64 range near n ≈ 515. The Helmholtz family uses real
//! orthonormal harmonics instead, which keep every value and every rotation entry of
//! order one at the degrees of 100 to 1000 that high-frequency levels need. They are
//! the real storage of N Rₙ (N the diagonal of §3.8) times √((2n + 1)/(4π)) per
//! degree.
//!
//! - [`legendre`]: the normalised associated Legendre functions P̄ₙᵐ (and P̄ₙᵐ / sin θ),
//!   overflow-free and accurate near the poles at any degree (spike S1).
//! - [`harmonics`] and [`harmonics_derivatives`]: Yₙᵐ(x̂) in the layout of §3.6, and
//!   ∂θ Yₙᵐ and (1/sin θ) ∂φ Yₙᵐ.
//! - [`rotation`]: the y-rotation blocks by the Gumerov–Duraiswami recursion, the π − β
//!   signs and the full z-y-z blocks.
//!
//! The functions are generic over `T`, but CONVENTIONS §3.15 tests the Helmholtz family
//! in f64 only until decision D3; f32 is checked at low degree only.

mod harmonics;
pub mod legendre;
pub mod rotation;

pub use harmonics::{DerivativeScratch, harmonics, harmonics_derivatives};
