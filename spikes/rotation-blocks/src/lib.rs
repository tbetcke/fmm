//! Orthogonal rotation blocks of large degree in the orthonormal real basis, a spike
//! (Helmholtz CH0.1, S1; docs/helmholtz/tasks/CH0.1.md, proposal §3.2, §5.2, §8 R1).
//!
//! The basis is the real storage (CONVENTIONS §3.6) of N Rₙ, with Rₙᵐ the Racah regular
//! harmonics of §3.3 and N = diag(√((n + |m|)! (n − |m|)!) · cₘ), c₀ = 1, cₘ = √2
//! otherwise (§3.8): slot +m holds √2 νₘ Re Rₙᵐ, slot −m √2 νₘ Im Rₙᵐ, slot 0 ν₀ Rₙ⁰, with
//! νₘ = √((n + |m|)! (n − |m|)!). These are the real orthonormal harmonics without the
//! Condon–Shortley phase, up to a factor per degree, which a rotation block does not see.
//! The block Oⁿ(Q) satisfies Yₙ(Qx) = Oⁿ(Q) Yₙ(x) and equals N Dⁿ(Q) N⁻¹ for the blocks
//! Dⁿ of `nd_fmm_math::rotation::blocks`.
//!
//! - [`legendre`]: P̄ₙᵐ(cos β) = √((n − m)!/(n + m)!) Pₙᵐ(cos β) to any degree, without
//!   underflow of the start values;
//! - [`gd`]: the y-rotation blocks Oⁿ(R_y(β)) by the Gumerov–Duraiswami recursion, in a
//!   compact layout (a cos half and a sin half), and full blocks of z-y-z rotations;
//! - [`conjugated`]: N Dⁿ N⁻¹ from `nd_fmm_math::rotation::blocks`, for comparison.
//!
//! SPIKE_REPORT.md has the method, the measurements and the answers.

pub mod conjugated;
pub mod gd;
pub mod legendre;

/// Index of slot (i, j), −n ≤ i, j ≤ n, of a row-major (2n + 1) × (2n + 1) block whose
/// rows and columns run over the slots m = −n to n, the layout of
/// `nd_fmm_math::rotation::blocks`.
#[inline]
pub fn entry(n: usize, i: isize, j: isize) -> usize {
    let n = n as isize;
    ((i + n) * (2 * n + 1) + (j + n)) as usize
}
