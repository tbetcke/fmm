//! Coaxial translation coefficients of the Helmholtz FMM, a spike (Helmholtz CH0.2, S2;
//! docs/helmholtz/tasks/CH0.2.md, proposal §3.1, §5.2, §5.3, §8 R2).
//!
//! The basis is the real orthonormal Yₙᵐ (CONVENTIONS §3.3's Legendre functions without
//! the Condon–Shortley phase, cos in slot +m and sin in slot −m as §3.6, normalised so
//! that Σₘ Yₙᵐ(x̂) Yₙᵐ(ŷ) = (2n + 1)/(4π) Pₙ(x̂ · ŷ), proposal §3.1), and
//! Fₙᵐ(r) = fₙ(k r) Yₙᵐ(r̂) with f = j (R) or h = h⁽¹⁾ (S). A translation by t = t ẑ is
//! Fₙᵐ(r + t) = Σ_{n′} (E|F)ᵐ_{n′n}(t) G_{n′}ᵐ(r), one coefficient for the slots ±m:
//! (R|R) everywhere, (S|R) for |r| < t, (S|S) = (R|R) for |r| > t.
//!
//! - [`radial`]: jₗ and hₗ of complex argument with the exponential factor removed;
//! - [`coaxial`]: the recursions in m and n, in a triangle closed under them or over
//!   the full rectangle, and the halved table;
//! - [`config`]: the degrees and shifts of the octree's levels;
//! - [`complex`]: a minimal complex type.
//!
//! SPIKE_REPORT.md has the method, the measurements and the answers.

pub mod coaxial;
pub mod complex;
pub mod config;
pub mod metric;
pub mod projection;
pub mod radial;

use complex::C64;

/// Which coefficients: (R|R) = (S|S) from jₗ, or (S|R) from hₗ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// (R|R) = (S|S): M2M and L2L.
    Regular,
    /// (S|R): M2L.
    Singular,
}

/// The start values f̂ₗ, l = 0..=l_max, in the spike's normalisation: jₗ(z) e^{−|Im z|} gˡ
/// for [`Op::Regular`], hₗ(z) e^{Im z} gˡ for [`Op::Singular`].
pub fn start_values(op: Op, z: C64, g: f64, l_max: usize) -> Vec<C64> {
    let mut out = vec![C64::ZERO; l_max + 1];
    match op {
        Op::Regular => {
            radial::bessel_j(z, &mut out);
            let mut s = 1.0;
            for v in out.iter_mut() {
                *v = *v * s;
                s *= g;
            }
        }
        Op::Singular => radial::hankel(z, g, &mut out),
    }
    out
}
