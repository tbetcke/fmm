//! The rounding bound of the diagonal form (proposal §3.6, amended at CH0.6).

use nd_fmm_math::Complex;

/// The constants of the bound E of proposal §3.6 for one way of building T and one
/// working precision (spike S6, `spikes/diagonal/SPIKE_REPORT.md` §3, §5 and §7; signed off
/// at CH0.6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundConstants {
    /// C_T, the factor of the Hankel term.
    pub c_t: f64,
    /// C₀, the floor in units of u.
    pub c_0: f64,
    /// u, the unit roundoff of the working precision.
    pub unit_roundoff: f64,
}

impl BoundConstants {
    /// f64, T computed in extended precision and rounded once: C_T = 0.15, C₀ = 200.
    pub const F64: Self = Self {
        c_t: 0.15,
        c_0: 200.0,
        unit_roundoff: f64::EPSILON / 2.0,
    };
    /// f32, T computed in extended precision and rounded once: C_T = 0.18, C₀ = 50.
    pub const F32: Self = Self {
        c_t: 0.18,
        c_0: 50.0,
        unit_roundoff: f32::EPSILON as f64 / 2.0,
    };
    /// f64, T interpolated in ŝ · t̂ from an extended-precision table: C_T = 0.23,
    /// C₀ = 200 (S6 §7; not built here).
    pub const F64_INTERPOLATED: Self = Self {
        c_t: 0.23,
        c_0: 200.0,
        unit_roundoff: f64::EPSILON / 2.0,
    };
}

/// θ of the rule of proposal §3.6: an offset class takes the diagonal form where
/// E ≤ θ ε. S6 fitted C_T so that the measured rounding stays within 2E = E/θ.
pub const THETA: f64 = 0.5;

/// The bound of proposal §3.6 for an offset of length `distance` = |t| on a level of
/// circumradius a and degree p (bandwidth 2P − 2 = 2p):
///
/// ```text
/// E = C_T |k||t| u max_{j ≤ 2p} (2j + 1)|hⱼ(k|t|)| e^{Im(k)(|t| + 2a)} + C₀ u
/// ```
///
/// The Hankel term is formed in log space (S6's `log10_term`: the upward recursion in f64,
/// rescaled by 10⁻²⁰⁰ when it grows), so E is finite, or +∞, where hⱼ leaves the exponent
/// range. It depends on k, the level, p and the class alone (constraint 9).
///
/// ```
/// use nd_fmm_math::Complex;
/// use nd_fmm_ref::helmholtz::diagonal::{BoundConstants, bound};
///
/// // A far class at moderate k a: the floor C₀ u dominates.
/// let k = Complex::new(8.0 / 3f64.sqrt() * 2.0, 0.0);
/// let e = bound(k, 3.0, 3f64.sqrt() / 2.0, 12, BoundConstants::F64);
/// assert!(e > 200.0 * f64::EPSILON / 2.0 && e < 1e-12);
/// ```
///
/// # Panics
///
/// If |t| ≤ 0 or k = 0.
pub fn bound(
    k: Complex<f64>,
    distance: f64,
    circumradius: f64,
    p: usize,
    constants: BoundConstants,
) -> f64 {
    assert!(distance > 0.0, "the offset must have a positive length");
    assert!(k.norm() > 0.0, "the wavenumber must be nonzero");
    let z = k * distance;
    let log10 = log10_hankel_term(z, 2 * p)
        + (k.im * (distance + 2.0 * circumradius)) * core::f64::consts::LOG10_E;
    constants.c_t * constants.unit_roundoff * 10f64.powf(log10)
        + constants.c_0 * constants.unit_roundoff
}

/// log₁₀ of |z| max_{j ≤ l_max} (2j + 1)|hⱼ(z)|.
fn log10_hankel_term(z: Complex<f64>, l_max: usize) -> f64 {
    let i = Complex::new(0.0, 1.0);
    let e = (i * z).exp();
    let mut h0 = -(i * e) / z;
    let mut h1 = -(e * (z + i)) / (z * z);
    let mut offset = 0.0f64; // the h are 10^offset times the values kept
    let mut best = h0.norm().log10();
    if l_max >= 1 {
        best = best.max((3.0 * h1.norm()).log10());
    }
    for l in 1..l_max {
        let h2 = h1 * (2 * l + 1) as f64 / z - h0;
        h0 = h1;
        h1 = h2;
        best = best.max(((2 * l + 3) as f64 * h1.norm()).log10() + offset);
        if h1.norm() > 1e200 {
            h0 *= 1e-200;
            h1 *= 1e-200;
            offset += 200.0;
        }
    }
    best + z.norm().log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The log-space term against the direct maximum where nothing overflows, and finite
    /// beyond the exponent range.
    #[test]
    fn hankel_term_in_log_space() {
        let z = Complex::new(7.5f64, 0.3);
        let l_max = 40;
        let mut h = vec![Complex::new(0.0, 0.0); l_max + 1];
        nd_fmm_math::radial::hankel_h(z, 1.0, &mut h, None);
        // hankel_h scales by e^{Im z}; undo it.
        let damp = (-z.im).exp();
        let direct = (0..=l_max)
            .map(|j| (2 * j + 1) as f64 * h[j].norm() * damp)
            .fold(0.0, f64::max)
            * z.norm();
        let log = log10_hankel_term(z, l_max);
        assert!(
            (log - direct.log10()).abs() < 1e-12,
            "{log} {}",
            direct.log10()
        );
        let far = log10_hankel_term(Complex::new(5.0, 0.0), 600);
        assert!(far.is_finite() && far > 400.0);
    }
}
