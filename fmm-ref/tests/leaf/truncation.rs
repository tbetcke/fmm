//! Truncation bound against the exact kernel, for the standard one-box separation of
//! design §2.5: the near points within √3 r of the centre, the far points at distance
//! at least (4 − √3) r.
//!
//! # Potential bound
//!
//! For a source at distance a and a target at distance d > a from the centre, the
//! tail of the Legendre series Σₙ aⁿ/dⁿ⁺¹ Pₙ(cos γ) of 1/|x − y| beyond degree p is at
//! most Σₙ>ₚ aⁿ/dⁿ⁺¹ = (a/d)ᵖ⁺¹ / (d − a), since |Pₙ| ≤ 1. P2M-M2P and P2L-L2P
//! compute exactly that series truncated at p (see `series`), so
//!
//! |φ − φₚ| ≤ Σⱼ |qⱼ| (aⱼ/dⱼ)ᵖ⁺¹ / (dⱼ − aⱼ).
//!
//! # Gradient bound
//!
//! |∇f| does not change under rotation, so rotate about the centre until the source
//! lies on the +z axis, and let ρ = a/d. The degree-n term, as a function of the
//! target, is then a multiple of the zonal harmonic Iₙ⁰ or Rₙ⁰. For real f,
//! |∇f|² = (∂z f)² + |(∂x + i∂y) f|², and the ladders of CONVENTIONS §3.4 (regular)
//! and of `nd_fmm_math::harmonics::irregular_grad` (irregular) give ∂z and ∂x + i∂y of
//! Iₙ⁰ and Rₙ⁰ as single harmonics of order 0 and 1. The addition theorem at γ = 0,
//! Σₘ (N − |m|)!/(N + |m|)! P_Nᵐ(t)² = 1 over |m| ≤ N (with the Legendre functions of
//! §3.3), gives (P_N¹)² ≤ N(N + 1)/2 · (1 − P_N²). Then
//!
//! - target farther (M2P): the term is aⁿ Iₙ⁰(x)/n!, its gradient has norm
//!   aⁿ/dⁿ⁺² √((n + 1)² Pₙ₊₁² + (Pₙ₊₁¹)²) ≤ (n + 1) aⁿ/dⁿ⁺², so the tail beyond p is
//!   at most Σₙ>ₚ (n + 1) ρⁿ / d² = ρᵖ⁺¹/d² · [(p + 2)/(1 − ρ) + ρ/(1 − ρ)²];
//! - target nearer (L2P): the term is n! Rₙ⁰(x)/dⁿ⁺¹, its gradient has norm
//!   aⁿ⁻¹/dⁿ⁺¹ √(n² Pₙ₋₁² + (Pₙ₋₁¹)²) ≤ n aⁿ⁻¹/dⁿ⁺¹, so the tail beyond p is at most
//!   Σₙ>ₚ n ρⁿ⁻¹ / d² = ρᵖ/d² · [(p + 1)/(1 − ρ) + ρ/(1 − ρ)²].
//!
//! Both sums use Σₖ≥₀ (N + k) ρᵏ = N/(1 − ρ) + ρ/(1 − ρ)². At p = 0 in the second case
//! the bound is 1/(d − a)², the largest possible |∇ 1/|x − y||, as it must be.
//!
//! # Rounding floor
//!
//! A tail far below the rounding error of φₚ is not measurable, so each bound gets a
//! floor: 1e-14 times the potential magnitude Σⱼ |qⱼ| / |x − yⱼ|, as the brief
//! prescribes, and likewise 1e-14 times the gradient magnitude Σⱼ |qⱼ| / |x − yⱼ|²
//! for gradients. The collinear corner cases attain both bounds, so there the error
//! exceeds the bound by its rounding error, which the floor covers.

use nd_fmm_ref::Workspace;

use crate::common::{P_MAX, SQRT3, SplitMix64, Target, Worst, exact, frames, norm, place, sub};
use crate::series::{p2l_l2p, p2m_m2p};

/// Potential rounding floor, relative to Σⱼ |qⱼ| / |x − yⱼ|.
const VALUE_FLOOR: f64 = 1e-14;
/// Gradient rounding floor, relative to Σⱼ |qⱼ| / |x − yⱼ|².
const GRADIENT_FLOOR: f64 = 1e-14;

/// The potential and gradient bounds of the module documentation, summed over
/// sources, for the target `x` in the frame with centre `centre`.
pub fn bounds(
    p: usize,
    centre: [f64; 3],
    sources: &[[f64; 3]],
    charges: &[f64],
    x: [f64; 3],
    which: Target,
) -> (f64, f64) {
    let dx = norm(sub(x, centre));
    let (mut value, mut gradient) = (0.0, 0.0);
    for (&y, &q) in sources.iter().zip(charges) {
        let dy = norm(sub(y, centre));
        let (a, d) = match which {
            Target::Far => (dy, dx),
            Target::Near => (dx, dy),
        };
        let rho = a / d;
        let pf = p as f64;
        value += q.abs() * rho.powi(p as i32 + 1) / (d - a);
        let tail = match which {
            Target::Far => {
                rho.powi(p as i32 + 1) * ((pf + 2.0) / (1.0 - rho) + rho / (1.0 - rho).powi(2))
            }
            Target::Near => {
                rho.powi(p as i32) * ((pf + 1.0) / (1.0 - rho) + rho / (1.0 - rho).powi(2))
            }
        };
        gradient += q.abs() * tail / (d * d);
    }
    (value, gradient)
}

/// One truncated evaluation: degree, frame centre, sources, charges, target and which
/// of the two points is the target.
pub struct Case<'a> {
    pub p: usize,
    pub centre: [f64; 3],
    pub sources: &'a [[f64; 3]],
    pub charges: &'a [f64],
    pub x: [f64; 3],
    pub which: Target,
}

/// How a truncated evaluation compares with its bound: the excess of the error over
/// the bound, relative to the magnitude (the check passes if it is at most the floor),
/// and the fraction of the bound the error attains.
#[derive(Clone, Copy, Debug)]
pub struct Excess {
    /// (|φ − φₚ| − bound) / Σⱼ |qⱼ| / |x − yⱼ|; must be at most `VALUE_FLOOR`.
    pub value: f64,
    /// (|∇φ − ∇φₚ| − bound) / Σⱼ |qⱼ| / |x − yⱼ|²; must be at most `GRADIENT_FLOOR`.
    pub gradient: f64,
    /// |φ − φₚ| / bound where the bound is measurable (see [`attained`]), else 0.
    pub value_attained: f64,
    /// |∇φ − ∇φₚ| / bound where the bound is measurable, else 0.
    pub gradient_attained: f64,
}

impl Excess {
    /// Whether both errors are within their bound plus floor.
    pub fn passes(&self) -> bool {
        self.value <= VALUE_FLOOR && self.gradient <= GRADIENT_FLOOR
    }
}

/// error / bound if the bound exceeds 10⁻¹⁰ times the magnitude, far above rounding,
/// and 0 otherwise: below that the error is rounding, not truncation.
fn attained(error: f64, bound: f64, magnitude: f64) -> f64 {
    if bound > 1e-10 * magnitude {
        error / bound
    } else {
        0.0
    }
}

impl Case<'_> {
    /// Compares the potential and gradient `got` with the exact kernel and the bounds.
    pub fn excess(&self, (value, gradient): (f64, [f64; 3])) -> Excess {
        let Case {
            p,
            centre,
            sources,
            charges,
            x,
            which,
        } = *self;
        let reference = exact(sources, charges, x);
        let (value_bound, gradient_bound) = bounds(p, centre, sources, charges, x, which);
        let value_error = (value - reference.value).abs();
        let gradient_error = norm(sub(gradient, reference.gradient));
        Excess {
            value: (value_error - value_bound) / reference.value_scale,
            gradient: (gradient_error - gradient_bound) / reference.gradient_scale,
            value_attained: attained(value_error, value_bound, reference.value_scale),
            gradient_attained: attained(gradient_error, gradient_bound, reference.gradient_scale),
        }
    }

    /// Asserts that both errors are within bound plus floor and records the excesses
    /// and the attained fractions of the bounds.
    fn check(&self, got: (f64, [f64; 3]), worst: &mut [Worst; 4]) {
        let e = self.excess(got);
        let (p, which) = (self.p, self.which);
        assert!(e.passes(), "{which:?} p = {p}: {e:?}");
        worst[0].update(e.value);
        worst[1].update(e.gradient);
        worst[2].update(e.value_attained);
        worst[3].update(e.gradient_attained);
    }
}

/// Trackers for the excesses and attained fractions of one operator pair.
fn trackers(pair: &'static str) -> [Worst; 4] {
    let name = |what: &str| -> &'static str { format!("{pair} {what}").leak() };
    [
        Worst::new(name("potential (error - bound) / magnitude")),
        Worst::new(name("gradient (error - bound) / magnitude")),
        Worst::new(name("potential error / bound, bound > 1e-10 magnitude")),
        Worst::new(name("gradient error / bound, bound > 1e-10 magnitude")),
    ]
}

/// Near points in the box (so within √3 r), including a corner, and far points at
/// scaled distance in [4 − √3, 8], including the closest point in the direction of the
/// corner, where the bound is attained.
fn geometry(rng: &mut SplitMix64) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let near_limit = 4.0 - SQRT3;
    let mut near = vec![[1.0, 1.0, 1.0], [-1.0, 1.0, -1.0]];
    let mut far = vec![[near_limit / SQRT3; 3], [0.0, 0.0, near_limit]];
    for _ in 0..6 {
        near.push(rng.in_cube());
        far.push(rng.in_shell(near_limit, 8.0));
    }
    (near, far)
}

#[test]
fn p2m_m2p_within_truncation_bound() {
    // Error measure: |φ − φₚ| and |∇φ − ∇φₚ| (Euclidean) against the exact kernel,
    // within the bounds of the module documentation plus the rounding floors. Checked
    // for one source at a time, where the corner-to-nearest-target pair attains the
    // potential bound, and for all sources together.
    let mut rng = SplitMix64::new(0x7a11);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = trackers("P2M-M2P");
    for frame in frames() {
        let (near, far) = geometry(&mut rng);
        let sources: Vec<[f64; 3]> = near.iter().map(|&u| place(&frame, u)).collect();
        let targets: Vec<[f64; 3]> = far.iter().map(|&v| place(&frame, v)).collect();
        let charges: Vec<f64> = sources.iter().map(|_| rng.charge()).collect();
        for p in 0..=P_MAX {
            for &x in &targets {
                for j in 0..sources.len() {
                    let (y, q) = (&sources[j..=j], &charges[j..=j]);
                    let got = p2m_m2p(p, &frame, y, q, x, &mut ws);
                    let case = Case {
                        p,
                        centre: frame.centre,
                        sources: y,
                        charges: q,
                        x,
                        which: Target::Far,
                    };
                    case.check(got, &mut worst);
                }
                let got = p2m_m2p(p, &frame, &sources, &charges, x, &mut ws);
                let case = Case {
                    p,
                    centre: frame.centre,
                    sources: &sources,
                    charges: &charges,
                    x,
                    which: Target::Far,
                };
                case.check(got, &mut worst);
            }
        }
    }
    // The collinear corner cases attain both bounds up to rounding.
    assert!(worst[2].value() > 0.99 && worst[3].value() > 0.99);
}

#[test]
fn p2l_l2p_within_truncation_bound() {
    // Error measure: as for P2M-M2P, with the roles swapped: sources at scaled distance
    // at least 4 − √3, targets in the box.
    let mut rng = SplitMix64::new(0x7a12);
    let mut ws = Workspace::new(P_MAX);
    let mut worst = trackers("P2L-L2P");
    for frame in frames() {
        let (near, far) = geometry(&mut rng);
        let sources: Vec<[f64; 3]> = far.iter().map(|&u| place(&frame, u)).collect();
        let targets: Vec<[f64; 3]> = near.iter().map(|&v| place(&frame, v)).collect();
        let charges: Vec<f64> = sources.iter().map(|_| rng.charge()).collect();
        for p in 0..=P_MAX {
            for &x in &targets {
                for j in 0..sources.len() {
                    let (y, q) = (&sources[j..=j], &charges[j..=j]);
                    let got = p2l_l2p(p, &frame, y, q, x, &mut ws);
                    let case = Case {
                        p,
                        centre: frame.centre,
                        sources: y,
                        charges: q,
                        x,
                        which: Target::Near,
                    };
                    case.check(got, &mut worst);
                }
                let got = p2l_l2p(p, &frame, &sources, &charges, x, &mut ws);
                let case = Case {
                    p,
                    centre: frame.centre,
                    sources: &sources,
                    charges: &charges,
                    x,
                    which: Target::Near,
                };
                case.check(got, &mut worst);
            }
        }
    }
    // The collinear corner cases attain both bounds up to rounding.
    assert!(worst[2].value() > 0.99 && worst[3].value() > 0.99);
}
