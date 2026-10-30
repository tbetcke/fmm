//! Real orthonormal spherical harmonics and their angular derivatives (CONVENTIONS
//! §3.15, "Angular basis").

use super::legendre::Columns;
use crate::{Layout, RealScalar};

/// The real orthonormal harmonics Yₙᵐ(x̂), n ≤ p, of the direction of `x` (CONVENTIONS
/// §3.15), written to `out` in the layout idx(n, m) = n² + n + m of §3.6: the cos part
/// in slot +m, the sin part in slot −m, no Condon–Shortley phase,
///
/// Yₙ⁰ = √((2n + 1)/(4π)) P̄ₙ⁰(cos θ),  Yₙ^{±m} = √2 √((2n + 1)/(4π)) P̄ₙᵐ(cos θ) {cos mφ, sin mφ},
///
/// with P̄ₙᵐ the normalised Legendre functions of [`super::legendre`]. cos θ, sin θ,
/// cos mφ and sin mφ come from the coordinates by the Cartesian recursion of
/// (x + iy)/ρ, without trigonometric calls (§3.5). On the z-axis, and at x = 0, φ = 0
/// (and θ = 0 at x = 0). Values of degree n do not depend on p; they stay finite and
/// accurate at every degree (S1: to n = 1000).
///
/// ```
/// use nd_fmm_math::Layout;
/// use nd_fmm_math::orthonormal::harmonics;
/// use std::f64::consts::PI;
///
/// let layout = Layout::new(1);
/// let mut y = vec![0.0; layout.len()];
/// harmonics(1, [0.0, 3.0, 4.0], &mut y);
/// // Y₀⁰ = 1/√(4π), Y₁⁰ = √(3/(4π)) cos θ, Y₁⁻¹ = √(3/(4π)) sin θ sin φ
/// assert!((y[layout.idx(0, 0)] - 1.0 / (4.0 * PI).sqrt()).abs() < 1e-15);
/// assert!((y[layout.idx(1, 0)] - (3.0 / (4.0 * PI)).sqrt() * 0.8).abs() < 1e-15);
/// assert!((y[layout.idx(1, -1)] - (3.0 / (4.0 * PI)).sqrt() * 0.6).abs() < 1e-15);
/// assert!(y[layout.idx(1, 1)].abs() < 1e-15);
/// ```
///
/// # Panics
///
/// If `out.len() != (p + 1)²`.
pub fn harmonics<T: RealScalar>(p: usize, x: [T; 3], out: &mut [T]) {
    let layout = Layout::new(p);
    check_len(out, layout, "out");
    let direction = Direction::of(x);
    let mut columns = Columns::new(direction.cos_theta, direction.sin_theta, false);
    let mut phase = (T::one(), T::zero());
    for m in 0..=p {
        if m > 0 {
            phase = direction.advance(phase);
        }
        let factor = order_factor::<T>(m);
        columns.next(p, |n, pbar| {
            let value = normalisation::<T>(n) * factor * pbar;
            store(out, layout, n, m, value, phase);
        });
    }
}

/// Work space of [`harmonics_derivatives`] up to degree p: three columns of the
/// normalised Legendre functions. It allocates once, in [`DerivativeScratch::new`].
pub struct DerivativeScratch<T> {
    p: usize,
    columns: [Vec<T>; 3],
}

impl<T: RealScalar> DerivativeScratch<T> {
    /// Work space for degrees up to `p`.
    pub fn new(p: usize) -> Self {
        Self {
            p,
            columns: core::array::from_fn(|_| vec![T::zero(); p + 1]),
        }
    }
}

/// The harmonics of [`harmonics`] and their angular derivatives ∂Yₙᵐ/∂θ and
/// (1/sin θ) ∂Yₙᵐ/∂φ, n ≤ p, each in the layout of §3.6.
///
/// Both derivatives are regular at the poles: ∂θ comes from the ladder
///
/// ∂θ P̄ₙ⁰ = −√(n(n + 1)) P̄ₙ¹,
/// ∂θ P̄ₙᵐ = ½ (√((n + m)(n − m + 1)) P̄ₙᵐ⁻¹ − √((n − m)(n + m + 1)) P̄ₙᵐ⁺¹)  (m ≥ 1),
///
/// and (1/sin θ) ∂φ from P̄ₙᵐ / sin θ ([`super::legendre::normalised_over_sin`]); neither
/// divides by sin θ. On the z-axis they are the derivatives in the frame φ = 0. With
/// θ̂ and φ̂ the unit vectors of that frame, the gradient of a solid function
/// f(r) Yₙᵐ(x̂) is f′(r) Yₙᵐ r̂ + (f(r)/r)(∂θ Yₙᵐ θ̂ + (1/sin θ) ∂φ Yₙᵐ φ̂).
///
/// ```
/// use nd_fmm_math::Layout;
/// use nd_fmm_math::orthonormal::{DerivativeScratch, harmonics_derivatives};
/// use std::f64::consts::PI;
///
/// let layout = Layout::new(1);
/// let (mut y, mut dt, mut dp) = (vec![0.0; 4], vec![0.0; 4], vec![0.0; 4]);
/// let mut scratch = DerivativeScratch::new(1);
/// // On the north pole: ∂θ Y₁¹ = √(3/(4π)) cos φ = √(3/(4π)) at φ = 0, and
/// // (1/sin θ) ∂φ Y₁⁻¹ = √(3/(4π)) cos φ.
/// harmonics_derivatives(1, [0.0, 0.0, 1.0], &mut y, &mut dt, &mut dp, &mut scratch);
/// let c = (3.0 / (4.0 * PI)).sqrt();
/// assert!((dt[layout.idx(1, 1)] - c).abs() < 1e-15);
/// assert!((dp[layout.idx(1, -1)] - c).abs() < 1e-15);
/// assert!(dt[layout.idx(1, 0)].abs() < 1e-15);
/// ```
///
/// # Panics
///
/// If a slice's length is not (p + 1)², or `scratch` is for a smaller degree.
pub fn harmonics_derivatives<T: RealScalar>(
    p: usize,
    x: [T; 3],
    value: &mut [T],
    d_theta: &mut [T],
    d_phi: &mut [T],
    scratch: &mut DerivativeScratch<T>,
) {
    let layout = Layout::new(p);
    check_len(value, layout, "value");
    check_len(d_theta, layout, "d_theta");
    check_len(d_phi, layout, "d_phi");
    assert!(scratch.p >= p, "scratch is for degree {} < {p}", scratch.p);
    let direction = Direction::of(x);
    let (c, s) = (direction.cos_theta, direction.sin_theta);
    let mut columns = Columns::new(c, s, false);
    let mut over_sin = Columns::new(c, s, true);
    // The columns of orders m − 1, m and m + 1, each indexed by n and zero for n below
    // its order.
    let [mut previous, mut current, mut following] = {
        let [a, b, c] = &mut scratch.columns;
        [&mut a[..=p], &mut b[..=p], &mut c[..=p]]
    };
    let fill = |columns: &mut Columns<T>, column: &mut [T]| {
        column.fill(T::zero());
        columns.next(p, |n, v| column[n] = v);
    };
    fill(&mut columns, current);
    fill(&mut columns, following);
    let mut phase = (T::one(), T::zero());
    for m in 0..=p {
        if m > 0 {
            phase = direction.advance(phase);
        }
        let factor = order_factor::<T>(m);
        let (cos_m, sin_m) = phase;
        let mi = int::<T>(m);
        over_sin.next(p, |n, q| {
            // ∂φ of cos mφ is −m sin mφ, of sin mφ it is m cos mφ.
            let scale = normalisation::<T>(n) * factor * mi * q;
            if m == 0 {
                d_phi[layout.idx(n, 0)] = T::zero();
            } else {
                d_phi[layout.idx(n, m as isize)] = -scale * sin_m;
                d_phi[layout.idx(n, -(m as isize))] = scale * cos_m;
            }
        });
        for n in m..=p {
            let norm = normalisation::<T>(n) * factor;
            store(value, layout, n, m, norm * current[n], phase);
            let derivative = if m == 0 {
                -int::<T>(n * (n + 1)).sqrt() * following[n]
            } else {
                let up = int::<T>((n + m) * (n - m + 1)).sqrt() * previous[n];
                let down = int::<T>((n - m) * (n + m + 1)).sqrt() * following[n];
                (up - down) * T::from_f64(0.5)
            };
            store(d_theta, layout, n, m, norm * derivative, phase);
        }
        // Move the window up by one order.
        core::mem::swap(&mut previous, &mut current);
        core::mem::swap(&mut current, &mut following);
        fill(&mut columns, following);
    }
}

/// cos θ, sin θ and (cos φ, sin φ) of a point, from its coordinates.
struct Direction<T> {
    cos_theta: T,
    sin_theta: T,
    cos_phi: T,
    sin_phi: T,
}

impl<T: RealScalar> Direction<T> {
    fn of([x, y, z]: [T; 3]) -> Self {
        let rho = x.hypot(y);
        let r = rho.hypot(z);
        if r == T::zero() {
            return Self {
                cos_theta: T::one(),
                sin_theta: T::zero(),
                cos_phi: T::one(),
                sin_phi: T::zero(),
            };
        }
        let (cos_phi, sin_phi) = if rho > T::zero() {
            (x / rho, y / rho)
        } else {
            (T::one(), T::zero())
        };
        Self {
            cos_theta: z / r,
            sin_theta: rho / r,
            cos_phi,
            sin_phi,
        }
    }

    /// (cos mφ, sin mφ) from (cos (m − 1)φ, sin (m − 1)φ).
    #[inline]
    fn advance(&self, (c, s): (T, T)) -> (T, T) {
        (
            c * self.cos_phi - s * self.sin_phi,
            s * self.cos_phi + c * self.sin_phi,
        )
    }
}

/// √((2n + 1)/(4π)).
#[inline]
fn normalisation<T: RealScalar>(n: usize) -> T {
    (int::<T>(2 * n + 1) / (T::from_f64(4.0) * T::PI())).sqrt()
}

/// 1 for m = 0, √2 otherwise.
#[inline]
fn order_factor<T: RealScalar>(m: usize) -> T {
    if m == 0 { T::one() } else { T::SQRT_2() }
}

/// Writes `value` · cos mφ to slot +m and `value` · sin mφ to slot −m (slot 0 alone for
/// m = 0).
#[inline]
fn store<T: RealScalar>(
    out: &mut [T],
    layout: Layout,
    n: usize,
    m: usize,
    value: T,
    (c, s): (T, T),
) {
    if m == 0 {
        out[layout.idx(n, 0)] = value;
    } else {
        out[layout.idx(n, m as isize)] = value * c;
        out[layout.idx(n, -(m as isize))] = value * s;
    }
}

fn check_len<T>(slice: &[T], layout: Layout, name: &str) {
    assert_eq!(
        slice.len(),
        layout.len(),
        "`{name}` must have length (p + 1)² = {}",
        layout.len()
    );
}

/// An integer as `T`.
#[inline(always)]
fn int<T: RealScalar>(n: usize) -> T {
    T::from_f64(n as f64)
}
