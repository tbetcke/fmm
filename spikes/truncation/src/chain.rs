//! The FMM operators of a chain and the direct sum, in either precision.
//!
//! The kernel is G(x, y) = e^{ik|x−y|}/(4π|x−y|) = ik Σₙ jₙ(k r_<) hₙ(k r_>) Σₘ Yₙᵐ(x̂) Yₙᵐ(ŷ)
//! (proposal §3.1, [`crate::harmonics`]' basis). A level with scale s keeps its multipoles
//! as M̂ₙ = Mₙ/sⁿ and its locals as L̂ₙ = Lₙ sⁿ (s = min(|k| a/8, 1), [`Setup::new`]), so
//!
//! - P2M: M̂ₙᵐ = Σ q ĵₙ(k|y − c|) Yₙᵐ(y − c), ĵ = j/sⁿ;
//! - P2L: L̂ₙᵐ = Σ q ĥₙ(k|y − c|) Yₙᵐ(y − c), ĥ = h sⁿ;
//! - L2P: φ = ik Σ L̂ₙᵐ ĵₙ(k|x − c|) Yₙᵐ(x − c); M2P: φ = ik Σ M̂ₙᵐ ĥₙ(k|x − c|) Yₙᵐ(x − c);
//! - M2L: L̂ = Ê M̂ with the (S|R) table scaled s^{n+n′} on the one level;
//! - M2M child (s_c) → parent (s_p) by t = c_p − c_c: M̂_{p,n′} = Σ E_{n′n} s_cⁿ s_p^{−n′} M̂_{c,n};
//! - L2L parent → child by t = c_c − c_p: L̂_{c,n′} = Σ E_{n′n} s_p^{−n} s_c^{n′} L̂_{p,n}
//!   (the factors per entry, [`ShiftScale`]);
//!
//! each translation as rotation, coaxial step, rotation back (RCR, [`rcr`]).

use crate::coaxial::{self, Op, ShiftScale, Table};
use crate::complex::Complex;
use crate::harmonics::{Harmonics, Polar, idx, len};
use crate::radial;
use crate::real::Real;
use crate::rotation::Rotation;

/// The kind of wavenumber, with k a meaning κ a (S2's set).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// k = κ.
    Real,
    /// k = κ (1 + iη).
    Damped(f64),
    /// k = iκ (Yukawa).
    Imaginary,
}

impl Kind {
    /// The four kinds of the sweep.
    pub const ALL: [Kind; 4] = [
        Kind::Real,
        Kind::Damped(0.1),
        Kind::Damped(1.0),
        Kind::Imaginary,
    ];

    /// A short name.
    pub fn name(self) -> String {
        match self {
            Kind::Real => "real".into(),
            Kind::Damped(eta) => format!("complex{eta}"),
            Kind::Imaginary => "imaginary".into(),
        }
    }

    /// The wavenumber with κ = κa / a.
    pub fn wavenumber<T: Real>(self, kappa: f64) -> Complex<T> {
        match self {
            Kind::Real => Complex::from_f64(kappa, 0.0),
            Kind::Damped(eta) => {
                Complex::new(T::from_f64(kappa), T::from_f64(kappa) * T::from_f64(eta))
            }
            Kind::Imaginary => Complex::from_f64(0.0, kappa),
        }
    }
}

/// √3/2, the circumradius of a unit box.
pub fn unit_radius<T: Real>() -> T {
    T::from_f64(3.0).sqrt() * T::from_f64(0.5)
}

/// The wavenumber and the three levels' scales of a chain: level 0 is the M2L level
/// (width 1), level 1 its children, level 2 its grandchildren.
#[derive(Clone, Debug)]
pub struct Setup<T> {
    /// The kind of k.
    pub kind: Kind,
    /// κ a at level 0.
    pub ka: f64,
    /// k.
    pub k: Complex<T>,
    /// s_l = min(|k| a_l / 8, 1).
    pub s: [T; 3],
    /// Forces the shift tables (M2M, L2L) to S2's projection (`Some(true)`) or recursion
    /// (`Some(false)`); None takes the rule of [`Setup::shift_table`]. For the test that
    /// records the projection's failure for L2L (SPIKE_REPORT.md §2.4).
    pub shift_projection: Option<bool>,
}

impl<T: Real> Setup<T> {
    /// The setup for κ a = `ka` at level 0.
    pub fn new(kind: Kind, ka: f64) -> Self {
        let a = unit_radius::<T>();
        let kappa = (T::from_f64(ka) / a).to_f64();
        let k = kind.wavenumber::<T>(kappa);
        // s = min(|k| a_l / 8, 1): FMM3D's form with the break at |k| a = 8 rather than
        // 1. The truncated chain does not depend on the scale; this one keeps the (S|R)
        // table, whose entries reach h_{2p}(k t) at the nearest distance, inside the
        // exponent range to p ≈ 130 at low frequency (min(|k| a, 1) overflows from
        // p ≈ 90 at k a = 1).
        let s = std::array::from_fn(|l| {
            let v = k.abs() * a * T::from_f64(0.5f64.powi(l as i32) / 8.0);
            if v < T::ONE { v } else { T::ONE }
        });
        Setup {
            kind,
            ka,
            k,
            s,
            shift_projection: None,
        }
    }

    /// The scale factors of an M2M (`m2m`) or L2L between level `child` and its parent,
    /// for degrees to p.
    pub fn shift_scale(&self, child: usize, m2m: bool, p: usize) -> ShiftScale<T> {
        let (sc, sp) = (self.s[child], self.s[child - 1]);
        if m2m {
            ShiftScale::new(true, sc, sp, p)
        } else {
            ShiftScale::new(false, sp, sc, p)
        }
    }

    /// Whether S2's recursion builds the tables of this setup (Re(k) a ≤ 40 in f64, the
    /// signed-off rule; ≤ 100 in double-double, where the recursion stays within 10⁻²⁴
    /// of the projection: 3·10⁻³⁰ at k a = 32, 6·10⁻²² at 128), else the projection.
    pub fn use_recursion(&self) -> bool {
        let re_ka = self.k.re.to_f64() * unit_radius::<f64>();
        re_ka <= if T::EPSILON < 1e-20 { 100.0 } else { 40.0 }
    }

    /// Gauss–Legendre nodes of a projection table of degree P.
    fn nodes(p: usize) -> usize {
        p + if T::EPSILON < 1e-20 { 100 } else { 40 }
    }

    /// The M2L table (S|R) of degree P for |t|² = `d2` (level 0), scaled by s₀.
    pub fn m2l_table(&self, d2: i32, p: usize) -> Table<T> {
        let t = T::from_usize(d2 as usize).sqrt();
        if self.use_recursion() {
            let start = coaxial::start_values(Op::Singular, self.k * t, self.s[0], 2 * p);
            coaxial::recursion(&start, self.s[0], p)
        } else {
            let rho = unit_radius::<T>();
            coaxial::projection(Op::Singular, self.k, t, rho, self.s[0], p, Self::nodes(p))
        }
    }

    /// The (R|R) table of degree P between level `child` and its parent (the shift
    /// a_child, half the parent's circumradius), unscaled.
    pub fn shift_table(&self, child: usize, p: usize) -> Table<T> {
        let w_parent = T::from_f64(0.5f64.powi(child as i32 - 1));
        let t = unit_radius::<T>() * w_parent * T::from_f64(0.5);
        // In double-double the recursion for complex k and to Re(k) a = 128 for real k (the
        // shift's argument is k a/2), the projection above: S2's projection on an outer
        // sphere, accurate in M2M's weights, is not for L2L with damped k
        // (SPIKE_REPORT.md §2.4).
        let recursion = match self.shift_projection {
            Some(projection) => !projection,
            None if T::EPSILON < 1e-20 => {
                self.k.im != T::ZERO || self.k.re.to_f64() * unit_radius::<f64>() <= 128.0
            }
            None => self.use_recursion(),
        };
        if recursion {
            let start = coaxial::start_values(Op::Regular, self.k * t, T::ONE, 2 * p);
            coaxial::recursion(&start, T::ONE, p)
        } else {
            // S2: the outer sphere at the parent's nearest M2L distance, 2 w_parent.
            let rho = T::from_f64(2.0) * w_parent;
            // The square table an L2L reads needs the integrand's degree, about
            // n + n′ + 2|k| t, resolved; M2M reads only n ≤ p_child.
            let kt = (self.k.abs() * t).to_f64().ceil() as usize;
            coaxial::projection(Op::Regular, self.k, t, rho, T::ONE, p, Self::nodes(p) + kt)
        }
    }
}

/// Buffers of one thread.
pub struct Work<T> {
    a: Vec<Complex<T>>,
    b: Vec<Complex<T>>,
    c: Vec<Complex<T>>,
    harm: Harmonics<T>,
    rv: Vec<Complex<T>>,
    rd: Vec<Complex<T>>,
    rw: Vec<Complex<T>>,
}

impl<T: Real> Work<T> {
    /// Buffers for degrees to p.
    pub fn new(p: usize) -> Self {
        Work {
            a: vec![Complex::ZERO; len(p)],
            b: vec![Complex::ZERO; len(p)],
            c: vec![Complex::ZERO; len(p)],
            harm: Harmonics::new(p),
            rv: vec![Complex::ZERO; p + 1],
            rd: vec![Complex::ZERO; p + 1],
            rw: vec![Complex::ZERO; p + 2],
        }
    }

    /// Makes the buffers fit degree p (the harmonics exactly p).
    fn ensure(&mut self, p: usize) {
        if self.harm.degree() != p {
            self.harm = Harmonics::new(p);
        }
        if self.rv.len() < p + 1 {
            self.rv.resize(p + 1, Complex::ZERO);
            self.rd.resize(p + 1, Complex::ZERO);
            self.rw.resize(p + 2, Complex::ZERO);
        }
        if self.a.len() < len(p) {
            self.a.resize(len(p), Complex::ZERO);
            self.b.resize(len(p), Complex::ZERO);
            self.c.resize(len(p), Complex::ZERO);
        }
    }
}

fn sub<T: Real>(x: [T; 3], c: [T; 3]) -> [T; 3] {
    [x[0] - c[0], x[1] - c[1], x[2] - c[2]]
}

/// P2M (`regular`, ĵ) or P2L (ĥ) to degree p about c at a level with scale s.
pub fn p2e<T: Real>(
    regular: bool,
    k: Complex<T>,
    s: T,
    c: [T; 3],
    points: &[[T; 3]],
    q: &[Complex<T>],
    p: usize,
) -> Vec<Complex<T>> {
    let mut out = vec![Complex::ZERO; len(p)];
    let mut harm = Harmonics::new(p);
    let mut rad = vec![Complex::ZERO; p + 1];
    for (y, &qy) in points.iter().zip(q) {
        let polar = Polar::new(sub(*y, c));
        harm.eval(&polar, false);
        let z = k * polar.r;
        if regular {
            radial::bessel_j(z, s, &mut rad);
        } else {
            radial::hankel(z, s, &mut rad);
        }
        for n in 0..=p {
            let f = qy * rad[n];
            for m in -(n as isize)..=n as isize {
                out[idx(n, m)] += f * harm.y[idx(n, m)];
            }
        }
    }
    out
}

/// Potential and gradient at a target.
pub type Field<T> = (Complex<T>, [Complex<T>; 3]);

/// L2P (`regular`, ĵ) or M2P (ĥ) of an expansion of degree p about c at the targets:
/// φ = ik Σ eₙᵐ f̂ₙ(k r) Yₙᵐ and its gradient.
#[allow(clippy::too_many_arguments)]
pub fn e2p<T: Real>(
    regular: bool,
    k: Complex<T>,
    s: T,
    c: [T; 3],
    e: &[Complex<T>],
    p: usize,
    targets: &[[T; 3]],
    work: &mut Work<T>,
    out: &mut Vec<Field<T>>,
) {
    out.clear();
    work.ensure(p);
    let ik = Complex::I * k;
    for x in targets {
        let polar = Polar::new(sub(*x, c));
        let z = k * polar.r;
        radial::with_derivative(
            regular,
            z,
            s,
            &mut work.rv[..=p],
            &mut work.rd[..=p],
            &mut work.rw[..p + 2],
        );
        let (rv, rd, harm) = (&work.rv, &work.rd, &mut work.harm);
        harm.eval(&polar, true);
        let (mut phi, mut ar, mut at, mut ap) =
            (Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO);
        for n in 0..=p {
            let (mut sy, mut st, mut sp) = (Complex::ZERO, Complex::ZERO, Complex::ZERO);
            for m in -(n as isize)..=n as isize {
                let i = idx(n, m);
                sy += e[i] * harm.y[i];
                st += e[i] * harm.dtheta[i];
                sp += e[i] * harm.dphi[i];
            }
            phi += sy * rv[n];
            ar += sy * rd[n];
            at += st * rv[n];
            ap += sp * rv[n];
        }
        // ∇(f(kr) Y) = k f′ Y r̂ + (f/r)(∂_θ Y θ̂ + (1/sin θ) ∂_φ Y φ̂).
        let [rh, th, ph] = polar.frame();
        let ar = ar * k;
        let inv_r = T::ONE / polar.r;
        let (at, ap) = (at * inv_r, ap * inv_r);
        let grad = std::array::from_fn(|d| (ar * rh[d] + at * th[d] + ap * ph[d]) * ik);
        out.push((phi * ik, grad));
    }
}

/// The direct sum of G at the targets.
pub fn direct<T: Real>(
    k: Complex<T>,
    sources: &[[T; 3]],
    q: &[Complex<T>],
    targets: &[[T; 3]],
) -> Vec<Field<T>> {
    let four_pi = T::from_f64(4.0) * T::pi();
    let ik = Complex::I * k;
    targets
        .iter()
        .map(|x| {
            let mut phi = Complex::ZERO;
            let mut grad = [Complex::ZERO; 3];
            for (y, &qy) in sources.iter().zip(q) {
                let d = sub(*x, *y);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let g = (ik * r).exp() / (four_pi * r) * qy;
                phi += g;
                let f = (ik - Complex::real(T::ONE / r)) * g / Complex::real(r);
                for dd in 0..3 {
                    grad[dd] += f * d[dd];
                }
            }
            (phi, grad)
        })
        .collect()
}

/// A translation by rotation, coaxial step and rotation back, from degree p_src to
/// p_tgt, with the scale factors of a shift (None for the M2L); the result in `out`
/// (length at least (p_tgt + 1)²).
#[allow(clippy::too_many_arguments)]
pub fn rcr<T: Real>(
    rot: &Rotation<T>,
    table: &Table<T>,
    p_src: usize,
    p_tgt: usize,
    scale: Option<&ShiftScale<T>>,
    c: &[Complex<T>],
    out: &mut [Complex<T>],
    work: &mut Work<T>,
) {
    let pm = p_src.max(p_tgt);
    if work.a.len() < len(pm) {
        work.a.resize(len(pm), Complex::ZERO);
        work.b.resize(len(pm), Complex::ZERO);
        work.c.resize(len(pm), Complex::ZERO);
    }
    rot.forward(p_src, c, &mut work.a, &mut work.c);
    table.apply(p_src, p_tgt, scale, &work.a, &mut work.b);
    rot.back(p_tgt, &work.b, out, &mut work.c);
}

/// The relative ℓ2 errors (potential, gradient) of `got` against `exact` (§9.1).
pub fn errors<T: Real>(got: &[Field<T>], exact: &[Field<T>]) -> (f64, f64) {
    let (mut np, mut dp, mut ng, mut dg) = (T::ZERO, T::ZERO, T::ZERO, T::ZERO);
    for ((pg, gg), (pe, ge)) in got.iter().zip(exact) {
        np += (*pg - *pe).norm_sqr();
        dp += pe.norm_sqr();
        for d in 0..3 {
            ng += (gg[d] - ge[d]).norm_sqr();
            dg += ge[d].norm_sqr();
        }
    }
    // A non-finite value (overflow at a degree far above the useful range) counts as
    // an infinite error, never as a pass.
    let finite = |e: f64| if e.is_finite() { e } else { f64::INFINITY };
    (
        finite((np / dp).sqrt().to_f64()),
        finite((ng / dg).sqrt().to_f64()),
    )
}
