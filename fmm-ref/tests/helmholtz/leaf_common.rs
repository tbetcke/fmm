//! Helpers shared by the leaf-operator tests (Helmholtz CH1.3b): the box, the kinds of k,
//! point sets on both sides of the box's sphere, and the two series P2M → M2P and
//! P2L → L2P.

use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};

use crate::common::{C, SplitMix64};

/// The test box: centre and half-width r (a power of two, as in an octree).
pub const CENTRE: [f64; 3] = [0.3, -0.2, 0.5];
pub const HALF: f64 = 0.5;

/// The box's circumradius a = √3 r.
pub fn circumradius() -> f64 {
    3.0_f64.sqrt() * HALF
}

pub fn frame() -> Frame<f64> {
    Frame::new(CENTRE, HALF)
}

/// The kinds of k of CONVENTIONS §3.15, plus a weakly damped complex k.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Real,
    /// arg k = π/4.
    Complex,
    /// Im k = Re k / 10.
    Weak,
    /// k = iκ.
    Imaginary,
}

pub const KINDS: [Kind; 4] = [Kind::Real, Kind::Complex, Kind::Weak, Kind::Imaginary];

/// The wavenumber of the kind with |k| = `modulus`.
pub fn wavenumber(kind: Kind, modulus: f64) -> C {
    match kind {
        Kind::Real => C::new(modulus, 0.0),
        Kind::Complex => C::from_polar(modulus, std::f64::consts::FRAC_PI_4),
        Kind::Weak => C::new(1.0, 0.1) * (modulus / 1.01_f64.sqrt()),
        Kind::Imaginary => C::new(0.0, modulus),
    }
}

/// The level of the test box for k.
pub fn level(k: C) -> Level<f64> {
    Level::new(k, HALF)
}

/// `n` points uniform in the box, then its 8 corners (ρ = a) and, with `centre`, the
/// centre itself.
pub fn inside(rng: &mut SplitMix64, n: usize, centre: bool) -> Vec<[f64; 3]> {
    let mut points = rng.points(n, CENTRE, HALF);
    for corner in 0..8 {
        let sign = |bit: usize| if corner >> bit & 1 == 1 { 1.0 } else { -1.0 };
        points.push(core::array::from_fn(|i| CENTRE[i] + sign(i) * HALF));
    }
    if centre {
        points.push(CENTRE);
    }
    points
}

/// `n` points at distance `radius` from the centre in random directions.
pub fn shell(rng: &mut SplitMix64, n: usize, radius: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            loop {
                let u: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
                let norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                if norm > 0.1 && norm <= 1.0 {
                    break core::array::from_fn(|i| CENTRE[i] + radius * u[i] / norm);
                }
            }
        })
        .collect()
}

/// The potentials and gradients of an expansion at the targets.
pub struct Field {
    pub potential: Vec<C>,
    pub gradient: Vec<[C; 3]>,
}

impl Field {
    pub fn zero(n: usize) -> Self {
        Self {
            potential: vec![C::new(0.0, 0.0); n],
            gradient: vec![[C::new(0.0, 0.0); 3]; n],
        }
    }
}

/// P2M → M2P at degree p.
pub fn multipole_series(
    p: usize,
    k: C,
    sources: &[[f64; 3]],
    charges: &[C],
    targets: &[[f64; 3]],
) -> Field {
    let (level, frame, mut ws) = (level(k), frame(), Workspace::new(p));
    let mut multipole = vec![0.0; 2 * (p + 1) * (p + 1)];
    leaf::p2m(p, &level, &frame, sources, charges, &mut ws, &mut multipole);
    let mut field = Field::zero(targets.len());
    leaf::m2p(
        p,
        &level,
        &frame,
        &multipole,
        targets,
        &mut ws,
        &mut field.potential,
        Some(&mut field.gradient),
    );
    field
}

/// P2L → L2P at degree p.
pub fn local_series(
    p: usize,
    k: C,
    sources: &[[f64; 3]],
    charges: &[C],
    targets: &[[f64; 3]],
) -> Field {
    let (level, frame, mut ws) = (level(k), frame(), Workspace::new(p));
    let mut local = vec![0.0; 2 * (p + 1) * (p + 1)];
    leaf::p2l(p, &level, &frame, sources, charges, &mut ws, &mut local);
    let mut field = Field::zero(targets.len());
    leaf::l2p(
        p,
        &level,
        &frame,
        &local,
        targets,
        &mut ws,
        &mut field.potential,
        Some(&mut field.gradient),
    );
    field
}
