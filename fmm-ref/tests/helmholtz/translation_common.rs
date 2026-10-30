//! Helpers shared by the translation tests (Helmholtz CH1.3c): the boxes (a parent of
//! half-width r = 0.5 at `CENTRE`, its eight children of r/2, and the boxes of one level at
//! the offsets of the M2L), the two methods, the inputs formed by the leaf operators from
//! point charges, and the evaluation of an output's field.

use nd_fmm_math::radial::{bessel_j, hankel_h};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, direct, leaf, rcr};

use crate::common::{C, SplitMix64};
use crate::leaf_common::{CENTRE, Field, HALF};

/// The half-width of the children.
pub const CHILD: f64 = HALF / 2.0;

/// RCR or the projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Rcr,
    Direct,
}

/// A translation and its geometry: M2M from child `c` to the parent, L2L from the parent
/// to child `c` (c's bits give the signs of the child's offset), M2L from the box at
/// `CENTRE` to the box at `CENTRE` + offset · 2r of the same level (half-width r).
#[derive(Clone, Copy, Debug)]
pub enum Translation {
    M2m(usize),
    L2l(usize),
    M2l([i32; 3]),
}

pub fn parent() -> Frame<f64> {
    Frame::new(CENTRE, HALF)
}

pub fn child(c: usize) -> Frame<f64> {
    let sign = |bit: usize| if c >> bit & 1 == 1 { 1.0 } else { -1.0 };
    Frame::new(core::array::from_fn(|i| CENTRE[i] + sign(i) * CHILD), CHILD)
}

pub fn neighbour(offset: [i32; 3]) -> Frame<f64> {
    Frame::new(
        core::array::from_fn(|i| CENTRE[i] + offset[i] as f64 * 2.0 * HALF),
        HALF,
    )
}

/// One expansion's degree, level and frame.
#[derive(Clone, Copy, Debug)]
pub struct Side {
    pub p: usize,
    pub level: Level<f64>,
    pub frame: Frame<f64>,
}

impl Translation {
    /// The input and output frames.
    pub fn frames(self) -> (Frame<f64>, Frame<f64>) {
        match self {
            Self::M2m(c) => (child(c), parent()),
            Self::L2l(c) => (parent(), child(c)),
            Self::M2l(offset) => (parent(), neighbour(offset)),
        }
    }

    pub fn input_is_multipole(self) -> bool {
        !matches!(self, Self::L2l(_))
    }

    pub fn output_is_multipole(self) -> bool {
        matches!(self, Self::M2m(_))
    }

    /// The input and output sides for k and the degrees.
    pub fn sides(self, k: C, p_in: usize, p_out: usize) -> (Side, Side) {
        let (from, to) = self.frames();
        let side = |p, frame: Frame<f64>| Side {
            p,
            level: Level::new(k, frame.radius),
            frame,
        };
        (side(p_in, from), side(p_out, to))
    }

    /// The complex translation of `input` by `method`, a fresh output.
    pub fn apply(self, method: Method, k: C, p_in: usize, p_out: usize, input: &[f64]) -> Vec<f64> {
        let (i, o) = self.sides(k, p_in, p_out);
        let mut ws = Workspace::new(p_in.max(p_out));
        let mut out = vec![0.0; 2 * (p_out + 1) * (p_out + 1)];
        let (a, b) = (&i, &o);
        match (self, method) {
            (Self::M2m(_), Method::Rcr) => {
                let table = rcr::m2m_table(a.p, b.p, &a.level);
                rcr::m2m(
                    a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &table, &mut ws, input,
                    &mut out,
                );
            }
            (Self::M2m(_), Method::Direct) => direct::m2m(
                a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &mut ws, input, &mut out,
            ),
            (Self::L2l(_), Method::Rcr) => {
                let table = rcr::l2l_table(a.p, b.p, &b.level);
                rcr::l2l(
                    a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &table, &mut ws, input,
                    &mut out,
                );
            }
            (Self::L2l(_), Method::Direct) => direct::l2l(
                a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &mut ws, input, &mut out,
            ),
            (Self::M2l(_), Method::Rcr) => {
                let table = rcr::m2l_table(a.p, b.p, &a.level, &a.frame, &b.frame);
                rcr::m2l(
                    a.p, b.p, &a.level, &a.frame, &b.frame, &table, &mut ws, input, &mut out,
                );
            }
            (Self::M2l(_), Method::Direct) => direct::m2l(
                a.p, b.p, &a.level, &a.frame, &b.frame, &mut ws, input, &mut out,
            ),
        }
        out
    }

    /// The Yukawa translation of the real `input` by `method`, k = iκ.
    pub fn apply_yukawa(
        self,
        method: Method,
        k: C,
        p_in: usize,
        p_out: usize,
        input: &[f64],
    ) -> Vec<f64> {
        let (i, o) = self.sides(k, p_in, p_out);
        let mut ws = Workspace::new(p_in.max(p_out));
        let mut out = vec![0.0; (p_out + 1) * (p_out + 1)];
        let (a, b) = (&i, &o);
        match (self, method) {
            (Self::M2m(_), Method::Rcr) => {
                let table = rcr::m2m_table(a.p, b.p, &a.level);
                rcr::m2m_yukawa(
                    a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &table, &mut ws, input,
                    &mut out,
                );
            }
            (Self::M2m(_), Method::Direct) => direct::m2m_yukawa(
                a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &mut ws, input, &mut out,
            ),
            (Self::L2l(_), Method::Rcr) => {
                let table = rcr::l2l_table(a.p, b.p, &b.level);
                rcr::l2l_yukawa(
                    a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &table, &mut ws, input,
                    &mut out,
                );
            }
            (Self::L2l(_), Method::Direct) => direct::l2l_yukawa(
                a.p, b.p, &a.level, &b.level, &a.frame, &b.frame, &mut ws, input, &mut out,
            ),
            (Self::M2l(_), Method::Rcr) => {
                let table = rcr::m2l_table(a.p, b.p, &a.level, &a.frame, &b.frame);
                rcr::m2l_yukawa(
                    a.p, b.p, &a.level, &a.frame, &b.frame, &table, &mut ws, input, &mut out,
                );
            }
            (Self::M2l(_), Method::Direct) => direct::m2l_yukawa(
                a.p, b.p, &a.level, &a.frame, &b.frame, &mut ws, input, &mut out,
            ),
        }
        out
    }

    /// Sources for the input: in the input box with its corners for a multipole, at
    /// 2.5 a_P from the parent's centre for a local.
    pub fn sources(self, rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
        let (from, _) = self.frames();
        if self.input_is_multipole() {
            in_box(rng, n, &from, false)
        } else {
            shell(rng, n, &from, 2.5 * 3.0_f64.sqrt() * from.radius)
        }
    }

    /// The complex input of degree p from `sources` and `charges` (P2M or P2L).
    pub fn input(self, k: C, p: usize, sources: &[[f64; 3]], charges: &[C]) -> Vec<f64> {
        let (from, _) = self.frames();
        let level = Level::new(k, from.radius);
        let mut ws = Workspace::new(p);
        let mut out = vec![0.0; 2 * (p + 1) * (p + 1)];
        if self.input_is_multipole() {
            leaf::p2m(p, &level, &from, sources, charges, &mut ws, &mut out);
        } else {
            leaf::p2l(p, &level, &from, sources, charges, &mut ws, &mut out);
        }
        out
    }

    /// The Yukawa input of degree p from real charges.
    pub fn input_yukawa(self, k: C, p: usize, sources: &[[f64; 3]], charges: &[f64]) -> Vec<f64> {
        let (from, _) = self.frames();
        let level = Level::new(k, from.radius);
        let mut ws = Workspace::new(p);
        let mut out = vec![0.0; (p + 1) * (p + 1)];
        if self.input_is_multipole() {
            leaf::p2m_yukawa(p, &level, &from, sources, charges, &mut ws, &mut out);
        } else {
            leaf::p2l_yukawa(p, &level, &from, sources, charges, &mut ws, &mut out);
        }
        out
    }

    /// The field of a complex output at the targets (M2P or L2P), with gradients.
    pub fn evaluate(self, k: C, p: usize, output: &[f64], targets: &[[f64; 3]]) -> Field {
        let (_, to) = self.frames();
        let level = Level::new(k, to.radius);
        let mut ws = Workspace::new(p);
        let mut field = Field::zero(targets.len());
        let (phi, grad) = (&mut field.potential, Some(&mut field.gradient[..]));
        if self.output_is_multipole() {
            leaf::m2p(p, &level, &to, output, targets, &mut ws, phi, grad);
        } else {
            leaf::l2p(p, &level, &to, output, targets, &mut ws, phi, grad);
        }
        field
    }
}

/// `n` points uniform in the box of `frame`, its corners and, with `centre`, its centre.
pub fn in_box(rng: &mut SplitMix64, n: usize, frame: &Frame<f64>, centre: bool) -> Vec<[f64; 3]> {
    let (c, r) = (frame.centre, frame.radius);
    let mut points = rng.points(n, c, r);
    for corner in 0..8 {
        let sign = |bit: usize| if corner >> bit & 1 == 1 { 1.0 } else { -1.0 };
        points.push(core::array::from_fn(|i| c[i] + sign(i) * r));
    }
    if centre {
        points.push(c);
    }
    points
}

/// `n` points at distance `radius` from the frame's centre in random directions.
pub fn shell(rng: &mut SplitMix64, n: usize, frame: &Frame<f64>, radius: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            loop {
                let u: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
                let norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                if norm > 0.1 && norm <= 1.0 {
                    break core::array::from_fn(|i| frame.centre[i] + radius * u[i] / norm);
                }
            }
        })
        .collect()
}

/// The offsets of the M2L tests in debug: the nearest (2, 0, 0), the axis ±z (θ = 0 and
/// π), the farthest (3, 3, 3) and four general ones.
pub const OFFSETS: [[i32; 3]; 8] = [
    [2, 0, 0],
    [0, 0, 2],
    [0, 0, -3],
    [3, 3, 3],
    [-2, 1, -1],
    [1, -3, 2],
    [-3, -2, 0],
    [2, 2, -3],
];

/// The 316 offsets of the V list (CONVENTIONS §3.12): max |offset| of 2 or 3.
pub fn all_offsets() -> Vec<[i32; 3]> {
    let mut offsets = Vec::new();
    for i in -3..=3 {
        for j in -3..=3 {
            for k in -3..=3 {
                let o: [i32; 3] = [i, j, k];
                if o.iter().any(|x| x.abs() >= 2) {
                    offsets.push(o);
                }
            }
        }
    }
    offsets
}

/// The FMM weights wₙ = √((2n + 1)/(4π)) |fₙ(kR)|, n ≤ p, of an expansion on `frame`'s
/// level: a multipole at the nearest points of its V list, R = (4 − √3) r,
/// fₙ = 4π ik ĥₙ(kR) e^{β − Im(k)R}; a local on its sphere, R = a, fₙ = ĵₙ(kR) e^{Im(k)R − β}.
pub fn weights(k: C, frame: &Frame<f64>, multipole: bool, p: usize) -> Vec<f64> {
    let level = Level::new(k, frame.radius);
    let mut f = vec![C::new(0.0, 0.0); p + 1];
    let e = if multipole {
        let r = (4.0 - 3.0_f64.sqrt()) * frame.radius;
        hankel_h(k * r, level.scale(), &mut f, None);
        (level.damping() - k.im * r).exp() * 4.0 * std::f64::consts::PI * k.norm()
    } else {
        let r = level.circumradius();
        bessel_j(k * r, level.scale(), &mut f, None);
        (k.im * r - level.damping()).exp()
    };
    f.iter()
        .enumerate()
        .map(|(n, x)| x.norm() * e * ((2 * n + 1) as f64 / (4.0 * std::f64::consts::PI)).sqrt())
        .collect()
}

/// ‖C‖_w = √(Σₙ wₙ² ‖Cₙ‖²) of two planes of degree p = weights.len() − 1, from terms
/// scaled by the largest, so that tiny expansions do not underflow.
pub fn weighted_norm(weights: &[f64], c: &[f64]) -> f64 {
    let len = c.len() / 2;
    let terms: Vec<f64> = (0..weights.len())
        .flat_map(|n| {
            (n * n..(n + 1) * (n + 1))
                .flat_map(move |i| [i, len + i])
                .map(move |j| (n, j))
        })
        .map(|(n, j)| weights[n] * c[j])
        .collect();
    let largest = terms.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if largest == 0.0 {
        return 0.0;
    }
    largest
        * terms
            .iter()
            .map(|v| (v / largest).powi(2))
            .sum::<f64>()
            .sqrt()
}
