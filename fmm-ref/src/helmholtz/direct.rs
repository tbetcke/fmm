//! Translations of the Helmholtz family by projection on a sphere (CONVENTIONS §3.15,
//! "Radial functions and scaling"): the trusted path that [`rcr`](super::rcr) is tested
//! against.
//!
//! Each operator evaluates the input expansion's field φ̃ with the [`leaf`]
//! evaluators ([`leaf::m2p`], [`leaf::l2p`] and their Yukawa forms) at the nodes of a
//! product quadrature on the output's sphere, of radius ρ = a′ (the output level's
//! circumradius) about the output centre c′: Gauss–Legendre in cos θ with Q nodes and
//! uniform in φ with 2Q, Q = p_in + p_out + ⌈|k| ρ⌉ + 40. With the real orthonormal
//! harmonics of the node's direction it forms
//!
//! ```text
//! Aₙᵐ = ∫ φ̃(c′ + ρ x̂) Yₙᵐ(x̂) dΩ,   Bₙᵐ = ∫ ρ ∂ρ φ̃(c′ + ρ x̂) Yₙᵐ(x̂) dΩ
//! ```
//!
//! (B from the evaluators' gradients, for a local output only) and divides by the output's
//! radial function in stored form (the leaf module's formulas at |v| = ρ):
//!
//! ```text
//! multipole  M̂ₙᵐ = Aₙᵐ / (4π ik ĥₙ(kρ) e^{β′ − Im(k)ρ})
//! local      L̂ₙᵐ = (conj(f) Aₙᵐ + conj(g) Bₙᵐ) / (|f|² + |g|²),
//!            f = ĵₙ(kρ) e^{Im(k)ρ − β′},  g = ρ k ĵₙ′(kρ) e^{Im(k)ρ − β′}
//! ```
//!
//! the local by least squares from the value and the radial derivative, since jₙ has zeros.
//! The Yukawa forms divide by 4πκ k̂ₙ(κρ) e^{β′ − κρ} and îₙ(κρ) e^{κρ − β′}. The
//! exponents are zero at ρ = a′.
//!
//! The projection gives the output coefficients of the input's truncated field, exactly
//! up to the quadrature's aliasing: the translation that RCR applies with rectangular
//! degrees. It is O(Q² p²) per call, an oracle, not a fast path. At low frequency the
//! division by ĵₙ(ka′) ~ (ka′)ⁿ/(2n + 1)!! amplifies the quadrature's rounding in the
//! high-degree local coefficients; their contribution to the field inside the box, where
//! a local is evaluated, stays at round-off. Compare locals by their fields, or by
//! coefficients weighted per degree by |ĵₙ(ka′)| (the FMM-weighted measure of the tests),
//! not coefficient by coefficient.
//!
//! The operators take the arguments of their [`rcr`](super::rcr) counterparts except the
//! table, and the same storage. The frames need not form an octree pair: M2M needs the
//! child's sphere inside the parent's (|t| + a_c ≤ a_P), L2L the same with the parent as
//! input, and M2L two boxes of one level whose spheres do not overlap (|t| ≥ 2a).
//!
//! # Tested range
//!
//! `tests/helmholtz/translation_*.rs` (Helmholtz CH1.3c) checks the projection against
//! [`rcr`](super::rcr) over the ranges listed there, the limit k → 0 against Laplace's
//! translations (its local coefficients carry rounding up to about 1e-11 of their degree's
//! size at |k| a = 10⁻⁶) and Yukawa two ways to κ a = 50.
//!
//! # Panics
//!
//! If the workspace was built for a smaller degree than max(p_in, p_out), if a frame's
//! radius is not its level's half-width, if the levels have different k, if a coefficient
//! slice has the wrong length, if the geometry above fails (with a relative slack of
//! 64 ε), and for the Yukawa forms if the level is not Yukawa.
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::Frame;
//! use nd_fmm_ref::helmholtz::{Level, Workspace, direct, rcr};
//!
//! let p = 6;
//! let k = Complex::new(2.0, 0.5);
//! let (child, parent) = (Level::new(k, 0.25), Level::new(k, 0.5));
//! let from = Frame::new([0.25, -0.25, 0.25], 0.25);
//! let to = Frame::new([0.0; 3], 0.5);
//! let mut ws = Workspace::new(p);
//! let len = 2 * (p + 1) * (p + 1);
//! let multipole: Vec<f64> = (0..len).map(|i| 1.0 / (1.0 + i as f64)).collect();
//!
//! let mut by_projection = vec![0.0; len];
//! direct::m2m(p, p, &child, &parent, &from, &to, &mut ws, &multipole, &mut by_projection);
//! let table = rcr::m2m_table(p, p, &child);
//! let mut by_rcr = vec![0.0; len];
//! rcr::m2m(p, p, &child, &parent, &from, &to, &table, &mut ws, &multipole, &mut by_rcr);
//! for (a, b) in by_projection.iter().zip(&by_rcr) {
//!     assert!((a - b).abs() < 1e-12);
//! }
//! ```

use nd_fmm_math::orthonormal::harmonics;
use nd_fmm_math::radial::{bessel_j, hankel_h, modified_i, modified_k};
use nd_fmm_math::{Complex, Layout, RealScalar};

use super::leaf::{self, check_coefficients, check_frame, check_yukawa, four_pi, four_pi_ik};
use super::rcr::{check_same_k, norm, shift};
use super::{Level, Workspace};
use crate::Frame;

/// M2M: adds the stored complex multipole `multipole_in` about `from` (degree `p_child`,
/// level `child`) as a multipole about `to` (degree `p_parent`, level `parent`) to
/// `multipole_out`, by projection (module documentation).
///
/// # Panics
///
/// As in the module documentation.
#[allow(clippy::too_many_arguments)]
pub fn m2m<T: RealScalar>(
    p_child: usize,
    p_parent: usize,
    child: &Level<T>,
    parent: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    multipole_in: &[T],
    multipole_out: &mut [T],
) {
    check_nested(child, parent, from, to);
    check_coefficients(p_child, multipole_in, 2, "multipole_in");
    check_coefficients(p_parent, multipole_out, 2, "multipole_out");
    let mut field = [Complex::new(T::zero(), T::zero())];
    project(p_child, p_parent, parent, to, ws, false, |ws, x, _| {
        field[0] = Complex::new(T::zero(), T::zero());
        leaf::m2p(
            p_child,
            child,
            from,
            multipole_in,
            &[x],
            ws,
            &mut field,
            None,
        );
        (field[0], Complex::new(T::zero(), T::zero()))
    });
    finish_multipole(p_parent, parent, ws, multipole_out);
}

/// L2L: adds the stored complex local `local_in` about `from` (degree `p_parent`, level
/// `parent`) as a local about `to` (degree `p_child`, level `child`) to `local_out`, by
/// projection (module documentation).
///
/// # Panics
///
/// As in the module documentation.
#[allow(clippy::too_many_arguments)]
pub fn l2l<T: RealScalar>(
    p_parent: usize,
    p_child: usize,
    parent: &Level<T>,
    child: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    local_in: &[T],
    local_out: &mut [T],
) {
    check_nested(child, parent, to, from);
    check_coefficients(p_parent, local_in, 2, "local_in");
    check_coefficients(p_child, local_out, 2, "local_out");
    let zero = Complex::new(T::zero(), T::zero());
    let (mut field, mut gradient) = ([zero], [[zero; 3]]);
    project(
        p_parent,
        p_child,
        child,
        to,
        ws,
        true,
        |ws, x, direction| {
            field[0] = zero;
            gradient[0] = [zero; 3];
            leaf::l2p(
                p_parent,
                parent,
                from,
                local_in,
                &[x],
                ws,
                &mut field,
                Some(&mut gradient),
            );
            (field[0], radial_part(gradient[0], direction))
        },
    );
    finish_local(p_child, child, ws, local_out);
}

/// M2L: adds the stored complex multipole `multipole` of the box `from` (degree `p_src`)
/// as a local about the box `to` of the same `level` (degree `p_tgt`) to `local`, by
/// projection (module documentation).
///
/// # Panics
///
/// As in the module documentation.
#[allow(clippy::too_many_arguments)]
pub fn m2l<T: RealScalar>(
    p_src: usize,
    p_tgt: usize,
    level: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    multipole: &[T],
    local: &mut [T],
) {
    check_apart(level, from, to);
    check_coefficients(p_src, multipole, 2, "multipole");
    check_coefficients(p_tgt, local, 2, "local");
    let zero = Complex::new(T::zero(), T::zero());
    let (mut field, mut gradient) = ([zero], [[zero; 3]]);
    project(p_src, p_tgt, level, to, ws, true, |ws, x, direction| {
        field[0] = zero;
        gradient[0] = [zero; 3];
        leaf::m2p(
            p_src,
            level,
            from,
            multipole,
            &[x],
            ws,
            &mut field,
            Some(&mut gradient),
        );
        (field[0], radial_part(gradient[0], direction))
    });
    finish_local(p_tgt, level, ws, local);
}

/// Yukawa M2M: as [`m2m`] on the real stored multipole M̂^Y, with the Yukawa evaluators
/// (module documentation).
///
/// # Panics
///
/// As [`m2m`], with (p + 1)² reals per expansion, and if the level is not Yukawa.
#[allow(clippy::too_many_arguments)]
pub fn m2m_yukawa<T: RealScalar>(
    p_child: usize,
    p_parent: usize,
    child: &Level<T>,
    parent: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    multipole_in: &[T],
    multipole_out: &mut [T],
) {
    check_yukawa(child);
    check_nested(child, parent, from, to);
    check_coefficients(p_child, multipole_in, 1, "multipole_in");
    check_coefficients(p_parent, multipole_out, 1, "multipole_out");
    let mut field = [T::zero()];
    project(p_child, p_parent, parent, to, ws, false, |ws, x, _| {
        field[0] = T::zero();
        leaf::m2p_yukawa(
            p_child,
            child,
            from,
            multipole_in,
            &[x],
            ws,
            &mut field,
            None,
        );
        (real(field[0]), real(T::zero()))
    });
    finish_multipole_yukawa(p_parent, parent, ws, multipole_out);
}

/// Yukawa L2L: as [`l2l`] on the real stored local L̂^Y, with the Yukawa evaluators
/// (module documentation).
///
/// # Panics
///
/// As [`l2l`], with (p + 1)² reals per expansion, and if the level is not Yukawa.
#[allow(clippy::too_many_arguments)]
pub fn l2l_yukawa<T: RealScalar>(
    p_parent: usize,
    p_child: usize,
    parent: &Level<T>,
    child: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    local_in: &[T],
    local_out: &mut [T],
) {
    check_yukawa(child);
    check_nested(child, parent, to, from);
    check_coefficients(p_parent, local_in, 1, "local_in");
    check_coefficients(p_child, local_out, 1, "local_out");
    let mut field = [T::zero()];
    project(p_parent, p_child, child, to, ws, false, |ws, x, _| {
        field[0] = T::zero();
        leaf::l2p_yukawa(p_parent, parent, from, local_in, &[x], ws, &mut field, None);
        (real(field[0]), real(T::zero()))
    });
    finish_local_yukawa(p_child, child, ws, local_out);
}

/// Yukawa M2L: as [`m2l`] on the real stored expansions, with the Yukawa evaluators
/// (module documentation).
///
/// # Panics
///
/// As [`m2l`], with (p + 1)² reals per expansion, and if the level is not Yukawa.
#[allow(clippy::too_many_arguments)]
pub fn m2l_yukawa<T: RealScalar>(
    p_src: usize,
    p_tgt: usize,
    level: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    multipole: &[T],
    local: &mut [T],
) {
    check_yukawa(level);
    check_apart(level, from, to);
    check_coefficients(p_src, multipole, 1, "multipole");
    check_coefficients(p_tgt, local, 1, "local");
    let mut field = [T::zero()];
    project(p_src, p_tgt, level, to, ws, false, |ws, x, _| {
        field[0] = T::zero();
        leaf::m2p_yukawa(p_src, level, from, multipole, &[x], ws, &mut field, None);
        (real(field[0]), real(T::zero()))
    });
    finish_local_yukawa(p_tgt, level, ws, local);
}

/// Fills the workspace's sums with A (the first (p_out + 1)²) and, with `derivative`, B
/// (the next (p_out + 1)²) on the sphere of radius ρ = a′ about `to`. `field(ws, x, x̂)`
/// returns φ̃(x) and ∂ρ φ̃(x) (the latter only read with `derivative`).
fn project<T: RealScalar>(
    p_in: usize,
    p_out: usize,
    out_level: &Level<T>,
    to: &Frame<T>,
    ws: &mut Workspace<T>,
    derivative: bool,
    mut field: impl FnMut(&mut Workspace<T>, [T; 3], [T; 3]) -> (Complex<T>, Complex<T>),
) {
    let len = Layout::new(p_out).len();
    let rho = out_level.circumradius();
    let bandwidth = (out_level.k().norm() * rho).to_f64().ceil() as usize;
    let q = p_in + p_out + bandwidth + 40;
    let azimuths = 2 * q;
    ws.translation(p_in.max(p_out)).sums[..2 * len].fill(Complex::new(T::zero(), T::zero()));
    let step = 2.0 * core::f64::consts::PI / azimuths as f64;
    for i in 0..q {
        let (x, weight) = gauss_legendre(q, i);
        let sin_theta = ((1.0 - x) * (1.0 + x)).sqrt();
        let weight = T::from_f64(weight * step);
        for j in 0..azimuths {
            let (sin_phi, cos_phi) = (step * j as f64).sin_cos();
            let direction = [
                T::from_f64(sin_theta * cos_phi),
                T::from_f64(sin_theta * sin_phi),
                T::from_f64(x),
            ];
            let node = core::array::from_fn(|d| to.centre[d] + rho * direction[d]);
            let (value, radial) = field(ws, node, direction);
            let buffers = ws.translation(p_in.max(p_out));
            let y = &mut buffers.harmonics[..len];
            harmonics(p_out, direction, y);
            let (a, b) = buffers.sums.split_at_mut(len);
            for (i, &yi) in y.iter().enumerate() {
                let w = weight * yi;
                a[i] = a[i] + value * w;
                if derivative {
                    b[i] = b[i] + radial * (w * rho);
                }
            }
        }
    }
}

/// Adds Aₙᵐ / (4π ik ĥₙ(ka′)) to the two planes of `out` (the exponent is zero at ρ = a′).
fn finish_multipole<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    ws: &mut Workspace<T>,
    out: &mut [T],
) {
    let len = Layout::new(p).len();
    let k = level.k();
    let rho = level.circumradius();
    let buffers = ws.translation(p);
    let radial = &mut buffers.radial[..=p];
    hankel_h(k * rho, level.scale(), radial, None);
    let e = (level.damping() - k.im * rho).exp();
    let (re, im) = out.split_at_mut(len);
    for (n, &h) in radial.iter().enumerate() {
        let denominator = four_pi_ik(k) * h * e;
        for i in n * n..(n + 1) * (n + 1) {
            let c = buffers.sums[i] / denominator;
            re[i] = re[i] + c.re;
            im[i] = im[i] + c.im;
        }
    }
}

/// Adds the least-squares local of the module documentation to the two planes of `out`.
fn finish_local<T: RealScalar>(p: usize, level: &Level<T>, ws: &mut Workspace<T>, out: &mut [T]) {
    let len = Layout::new(p).len();
    let k = level.k();
    let rho = level.circumradius();
    let buffers = ws.translation(p);
    let (values, derivatives) = (
        &mut buffers.radial[..=p],
        &mut buffers.radial_derivative[..=p],
    );
    bessel_j(k * rho, level.scale(), values, Some(derivatives));
    let e = (k.im * rho - level.damping()).exp();
    let (re, im) = out.split_at_mut(len);
    let (a, b) = buffers.sums.split_at(len);
    for n in 0..=p {
        let f = values[n] * e;
        let g = k * derivatives[n] * (rho * e);
        let denominator = f.norm_sqr() + g.norm_sqr();
        for i in n * n..(n + 1) * (n + 1) {
            let c = (f.conj() * a[i] + g.conj() * b[i]) / denominator;
            re[i] = re[i] + c.re;
            im[i] = im[i] + c.im;
        }
    }
}

/// Adds Re Aₙᵐ / (4πκ k̂ₙ(κa′)) to `out`.
fn finish_multipole_yukawa<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    ws: &mut Workspace<T>,
    out: &mut [T],
) {
    let kappa = level.k().im;
    let rho = level.circumradius();
    let buffers = ws.translation(p);
    let radial = &mut buffers.real[..=p];
    modified_k(kappa * rho, level.scale(), radial, None);
    let e = (level.damping() - kappa * rho).exp();
    for (n, &k) in radial.iter().enumerate() {
        add_degree(n, four_pi::<T>() * kappa * k * e, buffers.sums, out);
    }
}

/// Adds Re Aₙᵐ / îₙ(κa′) to `out`.
fn finish_local_yukawa<T: RealScalar>(
    p: usize,
    level: &Level<T>,
    ws: &mut Workspace<T>,
    out: &mut [T],
) {
    let kappa = level.k().im;
    let rho = level.circumradius();
    let buffers = ws.translation(p);
    let radial = &mut buffers.real[..=p];
    modified_i(kappa * rho, level.scale(), radial, None);
    let e = (kappa * rho - level.damping()).exp();
    for (n, &i) in radial.iter().enumerate() {
        add_degree(n, i * e, buffers.sums, out);
    }
}

/// Adds Re Aₙᵐ / `denominator` to the slots of degree n of the real `out`.
fn add_degree<T: RealScalar>(n: usize, denominator: T, sums: &[Complex<T>], out: &mut [T]) {
    let slots = n * n..(n + 1) * (n + 1);
    for (o, a) in out[slots.clone()].iter_mut().zip(&sums[slots]) {
        *o = *o + a.re / denominator;
    }
}

fn real<T: RealScalar>(x: T) -> Complex<T> {
    Complex::new(x, T::zero())
}

/// ∂ρ φ̃ = ∇φ̃ · x̂ at a node in the direction x̂.
fn radial_part<T: RealScalar>(gradient: [Complex<T>; 3], direction: [T; 3]) -> Complex<T> {
    gradient[0] * direction[0] + gradient[1] * direction[1] + gradient[2] * direction[2]
}

/// The i-th of q Gauss–Legendre nodes on [−1, 1] and its weight, by Newton's method on
/// the three-term recursion, in f64.
fn gauss_legendre(q: usize, i: usize) -> (f64, f64) {
    let mut x = (core::f64::consts::PI * (i as f64 + 0.75) / (q as f64 + 0.5)).cos();
    let mut derivative = 1.0;
    for _ in 0..100 {
        let (mut p0, mut p1) = (1.0, x);
        for j in 2..=q {
            let p2 = ((2 * j - 1) as f64 * x * p1 - (j - 1) as f64 * p0) / j as f64;
            p0 = p1;
            p1 = p2;
        }
        let (pq, pq1) = if q == 1 { (x, 1.0) } else { (p1, p0) };
        derivative = q as f64 * (x * pq - pq1) / (x * x - 1.0);
        let dx = pq / derivative;
        x -= dx;
        if dx.abs() <= 1e-16 * x.abs().max(1.0) {
            break;
        }
    }
    (x, 2.0 / ((1.0 - x * x) * derivative * derivative))
}

/// The child's sphere inside the parent's, both frames on their levels.
fn check_nested<T: RealScalar>(
    child: &Level<T>,
    parent: &Level<T>,
    child_frame: &Frame<T>,
    parent_frame: &Frame<T>,
) {
    check_frame(child, child_frame);
    check_frame(parent, parent_frame);
    check_same_k(child, parent);
    let t = norm(shift(child_frame, parent_frame));
    let slack = T::from_f64(64.0) * T::epsilon() * parent.circumradius();
    assert!(
        t + child.circumradius() <= parent.circumradius() + slack,
        "the child's sphere must lie inside the parent's"
    );
}

/// Two frames of `level` whose spheres do not overlap.
fn check_apart<T: RealScalar>(level: &Level<T>, from: &Frame<T>, to: &Frame<T>) {
    check_frame(level, from);
    check_frame(level, to);
    let a = level.circumradius();
    let slack = T::from_f64(64.0) * T::epsilon() * a;
    assert!(
        norm(shift(from, to)) + slack >= a + a,
        "the M2L needs boxes whose spheres do not overlap"
    );
}
