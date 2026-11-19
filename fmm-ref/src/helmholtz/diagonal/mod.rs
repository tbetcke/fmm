//! The diagonal form of the M2L on high-frequency levels (proposal §5.4; Helmholtz
//! CH1.3d1): S → F, the diagonal translation and F → R, with the transfer functions in
//! extended precision, the trusted path of CH4.1 and CH4.2.
//!
//! # The chain
//!
//! For the kernel e^{ikr}/r = 4π ik Σₙ jₙ hₙ Σₘ Yₙᵐ Yₙᵐ (CONVENTIONS §3.15, "Kernel and
//! factors") and a shift t = c_t − c_s between two boxes of one level, the identity
//!
//! ```text
//! e^{ik|x−y|}/(4π|x−y|) = (ik/16π²) ∫ e^{ik ŝ·(x − c_t)} T(ŝ) e^{−ik ŝ·(y − c_s)} dŝ,
//! T(ŝ) = Σₗ (2l + 1) iˡ hₗ(k|t|) Pₗ(ŝ · t̂)
//! ```
//!
//! (proposal §3.6; S6, `spikes/diagonal/SPIKE_REPORT.md` §1.1), with
//! e^{±ik ŝ·r} = 4π Σ (±i)ⁿ jₙ Σₘ Yₙᵐ(r̂) Yₙᵐ(ŝ), turns the M2L of §3.15 (multipole M of
//! degree p, unscaled, to local L) into
//!
//! - **S → F**: the signature F(ŝ) = Σₙ (−i)ⁿ Σₘ Mₙᵐ Yₙᵐ(ŝ) (for charges q at y,
//!   4π F(ŝ) = Σ q e^{−ik ŝ·(y − c_s)} up to the truncation);
//! - **the diagonal step**: G(ŝ) = T(ŝ) F(ŝ), pointwise;
//! - **F → R**: Lₙᵐ = 4π ik iⁿ ∫ Yₙᵐ(ŝ) G(ŝ) dŝ, by the [`Grid`]'s quadrature, exact for
//!   the product's degree when T is truncated at the bandwidth 2p.
//!
//! With T truncated at l ≤ 2p this is, in exact arithmetic, the truncated M2L that
//! [`rcr::m2l`](super::rcr::m2l) applies: the (S|R) coefficients of degrees n, n′ ≤ p
//! involve hₗ only for l ≤ n + n′. Its error is rounding alone, bounded by E of proposal
//! §3.6 ([`bound`]), which grows with max (2l + 1)|hₗ(k|t|)|: the nearest classes at small
//! k a fail it, and CH4.3's rule sends them to RCR.
//!
//! # Stored form
//!
//! With the stored coefficients M̂ = e^{−β} M / sⁿ and L̂ = e^β L sⁿ of §3.15 and the stored
//! transfer function T̂ = e^{Im z} T ([`Transfer`], z = k|t|):
//!
//! ```text
//! F̂(ŝ)   = Σₙ (−i)ⁿ sⁿ Σₘ M̂ₙᵐ Yₙᵐ(ŝ)                                [s2f]
//! L̂ₙᵐ   += sⁿ iⁿ Σᵢⱼ wᵢ (2π/N_φ) Yₙᵐ(ŝᵢⱼ) G(ŝᵢⱼ)                       [f2r]
//! m2l:     G = 4π ik e^{2β − Im(k)|t|} T̂ F̂,  L̂ += f2r(G)
//! ```
//!
//! the factor 4π ik e^{2β − Im(k)|t|} of RCR's M2L (`rcr`'s module documentation),
//! applied per sample after the product.
//!
//! # Storage and method
//!
//! Expansions are §3.15's complex ones, two real planes of (p + 1)² (real, then imaginary)
//! in the layout idx(n, m) of §3.6; the diagonal form needs the same degree at both ends.
//! A signature is (2p + 1) N_φ complex samples at i N_φ + j ([`Grid`]). Both transforms
//! factor through the rings, as in S6: on each ring a Legendre step per order m with the
//! ring's θ-factors, then the discrete Fourier transform along the ring written as a
//! product (O(p N_φ) per ring), O(p³) per box. This is the reference, not CH4.1's fast
//! transform.
//!
//! # Tested range
//!
//! `tests/helmholtz/diagonal.rs` (CH1.3d1, `docs/helmholtz/tasks/CH1.3d1.md`), f64 unless
//! said: the grid's quadrature (Yₙᵐ orthonormal to n = 2p, p ≤ 10); S → F → R the identity
//! to 10⁻¹³ (p ≤ 24, s = 1 and s < 1); the signature of a charge against e^{−ik ŝ·y} to
//! 10⁻¹²; the permuted transfer functions of the 316 offsets equal to direct ones within
//! 2 ulp of the largest sample; and [`m2l`] against [`rcr::m2l`](super::rcr::m2l) within 2E
//! ([`THETA`]) on the 34 classes for real k at k a = 2.83 to 54 and p = 7 to 82, weakly
//! damped k (Im k = Re k/10) at k a = 8, and in f32 at k a ≈ 16 (p = 30); on the 316
//! offsets through the permutations at k a = 5, p = 16; against the direct sum within
//! RCR's truncation plus 2E. Measured/E is at most 1.23 to k a = 32 and reaches 1.99 at
//! k a = 54 on the farthest classes, where E is the floor C₀ u and the f64 RCR rounds more
//! than the diagonal form: against the projection of [`direct::m2l`](super::direct::m2l)
//! the diagonal form errs 1.8–2.9·10⁻¹⁴ and RCR 3.4–6.9·10⁻¹⁴ (E = 2.7·10⁻¹⁴). Complex k
//! beyond weak damping and Yukawa are not tested: the diagonal form serves the
//! high-frequency levels of real and weakly damped k (proposal §3.6, "Regimes").
//!
//! # Panics
//!
//! The operators panic if the grid, transfer function or workspace has another degree, if a coefficient or signature slice has the wrong length,
//! if a frame's radius is not the level's half-width, and if the transfer function's shift
//! is not the frames' (to √ε of the half-width).
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::Frame;
//! use nd_fmm_ref::helmholtz::{Level, Workspace as LeafWorkspace, diagonal, leaf, rcr};
//!
//! let p = 12;
//! let level = Level::new(Complex::new(9.0_f64, 0.0), 0.5); // k a ≈ 7.8
//! let source = Frame::new([0.0; 3], 0.5);
//! let target = Frame::new([2.0, 1.0, -1.0], 0.5);
//! let mut ws = LeafWorkspace::new(p);
//! let mut multipole = vec![0.0; 2 * (p + 1) * (p + 1)];
//! leaf::p2m(p, &level, &source, &[[0.1, -0.2, 0.3]], &[Complex::new(1.0, 0.5)], &mut ws,
//!     &mut multipole);
//!
//! let grid = diagonal::Grid::new(p);
//! let transfer = diagonal::Transfer::new(&grid, &level, &source, &target);
//! let mut dws = diagonal::Workspace::new(p);
//! let mut local = vec![0.0; 2 * (p + 1) * (p + 1)];
//! diagonal::m2l(&grid, &level, &source, &target, &transfer, &mut dws, &multipole, &mut local);
//!
//! let table = rcr::m2l_table(p, p, &level, &source, &target);
//! let mut reference = vec![0.0; 2 * (p + 1) * (p + 1)];
//! rcr::m2l(p, p, &level, &source, &target, &table, &mut ws, &multipole, &mut reference);
//! let scale = reference.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
//! for (a, b) in local.iter().zip(&reference) {
//!     assert!((a - b).abs() < 1e-12 * scale);
//! }
//! ```

mod bound;
mod classes;
mod dd;
mod grid;
mod transfer;

pub use bound::{BoundConstants, THETA, bound};
pub use classes::{Class, Symmetry, class_of, classes, offsets};
pub use grid::{Grid, azimuth_count};
pub use transfer::Transfer;

use nd_fmm_math::orthonormal::legendre;
use nd_fmm_math::{Complex, Layout, RealScalar};

use super::Level;
use super::leaf::{check_coefficients, check_frame, four_pi_ik};
use crate::Frame;

/// Scratch memory of the diagonal form for degree p: one signature ((2p + 1) N_φ complex
/// values, 2.2 MB at p = 131 in f64), a complex accumulator of (p + 1)² and the per-ring
/// sums, 2(p + 1) complex values. A workspace of degree p serves the operators of degree p
/// only (the signature's length depends on it).
///
/// ```
/// use nd_fmm_ref::helmholtz::diagonal::Workspace;
///
/// let ws = Workspace::<f64>::new(8);
/// assert_eq!(ws.degree(), 8);
/// ```
pub struct Workspace<T> {
    p: usize,
    signature: Vec<Complex<T>>,
    accumulator: Vec<Complex<T>>,
    ring: Vec<Complex<T>>,
}

impl<T: RealScalar> Workspace<T> {
    /// The workspace for degree p.
    pub fn new(p: usize) -> Self {
        let zero = Complex::new(T::zero(), T::zero());
        Self {
            p,
            signature: vec![zero; (2 * p + 1) * azimuth_count(p)],
            accumulator: vec![zero; Layout::new(p).len()],
            ring: vec![zero; 2 * (p + 1)],
        }
    }

    /// The degree p.
    pub fn degree(&self) -> usize {
        self.p
    }
}

/// S → F: writes the signature F̂(ŝ) = Σₙ (−i)ⁿ sⁿ Σₘ M̂ₙᵐ Yₙᵐ(ŝ) of the stored complex
/// multipole (degree p = the grid's) to `signature`, at the grid's samples (module
/// documentation, "Stored form"). It overwrites `signature`.
///
/// # Panics
///
/// If the workspace has another degree than the grid, `multipole` is not 2(p + 1)² long
/// or `signature` not [`Grid::samples`] long.
pub fn s2f<T: RealScalar>(
    grid: &Grid<T>,
    level: &Level<T>,
    multipole: &[T],
    ws: &mut Workspace<T>,
    signature: &mut [Complex<T>],
) {
    let p = grid.degree();
    check_workspace(ws, p);
    check_coefficients(p, multipole, 2, "multipole");
    check_signature(grid, signature);
    signature_of(grid, level.scale(), multipole, &mut ws.ring, signature);
}

/// F → R: adds sⁿ iⁿ Σᵢⱼ wᵢ (2π/N_φ) Yₙᵐ(ŝᵢⱼ) G(ŝᵢⱼ) to the stored complex local (degree
/// p = the grid's), for the signature product G (module documentation, "Stored form").
/// It applies no factor 4π ik.
///
/// # Panics
///
/// As [`s2f`], with `local` in place of `multipole`.
pub fn f2r<T: RealScalar>(
    grid: &Grid<T>,
    level: &Level<T>,
    signature: &[Complex<T>],
    ws: &mut Workspace<T>,
    local: &mut [T],
) {
    let p = grid.degree();
    check_workspace(ws, p);
    check_coefficients(p, local, 2, "local");
    check_signature(grid, signature);
    local_of(
        grid,
        level.scale(),
        signature,
        &mut ws.ring,
        &mut ws.accumulator,
        local,
    );
}

/// The M2L by the diagonal form, from the stored complex multipole of the box `from` to
/// the stored complex local of the box `to` of `level`, both of the grid's degree p:
/// L̂ += f2r(4π ik e^{2β − Im(k)|t|} T̂ · s2f(M̂)) (module documentation). It adds into
/// `local` and allocates nothing.
///
/// # Panics
///
/// As [`s2f`] and [`f2r`]; if a frame's radius is not the level's half-width; if the
/// transfer function has another degree or another shift than to − from.
#[allow(clippy::too_many_arguments)]
pub fn m2l<T: RealScalar>(
    grid: &Grid<T>,
    level: &Level<T>,
    from: &Frame<T>,
    to: &Frame<T>,
    transfer: &Transfer<T>,
    ws: &mut Workspace<T>,
    multipole: &[T],
    local: &mut [T],
) {
    let p = grid.degree();
    check_workspace(ws, p);
    check_frame(level, from);
    check_frame(level, to);
    check_coefficients(p, multipole, 2, "multipole");
    check_coefficients(p, local, 2, "local");
    assert_eq!(
        transfer.degree(),
        p,
        "the transfer function has another degree than the grid"
    );
    let t: [T; 3] = core::array::from_fn(|i| to.centre[i] - from.centre[i]);
    let tolerance = T::epsilon().sqrt().to_f64() * level.half_width().to_f64();
    assert!(
        t.iter()
            .zip(transfer.shift())
            .all(|(&a, b)| (a.to_f64() - b).abs() <= tolerance),
        "the transfer function's shift {:?} is not the frames' {:?}",
        transfer.shift(),
        t.map(|x| x.to_f64())
    );
    let k = level.k();
    let distance = t[0].hypot(t[1]).hypot(t[2]);
    let factor = four_pi_ik(k) * (level.damping() + level.damping() - k.im * distance).exp();
    let s = level.scale();
    let Workspace {
        signature,
        accumulator,
        ring,
        ..
    } = ws;
    signature_of(grid, s, multipole, ring, signature);
    for (g, &tr) in signature.iter_mut().zip(transfer.samples()) {
        *g = tr * *g * factor;
    }
    local_of(grid, s, signature, ring, accumulator, local);
}

/// S → F into `signature`, with `ring` of 2(p + 1).
fn signature_of<T: RealScalar>(
    grid: &Grid<T>,
    s: T,
    multipole: &[T],
    ring: &mut [Complex<T>],
    signature: &mut [Complex<T>],
) {
    let p = grid.degree();
    let len = Layout::new(p).len();
    let (re, im) = multipole.split_at(len);
    let azimuths = grid.azimuths();
    let (cos, sin) = grid.azimuth_tables();
    let zero = Complex::new(T::zero(), T::zero());
    let (a, b) = ring.split_at_mut(p + 1);
    for i in 0..grid.rings() {
        let theta = grid.ring(i);
        // a_μ = Σₙ (−i)ⁿ sⁿ M̂_{n,+μ} Θₙ^μ, b_μ = Σₙ (−i)ⁿ sⁿ M̂_{n,−μ} Θₙ^μ.
        for mu in 0..=p {
            let (mut sa, mut sb) = (zero, zero);
            let mut power = s.powi(mu as i32);
            for n in mu..=p {
                let th = theta[legendre::index(n, mu)] * power;
                let plus = index(n, mu as isize);
                sa = sa + times_minus_i_pow(Complex::new(re[plus], im[plus]), n) * th;
                if mu > 0 {
                    let minus = index(n, -(mu as isize));
                    sb = sb + times_minus_i_pow(Complex::new(re[minus], im[minus]), n) * th;
                }
                power = power * s;
            }
            a[mu] = sa;
            b[mu] = sb;
        }
        // F(φⱼ) = a₀ + Σ_μ a_μ cos μφⱼ + b_μ sin μφⱼ.
        for (j, out) in signature[i * azimuths..(i + 1) * azimuths]
            .iter_mut()
            .enumerate()
        {
            let mut f = a[0];
            let mut at = 0;
            for mu in 1..=p {
                at += j;
                if at >= azimuths {
                    at -= azimuths;
                }
                f = f + a[mu] * cos[at] + b[mu] * sin[at];
            }
            *out = f;
        }
    }
}

/// F → R of `signature` added to `local`, with `ring` of 2(p + 1) and `accumulator` of
/// (p + 1)².
fn local_of<T: RealScalar>(
    grid: &Grid<T>,
    s: T,
    signature: &[Complex<T>],
    ring: &mut [Complex<T>],
    accumulator: &mut [Complex<T>],
    local: &mut [T],
) {
    let p = grid.degree();
    let len = Layout::new(p).len();
    let azimuths = grid.azimuths();
    let (cos, sin) = grid.azimuth_tables();
    let zero = Complex::new(T::zero(), T::zero());
    accumulator[..len].fill(zero);
    let (c, sn) = ring.split_at_mut(p + 1);
    for i in 0..grid.rings() {
        let weight = grid.weight(i);
        let theta = grid.ring(i);
        let g = &signature[i * azimuths..(i + 1) * azimuths];
        // C_μ = wᵢ Σⱼ Gⱼ cos μφⱼ, S_μ = wᵢ Σⱼ Gⱼ sin μφⱼ.
        for mu in 0..=p {
            let (mut sc, mut ss) = (zero, zero);
            let mut at = 0;
            for &gj in g {
                sc = sc + gj * cos[at];
                ss = ss + gj * sin[at];
                at += mu;
                if at >= azimuths {
                    at -= azimuths;
                }
            }
            c[mu] = sc * weight;
            sn[mu] = ss * weight;
        }
        for n in 0..=p {
            accumulator[index(n, 0)] =
                accumulator[index(n, 0)] + c[0] * theta[legendre::index(n, 0)];
            for mu in 1..=n {
                let th = theta[legendre::index(n, mu)];
                let (plus, minus) = (index(n, mu as isize), index(n, -(mu as isize)));
                accumulator[plus] = accumulator[plus] + c[mu] * th;
                accumulator[minus] = accumulator[minus] + sn[mu] * th;
            }
        }
    }
    // L̂ₙᵐ += sⁿ iⁿ (the accumulated sum).
    let (re, im) = local.split_at_mut(len);
    let mut power = T::one();
    for n in 0..=p {
        for m in -(n as isize)..=n as isize {
            let i = index(n, m);
            let v = times_i_pow(accumulator[i], n) * power;
            re[i] = re[i] + v.re;
            im[i] = im[i] + v.im;
        }
        power = power * s;
    }
}

/// idx(n, m) of §3.6.
fn index(n: usize, m: isize) -> usize {
    (n * n + n).wrapping_add_signed(m)
}

/// z iⁿ, exactly.
fn times_i_pow<T: RealScalar>(z: Complex<T>, n: usize) -> Complex<T> {
    match n % 4 {
        0 => z,
        1 => Complex::new(-z.im, z.re),
        2 => Complex::new(-z.re, -z.im),
        _ => Complex::new(z.im, -z.re),
    }
}

/// z (−i)ⁿ = z i^{3n}, exactly.
fn times_minus_i_pow<T: RealScalar>(z: Complex<T>, n: usize) -> Complex<T> {
    times_i_pow(z, 3 * (n % 4))
}

fn check_workspace<T>(ws: &Workspace<T>, p: usize) {
    assert_eq!(ws.p, p, "the workspace has degree {}, the grid {p}", ws.p);
}

fn check_signature<T: RealScalar>(grid: &Grid<T>, signature: &[Complex<T>]) {
    assert_eq!(
        signature.len(),
        grid.samples(),
        "the signature has length {}, expected {} for p = {}",
        signature.len(),
        grid.samples(),
        grid.degree()
    );
}
