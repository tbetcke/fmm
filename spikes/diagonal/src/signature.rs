//! S → F and F → R in any precision (proposal §5.4).
//!
//! With the kernel G(x, y) = e^{ik|x−y|}/(4π|x−y|) = ik Σₙ jₙ hₙ Σₘ Yₙᵐ Yₙᵐ of S7
//! (`nd_fmm_spike_truncation::chain`), multipoles Mₙᵐ = Σ q jₙ(k|y − c_s|) Yₙᵐ(y − c_s)
//! and locals evaluated as φ = ik Σ Lₙᵐ jₙ(k|x − c_t|) Yₙᵐ(x − c_t), the identity
//!
//! G(x, y) = (ik/16π²) ∫ e^{ikŝ·(x − c_t)} T(ŝ) e^{−ikŝ·(y − c_s)} dŝ,
//! T(ŝ) = Σₗ (2l + 1) iˡ hₗ(k|t|) Pₗ(ŝ · t̂), t = c_t − c_s
//!
//! (proposal §3.6; the seed of Appendix B.1), with e^{±ikŝ·r} = 4π Σ (±i)ⁿ jₙ Σₘ Y(r̂) Y(ŝ),
//! gives
//!
//! - S → F: F(ŝ) = Σₙ (−i)ⁿ Σₘ Mₙᵐ Yₙᵐ(ŝ), the signature, of bandwidth p;
//! - the diagonal step G(ŝ) = T(ŝ) F(ŝ);
//! - F → R: Lₙᵐ = iⁿ ∫ Yₙᵐ(ŝ) G(ŝ) dŝ, by the grid's quadrature.
//!
//! Both transforms factor through the rings: a Legendre step per order m with the ring's
//! Θₙᵐ, and a discrete Fourier transform along the ring written as a product (§6.7's
//! default), O(p N_φ) per ring.

use crate::grid::{Tables, tri};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::harmonics::idx;
use nd_fmm_spike_truncation::real::Real;

/// z iᵏ, exactly.
#[inline(always)]
pub fn times_i_pow<T: Real>(z: Complex<T>, k: usize) -> Complex<T> {
    match k % 4 {
        0 => z,
        1 => Complex::new(-z.im, z.re),
        2 => Complex::new(-z.re, -z.im),
        _ => Complex::new(z.im, -z.re),
    }
}

/// S → F on one ring: the signature of the multipole `m` (degree p) at the ring's N_φ
/// samples, into `out`; `theta` is the ring's Θ ([`crate::grid::Grid::theta`]) and `ab`
/// scratch of length 2(p + 1).
pub fn s2f_ring<T: Real>(
    p: usize,
    theta: &[T],
    m: &[Complex<T>],
    tab: &Tables<T>,
    ab: &mut [Complex<T>],
    out: &mut [Complex<T>],
) {
    let nphi = tab.cos.len();
    // a_μ = Σₙ (−i)ⁿ M_{n,+μ} Θₙ^μ, b_μ = Σₙ (−i)ⁿ M_{n,−μ} Θₙ^μ; (−i)ⁿ = i^{3n}.
    let (a, b) = ab.split_at_mut(p + 1);
    for mu in 0..=p {
        let (mut sa, mut sb) = (Complex::ZERO, Complex::ZERO);
        for n in mu..=p {
            let th = theta[tri(n, mu)];
            sa += times_i_pow(m[idx(n, mu as isize)], 3 * n) * th;
            if mu > 0 {
                sb += times_i_pow(m[idx(n, -(mu as isize))], 3 * n) * th;
            }
        }
        a[mu] = sa;
        b[mu] = sb;
    }
    for (j, o) in out[..nphi].iter_mut().enumerate() {
        let mut f = a[0];
        let mut at = 0;
        for mu in 1..=p {
            at += j;
            if at >= nphi {
                at -= nphi;
            }
            f += a[mu] * tab.cos[at] + b[mu] * tab.sin[at];
        }
        *o = f;
    }
}

/// F → R on one ring: adds ring i's share of ∫ Yₙᵐ G dŝ (without the iⁿ of
/// [`finish`]) for the products `g` at its N_φ samples to `acc` (degree p); `cs` is
/// scratch of length 2(p + 1).
pub fn f2r_ring<T: Real>(
    p: usize,
    theta: &[T],
    weight: T,
    g: &[Complex<T>],
    tab: &Tables<T>,
    cs: &mut [Complex<T>],
    acc: &mut [Complex<T>],
) {
    let nphi = tab.cos.len();
    let (c, s) = cs.split_at_mut(p + 1);
    // C_μ = Σⱼ Gⱼ cos μφⱼ, S_μ = Σⱼ Gⱼ sin μφⱼ.
    for mu in 0..=p {
        let (mut sc, mut ss) = (Complex::ZERO, Complex::ZERO);
        let mut at = 0;
        for &gj in &g[..nphi] {
            sc += gj * tab.cos[at];
            ss += gj * tab.sin[at];
            at += mu;
            if at >= nphi {
                at -= nphi;
            }
        }
        c[mu] = sc * weight;
        s[mu] = ss * weight;
    }
    for n in 0..=p {
        acc[idx(n, 0)] += c[0] * theta[tri(n, 0)];
        for mu in 1..=n {
            let th = theta[tri(n, mu)];
            acc[idx(n, mu as isize)] += c[mu] * th;
            acc[idx(n, -(mu as isize))] += s[mu] * th;
        }
    }
}

/// The iⁿ of F → R, once every ring has been added.
pub fn finish<T: Real>(p: usize, acc: &mut [Complex<T>]) {
    for n in 0..=p {
        for mu in -(n as isize)..=n as isize {
            let i = idx(n, mu);
            acc[i] = times_i_pow(acc[i], n);
        }
    }
}
