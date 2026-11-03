//! The pair chain of the Helmholtz exponential M2L on one level (Helmholtz CH6.1c;
//! CONVENTIONS §3.15, "Exponential M2L"): a stored multipole to plane waves, a shift, the
//! waves to a stored local, along any of the six directions of §3.14.
//!
//! # The representation
//!
//! The level's units are those of [`HelmholtzTables`](crate::helmholtz::HelmholtzTables):
//! half-width 1, k̂ = k r_l, a box represents r_l φ̃. The waves work in box widths
//! u = v/2, with κ = k w_l = 2k̂, where for ŝ = (t cos α, t sin α, iλ)/κ (ŝ·ŝ = 1)
//!
//! e^{iκŝ·u} = e^{−λ u_z + i t (u_x cos α + u_y sin α)}.
//!
//! The quadrature of [`super::quadrature`] (or of `nd_fmm_ref`'s generator) represents
//! e^{iκ|u|}/|u| = iκ h₀(κ|u|) by such waves for 1 ≤ u_z ≤ 4, |u_x|, |u_y| ≤ 4, and the
//! solid harmonics of ∇/(iκ) carry it to every degree:
//!
//! hₙ(κ|u|) Yₙᵐ(û) = 1/(iⁿ⁺¹ κ) ∫ dλ (1/2π) ∫ dα Yₙᵐ(ŝ) e^{iκŝ·u},   u_z > 0.
//!
//! Yₙ^{±m}(ŝ) is κ⁻ⁿ cₙₘ 𝒫̄ₙᵐ(iλ, t; κ²) {cos mα, sin mα}, 𝒫̄ₙᵐ the solid normalised
//! Legendre polynomial of [`harmonic_factors`] and cₙₘ = √((2 − δₘ₀)(2n + 1)/(4π)). With
//! the stored form M = e^β sⁿ M̂ of §3.15 and its power scale folded into the arguments,
//! the factor of node k is
//!
//! Pₖ(n, m) = i⁻ⁿ cₙₘ 𝒫̄ₙᵐ((s/κ) iλₖ, (s/κ) tₖ; s²),
//!
//! a polynomial, finite as κ → 0 (|s/κ| = √3/2 where s < 1). The **up waves** of a stored
//! multipole (target above) represent its r_l φ̃, r_l φ̃(c + w u) = Σₖ Σⱼ X e^{iκŝ·u}:
//!
//! X(λₖ, αⱼ) = 2π e^β (wₖ/Nₖ) Σ_{m ≤ Fₖ'} [Aₖ(m) cos mαⱼ + Bₖ(m) sin mαⱼ],
//! Aₖ(m) = Σₙ Pₖ(n, m) M̂ₙ^{+m},  Bₖ(m) = Σₙ Pₖ(n, m) M̂ₙ^{−m},
//!
//! with Fₖ' = min(Fₖ, p); the **down waves** (target below, e^{+λ u_z}) take
//! (−1)ⁿ⁻ᵐ Pₖ(n, m). A shift by d (box widths, the direction's frame) multiplies node
//! (k, j) by e^{∓λₖ d_z + i tₖ (d_x cos αⱼ + d_y sin αⱼ)} (up: −). From the incoming waves
//! V at the target's centre, by e^{iκŝ·u} = 4π Σₙ iⁿ jₙ(κ|u|) Σₘ Yₙᵐ(ŝ) Yₙᵐ(û),
//!
//! L̂ₙ^{+m} += 4π e^β (−1)ⁿ Pₖ(n, m) Σⱼ V cos mαⱼ,  L̂ₙ^{−m} the same with sin mαⱼ,
//!
//! for m ≤ min(Fₖ', n), with (−1)ⁿ⁻ᵐ more for down waves. Every wave value is complex
//! (the real basis has no conjugate symmetry to halve them, unlike §3.14's): Σₖ Nₖ
//! complex values per direction, stored as the real parts, then the imaginary parts,
//! node k's from [`NodeBlock::start`].
//!
//! # Directions and frames
//!
//! The six directions and their frames are §3.14's ([`Direction`], [`FRAMES`]). The
//! multipole goes into frame f by Oⁿ(Q) per degree and the local back by Oⁿ(Q)ᵀ (§3.15,
//! "Rotations": the basis is orthonormal, so the K and S of §3.14 do not appear), Oⁿ the
//! blocks of `CubeTransforms` (Helmholtz CH3.2b) for the two cube rotations of frames 1
//! and 2, built for those two alone.
//!
//! # The steps
//!
//! | Step | Here | Work per box |
//! | --- | --- | --- |
//! | the multipole into a frame | [`rotate_multipole`](PlaneWaveLevel::rotate_multipole) | O(p³) |
//! | its modes Aₖ(m), Bₖ(m), up and down | [`outgoing_modes`](PlaneWaveLevel::outgoing_modes) | O(s p²) |
//! | the waves from the modes | [`waves_from_modes`](PlaneWaveLevel::waves_from_modes) | O(W F) |
//! | a shift, an accumulation | [`shift_factors`](PlaneWaveLevel::shift_factors), [`shift_add`] | O(W) |
//! | the sums Σⱼ V cos mαⱼ, Σⱼ V sin mαⱼ | [`sums_from_waves`](PlaneWaveLevel::sums_from_waves) | O(W F) |
//! | the local from the sums | [`add_local_from_sums`](PlaneWaveLevel::add_local_from_sums) | O(s p²) |
//! | the local back from the frame | [`add_local_from_frame`](PlaneWaveLevel::add_local_from_frame) | O(p³) |
//!
//! W = Σₖ Nₖ and F the largest Fₖ' + 1. Each step runs in a fixed order and rounds every
//! product and sum apart; the tables are built in f64 and rounded to T, the shift factors
//! of [`m2l`](PlaneWaveLevel::m2l) are formed in f64 per call. [`PlaneWaveLevel::m2l`]
//! translates one V pair by these steps with one shift, from the source's centre to the
//! target's: the slower path that merge and shift (Helmholtz CH6.1d) equals to rounding.
//!
//! `tests/helmholtz/plane_wave_pair.rs` checks it against RCR (`HelmholtzTables::m2l`) on
//! every V offset, with FMM3D's and with generated quadratures; the identities are checked
//! in mpmath by `tools/fixtures/check_3_15_plane_wave.py`.

use core::f64::consts::{PI, TAU};

use nd_fmm_math::{Complex, Layout, RealScalar};
use nd_fmm_ref::helmholtz::Level;

use super::quadrature::Quadrature;
use crate::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use crate::helmholtz::classes::rotation_blocks;
use crate::plane_wave::geometry::Direction;
use crate::symmetry::SignedPermutation;

#[cfg(doc)]
use crate::plane_wave::FRAMES;

/// The complex scalar of the builders.
type C = Complex<f64>;

/// The 316 V-list offsets in table order (CONVENTIONS §3.12).
const OFFSETS: [[i64; 3]; M2L_OFFSET_COUNT] = m2l_offsets();

/// The nodes of a quadrature, from either source: CH6.1a's [`Quadrature`] or
/// `nd_fmm_ref::helmholtz::plane_wave::Quadrature` (its accessors of the same names).
#[derive(Clone, Copy, Debug)]
pub struct Nodes<'a> {
    /// The decay rates λₖ.
    pub lambda: &'a [Complex<f64>],
    /// The transverse wavenumbers tₖ = √(λₖ² + κ²).
    pub t: &'a [Complex<f64>],
    /// The weights wₖ (of dλ).
    pub weights: &'a [Complex<f64>],
    /// The α counts Nₖ.
    pub alpha: &'a [usize],
    /// The Fourier counts Fₖ: a source's orders |m| ≤ Fₖ are kept at λₖ.
    pub fourier: &'a [usize],
}

/// Where one λ node's values lie in the waves, the modes and the tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeBlock {
    /// Its first value among the W complex wave values of a direction.
    pub start: usize,
    /// Its values, Nₖ.
    pub len: usize,
    /// Its orders m = 0..`orders`, Fₖ' + 1 = min(Fₖ, p) + 1.
    pub orders: usize,
    /// Its first order among all the nodes' orders, Σ_{l<k} (F_l' + 1).
    pub mode_start: usize,
    /// Its first value in the cos and sin tables: `orders` rows of `len`, row m at
    /// `trig + m len`.
    pub trig: usize,
    /// Its first factor: for m = 0..`orders`, the degrees n = m..=p.
    pub factor: usize,
}

/// One level's tables of the exponential M2L (the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct PlaneWaveLevel<T: RealScalar> {
    k_hat: C,
    p: usize,
    lambda: Vec<C>,
    t: Vec<C>,
    blocks: Vec<NodeBlock>,
    /// The wave values W = Σₖ Nₖ.
    waves: usize,
    /// The orders Σₖ (Fₖ' + 1).
    orders: usize,
    /// cos αⱼ and sin αⱼ per wave value, in f64, for the shift factors.
    alpha_cos: Vec<f64>,
    alpha_sin: Vec<f64>,
    /// cos mαⱼ and sin mαⱼ per node, order and value ([`NodeBlock::trig`]).
    cos: Vec<T>,
    sin: Vec<T>,
    /// 2π e^β (wₖ/Nₖ) Pₖ(n, m), real and imaginary parts ([`NodeBlock::factor`]).
    out_re: Vec<T>,
    out_im: Vec<T>,
    /// 4π e^β (−1)ⁿ Pₖ(n, m).
    in_re: Vec<T>,
    in_im: Vec<T>,
    /// Oⁿ(Q) of frames 1 and 2, per degree row-major at Σ_{i<n} (2i + 1)².
    frames: [Vec<T>; 2],
}

/// The scratch of [`PlaneWaveLevel::m2l`] and [`PlaneWaveLevel::waves_of`], made by
/// [`PlaneWaveLevel::scratch`].
#[derive(Clone, Debug)]
pub struct PlaneWaveLevelScratch<T> {
    frame: Vec<T>,
    modes: Vec<T>,
    up: Vec<T>,
    down: Vec<T>,
    incoming: Vec<T>,
    factors: Vec<T>,
    sums: Vec<T>,
    local: Vec<T>,
}

/// Writes cₙₘ 𝒫̄ₙᵐ(z, ρ; r²), 0 ≤ m ≤ n ≤ p, to `out` at
/// `nd_fmm_math::orthonormal::legendre::index(n, m)`: the polar factors of the
/// orthonormal solid harmonics of CONVENTIONS §3.15 at a point x = (ρ cos α, ρ sin α, z),
/// complex or real, with r² = x·x, so that rⁿ Yₙ^{±m}(x̂) = out[(n, m)] {cos mα, sin mα},
/// cₙₘ = √((2 − δₘ₀)(2n + 1)/(4π)). 𝒫̄ₙᵐ is rⁿ P̄ₙᵐ(z/r) for the normalised Legendre
/// functions of `nd_fmm_math::orthonormal::legendre` (no Condon–Shortley phase), a
/// polynomial in z, ρ and r², by their recursion in n per order m:
///
/// 𝒫̄ₘᵐ = Πᵢ₌₁..ₘ √((2i − 1)/(2i)) ρᵐ,  𝒫̄ₘ₊₁ᵐ = √(2m + 1) z 𝒫̄ₘᵐ,
/// 𝒫̄ₙᵐ = ((2n − 1) z 𝒫̄ₙ₋₁ᵐ − cₙ₋₁ r² 𝒫̄ₙ₋₂ᵐ) / cₙ,  cₙ = √((n − m)(n + m)).
///
/// In f64; the plain recursion, which the degrees of the low-frequency levels (p of a few
/// tens) do not strain.
///
/// # Panics
///
/// If `out.len()` is not (p + 1)(p + 2)/2.
pub fn harmonic_factors(p: usize, z: C, rho: C, r2: C, out: &mut [C]) {
    assert_eq!(out.len(), (p + 1) * (p + 2) / 2, "factor slice length");
    let index = |n: usize, m: usize| n * (n + 1) / 2 + m;
    let mut start = C::new(1.0, 0.0);
    for m in 0..=p {
        if m > 0 {
            start = start * rho * ((2 * m - 1) as f64 / (2 * m) as f64).sqrt();
        }
        let mut previous = C::new(0.0, 0.0);
        let mut current = start;
        for n in m..=p {
            if n == m + 1 {
                previous = current;
                current = z * current * ((2 * m + 1) as f64).sqrt();
            } else if n > m + 1 {
                let c = |n: usize| (((n - m) * (n + m)) as f64).sqrt();
                let next = (z * current * (2 * n - 1) as f64 - r2 * previous * c(n - 1)) / c(n);
                previous = current;
                current = next;
            }
            let norm = if m == 0 { 1.0 } else { 2.0 };
            out[index(n, m)] = current * (norm * (2 * n + 1) as f64 / (4.0 * PI)).sqrt();
        }
    }
}

/// `acc` += `waves` · `factors` value by value, complex, on split planes of equal length:
/// per value the real part re_a re_f − im_a im_f and the imaginary part
/// re_a im_f + im_a re_f added, each product and sum rounded apart.
///
/// # Panics
///
/// If the three slices differ in length or have an odd length.
pub fn shift_add<T: RealScalar>(acc: &mut [T], waves: &[T], factors: &[T]) {
    assert!(
        acc.len() == waves.len() && waves.len() == factors.len() && acc.len().is_multiple_of(2),
        "wave slice lengths"
    );
    let w = acc.len() / 2;
    let (acc_re, acc_im) = acc.split_at_mut(w);
    let (x_re, x_im) = waves.split_at(w);
    let (f_re, f_im) = factors.split_at(w);
    for i in 0..w {
        acc_re[i] = acc_re[i] + (x_re[i] * f_re[i] - x_im[i] * f_im[i]);
        acc_im[i] = acc_im[i] + (x_re[i] * f_im[i] + x_im[i] * f_re[i]);
    }
}

/// The element of O_h that rotates vectors into `frame` ≥ 1 as [`Direction::rotate`].
fn frame_element(frame: usize) -> SignedPermutation {
    let direction = Direction::ALL[2 * frame];
    let unit = |a: usize| -> [i64; 3] { core::array::from_fn(|i| i64::from(i == a)) };
    SignedPermutation::all()
        .into_iter()
        .find(|g| (0..3).all(|a| g.apply(unit(a)) == direction.rotate(unit(a))))
        .expect("each frame's rotation is a cube element")
}

/// iⁿ for an integer n.
fn i_pow(n: i64) -> C {
    match n.rem_euclid(4) {
        0 => C::new(1.0, 0.0),
        1 => C::new(0.0, 1.0),
        2 => C::new(-1.0, 0.0),
        _ => C::new(0.0, -1.0),
    }
}

impl<T: RealScalar> PlaneWaveLevel<T> {
    /// The tables of the level with k̂ = k r_l (the [module documentation](self)) at degree
    /// p, with CH6.1a's quadrature of κ = 2k̂.
    ///
    /// # Panics
    ///
    /// If the quadrature's κ is not 2k̂, and as [`from_nodes`](Self::from_nodes).
    pub fn new(k_hat: Complex<f64>, p: usize, quadrature: &Quadrature) -> Self {
        assert!(
            quadrature.kappa() == k_hat * 2.0,
            "the quadrature's κ = {} is not 2k̂ = {}",
            quadrature.kappa(),
            k_hat * 2.0
        );
        Self::from_nodes(
            k_hat,
            p,
            Nodes {
                lambda: quadrature.lambda(),
                t: quadrature.t(),
                weights: quadrature.weights(),
                alpha: quadrature.alpha_counts(),
                fourier: quadrature.fourier_counts(),
            },
        )
    }

    /// The tables of the level with k̂ = k r_l at degree p with the nodes of a quadrature
    /// of κ = 2k̂ (for instance `nd_fmm_ref`'s generated one). Builds in f64 and rounds to
    /// T; allocates.
    ///
    /// # Panics
    ///
    /// Unless k̂ is finite with Re k̂ > 0 and Im k̂ ≥ 0 (as `HelmholtzTables`: Yukawa's real
    /// path, Re k = 0, is not served), the node slices have one length, and every Nₖ ≥ 1.
    pub fn from_nodes(k_hat: Complex<f64>, p: usize, nodes: Nodes<'_>) -> Self {
        assert!(
            k_hat.re.is_finite() && k_hat.im.is_finite() && k_hat.re > 0.0 && k_hat.im >= 0.0,
            "k̂ must be finite with Re > 0 and Im ≥ 0 (Yukawa's real path is not served)"
        );
        let s = nodes.lambda.len();
        assert!(
            [
                nodes.t.len(),
                nodes.weights.len(),
                nodes.alpha.len(),
                nodes.fourier.len()
            ]
            .iter()
            .all(|&n| n == s),
            "node slices of different lengths"
        );
        assert!(nodes.alpha.iter().all(|&n| n >= 1), "an α count of 0");
        let level = Level::new(k_hat, 1.0);
        let (scale, beta) = (level.scale(), level.damping());
        let kappa = k_hat * 2.0;
        let sk = C::new(scale, 0.0) / kappa;
        let r2 = C::new(scale * scale, 0.0);
        let growth = beta.exp();

        let mut blocks = Vec::with_capacity(s);
        let (mut waves, mut orders, mut trig, mut factor) = (0, 0, 0, 0);
        for k in 0..s {
            let block = NodeBlock {
                start: waves,
                len: nodes.alpha[k],
                orders: nodes.fourier[k].min(p) + 1,
                mode_start: orders,
                trig,
                factor,
            };
            waves += block.len;
            orders += block.orders;
            trig += block.orders * block.len;
            factor += (0..block.orders).map(|m| p + 1 - m).sum::<usize>();
            blocks.push(block);
        }

        let (mut alpha_cos, mut alpha_sin) = (Vec::with_capacity(waves), Vec::with_capacity(waves));
        let (mut cos, mut sin) = (Vec::with_capacity(trig), Vec::with_capacity(trig));
        let cast = |v: f64| T::from_f64(v);
        for b in &blocks {
            let n = b.len;
            for j in 0..n {
                let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
                alpha_cos.push(ca);
                alpha_sin.push(sa);
            }
            for m in 0..b.orders {
                for j in 0..n {
                    // mαⱼ reduced exactly: m j mod N of the N equal steps.
                    let (sa, ca) = (TAU * ((m * j) % n) as f64 / n as f64).sin_cos();
                    cos.push(cast(ca));
                    sin.push(cast(sa));
                }
            }
        }

        let mut harmonic = vec![C::new(0.0, 0.0); (p + 1) * (p + 2) / 2];
        let (mut out_re, mut out_im) = (Vec::with_capacity(factor), Vec::with_capacity(factor));
        let (mut in_re, mut in_im) = (Vec::with_capacity(factor), Vec::with_capacity(factor));
        for (k, b) in blocks.iter().enumerate() {
            let z = sk * C::i() * nodes.lambda[k];
            let rho = sk * nodes.t[k];
            harmonic_factors(p, z, rho, r2, &mut harmonic);
            let outgoing = nodes.weights[k] * (TAU * growth / b.len as f64);
            let incoming = 4.0 * PI * growth;
            for m in 0..b.orders {
                for n in m..=p {
                    let pnm = i_pow(-(n as i64)) * harmonic[n * (n + 1) / 2 + m];
                    let o = outgoing * pnm;
                    let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                    let i = pnm * (incoming * sign);
                    out_re.push(cast(o.re));
                    out_im.push(cast(o.im));
                    in_re.push(cast(i.re));
                    in_im.push(cast(i.im));
                }
            }
        }

        let frames = [1, 2].map(|f| {
            rotation_blocks(p, frame_element(f))
                .into_iter()
                .map(cast)
                .collect()
        });

        Self {
            k_hat,
            p,
            lambda: nodes.lambda.to_vec(),
            t: nodes.t.to_vec(),
            blocks,
            waves,
            orders,
            alpha_cos,
            alpha_sin,
            cos,
            sin,
            out_re,
            out_im,
            in_re,
            in_im,
            frames,
        }
    }

    /// k̂ = k r_l.
    pub fn k_hat(&self) -> Complex<f64> {
        self.k_hat
    }

    /// The degree p.
    pub fn p(&self) -> usize {
        self.p
    }

    /// The nodes' blocks.
    pub fn blocks(&self) -> &[NodeBlock] {
        &self.blocks
    }

    /// The reals of the waves of one direction, 2W (W = Σₖ Nₖ complex values).
    pub fn wave_len(&self) -> usize {
        2 * self.waves
    }

    /// The reals of the modes of [`outgoing_modes`](Self::outgoing_modes), 8 per order
    /// and node.
    pub fn mode_len(&self) -> usize {
        8 * self.orders
    }

    /// The reals of the sums of [`sums_from_waves`](Self::sums_from_waves), 4 per order
    /// and node.
    pub fn sum_len(&self) -> usize {
        4 * self.orders
    }

    /// The bytes of the tables: the nodes and the α angles in f64, the cos, sin, factor
    /// and frame tables in T.
    pub fn bytes(&self) -> usize {
        let f64s =
            2 * (self.lambda.len() + self.t.len()) + self.alpha_cos.len() + self.alpha_sin.len();
        let ts = self.cos.len()
            + self.sin.len()
            + self.out_re.len()
            + self.out_im.len()
            + self.in_re.len()
            + self.in_im.len()
            + self.frames.iter().map(Vec::len).sum::<usize>();
        f64s * size_of::<f64>() + ts * size_of::<T>()
    }

    /// The scratch of [`m2l`](Self::m2l) and [`waves_of`](Self::waves_of).
    pub fn scratch(&self) -> PlaneWaveLevelScratch<T> {
        let (n, w) = (2 * Layout::new(self.p).len(), self.wave_len());
        PlaneWaveLevelScratch {
            frame: vec![T::zero(); n],
            modes: vec![T::zero(); self.mode_len()],
            up: vec![T::zero(); w],
            down: vec![T::zero(); w],
            incoming: vec![T::zero(); w],
            factors: vec![T::zero(); w],
            sums: vec![T::zero(); self.sum_len()],
            local: vec![T::zero(); n],
        }
    }

    /// Writes the multipole of `planes` planes (1 or 2, (p + 1)² values each) rotated into
    /// `frame` to the first `planes` planes of `out`: per plane and degree Oⁿ(Q) x, slot i
    /// the sum over the slots j ascending (frame 0 copies).
    ///
    /// # Panics
    ///
    /// If `frame` ≥ 3, `planes` is not 1 or 2, `multipole` does not hold `planes` planes
    /// or `out` fewer.
    pub fn rotate_multipole(&self, frame: usize, multipole: &[T], planes: usize, out: &mut [T]) {
        assert!(frame < 3, "frame {frame}");
        let len = Layout::new(self.p).len();
        assert!(matches!(planes, 1 | 2), "{planes} planes");
        assert_eq!(multipole.len(), planes * len, "multipole length");
        assert!(out.len() >= planes * len, "rotated multipole length");
        if frame == 0 {
            out[..planes * len].copy_from_slice(multipole);
            return;
        }
        let blocks = &self.frames[frame - 1];
        for plane in 0..planes {
            let (x, y) = (&multipole[plane * len..], &mut out[plane * len..]);
            let mut at = 0;
            for n in 0..=self.p {
                let width = 2 * n + 1;
                for i in 0..width {
                    let row = &blocks[at + i * width..at + (i + 1) * width];
                    let mut sum = T::zero();
                    for j in 0..width {
                        sum = sum + (row[j] * x[n * n + j]);
                    }
                    y[n * n + i] = sum;
                }
                at += width * width;
            }
        }
    }

    /// `out` += the local `local` (two planes, in `frame`) rotated back to the box's frame:
    /// per plane and degree Oⁿ(Q)ᵀ L, slot j the sum over the slots i ascending, then added
    /// (frame 0 adds `local`).
    ///
    /// # Panics
    ///
    /// If `frame` ≥ 3 or a slice does not hold two planes.
    pub fn add_local_from_frame(&self, frame: usize, local: &[T], out: &mut [T]) {
        assert!(frame < 3, "frame {frame}");
        let len = Layout::new(self.p).len();
        assert!(
            local.len() == 2 * len && out.len() == 2 * len,
            "local lengths"
        );
        if frame == 0 {
            for (o, &x) in out.iter_mut().zip(local) {
                *o = *o + (x);
            }
            return;
        }
        let blocks = &self.frames[frame - 1];
        for plane in 0..2 {
            let (x, y) = (&local[plane * len..], &mut out[plane * len..]);
            let mut at = 0;
            for n in 0..=self.p {
                let width = 2 * n + 1;
                for j in 0..width {
                    let mut sum = T::zero();
                    for i in 0..width {
                        sum = sum + (blocks[at + i * width + j] * x[n * n + i]);
                    }
                    y[n * n + j] = y[n * n + j] + (sum);
                }
                at += width * width;
            }
        }
    }

    /// Writes the modes of a multipole in a frame (`planes` planes): per node k and order
    /// m (at [`NodeBlock::mode_start`] + m, 8 reals each) Aₖ(m) and Bₖ(m) of the up waves,
    /// then of the down waves, each complex (re, im), with the waves' weights
    /// 2π e^β wₖ/Nₖ folded in (the [module documentation](self)). Per order the sums over
    /// n ascending, the degrees of even and odd n − m apart (up their sum, down their
    /// difference). `multipole` holds `planes` planes, an imaginary plane absent as zero.
    ///
    /// # Panics
    ///
    /// If `planes` is not 1 or 2, `multipole` has fewer than `planes` planes or `modes`
    /// does not hold [`mode_len`](Self::mode_len) values.
    pub fn outgoing_modes(&self, multipole: &[T], planes: usize, modes: &mut [T]) {
        let len = Layout::new(self.p).len();
        assert!(matches!(planes, 1 | 2), "{planes} planes");
        assert!(multipole.len() >= planes * len, "multipole length");
        assert_eq!(modes.len(), self.mode_len(), "mode length");
        let (re, im) = (&multipole[..len], &multipole[len.min(multipole.len())..]);
        let im_of = |i: usize| if planes == 2 { im[i] } else { T::zero() };
        let layout = Layout::new(self.p);
        for b in &self.blocks {
            let mut f = b.factor;
            for m in 0..b.orders {
                // [even A, odd A, even B, odd B], (re, im) each.
                let mut acc = [T::zero(); 8];
                for n in m..=self.p {
                    let (fr, fi) = (self.out_re[f], self.out_im[f]);
                    f += 1;
                    let parity = 2 * ((n - m) % 2);
                    let i = layout.idx(n, m as isize);
                    let (ar, ai) = (re[i], im_of(i));
                    acc[parity] = acc[parity] + (fr * ar - fi * ai);
                    acc[parity + 1] = acc[parity + 1] + (fr * ai + fi * ar);
                    if m > 0 {
                        let i = layout.idx(n, -(m as isize));
                        let (br, bi) = (re[i], im_of(i));
                        acc[parity + 4] = acc[parity + 4] + (fr * br - fi * bi);
                        acc[parity + 5] = acc[parity + 5] + (fr * bi + fi * br);
                    }
                }
                let out = &mut modes[8 * (b.mode_start + m)..8 * (b.mode_start + m + 1)];
                // Up: A, B as even + odd; down: even − odd.
                for (q, base) in [(0, 0), (1, 4)] {
                    for c in 0..2 {
                        out[2 * q + c] = acc[base + c] + acc[base + 2 + c];
                        out[4 + 2 * q + c] = acc[base + c] - acc[base + 2 + c];
                    }
                }
            }
        }
    }

    /// Writes the up and down waves (2W reals each) from the modes of
    /// [`outgoing_modes`](Self::outgoing_modes): per node and value j, from zero, for every
    /// order m ascending A(m) cos mαⱼ then B(m) sin mαⱼ added, each product rounded apart.
    ///
    /// # Panics
    ///
    /// If a slice has another length.
    pub fn waves_from_modes(&self, modes: &[T], up: &mut [T], down: &mut [T]) {
        assert_eq!(modes.len(), self.mode_len(), "mode length");
        let w = self.waves;
        assert!(up.len() == 2 * w && down.len() == 2 * w, "wave lengths");
        for b in &self.blocks {
            for j in 0..b.len {
                // (re, im) of the up and of the down wave.
                let mut x = [[T::zero(); 2]; 2];
                for m in 0..b.orders {
                    let (c, s) = (
                        self.cos[b.trig + m * b.len + j],
                        self.sin[b.trig + m * b.len + j],
                    );
                    let md = &modes[8 * (b.mode_start + m)..8 * (b.mode_start + m + 1)];
                    for (x, md) in x.iter_mut().zip(md.as_chunks::<4>().0) {
                        x[0] = x[0] + (md[0] * c);
                        x[0] = x[0] + (md[2] * s);
                        x[1] = x[1] + (md[1] * c);
                        x[1] = x[1] + (md[3] * s);
                    }
                }
                let v = b.start + j;
                [up[v], up[w + v]] = x[0];
                [down[v], down[w + v]] = x[1];
            }
        }
    }

    /// Writes the shift factors by `d` (box widths, in the direction's frame) of the up
    /// (`up`) or down waves, e^{∓λₖ d_z + i tₖ (d_x cos αⱼ + d_y sin αⱼ)} per node and
    /// value, formed in f64 and rounded to T, 2W reals. Allocates nothing.
    ///
    /// # Panics
    ///
    /// If `out` does not hold 2W values.
    pub fn shift_factors(&self, d: [f64; 3], up: bool, out: &mut [T]) {
        let w = self.waves;
        assert_eq!(out.len(), 2 * w, "factor length");
        let sign = if up { -1.0 } else { 1.0 };
        for (k, b) in self.blocks.iter().enumerate() {
            for j in b.start..b.start + b.len {
                let lateral = d[0] * self.alpha_cos[j] + d[1] * self.alpha_sin[j];
                let e = (self.lambda[k] * (sign * d[2]) + C::i() * self.t[k] * lateral).exp();
                out[j] = T::from_f64(e.re);
                out[w + j] = T::from_f64(e.im);
            }
        }
    }

    /// Writes the sums of incoming waves `waves` (2W reals): per node k and order m (at
    /// [`NodeBlock::mode_start`] + m, 4 reals each) Σⱼ V cos mαⱼ then Σⱼ V sin mαⱼ,
    /// complex (re, im), over the values j ascending.
    ///
    /// # Panics
    ///
    /// If a slice has another length.
    pub fn sums_from_waves(&self, waves: &[T], sums: &mut [T]) {
        let w = self.waves;
        assert_eq!(waves.len(), 2 * w, "wave length");
        assert_eq!(sums.len(), self.sum_len(), "sum length");
        for b in &self.blocks {
            for m in 0..b.orders {
                let mut g = [T::zero(); 4];
                for j in 0..b.len {
                    let (c, s) = (
                        self.cos[b.trig + m * b.len + j],
                        self.sin[b.trig + m * b.len + j],
                    );
                    let (vr, vi) = (waves[b.start + j], waves[w + b.start + j]);
                    g[0] = g[0] + (vr * c);
                    g[1] = g[1] + (vi * c);
                    g[2] = g[2] + (vr * s);
                    g[3] = g[3] + (vi * s);
                }
                sums[4 * (b.mode_start + m)..4 * (b.mode_start + m + 1)].copy_from_slice(&g);
            }
        }
    }

    /// `local` (two planes, in the frame) += the local of the sums of
    /// [`sums_from_waves`](Self::sums_from_waves) of up (`up`) or down waves: per node k,
    /// order m and degree n ≥ m ascending, slot +m += Q Σ V cos and slot −m += Q Σ V sin,
    /// Q = 4π e^β (−1)ⁿ Pₖ(n, m) (times (−1)ⁿ⁻ᵐ for down waves), complex.
    ///
    /// # Panics
    ///
    /// If a slice has another length.
    pub fn add_local_from_sums(&self, sums: &[T], up: bool, local: &mut [T]) {
        let len = Layout::new(self.p).len();
        assert_eq!(sums.len(), self.sum_len(), "sum length");
        assert_eq!(local.len(), 2 * len, "local length");
        let layout = Layout::new(self.p);
        let (re, im) = local.split_at_mut(len);
        for b in &self.blocks {
            let mut f = b.factor;
            for m in 0..b.orders {
                let g = &sums[4 * (b.mode_start + m)..4 * (b.mode_start + m + 1)];
                for n in m..=self.p {
                    let (mut qr, mut qi) = (self.in_re[f], self.in_im[f]);
                    f += 1;
                    if !up && (n - m) % 2 == 1 {
                        (qr, qi) = (-qr, -qi);
                    }
                    let i = layout.idx(n, m as isize);
                    re[i] = re[i] + (qr * g[0] - qi * g[1]);
                    im[i] = im[i] + (qr * g[1] + qi * g[0]);
                    if m > 0 {
                        let i = layout.idx(n, -(m as isize));
                        re[i] = re[i] + (qr * g[2] - qi * g[3]);
                        im[i] = im[i] + (qr * g[3] + qi * g[2]);
                    }
                }
            }
        }
    }

    /// Writes the up and down waves (2W reals each) of a stored multipole of `planes`
    /// planes in `frame`: [`rotate_multipole`](Self::rotate_multipole),
    /// [`outgoing_modes`](Self::outgoing_modes), [`waves_from_modes`](Self::waves_from_modes).
    ///
    /// # Panics
    ///
    /// As those steps.
    pub fn waves_of(
        &self,
        frame: usize,
        multipole: &[T],
        planes: usize,
        up: &mut [T],
        down: &mut [T],
        scratch: &mut PlaneWaveLevelScratch<T>,
    ) {
        self.rotate_multipole(frame, multipole, planes, &mut scratch.frame);
        self.outgoing_modes(&scratch.frame, planes, &mut scratch.modes);
        self.waves_from_modes(&scratch.modes, up, down);
    }

    /// The value of up (`up`) or down waves `waves` at u (box widths from their centre, in
    /// their frame), Σₖ Σⱼ X e^{∓λₖ u_z + i tₖ (u_x cos αⱼ + u_y sin αⱼ)} in f64: the
    /// r_l φ̃ that the waves represent (the [module documentation](self)). For tests.
    ///
    /// # Panics
    ///
    /// If `waves` does not hold 2W values.
    pub fn evaluate(&self, waves: &[T], u: [f64; 3], up: bool) -> Complex<f64> {
        let w = self.waves;
        assert_eq!(waves.len(), 2 * w, "wave length");
        let sign = if up { -1.0 } else { 1.0 };
        let mut sum = C::new(0.0, 0.0);
        for (k, b) in self.blocks.iter().enumerate() {
            for j in b.start..b.start + b.len {
                let lateral = u[0] * self.alpha_cos[j] + u[1] * self.alpha_sin[j];
                let e = (self.lambda[k] * (sign * u[2]) + C::i() * self.t[k] * lateral).exp();
                sum += C::new(waves[j].to_f64(), waves[w + j].to_f64()) * e;
            }
        }
        sum
    }

    /// Adds the exponential M2L of the V pair with offset index `offset_index` (§3.12)
    /// along `direction` from `multipole` (`multipole_planes` planes: 2, or 1 for real k and
    /// real charges) to `local` (two planes), in the level's units: the source's waves in
    /// the direction's frame, one shift by the pair's offset, the local, and back to the
    /// box's frame. The trusted slower path of merge and shift; any direction along which
    /// the offset is 2 or 3 boxes lies in the quadrature's region. Allocates nothing.
    ///
    /// # Panics
    ///
    /// If `offset_index` ≥ 316, the offset is not 2 or 3 boxes along `direction`, a slice
    /// does not hold its planes of (p + 1)² values, or `scratch` has other lengths than
    /// [`scratch`](Self::scratch) makes.
    pub fn m2l(
        &self,
        offset_index: usize,
        direction: Direction,
        multipole: &[T],
        multipole_planes: usize,
        local: &mut [T],
        scratch: &mut PlaneWaveLevelScratch<T>,
    ) {
        let d = OFFSETS[offset_index];
        let offset = direction.rotate(d);
        let up = direction.is_up();
        let along = if up { offset[2] } else { -offset[2] };
        assert!(
            (2..=3).contains(&along),
            "the offset {d:?} is not 2 or 3 boxes along {direction:?}"
        );
        self.rotate_multipole(
            direction.frame(),
            multipole,
            multipole_planes,
            &mut scratch.frame,
        );
        let PlaneWaveLevelScratch {
            frame,
            modes,
            up: up_waves,
            down: down_waves,
            incoming,
            factors,
            sums,
            local: frame_local,
        } = scratch;
        self.outgoing_modes(frame, multipole_planes, modes);
        self.waves_from_modes(modes, up_waves, down_waves);
        self.shift_factors(offset.map(|x| x as f64), up, factors);
        incoming.fill(T::zero());
        shift_add(incoming, if up { up_waves } else { down_waves }, factors);
        self.sums_from_waves(incoming, sums);
        frame_local.fill(T::zero());
        self.add_local_from_sums(sums, up, frame_local);
        self.add_local_from_frame(direction.frame(), frame_local, local);
    }

    /// [`m2l`](Self::m2l) along [`Direction::of`] the offset: the first axis of z, y, x
    /// along which the target lies two or three boxes from the source.
    ///
    /// # Panics
    ///
    /// As [`m2l`](Self::m2l).
    pub fn m2l_of_offset(
        &self,
        offset_index: usize,
        multipole: &[T],
        multipole_planes: usize,
        local: &mut [T],
        scratch: &mut PlaneWaveLevelScratch<T>,
    ) {
        let direction =
            Direction::of(OFFSETS[offset_index]).expect("a V-list offset has a direction");
        self.m2l(
            offset_index,
            direction,
            multipole,
            multipole_planes,
            local,
            scratch,
        );
    }
}
