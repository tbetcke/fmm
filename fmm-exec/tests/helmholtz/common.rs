//! Helpers shared by the Helmholtz tests: a seeded generator, the direct sum with 1/(4π),
//! the relative L2 error and an output's vectors as complex values.

use std::f64::consts::PI;

use nd_fmm_exec::helmholtz::{Output, Wavenumber};
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p;

pub type C = Complex<f64>;

/// SplitMix64, uniform in [−1, 1).
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }
}

/// The relative L2 error of `a` against `b`.
pub fn relative_l2(a: &[C], b: &[C]) -> f64 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| (x - y).norm_sqr()).sum();
    let norm: f64 = b.iter().map(C::norm_sqr).sum();
    (diff / norm).sqrt()
}

/// φ and ∇φ of the kernel G_k (1/(4π) included) at `targets`, complex k = Re k + i Im k.
pub fn exact(
    k: &Wavenumber,
    sources: &[[f64; 3]],
    charges: &[C],
    targets: &[[f64; 3]],
) -> (Vec<C>, Vec<[C; 3]>) {
    let zero = C::new(0.0, 0.0);
    let mut phi = vec![zero; targets.len()];
    let mut grad = vec![[zero; 3]; targets.len()];
    let kc = C::new(k.re(), k.im());
    p2p::direct_sum(kc, sources, charges, targets, &mut phi, Some(&mut grad));
    let scale = 1.0 / (4.0 * PI);
    (
        phi.iter().map(|&x| x * scale).collect(),
        grad.iter().map(|g| g.map(|x| x * scale)).collect(),
    )
}

/// Vector v of `output` as complex values (imaginary parts zero on Yukawa's real path);
/// no gradients if the output has none.
pub fn output_of(output: &Output<f64>, v: usize) -> (Vec<C>, Vec<[C; 3]>) {
    let phi = output.potential_of(v);
    let im = |i: usize| phi.im.map_or(0.0, |im| im[i]);
    let potential = (0..phi.re.len())
        .map(|i| C::new(phi.re[i], im(i)))
        .collect();
    let Some(g) = output.gradient_of(v) else {
        return (potential, Vec::new());
    };
    let gradient = (0..g.re.len())
        .map(|i| core::array::from_fn(|a| C::new(g.re[i][a], g.im.map_or(0.0, |im| im[i][a]))))
        .collect();
    (potential, gradient)
}
