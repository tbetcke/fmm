//! The Racah-basis blocks of `nd_fmm_math::rotation::blocks` conjugated into the
//! orthonormal real basis, N Dⁿ N⁻¹ (CONVENTIONS §3.8), the low-degree reference of
//! proposal §3.2.

use nd_fmm_math::rotation::{block_range, blocks, blocks_len};

/// The ratios N_a / N_0, a = 0 to n, of degree n: 1 at a = 0, else
/// √(2 Πₜ₌₁..ₐ (n + t)/(n − t + 1)).
pub fn n_ratios(n: usize) -> Vec<f64> {
    let mut product = 1.0f64;
    (0..=n)
        .map(|a| {
            if a == 0 {
                return 1.0;
            }
            product *= (n + a) as f64 / (n - a + 1) as f64;
            (2.0 * product).sqrt()
        })
        .collect()
}

/// N Dⁿ(Q) N⁻¹, n ≤ p, in the layout of `nd_fmm_math::rotation::blocks`.
///
/// Entry (i, j) of Dⁿ is multiplied by N_|i| / N_|j|, from [`n_ratios`], so no factorial
/// is formed. In f64 the
/// ratio overflows near n ≈ 512 and the Racah entries themselves span 4ⁿ (proposal
/// §3.2); the spike measures where the accuracy is lost long before that.
pub fn conjugated_blocks(p: usize, q: &[[f64; 3]; 3]) -> Vec<f64> {
    let mut d = vec![0.0; blocks_len(p)];
    blocks(p, q, &mut d);
    for n in 0..=p {
        let block = &mut d[block_range(n)];
        let width = 2 * n + 1;
        let w = n_ratios(n);
        let ni = n as isize;
        for i in -ni..=ni {
            for j in -ni..=ni {
                let at = ((i + ni) as usize) * width + (j + ni) as usize;
                block[at] *= w[i.unsigned_abs()] / w[j.unsigned_abs()];
            }
        }
    }
    d
}
