//! Plain-Rust f64 reference GEMM, deterministic test data and error measure.

use std::thread;

/// Real element types benchmarked by the spike.
pub trait Real: cubecl::prelude::Float + cubecl::prelude::CubeElement + Copy + Send + Sync {
    /// Short name used in the result table.
    const NAME: &'static str;
    /// Rounds an f64 to this type (named to avoid `num_traits` clashes).
    fn narrow(x: f64) -> Self;
    /// Widens this value to f64.
    fn widen(self) -> f64;
}

impl Real for f32 {
    const NAME: &'static str = "f32";
    fn narrow(x: f64) -> Self {
        x as f32
    }
    fn widen(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    const NAME: &'static str = "f64";
    fn narrow(x: f64) -> Self {
        x
    }
    fn widen(self) -> f64 {
        self
    }
}

/// Fills a vector with uniform values in [-1, 1) from a fixed-seed xorshift generator.
pub fn uniform<F: Real>(len: usize, seed: u64) -> Vec<F> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            F::narrow(2.0 * unit - 1.0)
        })
        .collect()
}

/// Computes C = A X in f64, with A of shape nc x nc and X of shape nc x b, all row-major.
///
/// The inputs are widened from `F`, so the reference sees exactly the values the device
/// sees and the comparison measures GEMM error only. Rows of C are split across threads
/// and columns are blocked so that a block of X stays in cache.
pub fn gemm_f64<F: Real>(a: &[F], x: &[F], nc: usize, b: usize) -> Vec<f64> {
    const COL_BLOCK: usize = 256;
    let a: Vec<f64> = a.iter().map(|v| v.widen()).collect();
    let x: Vec<f64> = x.iter().map(|v| v.widen()).collect();
    let mut c = vec![0.0f64; nc * b];
    let threads = thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(nc);
    let rows_per_thread = nc.div_ceil(threads);
    thread::scope(|s| {
        for (chunk, c_rows) in c.chunks_mut(rows_per_thread * b).enumerate() {
            let (a, x) = (&a, &x);
            s.spawn(move || {
                let row0 = chunk * rows_per_thread;
                for j0 in (0..b).step_by(COL_BLOCK) {
                    let j1 = (j0 + COL_BLOCK).min(b);
                    for (r, c_row) in c_rows.chunks_mut(b).enumerate() {
                        let i = row0 + r;
                        let c_blk = &mut c_row[j0..j1];
                        for k in 0..nc {
                            let aik = a[i * nc + k];
                            let x_blk = &x[k * b + j0..k * b + j1];
                            for (cv, xv) in c_blk.iter_mut().zip(x_blk) {
                                *cv += aik * xv;
                            }
                        }
                    }
                }
            });
        }
    });
    c
}

/// Relative max-norm error max|C - C_ref| / max|C_ref|.
pub fn rel_max_error<F: Real>(c: &[F], c_ref: &[f64]) -> f64 {
    assert_eq!(c.len(), c_ref.len());
    let (mut diff, mut norm) = (0.0f64, 0.0f64);
    for (v, r) in c.iter().zip(c_ref) {
        diff = diff.max((v.widen() - r).abs());
        norm = norm.max(r.abs());
    }
    diff / norm
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The blocked, threaded reference agrees with the textbook triple loop.
    #[test]
    fn reference_matches_triple_loop() {
        let (nc, b) = (25, 777);
        let a = uniform::<f64>(nc * nc, 1);
        let x = uniform::<f64>(nc * b, 2);
        let c = gemm_f64(&a, &x, nc, b);
        let mut max_err = 0.0f64;
        for i in 0..nc {
            for j in 0..b {
                let s: f64 = (0..nc).map(|k| a[i * nc + k] * x[k * b + j]).sum();
                max_err = max_err.max((s - c[i * b + j]).abs());
            }
        }
        assert!(max_err < 1e-13, "max_err = {max_err}");
    }
}
