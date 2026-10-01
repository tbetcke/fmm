//! The C3.2 gate in its smallest form (docs/phase3/README.md, "Exit gate"): a uniform
//! level-4 tree with N = 10⁵ points uniform in a cube, sources equal to targets, charges
//! uniform in [−1, 1), f64, at p = 3 and 8. The full report, with p = 18, f32, the
//! gradients and the stage timings, is the `fmm_accuracy` example of nd-fmm-validate.
//!
//! Error measure: the relative L2 error of φ over 1,000 targets sampled with a fixed
//! seed (design §8.2), against `direct_sum` in f64 over all sources divided by 4π (the
//! oracle expands 1/|x − y|, `Fmm` returns Σ q / (4π |x − y|)), as the root mean square
//! over eight seeded charge vectors: √((1/8) Σₖ eₖ²) with eₖ = ‖φₖ − φₖ*‖ / ‖φₖ*‖
//! (docs/phase3/README.md, "Error measures"). Each vector counts equally; pooling the
//! sums instead would weight a vector by ‖φₖ*‖², and a vector with a large, smooth
//! potential (a small relative error) would dominate. The gate is twice the
//! single-translation prediction of Phase 1 (design
//! §7, "Single-translation accuracy", P2M → M2L → L2P): 1.77e-3 at p = 3 and 1.08e-5 at
//! p = 8.
//!
//! *Why several charge vectors.* With mixed-sign charges the far field of the coarsest
//! V-list level (64 boxes on level 2) partly cancels at each target, by an amount that
//! depends on the charge vector, while the truncation errors of its pairs do not
//! cancel. On this tree the error of one vector varied by a factor of 2.3 (p = 3) and
//! 2.7 (p = 8) across seven seeds, with every pair within its single-translation
//! error. The mean over several vectors measures the FMM, not the draw.
//!
//! Its own executable, because it initialises MPI; ignored, because it needs release
//! mode:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release -- --ignored
//! ```

use mpi::traits::*;
use nd_fmm_exec::fmm::FmmBuilder;
use nd_fmm_ref::p2p::direct_sum;

/// The single-translation prediction of the relative L2 error of φ (design §7).
const PREDICTION: [(usize, f64); 2] = [(3, 1.77e-3), (8, 1.08e-5)];

/// SplitMix64, as in the other tests.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }

    /// Uniform in 0..n (n ≤ 2³², with a negligible bias).
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
}

/// The seed of the points and the sample; charge vector k has the seed `SEED + 1 + k`.
const SEED: u64 = 0xc32;

/// The number of charge vectors the error is averaged over.
const CHARGE_VECTORS: u64 = 8;

#[test]
#[ignore = "release mode: N = 10^5 and eight direct sums at 1,000 targets"]
fn uniform_tree_within_twice_the_prediction() {
    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    assert_eq!(comm.size(), 1, "the gate runs on one rank");

    let n = 100_000;
    let mut rng = SplitMix64(SEED);
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
        .collect();
    // 1,000 distinct targets: a partial Fisher–Yates shuffle.
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..1000 {
        let j = i + rng.below(n - i);
        order.swap(i, j);
    }
    let sample = &order[..1000];
    let sampled: Vec<[f64; 3]> = sample.iter().map(|&i| points[i]).collect();
    let charges: Vec<Vec<f64>> = (0..CHARGE_VECTORS)
        .map(|k| {
            let mut rng = SplitMix64(SEED + 1 + k);
            (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
        })
        .collect();
    let exact: Vec<Vec<f64>> = charges
        .iter()
        .map(|q| {
            let mut phi = vec![0.0; sampled.len()];
            direct_sum(&points, q, &sampled, &mut phi, None);
            phi.iter()
                .map(|v| v / (4.0 * std::f64::consts::PI))
                .collect()
        })
        .collect();

    // Every p is measured and printed before the gate is checked.
    let mut failures = Vec::new();
    for (p, prediction) in PREDICTION {
        let builder = FmmBuilder::<f64>::new(p)
            .max_level(4)
            .max_points_per_leaf(1);
        let mut fmm = builder
            .build(&points, &points, &comm)
            .expect("the FMM builds");
        let leaves = fmm.plan().index().leaves();
        assert_eq!(fmm.nleaves(), 4096);
        assert!((0..fmm.nleaves()).all(|j| leaves.level(j) == 4), "uniform");
        let mut each = Vec::new();
        for (q, exact) in charges.iter().zip(&exact) {
            let output = fmm.evaluate(q).expect("the FMM evaluates");
            let (mut e2, mut r2) = (0.0f64, 0.0f64);
            for (&i, &e) in sample.iter().zip(exact) {
                e2 += (output.potential[i] - e).powi(2);
                r2 += e * e;
            }
            each.push((e2 / r2).sqrt());
        }
        let error = (each.iter().map(|e| e * e).sum::<f64>() / each.len() as f64).sqrt();
        let each: Vec<String> = each.iter().map(|e| format!("{e:.2e}")).collect();
        eprintln!(
            "uniform level-4 tree, N = {n}, p = {p}: relative L2 error of φ {error:.3e} \
             (root mean square over {CHARGE_VECTORS} charge vectors: {}), prediction \
             {prediction:.2e}, ratio {:.2}",
            each.join(", "),
            error / prediction
        );
        if error > 2.0 * prediction {
            failures.push(format!(
                "p = {p}: {error:e} exceeds twice the prediction {prediction:e}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
