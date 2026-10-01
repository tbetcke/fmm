//! The geometry lesson of CONVENTIONS §3.13: P2P between two neighbouring leaves on
//! level 12 of the generic domain (a = (0.1, −2.3, 7.9), w = 0.37, so r_12 ≈ 4.5e-5
//! and |x| ≈ 8), in f32.
//!
//! - **Leaf-scaled:** sources and targets stored as leaf-scaled f32 coordinates; the
//!   sources mapped into the target frame by ŷ = ĉ(s|t) + r̂(s|t) u_s with the exact
//!   `relative_frame`; `p2p` in f32 on (ŷ, u_t); the result φ̂ divided by r_t.
//! - **Absolute:** the same pair with the f32 roundings of the absolute coordinates.
//!
//! Both are compared with `direct_sum` in f64 over the original coordinates. Error
//! measure: per target |φ − φ_ref| / Σ |q| / |x − y|, the largest over the targets.
//! An absolute f32 coordinate near |x| ≈ 8 is off by up to ε₃₂ · 8 ≈ 5e-7, about 1e-2
//! of the leaf radius, so the absolute path loses about two digits; the leaf-scaled
//! path is off by ε₃₂ of the leaf radius. Both errors are printed; only the leaf-scaled
//! one is asserted, to 1e-6.

use nd_fmm_exec::geometry::{contains, leaf_coordinates, radius, relative_frame};
use nd_fmm_ref::p2p::{direct_sum, p2p};
use nd_octree::morton;

use crate::common::{SplitMix64, generic_domain};

/// Tolerance of the leaf-scaled f32 path, relative to Σ |q| / |x − y|.
const TOL: f64 = 1e-6;

#[test]
fn leaf_scaled_f32_p2p_beats_absolute_f32_at_level_12() {
    let mut rng = SplitMix64::new(0x730e);
    let domain = generic_domain();
    let level = 12;
    let r_t = radius(level, &domain);
    let (mut worst_scaled, mut worst_absolute) = (0.0_f64, 0.0_f64);
    for pair in 0..4 {
        // A target leaf and a neighbour across a face, an edge or a corner.
        let t = morton::from_index_and_level(
            [
                rng.below(4000) + 40,
                rng.below(4000) + 40,
                rng.below(4000) + 40,
            ],
            level,
        );
        let direction = [[1, 0, 0], [0, -1, 0], [1, 1, 0], [-1, 1, -1]][pair];
        let s = morton::key_in_direction(t, direction);
        let points = |rng: &mut SplitMix64, key, n| -> Vec<[f64; 3]> {
            let mut v = Vec::new();
            while v.len() < n {
                let x = rng.in_box(key, &domain);
                if contains(key, x, &domain) {
                    v.push(x);
                }
            }
            v
        };
        let sources = points(&mut rng, s, 40);
        let targets = points(&mut rng, t, 40);
        let charges: Vec<f64> = (0..sources.len()).map(|_| rng.range(-1.0, 1.0)).collect();
        let charges32: Vec<f32> = charges.iter().map(|&q| q as f32).collect();

        // Oracle and term magnitudes, in f64 over the original coordinates.
        let mut reference = vec![0.0; targets.len()];
        direct_sum(&sources, &charges, &targets, &mut reference, None);
        let magnitude: Vec<f64> = targets
            .iter()
            .map(|x| {
                sources
                    .iter()
                    .zip(&charges)
                    .map(|(y, q)| {
                        q.abs() / (0..3).map(|k| (x[k] - y[k]).powi(2)).sum::<f64>().sqrt()
                    })
                    .sum()
            })
            .collect();

        // Leaf-scaled: u_s, u_t in f32, ŷ = ĉ(s|t) + r̂(s|t) u_s in scratch.
        let frame = relative_frame::<f32>(s, t);
        let u_t: Vec<[f32; 3]> = targets
            .iter()
            .map(|&x| leaf_coordinates(x, t, &domain))
            .collect();
        let y_hat: Vec<[f32; 3]> = sources
            .iter()
            .map(|&y| {
                let u_s: [f32; 3] = leaf_coordinates(y, s, &domain);
                core::array::from_fn(|k| frame.centre[k] + frame.radius * u_s[k])
            })
            .collect();
        let mut phi_hat = vec![0.0_f32; targets.len()];
        p2p(&y_hat, &charges32, &u_t, &mut phi_hat, None);

        // Absolute: f32 roundings of the absolute coordinates.
        let to32 =
            |v: &[[f64; 3]]| -> Vec<[f32; 3]> { v.iter().map(|x| x.map(|c| c as f32)).collect() };
        let mut phi_absolute = vec![0.0_f32; targets.len()];
        p2p(
            &to32(&sources),
            &charges32,
            &to32(&targets),
            &mut phi_absolute,
            None,
        );

        for i in 0..targets.len() {
            let scaled = phi_hat[i] as f64 / r_t;
            worst_scaled = worst_scaled.max((scaled - reference[i]).abs() / magnitude[i]);
            worst_absolute =
                worst_absolute.max((phi_absolute[i] as f64 - reference[i]).abs() / magnitude[i]);
        }
    }
    println!(
        "level 12 P2P in f32, relative to sum |q| / |x - y|: leaf-scaled {worst_scaled:.3e}, \
         absolute {worst_absolute:.3e} (tolerance {TOL:.0e} on the leaf-scaled one)"
    );
    assert!(
        worst_scaled <= TOL,
        "leaf-scaled f32 P2P error {worst_scaled:e} exceeds {TOL:e}"
    );
}
