//! The degree up to which N Dⁿ N⁻¹ from `nd_fmm_math::rotation::blocks` stays accurate,
//! measured against the Gumerov–Duraiswami blocks (which tests/accuracy.rs checks against
//! the arbitrary-precision reference to n = 1000). Prints Markdown for SPIKE_REPORT.md.
//!
//! `cargo run -p nd-fmm-spike-rotation-blocks --release --example sweep`

use nd_fmm_math::rotation::{block_range, euler_zyz};
use nd_fmm_spike_rotation_blocks::conjugated::conjugated_blocks;
use nd_fmm_spike_rotation_blocks::gd::euler_blocks;

const P: usize = 600;
const THRESHOLDS: [f64; 6] = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-2];

fn main() {
    let m2m = (1.0f64 / 3.0f64.sqrt()).acos();
    let cases: [(&str, [f64; 3]); 5] = [
        ("y, β = arccos(1/√3) (M2M)", [0.0, m2m, 0.0]),
        (
            "y, β = atan(1/3) (V list)",
            [0.0, (1.0f64 / 3.0).atan(), 0.0],
        ),
        ("y, β = π/2", [0.0, std::f64::consts::FRAC_PI_2, 0.0]),
        ("z-y-z (0.3, 1.1, −0.4)", [0.3, 1.1, -0.4]),
        ("z-y-z (2.9, 2.7, 0.6)", [2.9, 2.7, 0.6]),
    ];
    println!("Max |N Dⁿ N⁻¹ − Oⁿ| per degree against the GD blocks, f64.\n");
    print!("| rotation |");
    for t in THRESHOLDS {
        print!(" first n above {t:.0e} |");
    }
    println!(" first non-finite n | error at n = 20, 30, 40, 60, 80, 100 |");
    println!("| --- |{}", " ---: |".repeat(THRESHOLDS.len() + 2));
    for (name, [a, b, g]) in cases {
        let ours = euler_blocks(P, a, b, g);
        let theirs = conjugated_blocks(P, &euler_zyz(a, b, g));
        let errors: Vec<f64> = (0..=P)
            .map(|n| {
                let r = block_range(n);
                ours[r.clone()]
                    .iter()
                    .zip(&theirs[r])
                    .fold(0.0f64, |acc, (x, y)| {
                        if y.is_finite() {
                            acc.max((x - y).abs())
                        } else {
                            f64::INFINITY
                        }
                    })
            })
            .collect();
        // The first degree from which the error stays above t (errors are not monotone).
        let first_above = |t: f64| {
            let last_below = (0..=P).rev().find(|&n| errors[n] <= t);
            match last_below {
                Some(n) if n == P => "> 600".to_string(),
                Some(n) => (n + 1).to_string(),
                None => "0".to_string(),
            }
        };
        print!("| {name} |");
        for t in THRESHOLDS {
            print!(" {} |", first_above(t));
        }
        let nonfinite = errors
            .iter()
            .position(|e| !e.is_finite())
            .map_or("> 600".to_string(), |n| n.to_string());
        let at: Vec<String> = [20, 30, 40, 60, 80, 100]
            .iter()
            .map(|&n| format!("{:.1e}", errors[n]))
            .collect();
        println!(" {nonfinite} | {} |", at.join(", "));
    }
}
