//! Accumulation onto nonzero initial values, and empty sources and targets.

use nd_fmm_ref::p2p::{direct_sum, p2p};

use crate::common::{SplitMix64, bits, gradient_bits, naive};

#[test]
fn accumulates_onto_initial_values() {
    // Error measure: bit-for-bit equality. `p2p` adds each term into the initial value,
    // so it equals the naive loop started from that value. `direct_sum` adds its
    // compensated sum in one final addition, so it equals the initial value plus the
    // result on zeros.
    let mut rng = SplitMix64::new(0xacc);
    let (sources, charges) = rng.charged(40, [0.0; 3], 1.0);
    let targets: Vec<[f64; 3]> = (0..24).map(|_| rng.in_cube([0.0; 3], 1.0)).collect();
    let n = targets.len();
    let initial: Vec<f64> = (0..n).map(|_| rng.range(-50.0, 50.0)).collect();
    let initial_gradient: Vec<[f64; 3]> = (0..n)
        .map(|_| [0; 3].map(|_| rng.range(-500.0, 500.0)))
        .collect();

    let (mut phi, mut grad) = (initial.clone(), initial_gradient.clone());
    naive(&sources, &charges, &targets, &mut phi, &mut grad);
    let (mut potential, mut gradient) = (initial.clone(), initial_gradient.clone());
    p2p(
        &sources,
        &charges,
        &targets,
        &mut potential,
        Some(&mut gradient),
    );
    assert_eq!(bits(&potential), bits(&phi));
    assert_eq!(gradient_bits(&gradient), gradient_bits(&grad));
    let mut potential = initial.clone();
    p2p(&sources, &charges, &targets, &mut potential, None);
    assert_eq!(bits(&potential), bits(&phi));

    let (mut fresh, mut fresh_gradient) = (vec![0.0; n], vec![[0.0; 3]; n]);
    direct_sum(
        &sources,
        &charges,
        &targets,
        &mut fresh,
        Some(&mut fresh_gradient),
    );
    let expected: Vec<f64> = initial.iter().zip(&fresh).map(|(a, b)| a + b).collect();
    let expected_gradient: Vec<[f64; 3]> = initial_gradient
        .iter()
        .zip(&fresh_gradient)
        .map(|(a, b)| [0, 1, 2].map(|k| a[k] + b[k]))
        .collect();
    let (mut potential, mut gradient) = (initial.clone(), initial_gradient.clone());
    direct_sum(
        &sources,
        &charges,
        &targets,
        &mut potential,
        Some(&mut gradient),
    );
    assert_eq!(bits(&potential), bits(&expected));
    assert_eq!(gradient_bits(&gradient), gradient_bits(&expected_gradient));
    let mut potential = initial.clone();
    direct_sum(&sources, &charges, &targets, &mut potential, None);
    assert_eq!(bits(&potential), bits(&expected));
}

#[test]
fn empty_sources_leave_outputs_unchanged() {
    // Error measure: bit-for-bit equality with the (nonzero) initial values.
    let targets = [[0.5, -1.0, 2.0], [3.0, 0.25, -0.75]];
    let initial = [1.5, -2.25];
    let initial_gradient = [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]];

    let (mut potential, mut gradient) = (initial, initial_gradient);
    p2p::<f64>(&[], &[], &targets, &mut potential, Some(&mut gradient));
    assert_eq!(bits(&potential), bits(&initial));
    assert_eq!(gradient_bits(&gradient), gradient_bits(&initial_gradient));
    p2p::<f64>(&[], &[], &targets, &mut potential, None);
    assert_eq!(bits(&potential), bits(&initial));

    let (mut potential, mut gradient) = (initial, initial_gradient);
    direct_sum(&[], &[], &targets, &mut potential, Some(&mut gradient));
    assert_eq!(bits(&potential), bits(&initial));
    assert_eq!(gradient_bits(&gradient), gradient_bits(&initial_gradient));
    direct_sum(&[], &[], &targets, &mut potential, None);
    assert_eq!(bits(&potential), bits(&initial));

    let mut potential32 = [1.5f32, -2.25];
    p2p::<f32>(&[], &[], &[[0.0; 3]; 2], &mut potential32, None);
    assert_eq!(potential32, [1.5, -2.25]);
}

#[test]
fn empty_targets_are_accepted() {
    // Error measure: none; the calls must return without panicking.
    let sources = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    let charges = [1.0, -1.0];
    p2p(&sources, &charges, &[], &mut [], None);
    p2p(&sources, &charges, &[], &mut [], Some(&mut []));
    direct_sum(&sources, &charges, &[], &mut [], None);
    direct_sum(&sources, &charges, &[], &mut [], Some(&mut []));
    p2p::<f32>(&[], &[], &[], &mut [], Some(&mut []));
    direct_sum(&[], &[], &[], &mut [], Some(&mut []));
}
