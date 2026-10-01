//! `Domain::new`: the boxes of `compute_global_bounding_box` (its formula reproduced
//! without MPI) and boxes built by hand as [a, fl(a + w)] are accepted, whatever their
//! distance from the origin; non-cubic, non-finite and degenerate boxes are rejected.

use nd_fmm_exec::geometry::{Domain, GeometryError, SIDE_TOLERANCE};
use nd_octree::PhysicalBox;

use crate::common::{EPS64, SplitMix64, Worst, padded_box, sides};

/// The largest pairwise side difference of a box, in units of ε₆₄ (M + w), the unit of
/// [`SIDE_TOLERANCE`].
fn mismatch(bounding_box: &PhysicalBox) -> f64 {
    let w = sides(bounding_box);
    let side = w[0].max(w[1]).max(w[2]);
    let narrowest = w[0].min(w[1]).min(w[2]);
    let largest_corner = bounding_box
        .coordinates()
        .iter()
        .fold(0.0_f64, |m, c| m.max(c.abs()));
    (side - narrowest) / (EPS64 * (largest_corner + side))
}

/// A seeded point cloud: `cloud` selects the distance of its centre from the origin
/// (0, 1, 10³ or 10⁹ times its extent) and the shape (equal extents, or extents in
/// the ratio 1 : 10⁻³ : 10⁻⁷, or a flat cloud with one extent zero).
fn cloud(rng: &mut SplitMix64, cloud: usize) -> Vec<[f64; 3]> {
    let extent = 10f64.powf(rng.range(-3.0, 3.0));
    let distance = match cloud % 4 {
        0 => 0.0,
        1 => 1.0,
        2 => 1e3 * extent,
        _ => 1e9 * extent,
    };
    let centre: [f64; 3] = core::array::from_fn(|_| distance * rng.range(-1.0, 1.0));
    let mut ratio = match (cloud / 4) % 3 {
        0 => [1.0, 1.0, 1.0],
        1 => [1.0, 1e-3, 1e-7],
        _ => [1.0, 0.3, 0.0],
    };
    // Rotate the shape so each axis is the long one in turn.
    ratio.rotate_right(cloud % 3);
    (0..40)
        .map(|_| core::array::from_fn(|k| centre[k] + ratio[k] * extent * rng.range(-1.0, 1.0)))
        .collect()
}

#[test]
fn global_bounding_boxes_are_accepted() {
    // Error measure: the largest side difference in units of ε₆₄ (M + w), printed;
    // acceptance is asserted.
    let mut rng = SplitMix64::new(0x7302);
    let mut worst = Worst::new("compute_global_bounding_box side mismatch / ε₆₄ (M + w)");
    let mut worst_far = Worst::new("  of which clouds 10⁹ extents from the origin");
    let mut worst_relative = Worst::new("  the same mismatch relative to w");
    for c in 0..100 {
        let bounding_box = padded_box(&cloud(&mut rng, c));
        let domain = Domain::new(&bounding_box)
            .unwrap_or_else(|e| panic!("cloud {c}: {} rejected: {e}", bounding_box));
        let w = sides(&bounding_box);
        assert_eq!(domain.side(), w[0].max(w[1]).max(w[2]), "cloud {c}");
        let c6 = bounding_box.coordinates();
        assert_eq!(domain.lower(), [c6[0], c6[1], c6[2]]);
        worst.record(mismatch(&bounding_box));
        let narrowest = w[0].min(w[1]).min(w[2]);
        worst_relative.record((domain.side() - narrowest) / domain.side());
        if c % 4 == 3 {
            worst_far.record(mismatch(&bounding_box));
        }
    }
    worst.check(SIDE_TOLERANCE);
    worst_far.check(SIDE_TOLERANCE);
    worst_relative.report();
}

#[test]
fn hand_built_boxes_are_accepted() {
    // Error measure: as above, for boxes [a, fl(a + w)] with |a| up to 10⁹ w.
    let mut rng = SplitMix64::new(0x7303);
    let mut worst = Worst::new("[a, fl(a + w)] side mismatch / ε₆₄ (M + w)");
    for case in 0..200 {
        let w = 10f64.powf(rng.range(-3.0, 3.0));
        let distance = [0.0, 1.0, 1e3 * w, 1e9 * w][case % 4];
        let a: [f64; 3] = core::array::from_fn(|_| distance * rng.range(-1.0, 1.0));
        let bounding_box = PhysicalBox::new([a[0], a[1], a[2], a[0] + w, a[1] + w, a[2] + w]);
        Domain::new(&bounding_box).unwrap_or_else(|e| panic!("{bounding_box} rejected: {e}"));
        worst.record(mismatch(&bounding_box));
    }
    worst.check(SIDE_TOLERANCE);
}

#[test]
fn unequal_sides_are_rejected_with_the_sides_named() {
    // Error measure: exact equality of the reported sides.
    let error = Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0 + 1e-9])).unwrap_err();
    match error {
        GeometryError::NotCubic {
            sides,
            difference,
            tolerance,
        } => {
            assert_eq!(sides, [1.0, 1.0, 1.0 + 1e-9]);
            assert!(difference > tolerance);
        }
        other => panic!("expected NotCubic, got {other:?}"),
    }
    let message = error.to_string();
    assert!(message.contains("[1.0, 1.0, 1.000000001]"), "{message}");
    println!("rejected: {message}");

    // A relative 1e-9 along x, near but not at the origin.
    let error =
        Domain::new(&PhysicalBox::new([-0.5, 0.25, 0.0, 0.5 + 1e-9, 1.25, 1.0])).unwrap_err();
    assert!(matches!(error, GeometryError::NotCubic { .. }), "{error:?}");
    println!("rejected: {error}");
}

#[test]
fn non_finite_and_degenerate_boxes_are_rejected() {
    // Error measure: the error variant and its payload, exactly.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for position in 0..6 {
            let mut coordinates = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
            coordinates[position] = bad;
            let error = Domain::new(&PhysicalBox::new(coordinates)).unwrap_err();
            assert!(
                matches!(error, GeometryError::NonFinite { .. }),
                "{coordinates:?}: {error:?}"
            );
        }
    }
    let degenerate = [
        [0.0; 6],
        [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
        [-1e308, -1e308, -1e308, 1e308, 1e308, 1e308],
    ];
    for coordinates in degenerate {
        let error = Domain::new(&PhysicalBox::new(coordinates)).unwrap_err();
        let want: [f64; 3] = core::array::from_fn(|k| coordinates[k + 3] - coordinates[k]);
        assert_eq!(
            error,
            GeometryError::InvalidSide { sides: want },
            "{coordinates:?}"
        );
        println!("rejected: {error}");
    }
}
