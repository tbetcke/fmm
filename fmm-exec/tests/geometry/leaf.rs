//! `leaf_coordinates` and `contains` (CONVENTIONS §3.13, "Leaf-scaled coordinates" and
//! "Containment"): u within the error bound of the exact value, u in [−1, 1]³ up to
//! β_k plus that bound for every point `contains` assigns to a leaf (also at the leaf
//! faces), and the f32 u as the cast of the f64 one.

use nd_fmm_exec::geometry::{Domain, contains, leaf_coordinates, radius};
use nd_octree::{MortonKey, morton};

use crate::common::{
    DEEPEST, SplitMix64, Worst, bound32, bound64, containment_bound, error, exact_u,
    generic_domain, padded_box, sides,
};

/// Checks u of `x` in `leaf` against the exact value and the bounds of §3.13, in f64 and
/// f32. Records error / bound and returns the absolute f64 and f32 errors.
fn check_point(x: [f64; 3], leaf: MortonKey, domain: &Domain, ratio: &mut Worst) -> (f64, f64) {
    let (level, index) = morton::decode(leaf);
    let (a, w) = (domain.lower(), domain.side());
    let r = radius(level, domain);
    let u64 = leaf_coordinates::<f64>(x, leaf, domain);
    let u32 = leaf_coordinates::<f32>(x, leaf, domain);
    let (mut e64, mut e32) = (0.0_f64, 0.0_f64);
    for k in 0..3 {
        // f32 is the cast of the f64 value, never a computation in f32.
        assert_eq!(u32[k].to_bits(), (u64[k] as f32).to_bits());
        let exact = exact_u(x[k], a[k], w, level, index[k]);
        let b64 = bound64(x[k] - a[k], r, exact.0);
        let b32 = bound32(b64, exact.0);
        let (err64, err32) = (error(u64[k], exact), error(u32[k] as f64, exact));
        ratio.record((err64 / b64).max(err32 / b32));
        (e64, e32) = (e64.max(err64), e32.max(err32));
    }
    (e64, e32)
}

#[test]
fn leaf_coordinates_are_within_the_error_bound() {
    // Error measure: componentwise |ũ − u| against the exact u (double-double), as a
    // fraction of the §3.13 bound in f64 and in f32; the absolute errors are printed.
    let mut rng = SplitMix64::new(0x730b);
    let domain = generic_domain();
    let mut ratio = Worst::new("leaf-scaled u: error / §3.13 bound (f64 and f32)");
    for level in [0, 4, 10, 16] {
        let (mut e64, mut e32) = (0.0_f64, 0.0_f64);
        for _ in 0..50 {
            let leaf = rng.key(level);
            for _ in 0..20 {
                let x = rng.in_box(leaf, &domain);
                // The leaf of x, which differs from `leaf` only at a face.
                let leaf = morton::from_physical_point(x, &domain.physical_box(), level);
                assert!(contains(leaf, x, &domain));
                let (a, b) = check_point(x, leaf, &domain, &mut ratio);
                (e64, e32) = (e64.max(a), e32.max(b));
            }
        }
        println!("level {level:2}: largest |u - u_exact| f64 {e64:.2e}, f32 {e32:.2e}");
        // In f64 within 2^(l−51) + ε₆₄ for |u| ≤ 1 (to first order; checked with margin).
        assert!(e64 <= 1.01 * (2f64.powi(level as i32 - 51) + f64::EPSILON / 2.0));
    }
    ratio.check(1.0);
}

/// Points within a few ulps of the faces of random leaves on `level`, and of the
/// domain's own faces, strictly inside the domain.
fn face_probes(rng: &mut SplitMix64, domain: &Domain, level: usize, count: usize) -> Vec<[f64; 3]> {
    let [x0, y0, z0, x1, y1, z1] = domain.physical_box().coordinates();
    let (lo, hi) = ([x0, y0, z0], [x1, y1, z1]);
    let mut probes = Vec::new();
    for _ in 0..count {
        let axis = rng.below(3);
        let base: [f64; 3] = core::array::from_fn(|k| rng.range(lo[k], hi[k]));
        let side = hi[axis] - lo[axis];
        let face =
            lo[axis] + (1 + rng.below((1 << level) - 1)) as f64 * (side / (1 << level) as f64);
        for target in [face, lo[axis], hi[axis]] {
            let mut x = target;
            for _ in 0..4 {
                x = x.next_down();
            }
            for _ in 0..9 {
                let mut point = base;
                point[axis] = x;
                if lo[axis] < x && x < hi[axis] {
                    probes.push(point);
                }
                x = x.next_up();
            }
        }
    }
    probes
}

#[test]
fn contained_points_lie_in_the_unit_cube_up_to_the_bound() {
    // Error measure: max over components of (|u| − 1) / (β_k + E), for the exact u
    // (E = 0) and the stored u in f64 and f32 (E = the §3.13 bound); asserted ≤ 1. The
    // domains come from the `compute_global_bounding_box` formula, so their sides
    // differ by rounding.
    let mut rng = SplitMix64::new(0x730c);
    let mut ratio = Worst::new("containment: (|u| - 1) / (beta + E)");
    let mut excess = Worst::new("containment: largest |u| - 1, stored f32");
    let mut bound_ratio = Worst::new("containment probes: u error / §3.13 bound");
    let mut probes_total = 0;
    for (centre, spread) in [
        ([3.7e5 + 0.3, -1.2e6 - 0.7, 8.1e5 + 0.1], 37.25),
        ([0.2, -0.35, 0.05], 1.3),
    ] {
        let cloud: Vec<[f64; 3]> = (0..1000)
            .map(|_| centre.map(|c| c + spread * rng.range(-1.0, 1.0)))
            .collect();
        let domain = Domain::new(&padded_box(&cloud)).unwrap();
        let (a, w) = (domain.lower(), domain.side());
        let w_k = sides(&domain.physical_box());
        let mut points: Vec<([f64; 3], usize)> = cloud.iter().map(|&x| (x, DEEPEST)).collect();
        points.extend(cloud[..300].iter().map(|&x| (x, rng.below(DEEPEST + 1))));
        for level in [3, 9, DEEPEST] {
            points.extend(
                face_probes(&mut rng, &domain, level, 40)
                    .into_iter()
                    .map(|x| (x, level)),
            );
        }
        probes_total += points.len();
        for (x, level) in points {
            let leaf = morton::from_physical_point(x, &domain.physical_box(), level);
            assert!(contains(leaf, x, &domain), "x = {x:?}, level {level}");
            // No neighbour across a face contains x.
            for axis in 0..3 {
                for step in [-1, 1] {
                    let mut direction = [0; 3];
                    direction[axis] = step;
                    let neighbour = morton::key_in_direction(leaf, direction);
                    if morton::is_valid(neighbour) {
                        assert!(!contains(neighbour, x, &domain), "x = {x:?} in two leaves");
                    }
                }
            }
            let (_, index) = morton::decode(leaf);
            let r = radius(level, &domain);
            let u64 = leaf_coordinates::<f64>(x, leaf, &domain);
            let u32 = leaf_coordinates::<f32>(x, leaf, &domain);
            for k in 0..3 {
                let beta = containment_bound(w_k[k], w, level);
                let exact = exact_u(x[k], a[k], w, level, index[k]);
                let u_exact = exact.0 + exact.1;
                let b64 = bound64(x[k] - a[k], r, exact.0);
                let b32 = bound32(b64, exact.0);
                ratio.record((u_exact.abs() - 1.0) / beta);
                ratio.record((u64[k].abs() - 1.0) / (beta + b64));
                ratio.record((u32[k].abs() as f64 - 1.0) / (beta + b32));
                excess.record(u32[k].abs() as f64 - 1.0);
                bound_ratio
                    .record((error(u64[k], exact) / b64).max(error(u32[k] as f64, exact) / b32));
            }
        }
    }
    println!("containment: {probes_total} points incl. leaf-face probes, 2 domains");
    excess.report();
    bound_ratio.check(1.0);
    ratio.check(1.0);
}

#[test]
fn contains_agrees_with_every_deeper_key() {
    // Error measure: exact equality of keys. The ancestor on level l of the key on
    // level 16 is the key that `contains` computes on level l.
    let mut rng = SplitMix64::new(0x730d);
    let domain = generic_domain();
    let bounding_box = domain.physical_box();
    for _ in 0..500 {
        let x = rng.in_box(morton::root(), &domain);
        let deepest = morton::from_physical_point(x, &bounding_box, DEEPEST);
        for level in 0..=DEEPEST {
            let leaf = morton::ancestor_at_level(deepest, level).unwrap();
            assert!(contains(leaf, x, &domain), "x = {x:?}, level {level}");
            assert_eq!(morton::from_physical_point(x, &bounding_box, level), leaf);
        }
    }
    // Points outside, on the boundary or NaN are in no leaf.
    let [x0, y0, z0, x1, _, _] = bounding_box.coordinates();
    for x in [
        [x0, y0 + 0.1, z0 + 0.1],
        [x1, y0 + 0.1, z0 + 0.1],
        [x0 - 1.0, y0, z0],
        [f64::NAN, y0, z0],
    ] {
        let leaf = morton::from_physical_point([x0 + 1e-3, y0 + 0.1, z0 + 0.1], &bounding_box, 0);
        assert!(!contains(leaf, x, &domain), "x = {x:?}");
    }
}
