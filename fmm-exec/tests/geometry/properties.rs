//! Random properties, 64 cases each: random keys on levels 0–16, random pairs and
//! triples of keys, and random points of the generic domain.

use nd_fmm_exec::geometry::{
    centre, contains, integer_centre, leaf_coordinates, radius, relative_frame,
};
use nd_octree::{MortonKey, morton};
use proptest::prelude::*;

use crate::common::{
    DEEPEST, bound32, bound64, containment_bound, dyadic_domain, dyadic_frame, error, exact_u,
    generic_domain, physical_midpoint, sides,
};

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(64)
}

/// A uniformly random level, then a uniformly random key on it.
fn key() -> impl Strategy<Value = MortonKey> {
    (0..=DEEPEST).prop_flat_map(|level| {
        let n = 1usize << level;
        [0..n, 0..n, 0..n].prop_map(move |index| morton::from_index_and_level(index, level))
    })
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn relative_frame_is_exact(s in key(), t in key()) {
        // Error measure: bit-for-bit equality with the dyadic f64 reference, f64 and f32.
        let (c, r) = dyadic_frame(s, t);
        let f64_frame = relative_frame::<f64>(s, t);
        let f32_frame = relative_frame::<f32>(s, t);
        prop_assert_eq!(f64_frame.centre, c);
        prop_assert_eq!(f64_frame.radius, r);
        prop_assert_eq!(f32_frame.centre.map(f64::from), c);
        prop_assert_eq!(f64::from(f32_frame.radius), r);
    }

    #[test]
    fn relative_frames_compose_and_reverse(s in key(), b in key(), t in key()) {
        // Error measure: exact equality, in f32 (the stricter precision).
        let (st, ts) = (relative_frame::<f32>(s, t), relative_frame::<f32>(t, s));
        let (sb, bt) = (relative_frame::<f32>(s, b), relative_frame::<f32>(b, t));
        let composed: [f32; 3] = core::array::from_fn(|k| sb.centre[k] * bt.radius + bt.centre[k]);
        prop_assert_eq!(composed, st.centre);
        prop_assert_eq!(sb.radius * bt.radius, st.radius);
        prop_assert_eq!(ts.centre, st.centre.map(|c| -c / st.radius));
        prop_assert_eq!(ts.radius, 1.0 / st.radius);
    }

    #[test]
    fn integer_centre_and_centre_agree(key in key(), finer in 0..=DEEPEST) {
        // Error measure: bit-for-bit equality in the dyadic domain.
        let domain = dyadic_domain();
        let level = morton::level(key);
        let reference = level.max(finer);
        let c = integer_centre(key, reference);
        let r = radius(reference, &domain);
        let a = domain.lower();
        let from_integer: [f64; 3] = core::array::from_fn(|k| a[k] + c[k] as f64 * r);
        prop_assert_eq!(from_integer, centre(key, &domain));
        let (mid, half) = physical_midpoint(key, &domain);
        prop_assert_eq!(centre(key, &domain), mid);
        prop_assert_eq!(half, [radius(level, &domain); 3]);
    }

    #[test]
    fn leaf_coordinates_are_bounded_and_contained(
        level in 0..=DEEPEST,
        reference in [0.0..1.0f64, 0.0..1.0f64, 0.0..1.0f64],
    ) {
        // Error measure: |ũ − u| / §3.13 bound ≤ 1 in f64 and f32, and |u| − 1 ≤ β_k
        // plus that bound.
        let domain = generic_domain();
        let (a, w) = (domain.lower(), domain.side());
        let w_k = sides(&domain.physical_box());
        let x: [f64; 3] = core::array::from_fn(|k| a[k] + reference[k] * w_k[k]);
        prop_assume!((0..3).all(|k| a[k] < x[k] && x[k] < a[k] + w_k[k]));
        let leaf = morton::from_physical_point(x, &domain.physical_box(), level);
        prop_assert!(contains(leaf, x, &domain));
        let (_, index) = morton::decode(leaf);
        let r = radius(level, &domain);
        let u64 = leaf_coordinates::<f64>(x, leaf, &domain);
        let u32 = leaf_coordinates::<f32>(x, leaf, &domain);
        for k in 0..3 {
            prop_assert_eq!(u32[k], u64[k] as f32);
            let exact = exact_u(x[k], a[k], w, level, index[k]);
            let b64 = bound64(x[k] - a[k], r, exact.0);
            let b32 = bound32(b64, exact.0);
            prop_assert!(error(u64[k], exact) <= b64);
            prop_assert!(error(u32[k] as f64, exact) <= b32);
            let beta = containment_bound(w_k[k], w, level);
            prop_assert!((exact.0 + exact.1).abs() <= 1.0 + beta);
            prop_assert!(f64::from(u32[k]).abs() <= 1.0 + beta + b32);
        }
    }
}
