//! Random properties, 64 cases each: random levels, keys, octants, offsets and grid
//! points, against `nd-fmm-ref` at the absolute frames of the dyadic domain (the
//! measures and tolerances of the deterministic tests, `common`).

use std::sync::Mutex;

use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_ref::{Workspace, direct, leaf, p2p};
use nd_fmm_tables::geometry::m2l_offsets;
use nd_octree::{MortonKey, morton};
use proptest::prelude::*;

use crate::common::{
    Basis, Kind, LEAF_TOL, SplitMix64, absolute_frame, degree_error, dense, dyadic_domain,
    grid_points, key_radius, len, norm, operator, p2p_terms, point_terms, random_coefficients,
    source_chunk, split_output, sub, target_chunk, touches, translation_tolerance,
};

const P: usize = 4;

fn config() -> ProptestConfig {
    ProptestConfig::with_cases(64)
}

/// The key on `level` at the fractional position `t` ∈ [0, 1)³ of the domain.
fn key_at(level: usize, t: [f64; 3]) -> MortonKey {
    let n = 1usize << level;
    morton::from_index_and_level(t.map(|t| ((t * n as f64) as usize).min(n - 1)), level)
}

/// Records the error `e` of the property `name` and prints each new worst value, so
/// the last line printed for a property is its worst error.
fn record(name: &'static str, e: f64) -> f64 {
    static WORST: Mutex<Vec<(&str, f64)>> = Mutex::new(Vec::new());
    let mut worst = WORST.lock().unwrap();
    let position = match worst.iter().position(|(n, _)| *n == name) {
        Some(position) => position,
        None => {
            worst.push((name, -1.0));
            worst.len() - 1
        }
    };
    if e > worst[position].1 {
        worst[position].1 = e;
        eprintln!("worst so far, {name}: {e:.3e}");
    }
    e
}

fn position() -> impl Strategy<Value = [f64; 3]> {
    [0.0..1.0f64, 0.0..1.0f64, 0.0..1.0f64]
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn m2l_equals_direct_for_random_v_list_pairs(
        level in 2usize..=16,
        offset in 0usize..316,
        t in position(),
        strategy in prop::sample::select(vec![
            M2lStrategy::Dense, M2lStrategy::Classes, M2lStrategy::Rotation,
        ]),
        seed: u64,
    ) {
        // Error measure: per degree relative to the dense terms; 1e-14 dense, 1e-13
        // otherwise. The source is moved into the domain along each axis that the
        // offset would leave.
        let d = m2l_offsets()[offset];
        let n = 1i64 << level;
        let (_, index) = morton::decode(key_at(level, t));
        let source_index: [usize; 3] = core::array::from_fn(|k| {
            (index[k] as i64).clamp((-d[k]).max(0), (n - 1 - d[k]).min(n - 1)) as usize
        });
        let target_index: [usize; 3] = core::array::from_fn(|k| (source_index[k] as i64 + d[k]) as usize);
        let source = morton::from_index_and_level(source_index, level);
        let target = morton::from_index_and_level(target_index, level);
        let domain = dyadic_domain();
        let mut rng = SplitMix64::new(seed);
        let x = random_coefficients(Kind::Multipole, P, &mut rng);
        let mut op = operator(strategy, P, 0);
        let mut got = vec![0.0; len(P)];
        op.m2l_pair(source, target, &x, &mut got);
        let mut want = vec![0.0; len(P)];
        let (from, to) = (absolute_frame(source, &domain), absolute_frame(target, &domain));
        direct::m2l(P, &from, &to, &mut Workspace::new(P), &x, &mut want);
        let e = degree_error(Kind::Local, P, &got, &want, &crate::common::terms(&dense(P).m2l, offset, &x));
        record("proptest M2L vs direct, p = 4 (terms)", e);
        prop_assert!(e <= translation_tolerance(strategy), "{strategy:?}, d = {d:?}: {e:.3e}");
    }

    #[test]
    fn m2m_and_l2l_equal_direct_for_random_children(
        level in 1usize..=16,
        t in position(),
        octant in 0usize..8,
        seed: u64,
    ) {
        // Error measure: per degree relative to the dense terms; 1e-13 (rotation).
        let parent = key_at(level - 1, t);
        let child = morton::children(parent).unwrap()[octant];
        let domain = dyadic_domain();
        let (c, pa) = (absolute_frame(child, &domain), absolute_frame(parent, &domain));
        let mut rng = SplitMix64::new(seed);
        let mut op = operator(M2lStrategy::Rotation, P, 0);
        let mut ws = Workspace::new(P);
        let x = random_coefficients(Kind::Multipole, P, &mut rng);
        let (mut got, mut want) = (vec![0.0; len(P)], vec![0.0; len(P)]);
        op.m2m_pair(child, parent, &x, &mut got);
        direct::m2m(P, &c, &pa, &mut ws, &x, &mut want);
        let scale = crate::common::terms(&dense(P).m2m, octant, &x);
        let e = record(
            "proptest M2M (rotation) vs direct, p = 4 (terms)",
            degree_error(Kind::Multipole, P, &got, &want, &scale),
        );
        prop_assert!(e <= 1e-13, "M2M: {e:.3e}");
        let x = random_coefficients(Kind::Local, P, &mut rng);
        let (mut got, mut want) = (vec![0.0; len(P)], vec![0.0; len(P)]);
        op.l2l_pair(parent, child, &x, &mut got);
        direct::l2l(P, &pa, &c, &mut ws, &x, &mut want);
        let scale = crate::common::terms(&dense(P).l2l, octant, &x);
        let e = record(
            "proptest L2L (rotation) vs direct, p = 4 (terms)",
            degree_error(Kind::Local, P, &got, &want, &scale),
        );
        prop_assert!(e <= 1e-13, "L2L: {e:.3e}");
    }

    #[test]
    fn p2p_equals_the_reference_for_random_touching_leaves(
        level in 1usize..=16,
        t in position(),
        direction in [-1i64..=1, -1i64..=1, -1i64..=1],
        relative_level in -1i64..=1,
        seed: u64,
    ) {
        // Error measure: potential and gradient relative to their terms; 1e-13. The
        // source is the box in `direction` from the target, on the target's level or
        // one level coarser or finer (a touching box that is not an ancestor or
        // descendant of the target), if it lies in the domain.
        let target = key_at(level, t);
        let (_, index) = morton::decode(target);
        let n = 1i64 << level;
        let neighbour: Option<[usize; 3]> = (0..3)
            .map(|k| {
                let i = index[k] as i64 + direction[k];
                (0..n).contains(&i).then_some(i as usize)
            })
            .collect::<Option<Vec<_>>>()
            .map(|v| [v[0], v[1], v[2]]);
        prop_assume!(neighbour.is_some());
        let neighbour = morton::from_index_and_level(neighbour.unwrap(), level);
        let source = match relative_level {
            -1 => morton::parent(neighbour).unwrap(),
            0 => neighbour,
            _ if level < 16 => morton::children(neighbour).unwrap()[(seed % 8) as usize],
            _ => neighbour,
        };
        prop_assume!(
            source == target
                || (touches(source, target)
                    && !morton::is_ancestor(source, target)
                    && !morton::is_ancestor(target, source))
        );
        let domain = dyadic_domain();
        let mut rng = SplitMix64::new(seed);
        let (y, u_s) = grid_points(source, &domain, 5, &mut rng);
        let (x, u_t) = if source == target { (y.clone(), u_s.clone()) } else { grid_points(target, &domain, 5, &mut rng) };
        let q = rng.charges(5);
        let mut op = operator(M2lStrategy::Rotation, 0, 5);
        let mut output = vec![0.0; 4 * x.len()];
        op.p2p_pair(source, target, &source_chunk(&u_s, &q), &target_chunk(&u_t), &mut output);
        let mut phi = vec![0.0; x.len()];
        let mut grad = vec![[0.0; 3]; x.len()];
        p2p::p2p(&y, &q, &x, &mut phi, Some(&mut grad));
        let r = key_radius(target, &domain);
        let (potential, gradient) = split_output(&output);
        for j in 0..x.len() {
            let (tp, tg) = p2p_terms(&y, &q, x[j]);
            let e = record(
                "proptest P2P vs p2p::p2p, potential (terms)",
                (potential[j] - r * phi[j]).abs() / (r * tp),
            );
            prop_assert!(e <= LEAF_TOL, "potential: {e:.3e}");
            let want = grad[j].map(|g| r * r * g);
            let e = record(
                "proptest P2P vs p2p::p2p, gradient (terms)",
                norm(sub(gradient[j], want)) / (r * r * tg),
            );
            prop_assert!(e <= LEAF_TOL, "gradient: {e:.3e}");
        }
    }

    #[test]
    fn p2m_equals_the_reference_for_random_leaves(
        level in 0usize..=16,
        t in position(),
        seed: u64,
    ) {
        // Error measure: per degree relative to the terms of P2M; 1e-13.
        let key = key_at(level, t);
        let domain = dyadic_domain();
        let frame = absolute_frame(key, &domain);
        let mut rng = SplitMix64::new(seed);
        let (x, u) = grid_points(key, &domain, 7, &mut rng);
        let q = rng.charges(7);
        let mut op = operator(M2lStrategy::Rotation, P, 0);
        let mut got = vec![0.0; len(P)];
        op.p2m_leaf(&source_chunk(&u, &q), &mut got);
        let mut want = vec![0.0; len(P)];
        leaf::p2m(P, &frame, &x, &q, &mut Workspace::new(P), &mut want);
        let scale = point_terms(Basis::Regular, P, &frame, &x, &q);
        let e = record(
            "proptest P2M vs leaf::p2m, p = 4 (terms)",
            degree_error(Kind::Multipole, P, &got, &want, &scale),
        );
        prop_assert!(e <= LEAF_TOL, "{e:.3e}");
    }
}
