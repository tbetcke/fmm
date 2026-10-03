//! f32: every operator of `LaplaceOperator<f32>` against `LaplaceOperator<f64>` on the
//! same keys, p ≤ 8, to 1e-5 (terms).
//!
//! The f32 tables are the f64 tables rounded entry by entry (CONVENTIONS §3.12). Inputs
//! are generated in f64 (coefficients, charges, and leaf-scaled points on the grid of
//! `common`, which f32 holds exactly) and rounded to f32 for the f32 operator; the f64
//! operator runs on the f64 inputs. Error measures as in the f64 tests (`common`), with
//! the terms of the f64 computation: per degree for coefficients, relative to the
//! terms of values (gradients as Euclidean norms). Levels 2, 9 and 16 of the dyadic
//! domain; M2M and L2L for every octant, M2L for every offset; P2L and M2P for one
//! X-list and one W-list pair, P2P for the self pair and every neighbour on the level,
//! with `P2pChoice::Reference` and every ISA this machine offers (f32 and f64 with the
//! same kernel).

use nd_fmm_exec::geometry::relative_frame;
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_ref::Frame;
use nd_fmm_tables::geometry::m2l_offsets;
use nd_octree::{MortonKey, morton};

use crate::common::{
    Basis, Kind, LEVELS, STRATEGIES, SplitMix64, Worst, degree_error, dense, dyadic_domain,
    evaluation_terms, grid_points, len, neighbours, norm, operator, p2p_choices, p2p_terms,
    point_terms, random_coefficients, random_pair, source_chunk, split_output, sub, target_chunk,
    terms, touches,
};

const TOL: f64 = 1e-5;

/// The degrees of the f32 test.
const DEGREES: [usize; 3] = [2, 5, 8];

fn to_f32(x: &[f64]) -> Vec<f32> {
    x.iter().map(|&v| v as f32).collect()
}

fn to_f64(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| v as f64).collect()
}

/// Runs `op` of both operators from zero outputs of length `n` and returns both.
fn both(
    n: usize,
    f32_run: impl FnOnce(&mut [f32]),
    f64_run: impl FnOnce(&mut [f64]),
) -> (Vec<f64>, Vec<f64>) {
    let mut single = vec![0.0f32; n];
    f32_run(&mut single);
    let mut double = vec![0.0f64; n];
    f64_run(&mut double);
    (to_f64(&single), double)
}

/// The errors of the potentials and gradients of `got` against `want` relative to
/// `terms` (one pair per target).
fn value_errors(got: &[f64], want: &[f64], terms: &[(f64, f64)], worst: &Worst) {
    let ((gp, gg), (wp, wg)) = (split_output(got), split_output(want));
    for j in 0..terms.len() {
        worst.check((gp[j] - wp[j]).abs() / terms[j].0, TOL, || {
            format!("target {j}")
        });
        if terms[j].1 > 0.0 {
            worst.check(norm(sub(gg[j], wg[j])) / terms[j].1, TOL, || {
                format!("target {j}, gradient")
            });
        }
    }
}

struct Pair {
    single: LaplaceOperator<f32>,
    double: LaplaceOperator<f64>,
}

fn check_translations(ops: &mut Pair, worst: &Worst, rng: &mut SplitMix64) {
    let p = ops.double.p();
    let n = len(p);
    let dense = dense(p);
    for level in LEVELS {
        for o in 0..8 {
            let parent = rng.key(level - 1);
            let child = morton::children(parent).unwrap()[o];
            let x = random_coefficients(Kind::Multipole, p, rng);
            let (got, want) = both(
                n,
                |y| ops.single.m2m_pair(child, parent, &to_f32(&x), y),
                |y| ops.double.m2m_pair(child, parent, &x, y),
            );
            let e = degree_error(Kind::Multipole, p, &got, &want, &terms(&dense.m2m, o, &x));
            worst.check(e, TOL, || format!("M2M p = {p}, level {level}, octant {o}"));

            let x = random_coefficients(Kind::Local, p, rng);
            let (got, want) = both(
                n,
                |y| ops.single.l2l_pair(parent, child, &to_f32(&x), y),
                |y| ops.double.l2l_pair(parent, child, &x, y),
            );
            let e = degree_error(Kind::Local, p, &got, &want, &terms(&dense.l2l, o, &x));
            worst.check(e, TOL, || format!("L2L p = {p}, level {level}, octant {o}"));
        }
        for (index, d) in m2l_offsets().into_iter().enumerate() {
            let (source, target) = random_pair(level, d, rng);
            let x = random_coefficients(Kind::Multipole, p, rng);
            let (got, want) = both(
                n,
                |y| ops.single.m2l_pair(source, target, &to_f32(&x), y),
                |y| ops.double.m2l_pair(source, target, &x, y),
            );
            let e = degree_error(Kind::Local, p, &got, &want, &terms(&dense.m2l, index, &x));
            worst.check(e, TOL, || format!("M2L p = {p}, level {level}, d = {d:?}"));
        }
    }
}

/// An X-list source of `target`, if it has one.
fn x_source(target: MortonKey) -> Option<MortonKey> {
    let parent = morton::parent(target)?;
    neighbours(parent)
        .into_iter()
        .find(|&s| !touches(s, target))
}

fn check_leaf_operators(ops: &mut Pair, worst: &Worst, rng: &mut SplitMix64) {
    let p = ops.double.p();
    let domain = dyadic_domain();
    let unit = Frame::new([0.0; 3], 1.0);
    const POINTS: usize = 5;
    for level in LEVELS {
        let target = rng.interior_key(level);
        let (_, u_t) = grid_points(target, &domain, POINTS, rng);
        let targets = target_chunk(&u_t);
        let targets32 = to_f32(&targets);

        // P2M, then L2P of a random local, at the target leaf.
        let q = rng.charges(POINTS);
        let chunk = source_chunk(&u_t, &q);
        let (got, want) = both(
            len(p),
            |y| ops.single.p2m_leaf(&to_f32(&chunk), y),
            |y| ops.double.p2m_leaf(&chunk, y),
        );
        let scale = point_terms(Basis::Regular, p, &unit, &u_t, &q);
        let e = degree_error(Kind::Multipole, p, &got, &want, &scale);
        worst.check(e, TOL, || format!("P2M p = {p}, level {level}"));

        let local = random_coefficients(Kind::Local, p, rng);
        let (got, want) = both(
            4 * POINTS,
            |y| ops.single.l2p_leaf(&to_f32(&local), &targets32, y),
            |y| ops.double.l2p_leaf(&local, &targets, y),
        );
        let scale: Vec<_> = u_t
            .iter()
            .map(|&u| evaluation_terms(Basis::Regular, p, &unit, &local, u))
            .collect();
        value_errors(&got, &want, &scale, worst);

        // P2L from an X-list source.
        let mut x_target = rng.key(level);
        let x = loop {
            if let Some(s) = x_source(x_target) {
                break s;
            }
            x_target = rng.key(level);
        };
        let (_, u_s) = grid_points(x, &domain, POINTS, rng);
        let chunk = source_chunk(&u_s, &q);
        let (got, want) = both(
            len(p),
            |y| ops.single.p2l_pair(x, x_target, &to_f32(&chunk), y),
            |y| ops.double.p2l_pair(x, x_target, &chunk, y),
        );
        let frame = relative_frame::<f64>(x_target, x);
        let scale = point_terms(Basis::Irregular, p, &frame, &u_s, &q);
        let e = degree_error(Kind::Local, p, &got, &want, &scale);
        worst.check(e, TOL, || format!("P2L p = {p}, level {level}"));

        // M2P from a W-list source (below level 16).
        if level < 16 {
            let w = neighbours(target)
                .into_iter()
                .flat_map(|n| morton::children(n).unwrap())
                .find(|&s| !touches(s, target))
                .unwrap();
            let multipole = random_coefficients(Kind::Multipole, p, rng);
            let (got, want) = both(
                4 * POINTS,
                |y| {
                    ops.single
                        .m2p_pair(w, target, &to_f32(&multipole), &targets32, y)
                },
                |y| ops.double.m2p_pair(w, target, &multipole, &targets, y),
            );
            let frame = relative_frame::<f64>(w, target);
            let scale: Vec<_> = u_t
                .iter()
                .map(|&u| evaluation_terms(Basis::Irregular, p, &frame, &multipole, u))
                .collect();
            value_errors(&got, &want, &scale, worst);
        }
    }
}

/// P2P from the leaf itself and from each neighbour on its level, levels 2, 9 and 16.
fn check_p2p(ops: &mut Pair, worst: &Worst, rng: &mut SplitMix64) {
    let domain = dyadic_domain();
    const POINTS: usize = 5;
    for level in LEVELS {
        let target = rng.interior_key(level);
        let (_, u_t) = grid_points(target, &domain, POINTS, rng);
        let targets = target_chunk(&u_t);
        let targets32 = to_f32(&targets);
        let q = rng.charges(POINTS);
        for source in neighbours(target).into_iter().chain([target]) {
            let (_, u_s) = if source == target {
                (Vec::new(), u_t.clone())
            } else {
                grid_points(source, &domain, POINTS, rng)
            };
            let chunk = source_chunk(&u_s, &q);
            let (got, want) = both(
                4 * POINTS,
                |y| {
                    ops.single
                        .p2p_pair(source, target, &to_f32(&chunk), &targets32, y)
                },
                |y| ops.double.p2p_pair(source, target, &chunk, &targets, y),
            );
            let frame = relative_frame::<f64>(source, target);
            let mapped: Vec<[f64; 3]> = u_s
                .iter()
                .map(|u| core::array::from_fn(|k| frame.centre[k] + frame.radius * u[k]))
                .collect();
            let scale: Vec<_> = u_t.iter().map(|&x| p2p_terms(&mapped, &q, x)).collect();
            value_errors(&got, &want, &scale, worst);
        }
    }
}

#[test]
fn every_operator_in_f32_matches_f64() {
    let mut rng = SplitMix64::new(0x7871);
    let translations = Worst::new("f32 vs f64, M2M, L2L, M2L, p ∈ {2, 5, 8} (terms)");
    let leaves = Worst::new("f32 vs f64, P2M, P2L, L2P, M2P, P2P, p ∈ {2, 5, 8} (terms)");
    let choices = p2p_choices("f32 vs f64, P2P");
    for p in DEGREES {
        for strategy in STRATEGIES {
            let mut ops = Pair {
                single: LaplaceOperator::new(Tables::<f32>::build(p, strategy), true, 8),
                double: operator(strategy, p, 8),
            };
            check_translations(&mut ops, &translations, &mut rng);
            if strategy == M2lStrategy::Rotation {
                check_leaf_operators(&mut ops, &leaves, &mut rng);
                // P2P with every kernel, f32 against f64 with the same kernel.
                for &choice in &choices {
                    let mut ops = Pair {
                        single: ops.single.clone().with_p2p(choice).unwrap(),
                        double: ops.double.clone().with_p2p(choice).unwrap(),
                    };
                    check_p2p(&mut ops, &leaves, &mut rng);
                }
            }
        }
    }
}
