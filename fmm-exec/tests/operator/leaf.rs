//! Leaf operators and P2P, the C3.1 criterion for the rest: P2M, P2L, L2P, M2P and P2P
//! of `LaplaceOperator` against `nd_fmm_ref::{leaf, p2p}` at the absolute frames of
//! CONVENTIONS §3.12, with the absolute coordinates the points were generated in, in
//! the dyadic domain, p ≤ 12, potentials and gradients.
//!
//! Points lie on the dyadic grid of `common`, so both sides see the same geometry
//! exactly. Reference potentials and gradients (of 1/|x − y|, no 4π) are converted to
//! the leaf-scaled units of §3.13, φ̂ = r_t φ and ĝ = r_t² ∇φ. Error measures (`common`):
//! coefficients per degree relative to their terms (P2M, P2L); values relative to the
//! magnitudes of their terms, the gradient as a Euclidean norm (L2P, M2P, P2P).
//! Tolerance 1e-13.
//!
//! Geometry:
//! - P2M and L2P at random leaves on levels 2, 9 and 16.
//! - P2L, X-list geometry: a random target box t on levels 2, 9 and 16 and every source
//!   leaf s on the level of its parent that touches the parent but not t.
//! - M2P, W-list geometry: a random target leaf t on levels 2, 9 and 15 and, for every
//!   neighbour of t on its level, every child s of that neighbour (level l + 1) that
//!   does not touch t; so the parent of s touches t and s does not.
//! - P2P: a random interior leaf t on levels 2, 9 and 16 with itself (the targets are
//!   the sources plus as many other points, so coincident pairs are excluded), with
//!   each of its 26 neighbours on its level, with every leaf one level coarser that
//!   touches it, and (below level 16) with every leaf one level finer that touches it.

use nd_fmm_ref::{Frame, Workspace, leaf, p2p};
use nd_octree::{MortonKey, morton};

use crate::common::{
    Basis, Kind, LEAF_TOL, LEVELS, SplitMix64, Worst, absolute_frame, degree_error, dyadic_domain,
    evaluation_terms, grid_points, key_radius, len, neighbours, norm, operator, p2p_terms,
    point_terms, random_coefficients, source_chunk, split_output, sub, target_chunk, touches,
};
use nd_fmm_exec::geometry::relative_frame;
use nd_fmm_exec::tables::M2lStrategy;

/// The degrees of the leaf tests.
const DEGREES: [usize; 5] = [0, 1, 4, 8, 12];

/// Points per box.
const POINTS: usize = 6;

/// The worst value and gradient errors of an evaluation.
struct Evaluation {
    potential: Worst,
    gradient: Worst,
}

impl Evaluation {
    fn new(name: &str) -> Self {
        Self {
            potential: Worst::new(format!("{name}, potential (terms)")),
            gradient: Worst::new(format!("{name}, gradient (terms)")),
        }
    }

    /// Compares the operator's output chunk with reference φ, ∇φ of 1/|x − y| at the
    /// targets, converted with r_t, relative to the given terms in leaf units.
    fn check(
        &self,
        output: &[f64],
        (phi, grad): (&[f64], &[[f64; 3]]),
        r_t: f64,
        terms: &[(f64, f64)],
        context: impl Fn() -> String,
    ) {
        let (potential, gradient) = split_output(output);
        for j in 0..phi.len() {
            let e = relative((potential[j] - r_t * phi[j]).abs(), terms[j].0);
            self.potential.check(e, LEAF_TOL, &context);
            let want = grad[j].map(|g| r_t * r_t * g);
            let e = relative(norm(sub(gradient[j], want)), terms[j].1);
            self.gradient.check(e, LEAF_TOL, &context);
        }
    }
}

/// The error `error` relative to `terms`; zero terms (the gradient at p = 0) require an
/// exact zero.
fn relative(error: f64, terms: f64) -> f64 {
    if terms > 0.0 {
        error / terms
    } else {
        assert_eq!(error, 0.0, "nonzero error with zero terms");
        0.0
    }
}

/// Runs the reference `leaf` evaluation `evaluate` at `targets`, from zero.
fn reference_values(
    targets: &[[f64; 3]],
    evaluate: impl FnOnce(&mut [f64], &mut [[f64; 3]]),
) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut phi = vec![0.0; targets.len()];
    let mut grad = vec![[0.0; 3]; targets.len()];
    evaluate(&mut phi, &mut grad);
    (phi, grad)
}

#[test]
fn p2m_and_l2p_equal_the_reference_on_levels_2_9_16() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7811);
    let p2m = Worst::new("P2M vs leaf::p2m, levels 2, 9, 16, p ≤ 12, per degree (terms)");
    let l2p = Evaluation::new("L2P vs leaf::l2p, levels 2, 9, 16, p ≤ 12");
    for p in DEGREES {
        // The leaf operators use no table; the rotation tables are the cheapest to build.
        let mut op = operator(M2lStrategy::Rotation, p, 0);
        let mut ws = Workspace::new(p);
        for level in LEVELS {
            for _ in 0..4 {
                let key = rng.key(level);
                let frame = absolute_frame(key, &domain);
                let context = || format!("p = {p}, level {level}, key {key}");

                let (x, u) = grid_points(key, &domain, POINTS, &mut rng);
                let q = rng.charges(POINTS);
                let mut got = vec![0.0; len(p)];
                op.p2m_leaf(&source_chunk(&u, &q), &mut got);
                let mut want = vec![0.0; len(p)];
                leaf::p2m(p, &frame, &x, &q, &mut ws, &mut want);
                let scale = point_terms(Basis::Regular, p, &frame, &x, &q);
                p2m.check(
                    degree_error(Kind::Multipole, p, &got, &want, &scale),
                    LEAF_TOL,
                    context,
                );

                let local = random_coefficients(Kind::Local, p, &mut rng);
                let mut output = vec![0.0; 4 * POINTS];
                op.l2p_leaf(&local, &target_chunk(&u), &mut output);
                let (phi, grad) = reference_values(&x, |phi, grad| {
                    leaf::l2p(p, &frame, &local, &x, &mut ws, phi, Some(grad))
                });
                let unit = Frame::new([0.0; 3], 1.0);
                let terms: Vec<_> = u
                    .iter()
                    .map(|&u| evaluation_terms(Basis::Regular, p, &unit, &local, u))
                    .collect();
                l2p.check(&output, (&phi, &grad), frame.radius, &terms, context);
            }
        }
    }
}

/// The X-list sources of `target`: the leaves on its parent's level that touch the
/// parent but not `target`.
fn x_sources(target: MortonKey) -> Vec<MortonKey> {
    let parent = morton::parent(target).unwrap();
    neighbours(parent)
        .into_iter()
        .filter(|&s| !touches(s, target))
        .collect()
}

#[test]
fn p2l_equals_the_reference_for_x_list_geometry() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7812);
    let worst = Worst::new("P2L vs leaf::p2l, X-list geometry, levels 2, 9, 16, p ≤ 12 (terms)");
    let mut pairs = 0;
    for p in DEGREES {
        let mut op = operator(M2lStrategy::Rotation, p, 0);
        let mut ws = Workspace::new(p);
        for level in LEVELS {
            let mut targets = 0;
            while targets < 2 {
                let target = rng.key(level);
                let sources = x_sources(target);
                if sources.is_empty() {
                    continue;
                }
                targets += 1;
                let frame = absolute_frame(target, &domain);
                for source in sources {
                    assert!(touches(source, morton::parent(target).unwrap()));
                    let (y, u) = grid_points(source, &domain, POINTS, &mut rng);
                    let q = rng.charges(POINTS);
                    let mut got = vec![0.0; len(p)];
                    op.p2l_pair(source, target, &source_chunk(&u, &q), &mut got);
                    let mut want = vec![0.0; len(p)];
                    leaf::p2l(p, &frame, &y, &q, &mut ws, &mut want);
                    let scale = point_terms(Basis::Irregular, p, &frame, &y, &q);
                    worst.check(
                        degree_error(Kind::Local, p, &got, &want, &scale),
                        LEAF_TOL,
                        || format!("p = {p}, source {source}, target {target}"),
                    );
                    pairs += 1;
                }
            }
        }
    }
    eprintln!("P2L: {pairs} X-list pairs");
}

/// The W-list sources of `target`: the children of its neighbours that do not touch it.
fn w_sources(target: MortonKey) -> Vec<MortonKey> {
    neighbours(target)
        .into_iter()
        .flat_map(|n| morton::children(n).unwrap())
        .filter(|&s| !touches(s, target))
        .collect()
}

#[test]
fn m2p_equals_the_reference_for_w_list_geometry() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7813);
    let worst = Evaluation::new("M2P vs leaf::m2p, W-list geometry, levels 2, 9, 15, p ≤ 12");
    let mut pairs = 0;
    for p in DEGREES {
        let mut op = operator(M2lStrategy::Rotation, p, 0);
        let mut ws = Workspace::new(p);
        for level in [2, 9, 15] {
            let target = rng.key(level);
            let (x, u) = grid_points(target, &domain, POINTS, &mut rng);
            let r_t = key_radius(target, &domain);
            for source in w_sources(target) {
                assert!(touches(morton::parent(source).unwrap(), target));
                let frame = absolute_frame(source, &domain);
                let multipole = random_coefficients(Kind::Multipole, p, &mut rng);
                let mut output = vec![0.0; 4 * POINTS];
                op.m2p_pair(source, target, &multipole, &target_chunk(&u), &mut output);
                let (phi, grad) = reference_values(&x, |phi, grad| {
                    leaf::m2p(p, &frame, &multipole, &x, &mut ws, phi, Some(grad))
                });
                let relative = relative_frame::<f64>(source, target);
                let terms: Vec<_> = u
                    .iter()
                    .map(|&u| evaluation_terms(Basis::Irregular, p, &relative, &multipole, u))
                    .collect();
                worst.check(&output, (&phi, &grad), r_t, &terms, || {
                    format!("p = {p}, source {source}, target {target}")
                });
                pairs += 1;
            }
        }
    }
    eprintln!("M2P: {pairs} W-list pairs");
}

/// The P2P sources of the interior leaf `target` of the test: itself, its 26
/// neighbours, the touching leaves one level coarser and, below level 16, one level
/// finer.
fn near_sources(target: MortonKey) -> Vec<MortonKey> {
    let level = morton::level(target);
    let mut sources = vec![target];
    let same = neighbours(target);
    assert_eq!(same.len(), 26);
    sources.extend(&same);
    let parent = morton::parent(target).unwrap();
    sources.extend(
        neighbours(parent)
            .into_iter()
            .filter(|&s| touches(s, target)),
    );
    if level < 16 {
        sources.extend(
            same.iter()
                .flat_map(|&n| morton::children(n).unwrap())
                .filter(|&s| touches(s, target)),
        );
    }
    sources
}

#[test]
fn p2p_equals_the_reference_for_near_pairs() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7814);
    let worst = Evaluation::new("P2P vs p2p::p2p, self, 26 neighbours, coarser, finer");
    let mut op = operator(M2lStrategy::Rotation, 0, POINTS);
    let mut counts = [0usize; 4];
    for level in LEVELS {
        let target = rng.interior_key(level);
        let r_t = key_radius(target, &domain);
        for source in near_sources(target) {
            let (y, u_s) = grid_points(source, &domain, POINTS, &mut rng);
            let q = rng.charges(POINTS);
            // For the self pair the targets are the sources and as many other points.
            let (x, u_t) = if source == target {
                let (mut x, mut u) = grid_points(target, &domain, POINTS, &mut rng);
                x.splice(0..0, y.iter().copied());
                u.splice(0..0, u_s.iter().copied());
                (x, u)
            } else {
                grid_points(target, &domain, POINTS, &mut rng)
            };
            let mut output = vec![0.0; 4 * x.len()];
            op.p2p_pair(
                source,
                target,
                &source_chunk(&u_s, &q),
                &target_chunk(&u_t),
                &mut output,
            );
            let (phi, grad) =
                reference_values(&x, |phi, grad| p2p::p2p(&y, &q, &x, phi, Some(grad)));
            let terms: Vec<_> = x
                .iter()
                .map(|&x| {
                    let (phi, grad) = p2p_terms(&y, &q, x);
                    (r_t * phi, r_t * r_t * grad)
                })
                .collect();
            worst.check(&output, (&phi, &grad), r_t, &terms, || {
                format!("level {level}, source {source}, target {target}")
            });
            let kind = match morton::level(source) {
                l if source == target => {
                    assert_eq!(l, level);
                    0
                }
                l if l == level => 1,
                l if l < level => 2,
                _ => 3,
            };
            counts[kind] += 1;
        }
    }
    eprintln!(
        "P2P pairs: {} self, {} same level, {} coarser, {} finer",
        counts[0], counts[1], counts[2], counts[3]
    );
    assert_eq!(counts[0], 3);
    assert_eq!(counts[1], 3 * 26);
    assert!(counts[2] > 0 && counts[3] > 0);
}
