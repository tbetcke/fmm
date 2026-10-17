//! Dipole sources (Phase 6 T8, C6.4): P2M, P2L and P2P of `LaplaceOperator` built
//! `with_sources(Dipoles)` and `with_sources(ChargesAndDipoles)` against
//! `nd_fmm_ref::{leaf, p2p}`'s dipole operators at the absolute frames of CONVENTIONS
//! §3.12, the C3.1 criterion for the new source types, in the dyadic domain, p ≤ 12.
//!
//! The chunks hold the moments d̂ in units of the source leaf (CONVENTIONS §3.13, "Source
//! chunks"), on the grid of `common`; the reference gets d = r_s d̂ in absolute units. With
//! charges and dipoles the reference is the charge operator plus the dipole operator.
//! Error measures (as `leaf.rs`): coefficients per degree relative to their terms
//! Σⱼ Σₖ |d̂ⱼ,ₖ ∂ₖ Xᵢ| (+ Σⱼ |qⱼ Xᵢ| with charges); P2P potentials and gradients relative to
//! Σⱼ |d̂ⱼ| / r² and Σⱼ 2 |d̂ⱼ| / r³ (+ the charge terms) in leaf units. Tolerance 1e-13, for
//! every P2P kernel the machine offers (`Reference` and every ISA).

use nd_fmm_exec::operator::{LaplaceOperator, SourceType};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::{Frame, Workspace, leaf, p2p};
use nd_octree::{MortonKey, morton};

use crate::common::Basis;
use crate::common::{
    Kind, LEAF_TOL, LEVELS, SplitMix64, Worst, absolute_frame, degree_error, dyadic_domain,
    grid_points, key_radius, len, neighbours, norm, p2p_choices, p2p_terms, point_terms, sub,
    tables, target_chunk, touches,
};

/// The degrees of the expansion tests.
const DEGREES: [usize; 4] = [1, 4, 8, 12];

/// Points per box.
const POINTS: usize = 5;

/// The two source types with dipoles.
const TYPES: [SourceType; 2] = [SourceType::Dipoles, SourceType::ChargesAndDipoles];

/// A source chunk of `sources` (CONVENTIONS §3.13): the coordinates, the charges if the
/// type has them, then the moments.
fn chunk(sources: SourceType, u: &[[f64; 3]], q: &[f64], d: &[[f64; 3]]) -> Vec<f64> {
    let mut out: Vec<f64> = u.iter().flatten().copied().collect();
    if sources.has_charges() {
        out.extend_from_slice(q);
    }
    out.extend(d.iter().flatten());
    out
}

/// The operator of `sources` at degree p with the rotation tables (the cheapest; the leaf
/// operators use none).
fn operator(sources: SourceType, p: usize, max_leaf_points: usize) -> LaplaceOperator<f64> {
    LaplaceOperator::new(
        tables(M2lStrategy::Rotation, p).clone(),
        true,
        max_leaf_points,
    )
    .with_sources(sources)
}

/// `count` moments on the grid of `common`.
fn moments(rng: &mut SplitMix64, count: usize) -> Vec<[f64; 3]> {
    (0..count).map(|_| [0; 3].map(|_| rng.grid())).collect()
}

/// The coefficient terms Σⱼ Σₖ |dⱼ,ₖ / R ∂ₖ Xᵢ(frame.scaled(yⱼ))| of a dipole P2M
/// (regular) or P2L (irregular), per slot.
fn dipole_terms(
    basis: Basis,
    p: usize,
    frame: &Frame<f64>,
    points: &[[f64; 3]],
    dipoles: &[[f64; 3]],
) -> Vec<f64> {
    let mut out = vec![0.0; len(p)];
    let mut values = vec![0.0; len(p)];
    let mut grad = [0; 3].map(|_| vec![0.0; len(p)]);
    for (&y, d) in points.iter().zip(dipoles) {
        let [gx, gy, gz] = &mut grad;
        let components = [&mut gx[..], &mut gy[..], &mut gz[..]];
        match basis {
            Basis::Regular => harmonics::regular_grad(p, frame.scaled(y), &mut values, components),
            Basis::Irregular => {
                harmonics::irregular_grad(p, frame.scaled(y), &mut values, components)
            }
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o += (0..3)
                .map(|k| (d[k] / frame.radius * grad[k][i]).abs())
                .sum::<f64>();
        }
    }
    out
}

#[test]
fn dipole_p2m_equals_the_reference_on_levels_2_9_16() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0xd1b1);
    let worst = Worst::new("dipole P2M vs leaf::p2m_dipoles, levels 2, 9, 16, p ≤ 12 (terms)");
    for sources in TYPES {
        for p in DEGREES {
            let mut op = operator(sources, p, 0);
            assert_eq!(op.sources(), sources);
            let mut ws = Workspace::new(p);
            for level in LEVELS {
                for _ in 0..3 {
                    let key = rng.key(level);
                    let frame = absolute_frame(key, &domain);
                    let (x, u) = grid_points(key, &domain, POINTS, &mut rng);
                    let q = rng.charges(POINTS);
                    let d_hat = moments(&mut rng, POINTS);
                    let d: Vec<[f64; 3]> =
                        d_hat.iter().map(|d| d.map(|c| c * frame.radius)).collect();
                    let mut got = vec![0.0; len(p)];
                    op.p2m_leaf(&chunk(sources, &u, &q, &d_hat), &mut got);
                    let mut want = vec![0.0; len(p)];
                    let mut scale = dipole_terms(Basis::Regular, p, &frame, &x, &d);
                    if sources.has_charges() {
                        leaf::p2m(p, &frame, &x, &q, &mut ws, &mut want);
                        let charge = point_terms(Basis::Regular, p, &frame, &x, &q);
                        scale.iter_mut().zip(charge).for_each(|(s, c)| *s += c);
                    }
                    leaf::p2m_dipoles(p, &frame, &x, &d, &mut ws, &mut want);
                    worst.check(
                        degree_error(Kind::Multipole, p, &got, &want, &scale),
                        LEAF_TOL,
                        || format!("{sources}, p = {p}, level {level}, key {key}"),
                    );
                }
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
fn dipole_p2l_equals_the_reference_for_x_list_geometry() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0xd1b2);
    let worst =
        Worst::new("dipole P2L vs leaf::p2l_dipoles, X-list geometry, levels 2, 9, 16 (terms)");
    for sources in TYPES {
        for p in DEGREES {
            let mut op = operator(sources, p, 0);
            let mut ws = Workspace::new(p);
            for level in LEVELS {
                let target = loop {
                    let target = rng.key(level);
                    if !x_sources(target).is_empty() {
                        break target;
                    }
                };
                let frame = absolute_frame(target, &domain);
                for source in x_sources(target) {
                    let (y, u) = grid_points(source, &domain, POINTS, &mut rng);
                    let r_s = key_radius(source, &domain);
                    let q = rng.charges(POINTS);
                    let d_hat = moments(&mut rng, POINTS);
                    let d: Vec<[f64; 3]> = d_hat.iter().map(|d| d.map(|c| c * r_s)).collect();
                    let mut got = vec![0.0; len(p)];
                    op.p2l_pair(source, target, &chunk(sources, &u, &q, &d_hat), &mut got);
                    let mut want = vec![0.0; len(p)];
                    let mut scale = dipole_terms(Basis::Irregular, p, &frame, &y, &d);
                    if sources.has_charges() {
                        leaf::p2l(p, &frame, &y, &q, &mut ws, &mut want);
                        let charge = point_terms(Basis::Irregular, p, &frame, &y, &q);
                        scale.iter_mut().zip(charge).for_each(|(s, c)| *s += c);
                    }
                    leaf::p2l_dipoles(p, &frame, &y, &d, &mut ws, &mut want);
                    worst.check(
                        degree_error(Kind::Local, p, &got, &want, &scale),
                        LEAF_TOL,
                        || format!("{sources}, p = {p}, source {source}, target {target}"),
                    );
                }
            }
        }
    }
}

/// The P2P sources of the interior leaf `target`: itself, its 26 neighbours, the touching
/// leaves one level coarser and, below level 16, one level finer.
fn near_sources(target: MortonKey) -> Vec<MortonKey> {
    let level = morton::level(target);
    let mut sources = vec![target];
    let same = neighbours(target);
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

/// Σⱼ |dⱼ| / r² and Σⱼ 2 |dⱼ| / r³ at `x`, coincident points excluded.
fn dipole_p2p_terms(sources: &[[f64; 3]], dipoles: &[[f64; 3]], x: [f64; 3]) -> (f64, f64) {
    let mut terms = (0.0, 0.0);
    for (y, d) in sources.iter().zip(dipoles) {
        if *y == x {
            continue;
        }
        let r = norm(sub(x, *y));
        terms.0 += norm(*d) / (r * r);
        terms.1 += 2.0 * norm(*d) / (r * r * r);
    }
    terms
}

#[test]
fn dipole_p2p_equals_the_reference_for_near_pairs() {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0xd1b3);
    let choices = p2p_choices("dipole P2P vs p2p::p2p_dipoles, near pairs");
    let mut pairs = 0;
    for sources in TYPES {
        for &choice in &choices {
            for gradients in [true, false] {
                let op = LaplaceOperator::new(
                    tables(M2lStrategy::Rotation, 0).clone(),
                    gradients,
                    POINTS,
                )
                .with_sources(sources)
                .with_p2p(choice)
                .expect("an available P2P kernel");
                let mut op = op;
                let worst = Worst::new(format!(
                    "dipole P2P ({sources}, {choice}, gradients {gradients}) vs p2p::p2p_dipoles \
                     (terms)"
                ));
                for level in LEVELS {
                    let target = rng.interior_key(level);
                    let r_t = key_radius(target, &domain);
                    for source in near_sources(target) {
                        let (y, u_s) = grid_points(source, &domain, POINTS, &mut rng);
                        let r_s = key_radius(source, &domain);
                        let q = rng.charges(POINTS);
                        let d_hat = moments(&mut rng, POINTS);
                        let d: Vec<[f64; 3]> = d_hat.iter().map(|d| d.map(|c| c * r_s)).collect();
                        let (x, u_t) = if source == target {
                            let (mut x, mut u) = grid_points(target, &domain, POINTS, &mut rng);
                            x.splice(0..0, y.iter().copied());
                            u.splice(0..0, u_s.iter().copied());
                            (x, u)
                        } else {
                            grid_points(target, &domain, POINTS, &mut rng)
                        };
                        let n = x.len();
                        let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
                        if sources.has_charges() {
                            p2p::p2p(&y, &q, &x, &mut phi, Some(&mut grad));
                        }
                        p2p::p2p_dipoles(&y, &d, &x, &mut phi, Some(&mut grad));
                        let mut output = vec![0.0; if gradients { 4 } else { 1 } * n];
                        op.p2p_pair(
                            source,
                            target,
                            &chunk(sources, &u_s, &q, &d_hat),
                            &target_chunk(&u_t),
                            &mut output,
                        );
                        for i in 0..n {
                            let (mut tp, mut tg) = dipole_p2p_terms(&y, &d, x[i]);
                            if sources.has_charges() {
                                let (cp, cg) = p2p_terms(&y, &q, x[i]);
                                tp += cp;
                                tg += cg;
                            }
                            let e = (output[i] - r_t * phi[i]).abs() / (r_t * tp);
                            worst.check(e, LEAF_TOL, || {
                                format!("φ, level {level}, source {source}, target {target}")
                            });
                            if gradients {
                                let got: [f64; 3] = core::array::from_fn(|k| output[n + 3 * i + k]);
                                let want = grad[i].map(|g| r_t * r_t * g);
                                let e = norm(sub(got, want)) / (r_t * r_t * tg);
                                worst.check(e, LEAF_TOL, || {
                                    format!("∇φ, level {level}, source {source}, target {target}")
                                });
                            }
                        }
                        pairs += 1;
                    }
                }
            }
        }
    }
    eprintln!("dipole P2P: {pairs} pairs over the source types, kernels and outputs");
}

#[test]
fn charges_keep_the_chunk_and_dipoles_change_it() {
    // Error measure: exact equality of sizes.
    use nd_fmm_plan::operator::FmmSizes;
    let base = LaplaceOperator::new(tables(M2lStrategy::Rotation, 2).clone(), false, 4);
    assert_eq!(base.sources(), SourceType::Charges);
    assert_eq!(base.source_point_size(), 4);
    for (sources, size) in [
        (SourceType::Charges, 4),
        (SourceType::Dipoles, 6),
        (SourceType::ChargesAndDipoles, 7),
    ] {
        let op = base.clone().with_sources(sources);
        assert_eq!(op.source_point_size(), size, "{sources}");
        assert_eq!(op.multipole_size(0), Layout::new(2).len());
    }
}

#[test]
#[should_panic(expected = "dipoles with 2 charge vectors: dipoles carry one set of moments")]
fn dipoles_refuse_several_vectors() {
    let _ = LaplaceOperator::new(tables(M2lStrategy::Rotation, 2).clone(), false, 4)
        .with_vectors(2)
        .with_sources(SourceType::Dipoles);
}

#[test]
#[should_panic(expected = "several vectors need charges only, not charges+dipoles")]
fn several_vectors_refuse_dipoles() {
    let _ = LaplaceOperator::new(tables(M2lStrategy::Rotation, 2).clone(), false, 4)
        .with_sources(SourceType::ChargesAndDipoles)
        .with_vectors(3);
}

#[test]
#[should_panic(expected = "a source chunk holds 6 values per point, got 8 values")]
fn a_dipole_chunk_has_six_values_per_point() {
    let mut op = operator(SourceType::Dipoles, 2, 4);
    let mut multipole = vec![0.0; len(2)];
    op.p2m_leaf(&[0.0; 8], &mut multipole);
}
