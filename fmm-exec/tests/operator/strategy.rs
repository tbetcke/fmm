//! Strategies: `Auto` resolves as documented, and `Dense`, `Classes` and `Rotation`
//! give the same translations.

use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_tables::cache::TableKind;
use nd_fmm_tables::geometry::m2l_offsets;
use nd_octree::morton;

use crate::common::{
    Kind, STRATEGIES, SplitMix64, Worst, degree_error, dense, len, operator, random_coefficients,
    random_pair, terms,
};

#[test]
fn auto_resolves_to_dense_up_to_p_8_and_to_rotation_above() {
    // Error measure: exact equality (enumerations).
    for p in 0..=20 {
        let want = if p <= 8 {
            M2lStrategy::Dense
        } else {
            M2lStrategy::Rotation
        };
        assert_eq!(M2lStrategy::Auto.resolve(p), want, "p = {p}");
        for strategy in STRATEGIES {
            assert_eq!(strategy.resolve(p), strategy, "p = {p}");
        }
    }
    assert_eq!(M2lStrategy::AUTO_DENSE_MAX_P, 8);
    assert_eq!(M2lStrategy::default(), M2lStrategy::Auto);
    // The tables hold the families of the resolved strategy, and only those.
    for (p, strategy, kinds) in [
        (
            2,
            M2lStrategy::Auto,
            &[TableKind::M2m, TableKind::L2l, TableKind::M2l][..],
        ),
        (
            2,
            M2lStrategy::Classes,
            &[TableKind::M2m, TableKind::L2l, TableKind::M2lClasses],
        ),
        (2, M2lStrategy::Rotation, &[TableKind::Rotation]),
        (9, M2lStrategy::Auto, &[TableKind::Rotation]),
    ] {
        let tables = Tables::<f64>::build(p, strategy);
        assert_eq!(tables.p(), p);
        assert_eq!(tables.strategy(), strategy.resolve(p));
        assert_eq!(tables.kinds(), kinds);
    }
}

#[test]
fn dense_classes_and_rotation_give_the_same_translations() {
    // Error measure: output coefficients per degree, relative to the terms of the dense
    // table (`common`), of `Classes` and `Rotation` against `Dense`: M2M and L2L for
    // every octant, M2L for every offset, on level 9 of the dyadic domain, p ∈ {2, 6}.
    // Tolerance 1e-13.
    let mut rng = SplitMix64::new(0x7831);
    let worst = Worst::new("Classes and Rotation vs Dense, p ∈ {2, 6} (terms, per degree)");
    for p in [2, 6] {
        let dense_tables = dense(p);
        let mut reference = operator(M2lStrategy::Dense, p, 0);
        for strategy in [M2lStrategy::Classes, M2lStrategy::Rotation] {
            let mut op = operator(strategy, p, 0);
            let context = || format!("{strategy:?}, p = {p}");
            for o in 0..8 {
                let parent = rng.key(8);
                let child = morton::children(parent).unwrap()[o];
                let x = random_coefficients(Kind::Multipole, p, &mut rng);
                let (mut got, mut want) = (vec![0.0; len(p)], vec![0.0; len(p)]);
                op.m2m_pair(child, parent, &x, &mut got);
                reference.m2m_pair(child, parent, &x, &mut want);
                let scale = terms(&dense_tables.m2m, o, &x);
                worst.check(
                    degree_error(Kind::Multipole, p, &got, &want, &scale),
                    1e-13,
                    context,
                );

                let x = random_coefficients(Kind::Local, p, &mut rng);
                let (mut got, mut want) = (vec![0.0; len(p)], vec![0.0; len(p)]);
                op.l2l_pair(parent, child, &x, &mut got);
                reference.l2l_pair(parent, child, &x, &mut want);
                let scale = terms(&dense_tables.l2l, o, &x);
                worst.check(
                    degree_error(Kind::Local, p, &got, &want, &scale),
                    1e-13,
                    context,
                );
            }
            for (index, d) in m2l_offsets().into_iter().enumerate() {
                let (source, target) = random_pair(9, d, &mut rng);
                let x = random_coefficients(Kind::Multipole, p, &mut rng);
                let (mut got, mut want) = (vec![0.0; len(p)], vec![0.0; len(p)]);
                op.m2l_pair(source, target, &x, &mut got);
                reference.m2l_pair(source, target, &x, &mut want);
                let scale = terms(&dense_tables.m2l, index, &x);
                worst.check(
                    degree_error(Kind::Local, p, &got, &want, &scale),
                    1e-13,
                    context,
                );
            }
        }
    }
}
