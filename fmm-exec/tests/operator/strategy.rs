//! Strategies: `Auto` resolves by the node rule of Phase 5N T4 (docs/design/node-m2l.md
//! §4), and `Dense`, `Classes` and `Rotation` give the same translations.

use nd_fmm_exec::tables::{M2lStrategy, NodeContext, StrategyChoice, Tables};
use nd_fmm_tables::cache::TableKind;
use nd_fmm_tables::geometry::m2l_offsets;
use nd_octree::morton;

use crate::common::{
    Kind, STRATEGIES, SplitMix64, Worst, degree_error, dense, len, operator, random_coefficients,
    random_pair, terms,
};

/// The rule of node-m2l.md §4 (rule B), restated: Dense iff the dense tables of one rank,
/// 332 (p + 1)⁴ values of `value_bytes`, are at most 9 MB and those of the node at most the
/// budget.
fn rule_b(p: usize, value_bytes: usize, ranks_per_node: usize, budget: usize) -> M2lStrategy {
    let rank = 332 * (p as u128 + 1).pow(4) * value_bytes as u128;
    if rank <= 9_000_000 && ranks_per_node as u128 * rank <= budget as u128 {
        M2lStrategy::Dense
    } else {
        M2lStrategy::Rotation
    }
}

#[test]
fn auto_resolves_by_the_node_rule() {
    // Error measure: exact equality (enumerations, byte counts and the reason's text).
    assert_eq!(M2lStrategy::AUTO_DENSE_RANK_BYTES, 9_000_000);
    assert_eq!(M2lStrategy::AUTO_DENSE_NODE_BYTES, 64_000_000);
    assert_eq!(M2lStrategy::default(), M2lStrategy::Auto);
    // Every input the design names: both precisions, p = 0–20, 1–128 ranks per node, the
    // default budget, the harness's 96 MB, and budgets at the edges.
    let mut cells = 0;
    for value_bytes in [4, 8] {
        for p in 0..=20 {
            for ranks in 1..=128 {
                for budget in [0, 1, 48_000_000, 64_000_000, 96_000_000, usize::MAX] {
                    let node = NodeContext::new(p, value_bytes, ranks, budget);
                    let got = M2lStrategy::Auto.resolve_on_node(&node);
                    assert_ne!(got, M2lStrategy::Auto);
                    assert_eq!(
                        got,
                        rule_b(p, value_bytes, ranks, budget),
                        "p = {p}, {value_bytes}-byte values, {ranks} ranks per node, budget \
                         {budget}"
                    );
                    assert_eq!(M2lStrategy::Auto.choose(&node).strategy(), got);
                    // A named strategy is never overridden.
                    for strategy in STRATEGIES {
                        assert_eq!(strategy.resolve_on_node(&node), strategy);
                        assert_eq!(strategy.choose(&node), StrategyChoice::Named(strategy));
                    }
                    cells += 1;
                }
            }
        }
    }
    assert_eq!(cells, 2 * 21 * 128 * 6);

    // The table sizes of node-m2l.md §2.2 and the cells of §3 and §4.
    let rank_mb = |p, b| NodeContext::new(p, b, 1, 0).dense_rank_bytes() as f64 / 1e6;
    for (p, b, mb) in [
        (3, 8, 0.68),
        (6, 8, 6.38),
        (6, 4, 3.19),
        (8, 8, 17.43),
        (8, 4, 8.71),
        (10, 8, 38.89),
        (12, 8, 75.85),
    ] {
        assert!(
            (rank_mb(p, b) - mb).abs() < 0.01,
            "p = {p}, {b} bytes: {}",
            rank_mb(p, b)
        );
    }
    let auto = |p, b, ranks, budget| {
        M2lStrategy::Auto.resolve_on_node(&NodeContext::new(p, b, ranks, budget))
    };
    let (dense, rotation) = (M2lStrategy::Dense, M2lStrategy::Rotation);
    let default = M2lStrategy::AUTO_DENSE_NODE_BYTES;
    // One rank: Dense where it wins on one rank (f64 p ≤ 6, f32 p ≤ 8), Rotation from
    // f64 p = 7 and f32 p = 9; whatever the budget above.
    for p in 0..=20 {
        assert_eq!(
            auto(p, 8, 1, default),
            if p <= 6 { dense } else { rotation },
            "f64 p = {p}"
        );
        assert_eq!(
            auto(p, 4, 1, default),
            if p <= 8 { dense } else { rotation },
            "f32 p = {p}"
        );
        assert_eq!(
            auto(p, 8, 1, usize::MAX),
            if p <= 6 { dense } else { rotation }
        );
    }
    // f64 p = 8 resolves to Rotation at every count (the best in all 48 cells of §2.1).
    for ranks in 1..=128 {
        assert_eq!(auto(8, 8, ranks, default), rotation);
        assert_eq!(auto(8, 8, ranks, 96_000_000), rotation);
    }
    // The budget: f64 p = 6 (6.4 MB) Dense to 10 ranks per node at 64 MB, to 15 at 96 MB;
    // f32 p = 8 (8.7 MB) to 7 and to 11; f64 p = 3 (0.68 MB) to 94 at 64 MB, so on every
    // core of locust (72) and of a Kathleen node (40).
    for (p, b, budget, last) in [
        (6, 8, default, 10),
        (6, 8, 96_000_000, 15),
        (8, 4, default, 7),
        (8, 4, 96_000_000, 11),
        (3, 8, default, 94),
    ] {
        assert_eq!(
            auto(p, b, last, budget),
            dense,
            "p = {p}, {b} bytes, {last} ranks"
        );
        assert_eq!(
            auto(p, b, last + 1, budget),
            rotation,
            "p = {p}, {b} bytes, {} ranks",
            last + 1
        );
    }

    // The reason, as the threading report shows it.
    let text = |p, b, ranks, budget| {
        M2lStrategy::Auto
            .choose(&NodeContext::new(p, b, ranks, budget))
            .to_string()
    };
    assert_eq!(
        text(8, 4, 12, default),
        "Auto → Rotation: 12 ranks per node × 8.7 MB > 64 MB"
    );
    assert_eq!(
        text(6, 8, 4, default),
        "Auto → Dense: 4 ranks per node × 6.4 MB ≤ 64 MB"
    );
    assert_eq!(
        text(3, 8, 1, 96_000_000),
        "Auto → Dense: 1 rank per node × 0.7 MB ≤ 96 MB"
    );
    assert_eq!(
        text(8, 8, 1, default),
        "Auto → Rotation: 17.4 MB per rank > 9 MB"
    );
    assert_eq!(
        StrategyChoice::Named(M2lStrategy::Classes).to_string(),
        "Classes (named)"
    );
    assert_eq!(
        StrategyChoice::Device(M2lStrategy::Dense).to_string(),
        "Auto → Dense (the device path)"
    );

    // The deprecated one-rank f64 form agrees with the rule.
    #[allow(deprecated)]
    for p in 0..=20 {
        assert_eq!(
            M2lStrategy::Auto.resolve(p),
            auto(p, 8, 1, default),
            "p = {p}"
        );
        assert_eq!(M2lStrategy::AUTO_DENSE_MAX_P, 8);
    }

    // The tables resolve `Auto` on one rank in their own precision, and hold the families
    // of the resolved strategy only.
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
        (7, M2lStrategy::Auto, &[TableKind::Rotation]),
    ] {
        let tables = Tables::<f64>::build(p, strategy);
        assert_eq!(tables.p(), p);
        assert_eq!(
            tables.strategy(),
            strategy.resolve_on_node(&NodeContext::one_rank::<f64>(p))
        );
        assert_eq!(tables.kinds(), kinds);
    }
    // f32 at p = 7: Dense on one rank (5.4 MB), where f64 (10.9 MB) resolves to Rotation.
    assert_eq!(
        Tables::<f32>::build(7, M2lStrategy::Auto).strategy(),
        M2lStrategy::Dense
    );
    assert_eq!(
        NodeContext::one_rank::<f32>(8),
        NodeContext::new(8, 4, 1, default)
    );
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
