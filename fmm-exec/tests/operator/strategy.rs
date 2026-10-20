//! Strategies: `Auto` resolves on the host by the rule of Phase 6 T11 (Batched below
//! f64 p = 7 and f32 p = 8, PlaneWave from there; decision 7 of docs/phase6/README.md),
//! `Dense`, `Classes` and `Rotation` give the same translations, and `Batched` (Phase 6
//! T2) holds `Dense`'s tables and names its column kernel.

use nd_fmm_exec::operator::{Isa, LaplaceOperator, MulAdd, P2pChoice};
use nd_fmm_exec::tables::{M2lStrategy, NodeContext, StrategyChoice, Tables};
use nd_fmm_tables::cache::TableKind;
use nd_fmm_tables::geometry::m2l_offsets;
use nd_octree::morton;

use crate::common::{
    Kind, STRATEGIES, SplitMix64, Worst, degree_error, dense, len, operator, random_coefficients,
    random_pair, terms,
};

/// The rule of Phase 6 T11, restated: PlaneWave from f64 p = 7 and f32 p = 8, Batched
/// below, whatever the ranks per node and the budget.
fn rule(p: usize, value_bytes: usize) -> M2lStrategy {
    let threshold = if value_bytes == 4 { 8 } else { 7 };
    if p >= threshold {
        M2lStrategy::PlaneWave
    } else {
        M2lStrategy::Batched
    }
}

#[test]
fn auto_resolves_by_the_node_rule() {
    // Error measure: exact equality (enumerations, byte counts and the reason's text).
    assert_eq!(M2lStrategy::AUTO_PLANE_WAVE_MIN_P_F64, 7);
    assert_eq!(M2lStrategy::AUTO_PLANE_WAVE_MIN_P_F32, 8);
    assert_eq!(M2lStrategy::auto_plane_wave_min_p(8), 7);
    assert_eq!(M2lStrategy::auto_plane_wave_min_p(4), 8);
    assert_eq!(M2lStrategy::AUTO_DENSE_RANK_BYTES, 9_000_000);
    assert_eq!(M2lStrategy::AUTO_DENSE_NODE_BYTES, 64_000_000);
    assert_eq!(M2lStrategy::default(), M2lStrategy::Auto);
    // Both precisions, p = 0–20, 1–128 ranks per node, the default budget, the harness's
    // 96 MB, and budgets at the edges: the ranks and the budget do not enter.
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
                        rule(p, value_bytes),
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

    // The table sizes of node-m2l.md §2.2, still reported.
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
    let default = M2lStrategy::AUTO_DENSE_NODE_BYTES;

    // The reason, as the threading report shows it.
    let text = |p, b, ranks, budget| {
        M2lStrategy::Auto
            .choose(&NodeContext::new(p, b, ranks, budget))
            .to_string()
    };
    assert_eq!(text(8, 4, 12, default), "Auto → PlaneWave: f32 p = 8 ≥ 8");
    assert_eq!(text(7, 4, 12, default), "Auto → Batched: f32 p = 7 < 8");
    assert_eq!(text(6, 8, 4, default), "Auto → Batched: f64 p = 6 < 7");
    assert_eq!(text(7, 8, 1, 96_000_000), "Auto → PlaneWave: f64 p = 7 ≥ 7");
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
        assert_eq!(M2lStrategy::Auto.resolve(p), rule(p, 8), "p = {p}");
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
    // f32 at p = 7: Batched, where f64 resolves to PlaneWave.
    assert_eq!(
        Tables::<f32>::build(7, M2lStrategy::Auto).strategy(),
        M2lStrategy::Batched
    );
    assert_eq!(
        Tables::<f64>::build(7, M2lStrategy::Auto).strategy(),
        M2lStrategy::PlaneWave
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

#[test]
fn batched_holds_the_dense_tables_and_names_its_kernel() {
    // Phase 6 T2 (C6.6). Error measures: exact equality of strategies, kinds, kernels and
    // the bits of every pair translation. The level calls of `Batched` (per block, through
    // the column kernel) are checked against `Dense` bit for bit in `tests/mpi_exec.rs`;
    // here its tables and settings.
    let p = 4;
    let (dense, batched) = (Tables::<f64>::build(p, M2lStrategy::Dense), {
        Tables::<f64>::build(p, M2lStrategy::Batched)
    });
    assert_eq!(batched.strategy(), M2lStrategy::Batched);
    assert_eq!(batched.kinds(), dense.kinds());
    assert_eq!(
        batched.kinds(),
        [TableKind::M2m, TableKind::L2l, TableKind::M2l]
    );
    // Named, never overridden; `Auto`'s choice at this degree (Phase 6 T11).
    for ranks in [1, 12, 128] {
        let node = NodeContext::new(p, 8, ranks, 64_000_000);
        assert_eq!(
            M2lStrategy::Batched.choose(&node),
            StrategyChoice::Named(M2lStrategy::Batched)
        );
        assert_eq!(
            M2lStrategy::Auto.resolve_on_node(&node),
            M2lStrategy::Batched
        );
    }
    // One pair at a time it applies the dense tables: every translation bit for bit.
    let n = len(p);
    let mut rng = SplitMix64::new(0xba7c);
    let (mut s_dense, mut s_batched) = (dense.scratch(), batched.scratch());
    for index in 0..316 {
        let x = random_coefficients(Kind::Multipole, p, &mut rng);
        let (mut a, mut b) = (vec![0.5; n], vec![0.5; n]);
        dense.m2l(index, &x, &mut a, &mut s_dense);
        batched.m2l(index, &x, &mut b, &mut s_batched);
        assert_eq!(a, b, "offset {index}");
    }
    for o in 0..8 {
        let x = random_coefficients(Kind::Multipole, p, &mut rng);
        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
        dense.m2m(o, &x, &mut a, &mut s_dense);
        batched.m2m(o, &x, &mut b, &mut s_batched);
        dense.l2l(o, &x, &mut a, &mut s_dense);
        batched.l2l(o, &x, &mut b, &mut s_batched);
        assert_eq!(a, b, "octant {o}");
    }

    // The column kernel: on the P2P ISA (the scalar path for the reference), separate
    // by default; none under another strategy.
    let op = LaplaceOperator::new(batched.clone(), false, 8);
    let kernel = op.m2l_kernel().expect("Batched has a column kernel");
    assert_eq!(
        (kernel.isa(), kernel.mul_add()),
        (Isa::detect(), MulAdd::Separate)
    );
    let mut op = op.with_m2l_mul_add(MulAdd::Fused);
    assert_eq!(op.m2l_kernel().unwrap().mul_add(), MulAdd::Fused);
    op.set_m2l_mul_add(MulAdd::Separate);
    assert_eq!(op.m2l_kernel().unwrap().mul_add(), MulAdd::Separate);
    let reference = op
        .with_m2l_mul_add(MulAdd::Fused)
        .with_p2p(P2pChoice::Reference)
        .unwrap();
    let kernel = reference.m2l_kernel().unwrap();
    assert_eq!(
        (kernel.isa(), kernel.mul_add()),
        (Isa::Scalar, MulAdd::Fused)
    );
    for isa in Isa::available() {
        let op = LaplaceOperator::new(batched.clone(), false, 8)
            .with_p2p(P2pChoice::Isa(isa))
            .unwrap();
        assert_eq!(op.m2l_kernel().unwrap().isa(), isa);
    }
    for strategy in STRATEGIES {
        let op = LaplaceOperator::new(Tables::<f64>::build(p, strategy), false, 8)
            .with_m2l_mul_add(MulAdd::Fused);
        assert!(op.m2l_kernel().is_none(), "{strategy:?}");
    }
}
