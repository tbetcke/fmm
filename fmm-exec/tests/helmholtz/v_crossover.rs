//! The V rule's cost model (Helmholtz CH4.0d2): `v_crossover` against its closed form, the
//! M2L's thread time per pair over the P2P's per interaction, from each machine class's
//! published constants; and the W and X rule's (Helmholtz CH4.0d3): `wx_crossovers` against
//! its closed form and the report's values at p = 15; and the crossover after the plan
//! (Helmholtz CH4.3b2b): `v_after_plan_crossover` against its closed form; and the raised
//! levels' crossover before the plan (Helmholtz CH4.3b2d): `raised_crossover` against its
//! closed form and against the after-plan crossover of the same counts.

use nd_fmm_exec::helmholtz::{
    DIAGONAL_COSTS_NEON, DIAGONAL_COSTS_X86, DiagonalCosts, DiagonalShape, HelmholtzM2l,
    LeafVShape, V_COSTS_NEON, V_COSTS_X86, V_P2P_POTENTIAL_SHARE, VLevelM2l, WX_COSTS_NEON,
    WX_COSTS_X86, raised_crossover, v_after_plan_crossover, v_costs, v_crossover, wx_costs,
    wx_crossovers,
};
use nd_fmm_exec::operator::Isa;

#[test]
fn the_crossover_is_the_m2ls_time_over_the_p2ps() {
    for costs in [V_COSTS_X86, V_COSTS_NEON] {
        for p in [3usize, 7, 11, 15, 24, 32] {
            let q = (p + 1) as f64;
            let rotation = (costs.rotation[0] + costs.rotation[1] * q.powi(3)) / costs.p2p;
            let batched = costs.batched * q.powi(4) / costs.p2p;
            let waves = costs.plane_wave * q * q / costs.p2p;
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b;
            let at = |m2l| v_crossover(&costs, m2l, p, true);
            assert!(close(at(HelmholtzM2l::Rotation), rotation));
            assert!(close(at(HelmholtzM2l::Batched), batched));
            // Dense and Classes are charged as Batched (one dense product per pair).
            assert!(close(at(HelmholtzM2l::Dense), batched));
            assert!(close(at(HelmholtzM2l::Classes), batched));
            assert!(close(at(HelmholtzM2l::PlaneWave), waves));
            // Without gradients the P2P is cheaper, so more pairs convert.
            assert!(close(
                v_crossover(&costs, HelmholtzM2l::Rotation, p, false),
                rotation / V_P2P_POTENTIAL_SHARE
            ));
        }
    }
    // The classes by ISA.
    assert_eq!(v_costs(Isa::Neon), V_COSTS_NEON);
    for isa in [Isa::Scalar, Isa::Avx2, Isa::Avx512] {
        assert_eq!(v_costs(isa), V_COSTS_X86);
    }
    // The fit's points (fmm-validate/results/helmholtz-ch4-0d2.md): on x86 Rotation's
    // crossover at p = 15 and 24 near 3,900 and 15,000, Batched's at p = 7 near 160.
    let x86 = |m2l, p| v_crossover(&V_COSTS_X86, m2l, p, true);
    assert!((3_500.0..4_300.0).contains(&x86(HelmholtzM2l::Rotation, 15)));
    assert!((13_500.0..16_500.0).contains(&x86(HelmholtzM2l::Rotation, 24)));
    assert!((120.0..200.0).contains(&x86(HelmholtzM2l::Batched, 7)));
}

/// N_W = (a + b (p + 1)²)_M2P / P2P and N_X = (a + b (p + 1)²)_P2L / P2P, growing with p, each
/// class's constants by its ISA, and at p = 15 the values of the report's §2 (about 1100 and
/// 360 on x86, 370 and 110 on NEON).
#[test]
fn the_w_and_x_crossovers_are_the_leaf_operators_time_over_the_p2ps() {
    for (wx, v) in [(WX_COSTS_X86, V_COSTS_X86), (WX_COSTS_NEON, V_COSTS_NEON)] {
        let mut last = [0.0; 2];
        for p in [3usize, 7, 11, 15, 24, 32] {
            let q2 = ((p + 1) * (p + 1)) as f64;
            let [w, x] = wx_crossovers(&wx, &v, p);
            let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b;
            assert!(close(w, (wx.m2p[0] + wx.m2p[1] * q2) / v.p2p), "p = {p}");
            assert!(close(x, (wx.p2l[0] + wx.p2l[1] * q2) / v.p2p), "p = {p}");
            assert!(w > last[0] && x > last[1], "p = {p}: rising");
            last = [w, x];
        }
    }
    let at = |isa| wx_crossovers(&wx_costs(isa), &v_costs(isa), 15);
    let [w, x] = at(Isa::Avx512);
    assert!(
        (1000.0..1200.0).contains(&w) && (330.0..390.0).contains(&x),
        "{w}, {x}"
    );
    let [w, x] = at(Isa::Neon);
    assert!(
        (340.0..400.0).contains(&w) && (100.0..120.0).contains(&x),
        "{w}, {x}"
    );
    assert_eq!(wx_costs(Isa::Avx2), WX_COSTS_X86);
    assert_eq!(wx_costs(Isa::Scalar), WX_COSTS_X86);
    assert_eq!(wx_costs(Isa::Neon), WX_COSTS_NEON);
}

/// After the plan (Helmholtz CH4.3b2b): the form's pair a (p + 1)² + b (p + 1)³ and the waves'
/// cost per box with a V row times those boxes over the level's pairs, each over the P2P's per
/// interaction (without gradients over its potential share), RCR's `v_crossover`'s; a level
/// without pairs 0. The waves' crossover falls as the pairs per box rise, so a sparse level's
/// lies far above the per-pair constant of the plan (`VCosts::plane_wave`), a dense one's near it.
#[test]
fn the_crossover_after_the_plan_is_the_runs_time_over_the_p2ps() {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b;
    let shape = |targets: u64, pairs: u64| {
        let mut class_pairs = [0u64; 34];
        // Spread over two classes: only the sum counts.
        class_pairs[0] = pairs / 2;
        class_pairs[33] = pairs - pairs / 2;
        DiagonalShape {
            targets,
            class_pairs,
        }
    };
    for (v, d) in [
        (V_COSTS_X86, DIAGONAL_COSTS_X86),
        (V_COSTS_NEON, DIAGONAL_COSTS_NEON),
    ] {
        for p in [8usize, 15, 22, 32, 44] {
            let q = (p + 1) as f64;
            let at = |m2l, s: &DiagonalShape, gradients| {
                v_after_plan_crossover(&v, &d, m2l, p, s, gradients)
            };
            let level = shape(1_000, 45_000);
            let form = (d.pair[0] * q * q + d.pair[1] * q.powi(3)) / v.p2p;
            assert!(close(at(VLevelM2l::Form, &level, true), form), "p = {p}");
            assert!(close(
                at(VLevelM2l::Form, &level, false),
                form / V_P2P_POTENTIAL_SHARE
            ));
            assert!(close(
                at(VLevelM2l::Rcr, &level, true),
                v_crossover(&v, HelmholtzM2l::Rotation, p, true)
            ));
            let waves = DiagonalCosts::unit(d.plane_wave, p) / 45.0 / v.p2p;
            assert!(close(at(VLevelM2l::Waves, &level, true), waves), "p = {p}");
            // Four times the pairs per box, a quarter of the crossover.
            let dense = shape(1_000, 180_000);
            assert!(close(at(VLevelM2l::Waves, &dense, true), waves / 4.0));
            assert_eq!(at(VLevelM2l::Waves, &shape(10, 0), true), 0.0);
            // At W1's 45 pairs per box the waves' mean exceeds the plan's per-pair constant.
            assert!(
                at(VLevelM2l::Waves, &level, true)
                    > v_crossover(&v, HelmholtzM2l::PlaneWave, p, true)
            );
        }
    }
}

/// Before the plan on a raised level (Helmholtz CH4.3b2d): the waves' cost per box with a V row
/// at p_w times the leaves with a V entry over their entries, over the P2P's per interaction
/// (without gradients over its potential share); a level without entries 0; equal to the
/// after-plan crossover of a level that keeps its waves with the same boxes and pairs; on W1's
/// sparse raised levels (a few pairs per leaf, CH4.3b2b's report §6: 79,211 at p_w = 32 on
/// x86) far above the per-pair constant, on a dense level (189 pairs per leaf) at most a few
/// times it.
#[test]
fn the_raised_crossover_is_the_waves_cost_per_box_over_the_pairs_per_leaf() {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b;
    for (v, d) in [
        (V_COSTS_X86, DIAGONAL_COSTS_X86),
        (V_COSTS_NEON, DIAGONAL_COSTS_NEON),
    ] {
        for p_w in [28usize, 32] {
            let shape = |leaves, pairs| LeafVShape { leaves, pairs };
            let at = |s, gradients| raised_crossover(&v, &d, p_w, s, gradients);
            let per_box = DiagonalCosts::unit(d.plane_wave, p_w);
            let sparse = shape(1_000, 16_000);
            assert!(
                close(at(sparse, true), per_box / 16.0 / v.p2p),
                "p_w = {p_w}"
            );
            assert!(close(
                at(sparse, false),
                per_box / 16.0 / (v.p2p * V_P2P_POTENTIAL_SHARE)
            ));
            assert_eq!(at(shape(0, 0), true), 0.0);
            // The after-plan crossover of the same boxes and pairs.
            let mut class_pairs = [0u64; 34];
            class_pairs[5] = 16_000;
            let level = DiagonalShape {
                targets: 1_000,
                class_pairs,
            };
            assert!(close(
                at(sparse, true),
                v_after_plan_crossover(&v, &d, VLevelM2l::Waves, p_w, &level, true)
            ));
            let per_pair = v_crossover(&v, HelmholtzM2l::PlaneWave, p_w, true);
            assert!(at(sparse, true) > 5.0 * per_pair, "p_w = {p_w}");
            assert!(
                at(shape(1_000, 189_000), true) < 3.0 * per_pair,
                "p_w = {p_w}"
            );
        }
    }
    // CH4.3b2b's sparse level (W1 at kD = 200, level 5, p_w = 32, x86): 79,211 after the plan,
    // at about 16.8 pairs per box.
    let x86 = raised_crossover(
        &V_COSTS_X86,
        &DIAGONAL_COSTS_X86,
        32,
        LeafVShape {
            leaves: 10_000,
            pairs: 168_000,
        },
        true,
    );
    assert!((70_000.0..90_000.0).contains(&x86), "{x86}");
}
