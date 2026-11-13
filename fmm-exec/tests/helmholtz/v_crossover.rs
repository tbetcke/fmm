//! The V rule's cost model (Helmholtz CH4.0d2): `v_crossover` against its closed form, the
//! M2L's thread time per pair over the P2P's per interaction, from the published constants.

use nd_fmm_exec::helmholtz::{
    HelmholtzM2l, V_BATCHED_NS, V_P2P_NS, V_P2P_POTENTIAL_SHARE, V_PLANE_WAVE_NS, V_ROTATION_NS,
    v_crossover,
};

#[test]
fn the_crossover_is_the_m2ls_time_over_the_p2ps() {
    for p in [3usize, 7, 11, 15, 24, 32] {
        let q = (p + 1) as f64;
        let rotation = (V_ROTATION_NS[0] + V_ROTATION_NS[1] * q.powi(3)) / V_P2P_NS;
        let batched = V_BATCHED_NS * q.powi(4) / V_P2P_NS;
        let waves = V_PLANE_WAVE_NS * q * q / V_P2P_NS;
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b;
        assert!(close(
            v_crossover(HelmholtzM2l::Rotation, p, true),
            rotation
        ));
        assert!(close(v_crossover(HelmholtzM2l::Batched, p, true), batched));
        // Dense and Classes are charged as Batched (one dense product per pair).
        assert!(close(v_crossover(HelmholtzM2l::Dense, p, true), batched));
        assert!(close(v_crossover(HelmholtzM2l::Classes, p, true), batched));
        assert!(close(v_crossover(HelmholtzM2l::PlaneWave, p, true), waves));
        // Without gradients the P2P is cheaper, so more pairs convert.
        assert!(close(
            v_crossover(HelmholtzM2l::Rotation, p, false),
            rotation / V_P2P_POTENTIAL_SHARE
        ));
    }
    // The fit's points (fmm-validate/results/helmholtz-ch4-0d2.md): Rotation's crossover at
    // p = 15 and 24 near 3,900 and 15,000, Batched's at p = 7 near 160.
    let rotation = |p| v_crossover(HelmholtzM2l::Rotation, p, true);
    assert!((3_500.0..4_300.0).contains(&rotation(15)));
    assert!((13_500.0..16_500.0).contains(&rotation(24)));
    assert!((120.0..200.0).contains(&v_crossover(HelmholtzM2l::Batched, 7, true)));
}
