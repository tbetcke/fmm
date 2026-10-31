//! The reference workloads as ported into `nd-fmm-validate` (Helmholtz CH3.0b,
//! `nd_fmm_validate::helmholtz_workloads`) equal this spike's point for point, so the
//! census's trees are the trees of the ported workloads.

use nd_fmm_spike_census::workloads::Workload as Spike;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_workloads::{self as ported, Workload as Ported};

#[test]
fn ported_workloads_equal_the_spike_bit_for_bit() {
    let bits = |points: Vec<[f64; 3]>| -> Vec<[u64; 3]> {
        points.into_iter().map(|p| p.map(f64::to_bits)).collect()
    };
    for kd in [1.0, 10.0, 37.5, 100.0] {
        assert_eq!(
            bits(Ported::W1.points(kd)),
            bits(Spike::W1.points(kd)),
            "W1 {kd}"
        );
        assert_eq!(
            bits(Ported::W2.points(kd)),
            bits(Spike::W2.points(kd)),
            "W2 {kd}"
        );
        assert_eq!(
            ported::w1_count(kd),
            nd_fmm_spike_census::workloads::w1_count(kd)
        );
        let (Ported::W3 { n: a }, Spike::W3 { n: b }) =
            (Ported::w3_resolved(kd), Spike::w3_resolved(kd))
        else {
            panic!("W3");
        };
        assert_eq!(a, b, "W3 resolved {kd}");
    }
    let n = 20_000;
    assert_eq!(
        bits(Ported::W3 { n }.points(1.0)),
        bits(Spike::W3 { n }.points(1.0))
    );
    for distribution in [
        Distribution::Cube,
        Distribution::Sphere,
        Distribution::Plummer,
        Distribution::Clusters,
    ] {
        assert_eq!(
            bits(Ported::W4 { distribution, n }.points(1.0)),
            bits(Spike::W4 { distribution, n }.points(1.0)),
            "W4 {}",
            distribution.name()
        );
    }
}
