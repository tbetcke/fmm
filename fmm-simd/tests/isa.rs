//! Acceptance tests for ISA detection and kernel construction (Phase 3S T3): what is
//! available, which constructors fail, lanes per ISA and precision, and names.
//!
//! Every test prints the ISAs this machine offers; run with `--show-output` to see it.

use nd_fmm_simd::{Isa, IsaUnavailable, P2pKernel};

/// Prints the ISAs this machine offers and the one detection picks.
fn print_isas(test: &str) {
    let available: Vec<String> = Isa::available().map(|isa| isa.to_string()).collect();
    println!(
        "{test}: available ISAs: {}; detected: {}",
        available.join(", "),
        Isa::detect()
    );
}

#[test]
fn detected_isa_is_available() {
    print_isas("detected_isa_is_available");
    assert!(Isa::detect().is_available());
    assert_eq!(P2pKernel::<f32>::detect().isa(), Isa::detect());
    assert_eq!(P2pKernel::<f64>::detect().isa(), Isa::detect());
}

#[test]
fn scalar_is_always_available() {
    print_isas("scalar_is_always_available");
    assert!(Isa::Scalar.is_available());
    assert_eq!(Isa::available().next(), Some(Isa::Scalar));
}

#[test]
fn available_contains_detect_and_scalar() {
    print_isas("available_contains_detect_and_scalar");
    let available: Vec<Isa> = Isa::available().collect();
    assert!(available.contains(&Isa::Scalar));
    assert!(available.contains(&Isa::detect()));
    assert!(available.iter().all(|isa| isa.is_available()));
    let unavailable: Vec<Isa> = Isa::all().filter(|isa| !available.contains(isa)).collect();
    assert!(unavailable.iter().all(|isa| !isa.is_available()));
}

#[test]
fn all_lists_every_variant_once() {
    print_isas("all_lists_every_variant_once");
    let all: Vec<Isa> = Isa::all().collect();
    assert_eq!(all, [Isa::Scalar, Isa::Neon, Isa::Avx2]);
}

#[test]
fn detection_matches_the_architecture() {
    print_isas("detection_matches_the_architecture");
    assert_eq!(Isa::Neon.is_available(), cfg!(target_arch = "aarch64"));
    #[cfg(target_arch = "x86_64")]
    {
        let detected = std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma");
        assert_eq!(Isa::Avx2.is_available(), detected);
        let expected = if detected { Isa::Avx2 } else { Isa::Scalar };
        assert_eq!(Isa::detect(), expected);
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert!(!Isa::Avx2.is_available());
    #[cfg(target_arch = "aarch64")]
    assert_eq!(Isa::detect(), Isa::Neon);
}

#[test]
fn new_fails_exactly_for_unavailable_isas() {
    print_isas("new_fails_exactly_for_unavailable_isas");
    for isa in Isa::all() {
        check_new::<f32>(isa);
        check_new::<f64>(isa);
    }
}

/// `P2pKernel::<T>::new(isa)` succeeds if and only if `isa` is available, and the
/// error names the ISA.
fn check_new<T: nd_fmm_simd::SimdScalar>(isa: Isa) {
    match P2pKernel::<T>::new(isa) {
        Ok(kernel) => {
            assert!(isa.is_available(), "{isa} is not available");
            assert_eq!(kernel.isa(), isa);
        }
        Err(error) => {
            assert!(!isa.is_available(), "{isa} is available");
            assert_eq!(error, IsaUnavailable { isa });
            assert_eq!(
                error.to_string(),
                format!("instruction set `{isa}` is not available on this machine")
            );
        }
    }
}

#[test]
fn lanes_per_isa_and_precision() {
    // docs/design/simd-p2p.md §3, requirement 7.
    print_isas("lanes_per_isa_and_precision");
    assert_eq!(Isa::Scalar.lanes::<f32>(), 1);
    assert_eq!(Isa::Scalar.lanes::<f64>(), 1);
    assert_eq!(Isa::Neon.lanes::<f32>(), 4);
    assert_eq!(Isa::Neon.lanes::<f64>(), 2);
    assert_eq!(Isa::Avx2.lanes::<f32>(), 8);
    assert_eq!(Isa::Avx2.lanes::<f64>(), 4);
}

#[test]
fn display_names() {
    print_isas("display_names");
    let names: Vec<String> = Isa::all().map(|isa| isa.to_string()).collect();
    assert_eq!(names, ["scalar", "neon", "avx2"]);
}

#[test]
fn kernel_is_copy_send_and_sync() {
    fn assert_traits<K: Copy + Send + Sync + 'static>() {}
    assert_traits::<P2pKernel<f32>>();
    assert_traits::<P2pKernel<f64>>();
}
