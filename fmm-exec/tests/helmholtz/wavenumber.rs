//! The classification of k (CONVENTIONS §3.15): Yukawa (Re k = 0), real (Im k = 0) or
//! complex, and the rejected values.

use std::f64::consts::TAU;

use nd_fmm_exec::helmholtz::{Wavenumber, WavenumberError, WavenumberKind};

#[test]
fn each_kind_is_classified() {
    let cases = [
        ((1.0, 0.0), WavenumberKind::Real),
        ((500.0, 0.0), WavenumberKind::Real),
        ((1e-300, 0.0), WavenumberKind::Real),
        ((1.0, 0.5), WavenumberKind::Complex),
        ((1e-3, 1e3), WavenumberKind::Complex),
        ((0.0, 2.0), WavenumberKind::Yukawa),
        ((0.0, 1e-300), WavenumberKind::Yukawa),
    ];
    for ((re, im), kind) in cases {
        let k = Wavenumber::new(re, im).unwrap();
        assert_eq!(k.kind(), kind, "k = {re} + {im}i");
        assert_eq!((k.re(), k.im()), (re, im));
    }
}

#[test]
fn signed_zeros_count_as_zero() {
    let yukawa = Wavenumber::new(-0.0, 3.0).unwrap();
    assert_eq!(yukawa.kind(), WavenumberKind::Yukawa);
    assert_eq!(yukawa.re().to_bits(), 0.0f64.to_bits(), "stored as +0");
    let real = Wavenumber::new(3.0, -0.0).unwrap();
    assert_eq!(real.kind(), WavenumberKind::Real);
    assert_eq!(real.im().to_bits(), 0.0f64.to_bits(), "stored as +0");
    assert_eq!(
        Wavenumber::new(-0.0, -0.0),
        Err(WavenumberError::Zero),
        "k = 0 with both signs negative"
    );
}

#[test]
fn rejected_values() {
    assert_eq!(Wavenumber::new(0.0, 0.0), Err(WavenumberError::Zero));
    for (re, im) in [
        (f64::NAN, 0.0),
        (1.0, f64::NAN),
        (f64::INFINITY, 0.0),
        (1.0, f64::INFINITY),
        (f64::NEG_INFINITY, 1.0),
    ] {
        assert!(
            matches!(
                Wavenumber::new(re, im),
                Err(WavenumberError::NotFinite { .. })
            ),
            "k = {re} + {im}i"
        );
    }
    assert_eq!(
        Wavenumber::new(-1.0, 0.0),
        Err(WavenumberError::Incoming { re: -1.0 })
    );
    assert_eq!(
        Wavenumber::new(-1.0, 1.0),
        Err(WavenumberError::Incoming { re: -1.0 })
    );
    assert_eq!(
        Wavenumber::new(1.0, -0.5),
        Err(WavenumberError::Growing { im: -0.5 })
    );
    assert_eq!(
        Wavenumber::new(0.0, -0.5),
        Err(WavenumberError::Growing { im: -0.5 })
    );
}

#[test]
fn wavelength_is_two_pi_over_re_k() {
    assert_eq!(
        Wavenumber::new(2.0, 0.0).unwrap().wavelength(),
        Some(TAU / 2.0)
    );
    assert_eq!(
        Wavenumber::new(4.0, 7.0).unwrap().wavelength(),
        Some(TAU / 4.0),
        "complex k: the wavelength of Re k"
    );
    assert_eq!(Wavenumber::new(0.0, 7.0).unwrap().wavelength(), None);
}

#[test]
fn errors_display() {
    let messages = [
        Wavenumber::new(f64::NAN, 0.0).unwrap_err().to_string(),
        Wavenumber::new(0.0, 0.0).unwrap_err().to_string(),
        Wavenumber::new(-1.0, 0.0).unwrap_err().to_string(),
        Wavenumber::new(1.0, -1.0).unwrap_err().to_string(),
    ];
    for m in &messages {
        assert!(!m.is_empty());
    }
    assert!(messages[1].contains("Laplace"), "{}", messages[1]);
}
