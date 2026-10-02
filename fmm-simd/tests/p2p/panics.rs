//! Argument checks: mismatched lengths panic on every ISA, with the reference's
//! message.

use std::panic::{AssertUnwindSafe, catch_unwind};

use nd_fmm_simd::P2pKernel;

use crate::common::kernels;

const SOURCES: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
const TARGETS: [[f64; 3]; 1] = [[0.0, 2.0, 0.0]];

/// The signature shared by `nd_fmm_ref::p2p::p2p` and `P2pKernel::evaluate`.
type P2p<'a> = &'a dyn Fn(&[[f64; 3]], &[f64], &[[f64; 3]], &mut [f64], Option<&mut [[f64; 3]]>);

/// The panic message of each mismatched call, or `None` for a call that returns.
fn messages(p2p: P2p<'_>) -> Vec<Option<String>> {
    let calls: [&dyn Fn(); 3] = [
        &|| p2p(&SOURCES, &[1.0], &TARGETS, &mut [0.0], None),
        &|| p2p(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [0.0; 2], None),
        &|| p2p(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [0.0], Some(&mut [])),
    ];
    calls
        .into_iter()
        .map(|call| {
            catch_unwind(AssertUnwindSafe(call)).err().map(|payload| {
                match payload.downcast::<String>() {
                    Ok(message) => *message,
                    Err(_) => "a non-string panic".to_owned(),
                }
            })
        })
        .collect()
}

#[test]
fn mismatched_lengths_panic_as_in_the_reference() {
    let expected = messages(&nd_fmm_ref::p2p::p2p);
    assert!(expected.iter().all(Option::is_some));
    for kernel in kernels::<f64>("mismatched_lengths_panic_as_in_the_reference") {
        let evaluate =
            |s: &[[f64; 3]],
             q: &[f64],
             t: &[[f64; 3]],
             p: &mut [f64],
             g: Option<&mut [[f64; 3]]>| { kernel.evaluate(s, q, t, p, g) };
        assert_eq!(messages(&evaluate), expected, "{}", kernel.isa());
    }
}

#[test]
#[should_panic(expected = "`sources` and `charges` must have the same length, got 2 and 1")]
fn mismatched_charges() {
    P2pKernel::<f64>::detect().evaluate(&SOURCES, &[1.0], &TARGETS, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "`targets` and `potential` must have the same length, got 1 and 2")]
fn mismatched_potential() {
    P2pKernel::<f64>::detect().evaluate(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [0.0; 2], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient` must have the same length, got 1 and 0")]
fn mismatched_gradient() {
    P2pKernel::<f64>::detect().evaluate(&SOURCES, &[1.0, 1.0], &TARGETS, &mut [0.0], Some(&mut []));
}
