//! Smoke test of the spike's core: one W1 cell per available ISA and precision, every
//! prototype against `direct_sum` (the check that precedes every timing), and the
//! chosen inverse square roots on a strided subset. Each test prints the ISAs it ran.

use crate::bench::{Variant, call};
use crate::green::Gk;
use crate::kernels::{Order, Prototypes};
use crate::machine;
use crate::rsqrt_study::Candidates;
use crate::simd::Elem;
use crate::workloads::{Form, Oracle, accuracy, w1};

fn smoke<E: Elem + Gk + Prototypes>() {
    let protos = E::prototypes();
    let mut isas: Vec<&str> = protos.iter().map(|p| p.isa).collect();
    isas.dedup();
    println!(
        "{}: prototypes run on ISAs {:?} (available: {:?})",
        E::NAME,
        isas,
        machine::isas()
    );
    assert!(!protos.is_empty() || machine::isas() == ["scalar"]);
    let mut pool = w1::<E>(24, 1);
    let set = &mut pool[0];
    let oracle = Oracle::new(set);
    for p in &protos {
        for form in [Form::Gathered, Form::PerPair] {
            set.clear();
            call(&Variant::Proto(*p), set, form, p.grad);
            let acc = accuracy(&oracle, &set.pot, p.grad.then_some(&set.grad[..]));
            assert!(
                acc.passes::<E>(set.sources.len(), p.flavour == "relaxed"),
                "{} {:?}: {acc:?}",
                p.name,
                form
            );
        }
        // Targets in lanes: per-pair calls give the bits of one gathered call.
        if p.order == Order::Til {
            set.clear();
            call(&Variant::Proto(*p), set, Form::PerPair, p.grad);
            let (pot, grad) = (set.pot.clone(), set.grad.clone());
            set.clear();
            call(&Variant::Proto(*p), set, Form::Gathered, p.grad);
            let bits = |v: &[E]| v.iter().map(|&x| x.to_f64().to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&pot), bits(&set.pot), "{}", p.name);
            assert_eq!(
                bits(grad.as_flattened()),
                bits(set.grad.as_flattened()),
                "{}",
                p.name
            );
        }
    }
    println!("{}: {} prototypes checked", E::NAME, protos.len());
}

#[test]
fn prototypes_match_direct_sum_f32() {
    smoke::<f32>();
}

#[test]
fn prototypes_match_direct_sum_f64() {
    smoke::<f64>();
}

/// The formulations the kernels use ("best" and "gk") stay within 4 u_T on a subset;
/// the report checks f32 exhaustively and f64 on 10⁷ samples.
#[test]
fn kernel_rsqrt_within_contract() {
    let (n32, n64) = (f32_names(), f64_names());
    for c in f32::candidates().iter().filter(|c| n32.contains(&c.name)) {
        let st = crate::rsqrt_study::errors_f32(c, 997);
        println!("{} f32 {}: {:.2} u", c.isa, c.name, st.max_u);
        assert!(st.max_u <= 4.0, "{} f32 {}: {st:?}", c.isa, c.name);
    }
    for c in f64::candidates().iter().filter(|c| n64.contains(&c.name)) {
        let st = crate::rsqrt_study::errors_f64(c, 20_000);
        println!("{} f64 {}: {:.2} u", c.isa, c.name, st.max_u);
        assert!(st.max_u <= 4.0, "{} f64 {}: {st:?}", c.isa, c.name);
    }
}

fn f32_names() -> Vec<&'static str> {
    f32::prototypes()
        .iter()
        .filter(|p| p.flavour == "best" || p.flavour == "gk")
        .map(|p| p.rsqrt)
        .collect()
}

fn f64_names() -> Vec<&'static str> {
    f64::prototypes()
        .iter()
        .filter(|p| p.flavour == "best" || p.flavour == "gk")
        .map(|p| p.rsqrt)
        .collect()
}
