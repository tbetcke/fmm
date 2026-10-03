//! The P2P kernel of `LaplaceOperator` (Phase 3S / T6, C3S.5): `Auto` is the default and
//! runs the detected ISA, every available ISA and the reference are selectable, an
//! unavailable ISA is rejected, and the text form of `P2pChoice` round-trips. Error
//! measure: exact equality.

use nd_fmm_exec::operator::{Isa, IsaUnavailable, LaplaceOperator, P2pChoice, UnknownP2pChoice};
use nd_fmm_exec::tables::M2lStrategy;

use crate::common::{operator, p2p_choices, tables};

#[test]
fn auto_is_the_default_and_runs_the_detected_isa() {
    assert_eq!(P2pChoice::default(), P2pChoice::Auto);
    let detected = P2pChoice::Isa(Isa::detect());
    assert_eq!(P2pChoice::Auto.resolve(), Ok(detected));
    let op = operator(M2lStrategy::Rotation, 2, 4);
    assert_eq!(op.p2p_kernel(), detected, "`new` uses Auto");
    let auto = op.clone().with_p2p(P2pChoice::Auto).unwrap();
    assert_eq!(auto.p2p_kernel(), detected);
    eprintln!("Auto runs {detected} on this machine");
}

#[test]
fn every_available_kernel_is_selectable() {
    for choice in p2p_choices("selectable") {
        assert_eq!(choice.resolve(), Ok(choice));
        let op = LaplaceOperator::new(tables(M2lStrategy::Rotation, 2).clone(), false, 4);
        assert_eq!(op.with_p2p(choice).unwrap().p2p_kernel(), choice);
    }
}

#[test]
fn an_unavailable_isa_is_rejected() {
    // NEON and AVX2 belong to different architectures, so one of them is unavailable.
    let unavailable: Vec<Isa> = Isa::all().filter(|isa| !isa.is_available()).collect();
    assert!(!unavailable.is_empty());
    for isa in unavailable {
        let choice = P2pChoice::Isa(isa);
        assert_eq!(choice.resolve(), Err(IsaUnavailable { isa }));
        let error = operator(M2lStrategy::Rotation, 2, 4).with_p2p(choice).err();
        assert_eq!(error, Some(IsaUnavailable { isa }));
        eprintln!("{isa}: rejected");
    }
}

#[test]
fn the_text_form_round_trips() {
    let choices = [P2pChoice::Auto, P2pChoice::Reference]
        .into_iter()
        .chain(Isa::all().map(P2pChoice::Isa));
    for choice in choices {
        let text = choice.to_string();
        assert_eq!(text.parse::<P2pChoice>(), Ok(choice), "{text}");
    }
    assert_eq!("neon".parse(), Ok(P2pChoice::Isa(Isa::Neon)));
    assert_eq!("avx2".parse(), Ok(P2pChoice::Isa(Isa::Avx2)));
    let error = "sse2".parse::<P2pChoice>().unwrap_err();
    assert_eq!(
        error,
        UnknownP2pChoice {
            text: "sse2".into()
        }
    );
    eprintln!("{error}");
}
