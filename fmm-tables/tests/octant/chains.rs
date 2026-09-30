//! Physical chains (as in Phase 1 T5) and compositions: P2M then the M2M table equals
//! P2M at the parent, L2P after the L2L table equals L2P of the original local, and
//! two table steps equal one `direct` translation over two levels.

use nd_fmm_ref::{Frame, Workspace, direct, leaf};
use nd_fmm_tables::geometry::OCTANT_COUNT;

use crate::common::{
    CHAIN_TOL, DEBUG_DEGREES, DYADIC, Domain, Kind, SplitMix64, Worst, apply, child_index,
    degree_error, l2l, l2p_terms, len, m2m, place, random_coefficients, terms,
};

/// Parent levels of the chain tests, with the canonical parent (0, 1) as `None`.
const CHAIN_LEVELS: [Option<u32>; 4] = [None, Some(0), Some(7), Some(14)];

/// The parent frame of a chain test: the canonical (0, 1), or a random box on `level`
/// of the dyadic domain, with its index.
fn parent(level: Option<u32>, rng: &mut SplitMix64) -> (Frame<f64>, Option<(u32, [u64; 3])>) {
    match level {
        None => (Frame::new([0.0; 3], 1.0), None),
        Some(l) => {
            let index = Domain::random_index(l, rng);
            (DYADIC.frame(l, index), Some((l, index)))
        }
    }
}

/// Child `o` of `parent`: from the box index on a dyadic level, or (½ s_o, ½) for the
/// canonical parent.
fn child(parent: &Frame<f64>, at: Option<(u32, [u64; 3])>, o: usize) -> Frame<f64> {
    match at {
        Some((l, index)) => DYADIC.frame(l + 1, child_index(index, o)),
        None => {
            let h = parent.radius / 2.0;
            let s = nd_fmm_tables::geometry::octant_direction(o);
            Frame::new(
                core::array::from_fn(|a| parent.centre[a] + h * s[a] as f64),
                h,
            )
        }
    }
}

#[test]
fn m2m_table_after_p2m_equals_p2m_at_the_parent() {
    // Error measure: multipole coefficients per degree in the orthonormal weighting Nₘ,
    // relative to the terms |Aᵢₖ xₖ| of the table application to the child P2M
    // (`common`).
    let mut rng = SplitMix64::new(0x0c7a_0101);
    let mut ws = Workspace::new(DEBUG_DEGREES[DEBUG_DEGREES.len() - 1]);
    let worst = Worst::new("P2M, M2M table vs P2M at parent, per degree (terms)");
    for level in CHAIN_LEVELS {
        let (parent, at) = parent(level, &mut rng);
        for o in 0..OCTANT_COUNT {
            let child = child(&parent, at, o);
            let sources: Vec<[f64; 3]> = (0..10).map(|_| place(&child, rng.in_cube())).collect();
            let charges: Vec<f64> = (0..10).map(|_| rng.charge()).collect();
            for p in DEBUG_DEGREES {
                let mut at_child = vec![0.0; len(p)];
                leaf::p2m(p, &child, &sources, &charges, &mut ws, &mut at_child);
                let got = apply(m2m(p).matrices(), o, &at_child);
                let mut want = vec![0.0; len(p)];
                leaf::p2m(p, &parent, &sources, &charges, &mut ws, &mut want);
                let scale = terms(m2m(p).matrices(), o, &at_child);
                let e = worst.update(degree_error(Kind::Multipole, p, &got, &want, &scale));
                assert!(e <= CHAIN_TOL, "{level:?}, o = {o}, p = {p}: {e:.3e}");
            }
        }
    }
}

#[test]
fn l2p_after_the_l2l_table_equals_l2p_of_the_original_local() {
    // Error measure: potential at targets inside the child box, relative to the sum of
    // the L2P terms of the table's terms |Aᵢₖ xₖ| in the child frame and of the L2P
    // terms of the original local in the parent frame.
    let mut rng = SplitMix64::new(0x0c7a_0102);
    let mut ws = Workspace::new(DEBUG_DEGREES[DEBUG_DEGREES.len() - 1]);
    let worst = Worst::new("L2L table, L2P vs L2P of the original, relative to terms");
    for level in CHAIN_LEVELS {
        let (parent, at) = parent(level, &mut rng);
        for o in 0..OCTANT_COUNT {
            let child = child(&parent, at, o);
            let targets: Vec<[f64; 3]> = (0..10).map(|_| place(&child, rng.in_cube())).collect();
            for p in DEBUG_DEGREES {
                let local = random_coefficients(Kind::Local, p, &mut rng);
                let translated = apply(l2l(p).matrices(), o, &local);
                let mut got = vec![0.0; targets.len()];
                leaf::l2p(p, &child, &translated, &targets, &mut ws, &mut got, None);
                let mut want = vec![0.0; targets.len()];
                leaf::l2p(p, &parent, &local, &targets, &mut ws, &mut want, None);
                let tau = terms(l2l(p).matrices(), o, &local);
                let magnitudes: Vec<f64> = local.iter().map(|v| v.abs()).collect();
                for ((g, w), &x) in got.iter().zip(&want).zip(&targets) {
                    let scale =
                        l2p_terms(p, &child, &tau, x) + l2p_terms(p, &parent, &magnitudes, x);
                    let e = worst.update((g - w).abs() / scale);
                    assert!(e <= CHAIN_TOL, "{level:?}, o = {o}, p = {p}: {e:.3e}");
                }
            }
        }
    }
}

#[test]
fn two_table_steps_equal_direct_over_two_levels() {
    // M2M table (o₂) after M2M table (o₁) against `direct::m2m` from the grandchild o₁
    // of child o₂ to the grandparent; L2L table (o₂) after L2L table (o₁) against
    // `direct::l2l` from the grandparent to the grandchild o₂ of child o₁.
    //
    // Error measure: coefficients per degree in the orthonormal weighting (Nₘ for M2M,
    // Nₘ/Sₘ for L2L), relative to the terms |A₂| (|A₁| |x|) of the two table steps.
    let mut rng = SplitMix64::new(0x0c7a_0103);
    let mut ws = Workspace::new(DEBUG_DEGREES[DEBUG_DEGREES.len() - 1]);
    let worst_m2m = Worst::new("M2M table composition vs direct, per degree (terms)");
    let worst_l2l = Worst::new("L2L table composition vs direct, per degree (terms)");
    for level in [None, Some(0), Some(9), Some(14)] {
        let (top, at) = parent(level, &mut rng);
        for p in DEBUG_DEGREES {
            let (a, b) = (m2m(p).matrices(), l2l(p).matrices());
            for o1 in 0..OCTANT_COUNT {
                for o2 in 0..OCTANT_COUNT {
                    // M2M: grandchild o₁ of child o₂ of `top`.
                    let middle = child(&top, at, o2);
                    let middle_at = at.map(|(l, index)| (l + 1, child_index(index, o2)));
                    let bottom = child(&middle, middle_at, o1);
                    let input = random_coefficients(Kind::Multipole, p, &mut rng);
                    let got = apply(a, o2, &apply(a, o1, &input));
                    let mut want = vec![0.0; len(p)];
                    direct::m2m(p, &bottom, &top, &mut ws, &input, &mut want);
                    let scale = terms(a, o2, &terms(a, o1, &input));
                    let e = worst_m2m.update(degree_error(Kind::Multipole, p, &got, &want, &scale));
                    assert!(
                        e <= CHAIN_TOL,
                        "M2M {level:?}, o₁ = {o1}, o₂ = {o2}, p = {p}: {e:.3e}"
                    );

                    // L2L: grandchild o₂ of child o₁ of `top`.
                    let middle = child(&top, at, o1);
                    let middle_at = at.map(|(l, index)| (l + 1, child_index(index, o1)));
                    let bottom = child(&middle, middle_at, o2);
                    let input = random_coefficients(Kind::Local, p, &mut rng);
                    let got = apply(b, o2, &apply(b, o1, &input));
                    let mut want = vec![0.0; len(p)];
                    direct::l2l(p, &top, &bottom, &mut ws, &input, &mut want);
                    let scale = terms(b, o2, &terms(b, o1, &input));
                    let e = worst_l2l.update(degree_error(Kind::Local, p, &got, &want, &scale));
                    assert!(
                        e <= CHAIN_TOL,
                        "L2L {level:?}, o₁ = {o1}, o₂ = {o2}, p = {p}: {e:.3e}"
                    );
                }
            }
        }
    }
}
