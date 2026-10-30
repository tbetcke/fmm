//! Argument checks: each kind of length mismatch and a non-finite wavenumber panic.

use nd_fmm_ref::helmholtz::p2p;

use crate::common::C;

const K: C = C::new(1.0, 0.5);
const ONE: [C; 1] = [C::new(1.0, 0.0)];
const X: [[f64; 3]; 1] = [[1.0, 2.0, 3.0]];

#[test]
#[should_panic(expected = "`sources` and `charges`")]
fn p2p_charges_length() {
    p2p::p2p(K, &X, &[ONE[0], ONE[0]], &X, &mut [C::new(0.0, 0.0)], None);
}

#[test]
#[should_panic(expected = "`targets` and `potential`")]
fn p2p_potential_length() {
    p2p::p2p(K, &X, &ONE, &X, &mut [], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient`")]
fn p2p_gradient_length() {
    let mut g = [[C::new(0.0, 0.0); 3]; 2];
    p2p::p2p(K, &X, &ONE, &X, &mut [C::new(0.0, 0.0)], Some(&mut g));
}

#[test]
#[should_panic(expected = "`sources` and `charges`")]
fn direct_sum_charges_length() {
    p2p::direct_sum(K, &X, &[], &X, &mut [C::new(0.0, 0.0)], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient`")]
fn direct_sum_gradient_length() {
    let mut g: [[C; 3]; 0] = [];
    p2p::direct_sum(K, &X, &ONE, &X, &mut [C::new(0.0, 0.0)], Some(&mut g));
}

#[test]
#[should_panic(expected = "`targets` and `potential`")]
fn direct_sum_potential_length() {
    p2p::direct_sum(K, &X, &ONE, &X, &mut [], None);
}

#[test]
#[should_panic(expected = "`targets` and `gradient`")]
fn yukawa_gradient_length() {
    let mut g = [[0.0; 3]; 2];
    p2p::p2p_yukawa(1.0, &X, &[1.0], &X, &mut [0.0], Some(&mut g));
}

#[test]
#[should_panic(expected = "`targets` and `potential`")]
fn yukawa_potential_length() {
    p2p::p2p_yukawa(1.0, &X, &[1.0], &X, &mut [0.0, 0.0], None);
}

#[test]
#[should_panic(expected = "`sources` and `charges`")]
fn direct_sum_yukawa_charges_length() {
    p2p::direct_sum_yukawa(1.0, &X, &[], &X, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn p2p_nan_k() {
    p2p::p2p(
        C::new(f64::NAN, 0.0),
        &X,
        &ONE,
        &X,
        &mut [C::new(0.0, 0.0)],
        None,
    );
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn direct_sum_infinite_k() {
    p2p::direct_sum(
        C::new(1.0, f64::INFINITY),
        &X,
        &ONE,
        &X,
        &mut [C::new(0.0, 0.0)],
        None,
    );
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn yukawa_nan_kappa() {
    p2p::p2p_yukawa(f64::NAN, &X, &[1.0], &X, &mut [0.0], None);
}

#[test]
#[should_panic(expected = "finite wavenumber")]
fn direct_sum_yukawa_infinite_kappa() {
    p2p::direct_sum_yukawa(f64::INFINITY, &X, &[1.0], &X, &mut [0.0], None);
}

mod leaf {
    //! The leaf operators (Helmholtz CH1.3b) and `Level`.

    use nd_fmm_ref::Frame;
    use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};

    use crate::common::C;

    const P: usize = 2;
    const LEN: usize = (P + 1) * (P + 1);
    const Y: [[f64; 3]; 1] = [[0.1, 0.2, 0.3]];
    const Q: [C; 1] = [C::new(1.0, 0.0)];

    fn level() -> Level<f64> {
        Level::new(C::new(1.0, 0.5), 0.5)
    }

    fn yukawa() -> Level<f64> {
        Level::new(C::new(0.0, 2.0), 0.5)
    }

    fn frame() -> Frame<f64> {
        Frame::new([0.0; 3], 0.5)
    }

    #[test]
    #[should_panic(expected = "nonzero with Re k ≥ 0 and Im k ≥ 0")]
    fn level_zero_k() {
        Level::new(C::new(0.0, 0.0), 0.5);
    }

    #[test]
    #[should_panic(expected = "nonzero with Re k ≥ 0 and Im k ≥ 0")]
    fn level_negative_real_part() {
        Level::new(C::new(-1.0, 0.5), 0.5);
    }

    #[test]
    #[should_panic(expected = "nonzero with Re k ≥ 0 and Im k ≥ 0")]
    fn level_growing_kernel() {
        Level::new(C::new(1.0, -0.5), 0.5);
    }

    #[test]
    #[should_panic(expected = "the wavenumber must be finite")]
    fn level_infinite_k() {
        Level::new(C::new(f64::INFINITY, 0.0), 0.5);
    }

    #[test]
    #[should_panic(expected = "the half-width must be finite and positive")]
    fn level_zero_half_width() {
        Level::new(C::new(1.0, 0.0), 0.0);
    }

    #[test]
    #[should_panic(expected = "multipole has length 9, expected 18 for p = 2")]
    fn p2m_coefficient_length() {
        let mut m = [0.0; LEN];
        leaf::p2m(
            P,
            &level(),
            &frame(),
            &Y,
            &Q,
            &mut Workspace::new(P),
            &mut m,
        );
    }

    #[test]
    #[should_panic(expected = "`sources` and `charges`")]
    fn p2l_charges_length() {
        let mut l = [0.0; 2 * LEN];
        let y = [[2.0, 0.0, 0.0]];
        leaf::p2l(
            P,
            &level(),
            &frame(),
            &y,
            &[],
            &mut Workspace::new(P),
            &mut l,
        );
    }

    #[test]
    #[should_panic(expected = "is not the level's half-width")]
    fn frame_radius_differs() {
        let mut m = [0.0; 2 * LEN];
        let frame = Frame::new([0.0; 3], 0.25);
        leaf::p2m(P, &level(), &frame, &Y, &Q, &mut Workspace::new(P), &mut m);
    }

    #[test]
    #[should_panic(expected = "Workspace built for p = 1 is too small for p = 2")]
    fn workspace_too_small() {
        let mut m = [0.0; 2 * LEN];
        leaf::p2m(
            P,
            &level(),
            &frame(),
            &Y,
            &Q,
            &mut Workspace::new(1),
            &mut m,
        );
    }

    #[test]
    #[should_panic(expected = "`targets` and `potential`")]
    fn l2p_potential_length() {
        let l = [0.0; 2 * LEN];
        leaf::l2p(
            P,
            &level(),
            &frame(),
            &l,
            &Y,
            &mut Workspace::new(P),
            &mut [],
            None,
        );
    }

    #[test]
    #[should_panic(expected = "`targets` and `gradient`")]
    fn m2p_gradient_length() {
        let m = [0.0; 2 * LEN];
        let x = [[2.0, 0.0, 0.0]];
        let mut phi = [C::new(0.0, 0.0)];
        let mut ws = Workspace::new(P);
        leaf::m2p(
            P,
            &level(),
            &frame(),
            &m,
            &x,
            &mut ws,
            &mut phi,
            Some(&mut []),
        );
    }

    #[test]
    #[should_panic(expected = "hankel_h needs z ≠ 0")]
    fn p2l_source_at_the_centre() {
        let mut l = [0.0; 2 * LEN];
        let y = [[0.0; 3]];
        leaf::p2l(
            P,
            &level(),
            &frame(),
            &y,
            &Q,
            &mut Workspace::new(P),
            &mut l,
        );
    }

    #[test]
    #[should_panic(expected = "modified_k needs a finite x > 0")]
    fn m2p_yukawa_target_at_the_centre() {
        let m = [0.0; LEN];
        let mut phi = [0.0];
        let mut ws = Workspace::new(P);
        leaf::m2p_yukawa(
            P,
            &yukawa(),
            &frame(),
            &m,
            &[[0.0; 3]],
            &mut ws,
            &mut phi,
            None,
        );
    }

    #[test]
    #[should_panic(expected = "Yukawa's real path needs k = iκ (Re k = 0)")]
    fn yukawa_on_a_complex_level() {
        let mut m = [0.0; LEN];
        leaf::p2m_yukawa(
            P,
            &level(),
            &frame(),
            &Y,
            &[1.0],
            &mut Workspace::new(P),
            &mut m,
        );
    }

    #[test]
    #[should_panic(expected = "local has length 18, expected 9 for p = 2")]
    fn yukawa_coefficient_length() {
        let l = [0.0; 2 * LEN];
        let mut phi = [0.0];
        let mut ws = Workspace::new(P);
        leaf::l2p_yukawa(P, &yukawa(), &frame(), &l, &Y, &mut ws, &mut phi, None);
    }
}

/// The translations (Helmholtz CH1.3c): geometry, tables, degrees and levels.
mod translations {
    use nd_fmm_ref::Frame;
    use nd_fmm_ref::helmholtz::{Level, Workspace, direct, rcr};

    use crate::common::C;
    use crate::translation_common::{CHILD, child, neighbour, parent};

    const P: usize = 2;
    const LEN: usize = (P + 1) * (P + 1);
    const K: C = C::new(1.0, 0.5);

    fn levels() -> (Level<f64>, Level<f64>) {
        (Level::new(K, CHILD), Level::new(K, 2.0 * CHILD))
    }

    fn yukawa() -> (Level<f64>, Level<f64>) {
        let k = C::new(0.0, 2.0);
        (Level::new(k, CHILD), Level::new(k, 2.0 * CHILD))
    }

    #[test]
    #[should_panic(expected = "the table has degrees 2 → 3, expected 2 → 2")]
    fn rcr_table_degrees() {
        let (c, p) = levels();
        let table = rcr::m2m_table(P, P + 1, &c);
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P + 1);
        rcr::m2m(
            P,
            P,
            &c,
            &p,
            &child(0),
            &parent(),
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the frames are not a parent and one of its children")]
    fn rcr_m2m_not_a_child() {
        let (c, p) = levels();
        let table = rcr::m2m_table(P, P, &c);
        let mut out = [0.0; 2 * LEN];
        let far = Frame::new([3.0, 0.0, 0.0], CHILD);
        let mut ws = Workspace::new(P);
        rcr::m2m(
            P,
            P,
            &c,
            &p,
            &far,
            &parent(),
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the parent's half-width must be twice the child's")]
    fn rcr_l2l_not_a_parent() {
        let (c, _) = levels();
        let table = rcr::l2l_table(P, P, &c);
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        let same = child(1);
        rcr::l2l(
            P,
            P,
            &c,
            &c,
            &same,
            &child(0),
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the two levels have different wavenumbers")]
    fn rcr_m2m_different_k() {
        let (c, _) = levels();
        let p = Level::new(C::new(2.0, 0.5), 2.0 * CHILD);
        let table = rcr::m2m_table(P, P, &c);
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        rcr::m2m(
            P,
            P,
            &c,
            &p,
            &child(0),
            &parent(),
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the M2L needs distinct centres")]
    fn rcr_m2l_zero_shift() {
        let (_, l) = levels();
        let table = rcr::m2l_table(P, P, &l, &parent(), &neighbour([2, 0, 0]));
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        rcr::m2l(
            P,
            P,
            &l,
            &parent(),
            &parent(),
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "Workspace built for p = 1 is too small for p = 2")]
    fn rcr_workspace_degree() {
        let (_, l) = levels();
        let to = neighbour([2, 0, 0]);
        let table = rcr::m2l_table(P, P, &l, &parent(), &to);
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P - 1);
        rcr::m2l(
            P,
            P,
            &l,
            &parent(),
            &to,
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "multipole has length 9, expected 18 for p = 2")]
    fn rcr_coefficient_length() {
        let (_, l) = levels();
        let to = neighbour([2, 0, 0]);
        let table = rcr::m2l_table(P, P, &l, &parent(), &to);
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        rcr::m2l(
            P,
            P,
            &l,
            &parent(),
            &to,
            &table,
            &mut ws,
            &[0.0; LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the frame's radius 0.25 is not the level's half-width 0.5")]
    fn rcr_frame_of_another_level() {
        let (_, l) = levels();
        let to = neighbour([2, 0, 0]);
        let table = rcr::m2l_table(P, P, &l, &parent(), &to);
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        rcr::m2l(
            P,
            P,
            &l,
            &child(0),
            &to,
            &table,
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "Yukawa's real path needs k = iκ (Re k = 0)")]
    fn rcr_yukawa_on_a_complex_level() {
        let (c, p) = levels();
        let table = rcr::m2m_table(P, P, &c);
        let mut out = [0.0; LEN];
        let mut ws = Workspace::new(P);
        rcr::m2m_yukawa(
            P,
            P,
            &c,
            &p,
            &child(0),
            &parent(),
            &table,
            &mut ws,
            &[0.0; LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the child's sphere must lie inside the parent's")]
    fn direct_m2m_outside() {
        let (c, p) = levels();
        let mut out = [0.0; 2 * LEN];
        let far = Frame::new([1.0, 0.0, 0.0], CHILD);
        let mut ws = Workspace::new(P);
        direct::m2m(
            P,
            P,
            &c,
            &p,
            &far,
            &parent(),
            &mut ws,
            &[0.0; 2 * LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "the M2L needs boxes whose spheres do not overlap")]
    fn direct_m2l_overlap() {
        let (_, l) = levels();
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        let to = neighbour([1, 0, 0]);
        direct::m2l(P, P, &l, &parent(), &to, &mut ws, &[0.0; 2 * LEN], &mut out);
    }

    #[test]
    #[should_panic(expected = "Yukawa's real path needs k = iκ (Re k = 0)")]
    fn direct_yukawa_on_a_complex_level() {
        let (c, p) = levels();
        let mut out = [0.0; LEN];
        let mut ws = Workspace::new(P);
        direct::l2l_yukawa(
            P,
            P,
            &p,
            &c,
            &parent(),
            &child(0),
            &mut ws,
            &[0.0; LEN],
            &mut out,
        );
    }

    #[test]
    #[should_panic(expected = "local_out has length 18, expected 9 for p = 2")]
    fn direct_yukawa_coefficient_length() {
        let (c, p) = yukawa();
        let mut out = [0.0; 2 * LEN];
        let mut ws = Workspace::new(P);
        direct::l2l_yukawa(
            P,
            P,
            &p,
            &c,
            &parent(),
            &child(0),
            &mut ws,
            &[0.0; LEN],
            &mut out,
        );
    }
}
