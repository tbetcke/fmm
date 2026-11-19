//! The reflection classes of `nd_fmm_ref::helmholtz::diagonal` as elements of the cube's group
//! (review 2026-11-19, C5; Helmholtz CH4.1b).
//!
//! The diagonal form groups the 316 M2L offsets into 34 classes under the 16 symmetries that
//! flip x, y and z and swap x and y (`nd_fmm_ref::helmholtz::diagonal::{classes, Symmetry}`,
//! proposal §5.4), the symmetries of its signature grid. They form a subgroup of the 48
//! elements of O_h with which [`crate::symmetry`] groups the same offsets into the 16 classes of
//! CONVENTIONS §3.12. [`signed_permutation`] names each as an element of O_h, so both class
//! sets are read on one group (the tests: the two offset sets are one, each of the 34 classes
//! lies in one of the 16).

use nd_fmm_ref::helmholtz::diagonal::Symmetry;

use crate::symmetry::SignedPermutation;

/// The element P of O_h with P o = S o for every offset o, S ref's flips then swap.
///
/// ```
/// use nd_fmm_ref::helmholtz::diagonal::class_of;
/// use nd_fmm_tables::helmholtz::diagonal::signed_permutation;
///
/// let (representative, s) = class_of([-3, 1, -2]);
/// let p = signed_permutation(s);
/// assert_eq!(p.apply([-3, 1, -2]), representative.map(i64::from));
/// ```
pub fn signed_permutation(symmetry: Symmetry) -> SignedPermutation {
    let unit = |a: usize| core::array::from_fn::<i64, 3, _>(|b| i64::from(a == b));
    SignedPermutation::all()
        .into_iter()
        .find(|g| (0..3).all(|a| g.apply(unit(a)) == symmetry.apply(unit(a))))
        .expect("flips and a swap are a signed permutation")
}
