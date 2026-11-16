//! The rule of the Rotation M2L by pairs (Helmholtz CH4.0i, `rcr_by_pairs`): by pairs under
//! `RcrSplit::Pairs` always, under `RcrSplit::Boxes` never, under `RcrSplit::Auto` on a level
//! of degree at least `RCR_PAIR_MIN_DEGREE` at every thread count, and of degree at least
//! `RCR_PAIR_FEW_DEGREE` on a pool of more than one thread where the level's targets with a
//! V row are fewer than `RCR_PAIR_SHARE` times the threads. The level call's bits are
//! `nd-fmm-exec`'s `tests/helmholtz_fmm.rs` (`check_rcr_split`): by pairs needs a plan, so MPI.

use nd_fmm_exec::helmholtz::{
    RCR_PAIR_FEW_DEGREE, RCR_PAIR_MIN_DEGREE, RCR_PAIR_SHARE, RcrSplit, rcr_by_pairs,
};

#[test]
fn the_rule_of_the_split_by_pairs() {
    for degree in [0, 7, 15, 29, 30, 47, 48, 72, 135] {
        for threads in [1, 2, 40, 72] {
            for targets in [0, 1, 63, 64, 159, 160, 287, 288, 512, 4096] {
                assert!(rcr_by_pairs(RcrSplit::Pairs, degree, targets, threads));
                assert!(!rcr_by_pairs(RcrSplit::Boxes, degree, targets, threads));
                assert_eq!(
                    rcr_by_pairs(RcrSplit::Auto, degree, targets, threads),
                    degree >= RCR_PAIR_MIN_DEGREE
                        || (threads > 1
                            && degree >= RCR_PAIR_FEW_DEGREE
                            && targets < RCR_PAIR_SHARE * threads),
                    "p = {degree}, {targets} targets on {threads} threads"
                );
            }
        }
    }
    // W1 at kD = 500: level 2 (64 boxes, p = 131) and level 3 (512, p = 72) by pairs at every
    // thread count; a level of p = 35 and 64 boxes on 40 threads, not serially; low degrees
    // never.
    for threads in [1, 40, 72] {
        assert!(rcr_by_pairs(RcrSplit::Auto, 131, 64, threads));
        assert!(rcr_by_pairs(RcrSplit::Auto, 72, 512, threads));
        assert!(!rcr_by_pairs(RcrSplit::Auto, 8, 64, threads));
        assert!(!rcr_by_pairs(RcrSplit::Auto, 24, 512, threads));
    }
    assert!(rcr_by_pairs(RcrSplit::Auto, 35, 64, 40) && !rcr_by_pairs(RcrSplit::Auto, 35, 64, 1));
    assert_eq!(RcrSplit::default(), RcrSplit::Auto);
}
