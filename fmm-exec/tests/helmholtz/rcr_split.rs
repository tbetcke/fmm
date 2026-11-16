//! The rule of the Rotation M2L by pairs (Helmholtz CH4.0i, `rcr_by_pairs`): by pairs under
//! `RcrSplit::Pairs` always, under `RcrSplit::Boxes` never, under `RcrSplit::Auto` on a pool
//! of more than one thread where the level's targets with a V row are fewer than
//! `RCR_PAIR_SHARE` times the threads. The level call's bits are `nd-fmm-exec`'s
//! `tests/helmholtz_fmm.rs` (`check_rcr_split`): by pairs needs a plan, so MPI.

use nd_fmm_exec::helmholtz::{RCR_PAIR_SHARE, RcrSplit, rcr_by_pairs};

#[test]
fn the_rule_of_the_split_by_pairs() {
    for threads in [1, 2, 40, 72] {
        for targets in [0, 1, 63, 64, 159, 160, 287, 288, 512, 4096] {
            assert!(rcr_by_pairs(RcrSplit::Pairs, targets, threads));
            assert!(!rcr_by_pairs(RcrSplit::Boxes, targets, threads));
            assert_eq!(
                rcr_by_pairs(RcrSplit::Auto, targets, threads),
                threads > 1 && targets < RCR_PAIR_SHARE * threads,
                "{targets} targets on {threads} threads"
            );
        }
    }
    // Level 2 (64 boxes) on Kathleen's 40 and locust's 72 threads; never serially.
    assert!(rcr_by_pairs(RcrSplit::Auto, 64, 40) && rcr_by_pairs(RcrSplit::Auto, 64, 72));
    assert!(!rcr_by_pairs(RcrSplit::Auto, 64, 1));
    assert_eq!(RcrSplit::default(), RcrSplit::Auto);
}
