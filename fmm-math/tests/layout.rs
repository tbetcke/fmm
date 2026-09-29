//! `Layout` index maps (CONVENTIONS §3.6): idx and nm are inverse to each other for
//! p ≤ 40, and the degrees tile the storage in order.

use nd_fmm_math::Layout;
use proptest::prelude::*;

fn degree_and_order() -> impl Strategy<Value = (usize, usize, isize)> {
    (0usize..=40)
        .prop_flat_map(|p| (Just(p), 0..=p))
        .prop_flat_map(|(p, n)| (Just(p), Just(n), -(n as isize)..=n as isize))
}

fn storage_index() -> impl Strategy<Value = (usize, usize)> {
    (0usize..=40).prop_flat_map(|p| (Just(p), 0..(p + 1) * (p + 1)))
}

proptest! {
    #[test]
    fn nm_inverts_idx((p, n, m) in degree_and_order()) {
        let layout = Layout::new(p);
        let i = layout.idx(n, m);
        prop_assert!(i < layout.len());
        prop_assert_eq!(layout.nm(i), (n, m));
    }

    #[test]
    fn idx_inverts_nm((p, i) in storage_index()) {
        let layout = Layout::new(p);
        let (n, m) = layout.nm(i);
        prop_assert!(n <= p && m.unsigned_abs() <= n);
        prop_assert_eq!(layout.idx(n, m), i);
    }
}

#[test]
fn degrees_tile_storage_in_order() {
    for p in 0..=40 {
        let layout = Layout::new(p);
        assert_eq!(layout.len(), (p + 1) * (p + 1));
        assert_eq!(layout.degrees().len(), p + 1);
        let mut next = 0;
        for (n, range) in layout.degrees() {
            assert_eq!(range, layout.degree(n));
            assert_eq!(range.start, next);
            assert_eq!(range.len(), 2 * n + 1);
            for (i, m) in range.clone().zip(-(n as isize)..=n as isize) {
                assert_eq!(layout.idx(n, m), i);
            }
            next = range.end;
        }
        assert_eq!(next, layout.len());
    }
}
