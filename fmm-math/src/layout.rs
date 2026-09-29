//! Index layout of real storage (CONVENTIONS §3.6).

use core::ops::Range;

/// Index layout of real storage for degrees 0 to `p` (CONVENTIONS §3.6).
///
/// Degree n and order m, −n ≤ m ≤ n, live at index n² + n + m, so degree n occupies
/// the contiguous range n²..(n + 1)² and the whole layout (p + 1)² slots. Slot m = 0
/// holds the real value, slot +m (m > 0) the real part and slot −m the imaginary part
/// of the order-m quantity. The same layout stores basis values, their gradients and
/// expansion coefficients.
///
/// ```
/// use nd_fmm_math::Layout;
///
/// let layout = Layout::new(2);
/// assert_eq!(layout.len(), 9);
/// assert_eq!(layout.idx(2, -1), 5);
/// assert_eq!(layout.nm(5), (2, -1));
/// assert_eq!(layout.degree(1), 1..4);
///
/// let sizes: Vec<usize> = layout.degrees().map(|(_, range)| range.len()).collect();
/// assert_eq!(sizes, [1, 3, 5]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    p: usize,
}

impl Layout {
    /// Layout for degrees 0 to `p`.
    #[inline]
    pub const fn new(p: usize) -> Self {
        Self { p }
    }

    /// Largest degree p.
    #[inline]
    pub const fn p(self) -> usize {
        self.p
    }

    /// Number of slots, (p + 1)².
    #[inline]
    pub const fn len(self) -> usize {
        (self.p + 1) * (self.p + 1)
    }

    /// Always `false`: every layout holds at least degree 0.
    #[inline]
    pub const fn is_empty(self) -> bool {
        false
    }

    /// Index n² + n + m of degree `n` and order `m`.
    ///
    /// Requires n ≤ p and |m| ≤ n; checked in debug builds only.
    #[inline]
    pub const fn idx(self, n: usize, m: isize) -> usize {
        debug_assert!(n <= self.p && m.unsigned_abs() <= n);
        (n * n + n).wrapping_add_signed(m)
    }

    /// Degree and order `(n, m)` of index `i`; the inverse of [`idx`](Self::idx).
    ///
    /// Requires i < (p + 1)²; checked in debug builds only.
    #[inline]
    pub const fn nm(self, i: usize) -> (usize, isize) {
        debug_assert!(i < self.len());
        let n = i.isqrt();
        (n, (i - n * n) as isize - n as isize)
    }

    /// Index range n²..(n + 1)² of degree `n`, orders −n to n in increasing order.
    ///
    /// Requires n ≤ p; checked in debug builds only.
    #[inline]
    pub const fn degree(self, n: usize) -> Range<usize> {
        debug_assert!(n <= self.p);
        n * n..(n + 1) * (n + 1)
    }

    /// Iterates over the degrees 0 to p in storage order, yielding each degree with its
    /// index range as returned by [`degree`](Self::degree).
    pub fn degrees(self) -> impl ExactSizeIterator<Item = (usize, Range<usize>)> {
        (0..self.p + 1).map(move |n| (n, self.degree(n)))
    }
}
