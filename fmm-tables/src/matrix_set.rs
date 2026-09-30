//! A family of square matrices, stored column-major and contiguously.

use nd_fmm_math::RealScalar;

/// `count` square matrices A₀, …, A_{count−1} of order `n`, stored column-major and
/// contiguously, matrix after matrix.
///
/// Entry (r, c) of matrix Aᵢ sits at index i n² + r + c n of [`MatrixSet::as_slice`].
/// A table of one operator family is one set: the matrix of each octant or offset is a
/// real (p + 1)² × (p + 1)² matrix with output = A · input, rows and columns indexed in
/// the storage of CONVENTIONS §3.6, and each matrix is a column-major GEMM operand.
///
/// ```
/// use nd_fmm_tables::MatrixSet;
///
/// let mut set = MatrixSet::<f64>::zeros(2, 1);
/// set.matrix_mut(0).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]); // A = [1 3; 2 4]
/// let mut y = [10.0, 20.0];
/// set.apply(0, &[1.0, 1.0], &mut y);
/// assert_eq!(y, [14.0, 26.0]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixSet<T: RealScalar> {
    n: usize,
    count: usize,
    data: Vec<T>,
}

impl<T: RealScalar> MatrixSet<T> {
    /// Creates `count` zero matrices of order `n`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or if n² · `count` overflows `usize`.
    pub fn zeros(n: usize, count: usize) -> Self {
        assert!(n > 0, "MatrixSet order n must be positive");
        let len = n
            .checked_mul(n)
            .and_then(|n2| n2.checked_mul(count))
            .expect("MatrixSet size n^2 * count overflows usize");
        Self {
            n,
            count,
            data: vec![T::zero(); len],
        }
    }

    /// Returns the order n of every matrix.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Returns the number of matrices.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns all entries, n² · `count` reals: matrix after matrix, each column-major.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns matrix Aᵢ, n² reals in column-major order.
    ///
    /// # Panics
    ///
    /// Panics if `i >= count`.
    #[inline]
    pub fn matrix(&self, i: usize) -> &[T] {
        let range = self.matrix_range(i);
        &self.data[range]
    }

    /// Returns matrix Aᵢ mutably, n² reals in column-major order.
    ///
    /// # Panics
    ///
    /// Panics if `i >= count`.
    #[inline]
    pub fn matrix_mut(&mut self, i: usize) -> &mut [T] {
        let range = self.matrix_range(i);
        &mut self.data[range]
    }

    /// Returns column k of matrix Aᵢ mutably, n reals. A builder writes the operator's
    /// response to the k-th unit vector here.
    ///
    /// # Panics
    ///
    /// Panics if `i >= count` or `k >= n`.
    #[inline]
    pub fn column_mut(&mut self, i: usize, k: usize) -> &mut [T] {
        let n = self.n;
        assert!(k < n, "column index {k} out of range for order {n}");
        &mut self.matrix_mut(i)[k * n..(k + 1) * n]
    }

    /// Adds Aᵢ x to y.
    ///
    /// Summation order: for k = 0, 1, …, n − 1 in turn, column k is scaled by xₖ and
    /// added into y, so each yᵣ is updated as yᵣ ← yᵣ + xₖ Aᵣₖ with k increasing,
    /// starting from the incoming yᵣ. Every product and sum is rounded separately (no
    /// fused multiply–add). The result is therefore deterministic and bit-identical to
    /// the naive double loop in that order. Allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics if `i >= count`, or if `x` or `y` does not have length n.
    pub fn apply(&self, i: usize, x: &[T], y: &mut [T]) {
        let n = self.n;
        assert_eq!(x.len(), n, "`x` must have length n = {n}");
        assert_eq!(y.len(), n, "`y` must have length n = {n}");
        for (&xk, column) in x.iter().zip(self.matrix(i).chunks_exact(n)) {
            for (yr, &ark) in y.iter_mut().zip(column) {
                *yr = *yr + xk * ark;
            }
        }
    }

    /// Returns the same set in precision `U`, every entry rounded to nearest through
    /// [`RealScalar::to_f64`] and [`RealScalar::from_f64`].
    ///
    /// From f64 to f32 this rounds each entry once, as `as f32` does; this is how an
    /// f32 table is made from the f64 build.
    pub fn cast<U: RealScalar>(&self) -> MatrixSet<U> {
        MatrixSet {
            n: self.n,
            count: self.count,
            data: self.data.iter().map(|&v| U::from_f64(v.to_f64())).collect(),
        }
    }

    /// The index range of matrix Aᵢ in `data`.
    #[inline]
    fn matrix_range(&self, i: usize) -> core::ops::Range<usize> {
        assert!(
            i < self.count,
            "matrix index {i} out of range for {} matrices",
            self.count
        );
        let n2 = self.n * self.n;
        i * n2..(i + 1) * n2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// SplitMix64: a small, deterministic generator, so the tests need no extra
    /// dependency.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// Uniform in [−1, 1).
        fn signed(&mut self) -> f64 {
            2.0 * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) - 1.0
        }
    }

    /// A set of `count` matrices of order `n` with entries uniform in [−1, 1).
    fn random_set<T: RealScalar>(rng: &mut SplitMix64, n: usize, count: usize) -> MatrixSet<T> {
        let mut set = MatrixSet::zeros(n, count);
        for i in 0..count {
            for v in set.matrix_mut(i) {
                *v = T::from_f64(rng.signed());
            }
        }
        set
    }

    fn random_vector<T: RealScalar>(rng: &mut SplitMix64, n: usize) -> Vec<T> {
        (0..n).map(|_| T::from_f64(rng.signed())).collect()
    }

    /// The naive double loop: columns k in increasing order, rows r within.
    fn naive_apply<T: RealScalar>(set: &MatrixSet<T>, i: usize, x: &[T], y: &mut [T]) {
        let n = set.n();
        let data = set.as_slice();
        for k in 0..n {
            for r in 0..n {
                y[r] = y[r] + x[k] * data[i * n * n + r + k * n];
            }
        }
    }

    /// Checks `apply` against [`naive_apply`] for every matrix of a random set, from
    /// the starting vector `y0`.
    fn check_apply<T: RealScalar>(set: &MatrixSet<T>, x: &[T], y0: &[T]) {
        // Error measure: bit-for-bit equality with the naive loop in the same order.
        for i in 0..set.count() {
            let mut fast = y0.to_vec();
            let mut naive = y0.to_vec();
            set.apply(i, x, &mut fast);
            naive_apply(set, i, x, &mut naive);
            let fast: Vec<u64> = fast.iter().map(|&v| v.to_f64().to_bits()).collect();
            let naive: Vec<u64> = naive.iter().map(|&v| v.to_f64().to_bits()).collect();
            assert_eq!(fast, naive, "matrix {i}, n = {}", set.n());
        }
    }

    #[test]
    fn zeros_has_shape_and_zero_entries() {
        // Error measure: exact equality (no arithmetic is involved).
        let set = MatrixSet::<f64>::zeros(4, 3);
        assert_eq!((set.n(), set.count()), (4, 3));
        assert_eq!(set.as_slice().len(), 48);
        assert!(set.as_slice().iter().all(|&v| v == 0.0));
        assert!(MatrixSet::<f32>::zeros(5, 0).as_slice().is_empty());
    }

    #[test]
    fn apply_matches_naive_loop_bit_for_bit() {
        let mut rng = SplitMix64(0x7ab1e5);
        for n in [1, 4, 81] {
            let set = random_set::<f64>(&mut rng, n, 3);
            let x = random_vector::<f64>(&mut rng, n);
            check_apply(&set, &x, &vec![0.0; n]);
            let set = random_set::<f32>(&mut rng, n, 3);
            let x = random_vector::<f32>(&mut rng, n);
            check_apply(&set, &x, &vec![0.0; n]);
        }
    }

    #[test]
    fn apply_accumulates_onto_nonzero_y() {
        let mut rng = SplitMix64(0xacc);
        for n in [1, 4, 81] {
            let set = random_set::<f64>(&mut rng, n, 2);
            let x = random_vector::<f64>(&mut rng, n);
            let y0 = random_vector::<f64>(&mut rng, n);
            // Error measure: bit-for-bit equality with the naive loop from the same y.
            check_apply(&set, &x, &y0);
            // Error measure: exact equality. With A = 0, y is unchanged; with A = I
            // and x = y0, every entry doubles exactly.
            let mut y = y0.clone();
            MatrixSet::zeros(n, 1).apply(0, &x, &mut y);
            assert_eq!(y, y0);
            let mut identity = MatrixSet::zeros(n, 1);
            for k in 0..n {
                identity.column_mut(0, k)[k] = 1.0;
            }
            identity.apply(0, &y0, &mut y);
            assert_eq!(y, y0.iter().map(|&v| 2.0 * v).collect::<Vec<_>>());
        }
    }

    #[test]
    fn entry_layout_is_column_major_and_contiguous() {
        // Error measure: exact equality (entries are small integers).
        let (n, count) = (3, 4);
        let mut set = MatrixSet::<f64>::zeros(n, count);
        let code = |i: usize, r: usize, c: usize| (100 * i + 10 * r + c) as f64;
        for i in 0..count {
            for c in 0..n {
                let column = set.column_mut(i, c);
                assert_eq!(column.len(), n);
                for (r, v) in column.iter_mut().enumerate() {
                    *v = code(i, r, c);
                }
            }
        }
        for i in 0..count {
            assert_eq!(set.matrix(i).len(), n * n);
            for r in 0..n {
                for c in 0..n {
                    assert_eq!(set.as_slice()[i * n * n + r + c * n], code(i, r, c));
                    assert_eq!(set.matrix(i)[r + c * n], code(i, r, c));
                }
            }
        }
        set.matrix_mut(2)[1 + 2 * n] = -1.0;
        assert_eq!(set.as_slice()[2 * n * n + 1 + 2 * n], -1.0);
    }

    #[test]
    fn cast_to_f32_equals_as_f32() {
        // Error measure: exact equality of every entry with `as f32`.
        let mut rng = SplitMix64(0xca57);
        let mut set = random_set::<f64>(&mut rng, 9, 4);
        // Entries that round differently from truncation, and a tie that rounds to even.
        set.matrix_mut(0)[0] = 1.0 + f64::EPSILON;
        set.matrix_mut(0)[1] = 1.0 / 3.0;
        set.matrix_mut(0)[2] = 1.0 + 2.0_f64.powi(-24);
        set.matrix_mut(0)[3] = 1.0e300;
        let single = set.cast::<f32>();
        assert_eq!((single.n(), single.count()), (set.n(), set.count()));
        for (&s, &d) in single.as_slice().iter().zip(set.as_slice()) {
            assert_eq!(s.to_bits(), (d as f32).to_bits());
        }
        // Back to f64 is exact.
        let double = single.cast::<f64>();
        for (&d, &s) in double.as_slice().iter().zip(single.as_slice()) {
            assert_eq!(d, f64::from(s));
        }
    }

    #[test]
    #[should_panic(expected = "`x` must have length n = 4")]
    fn apply_panics_on_wrong_x_length() {
        let set = MatrixSet::<f64>::zeros(4, 1);
        set.apply(0, &[0.0; 3], &mut [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "`y` must have length n = 4")]
    fn apply_panics_on_wrong_y_length() {
        let set = MatrixSet::<f64>::zeros(4, 1);
        set.apply(0, &[0.0; 4], &mut [0.0; 5]);
    }

    #[test]
    #[should_panic(expected = "matrix index 2 out of range for 2 matrices")]
    fn apply_panics_on_matrix_index_out_of_range() {
        let set = MatrixSet::<f64>::zeros(4, 2);
        set.apply(2, &[0.0; 4], &mut [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "matrix index 1 out of range for 1 matrices")]
    fn matrix_panics_on_index_out_of_range() {
        MatrixSet::<f32>::zeros(2, 1).matrix(1);
    }

    #[test]
    #[should_panic(expected = "column index 3 out of range for order 3")]
    fn column_mut_panics_on_index_out_of_range() {
        MatrixSet::<f64>::zeros(3, 1).column_mut(0, 3);
    }

    #[test]
    #[should_panic(expected = "MatrixSet order n must be positive")]
    fn zeros_panics_on_zero_order() {
        MatrixSet::<f64>::zeros(0, 1);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn apply_matches_naive_loop_for_random_shapes(
            n in 1usize..=20,
            count in 1usize..=4,
            seed in any::<u64>(),
        ) {
            let mut rng = SplitMix64(seed);
            let set = random_set::<f64>(&mut rng, n, count);
            let x = random_vector::<f64>(&mut rng, n);
            let y0 = random_vector::<f64>(&mut rng, n);
            // Error measure: bit-for-bit equality with the naive loop in the same order.
            for i in 0..count {
                let mut fast = y0.clone();
                let mut naive = y0.clone();
                set.apply(i, &x, &mut fast);
                naive_apply(&set, i, &x, &mut naive);
                let fast: Vec<u64> = fast.iter().map(|v| v.to_bits()).collect();
                let naive: Vec<u64> = naive.iter().map(|v| v.to_bits()).collect();
                prop_assert_eq!(fast, naive);
            }
        }
    }
}
