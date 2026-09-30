//! A seeded pseudo-random generator.

/// SplitMix64: a small, fast, seeded pseudo-random generator (Steele, Lea and Flood,
/// 2014), the same algorithm as the generators of the `nd-fmm-math` and `nd-fmm-ref`
/// tests, so this crate needs no `rand` dependency.
///
/// The sequence depends only on the seed, so every distribution and sweep built on it
/// is deterministic. It is not a cryptographic generator.
///
/// ```
/// use nd_fmm_validate::SplitMix64;
///
/// let mut a = SplitMix64::new(7);
/// let mut b = SplitMix64::new(7);
/// assert_eq!(a.next_u64(), b.next_u64());
/// let x = a.range(-1.0, 1.0);
/// assert!((-1.0..1.0).contains(&x));
/// ```
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Creates a generator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Returns the next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a number uniform in [0, 1), from the top 53 bits of [`next_u64`](Self::next_u64).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Returns a number uniform in [lo, hi) (up to the rounding of lo + (hi − lo) u).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_reference_sequence() {
        // The published SplitMix64 outputs for seed 0 (e.g. the reference implementation
        // of Vigna's splitmix64.c). Error measure: exact equality.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn deterministic_for_a_seed() {
        let a: Vec<u64> = {
            let mut rng = SplitMix64::new(42);
            (0..100).map(|_| rng.next_u64()).collect()
        };
        let mut rng = SplitMix64::new(42);
        assert!(a.iter().all(|&x| x == rng.next_u64()));
        let mut other = SplitMix64::new(43);
        assert_ne!(a[0], other.next_u64());
    }

    #[test]
    fn uniform_and_range_stay_in_their_intervals() {
        let mut rng = SplitMix64::new(1);
        let mut sum = 0.0;
        let n = 10_000;
        for _ in 0..n {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
            sum += u;
            let x = rng.range(-3.0, 5.0);
            assert!((-3.0..5.0).contains(&x));
        }
        // The mean of n uniform numbers has standard deviation 1/√(12 n) ≈ 0.003.
        assert!((sum / n as f64 - 0.5).abs() < 0.02);
    }
}
