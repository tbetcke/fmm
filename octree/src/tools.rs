//! Utility routines.

use itertools::izip;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::{
    MortonKey,
    constants::{DEEPEST_LEVEL, LEVEL_SIZE},
    morton,
};

/// Generate random keys for testing.
/// # Parameters
///
/// - `nkeys`: Number of distinct finest-level keys to generate.
/// - `rng`: Random source used for sampling.
///
/// # Examples
///
/// ```
/// let mut rng = nd_octree::tools::seeded_rng(7);
/// assert_eq!(nd_octree::tools::generate_random_keys(2, &mut rng).len(), 2);
/// ```
pub fn generate_random_keys<R: Rng>(nkeys: usize, rng: &mut R) -> Vec<MortonKey> {
    let mut result = Vec::<MortonKey>::with_capacity(nkeys);

    let xindices = rand::seq::index::sample(rng, LEVEL_SIZE as usize, nkeys);
    let yindices = rand::seq::index::sample(rng, LEVEL_SIZE as usize, nkeys);
    let zindices = rand::seq::index::sample(rng, LEVEL_SIZE as usize, nkeys);

    for (xval, yval, zval) in izip!(xindices.iter(), yindices.iter(), zindices.iter()) {
        result.push(morton::from_index_and_level(
            [xval, yval, zval],
            DEEPEST_LEVEL as usize,
        ));
    }

    result
}

/// Get a seeded rng
/// # Parameters
///
/// - `seed`: Deterministic seed for the random stream.
///
/// # Examples
///
/// ```
/// let _rng = nd_octree::tools::seeded_rng(7);
/// ```
pub fn seeded_rng(seed: usize) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed as u64)
}
