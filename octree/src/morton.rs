//! Routines for working with Morton keys.
//!
//! A Morton key is a 64 bit integer that uniquely encodes a node of an octree. This module
//! defines methods for working with Morton keys. Morton keys within this library support
//! octree data structures up to a depth of level 16, meaning we have on the deepest level
//! indices from 0 to 65535 in each direction. An octree node has 8 children meaning per level we
//! require 3 bits to uniquely identify a descendent. Hence, in total we require 48 bits. In addition
//! we have reserved 15 bits for level information, which is more than required. The level information
//! is stored in the lowest 15 bits and the index information is stored in the next 48 bits. Hence,
//! a valid key requires 63 bits. We use bit 64 to indicate an invalid key. If bit 64 is 1 then the
//! key is invalid. This is useful for initializing arrays of Morton keys with default values. The default
//! value of a Morton key is an invalid key.

use crate::constants::{
    BYTE_DISPLACEMENT, BYTE_MASK, DEEPEST_LEVEL, DIRECTIONS, LEVEL_DISPLACEMENT, LEVEL_MASK,
    LEVEL_SIZE, NINE_BIT_MASK, NSIBLINGS, X_LOOKUP_DECODE, X_LOOKUP_ENCODE, Y_LOOKUP_DECODE,
    Y_LOOKUP_ENCODE, Z_LOOKUP_DECODE, Z_LOOKUP_ENCODE,
};
use crate::geometry::PhysicalBox;
use itertools::Itertools;
use itertools::izip;
use std::collections::HashSet;

/// Type alias for a Morton key
pub type MortonKey = u64;

/// A key that is not valid or well formed but guaranteed to be larger than any valid key.
///
/// This is useful when a guaranteed upper bound is needed.
#[inline(always)]
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::upper_bound() > morton::invalid_key());
/// ```
pub fn upper_bound() -> MortonKey {
    u64::MAX
}

/// Create an invalid key.
#[inline(always)]
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(!morton::is_valid(morton::invalid_key()));
/// ```
pub fn invalid_key() -> MortonKey {
    1 << 63
}

/// Check if key is valid.
///
/// A key is not valid if its highest bit is 1.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::is_valid(morton::root()));
/// ```
pub fn is_valid(key: MortonKey) -> bool {
    // If the highest bit is 1 the key is by definition not valid.
    key >> 63 != 1
}

/// Create a root key.
///
/// A root key simply has the value `0`.
#[inline(always)]
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::root(), 0);
/// ```
pub fn root() -> MortonKey {
    0
}

/// Return the first deepest key.
///
/// This is the first key on the deepest level.
#[inline(always)]
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::level(morton::deepest_first()), 16);
/// ```
pub fn deepest_first() -> MortonKey {
    from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize)
}

/// Return the last deepest key.
///
/// This is the last key on the deepest level.
#[inline(always)]
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::level(morton::deepest_last()), 16);
/// ```
pub fn deepest_last() -> MortonKey {
    from_index_and_level(
        [
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
        ],
        DEEPEST_LEVEL as usize,
    )
}

/// Return the associated physical box.
///
/// Given the physical boundix bos of the octree this method returns the
/// physical box associated with the key.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
/// - `bounding_box`: Physical domain used for coordinate conversion.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// use nd_octree::PhysicalBox;
/// let domain = PhysicalBox::new([0.0; 6]);
/// let _ = morton::physical_box(morton::root(), &domain);
/// ```
pub fn physical_box(key: MortonKey, bounding_box: &PhysicalBox) -> PhysicalBox {
    let (level, [x, y, z]) = decode(key);
    let xind = x as f64;
    let yind = y as f64;
    let zind = z as f64;

    let [xmin, ymin, zmin, xmax, ymax, zmax] = bounding_box.coordinates();
    let level_size = (1 << level) as f64;
    let xlength = (xmax - xmin) / level_size;
    let ylength = (ymax - ymin) / level_size;
    let zlength = (zmax - zmin) / level_size;

    PhysicalBox::new([
        xmin + xind * xlength,
        ymin + yind * ylength,
        zmin + zind * zlength,
        xmin + (1.0 + xind) * xlength,
        ymin + (1.0 + yind) * ylength,
        zmin + (1.0 + zind) * zlength,
    ])
}

/// Return key in a given direction.
///
/// Returns an invalid key if there is no valid key in that direction.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
/// - `direction`: Cell offset in `[x, y, z]` order.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let key = morton::from_index_and_level([0, 0, 0], 1);
/// assert!(!morton::is_valid(morton::key_in_direction(key, [-1, 0, 0])));
/// ```
pub fn key_in_direction(key: MortonKey, direction: [i64; 3]) -> MortonKey {
    let (level, [x, y, z]) = decode(key);
    let level_size = 1 << level;

    let new_index = [
        x as i64 + direction[0],
        y as i64 + direction[1],
        z as i64 + direction[2],
    ];

    if 0 <= new_index[0]
        && new_index[0] < level_size
        && 0 <= new_index[1]
        && new_index[1] < level_size
        && 0 <= new_index[2]
        && new_index[2] < level_size
    {
        from_index_and_level(
            [
                new_index[0] as usize,
                new_index[1] as usize,
                new_index[2] as usize,
            ],
            level,
        )
    } else {
        invalid_key()
    }
}

/// A key is ill-formed if it has non-zero bits and positions that should be zero by the given level.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::is_well_formed(morton::root()));
/// ```
pub fn is_well_formed(key: MortonKey) -> bool {
    let level = key & LEVEL_MASK;
    // An encoded level beyond the deepest supported level is ill-formed. Checking
    // it here also keeps the shift below from underflowing.
    if level > DEEPEST_LEVEL {
        return false;
    }
    let key = key >> LEVEL_DISPLACEMENT;
    // Check that all the bits below the level of the key are zero.
    // Need to first create a suitable bitmask that has
    // all bits set to one at the last DEEPEST_LEVEL - level bits.
    let shift = 3 * (DEEPEST_LEVEL - level);
    // The mask has now bits set to one at the last `level_diff` bits
    let mask: u64 = (1 << shift) - 1;
    // Is zero if and only if all the bits of the key at the `level_diff` bits are zero.
    (mask & key) == 0
}

/// Map a physical point within a bounding box to a Morton key on a given level.
///
/// It is assumed that points are strictly contained within the bounding box.
#[inline(always)]
/// # Parameters
///
/// - `point`: Three Cartesian coordinates in the source coordinate system.
/// - `bounding_box`: Physical domain used for coordinate conversion.
/// - `level`: Octree level, from zero through the supported maximum.
///
/// # Examples
///
/// ```
/// use nd_octree::{morton, PhysicalBox};
/// let domain = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
/// let key = morton::from_physical_point([0.5, 0.5, 0.5], &domain, 1);
/// assert_eq!(morton::level(key), 1);
/// ```
pub fn from_physical_point(point: [f64; 3], bounding_box: &PhysicalBox, level: usize) -> MortonKey {
    let level_size: usize = 1 << level;
    let reference = bounding_box.physical_to_reference(point);
    // A point strictly inside the box can still round to a reference coordinate of
    // exactly one, which would index one cell past the last. Clamp so such a point
    // stays in the final cell instead of wrapping to the opposite corner.
    let index = |coordinate: f64| ((coordinate * level_size as f64) as usize).min(level_size - 1);

    from_index_and_level(
        [
            index(reference[0]),
            index(reference[1]),
            index(reference[2]),
        ],
        level,
    )
}

/// Create a new key by providing the [x, y, z] index and a level.
///
/// This is the preferred way to create a new Morton key.
#[inline(always)]
/// # Parameters
///
/// - `index`: `[x, y, z]` cell index at `level`.
/// - `level`: Octree level, from zero through the supported maximum.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let key = morton::from_index_and_level([1, 0, 1], 1);
/// assert_eq!(morton::decode(key), (1, [1, 0, 1]));
/// ```
pub fn from_index_and_level(index: [usize; 3], level: usize) -> MortonKey {
    let level = level as u64;
    debug_assert!(level <= DEEPEST_LEVEL);

    debug_assert!(index[0] < (1 << level));
    debug_assert!(index[1] < (1 << level));
    debug_assert!(index[2] < (1 << level));

    // If we are not on the deepest level we need to shift the box.
    // The box with x-index one on DEEPEST_LEVEL-1 has index two on
    // DEEPEST_LEVEL.

    let level_diff = DEEPEST_LEVEL - level;

    let x = (index[0] as u64) << level_diff;
    let y = (index[1] as u64) << level_diff;
    let z = (index[2] as u64) << level_diff;

    let key: u64 = X_LOOKUP_ENCODE[((x >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[((y >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[((z >> BYTE_DISPLACEMENT) & BYTE_MASK) as usize];

    let key = (key << 24)
        | X_LOOKUP_ENCODE[(x & BYTE_MASK) as usize]
        | Y_LOOKUP_ENCODE[(y & BYTE_MASK) as usize]
        | Z_LOOKUP_ENCODE[(z & BYTE_MASK) as usize];

    let key = key << LEVEL_DISPLACEMENT;
    key | level
}

/// Return the level of a key.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::level(morton::root()), 0);
/// ```
pub fn level(key: MortonKey) -> usize {
    (key & LEVEL_MASK) as usize
}

/// Decode a key and return a tuple of the form (level, [x, y, z]), where the latter is the index vector.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::decode(morton::root()), (0, [0, 0, 0]));
/// ```
pub fn decode(key: MortonKey) -> (usize, [usize; 3]) {
    #[inline(always)]
    /// # Parameters
    ///
    /// - `key`: Raw Morton position bits, without level interpretation.
    /// - `lookup_table`: 9-bit Morton chunk lookup table for one Cartesian axis.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // This nested helper is private to decoding.
    /// assert_eq!(decode_key_helper(0, &Z_LOOKUP_DECODE), 0);
    /// ```
    fn decode_key_helper(key: u64, lookup_table: &[u64; 512]) -> u64 {
        const N_LOOPS: u64 = 6; // 48 bits for the keys. Process in pairs of 9. So 6 passes enough.
        let mut coord: u64 = 0;

        for index in 0..N_LOOPS {
            coord |= lookup_table[((key >> (index * 9)) & NINE_BIT_MASK) as usize] << (3 * index);
        }

        coord
    }

    let level = level(key);
    debug_assert!(
        level as u64 <= DEEPEST_LEVEL,
        "cannot decode a key whose encoded level exceeds DEEPEST_LEVEL"
    );
    // Saturate so an ill-formed level yields a defined result in release builds
    // rather than an underflowing shift.
    let level_diff = DEEPEST_LEVEL.saturating_sub(level as u64);

    let key = key >> LEVEL_DISPLACEMENT;

    let x = decode_key_helper(key, &X_LOOKUP_DECODE);
    let y = decode_key_helper(key, &Y_LOOKUP_DECODE);
    let z = decode_key_helper(key, &Z_LOOKUP_DECODE);

    let x = x >> level_diff;
    let y = y >> level_diff;
    let z = z >> level_diff;

    (level, [x as usize, y as usize, z as usize])
}

/// Return the parent of a key.
///
/// Returns `None` if the key is root, otherwise `Some(parent)`
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::parent(morton::root()), None);
/// ```
pub fn parent(key: MortonKey) -> Option<MortonKey> {
    let level = level(key);
    if level == 0 {
        None
    } else {
        // We set the bits at our current level to zero and subtract 1 at the end to reduce the
        // level by one.

        let bit_displacement = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level as u64);
        let mask = !(7 << bit_displacement);

        Some((key & mask) - 1)
    }
}

/// Return ancestor of key on specified level
///
/// Return None if `level > level(key)`.
/// Return the key itself if `level == level(key)`.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
/// - `level`: Octree level, from zero through the supported maximum.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let key = morton::from_index_and_level([1, 0, 1], 1);
/// assert_eq!(morton::ancestor_at_level(key, 0), Some(morton::root()));
/// ```
pub fn ancestor_at_level(key: MortonKey, level: usize) -> Option<MortonKey> {
    let my_level = super::morton::level(key);

    if my_level < level {
        return None;
    }

    if my_level == level {
        return Some(key);
    }

    let key = key >> LEVEL_DISPLACEMENT;

    let bit_displacement = 3 * (DEEPEST_LEVEL - level as u64);
    // Sets the last bits to zero and shifts back
    let key = (key >> bit_displacement) << (bit_displacement + LEVEL_DISPLACEMENT);

    Some(key | level as u64)
}

/// Check if `key` is ancestor of `other`. If keys are identical also returns true.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
/// - `other`: Second Morton key whose relationship to `key` is tested.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::is_ancestor(morton::root(), morton::deepest_first()));
/// ```
pub fn is_ancestor(key: MortonKey, other: MortonKey) -> bool {
    let my_level = level(key);
    let other_level = level(other);

    if !is_valid(key) || !is_valid(other) {
        return false;
    }

    // An encoded level beyond the deepest supported level is ill-formed and has
    // no ancestry; returning early also keeps the shifts below from underflowing.
    if my_level > DEEPEST_LEVEL as usize || other_level > DEEPEST_LEVEL as usize {
        return false;
    }

    if key == other {
        true
    } else if my_level > other_level {
        false
    } else {
        // We shift both keys out to 3 * DEEPEST_LEVEL - my_level
        // This gives identical bit sequences if my_key is an ancestor of other_key
        let my_key = key >> (LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as u64));
        let other_key = other >> (LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - my_level as u64));

        my_key == other_key
    }
}

/// Return the finest common ancestor of two keys.
///
/// If the keys are identical return the key itself.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
/// - `other`: Second Morton key whose relationship to `key` is tested.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::finest_common_ancestor(morton::root(), morton::deepest_first()), morton::root());
/// ```
pub fn finest_common_ancestor(key: MortonKey, other: MortonKey) -> MortonKey {
    if key == other {
        return key;
    }

    let my_level = level(key);
    let other_level = level(other);

    // Want to bring both keys to the minimum of the two levels.
    let level = my_level.min(other_level);
    debug_assert!(
        level as u64 <= DEEPEST_LEVEL,
        "cannot compute a common ancestor of keys whose encoded level exceeds DEEPEST_LEVEL"
    );

    // Remove the level information and bring second key to the same level as first key
    // After the following operation the least significant bits are associated with `first_level`.

    let level_displacement = LEVEL_DISPLACEMENT + 3 * DEEPEST_LEVEL.saturating_sub(level as u64);

    let mut first_key = key >> level_displacement;
    let mut second_key = other >> level_displacement;

    // Now move both keys up until they are identical.
    // At the same time we reduce the first level.

    let mut count = 0;

    while first_key != second_key {
        count += 1;
        first_key >>= 3;
        second_key >>= 3;
    }

    // We now return the ancestor at the given level.

    let new_level = level - count;

    first_key <<= 3 * DEEPEST_LEVEL.saturating_sub(new_level as u64) + LEVEL_DISPLACEMENT;

    first_key | new_level as u64
}

/// Return true if key is equal to the root key.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::is_root(morton::root()));
/// ```
pub fn is_root(key: MortonKey) -> bool {
    key == 0
}

/// Return the 8 children of a key.
///
/// Returns `None` if at deepest level (or if the encoded level is ill-formed and
/// exceeds it), otherwise `Some(children)`.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::children(morton::root()).unwrap().len(), 8);
/// ```
pub fn children(key: MortonKey) -> Option<[MortonKey; 8]> {
    let level = level(key) as u64;
    if level >= DEEPEST_LEVEL {
        None
    } else {
        let child_level = 1 + level;

        let shift = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - child_level);
        Some([
            1 + (key | (0 << shift)),
            1 + (key | (1 << shift)),
            1 + (key | (2 << shift)),
            1 + (key | (3 << shift)),
            1 + (key | (4 << shift)),
            1 + (key | (5 << shift)),
            1 + (key | (6 << shift)),
            1 + (key | (7 << shift)),
        ])
    }
}

/// Return the 8 siblings of a key.
///
/// The key itself is part of the siblings.
///
/// Returns `None` if the key is root, otherwise `Some(siblings)`.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let key = morton::from_index_and_level([0, 0, 0], 1);
/// assert_eq!(morton::siblings(key).unwrap().len(), 8);
/// ```
pub fn siblings(key: MortonKey) -> Option<[MortonKey; 8]> {
    if is_root(key) {
        None
    } else {
        children(parent(key).unwrap())
    }
}

/// Return the neighbours of a key.
///
/// The key itself is not part of the neighbours.
/// If along a certain direction there is no neighbour then
///  an invalid key is stored.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::neighbours(morton::root()).len(), 26);
/// ```
pub fn neighbours(key: MortonKey) -> [MortonKey; 26] {
    let mut result = [invalid_key(); 26];

    let (level, [x, y, z]) = decode(key);

    if level == 0 {
        return result;
    }
    let level_size = 1 << level;

    for (direction, res) in izip!(DIRECTIONS, result.iter_mut()) {
        let new_index = [
            x as i64 + direction[0],
            y as i64 + direction[1],
            z as i64 + direction[2],
        ];
        if 0 <= new_index[0]
            && new_index[0] < level_size
            && 0 <= new_index[1]
            && new_index[1] < level_size
            && 0 <= new_index[2]
            && new_index[2] < level_size
        {
            *res = from_index_and_level(
                [
                    new_index[0] as usize,
                    new_index[1] as usize,
                    new_index[2] as usize,
                ],
                level,
            );
        }
    }
    result
}

/// Return the index of the key as a child of the parent, i.e. 0, 1, ..., 7.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let key = morton::from_index_and_level([1, 0, 0], 1);
/// assert_eq!(morton::child_index(key), 4);
/// ```
pub fn child_index(key: MortonKey) -> usize {
    if key == root() {
        return 0;
    }
    let level = level(key) as u64;

    let shift = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - level);

    ((key >> shift) % 8) as usize
}

/// Return the finest descendent that is opposite to the joint corner with the siblings.
#[inline(always)]
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let key = morton::from_index_and_level([1, 0, 0], 1);
/// assert_eq!(morton::level(morton::finest_outer_descendent(key)), 16);
/// ```
pub fn finest_outer_descendent(mut key: MortonKey) -> MortonKey {
    // First find out which child the current key is.

    let level = level(key) as u64;

    if level == DEEPEST_LEVEL {
        return key;
    }

    let mut child_level = 1 + level;
    let outer_index = child_index(key) as u64;

    while child_level <= DEEPEST_LEVEL {
        let shift = LEVEL_DISPLACEMENT + 3 * (DEEPEST_LEVEL - child_level);
        key = 1 + (key | outer_index << shift);
        child_level += 1;
    }

    key
}

/// Return the next possible Morton key on the deepest level that is not a descendent of the current key.
///
/// If the key is already the last possible key then return None.
/// # Parameters
///
/// - `key`: Morton key to inspect or transform.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::next_non_descendent_key(morton::deepest_last()), None);
/// ```
pub fn next_non_descendent_key(key: MortonKey) -> Option<MortonKey> {
    // If we are an ancestor of deepest_last we return None as then there
    // is next key.

    if is_ancestor(key, deepest_last()) {
        return None;
    }

    let level = level(key) as u64;

    let level_diff = DEEPEST_LEVEL - level;
    let shift = LEVEL_DISPLACEMENT + 3 * level_diff;

    // Need to know which sibling we are.
    let child_index = ((key >> shift) % 8) as usize;
    // If we are between 0 and 6 take the next sibling and go to deepest level.
    if child_index < 7 {
        Some(key + (1 << shift) + level_diff)
    } else {
        // If we are the last child go to the parent and take next key from there.
        next_non_descendent_key(parent(key).unwrap())
    }
}

/// Linearize by sorting and removing overlaps.
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let keys = morton::linearize(&[morton::root(), morton::root()]);
/// assert_eq!(keys, vec![morton::root()]);
/// ```
pub fn linearize(keys: &[MortonKey]) -> Vec<MortonKey> {
    let mut new_keys = Vec::<MortonKey>::new();
    if keys.is_empty() {
        new_keys
    } else {
        let mut keys = keys.to_vec();
        keys.sort_unstable();
        for (&m1, &m2) in keys.iter().tuple_windows() {
            if m1 == m2 || is_ancestor(m1, m2) {
                continue;
            }
            new_keys.push(m1)
        }
        new_keys.push(*keys.last().unwrap());
        new_keys
    }
}

/// Fill the region between two keys with a minimal number of keys.
/// # Parameters
///
/// - `key1`: One endpoint of the Morton interval.
/// - `key2`: Other endpoint of the Morton interval.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let left = morton::from_index_and_level([0, 0, 0], 1);
/// let right = morton::from_index_and_level([1, 0, 0], 1);
/// let _between = morton::fill_between_keys(left, right);
/// ```
pub fn fill_between_keys(key1: MortonKey, key2: MortonKey) -> Vec<MortonKey> {
    // Make sure that key1 is smaller or equal key2
    let (key1, key2) = if key1 < key2 {
        (key1, key2)
    } else {
        (key2, key1)
    };

    // If key1 is ancestor of key2 return empty list. Note that
    // is_ancestor is true if key1 is equal to key2.
    if is_ancestor(key1, key2) {
        return Vec::<MortonKey>::new();
    }

    // The finest common ancestor is always closer to the root than either key
    // if key1 is not an ancestor of key2 or vice versa.
    let ancestor = finest_common_ancestor(key1, key2);
    let children = super::morton::children(ancestor).unwrap();

    let mut result = Vec::<MortonKey>::new();

    let mut work_set = Vec::<MortonKey>::from_iter(children.iter().copied());

    while let Some(item) = work_set.pop() {
        // If the item is either key we don't want it in the result.
        if item == key1 || item == key2 {
            continue;
        }
        // We want items that are strictly between the two keys and are not ancestors of either.
        // We do not check specifically if item is an ancestor of key1 as then it would be smaller than key1.
        else if key1 < item && item < key2 && !is_ancestor(item, key2) {
            result.push(item);
        } else {
            // If the item is an ancestor of key1 or key2 just refine to the children and try again.
            // Note we already exclude that item is identical to key1 or key2.
            // So if item is an ancestor of either its children cannot have a level larger than key1 or key2.
            if is_ancestor(item, key1) || is_ancestor(item, key2) {
                let children = super::morton::children(item).unwrap();
                work_set.extend(children.iter());
            }
        }
    }

    result.sort_unstable();
    result
}

/// Complete a tree ensuring that the given keys are part of the leafs.
///
/// The given keys must not overlap.
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let tree = morton::complete_tree(&[]);
/// assert_eq!(tree, vec![morton::root()]);
/// ```
pub fn complete_tree(keys: &[MortonKey]) -> Vec<MortonKey> {
    // First make sure that the input sequence is sorted.
    let mut keys = keys.to_vec();
    keys.sort_unstable();

    let mut result = Vec::<MortonKey>::new();

    // Special case of empty keys.
    if keys.is_empty() {
        result.push(root());
        return result;
    }

    // If just the root is given return that.
    if keys.len() == 1 && *keys.first().unwrap() == root() {
        return keys.to_vec();
    }

    let deepest_first = deepest_first();
    let deepest_last = deepest_last();

    // If the first key is not an ancestor of the deepest possible first element in the
    // tree get the finest ancestor between the two and use the first child of that.

    let first_key = *keys.first().unwrap();
    let last_key = *keys.last().unwrap();

    if !is_ancestor(first_key, deepest_first) {
        let ancestor = finest_common_ancestor(deepest_first, first_key);
        keys.insert(0, children(ancestor).unwrap()[0]);
    }

    if !is_ancestor(last_key, deepest_last) {
        let ancestor = finest_common_ancestor(deepest_last, last_key);
        keys.push(children(ancestor).unwrap()[NSIBLINGS - 1]);
    }

    // Now just iterate over the keys by tuples of two and fill the region between two keys.

    for (&key1, &key2) in keys.iter().tuple_windows() {
        result.push(key1);
        result.extend_from_slice(fill_between_keys(key1, key2).as_slice());
    }

    // Push the final key
    result.push(*keys.last().unwrap());
    // We do not sort the keys. They are already sorted.
    result
}

/// Get all interior keys for an Octree represented by a list of Morton keys
///
/// Adds the root at level 0 if the root is not a leaf if the octree.
/// If `keys` only contains the root of the tree then the returned set is empty.
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::get_interior_keys(&[morton::root()]).is_empty());
/// ```
pub fn get_interior_keys(keys: &[MortonKey]) -> HashSet<MortonKey> {
    let mut interior_keys = HashSet::<MortonKey>::new();

    let keys = linearize(keys);

    for &key in &keys {
        if level(key) > 0 {
            let mut p = parent(key).unwrap();
            while level(p) > 0 && !interior_keys.contains(&p) {
                interior_keys.insert(p);
                p = parent(p).unwrap();
            }
        }
    }

    if !keys.contains(&root()) {
        interior_keys.insert(root());
    }

    interior_keys
}

/// Return a set consisting of `keys` and all their ancestors.
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::get_interior_and_leaf_keys(&[morton::root()]).contains(&morton::root()));
/// ```
pub fn get_interior_and_leaf_keys(keys: &[MortonKey]) -> HashSet<MortonKey> {
    let mut all_keys = get_interior_keys(keys);
    all_keys.extend(keys.iter());
    all_keys
}

/// Check if a list of Morton keys represent a complete linear tree.
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert!(morton::is_complete_linear_octree(&[morton::root()]));
/// ```
pub fn is_complete_linear_octree(keys: &[MortonKey]) -> bool {
    // A tree containing an invalid sentinel or an ill-formed key is never complete,
    // and the ancestry checks below are only meaningful for well-formed keys.
    if keys
        .iter()
        .any(|&key| !is_valid(key) || !is_well_formed(key))
    {
        return false;
    }

    // First check that the list is sorted and not overlapping.
    for (&key1, &key2) in keys.iter().tuple_windows() {
        if key1 > key2 || is_ancestor(key1, key2) {
            return false;
        }
    }
    // Now check that all interior keys have 8 children.

    let interior_keys = get_interior_keys(keys);
    let mut all_keys = HashSet::<MortonKey>::from_iter(interior_keys.iter().copied());
    all_keys.extend(keys.iter());

    for key in interior_keys {
        let children = children(key).unwrap();
        for child in children {
            if !all_keys.contains(&child) {
                return false;
            }
        }
    }

    true
}

/// 2:1 balance a list of Morton keys with respect to a root key.
///
/// The balanced tree only has keys that are descendents of root
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
/// - `root`: Root of the region to balance.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// assert_eq!(morton::balance(&[morton::root()], morton::root()), vec![morton::root()]);
/// ```
pub fn balance(keys: &[MortonKey], root: MortonKey) -> Vec<MortonKey> {
    let keys = keys
        .iter()
        .copied()
        .filter(|&key| is_ancestor(root, key))
        .collect_vec();

    if keys.is_empty() {
        return Vec::<MortonKey>::new();
    }

    let deepest_level = keys.iter().map(|&key| level(key)).max().unwrap();
    let root_level = level(root);

    if deepest_level == root_level {
        return vec![root];
    }

    // Start with keys at deepest level
    let mut work_list = keys
        .iter()
        .copied()
        .filter(|&key| level(key) == deepest_level)
        .collect_vec();

    let mut result = Vec::<MortonKey>::new();

    // Now go through and make sure that for each key siblings and neighbours of parents are added

    for level in ((1 + root_level)..=deepest_level).rev() {
        let mut parents = HashSet::<MortonKey>::new();
        let mut new_work_list = Vec::<MortonKey>::new();
        // We filter the work list by level and also make sure that
        // only one sibling of each of the parents children is added to
        // our current level list.
        for &key in work_list.iter() {
            let parent = super::morton::parent(key).unwrap();
            if !parents.contains(&parent) {
                parents.insert(parent);
                result.extend_from_slice(siblings(key).unwrap().as_slice());
                new_work_list.extend_from_slice(
                    neighbours(parent)
                        .iter()
                        .copied()
                        // Invalid keys are also filtered out by `is_ancestor`.
                        .filter(|&key| is_ancestor(root, key))
                        .collect_vec()
                        .as_slice(),
                );
            }
        }
        new_work_list.extend(
            keys.iter()
                .copied()
                .filter(|&key| super::morton::level(key) == level - 1),
        );

        work_list = new_work_list;
        // Now extend the work list with the
    }

    linearize(result.as_slice())
}

/// Returns true if an Octree is linear, complete, and, balanced.
/// # Parameters
///
/// - `keys`: Morton keys that define the input tree or query batch.
///
/// # Examples
///
/// ```
/// use nd_octree::morton;
/// let leaves = morton::children(morton::root()).unwrap();
/// assert!(morton::is_complete_linear_and_balanced(&leaves));
/// ```
pub fn is_complete_linear_and_balanced(keys: &[MortonKey]) -> bool {
    // First check that it is complete and linear.

    if !is_complete_linear_octree(keys) {
        return false;
    }

    // Now check that it is balanced.
    // We add for each key the neighbors of the parents. If
    // we then linearize and the set of keys is identicial the octree
    // was balanced. Otherwise, some key are replaced in the linearisation
    // through desendents on a deeper level and the two lists are not
    // identical.

    let mut new_keys = keys.to_vec();
    for &key in keys {
        // The root has no parent, so a root-only tree contributes no neighbours.
        let Some(parent) = parent(key) else {
            continue;
        };
        new_keys.extend(neighbours(parent).iter().copied().filter(|k| is_valid(*k)));
    }

    let new_keys = linearize(&new_keys);

    if new_keys.len() != keys.len() {
        return false;
    } else {
        for (&key1, key2) in izip!(keys, new_keys) {
            if key1 != key2 {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_z_decode_table() {
        for (index, &actual) in Z_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = (index & 1) as u64;
            expected |= (((index >> 3) & 1) << 1) as u64;
            expected |= (((index >> 6) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_y_decode_table() {
        for (index, &actual) in Y_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = ((index >> 1) & 1) as u64;
            expected |= (((index >> 4) & 1) << 1) as u64;
            expected |= (((index >> 7) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_x_decode_table() {
        for (index, &actual) in X_LOOKUP_DECODE.iter().enumerate() {
            let mut expected: u64 = ((index >> 2) & 1) as u64;
            expected |= (((index >> 5) & 1) << 1) as u64;
            expected |= (((index >> 8) & 1) << 2) as u64;

            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_z_encode_table() {
        for (mut index, actual) in Z_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_y_encode_table() {
        for (mut index, actual) in Y_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 1)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_x_encode_table() {
        for (mut index, actual) in X_LOOKUP_ENCODE.iter().enumerate() {
            let mut sum: u64 = 0;

            for shift in 0..8 {
                sum |= ((index & 1) << (3 * shift + 2)) as u64;
                index >>= 1;
            }

            assert_eq!(sum, *actual);
        }
    }

    #[test]
    fn test_encoding_decoding() {
        let index: [usize; 3] = [
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
            LEVEL_SIZE as usize - 1,
        ];

        let key = from_index_and_level(index, DEEPEST_LEVEL as usize);

        let (level, actual) = decode(key);

        assert_eq!(level, DEEPEST_LEVEL as usize);
        assert_eq!(index, actual);
    }

    #[test]
    fn test_parent() {
        let index = [15, 39, 45];
        let key = from_index_and_level(index, 9);
        let parent = parent(key).unwrap();

        let expected_index = [7, 19, 22];
        let (actual_level, actual_index) = decode(parent);
        assert_eq!(actual_level, 8);
        assert_eq!(actual_index, expected_index);
    }

    #[test]
    fn test_ancestor() {
        let index = [15, 39, 45];
        let key = from_index_and_level(index, 9);
        assert!(is_ancestor(key, key));
        let ancestor = parent(parent(key).unwrap()).unwrap();
        assert!(is_ancestor(ancestor, key));
    }

    #[test]
    fn test_ancestor_at_level() {
        let index = [15, 39, 45];
        let key = from_index_and_level(index, 9);
        assert!(is_ancestor(key, key));
        let ancestor = parent(parent(key).unwrap()).unwrap();
        assert!(ancestor_at_level(key, 10).is_none());
        assert_eq!(ancestor_at_level(key, 9).unwrap(), key);
        assert_eq!(ancestor_at_level(key, 7).unwrap(), ancestor);
    }

    #[test]
    fn test_finest_ancestor() {
        let index = [15, 39, 45];

        let key = from_index_and_level(index, 9);
        // The finest ancestor with itself is the key itself.
        assert_eq!(finest_common_ancestor(key, key), key);
        // Finest ancestor with ancestor two levels up is the ancestor.
        let ancestor = parent(parent(key).unwrap()).unwrap();
        assert_eq!(finest_common_ancestor(key, ancestor), ancestor);

        // Finest ancestor  of the following keys should be the root of the tree.

        let key1 = from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize - 1);
        let key2 = from_index_and_level(
            [
                LEVEL_SIZE as usize - 1,
                LEVEL_SIZE as usize - 1,
                LEVEL_SIZE as usize - 1,
            ],
            DEEPEST_LEVEL as usize,
        );

        assert_eq!(finest_common_ancestor(key1, key2), root(),);

        // The finest ancestor of these two keys should be at level 1.

        let key1 = from_index_and_level([0, 0, 62], 6);
        let key2 = from_index_and_level([0, 0, 63], 6);
        let expected = from_index_and_level([0, 0, 31], 5);

        assert_eq!(finest_common_ancestor(key1, key2), expected);
    }

    #[test]
    fn test_children() {
        let key = from_index_and_level([4, 9, 8], 4);
        let children = super::children(key).unwrap();

        // Check that all the children are different.

        let children_set =
            std::collections::HashSet::<MortonKey>::from_iter(children.iter().copied());
        assert_eq!(children_set.len(), 8);

        // Check that all children are on the correct level and that their parent is our key.

        for child in children {
            assert_eq!(level(child), 5);
            assert_eq!(parent(child).unwrap(), key);
        }
    }

    #[test]
    fn test_fill_between_keys() {
        // Do various checks
        fn sanity_checks(key1: MortonKey, key2: MortonKey, mut keys: Vec<MortonKey>) {
            // Check that keys are strictly sorted and that no key is ancestor of the next key.

            let max_level = level(key1).max(level(key2));

            keys.insert(0, key1);
            keys.push(key2);

            for (k1, k2) in keys.iter().tuple_windows() {
                assert!(k1 < k2);
                assert!(!is_ancestor(key1, key2));
            }

            // Check that level not higher than max_level

            for &k in keys.iter() {
                assert!(level(k) <= max_level);
            }
        }

        // Correct result for keys on one level
        let key1 = from_index_and_level([0, 1, 0], 4);
        let key2 = from_index_and_level([8, 4, 13], 4);
        let keys = fill_between_keys(key1, key2);
        assert!(!keys.is_empty());

        sanity_checks(key1, key2, keys);

        // Correct result for passing same key twice

        let keys = fill_between_keys(key1, key1);
        assert!(keys.is_empty());

        // Two consecutive keys should also be empty.

        let children = children(key2).unwrap();

        let keys = fill_between_keys(children[1], children[2]);
        assert!(keys.is_empty());
    }

    #[test]
    pub fn test_complete_region() {
        // Do various checks
        fn sanity_checks(keys: &[MortonKey], complete_region: &[MortonKey]) {
            // Check that keys are strictly sorted and that no key is ancestor of the next key.

            if !keys.is_empty() {
                // Max level of input keys.
                let max_level = keys.iter().map(|&item| level(item)).max().unwrap();
                for k in complete_region.iter() {
                    assert!(level(*k) <= max_level);
                }
            }

            // Check that completed region has sorted keys and no overlaps.
            for (&k1, &k2) in complete_region.iter().tuple_windows() {
                assert!(k1 < k2);
                assert!(!is_ancestor(k1, k2));
            }

            // Check that first key is ancestor of first in deepest level
            // and that last key is ancestor of last in deepest level.
            let deepest_first = from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize);
            let deepest_last = from_index_and_level(
                [
                    LEVEL_SIZE as usize - 1,
                    LEVEL_SIZE as usize - 1,
                    LEVEL_SIZE as usize - 1,
                ],
                DEEPEST_LEVEL as usize,
            );

            assert!(is_ancestor(
                *complete_region.first().unwrap(),
                deepest_first
            ));
            assert!(is_ancestor(*complete_region.last().unwrap(), deepest_last));
        }

        // Create 3 Morton keys around which to complete region.

        let key1 = from_index_and_level([17, 30, 55], 10);
        let key2 = from_index_and_level([17, 540, 55], 10);
        let key3 = from_index_and_level([17, 30, 799], 11);

        let keys = [key1, key2, key3];

        let complete_region = complete_tree(keys.as_slice());

        sanity_checks(keys.as_slice(), complete_region.as_slice());

        // For an empty slice the complete region method should just add the root of the tree.
        let keys = Vec::<MortonKey>::new();
        let complete_region = complete_tree(keys.as_slice());
        assert_eq!(complete_region.len(), 1);

        sanity_checks(keys.as_slice(), complete_region.as_slice());

        // Choose a region where the first and last key are ancestors of deepest first and deepest last.

        let keys = [deepest_first(), deepest_last()];

        let complete_region = complete_tree(keys.as_slice());

        sanity_checks(keys.as_slice(), complete_region.as_slice());
    }

    #[test]
    pub fn test_neighbour_directions_unique() {
        let neighbour_set: HashSet<[i64; 3]> = HashSet::from_iter(DIRECTIONS.iter().copied());
        assert_eq!(neighbour_set.len(), 26);
    }

    #[test]
    pub fn test_invalid_keys() {
        let invalid_key = invalid_key();

        // Make sure that an invalid key is invalid.
        assert!(!is_valid(invalid_key));

        // Make sure that an invalid key is not ill-formed.
        assert!(is_well_formed(invalid_key));
    }

    #[test]
    pub fn test_neighbours() {
        // Check that root only has invalid neighbors.
        let neighbours = super::neighbours(root());
        for key in neighbours {
            assert!(!is_valid(key));
        }

        // Now check inside a tree that all neighbours exist and that their distance to key corresponds
        // to the corresponding directions vector.

        let index = [33, 798, 56];
        let level = 11;
        let key = from_index_and_level(index, level);
        let neighbours = super::neighbours(key);
        for (dir, key) in izip!(DIRECTIONS, neighbours) {
            assert!(is_valid(key));
            let (level, key_index) = decode(key);
            assert_eq!(super::level(key), level);
            let direction: [i64; 3] = [
                key_index[0] as i64 - index[0] as i64,
                key_index[1] as i64 - index[1] as i64,
                key_index[2] as i64 - index[2] as i64,
            ];
            assert_eq!(direction, dir);
        }
    }

    #[test]
    pub fn test_key_in_direction() {
        // Now check inside a tree that all neighbours exist and that their distance to key corresponds
        // to the corresponding directions vector.

        let index = [33, 798, 56];
        let dir = [2, 5, -3];
        let level = 11;
        let key = from_index_and_level(index, level);
        let new_key = key_in_direction(key, dir);

        let (new_level, new_index) = decode(new_key);
        assert_eq!(new_level, level);
        assert_eq!(new_index[0] as i64, index[0] as i64 + dir[0]);
        assert_eq!(new_index[1] as i64, index[1] as i64 + dir[1]);
        assert_eq!(new_index[2] as i64, index[2] as i64 + dir[2]);

        // Now test a direction that gives an invalid key.

        let dir = [-34, 798, 56];
        let new_key = key_in_direction(key, dir);
        assert!(!is_valid(new_key));
    }

    #[test]
    pub fn test_balanced() {
        // Balance the second level of a tree.

        let balanced = balance([from_index_and_level([0, 1, 0], 2)].as_slice(), root());

        assert!(is_complete_linear_octree(&balanced));

        // Try a few keys on deeper levels

        let key1 = from_index_and_level([17, 35, 48], 9);
        let key2 = from_index_and_level([355, 25, 67], 9);
        let key3 = from_index_and_level([0, 0, 0], 8);

        // Just make sure one is not ancestor of the other. Does not matter for routine.
        // But want to avoid for unit test checks.
        assert!(!is_ancestor(key3, key1));

        let balanced = balance([key1, key2, key3].as_slice(), root());

        assert!(is_complete_linear_octree(&balanced));

        // Let us now check balancing with respec to a single given key.

        // We start with all keys on level 1. We replace the first key
        // by its descendents two levels down and linearize. The
        // resulting octree is complete and linear but not balanced.
        // However, the subtree under [0, 0, 0,] on level 1 is balanced.

        let mut keys = vec![
            from_index_and_level([0, 0, 0], 1),
            from_index_and_level([0, 0, 1], 1),
            from_index_and_level([0, 1, 0], 1),
            from_index_and_level([1, 0, 0], 1),
            from_index_and_level([0, 1, 1], 1),
            from_index_and_level([1, 1, 0], 1),
            from_index_and_level([1, 0, 1], 1),
            from_index_and_level([1, 1, 1], 1),
        ];

        // We recurse twice to get 64 children on level 3 from the first box.
        let seed = from_index_and_level([0, 0, 0], 1);
        let children = children(seed).unwrap();
        let mut descendents = Vec::<MortonKey>::new();
        for child in children {
            descendents.extend(super::children(child).unwrap());
        }

        // We add all those children to the tree and linearize.
        keys.extend(descendents.iter());

        let keys = linearize(&keys);

        let subtree_balanced = balance(&keys, from_index_and_level([0, 0, 0], 1));

        // Check that this balanced subtree has 64 elements.

        assert_eq!(subtree_balanced.len(), 64);

        // Check that each of the subtree elements lives on level 3.

        for &key in &subtree_balanced {
            assert_eq!(level(key), 3);
        }
    }

    #[test]
    pub fn test_is_complete_linear_and_balanced() {
        // First we create an unbalanced Octree.
        // We start with all keys at level 1 and then recurse one of the keys
        // two times and linearize.

        let mut keys = vec![
            from_index_and_level([0, 0, 0], 1),
            from_index_and_level([0, 0, 1], 1),
            from_index_and_level([0, 1, 0], 1),
            from_index_and_level([1, 0, 0], 1),
            from_index_and_level([0, 1, 1], 1),
            from_index_and_level([1, 1, 0], 1),
            from_index_and_level([1, 0, 1], 1),
            from_index_and_level([1, 1, 1], 1),
        ];

        // We recurse twice to get 64 children on level 3 from the first box.
        let seed = from_index_and_level([0, 0, 0], 1);
        let children = children(seed).unwrap();
        let mut descendents = Vec::<MortonKey>::new();
        for child in children {
            descendents.extend(super::children(child).unwrap());
        }

        // We add all those children to the tree and linearize.
        keys.extend(descendents.iter());

        let keys = linearize(&keys);

        // This tree should be complete and linear.
        assert!(is_complete_linear_octree(&keys));

        // However, it should not be balanced.
        assert!(!is_complete_linear_and_balanced(&keys));

        // Now balance it.
        let keys = balance(&keys, root());
        // Now the balancing check should be true
        assert!(is_complete_linear_and_balanced(&keys));

        // The balanced tree should have 120 keys. It should have
        // 64 keys on level 3 and then all the other 7 boxes on level 1
        // should have been replaced by their refinement on level 2. Hence,
        // we have 64 + 56 = 120 keys.
        assert_eq!(keys.len(), 120);
    }

    #[test]
    pub fn test_from_physical_point() {
        let bounding_box = PhysicalBox::new([-2.0, -3.0, -1.0, 4.0, 5.0, 6.0]);

        let point = [1.5, -2.5, 5.0];
        let level = 10;

        let key = from_physical_point(point, &bounding_box, level);

        let physical_box = physical_box(key, &bounding_box);

        let coords = physical_box.coordinates();

        assert!(coords[0] <= point[0] && point[0] < coords[3]);
        assert!(coords[1] <= point[1] && point[1] < coords[4]);
        assert!(coords[2] <= point[2] && point[2] < coords[5]);

        // Now compute the box.
    }

    #[test]
    pub fn test_child_index() {
        let key = from_index_and_level([1, 501, 718], 10);

        let children = children(key).unwrap();

        for (index, child) in children.iter().enumerate() {
            assert_eq!(index, super::child_index(*child));
        }
    }

    #[test]
    pub fn test_finest_outer_descendent() {
        let key = from_index_and_level([0, 0, 0], 1);

        let finest_outer_descendent = finest_outer_descendent(key);

        assert_eq!(
            finest_outer_descendent,
            from_index_and_level([0, 0, 0], DEEPEST_LEVEL as usize)
        );

        let key = from_index_and_level([1, 1, 0], 1);
        let finest_outer_descendent = super::finest_outer_descendent(key);

        assert_eq!(
            finest_outer_descendent,
            from_index_and_level(
                [LEVEL_SIZE as usize - 1, LEVEL_SIZE as usize - 1, 0],
                DEEPEST_LEVEL as usize
            )
        );
    }

    #[test]
    fn test_complete_tree_shallow_subsets_cover_depth_16_domain() {
        // This interval oracle deliberately uses the encoded depth-16 Morton
        // position, rather than the completion helper's ancestry routines.
        fn assert_exact_partition(keys: &[MortonKey]) {
            let mut expected_start = 0_u64;
            for &key in keys {
                let cell_width = 1_u64 << (3 * (DEEPEST_LEVEL as usize - level(key)));
                assert_eq!(key >> LEVEL_DISPLACEMENT, expected_start);
                expected_start += cell_width;
            }
            assert_eq!(expected_start, 1_u64 << (3 * DEEPEST_LEVEL));
        }

        let root_children = children(root()).unwrap();
        for selected in 1_u16..(1_u16 << NSIBLINGS) {
            let input = root_children
                .iter()
                .enumerate()
                .filter_map(|(index, &key)| (selected & (1 << index) != 0).then_some(key))
                .collect_vec();
            let completed = complete_tree(&input);

            assert_exact_partition(&completed);
            assert!(is_complete_linear_octree(&completed));
            assert_eq!(complete_tree(&completed), completed);
            for key in input {
                assert!(completed.contains(&key));
            }
        }
    }

    #[test]
    fn test_encoding_boundary_matrix_all_levels() {
        for level in 0..=DEEPEST_LEVEL as usize {
            let maximum = (1 << level) - 1;
            for x in [0, maximum] {
                for y in [0, maximum] {
                    for z in [0, maximum] {
                        let key = from_index_and_level([x, y, z], level);
                        assert_eq!(decode(key), (level, [x, y, z]));
                        assert!(is_well_formed(key));
                    }
                }
            }

            if level >= 9 {
                let alternating = maximum & 0xaaaa;
                let key = from_index_and_level([255, 256, alternating], level);
                assert_eq!(decode(key), (level, [255, 256, alternating]));
            }
        }
    }

    #[test]
    fn test_balance_matches_geometric_touch_oracle() {
        // Each leaf is represented on the common deepest-level integer lattice.
        // Distinct complete-tree leaves only touch when their closed boxes intersect.
        fn is_two_to_one_by_geometry(keys: &[MortonKey]) -> bool {
            for (index, &first) in keys.iter().enumerate() {
                let (first_level, first_index) = decode(first);
                let first_width = 1_i64 << (DEEPEST_LEVEL as usize - first_level);
                for &second in &keys[index + 1..] {
                    let (second_level, second_index) = decode(second);
                    let second_width = 1_i64 << (DEEPEST_LEVEL as usize - second_level);
                    let touches = (0..3).all(|axis| {
                        let first_start = first_index[axis] as i64 * first_width;
                        let second_start = second_index[axis] as i64 * second_width;
                        first_start <= second_start + second_width
                            && second_start <= first_start + first_width
                    });
                    if touches && first_level.abs_diff(second_level) > 1 {
                        return false;
                    }
                }
            }
            true
        }

        let level_one = children(root()).unwrap();
        let refined_corner = children(level_one[0]).unwrap();
        let refined_face = children(level_one[7]).unwrap();
        let mut input = level_one.to_vec();
        input.extend(refined_corner);
        input.extend(refined_face);
        // This level-two cell touches unrefined level-one cells at x = 1/2.
        input.extend(children(from_index_and_level([1, 0, 0], 2)).unwrap());
        let input = linearize(&input);
        assert!(is_complete_linear_octree(&input));
        assert!(!is_two_to_one_by_geometry(&input));

        let balanced = balance(&input, root());
        assert!(is_complete_linear_octree(&balanced));
        assert!(is_two_to_one_by_geometry(&balanced));
        // Balancing must refine, not coarsen away the original fine cells.
        for &key in &balanced {
            let start = key >> LEVEL_DISPLACEMENT;
            let end = start + (1_u64 << (3 * (DEEPEST_LEVEL as usize - level(key))));
            assert!(input.iter().any(|&original| {
                let original_start = original >> LEVEL_DISPLACEMENT;
                let original_end =
                    original_start + (1_u64 << (3 * (DEEPEST_LEVEL as usize - level(original))));
                original_start <= start && end <= original_end
            }));
        }
        assert_eq!(balance(&balanced, root()), balanced);
    }

    #[test]
    fn regression_root_only_tree_is_balanced() {
        assert!(is_complete_linear_and_balanced(&[root()]));
    }

    #[test]
    fn regression_completeness_rejects_invalid_sentinel() {
        assert!(!is_complete_linear_octree(&[root(), invalid_key()]));
    }

    #[test]
    fn regression_well_formed_rejects_unsupported_level() {
        assert!(!is_well_formed(17));
    }

    #[test]
    fn regression_physical_point_rounding_stays_in_last_cell() {
        let bounding_box = PhysicalBox::new([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        let almost_one = f64::from_bits(1.0_f64.to_bits() - 1);
        let key = from_physical_point([almost_one, 0.0, 0.0], &bounding_box, 16);
        assert_eq!(decode(key).1[0], LEVEL_SIZE as usize - 1);
    }

    #[test]
    pub fn test_next_nondescendent_key() {
        let key = from_index_and_level([25, 17, 6], 5);

        let children = children(key).unwrap();

        // Check the next nondescendent key for the first six children

        for (child, next_child) in children.iter().tuple_windows() {
            let next_key = next_non_descendent_key(*child).unwrap();
            assert_eq!(level(next_key), DEEPEST_LEVEL as usize);
            assert!(!is_ancestor(*child, next_key));
            assert!(is_ancestor(*next_child, next_key));
        }

        // Now check the next nondescendent key from the last child.

        let next_child = next_non_descendent_key(*children.last().unwrap()).unwrap();

        // Check that the next nondescendent key from the parent is the same as that of the last child.

        assert_eq!(next_non_descendent_key(key).unwrap(), next_child);

        // Check that it is not a descendent of the parent and that its level is correct.

        assert_eq!(level(next_child), DEEPEST_LEVEL as usize);
        assert!(!is_ancestor(key, next_child));

        // Finally make sure that an ancestor of deepest last returns None.

        assert!(next_non_descendent_key(deepest_last()).is_none());
    }

    #[test]
    fn test_linearize_sorts_and_removes_overlaps() {
        assert!(linearize(&[]).is_empty());

        let key = from_index_and_level([3, 5, 7], 4);
        let children = children(key).unwrap();

        // Unsorted input with a duplicate and two ancestors of the other keys.
        let input = [children[5], key, children[0], children[5], root()];
        let linear = linearize(&input);

        // Ancestors are dropped in favour of their descendents, duplicates collapse.
        assert_eq!(linear, vec![children[0], children[5]]);

        // Linearization leaves an already linear list unchanged.
        assert_eq!(linearize(&linear), linear);

        // The deepest key of a pure ancestor chain is the only survivor.
        assert_eq!(linearize(&[key, root(), parent(key).unwrap()]), vec![key]);
    }

    #[test]
    fn test_siblings_agree_with_children_and_child_index() {
        // The root has no parent and therefore no siblings.
        assert_eq!(siblings(root()), None);

        let key = from_index_and_level([9, 2, 13], 6);
        let sibs = siblings(key).unwrap();

        assert_eq!(sibs, children(parent(key).unwrap()).unwrap());
        assert_eq!(sibs.iter().unique().count(), NSIBLINGS);
        // `child_index` is the position of a key within its own sibling list.
        assert_eq!(sibs[child_index(key)], key);
        for &sibling in &sibs {
            assert_eq!(parent(sibling), parent(key));
            assert_eq!(level(sibling), level(key));
            assert_eq!(siblings(sibling), Some(sibs));
        }

        // A deepest level key still has siblings even though it has no children.
        assert_eq!(children(deepest_last()), None);
        assert!(siblings(deepest_last()).unwrap().contains(&deepest_last()));
    }

    #[test]
    fn test_is_well_formed_detects_stray_index_bits() {
        // Every key built through the constructor is well formed on every level.
        for level in 0..=DEEPEST_LEVEL as usize {
            let last = (1 << level) - 1;
            for index in [[0, 0, 0], [last, last, last], [last, 0, last]] {
                assert!(is_well_formed(from_index_and_level(index, level)));
            }
        }

        // Relabel a deepest level key as a coarser key without clearing the index
        // bits that the coarser level cannot represent.
        let deepest = from_index_and_level([1, 1, 1], DEEPEST_LEVEL as usize);
        let ill_formed = (deepest & !LEVEL_MASK) | (DEEPEST_LEVEL - 1);
        assert_eq!(level(ill_formed), DEEPEST_LEVEL as usize - 1);
        assert!(!is_well_formed(ill_formed));
        // Clearing those bits makes the same relabelling well formed again.
        assert!(is_well_formed(ill_formed & !(7 << LEVEL_DISPLACEMENT)));

        // Well-formedness says nothing about the invalid marker: the marker sits
        // above the index bits, so validity has to be checked separately.
        assert!(is_well_formed(invalid_key()));
        assert!(!is_valid(invalid_key()));
    }

    #[test]
    fn test_interior_keys_are_the_ancestors_of_the_leafs() {
        let key = from_index_and_level([5, 3, 1], 3);

        let interior = get_interior_keys(&[key]);
        let expected = HashSet::<MortonKey>::from_iter(
            (0..3).map(|level| ancestor_at_level(key, level).unwrap()),
        );

        assert_eq!(interior, expected);
        assert!(interior.contains(&root()));
        // A leaf is never one of the interior keys.
        assert!(!interior.contains(&key));

        let all = get_interior_and_leaf_keys(&[key]);
        assert_eq!(all.len(), 1 + interior.len());
        assert!(all.contains(&key));
        assert!(all.is_superset(&interior));

        // Overlapping input is linearized first, so an ancestor of another key
        // becomes an interior key rather than a leaf.
        assert_eq!(get_interior_keys(&[key, root()]), expected);

        // A root-only tree has no interior keys at all.
        assert!(get_interior_keys(&[root()]).is_empty());
        assert_eq!(
            get_interior_and_leaf_keys(&[root()]),
            HashSet::<MortonKey>::from_iter([root()])
        );
    }

    #[test]
    fn test_is_complete_linear_octree_rejects_broken_trees() {
        let octants = children(root()).unwrap();
        assert!(is_complete_linear_octree(&octants));
        assert!(is_complete_linear_octree(&[root()]));

        // Sorted order is part of the definition, so a permutation is rejected.
        let mut unsorted = octants.to_vec();
        unsorted.swap(0, 7);
        assert!(!is_complete_linear_octree(&unsorted));

        // A missing sibling leaves the root with seven children.
        assert!(!is_complete_linear_octree(&octants[..7]));

        // An ancestor and its descendents may not both be leafs.
        let mut overlapping = octants.to_vec();
        overlapping.extend_from_slice(&children(octants[0]).unwrap());
        overlapping.sort_unstable();
        assert!(!is_complete_linear_octree(&overlapping));

        // An ill-formed key is rejected before any ancestry reasoning happens.
        let ill_formed =
            (from_index_and_level([1, 1, 1], DEEPEST_LEVEL as usize) & !LEVEL_MASK) | 1;
        assert!(!is_complete_linear_octree(&[ill_formed]));
    }

    #[test]
    fn test_physical_box_tiles_the_domain() {
        let domain = PhysicalBox::new([-1.0, 2.0, 0.5, 3.0, 6.0, 4.5]);

        // The root covers the whole domain.
        assert_eq!(
            physical_box(root(), &domain).coordinates(),
            domain.coordinates()
        );

        // A level two cell is a quarter of the domain along each axis.
        let key = from_index_and_level([1, 0, 3], 2);
        assert_eq!(
            physical_box(key, &domain).coordinates(),
            [0.0, 2.0, 3.5, 1.0, 3.0, 4.5]
        );

        // The eight children partition their parent into equal octants.
        let parent_box = physical_box(key, &domain).coordinates();
        let child_boxes = children(key)
            .unwrap()
            .map(|child| physical_box(child, &domain).coordinates());
        for child_box in child_boxes {
            for axis in 0..3 {
                assert_eq!(
                    child_box[3 + axis] - child_box[axis],
                    0.5 * (parent_box[3 + axis] - parent_box[axis])
                );
                assert!(parent_box[axis] <= child_box[axis]);
                assert!(child_box[3 + axis] <= parent_box[3 + axis]);
            }
        }
        assert_eq!(
            child_boxes
                .iter()
                .map(|c| c.map(f64::to_bits))
                .unique()
                .count(),
            NSIBLINGS
        );
        for axis in 0..3 {
            let lower = child_boxes
                .iter()
                .map(|c| c[axis])
                .reduce(f64::min)
                .unwrap();
            let upper = child_boxes
                .iter()
                .map(|c| c[3 + axis])
                .reduce(f64::max)
                .unwrap();
            assert_eq!([lower, upper], [parent_box[axis], parent_box[3 + axis]]);
        }
    }

    #[test]
    fn test_physical_box_centre_maps_back_to_its_key() {
        let domain = PhysicalBox::new([-1.0, 2.0, 0.5, 3.0, 6.0, 4.5]);

        for key in [
            root(),
            from_index_and_level([1, 0, 3], 2),
            deepest_first(),
            deepest_last(),
        ] {
            let coords = physical_box(key, &domain).coordinates();
            let centre = [
                0.5 * (coords[0] + coords[3]),
                0.5 * (coords[1] + coords[4]),
                0.5 * (coords[2] + coords[5]),
            ];
            assert_eq!(from_physical_point(centre, &domain, level(key)), key);
        }
    }
}
