//! The partition map: the coarse blocks and their owners as keys (Phase 5S S1;
//! docs/design/scale-out.md §3.2).

use std::ops::Range;

use crate::{
    constants::{DEEPEST_LEVEL, LEVEL_DISPLACEMENT},
    morton::{self, MortonKey},
};

use super::{KeyType, first_finest_descendant};

/// Where a key lies relative to the blocks, with the index of the block that holds its
/// first finest-level descendant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Placement {
    /// The key is block `b`.
    Block(usize),
    /// The key is an ancestor of blocks; `b` is the first block below it.
    Ancestor(usize),
    /// The key lies strictly inside block `b`.
    Below(usize),
}

/// The coarse blocks and their owners, replicated on every rank as keys (Phase 5S S1;
/// docs/design/scale-out.md §3.2).
///
/// [`crate::Octree::new`] builds the complete coarse tree and cuts it into contiguous
/// ranges of blocks, one per rank (a range may be empty). The map keeps both on every
/// rank, with one bit per block for whether the block is a leaf of the tree, so that any
/// rank computes, without communication, the owner of any key, the ranks that share a
/// key above the blocks, and the kind of any *top-tree key* (a block or an ancestor of
/// one) relative to itself.
///
/// The owner of a key at or below a block is the block's rank; the owner of a key above
/// the blocks is the rank of its first descendant block (the owner of S2,
/// docs/design/scale-out.md §4.1).
///
/// Memory per rank: 8 bytes per block for its key, one bit per block for the leaf flag
/// and 8 (P + 1) bytes for the cut of P ranks. At 36 blocks per rank that is about
/// 30 kB at 10² ranks, 300 kB at 10³ and 3 MB at 10⁴ (*model*, scale-out.md §8.2), the
/// one structure of the octree that grows with P. Every lookup is a binary search over
/// the blocks and one over the cut: O(log B + log P) for B blocks.
///
/// # Examples
///
/// ```
/// use nd_octree::{morton, octree::{KeyType, PartitionMap}};
/// // The eight octants, two per rank on four ranks, seen from rank 1; octant 5 is
/// // refined, the others are leaves.
/// let octants = morton::children(morton::root()).unwrap().to_vec();
/// let leaves = [true, true, true, true, true, false, true, true];
/// let map = PartitionMap::new(octants.clone(), vec![0, 2, 4, 6, 8], &leaves, 1);
/// assert_eq!(map.rank_blocks(2), 4..6);
/// assert_eq!(map.owner(morton::deepest_last()), 3);
/// assert_eq!(map.owner(morton::root()), 0);
/// assert_eq!(map.sharers(morton::root()), 0..4);
/// assert_eq!(map.top_kind(morton::root()), Some(KeyType::Global));
/// assert_eq!(map.top_kind(octants[2]), Some(KeyType::LocalLeaf));
/// assert_eq!(map.top_kind(octants[5]), Some(KeyType::GhostInterior(2)));
/// assert_eq!(map.top_kind(morton::deepest_first()), None);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionMap {
    blocks: Vec<MortonKey>,
    offsets: Vec<usize>,
    leaf_bits: Vec<u64>,
    rank: usize,
}

impl PartitionMap {
    /// Build the map from the replicated coarse tree and its cut.
    ///
    /// # Parameters
    ///
    /// - `blocks`: Every coarse block, ascending: a complete, linear tree.
    /// - `offsets`: The `P + 1` range boundaries of the cut: rank `r` owns the blocks
    ///   `offsets[r]..offsets[r + 1]`. They start at 0, do not decrease and end at the
    ///   number of blocks.
    /// - `leaf_blocks`: For every block, whether it is a leaf of the tree.
    /// - `rank`: The rank the kinds of [`PartitionMap::top_kind`] are relative to.
    ///
    /// # Panics
    ///
    /// Panics when there is no block, when `offsets` or `leaf_blocks` do not fit the
    /// blocks, or when `rank` is not below P. Debug builds also check that the blocks
    /// form a complete, linear tree.
    pub fn new(
        blocks: Vec<MortonKey>,
        offsets: Vec<usize>,
        leaf_blocks: &[bool],
        rank: usize,
    ) -> Self {
        assert!(
            !blocks.is_empty(),
            "a partition map needs at least one block"
        );
        assert!(
            offsets.len() >= 2
                && offsets[0] == 0
                && *offsets.last().unwrap() == blocks.len()
                && offsets.windows(2).all(|pair| pair[0] <= pair[1]),
            "the cut must run from 0 to the number of blocks without decreasing"
        );
        assert!(
            rank + 1 < offsets.len(),
            "rank {rank} is not below the number of ranks"
        );
        assert_eq!(
            leaf_blocks.len(),
            blocks.len(),
            "every block needs a leaf flag"
        );
        debug_assert!(
            morton::is_complete_linear_octree(&blocks),
            "the blocks must form a complete, linear tree"
        );
        let mut leaf_bits = vec![0u64; blocks.len().div_ceil(64)];
        for (b, _) in leaf_blocks.iter().enumerate().filter(|(_, leaf)| **leaf) {
            leaf_bits[b / 64] |= 1 << (b % 64);
        }
        Self {
            blocks,
            offsets,
            leaf_bits,
            rank,
        }
    }

    /// Every coarse block, ascending.
    pub fn blocks(&self) -> &[MortonKey] {
        &self.blocks
    }

    /// The number of ranks P.
    pub fn nranks(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The rank that [`PartitionMap::top_kind`] classifies relative to.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The blocks of `rank`, as a range of indices into [`PartitionMap::blocks`]; empty
    /// for a rank without blocks (O4).
    ///
    /// # Panics
    ///
    /// Panics when `rank` is not below P.
    pub fn rank_blocks(&self, rank: usize) -> Range<usize> {
        self.offsets[rank]..self.offsets[rank + 1]
    }

    /// Whether block `b` (an index into [`PartitionMap::blocks`]) is a leaf of the tree.
    ///
    /// # Panics
    ///
    /// Panics when `b` is not an index of a block.
    pub fn is_leaf_block(&self, b: usize) -> bool {
        assert!(b < self.blocks.len(), "block {b} out of range");
        self.leaf_bits[b / 64] >> (b % 64) & 1 == 1
    }

    /// The rank of block `b` (an index into [`PartitionMap::blocks`]). O(log P).
    ///
    /// # Panics
    ///
    /// Panics when `b` is not an index of a block.
    pub fn block_owner(&self, b: usize) -> usize {
        assert!(b < self.blocks.len(), "block {b} out of range");
        // The last rank whose range starts at or before `b`. A rank without blocks
        // starts where the next one does, so it is never the last such rank.
        self.offsets[..self.nranks()].partition_point(|&offset| offset <= b) - 1
    }

    /// The owner of `key`: the rank of the block containing it, or, for a key above
    /// the blocks, the rank of its first descendant block (S2's owner). Never a rank
    /// without blocks. O(log B + log P).
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid Morton key.
    pub fn owner(&self, key: MortonKey) -> usize {
        let b = match self.placement(key) {
            Placement::Block(b) | Placement::Ancestor(b) | Placement::Below(b) => b,
        };
        self.block_owner(b)
    }

    /// The ranks with a block at or below `key`, a contiguous range; the owner alone for
    /// a key at or below a block. A rank without blocks inside the range has none below
    /// `key` either. O(log B + log P).
    ///
    /// # Panics
    ///
    /// Panics when `key` is not a valid Morton key.
    pub fn sharers(&self, key: MortonKey) -> Range<usize> {
        match self.placement(key) {
            Placement::Block(b) | Placement::Below(b) => {
                let owner = self.block_owner(b);
                owner..owner + 1
            }
            Placement::Ancestor(first) => {
                let last = self.descendant_blocks(key, first).end - 1;
                self.block_owner(first)..self.block_owner(last) + 1
            }
        }
    }

    /// The kind of a top-tree key (a block or an ancestor of one) relative to
    /// [`PartitionMap::rank`], or `None` for a key below the blocks or an invalid key.
    /// No communication; O(log B + log P).
    ///
    /// - An own block is [`KeyType::LocalLeaf`] or [`KeyType::LocalInterior`], by its
    ///   leaf flag.
    /// - An ancestor of an own block, a *shared ancestor*, is [`KeyType::Global`].
    /// - Another rank's block is [`KeyType::GhostLeaf`] or [`KeyType::GhostInterior`]
    ///   of its owner.
    /// - An ancestor of other ranks' blocks only is [`KeyType::GhostInterior`] of its
    ///   owner, the rank of its first descendant block.
    ///
    /// On one rank the only block is the root, which is `Global` unless it is a leaf,
    /// as [`crate::Octree::all_keys`] has it since Phase 5.
    pub fn top_kind(&self, key: MortonKey) -> Option<KeyType> {
        if !morton::is_valid(key) {
            return None;
        }
        match self.placement(key) {
            Placement::Below(_) => None,
            Placement::Block(b) => {
                let owner = self.block_owner(b);
                let leaf = self.is_leaf_block(b);
                Some(match (owner == self.rank, leaf) {
                    (_, true) if self.nranks() == 1 => KeyType::LocalLeaf,
                    (_, false) if self.nranks() == 1 => KeyType::Global,
                    (true, true) => KeyType::LocalLeaf,
                    (true, false) => KeyType::LocalInterior,
                    (false, true) => KeyType::GhostLeaf(owner),
                    (false, false) => KeyType::GhostInterior(owner),
                })
            }
            Placement::Ancestor(first) => {
                let below = self.descendant_blocks(key, first);
                let own = self.rank_blocks(self.rank);
                let shares = own.start.max(below.start) < own.end.min(below.end);
                Some(if shares {
                    KeyType::Global
                } else {
                    KeyType::GhostInterior(self.block_owner(first))
                })
            }
        }
    }

    /// Where `key` lies relative to the blocks. O(log B).
    ///
    /// The block holding the key's first finest-level descendant is the block containing
    /// the key, the key itself, or the first block below it.
    pub(crate) fn placement(&self, key: MortonKey) -> Placement {
        assert!(
            morton::is_valid(key),
            "key {key:#x} is not a valid Morton key"
        );
        let first = first_finest_descendant(key);
        // The first block is the first key's block, so every finest key has one at or
        // before it.
        let b = self.blocks.partition_point(|&block| block <= first) - 1;
        let block = self.blocks[b];
        if block == key {
            Placement::Block(b)
        } else if morton::is_ancestor(key, block) {
            Placement::Ancestor(b)
        } else {
            Placement::Below(b)
        }
    }

    /// The blocks below the ancestor `key`, whose first block is `first`, as a range of
    /// indices. The blocks below a key are contiguous in Morton order.
    fn descendant_blocks(&self, key: MortonKey, first: usize) -> Range<usize> {
        let last = last_finest_descendant(key);
        first..self.blocks.partition_point(|&block| block <= last)
    }
}

/// The last finest-level descendant of a key: the finest key at its far corner, whose
/// position bits below the key's level are all set.
pub(crate) fn last_finest_descendant(key: MortonKey) -> MortonKey {
    let below = 3 * (DEEPEST_LEVEL - morton::level(key) as u64);
    let position = (key >> LEVEL_DISPLACEMENT) | ((1 << below) - 1);
    (position << LEVEL_DISPLACEMENT) | DEEPEST_LEVEL
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::{PartitionMap, Placement, last_finest_descendant};
    use crate::{
        constants::DEEPEST_LEVEL,
        morton::{self, MortonKey},
        octree::{KeyType, first_finest_descendant, owner_rank_for_key, tree_bins},
    };

    /// A graded coarse tree of 29 blocks on levels 1–3: the octants, octant 0 refined
    /// with its first child refined again, and octant 7 refined.
    fn graded_blocks() -> Vec<MortonKey> {
        let octants = morton::children(morton::root()).unwrap();
        let first = morton::children(octants[0]).unwrap();
        let mut blocks = morton::children(first[0]).unwrap().to_vec();
        blocks.extend_from_slice(&first[1..]);
        blocks.extend_from_slice(&octants[1..7]);
        blocks.extend_from_slice(&morton::children(octants[7]).unwrap());
        blocks.sort_unstable();
        assert!(morton::is_complete_linear_octree(&blocks));
        blocks
    }

    /// Every key of the tree whose leaves are `blocks`, plus one level below each block.
    fn keys_around(blocks: &[MortonKey]) -> Vec<MortonKey> {
        let mut keys = morton::get_interior_keys(blocks).into_iter().collect_vec();
        keys.extend_from_slice(blocks);
        for &block in blocks {
            keys.extend(morton::children(block).unwrap());
            keys.push(first_finest_descendant(block));
            keys.push(last_finest_descendant(block));
        }
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// The brute-force placement of `key`: the containing block, the key itself, or the
    /// first block below it.
    fn brute_force(blocks: &[MortonKey], key: MortonKey) -> Placement {
        if let Some(b) = blocks.iter().position(|&block| block == key) {
            return Placement::Block(b);
        }
        if let Some(b) = blocks
            .iter()
            .position(|&block| morton::is_ancestor(block, key))
        {
            return Placement::Below(b);
        }
        Placement::Ancestor(
            blocks
                .iter()
                .position(|&block| morton::is_ancestor(key, block))
                .unwrap(),
        )
    }

    #[test]
    fn test_last_finest_descendant() {
        assert_eq!(
            last_finest_descendant(morton::root()),
            morton::deepest_last()
        );
        let key = morton::from_index_and_level([1, 0, 1], 1);
        let half = 1usize << (DEEPEST_LEVEL - 1);
        assert_eq!(
            last_finest_descendant(key),
            morton::from_index_and_level(
                [2 * half - 1, half - 1, 2 * half - 1],
                DEEPEST_LEVEL as usize
            )
        );
        assert_eq!(
            last_finest_descendant(morton::deepest_first()),
            morton::deepest_first()
        );
        // Both corners against the decoded index, on keys of every level.
        let mut rng = crate::tools::seeded_rng(17);
        for key in crate::tools::generate_random_keys(200, &mut rng) {
            for level in 0..=DEEPEST_LEVEL as usize {
                let key = morton::ancestor_at_level(key, level).unwrap();
                let shift = DEEPEST_LEVEL as usize - level;
                let index = morton::decode(key).1;
                let corner = |offset: usize| {
                    morton::from_index_and_level(
                        index.map(|i| (i << shift) | offset),
                        DEEPEST_LEVEL as usize,
                    )
                };
                assert_eq!(first_finest_descendant(key), corner(0));
                assert_eq!(last_finest_descendant(key), corner((1 << shift) - 1));
            }
        }
    }

    /// Owners, sharers and kinds against brute force, on every key of a graded tree and
    /// one level below its blocks, for several cuts with empty ranks (O4) at the start,
    /// in the middle and at the end, and from every rank.
    #[test]
    fn test_partition_map_lookups_against_brute_force() {
        let blocks = graded_blocks();
        let nblocks = blocks.len();
        let leaves = (0..nblocks).map(|b| b % 3 != 1).collect_vec();
        let keys = keys_around(&blocks);
        let cuts = [
            vec![0, nblocks],
            vec![0, 9, 20, nblocks],
            vec![0, 0, 5, 5, 6, 25, nblocks, nblocks],
            vec![0, 1, 2, 3, nblocks],
        ];
        for offsets in cuts {
            let nranks = offsets.len() - 1;
            let owner_of_block =
                |b: usize| (0..nranks).find(|&r| (offsets[r]..offsets[r + 1]).contains(&b));
            for rank in 0..nranks {
                let map = PartitionMap::new(blocks.clone(), offsets.clone(), &leaves, rank);
                assert_eq!(map.nranks(), nranks);
                assert_eq!(map.rank(), rank);
                for (b, &leaf) in leaves.iter().enumerate() {
                    assert_eq!(Some(map.block_owner(b)), owner_of_block(b));
                    assert_eq!(map.is_leaf_block(b), leaf);
                }
                for &key in &keys {
                    let placement = brute_force(&blocks, key);
                    assert_eq!(map.placement(key), placement, "{key}");
                    let below = blocks
                        .iter()
                        .positions(|&block| morton::is_ancestor(key, block))
                        .collect_vec();
                    let (owner, sharers) = match placement {
                        Placement::Block(b) | Placement::Below(b) => {
                            let owner = owner_of_block(b).unwrap();
                            (owner, owner..owner + 1)
                        }
                        Placement::Ancestor(first) => (
                            owner_of_block(first).unwrap(),
                            owner_of_block(below[0]).unwrap()
                                ..owner_of_block(*below.last().unwrap()).unwrap() + 1,
                        ),
                    };
                    assert_eq!(map.owner(key), owner, "{key}");
                    assert_eq!(map.sharers(key), sharers, "{key}");
                    // The owner is never a rank without blocks.
                    assert!(!map.rank_blocks(owner).is_empty());
                    let expected = match placement {
                        Placement::Below(_) => None,
                        Placement::Block(b) if nranks == 1 => Some(if leaves[b] {
                            KeyType::LocalLeaf
                        } else {
                            KeyType::Global
                        }),
                        Placement::Block(b) => Some(match (owner == rank, leaves[b]) {
                            (true, true) => KeyType::LocalLeaf,
                            (true, false) => KeyType::LocalInterior,
                            (false, true) => KeyType::GhostLeaf(owner),
                            (false, false) => KeyType::GhostInterior(owner),
                        }),
                        Placement::Ancestor(_) => Some(
                            if below.iter().any(|&b| owner_of_block(b).unwrap() == rank) {
                                KeyType::Global
                            } else {
                                KeyType::GhostInterior(owner)
                            },
                        ),
                    };
                    assert_eq!(map.top_kind(key), expected, "{key} from rank {rank}");
                }
                assert_eq!(map.top_kind(morton::invalid_key()), None);
            }
        }
    }

    /// The owner of a finest key is the owner of the partition bounds, which
    /// `Octree::owner_rank` uses, with empty ranks anywhere in the cut.
    #[test]
    fn test_partition_map_owner_matches_the_bounds() {
        let blocks = graded_blocks();
        let nblocks = blocks.len();
        let leaves = vec![true; nblocks];
        for offsets in [
            vec![0, 0, 3, 3, 3, 17, nblocks, nblocks],
            vec![0, nblocks, nblocks, nblocks],
        ] {
            let bounds = tree_bins(&blocks, &offsets);
            let map = PartitionMap::new(blocks.clone(), offsets, &leaves, 0);
            for &block in &blocks {
                for key in [
                    first_finest_descendant(block),
                    last_finest_descendant(block),
                ] {
                    assert_eq!(Some(map.owner(key)), owner_rank_for_key(&bounds, key));
                }
            }
        }
    }

    /// The root: on one rank the only block, `Global` unless a leaf (Phase 5); on
    /// several ranks with the root as the only block, local to its owner and a ghost
    /// elsewhere; above the blocks, `Global` on the ranks with blocks and a ghost of
    /// the first block's rank on a rank without.
    #[test]
    fn test_partition_map_root() {
        let root = morton::root();
        for leaf in [false, true] {
            let map = PartitionMap::new(vec![root], vec![0, 1], &[leaf], 0);
            let expected = if leaf {
                KeyType::LocalLeaf
            } else {
                KeyType::Global
            };
            assert_eq!(map.top_kind(root), Some(expected));
            assert_eq!(map.owner(morton::deepest_last()), 0);
            assert_eq!(map.top_kind(morton::deepest_last()), None);

            for rank in 0..3 {
                let map = PartitionMap::new(vec![root], vec![0, 0, 1, 1], &[leaf], rank);
                let expected = match (rank == 1, leaf) {
                    (true, true) => KeyType::LocalLeaf,
                    (true, false) => KeyType::LocalInterior,
                    (false, true) => KeyType::GhostLeaf(1),
                    (false, false) => KeyType::GhostInterior(1),
                };
                assert_eq!(map.top_kind(root), Some(expected));
                assert_eq!(map.owner(root), 1);
                assert_eq!(map.sharers(root), 1..2);
            }
        }

        let octants = morton::children(root).unwrap().to_vec();
        let leaves = [true; 8];
        for (rank, expected) in [
            (0, KeyType::GhostInterior(1)),
            (1, KeyType::Global),
            (2, KeyType::Global),
            (3, KeyType::GhostInterior(1)),
        ] {
            let map = PartitionMap::new(octants.clone(), vec![0, 0, 3, 8, 8], &leaves, rank);
            assert_eq!(map.top_kind(root), Some(expected));
            assert_eq!(map.owner(root), 1);
            assert_eq!(map.sharers(root), 1..3);
        }
    }

    #[test]
    #[should_panic(expected = "the cut must run")]
    fn test_partition_map_rejects_a_short_cut() {
        let octants = morton::children(morton::root()).unwrap().to_vec();
        let _ = PartitionMap::new(octants, vec![0, 4], &[true; 8], 0);
    }

    #[test]
    #[should_panic(expected = "is not below the number of ranks")]
    fn test_partition_map_rejects_a_rank_beyond_the_cut() {
        let _ = PartitionMap::new(vec![morton::root()], vec![0, 1], &[true], 1);
    }
}
