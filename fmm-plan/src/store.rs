//! Data stores: level buffers for expansions, leaf stores for point data.
//!
//! The new plan keeps every value of an evaluation in a few flat allocations, addressed
//! by the indices of the [`BoxIndex`] (design §5, `docs/design/fmm-plan-redesign.md`).
//!
//! # Level buffers
//!
//! A [`LevelBuffers`] holds one kind of expansion (multipoles, or locals) for every box
//! of every level, in one allocation:
//!
//! ```text
//! | level 0: box 0, box 1, … | level 1: box 0, box 1, … | … | level L: … |
//!   ^ offsets()[0]             ^ offsets()[1]                  ^ offsets()[L]
//! ```
//!
//! Box i of level l holds `size(l)` values at `offsets()[l] + i * size(l)`. A level is
//! therefore a column-major `size(l) × len(l)` matrix, a GEMM operand, and
//! `chunks_mut(size(l))` of it lines up with the rows of every box view of the level
//! (requirement 3). Every box the rank holds has a slot, ghosts included; sizes may
//! differ between levels but must be positive. Levels are stored in order, so two
//! levels can be borrowed at once ([`LevelBuffers::parent_child_mut`],
//! [`LevelBuffers::child_parent_mut`]) and the whole buffer is one device upload.
//!
//! # Leaf stores
//!
//! A [`LeafStore`] holds point data per leaf in compressed sparse row form: leaf j has
//! `count(j)` points of `point_size()` values each, contiguous, after the points of
//! leaf j − 1 (requirement 4):
//!
//! ```text
//! data:          | leaf 0: count(0) · point_size | leaf 1: … | … |
//! point_offsets: 0, count(0), count(0) + count(1), …
//! ```
//!
//! Counts may be zero; such a leaf has an empty chunk. The point size may be zero too
//! (a target input an operator does not need); the counts are kept regardless. The
//! layout of the values inside a chunk is the operator's (CONVENTIONS §3.13 for
//! Laplace). An evaluation uses three stores, all in the leaf numbering of
//! [`LeafNumbering`](super::index::LeafNumbering):
//!
//! | Store | Leaves | Counts | Exchanged |
//! | --- | --- | --- | --- |
//! | sources | local, then the ghost leaves of the U- and X-lists | caller for local leaves; the owners' counts for ghosts, from [`SourceExchange`](super::exchange::SourceExchange) | ghost tail, forward |
//! | target input | local | caller | never |
//! | target output | local | the target-input counts | never |
//!
//! [`LeafStore::range`] and [`LeafStore::range_mut`] give a range of leaves as one slice
//! with its offsets, for example the local leaves of one level for a level call, or the
//! ghost tail for the source exchange.
//!
//! # Guarantees
//!
//! - Chunks are disjoint and tile their buffer in index order, without gaps.
//! - Every store is zero-initialised (`T::default()`), and `clear` restores that state
//!   without changing the layout.
//! - The slice types give per-box and per-leaf chunks without allocation, and split
//!   mutably with safe Rust, so a host operator can hand every target to one thread
//!   (design §6.5).

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;

use std::ops::Range;

use super::index::BoxIndex;

/// One kind of expansion for every box of every level, in one allocation.
///
/// See the [module documentation](self).
#[derive(Clone, Debug, PartialEq)]
pub struct LevelBuffers<T> {
    data: Vec<T>,
    /// `nlevels + 1` value offsets: level l is `offsets[l]..offsets[l + 1]`.
    offsets: Vec<usize>,
    /// Values per box, by level.
    sizes: Vec<usize>,
    /// Boxes, by level.
    lens: Vec<usize>,
}

/// The chunks of one level of a [`LevelBuffers`], box i at `i * size()`.
#[derive(Clone, Copy, Debug)]
pub struct LevelSlice<'a, T> {
    data: &'a [T],
    size: usize,
}

/// The chunks of one level of a [`LevelBuffers`], mutably, box i at `i * size()`.
#[derive(Debug)]
pub struct LevelSliceMut<'a, T> {
    data: &'a mut [T],
    size: usize,
}

impl<T: Copy + Default> LevelBuffers<T> {
    /// Create zeroed buffers with `lens[l]` boxes of `sizes[l]` values on level l.
    ///
    /// # Panics
    ///
    /// Panics if `lens` and `sizes` differ in length, a size is zero, or the total
    /// number of values overflows `usize`.
    pub fn new(lens: &[usize], sizes: &[usize]) -> Self {
        assert_eq!(lens.len(), sizes.len(), "one size per level is required");
        assert!(
            sizes.iter().all(|&size| size > 0),
            "level sizes must be positive"
        );
        let mut offsets = Vec::with_capacity(lens.len() + 1);
        let mut total = 0usize;
        offsets.push(total);
        for (&len, &size) in lens.iter().zip(sizes) {
            total = len
                .checked_mul(size)
                .and_then(|values| total.checked_add(values))
                .expect("the number of values overflows usize");
            offsets.push(total);
        }
        Self {
            data: vec![T::default(); total],
            offsets,
            sizes: sizes.to_vec(),
            lens: lens.to_vec(),
        }
    }

    /// Create zeroed buffers for every box of `index`, with `sizes[l]` values per box
    /// on level l.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and if `sizes` does not have one entry per level of
    /// `index`.
    pub fn from_index(index: &BoxIndex, sizes: &[usize]) -> Self {
        let lens: Vec<usize> = (0..index.nlevels()).map(|level| index.len(level)).collect();
        Self::new(&lens, sizes)
    }

    /// Return the number of levels.
    pub fn nlevels(&self) -> usize {
        self.sizes.len()
    }

    /// Return the number of boxes of `level`.
    pub fn len(&self, level: usize) -> usize {
        self.lens[level]
    }

    /// Return true if no level has a box.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of values per box of `level`.
    pub fn size(&self, level: usize) -> usize {
        self.sizes[level]
    }

    /// Return the `nlevels() + 1` value offsets: level l occupies
    /// `offsets()[l]..offsets()[l + 1]` of [`as_slice`](Self::as_slice).
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Return the chunks of `level`.
    pub fn level(&self, level: usize) -> LevelSlice<'_, T> {
        LevelSlice {
            data: &self.data[self.offsets[level]..self.offsets[level + 1]],
            size: self.sizes[level],
        }
    }

    /// Return the chunks of `level` mutably.
    pub fn level_mut(&mut self, level: usize) -> LevelSliceMut<'_, T> {
        LevelSliceMut {
            data: &mut self.data[self.offsets[level]..self.offsets[level + 1]],
            size: self.sizes[level],
        }
    }

    /// Return level `coarse` mutably and level `coarse + 1` shared, as M2M needs them.
    ///
    /// # Panics
    ///
    /// Panics if `coarse + 1 >= nlevels()`.
    pub fn parent_child_mut(&mut self, coarse: usize) -> (LevelSliceMut<'_, T>, LevelSlice<'_, T>) {
        let (coarse_size, fine_size) = (self.sizes[coarse], self.sizes[coarse + 1]);
        let (low, high) = self.split_levels(coarse);
        (
            LevelSliceMut {
                data: low,
                size: coarse_size,
            },
            LevelSlice {
                data: high,
                size: fine_size,
            },
        )
    }

    /// Return level `coarse + 1` mutably and level `coarse` shared, as L2L needs them.
    ///
    /// # Panics
    ///
    /// Panics if `coarse + 1 >= nlevels()`.
    pub fn child_parent_mut(&mut self, coarse: usize) -> (LevelSliceMut<'_, T>, LevelSlice<'_, T>) {
        let (coarse_size, fine_size) = (self.sizes[coarse], self.sizes[coarse + 1]);
        let (low, high) = self.split_levels(coarse);
        (
            LevelSliceMut {
                data: high,
                size: fine_size,
            },
            LevelSlice {
                data: low,
                size: coarse_size,
            },
        )
    }

    /// Return the values of levels `coarse` and `coarse + 1`, both mutably.
    fn split_levels(&mut self, coarse: usize) -> (&mut [T], &mut [T]) {
        let [start, middle, end] = [coarse, coarse + 1, coarse + 2].map(|l| self.offsets[l]);
        let (low, high) = self.data[start..end].split_at_mut(middle - start);
        (low, high)
    }

    /// Return the chunk of box `i` of `level`.
    pub fn chunk(&self, level: usize, i: usize) -> &[T] {
        let start = self.offsets[level] + i * self.sizes[level];
        &self.data[start..start + self.sizes[level]]
    }

    /// Return the chunk of box `i` of `level` mutably.
    pub fn chunk_mut(&mut self, level: usize, i: usize) -> &mut [T] {
        let start = self.offsets[level] + i * self.sizes[level];
        &mut self.data[start..start + self.sizes[level]]
    }

    /// Return every value, levels in order, for one device upload.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Return every value mutably, levels in order.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Reset every value to `T::default()`.
    pub fn clear(&mut self) {
        self.data.fill(T::default());
    }
}

impl<'a, T> LevelSlice<'a, T> {
    /// Return a level without boxes, with chunks of `size` values, for the level below
    /// the deepest one.
    pub(crate) fn empty(size: usize) -> Self {
        debug_assert!(size > 0, "level sizes must be positive");
        Self { data: &[], size }
    }

    /// Return the number of boxes.
    pub fn len(&self) -> usize {
        self.data.len() / self.size
    }

    /// Return true if the level has no box.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of values per box.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return the chunk of box `i`.
    pub fn chunk(&self, i: usize) -> &'a [T] {
        &self.data[i * self.size..(i + 1) * self.size]
    }

    /// Iterate over the chunks of the boxes, in index order.
    pub fn chunks(&self) -> std::slice::ChunksExact<'a, T> {
        self.data.chunks_exact(self.size)
    }

    /// Return the values of the level, box after box.
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<'a, T> LevelSliceMut<'a, T> {
    /// Return the number of boxes.
    pub fn len(&self) -> usize {
        self.data.len() / self.size
    }

    /// Return true if the level has no box.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of values per box.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Return the chunk of box `i`.
    pub fn chunk(&self, i: usize) -> &[T] {
        &self.data[i * self.size..(i + 1) * self.size]
    }

    /// Return the chunk of box `i` mutably.
    pub fn chunk_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.data[i * self.size..(i + 1) * self.size]
    }

    /// Iterate mutably over the chunks of the boxes, in index order; chunk t lines up
    /// with row t of every box view of the level.
    pub fn chunks_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        self.data.chunks_exact_mut(self.size)
    }

    /// Return the values of the level, box after box.
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Return the values of the level mutably, for example for `par_chunks_mut(size)`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }
}

/// Point data per leaf, compressed sparse row: leaf j holds `count(j)` points of
/// `point_size()` values.
///
/// See the [module documentation](self).
#[derive(Clone, Debug, PartialEq)]
pub struct LeafStore<T> {
    data: Vec<T>,
    /// `nleaves + 1` point offsets: leaf j holds points
    /// `point_offsets[j]..point_offsets[j + 1]`.
    point_offsets: Vec<usize>,
    point_size: usize,
}

/// A range of leaves of a [`LeafStore`] as one slice with its offsets.
///
/// Leaf r of the slice is leaf `range.start + r` of the store.
#[derive(Clone, Copy, Debug)]
pub struct LeafSlice<'a, T> {
    data: &'a [T],
    /// `nleaves + 1` point offsets of the store, not rebased.
    point_offsets: &'a [usize],
    point_size: usize,
}

/// A range of leaves of a [`LeafStore`] as one mutable slice with its offsets.
///
/// Leaf r of the slice is leaf `range.start + r` of the store.
#[derive(Debug)]
pub struct LeafSliceMut<'a, T> {
    data: &'a mut [T],
    /// `nleaves + 1` point offsets of the store, not rebased.
    point_offsets: &'a [usize],
    point_size: usize,
}

impl<T: Copy + Default> LeafStore<T> {
    /// Create a zeroed store with `counts[j]` points of `point_size` values for leaf j.
    ///
    /// Zero counts and a zero point size are allowed.
    ///
    /// # Panics
    ///
    /// Panics if the total number of values overflows `usize`.
    pub fn new(counts: &[usize], point_size: usize) -> Self {
        let mut point_offsets = Vec::with_capacity(counts.len() + 1);
        let mut total = 0usize;
        point_offsets.push(total);
        for &count in counts {
            total = total
                .checked_add(count)
                .expect("the number of points overflows usize");
            point_offsets.push(total);
        }
        let len = total
            .checked_mul(point_size)
            .expect("the number of values overflows usize");
        Self {
            data: vec![T::default(); len],
            point_offsets,
            point_size,
        }
    }

    /// Return the number of leaves.
    pub fn nleaves(&self) -> usize {
        self.point_offsets.len() - 1
    }

    /// Return the number of values per point.
    pub fn point_size(&self) -> usize {
        self.point_size
    }

    /// Return the number of points of leaf `j`.
    pub fn count(&self, j: usize) -> usize {
        self.point_offsets[j + 1] - self.point_offsets[j]
    }

    /// Return true if the store has one leaf per entry of `counts`, with those counts.
    pub fn has_counts(&self, counts: &[usize]) -> bool {
        counts.len() == self.nleaves()
            && counts
                .iter()
                .zip(self.point_offsets.windows(2))
                .all(|(&count, bounds)| bounds[1] - bounds[0] == count)
    }

    /// Return the `nleaves() + 1` point offsets: leaf j holds points
    /// `point_offsets()[j]..point_offsets()[j + 1]`, so its values start at
    /// `point_size() * point_offsets()[j]`.
    pub fn point_offsets(&self) -> &[usize] {
        &self.point_offsets
    }

    /// Return the values of leaf `j`.
    pub fn chunk(&self, j: usize) -> &[T] {
        &self.data[self.values(j..j + 1)]
    }

    /// Return the values of leaf `j` mutably.
    pub fn chunk_mut(&mut self, j: usize) -> &mut [T] {
        let values = self.values(j..j + 1);
        &mut self.data[values]
    }

    /// Return the leaves `range` as one slice with their offsets.
    pub fn range(&self, range: Range<usize>) -> LeafSlice<'_, T> {
        LeafSlice {
            data: &self.data[self.values(range.clone())],
            point_offsets: &self.point_offsets[range.start..range.end + 1],
            point_size: self.point_size,
        }
    }

    /// Return the leaves `range` as one mutable slice with their offsets.
    pub fn range_mut(&mut self, range: Range<usize>) -> LeafSliceMut<'_, T> {
        let values = self.values(range.clone());
        LeafSliceMut {
            data: &mut self.data[values],
            point_offsets: &self.point_offsets[range.start..range.end + 1],
            point_size: self.point_size,
        }
    }

    /// Return every value, leaf after leaf.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Return every value mutably, leaf after leaf.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Reset every value to `T::default()`; the counts stay.
    pub fn clear(&mut self) {
        self.data.fill(T::default());
    }

    /// The value range of the leaves `leaves`.
    fn values(&self, leaves: Range<usize>) -> Range<usize> {
        self.point_size * self.point_offsets[leaves.start]
            ..self.point_size * self.point_offsets[leaves.end]
    }
}

impl<'a, T> LeafSlice<'a, T> {
    /// Return the number of leaves.
    pub fn nleaves(&self) -> usize {
        self.point_offsets.len() - 1
    }

    /// Return true if the slice holds no value.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of values per point.
    pub fn point_size(&self) -> usize {
        self.point_size
    }

    /// Return the number of points of leaf `r` of the slice.
    pub fn count(&self, r: usize) -> usize {
        self.point_offsets[r + 1] - self.point_offsets[r]
    }

    /// Return the values of leaf `r` of the slice.
    pub fn chunk(&self, r: usize) -> &'a [T] {
        let base = self.point_offsets[0];
        &self.data[self.point_size * (self.point_offsets[r] - base)
            ..self.point_size * (self.point_offsets[r + 1] - base)]
    }

    /// Return the values of every leaf of the slice, leaf after leaf.
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<'a, T> LeafSliceMut<'a, T> {
    /// Return the number of leaves.
    pub fn nleaves(&self) -> usize {
        self.point_offsets.len() - 1
    }

    /// Return true if the slice holds no value.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of values per point.
    pub fn point_size(&self) -> usize {
        self.point_size
    }

    /// Return the number of points of leaf `r` of the slice.
    pub fn count(&self, r: usize) -> usize {
        self.point_offsets[r + 1] - self.point_offsets[r]
    }

    /// Return the values of leaf `r` of the slice.
    pub fn chunk(&self, r: usize) -> &[T] {
        let range = self.values(r);
        &self.data[range]
    }

    /// Return the values of leaf `r` of the slice mutably.
    pub fn chunk_mut(&mut self, r: usize) -> &mut [T] {
        let range = self.values(r);
        &mut self.data[range]
    }

    /// Iterate mutably over the values of the leaves, leaf after leaf, without
    /// allocation; item r belongs to leaf r of the slice.
    pub fn chunks_mut(&mut self) -> impl Iterator<Item = &mut [T]> {
        let point_size = self.point_size;
        let mut rest: &mut [T] = self.data;
        self.point_offsets.windows(2).map(move |bounds| {
            let (chunk, tail) =
                std::mem::take(&mut rest).split_at_mut(point_size * (bounds[1] - bounds[0]));
            rest = tail;
            chunk
        })
    }

    /// Split into leaves `0..mid` and `mid..nleaves()` of the slice, both mutable.
    ///
    /// # Panics
    ///
    /// Panics if `mid > nleaves()`.
    pub fn split_at_mut(&mut self, mid: usize) -> (LeafSliceMut<'_, T>, LeafSliceMut<'_, T>) {
        let at = self.point_size * (self.point_offsets[mid] - self.point_offsets[0]);
        let (left, right) = self.data.split_at_mut(at);
        (
            LeafSliceMut {
                data: left,
                point_offsets: &self.point_offsets[..mid + 1],
                point_size: self.point_size,
            },
            LeafSliceMut {
                data: right,
                point_offsets: &self.point_offsets[mid..],
                point_size: self.point_size,
            },
        )
    }

    /// Return the values of every leaf of the slice, leaf after leaf.
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Return the values of every leaf of the slice mutably, leaf after leaf.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// The value range of leaf `r` of the slice.
    fn values(&self, r: usize) -> Range<usize> {
        let base = self.point_offsets[0];
        self.point_size * (self.point_offsets[r] - base)
            ..self.point_size * (self.point_offsets[r + 1] - base)
    }
}
