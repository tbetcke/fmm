//! The plan's views and the box and leaf coordinates on the device (device-path.md
//! §3.1, §6.1 "Launch hygiene", §6.4 "Details").
//!
//! The crate does not depend on `nd-fmm-plan`, so each view is uploaded from plain
//! `u32` arrays with the layout of the plan's (`Csr`, `GroupedCsr`), and validated on
//! the host first: offsets start at 0, never decrease and end at the entry count, and
//! every index lies below the bound the caller states. A view that fails a check is a
//! defect of the caller and panics, as on the host path (device-path.md §12). Kernels
//! read the views without bounds checks; the bounds every launch wrapper checks are
//! those the [`IndexBuffer`]s record at upload.
//!
//! | Type | Holds | Upload |
//! | --- | --- | --- |
//! | [`IndexView`] | a CSR: row offsets and entries | 2 index buffers |
//! | [`GroupedView`] | target-centric rows of (source, group), the same pairs in batches by group, and the row-to-batch map | 7 index buffers |
//! | [`BoxCoordinates`] | the index (i_x, i_y, i_z) of every box of every level | 1 buffer of 3 `u32` per box |
//! | [`LeafCoordinates`] | the level and index of every leaf | 1 buffer of 4 `u32` per leaf |
//! | [`PointOffsets`] | the point offsets of a leaf store (T6) | 1 index buffer |
//!
//! Each array is a buffer of its own, because an [`IndexBuffer`] records one bound for
//! the whole buffer (T4), and arrays that address different matrices must not share
//! one. Every upload is counted by [`Device::counters`].
//!
//! **The row-to-batch map** (device-path.md §6.4): for row entry e = (t, s, g) of a
//! grouped view, `row_to_batch[e]` is the position k in the batch arrays with
//! `batch_targets[k] = t`, `batch_sources[k] = s` in batch g, the column of a grouped
//! product that holds the entry's contribution. [`row_to_batch`] builds it on the host
//! in O(E) by walking the batches in group order with one cursor per target, and the
//! walk checks on the way that the rows and the batches hold the same pairs.

use crate::buffer::{DeviceBuffer, IndexBuffer, IndexSlice};
use crate::device::Device;
use crate::error::KernelError;

/// The deepest level of a box, that of a Morton key: box indices on level l lie in
/// 0..2ˡ.
pub const MAX_LEVEL: u32 = 16;

/// Checks that `offsets` are CSR offsets over `len` entries: non-empty, starting at 0,
/// non-decreasing, ending at `len`.
fn check_offsets(what: &str, offsets: &[u32], len: usize) {
    assert!(
        !offsets.is_empty(),
        "{what}: the offsets need at least one element"
    );
    assert_eq!(offsets[0], 0, "{what}: the offsets must start at 0");
    if let Some(i) = offsets.windows(2).position(|w| w[1] < w[0]) {
        panic!(
            "{what}: offset {} = {} is below offset {i} = {}",
            i + 1,
            offsets[i + 1],
            offsets[i]
        );
    }
    assert_eq!(
        *offsets.last().unwrap() as usize,
        len,
        "{what}: the last offset must be the entry count"
    );
}

/// Checks that every index of `indices` lies below `bound`.
fn check_bound(what: &str, indices: &[u32], bound: usize) {
    if let Some((i, &v)) = indices
        .iter()
        .enumerate()
        .find(|&(_, &v)| v as usize >= bound)
    {
        panic!("{what}: entry {i} = {v} is not below the bound {bound}");
    }
}

/// A CSR on the device (`nd_fmm_plan::lists::Csr`): row r holds
/// `entries[row_offsets[r]..row_offsets[r + 1]]`.
#[derive(Debug)]
pub struct IndexView {
    row_offsets: IndexBuffer,
    entries: IndexBuffer,
    columns: usize,
}

impl IndexView {
    /// Validates the CSR on the host and uploads its two arrays (two uploads).
    /// `columns` bounds the entries: each addresses one of `columns` columns (boxes or
    /// leaves) of the data the view reads.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If the offsets are not CSR offsets over `entries`, or an entry is not below
    /// `columns`.
    pub fn upload(
        device: &mut Device,
        row_offsets: &[u32],
        entries: &[u32],
        columns: usize,
    ) -> Result<Self, KernelError> {
        check_offsets("IndexView", row_offsets, entries.len());
        check_bound("IndexView entries", entries, columns);
        Ok(Self {
            row_offsets: device.upload_indices(row_offsets)?,
            entries: device.upload_indices(entries)?,
            columns,
        })
    }

    /// The number of rows.
    pub fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no row has an entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The bound of the entries stated at upload.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The `nrows() + 1` row offsets.
    pub fn row_offsets(&self) -> IndexSlice<'_> {
        self.row_offsets.as_slice()
    }

    /// The entries, row after row.
    pub fn entries(&self) -> IndexSlice<'_> {
        self.entries.as_slice()
    }

    /// Downloads both arrays, for tests and reports: two downloads.
    ///
    /// # Errors
    ///
    /// As [`Device::download`].
    pub fn download(&self, device: &mut Device) -> Result<CsrImage, KernelError> {
        Ok(CsrImage {
            row_offsets: download_indices(device, &self.row_offsets)?,
            entries: download_indices(device, &self.entries)?,
        })
    }
}

/// The arrays of an [`IndexView`] on the host, as downloaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CsrImage {
    /// The row offsets.
    pub row_offsets: Vec<u32>,
    /// The entries.
    pub entries: Vec<u32>,
}

/// The host arrays of a grouped view, in the layout of `nd_fmm_plan::lists::GroupedCsr`:
/// rows by target with their sources and groups, and batches by group with their
/// targets and sources. `G` is the group type (`u16` offset indices, `u8` octants),
/// widened to `u32` at upload.
#[derive(Clone, Copy, Debug)]
pub struct GroupedArrays<'a, G> {
    /// `nrows + 1` offsets into `sources` and `groups`.
    pub row_offsets: &'a [u32],
    /// The sources of every row, row after row.
    pub sources: &'a [u32],
    /// The groups, parallel to `sources`, strictly ascending within each row.
    pub groups: &'a [G],
    /// `ngroups + 1` offsets into `batch_targets` and `batch_sources`.
    pub batch_offsets: &'a [u32],
    /// The targets of every batch, strictly ascending within each batch.
    pub batch_targets: &'a [u32],
    /// The sources, parallel to `batch_targets`.
    pub batch_sources: &'a [u32],
}

/// Validates the grouped arrays and returns their row-to-batch map (module
/// documentation): `map[e]` is the position in the batch arrays of row entry e. O(E)
/// for E pairs.
///
/// Checks that the rows and the batches are CSR, that sources lie below `sources_bound`,
/// batch targets below the row count and groups below the batch count, that each row's
/// groups and each batch's targets ascend strictly, and that the batches hold exactly
/// the pairs of the rows.
///
/// # Panics
///
/// If a check fails.
pub fn row_to_batch<G: Copy + Into<u32>>(
    arrays: &GroupedArrays<'_, G>,
    sources_bound: usize,
) -> Vec<u32> {
    let GroupedArrays {
        row_offsets,
        sources,
        groups,
        batch_offsets,
        batch_targets,
        batch_sources,
    } = *arrays;
    assert_eq!(
        sources.len(),
        groups.len(),
        "GroupedView: sources and groups differ in length"
    );
    assert_eq!(
        batch_targets.len(),
        batch_sources.len(),
        "GroupedView: batch targets and sources differ in length"
    );
    check_offsets("GroupedView rows", row_offsets, sources.len());
    check_offsets("GroupedView batches", batch_offsets, batch_targets.len());
    assert_eq!(
        batch_targets.len(),
        sources.len(),
        "GroupedView: the batches hold {} pairs, the rows {}",
        batch_targets.len(),
        sources.len()
    );
    let (nrows, ngroups) = (row_offsets.len() - 1, batch_offsets.len() - 1);
    check_bound("GroupedView sources", sources, sources_bound);
    check_bound("GroupedView batch sources", batch_sources, sources_bound);
    check_bound("GroupedView batch targets", batch_targets, nrows);
    for t in 0..nrows {
        let row = &groups[row_offsets[t] as usize..row_offsets[t + 1] as usize];
        let row: Vec<u32> = row.iter().map(|&g| g.into()).collect();
        check_bound("GroupedView groups", &row, ngroups);
        assert!(
            row.windows(2).all(|w| w[0] < w[1]),
            "GroupedView: the groups of row {t} do not ascend strictly: {row:?}"
        );
    }
    let mut cursor: Vec<u32> = row_offsets[..nrows].to_vec();
    let mut map = vec![0u32; sources.len()];
    for g in 0..ngroups {
        let batch = batch_offsets[g] as usize..batch_offsets[g + 1] as usize;
        assert!(
            batch_targets[batch.clone()].windows(2).all(|w| w[0] < w[1]),
            "GroupedView: the targets of batch {g} do not ascend strictly"
        );
        for k in batch {
            let t = batch_targets[k] as usize;
            let e = cursor[t] as usize;
            assert!(
                e < row_offsets[t + 1] as usize
                    && groups[e].into() == g as u32
                    && sources[e] == batch_sources[k],
                "GroupedView: batch {g} pairs target {t} with source {}, which row {t} \
                 does not hold at its next entry",
                batch_sources[k]
            );
            map[e] = k as u32;
            cursor[t] += 1;
        }
    }
    // Every pair of the batches was met in a row, and both hold as many pairs, so every
    // row entry was met.
    debug_assert!((0..nrows).all(|t| cursor[t] == row_offsets[t + 1]));
    map
}

/// A grouped view on the device (`nd_fmm_plan::lists::GroupedCsr`: the V list, the
/// children of an upward pass, the parents of the downward pass): the rows, the batches
/// and the row-to-batch map, each in an index buffer of its own, the groups widened to
/// `u32`.
#[derive(Debug)]
pub struct GroupedView {
    row_offsets: IndexBuffer,
    sources: IndexBuffer,
    groups: IndexBuffer,
    batch_offsets: IndexBuffer,
    batch_targets: IndexBuffer,
    batch_sources: IndexBuffer,
    row_to_batch: IndexBuffer,
    sources_bound: usize,
}

impl GroupedView {
    /// Validates the arrays and builds the row-to-batch map on the host
    /// ([`row_to_batch`]), then uploads the seven arrays (seven uploads). `sources_bound`
    /// is the number of boxes the sources address.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// As [`row_to_batch`].
    pub fn upload<G: Copy + Into<u32>>(
        device: &mut Device,
        arrays: &GroupedArrays<'_, G>,
        sources_bound: usize,
    ) -> Result<Self, KernelError> {
        let map = row_to_batch(arrays, sources_bound);
        Ok(Self {
            row_offsets: device.upload_indices(arrays.row_offsets)?,
            sources: device.upload_indices(arrays.sources)?,
            groups: device.upload_indices_widened(arrays.groups)?,
            batch_offsets: device.upload_indices(arrays.batch_offsets)?,
            batch_targets: device.upload_indices(arrays.batch_targets)?,
            batch_sources: device.upload_indices(arrays.batch_sources)?,
            row_to_batch: device.upload_indices(&map)?,
            sources_bound,
        })
    }

    /// The number of rows (targets).
    pub fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// The number of groups (batches).
    pub fn ngroups(&self) -> usize {
        self.batch_offsets.len() - 1
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// True if there is no pair.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The bound of the sources stated at upload.
    pub fn sources_bound(&self) -> usize {
        self.sources_bound
    }

    /// The `nrows() + 1` row offsets.
    pub fn row_offsets(&self) -> IndexSlice<'_> {
        self.row_offsets.as_slice()
    }

    /// The sources of every row, row after row.
    pub fn sources(&self) -> IndexSlice<'_> {
        self.sources.as_slice()
    }

    /// The groups, parallel to [`sources`](Self::sources).
    pub fn groups(&self) -> IndexSlice<'_> {
        self.groups.as_slice()
    }

    /// The `ngroups() + 1` batch offsets.
    pub fn batch_offsets(&self) -> IndexSlice<'_> {
        self.batch_offsets.as_slice()
    }

    /// The targets of every batch, batch after batch.
    pub fn batch_targets(&self) -> IndexSlice<'_> {
        self.batch_targets.as_slice()
    }

    /// The sources of every batch, parallel to [`batch_targets`](Self::batch_targets).
    pub fn batch_sources(&self) -> IndexSlice<'_> {
        self.batch_sources.as_slice()
    }

    /// The row-to-batch map, parallel to [`sources`](Self::sources).
    pub fn row_to_batch(&self) -> IndexSlice<'_> {
        self.row_to_batch.as_slice()
    }

    /// Downloads the seven arrays, for tests and reports: seven downloads.
    ///
    /// # Errors
    ///
    /// As [`Device::download`].
    pub fn download(&self, device: &mut Device) -> Result<GroupedImage, KernelError> {
        Ok(GroupedImage {
            row_offsets: download_indices(device, &self.row_offsets)?,
            sources: download_indices(device, &self.sources)?,
            groups: download_indices(device, &self.groups)?,
            batch_offsets: download_indices(device, &self.batch_offsets)?,
            batch_targets: download_indices(device, &self.batch_targets)?,
            batch_sources: download_indices(device, &self.batch_sources)?,
            row_to_batch: download_indices(device, &self.row_to_batch)?,
        })
    }
}

/// The arrays of a [`GroupedView`] on the host, as downloaded; the groups widened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedImage {
    /// The row offsets.
    pub row_offsets: Vec<u32>,
    /// The sources of every row.
    pub sources: Vec<u32>,
    /// The groups, widened to `u32`.
    pub groups: Vec<u32>,
    /// The batch offsets.
    pub batch_offsets: Vec<u32>,
    /// The targets of every batch.
    pub batch_targets: Vec<u32>,
    /// The sources of every batch.
    pub batch_sources: Vec<u32>,
    /// The row-to-batch map.
    pub row_to_batch: Vec<u32>,
}

/// Checks that `index` is a box index on `level`: level at most [`MAX_LEVEL`], every
/// component below 2ˡ.
fn check_box(what: &str, i: usize, level: u32, index: [u32; 3]) {
    assert!(
        level <= MAX_LEVEL,
        "{what} {i}: level {level} exceeds {MAX_LEVEL}"
    );
    assert!(
        index.iter().all(|&c| u64::from(c) < 1u64 << level),
        "{what} {i}: index {index:?} is not a box index on level {level}"
    );
}

/// The integer index (i_x, i_y, i_z) of every box of every level (CONVENTIONS §3.12),
/// three `u32` per box, level after level in box order: the frames between boxes come
/// from them on the device (device-path.md §6.1, "Frames from integer coordinates").
#[derive(Debug)]
pub struct BoxCoordinates {
    buffer: DeviceBuffer<u32>,
    /// `nlevels + 1` offsets in boxes.
    offsets: Vec<usize>,
}

impl BoxCoordinates {
    /// Validates and uploads the indices, `levels[l]` holding the boxes of level l in
    /// box order (one upload).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If a component of an index on level l is not below 2ˡ, or there are more than
    /// [`MAX_LEVEL`] + 1 levels.
    pub fn upload(device: &mut Device, levels: &[Vec<[u32; 3]>]) -> Result<Self, KernelError> {
        assert!(
            levels.len() <= MAX_LEVEL as usize + 1,
            "BoxCoordinates: {} levels",
            levels.len()
        );
        let mut offsets = vec![0];
        let mut flat = Vec::with_capacity(3 * levels.iter().map(Vec::len).sum::<usize>());
        for (level, boxes) in levels.iter().enumerate() {
            for (i, &index) in boxes.iter().enumerate() {
                check_box("BoxCoordinates: box", i, level as u32, index);
                flat.extend(index);
            }
            offsets.push(offsets.last().unwrap() + boxes.len());
        }
        Ok(Self {
            buffer: device.upload(&flat)?,
            offsets,
        })
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The number of boxes of `level`.
    pub fn len(&self, level: usize) -> usize {
        self.offsets[level + 1] - self.offsets[level]
    }

    /// True if no level has a box.
    pub fn is_empty(&self) -> bool {
        self.offsets.last() == Some(&0)
    }

    /// The position of box 0 of `level` in boxes: its first value is at three times it.
    pub fn offset(&self, level: usize) -> usize {
        self.offsets[level]
    }

    /// The buffer of every level, three values per box.
    pub fn buffer(&self) -> &DeviceBuffer<u32> {
        &self.buffer
    }

    /// Downloads the indices, level after level, for tests and reports: one download.
    ///
    /// # Errors
    ///
    /// As [`Device::download`].
    pub fn download(&self, device: &mut Device) -> Result<Vec<Vec<[u32; 3]>>, KernelError> {
        let flat = download_buffer(device, &self.buffer)?;
        let triples = flat.as_chunks::<3>().0;
        Ok(self
            .offsets
            .windows(2)
            .map(|w| triples[w[0]..w[1]].to_vec())
            .collect())
    }
}

/// The level and index of every leaf, four `u32` per leaf (l, i_x, i_y, i_z), in leaf
/// order: P2P, P2L and M2P form their frames from them (device-path.md §6.1).
#[derive(Debug)]
pub struct LeafCoordinates {
    buffer: DeviceBuffer<u32>,
    len: usize,
}

impl LeafCoordinates {
    /// Validates and uploads `leaves[j] = (level, index)` (one upload).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If a level exceeds [`MAX_LEVEL`] or a component of an index on level l is not
    /// below 2ˡ.
    pub fn upload(device: &mut Device, leaves: &[(u32, [u32; 3])]) -> Result<Self, KernelError> {
        let mut flat = Vec::with_capacity(4 * leaves.len());
        for (j, &(level, index)) in leaves.iter().enumerate() {
            check_box("LeafCoordinates: leaf", j, level, index);
            flat.push(level);
            flat.extend(index);
        }
        Ok(Self {
            buffer: device.upload(&flat)?,
            len: leaves.len(),
        })
    }

    /// The number of leaves.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if there is no leaf.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer, four values per leaf.
    pub fn buffer(&self) -> &DeviceBuffer<u32> {
        &self.buffer
    }

    /// Downloads the leaves, for tests and reports: one download.
    ///
    /// # Errors
    ///
    /// As [`Device::download`].
    pub fn download(&self, device: &mut Device) -> Result<Vec<(u32, [u32; 3])>, KernelError> {
        let flat = download_buffer(device, &self.buffer)?;
        Ok(flat
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&[l, x, y, z]| (l, [x, y, z]))
            .collect())
    }
}

/// The point offsets of a leaf store (`nd_fmm_plan::store::LeafStore`): leaf j holds
/// points `offsets[j]..offsets[j + 1]`, so its chunk of a store with s values per point
/// is `s offsets[j]..s offsets[j + 1]` (CONVENTIONS §3.13, "Source chunks", "Target
/// input and output"). Validated at upload (CSR offsets: from 0, never decreasing), so
/// that a kernel may read a leaf's count and chunk from them without a bounds check:
/// every chunk lies within `s` [`total`](Self::total) values (T6).
#[derive(Debug)]
pub struct PointOffsets {
    offsets: IndexBuffer,
    total: usize,
}

impl PointOffsets {
    /// Validates `offsets` (one more than the leaves) on the host and uploads them (one
    /// upload).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If `offsets` is empty, does not start at 0 or decreases.
    pub fn upload(device: &mut Device, offsets: &[u32]) -> Result<Self, KernelError> {
        let total = offsets.last().map_or(0, |&t| t as usize);
        check_offsets("PointOffsets", offsets, total);
        Ok(Self {
            offsets: device.upload_indices(offsets)?,
            total,
        })
    }

    /// The number of leaves.
    pub fn nleaves(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The number of points of every leaf together: the last offset.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The `nleaves() + 1` offsets.
    pub fn offsets(&self) -> IndexSlice<'_> {
        self.offsets.as_slice()
    }

    /// The index buffer of the offsets.
    pub fn buffer(&self) -> &IndexBuffer {
        &self.offsets
    }

    /// Downloads the offsets, for tests and reports: one download.
    ///
    /// # Errors
    ///
    /// As [`Device::download`].
    pub fn download(&self, device: &mut Device) -> Result<Vec<u32>, KernelError> {
        download_indices(device, &self.offsets)
    }
}

/// The whole of an index buffer, downloaded.
fn download_indices(device: &mut Device, indices: &IndexBuffer) -> Result<Vec<u32>, KernelError> {
    download_buffer(device, indices.buffer())
}

/// The whole of a buffer, downloaded.
fn download_buffer(
    device: &mut Device,
    buffer: &DeviceBuffer<u32>,
) -> Result<Vec<u32>, KernelError> {
    let mut out = vec![0; buffer.len()];
    device.download(buffer.as_slice(), &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The six arrays of a grouped view: row offsets, sources, groups, batch offsets,
    /// batch targets, batch sources.
    type Arrays = (Vec<u32>, Vec<u32>, Vec<u8>, Vec<u32>, Vec<u32>, Vec<u32>);

    /// Two targets, three groups: row 0 = [(5, 0), (7, 2)], row 1 = [(6, 0), (5, 1)].
    fn sample() -> Arrays {
        (
            vec![0, 2, 4],
            vec![5, 7, 6, 5],
            vec![0, 2, 0, 1],
            vec![0, 2, 3, 4],
            vec![0, 1, 1, 0],
            vec![5, 6, 5, 7],
        )
    }

    fn arrays<'a>(s: &'a Arrays) -> GroupedArrays<'a, u8> {
        GroupedArrays {
            row_offsets: &s.0,
            sources: &s.1,
            groups: &s.2,
            batch_offsets: &s.3,
            batch_targets: &s.4,
            batch_sources: &s.5,
        }
    }

    #[test]
    fn row_to_batch_points_at_the_matching_batch_entry() {
        let s = sample();
        let map = row_to_batch(&arrays(&s), 8);
        // Row entries (0, 5, g0), (0, 7, g2), (1, 6, g0), (1, 5, g1).
        assert_eq!(map, vec![0, 3, 1, 2]);
        for (e, &k) in map.iter().enumerate() {
            assert_eq!(s.5[k as usize], s.1[e], "entry {e}");
        }
    }

    #[test]
    fn empty_views_are_valid() {
        let empty: GroupedArrays<'_, u16> = GroupedArrays {
            row_offsets: &[0, 0, 0],
            sources: &[],
            groups: &[],
            batch_offsets: &[0; 317],
            batch_targets: &[],
            batch_sources: &[],
        };
        assert!(row_to_batch(&empty, 0).is_empty());
        check_offsets("csr", &[0], 0);
    }

    #[test]
    #[should_panic(expected = "does not hold at its next entry")]
    fn batches_with_another_pair_are_rejected() {
        let mut s = sample();
        s.5[1] = 4;
        row_to_batch(&arrays(&s), 8);
    }

    #[test]
    #[should_panic(expected = "do not ascend strictly")]
    fn unordered_groups_are_rejected() {
        let mut s = sample();
        s.2.swap(0, 1);
        row_to_batch(&arrays(&s), 8);
    }

    #[test]
    #[should_panic(expected = "not below the bound")]
    fn sources_out_of_bounds_are_rejected() {
        let s = sample();
        row_to_batch(&arrays(&s), 7);
    }

    #[test]
    #[should_panic(expected = "must be the entry count")]
    fn short_offsets_are_rejected() {
        check_offsets("csr", &[0, 1], 2);
    }

    #[test]
    #[should_panic(expected = "is below offset")]
    fn decreasing_offsets_are_rejected() {
        check_offsets("csr", &[0, 2, 1, 3], 3);
    }

    #[test]
    #[should_panic(expected = "is not a box index on level 2")]
    fn box_indices_beyond_the_level_are_rejected() {
        check_box("box", 0, 2, [3, 4, 0]);
    }
}
