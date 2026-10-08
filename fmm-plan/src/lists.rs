//! Interaction lists in index form, one set of views per level.
//!
//! [`Plan`](super::plan::Plan) holds, for every level l, a [`LevelLists`] with the
//! views of design §4.3 (`docs/design/fmm-plan-redesign.md`). Every entry is a `u32`
//! index: a box index on a level of the [`BoxIndex`](super::index::BoxIndex), or a leaf
//! index of its leaf numbering. No view contains a key, so no evaluation step looks a
//! key up.
//!
//! # Views
//!
//! Every view has one row per **target**, in target order:
//!
//! | View of level l | Rows | Row entries, in this order | Grouping |
//! | --- | --- | --- | --- |
//! | [`v`](LevelLists::v) | boxes of l | (source box on l, offset index), by offset index | per offset index: (targets, sources) |
//! | [`x`](LevelLists::x) | boxes of l | source leaf, by the leaf's (level, key) | – |
//! | [`m2m_local`](LevelLists::m2m_local) | boxes of l (parents) | (child box on l + 1, octant), by octant; rows only for `LocalInterior` parents | per octant: (parents, children) |
//! | [`m2m_global`](LevelLists::m2m_global) | boxes of l (parents) | the same; rows only for `Global` parents | per octant: (parents, children) |
//! | [`l2l`](LevelLists::l2l) | boxes of l (children) | (parent box on l − 1, octant); rows for non-ghost boxes on l ≥ 1 | per octant: (children, parents) |
//! | [`p2m`](LevelLists::p2m) | boxes of l | the local leaf of the box, if any | – |
//! | [`near`](LevelLists::near) | local leaves of l | source leaf, by the leaf's (level, key), the leaf itself included | – |
//! | [`w`](LevelLists::w) | local leaves of l | source box on l + 1, by box index | – |
//! | [`l2p`](LevelLists::l2p) | local leaves of l | the box of the leaf | – |
//!
//! The offset index of a V-list pair is the position of d = index(target) −
//! index(source), the difference of the `morton::decode` indices, in
//! [`V_LIST_DIRECTIONS`] (CONVENTIONS §3.12); see [`offset_index`]. The octant of a
//! parent–child pair is `morton::child_index(child)`, the order of `morton::children`.
//!
//! # Guarantees
//!
//! - **Rows cover their targets once each.** A box view of level l has exactly one row
//!   per box of the level, ghosts included (with empty rows), so row t lines up with
//!   chunk t of `level_buffer.chunks_mut(size)`. A leaf view of level l has exactly one
//!   row per local leaf of the level, and row r belongs to leaf
//!   `local_leaves(l).start + r`, so the rows line up with the CSR offsets of a leaf
//!   store restricted to that range.
//! - **The lists are those of the per-key rule.** For every non-ghost box, `v`, `w`, `x`
//!   and `near` minus the box itself hold exactly its V-, W-, X- and U-list of the
//!   per-key rule of [`interaction_manager`](crate::interaction_manager), translated to
//!   indices. Ghost boxes have empty rows in every view.
//! - **Entry levels.** V entries lie on l, W entries on l + 1, X entries on l − 1 and U
//!   entries on l − 1, l or l + 1 (2:1 balance). V and W entries are boxes of any kind,
//!   U and X entries are leaves.
//! - **Sorted rows.** Rows of a grouped view are strictly ascending in their group (offset
//!   index or octant), so a target meets each group at most once. Rows of `x` and `near`
//!   are strictly ascending in the entry leaf's (level, key) (P1,
//!   `docs/design/distributed-fmm.md` §5.1): on one rank that is ascending leaf index,
//!   and on several ranks it puts a ghost leaf where the one-rank row has it, so the
//!   sums of P2L and P2P are those of one rank. The rows of the other [`Csr`] views are
//!   strictly ascending in their entry.
//! - **At most once per batch.** For a fixed offset index d the source of a V pair is the
//!   target minus d, so each target, and each source, appears at most once in batch
//!   (l, d). A parent has one child per octant, so it appears at most once in each
//!   `m2m_*` batch; in `l2l` each child appears once in total. The batches of a view
//!   partition its pairs, and each batch lists its targets in ascending order.
//! - **Both views agree.** A grouped view holds the same pairs, with the same groups, in
//!   its rows and in its batches. A row is in group order, which is the order in which a
//!   walk over the batches 0, 1, 2, … meets that target, so both walks add the
//!   contributions of each target in the same order.
//!
//! # Cost
//!
//! The lists are built once per plan, in O(E log K) for E list entries and K held
//! boxes: the per-key rule produces keys, which are translated by binary search. A V
//! pair takes 6 bytes in its row and 8 bytes in its batch, about 2.6 kB per box of a
//! uniform tree; everything else adds about 0.25 kB per box (design §4.5).

use nd_octree::MortonKey;

use crate::interaction_manager::V_LIST_DIRECTIONS;

/// The number of V-list offsets, the length of [`V_LIST_DIRECTIONS`].
pub const NOFFSETS: usize = V_LIST_DIRECTIONS.len();

/// The number of child octants.
pub const NOCTANTS: usize = 8;

/// The table index of every d in {−3..3}³, at 49 (d_x + 3) + 7 (d_y + 3) + (d_z + 3),
/// or `u16::MAX` for d in {−1..1}³.
const OFFSET_LOOKUP: [u16; 343] = {
    let mut lookup = [u16::MAX; 343];
    let mut d = 0;
    while d < NOFFSETS {
        let [x, y, z] = V_LIST_DIRECTIONS[d];
        lookup[(49 * (x + 3) + 7 * (y + 3) + (z + 3)) as usize] = d as u16;
        d += 1;
    }
    lookup
};

/// Return the position of `direction` in [`V_LIST_DIRECTIONS`], or `None` if it is not
/// a V-list offset (CONVENTIONS §3.12).
pub fn offset_index(direction: [i64; 3]) -> Option<usize> {
    if direction
        .iter()
        .any(|component| !(-3..=3).contains(component))
    {
        return None;
    }
    let [x, y, z] = direction;
    let d = OFFSET_LOOKUP[(49 * (x + 3) + 7 * (y + 3) + (z + 3)) as usize];
    (d != u16::MAX).then_some(d as usize)
}

/// Return the offset `index(target) - index(source)` of two keys of one level.
pub(crate) fn offset(target: MortonKey, source: MortonKey) -> [i64; 3] {
    let (_, target_index) = nd_octree::morton::decode(target);
    let (_, source_index) = nd_octree::morton::decode(source);
    [0, 1, 2].map(|dim| target_index[dim] as i64 - source_index[dim] as i64)
}

/// Rows of `u32` entries in compressed sparse row form.
///
/// Row t holds `entries()[row_offsets()[t]..row_offsets()[t + 1]]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Csr {
    row_offsets: Vec<u32>,
    entries: Vec<u32>,
}

impl Csr {
    /// Return the entries of row `t`.
    ///
    /// # Panics
    ///
    /// Panics if `t >= self.nrows()`.
    pub fn row(&self, t: usize) -> &[u32] {
        &self.entries[self.row_offsets[t] as usize..self.row_offsets[t + 1] as usize]
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// Return the number of entries of all rows.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true if no row has an entry.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return the `nrows() + 1` row offsets, starting at 0, for upload to a device.
    pub fn row_offsets(&self) -> &[u32] {
        &self.row_offsets
    }

    /// Return the entries of all rows, row after row, for upload to a device.
    pub fn entries(&self) -> &[u32] {
        &self.entries
    }
}

/// Target-centric rows of (source, group) pairs, together with the same pairs grouped
/// by group.
///
/// Row t holds the sources of target t with their groups, strictly ascending in the
/// group. Batch g holds the pairs of group g as two parallel arrays (targets,
/// sources), targets strictly ascending. Used as [`VList`] (groups are offset indices)
/// and as [`Children`] and [`Parents`] (groups are octants).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupedCsr<G> {
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<G>,
    batch_offsets: Vec<u32>,
    batch_targets: Vec<u32>,
    batch_sources: Vec<u32>,
}

/// The V-list pairs of one level, by target (in offset-index order) and by offset index.
pub type VList = GroupedCsr<u16>;

/// The parent–child pairs of an upward pass on one level, by parent (in octant order)
/// and by octant. Targets are the parents on the level, sources their children on the
/// next finer level.
pub type Children = GroupedCsr<u8>;

/// The child–parent pairs of the downward pass on one level, by child and by octant.
/// Targets are the boxes of the level, sources their parents on the next coarser level;
/// each row has at most one entry.
pub type Parents = GroupedCsr<u8>;

impl<G: Copy + Into<usize>> GroupedCsr<G> {
    /// Return the sources of target `t` and their groups, ascending in the group.
    ///
    /// # Panics
    ///
    /// Panics if `t >= self.nrows()`.
    pub fn row(&self, t: usize) -> (&[u32], &[G]) {
        let range = self.row_offsets[t] as usize..self.row_offsets[t + 1] as usize;
        (&self.sources[range.clone()], &self.groups[range])
    }

    /// Return the pairs of group `g` as (targets, sources); targets ascending, each at
    /// most once.
    ///
    /// # Panics
    ///
    /// Panics if `g >= self.ngroups()`.
    pub fn batch(&self, g: usize) -> (&[u32], &[u32]) {
        let range = self.batch_offsets[g] as usize..self.batch_offsets[g + 1] as usize;
        (
            &self.batch_targets[range.clone()],
            &self.batch_sources[range],
        )
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    /// Return the number of groups: [`NOFFSETS`] for a [`VList`], [`NOCTANTS`] for
    /// [`Children`] and [`Parents`].
    pub fn ngroups(&self) -> usize {
        self.batch_offsets.len() - 1
    }

    /// Return the number of pairs.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Return true if there is no pair.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Return the `nrows() + 1` row offsets, starting at 0.
    pub fn row_offsets(&self) -> &[u32] {
        &self.row_offsets
    }

    /// Return the sources of all rows, row after row.
    pub fn sources(&self) -> &[u32] {
        &self.sources
    }

    /// Return the groups of all rows, parallel to [`sources`](Self::sources).
    pub fn groups(&self) -> &[G] {
        &self.groups
    }

    /// Return the `ngroups() + 1` batch offsets into
    /// [`batch_targets`](Self::batch_targets) and [`batch_sources`](Self::batch_sources).
    pub fn batch_offsets(&self) -> &[u32] {
        &self.batch_offsets
    }

    /// Return the targets of all batches, batch after batch.
    pub fn batch_targets(&self) -> &[u32] {
        &self.batch_targets
    }

    /// Return the sources of all batches, parallel to
    /// [`batch_targets`](Self::batch_targets).
    pub fn batch_sources(&self) -> &[u32] {
        &self.batch_sources
    }
}

impl GroupedCsr<u16> {
    /// Return the offset indices of all rows, parallel to [`sources`](Self::sources).
    pub fn offset_indices(&self) -> &[u16] {
        &self.groups
    }
}

impl GroupedCsr<u8> {
    /// Return the octants of all rows, parallel to [`sources`](Self::sources).
    pub fn octants(&self) -> &[u8] {
        &self.groups
    }
}

/// Every view of one level; see the [module documentation](self).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelLists {
    pub(crate) v: VList,
    pub(crate) x: Csr,
    pub(crate) m2m_local: Children,
    pub(crate) m2m_global: Children,
    pub(crate) l2l: Parents,
    pub(crate) p2m: Csr,
    pub(crate) near: Csr,
    pub(crate) w: Csr,
    pub(crate) l2p: Csr,
}

impl LevelLists {
    /// Return the V-list pairs whose target lies on this level (M2L).
    pub fn v(&self) -> &VList {
        &self.v
    }

    /// Return the X-list sources, as leaf indices, of every box of this level (P2L), by
    /// the leaf's (level, key).
    pub fn x(&self) -> &Csr {
        &self.x
    }

    /// Return the children of the `LocalInterior` boxes of this level (M2M, local pass).
    pub fn m2m_local(&self) -> &Children {
        &self.m2m_local
    }

    /// Return the children of the `Global` boxes of this level (M2M, global pass).
    pub fn m2m_global(&self) -> &Children {
        &self.m2m_global
    }

    /// Return the parent of every non-ghost box of this level (L2L).
    pub fn l2l(&self) -> &Parents {
        &self.l2l
    }

    /// Return the local leaf, as a leaf index, of every box of this level (P2M).
    pub fn p2m(&self) -> &Csr {
        &self.p2m
    }

    /// Return the near list (U-list and the leaf itself), as leaf indices, of every
    /// local leaf of this level (P2P), by the leaf's (level, key).
    pub fn near(&self) -> &Csr {
        &self.near
    }

    /// Return the W-list sources, as box indices on the next finer level, of every local
    /// leaf of this level (M2P).
    pub fn w(&self) -> &Csr {
        &self.w
    }

    /// Return the box index of every local leaf of this level (L2P).
    pub fn l2p(&self) -> &Csr {
        &self.l2p
    }
}

/// Error raised when a view outgrows its `u32` offsets.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Overflow;

/// Convert an offset or a count into `u32`.
fn to_u32(value: usize) -> Result<u32, Overflow> {
    u32::try_from(value).map_err(|_| Overflow)
}

/// Builds a [`Csr`] row by row.
#[derive(Default)]
pub(crate) struct CsrBuilder {
    row_offsets: Vec<usize>,
    entries: Vec<u32>,
}

impl CsrBuilder {
    /// Append a row; `entries` must already be in row order.
    pub(crate) fn push_row(&mut self, entries: impl IntoIterator<Item = u32>) {
        if self.row_offsets.is_empty() {
            self.row_offsets.push(0);
        }
        self.entries.extend(entries);
        self.row_offsets.push(self.entries.len());
    }

    pub(crate) fn finish(mut self) -> Result<Csr, Overflow> {
        if self.row_offsets.is_empty() {
            self.row_offsets.push(0);
        }
        Ok(Csr {
            row_offsets: self
                .row_offsets
                .into_iter()
                .map(to_u32)
                .collect::<Result<_, _>>()?,
            entries: self.entries,
        })
    }
}

/// Builds a [`GroupedCsr`] row by row, then forms the batches.
pub(crate) struct GroupedBuilder<G> {
    ngroups: usize,
    row_offsets: Vec<usize>,
    sources: Vec<u32>,
    groups: Vec<G>,
}

impl<G: Copy + Into<usize>> GroupedBuilder<G> {
    pub(crate) fn new(ngroups: usize) -> Self {
        Self {
            ngroups,
            row_offsets: vec![0],
            sources: Vec::new(),
            groups: Vec::new(),
        }
    }

    /// Append a row; `pairs` are (group, source) and must be strictly ascending in the
    /// group.
    pub(crate) fn push_row(&mut self, pairs: impl IntoIterator<Item = (G, u32)>) {
        for (group, source) in pairs {
            debug_assert!(group.into() < self.ngroups);
            debug_assert!(
                self.sources.len() == *self.row_offsets.last().unwrap()
                    || (*self.groups.last().unwrap()).into() < group.into(),
                "groups of a row must be strictly ascending"
            );
            self.groups.push(group);
            self.sources.push(source);
        }
        self.row_offsets.push(self.sources.len());
    }

    pub(crate) fn finish(self) -> Result<GroupedCsr<G>, Overflow> {
        let nrows = self.row_offsets.len() - 1;
        to_u32(nrows)?;

        // Counting sort by group. Rows are visited in target order, so each batch lists
        // its targets in ascending order.
        let mut batch_offsets = vec![0usize; self.ngroups + 1];
        for &group in &self.groups {
            batch_offsets[group.into() + 1] += 1;
        }
        for g in 0..self.ngroups {
            batch_offsets[g + 1] += batch_offsets[g];
        }
        let mut cursor = batch_offsets.clone();
        let mut batch_targets = vec![0u32; self.sources.len()];
        let mut batch_sources = vec![0u32; self.sources.len()];
        for t in 0..nrows {
            for e in self.row_offsets[t]..self.row_offsets[t + 1] {
                let position = &mut cursor[self.groups[e].into()];
                batch_targets[*position] = t as u32;
                batch_sources[*position] = self.sources[e];
                *position += 1;
            }
        }

        let convert =
            |offsets: Vec<usize>| offsets.into_iter().map(to_u32).collect::<Result<_, _>>();
        Ok(GroupedCsr {
            row_offsets: convert(self.row_offsets)?,
            sources: self.sources,
            groups: self.groups,
            batch_offsets: convert(batch_offsets)?,
            batch_targets,
            batch_sources,
        })
    }
}
