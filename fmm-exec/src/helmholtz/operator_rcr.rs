//! The RCR M2L level call by pairs (Helmholtz CH4.0i), a private part of the
//! [`operator`](super) module ([module documentation](super#execution)).
//!
//! The shared driver runs a Rotation level target by target, one task per box with its V
//! row in series, so a level of fewer boxes than threads keeps threads idle (level 2 of W1
//! at kD = 500: 64 boxes at p = 131, 28–48 % of linear on 40 and 72 threads, Helmholtz
//! CH4.0e). By pairs the level call takes the level's V pairs in the plan's order (the rows
//! one after the other, `GroupedCsr::row_offsets`), in chunks of consecutive pairs whose
//! contributions fit the buffer, and runs two passes per chunk on the pool:
//!
//! 1. each pair writes its contribution, the terms `HelmholtzTables::m2l` would add to the
//!    target's local (`HelmholtzTables::m2l_contribution`), into its own slot, in any order
//!    on any thread: the chunk's pairs in the order of their polar angle and distance
//!    (`HelmholtzTables::m2l_classes`), so that a thread's consecutive pairs share their
//!    y-blocks (kept in its scratch where they are formed per call) and coaxial table;
//! 2. each target of the chunk adds its pairs' slots in its row's order
//!    (`nd_fmm_tables::helmholtz::add_m2l_contribution`).
//!
//! Each value of a local receives the same terms, each computed by the same operations and
//! added by the same one rounding in the same order as target by target, so the output is
//! the shared driver's bit for bit, at every thread count and buffer size (a row may span
//! two chunks; its pairs still add in its order). The split may therefore read the thread
//! count ([`RcrSplit::Auto`]): it changes no bit (proposal §2, constraints 2 and 9).

use nd_fmm_plan::operator::M2l;
use nd_fmm_tables::helmholtz::add_m2l_contribution;

use super::{HelmholtzOperator, SimdScalar};
use crate::driver;

/// How a Rotation (RCR) M2L level call splits over the pool (Helmholtz CH4.0i; the module
/// documentation of [`operator`](super), "Execution"). The output is the same bit for bit
/// under every setting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RcrSplit {
    /// By pairs on a level of degree p_l ≥ [`RCR_PAIR_MIN_DEGREE`], also serially, and on a
    /// level of degree p_l ≥ [`RCR_PAIR_FEW_DEGREE`] whose targets with a V row are fewer
    /// than [`RCR_PAIR_SHARE`] times the pool's threads; by boxes elsewhere ([`rcr_by_pairs`]).
    #[default]
    Auto,
    /// Target by target, each target's V row on one thread (the shared driver).
    Boxes,
    /// By pairs on every Rotation level, also serially.
    Pairs,
}

/// Under [`RcrSplit::Auto`] a Rotation level of degree at least this runs by pairs at every
/// thread count (Helmholtz CH4.0i): on one Grace core the pair path takes 0.93× the box path
/// at p = 72 and 0.60–0.66× at p = 131 (the y-blocks kept and the coaxial tables reused in
/// class order), and in the FMM levels of p = 66–135 ran 0.26–0.98× by boxes at every
/// split on locust and Kathleen, while levels of p = 24 and thousands of boxes ran 1.03–1.12×
/// (fmm-validate/results/helmholtz-ch4-0i.md).
pub const RCR_PAIR_MIN_DEGREE: usize = 48;

/// Under [`RcrSplit::Auto`] a Rotation level of degree at least this whose targets with a V
/// row are fewer than [`RCR_PAIR_SHARE`] times the pool's threads runs by pairs (Helmholtz
/// CH4.0i): levels of 64 boxes at p = 30–38 ran 0.39–0.90× by boxes on 72 threads, levels of
/// p = 7–15 1.1–3.8× at every box count.
pub const RCR_PAIR_FEW_DEGREE: usize = 30;

/// Under [`RcrSplit::Auto`] a Rotation level of degree at least [`RCR_PAIR_FEW_DEGREE`] runs
/// by pairs where its targets with a V row are fewer than this many times the pool's threads
/// (Helmholtz CH4.0i).
pub const RCR_PAIR_SHARE: usize = 4;

/// The default byte bound of the Rotation M2L's buffer of contributions by pairs, 1 GiB
/// ([`HelmholtzOperator::with_rcr_buffer`]; Helmholtz CH4.0i): level 2 of W1 at kD = 500
/// (about 3,000 pairs of 279 KB at p = 131) in one chunk.
pub const DEFAULT_RCR_BUFFER: usize = 1 << 30;

/// Whether a Rotation level call of degree `degree` with `targets` targets with a V row on
/// `threads` threads runs by pairs under `split` (Helmholtz CH4.0i; [`RcrSplit`]). No bit
/// depends on it.
pub fn rcr_by_pairs(split: RcrSplit, degree: usize, targets: usize, threads: usize) -> bool {
    match split {
        RcrSplit::Boxes => false,
        RcrSplit::Pairs => true,
        RcrSplit::Auto => {
            degree >= RCR_PAIR_MIN_DEGREE
                || (threads > 1
                    && degree >= RCR_PAIR_FEW_DEGREE
                    && targets < RCR_PAIR_SHARE * threads)
        }
    }
}

/// The counts of the Rotation level calls by pairs since the operator's M2L was last chosen
/// ([`HelmholtzOperator::rcr_counts`]; Helmholtz CH4.0i).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RcrCounts {
    /// Level calls by pairs, per level.
    pub level_calls: Vec<usize>,
    /// Chunks of pairs, each with its contributions in the buffer, summed over the levels.
    pub chunks: usize,
}

/// The pair split's buffer: the contributions of a chunk's pairs, sized for the plan's
/// levels and the bound ([`HelmholtzOperator::with_rcr_buffer`]), so that no level call
/// allocates.
pub(super) struct RcrBuffer<T> {
    values: Vec<T>,
    /// A chunk's pairs (from its first) in the order of their slots, and each pair's slot.
    order: Vec<u32>,
    slot_of: Vec<u32>,
    /// The byte bound of `values`, or `None` for every pair of a level.
    pub(super) bound: Option<usize>,
    /// Each level's V pairs and targets with a V row on this rank, if the plan named them.
    pub(super) levels: Vec<(usize, usize)>,
    pub(super) split: RcrSplit,
    pub(super) counts: RcrCounts,
    /// The most bytes `values` held.
    pub(super) peak: usize,
}

impl<T> Default for RcrBuffer<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            order: Vec::new(),
            slot_of: Vec::new(),
            bound: Some(DEFAULT_RCR_BUFFER),
            levels: Vec::new(),
            split: RcrSplit::Auto,
            counts: RcrCounts::default(),
            peak: 0,
        }
    }
}

impl<T: SimdScalar> RcrBuffer<T> {
    /// The pairs a chunk of a level of `pairs` pairs with `size` values per contribution
    /// holds: every pair without a bound, else as many as the bound allows, at least
    /// `threads` (one per thread), at most `pairs`.
    fn capacity(&self, pairs: usize, size: usize, threads: usize) -> usize {
        let bounded = self.bound.map_or(pairs, |bytes| {
            (bytes / (size * size_of::<T>()).max(1)).max(threads.max(1))
        });
        bounded.min(pairs)
    }

    /// Sizes the buffer for the plan's levels on which a level call on `threads` threads
    /// runs by pairs, `rotation(l)` telling the Rotation levels, `degree(l)` their degrees and
    /// `size(l)` their values per contribution; empty where none does.
    pub(super) fn size(
        &mut self,
        threads: usize,
        rotation: impl Fn(usize) -> bool,
        degree: impl Fn(usize) -> usize,
        size: impl Fn(usize) -> usize,
    ) {
        let (mut values, mut slots) = (0, 0);
        for (l, &(pairs, targets)) in self.levels.iter().enumerate() {
            if l >= 2
                && pairs > 0
                && rotation(l)
                && rcr_by_pairs(self.split, degree(l), targets, threads)
            {
                let capacity = self.capacity(pairs, size(l), threads);
                values = values.max(capacity * size(l));
                slots = slots.max(capacity);
            }
        }
        self.values = vec![T::zero(); values];
        self.order = vec![0; slots];
        self.slot_of = vec![0; slots];
        self.peak = self.peak.max(self.bytes());
    }

    /// Its bytes.
    pub(super) fn bytes(&self) -> usize {
        self.values.len() * size_of::<T>() + 4 * (self.order.len() + self.slot_of.len())
    }
}

impl<T: SimdScalar> HelmholtzOperator<T> {
    /// The M2L level call of a Rotation level: by pairs or target by target ([`RcrSplit`];
    /// module documentation above).
    pub(super) fn rotation_m2l(&mut self, batch: M2l<'_, T>) {
        let threads = match (&self.execution.pool, self.execution.serial) {
            (Some(_), false) => self.execution.scratch.len(),
            _ => 1,
        };
        let row_offsets = batch.pairs.row_offsets();
        let targets = row_offsets.windows(2).filter(|w| w[1] > w[0]).count();
        let degree = self.kernel.tables.m2l_degree(batch.level);
        if batch.pairs.is_empty() || !rcr_by_pairs(self.rcr_buffer.split, degree, targets, threads)
        {
            return driver::m2l(&self.kernel, &mut self.execution, batch);
        }
        let M2l {
            level,
            index,
            pairs,
            multipoles,
            mut locals,
        } = batch;
        let kernel = &self.kernel;
        let (planes, m) = (
            kernel.densities.multipole_planes(),
            kernel.densities.vectors(),
        );
        let size = kernel.tables.m2l_contribution_len(level, m);
        assert_eq!(size, locals.size(), "a contribution is a local");
        let buffer = &mut self.rcr_buffer;
        let npairs = pairs.len();
        let capacity = buffer.capacity(npairs, size, threads);
        if buffer.values.len() < capacity * size || buffer.order.len() < capacity {
            // The plan did not size the buffer for this level.
            buffer
                .values
                .resize(buffer.values.len().max(capacity * size), T::zero());
            buffer.order.resize(buffer.order.len().max(capacity), 0);
            buffer.slot_of.resize(buffer.slot_of.len().max(capacity), 0);
            buffer.peak = buffer.peak.max(buffer.bytes());
        }
        if buffer.counts.level_calls.len() <= level {
            buffer.counts.level_calls.resize(level + 1, 0);
        }
        buffer.counts.level_calls[level] += 1;
        let (sources, offsets) = (pairs.sources(), pairs.groups());
        let locals = locals.as_mut_slice();
        let mut lo = 0;
        while lo < npairs {
            let hi = (lo + capacity).min(npairs);
            buffer.counts.chunks += 1;
            // The slots of the pairs lo..hi by angle and distance; no bit depends on it.
            let n = hi - lo;
            let (order, slot_of) = (&mut buffer.order[..n], &mut buffer.slot_of[..n]);
            for (i, o) in order.iter_mut().enumerate() {
                *o = i as u32;
            }
            order.sort_unstable_by_key(|&i| {
                (
                    kernel.tables.m2l_classes(offsets[lo + i as usize] as usize),
                    i,
                )
            });
            for (j, &i) in order.iter().enumerate() {
                slot_of[i as usize] = j as u32;
            }
            let (order, slot_of) = (&*order, &*slot_of);
            // 1. The contributions of the pairs lo..hi, each into its slot.
            self.execution
                .chunks(&mut buffer.values[..n * size], size, |j, slot, scratch| {
                    let g = lo + order[j] as usize;
                    let (s, d) = (sources[g] as usize, offsets[g] as usize);
                    kernel.tables.m2l_contribution(
                        level,
                        d,
                        multipoles.chunk(s),
                        planes,
                        slot,
                        m,
                        &mut scratch.tables,
                    );
                });
            // 2. The targets whose rows meet lo..hi add their pairs' slots in row order.
            let first = row_offsets.partition_point(|&r| (r as usize) <= lo) - 1;
            let last = row_offsets.partition_point(|&r| (r as usize) < hi);
            let slots = &buffer.values[..n * size];
            self.execution.chunks(
                &mut locals[first * size..last * size],
                size,
                |i, local, _| {
                    let t = first + i;
                    let start = (row_offsets[t] as usize).max(lo);
                    let end = (row_offsets[t + 1] as usize).min(hi);
                    for g in start..end {
                        debug_assert_eq!(
                            driver::offset_index(
                                index.key(level, sources[g] as usize),
                                index.key(level, t)
                            ),
                            offsets[g] as usize,
                            "the offset index of a V-list entry disagrees with its keys"
                        );
                        let j = slot_of[g - lo] as usize;
                        add_m2l_contribution(&slots[j * size..(j + 1) * size], local);
                    }
                },
            );
            lo = hi;
        }
    }
}
