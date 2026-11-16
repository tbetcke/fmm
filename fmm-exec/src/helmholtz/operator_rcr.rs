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
//!    on any thread;
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
    /// By pairs on a level whose targets with a V row are fewer than [`RCR_PAIR_SHARE`]
    /// times the pool's threads, by boxes elsewhere and without a pool.
    #[default]
    Auto,
    /// Target by target, each target's V row on one thread (the shared driver).
    Boxes,
    /// By pairs on every Rotation level, also serially.
    Pairs,
}

/// Under [`RcrSplit::Auto`] a Rotation level runs by pairs where its targets with a V row
/// are fewer than this many times the pool's threads (Helmholtz CH4.0i).
pub const RCR_PAIR_SHARE: usize = 4;

/// The default byte bound of the Rotation M2L's buffer of contributions by pairs, 1 GiB
/// ([`HelmholtzOperator::with_rcr_buffer`]; Helmholtz CH4.0i): level 2 of W1 at kD = 500
/// (about 3,000 pairs of 279 KB at p = 131) in one chunk.
pub const DEFAULT_RCR_BUFFER: usize = 1 << 30;

/// Whether a Rotation level call with `targets` targets with a V row on `threads` threads
/// runs by pairs under `split` (Helmholtz CH4.0i; [`RcrSplit`]).
pub fn rcr_by_pairs(split: RcrSplit, targets: usize, threads: usize) -> bool {
    match split {
        RcrSplit::Boxes => false,
        RcrSplit::Pairs => true,
        RcrSplit::Auto => threads > 1 && targets < RCR_PAIR_SHARE * threads,
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
    /// runs by pairs, `rotation(l)` telling the Rotation levels and `size(l)` their values per
    /// contribution; empty where none does.
    pub(super) fn size(
        &mut self,
        threads: usize,
        rotation: impl Fn(usize) -> bool,
        size: impl Fn(usize) -> usize,
    ) {
        let mut values = 0;
        for (l, &(pairs, targets)) in self.levels.iter().enumerate() {
            if l >= 2 && pairs > 0 && rotation(l) && rcr_by_pairs(self.split, targets, threads) {
                values = values.max(self.capacity(pairs, size(l), threads) * size(l));
            }
        }
        self.values = vec![T::zero(); values];
        self.peak = self.peak.max(self.bytes());
    }

    /// Its bytes.
    pub(super) fn bytes(&self) -> usize {
        self.values.len() * size_of::<T>()
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
        if batch.pairs.is_empty() || !rcr_by_pairs(self.rcr_buffer.split, targets, threads) {
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
        if buffer.values.len() < capacity * size {
            // The plan did not size the buffer for this level.
            buffer.values.resize(capacity * size, T::zero());
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
            // 1. The contributions of the pairs lo..hi, each into its slot.
            self.execution.chunks(
                &mut buffer.values[..(hi - lo) * size],
                size,
                |i, slot, scratch| {
                    let (s, d) = (sources[lo + i] as usize, offsets[lo + i] as usize);
                    kernel.tables.m2l_contribution(
                        level,
                        d,
                        multipoles.chunk(s),
                        planes,
                        slot,
                        m,
                        &mut scratch.tables,
                    );
                },
            );
            // 2. The targets whose rows meet lo..hi add their pairs' slots in row order.
            let first = row_offsets.partition_point(|&r| (r as usize) <= lo) - 1;
            let last = row_offsets.partition_point(|&r| (r as usize) < hi);
            let slots = &buffer.values[..(hi - lo) * size];
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
                        add_m2l_contribution(&slots[(g - lo) * size..(g - lo + 1) * size], local);
                    }
                },
            );
            lo = hi;
        }
    }
}
