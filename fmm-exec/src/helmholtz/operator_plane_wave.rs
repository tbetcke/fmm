//! The exponential M2L level call (Helmholtz CH6.1d; CONVENTIONS §3.14, "Merge and shift",
//! and §3.15, "Exponential M2L"), a private part of the [`operator`](super) module: its
//! scratch, its order and its pair body ([module documentation](super#m2l)).

use core::ops::Range;
use std::sync::PoisonError;
use std::time::{Duration, Instant};

use nd_fmm_math::Layout;
use nd_fmm_plan::lists::VList;
use nd_fmm_plan::operator::M2l;
use nd_fmm_plan::store::LevelSlice;
use nd_fmm_tables::PlaneWaveTables;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::plane_wave::merge::{MergeLevel, PlaneWaveM2l, add, shift};
use nd_fmm_tables::helmholtz::plane_wave::pair::shift_add;
use nd_fmm_tables::helmholtz::plane_wave::{PlaneWaveLevel, PlaneWaveLevelScratch};
use nd_fmm_tables::plane_wave::geometry::slot_of;
use nd_fmm_tables::plane_wave::{CHILDREN, Direction, MergeGeometry, SLOTS};
use nd_octree::morton;

use super::{HelmholtzOperator, Kernel, Scratch, SimdScalar};
use crate::driver::{BoxCall, PARENT_GRAIN};

/// The slot of the source of V pair (child index o, offset index d) at `8 d + o`, or
/// [`NO_SLOT`] (`nd_fmm_tables::plane_wave::geometry::slot_of`, tabulated once).
static SLOT_OF: std::sync::LazyLock<[u8; 8 * M2L_OFFSET_COUNT]> = std::sync::LazyLock::new(|| {
    let offsets = m2l_offsets();
    core::array::from_fn(|i| slot_of(i % 8, offsets[i / 8]).map_or(NO_SLOT, |slot| slot as u8))
});

/// No slot: the offset does not reach the child from the parent's slots.
const NO_SLOT: u8 = u8::MAX;

/// A slot the parent's computed children do not name.
const NO_BOX: u32 = u32::MAX;

/// The exponential M2L's part of one thread's scratch: a box's multipole in a frame and its
/// modes (outgoing); a parent's slots, accumulators and children's incoming waves, a
/// child's sums and local in the frame (merge and shift); and per level with waves the
/// pair translation's scratch (the pair body). Sized at construction for the largest
/// level: 4n + M + S + (a + 16) W values and 216 slots, with n = (p + 1)², M and S the
/// modes and sums, W the reals of one direction's waves and a the most accumulators of a
/// direction; and per level with waves its own 4n + M + S + 4W.
pub(super) struct PlaneScratch<T> {
    rotated: Vec<T>,
    modes: Vec<T>,
    slots: [u32; SLOTS],
    present: Vec<bool>,
    accs: Vec<T>,
    incoming: Vec<T>,
    sums: Vec<T>,
    frame: Vec<T>,
    pairs: Vec<Option<PlaneWaveLevelScratch<T>>>,
    /// The values of `pairs`.
    pair_values: usize,
    /// This thread's time in merge and shift's parts since the level call last drained it.
    parts: Parts,
}

/// One thread's time in merge and shift's parts ([`PlaneWaveTimes`]'s thread-seconds).
#[derive(Clone, Copy, Debug, Default)]
struct Parts {
    shifts: Duration,
    children: Duration,
    conversions: Duration,
    busy: Duration,
}

impl<T: SimdScalar> PlaneScratch<T> {
    /// The scratch of `m2l`'s levels.
    pub(super) fn new(m2l: &PlaneWaveM2l<T>) -> Self {
        let levels = || {
            (0..m2l.nlevels())
                .filter_map(|l| m2l.level(l))
                .map(MergeLevel::tables)
        };
        let largest = |f: fn(&PlaneWaveLevel<T>) -> usize| levels().map(f).max().unwrap_or(0);
        let n = largest(|t| Layout::new(t.p()).len());
        let w = largest(PlaneWaveLevel::wave_len);
        let a = m2l.geometry().max_accumulators();
        Self {
            rotated: vec![T::zero(); 2 * n],
            modes: vec![T::zero(); largest(PlaneWaveLevel::mode_len)],
            slots: [NO_BOX; SLOTS],
            present: vec![false; a],
            accs: vec![T::zero(); a * w],
            incoming: vec![T::zero(); 2 * CHILDREN * w],
            sums: vec![T::zero(); largest(PlaneWaveLevel::sum_len)],
            frame: vec![T::zero(); 2 * n],
            pairs: (0..m2l.nlevels())
                .map(|l| m2l.level(l).map(|level| level.tables().scratch()))
                .collect(),
            // `PlaneWaveLevel::scratch`: the frame multipole and local, the modes, the up,
            // down and incoming waves and the factors, the sums.
            pair_values: levels()
                .map(|t| {
                    4 * Layout::new(t.p()).len() + t.mode_len() + 4 * t.wave_len() + t.sum_len()
                })
                .sum(),
            parts: Parts::default(),
        }
    }

    /// Its bytes: the values, the pair scratch of every level and the slots.
    pub(super) fn bytes(&self) -> usize {
        let values = self.rotated.len()
            + self.modes.len()
            + self.accs.len()
            + self.incoming.len()
            + self.sums.len()
            + self.frame.len()
            + self.pair_values;
        values * size_of::<T>() + self.present.len() + 4 * SLOTS
    }
}

impl<T: SimdScalar> Kernel<T> {
    /// The exponential M2L's tables of `level`, or `None` where the level runs another M2L.
    #[inline]
    pub(super) fn plane_wave_level(&self, level: usize) -> Option<&MergeLevel<T>> {
        self.plane_waves.as_ref().and_then(|m2l| m2l.level(level))
    }

    /// The pair body on a level with waves: each vector's pair translated along the
    /// direction merge and shift gives it (`PlaneWaveTables::pair_direction`, from the
    /// offset index `d` and the target's child index), by `PlaneWaveLevel::m2l`.
    pub(super) fn plane_wave_pair(
        &self,
        merge: &MergeLevel<T>,
        (level, d, child): (usize, usize, usize),
        multipole: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let tables = merge.tables();
        let planes = self.densities.multipole_planes();
        let n = Layout::new(tables.p()).len();
        let direction = PlaneWaveTables::<T>::pair_direction(d, child);
        let scratch = s
            .plane
            .as_mut()
            .and_then(|plane| plane.pairs[level].as_mut())
            .expect("a level with waves has its pair scratch");
        let multipoles = multipole.chunks_exact(planes * n);
        for (x, y) in multipoles.zip(local.chunks_exact_mut(2 * n)) {
            tables.m2l(d, direction, x, planes, y, scratch);
        }
    }
}

/// The fewest boxes the exponential M2L's waves buffer holds on a level (or the level's
/// boxes, if fewer): one parent's 216 slots, the most distinct sources its children read.
pub const MIN_BUFFER_BOXES: usize = SLOTS;

/// No position: the box is not a source of the chunk.
const NO_POSITION: u32 = u32::MAX;

/// The level call's buffers (module documentation, "M2L"): the up and down waves of one
/// frame of a chunk's sources, each source's position in it, the chunk's sources in that
/// order, and the counts of the level calls so far. Sized for the plan's levels and the
/// bound ([`HelmholtzOperator::with_plane_wave_buffer`]), so no level call allocates.
pub(super) struct PlaneBuffer<T> {
    waves: Vec<T>,
    position: Vec<u32>,
    sources: Vec<u32>,
    /// The byte bound of `waves`, or `None` for every box of a level.
    pub(super) bound: Option<usize>,
    /// The boxes of each level, own and ghost, if the plan named them.
    pub(super) level_boxes: Vec<usize>,
    pub(super) counts: PlaneWaveCounts,
    /// How merge and shift splits a chunk over the pool.
    pub(super) split: PlaneWaveSplit,
}

/// The counts of the exponential M2L's level calls since the operator was built or its M2L
/// last chosen ([`HelmholtzOperator::plane_wave_counts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaneWaveCounts {
    /// Level calls by merge and shift.
    pub level_calls: usize,
    /// Chunks of parents, each with its sources' waves in the buffer.
    pub chunks: usize,
    /// Boxes whose waves were computed, once per chunk, vector and frame: the sources of a
    /// chunk's computed children, a source read by several chunks once in each.
    pub waves: usize,
    /// Chunks whose merge and shift ran by child ([`PlaneWaveSplit`]; Helmholtz CH6.2f).
    pub by_child: usize,
}

/// How the exponential M2L's level call splits merge and shift over the pool (Helmholtz
/// CH6.2f; [`HelmholtzOperator::with_plane_wave_split`]). Every setting gives every local
/// the same arithmetic in the same order, so the output is the same bit for bit; only the
/// work and its spread over the threads differ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaneWaveSplit {
    /// By child on a chunk where the split by parents, which never cuts a piece below
    /// `PARENT_GRAIN` (64) boxes, makes at most half as many pieces holding a computed box
    /// as the pool has threads (the split by child repeats shared accumulators, measured
    /// 1.2–3× the parents' thread time on CH6.2f's cases); by parents elsewhere and without a
    /// pool.
    #[default]
    Auto,
    /// By parents: the pool halves a chunk at parent boundaries down to pieces of 64 boxes,
    /// and each parent's accumulators serve all its computed children.
    Parents,
    /// By child: one task per box of the chunk, each computed child summing the
    /// accumulators that serve it (siblings computing a shared accumulator again) and
    /// converting its waves.
    Children,
}

/// The wall time of the two passes of the exponential M2L's level calls on one level, summed
/// over its level calls, chunks, vectors and frames since the operator was built or its M2L
/// last chosen ([`HelmholtzOperator::plane_wave_times`]; Helmholtz CH6.2d). The rest of the
/// level's [`m2l_times`](HelmholtzOperator::m2l_times) entry is the chunking between them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlaneWaveTimes {
    /// The waves of each chunk's sources, box by box on the pool.
    pub waves: Duration,
    /// Merge and shift, by parents or by child on the pool ([`PlaneWaveSplit`]).
    pub parents: Duration,
    /// Merge and shift's thread-seconds (Helmholtz CH6.2f): the time of its tasks summed
    /// over the threads, so `busy` over `parents` is the threads it kept busy.
    pub busy: Duration,
    /// The thread-seconds of the accumulators, each summing its slots' waves times their
    /// source shifts (part of `busy`).
    pub shifts: Duration,
    /// The thread-seconds of the children's sums of their accumulators and their child
    /// shifts (part of `busy`).
    pub children: Duration,
    /// The thread-seconds of the children's conversions of their waves into locals, rotated
    /// back (part of `busy`).
    pub conversions: Duration,
}

impl<T> Default for PlaneBuffer<T> {
    fn default() -> Self {
        Self {
            waves: Vec::new(),
            position: Vec::new(),
            sources: Vec::new(),
            bound: None,
            level_boxes: Vec::new(),
            counts: PlaneWaveCounts::default(),
            split: PlaneWaveSplit::Auto,
        }
    }
}

impl<T: SimdScalar> PlaneBuffer<T> {
    /// The boxes a level call on a level of `boxes` boxes with `box_waves` reals of waves per
    /// box holds in the buffer: every box without a bound, else as many as the bound allows,
    /// at least [`MIN_BUFFER_BOXES`], at most `boxes`.
    pub(super) fn capacity(&self, boxes: usize, box_waves: usize) -> usize {
        let bounded = self.bound.map_or(boxes, |bytes| {
            (bytes / (box_waves * size_of::<T>()).max(1)).max(MIN_BUFFER_BOXES)
        });
        bounded.min(boxes)
    }

    /// Sizes the buffers for the levels with waves of `m2l` and the plan's boxes per level,
    /// or empties them where there are none.
    pub(super) fn size(&mut self, m2l: Option<&PlaneWaveM2l<T>>) {
        let (mut values, mut boxes, mut sources) = (0, 0, 0);
        if let Some(m2l) = m2l {
            for (l, &n) in self.level_boxes.iter().enumerate() {
                let Some(level) = m2l.level(l) else { continue };
                let box_waves = 2 * level.tables().wave_len();
                let capacity = self.capacity(n, box_waves);
                values = values.max(capacity * box_waves);
                boxes = boxes.max(n);
                sources = sources.max(capacity);
            }
        }
        self.waves = vec![T::zero(); values];
        self.position = vec![NO_POSITION; boxes];
        self.sources = vec![0; sources];
    }

    /// Its bytes: the waves, the positions and the sources.
    pub(super) fn bytes(&self) -> usize {
        self.waves.len() * size_of::<T>() + 4 * (self.position.len() + self.sources.len())
    }

    /// The bytes of its waves.
    pub(super) fn wave_bytes(&self) -> usize {
        self.waves.len() * size_of::<T>()
    }

    /// Grows the buffers for a level call of `boxes` boxes holding `capacity` of them, where
    /// the plan did not size them.
    fn reserve(&mut self, boxes: usize, capacity: usize, box_waves: usize) {
        if self.waves.len() < capacity * box_waves {
            self.waves.resize(capacity * box_waves, T::zero());
        }
        if self.position.len() < boxes {
            self.position.resize(boxes, NO_POSITION);
        }
        if self.sources.len() < capacity {
            self.sources.resize(capacity, 0);
        }
    }

    /// The chunk of whole parents from box `start` of the call's level: the parents in index
    /// order while their computed children's distinct sources fit `capacity` (one parent
    /// always does), each source's position its order of first reading. Returns the chunk's
    /// end and its number of sources, listed in `sources`.
    fn chunk(
        &mut self,
        call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
        start: usize,
        end: usize,
        capacity: usize,
    ) -> (usize, usize) {
        let mut count = 0;
        let mut t = start;
        while t < end {
            let parent = morton::parent(call.index.key(call.level, t));
            let mut next = t + 1;
            while next < end && morton::parent(call.index.key(call.level, next)) == parent {
                next += 1;
            }
            let before = count;
            let mut overflow = false;
            'parent: for u in t..next {
                for &source in call.view.row(u).0 {
                    let position = &mut self.position[source as usize];
                    if *position != NO_POSITION {
                        continue;
                    }
                    if count == capacity {
                        overflow = true;
                        break 'parent;
                    }
                    *position = count as u32;
                    self.sources[count] = source;
                    count += 1;
                }
            }
            // The parent's sources do not all fit: it starts the next chunk.
            if overflow {
                assert!(t > start, "one parent's sources fit the buffer");
                self.forget(before, count);
                return (t, before);
            }
            t = next;
        }
        (end, count)
    }

    /// Forgets the positions of the sources listed in `from..to`.
    fn forget(&mut self, from: usize, to: usize) {
        for &source in &self.sources[from..to] {
            self.position[source as usize] = NO_POSITION;
        }
    }
}

impl<T: SimdScalar> HelmholtzOperator<T> {
    /// The level call by merge and shift (module documentation, "M2L"): chunk by chunk of
    /// whole parents, per vector and frame the up and down waves of the chunk's sources on
    /// the pool, then merge and shift by parents (the pool splitting the chunk at parent
    /// boundaries) or by child ([`PlaneWaveSplit`]).
    pub(super) fn plane_wave_m2l(&mut self, batch: M2l<'_, T>) {
        let M2l {
            level,
            index,
            pairs,
            multipoles,
            locals,
        } = batch;
        if pairs.is_empty() {
            return;
        }
        let kernel = &self.kernel;
        let geometry = kernel
            .plane_waves
            .as_ref()
            .expect("a level with waves")
            .geometry();
        let merge = kernel.plane_wave_level(level).expect("a level with waves");
        let call = BoxCall {
            level,
            index,
            view: pairs,
            input: multipoles,
        };
        let tables = merge.tables();
        let (planes, m) = (
            kernel.densities.multipole_planes(),
            kernel.densities.vectors(),
        );
        let n = Layout::new(tables.p()).len();
        let box_waves = 2 * tables.wave_len();
        let boxes = multipoles.as_slice().len() / (m * planes * n);
        let threads = match (&self.execution.pool, self.execution.serial) {
            (Some(_), false) => self.execution.scratch.len(),
            _ => 1,
        };
        let buffer = &mut self.plane_buffer;
        let capacity = buffer.capacity(boxes, box_waves);
        buffer.reserve(boxes, capacity, box_waves);
        buffer.counts.level_calls += 1;
        let mut locals = locals;
        let size = locals.size();
        let locals = locals.as_mut_slice();
        let targets = locals.len() / size;
        let mut start = 0;
        while start < targets {
            let (end, count) = buffer.chunk(&call, start, targets, capacity);
            if count == 0 {
                start = end;
                continue;
            }
            let starts = |t: usize| {
                t == 0
                    || morton::parent(index.key(level, start + t))
                        != morton::parent(index.key(level, start + t - 1))
            };
            let computed = |t: usize| !call.view.row(start + t).0.is_empty();
            let by_child = match buffer.split {
                PlaneWaveSplit::Parents => false,
                PlaneWaveSplit::Children => true,
                PlaneWaveSplit::Auto => {
                    threads > 1 && 2 * pieces(0..end - start, &starts, &computed) <= threads
                }
            };
            buffer.counts.chunks += 1;
            buffer.counts.waves += 3 * m * count;
            buffer.counts.by_child += usize::from(by_child);
            let chunk = &mut locals[start * size..end * size];
            for v in 0..m {
                for frame in 0..3 {
                    let sources = &buffer.sources[..count];
                    let waves = &mut buffer.waves[..count * box_waves];
                    let clock = Instant::now();
                    self.execution.chunks(waves, box_waves, |b, out, scratch| {
                        let source = sources[b] as usize;
                        let x = &multipoles.chunk(source)[v * planes * n..(v + 1) * planes * n];
                        let plane = scratch
                            .plane
                            .as_mut()
                            .expect("the exponential M2L's scratch");
                        outgoing(tables, frame, x, planes, out, plane);
                    });
                    self.plane_wave_times[level].waves += clock.elapsed();
                    let clock = Instant::now();
                    let waves = (&buffer.waves[..count * box_waves], &buffer.position[..]);
                    let tables = (merge, geometry);
                    let wave = (frame, v, waves);
                    if by_child {
                        self.execution.chunks(chunk, size, |i, local, scratch| {
                            let plane = scratch
                                .plane
                                .as_mut()
                                .expect("the exponential M2L's scratch");
                            let busy = Instant::now();
                            child(&call, tables, wave, (start..end, start + i), local, plane);
                            plane.parts.busy += busy.elapsed();
                        });
                    } else {
                        self.execution
                            .parents(chunk, size, starts, |range, chunks, scratch| {
                                let plane = scratch
                                    .plane
                                    .as_mut()
                                    .expect("the exponential M2L's scratch");
                                let busy = Instant::now();
                                parents(
                                    &call,
                                    tables,
                                    wave,
                                    start + range.start..start + range.end,
                                    chunks,
                                    plane,
                                );
                                plane.parts.busy += busy.elapsed();
                            });
                    }
                    self.plane_wave_times[level].parents += clock.elapsed();
                }
            }
            buffer.forget(0, count);
            start = end;
        }
        // Every thread's parts, into the level's times.
        let times = &mut self.plane_wave_times[level];
        for set in &mut self.execution.scratch {
            let set = set.get_mut().unwrap_or_else(PoisonError::into_inner);
            if let Some(plane) = set.plane.as_mut() {
                let parts = core::mem::take(&mut plane.parts);
                times.busy += parts.busy;
                times.shifts += parts.shifts;
                times.children += parts.children;
                times.conversions += parts.conversions;
            }
        }
    }
}

/// The pieces with work into which the shared drivers' split by parents
/// (`Execution::parents`, `driver::split_parents`) cuts the boxes `boxes`, `starts(t)`
/// telling whether box t begins a parent and `computed(t)` whether it has a V row: halved at
/// the first parent boundary from the middle while a piece holds more than [`PARENT_GRAIN`]
/// boxes, the driver's rule restated (it decides only the split, never a bit), counting the
/// pieces that hold a computed box (on P ranks a level also holds boxes its rank does not
/// compute, whose pieces finish at once).
fn pieces(
    boxes: Range<usize>,
    starts: &impl Fn(usize) -> bool,
    computed: &impl Fn(usize) -> bool,
) -> usize {
    let mut mid = boxes.start + boxes.len() / 2;
    while mid < boxes.end && !starts(mid) {
        mid += 1;
    }
    if boxes.len() <= PARENT_GRAIN || mid == boxes.end {
        return usize::from(boxes.into_iter().any(computed));
    }
    pieces(boxes.start..mid, starts, computed) + pieces(mid..boxes.end, starts, computed)
}

/// The up and down waves of `frame` of one vector's multipole `x` (`planes` planes) into
/// `out`, 2W reals each.
fn outgoing<T: SimdScalar>(
    tables: &PlaneWaveLevel<T>,
    frame: usize,
    x: &[T],
    planes: usize,
    out: &mut [T],
    s: &mut PlaneScratch<T>,
) {
    let n = Layout::new(tables.p()).len();
    let rotated = &mut s.rotated[..planes * n];
    tables.rotate_multipole(frame, x, planes, rotated);
    let modes = &mut s.modes[..tables.mode_len()];
    tables.outgoing_modes(rotated, planes, modes);
    let (up, down) = out.split_at_mut(tables.wave_len());
    tables.waves_from_modes(modes, up, down);
}

/// Merge and shift of `frame` for vector `v` over the parents of the boxes `boxes`, whose
/// locals are `locals` (consecutive boxes of the level, whole parents): parent by parent
/// in index order.
fn parents<T: SimdScalar>(
    call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
    tables: (&MergeLevel<T>, &MergeGeometry),
    wave: (usize, usize, (&[T], &[u32])),
    boxes: Range<usize>,
    locals: &mut [T],
    s: &mut PlaneScratch<T>,
) {
    let size = locals.len() / boxes.len();
    let mut t = boxes.start;
    while t < boxes.end {
        let end = parent_end(call, t, boxes.end);
        let run = &mut locals[(t - boxes.start) * size..(end - boxes.start) * size];
        let children = sources_by_slot(call, tables.1, t..end, s);
        parent_call(tables, wave, children, (t, size), run, s);
        t = end;
    }
}

/// The end of the run of boxes from `t` that share its parent, at most `end`.
fn parent_end<T>(call: &BoxCall<'_, VList, LevelSlice<'_, T>>, t: usize, end: usize) -> usize {
    let parent = morton::parent(call.index.key(call.level, t));
    let mut next = t + 1;
    while next < end && morton::parent(call.index.key(call.level, next)) == parent {
        next += 1;
    }
    next
}

/// Merge and shift of `frame` for vector `v` into box `t` alone, whose local is `local`,
/// the boxes `chunk` being whole parents (the split by child): its parent's run of boxes
/// and their sources by slot, as the parent's call has them, then the call for this child
/// alone, so the child's local gets the parent's call's arithmetic in its order.
fn child<T: SimdScalar>(
    call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
    tables: (&MergeLevel<T>, &MergeGeometry),
    wave: (usize, usize, (&[T], &[u32])),
    (chunk, t): (Range<usize>, usize),
    local: &mut [T],
    s: &mut PlaneScratch<T>,
) {
    if call.view.row(t).0.is_empty() {
        return;
    }
    let parent = morton::parent(call.index.key(call.level, t));
    let mut first = t;
    while first > chunk.start && morton::parent(call.index.key(call.level, first - 1)) == parent {
        first -= 1;
    }
    let siblings = first..parent_end(call, t, chunk.end);
    let (boxes, computed) = sources_by_slot(call, tables.1, siblings, s);
    let o = morton::child_index(call.index.key(call.level, t));
    debug_assert!(computed & (1 << o) != 0, "a box with a V row is computed");
    parent_call(tables, wave, (boxes, 1 << o), (t, local.len()), local, s);
}

/// The children `children` of one parent (consecutive boxes) by child index, and the
/// computed ones (with a V row) as a mask; their sources into `s.slots` by slot.
fn sources_by_slot<T>(
    call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
    geometry: &MergeGeometry,
    children: Range<usize>,
    s: &mut PlaneScratch<T>,
) -> ([usize; CHILDREN], u8) {
    let mut boxes = [usize::MAX; CHILDREN];
    let mut computed = 0u8;
    for t in children {
        let o = morton::child_index(call.index.key(call.level, t));
        boxes[o] = t;
        if !call.view.row(t).0.is_empty() {
            computed |= 1 << o;
        }
    }
    if computed == 0 {
        return (boxes, computed);
    }
    s.slots.fill(NO_BOX);
    for (o, &t) in boxes.iter().enumerate() {
        if computed & (1 << o) == 0 {
            continue;
        }
        let (sources, offsets) = call.view.row(t);
        for (&source, &d) in sources.iter().zip(offsets) {
            let slot = SLOT_OF[8 * d as usize + o];
            assert_ne!(slot, NO_SLOT, "a V pair lies in the parent's slots");
            debug_assert!(geometry.direction(slot as usize, o).is_some(), "a V pair");
            s.slots[slot as usize] = source;
        }
    }
    (boxes, computed)
}

/// Merge and shift of `frame` for vector `v` into the children `serve` of one parent
/// (`boxes` by child index; the computed children, or one of them under the split by
/// child), the sources by slot in `s.slots`, whose locals are `locals` from box `first`,
/// `size` values per box: per direction of the frame, every accumulator whose mask holds a
/// child served sums its slots present in ascending order, each source's waves times its
/// source shift, from zero; each child served adds its accumulators in ascending mask order
/// and multiplies by its child shift; then each child served with waves converts its up
/// and its down waves into a local in the frame and adds it rotated back. A child's
/// arithmetic does not depend on the others served.
fn parent_call<T: SimdScalar>(
    (merge, geometry): (&MergeLevel<T>, &MergeGeometry),
    (frame, v, (waves, position)): (usize, usize, (&[T], &[u32])),
    (boxes, serve): ([usize; CHILDREN], u8),
    (first, size): (usize, usize),
    locals: &mut [T],
    s: &mut PlaneScratch<T>,
) {
    if serve == 0 {
        return;
    }
    let tables = merge.tables();
    let w = tables.wave_len();
    let n = Layout::new(tables.p()).len();
    // Per child, whether it received up waves (0) and down waves (1).
    let mut has = [[false; 2]; CHILDREN];
    for direction in [0, 1].map(|half| Direction::ALL[2 * frame + half]) {
        let half = usize::from(!direction.is_up());
        let accumulators = geometry.accumulators(direction);
        // The accumulators of the children served, each over its slots in order.
        let clock = Instant::now();
        for (a, accumulator) in accumulators.iter().enumerate() {
            s.present[a] = false;
            if accumulator.mask & serve == 0 {
                continue;
            }
            let acc = &mut s.accs[a * w..(a + 1) * w];
            for &slot in &accumulator.slots {
                let source = s.slots[slot as usize];
                if source == NO_BOX {
                    continue;
                }
                if !s.present[a] {
                    acc.fill(T::zero());
                    s.present[a] = true;
                }
                let at = (2 * position[source as usize] as usize + half) * w;
                let factors = merge
                    .source_shift(direction, slot as usize)
                    .expect("a slot of an accumulator has a shift");
                shift_add(acc, &waves[at..at + w], factors);
            }
        }
        let shifted = Instant::now();
        s.parts.shifts += shifted - clock;
        // Each child served: its accumulators in order, then its shift.
        for (o, has) in has.iter_mut().enumerate() {
            if serve & (1 << o) == 0 {
                continue;
            }
            let incoming = &mut s.incoming[(2 * o + half) * w..(2 * o + half + 1) * w];
            for (a, accumulator) in accumulators.iter().enumerate() {
                if s.present[a] && accumulator.mask & (1 << o) != 0 {
                    let acc = &s.accs[a * w..(a + 1) * w];
                    if has[half] {
                        add(incoming, acc);
                    } else {
                        incoming.copy_from_slice(acc);
                        has[half] = true;
                    }
                }
            }
            if has[half] {
                shift(incoming, merge.child_shift(direction, o));
            }
        }
        s.parts.children += shifted.elapsed();
    }
    // Each child's local: the sums of its up and down waves, the local in the frame, and
    // back into the box's frame.
    let clock = Instant::now();
    for (o, has) in has.iter().enumerate() {
        if !has[0] && !has[1] {
            continue;
        }
        let frame_local = &mut s.frame[..2 * n];
        frame_local.fill(T::zero());
        let sums = &mut s.sums[..tables.sum_len()];
        for (half, &has) in has.iter().enumerate() {
            if has {
                let incoming = &s.incoming[(2 * o + half) * w..(2 * o + half + 1) * w];
                tables.sums_from_waves(incoming, sums);
                tables.add_local_from_sums(sums, half == 0, frame_local);
            }
        }
        let at = (boxes[o] - first) * size + v * 2 * n;
        tables.add_local_from_frame(frame, frame_local, &mut locals[at..at + 2 * n]);
    }
    s.parts.conversions += clock.elapsed();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pieces` against the driver's halving: a level of 64 boxes is one piece, of 512 eight,
    /// and pieces without a computed box (on P ranks, boxes the rank holds but does not
    /// compute) are not counted.
    #[test]
    fn pieces_count_the_split_by_parents_with_work() {
        let starts = |t: usize| t.is_multiple_of(8);
        let all = |_: usize| true;
        assert_eq!(pieces(0..64, &starts, &all), 1);
        assert_eq!(pieces(0..512, &starts, &all), 8);
        assert_eq!(pieces(0..65, &starts, &all), 2);
        assert_eq!(pieces(0..512, &starts, &|t: usize| t < 128), 2);
        assert_eq!(pieces(0..512, &starts, &|t: usize| t == 300), 1);
        assert_eq!(pieces(0..512, &starts, &|_: usize| false), 0);
        // One parent of 8 boxes cannot be cut, however many boxes follow without a boundary.
        assert_eq!(pieces(0..200, &|t: usize| t == 0, &all), 1);
    }
}
