//! The exponential M2L level call (Helmholtz CH6.1d; CONVENTIONS §3.14, "Merge and shift",
//! and §3.15, "Exponential M2L"), a private part of the [`operator`](super) module: its
//! scratch, its order and its pair body ([module documentation](super#m2l)).

use core::ops::Range;

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
use crate::driver::BoxCall;

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
    /// the pool, then merge and shift parent by parent, the pool splitting the chunk at
    /// parent boundaries.
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
            buffer.counts.chunks += 1;
            buffer.counts.waves += 3 * m * count;
            let chunk = &mut locals[start * size..end * size];
            for v in 0..m {
                for frame in 0..3 {
                    let sources = &buffer.sources[..count];
                    let waves = &mut buffer.waves[..count * box_waves];
                    self.execution.chunks(waves, box_waves, |b, out, scratch| {
                        let source = sources[b] as usize;
                        let x = &multipoles.chunk(source)[v * planes * n..(v + 1) * planes * n];
                        let plane = scratch
                            .plane
                            .as_mut()
                            .expect("the exponential M2L's scratch");
                        outgoing(tables, frame, x, planes, out, plane);
                    });
                    let waves = (&buffer.waves[..count * box_waves], &buffer.position[..]);
                    let starts = |t: usize| {
                        t == 0
                            || morton::parent(index.key(level, start + t))
                                != morton::parent(index.key(level, start + t - 1))
                    };
                    self.execution
                        .parents(chunk, size, starts, |range, chunks, scratch| {
                            let plane = scratch
                                .plane
                                .as_mut()
                                .expect("the exponential M2L's scratch");
                            parents(
                                &call,
                                (merge, geometry),
                                (frame, v, waves),
                                start + range.start..start + range.end,
                                chunks,
                                plane,
                            );
                        });
                }
            }
            buffer.forget(0, count);
            start = end;
        }
    }
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
        let parent = morton::parent(call.index.key(call.level, t));
        let mut end = t + 1;
        while end < boxes.end && morton::parent(call.index.key(call.level, end)) == parent {
            end += 1;
        }
        let run = &mut locals[(t - boxes.start) * size..(end - boxes.start) * size];
        parent_call(call, tables, wave, t..end, run, s);
        t = end;
    }
}

/// Merge and shift of `frame` for vector `v` into the children `children` of one parent
/// (consecutive boxes) whose locals are `locals`: per direction of the frame, every
/// accumulator whose mask holds a computed child sums its slots present in ascending order,
/// each source's waves times its source shift, from zero; each computed child adds its
/// accumulators in ascending mask order and multiplies by its child shift; then each
/// computed child with waves converts its up and its down waves into a local in the frame
/// and adds it rotated back.
fn parent_call<T: SimdScalar>(
    call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
    (merge, geometry): (&MergeLevel<T>, &MergeGeometry),
    (frame, v, (waves, position)): (usize, usize, (&[T], &[u32])),
    children: Range<usize>,
    locals: &mut [T],
    s: &mut PlaneScratch<T>,
) {
    let tables = merge.tables();
    let w = tables.wave_len();
    let n = Layout::new(tables.p()).len();
    let size = locals.len() / children.len();
    // The children this rank computes (a V row), by child index.
    let mut boxes = [usize::MAX; CHILDREN];
    let mut computed = 0u8;
    for t in children.clone() {
        let o = morton::child_index(call.index.key(call.level, t));
        boxes[o] = t;
        if !call.view.row(t).0.is_empty() {
            computed |= 1 << o;
        }
    }
    if computed == 0 {
        return;
    }
    // The sources by slot, from the computed children's rows.
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
    // Per child, whether it received up waves (0) and down waves (1).
    let mut has = [[false; 2]; CHILDREN];
    for direction in [0, 1].map(|half| Direction::ALL[2 * frame + half]) {
        let half = usize::from(!direction.is_up());
        let accumulators = geometry.accumulators(direction);
        // The accumulators of the computed children, each over its slots in order.
        for (a, accumulator) in accumulators.iter().enumerate() {
            s.present[a] = false;
            if accumulator.mask & computed == 0 {
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
        // Each computed child: its accumulators in order, then its shift.
        for (o, has) in has.iter_mut().enumerate() {
            if computed & (1 << o) == 0 {
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
    }
    // Each child's local: the sums of its up and down waves, the local in the frame, and
    // back into the box's frame.
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
        let at = (boxes[o] - children.start) * size + v * 2 * n;
        tables.add_local_from_frame(frame, frame_local, &mut locals[at..at + 2 * n]);
    }
}
