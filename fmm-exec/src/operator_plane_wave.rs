//! The plane-wave M2L level call (Phase 6 T13, C6.5; CONVENTIONS §3.14), a private part
//! of the [`operator`](super) module: its kernels, scratch and order ([module
//! documentation](super#the-plane-wave-m2l)).

use core::ops::Range;

use nd_fmm_plan::lists::VList;
use nd_fmm_plan::store::LevelSlice;
use nd_fmm_simd::{Isa, IsaUnavailable, SimdScalar, WaveBlock, WaveKernel, WaveTables};
use nd_fmm_tables::PlaneWaveTables;
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::plane_wave::geometry::slot_of;
use nd_fmm_tables::plane_wave::{CHILDREN, Direction, SLOTS};
use nd_octree::morton;

use super::{BoxCall, Kernels, Scratch};

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

/// The boxes per piece below which the pool runs a piece of the parents serially.
pub(crate) const PARENT_GRAIN: usize = 64;

/// The plane-wave kernels of an operator: the vector kernels on the ISA of the P2P
/// choice (the scalar path for the reference) and the tables of their Fourier passes in
/// their layout.
#[derive(Clone, Debug)]
pub(crate) struct PlaneWaveKernel<T: SimdScalar> {
    pub(crate) kernel: WaveKernel<T>,
    tables: WaveTables<T>,
}

impl<T: SimdScalar> PlaneWaveKernel<T> {
    /// The kernels of `tables` on `isa`.
    pub(crate) fn new(tables: &PlaneWaveTables<T>, isa: Isa) -> Result<Self, IsaUnavailable> {
        let blocks: Vec<WaveBlock> = tables
            .blocks()
            .iter()
            .map(|b| WaveBlock {
                start: b.start,
                len: b.len,
                modes: b.modes,
                mode_start: b.mode_start,
                trig: b.trig,
            })
            .collect();
        let kernel = WaveKernel::new(isa)?;
        Ok(Self {
            tables: kernel.tables(&blocks, tables.cos(), tables.sin()),
            kernel,
        })
    }
}

/// The scratch of one thread's plane-wave work: a box's frame coefficients and modes
/// (outgoing), a parent's slots, offset lists, accumulators and children's incoming waves,
/// a child's sums and frame local (incoming), and the work buffer of the λ-power steps.
/// Sized at construction: (2 a + 32) H + 2 (p + 1)² + 8 F + w values and 3 · 216 + a
/// indices, with a the most accumulators of a direction, F the modes of all the λ nodes and
/// w = 4 (p + 1) s (`PlaneWaveTables::work_len`).
#[derive(Clone, Debug)]
pub(crate) struct PlaneScratch<T> {
    rotated: Vec<T>,
    modes: Vec<T>,
    slots: [u32; SLOTS],
    wave_offsets: Vec<usize>,
    shift_offsets: Vec<usize>,
    acc_offsets: Vec<usize>,
    present: Vec<bool>,
    accs: Vec<T>,
    incoming: Vec<T>,
    sums: Vec<T>,
    frame: Vec<T>,
    work: Vec<T>,
}

impl<T: SimdScalar> PlaneScratch<T> {
    /// Its bytes: the values, the offset lists' capacity and the slots.
    pub(crate) fn bytes(&self) -> usize {
        let values = self.rotated.len()
            + self.modes.len()
            + self.accs.len()
            + self.incoming.len()
            + self.sums.len()
            + self.frame.len()
            + self.work.len();
        let indices = self.wave_offsets.capacity()
            + self.shift_offsets.capacity()
            + self.acc_offsets.capacity();
        values * size_of::<T>() + indices * size_of::<usize>() + self.present.len() + 4 * SLOTS
    }

    /// The scratch of `tables`.
    pub(crate) fn new(tables: &PlaneWaveTables<T>) -> Self {
        let (n, w, a) = (
            (tables.p() + 1).pow(2),
            tables.wave_len(),
            tables.geometry().max_accumulators(),
        );
        Self {
            rotated: vec![T::zero(); n],
            modes: vec![T::zero(); 4 * tables.mode_count()],
            slots: [NO_BOX; SLOTS],
            wave_offsets: Vec::with_capacity(SLOTS),
            shift_offsets: Vec::with_capacity(SLOTS),
            acc_offsets: Vec::with_capacity(a),
            present: vec![false; a],
            accs: vec![T::zero(); a * w],
            incoming: vec![T::zero(); 2 * CHILDREN * w],
            sums: vec![T::zero(); 4 * tables.mode_count()],
            frame: vec![T::zero(); n],
            work: vec![T::zero(); tables.work_len()],
        }
    }
}

impl<T: SimdScalar> Kernels<T> {
    /// The plane-wave tables and kernels, under `PlaneWave`.
    #[inline]
    pub(crate) fn plane_waves(&self) -> Option<(&PlaneWaveTables<T>, &PlaneWaveKernel<T>)> {
        self.tables.plane_waves().zip(self.waves.as_ref())
    }

    /// The waves of `frame` of vector v of box `b`'s multipole (`multipole`, (p + 1)²
    /// values) into `out`, 4H values: the up waves, then the down waves.
    #[inline]
    pub(crate) fn plane_outgoing(
        &self,
        frame: usize,
        multipole: &[T],
        out: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (tables, waves) = self.plane_waves().expect("PlaneWave tables");
        let plane = scratch.plane.as_mut().expect("PlaneWave scratch");
        let rotated: &[T] = if frame == 0 {
            multipole
        } else {
            tables.rotate_multipole(frame, multipole, &mut plane.rotated);
            &plane.rotated
        };
        tables.outgoing_modes(rotated, &mut plane.modes, &mut plane.work);
        let (up, down) = out.split_at_mut(tables.wave_len());
        waves.kernel.waves(&waves.tables, &plane.modes, up, down);
    }

    /// Merge and shift of `frame` for vector `v` over the parents of the boxes `boxes`, whose
    /// locals are `locals` (consecutive boxes of the level, whole parents): parent by
    /// parent in index order ([module documentation](super#the-plane-wave-m2l)).
    pub(crate) fn plane_parents(
        &self,
        call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
        (frame, v, waves): (usize, usize, &[T]),
        boxes: Range<usize>,
        locals: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let size = self.box_size();
        let mut t = boxes.start;
        while t < boxes.end {
            let parent = morton::parent(call.index.key(call.level, t));
            let mut end = t + 1;
            while end < boxes.end && morton::parent(call.index.key(call.level, end)) == parent {
                end += 1;
            }
            let run = &mut locals[(t - boxes.start) * size..(end - boxes.start) * size];
            self.plane_parent(call, (frame, v, waves), t..end, run, scratch);
            t = end;
        }
    }

    /// Merge and shift of `frame` for vector `v` into the children `children` of one
    /// parent (consecutive boxes) whose locals are `locals`.
    fn plane_parent(
        &self,
        call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
        (frame, v, waves): (usize, usize, &[T]),
        children: Range<usize>,
        locals: &mut [T],
        scratch: &mut Scratch<T>,
    ) {
        let (tables, kernels) = self.plane_waves().expect("PlaneWave tables");
        let kernel = &kernels.kernel;
        let plane = scratch.plane.as_mut().expect("PlaneWave scratch");
        let (w, n, size) = (tables.wave_len(), (self.p + 1).pow(2), self.box_size());
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
        plane.slots.fill(NO_BOX);
        let geometry = tables.geometry();
        for (o, &t) in boxes.iter().enumerate() {
            if computed & (1 << o) == 0 {
                continue;
            }
            let (sources, offsets) = call.view.row(t);
            let slot_of = &*SLOT_OF;
            for (&s, &d) in sources.iter().zip(offsets) {
                let slot = slot_of[8 * d as usize + o];
                assert_ne!(slot, NO_SLOT, "a V pair lies in the parent's slots");
                debug_assert!(geometry.direction(slot as usize, o).is_some(), "a V pair");
                plane.slots[slot as usize] = s;
            }
        }
        let mut has = [[false; 2]; CHILDREN];
        for half in 0..2 {
            let direction = Direction::ALL[2 * frame + half];
            let accumulators = geometry.accumulators(direction);
            // The accumulators of the computed children, each over its slots in order.
            for (a, accumulator) in accumulators.iter().enumerate() {
                plane.present[a] = false;
                if accumulator.mask & computed == 0 {
                    continue;
                }
                plane.wave_offsets.clear();
                plane.shift_offsets.clear();
                for &slot in &accumulator.slots {
                    let s = plane.slots[slot as usize];
                    if s == NO_BOX {
                        continue;
                    }
                    plane.wave_offsets.push((2 * s as usize + half) * w);
                    plane.shift_offsets.push(
                        tables
                            .source_shift_offset(direction, slot as usize)
                            .expect("a slot of an accumulator has a shift"),
                    );
                }
                if plane.wave_offsets.is_empty() {
                    continue;
                }
                let acc = &mut plane.accs[a * w..(a + 1) * w];
                acc.fill(T::zero());
                kernel.merge(
                    acc,
                    waves,
                    &plane.wave_offsets,
                    tables.source_shifts(),
                    &plane.shift_offsets,
                );
                plane.present[a] = true;
            }
            // Each computed child: its accumulators in order, then its shift.
            for (o, has) in has.iter_mut().enumerate() {
                if computed & (1 << o) == 0 {
                    continue;
                }
                plane.acc_offsets.clear();
                for (a, accumulator) in accumulators.iter().enumerate() {
                    if plane.present[a] && accumulator.mask & (1 << o) != 0 {
                        plane.acc_offsets.push(a * w);
                    }
                }
                if plane.acc_offsets.is_empty() {
                    continue;
                }
                has[half] = true;
                let incoming = &mut plane.incoming[(2 * o + half) * w..(2 * o + half + 1) * w];
                kernel.gather(
                    incoming,
                    &plane.accs,
                    &plane.acc_offsets,
                    tables.child_shift(direction, o),
                );
            }
        }
        // Each child's local: the sums of its up and down waves, the local in the frame,
        // and back into the box's frame.
        for (o, has) in has.iter().enumerate() {
            if !has[0] && !has[1] {
                continue;
            }
            let incoming = &mut plane.incoming[2 * o * w..(2 * o + 2) * w];
            for (half, &has) in has.iter().enumerate() {
                if !has {
                    incoming[half * w..(half + 1) * w].fill(T::zero());
                }
            }
            let (up, down) = incoming.split_at(w);
            kernel.sums(&kernels.tables, up, down, &mut plane.sums);
            tables.local_from_sums(&plane.sums, &mut plane.frame);
            let at = (boxes[o] - children.start) * size + v * n;
            tables.add_local_from_frame(frame, &plane.frame, &mut locals[at..at + n]);
        }
    }
}
