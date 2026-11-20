//! A level's signatures and scratch, and the level driver of the diagonal form (proposal §5.4,
//! "Scratch and chunking"; Helmholtz CH4.1b).
//!
//! # The signature buffers
//!
//! [`Signatures`] holds the signatures of a set of boxes of one level, box after box. Each
//! signature is CONVENTIONS §3.15's (2p + 1) N_φ complex samples in its order i N_φ + j,
//! stored interleaved (real, imaginary) as `Complex<T>`; box b's first sample lies at
//! `b · stride`, the stride the samples rounded up to whole lines of [`SIGNATURE_ALIGN`] bytes,
//! so that every box starts on a line ([`Signatures::is_aligned`]) and a box's samples never
//! share a line with another box's. The padding samples are zero and nothing reads them. The
//! store is allocated once at its capacity and holds any number of boxes up to it
//! ([`Signatures::set_boxes`]), so a level call reuses it chunk after chunk. This layout is
//! the crate's, not a convention (CONVENTIONS §3.15, "Signatures", "Not fixed here").
//!
//! # The level's scratch
//!
//! A level call of the diagonal form keeps the source signatures of the whole call (each
//! source box converted once, S → F) and forms the targets' accumulators a chunk of targets
//! at a time: zeroed, every diagonal pair of each target applied in its row's order, F → R
//! into the target's local. [`scratch_shape`] sizes it from a byte budget per rank (the
//! caller's node budget over the ranks of the node, proposal §2, constraint 7): the
//! conversion tables, one [`ConversionScratch`] and gather buffer per thread, the sources'
//! signatures and the largest chunk of accumulators that fits (not the transfer functions,
//! which are CH4.2's tables); `None` when not even
//! [`BLOCK_BOXES`] targets fit, where the level runs RCR (proposal §7; CH4.3).
//!
//! # Determinism
//!
//! A target's accumulator is zeroed and receives its own pairs in its row's order; S → F and
//! F → R give each box the bits of its own input whatever its batch (the module docs of
//! [`super`], "Determinism"); the gathers and scatters are copies. So every local is the same
//! bits at every chunk size and every capacity of the buffers ([`m2l_level`]; tested), and a
//! threaded caller may split the source boxes and a chunk's targets over threads without
//! changing a bit (proposal §2, constraint 2).

use nd_fmm_math::{Complex, RealScalar};
use nd_fmm_ref::helmholtz::diagonal::azimuth_count;

use super::{BLOCK_BOXES, ConversionScratch, Conversions};

/// The alignment in bytes of every box's first sample in [`Signatures`], a cache line (and
/// the widest vector register, AVX-512).
pub const SIGNATURE_ALIGN: usize = 64;

/// The default byte budget of the diagonal form's scratch per node, 8 GiB (as nd-fmm-exec's
/// dense and plane-wave node budgets): a caller divides it over the node's ranks and passes
/// the share to [`scratch_shape`] (proposal §2, constraint 7; §7).
pub const DEFAULT_SIGNATURE_NODE_BUDGET: usize = 8 << 30;

/// The signatures of a set of boxes of one level, box after box, each on its own
/// [`SIGNATURE_ALIGN`]-byte line (module documentation, "The signature buffers"). A clone is
/// aligned on its own allocation.
#[derive(Debug)]
pub struct Signatures<T> {
    values: Vec<Complex<T>>,
    /// The first sample of box 0 in `values`.
    start: usize,
    samples: usize,
    stride: usize,
    capacity: usize,
    boxes: usize,
}

impl<T: RealScalar> Clone for Signatures<T> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.samples, self.capacity);
        let len = self.capacity * self.stride;
        out.values[out.start..out.start + len]
            .copy_from_slice(&self.values[self.start..self.start + len]);
        out.boxes = self.boxes;
        out
    }
}

impl<T: RealScalar> Signatures<T> {
    /// The samples per line of [`SIGNATURE_ALIGN`] bytes.
    fn line() -> usize {
        let size = size_of::<Complex<T>>();
        assert!(
            SIGNATURE_ALIGN.is_multiple_of(size),
            "a complex sample of {size} bytes does not divide a line of {SIGNATURE_ALIGN}"
        );
        SIGNATURE_ALIGN / size
    }

    /// The stride between two boxes' first samples: `samples` rounded up to whole lines.
    pub fn stride_for(samples: usize) -> usize {
        samples.next_multiple_of(Self::line())
    }

    /// The bytes of a store of `samples` samples per box for `boxes` boxes: the strides and
    /// the slack the alignment may take (one line less one sample).
    pub fn bytes_for(samples: usize, boxes: usize) -> usize {
        Self::values_for(samples, boxes) * size_of::<Complex<T>>()
    }

    fn values_for(samples: usize, boxes: usize) -> usize {
        if boxes == 0 {
            0
        } else {
            Self::stride_for(samples) * boxes + Self::line() - 1
        }
    }

    /// A zeroed store of `capacity` boxes of `samples` samples each, holding `capacity` boxes.
    /// The only allocation; nothing later allocates.
    pub fn new(samples: usize, capacity: usize) -> Self {
        let size = size_of::<Complex<T>>();
        let values = vec![Complex::new(T::zero(), T::zero()); Self::values_for(samples, capacity)];
        // The address as an integer: `Vec`'s buffer never moves, as the store never grows. An
        // allocation whose address is not a whole number of samples off a line cannot be
        // aligned; it starts at 0 (`is_aligned` says so). The bits never depend on it.
        let address = values.as_ptr() as usize;
        let start = if address.is_multiple_of(size) {
            ((SIGNATURE_ALIGN - address % SIGNATURE_ALIGN) % SIGNATURE_ALIGN) / size
        } else {
            0
        };
        Self {
            values,
            start,
            samples,
            stride: Self::stride_for(samples),
            capacity,
            boxes: capacity,
        }
    }

    /// The samples of a signature.
    pub fn samples(&self) -> usize {
        self.samples
    }

    /// The stride between two boxes' first samples ([`stride_for`](Self::stride_for)).
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// The boxes the store can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The boxes the store holds now.
    pub fn boxes(&self) -> usize {
        self.boxes
    }

    /// Holds `boxes` boxes (their values as they are).
    ///
    /// # Panics
    ///
    /// If `boxes` exceeds the capacity.
    pub fn set_boxes(&mut self, boxes: usize) {
        assert!(
            boxes <= self.capacity,
            "{boxes} boxes exceed the capacity of {}",
            self.capacity
        );
        self.boxes = boxes;
    }

    /// Whether every box's first sample lies on a [`SIGNATURE_ALIGN`]-byte line (true from
    /// every allocator whose blocks are aligned to a sample's size, the system allocators of
    /// Linux and macOS among them).
    pub fn is_aligned(&self) -> bool {
        self.capacity == 0
            || (self.values[self.start..].as_ptr() as usize).is_multiple_of(SIGNATURE_ALIGN)
    }

    /// The bytes the store holds ([`bytes_for`](Self::bytes_for) of its capacity).
    pub fn bytes(&self) -> usize {
        self.values.len() * size_of::<Complex<T>>()
    }

    /// Box b's signature.
    ///
    /// # Panics
    ///
    /// If b is not below [`boxes`](Self::boxes).
    pub fn signature(&self, b: usize) -> &[Complex<T>] {
        assert!(b < self.boxes, "box {b} of {}", self.boxes);
        let at = self.start + b * self.stride;
        &self.values[at..at + self.samples]
    }

    /// Box b's signature, to write.
    ///
    /// # Panics
    ///
    /// If b is not below [`boxes`](Self::boxes).
    pub fn signature_mut(&mut self, b: usize) -> &mut [Complex<T>] {
        assert!(b < self.boxes, "box {b} of {}", self.boxes);
        let at = self.start + b * self.stride;
        &mut self.values[at..at + self.samples]
    }

    /// The boxes' strides, one mutable slice of [`stride`](Self::stride) samples per box (its
    /// signature first, then the padding), for a caller that splits the boxes over threads.
    pub fn boxes_mut(&mut self) -> core::slice::ChunksExactMut<'_, Complex<T>> {
        let end = self.start + self.boxes * self.stride;
        self.values[self.start..end].chunks_exact_mut(self.stride)
    }

    /// The strides of the boxes it holds as one slice, box b's signature at `b *`
    /// [`stride`](Self::stride), for a caller that splits blocks of boxes over threads
    /// (`chunks_mut(n * stride)`) and converts each with
    /// [`Conversions::s2f_strides`] (Helmholtz CH4.2b1).
    pub fn strides_mut(&mut self) -> &mut [Complex<T>] {
        let end = self.start + self.boxes * self.stride;
        &mut self.values[self.start..end]
    }

    /// Sets the samples of every box it holds to zero.
    pub fn zero(&mut self) {
        let samples = self.samples;
        let zero = Complex::new(T::zero(), T::zero());
        for b in self.boxes_mut() {
            b[..samples].fill(zero);
        }
    }

    /// The strides of boxes `first..first + boxes` (the last box's padding left out).
    fn strided(&self, first: usize, boxes: usize) -> &[Complex<T>] {
        let at = self.start + first * self.stride;
        &self.values[at..at + (boxes - 1) * self.stride + self.samples]
    }

    fn strided_mut(&mut self, first: usize, boxes: usize) -> &mut [Complex<T>] {
        let at = self.start + first * self.stride;
        &mut self.values[at..at + (boxes - 1) * self.stride + self.samples]
    }
}

impl<T: RealScalar> Conversions<T> {
    /// S → F of a batch of boxes into boxes `first..` of `signatures`: each box's signature
    /// is [`s2f`](Self::s2f)'s bit for bit.
    ///
    /// # Panics
    ///
    /// If `multipoles` is not a whole number of expansions, the boxes do not fit
    /// `signatures` from `first`, its samples are not [`samples`](Self::samples), or the
    /// scratch was made for another degree.
    pub fn s2f_into(
        &self,
        multipoles: &[T],
        signatures: &mut Signatures<T>,
        first: usize,
        scratch: &mut ConversionScratch<T>,
    ) {
        let boxes = self.check_store(multipoles.len(), signatures, first, scratch);
        if boxes > 0 {
            let stride = signatures.stride();
            self.s2f_strided(
                multipoles,
                signatures.strided_mut(first, boxes),
                stride,
                scratch,
            );
        }
    }

    /// S → F of a batch of boxes into a slice of strides (a block of
    /// [`Signatures::strides_mut`]): box b's signature at `b * stride`, [`s2f`](Self::s2f)'s
    /// bits, the padding untouched (Helmholtz CH4.2b1).
    ///
    /// # Panics
    ///
    /// If `multipoles` is not a whole number of expansions, `stride` is below
    /// [`samples`](Self::samples), the boxes do not fit `strides`, or the scratch was made for
    /// another degree.
    pub fn s2f_strides(
        &self,
        multipoles: &[T],
        strides: &mut [Complex<T>],
        stride: usize,
        scratch: &mut ConversionScratch<T>,
    ) {
        assert_eq!(
            scratch.p, self.p,
            "the scratch has degree {}, the conversions {}",
            scratch.p, self.p
        );
        let len = self.expansion_len();
        assert!(
            multipoles.len().is_multiple_of(len),
            "the multipoles hold {} values, not a multiple of 2(p + 1)² = {len}",
            multipoles.len()
        );
        assert!(
            stride >= self.samples(),
            "a stride of {stride} below {} samples",
            self.samples()
        );
        let boxes = multipoles.len() / len;
        if boxes > 0 {
            let need = (boxes - 1) * stride + self.samples();
            assert!(
                strides.len() >= need,
                "{boxes} boxes need {need} samples of strides, not {}",
                strides.len()
            );
            self.s2f_strided(multipoles, &mut strides[..need], stride, scratch);
        }
    }

    /// F → R of boxes `first..` of `signatures` into a batch of locals, adding: each local
    /// receives [`f2r`](Self::f2r)'s terms bit for bit.
    ///
    /// # Panics
    ///
    /// As [`s2f_into`](Self::s2f_into), with `locals` in place of `multipoles`.
    pub fn f2r_from(
        &self,
        signatures: &Signatures<T>,
        first: usize,
        locals: &mut [T],
        scratch: &mut ConversionScratch<T>,
    ) {
        let boxes = self.check_store(locals.len(), signatures, first, scratch);
        if boxes > 0 {
            self.f2r_strided(
                signatures.strided(first, boxes),
                signatures.stride(),
                locals,
                scratch,
            );
        }
    }

    fn check_store(
        &self,
        expansions: usize,
        signatures: &Signatures<T>,
        first: usize,
        scratch: &ConversionScratch<T>,
    ) -> usize {
        assert_eq!(
            signatures.samples(),
            self.samples(),
            "the signatures hold {} samples, the conversions {}",
            signatures.samples(),
            self.samples()
        );
        assert_eq!(
            scratch.p, self.p,
            "the scratch has degree {}, the conversions {}",
            scratch.p, self.p
        );
        let len = self.expansion_len();
        assert!(
            expansions.is_multiple_of(len),
            "the expansions hold {expansions} values, not a multiple of 2(p + 1)² = {len}"
        );
        let boxes = expansions / len;
        assert!(
            first + boxes <= signatures.boxes(),
            "boxes {first}..{} of a store of {}",
            first + boxes,
            signatures.boxes()
        );
        boxes
    }

    /// The bytes of the tables of degree p in `T` ([`bytes`](Self::bytes) without building
    /// them): the Legendre tables (p + 1)²(p + 2) values (S → F and F → R), the φ tables
    /// 4 (⌊N_φ/8⌋ + 1)(p + 1), the azimuth map and the offsets per order.
    pub fn bytes_for(p: usize) -> usize {
        let half = p + 1;
        let azimuths = azimuth_count(p);
        let bases = azimuths / 8 + 1;
        (half * half * (half + 1) + 4 * bases * half) * size_of::<T>()
            + azimuths * size_of::<super::Azimuth>()
            + half * size_of::<[usize; 4]>()
    }
}

impl<T: RealScalar> ConversionScratch<T> {
    /// The bytes of the scratch of degree p ([`bytes`](Self::bytes) without making it).
    pub fn bytes_for(p: usize) -> usize {
        let w = 4 * BLOCK_BOXES;
        let columns = 2 * super::RING_BLOCK * w;
        let bases = azimuth_count(p) / 8 + 1;
        ((p + 1) * (p + 2) / 2 * w
            + 2 * super::RING_BLOCK * w
            + (p + 1) * columns
            + 2 * bases * columns)
            * size_of::<T>()
    }
}

/// The scratch of one level call of the diagonal form on one rank, in bytes but for the
/// chunk ([`scratch_shape`]; module documentation, "The level's scratch").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchShape {
    /// The conversion tables ([`Conversions::bytes_for`]).
    pub tables: usize,
    /// Per thread a [`ConversionScratch`] and a gather buffer of [`BLOCK_BOXES`] expansions,
    /// times the threads.
    pub threads: usize,
    /// The source signatures, kept for the level call ([`Signatures::bytes_for`]).
    pub sources: usize,
    /// The targets per chunk of accumulators: all the targets where they fit, else the most
    /// that fit, at least [`BLOCK_BOXES`] (or all targets, if fewer).
    pub chunk: usize,
    /// The accumulators of one chunk ([`Signatures::bytes_for`]).
    pub accumulators: usize,
}

impl ScratchShape {
    /// The bytes in all.
    pub fn bytes(&self) -> usize {
        self.tables + self.threads + self.sources + self.accumulators
    }

    /// The chunks of `targets` targets (0 for none).
    pub fn chunks(&self, targets: usize) -> usize {
        if self.chunk == 0 {
            0
        } else {
            targets.div_ceil(self.chunk)
        }
    }
}

/// The scratch of a level call of the diagonal form at degree p in `T`, with `sources`
/// source boxes and `targets` target boxes on this rank and `threads` threads, within
/// `budget` bytes (the rank's share of the node budget, [`DEFAULT_SIGNATURE_NODE_BUDGET`]):
/// the tables, the threads' scratch and the sources' signatures whole, then the largest chunk
/// of accumulators that fits; `None` when the floor of [`BLOCK_BOXES`] targets (or all, if
/// fewer) does not fit, where the level runs RCR. A function of its arguments alone.
pub fn scratch_shape<T: RealScalar>(
    p: usize,
    sources: usize,
    targets: usize,
    threads: usize,
    budget: usize,
) -> Option<ScratchShape> {
    let samples = (2 * p + 1) * azimuth_count(p);
    let expansion = 2 * (p + 1) * (p + 1) * size_of::<T>();
    let mut shape = ScratchShape {
        tables: Conversions::<T>::bytes_for(p),
        threads: threads.max(1) * (ConversionScratch::<T>::bytes_for(p) + BLOCK_BOXES * expansion),
        sources: Signatures::<T>::bytes_for(samples, sources),
        chunk: 0,
        accumulators: 0,
    };
    let fixed = shape.tables + shape.threads + shape.sources;
    let floor = targets.min(BLOCK_BOXES);
    if fixed + Signatures::<T>::bytes_for(samples, floor) > budget {
        return None;
    }
    if targets == 0 {
        return Some(shape);
    }
    // The most boxes c ≥ floor with bytes_for(c) ≤ budget − fixed, at most `targets`.
    let size = size_of::<Complex<T>>();
    let room = (budget - fixed) / size;
    let line = Signatures::<T>::line();
    let stride = Signatures::<T>::stride_for(samples);
    let most = room.saturating_sub(line - 1) / stride;
    shape.chunk = most.clamp(floor, targets);
    shape.accumulators = Signatures::<T>::bytes_for(samples, shape.chunk);
    debug_assert!(shape.bytes() <= budget);
    Some(shape)
}

/// The buffers of one level call of the diagonal form on one thread: the source signatures,
/// a chunk of target accumulators, the conversions' scratch and a gather buffer of
/// [`BLOCK_BOXES`] expansions. Sized at construction; [`m2l_level`] allocates nothing.
#[derive(Debug)]
pub struct LevelBuffers<T> {
    sources: Signatures<T>,
    accumulators: Signatures<T>,
    scratch: ConversionScratch<T>,
    expansions: Vec<T>,
}

impl<T: RealScalar> Clone for LevelBuffers<T> {
    fn clone(&self) -> Self {
        Self {
            sources: self.sources.clone(),
            accumulators: self.accumulators.clone(),
            scratch: self.scratch.clone(),
            expansions: self.expansions.clone(),
        }
    }
}

impl<T: RealScalar> LevelBuffers<T> {
    /// Buffers for `conversions`' degree with room for `sources` source signatures and a chunk
    /// of `chunk` targets ([`ScratchShape::chunk`]).
    ///
    /// # Panics
    ///
    /// If `chunk` is 0.
    pub fn new(conversions: &Conversions<T>, sources: usize, chunk: usize) -> Self {
        assert!(chunk > 0, "a chunk holds at least one target");
        let samples = conversions.samples();
        Self {
            sources: Signatures::new(samples, sources),
            accumulators: Signatures::new(samples, chunk),
            scratch: ConversionScratch::new(conversions),
            expansions: vec![T::zero(); BLOCK_BOXES * conversions.expansion_len()],
        }
    }

    /// The targets per chunk.
    pub fn chunk(&self) -> usize {
        self.accumulators.capacity()
    }

    /// The bytes the buffers hold: [`ScratchShape`]'s `sources`, `accumulators` and one
    /// thread's share of `threads`.
    pub fn bytes(&self) -> usize {
        self.sources.bytes()
            + self.accumulators.bytes()
            + self.scratch.bytes()
            + self.expansions.len() * size_of::<T>()
    }
}

/// One level call of the diagonal form (module documentation, "The level's scratch"):
///
/// 1. S → F of the source boxes `sources` (indices of boxes in `multipoles`, each a stored
///    expansion of [`Conversions::expansion_len`] values, box after box) into the buffers'
///    source signatures, `sources[s]` into slot s;
/// 2. the targets `targets` (indices of boxes in `locals`, distinct: a box listed twice in
///    one block would be gathered twice) in chunks of
///    [`LevelBuffers::chunk`] in their order: each target's accumulator zeroed, then for each
///    of its pairs `rows[i]..rows[i + 1]`, in that order, `pair(i, k, signature, accumulator)`
///    with the signature of slot `slots[k]`; then F → R of the chunk added into the locals.
///
/// `pair` adds the pair's term to the target's accumulator (CH4.2: the transfer function of
/// the pair's offset class times the source's signature). Each local's bits depend on its
/// own row alone, not on the chunk size or the order of the other targets (tested). A target
/// with an empty row receives F → R of a zero accumulator (+0 on every value, which turns a
/// −0 into +0); a caller passes the targets with diagonal pairs.
///
/// # Panics
///
/// If the lengths disagree (`rows` not `targets.len() + 1` ascending offsets into `slots`, a
/// slot not below `sources.len()`, a box out of range), the buffers have fewer source slots
/// than `sources`, or another degree.
#[allow(clippy::too_many_arguments)]
pub fn m2l_level<T: RealScalar, F>(
    conversions: &Conversions<T>,
    multipoles: &[T],
    sources: &[usize],
    locals: &mut [T],
    targets: &[usize],
    rows: &[usize],
    slots: &[usize],
    buffers: &mut LevelBuffers<T>,
    mut pair: F,
) where
    F: FnMut(usize, usize, &[Complex<T>], &mut [Complex<T>]),
{
    let len = conversions.expansion_len();
    assert_eq!(
        rows.len(),
        targets.len() + 1,
        "rows need targets + 1 offsets"
    );
    assert_eq!(
        rows.last().copied(),
        Some(slots.len()),
        "the rows end at the pairs"
    );
    assert!(
        sources.len() <= buffers.sources.capacity(),
        "{} sources exceed the buffers' {}",
        sources.len(),
        buffers.sources.capacity()
    );
    let LevelBuffers {
        sources: signatures,
        accumulators,
        scratch,
        expansions,
    } = buffers;

    // 1. S → F, a block of gathered multipoles at a time.
    signatures.set_boxes(sources.len());
    for first in (0..sources.len()).step_by(BLOCK_BOXES) {
        let block = &sources[first..sources.len().min(first + BLOCK_BOXES)];
        for (slot, &b) in block.iter().enumerate() {
            expansions[slot * len..(slot + 1) * len]
                .copy_from_slice(&multipoles[b * len..(b + 1) * len]);
        }
        conversions.s2f_into(&expansions[..block.len() * len], signatures, first, scratch);
    }

    // 2. The targets by chunks.
    let chunk = accumulators.capacity();
    for t0 in (0..targets.len()).step_by(chunk) {
        let n = chunk.min(targets.len() - t0);
        accumulators.set_boxes(n);
        accumulators.zero();
        for i in t0..t0 + n {
            assert!(rows[i] <= rows[i + 1], "row offsets must ascend");
            let accumulator = accumulators.signature_mut(i - t0);
            let (from, to) = (rows[i], rows[i + 1]);
            for (k, &slot) in slots[from..to].iter().enumerate() {
                pair(i, from + k, signatures.signature(slot), accumulator);
            }
        }
        // F → R by blocks: the locals gathered, added into and scattered back (copies).
        for first in (0..n).step_by(BLOCK_BOXES) {
            let block = &targets[t0 + first..t0 + n.min(first + BLOCK_BOXES)];
            for (slot, &b) in block.iter().enumerate() {
                expansions[slot * len..(slot + 1) * len]
                    .copy_from_slice(&locals[b * len..(b + 1) * len]);
            }
            let gathered = &mut expansions[..block.len() * len];
            conversions.f2r_from(accumulators, first, gathered, scratch);
            for (slot, &b) in block.iter().enumerate() {
                locals[b * len..(b + 1) * len]
                    .copy_from_slice(&expansions[slot * len..(slot + 1) * len]);
            }
        }
    }
}
