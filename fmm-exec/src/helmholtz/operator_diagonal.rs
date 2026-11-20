//! The diagonal M2L of the high-frequency levels (Helmholtz CH4.2b1; proposal §3.6, §5.4), a
//! private part of the [`operator`](super) module ([module documentation](super#m2l)).
//!
//! # The rule
//!
//! Under [`HelmholtzDiagonal::Rule`] or [`HelmholtzDiagonal::Tolerance`] each level whose M2L
//! is RCR's under the operator's M2L setting (no waves, no Dense, Classes or Batched tables)
//! takes the offset classes c with E(l, c) ≤ θ ε (`nd_fmm_tables::helmholtz::diagonal::
//! class_bounds` at the level's degree and wavenumber, θ = `nd_fmm_ref`'s `THETA` = 1/2) to the
//! diagonal form, by [`diagonal_rule_classes`]: S6's constants of f64, or of f32 at ε ≥
//! [`DIAGONAL_F32_MIN_TOLERANCE`] only (decision D3). Every other class stays on RCR. A level holds
//! the transfer functions of its classes (`DiagonalM2l`, one part per (class, ring) on the pool)
//! and its conversions (`Conversions`) only where some class passes and its scratch fits the
//! budget: the conversions, one thread's scratch, the signatures of the level's distinct V sources
//! for every charge vector and a chunk of accumulators (`scratch_shape`), the sources counted as
//! the largest count over the ranks ([`with_diagonal_sources`]), so that every rank takes the same
//! levels (constraints 2, 7 and 9). A function of k, the level's width, its degree, ε and T,
//! and of the agreed count.
//!
//! # The level call
//!
//! On a level with transfer functions:
//!
//! 1. each distinct source of a diagonal pair takes a slot, in ascending box index, and S → F
//!    writes the signature of every (slot, vector) into the level's [`Signatures`] (index
//!    slot · m + v), blocks of `BLOCK_BOXES` signatures split over the pool, each block's
//!    multipoles gathered into the thread's scratch (a real multipole widened to two planes,
//!    its imaginary plane zero);
//! 2. the targets in chunks of consecutive rows split over the pool: in each chunk every
//!    accumulator (target, vector) is zeroed and receives its target's diagonal pairs in row
//!    order (`DiagonalM2l::pair`); F → R adds each accumulator of a target with a diagonal pair
//!    into its local; then each target adds its RCR pairs in row order (`HelmholtzTables::m2l`,
//!    Rotation's body).
//!
//! A signature depends on its multipole alone and a local on its own row alone (its
//! accumulator's chain is its diagonal pairs in row order, F → R gives each box its own bits
//! in any batch, then its RCR pairs in row order), so the output is the same bits at every
//! thread count, every chunk width and on P ranks the one-rank output (proposal §2,
//! constraint 2; §5.4, step 3). The accumulators are each thread's, a chunk of targets × m
//! signatures in the [`Signatures`] layout; one allocation per store holds every level's layout
//! (`Signatures::relayout`). The operator's store is sized from the source counts
//! ([`with_diagonal_sources`]); without them, and for the slots of a level's boxes, the first
//! level call of the largest level grows the buffer once.
//!
//! The levels depend on the budget per rank, the node's over the ranks per node, as the dense
//! tables' do: where the budget binds, a level may take the diagonal form at one rank per node
//! and not at many, so the output on P ranks is the one-rank FMM's bit for bit where the levels
//! taken agree (always, where no level falls back).
//!
//! The pair body ([`PairOperator`](nd_fmm_plan::operator::PairOperator)) takes a diagonal
//! offset by S → F of the source, the pair and F → R of the pair alone: the level call to
//! rounding (there a target's diagonal pairs are summed as signatures before one F → R).
//!
//! [`with_diagonal_sources`]: super::HelmholtzOperator::with_diagonal_sources

use std::sync::PoisonError;
use std::time::{Duration, Instant};

use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::operator::M2l;
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_ref::helmholtz::diagonal::{BoundConstants, THETA};
use nd_fmm_tables::helmholtz::diagonal::{
    BLOCK_BOXES, ConversionScratch, Conversions, DiagonalM2l, Signatures, class_bounds,
    scratch_shape,
};
use nd_fmm_tables::parts::Parts;

use super::{HelmholtzOperator, Kernel, SimdScalar};

/// The diagonal M2L of the high-frequency levels (Helmholtz CH4.2b1; opt-in until CH4.3;
/// module documentation of [`operator`](super), "M2L").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum HelmholtzDiagonal {
    /// No level takes the diagonal form (the default).
    #[default]
    Off,
    /// The rule at the FMM's tolerance
    /// ([`with_plane_wave_tolerance`](super::HelmholtzOperator::with_plane_wave_tolerance),
    /// which `HelmholtzFmmBuilder` sets from `Degrees::Tolerance`); with named degrees and no
    /// tolerance no level.
    Rule,
    /// The rule at the named tolerance ε (for the tests and the measurements, with named
    /// degrees too).
    Tolerance(f64),
}

impl HelmholtzDiagonal {
    /// The tolerance the rule reads, given the FMM's: `None` under [`Off`](Self::Off) and under
    /// [`Rule`](Self::Rule) without one.
    pub fn tolerance(self, fmm: Option<f64>) -> Option<f64> {
        match self {
            Self::Off => None,
            Self::Rule => fmm,
            Self::Tolerance(epsilon) => Some(epsilon),
        }
    }
}

/// The diagonal form in f32 runs only at tolerances ε ≥ 10⁻⁴ (decision D3, signed off
/// 2026-11-06: f32 on the high-frequency levels at ε ≥ 10⁻⁴, each class by the bound in
/// u = 2⁻²⁴; f64 below).
pub const DIAGONAL_F32_MIN_TOLERANCE: f64 = 1e-4;

/// The offset classes (indices into `nd_fmm_ref::helmholtz::diagonal::classes()`) that take the
/// diagonal form on a level of degree p and wavenumber k̂ in its units at tolerance ε, in `T`:
/// those with E(l, c) ≤ θ ε (`class_bounds` with `BoundConstants::F64`, or `F32` for a 4-byte
/// `T`, where none below [`DIAGONAL_F32_MIN_TOLERANCE`]). A function of its arguments alone
/// (proposal §2, constraint 9).
pub fn diagonal_rule_classes<T: SimdScalar>(
    p: usize,
    k_hat: Complex<f64>,
    epsilon: f64,
) -> Vec<usize> {
    let single = size_of::<T>() == 4;
    if single && epsilon < DIAGONAL_F32_MIN_TOLERANCE {
        return Vec::new();
    }
    let constants = if single {
        BoundConstants::F32
    } else {
        BoundConstants::F64
    };
    class_bounds(p, k_hat, constants)
        .into_iter()
        .enumerate()
        .filter(|&(_, e)| e <= THETA * epsilon)
        .map(|(c, _)| c)
        .collect()
}

/// The width of the diagonal level call's chunks of targets on a level of `rows` boxes run on
/// `threads` threads: `BLOCK_BOXES` on one thread, else ⌈rows / threads⌉ within 1 and
/// `BLOCK_BOXES`. An execution split only (module documentation): no bit depends on it.
pub fn diagonal_chunk_targets(rows: usize, threads: usize) -> usize {
    if threads <= 1 {
        return BLOCK_BOXES;
    }
    rows.div_ceil(threads).clamp(1, BLOCK_BOXES)
}

/// The counts of the diagonal level calls since the operator's M2L was last chosen
/// ([`HelmholtzOperator::diagonal_counts`]; Helmholtz CH4.2b1), per level.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DiagonalCounts {
    /// Level calls of the diagonal form.
    pub level_calls: Vec<usize>,
    /// V pairs taken by the diagonal form, summed over the calls.
    pub diagonal_pairs: Vec<usize>,
    /// V pairs of those calls left on RCR.
    pub rcr_pairs: Vec<usize>,
    /// Source signatures formed (slots × vectors), summed over the calls.
    pub signatures: Vec<usize>,
    /// Accumulators converted back by F → R (targets with a diagonal pair × vectors), summed
    /// over the calls (Helmholtz CH4.2b2; counted by one serial pass over the level's pairs on
    /// the calling thread, outside both timed passes but inside the level's M2L time).
    pub back: Vec<usize>,
}

impl DiagonalCounts {
    fn reset(&mut self, nlevels: usize) {
        for counts in [
            &mut self.level_calls,
            &mut self.diagonal_pairs,
            &mut self.rcr_pairs,
            &mut self.signatures,
            &mut self.back,
        ] {
            counts.clear();
            counts.resize(nlevels, 0);
        }
    }
}

/// The diagonal level calls' times on one level since the operator's M2L was last chosen
/// ([`HelmholtzOperator::diagonal_times`]; Helmholtz CH4.2b2): the two passes' wall time on the
/// calling thread's clock, and the workers' thread-seconds by part, each worker's own clock
/// (a few clock reads per block of signatures and per chunk of targets; no value changes).
/// This rank's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiagonalTimes {
    /// The S → F pass (module documentation, "The level call", step 1), wall.
    pub signatures: Duration,
    /// The targets' pass (step 2: the diagonal pairs, F → R and the RCR pairs), wall.
    pub targets: Duration,
    /// S → F's thread-seconds, the multipoles' gather included (part of `signatures` × the
    /// threads).
    pub s2f: Duration,
    /// The diagonal pairs' thread-seconds, the accumulators' zeroing included (part of
    /// `targets` × the threads).
    pub pairs: Duration,
    /// F → R's thread-seconds.
    pub f2r: Duration,
    /// The RCR pairs' thread-seconds.
    pub rcr: Duration,
}

impl DiagonalTimes {
    fn add_parts(&mut self, parts: &Self) {
        self.s2f += parts.s2f;
        self.pairs += parts.pairs;
        self.f2r += parts.f2r;
        self.rcr += parts.rcr;
    }
}

/// The diagonal settings of the operator (module documentation, "The rule").
#[derive(Clone, Debug, Default)]
pub(super) struct DiagonalSettings {
    pub(super) choice: HelmholtzDiagonal,
    /// The byte budget of a level's scratch on this rank, or `None` for no bound.
    pub(super) budget: Option<usize>,
    /// Each level's distinct V sources, the largest count over the ranks, if known.
    pub(super) sources: Option<Vec<usize>>,
    /// The chunk width if named, else [`diagonal_chunk_targets`]'s.
    pub(super) chunk: Option<usize>,
}

impl DiagonalSettings {
    /// The widest chunk a level call may take.
    fn max_chunk(&self) -> usize {
        self.chunk.unwrap_or(BLOCK_BOXES).max(1)
    }
}

/// One level's tables of the diagonal form.
pub(super) struct DiagonalLevel<T> {
    pub(super) m2l: DiagonalM2l<T>,
    pub(super) conversions: Conversions<T>,
    /// The wall time of building the transfer functions and the conversions (Helmholtz
    /// CH4.2b2).
    pub(super) build: (Duration, Duration),
}

impl<T: SimdScalar> DiagonalLevel<T> {
    fn samples(&self) -> usize {
        self.conversions.samples()
    }
}

/// The levels that take the diagonal form and their tables (module documentation, "The
/// rule"): for each level l ≥ 2 with `rcr[l]` (its M2L RCR's), degree `degrees[l]` and
/// wavenumber `k_hat(l)` in its units, the classes of [`diagonal_rule_classes`] at the
/// setting's tolerance, where some class passes and the scratch fits the budget; `None`
/// elsewhere. The transfer functions are built on `parts`, one part per (class, ring).
pub(super) fn select<T: SimdScalar>(
    settings: &DiagonalSettings,
    fmm_tolerance: Option<f64>,
    rcr: &[bool],
    degrees: &[usize],
    k_hat: impl Fn(usize) -> Complex<f64>,
    vectors: usize,
    parts: &impl Parts,
) -> Vec<Option<DiagonalLevel<T>>> {
    let Some(epsilon) = settings.choice.tolerance(fmm_tolerance) else {
        return (0..degrees.len()).map(|_| None).collect();
    };
    (0..degrees.len())
        .map(|l| {
            let p = degrees[l];
            if l < 2 || !rcr[l] {
                return None;
            }
            let sources = settings
                .sources
                .as_ref()
                .map(|s| s.get(l).copied().unwrap_or(0));
            if sources == Some(0) {
                return None;
            }
            if let (Some(budget), Some(sources)) = (settings.budget, sources) {
                let accumulators = settings.max_chunk() * vectors;
                scratch_shape::<T>(p, sources * vectors, accumulators, 1, budget)?;
            }
            let k = k_hat(l);
            let classes = diagonal_rule_classes::<T>(p, k, epsilon);
            if classes.is_empty() {
                return None;
            }
            let level = Level::new(k, 1.0);
            let clock = Instant::now();
            let m2l = DiagonalM2l::build_on(p, k, &classes, parts);
            let transfer = clock.elapsed();
            let clock = Instant::now();
            let conversions = Conversions::new(p, level.scale());
            Some(DiagonalLevel {
                m2l,
                conversions,
                build: (transfer, clock.elapsed()),
            })
        })
        .collect()
}

/// A store of signatures with room for each of `layouts` (samples, boxes): made for the largest
/// in bytes, laid out again per level (`Signatures::relayout`).
fn store_for<T: SimdScalar>(layouts: impl Iterator<Item = (usize, usize)>) -> Signatures<T> {
    let (samples, boxes) = layouts
        .max_by_key(|&(samples, boxes)| Signatures::<T>::bytes_for(samples, boxes))
        .unwrap_or((1, 0));
    Signatures::new(samples, boxes)
}

/// One thread's scratch of the diagonal form: each level's conversion scratch, the
/// accumulators of a chunk of targets × vectors (at least two signatures, for the pair body),
/// and the gather buffer of `BLOCK_BOXES` expansions of two planes.
pub(super) struct DiagonalScratch<T> {
    conversions: Vec<Option<ConversionScratch<T>>>,
    accumulators: Signatures<T>,
    gather: Vec<T>,
    /// This thread's time by part since the level call last drained it (the thread-seconds of
    /// [`DiagonalTimes`]).
    parts: DiagonalTimes,
}

impl<T: SimdScalar> DiagonalScratch<T> {
    /// Scratch for `levels`, `vectors` charge vectors and chunks of at most `settings`' width;
    /// `None` without a diagonal level.
    pub(super) fn new(
        levels: &[Option<DiagonalLevel<T>>],
        settings: &DiagonalSettings,
        vectors: usize,
    ) -> Option<Self> {
        let p_max = levels.iter().flatten().map(|d| d.m2l.degree()).max()?;
        let boxes = (settings.max_chunk() * vectors).max(2);
        Some(Self {
            conversions: levels
                .iter()
                .map(|d| d.as_ref().map(|d| ConversionScratch::new(&d.conversions)))
                .collect(),
            accumulators: store_for(levels.iter().flatten().map(|d| (d.samples(), boxes))),
            gather: vec![T::zero(); BLOCK_BOXES * 2 * Layout::new(p_max).len()],
            parts: DiagonalTimes::default(),
        })
    }

    /// Its bytes.
    pub(super) fn bytes(&self) -> usize {
        self.conversions
            .iter()
            .flatten()
            .map(ConversionScratch::bytes)
            .sum::<usize>()
            + self.accumulators.bytes()
            + self.gather.len() * size_of::<T>()
    }
}

/// The operator's buffer of the diagonal level call: the source signatures of a level (slots ×
/// vectors) and the slots of its boxes.
pub(super) struct DiagonalBuffer<T> {
    signatures: Option<Signatures<T>>,
    /// Per box of the level, its slot + 1, or 0.
    slot_of: Vec<u32>,
    /// Per slot, its box.
    slot_source: Vec<u32>,
    pub(super) counts: DiagonalCounts,
    /// The level calls' times per level (Helmholtz CH4.2b2), reset with the counts.
    pub(super) times: Vec<DiagonalTimes>,
    /// The most bytes the buffer held.
    pub(super) peak: usize,
}

impl<T> Default for DiagonalBuffer<T> {
    fn default() -> Self {
        Self {
            signatures: None,
            slot_of: Vec::new(),
            slot_source: Vec::new(),
            counts: DiagonalCounts::default(),
            times: Vec::new(),
            peak: 0,
        }
    }
}

impl<T: SimdScalar> DiagonalBuffer<T> {
    /// Sizes the signatures for `levels` with the settings' source counts (none without them:
    /// the first level call grows them), and resets the counts.
    pub(super) fn size(
        &mut self,
        levels: &[Option<DiagonalLevel<T>>],
        settings: &DiagonalSettings,
        vectors: usize,
    ) {
        self.counts.reset(levels.len());
        self.times.clear();
        self.times.resize(levels.len(), DiagonalTimes::default());
        self.signatures = None;
        if let Some(sources) = &settings.sources {
            let layouts = levels.iter().enumerate().filter_map(|(l, d)| {
                d.as_ref()
                    .map(|d| (d.samples(), sources.get(l).copied().unwrap_or(0) * vectors))
            });
            let store = store_for(layouts);
            if store.capacity() > 0 {
                self.signatures = Some(store);
            }
        }
        self.peak = self.peak.max(self.bytes());
    }

    /// Its bytes.
    pub(super) fn bytes(&self) -> usize {
        self.signatures.as_ref().map_or(0, Signatures::bytes)
            + 4 * (self.slot_of.capacity() + self.slot_source.capacity())
    }
}

/// `multipole`'s one vector of `planes` planes as two planes in `out` (2 (p + 1)² values): a
/// real multipole's imaginary plane zero.
fn widen<T: SimdScalar>(multipole: &[T], planes: usize, out: &mut [T]) {
    if planes == 2 {
        out.copy_from_slice(multipole);
    } else {
        let len = multipole.len();
        out[..len].copy_from_slice(multipole);
        out[len..].fill(T::zero());
    }
}

impl<T: SimdScalar> Kernel<T> {
    /// The diagonal form's tables of `level`, if it has them.
    pub(super) fn diagonal_level(&self, level: usize) -> Option<&DiagonalLevel<T>> {
        self.diagonal.get(level).and_then(Option::as_ref)
    }

    /// The pair body of a diagonal offset `d` on `level` (module documentation): per vector
    /// S → F of the source, the pair into a zeroed accumulator, F → R added into the local.
    pub(super) fn diagonal_pair(
        &self,
        level: usize,
        d: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut DiagonalScratch<T>,
    ) {
        let diagonal = self
            .diagonal_level(level)
            .expect("a diagonal level has tables");
        let (m2l, conversions) = (&diagonal.m2l, &diagonal.conversions);
        let planes = self.densities.multipole_planes();
        let (len, samples) = (conversions.expansion_len(), conversions.samples());
        let mlen = planes * len / 2;
        let DiagonalScratch {
            conversions: scratches,
            accumulators,
            gather,
            ..
        } = scratch;
        let conversion = scratches[level]
            .as_mut()
            .expect("the scratch has the level's conversions");
        accumulators.relayout(samples, 2);
        let stride = accumulators.stride();
        for (v, local) in local.chunks_exact_mut(len).enumerate() {
            let expansion = &mut gather[..len];
            widen(&multipole[v * mlen..(v + 1) * mlen], planes, expansion);
            let (signature, accumulator) = accumulators.strides_mut().split_at_mut(stride);
            conversions.s2f_strides(expansion, signature, stride, conversion);
            let accumulator = &mut accumulator[..samples];
            accumulator.fill(Complex::new(T::zero(), T::zero()));
            m2l.pair(d, &signature[..samples], accumulator);
            conversions.f2r(accumulator, local, conversion);
        }
    }
}

impl<T: SimdScalar> HelmholtzOperator<T> {
    /// The M2L level call of a level with the diagonal form's tables (module documentation,
    /// "The level call").
    pub(super) fn diagonal_m2l(&mut self, batch: M2l<'_, T>) {
        let threads = match (&self.execution.pool, self.execution.serial) {
            (Some(_), false) => self.execution.scratch.len(),
            _ => 1,
        };
        let M2l {
            level,
            pairs,
            multipoles,
            mut locals,
            ..
        } = batch;
        let kernel = &self.kernel;
        let diagonal = kernel
            .diagonal_level(level)
            .expect("a diagonal level has tables");
        let (m2l, conversions) = (&diagonal.m2l, &diagonal.conversions);
        let (planes, m) = (
            kernel.densities.multipole_planes(),
            kernel.densities.vectors(),
        );
        let (len, samples) = (conversions.expansion_len(), conversions.samples());
        let mlen = planes * len / 2;
        let size = locals.size();
        assert_eq!(size, m * len, "a local is m expansions of two planes");
        let (row_offsets, sources, offsets) =
            (pairs.row_offsets(), pairs.sources(), pairs.groups());
        let buffer = &mut self.diagonal_buffer;

        // 1. The slots, in ascending box index.
        let nboxes = multipoles.len();
        // Grown at the first level call of the largest level (its boxes are the plan's, own
        // and ghost, which the operator is not given), then reused.
        if buffer.slot_of.len() < nboxes {
            buffer.slot_of.resize(nboxes, 0);
            buffer.slot_source.resize(nboxes, 0);
        }
        let slot_of = &mut buffer.slot_of[..nboxes];
        slot_of.fill(0);
        let (mut diagonal_pairs, mut rcr_pairs) = (0, 0);
        for (&s, &d) in sources.iter().zip(offsets) {
            if m2l.has_offset(d as usize) {
                slot_of[s as usize] = 1;
                diagonal_pairs += 1;
            } else {
                rcr_pairs += 1;
            }
        }
        let mut slots = 0;
        for (s, slot) in slot_of.iter_mut().enumerate() {
            if *slot != 0 {
                buffer.slot_source[slots] = s as u32;
                slots += 1;
                *slot = slots as u32;
            }
        }
        let counts = &mut buffer.counts;
        counts.level_calls[level] += 1;
        counts.diagonal_pairs[level] += diagonal_pairs;
        counts.rcr_pairs[level] += rcr_pairs;
        counts.signatures[level] += slots * m;

        // 2. S → F of every (slot, vector), blocks over the pool.
        if !buffer
            .signatures
            .as_ref()
            .is_some_and(|s| s.fits(samples, slots * m))
        {
            // Not sized for this level (no source counts given): grown here, once for the
            // largest level met.
            buffer.signatures = Some(Signatures::new(samples, slots * m));
        }
        buffer.peak = buffer.peak.max(buffer.bytes());
        let signatures = buffer
            .signatures
            .as_mut()
            .expect("the signatures are sized");
        signatures.relayout(samples, slots * m);
        let stride = signatures.stride();
        let slot_source = &buffer.slot_source[..slots];
        if slots > 0 {
            let clock = Instant::now();
            self.execution.chunks(
                signatures.strides_mut(),
                stride * BLOCK_BOXES,
                |j, strides, scratch| {
                    let DiagonalScratch {
                        conversions: scratches,
                        gather,
                        parts,
                        ..
                    } = scratch
                        .diagonal
                        .as_mut()
                        .expect("the scratch has the diagonal form's buffers");
                    let busy = Instant::now();
                    let first = j * BLOCK_BOXES;
                    let nb = strides.len() / stride;
                    for i in 0..nb {
                        let (slot, v) = ((first + i) / m, (first + i) % m);
                        let multipole = multipoles.chunk(slot_source[slot] as usize);
                        widen(
                            &multipole[v * mlen..(v + 1) * mlen],
                            planes,
                            &mut gather[i * len..(i + 1) * len],
                        );
                    }
                    let conversion = scratches[level]
                        .as_mut()
                        .expect("the scratch has the level's conversions");
                    conversions.s2f_strides(&gather[..nb * len], strides, stride, conversion);
                    parts.s2f += busy.elapsed();
                },
            );
            buffer.times[level].signatures += clock.elapsed();
        }

        // 3. The targets by chunks of rows: the diagonal pairs, F → R, the RCR pairs.
        let rows = row_offsets.len() - 1;
        let chunk = kernel
            .diagonal_settings
            .chunk
            .unwrap_or_else(|| diagonal_chunk_targets(rows, threads))
            .max(1);
        let signatures = &*signatures;
        let slot_of = &buffer.slot_of[..nboxes];
        let row = |t: usize| row_offsets[t] as usize..row_offsets[t + 1] as usize;
        let has_diagonal = |t: usize| row(t).any(|g| m2l.has_offset(offsets[g] as usize));
        buffer.counts.back[level] += (0..rows).filter(|&t| has_diagonal(t)).count() * m;
        let clock = Instant::now();
        self.execution
            .chunks(locals.as_mut_slice(), size * chunk, |j, block, scratch| {
                let super::Scratch {
                    diagonal, tables, ..
                } = scratch;
                let DiagonalScratch {
                    conversions: scratches,
                    accumulators,
                    parts,
                    ..
                } = diagonal
                    .as_mut()
                    .expect("the scratch has the diagonal form's buffers");
                let conversion = scratches[level]
                    .as_mut()
                    .expect("the scratch has the level's conversions");
                let (t0, n) = (j * chunk, block.len() / size);
                let busy = Instant::now();
                accumulators.relayout(samples, n * m);
                accumulators.zero();
                let mut diagonal_pairs = false;
                for i in 0..n {
                    for g in row(t0 + i) {
                        let d = offsets[g] as usize;
                        if !m2l.has_offset(d) {
                            continue;
                        }
                        diagonal_pairs = true;
                        let slot = slot_of[sources[g] as usize] as usize - 1;
                        for v in 0..m {
                            m2l.pair(
                                d,
                                signatures.signature(slot * m + v),
                                accumulators.signature_mut(i * m + v),
                            );
                        }
                    }
                }
                // F → R by runs of consecutive targets with a diagonal pair.
                let converted = Instant::now();
                if diagonal_pairs {
                    parts.pairs += converted - busy;
                }
                let mut i = 0;
                while i < n {
                    if !has_diagonal(t0 + i) {
                        i += 1;
                        continue;
                    }
                    let mut end = i + 1;
                    while end < n && has_diagonal(t0 + end) {
                        end += 1;
                    }
                    conversions.f2r_from(
                        accumulators,
                        i * m,
                        &mut block[i * size..end * size],
                        conversion,
                    );
                    i = end;
                }
                if diagonal_pairs {
                    parts.f2r += converted.elapsed();
                }
                // The RCR pairs, each target's in row order.
                let rcr = Instant::now();
                let mut rcr_pairs = false;
                for (i, local) in block.chunks_exact_mut(size).enumerate() {
                    for g in row(t0 + i) {
                        let d = offsets[g] as usize;
                        if m2l.has_offset(d) {
                            continue;
                        }
                        rcr_pairs = true;
                        kernel.tables.m2l(
                            level,
                            d,
                            multipoles.chunk(sources[g] as usize),
                            planes,
                            local,
                            m,
                            tables,
                        );
                    }
                }
                if rcr_pairs {
                    parts.rcr += rcr.elapsed();
                }
            });
        let times = &mut buffer.times[level];
        times.targets += clock.elapsed();
        // Every thread's parts, into the level's times.
        for set in &mut self.execution.scratch {
            let set = set.get_mut().unwrap_or_else(PoisonError::into_inner);
            if let Some(diagonal) = set.diagonal.as_mut() {
                times.add_parts(&core::mem::take(&mut diagonal.parts));
            }
        }
    }
}
