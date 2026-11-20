//! The diagonal M2L of the high-frequency levels (Helmholtz CH4.2b1; proposal §3.6, §5.4), a
//! private part of the [`operator`](super) module ([module documentation](super#m2l)).
//!
//! # The rule
//!
//! Under [`HelmholtzDiagonal::Rule`] or [`HelmholtzDiagonal::Tolerance`] each level whose M2L
//! is RCR's under the operator's M2L setting (no waves, no Dense, Classes or Batched tables)
//! takes the offset classes c with E(l, c) ≤ θ ε (`nd_fmm_tables::helmholtz::diagonal::
//! class_bounds` at the level's degree and wavenumber, θ = `nd_fmm_ref`'s `THETA` = 1/2) to the
//! diagonal form, by [`diagonal_rule_classes`]: S6's constants of the setting's build of the
//! transfer functions in f64 ([`TransferBuild::Direct`]'s C_T = 0.15, `Interpolated`'s 0.23,
//! the default [`DEFAULT_DIAGONAL_BUILD`]; Helmholtz CH4.2c2), or of f32 at ε ≥ [`DIAGONAL_F32_MIN_TOLERANCE`] only (decision D3) by
//! either build. Every other class stays on RCR. A level holds
//! the transfer functions of its classes (`DiagonalM2l::build_by` of the setting's build, one
//! part per (class, ring) on the pool)
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

use std::sync::{LazyLock, PoisonError};
use std::time::{Duration, Instant};

use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::operator::M2l;
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_ref::helmholtz::diagonal::{THETA, class_of, classes};
use nd_fmm_tables::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use nd_fmm_tables::helmholtz::diagonal::{
    BLOCK_BOXES, CLASS_COUNT, ConversionScratch, Conversions, DiagonalM2l, Signatures,
    TransferBuild, class_bounds, scratch_shape,
};
use nd_fmm_tables::parts::Parts;

use super::{HelmholtzOperator, Isa, Kernel, SimdScalar};

/// The diagonal M2L of the high-frequency levels (Helmholtz CH4.2b1; opt-in until CH4.3b2;
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
    /// The rule at the FMM's tolerance, then by time (Helmholtz CH4.3b1; module documentation
    /// of [`operator`](super), "M2L", the time rule), on the levels of k a ≥
    /// [`DIAGONAL_AUTO_MIN_KA`] only: a level takes the bound's classes only
    /// where the model ([`DiagonalCosts`]) puts its diagonal level below the M2L it would run
    /// otherwise by the level's agreed V lists ([`DiagonalShape`]): RCR's
    /// ([`diagonal_beats_rcr`]), or under [`HelmholtzM2l::Auto`](super::HelmholtzM2l::Auto) the
    /// exponential M2L's ([`diagonal_beats_plane_waves`]), which then runs the level at its own
    /// degree.
    Auto,
}

impl HelmholtzDiagonal {
    /// The tolerance the rule reads, given the FMM's: `None` under [`Off`](Self::Off) and under
    /// [`Rule`](Self::Rule) without one.
    pub fn tolerance(self, fmm: Option<f64>) -> Option<f64> {
        match self {
            Self::Off => None,
            Self::Rule | Self::Auto => fmm,
            Self::Tolerance(epsilon) => Some(epsilon),
        }
    }
}

/// The diagonal form in f32 runs only at tolerances ε ≥ 10⁻⁴ (decision D3, signed off
/// 2026-11-06: f32 on the high-frequency levels at ε ≥ 10⁻⁴, each class by the bound in
/// u = 2⁻²⁴; f64 below).
pub const DIAGONAL_F32_MIN_TOLERANCE: f64 = 1e-4;

/// The offset classes (indices into `nd_fmm_ref::helmholtz::diagonal::classes()`) that take the
/// diagonal form on a level of degree p and wavenumber k̂ in its units at tolerance ε, in `T`,
/// with transfer functions of `build`: those with E(l, c) ≤ θ ε (`class_bounds` with
/// [`TransferBuild::bound_constants`]: `BoundConstants::F64` direct, `F64_INTERPOLATED`
/// interpolated (Helmholtz CH4.2c2), or `F32` for a 4-byte `T` by either build, where none below
/// [`DIAGONAL_F32_MIN_TOLERANCE`]). A function of its arguments alone (proposal §2,
/// constraint 9).
pub fn diagonal_rule_classes<T: SimdScalar>(
    p: usize,
    k_hat: Complex<f64>,
    epsilon: f64,
    build: TransferBuild,
) -> Vec<usize> {
    let single = size_of::<T>() == 4;
    if single && epsilon < DIAGONAL_F32_MIN_TOLERANCE {
        return Vec::new();
    }
    class_bounds(p, k_hat, build.bound_constants(single))
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

/// The smallest k a = Re(k̂_l) √3 at which the time rule ([`HelmholtzDiagonal::Auto`]) takes a
/// level: S6's calibrated range (Helmholtz CH4.3b1, CH4.3a's question 3: below it one damped
/// class measured 2.9 E per pair, direct build, η = 0.1, k a 1.69). The bound's rule under
/// [`HelmholtzDiagonal::Rule`] and [`HelmholtzDiagonal::Tolerance`] reaches below it. On W1 and
/// W3 the time rule took no level below it before the bound (fmm-validate/results/
/// helmholtz-ch4-3b1.md): the levels there are small, and RCR is the cheaper.
pub const DIAGONAL_AUTO_MIN_KA: f64 = 2.8;

/// Whether the time rule may take a level of wavenumber `k_hat` in its units: k a ≥
/// [`DIAGONAL_AUTO_MIN_KA`], a = √3 the circumradius of a box of half-width 1.
pub fn diagonal_auto_in_range(k_hat: Complex<f64>) -> bool {
    k_hat.re * 3f64.sqrt() >= DIAGONAL_AUTO_MIN_KA
}

/// The class of each V offset, by the offset's table index (`nd_fmm_tables::geometry::
/// m2l_offsets`), an index into `nd_fmm_ref::helmholtz::diagonal::classes()`.
static OFFSET_CLASSES: LazyLock<[u8; M2L_OFFSET_COUNT]> = LazyLock::new(|| {
    let classes = classes();
    m2l_offsets().map(|d| {
        let (representative, _) = class_of(d.map(|x| x as i32));
        let class = classes
            .iter()
            .position(|c| c.representative == representative)
            .expect("every V offset lies in a class");
        u8::try_from(class).expect("34 classes")
    })
});

/// The class of the V offset of table index `offset` (an index into
/// `nd_fmm_ref::helmholtz::diagonal::classes()`; Helmholtz CH4.3b1).
///
/// # Panics
///
/// If `offset` is not below 316.
pub fn offset_class(offset: usize) -> usize {
    usize::from(OFFSET_CLASSES[offset])
}

/// A level's V lists as the diagonal form's time rule reads them (Helmholtz CH4.3b1): the
/// boxes with a V row and the V pairs of each offset class, each target counted by the rank
/// that computes it and summed over the ranks, so the same on every rank and at every rank
/// count (`HelmholtzFmmBuilder::build`'s one all-reduce).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagonalShape {
    /// The level's boxes with a V row.
    pub targets: u64,
    /// The V pairs of each class (indices into `nd_fmm_ref::helmholtz::diagonal::classes()`).
    pub class_pairs: [u64; CLASS_COUNT],
}

impl Default for DiagonalShape {
    fn default() -> Self {
        Self {
            targets: 0,
            class_pairs: [0; CLASS_COUNT],
        }
    }
}

impl DiagonalShape {
    /// The V pairs.
    pub fn pairs(&self) -> u64 {
        self.class_pairs.iter().sum()
    }

    /// The V lists of the 8 children of an interior box of a uniform level: each child's 189
    /// offsets, 8 targets and 1,512 pairs, by class. The time rule's shape on an operator
    /// without the builder's counts.
    pub fn uniform() -> Self {
        let mut shape = Self {
            targets: 8,
            ..Self::default()
        };
        for child in 0..8i64 {
            let c = [child & 1, (child >> 1) & 1, (child >> 2) & 1];
            for (offset, d) in m2l_offsets().iter().enumerate() {
                // The source child of the parent's neighbour: target + d must lie in a box
                // whose parent is the target's parent's neighbour (or the parent itself).
                let parent = [0, 1, 2].map(|k| (c[k] + d[k]).div_euclid(2));
                if parent.iter().all(|x| x.abs() <= 1) {
                    shape.class_pairs[offset_class(offset)] += 1;
                }
            }
        }
        shape
    }
}

/// The cost model of the diagonal form's time rule for one machine class (Helmholtz CH4.3b1,
/// [`HelmholtzDiagonal::Auto`]): each unit's cost a (p + 1)² + b (p + 1)³ thread-ns, [a, b],
/// at the level's degree p. Each is a level call's wall time × its threads per unit, idle
/// threads included (one rank, every core a thread), so that the rule compares whole level
/// calls: S → F per source and F → R per target, the diagonal pair, RCR's pair, and the
/// exponential M2L per box at its degree p_w. Fitted by `tools/helmholtz/ch4-3b1-fit.py`
/// (fmm-validate/results/helmholtz-ch4-3b1.md); [`VCosts`](super::super::VCosts) fits thread
/// time on the leaf levels for the V near rule, a different measure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagonalCosts {
    /// S → F per source signature.
    pub s2f: [f64; 2],
    /// F → R per target converted back.
    pub f2r: [f64; 2],
    /// The diagonal pair.
    pub pair: [f64; 2],
    /// RCR's pair (Rotation's M2L).
    pub rotation: [f64; 2],
    /// The exponential M2L per box with a V row, at its degree p_w: the waves of its source
    /// and merge and shift into it, both per box (on W1 and W3 alike, whatever the pairs per
    /// box; fmm-validate/results/helmholtz-ch4-3b1.md).
    pub plane_wave: [f64; 2],
}

/// The number of constants of a [`DiagonalCosts`], for its agreement over the ranks.
pub const DIAGONAL_COST_CONSTANTS: usize = 10;

impl DiagonalCosts {
    /// A unit's cost a (p + 1)² + b (p + 1)³ at degree p.
    pub fn unit(unit: [f64; 2], p: usize) -> f64 {
        let q = (p + 1) as f64;
        unit[0] * q * q + unit[1] * q * q * q
    }

    /// The modelled cost of a level of degree `p` and V lists `shape` with the classes
    /// `classes` diagonal and the others RCR's: S → F and F → R once per box with a V row (the
    /// model's one approximation: the distinct sources are taken as the targets, which the V
    /// relation makes them on a level whose boxes all hold sources and targets), each diagonal
    /// class's pairs at the pair's cost and the rest at RCR's.
    pub fn diagonal_level(&self, p: usize, shape: &DiagonalShape, classes: &[usize]) -> f64 {
        let boxes = shape.targets as f64 * (Self::unit(self.s2f, p) + Self::unit(self.f2r, p));
        let diagonal: u64 = classes.iter().map(|&c| shape.class_pairs[c]).sum();
        let rcr = shape.pairs() - diagonal;
        boxes
            + diagonal as f64 * Self::unit(self.pair, p)
            + rcr as f64 * Self::unit(self.rotation, p)
    }

    /// The modelled cost of the level by RCR alone.
    pub fn rcr_level(&self, p: usize, shape: &DiagonalShape) -> f64 {
        shape.pairs() as f64 * Self::unit(self.rotation, p)
    }

    /// The modelled cost of the level by the exponential M2L at its degree `p_w`: per box
    /// with a V row.
    pub fn plane_wave_level(&self, p_w: usize, shape: &DiagonalShape) -> f64 {
        shape.targets as f64 * Self::unit(self.plane_wave, p_w)
    }

    /// The constants in field order, for their agreement over the ranks.
    pub fn to_array(&self) -> [f64; DIAGONAL_COST_CONSTANTS] {
        let [s2f, f2r, pair, rotation, plane_wave] = [
            self.s2f,
            self.f2r,
            self.pair,
            self.rotation,
            self.plane_wave,
        ];
        [
            s2f[0],
            s2f[1],
            f2r[0],
            f2r[1],
            pair[0],
            pair[1],
            rotation[0],
            rotation[1],
            plane_wave[0],
            plane_wave[1],
        ]
    }

    /// The costs of [`to_array`](Self::to_array)'s constants.
    pub fn from_array(c: [f64; DIAGONAL_COST_CONSTANTS]) -> Self {
        Self {
            s2f: [c[0], c[1]],
            f2r: [c[2], c[3]],
            pair: [c[4], c[5]],
            rotation: [c[6], c[7]],
            plane_wave: [c[8], c[9]],
        }
    }
}

/// [`DiagonalCosts`] of x86 (AVX-512 and AVX2; measured on Kathleen, Xeon Gold 6248, one rank
/// × 40 threads).
pub const DIAGONAL_COSTS_X86: DiagonalCosts = DiagonalCosts {
    s2f: [420.0, 2.01],
    f2r: [246.0, 6.01],
    pair: [23.9, 0.525],
    rotation: [0.0, 8.62],
    plane_wave: [65.2, 2.44],
};

/// [`DiagonalCosts`] of NEON (measured on locust, Neoverse-V2, one rank × 72 threads; the M6
/// takes them unmeasured).
pub const DIAGONAL_COSTS_NEON: DiagonalCosts = DiagonalCosts {
    s2f: [108.0, 1.58],
    f2r: [92.0, 3.26],
    pair: [13.0, 0.224],
    rotation: [0.0, 3.2],
    plane_wave: [1210.0, 39.2],
};

/// The [`DiagonalCosts`] of the machine class of `isa`: NEON's on NEON, x86's elsewhere.
pub fn diagonal_costs(isa: Isa) -> DiagonalCosts {
    match isa {
        Isa::Neon => DIAGONAL_COSTS_NEON,
        _ => DIAGONAL_COSTS_X86,
    }
}

/// Whether the time rule ([`HelmholtzDiagonal::Auto`]) takes a level of degree `p` and V
/// lists `shape` from RCR with the bound's `classes`: the model's diagonal level below RCR's.
/// A function of its arguments alone (constraint 9).
pub fn diagonal_beats_rcr(
    costs: &DiagonalCosts,
    p: usize,
    shape: &DiagonalShape,
    classes: &[usize],
) -> bool {
    !classes.is_empty() && costs.diagonal_level(p, shape, classes) < costs.rcr_level(p, shape)
}

/// Whether the time rule takes a level of degree `p` and V lists `shape` from the exponential
/// M2L at its degree `p_w`, with the bound's `classes`: the model's diagonal level below the
/// waves'. Decided after the plan, on the agreed shape, since the waves cost per box and the
/// diagonal form per pair, so their order turns on the pairs per box (W1's surface about 45,
/// W3's volume about 165). A function of its arguments alone (constraint 9).
pub fn diagonal_beats_plane_waves(
    costs: &DiagonalCosts,
    p: usize,
    p_w: usize,
    shape: &DiagonalShape,
    classes: &[usize],
) -> bool {
    !classes.is_empty()
        && costs.diagonal_level(p, shape, classes) < costs.plane_wave_level(p_w, shape)
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

/// The default build of the diagonal form's transfer functions: interpolated in ŝ · t̂ from a
/// double-double table (Helmholtz CH4.2c2, its question 1, signed off 2026-11-20 on #241; the
/// rule's C_T = 0.23). On W1 at kD = 500 and 1000 it gave the direct build's errors and classes
/// with tables 4.0–6.4× faster to build (fmm-validate/results/helmholtz-ch4-2c2.md).
/// [`TransferBuild::Direct`] stays a named setting and the tests' trusted path.
pub const DEFAULT_DIAGONAL_BUILD: TransferBuild = TransferBuild::Interpolated;

/// The diagonal settings of the operator (module documentation, "The rule").
#[derive(Clone, Debug)]
pub(super) struct DiagonalSettings {
    pub(super) choice: HelmholtzDiagonal,
    /// The transfer functions' build (Helmholtz CH4.2c2), which the rule's constants follow.
    pub(super) build: TransferBuild,
    /// The byte budget of a level's scratch on this rank, or `None` for no bound.
    pub(super) budget: Option<usize>,
    /// Each level's distinct V sources, the largest count over the ranks, if known.
    pub(super) sources: Option<Vec<usize>>,
    /// The chunk width if named, else [`diagonal_chunk_targets`]'s.
    pub(super) chunk: Option<usize>,
    /// The time rule's costs if named (the builder's, agreed over the ranks), else those of the
    /// column kernel's ISA (Helmholtz CH4.3b1).
    pub(super) costs: Option<DiagonalCosts>,
    /// Each level's V lists for the time rule, agreed over the ranks, if known; else the
    /// uniform level's.
    pub(super) shapes: Option<Vec<DiagonalShape>>,
}

impl Default for DiagonalSettings {
    /// Off, [`DEFAULT_DIAGONAL_BUILD`], no budget, no source counts, the chunk width by rule,
    /// the costs of the ISA and no shapes.
    fn default() -> Self {
        Self {
            choice: HelmholtzDiagonal::default(),
            build: DEFAULT_DIAGONAL_BUILD,
            budget: None,
            sources: None,
            chunk: None,
            costs: None,
            shapes: None,
        }
    }
}

impl DiagonalSettings {
    /// Level l's V lists for the time rule: the agreed shape where given (a level past the
    /// shapes has none), else the uniform level's.
    pub(super) fn shape(&self, l: usize) -> DiagonalShape {
        self.shapes
            .as_ref()
            .map_or_else(DiagonalShape::uniform, |s| {
                s.get(l).copied().unwrap_or_default()
            })
    }

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
/// setting's tolerance and build, where some class passes and the scratch fits the budget,
/// and under [`HelmholtzDiagonal::Auto`] where the level is `from_waves[l]` (its exponential
/// M2L given up for the form by [`diagonal_beats_plane_waves`]) or [`diagonal_beats_rcr`] by
/// `costs` and the level's shape; `None` elsewhere. The transfer
/// functions are built by the setting's build on `parts`, one part per (class, ring).
#[allow(clippy::too_many_arguments)]
pub(super) fn select<T: SimdScalar>(
    settings: &DiagonalSettings,
    fmm_tolerance: Option<f64>,
    rcr: &[bool],
    from_waves: &[bool],
    costs: &DiagonalCosts,
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
            let classes = diagonal_rule_classes::<T>(p, k, epsilon, settings.build);
            if classes.is_empty() {
                return None;
            }
            if settings.choice == HelmholtzDiagonal::Auto && !diagonal_auto_in_range(k) {
                return None;
            }
            if settings.choice == HelmholtzDiagonal::Auto
                && !from_waves.get(l).copied().unwrap_or(false)
            {
                let shape = settings.shape(l);
                if !diagonal_beats_rcr(costs, p, &shape, &classes) {
                    return None;
                }
            }
            let level = Level::new(k, 1.0);
            let clock = Instant::now();
            let m2l = DiagonalM2l::build_by(p, k, &classes, settings.build, parts);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The uniform level: 189 V offsets per child, every class present, and each class's
    /// pairs over the 8 children by position: an offset with |d_k| = 3 along an axis lies in
    /// the V list of one child in two along it, one with |d_k| ≤ 2 in that of both.
    #[test]
    fn the_uniform_level() {
        let shape = DiagonalShape::uniform();
        assert_eq!(shape.targets, 8);
        assert_eq!(shape.pairs(), 8 * 189);
        let mut expected = [0u64; CLASS_COUNT];
        for (offset, d) in m2l_offsets().iter().enumerate() {
            let children: u64 = d
                .iter()
                .map(|&x| if x.abs() == 3 { 1 } else { 2 })
                .product();
            expected[offset_class(offset)] += children;
        }
        assert_eq!(shape.class_pairs, expected);
        assert!(shape.class_pairs.iter().all(|&n| n > 0));
        // Class 0, (0, 0, 2), holds the two offsets (0, 0, ±2), each in every child's list.
        assert_eq!(shape.class_pairs[0], 2 * 8);
    }

    /// The model by hand, and its two decisions.
    #[test]
    fn the_model_by_hand() {
        let costs = DiagonalCosts {
            s2f: [1.0, 0.0],
            f2r: [0.0, 1.0],
            pair: [2.0, 0.0],
            rotation: [0.0, 3.0],
            plane_wave: [5.0, 0.0],
        };
        let mut shape = DiagonalShape {
            targets: 10,
            ..DiagonalShape::default()
        };
        shape.class_pairs[0] = 7;
        shape.class_pairs[5] = 3;
        // p = 1: q = 2, q² = 4, q³ = 8.
        let boxes = 10.0 * (4.0 + 8.0);
        assert_eq!(
            costs.diagonal_level(1, &shape, &[0]),
            boxes + 7.0 * 8.0 + 3.0 * 24.0
        );
        assert_eq!(costs.diagonal_level(1, &shape, &[]), boxes + 10.0 * 24.0);
        assert_eq!(costs.rcr_level(1, &shape), 240.0);
        assert_eq!(costs.plane_wave_level(1, &shape), 200.0);
        // 120 + 56 + 72 = 248 against RCR's 240: RCR; with every pair diagonal, 200: diagonal.
        assert!(!diagonal_beats_rcr(&costs, 1, &shape, &[0]));
        assert!(diagonal_beats_rcr(&costs, 1, &shape, &[0, 5]));
        assert!(!diagonal_beats_rcr(&costs, 1, &shape, &[]));
        assert_eq!(
            DiagonalCosts::from_array(costs.to_array()),
            costs,
            "the constants' order"
        );
        // The waves cost per box and the form per pair: the order turns on the pairs per box.
        // 10 boxes at 5 + 1 each and 2 per pair against the waves' 20 per box: 60 + 2n < 200
        // for n < 70 pairs.
        let mut sparse = DiagonalShape {
            targets: 10,
            ..DiagonalShape::default()
        };
        sparse.class_pairs[0] = 60;
        let mut dense = sparse;
        dense.class_pairs[0] = 80;
        let waves = DiagonalCosts {
            s2f: [1.25, 0.0],
            f2r: [0.0, 0.125],
            pair: [0.5, 0.0],
            ..costs
        };
        assert_eq!(waves.plane_wave_level(1, &sparse), 200.0);
        assert!(diagonal_beats_plane_waves(&waves, 1, 1, &sparse, &[0]));
        assert!(!diagonal_beats_plane_waves(&waves, 1, 1, &dense, &[0]));
        assert!(!diagonal_beats_plane_waves(&waves, 1, 1, &sparse, &[]));
    }

    /// The per-class crossover (proposal §10 CH4.3): on both machine classes the diagonal pair
    /// costs less than RCR's at every degree of the high-frequency range, so once a level pays
    /// its conversions every class that passes the bound is faster diagonal, and the time
    /// rule decides per level.
    #[test]
    fn the_pair_against_rcrs_at_every_degree() {
        for (isa, costs) in [
            (Isa::Avx512, DIAGONAL_COSTS_X86),
            (Isa::Neon, DIAGONAL_COSTS_NEON),
        ] {
            assert_eq!(diagonal_costs(isa), costs);
            for p in 4..=256 {
                let ratio =
                    DiagonalCosts::unit(costs.pair, p) / DiagonalCosts::unit(costs.rotation, p);
                assert!(ratio < 1.0, "{isa:?} p = {p}: {ratio}");
            }
        }
        assert_eq!(diagonal_costs(Isa::Avx2), DIAGONAL_COSTS_X86);
        assert_eq!(diagonal_costs(Isa::Scalar), DIAGONAL_COSTS_X86);
    }
}
