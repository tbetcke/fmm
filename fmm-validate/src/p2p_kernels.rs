//! The P2P kernel benchmark of Phase 3S (C3S.6): the core of the `p2p_kernels` example,
//! and the inputs of the green-kernels comparison in `spikes/p2p-simd`.
//!
//! It times `nd_fmm_simd::P2pKernel` on every available ISA against
//! `nd_fmm_ref::p2p::p2p` on the workloads of docs/design/simd-p2p.md §8.2, and
//! measures every kernel against `direct_sum` first:
//!
//! - **W1, FMM-shaped** ([`w1`]): a target leaf of n_t ∈ [`W1_TARGETS`] points uniform
//!   in [−1, 1]³ and the 27 leaves of its 3 × 3 × 3 block, n_t sources each, in
//!   [−3, 3]³. The centre leaf's sources are the targets themselves, so its self pairs
//!   are coincident. A pool of [`POOL`] sets is cycled, so the data come from L2 or L3
//!   as in an FMM. Two [`Form`]s: per-pair, 27 calls of n_t sources (as
//!   `LaplaceOperator` calls the kernel), and gathered, one call of 27 n_t sources.
//! - **W2, all-pairs** ([`w2`]): N ∈ [`W2_POINTS`] sources uniform in [0, 1]³, and
//!   targets either N other uniform points or the sources themselves.
//!
//! Charges are uniform in [−1, 1]. Each cell runs in f32 and f64, with the potential
//! only and with gradients. The inputs are those of the Phase 3S T2 spike: the same
//! generator, seeds and order of draws, rounded to the precision of the run.
//!
//! # Measures (design §8.3)
//!
//! - **Pairs per second**, counting n_s n_t pairs per evaluation, coincident ones
//!   included, as the median of 15 batches of at least 20 ms
//!   ([`bench::median_time_per_call`](crate::bench::median_time_per_call)), one thread.
//! - **Accuracy** ([`Accuracy`]) against `direct_sum` on the same rounded inputs, on the
//!   first [`CHECKED_SETS`] sets of a pool: per target the error relative to the sum of
//!   its term magnitudes, Σ|q|/r for φ and Σ|q|/r² for each component of ∇φ, the
//!   largest over the targets ("max"); and the relative L2 error over all targets
//!   ("L2"). The per-component gradient measure is that of design §3, requirement 2,
//!   and of the T2 spike.
//! - **The check** ([`passes`]): requirement 2 as decided in Phase 3S T5, the largest
//!   error within 1e-6 (f32) or 1e-14 (f64), or within twice the reference's error on
//!   the same inputs where that is larger.
//! - **The model** ([`model`]): pairs per cycle of design §4.6, with the operation
//!   counts of the inverse square roots the T2 spike chose and, on NEON, the divider
//!   limit it measured.
//!
//! Timings are reported, never asserted. Nothing here needs MPI.

use std::time::Instant;

use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_simd::{Isa, P2pKernel, SimdScalar};

use crate::SplitMix64;
use crate::bench::median_time_per_call;

/// The number of sets in a W1 pool.
pub const POOL: usize = 64;

/// The target leaf sizes n_t of W1.
pub const W1_TARGETS: [usize; 6] = [8, 16, 24, 32, 64, 128];

/// The point counts N of W2.
pub const W2_POINTS: [usize; 2] = [1000, 10_000];

/// The seed of W1, as in the T2 spike.
pub const W1_SEED: u64 = 0x00F1_0001;

/// The seed of W2, as in the T2 spike.
pub const W2_SEED: u64 = 0x00F1_0002;

/// The number of sets of a pool that the accuracy is measured on.
pub const CHECKED_SETS: usize = 4;

/// The clock assumed for the Apple M3 Max to turn times into cycles: its P-core
/// maximum, 4.05 GHz, which the T2 spike checked against the latency of a dependent
/// FMA chain.
pub const M3_MAX_GHZ: f64 = 4.05;

/// How the sources of a W1 set are passed to the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Form {
    /// One call per source leaf, 27 calls of n_t sources.
    PerPair,
    /// One call with all 27 n_t sources.
    Gathered,
}

impl Form {
    /// "per-pair" or "gathered".
    pub fn name(self) -> &'static str {
        match self {
            Self::PerPair => "per-pair",
            Self::Gathered => "gathered",
        }
    }
}

/// One workload cell (design §8.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    /// W1 with `targets` points per leaf, in a form.
    W1 {
        /// The leaf size n_t.
        targets: usize,
        /// Per-pair or gathered.
        form: Form,
    },
    /// W2 with `points` sources and targets; `equal` if the targets are the sources.
    W2 {
        /// The number N of sources, and of targets.
        points: usize,
        /// Whether the targets are the sources themselves.
        equal: bool,
    },
}

impl Cell {
    /// Every cell: W1 at each n_t of [`W1_TARGETS`], per-pair then gathered, then W2 at
    /// each N of [`W2_POINTS`], targets distinct then equal.
    pub fn all() -> Vec<Cell> {
        let w1 = W1_TARGETS.iter().flat_map(|&targets| {
            [Form::PerPair, Form::Gathered].map(|form| Cell::W1 { targets, form })
        });
        let w2 = W2_POINTS
            .iter()
            .flat_map(|&points| [false, true].map(|equal| Cell::W2 { points, equal }));
        w1.chain(w2).collect()
    }

    /// The reduced set of `--quick`: W1 at n_t = 24 in both forms and W2 at N = 1,000
    /// with targets equal to the sources.
    pub fn quick() -> Vec<Cell> {
        vec![
            Cell::W1 {
                targets: 24,
                form: Form::PerPair,
            },
            Cell::W1 {
                targets: 24,
                form: Form::Gathered,
            },
            Cell::W2 {
                points: 1000,
                equal: true,
            },
        ]
    }

    /// The label of the reports, e.g. "W1 n_t=24 per-pair" or "W2 N=1000 t=s".
    pub fn label(self) -> String {
        match self {
            Cell::W1 { targets, form } => format!("W1 n_t={targets} {}", form.name()),
            Cell::W2 { points, equal } => {
                format!("W2 N={points} {}", if equal { "t=s" } else { "t≠s" })
            }
        }
    }

    /// The form the kernel is called in: gathered for W2.
    pub fn form(self) -> Form {
        match self {
            Cell::W1 { form, .. } => form,
            Cell::W2 { .. } => Form::Gathered,
        }
    }

    /// Whether the cell is FMM-shaped (W1).
    pub fn is_w1(self) -> bool {
        matches!(self, Cell::W1 { .. })
    }

    /// The inputs of the cell in precision `T`: the W1 pool of its n_t, or the one W2
    /// set.
    pub fn pool<T: RealScalar>(self) -> Vec<Set<T>> {
        match self {
            Cell::W1 { targets, .. } => w1(targets, W1_SEED),
            Cell::W2 { points, equal } => vec![w2(points, equal, W2_SEED)],
        }
    }
}

/// One source leaf of a [`Set`]: its sources and their charges.
#[derive(Clone, Debug, PartialEq)]
pub struct Leaf<T> {
    /// The source positions.
    pub sources: Vec<[T; 3]>,
    /// Their charges.
    pub charges: Vec<T>,
}

/// The inputs of one evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct Set<T> {
    /// The targets.
    pub targets: Vec<[T; 3]>,
    /// The source leaves of the per-pair form; W2 has one.
    pub leaves: Vec<Leaf<T>>,
    /// Every source, leaf after leaf: the gathered form.
    pub sources: Vec<[T; 3]>,
    /// Their charges.
    pub charges: Vec<T>,
}

impl<T: RealScalar> Set<T> {
    /// The set of `targets` and `leaves`, with the gathered sources.
    fn new(targets: Vec<[T; 3]>, leaves: Vec<Leaf<T>>) -> Self {
        let sources = leaves.iter().flat_map(|l| l.sources.clone()).collect();
        let charges = leaves.iter().flat_map(|l| l.charges.clone()).collect();
        Self {
            targets,
            leaves,
            sources,
            charges,
        }
    }

    /// The source–target pairs of one evaluation, n_s n_t, coincident ones included.
    pub fn pairs(&self) -> usize {
        self.targets.len() * self.sources.len()
    }
}

/// A point uniform in the cube `centre` + [−h, h]³, rounded to `T`.
fn point<T: RealScalar>(rng: &mut SplitMix64, centre: [f64; 3], h: f64) -> [T; 3] {
    centre.map(|c| T::from_f64(c + rng.range(-h, h)))
}

/// The W1 pool of leaf size `targets`: [`POOL`] sets drawn from `seed` (module
/// documentation).
pub fn w1<T: RealScalar>(targets: usize, seed: u64) -> Vec<Set<T>> {
    let mut rng = SplitMix64::new(seed ^ ((targets as u64) << 32));
    (0..POOL)
        .map(|_| {
            let target_points: Vec<[T; 3]> = (0..targets)
                .map(|_| point(&mut rng, [0.0; 3], 1.0))
                .collect();
            let mut leaves = Vec::with_capacity(27);
            for a in -1..=1 {
                for b in -1..=1 {
                    for c in -1..=1 {
                        let centre = [a, b, c].map(|k: i32| 2.0 * f64::from(k));
                        let sources = if (a, b, c) == (0, 0, 0) {
                            target_points.clone()
                        } else {
                            (0..targets).map(|_| point(&mut rng, centre, 1.0)).collect()
                        };
                        let charges = (0..targets)
                            .map(|_| T::from_f64(rng.range(-1.0, 1.0)))
                            .collect();
                        leaves.push(Leaf { sources, charges });
                    }
                }
            }
            Set::new(target_points, leaves)
        })
        .collect()
}

/// The W2 set of `points` sources in [0, 1]³ drawn from `seed`, with targets equal to
/// the sources if `equal`, else `points` other uniform points.
pub fn w2<T: RealScalar>(points: usize, equal: bool, seed: u64) -> Set<T> {
    let mut rng = SplitMix64::new(seed ^ ((points as u64) << 32));
    let sources: Vec<[T; 3]> = (0..points)
        .map(|_| point(&mut rng, [0.5; 3], 0.5))
        .collect();
    let charges = (0..points)
        .map(|_| T::from_f64(rng.range(-1.0, 1.0)))
        .collect();
    let targets = if equal {
        sources.clone()
    } else {
        (0..points)
            .map(|_| point(&mut rng, [0.5; 3], 0.5))
            .collect()
    };
    Set::new(targets, vec![Leaf { sources, charges }])
}

/// The outputs of one set: a potential and a gradient per target.
#[derive(Clone, Debug, PartialEq)]
pub struct Outputs<T> {
    /// φ per target.
    pub potential: Vec<T>,
    /// ∇φ per target.
    pub gradient: Vec<[T; 3]>,
}

impl<T: RealScalar> Outputs<T> {
    /// Zeroed outputs for `targets` targets.
    pub fn new(targets: usize) -> Self {
        Self {
            potential: vec![T::zero(); targets],
            gradient: vec![[T::zero(); 3]; targets],
        }
    }

    /// Zeroed outputs for every set of `pool`.
    pub fn for_pool(pool: &[Set<T>]) -> Vec<Self> {
        pool.iter().map(|s| Self::new(s.targets.len())).collect()
    }

    /// Sets every output to zero.
    pub fn clear(&mut self) {
        self.potential.fill(T::zero());
        self.gradient.fill([T::zero(); 3]);
    }
}

/// A P2P kernel under test: the reference or `nd-fmm-simd` on one ISA.
#[derive(Clone, Copy, Debug)]
pub enum Kernel<T: SimdScalar> {
    /// `nd_fmm_ref::p2p::p2p`, the P2P of Phase 3.
    Reference,
    /// `nd_fmm_simd::P2pKernel` on its ISA.
    Simd(P2pKernel<T>),
}

impl<T: SimdScalar> Kernel<T> {
    /// The reference, then `nd-fmm-simd` on every available ISA ([`Isa::available`]).
    pub fn all() -> Vec<Self> {
        std::iter::once(Self::Reference)
            .chain(Isa::available().map(|isa| {
                Self::Simd(P2pKernel::new(isa).expect("`Isa::available` lists available ISAs"))
            }))
            .collect()
    }

    /// The ISA of an `nd-fmm-simd` kernel, `None` for the reference.
    pub fn isa(&self) -> Option<Isa> {
        match self {
            Self::Reference => None,
            Self::Simd(kernel) => Some(kernel.isa()),
        }
    }

    /// "reference" or the ISA's name.
    pub fn name(&self) -> String {
        self.isa()
            .map_or_else(|| "reference".to_string(), |isa| isa.to_string())
    }

    /// One call of the kernel, adding into `potential` and, if given, `gradient`.
    pub fn call(
        &self,
        sources: &[[T; 3]],
        charges: &[T],
        targets: &[[T; 3]],
        potential: &mut [T],
        gradient: Option<&mut [[T; 3]]>,
    ) {
        match self {
            Self::Reference => nd_fmm_ref::p2p::p2p(sources, charges, targets, potential, gradient),
            Self::Simd(kernel) => kernel.evaluate(sources, charges, targets, potential, gradient),
        }
    }

    /// One evaluation of `set` in `form`, with gradients if `gradients`, adding into
    /// `out`.
    pub fn evaluate(&self, set: &Set<T>, form: Form, gradients: bool, out: &mut Outputs<T>) {
        match form {
            Form::Gathered => self.call(
                &set.sources,
                &set.charges,
                &set.targets,
                &mut out.potential,
                gradients.then_some(&mut out.gradient[..]),
            ),
            Form::PerPair => {
                for leaf in &set.leaves {
                    self.call(
                        &leaf.sources,
                        &leaf.charges,
                        &set.targets,
                        &mut out.potential,
                        gradients.then_some(&mut out.gradient[..]),
                    );
                }
            }
        }
    }
}

/// "f32" or "f64".
pub fn precision_name<T: RealScalar>() -> &'static str {
    if size_of::<T>() == 4 { "f32" } else { "f64" }
}

/// The f64 oracle of one set: `direct_sum` on its rounded inputs, and the sums of the
/// term magnitudes per target.
#[derive(Clone, Debug, PartialEq)]
pub struct Oracle {
    /// φ per target.
    pub potential: Vec<f64>,
    /// ∇φ per target.
    pub gradient: Vec<[f64; 3]>,
    /// Σ|q|/r per target, over the pairs with r² ≠ 0.
    pub potential_magnitude: Vec<f64>,
    /// Σ|q|/r² per target, over the pairs with r² ≠ 0.
    pub gradient_magnitude: Vec<f64>,
}

impl Oracle {
    /// The oracle of `set`, in f64 on its inputs as rounded to `T`.
    pub fn new<T: RealScalar>(set: &Set<T>) -> Self {
        let up = |v: &[[T; 3]]| -> Vec<[f64; 3]> {
            v.iter().map(|p| p.map(RealScalar::to_f64)).collect()
        };
        let sources = up(&set.sources);
        let targets = up(&set.targets);
        let charges: Vec<f64> = set.charges.iter().map(|&q| RealScalar::to_f64(q)).collect();
        let n = targets.len();
        let mut potential = vec![0.0; n];
        let mut gradient = vec![[0.0; 3]; n];
        direct_sum(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let mut potential_magnitude = vec![0.0; n];
        let mut gradient_magnitude = vec![0.0; n];
        for (i, x) in targets.iter().enumerate() {
            for (y, q) in sources.iter().zip(&charges) {
                let r2: f64 = (0..3).map(|k| (x[k] - y[k]).powi(2)).sum();
                if r2 > 0.0 {
                    potential_magnitude[i] += q.abs() / r2.sqrt();
                    gradient_magnitude[i] += q.abs() / r2;
                }
            }
        }
        Self {
            potential,
            gradient,
            potential_magnitude,
            gradient_magnitude,
        }
    }
}

/// The accuracy of one output against its [`Oracle`] (module documentation,
/// "Measures"). The gradient fields are 0 for an output without gradients.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Accuracy {
    /// The largest error of φ relative to Σ|q|/r of its target.
    pub max_potential: f64,
    /// The relative L2 error of φ over all targets.
    pub l2_potential: f64,
    /// The largest error of a component of ∇φ relative to Σ|q|/r² of its target.
    pub max_gradient: f64,
    /// The relative L2 error of ∇φ over all targets and components.
    pub l2_gradient: f64,
    /// Whether any output is not finite.
    pub non_finite: bool,
}

impl Accuracy {
    /// The worse of two measurements, field by field.
    pub fn worst(self, other: Accuracy) -> Accuracy {
        Accuracy {
            max_potential: self.max_potential.max(other.max_potential),
            l2_potential: self.l2_potential.max(other.l2_potential),
            max_gradient: self.max_gradient.max(other.max_gradient),
            l2_gradient: self.l2_gradient.max(other.l2_gradient),
            non_finite: self.non_finite || other.non_finite,
        }
    }

    /// The largest error relative to the term magnitudes, of φ and ∇φ.
    pub fn max(self) -> f64 {
        self.max_potential.max(self.max_gradient)
    }
}

/// The accuracy of `potential` and, if given, `gradient` against `oracle`.
///
/// # Panics
///
/// If the outputs and the oracle differ in length.
pub fn accuracy<T: RealScalar>(
    oracle: &Oracle,
    potential: &[T],
    gradient: Option<&[[T; 3]]>,
) -> Accuracy {
    assert_eq!(potential.len(), oracle.potential.len(), "one φ per target");
    let mut a = Accuracy::default();
    let (mut error, mut size) = (0.0, 0.0);
    for (i, &phi) in potential.iter().enumerate() {
        let v = RealScalar::to_f64(phi);
        a.non_finite |= !v.is_finite();
        let e = (v - oracle.potential[i]).abs();
        if oracle.potential_magnitude[i] > 0.0 {
            a.max_potential = a.max_potential.max(e / oracle.potential_magnitude[i]);
        }
        error += e * e;
        size += oracle.potential[i] * oracle.potential[i];
    }
    a.l2_potential = (error / size).sqrt();
    if let Some(gradient) = gradient {
        assert_eq!(gradient.len(), oracle.gradient.len(), "one ∇φ per target");
        let (mut error, mut size) = (0.0, 0.0);
        for (i, g) in gradient.iter().enumerate() {
            for (k, &gk) in g.iter().enumerate() {
                let v = RealScalar::to_f64(gk);
                a.non_finite |= !v.is_finite();
                let e = (v - oracle.gradient[i][k]).abs();
                if oracle.gradient_magnitude[i] > 0.0 {
                    a.max_gradient = a.max_gradient.max(e / oracle.gradient_magnitude[i]);
                }
                error += e * e;
                size += oracle.gradient[i][k] * oracle.gradient[i][k];
            }
        }
        a.l2_gradient = (error / size).sqrt();
    }
    a.non_finite |= a.max().is_nan();
    a
}

/// The sum tolerance of design §3, requirement 2: 1e-6 in f32 and 1e-14 in f64,
/// relative to the term magnitudes.
pub fn sum_tolerance<T: RealScalar>() -> f64 {
    if size_of::<T>() == 4 { 1e-6 } else { 1e-14 }
}

/// Whether `accuracy` passes requirement 2 as decided in Phase 3S T5: finite, and the
/// largest error within [`sum_tolerance`] or within twice `reference`'s, the error of
/// `nd_fmm_ref::p2p` on the same inputs, where that is larger.
pub fn passes<T: RealScalar>(accuracy: Accuracy, reference: Accuracy) -> bool {
    !accuracy.non_finite && accuracy.max() <= sum_tolerance::<T>().max(2.0 * reference.max())
}

/// The oracles of the first [`CHECKED_SETS`] sets of `pool`.
pub fn oracles<T: RealScalar>(pool: &[Set<T>]) -> Vec<Oracle> {
    pool.iter().take(CHECKED_SETS).map(Oracle::new).collect()
}

/// The worst accuracy of `kernel` over the sets that `oracles` belong to (the first of
/// `pool`), evaluated from zeroed outputs in `form`.
pub fn check<T: SimdScalar>(
    kernel: &Kernel<T>,
    pool: &[Set<T>],
    oracles: &[Oracle],
    form: Form,
    gradients: bool,
) -> Accuracy {
    pool.iter()
        .zip(oracles)
        .map(|(set, oracle)| {
            let mut out = Outputs::new(set.targets.len());
            kernel.evaluate(set, form, gradients, &mut out);
            accuracy(
                oracle,
                &out.potential,
                gradients.then_some(&out.gradient[..]),
            )
        })
        .fold(Accuracy::default(), Accuracy::worst)
}

/// Whether the per-pair calls of `kernel` on `set` give the bits of one gathered call
/// (design §3, requirement 4, chunk invariance).
pub fn per_pair_equals_gathered<T: SimdScalar>(
    kernel: &Kernel<T>,
    set: &Set<T>,
    gradients: bool,
) -> bool {
    let mut per_pair = Outputs::new(set.targets.len());
    let mut gathered = Outputs::new(set.targets.len());
    kernel.evaluate(set, Form::PerPair, gradients, &mut per_pair);
    kernel.evaluate(set, Form::Gathered, gradients, &mut gathered);
    let bits =
        |v: &[T]| -> Vec<u64> { v.iter().map(|&x| RealScalar::to_f64(x).to_bits()).collect() };
    bits(&per_pair.potential) == bits(&gathered.potential)
        && bits(per_pair.gradient.as_flattened()) == bits(gathered.gradient.as_flattened())
}

/// Pairs per second of `evaluate(i)`, one evaluation of set `i` of a pool of `len`
/// sets with `pairs` pairs each, cycling through the pool: the median of
/// [`BATCHES`](crate::bench::BATCHES) batches
/// ([`median_time_per_call`]).
pub fn pairs_per_second(pairs: usize, len: usize, mut evaluate: impl FnMut(usize)) -> f64 {
    let time = median_time_per_call(|calls| {
        let start = Instant::now();
        for c in 0..calls {
            evaluate(c % len);
        }
        start.elapsed()
    });
    pairs as f64 / time
}

/// Pairs per second of `kernel` on `pool` in `form` ([`pairs_per_second`]), adding into
/// `outputs`, one per set.
pub fn time_kernel<T: SimdScalar>(
    kernel: &Kernel<T>,
    pool: &[Set<T>],
    outputs: &mut [Outputs<T>],
    form: Form,
    gradients: bool,
) -> f64 {
    pairs_per_second(pool[0].pairs(), pool.len(), |i| {
        kernel.evaluate(&pool[i], form, gradients, &mut outputs[i]);
    })
}

/// The throughput model of design §4.6 for one ISA, precision and output, as corrected
/// by the T2 spike (spikes/p2p-simd/SPIKE_REPORT.md, "Fraction of the design §4.6
/// model").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    /// Vector FP operations per pair and lane (design §4.2 with the inverse square root
    /// of the kernel).
    pub ops: usize,
    /// The bound in pairs per cycle.
    pub pairs_per_cycle: f64,
}

/// The model of `isa` in precision `T`, with or without gradients; `None` for
/// [`Isa::Scalar`], which has none.
///
/// - Operations per pair and lane: 9 (φ) or 15 (φ, ∇φ) for the difference, r², the
///   mask and the accumulation (design §4.2), plus the inverse square root: FSQRT and
///   FDIV on NEON (2), `vrsqrtps` with the degree-2 correction in f32 (6) and the
///   degree-5 correction via f32 in f64 (11, with two conversions) on AVX2.
/// - NEON: four FP pipes, and the divider, which runs FSQRT and FDIV on a unit beside
///   them at about 3 cycles per vector (measured on the M3 Max by the spike). The bound
///   is W / max(ops / 4, 3) pairs per cycle.
/// - AVX2: two FMA pipes, 2 W / ops pairs per cycle; a model only, since no x86_64
///   machine times anything in Phase 3S.
pub fn model<T: SimdScalar>(isa: Isa, gradients: bool) -> Option<Model> {
    let base = if gradients { 15 } else { 9 };
    let lanes = isa.lanes::<T>() as f64;
    let f32 = size_of::<T>() == 4;
    match isa {
        Isa::Neon => {
            let ops = base + 2;
            let cycles = (ops as f64 / 4.0).max(3.0);
            Some(Model {
                ops,
                pairs_per_cycle: lanes / cycles,
            })
        }
        Isa::Avx2 => {
            let ops = base + if f32 { 6 } else { 11 };
            Some(Model {
                ops,
                pairs_per_cycle: 2.0 * lanes / ops as f64,
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workloads_have_the_shapes_of_design_8_2() {
        let pool = w1::<f64>(8, W1_SEED);
        assert_eq!(pool.len(), POOL);
        let set = &pool[0];
        assert_eq!(set.leaves.len(), 27);
        assert_eq!(set.sources.len(), 27 * 8);
        assert_eq!(set.pairs(), 27 * 64);
        // The centre leaf holds the targets themselves.
        assert_eq!(set.leaves[13].sources, set.targets);
        assert!(set.targets.iter().flatten().all(|c| c.abs() <= 1.0));
        assert!(set.sources.iter().flatten().all(|c| c.abs() <= 3.0));
        let equal = w2::<f32>(100, true, W2_SEED);
        assert_eq!(equal.targets, equal.sources);
        let distinct = w2::<f32>(100, false, W2_SEED);
        assert_eq!(distinct.sources, equal.sources);
        assert_ne!(distinct.targets, distinct.sources);
        assert_eq!(
            Cell::all().len(),
            2 * W1_TARGETS.len() + 2 * W2_POINTS.len()
        );
        assert_eq!(Cell::quick()[1].label(), "W1 n_t=24 gathered");
    }

    #[test]
    fn w1_reproduces_the_spike_inputs() {
        // The first draw of the spike's W1 pool at n_t = 8 (spikes/p2p-simd,
        // `workloads::w1`): SplitMix64 from W1_SEED ^ (8 << 32), the first target's x.
        let mut rng = SplitMix64::new(W1_SEED ^ (8 << 32));
        let x = rng.range(-1.0, 1.0);
        assert_eq!(w1::<f64>(8, W1_SEED)[0].targets[0][0], x);
    }

    #[test]
    fn accuracy_is_relative_to_the_term_magnitudes() {
        // Two opposite charges at distance 1 from the target: φ = 0, Σ|q|/r = 2.
        let set = Set::new(
            vec![[0.0; 3]],
            vec![Leaf {
                sources: vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                charges: vec![1.0, -1.0],
            }],
        );
        let oracle = Oracle::new(&set);
        assert_eq!(oracle.potential_magnitude, [2.0]);
        assert_eq!(oracle.gradient_magnitude, [2.0]);
        let a = accuracy(
            &oracle,
            &[1e-16],
            Some(&[[oracle.gradient[0][0] + 2f64.powi(-51), 0.0, 0.0]]),
        );
        assert_eq!(a.max_potential, 5e-17);
        // ∂x φ = −2 exactly; 2⁻⁵¹ off, relative to Σ|q|/r² = 2.
        assert_eq!(a.max_gradient, 2f64.powi(-52));
        assert!(!a.non_finite);
        assert!(passes::<f64>(a, Accuracy::default()));
        let bad = Accuracy {
            max_gradient: 3e-14,
            ..a
        };
        assert!(!passes::<f64>(bad, Accuracy::default()));
        // Twice the reference's error passes where it exceeds the tolerance.
        let reference = Accuracy {
            max_gradient: 2e-14,
            ..a
        };
        assert!(passes::<f64>(bad, reference));
        assert!(!passes::<f64>(
            Accuracy {
                non_finite: true,
                ..a
            },
            reference
        ));
    }

    #[test]
    fn models_follow_the_operation_counts() {
        let neon = model::<f32>(Isa::Neon, true).unwrap();
        assert_eq!(neon.ops, 17);
        assert!((neon.pairs_per_cycle - 16.0 / 17.0).abs() < 1e-15);
        // Potential only, the divider bounds NEON: 4 lanes per 3 cycles.
        assert!(
            (model::<f32>(Isa::Neon, false).unwrap().pairs_per_cycle - 4.0 / 3.0).abs() < 1e-15
        );
        assert_eq!(model::<f64>(Isa::Avx2, true).unwrap().ops, 26);
        assert_eq!(model::<f32>(Isa::Avx2, false).unwrap().ops, 15);
        assert!(model::<f64>(Isa::Scalar, false).is_none());
    }
}
