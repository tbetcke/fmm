//! The Helmholtz FMM against the sampled direct sum on the reference workloads
//! (Helmholtz CH3.1c; docs/design/helmholtz-fmm-proposal.md §9.1 and §10, CH3.1), on any
//! number of ranks.
//!
//! A [`Case`] names a workload of [`helmholtz_workloads`](crate::helmholtz_workloads), k,
//! the degrees and the charges; [`Problem`] draws the points, the sampled targets and the
//! charge vectors; [`Oracle`] is the direct sum in f64 at the sampled targets
//! (`nd_fmm_ref::helmholtz::p2p::direct_sum`, scaled by 1/(4π) as the FMM's output is); [`run`] builds the
//! [`HelmholtzFmm`] at each thread count, evaluates it and measures it:
//!
//! - **Points.** Sources equal targets. Rank r of P passes the contiguous block
//!   [`Problem::block`] of the workload's points; the FMM moves them to the ranks that own
//!   their leaves.
//! - **Charges.** §9.1: per vector, parts uniform in [−1, 1), drawn from the generator of
//!   `seed + 1 + v`, Re q at every point, then Im q (complex charges only).
//! - **Errors.** §9.1's norms over the sampled targets: the relative ℓ2 error of the
//!   complex potential, ‖φ − φ_ref‖₂ / ‖φ_ref‖₂, and the same over the 3-vector gradient
//!   field, per vector and over all vectors together (the sums of squares added), reduced
//!   over the ranks.
//! - **Checksum.** A rank-invariant checksum of the whole output ([`checksum`]): the
//!   wrapping sum over every output value of a mix of its global point index, its output
//!   slot and its bits, summed over the ranks. Equal outputs give equal checksums on any
//!   number of ranks. One changed value always changes it (the mix is a bijection);
//!   several changed values leave it equal with probability about 2⁻⁶⁴.
//!   Determinism (proposal §2, constraint 2) is shown by equal checksums across thread
//!   counts and rank counts.
//! - **Times.** The build and the evaluation, wall time of the slowest rank.
//! - **P2P kernel.** [`Case::p2p`] (Helmholtz CH3.4c): `Auto` by default, `Reference` for
//!   the times before the vectorised kernel; [`Row::p2p`] the kernel that ran.
//! - **M2L.** [`Case::m2l`] and [`Case::dense_budget`] (Helmholtz CH3.2a, CH3.2b, CH3.2c):
//!   Rotation by default, or Dense, Classes or Batched on the levels whose tables fit the
//!   budget, Batched in [`Case::m2l_mul_add`]'s mode (separate by default);
//!   [`Row::m2l`], [`Row::dense_levels`] and [`Row::dense_bytes`] what ran. Auto
//!   (Helmholtz CH3.2d), the rule per level, at the rule's crossover or at
//!   [`Case::m2l_max_degree`] (fused by default); [`Row::m2l_max_degree`] the agreed one.
//!   The exponential M2L (Helmholtz CH6.1e), each level's quadrature by
//!   [`Case::plane_wave`] (the rule by default, or named per level); [`Row::plane_wave`] the
//!   precision each level ran. [`Case::plane_wave_reach`] (Helmholtz CH6.2e) the lateral
//!   reach of the quadratures' α counts (the corner's by default); [`Row::plane_wave_reach`]
//!   the one built and [`Row::plane_wave_waves`] each level's waves per direction.
//!   [`Case::plane_wave_split`] (Helmholtz CH6.2f) how merge and shift splits over the
//!   threads (the library's `Auto` by default); [`Row::plane_wave_split`] the setting and
//!   [`Row::plane_wave_parts`] its thread-seconds by part per level. Auto with the
//!   exponential M2L (Helmholtz CH6.2h, D8): [`Case::auto_plane_wave`], at the rule's
//!   smallest degree or [`Case::m2l_plane_wave_min_degree`]; [`Row::m2l_plane_wave_min_degree`]
//!   the agreed one and [`Row::plane_wave`] the levels with waves.

use std::ops::Range;
use std::thread;
use std::time::{Duration, Instant};

use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;
use nd_fmm_exec::geometry::{Domain, radius};
use nd_fmm_exec::helmholtz::{
    ChargeKind, Charges, DEFAULT_DENSE_BUDGET, Degrees, HelmholtzFmm, HelmholtzFmmBuilder,
    HelmholtzFmmError, HelmholtzM2l, Output, PlaneWaveCounts, PlaneWavePrecision, PlaneWaveSplit,
    PlaneWaveTimes, Wavenumber,
};
use nd_fmm_exec::operator::{MulAdd, P2pChoice};
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p::direct_sum;
use nd_fmm_tables::helmholtz::plane_wave::Precision;

use crate::SplitMix64;
use crate::helmholtz_workloads::{SEED, Workload};
use crate::points;

/// One accuracy run: a workload at one k with the degrees, charges and thread counts.
#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    /// The workload (D = 1, so kD is k).
    pub workload: Workload,
    /// k as (Re k, Im k).
    pub k: (f64, f64),
    /// The degrees of the FMM's levels.
    pub degrees: Degrees,
    /// Real or complex charges.
    pub charges: ChargeKind,
    /// The number of charge vectors.
    pub vectors: usize,
    /// Whether the FMM computes gradients (and the errors include ∇φ).
    pub gradients: bool,
    /// The number of sampled targets (at most N).
    pub samples: usize,
    /// The seed of the samples and charges.
    pub seed: u64,
    /// The FMM's P2P kernel.
    pub p2p: P2pChoice,
    /// The FMM's M2L.
    pub m2l: HelmholtzM2l,
    /// The bytes of the dense, class or batched M2L tables per rank under
    /// [`HelmholtzM2l::Dense`], [`HelmholtzM2l::Classes`] or [`HelmholtzM2l::Batched`].
    pub dense_budget: usize,
    /// The batched M2L's multiply–add mode (read under [`HelmholtzM2l::Batched`] and
    /// [`HelmholtzM2l::Auto`]; `None` the library's default for the setting).
    pub m2l_mul_add: Option<MulAdd>,
    /// The crossover degree of [`HelmholtzM2l::Auto`] (`None` the rule's).
    pub m2l_max_degree: Option<usize>,
    /// Whether [`HelmholtzM2l::Auto`] takes the exponential M2L
    /// (`HelmholtzFmmBuilder::auto_plane_wave`, Helmholtz CH6.2h; `true` in [`Case::new`], the
    /// library's default since D8); named always, so that the earlier reports' `auto` runs
    /// repeat whatever the library's default.
    pub auto_plane_wave: bool,
    /// The smallest degree at which Auto takes the exponential M2L (`None` the rule's).
    pub m2l_plane_wave_min_degree: Option<usize>,
    /// The precision of each level's quadrature under [`HelmholtzM2l::PlaneWave`]
    /// (Helmholtz CH6.1e).
    pub plane_wave: PlaneWavePrecision,
    /// The bytes per node of its waves buffer (`HelmholtzFmmBuilder::plane_wave_budget`,
    /// Helmholtz CH6.2a; `None` the library's default, `usize::MAX` a whole level).
    pub plane_wave_budget: Option<usize>,
    /// The lateral reach of the quadratures' α counts under [`HelmholtzM2l::PlaneWave`]
    /// (`HelmholtzFmmBuilder::plane_wave_reach`, Helmholtz CH6.2e; `None` the library's, the
    /// region's corner 4√2).
    pub plane_wave_reach: Option<f64>,
    /// How merge and shift splits over the threads under [`HelmholtzM2l::PlaneWave`]
    /// (`HelmholtzFmmBuilder::plane_wave_split`, Helmholtz CH6.2f; `None` the library's,
    /// `Auto`). No bit depends on it.
    pub plane_wave_split: Option<PlaneWaveSplit>,
    /// The evaluations of one build (at least 1; Helmholtz CH6.2b): the row's times and
    /// errors are the last one's.
    pub evaluations: usize,
}

impl Case {
    /// A case with complex charges, one vector, gradients and 1,000 samples (§9.1), seeded
    /// with the workloads' [`SEED`], the P2P kernel of [`P2pChoice::Auto`] and M2L by
    /// [`HelmholtzM2l::Rotation`] (the dense budget [`DEFAULT_DENSE_BUDGET`]; under
    /// [`HelmholtzM2l::PlaneWave`] the precisions of [`PlaneWavePrecision::Rule`]).
    pub fn new(workload: Workload, k: (f64, f64), degrees: Degrees) -> Self {
        Self {
            workload,
            k,
            degrees,
            charges: ChargeKind::Complex,
            vectors: 1,
            gradients: true,
            samples: 1000,
            seed: SEED,
            p2p: P2pChoice::Auto,
            m2l: HelmholtzM2l::Rotation,
            dense_budget: DEFAULT_DENSE_BUDGET,
            m2l_mul_add: None,
            m2l_max_degree: None,
            auto_plane_wave: true,
            m2l_plane_wave_min_degree: None,
            plane_wave: PlaneWavePrecision::Rule,
            plane_wave_budget: None,
            plane_wave_reach: None,
            plane_wave_split: None,
            evaluations: 1,
        }
    }
}

/// The points, sampled targets and charge vectors of a [`Case`].
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    /// The workload's points.
    pub points: Vec<[f64; 3]>,
    /// The positions in `points` of the sampled targets, distinct.
    pub sample: Vec<usize>,
    /// Per vector, Re q at every point.
    pub re: Vec<Vec<f64>>,
    /// Per vector, Im q at every point (empty vectors for real charges).
    pub im: Vec<Vec<f64>>,
}

impl Problem {
    /// Draws the problem: the workload's points, the sample (a partial Fisher–Yates
    /// shuffle from the generator of `case.seed`, at most N targets) and charge vector v
    /// from the generator of `case.seed + 1 + v`.
    pub fn new(case: &Case) -> Self {
        let points = case.workload.points(case.k.0);
        let n = points.len();
        let mut rng = SplitMix64::new(case.seed);
        let mut order: Vec<usize> = (0..n).collect();
        let samples = case.samples.min(n);
        for i in 0..samples {
            let j = i + ((((rng.next_u64() >> 32) * (n - i) as u64) >> 32) as usize);
            order.swap(i, j);
        }
        order.truncate(samples);
        let (mut re, mut im) = (Vec::new(), Vec::new());
        for v in 0..case.vectors as u64 {
            let mut rng = SplitMix64::new(case.seed + 1 + v);
            re.push(points::charges(&mut rng, n));
            im.push(match case.charges {
                ChargeKind::Complex => points::charges(&mut rng, n),
                ChargeKind::Real => Vec::new(),
            });
        }
        Self {
            points,
            sample: order,
            re,
            im,
        }
    }

    /// The number of points N.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the problem has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The positions of the points that rank `rank` of `size` passes: a contiguous block,
    /// the first N mod P ranks one point more.
    pub fn block(&self, rank: usize, size: usize) -> Range<usize> {
        let (n, size) = (self.len(), size.max(1));
        let (base, extra) = (n / size, n % size);
        let start = rank * base + rank.min(extra);
        start..start + base + usize::from(rank < extra)
    }

    /// Charge vector v as complex numbers in f64 (Im q = 0 for real charges).
    pub fn complex_charges(&self, v: usize) -> Vec<Complex<f64>> {
        let im = &self.im[v];
        self.re[v]
            .iter()
            .enumerate()
            .map(|(i, &re)| Complex::new(re, im.get(i).copied().unwrap_or(0.0)))
            .collect()
    }
}

/// The exact potentials and gradients at the sampled targets, per vector, in f64, with
/// 1/(4π).
#[derive(Clone, Debug, PartialEq)]
pub struct Oracle {
    /// φ per vector and sampled target.
    pub potential: Vec<Vec<Complex<f64>>>,
    /// ∇φ per vector and sampled target (empty without gradients).
    pub gradient: Vec<Vec<[Complex<f64>; 3]>>,
}

impl Oracle {
    /// The oracle of `problem` at k, at the sampled targets with positions `targets` in
    /// `problem.sample`, on `threads` threads (each a contiguous share of the targets, so
    /// the values do not depend on the thread count). O(m N n_targets).
    pub fn of_samples(
        k: Complex<f64>,
        problem: &Problem,
        targets: &[usize],
        gradients: bool,
        threads: usize,
    ) -> Self {
        let points: Vec<[f64; 3]> = targets
            .iter()
            .map(|&s| problem.points[problem.sample[s]])
            .collect();
        let chunk = points.len().div_ceil(threads.max(1)).max(1);
        let (mut potential, mut gradient) = (Vec::new(), Vec::new());
        for v in 0..problem.re.len() {
            let charges = problem.complex_charges(v);
            let mut phi = vec![Complex::new(0.0, 0.0); points.len()];
            let mut grad =
                vec![[Complex::new(0.0, 0.0); 3]; if gradients { points.len() } else { 0 }];
            thread::scope(|scope| {
                let mut grad_chunks = grad.chunks_mut(chunk);
                for (x, phi) in points.chunks(chunk).zip(phi.chunks_mut(chunk)) {
                    let grad = grad_chunks.next();
                    let (sources, charges) = (&problem.points, &charges);
                    scope.spawn(move || direct_sum(k, sources, charges, x, phi, grad));
                }
            });
            // The direct sum leaves out 1/(4π); the FMM's output includes it (§3.15).
            let scale = 1.0 / (4.0 * std::f64::consts::PI);
            potential.push(phi.into_iter().map(|v| v * scale).collect());
            gradient.push(grad.into_iter().map(|g| g.map(|v| v * scale)).collect());
        }
        Self {
            potential,
            gradient,
        }
    }

    /// [`of_samples`](Self::of_samples) of every sampled target, computed once over the
    /// ranks of `comm`: each rank every P-th sample from its rank, and all-reduces (sums,
    /// zeros elsewhere) give every rank the whole oracle, the values of one rank bit for
    /// bit.
    ///
    /// # Collective operation
    ///
    /// On `comm`, every rank with the same arguments.
    pub fn sharded<C: CommunicatorCollectives>(
        k: Complex<f64>,
        problem: &Problem,
        gradients: bool,
        threads: usize,
        comm: &C,
    ) -> Self {
        let (rank, size) = (comm.rank() as usize, comm.size() as usize);
        let n = problem.sample.len();
        let mine: Vec<usize> = (rank..n).step_by(size).collect();
        let part = Self::of_samples(k, problem, &mine, gradients, threads);
        let (mut potential, mut gradient) = (Vec::new(), Vec::new());
        for v in 0..part.potential.len() {
            let width = if gradients { 8 } else { 2 };
            let mut local = vec![0.0f64; width * n];
            for (j, &s) in mine.iter().enumerate() {
                let phi = part.potential[v][j];
                local[width * s] = phi.re;
                local[width * s + 1] = phi.im;
                if gradients {
                    for (a, g) in part.gradient[v][j].iter().enumerate() {
                        local[width * s + 2 + 2 * a] = g.re;
                        local[width * s + 3 + 2 * a] = g.im;
                    }
                }
            }
            let mut all = vec![0.0f64; width * n];
            comm.all_reduce_into(&local[..], &mut all[..], SystemOperation::sum());
            let values = all.chunks_exact(width);
            potential.push(values.clone().map(|c| Complex::new(c[0], c[1])).collect());
            gradient.push(if gradients {
                values
                    .map(|c| std::array::from_fn(|a| Complex::new(c[2 + 2 * a], c[3 + 2 * a])))
                    .collect()
            } else {
                Vec::new()
            });
        }
        Self {
            potential,
            gradient,
        }
    }
}

/// A 64-bit mix (SplitMix64's finaliser).
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// This rank's part of the rank-invariant checksum of `output` (module documentation),
/// for targets whose global positions start at `first`: the wrapping sum over every value
/// of mix(bits + mix(slot + mix(i))), i the global position and slot the value's place in
/// the target's output (φ columns first, then ∇φ by column and axis). The ranks' parts
/// add up, wrapping, to the checksum of the whole output.
pub fn checksum(output: &Output<f64>, first: usize) -> u64 {
    let n = output.ntargets();
    let columns = output.potential.len() / n.max(1);
    let mut sum = 0u64;
    for i in 0..n {
        let point = mix(SEED ^ (first + i) as u64);
        for o in 0..columns {
            let value = output.potential[o * n + i];
            sum = sum.wrapping_add(mix(value.to_bits().wrapping_add(mix(point ^ o as u64))));
            if let Some(gradient) = &output.gradient {
                for (a, g) in gradient[o * n + i].iter().enumerate() {
                    let slot = (columns + 3 * o + a) as u64;
                    sum = sum.wrapping_add(mix(g.to_bits().wrapping_add(mix(point ^ slot))));
                }
            }
        }
    }
    sum
}

/// The wrapping sum of every rank's `part`.
///
/// # Collective operation
///
/// On `comm`.
pub fn reduce_checksum<C: CommunicatorCollectives>(part: u64, comm: &C) -> u64 {
    let mut parts = vec![0u64; comm.size() as usize];
    comm.all_gather_into(&part, &mut parts[..]);
    parts.into_iter().fold(0, u64::wrapping_add)
}

/// The relative ℓ2 errors of `output` at the sampled targets this rank holds, per vector,
/// [potential, gradient]: the sums of squares of error and reference, summed over the
/// ranks, and their ratios. The last entry is over all vectors together.
///
/// # Collective operation
///
/// On `comm`.
pub fn errors<C: CommunicatorCollectives>(
    output: &Output<f64>,
    problem: &Problem,
    block: &Range<usize>,
    oracle: &Oracle,
    comm: &C,
) -> Vec<[f64; 2]> {
    let vectors = output.vectors();
    // Per vector [φ error², φ reference², ∇φ error², ∇φ reference²].
    let mut sums = vec![0.0f64; 4 * vectors];
    for (s, &position) in problem.sample.iter().enumerate() {
        if !block.contains(&position) {
            continue;
        }
        let i = position - block.start;
        for v in 0..vectors {
            let phi = output.potential_of(v);
            let approx = Complex::new(phi.re[i], phi.im.map_or(0.0, |im| im[i]));
            let exact = oracle.potential[v][s];
            sums[4 * v] += (approx - exact).norm_sqr();
            sums[4 * v + 1] += exact.norm_sqr();
            if let Some(grad) = output.gradient_of(v) {
                for a in 0..3 {
                    let approx = Complex::new(grad.re[i][a], grad.im.map_or(0.0, |im| im[i][a]));
                    let exact = oracle.gradient[v][s][a];
                    sums[4 * v + 2] += (approx - exact).norm_sqr();
                    sums[4 * v + 3] += exact.norm_sqr();
                }
            }
        }
    }
    let mut all = vec![0.0f64; 4 * vectors];
    comm.all_reduce_into(&sums[..], &mut all[..], SystemOperation::sum());
    let ratio = |e: f64, r: f64| if r > 0.0 { (e / r).sqrt() } else { 0.0 };
    let mut rows: Vec<[f64; 2]> = all
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| [ratio(c[0], c[1]), ratio(c[2], c[3])])
        .collect();
    let total = |j: usize| all.iter().skip(j).step_by(4).sum::<f64>();
    rows.push([ratio(total(0), total(1)), ratio(total(2), total(3))]);
    rows
}

/// The crossovers of [`HelmholtzM2l::Auto`] that choose differently on a tree of `degrees`
/// (Helmholtz CH3.2d's sweep): `usize::MAX` (every level with tables), then each distinct
/// p_l of the levels with M2L (≥ 2) but the largest, descending, then one below the
/// smallest (Rotation on every level). Empty without a level of M2L.
pub fn sweep_crossovers(degrees: &[usize]) -> Vec<usize> {
    let mut distinct: Vec<usize> = degrees.iter().skip(2).copied().collect();
    distinct.sort_unstable();
    distinct.dedup();
    let Some(&smallest) = distinct.first() else {
        return Vec::new();
    };
    distinct.pop();
    std::iter::once(usize::MAX)
        .chain(distinct.into_iter().rev())
        .chain(smallest.checked_sub(1))
        .collect()
}

/// The outcome of one [`Case`] at one thread count.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The rayon threads per rank.
    pub threads: usize,
    /// The P2P kernel that ran (never `Auto`).
    pub p2p: P2pChoice,
    /// The M2L setting.
    pub m2l: HelmholtzM2l,
    /// The batched M2L's multiply–add mode.
    pub m2l_mul_add: MulAdd,
    /// The agreed crossover degree under [`HelmholtzM2l::Auto`], else `None`.
    pub m2l_max_degree: Option<usize>,
    /// The agreed smallest degree of Auto's exponential M2L, `None` unless
    /// [`Case::auto_plane_wave`] (Helmholtz CH6.2h).
    pub m2l_plane_wave_min_degree: Option<usize>,
    /// Per level, whether its M2L ran the setting's tables, Dense, Classes, Batched or the
    /// exponential M2L (else RCR).
    pub dense_levels: Vec<bool>,
    /// The bytes of the dense, class or batched M2L tables per rank, plus under
    /// [`HelmholtzM2l::PlaneWave`] (or Auto with [`Case::auto_plane_wave`]) the exponential
    /// M2L's tables and buffers (`plane_wave_bytes`).
    pub dense_bytes: usize,
    /// Per level, the precision of its quadrature under [`HelmholtzM2l::PlaneWave`] (and
    /// under Auto with [`Case::auto_plane_wave`]), `None` where it ran another M2L.
    pub plane_wave: Vec<Option<Precision>>,
    /// The bytes of the exponential M2L's waves buffer per rank (part of `dense_bytes`;
    /// Helmholtz CH6.2a).
    pub plane_wave_buffer: usize,
    /// The exponential M2L's level calls, chunks and waves computed in the evaluation, this
    /// rank's (Helmholtz CH6.2a).
    pub plane_wave_counts: PlaneWaveCounts,
    /// The lateral reach of the quadratures' α counts as built (Helmholtz CH6.2e).
    pub plane_wave_reach: f64,
    /// How merge and shift split over the threads (Helmholtz CH6.2f).
    pub plane_wave_split: PlaneWaveSplit,
    /// Per level, its waves per direction Σₖ Nₖ under [`HelmholtzM2l::PlaneWave`], 0 where
    /// it ran RCR (and under every other setting; Helmholtz CH6.2e).
    pub plane_wave_waves: Vec<usize>,
    /// The ranks.
    pub ranks: usize,
    /// The levels of the tree.
    pub levels: usize,
    /// The leaf rule's minimum leaf level.
    pub min_leaf_level: usize,
    /// The degree of every level.
    pub degrees: Vec<usize>,
    /// The circumradius √3 r_l of every level's boxes in the domain of the FMM (the
    /// points' bounding cube unless named), for x = Re(k) a_l (Helmholtz CH3.3).
    pub circumradii: Vec<f64>,
    /// The leaves of the tree, summed over the ranks.
    pub leaves: usize,
    /// The most points in a leaf.
    pub max_leaf_points: usize,
    /// The build, seconds (slowest rank).
    pub build: f64,
    /// The evaluation, seconds (slowest rank); the last of [`Case::evaluations`].
    pub evaluate: f64,
    /// The first evaluation, seconds (slowest rank); `evaluate` with one evaluation.
    pub first_evaluate: f64,
    /// Per level, the M2L level calls' seconds in the last evaluation, the slowest rank's
    /// (`HelmholtzOperator::m2l_times`; Helmholtz CH6.2b).
    pub m2l_seconds: Vec<f64>,
    /// Per level, the exponential M2L's passes in the last evaluation, [waves, merge and
    /// shift] seconds, each the slowest rank's (`HelmholtzOperator::plane_wave_times`;
    /// Helmholtz CH6.2d); zero where no level call ran merge and shift.
    pub plane_wave_seconds: Vec<[f64; 2]>,
    /// Per level, merge and shift's thread-seconds in the last evaluation, [busy, shifts,
    /// children, conversions], each the slowest rank's (`PlaneWaveTimes`' parts; Helmholtz
    /// CH6.2f): busy over merge and shift's seconds is the threads it kept busy.
    pub plane_wave_parts: Vec<[f64; 4]>,
    /// Per level, the V-list pairs of the plan, each target counted on the rank that owns it
    /// and summed over the ranks: the same on every rank count (Helmholtz CH6.2b).
    pub v_pairs: Vec<usize>,
    /// The relative ℓ2 errors [φ, ∇φ] per vector, then over all vectors.
    pub errors: Vec<[f64; 2]>,
    /// The checksum of the whole output.
    pub checksum: u64,
}

/// The slowest rank's `seconds`.
fn slowest<C: CommunicatorCollectives>(seconds: f64, comm: &C) -> f64 {
    let mut max = 0.0f64;
    comm.all_reduce_into(&seconds, &mut max, SystemOperation::max());
    max
}

/// Builds and evaluates the FMM of `case` on `problem` at `threads` threads per rank and
/// measures it against `oracle` (module documentation).
///
/// # Collective operation
///
/// On `comm`, every rank with the same arguments.
///
/// # Errors
///
/// The FMM's build and evaluation errors.
///
/// # Panics
///
/// If `case.k` is not a valid wavenumber.
pub fn run<C: CommunicatorCollectives>(
    case: &Case,
    problem: &Problem,
    oracle: &Oracle,
    threads: usize,
    comm: &C,
) -> Result<Row, HelmholtzFmmError> {
    run_output(case, problem, oracle, threads, comm).map(|run| run.row)
}

/// A [`Row`] with what it was measured on (Helmholtz CH3.6, the full-target crosscheck).
#[derive(Clone, Debug)]
pub struct Run {
    /// The row.
    pub row: Row,
    /// This rank's output, at the points of [`Problem::block`] in their order.
    pub output: Output<f64>,
    /// The FMM's domain.
    pub domain: Domain,
}

/// [`run`], also returning this rank's output and the FMM's domain.
///
/// # Collective operation
///
/// On `comm`, every rank with the same arguments.
///
/// # Errors
///
/// The FMM's build and evaluation errors.
///
/// # Panics
///
/// If `case.k` is not a valid wavenumber.
pub fn run_output<C: CommunicatorCollectives>(
    case: &Case,
    problem: &Problem,
    oracle: &Oracle,
    threads: usize,
    comm: &C,
) -> Result<Run, HelmholtzFmmError> {
    let k = Wavenumber::new(case.k.0, case.k.1).expect("a valid wavenumber");
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let block = problem.block(rank, size);
    let points = &problem.points[block.clone()];
    comm.barrier();
    let start = Instant::now();
    let mut builder = HelmholtzFmmBuilder::new(k, case.degrees.clone())
        .charges(case.charges)
        .vectors(case.vectors)
        .gradients(case.gradients)
        .threads(threads)
        .p2p_kernel(case.p2p)
        .m2l(case.m2l)
        .dense_budget(case.dense_budget)
        .auto_plane_wave(case.auto_plane_wave)
        .plane_wave_precision(case.plane_wave.clone());
    if let Some(mul_add) = case.m2l_mul_add {
        builder = builder.m2l_mul_add(mul_add);
    }
    if let Some(max_degree) = case.m2l_max_degree {
        builder = builder.m2l_max_degree(max_degree);
    }
    if let Some(min_degree) = case.m2l_plane_wave_min_degree {
        builder = builder.m2l_plane_wave_min_degree(min_degree);
    }
    if let Some(budget) = case.plane_wave_budget {
        builder = builder.plane_wave_budget(budget);
    }
    if let Some(reach) = case.plane_wave_reach {
        builder = builder.plane_wave_reach(reach);
    }
    if let Some(split) = case.plane_wave_split {
        builder = builder.plane_wave_split(split);
    }
    let mut fmm: HelmholtzFmm<'_, f64, C> = builder.build(points, points, comm)?;
    let build = slowest(start.elapsed().as_secs_f64(), comm);
    let charges: Vec<Charges<'_, f64>> = (0..case.vectors)
        .map(|v| match case.charges {
            ChargeKind::Real => Charges::Real(&problem.re[v][block.clone()]),
            ChargeKind::Complex => Charges::Complex {
                re: &problem.re[v][block.clone()],
                im: &problem.im[v][block.clone()],
            },
        })
        .collect();
    // The M2L times before the last evaluation, so that the row has the last one's.
    let (mut output, mut evaluate, mut first_evaluate) = (None, 0.0, 0.0);
    let mut before: Vec<Duration> = Vec::new();
    let mut passes_before: Vec<PlaneWaveTimes> = Vec::new();
    for evaluation in 0..case.evaluations.max(1) {
        before = fmm.operator().m2l_times().to_vec();
        passes_before = fmm.operator().plane_wave_times().to_vec();
        comm.barrier();
        let start = Instant::now();
        output = Some(fmm.evaluate(&charges)?);
        evaluate = slowest(start.elapsed().as_secs_f64(), comm);
        if evaluation == 0 {
            first_evaluate = evaluate;
        }
    }
    let output = output.expect("at least one evaluation");
    let errors = errors(&output, problem, &block, oracle, comm);
    let checksum = reduce_checksum(checksum(&output, block.start), comm);
    let mut leaves = 0usize;
    comm.all_reduce_into(&fmm.nleaves(), &mut leaves, SystemOperation::sum());
    let nlevels = fmm.nlevels();
    let seconds: Vec<f64> = fmm
        .operator()
        .m2l_times()
        .iter()
        .zip(&before)
        .map(|(after, before)| (*after - *before).as_secs_f64())
        .collect();
    let mut m2l_seconds = vec![0.0f64; nlevels];
    comm.all_reduce_into(&seconds[..], &mut m2l_seconds[..], SystemOperation::max());
    let passes: Vec<f64> = fmm
        .operator()
        .plane_wave_times()
        .iter()
        .zip(&passes_before)
        .flat_map(|(after, before)| {
            [
                (after.waves - before.waves).as_secs_f64(),
                (after.parents - before.parents).as_secs_f64(),
            ]
        })
        .collect();
    let mut slowest_passes = vec![0.0f64; 2 * nlevels];
    comm.all_reduce_into(&passes[..], &mut slowest_passes[..], SystemOperation::max());
    let plane_wave_seconds: Vec<[f64; 2]> = slowest_passes.as_chunks::<2>().0.to_vec();
    let parts: Vec<f64> = fmm
        .operator()
        .plane_wave_times()
        .iter()
        .zip(&passes_before)
        .flat_map(|(after, before)| {
            [
                (after.busy - before.busy).as_secs_f64(),
                (after.shifts - before.shifts).as_secs_f64(),
                (after.children - before.children).as_secs_f64(),
                (after.conversions - before.conversions).as_secs_f64(),
            ]
        })
        .collect();
    let mut slowest_parts = vec![0.0f64; 4 * nlevels];
    comm.all_reduce_into(&parts[..], &mut slowest_parts[..], SystemOperation::max());
    let plane_wave_parts: Vec<[f64; 4]> = slowest_parts.as_chunks::<4>().0.to_vec();
    // Each target once, on the rank that owns it (a replicated top box has rows on every
    // rank).
    let plan = fmm.plan();
    let pairs: Vec<u64> = (0..nlevels)
        .map(|l| {
            let v = plan.level(l).v();
            (0..v.nrows())
                .filter(|&t| plan.computes(l, t))
                .map(|t| v.row(t).0.len() as u64)
                .sum()
        })
        .collect();
    let mut v_pairs = vec![0u64; nlevels];
    comm.all_reduce_into(&pairs[..], &mut v_pairs[..], SystemOperation::sum());
    let row = Row {
        threads,
        p2p: fmm.p2p_kernel(),
        m2l: fmm.operator().m2l_choice(),
        m2l_mul_add: fmm.operator().m2l_mul_add(),
        m2l_max_degree: fmm.m2l_max_degree(),
        m2l_plane_wave_min_degree: fmm.m2l_plane_wave_min_degree(),
        dense_levels: (0..fmm.nlevels())
            .map(|l| fmm.operator().m2l_on(l) != HelmholtzM2l::Rotation)
            .collect(),
        dense_bytes: fmm.operator().dense_bytes()
            + fmm.operator().class_bytes()
            + fmm.operator().batched_bytes()
            + fmm.operator().plane_wave_bytes(),
        plane_wave: (0..fmm.nlevels())
            .map(|l| fmm.operator().plane_wave_precision(l))
            .collect(),
        plane_wave_buffer: fmm.operator().plane_wave_buffer_bytes(),
        plane_wave_counts: fmm.operator().plane_wave_counts(),
        plane_wave_reach: fmm.operator().plane_wave_reach(),
        plane_wave_split: fmm.operator().plane_wave_split(),
        plane_wave_waves: (0..fmm.nlevels())
            .map(|l| fmm.operator().plane_wave_waves(l))
            .collect(),
        ranks: size,
        levels: fmm.nlevels(),
        min_leaf_level: fmm.min_leaf_level(),
        degrees: fmm.degrees().to_vec(),
        circumradii: (0..fmm.nlevels())
            .map(|l| 3f64.sqrt() * radius(l, fmm.domain()))
            .collect(),
        leaves,
        max_leaf_points: fmm.max_leaf_points(),
        build,
        evaluate,
        first_evaluate,
        m2l_seconds,
        plane_wave_seconds,
        plane_wave_parts,
        v_pairs: v_pairs.into_iter().map(|n| n as usize).collect(),
        errors,
        checksum,
    };
    Ok(Run {
        row,
        output,
        domain: *fmm.domain(),
    })
}

/// k as a complex number.
pub fn complex_k(case: &Case) -> Complex<f64> {
    Complex::new(case.k.0, case.k.1)
}
