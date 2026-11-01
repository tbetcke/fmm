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
//!   number of ranks; a single changed bit changes it except with probability about 2⁻⁶⁴.
//!   Determinism (proposal §2, constraint 2) is shown by equal checksums across thread
//!   counts and rank counts.
//! - **Times.** The build and the evaluation, wall time of the slowest rank.

use std::ops::Range;
use std::thread;
use std::time::Instant;

use mpi::collective::SystemOperation;
use mpi::traits::CommunicatorCollectives;
use nd_fmm_exec::helmholtz::{
    ChargeKind, Charges, Degrees, HelmholtzFmm, HelmholtzFmmBuilder, HelmholtzFmmError, Output,
    Wavenumber,
};
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p::direct_sum;

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
}

impl Case {
    /// A case with complex charges, one vector, gradients and 1,000 samples (§9.1), seeded
    /// with the workloads' [`SEED`].
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

/// The outcome of one [`Case`] at one thread count.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The rayon threads per rank.
    pub threads: usize,
    /// The ranks.
    pub ranks: usize,
    /// The levels of the tree.
    pub levels: usize,
    /// The leaf rule's minimum leaf level.
    pub min_leaf_level: usize,
    /// The degree of every level.
    pub degrees: Vec<usize>,
    /// The leaves of the tree, summed over the ranks.
    pub leaves: usize,
    /// The most points in a leaf.
    pub max_leaf_points: usize,
    /// The build, seconds (slowest rank).
    pub build: f64,
    /// The evaluation, seconds (slowest rank).
    pub evaluate: f64,
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
    let k = Wavenumber::new(case.k.0, case.k.1).expect("a valid wavenumber");
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let block = problem.block(rank, size);
    let points = &problem.points[block.clone()];
    comm.barrier();
    let start = Instant::now();
    let mut fmm: HelmholtzFmm<'_, f64, C> = HelmholtzFmmBuilder::new(k, case.degrees.clone())
        .charges(case.charges)
        .vectors(case.vectors)
        .gradients(case.gradients)
        .threads(threads)
        .build(points, points, comm)?;
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
    comm.barrier();
    let start = Instant::now();
    let output = fmm.evaluate(&charges)?;
    let evaluate = slowest(start.elapsed().as_secs_f64(), comm);
    let errors = errors(&output, problem, &block, oracle, comm);
    let checksum = reduce_checksum(checksum(&output, block.start), comm);
    let mut leaves = 0usize;
    comm.all_reduce_into(&fmm.nleaves(), &mut leaves, SystemOperation::sum());
    Ok(Row {
        threads,
        ranks: size,
        levels: fmm.nlevels(),
        min_leaf_level: fmm.min_leaf_level(),
        degrees: fmm.degrees().to_vec(),
        leaves,
        max_leaf_points: fmm.max_leaf_points(),
        build,
        evaluate,
        errors,
        checksum,
    })
}

/// k as a complex number.
pub fn complex_k(case: &Case) -> Complex<f64> {
    Complex::new(case.k.0, case.k.1)
}
