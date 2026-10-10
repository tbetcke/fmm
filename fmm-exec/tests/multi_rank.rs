//! The C5.1 host gate (Phase 5 T6; docs/phase5/README.md, "Exit gate"): the Laplace FMM on
//! several ranks equals its one-rank result, at the workload points of the phase, from a
//! seeded random share of the points per rank.
//!
//! The problems (docs/phase5/README.md, "Workloads"), as the ignored one-rank gates draw
//! them:
//! - **cube**: the C3.2 problem of `tests/accuracy.rs`: N = 10⁵ points uniform in
//!   [−1, 1)³, seed `0xc32`, the uniform level-4 tree (`max_level` 4, one point per leaf);
//! - **plummer**: the C3.3 problem of `tests/adaptive.rs`: N = 10⁵ points of the Plummer
//!   sphere of scale 0.1, seed `0xc33`, `max_level` 16 and 64 points per leaf.
//!
//! Sources equal targets, gradients on, eight charge vectors uniform in [−1, 1) (vector k
//! from the seed + 1 + k, as `nd_fmm_validate::fmm_accuracy::Problem`; the odd vectors
//! evaluated with the exchanges overlapped, `Fmm::set_overlap`, Phase 5 T9), 1,000 sampled
//! targets; f64 at p = 3, 8 and 18, f32 at p = 3 and 8 (the f32 charges rounded, with an
//! oracle on the rounded charges); one thread per rank. Point i is passed on rank
//! `hash(i) % P` (a seeded random share). On every rank count, for every point:
//! - **Equal to one rank** (requirement 2, decision 7; docs/design/distributed-fmm.md
//!   §5.3, §5.4): every output value of every evaluation, φ and ∇φ, is bit for bit that
//!   of the one-rank `Fmm` of the same settings over the union of every rank's points in
//!   rank order, built on rank 0 on `SimpleCommunicator::self_comm()`; and within 100 u_T
//!   (1.1e-14 in f64, 6.0e-6 in f32) relative L2 over every target, φ and ∇φ, of the
//!   one-rank `Fmm` over the points in their own order (the "one-rank run"), whose
//!   relative max difference is printed too. On one rank both are the run itself.
//! - **Errors against the direct sum** (docs/phase5/README.md, "Accuracy measures"): the
//!   root mean square over the eight charge vectors of the relative L2 error at the
//!   sampled targets, of φ and of ∇φ, against `direct_sum` in f64 divided by 4π, within
//!   0.1% (f64) and 1% (f32) of the one-rank run's.
//! - **The C3.2 and C3.3 gates** on the multi-rank output: in f64, the cube's φ error
//!   within twice the single-translation prediction (1.77e-3, 1.08e-5 and, re-derived in
//!   Phase 3 T9, 6.74e-9 at p = 3, 8, 18), and the Plummer sphere's within twice the
//!   cube's at the same p; f32 is reported.
//!
//! Every rank builds and evaluates the multi-rank `Fmm`; rank 0 alone gathers the output
//! (all-gathers in rank order), computes the oracles and the two one-rank runs, and
//! checks. Its verdict is all-reduced before the test asserts, so the test fails on every
//! rank or on none. It prints the ranks and threads, every rank's leaves, owned points,
//! near pairs (U pairs and each leaf with itself) and V pairs, and per run the build and
//! the evaluation time (the slowest rank; the evaluation the median of the eight) with the
//! share of the moves (`BuildTimings::redistribute`, `StageTimings::forward_charges` and
//! `backward_output`), reported and never asserted. A barrier precedes every timed build,
//! so that no rank's time includes rank 0's checks.
//!
//! Its own executable, because it initialises MPI; ignored, because it needs release
//! mode. By hand at 1, 2, 4 and 8 ranks, under an external timeout (macOS loopback flags of
//! the root `CLAUDE.md`):
//!
//! ```text
//! cargo test -p nd-fmm-exec --release --test multi_rank --no-run
//! RUST_MIN_STACK=8388608 timeout 1800 mpirun --mca btl_tcp_if_include lo0 \
//!     --mca oob_tcp_if_include lo0 -n 4 target/release/deps/multi_rank-<hash> \
//!     --ignored --nocapture
//! ```

use std::time::Duration;

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{Fmm, FmmBuilder, Output};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::cache::Stored;

/// The number of points of each problem.
const N: usize = 100_000;

/// The number of sampled targets.
const SAMPLED: usize = 1000;

/// The number of charge vectors.
const CHARGE_VECTORS: u64 = 8;

/// The f64 degrees with the single-translation prediction of the relative L2 error of φ
/// (design §7; p = 18 re-derived in Phase 3 T9).
const F64_PS: [(usize, f64); 3] = [(3, 1.77e-3), (8, 1.08e-5), (18, 6.74e-9)];

/// The f32 degrees, at most 8 (design §4).
const F32_PS: [usize; 2] = [3, 8];

/// The seed of the random share.
const SHARE_SEED: u64 = 0x05ee_dc51;

/// How far the multi-rank errors against the direct sum may lie from the one-rank run's,
/// relative: 0.1% in f64, 1% in f32 (docs/phase5/README.md, "Accuracy measures").
fn error_ratio<T>() -> f64 {
    if size_of::<T>() == 4 { 0.01 } else { 0.001 }
}

/// 100 u_T: the tolerance between input distributions (design §5.4).
fn distribution_tolerance<T>() -> f64 {
    100.0
        * if size_of::<T>() == 4 {
            2f64.powi(-24)
        } else {
            2f64.powi(-53)
        }
}

/// SplitMix64, as `nd_fmm_validate::SplitMix64`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform in 0..n (n ≤ 2³², with a negligible bias).
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
}

/// The rank that point `i` is passed on: a seeded hash of i, modulo `size`.
fn share(i: usize, size: usize) -> usize {
    (SplitMix64(SHARE_SEED ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).next_u64()
        % size as u64) as usize
}

/// A point at distance `r` from the origin in a uniform direction (`points::sphere`).
fn at_radius(rng: &mut SplitMix64, r: f64) -> [f64; 3] {
    let v = loop {
        let v: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
        let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if (1e-6..=1.0).contains(&norm_sq) {
            break v;
        }
    };
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|vk| r * (vk / norm))
}

/// One problem: its points, sample, charge vectors and tree settings.
struct Workload {
    name: &'static str,
    points: Vec<[f64; 3]>,
    sample: Vec<usize>,
    charges: Vec<Vec<f64>>,
    max_level: usize,
    max_points_per_leaf: usize,
}

impl Workload {
    /// The problem `name` (module documentation): the points from the seed, then the
    /// sample from the same generator (a partial Fisher–Yates shuffle), and charge vector
    /// k from the seed + 1 + k.
    fn new(name: &'static str) -> Self {
        let (seed, max_level, max_points_per_leaf) = match name {
            "cube" => (0xc32, 4, 1),
            "plummer" => (0xc33, 16, 64),
            _ => unreachable!("{name}"),
        };
        let mut rng = SplitMix64(seed);
        let points: Vec<[f64; 3]> = match name {
            "cube" => (0..N)
                .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
                .collect(),
            _ => {
                let mass = 1000.0 / 101.0f64.powf(1.5);
                (0..N)
                    .map(|_| {
                        let u = mass * rng.uniform();
                        let r = if u > 0.0 {
                            0.1 / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
                        } else {
                            0.0
                        };
                        at_radius(&mut rng, r)
                    })
                    .collect()
            }
        };
        let mut order: Vec<usize> = (0..N).collect();
        for i in 0..SAMPLED {
            let j = i + rng.below(N - i);
            order.swap(i, j);
        }
        order.truncate(SAMPLED);
        let charges = (0..CHARGE_VECTORS)
            .map(|k| {
                let mut rng = SplitMix64(seed + 1 + k);
                (0..N).map(|_| rng.range(-1.0, 1.0)).collect()
            })
            .collect();
        Self {
            name,
            points,
            sample: order,
            charges,
            max_level,
            max_points_per_leaf,
        }
    }

    /// The settings at degree `p`, gradients on, one thread.
    fn builder<T>(&self, p: usize) -> FmmBuilder<T> {
        FmmBuilder::<T>::new(p)
            .max_level(self.max_level)
            .max_points_per_leaf(self.max_points_per_leaf)
            .gradients(true)
    }
}

/// The exact φ and ∇φ at the sampled targets per charge vector: `direct_sum` in f64,
/// divided by 4π.
type Oracle = Vec<(Vec<f64>, Vec<[f64; 3]>)>;

/// The oracle of `workload` for the charge vectors `charges`.
fn oracle(workload: &Workload, charges: &[Vec<f64>]) -> Oracle {
    let targets: Vec<[f64; 3]> = workload
        .sample
        .iter()
        .map(|&i| workload.points[i])
        .collect();
    let scale = 4.0 * std::f64::consts::PI;
    charges
        .iter()
        .map(|q| {
            let mut phi = vec![0.0; targets.len()];
            let mut grad = vec![[0.0; 3]; targets.len()];
            direct_sum(&workload.points, q, &targets, &mut phi, Some(&mut grad));
            (
                phi.iter().map(|v| v / scale).collect(),
                grad.iter().map(|g| g.map(|gk| gk / scale)).collect(),
            )
        })
        .collect()
}

/// The values of every rank's `local`, in rank order (an all-gather of the counts and one
/// of the values).
fn union<V: Equivalence + Copy + Default>(local: &[V], comm: &SimpleCommunicator) -> Vec<V> {
    let size = comm.size() as usize;
    let mut counts = vec![0i32; size];
    comm.all_gather_into(&(local.len() as i32), &mut counts[..]);
    let displacements: Vec<i32> = counts
        .iter()
        .scan(0, |start, &count| {
            let first = *start;
            *start += count;
            Some(first)
        })
        .collect();
    let total = counts.iter().map(|&n| n as usize).sum();
    let mut all = vec![V::default(); total];
    let mut partition = PartitionMut::new(&mut all[..], &counts[..], &displacements[..]);
    comm.all_gather_varcount_into(local, &mut partition);
    all
}

/// An output as f64: φ, and ∇φ flattened.
fn as_f64<T: RealScalar>(output: &Output<T>) -> (Vec<f64>, Vec<f64>) {
    let f = |v: &[T]| v.iter().map(|&x| RealScalar::to_f64(x)).collect::<Vec<_>>();
    (
        f(&output.potential),
        f(output.gradient.as_ref().expect("gradients").as_flattened()),
    )
}

/// The relative L2 and max differences of `a` from `b`.
fn differences(a: &[f64], b: &[f64]) -> (f64, f64) {
    let (mut d2, mut b2, mut dmax, mut bmax) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d2 += (x - y).powi(2);
        b2 += y * y;
        dmax = dmax.max((x - y).abs());
        bmax = bmax.max(y.abs());
    }
    ((d2 / b2).sqrt(), dmax / bmax)
}

/// The root mean square over the charge vectors of the relative L2 errors of φ and ∇φ of
/// `outputs` (f64, at the positions `at` of the sampled targets) against `oracle`.
fn errors(outputs: &[(Vec<f64>, Vec<f64>)], at: &[usize], oracle: &Oracle) -> (f64, f64) {
    let mut squares = (0.0f64, 0.0f64);
    for ((phi, grad), (exact_phi, exact_grad)) in outputs.iter().zip(oracle) {
        let (mut e, mut r, mut eg, mut rg) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (s, &i) in at.iter().enumerate() {
            e += (phi[i] - exact_phi[s]).powi(2);
            r += exact_phi[s].powi(2);
            for c in 0..3 {
                eg += (grad[3 * i + c] - exact_grad[s][c]).powi(2);
                rg += exact_grad[s][c].powi(2);
            }
        }
        squares.0 += e / r;
        squares.1 += eg / rg;
    }
    let k = outputs.len() as f64;
    ((squares.0 / k).sqrt(), (squares.1 / k).sqrt())
}

/// Evaluates every charge vector on the one-rank `Fmm` of `builder` over `points` on
/// `SimpleCommunicator::self_comm()`; the outputs as f64.
fn one_rank<T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    points: &[[f64; 3]],
    charges: &[Vec<T>],
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let one = SimpleCommunicator::self_comm();
    let mut fmm = builder
        .build(points, points, &one)
        .expect("the one-rank FMM builds");
    charges
        .iter()
        .map(|q| as_f64(&fmm.evaluate(q).expect("the one-rank FMM evaluates")))
        .collect()
}

/// The median of `times`.
fn median(mut times: Vec<Duration>) -> Duration {
    times.sort();
    times[times.len() / 2]
}

/// The largest `local` over all ranks.
fn slowest(local: Duration, comm: &SimpleCommunicator) -> Duration {
    let mut max = 0.0f64;
    comm.all_reduce_into(&local.as_secs_f64(), &mut max, SystemOperation::max());
    Duration::from_secs_f64(max)
}

/// The φ errors of a run, by (workload, precision, p), for the C3.3 gate.
type Errors = Vec<(&'static str, &'static str, usize, f64)>;

/// One run at degree `p` in `T` (module documentation): the multi-rank build and its
/// evaluations on every rank, the checks on rank 0. Appends rank 0's failures and the
/// φ error of the multi-rank run; prints every rank's tree if `first`.
fn run<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    (p, prediction): (usize, Option<f64>),
    (charges, oracle): (&[Vec<f64>], Option<&Oracle>),
    mine: &[usize],
    first: bool,
    (failures, errors_by_run): (&mut Vec<String>, &mut Errors),
    comm: &SimpleCommunicator,
) {
    let (rank, size) = (comm.rank(), comm.size() as usize);
    let precision = if size_of::<T>() == 4 { "f32" } else { "f64" };
    let what = format!("{}, {precision}, p = {p}", workload.name);
    let all_charges: Vec<Vec<T>> = charges
        .iter()
        .map(|q| q.iter().map(|&v| T::from_f64(v)).collect())
        .collect();
    let local_points: Vec<[f64; 3]> = mine.iter().map(|&i| workload.points[i]).collect();
    let local_charges: Vec<Vec<T>> = all_charges
        .iter()
        .map(|q| mine.iter().map(|&i| q[i]).collect())
        .collect();
    let builder = workload.builder::<T>(p);

    // The multi-rank run, on every rank.
    comm.barrier();
    let mut fmm: Fmm<'_, T> = builder
        .build(&local_points, &local_points, comm)
        .expect("the FMM builds");
    let build = fmm.build_timings();
    let mut outputs = Vec::new();
    let (mut totals, mut moves) = (Vec::new(), Vec::new());
    for (k, q) in local_charges.iter().enumerate() {
        // Every second charge vector with the exchanges overlapped (Phase 5 T9).
        fmm.set_overlap(k % 2 == 1);
        let output = fmm.evaluate(q).expect("the FMM evaluates");
        let t = output.timings;
        totals.push(t.total());
        moves.push(t.forward_charges + t.backward_output);
        outputs.push(as_f64(&output));
    }
    // Every rank's leaves, owned points, near pairs and V pairs, once per workload.
    if first {
        let sizes = fmm.list_sizes();
        let mine = [
            fmm.nleaves() as u64,
            fmm.owned_points().0 as u64,
            (sizes.u + fmm.nleaves()) as u64,
            sizes.v as u64,
            local_points.len() as u64,
        ];
        let all = union(&mine, comm);
        if rank == 0 {
            let rows: Vec<String> = all
                .chunks(5)
                .enumerate()
                .map(|(r, c)| {
                    format!(
                        "rank {r}: {} leaves, {} points owned ({} passed), {} near pairs, {} V \
                         pairs",
                        c[0], c[1], c[4], c[2], c[3]
                    )
                })
                .collect();
            let owned: Vec<u64> = all.chunks(5).map(|c| c[1]).collect();
            let mean = owned.iter().sum::<u64>() as f64 / owned.len() as f64;
            eprintln!(
                "{}: {size} ranks × 1 thread, points max/mean {:.3}; {}",
                workload.name,
                *owned.iter().max().unwrap() as f64 / mean,
                rows.join("; ")
            );
        }
    }
    let times = [
        slowest(build.total(), comm),
        slowest(build.redistribute, comm),
        slowest(median(totals), comm),
        slowest(median(moves), comm),
    ];
    // Every rank's output in rank order: the order of the union.
    let gathered: Vec<(Vec<f64>, Vec<f64>)> = outputs
        .iter()
        .map(|(phi, grad)| (union(phi, comm), union(grad, comm)))
        .collect();
    drop(fmm);

    if rank == 0 {
        let oracle = oracle.expect("rank 0 has the oracle");
        // The points in union order: rank 0's, then rank 1's, …
        let order: Vec<usize> = (0..size)
            .flat_map(|r| (0..N).filter(move |&i| share(i, size) == r))
            .collect();
        let mut position = vec![0; N];
        for (u, &i) in order.iter().enumerate() {
            position[i] = u;
        }
        let in_union = |values: &[Vec<T>]| -> Vec<Vec<T>> {
            values
                .iter()
                .map(|q| order.iter().map(|&i| q[i]).collect())
                .collect()
        };
        let union_points: Vec<[f64; 3]> = order.iter().map(|&i| workload.points[i]).collect();
        let (union_reference, point_order) = if size == 1 {
            (gathered.clone(), gathered.clone())
        } else {
            (
                one_rank(&builder, &union_points, &in_union(&all_charges)),
                one_rank(&builder, &workload.points, &all_charges),
            )
        };
        // Bit for bit the union in rank order.
        let differing: usize = gathered
            .iter()
            .zip(&union_reference)
            .map(|((phi, grad), (rphi, rgrad))| {
                phi.iter()
                    .chain(grad)
                    .zip(rphi.iter().chain(rgrad))
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count()
            })
            .sum();
        // Against the one-rank run in point order: the multi-rank output in point order.
        let (mut worst_l2, mut worst_max) = ([0.0f64; 2], [0.0f64; 2]);
        let in_points: Vec<(Vec<f64>, Vec<f64>)> = gathered
            .iter()
            .map(|(phi, grad)| {
                let phi: Vec<f64> = (0..N).map(|i| phi[position[i]]).collect();
                let grad: Vec<f64> = (0..N)
                    .flat_map(|i| grad[3 * position[i]..3 * position[i] + 3].to_vec())
                    .collect();
                (phi, grad)
            })
            .collect();
        for ((phi, grad), (ophi, ograd)) in in_points.iter().zip(&point_order) {
            for (k, (a, b)) in [(phi, ophi), (grad, ograd)].into_iter().enumerate() {
                let (l2, max) = differences(a, b);
                worst_l2[k] = worst_l2[k].max(l2);
                worst_max[k] = worst_max[k].max(max);
            }
        }
        let errors_multi = errors(&in_points, &workload.sample, oracle);
        let errors_one = errors(&point_order, &workload.sample, oracle);
        let ratios = (errors_multi.0 / errors_one.0, errors_multi.1 / errors_one.1);
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "| {} | {precision} | {p} | {size} | {} of {} | {:.1e} / {:.1e} | {:.1e} / {:.1e} \
             | {:.3e} / {:.3e} | {:.3e} / {:.3e} | {:.4} / {:.4} | {:.1} ({:.1}) | {:.2} ({:.3}) |",
            workload.name,
            differing,
            gathered.len() * 4 * N,
            worst_l2[0],
            worst_l2[1],
            worst_max[0],
            worst_max[1],
            errors_multi.0,
            errors_one.0,
            errors_multi.1,
            errors_one.1,
            ratios.0,
            ratios.1,
            ms(times[0]),
            ms(times[1]),
            ms(times[2]),
            ms(times[3]),
        );
        if differing > 0 {
            failures.push(format!(
                "{what}: {differing} values differ from the one-rank FMM over the union in rank \
                 order"
            ));
        }
        let tolerance = distribution_tolerance::<T>();
        if worst_l2.iter().any(|&d| d > tolerance) {
            failures.push(format!(
                "{what}: {worst_l2:?} from the one-rank run in point order, tolerance \
                 {tolerance:e}"
            ));
        }
        let allowed = error_ratio::<T>();
        if (ratios.0 - 1.0).abs() > allowed || (ratios.1 - 1.0).abs() > allowed {
            failures.push(format!(
                "{what}: errors {errors_multi:?} against the one-rank run's {errors_one:?}"
            ));
        }
        if let Some(prediction) = prediction.filter(|_| workload.name == "cube")
            && errors_multi.0 > 2.0 * prediction
        {
            failures.push(format!(
                "{what}: C3.2: {:e} exceeds twice the prediction {prediction:e}",
                errors_multi.0
            ));
        }
        errors_by_run.push((workload.name, precision, p, errors_multi.0));
    }
}

#[test]
#[ignore = "release mode, by hand at 1, 2, 4 and 8 ranks: N = 10^5 at five workload points"]
fn equal_to_one_rank_at_the_workload_points() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    let (rank, size) = (comm.rank(), comm.size() as usize);
    if rank == 0 {
        eprintln!(
            "C5.1 host gate: {size} ranks × 1 thread, a seeded random share per rank; BLAS: {}",
            [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"
            ]
            .map(|v| format!(
                "{v}={}",
                std::env::var(v).unwrap_or_else(|_| "unset".into())
            ))
            .join(", ")
        );
        eprintln!(
            "| workload | precision | p | ranks | values differing from the union in rank order \
             | φ / ∇φ relative L2 from the one-rank run in point order | relative max | φ \
             error, P ranks / one rank | ∇φ error | ratios | build ms (redistribute) | \
             evaluation ms (moves) |"
        );
        eprintln!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    }
    let mut failures = Vec::new();
    let mut errors_by_run = Errors::new();
    for name in ["cube", "plummer"] {
        let workload = Workload::new(name);
        let mine: Vec<usize> = (0..N)
            .filter(|&i| share(i, size) == rank as usize)
            .collect();
        let charges32: Vec<Vec<f64>> = workload
            .charges
            .iter()
            .map(|q| q.iter().map(|&v| f64::from(v as f32)).collect())
            .collect();
        let (oracle64, oracle32) = if rank == 0 {
            (
                Some(oracle(&workload, &workload.charges)),
                Some(oracle(&workload, &charges32)),
            )
        } else {
            (None, None)
        };
        for (k, (p, prediction)) in F64_PS.into_iter().enumerate() {
            run::<f64>(
                &workload,
                (p, Some(prediction)),
                (&workload.charges, oracle64.as_ref()),
                &mine,
                k == 0,
                (&mut failures, &mut errors_by_run),
                &comm,
            );
        }
        for p in F32_PS {
            run::<f32>(
                &workload,
                (p, None),
                (&charges32, oracle32.as_ref()),
                &mine,
                false,
                (&mut failures, &mut errors_by_run),
                &comm,
            );
        }
    }
    // The C3.3 gate in f64: the Plummer sphere's φ error within twice the cube's.
    if rank == 0 {
        for &(_, precision, p, error) in errors_by_run.iter().filter(|e| e.0 == "plummer") {
            let cube = errors_by_run
                .iter()
                .find(|e| e.0 == "cube" && e.1 == precision && e.2 == p)
                .expect("the cube runs every point")
                .3;
            let ratio = error / cube;
            eprintln!(
                "C3.3, {precision}, p = {p}: plummer {error:.3e} against the cube's {cube:.3e}, \
                 ratio {ratio:.2}{}",
                if precision == "f64" {
                    if ratio <= 2.0 {
                        " (gate: passes)"
                    } else {
                        " (gate: fails)"
                    }
                } else {
                    " (f32: reported)"
                }
            );
            if precision == "f64" && ratio > 2.0 {
                failures.push(format!("C3.3, p = {p}: ratio {ratio}"));
            }
        }
    }
    let mut failed = 0u64;
    comm.all_reduce_into(
        &(failures.len() as u64),
        &mut failed,
        SystemOperation::max(),
    );
    assert!(
        failed == 0,
        "rank {rank}: {failed} failures (rank 0 lists them): {}",
        failures.join("; ")
    );
    if rank == 0 {
        eprintln!("C5.1 host gate on {size} ranks: every check passes");
    }
}
