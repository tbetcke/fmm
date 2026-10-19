//! Rank-count invariance (Phase 5S T2, C5S.1; requirement 3 of docs/phase5s/README.md;
//! docs/design/scale-out.md §5.1; S7 of docs/design/distributed-fmm.md §14.4): a run on P
//! ranks equals a run on P/k ranks bit for bit when rank r of the small run holds the
//! concatenation of the inputs of ranks k r, …, k r + k − 1 of the large one, in that order.
//! It is the oracle wherever no one-rank run fits; here it runs where one does, so that
//! every later structural task (T3–T8) checks it in CI.
//!
//! **Generation by index.** Point i and its charge come from their own generator, seeded by
//! a hash of the workload's seed and i, and point i goes to world rank `hash(i) mod P` (a
//! seeded random share), each rank's points in index order. So any rank produces any rank's
//! input without holding the global set (`nd_fmm_validate::scaling::Workload` does the
//! same for the harness; this crate cannot depend on it).
//!
//! **The check.** The world (P ranks) builds and evaluates the `Fmm` first, on one path
//! (blocking, or overlapped through `Fmm::set_overlap`), and keeps its output. Then, for every k that divides P
//! (k > 1; on one rank k = 1, a second build on a communicator of its own), `split` of the
//! world gives the small communicator: colour 0 on the ranks k r, no colour elsewhere.
//! Every world rank enters the split; the others skip the small run and enter nothing of
//! the small communicator. Small rank r generates the inputs of world ranks k r … k r + k − 1,
//! concatenates them in that order, builds the same `Fmm`, evaluates it on the other path,
//! and sends each world rank k r + i (i ≥ 1) its slice of the output, point to point (the
//! sizes known on both sides); slice 0 stays. Each world rank compares its own targets'
//! values bit for bit with its own output. So every comparison is blocking against
//! overlapped across rank counts: overlap off on the large run and on in the small ones
//! in f64, the other way round in f32 (one evaluation per `Fmm` keeps the debug run
//! short). One all-reduce over the world agrees the count of differing values, so the
//! test fails on every rank or on none. Nothing is gathered.
//!
//! **What must be the same** (scale-out.md §5.1): the precision, p, `max_level`, the leaf
//! size and the domain (the bounding box of the same union), the near rule, and **the
//! strategy, named**:
//! `Auto` resolves by the ranks per node, which differ between the runs. Threads, overlap,
//! the table cache and the cut change no bit.
//!
//! **Workloads** (T2 brief), N = 10,000, sources equal to targets, gradients on,
//! `max_level` 16 and 64 points per leaf (the phase's tree):
//! - **cube**: uniform in the unit cube;
//! - **plummer**: the Plummer sphere of scale a = 0.1 at the origin, truncated at 10 a;
//! - **graded, one empty rank**: a fifth of the points over the unit cube and the rest in a
//!   cube of side 2⁻⁶ inside it (leaves on levels several apart, W and X lists), and on
//!   several ranks the last world rank passes nothing: its share goes to rank 0.
//!
//! each in f64 at p = 8 with `Rotation` and in f32 at p = 3 with `Dense`; the Plummer
//! sphere and the graded tree also with the near rule of the cost model (Phase 6 T12,
//! `NearRuleChoice::CostModel`), the same on every rank count since the rule decides each
//! pair from the leaf's count alone; and every workload in f64 at p = 8 with the plane-wave
//! M2L (Phase 6 T13, `M2lStrategy::PlaneWave`), the graded tree also in f32 at p = 6.
//!
//! Error measure: exact equality of the bit patterns of φ and ∇φ at every target.
//!
//! A deliberately broken input order (the concatenation in reverse rank order, its slices
//! still sent back by index) makes the check fail on every rank: shown once by hand (T2
//! report, `fmm-validate/results/phase5s-t2.md`), not committed as a failing test.
//!
//! Its own executable (docs/design/scale-out.md §9.1, T2's brief change (b)), so that
//! `mpi_exec` keeps its one-minute debug budget; one test initialises MPI, at
//! `Threading::Funneled`. CI's root job runs it at one rank, `run-tests-mpi` at 2 and 4
//! (`.github/scripts/run-mpi-tests.sh`); 8 by hand, under an external timeout (macOS
//! loopback flags of the root `CLAUDE.md`):
//!
//! ```text
//! cargo test -p nd-fmm-exec --test invariance --no-run   # note the executable
//! RUST_MIN_STACK=8388608 timeout 900 mpirun --mca btl_tcp_if_include lo0 \
//!     --mca oob_tcp_if_include lo0 -n 8 target/debug/deps/invariance-<hash> --nocapture
//! ```
//!
//! The builds load their tables from a cache in `CARGO_TARGET_TMPDIR`, built once. The
//! times it prints are for the record, never asserted.

use std::time::Instant;

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::{Color, SimpleCommunicator};
use mpi::traits::*;
use nd_fmm_exec::fmm::{FmmBuilder, NearRuleChoice, Output};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;

/// The number of points of each workload.
const N: usize = 10_000;

/// The table cache of every build: a directory of the test's own.
const TABLE_CACHE: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/invariance_tables");

/// SplitMix64, as `nd_fmm_validate::SplitMix64` (this crate cannot depend on it).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }

    /// A direction uniform on the unit sphere, by rejection in the unit ball.
    fn direction(&mut self) -> [f64; 3] {
        loop {
            let v: [f64; 3] = core::array::from_fn(|_| self.range(-1.0, 1.0));
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if (1e-3..=1.0).contains(&norm) {
                return v.map(|vk| vk / norm);
            }
        }
    }
}

/// A seeded hash of `i` (SplitMix64's finaliser): the seed of point i's generator, and of
/// its rank in the share.
fn hash(seed: u64, i: usize) -> u64 {
    let mut z = seed ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The point distributions of the workloads (module documentation).
#[derive(Clone, Copy, Debug)]
enum Distribution {
    Cube,
    Plummer,
    Graded,
}

/// A problem whose points, charges and share are generated by index.
#[derive(Clone, Copy, Debug)]
struct Workload {
    name: &'static str,
    distribution: Distribution,
    seed: u64,
    /// Whether the last rank passes nothing (on several ranks): its share goes to rank 0.
    empty_rank: bool,
}

/// The workloads of the module documentation.
const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "cube",
        distribution: Distribution::Cube,
        seed: 0x1c0be,
        empty_rank: false,
    },
    Workload {
        name: "plummer",
        distribution: Distribution::Plummer,
        seed: 0x91a3,
        empty_rank: false,
    },
    Workload {
        name: "graded, one empty rank",
        distribution: Distribution::Graded,
        seed: 0x96ade,
        empty_rank: true,
    },
];

/// The seed of the share, mixed into the workload's.
const SHARE_SALT: u64 = 0x5_4a7e;

impl Workload {
    /// Point `i`, from its own generator.
    fn point(&self, i: usize) -> [f64; 3] {
        let mut rng = SplitMix64(hash(self.seed, i));
        match self.distribution {
            Distribution::Cube => core::array::from_fn(|_| rng.range(0.0, 1.0)),
            Distribution::Plummer => {
                // The rule of `nd_fmm_validate::points::plummer`: the mass truncated at
                // r = 10 a, a direction uniform on the sphere.
                let a = 0.1;
                let u = 1000.0 / 101.0f64.powf(1.5) * rng.range(0.0, 1.0);
                let r = if u > 0.0 {
                    a / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
                } else {
                    0.0
                };
                rng.direction().map(|v| r * v)
            }
            Distribution::Graded => {
                if i.is_multiple_of(5) {
                    core::array::from_fn(|_| rng.range(0.0, 1.0))
                } else {
                    core::array::from_fn(|_| 0.3 + rng.range(0.0, 1.0 / 64.0))
                }
            }
        }
    }

    /// The charge of point `i`, uniform in [−1, 1).
    fn charge(&self, i: usize) -> f64 {
        SplitMix64(hash(self.seed.wrapping_add(1), i)).range(-1.0, 1.0)
    }

    /// The world rank of `size` that passes point `i`.
    fn owner(&self, i: usize, size: usize) -> usize {
        let owner = (hash(self.seed ^ SHARE_SALT, i) % size as u64) as usize;
        if self.empty_rank && size > 1 && owner == size - 1 {
            0
        } else {
            owner
        }
    }

    /// The indices that world rank `rank` of `size` passes, ascending.
    fn input(&self, rank: usize, size: usize) -> Vec<usize> {
        (0..N).filter(|&i| self.owner(i, size) == rank).collect()
    }
}

/// One output's values at the targets: φ, and ∇φ flattened.
struct Values<T> {
    potential: Vec<T>,
    gradient: Vec<T>,
}

impl<T: Copy> Values<T> {
    fn new(output: Output<T>) -> Self {
        let gradient = output.gradient.expect("built with gradients");
        Self {
            potential: output.potential,
            gradient: gradient.as_flattened().to_vec(),
        }
    }

    /// The values of `count` targets from `first`, φ then ∇φ: one message.
    fn slice(&self, first: usize, count: usize) -> Vec<T> {
        let mut values = self.potential[first..first + count].to_vec();
        values.extend_from_slice(&self.gradient[3 * first..3 * (first + count)]);
        values
    }

    /// Every value, φ then ∇φ.
    fn all(&self) -> Vec<T> {
        self.slice(0, self.potential.len())
    }
}

/// The number of positions where `a` and `b` differ in any bit (in f64, which holds an f32
/// exactly).
fn differing<T: RealScalar>(a: &[T], b: &[T]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|&(&x, &y)| RealScalar::to_f64(x).to_bits() != RealScalar::to_f64(y).to_bits())
        .count()
}

/// Builds the `Fmm` of `builder` on `comm` over the points `indices` of `workload` (sources
/// equal to targets) and evaluates its charges once per path of `overlapped` (blocking for
/// `false`). Returns the outputs in that order and the resolved strategy.
///
/// # Collective operation
///
/// On every rank of `comm`.
fn evaluate<T, C>(
    builder: &FmmBuilder<T>,
    workload: &Workload,
    indices: &[usize],
    overlapped: &[bool],
    comm: &C,
) -> (Vec<Values<T>>, M2lStrategy)
where
    T: Stored + SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    let points: Vec<[f64; 3]> = indices.iter().map(|&i| workload.point(i)).collect();
    let charges: Vec<T> = indices
        .iter()
        .map(|&i| T::from_f64(workload.charge(i)))
        .collect();
    let mut fmm = builder
        .build(&points, &points, comm)
        .unwrap_or_else(|error| panic!("{}: the FMM does not build: {error}", workload.name));
    let outputs = overlapped
        .iter()
        .map(|&overlap| {
            fmm.set_overlap(overlap);
            Values::new(fmm.evaluate(&charges).unwrap_or_else(|error| {
                panic!("{}: the FMM does not evaluate: {error}", workload.name)
            }))
        })
        .collect();
    (outputs, fmm.strategy())
}

/// The rank-count checks made, for the closing line.
static CHECKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The check of the module documentation for one workload, precision, degree and strategy,
/// at every k of `ks`: the large run evaluated overlapped if `large_overlapped`, blocking
/// otherwise, and the small runs on the other path.
///
/// # Collective operation
///
/// On every rank of `world`.
fn check<T: Stored + SimdScalar + Equivalence + Default>(
    workload: &Workload,
    (p, strategy, near): (usize, M2lStrategy, NearRuleChoice),
    large_overlapped: bool,
    ks: &[usize],
    world: &SimpleCommunicator,
) {
    let (rank, size) = (world.rank() as usize, world.size() as usize);
    let precision = if size_of::<T>() == 4 { "f32" } else { "f64" };
    let path = |overlapped: bool| if overlapped { "overlapped" } else { "blocking" };
    let what = format!(
        "{} {precision} p = {p} {strategy:?}, near rule {near}, the large run {}",
        workload.name,
        path(large_overlapped)
    );
    let builder = FmmBuilder::<T>::new(p)
        .max_level(16)
        .max_points_per_leaf(64)
        .gradients(true)
        .strategy(strategy)
        .near_rule(near)
        .table_cache(TABLE_CACHE);
    let mine = workload.input(rank, size);
    if workload.empty_rank && size > 1 && rank == size - 1 {
        assert!(mine.is_empty(), "rank {rank}: {what}: the empty rank");
    }

    let start = Instant::now();
    let (large, resolved) = evaluate(&builder, workload, &mine, &[large_overlapped], world);
    let large = large[0].all();
    assert_eq!(
        resolved, strategy,
        "rank {rank}: {what}: the strategy as named"
    );
    let large_seconds = start.elapsed().as_secs_f64();

    for &k in ks {
        let leader = rank / k * k;
        let color = if rank == leader {
            Color::with_value(0)
        } else {
            Color::undefined()
        };
        let start = Instant::now();
        let small_comm = world.split_by_color(color);
        // This rank's slice of the small run's output, on the other path.
        let small: Vec<T> = match &small_comm {
            Some(small) => {
                // The inputs of world ranks leader … leader + k − 1, concatenated in
                // `order`; each rank's slice found by its offset in the concatenation.
                let parts: Vec<Vec<usize>> =
                    (0..k).map(|i| workload.input(leader + i, size)).collect();
                let order: Vec<usize> = (0..k).collect();
                let mut offsets = vec![0usize; k];
                let mut concatenated = Vec::new();
                for &i in &order {
                    offsets[i] = concatenated.len();
                    concatenated.extend_from_slice(&parts[i]);
                }
                let (outputs, resolved) = evaluate(
                    &builder,
                    workload,
                    &concatenated,
                    &[!large_overlapped],
                    small,
                );
                assert_eq!(
                    resolved, strategy,
                    "rank {rank}: {what}: the small run's strategy"
                );
                for i in 1..k {
                    let to = world.process_at_rank((leader + i) as i32);
                    to.send(&outputs[0].slice(offsets[i], parts[i].len())[..]);
                }
                outputs[0].slice(offsets[0], parts[0].len())
            }
            None => {
                let from = world.process_at_rank(leader as i32);
                let mut values = vec![T::default(); 4 * mine.len()];
                from.receive_into(&mut values[..]);
                values
            }
        };
        drop(small_comm);
        let local = differing(&large, &small);
        if local > 0 {
            eprintln!(
                "rank {rank}: {what}: {local} of {} values differ between {size} ranks and \
                 {} ranks (k = {k})",
                4 * mine.len(),
                size / k
            );
        }
        let mut counts = [0usize; 3];
        world.all_reduce_into(
            &[local, usize::from(local > 0), 4 * mine.len()][..],
            &mut counts[..],
            SystemOperation::sum(),
        );
        let [total, ranks_differing, compared] = counts;
        if rank == 0 {
            eprintln!(
                "{what}: {size} ranks against {} (k = {k}): {total} of {compared} values \
                 differ, on {ranks_differing} rank(s); {large_seconds:.2} s for the large run, \
                 {:.2} s for the small one on rank 0",
                size / k,
                start.elapsed().as_secs_f64()
            );
        }
        assert_eq!(
            total,
            0,
            "rank {rank}: {what}: {total} values on {ranks_differing} rank(s) differ between \
             {size} ranks and {} ranks holding the concatenated inputs (k = {k})",
            size / k
        );
        assert_eq!(compared, 4 * N, "{what}: every target compared once");
        CHECKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[test]
fn rank_count_invariance() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let world = universe.world();
    let size = world.size() as usize;
    // Every divisor k > 1 of the world size; on one rank k = 1, a run against its repeat.
    let ks: Vec<usize> = if size == 1 {
        vec![1]
    } else {
        (2..=size).filter(|&k| size.is_multiple_of(k)).collect()
    };
    let start = Instant::now();
    let off = NearRuleChoice::Off;
    for workload in &WORKLOADS {
        check::<f64>(
            workload,
            (8, M2lStrategy::Rotation, off),
            false,
            &ks,
            &world,
        );
        check::<f32>(workload, (3, M2lStrategy::Dense, off), true, &ks, &world);
    }
    // The near rule (Phase 6 T12) on the adaptive workloads, whose W and X lists it
    // converts.
    let near = NearRuleChoice::CostModel;
    for workload in &WORKLOADS[1..] {
        check::<f64>(
            workload,
            (8, M2lStrategy::Rotation, near),
            false,
            &ks,
            &world,
        );
        check::<f32>(workload, (3, M2lStrategy::Dense, near), true, &ks, &world);
    }
    // The plane-wave M2L (Phase 6 T13), whose merge and shift sums every accumulator over
    // the same sources in the same order on any rank: every workload in f64 at p = 8, and
    // the graded tree in f32 at p = 6.
    for workload in &WORKLOADS {
        check::<f64>(
            workload,
            (8, M2lStrategy::PlaneWave, off),
            false,
            &ks,
            &world,
        );
    }
    check::<f32>(
        &WORKLOADS[2],
        (6, M2lStrategy::PlaneWave, off),
        true,
        &ks,
        &world,
    );
    if world.rank() == 0 {
        eprintln!(
            "rank-count invariance on {size} rank(s), k in {ks:?}: {} checks, 0 values \
             differ; {:.1} s",
            CHECKS.load(std::sync::atomic::Ordering::Relaxed),
            start.elapsed().as_secs_f64()
        );
    }
}
