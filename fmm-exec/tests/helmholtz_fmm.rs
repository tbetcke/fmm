//! The Helmholtz FMM (Helmholtz CH3.1b; docs/helmholtz/tasks/CH3.1b.md) on any number of
//! ranks: its own executable, whose one test initialises MPI at `Threading::Funneled`.
//!
//! Every scenario generates its points and charges by index from a seed (the same on every
//! rank), and each rank passes every P-th of them (`share`). Per scenario:
//!
//! - against the direct sum (`nd_fmm_ref::helmholtz::p2p::direct_sum` in f64, 1/(4π)
//!   included, over every rank's sources): the relative L2 error of φ and ∇φ over every
//!   rank's targets within `ERROR_FACTOR` times ε for S7's closed form, or within
//!   `P8_BOUND` for degree 8 on every level;
//! - the output pass bit for bit the reference pass of `Densities::write_leaf_output`;
//! - 4 threads bit for bit one (2 and 4 in release builds);
//! - the operator pair by pair (`PerPair`) bit for bit its level calls;
//! - on several ranks, the one-rank FMM over the union of every rank's points and charges
//!   in rank order (`SimpleCommunicator::self_comm()`, on every rank) bit for bit.
//!
//! Then vectors (vector v of an FMM of two bit for bit the one-vector FMM, of complex and
//! of real charges), real charges against complex ones with Im q = 0, and the errors, each
//! on every rank or on none. One scenario leaves rank 0 without points.
//!
//! Every comparison is agreed by an all-reduce, so a failure fails every rank. Run by hand
//! at several ranks under a timeout, e.g. on macOS
//! `cargo test -p nd-fmm-exec --test helmholtz_fmm --no-run`, then
//! `timeout 600 mpirun -n 4 --mca btl_tcp_if_include lo0 --mca oob_tcp_if_include lo0
//! <binary> --test-threads=1 --nocapture` (CI: `.github/scripts/run-mpi-tests.sh`).

use std::f64::consts::PI;
use std::time::Instant;

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives, Equivalence};
use nd_fmm_exec::helmholtz::{
    ChargeKind, Charges, Degrees, DensityError, HelmholtzFmm, HelmholtzFmmBuilder,
    HelmholtzFmmError, OperatorError, Output, Wavenumber,
};
use nd_fmm_math::Complex;
use nd_fmm_ref::helmholtz::p2p;
use nd_octree::PhysicalBox;

type C = Complex<f64>;

/// The error against the direct sum may be this many times ε: the closed form meets ε per
/// translation chain (S7, CH3.1a's chain tests), and an output sums many chains.
const ERROR_FACTOR: f64 = 10.0;

/// The error bound of the scenarios with degree 8 on every level, where truncation
/// dominates: about 5 to 13 times the errors measured on the M6 (7.5e-5 and 2.2e-4).
const P8_BOUND: f64 = 1e-3;

/// SplitMix64, uniform in [0, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// How the points lie in the unit cube (the domain [0, 1]³ is supplied).
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Uniform.
    Cube,
    /// Four Gaussian clusters of width 0.05, so the leaves lie on several levels.
    Clusters,
}

/// `n` points of `layout` strictly inside the unit cube, from `seed`.
fn points(layout: Layout, n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = Rng(seed);
    let centres = [
        [0.2, 0.3, 0.25],
        [0.7, 0.75, 0.3],
        [0.35, 0.7, 0.8],
        [0.8, 0.2, 0.7],
    ];
    (0..n)
        .map(|i| match layout {
            Layout::Cube => core::array::from_fn(|_| 0.001 + 0.998 * rng.next()),
            Layout::Clusters => {
                let c = centres[i % 4];
                core::array::from_fn(|a| {
                    // Box–Muller, clamped inside the cube.
                    let (u, v) = (rng.next().max(1e-12), rng.next());
                    let g = (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos();
                    (c[a] + 0.05 * g).clamp(0.001, 0.999)
                })
            }
        })
        .collect()
}

/// `n` values uniform in [−1, 1), from `seed`.
fn values(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    (0..n).map(|_| 2.0 * rng.next() - 1.0).collect()
}

/// This rank's share of `items`: every `size`-th item from `rank`.
fn share<V: Copy>(items: &[V], comm: &SimpleCommunicator) -> Vec<V> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    items.iter().skip(rank).step_by(size).copied().collect()
}

/// This rank's share of `items`, with `empty`: on several ranks, rank 0 has none and the
/// other ranks share every item; otherwise [`share`].
fn share_of<V: Copy>(items: &[V], empty: bool, comm: &SimpleCommunicator) -> Vec<V> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    if !empty || size == 1 {
        share(items, comm)
    } else if rank == 0 {
        Vec::new()
    } else {
        items
            .iter()
            .skip(rank - 1)
            .step_by(size - 1)
            .copied()
            .collect()
    }
}

/// The values of every rank's `local`, in rank order, and the position of this rank's
/// first value among them.
fn union<V: Equivalence + Copy + Default>(
    local: &[V],
    comm: &SimpleCommunicator,
) -> (Vec<V>, usize) {
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
    (all, displacements[comm.rank() as usize] as usize)
}

/// The points of every rank in rank order, and the position of this rank's first point.
fn union_points(local: &[[f64; 3]], comm: &SimpleCommunicator) -> (Vec<[f64; 3]>, usize) {
    let (flat, first) = union(local.as_flattened(), comm);
    (flat.as_chunks::<3>().0.to_vec(), first / 3)
}

/// The sum of `x` over every rank.
fn global_sum(x: f64, comm: &SimpleCommunicator) -> f64 {
    let mut sum = 0.0;
    comm.all_reduce_into(&x, &mut sum, SystemOperation::sum());
    sum
}

/// Whether `ok` holds on every rank.
fn everywhere(ok: bool, comm: &SimpleCommunicator) -> bool {
    let mut all = false;
    comm.all_reduce_into(&ok, &mut all, SystemOperation::logical_and());
    all
}

/// Every value of an output as bits: per column φ, then the gradient triples.
fn bits(output: &Output<f64>) -> Vec<u64> {
    let mut bits: Vec<u64> = output.potential.iter().map(|x| x.to_bits()).collect();
    if let Some(gradient) = &output.gradient {
        bits.extend(gradient.as_flattened().iter().map(|x| x.to_bits()));
    }
    bits
}

/// Asserts on every rank that `got` equals `want` bit for bit on every rank.
fn assert_same(what: &str, got: &Output<f64>, want: &Output<f64>, comm: &SimpleCommunicator) {
    let same = bits(got) == bits(want);
    if !same {
        eprintln!("rank {}: {what}: the outputs differ", comm.rank());
    }
    assert!(
        everywhere(same, comm),
        "{what}: not bit for bit on every rank"
    );
}

/// A scenario.
#[derive(Clone, Debug)]
struct Scenario {
    name: &'static str,
    layout: Layout,
    /// Sources, and targets (the same set if `None`).
    n: (usize, Option<usize>),
    k: (f64, f64),
    degrees: Degrees,
    /// The error bound against the direct sum (`ERROR_FACTOR` ε, or `P8_BOUND`).
    bound: f64,
    charges: ChargeKind,
    gradients: bool,
    /// Whether the scenario runs in debug builds (each takes a few seconds there).
    debug: bool,
    /// Whether rank 0 passes no point on several ranks.
    empty_rank0: bool,
}

/// The inputs of a scenario on this rank.
struct Inputs {
    sources: Vec<[f64; 3]>,
    targets: Vec<[f64; 3]>,
    /// Per vector: Re q, Im q.
    charges: Vec<(Vec<f64>, Vec<f64>)>,
}

impl Inputs {
    /// The scenario's inputs, m vectors, this rank's share.
    fn new(scenario: &Scenario, m: usize, comm: &SimpleCommunicator) -> Self {
        let (ns, nt) = scenario.n;
        let all_sources = points(scenario.layout, ns, 11);
        let all_targets = nt.map(|nt| points(scenario.layout, nt, 12));
        let empty = scenario.empty_rank0;
        let sources = share_of(&all_sources, empty, comm);
        let targets = share_of(all_targets.as_ref().unwrap_or(&all_sources), empty, comm);
        let charges = (0..m as u64)
            .map(|v| {
                let re = share_of(&values(ns, 100 + 2 * v), empty, comm);
                let im = match scenario.charges {
                    ChargeKind::Real => vec![0.0; re.len()],
                    ChargeKind::Complex => share_of(&values(ns, 101 + 2 * v), empty, comm),
                };
                (re, im)
            })
            .collect();
        Self {
            sources,
            targets,
            charges,
        }
    }

    /// The charges as the FMM takes them, for the path `kind`.
    fn charges(&self, kind: ChargeKind) -> Vec<Charges<'_, f64>> {
        self.charges
            .iter()
            .map(|(re, im)| match kind {
                ChargeKind::Real => Charges::Real(re),
                ChargeKind::Complex => Charges::Complex { re, im },
            })
            .collect()
    }
}

/// The builder of a scenario.
fn builder(scenario: &Scenario) -> HelmholtzFmmBuilder {
    let k = Wavenumber::new(scenario.k.0, scenario.k.1).unwrap();
    HelmholtzFmmBuilder::new(k, scenario.degrees.clone())
        .charges(scenario.charges)
        .gradients(scenario.gradients)
        .domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
        .max_points_per_leaf(32)
}

/// Builds and evaluates; panics on an error.
fn run<'o>(
    builder: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    kind: ChargeKind,
    comm: &'o SimpleCommunicator,
) -> (HelmholtzFmm<'o, f64>, Output<f64>) {
    let mut fmm = builder
        .build(&inputs.sources, &inputs.targets, comm)
        .unwrap_or_else(|error| panic!("rank {}: the build: {error}", comm.rank()));
    let output = fmm
        .evaluate(&inputs.charges(kind))
        .unwrap_or_else(|error| panic!("rank {}: the evaluation: {error}", comm.rank()));
    (fmm, output)
}

/// The relative L2 errors of φ and ∇φ of vector v of `output` against the direct sum over
/// every rank's sources, at this rank's targets, summed over every rank.
fn errors(
    scenario: &Scenario,
    inputs: &Inputs,
    output: &Output<f64>,
    v: usize,
    comm: &SimpleCommunicator,
) -> (f64, f64) {
    let k = C::new(scenario.k.0, scenario.k.1);
    let (all_sources, _) = union_points(&inputs.sources, comm);
    let (re, im) = &inputs.charges[v];
    let (re, _) = union(re, comm);
    let (im, _) = union(im, comm);
    let charges: Vec<C> = re.iter().zip(&im).map(|(&r, &i)| C::new(r, i)).collect();
    let n = inputs.targets.len();
    let zero = C::new(0.0, 0.0);
    let mut phi = vec![zero; n];
    let mut grad = vec![[zero; 3]; n];
    p2p::direct_sum(
        k,
        &all_sources,
        &charges,
        &inputs.targets,
        &mut phi,
        Some(&mut grad),
    );
    let scale = 1.0 / (4.0 * PI);
    let potential = output.potential_of(v);
    let at = |planes: &[f64], i| planes[i];
    let mut sums = [0.0f64; 4];
    for i in 0..n {
        let got = C::new(potential.re[i], potential.im.map_or(0.0, |im| at(im, i)));
        let want = phi[i] * scale;
        sums[0] += (got - want).norm_sqr();
        sums[1] += want.norm_sqr();
        if let Some(g) = output.gradient_of(v) {
            for a in 0..3 {
                let got = C::new(g.re[i][a], g.im.map_or(0.0, |im| im[i][a]));
                let want = grad[i][a] * scale;
                sums[2] += (got - want).norm_sqr();
                sums[3] += want.norm_sqr();
            }
        }
    }
    let sums = sums.map(|s| global_sum(s, comm));
    let gradient = if scenario.gradients {
        (sums[2] / sums[3]).sqrt()
    } else {
        0.0
    };
    ((sums[0] / sums[1]).sqrt(), gradient)
}

/// On several ranks, the one-rank FMM of `builder` over the union of every rank's inputs
/// in rank order, built by every rank on `self_comm` (so that a failure there fails every
/// rank), whose output at each rank's targets must equal `output` bit for bit.
fn check_one_rank(
    what: &str,
    builder: &HelmholtzFmmBuilder,
    inputs: &Inputs,
    kind: ChargeKind,
    output: &Output<f64>,
    comm: &SimpleCommunicator,
) {
    if comm.size() == 1 {
        return;
    }
    let (sources, _) = union_points(&inputs.sources, comm);
    let (targets, first) = union_points(&inputs.targets, comm);
    let charges: Vec<(Vec<f64>, Vec<f64>)> = inputs
        .charges
        .iter()
        .map(|(re, im)| (union(re, comm).0, union(im, comm).0))
        .collect();
    let union_inputs = Inputs {
        sources,
        targets,
        charges,
    };
    // Every rank builds the reference, so that a failure there fails every rank.
    let one = SimpleCommunicator::self_comm();
    let (fmm, all) = run(&builder.clone().threads(1), &union_inputs, kind, &one);
    let reference = bits(&all);
    let n = union_inputs.targets.len();
    let columns = fmm.densities().output_columns();
    let gradients = output.gradient.is_some();
    // This rank's targets of every column, φ then ∇φ.
    let local = inputs.targets.len();
    let (phi, grad) = reference.split_at(columns * n);
    let mut want = Vec::new();
    for column in 0..columns {
        want.extend_from_slice(&phi[column * n + first..column * n + first + local]);
    }
    if gradients {
        for column in 0..columns {
            let at = 3 * (column * n + first);
            want.extend_from_slice(&grad[at..at + 3 * local]);
        }
    }
    let same = bits(output) == want;
    if !same {
        eprintln!(
            "rank {}: {what}: differs from the one-rank FMM",
            comm.rank()
        );
    }
    assert!(
        everywhere(same, comm),
        "{what}: not the one-rank FMM over the union in rank order, bit for bit"
    );
}

/// The scenario's checks.
fn check(scenario: &Scenario, comm: &SimpleCommunicator) {
    let start = Instant::now();
    let rank = comm.rank();
    let inputs = Inputs::new(scenario, 1, comm);
    let builder = builder(scenario);
    let kind = scenario.charges;
    let (fmm, output) = run(&builder, &inputs, kind, comm);

    let (phi, grad) = errors(scenario, &inputs, &output, 0, comm);
    if rank == 0 {
        println!(
            "{}: {} levels, degrees {:?}, leaf level {}, {} leaves on rank 0, largest leaf {}: \
             relative L2 φ {phi:.2e}, ∇φ {grad:.2e} (bound {:.1e})",
            scenario.name,
            fmm.nlevels(),
            fmm.degrees(),
            fmm.min_leaf_level(),
            fmm.nleaves(),
            fmm.max_leaf_points(),
            scenario.bound,
        );
    }
    assert!(
        phi < scenario.bound && grad < scenario.bound,
        "{}: φ {phi:.2e}, ∇φ {grad:.2e} against the direct sum, bound {:.1e}",
        scenario.name,
        scenario.bound
    );

    assert_same(
        &format!(
            "{}: the output pass against write_leaf_output",
            scenario.name
        ),
        &output,
        &fmm.reference_output(),
        comm,
    );
    drop(fmm);

    // 2 threads in release builds only (the debug budget).
    let threads: &[usize] = if cfg!(debug_assertions) {
        &[4]
    } else {
        &[2, 4]
    };
    for &threads in threads {
        let (_, threaded) = run(&builder.clone().threads(threads), &inputs, kind, comm);
        assert_same(
            &format!("{}: {threads} threads", scenario.name),
            &threaded,
            &output,
            comm,
        );
    }

    let (_, pairs) = run(&builder.clone().pair_calls(true), &inputs, kind, comm);
    assert_same(
        &format!("{}: pair by pair", scenario.name),
        &pairs,
        &output,
        comm,
    );

    check_one_rank(scenario.name, &builder, &inputs, kind, &output, comm);
    if rank == 0 {
        println!(
            "{}: checks passed in {:.1?}",
            scenario.name,
            start.elapsed()
        );
    }
}

/// Two vectors: each bit for bit its one-vector FMM, the second against its direct sum;
/// and a reused output.
fn check_vectors(scenario: &Scenario, comm: &SimpleCommunicator) {
    let inputs = Inputs::new(scenario, 2, comm);
    let builder = builder(scenario);
    let kind = scenario.charges;
    let (mut fmm, both) = run(&builder.clone().vectors(2), &inputs, kind, comm);
    for v in 0..2 {
        let one = Inputs {
            sources: inputs.sources.clone(),
            targets: inputs.targets.clone(),
            charges: vec![inputs.charges[v].clone()],
        };
        let (_, single) = run(&builder, &one, kind, comm);
        let column = |output: &Output<f64>, v: usize| {
            let phi = output.potential_of(v);
            let mut bits: Vec<u64> = phi
                .re
                .iter()
                .chain(phi.im.unwrap_or(&[]))
                .map(|x| x.to_bits())
                .collect();
            if let Some(g) = output.gradient_of(v) {
                let im = g.im.unwrap_or(&[]);
                bits.extend(g.re.iter().chain(im).flatten().map(|x| x.to_bits()));
            }
            bits
        };
        let same = column(&both, v) == column(&single, 0);
        assert!(
            everywhere(same, comm),
            "{}: vector {v} of two is not the one-vector FMM bit for bit",
            scenario.name
        );
    }
    // Vector 1 against its own direct sum.
    let (phi, grad) = errors(scenario, &inputs, &both, 1, comm);
    assert!(
        phi < scenario.bound && grad < scenario.bound,
        "{}: vector 1 of two: φ {phi:.2e}, ∇φ {grad:.2e} against the direct sum",
        scenario.name
    );
    // A reused output of another path is replaced; one of this path is rewritten.
    let mut reused = Output::<f64>::new(
        &nd_fmm_exec::helmholtz::Densities::new(
            nd_fmm_exec::helmholtz::WavenumberKind::Real,
            ChargeKind::Complex,
            1,
        ),
        3,
        false,
    );
    for _ in 0..2 {
        fmm.evaluate_into(&inputs.charges(kind), &mut reused)
            .expect("evaluate_into");
        assert_same(
            &format!("{}: evaluate_into", scenario.name),
            &reused,
            &both,
            comm,
        );
    }
    if comm.rank() == 0 {
        println!("{}: two vectors passed", scenario.name);
    }
}

/// Real charges against complex ones with Im q = 0.
fn check_real_charges(scenario: &Scenario, comm: &SimpleCommunicator) {
    let real = Scenario {
        charges: ChargeKind::Real,
        ..scenario.clone()
    };
    let inputs = Inputs::new(&real, 1, comm);
    let (_, as_real) = run(&builder(&real), &inputs, ChargeKind::Real, comm);
    let complex = builder(&Scenario {
        charges: ChargeKind::Complex,
        ..scenario.clone()
    });
    let (_, as_complex) = run(&complex, &inputs, ChargeKind::Complex, comm);
    let diff = |a: &[f64], b: &[f64]| -> [f64; 2] {
        let d = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        [d, b.iter().map(|y| y * y).sum()]
    };
    let [d, n] = diff(&as_real.potential, &as_complex.potential).map(|s| global_sum(s, comm));
    let relative = (d / n).sqrt();
    if comm.rank() == 0 {
        println!(
            "{}: real charges against complex ones with Im q = 0: relative L2 {relative:.1e}",
            scenario.name
        );
    }
    assert!(
        relative < 1e-13,
        "{}: real charges differ from complex ones with Im q = 0 by {relative:.1e}",
        scenario.name
    );
}

/// The errors, each on every rank alike (or, for one rank's input, on that rank and
/// `OtherRank` elsewhere).
fn check_errors(comm: &SimpleCommunicator) {
    let scenario = &scenarios()[1];
    let inputs = Inputs::new(scenario, 1, comm);
    let ok = builder(scenario);
    let rank = comm.rank();

    let yukawa =
        HelmholtzFmmBuilder::new(Wavenumber::new(0.0, 5.0).unwrap(), Degrees::Tolerance(1e-3));
    let error = yukawa
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Operator(OperatorError::Yukawa))
        ),
        "rank {rank}: Yukawa: {error:?}"
    );

    let k = Wavenumber::new(1.0, 0.0).unwrap();
    let error = HelmholtzFmmBuilder::new(k, Degrees::Tolerance(2.0))
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .err();
    assert!(
        matches!(error, Some(HelmholtzFmmError::Degree(_))),
        "rank {rank}: a tolerance of 2: {error:?}"
    );

    let error = builder(scenario)
        .c_leaf(0.0)
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .err();
    assert!(
        matches!(error, Some(HelmholtzFmmError::LeafWidth(_))),
        "rank {rank}: c_leaf = 0: {error:?}"
    );

    // A point that is not finite on rank 0 only: its error there, `OtherRank` elsewhere.
    let mut bad = inputs.sources.clone();
    if rank == 0 {
        bad[0][1] = f64::NAN;
    }
    let error = ok.build::<f64, _>(&bad, &inputs.targets, comm).err();
    if rank == 0 {
        assert!(
            matches!(
                error,
                Some(HelmholtzFmmError::Build(
                    nd_fmm_exec::fmm::FmmError::NonFinitePoint { .. }
                ))
            ),
            "rank 0: a point that is not finite: {error:?}"
        );
    } else {
        assert!(
            matches!(error, Some(HelmholtzFmmError::OtherRank)),
            "rank {rank}: another rank's point that is not finite: {error:?}"
        );
    }

    let error = ok.build::<f64, _>(&[], &[], comm).err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Build(
                nd_fmm_exec::fmm::FmmError::NoPoints
            ))
        ),
        "rank {rank}: no points: {error:?}"
    );

    let mut fmm = ok
        .build::<f64, _>(&inputs.sources, &inputs.targets, comm)
        .unwrap();
    let error = fmm.evaluate(&[]).err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Charges(DensityError::VectorCount { .. }))
        ),
        "rank {rank}: no charge vector: {error:?}"
    );
    let (re, _) = &inputs.charges[0];
    let error = fmm.evaluate(&[Charges::Real(re)]).err();
    assert!(
        matches!(
            error,
            Some(HelmholtzFmmError::Charges(DensityError::ChargeKind { .. }))
        ),
        "rank {rank}: real charges to a complex FMM: {error:?}"
    );
    if comm.size() > 1 {
        // Rank 0 alone passes a short vector.
        let (re, im) = &inputs.charges[0];
        let n = if rank == 0 { re.len() - 1 } else { re.len() };
        let error = fmm
            .evaluate(&[Charges::Complex {
                re: &re[..n],
                im: &im[..n],
            }])
            .err();
        if rank == 0 {
            assert!(
                matches!(
                    error,
                    Some(HelmholtzFmmError::Charges(DensityError::Length { .. }))
                ),
                "rank 0: a short vector: {error:?}"
            );
        } else {
            assert!(
                matches!(error, Some(HelmholtzFmmError::OtherRank)),
                "rank {rank}: another rank's short vector: {error:?}"
            );
        }
    }
    // The FMM still evaluates after refused inputs.
    fmm.evaluate(&inputs.charges(ChargeKind::Complex))
        .expect("an evaluation after the errors");
    if rank == 0 {
        println!("errors passed");
    }
}

/// The scenarios, small trees (a few thousand points, at most 32 per leaf).
fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "cube, kD = 20 + 2i, ε = 1e-3, complex charges, gradients, rank 0 empty",
            layout: Layout::Cube,
            n: (1000, None),
            k: (20.0, 2.0),
            degrees: Degrees::Tolerance(1e-3),
            bound: ERROR_FACTOR * 1e-3,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: true,
        },
        Scenario {
            name: "cube, kD = 1, ε = 1e-3, complex charges, gradients",
            layout: Layout::Cube,
            n: (1500, None),
            k: (1.0, 0.0),
            degrees: Degrees::Tolerance(1e-3),
            bound: ERROR_FACTOR * 1e-3,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: false,
        },
        Scenario {
            name: "clusters, kD = 20, p = 8, real charges, separate targets",
            layout: Layout::Clusters,
            n: (1000, Some(700)),
            k: (20.0, 0.0),
            degrees: Degrees::PerLevel(vec![8; 17]),
            bound: P8_BOUND,
            charges: ChargeKind::Real,
            gradients: false,
            debug: true,
            empty_rank0: false,
        },
        Scenario {
            name: "cube, kD = 20 + 2i, p = 8, complex charges, gradients",
            layout: Layout::Cube,
            n: (1500, None),
            k: (20.0, 2.0),
            degrees: Degrees::PerLevel(vec![8; 17]),
            bound: P8_BOUND,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: true,
            empty_rank0: false,
        },
        Scenario {
            name: "cube, kD = 20, ε = 1e-6, complex charges, gradients",
            layout: Layout::Cube,
            n: (2000, None),
            k: (20.0, 0.0),
            degrees: Degrees::Tolerance(1e-6),
            bound: ERROR_FACTOR * 1e-6,
            charges: ChargeKind::Complex,
            gradients: true,
            debug: false,
            empty_rank0: false,
        },
        Scenario {
            name: "clusters, kD = 100, ε = 1e-6, complex charges",
            layout: Layout::Clusters,
            n: (3000, None),
            k: (100.0, 0.0),
            degrees: Degrees::Tolerance(1e-6),
            bound: ERROR_FACTOR * 1e-6,
            charges: ChargeKind::Complex,
            gradients: false,
            debug: false,
            empty_rank0: false,
        },
    ]
}

#[test]
fn helmholtz_fmm() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("MPI initialises once per test executable");
    assert!(
        provided >= Threading::Funneled,
        "MPI provides {provided:?}, below Funneled"
    );
    let comm = universe.world();
    let start = Instant::now();
    let debug = cfg!(debug_assertions);
    for scenario in scenarios() {
        if debug && !scenario.debug {
            if comm.rank() == 0 {
                println!("{}: release builds only", scenario.name);
            }
            continue;
        }
        check(&scenario, &comm);
    }
    // Two vectors of complex and of real charges.
    check_vectors(&scenarios()[1], &comm);
    check_vectors(&scenarios()[2], &comm);
    check_real_charges(&scenarios()[2], &comm);
    check_real_charges(&scenarios()[3], &comm);
    check_errors(&comm);
    if comm.rank() == 0 {
        println!(
            "helmholtz_fmm on {} ranks: {:.1?}",
            comm.size(),
            start.elapsed()
        );
    }
}
