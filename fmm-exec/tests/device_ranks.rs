//! The C5.1 device gate (Phase 5 T8; docs/phase5/README.md, "Exit gate"; docs/design/
//! distributed-fmm.md §7): the device FMM on several ranks against the host FMM on the same
//! ranks and against the one-rank host run, from a seeded random share of the points per
//! rank.
//!
//! Problems: the cube [−1, 1)³ and the Plummer sphere of scale a = 0.1 of `tests/device_fmm.rs`
//! (the rules of `nd_fmm_validate::fmm_accuracy::Distribution`, seed `0xc33`), sources equal
//! to targets, `max_level` 16 and 64 points per leaf, gradients on, the charges uniform in
//! [−1, 1) (seed `0xc33 + 1`; in f32 rounded), the default strategy and device layouts, one
//! thread per rank. Point i is passed on rank `hash(i) % P`. The workload points, one per
//! backend compiled in:
//! - the CPU runtime (feature `cpu`): N = 10⁴, f64 at p = 8 and f32 at p = 3;
//! - Metal (feature `metal`, by hand outside the macOS sandbox, 2 ranks sharing the GPU):
//!   N = 10⁵, f32 at p = 3 and 8;
//! - CUDA (feature `cuda`, by hand on locust, 2 and 4 ranks sharing the H100): N = 10⁵, f32
//!   at p = 3 and 8 and f64 at p = 8.
//!
//! At each point, on every rank:
//! - `tests/device_common::check_backend` against the host `Fmm` of the same settings on the
//!   same ranks: every kind on the host fallback bit for bit the host path; the default
//!   placement (every kind on the device) within the Phase 4 FMM bounds (relative L2 over
//!   the rank's targets 1e-12 in f64, 1e-5 in f32, φ and ∇φ), and so the multipoles and
//!   locals of every level; two evaluations and two builds bit-identical; the transfers,
//!   launches and syncs of every evaluation the formula of the design's §7.2
//!   (`device_common::expected_evaluation`, from `Fmm::exchange_sizes`);
//! - **against the one-rank host run**: the device output over every rank's targets, in
//!   point order, within the same bounds of the one-rank host `Fmm` over all the points in
//!   their own order (relative L2, φ and ∇φ; requirement 2: the host on P ranks lies within
//!   100 u_T of that run, design §5.4, far inside the bound). Rank 0 gathers and checks.
//!
//! **Transfers** (the T8 report): the cube at N = 10⁵, f32, p = 8 on the CPU runtime (and as
//! a workload point on Metal and CUDA) builds the device `Fmm` alone and checks its
//! evaluation against the formula and the one-rank host run. For every point rank 0 prints
//! each rank's transfers per evaluation by data kind (the charges, the ghost sources, the
//! sent multipoles, the coarse blocks, the received multipoles, the output: calls and
//! bytes), its syncs and launches, whether they equal the formula, and the device memory
//! of its exchange buffers and ghost slots (`DeviceReport::exchange_bytes`,
//! `ghost_bytes`).
//!
//! The device on several ranks is a correctness check only (docs/phase5/README.md,
//! decision 3): ranks share one GPU and are never timed here. The verdict is all-reduced
//! before the test asserts, so the test fails on every rank or on none, except that a
//! failed check inside `check_backend` panics on its rank (the external timeout ends the
//! run then).
//!
//! Its own executable, because it initialises MPI; ignored, because it needs release mode.
//! By hand at 2 and 4 ranks, under an external timeout (macOS loopback flags of the root
//! `CLAUDE.md`; none on locust):
//!
//! ```text
//! cargo test -p nd-fmm-exec --features cpu --release --test device_ranks --no-run
//! RUST_MIN_STACK=8388608 timeout 1800 mpirun --mca btl_tcp_if_include lo0 \
//!     --mca oob_tcp_if_include lo0 -n 4 target/release/deps/device_ranks-<hash> \
//!     --ignored --nocapture
//! ```
//!
//! Metal: build with `--features metal` inside the sandbox, run outside it at 2 ranks;
//! CUDA: `--features cuda` on locust through `tools/gh200/remote.sh`.

#![cfg(feature = "gpu")]

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::datatype::PartitionMut;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::device::DataKind;
use nd_fmm_exec::fmm::{Backend, Fmm, FmmBuilder, OperatorKind, Output};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;

mod device_common;
mod kind_common;

/// The seed of the points; the charges use `SEED + 1`.
const SEED: u64 = 0xc33;

/// The seed of the random share.
const SHARE_SEED: u64 = 0x05ee_dc58;

/// The data kinds of the transfer table, in its column order.
const TABLE_KINDS: [DataKind; 6] = [
    DataKind::Charges,
    DataKind::GhostSources,
    DataKind::SentMultipoles,
    DataKind::CoarseMultipoles,
    DataKind::ReceivedMultipoles,
    DataKind::Output,
];

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
    v.map(|vi| r * (vi / norm))
}

/// The `n` points of the distribution `name`, as `tests/device_fmm.rs` draws them.
fn points(name: &str, n: usize) -> Vec<[f64; 3]> {
    let mut rng = SplitMix64(SEED);
    match name {
        "cube" => (0..n)
            .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
            .collect(),
        "plummer" => {
            let mass = 1000.0 / 101.0f64.powf(1.5);
            (0..n)
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
        _ => unreachable!("{name}"),
    }
}

/// The rank that point `i` is passed on: a seeded hash of i, modulo `size`.
fn share(i: usize, size: usize) -> usize {
    (SplitMix64(SHARE_SEED ^ (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)).next_u64()
        % size as u64) as usize
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

/// The relative L2 difference of `a` from `b`.
fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
    let (mut d2, mut b2) = (0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        d2 += (x - y).powi(2);
        b2 += y * y;
    }
    if b2 > 0.0 {
        (d2 / b2).sqrt()
    } else {
        d2.sqrt()
    }
}

/// An output as f64: φ, and ∇φ flattened.
fn as_f64<T: RealScalar>(output: &Output<T>) -> (Vec<f64>, Vec<f64>) {
    let f = |v: &[T]| v.iter().map(|&x| RealScalar::to_f64(x)).collect::<Vec<_>>();
    (
        f(&output.potential),
        f(output.gradient.as_ref().expect("gradients").as_flattened()),
    )
}

/// The numbers of one rank's row of the transfer table: per kind of [`TABLE_KINDS`] its
/// calls and bytes (up or down), then syncs, launches, whether every count equals the
/// formula, the exchange buffers' and the ghost slots' bytes.
fn transfer_row<T: Stored + SimdScalar + Equivalence + Default>(fmm: &Fmm<'_, T>) -> Vec<u64> {
    let counters = fmm.device_counters().expect("a device backend");
    let report = fmm.device_report().expect("a device backend");
    let expected = device_common::expected_evaluation(fmm, &OperatorKind::ALL);
    let mut row = Vec::new();
    for kind in TABLE_KINDS {
        let t = counters.evaluation_traffic.get(kind);
        row.push(t.uploads + t.downloads);
        row.push(t.upload_bytes + t.download_bytes);
    }
    let equal = DataKind::ALL
        .iter()
        .all(|&k| counters.evaluation_traffic.get(k) == expected.traffic[k as usize])
        && counters.evaluation.syncs == expected.syncs
        && counters.evaluation.launches == expected.launches;
    row.extend([
        counters.evaluation.syncs,
        counters.evaluation.launches,
        u64::from(equal),
        report.exchange_bytes,
        report.ghost_bytes,
    ]);
    row
}

/// One workload point (module documentation): `check_backend` if `full`, then the device
/// `Fmm` against the one-rank host run, and the transfer table. Appends rank 0's failures.
#[allow(clippy::too_many_arguments)]
fn run<T: Stored + SimdScalar + Equivalence + Default>(
    backend: Backend,
    name: &'static str,
    n: usize,
    p: usize,
    full: bool,
    failures: &mut Vec<String>,
    comm: &SimpleCommunicator,
) {
    let (rank, size) = (comm.rank(), comm.size() as usize);
    let precision = if size_of::<T>() == 4 { "f32" } else { "f64" };
    let what = format!("{backend}, {name}, N = {n}, {precision}, p = {p}, {size} ranks");
    let all_points = points(name, n);
    let mut rng = SplitMix64(SEED + 1);
    let all_charges: Vec<T> = (0..n).map(|_| T::from_f64(rng.range(-1.0, 1.0))).collect();
    let mine: Vec<usize> = (0..n)
        .filter(|&i| share(i, size) == rank as usize)
        .collect();
    let local: Vec<[f64; 3]> = mine.iter().map(|&i| all_points[i]).collect();
    let charges: Vec<T> = mine.iter().map(|&i| all_charges[i]).collect();
    let builder = FmmBuilder::<T>::new(p)
        .max_level(16)
        .max_points_per_leaf(64)
        .gradients(true)
        .threads(1);
    let bound = device_common::fmm_bound::<T>();

    // Against the host on the same ranks.
    let checked = if full {
        let mut host = builder
            .build(&local, &local, comm)
            .unwrap_or_else(|error| panic!("rank {rank}, {what}: the host FMM: {error}"));
        let host_output = host.evaluate(&charges).expect("the host FMM evaluates");
        let d = device_common::check_backend(
            &builder,
            (&local, &local),
            &charges,
            &mut host,
            &host_output,
            backend,
            comm,
        );
        // The worst rank's differences.
        let mut worst = [0.0f64; 4];
        comm.all_reduce_into(
            &[d.potential, d.gradient, d.multipoles, d.locals][..],
            &mut worst[..],
            SystemOperation::max(),
        );
        Some(worst)
    } else {
        None
    };

    // The device on P ranks against the one-rank host run over the points in their order.
    let mut device = builder
        .clone()
        .backend(backend)
        .build(&local, &local, comm)
        .unwrap_or_else(|error| panic!("rank {rank}, {what}: the device FMM: {error}"));
    let report = device.device_report().expect("a device backend");
    assert_eq!(
        (report.ranks.rank, report.ranks.ranks),
        (rank as usize, size),
        "rank {rank}: {report}"
    );
    let output = device.evaluate(&charges).expect("the device FMM evaluates");
    let row = transfer_row(&device);
    let table = union(&row, comm);
    let (phi, grad) = as_f64(&output);
    let (phi, grad) = (union(&phi, comm), union(&grad, comm));
    let order = union(&mine.iter().map(|&i| i as u64).collect::<Vec<u64>>(), comm);
    drop(device);
    if rank == 0 {
        let one = SimpleCommunicator::self_comm();
        let mut reference = builder
            .build(&all_points, &all_points, &one)
            .expect("the one-rank host FMM builds");
        let (rphi, rgrad) = as_f64(&reference.evaluate(&all_charges).expect("evaluates"));
        let (mut pphi, mut pgrad) = (vec![0.0; n], vec![0.0; 3 * n]);
        for (u, &i) in order.iter().enumerate() {
            let i = i as usize;
            pphi[i] = phi[u];
            pgrad[3 * i..3 * i + 3].copy_from_slice(&grad[3 * u..3 * u + 3]);
        }
        let against_one = (relative_l2(&pphi, &rphi), relative_l2(&pgrad, &rgrad));
        let same_ranks = checked.map_or_else(
            || "not run".to_owned(),
            |w| {
                format!(
                    "φ {:.1e} / ∇φ {:.1e} (multipoles {:.1e}, locals {:.1e})",
                    w[0], w[1], w[2], w[3]
                )
            },
        );
        eprintln!(
            "{what}: the device against the host on the same ranks {same_ranks}; against the \
             one-rank host run φ {:.1e} / ∇φ {:.1e} (bound {bound:.0e})",
            against_one.0, against_one.1
        );
        if against_one.0 > bound || against_one.1 > bound {
            failures.push(format!(
                "{what}: {against_one:?} from the one-rank host run, bound {bound:e}"
            ));
        }
        eprintln!(
            "| rank | charges up | ghost sources up | sent multipoles down | coarse blocks up \
             | received multipoles up | output down | syncs | launches | = formula | exchange \
             buffers B | ghost slots B |"
        );
        eprintln!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
        let width = table.len() / size;
        for (r, row) in table.chunks(width).enumerate() {
            let kinds: Vec<String> = row[..2 * TABLE_KINDS.len()]
                .chunks(2)
                .map(|c| format!("{} × {} B", c[0], c[1]))
                .collect();
            let rest = &row[2 * TABLE_KINDS.len()..];
            eprintln!(
                "| {r} | {} | {} | {} | {} | {} | {} |",
                kinds.join(" | "),
                rest[0],
                rest[1],
                if rest[2] == 1 { "yes" } else { "NO" },
                rest[3],
                rest[4]
            );
            if rest[2] != 1 {
                failures.push(format!(
                    "{what}: rank {r}'s transfers differ from the formula"
                ));
            }
        }
    }
}

#[test]
#[ignore = "release mode, by hand at 2 and 4 ranks; Metal outside the sandbox, CUDA on locust"]
fn device_on_several_ranks() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    let (rank, size) = (comm.rank(), comm.size());
    let mut failures = Vec::new();
    let mut ran = Vec::new();
    if Backend::Cpu.is_compiled() {
        for name in ["cube", "plummer"] {
            run::<f64>(Backend::Cpu, name, 10_000, 8, true, &mut failures, &comm);
            run::<f32>(Backend::Cpu, name, 10_000, 3, true, &mut failures, &comm);
        }
        run::<f32>(
            Backend::Cpu,
            "cube",
            100_000,
            8,
            false,
            &mut failures,
            &comm,
        );
        ran.push("cpu (f64 p = 8, f32 p = 3 at N = 10^4; transfers at N = 10^5, f32 p = 8)");
    }
    if Backend::Metal.is_compiled() {
        let info = Backend::Metal
            .probe()
            .unwrap_or_else(|reason| panic!("Metal: {reason} (run outside the macOS sandbox)"));
        if rank == 0 {
            eprintln!("Metal: {info}");
        }
        for name in ["cube", "plummer"] {
            for p in [3, 8] {
                run::<f32>(Backend::Metal, name, 100_000, p, true, &mut failures, &comm);
            }
        }
        ran.push("metal (f32 p = 3 and 8 at N = 10^5)");
    }
    if Backend::Cuda.is_compiled() {
        for name in ["cube", "plummer"] {
            for p in [3, 8] {
                run::<f32>(Backend::Cuda, name, 100_000, p, true, &mut failures, &comm);
            }
            run::<f64>(Backend::Cuda, name, 100_000, 8, true, &mut failures, &comm);
        }
        ran.push("cuda (f32 p = 3 and 8, f64 p = 8 at N = 10^5)");
    }
    let mut failed = 0u64;
    comm.all_reduce_into(
        &(failures.len() as u64),
        &mut failed,
        SystemOperation::max(),
    );
    if rank == 0 {
        eprintln!(
            "C5.1 device gate on {size} ranks × 1 thread; backends run: {}; {}",
            if ran.is_empty() {
                "none (no device backend compiled in)".to_owned()
            } else {
                ran.join(", ")
            },
            kind_common::summary()
        );
    }
    assert!(
        failed == 0,
        "rank {rank}: {failed} failures (rank 0 lists them): {}",
        failures.join("; ")
    );
}
