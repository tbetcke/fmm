//! The dipole gate (Phase 6 T8, C6.4; docs/phase6/T8-dipoles.md, the gate signed off on
//! 2026-10-17): the dipole FMM, and the FMM of a charge and a dipole at every point,
//! against the direct sum at the C3.2 cube and the C3.3 Plummer sphere, f64 at p = 3, 8
//! and 12 and f32 at p = 3 and 8, on any number of ranks (run at 1, 2 and 4).
//!
//! The problems are those of the charge gates: the cube of `tests/accuracy.rs` (N = 10⁵
//! uniform in [−1, 1)³, seed `0xc32`, a uniform level-4 tree: `max_level` 4, one point per
//! leaf) and the Plummer sphere of `tests/adaptive.rs` (N = 10⁵, scale 0.1, seed `0xc33`,
//! `max_level` 16, 64 points per leaf), sources equal to targets, 1,000 sampled targets.
//! Eight strength vectors per problem: vector k has charges uniform in [−1, 1) from seed
//! `seed + 1 + k` (the charge gates' vectors) and moments with components uniform in
//! [−1, 1) from seed `seed + 101 + k`. Each rank passes every `size`-th point and its
//! strengths; the rank that passed a sampled point computes its direct sum
//! (`direct_sum_dipoles`, plus `direct_sum` with charges) and its error terms, reduced over
//! the ranks.
//!
//! Error measure (docs/phase3/README.md, "Error measures"): the relative L2 error of φ at
//! the sampled targets against the direct sum divided by 4π, as the root mean square over
//! the eight vectors. Gates (signed off with the brief): on the cube, at most twice the
//! source type's own single-translation prediction ([`PREDICTION`]: the P2M → M2L → L2P row
//! of `nd-fmm-validate`'s `accuracy` example with `--sources`; the charge bound of Phase 3
//! does not apply, the dipole error being 30–150 times the charge error at the same p); on
//! the Plummer sphere, at most twice the cube's error at the same source type, precision
//! and p (the C3.3 rule). Every case is measured and printed before the gates are checked.
//!
//! With every kind on the device (Phase 6 T8d): with a device feature each case is repeated
//! on the device backends compiled in, the CPU runtime in f64, Metal in f32 and CUDA in
//! both, every kind on the device (the default placement), and checked against the host run
//! of the same case with the device-against-host bounds of docs/phase4/README.md ("Accuracy
//! measures", FMM): the device's error within 0.1% (f64) and 5% (f32) of the host's, and
//! the device's φ within 1e-12 (f64) and 1e-5 (f32) of the host's (relative L2 over every
//! target, reduced over the ranks, the root mean square over the vectors). The device runs
//! with as many threads as the machine has cores (the CPU runtime's units per cube).
//!
//! Its own executable, because it initialises MPI; ignored, because it needs release mode:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release --test dipoles -- --ignored --nocapture
//! mpirun -n 2 target/release/deps/dipoles-<hash> --ignored --nocapture   # and -n 4
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release --test dipoles -- \
//!     --ignored --nocapture                                               # and metal, cuda
//! ```

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{Backend, FmmBuilder, SourceType, Sources};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::{direct_sum, direct_sum_dipoles};
use nd_fmm_tables::cache::Stored;

/// Points per problem.
const N: usize = 100_000;

/// Sampled targets per problem.
const SAMPLED: usize = 1000;

/// Strength vectors per problem.
const VECTORS: u64 = 8;

/// The predictions of one source type: (p, f64, f32) per degree.
type Predictions = [(usize, f64, f64); 3];

/// The single-translation prediction of the relative L2 error of φ, P2M → M2L → L2P over
/// the pooled targets of the 316 offsets (`nd-fmm-validate`'s `accuracy` example,
/// `--sources dipoles` and `--sources charges+dipoles`, f64 and f32), per source type and
/// degree: (p, f64, f32), f32 not at p = 12 (beyond its range, CONVENTIONS §3.9). Seed
/// 20260929 (`Config::STANDARD`), moments from the seed `^ MOMENT_SEED`; f32 agrees with f64
/// to the three printed digits at p = 3 and 8.
const PREDICTION: [(SourceType, Predictions); 2] = [
    (
        SourceType::Dipoles,
        [
            (3, 5.84e-2, 5.84e-2),
            (8, 5.15e-4, 5.15e-4),
            (12, 2.34e-5, f64::NAN),
        ],
    ),
    (
        SourceType::ChargesAndDipoles,
        [
            (3, 1.74e-2, 1.74e-2),
            (8, 1.50e-4, 1.50e-4),
            (12, 6.81e-6, f64::NAN),
        ],
    ),
];

/// SplitMix64, as in the other tests.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform in 0..n (n ≤ 2³², with a negligible bias).
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
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
    v.map(|c| r * c / norm)
}

/// The points and the sampled targets of a problem (as `tests/accuracy.rs` and
/// `tests/adaptive.rs` draw them), and its seed.
fn problem(name: &str) -> (Vec<[f64; 3]>, Vec<usize>, u64) {
    let seed = if name == "cube" { 0xc32 } else { 0xc33 };
    let mut rng = SplitMix64(seed);
    let points: Vec<[f64; 3]> = if name == "cube" {
        (0..N)
            .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
            .collect()
    } else {
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
    };
    let mut order: Vec<usize> = (0..N).collect();
    for i in 0..SAMPLED {
        let j = i + rng.below(N - i);
        order.swap(i, j);
    }
    order.truncate(SAMPLED);
    (points, order, seed)
}

/// The tree of a problem.
fn builder<T>(name: &str, p: usize, kind: SourceType) -> FmmBuilder<T> {
    let builder = FmmBuilder::<T>::new(p).sources(kind);
    if name == "cube" {
        builder.max_level(4).max_points_per_leaf(1)
    } else {
        builder.max_level(16).max_points_per_leaf(64)
    }
}

/// One measured case: on the host, or on a device backend (Phase 6 T8d) with its φ's
/// difference from the host's.
struct Case {
    name: &'static str,
    kind: SourceType,
    precision: &'static str,
    p: usize,
    backend: Backend,
    error: f64,
    difference: f64,
}

/// The device backends compiled in that run the gate in `precision` (Phase 6 T8d): the CPU
/// runtime in f64, Metal in f32, CUDA in both.
fn device_backends(precision: &str) -> Vec<Backend> {
    let mut backends = Vec::new();
    if precision == "f64" && Backend::Cpu.is_compiled() {
        backends.push(Backend::Cpu);
    }
    if precision == "f32" && Backend::Metal.is_compiled() {
        backends.push(Backend::Metal);
    }
    if Backend::Cuda.is_compiled() {
        backends.push(Backend::Cuda);
    }
    backends
}

/// The root mean square error of φ at the sampled targets over the vectors, for the FMM
/// of `name` with `kind` at degree p in T on the host; and the same on each of `devices`
/// with the root mean square over the vectors of the relative L2 difference of its φ from
/// the host's over every target.
#[allow(clippy::too_many_arguments)]
fn measure<T: Stored + SimdScalar + Equivalence + Default>(
    name: &str,
    kind: SourceType,
    p: usize,
    (points, local): (&[[f64; 3]], &[usize]),
    vectors: &[(Vec<f64>, Vec<[f64; 3]>)],
    exact: &[Vec<f64>],
    sampled: &[(usize, usize)],
    (devices, comm): (&[Backend], &SimpleCommunicator),
) -> (f64, Vec<(Backend, f64, f64)>) {
    let mine: Vec<[f64; 3]> = local.iter().map(|&i| points[i]).collect();
    let mut fmm = builder::<T>(name, p, kind)
        .build(&mine, &mine, comm)
        .unwrap_or_else(|error| panic!("{name}, {kind}, p = {p}: {error}"));
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut on_devices: Vec<_> = devices
        .iter()
        .map(|&backend| {
            let fmm = builder::<T>(name, p, kind)
                .backend(backend)
                .threads(threads)
                .build(&mine, &mine, comm)
                .unwrap_or_else(|error| panic!("{backend}, {name}, {kind}, p = {p}: {error}"));
            (backend, fmm)
        })
        .collect();
    let error = |output: &[T], exact: &[f64]| -> f64 {
        let mut sums = [0.0f64; 2];
        for (&(position, _), e) in sampled.iter().zip(exact) {
            let phi = RealScalar::to_f64(output[position]);
            sums[0] += (phi - e).powi(2);
            sums[1] += e * e;
        }
        let mut total = [0.0f64; 2];
        comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
        total[0] / total[1]
    };
    let mut squares = 0.0;
    let mut device_squares = vec![(0.0, 0.0); devices.len()];
    for ((charges, dipoles), exact) in vectors.iter().zip(exact) {
        let q: Vec<T> = local.iter().map(|&i| T::from_f64(charges[i])).collect();
        let d: Vec<[T; 3]> = local.iter().map(|&i| dipoles[i].map(T::from_f64)).collect();
        let sources = match kind {
            SourceType::Dipoles => Sources::dipoles(&d),
            _ => Sources::charges_and_dipoles(&q, &d),
        };
        let output = fmm.evaluate_sources(&sources).expect("the FMM evaluates");
        squares += error(&output.potential, exact);
        for ((backend, device), squares) in on_devices.iter_mut().zip(&mut device_squares) {
            let got = device
                .evaluate_sources(&sources)
                .unwrap_or_else(|error| panic!("{backend}: the device evaluates: {error}"));
            squares.0 += error(&got.potential, exact);
            let mut sums = [0.0f64; 2];
            for (&g, &h) in got.potential.iter().zip(&output.potential) {
                let (g, h) = (RealScalar::to_f64(g), RealScalar::to_f64(h));
                sums[0] += (g - h).powi(2);
                sums[1] += h * h;
            }
            let mut total = [0.0f64; 2];
            comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
            squares.1 += total[0] / total[1];
        }
    }
    let count = vectors.len() as f64;
    let on_devices = devices
        .iter()
        .zip(device_squares)
        .map(|(&backend, (e, d))| (backend, (e / count).sqrt(), (d / count).sqrt()))
        .collect();
    ((squares / count).sqrt(), on_devices)
}

#[test]
#[ignore = "release mode: N = 10^5, eight direct sums at 1,000 targets per problem"]
fn dipole_fmm_within_its_gates() {
    let (universe, _) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    let comm = universe.world();
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let mut cases = Vec::new();
    for name in ["cube", "plummer"] {
        let (points, sample, seed) = problem(name);
        let local: Vec<usize> = (0..N).filter(|i| i % size == rank).collect();
        // The sampled targets this rank passed: (position in its output, point).
        let sampled: Vec<(usize, usize)> = sample
            .iter()
            .filter(|&&i| i % size == rank)
            .map(|&i| (i / size, i))
            .collect();
        let targets: Vec<[f64; 3]> = sampled.iter().map(|&(_, i)| points[i]).collect();
        let vectors: Vec<(Vec<f64>, Vec<[f64; 3]>)> = (0..VECTORS)
            .map(|k| {
                let mut rng = SplitMix64(seed + 1 + k);
                let charges = (0..N).map(|_| rng.range(-1.0, 1.0)).collect();
                let mut rng = SplitMix64(seed + 101 + k);
                let dipoles = (0..N)
                    .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
                    .collect();
                (charges, dipoles)
            })
            .collect();
        for (kind, rows) in PREDICTION {
            let exact: Vec<Vec<f64>> = vectors
                .iter()
                .map(|(charges, dipoles)| {
                    let mut phi = vec![0.0; targets.len()];
                    if kind.has_charges() {
                        direct_sum(&points, charges, &targets, &mut phi, None);
                    }
                    direct_sum_dipoles(&points, dipoles, &targets, &mut phi, None);
                    phi.iter()
                        .map(|v| v / (4.0 * std::f64::consts::PI))
                        .collect()
                })
                .collect();
            for (p, prediction64, prediction32) in rows {
                for precision in ["f64", "f32"] {
                    let prediction = if precision == "f64" {
                        prediction64
                    } else {
                        prediction32
                    };
                    if prediction.is_nan() {
                        continue;
                    }
                    let data = (&points[..], &local[..]);
                    let devices = device_backends(precision);
                    let on = (&devices[..], &comm);
                    let (error, on_devices) = if precision == "f64" {
                        measure::<f64>(name, kind, p, data, &vectors, &exact, &sampled, on)
                    } else {
                        measure::<f32>(name, kind, p, data, &vectors, &exact, &sampled, on)
                    };
                    if rank == 0 {
                        eprintln!(
                            "{name}, {kind}, {precision}, p = {p}, {size} rank(s): relative L2 \
                             error of φ {error:.3e} (root mean square over {VECTORS} vectors), \
                             prediction {prediction:.2e}, ratio {:.2}",
                            error / prediction
                        );
                        for &(backend, device, difference) in &on_devices {
                            eprintln!(
                                "  {backend}, every kind on the device: error {device:.3e} \
                                 ({:.4} of the host's), φ {difference:.1e} from the host's",
                                device / error
                            );
                        }
                    }
                    cases.push(Case {
                        name,
                        kind,
                        precision,
                        p,
                        backend: Backend::Host,
                        error,
                        difference: 0.0,
                    });
                    for (backend, device, difference) in on_devices {
                        cases.push(Case {
                            name,
                            kind,
                            precision,
                            p,
                            backend,
                            error: device,
                            difference,
                        });
                    }
                }
            }
        }
    }
    let mut failures = Vec::new();
    for case in &cases {
        if case.backend != Backend::Host {
            // The device-against-host bounds of docs/phase4/README.md (Phase 6 T8d).
            let host = cases
                .iter()
                .find(|c| {
                    c.backend == Backend::Host
                        && c.name == case.name
                        && c.kind == case.kind
                        && c.precision == case.precision
                        && c.p == case.p
                })
                .expect("the host case");
            let (error_bound, difference_bound) = if case.precision == "f64" {
                (1e-3, 1e-12)
            } else {
                (5e-2, 1e-5)
            };
            if (case.error - host.error).abs() > error_bound * host.error
                || case.difference > difference_bound
            {
                failures.push(format!(
                    "{}, {}, {}, {}, p = {}: error {:e} against the host's {:e} (within {} of \
                     it), φ {:e} from the host's (bound {difference_bound:e})",
                    case.backend,
                    case.name,
                    case.kind,
                    case.precision,
                    case.p,
                    case.error,
                    host.error,
                    error_bound,
                    case.difference
                ));
            }
            continue;
        }
        let (_, rows) = PREDICTION.iter().find(|(k, _)| *k == case.kind).unwrap();
        let &(_, p64, p32) = rows.iter().find(|r| r.0 == case.p).unwrap();
        let prediction = if case.precision == "f64" { p64 } else { p32 };
        let bound = if case.name == "cube" {
            2.0 * prediction
        } else {
            let cube = cases
                .iter()
                .find(|c| {
                    c.name == "cube"
                        && c.backend == Backend::Host
                        && c.kind == case.kind
                        && c.precision == case.precision
                        && c.p == case.p
                })
                .expect("the cube case");
            2.0 * cube.error
        };
        if case.error > bound {
            failures.push(format!(
                "{}, {}, {}, p = {}: {:e} exceeds {bound:e}",
                case.name, case.kind, case.precision, case.p, case.error
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
