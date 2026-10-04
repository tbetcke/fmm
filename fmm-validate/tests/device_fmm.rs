//! A smoke run of the core of the `device_fmm` example (Phase 4 T13) on the CubeCL CPU
//! runtime: one small adaptive problem, the Plummer sphere at N = 3,000 with 32 points per
//! leaf (leaves on several levels, non-empty W and X lists, so every operator kind runs),
//! at p = 3 with gradients, against the host path of the same settings.
//!
//! Per precision (f64 under `Dense` and under `Rotation`, f32 under the device's static
//! rule):
//! - the device output within the FMM bounds of the host output (docs/phase4/README.md,
//!   "Accuracy measures"): relative L2 difference over every target and charge vector,
//!   φ and ∇φ, 1e-12 (f64) or 1e-5 (f32);
//! - each of the four errors against the direct sum (φ L2, φ max, ∇φ L2, ∇φ max: the
//!   root mean squares over the charge vectors at the sampled targets) within 0.1% (f64)
//!   or 5% (f32) of the host's;
//! - every kind on the device, one upload, one download and one sync per evaluation, two
//!   evaluations bit-identical, the stages timed synchronously, the strategy as asked.
//!
//! Timings are measured but never asserted. Its own executable, because it initialises
//! MPI; feature `cpu`:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-validate --features cpu --release --test device_fmm
//! ```
#![cfg(feature = "cpu")]

use mpi::topology::SimpleCommunicator;
use nd_fmm_exec::fmm::{Backend, OperatorKind};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_validate::calibration::{Precision, Reference};
use nd_fmm_validate::device_fmm::{Settings, bounds, compare, error_ratios, measure};
use nd_fmm_validate::fmm_accuracy::{Config, Distribution};

/// The smoke problem: an adaptive Plummer tree small enough for a debug run.
const SMOKE: Config = Config {
    distribution: Distribution::Plummer,
    n: 3000,
    max_level: 16,
    max_points_per_leaf: 32,
    sampled: 200,
    charge_vectors: 2,
    seed: 0x713,
};

/// The degree.
const P: usize = 3;

#[test]
fn device_fmm_core_on_the_cpu_runtime() {
    let universe = mpi::initialize().expect("this test owns MPI initialization");
    let comm = universe.world();
    let reference = Reference::new(&SMOKE);
    for (precision, strategy) in [
        (Precision::F64, M2lStrategy::Dense),
        (Precision::F64, M2lStrategy::Rotation),
        (Precision::F32, M2lStrategy::Auto),
    ] {
        check(&reference, precision, strategy, &comm);
    }
    println!("backends run: cpu (CubeCL CPU runtime); not run: metal, cuda (type-checked)");
}

/// Measures the host and the CPU runtime at `precision` and `strategy` and checks the
/// device against the host.
fn check(
    reference: &Reference,
    precision: Precision,
    strategy: M2lStrategy,
    comm: &SimpleCommunicator,
) {
    let host_strategy = match strategy {
        // The device's static rule in f32 is `Dense` (README decision 13).
        M2lStrategy::Auto => M2lStrategy::Dense,
        fixed => fixed,
    };
    let host = Settings {
        repeats: 2,
        ..Settings::host(1, host_strategy)
    };
    let device = Settings {
        repeats: 2,
        ..Settings::device(Backend::Cpu, strategy)
    };
    let host = measure(&SMOKE, reference, (precision, P), &host, comm).expect("the host builds");
    let device =
        measure(&SMOKE, reference, (precision, P), &device, comm).expect("the device builds");
    let (potential, gradient) = compare(&device, &host);
    let ratios = error_ratios(&device.run, &host.run);
    let (output_bound, error_bound) = bounds(precision);
    let run = device
        .run
        .device
        .as_ref()
        .expect("a device run reports its device");
    let [uploads, _, downloads, _, launches, syncs, _] = run.evaluation;
    println!(
        "{} {} p = {P}: device - host φ {potential:.2e}, ∇φ {gradient:.2e}; error ratios \
         {ratios:.4?}; {launches} launches; evaluate host {:.2} ms, device {:.2} ms; {}",
        precision.name(),
        device.settings,
        host.evaluation.as_secs_f64() * 1e3,
        device.evaluation.as_secs_f64() * 1e3,
        run.device,
    );

    assert!(
        potential <= output_bound && gradient <= output_bound,
        "device - host {potential:e}, {gradient:e} above {output_bound:e}"
    );
    for ratio in ratios {
        assert!(
            (ratio - 1.0).abs() <= error_bound,
            "error ratios {ratios:?} outside 1 ± {error_bound}"
        );
    }
    assert_eq!(
        run.on_device,
        OperatorKind::ALL.to_vec(),
        "every kind on the device"
    );
    assert_eq!(
        (uploads, downloads, syncs),
        (1, 1, 1),
        "the charges up, the output down, one sync"
    );
    assert!(
        device.identical && host.identical,
        "two evaluations bit-identical"
    );
    assert!(device.synchronous && !host.synchronous);
    assert_eq!(device.run.strategy, host_strategy, "the strategy as asked");
    assert!(device.tuning.is_none(), "no tuning without a cache");
    assert!(
        device.run.lists.w > 0 && device.run.lists.x > 0,
        "an adaptive tree with W and X lists"
    );
}
