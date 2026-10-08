//! The whole output at the C3.2 size, unchanged by Phase 4S T9 (docs/phase4s/T9-output-pass.md,
//! "Whole outputs unchanged"), shared by `tests/accuracy.rs` (the host) and
//! `tests/device_fmm.rs` (the device backends).
//!
//! On the C3.2 tree (N = 10⁵ points uniform in [−1, 1)³, a uniform level-4 tree), the
//! default build at p = 6, gradients off and on, evaluates its first charge vector. The
//! output equals the output pass before T9 on the same build (`Fmm::reference_output`, the
//! test oracle) bit for bit, and where the default runs the output pass on the device, a
//! build with `output_pass(OutputPass::Host)` gives the same bits too. Each line printed
//! carries an FNV-1a hash of the output's bits, for comparing builds and code versions by
//! hand: it depends on the machine (the P2P kernel's ISA) and is never asserted.
//!
//! Error measure: exact equality of the bit patterns.

use mpi::topology::SimpleCommunicator;
use mpi::traits::Equivalence;
use nd_fmm_exec::fmm::{Backend, FmmBuilder, Output, OutputPass, Placement};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::Stored;

/// The degree of the check.
pub const DEGREE: usize = 6;

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

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }
}

/// The C3.2 problem of `tests/accuracy.rs`: its 10⁵ points (seed `0xc32`) and its first
/// charge vector (seed `0xc33`).
#[allow(dead_code, reason = "tests/accuracy.rs draws the problem itself")]
pub fn c32_problem() -> (Vec<[f64; 3]>, Vec<f64>) {
    let n = 100_000;
    let mut rng = SplitMix64(0xc32);
    let points = (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(-1.0, 1.0)))
        .collect();
    let mut rng = SplitMix64(0xc33);
    let charges = (0..n).map(|_| rng.range(-1.0, 1.0)).collect();
    (points, charges)
}

/// The output's values as f64 bit patterns: potentials, then gradients.
fn bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let widen = |v: &T| RealScalar::to_f64(*v).to_bits();
    let mut values: Vec<u64> = output.potential.iter().map(widen).collect();
    if let Some(gradient) = &output.gradient {
        values.extend(gradient.as_flattened().iter().map(widen));
    }
    values
}

/// FNV-1a over `bits`, each value's eight bytes little-endian.
fn fnv1a(bits: &[u64]) -> u64 {
    bits.iter()
        .flat_map(|b| b.to_le_bytes())
        .fold(0xcbf2_9ce4_8422_2325, |h, byte| {
            (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
}

/// The check of the module documentation on `backend` in `T`, at each count of `threads`:
/// panics on a difference; returns one line per build, with the hash.
pub fn check_whole_output<T: Stored + SimdScalar + Equivalence + Default>(
    backend: Backend,
    threads: &[usize],
    (points, charges): (&[[f64; 3]], &[f64]),
    comm: &SimpleCommunicator,
) -> Vec<String> {
    let charges: Vec<T> = charges.iter().map(|&q| T::from_f64(q)).collect();
    let precision = if size_of::<T>() == 4 { "f32" } else { "f64" };
    let mut lines = Vec::new();
    for gradients in [false, true] {
        for &n in threads {
            let builder = FmmBuilder::<T>::new(DEGREE)
                .max_level(4)
                .max_points_per_leaf(1)
                .gradients(gradients)
                .threads(n)
                .backend(backend);
            let mut fmm = builder
                .build(points, points, comm)
                .unwrap_or_else(|error| panic!("{backend}: the FMM does not build: {error}"));
            let pass = fmm.output_pass();
            let output = fmm.evaluate(&charges).expect("the FMM evaluates");
            let want = bits(&output);
            let what = format!(
                "{backend}, {precision}, C3.2 tree, p = {DEGREE}, gradients {gradients}, {n} \
                 thread(s), output pass on the {pass}"
            );
            let reference = fmm.reference_output().expect("the reference output");
            assert!(
                bits(&reference) == want,
                "{what}: differs from the pass before T9"
            );
            let mut also = String::new();
            if pass == Placement::Device {
                let mut host = builder
                    .output_pass(OutputPass::Host)
                    .build(points, points, comm)
                    .expect("the FMM builds");
                let output = host.evaluate(&charges).expect("the FMM evaluates");
                assert!(
                    bits(&output) == want,
                    "{what}: the host pass differs from the device pass"
                );
                also = ", and the host pass".to_owned();
            }
            lines.push(format!(
                "{what}: {} values bit for bit the pass before T9{also}; output hash {:#018x}",
                want.len(),
                fnv1a(&want)
            ));
        }
    }
    lines
}
