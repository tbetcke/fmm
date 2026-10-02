//! Shared helpers: the kernels of every available ISA, grid-valued problems, runs of
//! the kernel and the reference, and bit patterns.

use nd_fmm_simd::{Isa, P2pKernel, SimdScalar};
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

/// Proptest cases per property, kept low for the debug-mode test time.
pub const CASES: u32 = 64;

/// The kernels of every available ISA, after printing them under the test's name.
pub fn kernels<T: SimdScalar>(test: &str) -> Vec<P2pKernel<T>> {
    let kernels: Vec<P2pKernel<T>> = Isa::available()
        .map(|isa| P2pKernel::new(isa).expect("an available ISA"))
        .collect();
    let names: Vec<String> = kernels.iter().map(|k| k.isa().to_string()).collect();
    println!(
        "{test} ({}): ISAs run: {}",
        std::any::type_name::<T>(),
        names.join(", ")
    );
    kernels
}

/// A proptest runner with [`CASES`] cases.
pub fn runner() -> TestRunner {
    TestRunner::new(Config {
        cases: CASES,
        ..Config::default()
    })
}

/// A P2P problem with grid-valued inputs, stored as integers so that one value serves
/// both precisions exactly.
#[derive(Clone, Debug)]
pub struct RawProblem {
    /// Source points, in units of 2⁻¹⁰.
    pub sources: Vec<[i32; 3]>,
    /// Charges, in units of 2⁻¹⁰.
    pub charges: Vec<i32>,
    /// Target points, in units of 2⁻¹⁰.
    pub targets: Vec<[i32; 3]>,
    /// Initial potential, in units of 2⁻⁸.
    pub potential: Vec<i32>,
    /// Initial gradient, in units of 2⁻⁸.
    pub gradient: Vec<[i32; 3]>,
}

/// A problem in precision `T`.
#[derive(Clone, Debug)]
pub struct Problem<T> {
    /// Source points.
    pub sources: Vec<[T; 3]>,
    /// Charges.
    pub charges: Vec<T>,
    /// Target points.
    pub targets: Vec<[T; 3]>,
    /// Initial potential.
    pub potential: Vec<T>,
    /// Initial gradient.
    pub gradient: Vec<[T; 3]>,
}

impl RawProblem {
    /// The problem in precision `T`; every value is exact in f32 and f64.
    pub fn cast<T: SimdScalar>(&self) -> Problem<T> {
        let point = |p: &[i32; 3]| p.map(|c| grid::<T>(c, 10));
        Problem {
            sources: self.sources.iter().map(point).collect(),
            charges: self.charges.iter().map(|&q| grid(q, 10)).collect(),
            targets: self.targets.iter().map(point).collect(),
            potential: self.potential.iter().map(|&v| grid(v, 8)).collect(),
            gradient: self
                .gradient
                .iter()
                .map(|g| g.map(|c| grid(c, 8)))
                .collect(),
        }
    }
}

/// k · 2⁻ˢ in precision `T`.
fn grid<T: SimdScalar>(k: i32, s: i32) -> T {
    T::from_f64(f64::from(k) * 2f64.powi(-s))
}

/// A grid coordinate in [−4, 4].
fn coordinate() -> impl Strategy<Value = i32> {
    -4096..=4096i32
}

/// A grid point in [−4, 4]³.
fn point() -> impl Strategy<Value = [i32; 3]> {
    [coordinate(), coordinate(), coordinate()]
}

/// Up to `max_sources` sources and `max_targets` targets. Some sources copy an
/// earlier source and some targets copy a source (chosen by index), so coincident
/// pairs and duplicated sources occur; the initial outputs are random.
pub fn problem(max_sources: usize, max_targets: usize) -> impl Strategy<Value = RawProblem> {
    let copy_or_point = || {
        prop_oneof![
            3 => point().prop_map(Err),
            1 => any::<usize>().prop_map(Ok),
        ]
    };
    let sources = prop::collection::vec((copy_or_point(), -1024..=1024i32), 0..=max_sources);
    let targets = prop::collection::vec(
        (
            copy_or_point(),
            -256..=256i32,
            [-256..=256i32, -256..=256i32, -256..=256i32],
        ),
        0..=max_targets,
    );
    (sources, targets).prop_map(|(sources, targets)| {
        let mut points: Vec<[i32; 3]> = Vec::with_capacity(sources.len());
        let mut charges = Vec::with_capacity(sources.len());
        for (p, q) in sources {
            let p = match p {
                Ok(i) if !points.is_empty() => points[i % points.len()],
                Ok(_) => [0, 0, 0],
                Err(p) => p,
            };
            points.push(p);
            charges.push(q);
        }
        let mut raw = RawProblem {
            sources: points,
            charges,
            targets: Vec::new(),
            potential: Vec::new(),
            gradient: Vec::new(),
        };
        for (p, v, g) in targets {
            let p = match p {
                Ok(i) if !raw.sources.is_empty() => raw.sources[i % raw.sources.len()],
                Ok(_) => [0, 0, 0],
                Err(p) => p,
            };
            raw.targets.push(p);
            raw.potential.push(v);
            raw.gradient.push(g);
        }
        raw
    })
}

/// The outputs of one evaluation: the potential and, if requested, the gradient.
pub type Outputs<T> = (Vec<T>, Option<Vec<[T; 3]>>);

/// The initial outputs of `problem`, with or without a gradient.
pub fn initial<T: SimdScalar>(problem: &Problem<T>, with_gradient: bool) -> Outputs<T> {
    (
        problem.potential.clone(),
        with_gradient.then(|| problem.gradient.clone()),
    )
}

/// `kernel.evaluate` on `problem`, from its initial outputs.
pub fn run<T: SimdScalar>(
    kernel: P2pKernel<T>,
    problem: &Problem<T>,
    with_gradient: bool,
) -> Outputs<T> {
    let (mut potential, mut gradient) = initial(problem, with_gradient);
    kernel.evaluate(
        &problem.sources,
        &problem.charges,
        &problem.targets,
        &mut potential,
        gradient.as_deref_mut(),
    );
    (potential, gradient)
}

/// `nd_fmm_ref::p2p::p2p` on `problem`, from its initial outputs.
pub fn run_reference<T: SimdScalar>(problem: &Problem<T>, with_gradient: bool) -> Outputs<T> {
    let (mut potential, mut gradient) = initial(problem, with_gradient);
    nd_fmm_ref::p2p::p2p(
        &problem.sources,
        &problem.charges,
        &problem.targets,
        &mut potential,
        gradient.as_deref_mut(),
    );
    (potential, gradient)
}

/// The bits of every output value, widened to f64 (exact and injective for finite
/// values), potential first, then the gradient components target by target.
pub fn bits<T: SimdScalar>(outputs: &Outputs<T>) -> Vec<u64> {
    let (potential, gradient) = outputs;
    let gradient = gradient.iter().flatten().flatten();
    potential
        .iter()
        .chain(gradient)
        .map(|&v| v.to_f64().to_bits())
        .collect()
}

/// A seeded problem with `n_sources` distinct random sources and `n_targets` random
/// targets, without coincident pairs unless the random points collide.
pub fn seeded(n_sources: usize, n_targets: usize, seed: u64) -> RawProblem {
    let mut state = seed;
    let mut next = |lo: i32, hi: i32| -> i32 {
        // SplitMix64.
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let span = u64::try_from(hi - lo + 1).expect("hi >= lo");
        lo + i32::try_from(z % span).expect("within the span")
    };
    let mut point = || [0; 3].map(|_| next(-4096, 4096));
    let sources: Vec<[i32; 3]> = (0..n_sources).map(|_| point()).collect();
    let targets: Vec<[i32; 3]> = (0..n_targets).map(|_| point()).collect();
    RawProblem {
        charges: (0..n_sources).map(|_| next(-1024, 1024)).collect(),
        potential: (0..n_targets).map(|_| next(-256, 256)).collect(),
        gradient: (0..n_targets)
            .map(|_| [0; 3].map(|_| next(-256, 256)))
            .collect(),
        sources,
        targets,
    }
}
