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

/// The scalar kernel alone, after printing it under the test's name.
pub fn scalar_kernel<T: SimdScalar>(test: &str) -> P2pKernel<T> {
    println!("{test} ({}): ISAs run: scalar", std::any::type_name::<T>());
    P2pKernel::new(Isa::Scalar).expect("the scalar ISA is always available")
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

/// The largest number of targets per block, K · W, over every ISA and precision with
/// the signed-off K (NEON: 2 × 4 in f32, 4 × 2 in f64; AVX2: 1 × 8 and 1 × 4); the
/// sweeps over n_t run from 0 to 3 · BLOCK + 1, which covers every n_t mod K · W of
/// every ISA three times over.
pub const BLOCK: usize = 8;

/// The precisions, with their unit roundoff and sum tolerance.
pub trait Precision: SimdScalar + std::fmt::Debug {
    /// The unit roundoff u_T: 2⁻²⁴ or 2⁻⁵³.
    const U: f64;
    /// The tolerance of a sum against `direct_sum`, relative to the sum of term
    /// magnitudes, for up to 4,096 sources (design §3, requirement 2), where the
    /// reference itself meets it ([`Oracle::tolerance`]).
    const SUM_TOL: f64;
    /// The smallest r² at which the gradient contract holds: 2⁻⁸⁴ in f32, where the
    /// intermediate (q ρ) ρ² stays normal (CONVENTIONS §3.13, "Range of the terms in
    /// f32"), and the domain's 2⁻¹⁰⁸ in f64.
    const GRADIENT_R2_MIN: f64;
}

impl Precision for f32 {
    const U: f64 = 5.960464477539063e-8;
    const SUM_TOL: f64 = 1e-6;
    const GRADIENT_R2_MIN: f64 = 5.169878828456423e-26;
}

impl Precision for f64 {
    const U: f64 = 1.1102230246251565e-16;
    const SUM_TOL: f64 = 1e-14;
    const GRADIENT_R2_MIN: f64 = DOMAIN_MIN;
}

/// The smallest nonzero r² of the kernel domain, 2⁻¹⁰⁸ (CONVENTIONS §3.13).
pub const DOMAIN_MIN: f64 = 3.0814879110195774e-33;
/// The largest r² of the kernel domain, 2⁷ (CONVENTIONS §3.13).
pub const DOMAIN_MAX: f64 = 128.0;

/// SplitMix64, seeded.
pub struct Rng(pub u64);

impl Rng {
    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 2f64.powi(-53)
    }

    /// Uniform in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// A point uniform in [−1, 1)³, rounded to `T`.
    pub fn point<T: SimdScalar>(&mut self) -> [T; 3] {
        [0; 3].map(|_| T::from_f64(self.uniform(-1.0, 1.0)))
    }
}

/// A problem with `n_sources` sources and `n_targets` targets uniform in [−1, 1)³,
/// charges uniform in [−1, 1), rounded to `T`, and zero initial outputs.
pub fn uniform<T: SimdScalar>(n_sources: usize, n_targets: usize, seed: u64) -> Problem<T> {
    let mut rng = Rng(seed);
    Problem {
        sources: (0..n_sources).map(|_| rng.point()).collect(),
        charges: (0..n_sources)
            .map(|_| T::from_f64(rng.uniform(-1.0, 1.0)))
            .collect(),
        targets: (0..n_targets).map(|_| rng.point()).collect(),
        potential: vec![T::zero(); n_targets],
        gradient: vec![[T::zero(); 3]; n_targets],
    }
}

/// `p` widened to f64, exactly.
pub fn widen<T: SimdScalar>(p: &[[T; 3]]) -> Vec<[f64; 3]> {
    p.iter().map(|x| x.map(|c| c.to_f64())).collect()
}

/// The oracle of a problem: `direct_sum` in f64 on its exactly widened inputs, from its
/// initial outputs, and per target the scale its error is measured against.
pub struct Oracle {
    /// The initial potential plus the compensated sum of the potential terms.
    pub potential: Vec<f64>,
    /// The same for the gradient.
    pub gradient: Vec<[f64; 3]>,
    /// |φ₀| + Σⱼ |qⱼ| / rᵢⱼ over the pairs with xᵢ ≠ yⱼ, per target.
    pub potential_scale: Vec<f64>,
    /// max |(∇φ₀)ₖ| + Σⱼ |qⱼ| / rᵢⱼ² over the same pairs, per target.
    pub gradient_scale: Vec<f64>,
    /// The errors of `nd_fmm_ref::p2p::p2p` on the problem, with gradients, in the
    /// measure of [`Oracle::errors`]: (potential, gradient).
    pub reference: (f64, f64),
}

impl Oracle {
    /// The oracle of `problem`.
    pub fn new<T: SimdScalar>(problem: &Problem<T>) -> Self {
        let sources = widen(&problem.sources);
        let targets = widen(&problem.targets);
        let charges: Vec<f64> = problem.charges.iter().map(|&q| q.to_f64()).collect();
        let mut potential: Vec<f64> = problem.potential.iter().map(|&v| v.to_f64()).collect();
        let mut gradient = widen(&problem.gradient);
        nd_fmm_ref::p2p::direct_sum(
            &sources,
            &charges,
            &targets,
            &mut potential,
            Some(&mut gradient),
        );
        let mut potential_scale = Vec::with_capacity(targets.len());
        let mut gradient_scale = Vec::with_capacity(targets.len());
        for (i, x) in targets.iter().enumerate() {
            let mut sp = problem.potential[i].to_f64().abs();
            let mut sg = problem.gradient[i]
                .iter()
                .map(|&g| g.to_f64().abs())
                .fold(0.0, f64::max);
            for (y, &q) in sources.iter().zip(&charges) {
                if x == y {
                    continue;
                }
                let r2: f64 = (0..3).map(|k| (x[k] - y[k]) * (x[k] - y[k])).sum();
                sp += q.abs() / r2.sqrt();
                sg += q.abs() / r2;
            }
            potential_scale.push(sp);
            gradient_scale.push(sg);
        }
        let mut oracle = Self {
            potential,
            gradient,
            potential_scale,
            gradient_scale,
            reference: (0.0, 0.0),
        };
        oracle.reference = oracle.errors(&run_reference(problem, true));
        oracle
    }

    /// The sum tolerances (potential, gradient) of design §3, requirement 2, as
    /// signed off in Phase 3S T5: [`Precision::SUM_TOL`], or twice the reference's own
    /// error on the same inputs where that is larger. In-order summation, which the
    /// kernel shares with the reference (requirement 4), exceeds 1e-6 in f32 with
    /// gradients on some FMM-shaped sets of a thousand or more sources.
    pub fn tolerance<T: Precision>(&self) -> (f64, f64) {
        (
            T::SUM_TOL.max(2.0 * self.reference.0),
            T::SUM_TOL.max(2.0 * self.reference.1),
        )
    }

    /// The largest error of `outputs` relative to the scales, for the potential and,
    /// if present, the gradient components. A non-finite output is an infinite error.
    pub fn errors<T: SimdScalar>(&self, outputs: &Outputs<T>) -> (f64, f64) {
        let relative = |got: T, want: f64, scale: f64| {
            let got = got.to_f64();
            if !got.is_finite() {
                return f64::INFINITY;
            }
            let e = (got - want).abs();
            if e == 0.0 { 0.0 } else { e / scale }
        };
        let (potential, gradient) = outputs;
        let mut ep = 0.0f64;
        for (i, &v) in potential.iter().enumerate() {
            ep = ep.max(relative(v, self.potential[i], self.potential_scale[i]));
        }
        let mut eg = 0.0f64;
        for (i, g) in gradient.iter().flatten().enumerate() {
            for (&gk, &want) in g.iter().zip(&self.gradient[i]) {
                eg = eg.max(relative(gk, want, self.gradient_scale[i]));
            }
        }
        (ep, eg)
    }
}

/// The largest errors per ISA of one test, printed by [`Worst::print`].
pub struct Worst {
    /// The test and what is measured.
    what: String,
    /// Per ISA: the largest potential and gradient errors.
    errors: Vec<(Isa, f64, f64)>,
    /// Per ISA: the checks where the reference's own error exceeded the sum
    /// tolerance, and the largest ratio of the kernel's error to the reference's there.
    above: Vec<(Isa, usize, f64)>,
}

impl Worst {
    /// No errors yet.
    pub fn new(what: impl Into<String>) -> Self {
        Self {
            what: what.into(),
            errors: Vec::new(),
            above: Vec::new(),
        }
    }

    /// Records the sum errors of a run on `isa` against `oracle`, also counting the
    /// checks where the reference itself exceeds [`Precision::SUM_TOL`] (the gradient
    /// only if the run had one).
    pub fn record_sum<T: Precision>(
        &mut self,
        isa: Isa,
        errors: (f64, f64),
        oracle: &Oracle,
        with_gradient: bool,
    ) {
        self.record(isa, errors);
        let pairs = [
            (errors.0, oracle.reference.0),
            (errors.1, oracle.reference.1),
        ];
        for (e, r) in pairs.into_iter().take(1 + usize::from(with_gradient)) {
            if r > T::SUM_TOL {
                match self.above.iter_mut().find(|a| a.0 == isa) {
                    Some(a) => {
                        a.1 += 1;
                        a.2 = a.2.max(e / r);
                    }
                    None => self.above.push((isa, 1, e / r)),
                }
            }
        }
    }

    /// Records the errors `(potential, gradient)` of a run on `isa`.
    pub fn record(&mut self, isa: Isa, (ep, eg): (f64, f64)) {
        match self.errors.iter_mut().find(|e| e.0 == isa) {
            Some(e) => {
                e.1 = e.1.max(ep);
                e.2 = e.2.max(eg);
            }
            None => self.errors.push((isa, ep, eg)),
        }
    }

    /// Prints the largest errors per ISA, in units of `unit` (named `unit_name`).
    pub fn print(&self, unit: f64, unit_name: &str) {
        for (isa, ep, eg) in &self.errors {
            println!(
                "{}: {isa} max error: potential {:.3e} {unit_name}, gradient {:.3e} {unit_name}",
                self.what,
                ep / unit,
                eg / unit
            );
        }
        for (isa, n, ratio) in &self.above {
            println!(
                "{}: {isa}: the reference exceeds the sum tolerance in {n} checks; there the \
                 error is at most {ratio:.3} times the reference's",
                self.what
            );
        }
    }
}

/// Asserts that `errors` are within the tolerances `(tp, tg)`, with the context `what`.
pub fn assert_within((ep, eg): (f64, f64), (tp, tg): (f64, f64), what: &dyn std::fmt::Display) {
    assert!(
        ep <= tp && eg <= tg,
        "{what}: errors {ep:e} (potential), {eg:e} (gradient) exceed {tp:e}, {tg:e}"
    );
}

/// Source leaves: the points and the charges of each.
pub type Leaves<T> = Vec<(Vec<[T; 3]>, Vec<T>)>;

/// A target leaf and its 26 neighbours on the same level, as `LaplaceOperator` maps
/// them (CONVENTIONS §3.13): `n` points per leaf with leaf-scaled coordinates u uniform
/// in [−1, 1)³, and the sources of neighbour s at ŷ = fl(ĉ(s|t) + r̂ u_s) with
/// ĉ ∈ {−2, 0, 2}³ and r̂ = 1. If `mixed`, each neighbour is instead on a random
/// level of the three that 2:1 balance allows: coarser (r̂ = 2, ĉₖ ∈ {±1, ±3}), the
/// same, or finer (r̂ = ½, ĉₖ ∈ {±½, ±3/2}), with the sign of ĉₖ the side of the
/// neighbour. The target leaf's own sources are its targets (the stored chunk, passed
/// unchanged), so the self pairs coincide.
///
/// Returns the 27 source leaves (points, charges) in the order of the target's near
/// list, and the targets.
pub fn fmm_shaped<T: SimdScalar>(n: usize, mixed: bool, seed: u64) -> (Leaves<T>, Vec<[T; 3]>) {
    let mut rng = Rng(seed);
    let targets: Vec<[T; 3]> = (0..n).map(|_| rng.point()).collect();
    let mut leaves = Vec::with_capacity(27);
    for offset in 0..27 {
        let side = [offset % 3, offset / 3 % 3, offset / 9].map(|o| o - 1);
        let charges: Vec<T> = (0..n)
            .map(|_| T::from_f64(rng.uniform(-1.0, 1.0)))
            .collect();
        if side == [0, 0, 0] {
            leaves.push((targets.clone(), charges));
            continue;
        }
        let level = if mixed { rng.below(3) } else { 1 };
        let (scale, centre): (f64, [f64; 3]) = match level {
            // Coarser: |ĉₖ| = 3 on the sides where it is a neighbour, 1 elsewhere
            // (the target lies inside its extent there).
            0 => (
                2.0,
                side.map(|s| if s == 0 { 1.0 } else { 3.0 * f64::from(s) }),
            ),
            1 => (1.0, side.map(|s| 2.0 * f64::from(s))),
            // Finer: |ĉₖ| = 3/2 on the neighbour's sides, ½ elsewhere.
            _ => (
                0.5,
                side.map(|s| if s == 0 { 0.5 } else { 1.5 * f64::from(s) }),
            ),
        };
        let (scale, centre) = (T::from_f64(scale), centre.map(T::from_f64));
        let points = (0..n)
            .map(|_| {
                let u: [T; 3] = rng.point();
                [0, 1, 2].map(|k| centre[k] + scale * u[k])
            })
            .collect();
        leaves.push((points, charges));
    }
    (leaves, targets)
}

/// The gathered form of `leaves`: every source leaf's points and charges, in order.
pub fn gathered<T: SimdScalar>(leaves: &[(Vec<[T; 3]>, Vec<T>)]) -> (Vec<[T; 3]>, Vec<T>) {
    (
        leaves.iter().flat_map(|l| l.0.iter().copied()).collect(),
        leaves.iter().flat_map(|l| l.1.iter().copied()).collect(),
    )
}
