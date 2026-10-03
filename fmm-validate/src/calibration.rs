//! The calibration of the degree p against the accuracy of the complete FMM (C3.4): the
//! core of the `calibrate` example.
//!
//! Design §4 starts from the Gumerov–Duraiswami figures, relative L2 errors near 1e-4,
//! 1e-7 and 1e-10 at p = 3, 8 and 18. This module measures the error of the FMM of
//! `nd-fmm-exec` against p, per precision and distribution, and finds the smallest p
//! for each target accuracy.
//!
//! - The problems ([`config`]) are those of [`Config::c33`] for every distribution, the
//!   uniform cube included: N points, sources equal to targets, an adaptive tree with
//!   `max_level` 16 and 64 points per leaf as the refinement target, eight charge
//!   vectors uniform in [−1, 1), 1,000 sampled targets.
//! - [`Reference`] draws the problem and computes the oracles once per distribution:
//!   `direct_sum` in f64 on the charges, and on the charges rounded to f32 for the f32
//!   runs.
//! - [`sweep`] runs [`fmm_accuracy::run`] for every degree of a precision, with the
//!   default strategy (`Auto`: `Dense` for p ≤ 8, `Rotation` above).
//! - [`smallest_p`] and [`worst`] read the calibration off the sweeps; [`floor`] is the
//!   smallest error a sweep reaches, the f32 rounding floor.
//! - [`leaf_study`] runs one degree at several refinement targets on the same problem.
//!
//! The error measure is that of [`fmm_accuracy`]: per charge vector the relative L2 and
//! max errors of φ and ∇φ at the sampled targets, then the root mean square over the
//! vectors. The errors are deterministic for the seed and the same for every number of
//! threads (C3.5), and change with the P2P kernel only in the last bits (C3S.5); the
//! timings change with both.
//!
//! [`fmm_accuracy`]: crate::fmm_accuracy

use std::fmt;
use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;

use crate::fmm_accuracy::{self, Config, Distribution, Execution, Oracle, Problem, Run};

/// The degrees of the f64 sweep: every degree that CONVENTIONS §3.9 tests for M2L.
pub const F64_DEGREES: [usize; 20] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
];

/// The degrees of the f32 sweep, at most 8 (design §4).
pub const F32_DEGREES: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

/// The target accuracies 10⁻ᵏ of the calibration table, as the exponents k.
pub const TARGET_EXPONENTS: [i32; 10] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// The refinement targets of the leaf-size study.
pub const LEAF_SIZES: [usize; 5] = [16, 32, 64, 128, 256];

/// The degree of the leaf-size study.
pub const LEAF_STUDY_DEGREE: usize = 8;

/// The distributions of the leaf-size study: the uniform cube and the Plummer sphere.
pub const LEAF_STUDY_DISTRIBUTIONS: [Distribution; 2] = [Distribution::Cube, Distribution::Plummer];

/// The calibration problem of `distribution` with `n` points: [`Config::c33`] with `n`
/// in place of 10⁵.
pub const fn config(distribution: Distribution, n: usize) -> Config {
    Config {
        n,
        ..Config::c33(distribution)
    }
}

/// The problem of a [`Config`] and its oracles, computed once and shared by every run.
#[derive(Clone, Debug, PartialEq)]
pub struct Reference {
    /// The points, the sampled targets and the f64 charge vectors.
    pub problem: Problem,
    /// The charge vectors rounded to f32.
    pub charges32: Vec<Vec<f32>>,
    /// The oracle of the f64 charges.
    pub oracle64: Oracle,
    /// The oracle of the f32 charges, in f64.
    pub oracle32: Oracle,
    /// The wall time of drawing the problem and of both oracles (one thread).
    pub time: Duration,
}

impl Reference {
    /// Draws the problem of `config` and computes both oracles.
    pub fn new(config: &Config) -> Self {
        let start = Instant::now();
        let problem = Problem::new(config);
        let oracle64 = Oracle::new(&problem, &problem.charges);
        let charges32 = problem.charges_as::<f32>();
        let rounded: Vec<Vec<f64>> = charges32
            .iter()
            .map(|q| q.iter().map(|&v| f64::from(v)).collect())
            .collect();
        let oracle32 = Oracle::new(&problem, &rounded);
        Self {
            problem,
            charges32,
            oracle64,
            oracle32,
            time: start.elapsed(),
        }
    }
}

/// The floating-point type of a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// f32, with the f32 charges and their oracle.
    F32,
    /// f64.
    F64,
}

impl Precision {
    /// "f32" or "f64", as in [`Run::precision`].
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The degrees of the calibration sweep: [`F32_DEGREES`] or [`F64_DEGREES`].
    pub fn degrees(self) -> &'static [usize] {
        match self {
            Self::F32 => &F32_DEGREES,
            Self::F64 => &F64_DEGREES,
        }
    }
}

/// Runs the FMM of `reference` in `precision` at every degree of `degrees`, with
/// `execution`'s threads and P2P kernel, with `config`'s tree.
///
/// # Collective operation
///
/// On `comm`, which must have one rank ([`fmm_accuracy::run`]).
///
/// # Panics
///
/// As [`fmm_accuracy::run`].
pub fn sweep(
    config: &Config,
    reference: &Reference,
    precision: Precision,
    degrees: &[usize],
    execution: Execution,
    comm: &SimpleCommunicator,
) -> Vec<Run> {
    let problem = &reference.problem;
    degrees
        .iter()
        .map(|&p| match precision {
            Precision::F64 => fmm_accuracy::run::<f64>(
                config,
                problem,
                &problem.charges,
                &reference.oracle64,
                (p, execution),
                comm,
            ),
            Precision::F32 => fmm_accuracy::run::<f32>(
                config,
                problem,
                &reference.charges32,
                &reference.oracle32,
                (p, execution),
                comm,
            ),
        })
        .collect()
}

/// Runs the f64 FMM of `reference` at degree `p` once for each refinement target of
/// `sizes`, otherwise with `config`'s tree. The problem, and so the oracle, does not
/// depend on the refinement target.
///
/// # Collective operation
///
/// On `comm`, which must have one rank ([`fmm_accuracy::run`]).
///
/// # Panics
///
/// As [`fmm_accuracy::run`].
pub fn leaf_study(
    config: &Config,
    reference: &Reference,
    p: usize,
    sizes: &[usize],
    execution: Execution,
    comm: &SimpleCommunicator,
) -> Vec<Run> {
    sizes
        .iter()
        .map(|&max_points_per_leaf| {
            let config = Config {
                max_points_per_leaf,
                ..*config
            };
            sweep(&config, reference, Precision::F64, &[p], execution, comm).remove(0)
        })
        .collect()
}

/// Which output a calibration reads: the relative L2 error of φ or of ∇φ.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Measure {
    /// The relative L2 error of φ.
    Potential,
    /// The relative L2 error of ∇φ.
    Gradient,
}

impl Measure {
    /// The relative L2 error of `run` in this measure (root mean square over the
    /// charge vectors).
    pub fn of(self, run: &Run) -> f64 {
        match self {
            Self::Potential => run.potential.l2,
            Self::Gradient => run.gradient.l2,
        }
    }

    /// "φ" or "∇φ".
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Potential => "φ",
            Self::Gradient => "∇φ",
        }
    }
}

/// The smallest degree that reaches a target, or that none of a sweep does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reached {
    /// The smallest degree whose error is below the target.
    At(usize),
    /// No degree up to the given largest one reaches the target.
    Beyond(usize),
}

impl fmt::Display for Reached {
    /// The degree, or "> p_max".
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::At(p) => write!(f, "{p}"),
            Self::Beyond(p) => write!(f, "> {p}"),
        }
    }
}

/// The smallest degree p of `runs` whose error in `measure` is below `target`, or
/// [`Reached::Beyond`] the largest degree of `runs` if there is none.
///
/// The degree is the first one below the target, whether or not every larger degree
/// stays below it; the sweeps print the errors, so a non-monotone tail (near the f32
/// floor) is visible there.
///
/// # Panics
///
/// If `runs` is empty.
pub fn smallest_p(runs: &[Run], measure: Measure, target: f64) -> Reached {
    let largest = runs.iter().map(|r| r.p).max().expect("a non-empty sweep");
    runs.iter()
        .filter(|r| measure.of(r) < target)
        .map(|r| r.p)
        .min()
        .map_or(Reached::Beyond(largest), Reached::At)
}

/// The worst of several calibrations of one target: the largest degree, and
/// [`Reached::Beyond`] if any one does not reach it.
///
/// # Panics
///
/// If `reached` is empty.
pub fn worst(reached: &[Reached]) -> Reached {
    *reached.iter().max().expect("at least one calibration")
}

/// The smallest error in `measure` over `runs` and the degree that reaches it: for f32,
/// the floor set by rounding.
///
/// # Panics
///
/// If `runs` is empty.
pub fn floor(runs: &[Run], measure: Measure) -> (f64, usize) {
    runs.iter()
        .map(|r| (measure.of(r), r.p))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("a non-empty sweep")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_is_the_largest_degree_or_beyond() {
        assert_eq!(worst(&[Reached::At(4), Reached::At(7)]), Reached::At(7));
        assert_eq!(
            worst(&[Reached::At(20), Reached::Beyond(20), Reached::At(3)]),
            Reached::Beyond(20)
        );
        assert_eq!(Reached::At(12).to_string(), "12");
        assert_eq!(Reached::Beyond(8).to_string(), "> 8");
    }

    #[test]
    fn sweeps_cover_the_tested_degrees() {
        assert_eq!(Precision::F64.degrees(), (1..=20).collect::<Vec<_>>());
        assert_eq!(Precision::F32.degrees(), (1..=8).collect::<Vec<_>>());
        assert_eq!(TARGET_EXPONENTS, std::array::from_fn(|i| i as i32 + 3));
        let c = config(Distribution::Plummer, 1000);
        assert_eq!(
            c,
            Config {
                n: 1000,
                ..Config::c33(Distribution::Plummer)
            }
        );
    }
}
