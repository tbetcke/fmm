//! The calibration of the Helmholtz FMM's per-level degree p_l over (k a_l, ε) in the full
//! FMM (Helmholtz CH3.3; docs/design/helmholtz-fmm-proposal.md §3.5 and §10, CH3.3;
//! docs/helmholtz/tasks/CH3.3.md). Free of MPI and of the FMM: the `helmholtz_calibrate`
//! example measures the error curves with [`helmholtz_accuracy`](crate::helmholtz_accuracy),
//! and this module finds the degrees in them and fits the rule.
//!
//! - **Curves.** A [`Curve`] holds the relative ℓ2 errors [φ, ∇φ] of the FMM at consecutive
//!   degrees p = `first`, `first` + 1, …, with every other setting fixed: one common
//!   degree on every level ≥ 2 (a plateau scan), or one level's degree with the others held
//!   high (a level scan).
//! - **The smallest degree.** [`Curve::smallest_degree`] is the smallest p at which the
//!   larger of the two errors meets the target at p, p + 1 and p + 2 ([`RUN`]), as S7's
//!   search (`spikes/truncation/SPIKE_REPORT.md` §1.4): the errors are not monotone in p to
//!   the last digit, so one lucky degree below the threshold is not taken.
//! - **Points.** A [`Point`] is one measured degree: the level's x = Re(k) a_l (a_l the
//!   circumradius), the tolerance ε and the degree that meets the target (ε/2, the plan's
//!   margin).
//! - **The rule.** [`DegreeRule`] has S7's shape with fitted constants,
//!
//!   ```text
//!   p(x, ε) = max(⌈α log₁₀(1/ε) + β⌉, ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉),
//!   ```
//!
//!   the low-frequency plateau and the high-frequency branch. [`fit_plateau`] fits α and
//!   β to the plateau points, [`fit_high`] fits C to the points the plateau does not cover,
//!   and [`covers`] checks that the rule is at least every point's degree.

use nd_fmm_exec::helmholtz::DegreeRule;

/// The consecutive degrees that must each meet a target (module documentation).
pub const RUN: usize = 3;

/// The errors of the FMM at consecutive degrees (module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct Curve {
    /// The degree of `errors[0]`.
    pub first: usize,
    /// The relative ℓ2 errors [φ, ∇φ] at `first`, `first` + 1, ….
    pub errors: Vec<[f64; 2]>,
}

impl Curve {
    /// An empty curve starting at degree `first`.
    pub fn new(first: usize) -> Self {
        Self {
            first,
            errors: Vec::new(),
        }
    }

    /// The degree the next measurement is for.
    pub fn next_degree(&self) -> usize {
        self.first + self.errors.len()
    }

    /// Appends the errors at [`Curve::next_degree`].
    pub fn push(&mut self, errors: [f64; 2]) {
        self.errors.push(errors);
    }

    /// The smallest degree p of the curve whose larger error meets `target` (≤) at p and at
    /// the next [`RUN`] − 1 degrees, all of them measured; `None` if there is none. A NaN
    /// error never meets a target.
    pub fn smallest_degree(&self, target: f64) -> Option<usize> {
        let meets: Vec<bool> = self
            .errors
            .iter()
            .map(|e| e[0].max(e[1]) <= target)
            .collect();
        meets
            .windows(RUN)
            .position(|w| w.iter().all(|&m| m))
            .map(|i| self.first + i)
    }

    /// Whether the last [`RUN`] degrees all meet `target`: the scan for every target at
    /// least as large is complete.
    pub fn reached(&self, target: f64) -> bool {
        self.errors.len() >= RUN
            && self.errors[self.errors.len() - RUN..]
                .iter()
                .all(|e| e[0].max(e[1]) <= target)
    }
}

/// One measured degree (module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    /// x = Re(k) a_l of the level, a_l its circumradius; for a plateau scan the largest x
    /// of the scanned levels.
    pub x: f64,
    /// The tolerance ε.
    pub epsilon: f64,
    /// The smallest degree that meets the target at ε.
    pub degree: usize,
}

/// The step of the grids of α, β and C: the constants are multiples of it, rounded up.
pub const STEP: f64 = 0.05;

/// Whether `rule` is at least every point's degree.
pub fn covers(rule: &DegreeRule, points: &[Point]) -> bool {
    points
        .iter()
        .all(|p| rule.degree(p.x, p.epsilon) >= p.degree)
}

/// The smallest multiple of [`STEP`] above `value` + 10⁻⁹: a constant at which a
/// [`DegreeRule`] ceiling of a quantity that must exceed p − 1 reaches p.
fn step_above(value: f64) -> f64 {
    (((value + 1e-9) / STEP).floor() + 1.0) * STEP
}

/// α and β of the plateau: over α on the grid of [`STEP`] in [0, 10], β the smallest
/// multiple of [`STEP`] with ⌈α log₁₀(1/ε) + β⌉ ≥ p at every point, the pair with the least
/// total excess Σ (⌈α L + β⌉ − p) over the points, ties to the least Σ (α L + β). The
/// x of the points is not used. `None` without points.
pub fn fit_plateau(points: &[Point]) -> Option<(f64, f64)> {
    if points.is_empty() {
        return None;
    }
    let decades: Vec<f64> = points.iter().map(|p| (1.0 / p.epsilon).log10()).collect();
    let mut best: Option<(usize, f64, f64, f64)> = None;
    for i in 0..=(10.0 / STEP) as usize {
        let alpha = i as f64 * STEP;
        let beta = step_above(
            points
                .iter()
                .zip(&decades)
                .map(|(p, l)| p.degree as f64 - 1.0 - alpha * l)
                .fold(f64::NEG_INFINITY, f64::max),
        );
        let rule = DegreeRule {
            alpha,
            beta,
            c: 0.0,
        };
        let excess = points
            .iter()
            .map(|p| rule.plateau(p.epsilon) - p.degree)
            .sum::<usize>();
        let total = decades.iter().map(|l| alpha * l + beta).sum::<f64>();
        if best.is_none_or(|(e, t, _, _)| excess < e || (excess == e && total < t)) {
            best = Some((excess, total, alpha, beta));
        }
    }
    best.map(|(_, _, alpha, beta)| (alpha, beta))
}

/// C of the high-frequency branch for the plateau α, β: the smallest multiple of [`STEP`]
/// with ⌈x + C (ln 1/ε)^{2/3} x^{1/3}⌉ ≥ p at every point whose degree the plateau does
/// not reach; 0 if the plateau reaches every one. Points with x = 0 that the plateau misses
/// cannot be covered: `None`.
pub fn fit_high(points: &[Point], alpha: f64, beta: f64) -> Option<f64> {
    let plateau = DegreeRule {
        alpha,
        beta,
        c: 0.0,
    };
    let mut c: Option<f64> = None;
    for p in points
        .iter()
        .filter(|p| plateau.plateau(p.epsilon) < p.degree)
    {
        if p.x <= 0.0 {
            return None;
        }
        let needed =
            (p.degree as f64 - 1.0 - p.x) / ((1.0 / p.epsilon).ln().powf(2.0 / 3.0) * p.x.cbrt());
        c = Some(c.map_or(needed, |c: f64| c.max(needed)));
    }
    Some(c.map_or(0.0, |c| step_above(c).max(0.0)))
}

/// The rule fitted to the plateau points (all levels low-frequency) and the other points
/// (the level scans, and plateau scans of trees with a high-frequency level): α and β from
/// the first, C from both. `None` if either fit fails.
pub fn fit(plateau: &[Point], others: &[Point]) -> Option<DegreeRule> {
    let (alpha, beta) = fit_plateau(plateau)?;
    let all: Vec<Point> = plateau.iter().chain(others).copied().collect();
    let c = fit_high(&all, alpha, beta)?;
    let rule = DegreeRule { alpha, beta, c };
    covers(&rule, &all).then_some(rule)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A curve whose larger error is 10^(−p/3), with optional overrides.
    fn curve(first: usize, last: usize, overrides: &[(usize, f64)]) -> Curve {
        let mut c = Curve::new(first);
        for p in first..=last {
            let e = overrides
                .iter()
                .find(|(q, _)| *q == p)
                .map_or(10f64.powf(-(p as f64) / 3.0), |&(_, e)| e);
            c.push([e / 10.0, e]);
        }
        c
    }

    #[test]
    fn smallest_degree_of_a_monotone_curve() {
        let c = curve(1, 30, &[]);
        // 10^(−p/3) ≤ 10⁻³ from p = 9 on.
        assert_eq!(c.smallest_degree(1e-3), Some(9));
        assert_eq!(c.smallest_degree(1.0), Some(1));
        assert_eq!(c.next_degree(), 31);
        // The last three are 28, 29 and 30; 10^(−28/3) ≈ 4.6e-10.
        assert_eq!(c.smallest_degree(5e-10), Some(28));
        assert!(c.reached(5e-10));
        assert!(!c.reached(1e-10));
        assert_eq!(c.smallest_degree(1e-10), None);
    }

    #[test]
    fn a_lucky_degree_below_the_threshold_is_not_taken() {
        // p = 6 dips below 10⁻³ but p = 7 does not.
        let c = curve(1, 20, &[(6, 1e-4), (7, 2e-3)]);
        assert_eq!(c.smallest_degree(1e-3), Some(9));
        // A bump after the threshold moves it past the bump.
        let c = curve(1, 20, &[(10, 5e-3)]);
        assert_eq!(c.smallest_degree(1e-3), Some(11));
    }

    #[test]
    fn nan_never_meets_and_short_curves_have_none() {
        let c = curve(1, 20, &[(10, f64::NAN)]);
        assert_eq!(c.smallest_degree(1e-3), Some(11));
        assert_eq!(curve(5, 6, &[]).smallest_degree(1.0), None);
        assert!(!curve(5, 6, &[]).reached(1.0));
        assert_eq!(Curve::new(3).smallest_degree(1.0), None);
    }

    #[test]
    fn the_larger_error_decides() {
        let mut c = Curve::new(1);
        for _ in 0..3 {
            c.push([1e-9, 2e-3]);
        }
        assert_eq!(c.smallest_degree(1e-3), None);
        assert_eq!(c.smallest_degree(2e-3), Some(1));
    }

    /// Points made by a known rule at the decades 3..=10 and a few x.
    fn points_of(rule: &DegreeRule, xs: &[f64]) -> Vec<Point> {
        (3..=10)
            .flat_map(|d| {
                let epsilon = 10f64.powi(-d);
                xs.iter().map(move |&x| Point {
                    x,
                    epsilon,
                    degree: rule.degree(x, epsilon),
                })
            })
            .collect()
    }

    #[test]
    fn the_plateau_fit_recovers_a_rule_and_covers_its_points() {
        let truth = DegreeRule {
            alpha: 3.3,
            beta: -4.0,
            c: 0.0,
        };
        let points = points_of(&truth, &[0.5]);
        let (alpha, beta) = fit_plateau(&points).unwrap();
        let fitted = DegreeRule {
            alpha,
            beta,
            c: 0.0,
        };
        assert!(covers(&fitted, &points));
        // The ceilings of the data are matched exactly: no excess.
        for p in &points {
            assert_eq!(fitted.plateau(p.epsilon), p.degree, "{p:?} {fitted:?}");
        }
        assert_eq!(fit_plateau(&[]), None);
    }

    #[test]
    fn the_plateau_fit_is_never_below_a_point() {
        let points =
            [(1e-3, 6), (1e-4, 9), (1e-5, 13), (1e-6, 15), (1e-8, 23)].map(|(epsilon, degree)| {
                Point {
                    x: 1.0,
                    epsilon,
                    degree,
                }
            });
        let (alpha, beta) = fit_plateau(&points).unwrap();
        let rule = DegreeRule {
            alpha,
            beta,
            c: 0.0,
        };
        assert!(covers(&rule, &points), "{rule:?}");
    }

    #[test]
    fn the_full_fit_covers_plateau_and_level_points() {
        let truth = DegreeRule {
            alpha: 3.0,
            beta: -2.0,
            c: 0.8,
        };
        let plateau = points_of(&truth, &[0.1, 1.0]);
        let levels = points_of(&truth, &[8.0, 21.7, 43.3, 64.9]);
        let rule = fit(&plateau, &levels).unwrap();
        assert!(
            covers(&rule, &plateau) && covers(&rule, &levels),
            "{rule:?}"
        );
        assert!(rule.c <= truth.c + 1e-12, "{rule:?}");
        // The plateau's constants come from the plateau alone.
        assert_eq!(fit_plateau(&plateau).unwrap(), (rule.alpha, rule.beta));
    }

    #[test]
    fn a_plateau_point_at_x_zero_above_the_plateau_cannot_be_fitted() {
        let points = [Point {
            x: 0.0,
            epsilon: 1e-3,
            degree: 10,
        }];
        assert_eq!(fit_high(&points, 0.0, 0.0), None);
        assert_eq!(fit_high(&points, 0.0, 10.0), Some(0.0));
    }

    #[test]
    fn steps_lie_above() {
        assert!((step_above(0.0) - 0.05).abs() < 1e-12);
        assert!((step_above(0.051) - 0.10).abs() < 1e-12);
        assert!((step_above(0.05) - 0.10).abs() < 1e-12);
        assert!((step_above(-0.07) - -0.05).abs() < 1e-12);
    }
}
