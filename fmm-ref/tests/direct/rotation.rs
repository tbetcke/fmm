//! Acceptance tests for the rotation-based translations M2M, L2L and M2L
//! (Phase 1 / T6): agreement with `direct` on random frames, the 8 octant children,
//! the 316 same-level offsets and at high degree; the special shifts (zero, along
//! ±z, near the z axis and along the other axes); the coaxial step alone; f32 against
//! f64; accumulation; argument checks.
//!
//! # Error measures
//!
//! Every test asserts on the measure of the `direct` tests (`common`): coefficients
//! per degree in the orthonormal weighting of CONVENTIONS §3.8 (Nₘ for multipoles,
//! Nₘ/Sₘ for locals), relative to the term magnitudes sⱼ of the §3.11 sum that the
//! translation equals. This is the measure the rotation steps preserve: in this
//! weighting both rotation rules are orthogonal (§3.8), so the rounding errors of the
//! two rotations are bounded by a few units of rounding times the weighted norms of
//! the rotated input and of the coaxial output, which the term magnitudes bound.
//!
//! For transparency each agreement test also reports, without asserting on it, the
//! "strict" error: the same per-degree weighted error, relative to the weighted
//! per-degree norm of the direct result itself. It exceeds the asserted error only
//! where the terms of a degree cancel, which rounding cannot resolve for either
//! method. Both worst errors print with `--nocapture`.

mod agreement;
mod panics;
mod precision;
mod special;

use nd_fmm_ref::{Frame, Workspace};

use crate::common::{Op, Worst, apply, degree_error, degree_norms, len};

/// Tolerance of rotation against direct for p ≤ 20, per degree relative to the term
/// magnitudes (brief T6; the Phase 1 gate).
pub const ROTATION_TOL: f64 = 1e-13;

/// Tolerance of rotation against direct for 20 < p ≤ 30 (M2M and L2L only), per
/// degree relative to the term magnitudes: the accuracy of the rotation blocks at
/// n ≤ 30 (Phase 0 T5).
pub const HIGH_DEGREE_TOL: f64 = 1e-11;

/// Tolerance of the coaxial step alone (shifts on the z axis, where no rotation is
/// involved) against direct, per degree relative to the term magnitudes (brief T6).
pub const COAXIAL_TOL: f64 = 1e-14;

/// Largest degree of the 1e-13 gate (brief T6).
pub const P_GATE: usize = 20;

/// The tolerance of rotation against direct at degree `p`.
pub fn tolerance(p: usize) -> f64 {
    if p <= P_GATE {
        ROTATION_TOL
    } else {
        HIGH_DEGREE_TOL
    }
}

/// Runs `op` by rotation, in f64 from zero output.
pub fn rotated(
    op: Op,
    p: usize,
    from: &Frame<f64>,
    to: &Frame<f64>,
    ws: &mut Workspace<f64>,
    input: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; len(p)];
    op.rotation()(p, from, to, ws, input, &mut out);
    out
}

/// Worst per-degree error of `got` against `want`, relative to the weighted degree
/// norms of `want` itself (the "strict" measure of the module documentation). A
/// degree whose norm is zero counts as 0 if its error is zero too, and as infinite
/// otherwise.
pub fn strict_error(op: Op, p: usize, got: &[f64], want: &[f64]) -> f64 {
    let diff: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    let errors = degree_norms(op.output(), p, &diff);
    let norms = degree_norms(op.output(), p, want);
    errors
        .iter()
        .zip(&norms)
        .map(|(&e, &n)| match (e == 0.0, n > 0.0) {
            (true, _) => 0.0,
            (false, true) => e / n,
            (false, false) => f64::INFINITY,
        })
        .fold(0.0, f64::max)
}

/// The two errors of one comparison of rotation against direct.
pub struct Errors {
    /// Per degree relative to the term magnitudes (asserted).
    pub terms: f64,
    /// Per degree relative to the direct result's own degree norms (reported).
    pub strict: f64,
}

/// Translates `input` by `op` from `from` to `to` with both methods and returns the
/// errors of the rotation result against the direct one.
pub fn compare(
    op: Op,
    p: usize,
    from: &Frame<f64>,
    to: &Frame<f64>,
    ws: &mut Workspace<f64>,
    input: &[f64],
) -> Errors {
    let got = rotated(op, p, from, to, ws, input);
    let want = apply(op, p, from, to, ws, input);
    let scale = op.scale(p, from, to, input);
    Errors {
        terms: degree_error(op.output(), p, &got, &want, &scale),
        strict: strict_error(op, p, &got, &want),
    }
}

/// Tracks the worst asserted and the worst strict error of a test.
pub struct Tracker {
    terms: Worst,
    strict: Worst,
}

impl Tracker {
    /// `name` names the asserted measure and `strict` the reported one.
    pub fn new(terms: &'static str, strict: &'static str) -> Self {
        Self {
            terms: Worst::new(terms),
            strict: Worst::new(strict),
        }
    }

    /// Records `errors` and returns the asserted error.
    pub fn update(&self, errors: &Errors) -> f64 {
        self.strict.update(errors.strict);
        self.terms.update(errors.terms)
    }
}
