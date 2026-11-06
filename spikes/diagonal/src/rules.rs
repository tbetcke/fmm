//! The degree rules whose P the sweep runs (P = p + 1).
//!
//! - [`formula`]: proposal §3.5 (Gumerov and Duraiswami 2009, eq. 18, δ = 2), S7's
//!   `calibrate::formula_p`;
//! - [`S7Table`]: S7's per-level degree p* (`spikes/truncation/results/summary.md`, signed
//!   off at CH0.7 as S7's output for S6), and off its grid S7's closed form
//!   max(⌈6 log₁₀(1/ε)⌉ − 2, ⌈x + (ln 1/ε)^{2/3} x^{1/3}⌉);
//! - [`calibrated`]: CH3.3's `DegreeRule::CALIBRATED`, the full FMM's rule
//!   (`fmm-exec/src/helmholtz/degree.rs`, copied).
//!
//! Every rule is taken at x = Re(k) a, a the circumradius, as the FMM's rule is.

use nd_fmm_spike_truncation::calibrate::formula_p;

/// The ε of the sweep, 10⁻³ … 10⁻¹⁰.
pub const EPSILONS: [f64; 8] = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10];

/// Proposal §3.5's degree at x = Re(k) a.
pub fn formula(x: f64, eps: f64) -> usize {
    formula_p(x, eps)
}

/// ⌈v − 10⁻⁹⌉, at least 0 (as `degree.rs`'s `rounded_up`).
fn rounded_up(v: f64) -> usize {
    (v - 1e-9).ceil().max(0.0) as usize
}

/// CH3.3's calibrated rule: max(⌈0.30 L² + 0.25 L + 2.60⌉, ⌈x + 0.80 (ln 1/ε)^{2/3} x^{1/3}⌉),
/// L = log₁₀(1/ε).
pub fn calibrated(x: f64, eps: f64) -> usize {
    let l = (1.0 / eps).log10();
    let plateau = rounded_up((0.30 * l + 0.25) * l + 2.60);
    let high = rounded_up(x + 0.80 * (1.0 / eps).ln().powf(2.0 / 3.0) * x.cbrt());
    plateau.max(high)
}

/// S7's closed form.
pub fn s7_closed(x: f64, eps: f64) -> usize {
    let floor = (6.0 * (1.0 / eps).log10()).ceil() - 2.0;
    let high = (x + (1.0 / eps).ln().powf(2.0 / 3.0) * x.cbrt()).ceil();
    floor.max(high) as usize
}

/// S7's p* tables, per kind of k.
#[derive(Clone, Debug, Default)]
pub struct S7Table {
    /// (kind, k a, p* per ε of [`EPSILONS`]).
    rows: Vec<(String, f64, [usize; 8])>,
}

impl S7Table {
    /// The tables of S7's `results/summary.md` (the "Per-level degree p*" table of each
    /// kind's section).
    pub fn parse(text: &str) -> Self {
        let mut rows = Vec::new();
        let mut kind: Option<String> = None;
        let mut in_pstar = false;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("## ") {
                kind = rest.split(':').next().map(|s| s.trim().to_string());
                in_pstar = false;
            } else if line.starts_with("Per-level degree p*") {
                in_pstar = true;
            } else if in_pstar && line.starts_with("| ") && !line.starts_with("| k a") {
                let cells: Vec<&str> = line.split('|').map(str::trim).collect();
                let (Some(k), Ok(ka)) = (kind.clone(), cells[1].parse::<f64>()) else {
                    continue;
                };
                let mut ps = [0usize; 8];
                for (e, p) in ps.iter_mut().enumerate() {
                    let cell = cells[2 + e].trim_end_matches('*');
                    *p = cell.split('/').next().unwrap().trim().parse().unwrap();
                }
                rows.push((k, ka, ps));
            } else if in_pstar && !line.starts_with('|') && !line.is_empty() {
                in_pstar = false;
            }
        }
        S7Table { rows }
    }

    /// S7's results file, read from the spike next to this one.
    pub fn load() -> Self {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../truncation/results/summary.md"
        );
        Self::parse(&std::fs::read_to_string(path).expect("S7's results/summary.md"))
    }

    /// p* for (kind, k a, ε of [`EPSILONS`] index e), or S7's closed form at x = Re(k) a
    /// off the table (then `true`, extrapolated).
    pub fn degree(&self, kind: &str, ka: f64, x: f64, e: usize) -> (usize, bool) {
        self.rows
            .iter()
            .find(|(k, a, _)| k == kind && ((a - ka) / ka).abs() < 2e-3)
            .map_or((s7_closed(x, EPSILONS[e]), true), |(_, _, ps)| {
                (ps[e], false)
            })
    }

    /// Number of rows read.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether no row was read.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
