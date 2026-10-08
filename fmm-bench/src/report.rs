//! The measurements of a run ([`Report`]) and their Markdown form ([`markdown`]).
//!
//! Times are stored in seconds and written in milliseconds (evaluations, kinds) or
//! seconds (builds), with three decimals; errors in scientific notation with two. A
//! combination the backend refused, or whose evaluation failed, keeps its row with the
//! reason in the strategy column and "–" elsewhere.

use std::fmt::Write;

use nd_fmm_exec::fmm::{Backend, DEFAULT_MAX_LEVEL, DEFAULT_MAX_POINTS_PER_LEAF, KindTiming};
use nd_fmm_validate::calibration::Precision;

use crate::options::{Combination, Options, kinds_name, output_pass_name, strategy_name};

/// The machine and the software of a run: the report's header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header {
    /// The start of the run, UTC: `2026-10-07 14:03:59 UTC`.
    pub date: String,
    /// The host name.
    pub host: String,
    /// The CPU model.
    pub cpu: String,
    /// The core count: physical and logical, and the performance cores where the cores
    /// differ in kind (macOS).
    pub cores: String,
    /// One line per device backend asked for: its `DeviceInfo`, or why it did not open.
    pub devices: Vec<String>,
    /// With CUDA asked for: the driver and the toolkit, as `nvidia-smi` and `nvcc`
    /// report them (CubeCL reports neither).
    pub cuda: Option<String>,
    /// `rustc -V`.
    pub rustc: String,
    /// The target and profile.
    pub target: String,
    /// The CubeCL version, or why there is none.
    pub cubecl: String,
    /// The source revision: `git describe --always --dirty`, or the `.source-revision`
    /// file of a synced tree.
    pub revision: String,
    /// The full command line.
    pub command: String,
    /// The backends compiled into the binary.
    pub compiled: String,
    /// The BLAS, OpenMP and rayon thread variables.
    pub environment: String,
    /// The threads of the run ([`Options::threads`]).
    pub threads: usize,
}

/// A statistic over repeated wall times, in seconds.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Stats {
    /// The smallest.
    pub min: f64,
    /// The median (the mean of the middle two of an even count).
    pub median: f64,
    /// The mean.
    pub mean: f64,
    /// The largest.
    pub max: f64,
    /// The sample standard deviation (n − 1 in the denominator; 0 for one sample).
    pub std: f64,
}

impl Stats {
    /// The statistics of `samples`.
    ///
    /// # Panics
    ///
    /// If `samples` is empty.
    pub fn of(samples: &[f64]) -> Self {
        assert!(!samples.is_empty(), "at least one sample");
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let mean = sorted.iter().sum::<f64>() / n as f64;
        let median = if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        };
        let std = if n > 1 {
            (sorted.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt()
        } else {
            0.0
        };
        Self {
            min: sorted[0],
            median,
            mean,
            max: sorted[n - 1],
            std,
        }
    }
}

/// The tree of a combination.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tree {
    /// The leaves.
    pub leaves: usize,
    /// The levels.
    pub levels: usize,
    /// Points per leaf, empty leaves included: min, mean, max.
    pub points_per_leaf: (usize, f64, usize),
    /// The interaction lists U, V, W, X.
    pub lists: [usize; 4],
}

/// One operator kind of the kind-timed evaluations.
#[derive(Clone, Debug, PartialEq)]
pub struct KindRow {
    /// `P2M`, `M2M`, …
    pub kind: &'static str,
    /// Where it ran: `host` or `device`.
    pub placement: String,
    /// Level calls per evaluation (with an entry in their view; M2M both passes).
    pub calls: usize,
    /// Its per-evaluation total over the evaluations, seconds.
    pub min: f64,
    /// See [`min`](Self::min).
    pub mean: f64,
    /// See [`min`](Self::min).
    pub max: f64,
}

/// The kind-timed evaluations of a combination (step 4).
#[derive(Clone, Debug, PartialEq)]
pub struct Kinds {
    /// The mode.
    pub mode: KindTiming,
    /// Every kind, in [`KIND_ORDER`] order.
    pub rows: Vec<KindRow>,
    /// Loading the charges (`StageTimings::load`): writing them into leaf order, and on
    /// a device also uploading and scattering them, seconds (Phase 4S T9).
    pub load: Stats,
    /// The output pass (`StageTimings::output`): on a device the download, then the
    /// scaling into the caller's order, seconds (Phase 4S T9).
    pub output: Stats,
    /// The mean of the download's part of [`output`](Self::output)
    /// (`StageTimings::download`), seconds; zero on the host.
    pub download: f64,
    /// The mean of the stages' remainder outside the level calls, less
    /// [`load`](Self::load) and [`output`](Self::output) (at least zero per evaluation),
    /// seconds: the exchanges and, on a device, the enqueueing between calls.
    pub other: f64,
    /// The mean of the sum over the kinds, seconds.
    pub sum: f64,
    /// The mean wall time of these evaluations, seconds.
    pub evaluation: f64,
    /// The evaluations with kind times (with `Device`, one whose window was not timed has
    /// none).
    pub timed: usize,
}

/// The order of the kinds in the report's tables.
pub const KIND_ORDER: [&str; 8] = ["P2M", "M2M", "M2L", "L2L", "P2L", "M2P", "L2P", "P2P"];

/// The outcome of step 4.
#[derive(Clone, Debug, PartialEq)]
pub enum KindOutcome {
    /// `--kinds off`.
    Off,
    /// The second build or its evaluations failed: the error.
    Refused(String),
    /// The kinds.
    Measured(Kinds),
}

/// What a device run ran and moved.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceFacts {
    /// The device (`DeviceInfo`).
    pub device: String,
    /// The kinds on the device.
    pub on_device: String,
    /// The P2P, leaf, GEMM and rotation layouts.
    pub layouts: String,
    /// One line per device level call of M2M, L2L and M2L: level, pairs, GEMM, chunks.
    pub calls: Vec<String>,
    /// The device memory the operator allocates, bytes.
    pub memory: u64,
    /// The transfers, launches and syncs of one evaluation.
    pub evaluation: String,
    /// The tuning, with a tuning cache.
    pub tuning: Option<String>,
}

/// A measured combination.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    /// The M2L strategy as it ran (on a device, as the device runs it).
    pub strategy: String,
    /// The P2P kernel of the host path and the host fallback.
    pub p2p: String,
    /// The threading report (`ThreadingReport`).
    pub threading: String,
    /// The tree.
    pub tree: Tree,
    /// The build: total, tables, device part, seconds.
    pub build: (f64, f64, f64),
    /// Whether every table family came from the table cache.
    pub tables_loaded: bool,
    /// The timed evaluations (step 3).
    pub evaluation: Stats,
    /// Whether every timed evaluation gave the bits of the first.
    pub identical: bool,
    /// The relative L2 errors of φ and of ∇φ (`None` without gradients); `None` with
    /// `--accuracy off`.
    pub errors: Option<(f64, Option<f64>)>,
    /// Step 4.
    pub kinds: KindOutcome,
    /// On a device.
    pub device: Option<DeviceFacts>,
}

/// The outcome of a combination.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The build refused the combination: the error.
    Refused(String),
    /// An evaluation failed: the error.
    Failed(String),
    /// The measurements.
    Measured(Box<Measured>),
}

/// One combination and its outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// The combination.
    pub combination: Combination,
    /// Its outcome.
    pub outcome: Outcome,
}

/// A run: the header, the settings and a row per combination measured so far.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    /// The machine and software.
    pub header: Header,
    /// The settings.
    pub options: Options,
    /// The seed of the points and charges.
    pub seed: u64,
    /// The rows, in the order of [`Options::combinations`].
    pub rows: Vec<Row>,
}

/// Milliseconds with three decimals.
fn ms(seconds: f64) -> String {
    format!("{:.3}", seconds * 1e3)
}

/// Seconds with three decimals.
fn s(seconds: f64) -> String {
    format!("{seconds:.3}")
}

/// An error in scientific notation.
fn e(error: f64) -> String {
    format!("{error:.2e}")
}

fn precision_name(precision: Precision) -> &'static str {
    precision.name()
}

/// "host, f32, N = 2000, p = 3".
pub fn title(c: &Combination) -> String {
    format!(
        "{}, {}, N = {}, p = {}",
        c.backend,
        precision_name(c.precision),
        c.n,
        c.p
    )
}

/// The first four cells of a row: backend, precision, N, p.
fn key(c: &Combination) -> String {
    format!(
        "| {} | {} | {} | {} |",
        c.backend,
        precision_name(c.precision),
        c.n,
        c.p
    )
}

/// The header of the summary table, with its alignment row.
pub const SUMMARY_HEADER: &str = "| backend | precision | N | p | strategy | build (s) | min (ms) \
| median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |";

/// The summary table: one row per combination.
pub fn summary_table(rows: &[Row]) -> String {
    let mut out = String::from(SUMMARY_HEADER);
    out.push('\n');
    for row in rows {
        let key = key(&row.combination);
        let line = match &row.outcome {
            Outcome::Refused(error) => {
                format!("{key} refused: {} |{}", cell(error), " – |".repeat(9))
            }
            Outcome::Failed(error) => {
                format!("{key} failed: {} |{}", cell(error), " – |".repeat(9))
            }
            Outcome::Measured(m) => {
                let t = &m.evaluation;
                let (phi, grad) = match m.errors {
                    None => ("–".to_owned(), "–".to_owned()),
                    Some((phi, grad)) => (e(phi), grad.map_or("–".to_owned(), e)),
                };
                format!(
                    "{key} {} | {} | {} | {} | {} | {} | {} | {phi} | {grad} | {} |",
                    m.strategy,
                    s(m.build.0),
                    ms(t.min),
                    ms(t.median),
                    ms(t.mean),
                    ms(t.max),
                    ms(t.std),
                    if m.identical { "yes" } else { "**no**" }
                )
            }
        };
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// The header of the kind table, with its alignment row.
pub const KIND_HEADER: &str = "| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P \
| L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |";

/// The kind table: one row per combination, mean ms per kind.
pub fn kind_table(rows: &[Row]) -> String {
    let mut out = String::from(KIND_HEADER);
    out.push('\n');
    for row in rows {
        let key = key(&row.combination);
        let rest = match &row.outcome {
            Outcome::Refused(_) | Outcome::Failed(_) => " – |".repeat(14),
            Outcome::Measured(m) => match &m.kinds {
                KindOutcome::Off => format!(" off |{}", " – |".repeat(13)),
                KindOutcome::Refused(error) => {
                    format!(" refused: {} |{}", cell(error), " – |".repeat(13))
                }
                KindOutcome::Measured(k) => {
                    let mut cells: Vec<String> = k.rows.iter().map(|r| ms(r.mean)).collect();
                    cells.push(ms(k.load.mean));
                    cells.push(ms(k.output.mean));
                    cells.push(ms(k.other));
                    cells.push(ms(k.sum));
                    cells.push(ms(k.evaluation));
                    cells.push(format!("{:.3}", k.sum / m.evaluation.mean));
                    cells.iter().map(|c| format!(" {c} |")).collect()
                }
            },
        };
        out.push_str(&key);
        out.push_str(&rest);
        out.push('\n');
    }
    out
}

/// The call the evaluations time: `Fmm::evaluate`, or with `--reuse-output`
/// `Fmm::evaluate_into` with one reused `Output` (Phase 4S T9).
fn evaluation_call(options: &Options) -> &'static str {
    if options.reuse_output {
        "`Fmm::evaluate_into` (one reused `Output`)"
    } else {
        "`Fmm::evaluate`"
    }
}

/// Text for a table cell: no `|` or line breaks.
fn cell(text: &str) -> String {
    text.replace('|', "/").replace('\n', " ")
}

/// The report as Markdown: the header, the summary and kind tables, a section per
/// combination and the footer.
pub fn markdown(report: &Report) -> String {
    let mut out = String::new();
    let h = &report.header;
    let o = &report.options;
    // Writing to a String does not fail.
    let _ = writeln!(out, "# FMM benchmark: {}, {}", h.host, h.date);
    out.push('\n');
    out.push_str(&header_table(report));
    out.push('\n');
    out.push_str("## Summary\n\n");
    let _ = writeln!(
        out,
        "Evaluation: the wall time of {} over {} timed evaluations with kind \
         timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled \
         targets against the f64 direct sum, from one evaluation. Bit-identical: every timed \
         evaluation gave the bits of the first.",
        evaluation_call(o),
        o.repeats
    );
    out.push('\n');
    out.push_str(&summary_table(&report.rows));
    out.push('\n');
    out.push_str("## Per operator kind\n\n");
    let _ = writeln!(
        out,
        "Mean ms per evaluation over a second build's evaluations with kind timings `{}` \
         (min and max in each combination's section). Load: writing the charges into leaf \
         order, on a device with their upload and scatter. Output: the output pass into the \
         caller's order, on a device with the download, the evaluation's one wait for the \
         device. Other: the rest of the stages outside the level calls (the exchanges, on a \
         device the enqueueing). Sum: the kinds together, without load, output and other. \
         Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the \
         summary's mean, the overhead of the mode.",
        kinds_name(o.kinds)
    );
    out.push('\n');
    out.push_str(&kind_table(&report.rows));
    for row in &report.rows {
        out.push('\n');
        out.push_str(&section(row, o));
    }
    out.push('\n');
    out.push_str(&footer(report));
    out
}

/// The header table: everything needed to repeat the run.
fn header_table(report: &Report) -> String {
    let h = &report.header;
    let o = &report.options;
    let mut rows: Vec<(&str, String)> = vec![
        ("date", h.date.clone()),
        ("host", h.host.clone()),
        ("CPU", format!("{}; {}", h.cpu, h.cores)),
        (
            "devices",
            if h.devices.is_empty() {
                "none asked for (host only)".to_owned()
            } else {
                h.devices.join("; ")
            },
        ),
    ];
    if let Some(cuda) = &h.cuda {
        rows.push(("CUDA", cuda.clone()));
    }
    rows.extend([
        (
            "software",
            format!("{}; {}; CubeCL {}", h.rustc, h.target, h.cubecl),
        ),
        ("compiled backends", h.compiled.clone()),
        ("source revision", h.revision.clone()),
        ("command", format!("`{}`", h.command)),
        ("environment", h.environment.clone()),
        (
            "problem",
            format!(
                "N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed {:#x}), \
                 sources equal to targets, charges uniform in [-1, 1) from the same generator; \
                 gradients {}",
                report.seed,
                if o.gradients { "on" } else { "off" }
            ),
        ),
        (
            "tree",
            format!(
                "adaptive; max_level {}; refinement target {} points per leaf",
                o.max_level.map_or_else(
                    || format!("{DEFAULT_MAX_LEVEL} (default)"),
                    |l| l.to_string()
                ),
                o.leaf_size.map_or_else(
                    || format!("{DEFAULT_MAX_POINTS_PER_LEAF} (default)"),
                    |l| l.to_string()
                )
            ),
        ),
        ("M2L strategy", strategy_name(o.strategy).to_owned()),
        (
            "threads",
            format!(
                "{}{}; host and host-fallback threads (on the CPU runtime, its units per cube)",
                h.threads,
                if o.threads.is_none() {
                    " (default: every core)"
                } else {
                    ""
                }
            ),
        ),
        (
            "timing",
            format!(
                "{} warm-up, then {} timed evaluations; kind timings `{}` in a second build",
                o.warmup,
                o.repeats,
                kinds_name(o.kinds)
            ),
        ),
        (
            "output",
            format!(
                "{}; output pass `{}`",
                if o.reuse_output {
                    "one `Output` reused by every evaluation (`Fmm::evaluate_into`, \
                     `--reuse-output`)"
                } else {
                    "a fresh `Output` per evaluation (`Fmm::evaluate`)"
                },
                output_pass_name(o.output_pass)
            ),
        ),
        (
            "accuracy",
            o.accuracy.map_or_else(
                || "off".to_owned(),
                |k| format!("relative L2 at {k} sampled targets against the f64 direct sum"),
            ),
        ),
        (
            "table cache",
            o.table_cache
                .as_ref()
                .map_or("none".to_owned(), |d| format!("`{}`", d.display())),
        ),
        (
            "tuning cache",
            o.tuning_cache
                .as_ref()
                .map_or("none".to_owned(), |d| format!("`{}`", d.display())),
        ),
    ]);
    let mut out = String::from("| item | value |\n| --- | --- |\n");
    for (item, value) in rows {
        let _ = writeln!(out, "| {item} | {} |", cell(&value));
    }
    out
}

/// The section of one combination.
fn section(row: &Row, options: &Options) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "## {}", title(&row.combination));
    out.push('\n');
    let m = match &row.outcome {
        Outcome::Refused(error) => {
            let _ = writeln!(out, "Refused: {error}");
            return out;
        }
        Outcome::Failed(error) => {
            let _ = writeln!(out, "Failed: {error}");
            return out;
        }
        Outcome::Measured(m) => m,
    };
    let t = &m.tree;
    let _ = writeln!(
        out,
        "- Tree: {} leaves on {} levels; points per leaf {} / {:.1} / {} (min / mean / max); \
         lists U {}, V {}, W {}, X {}.",
        t.leaves,
        t.levels,
        t.points_per_leaf.0,
        t.points_per_leaf.1,
        t.points_per_leaf.2,
        t.lists[0],
        t.lists[1],
        t.lists[2],
        t.lists[3]
    );
    let _ = writeln!(
        out,
        "- Run: strategy {}; P2P kernel {}; {}.",
        m.strategy, m.p2p, m.threading
    );
    let _ = writeln!(
        out,
        "- Build: {} s, of it tables {} s ({}) and device {} s.",
        s(m.build.0),
        s(m.build.1),
        if m.tables_loaded {
            "loaded from the cache"
        } else {
            "built"
        },
        s(m.build.2)
    );
    let ev = &m.evaluation;
    let _ = writeln!(
        out,
        "- Evaluation over {} repeats, ms: min {}, median {}, mean {}, max {}, std {}; \
         bit-identical across the repeats: {}.",
        options.repeats,
        ms(ev.min),
        ms(ev.median),
        ms(ev.mean),
        ms(ev.max),
        ms(ev.std),
        if m.identical { "yes" } else { "**no**" }
    );
    if let (Some((phi, grad)), Some(k)) = (m.errors, options.accuracy) {
        let _ = writeln!(
            out,
            "- Accuracy at {} sampled targets: φ {}, ∇φ {}.",
            k.min(row.combination.n),
            e(phi),
            grad.map_or("–".to_owned(), e)
        );
    }
    if let Some(d) = &m.device {
        let _ = writeln!(
            out,
            "- Device: {}; on the device: {}.",
            d.device, d.on_device
        );
        let _ = writeln!(
            out,
            "- Layouts: {}; device memory {:.1} MB.",
            d.layouts,
            d.memory as f64 / 1e6
        );
        let _ = writeln!(out, "- One evaluation: {}.", d.evaluation);
        if let Some(tuning) = &d.tuning {
            let _ = writeln!(out, "- Tuning: {tuning}.");
        }
        if !d.calls.is_empty() {
            let _ = writeln!(out, "- Device level calls of M2M, L2L and M2L:");
            for call in &d.calls {
                let _ = writeln!(out, "  - {call}");
            }
        }
    }
    match &m.kinds {
        KindOutcome::Off => {}
        KindOutcome::Refused(error) => {
            let _ = writeln!(out, "- Kind timings: refused: {error}");
        }
        KindOutcome::Measured(k) => {
            out.push('\n');
            let _ = writeln!(
                out,
                "Kind timings `{}` ({}), over {} evaluations, ms per evaluation:",
                kinds_name(k.mode),
                k.mode,
                k.timed
            );
            out.push('\n');
            out.push_str(
                "| kind | where | calls | min | mean | max | share |\n\
                 | --- | --- | ---: | ---: | ---: | ---: | ---: |\n",
            );
            for r in &k.rows {
                let share = if k.sum > 0.0 {
                    format!("{:.1}%", 100.0 * r.mean / k.sum)
                } else {
                    "–".to_owned()
                };
                let _ = writeln!(
                    out,
                    "| {} | {} | {} | {} | {} | {} | {share} |",
                    r.kind,
                    r.placement,
                    r.calls,
                    ms(r.min),
                    ms(r.mean),
                    ms(r.max)
                );
            }
            for (name, stats) in [("load", &k.load), ("output", &k.output)] {
                let _ = writeln!(
                    out,
                    "| {name} | | | {} | {} | {} | |",
                    ms(stats.min),
                    ms(stats.mean),
                    ms(stats.max)
                );
            }
            let _ = writeln!(out, "| other | | | | {} | | |", ms(k.other));
            let _ = writeln!(out, "| sum | | | | {} | | 100.0% |", ms(k.sum));
            if row.combination.backend.is_device() {
                out.push('\n');
                let _ = writeln!(
                    out,
                    "Output: of it the download (`read_output`, with its sync) {} ms on average.",
                    ms(k.download)
                );
            }
        }
    }
    out
}

/// The timing method, the backends that ran and the measurement line.
fn footer(report: &Report) -> String {
    let o = &report.options;
    let mut ran: Vec<String> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    for backend in Backend::ALL {
        let rows: Vec<&Row> = report
            .rows
            .iter()
            .filter(|r| r.combination.backend == backend)
            .collect();
        let precisions = |measured: bool| -> Vec<&'static str> {
            let mut names: Vec<&'static str> = rows
                .iter()
                .filter(|r| matches!(r.outcome, Outcome::Measured(_)) == measured)
                .map(|r| precision_name(r.combination.precision))
                .collect();
            names.dedup();
            names
        };
        let (yes, no) = (precisions(true), precisions(false));
        if !yes.is_empty() {
            ran.push(format!("{backend} ({})", yes.join(", ")));
        }
        if !no.is_empty() {
            refused.push(format!("{backend} ({})", no.join(", ")));
        }
    }
    let none = || "none".to_owned();
    let mut out = String::from("## Method\n\n");
    let _ = writeln!(
        out,
        "Each combination builds the FMM once and runs {} warm-up evaluations, which compile \
         the device kernels and are not counted, then times {} evaluations of the same charges \
         by the wall clock around {}, with kind timings off. The kind times come \
         from a second build with kind timings `{}`, which runs its own warm-up and {} timed \
         evaluations; in `sync` mode every level call on a device carries a sync, so the kinds \
         may add up to more than the summary's mean. The errors come from one evaluation \
         against the f64 direct sum, as a sanity check, not a gate.",
        o.warmup,
        o.repeats,
        evaluation_call(o),
        kinds_name(o.kinds),
        o.repeats
    );
    out.push('\n');
    let _ = writeln!(
        out,
        "Backends run: {}; refused or failed: {}.",
        if ran.is_empty() {
            none()
        } else {
            ran.join(", ")
        },
        if refused.is_empty() {
            none()
        } else {
            refused.join(", ")
        }
    );
    out.push('\n');
    let _ = writeln!(out, "Measured on {}; never asserted.", report.header.host);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nd_fmm_exec::tables::M2lStrategy;

    #[test]
    fn stats_of_a_few_samples() {
        // Error measure: absolute error, the values are exact in binary or nearly.
        let s = Stats::of(&[3.0, 1.0, 2.0, 6.0]);
        assert_eq!((s.min, s.median, s.mean, s.max), (1.0, 2.5, 3.0, 6.0));
        // Sample variance: (4 + 1 + 0 + 9) / 3.
        assert!((s.std - (14.0_f64 / 3.0).sqrt()).abs() < 1e-15);
        let one = Stats::of(&[0.5]);
        assert_eq!(
            (one.min, one.median, one.max, one.std),
            (0.5, 0.5, 0.5, 0.0)
        );
        assert_eq!(Stats::of(&[1.0, 5.0, 2.0]).median, 2.0);
    }

    /// The fixed report of the golden test: a measured host row with kinds, a refused
    /// row, and a measured device row with kind timings refused.
    fn fixture() -> Report {
        let options = Options {
            n: vec![2000],
            precision: vec![Precision::F32, Precision::F64],
            degree: vec![3],
            backend: vec![Backend::Host, Backend::Metal],
            threads: Some(2),
            strategy: M2lStrategy::Auto,
            repeats: 2,
            warmup: 1,
            kinds: KindTiming::Synchronous,
            accuracy: Some(100),
            gradients: true,
            reuse_output: false,
            output_pass: nd_fmm_exec::fmm::OutputPass::Auto,
            leaf_size: None,
            max_level: Some(8),
            table_cache: None,
            tuning_cache: Some("tune".into()),
            output: None,
            quick: false,
        };
        let header = Header {
            date: "2026-10-07 12:00:00 UTC".into(),
            host: "bench-host".into(),
            cpu: "Test CPU".into(),
            cores: "4 physical, 8 logical".into(),
            devices: vec!["metal (wgpu<msl>), Test GPU, CubeCL 0.11.0-pre.4, f32".into()],
            cuda: None,
            rustc: "rustc 1.99.0".into(),
            target: "aarch64-macos, release build".into(),
            cubecl: "0.11.0-pre.4".into(),
            revision: "abc1234-dirty".into(),
            command: "nd-fmm-bench --n 2000".into(),
            compiled: "host, metal".into(),
            environment: "OPENBLAS_NUM_THREADS=1, RAYON_NUM_THREADS unset".into(),
            threads: 2,
        };
        let combination = |backend, precision| Combination {
            backend,
            precision,
            n: 2000,
            p: 3,
        };
        let tree = Tree {
            leaves: 64,
            levels: 3,
            points_per_leaf: (20, 31.25, 45),
            lists: [1000, 2000, 0, 0],
        };
        let kind = |kind, placement: &str, calls, mean: f64| KindRow {
            kind,
            placement: placement.into(),
            calls,
            min: mean * 0.5,
            mean,
            max: mean * 1.5,
        };
        let host = Measured {
            strategy: "Dense".into(),
            p2p: "Neon".into(),
            threading: "rayon threads 2; MPI Funneled".into(),
            tree: tree.clone(),
            build: (0.0125, 0.004, 0.0),
            tables_loaded: false,
            evaluation: Stats {
                min: 0.001,
                median: 0.0011,
                mean: 0.0012,
                max: 0.0014,
                std: 0.0002,
            },
            identical: true,
            errors: Some((1.234e-3, Some(4.5e-3))),
            kinds: KindOutcome::Measured(Kinds {
                mode: KindTiming::Synchronous,
                rows: vec![
                    kind("P2M", "host", 1, 0.0001),
                    kind("M2M", "host", 2, 0.00005),
                    kind("M2L", "host", 2, 0.0004),
                    kind("L2L", "host", 2, 0.00005),
                    kind("P2L", "host", 0, 0.0),
                    kind("M2P", "host", 0, 0.0),
                    kind("L2P", "host", 1, 0.0001),
                    kind("P2P", "host", 1, 0.0003),
                ],
                load: Stats::of(&[0.00002, 0.00003]),
                output: Stats::of(&[0.00004, 0.00006]),
                download: 0.0,
                other: 0.00002,
                sum: 0.001,
                evaluation: 0.00115,
                timed: 2,
            }),
            device: None,
        };
        let device = Measured {
            strategy: "Dense".into(),
            p2p: "Neon".into(),
            threading: "rayon threads 2; MPI Funneled".into(),
            tree,
            build: (1.5, 0.004, 1.2),
            tables_loaded: true,
            evaluation: Stats {
                min: 0.0005,
                median: 0.0005,
                mean: 0.0005,
                max: 0.0005,
                std: 0.0,
            },
            identical: false,
            errors: Some((1.3e-3, None)),
            kinds: KindOutcome::Refused("kind timings refused | here".into()),
            device: Some(DeviceFacts {
                device: "metal (wgpu<msl>), Test GPU, CubeCL 0.11.0-pre.4, f32".into(),
                on_device: "P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P".into(),
                layouts: "P2P cube 64; leaf cube 64/32; GEMM cube; rotation cube 16".into(),
                calls: vec!["M2L level 2: 1000 pairs, hand-written cube, 1 chunk".into()],
                memory: 12_345_678,
                evaluation: "uploads 1 (8000 B), downloads 1 (32000 B), launches 40, syncs 1"
                    .into(),
                tuning: Some("0.25 s; strategy: Dense (tuned)".into()),
            }),
        };
        Report {
            header,
            options,
            seed: 0xbe9c,
            rows: vec![
                Row {
                    combination: combination(Backend::Host, Precision::F32),
                    outcome: Outcome::Measured(Box::new(host)),
                },
                Row {
                    combination: combination(Backend::Metal, Precision::F64),
                    outcome: Outcome::Refused("f64 is not supported".into()),
                },
                Row {
                    combination: combination(Backend::Metal, Precision::F32),
                    outcome: Outcome::Measured(Box::new(device)),
                },
            ],
        }
    }

    #[test]
    fn the_markdown_of_a_fixed_report() {
        // Error measure: the exact text. Column order, alignment rows, units (s for the
        // build, ms with three decimals for times, two-digit mantissas for errors), the
        // refused and failed forms, and the escaping of `|` inside a cell.
        let got = markdown(&fixture());
        let want = GOLDEN;
        if got != want {
            for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
                assert_eq!(g, w, "line {}", i + 1);
            }
            assert_eq!(got.lines().count(), want.lines().count(), "line count");
            assert_eq!(got, want);
        }
    }

    #[test]
    fn reused_outputs_and_the_download() {
        // Phase 4S T9: `--reuse-output` named in the header, the summary and the method;
        // a device row's kinds with the download's part of the output.
        let mut report = fixture();
        report.options.reuse_output = true;
        let host_kinds = match &report.rows[0].outcome {
            Outcome::Measured(m) => m.kinds.clone(),
            _ => unreachable!("the fixture's host row is measured"),
        };
        if let Outcome::Measured(m) = &mut report.rows[2].outcome {
            m.kinds = host_kinds;
            if let KindOutcome::Measured(k) = &mut m.kinds {
                k.download = 0.00001;
            }
        }
        let text = markdown(&report);
        assert!(text.contains(
            "| output | one `Output` reused by every evaluation (`Fmm::evaluate_into`, \
             `--reuse-output`); output pass `auto` |\n"
        ));
        assert!(text.contains(
            "Evaluation: the wall time of `Fmm::evaluate_into` (one reused `Output`) over 2"
        ));
        assert!(text.contains("by the wall clock around `Fmm::evaluate_into` (one reused"));
        assert_eq!(
            text.matches("Output: of it the download (`read_output`, with its sync) 0.010 ms")
                .count(),
            1,
            "the device row alone"
        );
    }

    #[test]
    fn failed_rows_and_kinds_off() {
        let mut report = fixture();
        report.rows[0].outcome = Outcome::Failed("device lost".into());
        if let Outcome::Measured(m) = &mut report.rows[2].outcome {
            m.kinds = KindOutcome::Off;
            m.errors = None;
        }
        let summary = summary_table(&report.rows);
        assert!(summary.contains(
            "| host | f32 | 2000 | 3 | failed: device lost | – | – | – | – | – | – | – | – | – |\n"
        ));
        assert!(summary.contains("| 0.500 | 0.000 | – | – | **no** |\n"));
        let kinds = kind_table(&report.rows);
        assert!(kinds.contains(
            "| metal | f32 | 2000 | 3 | off | – | – | – | – | – | – | – | – | – | – | – | – | – |\n"
        ));
        let text = markdown(&report);
        assert!(text.contains("## host, f32, N = 2000, p = 3\n\nFailed: device lost\n"));
        assert!(
            text.contains(
                "Backends run: metal (f32); refused or failed: host (f32), metal (f64).\n"
            )
        );
        // Every table row has as many cells as its header.
        for table in [summary, kinds] {
            let mut lines = table.lines();
            let columns = lines.next().unwrap().matches('|').count();
            assert!(lines.all(|l| l.matches('|').count() == columns), "{table}");
        }
    }

    /// The expected Markdown of [`fixture`].
    const GOLDEN: &str = "\
# FMM benchmark: bench-host, 2026-10-07 12:00:00 UTC

| item | value |
| --- | --- |
| date | 2026-10-07 12:00:00 UTC |
| host | bench-host |
| CPU | Test CPU; 4 physical, 8 logical |
| devices | metal (wgpu<msl>), Test GPU, CubeCL 0.11.0-pre.4, f32 |
| software | rustc 1.99.0; aarch64-macos, release build; CubeCL 0.11.0-pre.4 |
| compiled backends | host, metal |
| source revision | abc1234-dirty |
| command | `nd-fmm-bench --n 2000` |
| environment | OPENBLAS_NUM_THREADS=1, RAYON_NUM_THREADS unset |
| problem | N points uniform in the unit cube [0, 1]^3 (`points::cube`, seed 0xbe9c), sources equal to targets, charges uniform in [-1, 1) from the same generator; gradients on |
| tree | adaptive; max_level 8; refinement target 64 (default) points per leaf |
| M2L strategy | auto |
| threads | 2; host and host-fallback threads (on the CPU runtime, its units per cube) |
| timing | 1 warm-up, then 2 timed evaluations; kind timings `sync` in a second build |
| output | a fresh `Output` per evaluation (`Fmm::evaluate`); output pass `auto` |
| accuracy | relative L2 at 100 sampled targets against the f64 direct sum |
| table cache | none |
| tuning cache | `tune` |

## Summary

Evaluation: the wall time of `Fmm::evaluate` over 2 timed evaluations with kind timings off, in ms. Build: the whole build, in s. Errors: relative L2 at the sampled targets against the f64 direct sum, from one evaluation. Bit-identical: every timed evaluation gave the bits of the first.

| backend | precision | N | p | strategy | build (s) | min (ms) | median (ms) | mean (ms) | max (ms) | std (ms) | error φ | error ∇φ | bit-identical |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| host | f32 | 2000 | 3 | Dense | 0.013 | 1.000 | 1.100 | 1.200 | 1.400 | 0.200 | 1.23e-3 | 4.50e-3 | yes |
| metal | f64 | 2000 | 3 | refused: f64 is not supported | – | – | – | – | – | – | – | – | – |
| metal | f32 | 2000 | 3 | Dense | 1.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.000 | 1.30e-3 | – | **no** |

## Per operator kind

Mean ms per evaluation over a second build's evaluations with kind timings `sync` (min and max in each combination's section). Load: writing the charges into leaf order, on a device with their upload and scatter. Output: the output pass into the caller's order, on a device with the download, the evaluation's one wait for the device. Other: the rest of the stages outside the level calls (the exchanges, on a device the enqueueing). Sum: the kinds together, without load, output and other. Evaluation: the mean wall time of those evaluations. Sum / mean: the sum against the summary's mean, the overhead of the mode.

| backend | precision | N | p | P2M | M2M | M2L | L2L | P2L | M2P | L2P | P2P | load | output | other | sum | evaluation | sum / mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| host | f32 | 2000 | 3 | 0.100 | 0.050 | 0.400 | 0.050 | 0.000 | 0.000 | 0.100 | 0.300 | 0.025 | 0.050 | 0.020 | 1.000 | 1.150 | 0.833 |
| metal | f64 | 2000 | 3 | – | – | – | – | – | – | – | – | – | – | – | – | – | – |
| metal | f32 | 2000 | 3 | refused: kind timings refused / here | – | – | – | – | – | – | – | – | – | – | – | – | – |

## host, f32, N = 2000, p = 3

- Tree: 64 leaves on 3 levels; points per leaf 20 / 31.2 / 45 (min / mean / max); lists U 1000, V 2000, W 0, X 0.
- Run: strategy Dense; P2P kernel Neon; rayon threads 2; MPI Funneled.
- Build: 0.013 s, of it tables 0.004 s (built) and device 0.000 s.
- Evaluation over 2 repeats, ms: min 1.000, median 1.100, mean 1.200, max 1.400, std 0.200; bit-identical across the repeats: yes.
- Accuracy at 100 sampled targets: φ 1.23e-3, ∇φ 4.50e-3.

Kind timings `sync` (synchronous, a sync after each call on a device), over 2 evaluations, ms per evaluation:

| kind | where | calls | min | mean | max | share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| P2M | host | 1 | 0.050 | 0.100 | 0.150 | 10.0% |
| M2M | host | 2 | 0.025 | 0.050 | 0.075 | 5.0% |
| M2L | host | 2 | 0.200 | 0.400 | 0.600 | 40.0% |
| L2L | host | 2 | 0.025 | 0.050 | 0.075 | 5.0% |
| P2L | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| M2P | host | 0 | 0.000 | 0.000 | 0.000 | 0.0% |
| L2P | host | 1 | 0.050 | 0.100 | 0.150 | 10.0% |
| P2P | host | 1 | 0.150 | 0.300 | 0.450 | 30.0% |
| load | | | 0.020 | 0.025 | 0.030 | |
| output | | | 0.040 | 0.050 | 0.060 | |
| other | | | | 0.020 | | |
| sum | | | | 1.000 | | 100.0% |

## metal, f64, N = 2000, p = 3

Refused: f64 is not supported

## metal, f32, N = 2000, p = 3

- Tree: 64 leaves on 3 levels; points per leaf 20 / 31.2 / 45 (min / mean / max); lists U 1000, V 2000, W 0, X 0.
- Run: strategy Dense; P2P kernel Neon; rayon threads 2; MPI Funneled.
- Build: 1.500 s, of it tables 0.004 s (loaded from the cache) and device 1.200 s.
- Evaluation over 2 repeats, ms: min 0.500, median 0.500, mean 0.500, max 0.500, std 0.000; bit-identical across the repeats: **no**.
- Accuracy at 100 sampled targets: φ 1.30e-3, ∇φ –.
- Device: metal (wgpu<msl>), Test GPU, CubeCL 0.11.0-pre.4, f32; on the device: P2M, M2M, M2L, P2L, L2L, L2P, M2P, P2P.
- Layouts: P2P cube 64; leaf cube 64/32; GEMM cube; rotation cube 16; device memory 12.3 MB.
- One evaluation: uploads 1 (8000 B), downloads 1 (32000 B), launches 40, syncs 1.
- Tuning: 0.25 s; strategy: Dense (tuned).
- Device level calls of M2M, L2L and M2L:
  - M2L level 2: 1000 pairs, hand-written cube, 1 chunk
- Kind timings: refused: kind timings refused | here

## Method

Each combination builds the FMM once and runs 1 warm-up evaluations, which compile the device kernels and are not counted, then times 2 evaluations of the same charges by the wall clock around `Fmm::evaluate`, with kind timings off. The kind times come from a second build with kind timings `sync`, which runs its own warm-up and 2 timed evaluations; in `sync` mode every level call on a device carries a sync, so the kinds may add up to more than the summary's mean. The errors come from one evaluation against the f64 direct sum, as a sanity check, not a gate.

Backends run: host (f32), metal (f32); refused or failed: metal (f64).

Measured on bench-host; never asserted.
";
}
