//! The command line ([`parse`], [`HELP`]).
//!
//! Every list option takes one value or a comma-separated list, and the run is the
//! Cartesian product of the lists ([`Options::combinations`]). An option is written
//! `--name value` or `--name=value`, at most once. Counts accept `1000000`, `1_000_000`
//! and `1e6` (any decimal number with an exponent whose value is a whole number).
//! `--quick` changes only the defaults of `--n` and `--repeats`; given options win.

use std::collections::HashSet;
use std::path::PathBuf;

use nd_fmm_exec::fmm::{
    Backend, DEFAULT_MAX_LEVEL, DeviceGather, DeviceGemm, DeviceReduction, KindTiming, MAX_DEGREE,
    OutputPass,
};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_validate::calibration::Precision;
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::scaling::Workload;

use crate::problem::{DEFAULT_VECTORS, ProblemSpec};

/// The default number of points.
pub const DEFAULT_N: usize = 1_000_000;

/// The default expansion degree p.
pub const DEFAULT_DEGREE: usize = 6;

/// The default number of timed evaluations.
pub const DEFAULT_REPEATS: usize = 10;

/// The default number of warm-up evaluations.
pub const DEFAULT_WARMUP: usize = 2;

/// The default number of sampled targets of the accuracy column.
pub const DEFAULT_ACCURACY_TARGETS: usize = 1000;

/// The number of points of `--quick`.
pub const QUICK_N: usize = 10_000;

/// The timed evaluations of `--quick`.
pub const QUICK_REPEATS: usize = 2;

/// The one-line usage, printed with every command-line error.
pub const USAGE: &str = "usage: nd-fmm-bench [--n N,...] [--precision f32|f64,...] \
[--degree p,...] [--backend host|cpu|metal|cuda,...] [--threads n] \
[--strategy auto|dense|classes|rotation|batched] [--repeats r] [--warmup w] \
[--kinds sync|device|off] [--accuracy TARGETS|off] [--no-gradients] [--reuse-output] \
[--output-pass auto|host|device] [--device-gemm auto|library|hand|shared] \
[--gather auto|separate|fused] [--reduction by-chunk|walk] [--scratch MB] [--leaf-size n] \
[--max-level l] [--table-cache DIR] [--tuning-cache DIR] [--output FILE.md] [--quick] [--help]
       nd-fmm-bench --write-problem FILE [--dist cube|sphere|plummer|clusters] [--n N] \
[--vectors m] [--accuracy TARGETS] [--threads n]
       nd-fmm-bench --check-output OUT,... --problem FILE
       nd-fmm-bench --write-output OUT --problem FILE [--degree p] [--strategy S] [--threads n]
       nd-fmm-bench --compare REF,OUT,...";

/// The text of `--help`.
pub const HELP: &str = "\
nd-fmm-bench: the FMM of nd-fmm-exec on N points uniform in the unit cube [0, 1]^3,
sources equal to targets, charges uniform in [-1, 1). Writes a Markdown report (summary
and per-kind tables) and prints the summary on stdout, progress on stderr. Every list
option takes one value or a comma-separated list; the run is their Cartesian product.

  --n N,...                points (default 1000000; accepts 1e6)
  --precision f32|f64,...  value type (default f32)
  --degree p,...           expansion degree p, 0 to 20 (default 6)
  --backend B,...          host, cpu (CubeCL CPU runtime), metal, cuda (default host)
  --threads n              host and host-fallback threads (default: every core)
  --strategy S             auto, dense, classes, rotation or batched (default auto)
  --repeats r              timed evaluations (default 10)
  --warmup w               untimed evaluations first, which compile the kernels (default 2)
  --kinds sync|device|off  per-kind timing mode (default sync)
  --accuracy TARGETS|off   sampled targets of the error columns (default 1000)
  --no-gradients           potentials only (gradients are on by default)
  --reuse-output           evaluate into one reused output (Fmm::evaluate_into) rather
                           than a fresh one per evaluation (Fmm::evaluate, the default)
  --output-pass P          auto, host or device: where the output pass runs (default
                           auto: on the device where it does f64 arithmetic)
  --device-gemm G          auto, library, hand or shared: the GEMM of the device's M2M,
                           L2L and dense M2L (default auto; shared is the shared-memory
                           kernel of Phase 6 T7)
  --gather G               auto, separate or fused: how the device's hand-written GEMM
                           gets its input columns (default auto; Phase 6 T7)
  --reduction R            by-chunk or walk: the device's reduction of M2M and M2L
                           (default by-chunk; walk is Phase 4's, Phase 6 T7)
  --scratch MB             the device's chunk budget in MB (default: the device's, per
                           charge vector)
  --leaf-size n            refinement target, points per leaf (default: the library's, 64)
  --max-level l            deepest leaf level, 0 to 16 (default: the library's, 16)
  --table-cache DIR        load and store the operator tables there (default: none)
  --tuning-cache DIR       tune the device path and keep the choices there (default: none)
  --output FILE.md         the report (default bench-results/<host>-<date>-<time>.md)
  --quick                  N = 10^4 and --repeats 2 unless given: a smoke run
  --help                   this text

A combination the backend refuses (f64 on Metal, a backend not compiled in) becomes a
row \"refused: <error>\" and the run goes on. Run it through tools/bench/run.sh, which picks
the cargo features from --backend and sets every BLAS thread variable to 1.

A problem for another code (Phase 6 T9, tools/fmm3d/README.md), without MPI:

  --write-problem FILE     write the points and charges of the scaling example's
                           workload, its sampled targets and the f64 direct sum there
                           (divided by 4 pi) to FILE, a binary file (module problem)
    --dist D               cube, sphere, plummer or clusters (default cube)
    --n N                  points (default 1000000)
    --vectors m            charge vectors (default 8)
    --accuracy TARGETS     sampled targets (default 1000)
    --threads n            threads of the direct sum (default: every core)
  --check-output OUT,...   the errors of each output file (phi and grad phi at the
                           sampled targets) against the direct sum of
    --problem FILE         the problem file; one line per output, the scaling
                           example's error fields (an output of every point, as
                           --write-output writes, is checked at the sampled targets)
  --write-output OUT       our FMM on the points and charges of --problem FILE, every
                           vector, phi and grad phi at every point to OUT (on one rank;
                           --degree p, default 6, --strategy, default batched,
                           --threads)
  --compare REF,OUT,...    each output of every point against the first, over every
                           point: one line per pair, the error fields as differences";

/// The settings of a run.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The numbers of points.
    pub n: Vec<usize>,
    /// The precisions.
    pub precision: Vec<Precision>,
    /// The expansion degrees p.
    pub degree: Vec<usize>,
    /// The backends.
    pub backend: Vec<Backend>,
    /// The host and host-fallback threads (`FmmBuilder::threads`); `None` for every core
    /// ([`threads`](Self::threads)).
    pub threads: Option<usize>,
    /// The M2L strategy (`FmmBuilder::strategy`).
    pub strategy: M2lStrategy,
    /// The timed evaluations, at least 1.
    pub repeats: usize,
    /// The warm-up evaluations before the timed ones.
    pub warmup: usize,
    /// The per-kind timing mode of the second build; `Off` skips it.
    pub kinds: KindTiming,
    /// The sampled targets of the accuracy columns; `None` for no accuracy.
    pub accuracy: Option<usize>,
    /// Whether the FMM computes gradients.
    pub gradients: bool,
    /// Whether the evaluations go through `Fmm::evaluate_into` with one `Output` reused
    /// (`--reuse-output`, Phase 4S T9) rather than `Fmm::evaluate`, the default.
    pub reuse_output: bool,
    /// Where the output pass runs (`FmmBuilder::output_pass`, Phase 4S T9).
    pub output_pass: OutputPass,
    /// The GEMM of the device translations (`FmmBuilder::device_gemm`, Phase 6 T7).
    pub device_gemm: DeviceGemm,
    /// How the device's hand-written GEMM gets its input columns
    /// (`FmmBuilder::device_gather`, Phase 6 T7).
    pub gather: DeviceGather,
    /// The device's reduction (`FmmBuilder::device_reduction`, Phase 6 T7).
    pub reduction: DeviceReduction,
    /// The device's chunk budget in bytes (`FmmBuilder::device_scratch_budget`); `None`
    /// for the device's.
    pub scratch: Option<u64>,
    /// The refinement target (`FmmBuilder::max_points_per_leaf`); `None` for the
    /// library's default.
    pub leaf_size: Option<usize>,
    /// The deepest leaf level (`FmmBuilder::max_level`); `None` for the library's
    /// default.
    pub max_level: Option<usize>,
    /// The table-cache directory (`FmmBuilder::table_cache`).
    pub table_cache: Option<PathBuf>,
    /// The tuning-cache directory (`FmmBuilder::tuning_cache`).
    pub tuning_cache: Option<PathBuf>,
    /// The report file; `None` for `bench-results/<host>-<date>-<time>.md`.
    pub output: Option<PathBuf>,
    /// Whether `--quick` was given.
    pub quick: bool,
}

impl Default for Options {
    /// The defaults: N = 10⁶, f32, p = 6, the host, every core, `Auto`, 10 repeats after
    /// 2 warm-ups, synchronous kind timings, 1,000 sampled targets, gradients, the
    /// library's tree, no caches.
    fn default() -> Self {
        Self {
            n: vec![DEFAULT_N],
            precision: vec![Precision::F32],
            degree: vec![DEFAULT_DEGREE],
            backend: vec![Backend::Host],
            threads: None,
            strategy: M2lStrategy::Auto,
            repeats: DEFAULT_REPEATS,
            warmup: DEFAULT_WARMUP,
            kinds: KindTiming::Synchronous,
            accuracy: Some(DEFAULT_ACCURACY_TARGETS),
            gradients: true,
            reuse_output: false,
            output_pass: OutputPass::Auto,
            device_gemm: DeviceGemm::Auto,
            gather: DeviceGather::Auto,
            reduction: DeviceReduction::ByChunk,
            scratch: None,
            leaf_size: None,
            max_level: None,
            table_cache: None,
            tuning_cache: None,
            output: None,
            quick: false,
        }
    }
}

/// One point of the Cartesian product of the lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Combination {
    /// The backend.
    pub backend: Backend,
    /// The precision.
    pub precision: Precision,
    /// The number of points.
    pub n: usize,
    /// The expansion degree.
    pub p: usize,
}

impl Options {
    /// The Cartesian product of the lists, in the column order of the report: by
    /// backend, then precision, then N, then p, each in the order given.
    pub fn combinations(&self) -> Vec<Combination> {
        let mut all = Vec::new();
        for &backend in &self.backend {
            for &precision in &self.precision {
                for &n in &self.n {
                    for &p in &self.degree {
                        all.push(Combination {
                            backend,
                            precision,
                            n,
                            p,
                        });
                    }
                }
            }
        }
        all
    }

    /// The threads of the run: `--threads`, else every core
    /// (`std::thread::available_parallelism`).
    pub fn threads(&self) -> usize {
        self.threads.unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
        })
    }
}

/// What the command line asks for.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// A benchmark run.
    Run(Box<Options>),
    /// `--write-problem`: a problem file for another code
    /// ([`problem::write_problem`](crate::problem::write_problem); Phase 6 T9).
    WriteProblem(ProblemSpec),
    /// `--check-output`: the errors of other codes' output files against the direct sum of
    /// a problem file ([`problem::check_output`](crate::problem::check_output)).
    CheckOutput {
        /// The problem file.
        problem: PathBuf,
        /// The output files, in the order given.
        outputs: Vec<PathBuf>,
    },
    /// `--write-output`: our FMM on a problem file, φ and ∇φ at every point
    /// ([`problem::evaluate_problem`](crate::problem::evaluate_problem); Phase 6 T9's
    /// cross-check against another code's output).
    WriteOutput {
        /// The problem file.
        problem: PathBuf,
        /// The full output file.
        output: PathBuf,
        /// The degree p.
        degree: usize,
        /// The M2L strategy (`batched` by default: Phase 6 T2's, bit for bit `dense`).
        strategy: M2lStrategy,
        /// The threads; every core if `None`.
        threads: Option<usize>,
    },
    /// `--compare`: every full output file after the first against the first, over every
    /// point ([`problem::compare_outputs`](crate::problem::compare_outputs)).
    Compare {
        /// The output files, the reference first.
        outputs: Vec<PathBuf>,
    },
    /// `--help`: print [`HELP`].
    Help,
}

/// Parses the arguments after the program name.
///
/// # Errors
///
/// A message naming the offending option for an unknown option, a missing or malformed
/// value, a value out of range, or an option given twice. The binary prints it with
/// [`USAGE`] and exits with status 2.
pub fn parse<S: AsRef<str>>(args: &[S]) -> Result<Command, String> {
    let names = || {
        args.iter().map(|a| {
            a.as_ref()
                .split_once('=')
                .map_or(a.as_ref(), |(name, _)| name)
        })
    };
    if names().any(|name| name == "--help" || name == "-h") {
        return Ok(Command::Help);
    }
    if names().any(|name| PROBLEM_COMMANDS.iter().any(|(command, _)| *command == name)) {
        return parse_problem(args);
    }
    let mut options = Options::default();
    let (mut n, mut repeats) = (None, None);
    let mut seen = HashSet::new();
    let mut rest = args.iter().map(AsRef::as_ref);
    while let Some(arg) = rest.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value)),
            _ => (arg, None),
        };
        if !seen.insert(name.to_owned()) {
            return Err(format!("`{name}` is given twice"));
        }
        match name {
            "--help" | "-h" => return Ok(Command::Help),
            "--quick" | "--no-gradients" | "--reuse-output" => {
                if inline.is_some() {
                    return Err(format!("`{name}` takes no value"));
                }
                match name {
                    "--quick" => options.quick = true,
                    "--no-gradients" => options.gradients = false,
                    _ => options.reuse_output = true,
                }
                continue;
            }
            _ => {}
        }
        let value = match inline.or_else(|| rest.next()) {
            Some(value) => value,
            None if name.starts_with("--") => return Err(format!("`{name}` needs a value")),
            None => return Err(format!("unexpected argument `{name}`")),
        };
        let at = |message: &str| format!("`{name} {value}`: {message}");
        match name {
            "--n" => {
                let list = list(value, count).map_err(|e| at(&e))?;
                if list.contains(&0) {
                    return Err(at("N must be at least 1"));
                }
                n = Some(list);
            }
            "--precision" => options.precision = list(value, precision).map_err(|e| at(&e))?,
            "--degree" => {
                let list = list(value, count).map_err(|e| at(&e))?;
                if list.iter().any(|&p| p > MAX_DEGREE) {
                    return Err(at(&format!("the degree p must be at most {MAX_DEGREE}")));
                }
                options.degree = list;
            }
            "--backend" => {
                options.backend = list(value, |s| s.parse::<Backend>().map_err(|e| e.to_string()))
                    .map_err(|e| at(&e))?;
            }
            "--threads" => options.threads = Some(positive(value).map_err(|e| at(&e))?),
            "--strategy" => options.strategy = strategy(value).map_err(|e| at(&e))?,
            "--repeats" => repeats = Some(positive(value).map_err(|e| at(&e))?),
            "--warmup" => options.warmup = count(value).map_err(|e| at(&e))?,
            "--kinds" => options.kinds = kinds(value).map_err(|e| at(&e))?,
            "--output-pass" => options.output_pass = output_pass(value).map_err(|e| at(&e))?,
            "--device-gemm" => options.device_gemm = device_gemm(value).map_err(|e| at(&e))?,
            "--gather" => options.gather = gather(value).map_err(|e| at(&e))?,
            "--reduction" => options.reduction = reduction(value).map_err(|e| at(&e))?,
            "--scratch" => {
                let mb = positive(value).map_err(|e| at(&e))?;
                options.scratch = Some((mb as u64) << 20);
            }
            "--accuracy" => {
                options.accuracy = match value {
                    "off" => None,
                    v => Some(positive(v).map_err(|e| at(&e))?),
                };
            }
            "--leaf-size" => options.leaf_size = Some(positive(value).map_err(|e| at(&e))?),
            "--max-level" => {
                let level = count(value).map_err(|e| at(&e))?;
                if level > DEFAULT_MAX_LEVEL {
                    return Err(at(&format!("the deepest level is {DEFAULT_MAX_LEVEL}")));
                }
                options.max_level = Some(level);
            }
            "--table-cache" => options.table_cache = Some(path(value).map_err(|e| at(&e))?),
            "--tuning-cache" => options.tuning_cache = Some(path(value).map_err(|e| at(&e))?),
            "--output" => options.output = Some(path(value).map_err(|e| at(&e))?),
            _ if name.starts_with("--") => return Err(format!("unknown option `{name}`")),
            _ => return Err(format!("unexpected argument `{name}`")),
        }
    }
    let quick = options.quick;
    options.n = n.unwrap_or_else(|| vec![if quick { QUICK_N } else { DEFAULT_N }]);
    options.repeats = repeats.unwrap_or(if quick {
        QUICK_REPEATS
    } else {
        DEFAULT_REPEATS
    });
    Ok(Command::Run(Box::new(options)))
}

/// The commands of the problem files (Phase 6 T9) and the options each takes.
const PROBLEM_COMMANDS: [(&str, &[&str]); 4] = [
    (
        "--write-problem",
        &["--dist", "--n", "--vectors", "--accuracy", "--threads"],
    ),
    ("--check-output", &["--problem"]),
    (
        "--write-output",
        &["--problem", "--degree", "--strategy", "--threads"],
    ),
    ("--compare", &[]),
];

/// The options of `--write-problem`, `--check-output`, `--write-output` and `--compare`
/// ([`Command::WriteProblem`], [`Command::CheckOutput`], [`Command::WriteOutput`],
/// [`Command::Compare`]): exactly one of the four, and only its own options.
fn parse_problem<S: AsRef<str>>(args: &[S]) -> Result<Command, String> {
    let mut problem = None;
    let (mut distribution, mut n, mut vectors) = (Distribution::Cube, DEFAULT_N, DEFAULT_VECTORS);
    let (mut sampled, mut threads) = (DEFAULT_ACCURACY_TARGETS, None);
    let (mut degree, mut strategy_choice) = (DEFAULT_DEGREE, M2lStrategy::Batched);
    let (mut command, mut file, mut files) = (None, None, None);
    let mut seen = Vec::new();
    let mut rest = args.iter().map(AsRef::as_ref);
    while let Some(arg) = rest.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) if name.starts_with("--") => (name, Some(value)),
            _ => (arg, None),
        };
        if seen.contains(&name) {
            return Err(format!("`{name}` is given twice"));
        }
        seen.push(name);
        let value = match inline.or_else(|| rest.next()) {
            Some(value) => value,
            None if name.starts_with("--") => return Err(format!("`{name}` needs a value")),
            None => return Err(format!("unexpected argument `{name}`")),
        };
        let at = |message: &str| format!("`{name} {value}`: {message}");
        match name {
            "--write-problem" | "--write-output" | "--check-output" | "--compare" => {
                if let Some(other) = command {
                    return Err(format!("`{other}` and `{name}` exclude each other"));
                }
                command = Some(name);
                if name == "--write-problem" || name == "--write-output" {
                    file = Some(path(value).map_err(|e| at(&e))?);
                } else {
                    files = Some(list(value, path).map_err(|e| at(&e))?);
                }
            }
            "--problem" => problem = Some(path(value).map_err(|e| at(&e))?),
            "--dist" => {
                distribution = Distribution::from_name(value).ok_or_else(|| {
                    at("not a distribution; expected `cube`, `sphere`, `plummer` or `clusters`")
                })?;
            }
            "--n" => n = positive(value).map_err(|e| at(&e))?,
            "--vectors" => vectors = positive(value).map_err(|e| at(&e))?,
            "--accuracy" => sampled = positive(value).map_err(|e| at(&e))?,
            "--threads" => threads = Some(positive(value).map_err(|e| at(&e))?),
            "--degree" => {
                degree = count(value).map_err(|e| at(&e))?;
                if degree > MAX_DEGREE {
                    return Err(at(&format!("the degree p must be at most {MAX_DEGREE}")));
                }
            }
            "--strategy" => strategy_choice = strategy(value).map_err(|e| at(&e))?,
            _ if name.starts_with("--") => {
                return Err(format!(
                    "`{name}` does not apply to `--write-problem`, `--check-output`, \
                     `--write-output` or `--compare`"
                ));
            }
            _ => return Err(format!("unexpected argument `{name}`")),
        }
    }
    let command = command.expect("parse calls this for one of the commands");
    let allowed = PROBLEM_COMMANDS
        .iter()
        .find(|(c, _)| *c == command)
        .map(|(_, options)| *options)
        .unwrap_or_default();
    if let Some(name) = seen.iter().find(|n| **n != command && !allowed.contains(n)) {
        return Err(format!("`{name}` does not apply to `{command}`"));
    }
    let needs_problem = || {
        problem
            .clone()
            .ok_or(format!("`{command}` needs `--problem FILE`"))
    };
    match command {
        "--write-problem" => Ok(Command::WriteProblem(ProblemSpec {
            file: file.expect("given with its value"),
            workload: Workload::new(distribution, n),
            vectors,
            sampled: sampled.min(n),
            threads,
        })),
        "--check-output" => Ok(Command::CheckOutput {
            problem: needs_problem()?,
            outputs: files.expect("given with its value"),
        }),
        "--write-output" => Ok(Command::WriteOutput {
            problem: needs_problem()?,
            output: file.expect("given with its value"),
            degree,
            strategy: strategy_choice,
            threads,
        }),
        _ => {
            let outputs = files.expect("given with its value");
            if outputs.len() < 2 {
                return Err("`--compare` needs two output files or more".into());
            }
            Ok(Command::Compare { outputs })
        }
    }
}

/// A comma-separated list of values, none empty.
fn list<V>(value: &str, item: impl Fn(&str) -> Result<V, String>) -> Result<Vec<V>, String> {
    value
        .split(',')
        .map(|s| {
            if s.is_empty() {
                Err("an empty list entry".to_owned())
            } else {
                item(s)
            }
        })
        .collect()
}

/// A count: digits (with `_` separators allowed), or a decimal number with an exponent
/// whose value is a whole number, such as `1e6` or `2.5e5`.
pub fn count(s: &str) -> Result<usize, String> {
    let bad = || format!("`{s}` is not a count (write 1000000, 1_000_000 or 1e6)");
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit() || b == b'_') {
        return s.replace('_', "").parse().map_err(|_| bad());
    }
    if !s.contains(['e', 'E'])
        || !s
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return Err(bad());
    }
    let value: f64 = s.parse().map_err(|_| bad())?;
    // 2⁵³: every whole number up to it is exact in f64.
    if value.is_finite() && value >= 0.0 && value.fract() == 0.0 && value <= 9_007_199_254_740_992.0
    {
        Ok(value as usize)
    } else {
        Err(bad())
    }
}

/// A count of at least 1.
fn positive(s: &str) -> Result<usize, String> {
    match count(s)? {
        0 => Err("must be at least 1".to_owned()),
        v => Ok(v),
    }
}

/// `f32` or `f64`.
fn precision(s: &str) -> Result<Precision, String> {
    match s {
        "f32" => Ok(Precision::F32),
        "f64" => Ok(Precision::F64),
        _ => Err(format!("`{s}` is not a precision; expected `f32` or `f64`")),
    }
}

/// `auto`, `dense`, `classes`, `rotation` or `batched`.
fn strategy(s: &str) -> Result<M2lStrategy, String> {
    match s {
        "auto" => Ok(M2lStrategy::Auto),
        "dense" => Ok(M2lStrategy::Dense),
        "classes" => Ok(M2lStrategy::Classes),
        "rotation" => Ok(M2lStrategy::Rotation),
        "batched" => Ok(M2lStrategy::Batched),
        _ => Err(format!(
            "`{s}` is not a strategy; expected `auto`, `dense`, `classes`, `rotation` or \
             `batched`"
        )),
    }
}

/// `sync`, `device` or `off`.
fn kinds(s: &str) -> Result<KindTiming, String> {
    match s {
        "sync" => Ok(KindTiming::Synchronous),
        "device" => Ok(KindTiming::Device),
        "off" => Ok(KindTiming::Off),
        _ => Err(format!(
            "`{s}` is not a kind-timing mode; expected `sync`, `device` or `off`"
        )),
    }
}

/// `auto`, `host` or `device`.
fn output_pass(s: &str) -> Result<OutputPass, String> {
    match s {
        "auto" => Ok(OutputPass::Auto),
        "host" => Ok(OutputPass::Host),
        "device" => Ok(OutputPass::Device),
        _ => Err(format!(
            "`{s}` is not an output pass; expected `auto`, `host` or `device`"
        )),
    }
}

/// `auto`, `library`, `hand` or `shared`.
fn device_gemm(s: &str) -> Result<DeviceGemm, String> {
    match s {
        "auto" => Ok(DeviceGemm::Auto),
        "library" => Ok(DeviceGemm::Library),
        "hand" => Ok(DeviceGemm::HandWritten),
        "shared" => Ok(DeviceGemm::Shared),
        _ => Err(format!(
            "`{s}` is not a device GEMM; expected `auto`, `library`, `hand` or `shared`"
        )),
    }
}

/// `auto`, `separate` or `fused`.
fn gather(s: &str) -> Result<DeviceGather, String> {
    match s {
        "auto" => Ok(DeviceGather::Auto),
        "separate" => Ok(DeviceGather::Separate),
        "fused" => Ok(DeviceGather::Fused),
        _ => Err(format!(
            "`{s}` is not a gather; expected `auto`, `separate` or `fused`"
        )),
    }
}

/// `by-chunk` or `walk`.
fn reduction(s: &str) -> Result<DeviceReduction, String> {
    match s {
        "by-chunk" => Ok(DeviceReduction::ByChunk),
        "walk" => Ok(DeviceReduction::Walk),
        _ => Err(format!(
            "`{s}` is not a reduction; expected `by-chunk` or `walk`"
        )),
    }
}

/// The device translations of the report's header: the GEMM, the gather, the reduction
/// and the chunk budget, as the command line names them.
pub fn device_translations_name(options: &Options) -> String {
    let gemm = match options.device_gemm {
        DeviceGemm::Auto => "auto",
        DeviceGemm::Library => "library",
        DeviceGemm::HandWritten => "hand",
        DeviceGemm::Shared => "shared",
    };
    let gather = match options.gather {
        DeviceGather::Auto => "auto",
        DeviceGather::Separate => "separate",
        DeviceGather::Fused => "fused",
    };
    let reduction = match options.reduction {
        DeviceReduction::ByChunk => "by-chunk",
        DeviceReduction::Walk => "walk",
    };
    let scratch = options
        .scratch
        .map_or("the device's".to_owned(), |b| format!("{} MB", b >> 20));
    format!("GEMM `{gemm}`, gather `{gather}`, reduction `{reduction}`, chunk budget {scratch}")
}

/// The name of an [`OutputPass`] on the command line: `auto`, `host` or `device`.
pub fn output_pass_name(pass: OutputPass) -> &'static str {
    match pass {
        OutputPass::Auto => "auto",
        OutputPass::Host => "host",
        OutputPass::Device => "device",
    }
}

/// A non-empty path.
fn path(s: &str) -> Result<PathBuf, String> {
    if s.is_empty() {
        Err("an empty path".to_owned())
    } else {
        Ok(PathBuf::from(s))
    }
}

/// The name of a [`KindTiming`] on the command line: `sync`, `device` or `off`.
pub fn kinds_name(mode: KindTiming) -> &'static str {
    match mode {
        KindTiming::Off => "off",
        KindTiming::Synchronous => "sync",
        KindTiming::Device => "device",
    }
}

/// The name of an [`M2lStrategy`] on the command line: `auto`, `dense`, `classes`,
/// `rotation` or `batched`.
pub fn strategy_name(strategy: M2lStrategy) -> &'static str {
    match strategy {
        M2lStrategy::Auto => "auto",
        M2lStrategy::Dense => "dense",
        M2lStrategy::Classes => "classes",
        M2lStrategy::Rotation => "rotation",
        M2lStrategy::Batched => "batched",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Options {
        match parse(args) {
            Ok(Command::Run(options)) => *options,
            other => panic!("{args:?} gave {other:?}"),
        }
    }

    fn error(args: &[&str]) -> String {
        match parse(args) {
            Err(message) => message,
            other => panic!("{args:?} should be refused, gave {other:?}"),
        }
    }

    #[test]
    fn no_arguments_give_the_defaults() {
        let options = run(&[]);
        assert_eq!(options, Options::default());
        assert_eq!(options.n, [1_000_000]);
        assert_eq!(options.precision, [Precision::F32]);
        assert_eq!(options.degree, [6]);
        assert_eq!(options.backend, [Backend::Host]);
        assert_eq!(options.threads, None);
        assert_eq!(options.strategy, M2lStrategy::Auto);
        assert_eq!((options.repeats, options.warmup), (10, 2));
        assert_eq!(options.kinds, KindTiming::Synchronous);
        assert_eq!(options.accuracy, Some(1000));
        assert!(options.gradients && !options.quick && !options.reuse_output);
        assert_eq!((options.leaf_size, options.max_level), (None, None));
        assert_eq!(
            (options.table_cache, options.tuning_cache, options.output),
            (None, None, None)
        );
    }

    #[test]
    fn every_option_parses() {
        let options = run(&[
            "--n",
            "1000,2e3",
            "--precision",
            "f64,f32",
            "--degree",
            "0,3,20",
            "--backend",
            "host,cpu,metal,cuda",
            "--threads",
            "4",
            "--strategy",
            "rotation",
            "--repeats",
            "3",
            "--warmup",
            "0",
            "--kinds",
            "device",
            "--accuracy",
            "50",
            "--no-gradients",
            "--reuse-output",
            "--output-pass",
            "host",
            "--device-gemm",
            "shared",
            "--gather",
            "fused",
            "--reduction",
            "walk",
            "--scratch",
            "64",
            "--leaf-size",
            "32",
            "--max-level",
            "12",
            "--table-cache",
            "tables",
            "--tuning-cache",
            "tune",
            "--output",
            "out/report.md",
        ]);
        assert_eq!(
            options,
            Options {
                n: vec![1000, 2000],
                precision: vec![Precision::F64, Precision::F32],
                degree: vec![0, 3, 20],
                backend: Backend::ALL.to_vec(),
                threads: Some(4),
                strategy: M2lStrategy::Rotation,
                repeats: 3,
                warmup: 0,
                kinds: KindTiming::Device,
                accuracy: Some(50),
                gradients: false,
                reuse_output: true,
                output_pass: OutputPass::Host,
                device_gemm: DeviceGemm::Shared,
                gather: DeviceGather::Fused,
                reduction: DeviceReduction::Walk,
                scratch: Some(64 << 20),
                leaf_size: Some(32),
                max_level: Some(12),
                table_cache: Some("tables".into()),
                tuning_cache: Some("tune".into()),
                output: Some("out/report.md".into()),
                quick: false,
            }
        );
        // The `=` form, and the other values of the enumerations.
        let options = run(&[
            "--strategy=dense",
            "--kinds=off",
            "--accuracy=off",
            "--backend=metal",
        ]);
        assert_eq!(options.strategy, M2lStrategy::Dense);
        assert_eq!(options.kinds, KindTiming::Off);
        assert_eq!(options.accuracy, None);
        assert_eq!(options.backend, [Backend::Metal]);
        assert_eq!(
            run(&["--strategy", "classes"]).strategy,
            M2lStrategy::Classes
        );
        assert_eq!(run(&["--strategy", "auto"]).strategy, M2lStrategy::Auto);
        assert_eq!(run(&["--kinds", "sync"]).kinds, KindTiming::Synchronous);
        assert_eq!(parse(&["--help"]), Ok(Command::Help));
        assert_eq!(parse(&["--n", "5", "-h"]), Ok(Command::Help));
    }

    #[test]
    fn counts_accept_exponents_and_separators() {
        assert_eq!(count("1000000"), Ok(1_000_000));
        assert_eq!(count("1_000_000"), Ok(1_000_000));
        assert_eq!(count("1e6"), Ok(1_000_000));
        assert_eq!(count("1E6"), Ok(1_000_000));
        assert_eq!(count("2.5e5"), Ok(250_000));
        assert_eq!(count("4e+6"), Ok(4_000_000));
        assert_eq!(count("0"), Ok(0));
        for bad in [
            "", "1.5", "1e-1", "2.5e0", "-1e3", "1e400", "e6", "1e", "x", "1,0", "0x10", "inf",
            "nan", "1e6 ",
        ] {
            assert!(count(bad).is_err(), "{bad:?} is not a count");
        }
        assert_eq!(run(&["--n", "1e6,4e6"]).n, [1_000_000, 4_000_000]);
    }

    #[test]
    fn quick_sets_the_defaults_only() {
        let options = run(&["--quick"]);
        assert!(options.quick);
        assert_eq!((options.n.as_slice(), options.repeats), (&[10_000][..], 2));
        // Given options win, before or after `--quick`.
        let options = run(&["--n", "500", "--quick", "--repeats", "7"]);
        assert_eq!((options.n.as_slice(), options.repeats), (&[500][..], 7));
    }

    #[test]
    fn bad_input_is_refused_with_its_option() {
        assert_eq!(error(&["--bogus", "1"]), "unknown option `--bogus`");
        assert_eq!(error(&["stray"]), "unexpected argument `stray`");
        assert_eq!(error(&["--n"]), "`--n` needs a value");
        assert_eq!(error(&["--n", "5", "--n", "6"]), "`--n` is given twice");
        assert_eq!(error(&["--quick=yes"]), "`--quick` takes no value");
        assert_eq!(
            error(&["--reuse-output=yes"]),
            "`--reuse-output` takes no value"
        );
        assert!(error(&["--n", "abc"]).starts_with("`--n abc`: `abc` is not a count"));
        assert_eq!(
            error(&["--output-pass", "gpu"]),
            "`--output-pass gpu`: `gpu` is not an output pass; expected `auto`, `host` or \
             `device`"
        );
        assert_eq!(error(&["--n", "0"]), "`--n 0`: N must be at least 1");
        assert_eq!(
            error(&["--n", "10,,20"]),
            "`--n 10,,20`: an empty list entry"
        );
        assert_eq!(
            error(&["--degree", "21"]),
            "`--degree 21`: the degree p must be at most 20"
        );
        assert_eq!(
            error(&["--precision", "f16"]),
            "`--precision f16`: `f16` is not a precision; expected `f32` or `f64`"
        );
        assert!(error(&["--backend", "host,opencl"]).contains("`opencl` is not a backend"));
        assert_eq!(
            error(&["--threads", "0"]),
            "`--threads 0`: must be at least 1"
        );
        assert_eq!(
            error(&["--repeats", "0"]),
            "`--repeats 0`: must be at least 1"
        );
        assert!(error(&["--warmup", "-1"]).starts_with("`--warmup -1`:"));
        assert!(error(&["--strategy", "fast"]).contains("`fast` is not a strategy"));
        assert!(error(&["--kinds", "on"]).contains("`on` is not a kind-timing mode"));
        assert_eq!(
            error(&["--accuracy", "0"]),
            "`--accuracy 0`: must be at least 1"
        );
        assert_eq!(
            error(&["--leaf-size", "0"]),
            "`--leaf-size 0`: must be at least 1"
        );
        assert_eq!(
            error(&["--max-level", "17"]),
            "`--max-level 17`: the deepest level is 16"
        );
        assert_eq!(error(&["--output="]), "`--output `: an empty path");
    }

    #[test]
    fn the_product_runs_backend_then_precision_then_n_then_p() {
        let options = run(&[
            "--backend",
            "host,cpu",
            "--precision",
            "f32,f64",
            "--n",
            "100,200",
            "--degree",
            "3,6,8",
        ]);
        let all = options.combinations();
        assert_eq!(all.len(), 2 * 2 * 2 * 3);
        assert_eq!(
            all[0],
            Combination {
                backend: Backend::Host,
                precision: Precision::F32,
                n: 100,
                p: 3
            }
        );
        assert_eq!(all[1], Combination { p: 6, ..all[0] });
        assert_eq!(all[3], Combination { n: 200, ..all[0] });
        assert_eq!(
            all[6],
            Combination {
                precision: Precision::F64,
                ..all[0]
            }
        );
        assert_eq!(
            all[12],
            Combination {
                backend: Backend::Cpu,
                ..all[0]
            }
        );
        // Every combination once.
        for (i, a) in all.iter().enumerate() {
            assert!(!all[i + 1..].contains(a), "{a:?} twice");
        }
    }

    #[test]
    fn the_problem_commands_parse() {
        let spec = |args: &[&str]| match parse(args) {
            Ok(Command::WriteProblem(spec)) => spec,
            other => panic!("{args:?} gave {other:?}"),
        };
        assert_eq!(
            spec(&["--write-problem", "p.bin"]),
            ProblemSpec {
                file: "p.bin".into(),
                workload: Workload::new(Distribution::Cube, DEFAULT_N),
                vectors: 8,
                sampled: 1000,
                threads: None,
            }
        );
        assert_eq!(
            spec(&[
                "--dist=plummer",
                "--n",
                "1e7",
                "--write-problem",
                "p.bin",
                "--vectors",
                "2",
                "--accuracy",
                "500",
                "--threads",
                "40",
            ]),
            ProblemSpec {
                file: "p.bin".into(),
                workload: Workload::new(Distribution::Plummer, 10_000_000),
                vectors: 2,
                sampled: 500,
                threads: Some(40),
            }
        );
        // The sampled targets are at most N.
        assert_eq!(spec(&["--write-problem", "p", "--n", "10"]).sampled, 10);
        assert_eq!(
            parse(&["--check-output", "a.out,b.out", "--problem", "p.bin"]),
            Ok(Command::CheckOutput {
                problem: "p.bin".into(),
                outputs: vec!["a.out".into(), "b.out".into()],
            })
        );
        assert_eq!(
            parse(&["--write-problem", "p", "--help"]),
            Ok(Command::Help)
        );
        assert_eq!(
            parse(&["--write-output", "o.out", "--problem", "p.bin"]),
            Ok(Command::WriteOutput {
                problem: "p.bin".into(),
                output: "o.out".into(),
                degree: DEFAULT_DEGREE,
                strategy: M2lStrategy::Batched,
                threads: None,
            })
        );
        assert_eq!(
            parse(&[
                "--problem=p.bin",
                "--write-output=o.out",
                "--degree",
                "12",
                "--strategy",
                "rotation",
                "--threads",
                "40",
            ]),
            Ok(Command::WriteOutput {
                problem: "p.bin".into(),
                output: "o.out".into(),
                degree: 12,
                strategy: M2lStrategy::Rotation,
                threads: Some(40),
            })
        );
        assert_eq!(
            parse(&["--compare", "a.out,b.out,c.out"]),
            Ok(Command::Compare {
                outputs: vec!["a.out".into(), "b.out".into(), "c.out".into()],
            })
        );
        for (args, message) in [
            (
                &["--write-problem", "p", "--degree", "3"][..],
                "`--degree` does not apply to `--write-problem`",
            ),
            (
                &["--write-problem", "p", "--repeats", "3"][..],
                "`--repeats` does not apply to `--write-problem`, `--check-output`, \
                 `--write-output` or `--compare`",
            ),
            (
                &["--write-problem", "p", "--check-output", "o"][..],
                "`--write-problem` and `--check-output` exclude each other",
            ),
            (
                &["--write-problem", "p", "--problem", "q"][..],
                "`--problem` does not apply to `--write-problem`",
            ),
            (
                &["--check-output", "o", "--problem", "p", "--n", "5"][..],
                "`--n` does not apply to `--check-output`",
            ),
            (
                &["--check-output", "o"][..],
                "`--check-output` needs `--problem FILE`",
            ),
            (
                &["--write-output", "o"][..],
                "`--write-output` needs `--problem FILE`",
            ),
            (
                &["--compare", "a"][..],
                "`--compare` needs two output files or more",
            ),
            (
                &["--compare", "a,b", "--problem", "p"][..],
                "`--problem` does not apply to `--compare`",
            ),
            (
                &["--write-output", "o", "--problem", "p", "--degree", "21"][..],
                "`--degree 21`: the degree p must be at most 20",
            ),
            (
                &["--write-problem", "p", "--dist", "torus"][..],
                "`--dist torus`: not a distribution; expected `cube`, `sphere`, `plummer` or \
                 `clusters`",
            ),
            (
                &["--write-problem", "p", "--n", "0"][..],
                "`--n 0`: must be at least 1",
            ),
            (&["--write-problem"][..], "`--write-problem` needs a value"),
            (
                &["--write-problem", "p", "--write-problem", "q"][..],
                "`--write-problem` is given twice",
            ),
            (
                &["--check-output", "a,,b", "--problem", "p"][..],
                "`--check-output a,,b`: an empty list entry",
            ),
        ] {
            assert_eq!(error(args), message, "{args:?}");
        }
    }

    #[test]
    fn names_round_trip() {
        for mode in [KindTiming::Off, KindTiming::Synchronous, KindTiming::Device] {
            assert_eq!(kinds(kinds_name(mode)), Ok(mode));
        }
        for s in [
            M2lStrategy::Auto,
            M2lStrategy::Dense,
            M2lStrategy::Classes,
            M2lStrategy::Rotation,
            M2lStrategy::Batched,
        ] {
            assert_eq!(strategy(strategy_name(s)), Ok(s));
        }
        for pass in [OutputPass::Auto, OutputPass::Host, OutputPass::Device] {
            assert_eq!(output_pass(output_pass_name(pass)), Ok(pass));
        }
    }
}
