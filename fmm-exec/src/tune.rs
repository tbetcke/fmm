//! Autotune of the device path (Phase 4 T12, C4.7; feature `gpu`): which M2L strategy,
//! which GEMM per level and which P2P layout a device [`Fmm`](crate::fmm::Fmm) runs,
//! timed at build on the `Fmm`'s own levels, chosen once and kept in a cache file in a
//! directory the caller passes ([`FmmBuilder::tuning_cache`]). The design is
//! docs/design/device-path.md §10, signed off on 2026-10-03; its sections are cited as §x.
//!
//! # What is chosen ([`Decision`])
//!
//! | Decision | When | Candidates ([`Candidate`]), the static rule's first | Timed on |
//! | --- | --- | --- | --- |
//! | [`Strategy`](Decision::Strategy) | the builder's strategy is `M2lStrategy::Auto` and M2L runs on the device | `Dense` with each M2L GEMM candidate, `Rotation` | the `Fmm`'s largest V level |
//! | [`Gemm`](Decision::Gemm) of M2M, L2L or dense M2L, per pair bucket | `DeviceGemm::Auto` and the kind on the device | the hand-written GEMM in the backend's default layout and two others, the library (f32, p ≥ 8, a GPU), the default chunk budget and 16 MB | the largest level call of the kind in the bucket |
//! | [`P2p`](Decision::P2p), per points-per-leaf bucket | `DeviceP2pLayout::Auto` and P2P on the device | GPU: cube of 64 (the default), 32 and 128 units, plane layouts of 2 and 4 planes; CPU runtime: vectors of the host's width, 64, 128 and 256 bits | the level with the most near-field pairs |
//!
//! A setting the builder names explicitly (a strategy other than `Auto`,
//! `DeviceGemm::Library` or `HandWritten`, a scratch budget, a P2P layout) is never tuned:
//! it is used as given. A bucket is the count rounded up to a power of two ([`bucket`]):
//! the level's V pairs (M2L), octant pairs (M2M, both passes, and L2L), or the mean points
//! per leaf, (N_s + N_t) / (2 leaves) rounded up (P2P). Level calls of fewer than
//! [`MIN_TUNED_PAIRS`] pairs keep the static rule; they take microseconds.
//!
//! **Registration.** A candidate is timed only if it is registered: its layout must pass
//! the device's check, and a library GEMM must multiply in T itself. The library is
//! registered only where its plan's probe at build succeeds and the input-precision guard
//! of `nd_fmm_kernels::translate` passes for every chunk shape (T9); a library strategy
//! whose [`InputPrecision`] is lower than T (TF32, F16 or BF16 for f32 data) is never
//! registered, never timed and never chosen, whatever a test hook offers or claims
//! (docs/phase4/README.md, "M2L strategies").
//!
//! **The coefficient-major layout** (README decision 12, deferred from T9 to T12): the
//! gathered inputs and products laid out with each coefficient's values across the batch
//! contiguous (`nd_fmm_kernels::translate::Orientation::CoefficientMajor`), the GEMM
//! spike's orientation, is **not a candidate** (README decision 13, signed off on
//! 2026-10-04): it was faster than nothing the tuner could choose instead. Measured on the M3 Max, Metal f32:
//! - with `m2l_kernels --orientation` (the C3.2 cube and the Plummer sphere, p = 3–16,
//!   every V level): the hand-written kernel 1.09–1.49× *slower* coefficient-major; the
//!   library 2–14% faster than the box-major library, but 1.1–1.8× slower than the
//!   box-major hand-written kernel on every level;
//! - as a registered candidate of every library decision in the autotune tables (the
//!   cube and the Plummer sphere at p = 8): on M2M and L2L 1.23–1.26× slower than the
//!   box-major library (the choice), on M2L 1.6–1.8× slower than the hand-written kernel.
//!
//! The layout stays an option of `nd_fmm_kernels::translate::PlanSettings` (its
//! hand-written products are those of the box-major layout bit for bit, tested), for a
//! later device to measure; adding it back to the candidates bumps
//! [`CANDIDATE_SET_VERSION`].
//!
//! # The static rule ([`static_strategy`], [`static_gemm`], [`static_p2p`])
//!
//! Where no tuning result exists (no directory, a key the cache lacks and no time left,
//! or a decision the tuner does not take), the static rule of docs/phase4/README.md,
//! "Strategy selection", applies:
//! - f32: `Dense` at every p; M2M and L2L with the library GEMM where CMMA applies (f32,
//!   p ≥ 8, a GPU, the probe and the guard passed) and the hand-written GEMM otherwise;
//!   M2L with the hand-written GEMM (decided after T9, `DeviceGemm::Auto`);
//! - f64: `Dense` for p ≤ [`STATIC_F64_DENSE_MAX_P`] = 11 and `Rotation` from p = 12,
//!   the hand-written GEMM;
//! - the P2P layout of `P2pLayout::default_for`, the default chunk budget, box-major.
//!
//! The f64 boundary is **provisional**: nothing here can time f64 on a GPU, so this rule
//! is what f64 on CUDA gets (device-path.md §10.5). T10 measured the Metal f32 rotation at
//! 0.7–3.0% of f32 peak (1.8–3.0% for p = 4–16), below every break-even efficiency of the
//! spike's f64 model (13.2%, 5.4% and 3.8% of f64 peak at p = 8, 12 and 16 on an A100 or
//! H100, central); if a CUDA f64 rotation reached the same share of its peak, dense would
//! win to p = 16 at least, and the boundary would move up. Keeping the boundary at
//! p = 12, provisional, was signed off with README decision 13; moving it is a separate
//! decision.
//!
//! The rule changes the device path's `M2lStrategy::Auto` only: with M2L on the device,
//! `Auto` resolves by this rule (or the tuned strategy), and the host tables are built for
//! that strategy, so that the host fallback of M2M, L2L and M2L agrees with the device. With
//! M2L on the host fallback, `Auto` resolves by the host rule, so that every kind on the
//! host still gives the host path's bits. The host path's `Auto` (`Dense` to p = 8) does
//! not change.
//!
//! # When tuning runs (§10.4)
//!
//! Only with a device backend and a tuning-cache directory, and only for the decisions the
//! cache lacks. The strategy is decided before the tables are built, since it decides
//! which tables the host and the device build: rotation against dense on the largest V
//! level, with temporary buffers on the device and the tables through `table_cache` (at
//! p ≥ 12 without `table_cache` the dense candidate would take seconds to build, so the
//! strategy keeps the static rule there and is not stored). The other decisions are taken
//! after the uploads, on the operator's own buffers, and the plans of the affected level
//! calls rebuilt. Each candidate is launched once and synced first (compilation
//! excluded), then timed in batches of launches queued without a sync, doubling until a
//! batch takes at least [`MIN_BATCH`] (10 ms), and its time is the median of [`BATCHES`]
//! (5) such batches. The fastest is chosen; the first measured wins a tie.
//!
//! **Budget.** Tuning stops at a deadline, [`DEFAULT_BUDGET`] (10 s) per build or
//! [`FmmBuilder::tuning_budget`], checked before every candidate and between batches: no
//! candidate starts after it, and a candidate running at the deadline stops after its
//! current batch. The static rule's candidate is timed first, so a cut leaves at least
//! it; the cut decision is chosen from what was measured, stored and marked `cut`. A
//! decision that had not started keeps the static rule and is not stored, so a later
//! build tunes it. Tuning time is reported in `BuildTimings::device` and in
//! [`TuningReport::time`], never inside `evaluate`.
//!
//! # The cache (§10.5)
//!
//! [`TuningCache`]: one file per (backend, device, precision, p) in the caller's
//! directory, with the rules of `nd_fmm_tables::TableCache`: written whole to a uniquely
//! named temporary file in the directory, synced, then renamed over the final name, so
//! every rank may read and write it at once and a reader sees one whole file or none; a
//! checksum over the body. The file is text: a magic line, the checksum, then the key and
//! the versions, then one entry per decision with the candidates' times, so that a report
//! can print what was measured. A file whose magic, checksum or syntax is wrong
//! (truncated, corrupted) or whose key or versions differ ([`TuningKey`]: backend, device
//! name, compiler, precision, p; the CubeCL version, the `nd-fmm-kernels` version,
//! `CONVENTION_VERSION`, [`CANDIDATE_SET_VERSION`] and [`FORMAT_VERSION`]) is rejected
//! without a panic, reported ([`CacheState::Rejected`]) and replaced by a new tuning: a
//! stale entry is never trusted. Entries are keyed in the file by the gradients setting and
//! the decision; a build adds the entries it tuned to those of the file and stores the
//! union. Nothing reads an environment variable, and nothing is written without a
//! directory.
//!
//! # Determinism (requirement 6)
//!
//! Every choice is made at build and fixed for the `Fmm`'s lifetime; `evaluate` never
//! tunes. Two `Fmm`s built from the same cache make the same choices and give
//! bit-identical output; without a directory every build takes the static rule. A
//! re-tune (a cache rejected as stale, or a decision new to the file) times the
//! candidates again and **can choose differently**, since timings vary from run to run
//! (about ±25% on a laptop GPU): its output then agrees with the previous choice within
//! the FMM bounds, not bit for bit. [`TuningReport`] lists every decision with where its
//! choice came from ([`Source`]) and the times of its candidates.
//!
//! [`FmmBuilder::tuning_cache`]: crate::fmm::FmmBuilder::tuning_cache
//! [`FmmBuilder::tuning_budget`]: crate::fmm::FmmBuilder::tuning_budget

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use nd_fmm_kernels::p2p::{CPU_VECTOR_BITS, P2pLayout};
use nd_fmm_kernels::translate::{
    CPU_GEMM_BLOCK, DEFAULT_SCRATCH_BYTES, GemmLayout, GemmPolicy, Orientation, PlanSettings,
};
use nd_fmm_kernels::{BackendKind, Device, DeviceInfo, KernelError, Precision};
use nd_fmm_math::CONVENTION_VERSION;

use crate::fmm::{DeviceGemm, OperatorKind};
use crate::tables::M2lStrategy;

/// The version of the tuning-cache file format. Any change to the file bumps it.
pub const FORMAT_VERSION: u32 = 1;

/// The version of the candidate sets of this module. Any change to a candidate set (or
/// to what a candidate means) bumps it, which makes every cached entry stale.
pub const CANDIDATE_SET_VERSION: u32 = 1;

/// The tuning budget of one build by default (device-path.md §10.4).
pub const DEFAULT_BUDGET: Duration = Duration::from_secs(10);

/// The shortest timed batch of launches.
pub const MIN_BATCH: Duration = Duration::from_millis(10);

/// The timed batches of a candidate; its time is their median.
pub const BATCHES: usize = 5;

/// The largest degree at which the static rule takes `Dense` in f64 (device-path.md
/// §10.5, provisional).
pub const STATIC_F64_DENSE_MAX_P: usize = 11;

/// Level calls with fewer pairs keep the static rule.
pub const MIN_TUNED_PAIRS: usize = 512;

/// The smaller chunk budget among the GEMM candidates (the default is
/// `DEFAULT_SCRATCH_BYTES`, 128 MB).
pub const SMALL_SCRATCH_BYTES: u64 = 16 << 20;

/// The magic first line of a tuning-cache file.
const MAGIC: &str = "NDFMMTUN";

/// The bucket of a count: rounded up to a power of two (0 and 1 give 1).
pub fn bucket(count: usize) -> u64 {
    count.max(1).next_power_of_two() as u64
}

/// The precision in which a library GEMM multiplies its inputs: its stage and register
/// types (T9's input-precision guard reads them from the resolved `MatmulElems`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputPrecision {
    /// f64.
    F64,
    /// f32.
    F32,
    /// TF32: f32 with a 10-bit mantissa (CUDA's tensor cores).
    Tf32,
    /// IEEE half precision.
    F16,
    /// bfloat16.
    Bf16,
}

impl InputPrecision {
    /// The input precision of data in `precision`.
    pub fn of(precision: Precision) -> Self {
        match precision {
            Precision::F32 => Self::F32,
            Precision::F64 => Self::F64,
        }
    }

    /// Whether inputs in this precision keep data in `precision`: only `of(precision)`.
    pub fn keeps(self, precision: Precision) -> bool {
        self == Self::of(precision)
    }

    fn name(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::F32 => "f32",
            Self::Tf32 => "tf32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        [Self::F64, Self::F32, Self::Tf32, Self::F16, Self::Bf16]
            .into_iter()
            .find(|p| p.name() == text)
    }
}

/// The GEMM of a [`GemmChoice`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmKind {
    /// The hand-written kernel in this layout.
    HandWritten(GemmLayout),
    /// The library matmul (`cubek-matmul`, `SimpleCyclicCmma` named explicitly), which
    /// multiplies in `inputs`. A level whose shapes it does not take runs the hand-written
    /// kernel in the backend's default layout, decided at build.
    Library {
        /// The precision of its inputs; registered only if it keeps T.
        inputs: InputPrecision,
    },
}

/// The GEMM, layout of the operands and chunk budget of a grouped level call (M2M, L2L,
/// dense M2L).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemmChoice {
    /// The GEMM.
    pub gemm: GemmKind,
    /// The layout of the gathered inputs and products.
    pub orientation: Orientation,
    /// The bytes of a chunk's inputs and products together.
    pub budget: u64,
}

impl GemmChoice {
    /// The hand-written kernel in `layout`, box-major, default budget.
    pub fn hand_written(layout: GemmLayout) -> Self {
        Self {
            gemm: GemmKind::HandWritten(layout),
            orientation: Orientation::BoxMajor,
            budget: DEFAULT_SCRATCH_BYTES,
        }
    }

    /// The library in `precision`, box-major, default budget.
    pub fn library(precision: Precision) -> Self {
        Self {
            gemm: GemmKind::Library {
                inputs: InputPrecision::of(precision),
            },
            orientation: Orientation::BoxMajor,
            budget: DEFAULT_SCRATCH_BYTES,
        }
    }

    /// The plan settings of this choice for tables of order n on `info`: a library
    /// choice takes `GemmPolicy::Auto` with the default layout as its fallback.
    pub fn settings(&self, info: &DeviceInfo, n: usize) -> PlanSettings {
        let (layout, policy) = match self.gemm {
            GemmKind::HandWritten(layout) => (layout, GemmPolicy::HandWritten),
            GemmKind::Library { .. } => (GemmLayout::default_for(info, n), GemmPolicy::Auto),
        };
        PlanSettings {
            n,
            layout,
            policy,
            budget: self.budget,
            orientation: self.orientation,
        }
    }

    /// Whether this is a library choice.
    pub fn is_library(&self) -> bool {
        matches!(self.gemm, GemmKind::Library { .. })
    }
}

impl fmt::Display for GemmChoice {
    /// `hand-written cube(32 x 2 units, 4 columns per unit), box-major, 128 MB`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.gemm {
            GemmKind::HandWritten(layout) => write!(f, "hand-written {layout}")?,
            GemmKind::Library { inputs } => write!(f, "library ({} inputs)", inputs.name())?,
        }
        write!(f, ", {}, {} MB", self.orientation, self.budget >> 20)
    }
}

/// A candidate of a [`Decision`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Candidate {
    /// Strategy `Dense` with this M2L GEMM.
    Dense(GemmChoice),
    /// Strategy `Rotation`, the backend's default rotation layout.
    Rotation,
    /// A GEMM of a grouped level call.
    Gemm(GemmChoice),
    /// A P2P layout.
    P2p(P2pLayout),
}

impl Candidate {
    /// The strategy of a [`Decision::Strategy`] candidate.
    pub fn strategy(&self) -> Option<M2lStrategy> {
        match self {
            Self::Dense(_) => Some(M2lStrategy::Dense),
            Self::Rotation => Some(M2lStrategy::Rotation),
            _ => None,
        }
    }

    /// The GEMM of a `Dense` or `Gemm` candidate.
    pub fn gemm(&self) -> Option<GemmChoice> {
        match *self {
            Self::Dense(choice) | Self::Gemm(choice) => Some(choice),
            _ => None,
        }
    }

    /// The layout of a `P2p` candidate.
    pub fn p2p(&self) -> Option<P2pLayout> {
        match *self {
            Self::P2p(layout) => Some(layout),
            _ => None,
        }
    }
}

impl fmt::Display for Candidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(choice) => write!(f, "Dense, {choice}"),
            Self::Rotation => f.write_str("Rotation"),
            Self::Gemm(choice) => write!(f, "{choice}"),
            Self::P2p(layout) => write!(f, "P2P {layout}"),
        }
    }
}

/// What the tuner decides (module documentation, "What is chosen").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Decision {
    /// The M2L strategy.
    Strategy,
    /// The GEMM of the level calls of `kind` (M2M, L2L or M2L) in `bucket`.
    Gemm {
        /// [`OperatorKind::M2m`], [`OperatorKind::L2l`] or [`OperatorKind::M2l`].
        kind: OperatorKind,
        /// The pairs of the level call, rounded up to a power of two ([`bucket`]).
        bucket: u64,
    },
    /// The P2P layout for a mean of points per leaf in `bucket`.
    P2p {
        /// The mean points per leaf, rounded up to a power of two.
        bucket: u64,
    },
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Strategy => f.write_str("M2L strategy"),
            Self::Gemm { kind, bucket } => write!(f, "{kind} GEMM, ≤ {bucket} pairs"),
            Self::P2p { bucket } => write!(f, "P2P layout, ≤ {bucket} points per leaf"),
        }
    }
}

/// What happened to one candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Timing {
    /// Timed: the median time of one launch sequence (the level call).
    Measured(Duration),
    /// Registered but not timed: past the deadline, or it does not fit.
    Skipped(String),
    /// Registered, but a launch failed.
    Failed(String),
    /// Not registered (module documentation, "Registration").
    Unregistered(String),
}

impl fmt::Display for Timing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Measured(time) => write!(f, "{:.1} µs", time.as_secs_f64() * 1e6),
            Self::Skipped(reason) => write!(f, "skipped: {reason}"),
            Self::Failed(reason) => write!(f, "failed: {reason}"),
            Self::Unregistered(reason) => write!(f, "not registered: {reason}"),
        }
    }
}

/// Where a decision's choice came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    /// The static rule: no directory, a decision the tuner does not take, or no time left.
    Static,
    /// The tuning cache.
    Cached,
    /// Tuned in this build.
    Tuned,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Static => "static rule",
            Self::Cached => "cache",
            Self::Tuned => "tuned",
        })
    }
}

/// One decision of a build, with where its choice came from and what was measured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionReport {
    /// The decision.
    pub decision: Decision,
    /// The gradients setting of the build that tuned it (part of the key).
    pub gradients: bool,
    /// Where the choice came from.
    pub source: Source,
    /// The choice.
    pub choice: Candidate,
    /// The level timed, if any.
    pub level: Option<usize>,
    /// The pairs of the level timed (V or octant pairs, or the near-field pairs of the
    /// level's leaves).
    pub pairs: usize,
    /// Whether the deadline cut the candidates (only some were timed).
    pub cut: bool,
    /// The candidates in the order they were registered or offered, with their timing.
    pub times: Vec<(Candidate, Timing)>,
    /// Why the static rule applies, where it does.
    pub note: Option<String>,
}

impl DecisionReport {
    /// The fastest measured candidate's time, if any.
    pub fn best(&self) -> Option<Duration> {
        self.times
            .iter()
            .filter_map(|(_, t)| match t {
                Timing::Measured(d) => Some(*d),
                _ => None,
            })
            .min()
    }
}

/// What the build found in the tuning cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheState {
    /// No directory was given: no tuning, the static rule.
    NoDirectory,
    /// The file of the key did not exist.
    Missing,
    /// The file was loaded.
    Loaded,
    /// The file was rejected (stale or corrupt) and is replaced by this build's tuning.
    Rejected(String),
}

impl fmt::Display for CacheState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDirectory => f.write_str("no tuning-cache directory: static rule"),
            Self::Missing => f.write_str("no file for the key"),
            Self::Loaded => f.write_str("loaded"),
            Self::Rejected(reason) => write!(f, "rejected ({reason})"),
        }
    }
}

/// The key of a tuning-cache file and of its entries (module documentation, "The cache").
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TuningKey {
    /// The backend.
    pub backend: BackendKind,
    /// The device's name.
    pub device: String,
    /// The runtime and compiler.
    pub compiler: String,
    /// The precision of the FMM.
    pub precision: Precision,
    /// The degree.
    pub p: usize,
    /// Whether the FMM computes gradients (a key of the entries within the file).
    pub gradients: bool,
}

impl TuningKey {
    /// The key of an FMM of degree `p` in `precision` on the device `info`.
    pub fn new(info: &DeviceInfo, precision: Precision, p: usize, gradients: bool) -> Self {
        Self {
            backend: info.backend,
            device: info.name.clone(),
            compiler: info.compiler.clone(),
            precision,
            p,
            gradients,
        }
    }

    /// The file name: `tuning-<backend>-<device>-<hash>-<precision>-p<pp>.txt`, the device
    /// name reduced to lower-case letters, digits and dashes, and the hash (FNV-1a, 32
    /// bits) of the full device name and compiler, so that different devices never share
    /// a file.
    pub fn file_name(&self) -> String {
        let slug: String = self
            .device
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let slug = slug
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-");
        let hash = fnv1a(format!("{}\n{}", self.device, self.compiler).as_bytes());
        format!(
            "tuning-{}-{slug}-{:08x}-{}-p{:02}.txt",
            self.backend, hash as u32, self.precision, self.p
        )
    }

    /// The header lines of a file of this key: every field the loader checks.
    fn header(&self) -> Vec<(&'static str, String)> {
        vec![
            ("format", FORMAT_VERSION.to_string()),
            ("convention", CONVENTION_VERSION.to_string()),
            ("candidates", CANDIDATE_SET_VERSION.to_string()),
            ("cubecl", nd_fmm_kernels::CUBECL_VERSION.to_owned()),
            ("kernels", nd_fmm_kernels::VERSION.to_owned()),
            ("backend", self.backend.to_string()),
            ("device", self.device.clone()),
            ("compiler", self.compiler.clone()),
            ("precision", self.precision.to_string()),
            ("p", self.p.to_string()),
        ]
    }
}

/// What the tuner did in one build: the key, the cache, every decision and the time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningReport {
    /// The key of the build.
    pub key: TuningKey,
    /// The cache file, with a directory.
    pub file: Option<PathBuf>,
    /// What the build found in the cache.
    pub cache: CacheState,
    /// Every decision of the build, in the order taken.
    pub decisions: Vec<DecisionReport>,
    /// The time spent tuning (tables built or loaded for a candidate, plans, launches).
    pub time: Duration,
    /// The budget.
    pub budget: Duration,
    /// When the last candidate started, within the tuning time: before the budget.
    pub last_start: Duration,
    /// Whether the deadline cut a decision or left one untuned.
    pub deadline_hit: bool,
    /// The result of storing the tuned entries: `None` if nothing was stored.
    pub stored: Option<Result<(), String>>,
}

impl TuningReport {
    /// The decision `decision`, if the build took it.
    pub fn decision(&self, decision: Decision) -> Option<&DecisionReport> {
        self.decisions.iter().find(|d| d.decision == decision)
    }

    /// Whether any candidate was timed in this build.
    pub fn tuned(&self) -> bool {
        self.decisions.iter().any(|d| d.source == Source::Tuned)
    }
}

impl fmt::Display for TuningReport {
    /// A summary line and one line per decision.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tuning: cache {}; {:.3} s of a {:.0} s budget",
            self.cache,
            self.time.as_secs_f64(),
            self.budget.as_secs_f64()
        )?;
        if self.deadline_hit {
            f.write_str(", deadline reached")?;
        }
        match &self.stored {
            Some(Ok(())) => f.write_str("; stored")?,
            Some(Err(error)) => write!(f, "; not stored: {error}")?,
            None => {}
        }
        for d in &self.decisions {
            write!(f, "\n  {}: {} ({}", d.decision, d.choice, d.source)?;
            if let Some(level) = d.level {
                write!(f, ", level {level}, {} pairs", d.pairs)?;
            }
            if d.cut {
                f.write_str(", cut by the deadline")?;
            }
            if let Some(note) = &d.note {
                write!(f, "; {note}")?;
            }
            f.write_str(")")?;
        }
        Ok(())
    }
}

/// A test aid (README T12, "Tests that define done"): adjusts the measured time of a
/// candidate, and offers extra candidates to a decision, which go through registration
/// like every other. Default: neither. Never used outside tests and reports.
#[derive(Clone, Default)]
pub struct TuningHook {
    adjust: Option<Arc<AdjustFn>>,
    offer: Option<Arc<OfferFn>>,
}

/// The time adjustment of a [`TuningHook`].
type AdjustFn = dyn Fn(&Decision, &Candidate, Duration) -> Duration + Send + Sync;

/// The candidate offer of a [`TuningHook`].
type OfferFn = dyn Fn(&Decision) -> Vec<Candidate> + Send + Sync;

impl fmt::Debug for TuningHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TuningHook")
            .field("adjust", &self.adjust.is_some())
            .field("offer", &self.offer.is_some())
            .finish()
    }
}

impl TuningHook {
    /// A hook that changes nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces each measured time by `adjust(decision, candidate, measured)`, for example
    /// to make one candidate artificially slow. It runs inside the tuning time.
    pub fn adjust(
        mut self,
        adjust: impl Fn(&Decision, &Candidate, Duration) -> Duration + Send + Sync + 'static,
    ) -> Self {
        self.adjust = Some(Arc::new(adjust));
        self
    }

    /// Offers `offer(decision)` to each decision, after its own candidates.
    pub fn offer(
        mut self,
        offer: impl Fn(&Decision) -> Vec<Candidate> + Send + Sync + 'static,
    ) -> Self {
        self.offer = Some(Arc::new(offer));
        self
    }
}

/// The static strategy of the device path for `Auto` (module documentation, "The static
/// rule"): `Dense` in f32; in f64 `Dense` for p ≤ 11 and `Rotation` from p = 12.
pub fn static_strategy(precision: Precision, p: usize) -> M2lStrategy {
    match precision {
        Precision::F32 => M2lStrategy::Dense,
        Precision::F64 if p <= STATIC_F64_DENSE_MAX_P => M2lStrategy::Dense,
        Precision::F64 => M2lStrategy::Rotation,
    }
}

/// Whether the library GEMM is a candidate for tables of order n in `precision` on
/// `backend`: f32, p ≥ 8 (n ≥ 81), a GPU (`PlanSettings::library_candidate`).
pub fn library_applies(backend: BackendKind, precision: Precision, n: usize) -> bool {
    precision == Precision::F32 && n >= 81 && backend.is_gpu()
}

/// The GEMM of a grouped level call of `kind` under the builder's `setting` (module
/// documentation, "The static rule"): under `DeviceGemm::Auto` the library for M2M and
/// L2L where it applies and the hand-written kernel otherwise and for M2L; under
/// `Library` the library wherever it applies; under `HandWritten` the hand-written
/// kernel; always box-major, in the backend's default layout and with the default chunk
/// budget (or `budget`).
pub fn static_gemm(
    info: &DeviceInfo,
    precision: Precision,
    n: usize,
    kind: OperatorKind,
    setting: DeviceGemm,
    budget: Option<u64>,
) -> GemmChoice {
    let library = library_applies(info.backend, precision, n)
        && match setting {
            DeviceGemm::Auto => kind != OperatorKind::M2l,
            DeviceGemm::Library => true,
            DeviceGemm::HandWritten => false,
        };
    let mut choice = if library {
        GemmChoice::library(precision)
    } else {
        GemmChoice::hand_written(GemmLayout::default_for(info, n))
    };
    if let Some(budget) = budget {
        choice.budget = budget;
    }
    choice
}

/// The P2P layout of the static rule: `P2pLayout::default_for`.
pub fn static_p2p(info: &DeviceInfo) -> P2pLayout {
    P2pLayout::default_for(info)
}

/// The GEMM candidates of a grouped level call of `kind` for tables of order n, the
/// static rule's first (module documentation, "What is chosen"): the hand-written kernel
/// in the backend's default layout and two others (on a GPU 2 and 8 columns per unit; on
/// the CPU runtime blocks of 4 rows of 8 columns and of 16 rows of 2 columns); the library
/// where it applies; with `budget_tunable`, the static choice with a chunk budget of
/// [`SMALL_SCRATCH_BYTES`]. Box-major throughout (module documentation, "The
/// coefficient-major layout").
pub fn gemm_candidates(
    info: &DeviceInfo,
    precision: Precision,
    n: usize,
    kind: OperatorKind,
    budget_tunable: bool,
) -> Vec<GemmChoice> {
    let first = static_gemm(info, precision, n, kind, DeviceGemm::Auto, None);
    let default = GemmLayout::default_for(info, n);
    let mut out = vec![first, GemmChoice::hand_written(default)];
    let others = match default {
        GemmLayout::Cube {
            rows,
            columns,
            per_unit,
        } => vec![
            GemmLayout::Cube {
                rows,
                columns,
                per_unit: (per_unit / 2).max(1),
            },
            GemmLayout::Cube {
                rows,
                columns,
                per_unit: per_unit * 2,
            },
        ],
        GemmLayout::Cpu { .. } => {
            let rows = |b: u32| b.min(n.max(1) as u32);
            vec![
                GemmLayout::Cpu {
                    block: rows(CPU_GEMM_BLOCK / 2),
                    per_unit: 8,
                },
                GemmLayout::Cpu {
                    block: rows(CPU_GEMM_BLOCK * 2),
                    per_unit: 2,
                },
            ]
        }
    };
    out.extend(others.into_iter().map(GemmChoice::hand_written));
    if library_applies(info.backend, precision, n) {
        out.push(GemmChoice::library(precision));
    }
    if budget_tunable {
        out.push(GemmChoice {
            budget: SMALL_SCRATCH_BYTES,
            ..first
        });
    }
    dedup(out)
}

/// The P2P candidates of a device, the static rule's first (module documentation).
pub fn p2p_candidates(info: &DeviceInfo) -> Vec<P2pLayout> {
    let first = static_p2p(info);
    let mut out = vec![first];
    match info.backend {
        BackendKind::Cpu => {
            for bits in [64, 128, 256] {
                if bits != CPU_VECTOR_BITS {
                    out.push(P2pLayout::Cpu { vector_bits: bits });
                }
            }
        }
        BackendKind::Metal | BackendKind::Cuda => {
            for units in [32, 64, 128] {
                out.push(P2pLayout::Cube { units });
            }
            for planes in [2, 4] {
                out.push(P2pLayout::Plane { planes });
            }
        }
    }
    dedup(out)
}

/// `items` without repeats, in first-seen order.
fn dedup<T: PartialEq>(items: Vec<T>) -> Vec<T> {
    let mut out: Vec<T> = Vec::with_capacity(items.len());
    for item in items {
        if !out.contains(&item) {
            out.push(item);
        }
    }
    out
}

/// Why a candidate was not timed, from the function that times it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Rejection {
    /// Not registered: the device cannot run it, or the library refused the shape.
    Unregistered(String),
    /// It does not fit (or another permanent reason): registered, not timed.
    Skipped(String),
    /// A launch failed.
    Failed(String),
}

impl From<KernelError> for Rejection {
    fn from(error: KernelError) -> Self {
        Self::Failed(error.to_string())
    }
}

/// Times `launch` (one level call) on `device` (module documentation, "When tuning
/// runs"): one launch and a sync first, then batches of launches between syncs, doubling
/// until a batch takes [`MIN_BATCH`], then [`BATCHES`] batches; the median time per
/// launch. Past `deadline` it stops after the current batch (at least one is timed).
pub(crate) fn time_launches(
    device: &mut Device,
    deadline: Instant,
    mut launch: impl FnMut(&mut Device) -> Result<(), KernelError>,
) -> Result<Duration, KernelError> {
    launch(device)?;
    device.sync()?;
    let mut batch = |device: &mut Device, calls: u32| -> Result<Duration, KernelError> {
        let start = Instant::now();
        for _ in 0..calls {
            launch(device)?;
        }
        device.sync()?;
        Ok(start.elapsed())
    };
    let mut calls = 1u32;
    loop {
        let took = batch(device, calls)?;
        if took >= MIN_BATCH || Instant::now() >= deadline || calls >= 1 << 20 {
            break;
        }
        calls *= 2;
    }
    let mut times = Vec::with_capacity(BATCHES);
    for _ in 0..BATCHES {
        times.push(batch(device, calls)? / calls);
        if Instant::now() >= deadline {
            break;
        }
    }
    times.sort();
    Ok(times[times.len() / 2])
}

/// The tuner of one build: the key, the cache and its entries, the budget and the hook,
/// and the decisions taken so far (module documentation).
#[derive(Debug)]
pub struct Tuner {
    key: TuningKey,
    cache: Option<TuningCache>,
    state: CacheState,
    /// The entries of the cache file, every gradients setting (none if rejected).
    loaded: Vec<DecisionReport>,
    /// The entries tuned by this build, to store.
    tuned: Vec<DecisionReport>,
    hook: TuningHook,
    budget: Duration,
    /// Tuning time before the current phase.
    spent: Duration,
    /// The start of the current phase, if one is running.
    phase: Option<Instant>,
    last_start: Duration,
    deadline_hit: bool,
    decisions: Vec<DecisionReport>,
    stored: Option<Result<(), String>>,
}

impl Tuner {
    /// The tuner of a build with `key`, the tuning-cache directory `dir` (none: the static
    /// rule), the budget and the hook. Reads the cache file of the key, if any.
    pub fn new(key: TuningKey, dir: Option<&Path>, budget: Duration, hook: TuningHook) -> Self {
        let cache = dir.map(TuningCache::new);
        let (state, loaded) = match &cache {
            None => (CacheState::NoDirectory, Vec::new()),
            Some(cache) => match cache.load(&key) {
                Ok(entries) => (CacheState::Loaded, entries),
                Err(TuningCacheError::Missing) => (CacheState::Missing, Vec::new()),
                Err(error) => (CacheState::Rejected(error.to_string()), Vec::new()),
            },
        };
        Self {
            key,
            cache,
            state,
            loaded,
            tuned: Vec::new(),
            hook,
            budget,
            spent: Duration::ZERO,
            phase: None,
            last_start: Duration::ZERO,
            deadline_hit: false,
            decisions: Vec::new(),
            stored: None,
        }
    }

    /// The key.
    pub fn key(&self) -> &TuningKey {
        &self.key
    }

    /// Whether a directory was given, so that missing decisions are tuned.
    pub fn enabled(&self) -> bool {
        self.cache.is_some()
    }

    /// The cached choice of `decision` for this build's gradients setting.
    pub fn cached(&self, decision: Decision) -> Option<Candidate> {
        self.find(&self.loaded, decision).map(|e| e.choice)
    }

    /// The choice of `decision` made earlier in this build (tuned, cached or static), if
    /// any.
    pub fn decided(&self, decision: Decision) -> Option<Candidate> {
        self.decisions
            .iter()
            .find(|d| d.decision == decision)
            .map(|d| d.choice)
    }

    fn find<'a>(
        &self,
        entries: &'a [DecisionReport],
        decision: Decision,
    ) -> Option<&'a DecisionReport> {
        entries
            .iter()
            .find(|e| e.decision == decision && e.gradients == self.key.gradients)
    }

    /// Starts a tuning phase: its time counts toward the budget. Without a directory
    /// nothing is tuned, and no time counts.
    pub(crate) fn start_phase(&mut self) {
        if self.enabled() {
            self.phase = Some(Instant::now());
        }
    }

    /// Ends a tuning phase.
    pub(crate) fn end_phase(&mut self) {
        if let Some(start) = self.phase.take() {
            self.spent += start.elapsed();
        }
    }

    /// The tuning time so far.
    fn elapsed(&self) -> Duration {
        self.spent + self.phase.map_or(Duration::ZERO, |s| s.elapsed())
    }

    /// The deadline of the running phase.
    fn deadline(&self) -> Instant {
        let now = Instant::now();
        now + self.budget.saturating_sub(self.elapsed())
    }

    fn over(&self) -> bool {
        self.elapsed() >= self.budget
    }

    /// Records a decision taken by the static rule, with the reason.
    pub(crate) fn record_static(
        &mut self,
        decision: Decision,
        choice: Candidate,
        note: impl Into<Option<String>>,
    ) -> Candidate {
        self.decisions.push(DecisionReport {
            decision,
            gradients: self.key.gradients,
            source: Source::Static,
            choice,
            level: None,
            pairs: 0,
            cut: false,
            times: Vec::new(),
            note: note.into(),
        });
        choice
    }

    /// Records `report` as a decision of this build, tuned now (it is stored).
    fn record_tuned(&mut self, report: DecisionReport) {
        self.tuned.push(report.clone());
        self.decisions.push(report);
    }

    /// Decides `decision` (module documentation): from earlier in the build, from the
    /// cache, by the static rule (no directory, too small, or past the deadline: `static`
    /// is the static rule's candidate, the first of `candidates`), or by timing each
    /// registered candidate in order with `measure` and taking the fastest. `level` and
    /// `pairs` describe the level timed.
    pub(crate) fn decide(
        &mut self,
        decision: Decision,
        (level, pairs): (Option<usize>, usize),
        candidates: Vec<Candidate>,
        mut measure: impl FnMut(&Candidate, Instant) -> Result<Duration, Rejection>,
    ) -> Candidate {
        let first = candidates[0];
        if let Some(choice) = self.decided(decision) {
            return choice;
        }
        if let Some(entry) = self.find(&self.loaded, decision).cloned() {
            self.decisions.push(DecisionReport {
                source: Source::Cached,
                ..entry
            });
            return entry.choice;
        }
        if !self.enabled() {
            return self.record_static(decision, first, None);
        }
        if pairs < MIN_TUNED_PAIRS && !matches!(decision, Decision::P2p { .. }) {
            return self.record_static(
                decision,
                first,
                format!("fewer than {MIN_TUNED_PAIRS} pairs: not tuned"),
            );
        }
        if self.over() {
            self.deadline_hit = true;
            return self.record_static(
                decision,
                first,
                "past the tuning deadline: not tuned, not stored".to_owned(),
            );
        }
        let mut offered = candidates;
        if let Some(offer) = &self.hook.offer {
            offered.extend(offer(&decision));
        }
        let precision = self.key.precision;
        let mut times = Vec::with_capacity(offered.len());
        let mut cut = false;
        for candidate in offered {
            // The input-precision guard: a library that rounds T's inputs is never
            // registered.
            if let Some(GemmChoice {
                gemm: GemmKind::Library { inputs },
                ..
            }) = candidate.gemm()
                && !inputs.keeps(precision)
            {
                times.push((
                    candidate,
                    Timing::Unregistered(format!(
                        "input-precision guard: {} inputs for {precision} data",
                        inputs.name()
                    )),
                ));
                continue;
            }
            let measured = times.iter().any(|(_, t)| matches!(t, Timing::Measured(_)));
            if self.over() && measured {
                cut = true;
                self.deadline_hit = true;
                times.push((
                    candidate,
                    Timing::Skipped("past the tuning deadline".into()),
                ));
                continue;
            }
            self.last_start = self.elapsed();
            let timing = match measure(&candidate, self.deadline()) {
                Ok(time) => Timing::Measured(match &self.hook.adjust {
                    Some(adjust) => adjust(&decision, &candidate, time),
                    None => time,
                }),
                Err(Rejection::Unregistered(reason)) => Timing::Unregistered(reason),
                Err(Rejection::Skipped(reason)) => Timing::Skipped(reason),
                Err(Rejection::Failed(reason)) => Timing::Failed(reason),
            };
            times.push((candidate, timing));
        }
        let best = times
            .iter()
            .filter_map(|(c, t)| match t {
                Timing::Measured(d) => Some((*c, *d)),
                _ => None,
            })
            .min_by_key(|&(_, d)| d);
        let Some((choice, _)) = best else {
            self.decisions.push(DecisionReport {
                decision,
                gradients: self.key.gradients,
                source: Source::Static,
                choice: first,
                level,
                pairs,
                cut,
                times,
                note: Some("no candidate could be timed: not stored".into()),
            });
            return first;
        };
        self.record_tuned(DecisionReport {
            decision,
            gradients: self.key.gradients,
            source: Source::Tuned,
            choice,
            level,
            pairs,
            cut,
            times,
            note: None,
        });
        choice
    }

    /// Where the strategy was tuned now and `Dense` won: records the GEMM it won with as
    /// the decision of the M2L level calls in `bucket` (the largest V level's), with the
    /// strategy's times of the dense candidates, so that they take it without timing it
    /// again.
    pub(crate) fn record_dense_gemm(&mut self, bucket: u64) {
        let Some(strategy) = self
            .decisions
            .iter()
            .find(|d| d.decision == Decision::Strategy && d.source == Source::Tuned)
            .cloned()
        else {
            return;
        };
        let Candidate::Dense(choice) = strategy.choice else {
            return;
        };
        let decision = Decision::Gemm {
            kind: OperatorKind::M2l,
            bucket,
        };
        if self.decided(decision).is_some() {
            return;
        }
        let times = strategy
            .times
            .iter()
            .filter_map(|(c, t)| match c {
                Candidate::Dense(g) => Some((Candidate::Gemm(*g), t.clone())),
                _ => None,
            })
            .collect();
        self.record_tuned(DecisionReport {
            decision,
            choice: Candidate::Gemm(choice),
            times,
            note: Some("timed with the strategy".into()),
            ..strategy
        });
    }

    /// Stores the entries tuned by this build with those of the file (an entry of the same
    /// gradients and decision replaced), if any were tuned; and ends the tuning.
    pub(crate) fn finish(mut self) -> TuningReport {
        self.end_phase();
        if let Some(cache) = &self.cache
            && !self.tuned.is_empty()
        {
            let mut entries: Vec<DecisionReport> = self
                .loaded
                .iter()
                .filter(|e| {
                    !self
                        .tuned
                        .iter()
                        .any(|t| t.decision == e.decision && t.gradients == e.gradients)
                })
                .cloned()
                .collect();
            entries.extend(self.tuned.iter().cloned());
            self.stored = Some(cache.store(&self.key, &entries).map_err(|e| e.to_string()));
        }
        TuningReport {
            file: self.cache.as_ref().map(|c| c.path(&self.key)),
            key: self.key,
            cache: self.state,
            decisions: self.decisions,
            time: self.spent,
            budget: self.budget,
            last_start: self.last_start,
            deadline_hit: self.deadline_hit,
            stored: self.stored,
        }
    }
}

/// Why a tuning-cache file was not used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TuningCacheError {
    /// There is no file for the key.
    Missing,
    /// The file could not be read or written.
    Io(String),
    /// The file is not a whole tuning-cache file: magic, checksum or syntax.
    Corrupt(String),
    /// The file is of another key or version.
    Stale(String),
}

impl fmt::Display for TuningCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => f.write_str("no file"),
            Self::Io(error) => write!(f, "i/o: {error}"),
            Self::Corrupt(reason) => write!(f, "corrupt: {reason}"),
            Self::Stale(reason) => write!(f, "stale: {reason}"),
        }
    }
}

impl std::error::Error for TuningCacheError {}

/// The tuning cache in a directory the caller chose (module documentation, "The cache").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuningCache {
    dir: PathBuf,
}

/// Distinguishes the temporary files of one process.
static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

impl TuningCache {
    /// The cache in `dir`. Nothing is touched until a load or a store; a store creates the
    /// directory if it is missing.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The path of the file of `key` ([`TuningKey::file_name`]).
    pub fn path(&self, key: &TuningKey) -> PathBuf {
        self.dir.join(key.file_name())
    }

    /// The entries of the file of `key`, every gradients setting.
    ///
    /// # Errors
    ///
    /// [`TuningCacheError::Missing`] without a file; `Corrupt` for a file whose magic,
    /// checksum or syntax is wrong (truncated or changed); `Stale` for a file of another
    /// key or version; `Io` if it cannot be read. Never panics on the file's contents.
    pub fn load(&self, key: &TuningKey) -> Result<Vec<DecisionReport>, TuningCacheError> {
        let path = self.path(key);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(TuningCacheError::Missing);
            }
            Err(error) => return Err(TuningCacheError::Io(format!("{}: {error}", path.display()))),
        };
        decode(&bytes, key)
    }

    /// Stores `entries` as the file of `key`, atomically: written to a new temporary file
    /// in the directory, synced, renamed over the final name. Creates the directory if
    /// needed; on failure removes the temporary file and leaves the final name as it was.
    ///
    /// # Errors
    ///
    /// [`TuningCacheError::Io`] if the directory or a file cannot be created, written or
    /// renamed.
    pub fn store(
        &self,
        key: &TuningKey,
        entries: &[DecisionReport],
    ) -> Result<(), TuningCacheError> {
        let io = |what: &Path| {
            let what = what.to_path_buf();
            move |error: io::Error| TuningCacheError::Io(format!("{}: {error}", what.display()))
        };
        fs::create_dir_all(&self.dir).map_err(io(&self.dir))?;
        let path = self.path(key);
        let bytes = encode(key, entries);
        let (temporary, mut file) = loop {
            let counter = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!(".{}.{}-{counter}.tmp", key.file_name(), std::process::id());
            let temporary = self.dir.join(name);
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
            {
                Ok(file) => break (temporary, file),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(io(&temporary)(error)),
            }
        };
        let written = file
            .write_all(&bytes)
            .and_then(|()| file.sync_all())
            .map_err(io(&temporary))
            .and_then(|()| fs::rename(&temporary, &path).map_err(io(&path)));
        if written.is_err() {
            // Best effort: the error of the store is the one to report.
            let _ = fs::remove_file(&temporary);
        }
        written
    }
}

/// FNV-1a, 64 bits.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// The text of a file: `NDFMMTUN`, the checksum of the body, the body.
fn encode(key: &TuningKey, entries: &[DecisionReport]) -> Vec<u8> {
    let mut body = String::new();
    for (name, value) in key.header() {
        body.push_str(&format!("{name} {value}\n"));
    }
    for e in entries {
        body.push_str(&format!(
            "entry {} {} {} {} {} {}\n",
            u8::from(e.gradients),
            encode_decision(&e.decision),
            if e.cut { "cut" } else { "whole" },
            e.level.map_or("-".to_owned(), |l| l.to_string()),
            e.pairs,
            encode_candidate(&e.choice),
        ));
        for (candidate, timing) in &e.times {
            body.push_str(&format!(
                "time {} {}\n",
                encode_candidate(candidate),
                encode_timing(timing)
            ));
        }
    }
    format!("{MAGIC}\nchecksum {:016x}\n{body}", fnv1a(body.as_bytes())).into_bytes()
}

/// The entries of a file of `key` (module documentation, "The cache").
fn decode(bytes: &[u8], key: &TuningKey) -> Result<Vec<DecisionReport>, TuningCacheError> {
    let corrupt = |reason: &str| TuningCacheError::Corrupt(reason.to_owned());
    let text = std::str::from_utf8(bytes).map_err(|_| corrupt("not UTF-8"))?;
    let (magic, rest) = text
        .split_once('\n')
        .ok_or_else(|| corrupt("no magic line"))?;
    if magic != MAGIC {
        return Err(corrupt("wrong magic"));
    }
    let (checksum, body) = rest
        .split_once('\n')
        .ok_or_else(|| corrupt("no checksum line"))?;
    let checksum = checksum
        .strip_prefix("checksum ")
        .and_then(|h| u64::from_str_radix(h, 16).ok())
        .ok_or_else(|| corrupt("malformed checksum line"))?;
    if fnv1a(body.as_bytes()) != checksum {
        return Err(corrupt("checksum mismatch"));
    }
    let mut lines = body.lines();
    for (name, want) in key.header() {
        let line = lines
            .next()
            .ok_or_else(|| corrupt("the header ends early"))?;
        let (field, value) = line.split_once(' ').unwrap_or((line, ""));
        if field != name {
            return Err(corrupt(&format!(
                "header field `{field}`, expected `{name}`"
            )));
        }
        if value != want {
            return Err(TuningCacheError::Stale(format!(
                "{name} is `{value}`, expected `{want}`"
            )));
        }
    }
    let mut entries: Vec<DecisionReport> = Vec::new();
    for line in lines {
        let (kind, rest) = line
            .split_once(' ')
            .ok_or_else(|| corrupt("a malformed line"))?;
        match kind {
            "entry" => {
                let fields: Vec<&str> = rest.split(' ').collect();
                let [gradients, decision, cut, level, pairs, choice] = fields[..] else {
                    return Err(corrupt("a malformed entry"));
                };
                let entry = DecisionReport {
                    decision: decode_decision(decision).ok_or_else(|| corrupt("a decision"))?,
                    gradients: match gradients {
                        "0" => false,
                        "1" => true,
                        _ => return Err(corrupt("the gradients of an entry")),
                    },
                    source: Source::Cached,
                    choice: decode_candidate(choice).ok_or_else(|| corrupt("a choice"))?,
                    level: match level {
                        "-" => None,
                        l => Some(l.parse().map_err(|_| corrupt("a level"))?),
                    },
                    pairs: pairs
                        .parse()
                        .map_err(|_| corrupt("the pairs of an entry"))?,
                    cut: match cut {
                        "cut" => true,
                        "whole" => false,
                        _ => return Err(corrupt("the cut flag of an entry")),
                    },
                    times: Vec::new(),
                    note: None,
                };
                entries.push(entry);
            }
            "time" => {
                let (candidate, timing) = rest
                    .split_once(' ')
                    .ok_or_else(|| corrupt("a malformed time"))?;
                let entry = entries
                    .last_mut()
                    .ok_or_else(|| corrupt("a time before any entry"))?;
                entry.times.push((
                    decode_candidate(candidate).ok_or_else(|| corrupt("a timed candidate"))?,
                    decode_timing(timing).ok_or_else(|| corrupt("a timing"))?,
                ));
            }
            _ => return Err(corrupt(&format!("an unknown line `{kind}`"))),
        }
    }
    Ok(entries)
}

/// `strategy`, `gemm:M2L:65536` or `p2p:64`.
fn encode_decision(decision: &Decision) -> String {
    match decision {
        Decision::Strategy => "strategy".into(),
        Decision::Gemm { kind, bucket } => format!("gemm:{kind}:{bucket}"),
        Decision::P2p { bucket } => format!("p2p:{bucket}"),
    }
}

fn decode_decision(text: &str) -> Option<Decision> {
    let parts: Vec<&str> = text.split(':').collect();
    match parts[..] {
        ["strategy"] => Some(Decision::Strategy),
        ["gemm", kind, bucket] => Some(Decision::Gemm {
            kind: [OperatorKind::M2m, OperatorKind::L2l, OperatorKind::M2l]
                .into_iter()
                .find(|k| k.name() == kind)?,
            bucket: bucket.parse().ok()?,
        }),
        ["p2p", bucket] => Some(Decision::P2p {
            bucket: bucket.parse().ok()?,
        }),
        _ => None,
    }
}

/// `hand/cube/32/2/4/bm/134217728`, `hand/cpu/8/4/bm/…` or `library/f32/cm/…`.
fn encode_gemm(choice: &GemmChoice) -> String {
    let gemm = match choice.gemm {
        GemmKind::HandWritten(GemmLayout::Cube {
            rows,
            columns,
            per_unit,
        }) => format!("hand/cube/{rows}/{columns}/{per_unit}"),
        GemmKind::HandWritten(GemmLayout::Cpu { block, per_unit }) => {
            format!("hand/cpu/{block}/{per_unit}")
        }
        GemmKind::Library { inputs } => format!("library/{}", inputs.name()),
    };
    let orientation = match choice.orientation {
        Orientation::BoxMajor => "bm",
        Orientation::CoefficientMajor => "cm",
    };
    format!("{gemm}/{orientation}/{}", choice.budget)
}

fn decode_gemm(text: &str) -> Option<GemmChoice> {
    let parts: Vec<&str> = text.split('/').collect();
    let (gemm, rest) = match parts[..] {
        ["hand", "cube", rows, columns, per_unit, ref rest @ ..] => (
            GemmKind::HandWritten(GemmLayout::Cube {
                rows: rows.parse().ok()?,
                columns: columns.parse().ok()?,
                per_unit: per_unit.parse().ok()?,
            }),
            rest,
        ),
        ["hand", "cpu", block, per_unit, ref rest @ ..] => (
            GemmKind::HandWritten(GemmLayout::Cpu {
                block: block.parse().ok()?,
                per_unit: per_unit.parse().ok()?,
            }),
            rest,
        ),
        ["library", inputs, ref rest @ ..] => (
            GemmKind::Library {
                inputs: InputPrecision::parse(inputs)?,
            },
            rest,
        ),
        _ => return None,
    };
    let [orientation, budget] = rest[..] else {
        return None;
    };
    Some(GemmChoice {
        gemm,
        orientation: match orientation {
            "bm" => Orientation::BoxMajor,
            "cm" => Orientation::CoefficientMajor,
            _ => return None,
        },
        budget: budget.parse().ok()?,
    })
}

/// `dense=<gemm>`, `rotation`, `gemm=<gemm>`, `p2p=cube/64`, `p2p=plane/2`, `p2p=cpu/128`.
fn encode_candidate(candidate: &Candidate) -> String {
    match candidate {
        Candidate::Dense(choice) => format!("dense={}", encode_gemm(choice)),
        Candidate::Rotation => "rotation".into(),
        Candidate::Gemm(choice) => format!("gemm={}", encode_gemm(choice)),
        Candidate::P2p(P2pLayout::Cube { units }) => format!("p2p=cube/{units}"),
        Candidate::P2p(P2pLayout::Plane { planes }) => format!("p2p=plane/{planes}"),
        Candidate::P2p(P2pLayout::Cpu { vector_bits }) => format!("p2p=cpu/{vector_bits}"),
    }
}

fn decode_candidate(text: &str) -> Option<Candidate> {
    if text == "rotation" {
        return Some(Candidate::Rotation);
    }
    let (kind, value) = text.split_once('=')?;
    match kind {
        "dense" => Some(Candidate::Dense(decode_gemm(value)?)),
        "gemm" => Some(Candidate::Gemm(decode_gemm(value)?)),
        "p2p" => {
            let (layout, number) = value.split_once('/')?;
            let number: u32 = number.parse().ok()?;
            Some(Candidate::P2p(match layout {
                "cube" => P2pLayout::Cube { units: number },
                "plane" => P2pLayout::Plane { planes: number },
                "cpu" => P2pLayout::Cpu {
                    vector_bits: number,
                },
                _ => return None,
            }))
        }
        _ => None,
    }
}

/// `measured <ns>`, `skipped <reason>`, `failed <reason>` or `unregistered <reason>`; a
/// reason on one line.
fn encode_timing(timing: &Timing) -> String {
    let line = |reason: &str| reason.replace(['\n', '\r'], " ");
    match timing {
        Timing::Measured(time) => format!("measured {}", time.as_nanos()),
        Timing::Skipped(reason) => format!("skipped {}", line(reason)),
        Timing::Failed(reason) => format!("failed {}", line(reason)),
        Timing::Unregistered(reason) => format!("unregistered {}", line(reason)),
    }
}

fn decode_timing(text: &str) -> Option<Timing> {
    let (kind, value) = text.split_once(' ').unwrap_or((text, ""));
    match kind {
        "measured" => Some(Timing::Measured(Duration::from_nanos(value.parse().ok()?))),
        "skipped" => Some(Timing::Skipped(value.to_owned())),
        "failed" => Some(Timing::Failed(value.to_owned())),
        "unregistered" => Some(Timing::Unregistered(value.to_owned())),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(backend: BackendKind) -> DeviceInfo {
        DeviceInfo {
            backend,
            name: "Test Device (v2)".into(),
            compiler: "test<msl>".into(),
            cubecl_version: nd_fmm_kernels::CUBECL_VERSION,
            f32: true,
            f64: !backend.is_gpu(),
            plane_size: (32, 32),
            max_shared_memory: 32_768,
            max_units_per_cube: 1024,
            max_cube_count: (65_535, 65_535, 65_535),
            max_memory: None,
        }
    }

    #[test]
    fn the_static_strategy_follows_the_readme_rule() {
        for p in 0..=20 {
            assert_eq!(
                static_strategy(Precision::F32, p),
                M2lStrategy::Dense,
                "f32 p = {p}"
            );
            let want = if p <= 11 {
                M2lStrategy::Dense
            } else {
                M2lStrategy::Rotation
            };
            assert_eq!(static_strategy(Precision::F64, p), want, "f64 p = {p}");
        }
    }

    #[test]
    fn the_static_gemm_follows_the_readme_rule() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let info = info(backend);
            for p in 0..=20usize {
                let n = (p + 1) * (p + 1);
                let hand = GemmChoice::hand_written(GemmLayout::default_for(&info, n));
                for precision in [Precision::F32, Precision::F64] {
                    let library = precision == Precision::F32 && p >= 8 && backend.is_gpu();
                    for kind in [OperatorKind::M2m, OperatorKind::L2l, OperatorKind::M2l] {
                        let got = static_gemm(&info, precision, n, kind, DeviceGemm::Auto, None);
                        let want = if library && kind != OperatorKind::M2l {
                            GemmChoice::library(precision)
                        } else {
                            hand
                        };
                        assert_eq!(got, want, "{backend} {precision} p = {p} {kind}");
                        assert_eq!(got.orientation, Orientation::BoxMajor);
                        assert_eq!(got.budget, DEFAULT_SCRATCH_BYTES);
                        let forced =
                            static_gemm(&info, precision, n, kind, DeviceGemm::HandWritten, None);
                        assert_eq!(forced, hand);
                        let all = static_gemm(&info, precision, n, kind, DeviceGemm::Library, None);
                        assert_eq!(all.is_library(), library);
                    }
                }
            }
            assert_eq!(static_p2p(&info), P2pLayout::default_for(&info));
        }
    }

    #[test]
    fn candidates_start_with_the_static_rule_and_guard_the_library() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            let info = info(backend);
            for p in [3usize, 8, 12] {
                let n = (p + 1) * (p + 1);
                for precision in [Precision::F32, Precision::F64] {
                    for kind in [OperatorKind::M2m, OperatorKind::L2l, OperatorKind::M2l] {
                        let all = gemm_candidates(&info, precision, n, kind, true);
                        assert_eq!(
                            all[0],
                            static_gemm(&info, precision, n, kind, DeviceGemm::Auto, None)
                        );
                        let mut seen = all.clone();
                        seen.dedup();
                        assert_eq!(seen.len(), all.len(), "no repeats");
                        let libraries: Vec<_> = all.iter().filter(|c| c.is_library()).collect();
                        let defaults = libraries
                            .iter()
                            .filter(|c| c.budget == DEFAULT_SCRATCH_BYTES)
                            .count();
                        assert_eq!(
                            defaults,
                            usize::from(library_applies(backend, precision, n)),
                            "{backend} {precision} p = {p}"
                        );
                        assert!(
                            all.iter().all(|c| c.orientation == Orientation::BoxMajor),
                            "box-major only"
                        );
                        for c in libraries {
                            let GemmKind::Library { inputs } = c.gemm else {
                                unreachable!()
                            };
                            assert!(inputs.keeps(precision));
                        }
                    }
                }
            }
            let p2p = p2p_candidates(&info);
            assert_eq!(p2p[0], static_p2p(&info));
        }
        assert!(!InputPrecision::Tf32.keeps(Precision::F32));
        assert!(!InputPrecision::F16.keeps(Precision::F32));
        assert!(!InputPrecision::Bf16.keeps(Precision::F32));
        assert!(!InputPrecision::F32.keeps(Precision::F64));
        assert!(InputPrecision::F32.keeps(Precision::F32));
    }

    #[test]
    fn encodings_round_trip() {
        let info = info(BackendKind::Metal);
        let mut candidates = vec![Candidate::Rotation];
        for c in gemm_candidates(&info, Precision::F32, 81, OperatorKind::M2m, true) {
            candidates.push(Candidate::Gemm(c));
            candidates.push(Candidate::Dense(c));
        }
        for c in gemm_candidates(
            &self::info(BackendKind::Cpu),
            Precision::F64,
            81,
            OperatorKind::M2l,
            true,
        ) {
            candidates.push(Candidate::Gemm(c));
        }
        candidates.push(Candidate::Gemm(GemmChoice {
            gemm: GemmKind::Library {
                inputs: InputPrecision::Tf32,
            },
            ..GemmChoice::library(Precision::F32)
        }));
        for l in p2p_candidates(&info)
            .into_iter()
            .chain(p2p_candidates(&self::info(BackendKind::Cpu)))
        {
            candidates.push(Candidate::P2p(l));
        }
        for c in &candidates {
            assert_eq!(decode_candidate(&encode_candidate(c)), Some(*c), "{c}");
        }
        for d in [
            Decision::Strategy,
            Decision::Gemm {
                kind: OperatorKind::L2l,
                bucket: 4096,
            },
            Decision::P2p { bucket: 64 },
        ] {
            assert_eq!(decode_decision(&encode_decision(&d)), Some(d));
        }
        for t in [
            Timing::Measured(Duration::from_nanos(123_456)),
            Timing::Skipped("past the deadline".into()),
            Timing::Failed("a launch".into()),
            Timing::Unregistered("input-precision guard: tf32".into()),
        ] {
            assert_eq!(decode_timing(&encode_timing(&t)), Some(t));
        }
        assert_eq!(bucket(0), 1);
        assert_eq!(bucket(1), 1);
        assert_eq!(bucket(600_000), 1 << 20);
        assert_eq!(bucket(512), 512);
    }

    #[test]
    fn file_names_separate_devices_and_keys() {
        let key = TuningKey::new(&info(BackendKind::Metal), Precision::F32, 8, true);
        let name = key.file_name();
        assert!(name.starts_with("tuning-metal-test-device-v2-"), "{name}");
        assert!(name.ends_with("-f32-p08.txt"), "{name}");
        let other = TuningKey {
            compiler: "test<wgsl>".into(),
            ..key.clone()
        };
        assert_ne!(other.file_name(), name);
        let gradients = TuningKey {
            gradients: false,
            ..key.clone()
        };
        assert_eq!(
            gradients.file_name(),
            name,
            "gradients key the entries, not the file"
        );
    }
}
