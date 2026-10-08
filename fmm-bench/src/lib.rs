//! # `nd-fmm-bench`
//!
//! A benchmark of the nd-project FMM that anyone can run with one command (Phase 4S T6,
//! C4S.6): N points uniform in the unit cube, f32 or f64, an expansion degree p and a
//! backend; the evaluation time as min, median, mean, max and standard deviation, the
//! time of every operator kind, and the error against the direct sum, written as one
//! Markdown file.
//!
//! ```text
//! tools/bench/run.sh --backend host,cuda --precision f32,f64 --degree 3,6,8
//! ```
//!
//! The binary `nd-fmm-bench` parses the command line ([`options`]), initialises MPI on
//! one rank and calls [`measure::run`]; this library holds the configuration, the
//! measurement and the Markdown writer ([`report`]), so that they can be tested.
//!
//! ## The problem
//!
//! N points uniform in [0, 1)³ (`nd_fmm_validate::points::cube`), then N charges uniform
//! in [−1, 1) from the same seeded generator ([`measure::SEED`]); sources equal targets;
//! gradients on unless `--no-gradients`. The tree is adaptive, with the library's
//! default `max_level` (16) and refinement target (64 points per leaf) unless
//! `--max-level` or `--leaf-size` sets them. f32 runs round the charges to f32; the
//! points stay f64, as `FmmBuilder::build` takes them.
//!
//! ## Measurement
//!
//! Per combination of backend, precision, N and p ([`options::Options::combinations`]):
//! 1. **Build** the `Fmm` once and record its `BuildTimings`: the total, the tables and
//!    the device part (opening, uploads, tuning).
//! 2. **Warm up** with `--warmup` evaluations, which compile the device kernels and are
//!    not counted.
//! 3. **Time** `--repeats` evaluations with kind timings off: the wall time of
//!    `Fmm::evaluate` (with `--reuse-output`, of `Fmm::evaluate_into` into one reused
//!    `Output`; Phase 4S T9), as min, median, mean, max and sample standard deviation. Every
//!    output is compared bit for bit with the first; a difference flags the row
//!    ("bit-identical: **no**") rather than aborting.
//! 4. **Per kind**, unless `--kinds off`: a second `Fmm` with
//!    `FmmBuilder::kind_timings` (`sync` = `KindTiming::Synchronous`, `device` =
//!    `KindTiming::Device`), its own warm-up and `--repeats` evaluations; per kind the
//!    min, mean and max of its per-evaluation total, its share and calls, the remainder
//!    of the stages split into the charge load, the output pass and the rest ("load",
//!    "output", "other"; Phase 4S T9) and the sum against step 3's mean.
//! 5. **Accuracy**: the relative L2 error of φ and ∇φ of the first timed evaluation at
//!    `--accuracy` sampled targets against the f64 direct sum
//!    (`nd_fmm_validate::fmm_accuracy::Oracle`). A sanity column, not a gate.
//! 6. **Device facts**: the device (`DeviceInfo`), the resolved strategy, the layouts, the
//!    GEMM of every device level call (`DeviceReport`) and the transfers, launches and
//!    syncs of one evaluation.
//!
//! A combination the backend refuses (f64 on Metal, a backend not compiled in) becomes a
//! row `refused: <error>` and the run goes on. The file is rewritten after every
//! combination, so an interrupted run keeps its rows. Nothing here asserts a timing.
//!
//! ## Features
//!
//! `gpu`, `cpu`, `metal` and `cuda` pass through to `nd-fmm-exec` and
//! `nd-fmm-validate`, as in `nd-fmm-validate`. Without them only the host runs; a device
//! backend asked for is refused (`SettingsError::BackendNotCompiled`).

pub mod machine;
pub mod measure;
pub mod options;
pub mod report;
