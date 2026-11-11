//! The Helmholtz FMM against the sampled direct sum on the reference workloads
//! (Helmholtz CH3.1c; docs/design/helmholtz-fmm-proposal.md §9.1 and §10, CH3.1). Prints
//! Markdown on stdout.
//!
//! One case per kD and ε: the workload's points as sources and targets, the degrees of
//! S7's closed form at ε (or named, the same on every level), real or complex charges,
//! m vectors, gradients, against `nd_fmm_ref::helmholtz::p2p::direct_sum` in f64 at the
//! sampled targets (1,000 by default). Each case runs at every thread count of
//! `--threads`, and each row prints the tree, the degrees, the build and evaluation times
//! (slowest rank), the relative ℓ2 errors of φ and ∇φ over all vectors, and the
//! rank-invariant checksum of the whole output: equal checksums across thread counts and
//! rank counts show the run deterministic (`nd_fmm_validate::helmholtz_accuracy`).
//!
//! Options (each value may be a comma-separated list where marked):
//!
//! - `--workload w`: `w1` (default), `w2`, `w3` (ten points per wavelength), `w3:<n>`,
//!   `w4-<cube|sphere|plummer|clusters>:<n>`;
//! - `--kd x,…`: Re kD (default 10); `--kim y`: Im kD (default 0);
//! - `--eps e,…`: the tolerances (default 1e-3); `--degree p` names p on every level
//!   in place of a tolerance (levels 0 and 1 hold expansions nothing reads);
//! - `--charges real|complex` (default complex), `--vectors m` (default 1),
//!   `--samples n` (default 1000), `--no-gradients`;
//! - `--threads t,…` (default 1), `--oracle-threads t` (default the largest of
//!   `--threads`);
//! - `--rule calibrated|s7` (default `calibrated`, Helmholtz CH3.3): with `s7`, a tolerance
//!   gives S7's closed form (`s7_degree`) per level in place of the default rule, through
//!   one run at degree 1 for the levels' circumradii (the times before CH3.3's rule);
//! - `--p2p auto|reference|<isa>,…` (default `auto`): the P2P kernels (Helmholtz CH3.4c),
//!   each case run with each in turn; an ISA this machine lacks is refused before MPI
//!   starts.
//! - `--m2l rotation|dense|classes|batched|batched-fused,…` (default `rotation`, Helmholtz
//!   CH3.2a, CH3.2b, CH3.2c; `batched-fused` the batched M2L in the fused mode): the M2L,
//!   each case run with each in turn (and each P2P kernel);
//!   `--dense-budget MB` the dense, class or batched tables' bytes per rank (default
//!   1024 MiB). The column "M2L" names the levels with tables and their MB. Also (Helmholtz
//!   CH3.2d) `auto`, the rule per level at the rule's crossover, `auto:<d>` at the crossover
//!   d, and `sweep`: Auto at every crossover that chooses differently, from Batched on every
//!   level that fits (crossover ∞) down to Rotation on every level (one below the smallest
//!   p_l), the crossovers after the first from its degrees; the column names the crossover
//!   (`auto ≤ d`). Also (Helmholtz CH6.2h) `auto-pw`, Auto with the exponential M2L from the
//!   rule's smallest degree (`HelmholtzFmmBuilder::auto_plane_wave`), and `auto-pw:<e>` from
//!   degree e, each at each split of `--split`; the column `auto ≤ d, pw ≥ e: …`; `auto`
//!   names the setting off whatever the library's default. Also (Helmholtz CH6.1e) `planewave`, the exponential M2L with each level's
//!   quadrature by the rule, `planewave:<1|2|3>`, that iprec on every level whose κ_l = k w_l
//!   lies in the quadrature's tables, and `planewave-scan`: Rotation, then each such level
//!   ≥ 2 at iprec 1, 2 and 3 alone, the other levels Rotation; the column names the levels
//!   with waves and their precisions (`planewave 3:i2`). Also (Helmholtz CH6.2e)
//!   `planewave-scan:<1|2|3>`, the scan at that precision alone.
//! - `--reach r,…` (Helmholtz CH6.2e): the lateral reach of the quadratures' α counts in box
//!   widths (`corner`, the default, the library's 4√2), each exponential-M2L run (the scan's
//!   included) at each reach in turn; the column "M2L" of a row at another reach ends with
//!   `ρ r`.
//! - `--split auto|parents|children,…` (Helmholtz CH6.2f): how merge and shift splits over
//!   the threads (`auto`, the default, the library's), each exponential-M2L run (the scan's
//!   at the first) at each split in turn; the column "M2L" of a row at another split ends
//!   with `/parents` or `/children`. No bit depends on it.
//! - `--raise level|leaves|m2l,…` (Helmholtz CH6.1h, `PlaneWaveRaise`): which operators of a
//!   level the rule raises below 10⁻⁶ run at the raised degree (`level`, the default, the
//!   library's: every one; `leaves`: the level's translations to and from its parent at the
//!   degree given; `m2l`: those and its leaves' operators), each exponential-M2L run (the
//!   scan's at the first; Auto's with waves too) at each in turn, inside `--split`'s loop;
//!   the column "M2L" of a row ends with `/leaves` or `/m2l` where named.
//! - `--plane-wave-budget MB|whole` (Helmholtz CH6.2a): the exponential M2L's waves buffer
//!   per node (default the library's, `DEFAULT_PLANE_WAVE_NODE_BUDGET`; `whole` a whole
//!   level). The column "M2L" of an exponential-M2L row gives rank 0's waves buffer, its
//!   chunks and level calls and the waves computed (box × frame × vector), and the run ends
//!   with rank 0's peak resident size (Linux).
//!
//! - `--levels` (Helmholtz CH6.2b): after the table, a second one with a row per run and
//!   level with V pairs: the level's degree, the M2L that ran there (`rotation`, `dense`,
//!   `classes`, `batched`, `batched-fused` or `planewave iq`), its V pairs summed over the
//!   ranks, the M2L level calls' seconds (the slowest rank's, `Row::m2l_seconds`) and
//!   nanoseconds per pair, and (Helmholtz CH6.2d) where merge and shift ran its two passes'
//!   seconds, the waves and merge and shift (each the slowest rank's,
//!   `Row::plane_wave_seconds`), and (Helmholtz CH6.2e) the level's waves per direction
//!   Σₖ Nₖ (`Row::plane_wave_waves`; `–` without waves), and (Helmholtz CH6.2f) merge and
//!   shift's thread-seconds, busy, shifts, children and conversions (`Row::plane_wave_parts`),
//!   with the threads it kept busy (busy over merge and shift's seconds); then a row `Σ` with the levels'
//!   seconds summed and the evaluation's (with several evaluations also the first's).
//! - `--evaluations n` (default 1; Helmholtz CH6.2b): n evaluations of each build; the
//!   row's evaluation time, errors, checksum and M2L times are the last one's (the first
//!   touches the buffers, which on a large exponential M2L's waves costs seconds).
//!
//! The last column, "÷ Rotation", is the row's error of φ and of ∇φ over the case's
//! Rotation row at the same P2P kernel and threads, where one ran before it.
//!
//! Run in release mode, on any number of ranks (each rank passes a contiguous block of
//! the points, the oracle is sharded over the ranks, rank 0 prints), with one BLAS
//! thread:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example helmholtz_accuracy -- \
//!     --workload w1 --kd 30,100 --eps 1e-3,1e-6 --threads 1,6
//! mpirun -n 3 target/release/examples/helmholtz_accuracy --workload w1 --kd 100 --threads 2
//! ```
//!
//! MPI is initialised with `Threading::Funneled`. Timings are wall times on this machine,
//! reported and never asserted.

use std::time::Instant;

use mpi::Threading;
use mpi::traits::*;
use nd_fmm_exec::helmholtz::{
    ChargeKind, DEFAULT_DENSE_BUDGET, Degrees, HelmholtzM2l, PlaneWavePrecision, PlaneWaveRaise,
    PlaneWaveSplit, Wavenumber, s7_degree,
};
use nd_fmm_exec::operator::{MulAdd, P2pChoice};
use nd_fmm_math::Complex;
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{RHO_MAX, domain_of};
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{
    Case, Oracle, Problem, Row, complex_k, run, sweep_crossovers,
};
use nd_fmm_validate::helmholtz_workloads::Workload;
use nd_fmm_validate::scaling::resident_memory;

/// The command line.
struct Arguments {
    workload: String,
    kds: Vec<f64>,
    kim: f64,
    degrees: Vec<Degrees>,
    charges: ChargeKind,
    vectors: usize,
    samples: usize,
    gradients: bool,
    threads: Vec<usize>,
    oracle_threads: Option<usize>,
    p2p: Vec<P2pChoice>,
    m2l: Vec<M2lRun>,
    dense_budget: usize,
    plane_wave_budget: Option<usize>,
    reaches: Vec<Option<f64>>,
    splits: Vec<Option<PlaneWaveSplit>>,
    raises: Vec<Option<PlaneWaveRaise>>,
    s7: bool,
    levels: bool,
    evaluations: usize,
}

/// One entry of `--m2l`.
#[derive(Clone, Copy, Debug)]
enum M2lRun {
    /// A setting, its multiply–add mode and crossover (`None`: the library's).
    Fixed(HelmholtzM2l, Option<MulAdd>, Option<usize>),
    /// Auto at every crossover that chooses differently (module documentation).
    Sweep,
    /// The exponential M2L, by the rule (`None`) or one precision on every level.
    PlaneWave(Option<Precision>),
    /// Auto with the exponential M2L from the rule's smallest degree (`None`) or a named one
    /// (Helmholtz CH6.2h).
    AutoPlaneWave(Option<usize>),
    /// Rotation, then each level with a quadrature at each precision alone (`None`) or at
    /// one.
    PlaneWaveScan(Option<Precision>),
}

/// The levels ≥ 2 of a row whose κ_l = k w_l lies in the quadrature's tables.
fn plane_wave_levels(row: &Row, k: Complex<f64>) -> Vec<usize> {
    (2..row.levels)
        .filter(|&l| {
            // The width w_l = 2 r_l, from the circumradius √3 r_l.
            let width = 2.0 * row.circumradii[l] / 3f64.sqrt();
            domain_of(k * width).is_some()
        })
        .collect()
}

/// Parses a comma-separated list.
fn list<T: std::str::FromStr>(value: &str) -> Option<Vec<T>> {
    value.split(',').map(|v| v.parse().ok()).collect()
}

/// The workload of a name at kD (W1 and W3 resolved depend on it).
fn workload(name: &str, kd: f64) -> Option<Workload> {
    let count = |n: &str| n.parse::<usize>().ok().filter(|&n| n > 0);
    match name.split_once(':') {
        None => match name {
            "w1" => Some(Workload::W1),
            "w2" => Some(Workload::W2),
            "w3" => Some(Workload::w3_resolved(kd)),
            _ => None,
        },
        Some(("w3", n)) => Some(Workload::W3 { n: count(n)? }),
        Some((w4, n)) => Some(Workload::W4 {
            distribution: Distribution::from_name(w4.strip_prefix("w4-")?)?,
            n: count(n)?,
        }),
    }
}

/// Parses the options of the module documentation; exits with a message on anything else.
fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: helmholtz_accuracy [--workload w1|w2|w3|w3:<n>|w4-<distribution>:<n>] \
             [--kd x,…] [--kim y] [--eps e,…|--degree p] [--charges real|complex] \
             [--vectors m] [--samples n] [--no-gradients] [--threads t,…] \
             [--oracle-threads t] [--rule calibrated|s7] [--p2p auto|reference|<isa>,…] \
             [--m2l rotation|dense|classes|batched|batched-fused|auto|auto:<d>|auto-pw|\
             auto-pw:<d>|sweep|\
             planewave|planewave:<1|2|3>|planewave-scan|planewave-scan:<1|2|3>,…] \
             [--dense-budget MB] [--plane-wave-budget MB|whole] [--reach corner|r,…] \
             [--split auto|parents|children,…] [--raise level|leaves|m2l,…] [--levels] \
             [--evaluations n]; got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        workload: "w1".into(),
        kds: vec![10.0],
        kim: 0.0,
        degrees: vec![Degrees::Tolerance(1e-3)],
        charges: ChargeKind::Complex,
        vectors: 1,
        samples: 1000,
        gradients: true,
        threads: vec![1],
        oracle_threads: None,
        p2p: vec![P2pChoice::Auto],
        m2l: vec![M2lRun::Fixed(HelmholtzM2l::Rotation, None, None)],
        dense_budget: DEFAULT_DENSE_BUDGET,
        plane_wave_budget: None,
        reaches: vec![None],
        splits: vec![None],
        raises: vec![None],
        s7: false,
        levels: false,
        evaluations: 1,
    };
    let mut rest = args.as_slice();
    loop {
        rest = match rest {
            [] => break,
            [flag, tail @ ..] if flag == "--no-gradients" => {
                parsed.gradients = false;
                tail
            }
            [flag, tail @ ..] if flag == "--levels" => {
                parsed.levels = true;
                tail
            }
            [flag, value, tail @ ..] => {
                let ok = match flag.as_str() {
                    "--workload" => {
                        parsed.workload = value.clone();
                        workload(value, 1.0).is_some()
                    }
                    "--kd" => list(value).map(|v| parsed.kds = v).is_some(),
                    "--kim" => value.parse().map(|v| parsed.kim = v).is_ok(),
                    "--eps" => list::<f64>(value)
                        .map(|v| parsed.degrees = v.into_iter().map(Degrees::Tolerance).collect())
                        .is_some(),
                    "--degree" => value
                        .parse::<usize>()
                        .map(|p| parsed.degrees = vec![Degrees::PerLevel(vec![p; 64])])
                        .is_ok(),
                    "--charges" => match value.as_str() {
                        "real" => Some(ChargeKind::Real),
                        "complex" => Some(ChargeKind::Complex),
                        _ => None,
                    }
                    .map(|c| parsed.charges = c)
                    .is_some(),
                    "--vectors" => value.parse().map(|v| parsed.vectors = v).is_ok(),
                    "--samples" => value.parse().map(|v| parsed.samples = v).is_ok(),
                    "--threads" => list(value).map(|v| parsed.threads = v).is_some(),
                    "--p2p" => list::<P2pChoice>(value)
                        .filter(|c| c.iter().all(|c| c.resolve().is_ok()))
                        .map(|c| parsed.p2p = c)
                        .is_some(),
                    "--m2l" => value
                        .split(',')
                        .map(|v| {
                            let fixed = |m2l, mul_add| Some(M2lRun::Fixed(m2l, mul_add, None));
                            match v {
                                "rotation" => fixed(HelmholtzM2l::Rotation, None),
                                "dense" => fixed(HelmholtzM2l::Dense, None),
                                "classes" => fixed(HelmholtzM2l::Classes, None),
                                "batched" => fixed(HelmholtzM2l::Batched, Some(MulAdd::Separate)),
                                "batched-fused" => {
                                    fixed(HelmholtzM2l::Batched, Some(MulAdd::Fused))
                                }
                                "auto" => fixed(HelmholtzM2l::Auto, None),
                                "auto-pw" => Some(M2lRun::AutoPlaneWave(None)),
                                "sweep" => Some(M2lRun::Sweep),
                                "planewave" => Some(M2lRun::PlaneWave(None)),
                                "planewave-scan" => Some(M2lRun::PlaneWaveScan(None)),
                                "planewave-scan:1" => {
                                    Some(M2lRun::PlaneWaveScan(Some(Precision::Iprec1)))
                                }
                                "planewave-scan:2" => {
                                    Some(M2lRun::PlaneWaveScan(Some(Precision::Iprec2)))
                                }
                                "planewave-scan:3" => {
                                    Some(M2lRun::PlaneWaveScan(Some(Precision::Iprec3)))
                                }
                                "planewave:1" => Some(M2lRun::PlaneWave(Some(Precision::Iprec1))),
                                "planewave:2" => Some(M2lRun::PlaneWave(Some(Precision::Iprec2))),
                                "planewave:3" => Some(M2lRun::PlaneWave(Some(Precision::Iprec3))),
                                _ => {
                                    match v.strip_prefix("auto-pw:") {
                                        Some(d) => {
                                            d.parse().ok().map(|d| M2lRun::AutoPlaneWave(Some(d)))
                                        }
                                        None => v.strip_prefix("auto:")?.parse().ok().map(|d| {
                                            M2lRun::Fixed(HelmholtzM2l::Auto, None, Some(d))
                                        }),
                                    }
                                }
                            }
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|m| parsed.m2l = m)
                        .is_some(),
                    "--dense-budget" => value
                        .parse::<usize>()
                        .ok()
                        .and_then(|mb| mb.checked_mul(1 << 20))
                        .map(|bytes| parsed.dense_budget = bytes)
                        .is_some(),
                    "--plane-wave-budget" => match value.as_str() {
                        "whole" => Some(usize::MAX),
                        mb => mb
                            .parse::<usize>()
                            .ok()
                            .and_then(|mb| mb.checked_mul(1 << 20)),
                    }
                    .map(|bytes| parsed.plane_wave_budget = Some(bytes))
                    .is_some(),
                    "--reach" => value
                        .split(',')
                        .map(|v| match v {
                            "corner" => Some(None),
                            r => r
                                .parse::<f64>()
                                .ok()
                                .filter(|&r| r.is_finite() && r > 0.75)
                                .map(Some),
                        })
                        .collect::<Option<Vec<_>>>()
                        .filter(|r| !r.is_empty())
                        .map(|r| parsed.reaches = r)
                        .is_some(),
                    "--split" => value
                        .split(',')
                        .map(|v| match v {
                            "auto" => Some(None),
                            "parents" => Some(Some(PlaneWaveSplit::Parents)),
                            "children" => Some(Some(PlaneWaveSplit::Children)),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|s| parsed.splits = s)
                        .is_some(),
                    "--raise" => value
                        .split(',')
                        .map(|v| match v {
                            "level" => Some(None),
                            "leaves" => Some(Some(PlaneWaveRaise::Leaves)),
                            "m2l" => Some(Some(PlaneWaveRaise::M2l)),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|r| parsed.raises = r)
                        .is_some(),
                    "--rule" => match value.as_str() {
                        "calibrated" => Some(false),
                        "s7" => Some(true),
                        _ => None,
                    }
                    .map(|s7| parsed.s7 = s7)
                    .is_some(),
                    "--evaluations" => value
                        .parse()
                        .ok()
                        .filter(|&n| n > 0)
                        .map(|n| parsed.evaluations = n)
                        .is_some(),
                    "--oracle-threads" => value
                        .parse()
                        .map(|v| parsed.oracle_threads = Some(v))
                        .is_ok(),
                    _ => false,
                };
                if !ok {
                    usage();
                }
                tail
            }
            _ => usage(),
        };
    }
    if parsed.vectors == 0
        || parsed.threads.contains(&0)
        || parsed.kds.is_empty()
        || parsed.p2p.is_empty()
        || parsed.m2l.is_empty()
    {
        usage();
    }
    parsed
}

/// The M2L of a row: `rotation`, or `dense l…`, `classes l…`, `batched l…` or
/// `batched-fused l…` with the levels that ran the setting's tables and their MB; under
/// Auto prefixed with `auto ≤ d: ` (`∞` for no bound; `auto ≤ d, pw ≥ e: ` with waves,
/// whose levels follow as `; planewave l:iq…` with the waves buffer's MB, chunks, level calls
/// and waves computed); `planewave l:iq…` with each level's precision, its MB, and the same.
fn m2l_string(row: &Row) -> String {
    if row.m2l == HelmholtzM2l::PlaneWave {
        let levels: Vec<String> = (0..row.plane_wave.len())
            .filter_map(|l| Some(format!("{l}:i{}", row.plane_wave[l]?.iprec())))
            .collect();
        return match levels.is_empty() {
            true => "rotation".into(),
            false => format!(
                "planewave {} ({:.0} MB; waves {:.0} MB, {} chunks in {} calls, {} waves){}",
                levels.join(","),
                row.dense_bytes as f64 / 1e6,
                row.plane_wave_buffer as f64 / 1e6,
                row.plane_wave_counts.chunks,
                row.plane_wave_counts.level_calls,
                row.plane_wave_counts.waves,
                reach_suffix(row),
            ),
        };
    }
    let (m2l, mul_add, dense, bytes) = (
        row.m2l,
        row.m2l_mul_add,
        row.dense_levels.as_slice(),
        row.dense_bytes,
    );
    // Under Auto with the exponential M2L its levels are listed apart.
    let levels: Vec<String> = (0..dense.len())
        .filter(|&l| dense[l] && row.plane_wave[l].is_none())
        .map(|l| l.to_string())
        .collect();
    let name = match m2l {
        HelmholtzM2l::Classes => "classes",
        HelmholtzM2l::Batched | HelmholtzM2l::Auto if mul_add == MulAdd::Fused => "batched-fused",
        HelmholtzM2l::Batched | HelmholtzM2l::Auto => "batched",
        _ => "dense",
    };
    let waves: Vec<String> = (0..row.plane_wave.len())
        .filter_map(|l| Some(format!("{l}:i{}", row.plane_wave[l]?.iprec())))
        .collect();
    // With waves the bytes are both methods' and go after them.
    let mut ran = match (levels.is_empty(), waves.is_empty()) {
        (true, _) => "rotation".into(),
        (false, true) => format!("{name} {} ({:.0} MB)", levels.join(","), bytes as f64 / 1e6),
        (false, false) => format!("{name} {}", levels.join(",")),
    };
    if !waves.is_empty() {
        ran = format!(
            "{ran}; planewave {} ({:.0} MB in all; waves {:.0} MB, {} chunks in {} calls, {} \
             waves){}",
            waves.join(","),
            bytes as f64 / 1e6,
            row.plane_wave_buffer as f64 / 1e6,
            row.plane_wave_counts.chunks,
            row.plane_wave_counts.level_calls,
            row.plane_wave_counts.waves,
            reach_suffix(row),
        );
    }
    match auto_name(row) {
        Some(name) => format!("{name}: {ran}"),
        None => ran,
    }
}

/// Auto's name, `auto ≤ d` (`∞` for no bound), with `, pw ≥ e` where it takes the
/// exponential M2L from degree e (Helmholtz CH6.2h); `None` under the other settings.
fn auto_name(row: &Row) -> Option<String> {
    let batched = match row.m2l_max_degree? {
        usize::MAX => "auto ≤ ∞".to_string(),
        d => format!("auto ≤ {d}"),
    };
    Some(match row.m2l_plane_wave_min_degree {
        Some(e) => format!("{batched}, pw ≥ {e}"),
        None => batched,
    })
}

/// For an exponential-M2L row (or Auto's with waves), ` ρ r` where its counts' reach is not
/// the corner's, `/parents` or `/children` where its split is named and `/leaves` or `/m2l`
/// where the raised levels' form is; else empty.
fn reach_suffix(row: &Row) -> String {
    if row.m2l != HelmholtzM2l::PlaneWave && row.m2l_plane_wave_min_degree.is_none() {
        return String::new();
    }
    let reach = match row.plane_wave_reach != RHO_MAX {
        true => format!(" ρ {}", row.plane_wave_reach),
        false => String::new(),
    };
    let split = match row.plane_wave_split {
        PlaneWaveSplit::Auto => "",
        PlaneWaveSplit::Parents => "/parents",
        PlaneWaveSplit::Children => "/children",
    };
    let raise = match row.plane_wave_raise {
        PlaneWaveRaise::Level => "",
        PlaneWaveRaise::Leaves => "/leaves",
        PlaneWaveRaise::M2l => "/m2l",
    };
    format!("{reach}{split}{raise}")
}

/// The setting of a row in one word: `rotation`, `dense`, `classes`, `batched`,
/// `batched-fused`, `auto ≤ d` (`auto ≤ d, pw ≥ e` with waves) or `planewave` (with ` ρ r`
/// at another reach).
fn setting_name(row: &Row) -> String {
    match row.m2l {
        HelmholtzM2l::Rotation => "rotation".into(),
        HelmholtzM2l::Dense => "dense".into(),
        HelmholtzM2l::Classes => "classes".into(),
        HelmholtzM2l::Batched if row.m2l_mul_add == MulAdd::Fused => "batched-fused".into(),
        HelmholtzM2l::Batched => "batched".into(),
        HelmholtzM2l::Auto => match auto_name(row) {
            Some(name) => format!("{name}{}", reach_suffix(row)),
            None => "auto ≤ ∞".into(),
        },
        HelmholtzM2l::PlaneWave => format!("planewave{}", reach_suffix(row)),
    }
}

/// The M2L that ran on `level` of a row: `planewave iq`, the setting's tables (`dense`,
/// `classes`, `batched`, `batched-fused`; Auto's are batched) or `rotation`.
fn level_m2l(row: &Row, level: usize) -> String {
    if let Some(q) = row.plane_wave[level] {
        return format!("planewave i{}", q.iprec());
    }
    if !row.dense_levels[level] {
        return "rotation".into();
    }
    match row.m2l {
        HelmholtzM2l::Classes => "classes".into(),
        HelmholtzM2l::Batched | HelmholtzM2l::Auto if row.m2l_mul_add == MulAdd::Fused => {
            "batched-fused".into()
        }
        HelmholtzM2l::Batched | HelmholtzM2l::Auto => "batched".into(),
        _ => "dense".into(),
    }
}

/// The degrees as a short string: runs of equal degrees as `p×count`.
fn degrees_string(degrees: &[usize]) -> String {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &p in degrees {
        match runs.last_mut() {
            Some((q, count)) if *q == p => *count += 1,
            _ => runs.push((p, 1)),
        }
    }
    runs.iter()
        .map(|&(p, count)| {
            if count == 1 {
                p.to_string()
            } else {
                format!("{p}×{count}")
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn main() {
    let arguments = arguments();
    let (universe, _) =
        mpi::initialize_with_threading(Threading::Funneled).expect("MPI initialises");
    let world = universe.world();
    let root = world.rank() == 0;
    let oracle_threads = arguments
        .oracle_threads
        .unwrap_or_else(|| *arguments.threads.iter().max().unwrap_or(&1));
    if root {
        println!("# Helmholtz FMM against the sampled direct sum");
        println!();
        println!(
            "Machine: {}, {}, {}; {} ranks; BLAS threads: OPENBLAS_NUM_THREADS={}",
            cpu_model(),
            cores(),
            target(),
            world.size(),
            std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        );
        println!();
        println!(
            "| Workload | N | kD | degrees | charges | m | P2P | M2L | ranks × threads | levels \
             | leaf level | p by level | leaves | max leaf | build s | evaluate s | φ | ∇φ \
             | checksum | ÷ Rotation |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- | ---: | ---: \
             | ---: | ---: | ---: | ---: | --- | --- |"
        );
    }
    let mut levels: Vec<String> = Vec::new();
    for &kd in &arguments.kds {
        let workload = workload(&arguments.workload, kd).expect("checked");
        for degrees in &arguments.degrees {
            let mut case = Case::new(workload, (kd, arguments.kim), degrees.clone());
            case.charges = arguments.charges;
            case.vectors = arguments.vectors;
            case.samples = arguments.samples;
            case.gradients = arguments.gradients;
            case.dense_budget = arguments.dense_budget;
            case.plane_wave_budget = arguments.plane_wave_budget;
            case.evaluations = arguments.evaluations;
            let problem = Problem::new(&case);
            let start = Instant::now();
            let oracle = Oracle::sharded(
                complex_k(&case),
                &problem,
                case.gradients,
                oracle_threads,
                &world,
            );
            let oracle_seconds = start.elapsed().as_secs_f64();
            let kd_string = if arguments.kim == 0.0 {
                format!("{kd}")
            } else {
                format!("{kd} + {}i", arguments.kim)
            };
            let degrees_name = match degrees {
                Degrees::Tolerance(eps) if arguments.s7 => format!("S7 ε = {eps:e}"),
                Degrees::Tolerance(eps) => format!("ε = {eps:e}"),
                Degrees::PerLevel(p) => format!("p = {}", p[0]),
            };
            if let (true, &Degrees::Tolerance(eps)) = (arguments.s7, degrees) {
                case.degrees = Degrees::PerLevel(vec![1; 64]);
                let probe = run(&case, &problem, &oracle, oracle_threads, &world)
                    .expect("the FMM at degree 1 runs");
                let k = Wavenumber::new(kd, arguments.kim).expect("a valid wavenumber");
                case.degrees = Degrees::PerLevel(
                    (0..64)
                        .map(|l| match l {
                            2.. if l < probe.levels => s7_degree(&k, probe.circumradii[l], eps),
                            _ => 0,
                        })
                        .collect(),
                );
            }
            let charges = match case.charges {
                ChargeKind::Real => "real",
                ChargeKind::Complex => "complex",
            };
            let print = |row: &Row, rotation: Option<[f64; 2]>| {
                let all = row.errors.last().expect("an entry over all vectors");
                let ratio = match rotation {
                    Some(r) if case.gradients => {
                        format!("{:.3}/{:.3}", all[0] / r[0], all[1] / r[1])
                    }
                    Some(r) => format!("{:.3}", all[0] / r[0]),
                    None => "–".into(),
                };
                let gradient = if case.gradients {
                    format!("{:.2e}", all[1])
                } else {
                    "–".into()
                };
                println!(
                    "| {} | {} | {kd_string} | {degrees_name} | {charges} | {} | {} | {} | {} × {} \
                     | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2e} | {gradient} \
                     | {:016x} | {ratio} |",
                    workload.name(),
                    problem.len(),
                    case.vectors,
                    row.p2p,
                    m2l_string(row),
                    row.ranks,
                    row.threads,
                    row.levels,
                    row.min_leaf_level,
                    degrees_string(&row.degrees),
                    row.leaves,
                    row.max_leaf_points,
                    row.build,
                    row.evaluate,
                    all[0],
                    row.checksum,
                );
                if case.vectors > 1 {
                    let per: Vec<String> = row.errors[..case.vectors]
                        .iter()
                        .map(|e| format!("{:.2e}/{:.2e}", e[0], e[1]))
                        .collect();
                    println!("|  | per vector φ/∇φ: {} |", per.join(", "));
                }
            };
            // `--levels`: the rows of the second table, a row per level with V pairs and Σ.
            let per_level = |row: &Row| -> Vec<String> {
                let head = format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {} | {} × {}",
                    workload.name(),
                    problem.len(),
                    setting_name(row),
                    row.ranks,
                    row.threads,
                );
                let mut lines: Vec<String> = (0..row.levels)
                    .filter(|&l| row.v_pairs[l] > 0)
                    .map(|l| {
                        let [waves, parents] = row.plane_wave_seconds[l];
                        let passes = if waves + parents > 0.0 {
                            format!("{waves:.6} | {parents:.6}")
                        } else {
                            "– | –".into()
                        };
                        let waves = match row.plane_wave_waves[l] {
                            0 => "–".to_string(),
                            w => w.to_string(),
                        };
                        let [busy, shifts, children, conversions] = row.plane_wave_parts[l];
                        let parts = if busy > 0.0 {
                            format!(
                                "{busy:.6} | {shifts:.6} | {children:.6} | {conversions:.6} \
                                 | {:.1}",
                                busy / parents
                            )
                        } else {
                            "– | – | – | – | –".into()
                        };
                        format!(
                            "{head} | {l} | {} | {} | {} | {:.6} | {:.0} | {passes} | {waves} \
                             | {parts} |",
                            row.degrees[l],
                            level_m2l(row, l),
                            row.v_pairs[l],
                            row.m2l_seconds[l],
                            1e9 * row.m2l_seconds[l] / row.v_pairs[l] as f64,
                        )
                    })
                    .collect();
                let sum = |j: usize| row.plane_wave_parts.iter().map(|p| p[j]).sum::<f64>();
                let evaluate = match arguments.evaluations {
                    1 => format!("evaluate {:.4} s", row.evaluate),
                    n => format!(
                        "evaluate {:.4} s (of {n}; first {:.4} s)",
                        row.evaluate, row.first_evaluate
                    ),
                };
                lines.push(format!(
                    "{head} | Σ | – | {evaluate} | {} | {:.6} | – | {:.6} | {:.6} | {} | {:.6} \
                     | {:.6} | {:.6} | {:.6} | – |",
                    row.v_pairs.iter().sum::<usize>(),
                    row.m2l_seconds.iter().sum::<f64>(),
                    row.plane_wave_seconds.iter().map(|p| p[0]).sum::<f64>(),
                    row.plane_wave_seconds.iter().map(|p| p[1]).sum::<f64>(),
                    row.plane_wave_waves.iter().sum::<usize>(),
                    sum(0),
                    sum(1),
                    sum(2),
                    sum(3),
                ));
                lines
            };
            // The errors of the case's Rotation rows by P2P kernel and threads.
            let mut rotation: Vec<((P2pChoice, usize), [f64; 2])> = Vec::new();
            let k = complex_k(&case);
            for &m2l_run in &arguments.m2l {
                for &p2p in &arguments.p2p {
                    for &threads in &arguments.threads {
                        case.p2p = p2p;
                        // The runs of the entry: (setting, mode, crossover, precisions,
                        // reach), one (the exponential M2L's at each reach), or the sweep's
                        // (the first ∞, the others from its degrees), or the scan's
                        // (Rotation first, the levels from its row, each at each reach).
                        let fixed = |m2l, mul_add, max_degree| {
                            (
                                m2l,
                                mul_add,
                                max_degree,
                                PlaneWavePrecision::Rule,
                                None,
                                None,
                                None,
                                None,
                            )
                        };
                        let raises = &arguments.raises;
                        let mut runs = match m2l_run {
                            M2lRun::Fixed(m2l, mul_add, max_degree) => {
                                vec![fixed(m2l, mul_add, max_degree)]
                            }
                            M2lRun::Sweep => {
                                vec![fixed(HelmholtzM2l::Auto, None, Some(usize::MAX))]
                            }
                            M2lRun::PlaneWave(precision) => arguments
                                .reaches
                                .iter()
                                .flat_map(|&reach| {
                                    arguments.splits.iter().flat_map(move |&split| {
                                        raises.iter().map(move |&raise| (reach, split, raise))
                                    })
                                })
                                .map(|(reach, split, raise)| {
                                    (
                                        HelmholtzM2l::PlaneWave,
                                        None,
                                        None,
                                        match precision {
                                            None => PlaneWavePrecision::Rule,
                                            Some(q) => {
                                                PlaneWavePrecision::PerLevel(vec![Some(q); 64])
                                            }
                                        },
                                        reach,
                                        split,
                                        None,
                                        raise,
                                    )
                                })
                                .collect(),
                            M2lRun::AutoPlaneWave(min_degree) => arguments
                                .splits
                                .iter()
                                .flat_map(|&split| raises.iter().map(move |&raise| (split, raise)))
                                .map(|(split, raise)| {
                                    (
                                        HelmholtzM2l::Auto,
                                        None,
                                        None,
                                        PlaneWavePrecision::Rule,
                                        None,
                                        split,
                                        Some(min_degree),
                                        raise,
                                    )
                                })
                                .collect(),
                            M2lRun::PlaneWaveScan(_) => {
                                vec![fixed(HelmholtzM2l::Rotation, None, None)]
                            }
                        };
                        let mut next = 0;
                        while next < runs.len() {
                            let auto_plane_wave;
                            (
                                case.m2l,
                                case.m2l_mul_add,
                                case.m2l_max_degree,
                                case.plane_wave,
                                case.plane_wave_reach,
                                case.plane_wave_split,
                                auto_plane_wave,
                                case.plane_wave_raise,
                            ) = runs[next].clone();
                            case.auto_plane_wave = auto_plane_wave.is_some();
                            case.m2l_plane_wave_min_degree = auto_plane_wave.flatten();
                            next += 1;
                            match run(&case, &problem, &oracle, threads, &world) {
                                Ok(row) => {
                                    let all =
                                        *row.errors.last().expect("an entry over all vectors");
                                    let reference = rotation
                                        .iter()
                                        .find(|(key, _)| *key == (p2p, threads))
                                        .map(|&(_, errors)| errors);
                                    let is_rotation = case.m2l == HelmholtzM2l::Rotation;
                                    if root {
                                        print(&row, reference.filter(|_| !is_rotation));
                                        if arguments.levels {
                                            levels.extend(per_level(&row));
                                        }
                                    }
                                    if is_rotation && reference.is_none() {
                                        rotation.push(((p2p, threads), all));
                                    }
                                    if next == 1 {
                                        match m2l_run {
                                            M2lRun::Sweep => runs.extend(
                                                sweep_crossovers(&row.degrees)[1..].iter().map(
                                                    |&d| fixed(HelmholtzM2l::Auto, None, Some(d)),
                                                ),
                                            ),
                                            M2lRun::PlaneWaveScan(precision) => {
                                                let precisions = match precision {
                                                    Some(q) => vec![q],
                                                    None => Precision::ALL.to_vec(),
                                                };
                                                for l in plane_wave_levels(&row, k) {
                                                    for &q in &precisions {
                                                        for &reach in &arguments.reaches {
                                                            let mut named = vec![None; row.levels];
                                                            named[l] = Some(q);
                                                            runs.push((
                                                                HelmholtzM2l::PlaneWave,
                                                                None,
                                                                None,
                                                                PlaneWavePrecision::PerLevel(named),
                                                                reach,
                                                                arguments.splits[0],
                                                                None,
                                                                arguments.raises[0],
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                                Err(error) => {
                                    if root {
                                        eprintln!(
                                            "{} at kD = {kd_string}: {error}",
                                            workload.name()
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if root {
                eprintln!(
                    "{} kD = {kd_string} {degrees_name}: oracle {oracle_seconds:.1} s",
                    workload.name()
                );
            }
        }
    }
    if root && arguments.levels {
        println!();
        println!("## M2L per level");
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | level | p | ran | V pairs \
             | M2L s | ns per pair | waves s | merge and shift s | waves | busy s | shifts s \
             | children s | conversions s | threads busy |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for line in &levels {
            println!("{line}");
        }
    }
    if root && let Some((bytes, how)) = resident_memory() {
        println!();
        println!("Rank 0: {:.0} MB {how}.", bytes / 1e6);
    }
}
