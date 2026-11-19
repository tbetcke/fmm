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
//! - `--raise rule|level|leaves|m2l,…` (Helmholtz CH6.1h, `PlaneWaveRaise`): which operators
//!   of a level the rule raises below 10⁻⁶ run at the raised degree (`rule`, the default, the
//!   library's: `m2l` at 10⁻⁷ ≤ ε < 10⁻⁶, `leaves` below; `level`: every one; `leaves`: all
//!   but the level's translations to and from its parent; `m2l`: the M2L alone), each
//!   exponential-M2L run (the scan's at the first; Auto's with waves too) at each in turn,
//!   inside `--split`'s loop; the column "M2L" of a row ends with `/level`, `/leaves` or
//!   `/m2l` where named.
//! - `--empty-boxes on|off,…` (Helmholtz CH4.0d1, `HelmholtzFmmBuilder::empty_boxes`): whether
//!   the plan leaves out the work of the boxes whose subtree holds no target or no source
//!   point (`on`, the default, the library's), every run at each in turn (the outermost of
//!   the run loops); the column "M2L" of a row with the rule off ends with `/full`. A last
//!   table has a row per run with what the rule left out (`Row::left_out`, summed over the
//!   ranks) and the multipoles the ranks read from other ranks (`Row::remote_multipoles`).
//! - `--near-v off|cost|n|inf,…` (Helmholtz CH4.0d2, `HelmholtzFmmBuilder::near_v`): the V
//!   pairs between leaves with n_s n_t below n become near pairs (`off`, the default, the
//!   library's; `cost` the cost model's crossover per level; `inf` every V pair between
//!   leaves), every run at each in turn, inside
//!   `--empty-boxes`' loop; the column "M2L" of a row with the rule on ends with `/v<n>`. A
//!   last table "Near pairs" has a row per run with the V pairs offered and converted
//!   (`Row::v_near`), the V pairs that ran, the multipoles read from other ranks, the source
//!   points sent and received (`Row::source_points`), the P2P's seconds and near
//!   interactions with its thread-nanoseconds per interaction, the M2L's seconds and the
//!   evaluation; with `--levels` also a row per level with P2P.
//! - `--leaf-size auto|n,…` and `--c-leaf auto|c,…` (Helmholtz CH4.0d3,
//!   `HelmholtzFmmBuilder::{max_points_per_leaf, c_leaf}`): the octree's points per leaf and
//!   the maximum leaf width in wavelengths (`auto`, the default, the library's), every run at
//!   each pair in turn (outside `--empty-boxes`' loop: each a new tree); where one is named
//!   the column "M2L" of a row ends with `/n<size>` or `/c<c>`. A last table "Leaves" has a
//!   row per run with the leaf size and c_leaf as built, the leaves and those holding a
//!   point (also per level, `occupied/leaves`), the points per occupied leaf, the largest
//!   degree, the P2P's seconds and share of the evaluation, the M2L's seconds, the P2L's,
//!   the M2P's and the other kinds' seconds (`Row::kind_seconds`), the X and W entries
//!   (`Row::far_leaf_entries`), the bound
//!   on the P2P's phase Re(k) r (`Row::phase_bound`, CH3.4b's contract range 22 c_leaf
//!   without V pairs made near) and over c_leaf, the build, the evaluation and the errors.
//! - `--near-wx off|cost|n|inf,…` (Helmholtz CH4.0d3, `HelmholtzFmmBuilder::near_wx`): the W
//!   pairs whose finer leaf holds fewer than n source points and the X pairs whose finer leaf
//!   holds fewer than n target points become near pairs (`off`, the default, the library's;
//!   `cost` the cost model's crossovers per level; `inf` every one), every run at each in
//!   turn, inside `--near-v`'s loop; the column "M2L" of a row with the rule on ends with
//!   `/wx<n>`. A last table "Leaf operators" has a row per run with the P2P's, P2L's and
//!   M2P's seconds, point interactions (`Row::far_interactions`) and thread-nanoseconds per
//!   interaction, the W and X pairs offered and converted (`Row::wx_near`) and the
//!   evaluation; with `--levels` also a row per level with that work and its degree.
//! - `--term on|off,…` (Helmholtz CH4.0, `HelmholtzFmmBuilder::batched_term`): whether Auto
//!   reads the levels' V lists (`on`, the default, the library's), each Auto run (`auto`,
//!   `auto:<d>`, `sweep`, `auto-pw`) at each in turn; the column "M2L" of a row with the term
//!   off ends with `/no-term`.
//! - `--block auto|w,…` (Helmholtz CH4.0, `HelmholtzFmmBuilder::m2l_block_targets`): the
//!   width of Batched's blocks (`auto`, the default, the library's split by the threads),
//!   each Batched and Auto run at each in turn, inside `--term`'s loop; the column "M2L" of a
//!   row at a named width ends with `/wN`. No bit depends on it.
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
//!   with the threads it kept busy (busy over merge and shift's seconds), and (Helmholtz
//!   CH4.0) the level's boxes (`Row::v_targets`), m̂, the estimate of Batched's columns per
//!   table read that Auto's term reads (`Row::batched_columns`), Batched's (block, offset)
//!   visits at blocks of 64 summed over the ranks (`Row::block_visits`) and the V pairs over
//!   them, the exact columns per read on one rank, and (Helmholtz CH4.0e) whether the
//!   level's rotation blocks are stored or formed per V pair (`Row::m2l_blocks_stored`);
//!   then a row `Σ` with the levels' seconds summed and the evaluation's (with several
//!   evaluations also the first's).
//!   Then (Helmholtz CH4.1b) a table "Diagonal-form scratch", a row per run and level with V
//!   pairs: the diagonal form's scratch as if every V pair of the level ran it
//!   (`Row::signature_scratch`, `nd_fmm_tables::helmholtz::diagonal::scratch_shape` at the
//!   level's degree and the run's threads under 8 GiB per node over the ranks per node): the
//!   most targets with a V row and V sources on one rank, the budget, the most MiB of a
//!   rank's source signatures and whole scratch, the fewest targets per chunk, the most
//!   chunks, and the ranks over the budget.
//! - `--block-budget MB` (Helmholtz CH4.0e): the bytes of the M2L's stored rotation blocks
//!   per rank (default the library's, 512 MiB); the levels that do not fit, coarsest first,
//!   form their blocks per V pair. No bit depends on it.
//! - `--rcr-split auto|boxes|pairs,…` (Helmholtz CH4.0i, `HelmholtzFmmBuilder::rcr_split`):
//!   how the Rotation M2L's level calls split over the threads (`auto`, the default, the
//!   library's), the outermost loop of the runs; the column "M2L" of a row with a named split
//!   ends with `/rcr-boxes` or `/rcr-pairs`, and `--levels`' column "ran" says `rotation by
//!   pairs` where a level call ran by pairs. No bit depends on it.
//! - `--evaluations n` (default 1; Helmholtz CH6.2b): n evaluations of each build; the
//!   row's evaluation time, errors, checksum and M2L times are the last one's (the first
//!   touches the buffers, which on a large exponential M2L's waves costs seconds).
//!
//! The last column, "÷ Rotation", is the row's error of φ and of ∇φ over the case's
//! Rotation row at the same P2P kernel and threads, where one ran before it.
//!
//! A last table "Build" (Helmholtz CH4.0h) has a row per run with the build's seconds and
//! its parts (`Row::build_parts`, in the order of `BUILD_PARTS`, each the slowest rank's;
//! the operator's tables by part within "operator"), the evaluation and the checksum.
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
    PlaneWaveSplit, RcrSplit, VNear, Wavenumber, WxNear, s7_degree,
};
use nd_fmm_exec::operator::{MulAdd, P2pChoice};
use nd_fmm_math::Complex;
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{RHO_MAX, domain_of};
use nd_fmm_validate::bench::{cores, cpu_model, target};
use nd_fmm_validate::fmm_accuracy::Distribution;
use nd_fmm_validate::helmholtz_accuracy::{
    BUILD_PARTS, Case, Oracle, Problem, Row, complex_k, run, sweep_crossovers,
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
    block_budget: Option<usize>,
    reaches: Vec<Option<f64>>,
    splits: Vec<Option<PlaneWaveSplit>>,
    rcr_splits: Vec<Option<RcrSplit>>,
    raises: Vec<Option<PlaneWaveRaise>>,
    terms: Vec<bool>,
    blocks: Vec<Option<usize>>,
    empty_boxes: Vec<bool>,
    near_v: Vec<VNear>,
    near_wx: Vec<WxNear>,
    leaf_sizes: Vec<Option<usize>>,
    c_leaves: Vec<Option<f64>>,
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
             [--dense-budget MB] [--plane-wave-budget MB|whole] [--block-budget MB] \
             [--reach corner|r,…] \
             [--split auto|parents|children,…] [--raise rule|level|leaves|m2l,…] \
             [--rcr-split auto|boxes|pairs,…] \
             [--term on|off,…] [--block auto|w,…] [--empty-boxes on|off,…] [--near-v off|cost|n|inf,…] [--near-wx off|cost|n|inf,…] \
             [--leaf-size auto|n,…] [--c-leaf auto|c,…] [--levels] \
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
        block_budget: None,
        reaches: vec![None],
        splits: vec![None],
        rcr_splits: vec![None],
        raises: vec![None],
        terms: vec![true],
        blocks: vec![None],
        empty_boxes: vec![true],
        near_v: vec![VNear::Off],
        near_wx: vec![WxNear::Off],
        leaf_sizes: vec![None],
        c_leaves: vec![None],
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
                    "--block-budget" => value
                        .parse::<usize>()
                        .ok()
                        .and_then(|mb| mb.checked_mul(1 << 20))
                        .map(|bytes| parsed.block_budget = Some(bytes))
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
                    "--rcr-split" => value
                        .split(',')
                        .map(|v| match v {
                            "auto" => Some(None),
                            "boxes" => Some(Some(RcrSplit::Boxes)),
                            "pairs" => Some(Some(RcrSplit::Pairs)),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|s| parsed.rcr_splits = s)
                        .is_some(),
                    "--raise" => value
                        .split(',')
                        .map(|v| match v {
                            "rule" => Some(None),
                            "level" => Some(Some(PlaneWaveRaise::Level)),
                            "leaves" => Some(Some(PlaneWaveRaise::Leaves)),
                            "m2l" => Some(Some(PlaneWaveRaise::M2l)),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|r| parsed.raises = r)
                        .is_some(),
                    "--term" => value
                        .split(',')
                        .map(|v| match v {
                            "on" => Some(true),
                            "off" => Some(false),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|t| parsed.terms = t)
                        .is_some(),
                    "--empty-boxes" => value
                        .split(',')
                        .map(|v| match v {
                            "on" => Some(true),
                            "off" => Some(false),
                            _ => None,
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|e| parsed.empty_boxes = e)
                        .is_some(),
                    "--leaf-size" => value
                        .split(',')
                        .map(|v| match v {
                            "auto" => Some(None),
                            n => n.parse::<usize>().ok().filter(|&n| n > 0).map(Some),
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|n| parsed.leaf_sizes = n)
                        .is_some(),
                    "--c-leaf" => value
                        .split(',')
                        .map(|v| match v {
                            "auto" => Some(None),
                            c => c
                                .parse::<f64>()
                                .ok()
                                .filter(|c| *c > 0.0 && c.is_finite())
                                .map(Some),
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|c| parsed.c_leaves = c)
                        .is_some(),
                    "--near-wx" => value
                        .split(',')
                        .map(|v| match v {
                            "off" => Some(WxNear::Off),
                            "cost" => Some(WxNear::CostModel),
                            "inf" => Some(WxNear::Crossovers {
                                w: f64::INFINITY,
                                x: f64::INFINITY,
                            }),
                            n => n
                                .parse::<f64>()
                                .ok()
                                .filter(|n| *n >= 0.0)
                                .map(|n| WxNear::Crossovers { w: n, x: n }),
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|n| parsed.near_wx = n)
                        .is_some(),
                    "--near-v" => value
                        .split(',')
                        .map(|v| match v {
                            "off" => Some(VNear::Off),
                            "cost" => Some(VNear::CostModel),
                            "inf" => Some(VNear::Crossover(f64::INFINITY)),
                            n => n
                                .parse::<f64>()
                                .ok()
                                .filter(|n| *n >= 0.0)
                                .map(VNear::Crossover),
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|n| parsed.near_v = n)
                        .is_some(),
                    "--block" => value
                        .split(',')
                        .map(|v| match v {
                            "auto" => Some(None),
                            w => w.parse::<usize>().ok().filter(|&w| w > 0).map(Some),
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|b| parsed.blocks = b)
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
        Some(name) => format!("{name}{}: {ran}", batched_suffix(row)),
        None => format!("{ran}{}", batched_suffix(row)),
    }
}

/// For a row with the empty-box rule off `/full` (Helmholtz CH4.0d1), with V pairs as
/// near pairs `/v<n>` (Helmholtz CH4.0d2) and with W and X pairs as near pairs `/wx<n>`
/// (Helmholtz CH4.0d3); else empty.
fn empty_suffix(row: &Row) -> String {
    let full = if row.empty_boxes { "" } else { "/full" };
    let v = match row.near_v {
        VNear::Off => String::new(),
        near => format!("/v{near}"),
    };
    let wx = match row.near_wx {
        WxNear::Off => String::new(),
        WxNear::Crossovers { w, .. } => format!("/wx{w}"),
        WxNear::CostModel => "/wxcost".to_string(),
    };
    format!("{full}{v}{wx}")
}

/// For a row of a named leaf size `/n<size>` and of a named c_leaf `/c<c>` (Helmholtz
/// CH4.0d3; `named` says which the command line named); else empty.
fn leaf_suffix(row: &Row, named: (bool, bool)) -> String {
    let size = match named.0 {
        true => format!("/n{}", row.max_points_per_leaf),
        false => String::new(),
    };
    let width = match named.1 {
        true => format!("/c{}", row.c_leaf),
        false => String::new(),
    };
    format!("{size}{width}")
}

/// For an Auto row without the term `/no-term`, and for a Batched or Auto row at a named
/// block width `/wN` (Helmholtz CH4.0); else empty.
fn batched_suffix(row: &Row) -> String {
    let term = match (row.m2l, row.batched_term) {
        (HelmholtzM2l::Auto, false) => "/no-term",
        _ => "",
    };
    let width = match (row.m2l, row.m2l_block_targets) {
        (HelmholtzM2l::Auto | HelmholtzM2l::Batched, Some(w)) => format!("/w{w}"),
        _ => String::new(),
    };
    format!("{term}{width}")
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
/// the corner's, `/parents` or `/children` where its split is named and `/level`, `/leaves`
/// or `/m2l` where the raised levels' form is; else empty.
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
        PlaneWaveRaise::Rule => "",
        PlaneWaveRaise::Level => "/level",
        PlaneWaveRaise::Leaves => "/leaves",
        PlaneWaveRaise::M2l => "/m2l",
    };
    format!("{reach}{split}{raise}")
}

/// The setting of a row in one word: `rotation`, `dense`, `classes`, `batched`,
/// `batched-fused`, `auto ≤ d` (`auto ≤ d, pw ≥ e` with waves) or `planewave` (with ` ρ r`
/// at another reach), ending with `/rcr-boxes` or `/rcr-pairs` where the Rotation M2L's
/// split is named (Helmholtz CH4.0i).
fn setting_name(row: &Row) -> String {
    let split = match row.rcr_split {
        RcrSplit::Auto => "",
        RcrSplit::Boxes => "/rcr-boxes",
        RcrSplit::Pairs => "/rcr-pairs",
    };
    format!("{}{split}", m2l_setting_name(row))
}

/// [`setting_name`] without the Rotation M2L's split.
fn m2l_setting_name(row: &Row) -> String {
    match row.m2l {
        HelmholtzM2l::Rotation => "rotation".into(),
        HelmholtzM2l::Dense => "dense".into(),
        HelmholtzM2l::Classes => "classes".into(),
        HelmholtzM2l::Batched if row.m2l_mul_add == MulAdd::Fused => {
            format!("batched-fused{}", batched_suffix(row))
        }
        HelmholtzM2l::Batched => format!("batched{}", batched_suffix(row)),
        HelmholtzM2l::Auto => match auto_name(row) {
            Some(name) => format!("{name}{}{}", reach_suffix(row), batched_suffix(row)),
            None => "auto ≤ ∞".into(),
        },
        HelmholtzM2l::PlaneWave => format!("planewave{}", reach_suffix(row)),
    }
}

/// The M2L that ran on `level` of a row: `planewave iq`, the setting's tables (`dense`,
/// `classes`, `batched`, `batched-fused`; Auto's are batched), `rotation`, or `rotation by
/// pairs` where a level call ran by pairs (Helmholtz CH4.0i).
fn level_m2l(row: &Row, level: usize) -> String {
    if let Some(q) = row.plane_wave[level] {
        return format!("planewave i{}", q.iprec());
    }
    if !row.dense_levels[level] {
        return match row.rcr_by_pairs[level] {
            true => "rotation by pairs".into(),
            false => "rotation".into(),
        };
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
    let mut scratch_rows: Vec<String> = Vec::new();
    let mut left_out: Vec<String> = Vec::new();
    let mut near: Vec<String> = Vec::new();
    let mut leaf_rows: Vec<String> = Vec::new();
    let mut build_rows: Vec<String> = Vec::new();
    let mut leaf_operators: Vec<String> = Vec::new();
    // Whether `--leaf-size` and `--c-leaf` named a value (Helmholtz CH4.0d3): the column
    // "M2L" then ends with it.
    let named = (
        arguments.leaf_sizes.iter().any(Option::is_some),
        arguments.c_leaves.iter().any(Option::is_some),
    );
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
            case.block_budget = arguments.block_budget;
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
                let m2l = format!(
                    "{}{}{}",
                    m2l_string(row),
                    empty_suffix(row),
                    leaf_suffix(row, named)
                );
                println!(
                    "| {} | {} | {kd_string} | {degrees_name} | {charges} | {} | {} | {} | {} × {} \
                     | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2e} | {gradient} \
                     | {:016x} | {ratio} |",
                    workload.name(),
                    problem.len(),
                    case.vectors,
                    row.p2p,
                    m2l,
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
                let setting = format!(
                    "{}{}{}",
                    setting_name(row),
                    empty_suffix(row),
                    leaf_suffix(row, named)
                );
                let head = format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {} | {} × {}",
                    workload.name(),
                    problem.len(),
                    setting,
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
                             | {parts} | {} | {:.2} | {} | {:.2} | {} |",
                            row.degrees[l],
                            level_m2l(row, l),
                            row.v_pairs[l],
                            row.m2l_seconds[l],
                            1e9 * row.m2l_seconds[l] / row.v_pairs[l] as f64,
                            row.v_targets[l],
                            row.batched_columns[l],
                            row.block_visits[l],
                            row.v_pairs[l] as f64 / row.block_visits[l].max(1) as f64,
                            if row.m2l_blocks_stored[l] {
                                "stored"
                            } else {
                                "per pair"
                            },
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
                     | {:.6} | {:.6} | {:.6} | – | – | – | – | – | – |",
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
            // `--levels`: the rows of the table "Diagonal-form scratch" (Helmholtz CH4.1b), a
            // row per level with V pairs.
            let scratch_lines = |row: &Row| -> Vec<String> {
                let mb = |bytes: usize| bytes as f64 / (1u64 << 20) as f64;
                (0..row.levels)
                    .filter(|&l| row.v_pairs[l] > 0)
                    .map(|l| {
                        let s = row.signature_scratch[l];
                        format!(
                            "| {} | {} | {kd_string} | {degrees_name} | {} | {} × {} | {l} | {} \
                             | {} | {} | {:.0} | {:.0} | {:.0} | {} | {} | {} |",
                            workload.name(),
                            problem.len(),
                            setting_name(row),
                            row.ranks,
                            row.threads,
                            row.degrees[l],
                            s.targets,
                            s.sources,
                            mb(s.budget),
                            mb(s.source_bytes),
                            mb(s.bytes),
                            s.chunk,
                            s.chunks,
                            s.over_budget,
                        )
                    })
                    .collect()
            };
            // The row of the table "Empty boxes" (Helmholtz CH4.0d1).
            let left_out_line = |row: &Row| -> String {
                let [boxes, v, x, w, l2l, v_sources, w_sources, m2m] = row.left_out;
                format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {} | {} × {} | {} | {boxes} | {v} \
                     | {x} | {w} | {l2l} | {v_sources} | {w_sources} | {m2m} | {} | {} |",
                    workload.name(),
                    problem.len(),
                    setting_name(row),
                    row.ranks,
                    row.threads,
                    if row.empty_boxes { "on" } else { "off" },
                    row.v_pairs.iter().sum::<usize>(),
                    row.remote_multipoles,
                )
            };
            // The rows of the table "Near pairs" (Helmholtz CH4.0d2): the run's, and with
            // `--levels` one per level with P2P.
            let near_lines = |row: &Row| -> Vec<String> {
                let head = format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {}{}{} | {} × {}",
                    workload.name(),
                    problem.len(),
                    setting_name(row),
                    empty_suffix(row),
                    leaf_suffix(row, named),
                    row.ranks,
                    row.threads,
                );
                // Thread-nanoseconds per interaction: the slowest rank's seconds times its
                // threads over the interactions of every rank (exact on one rank).
                let per = |seconds: f64, interactions: u64| match interactions {
                    0 => "–".to_string(),
                    n => format!("{:.3}", seconds * row.threads as f64 * 1e9 / n as f64),
                };
                let p2p: f64 = row.p2p_seconds.iter().sum();
                let interactions: u64 = row.near_interactions.iter().sum();
                let mut lines = vec![format!(
                    "{head} | Σ | {} | {} | {} | {} | {} | {} | {p2p:.4} | {interactions} | {} \
                     | {:.4} | {:.4} |",
                    row.v_near[0],
                    row.v_near[1],
                    row.v_pairs.iter().sum::<usize>(),
                    row.remote_multipoles,
                    row.source_points[0],
                    row.source_points[1],
                    per(p2p, interactions),
                    row.m2l_seconds.iter().sum::<f64>(),
                    row.evaluate,
                )];
                if arguments.levels {
                    for l in (0..row.levels).filter(|&l| row.near_interactions[l] > 0) {
                        lines.push(format!(
                            "{head} | {l} | – | – | {} | – | – | – | {:.4} | {} | {} | {:.4} | – |",
                            row.v_pairs[l],
                            row.p2p_seconds[l],
                            row.near_interactions[l],
                            per(row.p2p_seconds[l], row.near_interactions[l]),
                            row.m2l_seconds[l],
                        ));
                    }
                }
                lines
            };
            // The row of the table "Leaves" (Helmholtz CH4.0d3).
            let re_k = complex_k(&case).re;
            let leaves_line = |row: &Row| -> String {
                let all = row.errors.last().expect("an entry over all vectors");
                let occupied: usize = row.leaves_by_level.iter().map(|c| c[1]).sum();
                let by_level: Vec<String> = row
                    .leaves_by_level
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| c[0] > 0)
                    .map(|(l, c)| format!("{l}:{}/{}", c[1], c[0]))
                    .collect();
                let p2p: f64 = row.p2p_seconds.iter().sum();
                let m2l: f64 = row.m2l_seconds.iter().sum();
                let kind = |j: usize| row.kind_seconds.iter().map(|k| k[j]).sum::<f64>();
                let entries = |j: usize| row.far_leaf_entries.iter().map(|e| e[j]).sum::<usize>();
                let phase = row.phase_bound(re_k);
                format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {}{} | {} × {} | {} | {} | {} | {} \
                     | {} | {} | {} | {:.1} | {} | {:.4} | {:.2} | {:.4} | {:.4} | {:.4} | {:.4} \
                     | {} | {} | {:.1} | {:.1} | {:.3} | {:.3} | {:.2e} | {:.2e} |",
                    workload.name(),
                    problem.len(),
                    setting_name(row),
                    empty_suffix(row),
                    row.ranks,
                    row.threads,
                    row.max_points_per_leaf,
                    row.c_leaf,
                    row.levels,
                    row.min_leaf_level,
                    row.leaves,
                    occupied,
                    by_level.join(" "),
                    problem.len() as f64 / occupied.max(1) as f64,
                    row.degrees.iter().max().copied().unwrap_or(0),
                    p2p,
                    p2p / row.evaluate,
                    m2l,
                    kind(4),
                    kind(5),
                    kind(0) + kind(1) + kind(2) + kind(3),
                    entries(0),
                    entries(1),
                    phase,
                    phase / row.c_leaf,
                    row.build,
                    row.evaluate,
                    all[0],
                    all[1],
                )
            };
            // The row of the table "Build" (Helmholtz CH4.0h): the build by part.
            let build_line = |row: &Row| -> String {
                let parts: Vec<String> =
                    row.build_parts.iter().map(|s| format!("{s:.3}")).collect();
                format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {}{}{} | {} × {} | {:.3} | {} \
                     | {:.3} | {:016x} |",
                    workload.name(),
                    problem.len(),
                    setting_name(row),
                    empty_suffix(row),
                    leaf_suffix(row, named),
                    row.ranks,
                    row.threads,
                    row.build,
                    parts.join(" | "),
                    row.evaluate,
                    row.checksum,
                )
            };
            // The rows of the table "Leaf operators" (Helmholtz CH4.0d3): the run's, and with
            // `--levels` one per level with P2P, P2L or M2P work.
            let operator_lines = |row: &Row| -> Vec<String> {
                let head = format!(
                    "| {} | {} | {kd_string} | {degrees_name} | {}{}{} | {} × {}",
                    workload.name(),
                    problem.len(),
                    setting_name(row),
                    empty_suffix(row),
                    leaf_suffix(row, named),
                    row.ranks,
                    row.threads,
                );
                // Thread-nanoseconds per interaction, as in "Near pairs".
                let per = |seconds: f64, interactions: u64| match interactions {
                    0 => "–".to_string(),
                    n => format!("{:.3}", seconds * row.threads as f64 * 1e9 / n as f64),
                };
                let line =
                    |level: String, p: String, [p2p, p2l, m2p]: [f64; 3], [n, x, w]: [u64; 3]| {
                        format!(
                            "{head} | {level} | {p} | {p2p:.4} | {n} | {} | {p2l:.4} | {x} | {} \
                         | {m2p:.4} | {w} | {} | {} | {} | {} | {} | {:.4} |",
                            per(p2p, n),
                            per(p2l, x),
                            per(m2p, w),
                            row.wx_near[0],
                            row.wx_near[1],
                            row.wx_near[2],
                            row.wx_near[3],
                            row.evaluate,
                        )
                    };
                let sum = |f: &dyn Fn(usize) -> f64| (0..row.levels).map(f).sum::<f64>();
                let total = |f: &dyn Fn(usize) -> u64| (0..row.levels).map(f).sum::<u64>();
                let mut lines = vec![line(
                    "Σ".into(),
                    "–".into(),
                    [
                        sum(&|l| row.p2p_seconds[l]),
                        sum(&|l| row.kind_seconds[l][4]),
                        sum(&|l| row.kind_seconds[l][5]),
                    ],
                    [
                        total(&|l| row.near_interactions[l]),
                        total(&|l| row.far_interactions[l][0]),
                        total(&|l| row.far_interactions[l][1]),
                    ],
                )];
                if arguments.levels {
                    for l in 0..row.levels {
                        let [x, w] = row.far_interactions[l];
                        if row.near_interactions[l] + x + w == 0 {
                            continue;
                        }
                        lines.push(line(
                            l.to_string(),
                            row.degrees[l].to_string(),
                            [
                                row.p2p_seconds[l],
                                row.kind_seconds[l][4],
                                row.kind_seconds[l][5],
                            ],
                            [row.near_interactions[l], x, w],
                        ));
                    }
                }
                lines
            };
            // The errors of the case's Rotation rows by P2P kernel and threads.
            let mut rotation: Vec<((P2pChoice, usize), [f64; 2])> = Vec::new();
            let k = complex_k(&case);
            // `--empty-boxes`, the outermost of the run loops, then `--near-v`, then
            // `--near-wx`.
            let near_wx = &arguments.near_wx;
            let rules: Vec<(bool, VNear, WxNear)> = arguments
                .empty_boxes
                .iter()
                .flat_map(|&empty_boxes| {
                    arguments.near_v.iter().flat_map(move |&near_v| {
                        near_wx
                            .iter()
                            .map(move |&near_wx| (empty_boxes, near_v, near_wx))
                    })
                })
                .collect();
            // `--leaf-size` and `--c-leaf` outside them (Helmholtz CH4.0d3): each a new tree.
            let trees: Vec<(Option<usize>, Option<f64>)> = arguments
                .leaf_sizes
                .iter()
                .flat_map(|&n| arguments.c_leaves.iter().map(move |&c| (n, c)))
                .collect();
            // `--rcr-split` outermost (Helmholtz CH4.0i): no bit depends on it.
            let mut runs_by_rule = Vec::new();
            for &rcr in &arguments.rcr_splits {
                for &tree in &trees {
                    for &rule in &rules {
                        runs_by_rule.push((rcr, tree, rule));
                    }
                }
            }
            let rules = runs_by_rule;
            for (rcr_split, (leaf_size, c_leaf), (empty_boxes, near_v, near_wx)) in rules {
                case.rcr_split = rcr_split;
                case.max_points_per_leaf = leaf_size;
                case.c_leaf = c_leaf;
                case.empty_boxes = empty_boxes;
                case.near_v = near_v;
                case.near_wx = near_wx;
                for &m2l_run in &arguments.m2l {
                    // `--term` for the Auto entries, `--block` for those and Batched's.
                    let (auto, batched) = match m2l_run {
                        M2lRun::Fixed(HelmholtzM2l::Auto, ..)
                        | M2lRun::Sweep
                        | M2lRun::AutoPlaneWave(_) => (true, true),
                        M2lRun::Fixed(HelmholtzM2l::Batched, ..) => (false, true),
                        _ => (false, false),
                    };
                    let terms = if auto {
                        &arguments.terms[..]
                    } else {
                        &arguments.terms[..1]
                    };
                    let blocks = if batched {
                        &arguments.blocks[..]
                    } else {
                        &arguments.blocks[..1]
                    };
                    let variants = terms
                        .iter()
                        .flat_map(|&term| blocks.iter().map(move |&block| (term, block)));
                    for (term, block) in variants {
                        case.batched_term = term;
                        case.m2l_block_targets = block;
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
                                                raises
                                                    .iter()
                                                    .map(move |&raise| (reach, split, raise))
                                            })
                                        })
                                        .map(|(reach, split, raise)| {
                                            (
                                                HelmholtzM2l::PlaneWave,
                                                None,
                                                None,
                                                match precision {
                                                    None => PlaneWavePrecision::Rule,
                                                    Some(q) => PlaneWavePrecision::PerLevel(vec![
                                                            Some(q);
                                                            64
                                                        ]),
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
                                        .flat_map(|&split| {
                                            raises.iter().map(move |&raise| (split, raise))
                                        })
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
                                            let all = *row
                                                .errors
                                                .last()
                                                .expect("an entry over all vectors");
                                            let reference = rotation
                                                .iter()
                                                .find(|(key, _)| *key == (p2p, threads))
                                                .map(|&(_, errors)| errors);
                                            let is_rotation = case.m2l == HelmholtzM2l::Rotation;
                                            if root {
                                                print(&row, reference.filter(|_| !is_rotation));
                                                if arguments.levels {
                                                    levels.extend(per_level(&row));
                                                    scratch_rows.extend(scratch_lines(&row));
                                                }
                                                left_out.push(left_out_line(&row));
                                                near.extend(near_lines(&row));
                                                leaf_rows.push(leaves_line(&row));
                                                build_rows.push(build_line(&row));
                                                leaf_operators.extend(operator_lines(&row));
                                            }
                                            if is_rotation && reference.is_none() {
                                                rotation.push(((p2p, threads), all));
                                            }
                                            if next == 1 {
                                                match m2l_run {
                                                    M2lRun::Sweep => runs.extend(
                                                        sweep_crossovers(&row.degrees)[1..]
                                                            .iter()
                                                            .map(|&d| {
                                                                fixed(
                                                                    HelmholtzM2l::Auto,
                                                                    None,
                                                                    Some(d),
                                                                )
                                                            }),
                                                    ),
                                                    M2lRun::PlaneWaveScan(precision) => {
                                                        let precisions = match precision {
                                                            Some(q) => vec![q],
                                                            None => Precision::ALL.to_vec(),
                                                        };
                                                        for l in plane_wave_levels(&row, k) {
                                                            for &q in &precisions {
                                                                for &reach in &arguments.reaches {
                                                                    let mut named =
                                                                        vec![None; row.levels];
                                                                    named[l] = Some(q);
                                                                    runs.push((
                                                                        HelmholtzM2l::PlaneWave,
                                                                        None,
                                                                        None,
                                                                        PlaneWavePrecision::PerLevel(
                                                                            named,
                                                                        ),
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
             | children s | conversions s | threads busy | boxes | m̂ | visits | columns per read \
             | blocks |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        );
        for line in &levels {
            println!("{line}");
        }
        println!();
        println!("## Diagonal-form scratch");
        println!();
        println!(
            "Per level with V pairs, as if every V pair ran the diagonal form (Helmholtz CH4.1b; \
             `scratch_shape` in f64 at the level's degree and the run's threads): the most \
             targets with a V row and distinct V sources on one rank, the rank's budget (8 GiB \
             per node over the ranks per node), the most MiB of a rank's source signatures and \
             of its whole scratch (among the ranks that fit), the fewest targets per chunk, the \
             most chunks, and the ranks whose scratch does not fit (RCR there)."
        );
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | level | p | targets | sources \
             | budget MiB | sources MiB | scratch MiB | chunk | chunks | over budget |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: |"
        );
        for line in &scratch_rows {
            println!("{line}");
        }
    }
    if root {
        println!();
        println!("## Empty boxes");
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | rule | boxes without targets \
             | their V | X | W | L2L | V without sources | W | M2M | V pairs | remote multipoles |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: | ---: |"
        );
        for line in &left_out {
            println!("{line}");
        }
        println!();
        println!("## Near pairs");
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | level | V pairs offered \
             | converted | V pairs | remote multipoles | source points sent | received | P2P s \
             | near interactions | ns per interaction | M2L s | evaluate s |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: | ---: | ---: |"
        );
        for line in &near {
            println!("{line}");
        }
        println!();
        println!("## Leaves");
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | leaf size | c_leaf | levels \
             | leaf level | leaves | occupied | occupied/leaves by level | N per occupied \
             | largest p | P2P s | P2P share | M2L s | P2L s | M2P s | P2M, M2M, L2L, L2P s \
             | X entries | W entries | Re(k) r bound | ÷ c_leaf | build s | evaluate s | φ | ∇φ |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- \
             | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: | ---: | ---: |"
        );
        for line in &leaf_rows {
            println!("{line}");
        }
        println!();
        println!("## Leaf operators");
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | level | p | P2P s \
             | near interactions | ns per interaction | P2L s | X interactions | ns per X \
             | M2P s | W interactions | ns per W | W pairs offered | converted \
             | X pairs offered | converted | evaluate s |"
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: \
             | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for line in &leaf_operators {
            println!("{line}");
        }
        println!();
        println!("## Build");
        println!();
        println!(
            "| Workload | N | kD | degrees | M2L | ranks × threads | build s | {} s | evaluate s \
             | checksum |",
            BUILD_PARTS.join(" s | ")
        );
        println!(
            "| --- | ---: | --- | --- | --- | --- | ---: | {}---: | --- |",
            "---: | ".repeat(BUILD_PARTS.len())
        );
        for line in &build_rows {
            println!("{line}");
        }
    }
    if root && let Some((bytes, how)) = resident_memory() {
        println!();
        println!("Rank 0: {:.0} MB {how}.", bytes / 1e6);
    }
}
