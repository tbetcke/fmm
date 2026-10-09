//! The device path against the host path (Phase 4 T5, C4.1; T6, C4.2; T7, C4.3; T8, C4.4;
//! T9, C4.5; T10, C4.6), shared by
//! `tests/mpi_exec.rs` (the CPU runtime), `tests/device_metal.rs` (Metal, ignored) and
//! `tests/device_cuda.rs` (CUDA, ignored; Phase 4S T4).
//!
//! [`check_backend`] builds an `Fmm` on a device backend at one thread twice, and checks,
//! for the scenario it is given:
//!
//! With every operator kind on the host fallback (T5, requirement 8):
//! - the output equals the host path's bit for bit, for the scenario's charges and for
//!   a second charge vector (the charges reversed), and again for the first charges
//!   (two evaluations of the device path are bit-identical);
//! - the transfers, launches and syncs of each evaluation equal the formula of
//!   docs/design/device-path.md §4.1 and §7.2 for the scenario's plan
//!   ([`expected_evaluation`]), and no evaluation moves points, plan views, geometry or
//!   tables, which the build uploaded once;
//! - the build uploaded the points and the tables the report lists;
//! - every view on the device equals the plan's view it was uploaded from, the
//!   row-to-batch maps point at the batch entry of each row entry, and the box and leaf
//!   indices, point offsets and charge slots are those of the plan and the counts;
//! - the output pass (Phase 4S T9) runs on the device where it does f64 arithmetic (the
//!   CPU runtime, CUDA) and on the host otherwise (Metal), and every output equals the
//!   pass before T9 (`Fmm::reference_output`) bit for bit.
//! - the operator holds the host mirrors of the fallback, the target output's included
//!   (Phase 4S T11, [`expected_mirror_bytes`]).
//!
//! With the default placement (T10): every kind on the device under every strategy
//! ([`device_kinds`]):
//! - the kinds are placed so, with the backend's default P2P, leaf-operator, GEMM and
//!   rotation layouts, and the report lists one M2M (local and global pass), L2L or (under
//!   `Dense` and `Classes`) M2L level call for each view with a pair, with its pairs and
//!   chunks, or under `Rotation` one rotation M2L level call for each V view with a pair,
//!   with its pairs, and names the strategy as the device runs it ("Classes, run as dense
//!   on the device");
//! - the output lies within the FMM bounds of the host output (docs/phase4/README.md,
//!   "Accuracy measures"): relative L2 over all targets within 1e-12 (f64) or 1e-5
//!   (f32), for φ and for ∇φ, for both charge vectors;
//! - the multipoles and the locals of every level lie within the same bounds of the host
//!   path's (relative L2 per level): on one rank the root's multipole comes from the
//!   global pass, so this checks the device M2M of the `m2m_global` view against the
//!   host's;
//! - two evaluations of the first charges are bit-identical, and so is the output of a
//!   second `Fmm` built from the same input (T11, requirement 6);
//! - the operator holds no host mirror, and so no host copy of the target output: the
//!   evaluation's download is read in place (Phase 4S T11);
//! - the output pass (Phase 4S T9): every output equals the pass before T9 bit for bit,
//!   and a build with `output_pass(OutputPass::Host)` gives the same bits, with one launch
//!   fewer where the default ran the pass on the device;
//! - the transfers, launches and syncs of each evaluation equal the formula with the
//!   fallback transfers of each device kind replaced by one launch per level call, or for
//!   M2M, L2L and dense M2L three per chunk (gather, GEMM, reduction or scatter-add;
//!   device-path.md §8.1; rotation M2L is one launch per level call); with every kind on
//!   the device, an evaluation moves only the charges (one upload of N_s s bytes) and the
//!   output (one download of o N_t s bytes) and syncs once, at that download
//!   (device-path.md §4.1, §8.2; T11);
//! - the stages are timed by the host clock, with no timing window (T11, device-path.md
//!   §8.3); a build with `device_timestamps(true)` gives the same bits and, where the
//!   device times on itself (Metal, CUDA), opens one timing window per stage with device work
//!   (five) with no sync of its own and returns the device times with the output, while
//!   on the CPU runtime, whose windows wait for it, it opens none;
//! - per-kind timings (Phase 4S T5, `tests/kind_common`): a build with
//!   `kind_timings(Synchronous)` gives the same bits, two evaluations, each with one sync
//!   after the charge upload and one after every level call with a pair besides the
//!   download's, no window, the launches of the formula, and the calls of the plan per
//!   kind and level; where the device times on itself, so do builds with
//!   `kind_timings(Device)`, one window per call and the one sync, alone and with
//!   `device_timestamps(true)` (its five stage windows around the call windows).
//!
//! On several ranks (Phase 5 T8, docs/design/distributed-fmm.md §7) every check runs as on
//! one rank, against the host path on the same ranks: the host output and expansions there,
//! and the formula extended by the exchanges' transfers on this rank
//! ([`expected_evaluation`], from `Fmm::exchange_sizes`): the ghost tail of the sources up,
//! the sent multipoles down in one download (one more sync), the other ranks' coarse
//! blocks up, and each level's received ghost multipoles up, with a gather and a scatter
//! launch each. On one rank these are all zero, and the Phase 4 formula holds unchanged.
//!
//! Error measures: exact equality (bit patterns, counts and bytes), and the relative L2
//! difference from the host output.

use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, Equivalence};
use nd_fmm_exec::device::{DataKind, GroupedImage, StageTiming, Traffic};
use nd_fmm_exec::fmm::{
    Backend, Fmm, FmmBuilder, KindTiming, OperatorKind, Output, OutputPass, Placement,
};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_kernels::Precision;
use nd_fmm_kernels::leaf::LeafLayout;
use nd_fmm_kernels::p2p::P2pLayout;
use nd_fmm_kernels::rotation::RotationLayout;
use nd_fmm_kernels::translate::GemmLayout;
use nd_fmm_math::RealScalar;
use nd_fmm_plan::lists::GroupedCsr;
use nd_fmm_plan::operator::UpwardPass;
use nd_fmm_plan::plan::Plan;
use nd_fmm_tables::cache::Stored;
use nd_octree::morton;

/// The largest differences of the default placement from the host path (relative L2):
/// of the output (φ, ∇φ), and per level of the multipoles and the locals, with the
/// root's multipole (the global pass on one rank) on its own.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Differences {
    /// φ over all targets.
    pub potential: f64,
    /// ∇φ over all targets.
    pub gradient: f64,
    /// The multipoles, the worst level.
    pub multipoles: f64,
    /// The locals, the worst level.
    pub locals: f64,
    /// The multipole of the root.
    pub root: f64,
}

/// The values as f64 bit patterns: potentials, then gradients.
pub fn output_bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut values: Vec<u64> = output
        .potential
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect();
    if let Some(gradient) = &output.gradient {
        values.extend(
            gradient
                .as_flattened()
                .iter()
                .map(|&v| RealScalar::to_f64(v).to_bits()),
        );
    }
    values
}

/// Asserts that two outputs agree bit for bit, naming the first difference.
fn assert_same<T: RealScalar>(what: &str, device: &Output<T>, host: &Output<T>) {
    let (a, b) = (output_bits(device), output_bits(host));
    assert_eq!(a.len(), b.len(), "{what}: output lengths");
    let differing = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    if let Some(i) = a.iter().zip(&b).position(|(x, y)| x != y) {
        panic!(
            "{what}: {differing} of {} values differ from the host path, the first at {i}: \
             {:e} against {:e}",
            a.len(),
            f64::from_bits(a[i]),
            f64::from_bits(b[i])
        );
    }
}

/// The transfers, launches and syncs of one evaluation, by the formula of the device
/// module's documentation: the data kinds in [`DataKind::ALL`] order, then the launches
/// and the syncs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Expected {
    /// Transfers per kind of data.
    pub traffic: [Traffic; DataKind::ALL.len()],
    /// Kernel launches.
    pub launches: u64,
    /// Syncs: one per download.
    pub syncs: u64,
}

impl Expected {
    fn down(&mut self, data: DataKind, bytes: usize) {
        let t = &mut self.traffic[data as usize];
        t.downloads += 1;
        t.download_bytes += bytes as u64;
        self.syncs += 1;
    }

    fn up(&mut self, data: DataKind, bytes: usize) {
        let t = &mut self.traffic[data as usize];
        t.uploads += 1;
        t.upload_bytes += bytes as u64;
    }

    /// The sum over every kind of data.
    fn total(&self) -> Traffic {
        self.traffic
            .iter()
            .fold(Traffic::default(), |a, b| Traffic {
                uploads: a.uploads + b.uploads,
                upload_bytes: a.upload_bytes + b.upload_bytes,
                downloads: a.downloads + b.downloads,
                download_bytes: a.download_bytes + b.download_bytes,
            })
    }
}

/// The transfers of one evaluation of `fmm` with every kind on the host fallback
/// (device-path.md §4.1, §7.2; the device module's tables). With s bytes per value,
/// n_c = (p + 1)², o values per target, K_l boxes and N_l targets on level l:
///
/// - the charges: one upload of N_s s bytes; the output: one download of o N_t s;
/// - per level call whose view has an entry (and, for L2P, M2P and P2P, whose leaves
///   have a target): P2M down and up K_l n_c s; M2M (each pass) down (K_l + K_{l+1})
///   n_c s in one call, up K_l n_c s; M2L down K_l n_c s twice, up once; P2L down and up
///   K_l n_c s; L2L down (K_{l−1} + K_l) n_c s, up K_l n_c s; L2P down K_l n_c s and
///   o N_l s, up o N_l s; M2P down K_{l+1} n_c s and o N_l s, up o N_l s; P2P down and
///   up o N_l s;
/// - launches: the three zero kernels of non-empty stores and the charge scatter if
///   there is a source, and with the output pass on the device (Phase 4S T9) its gather
///   if there is a target; syncs: one per download;
/// - on several ranks (Phase 5 T8, design §7.2), with G_s ghost source points, B_o and B_r
///   sent and received coarse blocks and S_l and R_l sent and received boxes of level l
///   (`Fmm::exchange_sizes`): up 4 G_s s (the ghost tail, if G_s > 0); down (B_o +
///   Σ_l S_l) n_c s in one download after one gather launch (if anything is sent); up
///   B_r n_c s and one scatter launch (if B_r > 0); per level with R_l > 0 up R_l n_c s
///   and one scatter launch. Every placement moves them, the host fallback as the device.
///
/// A kind in `on_device` (T6: P2P; T7: P2M, P2L, L2P, M2P; T8: M2M, L2L; T9: M2L) moves
/// nothing: each of its level calls is one launch instead of its downloads and uploads, or
/// for M2M, L2L and M2L three launches per chunk, with the chunks of the device report
/// (which this checks against the plan's views).
/// The bytes of the host mirrors a device `Fmm` holds (Phase 4S T11,
/// `DeviceReport::host_mirror_bytes`): the multipoles and the locals of every level with any
/// kind on the host fallback, and the target output (o N_t values) with L2P, M2P or P2P on
/// it; nothing with every kind on the device, where the evaluation's download is read in
/// place and the operator keeps no host copy of the output.
pub fn expected_mirror_bytes<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
) -> u64 {
    let host = |kind: OperatorKind| fmm.placement(kind) == Placement::Host;
    let index = fmm.plan().index();
    let nc = (fmm.p() + 1) * (fmm.p() + 1);
    let o = if fmm.gradients() { 4 } else { 1 };
    let mut values = 0;
    if OperatorKind::ALL.into_iter().any(host) {
        values += 2
            * nc
            * (0..fmm.plan().nlevels())
                .map(|l| index.len(l))
                .sum::<usize>();
    }
    if [OperatorKind::L2p, OperatorKind::M2p, OperatorKind::P2p]
        .into_iter()
        .any(host)
    {
        values += o * fmm.owned_points().1;
    }
    (values * size_of::<T>()) as u64
}

pub fn expected_evaluation<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
    on_device: &[OperatorKind],
) -> Expected {
    let device = |kind: OperatorKind| on_device.contains(&kind);
    let plan = fmm.plan();
    let index = plan.index();
    let nlevels = plan.nlevels();
    let s = size_of::<T>();
    let nc = (fmm.p() + 1) * (fmm.p() + 1);
    let o = if fmm.gradients() { 4 } else { 1 };
    let boxes = |l: usize| index.len(l) * nc * s;
    let level_targets = |l: usize| -> usize {
        fmm.target_counts()[index.leaves().local(l)]
            .iter()
            .sum::<usize>()
            * o
            * s
    };
    let mut e = Expected::default();
    let (multipoles, locals, output) = (
        DataKind::FallbackMultipoles,
        DataKind::FallbackLocals,
        DataKind::FallbackTargetOutput,
    );
    // The points this rank owns: the device holds those (Phase 5 T6, T8).
    let (nsources, ntargets) = fmm.owned_points();
    let nboxes: usize = (0..nlevels).map(|l| index.len(l)).sum();

    e.up(DataKind::Charges, nsources * s);
    e.launches += 2 * u64::from(nboxes > 0) + u64::from(ntargets > 0) + u64::from(nsources > 0);
    // The exchanges (Phase 5 T8): none on one rank.
    let x = fmm.exchange_sizes();
    if x.ghost_sources > 0 {
        e.up(DataKind::GhostSources, 4 * x.ghost_sources * s);
    }
    let sent = x.sent_blocks + x.sent_boxes.iter().sum::<usize>();
    if sent > 0 {
        e.launches += 1;
        e.down(DataKind::SentMultipoles, sent * nc * s);
    }
    if x.received_blocks > 0 {
        e.up(DataKind::CoarseMultipoles, x.received_blocks * nc * s);
        e.launches += 1;
    }
    for &r in &x.received_boxes {
        if r > 0 {
            e.up(DataKind::ReceivedMultipoles, r * nc * s);
            e.launches += 1;
        }
    }
    for l in 0..nlevels {
        let lists = plan.level(l);
        if !lists.p2m().is_empty() {
            if device(OperatorKind::P2m) {
                e.launches += 1;
            } else {
                e.down(multipoles, boxes(l));
                e.up(multipoles, boxes(l));
            }
        }
    }
    // The chunks of a device M2M or L2L level call, from the report.
    let chunks = |kind: OperatorKind, pass: Option<UpwardPass>, level: usize, pairs: usize| {
        let report = fmm.device_report().expect("a device backend");
        let calls: Vec<_> = report
            .translations
            .iter()
            .filter(|t| t.kind == kind && t.pass == pass && t.level == level)
            .collect();
        assert_eq!(
            calls.len(),
            1,
            "{kind} {pass:?} level {level}: one report entry"
        );
        assert_eq!(
            calls[0].pairs, pairs,
            "{kind} {pass:?} level {level}: pairs"
        );
        assert!(calls[0].chunks >= 1);
        calls[0].chunks as u64
    };
    for l in 0..nlevels.saturating_sub(1) {
        for (pass, view) in [
            (UpwardPass::Local, plan.level(l).m2m_local()),
            (UpwardPass::Global, plan.level(l).m2m_global()),
        ] {
            if !view.is_empty() {
                if device(OperatorKind::M2m) {
                    e.launches += 3 * chunks(OperatorKind::M2m, Some(pass), l, view.len());
                } else {
                    e.down(multipoles, boxes(l) + boxes(l + 1));
                    e.up(multipoles, boxes(l));
                }
            }
        }
    }
    for l in 1..nlevels {
        let lists = plan.level(l);
        if !lists.l2l().is_empty() {
            if device(OperatorKind::L2l) {
                e.launches += 3 * chunks(OperatorKind::L2l, None, l, lists.l2l().len());
            } else {
                e.down(locals, boxes(l - 1) + boxes(l));
                e.up(locals, boxes(l));
            }
        }
        if !lists.v().is_empty() {
            if device(OperatorKind::M2l) && fmm.strategy() == M2lStrategy::Rotation {
                // One launch of the rotation kernel (T10), reported with its pairs.
                let report = fmm.device_report().expect("a device backend");
                let calls: Vec<_> = report.rotations.iter().filter(|r| r.level == l).collect();
                assert_eq!(calls.len(), 1, "rotation M2L level {l}: one report entry");
                assert_eq!(
                    calls[0].pairs,
                    lists.v().len(),
                    "rotation M2L level {l}: pairs"
                );
                e.launches += 1;
            } else if device(OperatorKind::M2l) {
                e.launches += 3 * chunks(OperatorKind::M2l, None, l, lists.v().len());
            } else {
                e.down(multipoles, boxes(l));
                e.down(locals, boxes(l));
                e.up(locals, boxes(l));
            }
        }
        if !lists.x().is_empty() {
            if device(OperatorKind::P2l) {
                e.launches += 1;
            } else {
                e.down(locals, boxes(l));
                e.up(locals, boxes(l));
            }
        }
    }
    for l in 0..nlevels {
        let lists = plan.level(l);
        let targets = level_targets(l);
        if targets == 0 {
            continue;
        }
        if !lists.l2p().is_empty() {
            if device(OperatorKind::L2p) {
                e.launches += 1;
            } else {
                e.down(locals, boxes(l));
                e.down(output, targets);
                e.up(output, targets);
            }
        }
        if !lists.w().is_empty() {
            if device(OperatorKind::M2p) {
                e.launches += 1;
            } else {
                e.down(multipoles, boxes(l + 1));
                e.down(output, targets);
                e.up(output, targets);
            }
        }
        if !lists.near().is_empty() {
            if device(OperatorKind::P2p) {
                e.launches += 1;
            } else {
                e.down(output, targets);
                e.up(output, targets);
            }
        }
    }
    if fmm.output_pass() == Placement::Device && ntargets > 0 {
        e.launches += 1;
    }
    e.down(DataKind::Output, ntargets * o * s);
    e
}

/// The exchanges' kinds of data (Phase 5 T8), which move only on several ranks.
const EXCHANGES: [DataKind; 4] = [
    DataKind::GhostSources,
    DataKind::SentMultipoles,
    DataKind::CoarseMultipoles,
    DataKind::ReceivedMultipoles,
];

/// Where the output pass runs on `backend` by default (Phase 4S T9): on the device where
/// it does f64 arithmetic.
fn default_output_pass<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
) -> Placement {
    if fmm.device_report().expect("a device backend").info.f64 {
        Placement::Device
    } else {
        Placement::Host
    }
}

/// Checks `output`, the last evaluation of `fmm`, against the output pass before Phase 4S
/// T9 (`Fmm::reference_output`, the test oracle) bit for bit.
pub fn check_reference<T: Stored + SimdScalar + Equivalence + Default>(
    what: &str,
    fmm: &mut Fmm<'_, T>,
    output: &Output<T>,
) {
    let reference = fmm.reference_output().expect("the reference output");
    assert_same(
        &format!("{what}, against the pass before T9"),
        output,
        &reference,
    );
}

/// With every kind on the device, `expected` is the design's minimum (device-path.md
/// §4.1, §8.1, §8.2): one upload of the charges, one download of the output, one sync,
/// and nothing moved by a host-fallback call; on several ranks (Phase 5 T8, design §7.2)
/// besides the exchanges' transfers, and one more sync where the rank sends multipoles.
fn check_minimum<T: Stored + SimdScalar + Equivalence + Default>(
    backend: Backend,
    fmm: &Fmm<'_, T>,
    expected: &Expected,
) {
    let s = size_of::<T>() as u64;
    let o = if fmm.gradients() { 4 } else { 1 };
    let (nsources, ntargets) = fmm.owned_points();
    for data in DataKind::ALL {
        if EXCHANGES.contains(&data) {
            continue;
        }
        let want = match data {
            DataKind::Charges => Traffic {
                uploads: 1,
                upload_bytes: nsources as u64 * s,
                ..Traffic::default()
            },
            DataKind::Output => Traffic {
                downloads: 1,
                download_bytes: ntargets as u64 * o * s,
                ..Traffic::default()
            },
            _ => Traffic::default(),
        };
        assert_eq!(
            expected.traffic[data as usize], want,
            "{backend}: every kind on the device, {data} per evaluation"
        );
    }
    let sends = expected.traffic[DataKind::SentMultipoles as usize].downloads;
    assert!(
        sends <= 1,
        "{backend}: at most one download of the sent multipoles"
    );
    assert_eq!(
        expected.syncs,
        1 + sends,
        "{backend}: one sync per evaluation, one more where the rank sends"
    );
}

/// Checks the received multipoles of the last evaluation of `fmm` level by level
/// (`DeviceCounters::received_levels`, Phase 5 T8): one upload of R_l n_c s bytes on each
/// level with a ghost box, nothing elsewhere and nothing downloaded.
fn check_received_levels<T: Stored + SimdScalar + Equivalence + Default>(
    what: &str,
    fmm: &Fmm<'_, T>,
) {
    let counters = fmm.device_counters().expect("a device backend");
    let nc = ((fmm.p() + 1) * (fmm.p() + 1)) as u64;
    let s = size_of::<T>() as u64;
    let received = fmm.exchange_sizes().received_boxes;
    for (level, traffic) in counters.received_levels.iter().enumerate() {
        let r = received.get(level).copied().unwrap_or(0) as u64;
        let want = Traffic {
            uploads: u64::from(r > 0),
            upload_bytes: r * nc * s,
            ..Traffic::default()
        };
        assert_eq!(
            *traffic, want,
            "{what}: received multipoles of level {level}"
        );
    }
}

/// Checks the counters of the last evaluation of `fmm` against `expected`.
fn check_evaluation_counters<T: Stored + SimdScalar + Equivalence + Default>(
    what: &str,
    fmm: &Fmm<'_, T>,
    expected: &Expected,
) {
    let counters = fmm.device_counters().expect("a device backend");
    for data in DataKind::ALL {
        assert_eq!(
            counters.evaluation_traffic.get(data),
            expected.traffic[data as usize],
            "{what}: transfers of {data} in an evaluation"
        );
    }
    let total = expected.total();
    let c = counters.evaluation;
    assert_eq!(
        (
            c.uploads,
            c.upload_bytes,
            c.downloads,
            c.download_bytes,
            c.launches,
            c.syncs
        ),
        (
            total.uploads,
            total.upload_bytes,
            total.downloads,
            total.download_bytes,
            expected.launches,
            expected.syncs
        ),
        "{what}: counters of an evaluation (uploads, bytes, downloads, bytes, launches, \
         syncs)"
    );
    check_received_levels(what, fmm);
}

/// Checks that the device's copy of `view` is the plan's, and that its row-to-batch map
/// points at the batch entry of each row entry.
fn check_grouped<G: Copy + Into<usize> + Into<u32>>(
    what: &str,
    image: &GroupedImage,
    view: &GroupedCsr<G>,
) {
    assert_eq!(image.row_offsets, view.row_offsets(), "{what}: row offsets");
    assert_eq!(image.sources, view.sources(), "{what}: sources");
    let groups: Vec<u32> = view.groups().iter().map(|&g| g.into()).collect();
    assert_eq!(image.groups, groups, "{what}: groups");
    assert_eq!(
        image.batch_offsets,
        view.batch_offsets(),
        "{what}: batch offsets"
    );
    assert_eq!(
        image.batch_targets,
        view.batch_targets(),
        "{what}: batch targets"
    );
    assert_eq!(
        image.batch_sources,
        view.batch_sources(),
        "{what}: batch sources"
    );
    for t in 0..view.nrows() {
        let row = view.row_offsets()[t] as usize..view.row_offsets()[t + 1] as usize;
        for e in row {
            let k = image.row_to_batch[e] as usize;
            let g = groups[e] as usize;
            let batch = view.batch_offsets()[g] as usize..view.batch_offsets()[g + 1] as usize;
            assert!(
                batch.contains(&k)
                    && view.batch_targets()[k] == t as u32
                    && view.batch_sources()[k] == view.sources()[e],
                "{what}: row {t}, entry {e}: the row-to-batch map points at {k}"
            );
        }
    }
}

/// Checks every array on the device against `plan` and the counts of `fmm`.
fn check_views<T: Stored + SimdScalar + Equivalence + Default>(fmm: &mut Fmm<'_, T>) {
    let image = fmm
        .download_device_views()
        .expect("a device backend")
        .expect("the views download");
    let plan: &Plan = fmm.plan();
    let index = plan.index();
    assert_eq!(image.levels.len(), plan.nlevels());
    for (l, (level, lists)) in image.levels.iter().zip(plan.levels()).enumerate() {
        for (name, csr, view) in [
            ("P2M", &level.p2m, lists.p2m()),
            ("X", &level.x, lists.x()),
            ("near", &level.near, lists.near()),
            ("W", &level.w, lists.w()),
            ("L2P", &level.l2p, lists.l2p()),
        ] {
            assert_eq!(csr.row_offsets, view.row_offsets(), "level {l}, {name}");
            assert_eq!(csr.entries, view.entries(), "level {l}, {name}");
        }
        check_grouped(
            &format!("level {l}, M2M local"),
            &level.m2m_local,
            lists.m2m_local(),
        );
        check_grouped(
            &format!("level {l}, M2M global"),
            &level.m2m_global,
            lists.m2m_global(),
        );
        check_grouped(&format!("level {l}, L2L"), &level.l2l, lists.l2l());
        check_grouped(&format!("level {l}, V"), &level.v, lists.v());
    }
    let decode = |key| {
        let (level, i) = morton::decode(key);
        (level as u32, i.map(|c| c as u32))
    };
    for (l, boxes) in image.boxes.iter().enumerate() {
        let want: Vec<[u32; 3]> = index.keys(l).iter().map(|&k| decode(k).1).collect();
        assert_eq!(boxes, &want, "box indices of level {l}");
    }
    let leaves: Vec<(u32, [u32; 3])> = index.leaves().keys().iter().map(|&k| decode(k)).collect();
    assert_eq!(image.leaves, leaves, "leaf levels and indices");
    let offsets = |counts: &[usize]| -> Vec<u32> {
        std::iter::once(0)
            .chain(counts.iter().scan(0, |total, &n| {
                *total += n;
                Some(*total as u32)
            }))
            .collect()
    };
    // The source offsets cover every leaf of the numbering: the local leaves', then the
    // ghost leaves' (Phase 5 T8), whose counts come with the source exchange.
    let local = offsets(fmm.source_counts());
    let nlocal = fmm.source_counts().len();
    assert_eq!(image.source_offsets.len(), index.leaves().len() + 1);
    assert_eq!(
        image.source_offsets[..=nlocal],
        local[..],
        "local source offsets"
    );
    assert!(image.source_offsets.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(
        *image.source_offsets.last().unwrap() as usize - local[nlocal] as usize,
        fmm.exchange_sizes().ghost_sources,
        "the ghost leaves' source offsets"
    );
    assert_eq!(image.target_offsets, offsets(fmm.target_counts()));
    let mut slots = Vec::new();
    let mut start = 0;
    for &n in fmm.source_counts() {
        slots.extend((0..n).map(|k| (start + 3 * n + k) as u32));
        start += 4 * n;
    }
    assert_eq!(image.charge_slots, slots, "charge slots");
}

/// The relative L2 difference of `got` from `want` over all targets, for φ and for ∇φ
/// (0 for an output without gradients or without targets).
pub fn relative_l2<T: RealScalar>(got: &Output<T>, want: &Output<T>) -> (f64, f64) {
    let difference = |a: &[T], b: &[T]| -> f64 {
        let (mut d2, mut r2) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (RealScalar::to_f64(x), RealScalar::to_f64(y));
            d2 += (x - y) * (x - y);
            r2 += y * y;
        }
        if r2 > 0.0 {
            (d2 / r2).sqrt()
        } else {
            d2.sqrt()
        }
    };
    let potential = difference(&got.potential, &want.potential);
    let gradient = match (&got.gradient, &want.gradient) {
        (Some(a), Some(b)) => difference(a.as_flattened(), b.as_flattened()),
        _ => 0.0,
    };
    (potential, gradient)
}

/// The FMM bound of the device output against the host output (docs/phase4/README.md,
/// "Accuracy measures"): 1e-12 in f64, 1e-5 in f32, relative L2.
pub fn fmm_bound<T>() -> f64 {
    if size_of::<T>() == 4 { 1e-5 } else { 1e-12 }
}

/// Builds `builder` on `backend` at one thread and checks it against the host path:
/// `host` is the one-thread host `Fmm` of the same settings, `host_output` its output
/// for `charges` (module documentation): first with every kind on the host fallback,
/// then with the default placement ([`device_kinds`] on the device), on every rank count
/// (Phase 5 T8). Returns the largest relative L2 differences of the default placement from
/// the host path ([`Differences`]); panics on a failed check.
pub fn check_backend<'o, T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    (sources, targets): (&[[f64; 3]], &[[f64; 3]]),
    charges: &[T],
    host: &mut Fmm<'o, T>,
    host_output: &Output<T>,
    backend: Backend,
    comm: &'o SimpleCommunicator,
) -> Differences {
    check_fallback(
        builder,
        (sources, targets),
        charges,
        host,
        host_output,
        backend,
        comm,
    );
    check_default(
        builder,
        (sources, targets),
        charges,
        host,
        host_output,
        backend,
        comm,
    )
}

/// The kinds the device runs by default from T10 under `strategy`: every kind, M2L by the
/// dense tables or (under `Rotation`) by the rotation kernel.
pub fn device_kinds(strategy: M2lStrategy) -> Vec<OperatorKind> {
    let _ = strategy;
    OperatorKind::ALL.to_vec()
}

/// The worst relative L2 difference per level of the device's multipoles and locals
/// from the host path's, after an evaluation of the same charges on both: (multipoles,
/// locals, the root's multipole). Asserts each within `bound`.
fn check_expansions<T: Stored + SimdScalar + Equivalence + Default>(
    what: &str,
    device: &mut Fmm<'_, T>,
    host: &mut Fmm<'_, T>,
    bound: f64,
) -> (f64, f64, f64) {
    let (dm, dl) = device.expansions().expect("the device expansions download");
    let (hm, hl) = host.expansions().expect("the host expansions");
    let plan = host.plan();
    let n = (host.p() + 1) * (host.p() + 1);
    let mut offsets = vec![0];
    for l in 0..plan.nlevels() {
        offsets.push(offsets[l] + plan.index().len(l) * n);
    }
    let mut worst = (0.0f64, 0.0f64, 0.0f64);
    for (name, got, want) in [("multipoles", &dm, &hm), ("locals", &dl, &hl)] {
        assert_eq!(got.len(), want.len(), "{what}: {name}");
        for l in 0..plan.nlevels() {
            let range = offsets[l]..offsets[l + 1];
            let (mut d2, mut r2) = (0.0f64, 0.0f64);
            for (&x, &y) in got[range.clone()].iter().zip(&want[range]) {
                let (x, y) = (RealScalar::to_f64(x), RealScalar::to_f64(y));
                d2 += (x - y) * (x - y);
                r2 += y * y;
            }
            let e = if r2 > 0.0 {
                (d2 / r2).sqrt()
            } else {
                d2.sqrt()
            };
            assert!(
                e <= bound,
                "{what}: {name} of level {l} {e:e} from the host path's (bound {bound:e})"
            );
            if name == "multipoles" {
                worst.0 = worst.0.max(e);
                if l == 0 {
                    worst.2 = e;
                }
            } else {
                worst.1 = worst.1.max(e);
            }
        }
    }
    worst
}

/// The default-placement half of [`check_backend`] (module documentation), on one rank.
fn check_default<'o, T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    (sources, targets): (&[[f64; 3]], &[[f64; 3]]),
    charges: &[T],
    host: &mut Fmm<'o, T>,
    host_output: &Output<T>,
    backend: Backend,
    comm: &'o SimpleCommunicator,
) -> Differences {
    let mut fmm = builder
        .clone()
        .threads(1)
        .backend(backend)
        .build(sources, targets, comm)
        .unwrap_or_else(|error| panic!("{backend}: the FMM does not build: {error}"));
    let report = fmm.device_report().expect("a device backend");
    let kinds = device_kinds(fmm.strategy());
    for kind in OperatorKind::ALL {
        let want = if kinds.contains(&kind) {
            Placement::Device
        } else {
            Placement::Host
        };
        assert_eq!(fmm.placement(kind), want, "{backend}: {kind}");
    }
    assert_eq!(
        report.host_mirror_bytes,
        expected_mirror_bytes(&fmm),
        "{backend}: the host mirrors"
    );
    if kinds.len() == OperatorKind::ALL.len() {
        assert_eq!(
            report.host_mirror_bytes, 0,
            "{backend}: no host copy of the output with every kind on the device"
        );
    }
    assert_eq!(
        report.p2p_layout,
        P2pLayout::default_for(&report.info),
        "{backend}: the default P2P layout"
    );
    let precision = if size_of::<T>() == 4 {
        Precision::F32
    } else {
        Precision::F64
    };
    assert_eq!(
        report.leaf_layout,
        LeafLayout::default_for(&report.info, fmm.p(), precision),
        "{backend}: the default leaf-operator layout"
    );
    let n = (fmm.p() + 1) * (fmm.p() + 1);
    assert_eq!(
        report.gemm_layout,
        GemmLayout::default_for(&report.info, n),
        "{backend}: the default GEMM layout"
    );
    assert_eq!(
        report.rotation_layout,
        RotationLayout::default_for(&report.info, fmm.p()),
        "{backend}: the default rotation layout"
    );
    let strategy_name = match fmm.strategy() {
        M2lStrategy::Dense => "Dense",
        M2lStrategy::Classes => "Classes, run as dense on the device",
        _ => "Rotation",
    };
    assert_eq!(
        report.strategy_name(),
        strategy_name,
        "{backend}: the strategy"
    );
    let m2l_calls = report.translations_of(OperatorKind::M2l).count();
    let v_views = (0..fmm.nlevels())
        .filter(|&l| !fmm.plan().level(l).v().is_empty())
        .count();
    let rotation = fmm.strategy() == M2lStrategy::Rotation;
    assert_eq!(
        (m2l_calls, report.rotations.len()),
        match (kinds.contains(&OperatorKind::M2l), rotation) {
            (false, _) => (0, 0),
            (true, false) => (v_views, 0),
            (true, true) => (0, v_views),
        },
        "{backend}: one M2L report entry per V view with a pair"
    );
    let expected = expected_evaluation(&fmm, &kinds);
    if kinds.len() == OperatorKind::ALL.len() {
        check_minimum(backend, &fmm, &expected);
    }
    assert_eq!(
        report.stage_timing,
        StageTiming::Enqueue,
        "{backend}: the stage timing"
    );
    let second: Vec<T> = charges.iter().rev().copied().collect();
    let host_second = host.evaluate(&second).expect("the host FMM evaluates");
    let bound = fmm_bound::<T>();
    let mut worst = (0.0f64, 0.0f64);
    let mut first_bits = None;
    for (what, q, want) in [
        ("first charges", charges, host_output),
        ("second charges", &second[..], &host_second),
        ("first charges again", charges, host_output),
    ] {
        let output = fmm.evaluate(q).expect("the device FMM evaluates");
        let what = format!("{backend}, default placement, {what}");
        let (potential, gradient) = relative_l2(&output, want);
        assert!(
            potential <= bound && gradient <= bound,
            "{what}: relative L2 difference from the host φ {potential:e}, ∇φ {gradient:e} \
             exceeds {bound:e}"
        );
        worst = (worst.0.max(potential), worst.1.max(gradient));
        check_evaluation_counters(&what, &fmm, &expected);
        let windows = fmm.device_counters().unwrap().evaluation.windows;
        assert_eq!(windows, 0, "{what}: no timing window");
        assert!(
            output.timings.device.is_none(),
            "{what}: no device stage times"
        );
        let bits = output_bits(&output);
        match (&first_bits, std::ptr::eq(q, charges)) {
            (None, true) => first_bits = Some(bits),
            (Some(first), true) => assert_eq!(&bits, first, "{what}: two evaluations differ"),
            _ => {}
        }
        check_reference(&what, &mut fmm, &output);
    }
    assert_eq!(
        fmm.output_pass(),
        default_output_pass(&fmm),
        "{backend}: the output pass"
    );
    // A second build of the same input gives the same bits (requirement 6), and so does
    // one with device timestamps (device-path.md §8.3), which adds timing windows where
    // the device times on itself and no sync.
    let first_bits = first_bits.expect("the first charges ran");
    let timestamps = fmm
        .device_report()
        .expect("a device backend")
        .info
        .backend
        .is_gpu();
    for (what, builder) in [
        ("a second build", builder.clone()),
        (
            "device_timestamps(true)",
            builder.clone().device_timestamps(true),
        ),
        (
            "output_pass(Host)",
            builder.clone().output_pass(OutputPass::Host),
        ),
    ] {
        let mut again = builder
            .threads(1)
            .backend(backend)
            .build(sources, targets, comm)
            .unwrap_or_else(|error| panic!("{backend}: {what} does not build: {error}"));
        let output = again.evaluate(charges).expect("the device FMM evaluates");
        assert_eq!(
            output_bits(&output),
            first_bits,
            "{backend}, default placement: {what} differs from the first build"
        );
        if what.starts_with("output_pass") {
            // The host pass: one launch fewer where the default ran it on the device.
            assert_eq!(again.output_pass(), Placement::Host);
            let gather =
                u64::from(fmm.output_pass() == Placement::Device && fmm.owned_points().1 > 0);
            let counters = again.device_counters().unwrap().evaluation;
            assert_eq!(
                (counters.syncs, counters.launches, counters.download_bytes),
                (
                    expected.syncs,
                    expected.launches - gather,
                    expected.total().download_bytes
                ),
                "{backend}, {what}: syncs, launches and download bytes of an evaluation"
            );
            check_reference(&format!("{backend}, {what}"), &mut again, &output);
        }
        if what.starts_with("device_timestamps") {
            let windowed = timestamps;
            let report = again.device_report().expect("a device backend");
            assert_eq!(
                report.stage_timing,
                if windowed {
                    StageTiming::DeviceTimestamps
                } else {
                    StageTiming::Enqueue
                },
                "{backend}, {what}: the stage timing"
            );
            let counters = again.device_counters().unwrap().evaluation;
            assert_eq!(
                (counters.windows, counters.syncs, counters.launches),
                (
                    if windowed { 5 } else { 0 },
                    expected.syncs,
                    expected.launches
                ),
                "{backend}, {what}: windows, syncs and launches of an evaluation"
            );
            assert_eq!(
                output.timings.device.is_some(),
                windowed,
                "{backend}, {what}: the device stage times"
            );
        }
    }
    // Per-kind timings (Phase 4S T5): the same bits in every mode. `Synchronous` adds one
    // sync after the charge upload and one after every level call with a pair, and no
    // window; `Device`, where the device times on itself, one window per such call and no
    // sync, also inside the stage windows of `device_timestamps`.
    let mut modes = vec![(KindTiming::Synchronous, false)];
    if timestamps {
        modes.extend([(KindTiming::Device, false), (KindTiming::Device, true)]);
    }
    for (mode, stage_windows) in modes {
        let what = format!(
            "{backend}, default placement, kind_timings({mode:?}){}",
            if stage_windows {
                " with device_timestamps(true)"
            } else {
                ""
            }
        );
        let mut timed = builder
            .clone()
            .threads(1)
            .backend(backend)
            .kind_timings(mode)
            .device_timestamps(stage_windows)
            .build(sources, targets, comm)
            .unwrap_or_else(|error| panic!("{what} does not build: {error}"));
        for _ in 0..2 {
            let output = timed.evaluate(charges).expect("the device FMM evaluates");
            assert_eq!(output_bits(&output), first_bits, "{what}: differs from Off");
            let calls = crate::kind_common::check_kinds(&what, &timed, &output, mode);
            let counters = timed.device_counters().unwrap().evaluation;
            let (windows, syncs) = match mode {
                KindTiming::Synchronous => (0, expected.syncs + 1 + calls as u64),
                _ => (
                    calls as u64 + if stage_windows { 5 } else { 0 },
                    expected.syncs,
                ),
            };
            assert_eq!(
                (counters.windows, counters.syncs, counters.launches),
                (windows, syncs, expected.launches),
                "{what}: windows, syncs and launches of an evaluation ({calls} calls)"
            );
            assert_eq!(
                output.timings.device.is_some(),
                stage_windows,
                "{what}: the device stage times"
            );
            crate::kind_common::record(&format!("{backend} {mode:?}"), calls);
        }
    }
    // The expansions after the first charges on both paths (the host last evaluated the
    // second charges).
    host.evaluate(charges).expect("the host FMM evaluates");
    let (multipoles, locals, root) = check_expansions(
        &format!("{backend}, default placement"),
        &mut fmm,
        host,
        bound,
    );
    Differences {
        potential: worst.0,
        gradient: worst.1,
        multipoles,
        locals,
        root,
    }
}

/// The host-fallback half of [`check_backend`] (module documentation): every kind on the
/// host fallback, bit for bit the host path.
fn check_fallback<'o, T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    (sources, targets): (&[[f64; 3]], &[[f64; 3]]),
    charges: &[T],
    host: &mut Fmm<'o, T>,
    host_output: &Output<T>,
    backend: Backend,
    comm: &'o SimpleCommunicator,
) {
    let mut fmm = builder
        .clone()
        .threads(1)
        .backend(backend)
        .host_fallback(OperatorKind::ALL)
        .build(sources, targets, comm)
        .unwrap_or_else(|error| {
            panic!(
                "rank {}: {backend} on {} rank(s): the FMM does not build: {error}",
                comm.rank(),
                comm.size()
            )
        });
    assert_eq!(fmm.backend(), backend);
    assert_eq!(fmm.operator().threads(), 1);
    assert_eq!(
        fmm.output_pass(),
        default_output_pass(&fmm),
        "{backend}: the output pass"
    );
    let report = fmm.device_report().expect("a device backend");
    for kind in OperatorKind::ALL {
        assert_eq!(
            fmm.placement(kind),
            Placement::Host,
            "{kind}: every kind on the host fallback"
        );
    }
    assert_eq!(report.info.backend.name(), backend.name());
    assert_eq!(
        report.host_mirror_bytes,
        expected_mirror_bytes(&fmm),
        "{backend}: the host mirrors, the target output's included"
    );

    // The build: the points in two uploads, the source store with its ghost tail (zero
    // until the first source exchange; Phase 5 T8), the tables the report lists.
    let s = size_of::<T>();
    let build = fmm.device_counters().unwrap().build_traffic;
    let (nsources, ntargets) = fmm.owned_points();
    let stored = nsources + fmm.exchange_sizes().ghost_sources;
    assert_eq!(
        build.get(DataKind::Points),
        Traffic {
            uploads: 2,
            upload_bytes: ((4 * stored + 3 * ntargets) * s) as u64,
            ..Traffic::default()
        },
        "{backend}: points uploaded at build"
    );
    let tables: u64 = report.tables.iter().map(|t| t.bytes).sum();
    assert_eq!(
        build.get(DataKind::Tables),
        Traffic {
            uploads: report.tables.len() as u64,
            upload_bytes: tables,
            ..Traffic::default()
        },
        "{backend}: tables uploaded at build"
    );
    assert!(build.get(DataKind::Indices).uploads > 0 && build.get(DataKind::Geometry).uploads == 2);
    assert_eq!(
        build.total().downloads,
        0,
        "{backend}: the build downloads nothing"
    );

    let expected = expected_evaluation(&fmm, &[]);
    let second: Vec<T> = charges.iter().rev().copied().collect();
    let host_second = host.evaluate(&second).expect("the host FMM evaluates");
    for (what, q, want) in [
        ("first charges", charges, host_output),
        ("second charges", &second[..], &host_second),
        ("first charges again", charges, host_output),
    ] {
        let output = fmm.evaluate(q).expect("the device FMM evaluates");
        assert_same(&format!("{backend}, {what}"), &output, want);
        check_evaluation_counters(&format!("{backend}, {what}"), &fmm, &expected);
        let counters = fmm.device_counters().unwrap();
        for data in [
            DataKind::Points,
            DataKind::Indices,
            DataKind::Geometry,
            DataKind::Tables,
        ] {
            assert_eq!(
                counters.evaluation_traffic.get(data),
                Traffic::default(),
                "{backend}, {what}: {data} moved in an evaluation"
            );
        }
        check_reference(&format!("{backend}, {what}"), &mut fmm, &output);
    }
    check_views(&mut fmm);
}
