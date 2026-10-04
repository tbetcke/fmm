//! The device path against the host path (Phase 4 T5, C4.1; T6, C4.2; T7, C4.3; T8, C4.4),
//! shared by
//! `tests/mpi_exec.rs` (the CPU runtime) and `tests/device_metal.rs` (Metal, ignored).
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
//!   indices, point offsets and charge slots are those of the plan and the counts.
//!
//! With the default placement (T8): every kind but M2L on the device, M2L on the host
//! fallback:
//! - the seven kinds are placed on the device, with the backend's default P2P,
//!   leaf-operator and GEMM layouts, and the report lists one M2M (local and global
//!   pass) or L2L level call for each view with a pair, with its pairs and chunks;
//! - the output lies within the FMM bounds of the host output (docs/phase4/README.md,
//!   "Accuracy measures"): relative L2 over all targets within 1e-12 (f64) or 1e-5
//!   (f32), for φ and for ∇φ, for both charge vectors;
//! - the multipoles and the locals of every level lie within the same bounds of the host
//!   path's (relative L2 per level): on one rank the root's multipole comes from the
//!   global pass, so this checks the device M2M of the `m2m_global` view against the
//!   host's;
//! - two evaluations of the first charges are bit-identical;
//! - the transfers, launches and syncs of each evaluation equal the formula with the
//!   fallback transfers of each device kind replaced by one launch per level call, or for
//!   M2M and L2L three per chunk (gather, GEMM, reduction or scatter-add; device-path.md
//!   §8.1).
//!
//! On several ranks the device build returns `DeviceNeedsOneRank` on every rank instead
//! (device-path.md §4.4), which [`check_backend`] checks and reports.
//!
//! Error measures: exact equality (bit patterns, counts and bytes), and the relative L2
//! difference from the host output.

use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, Equivalence};
use nd_fmm_exec::device::{DataKind, GroupedImage, Traffic};
use nd_fmm_exec::fmm::{
    Backend, Fmm, FmmBuilder, FmmError, OperatorKind, Output, Placement, SettingsError,
};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_kernels::Precision;
use nd_fmm_kernels::leaf::LeafLayout;
use nd_fmm_kernels::p2p::P2pLayout;
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

/// What [`check_backend`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The device path ran and passed every check.
    Ran,
    /// Several ranks: every rank returned `DeviceNeedsOneRank`.
    OneRankOnly,
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
    pub traffic: [Traffic; 9],
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
///   there is a source; syncs: one per download.
///
/// A kind in `on_device` (T6: P2P; T7: P2M, P2L, L2P, M2P; T8: M2M, L2L) moves nothing:
/// each of its level calls is one launch instead of its downloads and uploads, or for M2M
/// and L2L three launches per chunk, with the chunks of the device report (which this
/// checks against the plan's views).
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

    e.up(DataKind::Charges, fmm.nsources() * s);
    e.launches += 2 + u64::from(fmm.ntargets() > 0) + u64::from(fmm.nsources() > 0);
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
            e.down(multipoles, boxes(l));
            e.down(locals, boxes(l));
            e.up(locals, boxes(l));
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
    e.down(DataKind::Output, fmm.ntargets() * o * s);
    e
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
    assert_eq!(image.source_offsets, offsets(fmm.source_counts()));
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
/// then with the default placement ([`DEVICE_KINDS`] on the device). Returns what it did,
/// and the largest relative L2 differences of the default placement from the host path
/// ([`Differences`]); panics on a failed check.
pub fn check_backend<'o, T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    (sources, targets): (&[[f64; 3]], &[[f64; 3]]),
    charges: &[T],
    host: &mut Fmm<'o, T>,
    host_output: &Output<T>,
    backend: Backend,
    comm: &'o SimpleCommunicator,
) -> (Outcome, Differences) {
    let outcome = check_fallback(
        builder,
        (sources, targets),
        charges,
        host,
        host_output,
        backend,
        comm,
    );
    if outcome == Outcome::OneRankOnly {
        return (outcome, Differences::default());
    }
    let difference = check_default(
        builder,
        (sources, targets),
        charges,
        host,
        host_output,
        backend,
        comm,
    );
    (outcome, difference)
}

/// The kinds the device runs by default after T8.
pub const DEVICE_KINDS: [OperatorKind; 7] = [
    OperatorKind::P2m,
    OperatorKind::M2m,
    OperatorKind::P2l,
    OperatorKind::L2l,
    OperatorKind::L2p,
    OperatorKind::M2p,
    OperatorKind::P2p,
];

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
    for kind in OperatorKind::ALL {
        let want = if DEVICE_KINDS.contains(&kind) {
            Placement::Device
        } else {
            Placement::Host
        };
        assert_eq!(fmm.placement(kind), want, "{backend}: {kind}");
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
    let expected = expected_evaluation(&fmm, &DEVICE_KINDS);
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
        let bits = output_bits(&output);
        match (&first_bits, std::ptr::eq(q, charges)) {
            (None, true) => first_bits = Some(bits),
            (Some(first), true) => assert_eq!(&bits, first, "{what}: two evaluations differ"),
            _ => {}
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
) -> Outcome {
    let built = builder
        .clone()
        .threads(1)
        .backend(backend)
        .host_fallback(OperatorKind::ALL)
        .build(sources, targets, comm);
    if comm.size() > 1 {
        let ranks = comm.size() as usize;
        match built {
            Err(FmmError::InvalidSettings(SettingsError::DeviceNeedsOneRank { ranks: r }))
                if r == ranks => {}
            Err(error) => panic!("rank {}: {backend} on {ranks} ranks: {error}", comm.rank()),
            Ok(_) => panic!("rank {}: {backend} builds on {ranks} ranks", comm.rank()),
        }
        return Outcome::OneRankOnly;
    }
    let mut fmm =
        built.unwrap_or_else(|error| panic!("{backend}: the FMM does not build: {error}"));
    assert_eq!(fmm.backend(), backend);
    assert_eq!(fmm.operator().threads(), 1);
    let report = fmm.device_report().expect("a device backend");
    for kind in OperatorKind::ALL {
        assert_eq!(
            fmm.placement(kind),
            Placement::Host,
            "{kind}: every kind on the host fallback"
        );
    }
    assert_eq!(report.info.backend.name(), backend.name());

    // The build: the points in two uploads, the tables the report lists.
    let s = size_of::<T>();
    let build = fmm.device_counters().unwrap().build_traffic;
    assert_eq!(
        build.get(DataKind::Points),
        Traffic {
            uploads: 2,
            upload_bytes: ((4 * fmm.nsources() + 3 * fmm.ntargets()) * s) as u64,
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
    }
    check_views(&mut fmm);
    Outcome::Ran
}
