//! Whether the call windows of `KindTiming::Device` overlap (Phase 4S T5), shared by
//! `tests/device_metal.rs` and `tests/device_cuda.rs`, where the device times on itself.
//!
//! [`call_windows`] runs the uniform cube on `backend` in two builds, each evaluation
//! with the output of an untimed build bit for bit:
//! - with `kind_timings(Device)` alone, after two warm-up evaluations, ten evaluations
//!   (one window per call and one sync each), then on the same build ten with
//!   `Synchronous` (`Fmm::set_kind_timings`; a sync after the charge upload and after
//!   each call): it prints per kind the median time of each mode, and the call windows'
//!   sum against the wall time of `evaluate`;
//! - with `kind_timings(Device)` and `device_timestamps(true)`, so that the call windows
//!   nest in the five stage windows: ten evaluations (one window per call plus five, one
//!   sync), each printed with the wall time of `evaluate`, the sum of the stage spans, the
//!   sum of the call windows, and per stage group the call windows against the span of
//!   their stages: the upward stages (P2M and M2M against `upward_local` plus
//!   `upward_global`), `downward` (L2L, M2L, P2L) and `evaluate_leaves` (L2P, M2P, P2P).
//!   The calls of a stage run one after another on the one stream, so call windows that
//!   do not overlap add up to at most their stage's span; a larger sum shows overlap (or
//!   windows the device misattributes).
//!
//! Timings are reported, never asserted.

use std::time::{Duration, Instant};

use mpi::topology::SimpleCommunicator;
use nd_fmm_exec::fmm::{
    Backend, DeviceStage, FmmBuilder, KindTiming, KindTimings, OperatorKind, Output,
};

use crate::device_common::output_bits;
use crate::kind_common::{check_kinds, expected_calls};
use crate::tune_common::Real;

/// The evaluations timed after the warm-up.
const EVALUATIONS: usize = 10;

/// The kinds of each stage group, with the stages whose spans they lie in.
const GROUPS: [(&str, &[OperatorKind], &[DeviceStage]); 3] = [
    (
        "upward",
        &[OperatorKind::P2m, OperatorKind::M2m],
        &[DeviceStage::UpwardLocal, DeviceStage::UpwardGlobal],
    ),
    (
        "downward",
        &[OperatorKind::L2l, OperatorKind::M2l, OperatorKind::P2l],
        &[DeviceStage::Downward],
    ),
    (
        "leaves",
        &[OperatorKind::L2p, OperatorKind::M2p, OperatorKind::P2p],
        &[DeviceStage::EvaluateLeaves],
    ),
];

/// The sum of the times of `kinds`.
fn kinds_time(timings: &KindTimings, kinds: &[OperatorKind]) -> Duration {
    kinds.iter().map(|&kind| timings.get(kind).total()).sum()
}

/// The call windows against the stage windows on `backend`, the uniform cube of `n`
/// points at p = `p` with gradients, in T (module documentation). On one rank.
pub fn call_windows<T: Real>(
    backend: Backend,
    (points, charges): (&[[f64; 3]], &[T]),
    p: usize,
    comm: &SimpleCommunicator,
) {
    let precision = if size_of::<T>() == 4 { "f32" } else { "f64" };
    let builder = FmmBuilder::<T>::new(p).gradients(true).backend(backend);
    let first: Output<T> = builder
        .clone()
        .build(points, points, comm)
        .unwrap()
        .evaluate(charges)
        .unwrap();
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    // The call windows alone, against synchronous kind timings of the same build.
    let mut fmm = builder
        .clone()
        .kind_timings(KindTiming::Device)
        .build(points, points, comm)
        .unwrap();
    let calls: usize = expected_calls(fmm.plan()).iter().flatten().sum();
    for _ in 0..2 {
        fmm.evaluate(charges).unwrap();
    }
    let mut medians = Vec::new();
    for mode in [KindTiming::Device, KindTiming::Synchronous] {
        fmm.set_kind_timings(mode).unwrap();
        let mut times: [Vec<Duration>; 8] = Default::default();
        let mut ratios = Vec::new();
        for _ in 0..EVALUATIONS {
            let start = Instant::now();
            let output = fmm.evaluate(charges).unwrap();
            let wall = start.elapsed();
            assert_eq!(
                output_bits(&output),
                output_bits(&first),
                "{backend} {precision}: kind_timings({mode:?}) changes the output"
            );
            check_kinds("call windows", &fmm, &output, mode);
            let counters = fmm.device_counters().unwrap().evaluation;
            assert_eq!(
                (counters.windows, counters.syncs),
                match mode {
                    KindTiming::Device => (calls as u64, 1),
                    _ => (0, 2 + calls as u64),
                },
                "{backend} {precision} {mode:?}: windows and syncs of an evaluation"
            );
            let kinds = output.timings.kinds.expect("kind times");
            for (kind, time) in kinds.iter() {
                times[kind as usize].push(time.total());
            }
            ratios.push(kinds.total().as_secs_f64() / wall.as_secs_f64());
        }
        ratios.sort_by(f64::total_cmp);
        eprintln!(
            "rank 0: {backend} {precision} kind_timings({mode:?}), uniform cube N = {}, p = \
             {p} ({:?}), {calls} calls: the sum over the kinds is {:.3}–{:.3} of the wall \
             time of evaluate (median {:.3}, {EVALUATIONS} evaluations)",
            points.len(),
            fmm.strategy(),
            ratios[0],
            ratios[EVALUATIONS - 1],
            ratios[EVALUATIONS / 2]
        );
        medians.push(times.map(|mut t| {
            t.sort();
            t[EVALUATIONS / 2]
        }));
    }
    let kinds: Vec<String> = OperatorKind::ALL
        .iter()
        .filter(|&&kind| medians[0][kind as usize] + medians[1][kind as usize] > Duration::ZERO)
        .map(|&kind| {
            format!(
                "{kind} {:.3} / {:.3}",
                ms(medians[0][kind as usize]),
                ms(medians[1][kind as usize])
            )
        })
        .collect();
    eprintln!(
        "rank 0: {backend} {precision} median by kind, ms, Device / Synchronous: {}",
        kinds.join(", ")
    );

    // The call windows inside the stage windows.
    let mut fmm = builder
        .kind_timings(KindTiming::Device)
        .device_timestamps(true)
        .build(points, points, comm)
        .unwrap();
    for _ in 0..2 {
        fmm.evaluate(charges).unwrap();
    }
    eprintln!(
        "rank 0: {backend} {precision} kind_timings(Device) with device_timestamps(true), \
         uniform cube N = {}, p = {p} ({:?}), {calls} calls, in ms: wall of evaluate | sum \
         of the stage spans | sum of the call windows | {}",
        points.len(),
        fmm.strategy(),
        GROUPS
            .map(|(name, ..)| format!("{name}: calls / span"))
            .join(" | ")
    );
    let (mut overlapping, mut over_wall) = (0, 0);
    let mut ratios = Vec::new();
    let mut last = None;
    for _ in 0..EVALUATIONS {
        let start = Instant::now();
        let output = fmm.evaluate(charges).unwrap();
        let wall = start.elapsed();
        assert_eq!(
            output_bits(&output),
            output_bits(&first),
            "{backend} {precision}: kind_timings(Device) changes the output"
        );
        check_kinds("call windows", &fmm, &output, KindTiming::Device);
        let counters = fmm.device_counters().unwrap().evaluation;
        assert_eq!(
            (counters.windows, counters.syncs),
            (calls as u64 + DeviceStage::ALL.len() as u64, 1),
            "{backend} {precision}: windows and syncs of an evaluation"
        );
        let stages = output.timings.device.expect("device stage times");
        let kinds = output.timings.kinds.expect("kind times");
        let mut groups = Vec::new();
        let mut overlap = false;
        for (_, group, spans) in GROUPS {
            let (inner, span) = (
                kinds_time(&kinds, group),
                spans.iter().map(|&s| stages.get(s)).sum::<Duration>(),
            );
            overlap |= inner > span;
            groups.push(format!("{:.3} / {:.3}", ms(inner), ms(span)));
        }
        overlapping += usize::from(overlap);
        over_wall += usize::from(kinds.total() > wall);
        ratios.push(kinds.total().as_secs_f64() / wall.as_secs_f64());
        eprintln!(
            "rank 0:   {:.3} | {:.3} | {:.3} | {}",
            ms(wall),
            ms(stages.total()),
            ms(kinds.total()),
            groups.join(" | ")
        );
        last = Some(kinds);
    }
    ratios.sort_by(f64::total_cmp);
    eprintln!(
        "rank 0: {backend} {precision} call windows: {calls} windows plus five and one sync \
         per evaluation, the output bit for bit the default's; the call windows add up to \
         {:.3}–{:.3} of the wall time of evaluate (median {:.3}), more than it in \
         {over_wall} of {EVALUATIONS}; a stage group's call windows exceed its stage spans \
         (overlap) in {overlapping} of {EVALUATIONS} evaluations",
        ratios[0],
        ratios[EVALUATIONS - 1],
        ratios[EVALUATIONS / 2]
    );
    let last = last.expect("ten evaluations");
    let kinds: Vec<String> = last
        .iter()
        .filter(|(_, time)| time.calls > 0)
        .map(|(kind, time)| format!("{kind} {:.3} ({} calls)", ms(time.total()), time.calls))
        .collect();
    eprintln!(
        "rank 0: {backend} {precision} by kind, the last evaluation, ms: {}",
        kinds.join(", ")
    );
}
