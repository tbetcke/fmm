//! The per-kind timings of `FmmBuilder::kind_timings` (Phase 4S T5, C4S.5), shared by
//! `tests/mpi_exec.rs` (the host and the CPU runtime), `tests/device_metal.rs` (Metal,
//! ignored) and `tests/device_cuda.rs` (CUDA, ignored), through `tests/device_common` for
//! the device backends.
//!
//! [`check_kinds`] checks the `StageTimings::kinds` of one evaluation against the plan:
//! - the calls of every kind equal the level calls the evaluator makes with an entry in
//!   their view ([`expected_calls`], from the plan's views: the evaluator calls every
//!   kind on every level, also with an empty view, and an empty call is not counted), per
//!   level; a level without such a call has zero time, and so a kind whose views are all
//!   empty has zero calls and zero time;
//! - every kind reports the placement `Fmm::placement` gives it;
//! - with `KindTiming::Synchronous`, the sum over the kinds is at most the sum of the
//!   stages that call operators (`upward_local`, `upward_global`, `downward`,
//!   `evaluate_leaves`), whose host timers enclose every call's: a bound, not a timing
//!   assertion.
//!
//! Error measure: exact equality of counts and of zero times.

use std::time::Duration;

use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_fmm_exec::fmm::{Fmm, KindTiming, MAX_LEVELS, OperatorKind, Output};
use nd_fmm_exec::operator::SimdScalar;
use nd_fmm_plan::plan::Plan;
use nd_fmm_tables::cache::Stored;

/// The level calls with an entry in their view that an evaluation of `plan` makes, by
/// kind (in `OperatorKind::ALL` order) and level, in the pass order of
/// `nd_fmm_plan::evaluator`: P2M on every level, M2M (local and global pass) for the
/// parents on levels 0 to L − 1, L2L, M2L and P2L on levels 1 to L, and L2P, M2P and P2P
/// on every level.
pub fn expected_calls(plan: &Plan) -> [[usize; MAX_LEVELS]; 8] {
    let mut calls = [[0; MAX_LEVELS]; 8];
    let mut count = |kind: OperatorKind, level: usize, empty: bool| {
        calls[kind as usize][level] += usize::from(!empty);
    };
    let nlevels = plan.nlevels();
    for l in 0..nlevels {
        let lists = plan.level(l);
        count(OperatorKind::P2m, l, lists.p2m().is_empty());
        if l + 1 < nlevels {
            count(OperatorKind::M2m, l, lists.m2m_local().is_empty());
            count(OperatorKind::M2m, l, lists.m2m_global().is_empty());
        }
        if l >= 1 {
            count(OperatorKind::L2l, l, lists.l2l().is_empty());
            count(OperatorKind::M2l, l, lists.v().is_empty());
            count(OperatorKind::P2l, l, lists.x().is_empty());
        }
        count(OperatorKind::L2p, l, lists.l2p().is_empty());
        count(OperatorKind::M2p, l, lists.w().is_empty());
        count(OperatorKind::P2p, l, lists.near().is_empty());
    }
    calls
}

/// Checks the kind timings of `output`, an evaluation of `fmm` built with `mode` (module
/// documentation): `None` for `KindTiming::Off`, otherwise the calls of the plan per kind
/// and level, zero time where there is no call, each kind's placement, and with
/// `Synchronous` the sum over the kinds within the stages that call operators. Returns
/// the calls of the evaluation.
pub fn check_kinds<T, C>(
    what: &str,
    fmm: &Fmm<'_, T, C>,
    output: &Output<T>,
    mode: KindTiming,
) -> usize
where
    T: Stored + SimdScalar + Equivalence + Default,
    C: CommunicatorCollectives,
{
    let timings = &output.timings;
    let Some(kinds) = timings.kinds else {
        assert_eq!(
            mode,
            KindTiming::Off,
            "{what}: no kind timings with {mode:?}"
        );
        assert_eq!(
            timings.remainder(),
            None,
            "{what}: no remainder without kinds"
        );
        return 0;
    };
    assert_ne!(mode, KindTiming::Off, "{what}: kind timings with Off");
    assert_eq!(kinds.mode, mode, "{what}: the mode of the kind timings");
    let expected = expected_calls(fmm.plan());
    for (kind, time) in kinds.iter() {
        let want = &expected[kind as usize];
        assert_eq!(
            time.calls,
            want.iter().sum::<usize>(),
            "{what}: {kind} calls (by level {want:?})"
        );
        for (level, (&calls, &t)) in want.iter().zip(&time.levels).enumerate() {
            if calls == 0 {
                assert_eq!(
                    t,
                    Duration::ZERO,
                    "{what}: {kind} on level {level} without a call"
                );
            }
        }
        assert_eq!(
            time.placement,
            fmm.placement(kind),
            "{what}: {kind} placement"
        );
    }
    let operators =
        timings.upward_local + timings.upward_global + timings.downward + timings.evaluate_leaves;
    if mode == KindTiming::Synchronous {
        assert!(
            kinds.total() <= operators,
            "{what}: the kinds' {:?} exceed the operator stages' {operators:?}",
            kinds.total()
        );
    }
    assert_eq!(
        timings.remainder(),
        Some(timings.total().saturating_sub(kinds.total())),
        "{what}: the remainder"
    );
    expected.iter().flatten().sum()
}

/// The evaluations with kind timings checked so far, per label: (label, evaluations,
/// fewest calls, most calls).
static RUNS: std::sync::Mutex<Vec<(String, usize, usize, usize)>> =
    std::sync::Mutex::new(Vec::new());

/// Records an evaluation with kind timings under `label` (a backend and mode), with its
/// `calls`, for [`summary`].
pub fn record(label: &str, calls: usize) {
    let mut runs = RUNS.lock().unwrap();
    match runs.iter_mut().find(|r| r.0 == label) {
        Some(r) => {
            r.1 += 1;
            r.2 = r.2.min(calls);
            r.3 = r.3.max(calls);
        }
        None => runs.push((label.to_owned(), 1, calls, calls)),
    }
}

/// "kind timings: …", what [`record`] recorded: for each label the evaluations, each bit
/// for bit the output without them, and the level calls per evaluation.
pub fn summary() -> String {
    let runs = RUNS.lock().unwrap();
    if runs.is_empty() {
        return "kind timings: none checked".to_owned();
    }
    let runs: Vec<String> = runs
        .iter()
        .map(|(label, n, lo, hi)| format!("{label} {n} evaluations, {lo}–{hi} calls each"))
        .collect();
    format!(
        "kind timings (bit for bit without them, calls per kind and level as the plan's): {}",
        runs.join("; ")
    )
}
