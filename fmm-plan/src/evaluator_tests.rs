//! Pure local tests of the passes of the evaluator, on the synthetic trees of
//! `plan_tests.rs` without ghosts. The exchanges need MPI; on these trees there is
//! nothing to exchange, so the passes run on their own. No MPI.
use nd_octree::MortonKey;

use super::{Data, EvaluatorError, validate};
use crate::index_fmm::{BatchedIndexFmm, GlobalLeaves, IndexFmm, Walk, check_counts};
use crate::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PerPair, UpwardPass,
};
use crate::plan::{
    Plan,
    tests::{cases, plan},
};

/// A count from a hash of the key, zero for about one key in five.
fn hashed(key: MortonKey) -> usize {
    (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 32) as usize % 5
}

/// Every plan of `plan_tests.rs` without ghost keys, with a name.
fn local_plans() -> Vec<(String, Plan)> {
    cases()
        .into_iter()
        .filter(|(_, map)| map.values().all(|kind| !kind.is_ghost()))
        .map(|(name, map)| (name, plan(&map)))
        .collect()
}

/// Run the four local stages on fresh stores and return the target output.
fn run<Op: FmmOperator<Value = u32>>(
    plan: &Plan,
    mut op: Op,
    source_counts: &[usize],
    target_counts: &[usize],
) -> (Data<u32>, Op) {
    validate(plan, &op, source_counts, target_counts).unwrap();
    let nlocal = plan.index().leaves().nlocal();
    let leaves = GlobalLeaves::with_offset(plan, 0, nlocal);
    let mut data = Data::new(plan, &op, source_counts, target_counts);
    leaves.fill_sources(data.sources.range_mut(0..nlocal));
    data.upward_local(plan, &mut op);
    data.upward_global(plan, &mut op);
    data.downward(plan, &mut op);
    data.evaluate_leaves(plan, &mut op);
    (data, op)
}

#[test]
fn the_index_fmm_counts_every_source_point_once_on_every_path() {
    let plans = local_plans();
    assert!(plans.len() >= 8, "both the local and the global-root trees");
    for (name, plan) in &plans {
        let leaves = plan.index().leaves();
        let nlocal = leaves.nlocal();
        assert_eq!(leaves.len(), nlocal, "{name}: no ghost leaf");
        let numbering = GlobalLeaves::with_offset(plan, 0, nlocal);
        let key = |j: usize| leaves.key(j);
        let variants: [(&str, Vec<usize>, Vec<usize>); 3] = [
            ("ones", vec![1; nlocal], vec![1; nlocal]),
            (
                "hashed",
                (0..nlocal).map(|j| hashed(key(j))).collect(),
                (0..nlocal).map(|j| hashed(key(j) ^ 0x5555)).collect(),
            ),
            (
                "several",
                (0..nlocal).map(|j| 1 + hashed(key(j)) % 4).collect(),
                (0..nlocal).map(|j| 1 + hashed(key(j) + 1) % 3).collect(),
            ),
        ];
        for (variant, sources, targets) in variants {
            let global = numbering.by_global_index(&sources);
            let outputs = [
                run(plan, PerPair(IndexFmm::new(nlocal)), &sources, &targets).0,
                run(
                    plan,
                    BatchedIndexFmm::new(nlocal, Walk::Rows),
                    &sources,
                    &targets,
                )
                .0,
                run(
                    plan,
                    BatchedIndexFmm::new(nlocal, Walk::Groupings),
                    &sources,
                    &targets,
                )
                .0,
            ];
            for (path, data) in ["per pair", "rows", "groupings"].iter().zip(&outputs) {
                if let Err(message) = check_counts(plan.index(), &data.target_output, &global) {
                    panic!("{name}, {variant}, {path}: {message}");
                }
            }
            // The paths agree on every value, coefficients included.
            for data in &outputs[1..] {
                assert_eq!(data.target_output, outputs[0].target_output, "{name}");
                assert_eq!(data.multipoles, outputs[0].multipoles, "{name}");
                assert_eq!(data.locals, outputs[0].locals, "{name}");
            }
        }
    }
}

#[test]
fn a_second_evaluation_after_reset_is_identical() {
    for (name, plan) in local_plans() {
        let nlocal = plan.index().leaves().nlocal();
        let counts: Vec<usize> = (0..nlocal)
            .map(|j| 1 + hashed(plan.index().leaf_key(j)))
            .collect();
        let (mut data, mut op) = run(
            &plan,
            BatchedIndexFmm::new(nlocal, Walk::Groupings),
            &counts,
            &counts,
        );
        let first = data.clone();
        data.reset();
        assert!(data.target_output.as_slice().iter().all(|&v| v == 0));
        assert!(data.multipoles.as_slice().iter().all(|&v| v == 0));
        assert!(data.locals.as_slice().iter().all(|&v| v == 0));
        assert_eq!(data.sources, first.sources, "{name}: sources are kept");
        data.upward_local(&plan, &mut op);
        data.upward_global(&plan, &mut op);
        data.downward(&plan, &mut op);
        data.evaluate_leaves(&plan, &mut op);
        assert_eq!(data.target_output, first.target_output, "{name}");
        assert_eq!(data.multipoles, first.multipoles, "{name}");
        assert_eq!(data.locals, first.locals, "{name}");
    }
}

/// One level call: (method, level, pass of an M2M).
type Call = (&'static str, usize, Option<UpwardPass>);

/// Records every call and checks that its buffers line up with its views.
struct CallRecorder<'p> {
    plan: &'p Plan,
    calls: Vec<Call>,
}

impl FmmSizes for CallRecorder<'_> {
    type Value = u32;

    fn multipole_size(&self, level: usize) -> usize {
        1 + level
    }

    fn local_size(&self, level: usize) -> usize {
        2 + level
    }

    fn source_point_size(&self) -> usize {
        4
    }

    fn target_input_point_size(&self) -> usize {
        3
    }

    fn target_output_point_size(&self) -> usize {
        1
    }
}

impl CallRecorder<'_> {
    fn boxes(&self, level: usize) -> usize {
        self.plan.index().len(level)
    }
}

impl FmmOperator for CallRecorder<'_> {
    fn p2m(&mut self, b: P2m<'_, u32>) {
        assert!(std::ptr::eq(b.leaves, self.plan.level(b.level).p2m()));
        assert_eq!(b.multipoles.len(), self.boxes(b.level));
        assert_eq!(b.multipoles.size(), 1 + b.level);
        assert_eq!(b.sources.nleaves(), self.plan.index().leaves().len());
        self.calls.push(("p2m", b.level, None));
    }

    fn m2m(&mut self, b: M2m<'_, u32>) {
        let lists = self.plan.level(b.level);
        let view = match b.pass {
            UpwardPass::Local => lists.m2m_local(),
            UpwardPass::Global => lists.m2m_global(),
        };
        assert!(std::ptr::eq(b.children, view));
        assert_eq!(b.multipoles.len(), self.boxes(b.level));
        assert_eq!(b.child_multipoles.len(), self.boxes(b.level + 1));
        assert_eq!(b.child_multipoles.size(), 2 + b.level);
        self.calls.push(("m2m", b.level, Some(b.pass)));
    }

    fn m2l(&mut self, b: M2l<'_, u32>) {
        assert!(std::ptr::eq(b.pairs, self.plan.level(b.level).v()));
        assert_eq!(b.multipoles.len(), self.boxes(b.level));
        assert_eq!(b.locals.len(), self.boxes(b.level));
        self.calls.push(("m2l", b.level, None));
    }

    fn p2l(&mut self, b: P2l<'_, u32>) {
        assert!(std::ptr::eq(b.x, self.plan.level(b.level).x()));
        assert_eq!(b.locals.len(), self.boxes(b.level));
        self.calls.push(("p2l", b.level, None));
    }

    fn l2l(&mut self, b: L2l<'_, u32>) {
        assert!(std::ptr::eq(b.parents, self.plan.level(b.level).l2l()));
        assert_eq!(b.locals.len(), self.boxes(b.level));
        assert_eq!(b.locals.size(), 2 + b.level);
        assert_eq!(b.parent_locals.len(), self.boxes(b.level - 1));
        self.calls.push(("l2l", b.level, None));
    }

    fn l2p(&mut self, b: L2p<'_, u32>) {
        assert!(std::ptr::eq(b.boxes, self.plan.level(b.level).l2p()));
        assert_eq!(b.leaves, self.plan.index().leaves().local(b.level));
        assert_eq!(b.target_input.nleaves(), b.leaves.len());
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        assert_eq!(b.target_input.point_size(), 3);
        assert_eq!(b.locals.len(), self.boxes(b.level));
        self.calls.push(("l2p", b.level, None));
    }

    fn m2p(&mut self, b: M2p<'_, u32>) {
        assert!(std::ptr::eq(b.w, self.plan.level(b.level).w()));
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        let finer = if b.level + 1 < self.plan.nlevels() {
            self.boxes(b.level + 1)
        } else {
            0
        };
        assert_eq!(b.multipoles.len(), finer);
        self.calls.push(("m2p", b.level, None));
    }

    fn p2p(&mut self, b: P2p<'_, u32>) {
        assert!(std::ptr::eq(b.near, self.plan.level(b.level).near()));
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        assert_eq!(b.sources.nleaves(), self.plan.index().leaves().len());
        self.calls.push(("p2p", b.level, None));
    }
}

/// The calls of the four local stages, in the order of design §7.2.
fn expected_calls(nlevels: usize) -> Vec<Call> {
    let deepest = nlevels - 1;
    let mut calls = Vec::new();
    for level in (0..=deepest).rev() {
        calls.push(("p2m", level, None));
        if level > 0 {
            calls.push(("m2m", level - 1, Some(UpwardPass::Local)));
        }
    }
    for level in (0..deepest).rev() {
        calls.push(("m2m", level, Some(UpwardPass::Global)));
    }
    for level in 1..=deepest {
        calls.extend([
            ("l2l", level, None),
            ("m2l", level, None),
            ("p2l", level, None),
        ]);
    }
    for level in 0..=deepest {
        calls.extend([
            ("l2p", level, None),
            ("m2p", level, None),
            ("p2p", level, None),
        ]);
    }
    calls
}

#[test]
fn every_level_gets_every_call_once_in_pass_order() {
    // Also the trees with ghosts: the call order does not depend on the data.
    for (name, map) in cases() {
        let plan = plan(&map);
        let nlocal = plan.index().leaves().nlocal();
        let counts = vec![2; nlocal];
        let mut op = CallRecorder {
            plan: &plan,
            calls: Vec::new(),
        };
        validate(&plan, &op, &counts, &counts).unwrap();
        let source_counts = vec![1; plan.index().leaves().len()];
        let mut data = Data::new(&plan, &op, &source_counts, &counts);
        data.upward_local(&plan, &mut op);
        data.upward_global(&plan, &mut op);
        data.downward(&plan, &mut op);
        data.evaluate_leaves(&plan, &mut op);
        assert_eq!(op.calls, expected_calls(plan.nlevels()), "{name}");
    }
}

#[test]
fn invalid_counts_and_sizes_are_errors() {
    let (_, plan) = local_plans().swap_remove(0);
    let nlocal = plan.index().leaves().nlocal();
    let ones = vec![1; nlocal];
    let op = IndexFmm::new(nlocal);
    assert_eq!(validate(&plan, &PerPair(op), &ones, &ones), Ok(()));
    assert_eq!(
        validate(&plan, &PerPair(op), &ones[1..], &ones),
        Err(EvaluatorError::SourceCountsLength {
            expected: nlocal,
            actual: nlocal - 1
        })
    );
    assert_eq!(
        validate(
            &plan,
            &PerPair(op),
            &ones,
            &[ones.clone(), vec![1]].concat()
        ),
        Err(EvaluatorError::TargetCountsLength {
            expected: nlocal,
            actual: nlocal + 1
        })
    );
    let huge = vec![usize::MAX / 2; nlocal];
    assert_eq!(
        validate(&plan, &PerPair(op), &huge, &ones),
        Err(EvaluatorError::Overflow)
    );
    assert_eq!(
        validate(&plan, &PerPair(op), &ones, &huge),
        Err(EvaluatorError::Overflow)
    );

    /// The index FMM with one size set to zero.
    struct Zero(&'static str);
    impl FmmSizes for Zero {
        type Value = u32;
        fn multipole_size(&self, level: usize) -> usize {
            usize::from(self.0 != "multipole_size" || level == 0)
        }
        fn local_size(&self, _: usize) -> usize {
            usize::from(self.0 != "local_size")
        }
        fn source_point_size(&self) -> usize {
            usize::from(self.0 != "source_point_size")
        }
        fn target_input_point_size(&self) -> usize {
            0
        }
        fn target_output_point_size(&self) -> usize {
            usize::from(self.0 != "target_output_point_size")
        }
    }
    impl crate::operator::PairOperator for Zero {
        fn p2m(&mut self, _: MortonKey, _: &[u32], _: &mut [u32]) {}
        fn m2m(&mut self, _: MortonKey, _: MortonKey, _: usize, _: &[u32], _: &mut [u32]) {}
        fn m2l(&mut self, _: MortonKey, _: MortonKey, _: usize, _: &[u32], _: &mut [u32]) {}
        fn p2l(&mut self, _: MortonKey, _: MortonKey, _: &[u32], _: &mut [u32]) {}
        fn l2l(&mut self, _: MortonKey, _: MortonKey, _: usize, _: &[u32], _: &mut [u32]) {}
        fn l2p(&mut self, _: MortonKey, _: &[u32], _: &[u32], _: &mut [u32]) {}
        fn m2p(&mut self, _: MortonKey, _: MortonKey, _: &[u32], _: &[u32], _: &mut [u32]) {}
        fn p2p(&mut self, _: MortonKey, _: MortonKey, _: &[u32], _: &[u32], _: &mut [u32]) {}
    }
    for size in [
        "multipole_size",
        "local_size",
        "source_point_size",
        "target_output_point_size",
    ] {
        assert_eq!(
            validate(&plan, &PerPair(Zero(size)), &ones, &ones),
            Err(EvaluatorError::ZeroSize { size })
        );
    }
    assert_eq!(validate(&plan, &PerPair(Zero("")), &ones, &ones), Ok(()));
}
