//! Pure local tests of the passes of the evaluator, on the synthetic trees of
//! `plan_tests.rs`. The exchanges need MPI; on the trees without ghosts there is
//! nothing to exchange, so the passes run on their own. No MPI.
use nd_octree::MortonKey;

use super::{Data, EvaluatorError, validate};
use crate::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PerPair, UpwardPass,
};
use crate::plan::{
    Plan,
    tests::{cases, plan},
};
use crate::store::{LeafSlice, LevelSlice};

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

/// The marker of box `i` of `level`: [`Markers`] writes it into the multipole of every
/// box, and into the local of every box below level 0.
fn box_mark(level: usize, i: usize) -> u32 {
    ((level as u32) << 24) + i as u32 + 1
}

/// The marker of leaf `j`: every value of its sources, its target input and its target
/// output.
fn leaf_mark(j: usize) -> u32 {
    0x8000_0000 + j as u32
}

/// Values per source point, target input point and target output point of [`Markers`].
const POINT_SIZES: [usize; 3] = [3, 2, 2];

/// A test operator that computes nothing: it checks that every batch hands the views
/// and the slices of its level and leaves, by markers.
///
/// Sources and target inputs carry [`leaf_mark`]. Every call checks the marker of every
/// chunk it reads, which also checks the chunk lengths against the counts, and fills
/// the chunks it writes with the marker of their box or leaf: P2M and M2M the
/// multipoles, L2L the locals, L2P the target output. M2L and P2L then check that the
/// locals of their targets were written by L2L, and M2P and P2P that the target output
/// was written by L2P. Holds on trees without ghosts, whose boxes are all written.
struct Markers<'p> {
    /// The tree and counts, for the messages.
    name: String,
    plan: &'p Plan,
    /// Per leaf of the numbering.
    source_counts: Vec<usize>,
    /// Per local leaf.
    target_counts: Vec<usize>,
}

impl FmmSizes for Markers<'_> {
    type Value = u32;

    fn multipole_size(&self, level: usize) -> usize {
        1 + level
    }

    fn local_size(&self, level: usize) -> usize {
        2 + level % 3
    }

    fn source_point_size(&self) -> usize {
        POINT_SIZES[0]
    }

    fn target_input_point_size(&self) -> usize {
        POINT_SIZES[1]
    }

    fn target_output_point_size(&self) -> usize {
        POINT_SIZES[2]
    }
}

/// Check that `chunk` holds `len` values, all equal to `mark`.
fn assert_marked(chunk: &[u32], len: usize, mark: u32, what: std::fmt::Arguments<'_>) {
    assert_eq!(chunk.len(), len, "{what}: length");
    assert!(chunk.iter().all(|&v| v == mark), "{what}: {chunk:?}");
}

impl Markers<'_> {
    /// The expected local of box `i` of `level`: nothing reaches level 0.
    fn local_mark(level: usize, i: usize) -> u32 {
        if level == 0 { 0 } else { box_mark(level, i) }
    }

    fn check_sources(&self, sources: &LeafSlice<'_, u32>, j: u32) {
        let j = j as usize;
        assert_eq!(sources.nleaves(), self.plan.index().leaves().len());
        let len = self.source_counts[j] * POINT_SIZES[0];
        let what = format_args!("{}: sources of leaf {j}", self.name);
        assert_marked(sources.chunk(j), len, leaf_mark(j), what);
    }

    fn check_multipole(&self, multipoles: &LevelSlice<'_, u32>, level: usize, i: u32) {
        let i = i as usize;
        let len = self.multipole_size(level);
        let what = format_args!("{}: multipole {i} of level {level}", self.name);
        assert_marked(multipoles.chunk(i), len, box_mark(level, i), what);
    }

    fn check_parent_local(&self, locals: &LevelSlice<'_, u32>, level: usize, i: u32) {
        let i = i as usize;
        let len = self.local_size(level);
        let mark = Self::local_mark(level, i);
        let what = format_args!("{}: local {i} of level {level}", self.name);
        assert_marked(locals.chunk(i), len, mark, what);
    }

    /// Check the target input of row `r` of `leaves`, and that its target output holds
    /// `output`, the marker of the leaf once L2P has run.
    fn check_targets(
        &self,
        leaves: &std::ops::Range<usize>,
        target_input: &LeafSlice<'_, u32>,
        target_output: &[u32],
        r: usize,
        output: u32,
    ) {
        let j = leaves.start + r;
        let count = self.target_counts[j];
        let input = target_input.chunk(r);
        let what = format_args!("{}: target input of leaf {j}", self.name);
        assert_marked(input, count * POINT_SIZES[1], leaf_mark(j), what);
        let what = format_args!("{}: target output of leaf {j}", self.name);
        assert_marked(target_output, count * POINT_SIZES[2], output, what);
    }
}

impl FmmOperator for Markers<'_> {
    fn p2m(&mut self, mut b: P2m<'_, u32>) {
        assert!(std::ptr::eq(b.leaves, self.plan.level(b.level).p2m()));
        assert_eq!(b.multipoles.size(), self.multipole_size(b.level));
        for t in 0..b.leaves.nrows() {
            let row = b.leaves.row(t);
            for &j in row {
                self.check_sources(&b.sources, j);
            }
            if !row.is_empty() {
                b.multipoles.chunk_mut(t).fill(box_mark(b.level, t));
            }
        }
    }

    fn m2m(&mut self, mut b: M2m<'_, u32>) {
        assert_eq!(b.multipoles.size(), self.multipole_size(b.level));
        for t in 0..b.children.nrows() {
            let children = b.children.row(t).0;
            for &c in children {
                self.check_multipole(&b.child_multipoles, b.level + 1, c);
            }
            if !children.is_empty() {
                b.multipoles.chunk_mut(t).fill(box_mark(b.level, t));
            }
        }
    }

    fn m2l(&mut self, b: M2l<'_, u32>) {
        assert!(std::ptr::eq(b.pairs, self.plan.level(b.level).v()));
        for t in 0..b.pairs.nrows() {
            let sources = b.pairs.row(t).0;
            for &s in sources {
                self.check_multipole(&b.multipoles, b.level, s);
            }
            if !sources.is_empty() {
                let len = self.local_size(b.level);
                let mark = box_mark(b.level, t);
                let what = format_args!("{}: local {t} of level {} after L2L", self.name, b.level);
                assert_marked(b.locals.chunk(t), len, mark, what);
            }
        }
    }

    fn p2l(&mut self, b: P2l<'_, u32>) {
        assert!(std::ptr::eq(b.x, self.plan.level(b.level).x()));
        for t in 0..b.x.nrows() {
            let row = b.x.row(t);
            for &j in row {
                self.check_sources(&b.sources, j);
            }
            if !row.is_empty() {
                let len = self.local_size(b.level);
                let mark = box_mark(b.level, t);
                let what = format_args!("{}: local {t} of level {} after L2L", self.name, b.level);
                assert_marked(b.locals.chunk(t), len, mark, what);
            }
        }
    }

    fn l2l(&mut self, mut b: L2l<'_, u32>) {
        assert!(std::ptr::eq(b.parents, self.plan.level(b.level).l2l()));
        assert_eq!(b.locals.size(), self.local_size(b.level));
        for t in 0..b.parents.nrows() {
            let parents = b.parents.row(t).0;
            for &p in parents {
                self.check_parent_local(&b.parent_locals, b.level - 1, p);
            }
            if !parents.is_empty() {
                b.locals.chunk_mut(t).fill(box_mark(b.level, t));
            }
        }
    }

    fn l2p(&mut self, mut b: L2p<'_, u32>) {
        assert!(std::ptr::eq(b.boxes, self.plan.level(b.level).l2p()));
        assert_eq!(b.leaves, self.plan.index().leaves().local(b.level));
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        for r in 0..b.boxes.nrows() {
            for &i in b.boxes.row(r) {
                self.check_parent_local(&b.locals, b.level, i);
            }
            // Nothing has written the target output yet.
            let output = b.target_output.chunk(r);
            self.check_targets(&b.leaves, &b.target_input, output, r, 0);
            let j = b.leaves.start + r;
            b.target_output.chunk_mut(r).fill(leaf_mark(j));
        }
    }

    fn m2p(&mut self, b: M2p<'_, u32>) {
        assert!(std::ptr::eq(b.w, self.plan.level(b.level).w()));
        assert_eq!(b.leaves, self.plan.index().leaves().local(b.level));
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        for r in 0..b.w.nrows() {
            for &s in b.w.row(r) {
                self.check_multipole(&b.multipoles, b.level + 1, s);
            }
            let output = b.target_output.chunk(r);
            let mark = leaf_mark(b.leaves.start + r);
            self.check_targets(&b.leaves, &b.target_input, output, r, mark);
        }
    }

    fn p2p(&mut self, b: P2p<'_, u32>) {
        assert!(std::ptr::eq(b.near, self.plan.level(b.level).near()));
        assert_eq!(b.leaves, self.plan.index().leaves().local(b.level));
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        for r in 0..b.near.nrows() {
            for &j in b.near.row(r) {
                self.check_sources(&b.sources, j);
            }
            let output = b.target_output.chunk(r);
            let mark = leaf_mark(b.leaves.start + r);
            self.check_targets(&b.leaves, &b.target_input, output, r, mark);
        }
    }
}

/// Fill the sources and target inputs of `data` with their leaf markers.
fn fill_marks(data: &mut Data<u32>) {
    for j in 0..data.sources.nleaves() {
        data.sources.chunk_mut(j).fill(leaf_mark(j));
    }
    for j in 0..data.target_input.nleaves() {
        data.target_input.chunk_mut(j).fill(leaf_mark(j));
    }
}

/// Run the four local stages.
fn run_stages<Op: FmmOperator<Value = u32>>(data: &mut Data<u32>, plan: &Plan, op: &mut Op) {
    data.upward_local(plan, op);
    data.upward_global(plan, op);
    data.downward(plan, op);
    data.evaluate_leaves(plan, op);
}

/// Run the four local stages with [`Markers`] on fresh, marked stores, and check that
/// every box and every local leaf was written.
fn run_markers<'p>(
    name: String,
    plan: &'p Plan,
    source_counts: &[usize],
    target_counts: &[usize],
) -> (Data<u32>, Markers<'p>) {
    let mut op = Markers {
        name,
        plan,
        source_counts: source_counts.to_vec(),
        target_counts: target_counts.to_vec(),
    };
    validate(plan, &op, source_counts, target_counts).unwrap();
    let mut data = Data::new(plan, &op, source_counts, target_counts);
    fill_marks(&mut data);
    run_stages(&mut data, plan, &mut op);

    let index = plan.index();
    for level in 0..plan.nlevels() {
        for i in 0..index.len(level) {
            let multipole = data.multipoles.chunk(level, i);
            let what = format_args!("{}: multipole {i} of level {level}", op.name);
            assert_marked(multipole, 1 + level, box_mark(level, i), what);
            let local = data.locals.chunk(level, i);
            let (len, mark) = (op.local_size(level), Markers::local_mark(level, i));
            let what = format_args!("{}: local {i} of level {level}", op.name);
            assert_marked(local, len, mark, what);
        }
    }
    for (j, &count) in target_counts.iter().enumerate() {
        let output = data.target_output.chunk(j);
        let what = format_args!("{}: target output of leaf {j}", op.name);
        assert_marked(output, count * POINT_SIZES[2], leaf_mark(j), what);
    }
    (data, op)
}

/// Source and target counts per leaf for `plan`: ones, hashed (zeros included) and
/// several points per leaf.
fn count_variants(plan: &Plan) -> [(&'static str, Vec<usize>, Vec<usize>); 3] {
    let leaves = plan.index().leaves();
    let nlocal = leaves.nlocal();
    let key = |j: usize| leaves.key(j);
    [
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
    ]
}

#[test]
fn every_batch_hands_the_slices_of_its_level_and_leaves() {
    let plans = local_plans();
    assert!(plans.len() >= 8, "both the local and the global-root trees");
    for (name, plan) in &plans {
        let leaves = plan.index().leaves();
        assert_eq!(leaves.len(), leaves.nlocal(), "{name}: no ghost leaf");
        for (variant, sources, targets) in count_variants(plan) {
            run_markers(format!("{name}, {variant}"), plan, &sources, &targets);
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
        let (mut data, mut op) = run_markers(name.clone(), &plan, &counts, &counts);
        let first = data.clone();
        data.reset();
        assert!(data.target_output.as_slice().iter().all(|&v| v == 0));
        assert!(data.multipoles.as_slice().iter().all(|&v| v == 0));
        assert!(data.locals.as_slice().iter().all(|&v| v == 0));
        assert_eq!(data.sources, first.sources, "{name}: sources are kept");
        assert_eq!(
            data.target_input, first.target_input,
            "{name}: target input is kept"
        );
        run_stages(&mut data, &plan, &mut op);
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

/// A per-pair operator that computes nothing, with one size set to zero (none for
/// `Zero("")`).
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

#[test]
fn invalid_counts_and_sizes_are_errors() {
    let (_, plan) = local_plans().swap_remove(0);
    let nlocal = plan.index().leaves().nlocal();
    assert!(nlocal >= 3, "enough leaves for the counts to overflow");
    let ones = vec![1; nlocal];
    let op = || PerPair(Zero(""));
    assert_eq!(validate(&plan, &op(), &ones, &ones), Ok(()));
    assert_eq!(
        validate(&plan, &op(), &ones[1..], &ones),
        Err(EvaluatorError::SourceCountsLength {
            expected: nlocal,
            actual: nlocal - 1
        })
    );
    assert_eq!(
        validate(&plan, &op(), &ones, &[ones.clone(), vec![1]].concat()),
        Err(EvaluatorError::TargetCountsLength {
            expected: nlocal,
            actual: nlocal + 1
        })
    );
    let huge = vec![usize::MAX / 2; nlocal];
    assert_eq!(
        validate(&plan, &op(), &huge, &ones),
        Err(EvaluatorError::Overflow)
    );
    assert_eq!(
        validate(&plan, &op(), &ones, &huge),
        Err(EvaluatorError::Overflow)
    );

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
}
