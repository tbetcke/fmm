//! Pure local tests of the per-pair adapter on hand-made batches. No MPI.
use nd_octree::{MortonKey, morton, octree::KeyType};

use super::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator, PerPair,
    UpwardPass,
};
use crate::v2::index::BoxIndex;
use crate::v2::lists::{Csr, CsrBuilder, GroupedBuilder, GroupedCsr, NOCTANTS, NOFFSETS};
use crate::v2::store::{LeafStore, LevelBuffers, LevelSlice};

/// A three-level index: the root, its eight children (child 0 interior, the others
/// leaves) and the eight children of child 0 (leaves). Local leaves 0..7 are boxes 1..8
/// of level 1, leaves 7..15 the boxes of level 2.
pub(crate) fn hand_index() -> BoxIndex {
    let level1 = morton::children(morton::root()).unwrap().to_vec();
    let level2 = morton::children(level1[0]).unwrap().to_vec();
    let mut kinds1 = vec![KeyType::LocalLeaf; 8];
    kinds1[0] = KeyType::LocalInterior;
    BoxIndex::new(
        vec![vec![morton::root()], level1, level2],
        vec![
            vec![KeyType::LocalInterior],
            kinds1,
            vec![KeyType::LocalLeaf; 8],
        ],
    )
}

/// A CSR view with the given rows.
pub(crate) fn csr(rows: &[&[u32]]) -> Csr {
    let mut builder = CsrBuilder::default();
    for row in rows {
        builder.push_row(row.iter().copied());
    }
    builder.finish().unwrap()
}

/// A grouped view with the given rows of (group, source).
pub(crate) fn grouped<G: Copy + Into<usize>>(
    ngroups: usize,
    rows: &[&[(G, u32)]],
) -> GroupedCsr<G> {
    let mut builder = GroupedBuilder::new(ngroups);
    for row in rows {
        builder.push_row(row.iter().copied());
    }
    builder.finish().unwrap()
}

/// A distinct value for value `k` of chunk `i` of a store tagged `tag`.
pub(crate) fn value(tag: u64, i: usize, k: usize) -> u64 {
    10_000 * tag + 100 * i as u64 + k as u64
}

/// Level buffers with `sizes` on the levels of [`hand_index`], every value distinct.
fn levels(tag: u64, sizes: &[usize]) -> LevelBuffers<u64> {
    let mut buffers = LevelBuffers::new(&[1, 8, 8], sizes);
    for level in 0..3 {
        for (i, chunk) in buffers.level_mut(level).chunks_mut().enumerate() {
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = value(tag + level as u64, i, k);
            }
        }
    }
    buffers
}

/// A leaf store with `counts`, every value distinct.
fn leaves(tag: u64, counts: &[usize], point_size: usize) -> LeafStore<u64> {
    let mut store = LeafStore::new(counts, point_size);
    for j in 0..counts.len() {
        for (k, v) in store.chunk_mut(j).iter_mut().enumerate() {
            *v = value(tag, j, k);
        }
    }
    store
}

/// One pair as the adapter issued it: (method, source key, target key, octant or offset
/// index, source chunk, target input chunk).
type Event = (
    &'static str,
    MortonKey,
    MortonKey,
    Option<usize>,
    Vec<u64>,
    Vec<u64>,
);

/// Records every pair and adds one to every value of its output chunk, so each output
/// chunk ends up holding the number of pairs that wrote it.
#[derive(Default)]
struct Recorder {
    events: Vec<Event>,
}

impl Recorder {
    fn record(
        &mut self,
        method: &'static str,
        (source, target, group): (MortonKey, MortonKey, Option<usize>),
        input: &[u64],
        target_input: &[u64],
        output: &mut [u64],
    ) {
        self.events.push((
            method,
            source,
            target,
            group,
            input.to_vec(),
            target_input.to_vec(),
        ));
        for v in output {
            *v += 1;
        }
    }
}

impl FmmSizes for Recorder {
    type Value = u64;

    fn multipole_size(&self, level: usize) -> usize {
        2 + level
    }

    fn local_size(&self, _level: usize) -> usize {
        3
    }

    fn source_point_size(&self) -> usize {
        2
    }

    fn target_input_point_size(&self) -> usize {
        1
    }

    fn target_output_point_size(&self) -> usize {
        2
    }
}

impl PairOperator for Recorder {
    fn p2m(&mut self, leaf: MortonKey, sources: &[u64], multipole: &mut [u64]) {
        self.record("p2m", (leaf, leaf, None), sources, &[], multipole);
    }

    fn m2m(&mut self, c: MortonKey, p: MortonKey, o: usize, input: &[u64], out: &mut [u64]) {
        self.record("m2m", (c, p, Some(o)), input, &[], out);
    }

    fn m2l(&mut self, s: MortonKey, t: MortonKey, d: usize, input: &[u64], out: &mut [u64]) {
        self.record("m2l", (s, t, Some(d)), input, &[], out);
    }

    fn p2l(&mut self, s: MortonKey, t: MortonKey, input: &[u64], out: &mut [u64]) {
        self.record("p2l", (s, t, None), input, &[], out);
    }

    fn l2l(&mut self, p: MortonKey, c: MortonKey, o: usize, input: &[u64], out: &mut [u64]) {
        self.record("l2l", (p, c, Some(o)), input, &[], out);
    }

    fn l2p(&mut self, leaf: MortonKey, local: &[u64], ti: &[u64], out: &mut [u64]) {
        self.record("l2p", (leaf, leaf, None), local, ti, out);
    }

    fn m2p(&mut self, s: MortonKey, t: MortonKey, input: &[u64], ti: &[u64], out: &mut [u64]) {
        self.record("m2p", (s, t, None), input, ti, out);
    }

    fn p2p(&mut self, s: MortonKey, t: MortonKey, input: &[u64], ti: &[u64], out: &mut [u64]) {
        self.record("p2p", (s, t, None), input, ti, out);
    }
}

/// Check that every chunk of `output` holds its number of pairs, `pairs[i]`, in every
/// value.
fn assert_counted(chunks: impl Iterator<Item = Vec<u64>>, before: &[Vec<u64>], pairs: &[u64]) {
    for (i, (after, before)) in chunks.zip(before).enumerate() {
        let expected: Vec<u64> = before.iter().map(|v| v + pairs[i]).collect();
        assert_eq!(after, expected, "output chunk {i}");
    }
}

#[test]
fn per_pair_sizes_delegate() {
    let op = PerPair(Recorder::default());
    assert_eq!(op.multipole_size(3), 5);
    assert_eq!(op.local_size(1), 3);
    assert_eq!(op.source_point_size(), 2);
    assert_eq!(op.target_input_point_size(), 1);
    assert_eq!(op.target_output_point_size(), 2);
}

#[test]
fn box_operators_issue_every_row_entry_in_target_then_row_order() {
    let index = hand_index();
    let key = |level, i| index.key(level, i);
    // Leaves 7..15 are the boxes of level 2; leaves 0..7 the leaf boxes 1..8 of level 1.
    let counts = [1, 0, 2, 1, 1, 3, 1, 2, 0, 1, 1, 1, 4, 1, 1];
    let sources = leaves(9, &counts, 2);
    let mut op = PerPair(Recorder::default());

    // P2M on level 2: boxes 0, 1 and 4 have their leaves; the others are skipped.
    let mut multipoles = levels(1, &[2, 3, 4]);
    let before: Vec<Vec<u64>> = multipoles.level(2).chunks().map(<[_]>::to_vec).collect();
    let rows = csr(&[&[7], &[8], &[], &[], &[11], &[], &[], &[]]);
    op.p2m(P2m {
        level: 2,
        index: &index,
        leaves: &rows,
        sources: sources.range(0..15),
        multipoles: multipoles.level_mut(2),
    });
    let expected: Vec<Event> = [(0, 7), (1, 8), (4, 11)]
        .map(|(_, j)| {
            let leaf = index.leaf_key(j);
            ("p2m", leaf, leaf, None, sources.chunk(j).to_vec(), vec![])
        })
        .to_vec();
    assert_eq!(op.0.events, expected);
    assert_counted(
        multipoles.level(2).chunks().map(<[_]>::to_vec),
        &before,
        &[1, 1, 0, 0, 1, 0, 0, 0],
    );

    // M2M from level 2 into level 1, rows by octant: only box 0 of level 1 is a parent.
    op.0.events.clear();
    let children = grouped::<u8>(
        NOCTANTS,
        &[&[(0, 0), (3, 3), (7, 7)], &[], &[], &[], &[], &[], &[], &[]],
    );
    let before: Vec<Vec<u64>> = multipoles.level(1).chunks().map(<[_]>::to_vec).collect();
    let (parents, kids) = multipoles.parent_child_mut(1);
    let kid_chunks: Vec<Vec<u64>> = kids.chunks().map(<[_]>::to_vec).collect();
    op.m2m(M2m {
        level: 1,
        index: &index,
        pass: UpwardPass::Local,
        children: &children,
        child_multipoles: kids,
        multipoles: parents,
    });
    let expected: Vec<Event> = [0, 3, 7]
        .map(|c| {
            (
                "m2m",
                key(2, c),
                key(1, 0),
                Some(c),
                kid_chunks[c].clone(),
                vec![],
            )
        })
        .to_vec();
    assert_eq!(op.0.events, expected);
    assert_counted(
        multipoles.level(1).chunks().map(<[_]>::to_vec),
        &before,
        &[3, 0, 0, 0, 0, 0, 0, 0],
    );

    // M2L on level 2: rows by offset index; a target may have several sources.
    op.0.events.clear();
    let mut locals = levels(5, &[3, 3, 3]);
    let pairs = grouped::<u16>(
        NOFFSETS,
        &[
            &[(5, 2), (40, 6)],
            &[],
            &[(0, 1)],
            &[],
            &[],
            &[],
            &[(315, 0)],
            &[],
        ],
    );
    let before: Vec<Vec<u64>> = locals.level(2).chunks().map(<[_]>::to_vec).collect();
    op.m2l(M2l {
        level: 2,
        index: &index,
        pairs: &pairs,
        multipoles: multipoles.level(2),
        locals: locals.level_mut(2),
    });
    let m = |s: usize| multipoles.chunk(2, s).to_vec();
    let expected: Vec<Event> = vec![
        ("m2l", key(2, 2), key(2, 0), Some(5), m(2), vec![]),
        ("m2l", key(2, 6), key(2, 0), Some(40), m(6), vec![]),
        ("m2l", key(2, 1), key(2, 2), Some(0), m(1), vec![]),
        ("m2l", key(2, 0), key(2, 6), Some(315), m(0), vec![]),
    ];
    assert_eq!(op.0.events, expected);
    let counted = [2, 0, 1, 0, 0, 0, 1, 0];
    assert_counted(
        locals.level(2).chunks().map(<[_]>::to_vec),
        &before,
        &counted,
    );

    // P2L on level 2: X-entries are leaves of level 1, by leaf index.
    op.0.events.clear();
    let x = csr(&[&[], &[1, 4], &[], &[], &[], &[], &[], &[0]]);
    op.p2l(P2l {
        level: 2,
        index: &index,
        x: &x,
        sources: sources.range(0..15),
        locals: locals.level_mut(2),
    });
    let s = |j: usize| sources.chunk(j).to_vec();
    let expected: Vec<Event> = vec![
        ("p2l", index.leaf_key(1), key(2, 1), None, s(1), vec![]),
        ("p2l", index.leaf_key(4), key(2, 1), None, s(4), vec![]),
        ("p2l", index.leaf_key(0), key(2, 7), None, s(0), vec![]),
    ];
    assert_eq!(op.0.events, expected);
    let counted: Vec<u64> = [2, 2, 1, 0, 0, 0, 1, 1].to_vec();
    assert_counted(
        locals.level(2).chunks().map(<[_]>::to_vec),
        &before,
        &counted,
    );

    // L2L into level 2 from level 1: one parent per row, with the child's octant.
    op.0.events.clear();
    let rows: Vec<[(u8, u32); 1]> = (0..8).map(|o| [(o, 0)]).collect();
    let rows: Vec<&[(u8, u32)]> = rows.iter().map(|row| &row[..]).collect();
    let parents = grouped::<u8>(NOCTANTS, &rows);
    let before: Vec<Vec<u64>> = locals.level(2).chunks().map(<[_]>::to_vec).collect();
    let (children_locals, parent_locals) = locals.child_parent_mut(1);
    let parent_chunk = parent_locals.chunk(0).to_vec();
    op.l2l(L2l {
        level: 2,
        index: &index,
        parents: &parents,
        parent_locals,
        locals: children_locals,
    });
    let expected: Vec<Event> = (0..8)
        .map(|c| {
            (
                "l2l",
                key(1, 0),
                key(2, c),
                Some(c),
                parent_chunk.clone(),
                vec![],
            )
        })
        .collect();
    assert_eq!(op.0.events, expected);
    assert_counted(
        locals.level(2).chunks().map(<[_]>::to_vec),
        &before,
        &[1; 8],
    );
}

#[test]
fn leaf_operators_issue_every_row_entry_with_the_rows_leaf_chunks() {
    let index = hand_index();
    // Leaves 7..15 are the local leaves of level 2; rows r = 0..8 are leaves 7 + r.
    let source_counts = [1, 0, 2, 1, 1, 3, 1, 2, 0, 1, 1, 1, 4, 1, 1];
    let target_counts = [2, 1, 0, 1, 3, 1, 1, 1, 2, 0, 1, 1, 2, 1, 1];
    let sources = leaves(9, &source_counts, 2);
    let target_input = leaves(3, &target_counts, 1);
    let mut target_output = leaves(4, &target_counts, 2);
    let multipoles = levels(1, &[2, 3, 4]);
    let locals = levels(5, &[3, 3, 3]);
    let level_leaves = index.leaves().local(2);
    assert_eq!(level_leaves, 7..15);
    let snapshot = |store: &LeafStore<u64>| -> Vec<Vec<u64>> {
        level_leaves
            .clone()
            .map(|j| store.chunk(j).to_vec())
            .collect()
    };
    let ti = |r: usize| target_input.chunk(7 + r).to_vec();
    let mut op = PerPair(Recorder::default());

    // L2P: every row holds the box of its leaf.
    let before = snapshot(&target_output);
    let rows: Vec<[u32; 1]> = (0..8).map(|i| [i]).collect();
    let rows: Vec<&[u32]> = rows.iter().map(|row| &row[..]).collect();
    let boxes = csr(&rows);
    op.l2p(L2p {
        level: 2,
        index: &index,
        leaves: level_leaves.clone(),
        boxes: &boxes,
        locals: locals.level(2),
        target_input: target_input.range(level_leaves.clone()),
        target_output: target_output.range_mut(level_leaves.clone()),
    });
    let expected: Vec<Event> = (0..8)
        .map(|r| {
            let leaf = index.leaf_key(7 + r);
            ("l2p", leaf, leaf, None, locals.chunk(2, r).to_vec(), ti(r))
        })
        .collect();
    assert_eq!(op.0.events, expected);
    assert_counted(snapshot(&target_output).into_iter(), &before, &[1; 8]);

    // M2P on the deepest level: the rows are empty and the multipoles of the level
    // below are an empty slice.
    op.0.events.clear();
    let before = snapshot(&target_output);
    let empty = csr(&[&[][..]; 8]);
    op.m2p(M2p {
        level: 2,
        index: &index,
        leaves: level_leaves.clone(),
        w: &empty,
        multipoles: LevelSlice::empty(4),
        target_input: target_input.range(level_leaves.clone()),
        target_output: target_output.range_mut(level_leaves.clone()),
    });
    assert!(op.0.events.is_empty());
    assert_counted(snapshot(&target_output).into_iter(), &before, &[0; 8]);

    // M2P on level 1: leaves 0..7 (boxes 1..8) with W-entries among the level-2 boxes.
    let level1 = index.leaves().local(1);
    assert_eq!(level1, 0..7);
    let before: Vec<Vec<u64>> = level1
        .clone()
        .map(|j| target_output.chunk(j).to_vec())
        .collect();
    let w = csr(&[&[0, 2], &[], &[], &[5], &[], &[], &[1]]);
    op.m2p(M2p {
        level: 1,
        index: &index,
        leaves: level1.clone(),
        w: &w,
        multipoles: multipoles.level(2),
        target_input: target_input.range(level1.clone()),
        target_output: target_output.range_mut(level1.clone()),
    });
    let m = |s: usize| multipoles.chunk(2, s).to_vec();
    let t1 = |r: usize| target_input.chunk(r).to_vec();
    let expected: Vec<Event> = vec![
        ("m2p", index.key(2, 0), index.leaf_key(0), None, m(0), t1(0)),
        ("m2p", index.key(2, 2), index.leaf_key(0), None, m(2), t1(0)),
        ("m2p", index.key(2, 5), index.leaf_key(3), None, m(5), t1(3)),
        ("m2p", index.key(2, 1), index.leaf_key(6), None, m(1), t1(6)),
    ];
    assert_eq!(op.0.events, expected);
    let after: Vec<Vec<u64>> = level1
        .clone()
        .map(|j| target_output.chunk(j).to_vec())
        .collect();
    assert_counted(after.into_iter(), &before, &[2, 0, 0, 1, 0, 0, 1]);

    // P2P on level 2: the near list holds the leaf itself and its neighbours, by leaf
    // index; sources come from every leaf, targets from the row's leaf.
    op.0.events.clear();
    let before = snapshot(&target_output);
    let near = csr(&[&[3, 7, 8], &[8], &[9], &[10], &[11], &[2, 12], &[13], &[14]]);
    op.p2p(P2p {
        level: 2,
        index: &index,
        leaves: level_leaves.clone(),
        near: &near,
        sources: sources.range(0..15),
        target_input: target_input.range(level_leaves.clone()),
        target_output: target_output.range_mut(level_leaves.clone()),
    });
    let mut expected: Vec<Event> = Vec::new();
    for r in 0..8 {
        for &j in near.row(r) {
            let j = j as usize;
            expected.push((
                "p2p",
                index.leaf_key(j),
                index.leaf_key(7 + r),
                None,
                sources.chunk(j).to_vec(),
                ti(r),
            ));
        }
    }
    assert_eq!(expected.len(), 11);
    assert_eq!(expected[0].1, index.leaf_key(3));
    assert_eq!(expected[1].1, expected[1].2, "the self pair");
    assert_eq!(op.0.events, expected);
    assert_counted(
        snapshot(&target_output).into_iter(),
        &before,
        &[3, 1, 1, 1, 1, 2, 1, 1],
    );
}
