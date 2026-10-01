//! Pure local tests of the index operators on hand-made chunks with several points per
//! leaf, through the per-pair adapter and both batched walks. No MPI.
use super::{BatchedIndexFmm, IndexFmm, Walk, check_counts};
use crate::v2::index::BoxIndex;
use crate::v2::lists::{NOCTANTS, NOFFSETS};
use crate::v2::operator::tests::{csr, grouped, hand_index};
use crate::v2::operator::{
    FmmOperator, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PerPair, UpwardPass,
};
use crate::v2::store::{LeafStore, LevelBuffers};

/// The leaves of [`hand_index`]; leaf j of these tests carries the global index j.
const NLEAVES: usize = 15;

/// Source points per leaf: several, one and none.
const SOURCE_COUNTS: [usize; NLEAVES] = [3, 0, 2, 1, 1, 4, 1, 2, 0, 1, 3, 1, 2, 1, 1];

/// Target points per leaf.
const TARGET_COUNTS: [usize; NLEAVES] = [2, 1, 0, 1, 3, 1, 1, 2, 2, 0, 1, 1, 3, 1, 1];

/// The three implementations of the index FMM.
fn operators() -> Vec<(&'static str, Box<dyn FmmOperator<Value = u32>>)> {
    vec![
        ("per pair", Box::new(PerPair(IndexFmm::new(NLEAVES)))),
        ("rows", Box::new(BatchedIndexFmm::new(NLEAVES, Walk::Rows))),
        (
            "groupings",
            Box::new(BatchedIndexFmm::new(NLEAVES, Walk::Groupings)),
        ),
    ]
}

/// A source store in which every point of leaf j carries j.
fn sources() -> LeafStore<u32> {
    let mut store = LeafStore::new(&SOURCE_COUNTS, 1);
    for j in 0..NLEAVES {
        store.chunk_mut(j).fill(j as u32);
    }
    store
}

/// Level buffers of [`hand_index`] with `NLEAVES` distinct values per box.
fn coefficients(tag: u32) -> LevelBuffers<u32> {
    let mut buffers = LevelBuffers::new(&[1, 8, 8], &[NLEAVES; 3]);
    for level in 0..3 {
        for (i, chunk) in buffers.level_mut(level).chunks_mut().enumerate() {
            for (k, v) in chunk.iter_mut().enumerate() {
                *v = tag + (100 * level + 10 * i + k) as u32;
            }
        }
    }
    buffers
}

/// `count` copies of the vector `point`, one per target point.
fn at_points(count: usize, point: &[u32]) -> Vec<u32> {
    point.repeat(count)
}

/// The vector with `SOURCE_COUNTS[j]` at every index j of `leaves` (an index FMM sum over
/// the source points of those leaves).
fn indices(leaves: &[u32]) -> Vec<u32> {
    let mut vector = vec![0u32; NLEAVES];
    for &j in leaves {
        vector[j as usize] += SOURCE_COUNTS[j as usize] as u32;
    }
    vector
}

/// The value-by-value sum of the chunks `chunks` of `level` of `buffers`.
fn sum(buffers: &LevelBuffers<u32>, level: usize, chunks: &[u32]) -> Vec<u32> {
    let mut vector = vec![0u32; NLEAVES];
    for &i in chunks {
        for (v, &value) in vector.iter_mut().zip(buffers.chunk(level, i as usize)) {
            *v += value;
        }
    }
    vector
}

#[test]
fn sizes_follow_the_number_of_leaves() {
    for (name, op) in operators() {
        assert_eq!(op.multipole_size(3), NLEAVES, "{name}");
        assert_eq!(op.local_size(0), NLEAVES, "{name}");
        assert_eq!(op.source_point_size(), 1, "{name}");
        assert_eq!(op.target_input_point_size(), 0, "{name}");
        assert_eq!(op.target_output_point_size(), NLEAVES, "{name}");
    }
    assert_eq!(IndexFmm::new(4).nleaves(), 4);
    let batched = BatchedIndexFmm::new(4, Walk::Groupings);
    assert_eq!((batched.nleaves(), batched.walk()), (4, Walk::Groupings));
}

#[test]
#[should_panic(expected = "at least one leaf")]
fn an_empty_tree_is_rejected() {
    IndexFmm::new(0);
}

#[test]
#[should_panic(expected = "at least one leaf")]
fn an_empty_tree_is_rejected_by_the_batched_variant() {
    BatchedIndexFmm::new(0, Walk::Rows);
}

#[test]
fn particle_operators_add_one_per_source_point_at_its_leaf() {
    let index: BoxIndex = hand_index();
    let sources = sources();
    // P2M on level 2 (box t is leaf 7 + t), P2L into level 2 from level-1 leaves.
    let p2m_rows = csr(&[&[7], &[8], &[9], &[10], &[11], &[12], &[13], &[14]]);
    let x_rows: [&[u32]; 8] = [&[0, 2, 5], &[], &[3], &[], &[], &[], &[1, 4], &[]];
    let x = csr(&x_rows);
    for (name, mut op) in operators() {
        let mut multipoles = LevelBuffers::<u32>::new(&[1, 8, 8], &[NLEAVES; 3]);
        op.p2m(P2m {
            level: 2,
            index: &index,
            leaves: &p2m_rows,
            sources: sources.range(0..NLEAVES),
            multipoles: multipoles.level_mut(2),
        });
        for t in 0..8u32 {
            assert_eq!(
                multipoles.chunk(2, t as usize),
                indices(&[7 + t]),
                "{name}: p2m"
            );
        }

        let mut locals = coefficients(0);
        let before = locals.clone();
        op.p2l(P2l {
            level: 2,
            index: &index,
            x: &x,
            sources: sources.range(0..NLEAVES),
            locals: locals.level_mut(2),
        });
        for (t, row) in x_rows.iter().enumerate() {
            let expected: Vec<u32> = before
                .chunk(2, t)
                .iter()
                .zip(indices(row))
                .map(|(a, b)| a + b)
                .collect();
            assert_eq!(locals.chunk(2, t), expected, "{name}: p2l of box {t}");
        }
    }
}

#[test]
fn translations_add_their_inputs_in_row_and_in_batch_order() {
    let index = hand_index();
    // M2M into box 0 of level 1 from all eight children; M2L on level 2 with several
    // sources per target and offsets out of source order; L2L into level 2.
    let children_rows: Vec<[(u8, u32); 8]> = vec![std::array::from_fn(|o| (o as u8, o as u32))];
    let mut m2m_rows: Vec<&[(u8, u32)]> = vec![&children_rows[0]];
    m2m_rows.extend([&[][..]; 7]);
    let children = grouped::<u8>(NOCTANTS, &m2m_rows);
    let v_rows: [&[(u16, u32)]; 8] = [
        &[(3, 5), (17, 2), (300, 7)],
        &[],
        &[(17, 6)],
        &[(0, 1), (3, 0)],
        &[],
        &[],
        &[(315, 4)],
        &[(3, 3)],
    ];
    let pairs = grouped::<u16>(NOFFSETS, &v_rows);
    let parent_rows: Vec<[(u8, u32); 1]> = (0..8).map(|o| [(o, 0)]).collect();
    let l2l_rows: Vec<&[(u8, u32)]> = parent_rows.iter().map(|row| &row[..]).collect();
    let parents = grouped::<u8>(NOCTANTS, &l2l_rows);

    for (name, mut op) in operators() {
        let mut multipoles = coefficients(1);
        let before = multipoles.clone();
        let (parent_level, child_level) = multipoles.parent_child_mut(1);
        op.m2m(M2m {
            level: 1,
            index: &index,
            pass: UpwardPass::Local,
            children: &children,
            child_multipoles: child_level,
            multipoles: parent_level,
        });
        let expected: Vec<u32> = before
            .chunk(1, 0)
            .iter()
            .zip(sum(&before, 2, &[0, 1, 2, 3, 4, 5, 6, 7]))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(multipoles.chunk(1, 0), expected, "{name}: m2m");
        for t in 1..8 {
            assert_eq!(multipoles.chunk(1, t), before.chunk(1, t), "{name}: m2m");
        }

        let mut locals = coefficients(7);
        let before_locals = locals.clone();
        op.m2l(M2l {
            level: 2,
            index: &index,
            pairs: &pairs,
            multipoles: multipoles.level(2),
            locals: locals.level_mut(2),
        });
        for (t, row) in v_rows.iter().enumerate() {
            let sources: Vec<u32> = row.iter().map(|&(_, s)| s).collect();
            let expected: Vec<u32> = before_locals
                .chunk(2, t)
                .iter()
                .zip(sum(&multipoles, 2, &sources))
                .map(|(a, b)| a + b)
                .collect();
            assert_eq!(locals.chunk(2, t), expected, "{name}: m2l of box {t}");
        }

        let after_m2l = locals.clone();
        let (child_locals, parent_locals) = locals.child_parent_mut(1);
        op.l2l(L2l {
            level: 2,
            index: &index,
            parents: &parents,
            parent_locals,
            locals: child_locals,
        });
        for t in 0..8 {
            let expected: Vec<u32> = after_m2l
                .chunk(2, t)
                .iter()
                .zip(after_m2l.chunk(1, 0))
                .map(|(a, b)| a + b)
                .collect();
            assert_eq!(locals.chunk(2, t), expected, "{name}: l2l of box {t}");
        }
    }
}

#[test]
fn leaf_operators_act_on_every_target_point() {
    let index = hand_index();
    let sources = sources();
    let multipoles = coefficients(1);
    let locals = coefficients(5);
    let target_input = LeafStore::<u32>::new(&TARGET_COUNTS, 0);
    let level1 = index.leaves().local(1);
    let level2 = index.leaves().local(2);
    let w_rows: [&[u32]; 7] = [&[0, 3], &[], &[7], &[], &[1, 2, 6], &[], &[5]];
    let w = csr(&w_rows);
    let near_rows: [&[u32]; 8] = [
        &[0, 7, 8],
        &[8, 9],
        &[9],
        &[2, 10, 14],
        &[11],
        &[12],
        &[1, 13],
        &[14],
    ];
    let near = csr(&near_rows);
    let boxes = csr(&[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7]]);

    for (name, mut op) in operators() {
        let mut target_output = LeafStore::<u32>::new(&TARGET_COUNTS, NLEAVES);
        op.l2p(L2p {
            level: 2,
            index: &index,
            leaves: level2.clone(),
            boxes: &boxes,
            locals: locals.level(2),
            target_input: target_input.range(level2.clone()),
            target_output: target_output.range_mut(level2.clone()),
        });
        op.p2p(P2p {
            level: 2,
            index: &index,
            leaves: level2.clone(),
            near: &near,
            sources: sources.range(0..NLEAVES),
            target_input: target_input.range(level2.clone()),
            target_output: target_output.range_mut(level2.clone()),
        });
        for (r, row) in near_rows.iter().enumerate() {
            let point: Vec<u32> = locals
                .chunk(2, r)
                .iter()
                .zip(indices(row))
                .map(|(a, b)| a + b)
                .collect();
            let j = level2.start + r;
            assert_eq!(
                target_output.chunk(j),
                at_points(TARGET_COUNTS[j], &point),
                "{name}: l2p and p2p of leaf {j}"
            );
        }

        op.m2p(M2p {
            level: 1,
            index: &index,
            leaves: level1.clone(),
            w: &w,
            multipoles: multipoles.level(2),
            target_input: target_input.range(level1.clone()),
            target_output: target_output.range_mut(level1.clone()),
        });
        for (r, row) in w_rows.iter().enumerate() {
            assert_eq!(
                target_output.chunk(r),
                at_points(TARGET_COUNTS[r], &sum(&multipoles, 2, row)),
                "{name}: m2p of leaf {r}"
            );
        }
    }
}

#[test]
fn the_count_check_finds_missing_and_doubled_interactions() {
    let index = hand_index();
    let counts = [2usize, 0, 1];
    let global = [3usize, 0, 1];
    let mut targets = LeafStore::<u32>::new(&counts, 3);
    for j in 0..3 {
        targets.chunk_mut(j).as_chunks_mut::<3>().0.fill([3, 0, 1]);
    }
    assert_eq!(check_counts(&index, &targets, &global), Ok(()));

    targets.chunk_mut(0)[5] = 2;
    let message = check_counts(&index, &targets, &global).unwrap_err();
    assert!(message.contains("point 1"), "{message}");
    assert!(message.contains("(2, 2, 1)"), "{message}");

    targets.chunk_mut(0)[5] = 1;
    targets.chunk_mut(2)[0] = 6;
    let message = check_counts(&index, &targets, &global).unwrap_err();
    assert!(message.contains("(0, 6, 3)"), "{message}");
}
