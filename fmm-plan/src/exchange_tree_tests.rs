//! Serial tests of the forwarding trees' rule: `split` and `level_trees` (Phase 5S T8s).

use std::collections::{HashMap, HashSet};

use super::{Edge, level_trees, split};

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

#[test]
fn split_on_one_node() {
    let node = |_: usize| 0;
    let list: Vec<usize> = (1..=20).collect();
    let parts = split(&list, 8, 0, &node);
    assert_eq!(parts.len(), 8);
    assert_eq!(parts.concat(), list);
    let sizes: Vec<usize> = parts.iter().map(Vec::len).collect();
    assert!(sizes.iter().all(|&n| n == 2 || n == 3), "{sizes:?}");
    // Fewer readers than the fan-out: one each.
    assert_eq!(split(&list[..3], 8, 0, &node).len(), 3);
    // Another node's readers are one part: that node's head splits them.
    assert_eq!(split(&list, 8, 1, &node), vec![list.clone()]);
}

#[test]
fn split_by_nodes() {
    // Ten ranks per node; readers on three nodes, the sender on a fourth.
    let node = |r: usize| r / 10;
    let list: Vec<usize> = (0..10).chain(10..12).chain(20..25).collect();
    let parts = split(&list, 8, 3, &node);
    assert!(parts.len() <= 8);
    assert_eq!(parts.concat(), list);
    for part in &parts {
        assert!(part.iter().all(|&r| node(r) == node(part[0])), "{parts:?}");
    }
    // Every node gets a part; the node with most readers gets the most.
    let per_node = |n: usize| parts.iter().filter(|p| node(p[0]) == n).count();
    assert!(per_node(0) >= per_node(2) && per_node(2) >= per_node(1) && per_node(1) >= 1);

    // More nodes than the fan-out: runs of whole nodes.
    let list: Vec<usize> = (0..120).step_by(3).collect();
    let parts = split(&list, 4, 99, &node);
    assert_eq!(parts.len(), 4);
    assert_eq!(parts.concat(), list);
    for pair in parts.windows(2) {
        assert_ne!(
            node(*pair[0].last().unwrap()),
            node(pair[1][0]),
            "a node is cut: {parts:?}"
        );
    }
}

/// Check the properties of `edges`, the tree of one box of `owner` to `readers`.
fn check_tree(
    owner: usize,
    readers: &[usize],
    edges: &[Edge],
    fan_out: usize,
    node: &dyn Fn(usize) -> usize,
) {
    let mut received: HashMap<usize, usize> = HashMap::new();
    for &(_, receiver, stage) in edges {
        assert!(
            received.insert(receiver, stage).is_none(),
            "{receiver} receives twice"
        );
    }
    let expected: HashSet<usize> = readers.iter().copied().collect();
    let got: HashSet<usize> = received.keys().copied().collect();
    assert_eq!(
        got, expected,
        "every reader, and only readers, receive the box"
    );
    let mut copies: HashMap<usize, usize> = HashMap::new();
    for &(sender, _, stage) in edges {
        *copies.entry(sender).or_default() += 1;
        if sender == owner {
            assert_eq!(stage, 0, "the owner sends at stage 0");
        } else {
            assert_eq!(
                received.get(&sender).map(|s| s + 1),
                Some(stage),
                "a forwarder sends one stage after it received"
            );
        }
    }
    assert!(copies.values().all(|&c| c <= fan_out), "more than F copies");
    let depth = 1 + edges.iter().map(|e| e.2).max().unwrap();
    // The rule cuts no node but its own, so the depth is bounded by the levels of a tree
    // over the nodes plus those of a tree within one node.
    let mut per_node: HashMap<usize, usize> = HashMap::new();
    for &r in readers {
        *per_node.entry(node(r)).or_default() += 1;
    }
    let levels = |n: usize| {
        let (mut k, mut reach) = (0, 1);
        while reach < n {
            reach *= fan_out;
            k += 1;
        }
        k
    };
    let bound = levels(per_node.len()) + levels(*per_node.values().max().unwrap()) + 2;
    assert!(
        depth <= bound,
        "depth {depth} for {} readers on {} nodes, F = {fan_out}",
        readers.len(),
        per_node.len()
    );
}

#[test]
fn trees_on_random_readers() {
    let mut rng = Rng(7);
    for case in 0..200 {
        let nranks = 2 + rng.below(300);
        let per_node = 1 + rng.below(50);
        let fan_out = 2 + rng.below(9);
        let owner = rng.below(nranks);
        let node = move |r: usize| r / per_node;
        let nboxes = 1 + rng.below(12);
        let readers: Vec<Vec<usize>> = (0..nboxes)
            .map(|_| {
                let mut set: Vec<usize> = (0..nranks).filter(|&r| r != owner).collect();
                let keep = fan_out + 1 + rng.below(set.len().max(fan_out + 1) - fan_out);
                // A random subset of `keep` readers (or every rank).
                for i in 0..set.len() {
                    let j = i + rng.below(set.len() - i);
                    set.swap(i, j);
                }
                set.truncate(keep.min(set.len()));
                set
            })
            .filter(|set| set.len() > fan_out)
            .collect();
        let trees = level_trees(owner, &node, &readers, fan_out);
        assert_eq!(trees.len(), readers.len(), "case {case}");
        for (box_readers, edges) in readers.iter().zip(&trees) {
            check_tree(owner, box_readers, edges, fan_out, &node);
        }
        // Deterministic: the same input gives the same trees.
        assert_eq!(level_trees(owner, &node, &readers, fan_out), trees);
    }
}

#[test]
fn trees_share_first_hops() {
    // Four boxes read by the same 30 ranks on one node: one set of heads for all.
    let node = |_: usize| 0;
    let readers: Vec<Vec<usize>> = (0..4)
        .map(|b| (1..=30).rev().skip(b % 2).collect())
        .collect();
    let trees = level_trees(0, &node, &readers, 8);
    let heads = |edges: &[Edge]| -> HashSet<usize> {
        edges.iter().filter(|e| e.0 == 0).map(|e| e.1).collect()
    };
    let all: HashSet<usize> = trees.iter().flat_map(|t| heads(t)).collect();
    assert!(all.len() <= 9, "the owner's first hops: {all:?}");
}

#[test]
fn heads_on_other_nodes() {
    // The owner on node 0, readers on nodes 1 and 2: each node's copies enter through
    // heads on that node, and within a node the copies stay on it.
    let node = |r: usize| r / 20;
    let readers = vec![(20..60).collect::<Vec<usize>>()];
    let trees = level_trees(0, &node, &readers, 4);
    check_tree(0, &readers[0], &trees[0], 4, &node);
    for &(sender, receiver, _) in &trees[0] {
        if sender != 0 {
            assert_eq!(node(sender), node(receiver), "a forward crosses nodes");
        }
    }
}
