//! Per-level counts of a tree and its plan: boxes, leaves, and the U, V, W and X lists,
//! each as the plan holds it (empty boxes included) and over occupied boxes only (the
//! work of an FMM that skips empty boxes).

use std::collections::BTreeMap;

use mpi::traits::CommunicatorCollectives;
use nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS;
use nd_fmm_plan::plan::Plan;
use nd_octree::{MortonKey, Octree, morton};

/// The squared distances |t|²/w² of the V-list offsets, ascending: 15 distance classes.
pub const DISTANCES: [i64; 15] = [4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 27];

/// The number of distance classes.
pub const NDIST: usize = DISTANCES.len();

/// The number of reflection classes of the V-list offsets.
pub const NCLASSES: usize = 34;

/// The distance class of a V-list offset: the position of |d|² in [`DISTANCES`].
pub fn distance_class(d: [i64; 3]) -> usize {
    let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    DISTANCES
        .iter()
        .position(|&x| x == d2)
        .expect("a V-list offset")
}

/// The representative of an offset's class under x → −x, y → −y, z → −z and x ↔ y, the
/// symmetries of the signature grid (proposal §5.4): (min(|x|, |y|), max(|x|, |y|), |z|).
pub fn reflection_representative(d: [i64; 3]) -> [i64; 3] {
    let (a, b) = (d[0].abs(), d[1].abs());
    [a.min(b), a.max(b), d[2].abs()]
}

/// The reflection class representatives of the 316 offsets, ascending.
pub fn reflection_classes() -> Vec<[i64; 3]> {
    let mut classes: Vec<[i64; 3]> = V_LIST_DIRECTIONS
        .iter()
        .map(|&d| reflection_representative(d))
        .collect();
    classes.sort_unstable();
    classes.dedup();
    classes
}

/// The counts of one level.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LevelCensus {
    /// The level.
    pub level: usize,
    /// Boxes the plan holds on the level.
    pub boxes: u64,
    /// Boxes holding at least one point.
    pub occupied: u64,
    /// Leaves on the level.
    pub leaves: u64,
    /// Leaves holding at least one point.
    pub occupied_leaves: u64,
    /// Points in the leaves of the level.
    pub leaf_points: u64,
    /// The most points in one leaf of the level.
    pub max_leaf_points: u64,
    /// U-list entries of the level's leaves (the near list without the leaf itself).
    pub u_entries: u64,
    /// U-list entries between occupied leaves.
    pub u_entries_occupied: u64,
    /// Point pairs of P2P: Σ over the level's leaves of n_t Σ_{s ∈ near(t)} n_s, the leaf
    /// itself included.
    pub p2p_pairs: u64,
    /// V-list entries of the level's boxes.
    pub v_entries: u64,
    /// V pairs between occupied boxes, per distance class.
    pub v_by_distance: [u64; NDIST],
    /// V pairs between occupied boxes, per reflection class (in [`reflection_classes`]
    /// order).
    pub v_by_class: Vec<u64>,
    /// For the occupied V sources: the mask of distance classes each is a source of, and
    /// how many boxes have that mask.
    pub source_masks: BTreeMap<u16, u64>,
    /// The same for the occupied V targets.
    pub target_masks: BTreeMap<u16, u64>,
    /// W-list entries of the level's leaves (boxes on the next level).
    pub w_entries: u64,
    /// W entries with an occupied leaf and an occupied box.
    pub w_entries_occupied: u64,
    /// Σ n_t over the occupied W entries: M2P evaluations of a level + 1 multipole.
    pub w_points: u64,
    /// X-list entries of the level's boxes (leaves on the previous level).
    pub x_entries: u64,
    /// X entries with an occupied box and an occupied leaf.
    pub x_entries_occupied: u64,
    /// Σ n_s over the occupied X entries: P2L evaluations into a level-l local.
    pub x_points: u64,
}

impl LevelCensus {
    /// V pairs between occupied boxes.
    pub fn v_pairs(&self) -> u64 {
        self.v_by_distance.iter().sum()
    }

    /// Occupied boxes that are the source (target) of at least one V pair in a class of
    /// `mask`.
    pub fn sources_in(&self, mask: u16) -> u64 {
        self.source_masks
            .iter()
            .filter(|(m, _)| *m & mask != 0)
            .map(|(_, n)| n)
            .sum()
    }

    /// Occupied boxes that are the target of at least one V pair in a class of `mask`.
    pub fn targets_in(&self, mask: u16) -> u64 {
        self.target_masks
            .iter()
            .filter(|(m, _)| *m & mask != 0)
            .map(|(_, n)| n)
            .sum()
    }
}

/// The census of a tree: every level, root first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Census {
    /// The number of points.
    pub npoints: u64,
    /// The per-level counts.
    pub levels: Vec<LevelCensus>,
}

impl Census {
    /// Counts the plan of `octree` with the points of `keys` (level-16 keys, one per
    /// point). One rank.
    pub fn new<C: CommunicatorCollectives>(octree: &Octree<'_, C>, keys: &[MortonKey]) -> Self {
        let plan = Plan::new(octree).expect("a plan of the tree");
        Self::of_plan(&plan, keys)
    }

    /// Counts `plan` with the points of `keys`.
    pub fn of_plan(plan: &Plan, keys: &[MortonKey]) -> Self {
        let index = plan.index();
        let nlevels = plan.nlevels();
        let leaves = index.leaves();

        // Points per leaf, then per box.
        let mut leaf_count = vec![0u64; leaves.len()];
        for &key in keys {
            let j = index
                .local_leaf_containing(key)
                .expect("every point in a leaf");
            leaf_count[j as usize] += 1;
        }
        let mut counts: Vec<Vec<u64>> = (0..nlevels).map(|l| vec![0; index.len(l)]).collect();
        for (j, &c) in leaf_count.iter().enumerate().take(leaves.nlocal()) {
            if c == 0 {
                continue;
            }
            let key = index.leaf_key(j);
            for (l, level) in counts.iter_mut().enumerate().take(morton::level(key) + 1) {
                let (_, i) = index
                    .find(morton::ancestor_at_level(key, l).unwrap())
                    .unwrap();
                level[i as usize] += c;
            }
        }

        let dist: Vec<usize> = V_LIST_DIRECTIONS
            .iter()
            .map(|&d| distance_class(d))
            .collect();
        let classes = reflection_classes();
        let class: Vec<usize> = V_LIST_DIRECTIONS
            .iter()
            .map(|&d| {
                classes
                    .binary_search(&reflection_representative(d))
                    .unwrap()
            })
            .collect();

        let mut levels = Vec::with_capacity(nlevels);
        for l in 0..nlevels {
            let lists = plan.level(l);
            let n = &counts[l];
            let mut c = LevelCensus {
                level: l,
                boxes: n.len() as u64,
                occupied: n.iter().filter(|&&x| x > 0).count() as u64,
                v_by_class: vec![0; NCLASSES],
                ..Default::default()
            };

            // Leaves, U and P2P.
            let local = leaves.local(l);
            for (r, j) in local.clone().enumerate() {
                let nt = leaf_count[j];
                c.leaves += 1;
                if nt > 0 {
                    c.occupied_leaves += 1;
                }
                c.leaf_points += nt;
                c.max_leaf_points = c.max_leaf_points.max(nt);
                let row = lists.near().row(r);
                c.u_entries += row.len() as u64 - 1;
                let mut sources = 0;
                for &s in row {
                    let ns = leaf_count[s as usize];
                    sources += ns;
                    if s as usize != j && nt > 0 && ns > 0 {
                        c.u_entries_occupied += 1;
                    }
                }
                c.p2p_pairs += nt * sources;

                // W: boxes on l + 1.
                let row = lists.w().row(r);
                c.w_entries += row.len() as u64;
                if nt > 0 {
                    for &s in row {
                        if counts[l + 1][s as usize] > 0 {
                            c.w_entries_occupied += 1;
                            c.w_points += nt;
                        }
                    }
                }
            }

            // V and X: rows per box.
            let v = lists.v();
            let mut source_mask = vec![0u16; n.len()];
            for t in 0..v.nrows() {
                let (sources, groups) = v.row(t);
                c.v_entries += sources.len() as u64;
                let row = lists.x().row(t);
                c.x_entries += row.len() as u64;
                if n[t] == 0 {
                    continue;
                }
                let mut target_mask = 0u16;
                for (&s, &g) in sources.iter().zip(groups) {
                    if n[s as usize] == 0 {
                        continue;
                    }
                    let d = dist[g as usize];
                    c.v_by_distance[d] += 1;
                    c.v_by_class[class[g as usize]] += 1;
                    target_mask |= 1 << d;
                    source_mask[s as usize] |= 1 << d;
                }
                if target_mask != 0 {
                    *c.target_masks.entry(target_mask).or_default() += 1;
                }
                for &s in row {
                    let ns = leaf_count[s as usize];
                    if ns > 0 {
                        c.x_entries_occupied += 1;
                        c.x_points += ns;
                    }
                }
            }
            for &m in source_mask.iter().filter(|&&m| m != 0) {
                *c.source_masks.entry(m).or_default() += 1;
            }
            levels.push(c);
        }
        Census {
            npoints: keys.len() as u64,
            levels,
        }
    }

    /// The deepest level with an occupied leaf.
    pub fn deepest_leaf_level(&self) -> usize {
        self.levels
            .iter()
            .rposition(|c| c.occupied_leaves > 0)
            .unwrap_or(0)
    }

    /// The shallowest level with an occupied leaf.
    pub fn shallowest_leaf_level(&self) -> usize {
        self.levels
            .iter()
            .position(|c| c.occupied_leaves > 0)
            .unwrap_or(0)
    }
}
