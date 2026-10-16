//! The modelled cost of an evaluation, for the work-weighted cut (Phase 5S T7, S3;
//! docs/design/scale-out.md §6).
//!
//! On several ranks the octree first cuts its coarse blocks among the ranks by points
//! (O3). The work of an evaluation follows the interaction lists rather than the points,
//! so on adaptive trees the ranks' compute grows apart with P (scale-out.md §6.2: by
//! points the busiest rank's modelled cost reaches 1.19–2.69 times the mean at 512
//! ranks). [`FmmBuilder::build`](crate::fmm::FmmBuilder::build) therefore cuts again
//! ([`FmmBuilder::work_cut`](crate::fmm::FmmBuilder::work_cut)): it counts, from the
//! plan of the first cut, the work of every own block by operator kind
//! ([`work_counts`], [`KindCounts`]), weighs the counts with per-kind constants
//! ([`CostModel`]) and hands one weight per block to `nd_octree::Octree::repartition`,
//! which knows nothing of the kernel. The new cut is taken where it lowers the busiest
//! rank's block cost by at least [`WORK_CUT_MIN_GAIN`](crate::fmm::WORK_CUT_MIN_GAIN).
//!
//! # The model
//!
//! The compute of an evaluation on a rank is modelled as Σ_k α_k c_k over the eight
//! operator kinds, with the counts c_k of the plan's views (one rank's level calls) and
//! constants α_k in nanoseconds per counted unit (scale-out.md §6.1):
//!
//! | Kind | Count c_k |
//! | --- | --- |
//! | P2M | source points of the local leaves |
//! | M2M | parent–child pairs, both passes |
//! | M2L | V pairs |
//! | P2L | X pairs times the source leaf's points |
//! | L2L | child–parent pairs |
//! | L2P | target points of the local leaves |
//! | M2P | W pairs times the target leaf's points |
//! | P2P | near interactions: Σ over the local leaves of n_t times Σ over the near list of n_s |
//!
//! A block's cost is that of the rows of the boxes and leaves at or below it. The rows of
//! the `Global` boxes (the shared ancestors: their global M2M on the owner, their M2L,
//! P2L and L2L on every rank that shares them) belong to no block and stay with the rank
//! ([`WorkCounts::rest`]): the cut cannot move them.
//!
//! # The constants
//!
//! The constants depend on the machine, the precision, p and the M2L strategy. Measured
//! by the per-kind timings (`KindTiming::Synchronous`, one thread per rank) as the time
//! of a kind summed over the ranks over its count summed over the ranks, the model
//! reproduced every rank's compute within 1.0–1.4 % on locust and the measured max over
//! mean within 0.05 on Kathleen (scale-out.md §6.1). [`CostModel::calibrated`] holds the
//! constants measured on Kathleen (the primary machine of Phase 5S) and scales them to
//! other degrees; [`FmmBuilder::cost_model`](crate::fmm::FmmBuilder::cost_model) names
//! others (for example locust's, whose M2L is relatively cheaper). Only the ratios
//! between the constants move the cut, and only the partition depends on them: a wrong
//! constant costs balance, never a bit of the output. The constants are fixed inputs,
//! never timed during a build, so the partition is deterministic.

use std::fmt;

use mpi::traits::{CommunicatorCollectives, Equivalence};
use nd_fmm_plan::exchange::{ExchangeError, SourceExchange};
use nd_fmm_plan::plan::Plan;
use nd_octree::{MortonKey, Octree, morton};
use rlst::distributed_tools::all_to_allv;
use thiserror::Error;

use crate::fmm::OperatorKind;
use crate::tables::M2lStrategy;

/// The work of a rank or a block by operator kind, in the units of the [module
/// documentation](self#the-model).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindCounts {
    counts: [u64; 8],
}

impl KindCounts {
    /// The count of `kind`.
    pub fn get(&self, kind: OperatorKind) -> u64 {
        self.counts[kind as usize]
    }

    /// Every kind with its count, in [`OperatorKind::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = (OperatorKind, u64)> + '_ {
        OperatorKind::ALL.into_iter().zip(self.counts)
    }

    /// The counts in [`OperatorKind::ALL`] order.
    pub fn to_array(&self) -> [u64; 8] {
        self.counts
    }

    /// Adds `count` to `kind`.
    fn add(&mut self, kind: OperatorKind, count: u64) {
        self.counts[kind as usize] += count;
    }

    /// The sum of two counts, kind by kind.
    pub fn sum(&self, other: &Self) -> Self {
        let mut counts = self.counts;
        for (count, other) in counts.iter_mut().zip(other.counts) {
            *count += other;
        }
        Self { counts }
    }
}

/// The work of a rank split by its coarse blocks ([`work_counts`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkCounts {
    /// The counts of every own block, in block order: the rows of the boxes and leaves at
    /// or below it.
    pub blocks: Vec<KindCounts>,
    /// The counts of the `Global` boxes' rows, which belong to no block.
    pub rest: KindCounts,
}

impl WorkCounts {
    /// The counts of the whole rank: every block and the rest.
    pub fn total(&self) -> KindCounts {
        self.blocks
            .iter()
            .fold(self.rest, |sum, block| sum.sum(block))
    }
}

/// Counts the work of every own block of `plan`, and the rest, from its views.
///
/// # Parameters
///
/// - `plan`: The plan of this rank.
/// - `blocks`: This rank's coarse blocks, ascending (`Octree::coarse_tree_leafs`).
/// - `sources`: The source points of every leaf of the plan's numbering, the ghost
///   leaves included (`SourceExchange::leaf_counts`).
/// - `targets`: The target points of every local leaf.
///
/// # Panics
///
/// Panics if `sources` or `targets` is shorter than the leaves it covers. A box at or
/// below none of `blocks` (a `Global` box) counts as [`WorkCounts::rest`].
pub fn work_counts(
    plan: &Plan,
    blocks: &[MortonKey],
    sources: &[usize],
    targets: &[usize],
) -> WorkCounts {
    use OperatorKind::{L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p};

    let index = plan.index();
    let leaves = index.leaves();
    let mut work = WorkCounts {
        blocks: vec![KindCounts::default(); blocks.len()],
        rest: KindCounts::default(),
    };
    // The block at or above an own key; `None` for a `Global` key, above every block.
    let block_of = |key: MortonKey| {
        let after = blocks.partition_point(|&block| block <= key);
        after
            .checked_sub(1)
            .filter(|&b| blocks[b] == key || morton::is_ancestor(blocks[b], key))
    };
    let mut add = |key: MortonKey, kind: OperatorKind, count: u64| {
        if count == 0 {
            return;
        }
        match block_of(key) {
            Some(b) => work.blocks[b].add(kind, count),
            None => work.rest.add(kind, count),
        }
    };

    for level in 0..plan.nlevels() {
        let lists = plan.level(level);
        let keys = index.keys(level);
        // Box views: one row per box of the level, empty for ghosts.
        for (t, &key) in keys.iter().enumerate() {
            add(key, M2l, lists.v().row(t).0.len() as u64);
            let x: usize = lists.x().row(t).iter().map(|&j| sources[j as usize]).sum();
            add(key, P2l, x as u64);
            let children = lists.m2m_local().row(t).0.len() + lists.m2m_global().row(t).0.len();
            add(key, M2m, children as u64);
            add(key, L2l, lists.l2l().row(t).0.len() as u64);
            let p2m: usize = lists
                .p2m()
                .row(t)
                .iter()
                .map(|&j| sources[j as usize])
                .sum();
            add(key, P2m, p2m as u64);
        }
        // Leaf views: one row per local leaf of the level.
        for (r, j) in leaves.local(level).enumerate() {
            let key = leaves.key(j);
            let n_t = targets[j] as u64;
            let near: usize = lists
                .near()
                .row(r)
                .iter()
                .map(|&s| sources[s as usize])
                .sum();
            add(key, P2p, n_t * near as u64);
            add(key, M2p, n_t * lists.w().row(r).len() as u64);
            add(key, L2p, n_t * lists.l2p().row(r).len() as u64);
        }
    }
    work
}

/// A distinct finest-level key with its numbers of source and target points on a rank,
/// as [`leaf_point_counts`] sends it to the key's owner.
///
/// `Default` is required by the collective helpers, which need a placeholder element.
#[derive(Clone, Copy, Debug, Default, Equivalence)]
struct CountedKey {
    key: MortonKey,
    sources: u64,
    targets: u64,
}

/// The source and target points of every local leaf of `plan`, counted where the
/// points are, before they move (the second cut of [`block_weights`]).
///
/// Every rank sorts the keys of its points, merges them into distinct keys with their
/// numbers of sources and targets, and sends each to the owner of its leaf by the
/// octree's partition bounds, which adds them up per local leaf: 24 bytes per distinct
/// key and rank, against the 12 bytes per point of each of `Fmm`'s two
/// `Redistribution`s.
///
/// # Collective operation
///
/// One all-to-all-v of the counted keys (with its all-to-all of counts and the
/// agreements of `rlst`'s `all_to_allv`); every rank must enter.
pub(crate) fn leaf_point_counts<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    plan: &Plan,
    source_keys: &[MortonKey],
    target_keys: &[MortonKey],
    comm: &C,
) -> (Vec<usize>, Vec<usize>) {
    let mut sources = source_keys.to_vec();
    sources.sort_unstable();
    let mut targets = target_keys.to_vec();
    targets.sort_unstable();
    // (key, sources, targets) of every distinct key, ascending.
    let mut counted: Vec<CountedKey> = Vec::new();
    let (mut s, mut t) = (0, 0);
    while s < sources.len() || t < targets.len() {
        let key = match (sources.get(s), targets.get(t)) {
            (Some(&a), Some(&b)) => a.min(b),
            (Some(&a), None) => a,
            (None, Some(&b)) => b,
            (None, None) => unreachable!(),
        };
        let s_end = s + sources[s..].partition_point(|&k| k == key);
        let t_end = t + targets[t..].partition_point(|&k| k == key);
        counted.push(CountedKey {
            key,
            sources: (s_end - s) as u64,
            targets: (t_end - t) as u64,
        });
        (s, t) = (s_end, t_end);
    }
    drop((sources, targets));
    // The owners do not decrease with the key.
    let size = comm.size() as usize;
    let mut per_rank = vec![0usize; size];
    for item in &counted {
        let owner = octree
            .owner_rank(item.key)
            .expect("the keys of points inside the domain are valid finest-level keys");
        per_rank[owner] += 1;
    }
    let (_received_counts, received) = all_to_allv(comm, &per_rank, &counted);
    drop(counted);

    let index = plan.index();
    let nlocal = index.leaves().nlocal();
    let mut leaf_sources = vec![0usize; nlocal];
    let mut leaf_targets = vec![0usize; nlocal];
    for CountedKey {
        key,
        sources: n_s,
        targets: n_t,
    } in received
    {
        let leaf = octree
            .local_leaf(key)
            .expect("a received key is a valid finest-level key")
            .expect("every rank routes by the same partition bounds");
        let j = index
            .find_leaf(leaf)
            .expect("every leaf of the octree is a local leaf of the plan")
            as usize;
        leaf_sources[j] += n_s as usize;
        leaf_targets[j] += n_t as usize;
    }
    (leaf_sources, leaf_targets)
}

/// The modelled cost of every own block of `octree` under `model`, from its plan and
/// the keys of this rank's points: the weights of the second cut
/// (`Octree::repartition`).
///
/// # Collective operation
///
/// [`leaf_point_counts`], then the build of a source exchange of one value per point,
/// which delivers the source counts of the ghost leaves (`SourceExchange::new`); every
/// rank must enter.
///
/// # Errors
///
/// The source exchange's, on every rank alike.
pub(crate) fn block_weights<C: CommunicatorCollectives>(
    octree: &Octree<'_, C>,
    plan: &Plan,
    source_keys: &[MortonKey],
    target_keys: &[MortonKey],
    model: &CostModel,
    comm: &C,
) -> Result<Vec<f64>, ExchangeError> {
    let (sources, targets) = leaf_point_counts(octree, plan, source_keys, target_keys, comm);
    let exchange = SourceExchange::<u8>::new(plan, comm, &sources, 1)?;
    let work = work_counts(
        plan,
        octree.coarse_tree_leafs(),
        exchange.leaf_counts(),
        &targets,
    );
    Ok(work
        .blocks
        .iter()
        .map(|counts| model.cost(counts))
        .collect())
}

/// Why [`CostModel::new`] rejects constants.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum CostModelError {
    /// A constant is negative, NaN or infinite.
    #[error("the constant of {kind} is {value}, not a finite, non-negative number")]
    Invalid {
        /// The kind of the constant.
        kind: OperatorKind,
        /// The constant.
        value: f64,
    },
    /// Every constant is zero: no work to weigh.
    #[error("every constant is zero")]
    AllZero,
}

/// Per-kind constants α_k of the modelled cost Σ_k α_k c_k, in nanoseconds per counted
/// unit (the [module documentation](self#the-model)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    alpha: [f64; 8],
}

/// One calibration of [`CostModel::calibrated`]: the constants measured for a strategy,
/// a value size and a degree.
struct Calibration {
    strategy: M2lStrategy,
    value_bytes: usize,
    p: usize,
    alpha: [f64; 8],
}

/// The constants measured on Kathleen (Phase 5S T7, job 603049; `nd-fmm-validate`'s
/// `scaling` with `--kinds`, the fit of `scaling::ModelFit`: one thread per rank, 40 ranks
/// on one node, the first cut, the cube, the Plummer sphere and the clusters at N = 10⁷;
/// per kind the median over the three workloads), in [`OperatorKind::ALL`] order: P2M,
/// M2M, M2L, P2L, L2L, L2P, M2P, P2P, in ns. With them the model was within 1.0–3.8 % of
/// every rank's sum of kind times (f64 p = 8 `Rotation`), 1.1–4.2 % (f64 p = 3 `Dense`),
/// 2.1–3.4 % (f32 `Rotation`) and 4.2–8.0 % (f32 `Dense`, whose M2L streams each rank's
/// own 17 MB of tables).
const CALIBRATIONS: [Calibration; 4] = [
    Calibration {
        strategy: M2lStrategy::Rotation,
        value_bytes: 8,
        p: 8,
        alpha: [
            664.03, 2754.50, 3129.00, 568.13, 2799.44, 1086.56, 1299.22, 3.0852,
        ],
    },
    Calibration {
        strategy: M2lStrategy::Dense,
        value_bytes: 8,
        p: 3,
        alpha: [
            112.66, 111.56, 117.42, 108.76, 124.71, 232.43, 329.63, 3.0675,
        ],
    },
    Calibration {
        strategy: M2lStrategy::Rotation,
        value_bytes: 4,
        p: 8,
        alpha: [
            596.01, 2561.45, 2805.80, 599.76, 2603.79, 1076.63, 1273.20, 1.3470,
        ],
    },
    Calibration {
        strategy: M2lStrategy::Dense,
        value_bytes: 4,
        p: 8,
        alpha: [
            600.07, 1417.28, 4322.29, 629.35, 1557.76, 1098.15, 1288.16, 1.3556,
        ],
    },
];

impl CostModel {
    /// Constants α_k in nanoseconds per counted unit, in [`OperatorKind::ALL`] order.
    ///
    /// # Errors
    ///
    /// [`CostModelError::Invalid`] for a negative, NaN or infinite constant,
    /// [`CostModelError::AllZero`] if every constant is zero.
    pub fn new(alpha: [f64; 8]) -> Result<Self, CostModelError> {
        if let Some((kind, &value)) = OperatorKind::ALL
            .into_iter()
            .zip(&alpha)
            .find(|(_, value)| !(value.is_finite() && **value >= 0.0))
        {
            return Err(CostModelError::Invalid { kind, value });
        }
        if alpha.iter().all(|&value| value == 0.0) {
            return Err(CostModelError::AllZero);
        }
        Ok(Self { alpha })
    }

    /// The constants measured on Kathleen for `strategy`, values of `value_bytes` bytes
    /// (8 for f64, 4 for f32) and degree `p`.
    ///
    /// The table holds f64 at p = 3 (`Dense`) and p = 8 (`Rotation`), and f32 at p = 8
    /// (`Dense` and `Rotation`), the configurations of Phase 5S. Another degree takes the
    /// calibration of the same strategy and precision nearest in p (`Classes` and
    /// `Batched` take `Dense`'s, `Auto` `Rotation`'s; another precision f64's) and scales
    /// it: the leaf
    /// operators by (p + 1)², the translations by (p + 1)³ under `Rotation` and (p + 1)⁴
    /// otherwise, P2P not at all.
    pub fn calibrated(strategy: M2lStrategy, value_bytes: usize, p: usize) -> Self {
        let strategy = match strategy {
            M2lStrategy::Classes | M2lStrategy::Batched => M2lStrategy::Dense,
            M2lStrategy::Auto => M2lStrategy::Rotation,
            named => named,
        };
        let candidates = |bytes: usize| {
            CALIBRATIONS
                .iter()
                .filter(move |c| c.strategy == strategy && c.value_bytes == bytes)
        };
        let bytes = if candidates(value_bytes).next().is_some() {
            value_bytes
        } else {
            8
        };
        let nearest = candidates(bytes)
            .min_by_key(|c| c.p.abs_diff(p))
            .expect("every strategy has an f64 calibration");
        let ratio = (p as f64 + 1.0) / (nearest.p as f64 + 1.0);
        let translation = if strategy == M2lStrategy::Rotation {
            ratio.powi(3)
        } else {
            ratio.powi(4)
        };
        let alpha = OperatorKind::ALL.map(|kind| {
            let scale = match kind {
                OperatorKind::P2m | OperatorKind::L2p | OperatorKind::P2l | OperatorKind::M2p => {
                    ratio.powi(2)
                }
                OperatorKind::M2m | OperatorKind::M2l | OperatorKind::L2l => translation,
                OperatorKind::P2p => 1.0,
            };
            nearest.alpha[kind as usize] * scale
        });
        Self { alpha }
    }

    /// The constant of `kind`, in nanoseconds per counted unit.
    pub fn alpha(&self, kind: OperatorKind) -> f64 {
        self.alpha[kind as usize]
    }

    /// The constants in [`OperatorKind::ALL`] order.
    pub fn to_array(&self) -> [f64; 8] {
        self.alpha
    }

    /// The modelled cost of `counts`, Σ_k α_k c_k, in nanoseconds.
    pub fn cost(&self, counts: &KindCounts) -> f64 {
        counts
            .iter()
            .map(|(kind, count)| self.alpha(kind) * count as f64)
            .sum()
    }
}

impl fmt::Display for CostModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, kind) in OperatorKind::ALL.into_iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{kind} {:.4} ns", self.alpha(kind))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_are_checked() {
        assert!(CostModel::new([1.0; 8]).is_ok());
        assert_eq!(CostModel::new([0.0; 8]), Err(CostModelError::AllZero));
        let mut alpha = [1.0; 8];
        alpha[2] = -1.0;
        assert_eq!(
            CostModel::new(alpha),
            Err(CostModelError::Invalid {
                kind: OperatorKind::M2l,
                value: -1.0
            })
        );
        alpha[2] = f64::NAN;
        assert!(matches!(
            CostModel::new(alpha),
            Err(CostModelError::Invalid {
                kind: OperatorKind::M2l,
                ..
            })
        ));
    }

    #[test]
    fn cost_is_the_weighted_sum() {
        let model = CostModel::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.5]).unwrap();
        let mut counts = KindCounts::default();
        counts.add(OperatorKind::M2l, 10);
        counts.add(OperatorKind::P2p, 4);
        assert_eq!(model.cost(&counts), 32.0);
        assert_eq!(counts.sum(&counts).get(OperatorKind::M2l), 20);
    }

    #[test]
    fn calibrations_scale_with_p() {
        for calibration in &CALIBRATIONS {
            let model =
                CostModel::calibrated(calibration.strategy, calibration.value_bytes, calibration.p);
            assert_eq!(model.to_array(), calibration.alpha);
            assert!(CostModel::new(calibration.alpha).is_ok());
        }
        // Rotation at f64 p = 17: from p = 8, translations by (18 / 9)³, leaf operators
        // by (18 / 9)², P2P unchanged.
        let base = CostModel::calibrated(M2lStrategy::Rotation, 8, 8);
        let scaled = CostModel::calibrated(M2lStrategy::Rotation, 8, 17);
        assert_eq!(
            scaled.alpha(OperatorKind::M2l),
            8.0 * base.alpha(OperatorKind::M2l)
        );
        assert_eq!(
            scaled.alpha(OperatorKind::P2m),
            4.0 * base.alpha(OperatorKind::P2m)
        );
        assert_eq!(
            scaled.alpha(OperatorKind::P2p),
            base.alpha(OperatorKind::P2p)
        );
        // Classes takes Dense's; Auto Rotation's; a precision without a calibration f64's.
        assert_eq!(
            CostModel::calibrated(M2lStrategy::Classes, 8, 3),
            CostModel::calibrated(M2lStrategy::Dense, 8, 3)
        );
        assert_eq!(
            CostModel::calibrated(M2lStrategy::Auto, 8, 8),
            CostModel::calibrated(M2lStrategy::Rotation, 8, 8)
        );
        assert_eq!(
            CostModel::calibrated(M2lStrategy::Dense, 2, 3),
            CostModel::calibrated(M2lStrategy::Dense, 8, 3)
        );
        // Dense at f64 p = 5 scales from p = 3 by (6 / 4)⁴ for the translations.
        let dense = CostModel::calibrated(M2lStrategy::Dense, 8, 3);
        let dense5 = CostModel::calibrated(M2lStrategy::Dense, 8, 5);
        let expected = dense.alpha(OperatorKind::M2m) * 1.5f64.powi(4);
        assert!((dense5.alpha(OperatorKind::M2m) - expected).abs() <= 1e-12 * expected);
    }
}
