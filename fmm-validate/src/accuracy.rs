//! The single-translation accuracy sweep: the operator chains of `nd-fmm-ref` for one
//! pair of well-separated boxes, against `nd_fmm_ref::p2p::direct_sum`.
//!
//! Its table is the single-translation prediction against which the uniform-tree FMM
//! is accepted in Phase 3 (design §7, C3.2), and the starting point of the p
//! calibration of C3.4.
//!
//! # Geometry
//!
//! The standard one-box separation of a uniform octree level (design §2.5): a source
//! box with centre c = (½, ½, ½) and half-width r = ½ (the scaling radius of
//! CONVENTIONS §3.7), and a target box of the same size with centre c + 2r·d for a
//! V-list offset d ∈ {−3..3}³ \ {−1..1}³, 316 offsets in all ([`v_list_offsets`]).
//! The six face-adjacent offsets (d a permutation of (±2, 0, 0)) have the smallest
//! centre distance, |b| = 4 in scaled units, and so the worst convergence ratio
//! 2√3/4 ≈ 0.87 of the M2L bound (CONVENTIONS §3.11, "M2L"); they are equivalent under
//! the cube symmetries, and which of them gives the largest error depends on the
//! random sample. The sweep therefore finds the worst offset by measurement: the one
//! with the largest error among all 316.
//!
//! Sources are uniform in the source box with charges uniform in [−1, 1); targets are
//! uniform in the target box, at the same box-relative positions for every offset.
//! The frames and points are rounded to the working precision T first, and the oracle
//! gets the rounded values, so input rounding is not counted as operator error.
//!
//! # Chains
//!
//! Every expansion has degree p; translations are those of `nd_fmm_ref::direct`
//! (the `rotation` operators agree with them to 1e-13, Phase 1 T6).
//!
//! - [`Chain::P2mM2p`]: P2M in the source box, M2P at the targets.
//! - [`Chain::P2lL2p`]: P2L of the sources in the target box, L2P at the targets.
//! - [`Chain::P2mM2mM2p`]: P2M in each of the 8 children of the source box (half-width
//!   r/2), M2M child to parent, M2P at the targets: the upward pass.
//! - [`Chain::P2mM2lL2p`]: P2M in the source box, M2L to the target box, L2P.
//! - [`Chain::P2mM2lL2lL2p`]: as the previous chain, then L2L to each of the 8 children
//!   of the target box and L2P at the targets in that child: the downward pass.
//!
//! M2M and L2L are exact to degree p (CONVENTIONS §3.11), so the third chain agrees
//! with the first, and the fifth with the fourth, up to rounding.
//!
//! # Error measure
//!
//! Relative L2 and max error of the potential and of the gradient (Euclidean norm per
//! target) against the f64 direct sum, as in design §8.2 ([`ErrorNorms`]), reported
//! twice:
//!
//! - *worst offset*: each error is measured over the targets of one offset, and the
//!   largest over the 316 offsets is reported, separately for each of the four
//!   measures ([`Row::worst`]), with the offset of the largest potential L2 error
//!   ([`Row::worst_offset`]);
//! - *all offsets*: each error is measured over the union of the targets of all 316
//!   offsets, as if they were one set ([`ErrorAccumulator`]).
//!
//! The charges have both signs, so potentials partly cancel; these figures are the
//! end-to-end measure of design §8.2, not a rounding-level test.
//!
//! ```
//! use nd_fmm_validate::accuracy::{Chain, Config, sweep};
//!
//! let config = Config { sources: 20, targets: 20, seed: 1 };
//! let rows = sweep::<f64>(&config, 2);
//! assert_eq!(rows.len(), Chain::ALL.len() * 2);
//! assert!(rows.iter().all(|row| row.all.potential.l2 < 1.0));
//! ```

use core::ops::Range;

use nd_fmm_math::{Layout, RealScalar};
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_ref::{Frame, Workspace, direct, leaf};

use crate::metrics::{ErrorAccumulator, ErrorNorms};
use crate::{SplitMix64, points};

/// Centre of the source box.
pub const SOURCE_CENTRE: [f64; 3] = [0.5, 0.5, 0.5];

/// Half-width r of both boxes, their scaling radius (CONVENTIONS §3.7).
pub const RADIUS: f64 = 0.5;

/// The 316 V-list offsets d ∈ {−3..3}³ \ {−1..1}³ of a uniform octree level (design
/// §2.5), in lexicographic order of (x, y, z). The target box centre is c + 2r·d.
pub fn v_list_offsets() -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(316);
    for x in -3..=3 {
        for y in -3..=3 {
            for z in -3..=3 {
                let d: [i32; 3] = [x, y, z];
                if d.iter().any(|c: &i32| c.abs() > 1) {
                    out.push(d);
                }
            }
        }
    }
    out
}

/// One of the operator chains of the sweep (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chain {
    /// P2M → M2P.
    P2mM2p,
    /// P2L → L2P.
    P2lL2p,
    /// P2M (children) → M2M → M2P.
    P2mM2mM2p,
    /// P2M → M2L → L2P.
    P2mM2lL2p,
    /// P2M → M2L → L2L (children) → L2P.
    P2mM2lL2lL2p,
}

impl Chain {
    /// All chains, in the order of the report.
    pub const ALL: [Chain; 5] = [
        Chain::P2mM2p,
        Chain::P2lL2p,
        Chain::P2mM2mM2p,
        Chain::P2mM2lL2p,
        Chain::P2mM2lL2lL2p,
    ];

    /// The chain written with arrows, e.g. "P2M → M2L → L2P".
    pub fn name(self) -> &'static str {
        match self {
            Chain::P2mM2p => "P2M → M2P",
            Chain::P2lL2p => "P2L → L2P",
            Chain::P2mM2mM2p => "P2M → M2M → M2P",
            Chain::P2mM2lL2p => "P2M → M2L → L2P",
            Chain::P2mM2lL2lL2p => "P2M → M2L → L2L → L2P",
        }
    }
}

/// Size and seed of a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Number of sources in the source box.
    pub sources: usize,
    /// Number of targets in each target box.
    pub targets: usize,
    /// Seed of the [`SplitMix64`] that draws sources, charges and targets.
    pub seed: u64,
}

impl Config {
    /// The configuration of the Phase 1 report: 1,000 sources and 1,000 targets.
    pub const STANDARD: Config = Config {
        sources: 1000,
        targets: 1000,
        seed: 20260929,
    };
}

/// Relative errors of the potential and of the gradient.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Errors {
    /// Relative L2 and max error of the potential.
    pub potential: ErrorNorms,
    /// Relative L2 and max error of the gradient (Euclidean norm per target).
    pub gradient: ErrorNorms,
}

/// The result of one chain at one degree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Row {
    /// The chain.
    pub chain: Chain,
    /// The degree p of every expansion in the chain.
    pub p: usize,
    /// The largest per-offset errors over the 316 offsets, each of the four maximised
    /// separately.
    pub worst: Errors,
    /// The offset d whose target box has the largest potential L2 error.
    pub worst_offset: [i32; 3],
    /// Errors over the targets of all 316 offsets together.
    pub all: Errors,
}

/// Accumulators of one chain at one degree, for the potential and the gradient.
#[derive(Clone, Copy, Default)]
struct Accumulators {
    potential: ErrorAccumulator,
    gradient: ErrorAccumulator,
}

impl Accumulators {
    fn add<T: RealScalar>(
        &mut self,
        potential: &[T],
        gradient: &[[T; 3]],
        exact: &[f64],
        exact_gradient: &[[f64; 3]],
    ) {
        self.potential.add_values(potential, exact);
        self.gradient.add_vectors(gradient, exact_gradient);
    }

    fn finish(&self) -> Errors {
        Errors {
            potential: self.potential.finish(),
            gradient: self.gradient.finish(),
        }
    }
}

/// The largest per-offset errors seen so far, and the offset of the largest potential
/// L2 error.
#[derive(Clone, Copy, Default)]
struct Worst {
    errors: Errors,
    offset: [i32; 3],
}

impl Worst {
    fn update(&mut self, errors: Errors, offset: [i32; 3]) {
        let max = |a: ErrorNorms, b: ErrorNorms| ErrorNorms {
            l2: a.l2.max(b.l2),
            max: a.max.max(b.max),
        };
        if errors.potential.l2 > self.errors.potential.l2 {
            self.offset = offset;
        }
        self.errors = Errors {
            potential: max(self.errors.potential, errors.potential),
            gradient: max(self.errors.gradient, errors.gradient),
        };
    }
}

/// Points rounded to T, with their f64 values (the rounded ones, for the oracle), sorted
/// by the child octant of a box they fall in, and the index range of each octant.
struct Sorted<T> {
    points: Vec<[T; 3]>,
    points64: Vec<[f64; 3]>,
    order: Vec<usize>,
    octants: [Range<usize>; 8],
}

/// Octant 4x + 2y + z of `u` relative to the centre: bit set where the coordinate is
/// ≥ 0.
fn octant(u: [f64; 3]) -> usize {
    (0..3).fold(0, |acc, i| (acc << 1) | usize::from(u[i] >= 0.0))
}

/// The frame of child `octant` (4x + 2y + z, bit set for the + side) of the box
/// (`centre`, `radius`): half the radius, centre moved by ± radius/2.
fn child(centre: [f64; 3], radius: f64, octant: usize) -> ([f64; 3], f64) {
    let h = radius / 2.0;
    let c = core::array::from_fn(|i| {
        let bit = octant >> (2 - i) & 1;
        centre[i] + if bit == 1 { h } else { -h }
    });
    (c, h)
}

/// Sorts `local` (box-relative scaled positions) by octant, and returns the order and
/// the octant ranges.
fn sort_by_octant(local: &[[f64; 3]]) -> (Vec<usize>, [Range<usize>; 8]) {
    let mut order: Vec<usize> = (0..local.len()).collect();
    order.sort_by_key(|&i| octant(local[i]));
    let mut ranges: [Range<usize>; 8] = Default::default();
    let mut start = 0;
    for (k, range) in ranges.iter_mut().enumerate() {
        let count = order.iter().filter(|&&i| octant(local[i]) == k).count();
        *range = start..start + count;
        start += count;
    }
    (order, ranges)
}

/// Places the box-relative positions `local` (sorted by `order`) in the box
/// (`centre`, `radius`) and rounds them to T.
fn place<T: RealScalar>(
    local: &[[f64; 3]],
    order: &[usize],
    centre: [f64; 3],
    radius: f64,
) -> (Vec<[T; 3]>, Vec<[f64; 3]>) {
    let points: Vec<[T; 3]> = order
        .iter()
        .map(|&i| core::array::from_fn(|k| T::from_f64(centre[k] + radius * local[i][k])))
        .collect();
    let points64 = points
        .iter()
        .map(|x| x.map(|c| RealScalar::to_f64(c)))
        .collect();
    (points, points64)
}

impl<T: RealScalar> Sorted<T> {
    fn new(local: &[[f64; 3]], centre: [f64; 3], radius: f64) -> Self {
        let (order, octants) = sort_by_octant(local);
        let (points, points64) = place(local, &order, centre, radius);
        Self {
            points,
            points64,
            order,
            octants,
        }
    }

    /// Moves the same box-relative points into the box (`centre`, `radius`).
    fn move_to(&mut self, local: &[[f64; 3]], centre: [f64; 3], radius: f64) {
        (self.points, self.points64) = place(local, &self.order, centre, radius);
    }
}

fn frame<T: RealScalar>(centre: [f64; 3], radius: f64) -> Frame<T> {
    Frame::new(centre.map(T::from_f64), T::from_f64(radius))
}

/// Runs every [`Chain`] at every degree p = 1..=`p_max` in precision T, for the target
/// boxes at all 316 V-list offsets, and returns one [`Row`] per chain and degree
/// (chains in the order of [`Chain::ALL`], then increasing p).
///
/// Serial; the cost is dominated by the 316 direct sums of `config.sources` ×
/// `config.targets` pairs and by the evaluations at `316 · config.targets` targets per
/// chain and degree.
///
/// # Panics
///
/// If `p_max == 0`, or `config` has no sources or no targets.
pub fn sweep<T: RealScalar>(config: &Config, p_max: usize) -> Vec<Row> {
    assert!(p_max >= 1, "the sweep starts at p = 1");
    assert!(
        config.sources > 0 && config.targets > 0,
        "the sweep needs sources and targets"
    );
    let mut rng = SplitMix64::new(config.seed);
    let source_local = points::cube(&mut rng, config.sources, [0.0; 3], 1.0);
    let charges_unsorted = points::charges(&mut rng, config.sources);
    let target_local = points::cube(&mut rng, config.targets, [0.0; 3], 1.0);

    let sources = Sorted::<T>::new(&source_local, SOURCE_CENTRE, RADIUS);
    let charges: Vec<T> = sources
        .order
        .iter()
        .map(|&i| T::from_f64(charges_unsorted[i]))
        .collect();
    let charges64: Vec<f64> = charges.iter().map(|q| RealScalar::to_f64(*q)).collect();
    let mut targets = Sorted::<T>::new(&target_local, SOURCE_CENTRE, RADIUS);

    let source_frame = frame::<T>(SOURCE_CENTRE, RADIUS);
    let source_children: [Frame<T>; 8] = core::array::from_fn(|k| {
        let (c, r) = child(SOURCE_CENTRE, RADIUS, k);
        frame(c, r)
    });

    let mut ws = Workspace::<T>::new(p_max);
    let max_len = Layout::new(p_max).len();

    // Source side, independent of the offset: P2M in the source box, and P2M in the
    // children followed by M2M to the source box.
    let mut multipoles = Vec::with_capacity(p_max + 1);
    let mut merged = Vec::with_capacity(p_max + 1);
    let mut child_multipole = vec![T::zero(); max_len];
    for p in 0..=p_max {
        let len = Layout::new(p).len();
        let mut m = vec![T::zero(); len];
        leaf::p2m(p, &source_frame, &sources.points, &charges, &mut ws, &mut m);
        multipoles.push(m);
        let mut m = vec![T::zero(); len];
        for (k, child) in source_children.iter().enumerate() {
            let range = sources.octants[k].clone();
            if range.is_empty() {
                continue;
            }
            let cm = &mut child_multipole[..len];
            cm.fill(T::zero());
            leaf::p2m(
                p,
                child,
                &sources.points[range.clone()],
                &charges[range],
                &mut ws,
                cm,
            );
            direct::m2m(p, child, &source_frame, &mut ws, cm, &mut m);
        }
        merged.push(m);
    }

    let n_chains = Chain::ALL.len();
    let mut all = vec![Accumulators::default(); n_chains * (p_max + 1)];
    let mut worst = vec![Worst::default(); n_chains * (p_max + 1)];
    let slot = |c: usize, p: usize| c * (p_max + 1) + p;

    let n_t = config.targets;
    let mut exact = vec![0.0; n_t];
    let mut exact_gradient = vec![[0.0; 3]; n_t];
    let mut potential = vec![T::zero(); n_t];
    let mut gradient = vec![[T::zero(); 3]; n_t];
    let mut local = vec![T::zero(); max_len];
    let mut child_local = vec![T::zero(); max_len];

    for offset in v_list_offsets() {
        let centre: [f64; 3] =
            core::array::from_fn(|i| SOURCE_CENTRE[i] + 2.0 * RADIUS * f64::from(offset[i]));
        targets.move_to(&target_local, centre, RADIUS);
        let target_frame = frame::<T>(centre, RADIUS);
        let target_children: [Frame<T>; 8] = core::array::from_fn(|k| {
            let (c, r) = child(centre, RADIUS, k);
            frame(c, r)
        });

        exact.fill(0.0);
        exact_gradient.fill([0.0; 3]);
        direct_sum(
            &sources.points64,
            &charges64,
            &targets.points64,
            &mut exact,
            Some(&mut exact_gradient),
        );

        for p in 1..=p_max {
            let len = Layout::new(p).len();
            for (c, chain) in Chain::ALL.into_iter().enumerate() {
                potential.fill(T::zero());
                gradient.fill([T::zero(); 3]);
                let local = &mut local[..len];
                local.fill(T::zero());
                let (pot, grad) = (&mut potential[..], Some(&mut gradient[..]));
                match chain {
                    Chain::P2mM2p | Chain::P2mM2mM2p => {
                        let m = if chain == Chain::P2mM2p {
                            &multipoles[p]
                        } else {
                            &merged[p]
                        };
                        leaf::m2p(p, &source_frame, m, &targets.points, &mut ws, pot, grad);
                    }
                    Chain::P2lL2p => {
                        leaf::p2l(p, &target_frame, &sources.points, &charges, &mut ws, local);
                        leaf::l2p(p, &target_frame, local, &targets.points, &mut ws, pot, grad);
                    }
                    Chain::P2mM2lL2p => {
                        direct::m2l(
                            p,
                            &source_frame,
                            &target_frame,
                            &mut ws,
                            &multipoles[p],
                            local,
                        );
                        leaf::l2p(p, &target_frame, local, &targets.points, &mut ws, pot, grad);
                    }
                    Chain::P2mM2lL2lL2p => {
                        direct::m2l(
                            p,
                            &source_frame,
                            &target_frame,
                            &mut ws,
                            &multipoles[p],
                            local,
                        );
                        for (k, child) in target_children.iter().enumerate() {
                            let range = targets.octants[k].clone();
                            if range.is_empty() {
                                continue;
                            }
                            let cl = &mut child_local[..len];
                            cl.fill(T::zero());
                            direct::l2l(p, &target_frame, child, &mut ws, local, cl);
                            leaf::l2p(
                                p,
                                child,
                                cl,
                                &targets.points[range.clone()],
                                &mut ws,
                                &mut potential[range.clone()],
                                Some(&mut gradient[range]),
                            );
                        }
                    }
                }
                all[slot(c, p)].add(&potential, &gradient, &exact, &exact_gradient);
                let mut this = Accumulators::default();
                this.add(&potential, &gradient, &exact, &exact_gradient);
                worst[slot(c, p)].update(this.finish(), offset);
            }
        }
    }

    Chain::ALL
        .into_iter()
        .enumerate()
        .flat_map(|(c, chain)| {
            let (all, worst) = (&all, &worst);
            (1..=p_max).map(move |p| Row {
                chain,
                p,
                worst: worst[slot(c, p)].errors,
                worst_offset: worst[slot(c, p)].offset,
                all: all[slot(c, p)].finish(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_the_v_list() {
        let offsets = v_list_offsets();
        assert_eq!(offsets.len(), 316);
        assert!(offsets.contains(&[2, 0, 0]) && offsets.contains(&[-3, 3, -1]));
        assert!(!offsets.contains(&[1, -1, 1]));
        for d in &offsets {
            let cheb = d.iter().map(|c| c.abs()).max().unwrap();
            assert!((2..=3).contains(&cheb), "{d:?}");
        }
        let mut sorted = offsets.clone();
        sorted.dedup();
        assert_eq!(sorted.len(), 316);
    }

    #[test]
    fn children_tile_their_parent() {
        // Error measure: exact equality (dyadic values).
        for k in 0..8 {
            let (c, r) = child([0.5; 3], 0.5, k);
            assert_eq!(r, 0.25);
            let u: [f64; 3] = core::array::from_fn(|i| c[i] - 0.5);
            assert_eq!(octant(u), k);
        }
        let (order, ranges) = sort_by_octant(&[[0.1, -0.2, 0.3], [-0.5, 0.5, -0.5], [0.0; 3]]);
        assert_eq!(order, [1, 0, 2]);
        assert_eq!(ranges[2], 0..1);
        assert_eq!(ranges[5], 1..2);
        assert_eq!(ranges[7], 2..3);
        assert!(ranges[0].is_empty());
    }
}
