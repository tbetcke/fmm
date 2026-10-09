//! The shadow check of the host-data hook (Phase 5 T7, C5.1 device;
//! docs/design/distributed-fmm.md §6.4), used by `tests/mpi_exec.rs`.
//!
//! [`Shadow`] wraps `LaplaceOperator` and keeps its own copies of the five stores of an
//! evaluation, laid out as the evaluator's. Every level call runs the wrapped operator on
//! a batch rebuilt from those copies, never on the evaluator's slices, with the batch's
//! own views. It learns of the caller's data as a device does, from the test
//! ([`Shadow::load_points`] once, [`Shadow::begin_evaluation`] per evaluation), and of
//! the evaluator's own data movements only through `FmmOperator::host_data`:
//! - `Reset`: it zeroes its multipoles, locals and target output;
//! - `SendSources`, `SendMultipoles`: it writes its values of the listed leaves, coarse
//!   blocks and boxes into the host store, which the exchange then sends;
//! - `ReceivedSources`: it copies the ghost tail from the host store; `ReceivedCoarse`
//!   and `ReceivedMultipoles`: it copies the received values from the exchanges' packed
//!   buffers (`CoarseExchange::chunk`, `MultipoleExchange::receive_buffer`).
//!
//! So its output equals the plain operator's bit for bit exactly when the events cover
//! every host read and write of the evaluator. With the hook disabled it ignores every
//! event: on one rank, where no event carries data, one evaluation from fresh copies is
//! still exact; on several ranks the ghosts and the other ranks' coarse blocks stay zero
//! and the output differs ([`check_fmm`] reports how many values differ).
//!
//! [`check_fmm`] drives the plan's `Evaluator` with a [`Shadow`] on the octree and plan of
//! a built `Fmm`, moves the points and charges to their owners with its own
//! `Redistribution`s (as `Fmm::build` and `Fmm::evaluate` do), and compares the scaled
//! output, moved back to the caller's order, with the `Fmm`'s output bit for bit.

use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{Fmm, Output};
use nd_fmm_exec::geometry::{leaf_coordinates, radius};
use nd_fmm_exec::operator::{LaplaceOperator, SimdScalar};
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, HostData, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p,
};
use nd_fmm_plan::redistribute::Redistribution;
use nd_fmm_plan::store::{LeafStore, LevelBuffers};
use nd_fmm_tables::cache::Stored;
use nd_octree::{MortonKey, constants::DEEPEST_LEVEL, morton};

/// `LaplaceOperator` on its own copies of the stores (module documentation).
pub struct Shadow<T: SimdScalar + Equivalence + Default> {
    operator: LaplaceOperator<T>,
    /// Whether it follows the host-data events; without, it ignores them.
    hook: bool,
    /// The host-data events received.
    events: usize,
    multipoles: LevelBuffers<T>,
    locals: LevelBuffers<T>,
    /// Every leaf of the numbering: local, then ghost.
    sources: LeafStore<T>,
    /// The local leaves.
    target_input: LeafStore<T>,
    /// The local leaves.
    target_output: LeafStore<T>,
}

impl<T: SimdScalar + Equivalence + Default> Shadow<T> {
    /// A shadow of `operator`, without stores until [`attach`](Self::attach); `hook`
    /// says whether it follows the host-data events.
    pub fn new(operator: LaplaceOperator<T>, hook: bool) -> Self {
        Self {
            operator,
            hook,
            events: 0,
            multipoles: LevelBuffers::new(&[], &[]),
            locals: LevelBuffers::new(&[], &[]),
            sources: LeafStore::new(&[], 0),
            target_input: LeafStore::new(&[], 0),
            target_output: LeafStore::new(&[], 0),
        }
    }

    /// Allocates its stores zeroed, laid out as the evaluator's for `index`:
    /// `source_counts` for every leaf of the numbering (ghosts included, as the
    /// evaluator's source store has them), `target_counts` for the local leaves.
    pub fn attach(&mut self, index: &BoxIndex, source_counts: &[usize], target_counts: &[usize]) {
        let op = &self.operator;
        let multipole_sizes: Vec<usize> =
            (0..index.nlevels()).map(|l| op.multipole_size(l)).collect();
        let local_sizes: Vec<usize> = (0..index.nlevels()).map(|l| op.local_size(l)).collect();
        self.multipoles = LevelBuffers::from_index(index, &multipole_sizes);
        self.locals = LevelBuffers::from_index(index, &local_sizes);
        self.sources = LeafStore::new(source_counts, op.source_point_size());
        self.target_input = LeafStore::new(target_counts, op.target_input_point_size());
        self.target_output = LeafStore::new(target_counts, op.target_output_point_size());
    }

    /// Loads the leaf-scaled points of the local leaves (CONVENTIONS §3.13), three
    /// coordinates per point in leaf order: `sources` into its source chunks, `targets`
    /// into its target input.
    pub fn load_points(&mut self, sources: &[T], targets: &[T]) {
        let (mut s, mut t) = (0, 0);
        for j in 0..self.target_input.nleaves() {
            let n = self.sources.count(j);
            self.sources.chunk_mut(j)[..3 * n].copy_from_slice(&sources[3 * s..3 * (s + n)]);
            s += n;
            let m = self.target_input.count(j);
            self.target_input
                .chunk_mut(j)
                .copy_from_slice(&targets[3 * t..3 * (t + m)]);
            t += m;
        }
        assert_eq!((3 * s, 3 * t), (sources.len(), targets.len()));
    }

    /// Loads the charges of this evaluation, one per source point in leaf order.
    pub fn begin_evaluation(&mut self, charges: &[T]) {
        let mut s = 0;
        for j in 0..self.target_input.nleaves() {
            let n = self.sources.count(j);
            self.sources.chunk_mut(j)[3 * n..].copy_from_slice(&charges[s..s + n]);
            s += n;
        }
        assert_eq!(s, charges.len());
    }

    /// Returns its target output.
    pub fn target_output(&self) -> &LeafStore<T> {
        &self.target_output
    }

    /// Returns the number of host-data events received.
    pub fn events(&self) -> usize {
        self.events
    }
}

impl<T: SimdScalar + Equivalence + Default> FmmSizes for Shadow<T> {
    type Value = T;

    fn multipole_size(&self, level: usize) -> usize {
        self.operator.multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.operator.local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.operator.source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.operator.target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.operator.target_output_point_size()
    }
}

/// Every level call: the batch's views and level, the shadow's own slices. The
/// evaluator's slices only have their shapes checked.
impl<T: SimdScalar + Equivalence + Default> FmmOperator for Shadow<T> {
    fn p2m(&mut self, b: P2m<'_, T>) {
        assert_eq!(b.multipoles.len(), self.multipoles.len(b.level));
        self.operator.p2m(P2m {
            sources: self.sources.range(0..self.sources.nleaves()),
            multipoles: self.multipoles.level_mut(b.level),
            ..b
        });
    }

    fn m2m(&mut self, b: M2m<'_, T>) {
        assert_eq!(b.multipoles.len(), self.multipoles.len(b.level));
        let (multipoles, child_multipoles) = self.multipoles.parent_child_mut(b.level);
        self.operator.m2m(M2m {
            child_multipoles,
            multipoles,
            ..b
        });
    }

    fn m2l(&mut self, b: M2l<'_, T>) {
        assert_eq!(b.locals.len(), self.locals.len(b.level));
        self.operator.m2l(M2l {
            multipoles: self.multipoles.level(b.level),
            locals: self.locals.level_mut(b.level),
            ..b
        });
    }

    fn p2l(&mut self, b: P2l<'_, T>) {
        assert_eq!(b.sources.nleaves(), self.sources.nleaves());
        self.operator.p2l(P2l {
            sources: self.sources.range(0..self.sources.nleaves()),
            locals: self.locals.level_mut(b.level),
            ..b
        });
    }

    fn l2l(&mut self, b: L2l<'_, T>) {
        assert_eq!(b.locals.len(), self.locals.len(b.level));
        let (locals, parent_locals) = self.locals.child_parent_mut(b.level - 1);
        self.operator.l2l(L2l {
            parent_locals,
            locals,
            ..b
        });
    }

    fn l2p(&mut self, b: L2p<'_, T>) {
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        let leaves = b.leaves.clone();
        self.operator.l2p(L2p {
            locals: self.locals.level(b.level),
            target_input: self.target_input.range(leaves.clone()),
            target_output: self.target_output.range_mut(leaves),
            ..b
        });
    }

    fn m2p(&mut self, b: M2p<'_, T>) {
        assert_eq!(b.target_output.nleaves(), b.leaves.len());
        let leaves = b.leaves.clone();
        // On the deepest level the batch's own empty level, which carries no data.
        let multipoles = if b.level + 1 < self.multipoles.nlevels() {
            self.multipoles.level(b.level + 1)
        } else {
            b.multipoles
        };
        self.operator.m2p(M2p {
            multipoles,
            target_input: self.target_input.range(leaves.clone()),
            target_output: self.target_output.range_mut(leaves),
            ..b
        });
    }

    fn p2p(&mut self, b: P2p<'_, T>) {
        assert_eq!(b.sources.nleaves(), self.sources.nleaves());
        let leaves = b.leaves.clone();
        self.operator.p2p(P2p {
            sources: self.sources.range(0..self.sources.nleaves()),
            target_input: self.target_input.range(leaves.clone()),
            target_output: self.target_output.range_mut(leaves),
            ..b
        });
    }

    fn host_data(&mut self, event: HostData<'_, T>) {
        self.events += 1;
        if !self.hook {
            return;
        }
        match event {
            HostData::Reset => {
                self.multipoles.clear();
                self.locals.clear();
                self.target_output.clear();
            }
            HostData::SendSources { leaves, sources } => {
                for &j in leaves {
                    let j = j as usize;
                    sources.chunk_mut(j).copy_from_slice(self.sources.chunk(j));
                }
            }
            HostData::ReceivedSources { leaves, sources } => {
                self.sources
                    .range_mut(leaves.clone())
                    .as_mut_slice()
                    .copy_from_slice(sources.range(leaves).as_slice());
            }
            HostData::SendMultipoles {
                coarse,
                exchange,
                multipoles,
            } => {
                for b in coarse.sent_blocks() {
                    let (level, i) = coarse.block(b);
                    let i = i as usize;
                    multipoles
                        .chunk_mut(level, i)
                        .copy_from_slice(self.multipoles.chunk(level, i));
                }
                for level in 0..exchange.nlevels() {
                    for &i in exchange.send_boxes(level) {
                        let i = i as usize;
                        multipoles
                            .chunk_mut(level, i)
                            .copy_from_slice(self.multipoles.chunk(level, i));
                    }
                }
            }
            HostData::ReceivedCoarse { coarse, .. } => {
                for b in coarse.received_blocks() {
                    let (level, i) = coarse.block(b);
                    self.multipoles
                        .chunk_mut(level, i as usize)
                        .copy_from_slice(coarse.chunk(b));
                }
            }
            HostData::ReceivedMultipoles {
                level, exchange, ..
            } => {
                let size = exchange.size(level);
                let values = exchange.receive_buffer(level).chunks_exact(size);
                for (&i, chunk) in exchange.receive_boxes(level).iter().zip(values) {
                    self.multipoles
                        .chunk_mut(level, i as usize)
                        .copy_from_slice(chunk);
                }
            }
        }
    }
}

/// The counts of every leaf of `store`: for the evaluator's source store, the local
/// counts and the ghost leaves' counts that its source exchange delivered.
pub fn counts<T: Copy + Default>(store: &LeafStore<T>) -> Vec<usize> {
    (0..store.nleaves()).map(|j| store.count(j)).collect()
}

/// The outcome of [`check_fmm`], summed over every rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// Output values that differ from the `Fmm`'s, over every evaluation.
    pub differing: usize,
    /// Output values compared.
    pub values: usize,
    /// Values of the evaluator's own target output and locals that are not zero (the
    /// shadow never writes the evaluator's slices, so there must be none).
    pub host_written: usize,
    /// Ranks on which an evaluation did not receive 5 + nlevels events.
    pub event_counts: usize,
}

/// The sum of `values` over every rank.
fn global_sums<const N: usize>(values: [usize; N], comm: &SimpleCommunicator) -> [usize; N] {
    let mut total = [0usize; N];
    comm.all_reduce_into(&values[..], &mut total[..], SystemOperation::sum());
    total
}

/// The keys of `points` on the deepest level, as `Fmm::build` computes them.
fn keys(
    points: &[[f64; 3]],
    fmm: &Fmm<'_, impl Stored + SimdScalar + Equivalence + Default>,
) -> Vec<MortonKey> {
    let domain = fmm.domain().physical_box();
    points
        .iter()
        .map(|&x| morton::from_physical_point(x, &domain, DEEPEST_LEVEL as usize))
        .collect()
}

/// The shadow check of `fmm`, built from this rank's `sources` and `targets` with one
/// thread on the host: the evaluator with a [`Shadow`] of `fmm`'s operator (`hook` on or
/// off) on `fmm`'s plan, every `(charges, output)` of `evaluations` evaluated in turn on
/// the same evaluator (so a second one depends on `Reset`), each output compared with the
/// `Fmm`'s bit for bit. Collective; the verdict is the same on every rank.
pub fn check_fmm<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
    (sources, targets): (&[[f64; 3]], &[[f64; 3]]),
    evaluations: &[(&[T], &Output<T>)],
    hook: bool,
    comm: &SimpleCommunicator,
) -> Verdict {
    let (octree, plan, domain) = (fmm.octree(), fmm.plan(), fmm.domain());
    let index = plan.index();
    let leaves = index.leaves();
    let nlocal = leaves.nlocal();
    // The moves of `Fmm::build`, routed by the same keys: the points to the ranks that own
    // their leaves, in leaf order.
    let source_route =
        Redistribution::new(octree, plan, &keys(sources, fmm)).expect("the sources route");
    let target_route =
        Redistribution::new(octree, plan, &keys(targets, fmm)).expect("the targets route");
    assert_eq!(source_route.counts(), fmm.source_counts());
    assert_eq!(target_route.counts(), fmm.target_counts());
    let scaled = |route: &Redistribution, points: &[[f64; 3]]| -> Vec<T> {
        let owned = route.forward(points.as_flattened(), 3);
        let mut owned = owned.as_chunks::<3>().0.iter();
        let mut scaled = Vec::with_capacity(3 * route.nreceived());
        for (j, &count) in route.counts().iter().enumerate() {
            for &x in owned.by_ref().take(count) {
                scaled.extend(leaf_coordinates::<T>(x, leaves.key(j), domain));
            }
        }
        scaled
    };
    let (source_points, target_points) = (
        scaled(&source_route, sources),
        scaled(&target_route, targets),
    );

    let operator = fmm.operator().clone();
    assert_eq!(operator.threads(), 1, "the shadow check runs on one thread");
    let gradients = operator.gradients();
    let mut evaluator = Evaluator::new(
        plan,
        comm,
        Shadow::new(operator, hook),
        fmm.source_counts(),
        fmm.target_counts(),
    )
    .expect("the shadow's evaluator builds");
    let all_counts = counts(evaluator.source_store());
    let shadow = evaluator.operator_mut();
    shadow.attach(index, &all_counts, fmm.target_counts());
    shadow.load_points(&source_points, &target_points);

    let mut verdict = Verdict::default();
    for &(charges, output) in evaluations {
        let charges = source_route.forward(charges, 1);
        let before = evaluator.operator().events();
        evaluator.operator_mut().begin_evaluation(&charges);
        evaluator.evaluate();
        let events = evaluator.operator().events() - before;
        verdict.event_counts += usize::from(events != 5 + plan.nlevels());
        let host = evaluator.target_output_store().as_slice().iter();
        verdict.host_written += host
            .chain(evaluator.locals().as_slice())
            .filter(|&&v| v != T::zero())
            .count();

        // The output pass of `Fmm::reference_output`: φ̂ / (4π r_t), ĝ / (4π r_t²) in f64,
        // then back to the caller's order.
        let store = evaluator.operator().target_output();
        let ntargets = target_route.nreceived();
        let mut potential = Vec::with_capacity(ntargets);
        let mut gradient = Vec::with_capacity(3 * ntargets);
        for j in 0..nlocal {
            let n = store.count(j);
            let r = radius(leaves.level(j), domain);
            let (phi_scale, g_scale) = (
                4.0 * std::f64::consts::PI * r,
                4.0 * std::f64::consts::PI * r * r,
            );
            let (phi, g) = store.chunk(j).split_at(n);
            potential.extend(
                phi.iter()
                    .map(|&v| T::from_f64(RealScalar::to_f64(v) / phi_scale)),
            );
            if gradients {
                gradient.extend(
                    g.iter()
                        .map(|&v| T::from_f64(RealScalar::to_f64(v) / g_scale)),
                );
            }
        }
        let potential = target_route.backward(&potential, 1);
        let mut got: Vec<T> = potential;
        let mut want: Vec<T> = output.potential.clone();
        if gradients {
            got.extend(target_route.backward(&gradient, 3));
            want.extend_from_slice(output.gradient.as_ref().expect("gradients").as_flattened());
        }
        assert_eq!(got.len(), want.len());
        verdict.values += got.len();
        verdict.differing += got
            .iter()
            .zip(&want)
            .filter(|(a, b)| RealScalar::to_f64(**a).to_bits() != RealScalar::to_f64(**b).to_bits())
            .count();
    }
    let [differing, values, host_written, event_counts] = global_sums(
        [
            verdict.differing,
            verdict.values,
            verdict.host_written,
            verdict.event_counts,
        ],
        comm,
    );
    Verdict {
        differing,
        values,
        host_written,
        event_counts,
    }
}
