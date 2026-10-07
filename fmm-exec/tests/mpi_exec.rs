//! The MPI scenarios of `nd-fmm-exec`. Keep MPI initialisation in this one test: MPI
//! cannot be initialised again after finalisation within the same process. Add
//! scenarios to `cases`, not new `#[test]`s (T9, T10 and T11 extend it).
//!
//! CI runs the scenarios on one rank. Each one is written for any rank count: every
//! rank generates the same global point set, passes its share of the keys to the
//! octree, and loads the points that fall into its own leaves, so a run by hand on
//! several ranks (under an external timeout, with the loopback flags of the root
//! `CLAUDE.md` on macOS) exercises ghosts and the global levels.
//!
//! Scenarios:
//! - **table order against the plan** (CONVENTIONS §3.12): on a uniform level-3 tree in
//!   the dyadic domain, every (level, offset) batch of the V-list view has the offset
//!   index of its direction, both as its position in `V_LIST_DIRECTIONS` and as
//!   `m2l_offset_index`; every octant batch of the M2M and L2L views has the child
//!   index of its children; and the centre difference of every V pair, from
//!   `morton::physical_box`, is 2 r_l d exactly.
//! - **batched against per-pair**: on a small adaptive tree with W and X lists, points
//!   loaded by hand in the leaf-scaled layout of §3.13, the evaluator with
//!   `LaplaceOperator` and with `PerPair<LaplaceOperator>` gives bit-identical target
//!   output. As a smoke check (not the C3.2 gate, which is T9's), its potentials and
//!   gradients match the direct sum to a loose tolerance.
//!
//! The scenarios of `Fmm` (T9) compare with `direct_sum` in f64 over all sources at
//! every target, divided by 4π (the oracle expands 1/|x − y|, `Fmm` returns
//! Σ q / (4π |x − y|)). Error measure: relative L2 error over all targets of all ranks
//! (docs/phase3/README.md, "Error measures"), charges uniform in [−1, 1).
//! - **uniform cube, every strategy**: N = 2,000, sources equal to targets, the default
//!   tree; for p ∈ {2, 4, 6, 8} and `Dense`, `Classes` and `Rotation`, the error
//!   decreases with p and is below 1e-4 at p = 8, and the three strategies agree to
//!   1e-12 (relative L2 between them).
//! - **distinct and nested point sets**: sources and targets disjoint, and the sources
//!   a subset of the targets, p = 6.
//! - **gradients**: ∇φ against the `direct_sum` gradient, p = 6; the potentials equal
//!   those of the FMM without gradients bit for bit with the reference P2P, and to the
//!   tolerance of an ISA against the reference with the SIMD kernel (its potential-only
//!   loop contracts φ + q ρ into one fma).
//! - **uniform tree**: `max_level` 3, `max_points_per_leaf` 1, four points in every
//!   level-3 box of a supplied unit-cube domain: every leaf lies on level 3, the W and X
//!   lists are empty.
//! - **single leaf**: `max_level` 0, the root the only leaf: P2P alone, equal to the
//!   direct sum to 1e-14. One rank only (a root-only coarse tree has one block).
//! - **no targets on this rank**: rank 0 passes no targets and gets empty output.
//! - **input errors**: a supplied non-cubic domain, a point outside a supplied domain,
//!   p > 20, a non-finite point and a charge vector of the wrong length each give
//!   their `FmmError`, agreed on every rank.
//! - **f32 against f64**: p = 6; the f32 output against the f64 output, and both
//!   against the direct sum.
//! - **repeatability**: `evaluate` twice gives bit-identical output, and a second charge
//!   vector gives bit for bit the output of a fresh build with it.
//! - **ownership**: every rank passes the same complete point set. On one rank the FMM
//!   runs; on several, every rank returns `PointsNotOwned` with the same count.
//!
//! Threads (T10, C3.5). MPI is initialised with `Threading::Funneled`, so that `Fmm` may
//! use threads. Error measure: exact equality of the bit patterns.
//! - Every `Fmm` evaluation above runs at one thread and again in a fresh build at 2, 4
//!   and 8 threads, through `evaluate_threaded`; every output equals the one-thread
//!   output bit for bit. Each threaded build reports its threads and MPI level
//!   (`Fmm::threading`), and the P2P buffer of every per-thread scratch set keeps its
//!   capacity through the evaluation (no allocation per pair). These builds share a
//!   table cache in `CARGO_TARGET_TMPDIR`, which loads tables bit for bit, so each
//!   table set is built once: in debug mode the table builds, not the evaluations,
//!   would otherwise take most of the minute the test may run.
//! - **batched against per-pair** also runs the operator with a pool of 2, 4 and 8
//!   threads (`LaplaceOperator::with_pool`), bit-identical to the serial operator.
//! - **threads, every strategy and precision**: an adaptive tree with W and X lists, in
//!   f32 and f64, `Dense`, `Classes` and `Rotation`, gradients off and on, at 1, 2, 4
//!   and 8 threads.
//! - **repeatability** also checks that two threaded evaluations agree bit for bit, and
//!   a threaded evaluation after a serial one (`Fmm::set_serial`) on the same `Fmm`.
//! - **input errors** also checks that `threads(0)` is rejected.
//!
//! Adaptive trees (T11, C3.3), where the W and X lists (M2P and P2L) are non-empty. Error
//! measure: as for T9, relative L2 against `direct_sum` divided by 4π over all targets,
//! unless stated otherwise. The distributions follow the rules of
//! `nd_fmm_validate::points` (this crate cannot depend on it), scaled to N = 2,000.
//! - **clustered distributions**: points on a sphere surface, in a Plummer sphere and in
//!   two Gaussian clusters, sources equal to targets, each with the largest refinement
//!   target of 16, 32 and 64 points per leaf at which its leaves lie on three levels.
//!   The leaves span at least three levels and W and X are non-empty. At p = 4 `Dense`,
//!   `Classes` and `Rotation` agree to 1e-12 (relative L2 between them); at p = 4 and 8
//!   the error is within twice that of the uniform cube at the same N (the C3.3 rule).
//!   In debug mode every strategy at p = 8 would take the whole minute, so p = 8 runs
//!   `Dense` alone here; every strategy at p = 8 on the trees of C3.3 is checked by the
//!   ignored `tests/adaptive.rs`.
//! - **strongly graded tree**: a dense blob of side 2⁻⁶ next to a sparse cloud, leaves
//!   on levels six or more apart, p = 8, φ and ∇φ.
//! - **coincident points in a level-16 leaf**: four refinement targets of points per
//!   leaf, and one level-16 leaf with 100 points (ten copies each of four positions, and
//!   60 points within 10⁻³ of its half-width of its centre) next to its seven siblings,
//!   in the unit domain, where the leaf-scaled coordinates are exact. The cloud around
//!   it carries no charge, so at the cluster every contribution is a P2P term: the
//!   largest error there, relative to the sum of the term magnitudes, is below 1e-14.
//!   Every output is finite, and copies of a point get the same output bit for bit
//!   (coincident pairs are excluded, as in `direct_sum`); the far field of the cluster
//!   at the cloud matches to 1e-2 (p = 4).
//! - **sources-only next to targets-only leaves**: sources and targets disjoint, by the
//!   parity of their level-2 cell, so every leaf holds one kind; some of each kind are
//!   U-list neighbours. p = 6, φ and ∇φ.
//! - **points on box faces and domain corners**: the points of the unit domain closest
//!   to its 8 corners, 12 edge midpoints and 6 face centres, and points with dyadic
//!   coordinates j / 2^m (m = 1–10) on the faces, edges and corners of boxes, half of
//!   them in a corner blob. p = 6, φ and ∇φ.
//!
//! Each runs through `evaluate_threaded` at 2, 4 and 8 threads, except the p = 8
//! evaluations and `Classes` and `Rotation` at p = 4 (one thread), to keep the debug
//! run under a minute.
//!
//! P2P kernels (Phase 3S T6, C3S.5). Every `Fmm` above runs the default kernel,
//! `P2pChoice::Auto` (the SIMD kernel on the ISA of `Isa::detect`). Error measures: exact
//! equality of the bit patterns, and the relative L2 difference of potential and
//! gradient from the `Reference` output over all ranks, 1e-13 in f64 and 1e-6 in f32.
//! - **batched against per-pair** runs the operator with `Reference` and with every
//!   ISA the machine offers: per-pair and 2, 4 and 8 threads equal the batched output
//!   bit for bit for each.
//! - `evaluate_every_kernel` repeats a scenario with `Reference` and every available
//!   ISA, each at 1, 2, 4 and 8 threads bit for bit, checks that `Auto` gives the output
//!   of the detected ISA bit for bit and that every ISA lies within the tolerance of
//!   `Reference`, and prints the kernels it ran. It runs in **gradients**, **single
//!   leaf**, **no targets on this rank**, **f32 against f64**, **threads, every
//!   strategy and precision** (`Dense`, f32 and f64, gradients off and on), **coincident
//!   points in a level-16 leaf** and **points on box faces and domain corners**: P2P
//!   alone, both precisions and both outputs, the coincident-pair rule and the extreme
//!   coordinates. The other scenarios run `Auto` only, to keep the debug run under a
//!   minute; the ignored `tests/accuracy.rs` and `tests/adaptive.rs` compare `Auto` with
//!   `Reference` on the large problems of C3.2 and C3.3.
//! - **input errors** also checks that a P2P kernel on an ISA the machine cannot run is
//!   rejected with `SettingsError::P2pIsaUnavailable`.
//!
//! Device path (Phase 4 T5, C4.1; T6, C4.2; T7, C4.3; T8, C4.4), with the `gpu` feature. Error
//! measures:
//! exact equality of the bit patterns, counts and bytes, and the relative L2 difference
//! of φ and ∇φ from the host output over all targets.
//! - Every `evaluate_threaded` call above, and so every `Fmm` scenario (uniform and
//!   adaptive trees, gradients on and off, empty leaves, coincident points, every
//!   strategy, f32 and f64, every P2P kernel of `evaluate_every_kernel`), is repeated at
//!   one thread on every device backend compiled in (`device_backends`; the CPU
//!   runtime, while Metal runs in the ignored `tests/device_metal.rs`), through
//!   `device_common::check_backend`, twice:
//!   - with every operator kind on the host fallback: the output equals the host path's
//!     bit for bit for the scenario's charges, a second charge vector and the first
//!     again; the transfers, launches and syncs of each evaluation equal the formula of
//!     docs/design/device-path.md §4.1 and §7.2; no evaluation moves points, views,
//!     geometry or tables; and every view on the device equals the plan's;
//!   - with the default placement (T10: every kind on the device under every strategy;
//!     `Classes` runs as dense on the device, `Rotation` M2L by the rotation kernel): the
//!     output within the FMM bounds of the host output (1e-12 in f64, 1e-5 in f32), and
//!     so the multipoles and locals of every level (relative L2 per level; on one rank
//!     the root's multipole is the global pass's M2M), two evaluations bit-identical, and
//!     the transfers of the formula with each device kind's fallback transfers replaced by
//!     one launch per level call, three per chunk for M2M, L2L and dense M2L.
//!
//!   On several ranks the device build returns `DeviceNeedsOneRank` on every rank. The
//!   test prints the backends it ran, and the largest differences, at the end.
//! - **device backends**: `Host` is the default and reports every kind on the host; a
//!   backend not compiled in gives `BackendNotCompiled`, also when only rank 0 asks for
//!   it (the others return `OtherRank`: the check rides on step 1's agreement); with the
//!   CPU runtime, `threads(4)` builds no rayon pool and caps the units per cube at 4
//!   (device-path.md §11), with the output of one unit bit for bit and within 1e-12 of
//!   the host's; the cube layout of the leaf operators (up to 8 units, one per core,
//!   tiles of up to 4 points) within 1e-12 of the host's too; M2M, L2L and M2L with a
//!   scratch budget of one column per chunk (as many chunks as pairs) and with the hand-written
//!   GEMM bit for bit the default; `synchronous_stages` adds seven syncs (after the
//!   charge upload and each stage) and changes no bit; on several ranks a device build with
//!   points other ranks own returns `PointsNotOwned`, which wins over
//!   `DeviceNeedsOneRank`, and one with points every rank owns (each rank passing its
//!   share of the level-1 octants of a uniform level-3 tree) returns
//!   `DeviceNeedsOneRank` on every rank.
//!
//! Every `Fmm` scenario but **ownership** passes each rank its share of the points (every
//! `size`-th). On several ranks those points generally lie in leaves of other ranks;
//! the scenario then checks that every rank returns `PointsNotOwned` and stops, since
//! points are not redistributed until C5.1. If the build succeeds instead, the
//! checks run distributed.

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use nd_fmm_exec::fmm::{
    Backend, Fmm, FmmBuilder, FmmError, OperatorKind, Output, Placement, PointSet, SettingsError,
};
use nd_fmm_exec::geometry::{Domain, GeometryError, leaf_coordinates, radius};
use nd_fmm_exec::operator::{Isa, LaplaceOperator, P2pChoice, SimdScalar};
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_math::RealScalar;
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS;
use nd_fmm_plan::lists::GroupedCsr;
use nd_fmm_plan::operator::{FmmOperator, PerPair};
use nd_fmm_plan::plan::Plan;
use nd_fmm_ref::p2p::direct_sum;
use nd_fmm_tables::cache::Stored;
use nd_fmm_tables::geometry::m2l_offset_index;
use nd_octree::{MortonKey, Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, morton};

#[cfg(feature = "gpu")]
mod device_common;

type Scenario = fn(&SimpleCommunicator);

/// The device runs of `evaluate_threaded` per backend compiled in (`device_backends`),
/// for the closing report line.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug)]
struct DeviceRuns {
    backend: Backend,
    f32_runs: usize,
    f64_runs: usize,
    /// Builds refused with `DeviceNeedsOneRank` on several ranks.
    refused: usize,
    /// The largest relative L2 differences of the default placement from the host path,
    /// [φ, ∇φ, multipoles, locals, the root's multipole], in f32 and in f64.
    worst: [[f64; 5]; 2],
}

#[cfg(feature = "gpu")]
static DEVICE_RUNS: std::sync::Mutex<Vec<DeviceRuns>> = std::sync::Mutex::new(Vec::new());

/// The device backends every `Fmm` scenario is repeated on: the CPU runtime if it is
/// compiled in. Metal runs in its own ignored executable (`tests/device_metal.rs`),
/// outside the macOS sandbox, and CUDA in its own (`tests/device_cuda.rs`), by hand on
/// locust.
#[cfg(feature = "gpu")]
fn device_backends() -> Vec<Backend> {
    [Backend::Cpu]
        .into_iter()
        .filter(|b| b.is_compiled())
        .collect()
}

/// The device runs of one backend for [`backends_line`]: the backend, f32 and f64 runs,
/// refused builds, and the largest differences of the default placement in f32 and f64
/// ([φ, ∇φ, multipoles, locals, root]).
type Runs = (Backend, usize, usize, usize, [[f64; 5]; 2]);

/// "backends run: …; not run: …" for this test.
fn backends_line() -> String {
    #[cfg(feature = "gpu")]
    let runs: Vec<Runs> = DEVICE_RUNS
        .lock()
        .unwrap()
        .iter()
        .map(|r| (r.backend, r.f32_runs, r.f64_runs, r.refused, r.worst))
        .collect();
    #[cfg(not(feature = "gpu"))]
    let runs: Vec<Runs> = Vec::new();
    let mut ran = vec!["host (every scenario)".to_owned()];
    ran.extend(
        runs.iter()
            .filter(|(_, f32_runs, f64_runs, ..)| f32_runs + f64_runs > 0)
            .map(|(backend, f32_runs, f64_runs, _, worst)| {
                format!(
                    "{backend} ({f32_runs} f32 and {f64_runs} f64 scenarios: on the host \
                     fallback bit for bit; with every kind on the device (M2L by rotation \
                     under Rotation) within the FMM bounds, largest relative L2 difference φ {:.1e} / ∇φ {:.1e}, per \
                     level multipoles {:.1e} (root, the global M2M: {:.1e}) / locals {:.1e} \
                     (f32); {:.1e} / {:.1e}, {:.1e} ({:.1e}) / {:.1e} (f64))",
                    worst[0][0],
                    worst[0][1],
                    worst[0][2],
                    worst[0][4],
                    worst[0][3],
                    worst[1][0],
                    worst[1][1],
                    worst[1][2],
                    worst[1][4],
                    worst[1][3]
                )
            }),
    );
    let refused: Vec<String> = runs
        .iter()
        .filter(|(_, f32_runs, f64_runs, refused, _)| f32_runs + f64_runs == 0 && *refused > 0)
        .map(|(backend, _, _, refused, _)| {
            format!("{backend} (DeviceNeedsOneRank on every rank, {refused} scenario(s))")
        })
        .collect();
    let not_run: Vec<String> = Backend::ALL
        .into_iter()
        .filter(|b| b.is_device() && !runs.iter().any(|(r, ..)| r == b))
        .map(|b| match (b, b.is_compiled()) {
            (_, false) => format!("{b} (not compiled)"),
            (Backend::Metal, true) => format!("{b} (ignored test tests/device_metal.rs)"),
            (Backend::Cuda, true) => format!("{b} (ignored test tests/device_cuda.rs)"),
            _ => format!("{b} (not run)"),
        })
        .chain(refused)
        .collect();
    format!(
        "backends run: {}; not run: {}",
        ran.join(", "),
        not_run.join(", ")
    )
}

/// The table cache of `evaluate_threaded`, in the scratch directory Cargo provides for
/// integration tests.
const TABLE_CACHE: &str = concat!(env!("CARGO_TARGET_TMPDIR"), "/mpi_exec_tables");

/// The thread counts every `Fmm` scenario is repeated at, besides one thread.
const THREADS: [usize; 3] = [2, 4, 8];

#[test]
fn distributed_scenarios() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(
        provided >= Threading::Funneled,
        "the threaded scenarios need MPI at Funneled, it provides {provided:?}"
    );
    let comm = universe.world();
    let cases: [(&str, Scenario); 19] = [
        ("table order against the plan", table_order),
        ("batched against per-pair", batched_against_per_pair),
        ("uniform cube, every strategy", uniform_cube_strategies),
        ("distinct and nested point sets", distinct_and_nested),
        ("gradients", gradients),
        ("uniform tree", uniform_tree),
        ("single leaf", single_leaf),
        ("no targets on this rank", no_targets_on_this_rank),
        ("input errors", input_errors),
        ("f32 against f64", f32_against_f64),
        ("repeatability", repeatability),
        ("ownership", ownership),
        (
            "threads, every strategy and precision",
            threads_every_strategy,
        ),
        ("clustered distributions", clustered_distributions),
        ("strongly graded tree", strongly_graded_tree),
        ("coincident points in a level-16 leaf", coincident_points),
        ("sources-only next to targets-only leaves", one_sided_leaves),
        ("points on box faces and domain corners", faces_and_corners),
        ("device backends", device_backends_scenario),
    ];
    for (name, scenario) in cases {
        eprintln!("rank {}: {name}", comm.rank());
        let start = std::time::Instant::now();
        scenario(&comm);
        eprintln!(
            "rank {}: {name}: {:.1} s",
            comm.rank(),
            start.elapsed().as_secs_f64()
        );
    }
    eprintln!("rank {}: {}", comm.rank(), backends_line());
}

/// The dyadic domain of Phase 2, a = (−1.25, 0.5, 2) and w = 3: every centre, centre
/// difference and half-width on levels 0–16 is exact in f64.
fn dyadic_domain() -> Domain {
    Domain::new(&PhysicalBox::new([-1.25, 0.5, 2.0, 1.75, 3.5, 5.0])).unwrap()
}

/// SplitMix64, as in the operator tests.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }

    /// A standard normal number: Box–Muller, the cosine branch.
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * (1.0 - self.range(0.0, 1.0)).ln()).sqrt();
        radius * (2.0 * std::f64::consts::PI * self.range(0.0, 1.0)).cos()
    }

    /// A direction uniform on the unit sphere, by rejection in the unit ball.
    fn direction(&mut self) -> [f64; 3] {
        loop {
            let v: [f64; 3] = core::array::from_fn(|_| self.range(-1.0, 1.0));
            let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if (1e-3..=1.0).contains(&norm) {
                return v.map(|vk| vk / norm);
            }
        }
    }
}

/// The octree of this rank's share of `points` (every `size`-th point from `rank`), with
/// the ghost-children layer that `Plan::new` requires.
fn octree<'c>(
    points: &[[f64; 3]],
    domain: &Domain,
    max_level: usize,
    capacity: usize,
    comm: &'c SimpleCommunicator,
) -> Octree<'c, SimpleCommunicator> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    let fine_keys: Vec<MortonKey> = points
        .iter()
        .skip(rank)
        .step_by(size)
        .map(|&x| morton::from_physical_point(x, &domain.physical_box(), DEEPEST_LEVEL as usize))
        .collect();
    let options = OctreeOptions::new()
        .with_max_level(max_level)
        .with_max_fine_keys(capacity)
        .with_ghost_children(true);
    Octree::new(&fine_keys, options, comm)
}

/// The midpoint of `morton::physical_box(key)`, per axis.
fn midpoint(key: MortonKey, domain: &Domain) -> [f64; 3] {
    let c = morton::physical_box(key, &domain.physical_box()).coordinates();
    core::array::from_fn(|k| (c[k] + c[k + 3]) / 2.0)
}

/// Checks the octant of every pair of a parent–child view, rows and batches: `child`
/// and `parent` give the keys of an entry's child and parent.
fn check_octants<G: Copy + Into<usize>>(
    view: &GroupedCsr<G>,
    pair: impl Fn(usize, usize) -> (MortonKey, MortonKey),
) -> usize {
    for t in 0..view.nrows() {
        let (sources, groups) = view.row(t);
        for (&s, &o) in sources.iter().zip(groups) {
            let (child, parent) = pair(t, s as usize);
            assert_eq!(morton::child_index(child), o.into(), "row {t}");
            assert_eq!(morton::parent(child), Some(parent));
        }
    }
    let mut pairs = 0;
    for o in 0..view.ngroups() {
        let (targets, sources) = view.batch(o);
        for (&t, &s) in targets.iter().zip(sources) {
            let (child, parent) = pair(t as usize, s as usize);
            assert_eq!(morton::child_index(child), o, "octant batch {o}");
            assert_eq!(morton::parent(child), Some(parent));
            pairs += 1;
        }
    }
    pairs
}

/// The table order of `nd-fmm-tables` against the views of the plan (module
/// documentation). Error measure: exact equality (integers, and centre differences that
/// are exact in the dyadic domain).
fn table_order(comm: &SimpleCommunicator) {
    let domain = dyadic_domain();
    // One point at the centre of every level-3 box: a uniform level-3 tree.
    let points: Vec<[f64; 3]> = (0..512)
        .map(|j| {
            let index = [j >> 6, (j >> 3) & 7, j & 7];
            midpoint(morton::from_index_and_level(index, 3), &domain)
        })
        .collect();
    let octree = octree(&points, &domain, 3, 1, comm);
    let plan = Plan::new(&octree).expect("the plan builds");
    let index = plan.index();
    let mut counts = [0usize; 3];
    for level in 0..plan.nlevels() {
        let lists = plan.level(level);
        let r = radius(level, &domain);
        let v = lists.v();
        let check_pair = |t: usize, s: usize, d: usize| {
            let (target, source) = (index.key(level, t), index.key(level, s));
            let ((_, ti), (_, si)) = (morton::decode(target), morton::decode(source));
            let offset: [i64; 3] = core::array::from_fn(|k| ti[k] as i64 - si[k] as i64);
            assert_eq!(
                offset, V_LIST_DIRECTIONS[d],
                "level {level}, offset index {d}"
            );
            assert_eq!(m2l_offset_index(offset), Some(d));
            let (ct, cs) = (midpoint(target, &domain), midpoint(source, &domain));
            let shift: [f64; 3] = core::array::from_fn(|k| ct[k] - cs[k]);
            assert_eq!(shift, offset.map(|dk| 2.0 * r * dk as f64), "level {level}");
        };
        for t in 0..v.nrows() {
            let (sources, offsets) = v.row(t);
            for (&s, &d) in sources.iter().zip(offsets) {
                check_pair(t, s as usize, d as usize);
            }
        }
        for d in 0..v.ngroups() {
            let (targets, sources) = v.batch(d);
            for (&t, &s) in targets.iter().zip(sources) {
                check_pair(t as usize, s as usize, d);
                counts[0] += 1;
            }
        }
        for children in [lists.m2m_local(), lists.m2m_global()] {
            counts[1] += check_octants(children, |t, c| {
                (index.key(level + 1, c), index.key(level, t))
            });
        }
        counts[2] += check_octants(lists.l2l(), |t, p| {
            (index.key(level, t), index.key(level - 1, p))
        });
    }
    let mut total = [0usize; 3];
    comm.all_reduce_into(&counts[..], &mut total[..], SystemOperation::sum());
    eprintln!(
        "rank {}: table order: {} V pairs, {} M2M pairs, {} L2L pairs checked ({} in all)",
        comm.rank(),
        counts[0],
        counts[1],
        counts[2],
        total.iter().sum::<usize>()
    );
    assert!(
        total.iter().all(|&n| n > 0),
        "every view has pairs: {total:?}"
    );
}

/// The points of the adaptive scenario: a cloud over the whole domain and a dense blob
/// near one corner, so the tree is graded and has W and X lists.
fn adaptive_points(domain: &Domain) -> Vec<[f64; 3]> {
    let mut rng = SplitMix64(0x7891);
    let (a, w) = (domain.lower(), domain.side());
    let mut points: Vec<[f64; 3]> = (0..80)
        .map(|_| core::array::from_fn(|k| a[k] + w * rng.range(0.001, 0.999)))
        .collect();
    points.extend(
        (0..80).map(|_| -> [f64; 3] { core::array::from_fn(|k| a[k] + w * rng.range(0.03, 0.09)) }),
    );
    points
}

/// The points of `points` that lie in each local leaf of `index`, in point order.
fn points_by_leaf(points: &[[f64; 3]], index: &BoxIndex, domain: &Domain) -> Vec<Vec<usize>> {
    let mut leaves = vec![Vec::new(); index.leaves().nlocal()];
    for (j, &x) in points.iter().enumerate() {
        let fine = morton::from_physical_point(x, &domain.physical_box(), DEEPEST_LEVEL as usize);
        if let Some(leaf) = index.local_leaf_containing(fine) {
            leaves[leaf as usize].push(j);
        }
    }
    leaves
}

/// Runs the evaluator with `operator` on the points loaded by leaf; returns the target
/// output store, every local leaf after another.
fn evaluate<Op: FmmOperator<Value = f64>>(
    plan: &Plan,
    comm: &SimpleCommunicator,
    operator: Op,
    domain: &Domain,
    (sources, charges, source_leaves): (&[[f64; 3]], &[f64], &[Vec<usize>]),
    (targets, target_leaves): (&[[f64; 3]], &[Vec<usize>]),
) -> Vec<f64> {
    let index = plan.index();
    let source_counts: Vec<usize> = source_leaves.iter().map(Vec::len).collect();
    let target_counts: Vec<usize> = target_leaves.iter().map(Vec::len).collect();
    let mut evaluator = Evaluator::new(plan, comm, operator, &source_counts, &target_counts)
        .expect("the evaluator builds");
    for (leaf, (in_sources, in_targets)) in source_leaves.iter().zip(target_leaves).enumerate() {
        let key = index.leaf_key(leaf);
        let chunk = evaluator.sources_mut(leaf);
        let n = in_sources.len();
        for (r, &j) in in_sources.iter().enumerate() {
            chunk[3 * r..3 * r + 3]
                .copy_from_slice(&leaf_coordinates::<f64>(sources[j], key, domain));
            chunk[3 * n + r] = charges[j];
        }
        let chunk = evaluator.target_input_mut(leaf);
        for (r, &j) in in_targets.iter().enumerate() {
            chunk[3 * r..3 * r + 3]
                .copy_from_slice(&leaf_coordinates::<f64>(targets[j], key, domain));
        }
    }
    evaluator.evaluate();
    evaluator.target_output_store().as_slice().to_vec()
}

/// The batched operator against the per-pair adapter, and a smoke check against the
/// direct sum (module documentation).
fn batched_against_per_pair(comm: &SimpleCommunicator) {
    let domain = dyadic_domain();
    let sources = adaptive_points(&domain);
    let mut rng = SplitMix64(0x7892);
    let charges: Vec<f64> = sources.iter().map(|_| rng.range(-1.0, 1.0)).collect();
    // The targets: the sources (coincident pairs excluded) and a few other points.
    let mut targets = sources.clone();
    let (a, w) = (domain.lower(), domain.side());
    targets.extend(
        (0..20)
            .map(|_| -> [f64; 3] { core::array::from_fn(|k| a[k] + w * rng.range(0.001, 0.999)) }),
    );

    let octree = octree(&sources, &domain, 6, 4, comm);
    let plan = Plan::new(&octree).expect("the plan builds");
    let index = plan.index();
    let source_leaves = points_by_leaf(&sources, index, &domain);
    let target_leaves = points_by_leaf(&targets, index, &domain);

    // The tree must exercise every list.
    let local = [
        (0..plan.nlevels())
            .map(|l| plan.level(l).w().len())
            .sum::<usize>(),
        (0..plan.nlevels())
            .map(|l| plan.level(l).x().len())
            .sum::<usize>(),
        (0..plan.nlevels())
            .map(|l| plan.level(l).v().len())
            .sum::<usize>(),
    ];
    let mut lists = [0usize; 3];
    comm.all_reduce_into(&local[..], &mut lists[..], SystemOperation::sum());
    assert!(lists.iter().all(|&n| n > 0), "W, X and V pairs: {lists:?}");

    // The scratch must hold the largest source leaf of any rank.
    let local_max = source_leaves.iter().map(Vec::len).max().unwrap_or(0);
    let mut max_leaf_points = 0usize;
    comm.all_reduce_into(&local_max, &mut max_leaf_points, SystemOperation::max());

    let p = 6;
    let op = LaplaceOperator::new(Tables::build(p, M2lStrategy::Auto), true, max_leaf_points);
    assert_eq!(
        op.p2p_kernel(),
        P2pChoice::Isa(Isa::detect()),
        "Auto by default"
    );
    let input = (&sources[..], &charges[..], &source_leaves[..]);
    for choice in p2p_choices() {
        let op = op
            .clone()
            .with_p2p(choice)
            .expect("an available P2P kernel");
        let batched = evaluate(
            &plan,
            comm,
            op.clone(),
            &domain,
            input,
            (&targets, &target_leaves),
        );
        // The same operator with a pool: bit-identical to the serial operator (C3.5).
        for n in THREADS {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("the pool builds");
            let threaded = op.clone().with_pool(std::sync::Arc::new(pool));
            assert_eq!(threaded.threads(), n);
            let output = evaluate(
                &plan,
                comm,
                threaded,
                &domain,
                input,
                (&targets, &target_leaves),
            );
            assert_eq!(
                bits(&output),
                bits(&batched),
                "{choice}: {n} threads against serial"
            );
        }
        let per_pair = evaluate(
            &plan,
            comm,
            PerPair(op),
            &domain,
            input,
            (&targets, &target_leaves),
        );
        assert_eq!(batched.len(), per_pair.len());
        let differing = batched
            .iter()
            .zip(&per_pair)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "{choice}: batched and per-pair output differ in {differing} values"
        );

        // Smoke check against the direct sum over all sources: φ = φ̂ / r_t, ∇φ = ĝ / r_t²
        // (1/|x − y| units, no 4π), relative L2 over the local targets.
        let mut offset = 0;
        let mut sums = [0.0f64; 4];
        for (leaf, in_targets) in target_leaves.iter().enumerate() {
            let r = radius(index.leaves().level(leaf), &domain);
            let n = in_targets.len();
            let (potential, gradient) = batched[offset..offset + 4 * n].split_at(n);
            offset += 4 * n;
            let x: Vec<[f64; 3]> = in_targets.iter().map(|&j| targets[j]).collect();
            let mut phi = vec![0.0; n];
            let mut grad = vec![[0.0; 3]; n];
            direct_sum(&sources, &charges, &x, &mut phi, Some(&mut grad));
            for j in 0..n {
                sums[0] += (potential[j] / r - phi[j]).powi(2);
                sums[1] += phi[j].powi(2);
                for k in 0..3 {
                    sums[2] += (gradient[3 * j + k] / (r * r) - grad[j][k]).powi(2);
                    sums[3] += grad[j][k].powi(2);
                }
            }
        }
        assert_eq!(offset, batched.len());
        let mut total = [0.0f64; 4];
        comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
        let (potential_error, gradient_error) =
            ((total[0] / total[1]).sqrt(), (total[2] / total[3]).sqrt());
        eprintln!(
            "rank {}: P2P kernel {choice}: batched = per-pair = 2, 4, 8 threads bit for bit \
             ({} values); W, X, V pairs {lists:?}; p = {p}: relative L2 error vs direct sum \
             {potential_error:.3e} (potential), {gradient_error:.3e} (gradient)",
            comm.rank(),
            batched.len()
        );
        assert!(potential_error < 1e-3, "potential: {potential_error:e}");
        assert!(gradient_error < 1e-2, "gradient: {gradient_error:e}");
    }
}

/// `n` points uniform in the unit cube [0, 1)³.
fn unit_cube_points(rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| core::array::from_fn(|_| rng.range(0.0, 1.0)))
        .collect()
}

/// `n` charges uniform in [−1, 1).
fn random_charges(rng: &mut SplitMix64, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.range(-1.0, 1.0)).collect()
}

/// This rank's share of `items`: every `size`-th item from `rank`.
fn share<V: Copy>(items: &[V], comm: &SimpleCommunicator) -> Vec<V> {
    let (rank, size) = (comm.rank() as usize, comm.size() as usize);
    items.iter().skip(rank).step_by(size).copied().collect()
}

/// The FMM, or `None` on several ranks if the points are not owned (module
/// documentation). Panics on any other error.
fn built<'o, T: Stored + SimdScalar + Equivalence + Default>(
    result: Result<Fmm<'o, T>, FmmError>,
    comm: &SimpleCommunicator,
) -> Option<Fmm<'o, T>> {
    match result {
        Ok(fmm) => Some(fmm),
        Err(FmmError::PointsNotOwned { count }) if comm.size() > 1 => {
            eprintln!(
                "rank {}: {count} points not owned; not redistributed until C5.1, scenario \
                 stops",
                comm.rank()
            );
            None
        }
        Err(error) => panic!("rank {}: the FMM does not build: {error}", comm.rank()),
    }
}

/// Builds the FMM of `builder` at one thread and evaluates `charges`, then builds it
/// again at each count of `threads` and checks the threaded evaluation
/// (`check_threaded`) against the one-thread output (C3.5). Returns the one-thread FMM
/// and its output, or `None` if the points are not owned (`built`).
///
/// Every build loads its tables from [`TABLE_CACHE`], building and storing them once.
fn evaluate_threaded<'o, T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    (sources, targets): (&[[f64; 3]], &[[f64; 3]]),
    charges: &[T],
    threads: &[usize],
    comm: &'o SimpleCommunicator,
) -> Option<(Fmm<'o, T>, Output<T>)> {
    let builder = builder.clone().table_cache(TABLE_CACHE);
    let mut fmm = built(
        builder.clone().threads(1).build(sources, targets, comm),
        comm,
    )?;
    assert_eq!(fmm.threading().threads, 1);
    assert_eq!(fmm.operator().threads(), 1);
    let output = fmm.evaluate(charges).expect("the FMM evaluates");
    let reference = output_bits(&output);
    #[cfg(feature = "gpu")]
    for backend in device_backends() {
        let (outcome, difference) = device_common::check_backend(
            &builder,
            (sources, targets),
            charges,
            &mut fmm,
            &output,
            backend,
            comm,
        );
        let mut runs = DEVICE_RUNS.lock().unwrap();
        let entry = match runs.iter().position(|r| r.backend == backend) {
            Some(i) => &mut runs[i],
            None => {
                runs.push(DeviceRuns {
                    backend,
                    f32_runs: 0,
                    f64_runs: 0,
                    refused: 0,
                    worst: [[0.0; 5]; 2],
                });
                runs.last_mut().unwrap()
            }
        };
        let precision = usize::from(size_of::<T>() == 8);
        let worst = &mut entry.worst[precision];
        for (w, d) in worst.iter_mut().zip([
            difference.potential,
            difference.gradient,
            difference.multipoles,
            difference.locals,
            difference.root,
        ]) {
            *w = w.max(d);
        }
        match outcome {
            device_common::Outcome::Ran if precision == 0 => entry.f32_runs += 1,
            device_common::Outcome::Ran => entry.f64_runs += 1,
            device_common::Outcome::OneRankOnly => entry.refused += 1,
        }
    }
    for &n in threads {
        let builder = builder.clone().threads(n);
        let mut threaded = built(builder.build(sources, targets, comm), comm)?;
        check_threaded(&mut threaded, charges, &reference, n);
    }
    Some((fmm, output))
}

/// The P2P kernels besides `Auto` that [`evaluate_every_kernel`] runs: `Reference` and
/// every ISA this machine offers.
fn p2p_choices() -> Vec<P2pChoice> {
    std::iter::once(P2pChoice::Reference)
        .chain(Isa::available().map(P2pChoice::Isa))
        .collect()
}

/// The largest relative L2 difference of an ISA's output from the `Reference` output of
/// the same settings (C3S.5): 1e-13 in f64, 1e-6 in f32.
fn kernel_tolerance<T: RealScalar>() -> f64 {
    if size_of::<T>() == 4 { 1e-6 } else { 1e-13 }
}

/// [`evaluate_threaded`] with the builder's P2P kernel (`Auto`, unless it sets one), and
/// again with `Reference` and with every ISA this machine offers (C3S.5). Each kernel's
/// output is bit-identical at one thread and at every count of `threads`; `Auto` runs
/// the ISA of `Isa::detect` and gives its output bit for bit, so that ISA repeats only
/// the one-thread run; and every ISA's potential and gradient lie within
/// [`kernel_tolerance`] of the `Reference` output (relative L2 over all ranks). Prints
/// the kernels it ran. Returns the `Auto` run, or `None` if the points are not owned.
fn evaluate_every_kernel<'o, T: Stored + SimdScalar + Equivalence + Default>(
    builder: &FmmBuilder<T>,
    sets: (&[[f64; 3]], &[[f64; 3]]),
    charges: &[T],
    threads: &[usize],
    comm: &'o SimpleCommunicator,
) -> Option<(Fmm<'o, T>, Output<T>)> {
    let (fmm, output) = evaluate_threaded(builder, sets, charges, threads, comm)?;
    let detected = P2pChoice::Isa(Isa::detect());
    assert_eq!(fmm.p2p_kernel(), detected, "Auto runs the detected ISA");
    let mut reference = None;
    let mut isas = Vec::new();
    for choice in p2p_choices() {
        let builder = builder.clone().p2p_kernel(choice);
        let repeats = if choice == detected { &[][..] } else { threads };
        let (kernel, kernel_output) = evaluate_threaded(&builder, sets, charges, repeats, comm)?;
        assert_eq!(kernel.p2p_kernel(), choice);
        if choice == detected {
            assert_eq!(
                output_bits(&kernel_output),
                output_bits(&output),
                "Auto and {choice} differ"
            );
        }
        match choice {
            P2pChoice::Reference => reference = Some(kernel_output),
            _ => isas.push((choice, kernel_output)),
        }
    }
    let reference = reference.expect("`p2p_choices` holds Reference");
    let tolerance = kernel_tolerance::<T>();
    let mut differences = Vec::new();
    for (choice, kernel_output) in &isas {
        let potential = difference(&kernel_output.potential, &reference.potential, comm);
        let gradient = match (&kernel_output.gradient, &reference.gradient) {
            (Some(g), Some(r)) => difference(g.as_flattened(), r.as_flattened(), comm),
            _ => 0.0,
        };
        assert!(
            potential < tolerance && gradient < tolerance,
            "{choice}: relative L2 difference from Reference {potential:e} (φ), {gradient:e} \
             (∇φ), tolerance {tolerance:e}"
        );
        differences.push(match kernel_output.gradient {
            Some(_) => format!("{choice} {potential:.1e} (φ), {gradient:.1e} (∇φ)"),
            None => format!("{choice} {potential:.1e} (φ)"),
        });
    }
    let names: Vec<String> = isas.iter().map(|(c, _)| c.to_string()).collect();
    eprintln!(
        "rank {}: P2P kernels auto ({detected}), reference, {}, each bit for bit at 1 and \
         {threads:?} threads; relative L2 difference from reference: {}",
        comm.rank(),
        names.join(", "),
        differences.join("; ")
    );
    Some((fmm, output))
}

/// The relative L2 difference ‖a − b‖₂ / ‖b‖₂ over the values of all ranks, or ‖a‖₂
/// if b is zero on every rank (a scenario without targets).
fn difference<T: RealScalar>(a: &[T], b: &[T], comm: &SimpleCommunicator) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sums = [0.0f64; 2];
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (RealScalar::to_f64(x), RealScalar::to_f64(y));
        sums[0] += (x - y).powi(2);
        sums[1] += y * y;
    }
    let mut total = [0.0f64; 2];
    comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
    if total[1] > 0.0 {
        (total[0] / total[1]).sqrt()
    } else {
        total[0].sqrt()
    }
}

/// Evaluates `charges` with `fmm`, built with `n` threads, and checks its threading
/// report, the capacity of every per-thread scratch set before and after, and the output
/// bit for bit against `reference`.
fn check_threaded<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &mut Fmm<'_, T>,
    charges: &[T],
    reference: &[u64],
    n: usize,
) {
    let report = fmm.threading();
    assert_eq!(report.threads, n, "{report}");
    assert!(report.mpi >= Threading::Funneled, "{report}");
    assert_eq!(fmm.operator().threads(), n);
    let capacities = fmm.operator().scratch_capacities();
    assert_eq!(
        capacities,
        vec![fmm.max_leaf_points(); n],
        "one scratch set per thread"
    );
    let output = fmm.evaluate(charges).expect("the threaded FMM evaluates");
    assert_eq!(
        fmm.operator().scratch_capacities(),
        capacities,
        "{n} threads: a scratch set reallocated"
    );
    let bits = output_bits(&output);
    assert_eq!(bits.len(), reference.len());
    let differing = bits.iter().zip(reference).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "{n} threads: {differing} of {} values differ from one thread",
        bits.len()
    );
}

/// The exact potential (and gradient) of `charges` at `sources` at every target:
/// `direct_sum` divided by 4π.
fn exact(sources: &[[f64; 3]], charges: &[f64], targets: &[[f64; 3]]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let mut phi = vec![0.0; targets.len()];
    let mut grad = vec![[0.0; 3]; targets.len()];
    direct_sum(sources, charges, targets, &mut phi, Some(&mut grad));
    let scale = 4.0 * std::f64::consts::PI;
    (
        phi.iter().map(|v| v / scale).collect(),
        grad.iter().map(|g| g.map(|gk| gk / scale)).collect(),
    )
}

/// The relative L2 error ‖a − e‖₂ / ‖e‖₂ over the values of all ranks.
fn relative_l2<T: RealScalar>(approx: &[T], exact: &[f64], comm: &SimpleCommunicator) -> f64 {
    assert_eq!(approx.len(), exact.len());
    let mut sums = [0.0f64; 2];
    for (&a, &e) in approx.iter().zip(exact) {
        sums[0] += (RealScalar::to_f64(a) - e).powi(2);
        sums[1] += e * e;
    }
    let mut total = [0.0f64; 2];
    comm.all_reduce_into(&sums[..], &mut total[..], SystemOperation::sum());
    (total[0] / total[1]).sqrt()
}

/// The relative L2 error of 3-vectors over all ranks, with the Euclidean norm per
/// vector.
fn relative_l2_vectors<T: RealScalar>(
    approx: &[[T; 3]],
    exact: &[[f64; 3]],
    comm: &SimpleCommunicator,
) -> f64 {
    let approx: Vec<T> = approx.iter().flatten().copied().collect();
    let exact: Vec<f64> = exact.iter().flatten().copied().collect();
    relative_l2(&approx, &exact, comm)
}

/// The values as f64 bit patterns (the conversion to f64 is exact).
fn bits<T: RealScalar>(values: &[T]) -> Vec<u64> {
    values
        .iter()
        .map(|&v| RealScalar::to_f64(v).to_bits())
        .collect()
}

/// The output's values as bit patterns: potentials, then gradients.
fn output_bits<T: RealScalar>(output: &Output<T>) -> Vec<u64> {
    let mut values = bits(&output.potential);
    if let Some(gradient) = &output.gradient {
        values.extend(bits(gradient.as_flattened()));
    }
    values
}

/// The sum of `local` over all ranks.
fn global_sum(local: usize, comm: &SimpleCommunicator) -> usize {
    let mut total = 0usize;
    comm.all_reduce_into(&local, &mut total, SystemOperation::sum());
    total
}

/// Uniform cube, sources equal to targets, p ∈ {2, 4, 6, 8}, every strategy (module
/// documentation).
fn uniform_cube_strategies(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7893);
    let points = unit_cube_points(&mut rng, 2000);
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let (phi, _) = exact(&points, &charges, &sources);
    let strategies = [
        M2lStrategy::Dense,
        M2lStrategy::Classes,
        M2lStrategy::Rotation,
    ];
    let ps = [2, 4, 6, 8];
    let mut errors = [[0.0f64; 4]; 3];
    for (k, &p) in ps.iter().enumerate() {
        let mut outputs: Vec<Vec<f64>> = Vec::new();
        for (s, &strategy) in strategies.iter().enumerate() {
            let builder = FmmBuilder::<f64>::new(p).strategy(strategy);
            let sets = (&sources[..], &sources[..]);
            let Some((fmm, output)) =
                evaluate_threaded(&builder, sets, &local_charges, &THREADS, comm)
            else {
                return;
            };
            assert_eq!(fmm.strategy(), strategy);
            errors[s][k] = relative_l2(&output.potential, &phi, comm);
            outputs.push(output.potential);
        }
        for s in 1..strategies.len() {
            let difference = relative_l2(&outputs[s], &outputs[0], comm);
            assert!(
                difference < 1e-12,
                "p = {p}: {:?} differs from Dense by {difference:e} (relative L2)",
                strategies[s]
            );
        }
    }
    for (s, strategy) in strategies.iter().enumerate() {
        eprintln!(
            "rank {}: uniform cube, N = {}, {strategy:?}: relative L2 error of φ at p = \
             {ps:?}: {}",
            comm.rank(),
            points.len(),
            errors[s].map(|e| format!("{e:.3e}")).join(", ")
        );
        assert!(
            errors[s].windows(2).all(|e| e[1] < e[0]),
            "{strategy:?}: the error does not decrease with p: {:?}",
            errors[s]
        );
        assert!(
            errors[s][3] < 1e-4,
            "{strategy:?}: p = 8: {:e}",
            errors[s][3]
        );
    }
}

/// Sources and targets disjoint, and sources a subset of the targets (module
/// documentation). Tolerance 1e-3, ten times the p = 6 error of the strategy scenario:
/// a smoke check of the permutations, which a wrong order fails by O(1).
fn distinct_and_nested(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7894);
    let sources = unit_cube_points(&mut rng, 1500);
    let charges = random_charges(&mut rng, sources.len());
    let others = unit_cube_points(&mut rng, 1000);
    // The sources in the middle of the targets, so their positions shift.
    let nested: Vec<[f64; 3]> = others[..500]
        .iter()
        .chain(&sources)
        .chain(&others[500..])
        .copied()
        .collect();
    let (local_sources, local_charges) = (share(&sources, comm), share(&charges, comm));
    for (name, targets) in [("disjoint", &others), ("nested", &nested)] {
        let local_targets = share(targets, comm);
        let builder = FmmBuilder::<f64>::new(6);
        let sets = (&local_sources[..], &local_targets[..]);
        let Some((fmm, output)) = evaluate_threaded(&builder, sets, &local_charges, &THREADS, comm)
        else {
            return;
        };
        assert_eq!(fmm.strategy(), M2lStrategy::Dense, "Auto at p = 6");
        assert_eq!(
            (fmm.nsources(), fmm.ntargets()),
            (local_sources.len(), local_targets.len())
        );
        assert_eq!(output.potential.len(), local_targets.len());
        assert!(output.gradient.is_none());
        let (phi, _) = exact(&sources, &charges, &local_targets);
        let error = relative_l2(&output.potential, &phi, comm);
        eprintln!(
            "rank {}: {name}: {} sources, {} targets, p = 6: relative L2 error of φ {error:.3e}",
            comm.rank(),
            sources.len(),
            targets.len()
        );
        assert!(error < 1e-3, "{name}: {error:e}");
    }
}

/// Gradients against the `direct_sum` gradient (module documentation). Tolerance 1e-2,
/// the smoke tolerance of the batched-against-per-pair scenario.
fn gradients(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7895);
    let points = unit_cube_points(&mut rng, 2000);
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(6).gradients(true);
    let sets = (&sources[..], &sources[..]);
    let Some((fmm, output)) = evaluate_every_kernel(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };
    assert!(fmm.gradients());
    let gradient = output.gradient.as_ref().expect("gradients requested");
    assert_eq!(gradient.len(), sources.len());
    let (phi, grad) = exact(&points, &charges, &sources);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(gradient, &grad, comm);
    eprintln!(
        "rank {}: gradients, N = {}, p = 6: relative L2 error {potential_error:.3e} (φ), \
         {gradient_error:.3e} (∇φ)",
        comm.rank(),
        points.len()
    );
    assert!(potential_error < 1e-3, "φ: {potential_error:e}");
    assert!(gradient_error < 1e-2, "∇φ: {gradient_error:e}");

    // The potentials do not depend on whether gradients are computed: bit for bit with
    // the reference P2P, which computes each term alike either way. The SIMD kernels
    // compute φ += q ρ with one fma without gradients, and t = q ρ, φ + t with them
    // (`nd_fmm_simd`, "The kernel"), so there the potentials agree to rounding, within
    // the tolerance of an ISA against the reference.
    let builder = FmmBuilder::<f64>::new(6);
    let Some((_, without)) = evaluate_threaded(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };
    let kernel_difference = difference(&without.potential, &output.potential, comm);
    assert!(
        kernel_difference < kernel_tolerance::<f64>(),
        "{kernel_difference:e}"
    );
    let mut reference = Vec::new();
    for gradients in [false, true] {
        let builder = builder
            .clone()
            .gradients(gradients)
            .p2p_kernel(P2pChoice::Reference);
        let Some((_, output)) = evaluate_threaded(&builder, sets, &local_charges, &[], comm) else {
            return;
        };
        reference.push(output.potential);
    }
    assert_eq!(bits(&reference[0]), bits(&reference[1]), "Reference");
    eprintln!(
        "rank {}: gradients: φ with and without gradients bit for bit with the reference \
         P2P, relative L2 difference {kernel_difference:.1e} with {}",
        comm.rank(),
        P2pChoice::Auto.resolve().unwrap()
    );
}

/// A uniform level-3 tree: no W or X list (module documentation).
fn uniform_tree(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7896);
    // Four points in every level-3 box of the unit cube, away from its faces.
    let points: Vec<[f64; 3]> = (0..512)
        .flat_map(|b: usize| {
            let index = [b >> 6, (b >> 3) & 7, b & 7];
            (0..4)
                .map(|_| core::array::from_fn(|k| (index[k] as f64 + rng.range(0.05, 0.95)) / 8.0))
                .collect::<Vec<[f64; 3]>>()
        })
        .collect();
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(4)
        .max_level(3)
        .max_points_per_leaf(1)
        .domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
    let sets = (&sources[..], &sources[..]);
    let Some((fmm, output)) = evaluate_threaded(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };
    assert_eq!(fmm.nlevels(), 4);
    let leaves = fmm.plan().index().leaves();
    assert!(
        (0..fmm.nleaves()).all(|j| leaves.level(j) == 3),
        "every leaf lies on level 3"
    );
    assert_eq!(global_sum(fmm.nleaves(), comm), 512);
    let sizes = fmm.list_sizes();
    let (w, x) = (global_sum(sizes.w, comm), global_sum(sizes.x, comm));
    assert_eq!((w, x), (0, 0), "W and X lists of a uniform tree");
    assert!(global_sum(sizes.v, comm) > 0 && global_sum(sizes.u, comm) > 0);
    let (phi, _) = exact(&points, &charges, &sources);
    let error = relative_l2(&output.potential, &phi, comm);
    eprintln!(
        "rank {}: uniform level-3 tree, N = {}, p = 4: {} V and {} U pairs; relative L2 \
         error of φ {error:.3e}",
        comm.rank(),
        points.len(),
        global_sum(sizes.v, comm),
        global_sum(sizes.u, comm)
    );
    assert!(error < 1e-2, "{error:e}");
}

/// The root as the only leaf: P2P alone (module documentation). Error measure: relative
/// L2 against the direct sum, 1e-14.
fn single_leaf(comm: &SimpleCommunicator) {
    if comm.size() > 1 {
        eprintln!("rank {}: single leaf: one rank only", comm.rank());
        return;
    }
    let mut rng = SplitMix64(0x7897);
    let points = unit_cube_points(&mut rng, 200);
    let charges = random_charges(&mut rng, points.len());
    let builder = FmmBuilder::<f64>::new(3).max_level(0).gradients(true);
    let (fmm, output) =
        evaluate_every_kernel(&builder, (&points, &points), &charges, &THREADS, comm)
            .expect("the FMM builds");
    assert_eq!((fmm.nlevels(), fmm.nleaves()), (1, 1));
    let (phi, grad) = exact(&points, &charges, &points);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(output.gradient.as_ref().unwrap(), &grad, comm);
    eprintln!(
        "rank {}: single leaf, N = {}: relative L2 error {potential_error:.3e} (φ), \
         {gradient_error:.3e} (∇φ)",
        comm.rank(),
        points.len()
    );
    assert!(potential_error < 1e-14, "φ: {potential_error:e}");
    assert!(gradient_error < 1e-14, "∇φ: {gradient_error:e}");
}

/// Rank 0 has no targets (module documentation).
fn no_targets_on_this_rank(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x7898);
    let points = unit_cube_points(&mut rng, 500);
    let charges = random_charges(&mut rng, points.len());
    let targets = unit_cube_points(&mut rng, 300);
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let local_targets = if comm.rank() == 0 {
        Vec::new()
    } else {
        share(&targets, comm)
    };
    let builder = FmmBuilder::<f64>::new(4).gradients(true);
    let sets = (&sources[..], &local_targets[..]);
    let Some((fmm, output)) = evaluate_every_kernel(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };
    assert_eq!(output.potential.len(), local_targets.len());
    assert_eq!(output.gradient.as_ref().unwrap().len(), local_targets.len());
    if comm.rank() == 0 {
        assert_eq!(fmm.ntargets(), 0);
        assert!(fmm.target_counts().iter().all(|&n| n == 0));
    }
    eprintln!(
        "rank {}: no targets on rank 0: {} targets here",
        comm.rank(),
        output.potential.len()
    );
}

/// Every input error gives its `FmmError`, agreed on every rank (module documentation).
fn input_errors(comm: &SimpleCommunicator) {
    let (rank, last) = (comm.rank(), comm.size() - 1);
    let mut rng = SplitMix64(0x7899);
    let points = share(&unit_cube_points(&mut rng, 400), comm);
    let unit = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

    // A supplied domain that is not a cube: the same on every rank.
    let builder =
        FmmBuilder::<f64>::new(4).domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 2.0]));
    let error = builder.build(&points, &points, comm).err();
    assert!(
        matches!(
            error,
            Some(FmmError::InvalidDomain(GeometryError::NotCubic { .. }))
        ),
        "{error:?}"
    );

    // A source of rank 0 outside the supplied domain.
    let mut outside = points.clone();
    if rank == 0 {
        outside[3] = [1.5, 0.5, 0.5];
    }
    let builder = FmmBuilder::<f64>::new(4).domain(unit);
    let error = builder.build(&outside, &points, comm).err();
    match (rank, &error) {
        (
            0,
            Some(FmmError::PointOutsideDomain {
                set, index, point, ..
            }),
        ) => {
            assert_eq!(
                (*set, *index, *point),
                (PointSet::Sources, 3, [1.5, 0.5, 0.5])
            );
        }
        (r, Some(FmmError::OtherRank)) if r != 0 => {}
        _ => panic!("rank {rank}: point outside the domain: {error:?}"),
    }

    // A target of the last rank that is not finite.
    let mut non_finite = points.clone();
    if rank == last {
        non_finite[0] = [0.5, f64::NAN, 0.5];
    }
    let error = FmmBuilder::<f64>::new(4)
        .build(&points, &non_finite, comm)
        .err();
    match (rank, &error) {
        (r, Some(FmmError::NonFinitePoint { set, index, .. })) if r == last => {
            assert_eq!((*set, *index), (PointSet::Targets, 0));
        }
        (r, Some(FmmError::OtherRank)) if r != last => {}
        _ => panic!("rank {rank}: non-finite point: {error:?}"),
    }

    // Settings out of range.
    for (builder, expected) in [
        (
            FmmBuilder::<f64>::new(21),
            SettingsError::DegreeTooLarge { p: 21 },
        ),
        (
            FmmBuilder::<f64>::new(4).max_level(17),
            SettingsError::MaxLevelTooDeep { max_level: 17 },
        ),
        (
            FmmBuilder::<f64>::new(4).max_points_per_leaf(0),
            SettingsError::ZeroPointsPerLeaf,
        ),
        (
            FmmBuilder::<f64>::new(4).threads(0),
            SettingsError::ZeroThreads,
        ),
    ] {
        let error = builder.build(&points, &points, comm).err();
        assert_eq!(error, Some(FmmError::InvalidSettings(expected)));
    }

    // A P2P kernel on an ISA this machine cannot run (NEON on x86_64, AVX2 on aarch64
    // and on x86_64 CPUs without it). The check is local; `build` agrees it with every
    // other input error of step 1, so on mixed CPUs the ranks that can run the ISA
    // return `OtherRank`.
    let unavailable: Vec<Isa> = Isa::all().filter(|isa| !isa.is_available()).collect();
    assert!(!unavailable.is_empty(), "NEON and AVX2 exclude each other");
    for &isa in &unavailable {
        let builder = FmmBuilder::<f64>::new(4).p2p_kernel(P2pChoice::Isa(isa));
        let error = builder.build(&points, &points, comm).err();
        let expected = SettingsError::P2pIsaUnavailable { isa };
        assert_eq!(error, Some(FmmError::InvalidSettings(expected)), "{isa}");
    }

    // No points on any rank, and points that span no volume.
    let error = FmmBuilder::<f64>::new(4).build(&[], &[], comm).err();
    assert_eq!(error, Some(FmmError::NoPoints));
    let error = FmmBuilder::<f64>::new(4)
        .build(&[[0.5; 3]], &[[0.5; 3]], comm)
        .err();
    assert_eq!(error, Some(FmmError::DegenerateExtent { extent: 0.0 }));

    // A charge vector of the wrong length on rank 0.
    let Some(mut fmm) = built(
        FmmBuilder::<f64>::new(2).build(&points, &points, comm),
        comm,
    ) else {
        return;
    };
    let charges = vec![1.0; points.len() - usize::from(rank == 0)];
    let error = fmm.evaluate(&charges).err();
    match (rank, &error) {
        (0, Some(FmmError::ChargesLength { expected, actual })) => {
            assert_eq!((*expected, *actual), (points.len(), points.len() - 1));
        }
        (r, Some(FmmError::OtherRank)) if r != 0 => {}
        _ => panic!("rank {rank}: wrong charge length: {error:?}"),
    }
    eprintln!(
        "rank {rank}: input errors as expected, the P2P kernel on {} among them",
        unavailable
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// f32 at p = 6 against f64 (module documentation). The f64 run and the oracle use the
/// f32 charges, so only the arithmetic differs. Tolerances: the f32 output within 1e-5
/// of the f64 output (relative L2, about 100 ε₃₂), and its error against the direct sum
/// within 1e-5 of the f64 error.
fn f32_against_f64(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789a);
    let points = unit_cube_points(&mut rng, 2000);
    let charges32: Vec<f32> = random_charges(&mut rng, points.len())
        .iter()
        .map(|&q| q as f32)
        .collect();
    let charges: Vec<f64> = charges32.iter().map(|&q| f64::from(q)).collect();
    let sources = share(&points, comm);
    let (local32, local64) = (share(&charges32, comm), share(&charges, comm));
    let sets = (&sources[..], &sources[..]);
    let builder = FmmBuilder::<f32>::new(6).gradients(true);
    let Some((_, output32)) = evaluate_every_kernel(&builder, sets, &local32, &THREADS, comm)
    else {
        return;
    };
    let builder = FmmBuilder::<f64>::new(6).gradients(true);
    let Some((_, output64)) = evaluate_every_kernel(&builder, sets, &local64, &THREADS, comm)
    else {
        return;
    };
    let (phi, _) = exact(&points, &charges, &sources);
    let difference = relative_l2(&output32.potential, &output64.potential, comm);
    let gradient_difference = relative_l2_vectors(
        output32.gradient.as_ref().unwrap(),
        output64.gradient.as_ref().unwrap(),
        comm,
    );
    let (error32, error64) = (
        relative_l2(&output32.potential, &phi, comm),
        relative_l2(&output64.potential, &phi, comm),
    );
    eprintln!(
        "rank {}: f32 against f64, N = {}, p = 6: relative L2 difference {difference:.3e} (φ), \
         {gradient_difference:.3e} (∇φ); error of φ {error32:.3e} (f32), {error64:.3e} (f64)",
        comm.rank(),
        points.len()
    );
    assert!(difference < 1e-5, "φ: {difference:e}");
    assert!(gradient_difference < 1e-5, "∇φ: {gradient_difference:e}");
    assert!(
        (error32 - error64).abs() < 1e-5,
        "{error32:e} against {error64:e}"
    );
}

/// Two evaluations agree bit for bit, and a second charge vector gives the output of a
/// fresh build (module documentation).
fn repeatability(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789b);
    let points = unit_cube_points(&mut rng, 1000);
    let first = random_charges(&mut rng, points.len());
    let second = random_charges(&mut rng, points.len());
    let sources = share(&points, comm);
    let (first, second) = (share(&first, comm), share(&second, comm));
    let builder = FmmBuilder::<f64>::new(4).gradients(true);
    let Some(mut fmm) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    let a = fmm.evaluate(&first).unwrap();
    let b = fmm.evaluate(&first).unwrap();
    assert_eq!(output_bits(&a), output_bits(&b), "two evaluations differ");
    let again = fmm.evaluate(&second).unwrap();
    let Some(mut fresh) = built(builder.build(&sources, &sources, comm), comm) else {
        return;
    };
    let reference = fresh.evaluate(&second).unwrap();
    assert_eq!(
        output_bits(&again),
        output_bits(&reference),
        "a second charge vector differs from a fresh build"
    );
    assert_ne!(output_bits(&a), output_bits(&again));

    // Threaded (C3.5): two evaluations, a serial one on the same `Fmm` in between and a
    // threaded one after it, and the second charge vector, each bit for bit.
    let Some(mut threaded) = built(
        builder.clone().threads(4).build(&sources, &sources, comm),
        comm,
    ) else {
        return;
    };
    for (name, serial) in [
        ("first threaded", false),
        ("second threaded", false),
        ("serial", true),
        ("threaded after serial", false),
    ] {
        threaded.set_serial(serial);
        assert_eq!(threaded.operator().threads(), if serial { 1 } else { 4 });
        let output = threaded.evaluate(&first).unwrap();
        assert_eq!(output_bits(&output), output_bits(&a), "{name} evaluation");
    }
    let output = threaded.evaluate(&second).unwrap();
    assert_eq!(
        output_bits(&output),
        output_bits(&again),
        "second charge vector"
    );
    eprintln!(
        "rank {}: repeatability: {} values bit for bit, serial and at 4 threads",
        comm.rank(),
        output_bits(&a).len()
    );
}

/// Every rank passes the complete point set: the FMM runs on one rank, and on several
/// every rank returns `PointsNotOwned` with the same count (module documentation).
fn ownership(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789c);
    let points = unit_cube_points(&mut rng, 600);
    let charges = random_charges(&mut rng, points.len());
    let builder = FmmBuilder::<f64>::new(4);
    if comm.size() == 1 {
        let (_, output) = evaluate_threaded(&builder, (&points, &points), &charges, &THREADS, comm)
            .expect("on one rank every point is owned");
        let (phi, _) = exact(&points, &charges, &points);
        let error = relative_l2(&output.potential, &phi, comm);
        eprintln!("rank 0: ownership: one rank, relative L2 error of φ {error:.3e} (p = 4)");
        assert!(error < 1e-2, "{error:e}");
        return;
    }
    let count = match builder.build(&points, &points, comm) {
        Err(FmmError::PointsNotOwned { count }) => count,
        Err(error) => panic!("rank {}: {error}", comm.rank()),
        Ok(_) => panic!(
            "rank {}: every rank holds every point, yet the build succeeds",
            comm.rank()
        ),
    };
    let (mut lowest, mut highest) = (0u64, 0u64);
    comm.all_reduce_into(&count, &mut lowest, SystemOperation::min());
    comm.all_reduce_into(&count, &mut highest, SystemOperation::max());
    assert_eq!(lowest, highest, "the ranks disagree on the count");
    assert!(count > 0);
    // A device backend: PointsNotOwned wins over DeviceNeedsOneRank (device-path.md §4.4).
    #[cfg(feature = "gpu")]
    for backend in device_backends() {
        let error = builder
            .clone()
            .backend(backend)
            .build(&points, &points, comm)
            .err();
        assert_eq!(
            error,
            Some(FmmError::PointsNotOwned { count }),
            "rank {}: {backend}",
            comm.rank()
        );
    }
    eprintln!(
        "rank {}: ownership: PointsNotOwned with {count} points, on every rank",
        comm.rank()
    );
}

/// An adaptive tree with W and X lists, in f32 and f64, every strategy, gradients off
/// and on, at 1, 2, 4 and 8 threads, bit for bit (module documentation).
fn threads_every_strategy(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789d);
    // A cloud over the unit cube and a dense blob near one corner: a graded tree.
    let mut points = unit_cube_points(&mut rng, 200);
    points.extend((0..200).map(|_| -> [f64; 3] { core::array::from_fn(|_| rng.range(0.02, 0.1)) }));
    let charges = random_charges(&mut rng, points.len());
    let (sources, charges64) = (share(&points, comm), share(&charges, comm));
    let charges32: Vec<f32> = charges64.iter().map(|&q| q as f32).collect();
    let configurations = threads_every_strategy_in::<f64>(&sources, &charges64, comm)
        + threads_every_strategy_in::<f32>(&sources, &charges32, comm);
    eprintln!(
        "rank {}: threads: {configurations} configurations (f32 and f64, Dense, Classes, \
         Rotation, gradients off and on) bit for bit at 1, 2, 4 and 8 threads",
        comm.rank()
    );
}

/// The configurations of `threads_every_strategy` in precision `T`; returns how many
/// ran.
fn threads_every_strategy_in<T: Stored + SimdScalar + Equivalence + Default>(
    sources: &[[f64; 3]],
    charges: &[T],
    comm: &SimpleCommunicator,
) -> usize {
    let mut configurations = 0;
    for strategy in [
        M2lStrategy::Dense,
        M2lStrategy::Classes,
        M2lStrategy::Rotation,
    ] {
        for gradients in [false, true] {
            let builder = FmmBuilder::<T>::new(3)
                .strategy(strategy)
                .gradients(gradients)
                .max_points_per_leaf(8);
            // Every P2P kernel with Dense; the strategies do not touch P2P.
            let evaluate = if strategy == M2lStrategy::Dense {
                evaluate_every_kernel
            } else {
                evaluate_threaded
            };
            let Some((fmm, output)) =
                evaluate(&builder, (sources, sources), charges, &THREADS, comm)
            else {
                return configurations;
            };
            let sizes = fmm.list_sizes();
            let (w, x) = (global_sum(sizes.w, comm), global_sum(sizes.x, comm));
            assert!(w > 0 && x > 0, "W and X lists: {w}, {x}");
            assert_eq!(output.gradient.is_some(), gradients);
            configurations += 1;
        }
    }
    configurations
}

/// `n` points uniform on the sphere of radius `radius` about `centre`.
fn sphere_points(rng: &mut SplitMix64, n: usize, centre: [f64; 3], radius: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let v = rng.direction();
            core::array::from_fn(|k| centre[k] + radius * v[k])
        })
        .collect()
}

/// `n` points of a Plummer sphere of scale `a` about `centre`, truncated at 10 a, by
/// inverse transform of the cumulative mass: the rule of
/// `nd_fmm_validate::points::plummer`, which this crate cannot depend on.
fn plummer_points(rng: &mut SplitMix64, n: usize, centre: [f64; 3], a: f64) -> Vec<[f64; 3]> {
    let mass = 1000.0 / 101.0f64.powf(1.5);
    (0..n)
        .map(|_| {
            let u = mass * rng.range(0.0, 1.0);
            let r = if u > 0.0 {
                a / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
            } else {
                0.0
            };
            let v = rng.direction();
            core::array::from_fn(|k| centre[k] + r * v[k])
        })
        .collect()
}

/// `n` points spread evenly over Gaussian clusters of width `width` at `centres`,
/// cluster after cluster, truncated at 4 σ: the rule of
/// `nd_fmm_validate::points::gaussian_clusters`.
fn cluster_points(
    rng: &mut SplitMix64,
    n: usize,
    centres: &[[f64; 3]],
    width: f64,
) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let c = centres[i * centres.len() / n];
            let z = loop {
                let z: [f64; 3] = core::array::from_fn(|_| rng.normal());
                if z[0] * z[0] + z[1] * z[1] + z[2] * z[2] <= 16.0 {
                    break z;
                }
            };
            core::array::from_fn(|k| c[k] + width * z[k])
        })
        .collect()
}

/// The number of leaves on each level 0–16 and the U, V, W and X list sizes of `fmm`,
/// over all ranks.
fn tree_summary<T: Stored + SimdScalar + Equivalence + Default>(
    fmm: &Fmm<'_, T>,
    comm: &SimpleCommunicator,
) -> ([usize; 17], [usize; 4]) {
    let leaves = fmm.plan().index().leaves();
    let mut local = [0usize; 17];
    for j in 0..fmm.nleaves() {
        local[leaves.level(j)] += 1;
    }
    let mut histogram = [0usize; 17];
    comm.all_reduce_into(&local[..], &mut histogram[..], SystemOperation::sum());
    let sizes = fmm.list_sizes();
    let local = [sizes.u, sizes.v, sizes.w, sizes.x];
    let mut lists = [0usize; 4];
    comm.all_reduce_into(&local[..], &mut lists[..], SystemOperation::sum());
    (histogram, lists)
}

/// The levels that hold leaves, from the histogram of `tree_summary`.
fn leaf_levels(histogram: &[usize; 17]) -> Vec<usize> {
    (0..17).filter(|&l| histogram[l] > 0).collect()
}

/// The number of local leaves per level, as "level: count" pairs.
fn levels_text(histogram: &[usize; 17]) -> String {
    leaf_levels(histogram)
        .iter()
        .map(|&l| format!("{l}: {}", histogram[l]))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The clustered distributions of N = 2,000 points, each with its refinement target:
/// the largest of 16, 32 and 64 points per leaf at which its tree has leaves on three
/// levels. The same rules as `nd_fmm_validate::points` (which this crate cannot depend
/// on), with parameters for a small N.
fn clustered_points(rng: &mut SplitMix64) -> [(&'static str, Vec<[f64; 3]>, usize); 3] {
    let centres = [[0.3, 0.3, 0.3], [0.7, 0.6, 0.5]];
    [
        ("sphere", sphere_points(rng, 2000, [0.5; 3], 0.4), 16),
        ("plummer", plummer_points(rng, 2000, [0.5; 3], 0.04), 64),
        ("clusters", cluster_points(rng, 2000, &centres, 0.05), 64),
    ]
}

/// Each clustered distribution at N = 2,000: every strategy at p = 4, Dense at p = 8
/// (module documentation).
fn clustered_distributions(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789e);
    let strategies = [
        M2lStrategy::Dense,
        M2lStrategy::Classes,
        M2lStrategy::Rotation,
    ];
    for (name, points, leaf) in clustered_points(&mut rng) {
        let charges = random_charges(&mut rng, points.len());
        let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
        let (phi, _) = exact(&points, &charges, &sources);
        let sets = (&sources[..], &sources[..]);
        let builder = |p: usize, strategy| {
            FmmBuilder::<f64>::new(p)
                .strategy(strategy)
                .max_points_per_leaf(leaf)
        };
        // p = 4: every strategy, Dense also at 2, 4 and 8 threads.
        let mut outputs: Vec<Vec<f64>> = Vec::new();
        for strategy in strategies {
            let threads: &[usize] = if strategy == M2lStrategy::Dense {
                &THREADS
            } else {
                &[]
            };
            let Some((fmm, output)) =
                evaluate_threaded(&builder(4, strategy), sets, &local_charges, threads, comm)
            else {
                return;
            };
            if strategy == M2lStrategy::Dense {
                let (histogram, [u, v, w, x]) = tree_summary(&fmm, comm);
                eprintln!(
                    "rank {}: {name}, N = {}, {leaf} points per leaf: leaves per level {}; U \
                     {u}, V {v}, W {w}, X {x} pairs",
                    comm.rank(),
                    points.len(),
                    levels_text(&histogram)
                );
                assert!(w > 0 && x > 0, "{name}: W and X lists: {w}, {x}");
                let levels = leaf_levels(&histogram);
                assert!(levels.len() >= 3, "{name}: leaves on levels {levels:?}");
            }
            outputs.push(output.potential);
        }
        for s in 1..strategies.len() {
            let difference = relative_l2(&outputs[s], &outputs[0], comm);
            assert!(
                difference < 1e-12,
                "{name}, p = 4: {:?} differs from Dense by {difference:e}",
                strategies[s]
            );
        }
        // p = 8: Dense at one thread.
        let Some((_, output)) = evaluate_threaded(
            &builder(8, M2lStrategy::Dense),
            sets,
            &local_charges,
            &[],
            comm,
        ) else {
            return;
        };
        let errors = [
            relative_l2(&outputs[0], &phi, comm),
            relative_l2(&output.potential, &phi, comm),
        ];
        eprintln!(
            "rank {}: {name}: relative L2 error of φ {:.3e} (p = 4), {:.3e} (p = 8); at p = \
             4 every strategy within 1e-12 of Dense",
            comm.rank(),
            errors[0],
            errors[1]
        );
        for (k, p) in [4, 8].into_iter().enumerate() {
            assert!(
                errors[k] < CLUSTERED_TOLERANCE[k],
                "{name}, p = {p}: {:e}",
                errors[k]
            );
        }
    }
}

/// The relative L2 error of φ that the clustered scenarios allow at p = 4 and 8: twice
/// the uniform-cube errors of `uniform_cube_strategies` at the same N (6.1e-4 and
/// 1.2e-5), the C3.3 rule, rounded up.
const CLUSTERED_TOLERANCE: [f64; 2] = [1.3e-3, 2.5e-5];

/// A dense blob next to a sparse cloud, p = 8 (module documentation).
fn strongly_graded_tree(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x789f);
    // 100 points over the unit cube and 400 in a cube of side 2⁻⁶ inside it.
    let mut points = unit_cube_points(&mut rng, 100);
    points.extend(
        (0..400)
            .map(|_| -> [f64; 3] { core::array::from_fn(|_| 0.3 + rng.range(0.0, 1.0 / 64.0)) }),
    );
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(8)
        .max_points_per_leaf(32)
        .gradients(true);
    let sets = (&sources[..], &sources[..]);
    // One thread only: p = 8 on this tree is among the most expensive evaluations of the debug
    // run, and the thread counts are covered by the other adaptive scenarios.
    let Some((fmm, output)) = evaluate_threaded(&builder, sets, &local_charges, &[], comm) else {
        return;
    };
    let (histogram, [u, v, w, x]) = tree_summary(&fmm, comm);
    let levels = leaf_levels(&histogram);
    let span = levels.last().unwrap() - levels.first().unwrap();
    let (phi, grad) = exact(&points, &charges, &sources);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(output.gradient.as_ref().unwrap(), &grad, comm);
    eprintln!(
        "rank {}: strongly graded tree, N = {}, p = 8: leaves per level {}; U {u}, V {v}, \
         W {w}, X {x} pairs; relative L2 error {potential_error:.3e} (φ), \
         {gradient_error:.3e} (∇φ)",
        comm.rank(),
        points.len(),
        levels_text(&histogram)
    );
    assert!(span >= 6, "leaves on levels {levels:?}");
    assert!(w > 0 && x > 0, "W and X lists: {w}, {x}");
    assert!(
        potential_error < CLUSTERED_TOLERANCE[1],
        "φ: {potential_error:e}"
    );
    assert!(gradient_error < 1e-4, "∇φ: {gradient_error:e}");
}

/// The unit cube as a supplied domain. With a = 0 and w = 1 the leaf-scaled coordinates
/// u = x 2^(l+1) − (2i + 1) are exact in f64 (CONVENTIONS §3.13).
fn unit_domain() -> PhysicalBox {
    PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
}

/// The relative L2 error of φ over the local targets `selected` only, over all ranks.
fn relative_l2_at(
    approx: &[f64],
    exact: &[f64],
    selected: &[bool],
    comm: &SimpleCommunicator,
) -> f64 {
    let pick = |values: &[f64]| -> Vec<f64> {
        values
            .iter()
            .zip(selected)
            .filter(|&(_, &s)| s)
            .map(|(&v, _)| v)
            .collect()
    };
    relative_l2(&pick(approx), &pick(exact), comm)
}

/// A level-16 leaf with coincident and nearly coincident points beyond
/// `max_points_per_leaf`, sources equal to targets (module documentation).
fn coincident_points(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x78a0);
    // The level-15 box with index `base`: one point at the centre of each of its
    // children 1–7, so with four points per leaf it is refined to level 16, and the
    // cluster in child 0.
    let base = [2usize, 3, 1];
    let child_centre = |o: usize| -> [f64; 3] {
        core::array::from_fn(|k| ((2 * base[k] + ((o >> (2 - k)) & 1)) as f64 + 0.5) / 65536.0)
    };
    let mut points: Vec<[f64; 3]> = (1..8).map(child_centre).collect();
    let (centre, half) = (child_centre(0), 0.5 / 65536.0);
    // Four positions with ten exact copies each.
    for d in 0..4 {
        let x: [f64; 3] = core::array::from_fn(|k| {
            centre[k] + half * 0.2 * (d as f64 - 1.5) * [1.0, -0.5, 0.25][k]
        });
        points.extend(std::iter::repeat_n(x, 10));
    }
    // 60 nearly coincident points, within 10⁻³ of the leaf's half-width of its centre.
    points.extend((0..60).map(|_| -> [f64; 3] {
        core::array::from_fn(|k| centre[k] + half * 1e-3 * rng.range(-1.0, 1.0))
    }));
    let in_cluster = points.len();
    // A cloud without charge, so that every non-zero contribution at a target of the
    // cluster is a P2P term (its own leaf and its siblings): the far field of the cloud
    // cannot hide the error of the near terms. The cloud's targets measure the far
    // field of the cluster.
    points.extend(
        (0..100).map(|_| -> [f64; 3] { core::array::from_fn(|_| rng.range(0.001, 0.999)) }),
    );
    let mut charges = random_charges(&mut rng, in_cluster);
    charges.resize(points.len(), 0.0);
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(4)
        .max_points_per_leaf(4)
        .gradients(true)
        .domain(unit_domain());
    let sets = (&sources[..], &sources[..]);
    let Some((fmm, output)) = evaluate_every_kernel(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };

    // The leaf of the cluster lies on level 16 and holds every point of the cluster.
    let fine = morton::from_physical_point(centre, &unit_domain(), DEEPEST_LEVEL as usize);
    let index = fmm.plan().index();
    let found = index.local_leaf_containing(fine).map(|j| {
        let j = j as usize;
        (index.leaves().level(j), fmm.source_counts()[j])
    });
    if let Some((level, count)) = found {
        assert_eq!((level, count), (16, 100), "the cluster's leaf");
    }
    assert_eq!(
        global_sum(usize::from(found.is_some()), comm),
        1,
        "one rank owns it"
    );

    // Coincident pairs are excluded: every output is finite, and the copies of a point
    // get the same output bit for bit.
    let gradient = output.gradient.as_ref().unwrap();
    assert!(output.potential.iter().all(|v| v.is_finite()));
    assert!(gradient.iter().flatten().all(|v| v.is_finite()));
    for i in 0..sources.len() {
        for j in 0..i {
            if sources[i] == sources[j] {
                assert_eq!(output.potential[i].to_bits(), output.potential[j].to_bits());
                assert_eq!(bits(&gradient[i]), bits(&gradient[j]));
            }
        }
    }

    // The rest matches the direct sum, which skips exactly coincident pairs too.
    let (phi, grad) = exact(&points, &charges, &sources);
    let cluster: Vec<bool> = (0..points.len())
        .skip(comm.rank() as usize)
        .step_by(comm.size() as usize)
        .map(|i| (7..in_cluster).contains(&i))
        .collect();
    let others: Vec<bool> = cluster.iter().map(|&c| !c).collect();
    // At a target of the cluster the near terms, of both signs, can cancel, so its error
    // is measured against the sum of the term magnitudes Σ |q| / (4π |x − y|)
    // (docs/phase1/README.md, "Error measures"), the largest over the cluster: a few
    // roundings per term and the plain summation of about 100 terms.
    let magnitudes: Vec<f64> = charges.iter().map(|q| q.abs()).collect();
    let (terms, _) = exact(&points, &magnitudes, &sources);
    let local = (0..sources.len())
        .filter(|&j| cluster[j])
        .map(|j| (output.potential[j] - phi[j]).abs() / terms[j])
        .fold(0.0f64, f64::max);
    let mut cluster_error = 0.0f64;
    comm.all_reduce_into(&local, &mut cluster_error, SystemOperation::max());
    let other_error = relative_l2_at(&output.potential, &phi, &others, comm);
    let gradient_error = relative_l2_vectors(gradient, &grad, comm);
    eprintln!(
        "rank {}: coincident points: 100 in one level-16 leaf (40 as ten copies of four), \
         {} in all, p = 4: largest error of φ in the cluster {cluster_error:.3e} (terms); \
         relative L2 error of φ at the other targets {other_error:.3e}, of ∇φ at all \
         {gradient_error:.3e}",
        comm.rank(),
        points.len()
    );
    assert!(cluster_error < 1e-14, "cluster: {cluster_error:e}");
    assert!(other_error < 1e-2, "other targets: {other_error:e}");
    assert!(gradient_error < 1e-12, "∇φ: {gradient_error:e}");
}

/// Leaves with sources only next to leaves with targets only (module documentation).
fn one_sided_leaves(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x78a1);
    // A cloud and a blob that straddles level-2 faces, in the unit domain; a point is a
    // source if the indices of its level-2 cell have an even sum, else a target, so
    // every leaf on level 2 or below holds only one kind.
    let mut points: Vec<[f64; 3]> = (0..300)
        .map(|_| core::array::from_fn(|_| rng.range(0.001, 0.999)))
        .collect();
    points.extend((0..300).map(|_| -> [f64; 3] { core::array::from_fn(|_| rng.range(0.2, 0.3)) }));
    let source = |x: &[f64; 3]| x.iter().map(|&c| (4.0 * c) as usize).sum::<usize>() % 2 == 0;
    let sources: Vec<[f64; 3]> = points.iter().filter(|x| source(x)).copied().collect();
    let targets: Vec<[f64; 3]> = points.iter().filter(|x| !source(x)).copied().collect();
    let charges = random_charges(&mut rng, sources.len());
    let (local_sources, local_charges) = (share(&sources, comm), share(&charges, comm));
    let local_targets = share(&targets, comm);
    let builder = FmmBuilder::<f64>::new(6)
        .max_points_per_leaf(8)
        .gradients(true)
        .domain(unit_domain());
    let sets = (&local_sources[..], &local_targets[..]);
    let Some((fmm, output)) = evaluate_threaded(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };

    // Count the one-sided leaves and the U-list pairs between the two kinds.
    let (s, t) = (fmm.source_counts(), fmm.target_counts());
    let leaves = fmm.plan().index().leaves();
    assert!(
        (0..fmm.nleaves()).all(|j| s[j] == 0 || t[j] == 0),
        "a leaf holds both kinds"
    );
    let mut adjacent = 0;
    for level in 0..fmm.nlevels() {
        let near = fmm.plan().level(level).near();
        for (r, j) in leaves.local(level).enumerate() {
            if s[j] > 0 {
                adjacent += near
                    .row(r)
                    .iter()
                    .filter(|&&k| (k as usize) < fmm.nleaves() && t[k as usize] > 0)
                    .count();
            }
        }
    }
    let local = [
        (0..fmm.nleaves()).filter(|&j| s[j] > 0).count(),
        (0..fmm.nleaves()).filter(|&j| t[j] > 0).count(),
        adjacent,
    ];
    let mut counts = [0usize; 3];
    comm.all_reduce_into(&local[..], &mut counts[..], SystemOperation::sum());
    let (histogram, [_, _, w, x]) = tree_summary(&fmm, comm);
    let (phi, grad) = exact(&sources, &charges, &local_targets);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(output.gradient.as_ref().unwrap(), &grad, comm);
    eprintln!(
        "rank {}: one-sided leaves: {} sources, {} targets; {} leaves with sources only, {} \
         with targets only, {} adjacent pairs of the two; leaves per level {}; W {w}, X {x}; \
         p = 6: relative L2 error {potential_error:.3e} (φ), {gradient_error:.3e} (∇φ)",
        comm.rank(),
        sources.len(),
        targets.len(),
        counts[0],
        counts[1],
        counts[2],
        levels_text(&histogram)
    );
    assert!(counts.iter().all(|&n| n > 0), "{counts:?}");
    assert!(w > 0 && x > 0, "W and X lists: {w}, {x}");
    assert!(potential_error < 1e-3, "φ: {potential_error:e}");
    assert!(gradient_error < 1e-2, "∇φ: {gradient_error:e}");
}

/// Points on box faces, edges and corners, and at the corners and faces of the domain
/// (module documentation).
fn faces_and_corners(comm: &SimpleCommunicator) {
    let mut rng = SplitMix64(0x78a2);
    // The coordinates closest to the faces of the unit domain: the smallest positive
    // double and the largest double below 1.
    let (lo, hi) = (f64::from_bits(1), 1.0 - f64::EPSILON / 2.0);
    let ends = [lo, 0.5, hi];
    // The 8 corners, 12 edge midpoints and 6 face centres of the domain (and its
    // centre), each just inside it.
    let mut points: Vec<[f64; 3]> = (0..27)
        .map(|i| [ends[i / 9], ends[(i / 3) % 3], ends[i % 3]])
        .collect();
    // Points with dyadic coordinates j / 2^m, m = 1–10: on the faces, edges and corners
    // of the boxes of level m and below, half of them in a corner blob of side 2⁻⁴, so
    // the tree there is deep.
    for i in 0..400 {
        let shift = if i % 2 == 0 { 0 } else { 4 };
        points.push(core::array::from_fn(|_| {
            let m = 1 + rng.next_u64() % 10;
            let j = 1 + rng.next_u64() % ((1 << m) - 1);
            j as f64 / (1u64 << (m + shift)) as f64
        }));
    }
    let charges = random_charges(&mut rng, points.len());
    let (sources, local_charges) = (share(&points, comm), share(&charges, comm));
    let builder = FmmBuilder::<f64>::new(6)
        .max_points_per_leaf(8)
        .gradients(true)
        .domain(unit_domain());
    let sets = (&sources[..], &sources[..]);
    let Some((fmm, output)) = evaluate_every_kernel(&builder, sets, &local_charges, &THREADS, comm)
    else {
        return;
    };
    let (histogram, [_, _, w, x]) = tree_summary(&fmm, comm);
    let gradient = output.gradient.as_ref().unwrap();
    assert!(output.potential.iter().all(|v| v.is_finite()));
    assert!(gradient.iter().flatten().all(|v| v.is_finite()));
    let (phi, grad) = exact(&points, &charges, &sources);
    let potential_error = relative_l2(&output.potential, &phi, comm);
    let gradient_error = relative_l2_vectors(gradient, &grad, comm);
    eprintln!(
        "rank {}: faces and corners: {} points, leaves per level {}; W {w}, X {x}; p = 6: \
         relative L2 error {potential_error:.3e} (φ), {gradient_error:.3e} (∇φ)",
        comm.rank(),
        points.len(),
        levels_text(&histogram)
    );
    assert!(w > 0 && x > 0, "W and X lists: {w}, {x}");
    assert!(potential_error < 1e-3, "φ: {potential_error:e}");
    assert!(gradient_error < 1e-2, "∇φ: {gradient_error:e}");
}

/// The backend setting (module documentation, "device backends"): the default, a
/// backend not compiled in, its agreement across ranks, and the threads rule of the CPU
/// runtime.
fn device_backends_scenario(comm: &SimpleCommunicator) {
    let rank = comm.rank();
    let mut rng = SplitMix64(0x78a0);
    let points = share(&unit_cube_points(&mut rng, 400), comm);

    // `Host` is the default, with every kind on the host.
    if let Some(fmm) = built(
        FmmBuilder::<f64>::new(3).build(&points, &points, comm),
        comm,
    ) {
        assert_eq!(fmm.backend(), Backend::Host);
        for kind in OperatorKind::ALL {
            assert_eq!(fmm.placement(kind), Placement::Host);
        }
    }
    assert_eq!(
        FmmBuilder::<f64>::new(3).build(&[], &[], comm).err(),
        Some(FmmError::NoPoints)
    );

    // Every backend that is not compiled in is refused at step 1, on every rank; and
    // when only rank 0 asks for one, rank 0 returns the error and the others `OtherRank`:
    // the check is an input error of step 1, agreed by its existing all-reduce.
    let missing: Vec<Backend> = Backend::ALL
        .into_iter()
        .filter(|b| !b.is_compiled())
        .collect();
    for &backend in &missing {
        let error = FmmBuilder::<f64>::new(3)
            .backend(backend)
            .build(&points, &points, comm)
            .err();
        let expected = SettingsError::BackendNotCompiled { backend };
        assert_eq!(
            error,
            Some(FmmError::InvalidSettings(expected)),
            "{backend}"
        );
    }
    if let Some(&backend) = missing.first() {
        let mine = if rank == 0 { backend } else { Backend::Host };
        let error = FmmBuilder::<f64>::new(3)
            .backend(mine)
            .build(&points, &points, comm)
            .err();
        match (rank, &error) {
            (
                0,
                Some(FmmError::InvalidSettings(SettingsError::BackendNotCompiled { backend: b })),
            ) if *b == backend => {}
            (r, Some(FmmError::OtherRank)) if r != 0 => {}
            _ => panic!("rank {rank}: {backend} on rank 0 only: {error:?}"),
        }
    }

    // The threads rule with the CPU runtime: no rayon pool, the units capped at n.
    #[cfg(feature = "cpu")]
    {
        let charges = random_charges(&mut rng, points.len());
        let builder = FmmBuilder::<f64>::new(3).backend(Backend::Cpu).threads(4);
        match builder.build(&points, &points, comm) {
            Ok(mut fmm) => {
                assert_eq!(fmm.threading().threads, 1, "{}", fmm.threading());
                assert_eq!(fmm.operator().threads(), 1);
                let report = fmm.device_report().expect("a device backend");
                assert_eq!(report.cpu_units, Some(4), "{report}");
                let report = report.to_string().replace('\n', "; ");
                let output = fmm
                    .evaluate(&charges)
                    .expect("the CPU-runtime FMM evaluates");
                let mut host = built(
                    FmmBuilder::<f64>::new(3).build(&points, &points, comm),
                    comm,
                )
                .expect("one rank");
                // P2P and the leaf operators on the device (T6, T7): within the FMM bound
                // of the host output, and the units cap changes no bit (each unit owns
                // whole boxes and target leaves).
                let host_output = host.evaluate(&charges).unwrap();
                let (potential, _) = device_common::relative_l2(&output, &host_output);
                assert!(
                    potential <= 1e-12,
                    "threads(4): {potential:e} from the host"
                );
                let want = output_bits(&output);
                let mut one_unit = builder
                    .clone()
                    .threads(1)
                    .build(&points, &points, comm)
                    .expect("one rank");
                assert_eq!(
                    output_bits(&one_unit.evaluate(&charges).unwrap()),
                    want,
                    "one unit against four"
                );
                // The cube layout of the leaf operators (T7; correctness only on the CPU
                // runtime, which allows one unit per core): within the FMM bound of the
                // host output too.
                let units = std::thread::available_parallelism()
                    .map_or(1, |n| n.get() as u32)
                    .min(8);
                let tile = units.min(4);
                let mut cube = builder
                    .clone()
                    .device_leaf_layout(nd_fmm_exec::fmm::DeviceLeafLayout::Cube { units, tile })
                    .build(&points, &points, comm)
                    .expect("one rank");
                assert_eq!(
                    cube.device_report().unwrap().leaf_layout.to_string(),
                    format!("cube ({units} units, tile {tile})")
                );
                let (potential, _) =
                    device_common::relative_l2(&cube.evaluate(&charges).unwrap(), &host_output);
                assert!(
                    potential <= 1e-12,
                    "the cube leaf layout: {potential:e} from the host"
                );
                // The scratch budget of M2M, L2L and M2L (T8, T9): with room for one column per
                // chunk, every level call runs in as many chunks as it has pairs (three
                // launches each), with the same output bit for bit; and the hand-written
                // GEMM, which the CPU runtime runs anyway, gives the same bits.
                let mut chunked = builder
                    .clone()
                    .device_scratch_budget(1)
                    .build(&points, &points, comm)
                    .expect("one rank");
                let translations = &chunked.device_report().unwrap().translations;
                assert!(!translations.is_empty());
                assert!(
                    translations.iter().all(|t| t.chunks == t.pairs),
                    "one column per chunk: {translations:?}"
                );
                let calls = translations.len();
                assert_eq!(
                    output_bits(&chunked.evaluate(&charges).unwrap()),
                    want,
                    "M2M, L2L and M2L in chunks of one column"
                );
                let mut hand = builder
                    .clone()
                    .device_gemm(nd_fmm_exec::fmm::DeviceGemm::HandWritten)
                    .build(&points, &points, comm)
                    .expect("one rank");
                assert_eq!(
                    output_bits(&hand.evaluate(&charges).unwrap()),
                    want,
                    "the hand-written GEMM"
                );
                // Synchronous stages: seven more syncs (after the charge upload and each
                // of the six stages), the same output.
                let syncs = fmm.device_counters().unwrap().evaluation.syncs;
                let mut synchronous = builder
                    .clone()
                    .synchronous_stages(true)
                    .build(&points, &points, comm)
                    .expect("one rank");
                let output = synchronous.evaluate(&charges).unwrap();
                assert_eq!(output_bits(&output), want, "synchronous stages");
                let counters = synchronous.device_counters().unwrap().evaluation;
                assert_eq!(counters.syncs, syncs + 7, "syncs of synchronous stages");
                eprintln!(
                    "rank {rank}: threads(4) with the CPU runtime: no pool; M2M, L2L and M2L in \
                     chunks of one column ({calls} level calls) bit for bit; synchronous \
                     stages: {} syncs against {syncs}; {report}",
                    counters.syncs
                );
            }
            Err(FmmError::PointsNotOwned { .. }) if comm.size() > 1 => {}
            Err(FmmError::InvalidSettings(SettingsError::DeviceNeedsOneRank { ranks }))
                if comm.size() > 1 =>
            {
                assert_eq!(ranks, comm.size() as usize);
            }
            Err(error) => panic!("rank {rank}: CPU runtime with threads(4): {error}"),
        }
    }
    // Points that every rank owns: four in every level-3 box of the unit cube, each rank
    // passing those of its share of the level-1 octants (2, 4 or 8 ranks; one rank all).
    // The partition of the octree then falls on octant boundaries, so the host build
    // succeeds on several ranks, and a device build returns `DeviceNeedsOneRank` on every
    // rank after step 5 (device-path.md §4.4). On one rank the device path runs.
    let size = comm.size() as usize;
    let owned: Vec<[f64; 3]> = (0..512)
        .filter(|&b: &usize| {
            let octant = 4 * (b >> 8) + 2 * ((b >> 5) & 1) + ((b >> 2) & 1);
            8 % size != 0 || octant * size / 8 == rank as usize
        })
        .flat_map(|b| {
            let index = [b >> 6, (b >> 3) & 7, b & 7];
            (0..4)
                .map(|_| core::array::from_fn(|k| (index[k] as f64 + rng.range(0.05, 0.95)) / 8.0))
                .collect::<Vec<[f64; 3]>>()
        })
        .collect();
    let builder = FmmBuilder::<f64>::new(3)
        .max_level(3)
        .max_points_per_leaf(1)
        .domain(PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
    let owned_charges = random_charges(&mut rng, owned.len());
    match builder.build(&owned, &owned, comm) {
        Ok(_) => {
            eprintln!(
                "rank {rank}: device backends: every point owned on {size} rank(s), the host \
                 path builds"
            );
            evaluate_threaded(&builder, (&owned, &owned), &owned_charges, &[], comm)
                .expect("the points are owned");
        }
        Err(FmmError::PointsNotOwned { count }) if size > 1 => eprintln!(
            "rank {rank}: device backends: {count} points not owned on {size} ranks; \
             DeviceNeedsOneRank not reached"
        ),
        Err(error) => panic!("rank {rank}: owned points: {error}"),
    }
    eprintln!(
        "rank {rank}: device backends: Host the default; not compiled in and refused: {}",
        missing
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    );
}
