//! The C3.3 problems at N = 10⁵ (docs/phase3/T11-adaptive-fmm.md): the error of the
//! FMM split by interaction list, every strategy at p = 8, and threads.
//!
//! The problems are those of `nd_fmm_validate::fmm_accuracy::Config::c33`: N = 10⁵
//! points on the unit sphere, in the Plummer sphere of scale 0.1 and in five Gaussian
//! clusters of width 0.02, seed `0xc33`, `max_level` 16 and 64 points per leaf, the
//! domain from `compute_global_bounding_box`, sources equal to targets, 1,000 targets
//! sampled from the same generator. This crate cannot depend on nd-fmm-validate, so the
//! distributions are reproduced here by the same rules, draw for draw. Only the first
//! charge vector (seed `0xc33 + 1`) is used; the example reports the root mean square
//! over eight.
//!
//! **Error per list.** The potential at a target in leaf τ is the sum of four parts,
//! one per list: U (P2P with the near leaves of τ), V (M2L into τ and its ancestors,
//! carried down by L2L and L2P), W (M2P from the W boxes of τ) and X (P2L into τ and
//! its ancestors). [`Masked`], a test wrapper around `LaplaceOperator`, leaves out M2L,
//! M2P or P2L, so the FMM's approximation of the V, W and X parts is the full output
//! minus the masked one, and that of the U part the rest. The exact parts are
//! `direct_sum` over the sources of the pairs of each list, from the plan's own lists:
//! every (target leaf, source leaf) pair belongs to exactly one list, which the test
//! asserts, and the four exact parts add up to the direct sum over all sources.
//!
//! Error measure (docs/phase3/README.md, "Error measures"): at the sampled targets, in
//! the units of the oracle (1/|x − y|, no 4π), for each part c with approximation A_c
//! and exact value E_c, ‖A_c − E_c‖₂ / ‖E‖₂ (its contribution to the relative L2 error of
//! φ), ‖A_c − E_c‖₂ / ‖E_c‖₂ (the relative error of the part) and
//! max |A_c − E_c| / max |E| (the worst target). The test prints them as a Markdown
//! table and asserts that the parts add up to the direct sum (relative L2 1e-13), that
//! the U part is exact up to the rounding of the leaf-scaled coordinates ([`U_BOUND`]),
//! and that the W and X parts each have a relative error within the C3.2 gate at the
//! same p ([`GATE`]), like the V list of a uniform tree.
//!
//! **Strategies and threads at p = 8.** `Dense`, `Classes` and `Rotation` agree to
//! 1e-12 (relative L2 over all targets), and four threads give the one-thread output bit
//! for bit (C3.5).
//!
//! **P2P kernels** (Phase 3S T6, C3S.5). The errors per list are measured with the
//! default P2P kernel, `P2pChoice::Auto`, and again with `P2pChoice::Reference` on the
//! same plan; the table has a row per kernel. Both pass the bounds above; with `Auto` the
//! error of the whole FMM lies within 1% of that with `Reference`, and its output within
//! 1e-13 of the `Reference` output (relative L2 over all points).
//!
//! **Device path** (Phase 4 T6, C4.2; T7, C4.3), with a backend feature: the complete
//! `Fmm` with the default placement (P2M, P2L, L2P, M2P and P2P on the device, M2M, M2L
//! and L2L on the host fallback), on the CPU runtime in f64
//! at every p of [`PS`] (feature `cpu`) and on Metal in f32 at p = 3 and 8 (feature
//! `metal`, by hand outside the macOS sandbox), against the host `Fmm` in the same
//! precision: the relative L2 error of φ at the sampled targets against the direct sum
//! (the four exact parts together) within 0.1% (f64) or 5% (f32) of the host's
//! (docs/phase4/README.md, "Accuracy measures"), and the output within 1e-12 (f64) or 1e-5
//! (f32) of the host output (relative L2 over all points). The device differs in the U
//! list (P2P), the W list (M2P), the X list (P2L) and the leaves' own expansions (P2M,
//! L2P).
//!
//! Its own executable, because it initialises MPI (at `Threading::Funneled`, for the
//! threaded evaluations); ignored, because it needs release mode:
//!
//! ```text
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --release --test adaptive -- --ignored --nocapture
//! RUST_MIN_STACK=8388608 cargo test -p nd-fmm-exec --features cpu --release --test adaptive -- --ignored --nocapture
//! ```

use std::collections::HashMap;

use mpi::Threading;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
#[cfg(feature = "gpu")]
use nd_fmm_exec::fmm::Backend;
use nd_fmm_exec::fmm::{Fmm, FmmBuilder};
use nd_fmm_exec::geometry::{Domain, leaf_coordinates, radius};
use nd_fmm_exec::operator::{Isa, LaplaceOperator, P2pChoice};
use nd_fmm_exec::tables::M2lStrategy;
use nd_fmm_plan::evaluator::Evaluator;
use nd_fmm_plan::operator::{FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p};
use nd_fmm_plan::plan::Plan;
use nd_fmm_ref::p2p::direct_sum;
use nd_octree::constants::DEEPEST_LEVEL;
use nd_octree::{MortonKey, morton};

/// The number of points.
const N: usize = 100_000;

/// The number of sampled targets.
const SAMPLED: usize = 1000;

/// The seed of the points and the sample; the charges use `SEED + 1`.
const SEED: u64 = 0xc33;

/// The degrees: those of the C3.3 gate.
const PS: [usize; 3] = [3, 8, 18];

/// The C3.2 gate, twice the single-translation prediction of the relative L2 error of
/// φ (docs/phase3/README.md, "Predictions"), at each degree of [`PS`].
const GATE: [f64; 3] = [3.54e-3, 2.16e-5, 1.348e-8];

/// The bound on the U part: P2P is exact up to rounding, but it runs on the leaf-scaled
/// coordinates of CONVENTIONS §3.13, whose rounding of x − a, about ε₆₄ |x − a|, the
/// direct sum on the original doubles does not share. Relative to a pair's distance
/// d that is ε₆₄ |x − a| / d; for 10⁵ points on a unit sphere the closest pairs lie
/// about 3e-5 apart, so single terms carry about 1e-11 and the U part about 3e-13
/// (sphere; 4e-14 and 7e-14 for Plummer and the clusters). The coordinates themselves
/// are only known to ε₆₄ |x|, so this stays within the input's own precision.
const U_BOUND: f64 = 1e-12;

/// The threads of the masked evaluations (the output is the same for every count).
const THREADS: usize = 8;

/// How far the error of the FMM with `Auto` may lie from that with `Reference`,
/// relative.
const KERNEL_ERROR_RATIO: f64 = 0.01;

/// The largest relative L2 difference of the `Auto` output from the `Reference` output.
const KERNEL_DIFFERENCE: f64 = 1e-13;

/// SplitMix64, as `nd_fmm_validate::SplitMix64`.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Uniform in 0..n (n ≤ 2³², with a negligible bias).
    fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }
}

/// A point uniform in the unit ball with norm at least `min_norm`
/// (`nd_fmm_validate::points`).
fn in_unit_ball(rng: &mut SplitMix64, min_norm: f64) -> [f64; 3] {
    loop {
        let v: [f64; 3] = core::array::from_fn(|_| rng.range(-1.0, 1.0));
        let norm_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        if norm_sq <= 1.0 && norm_sq >= min_norm * min_norm {
            return v;
        }
    }
}

/// A point at distance `r` from `centre` in a uniform direction (`points::sphere`).
fn at_radius(rng: &mut SplitMix64, centre: [f64; 3], r: f64) -> [f64; 3] {
    let v = in_unit_ball(rng, 1e-3);
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    core::array::from_fn(|i| centre[i] + r * (v[i] / norm))
}

/// A standard normal number (`points::gaussian_clusters`).
fn normal(rng: &mut SplitMix64) -> f64 {
    let radius = (-2.0 * (1.0 - rng.uniform()).ln()).sqrt();
    radius * (2.0 * std::f64::consts::PI * rng.uniform()).cos()
}

/// The C3.3 distributions, by name: the rules and parameters of
/// `nd_fmm_validate::fmm_accuracy::Distribution`.
fn distribution(name: &str, rng: &mut SplitMix64, n: usize) -> Vec<[f64; 3]> {
    match name {
        "sphere" => (0..n).map(|_| at_radius(rng, [0.0; 3], 1.0)).collect(),
        "plummer" => {
            let mass = 1000.0 / 101.0f64.powf(1.5);
            (0..n)
                .map(|_| {
                    let u = mass * rng.uniform();
                    let r = if u > 0.0 {
                        0.1 / (u.powf(-2.0 / 3.0) - 1.0).sqrt()
                    } else {
                        0.0
                    };
                    at_radius(rng, [0.0; 3], r)
                })
                .collect()
        }
        "clusters" => {
            let centres = [
                [-0.5, -0.5, -0.5],
                [0.5, -0.4, 0.3],
                [-0.3, 0.6, 0.2],
                [0.4, 0.5, -0.6],
                [0.6, -0.3, 0.45],
            ];
            let k = centres.len();
            let mut points = Vec::with_capacity(n);
            for (j, c) in centres.iter().enumerate() {
                for _ in 0..n / k + usize::from(j < n % k) {
                    let z = loop {
                        let z: [f64; 3] = core::array::from_fn(|_| normal(rng));
                        if z[0] * z[0] + z[1] * z[1] + z[2] * z[2] <= 16.0 {
                            break z;
                        }
                    };
                    points.push(core::array::from_fn(|i| c[i] + 0.02 * z[i]));
                }
            }
            points
        }
        _ => unreachable!("{name}"),
    }
}

/// The points, the sampled targets (a partial Fisher–Yates shuffle from the same
/// generator) and the first charge vector, as `fmm_accuracy::Problem::new`.
fn problem(name: &str) -> (Vec<[f64; 3]>, Vec<usize>, Vec<f64>) {
    let mut rng = SplitMix64(SEED);
    let points = distribution(name, &mut rng, N);
    let mut order: Vec<usize> = (0..N).collect();
    for i in 0..SAMPLED {
        let j = i + rng.below(N - i);
        order.swap(i, j);
    }
    order.truncate(SAMPLED);
    let mut rng = SplitMix64(SEED + 1);
    let charges = (0..N).map(|_| rng.range(-1.0, 1.0)).collect();
    (points, order, charges)
}

/// The operator kind that [`Masked`] leaves out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Skip {
    /// Nothing: the full FMM.
    Nothing,
    /// M2L, the V list.
    M2l,
    /// M2P, the W list.
    M2p,
    /// P2L, the X list.
    P2l,
}

/// An operator that forwards every call to `inner` except the one of `skip`: a test
/// wrapper that switches one list off.
#[derive(Clone)]
struct Masked<Op> {
    inner: Op,
    skip: Skip,
}

impl<Op: FmmSizes> FmmSizes for Masked<Op> {
    type Value = Op::Value;

    fn multipole_size(&self, level: usize) -> usize {
        self.inner.multipole_size(level)
    }

    fn local_size(&self, level: usize) -> usize {
        self.inner.local_size(level)
    }

    fn source_point_size(&self) -> usize {
        self.inner.source_point_size()
    }

    fn target_input_point_size(&self) -> usize {
        self.inner.target_input_point_size()
    }

    fn target_output_point_size(&self) -> usize {
        self.inner.target_output_point_size()
    }
}

impl<Op: FmmOperator> FmmOperator for Masked<Op> {
    fn p2m(&mut self, batch: P2m<'_, Self::Value>) {
        self.inner.p2m(batch);
    }

    fn m2m(&mut self, batch: M2m<'_, Self::Value>) {
        self.inner.m2m(batch);
    }

    fn m2l(&mut self, batch: M2l<'_, Self::Value>) {
        if self.skip != Skip::M2l {
            self.inner.m2l(batch);
        }
    }

    fn p2l(&mut self, batch: P2l<'_, Self::Value>) {
        if self.skip != Skip::P2l {
            self.inner.p2l(batch);
        }
    }

    fn l2l(&mut self, batch: L2l<'_, Self::Value>) {
        self.inner.l2l(batch);
    }

    fn l2p(&mut self, batch: L2p<'_, Self::Value>) {
        self.inner.l2p(batch);
    }

    fn m2p(&mut self, batch: M2p<'_, Self::Value>) {
        if self.skip != Skip::M2p {
            self.inner.m2p(batch);
        }
    }

    fn p2p(&mut self, batch: P2p<'_, Self::Value>) {
        self.inner.p2p(batch);
    }
}

/// The points of each local leaf of `plan`, in point order.
fn points_by_leaf(points: &[[f64; 3]], plan: &Plan, domain: &Domain) -> Vec<Vec<usize>> {
    let index = plan.index();
    let mut leaves = vec![Vec::new(); index.leaves().nlocal()];
    for (j, &x) in points.iter().enumerate() {
        let fine = morton::from_physical_point(x, &domain.physical_box(), DEEPEST_LEVEL as usize);
        let leaf = index
            .local_leaf_containing(fine)
            .expect("one rank owns every leaf");
        leaves[leaf as usize].push(j);
    }
    leaves
}

/// Runs the evaluator of `plan` with `operator` masked by `skip`, sources equal to
/// targets; returns Σ q / |x − y| (φ̂ / r_t) at every point, in point order.
fn evaluate_masked(
    plan: &Plan,
    comm: &SimpleCommunicator,
    operator: &LaplaceOperator<f64>,
    skip: Skip,
    (points, charges, domain): (&[[f64; 3]], &[f64], &Domain),
    leaves: &[Vec<usize>],
) -> Vec<f64> {
    let counts: Vec<usize> = leaves.iter().map(Vec::len).collect();
    let masked = Masked {
        inner: operator.clone(),
        skip,
    };
    let mut evaluator =
        Evaluator::new(plan, comm, masked, &counts, &counts).expect("the evaluator builds");
    let index = plan.index();
    for (leaf, members) in leaves.iter().enumerate() {
        let key = index.leaf_key(leaf);
        let n = members.len();
        let chunk = evaluator.sources_mut(leaf);
        for (r, &j) in members.iter().enumerate() {
            chunk[3 * r..3 * r + 3]
                .copy_from_slice(&leaf_coordinates::<f64>(points[j], key, domain));
            chunk[3 * n + r] = charges[j];
        }
        let chunk = evaluator.target_input_mut(leaf);
        for (r, &j) in members.iter().enumerate() {
            chunk[3 * r..3 * r + 3]
                .copy_from_slice(&leaf_coordinates::<f64>(points[j], key, domain));
        }
    }
    evaluator.evaluate();
    let mut potential = vec![0.0; points.len()];
    for (leaf, members) in leaves.iter().enumerate() {
        let r = radius(index.leaves().level(leaf), domain);
        for (&phi, &j) in evaluator.target_output(leaf).iter().zip(members) {
            potential[j] = phi / r;
        }
    }
    potential
}

/// The four interaction lists.
const LISTS: [&str; 4] = ["U", "V", "W", "X"];

/// The list of every source box of the pairs of target leaf `leaf`, by key: U for its
/// near leaves (itself included), W for its W boxes, and V and X for the V boxes and X
/// leaves of the leaf and of each of its ancestors. Panics if a key is in two lists.
fn lists_of(plan: &Plan, leaf: usize) -> HashMap<MortonKey, usize> {
    let index = plan.index();
    let level = index.leaves().level(leaf);
    let row = leaf - index.leaves().local(level).start;
    let mut lists = HashMap::new();
    let mut insert = |key: MortonKey, list: usize| {
        let previous = lists.insert(key, list);
        assert!(previous.is_none(), "{key:?} is in two lists of leaf {leaf}");
    };
    for &k in plan.level(level).near().row(row) {
        insert(index.leaf_key(k as usize), 0);
    }
    for &b in plan.level(level).w().row(row) {
        insert(index.key(level + 1, b as usize), 2);
    }
    let mut key = index.leaf_key(leaf);
    loop {
        let (l, i) = index.find(key).expect("every ancestor is held");
        for &s in plan.level(l).v().row(i as usize).0 {
            insert(index.key(l, s as usize), 1);
        }
        for &s in plan.level(l).x().row(i as usize) {
            insert(index.leaf_key(s as usize), 3);
        }
        match morton::parent(key) {
            Some(parent) => key = parent,
            None => break,
        }
    }
    lists
}

/// The exact part of each list at the sampled targets, Σ q / |x − y| over the sources of
/// its pairs, and the number of (target leaf, source leaf) pairs per list over the
/// sampled leaves. Asserts that every pair is in exactly one list.
fn exact_parts(
    plan: &Plan,
    (points, charges): (&[[f64; 3]], &[f64]),
    leaves: &[Vec<usize>],
    sample: &[usize],
) -> ([Vec<f64>; 4], [usize; 4]) {
    let index = plan.index();
    let mut leaf_of = vec![0; points.len()];
    for (leaf, members) in leaves.iter().enumerate() {
        for &j in members {
            leaf_of[j] = leaf;
        }
    }
    let mut by_leaf: HashMap<usize, Vec<usize>> = HashMap::new();
    for (r, &i) in sample.iter().enumerate() {
        by_leaf.entry(leaf_of[i]).or_default().push(r);
    }
    let mut parts: [Vec<f64>; 4] = core::array::from_fn(|_| vec![0.0; sample.len()]);
    let mut pairs = [0usize; 4];
    for (&leaf, rows) in &by_leaf {
        let lists = lists_of(plan, leaf);
        let mut sources: [(Vec<[f64; 3]>, Vec<f64>); 4] = Default::default();
        for (s, members) in leaves.iter().enumerate() {
            let mut found = Vec::new();
            let mut key = index.leaf_key(s);
            loop {
                if let Some(&list) = lists.get(&key) {
                    found.push(list);
                }
                match morton::parent(key) {
                    Some(parent) => key = parent,
                    None => break,
                }
            }
            assert_eq!(
                found.len(),
                1,
                "source leaf {s} of target leaf {leaf}: {found:?}"
            );
            pairs[found[0]] += 1;
            let (x, q) = &mut sources[found[0]];
            x.extend(members.iter().map(|&j| points[j]));
            q.extend(members.iter().map(|&j| charges[j]));
        }
        let targets: Vec<[f64; 3]> = rows.iter().map(|&r| points[sample[r]]).collect();
        for (c, (x, q)) in sources.iter().enumerate() {
            let mut phi = vec![0.0; targets.len()];
            direct_sum(x, q, &targets, &mut phi, None);
            for (&r, &v) in rows.iter().zip(&phi) {
                parts[c][r] = v;
            }
        }
    }
    (parts, pairs)
}

/// ‖v‖₂.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The relative L2 difference of two outputs over all points.
fn relative_l2(a: &[f64], b: &[f64]) -> f64 {
    let difference: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    norm(&difference) / norm(b)
}

/// The errors per list of one distribution at degree `p` (module documentation), with
/// the operator of `fmm` on the P2P kernel `kernel`, printed as table rows. Returns the
/// contribution ‖A_c − E_c‖₂ / ‖E‖₂ of each list and of all four, the relative error
/// ‖A_c − E_c‖₂ / ‖E_c‖₂ of each list, and the full output at every point.
fn list_errors(
    (name, kernel): (&str, P2pChoice),
    fmm: &Fmm<'_, f64>,
    comm: &SimpleCommunicator,
    (points, charges, sample): (&[[f64; 3]], &[f64], &[usize]),
    (leaves, parts): (&[Vec<usize>], &[Vec<f64>; 4]),
) -> ([f64; 5], [f64; 4], Vec<f64>) {
    let plan = fmm.plan();
    let input = (points, charges, fmm.domain());
    let operator = fmm.operator().clone().with_p2p(kernel).unwrap();
    assert_eq!(operator.p2p_kernel(), kernel.resolve().unwrap());
    let run = |skip| evaluate_masked(plan, comm, &operator, skip, input, leaves);
    let at_sample = |v: &[f64]| -> Vec<f64> { sample.iter().map(|&i| v[i]).collect() };
    let all = run(Skip::Nothing);
    let full = at_sample(&all);
    let without = [Skip::M2l, Skip::M2p, Skip::P2l].map(|skip| at_sample(&run(skip)));
    // A_V, A_W, A_X: the full output minus the masked one; A_U: the rest.
    let mut approx: [Vec<f64>; 4] = Default::default();
    for (c, masked) in without.iter().enumerate() {
        approx[c + 1] = full.iter().zip(masked).map(|(f, m)| f - m).collect();
    }
    approx[0] = (0..sample.len())
        .map(|r| full[r] - approx[1][r] - approx[2][r] - approx[3][r])
        .collect();
    let exact: Vec<f64> = (0..sample.len())
        .map(|r| parts.iter().map(|part| part[r]).sum())
        .collect();
    let (exact_norm, exact_max) = (
        norm(&exact),
        exact.iter().fold(0.0f64, |m, v| m.max(v.abs())),
    );
    let mut contributions = [0.0; 5];
    let mut relative = [0.0; 4];
    let mut rows = Vec::new();
    for c in 0..4 {
        let error: Vec<f64> = approx[c]
            .iter()
            .zip(&parts[c])
            .map(|(a, e)| a - e)
            .collect();
        let worst = error.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        contributions[c] = norm(&error) / exact_norm;
        relative[c] = norm(&error) / norm(&parts[c]);
        rows.push(format!(
            "| {name} | {} | {kernel} | {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} |",
            fmm.p(),
            LISTS[c],
            norm(&parts[c]) / exact_norm,
            contributions[c],
            relative[c],
            worst / exact_max
        ));
    }
    let error: Vec<f64> = full.iter().zip(&exact).map(|(a, e)| a - e).collect();
    contributions[4] = norm(&error) / exact_norm;
    rows.push(format!(
        "| {name} | {} | {kernel} | all | 1 | {:.3e} | {:.3e} | {:.3e} |",
        fmm.p(),
        contributions[4],
        contributions[4],
        error.iter().fold(0.0f64, |m, v| m.max(v.abs())) / exact_max
    ));
    for row in rows {
        println!("{row}");
    }
    (contributions, relative, all)
}

/// Every strategy at p = 8 agrees with Dense, and four threads give the one-thread
/// output bit for bit (module documentation).
fn strategies_and_threads(
    name: &str,
    points: &[[f64; 3]],
    charges: &[f64],
    comm: &SimpleCommunicator,
) {
    let build = |strategy, threads| {
        FmmBuilder::<f64>::new(8)
            .strategy(strategy)
            .threads(threads)
            .build(points, points, comm)
            .expect("the FMM builds")
    };
    let dense = build(M2lStrategy::Dense, 1)
        .evaluate(charges)
        .unwrap()
        .potential;
    let threaded = build(M2lStrategy::Dense, 4)
        .evaluate(charges)
        .unwrap()
        .potential;
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&threaded), bits(&dense), "{name}: four threads");
    for strategy in [M2lStrategy::Classes, M2lStrategy::Rotation] {
        let output = build(strategy, 1).evaluate(charges).unwrap().potential;
        let difference = relative_l2(&output, &dense);
        eprintln!("{name}, p = 8: {strategy:?} against Dense: {difference:.3e} (relative L2)");
        assert!(difference < 1e-12, "{name}: {strategy:?}: {difference:e}");
    }
}

/// The device path on each backend compiled in (module documentation, "Device path"):
/// the CPU runtime in f64 at every p of [`PS`], Metal in f32 at p = 3 and 8.
#[cfg(feature = "gpu")]
fn device_gates(
    name: &str,
    (points, charges): (&[[f64; 3]], &[f64]),
    (sample, exact): (&[usize], &[f64]),
    comm: &SimpleCommunicator,
    failures: &mut Vec<String>,
) {
    let mut ran = Vec::new();
    if Backend::Cpu.is_compiled() {
        for p in PS {
            device_gate::<f64>(
                name,
                Backend::Cpu,
                p,
                (points, charges),
                (sample, exact),
                comm,
                failures,
            );
        }
        ran.push("cpu (f64)");
    }
    if Backend::Metal.is_compiled() {
        for p in [3, 8] {
            device_gate::<f32>(
                name,
                Backend::Metal,
                p,
                (points, charges),
                (sample, exact),
                comm,
                failures,
            );
        }
        ran.push("metal (f32)");
    }
    eprintln!(
        "{name}: backends run: host, {}; not run: cuda (type-checked, not run)",
        ran.join(", ")
    );
}

/// The host and the device `Fmm` in `T` at degree `p` against the exact potential at the
/// sample (module documentation, "Device path").
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn device_gate<
    T: nd_fmm_tables::cache::Stored
        + nd_fmm_exec::operator::SimdScalar
        + Equivalence
        + Default
        + nd_fmm_math::RealScalar,
>(
    name: &str,
    backend: Backend,
    p: usize,
    (points, charges): (&[[f64; 3]], &[f64]),
    (sample, exact): (&[usize], &[f64]),
    comm: &SimpleCommunicator,
    failures: &mut Vec<String>,
) {
    let f64_run = size_of::<T>() == 8;
    let (error_ratio, difference_bound) = if f64_run { (1e-3, 1e-12) } else { (5e-2, 1e-5) };
    let q: Vec<T> = charges.iter().map(|&v| T::from_f64(v)).collect();
    let widen = |v: &[T]| -> Vec<f64> {
        v.iter()
            .map(|&x| 4.0 * std::f64::consts::PI * x.to_f64())
            .collect()
    };
    let builder = FmmBuilder::<T>::new(p);
    let host = widen(
        &builder
            .clone()
            .threads(THREADS)
            .build(points, points, comm)
            .expect("the host FMM builds")
            .evaluate(&q)
            .expect("the host FMM evaluates")
            .potential,
    );
    let mut fmm = builder
        .clone()
        .backend(backend)
        .build(points, points, comm)
        .unwrap_or_else(|error| panic!("{backend}: the device FMM does not build: {error}"));
    let report = fmm.device_report().expect("a device backend");
    let layout = format!(
        "P2P {}, leaf operators {}",
        report.p2p_layout, report.leaf_layout
    );
    let device = widen(
        &fmm.evaluate(&q)
            .expect("the device FMM evaluates")
            .potential,
    );
    let at_sample = |v: &[f64]| -> Vec<f64> { sample.iter().map(|&i| v[i]).collect() };
    let error = |v: &[f64]| relative_l2(&at_sample(v), exact);
    let (error_host, error_device) = (error(&host), error(&device));
    let ratio = error_device / error_host;
    let difference = relative_l2(&device, &host);
    eprintln!(
        "{name}, {backend}, {}, p = {p}, P2P and leaf operators on the device ({layout}): relative L2 error of φ \
         {error_device:.4e} against the host's {error_host:.4e} (ratio {ratio:.6}); output \
         within {difference:.1e} of the host's",
        if f64_run { "f64" } else { "f32" }
    );
    if (ratio - 1.0).abs() > error_ratio {
        failures.push(format!(
            "{name}, {backend}, p = {p}: the device error is {ratio} times the host's"
        ));
    }
    if difference > difference_bound {
        failures.push(format!(
            "{name}, {backend}, p = {p}: the device output differs from the host's by \
             {difference:e}"
        ));
    }
}

#[test]
#[ignore = "release mode: N = 10^5, four evaluations per p and the direct sums per list"]
fn errors_per_list_on_clustered_trees() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("this test owns MPI initialization");
    assert!(provided >= Threading::Funneled, "MPI provides {provided:?}");
    let comm = universe.world();
    assert_eq!(
        comm.size(),
        1,
        "one rank: the points are not redistributed (C5.1)"
    );

    println!(
        "| distribution | p | P2P | list | ‖E_c‖ / ‖E‖ | ‖A_c − E_c‖ / ‖E‖ | ‖A_c − E_c‖ / ‖E_c‖ | max |A_c − E_c| / max |E| |"
    );
    println!("|---|---:|---|---|---:|---:|---:|---:|");
    let detected = P2pChoice::Isa(Isa::detect());
    let mut failures = Vec::new();
    for name in ["sphere", "plummer", "clusters"] {
        let (points, sample, charges) = problem(name);
        let mut parts = None;
        for p in PS {
            let fmm = FmmBuilder::<f64>::new(p)
                .threads(THREADS)
                .build(&points, &points, &comm)
                .expect("the FMM builds");
            let leaves = points_by_leaf(&points, fmm.plan(), fmm.domain());
            let sizes = fmm.list_sizes();
            assert!(sizes.w > 0 && sizes.x > 0, "{name}: {sizes:?}");
            // The tree, and so the exact parts, are the same for every p.
            let (parts, pairs) = parts.get_or_insert_with(|| {
                let parts = exact_parts(fmm.plan(), (&points, &charges), &leaves, &sample);
                let mut all = vec![0.0; sample.len()];
                let targets: Vec<[f64; 3]> = sample.iter().map(|&i| points[i]).collect();
                direct_sum(&points, &charges, &targets, &mut all, None);
                let sum: Vec<f64> = (0..sample.len())
                    .map(|r| parts.0.iter().map(|c| c[r]).sum())
                    .collect();
                let difference = relative_l2(&sum, &all);
                eprintln!(
                    "{name}: {} leaves; pairs per list over the sampled leaves {:?}; the \
                     parts add up to the direct sum to {difference:.1e}; lists {sizes:?}",
                    fmm.nleaves(),
                    parts.1
                );
                assert!(difference < 1e-13, "{name}: {difference:e}");
                parts
            });
            assert!(pairs.iter().all(|&n| n > 0), "{name}: {pairs:?}");
            let input = (&points[..], &charges[..], &sample[..]);
            assert_eq!(fmm.p2p_kernel(), detected, "Auto by default");
            let mut alls = Vec::new();
            let mut totals = Vec::new();
            for kernel in [P2pChoice::Auto, P2pChoice::Reference] {
                let (contributions, relative, all) =
                    list_errors((name, kernel), &fmm, &comm, input, (&leaves, parts));
                // U is P2P alone, exact up to the rounding of [`U_BOUND`]. Each adaptive
                // list on its own is within the C3.2 gate, as the V list of a uniform
                // tree.
                if contributions[0] > U_BOUND {
                    failures.push(format!(
                        "{name}, p = {p}, {kernel}: U part {:e}",
                        contributions[0]
                    ));
                }
                let gate = GATE[PS.iter().position(|&q| q == p).unwrap()];
                for c in [2, 3] {
                    if relative[c] > gate {
                        failures.push(format!(
                            "{name}, p = {p}, {kernel}: {} part {:e} > {gate:e}",
                            LISTS[c], relative[c]
                        ));
                    }
                }
                alls.push(all);
                totals.push(contributions[4]);
            }
            // Auto against Reference: the error of the FMM, and the output.
            let (ratio, difference) = (totals[0] / totals[1], relative_l2(&alls[0], &alls[1]));
            eprintln!(
                "{name}, p = {p}: relative L2 error of φ {:.3e} with {detected}, {:.3e} with \
                 reference (ratio {ratio:.6}); outputs within {difference:.1e}",
                totals[0], totals[1]
            );
            if (ratio - 1.0).abs() > KERNEL_ERROR_RATIO {
                failures.push(format!(
                    "{name}, p = {p}: the error with {detected} is {ratio} times that with \
                     reference"
                ));
            }
            if difference > KERNEL_DIFFERENCE {
                failures.push(format!(
                    "{name}, p = {p}: {detected} differs from reference by {difference:e}"
                ));
            }
        }
        #[cfg(feature = "gpu")]
        {
            let (parts, _) = parts.as_ref().expect("measured above");
            let exact: Vec<f64> = (0..sample.len())
                .map(|r| parts.iter().map(|part| part[r]).sum())
                .collect();
            device_gates(
                name,
                (&points, &charges),
                (&sample, &exact),
                &comm,
                &mut failures,
            );
        }
        strategies_and_threads(name, &points, &charges, &comm);
    }
    assert!(failures.is_empty(), "{}", failures.join("; "));
}
