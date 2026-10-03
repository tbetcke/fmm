//! The device P2P (T6, C4.2) on every layout ([`P2pLayout`]) the backend runs, in f32
//! and, where the device supports it, f64, potential only and with gradients.
//!
//! Error measures (docs/phase4/README.md, "Accuracy measures", the signed-off device P2P
//! contract, C3S.4 unchanged):
//! - **pair terms**, against `nd_fmm_ref::p2p` in the same precision: the potential
//!   relative to the reference's term, within 8 u_T; each gradient component relative to
//!   |q| / r², within 16 u_T (in f32 over 2⁻⁸⁴ ≤ r² ≤ 2⁷ and |q| ≤ 1, CONVENTIONS §3.13).
//!   On Metal, which flushes subnormals, the charges are 0 or |q| ≥ 2⁻¹⁰⁰.
//! - **sums**, against `direct_sum` in f64 on the inputs as rounded to T (the mapped ŷ
//!   of the host, fl(ĉ + r̂ u_s)): per target the error relative to the sum of its term
//!   magnitudes (Σ|q|/r for φ, Σ|q|/r² per component of ∇φ), within 1e-6 (f32) or 1e-14
//!   (f64), or within twice the error of `nd_fmm_ref::p2p` on the same inputs in the
//!   same order where that is larger.
//! - **bits**: exact equality of the bit patterns (frames, invariance, determinism,
//!   empty calls).
//!
//! Every test prints the device and, per layout and precision, its worst errors.

use nd_fmm_kernels::p2p::{P2pInputs, P2pLayout, near_frames, p2p};
use nd_fmm_kernels::view::{IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p::direct_sum;

use crate::common::{Rng, TestFloat, tests_on};

/// The float types of the tests, with their unit roundoff.
trait Real: TestFloat + RealScalar {
    /// u_T: 2⁻²⁴ or 2⁻⁵³.
    const U: f64;
}

impl Real for f32 {
    const U: f64 = 1.0 / 16_777_216.0;
}

impl Real for f64 {
    const U: f64 = 1.0 / 9_007_199_254_740_992.0;
}

/// `x` rounded to `T`.
fn t<T: Real>(x: f64) -> T {
    <T as RealScalar>::from_f64(x)
}

/// `x` widened to f64.
fn w<T: Real>(x: T) -> f64 {
    <T as RealScalar>::to_f64(x)
}

/// The sum tolerance: 1e-6 in f32, 1e-14 in f64.
fn sum_tolerance<T: Real>() -> f64 {
    if T::FLOAT == Precision::F32 {
        1e-6
    } else {
        1e-14
    }
}

/// The layouts a backend runs in these tests: its default and the others. On the CPU
/// runtime the CPU layout runs with 128- and 256-bit vectors (the defaults of aarch64 and
/// x86_64, so that either host tests both), the cube layout has up to 8 units (it allows
/// at most one unit per core: 4 on the CI runner) and the plane layout's planes are one
/// unit, up to 4 of them.
fn layouts(device: &Device) -> Vec<P2pLayout> {
    match device.backend() {
        BackendKind::Cpu => {
            let units = device.info().max_units_per_cube.clamp(1, 8);
            vec![
                P2pLayout::Cpu { vector_bits: 128 },
                P2pLayout::Cpu { vector_bits: 256 },
                P2pLayout::Cube { units },
                P2pLayout::Plane {
                    planes: units.min(4),
                },
            ]
        }
        _ => vec![
            P2pLayout::Cube { units: 64 },
            P2pLayout::Plane { planes: 2 },
            P2pLayout::Cpu { vector_bits: 128 },
        ],
    }
}

/// A leaf of a test problem: its level and index, its sources (leaf-scaled, with
/// charges) and its targets (leaf-scaled).
#[derive(Clone, Debug)]
struct Leaf<T> {
    key: (u32, [u32; 3]),
    sources: Vec<[T; 3]>,
    charges: Vec<T>,
    targets: Vec<[T; 3]>,
}

impl<T: Real> Leaf<T> {
    fn new(key: (u32, [u32; 3])) -> Self {
        Self {
            key,
            sources: Vec::new(),
            charges: Vec::new(),
            targets: Vec::new(),
        }
    }
}

/// One level's call: the leaves, and the near rows of the target leaves
/// `first..first + rows.len()`.
#[derive(Clone, Debug)]
struct Problem<T> {
    leaves: Vec<Leaf<T>>,
    first: usize,
    rows: Vec<Vec<u32>>,
}

/// The frame (ĉ(s|t), r̂(s|t)) of CONVENTIONS §3.13 on the host, in f64 (exact).
fn frame(s: (u32, [u32; 3]), t: (u32, [u32; 3])) -> ([f64; 3], f64) {
    let big = s.0.max(t.0);
    let scale = 2f64.powi(t.0 as i32 - big as i32);
    let centre = std::array::from_fn(|k| {
        let cs = (2 * i64::from(s.1[k]) + 1) << (big - s.0);
        let ct = (2 * i64::from(t.1[k]) + 1) << (big - t.0);
        (cs - ct) as f64 * scale
    });
    (centre, 2f64.powi(t.0 as i32 - s.0 as i32))
}

impl<T: Real> Problem<T> {
    /// The sources of row r in the order the kernel adds them, mapped into the frame of
    /// the row's leaf on the host: ŷ = fl(ĉ + r̂ u_s) (r̂ u_s exact), u_s itself for the
    /// leaf itself.
    fn row_sources(&self, r: usize) -> (Vec<[T; 3]>, Vec<T>) {
        let target = self.first + r;
        let (mut y, mut q) = (Vec::new(), Vec::new());
        for &j in &self.rows[r] {
            let leaf = &self.leaves[j as usize];
            if j as usize == target {
                y.extend(&leaf.sources);
            } else {
                let (c, ratio) = frame(leaf.key, self.leaves[target].key);
                let (c, ratio) = (c.map(t::<T>), t::<T>(ratio));
                y.extend(
                    leaf.sources
                        .iter()
                        .map(|u| std::array::from_fn(|k| c[k] + ratio * u[k])),
                );
            }
            q.extend(&leaf.charges);
        }
        (y, q)
    }

    /// The point offsets of the source store and of the target stores.
    fn offsets(&self) -> (Vec<u32>, Vec<u32>) {
        let scan = |count: &dyn Fn(&Leaf<T>) -> usize| -> Vec<u32> {
            std::iter::once(0)
                .chain(self.leaves.iter().scan(0u32, |total, leaf| {
                    *total += count(leaf) as u32;
                    Some(*total)
                }))
                .collect()
        };
        (scan(&|l| l.sources.len()), scan(&|l| l.targets.len()))
    }

    /// The target output store, every value `value(leaf, i)` for the i-th value of the
    /// leaf's chunk.
    fn output_store(&self, gradients: bool, mut value: impl FnMut(usize, usize) -> T) -> Vec<T> {
        let per_point = if gradients { 4 } else { 1 };
        self.leaves
            .iter()
            .enumerate()
            .flat_map(|(j, leaf)| (0..per_point * leaf.targets.len()).map(move |i| (j, i)))
            .map(|(j, i)| value(j, i))
            .collect::<Vec<T>>()
    }

    /// The value range of leaf j's chunk in the target output store.
    fn output_range(&self, j: usize, gradients: bool) -> std::ops::Range<usize> {
        let per_point = if gradients { 4 } else { 1 };
        let before: usize = self.leaves[..j].iter().map(|l| l.targets.len()).sum();
        per_point * before..per_point * (before + self.leaves[j].targets.len())
    }
}

/// A problem on the device.
struct OnDevice<T: Real> {
    near: IndexView,
    leaves: LeafCoordinates,
    source_offsets: PointOffsets,
    target_offsets: PointOffsets,
    sources: DeviceBuffer<T>,
    target_input: DeviceBuffer<T>,
    output: DeviceBuffer<T>,
    first: usize,
}

/// Uploads `problem` with the target output `output`, and its rows as the near view.
fn upload<T: Real>(device: &mut Device, problem: &Problem<T>, output: &[T]) -> OnDevice<T> {
    upload_rows(device, problem, &problem.rows, output)
}

/// Uploads `problem` with the near view `rows` instead of its own.
fn upload_rows<T: Real>(
    device: &mut Device,
    problem: &Problem<T>,
    rows: &[Vec<u32>],
    output: &[T],
) -> OnDevice<T> {
    let (so, to) = problem.offsets();
    let mut sources = Vec::new();
    let mut input = Vec::new();
    for leaf in &problem.leaves {
        sources.extend(leaf.sources.iter().flatten());
        sources.extend(&leaf.charges);
        input.extend(leaf.targets.iter().flatten());
    }
    let row_offsets: Vec<u32> = std::iter::once(0)
        .chain(rows.iter().scan(0u32, |total, row| {
            *total += row.len() as u32;
            Some(*total)
        }))
        .collect();
    let entries: Vec<u32> = rows.iter().flatten().copied().collect();
    let keys: Vec<(u32, [u32; 3])> = problem.leaves.iter().map(|l| l.key).collect();
    OnDevice {
        near: IndexView::upload(device, &row_offsets, &entries, problem.leaves.len()).unwrap(),
        leaves: LeafCoordinates::upload(device, &keys).unwrap(),
        source_offsets: PointOffsets::upload(device, &so).unwrap(),
        target_offsets: PointOffsets::upload(device, &to).unwrap(),
        sources: device.upload(&sources).unwrap(),
        target_input: device.upload(&input).unwrap(),
        output: device.upload(output).unwrap(),
        first: problem.first,
    }
}

/// Launches the P2P of `on` in `layout`, adding into its output.
fn launch<T: Real>(device: &mut Device, on: &mut OnDevice<T>, layout: P2pLayout, gradients: bool) {
    let inputs = P2pInputs {
        near: &on.near,
        first_leaf: on.first,
        leaves: &on.leaves,
        source_offsets: &on.source_offsets,
        sources: on.sources.as_slice(),
        target_offsets: &on.target_offsets,
        target_input: on.target_input.as_slice(),
    };
    p2p(device, layout, gradients, &inputs, on.output.as_slice_mut()).unwrap();
}

/// The target output store of `on`, downloaded.
fn download<T: Real>(device: &mut Device, on: &OnDevice<T>) -> Vec<T> {
    let mut out = vec![t::<T>(0.0); on.output.len()];
    device.download(on.output.as_slice(), &mut out).unwrap();
    out
}

/// Runs `problem` from a zero output and returns the output store.
fn run<T: Real>(
    device: &mut Device,
    problem: &Problem<T>,
    layout: P2pLayout,
    gradients: bool,
) -> Vec<T> {
    let zeros = problem.output_store(gradients, |_, _| t::<T>(0.0));
    let mut on = upload(device, problem, &zeros);
    launch(device, &mut on, layout, gradients);
    download(device, &on)
}

/// The worst errors of a set of outputs.
#[derive(Clone, Copy, Debug, Default)]
struct Worst {
    potential: f64,
    gradient: f64,
    reference_potential: f64,
    reference_gradient: f64,
    targets: usize,
    /// f32 targets with a pair closer than r² = 2⁻⁸⁴, whose gradient is not checked.
    outside_gradient_range: usize,
}

impl Worst {
    fn print(&self, what: &str) {
        println!(
            "    {what}: max φ {:.2e} (reference {:.2e}), max ∇φ {:.2e} (reference {:.2e}), \
             {} targets{}",
            self.potential,
            self.reference_potential,
            self.gradient,
            self.reference_gradient,
            self.targets,
            if self.outside_gradient_range > 0 {
                format!(
                    " ({} with a pair below the f32 gradient range, ∇φ not checked)",
                    self.outside_gradient_range
                )
            } else {
                String::new()
            }
        );
    }
}

/// Splits a target output chunk of n targets into φ̂ and, with gradients, ĝ.
fn split<T: Real>(chunk: &[T], gradients: bool) -> (Vec<T>, Vec<[T; 3]>) {
    let n = if gradients {
        chunk.len() / 4
    } else {
        chunk.len()
    };
    let phi = chunk[..n].to_vec();
    let grad = if gradients {
        chunk[n..].as_chunks::<3>().0.to_vec()
    } else {
        Vec::new()
    };
    (phi, grad)
}

/// Checks every row of `problem` in `output` (computed from a zero output) against
/// `direct_sum` within the sum contract, and returns the worst errors.
fn check_sums<T: Real>(what: &str, problem: &Problem<T>, output: &[T], gradients: bool) -> Worst {
    check_sums_from(what, problem, None, output, gradients)
}

/// [`check_sums`] for an output computed from `initial` (zero if `None`): the oracle and
/// the reference start from it, and the term magnitudes include it.
fn check_sums_from<T: Real>(
    what: &str,
    problem: &Problem<T>,
    initial: Option<&[T]>,
    output: &[T],
    gradients: bool,
) -> Worst {
    let mut worst = Worst::default();
    for r in 0..problem.rows.len() {
        let j = problem.first + r;
        let targets = &problem.leaves[j].targets;
        if targets.is_empty() {
            continue;
        }
        let (y, q) = problem.row_sources(r);
        let range = problem.output_range(j, gradients);
        let (phi, grad) = split(&output[range.clone()], gradients);
        let n = targets.len();
        let (phi0, grad0) = match initial {
            Some(initial) => {
                let (p, g) = split(&initial[range], gradients);
                (
                    p,
                    if gradients {
                        g
                    } else {
                        vec![[t::<T>(0.0); 3]; n]
                    },
                )
            }
            None => (vec![t::<T>(0.0); n], vec![[t::<T>(0.0); 3]; n]),
        };
        let up = |v: &[[T; 3]]| -> Vec<[f64; 3]> { v.iter().map(|p| p.map(w::<T>)).collect() };
        let (y64, x64) = (up(&y), up(targets));
        let q64: Vec<f64> = q.iter().map(|&v| w(v)).collect();
        let mut ophi: Vec<f64> = phi0.iter().map(|&v| w(v)).collect();
        let mut ograd: Vec<[f64; 3]> = grad0.iter().map(|g| g.map(w::<T>)).collect();
        direct_sum(&y64, &q64, &x64, &mut ophi, Some(&mut ograd));
        let (mut rphi, mut rgrad) = (phi0.clone(), grad0.clone());
        nd_fmm_ref::p2p::p2p(&y, &q, targets, &mut rphi, Some(&mut rgrad));
        let (mut row_worst, mut row_reference) = ([0.0f64; 2], [0.0f64; 2]);
        for i in 0..n {
            let mut mp = w(phi0[i]).abs();
            let mut mg = grad0[i].iter().fold(0.0f64, |m, &v| m.max(w(v).abs()));
            let mut closest = f64::INFINITY;
            for (yk, qk) in y64.iter().zip(&q64) {
                let r2: f64 = (0..3).map(|k| (x64[i][k] - yk[k]).powi(2)).sum();
                if r2 > 0.0 {
                    mp += qk.abs() / r2.sqrt();
                    mg += qk.abs() / r2;
                    closest = closest.min(r2);
                }
            }
            // In f32 the gradient contract holds for 2⁻⁸⁴ ≤ r² only (CONVENTIONS §3.13,
            // "Range of the terms in f32"); the reference's term is inaccurate or infinite
            // below it too.
            let gradient_in_range = T::FLOAT == Precision::F64 || closest >= 2f64.powi(-84);
            if gradients && !gradient_in_range {
                worst.outside_gradient_range += 1;
            }
            assert!(
                w(phi[i]).is_finite(),
                "{what}: row {r}, target {i}: φ = {:?}",
                phi[i]
            );
            if mp > 0.0 {
                row_worst[0] = row_worst[0].max((w(phi[i]) - ophi[i]).abs() / mp);
                row_reference[0] = row_reference[0].max((w(rphi[i]) - ophi[i]).abs() / mp);
            }
            if gradients && gradient_in_range {
                for k in 0..3 {
                    assert!(
                        w(grad[i][k]).is_finite(),
                        "{what}: row {r}, target {i}: ∇φ = {:?}",
                        grad[i]
                    );
                    if mg > 0.0 {
                        row_worst[1] = row_worst[1].max((w(grad[i][k]) - ograd[i][k]).abs() / mg);
                        row_reference[1] =
                            row_reference[1].max((w(rgrad[i][k]) - ograd[i][k]).abs() / mg);
                    }
                }
            }
        }
        for (c, name) in [(0, "φ"), (1, "∇φ")] {
            let bound = sum_tolerance::<T>().max(2.0 * row_reference[c]);
            assert!(
                row_worst[c] <= bound,
                "{what}: row {r} ({n} targets, {} sources): {name} error {:e} exceeds {bound:e} \
                 (reference {:e})",
                y.len(),
                row_worst[c],
                row_reference[c]
            );
        }
        worst.potential = worst.potential.max(row_worst[0]);
        worst.gradient = worst.gradient.max(row_worst[1]);
        worst.reference_potential = worst.reference_potential.max(row_reference[0]);
        worst.reference_gradient = worst.reference_gradient.max(row_reference[1]);
        worst.targets += n;
    }
    worst
}

/// Asserts two output stores equal bit for bit.
fn assert_same_bits<T: Real>(what: &str, got: &[T], want: &[T]) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if let Some(i) = (0..got.len()).find(|&i| got[i].bits() != want[i].bits()) {
        panic!(
            "{what}: {} of {} values differ, the first at {i}: {:?} against {:?}",
            (0..got.len())
                .filter(|&i| got[i].bits() != want[i].bits())
                .count(),
            got.len(),
            got[i],
            want[i]
        );
    }
}

/// The precisions `device` runs, with a closure per precision.
macro_rules! each_precision {
    ($device:expr, $body:ident) => {{
        $body::<f32>($device);
        if $device.supports(Precision::F64) {
            $body::<f64>($device);
        } else {
            println!("  f64: not supported by this device, not run");
        }
    }};
}

/// A uniform leaf-scaled coordinate in [−1, 1], on the grid 2⁻⁵² and, unless zero, at
/// least 2⁻²⁰ in magnitude: stored values of §3.13 lie in G₅₃.
fn coordinate<T: Real>(rng: &mut Rng) -> T {
    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
    let u = (u * 2f64.powi(52)).round() / 2f64.powi(52);
    t(if u.abs() < 2f64.powi(-20) { 0.0 } else { u })
}

/// A uniform charge in [−1, 1).
fn charge<T: Real>(rng: &mut Rng) -> T {
    t((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
}

/// `n` random points of a leaf.
fn points<T: Real>(rng: &mut Rng, n: usize) -> Vec<[T; 3]> {
    (0..n)
        .map(|_| [0; 3].map(|_| coordinate::<T>(rng)))
        .collect()
}

/// Whether the closed boxes (level, index) a and b share a point, in cells of level 16.
fn touches(a: (u32, [u32; 3]), b: (u32, [u32; 3])) -> bool {
    let bounds = |key: (u32, [u32; 3])| {
        let size = 1u64 << (16 - key.0);
        key.1
            .map(|i| [u64::from(i) * size, (u64::from(i) + 1) * size])
    };
    let (a, b) = (bounds(a), bounds(b));
    (0..3).all(|k| a[k][0] <= b[k][1] && b[k][0] <= a[k][1])
}

/// The near sources of the interior box `target` that the tests use, as
/// `LaplaceOperator`'s near rows hold them: the same-level neighbours, the touching boxes
/// one level coarser and, below level 16, one level finer (more than 2:1 balance would
/// allow at once, to test every frame).
fn near_keys(target: (u32, [u32; 3])) -> Vec<(u32, [u32; 3])> {
    let (level, index) = target;
    let offsets = || {
        (0..27)
            .map(|d| [d / 9, (d / 3) % 3, d % 3].map(|o| o as i64 - 1))
            .filter(|d| *d != [0, 0, 0])
    };
    let shift = |i: [u32; 3], d: [i64; 3], level: u32| -> Option<[u32; 3]> {
        let n = 1i64 << level;
        let j: [i64; 3] = std::array::from_fn(|k| i64::from(i[k]) + d[k]);
        j.iter()
            .all(|&c| (0..n).contains(&c))
            .then(|| j.map(|c| c as u32))
    };
    let same: Vec<(u32, [u32; 3])> = offsets()
        .filter_map(|d| shift(index, d, level).map(|i| (level, i)))
        .collect();
    let mut keys = same.clone();
    if level > 0 {
        let parent = index.map(|i| i / 2);
        keys.extend(
            offsets()
                .filter_map(|d| shift(parent, d, level - 1).map(|i| (level - 1, i)))
                .filter(|&k| touches(k, target)),
        );
    }
    if level < 16 {
        for &(_, i) in &same {
            for c in 0..8u32 {
                let child = (
                    level + 1,
                    [0, 1, 2].map(|k| 2 * i[k] + ((c >> (2 - k)) & 1)),
                );
                if touches(child, target) {
                    keys.push(child);
                }
            }
        }
    }
    keys
}

/// An interior index on `level` (every neighbour and the parent's neighbours exist).
fn interior(rng: &mut Rng, level: u32) -> [u32; 3] {
    let n = 1usize << level;
    [0; 3].map(|_| (2 + rng.below(n.saturating_sub(4).max(1))).min(n - 1) as u32)
}

/// FMM-shaped rows: per entry of `targets` a target leaf on a level of 2, 9 and 16 with
/// that many targets and its near sources (`near_keys`) with random counts, at most
/// `max_sources` sources per row, in ascending leaf order (the leaf itself first).
fn fmm_rows<T: Real>(rng: &mut Rng, targets: &[usize], max_sources: usize) -> Problem<T> {
    let levels = [2u32, 9, 16];
    let mut leaves: Vec<Leaf<T>> = targets
        .iter()
        .enumerate()
        .map(|(r, &n_t)| {
            let level = levels[r % 3];
            let mut leaf = Leaf::new((level, interior(rng, level)));
            leaf.targets = points(rng, n_t);
            leaf
        })
        .collect();
    let mut rows = Vec::new();
    for r in 0..targets.len() {
        let keys = near_keys(leaves[r].key);
        let per_leaf = max_sources / (keys.len() + 1);
        let mut budget = max_sources;
        let mut row = vec![r as u32];
        // The leaf's own sources: its targets first (coincident pairs), then others.
        let own = rng.below(2 * per_leaf + 1).min(budget);
        budget -= own;
        let mut own_sources = leaves[r].targets.clone();
        own_sources.truncate(own);
        own_sources.extend(points::<T>(rng, own - own_sources.len()));
        leaves[r].charges = (0..own).map(|_| charge::<T>(rng)).collect();
        leaves[r].sources = own_sources;
        for key in keys {
            let n = rng.below(2 * per_leaf + 1).min(budget);
            budget -= n;
            let mut leaf = Leaf::new(key);
            leaf.sources = points(rng, n);
            leaf.charges = (0..n).map(|_| charge::<T>(rng)).collect();
            row.push(leaves.len() as u32);
            leaves.push(leaf);
        }
        rows.push(row);
    }
    Problem {
        leaves,
        first: 0,
        rows,
    }
}

/// The leaf sizes of the sums: 0, 1, around one tile, two and three tiles + 1, and two
/// sizes that span several target blocks of every layout.
fn sizes(tile: usize) -> Vec<usize> {
    let mut sizes = vec![
        0,
        1,
        tile.saturating_sub(1),
        tile,
        tile + 1,
        2 * tile,
        3 * tile + 1,
    ];
    sizes.extend([17, 37]);
    sizes.dedup();
    sizes
}

/// Pair terms: one source and one target per row over seeded separations spanning the
/// kernel domain 2⁻¹⁰⁸ ≤ r² ≤ 2⁷, within 8 u_T (φ) and 16 u_T (∇φ) of `nd_fmm_ref::p2p`.
fn pair_terms(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_0001);
        let flushing = device.backend() == BackendKind::Metal;
        let count = 32_768;
        let gradient_floor = if T::FLOAT == Precision::F32 {
            2f64.powi(-84)
        } else {
            0.0
        };
        let mut leaves = Vec::with_capacity(count);
        while leaves.len() < count {
            let scale = 2f64.powf(-(rng.below(40) as f64));
            let uniform = |rng: &mut Rng| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let y: [T; 3] = [0; 3].map(|_| t(scale * (2.0 * uniform(&mut rng) - 1.0)));
            let mut dir: [f64; 3] = [0; 3].map(|_| 2.0 * uniform(&mut rng) - 1.0);
            let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
            let len = 2f64.powf(-54.0 + 57.5 * uniform(&mut rng));
            dir = dir.map(|v| v / norm * len);
            let x: [T; 3] = std::array::from_fn(|k| t(w(y[k]) + dir[k]));
            let r2: f64 = (0..3).map(|k| (w(x[k]) - w(y[k])).powi(2)).sum();
            if !(2f64.powi(-108)..=2f64.powi(7)).contains(&r2) {
                continue;
            }
            // Charges: every 64th zero; on a flushing backend none below 2⁻¹⁰⁰.
            let q: T = if leaves.len() % 64 == 0 {
                t(0.0)
            } else {
                charge(&mut rng)
            };
            if flushing && w(q) != 0.0 && w(q).abs() < 2f64.powi(-100) {
                continue;
            }
            let mut leaf = Leaf::new((0, [0; 3]));
            leaf.sources = vec![y];
            leaf.charges = vec![q];
            leaf.targets = vec![x];
            leaves.push(leaf);
        }
        let problem = Problem {
            rows: (0..count).map(|r| vec![r as u32]).collect(),
            leaves,
            first: 0,
        };
        for layout in layouts(device) {
            let output = run(device, &problem, layout, true);
            let (mut ep, mut eg) = (0.0f64, 0.0f64);
            for (r, leaf) in problem.leaves.iter().enumerate() {
                let (x, y, q) = (leaf.targets[0], leaf.sources[0], leaf.charges[0]);
                let (mut rphi, mut rg) = ([t::<T>(0.0)], [[t::<T>(0.0); 3]]);
                nd_fmm_ref::p2p::p2p(&[y], &[q], &[x], &mut rphi, Some(&mut rg));
                let (phi, grad) = split(&output[problem.output_range(r, true)], true);
                let (phi, grad) = (phi[0], grad[0]);
                let e = if w(rphi[0]) == 0.0 {
                    if w(phi) == 0.0 { 0.0 } else { f64::INFINITY }
                } else {
                    ((w(phi) - w(rphi[0])) / w(rphi[0])).abs() / T::U
                };
                ep = ep.max(if e.is_nan() { f64::INFINITY } else { e });
                let r2: f64 = (0..3).map(|k| (w(x[k]) - w(y[k])).powi(2)).sum();
                if r2 >= gradient_floor && w(q) != 0.0 {
                    let scale = w(q).abs() / r2;
                    for k in 0..3 {
                        let e = (w(grad[k]) - w(rg[0][k])).abs() / scale / T::U;
                        eg = eg.max(if e.is_nan() { f64::INFINITY } else { e });
                    }
                } else if w(q) == 0.0 {
                    assert!(
                        grad.iter().all(|&g| w(g) == 0.0) && w(phi) == 0.0,
                        "{layout}: q = 0 gives a nonzero term"
                    );
                }
            }
            println!(
                "  {layout}, {}: {count} pairs, φ {ep:.2} u_T, ∇φ {eg:.2} u_T (bounds 8, 16)",
                T::FLOAT
            );
            assert!(ep <= 8.0, "{layout}: potential term {ep} u_T");
            assert!(eg <= 16.0, "{layout}: gradient term {eg} u_T");
        }
    }
    each_precision!(device, body);
}

/// Sums: FMM-shaped rows (a target leaf, its neighbours on its own, the coarser and the
/// finer level, mapped as `LaplaceOperator` maps them), n_t from 0 to 3 tiles + 1, up to
/// 4,096 sources per row, and a cancelling row, against `direct_sum`.
fn sums(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        for layout in layouts(device) {
            let tile = layout.tile(device.info(), T::FLOAT);
            let mut rng = Rng::new(0x7e57_0002 ^ tile as u64);
            let mut problem = fmm_rows::<T>(&mut rng, &sizes(tile), 4096);
            // A cancelling row: dipoles, pairs of opposite charges 2⁻¹⁰ apart, around a
            // leaf of targets, so that φ is far below Σ|q|/r.
            let first_extra = problem.leaves.len();
            let mut target = Leaf::new((9, [100, 101, 102]));
            target.targets = points(&mut rng, 24);
            let mut dipoles = Leaf::new((9, [101, 101, 102]));
            for _ in 0..200 {
                let y: [T; 3] = [0; 3].map(|_| coordinate::<T>(&mut rng));
                let mut z = y;
                z[0] = t(w(y[0]) - 2f64.powi(-10).copysign(w(y[0])));
                let q = charge::<T>(&mut rng);
                dipoles.sources.extend([y, z]);
                dipoles.charges.extend([q, t::<T>(-w(q))]);
            }
            // The cancelling row needs its target leaf among the rows: rebuild with it
            // first.
            let mut leaves = vec![target];
            leaves.append(&mut problem.leaves);
            leaves.push(dipoles);
            let shifted: Vec<Vec<u32>> = problem
                .rows
                .iter()
                .map(|row| row.iter().map(|&j| j + 1).collect())
                .collect();
            let mut rows = vec![vec![(first_extra + 1) as u32]];
            rows.extend(shifted);
            let problem = Problem {
                leaves,
                first: 0,
                rows,
            };
            let max_sources = (0..problem.rows.len())
                .map(|r| problem.row_sources(r).0.len())
                .max()
                .unwrap();
            for gradients in [false, true] {
                let output = run(device, &problem, layout, gradients);
                let worst = check_sums(&format!("{layout}"), &problem, &output, gradients);
                worst.print(&format!(
                    "{layout}, {}, gradients {gradients}, up to {max_sources} sources",
                    T::FLOAT
                ));
            }
        }
    }
    each_precision!(device, body);
}

/// Coincident points: targets equal to the sources of their leaf, duplicated sources,
/// mapped neighbour sources that equal or round onto a target, and adversarial points
/// at the leaf centres and faces on levels 2, 9 and 16. Every output is finite and
/// equals the sum over the non-coincident pairs within the sum contract.
fn coincident(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_0003);
        // Below half a spacing of the sums it is added to (ĉ of magnitude ½ to 1).
        let tiny = if T::FLOAT == Precision::F32 {
            2f64.powi(-30)
        } else {
            2f64.powi(-60)
        };
        // The source u_s of leaf `source` whose mapped ŷ = fl(ĉ + r̂ u_s) seen from
        // `target` is x: exact where (x − ĉ) / r̂ is not 0, else off by `tiny`, which the
        // rounding of ĉ + r̂ u_s removes.
        let onto = |x: [f64; 3], source: (u32, [u32; 3]), target: (u32, [u32; 3])| -> [T; 3] {
            let (c, ratio) = frame(source, target);
            std::array::from_fn(|k| {
                let exact = (x[k] - c[k]) / ratio;
                t(if exact == 0.0 && c[k].abs() >= 0.5 {
                    tiny
                } else {
                    exact
                })
            })
        };
        let special: Vec<T> = [
            0.0,
            -2f64.powi(-53),
            2f64.powi(-53),
            1.0,
            -1.0,
            1.0 - 2f64.powi(-24),
            -1.0 + 2f64.powi(-24),
            0.5,
            -0.5,
        ]
        .iter()
        .map(|&v| t::<T>(v))
        .collect();
        let grid: Vec<[T; 3]> = (0..27)
            .map(|i| {
                [
                    special[i % 9],
                    special[(i / 3) % 9],
                    special[(5 * i + 1) % 9],
                ]
            })
            .collect();
        // The target leaves: the constructed one, then one per adversarial level.
        let target_key = (9u32, [101u32, 100, 100]);
        let adversarial: Vec<(u32, [u32; 3])> = [2u32, 9, 16]
            .iter()
            .map(|&level| (level, interior(&mut rng, level)))
            .collect();
        let mut leaves = vec![Leaf::new(target_key)];
        leaves.extend(adversarial.iter().map(|&key| Leaf::new(key)));
        // Row 0: targets equal to the sources, every source three times; a same-level
        // neighbour (ĉ = (2, 0, 0)) whose sources map exactly onto the targets; a coarser
        // one (ĉ = (3, 1, 1), r̂ = 2) and a finer one (ĉ = (−1.5, −0.5, 0.5), r̂ = ½) with
        // a source each that rounds onto a target.
        let corner = [1.0, 1.0, 1.0];
        let face = [-1.0, -0.5, 0.5];
        // Targets on the grid 2⁻¹², so that x − ĉ is exact for every frame here.
        let mut targets: Vec<[T; 3]> = points::<T>(&mut rng, 13)
            .iter()
            .map(|p| p.map(|v| t::<T>((w(v) * 4096.0).round() / 4096.0)))
            .collect();
        targets.push(corner.map(t::<T>));
        targets.push(face.map(t::<T>));
        leaves[0].targets = targets.clone();
        for &x in &targets {
            for _ in 0..3 {
                leaves[0].sources.push(x);
                leaves[0].charges.push(charge(&mut rng));
            }
        }
        let same_key = (9u32, [102u32, 100, 100]);
        let coarse_key = (8u32, [51u32, 50, 50]);
        let fine_key = (10u32, [201u32, 200, 201]);
        for key in [same_key, coarse_key, fine_key] {
            assert!(touches(key, target_key), "{key:?}");
        }
        let mut same = Leaf::new(same_key);
        for x in &targets {
            same.sources.push(onto(x.map(w::<T>), same_key, target_key));
        }
        let mut coarse = Leaf::new(coarse_key);
        coarse.sources.push(onto(corner, coarse_key, target_key));
        let mut fine = Leaf::new(fine_key);
        fine.sources.push(onto(face, fine_key, target_key));
        let mut row0 = vec![0u32];
        for mut leaf in [same, coarse, fine] {
            leaf.sources.extend(points::<T>(&mut rng, 6));
            leaf.charges = leaf.sources.iter().map(|_| charge::<T>(&mut rng)).collect();
            row0.push(leaves.len() as u32);
            leaves.push(leaf);
        }
        let mut rows = vec![row0];
        // Adversarial rows: the special points as targets and sources of the leaf, and
        // in every near leaf the points that map onto them and the special points
        // themselves.
        for (a, &key) in adversarial.iter().enumerate() {
            let r = a + 1;
            leaves[r].targets = grid.clone();
            leaves[r].sources = grid.clone();
            leaves[r].charges = grid.iter().map(|_| charge::<T>(&mut rng)).collect();
            let mut row = vec![r as u32];
            for near in near_keys(key) {
                let mut source = Leaf::new(near);
                for p in &grid {
                    source.sources.push(onto(p.map(w::<T>), near, key));
                }
                source.sources.extend(grid.iter().copied());
                source.charges = source
                    .sources
                    .iter()
                    .map(|_| charge::<T>(&mut rng))
                    .collect();
                row.push(leaves.len() as u32);
                leaves.push(source);
            }
            rows.push(row);
        }
        let problem = Problem {
            leaves,
            first: 0,
            rows,
        };
        // The constructed coincidences happen on the host: 3 per target in the leaf, one
        // per target from the same-level neighbour, and the coarser and finer ones.
        let (y, _) = problem.row_sources(0);
        let hits = y
            .iter()
            .filter(|p| targets.iter().any(|x| x.map(w::<T>) == p.map(w::<T>)))
            .count();
        assert_eq!(hits, 4 * targets.len() + 2, "coincidences constructed");
        let coincident_pairs: usize = (0..problem.rows.len())
            .map(|r| {
                let (y, _) = problem.row_sources(r);
                let x = &problem.leaves[r].targets;
                x.iter()
                    .map(|x| y.iter().filter(|p| p.map(w::<T>) == x.map(w::<T>)).count())
                    .sum::<usize>()
            })
            .sum();
        for layout in layouts(device) {
            for gradients in [false, true] {
                // `check_sums` asserts that φ, and ∇φ within its range, are finite.
                let output = run(device, &problem, layout, gradients);
                let worst = check_sums(&format!("{layout}"), &problem, &output, gradients);
                worst.print(&format!(
                    "{layout}, {}, gradients {gradients}, {coincident_pairs} coincident pairs",
                    T::FLOAT
                ));
            }
        }
    }
    each_precision!(device, body);
}

/// Frames: the device frames of every entry (`near_frames`) equal the host's exact
/// (ĉ, r̂) bit for bit, for random pairs over every combination of levels 0–16, the
/// largest numerator (boxes in opposite corners of level 16) and the leaf itself.
fn frames(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_0004);
        let mut keys: Vec<(u32, [u32; 3])> = Vec::new();
        let mut rows: Vec<Vec<u32>> = Vec::new();
        let key = |rng: &mut Rng, level: u32| -> (u32, [u32; 3]) {
            (level, [0; 3].map(|_| rng.below(1 << level) as u32))
        };
        for lt in 0..=16u32 {
            keys.push(key(&mut rng, lt));
        }
        let targets = keys.len();
        for (r, row_level) in (0..=16u32).enumerate() {
            let mut row = vec![r as u32];
            for ls in 0..=16u32 {
                for _ in 0..3 {
                    row.push(keys.len() as u32);
                    keys.push(key(&mut rng, ls));
                }
            }
            if row_level == 16 {
                keys[r] = (16, [0, 0, 0]);
                row.push(keys.len() as u32);
                keys.push((16, [65_535; 3]));
            }
            rows.push(row);
        }
        let row_offsets: Vec<u32> = std::iter::once(0)
            .chain(rows.iter().scan(0u32, |total, row| {
                *total += row.len() as u32;
                Some(*total)
            }))
            .collect();
        let entries: Vec<u32> = rows.iter().flatten().copied().collect();
        let near = IndexView::upload(device, &row_offsets, &entries, keys.len()).unwrap();
        let leaves = LeafCoordinates::upload(device, &keys).unwrap();
        let mut out = device.alloc::<T>(4 * entries.len()).unwrap();
        near_frames(device, &near, 0, &leaves, out.as_slice_mut()).unwrap();
        let mut got = vec![t::<T>(0.0); out.len()];
        device.download(out.as_slice(), &mut got).unwrap();
        let mut largest = 0.0f64;
        for (r, row) in rows.iter().enumerate() {
            for (e, &j) in (row_offsets[r] as usize..).zip(row) {
                let want: [T; 4] = if j as usize == r {
                    [0.0, 0.0, 0.0, 1.0].map(t::<T>)
                } else {
                    let (c, ratio) = frame(keys[j as usize], keys[r]);
                    largest = largest.max(c.iter().fold(0.0, |m: f64, v| m.max(v.abs())));
                    [c[0], c[1], c[2], ratio].map(t::<T>)
                };
                for k in 0..4 {
                    assert_eq!(
                        got[4 * e + k].bits(),
                        want[k].bits(),
                        "row {r} entry {e}: component {k}: {:?} against {:?}",
                        got[4 * e + k],
                        want[k]
                    );
                }
            }
        }
        assert_eq!(largest, 131_070.0, "the largest numerator");
        println!(
            "  {}: {} frames over {targets} target levels bit for bit, largest |ĉ| {largest}",
            T::FLOAT,
            entries.len()
        );
    }
    each_precision!(device, body);
}

/// Domain ends: pairs at r² = 2⁻¹⁰⁸ and 2⁻⁸⁴ and near 2⁷, finite and within the term
/// bounds; in f32 the gradient only from 2⁻⁸⁴ (§3.13, "Range of the terms in f32").
fn domain_ends(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        // (target, source, charge): r² = 2⁻¹⁰⁸, 3 · 2⁻¹⁰⁸ (two components), 2⁻⁸⁴, and
        // 6.5² · 3 = 126.75 < 2⁷.
        let cases: Vec<([f64; 3], [f64; 3], f64)> = vec![
            ([2f64.powi(-54), 0.0, 0.0], [0.0; 3], 0.75),
            ([-2f64.powi(-53), 0.5, -0.25], [0.0, 0.5, -0.25], -1.0),
            ([2f64.powi(-42), 0.0, 0.0], [0.0; 3], 1.0),
            ([1.0, 1.0, -1.0], [-5.5, -5.5, 5.5], 0.5),
            ([-1.0, 1.0, 1.0], [5.25, -5.0, -4.75], -0.125),
        ];
        let mut leaves = Vec::new();
        for &(x, y, q) in &cases {
            let mut leaf = Leaf::new((3, [2, 2, 2]));
            leaf.targets = vec![x.map(t::<T>)];
            leaf.sources = vec![y.map(t::<T>)];
            leaf.charges = vec![t(q)];
            leaves.push(leaf);
        }
        let problem = Problem {
            rows: (0..cases.len()).map(|r| vec![r as u32]).collect(),
            leaves,
            first: 0,
        };
        for layout in layouts(device) {
            let output = run(device, &problem, layout, true);
            for (r, &(x, y, q)) in cases.iter().enumerate() {
                let r2: f64 = (0..3).map(|k| (x[k] - y[k]).powi(2)).sum();
                let (mut rphi, mut rg) = ([t::<T>(0.0)], [[t::<T>(0.0); 3]]);
                let leaf = &problem.leaves[r];
                nd_fmm_ref::p2p::p2p(
                    &leaf.sources,
                    &leaf.charges,
                    &leaf.targets,
                    &mut rphi,
                    Some(&mut rg),
                );
                let (phi, grad) = split(&output[problem.output_range(r, true)], true);
                let phi = w(phi[0]);
                assert!(phi.is_finite(), "{layout}: r² = {r2:e}: φ = {phi}");
                let ep = ((phi - w(rphi[0])) / w(rphi[0])).abs() / T::U;
                assert!(ep <= 8.0, "{layout}: r² = {r2:e}: φ {ep} u_T");
                let gradient_in_range = T::FLOAT == Precision::F64 || r2 >= 2f64.powi(-84);
                if gradient_in_range {
                    for k in 0..3 {
                        let g = w(grad[0][k]);
                        assert!(g.is_finite(), "{layout}: r² = {r2:e}: ∇φ = {g}");
                        let eg = (g - w(rg[0][k])).abs() / (q.abs() / r2) / T::U;
                        assert!(eg <= 16.0, "{layout}: r² = {r2:e}: ∇φ {eg} u_T");
                    }
                }
            }
            println!(
                "  {layout}, {}: r² from 2⁻¹⁰⁸ to 126.75 finite and within the term bounds",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

/// Accumulation onto a nonzero output, and calls that add nothing: rows without
/// entries, leaves without points and a level without rows leave every bit as it was
/// (the last launches nothing).
fn accumulation_and_empty_calls(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_0005);
        let mut problem = fmm_rows::<T>(&mut rng, &[0, 5, 9, 16], 600);
        // Row 0 has no targets; row 1 gets no entries and row 2 one leaf without points.
        problem.rows[1].clear();
        let empty_source = problem.leaves.len() as u32;
        problem.leaves.push(Leaf::new((9, [7, 7, 7])));
        problem.rows[2] = vec![empty_source];
        for layout in layouts(device) {
            for gradients in [false, true] {
                let initial = problem.output_store(gradients, |_, _| T::random_normal(&mut rng));
                let mut on = upload(device, &problem, &initial);
                launch(device, &mut on, layout, gradients);
                let output = download(device, &on);
                for j in [1, 2] {
                    let range = problem.output_range(j, gradients);
                    assert_same_bits(
                        &format!("{layout}: leaf {j}"),
                        &output[range.clone()],
                        &initial[range],
                    );
                }
                let worst = check_sums_from(
                    &format!("{layout}"),
                    &problem,
                    Some(&initial),
                    &output,
                    gradients,
                );
                worst.print(&format!(
                    "{layout}, {}, gradients {gradients}, onto nonzero output",
                    T::FLOAT
                ));
                // A level without rows launches nothing and changes nothing.
                let none = Problem {
                    rows: Vec::new(),
                    ..problem.clone()
                };
                let mut on = upload(device, &none, &initial);
                let before = device.counters().launches;
                launch(device, &mut on, layout, gradients);
                assert_eq!(device.counters().launches, before, "{layout}: empty level");
                assert_same_bits(
                    &format!("{layout}: empty level"),
                    &download(device, &on),
                    &initial,
                );
            }
            println!(
                "  {layout}, {}: empty rows, leaves without points and an empty level add \
                 nothing",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

/// Invariance and determinism, bit for bit: a near row split into one call per leaf
/// (entry k of every row in call k) gives the bits of one call; permuting the target
/// leaves of the level changes no leaf's bits; repeated launches give the same bits.
fn invariance_and_determinism(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_0006);
        let problem = fmm_rows::<T>(&mut rng, &[3, 11, 0, 20, 8, 1], 1500);
        for layout in layouts(device) {
            for gradients in [false, true] {
                let whole = run(device, &problem, layout, gradients);
                assert_same_bits(
                    &format!("{layout}: repeated launch"),
                    &run(device, &problem, layout, gradients),
                    &whole,
                );
                // Split at every leaf boundary.
                let zeros = problem.output_store(gradients, |_, _| t::<T>(0.0));
                let longest = problem.rows.iter().map(Vec::len).max().unwrap();
                let mut on = upload(device, &problem, &zeros);
                for k in 0..longest {
                    let rows: Vec<Vec<u32>> = problem
                        .rows
                        .iter()
                        .map(|row| row.get(k).map(|&j| vec![j]).unwrap_or_default())
                        .collect();
                    let mut part = upload_rows(device, &problem, &rows, &[]);
                    // Reuse the accumulated output: swap the buffers in.
                    std::mem::swap(&mut part.output, &mut on.output);
                    launch(device, &mut part, layout, gradients);
                    std::mem::swap(&mut part.output, &mut on.output);
                }
                assert_same_bits(
                    &format!("{layout}: split at every leaf boundary"),
                    &download(device, &on),
                    &whole,
                );
                // Permute the target leaves: new row i is old row π(i).
                let nrows = problem.rows.len();
                let perm = rng.permutation(nrows);
                let mut index = vec![0u32; problem.leaves.len()];
                for (j, slot) in index.iter_mut().enumerate() {
                    *slot = j as u32;
                }
                for (i, &old) in perm.iter().enumerate() {
                    index[old as usize] = i as u32;
                }
                let mut leaves = problem.leaves.clone();
                for (i, &old) in perm.iter().enumerate() {
                    leaves[i] = problem.leaves[old as usize].clone();
                }
                let rows: Vec<Vec<u32>> = perm
                    .iter()
                    .map(|&old| {
                        problem.rows[old as usize]
                            .iter()
                            .map(|&j| index[j as usize])
                            .collect()
                    })
                    .collect();
                let permuted = Problem {
                    leaves,
                    first: 0,
                    rows,
                };
                let output = run(device, &permuted, layout, gradients);
                for (i, &old) in perm.iter().enumerate() {
                    assert_same_bits(
                        &format!("{layout}: target leaf {old} at position {i}"),
                        &output[permuted.output_range(i, gradients)],
                        &whole[problem.output_range(old as usize, gradients)],
                    );
                }
            }
            println!(
                "  {layout}, {}: split per leaf, permuted targets and repeated launches bit \
                 for bit",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

/// On the CPU runtime every operation of the formulation is correctly rounded or an
/// explicit fma, so the three layouts give the same bits; reported, not part of the
/// contract (§3.13 allows contraction to differ).
fn layouts_agree(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_0007);
        let problem = fmm_rows::<T>(&mut rng, &[7, 19, 33], 2000);
        for gradients in [false, true] {
            let outputs: Vec<(P2pLayout, Vec<T>)> = layouts(device)
                .into_iter()
                .map(|layout| (layout, run(device, &problem, layout, gradients)))
                .collect();
            for (layout, output) in &outputs[1..] {
                let same = output
                    .iter()
                    .zip(&outputs[0].1)
                    .filter(|(a, b)| a.bits() == b.bits())
                    .count();
                println!(
                    "  {}, gradients {gradients}: {layout} equals {} in {same} of {} values",
                    T::FLOAT,
                    outputs[0].0,
                    output.len()
                );
            }
        }
    }
    each_precision!(device, body);
}

tests_on!(
    cpu: pair_terms,
    sums,
    coincident,
    frames,
    domain_ends,
    accumulation_and_empty_calls,
    invariance_and_determinism,
    layouts_agree,
);
tests_on!(
    metal: pair_terms,
    sums,
    coincident,
    frames,
    domain_ends,
    accumulation_and_empty_calls,
    invariance_and_determinism,
    layouts_agree,
);
