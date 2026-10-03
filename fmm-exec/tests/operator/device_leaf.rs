//! The device leaf operators P2M, L2P, P2L and M2P (Phase 4 T7, C4.3) as the device
//! operator runs them, against `nd_fmm_ref::leaf` at the absolute frames of CONVENTIONS
//! §3.12, as T8 of Phase 3 checks the host operators (`leaf.rs`), and against the host
//! operator (`LaplaceOperator`'s `p2m_leaf`, `l2p_leaf`, `p2l_pair`, `m2p_pair`) in the
//! same precision.
//!
//! Geometry, in the dyadic domain, points on the grid of `common` (so that every side sees
//! the same geometry exactly):
//! - P2M and L2P: four random leaves on each of levels 2, 9 and 16, with 6, 0, 1 and 40
//!   points (the first with the leaf centre, two face points and a corner);
//! - P2L, X-list geometry: two random target boxes on levels 2, 9 and 16 and every source
//!   leaf on the level of the parent that touches the parent but not the target (the
//!   level difference the X list allows), with 6, 1, 0 and 40 points in turn;
//! - M2P, W-list geometry: two random target leaves on levels 2, 9 and 15, and every child
//!   of a neighbour that does not touch the target (on the level below, the difference
//!   the W list allows), random multipoles;
//! - every degree of [`degrees`] (f64: 0, 1, 4, 8, 12, 20; f32: 0, 1, 4, 8), L2P and M2P
//!   with gradients and without; one launch per operator, level and degree, the rows
//!   all targets of the level as the device operator's views hold them.
//!
//! Error measures (`common`): coefficients per degree in the §3.8 weighting relative to
//! their terms (P2M, P2L); values relative to the magnitudes of their terms, the gradient
//! as a Euclidean norm (L2P, M2P), in the leaf units of §3.13. Tolerance 1e-13 in f64
//! (C3.1) and 1e-5 in f32 against the f64 reference (docs/phase4/README.md, "Accuracy
//! measures"), the inputs rounded to T first.
//!
//! Against the host operator (T7 brief): per operator, precision, level and quantity
//! (coefficients, φ, ∇φ), the device's worst error against the reference is compared with
//! the host operator's worst error against the same reference.
//! - In f32 the host operator's error is its own rounding (f32 against the f64
//!   reference), and the device's must lie within twice it: asserted.
//! - In f64 on this exact geometry the host operator does the reference's arithmetic on
//!   the same scaled points, so its error is zero (P2M, P2L: bit for bit) or only the
//!   rounding of the reference's own unit conversion, φ̂ = r_t φ with r_t = 3 · 2⁻ˡ⁻¹ (L2P,
//!   M2P, about 1e-16). The device, whose multiply–adds cubecl-opt fuses
//!   (spikes/device-arith/REPORT.md, rule 6), cannot be held to twice that; its ratio is
//!   printed, not asserted, and the 1e-13 bound applies. (Measured on the CPU runtime: L2P
//!   up to 2.4 times the host operator's 1e-16, all within 2.2e-16; M2P within 1.0.)
//!
//! Frames: the frames the kernels form for the X and W lists (`x_frames`, `w_frames`)
//! equal `geometry::relative_frame` bit for bit, in f32 and f64.
//!
//! The CPU runtime runs with the `cpu` feature (its default layout and the cube layout);
//! Metal (f32) with `metal`, ignored, by hand outside the macOS sandbox. Each test prints
//! the device, the worst errors per operator, precision and level (with the worst degree
//! for coefficients) and the backends it ran.

use std::collections::BTreeMap;

use nd_fmm_exec::geometry::relative_frame;
use nd_fmm_exec::operator::LaplaceOperator;
use nd_fmm_exec::tables::{M2lStrategy, Tables};
use nd_fmm_kernels::leaf::{
    LeafLayout, SourceInputs, TargetInputs, l2p, m2p, p2l, p2m, w_frames, x_frames,
};
use nd_fmm_kernels::view::{BoxCoordinates, IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::{Frame, Workspace, leaf};
use nd_octree::{MortonKey, morton};

use crate::common::{
    Basis, Kind, SplitMix64, absolute_frame, degree_error, dyadic_domain, evaluation_terms,
    grid_points, key_radius, len, neighbours, point_terms, random_coefficients, source_chunk,
    target_chunk, touches,
};

/// The precisions of these tests.
trait Real: DeviceFloat + nd_fmm_exec::operator::SimdScalar {}

impl Real for f32 {}

impl Real for f64 {}

/// `x` widened to f64.
fn w<T: Real>(x: T) -> f64 {
    RealScalar::to_f64(x)
}

/// `x` rounded to T.
fn t<T: Real>(x: f64) -> T {
    <T as RealScalar>::from_f64(x)
}

/// `x` rounded to T and widened again.
fn round<T: Real>(x: f64) -> f64 {
    w(t::<T>(x))
}

/// The degrees of the check: p ≤ 20 in f64, p ≤ 8 in f32.
fn degrees<T: Real>() -> &'static [usize] {
    if T::FLOAT == Precision::F64 {
        &[0, 1, 4, 8, 12, 20]
    } else {
        &[0, 1, 4, 8]
    }
}

/// The tolerance in T.
fn tolerance<T: Real>() -> f64 {
    if T::FLOAT == Precision::F64 {
        1e-13
    } else {
        1e-5
    }
}

/// A key's level and index as the device holds them.
fn key_index(key: MortonKey) -> (u32, [u32; 3]) {
    let (level, index) = morton::decode(key);
    (level as u32, index.map(|c| c as u32))
}

/// The points of a leaf: absolute and leaf-scaled.
type Points = (Vec<[f64; 3]>, Vec<[f64; 3]>);

/// The points of the leaf `key` (absolute and leaf-scaled): `count` grid points, the
/// first replaced by the leaf centre, two face points and a corner if `special`.
fn leaf_points(key: MortonKey, count: usize, special: bool, rng: &mut SplitMix64) -> Points {
    let domain = dyadic_domain();
    let (mut x, mut u) = grid_points(key, &domain, count, rng);
    if special {
        let frame = absolute_frame(key, &domain);
        let fixed = [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, -0.25],
            [-0.75, -1.0, 0.125],
            [1.0, -1.0, 1.0],
        ];
        for (i, f) in fixed.iter().enumerate().take(count) {
            u[i] = *f;
            x[i] = std::array::from_fn(|k| frame.centre[k] + frame.radius * f[k]);
        }
    }
    (x, u)
}

/// The worst errors of one (operator, precision, level, quantity): the device's with its
/// degree, and the host operator's.
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    device: f64,
    degree: Option<usize>,
    host: f64,
}

impl Cell {
    fn add(&mut self, device: f64, degree: Option<usize>, host: f64) {
        if device >= self.device {
            self.device = device;
            self.degree = degree;
        }
        self.host = self.host.max(host);
    }
}

/// Every cell of a run, by (operator, quantity, level).
type Cells = BTreeMap<(&'static str, &'static str, usize), Cell>;

/// Records `device` (and its degree) and `host`, asserting the operator bound.
fn record<T: Real>(
    cells: &mut Cells,
    key: (&'static str, &'static str, usize),
    (device, degree): (f64, Option<usize>),
    host: f64,
    context: impl Fn() -> String,
) {
    assert!(
        device <= tolerance::<T>(),
        "{}, {}, {}: {} error {device:e} exceeds {:e} (host operator {host:e}); {}",
        key.0,
        T::FLOAT,
        key.2,
        key.1,
        tolerance::<T>(),
        context()
    );
    cells.entry(key).or_default().add(device, degree, host);
}

/// The host operator in T at degree p, potentials and gradients if `gradients`.
fn host_operator<T: Real>(p: usize, gradients: bool) -> LaplaceOperator<T> {
    // The leaf operators use no table; the rotation tables are the cheapest to build.
    LaplaceOperator::new(Tables::<T>::build(p, M2lStrategy::Rotation), gradients, 64)
}

/// Uploads the leaf coordinates, the point offsets of `counts` and the box coordinates of
/// `level` holding `boxes`.
fn geometry(
    device: &mut Device,
    leaves: &[MortonKey],
    counts: &[usize],
    (level, boxes): (usize, &[MortonKey]),
) -> (LeafCoordinates, PointOffsets, BoxCoordinates) {
    let keys: Vec<(u32, [u32; 3])> = leaves.iter().map(|&k| key_index(k)).collect();
    let offsets: Vec<u32> = std::iter::once(0)
        .chain(counts.iter().scan(0u32, |total, &n| {
            *total += n as u32;
            Some(*total)
        }))
        .collect();
    let levels: Vec<Vec<[u32; 3]>> = (0..=level)
        .map(|l| {
            if l == level {
                boxes.iter().map(|&k| key_index(k).1).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    (
        LeafCoordinates::upload(device, &keys).unwrap(),
        PointOffsets::upload(device, &offsets).unwrap(),
        BoxCoordinates::upload(device, &levels).unwrap(),
    )
}

/// A CSR view of `rows`.
fn view(device: &mut Device, rows: &[Vec<u32>], columns: usize) -> IndexView {
    let offsets: Vec<u32> = std::iter::once(0)
        .chain(rows.iter().scan(0u32, |total, row| {
            *total += row.len() as u32;
            Some(*total)
        }))
        .collect();
    let entries: Vec<u32> = rows.iter().flatten().copied().collect();
    IndexView::upload(device, &offsets, &entries, columns).unwrap()
}

/// A set of source leaves: keys, absolute and leaf-scaled points, charges rounded to T.
struct Sources {
    keys: Vec<MortonKey>,
    x: Vec<Vec<[f64; 3]>>,
    u: Vec<Vec<[f64; 3]>>,
    q: Vec<Vec<f64>>,
}

impl Sources {
    /// The source store in T and the counts.
    fn store<T: Real>(&self) -> (Vec<T>, Vec<usize>) {
        let mut store = Vec::new();
        for (u, q) in self.u.iter().zip(&self.q) {
            store.extend(u.iter().flatten().map(|&c| t::<T>(c)));
            store.extend(q.iter().map(|&c| t::<T>(c)));
        }
        (store, self.u.iter().map(Vec::len).collect())
    }
}

/// P2M (rows: boxes, entries: own leaf) or P2L (entries: X list) of one level on the
/// device: `rows[t]` lists the leaves of target box `targets[t]`.
fn run_sources<T: Real>(
    device: &mut Device,
    (layout, p, irregular): (LeafLayout, usize, bool),
    (level, targets): (usize, &[MortonKey]),
    sources: &Sources,
    rows: &[Vec<u32>],
) -> (Vec<T>, Vec<[T; 4]>) {
    let (store, counts) = sources.store::<T>();
    let (leaves, source_offsets, boxes) =
        geometry(device, &sources.keys, &counts, (level, targets));
    let view = view(device, rows, sources.keys.len());
    let store = device.upload(&store).unwrap();
    let n = len(p);
    let mut out = device.alloc::<T>(n * targets.len()).unwrap();
    let inputs = SourceInputs {
        view: &view,
        level,
        boxes: &boxes,
        leaves: &leaves,
        source_offsets: &source_offsets,
        sources: store.as_slice(),
    };
    if irregular {
        p2l(device, layout, p, &inputs, out.as_slice_mut()).unwrap();
    } else {
        p2m(device, layout, p, &inputs, out.as_slice_mut()).unwrap();
    }
    let mut got = vec![t::<T>(0.0); out.len()];
    device.download(out.as_slice(), &mut got).unwrap();
    let mut frames = device.alloc::<T>(4 * view.len()).unwrap();
    x_frames(device, &inputs, frames.as_slice_mut()).unwrap();
    let mut flat = vec![t::<T>(0.0); frames.len()];
    device.download(frames.as_slice(), &mut flat).unwrap();
    (got, flat.as_chunks::<4>().0.to_vec())
}

/// L2P (entries: own box on `level`) or M2P (entries: W list on `level + 1`) of one level
/// on the device: target leaves `targets` with leaf-scaled points `u`, the expansions of
/// `boxes`; returns the target output store and the W frames.
#[allow(clippy::too_many_arguments)]
fn run_targets<T: Real>(
    device: &mut Device,
    (layout, p, irregular, gradients): (LeafLayout, usize, bool, bool),
    level: usize,
    (targets, u): (&[MortonKey], &[Vec<[f64; 3]>]),
    (boxes, coefficients): (&[MortonKey], &[Vec<f64>]),
    rows: &[Vec<u32>],
) -> (Vec<T>, Vec<[T; 4]>) {
    let counts: Vec<usize> = u.iter().map(Vec::len).collect();
    let entry_level = level + usize::from(irregular);
    let (leaves, target_offsets, box_coordinates) =
        geometry(device, targets, &counts, (entry_level, boxes));
    let view = view(device, rows, boxes.len());
    let input: Vec<T> = u.iter().flatten().flatten().map(|&c| t::<T>(c)).collect();
    let input = device.upload(&input).unwrap();
    let flat: Vec<T> = coefficients.iter().flatten().map(|&c| t::<T>(c)).collect();
    let coefficients = device.upload(&flat).unwrap();
    let per_point = if gradients { 4 } else { 1 };
    let total: usize = counts.iter().sum();
    let mut output = device.alloc::<T>(per_point * total).unwrap();
    let inputs = TargetInputs {
        view: &view,
        level,
        first_leaf: 0,
        boxes: &box_coordinates,
        leaves: &leaves,
        target_offsets: &target_offsets,
        target_input: input.as_slice(),
    };
    if irregular {
        m2p(
            device,
            layout,
            p,
            gradients,
            &inputs,
            coefficients.as_slice(),
            output.as_slice_mut(),
        )
        .unwrap();
    } else {
        l2p(
            device,
            layout,
            p,
            gradients,
            &inputs,
            coefficients.as_slice(),
            output.as_slice_mut(),
        )
        .unwrap();
    }
    let mut got = vec![t::<T>(0.0); output.len()];
    device.download(output.as_slice(), &mut got).unwrap();
    let mut frames = device.alloc::<T>(4 * view.len()).unwrap();
    if irregular {
        w_frames(device, &inputs, frames.as_slice_mut()).unwrap();
    }
    let mut flat = vec![t::<T>(0.0); frames.len()];
    device.download(frames.as_slice(), &mut flat).unwrap();
    (got, flat.as_chunks::<4>().0.to_vec())
}

/// Asserts that a device frame equals `relative_frame::<T>(s, t)` bit for bit.
fn check_frame<T: Real>(got: [T; 4], s: MortonKey, t: MortonKey, what: &str) {
    let f = relative_frame::<T>(s, t);
    let want = [f.centre[0], f.centre[1], f.centre[2], f.radius];
    for k in 0..4 {
        assert_eq!(
            w(got[k]).to_bits(),
            w(want[k]).to_bits(),
            "{what}: frame of {s} from {t}, component {k}"
        );
    }
}

/// The values of a target output chunk of n targets, potentials and (with gradients)
/// gradients, widened.
fn split<T: Real>(chunk: &[T], n: usize, gradients: bool) -> (Vec<f64>, Vec<[f64; 3]>) {
    let phi = chunk[..n].iter().map(|&v| w(v)).collect();
    let grad = if gradients {
        chunk[n..]
            .as_chunks::<3>()
            .0
            .iter()
            .map(|g| g.map(w::<T>))
            .collect()
    } else {
        Vec::new()
    };
    (phi, grad)
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn relative(error: f64, terms: f64) -> f64 {
    if terms > 0.0 {
        error / terms
    } else {
        assert_eq!(error, 0.0, "nonzero error with zero terms");
        0.0
    }
}

/// Compares the target outputs `device` and `host` of target leaf `key` (points x, u)
/// with the reference φ, ∇φ (of 1/|x − y|, absolute units) converted with r_t, relative
/// to `terms` in leaf units; records the cells of `name` on `level`.
#[allow(clippy::too_many_arguments)]
fn compare_values<T: Real>(
    cells: &mut Cells,
    (name, level): (&'static str, usize),
    (device, host): (&[T], &[T]),
    (phi, grad): (&[f64], &[[f64; 3]]),
    r_t: f64,
    terms: &[(f64, f64)],
    gradients: bool,
    context: impl Fn() -> String,
) {
    let n = phi.len();
    let (dp, dg) = split(device, n, gradients);
    let (hp, hg) = split(host, n, gradients);
    for j in 0..n {
        let want = r_t * phi[j];
        let e = relative((dp[j] - want).abs(), terms[j].0);
        let h = relative((hp[j] - want).abs(), terms[j].0);
        record::<T>(cells, (name, "φ", level), (e, None), h, &context);
        if gradients {
            let want = grad[j].map(|v| r_t * r_t * v);
            let sub = |a: [f64; 3]| std::array::from_fn(|k| a[k] - want[k]);
            let e = relative(norm(sub(dg[j])), terms[j].1);
            let h = relative(norm(sub(hg[j])), terms[j].1);
            record::<T>(cells, (name, "∇φ", level), (e, None), h, &context);
        }
    }
}

/// P2M and L2P at four random leaves on each level.
fn p2m_and_l2p<T: Real>(device: &mut Device, layout: LeafLayout, cells: &mut Cells) {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7e70);
    for level in [2usize, 9, 16] {
        for &p in degrees::<T>() {
            let mut keys = Vec::new();
            while keys.len() < 4 {
                let key = rng.key(level);
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            let counts = [6, 0, 1, 40];
            let mut sources = Sources {
                keys: keys.clone(),
                x: Vec::new(),
                u: Vec::new(),
                q: Vec::new(),
            };
            for (j, &key) in keys.iter().enumerate() {
                let (x, u) = leaf_points(key, counts[j], j == 0, &mut rng);
                sources.q.push(
                    rng.charges(counts[j])
                        .iter()
                        .map(|&q| round::<T>(q))
                        .collect(),
                );
                sources.x.push(x);
                sources.u.push(u);
            }
            // P2M: row t is leaf t.
            let rows: Vec<Vec<u32>> = (0..4).map(|j| vec![j]).collect();
            let (got, _) =
                run_sources::<T>(device, (layout, p, false), (level, &keys), &sources, &rows);
            let mut host = host_operator::<T>(p, false);
            let mut ws = Workspace::new(p);
            let n = len(p);
            for (j, &key) in keys.iter().enumerate() {
                let frame = absolute_frame(key, &domain);
                let mut want = vec![0.0; n];
                leaf::p2m(p, &frame, &sources.x[j], &sources.q[j], &mut ws, &mut want);
                let chunk: Vec<T> = source_chunk(&sources.u[j], &sources.q[j])
                    .iter()
                    .map(|&c| t::<T>(c))
                    .collect();
                let mut in_t = vec![t::<T>(0.0); n];
                host.p2m_leaf(&chunk, &mut in_t);
                let scale = point_terms(Basis::Regular, p, &frame, &sources.x[j], &sources.q[j]);
                let widen = |v: &[T]| -> Vec<f64> { v.iter().map(|&c| w(c)).collect() };
                let device_error = degree_error_at(
                    Kind::Multipole,
                    p,
                    &widen(&got[j * n..(j + 1) * n]),
                    &want,
                    &scale,
                );
                let host_error =
                    degree_error_at(Kind::Multipole, p, &widen(&in_t), &want, &scale).0;
                record::<T>(
                    cells,
                    ("P2M", "coefficients", level),
                    (device_error.0, Some(device_error.1)),
                    host_error,
                    || format!("p = {p}, leaf {key}, {} points", counts[j]),
                );
            }
            // L2P: row r is the leaf's own box r; random locals.
            let targets: Vec<Points> = keys
                .iter()
                .enumerate()
                .map(|(j, &key)| leaf_points(key, counts[j], j == 0, &mut rng))
                .collect();
            let locals: Vec<Vec<f64>> = (0..4)
                .map(|_| {
                    random_coefficients(Kind::Local, p, &mut rng)
                        .iter()
                        .map(|&c| round::<T>(c))
                        .collect()
                })
                .collect();
            let u: Vec<Vec<[f64; 3]>> = targets.iter().map(|(_, u)| u.clone()).collect();
            for gradients in [false, true] {
                let (got, _) = run_targets::<T>(
                    device,
                    (layout, p, false, gradients),
                    level,
                    (&keys, &u),
                    (&keys, &locals),
                    &rows,
                );
                let mut host = host_operator::<T>(p, gradients);
                let per_point = if gradients { 4 } else { 1 };
                let mut start = 0;
                for (j, &key) in keys.iter().enumerate() {
                    let (x, u) = &targets[j];
                    let frame = absolute_frame(key, &domain);
                    let (mut phi, mut grad) = (vec![0.0; x.len()], vec![[0.0; 3]; x.len()]);
                    leaf::l2p(p, &frame, &locals[j], x, &mut ws, &mut phi, Some(&mut grad));
                    let local_t: Vec<T> = locals[j].iter().map(|&c| t::<T>(c)).collect();
                    let input: Vec<T> = target_chunk(u).iter().map(|&c| t::<T>(c)).collect();
                    let mut in_t = vec![t::<T>(0.0); per_point * x.len()];
                    host.l2p_leaf(&local_t, &input, &mut in_t);
                    let unit = Frame::new([0.0; 3], 1.0);
                    let terms: Vec<(f64, f64)> = u
                        .iter()
                        .map(|&u| evaluation_terms(Basis::Regular, p, &unit, &locals[j], u))
                        .collect();
                    let end = start + per_point * x.len();
                    compare_values::<T>(
                        cells,
                        ("L2P", level),
                        (&got[start..end], &in_t),
                        (&phi, &grad),
                        key_radius(key, &domain),
                        &terms,
                        gradients,
                        || format!("p = {p}, leaf {key}, gradients {gradients}"),
                    );
                    start = end;
                }
            }
        }
    }
}

/// `degree_error` with the degree of the worst error.
fn degree_error_at(kind: Kind, p: usize, got: &[f64], want: &[f64], terms: &[f64]) -> (f64, usize) {
    let mut worst = (0.0f64, 0usize);
    for n in 0..=p {
        let range = n * n..(n + 1) * (n + 1);
        let mut g = vec![0.0; len(p)];
        let mut wv = vec![0.0; len(p)];
        let mut tm = vec![0.0; len(p)];
        g[range.clone()].copy_from_slice(&got[range.clone()]);
        wv[range.clone()].copy_from_slice(&want[range.clone()]);
        tm[range.clone()].copy_from_slice(&terms[range]);
        let e = degree_error(kind, p, &g, &wv, &tm);
        if e > worst.0 {
            worst = (e, n);
        }
    }
    worst
}

/// The X-list sources of `target`: the leaves on its parent's level that touch the
/// parent but not `target`.
fn x_sources(target: MortonKey) -> Vec<MortonKey> {
    let parent = morton::parent(target).unwrap();
    neighbours(parent)
        .into_iter()
        .filter(|&s| !touches(s, target))
        .collect()
}

/// P2L at two target boxes per level with their X lists, one launch per level.
fn p2l_x_list<T: Real>(device: &mut Device, layout: LeafLayout, cells: &mut Cells) {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7e71);
    for level in [2usize, 9, 16] {
        for &p in degrees::<T>() {
            let mut targets = Vec::new();
            while targets.len() < 2 {
                let target = rng.key(level);
                if !x_sources(target).is_empty() && !targets.contains(&target) {
                    targets.push(target);
                }
            }
            let mut sources = Sources {
                keys: Vec::new(),
                x: Vec::new(),
                u: Vec::new(),
                q: Vec::new(),
            };
            let mut rows = Vec::new();
            for &target in &targets {
                let mut row = Vec::new();
                for (k, source) in x_sources(target).into_iter().enumerate() {
                    assert_eq!(morton::level(source) + 1, level);
                    let count = [6, 1, 0, 40][k % 4];
                    let (x, u) = leaf_points(source, count, k == 0, &mut rng);
                    row.push(sources.keys.len() as u32);
                    sources.keys.push(source);
                    sources
                        .q
                        .push(rng.charges(count).iter().map(|&q| round::<T>(q)).collect());
                    sources.x.push(x);
                    sources.u.push(u);
                }
                rows.push(row);
            }
            let (got, frames) = run_sources::<T>(
                device,
                (layout, p, true),
                (level, &targets),
                &sources,
                &rows,
            );
            let n = len(p);
            let mut ws = Workspace::new(p);
            let mut host = host_operator::<T>(p, false);
            let mut e = 0;
            for (r, &target) in targets.iter().enumerate() {
                let frame = absolute_frame(target, &domain);
                let mut want = vec![0.0; n];
                let mut in_t = vec![t::<T>(0.0); n];
                let mut scale = vec![0.0; n];
                for &j in &rows[r] {
                    let j = j as usize;
                    let source = sources.keys[j];
                    check_frame(frames[e], target, source, "P2L");
                    e += 1;
                    leaf::p2l(p, &frame, &sources.x[j], &sources.q[j], &mut ws, &mut want);
                    let chunk: Vec<T> = source_chunk(&sources.u[j], &sources.q[j])
                        .iter()
                        .map(|&c| t::<T>(c))
                        .collect();
                    host.p2l_pair(source, target, &chunk, &mut in_t);
                    let terms =
                        point_terms(Basis::Irregular, p, &frame, &sources.x[j], &sources.q[j]);
                    for (s, v) in scale.iter_mut().zip(terms) {
                        *s += v;
                    }
                }
                let widen = |v: &[T]| -> Vec<f64> { v.iter().map(|&c| w(c)).collect() };
                let device_error = degree_error_at(
                    Kind::Local,
                    p,
                    &widen(&got[r * n..(r + 1) * n]),
                    &want,
                    &scale,
                );
                let host_error = degree_error_at(Kind::Local, p, &widen(&in_t), &want, &scale).0;
                record::<T>(
                    cells,
                    ("P2L", "coefficients", level),
                    (device_error.0, Some(device_error.1)),
                    host_error,
                    || format!("p = {p}, target {target}, {} sources", rows[r].len()),
                );
            }
        }
    }
}

/// The W-list sources of `target`: the children of its neighbours that do not touch it.
fn w_sources(target: MortonKey) -> Vec<MortonKey> {
    neighbours(target)
        .into_iter()
        .flat_map(|n| morton::children(n).unwrap())
        .filter(|&s| !touches(s, target))
        .collect()
}

/// M2P at two target leaves per level with their W lists, one launch per level.
fn m2p_w_list<T: Real>(device: &mut Device, layout: LeafLayout, cells: &mut Cells) {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7e72);
    for level in [2usize, 9, 15] {
        for &p in degrees::<T>() {
            let mut targets = Vec::new();
            while targets.len() < 2 {
                let target = rng.key(level);
                if !targets.contains(&target) {
                    targets.push(target);
                }
            }
            let counts = [6, 40];
            let points: Vec<Points> = targets
                .iter()
                .enumerate()
                .map(|(r, &key)| leaf_points(key, counts[r], r == 0, &mut rng))
                .collect();
            let (mut boxes, mut multipoles, mut rows) = (Vec::new(), Vec::new(), Vec::new());
            for &target in &targets {
                let mut row = Vec::new();
                for source in w_sources(target) {
                    assert_eq!(morton::level(source), level + 1);
                    row.push(boxes.len() as u32);
                    boxes.push(source);
                    multipoles.push(
                        random_coefficients(Kind::Multipole, p, &mut rng)
                            .iter()
                            .map(|&c| round::<T>(c))
                            .collect::<Vec<f64>>(),
                    );
                }
                rows.push(row);
            }
            let u: Vec<Vec<[f64; 3]>> = points.iter().map(|(_, u)| u.clone()).collect();
            let mut ws = Workspace::new(p);
            for gradients in [false, true] {
                let (got, frames) = run_targets::<T>(
                    device,
                    (layout, p, true, gradients),
                    level,
                    (&targets, &u),
                    (&boxes, &multipoles),
                    &rows,
                );
                let mut host = host_operator::<T>(p, gradients);
                let per_point = if gradients { 4 } else { 1 };
                let (mut start, mut e) = (0, 0);
                for (r, &target) in targets.iter().enumerate() {
                    let (x, u) = &points[r];
                    let (mut phi, mut grad) = (vec![0.0; x.len()], vec![[0.0; 3]; x.len()]);
                    let input: Vec<T> = target_chunk(u).iter().map(|&c| t::<T>(c)).collect();
                    let mut in_t = vec![t::<T>(0.0); per_point * x.len()];
                    let mut terms = vec![(0.0, 0.0); x.len()];
                    for &s in &rows[r] {
                        let s = s as usize;
                        let source = boxes[s];
                        check_frame(frames[e], source, target, "M2P");
                        e += 1;
                        let frame = absolute_frame(source, &domain);
                        leaf::m2p(
                            p,
                            &frame,
                            &multipoles[s],
                            x,
                            &mut ws,
                            &mut phi,
                            Some(&mut grad),
                        );
                        let multipole: Vec<T> = multipoles[s].iter().map(|&c| t::<T>(c)).collect();
                        host.m2p_pair(source, target, &multipole, &input, &mut in_t);
                        let relative = relative_frame::<f64>(source, target);
                        for (term, &u) in terms.iter_mut().zip(u) {
                            let (tp, tg) =
                                evaluation_terms(Basis::Irregular, p, &relative, &multipoles[s], u);
                            term.0 += tp;
                            term.1 += tg;
                        }
                    }
                    let end = start + per_point * x.len();
                    compare_values::<T>(
                        cells,
                        ("M2P", level),
                        (&got[start..end], &in_t),
                        (&phi, &grad),
                        key_radius(target, &domain),
                        &terms,
                        gradients,
                        || format!("p = {p}, target {target}, gradients {gradients}"),
                    );
                    start = end;
                }
            }
        }
    }
}

/// Prints the cells of a run and checks the comparison with the host operator (module
/// documentation).
fn report<T: Real>(cells: &Cells, layout: LeafLayout) {
    for (&(name, quantity, level), cell) in cells {
        let degree = cell
            .degree
            .map_or(String::new(), |n| format!(" at degree {n}"));
        let comparison = if cell.host == 0.0 {
            "host operator equals the reference bit for bit; the bound applies".to_owned()
        } else if T::FLOAT == Precision::F64 {
            // Reported, not asserted (module documentation, "Against the host operator").
            format!(
                "host operator {:.2e}, ratio {:.2}, reported only in f64",
                cell.host,
                cell.device / cell.host
            )
        } else {
            assert!(
                cell.device <= 2.0 * cell.host,
                "{name}, {}, level {level}, {quantity}, {layout}: the device's {:e} exceeds \
                 twice the host operator's {:e}",
                T::FLOAT,
                cell.device,
                cell.host
            );
            format!(
                "host operator {:.2e}, ratio {:.2}, within twice",
                cell.host,
                cell.device / cell.host
            )
        };
        eprintln!(
            "  {name}, {}, level {level}, {quantity}, {layout}: worst {:.2e}{degree} \
             (bound {:.0e}; {comparison})",
            T::FLOAT,
            cell.device,
            tolerance::<T>()
        );
    }
}

/// Every check on the device in T and each layout the device operator offers on it.
fn check_precision<T: Real>(device: &mut Device) {
    let mut layouts = vec![LeafLayout::default_for(device.info(), 8, T::FLOAT)];
    if device.backend() == BackendKind::Cpu {
        layouts.push(LeafLayout::Cube { units: 8, tile: 4 });
    }
    for layout in layouts {
        let mut cells = Cells::new();
        p2m_and_l2p::<T>(device, layout, &mut cells);
        p2l_x_list::<T>(device, layout, &mut cells);
        m2p_w_list::<T>(device, layout, &mut cells);
        report::<T>(&cells, layout);
    }
}

/// Every check on the device of `kind`, in f32 and, where it runs, f64.
fn run(kind: BackendKind) {
    let mut device = Device::open(kind).unwrap_or_else(|error| panic!("{kind}: {error}"));
    eprintln!(
        "device leaf operators against nd_fmm_ref::leaf: on {}",
        device.info()
    );
    check_precision::<f32>(&mut device);
    if device.supports(Precision::F64) {
        check_precision::<f64>(&mut device);
    } else {
        eprintln!("  f64: not supported by this device, not run");
    }
    eprintln!("  frames: every X and W pair bit for bit against relative_frame");
    let not_run: Vec<&str> = BackendKind::ALL
        .into_iter()
        .filter(|&k| k != kind)
        .map(|k| match k {
            BackendKind::Cuda => "cuda (type-checked, not run)",
            BackendKind::Metal => "metal (its own ignored test)",
            BackendKind::Cpu => "cpu (its own test)",
        })
        .collect();
    eprintln!("backends run: {kind}; not run: {}", not_run.join(", "));
}

#[cfg(feature = "cpu")]
#[test]
fn device_leaf_operators_equal_the_reference_on_the_cpu_runtime() {
    run(BackendKind::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Metal: run by hand, outside the sandbox"]
fn device_leaf_operators_equal_the_reference_on_metal() {
    run(BackendKind::Metal);
}
