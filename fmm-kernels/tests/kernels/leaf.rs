//! The device leaf operators (T7, C4.3) and the harmonics they evaluate, on every layout
//! ([`LeafLayout`]) the backend runs, in f32 and, where the device supports it, f64, with
//! and without gradients, at p ∈ {0, 3, 8} (p = 20 in f64 in the ignored sweep).
//!
//! Error measures (docs/phase4/README.md, "Accuracy measures"; the measures of
//! fmm-exec/tests/operator/common.rs):
//! - **harmonics**, against `nd_fmm_math::harmonics` in f64 at the point as rounded to T:
//!   per point and degree, in the orthonormal weighting of CONVENTIONS §3.8 (regular
//!   harmonics weighted as multipoles, irregular as locals), the weighted norm of the
//!   difference relative to the weighted norm of the values; the gradient's three
//!   components together.
//! - **coefficients** (P2M, P2L), against `nd_fmm_ref::leaf` in f64 at the relative
//!   frames of CONVENTIONS §3.13 on the inputs as rounded to T: per degree in the §3.8
//!   weighting, relative to the term magnitudes τᵢ = Σⱼ |qⱼ Xᵢ(vⱼ)|.
//! - **values** (L2P, M2P): relative to the magnitudes of their terms, Σᵢ wᵢ |Cᵢ Xᵢ(v)| / r̂
//!   (wᵢ = 1 for m = 0, 2 otherwise), the gradient as a Euclidean norm over the three
//!   components, its terms divided by r̂².
//!
//! Tolerance 1e-13 in f64 and 1e-5 in f32 (the leaf-operator bounds of the README). The
//! same errors of `nd_fmm_ref::leaf` (and `nd-fmm-math`) in T are printed alongside. Bit
//! identity with the host does not apply: cubecl-opt fuses every lone a · b ± c of the
//! recursion and the contractions (spikes/device-arith/REPORT.md, "Recommendation", rule
//! 6); determinism, split invariance, frames and empty calls are checked bit for bit.
//!
//! Geometry: boxes and leaves on levels 2, 9 and 16 (M2P: target leaves on 2, 9 and 15,
//! W boxes one level below); leaves with 0, 1 and many points, points at the leaf centre,
//! on its faces and at its corners; P2L sources from the X list (the leaves on the level
//! above that touch the target's parent but not the target) and, beyond what the plan
//! forms, same-level leaves two boxes away; M2P sources from the W list (children of the
//! target's neighbours that do not touch it).

use nd_fmm_kernels::leaf::{
    Basis, LeafLayout, SourceInputs, TargetInputs, coefficients, harmonics, l2p, m2p, p2l, p2m,
    w_frames, x_frames,
};
use nd_fmm_kernels::view::{BoxCoordinates, IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{BackendKind, Device, DeviceBuffer, Precision};
use nd_fmm_math::{Layout, RealScalar, harmonics as host};
use nd_fmm_ref::{Frame, Workspace, leaf as reference};

use crate::common::{Rng, TestFloat, tests_on};

/// The float types of the tests.
trait Real: TestFloat + RealScalar {}

impl Real for f32 {}

impl Real for f64 {}

/// `x` rounded to `T`.
fn t<T: Real>(x: f64) -> T {
    <T as RealScalar>::from_f64(x)
}

/// `x` widened to f64.
fn w<T: Real>(x: T) -> f64 {
    <T as RealScalar>::to_f64(x)
}

/// The operator tolerance: 1e-13 in f64, 1e-5 in f32 against the f64 reference.
fn tolerance<T: Real>() -> f64 {
    if T::FLOAT == Precision::F64 {
        1e-13
    } else {
        1e-5
    }
}

/// The degrees of the default tests (the column sizes of fmm-kernels/CLAUDE.md).
const DEGREES: [usize; 3] = [0, 3, 8];

/// The levels of the C4.3 criterion.
const LEVELS: [u32; 3] = [2, 9, 16];

/// A box or leaf: level and index.
type Key = (u32, [u32; 3]);

/// The layouts a backend runs in these tests: on the CPU runtime the CPU layout (its
/// default) and the cube layout with up to 8 units and tiles of up to 4 points (at most
/// one unit per core there: 4 on the CI runner); on a GPU the default cube layout, a
/// smaller one and the CPU layout.
fn layouts(device: &Device) -> Vec<LeafLayout> {
    match device.backend() {
        BackendKind::Cpu => {
            let units = device.info().max_units_per_cube.clamp(1, 8);
            vec![
                LeafLayout::Cpu,
                LeafLayout::Cube {
                    units,
                    tile: units.min(4),
                },
            ]
        }
        _ => vec![
            LeafLayout::Cube {
                units: 64,
                tile: 32,
            },
            LeafLayout::Cube { units: 32, tile: 8 },
            LeafLayout::Cpu,
        ],
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

// --- Measures --------------------------------------------------------------------------

fn factorial(k: usize) -> f64 {
    (1..=k).map(|j| j as f64).product()
}

/// The weight of slot (n, m) in the orthonormal basis of CONVENTIONS §3.8: as a multipole
/// (regular harmonics, P2M) √((n + |m|)! (n − |m|)!) cₘ, as a local (irregular harmonics,
/// P2L) cₘ / √((n + |m|)! (n − |m|)!), with c₀ = 1 and cₘ = √2 otherwise.
fn weight(multipole: bool, n: usize, m: isize) -> f64 {
    let k = m.unsigned_abs();
    let c = if k == 0 { 1.0 } else { 2f64.sqrt() };
    let f = (factorial(n + k) * factorial(n - k)).sqrt();
    if multipole { f * c } else { c / f }
}

/// The worst per-degree error of `got` against `want` relative to `terms` (each a set of
/// slices of (p + 1)² values, combined in one norm per degree), in the weighting of
/// `multipole`: (error, degree). A degree whose terms vanish must have no error.
fn degree_error(
    multipole: bool,
    p: usize,
    got: &[&[f64]],
    want: &[&[f64]],
    terms: &[&[f64]],
) -> (f64, usize) {
    let layout = Layout::new(p);
    let mut worst = (0.0f64, 0usize);
    for (n, range) in layout.degrees() {
        let (mut e, mut s) = (0.0f64, 0.0f64);
        for ((g, wv), tm) in got.iter().zip(want).zip(terms) {
            for i in range.clone() {
                let wt = weight(multipole, n, layout.nm(i).1);
                e += (wt * (g[i] - wv[i])).powi(2);
                s += (wt * tm[i]).powi(2);
            }
        }
        let error = if s > 0.0 {
            (e / s).sqrt()
        } else {
            assert_eq!(e, 0.0, "degree {n}: error {e:e} with zero terms");
            0.0
        };
        if error > worst.0 {
            worst = (error, n);
        }
    }
    worst
}

/// The worst errors of a check, with the degree of the worst coefficient error.
#[derive(Clone, Copy, Debug, Default)]
struct Worst {
    device: f64,
    degree: usize,
    host: f64,
    gradient: f64,
    host_gradient: f64,
    cases: usize,
}

impl Worst {
    fn add(&mut self, (device, degree): (f64, usize), (host, _): (f64, usize)) {
        if device > self.device {
            self.device = device;
            self.degree = degree;
        }
        self.host = self.host.max(host);
        self.cases += 1;
    }

    /// Prints a coefficient check (P2M, P2L): the worst error and its degree.
    fn print_coefficients(&self, what: &str) {
        println!(
            "    {what}: max {:.2e} at degree {} (reference in T {:.2e}), {} boxes",
            self.device, self.degree, self.host, self.cases
        );
    }

    /// Prints a value check (L2P, M2P), with the gradient's errors if `gradients`.
    fn print_values(&self, what: &str, gradients: bool) {
        let gradient = if gradients {
            format!(
                ", ∇φ {:.2e} (reference in T {:.2e})",
                self.gradient, self.host_gradient
            )
        } else {
            String::new()
        };
        println!(
            "    {what}: max φ {:.2e} (reference in T {:.2e}){gradient}, {} targets",
            self.device, self.host, self.cases
        );
    }
}

/// The values and gradient components of the harmonics of `basis` at `v` in f64 (the
/// gradients only with `gradients`).
fn host_harmonics(basis: Basis, p: usize, v: [f64; 3], gradients: bool) -> Vec<Vec<f64>> {
    let n = coefficients(p);
    let mut values = vec![0.0; n];
    if !gradients {
        match basis {
            Basis::Regular => host::regular(p, v, &mut values),
            Basis::Irregular => host::irregular(p, v, &mut values),
        }
        return vec![values];
    }
    let [mut gx, mut gy, mut gz] = [0; 3].map(|_| vec![0.0; n]);
    let components = [&mut gx[..], &mut gy[..], &mut gz[..]];
    match basis {
        Basis::Regular => host::regular_grad(p, v, &mut values, components),
        Basis::Irregular => host::irregular_grad(p, v, &mut values, components),
    }
    vec![values, gx, gy, gz]
}

/// The harmonics of `basis` in T at `v` (as `nd-fmm-math` evaluates them in T).
fn host_harmonics_in<T: Real>(basis: Basis, p: usize, v: [T; 3], gradients: bool) -> Vec<Vec<f64>> {
    let n = coefficients(p);
    let mut values = vec![t::<T>(0.0); n];
    let mut out = Vec::new();
    if gradients {
        let [mut gx, mut gy, mut gz] = [0; 3].map(|_| vec![t::<T>(0.0); n]);
        let components = [&mut gx[..], &mut gy[..], &mut gz[..]];
        match basis {
            Basis::Regular => host::regular_grad(p, v, &mut values, components),
            Basis::Irregular => host::irregular_grad(p, v, &mut values, components),
        }
        out.push(values);
        out.extend([gx, gy, gz]);
    } else {
        match basis {
            Basis::Regular => host::regular(p, v, &mut values),
            Basis::Irregular => host::irregular(p, v, &mut values),
        }
        out.push(values);
    }
    out.into_iter()
        .map(|v| v.into_iter().map(w::<T>).collect())
        .collect()
}

/// The evaluation terms of `coefficients` of `basis` at `v` with the factors 1/r̂ and
/// 1/r̂²: the potential's and the Euclidean norm of the gradient's.
fn evaluation_terms(
    basis: Basis,
    p: usize,
    coefficients: &[f64],
    v: [f64; 3],
    ratio: f64,
) -> (f64, f64) {
    let layout = Layout::new(p);
    let h = host_harmonics(basis, p, v, true);
    let sum = |values: &[f64]| -> f64 {
        (0..layout.len())
            .map(|i| {
                let wt = if layout.nm(i).1 == 0 { 1.0 } else { 2.0 };
                wt * (coefficients[i] * values[i]).abs()
            })
            .sum()
    };
    let gradient = h[1..].iter().map(|g| sum(g).powi(2)).sum::<f64>().sqrt();
    (sum(&h[0]) / ratio, gradient / (ratio * ratio))
}

// --- Geometry ----------------------------------------------------------------------------

/// The frame (ĉ(s|t), r̂(s|t)) of CONVENTIONS §3.13 on the host, in f64 (exact).
fn frame(s: Key, t: Key) -> ([f64; 3], f64) {
    let big = s.0.max(t.0);
    let scale = 2f64.powi(t.0 as i32 - big as i32);
    let centre = std::array::from_fn(|k| {
        let cs = (2 * i64::from(s.1[k]) + 1) << (big - s.0);
        let ct = (2 * i64::from(t.1[k]) + 1) << (big - t.0);
        (cs - ct) as f64 * scale
    });
    (centre, 2f64.powi(t.0 as i32 - s.0 as i32))
}

/// Whether the closed boxes a and b share a point.
fn touches(a: Key, b: Key) -> bool {
    let bounds = |key: Key| {
        let size = 1u64 << (16 - key.0);
        key.1
            .map(|i| [u64::from(i) * size, (u64::from(i) + 1) * size])
    };
    let (a, b) = (bounds(a), bounds(b));
    (0..3).all(|k| a[k][0] <= b[k][1] && b[k][0] <= a[k][1])
}

/// The boxes on the level of `key` that share a point with it.
fn neighbours(key: Key) -> Vec<Key> {
    let n = 1i64 << key.0;
    (0..27)
        .map(|d| [d / 9, (d / 3) % 3, d % 3].map(|o| o as i64 - 1))
        .filter(|d| *d != [0, 0, 0])
        .filter_map(|d| {
            let j: [i64; 3] = std::array::from_fn(|k| i64::from(key.1[k]) + d[k]);
            j.iter()
                .all(|&c| (0..n).contains(&c))
                .then(|| (key.0, j.map(|c| c as u32)))
        })
        .collect()
}

/// The eight children of `key`.
fn children(key: Key) -> Vec<Key> {
    (0..8u32)
        .map(|c| {
            (
                key.0 + 1,
                [0, 1, 2].map(|k| 2 * key.1[k] + ((c >> (2 - k)) & 1)),
            )
        })
        .collect()
}

/// An interior index on `level` (every neighbour and the parent's neighbours exist).
fn interior(rng: &mut Rng, level: u32) -> [u32; 3] {
    let n = 1usize << level;
    [0; 3].map(|_| (2 + rng.below(n.saturating_sub(4).max(1))).min(n - 1) as u32)
}

/// A uniform leaf-scaled coordinate in [−1, 1] on the grid 2⁻⁵².
fn coordinate(rng: &mut Rng) -> f64 {
    let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
    (u * 2f64.powi(52)).round() / 2f64.powi(52)
}

/// A uniform charge in [−1, 1).
fn charge(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// `n` points of a leaf: with `special`, first the centre, two face points and a corner,
/// then uniform ones.
fn points<T: Real>(rng: &mut Rng, n: usize, special: bool) -> Vec<[T; 3]> {
    let fixed: [[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [1.0, 0.25, -0.5],
        [-0.75, -1.0, 0.125],
        [1.0, -1.0, 1.0],
    ];
    (0..n)
        .map(|i| {
            if special && i < fixed.len() {
                fixed[i].map(t::<T>)
            } else {
                [0; 3].map(|_| t::<T>(coordinate(rng)))
            }
        })
        .collect()
}

/// Random coefficients of degree ≤ p whose weighted entries are uniform in [−1, 1].
fn random_coefficients<T: Real>(rng: &mut Rng, p: usize, multipole: bool) -> Vec<T> {
    let layout = Layout::new(p);
    (0..layout.len())
        .map(|i| {
            let (n, m) = layout.nm(i);
            t::<T>(charge(rng) / weight(multipole, n, m))
        })
        .collect()
}

/// The point counts of the leaves of a case: 0, 1 and many (around and past the tiles).
const COUNTS: [usize; 6] = [5, 0, 1, 9, 37, 70];

/// CSR offsets of `rows`.
fn csr(rows: &[Vec<u32>]) -> (Vec<u32>, Vec<u32>) {
    let offsets = std::iter::once(0)
        .chain(rows.iter().scan(0u32, |total, row| {
            *total += row.len() as u32;
            Some(*total)
        }))
        .collect();
    (offsets, rows.iter().flatten().copied().collect())
}

/// CSR point offsets of `counts`.
fn offsets(counts: impl Iterator<Item = usize>) -> Vec<u32> {
    std::iter::once(0)
        .chain(counts.scan(0u32, |total, n| {
            *total += n as u32;
            Some(*total)
        }))
        .collect()
}

/// The boxes of every level 0..=16, `level` holding `keys`.
fn levels(level: u32, keys: &[[u32; 3]]) -> Vec<Vec<[u32; 3]>> {
    (0..=16)
        .map(|l| {
            if l == level {
                keys.to_vec()
            } else {
                Vec::new()
            }
        })
        .collect()
}

// --- Source operators (P2M, P2L) -------------------------------------------------------

/// One level's P2M or P2L call: boxes of `level` (rows), source leaves, rows of leaf
/// indices.
#[derive(Clone, Debug)]
struct SourceCase<T> {
    level: u32,
    boxes: Vec<[u32; 3]>,
    leaves: Vec<Key>,
    sources: Vec<Vec<[T; 3]>>,
    charges: Vec<Vec<T>>,
    rows: Vec<Vec<u32>>,
}

/// A source case on the device.
struct SourcesOnDevice<T: Real> {
    view: IndexView,
    level: usize,
    boxes: BoxCoordinates,
    leaves: LeafCoordinates,
    source_offsets: PointOffsets,
    sources: DeviceBuffer<T>,
    out: DeviceBuffer<T>,
}

impl<T: Real> SourceCase<T> {
    /// Uploads the case with `rows` as its view and `out` as the coefficients.
    fn upload(&self, device: &mut Device, rows: &[Vec<u32>], out: &[T]) -> SourcesOnDevice<T> {
        let (ro, en) = csr(rows);
        let mut store = Vec::new();
        for (u, q) in self.sources.iter().zip(&self.charges) {
            store.extend(u.iter().flatten());
            store.extend(q);
        }
        SourcesOnDevice {
            view: IndexView::upload(device, &ro, &en, self.leaves.len()).unwrap(),
            level: self.level as usize,
            boxes: BoxCoordinates::upload(device, &levels(self.level, &self.boxes)).unwrap(),
            leaves: LeafCoordinates::upload(device, &self.leaves).unwrap(),
            source_offsets: PointOffsets::upload(
                device,
                &offsets(self.sources.iter().map(Vec::len)),
            )
            .unwrap(),
            sources: device.upload(&store).unwrap(),
            out: device.upload(out).unwrap(),
        }
    }

    /// Runs P2M (`p2l` false) or P2L on the device from `initial` (zero if `None`) with
    /// the case's rows, and returns the coefficients.
    fn run(
        &self,
        device: &mut Device,
        layout: LeafLayout,
        p: usize,
        irregular: bool,
        initial: Option<&[T]>,
    ) -> Vec<T> {
        let zeros = vec![t::<T>(0.0); coefficients(p) * self.boxes.len()];
        let mut on = self.upload(device, &self.rows, initial.unwrap_or(&zeros));
        launch_sources(device, &mut on, layout, p, irregular);
        download(device, &on.out)
    }

    /// The sources of entry leaf j of box t at the frame of the operator, in f64: u_s
    /// (P2M) or (u_s − ĉ(t|s)) / r̂(t|s) (P2L), with the frame of `nd_fmm_ref::leaf`.
    fn frame_of(&self, t_box: usize, j: usize, irregular: bool) -> Frame<f64> {
        if irregular {
            let (c, r) = frame((self.level, self.boxes[t_box]), self.leaves[j]);
            Frame::new(c, r)
        } else {
            Frame::new([0.0; 3], 1.0)
        }
    }

    /// The reference coefficients of box t in f64 and in T, and their terms, from
    /// `initial`.
    fn reference(&self, p: usize, irregular: bool, t_box: usize, initial: &[T]) -> [Vec<f64>; 3] {
        let n = coefficients(p);
        let start = &initial[t_box * n..(t_box + 1) * n];
        let mut want: Vec<f64> = start.iter().map(|&v| w(v)).collect();
        let mut in_t: Vec<T> = start.to_vec();
        let mut terms: Vec<f64> = want.iter().map(|v| v.abs()).collect();
        let mut ws64 = Workspace::<f64>::new(p);
        let mut ws = Workspace::<T>::new(p);
        for &j in &self.rows[t_box] {
            let j = j as usize;
            let frame = self.frame_of(t_box, j, irregular);
            let frame_t = Frame::new(frame.centre.map(t::<T>), t::<T>(frame.radius));
            let u64s: Vec<[f64; 3]> = self.sources[j].iter().map(|u| u.map(w::<T>)).collect();
            let q64: Vec<f64> = self.charges[j].iter().map(|&q| w(q)).collect();
            if irregular {
                reference::p2l(p, &frame, &u64s, &q64, &mut ws64, &mut want);
                reference::p2l(
                    p,
                    &frame_t,
                    &self.sources[j],
                    &self.charges[j],
                    &mut ws,
                    &mut in_t,
                );
            } else {
                reference::p2m(p, &frame, &u64s, &q64, &mut ws64, &mut want);
                reference::p2m(
                    p,
                    &frame_t,
                    &self.sources[j],
                    &self.charges[j],
                    &mut ws,
                    &mut in_t,
                );
            }
            let basis = if irregular {
                Basis::Irregular
            } else {
                Basis::Regular
            };
            for (y, q) in u64s.iter().zip(&q64) {
                let values = &host_harmonics(basis, p, frame.scaled(*y), false)[0];
                for (o, v) in terms.iter_mut().zip(values) {
                    *o += (q * v).abs();
                }
            }
        }
        [want, in_t.into_iter().map(w::<T>).collect(), terms]
    }

    /// Checks every box of `got` (computed from `initial`) against the reference; adds
    /// the errors to `worst`.
    fn check(
        &self,
        what: &str,
        p: usize,
        irregular: bool,
        initial: &[T],
        got: &[T],
        worst: &mut Worst,
    ) {
        let n = coefficients(p);
        for t_box in 0..self.boxes.len() {
            let [want, in_t, terms] = self.reference(p, irregular, t_box, initial);
            let got: Vec<f64> = got[t_box * n..(t_box + 1) * n]
                .iter()
                .map(|&v| w(v))
                .collect();
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{what}: box {t_box}: not finite"
            );
            let multipole = !irregular;
            let device = degree_error(multipole, p, &[&got], &[&want], &[&terms]);
            let host = degree_error(multipole, p, &[&in_t], &[&want], &[&terms]);
            assert!(
                device.0 <= tolerance::<T>(),
                "{what}: p = {p}, box {t_box}: error {:e} at degree {} exceeds {:e} (reference \
                 in T {:e})",
                device.0,
                device.1,
                tolerance::<T>(),
                host.0
            );
            worst.add(device, host);
        }
    }
}

/// Launches P2M or P2L of `on` in `layout`.
fn launch_sources<T: Real>(
    device: &mut Device,
    on: &mut SourcesOnDevice<T>,
    layout: LeafLayout,
    p: usize,
    irregular: bool,
) {
    let inputs = SourceInputs {
        view: &on.view,
        level: on.level,
        boxes: &on.boxes,
        leaves: &on.leaves,
        source_offsets: &on.source_offsets,
        sources: on.sources.as_slice(),
    };
    let out = on.out.as_slice_mut();
    if irregular {
        p2l(device, layout, p, &inputs, out).unwrap();
    } else {
        p2m(device, layout, p, &inputs, out).unwrap();
    }
}

/// A buffer downloaded.
fn download<T: Real>(device: &mut Device, buffer: &DeviceBuffer<T>) -> Vec<T> {
    let mut out = vec![t::<T>(0.0); buffer.len()];
    device.download(buffer.as_slice(), &mut out).unwrap();
    out
}

/// A P2M case on `level`: six boxes, each its own leaf with the point counts of
/// [`COUNTS`] (the first with the special points), and a seventh box without a row.
fn p2m_case<T: Real>(rng: &mut Rng, level: u32) -> SourceCase<T> {
    let mut boxes: Vec<[u32; 3]> = Vec::new();
    while boxes.len() < COUNTS.len() + 1 {
        let n = 1usize << level;
        let index = [0; 3].map(|_| rng.below(n) as u32);
        if !boxes.contains(&index) {
            boxes.push(index);
        }
    }
    let leaves: Vec<Key> = boxes.iter().map(|&b| (level, b)).collect();
    let sources: Vec<Vec<[T; 3]>> = (0..leaves.len())
        .map(|j| points(rng, *COUNTS.get(j).unwrap_or(&3), j == 0))
        .collect();
    let charges = sources
        .iter()
        .map(|s| s.iter().map(|_| t::<T>(charge(rng))).collect())
        .collect();
    let mut rows: Vec<Vec<u32>> = (0..boxes.len() as u32).map(|j| vec![j]).collect();
    rows[COUNTS.len()].clear();
    SourceCase {
        level,
        boxes,
        leaves,
        sources,
        charges,
        rows,
    }
}

/// A P2L case on `level`: three target boxes, each with its X list (the leaves
/// on the level above that touch its parent but not it; at most `max_sources` of them,
/// in ascending order) and two same-level leaves two boxes away; and a fourth box
/// without a row.
fn p2l_case<T: Real>(rng: &mut Rng, level: u32, max_sources: usize) -> SourceCase<T> {
    let mut case = SourceCase {
        level,
        boxes: Vec::new(),
        leaves: Vec::new(),
        sources: Vec::new(),
        charges: Vec::new(),
        rows: Vec::new(),
    };
    for r in 0..3 {
        // A target whose X list is not empty: on every level some boxes have none (those
        // facing the inside of their parent).
        let (target, mut keys) = loop {
            let target = (level, [0; 3].map(|_| rng.below(1 << level) as u32));
            let parent = (level - 1, target.1.map(|i| i / 2));
            let keys: Vec<Key> = neighbours(parent)
                .into_iter()
                .filter(|&s| !touches(s, target))
                .collect();
            if !keys.is_empty() {
                break (target, keys);
            }
        };
        keys.truncate(max_sources);
        let n = 1i64 << level;
        for d in [[2i64, 0, 1], [-2, 2, -2], [0, -3, 2]] {
            let j: [i64; 3] = std::array::from_fn(|k| i64::from(target.1[k]) + d[k]);
            if j.iter().all(|&c| (0..n).contains(&c)) {
                keys.push((level, j.map(|c| c as u32)));
            }
        }
        let mut row = Vec::new();
        for (k, key) in keys.into_iter().enumerate() {
            let count = COUNTS[(r + k) % COUNTS.len()];
            row.push(case.leaves.len() as u32);
            case.leaves.push(key);
            let s = points(rng, count, k == 0);
            case.charges
                .push(s.iter().map(|_| t::<T>(charge(rng))).collect());
            case.sources.push(s);
        }
        case.boxes.push(target.1);
        case.rows.push(row);
    }
    let extra = (level, interior(rng, level));
    case.boxes.push(extra.1);
    case.rows.push(Vec::new());
    case
}

// --- Target operators (L2P, M2P) -------------------------------------------------------

/// One level's L2P or M2P call: target leaves (rows), the boxes of the entry level and
/// their coefficients, rows of box indices.
#[derive(Clone, Debug)]
struct TargetCase<T> {
    level: u32,
    leaves: Vec<Key>,
    targets: Vec<Vec<[T; 3]>>,
    /// The boxes of the entry level: `level` for L2P, `level + 1` for M2P.
    boxes: Vec<[u32; 3]>,
    coefficients: Vec<T>,
    rows: Vec<Vec<u32>>,
}

/// A target case on the device.
struct TargetsOnDevice<T: Real> {
    view: IndexView,
    level: usize,
    boxes: BoxCoordinates,
    leaves: LeafCoordinates,
    target_offsets: PointOffsets,
    target_input: DeviceBuffer<T>,
    coefficients: DeviceBuffer<T>,
    output: DeviceBuffer<T>,
}

impl<T: Real> TargetCase<T> {
    fn entry_level(&self, irregular: bool) -> u32 {
        self.level + u32::from(irregular)
    }

    /// Uploads the case with `rows` as its view and `output` as the target output.
    fn upload(
        &self,
        device: &mut Device,
        irregular: bool,
        rows: &[Vec<u32>],
        output: &[T],
    ) -> TargetsOnDevice<T> {
        let (ro, en) = csr(rows);
        let input: Vec<T> = self.targets.iter().flatten().flatten().copied().collect();
        TargetsOnDevice {
            view: IndexView::upload(device, &ro, &en, self.boxes.len()).unwrap(),
            level: self.level as usize,
            boxes: BoxCoordinates::upload(
                device,
                &levels(self.entry_level(irregular), &self.boxes),
            )
            .unwrap(),
            leaves: LeafCoordinates::upload(device, &self.leaves).unwrap(),
            target_offsets: PointOffsets::upload(
                device,
                &offsets(self.targets.iter().map(Vec::len)),
            )
            .unwrap(),
            target_input: device.upload(&input).unwrap(),
            coefficients: device.upload(&self.coefficients).unwrap(),
            output: device.upload(output).unwrap(),
        }
    }

    /// The target output store, every value from `value`.
    fn output_store(&self, gradients: bool, mut value: impl FnMut() -> T) -> Vec<T> {
        let per_point = if gradients { 4 } else { 1 };
        let n: usize = self.targets.iter().map(Vec::len).sum();
        (0..per_point * n).map(|_| value()).collect()
    }

    /// The value range of leaf r's chunk in the target output.
    fn output_range(&self, r: usize, gradients: bool) -> std::ops::Range<usize> {
        let per_point = if gradients { 4 } else { 1 };
        let before: usize = self.targets[..r].iter().map(Vec::len).sum();
        per_point * before..per_point * (before + self.targets[r].len())
    }

    /// Runs L2P (`irregular` false) or M2P on the device from `initial` (zero if `None`).
    fn run(
        &self,
        device: &mut Device,
        layout: LeafLayout,
        p: usize,
        (irregular, gradients): (bool, bool),
        initial: Option<&[T]>,
    ) -> Vec<T> {
        let zeros = self.output_store(gradients, || t::<T>(0.0));
        let mut on = self.upload(device, irregular, &self.rows, initial.unwrap_or(&zeros));
        launch_targets(device, &mut on, layout, p, (irregular, gradients));
        download(device, &on.output)
    }

    /// The frame of entry box s seen from target leaf r for M2P, the unit frame for L2P.
    fn frame_of(&self, r: usize, s: usize, irregular: bool) -> Frame<f64> {
        if irregular {
            let (c, ratio) = frame((self.level + 1, self.boxes[s]), self.leaves[r]);
            Frame::new(c, ratio)
        } else {
            Frame::new([0.0; 3], 1.0)
        }
    }

    /// Checks every target of `got` (from `initial`) against the reference.
    fn check(
        &self,
        what: &str,
        p: usize,
        (irregular, gradients): (bool, bool),
        initial: &[T],
        got: &[T],
        worst: &mut Worst,
    ) {
        let n = coefficients(p);
        let basis = if irregular {
            Basis::Irregular
        } else {
            Basis::Regular
        };
        let mut ws64 = Workspace::<f64>::new(p);
        let mut ws = Workspace::<T>::new(p);
        for r in 0..self.leaves.len() {
            let x = &self.targets[r];
            let count = x.len();
            if count == 0 {
                continue;
            }
            let range = self.output_range(r, gradients);
            let split = |chunk: &[T]| -> (Vec<f64>, Vec<[f64; 3]>) {
                let phi = chunk[..count].iter().map(|&v| w(v)).collect();
                let grad = if gradients {
                    chunk[count..]
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|g| g.map(w::<T>))
                        .collect()
                } else {
                    vec![[0.0; 3]; count]
                };
                (phi, grad)
            };
            let (phi0, grad0) = split(&initial[range.clone()]);
            let (phi, grad) = split(&got[range]);
            let (mut want_phi, mut want_grad) = (phi0.clone(), grad0.clone());
            let start_t = |v: &[f64]| -> Vec<T> { v.iter().map(|&v| t::<T>(v)).collect() };
            let (mut t_phi, mut t_grad) = (
                start_t(&phi0),
                grad0.iter().map(|g| g.map(t::<T>)).collect::<Vec<[T; 3]>>(),
            );
            let mut terms: Vec<(f64, f64)> = phi0
                .iter()
                .zip(&grad0)
                .map(|(p0, g0)| (p0.abs(), g0.iter().map(|v| v * v).sum::<f64>().sqrt()))
                .collect();
            let x64: Vec<[f64; 3]> = x.iter().map(|u| u.map(w::<T>)).collect();
            for &s in &self.rows[r] {
                let s = s as usize;
                let c_t = &self.coefficients[s * n..(s + 1) * n];
                let c64: Vec<f64> = c_t.iter().map(|&v| w(v)).collect();
                let frame = self.frame_of(r, s, irregular);
                let frame_t = Frame::new(frame.centre.map(t::<T>), t::<T>(frame.radius));
                let g64 = gradients.then_some(&mut want_grad[..]);
                let gt = gradients.then_some(&mut t_grad[..]);
                if irregular {
                    reference::m2p(p, &frame, &c64, &x64, &mut ws64, &mut want_phi, g64);
                    reference::m2p(p, &frame_t, c_t, x, &mut ws, &mut t_phi, gt);
                } else {
                    reference::l2p(p, &frame, &c64, &x64, &mut ws64, &mut want_phi, g64);
                    reference::l2p(p, &frame_t, c_t, x, &mut ws, &mut t_phi, gt);
                }
                for (i, xi) in x64.iter().enumerate() {
                    let (tp, tg) =
                        evaluation_terms(basis, p, &c64, frame.scaled(*xi), frame.radius);
                    terms[i].0 += tp;
                    terms[i].1 += tg;
                }
            }
            for i in 0..count {
                let rel = |e: f64, s: f64| if s > 0.0 { e / s } else { e };
                assert!(
                    phi[i].is_finite(),
                    "{what}: leaf {r}, target {i}: φ = {}",
                    phi[i]
                );
                let e = rel((phi[i] - want_phi[i]).abs(), terms[i].0);
                let h = rel((w(t_phi[i]) - want_phi[i]).abs(), terms[i].0);
                assert!(
                    e <= tolerance::<T>(),
                    "{what}: p = {p}, leaf {r}, target {i}: φ error {e:e} exceeds {:e} \
                     (reference in T {h:e})",
                    tolerance::<T>()
                );
                worst.add((e, 0), (h, 0));
                if gradients {
                    let norm = |a: [f64; 3], b: [f64; 3]| {
                        (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
                    };
                    let e = rel(norm(grad[i], want_grad[i]), terms[i].1);
                    let h = rel(norm(t_grad[i].map(w::<T>), want_grad[i]), terms[i].1);
                    assert!(
                        e <= tolerance::<T>(),
                        "{what}: p = {p}, leaf {r}, target {i}: ∇φ error {e:e} exceeds {:e} \
                         (reference in T {h:e})",
                        tolerance::<T>()
                    );
                    worst.gradient = worst.gradient.max(e);
                    worst.host_gradient = worst.host_gradient.max(h);
                }
            }
        }
    }
}

/// Launches L2P or M2P of `on` in `layout`.
fn launch_targets<T: Real>(
    device: &mut Device,
    on: &mut TargetsOnDevice<T>,
    layout: LeafLayout,
    p: usize,
    (irregular, gradients): (bool, bool),
) {
    let inputs = TargetInputs {
        view: &on.view,
        level: on.level,
        first_leaf: 0,
        boxes: &on.boxes,
        leaves: &on.leaves,
        target_offsets: &on.target_offsets,
        target_input: on.target_input.as_slice(),
    };
    let coefficients = on.coefficients.as_slice();
    let output = on.output.as_slice_mut();
    if irregular {
        m2p(device, layout, p, gradients, &inputs, coefficients, output).unwrap();
    } else {
        l2p(device, layout, p, gradients, &inputs, coefficients, output).unwrap();
    }
}

/// An L2P case on `level`: six target leaves with the point counts of [`COUNTS`] (the
/// first with the special points), each its own box, random locals; a seventh leaf
/// without a row.
fn l2p_case<T: Real>(rng: &mut Rng, level: u32, p: usize) -> TargetCase<T> {
    let source = p2m_case::<T>(rng, level);
    let targets: Vec<Vec<[T; 3]>> = (0..source.leaves.len())
        .map(|j| points(rng, *COUNTS.get(j).unwrap_or(&3), j == 0))
        .collect();
    let coefficients = (0..source.boxes.len())
        .flat_map(|_| random_coefficients::<T>(rng, p, false))
        .collect();
    TargetCase {
        level,
        leaves: source.leaves,
        targets,
        boxes: source.boxes,
        coefficients,
        rows: source.rows,
    }
}

/// An M2P case on `level`: three interior target leaves with point counts from
/// [`COUNTS`] (the first with the special points), each with at most `max_sources` of its
/// W list (children of its neighbours that do not touch it, ascending), random
/// multipoles; a fourth leaf without a row.
fn m2p_case<T: Real>(rng: &mut Rng, level: u32, p: usize, max_sources: usize) -> TargetCase<T> {
    let mut case = TargetCase {
        level,
        leaves: Vec::new(),
        targets: Vec::new(),
        boxes: Vec::new(),
        coefficients: Vec::new(),
        rows: Vec::new(),
    };
    for r in 0..4 {
        let target = (level, interior(rng, level));
        let mut keys: Vec<Key> = neighbours(target)
            .into_iter()
            .flat_map(children)
            .filter(|&s| !touches(s, target))
            .collect();
        // A spread-out subset, in the order of the list.
        let stride = keys.len().div_ceil(max_sources).max(1);
        keys = keys.into_iter().step_by(stride).collect();
        if r == 3 {
            keys.clear();
        }
        let mut row = Vec::new();
        for key in keys {
            row.push(case.boxes.len() as u32);
            case.boxes.push(key.1);
            case.coefficients
                .extend(random_coefficients::<T>(rng, p, true));
        }
        case.leaves.push(target);
        case.targets.push(points(rng, [5, 70, 1, 9][r], r == 0));
        case.rows.push(row);
    }
    case
}

// --- Tests --------------------------------------------------------------------------------

/// Harmonics: regular at points inside the unit ball and the box's sphere (|x| ≤ √3, the
/// centre, faces and corners included), irregular outside (|x| in [2, 11]), with and
/// without gradients, against `nd_fmm_math::harmonics` (module documentation).
fn harmonics_match_nd_fmm_math(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        harmonics_at::<T>(device, &DEGREES);
    }
    each_precision!(device, body);
}

/// The harmonics check at `degrees`.
fn harmonics_at<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = Rng::new(0x7e57_7001);
    let count = 400;
    for basis in [Basis::Regular, Basis::Irregular] {
        let mut pts: Vec<[T; 3]> = points(&mut rng, count, true);
        if basis == Basis::Irregular {
            for p in &mut pts {
                let v = p.map(w::<T>);
                let norm = v.iter().map(|c| c * c).sum::<f64>().sqrt().max(1e-3);
                let radius = 2.0 + 9.0 * charge(&mut rng).abs();
                let mut scaled = v.map(|c| c / norm * radius);
                if norm <= 1e-3 {
                    scaled = [0.0, 0.0, radius];
                }
                *p = scaled.map(t::<T>);
            }
        } else {
            // Inside the unit ball too: every eighth point scaled by a random factor.
            for (i, p) in pts.iter_mut().enumerate().skip(4) {
                if i % 8 == 0 {
                    let f = 0.25 + 0.75 * charge(&mut rng).abs();
                    *p = p.map(|c| t::<T>(w(c) * f));
                }
            }
        }
        let flat: Vec<T> = pts.iter().flatten().copied().collect();
        let input = device.upload(&flat).unwrap();
        for &p in degrees {
            for gradients in [false, true] {
                let per_point = coefficients(p) * if gradients { 4 } else { 1 };
                let mut out = device.alloc::<T>(per_point * count).unwrap();
                harmonics(
                    device,
                    p,
                    basis,
                    gradients,
                    input.as_slice(),
                    out.as_slice_mut(),
                )
                .unwrap();
                let got = download(device, &out);
                let (mut worst, mut worst_degree, mut worst_host, mut identical) =
                    (0.0f64, 0, 0.0f64, 0);
                let multipole = basis == Basis::Regular;
                for (i, x) in pts.iter().enumerate() {
                    let want = host_harmonics(basis, p, x.map(w::<T>), gradients);
                    let in_t = host_harmonics_in::<T>(basis, p, *x, gradients);
                    let chunk = &got[i * per_point..(i + 1) * per_point];
                    let n = coefficients(p);
                    let got_parts: Vec<Vec<f64>> = chunk
                        .chunks(n)
                        .map(|c| c.iter().map(|&v| w(v)).collect())
                        .collect();
                    identical += got_parts
                        .iter()
                        .flatten()
                        .zip(in_t.iter().flatten())
                        .filter(|(a, b)| a.to_bits() == b.to_bits())
                        .count();
                    let abs: Vec<Vec<f64>> = want
                        .iter()
                        .map(|v| v.iter().map(|x| x.abs()).collect())
                        .collect();
                    // The values, then the three gradient components together.
                    let groups: &[(usize, usize)] = if gradients {
                        &[(0, 1), (1, 4)]
                    } else {
                        &[(0, 1)]
                    };
                    for &(first, end) in groups {
                        let g = first..end;
                        let gp: Vec<&[f64]> =
                            got_parts[g.clone()].iter().map(Vec::as_slice).collect();
                        let wp: Vec<&[f64]> = want[g.clone()].iter().map(Vec::as_slice).collect();
                        let tp: Vec<&[f64]> = abs[g.clone()].iter().map(Vec::as_slice).collect();
                        let hp: Vec<&[f64]> = in_t[g].iter().map(Vec::as_slice).collect();
                        let (e, degree) = degree_error(multipole, p, &gp, &wp, &tp);
                        let (h, _) = degree_error(multipole, p, &hp, &wp, &tp);
                        assert!(
                            e <= tolerance::<T>(),
                            "{basis:?}, {}, p = {p}, gradients {gradients}, point {i} {x:?}: \
                             error {e:e} at degree {degree} (nd-fmm-math in T {h:e})",
                            T::FLOAT
                        );
                        if e > worst {
                            worst = e;
                            worst_degree = degree;
                        }
                        worst_host = worst_host.max(h);
                    }
                }
                println!(
                    "  {basis:?}, {}, p = {p}, gradients {gradients}: max {worst:.2e} at degree \
                     {worst_degree} (nd-fmm-math in T {worst_host:.2e}); {identical} of {} \
                     values equal nd-fmm-math in T bit for bit",
                    T::FLOAT,
                    count * per_point
                );
            }
        }
    }
}

/// P2M and L2P on levels 2, 9 and 16 against `nd_fmm_ref::leaf` at the unit frame.
fn p2m_and_l2p(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        p2m_and_l2p_at::<T>(device, &DEGREES);
    }
    each_precision!(device, body);
}

fn p2m_and_l2p_at<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = Rng::new(0x7e57_7002);
    for layout in layouts(device) {
        for &level in &LEVELS {
            let mut worst_p2m = Worst::default();
            let mut worst_l2p = [Worst::default(); 2];
            for &p in degrees {
                let case = p2m_case::<T>(&mut rng, level);
                let initial = vec![t::<T>(0.0); coefficients(p) * case.boxes.len()];
                let got = case.run(device, layout, p, false, None);
                case.check(
                    &format!("P2M, {layout}, level {level}"),
                    p,
                    false,
                    &initial,
                    &got,
                    &mut worst_p2m,
                );
                let case = l2p_case::<T>(&mut rng, level, p);
                for (g, gradients) in [false, true].into_iter().enumerate() {
                    let initial = case.output_store(gradients, || t::<T>(0.0));
                    let got = case.run(device, layout, p, (false, gradients), None);
                    case.check(
                        &format!("L2P, {layout}, level {level}"),
                        p,
                        (false, gradients),
                        &initial,
                        &got,
                        &mut worst_l2p[g],
                    );
                }
            }
            worst_p2m.print_coefficients(&format!("P2M, {layout}, {}, level {level}", T::FLOAT));
            worst_l2p[0].print_values(
                &format!("L2P, {layout}, {}, level {level}", T::FLOAT),
                false,
            );
            worst_l2p[1].print_values(
                &format!("L2P, {layout}, {}, level {level}, gradients", T::FLOAT),
                true,
            );
        }
    }
}

/// P2L for X-list geometry (and same-level leaves two boxes away) on levels 2, 9 and 16.
fn p2l_x_list(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        p2l_at::<T>(device, &DEGREES);
    }
    each_precision!(device, body);
}

fn p2l_at<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = Rng::new(0x7e57_7003);
    for layout in layouts(device) {
        for &level in &LEVELS {
            let mut worst = Worst::default();
            for &p in degrees {
                let case = p2l_case::<T>(&mut rng, level, 12);
                let initial = vec![t::<T>(0.0); coefficients(p) * case.boxes.len()];
                let got = case.run(device, layout, p, true, None);
                case.check(
                    &format!("P2L, {layout}, level {level}"),
                    p,
                    true,
                    &initial,
                    &got,
                    &mut worst,
                );
            }
            worst.print_coefficients(&format!("P2L, {layout}, {}, level {level}", T::FLOAT));
        }
    }
}

/// M2P for W-list geometry, target leaves on levels 2, 9 and 15.
fn m2p_w_list(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        m2p_at::<T>(device, &DEGREES);
    }
    each_precision!(device, body);
}

fn m2p_at<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = Rng::new(0x7e57_7004);
    for layout in layouts(device) {
        for level in [2u32, 9, 15] {
            let mut worst = [Worst::default(); 2];
            for &p in degrees {
                let case = m2p_case::<T>(&mut rng, level, p, 12);
                for (g, gradients) in [false, true].into_iter().enumerate() {
                    let initial = case.output_store(gradients, || t::<T>(0.0));
                    let got = case.run(device, layout, p, (true, gradients), None);
                    case.check(
                        &format!("M2P, {layout}, level {level}"),
                        p,
                        (true, gradients),
                        &initial,
                        &got,
                        &mut worst[g],
                    );
                }
            }
            worst[0].print_values(
                &format!("M2P, {layout}, {}, level {level}", T::FLOAT),
                false,
            );
            worst[1].print_values(
                &format!("M2P, {layout}, {}, level {level}, gradients", T::FLOAT),
                true,
            );
        }
    }
}

/// Frames: the device frames of every entry of an X view and a W view equal the host's
/// exact (ĉ, r̂) bit for bit, for random keys on every level, the levels 0–16 and the
/// opposite corners of the domain.
fn frames(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_7005);
        let key = |rng: &mut Rng, level: u32| -> Key {
            (level, [0; 3].map(|_| rng.below(1 << level) as u32))
        };
        let mut checked = 0;
        for level in 0..=15u32 {
            // X: boxes of `level`, leaves anywhere; W: leaves anywhere, boxes of level + 1.
            let boxes: Vec<[u32; 3]> = (0..3).map(|_| key(&mut rng, level).1).collect();
            let mut leaves: Vec<Key> = (0..=16).map(|l| key(&mut rng, l)).collect();
            leaves.push((16, [65_535; 3]));
            let rows: Vec<Vec<u32>> = (0..3).map(|_| (0..leaves.len() as u32).collect()).collect();
            let (ro, en) = csr(&rows);
            let view = IndexView::upload(device, &ro, &en, leaves.len()).unwrap();
            let mut corner = boxes.clone();
            if level == 15 {
                corner[0] = [0; 3];
            }
            let coordinates = BoxCoordinates::upload(device, &levels(level, &corner)).unwrap();
            let leaf_coordinates = LeafCoordinates::upload(device, &leaves).unwrap();
            let offsets = PointOffsets::upload(device, &vec![0; leaves.len() + 1]).unwrap();
            let sources = device.alloc::<T>(0).unwrap();
            let inputs = SourceInputs {
                view: &view,
                level: level as usize,
                boxes: &coordinates,
                leaves: &leaf_coordinates,
                source_offsets: &offsets,
                sources: sources.as_slice(),
            };
            let mut out = device.alloc::<T>(4 * en.len()).unwrap();
            x_frames(device, &inputs, out.as_slice_mut()).unwrap();
            let got = download(device, &out);
            for (r, row) in rows.iter().enumerate() {
                for (e, &j) in (ro[r] as usize..).zip(row) {
                    let (c, ratio) = frame((level, corner[r]), leaves[j as usize]);
                    let want = [c[0], c[1], c[2], ratio].map(t::<T>);
                    for k in 0..4 {
                        assert_eq!(
                            got[4 * e + k].bits(),
                            want[k].bits(),
                            "X, level {level}, row {r}, entry {e}, {k}"
                        );
                    }
                    checked += 1;
                }
            }
            // W: rows are the leaves, entries the boxes of level + 1.
            let wboxes: Vec<[u32; 3]> = (0..4).map(|_| key(&mut rng, level + 1).1).collect();
            let wrows: Vec<Vec<u32>> = (0..leaves.len()).map(|_| (0..4).collect()).collect();
            let (wro, wen) = csr(&wrows);
            let wview = IndexView::upload(device, &wro, &wen, wboxes.len()).unwrap();
            let wcoordinates = BoxCoordinates::upload(device, &levels(level + 1, &wboxes)).unwrap();
            let target_offsets = PointOffsets::upload(device, &vec![0; leaves.len() + 1]).unwrap();
            let inputs = TargetInputs {
                view: &wview,
                level: level as usize,
                first_leaf: 0,
                boxes: &wcoordinates,
                leaves: &leaf_coordinates,
                target_offsets: &target_offsets,
                target_input: sources.as_slice(),
            };
            let mut out = device.alloc::<T>(4 * wen.len()).unwrap();
            w_frames(device, &inputs, out.as_slice_mut()).unwrap();
            let got = download(device, &out);
            for (r, row) in wrows.iter().enumerate() {
                for (e, &s) in (wro[r] as usize..).zip(row) {
                    let (c, ratio) = frame((level + 1, wboxes[s as usize]), leaves[r]);
                    let want = [c[0], c[1], c[2], ratio].map(t::<T>);
                    for k in 0..4 {
                        assert_eq!(
                            got[4 * e + k].bits(),
                            want[k].bits(),
                            "W, level {level}, row {r}, entry {e}, {k}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        println!("  {}: {checked} X and W frames bit for bit", T::FLOAT);
    }
    each_precision!(device, body);
}

/// Accumulation onto a nonzero output and calls that add nothing: rows without entries
/// and leaves without points leave every bit as it was, and a call without entries
/// launches nothing.
fn accumulation_and_empty_calls(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_7006);
        let p = 3;
        for layout in layouts(device) {
            // P2M and P2L onto random coefficients.
            for irregular in [false, true] {
                let case = if irregular {
                    p2l_case::<T>(&mut rng, 9, 8)
                } else {
                    p2m_case::<T>(&mut rng, 9)
                };
                let initial: Vec<T> = (0..coefficients(p) * case.boxes.len())
                    .map(|_| T::random_normal(&mut rng))
                    .collect();
                let got = case.run(device, layout, p, irregular, Some(&initial));
                let mut worst = Worst::default();
                case.check("onto nonzero", p, irregular, &initial, &got, &mut worst);
                let n = coefficients(p);
                for (r, row) in case.rows.iter().enumerate() {
                    let points: usize = row.iter().map(|&j| case.sources[j as usize].len()).sum();
                    if points == 0 {
                        assert!(
                            (r * n..(r + 1) * n).all(|i| got[i].bits() == initial[i].bits()),
                            "{layout}: box {r} without sources changed"
                        );
                    }
                }
                // No entry at all: nothing launched, nothing changed.
                let none = vec![Vec::new(); case.rows.len()];
                let mut on = case.upload(device, &none, &initial);
                let before = device.counters().launches;
                launch_sources(device, &mut on, layout, p, irregular);
                assert_eq!(device.counters().launches, before, "{layout}: empty call");
                assert!(
                    download(device, &on.out)
                        .iter()
                        .zip(&initial)
                        .all(|(a, b)| a.bits() == b.bits())
                );
            }
            // L2P and M2P onto random outputs.
            for irregular in [false, true] {
                let case = if irregular {
                    m2p_case::<T>(&mut rng, 9, p, 8)
                } else {
                    l2p_case::<T>(&mut rng, 9, p)
                };
                for gradients in [false, true] {
                    let initial = case.output_store(gradients, || T::random_normal(&mut rng));
                    let got = case.run(device, layout, p, (irregular, gradients), Some(&initial));
                    let mut worst = Worst::default();
                    case.check(
                        "onto nonzero",
                        p,
                        (irregular, gradients),
                        &initial,
                        &got,
                        &mut worst,
                    );
                    for (r, row) in case.rows.iter().enumerate() {
                        if row.is_empty() {
                            let range = case.output_range(r, gradients);
                            assert!(
                                range.clone().all(|i| got[i].bits() == initial[i].bits()),
                                "{layout}: leaf {r} without entries changed"
                            );
                        }
                    }
                    let none = vec![Vec::new(); case.rows.len()];
                    let mut on = case.upload(device, irregular, &none, &initial);
                    let before = device.counters().launches;
                    launch_targets(device, &mut on, layout, p, (irregular, gradients));
                    assert_eq!(device.counters().launches, before, "{layout}: empty call");
                    assert!(
                        download(device, &on.output)
                            .iter()
                            .zip(&initial)
                            .all(|(a, b)| a.bits() == b.bits())
                    );
                }
            }
            println!(
                "  {layout}, {}: onto nonzero outputs within the bounds; rows without entries, \
                 leaves without points and calls without entries change nothing",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

/// Determinism and invariance, bit for bit: repeated launches give the same bits (the
/// per-coefficient sums of P2M and P2L included); a row split into one call per entry
/// gives the bits of one call (P2L, M2P).
fn invariance_and_determinism(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_7007);
        let p = 8;
        let same = |a: &[T], b: &[T]| a.iter().zip(b).all(|(x, y)| x.bits() == y.bits());
        for layout in layouts(device) {
            for irregular in [false, true] {
                let case = if irregular {
                    p2l_case::<T>(&mut rng, 9, 12)
                } else {
                    p2m_case::<T>(&mut rng, 9)
                };
                let whole = case.run(device, layout, p, irregular, None);
                assert!(
                    same(&case.run(device, layout, p, irregular, None), &whole),
                    "{layout}: repeated"
                );
                // One call per entry, accumulating in the same buffer.
                let zeros = vec![t::<T>(0.0); whole.len()];
                let mut on = case.upload(device, &case.rows, &zeros);
                let longest = case.rows.iter().map(Vec::len).max().unwrap();
                for k in 0..longest {
                    let rows: Vec<Vec<u32>> = case
                        .rows
                        .iter()
                        .map(|row| row.get(k).map(|&j| vec![j]).unwrap_or_default())
                        .collect();
                    let mut part = case.upload(device, &rows, &[]);
                    std::mem::swap(&mut part.out, &mut on.out);
                    launch_sources(device, &mut part, layout, p, irregular);
                    std::mem::swap(&mut part.out, &mut on.out);
                }
                assert!(
                    same(&download(device, &on.out), &whole),
                    "{layout}: split per entry"
                );
            }
            for irregular in [false, true] {
                let case = if irregular {
                    m2p_case::<T>(&mut rng, 9, p, 12)
                } else {
                    l2p_case::<T>(&mut rng, 9, p)
                };
                for gradients in [false, true] {
                    let whole = case.run(device, layout, p, (irregular, gradients), None);
                    assert!(
                        same(
                            &case.run(device, layout, p, (irregular, gradients), None),
                            &whole
                        ),
                        "{layout}: repeated"
                    );
                    let zeros = case.output_store(gradients, || t::<T>(0.0));
                    let mut on = case.upload(device, irregular, &case.rows, &zeros);
                    let longest = case.rows.iter().map(Vec::len).max().unwrap();
                    for k in 0..longest {
                        let rows: Vec<Vec<u32>> = case
                            .rows
                            .iter()
                            .map(|row| row.get(k).map(|&j| vec![j]).unwrap_or_default())
                            .collect();
                        let mut part = case.upload(device, irregular, &rows, &[]);
                        std::mem::swap(&mut part.output, &mut on.output);
                        launch_targets(device, &mut part, layout, p, (irregular, gradients));
                        std::mem::swap(&mut part.output, &mut on.output);
                    }
                    assert!(
                        same(&download(device, &on.output), &whole),
                        "{layout}: split per entry"
                    );
                }
            }
            println!(
                "  {layout}, {}: repeated launches and rows split per entry bit for bit (P2M, \
                 P2L, L2P, M2P, with and without gradients)",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

/// The layouts against each other: how many values they give bit for bit alike;
/// reported, not part of the contract (contraction may differ between kernels).
fn layouts_agree(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0x7e57_7008);
        let p = 8;
        let all = layouts(device);
        let source = p2l_case::<T>(&mut rng, 9, 12);
        let target = m2p_case::<T>(&mut rng, 9, p, 12);
        let outputs: Vec<(Vec<T>, Vec<T>)> = all
            .iter()
            .map(|&layout| {
                (
                    source.run(device, layout, p, true, None),
                    target.run(device, layout, p, (true, true), None),
                )
            })
            .collect();
        for (layout, (a, b)) in all.iter().zip(&outputs).skip(1) {
            let count = |x: &[T], y: &[T]| {
                x.iter()
                    .zip(y)
                    .filter(|(u, v)| u.bits() == v.bits())
                    .count()
            };
            println!(
                "  {}, p = {p}: {layout} equals {} in {} of {} P2L and {} of {} M2P values",
                T::FLOAT,
                all[0],
                count(a, &outputs[0].0),
                a.len(),
                count(b, &outputs[0].1),
                b.len()
            );
        }
    }
    each_precision!(device, body);
}

/// The p = 20 sweep in f64 (CONVENTIONS §3.9; the f64 bound up to p = 20): harmonics
/// and every operator on every layout, ignored by default (compile time).
fn degree_20_sweep(device: &mut Device) {
    if !device.supports(Precision::F64) {
        println!("  f64: not supported by this device, not run");
        return;
    }
    harmonics_at::<f64>(device, &[20]);
    p2m_and_l2p_at::<f64>(device, &[20]);
    p2l_at::<f64>(device, &[20]);
    m2p_at::<f64>(device, &[20]);
}

tests_on!(
    cpu: harmonics_match_nd_fmm_math,
    p2m_and_l2p,
    p2l_x_list,
    m2p_w_list,
    frames,
    accumulation_and_empty_calls,
    invariance_and_determinism,
    layouts_agree,
);
tests_on!(
    metal: harmonics_match_nd_fmm_math,
    p2m_and_l2p,
    p2l_x_list,
    m2p_w_list,
    frames,
    accumulation_and_empty_calls,
    invariance_and_determinism,
    layouts_agree,
);

/// The p = 20 sweep on the CPU runtime: ignored (compile time; fmm-kernels/CLAUDE.md, test
/// budget).
#[cfg(feature = "cpu")]
#[test]
#[ignore = "p = 20 in f64: long kernel compilation on the CPU runtime"]
fn degree_20_sweep_on_the_cpu_runtime() {
    crate::common::run(
        BackendKind::Cpu,
        "leaf::degree_20_sweep_on_the_cpu_runtime",
        degree_20_sweep,
    );
}

const _: fn(&mut Device) = degree_20_sweep;
