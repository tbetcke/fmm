//! The device P2P (Phase 4 T6, C4.2) as the device operator runs it, against
//! `nd_fmm_ref::p2p` at the absolute frames of CONVENTIONS §3.12, as T8 of Phase 3
//! checks the host P2P (`leaf.rs`): a random interior leaf t on levels 2, 9 and 16 with
//! itself (the targets are the sources plus as many other points), each of its 26
//! neighbours, every touching leaf one level coarser and, below level 16, every touching
//! leaf one level finer, points on the dyadic grid of `common` in the dyadic domain, with
//! gradients and without, in f64 and f32, with the layouts the device operator offers on
//! the backend (its default first). Every pair is one row of one launch per level, as
//! `nd_fmm_kernels::p2p::p2p` takes a level's near view.
//!
//! Error measure (`common`): values relative to the magnitudes of their terms, the
//! gradient as a Euclidean norm, in the leaf units of §3.13 (φ̂ = r_t φ, ĝ = r_t² ∇φ).
//! Tolerance 1e-13 in f64 (C3.1) and 1e-5 in f32 (the f32 operator bound of
//! docs/phase4/README.md, "Accuracy measures").
//!
//! Frames: the frames the kernel forms from the leaves' integer indices
//! (`nd_fmm_kernels::p2p::near_frames`) equal `geometry::relative_frame` bit for bit, in
//! f32 and f64, for every pair above and for random key pairs over every combination of
//! levels 0–16, the opposite corners of level 16 included (device-path.md §6.1).
//!
//! The CPU runtime runs with the `cpu` feature; Metal (f32) with `metal`, ignored, by
//! hand outside the macOS sandbox; CUDA (f32 and f64) with `cuda`, ignored, by hand on
//! locust (Phase 4S T4). Each test prints the device and the backends it ran.

use nd_fmm_exec::geometry::relative_frame;
use nd_fmm_kernels::p2p::{P2pInputs, P2pLayout, near_frames, p2p};
use nd_fmm_kernels::view::{IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, Precision};
use nd_fmm_math::RealScalar;
use nd_fmm_ref::p2p as reference;
use nd_octree::{MortonKey, morton};

use crate::common::{
    LEVELS, SplitMix64, dyadic_domain, grid_points, key_radius, neighbours, norm, p2p_terms, sub,
    touches,
};

/// Points per box.
const POINTS: usize = 6;

/// The precisions of these tests.
trait Real: DeviceFloat + RealScalar {}

/// `x` widened to f64.
fn w<T: Real>(x: T) -> f64 {
    RealScalar::to_f64(x)
}

impl Real for f32 {}

impl Real for f64 {}

/// The P2P sources of the interior leaf `target`: itself, its 26 neighbours, the
/// touching leaves one level coarser and, below level 16, one level finer (as `leaf.rs`).
fn near_sources(target: MortonKey) -> Vec<MortonKey> {
    let level = morton::level(target);
    let mut sources = vec![target];
    let same = neighbours(target);
    assert_eq!(same.len(), 26);
    sources.extend(&same);
    let parent = morton::parent(target).unwrap();
    sources.extend(
        neighbours(parent)
            .into_iter()
            .filter(|&s| touches(s, target)),
    );
    if level < 16 {
        sources.extend(
            same.iter()
                .flat_map(|&n| morton::children(n).unwrap())
                .filter(|&s| touches(s, target)),
        );
    }
    sources
}

/// One source–target leaf pair: keys, absolute and leaf-scaled points, charges.
struct Pair {
    source: MortonKey,
    target: MortonKey,
    x: Vec<[f64; 3]>,
    u_t: Vec<[f64; 3]>,
    y: Vec<[f64; 3]>,
    u_s: Vec<[f64; 3]>,
    q: Vec<f64>,
}

/// The pairs of one level (module documentation).
fn pairs(level: usize, rng: &mut SplitMix64) -> Vec<Pair> {
    let domain = dyadic_domain();
    let target = rng.interior_key(level);
    near_sources(target)
        .into_iter()
        .map(|source| {
            let (y, u_s) = grid_points(source, &domain, POINTS, rng);
            let q = rng.charges(POINTS);
            let (x, u_t) = if source == target {
                let (mut x, mut u) = grid_points(target, &domain, POINTS, rng);
                x.splice(0..0, y.iter().copied());
                u.splice(0..0, u_s.iter().copied());
                (x, u)
            } else {
                grid_points(target, &domain, POINTS, rng)
            };
            Pair {
                source,
                target,
                x,
                u_t,
                y,
                u_s,
                q,
            }
        })
        .collect()
}

/// A key's level and index as the device holds them.
fn key_index(key: MortonKey) -> (u32, [u32; 3]) {
    let (level, index) = morton::decode(key);
    (level as u32, index.map(|c| c as u32))
}

/// The pairs as one level's call: row r the target leaf r of pair r, its entry the
/// row's own leaf for the self pair and otherwise a leaf of the source after the rows.
/// Returns the output chunk of every row and the frames of every entry.
fn launch<T: Real>(
    device: &mut Device,
    layout: P2pLayout,
    gradients: bool,
    pairs: &[Pair],
) -> (Vec<Vec<T>>, Vec<[T; 4]>) {
    let narrow = |v: &[[f64; 3]]| -> Vec<T> {
        v.iter()
            .flatten()
            .map(|&c| <T as RealScalar>::from_f64(c))
            .collect()
    };
    let rows = pairs.len();
    let mut keys = Vec::new();
    let mut entries = Vec::new();
    let (mut source_counts, mut target_counts) = (Vec::new(), Vec::new());
    let (mut sources, mut input) = (Vec::<T>::new(), Vec::<T>::new());
    let self_sources = |pair: &Pair, sources: &mut Vec<T>| {
        sources.extend(narrow(&pair.u_s));
        sources.extend(pair.q.iter().map(|&q| <T as RealScalar>::from_f64(q)));
    };
    for (r, pair) in pairs.iter().enumerate() {
        keys.push(key_index(pair.target));
        target_counts.push(pair.x.len() as u32);
        input.extend(narrow(&pair.u_t));
        if pair.source == pair.target {
            source_counts.push(POINTS as u32);
            self_sources(pair, &mut sources);
            entries.push(r as u32);
        } else {
            source_counts.push(0);
            entries.push((rows + entries.iter().filter(|&&e| e as usize >= rows).count()) as u32);
        }
    }
    for pair in pairs.iter().filter(|p| p.source != p.target) {
        keys.push(key_index(pair.source));
        source_counts.push(POINTS as u32);
        target_counts.push(0);
        self_sources(pair, &mut sources);
    }
    let offsets = |counts: &[u32]| -> Vec<u32> {
        std::iter::once(0)
            .chain(counts.iter().scan(0, |total, &n| {
                *total += n;
                Some(*total)
            }))
            .collect()
    };
    let row_offsets: Vec<u32> = (0..=rows as u32).collect();
    let near = IndexView::upload(device, &row_offsets, &entries, keys.len()).unwrap();
    let leaves = LeafCoordinates::upload(device, &keys).unwrap();
    let source_offsets = PointOffsets::upload(device, &offsets(&source_counts)).unwrap();
    let target_offsets = PointOffsets::upload(device, &offsets(&target_counts)).unwrap();
    let sources = device.upload(&sources).unwrap();
    let target_input = device.upload(&input).unwrap();
    let per_point = if gradients { 4 } else { 1 };
    let mut output = device.alloc::<T>(per_point * input.len() / 3).unwrap();
    let inputs = P2pInputs {
        near: &near,
        first_leaf: 0,
        leaves: &leaves,
        source_offsets: &source_offsets,
        sources: sources.as_slice(),
        target_offsets: &target_offsets,
        target_input: target_input.as_slice(),
    };
    p2p(device, layout, gradients, &inputs, output.as_slice_mut()).unwrap();
    let zero = <T as RealScalar>::from_f64(0.0);
    let mut all = vec![zero; output.len()];
    device.download(output.as_slice(), &mut all).unwrap();
    let mut frames = device.alloc::<T>(4 * rows).unwrap();
    near_frames(device, &near, 0, &leaves, frames.as_slice_mut()).unwrap();
    let mut flat = vec![zero; frames.len()];
    device.download(frames.as_slice(), &mut flat).unwrap();
    let mut chunks = Vec::with_capacity(rows);
    let mut start = 0;
    for pair in pairs {
        let len = per_point * pair.x.len();
        chunks.push(all[start..start + len].to_vec());
        start += len;
    }
    (chunks, flat.as_chunks::<4>().0.to_vec())
}

/// The tolerance in `T`.
fn tolerance<T: Real>() -> f64 {
    if T::FLOAT == Precision::F64 {
        1e-13
    } else {
        1e-5
    }
}

/// The P2P of every level against the reference, and its frames bit for bit; returns
/// the worst potential and gradient errors.
fn near_pairs<T: Real>(device: &mut Device, layout: P2pLayout) -> (f64, f64) {
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7e60);
    let (mut worst_potential, mut worst_gradient) = (0.0f64, 0.0f64);
    for level in LEVELS {
        let pairs = pairs(level, &mut rng);
        for gradients in [false, true] {
            let (chunks, frames) = launch::<T>(device, layout, gradients, &pairs);
            for (pair, (chunk, frame)) in pairs.iter().zip(chunks.iter().zip(&frames)) {
                let context = format!(
                    "{layout}, {}, level {level}, source {:?} target {:?}, gradients {gradients}",
                    T::FLOAT,
                    morton::decode(pair.source),
                    morton::decode(pair.target)
                );
                // Frames bit for bit.
                let want: [T; 4] = if pair.source == pair.target {
                    [0.0, 0.0, 0.0, 1.0].map(<T as RealScalar>::from_f64)
                } else {
                    let f = relative_frame::<T>(pair.source, pair.target);
                    [f.centre[0], f.centre[1], f.centre[2], f.radius]
                };
                for k in 0..4 {
                    assert_eq!(
                        w(frame[k]).to_bits(),
                        w(want[k]).to_bits(),
                        "{context}: frame component {k}"
                    );
                }
                // Values against the reference at the absolute points, in leaf units.
                let q: Vec<f64> = pair
                    .q
                    .iter()
                    .map(|&q| w(<T as RealScalar>::from_f64(q)))
                    .collect();
                let n = pair.x.len();
                let (mut phi, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n]);
                reference::p2p(&pair.y, &q, &pair.x, &mut phi, Some(&mut grad));
                let r_t = key_radius(pair.target, &domain);
                for j in 0..n {
                    let (terms_phi, terms_grad) = p2p_terms(&pair.y, &q, pair.x[j]);
                    let got = w(chunk[j]);
                    let e = (got - r_t * phi[j]).abs() / (r_t * terms_phi);
                    assert!(
                        e <= tolerance::<T>(),
                        "{context}: target {j}: φ error {e:e}"
                    );
                    worst_potential = worst_potential.max(e);
                    if gradients {
                        let g: [f64; 3] = std::array::from_fn(|k| w(chunk[n + 3 * j + k]));
                        let want = grad[j].map(|v| r_t * r_t * v);
                        let e = norm(sub(g, want)) / (r_t * r_t * terms_grad);
                        assert!(
                            e <= tolerance::<T>(),
                            "{context}: target {j}: ∇φ error {e:e}"
                        );
                        worst_gradient = worst_gradient.max(e);
                    }
                }
            }
        }
    }
    (worst_potential, worst_gradient)
}

/// Random key pairs over every combination of levels 0–16, and the opposite corners of
/// level 16: the device frames equal `relative_frame` bit for bit.
fn random_frames<T: Real>(device: &mut Device) -> usize {
    let mut rng = SplitMix64::new(0x7e61);
    let mut pairs = Vec::new();
    for lt in 0..=16 {
        for ls in 0..=16 {
            for _ in 0..2 {
                pairs.push((rng.key(ls), rng.key(lt)));
            }
        }
    }
    pairs.push((
        morton::from_index_and_level([65_535; 3], 16),
        morton::from_index_and_level([0; 3], 16),
    ));
    // One row per pair: the target leaves first, then the sources.
    let rows = pairs.len();
    let mut keys: Vec<(u32, [u32; 3])> = pairs.iter().map(|p| key_index(p.1)).collect();
    keys.extend(pairs.iter().map(|p| key_index(p.0)));
    let row_offsets: Vec<u32> = (0..=rows as u32).collect();
    let entries: Vec<u32> = (rows as u32..2 * rows as u32).collect();
    let near = IndexView::upload(device, &row_offsets, &entries, keys.len()).unwrap();
    let leaves = LeafCoordinates::upload(device, &keys).unwrap();
    let mut frames = device.alloc::<T>(4 * rows).unwrap();
    near_frames(device, &near, 0, &leaves, frames.as_slice_mut()).unwrap();
    let mut got = vec![<T as RealScalar>::from_f64(0.0); frames.len()];
    device.download(frames.as_slice(), &mut got).unwrap();
    for (e, &(s, t)) in pairs.iter().enumerate() {
        let f = relative_frame::<T>(s, t);
        let want = [f.centre[0], f.centre[1], f.centre[2], f.radius];
        for k in 0..4 {
            assert_eq!(
                w(got[4 * e + k]).to_bits(),
                w(want[k]).to_bits(),
                "{}: pair {e} ({:?} from {:?}), component {k}",
                T::FLOAT,
                morton::decode(s),
                morton::decode(t)
            );
        }
    }
    rows
}

/// The layouts the device operator offers on `device`: the default first.
fn layouts(device: &Device) -> Vec<P2pLayout> {
    let default = P2pLayout::default_for(device.info());
    let mut layouts = vec![default];
    if device.backend() != BackendKind::Cpu {
        layouts.push(P2pLayout::Plane { planes: 2 });
    }
    layouts
}

/// Every check on the device of `kind`, in f32 and, where it runs, f64.
fn run(kind: BackendKind) {
    let mut device = Device::open(kind).unwrap_or_else(|error| panic!("{kind}: {error}"));
    eprintln!("device P2P against p2p::p2p: on {}", device.info());
    for layout in layouts(&device) {
        let (p, g) = near_pairs::<f32>(&mut device, layout);
        eprintln!("  {layout}, f32: worst φ {p:.2e}, ∇φ {g:.2e} (terms; bound 1e-5)");
        if device.supports(Precision::F64) {
            let (p, g) = near_pairs::<f64>(&mut device, layout);
            eprintln!("  {layout}, f64: worst φ {p:.2e}, ∇φ {g:.2e} (terms; bound 1e-13)");
        }
    }
    let rows = random_frames::<f32>(&mut device);
    if device.supports(Precision::F64) {
        random_frames::<f64>(&mut device);
    }
    eprintln!(
        "  frames: every near pair and {rows} random key pairs bit for bit against \
         relative_frame"
    );
    let not_run: Vec<String> = BackendKind::ALL
        .into_iter()
        .filter(|&k| k != kind)
        .map(|k| match (k, k.is_compiled()) {
            (_, false) => format!("{k} (not compiled)"),
            (BackendKind::Cpu, true) => format!("{k} (its own test)"),
            _ => format!("{k} (its own ignored test)"),
        })
        .collect();
    eprintln!("backends run: {kind}; not run: {}", not_run.join(", "));
}

#[cfg(feature = "cpu")]
#[test]
fn device_p2p_equals_the_reference_on_the_cpu_runtime() {
    run(BackendKind::Cpu);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Metal: run by hand, outside the sandbox"]
fn device_p2p_equals_the_reference_on_metal() {
    run(BackendKind::Metal);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "CUDA: run by hand on locust"]
fn device_p2p_equals_the_reference_on_cuda() {
    run(BackendKind::Cuda);
}
