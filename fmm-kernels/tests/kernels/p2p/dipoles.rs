//! The device P2P with dipole sources (Phase 6 T8d, C6.4) on every layout the backend
//! runs, in f32 and, where the device supports it, f64, potential only and with gradients,
//! for `SourceKind::Dipoles` and `ChargesAndDipoles`.
//!
//! Error measures (the dipole contract of CONVENTIONS §3.13, "Device kernels", the host
//! kernel's of docs/design/simd-p2p.md §4.8):
//! - **pair terms**, against `nd_fmm_ref::p2p::p2p_dipoles` in the same precision, on the
//!   mapped inputs of the host (ŷ = fl(ĉ + r̂ u_s), d_t = r̂ d̂): the potential relative to
//!   |d_t| / r², within 32 u_T; each gradient component relative to |d_t| / r³, within
//!   176 u_T; in f32 over the dipole range 2^(−80 + 2e/3) ≤ r² ≤ 2⁷ (|d_t,k| ≤ 2^e), and
//!   for d_t = 0 or |d_t| r ≥ 2⁻⁹⁶ (subnormals flush on Metal);
//! - **sums**, against `direct_sum` and `direct_sum_dipoles` in f64 on the inputs as
//!   rounded to T and mapped as the host maps them: per target the error relative to the
//!   sum of its term magnitudes (Σ|q|/r + Σ|d_t|/r² for φ, Σ|q|/r² + Σ|d_t|/r³ per
//!   component of ∇φ), within 1e-5 (f32) or 1e-13 (f64), or within twice the error of the
//!   reference (`p2p`, `p2p_dipoles`) on the same inputs where that is larger;
//! - **bits**: invariance (a row split per leaf, permuted target leaves, repeated
//!   launches), and the order: a launch with charges and dipoles equals, leaf by leaf, a
//!   launch of the charges followed by one of the moments.

use nd_fmm_kernels::p2p::{P2pInputs, P2pLayout, p2p};
use nd_fmm_kernels::{Device, Precision, SourceKind};
use nd_fmm_ref::p2p::{direct_sum, direct_sum_dipoles, p2p_dipoles};

use super::{
    Leaf, OnDevice, Problem, Real, Worst, assert_same_bits, charge, coincident_problem, fmm_rows,
    frame, layouts, sizes, split, t, upload_rows, w,
};
use crate::common::{Rng, tests_on};

/// The two source kinds with dipoles.
const KINDS: [SourceKind; 2] = [SourceKind::Dipoles, SourceKind::ChargesAndDipoles];

/// The sum tolerance of the dipole kernels: 1e-5 in f32, 1e-13 in f64 (the host kernel's,
/// `nd-fmm-simd`'s `tests/p2p/dipoles.rs`).
fn sum_tolerance<T: Real>() -> f64 {
    if T::FLOAT == Precision::F32 {
        1e-5
    } else {
        1e-13
    }
}

/// The smallest r² of the f32 dipole range for |d_t,k| ≤ 2 (moments in [−1, 1)³ mapped by
/// r̂ ≤ 2): 2^(−80 + 2/3), rounded up to 2⁻⁷⁹; 0 in f64.
fn dipole_floor<T: Real>() -> f64 {
    if T::FLOAT == Precision::F32 {
        2f64.powi(-79)
    } else {
        0.0
    }
}

/// A moment uniform in [−1, 1)³.
fn moment<T: Real>(rng: &mut Rng) -> [T; 3] {
    [0; 3].map(|_| charge::<T>(rng))
}

/// Gives every source of `problem` a moment uniform in [−1, 1)³.
fn with_moments<T: Real>(rng: &mut Rng, mut problem: Problem<T>) -> Problem<T> {
    for leaf in &mut problem.leaves {
        leaf.moments = (0..leaf.sources.len()).map(|_| moment::<T>(rng)).collect();
    }
    problem
}

/// The source store of `problem` with the strengths of `kind`: per leaf its triples, its
/// charges (with charges) and its moments (with dipoles), CONVENTIONS §3.13.
fn store<T: Real>(problem: &Problem<T>, kind: SourceKind) -> Vec<T> {
    let mut store = Vec::new();
    for leaf in &problem.leaves {
        store.extend(leaf.sources.iter().flatten());
        if kind.has_charges() {
            store.extend(&leaf.charges);
        }
        if kind.has_dipoles() {
            store.extend(leaf.moments.iter().flatten());
        }
    }
    store
}

/// Uploads `problem` with the near view `rows`, the target output `output` and the
/// source store of `kind`.
fn upload_kind<T: Real>(
    device: &mut Device,
    problem: &Problem<T>,
    rows: &[Vec<u32>],
    output: &[T],
    kind: SourceKind,
) -> OnDevice<T> {
    let mut on = upload_rows(device, problem, rows, output);
    on.sources = device.upload(&store(problem, kind)).unwrap();
    on
}

/// Launches the P2P of `on`, whose store holds `kind`, adding into its output.
fn launch_kind<T: Real>(
    device: &mut Device,
    on: &mut OnDevice<T>,
    (layout, gradients, kind): (P2pLayout, bool, SourceKind),
) {
    let inputs = P2pInputs {
        near: &on.near,
        first_leaf: on.first,
        leaves: &on.leaves,
        source_offsets: &on.source_offsets,
        sources: on.sources.as_slice(),
        target_offsets: &on.target_offsets,
        target_input: on.target_input.as_slice(),
        vectors: 1,
        source_kind: kind,
    };
    p2p(device, layout, gradients, &inputs, on.output.as_slice_mut()).unwrap();
}

/// Runs `problem` with the strengths of `kind` from a zero output and returns the output
/// store.
fn run_kind<T: Real>(
    device: &mut Device,
    problem: &Problem<T>,
    (layout, gradients, kind): (P2pLayout, bool, SourceKind),
) -> Vec<T> {
    let zeros = problem.output_store(gradients, |_, _| t::<T>(0.0));
    let mut on = upload_kind(device, problem, &problem.rows, &zeros, kind);
    launch_kind(device, &mut on, (layout, gradients, kind));
    super::download(device, &on)
}

/// The sources of row r as the host maps them, leaf by leaf in row order: (ŷ, q, d_t) of
/// each leaf, ŷ = fl(ĉ + r̂ u_s) and d_t = r̂ d̂ (exact), u_s and d̂ themselves for the
/// leaf itself.
type Mapped<T> = (Vec<[T; 3]>, Vec<T>, Vec<[T; 3]>);

fn row_leaves<T: Real>(problem: &Problem<T>, r: usize) -> Vec<Mapped<T>> {
    let target = problem.first + r;
    problem.rows[r]
        .iter()
        .map(|&j| {
            let leaf = &problem.leaves[j as usize];
            if j as usize == target {
                (
                    leaf.sources.clone(),
                    leaf.charges.clone(),
                    leaf.moments.clone(),
                )
            } else {
                let (c, ratio) = frame(leaf.key, problem.leaves[target].key);
                let (c, ratio) = (c.map(t::<T>), t::<T>(ratio));
                (
                    leaf.sources
                        .iter()
                        .map(|u| std::array::from_fn(|k| c[k] + ratio * u[k]))
                        .collect(),
                    leaf.charges.clone(),
                    leaf.moments.iter().map(|d| d.map(|v| ratio * v)).collect(),
                )
            }
        })
        .collect()
}

/// The Euclidean norm of a moment in f64.
fn norm<T: Real>(d: &[T; 3]) -> f64 {
    d.iter().map(|&v| w(v).powi(2)).sum::<f64>().sqrt()
}

/// Checks every row of `problem` in `output` (computed from `initial`, zero if `None`)
/// against the f64 direct sums of `kind` within the sum contract; returns the worst errors.
fn check_sums<T: Real>(
    what: &str,
    problem: &Problem<T>,
    (initial, output): (Option<&[T]>, &[T]),
    (gradients, kind): (bool, SourceKind),
) -> Worst {
    let mut worst = Worst::default();
    let up = |v: &[[T; 3]]| -> Vec<[f64; 3]> { v.iter().map(|p| p.map(w::<T>)).collect() };
    for r in 0..problem.rows.len() {
        let j = problem.first + r;
        let targets = &problem.leaves[j].targets;
        if targets.is_empty() {
            continue;
        }
        let n = targets.len();
        let range = problem.output_range(j, gradients);
        let (phi, grad) = split(&output[range.clone()], gradients);
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
        let x64 = up(targets);
        let mut ophi: Vec<f64> = phi0.iter().map(|&v| w(v)).collect();
        let mut ograd: Vec<[f64; 3]> = grad0.iter().map(|g| g.map(w::<T>)).collect();
        let (mut rphi, mut rgrad) = (phi0.clone(), grad0.clone());
        let mut terms: Vec<([f64; 3], f64, f64)> = Vec::new();
        for (y, q, d) in row_leaves(problem, r) {
            let y64 = up(&y);
            if kind.has_charges() {
                let q64: Vec<f64> = q.iter().map(|&v| w(v)).collect();
                direct_sum(&y64, &q64, &x64, &mut ophi, Some(&mut ograd));
                nd_fmm_ref::p2p::p2p(&y, &q, targets, &mut rphi, Some(&mut rgrad));
            }
            if kind.has_dipoles() {
                direct_sum_dipoles(&y64, &up(&d), &x64, &mut ophi, Some(&mut ograd));
                p2p_dipoles(&y, &d, targets, &mut rphi, Some(&mut rgrad));
            }
            for (k, &yk) in y64.iter().enumerate() {
                let qk = if kind.has_charges() {
                    w(q[k]).abs()
                } else {
                    0.0
                };
                let dk = if kind.has_dipoles() { norm(&d[k]) } else { 0.0 };
                terms.push((yk, qk, dk));
            }
        }
        let (mut row_worst, mut row_reference) = ([0.0f64; 2], [0.0f64; 2]);
        for i in 0..n {
            let mut mp = w(phi0[i]).abs();
            let mut mg = grad0[i].iter().fold(0.0f64, |m, &v| m.max(w(v).abs()));
            let mut closest = f64::INFINITY;
            for &(yk, qk, dk) in &terms {
                let r2: f64 = (0..3).map(|c| (x64[i][c] - yk[c]).powi(2)).sum();
                if r2 > 0.0 {
                    mp += qk / r2.sqrt() + dk / r2;
                    mg += qk / r2 + dk / (r2 * r2.sqrt());
                    closest = closest.min(r2);
                }
            }
            if closest < dipole_floor::<T>() {
                worst.outside_gradient_range += 1;
                continue;
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
            if gradients {
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
                "{what}: row {r} ({n} targets): {name} error {:e} exceeds {bound:e} (reference \
                 {:e})",
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

/// Pair terms: one source and one target per row, the source in the leaf itself or in a
/// same-level, coarser or finer neighbour (the moment mapped by r̂ = 1, 2 or ½), over
/// seeded separations across the kernel domain and moments from 2⁻⁴⁰ to 1 in magnitude
/// (every 64th zero, every 16th with zero components) and zero charges, within 32 u_T (φ)
/// and 176 u_T (∇φ) of `p2p_dipoles`.
fn dipole_pair_terms(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0xd1b0_0001);
        let count = 32_768;
        let uniform = |rng: &mut Rng| (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let target_key = (9u32, [101u32, 100, 100]);
        let sources = [
            target_key,
            (9, [102, 100, 100]),
            (8, [51, 50, 50]),
            (10, [201, 200, 201]),
        ];
        // Per row its target leaf and, for a neighbour, its source leaf.
        let mut pairs: Vec<(Leaf<T>, Option<Leaf<T>>)> = Vec::with_capacity(count);
        let mut outside = 0usize;
        while pairs.len() < count {
            let source_key = sources[pairs.len() % 4];
            let scale = 2f64.powf(-(rng.below(40) as f64));
            let x: [T; 3] = [0; 3].map(|_| t(scale * (2.0 * uniform(&mut rng) - 1.0)));
            let mut dir: [f64; 3] = [0; 3].map(|_| 2.0 * uniform(&mut rng) - 1.0);
            let length = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
            let len = 2f64.powf(-54.0 + 57.5 * uniform(&mut rng));
            dir = dir.map(|v| v / length * len);
            // The source in its own leaf's frame, u_s = (ŷ − ĉ) / r̂ rounded to T; the host
            // maps it back (`row_leaves`).
            let (c, ratio) = frame(source_key, target_key);
            let u: [T; 3] = std::array::from_fn(|k| t((w(x[k]) + dir[k] - c[k]) / ratio));
            let magnitude = 2f64.powf(-(rng.below(41) as f64));
            let mut d = moment::<T>(&mut rng).map(|v| t(w(v) * magnitude));
            if pairs.len().is_multiple_of(64) {
                d = [t(0.0); 3];
            } else if pairs.len().is_multiple_of(16) {
                d[pairs.len() / 16 % 3] = t(0.0);
            }
            let mut source = Leaf::new(source_key);
            source.sources = vec![u];
            // No charge: the terms are the moments' (the charge stage still runs for
            // `ChargesAndDipoles`, adding zero).
            source.charges = vec![t(0.0)];
            source.moments = vec![d];
            let mut target = Leaf::new(target_key);
            target.targets = vec![x];
            let pair = if source_key == target_key {
                target.sources = source.sources;
                target.charges = source.charges;
                target.moments = source.moments;
                (target, None)
            } else {
                (target, Some(source))
            };
            // The mapped pair as the host forms it.
            let one = Problem {
                leaves: std::iter::once(pair.0.clone())
                    .chain(pair.1.clone())
                    .collect(),
                first: 0,
                rows: vec![if pair.1.is_some() {
                    vec![0, 1]
                } else {
                    vec![0]
                }],
            };
            let (y, _, dt) = row_leaves(&one, 0).pop().unwrap();
            let r2: f64 = (0..3).map(|k| (w(x[k]) - w(y[0][k])).powi(2)).sum();
            if !(2f64.powi(-108)..=2f64.powi(7)).contains(&r2) {
                continue;
            }
            if r2 < dipole_floor::<T>() {
                outside += 1;
                continue;
            }
            let dn = norm(&dt[0]);
            if dn != 0.0 && dn * r2.sqrt() < 2f64.powi(-96) {
                continue;
            }
            pairs.push(pair);
        }
        // The target leaves first (rows are consecutive leaves), then the neighbours.
        let mut leaves = Vec::with_capacity(2 * count);
        let mut neighbours = Vec::new();
        let mut rows = Vec::with_capacity(count);
        for (r, (target, source)) in pairs.into_iter().enumerate() {
            leaves.push(target);
            match source {
                Some(source) => {
                    rows.push(vec![r as u32, (count + neighbours.len()) as u32]);
                    neighbours.push(source);
                }
                None => rows.push(vec![r as u32]),
            }
        }
        leaves.extend(neighbours);
        let problem = Problem {
            leaves,
            first: 0,
            rows,
        };
        for layout in layouts(device) {
            for kind in KINDS {
                let output = run_kind(device, &problem, (layout, true, kind));
                let (mut ep, mut eg) = (0.0f64, 0.0f64);
                for r in 0..count {
                    let x = problem.leaves[r].targets[0];
                    let mapped = row_leaves(&problem, r);
                    let (y, q, d) = mapped.last().unwrap();
                    let (mut rphi, mut rg) = ([t::<T>(0.0)], [[t::<T>(0.0); 3]]);
                    if kind.has_charges() {
                        nd_fmm_ref::p2p::p2p(y, q, &[x], &mut rphi, Some(&mut rg));
                    }
                    p2p_dipoles(y, d, &[x], &mut rphi, Some(&mut rg));
                    let (phi, grad) = split(&output[problem.output_range(r, true)], true);
                    let (phi, grad) = (phi[0], grad[0]);
                    let r2: f64 = (0..3).map(|k| (w(x[k]) - w(y[0][k])).powi(2)).sum();
                    let dn = norm(&d[0]);
                    if dn == 0.0 {
                        assert!(
                            w(phi) == 0.0 && grad.iter().all(|&g| w(g) == 0.0),
                            "{layout}, {kind}: d = 0 gives a nonzero term"
                        );
                        continue;
                    }
                    let e = (w(phi) - w(rphi[0])).abs() / (dn / r2) / T::U;
                    ep = ep.max(if e.is_nan() { f64::INFINITY } else { e });
                    for k in 0..3 {
                        let e = (w(grad[k]) - w(rg[0][k])).abs() / (dn / (r2 * r2.sqrt())) / T::U;
                        eg = eg.max(if e.is_nan() { f64::INFINITY } else { e });
                    }
                }
                println!(
                    "  {layout}, {}, {kind}: {count} pairs ({outside} drawn below the f32 dipole \
                     range, redrawn), φ {ep:.2} u_T, ∇φ {eg:.2} u_T (bounds 32, 176)",
                    T::FLOAT
                );
                assert!(ep <= 32.0, "{layout}, {kind}: potential term {ep} u_T");
                assert!(eg <= 176.0, "{layout}, {kind}: gradient term {eg} u_T");
            }
        }
    }
    each_precision!(device, body);
}

/// Sums: FMM-shaped rows (a target leaf, its neighbours on its own, the coarser and the
/// finer level, mapped as `LaplaceOperator` maps them), n_t from 0 to 3 tiles + 1, up to
/// 4,096 sources per row, with moments (and charges), against the f64 direct sums.
fn dipole_sums(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        for layout in layouts(device) {
            let tile = layout.tile(device.info(), T::FLOAT);
            let mut rng = Rng::new(0xd1b0_0002 ^ tile as u64);
            let problem = fmm_rows::<T>(&mut rng, &sizes(tile), 4096);
            let problem = with_moments(&mut rng, problem);
            for kind in KINDS {
                for gradients in [false, true] {
                    let output = run_kind(device, &problem, (layout, gradients, kind));
                    let what = format!("{layout}, {kind}");
                    let worst = check_sums(&what, &problem, (None, &output), (gradients, kind));
                    worst.print(&format!("{what}, {}, gradients {gradients}", T::FLOAT));
                }
            }
        }
    }
    each_precision!(device, body);
}

/// Coincident points with moments: the rows of the charge test `coincident` (targets equal
/// to the sources of their leaf, duplicated sources, mapped neighbour sources that equal or
/// round onto a target, the adversarial points of levels 2, 9 and 16), every source with a
/// moment. Every output is finite and equals the sum over the non-coincident pairs within
/// the sum contract.
fn dipole_coincident(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0xd1b0_0003);
        let (problem, coincident_pairs) = coincident_problem::<T>(&mut rng);
        let problem = with_moments(&mut rng, problem);
        for layout in layouts(device) {
            for kind in KINDS {
                for gradients in [false, true] {
                    let output = run_kind(device, &problem, (layout, gradients, kind));
                    let what = format!("{layout}, {kind}");
                    let worst = check_sums(&what, &problem, (None, &output), (gradients, kind));
                    worst.print(&format!(
                        "{what}, {}, gradients {gradients}, {coincident_pairs} coincident pairs",
                        T::FLOAT
                    ));
                }
            }
        }
    }
    each_precision!(device, body);
}

/// Domain ends with moments: pairs at r² = 2⁻¹⁰⁸ and 3 · 2⁻¹⁰⁸ (f64), at the f32 dipole
/// range 2⁻⁸⁰ (|d| ≤ 1) and 2⁻⁷⁶ (|d_k| ≤ 2⁶), near 2⁷, and the smallest moment of the
/// flushing rule (|d| r = 2⁻⁹⁰), finite and within the term bounds.
fn dipole_domain_ends(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let f64_only = T::FLOAT == Precision::F64;
        // (target, source, moment, f64 only).
        type Case = ([f64; 3], [f64; 3], [f64; 3], bool);
        let cases: Vec<Case> = vec![
            (
                [2f64.powi(-54), 0.0, 0.0],
                [0.0; 3],
                [0.75, -0.5, 0.25],
                true,
            ),
            (
                [-2f64.powi(-53), 0.5, -0.25],
                [0.0, 0.5, -0.25],
                [-1.0, 0.0, 0.5],
                true,
            ),
            (
                [2f64.powi(-40), 0.0, 0.0],
                [0.0; 3],
                [1.0, -1.0, 0.5],
                false,
            ),
            (
                [2f64.powi(-38), 0.0, 0.0],
                [0.0; 3],
                [64.0, -32.0, 64.0],
                false,
            ),
            (
                [2f64.powi(-40), 0.0, 0.0],
                [0.0; 3],
                [2f64.powi(-50), 0.0, -2f64.powi(-51)],
                false,
            ),
            ([1.0, 1.0, -1.0], [-5.5, -5.5, 5.5], [0.5, 0.5, -0.5], false),
            (
                [-1.0, 1.0, 1.0],
                [5.25, -5.0, -4.75],
                [-0.125, 1.0, 0.0],
                false,
            ),
        ];
        let cases: Vec<_> = cases
            .into_iter()
            .filter(|case| f64_only || !case.3)
            .collect();
        let mut leaves = Vec::new();
        for &(x, y, d, _) in &cases {
            let mut leaf = Leaf::new((3, [2, 2, 2]));
            leaf.targets = vec![x.map(t::<T>)];
            leaf.sources = vec![y.map(t::<T>)];
            leaf.charges = vec![t(0.5)];
            leaf.moments = vec![d.map(t::<T>)];
            leaves.push(leaf);
        }
        let problem = Problem {
            rows: (0..cases.len()).map(|r| vec![r as u32]).collect(),
            leaves,
            first: 0,
        };
        for layout in layouts(device) {
            let output = run_kind(device, &problem, (layout, true, SourceKind::Dipoles));
            for (r, &(x, y, d, _)) in cases.iter().enumerate() {
                let r2: f64 = (0..3).map(|k| (x[k] - y[k]).powi(2)).sum();
                let leaf = &problem.leaves[r];
                let (mut rphi, mut rg) = ([t::<T>(0.0)], [[t::<T>(0.0); 3]]);
                p2p_dipoles(
                    &leaf.sources,
                    &leaf.moments,
                    &leaf.targets,
                    &mut rphi,
                    Some(&mut rg),
                );
                let (phi, grad) = split(&output[problem.output_range(r, true)], true);
                let dn = d.iter().map(|v| v * v).sum::<f64>().sqrt();
                let phi = w(phi[0]);
                assert!(phi.is_finite(), "{layout}: r² = {r2:e}: φ = {phi}");
                let ep = (phi - w(rphi[0])).abs() / (dn / r2) / T::U;
                assert!(ep <= 32.0, "{layout}: r² = {r2:e}: φ {ep} u_T");
                for k in 0..3 {
                    let g = w(grad[0][k]);
                    assert!(g.is_finite(), "{layout}: r² = {r2:e}: ∇φ = {g}");
                    let eg = (g - w(rg[0][k])).abs() / (dn / (r2 * r2.sqrt())) / T::U;
                    assert!(eg <= 176.0, "{layout}: r² = {r2:e}: ∇φ {eg} u_T");
                }
            }
            println!(
                "  {layout}, {}: {} dipole pairs at the domain ends finite and within the term \
                 bounds",
                T::FLOAT,
                cases.len()
            );
        }
    }
    each_precision!(device, body);
}

/// Invariance, determinism and the order, bit for bit, for both kinds: a near row split
/// into one call per leaf gives the bits of one call; permuting the target leaves changes
/// no leaf's bits; repeated launches give the same bits; and charges and dipoles in one
/// launch equal, leaf by leaf, a launch of the charges followed by a launch of the moments
/// (each source leaf adds its charges first, then its moments).
fn dipole_invariance_and_order(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0xd1b0_0006);
        let problem = fmm_rows::<T>(&mut rng, &[3, 11, 0, 20, 8, 1], 1500);
        let problem = with_moments(&mut rng, problem);
        let longest = problem.rows.iter().map(Vec::len).max().unwrap();
        let per_leaf = |k: usize| -> Vec<Vec<u32>> {
            problem
                .rows
                .iter()
                .map(|row| row.get(k).map(|&j| vec![j]).unwrap_or_default())
                .collect()
        };
        for layout in layouts(device) {
            for gradients in [false, true] {
                for kind in KINDS {
                    let what = format!("{layout}, {kind}, gradients {gradients}");
                    let whole = run_kind(device, &problem, (layout, gradients, kind));
                    assert_same_bits(
                        &format!("{what}: repeated launch"),
                        &run_kind(device, &problem, (layout, gradients, kind)),
                        &whole,
                    );
                    // Split at every leaf boundary.
                    let zeros = problem.output_store(gradients, |_, _| t::<T>(0.0));
                    let mut on = upload_kind(device, &problem, &problem.rows, &zeros, kind);
                    for k in 0..longest {
                        let mut part = upload_kind(device, &problem, &per_leaf(k), &[], kind);
                        std::mem::swap(&mut part.output, &mut on.output);
                        launch_kind(device, &mut part, (layout, gradients, kind));
                        std::mem::swap(&mut part.output, &mut on.output);
                    }
                    assert_same_bits(
                        &format!("{what}: split at every leaf boundary"),
                        &super::download(device, &on),
                        &whole,
                    );
                    // Permute the target leaves: new row i is old row π(i).
                    let nrows = problem.rows.len();
                    let perm = rng.permutation(nrows);
                    let mut index: Vec<u32> = (0..problem.leaves.len() as u32).collect();
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
                    let output = run_kind(device, &permuted, (layout, gradients, kind));
                    for (i, &old) in perm.iter().enumerate() {
                        assert_same_bits(
                            &format!("{what}: target leaf {old} at position {i}"),
                            &output[permuted.output_range(i, gradients)],
                            &whole[problem.output_range(old as usize, gradients)],
                        );
                    }
                }
                // The order: leaf by leaf, the charges' launch, then the moments'.
                let both = run_kind(
                    device,
                    &problem,
                    (layout, gradients, SourceKind::ChargesAndDipoles),
                );
                let zeros = problem.output_store(gradients, |_, _| t::<T>(0.0));
                let mut on =
                    upload_kind(device, &problem, &problem.rows, &zeros, SourceKind::Charges);
                for k in 0..longest {
                    for kind in [SourceKind::Charges, SourceKind::Dipoles] {
                        let mut part = upload_kind(device, &problem, &per_leaf(k), &[], kind);
                        std::mem::swap(&mut part.output, &mut on.output);
                        launch_kind(device, &mut part, (layout, gradients, kind));
                        std::mem::swap(&mut part.output, &mut on.output);
                    }
                }
                assert_same_bits(
                    &format!(
                        "{layout}, gradients {gradients}: charges and dipoles against the charges \
                         and then the moments of each leaf"
                    ),
                    &super::download(device, &on),
                    &both,
                );
            }
            println!(
                "  {layout}, {}: split per leaf, permuted targets, repeated launches and the \
                 order of charges and moments bit for bit",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

/// The refusals: dipoles with more than one vector, and a store of the wrong length.
fn dipole_refusals(device: &mut Device) {
    let mut rng = Rng::new(0xd1b0_0007);
    let problem = fmm_rows::<f32>(&mut rng, &[3, 5], 200);
    let problem = with_moments(&mut rng, problem);
    let zeros = problem.output_store(false, |_, _| 0.0f32);
    let layout = P2pLayout::default_for(device.info());
    for (kind, vectors, store_kind) in [
        (SourceKind::Dipoles, 2, SourceKind::Dipoles),
        (SourceKind::Dipoles, 1, SourceKind::Charges),
        (SourceKind::ChargesAndDipoles, 1, SourceKind::Dipoles),
    ] {
        let mut on = upload_kind(device, &problem, &problem.rows, &zeros, store_kind);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let inputs = P2pInputs {
                near: &on.near,
                first_leaf: on.first,
                leaves: &on.leaves,
                source_offsets: &on.source_offsets,
                sources: on.sources.as_slice(),
                target_offsets: &on.target_offsets,
                target_input: on.target_input.as_slice(),
                vectors,
                source_kind: kind,
            };
            let _ = p2p(device, layout, false, &inputs, on.output.as_slice_mut());
        }));
        assert!(
            refused.is_err(),
            "{kind} with {vectors} vectors on a store of {store_kind}: not refused"
        );
    }
    println!("  dipoles with two vectors and stores of another kind refused");
}

tests_on!(
    cpu: dipole_pair_terms,
    dipole_sums,
    dipole_coincident,
    dipole_domain_ends,
    dipole_invariance_and_order,
    dipole_refusals,
);
tests_on!(
    metal: dipole_pair_terms,
    dipole_sums,
    dipole_coincident,
    dipole_domain_ends,
    dipole_invariance_and_order,
    dipole_refusals,
);
tests_on!(
    cuda: dipole_pair_terms,
    dipole_sums,
    dipole_coincident,
    dipole_domain_ends,
    dipole_invariance_and_order,
    dipole_refusals,
);
