//! P2P candidates on the device: the inverse square root as `sqrt` and division,
//! `inverse_sqrt` alone, or `inverse_sqrt` with one Newton step, each with and without
//! explicit fma, potential only and with gradients.
//!
//! - **Pair terms** ([`pair_errors`]): one term per pair, over pairs spanning the §3.13
//!   kernel domain, against the term of `nd_fmm_ref::p2p` in u_T (potential relative;
//!   each gradient component relative to |q| / r²).
//! - **Sums** ([`leaf_check`]): a GPU-shaped kernel (one unit per target) over FMM-shaped
//!   leaf sets, a target leaf and its 26 neighbours mapped as `LaplaceOperator` maps them,
//!   ŷ = ĉ + r̂ u_s (u_s itself for the leaf itself), against `direct_sum`, relative to the
//!   term magnitudes ([`nd_fmm_validate::p2p_kernels::accuracy`]).
//! - **Throughput** ([`leaf_rate`]): the same kernel timed quickly on a GPU, to rank the
//!   candidates. T6 times the production kernel.

use std::time::Instant;

use cubecl::prelude::*;
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::median_time_per_call;
use nd_fmm_validate::p2p_kernels::{Accuracy, Form, Kernel, Set, accuracy, check, oracles, w1};

use crate::backend::{Backend, read, sync};
use crate::primitives::shape;
use crate::real::Real;

/// An inverse square root of a candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rsqrt {
    /// ρ = 1 / √r².
    SqrtDiv,
    /// ρ = `inverse_sqrt(r²)`.
    Rsqrt,
    /// `inverse_sqrt` and one Newton step.
    RsqrtNewton,
}

/// One P2P candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    /// The inverse square root.
    pub rsqrt: Rsqrt,
    /// Whether r², ŷ, the Newton step and the accumulation use explicit `fma` calls.
    pub fma: bool,
}

impl Candidate {
    /// Every candidate.
    pub fn all() -> Vec<Candidate> {
        [Rsqrt::SqrtDiv, Rsqrt::Rsqrt, Rsqrt::RsqrtNewton]
            .into_iter()
            .flat_map(|rsqrt| [false, true].map(|fma| Candidate { rsqrt, fma }))
            .collect()
    }

    /// The name in the tables.
    pub fn name(self) -> String {
        let r = match self.rsqrt {
            Rsqrt::SqrtDiv => "sqrt + div",
            Rsqrt::Rsqrt => "inverse_sqrt",
            Rsqrt::RsqrtNewton => "inverse_sqrt + Newton",
        };
        format!(
            "{r}{}",
            if self.fma {
                ", explicit fma"
            } else {
                ", as written"
            }
        )
    }

    fn code(self) -> u32 {
        match self.rsqrt {
            Rsqrt::SqrtDiv => 0,
            Rsqrt::Rsqrt => 1,
            Rsqrt::RsqrtNewton => 2,
        }
    }
}

/// r² of d, written or with explicit fma.
#[cube]
fn r_squared<F: Float>(d0: F, d1: F, d2: F, #[comptime] use_fma: bool) -> F {
    if use_fma {
        fma(d2, d2, fma(d1, d1, d0 * d0))
    } else {
        d0 * d0 + d1 * d1 + d2 * d2
    }
}

/// ρ = 1/√r², cleared to 0 where r² = 0.
#[cube]
fn inverse_r<F: Float>(r2: F, #[comptime] rsqrt: u32, #[comptime] use_fma: bool) -> F {
    let rho = match comptime!(rsqrt) {
        0 => F::new(1.0f32) / r2.sqrt(),
        1 => r2.inverse_sqrt(),
        _ => {
            let y = r2.inverse_sqrt();
            if use_fma {
                let h = F::new(0.5f32) * r2;
                let e = fma(-(h * y), y, F::new(0.5f32));
                fma(y, e, y)
            } else {
                y * (F::new(1.5f32) - F::new(0.5f32) * r2 * y * y)
            }
        }
    };
    select(r2 == F::new(0.0f32), F::new(0.0f32), rho)
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pair_kernel<F: Float>(
    x: &[F],
    y: &[F],
    q: &[F],
    phi: &mut [F],
    grad: &mut [F],
    n: u32,
    stride: u32,
    #[comptime] rsqrt: u32,
    #[comptime] use_fma: bool,
) {
    let mut i = ABSOLUTE_POS;
    let (n, stride) = (n as usize, stride as usize);
    while i < n {
        let d0 = x[3 * i] - y[3 * i];
        let d1 = x[3 * i + 1] - y[3 * i + 1];
        let d2 = x[3 * i + 2] - y[3 * i + 2];
        let r2 = r_squared::<F>(d0, d1, d2, use_fma);
        let rho = inverse_r::<F>(r2, rsqrt, use_fma);
        let qr = q[i] * rho;
        phi[i] = qr;
        let w = qr * (rho * rho);
        grad[3 * i] = -(w * d0);
        grad[3 * i + 1] = -(w * d1);
        grad[3 * i + 2] = -(w * d2);
        i += stride;
    }
}

/// Seeded pairs (x, y, q) spanning the kernel domain 2⁻¹⁰⁸ ≤ r² ≤ 2⁷ of §3.13: y in
/// 2^−e [−1, 1]³, x = y + δ with |δ| log-uniform, |q| ≤ 1, kept if r² (in f64, from the
/// values in T) is in the domain. Each run of `group` consecutive pairs shares its y and q
/// (the CPU-shaped kernel measures one source against a block of targets).
pub fn domain_pairs<T: Real>(count: usize, group: usize, seed: u64) -> Vec<([T; 3], [T; 3], T)> {
    let mut rng = SplitMix64::new(seed);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let scale = 2f64.powf(-rng.range(0.0, 40.0));
        let y: [T; 3] = std::array::from_fn(|_| T::narrow(scale * rng.range(-1.0, 1.0)));
        let q = T::narrow(rng.range(-1.0, 1.0));
        let mut members = 0;
        while members < group {
            let mut dir: [f64; 3] = std::array::from_fn(|_| rng.range(-1.0, 1.0));
            let norm = dir.iter().map(|v| v * v).sum::<f64>().sqrt();
            let len = 2f64.powf(rng.range(-54.0, 3.5));
            dir = dir.map(|v| v / norm * len);
            let x: [T; 3] = std::array::from_fn(|k| T::narrow(y[k].widen() + dir[k]));
            let r2: f64 = (0..3).map(|k| (x[k].widen() - y[k].widen()).powi(2)).sum();
            if (2f64.powi(-108)..=2f64.powi(7)).contains(&r2) {
                out.push((x, y, q));
                members += 1;
            }
        }
    }
    out.truncate(count - count % group);
    out
}

/// The worst pair-term errors of a candidate, in u_T.
#[derive(Clone, Copy, Debug, Default)]
pub struct PairErrors {
    /// Potential, relative to the reference's term.
    pub potential: f64,
    /// Gradient component, relative to |q| / r², over the pairs where the gradient
    /// contract applies (in f32, r² ≥ 2⁻⁸⁴).
    pub gradient: f64,
    /// [`PairErrors::potential`] over the pairs whose nonzero |dₖ| are all at least 2⁻⁵³, as
    /// on leaf-scaled data (§3.13, "Fast kernels"): no dₖ² is subnormal in f32 there.
    pub potential_grid: f64,
    /// [`PairErrors::gradient`] over the same pairs.
    pub gradient_grid: f64,
    /// Pairs measured.
    pub pairs: usize,
}

/// The terms of `nd_fmm_ref::p2p` for each pair: (φ, ∇φ).
pub fn reference_terms<T: Real>(pairs: &[([T; 3], [T; 3], T)]) -> Vec<(T, [T; 3])> {
    pairs
        .iter()
        .map(|(x, y, q)| {
            let mut phi = [T::narrow(0.0)];
            let mut g = [[T::narrow(0.0); 3]];
            nd_fmm_ref::p2p::p2p(&[*y], &[*q], &[*x], &mut phi, Some(&mut g));
            (phi[0], g[0])
        })
        .collect()
}

/// The terms of one pair each, computed on the device by `candidate`.
pub fn device_terms<T: Real>(
    client: &Client,
    backend: Backend,
    candidate: Candidate,
    pairs: &[([T; 3], [T; 3], T)],
) -> (Vec<T>, Vec<T>) {
    let n = pairs.len();
    let x: Vec<T> = pairs.iter().flat_map(|p| p.0).collect();
    let y: Vec<T> = pairs.iter().flat_map(|p| p.1).collect();
    let q: Vec<T> = pairs.iter().map(|p| p.2).collect();
    let (hx, hy, hq) = (
        client.create_from_slice(T::as_bytes(&x)),
        client.create_from_slice(T::as_bytes(&y)),
        client.create_from_slice(T::as_bytes(&q)),
    );
    let phi = client.empty(n * size_of::<T>());
    let grad = client.empty(3 * n * size_of::<T>());
    let (units, cubes) = shape(backend);
    // SAFETY: the buffers hold 3n, 3n, n, n and 3n elements, and the kernel touches pairs
    // below n only.
    unsafe {
        pair_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(hx, 3 * n),
            BufferArg::from_raw_parts(hy, 3 * n),
            BufferArg::from_raw_parts(hq, n),
            BufferArg::from_raw_parts(phi.clone(), n),
            BufferArg::from_raw_parts(grad.clone(), 3 * n),
            n as u32,
            units * cubes,
            candidate.code(),
            candidate.fma,
        );
    }
    (read(client, phi), read(client, grad))
}

/// The worst pair-term errors of terms (φ, flattened ∇φ) against the reference's.
pub fn term_errors<T: Real>(
    pairs: &[([T; 3], [T; 3], T)],
    reference: &[(T, [T; 3])],
    phi: &[T],
    grad: &[T],
) -> PairErrors {
    let gradient_floor = if T::BITS == 24 { 2f64.powi(-84) } else { 0.0 };
    let mut e = PairErrors {
        pairs: pairs.len(),
        ..PairErrors::default()
    };
    for (i, ((x, y, q), (rphi, rg))) in pairs.iter().zip(reference).enumerate() {
        let r2: f64 = (0..3).map(|k| (x[k].widen() - y[k].widen()).powi(2)).sum();
        let grid = (0..3).all(|k| {
            let d = (x[k].widen() - y[k].widen()).abs();
            d == 0.0 || d >= 2f64.powi(-53)
        });
        let ep = ((phi[i].widen() - rphi.widen()) / rphi.widen()).abs() / T::U;
        let ep = if ep.is_nan() { f64::INFINITY } else { ep };
        e.potential = e.potential.max(ep);
        if grid {
            e.potential_grid = e.potential_grid.max(ep);
        }
        if r2 >= gradient_floor {
            let scale = q.widen().abs() / r2;
            for k in 0..3 {
                let eg = (grad[3 * i + k].widen() - rg[k].widen()).abs() / scale / T::U;
                let eg = if eg.is_nan() { f64::INFINITY } else { eg };
                e.gradient = e.gradient.max(eg);
                if grid {
                    e.gradient_grid = e.gradient_grid.max(eg);
                }
            }
        }
    }
    e
}

/// The pair-term errors of `candidate` over `pairs`.
pub fn pair_errors<T: Real>(
    client: &Client,
    backend: Backend,
    candidate: Candidate,
    pairs: &[([T; 3], [T; 3], T)],
    reference: &[(T, [T; 3])],
) -> PairErrors {
    let (phi, grad) = device_terms(client, backend, candidate, pairs);
    term_errors(pairs, reference, &phi, &grad)
}

/// The index of the target leaf itself among the 27 leaves of a W1 set.
const SELF_LEAF: usize = 13;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn leaf_kernel<F: Float>(
    targets: &[F],
    sources: &[F],
    charges: &[F],
    frames: &[F],
    phi: &mut [F],
    grad: &mut [F],
    n_t: u32,
    pool: u32,
    total: u32,
    #[comptime] rsqrt: u32,
    #[comptime] use_fma: bool,
    #[comptime] gradients: bool,
) {
    let g = ABSOLUTE_POS;
    let (n_t, pool, total) = (n_t as usize, pool as usize, total as usize);
    if g >= total {
        terminate!();
    }
    let set = (g / n_t) % pool;
    let t = set * n_t + g % n_t;
    let x0 = targets[3 * t];
    let x1 = targets[3 * t + 1];
    let x2 = targets[3 * t + 2];
    let mut p = phi[g];
    let mut g0 = F::new(0.0f32);
    let mut g1 = F::new(0.0f32);
    let mut g2 = F::new(0.0f32);
    if gradients {
        g0 = grad[3 * g];
        g1 = grad[3 * g + 1];
        g2 = grad[3 * g + 2];
    }
    for j in 0..27usize {
        let c0 = frames[4 * j];
        let c1 = frames[4 * j + 1];
        let c2 = frames[4 * j + 2];
        let r = frames[4 * j + 3];
        let base = (set * 27 + j) * n_t;
        for k in 0..n_t {
            let s = base + k;
            let u0 = sources[3 * s];
            let u1 = sources[3 * s + 1];
            let u2 = sources[3 * s + 2];
            let mut y0 = u0;
            let mut y1 = u1;
            let mut y2 = u2;
            if j != 13 {
                if use_fma {
                    y0 = fma(r, u0, c0);
                    y1 = fma(r, u1, c1);
                    y2 = fma(r, u2, c2);
                } else {
                    y0 = c0 + r * u0;
                    y1 = c1 + r * u1;
                    y2 = c2 + r * u2;
                }
            }
            let d0 = x0 - y0;
            let d1 = x1 - y1;
            let d2 = x2 - y2;
            let r2 = r_squared::<F>(d0, d1, d2, use_fma);
            let rho = inverse_r::<F>(r2, rsqrt, use_fma);
            let q = charges[s];
            if use_fma {
                p = fma(q, rho, p);
            } else {
                p += q * rho;
            }
            if gradients {
                let w = (q * rho) * (rho * rho);
                if use_fma {
                    g0 = fma(-w, d0, g0);
                    g1 = fma(-w, d1, g1);
                    g2 = fma(-w, d2, g2);
                } else {
                    g0 -= w * d0;
                    g1 -= w * d1;
                    g2 -= w * d2;
                }
            }
        }
    }
    phi[g] = p;
    if gradients {
        grad[3 * g] = g0;
        grad[3 * g + 1] = g1;
        grad[3 * g + 2] = g2;
    }
}

/// A W1 pool in leaf-scaled form, on the device: targets u_t, per leaf the sources u_s in
/// their own frame, the charges, and the 27 frames (ĉ, r̂).
pub struct LeafPool {
    /// n_t.
    pub n_t: usize,
    /// The number of sets.
    pub sets: usize,
    targets: cubecl::server::Handle,
    sources: cubecl::server::Handle,
    charges: cubecl::server::Handle,
    frames: cubecl::server::Handle,
}

/// The frame of leaf j of a W1 set: ĉ = 2 (a, b, c) for the block offsets a, b, c ∈ {−1,
/// 0, 1}, r̂ = 1 (same level).
fn w1_frame(j: usize) -> [f64; 3] {
    [j / 9, (j / 3) % 3, j % 3].map(|o| 2.0 * (o as f64 - 1.0))
}

/// Uploads a W1 pool in leaf-scaled form. W1's sources are absolute points ŷ in the
/// 3 × 3 × 3 block of leaves with centres 2(a, b, c) and half-width 1, which is a target
/// leaf and its same-level neighbours in leaf-scaled coordinates (ĉ = 2(a, b, c),
/// r̂ = 1); u_s = ŷ − ĉ is exact (Sterbenz), and so is ĉ + u_s = ŷ.
pub fn upload_pool<T: Real>(client: &Client, pool: &[Set<T>]) -> LeafPool {
    let n_t = pool[0].targets.len();
    let mut targets = Vec::new();
    let mut sources = Vec::new();
    let mut charges = Vec::new();
    for set in pool {
        targets.extend(set.targets.iter().flatten().copied());
        for (j, leaf) in set.leaves.iter().enumerate() {
            let c = w1_frame(j).map(T::narrow);
            for y in &leaf.sources {
                for k in 0..3 {
                    let u = y[k] - c[k];
                    assert_eq!((c[k] + u).bits(), y[k].bits(), "u_s = ŷ − ĉ must be exact");
                    sources.push(if j == SELF_LEAF { y[k] } else { u });
                }
            }
            charges.extend(&leaf.charges);
        }
    }
    let frames: Vec<T> = (0..27)
        .flat_map(|j| {
            let c = w1_frame(j);
            [c[0], c[1], c[2], 1.0].map(T::narrow)
        })
        .collect();
    LeafPool {
        n_t,
        sets: pool.len(),
        targets: client.create_from_slice(T::as_bytes(&targets)),
        sources: client.create_from_slice(T::as_bytes(&sources)),
        charges: client.create_from_slice(T::as_bytes(&charges)),
        frames: client.create_from_slice(T::as_bytes(&frames)),
    }
}

/// Launches the leaf kernel over `leaves` target leaves (the pool cycled), adding into
/// `phi` and `grad`.
#[allow(clippy::too_many_arguments)]
fn launch_leaves<T: Real>(
    client: &Client,
    pool: &LeafPool,
    candidate: Candidate,
    gradients: bool,
    leaves: usize,
    phi: &cubecl::server::Handle,
    grad: &cubecl::server::Handle,
    units: u32,
) {
    let total = leaves * pool.n_t;
    let n = pool.sets * pool.n_t;
    // SAFETY: targets hold 3n, sources 3 · 27n, charges 27n and frames 108 elements; phi
    // and grad hold `total` and 3 `total`; the kernel stops at `total` and indexes the pool
    // modulo its size.
    unsafe {
        leaf_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(total.div_ceil(units as usize) as u32, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(pool.targets.clone(), 3 * n),
            BufferArg::from_raw_parts(pool.sources.clone(), 81 * n),
            BufferArg::from_raw_parts(pool.charges.clone(), 27 * n),
            BufferArg::from_raw_parts(pool.frames.clone(), 108),
            BufferArg::from_raw_parts(phi.clone(), total),
            BufferArg::from_raw_parts(grad.clone(), 3 * total),
            pool.n_t as u32,
            pool.sets as u32,
            total as u32,
            candidate.code(),
            candidate.fma,
            gradients,
        );
    }
}

/// The sum accuracy of `candidate` (with gradients) on the first sets of a W1 pool,
/// against `direct_sum`, and the reference's on the same sets.
pub fn leaf_check<T: Real>(
    client: &Client,
    backend: Backend,
    candidate: Candidate,
    n_t: usize,
) -> (Accuracy, Accuracy) {
    let pool = w1::<T>(n_t, nd_fmm_validate::p2p_kernels::W1_SEED);
    let oracles = oracles(&pool);
    let checked = &pool[..oracles.len()];
    let device = upload_pool(client, checked);
    let total = checked.len() * n_t;
    let zeros = vec![T::narrow(0.0); 3 * total];
    let phi = client.create_from_slice(T::as_bytes(&zeros[..total]));
    let grad = client.create_from_slice(T::as_bytes(&zeros));
    let units = if backend.is_gpu() { 64 } else { 16 };
    launch_leaves::<T>(
        client,
        &device,
        candidate,
        true,
        checked.len(),
        &phi,
        &grad,
        units,
    );
    let phi = read::<T>(client, phi);
    let grad = read::<T>(client, grad);
    let mut worst = Accuracy::default();
    for (i, oracle) in oracles.iter().enumerate() {
        let g: Vec<[T; 3]> = grad[3 * i * n_t..3 * (i + 1) * n_t]
            .as_chunks::<3>()
            .0
            .to_vec();
        worst = worst.worst(accuracy(oracle, &phi[i * n_t..(i + 1) * n_t], Some(&g)));
    }
    let reference = check(
        &Kernel::<T>::Reference,
        checked,
        &oracles,
        Form::Gathered,
        true,
    );
    (worst, reference)
}

/// Pairs per second of `candidate` on a GPU over `leaves` target leaves of a W1 pool per
/// launch: launches queued between syncs, compilation excluded, the median of 15 batches
/// of at least 20 ms.
pub fn leaf_rate<T: Real>(
    client: &Client,
    candidate: Candidate,
    n_t: usize,
    gradients: bool,
    leaves: usize,
) -> f64 {
    let pool = w1::<T>(n_t, nd_fmm_validate::p2p_kernels::W1_SEED);
    let device = upload_pool(client, &pool);
    let total = leaves * n_t;
    let zeros = vec![T::narrow(0.0); 3 * total];
    let phi = client.create_from_slice(T::as_bytes(&zeros[..total]));
    let grad = client.create_from_slice(T::as_bytes(&zeros));
    let launch = || {
        launch_leaves::<T>(
            client, &device, candidate, gradients, leaves, &phi, &grad, 64,
        )
    };
    launch();
    sync(client);
    let time = median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            launch();
        }
        sync(client);
        start.elapsed()
    });
    (total * 27 * n_t) as f64 / time
}
