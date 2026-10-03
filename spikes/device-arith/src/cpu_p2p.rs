//! A CPU-shaped P2P on the CubeCL CPU runtime, against `nd_fmm_simd::P2pKernel` (decision 10).
//!
//! The kernel mirrors the Phase 3S NEON kernel (docs/design/simd-p2p.md §4.1–§4.2,
//! spikes/p2p-simd/SPIKE_REPORT.md): targets in `Vector<T, N>` lanes with N the NEON width
//! (4 in f32, 2 in f64), K vectors per block (K = 2 in f32, 4 in f64), sources broadcast
//! in input order, explicit `fma` for r² and the terms, ρ = 1/√r² by `sqrt` and division,
//! cleared to 0 where r² = 0. No shared memory and no `sync_cube`. Each unit takes a
//! contiguous share of the target leaves; one unit per core.
//!
//! Inputs are W1's gathered form (`nd_fmm_validate::p2p_kernels`): per target leaf its n_t
//! targets and 27 n_t sources. Targets are uploaded transposed, one vector of x, y and z
//! per N targets, and outputs come back the same way: the kernel's own layout, transposed
//! on the host and not timed (it is O(n_t) per leaf against O(27 n_t²) pairs).

use std::time::Instant;

use cubecl::prelude::*;
use nd_fmm_simd::P2pKernel;
use nd_fmm_validate::bench::median_time_per_call;
use nd_fmm_validate::p2p_kernels::{Accuracy, Set, accuracy};
use rayon::prelude::*;

use crate::backend::{read, sync};
use crate::real::Real;

/// The vector width N and the block K of a precision: NEON's width and Phase 3S's K.
pub fn layout<T: Real>() -> (usize, usize) {
    if T::BITS == 24 { (4, 2) } else { (2, 4) }
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cpu_p2p_kernel<F: Float, N: Size>(
    targets: &[Vector<F, N>],
    sources: &[F],
    charges: &[F],
    phi: &mut [Vector<F, N>],
    grad: &mut [Vector<F, N>],
    n_t: u32,
    n_s: u32,
    pool: u32,
    leaves: u32,
    units: u32,
    #[comptime] k: usize,
    #[comptime] gradients: bool,
) {
    let width = phi.vector_size();
    let u = UNIT_POS_X as usize;
    let (n_s, pool, leaves, units) = (n_s as usize, pool as usize, leaves as usize, units as usize);
    let vectors = n_t as usize / width;
    let first = leaves * u / units;
    let last = leaves * (u + 1) / units;
    let zero = Vector::<F, N>::new(F::new(0.0f32));
    let one = Vector::<F, N>::new(F::new(1.0f32));
    let mut tx = Array::<Vector<F, N>>::new(k);
    let mut ty = Array::<Vector<F, N>>::new(k);
    let mut tz = Array::<Vector<F, N>>::new(k);
    let mut ap = Array::<Vector<F, N>>::new(k);
    let mut ax = Array::<Vector<F, N>>::new(k);
    let mut ay = Array::<Vector<F, N>>::new(k);
    let mut az = Array::<Vector<F, N>>::new(k);
    for leaf in first..last {
        let set = leaf % pool;
        for block in 0..vectors / k {
            #[unroll]
            for b in 0..k {
                let v = block * k + b;
                let t = (set * vectors + v) * 3;
                tx[b] = targets[t];
                ty[b] = targets[t + 1];
                tz[b] = targets[t + 2];
                let o = leaf * vectors + v;
                ap[b] = phi[o];
                if gradients {
                    ax[b] = grad[3 * o];
                    ay[b] = grad[3 * o + 1];
                    az[b] = grad[3 * o + 2];
                }
            }
            for j in 0..n_s {
                let s = set * n_s + j;
                let sx = Vector::<F, N>::new(sources[3 * s]);
                let sy = Vector::<F, N>::new(sources[3 * s + 1]);
                let sz = Vector::<F, N>::new(sources[3 * s + 2]);
                let q = Vector::<F, N>::new(charges[s]);
                #[unroll]
                for b in 0..k {
                    let dx = tx[b] - sx;
                    let dy = ty[b] - sy;
                    let dz = tz[b] - sz;
                    let r2 = fma(dz, dz, fma(dy, dy, dx * dx));
                    let rho = select_many(r2.equal(&zero), zero, one / r2.sqrt());
                    ap[b] = fma(q, rho, ap[b]);
                    if gradients {
                        let w = (q * rho) * (rho * rho);
                        ax[b] = fma(-w, dx, ax[b]);
                        ay[b] = fma(-w, dy, ay[b]);
                        az[b] = fma(-w, dz, az[b]);
                    }
                }
            }
            #[unroll]
            for b in 0..k {
                let o = leaf * vectors + block * k + b;
                phi[o] = ap[b];
                if gradients {
                    grad[3 * o] = ax[b];
                    grad[3 * o + 1] = ay[b];
                    grad[3 * o + 2] = az[b];
                }
            }
        }
    }
}

/// A pool of target leaves on the device in the kernel's layout.
pub struct CpuPool {
    n_t: usize,
    n_s: usize,
    sets: usize,
    targets: cubecl::server::Handle,
    sources: cubecl::server::Handle,
    charges: cubecl::server::Handle,
}

/// Uploads `pool` (gathered form): targets transposed per N, sources and charges as they
/// are.
///
/// # Panics
///
/// If the sets differ in size, or n_t is not a multiple of K N.
pub fn upload<T: Real>(client: &Client, pool: &[Set<T>]) -> CpuPool {
    let (n, k) = layout::<T>();
    let (n_t, n_s) = (pool[0].targets.len(), pool[0].sources.len());
    assert_eq!(
        n_t % (n * k),
        0,
        "n_t must be a multiple of K N = {}",
        n * k
    );
    let mut targets = Vec::with_capacity(pool.len() * 3 * n_t);
    for set in pool {
        assert_eq!((set.targets.len(), set.sources.len()), (n_t, n_s));
        for v in set.targets.chunks_exact(n) {
            for c in 0..3 {
                targets.extend(v.iter().map(|t| t[c]));
            }
        }
    }
    let sources: Vec<T> = pool
        .iter()
        .flat_map(|s| s.sources.iter().flatten().copied())
        .collect();
    let charges: Vec<T> = pool
        .iter()
        .flat_map(|s| s.charges.iter().copied())
        .collect();
    CpuPool {
        n_t,
        n_s,
        sets: pool.len(),
        targets: client.create_from_slice(T::as_bytes(&targets)),
        sources: client.create_from_slice(T::as_bytes(&sources)),
        charges: client.create_from_slice(T::as_bytes(&charges)),
    }
}

/// Output buffers for `leaves` target leaves of n_t targets, zeroed.
pub struct CpuOutputs {
    phi: cubecl::server::Handle,
    grad: cubecl::server::Handle,
    leaves: usize,
    n_t: usize,
}

/// Zeroed outputs for `leaves` leaves of `pool`.
pub fn outputs<T: Real>(client: &Client, pool: &CpuPool, leaves: usize) -> CpuOutputs {
    let zeros = vec![T::narrow(0.0); 3 * leaves * pool.n_t];
    CpuOutputs {
        phi: client.create_from_slice(T::as_bytes(&zeros[..leaves * pool.n_t])),
        grad: client.create_from_slice(T::as_bytes(&zeros)),
        leaves,
        n_t: pool.n_t,
    }
}

/// Launches the kernel over `out.leaves` target leaves (the pool cycled) with `units`
/// units in one cube.
pub fn launch<T: Real>(
    client: &Client,
    pool: &CpuPool,
    out: &CpuOutputs,
    gradients: bool,
    units: u32,
) {
    let (n, k) = layout::<T>();
    let targets = pool.sets * pool.n_t * 3 / n;
    let outs = out.leaves.max(1) * pool.n_t / n;
    // SAFETY: targets hold 3 n_t / N vectors per set, sources 3 n_s and charges n_s scalars
    // per set; phi and grad hold n_t / N and 3 n_t / N vectors per launched leaf. The kernel
    // reads set `leaf % pool` and writes leaves below `out.leaves` only.
    unsafe {
        cpu_p2p_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(units),
            n,
            BufferArg::from_raw_parts(pool.targets.clone(), targets),
            BufferArg::from_raw_parts(pool.sources.clone(), pool.sets * pool.n_s * 3),
            BufferArg::from_raw_parts(pool.charges.clone(), pool.sets * pool.n_s),
            BufferArg::from_raw_parts(out.phi.clone(), outs),
            BufferArg::from_raw_parts(out.grad.clone(), 3 * outs),
            pool.n_t as u32,
            pool.n_s as u32,
            pool.sets as u32,
            out.leaves as u32,
            units,
            k,
            gradients,
        );
    }
}

/// Reads the outputs back, per leaf: φ per target and ∇φ per target.
pub fn read_outputs<T: Real>(client: &Client, out: CpuOutputs) -> Vec<(Vec<T>, Vec<[T; 3]>)> {
    let (n, _) = layout::<T>();
    let phi = read::<T>(client, out.phi);
    let grad = read::<T>(client, out.grad);
    (0..out.leaves)
        .map(|leaf| {
            let p = phi[leaf * out.n_t..(leaf + 1) * out.n_t].to_vec();
            let g = (0..out.n_t)
                .map(|t| {
                    let (v, lane) = (leaf * out.n_t / n + t / n, t % n);
                    std::array::from_fn(|c| grad[(3 * v + c) * n + lane])
                })
                .collect();
            (p, g)
        })
        .collect()
}

/// The sum accuracy of the kernel on `sets` (each its own leaf) against `oracles`, with
/// gradients if `gradients`.
pub fn accuracy_on<T: Real>(
    client: &Client,
    sets: &[Set<T>],
    oracles: &[nd_fmm_validate::p2p_kernels::Oracle],
    gradients: bool,
    units: u32,
) -> Accuracy {
    let pool = upload(client, sets);
    let out = outputs::<T>(client, &pool, sets.len());
    launch::<T>(client, &pool, &out, gradients, units);
    read_outputs::<T>(client, out)
        .iter()
        .zip(oracles)
        .map(|((p, g), o)| accuracy(o, p, gradients.then_some(&g[..])))
        .fold(Accuracy::default(), Accuracy::worst)
}

/// How many outputs (φ and ∇φ components) of the kernel on `sets` equal those of
/// `kernel` (`nd-fmm-simd`) bit for bit, and how many there are.
pub fn identical_to_simd<T: Real>(
    client: &Client,
    kernel: P2pKernel<T>,
    sets: &[Set<T>],
    gradients: bool,
) -> (usize, usize) {
    let pool = upload(client, sets);
    let out = outputs::<T>(client, &pool, sets.len());
    launch::<T>(client, &pool, &out, gradients, 1);
    let (mut same, mut total) = (0, 0);
    for ((p, g), set) in read_outputs::<T>(client, out).iter().zip(sets) {
        let n_t = set.targets.len();
        let (mut hp, mut hg) = (vec![T::narrow(0.0); n_t], vec![[T::narrow(0.0); 3]; n_t]);
        kernel.evaluate(
            &set.sources,
            &set.charges,
            &set.targets,
            &mut hp,
            gradients.then_some(&mut hg[..]),
        );
        for t in 0..n_t {
            same += usize::from(p[t].bits() == hp[t].bits());
            total += 1;
            if gradients {
                for c in 0..3 {
                    same += usize::from(g[t][c].bits() == hg[t][c].bits());
                    total += 1;
                }
            }
        }
    }
    (same, total)
}

/// The pair-term errors of the kernel in u_T: groups of K N targets around one source each
/// (one leaf of K N targets and one source), against the reference's terms.
pub fn pair_errors<T: Real>(
    client: &Client,
    pairs: &[([T; 3], [T; 3], T)],
) -> crate::p2p::PairErrors {
    let (n, k) = layout::<T>();
    let group = n * k;
    let sets: Vec<Set<T>> = pairs
        .chunks_exact(group)
        .map(|g| Set {
            targets: g.iter().map(|p| p.0).collect(),
            leaves: Vec::new(),
            sources: vec![g[0].1],
            charges: vec![g[0].2],
        })
        .collect();
    let used = sets.len() * group;
    let pool = upload(client, &sets);
    let out = outputs::<T>(client, &pool, sets.len());
    launch::<T>(client, &pool, &out, true, 16);
    let results = read_outputs(client, out);
    let phi: Vec<T> = results
        .iter()
        .flat_map(|(p, _)| p.iter().copied())
        .collect();
    let grad: Vec<T> = results
        .iter()
        .flat_map(|(_, g)| g.iter().flatten().copied())
        .collect();
    let pairs = &pairs[..used];
    let reference = crate::p2p::reference_terms(pairs);
    crate::p2p::term_errors(pairs, &reference, &phi, &grad)
}

/// Seconds per launch over `leaves` leaves, launches queued between syncs (the median of
/// 15 batches of at least 20 ms), after a warm-up launch that compiles the kernel.
pub fn time_launch<T: Real>(
    client: &Client,
    pool: &CpuPool,
    leaves: usize,
    gradients: bool,
    units: u32,
) -> f64 {
    let out = outputs::<T>(client, pool, leaves);
    launch::<T>(client, pool, &out, gradients, units);
    sync(client);
    median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            launch::<T>(client, pool, &out, gradients, units);
        }
        sync(client);
        start.elapsed()
    })
}

/// Seconds of one launch followed by a sync (latency), the median of 15.
pub fn launch_latency<T: Real>(client: &Client, pool: &CpuPool, leaves: usize, units: u32) -> f64 {
    let out = outputs::<T>(client, pool, leaves);
    launch::<T>(client, pool, &out, false, units);
    sync(client);
    let mut times: Vec<f64> = (0..15)
        .map(|_| {
            let start = Instant::now();
            launch::<T>(client, pool, &out, false, units);
            sync(client);
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[7]
}

/// Seconds per evaluation of `leaves` leaves (the pool cycled) by `nd_fmm_simd`'s kernel,
/// on one thread or on a rayon pool of `threads` threads.
pub fn time_simd<T: Real>(
    kernel: P2pKernel<T>,
    pool: &[Set<T>],
    leaves: usize,
    gradients: bool,
    threads: usize,
) -> f64 {
    let n_t = pool[0].targets.len();
    let mut out: Vec<(Vec<T>, Vec<[T; 3]>)> = (0..leaves)
        .map(|_| (vec![T::narrow(0.0); n_t], vec![[T::narrow(0.0); 3]; n_t]))
        .collect();
    let one = |leaf: usize, (p, g): &mut (Vec<T>, Vec<[T; 3]>)| {
        let set = &pool[leaf % pool.len()];
        kernel.evaluate(
            &set.sources,
            &set.charges,
            &set.targets,
            p,
            gradients.then_some(&mut g[..]),
        );
    };
    if threads == 1 {
        median_time_per_call(|calls| {
            let start = Instant::now();
            for _ in 0..calls {
                out.iter_mut()
                    .enumerate()
                    .for_each(|(leaf, o)| one(leaf, o));
            }
            start.elapsed()
        })
    } else {
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        threads.install(|| {
            median_time_per_call(|calls| {
                let start = Instant::now();
                for _ in 0..calls {
                    out.par_iter_mut()
                        .enumerate()
                        .for_each(|(leaf, o)| one(leaf, o));
                }
                start.elapsed()
            })
        })
    }
}
