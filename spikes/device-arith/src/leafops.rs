//! Leaf-operator and GEMM implications: does contraction (or reassociation) change the
//! harmonics recursion (C4.3) or the GEMM (C4.4, C4.5) beyond their tolerances?
//!
//! - Regular and irregular solid harmonics to degree p, the recursion of
//!   `nd_fmm_math::harmonics::{regular, irregular}` (CONVENTIONS §3.5) ported operation by
//!   operation, one unit per point, against `nd-fmm-math` in the same precision and in f64.
//! - y ← y + A x per column in the order of `nd_fmm_tables::MatrixSet::apply` (for each
//!   output yᵣ, k increasing), against `apply` (unfused) and against the same order with
//!   every multiply–add fused on the host.

use cubecl::prelude::*;
use nd_fmm_math::{Layout, harmonics};
use nd_fmm_tables::MatrixSet;
use nd_fmm_validate::SplitMix64;

use crate::backend::{Backend, read};
use crate::real::Real;

#[cube]
fn at(n: usize, m: usize) -> usize {
    n * n + n + m
}

#[cube]
fn at_neg(n: usize, m: usize) -> usize {
    n * n + n - m
}

#[cube(launch_unchecked)]
fn harmonics_kernel<F: Float>(
    points: &[F],
    out: &mut [F],
    count: u32,
    #[comptime] p: usize,
    #[comptime] irregular: bool,
) {
    let i = ABSOLUTE_POS;
    if i >= count as usize {
        terminate!();
    }
    let o = i * (p + 1) * (p + 1);
    let x = points[3 * i];
    let y = points[3 * i + 1];
    let z = points[3 * i + 2];
    let r2 = x * x + y * y + z * z;
    let inv_r2 = F::new(1.0f32) / r2;
    let mut dre = F::new(1.0f32);
    if irregular {
        dre = F::new(1.0f32) / r2.sqrt();
    }
    let mut dim = F::new(0.0f32);
    for m in 0..p + 1 {
        if m > 0 {
            let mut s = F::new(1.0f32) / F::cast_from(2 * m);
            if irregular {
                s = F::cast_from(2 * m - 1) * inv_r2;
            }
            let re = (x * dre - y * dim) * s;
            let im = (x * dim + y * dre) * s;
            dre = re;
            dim = im;
        }
        out[o + at(m, m)] = dre;
        if m > 0 {
            out[o + at_neg(m, m)] = dim;
        }
        if m < p {
            let mut c = z;
            if irregular {
                c = F::cast_from(2 * m + 1) * z * inv_r2;
            }
            out[o + at(m + 1, m)] = c * dre;
            if m > 0 {
                out[o + at_neg(m + 1, m)] = c * dim;
            }
            for n in m + 2..p + 1 {
                // `irregular` is comptime: one branch is compiled.
                let (c1, c2) = if irregular {
                    (
                        F::cast_from(2 * n - 1) * z * inv_r2,
                        -F::cast_from((n - 1 - m) * (n - 1 + m)) * inv_r2,
                    )
                } else {
                    let den = F::cast_from((n + m) * (n - m));
                    (F::cast_from(2 * n - 1) * z / den, -r2 / den)
                };
                out[o + at(n, m)] = c1 * out[o + at(n - 1, m)] + c2 * out[o + at(n - 2, m)];
                if m > 0 {
                    out[o + at_neg(n, m)] =
                        c1 * out[o + at_neg(n - 1, m)] + c2 * out[o + at_neg(n - 2, m)];
                }
            }
        }
    }
}

/// The harmonics comparison of one kind and precision.
#[derive(Clone, Copy, Debug, Default)]
pub struct HarmonicsResult {
    /// Values bit-identical to `nd-fmm-math` in the same precision, and their count.
    pub identical: usize,
    /// All values.
    pub values: usize,
    /// The largest difference from `nd-fmm-math` in T, per point and degree relative to the
    /// largest value of that degree at that point, in u_T.
    pub vs_host: f64,
    /// [`HarmonicsResult::vs_host`] over the (point, degree) cells whose largest value is at
    /// least 2^(BITS) times the smallest normal of T, so that no value that matters is
    /// subnormal (where a backend that flushes to zero loses it).
    pub vs_host_normal: f64,
    /// Values of the host in T that are nonzero and below the normal range.
    pub subnormal: usize,
    /// The same against `nd-fmm-math` in f64, for the device and for the host in T.
    pub device_vs_f64: f64,
    /// See [`HarmonicsResult::device_vs_f64`].
    pub host_vs_f64: f64,
}

/// Points for regular (|x| ≤ √3, uniform in [−1, 1]³) or irregular harmonics (|x| in [2, 11]).
fn harmonic_points<T: Real>(count: usize, irregular: bool, seed: u64) -> Vec<[T; 3]> {
    let mut rng = SplitMix64::new(seed);
    (0..count)
        .map(|_| {
            let v: [f64; 3] = std::array::from_fn(|_| rng.range(-1.0, 1.0));
            if irregular {
                let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
                let r = rng.range(2.0, 11.0);
                v.map(|a| T::narrow(a / norm * r))
            } else {
                v.map(T::narrow)
            }
        })
        .collect()
}

/// Degree-p harmonics on the device against `nd-fmm-math`.
pub fn harmonics<T: Real>(
    client: &Client,
    backend: Backend,
    p: usize,
    irregular: bool,
    count: usize,
) -> HarmonicsResult {
    let layout = Layout::new(p);
    let len = layout.len();
    let points = harmonic_points::<T>(count, irregular, 41 + p as u64);
    let flat: Vec<T> = points.iter().flatten().copied().collect();
    let hp = client.create_from_slice(T::as_bytes(&flat));
    let out = client.empty(count * len * size_of::<T>());
    let units = if backend.is_gpu() { 64 } else { 16 };
    // SAFETY: points holds 3 count and out count (p + 1)² elements; units past count stop.
    unsafe {
        harmonics_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(count.div_ceil(units) as u32, 1, 1),
            CubeDim::new_1d(units as u32),
            BufferArg::from_raw_parts(hp, 3 * count),
            BufferArg::from_raw_parts(out.clone(), count * len),
            count as u32,
            p,
            irregular,
        );
    }
    let dev = read::<T>(client, out);
    let mut res = HarmonicsResult {
        values: count * len,
        ..HarmonicsResult::default()
    };
    let mut host = vec![T::narrow(0.0); len];
    let mut exact = vec![0.0f64; len];
    for (i, x) in points.iter().enumerate() {
        let x64 = x.map(|v| v.widen());
        if irregular {
            harmonics::irregular(p, *x, &mut host);
            harmonics::irregular(p, x64, &mut exact);
        } else {
            harmonics::regular(p, *x, &mut host);
            harmonics::regular(p, x64, &mut exact);
        }
        let d = &dev[i * len..(i + 1) * len];
        for n in 0..=p {
            let range = layout.degree(n);
            let scale = exact[range.clone()]
                .iter()
                .fold(0.0f64, |a, v| a.max(v.abs()));
            let normal = scale >= T::MIN_NORMAL.widen() / T::U;
            for j in range {
                res.identical += usize::from(d[j].bits() == host[j].bits());
                let e = |v: f64, w: f64| (v - w).abs() / scale / T::U;
                res.vs_host = res.vs_host.max(e(d[j].widen(), host[j].widen()));
                if normal {
                    res.vs_host_normal = res.vs_host_normal.max(e(d[j].widen(), host[j].widen()));
                }
                let h = host[j].widen().abs();
                res.subnormal += usize::from(h > 0.0 && h < T::MIN_NORMAL.widen());
                res.device_vs_f64 = res.device_vs_f64.max(e(d[j].widen(), exact[j]));
                res.host_vs_f64 = res.host_vs_f64.max(e(host[j].widen(), exact[j]));
            }
        }
    }
    res
}

#[cube(launch_unchecked)]
fn gemm_kernel<F: Float>(a: &[F], x: &[F], y: &mut [F], cols: u32, #[comptime] n: usize) {
    let c = ABSOLUTE_POS;
    if c >= cols as usize {
        terminate!();
    }
    for r in 0..n {
        let mut acc = y[c * n + r];
        for k in 0..n {
            acc += x[c * n + k] * a[r + k * n];
        }
        y[c * n + r] = acc;
    }
}

/// The GEMM comparison.
#[derive(Clone, Copy, Debug, Default)]
pub struct GemmResult {
    /// Outputs bit-identical to `MatrixSet::apply`.
    pub equals_apply: usize,
    /// Outputs bit-identical to the host with every multiply–add fused, in the same order.
    pub equals_fused: usize,
    /// Outputs.
    pub outputs: usize,
    /// The largest difference from `apply`, relative to the largest |y|, in u_T.
    pub vs_apply: f64,
}

/// y ← y + A x for `cols` columns, A of order n = (p + 1)², on the device and on the host.
pub fn gemm<T: Real>(client: &Client, backend: Backend, p: usize, cols: usize) -> GemmResult {
    let n = (p + 1) * (p + 1);
    let mut rng = SplitMix64::new(97);
    let mut set = MatrixSet::<T>::zeros(n, 1);
    for v in set.matrix_mut(0) {
        *v = T::narrow(rng.range(-1.0, 1.0));
    }
    let x: Vec<T> = (0..n * cols)
        .map(|_| T::narrow(rng.range(-1.0, 1.0)))
        .collect();
    let y0: Vec<T> = (0..n * cols)
        .map(|_| T::narrow(rng.range(-1.0, 1.0)))
        .collect();
    let ha = client.create_from_slice(T::as_bytes(set.matrix(0)));
    let hx = client.create_from_slice(T::as_bytes(&x));
    let hy = client.create_from_slice(T::as_bytes(&y0));
    let units = if backend.is_gpu() { 64 } else { 16 };
    // SAFETY: a holds n², x and y n cols elements; units past cols stop.
    unsafe {
        gemm_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(cols.div_ceil(units) as u32, 1, 1),
            CubeDim::new_1d(units as u32),
            BufferArg::from_raw_parts(ha, n * n),
            BufferArg::from_raw_parts(hx, n * cols),
            BufferArg::from_raw_parts(hy.clone(), n * cols),
            cols as u32,
            n,
        );
    }
    let dev = read::<T>(client, hy);
    let a = set.matrix(0);
    let mut res = GemmResult {
        outputs: n * cols,
        ..GemmResult::default()
    };
    let mut host = y0.clone();
    for c in 0..cols {
        set.apply(0, &x[c * n..(c + 1) * n], &mut host[c * n..(c + 1) * n]);
    }
    let scale = host.iter().fold(0.0f64, |m, v| m.max(v.widen().abs()));
    for c in 0..cols {
        for r in 0..n {
            let mut fused = y0[c * n + r];
            for k in 0..n {
                fused = T::host_fma(x[c * n + k], a[r + k * n], fused);
            }
            let (d, h) = (dev[c * n + r], host[c * n + r]);
            res.equals_apply += usize::from(d.bits() == h.bits());
            res.equals_fused += usize::from(d.bits() == fused.bits());
            res.vs_apply = res
                .vs_apply
                .max((d.widen() - h.widen()).abs() / scale / T::U);
        }
    }
    res
}
