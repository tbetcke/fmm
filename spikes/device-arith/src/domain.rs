//! The §3.13 argument on the device: ŷ, dₖ and r² of leaf-scaled pairs computed in a
//! device kernel as a P2P kernel computes them, checked against the rule that r² = 0
//! exactly for coincident stored points and every other r² is at least 2⁻¹⁰⁶.
//!
//! Formulations, per pair (ŷ = u_s for s = t, with no arithmetic):
//! - **written**: ŷ = ĉ + r̂ u_s and d = u_t − (ĉ + r̂ u_s) in one expression, as source
//!   code would write it (cubecl-opt fuses the product into an fma; the product is exact,
//!   so that changes no bit, but a reassociating compiler could form (u_t − ĉ) − r̂ u_s);
//!   r² = d₀d₀ + d₁d₁ + d₂d₂ as written;
//! - **fma forward / backward**: r² = fma(d₂, d₂, fma(d₁, d₁, d₀²)) and
//!   fma(d₀, d₀, fma(d₁, d₁, d₂²)), from the written d;
//! - **explicit**: ŷ = fma(r̂, u_s, ĉ), d = u_t − ŷ, r² forward;
//! - **unfused ŷ**: the product r̂ u_s also stored (a second use, so cubecl-opt does not
//!   fuse it), d = u_t − (ĉ + r̂ u_s) in one expression, r² forward: what a compiler that
//!   reassociates does to the written form when no fma shields it.

use cubecl::prelude::*;

use crate::backend::{Backend, read};
use crate::pairs::Pair;
use crate::primitives::shape;
use crate::real::Real;

/// The r² formulations, in output order.
pub const FORMULATIONS: [&str; 5] = [
    "written",
    "fma forward",
    "fma backward",
    "explicit fma ŷ",
    "unfused ŷ (product with a second use)",
];

/// The number of formulations.
const F: usize = FORMULATIONS.len();

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn domain_kernel<F: Float>(
    ut: &[F],
    us: &[F],
    c: &[F],
    r: &[F],
    self_pair: &[u32],
    y_out: &mut [F],
    d_out: &mut [F],
    d_unfused: &mut [F],
    products: &mut [F],
    r2_out: &mut [F],
    n: u32,
    stride: u32,
) {
    let mut i = ABSOLUTE_POS;
    let (n, stride) = (n as usize, stride as usize);
    while i < n {
        let rr = r[i];
        // ŷ = u_s for s = t; branch-free, so both forms are compiled for every pair.
        let is_self = self_pair[i] != 0;
        let u0 = us[3 * i];
        let u1 = us[3 * i + 1];
        let u2 = us[3 * i + 2];
        let d0 = ut[3 * i] - select(is_self, u0, c[3 * i] + rr * u0);
        let d1 = ut[3 * i + 1] - select(is_self, u1, c[3 * i + 1] + rr * u1);
        let d2 = ut[3 * i + 2] - select(is_self, u2, c[3 * i + 2] + rr * u2);
        let y0 = select(is_self, u0, fma(rr, u0, c[3 * i]));
        let y1 = select(is_self, u1, fma(rr, u1, c[3 * i + 1]));
        let y2 = select(is_self, u2, fma(rr, u2, c[3 * i + 2]));
        y_out[3 * i] = y0;
        y_out[3 * i + 1] = y1;
        y_out[3 * i + 2] = y2;
        let e0 = ut[3 * i] - y0;
        let e1 = ut[3 * i + 1] - y1;
        let e2 = ut[3 * i + 2] - y2;
        d_out[3 * i] = d0;
        d_out[3 * i + 1] = d1;
        d_out[3 * i + 2] = d2;
        let p0 = rr * u0;
        let p1 = rr * u1;
        let p2 = rr * u2;
        products[3 * i] = p0;
        products[3 * i + 1] = p1;
        products[3 * i + 2] = p2;
        let f0 = ut[3 * i] - select(is_self, u0, c[3 * i] + p0);
        let f1 = ut[3 * i + 1] - select(is_self, u1, c[3 * i + 1] + p1);
        let f2 = ut[3 * i + 2] - select(is_self, u2, c[3 * i + 2] + p2);
        d_unfused[3 * i] = f0;
        d_unfused[3 * i + 1] = f1;
        d_unfused[3 * i + 2] = f2;
        r2_out[5 * i] = d0 * d0 + d1 * d1 + d2 * d2;
        r2_out[5 * i + 1] = fma(d2, d2, fma(d1, d1, d0 * d0));
        r2_out[5 * i + 2] = fma(d0, d0, fma(d1, d1, d2 * d2));
        r2_out[5 * i + 3] = fma(e2, e2, fma(e1, e1, e0 * e0));
        r2_out[5 * i + 4] = fma(f2, f2, fma(f1, f1, f0 * f0));
        i += stride;
    }
}

/// What the device computed for a list of pairs.
pub struct DeviceValues<T> {
    /// ŷ (from the explicit fma), 3 per pair.
    pub y: Vec<T>,
    /// d (written), 3 per pair.
    pub d: Vec<T>,
    /// d with the unfused ŷ, 3 per pair.
    pub d_unfused: Vec<T>,
    /// r², one per formulation, 5 per pair.
    pub r2: Vec<T>,
}

/// Runs the domain kernel over `pairs`.
pub fn run<T: Real>(client: &Client, backend: Backend, pairs: &[Pair<T>]) -> DeviceValues<T> {
    let n = pairs.len();
    let flat = |f: &dyn Fn(&Pair<T>) -> [T; 3]| -> Vec<T> { pairs.iter().flat_map(f).collect() };
    let ut = client.create_from_slice(T::as_bytes(&flat(&|p| p.ut)));
    let us = client.create_from_slice(T::as_bytes(&flat(&|p| p.us)));
    let c = client.create_from_slice(T::as_bytes(&flat(&|p| p.c)));
    let r: Vec<T> = pairs.iter().map(|p| p.r).collect();
    let r = client.create_from_slice(T::as_bytes(&r));
    let s: Vec<u32> = pairs.iter().map(|p| u32::from(p.self_pair)).collect();
    let s = client.create_from_slice(u32::as_bytes(&s));
    let y = client.empty(3 * n * size_of::<T>());
    let d = client.empty(3 * n * size_of::<T>());
    let du = client.empty(3 * n * size_of::<T>());
    let products = client.empty(3 * n * size_of::<T>());
    let r2 = client.empty(F * n * size_of::<T>());
    let (units, cubes) = shape(backend);
    // SAFETY: every buffer holds the number of elements passed with it, and the kernel
    // touches pairs below n only.
    unsafe {
        domain_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(ut, 3 * n),
            BufferArg::from_raw_parts(us, 3 * n),
            BufferArg::from_raw_parts(c, 3 * n),
            BufferArg::from_raw_parts(r, n),
            BufferArg::from_raw_parts(s, n),
            BufferArg::from_raw_parts(y.clone(), 3 * n),
            BufferArg::from_raw_parts(d.clone(), 3 * n),
            BufferArg::from_raw_parts(du.clone(), 3 * n),
            BufferArg::from_raw_parts(products, 3 * n),
            BufferArg::from_raw_parts(r2.clone(), F * n),
            n as u32,
            units * cubes,
        );
    }
    DeviceValues {
        y: read(client, y),
        d: read(client, d),
        d_unfused: read(client, du),
        r2: read(client, r2),
    }
}

/// The outcome of the check on one set of pairs, per formulation.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    /// Pairs checked.
    pub pairs: usize,
    /// Pairs the reference skips (u_t == ŷ in all components).
    pub coincident: usize,
    /// Of those, from distinct points.
    pub coincident_distinct: usize,
    /// Components whose device ŷ differs from the host's fl(ĉ + r̂ u_s) in its bits.
    pub y_differs: usize,
    /// Components whose device d differs from the host's fl(u_t − ŷ).
    pub d_differs: usize,
    /// Components whose device d with the unfused ŷ differs from the host's.
    pub d_unfused_differs: usize,
    /// Components where the reassociated (u_t − ĉ) − r̂ u_s, on the host, differs from
    /// fl(u_t − fl(ĉ + r̂ u_s)): how sensitive the pairs are to reassociation.
    pub reassociation_sensitive: usize,
    /// Of those, components where the reassociated d is zero and the correct one is not,
    /// or the reverse (a coincidence gained or lost).
    pub reassociation_breaks_zero: usize,
    /// Components with d = 0 but u_t ≠ ŷ, or the reverse.
    pub d_zero_wrong: usize,
    /// The smallest nonzero |d|.
    pub min_d: f64,
    /// Per formulation: pairs where r² = 0 disagrees with the reference rule.
    pub r2_zero_wrong: [usize; F],
    /// Per formulation: nonzero r² below 2⁻¹⁰⁶.
    pub r2_below: [usize; F],
    /// Per formulation: r² above 2⁷.
    pub r2_above: [usize; F],
    /// Per formulation: the smallest nonzero r².
    pub min_r2: [f64; F],
    /// Per formulation: the largest r².
    pub max_r2: [f64; F],
    /// Per formulation: r² equal to the host's plain (unfused) r² bit for bit.
    pub r2_equals_plain: [usize; F],
    /// Per formulation: r² equal to the host's forward-fma r² bit for bit.
    pub r2_equals_forward: [usize; F],
}

impl Outcome {
    /// Whether the §3.13 rule holds for formulation `f`: d and r² zero exactly for
    /// coincident points, every nonzero r² in [2⁻¹⁰⁶, 2⁷].
    pub fn holds(&self, f: usize) -> bool {
        self.d_zero_wrong == 0
            && self.r2_zero_wrong[f] == 0
            && self.r2_below[f] == 0
            && self.r2_above[f] == 0
    }
}

/// Checks the device values of `pairs` against the host.
pub fn check<T: Real>(pairs: &[Pair<T>], dev: &DeviceValues<T>) -> Outcome {
    let lower = 2f64.powi(-106);
    let upper = 2f64.powi(7);
    let mut o = Outcome {
        pairs: pairs.len(),
        min_d: f64::INFINITY,
        min_r2: [f64::INFINITY; F],
        ..Outcome::default()
    };
    for (i, p) in pairs.iter().enumerate() {
        let y = p.y();
        let d = p.d();
        let coincident = p.coincident();
        o.coincident += usize::from(coincident);
        o.coincident_distinct += usize::from(coincident && p.multi);
        for k in 0..3 {
            let (dy, dd) = (dev.y[3 * i + k], dev.d[3 * i + k]);
            o.y_differs += usize::from(dy.bits() != y[k].bits());
            o.d_differs += usize::from(dd.bits() != d[k].bits());
            o.d_unfused_differs += usize::from(dev.d_unfused[3 * i + k].bits() != d[k].bits());
            if !p.self_pair {
                let re = (p.ut[k] - p.c[k]) - p.r * p.us[k];
                o.reassociation_sensitive += usize::from(re.bits() != d[k].bits());
                o.reassociation_breaks_zero +=
                    usize::from((re.widen() == 0.0) != (d[k].widen() == 0.0));
            }
            let zero = dd.widen() == 0.0;
            o.d_zero_wrong += usize::from(zero != (p.ut[k] == y[k]));
            if !zero {
                o.min_d = o.min_d.min(dd.widen().abs());
            }
        }
        let (plain, forward) = (p.r2_plain(), p.r2_forward());
        for f in 0..F {
            let r2 = dev.r2[F * i + f];
            let v = r2.widen();
            o.r2_zero_wrong[f] += usize::from((v == 0.0) != coincident);
            if v != 0.0 {
                o.r2_below[f] += usize::from(v < lower || v.is_nan());
                o.r2_above[f] += usize::from(v > upper || v.is_nan());
                o.min_r2[f] = o.min_r2[f].min(v);
                o.max_r2[f] = o.max_r2[f].max(v);
            }
            o.r2_equals_plain[f] += usize::from(r2.bits() == plain.bits());
            o.r2_equals_forward[f] += usize::from(r2.bits() == forward.bits());
        }
    }
    o
}
