//! Prototype P2P kernels with the signature of `nd_fmm_ref::p2p::p2p`:
//! φᵢ += Σⱼ qⱼ / rᵢⱼ and ∇φᵢ −= Σⱼ qⱼ dᵢⱼ / rᵢⱼ³ with d = xᵢ − yⱼ, a pair with r² = 0
//! contributing nothing (CONVENTIONS §3.13, "Fast kernels").
//!
//! - [`til`]: targets in lanes (design simd-p2p.md §4.1), K vectors of W targets per
//!   block, each source broadcast, each target's sources added in input order. The
//!   last block runs the same code on a padded stack copy that repeats the last target.
//! - [`sil`]: sources in lanes (green-kernels' order), T targets per block broadcast,
//!   W sources per vector, a horizontal sum per target at the end of the call. The
//!   last partial vector of sources is a padded copy with zero charges, so there is no
//!   scalar tail.
//!
//! Both are generic over the vector type and the inverse square root, and are
//! instantiated per ISA in [`neon`] and [`avx2`] as `#[inline(never)]` entry points,
//! which the disassembly script (`inner_loops.py`) finds by name.

use crate::simd::{Elem, Rsqrt, Vf};

/// A prototype entry point: the signature of `nd_fmm_ref::p2p::p2p`.
pub type KernelFn<E> = unsafe fn(&[[E; 3]], &[E], &[[E; 3]], &mut [E], Option<&mut [[E; 3]]>);

/// Loop order of a prototype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// Targets in lanes.
    Til,
    /// Sources in lanes.
    Sil,
}

/// One instantiated prototype.
#[derive(Clone, Copy)]
pub struct Proto<E> {
    /// Entry-point name, as in the disassembly.
    pub name: &'static str,
    /// "neon" or "avx2".
    pub isa: &'static str,
    /// Loop order.
    pub order: Order,
    /// K (target vectors per block) or T (targets per block).
    pub block: usize,
    /// Inverse-square-root formulation.
    pub rsqrt: &'static str,
    /// Its vector FP operations, including the estimate.
    pub rsqrt_ops: usize,
    /// "best" (the contract's cheapest), "gk" (green-kernels' formulation) or "relaxed".
    pub flavour: &'static str,
    /// With gradients.
    pub grad: bool,
    /// Lanes per vector.
    pub lanes: usize,
    /// The entry point; call only where `isa` is available.
    pub f: KernelFn<E>,
}

impl<E> Proto<E> {
    /// FP operations per pair and lane by the model of design §4.2, with this
    /// prototype's inverse square root: d (3), r² (3), mask (2), φ (1 without, 2 with
    /// gradients), gradient (5).
    pub fn model_ops(&self) -> usize {
        let base = if self.grad {
            3 + 3 + 2 + 2 + 5
        } else {
            3 + 3 + 2 + 1
        };
        base + self.rsqrt_ops
    }

    /// Calls the prototype.
    pub fn call(
        &self,
        sources: &[[E; 3]],
        charges: &[E],
        targets: &[[E; 3]],
        potential: &mut [E],
        gradient: Option<&mut [[E; 3]]>,
    ) {
        assert_eq!(
            self.grad,
            gradient.is_some(),
            "prototype {} and output",
            self.name
        );
        // SAFETY: prototypes are listed only for ISAs detected on this CPU
        // (`crate::prototypes`).
        unsafe { (self.f)(sources, charges, targets, potential, gradient) }
    }
}

/// Largest K·W (4 × 8) and W (8).
const MAX_BLOCK: usize = 32;
const MAX_W: usize = 8;

fn check<E>(s: &[[E; 3]], q: &[E], t: &[[E; 3]], p: &[E], g: Option<&[[E; 3]]>, grad: bool) {
    assert_eq!(s.len(), q.len());
    assert_eq!(t.len(), p.len());
    assert_eq!(grad, g.is_some());
    if let Some(g) = g {
        assert_eq!(t.len(), g.len());
    }
}

/// Targets in lanes, K vectors per block.
///
/// # Safety
/// The CPU has the ISA of `V`.
#[inline(always)]
pub unsafe fn til<V: Vf, R: Rsqrt<V>, const K: usize, const G: bool>(
    src: &[[V::E; 3]],
    q: &[V::E],
    tgt: &[[V::E; 3]],
    pot: &mut [V::E],
    mut grad: Option<&mut [[V::E; 3]]>,
) {
    check(src, q, tgt, pot, grad.as_deref(), G);
    let w = V::W;
    let b = K * w;
    assert!(b <= MAX_BLOCK);
    let zero = V::E::ZERO;
    let n = tgt.len();
    let mut start = 0;
    while start < n {
        let m = (n - start).min(b);
        // Per-lane copies of the block; padding lanes repeat the last target.
        let mut bx = [[zero; MAX_BLOCK]; 3];
        let mut bp = [zero; MAX_BLOCK];
        let mut bg = [[zero; MAX_BLOCK]; 3];
        for l in 0..b {
            let i = start + l.min(m - 1);
            for c in 0..3 {
                bx[c][l] = tgt[i][c];
            }
            if l < m {
                bp[l] = pot[i];
                if let Some(g) = grad.as_deref() {
                    for c in 0..3 {
                        bg[c][l] = g[i][c];
                    }
                }
            }
        }
        // SAFETY: the CPU has V's ISA (caller); every load and store below reads or
        // writes K·W ≤ MAX_BLOCK lanes of the stack buffers, or one source triple and
        // one charge from in-bounds references.
        unsafe {
            let tx: [[V; K]; 3] = core::array::from_fn(|c| {
                core::array::from_fn(|k| V::load(bx[c].as_ptr().add(k * w)))
            });
            let mut phi: [V; K] = core::array::from_fn(|k| V::load(bp.as_ptr().add(k * w)));
            let mut g: [[V; K]; 3] = core::array::from_fn(|c| {
                core::array::from_fn(|k| V::load(bg[c].as_ptr().add(k * w)))
            });
            for (s, &qj) in src.iter().zip(q) {
                let [sx, sy, sz] = V::splat3(s.as_ptr());
                let qv = V::splat(qj);
                for k in 0..K {
                    let dx = tx[0][k].sub(sx);
                    let dy = tx[1][k].sub(sy);
                    let dz = tx[2][k].sub(sz);
                    let r2 = V::fma(dz, dz, V::fma(dy, dy, dx.mul(dx)));
                    let rho = V::mask_zero(r2, R::rsqrt(r2));
                    if G {
                        let qr = qv.mul(rho);
                        phi[k] = phi[k].add(qr);
                        let qr3 = qr.mul(rho.mul(rho));
                        g[0][k] = V::fnma(qr3, dx, g[0][k]);
                        g[1][k] = V::fnma(qr3, dy, g[1][k]);
                        g[2][k] = V::fnma(qr3, dz, g[2][k]);
                    } else {
                        phi[k] = V::fma(qv, rho, phi[k]);
                    }
                }
            }
            for k in 0..K {
                phi[k].store(bp.as_mut_ptr().add(k * w));
                if G {
                    for c in 0..3 {
                        g[c][k].store(bg[c].as_mut_ptr().add(k * w));
                    }
                }
            }
        }
        pot[start..start + m].copy_from_slice(&bp[..m]);
        if let Some(gr) = grad.as_deref_mut() {
            for l in 0..m {
                gr[start + l] = [bg[0][l], bg[1][l], bg[2][l]];
            }
        }
        start += m;
    }
}

/// Sources in lanes, T targets per block.
///
/// # Safety
/// The CPU has the ISA of `V`.
#[inline(always)]
pub unsafe fn sil<V: Vf, R: Rsqrt<V>, const T: usize, const G: bool>(
    src: &[[V::E; 3]],
    q: &[V::E],
    tgt: &[[V::E; 3]],
    pot: &mut [V::E],
    mut grad: Option<&mut [[V::E; 3]]>,
) {
    check(src, q, tgt, pot, grad.as_deref(), G);
    let w = V::W;
    assert!(w <= MAX_W);
    let zero = V::E::ZERO;
    let ns = src.len();
    let full = ns / w;
    // The last partial vector: the remaining sources, then copies of the last source
    // with charge zero (finite terms times zero, or masked where r² = 0).
    let mut tail_s = [zero; 3 * MAX_W];
    let mut tail_q = [zero; MAX_W];
    let has_tail = ns > full * w;
    if has_tail {
        for l in 0..w {
            let j = (full * w + l).min(ns - 1);
            tail_s[3 * l..3 * l + 3].copy_from_slice(&src[j]);
            tail_q[l] = if full * w + l < ns { q[j] } else { zero };
        }
    }
    let n = tgt.len();
    let mut start = 0;
    while start < n {
        let m = (n - start).min(T);
        // SAFETY: the CPU has V's ISA (caller). Full vectors read W triples and W
        // charges at offsets c·W < full·W ≤ ns; the tail reads the stack copies.
        unsafe {
            let tv: [[V; 3]; T] = core::array::from_fn(|t| {
                let x = tgt[start + t.min(m - 1)];
                [V::splat(x[0]), V::splat(x[1]), V::splat(x[2])]
            });
            let mut acc = [V::splat(zero); T];
            let mut gacc = [[V::splat(zero); 3]; T];
            let ps = src.as_ptr() as *const V::E;
            for c in 0..full {
                sil_chunk::<V, R, T, G>(
                    ps.add(3 * c * w),
                    q.as_ptr().add(c * w),
                    &tv,
                    &mut acc,
                    &mut gacc,
                );
            }
            if has_tail {
                sil_chunk::<V, R, T, G>(tail_s.as_ptr(), tail_q.as_ptr(), &tv, &mut acc, &mut gacc);
            }
            for t in 0..m {
                pot[start + t] = pot[start + t] + acc[t].hsum();
                if let Some(gr) = grad.as_deref_mut() {
                    for c in 0..3 {
                        gr[start + t][c] = gr[start + t][c] + gacc[t][c].hsum();
                    }
                }
            }
        }
        start += m;
    }
}

/// One vector of W sources against T broadcast targets.
#[inline(always)]
unsafe fn sil_chunk<V: Vf, R: Rsqrt<V>, const T: usize, const G: bool>(
    ps: *const V::E,
    pq: *const V::E,
    tv: &[[V; 3]; T],
    acc: &mut [V; T],
    gacc: &mut [[V; 3]; T],
) {
    // SAFETY: the caller passes W readable triples at `ps` and W charges at `pq`.
    unsafe {
        let [sx, sy, sz] = V::load3(ps);
        let qv = V::load(pq);
        for t in 0..T {
            let dx = tv[t][0].sub(sx);
            let dy = tv[t][1].sub(sy);
            let dz = tv[t][2].sub(sz);
            let r2 = V::fma(dz, dz, V::fma(dy, dy, dx.mul(dx)));
            let rho = V::mask_zero(r2, R::rsqrt(r2));
            if G {
                let qr = qv.mul(rho);
                acc[t] = acc[t].add(qr);
                let qr3 = qr.mul(rho.mul(rho));
                gacc[t][0] = V::fnma(qr3, dx, gacc[t][0]);
                gacc[t][1] = V::fnma(qr3, dy, gacc[t][1]);
                gacc[t][2] = V::fnma(qr3, dz, gacc[t][2]);
            } else {
                acc[t] = V::fma(qv, rho, acc[t]);
            }
        }
    }
}

/// Entry points of one (ISA, precision, output) as `#[inline(never)]` functions, and
/// their list. `$order<$R, $K>` instantiates [`til`] or [`sil`].
macro_rules! entries {
    ($m:ident, $V:ty, $E:ty, $G:literal, $( $name:ident : $order:ident < $R:ty, $K:literal, $flavour:literal > ),* $(,)?) => {
        pub mod $m {
            use super::super::{sil, til, KernelFn, Order, Proto};
            #[allow(unused_imports)]
            use crate::simd::*;
            use crate::simd::{Rsqrt, Vf};
            #[allow(unused_imports)]
            use super::*;
            $(
                /// Prototype entry point.
                ///
                /// # Safety
                /// The CPU has the ISA of the vector type.
                #[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
                #[inline(never)]
                pub unsafe fn $name(s: &[[$E; 3]], q: &[$E], t: &[[$E; 3]], p: &mut [$E], g: Option<&mut [[$E; 3]]>) {
                    // SAFETY: forwarded from the caller.
                    unsafe { $order::<$V, $R, $K, $G>(s, q, t, p, g) }
                }
            )*
            /// Every entry point of this module.
            pub fn list() -> Vec<Proto<$E>> {
                vec![$(
                    Proto {
                        name: concat!(stringify!($m), "::", stringify!($name)),
                        isa: <$V as Vf>::ISA,
                        order: if stringify!($order) == "til" { Order::Til } else { Order::Sil },
                        block: $K,
                        rsqrt: <$R as Rsqrt<$V>>::NAME,
                        rsqrt_ops: <$R as Rsqrt<$V>>::OPS,
                        flavour: $flavour,
                        grad: $G,
                        lanes: <$V as Vf>::W,
                        f: $name as KernelFn<$E>,
                    }
                ),*]
            }
        }
    };
}

/// The standard set: both orders, three blockings, the best and the green-kernels
/// formulation; `$extra` adds rows (the relaxed f64 levels).
macro_rules! standard {
    ($m:ident, $V:ty, $E:ty, $G:literal, $Best:ty, $Gk:ty $(, $name:ident : $order:ident < $R:ty, $K:literal, $flavour:literal >)* ) => {
        entries!($m, $V, $E, $G,
            til_k1_best: til<$Best, 1, "best">,
            til_k2_best: til<$Best, 2, "best">,
            til_k4_best: til<$Best, 4, "best">,
            til_k1_gk: til<$Gk, 1, "gk">,
            til_k2_gk: til<$Gk, 2, "gk">,
            til_k4_gk: til<$Gk, 4, "gk">,
            sil_t1_best: sil<$Best, 1, "best">,
            sil_t2_best: sil<$Best, 2, "best">,
            sil_t4_best: sil<$Best, 4, "best">,
            sil_t1_gk: sil<$Gk, 1, "gk">,
            sil_t2_gk: sil<$Gk, 2, "gk">,
            sil_t4_gk: sil<$Gk, 4, "gk">
            $(, $name: $order<$R, $K, $flavour>)*
        );
    };
}

/// NEON prototypes. "best" is the formulation chosen by the study (SPIKE_REPORT.md):
/// the cheapest in the kernel among those within 4 u_T; "gk" is green-kernels' NEON
/// formulation (FRSQRTE, 2 / 3 FRSQRTS steps). "cand" rows (targets in lanes, K = 2)
/// compare every contract candidate inside the kernel; "relaxed" rows are the f64
/// levels below full precision.
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use crate::simd::SqrtDiv;
    use crate::simd::neon::{F32x4, F64x2};

    /// The NEON f32 formulation chosen by the study: `sqrt` then a division, the
    /// fastest inside the kernel on the M3 Max (the divider runs beside the FP pipes).
    pub type BestF32 = SqrtDiv;
    /// The NEON f64 formulation chosen by the study (as for f32).
    pub type BestF64 = SqrtDiv;

    macro_rules! neon_f32 {
        ($m:ident, $G:literal) => {
            standard!($m, F32x4, f32, $G, BestF32, Steps<2>,
                til_k2_c_n2: til<Newton<2>, 2, "cand">,
                til_k2_c_n3: til<Newton<3>, 2, "cand">,
                til_k2_c_s2: til<Steps<2>, 2, "cand">,
                til_k2_c_s3: til<Steps<3>, 2, "cand">,
                til_k2_c_p2: til<Poly<2>, 2, "cand">,
                til_k2_c_p3: til<Poly<3>, 2, "cand">,
                til_k2_c_p4: til<Poly<4>, 2, "cand">,
                til_k2_c_sd: til<SqrtDiv, 2, "cand">);
        };
    }
    macro_rules! neon_f64 {
        ($m:ident, $G:literal) => {
            standard!($m, F64x2, f64, $G, BestF64, Steps<3>,
                til_k2_c_n3: til<Newton<3>, 2, "cand">,
                til_k2_c_n4: til<Newton<4>, 2, "cand">,
                til_k2_c_s3: til<Steps<3>, 2, "cand">,
                til_k2_c_s4: til<Steps<4>, 2, "cand">,
                til_k2_c_p6: til<Poly<6>, 2, "cand">,
                til_k2_c_p7: til<Poly<7>, 2, "cand">,
                til_k2_c_p8: til<Poly<8>, 2, "cand">,
                til_k2_c_n1p3: til<NewtonPoly<1, 3>, 2, "cand">,
                til_k2_c_sd: til<SqrtDiv, 2, "cand">,
                til_k1_rel_s2: til<Steps<2>, 1, "relaxed">,
                til_k2_rel_s2: til<Steps<2>, 2, "relaxed">,
                til_k4_rel_s2: til<Steps<2>, 4, "relaxed">,
                til_k1_rel_p4: til<Poly<4>, 1, "relaxed">,
                til_k2_rel_p4: til<Poly<4>, 2, "relaxed">,
                til_k4_rel_p4: til<Poly<4>, 4, "relaxed">,
                til_k1_rel_n1p2: til<NewtonPoly<1, 2>, 1, "relaxed">,
                til_k2_rel_n1p2: til<NewtonPoly<1, 2>, 2, "relaxed">,
                til_k4_rel_n1p2: til<NewtonPoly<1, 2>, 4, "relaxed">);
        };
    }
    neon_f32!(f32_pot, false);
    neon_f32!(f32_grad, true);
    neon_f64!(f64_pot, false);
    neon_f64!(f64_grad, true);
}

/// AVX2 + FMA prototypes. The "best" formulations are chosen from the documented
/// estimate bound and the operation count (SPIKE_REPORT.md); "gk" is green-kernels'
/// AVX2 formulation (`rsqrtps` and 1 step for f32; via f32 and 2 steps for f64).
#[cfg(target_arch = "x86_64")]
pub mod avx2 {
    use crate::simd::Poly;
    use crate::simd::avx2::{F32x8, F64x4};

    /// The AVX2 f32 formulation chosen from the bound.
    pub type BestF32 = Poly<2>;
    /// The AVX2 f64 formulation chosen from the bound.
    pub type BestF64 = Poly<5>;

    standard!(f32_pot, F32x8, f32, false, BestF32, Steps<1>);
    standard!(f32_grad, F32x8, f32, true, BestF32, Steps<1>);
    standard!(f64_pot, F64x4, f64, false, BestF64, Steps<2>);
    standard!(f64_grad, F64x4, f64, true, BestF64, Steps<2>);
}

/// The prototypes of one precision available on this CPU, potential-only and
/// gradient entries together.
pub trait Prototypes: Elem {
    /// The list.
    fn prototypes() -> Vec<Proto<Self>>;
}

impl Prototypes for f32 {
    fn prototypes() -> Vec<Proto<f32>> {
        #[allow(unused_mut)]
        let mut out = Vec::new();
        #[cfg(target_arch = "aarch64")]
        {
            out.extend(neon::f32_pot::list());
            out.extend(neon::f32_grad::list());
        }
        #[cfg(target_arch = "x86_64")]
        if crate::machine::avx2_fma() {
            out.extend(avx2::f32_pot::list());
            out.extend(avx2::f32_grad::list());
        }
        out
    }
}

impl Prototypes for f64 {
    fn prototypes() -> Vec<Proto<f64>> {
        #[allow(unused_mut)]
        let mut out = Vec::new();
        #[cfg(target_arch = "aarch64")]
        {
            out.extend(neon::f64_pot::list());
            out.extend(neon::f64_grad::list());
        }
        #[cfg(target_arch = "x86_64")]
        if crate::machine::avx2_fma() {
            out.extend(avx2::f64_pot::list());
            out.extend(avx2::f64_grad::list());
        }
        out
    }
}
