//! The inverse-square-root study (design simd-p2p.md §4.3): every candidate's error in
//! units of u_T, its behaviour at 0 and at subnormal inputs, its throughput and
//! latency, and the rates of the single instructions it is built from.
//!
//! Errors:
//! - f32: exhaustively over [1, 4), against 1/√x in f64. The estimates and every
//!   refinement commute with scaling by 4ᵏ for normal inputs, so one period covers
//!   every normal input; the domain ends, the powers of two in the domain and their
//!   neighbours are checked as well.
//! - f64: on seeded log-uniform samples over the kernel domain [2⁻¹⁰⁸, 2⁷]
//!   (CONVENTIONS §3.13), the powers of two in it, their neighbours and the ends. The
//!   relative error e of y follows from the residual ρ = 1 − x y², computed with
//!   error-free products: (1 + e)² = 1 − ρ, so e = −ρ / (1 + √(1 − ρ)).

use std::hint::black_box;

use nd_fmm_math::RealScalar;
use std::time::Instant;

use crate::machine::{ASSUMED_GHZ, median_time_per_call};
use crate::rng::Rng;
use crate::simd::{Elem, Rsqrt, Vf};

/// One candidate of one ISA and precision.
pub struct Cand<E> {
    /// Formulation.
    pub name: &'static str,
    /// "neon" or "avx2".
    pub isa: &'static str,
    /// Vector FP operations, including the estimate.
    pub ops: usize,
    /// Lanes.
    pub lanes: usize,
    /// Applies the candidate to a slice whose length is a multiple of `lanes`.
    pub apply: unsafe fn(&[E], &mut [E]),
    /// A dependent chain y ← rsqrt(y) of the given length, from the given start.
    pub chain: unsafe fn(usize, E) -> E,
}

impl<E: Elem> Cand<E> {
    /// Applies the candidate to `x`, padding the last vector.
    pub fn eval(&self, x: &[E], out: &mut [E]) {
        let full = x.len() / self.lanes * self.lanes;
        // SAFETY: candidates are listed only for ISAs detected on this CPU; `full`
        // is a multiple of the lane count.
        unsafe { (self.apply)(&x[..full], &mut out[..full]) };
        if full < x.len() {
            let mut px = vec![E::ONE; self.lanes];
            let mut po = vec![E::ZERO; self.lanes];
            px[..x.len() - full].copy_from_slice(&x[full..]);
            // SAFETY: as above; the padded copies hold one vector.
            unsafe { (self.apply)(&px, &mut po) };
            out[full..].copy_from_slice(&po[..x.len() - full]);
        }
    }
}

/// Applies `R` to whole vectors.
///
/// # Safety
/// The CPU has V's ISA; the lengths are equal multiples of W.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[inline(never)]
unsafe fn apply<V: Vf, R: Rsqrt<V>>(x: &[V::E], out: &mut [V::E]) {
    assert_eq!(x.len(), out.len());
    assert_eq!(x.len() % V::W, 0);
    for i in (0..x.len()).step_by(V::W) {
        // SAFETY: the CPU has V's ISA (caller); i + W ≤ len for both slices.
        unsafe { R::rsqrt(V::load(x.as_ptr().add(i))).store(out.as_mut_ptr().add(i)) }
    }
}

/// A dependent chain y ← rsqrt(y); it converges to 1, so every input stays normal.
///
/// # Safety
/// The CPU has V's ISA.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[inline(never)]
unsafe fn chain<V: Vf, R: Rsqrt<V>>(iters: usize, x0: V::E) -> V::E {
    // SAFETY: the CPU has V's ISA (caller).
    unsafe {
        let mut v = V::splat(x0);
        for _ in 0..iters {
            v = R::rsqrt(v);
        }
        v.hsum()
    }
}

macro_rules! cands {
    ($V:ty, $E:ty; $($R:ty),* $(,)?) => {
        vec![$(
            Cand::<$E> {
                name: <$R as Rsqrt<$V>>::NAME,
                isa: <$V as Vf>::ISA,
                ops: <$R as Rsqrt<$V>>::OPS,
                lanes: <$V as Vf>::W,
                apply: apply::<$V, $R>,
                chain: chain::<$V, $R>,
            }
        ),*]
    };
}

/// The candidates of one precision on this CPU.
pub trait Candidates: Elem {
    /// The list, in report order.
    fn candidates() -> Vec<Cand<Self>>;
    /// The error study of one candidate; `quick` checks a subset.
    fn errors(c: &Cand<Self>, quick: bool) -> ErrorStats;
}

impl Candidates for f32 {
    fn candidates() -> Vec<Cand<f32>> {
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(target_arch = "aarch64")]
        {
            use crate::simd::neon::F32x4;
            use crate::simd::{Newton, Poly, SqrtDiv, Steps};
            v.extend(cands!(F32x4, f32;
                Newton<0>, Newton<1>, Newton<2>, Newton<3>,
                Steps<1>, Steps<2>, Steps<3>,
                Poly<2>, Poly<3>, Poly<4>,
                SqrtDiv));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::machine::avx2_fma() {
            use crate::simd::avx2::F32x8;
            use crate::simd::{Newton, Poly, SqrtDiv, Steps};
            v.extend(cands!(F32x8, f32;
                Newton<0>, Newton<1>, Newton<2>,
                Steps<1>, Steps<2>,
                Poly<2>, Poly<3>,
                SqrtDiv));
        }
        v
    }
    fn errors(c: &Cand<f32>, quick: bool) -> ErrorStats {
        errors_f32(c, if quick { 64 } else { 1 })
    }
}

impl Candidates for f64 {
    fn candidates() -> Vec<Cand<f64>> {
        #[allow(unused_mut)]
        let mut v = Vec::new();
        #[cfg(target_arch = "aarch64")]
        {
            use crate::simd::neon::F64x2;
            use crate::simd::{Newton, NewtonPoly, Poly, SqrtDiv, Steps};
            v.extend(cands!(F64x2, f64;
                Newton<0>, Newton<2>, Newton<3>, Newton<4>,
                Steps<2>, Steps<3>, Steps<4>,
                Poly<4>, Poly<6>, Poly<7>, Poly<8>,
                NewtonPoly<1, 2>, NewtonPoly<1, 3>,
                SqrtDiv));
        }
        #[cfg(target_arch = "x86_64")]
        if crate::machine::avx2_fma() {
            use crate::simd::avx2::{BitTrick, F64x4};
            use crate::simd::{Newton, NewtonPoly, Poly, SqrtDiv, Steps};
            v.extend(cands!(F64x4, f64;
                Newton<0>, Newton<2>, Newton<3>,
                Steps<2>, Steps<3>,
                Poly<3>, Poly<4>, Poly<5>,
                NewtonPoly<1, 2>,
                BitTrick<3>, BitTrick<4>,
                SqrtDiv));
        }
        v
    }
    fn errors(c: &Cand<f64>, quick: bool) -> ErrorStats {
        errors_f64(c, if quick { 100_000 } else { 10_000_000 })
    }
}

/// The measured error of one candidate.
#[derive(Clone, Copy, Debug)]
pub struct ErrorStats {
    /// Largest |relative error| / u_T.
    pub max_u: f64,
    /// The input where it occurs.
    pub at: f64,
    /// Number of inputs checked.
    pub count: u64,
}

impl ErrorStats {
    fn new() -> Self {
        Self {
            max_u: 0.0,
            at: f64::NAN,
            count: 0,
        }
    }
    fn push(&mut self, x: f64, rel: f64, u: f64) {
        self.count += 1;
        let e = rel.abs() / u;
        if e.is_nan() || e > self.max_u {
            // NaN counts as the worst.
            self.max_u = if e.is_nan() { f64::INFINITY } else { e };
            self.at = x;
        }
    }
}

/// Relative error of `y` ≈ 1/√x, both f64, from the residual with error-free products.
pub fn rel_err_f64(x: f64, y: f64) -> f64 {
    if !y.is_finite() || y <= 0.0 {
        return f64::NAN;
    }
    let p = y * y;
    let pe = y.mul_add(y, -p);
    let t = x * p;
    let te = x.mul_add(p, -t);
    // 1 − t is exact for t in [½, 2] (Sterbenz), which holds for every candidate.
    let rho = (1.0 - t) - te - x * pe;
    -rho / (1.0 + (1.0 - rho).sqrt())
}

/// Relative error of an f32 result against 1/√x in f64.
pub fn rel_err_f32(x: f32, y: f32) -> f64 {
    let t = 1.0 / f64::from(x).sqrt();
    (f64::from(y) - t) / t
}

/// The domain ends, the powers of two of the kernel domain and their neighbours.
pub fn special_inputs<E: Elem>() -> Vec<E> {
    let mut v = Vec::new();
    for k in -108..=7 {
        let p = 2f64.powi(k);
        let pe = E::from_f64(p);
        v.push(pe);
        let bits_up = next_up(pe);
        let bits_down = next_down(pe);
        if bits_down.to_f64() >= 2f64.powi(-108) {
            v.push(bits_down);
        }
        if bits_up.to_f64() <= 128.0 {
            v.push(bits_up);
        }
    }
    v
}

fn next_up<E: Elem>(x: E) -> E {
    if E::NAME == "f32" {
        E::from_f64(f64::from(f32::from_bits((x.to_f64() as f32).to_bits() + 1)))
    } else {
        E::from_f64(f64::from_bits(x.to_f64().to_bits() + 1))
    }
}

fn next_down<E: Elem>(x: E) -> E {
    if E::NAME == "f32" {
        E::from_f64(f64::from(f32::from_bits((x.to_f64() as f32).to_bits() - 1)))
    } else {
        E::from_f64(f64::from_bits(x.to_f64().to_bits() - 1))
    }
}

/// f32 errors: every input in [1, 4) (or every `stride`-th), and the special inputs.
pub fn errors_f32(c: &Cand<f32>, stride: u32) -> ErrorStats {
    let mut st = ErrorStats::new();
    let lo = 1.0f32.to_bits();
    let hi = 4.0f32.to_bits();
    let chunk = 1 << 16;
    let mut x = Vec::with_capacity(chunk);
    let mut y = vec![0.0f32; chunk];
    let mut b = lo;
    while b < hi {
        x.clear();
        while x.len() < chunk && b < hi {
            x.push(f32::from_bits(b));
            b += stride;
        }
        c.eval(&x, &mut y[..x.len()]);
        for (&xi, &yi) in x.iter().zip(&y) {
            st.push(f64::from(xi), rel_err_f32(xi, yi), f32::U);
        }
    }
    let sp = special_inputs::<f32>();
    let mut out = vec![0.0f32; sp.len()];
    c.eval(&sp, &mut out);
    for (&xi, &yi) in sp.iter().zip(&out) {
        st.push(f64::from(xi), rel_err_f32(xi, yi), f32::U);
    }
    st
}

/// f64 errors on `samples` seeded log-uniform inputs over [2⁻¹⁰⁸, 2⁷], and the special
/// inputs.
pub fn errors_f64(c: &Cand<f64>, samples: usize) -> ErrorStats {
    let mut st = ErrorStats::new();
    let mut rng = Rng::new(0x5EED_0F64);
    let chunk = 1 << 16;
    let mut x = Vec::with_capacity(chunk);
    let mut y = vec![0.0f64; chunk];
    let mut done = 0;
    while done < samples {
        x.clear();
        while x.len() < chunk && done < samples {
            x.push(2f64.powf(rng.range(-108.0, 7.0)));
            done += 1;
        }
        c.eval(&x, &mut y[..x.len()]);
        for (&xi, &yi) in x.iter().zip(&y) {
            st.push(xi, rel_err_f64(xi, yi), f64::U);
        }
    }
    let sp = special_inputs::<f64>();
    let mut out = vec![0.0f64; sp.len()];
    c.eval(&sp, &mut out);
    for (&xi, &yi) in sp.iter().zip(&out) {
        st.push(xi, rel_err_f64(xi, yi), f64::U);
    }
    st
}

/// The candidate's raw outputs (before the r² = 0 mask) at 0, the smallest subnormal
/// and a mid-range subnormal, as text.
pub fn edge_behaviour<E: Elem>(c: &Cand<E>) -> String {
    let inputs: [f64; 3] = if E::NAME == "f32" {
        [0.0, f64::from(f32::from_bits(1)), 1e-40]
    } else {
        [0.0, f64::from_bits(1), 1e-310]
    };
    let x: Vec<E> = inputs.iter().map(|&v| E::from_f64(v)).collect();
    let mut y = vec![E::ZERO; 3];
    c.eval(&x, &mut y);
    let show = |v: E| {
        let f = v.to_f64();
        if f.is_nan() {
            "NaN".to_string()
        } else if f.is_infinite() {
            "+inf".to_string()
        } else {
            format!("{f:.3e}")
        }
    };
    // Relative error at the mid subnormal, against 1/√x in f64.
    let t = 1.0 / inputs[2].sqrt();
    let rel = (y[2].to_f64() - t) / t;
    format!(
        "0 → {}; min sub → {}; {:.0e} → {} (rel {:.1e})",
        show(y[0]),
        show(y[1]),
        inputs[2],
        show(y[2]),
        rel
    )
}

/// Throughput on an L1-resident buffer: (ns per element, cycles per vector).
pub fn throughput<E: Elem>(c: &Cand<E>) -> (f64, f64) {
    let n = 4096;
    let mut rng = Rng::new(7);
    let x: Vec<E> = (0..n)
        .map(|_| E::from_f64(2f64.powf(rng.range(-20.0, 7.0))))
        .collect();
    let mut y = vec![E::ZERO; n];
    let t = median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            // SAFETY: as in `Cand::eval`; n is a multiple of every lane count.
            unsafe { (c.apply)(black_box(&x), black_box(&mut y)) };
        }
        start.elapsed()
    });
    let ns = t * 1e9 / n as f64;
    (ns, ns * c.lanes as f64 * ASSUMED_GHZ)
}

/// Latency of one evaluation in a dependent chain: (ns, cycles).
pub fn latency<E: Elem>(c: &Cand<E>) -> (f64, f64) {
    let iters = 1 << 12;
    let t = median_time_per_call(|calls| {
        let start = Instant::now();
        for _ in 0..calls {
            // SAFETY: as in `Cand::eval`.
            black_box(unsafe { (c.chain)(iters, black_box(E::from_f64(2.0))) });
        }
        start.elapsed()
    });
    let ns = t * 1e9 / iters as f64;
    (ns, ns * ASSUMED_GHZ)
}

/// The single instructions of the formulations.
#[derive(Clone, Copy, Debug)]
pub enum Prim {
    Fma,
    Mul,
    Add,
    Est,
    Step,
    Sqrt,
    Div,
}

impl Prim {
    /// Every instruction, in report order.
    pub const ALL: [Prim; 7] = [
        Prim::Fma,
        Prim::Mul,
        Prim::Add,
        Prim::Est,
        Prim::Step,
        Prim::Sqrt,
        Prim::Div,
    ];

    /// Name, with the NEON / AVX2 instruction.
    pub fn name(self) -> &'static str {
        match self {
            Prim::Fma => "fma (FMLA / vfmadd)",
            Prim::Mul => "mul (FMUL / vmulp)",
            Prim::Add => "add (FADD / vaddp)",
            Prim::Est => "estimate (FRSQRTE / vrsqrtps)",
            Prim::Step => "step (FRSQRTS / fnmadd)",
            Prim::Sqrt => "sqrt (FSQRT / vsqrtp)",
            Prim::Div => "div (FDIV / vdivp)",
        }
    }
}

/// `iters` rounds of one instruction on C independent chains.
///
/// # Safety
/// The CPU has V's ISA.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[inline(never)]
unsafe fn prim_chains<V: Vf, const OP: u8, const C: usize>(iters: usize) -> V::E {
    // SAFETY: the CPU has V's ISA (caller).
    unsafe {
        let one = black_box(V::splat(V::E::ONE));
        let zero = black_box(V::splat(V::E::ZERO));
        // Opaque starts, so that no chain folds at compile time (sqrt(1) = 1).
        let mut v: [V; C] =
            core::array::from_fn(|i| black_box(V::splat(V::E::from_f64(1.5 + 0.01 * i as f64))));
        for _ in 0..iters {
            for x in v.iter_mut() {
                *x = match OP {
                    // Accumulator form, as in the kernel: x + 1·0.
                    0 => V::fma(one, zero, *x),
                    1 => x.mul(one),
                    2 => x.add(zero),
                    3 => V::est(*x),
                    4 => V::step(*x, one),
                    5 => x.sqrt(),
                    _ => one.div(*x),
                };
            }
        }
        v.iter().fold(V::splat(V::E::ZERO), |a, &b| a.add(b)).hsum()
    }
}

/// Latency (cycles, one chain) and throughput (instructions per cycle, 16 chains) of one
/// instruction, at the assumed clock.
///
/// # Safety
/// The CPU has V's ISA.
pub unsafe fn prim_rates<V: Vf>(p: Prim) -> (f64, f64) {
    let iters = 1 << 12;
    let f1: unsafe fn(usize) -> V::E;
    let f16: unsafe fn(usize) -> V::E;
    match p {
        Prim::Fma => (f1, f16) = (prim_chains::<V, 0, 1>, prim_chains::<V, 0, 16>),
        Prim::Mul => (f1, f16) = (prim_chains::<V, 1, 1>, prim_chains::<V, 1, 16>),
        Prim::Add => (f1, f16) = (prim_chains::<V, 2, 1>, prim_chains::<V, 2, 16>),
        Prim::Est => (f1, f16) = (prim_chains::<V, 3, 1>, prim_chains::<V, 3, 16>),
        Prim::Step => (f1, f16) = (prim_chains::<V, 4, 1>, prim_chains::<V, 4, 16>),
        Prim::Sqrt => (f1, f16) = (prim_chains::<V, 5, 1>, prim_chains::<V, 5, 16>),
        Prim::Div => (f1, f16) = (prim_chains::<V, 6, 1>, prim_chains::<V, 6, 16>),
    }
    let time = |f: unsafe fn(usize) -> V::E| {
        median_time_per_call(|calls| {
            let start = Instant::now();
            for _ in 0..calls {
                // SAFETY: the CPU has V's ISA (caller).
                black_box(unsafe { f(black_box(iters)) });
            }
            start.elapsed()
        })
    };
    let lat = time(f1) * 1e9 / iters as f64 * ASSUMED_GHZ;
    let thr = (16 * iters) as f64 / (time(f16) * 1e9 * ASSUMED_GHZ);
    (lat, thr)
}
