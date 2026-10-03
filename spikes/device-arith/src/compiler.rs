//! Compiler behaviour per backend: inputs built so that each effect shows in the bits.
//!
//! One kernel, one unit, reads every operand from a buffer (so nothing can be folded on
//! the host side) and writes one value per probe. Each probe has the IEEE value of the
//! expression as written, evaluated on the host in Rust (which never contracts), and the
//! value the effect it probes would give.
//!
//! With e = 2^−⌈p/2⌉ (2⁻¹² in f32, 2⁻²⁷ in f64) and a = b = 1 + e, the exact product
//! a b = 1 + 2e + e² rounds to 1 + 2e (e² is half an ulp, a tie to even), so
//! a b − (1 + 2e) is 0 unfused and e² fused.

use cubecl::prelude::*;

use crate::backend::read;
use crate::real::Real;

/// Input slots of the probe kernel.
mod slot {
    pub const A: usize = 0; // 1 + e
    pub const NEG_C: usize = 1; // −(1 + 2e)
    pub const C: usize = 2; // 1 + 2e
    pub const ONE: usize = 3; // 1
    pub const TINY: usize = 4; // b, with fl(1 + b) = 1
    pub const N15: usize = 5; // 1.5 × smallest normal
    pub const N125: usize = 6; // 1.25 × smallest normal
    pub const SMALL: usize = 7; // 2^−100 (f32), 2^−1000 (f64)
    pub const SMALL2: usize = 8; // 2^−30 (f32), 2^−60 (f64): SMALL × SMALL2 is subnormal
    pub const SUB: usize = 9; // a subnormal
    pub const NEG_ZERO: usize = 10; // −0
    pub const ZERO: usize = 11; // +0
    pub const Q: usize = 12; // 3
    pub const INF: usize = 13; // +∞
    pub const NEG_A: usize = 14; // −(1 + e)
    /// Eight more copies of 1 + e, one per contraction probe: LLVM's CSE runs before
    /// cubecl-opt's InstCombine on the CPU runtime, so a shared a·b would have several
    /// uses and never fuse.
    pub const A_COPIES: usize = 15;
    pub const ONE2: usize = 23; // 1 again, a distinct load
    pub const COUNT: usize = 24;
}

/// The number of probes.
pub const PROBES: usize = 23;

// `(one - one)` and `inf - inf` repeat an operand on purpose: they probe cubecl-opt's folds.
#[allow(clippy::eq_op)]
#[cube(launch_unchecked)]
fn probe_kernel<F: Float>(v: &[F], out: &mut [F]) {
    let a = v[0];
    let neg_c = v[1];
    let c = v[2];
    let one = v[3];
    let tiny = v[4];
    let n15 = v[5];
    let n125 = v[6];
    let small = v[7];
    let small2 = v[8];
    let sub = v[9];
    let neg_zero = v[10];
    let zero = v[11];
    let q = v[12];
    let inf = v[13];
    let neg_a = v[14];

    // Contraction, each product from its own copy of a.
    let a0 = v[15];
    let a1 = v[16];
    let a2 = v[17];
    let a3 = v[18];
    let a4 = v[19];
    let a5 = v[20];
    let a6 = v[21];
    out[0] = a0 * a0 + neg_c;
    out[1] = neg_c + a1 * a1;
    out[2] = a2 * a2 - c;
    out[3] = c - a3 * a3;
    let p = a4 * a4;
    out[4] = p + neg_c;
    out[5] = p;
    out[6] = fma(a5, a5, neg_c);
    out[7] = a6 * a6 + neg_a * a;
    // Reassociation.
    out[8] = (one + tiny) - one;
    out[9] = one - (one + tiny);
    out[10] = (one - one) - tiny;
    // Subnormals.
    out[11] = n15 - n125;
    out[12] = small * small2;
    out[13] = sub * one;
    out[14] = sub + sub;
    // Signed zero and comparisons.
    out[15] = select(neg_zero == F::new(0.0f32), F::new(1.0f32), F::new(2.0f32));
    out[16] = select(sub == F::new(0.0f32), F::new(1.0f32), F::new(2.0f32));
    // Masking a non-finite 1/r at r² = 0.
    let rho = F::new(1.0f32) / zero.sqrt();
    out[17] = q * select(zero == F::new(0.0f32), F::new(0.0f32), rho);
    let mut t = q * rho;
    if zero == F::new(0.0f32) {
        t = F::new(0.0f32);
    }
    out[18] = t;
    // Algebraic folds (cubecl-opt's SCCP / SimplifyOps).
    out[19] = inf - inf;
    out[20] = neg_zero + F::new(0.0f32);
    // Reassociation with distinct values (x and y equal but loaded from different slots).
    let one2 = v[23];
    out[21] = (one + tiny) - one2;
    out[22] = one - (one2 + tiny);
}

/// One probe: what it computes, the IEEE value as written, and the value of the effect.
pub struct Probe<T> {
    /// The expression.
    pub name: &'static str,
    /// The IEEE value of the expression as written, unfused.
    pub ieee: T,
    /// The value if the effect occurs, and its name.
    pub effect: (T, &'static str),
}

/// The probes in precision T, in output order.
pub fn probes<T: Real>() -> (Vec<T>, Vec<Probe<T>>) {
    let half = (T::BITS + 1) / 2;
    let e = T::narrow(2f64.powi(-half));
    let one = T::narrow(1.0);
    let a = one + e;
    let c = one + e + e;
    let tiny = T::narrow(2f64.powi(-(T::BITS + 6)));
    let min = T::MIN_NORMAL;
    let (small, small2) = if T::BITS == 24 {
        (T::narrow(2f64.powi(-100)), T::narrow(2f64.powi(-30)))
    } else {
        (T::narrow(2f64.powi(-1000)), T::narrow(2f64.powi(-60)))
    };
    let sub = T::SUBNORMAL;
    let zero = T::narrow(0.0);
    let q = T::narrow(3.0);
    let inf = T::narrow(f64::INFINITY);
    let mut v = vec![zero; slot::COUNT];
    v[slot::A] = a;
    v[slot::NEG_C] = zero - c;
    v[slot::C] = c;
    v[slot::ONE] = one;
    v[slot::TINY] = tiny;
    v[slot::N15] = T::narrow(1.5) * min;
    v[slot::N125] = T::narrow(1.25) * min;
    v[slot::SMALL] = small;
    v[slot::SMALL2] = small2;
    v[slot::SUB] = sub;
    v[slot::NEG_ZERO] = T::narrow(-0.0);
    v[slot::ZERO] = zero;
    v[slot::Q] = q;
    v[slot::INF] = inf;
    v[slot::NEG_A] = zero - a;
    v[slot::ONE2] = one;
    for s in &mut v[slot::A_COPIES..slot::ONE2] {
        *s = a;
    }
    let e2 = e * e;
    let ab = a * a;
    let flushed = zero;
    let probes = vec![
        Probe {
            name: "a·b + c",
            ieee: ab - c,
            effect: (e2, "fused"),
        },
        Probe {
            name: "c + a·b",
            ieee: ab - c,
            effect: (e2, "fused"),
        },
        Probe {
            name: "a·b − c",
            ieee: ab - c,
            effect: (e2, "fused"),
        },
        Probe {
            name: "c − a·b",
            ieee: c - ab,
            effect: (zero - e2, "fused"),
        },
        Probe {
            name: "p = a·b (two uses); p + c",
            ieee: ab - c,
            effect: (e2, "fused"),
        },
        Probe {
            name: "p (the second use)",
            ieee: ab,
            effect: (ab, "—"),
        },
        Probe {
            name: "fma(a, b, c) explicit",
            ieee: e2,
            effect: (zero, "not fused"),
        },
        Probe {
            name: "a·a + (−a)·a",
            ieee: zero,
            effect: (e2, "fused (either product)"),
        },
        Probe {
            name: "(x + b) − x (one value x = 1)",
            ieee: zero,
            effect: (tiny, "simplified to b"),
        },
        Probe {
            name: "x − (x + z) (one value x = 1)",
            ieee: zero,
            effect: (zero - tiny, "simplified to −z"),
        },
        Probe {
            name: "(x − y) − z (control)",
            ieee: zero - tiny,
            effect: (zero, "—"),
        },
        Probe {
            name: "1.5 min − 1.25 min (subnormal out)",
            ieee: T::narrow(1.5) * min - T::narrow(1.25) * min,
            effect: (flushed, "flushed"),
        },
        Probe {
            name: "small · small (subnormal out)",
            ieee: small * small2,
            effect: (flushed, "flushed"),
        },
        Probe {
            name: "subnormal · 1 (subnormal in)",
            ieee: sub,
            effect: (flushed, "flushed"),
        },
        Probe {
            name: "subnormal + subnormal",
            ieee: sub + sub,
            effect: (flushed, "flushed"),
        },
        Probe {
            name: "select(−0 == 0, 1, 2)",
            ieee: one,
            effect: (T::narrow(2.0), "−0 ≠ 0"),
        },
        Probe {
            name: "select(subnormal == 0, 1, 2)",
            ieee: T::narrow(2.0),
            effect: (one, "flushed in compare"),
        },
        Probe {
            name: "q · select(r² == 0, 0, 1/√r²), r² = 0",
            ieee: zero,
            effect: (T::narrow(f64::NAN), "mask lost"),
        },
        Probe {
            name: "t = q/√r²; if r² == 0 { t = 0 }",
            ieee: zero,
            effect: (T::narrow(f64::NAN), "mask lost"),
        },
        Probe {
            name: "∞ − ∞ (same value)",
            ieee: T::narrow(f64::NAN),
            effect: (zero, "folded x − x → 0"),
        },
        Probe {
            name: "−0 + 0.0 (literal)",
            ieee: zero,
            effect: (T::narrow(-0.0), "folded x + 0 → x"),
        },
        Probe {
            name: "(x + b) − y (x = y = 1, distinct loads)",
            ieee: zero,
            effect: (tiny, "reassociated"),
        },
        Probe {
            name: "x − (y + z) (x = y = 1, distinct loads)",
            ieee: zero,
            effect: (zero - tiny, "reassociated"),
        },
    ];
    assert_eq!(probes.len(), PROBES);
    (v, probes)
}

/// The probe results on `client`: per probe its device value and what it shows.
pub fn run<T: Real>(client: &Client) -> Vec<(Probe<T>, T, String)> {
    let (v, probes) = probes::<T>();
    let hv = client.create_from_slice(T::as_bytes(&v));
    let out = client.empty(PROBES * size_of::<T>());
    // SAFETY: the kernel reads slot::COUNT inputs and writes PROBES outputs, the sizes of
    // the two buffers.
    unsafe {
        probe_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(hv, slot::COUNT),
            BufferArg::from_raw_parts(out.clone(), PROBES),
        );
    }
    let y = read::<T>(client, out);
    probes
        .into_iter()
        .zip(y)
        .map(|(p, y)| {
            let same =
                |a: T, b: T| a.bits() == b.bits() || (a.widen().is_nan() && b.widen().is_nan());
            let verdict = if same(y, p.ieee) {
                "as written".to_string()
            } else if same(y, p.effect.0) {
                format!("**{}**", p.effect.1)
            } else {
                "**other**".to_string()
            };
            (p, y, verdict)
        })
        .collect()
}
