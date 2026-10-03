//! Primitive accuracy: `sqrt`, division, `recip`, `inverse_sqrt` and `inverse_sqrt` with one
//! Newton step, applied on the device to inputs uploaded from the host and compared with a
//! correctly rounded host reference.

use cubecl::prelude::*;
use nd_fmm_validate::SplitMix64;

use crate::backend::{Backend, read};
use crate::real::Real;

/// The operations measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `x.sqrt()`.
    Sqrt,
    /// `a / x`, a uniform in [1, 2).
    Div,
    /// `F::new(1.0f32) / x`.
    OneDiv,
    /// `x.recip()`.
    Recip,
    /// `x.inverse_sqrt()`.
    Rsqrt,
    /// `x.inverse_sqrt()` and one Newton step, y (3/2 − x y²/2) with two fmas.
    RsqrtNewton,
    /// `F::new(1.0f32) / x.sqrt()`: the 1/r of the `sqrt` + division P2P.
    SqrtDiv,
}

impl Op {
    /// Every operation.
    pub const ALL: [Op; 7] = [
        Op::Sqrt,
        Op::Div,
        Op::OneDiv,
        Op::Recip,
        Op::Rsqrt,
        Op::RsqrtNewton,
        Op::SqrtDiv,
    ];

    /// The name in the tables.
    pub fn name(self) -> &'static str {
        match self {
            Op::Sqrt => "sqrt(x)",
            Op::Div => "a / x",
            Op::OneDiv => "1 / x",
            Op::Recip => "recip(x)",
            Op::Rsqrt => "inverse_sqrt(x)",
            Op::RsqrtNewton => "inverse_sqrt(x) + Newton",
            Op::SqrtDiv => "1 / sqrt(x)",
        }
    }

    fn code(self) -> u32 {
        Op::ALL.iter().position(|&o| o == self).unwrap() as u32
    }

    /// The error of the device result y in u_T.
    fn error<T: Real>(self, a: T, x: T, y: T) -> f64 {
        match self {
            Op::Sqrt => T::sqrt_error(x, y),
            Op::Div => T::div_error(a, x, y),
            Op::OneDiv | Op::Recip => T::div_error(T::narrow(1.0), x, y),
            Op::Rsqrt | Op::RsqrtNewton | Op::SqrtDiv => T::rsqrt_error(x, y),
        }
    }

    /// The host value the device result is compared with bit for bit: correctly rounded
    /// for `sqrt` and the divisions, fl(1 / fl(√x)) for the inverse square roots.
    fn host<T: Real>(self, a: T, x: T) -> T {
        let one = T::narrow(1.0);
        match self {
            Op::Sqrt => T::host_sqrt(x),
            Op::Div => a / x,
            Op::OneDiv | Op::Recip => one / x,
            Op::Rsqrt | Op::RsqrtNewton | Op::SqrtDiv => one / T::host_sqrt(x),
        }
    }
}

/// Applies operation `op` (the index into [`Op::ALL`]) to every x, with numerator a.
#[cube]
fn apply<F: Float>(a: F, x: F, #[comptime] op: u32) -> F {
    match comptime!(op) {
        0 => x.sqrt(),
        1 => a / x,
        2 => F::new(1.0f32) / x,
        3 => x.recip(),
        4 => x.inverse_sqrt(),
        5 => {
            let y = x.inverse_sqrt();
            let h = F::new(0.5f32) * x;
            let e = fma(-(h * y), y, F::new(0.5f32));
            fma(y, e, y)
        }
        _ => F::new(1.0f32) / x.sqrt(),
    }
}

#[cube(launch_unchecked)]
fn primitive_kernel<F: Float>(
    a: &[F],
    x: &[F],
    out: &mut [F],
    n: u32,
    stride: u32,
    #[comptime] op: u32,
) {
    let mut i = ABSOLUTE_POS;
    let (n, stride) = (n as usize, stride as usize);
    while i < n {
        out[i] = apply::<F>(a[i], x[i], op);
        i += stride;
    }
}

/// The launch shape of an elementwise kernel: (units per cube, cubes).
pub fn shape(backend: Backend) -> (u32, u32) {
    if backend.is_gpu() {
        (256, 4096)
    } else {
        (16, 16)
    }
}

/// Runs `op` on the device over `a` and `x`.
pub fn run<T: Real>(client: &Client, backend: Backend, op: Op, a: &[T], x: &[T]) -> Vec<T> {
    let n = x.len();
    let ha = client.create_from_slice(T::as_bytes(a));
    let hx = client.create_from_slice(T::as_bytes(x));
    let out = client.empty(size_of_val(x));
    let (units, cubes) = shape(backend);
    // SAFETY: the buffers hold n elements each, and the kernel touches indices below n only.
    unsafe {
        primitive_kernel::launch_unchecked::<T>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(ha, n),
            BufferArg::from_raw_parts(hx, n),
            BufferArg::from_raw_parts(out.clone(), n),
            n as u32,
            units * cubes,
            op.code(),
        );
    }
    read::<T>(client, out)
}

/// The inputs of one measurement.
pub struct Inputs<T> {
    /// What they are, for the table.
    pub label: String,
    /// The numerators of `a / x`.
    pub a: Vec<T>,
    /// The arguments.
    pub x: Vec<T>,
}

fn numerators<T: Real>(n: usize, seed: u64) -> Vec<T> {
    let mut rng = SplitMix64::new(seed);
    (0..n).map(|_| T::narrow(rng.range(1.0, 2.0))).collect()
}

/// f32 exhaustively over [1, 4): every float in the two binades.
pub fn exhaustive_f32() -> Inputs<f32> {
    let x: Vec<f32> = (1.0f32.to_bits()..4.0f32.to_bits())
        .map(f32::from_bits)
        .collect();
    Inputs {
        label: format!("[1, 4) exhaustive ({} values)", x.len()),
        a: numerators(x.len(), 11),
        x,
    }
}

/// Powers of two across the §3.13 domain: for every k in −108..=7, x = 2^k m for the 64
/// floats m from 1 up, the 64 below 2 and 4, and 256 seeded m in [1, 4).
pub fn powers_of_two<T: Real>(seed: u64) -> Inputs<T> {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::new();
    for k in -108..=7 {
        let p = 2f64.powi(k);
        for base in [1.0, 2.0, 4.0] {
            let mut m = T::narrow(base);
            for _ in 0..64 {
                if base == 1.0 {
                    x.push(T::narrow(p * m.widen()));
                    m = T::from_bits(m.bits() + 1);
                } else {
                    m = T::from_bits(m.bits() - 1);
                    x.push(T::narrow(p * m.widen()));
                }
            }
        }
        for _ in 0..256 {
            x.push(T::narrow(p * rng.range(1.0, 4.0)));
        }
    }
    Inputs {
        label: format!("2^k m, k = −108..7 ({} values)", x.len()),
        a: numerators(x.len(), seed + 1),
        x,
    }
}

/// `count` seeded log-uniform samples over [2⁻¹⁰⁸, 2⁷].
pub fn log_uniform<T: Real>(count: usize, seed: u64) -> Inputs<T> {
    let mut rng = SplitMix64::new(seed);
    let x = (0..count)
        .map(|_| T::narrow(2f64.powf(rng.range(-108.0, 7.0))))
        .collect();
    Inputs {
        label: format!("log-uniform on [2^-108, 2^7] ({count} samples)"),
        a: numerators(count, seed + 1),
        x,
    }
}

/// The measured accuracy of one operation on one input set.
pub struct Accuracy {
    /// The largest error in u_T.
    pub max_error: f64,
    /// The argument where it occurs.
    pub worst_x: f64,
    /// How many results differ from the host value of [`Op::host`] in their bits.
    pub differ: usize,
    /// The number of inputs.
    pub count: usize,
}

/// Measures `op` on the device against the host.
pub fn measure<T: Real>(client: &Client, backend: Backend, op: Op, inputs: &Inputs<T>) -> Accuracy {
    let y = run(client, backend, op, &inputs.a, &inputs.x);
    let mut acc = Accuracy {
        max_error: 0.0,
        worst_x: 0.0,
        differ: 0,
        count: y.len(),
    };
    for ((&a, &x), &y) in inputs.a.iter().zip(&inputs.x).zip(&y) {
        let e = op.error(a, x, y);
        if e > acc.max_error || e.is_nan() {
            acc.max_error = e;
            acc.worst_x = x.widen();
        }
        acc.differ += usize::from(op.host(a, x).bits() != y.bits());
    }
    acc
}

/// The edge-case inputs: 0, −0, the smallest normal, a subnormal, ∞ and NaN.
pub fn edge_inputs<T: Real>() -> Vec<(&'static str, T)> {
    vec![
        ("+0", T::narrow(0.0)),
        ("-0", T::narrow(-0.0)),
        ("min normal", T::MIN_NORMAL),
        ("subnormal", T::SUBNORMAL),
        ("+inf", T::narrow(f64::INFINITY)),
        ("NaN", T::narrow(f64::NAN)),
    ]
}

/// A value for the edge-case table: ±0, ±inf, NaN, or the value relative to the IEEE
/// result.
pub fn show<T: Real>(y: T, ieee: T) -> String {
    let v = y.widen();
    let text = if v.is_nan() {
        "NaN".to_string()
    } else if v == 0.0 {
        if v.is_sign_negative() { "-0" } else { "+0" }.to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "+inf" } else { "-inf" }.to_string()
    } else {
        format!("{v:.6e}")
    };
    let same = y.bits() == ieee.bits() || (v.is_nan() && ieee.widen().is_nan());
    if same {
        text
    } else {
        format!("**{text}** (IEEE {})", show(ieee, ieee))
    }
}

/// The edge cases of every operation: one row per input, one column per operation,
/// the device value and, where it differs, the IEEE value.
pub fn edge_cases<T: Real>(client: &Client, backend: Backend) -> Vec<(String, Vec<String>)> {
    let inputs = edge_inputs::<T>();
    let x: Vec<T> = inputs.iter().map(|e| e.1).collect();
    let a = vec![T::narrow(1.0); x.len()];
    let columns: Vec<Vec<T>> = Op::ALL
        .iter()
        .map(|&op| run(client, backend, op, &a, &x))
        .collect();
    inputs
        .iter()
        .enumerate()
        .map(|(i, (name, xv))| {
            let cells = Op::ALL
                .iter()
                .zip(&columns)
                .map(|(&op, col)| show(col[i], ieee(op, *xv)))
                .collect();
            (name.to_string(), cells)
        })
        .collect()
}

/// The IEEE result of `op` at x (a = 1), in f64 then rounded: exact for the special values.
fn ieee<T: Real>(op: Op, x: T) -> T {
    let v = x.widen();
    T::narrow(match op {
        Op::Sqrt => v.sqrt(),
        Op::Div | Op::OneDiv | Op::Recip => 1.0 / v,
        Op::Rsqrt | Op::RsqrtNewton | Op::SqrtDiv => 1.0 / v.sqrt(),
    })
}
