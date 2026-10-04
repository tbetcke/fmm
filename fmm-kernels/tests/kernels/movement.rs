//! Zero, gather (box-major and, T12, coefficient-major in blocks), scatter-add and scatter
//! against plain host loops, bit for bit.
//!
//! Every launch works on ranges at odd offsets inside larger buffers, and the elements
//! around them are checked to be untouched. Column sizes are (p + 1)² for
//! p ∈ {0, 3, 8, 20}; index arrays are empty, of one element, permuted and (gather
//! only) repeated; the scatter-add accumulates onto nonzero data.

use nd_fmm_kernels::movement::{
    gather_coefficients, gather_columns, scatter_add_columns, scatter_values, zero,
};
use nd_fmm_kernels::{Device, DeviceElement};

use crate::common::{Rng, TestFloat, assert_bits, tests_on};

/// The degrees whose column sizes (p + 1)² are tested.
const DEGREES: [usize; 4] = [0, 3, 8, 20];
/// Columns of the matrices: at p = 20, 41 columns are 18,081 values, enough for a
/// CPU-runtime launch of several units.
const COLUMNS: usize = 41;
/// Elements before and after every range, which no launch may touch.
const PAD: usize = 3;

/// Element types of the movement tests, with their test data and comparison.
trait Data: DeviceElement + Default + std::fmt::Debug {
    /// Element i of a test vector: special values first, then random bits.
    fn sample(rng: &mut Rng, i: usize) -> Self;
    /// Asserts equality bit for bit (a NaN need only stay a NaN); returns whether every
    /// NaN kept its payload too.
    fn check(what: &str, got: &[Self], want: &[Self]) -> bool;
}

impl Data for f32 {
    fn sample(rng: &mut Rng, i: usize) -> Self {
        Self::specials()
            .get(i)
            .copied()
            .unwrap_or_else(|| Self::random_bits(rng))
    }
    fn check(what: &str, got: &[Self], want: &[Self]) -> bool {
        assert_bits(what, got, want)
    }
}

impl Data for f64 {
    fn sample(rng: &mut Rng, i: usize) -> Self {
        Self::specials()
            .get(i)
            .copied()
            .unwrap_or_else(|| Self::random_bits(rng))
    }
    fn check(what: &str, got: &[Self], want: &[Self]) -> bool {
        assert_bits(what, got, want)
    }
}

impl Data for u32 {
    fn sample(rng: &mut Rng, _: usize) -> Self {
        rng.next_u64() as u32
    }
    fn check(what: &str, got: &[Self], want: &[Self]) -> bool {
        assert_eq!(got, want, "{what}");
        true
    }
}

/// Prints what a copying kernel kept of the special values.
fn report_copies<E: Data>(what: &str, payloads: bool) {
    println!(
        "  {what} {}: every non-NaN bit pattern copied (−0, subnormals, ±∞ included); \
         NaN payloads {}",
        std::any::type_name::<E>(),
        if payloads {
            "kept"
        } else {
            "changed (NaN stays NaN)"
        }
    );
}

fn samples<E: Data>(rng: &mut Rng, n: usize) -> Vec<E> {
    (0..n).map(|i| E::sample(rng, i)).collect()
}

/// `inner` with `PAD` random elements before and after.
fn padded<E: Data>(rng: &mut Rng, inner: &[E]) -> Vec<E> {
    let mut v = samples::<E>(rng, PAD);
    v.extend_from_slice(inner);
    v.extend(samples::<E>(rng, PAD));
    v
}

/// The index arrays of a launch over `k` columns.
fn index_cases(rng: &mut Rng, k: usize, repeated: bool) -> Vec<(&'static str, Vec<u32>)> {
    let mut cases = vec![
        ("empty", vec![]),
        ("one", vec![(k as u32) / 2]),
        ("permuted", rng.permutation(k)),
    ];
    if repeated {
        let last = k as u32 - 1;
        cases.push(("repeated", vec![3, 3, 0, last, 3, 1, last, 1]));
    }
    cases
}

/// Uploads `indices` between two leading and trailing zeros; the launch uses the
/// range [`PAD`, `PAD` + len).
fn upload_indices(device: &mut Device, indices: &[u32]) -> nd_fmm_kernels::IndexBuffer {
    let mut v = vec![0; PAD];
    v.extend_from_slice(indices);
    v.extend([0; PAD]);
    device.upload_indices(&v).unwrap()
}

fn download<E: Data>(device: &mut Device, buffer: &nd_fmm_kernels::DeviceBuffer<E>) -> Vec<E> {
    let mut out = vec![E::default(); buffer.len()];
    device.download(buffer.as_slice(), &mut out).unwrap();
    out
}

fn zero_onto<E: Data>(device: &mut Device) {
    let mut rng = Rng::new(1);
    for len in [1, 7, 1000, 100_003] {
        let data = {
            let inner = samples::<E>(&mut rng, len);
            padded(&mut rng, &inner)
        };
        let mut buffer = device.upload(&data).unwrap();
        // A range in the upper half, an empty range, then the whole inner range.
        let mut want = data.clone();
        for range in [PAD + len / 2..PAD + len, PAD..PAD, PAD..PAD + len] {
            zero(device, buffer.slice_mut(range.clone())).unwrap();
            want[range.clone()].fill(E::default());
            E::check(
                &format!("zero {range:?} of {len}"),
                &download(device, &buffer),
                &want,
            );
        }
    }
}

fn gather<E: Data>(device: &mut Device) {
    let mut rng = Rng::new(2);
    let mut payloads = true;
    for p in DEGREES {
        let n = (p + 1) * (p + 1);
        let matrix = samples::<E>(&mut rng, COLUMNS * n);
        let x_data = padded(&mut rng, &matrix);
        let x = device.upload(&x_data).unwrap();
        for (case, indices) in index_cases(&mut rng, COLUMNS, true) {
            let m = indices.len();
            let y_data = {
                let inner = samples::<E>(&mut rng, m * n);
                padded(&mut rng, &inner)
            };
            let mut y = device.upload(&y_data).unwrap();
            let idx = upload_indices(device, &indices);
            gather_columns(
                device,
                n,
                x.slice(PAD..PAD + COLUMNS * n),
                idx.slice(PAD..PAD + m),
                y.slice_mut(PAD..PAD + m * n),
            )
            .unwrap();
            let mut want = y_data.clone();
            for (j, &c) in indices.iter().enumerate() {
                for r in 0..n {
                    want[PAD + j * n + r] = matrix[c as usize * n + r];
                }
            }
            payloads &= E::check(
                &format!("gather, p = {p}, {case}"),
                &download(device, &y),
                &want,
            );
            // Coefficient-major, in one block, blocks of one column and blocks of the
            // smallest divisor above one: coefficient k of column b w + c at (b n + k) w + c.
            let divisor = (2..m).find(|d| m % d == 0).unwrap_or(m);
            for block in [m, 1, divisor] {
                if block == 0 {
                    continue;
                }
                let mut y = device.upload(&y_data).unwrap();
                gather_coefficients(
                    device,
                    n,
                    x.slice(PAD..PAD + COLUMNS * n),
                    idx.slice(PAD..PAD + m),
                    block,
                    y.slice_mut(PAD..PAD + m * n),
                )
                .unwrap();
                let mut want = y_data.clone();
                for (j, &c) in indices.iter().enumerate() {
                    let (b, col) = (j / block, j % block);
                    for k in 0..n {
                        want[PAD + (b * n + k) * block + col] = matrix[c as usize * n + k];
                    }
                }
                payloads &= E::check(
                    &format!("gather coefficient-major, p = {p}, {case}, blocks of {block}"),
                    &download(device, &y),
                    &want,
                );
            }
        }
    }
    report_copies::<E>("gather", payloads);
}

fn scatter_add<T: TestFloat + Data>(device: &mut Device) {
    let mut rng = Rng::new(3);
    for p in DEGREES {
        let n = (p + 1) * (p + 1);
        let normals =
            |rng: &mut Rng, len| (0..len).map(|_| T::random_normal(rng)).collect::<Vec<T>>();
        let x_data = {
            let inner = normals(&mut rng, COLUMNS * n);
            padded(&mut rng, &inner)
        };
        for (case, indices) in index_cases(&mut rng, COLUMNS, false) {
            let m = indices.len();
            let y_packed = normals(&mut rng, m * n);
            let y = device.upload(&padded(&mut rng, &y_packed)).unwrap();
            let mut x = device.upload(&x_data).unwrap();
            let idx = upload_indices(device, &indices);
            scatter_add_columns(
                device,
                n,
                y.slice(PAD..PAD + m * n),
                idx.slice(PAD..PAD + m),
                x.slice_mut(PAD..PAD + COLUMNS * n),
            )
            .unwrap();
            let mut want = x_data.clone();
            for (j, &c) in indices.iter().enumerate() {
                for r in 0..n {
                    let k = PAD + c as usize * n + r;
                    want[k] += y_packed[j * n + r];
                }
            }
            T::check(
                &format!("scatter-add, p = {p}, {case}"),
                &download(device, &x),
                &want,
            );
        }
    }
}

fn scatter<E: Data>(device: &mut Device) {
    let mut rng = Rng::new(4);
    let len = 1000;
    let mut payloads = true;
    let x_data = {
        let inner = samples::<E>(&mut rng, len);
        padded(&mut rng, &inner)
    };
    for m in [0, 1, 500, len] {
        let indices = rng.permutation(len)[..m].to_vec();
        let values = samples::<E>(&mut rng, m);
        let y = device.upload(&padded(&mut rng, &values)).unwrap();
        let mut x = device.upload(&x_data).unwrap();
        let idx = upload_indices(device, &indices);
        scatter_values(
            device,
            y.slice(PAD..PAD + m),
            idx.slice(PAD..PAD + m),
            x.slice_mut(PAD..PAD + len),
        )
        .unwrap();
        let mut want = x_data.clone();
        for (&i, &v) in indices.iter().zip(&values) {
            want[PAD + i as usize] = v;
        }
        payloads &= E::check(
            &format!("scatter of {m} values"),
            &download(device, &x),
            &want,
        );
    }
    report_copies::<E>("scatter", payloads);
}

/// A launch with nothing to do launches nothing.
fn empty_launches_nothing(device: &mut Device) {
    let x = device.upload(&[1.0f32; 8]).unwrap();
    let mut y = device.alloc::<f32>(8).unwrap();
    let idx = device.upload_indices(&[]).unwrap();
    device.reset_counters();
    zero(device, y.slice_mut(4..4)).unwrap();
    gather_columns(device, 4, x.as_slice(), idx.as_slice(), y.slice_mut(..0)).unwrap();
    gather_coefficients(device, 4, x.as_slice(), idx.as_slice(), 1, y.slice_mut(..0)).unwrap();
    scatter_add_columns(device, 4, x.slice(..0), idx.as_slice(), y.as_slice_mut()).unwrap();
    scatter_values(device, x.slice(..0), idx.as_slice(), y.as_slice_mut()).unwrap();
    assert_eq!(device.counters().launches, 0);
}

fn zero_f32(device: &mut Device) {
    zero_onto::<f32>(device);
}
fn zero_f64(device: &mut Device) {
    zero_onto::<f64>(device);
}
fn zero_u32(device: &mut Device) {
    zero_onto::<u32>(device);
}
fn gather_f32(device: &mut Device) {
    gather::<f32>(device);
}
fn gather_f64(device: &mut Device) {
    gather::<f64>(device);
}
fn gather_u32(device: &mut Device) {
    gather::<u32>(device);
}
fn scatter_add_f32(device: &mut Device) {
    scatter_add::<f32>(device);
}
fn scatter_add_f64(device: &mut Device) {
    scatter_add::<f64>(device);
}
fn scatter_f32(device: &mut Device) {
    scatter::<f32>(device);
}
fn scatter_f64(device: &mut Device) {
    scatter::<f64>(device);
}
fn scatter_u32(device: &mut Device) {
    scatter::<u32>(device);
}

tests_on!(
    cpu: zero_f32,
    zero_f64,
    zero_u32,
    gather_f32,
    gather_f64,
    gather_u32,
    scatter_add_f32,
    scatter_add_f64,
    scatter_f32,
    scatter_f64,
    scatter_u32,
    empty_launches_nothing,
);
tests_on!(
    metal: zero_f32,
    zero_u32,
    gather_f32,
    gather_u32,
    scatter_add_f32,
    scatter_f32,
    scatter_u32,
    empty_launches_nothing,
);

/// The wrappers refuse what would make a kernel read or write out of bounds, before
/// launching.
#[cfg(feature = "cpu")]
mod refusals {
    use nd_fmm_kernels::BackendKind;
    use nd_fmm_kernels::movement::gather_columns;

    use crate::common::run;

    #[test]
    #[should_panic(expected = "address a matrix of 2 columns")]
    fn gather_refuses_an_index_out_of_range() {
        run(
            BackendKind::Cpu,
            "gather_refuses_an_index_out_of_range",
            |device| {
                let x = device.upload(&[0.0f32; 8]).unwrap();
                let mut y = device.alloc::<f32>(4).unwrap();
                let idx = device.upload_indices(&[2]).unwrap();
                let _ = gather_columns(device, 4, x.as_slice(), idx.as_slice(), y.as_slice_mut());
            },
        );
    }

    #[test]
    #[should_panic(expected = "need 8 values, not 4")]
    fn gather_refuses_a_short_output() {
        run(
            BackendKind::Cpu,
            "gather_refuses_a_short_output",
            |device| {
                let x = device.upload(&[0.0f64; 8]).unwrap();
                let mut y = device.alloc::<f64>(4).unwrap();
                let idx = device.upload_indices(&[0, 1]).unwrap();
                let _ = gather_columns(device, 4, x.as_slice(), idx.as_slice(), y.as_slice_mut());
            },
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index 1 repeats")]
    fn scatter_add_refuses_repeated_indices_in_debug_builds() {
        run(
            BackendKind::Cpu,
            "scatter_add_refuses_repeated_indices",
            |device| {
                let y = device.upload(&[1.0f32; 2]).unwrap();
                let mut x = device.alloc::<f32>(2).unwrap();
                let idx = device.upload_indices(&[1, 1]).unwrap();
                let _ = nd_fmm_kernels::movement::scatter_add_columns(
                    device,
                    1,
                    y.as_slice(),
                    idx.as_slice(),
                    x.as_slice_mut(),
                );
            },
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "index 0 repeats")]
    fn scatter_refuses_repeated_indices_in_debug_builds() {
        run(
            BackendKind::Cpu,
            "scatter_refuses_repeated_indices",
            |device| {
                let y = device.upload(&[1u32, 2]).unwrap();
                let mut x = device.alloc::<u32>(2).unwrap();
                let idx = device.upload_indices(&[0, 0]).unwrap();
                let _ = nd_fmm_kernels::movement::scatter_values(
                    device,
                    y.as_slice(),
                    idx.as_slice(),
                    x.as_slice_mut(),
                );
            },
        );
    }
}
