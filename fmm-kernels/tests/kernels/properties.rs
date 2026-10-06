//! Property tests on the CPU runtime and CUDA: gather and scatter-add over random column sizes,
//! matrix sizes, offsets and index arrays, against the host loops, bit for bit.

use std::cell::RefCell;

use nd_fmm_kernels::Device;
use nd_fmm_kernels::movement::{gather_columns, scatter_add_columns};
use proptest::prelude::*;
use proptest::test_runner::{Config, TestRunner};

use crate::common::{Rng, TestFloat, assert_bits, tests_on};

/// Column sizes: (p + 1)² for p = 0, 3, 8, the kernels the other tests compile.
const COLUMN_SIZES: [usize; 3] = [1, 16, 81];
/// Cases per property: each costs a few transfers and one launch.
const CASES: u32 = 64;

/// One random case: column size, columns, launch size, offsets of the three ranges,
/// and the seed of the data and indices.
fn cases() -> impl Strategy<Value = (usize, usize, usize, [usize; 3], u64)> {
    (
        0..COLUMN_SIZES.len(),
        1..60usize,
        0..120usize,
        [0..5usize, 0..5, 0..5],
        any::<u64>(),
    )
        .prop_map(|(c, k, m, offsets, seed)| (COLUMN_SIZES[c], k, m, offsets, seed))
}

fn values<T: TestFloat>(rng: &mut Rng, len: usize) -> Vec<T> {
    (0..len).map(|_| T::random_normal(rng)).collect()
}

/// Runs `CASES` cases of `property` on the device.
fn check<F>(device: &mut Device, name: &str, property: F)
where
    F: Fn(&mut Device, (usize, usize, usize, [usize; 3], u64)),
{
    let device = RefCell::new(device);
    let config = Config {
        cases: CASES,
        failure_persistence: None,
        ..Config::default()
    };
    TestRunner::new(config)
        .run(&cases(), |case| {
            property(&mut device.borrow_mut(), case);
            Ok(())
        })
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    println!("  {name}: {CASES} cases");
}

fn gather_property<T: TestFloat>(
    device: &mut Device,
    (n, k, m, [ox, oi, oy], seed): (usize, usize, usize, [usize; 3], u64),
) {
    let mut rng = Rng::new(seed);
    let x_data = values::<T>(&mut rng, ox + k * n);
    let indices: Vec<u32> = (0..m).map(|_| rng.below(k) as u32).collect();
    let mut index_data = vec![0; oi];
    index_data.extend(&indices);
    let y_data = values::<T>(&mut rng, oy + m * n);
    let x = device.upload(&x_data).unwrap();
    let idx = device.upload_indices(&index_data).unwrap();
    let mut y = device.upload(&y_data).unwrap();
    gather_columns(device, n, x.slice(ox..), idx.slice(oi..), y.slice_mut(oy..)).unwrap();
    let mut want = y_data;
    for (j, &c) in indices.iter().enumerate() {
        for r in 0..n {
            want[oy + j * n + r] = x_data[ox + c as usize * n + r];
        }
    }
    let mut got = want.clone();
    device.download(y.as_slice(), &mut got).unwrap();
    assert_bits("gather", &got, &want);
}

fn scatter_add_property<T: TestFloat>(
    device: &mut Device,
    (n, k, m, [ox, oi, oy], seed): (usize, usize, usize, [usize; 3], u64),
) {
    let mut rng = Rng::new(seed);
    let m = m.min(k);
    let x_data = values::<T>(&mut rng, ox + k * n);
    let indices = rng.permutation(k)[..m].to_vec();
    let mut index_data = vec![0; oi];
    index_data.extend(&indices);
    let y_data = values::<T>(&mut rng, oy + m * n);
    let y = device.upload(&y_data).unwrap();
    let idx = device.upload_indices(&index_data).unwrap();
    let mut x = device.upload(&x_data).unwrap();
    scatter_add_columns(device, n, y.slice(oy..), idx.slice(oi..), x.slice_mut(ox..)).unwrap();
    let mut want = x_data;
    for (j, &c) in indices.iter().enumerate() {
        for r in 0..n {
            let i = ox + c as usize * n + r;
            want[i] += y_data[oy + j * n + r];
        }
    }
    let mut got = want.clone();
    device.download(x.as_slice(), &mut got).unwrap();
    assert_bits("scatter-add", &got, &want);
}

fn gather_properties(device: &mut Device) {
    check(device, "gather f32", gather_property::<f32>);
    check(device, "gather f64", gather_property::<f64>);
}

fn scatter_add_properties(device: &mut Device) {
    check(device, "scatter-add f32", scatter_add_property::<f32>);
    check(device, "scatter-add f64", scatter_add_property::<f64>);
}

tests_on!(cpu: gather_properties, scatter_add_properties);
tests_on!(cuda: gather_properties, scatter_add_properties);
