//! Device buffers round-trip bit for bit: upload, allocation, writes and downloads of
//! whole buffers and ranges, for f32, f64 and u32 (README exit gate C4.1); and (Phase 4S
//! T11) `Device::download_view` equals `download`, and `write_owned` equals `write`, bit
//! for bit and in their counters.

use nd_fmm_kernels::{Device, DeviceElement};

use crate::common::{Rng, TestFloat, assert_bits, tests_on};

/// The sizes of the default run; 10⁶ has its own tests.
const SIZES: [usize; 4] = [0, 1, 7, 1000];

/// Uploads `data`, downloads it whole and in ranges, writes into ranges, and checks
/// every result with `check(what, got, want)`.
fn round_trip<E: DeviceElement + Default + std::fmt::Debug>(
    device: &mut Device,
    data: &[E],
    check: &mut impl FnMut(&str, &[E], &[E]),
) {
    let n = data.len();
    let mut buffer = device.upload(data).unwrap();
    assert_eq!(buffer.len(), n);
    let mut out = vec![data.first().copied().unwrap_or(E::default()); n];
    device.download(buffer.as_slice(), &mut out).unwrap();
    check("upload, download", &out, data);

    // A range at an odd offset, and a write into another range.
    if n >= 3 {
        let range = 1..n - 1;
        let mut part = vec![data[0]; range.len()];
        device
            .download(buffer.slice(range.clone()), &mut part)
            .unwrap();
        check("download of a range", &part, &data[range.clone()]);

        let new: Vec<E> = data[range.clone()].iter().rev().copied().collect();
        device.write(buffer.slice_mut(range.clone()), &new).unwrap();
        let mut want = data.to_vec();
        want[range].copy_from_slice(&new);
        device.download(buffer.as_slice(), &mut out).unwrap();
        check("write into a range", &out, &want);
    }

    // Allocation gives zeros (+0 bits); a write fills it.
    let mut zeroed = device.alloc::<E>(n).unwrap();
    device.download(zeroed.as_slice(), &mut out).unwrap();
    check("alloc", &out, &vec![E::default(); n]);
    device.write(zeroed.as_slice_mut(), data).unwrap();
    device.download(zeroed.as_slice(), &mut out).unwrap();
    check("alloc, write", &out, data);
}

/// The sizes of the view and owned-write tests (Phase 4S T11).
const VIEW_SIZES: [usize; 4] = [0, 1, 7, 100_000];

/// `data` at the odd offset 3 of a larger buffer: `download_view` of that range and of the
/// whole buffer equals `download` bit for bit, with the same counters (one download of
/// the range's bytes, one sync); `write_owned` into the range equals `write`, with the same
/// counters (one upload of the range's bytes).
fn views_and_owned_writes<E: DeviceElement + Default + std::fmt::Debug>(
    device: &mut Device,
    data: &[E],
    check: &mut impl FnMut(&str, &[E], &[E]),
) {
    let n = data.len();
    let pad = data.first().copied().unwrap_or_default();
    let mut padded = vec![pad; n + 5];
    padded[3..3 + n].copy_from_slice(data);
    let size = size_of::<E>() as u64;
    let buffer = device.upload(&padded).unwrap();
    for (what, range) in [("range", 3..3 + n), ("whole buffer", 0..n + 5)] {
        let mut out = vec![E::default(); range.len()];
        device.reset_counters();
        device
            .download(buffer.slice(range.clone()), &mut out)
            .unwrap();
        let by_download = device.counters();
        device.reset_counters();
        let view = device.download_view(buffer.slice(range.clone())).unwrap();
        assert_eq!(view.len(), range.len());
        let got = view.to_vec();
        drop(view);
        let by_view = device.counters();
        check(&format!("view of the {what}"), &got, &out);
        check(
            &format!("download of the {what}"),
            &out,
            &padded[range.clone()],
        );
        assert_eq!(by_view, by_download, "{what}");
        assert_eq!(
            (by_view.downloads, by_view.syncs, by_view.download_bytes),
            (1, 1, range.len() as u64 * size),
            "{what}"
        );
        assert_eq!((by_view.uploads, by_view.launches), (0, 0), "{what}");
    }

    let new: Vec<E> = data.iter().rev().copied().collect();
    let mut want = padded.clone();
    want[3..3 + n].copy_from_slice(&new);
    let mut written = device.upload(&padded).unwrap();
    let mut owned = device.upload(&padded).unwrap();
    device.reset_counters();
    device.write(written.slice_mut(3..3 + n), &new).unwrap();
    let by_write = device.counters();
    device.reset_counters();
    device
        .write_owned(owned.slice_mut(3..3 + n), new.clone())
        .unwrap();
    let by_owned = device.counters();
    assert_eq!(by_owned, by_write);
    assert_eq!(
        (by_owned.uploads, by_owned.upload_bytes, by_owned.syncs),
        (1, n as u64 * size, 0)
    );
    let mut out = vec![E::default(); n + 5];
    let mut out_owned = vec![E::default(); n + 5];
    device.download(written.as_slice(), &mut out).unwrap();
    device.download(owned.as_slice(), &mut out_owned).unwrap();
    check("write into a range", &out, &want);
    check("write_owned into a range", &out_owned, &want);
}

fn views_float<T: TestFloat>(device: &mut Device) {
    let mut payloads = true;
    for n in VIEW_SIZES {
        for data in float_patterns::<T>(n, 29 + n as u64) {
            views_and_owned_writes(device, &data, &mut |what, got, want| {
                payloads &= assert_bits(&format!("n = {n}: {what}"), got, want);
            });
        }
    }
    println!(
        "  {}: download_view = download and write_owned = write, bit for bit and in their \
         counters, sizes {VIEW_SIZES:?} at offset 3 and whole; NaN payloads {}",
        std::any::type_name::<T>(),
        if payloads { "kept" } else { "NOT kept" }
    );
}

fn views_f32(device: &mut Device) {
    views_float::<f32>(device);
}

fn views_f64(device: &mut Device) {
    views_float::<f64>(device);
}

fn views_u32(device: &mut Device) {
    for n in VIEW_SIZES {
        let data = Rng::new(41 + n as u64).u32s(n);
        views_and_owned_writes(device, &data, &mut |what, got, want| {
            assert_eq!(got, want, "n = {n}: {what}");
        });
    }
    println!("  u32: download_view = download and write_owned = write, sizes {VIEW_SIZES:?}");
}

/// Float data of length n: the special values cycled, then any bit pattern.
fn float_patterns<T: TestFloat>(n: usize, seed: u64) -> [Vec<T>; 2] {
    let specials = T::specials();
    let mut rng = Rng::new(seed);
    [
        (0..n).map(|i| specials[i % specials.len()]).collect(),
        (0..n).map(|_| T::random_bits(&mut rng)).collect(),
    ]
}

fn round_trip_float<T: TestFloat>(device: &mut Device, sizes: &[usize]) {
    let mut payloads = true;
    for &n in sizes {
        for data in float_patterns::<T>(n, 17 + n as u64) {
            round_trip(device, &data, &mut |what, got, want| {
                payloads &= assert_bits(&format!("n = {n}: {what}"), got, want);
            });
        }
    }
    println!(
        "  {}: sizes {sizes:?} bit for bit; −0, smallest normal, subnormals, ±max, ±∞ kept \
         (no flush in transfers); NaN payloads {}",
        std::any::type_name::<T>(),
        if payloads {
            "kept"
        } else {
            "NOT kept (NaN stays NaN)"
        }
    );
}

fn round_trip_u32_sizes(device: &mut Device, sizes: &[usize]) {
    for &n in sizes {
        let mut rng = Rng::new(n as u64);
        let mut data = rng.u32s(n);
        for (d, s) in data.iter_mut().zip([0, 1, u32::MAX, 1 << 31]) {
            *d = s;
        }
        round_trip(device, &data, &mut |what, got, want| {
            assert_eq!(got, want, "n = {n}: {what}");
        });
    }
    println!("  u32: sizes {sizes:?} bit for bit");
}

fn round_trip_f32(device: &mut Device) {
    round_trip_float::<f32>(device, &SIZES);
}

fn round_trip_f64(device: &mut Device) {
    round_trip_float::<f64>(device, &SIZES);
}

fn round_trip_u32(device: &mut Device) {
    round_trip_u32_sizes(device, &SIZES);
}

fn round_trip_large_f32(device: &mut Device) {
    round_trip_float::<f32>(device, &[1_000_000]);
}

fn round_trip_large_f64(device: &mut Device) {
    round_trip_float::<f64>(device, &[1_000_000]);
}

fn round_trip_large_u32(device: &mut Device) {
    round_trip_u32_sizes(device, &[1_000_000]);
}

/// Indices uploaded from `u16` and `u8` are widened to `u32`.
fn widened_indices(device: &mut Device) {
    let small: Vec<u16> = vec![0, 1, 315, u16::MAX];
    let tiny: Vec<u8> = vec![7, 0, u8::MAX];
    for (indices, want) in [
        (
            device.upload_indices_widened(&small).unwrap(),
            vec![0, 1, 315, 65_535],
        ),
        (
            device.upload_indices_widened(&tiny).unwrap(),
            vec![7, 0, 255],
        ),
        (device.upload_indices(&[]).unwrap(), vec![]),
    ] {
        assert_eq!(
            indices.bound(),
            want.iter().max().map_or(0, |&m: &u32| u64::from(m) + 1)
        );
        let mut out = vec![0u32; want.len()];
        device
            .download(indices.buffer().as_slice(), &mut out)
            .unwrap();
        assert_eq!(out, want);
    }
}

tests_on!(
    cpu: round_trip_f32,
    round_trip_f64,
    round_trip_u32,
    views_f32,
    views_f64,
    views_u32,
    round_trip_large_f32,
    round_trip_large_f64,
    round_trip_large_u32,
    widened_indices,
);
tests_on!(
    metal: round_trip_f32,
    round_trip_u32,
    views_f32,
    views_u32,
    round_trip_large_f32,
    round_trip_large_u32,
    widened_indices,
);
tests_on!(
    cuda: round_trip_f32,
    round_trip_f64,
    round_trip_u32,
    views_f32,
    views_f64,
    views_u32,
    round_trip_large_f32,
    round_trip_large_f64,
    round_trip_large_u32,
    widened_indices,
);
