//! The views of `view` round-trip through the device, the row-to-batch map points at
//! the batch entry of every row entry, and the cap on the CPU units changes no result.
//! Error measure: exact equality.

use nd_fmm_kernels::movement::gather_columns;
use nd_fmm_kernels::view::{
    BoxCoordinates, GroupedArrays, GroupedView, IndexView, LeafCoordinates, MAX_LEVEL,
};
use nd_fmm_kernels::{CPU_MAX_UNITS, Device};

use crate::common::{Rng, tests_on};

/// The six arrays of a grouped view: row offsets, sources, groups, batch offsets,
/// batch targets, batch sources.
type Arrays = (Vec<u32>, Vec<u32>, Vec<u16>, Vec<u32>, Vec<u32>, Vec<u32>);

/// A random grouped view of `nrows` targets, `nsources` sources and `ngroups` groups:
/// each target takes each group with probability one half, with a random source.
fn grouped(rng: &mut Rng, nrows: usize, nsources: usize, ngroups: usize) -> Arrays {
    let mut rows: Vec<Vec<(u32, u16)>> = vec![Vec::new(); nrows];
    for row in rows.iter_mut() {
        for g in 0..ngroups {
            if rng.next_u64() & 1 == 0 {
                row.push((rng.below(nsources) as u32, g as u16));
            }
        }
    }
    let mut row_offsets = vec![0u32];
    let (mut sources, mut groups) = (Vec::new(), Vec::new());
    for row in &rows {
        for &(s, g) in row {
            sources.push(s);
            groups.push(g);
        }
        row_offsets.push(sources.len() as u32);
    }
    let mut batch_offsets = vec![0u32];
    let (mut batch_targets, mut batch_sources) = (Vec::new(), Vec::new());
    for g in 0..ngroups {
        for (t, row) in rows.iter().enumerate() {
            if let Some(&(s, _)) = row.iter().find(|&&(_, h)| h as usize == g) {
                batch_targets.push(t as u32);
                batch_sources.push(s);
            }
        }
        batch_offsets.push(batch_targets.len() as u32);
    }
    (
        row_offsets,
        sources,
        groups,
        batch_offsets,
        batch_targets,
        batch_sources,
    )
}

/// An `IndexView` and a `GroupedView` come back as uploaded, also empty ones, and the
/// row-to-batch map points at the entry of the same pair in the batch of its group.
fn views_round_trip(device: &mut Device) {
    let mut rng = Rng::new(0x5ee5);
    for (nrows, nsources, ngroups) in [(0, 0, 8), (1, 1, 8), (37, 50, 8), (64, 100, 316)] {
        let (row_offsets, sources, groups, batch_offsets, batch_targets, batch_sources) =
            grouped(&mut rng, nrows, nsources, ngroups);
        let arrays = GroupedArrays {
            row_offsets: &row_offsets,
            sources: &sources,
            groups: &groups,
            batch_offsets: &batch_offsets,
            batch_targets: &batch_targets,
            batch_sources: &batch_sources,
        };
        let view = GroupedView::upload(device, &arrays, nsources).unwrap();
        assert_eq!(
            (view.nrows(), view.ngroups(), view.len()),
            (nrows, ngroups, sources.len())
        );
        let image = view.download(device).unwrap();
        assert_eq!(image.row_offsets, row_offsets);
        assert_eq!(image.sources, sources);
        assert_eq!(
            image.groups,
            groups.iter().map(|&g| u32::from(g)).collect::<Vec<_>>()
        );
        assert_eq!(image.batch_offsets, batch_offsets);
        assert_eq!(image.batch_targets, batch_targets);
        assert_eq!(image.batch_sources, batch_sources);
        for t in 0..nrows {
            for e in row_offsets[t] as usize..row_offsets[t + 1] as usize {
                let k = image.row_to_batch[e] as usize;
                let g = groups[e] as usize;
                assert!((batch_offsets[g] as usize..batch_offsets[g + 1] as usize).contains(&k));
                assert_eq!((batch_targets[k], batch_sources[k]), (t as u32, sources[e]));
            }
        }

        let csr = IndexView::upload(device, &row_offsets, &sources, nsources).unwrap();
        assert_eq!((csr.nrows(), csr.len()), (nrows, sources.len()));
        let image = csr.download(device).unwrap();
        assert_eq!(
            (image.row_offsets, image.entries),
            (row_offsets.clone(), sources.clone())
        );
    }
}

/// Box and leaf coordinates come back as uploaded, the extremes of each level included.
fn coordinates_round_trip(device: &mut Device) {
    let mut rng = Rng::new(0xc00d);
    let levels: Vec<Vec<[u32; 3]>> = (0..=MAX_LEVEL)
        .map(|l| {
            let top = (1u32 << l) - 1;
            let mut boxes = vec![[0, 0, 0], [top, top, top], [top, 0, top]];
            boxes.extend((0..5).map(|_| [0; 3].map(|_| (rng.next_u64() as u32) & top)));
            boxes
        })
        .collect();
    let boxes = BoxCoordinates::upload(device, &levels).unwrap();
    assert_eq!(boxes.nlevels(), levels.len());
    assert_eq!(boxes.offset(3), 3 * 8);
    assert_eq!(boxes.download(device).unwrap(), levels);

    let leaves: Vec<(u32, [u32; 3])> = levels
        .iter()
        .enumerate()
        .flat_map(|(l, b)| b.iter().map(move |&i| (l as u32, i)))
        .collect();
    let coordinates = LeafCoordinates::upload(device, &leaves).unwrap();
    assert_eq!(coordinates.len(), leaves.len());
    assert_eq!(coordinates.download(device).unwrap(), leaves);
    let empty = LeafCoordinates::upload(device, &[]).unwrap();
    assert!(empty.download(device).unwrap().is_empty());
}

/// A gather with the CPU units capped at 1, 3 and the default gives the same bits, and
/// the cap is reported as set (clamped to 1–16).
fn units_cap_changes_no_result(device: &mut Device) {
    let n = 81;
    let columns = 400;
    let x: Vec<f32> = (0..n * columns).map(|i| i as f32 * 0.5).collect();
    let indices: Vec<u32> = (0..columns as u32).rev().collect();
    let xd = device.upload(&x).unwrap();
    let id = device.upload_indices(&indices).unwrap();
    let mut outputs = Vec::new();
    for cap in [1, 3, CPU_MAX_UNITS, 0, 100] {
        device.limit_units(cap);
        assert_eq!(device.units_cap(), cap.clamp(1, CPU_MAX_UNITS));
        let mut y = device.alloc::<f32>(n * columns).unwrap();
        gather_columns(device, n, xd.as_slice(), id.as_slice(), y.as_slice_mut()).unwrap();
        let mut out = vec![0.0f32; n * columns];
        device.download(y.as_slice(), &mut out).unwrap();
        outputs.push(out);
    }
    device.limit_units(CPU_MAX_UNITS);
    let want: Vec<f32> = indices
        .iter()
        .flat_map(|&c| x[c as usize * n..(c as usize + 1) * n].iter().copied())
        .collect();
    for out in &outputs {
        assert_eq!(out, &want);
    }
}

/// The available memory, where the backend reports a limit, lies below it and shrinks
/// by at least a buffer's bytes while the buffer lives.
fn available_memory_shrinks(device: &mut Device) {
    let Some(limit) = device.info().max_memory else {
        println!("  the backend reports no memory limit");
        return;
    };
    let before = device.available_memory().unwrap();
    assert!(before <= limit);
    let len = 1 << 22;
    let buffer = device.alloc::<f32>(len).unwrap();
    let during = device.available_memory().unwrap();
    assert!(
        during + Device::buffer_bytes::<f32>(len) <= before,
        "{during} + {} > {before}",
        Device::buffer_bytes::<f32>(len)
    );
    drop(buffer);
    println!("  available memory {before} B of {limit} B");
}

tests_on!(cpu: views_round_trip, coordinates_round_trip, units_cap_changes_no_result, available_memory_shrinks);
tests_on!(metal: views_round_trip, coordinates_round_trip, units_cap_changes_no_result, available_memory_shrinks);
