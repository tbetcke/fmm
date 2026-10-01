//! Measure the time and memory of building the index-form plan against the old
//! interaction manager, on one rank.
//!
//! Reports, per tree, the best of several builds and the heap memory: the peak during the
//! build and what the result retains, both counted by a wrapping allocator. Nothing is
//! asserted. Run in release mode:
//! `cargo run --release -p nd-fmm-plan --example plan_build_cost`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nd_fmm_plan::{interaction_manager::InteractionManager, v2::plan::Plan};
use nd_octree::{Octree, OctreeOptions, PhysicalBox, constants::DEEPEST_LEVEL, points_to_morton};
use rand_chacha::{ChaCha8Rng, rand_core::SeedableRng};
use rlst::rlst_dynamic_array;

/// The system allocator, counting the bytes in use and their peak.
struct Counting;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let current = CURRENT.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(current, Ordering::Relaxed);
}

// SAFETY: every call is forwarded to `System` unchanged; only the counters are added.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grow(new_size);
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Build `f` `repeats` times; return the best time in ms, the peak heap growth during one
/// build and the heap retained by its result, both in MB.
fn measure<T>(repeats: usize, mut f: impl FnMut() -> T) -> (f64, f64, f64) {
    let mut best = f64::INFINITY;
    for _ in 0..repeats {
        let start = Instant::now();
        let result = f();
        best = best.min(start.elapsed().as_secs_f64() * 1e3);
        drop(result);
    }
    let before = CURRENT.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let result = f();
    let peak = PEAK.load(Ordering::Relaxed) - before;
    let retained = CURRENT.load(Ordering::Relaxed) - before;
    drop(result);
    let mb = |bytes: usize| bytes as f64 / 1e6;
    (best, mb(peak), mb(retained))
}

fn keys(points: &[[f64; 3]]) -> Vec<u64> {
    let mut array = rlst_dynamic_array!(f64, [3, points.len()]);
    for (j, point) in points.iter().enumerate() {
        for i in 0..3 {
            array[[i, j]] = point[i];
        }
    }
    points_to_morton(
        &array,
        DEEPEST_LEVEL as usize,
        &PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
    )
}

/// The sources and targets of the "graded corner blob" scenario of
/// `tests/mpi_regressions.rs` on rank 0.
fn graded_corner_blob() -> Vec<u64> {
    let cloud: Vec<[f64; 3]> = (0..48)
        .map(|j| {
            [
                ((13 * j + 7) % 97) as f64 / 97.0,
                ((29 * j + 3) % 89) as f64 / 89.0,
                ((11 * j + 5) % 83) as f64 / 83.0,
            ]
        })
        .collect();
    let corner = |n: usize| 0.005 + (n % 8) as f64 / 8.0;
    let origin = [corner(0), corner(1), corner(2)];
    let blob: Vec<[f64; 3]> = (0..64usize)
        .map(|i| {
            let coordinate = |k: usize, d: usize| origin[d] + (k % 4) as f64 / 64.0;
            [
                coordinate(i, 0),
                coordinate(i / 4, 1),
                coordinate(i / 16, 2),
            ]
        })
        .collect();
    let mut points: Vec<[f64; 3]> = cloud.iter().chain(&blob).copied().collect();
    points.extend(cloud.iter().rev().chain(&blob));
    keys(&points)
}

fn main() {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();

    let npoints = 100_000;
    let mut uniform = rlst_dynamic_array!(f64, [3, npoints]);
    uniform.fill_from_equally_distributed(&mut ChaCha8Rng::seed_from_u64(0));
    let bounding_box = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let uniform = points_to_morton(&uniform, DEEPEST_LEVEL as usize, &bounding_box);

    let cases = [
        ("graded corner blob", graded_corner_blob(), 6, 1),
        ("1e5 uniform, level 4", uniform.clone(), 4, 1),
        (
            "1e5 uniform, 32 per leaf",
            uniform,
            DEEPEST_LEVEL as usize,
            32,
        ),
    ];
    println!("| Tree | Leaves | Boxes held | Build | Time (ms) | Peak heap (MB) | Retained (MB) |");
    println!("| --- | --- | --- | --- | --- | --- | --- |");
    for (name, fine_keys, max_level, capacity) in cases {
        let options = OctreeOptions::new()
            .with_max_level(max_level)
            .with_max_fine_keys(capacity)
            .with_ghost_children(true);
        let octree = Octree::new(&fine_keys, options, &comm);
        let (nleaves, nboxes) = (octree.leaf_keys().len(), octree.all_keys().len());
        let repeats = 5;
        for (build, (time, peak, retained)) in [
            (
                "`InteractionManager::new`",
                measure(repeats, || InteractionManager::new(&octree)),
            ),
            (
                "`Plan::new`",
                measure(repeats, || Plan::new(&octree).unwrap()),
            ),
        ] {
            println!(
                "| {name} | {nleaves} | {nboxes} | {build} | {time:.2} | {peak:.2} | {retained:.2} |"
            );
        }
    }
}
