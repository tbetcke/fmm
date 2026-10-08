//! Phase 4S T11 spike: the download and upload paths of a device evaluation, step by step.
//! Prints Markdown on stdout and progress on stderr. See docs/phase4s/T11-download-path.md
//! and REPORT.md.
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
//!     VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-spike-download-path --features metal -- \
//!     [--backends metal,cuda,cpu] [--precisions f32,f64] \
//!     [--sections download,pool,upload,zero,device] [--sizes 16,160,320] \
//!     [--upload-sizes 4,8,40,80,160] [--repeats 10] [--warmup 2] \
//!     [--pool-sequences steady,mixed,evaluation,persistent] [--pool-iterations 10]
//! ```
//!
//! Sizes are in MB (10⁶ bytes). Sections:
//! - `download`: CubeCL's read path split into the wait for queued work (a sync first),
//!   `Client::read_one` (the device-to-host copy, the host pool, the mapping), the copy
//!   into a host slice on one thread (warm, as into a reused buffer; and cold, into a
//!   fresh `vec!`, page faults included), the same copy in parallel on rayon's pool, and
//!   dropping CubeCL's `Bytes`; and `read_one` plus the serial copy without the sync
//!   first, which is what `nd_fmm_kernels::Device::download` does;
//! - `pool`: `read_one` per download over a sequence of downloads, to see whether the
//!   host pool (CUDA's pinned pool, wgpu's staging pool) allocates in every download or
//!   only on growth. Run it in a process of its own (`--sections pool`) so that the other
//!   sections have not grown the pools; under `nsys` on CUDA, `cuMemAllocHost` counts;
//! - `upload`: `Device::write`'s path split into the `to_vec` copy, `Bytes::from_elems`,
//!   `Client::write` (to its return) and the transfer (to a following sync), and the same
//!   without `to_vec` (an owned buffer handed over);
//! - `zero`: the three zero kernels of `DeviceOperator::begin_evaluation` (multipoles,
//!   locals, target output) at the benchmark's sizes, by timing windows (CUDA events,
//!   Metal timestamps; the host clock on the CPU runtime);
//! - `device`: `nd_fmm_kernels::Device::write` and `Device::download` end to end, with
//!   their counters, the production path the other sections take apart.
//!
//! Metal needs a process with GPU access (outside the macOS sandbox). CUDA runs on locust.
//! Timings are reported, never asserted.

use std::time::Instant;

use cubecl::bytes::Bytes;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;
use nd_fmm_kernels::{BackendKind, Device, DeviceFloat, movement};
use rayon::prelude::*;

/// x[i] = i + seed (as E) for i < len, grid-stride.
#[cube(launch_unchecked)]
fn fill_kernel<E: Float>(x: &mut [E], seed: u32, len: u32, threads: u32) {
    let len = len as usize;
    let mut i = ABSOLUTE_POS;
    while i < len {
        x[i] = E::cast_from(i as u32 + seed);
        i += threads as usize;
    }
}

/// A precision of the spike.
trait Elem: DeviceFloat + Default + std::fmt::Debug {
    const NAME: &'static str;
    /// The value the fill kernel writes for `v` = i + seed (exact below 2²⁴ in f32).
    fn of(v: u32) -> Self;
}

impl Elem for f32 {
    const NAME: &'static str = "f32";
    fn of(v: u32) -> Self {
        v as f32
    }
}

impl Elem for f64 {
    const NAME: &'static str = "f64";
    fn of(v: u32) -> Self {
        f64::from(v)
    }
}

/// A CubeCL backend compiled into the spike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cpu,
    Metal,
    Cuda,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    fn parse(name: &str) -> Self {
        [Self::Cpu, Self::Metal, Self::Cuda]
            .into_iter()
            .find(|b| b.name() == name)
            .unwrap_or_else(|| panic!("unknown backend {name}"))
    }

    fn kind(self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::Cpu,
            Self::Metal => BackendKind::Metal,
            Self::Cuda => BackendKind::Cuda,
        }
    }

    /// A CubeCL client of the backend's default device, as `Device::open` makes it.
    fn client(self) -> Client {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu => cubecl::Device::cpu().expect("CPU runtime").client(),
            #[cfg(feature = "metal")]
            Self::Metal => cubecl::Device::metal_msl(cubecl::device::WgpuDeviceKind::DefaultDevice)
                .expect("Metal device (wgpu, MSL)")
                .client(),
            #[cfg(feature = "cuda")]
            Self::Cuda => cubecl::Device::cuda(0).expect("CUDA device 0").client(),
            #[allow(unreachable_patterns)]
            other => panic!("backend {} is not compiled in", other.name()),
        }
    }

    fn is_gpu(self) -> bool {
        self != Self::Cpu
    }
}

/// Command-line options.
struct Options {
    backends: Vec<Backend>,
    precisions: Vec<String>,
    sections: Vec<String>,
    sizes: Vec<f64>,
    upload_sizes: Vec<f64>,
    repeats: usize,
    warmup: usize,
    pool_sequences: Vec<String>,
    pool_iterations: usize,
}

fn list(value: &str) -> Vec<String> {
    value.split(',').map(str::to_string).collect()
}

fn numbers(value: &str) -> Vec<f64> {
    value
        .split(',')
        .map(|v| v.parse().expect("a number"))
        .collect()
}

fn parse() -> Options {
    let mut o = Options {
        backends: Vec::new(),
        precisions: list("f32,f64"),
        sections: list("download,upload,zero,device"),
        sizes: vec![16.0, 160.0, 320.0],
        upload_sizes: vec![4.0, 8.0, 40.0, 80.0, 160.0],
        repeats: 10,
        warmup: 2,
        pool_sequences: list("steady,mixed,evaluation,persistent"),
        pool_iterations: 10,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().expect("a value after the flag").as_str();
        match flag.as_str() {
            "--backends" => o.backends = list(value()).iter().map(|b| Backend::parse(b)).collect(),
            "--precisions" => o.precisions = list(value()),
            "--sections" => o.sections = list(value()),
            "--sizes" => o.sizes = numbers(value()),
            "--upload-sizes" => o.upload_sizes = numbers(value()),
            "--repeats" => o.repeats = value().parse().expect("repeats"),
            "--warmup" => o.warmup = value().parse().expect("warmup"),
            "--pool-sequences" => o.pool_sequences = list(value()),
            "--pool-iterations" => o.pool_iterations = value().parse().expect("iterations"),
            other => panic!("unknown flag {other}"),
        }
    }
    if o.backends.is_empty() {
        o.backends = [Backend::Metal, Backend::Cuda, Backend::Cpu]
            .into_iter()
            .filter(|b| b.kind().is_compiled())
            .collect();
    }
    o
}

/// Milliseconds since `t`.
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Times in ms over the timed repeats.
#[derive(Default, Clone)]
struct Samples(Vec<f64>);

impl Samples {
    fn push(&mut self, v: f64) {
        self.0.push(v);
    }

    fn sorted(&self) -> Vec<f64> {
        let mut v = self.0.clone();
        v.sort_by(f64::total_cmp);
        v
    }

    fn median(&self) -> f64 {
        let v = self.sorted();
        if v.is_empty() {
            return f64::NAN;
        }
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    }

    fn max(&self) -> f64 {
        self.sorted().last().copied().unwrap_or(f64::NAN)
    }

    /// "median (max)", in ms.
    fn cell(&self) -> String {
        format!("{:.3} ({:.3})", self.median(), self.max())
    }
}

/// GB/s for `bytes` in `ms`.
fn rate(bytes: usize, ms: f64) -> String {
    format!("{:.1}", bytes as f64 / ms / 1e6)
}

/// Elements of `E` in `mb` MB.
fn elements<E: Elem>(mb: f64) -> usize {
    (mb * 1e6) as usize / size_of::<E>()
}

/// Queues the fill kernel over `len` elements of `handle`.
fn fill<E: Elem>(backend: Backend, client: &Client, handle: &Handle, len: usize, seed: u32) {
    let (units, cubes) = if backend.is_gpu() {
        (256u32, len.div_ceil(256).min(65_535) as u32)
    } else {
        (16, 1)
    };
    // SAFETY: `handle` holds `len` elements of E (allocated with that many bytes), and the
    // kernel writes x[i] for i < len only.
    unsafe {
        fill_kernel::launch_unchecked::<E>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(handle.clone(), len),
            seed,
            len as u32,
            units * cubes,
        );
    }
}

fn sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("device sync");
}

/// Copies `src` into `dst` on rayon's pool, in chunks of 2¹⁶ elements.
fn par_copy<E: Copy + Send + Sync>(dst: &mut [E], src: &[E]) {
    dst.par_chunks_mut(1 << 16)
        .zip(src.par_chunks(1 << 16))
        .for_each(|(d, s)| d.copy_from_slice(s));
}

/// Checks that `values` holds the fill of `seed` at a few indices.
fn check_fill<E: Elem>(values: &[E], seed: u32) {
    for k in [0, 1, 7, values.len() / 3, values.len() - 1] {
        let v = k as u32 + seed;
        if v < 1 << 24 {
            assert_eq!(values[k], E::of(v), "element {k} of the download");
        }
    }
}

fn download_section<E: Elem>(backend: Backend, o: &Options) {
    let client = backend.client();
    println!("\n### Download, {}, {}\n", backend.name(), E::NAME);
    println!(
        "Median (max) over {} downloads after {} warm-ups, ms. A fill kernel writes the \
         buffer before each download. `sync`: the wait for it; `read_one`: \
         `Client::read_one` after that sync; `copy`: `copy_from_slice` into a reused, \
         touched host slice (one thread); `par copy`: the same on rayon's pool ({} \
         threads); `cold copy`: a fresh `vec!` and the copy into it; `drop`: dropping \
         CubeCL's `Bytes`; `unsynced`: `read_one` and the copy without the sync first, \
         as `Device::download` does it, the fill's wait included.\n",
        o.repeats,
        o.warmup,
        rayon::current_num_threads()
    );
    println!(
        "| MB | sync | read_one | copy | par copy | cold copy | drop | unsynced | read_one GB/s | copy GB/s | par copy GB/s |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for &mb in &o.sizes {
        eprintln!("download {} {} {mb} MB", backend.name(), E::NAME);
        let len = elements::<E>(mb);
        let bytes = len * size_of::<E>();
        let handle = client.empty(bytes);
        let mut warm = vec![E::of(1); len];
        let mut warm_par = vec![E::of(2); len];
        let (mut s_sync, mut s_read, mut s_copy, mut s_par, mut s_cold, mut s_drop, mut s_un) = (
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
        );
        for i in 0..o.warmup + o.repeats {
            let seed = i as u32;
            fill::<E>(backend, &client, &handle, len, seed);
            let t = Instant::now();
            sync(&client);
            let t_sync = ms(t);
            let t = Instant::now();
            let data = client.read_one(handle.clone()).expect("read_one");
            let t_read = ms(t);
            let src = E::from_bytes(&data);
            let t = Instant::now();
            warm.copy_from_slice(src);
            let t_copy = ms(t);
            let t = Instant::now();
            par_copy(&mut warm_par, src);
            let t_par = ms(t);
            let t = Instant::now();
            let mut cold = vec![E::default(); len];
            cold.copy_from_slice(src);
            let t_cold = ms(t);
            check_fill(&warm, seed);
            assert!(warm == warm_par && warm == cold, "the copies differ");
            drop(cold);
            let t = Instant::now();
            drop(data);
            let t_drop = ms(t);

            // As `Device::download`: read_one at once, then the serial copy.
            fill::<E>(backend, &client, &handle, len, seed + 1);
            let t = Instant::now();
            let data = client.read_one(handle.clone()).expect("read_one");
            warm.copy_from_slice(E::from_bytes(&data));
            let t_un = ms(t);
            drop(data);
            check_fill(&warm, seed + 1);

            if i >= o.warmup {
                s_sync.push(t_sync);
                s_read.push(t_read);
                s_copy.push(t_copy);
                s_par.push(t_par);
                s_cold.push(t_cold);
                s_drop.push(t_drop);
                s_un.push(t_un);
            }
        }
        println!(
            "| {mb} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            s_sync.cell(),
            s_read.cell(),
            s_copy.cell(),
            s_par.cell(),
            s_cold.cell(),
            s_drop.cell(),
            s_un.cell(),
            rate(bytes, s_read.median()),
            rate(bytes, s_copy.median()),
            rate(bytes, s_par.median()),
        );
    }
}

fn pool_section<E: Elem>(backend: Backend, o: &Options) {
    let client = backend.client();
    // The output (4 N values) and the charges (N values) of N = 10⁷.
    let big = (4e7 * size_of::<E>() as f64) / 1e6;
    let charges_mb = big / 4.0;
    println!("\n### Host pool, {}, {}\n", backend.name(), E::NAME);
    println!(
        "`read_one` per download in sequence (ms), after a fill and a sync, in a process \
         that ran the sections {} in that order. `steady`: {} downloads of {big} MB (the \
         output of N = 10⁷ with gradients); `mixed`: the sizes {:?} MB in turn; \
         `evaluation`: per iteration a `Client::write` of {charges_mb} MB (the charges of \
         N = 10⁷), a fill and a download of {big} MB; `persistent`: `steady` with each \
         `read_one` inside `Client::memory_persistent_allocation`.\n",
        o.sections.join(", "),
        o.pool_iterations,
        o.sizes,
    );
    let max_mb = o.sizes.iter().copied().fold(big, f64::max);
    let max_len = elements::<E>(max_mb);
    let handle = client.empty(max_len * size_of::<E>());
    let upload = client.empty(elements::<E>(charges_mb) * size_of::<E>());
    let charges = vec![E::of(3); elements::<E>(charges_mb)];
    for sequence in &o.pool_sequences {
        eprintln!("pool {} {} {sequence}", backend.name(), E::NAME);
        let mut cells = Vec::new();
        for i in 0..o.pool_iterations {
            let mb = match sequence.as_str() {
                "mixed" => o.sizes[i % o.sizes.len()],
                _ => big,
            };
            let len = elements::<E>(mb);
            let range = handle
                .clone()
                .offset_end(((max_len - len) * size_of::<E>()) as u64);
            if sequence == "evaluation" {
                client.write(&upload, Bytes::from_elems(charges.clone()));
            }
            fill::<E>(backend, &client, &handle, len, i as u32);
            sync(&client);
            let t = Instant::now();
            let data = if sequence == "persistent" {
                client.memory_persistent_allocation(range, |range| client.read_one(range))
            } else {
                client.read_one(range)
            }
            .expect("read_one");
            let t_read = ms(t);
            assert_eq!(data.len(), len * size_of::<E>());
            check_fill(E::from_bytes(&data), i as u32);
            drop(data);
            cells.push(if sequence == "mixed" {
                format!("{t_read:.2} ({mb})")
            } else {
                format!("{t_read:.2}")
            });
        }
        println!("- {sequence}: {}", cells.join(", "));
    }
}

fn upload_section<E: Elem>(backend: Backend, o: &Options) {
    let client = backend.client();
    println!("\n### Upload, {}, {}\n", backend.name(), E::NAME);
    println!(
        "Median (max) over {} uploads after {} warm-ups, ms, each from an idle stream. \
         `to_vec`: `Device::write`'s copy of the caller's slice; `from_elems`: \
         `Bytes::from_elems` of that `Vec`; `write`: `Client::write` until it returns; \
         `sync`: the transfer, to a following sync; `owned`: `from_elems`, `write` and \
         `sync` of a `Vec` made before the timing (no `to_vec`). Each path's device \
         contents are read back once and compared bit for bit with the source.\n",
        o.repeats, o.warmup
    );
    println!("| MB | to_vec | from_elems | write | sync | total | owned | to_vec GB/s |");
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for &mb in &o.upload_sizes {
        eprintln!("upload {} {} {mb} MB", backend.name(), E::NAME);
        let len = elements::<E>(mb);
        let bytes = len * size_of::<E>();
        let handle = client.empty(bytes);
        let data: Vec<E> = (0..len).map(|i| E::of(i as u32 & 0xff_ffff)).collect();
        let (mut s_vec, mut s_from, mut s_write, mut s_sync, mut s_total, mut s_owned) = (
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
            Samples::default(),
        );
        for i in 0..o.warmup + o.repeats {
            sync(&client);
            let t0 = Instant::now();
            let t = Instant::now();
            let v = data.to_vec();
            let t_vec = ms(t);
            let t = Instant::now();
            let b = Bytes::from_elems(v);
            let t_from = ms(t);
            let t = Instant::now();
            client.write(&handle, b);
            let t_write = ms(t);
            let t = Instant::now();
            sync(&client);
            let t_sync = ms(t);
            let t_total = ms(t0);
            if i == 0 {
                let back = client.read_one(handle.clone()).expect("read_one");
                assert!(E::from_bytes(&back) == data.as_slice(), "write differs");
            }

            let owned = data.clone();
            sync(&client);
            let t = Instant::now();
            client.write(&handle, Bytes::from_elems(owned));
            sync(&client);
            let t_owned = ms(t);
            if i == 0 {
                let back = client.read_one(handle.clone()).expect("read_one");
                assert!(
                    E::from_bytes(&back) == data.as_slice(),
                    "owned write differs"
                );
            }
            if i >= o.warmup {
                s_vec.push(t_vec);
                s_from.push(t_from);
                s_write.push(t_write);
                s_sync.push(t_sync);
                s_total.push(t_total);
                s_owned.push(t_owned);
            }
        }
        println!(
            "| {mb} | {} | {} | {} | {} | {} | {} | {} |",
            s_vec.cell(),
            s_from.cell(),
            s_write.cell(),
            s_sync.cell(),
            s_total.cell(),
            s_owned.cell(),
            rate(bytes, s_vec.median()),
        );
    }
}

/// The benchmark's boxes on all levels (the unit cube, 64 points per leaf), from the
/// level calls of T9's reports: 37,449 at N = 10⁶ (levels 0–5) and 299,673 at N = 10⁷
/// (levels 0–6 full, 80 boxes on level 7).
const BOXES: [(usize, usize); 2] = [(1_000_000, 37_449), (10_000_000, 299_673)];

fn zero_section<E: Elem>(backend: Backend, o: &Options) {
    let mut device = Device::open(backend.kind()).expect("open the device");
    println!("\n### Zeroing, {}, {}\n", backend.name(), E::NAME);
    println!(
        "The three zero kernels of `begin_evaluation` at the benchmark's sizes (gradients \
         on: the target output 4 N values; multipoles and locals (p + 1)² values per box, \
         every level), median (max) over {} evaluations after {} warm-ups, ms, by timing \
         windows ({}): one window per kernel, then one over the three.\n",
        o.repeats,
        o.warmup,
        if device.times_on_device() {
            "on the device"
        } else {
            "the host clock between two syncs"
        },
    );
    println!(
        "| N | p | multipoles MB | output MB | multipoles | locals | output | all three | GB/s (all) |"
    );
    println!("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    for (n, boxes) in BOXES {
        for p in [3usize, 8] {
            eprintln!("zero {} {} N {n} p {p}", backend.name(), E::NAME);
            let coeffs = boxes * (p + 1) * (p + 1);
            let mut multipoles = device.alloc::<E>(coeffs).expect("alloc");
            let mut locals = device.alloc::<E>(coeffs).expect("alloc");
            let mut output = device.alloc::<E>(4 * n).expect("alloc");
            let (mut s_m, mut s_l, mut s_o, mut s_all) = (
                Samples::default(),
                Samples::default(),
                Samples::default(),
                Samples::default(),
            );
            for i in 0..o.warmup + o.repeats {
                let mut times = Vec::new();
                for buffer in [&mut multipoles, &mut locals, &mut output] {
                    let w = device.open_window().expect("window");
                    movement::zero(&mut device, buffer.as_slice_mut()).expect("zero");
                    times.push(device.close_window(w).expect("window"));
                }
                let w = device.open_window().expect("window");
                for buffer in [&mut multipoles, &mut locals, &mut output] {
                    movement::zero(&mut device, buffer.as_slice_mut()).expect("zero");
                }
                times.push(device.close_window(w).expect("window"));
                device.sync().expect("sync");
                let t: Vec<f64> = times
                    .into_iter()
                    .map(|w| w.resolve().map_or(f64::NAN, |d| d.as_secs_f64() * 1e3))
                    .collect();
                if i >= o.warmup {
                    s_m.push(t[0]);
                    s_l.push(t[1]);
                    s_o.push(t[2]);
                    s_all.push(t[3]);
                }
            }
            let bytes = (2 * coeffs + 4 * n) * size_of::<E>();
            println!(
                "| {n} | {p} | {:.1} | {:.1} | {} | {} | {} | {} | {} |",
                (coeffs * size_of::<E>()) as f64 / 1e6,
                (4 * n * size_of::<E>()) as f64 / 1e6,
                s_m.cell(),
                s_l.cell(),
                s_o.cell(),
                s_all.cell(),
                rate(bytes, s_all.median()),
            );
        }
    }
}

fn device_section<E: Elem>(backend: Backend, o: &Options) {
    let mut device = Device::open(backend.kind()).expect("open the device");
    println!(
        "\n### `Device::write` and `Device::download`, {}, {}\n",
        backend.name(),
        E::NAME
    );
    println!(
        "The production path, median (max) over {} calls after {} warm-ups, ms. `write`: \
         `Device::write` and a `Device::sync`; `download`: `Device::download` from an \
         idle stream (its own sync included) into a reused slice; counters per call.\n",
        o.repeats, o.warmup
    );
    println!(
        "| MB | write + sync | download | download GB/s | counters (write) | counters (download) |"
    );
    println!("| ---: | ---: | ---: | ---: | --- | --- |");
    let mut sizes = o.sizes.clone();
    sizes.extend(
        o.upload_sizes
            .iter()
            .copied()
            .filter(|s| !o.sizes.contains(s)),
    );
    sizes.sort_by(f64::total_cmp);
    for mb in sizes {
        eprintln!("device {} {} {mb} MB", backend.name(), E::NAME);
        let len = elements::<E>(mb);
        let data: Vec<E> = (0..len).map(|i| E::of(i as u32 & 0xff_ffff)).collect();
        let mut out = vec![E::of(1); len];
        let mut buffer = device.alloc::<E>(len).expect("alloc");
        let (mut s_w, mut s_d) = (Samples::default(), Samples::default());
        let (mut c_w, mut c_d) = Default::default();
        for i in 0..o.warmup + o.repeats {
            device.sync().expect("sync");
            device.reset_counters();
            let t = Instant::now();
            device.write(buffer.as_slice_mut(), &data).expect("write");
            device.sync().expect("sync");
            let t_w = ms(t);
            c_w = device.counters();
            device.sync().expect("sync");
            device.reset_counters();
            let t = Instant::now();
            device
                .download(buffer.as_slice(), &mut out)
                .expect("download");
            let t_d = ms(t);
            c_d = device.counters();
            assert!(out == data, "download differs from the write");
            out.fill(E::of(1));
            if i >= o.warmup {
                s_w.push(t_w);
                s_d.push(t_d);
            }
        }
        let counters = |c: nd_fmm_kernels::Counters| {
            format!(
                "up {} ({} B), down {} ({} B), syncs {}",
                c.uploads, c.upload_bytes, c.downloads, c.download_bytes, c.syncs
            )
        };
        println!(
            "| {mb} | {} | {} | {} | {} | {} |",
            s_w.cell(),
            s_d.cell(),
            rate(len * size_of::<E>(), s_d.median()),
            counters(c_w),
            counters(c_d),
        );
    }
}

fn run<E: Elem>(backend: Backend, o: &Options) {
    for section in &o.sections {
        match section.as_str() {
            "download" => download_section::<E>(backend, o),
            "pool" => pool_section::<E>(backend, o),
            "upload" => upload_section::<E>(backend, o),
            "zero" => zero_section::<E>(backend, o),
            "device" => device_section::<E>(backend, o),
            other => panic!("unknown section {other}"),
        }
    }
}

fn main() {
    let o = parse();
    use nd_fmm_validate::bench::{cores, cpu_model, target, toolchain};
    println!("## Download and upload paths (Phase 4S T11 spike)\n");
    println!(
        "- Host: {} ({}), {}, {}; rayon threads {}.",
        cpu_model(),
        cores(),
        target(),
        toolchain(),
        rayon::current_num_threads()
    );
    println!(
        "- Arguments: {}.",
        std::env::args().skip(1).collect::<Vec<_>>().join(" ")
    );
    for &backend in &o.backends {
        let device = Device::open(backend.kind()).expect("open the device");
        println!("- Device: {}.", device.info());
        drop(device);
        for precision in &o.precisions {
            match precision.as_str() {
                "f32" => run::<f32>(backend, &o),
                "f64" if backend == Backend::Metal => {
                    println!("\n(f64 not run on Metal: no f64 arithmetic.)");
                }
                "f64" => run::<f64>(backend, &o),
                other => panic!("unknown precision {other}"),
            }
        }
    }
}
