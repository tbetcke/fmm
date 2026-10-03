//! Phase 0 / T6 spike: CubeCL GEMM throughput for dense M2L shapes.
//!
//! C = A X with A of shape Nc x Nc, Nc = (p+1)^2, and X of shape Nc x B. Every result is
//! checked against a plain-Rust f64 reference (relative 1e-12 for f64, 1e-5 for f32).
//! See docs/phase0/T6-cubecl-gemm-spike.md and SPIKE_REPORT.md.
//!
//! Usage: `cargo run -p nd-fmm-spike-cubecl-gemm --release -- [options]`
//!
//! - `--backends metal,cpu,cuda` (default: every backend compiled in)
//! - `--p 4,8,12,16` and `--b 1000,10000,100000` (defaults as shown)
//! - `--precisions f32,f64` (default: both; skipped where the device lacks the type)
//! - `--quick` for short timing runs
//! - `--no-lib` to skip the library matmul (its f32 kernels for B = 1e4 did not finish
//!   compiling within minutes on the CubeCL CPU runtime)
//! - `--force-f64` to run f64 even where the runtime does not register the type (CubeCL
//!   0.10 disabled f64 on CUDA; 0.11.0-pre.4 registers it again; see SPIKE_REPORT.md)
//!
//! The first launch of every implementation (compilation included) is timed and printed
//! to stderr.

mod bench;
mod reference;
mod tiled;

use std::io::Write;
use std::time::Duration;

use cubecl::prelude::*;

use bench::{Row, Settings, run_case};
use reference::Real;

struct Options {
    backends: Vec<String>,
    ps: Vec<usize>,
    bs: Vec<usize>,
    precisions: Vec<String>,
    force_f64: bool,
    no_lib: bool,
    settings: Settings,
}

fn list<T: std::str::FromStr>(value: Option<String>, flag: &str) -> Vec<T> {
    value
        .unwrap_or_else(|| panic!("{flag} needs a value"))
        .split(',')
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| panic!("bad value {s:?} for {flag}"))
        })
        .collect()
}

fn parse_options() -> Options {
    let compiled: Vec<String> = [
        (cfg!(feature = "metal"), "metal"),
        (cfg!(feature = "cpu"), "cpu"),
        (cfg!(feature = "cuda"), "cuda"),
    ]
    .iter()
    .filter(|(on, _)| *on)
    .map(|(_, name)| name.to_string())
    .collect();
    let mut options = Options {
        backends: compiled,
        ps: vec![4, 8, 12, 16],
        bs: vec![1_000, 10_000, 100_000],
        precisions: vec!["f32".into(), "f64".into()],
        force_f64: false,
        no_lib: false,
        settings: Settings {
            target: Duration::from_millis(1000),
            min_reps: 5,
            max_reps: 50,
            batch: Duration::from_millis(50),
            max_batch: 1024,
        },
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--backends" => options.backends = list(args.next(), &flag),
            "--p" => options.ps = list(args.next(), &flag),
            "--b" => options.bs = list(args.next(), &flag),
            "--precisions" => options.precisions = list(args.next(), &flag),
            "--force-f64" => options.force_f64 = true,
            "--no-lib" => options.no_lib = true,
            "--quick" => {
                options.settings = Settings {
                    target: Duration::from_millis(200),
                    min_reps: 3,
                    max_reps: 20,
                    batch: Duration::from_millis(20),
                    max_batch: 256,
                }
            }
            _ => panic!("unknown option {flag}"),
        }
    }
    options
}

fn print_header() {
    println!(
        "| backend | precision | p | Nc | B | implementation | median ms | reps | GFLOP/s | rel. error | check |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
}

fn print_row(row: &Row) -> bool {
    let nc = (row.p + 1) * (row.p + 1);
    let prefix = format!(
        "| {} | {} | {} | {} | {} | {} |",
        row.backend, row.precision, row.p, nc, row.b, row.implementation
    );
    let passed = match &row.outcome {
        Ok(m) => {
            println!(
                "{prefix} {:.4} | {} | {:.1} | {:.2e} | {} |",
                m.median.as_secs_f64() * 1e3,
                m.reps,
                m.gflops,
                m.rel_error,
                if m.passed { "pass" } else { "FAIL" }
            );
            m.passed
        }
        Err(e) => {
            let e: String = e.lines().next().unwrap_or("").chars().take(80).collect();
            println!("{prefix} – | – | – | – | n/a: {} |", e.replace('|', "/"));
            true
        }
    };
    std::io::stdout().flush().ok();
    passed
}

/// Prints the device facts the report needs, and whether f64 is supported.
fn describe(client: &Client, backend: &str) -> bool {
    let props = client.properties();
    let f64_ok = props.supports_type(f64::elem_type_native());
    let hw = &props.hardware;
    eprintln!(
        "[{backend}] runtime {}: f64 supported = {f64_ok}, plane size {}..{}, \
         max shared memory {} B, SMs {:?}, CPU cores {:?}, tensor-core min dim {:?}",
        client.name(),
        hw.plane_size_min,
        hw.plane_size_max,
        hw.max_shared_memory_size,
        hw.num_streaming_multiprocessors,
        hw.num_cpu_cores,
        hw.min_tensor_cores_dim,
    );
    f64_ok
}

fn run_precision<F: Real>(
    client: &Client,
    backend: &'static str,
    options: &Options,
    failures: &mut usize,
) {
    for &p in &options.ps {
        for &b in &options.bs {
            eprintln!("[{backend}] {} p = {p}, B = {b}", F::NAME);
            for row in run_case::<F>(client, backend, p, b, !options.no_lib, options.settings) {
                if !print_row(&row) {
                    *failures += 1;
                }
            }
        }
    }
}

fn run_backend(client: Client, backend: &'static str, options: &Options, failures: &mut usize) {
    let f64_ok = describe(&client, backend) || options.force_f64;
    for precision in &options.precisions {
        match precision.as_str() {
            "f32" => run_precision::<f32>(&client, backend, options, failures),
            "f64" if f64_ok => run_precision::<f64>(&client, backend, options, failures),
            "f64" => eprintln!("[{backend}] f64 not supported by the device: skipped"),
            other => panic!("unknown precision {other}"),
        }
    }
}

fn main() {
    // The CubeCL CPU runtime's default 64 MB worker stack overflows at p = 16, B = 1e5.
    if std::env::var_os("CUBECL_CPU_STACK_MB").is_none() {
        // SAFETY: no other thread exists yet, so nothing reads the environment concurrently.
        unsafe { std::env::set_var("CUBECL_CPU_STACK_MB", "1024") };
    }
    let options = parse_options();
    let mut failures = 0;
    print_header();
    for backend in options.backends.clone() {
        match backend.as_str() {
            #[cfg(feature = "metal")]
            "metal" => {
                use cubecl::{Device, device::WgpuDeviceKind};
                let client = Device::metal_msl(WgpuDeviceKind::DefaultDevice)
                    .expect("Metal device")
                    .client();
                run_backend(client, "metal", &options, &mut failures);
            }
            #[cfg(feature = "cpu")]
            "cpu" => {
                use cubecl::{Device, device::CpuDevice};
                let client = Device::Cpu(CpuDevice).client();
                run_backend(client, "cpu", &options, &mut failures);
            }
            #[cfg(feature = "cuda")]
            "cuda" => {
                use cubecl::{Device, device::CudaDevice};
                let client = Device::Cuda(CudaDevice::default()).client();
                run_backend(client, "cuda", &options, &mut failures);
            }
            other => panic!("backend {other} is not compiled in (see the crate features)"),
        }
    }
    if failures > 0 {
        eprintln!("{failures} result(s) failed the accuracy check");
        std::process::exit(1);
    }
}
