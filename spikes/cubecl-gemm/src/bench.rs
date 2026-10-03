//! Timing and correctness check of one (runtime, precision, p, B) case.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::{Shape, Strides};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::launch::launch_ref;
use cubek_matmul::multi_level;
use cubek_matmul::strategy::Strategy;
use cubek_std::InputBinding;

use crate::reference::{Real, gemm_f64, rel_max_error, uniform};
use crate::tiled;

/// One row of the result table.
pub struct Row {
    pub backend: &'static str,
    pub precision: &'static str,
    pub p: usize,
    pub b: usize,
    pub implementation: String,
    /// Timing and accuracy, or the reason the implementation did not run.
    pub outcome: Result<Measured, String>,
}

/// Timing and accuracy of an implementation that ran.
pub struct Measured {
    /// Median time per GEMM.
    pub median: Duration,
    /// Number of GEMMs timed.
    pub reps: usize,
    pub gflops: f64,
    pub rel_error: f64,
    pub passed: bool,
}

/// Benchmark settings.
#[derive(Clone, Copy)]
pub struct Settings {
    /// Keep timing batches until this much time has been spent (after warm-up).
    pub target: Duration,
    /// Minimum and maximum number of timed batches.
    pub min_reps: usize,
    pub max_reps: usize,
    /// Launches per batch are doubled until a batch takes at least this long.
    pub batch: Duration,
    /// Upper bound on launches per batch.
    pub max_batch: usize,
}

/// Library strategies tried for every case. `Auto` is what a caller gets by default;
/// the others show whether an explicit choice does better.
fn library_strategies() -> Vec<Strategy> {
    vec![
        Strategy::Auto,
        multi_level::Strategy::SimpleUnit(Default::default()).into(),
        multi_level::Strategy::DoubleUnit(Default::default()).into(),
        multi_level::Strategy::SimpleCyclicCmma(Default::default()).into(),
        multi_level::Strategy::DoubleCyclicCmma(Default::default()).into(),
    ]
}

/// Relative tolerance of the check against the f64 reference.
fn tolerance<F: Real>() -> f64 {
    if F::NAME == "f64" { 1e-12 } else { 1e-5 }
}

fn sync(client: &Client) -> Result<(), String> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))
}

fn message(panic: Box<dyn std::any::Any + Send>) -> String {
    panic
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "panic".into())
}

/// Time of `count` back-to-back launches followed by one sync.
fn time_batch(
    client: &Client,
    count: usize,
    launch: &mut impl FnMut() -> Result<(), String>,
) -> Result<Duration, String> {
    let t = Instant::now();
    for _ in 0..count {
        launch()?;
    }
    sync(client)?;
    Ok(t.elapsed())
}

/// Warms up, then returns the median time per launch and the number of launches timed.
///
/// The warm-up launch includes the kernel's compilation; its time is printed to stderr.
///
/// A sync costs about 1.5 ms on wgpu/Metal regardless of the kernel, so launches are
/// queued in batches (doubled until a batch takes `settings.batch`) with one sync per
/// batch. This matches an FMM pass, which queues many kernels between syncs.
fn time(
    client: &Client,
    settings: Settings,
    label: &str,
    mut launch: impl FnMut() -> Result<(), String>,
) -> Result<(Duration, usize), String> {
    let first = time_batch(client, 1, &mut launch)?;
    eprintln!(
        "  {label}: first launch (compilation included) {:.3} s",
        first.as_secs_f64()
    );
    let mut count = 1;
    while count < settings.max_batch && time_batch(client, count, &mut launch)? < settings.batch {
        count *= 2;
    }
    let mut times = Vec::new();
    let start = Instant::now();
    while times.len() < settings.min_reps
        || (start.elapsed() < settings.target && times.len() < settings.max_reps)
    {
        times.push(time_batch(client, count, &mut launch)? / count as u32);
    }
    times.sort();
    Ok((times[times.len() / 2], times.len() * count))
}

/// Runs every implementation for one case and checks it against the f64 reference.
pub fn run_case<F: Real>(
    client: &Client,
    backend: &'static str,
    p: usize,
    b: usize,
    with_library: bool,
    settings: Settings,
) -> Vec<Row> {
    let nc = (p + 1) * (p + 1);
    let a_host = uniform::<F>(nc * nc, 1 + p as u64);
    let x_host = uniform::<F>(nc * b, 1000 + b as u64);
    let c_ref = gemm_f64(&a_host, &x_host, nc, b);
    let a = client.create_from_slice(F::as_bytes(&a_host));
    let x = client.create_from_slice(F::as_bytes(&x_host));
    let flops = 2.0 * (nc * nc * b) as f64;
    let elem = F::elem_type_native();

    // Times a launch closure, turning panics (e.g. from `Strategy::Auto`) into errors.
    let timed = |label: &str, launch: &mut dyn FnMut() -> Result<(), String>| {
        catch_unwind(AssertUnwindSafe(|| time(client, settings, label, launch)))
            .map_err(message)
            .and_then(|r| r)
    };

    let mut rows = Vec::new();
    let mut record = |implementation: String,
                      outcome: Result<(Duration, usize, Handle), String>| {
        let outcome = outcome.and_then(|(median, reps, c)| {
            let bytes = client.read_one(c).map_err(|e| format!("{e:?}"))?;
            let rel_error = rel_max_error(F::from_bytes(&bytes), &c_ref);
            Ok(Measured {
                median,
                reps,
                gflops: flops / median.as_secs_f64() / 1e9,
                rel_error,
                passed: rel_error <= tolerance::<F>(),
            })
        });
        rows.push(Row {
            backend,
            precision: F::NAME,
            p,
            b,
            implementation,
            outcome,
        });
    };

    let strategies = if with_library {
        library_strategies()
    } else {
        Vec::new()
    };
    for strategy in strategies {
        let name = format!("lib:{strategy}");
        let c = client.empty(nc * b * size_of::<F>());
        let binding = |h: &Handle, rows: usize, cols: usize| unsafe {
            TensorBinding::from_raw_parts(
                h.clone(),
                Strides::from([cols, 1]),
                Shape::from([rows, cols]),
            )
        };
        let outcome = timed(&name, &mut || {
            let mut dtypes = MatmulElems::from_single_dtype(F::elem_type_native());
            launch_ref(
                &strategy,
                client,
                InputBinding::Normal(binding(&a, nc, nc), elem),
                InputBinding::Normal(binding(&x, nc, b), elem),
                binding(&c, nc, b),
                &mut dtypes,
            )
            .map_err(|e| format!("{e:?}"))
        })
        .map(|(median, reps)| (median, reps, c));
        record(name, outcome);
    }

    let smem_name = format!("tiled-smem(tm={})", tiled::rows_per_unit(nc));
    if backend == "cpu" {
        // One p = 4, B = 1e3 GEMM took 33 s on the CPU runtime: 256-unit cubes with
        // barriers are emulated very slowly there. `tiled-reg` is the CPU-shaped kernel.
        record(
            smem_name,
            Err("skipped on the CPU runtime (too slow)".into()),
        );
    } else {
        let c = client.empty(nc * b * size_of::<F>());
        let outcome = timed(&smem_name, &mut || {
            tiled::launch_smem::<F>(client, p, b, &a, &x, &c);
            Ok(())
        })
        .map(|(median, reps)| (median, reps, c));
        record(smem_name, outcome);
    }

    for rows_fast in [false, true] {
        let name = format!(
            "tiled-reg-{}({}x{}, vec {})",
            if rows_fast { "rows" } else { "cols" },
            tiled::REG_TM,
            tiled::REG_TNV * tiled::REG_VEC,
            tiled::REG_VEC
        );
        let c = client.empty(nc * b * size_of::<F>());
        let outcome = timed(&name, &mut || {
            tiled::launch_reg::<F>(client, p, b, rows_fast, &a, &x, &c);
            Ok(())
        })
        .map(|(median, reps)| (median, reps, c));
        record(name, outcome);
    }

    rows
}
