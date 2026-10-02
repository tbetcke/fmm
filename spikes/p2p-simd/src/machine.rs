//! The machine description of the report. `cpu_model`, `cores`, `target` and the
//! timing loop are copies of `nd_fmm_validate::bench` (Phase 3), so that the spike
//! does not depend on MPI.

use std::time::Duration;

/// Number of timed batches per figure; the median is reported.
pub const BATCHES: usize = 15;

/// Minimum duration of one batch.
pub const MIN_BATCH: Duration = Duration::from_millis(20);

/// Median time per call, in seconds, over [`BATCHES`] batches of at least
/// [`MIN_BATCH`] (`nd_fmm_validate::bench::median_time_per_call`).
pub fn median_time_per_call(mut batch: impl FnMut(usize) -> Duration) -> f64 {
    let mut calls = 1;
    while batch(calls) < MIN_BATCH {
        calls *= 2;
    }
    let mut times: Vec<f64> = (0..BATCHES)
        .map(|_| batch(calls).as_secs_f64() / calls as f64)
        .collect();
    times.sort_by(f64::total_cmp);
    times[BATCHES / 2]
}

/// The value of a `sysctl` key (macOS), if it can be read.
fn sysctl(key: &str) -> Option<String> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .ok()?;
    let s = String::from_utf8(out.stdout).ok()?;
    let s = s.trim();
    (out.status.success() && !s.is_empty()).then(|| s.to_string())
}

/// The CPU model, from `sysctl` on macOS and `/proc/cpuinfo` on Linux, or "unknown".
pub fn cpu_model() -> String {
    let from_proc = || {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        info.lines()
            .find(|l| l.starts_with("model name") || l.starts_with("Model"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
    };
    let model = if cfg!(target_os = "macos") {
        sysctl("machdep.cpu.brand_string")
    } else {
        from_proc()
    };
    model.unwrap_or_else(|| "unknown".to_string())
}

/// The core count: logical CPUs, and on macOS the physical cores.
pub fn cores() -> String {
    let logical = std::thread::available_parallelism().map_or(0, |n| n.get());
    match sysctl("hw.physicalcpu") {
        Some(physical) if cfg!(target_os = "macos") => {
            format!("{physical} physical, {logical} logical")
        }
        _ => format!("{logical} logical"),
    }
}

/// The compilation target and profile; a debug build is flagged.
pub fn target() -> String {
    format!(
        "{}-{}{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        if cfg!(debug_assertions) {
            " (DEBUG BUILD: timings are not meaningful; use --release)"
        } else {
            ", release build"
        }
    )
}

/// The compiler that built the spike.
pub fn toolchain() -> &'static str {
    env!("SPIKE_RUSTC_VERSION")
}

/// Whether AVX2 and FMA are both available (x86_64).
#[cfg(target_arch = "x86_64")]
pub fn avx2_fma() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

/// The ISAs with prototypes on this CPU.
pub fn isas() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = vec!["scalar"];
    #[cfg(target_arch = "aarch64")]
    v.push("neon");
    #[cfg(target_arch = "x86_64")]
    if avx2_fma() {
        v.push("avx2");
    }
    v
}

/// The pulp backend green-kernels dispatches to, as pulp names it.
pub fn pulp_backend() -> String {
    format!("{:?}", pulp::Arch::new())
}

/// Clock frequency assumed to convert times to cycles: the P-core maximum of the
/// Apple M3 Max, 4.05 GHz. The report checks it against the measured latency of a
/// dependent FMA chain (4 cycles on Apple P-cores).
pub const ASSUMED_GHZ: f64 = 4.05;
