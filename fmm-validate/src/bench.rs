//! Helpers of the timing reports: the machine description, the median time per call
//! over repeated batches, the fitted exponent of t ∝ pᵏ and the crossover between two
//! methods.
//!
//! The examples `timing`, `tables`, `p2p_kernels` and `p2p_fmm` use them. Nothing here asserts a timing; the
//! tests cover the fit and the crossover on made-up data only.
//!
//! ```
//! use nd_fmm_validate::bench::{Crossover, crossover, fitted_exponent};
//!
//! let ps = [4, 8, 16];
//! let cubic: Vec<f64> = ps.iter().map(|&p| (p as f64).powi(3)).collect();
//! assert!((fitted_exponent(&ps, &cubic, 8) - 3.0).abs() < 1e-12);
//! let quartic: Vec<f64> = ps.iter().map(|&p| (p as f64).powi(4) / 8.0).collect();
//! assert_eq!(
//!     crossover(&ps, &quartic, &cubic),
//!     Crossover::From { second_from: 16, first_at: 8 }
//! );
//! ```

use std::time::Duration;

/// Number of timed batches per figure; the median is reported.
pub const BATCHES: usize = 15;

/// Minimum duration of one batch.
pub const MIN_BATCH: Duration = Duration::from_millis(20);

/// Median time per call, in seconds, over [`BATCHES`] batches.
///
/// `batch(calls)` performs `calls` calls and returns the time they took. The number of
/// calls per batch is found first, by doubling from one until a batch takes at least
/// [`MIN_BATCH`]; that also warms up caches and branch predictors.
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

/// Least-squares slope k of ln t against ln p over the points with p ≥ `from`: the
/// exponent of t ∝ pᵏ.
///
/// # Panics
///
/// If fewer than two degrees are at least `from`, or `ps` and `times` differ in length.
pub fn fitted_exponent(ps: &[usize], times: &[f64], from: usize) -> f64 {
    assert_eq!(ps.len(), times.len(), "one time per degree");
    let points: Vec<(f64, f64)> = ps
        .iter()
        .zip(times)
        .filter(|(p, _)| **p >= from)
        .map(|(&p, &t)| ((p as f64).ln(), t.ln()))
        .collect();
    assert!(points.len() >= 2, "the fit needs two degrees p ≥ {from}");
    let n = points.len() as f64;
    let (mx, my) = points
        .iter()
        .fold((0.0, 0.0), |(a, b), (x, y)| (a + x / n, b + y / n));
    let sxy: f64 = points.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = points.iter().map(|(x, _)| (x - mx) * (x - mx)).sum();
    sxy / sxx
}

/// Where the second of two methods becomes faster than the first, over the measured
/// degrees ([`crossover`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossover {
    /// The second method is faster at every measured p.
    SecondAlways,
    /// The first method is at least as fast at the largest measured p.
    FirstAtLargest,
    /// The second method is faster at `second_from` and every larger measured p; the
    /// first is at least as fast at `first_at`, the measured p just below.
    From {
        /// The smallest measured p from which the second method is always faster.
        second_from: usize,
        /// The largest measured p at which the first method is at least as fast.
        first_at: usize,
    },
}

impl Crossover {
    /// The crossover in words, naming the methods `first` and `second`, e.g.
    /// "rotation is faster for p ≥ 12 (direct is faster at p = 10)".
    pub fn describe(self, first: &str, second: &str) -> String {
        match self {
            Crossover::SecondAlways => format!("{second} is faster at every measured p"),
            Crossover::FirstAtLargest => format!("{first} is faster at the largest measured p"),
            Crossover::From {
                second_from,
                first_at,
            } => format!(
                "{second} is faster for p ≥ {second_from} ({first} is faster at p = {first_at})"
            ),
        }
    }
}

/// The crossover of `second` against `first`, times per call at the degrees `ps`: the
/// smallest measured p from which `second` is faster at every larger measured p, and
/// the largest measured p at which `first` is at least as fast.
///
/// # Panics
///
/// If the three slices differ in length.
pub fn crossover(ps: &[usize], first: &[f64], second: &[f64]) -> Crossover {
    assert!(
        ps.len() == first.len() && ps.len() == second.len(),
        "one time per degree and method"
    );
    match (0..ps.len()).rev().find(|&i| first[i] <= second[i]) {
        None => Crossover::SecondAlways,
        Some(i) if i + 1 == ps.len() => Crossover::FirstAtLargest,
        Some(i) => Crossover::From {
            second_from: ps[i + 1],
            first_at: ps[i],
        },
    }
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
/// On Linux the `model name` line names it (x86_64); where there is none (aarch64, e.g.
/// NVIDIA Grace), the `CPU implementer` and `CPU part` lines do ([`cpuinfo_model`]).
pub fn cpu_model() -> String {
    let model = if cfg!(target_os = "macos") {
        sysctl("machdep.cpu.brand_string")
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|info| cpuinfo_model(&info))
    };
    model.unwrap_or_else(|| "unknown".to_string())
}

/// The CPU model in the text of `/proc/cpuinfo`: its `model name` (or `Model`) line, else
/// the first `CPU implementer` and `CPU part` of an Arm core, named where the part is known
/// ("Neoverse-V2 (implementer 0x41, part 0xd4f)" on NVIDIA Grace), else by the numbers
/// alone; `None` if the text has neither.
pub fn cpuinfo_model(info: &str) -> Option<String> {
    let value = |key: &str| {
        info.lines()
            .find(|l| l.split(':').next().is_some_and(|k| k.trim() == key))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    if let Some(model) = value("model name").or_else(|| value("Model")) {
        return Some(model);
    }
    let (implementer, part) = (value("CPU implementer")?, value("CPU part")?);
    let parse = |s: &str| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok();
    // Arm Ltd's (0x41) server and recent cores, from the Arm technical reference manuals'
    // MIDR_EL1 part numbers.
    let name = match (parse(&implementer), parse(&part)) {
        (Some(0x41), Some(0xd0c)) => Some("Neoverse-N1"),
        (Some(0x41), Some(0xd40)) => Some("Neoverse-V1"),
        (Some(0x41), Some(0xd49)) => Some("Neoverse-N2"),
        (Some(0x41), Some(0xd4f)) => Some("Neoverse-V2"),
        (Some(0x41), Some(0xd84)) => Some("Neoverse-V3"),
        (Some(0x41), Some(0xd8e)) => Some("Neoverse-N3"),
        _ => None,
    };
    let numbers = format!("implementer {implementer}, part {part}");
    Some(name.map_or_else(
        || format!("Arm CPU ({numbers})"),
        |name| format!("{name} ({numbers})"),
    ))
}

/// The core count: logical CPUs from `std::thread::available_parallelism`, and the
/// physical cores from `sysctl` on macOS and from `/sys/devices/system/cpu` on Linux
/// (the distinct `core_cpus_list` of the CPUs' topologies).
pub fn cores() -> String {
    let logical = std::thread::available_parallelism().map_or(0, |n| n.get());
    let physical = if cfg!(target_os = "macos") {
        sysctl("hw.physicalcpu")
    } else {
        linux_physical_cores().map(|n| n.to_string())
    };
    match physical {
        Some(physical) => format!("{physical} physical, {logical} logical"),
        None => format!("{logical} logical"),
    }
}

/// The physical cores of a Linux machine: the distinct sets of logical CPUs that share a
/// core (`topology/core_cpus_list`, or `thread_siblings_list` on older kernels) over
/// `/sys/devices/system/cpu/cpu<n>`; `None` where they cannot be read.
fn linux_physical_cores() -> Option<usize> {
    let mut cores = std::collections::BTreeSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_str()?;
        if !name
            .strip_prefix("cpu")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
        {
            continue;
        }
        let topology = path.join("topology");
        let siblings = std::fs::read_to_string(topology.join("core_cpus_list"))
            .or_else(|_| std::fs::read_to_string(topology.join("thread_siblings_list")))
            .ok()?;
        cores.insert(siblings.trim().to_string());
    }
    (!cores.is_empty()).then_some(cores.len())
}

/// The number of performance cores: on macOS `hw.perflevel0.physicalcpu` (12 on the
/// Apple M3 Max), elsewhere `None`, where the cores are not split by kind.
pub fn performance_cores() -> Option<usize> {
    if cfg!(target_os = "macos") {
        sysctl("hw.perflevel0.physicalcpu").and_then(|s| s.parse().ok())
    } else {
        None
    }
}

/// The version of the `rustc` on the path at run time, e.g. "rustc 1.99.0 (…)", or
/// "unknown". Under `cargo run` it is the compiler that built the program, unless the
/// toolchain was overridden for the build only.
pub fn toolchain() -> String {
    std::process::Command::new("rustc")
        .arg("-V")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}

/// The compilation target and profile, e.g. "aarch64-macos, release build"; a debug
/// build is flagged as giving meaningless timings.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cpu_model_comes_from_cpuinfo() {
        // Error measure: exact strings. Excerpts of /proc/cpuinfo: x86_64 with a model
        // name, NVIDIA Grace (locust, Phase 4S T4) without one, an unknown Arm part, and
        // neither.
        let x86 = "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel\t\t: 85\n\
                   model name\t: Intel(R) Xeon(R) Gold 6248\n";
        assert_eq!(
            cpuinfo_model(x86).as_deref(),
            Some("Intel(R) Xeon(R) Gold 6248")
        );
        let grace = "processor\t: 0\nBogoMIPS\t: 2000.00\nFeatures\t: fp asimd sve2\n\
                     CPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x0\n\
                     CPU part\t: 0xd4f\nCPU revision\t: 0\n\nprocessor\t: 1\n\
                     CPU implementer\t: 0x41\nCPU part\t: 0xd4f\n";
        assert_eq!(
            cpuinfo_model(grace).as_deref(),
            Some("Neoverse-V2 (implementer 0x41, part 0xd4f)")
        );
        let other = "CPU implementer\t: 0x61\nCPU part\t: 0x022\n";
        assert_eq!(
            cpuinfo_model(other).as_deref(),
            Some("Arm CPU (implementer 0x61, part 0x022)")
        );
        assert_eq!(cpuinfo_model("processor\t: 0\n"), None);
    }

    #[test]
    fn fit_recovers_the_exponent_of_a_power_law() {
        // Error measure: absolute error of the slope; the data are exact powers.
        let ps = [2, 4, 8, 12, 16, 20];
        let times: Vec<f64> = ps.iter().map(|&p| 3e-7 * (p as f64).powf(3.5)).collect();
        assert!((fitted_exponent(&ps, &times, 8) - 3.5).abs() < 1e-12);
        // Points below `from` are ignored, however far off.
        let mut noisy = times.clone();
        noisy[0] = 1.0;
        noisy[1] = 1e-12;
        assert!((fitted_exponent(&ps, &noisy, 8) - 3.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "the fit needs two degrees p ≥ 8")]
    fn fit_needs_two_points() {
        let _ = fitted_exponent(&[4, 8], &[1.0, 2.0], 8);
    }

    #[test]
    fn crossover_cases() {
        let ps = [2, 4, 8];
        let first = [1.0, 2.0, 3.0];
        assert_eq!(
            crossover(&ps, &first, &[2.0, 3.0, 4.0]),
            Crossover::FirstAtLargest
        );
        assert_eq!(
            crossover(&ps, &first, &[0.5, 1.0, 1.5]),
            Crossover::SecondAlways
        );
        // A tie counts for the first method.
        assert_eq!(
            crossover(&ps, &first, &[1.0, 2.0, 2.5]),
            Crossover::From {
                second_from: 8,
                first_at: 4
            }
        );
        // The last p at which the first is faster decides, not the first crossing.
        assert_eq!(
            crossover(&ps, &first, &[2.0, 1.0, 3.0]),
            Crossover::FirstAtLargest
        );
        assert_eq!(
            Crossover::From {
                second_from: 12,
                first_at: 10
            }
            .describe("direct", "rotation"),
            "rotation is faster for p ≥ 12 (direct is faster at p = 10)"
        );
        assert_eq!(
            Crossover::SecondAlways.describe("direct", "rotation"),
            "rotation is faster at every measured p"
        );
        assert_eq!(
            Crossover::FirstAtLargest.describe("direct", "rotation"),
            "direct is faster at the largest measured p"
        );
    }

    #[test]
    fn median_is_taken_over_the_batches() {
        // Error measure: exact equality; the batch reports a made-up duration.
        let mut runs = Vec::new();
        let t = median_time_per_call(|calls| {
            runs.push(calls);
            // 1 ms per call, plus 1 ms per batch after the calibration.
            let extra = u64::from(runs.len() > 6);
            Duration::from_millis(calls as u64 + extra * calls as u64)
        });
        // Calibration: 1, 2, 4, 8, 16 and 32 calls; 32 ms ≥ 20 ms.
        assert_eq!(runs[..6], [1, 2, 4, 8, 16, 32]);
        assert_eq!(runs.len(), 6 + BATCHES);
        assert!(runs[6..].iter().all(|&c| c == 32));
        assert_eq!(t, 2e-3);
    }
}
