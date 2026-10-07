//! The report's header ([`Header::collect`]): the date, the machine, the devices and the
//! software, from `nd_fmm_validate::bench`'s machine lines and a few commands
//! (`hostname`, `git`, `nvidia-smi`, `nvcc`), each optional: a fact that cannot be read
//! is reported as unknown, never guessed.

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use nd_fmm_exec::fmm::Backend;
use nd_fmm_validate::bench::{cores, cpu_model, performance_cores, target, toolchain};

use crate::options::Options;
use crate::report::Header;

/// The thread variables the header reports: every BLAS and OpenMP variable that
/// `tools/bench/run.sh` sets to 1, and rayon's.
pub const THREAD_VARIABLES: [&str; 7] = [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "RAYON_NUM_THREADS",
];

impl Header {
    /// The header of a run of `options` started now, with the command line `command`.
    /// Opens (and closes again) the device of every device backend in `options`, to name
    /// it.
    pub fn collect(options: &Options, command: String) -> Self {
        let (date, _) = utc_now();
        let devices: Vec<Backend> = Backend::ALL
            .into_iter()
            .filter(|b| b.is_device() && options.backend.contains(b))
            .collect();
        let (devices, cubecl) = describe_devices(&devices);
        Self {
            date,
            host: host_name(),
            cpu: cpu_model(),
            cores: format!(
                "{}{}",
                cores(),
                performance_cores().map_or(String::new(), |p| format!(", {p} performance"))
            ),
            devices,
            cuda: options.backend.contains(&Backend::Cuda).then(cuda_versions),
            rustc: toolchain(),
            target: target(),
            cubecl,
            revision: source_revision(),
            command,
            compiled: Backend::ALL
                .into_iter()
                .filter(|b| b.is_compiled())
                .map(Backend::name)
                .collect::<Vec<_>>()
                .join(", "),
            environment: THREAD_VARIABLES
                .iter()
                .map(|v| match std::env::var(v) {
                    Ok(value) => format!("{v}={value}"),
                    Err(_) => format!("{v} unset"),
                })
                .collect::<Vec<_>>()
                .join(", "),
            threads: options.threads(),
        }
    }
}

/// One line per device backend, its `DeviceInfo` or why it did not open, and the CubeCL
/// version of the first that opened.
#[cfg(feature = "gpu")]
fn describe_devices(backends: &[Backend]) -> (Vec<String>, String) {
    let mut cubecl = None;
    let lines = backends
        .iter()
        .map(|&backend| match backend.probe() {
            Ok(info) => {
                cubecl.get_or_insert(info.cubecl_version);
                info.to_string()
            }
            Err(reason) => format!("{backend}: not available ({reason})"),
        })
        .collect();
    (
        lines,
        cubecl.map_or_else(|| "(no device opened)".to_owned(), str::to_owned),
    )
}

/// Without the `gpu` feature no device backend is compiled in.
#[cfg(not(feature = "gpu"))]
fn describe_devices(backends: &[Backend]) -> (Vec<String>, String) {
    let lines = backends
        .iter()
        .map(|b| format!("{b}: not compiled in (build with --features {b})"))
        .collect();
    (lines, "not compiled in".to_owned())
}

/// The output of a command, trimmed, if it ran and succeeded with some output.
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let text = text.trim();
    (out.status.success() && !text.is_empty()).then(|| text.to_owned())
}

/// The host name: `hostname`, else `HOSTNAME`, else "unknown".
pub fn host_name() -> String {
    command_output("hostname", &[])
        .or_else(|| std::env::var("HOSTNAME").ok().filter(|h| !h.is_empty()))
        .unwrap_or_else(|| "unknown".to_owned())
}

/// The host name for a file name: up to the first `.`, and only ASCII letters, digits,
/// `-` and `_`.
pub fn short_host(host: &str) -> String {
    let short: String = host
        .split('.')
        .next()
        .unwrap_or_default()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if short.is_empty() {
        "unknown".to_owned()
    } else {
        short
    }
}

/// The source revision: `git describe --always --dirty` in the checkout this binary was
/// built from, else the first line of its `.source-revision` (written by
/// tools/gh200/sync.sh, with the branch on the second), else "unknown".
pub fn source_revision() -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(Path::new("."));
    if let Some(root) = root.to_str()
        && let Some(revision) =
            command_output("git", &["-C", root, "describe", "--always", "--dirty"])
    {
        return format!("{revision} (git describe)");
    }
    std::fs::read_to_string(root.join(".source-revision"))
        .ok()
        .and_then(|text| {
            let mut lines = text.lines().map(str::trim);
            let revision = lines.next().filter(|r| !r.is_empty())?;
            Some(match lines.next().filter(|b| !b.is_empty()) {
                Some(branch) => format!("{revision} on {branch} (.source-revision)"),
                None => format!("{revision} (.source-revision)"),
            })
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

/// The CUDA driver, GPU and clocks (`nvidia-smi`) and the toolkit (`nvcc`); CubeCL
/// reports neither.
fn cuda_versions() -> String {
    let smi = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total,clocks.sm,clocks.max.sm",
            "--format=csv,noheader",
        ],
    )
    .map_or_else(
        || "nvidia-smi: not available".to_owned(),
        |s| {
            format!(
                "nvidia-smi: {} (name, driver, memory, SM clock, max SM clock)",
                s.lines().next().unwrap_or_default()
            )
        },
    );
    let nvcc = command_output("nvcc", &["--version"])
        .and_then(|s| {
            s.lines()
                .find(|l| l.contains("release"))
                .map(|l| l.trim().to_owned())
        })
        .map_or_else(|| "nvcc: not on PATH".to_owned(), |l| format!("nvcc: {l}"));
    format!("{smi}; {nvcc}")
}

/// The current UTC time as `YYYY-MM-DD hh:mm:ss UTC` and as `YYYY-MM-DD-hhmmss` (for
/// file names).
pub fn utc_now() -> (String, String) {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    utc(seconds)
}

/// The UTC time `seconds` after the Unix epoch, in the two forms of [`utc_now`].
pub fn utc(seconds: u64) -> (String, String) {
    let days = (seconds / 86_400) as i64;
    let rest = seconds % 86_400;
    let (h, m, s) = (rest / 3600, rest % 3600 / 60, rest % 60);
    let (year, month, day) = civil_from_days(days);
    (
        format!("{year:04}-{month:02}-{day:02} {h:02}:{m:02}:{s:02} UTC"),
        format!("{year:04}-{month:02}-{day:02}-{h:02}{m:02}{s:02}"),
    )
}

/// The proleptic Gregorian date of the day `days` after 1970-01-01 (H. Hinnant's
/// `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_of_known_instants() {
        // Error measure: exact strings, against `date -u -r <seconds>`.
        assert_eq!(
            utc(0),
            (
                "1970-01-01 00:00:00 UTC".to_owned(),
                "1970-01-01-000000".to_owned()
            )
        );
        assert_eq!(utc(951_782_400).0, "2000-02-29 00:00:00 UTC");
        assert_eq!(utc(1_709_210_096).0, "2024-02-29 12:34:56 UTC");
        assert_eq!(
            utc(1_791_374_639),
            (
                "2026-10-07 12:03:59 UTC".to_owned(),
                "2026-10-07-120359".to_owned()
            )
        );
        assert_eq!(utc(4_102_444_799).0, "2099-12-31 23:59:59 UTC");
    }

    #[test]
    fn short_host_names_are_file_safe() {
        assert_eq!(short_host("locust.rc.ucl.ac.uk"), "locust");
        assert_eq!(short_host("Timos-MacBook Pro"), "Timos-MacBook_Pro");
        assert_eq!(short_host(""), "unknown");
        assert_eq!(short_host("a/b"), "a_b");
    }
}
