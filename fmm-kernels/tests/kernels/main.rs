//! Tests of `nd-fmm-kernels` on every backend compiled in: one executable, so the
//! CubeCL runtimes link once and each kernel compiles once per backend and process.
//!
//! - No feature: the host-side tests only (`capability`), in seconds.
//! - `--features cpu`: every runtime test on the CubeCL CPU runtime, in f32 and f64.
//! - `--features metal`: the same tests on Metal in f32, `#[ignore]`d, run by hand
//!   outside the macOS sandbox with `-- --ignored`.
//!
//! Every runtime test prints the device it ran on and a closing line "backends run:
//! …; not run: …"; run with `--show-output` to see them.

mod common;

mod capability;
mod movement;
mod properties;
mod round_trip;
