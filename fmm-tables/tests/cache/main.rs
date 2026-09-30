//! Acceptance tests for the versioned on-disk cache (Phase 2 / T7, C2.4): round trips
//! of every family in both precisions, `load_or_build` cold and warm, rejection of
//! stale, mismatched, truncated and corrupt files, the documented header, determinism,
//! atomic writes and temporary files, an unwritable directory, and the f32 files.
//!
//! The oracle is the cold build. Every test names its error measure; all are exact:
//! bit-for-bit equality of tables and byte-for-byte equality of files. Each test works
//! in its own subdirectory of `CARGO_TARGET_TMPDIR` and removes it at the end.

mod common;
mod files;
mod rejection;
mod round_trip;
