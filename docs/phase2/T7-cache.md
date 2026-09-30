# Phase 2 / T7 — nd-fmm-tables: versioned on-disk cache (C2.4)

Building the tables takes tens of seconds at p = 20 (T4). Phase 3 needs them for every
run and on every MPI rank. This task stores them on disk, keyed so that a stale table
can never be loaded.

Read first: docs/CONVENTIONS.md §3.10 and §3.12; docs/phase2/README.md ("Design
decisions": serial and deterministic, no serialiser dependency); fmm-tables/CLAUDE.md;
the T3–T6 code; `nd_fmm_math::CONVENTION_VERSION`.

Do:
- Add `thiserror` (`workspace = true`) to nd-fmm-tables.
- Module `cache`:
  - A sealed trait `Stored: RealScalar`, implemented for f32 and f64, with a
    `PRECISION` constant and little-endian byte conversion.
  - `TableKind::{M2m, L2l, M2l, M2lClasses, Rotation}`, and a trait that each table
    family implements. It gives the family's kind, `build(p)`, and the payload encode
    and decode.
  - `TableCache::new(dir)`. The caller supplies the directory. This crate reads no
    environment variable and picks no default; `nd-fmm-exec` decides in Phase 3.
  - `load::<F>(p) -> Result<F, CacheError>` and `store(&F) -> Result<(), CacheError>`.
  - `load_or_build::<F>(p) -> (F, CacheOutcome)`. It never fails because of the cache:
    `CacheOutcome` is `Loaded`, `Built`, `Rebuilt(CacheError)` (a stale, mismatched or
    corrupt file was replaced) or `NotStored(CacheError)` (the table was built but could
    not be written).
- Key and file name: the kind, p, precision, `CONVENTION_VERSION` and a
  `FORMAT_VERSION` of this crate. Derive the file name from the key, for example
  `m2l-p08-f64-c1-v1.bin`.
- File format, documented as a table in the module docs; all integers little-endian:
  - magic bytes;
  - format version, convention version, kind, p and precision;
  - payload length and an FNV-1a 64-bit checksum of the payload, hand-written (a few
    lines);
  - the payload: the values in the in-memory order of the family (`MatrixSet` layout,
    then any index tables), little-endian.
- Writes are atomic:
  - Write to a uniquely named temporary file in the same directory, then rename it
    over the final name. No partial file ever appears under a final name.
  - Two processes that build the same key at the same time both succeed, and their
    files are byte-identical (determinism).
  - Loads ignore temporary files.
- An f32 file holds the rounded f64 table, never a table built in f32.
- Errors (`thiserror`): I/O, bad magic, format version, convention version, key
  mismatch (kind, p or precision), length and checksum, each carrying what was found
  and what was expected.

Tests that define done. Use subdirectories of `env!("CARGO_TARGET_TMPDIR")`, one per
test, and remove them at the end.
- Round trip: for every family and both precisions, at p ∈ {0, 1, 4}, `store` then
  `load` is bit-identical to the built table.
- `load_or_build`: `Built` when cold, then `Loaded`, bit-identical to the cold build.
- Rejection, each also through `load_or_build`, which must report `Rebuilt` with the
  right error:
  - a stale convention version or format version (header patched by the test);
  - a file renamed to another key's name (key mismatch);
  - a truncated file (length);
  - one flipped payload byte (checksum);
  - bad magic.
- Storing the same table twice gives byte-identical files.
- No temporary file remains after a successful store. A leftover temporary file (a
  crashed writer, simulated by the test) does not affect loading.
- An unwritable directory gives `NotStored`, together with a correct table. Use a
  path below a regular file, which fails even when tests run as root.
- The f32 file equals the cast of the f64 build.

Must pass:
- `cargo test -p nd-fmm-tables` (debug run under a minute; keep cached tables at
  p ≤ 4 in tests);
- `cargo clippy -p nd-fmm-tables --all-targets -- -D warnings`;
- `cargo doc -p nd-fmm-tables --no-deps` without warnings;
- the root checks.

Report the file sizes of every family at p = 8 (T8 times loading).

Do not:
- add serde, bincode, a checksum crate or file locking;
- choose a default cache directory or read environment variables;
- write outside the directory given;
- commit any cache file (`.gitignore` covers `target/` only).
