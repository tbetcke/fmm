//! Versioned on-disk cache of the operator tables (CONVENTIONS §3.10, §3.12; C2.4).
//!
//! Building the tables takes tens of seconds at p = 20, and every run of the FMM needs
//! them on every rank. A [`TableCache`] stores each table in one file of a directory
//! that the caller supplies, and loads it back bit for bit. This crate reads no
//! environment variable and picks no default directory.
//!
//! # Keys and file names
//!
//! A file is keyed by the table family ([`TableKind`]), the degree p, the stored
//! [`Precision`], [`CONVENTION_VERSION`] of `nd-fmm-math` and the [`FORMAT_VERSION`]
//! of this module. The file name is derived from the key ([`CacheKey::file_name`]),
//! for example `m2l-p08-f64-c1-v1.bin`, and the key is repeated in the header, so a
//! file that was renamed, or written under another convention or format, is rejected
//! on loading rather than used. By CONVENTIONS §3.10 any change to the conventions
//! the tables depend on, §3.12 included, bumps the convention version and so
//! invalidates every cached table.
//!
//! # File format
//!
//! All integers are little-endian. The header is [`HEADER_LEN`] = 44 bytes:
//!
//! | Offset | Size | Field |
//! | --- | --- | --- |
//! | 0 | 8 | magic bytes [`MAGIC`], `NDFMMTAB` |
//! | 8 | 4 | format version, u32 ([`FORMAT_VERSION`]) |
//! | 12 | 4 | convention version, u32 ([`CONVENTION_VERSION`]) |
//! | 16 | 4 | kind, u32 ([`TableKind::code`]) |
//! | 20 | 4 | degree p, u32 |
//! | 24 | 4 | precision, u32: bits per value, 32 or 64 ([`Precision::code`]) |
//! | 28 | 8 | payload length in bytes, u64 |
//! | 36 | 8 | FNV-1a 64-bit checksum of the payload, u64 |
//! | 44 | payload length | payload |
//!
//! The payload holds the values of the table in its in-memory order, each value in the
//! stored precision (f64 or f32, little-endian IEEE 754), then any index tables as u32.
//! A [`MatrixSet`] is written as [`MatrixSet::as_slice`]: matrix after matrix, each
//! column-major. With n = (p + 1)², B = `blocks_len(p)` and C the number of coaxial
//! factors per distance:
//!
//! | Kind | Payload |
//! | --- | --- |
//! | [`TableKind::M2m`], [`TableKind::L2l`] | the 8 matrices, 8 n² values |
//! | [`TableKind::M2l`] | the 316 matrices, 316 n² values |
//! | [`TableKind::M2lClasses`] | the 16 class matrices, 16 n² values; T_M(P) of the 48 group elements in enumeration order, B values each; T_L(P) likewise; for each of the 316 offsets its class and group element index, two u32 |
//! | [`TableKind::Rotation`] | the M2M, L2L and M2L [`ShiftTables`] in turn, each as below |
//!
//! Each [`ShiftTables`] is written as: the numbers of
//! stored polar angles, azimuths and distances, three u32; the polar angles, azimuths
//! and distances, always as f64; the forward blocks, the backward blocks, the azimuth
//! factors and the coaxial factors, in the stored precision; and for each entry its
//! [`Shift`] as four u32: the alignment (0 up, 1 down,
//! 2 rotated), the polar angle and azimuth indices (0 unless rotated) and the distance
//! index.
//!
//! An f32 file holds the f64 build rounded entry by entry, since
//! [`CachedTable::build`] builds in f64 in every precision. Building and encoding are
//! deterministic, so storing the same key twice, from one process or from two, gives
//! byte-identical files.
//!
//! # Writing and loading
//!
//! [`TableCache::store`] writes to a uniquely named temporary file in the cache
//! directory, `.<file name>.<process id>-<counter>.tmp`, and renames it over the
//! final name once it is complete and synced. The header, with the magic bytes, is
//! written last. No partial file ever appears under a final name, and two processes
//! that store the same key at the same time both succeed. [`TableCache::load`] opens
//! only the final name, so temporary files, including those a crashed writer left
//! behind, never affect loading. The cache uses no file locking.
//!
//! A load checks, in this order: the header length, the magic bytes, the format
//! version, the convention version, the key (kind, p, precision), the payload length
//! against the file, the checksum, and finally the structure of the payload. Each
//! failure is a [`CacheError`] that carries what was found and what was expected.
//! [`TableCache::load_or_build`] never fails because of the cache: it builds the table
//! when the file is missing or rejected and reports what happened in a
//! [`CacheOutcome`].
//!
//! ```no_run
//! use nd_fmm_tables::M2lTables;
//! use nd_fmm_tables::cache::{CacheOutcome, TableCache};
//!
//! let cache = TableCache::new("/path/chosen/by/the/caller");
//! let (tables, outcome) = cache.load_or_build::<M2lTables<f64>>(8);
//! match outcome {
//!     CacheOutcome::Loaded | CacheOutcome::Built => {}
//!     CacheOutcome::Rebuilt(error) => eprintln!("replaced a rejected cache file: {error}"),
//!     CacheOutcome::NotStored(error) => eprintln!("could not write the cache: {error}"),
//! }
//! assert_eq!(tables.p(), 8);
//! ```

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use nd_fmm_math::rotation::blocks_len;
use nd_fmm_math::{CONVENTION_VERSION, Layout, RealScalar};

use crate::MatrixSet;
use crate::geometry::{M2L_OFFSET_COUNT, OCTANT_COUNT};
use crate::m2l::{M2lClasses, M2lTables};
use crate::octant::{L2lTables, M2mTables, OctantTables};
use crate::rotation::{Alignment, Operator, RotationTables, Shift, ShiftTables, coaxial_len};
use crate::symmetry::{CoefficientTransform, GROUP_ORDER, M2L_CLASS_COUNT, SignedPermutation};

/// The magic bytes that open every cache file.
pub const MAGIC: [u8; 8] = *b"NDFMMTAB";

/// The version of the file format of this module. Any change to the header or to the
/// payload of any family bumps it.
pub const FORMAT_VERSION: u32 = 1;

/// The length of the file header in bytes (module documentation, "File format").
pub const HEADER_LEN: usize = 44;

/// Byte offset of the payload length in the header.
const LENGTH_OFFSET: usize = 28;

/// The size of the buffers that convert values to and from bytes.
const CHUNK: usize = 8192;

/// FNV-1a, 64 bits: the offset basis.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a, 64 bits: the prime.
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Continues the FNV-1a 64-bit hash `hash` over `bytes`.
fn fnv1a(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

mod sealed {
    /// Seals [`Stored`](super::Stored) and [`CachedTable`](super::CachedTable).
    pub trait Sealed {}
}

/// The precision of the stored values of a table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

impl Precision {
    /// Returns the code of the header: the number of bits per value, 32 or 64.
    pub const fn code(self) -> u32 {
        match self {
            Precision::F32 => 32,
            Precision::F64 => 64,
        }
    }

    /// Returns the name used in file names, `f32` or `f64`.
    pub const fn name(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }

    /// Returns the number of bytes per value, 4 or 8.
    pub const fn bytes(self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }
}

/// A scalar type in which tables are stored: f32 or f64. Sealed.
pub trait Stored: RealScalar + sealed::Sealed {
    /// The precision of the type.
    const PRECISION: Precision;

    /// Writes the value into `out` as little-endian IEEE 754 bytes.
    ///
    /// # Panics
    ///
    /// If `out` does not have length `PRECISION.bytes()`.
    fn write_le(self, out: &mut [u8]);

    /// Reads a value from little-endian IEEE 754 bytes, bit for bit.
    ///
    /// # Panics
    ///
    /// If `bytes` does not have length `PRECISION.bytes()`.
    fn read_le(bytes: &[u8]) -> Self;
}

impl sealed::Sealed for f32 {}

impl Stored for f32 {
    const PRECISION: Precision = Precision::F32;

    #[inline]
    fn write_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }

    #[inline]
    fn read_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().expect("an f32 has 4 bytes"))
    }
}

impl sealed::Sealed for f64 {}

impl Stored for f64 {
    const PRECISION: Precision = Precision::F64;

    #[inline]
    fn write_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }

    #[inline]
    fn read_le(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().expect("an f64 has 8 bytes"))
    }
}

/// A table family that the cache stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TableKind {
    /// [`M2mTables`].
    M2m,
    /// [`L2lTables`].
    L2l,
    /// [`M2lTables`].
    M2l,
    /// [`M2lClasses`].
    M2lClasses,
    /// [`RotationTables`].
    Rotation,
}

impl TableKind {
    /// Returns the code of the header: 1 to 5 in the order of the variants.
    pub const fn code(self) -> u32 {
        match self {
            TableKind::M2m => 1,
            TableKind::L2l => 2,
            TableKind::M2l => 3,
            TableKind::M2lClasses => 4,
            TableKind::Rotation => 5,
        }
    }

    /// Returns the name used in file names: `m2m`, `l2l`, `m2l`, `m2l-classes` or
    /// `rotation`.
    pub const fn name(self) -> &'static str {
        match self {
            TableKind::M2m => "m2m",
            TableKind::L2l => "l2l",
            TableKind::M2l => "m2l",
            TableKind::M2lClasses => "m2l-classes",
            TableKind::Rotation => "rotation",
        }
    }
}

/// The key of a cache file: family, degree and precision. The convention and format
/// versions are those of this build, [`CONVENTION_VERSION`] and [`FORMAT_VERSION`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// The table family.
    pub kind: TableKind,
    /// The degree p.
    pub p: usize,
    /// The stored precision.
    pub precision: Precision,
}

impl CacheKey {
    /// Returns the key of the table family `F` at degree `p`.
    pub fn of<F: CachedTable>(p: usize) -> Self {
        Self {
            kind: F::KIND,
            p,
            precision: <F::Scalar as Stored>::PRECISION,
        }
    }

    /// Returns the file name of the key,
    /// `<kind>-p<p, two digits>-<precision>-c<convention>-v<format>.bin`, for example
    /// `m2l-p08-f64-c1-v1.bin`.
    pub fn file_name(&self) -> String {
        format!(
            "{}-p{:02}-{}-c{}-v{}.bin",
            self.kind.name(),
            self.p,
            self.precision.name(),
            CONVENTION_VERSION,
            FORMAT_VERSION
        )
    }

    /// Returns the header of a file of this key with the given payload length and
    /// checksum.
    fn header(&self, payload_len: u64, checksum: u64) -> [u8; HEADER_LEN] {
        let p = u32::try_from(self.p).expect("the degree p fits in a u32");
        let mut header = [0; HEADER_LEN];
        header[0..8].copy_from_slice(&MAGIC);
        header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&CONVENTION_VERSION.to_le_bytes());
        header[16..20].copy_from_slice(&self.kind.code().to_le_bytes());
        header[20..24].copy_from_slice(&p.to_le_bytes());
        header[24..28].copy_from_slice(&self.precision.code().to_le_bytes());
        header[LENGTH_OFFSET..36].copy_from_slice(&payload_len.to_le_bytes());
        header[36..44].copy_from_slice(&checksum.to_le_bytes());
        header
    }

    /// Checks `header` against this key and returns the payload length and checksum it
    /// records.
    fn check(&self, header: &[u8; HEADER_LEN]) -> Result<(u64, u64), CacheError> {
        let u32_at = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
        let magic: [u8; 8] = header[0..8].try_into().unwrap();
        if magic != MAGIC {
            return Err(CacheError::BadMagic {
                found: magic,
                expected: MAGIC,
            });
        }
        let format = u32_at(8);
        if format != FORMAT_VERSION {
            return Err(CacheError::FormatVersion {
                found: format,
                expected: FORMAT_VERSION,
            });
        }
        let convention = u32_at(12);
        if convention != CONVENTION_VERSION {
            return Err(CacheError::ConventionVersion {
                found: convention,
                expected: CONVENTION_VERSION,
            });
        }
        let fields = [
            (KeyField::Kind, u32_at(16), u64::from(self.kind.code())),
            (KeyField::P, u32_at(20), self.p as u64),
            (
                KeyField::Precision,
                u32_at(24),
                u64::from(self.precision.code()),
            ),
        ];
        for (field, found, expected) in fields {
            if u64::from(found) != expected {
                return Err(CacheError::KeyMismatch {
                    field,
                    found: found.into(),
                    expected,
                });
            }
        }
        Ok((u64_at(LENGTH_OFFSET), u64_at(36)))
    }
}

/// A field of the key in the file header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyField {
    /// The table family, as [`TableKind::code`].
    Kind,
    /// The degree p.
    P,
    /// The precision, as [`Precision::code`].
    Precision,
}

impl fmt::Display for KeyField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            KeyField::Kind => "kind",
            KeyField::P => "degree p",
            KeyField::Precision => "precision",
        })
    }
}

/// Why a cache file could not be loaded or stored. Each variant carries what was found
/// and what was expected.
#[derive(Debug, thiserror::Error)]
pub enum CacheError {
    /// Reading, writing, creating or renaming `path` failed; a missing file is an
    /// [`io::ErrorKind::NotFound`] error ([`CacheError::is_not_found`]).
    #[error("I/O error on {}: {source}", path.display())]
    Io {
        /// The file or directory.
        path: PathBuf,
        /// The error of the operating system.
        #[source]
        source: io::Error,
    },
    /// The file does not start with [`MAGIC`].
    #[error("bad magic bytes: found {found:02x?}, expected {expected:02x?}")]
    BadMagic {
        /// The first 8 bytes of the file.
        found: [u8; 8],
        /// [`MAGIC`].
        expected: [u8; 8],
    },
    /// The file was written in another file format.
    #[error("file format version {found}, expected {expected}")]
    FormatVersion {
        /// The version in the header.
        found: u32,
        /// [`FORMAT_VERSION`].
        expected: u32,
    },
    /// The file was written under another convention version (CONVENTIONS §3.10).
    #[error("convention version {found}, expected {expected}")]
    ConventionVersion {
        /// The version in the header.
        found: u32,
        /// [`CONVENTION_VERSION`].
        expected: u32,
    },
    /// The header holds another key than the file name: another kind, p or precision.
    #[error("key mismatch in the {field}: found {found}, expected {expected}")]
    KeyMismatch {
        /// The first field that differs.
        field: KeyField,
        /// Its value in the header.
        found: u64,
        /// Its value in the requested key.
        expected: u64,
    },
    /// A length in bytes differs from what the header or the family requires: the
    /// file is shorter than the header, the payload differs from the length recorded
    /// in the header, or the payload is shorter or longer than its table needs.
    #[error("length mismatch: found {found} bytes, expected {expected}")]
    Length {
        /// The length present.
        found: u64,
        /// The length required.
        expected: u64,
    },
    /// The FNV-1a checksum of the payload differs from the header.
    #[error("payload checksum {found:#018x}, expected {expected:#018x}")]
    Checksum {
        /// The checksum of the payload in the file.
        found: u64,
        /// The checksum in the header.
        expected: u64,
    },
    /// The payload has a valid checksum but an index out of range; only a file written
    /// by other code can have one.
    #[error("invalid {field} {found} in the payload, expected below {bound}")]
    InvalidPayload {
        /// What the index counts.
        field: &'static str,
        /// The index in the file.
        found: u64,
        /// The exclusive upper bound.
        bound: u64,
    },
}

impl CacheError {
    /// Whether this is an I/O error because the file does not exist.
    pub fn is_not_found(&self) -> bool {
        matches!(self, CacheError::Io { source, .. } if source.kind() == io::ErrorKind::NotFound)
    }
}

/// Returns a closure that wraps an [`io::Error`] on `path`.
fn io_error(path: &Path) -> impl Fn(io::Error) -> CacheError + '_ {
    move |source| CacheError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// What [`TableCache::load_or_build`] did.
#[derive(Debug)]
pub enum CacheOutcome {
    /// The table was loaded from its file.
    Loaded,
    /// There was no file; the table was built and stored.
    Built,
    /// The file was rejected for the given reason; the table was built and the file
    /// replaced.
    Rebuilt(CacheError),
    /// The table was built but could not be stored, for the given reason.
    NotStored(CacheError),
}

/// A table family that the cache stores: its kind, how to build it, and the encoding
/// of its payload (module documentation, "File format"). Sealed; implemented for
/// [`M2mTables`], [`L2lTables`], [`M2lTables`], [`M2lClasses`] and [`RotationTables`]
/// in f32 and f64.
pub trait CachedTable: Sized + sealed::Sealed {
    /// The stored scalar type.
    type Scalar: Stored;

    /// The family.
    const KIND: TableKind;

    /// Builds the table of degree `p`, in f64 and rounded entry by entry to
    /// [`CachedTable::Scalar`]: the family's own `build`.
    fn build(p: usize) -> Self;

    /// Returns the degree p.
    fn p(&self) -> usize;

    /// Writes the payload.
    fn encode(&self, out: &mut PayloadWriter<'_>);

    /// Reads the payload of a table of degree `p`, as written by
    /// [`CachedTable::encode`].
    ///
    /// # Errors
    ///
    /// [`CacheError::Length`] if the payload is too short, [`CacheError::InvalidPayload`]
    /// if an index is out of range, and [`CacheError::Io`] if reading fails.
    fn decode(p: usize, input: &mut PayloadReader<'_>) -> Result<Self, CacheError>;
}

/// The payload of a cache file being written: counts its bytes and hashes them. Created
/// only by [`TableCache::store`].
pub struct PayloadWriter<'a> {
    out: &'a mut dyn Write,
    len: u64,
    hash: u64,
    /// The first write error; later writes are skipped.
    error: Option<io::Error>,
}

impl<'a> PayloadWriter<'a> {
    /// Starts an empty payload on `out`.
    fn new(out: &'a mut dyn Write) -> Self {
        Self {
            out,
            len: 0,
            hash: FNV_OFFSET,
            error: None,
        }
    }

    /// Writes `bytes`, unless an earlier write failed.
    fn write_bytes(&mut self, bytes: &[u8]) {
        if self.error.is_some() {
            return;
        }
        self.len += bytes.len() as u64;
        self.hash = fnv1a(self.hash, bytes);
        if let Err(error) = self.out.write_all(bytes) {
            self.error = Some(error);
        }
    }

    /// Writes `values` in order, each as little-endian bytes of its precision.
    pub fn write_values<T: Stored>(&mut self, values: &[T]) {
        let width = T::PRECISION.bytes();
        let mut buffer = [0; CHUNK];
        for chunk in values.chunks(CHUNK / width) {
            let bytes = &mut buffer[..chunk.len() * width];
            for (&v, out) in chunk.iter().zip(bytes.chunks_exact_mut(width)) {
                v.write_le(out);
            }
            self.write_bytes(bytes);
        }
    }

    /// Writes an index as a little-endian u32.
    ///
    /// # Panics
    ///
    /// If `index` does not fit in a u32.
    pub fn write_index(&mut self, index: usize) {
        let index = u32::try_from(index).expect("a table index fits in a u32");
        self.write_bytes(&index.to_le_bytes());
    }

    /// Returns the payload length and checksum, or the first write error.
    fn finish(self) -> io::Result<(u64, u64)> {
        match self.error {
            Some(error) => Err(error),
            None => Ok((self.len, self.hash)),
        }
    }
}

/// The payload of a cache file being read: bounds reads by the length in the header
/// and hashes what it reads. Created only by [`TableCache::load`].
pub struct PayloadReader<'a> {
    input: &'a mut dyn Read,
    path: &'a Path,
    /// The payload length recorded in the header.
    len: u64,
    /// The bytes read so far.
    read: u64,
    hash: u64,
}

impl<'a> PayloadReader<'a> {
    /// Starts reading a payload of `len` bytes from `input`, the file `path`.
    fn new(input: &'a mut dyn Read, path: &'a Path, len: u64) -> Self {
        Self {
            input,
            path,
            len,
            read: 0,
            hash: FNV_OFFSET,
        }
    }

    /// Checks that `bytes` more bytes are left.
    fn reserve(&self, bytes: u64) -> Result<(), CacheError> {
        let needed = self.read.saturating_add(bytes);
        if needed > self.len {
            return Err(CacheError::Length {
                found: self.len,
                expected: needed,
            });
        }
        Ok(())
    }

    /// Fills `out` from the payload.
    fn read_bytes(&mut self, out: &mut [u8]) -> Result<(), CacheError> {
        self.reserve(out.len() as u64)?;
        self.input.read_exact(out).map_err(io_error(self.path))?;
        self.read += out.len() as u64;
        self.hash = fnv1a(self.hash, out);
        Ok(())
    }

    /// Reads `count` values, each as little-endian bytes of its precision.
    ///
    /// # Errors
    ///
    /// [`CacheError::Length`] if fewer are left, before anything is read or allocated;
    /// [`CacheError::Io`] if reading fails.
    pub fn read_values<T: Stored>(&mut self, count: usize) -> Result<Vec<T>, CacheError> {
        let width = T::PRECISION.bytes();
        self.reserve((count as u64).saturating_mul(width as u64))?;
        let mut values = Vec::with_capacity(count);
        let mut buffer = [0; CHUNK];
        let mut left = count;
        while left > 0 {
            let take = left.min(CHUNK / width);
            let bytes = &mut buffer[..take * width];
            self.read_bytes(bytes)?;
            values.extend(bytes.chunks_exact(width).map(T::read_le));
            left -= take;
        }
        Ok(values)
    }

    /// Reads an index written by [`PayloadWriter::write_index`] and checks that it is
    /// below `bound`.
    ///
    /// # Errors
    ///
    /// [`CacheError::InvalidPayload`] naming `field` if it is not below `bound`;
    /// [`CacheError::Length`] and [`CacheError::Io`] as for
    /// [`PayloadReader::read_values`].
    pub fn read_index(&mut self, field: &'static str, bound: usize) -> Result<usize, CacheError> {
        let mut bytes = [0; 4];
        self.read_bytes(&mut bytes)?;
        let index = u32::from_le_bytes(bytes);
        if u64::from(index) < bound as u64 {
            Ok(index as usize)
        } else {
            Err(CacheError::InvalidPayload {
                field,
                found: index.into(),
                bound: bound as u64,
            })
        }
    }

    /// Reads and hashes the rest of the payload.
    fn drain(&mut self) -> Result<(), CacheError> {
        let mut buffer = [0; CHUNK];
        while self.read < self.len {
            let take = (self.len - self.read).min(CHUNK as u64) as usize;
            self.read_bytes(&mut buffer[..take])?;
        }
        Ok(())
    }
}

/// A directory of cached tables (module documentation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableCache {
    dir: PathBuf,
}

/// Distinguishes the temporary files of one process.
static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

impl TableCache {
    /// A cache in the directory `dir`, chosen by the caller. Nothing is touched until
    /// the first load or store; [`TableCache::store`] creates `dir` if it is missing.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Returns the cache directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns the path of the file of `key`: [`CacheKey::file_name`] in the cache
    /// directory.
    pub fn path(&self, key: &CacheKey) -> PathBuf {
        self.dir.join(key.file_name())
    }

    /// Loads the table of family `F` and degree `p` from its file.
    ///
    /// Reads the file once, without holding a second copy of the payload, and checks
    /// the header, the payload length, the checksum and the payload structure in the
    /// order of the module documentation. The result is bit-identical to the table
    /// that was stored.
    ///
    /// # Errors
    ///
    /// Any [`CacheError`]; a missing file gives [`CacheError::Io`] with
    /// [`CacheError::is_not_found`].
    pub fn load<F: CachedTable>(&self, p: usize) -> Result<F, CacheError> {
        let key = CacheKey::of::<F>(p);
        let path = self.path(&key);
        let file = File::open(&path).map_err(io_error(&path))?;
        let file_len = file.metadata().map_err(io_error(&path))?.len();
        if file_len < HEADER_LEN as u64 {
            return Err(CacheError::Length {
                found: file_len,
                expected: HEADER_LEN as u64,
            });
        }
        let mut input = BufReader::with_capacity(1 << 16, file);
        let mut header = [0; HEADER_LEN];
        input.read_exact(&mut header).map_err(io_error(&path))?;
        let (payload_len, checksum) = key.check(&header)?;
        if file_len - HEADER_LEN as u64 != payload_len {
            return Err(CacheError::Length {
                found: file_len - HEADER_LEN as u64,
                expected: payload_len,
            });
        }
        let mut reader = PayloadReader::new(&mut input, &path, payload_len);
        let decoded = match F::decode(p, &mut reader) {
            Err(error @ CacheError::Io { .. }) => return Err(error),
            decoded => decoded,
        };
        let consumed = reader.read;
        // A corrupt payload is reported as such, even if decoding failed first.
        reader.drain()?;
        if reader.hash != checksum {
            return Err(CacheError::Checksum {
                found: reader.hash,
                expected: checksum,
            });
        }
        let table = decoded?;
        if consumed != payload_len {
            return Err(CacheError::Length {
                found: payload_len,
                expected: consumed,
            });
        }
        Ok(table)
    }

    /// Stores `table` under its key, atomically: into a new temporary file in the cache
    /// directory, synced, then renamed over the final name (module documentation).
    /// Creates the cache directory, and its missing parents, if needed. On failure the
    /// temporary file is removed and no file under the final name is changed.
    ///
    /// # Errors
    ///
    /// [`CacheError::Io`] if the directory or a file cannot be created, written or
    /// renamed.
    pub fn store<F: CachedTable>(&self, table: &F) -> Result<(), CacheError> {
        let key = CacheKey::of::<F>(table.p());
        fs::create_dir_all(&self.dir).map_err(io_error(&self.dir))?;
        let path = self.path(&key);
        let (temporary, file) = self.create_temporary(&key)?;
        let stored = write_file(file, &temporary, &key, table)
            .and_then(|()| fs::rename(&temporary, &path).map_err(io_error(&path)));
        if stored.is_err() {
            // Best effort: the error of the store is the one to report.
            let _ = fs::remove_file(&temporary);
        }
        stored
    }

    /// Loads the table of family `F` and degree `p`, or builds it with
    /// [`CachedTable::build`] and stores it if the file is missing or rejected. Never
    /// fails because of the cache; the [`CacheOutcome`] says what happened.
    pub fn load_or_build<F: CachedTable>(&self, p: usize) -> (F, CacheOutcome) {
        match self.load::<F>(p) {
            Ok(table) => (table, CacheOutcome::Loaded),
            Err(rejected) => {
                let table = F::build(p);
                let outcome = match self.store(&table) {
                    Ok(()) if rejected.is_not_found() => CacheOutcome::Built,
                    Ok(()) => CacheOutcome::Rebuilt(rejected),
                    Err(error) => CacheOutcome::NotStored(error),
                };
                (table, outcome)
            }
        }
    }

    /// Creates a new temporary file for `key` in the cache directory, named
    /// `.<file name>.<process id>-<counter>.tmp`; skips names that exist.
    fn create_temporary(&self, key: &CacheKey) -> Result<(PathBuf, File), CacheError> {
        loop {
            let counter = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
            let name = format!(".{}.{}-{counter}.tmp", key.file_name(), std::process::id());
            let path = self.dir.join(name);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(file) => return Ok((path, file)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(io_error(&path)(error)),
            }
        }
    }
}

/// Writes the file of `table` into `file`, at `path`: a zeroed header, the payload,
/// then the header itself, and syncs it.
fn write_file<F: CachedTable>(
    file: File,
    path: &Path,
    key: &CacheKey,
    table: &F,
) -> Result<(), CacheError> {
    let mut out = BufWriter::with_capacity(1 << 16, file);
    out.write_all(&[0; HEADER_LEN]).map_err(io_error(path))?;
    let mut payload = PayloadWriter::new(&mut out);
    table.encode(&mut payload);
    let (len, checksum) = payload.finish().map_err(io_error(path))?;
    let mut file = out
        .into_inner()
        .map_err(|error| io_error(path)(error.into_error()))?;
    file.seek(SeekFrom::Start(0)).map_err(io_error(path))?;
    file.write_all(&key.header(len, checksum))
        .map_err(io_error(path))?;
    file.sync_all().map_err(io_error(path))
}

/// Reads `count` matrices of degree `p`.
fn decode_matrices<T: Stored>(
    p: usize,
    count: usize,
    input: &mut PayloadReader<'_>,
) -> Result<MatrixSet<T>, CacheError> {
    let n = Layout::new(p).len();
    let data = input.read_values(n.saturating_mul(n).saturating_mul(count))?;
    Ok(MatrixSet::from_data(n, count, data))
}

/// Reads the M2M or L2L tables of degree `p`.
fn decode_octant<T: Stored>(
    p: usize,
    input: &mut PayloadReader<'_>,
) -> Result<OctantTables<T>, CacheError> {
    Ok(OctantTables {
        p,
        matrices: decode_matrices(p, OCTANT_COUNT, input)?,
    })
}

impl<T: Stored> sealed::Sealed for M2mTables<T> {}

impl<T: Stored> CachedTable for M2mTables<T> {
    type Scalar = T;
    const KIND: TableKind = TableKind::M2m;

    fn build(p: usize) -> Self {
        M2mTables::build(p)
    }

    fn p(&self) -> usize {
        self.inner.p
    }

    fn encode(&self, out: &mut PayloadWriter<'_>) {
        out.write_values(self.inner.matrices.as_slice());
    }

    fn decode(p: usize, input: &mut PayloadReader<'_>) -> Result<Self, CacheError> {
        Ok(Self {
            inner: decode_octant(p, input)?,
        })
    }
}

impl<T: Stored> sealed::Sealed for L2lTables<T> {}

impl<T: Stored> CachedTable for L2lTables<T> {
    type Scalar = T;
    const KIND: TableKind = TableKind::L2l;

    fn build(p: usize) -> Self {
        L2lTables::build(p)
    }

    fn p(&self) -> usize {
        self.inner.p
    }

    fn encode(&self, out: &mut PayloadWriter<'_>) {
        out.write_values(self.inner.matrices.as_slice());
    }

    fn decode(p: usize, input: &mut PayloadReader<'_>) -> Result<Self, CacheError> {
        Ok(Self {
            inner: decode_octant(p, input)?,
        })
    }
}

impl<T: Stored> sealed::Sealed for M2lTables<T> {}

impl<T: Stored> CachedTable for M2lTables<T> {
    type Scalar = T;
    const KIND: TableKind = TableKind::M2l;

    fn build(p: usize) -> Self {
        M2lTables::build(p)
    }

    fn p(&self) -> usize {
        self.p
    }

    fn encode(&self, out: &mut PayloadWriter<'_>) {
        out.write_values(self.matrices.as_slice());
    }

    fn decode(p: usize, input: &mut PayloadReader<'_>) -> Result<Self, CacheError> {
        Ok(Self {
            p,
            matrices: decode_matrices(p, M2L_OFFSET_COUNT, input)?,
        })
    }
}

impl<T: Stored> sealed::Sealed for M2lClasses<T> {}

impl<T: Stored> CachedTable for M2lClasses<T> {
    type Scalar = T;
    const KIND: TableKind = TableKind::M2lClasses;

    fn build(p: usize) -> Self {
        M2lClasses::build(p)
    }

    fn p(&self) -> usize {
        self.p
    }

    fn encode(&self, out: &mut PayloadWriter<'_>) {
        out.write_values(self.matrices.as_slice());
        for transform in self.multipole.iter().chain(&self.local) {
            out.write_values(&transform.blocks);
        }
        for &(class, element) in &self.offsets {
            out.write_index(class);
            out.write_index(element.index());
        }
    }

    fn decode(p: usize, input: &mut PayloadReader<'_>) -> Result<Self, CacheError> {
        let matrices = decode_matrices(p, M2L_CLASS_COUNT, input)?;
        let mut transforms = || {
            (0..GROUP_ORDER)
                .map(|_| {
                    Ok(CoefficientTransform {
                        p,
                        blocks: input.read_values(blocks_len(p))?,
                    })
                })
                .collect::<Result<Vec<_>, CacheError>>()
        };
        let multipole = transforms()?;
        let local = transforms()?;
        let offsets = (0..M2L_OFFSET_COUNT)
            .map(|_| {
                let class = input.read_index("class", M2L_CLASS_COUNT)?;
                let element = input.read_index("group element", GROUP_ORDER)?;
                Ok((class, SignedPermutation::from_index(element)))
            })
            .collect::<Result<_, CacheError>>()?;
        Ok(Self {
            p,
            matrices,
            multipole,
            local,
            offsets,
        })
    }
}

/// Writes the tables of one translation family (module documentation).
fn encode_shift_tables<T: Stored>(tables: &ShiftTables<T>, out: &mut PayloadWriter<'_>) {
    out.write_index(tables.polar_angles.len());
    out.write_index(tables.azimuth_angles.len());
    out.write_index(tables.distances.len());
    out.write_values(&tables.polar_angles);
    out.write_values(&tables.azimuth_angles);
    out.write_values(&tables.distances);
    out.write_values(&tables.forward);
    out.write_values(&tables.backward);
    out.write_values(&tables.azimuth);
    out.write_values(&tables.coaxial);
    for shift in &tables.shifts {
        let (tag, polar, azimuth) = match shift.alignment {
            Alignment::Up => (0, 0, 0),
            Alignment::Down => (1, 0, 0),
            Alignment::Rotated { polar, azimuth } => (2, polar, azimuth),
        };
        out.write_index(tag);
        out.write_index(polar);
        out.write_index(azimuth);
        out.write_index(shift.distance);
    }
}

/// Reads the tables of the translation family `operator` at degree `p`.
fn decode_shift_tables<T: Stored>(
    p: usize,
    operator: Operator,
    input: &mut PayloadReader<'_>,
) -> Result<ShiftTables<T>, CacheError> {
    // Every stored angle and distance is that of at least one entry.
    let entries = operator.count();
    let polar_count = input.read_index("number of polar angles", entries + 1)?;
    let azimuth_count = input.read_index("number of azimuths", entries + 1)?;
    let distance_count = input.read_index("number of distances", entries + 1)?;
    let polar_angles = input.read_values(polar_count)?;
    let azimuth_angles = input.read_values(azimuth_count)?;
    let distances = input.read_values(distance_count)?;
    let forward = input.read_values(polar_count * blocks_len(p))?;
    let backward = input.read_values(polar_count * blocks_len(p))?;
    let azimuth = input.read_values(azimuth_count * 2 * p)?;
    let coaxial = input.read_values(distance_count * coaxial_len(p))?;
    let shifts = (0..entries)
        .map(|_| {
            let tag = input.read_index("alignment", 3)?;
            // A shift on the axis stores the rotation indices 0.
            let (polar_bound, azimuth_bound) = if tag == 2 {
                (polar_count, azimuth_count)
            } else {
                (1, 1)
            };
            let polar = input.read_index("polar angle index", polar_bound)?;
            let azimuth = input.read_index("azimuth index", azimuth_bound)?;
            let distance = input.read_index("distance index", distance_count)?;
            let alignment = match tag {
                0 => Alignment::Up,
                1 => Alignment::Down,
                _ => Alignment::Rotated { polar, azimuth },
            };
            Ok(Shift {
                alignment,
                distance,
            })
        })
        .collect::<Result<_, CacheError>>()?;
    Ok(ShiftTables {
        operator,
        p,
        polar_angles,
        azimuth_angles,
        distances,
        forward,
        backward,
        azimuth,
        coaxial,
        shifts,
    })
}

impl<T: Stored> sealed::Sealed for RotationTables<T> {}

impl<T: Stored> CachedTable for RotationTables<T> {
    type Scalar = T;
    const KIND: TableKind = TableKind::Rotation;

    fn build(p: usize) -> Self {
        RotationTables::build(p)
    }

    fn p(&self) -> usize {
        self.p
    }

    fn encode(&self, out: &mut PayloadWriter<'_>) {
        encode_shift_tables(&self.m2m, out);
        encode_shift_tables(&self.l2l, out);
        encode_shift_tables(&self.m2l, out);
    }

    fn decode(p: usize, input: &mut PayloadReader<'_>) -> Result<Self, CacheError> {
        Ok(Self {
            p,
            m2m: decode_shift_tables(p, Operator::M2m, input)?,
            l2l: decode_shift_tables(p, Operator::L2l, input)?,
            m2l: decode_shift_tables(p, Operator::M2l, input)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a_matches_reference_vectors() {
        // Error measure: exact equality with the published FNV-1a 64-bit test vectors.
        assert_eq!(fnv1a(FNV_OFFSET, b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(FNV_OFFSET, b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(FNV_OFFSET, b"foobar"), 0x8594_4171_f739_67e8);
        // Hashing in pieces equals hashing at once.
        assert_eq!(
            fnv1a(fnv1a(FNV_OFFSET, b"foo"), b"bar"),
            0x8594_4171_f739_67e8
        );
    }

    #[test]
    fn file_names_follow_the_key() {
        // Error measure: exact string equality.
        let key = CacheKey::of::<M2lTables<f64>>(8);
        assert_eq!(
            key.file_name(),
            format!("m2l-p08-f64-c{CONVENTION_VERSION}-v{FORMAT_VERSION}.bin")
        );
        let key = CacheKey::of::<M2lClasses<f32>>(12);
        assert_eq!(
            key.file_name(),
            format!("m2l-classes-p12-f32-c{CONVENTION_VERSION}-v{FORMAT_VERSION}.bin")
        );
    }

    #[test]
    fn header_round_trips_and_checks_the_key() {
        // Error measure: exact equality of the decoded fields.
        let key = CacheKey::of::<RotationTables<f32>>(3);
        let header = key.header(1234, 0xdead_beef);
        assert_eq!(key.check(&header).unwrap(), (1234, 0xdead_beef));
        let other = CacheKey::of::<RotationTables<f64>>(3);
        assert!(matches!(
            other.check(&header),
            Err(CacheError::KeyMismatch {
                field: KeyField::Precision,
                found: 32,
                expected: 64
            })
        ));
    }

    #[test]
    fn values_round_trip_bit_for_bit() {
        // Error measure: bit-for-bit equality, including −0, subnormals and NaN.
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.5e-310,
            f64::MAX,
            f64::NAN,
            f64::INFINITY,
            1.0 / 3.0,
        ];
        let many: Vec<f64> = (0..3 * CHUNK).map(|i| i as f64 * 0.1).collect();
        let mut bytes = Vec::new();
        let mut writer = PayloadWriter::new(&mut bytes);
        writer.write_values(&values);
        writer.write_values(&many);
        writer.write_values(&values.map(|v| v as f32));
        writer.write_index(7);
        let (len, hash) = writer.finish().unwrap();
        assert_eq!(len as usize, bytes.len());
        assert_eq!(hash, fnv1a(FNV_OFFSET, &bytes));

        let mut input: &[u8] = &bytes;
        let path = Path::new("memory");
        let mut reader = PayloadReader::new(&mut input, path, len);
        let back: Vec<f64> = reader.read_values(values.len()).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&back), bits(&values));
        assert_eq!(bits(&reader.read_values(many.len()).unwrap()), bits(&many));
        let single: Vec<f32> = reader.read_values(values.len()).unwrap();
        for (s, v) in single.iter().zip(values) {
            assert_eq!(s.to_bits(), (v as f32).to_bits());
        }
        assert!(matches!(
            reader.read_index("test index", 7),
            Err(CacheError::InvalidPayload {
                found: 7,
                bound: 7,
                ..
            })
        ));
        assert!(matches!(
            reader.read_values::<f64>(1),
            Err(CacheError::Length { .. })
        ));
        assert_eq!(reader.hash, hash);
    }
}
