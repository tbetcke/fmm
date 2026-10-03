//! The error type of the crate (device-path.md §3.1, §12).

use crate::device::{BackendKind, Precision};

/// What can fail when opening a device, moving data or launching a kernel.
///
/// Violated preconditions of the launch wrappers (slice lengths that do not match, an
/// index out of range, repeated indices where they must be distinct) are invariants
/// of the caller and panic instead, as on the host path (device-path.md §12).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum KernelError {
    /// The backend's cargo feature is not enabled in this build.
    #[error("backend {backend} is not compiled in (enable the `{backend}` feature)")]
    NotCompiled {
        /// The backend asked for.
        backend: BackendKind,
    },
    /// The backend is compiled in, but no usable device came up: no adapter, or (for
    /// Metal) a device without the MSL compiler.
    #[error("no {backend} device: {reason}")]
    NoDevice {
        /// The backend asked for.
        backend: BackendKind,
        /// Why, in CubeCL's words or ours.
        reason: String,
    },
    /// The device does no arithmetic in this precision (f64 on Metal).
    #[error("backend {backend} does not support {precision}")]
    UnsupportedPrecision {
        /// The backend of the device.
        backend: BackendKind,
        /// The precision asked for.
        precision: Precision,
    },
    /// The request does not fit in the device memory the backend reports.
    #[error("{requested} bytes requested, {limit} bytes available on the device")]
    OutOfMemory {
        /// Bytes requested.
        requested: u64,
        /// Bytes still available: the reported limit less the bytes in use.
        limit: u64,
    },
    /// A buffer or launch needs more elements than the kernels index.
    #[error(
        "{what} of {len} elements exceeds the device limit of {} elements",
        crate::MAX_ELEMENTS
    )]
    TooLarge {
        /// What was too large.
        what: &'static str,
        /// Its length in elements.
        len: usize,
    },
    /// A buffer of another [`Device`](crate::Device).
    #[error("a buffer of another device")]
    WrongDevice,
    /// A kernel layout the device cannot run (T6): more units per cube or more shared
    /// memory than it has, or a plane layout on a device whose plane size varies.
    #[error("layout {layout} does not fit the device: {reason}")]
    UnsupportedLayout {
        /// The layout, as displayed.
        layout: String,
        /// Why it does not fit.
        reason: String,
    },
    /// A launch or transfer failed on the device, with CubeCL's message.
    #[error("device error: {reason}")]
    Device {
        /// CubeCL's message.
        reason: String,
    },
}
