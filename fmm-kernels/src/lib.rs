//! # `nd-fmm-kernels`
//!
//! The CubeCL device kernels of the nd-project FMM (Phase 4). The crate hides CubeCL
//! behind its own [`Device`], [`DeviceBuffer`] and safe launch wrappers, so that
//! `nd-fmm-exec` reaches the device without a `cubecl` dependency and without
//! `unsafe`. Its design is [docs/design/device-path.md][design], §3.1; it needs no MPI,
//! no `nd-fmm-plan` and no rayon, and takes plain `u32` index arrays and `f32`/`f64`
//! data.
//!
//! **Status.** Phase 4 T4: backends, the capability check, device buffers and the data
//! movement primitives ([`movement`]). The operator kernels follow in T6–T10.
//!
//! ## Backends and features
//!
//! The backend is a run-time value, [`BackendKind`], never a type parameter: CubeCL 0.11's
//! client is one type for every runtime (device-path.md §3.2). Each backend is a cargo
//! feature, and none is on by default.
//!
//! | Feature | [`BackendKind`] | CubeCL runtime | Precisions | Run here |
//! | --- | --- | --- | --- | --- |
//! | none | — | none: kernel definitions and host-side types only | — | [`Device::open`] gives [`KernelError::NotCompiled`] |
//! | `cpu` | [`BackendKind::Cpu`] | the CPU runtime (LLVM JIT; its build downloads the `tracel-llvm` bundle) | f32, f64 | yes: the correctness backend, and the CI candidate |
//! | `metal` | [`BackendKind::Metal`] | wgpu with the MSL compiler (`cubecl/metal` = `wgpu-msl`) | f32 | by hand, outside the macOS sandbox |
//! | `cuda` | [`BackendKind::Cuda`] | CUDA (LLVM NVPTX) | f32, f64 (registered) | never: type-checked only |
//!
//! [`Device::open`] uses CubeCL's fallible device constructors and refuses a Metal
//! device that came up without the MSL compiler (a silent WGSL fallback), recognised by
//! the plane-sync and CMMA features only the MSL path registers. [`DeviceInfo`] names
//! the backend, compiler, device and CubeCL version for reports; its `Display` is one
//! line, `metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32`.
//!
//! The crate never reads an environment variable to choose a backend or device.
//! CubeCL reads its own (`CUBECL_CPU_STACK_MB`, `CUBECL_DEBUG_LOG`, …) and a
//! `cubecl.toml` found upward from the working directory (device-path.md F24); the
//! crate sets none of them. CubeCL's `persistence` feature stays off with `cpu` and
//! `cuda` (the workspace entries set `default-features = false`). With `metal` it is on
//! regardless, through CubeCL's own manifests (`cubecl-wgpu` takes `cubecl-cpp` with
//! its defaults), and opening a Metal device creates an empty store
//! `target/environment/default.db`. The crate never autotunes or measures throughput
//! through CubeCL, so nothing is recorded in it.
//!
//! ## Capability
//!
//! f32 runs on every backend. f64 runs where the device does arithmetic in it, as
//! CubeCL's type registry reports ([`Device::supports`]): the CPU runtime here, CUDA in
//! principle, never Metal. Where it does not, [`Device::require`] and every
//! allocation of an `f64` buffer return [`KernelError::UnsupportedPrecision`]: f64 is
//! refused with a typed error, never by a panic or a cast (README requirement 7).
//!
//! ## Buffers and transfers
//!
//! A [`DeviceBuffer<E>`] holds `f32`, `f64` or `u32` ([`DeviceElement`]). Index arrays
//! are `u32` only on the device; `u16` and `u8` indices are widened at upload
//! ([`Device::upload_indices_widened`]). Sizes 0 and 1 work. [`Device::upload`] and
//! [`Device::alloc`] check the length and the device memory first, because CubeCL
//! panics on a failed allocation. [`Device::write`] overwrites a range without a sync;
//! [`Device::download`] waits for the queued launches (one sync) and reports a failed
//! launch that wrote the buffer as [`KernelError::Device`]. [`Device::counters`] counts
//! transfers (calls and bytes), launches and syncs.
//!
//! Transfers are copies: every bit pattern round-trips, −0, subnormals, ±∞ and NaN
//! payloads included, on every backend run (Metal flushes subnormals in arithmetic
//! only, CONVENTIONS §3.13, "Device kernels").
//!
//! ## Data movement
//!
//! [`movement`] holds the primitives every later kernel needs, each a safe wrapper
//! over one `#[cube]` kernel that allocates nothing and launches nothing for an empty
//! range:
//!
//! - [`movement::zero`]: `x[range] = 0`;
//! - [`movement::gather_columns`]: `y[:, j] = x[:, idx[j]]`, columns of n values (n
//!   comptime); indices may repeat;
//! - [`movement::scatter_add_columns`]: `x[:, idx[j]] += y[:, j]`, the indices of one
//!   launch distinct (checked on the host in debug builds);
//! - [`movement::scatter_values`]: `x[idx[j]] = y[j]`, the indices distinct.
//!
//! Index arrays are uploaded as an [`IndexBuffer`], which records on the host the bound
//! the wrappers check, so the kernels launch without bounds checks. Kernels take whole
//! buffers and element offsets as scalars; the launch shape is chosen per backend
//! (GPU: 256 units per cube, coalesced; CPU runtime: one cube, a contiguous block per
//! unit). No kernel uses atomics.
//!
//! ## Tests on each backend
//!
//! From the repository root (`cargo test` prints the backends it ran with
//! `--show-output`):
//!
//! ```sh
//! cargo test -p nd-fmm-kernels                                   # no runtime: seconds
//! cargo test -p nd-fmm-kernels --features cpu --release          # the CPU runtime
//! cargo test -p nd-fmm-kernels --features metal --release -- --ignored   # Metal, by hand
//! cargo check -p nd-fmm-kernels --features cuda                  # CUDA: type-checked only
//! ```
//!
//! Metal needs a process with GPU access: it fails inside the macOS sandbox ("No
//! possible adapter available"). The `cpu` feature's first build downloads the
//! `tracel-llvm` bundle (fmm-kernels/CLAUDE.md has the sandbox notes).
//!
//! [design]: https://github.com/tbetcke/fmm/blob/main/docs/design/device-path.md

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]

mod buffer;
mod device;
mod error;
pub mod movement;

pub use buffer::{
    DeviceBuffer, DeviceElement, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexBuffer, IndexSlice,
};
pub use device::{
    BackendKind, CUBECL_VERSION, Counters, Device, DeviceInfo, MAX_ELEMENTS, Precision,
};
pub use error::KernelError;
