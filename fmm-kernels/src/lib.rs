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
//! movement primitives ([`movement`]). T5: the plan's views and the box and leaf
//! coordinates on the device ([`view`]), and the cap on the CPU runtime's units per
//! cube ([`Device::limit_units`]). T6: the P2P kernel ([`p2p`]) in three layouts, and
//! the leaf stores' point offsets ([`view::PointOffsets`]). T7: the leaf operators P2M,
//! L2P, P2L and M2P ([`leaf`]). T8: the grouped translations with dense tables
//! ([`translate`]: the hand-written GEMM, the library GEMM, M2M and L2L; T9 adds dense
//! M2L). T10: rotation M2L ([`rotation`]). T11: timing windows ([`Device::open_window`]).
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
//! | `cuda` | [`BackendKind::Cuda`] | CUDA (LLVM NVPTX) | f32, f64 | by hand on locust (an H100); type-checked in CI |
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
//! transfers (calls and bytes), launches, syncs and timing windows.
//!
//! [`Device::open_window`] and [`Device::close_window`] bracket queued work in a timing
//! window ([`TimingWindow`], [`WindowTime`]; device-path.md §8.3), read after the work has
//! run. On a device that [times on itself](Device::times_on_device) (Metal with timestamp
//! queries, CUDA) a window waits for nothing; on the CPU runtime it drains the stream at
//! both ends, counted as two syncs.
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
//! On the CPU runtime the units per cube of these launches are capped by
//! [`Device::limit_units`] (default [`CPU_MAX_UNITS`]); `nd-fmm-exec` sets the cap from
//! its `threads(n)`, so that a rank keeps at most n of CubeCL's workers busy in them
//! (device-path.md §11).
//!
//! Index arrays are uploaded as an [`IndexBuffer`], which records on the host the bound
//! the wrappers check, so the kernels launch without bounds checks. Kernels take whole
//! buffers and element offsets as scalars; the launch shape is chosen per backend
//! (GPU: 256 units per cube, coalesced; CPU runtime: one cube, a contiguous block per
//! unit). No kernel uses atomics.
//!
//! ## Views
//!
//! [`view`] holds the plan's views on the device, uploaded once per FMM from plain `u32`
//! arrays (the crate does not depend on `nd-fmm-plan`) and validated on the host:
//! [`view::IndexView`] (a CSR), [`view::GroupedView`] (rows, batches and the
//! row-to-batch map of the grouped translations), [`view::BoxCoordinates`] and
//! [`view::LeafCoordinates`] (the integer indices the kernels form frames from), and
//! [`view::PointOffsets`] (the point offsets of a leaf store, validated as CSR offsets).
//!
//! ## P2P (T6)
//!
//! [`p2p::p2p`] adds one level's near field into the target output: potentials and,
//! optionally, gradients, f32 and f64, in the formulation of CONVENTIONS §3.13, "Device
//! kernels", and in the order of the host path. Three layouts of one kernel
//! ([`p2p::P2pLayout`]): one cube per target leaf with a shared-memory tile (the default
//! on Metal and CUDA), one plane per target leaf, and a CPU layout with targets in vector
//! lanes (the default on the CPU runtime, within 1.5× of `nd-fmm-simd` per pair at one
//! thread, decision 10). [`p2p::near_frames`] writes the frames the kernels form.
//!
//! ## Leaf operators (T7)
//!
//! [`leaf::p2m`], [`leaf::p2l`], [`leaf::l2p`] and [`leaf::m2p`] add one level's leaf
//! expansion operators, `nd_fmm_ref::leaf`'s operators at the exact frames of CONVENTIONS
//! §3.13, f32 and f64, potentials and optionally gradients, with p comptime up to
//! [`leaf::MAX_DEGREE`]. The solid harmonics follow the recursion of `nd_fmm_math::harmonics`
//! operation for operation, unrolled into a local array; they agree with the host to
//! rounding, not bit for bit (cubecl-opt's contraction). Two layouts
//! ([`leaf::LeafLayout`]): one cube per box or target leaf with coefficient owners and
//! shared-memory tiles (the default on Metal and CUDA), and a CPU layout without shared
//! memory (the default on the CPU runtime). [`leaf::harmonics`], [`leaf::x_frames`] and
//! [`leaf::w_frames`] expose the harmonics and frames for tests.
//!
//! ## Grouped translations (T8)
//!
//! [`translate::grouped`] runs one level's M2M or L2L (and, from T9, dense M2L) as in
//! device-path.md §6.4: per chunk of the view's batches a gather of the input columns
//! in batch order, one grouped GEMM over the level's groups into a temporary, and a
//! reduction per target in row order or a scatter-add; from T12 the gathered inputs and
//! products may also be laid out coefficient-major ([`translate::Orientation`], the GEMM
//! spike's orientation; the hand-written kernel's products are the same bits either way).
//! The hand-written GEMM
//! ([`translate::gemm`], [`translate::GemmLayout`]: a cube per tile on the GPUs, one unit per
//! core on the CPU runtime) sums each output from zero with explicit fmas in ascending
//! order, which a host `mul_add` loop repeats bit for bit; the library GEMM
//! (`cubek-matmul`, `SimpleCyclicCmma` named explicitly) runs f32 at p ≥ 8 on a GPU where
//! a probe at build accepts the shape and the input-precision guard passes
//! ([`translate::GroupedPlan`]). No atomics: the GEMM writes distinct columns, and the
//! reduction owns each target.
//!
//! ## Rotation M2L (T10)
//!
//! [`rotation::m2l`] runs one level's M2L by point and shoot (device-path.md §6.6): the M2L
//! family of the Phase 2 rotation tables uploaded once in its own storage
//! ([`rotation::RotationTables`]), one cube per box with a V pair (or a unit's share of
//! the boxes on the CPU runtime), each pair `ShiftTables::apply` step for step (z-rotation,
//! y-blocks, coaxial step, y-blocks back, z-rotation back, each followed by `sync_cube`;
//! the coaxial step alone on the z axis), the last step added into the box's accumulator
//! in row order. Every multiply–add is an explicit fma, so the kernel equals a host
//! `mul_add` loop in its order bit for bit, and `RotationTables::m2l` to rounding.
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
//! cargo check -p nd-fmm-kernels --features cuda                  # CUDA: the CI type-check
//! cargo test -p nd-fmm-kernels --features cuda --release -- --ignored    # CUDA, on locust
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
mod frame;
pub mod leaf;
pub mod movement;
pub mod p2p;
pub mod rotation;
pub mod translate;
pub mod view;

pub use buffer::{
    DeviceBuffer, DeviceElement, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexBuffer, IndexSlice,
};
pub use device::{
    BackendKind, CPU_MAX_UNITS, CUBECL_VERSION, Counters, Device, DeviceInfo, MAX_ELEMENTS,
    Precision, TimingWindow, WindowTime,
};
pub use error::KernelError;

/// The version of this crate, part of the key of every tuning-cache entry of
/// `nd-fmm-exec` (Phase 4 T12): a cache written by another version is stale.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
