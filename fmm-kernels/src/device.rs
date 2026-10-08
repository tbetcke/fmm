//! Backends, devices, the capability query, transfers and counters
//! (device-path.md §3.1, §5.1).

use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};

use cubecl::client::Client;
use cubecl::features::Plane;
use cubecl::prelude::*;
use cubecl::profile::{ProfileDuration, TimingMethod};
use cubecl::server::ProfileError;

use crate::buffer::{DeviceBuffer, DeviceElement, DeviceSlice, DeviceSliceMut, IndexBuffer};
use crate::error::KernelError;
use crate::movement;

/// The CubeCL version this crate is built against (the workspace pin), for reports.
pub const CUBECL_VERSION: &str = "0.11.0-pre.4";

/// The most elements a buffer may hold. Kernels index with 32-bit arithmetic, and
/// keeping lengths below 2³¹ leaves room for a grid stride past the end.
pub const MAX_ELEMENTS: usize = (1 << 31) - 1;

/// A backend this crate can name. Whether it is compiled in is a run-time question, so a
/// build without a backend can still name it and refuse it with
/// [`KernelError::NotCompiled`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    /// The CubeCL CPU runtime (LLVM JIT), feature `cpu`: f32 and f64, the correctness
    /// backend for f64.
    Cpu,
    /// wgpu with the MSL compiler, feature `metal`: f32 only.
    Metal,
    /// CUDA (LLVM NVPTX), feature `cuda`: f32 and f64, run by hand on locust's H100
    /// (Phase 4S T2); type-checked in CI.
    Cuda,
}

impl BackendKind {
    /// Every backend, in report order.
    pub const ALL: [Self; 3] = [Self::Cpu, Self::Metal, Self::Cuda];

    /// The name used in reports and on command lines: `cpu`, `metal` or `cuda`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// The backend of a [`name`](Self::name), compiled in or not.
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == name)
    }

    /// True if this build has the backend's cargo feature.
    pub fn is_compiled(self) -> bool {
        match self {
            Self::Cpu => cfg!(feature = "cpu"),
            Self::Metal => cfg!(feature = "metal"),
            Self::Cuda => cfg!(feature = "cuda"),
        }
    }

    /// True for the GPU backends, whose kernels use GPU-shaped launches.
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A floating-point precision a device may or may not do arithmetic in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
}

impl Precision {
    /// `f32` or `f64`.
    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What an opened device reports, for reports and for per-backend choices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The backend.
    pub backend: BackendKind,
    /// The device's name as the runtime reports it (the adapter name on Metal).
    pub name: String,
    /// The runtime and compiler (`client.name()`): `wgpu<msl>` on Metal.
    pub compiler: String,
    /// The CubeCL version, [`CUBECL_VERSION`].
    pub cubecl_version: &'static str,
    /// Whether the device does arithmetic in f32.
    pub f32: bool,
    /// Whether the device does arithmetic in f64.
    pub f64: bool,
    /// The smallest and largest plane (SIMD group) size.
    pub plane_size: (u32, u32),
    /// The largest shared memory per cube, in bytes.
    pub max_shared_memory: usize,
    /// The most units per cube.
    pub max_units_per_cube: u32,
    /// The most cubes per launch in each dimension.
    pub max_cube_count: (u32, u32, u32),
    /// The device memory the backend reports as usable, in bytes, if it reports one.
    pub max_memory: Option<u64>,
}

impl DeviceInfo {
    /// The precisions the device does arithmetic in, in order.
    pub fn precisions(&self) -> Vec<Precision> {
        [(Precision::F32, self.f32), (Precision::F64, self.f64)]
            .into_iter()
            .filter_map(|(p, on)| on.then_some(p))
            .collect()
    }

    /// Whether the device does arithmetic in `precision`.
    pub fn supports(&self, precision: Precision) -> bool {
        match precision {
            Precision::F32 => self.f32,
            Precision::F64 => self.f64,
        }
    }
}

impl fmt::Display for DeviceInfo {
    /// One line: `metal (wgpu<msl>), Apple M3 Max, CubeCL 0.11.0-pre.4, f32`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let precisions: Vec<_> = self.precisions().iter().map(|p| p.name()).collect();
        write!(
            f,
            "{} ({}), {}, CubeCL {}, {}",
            self.backend,
            self.compiler,
            self.name,
            self.cubecl_version,
            if precisions.is_empty() {
                "no float precision".to_string()
            } else {
                precisions.join(" ")
            }
        )
    }
}

/// Transfers, launches and syncs of a [`Device`] since it was opened or the counters
/// were last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counters {
    /// Bytes copied from the host to the device.
    pub upload_bytes: u64,
    /// Host-to-device copies ([`Device::upload`], [`Device::upload_indices`],
    /// [`Device::write`], [`Device::write_owned`]), empty ones included.
    pub uploads: u64,
    /// Bytes copied from the device to the host.
    pub download_bytes: u64,
    /// Device-to-host copies ([`Device::download`], [`Device::download_view`]), empty
    /// ones included.
    pub downloads: u64,
    /// Kernel launches. A call with nothing to do launches nothing.
    pub launches: u64,
    /// Waits for the device: one per [`Device::download`], [`Device::download_view`] and
    /// [`Device::sync`], and two per timing window not timed on the device
    /// ([`Device::close_window`]).
    pub syncs: u64,
    /// Timing windows closed ([`Device::close_window`]).
    pub windows: u64,
}

/// A timing window open on a device's stream ([`Device::open_window`]).
#[derive(Debug)]
pub struct TimingWindow {
    window: cubecl::client::ProfileWindow,
    /// The device's launches before the window opened ([`Device`]'s `launched`).
    launched: u64,
}

/// The time of a closed [`TimingWindow`] ([`Device::close_window`]), read with
/// [`resolve`](Self::resolve) once the window's work has run.
pub struct WindowTime {
    /// `None` for a window without device work.
    duration: Option<ProfileDuration>,
    on_device: bool,
}

impl fmt::Debug for WindowTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WindowTime")
            .field("on_device", &self.on_device)
            .finish_non_exhaustive()
    }
}

impl WindowTime {
    /// Whether the device timed the window without waiting for the stream
    /// ([`Device::times_on_device`]).
    pub fn on_device(&self) -> bool {
        self.on_device
    }

    /// The time of the window: on a device that times on the device, from its
    /// timestamps, which waits for the buffer that carries them (call it after the
    /// download that ends the work, when the stream is idle); elsewhere the host time
    /// between the window's two waits. `None` if the runtime measured nothing (a window
    /// without device work).
    pub fn resolve(self) -> Option<std::time::Duration> {
        let duration = self.duration?;
        cubecl::future::block_on(duration.resolve()).map(|ticks| ticks.duration())
    }
}

/// A launch shape for kernels that cover `work` elements: `cubes` cubes of `units`
/// units, each unit taking blocks of `chunk` consecutive elements, one block per
/// stride of `units · cubes` blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Grid {
    pub units: u32,
    pub cubes: u32,
    pub chunk: u32,
}

impl Grid {
    /// The units of a launch, `units · cubes`.
    pub fn threads(self) -> u32 {
        self.units * self.cubes
    }

    /// The shape of an elementwise launch over `work ≥ 1` elements, with the CPU
    /// runtime's units per cube capped at `cpu_units` ([`Device::limit_units`]).
    ///
    /// - GPU: 256 units per cube, one element per unit and stride (coalesced), as many
    ///   cubes as cover `work`, capped at the device's limit (the units then stride).
    ///   `cpu_units` does not apply.
    /// - CPU runtime: each unit is a task on CubeCL's worker pool and loops over the
    ///   cubes (device-path.md F18), so one cube, and each unit one contiguous block:
    ///   one unit below [`CPU_MIN_CHUNK`] elements, else up to `max_units_per_cube`
    ///   (at most [`CPU_MAX_UNITS`] and at most `cpu_units`). Only two cube sizes occur
    ///   per cap, so few kernel variants are compiled.
    pub fn elementwise(info: &DeviceInfo, work: usize, cpu_units: u32) -> Self {
        debug_assert!((1..=MAX_ELEMENTS).contains(&work));
        if info.backend.is_gpu() {
            let units = GPU_UNITS.min(info.max_units_per_cube.max(1));
            let cubes = work
                .div_ceil(units as usize)
                .min(info.max_cube_count.0.clamp(1, GPU_MAX_CUBES) as usize);
            Self {
                units,
                cubes: cubes as u32,
                chunk: 1,
            }
        } else {
            let max_units = info
                .max_units_per_cube
                .clamp(1, CPU_MAX_UNITS)
                .min(cpu_units.max(1));
            let units = if work < CPU_MIN_CHUNK { 1 } else { max_units };
            Self {
                units,
                cubes: 1,
                chunk: work.div_ceil(units as usize) as u32,
            }
        }
    }
}

/// Units per cube of the GPU elementwise launches.
const GPU_UNITS: u32 = 256;
/// The most cubes of a GPU elementwise launch (wgpu's limit per dimension).
const GPU_MAX_CUBES: u32 = 65_535;
/// The most units per cube of a CPU-runtime elementwise launch (T2, T3: 12–16 units
/// per cube), and the default of [`Device::limit_units`].
pub const CPU_MAX_UNITS: u32 = 16;
/// Below this many elements a CPU-runtime elementwise launch uses one unit.
pub(crate) const CPU_MIN_CHUNK: usize = 1 << 14;

/// Values downloaded from a device ([`Device::download_view`]), read in place in
/// CubeCL's host memory: pinned memory from the CUDA runtime's pool, a mapped staging
/// buffer on wgpu, host memory on the CPU runtime. Derefs to `&[E]`, without a copy into a
/// caller's slice.
///
/// It owns CubeCL's host buffer, which goes back to CubeCL's pool when it is dropped.
/// Drop it before the next download: a view held across a later download makes the pool
/// keep a second buffer beside it (spikes/download-path/REPORT.md, "CubeCL's host pools").
/// `nd-fmm-exec` ties its views to a borrow of the operator, so none outlives its
/// evaluation.
pub struct HostValues<E: DeviceElement> {
    /// `None` for an empty range.
    bytes: Option<cubecl::bytes::Bytes>,
    _element: PhantomData<E>,
}

impl<E: DeviceElement> HostValues<E> {
    fn new(bytes: Option<cubecl::bytes::Bytes>) -> Self {
        Self {
            bytes,
            _element: PhantomData,
        }
    }
}

impl<E: DeviceElement> Deref for HostValues<E> {
    type Target = [E];

    fn deref(&self) -> &[E] {
        self.bytes
            .as_ref()
            .map_or(&[], |bytes| E::from_bytes(bytes))
    }
}

impl<E: DeviceElement> fmt::Debug for HostValues<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostValues")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

/// The source of [`Device`] identities, so that buffers can name their device.
static NEXT_DEVICE_ID: AtomicU64 = AtomicU64::new(0);

/// An opened device: one `cubecl::Device` and its client, what it reports, and the
/// counters of transfers, launches and syncs.
///
/// Every operation is issued from the thread that holds the `Device`, so all of them
/// land on one CubeCL stream and run in order (device-path.md F10, §6.1). The type is
/// `Send`; move it between threads, never share it.
pub struct Device {
    client: Client,
    info: DeviceInfo,
    id: u64,
    counters: Counters,
    /// Every launch since the device was opened, never reset: a timing window with none
    /// in it measures nothing.
    launched: u64,
    /// The cap on the units per cube of the CPU runtime's launches.
    cpu_units: u32,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("info", &self.info)
            .field("id", &self.id)
            .field("counters", &self.counters)
            .field("cpu_units", &self.cpu_units)
            .finish_non_exhaustive()
    }
}

impl Device {
    /// Opens the default device of `kind` through CubeCL 0.11's fallible constructors
    /// (`cubecl::Device::cpu`, `metal_msl`, `cuda(0)`).
    ///
    /// Metal must come up with the MSL compiler. wgpu-msl falls back to WGSL silently
    /// when the GPU family check or the MSL 3.2 canary fails, and `client.name()` says
    /// `wgpu<msl>` either way; only the MSL path registers `Plane::Sync` and the CMMA
    /// combinations, so a device without both is refused (device-path.md §3.1, F4).
    ///
    /// # Errors
    ///
    /// [`KernelError::NotCompiled`] if the backend's feature is off;
    /// [`KernelError::NoDevice`] if the runtime finds no such device (Metal inside the
    /// macOS sandbox) or Metal came up without MSL.
    ///
    /// A panic inside CubeCL's client creation (device-path.md F3) is not caught; the
    /// constructors enumerate the devices first, so it is not expected.
    pub fn open(kind: BackendKind) -> Result<Self, KernelError> {
        if !kind.is_compiled() {
            return Err(KernelError::NotCompiled { backend: kind });
        }
        let device = match kind {
            BackendKind::Cpu => cubecl::Device::cpu(),
            BackendKind::Metal => {
                cubecl::Device::metal_msl(cubecl::device::WgpuDeviceKind::DefaultDevice)
            }
            BackendKind::Cuda => cubecl::Device::cuda(0),
        }
        .map_err(|e| match e {
            cubecl::device::DeviceUnavailable::NotLinked(_) => {
                KernelError::NotCompiled { backend: kind }
            }
            other => KernelError::NoDevice {
                backend: kind,
                reason: other.to_string(),
            },
        })?;
        let client = device.client();
        let props = client.properties();
        if kind == BackendKind::Metal
            && !(props.features.plane.contains(Plane::Sync)
                && !props.features.matmul.cmma.is_empty())
        {
            return Err(KernelError::NoDevice {
                backend: kind,
                reason: "the device came up without the MSL compiler (WGSL fallback)".into(),
            });
        }
        let hw = &props.hardware;
        let info = DeviceInfo {
            backend: kind,
            name: props.identity.name.clone(),
            compiler: client.name().to_string(),
            cubecl_version: CUBECL_VERSION,
            f32: does_arithmetic::<f32>(&client),
            f64: does_arithmetic::<f64>(&client),
            plane_size: (hw.plane_size_min, hw.plane_size_max),
            max_shared_memory: hw.max_shared_memory_size,
            max_units_per_cube: hw.max_units_per_cube,
            max_cube_count: hw.max_cube_count,
            max_memory: props.memory.max_memory(),
        };
        Ok(Self {
            client,
            info,
            id: NEXT_DEVICE_ID.fetch_add(1, Ordering::Relaxed),
            counters: Counters::default(),
            launched: 0,
            cpu_units: CPU_MAX_UNITS,
        })
    }

    /// What the device reports.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// The backend of the device.
    pub fn backend(&self) -> BackendKind {
        self.info.backend
    }

    /// Whether the device does arithmetic in `precision`: CubeCL's type registry lists
    /// the type with arithmetic among its uses (device-path.md F12, §5.1).
    pub fn supports(&self, precision: Precision) -> bool {
        self.info.supports(precision)
    }

    /// Refuses a precision the device does not do arithmetic in.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedPrecision`], for f64 on Metal.
    pub fn require(&self, precision: Precision) -> Result<(), KernelError> {
        if self.supports(precision) {
            Ok(())
        } else {
            Err(KernelError::UnsupportedPrecision {
                backend: self.backend(),
                precision,
            })
        }
    }

    /// Caps the units per cube of the CPU runtime's layouts at `units` (at least 1):
    /// the elementwise launches of [`movement`](crate::movement) use at most that many,
    /// so that a rank keeps at most `units` of CubeCL's worker threads busy in them
    /// (device-path.md §11; `nd-fmm-exec` passes its `threads(n)`). The default is
    /// [`CPU_MAX_UNITS`], 16; a larger cap changes nothing. GPU backends ignore it.
    /// Kernels with shared memory or `sync_cube` keep their own cube size.
    pub fn limit_units(&mut self, units: u32) {
        self.cpu_units = units.clamp(1, CPU_MAX_UNITS);
    }

    /// The cap of [`limit_units`](Self::limit_units): 1–16, [`CPU_MAX_UNITS`] by
    /// default.
    pub fn units_cap(&self) -> u32 {
        self.cpu_units
    }

    /// The bytes still available on the device: the limit the backend reports
    /// ([`DeviceInfo::max_memory`]) less the bytes CubeCL holds, or `None` if the
    /// backend reports no limit. [`alloc`](Self::alloc) and [`upload`](Self::upload)
    /// check every request against it; a caller sums its allocations first to refuse a
    /// configuration before allocating anything (device-path.md §4.6).
    pub fn available_memory(&self) -> Option<u64> {
        self.info.max_memory.map(|limit| {
            let usage = self.client.memory_usage();
            limit.saturating_sub(usage.bytes_in_use + usage.bytes_padding)
        })
    }

    /// The bytes a buffer of `len` elements of `E` occupies on the device: at least one
    /// element, so that an empty buffer still has a handle to bind. The sum over a
    /// configuration's buffers is what [`available_memory`](Self::available_memory)
    /// must cover.
    pub fn buffer_bytes<E: DeviceElement>(len: usize) -> u64 {
        storage_bytes::<E>(len) as u64
    }

    /// The counters since the device was opened or [`reset_counters`](Self::reset_counters).
    pub fn counters(&self) -> Counters {
        self.counters
    }

    /// Sets every counter to zero.
    pub fn reset_counters(&mut self) {
        self.counters = Counters::default();
    }

    /// Allocates `len` elements set to zero (+0.0 for floats) by a kernel, with no
    /// transfer.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedPrecision`] for a float type the device does not
    /// support, [`KernelError::TooLarge`] past [`MAX_ELEMENTS`],
    /// [`KernelError::OutOfMemory`] if the device reports a limit and `len` elements do
    /// not fit under it. The check comes first because CubeCL panics when an
    /// allocation fails (device-path.md F7).
    pub fn alloc<E: DeviceElement>(&mut self, len: usize) -> Result<DeviceBuffer<E>, KernelError> {
        self.check_element::<E>(len)?;
        let handle = self.client.empty(storage_bytes::<E>(len));
        let mut buffer = DeviceBuffer::new(handle, len, self.id);
        movement::zero(self, buffer.as_slice_mut())?;
        Ok(buffer)
    }

    /// Allocates a buffer holding a copy of `data`: one upload.
    ///
    /// # Errors
    ///
    /// As [`alloc`](Self::alloc).
    pub fn upload<E: DeviceElement>(&mut self, data: &[E]) -> Result<DeviceBuffer<E>, KernelError> {
        self.check_element::<E>(data.len())?;
        let handle = if data.is_empty() {
            self.client.empty(storage_bytes::<E>(0))
        } else {
            self.client.create_from_slice(E::as_bytes(data))
        };
        self.count_upload(size_of_val(data));
        Ok(DeviceBuffer::new(handle, data.len(), self.id))
    }

    /// Uploads an index array and records, on the host, the bound every launch checks
    /// its indices against: one more than the largest entry. Debug builds also keep a
    /// host copy, against which the scatters check that the indices of a launch are
    /// distinct.
    ///
    /// # Errors
    ///
    /// As [`alloc`](Self::alloc).
    pub fn upload_indices(&mut self, indices: &[u32]) -> Result<IndexBuffer, KernelError> {
        let buffer = self.upload(indices)?;
        Ok(IndexBuffer::new(buffer, indices))
    }

    /// [`upload_indices`](Self::upload_indices) for `u16` or `u8` indices (the offset
    /// indices and octants of `GroupedCsr`), widened to `u32` on the host first: the
    /// device holds `u32` indices only (device-path.md §3.1).
    ///
    /// # Errors
    ///
    /// As [`alloc`](Self::alloc).
    pub fn upload_indices_widened<I: Copy + Into<u32>>(
        &mut self,
        indices: &[I],
    ) -> Result<IndexBuffer, KernelError> {
        let wide: Vec<u32> = indices.iter().map(|&i| i.into()).collect();
        self.upload_indices(&wide)
    }

    /// Overwrites `slice` with `data`, without a sync: one upload, ordered after the
    /// launches already queued. The data is copied first (CubeCL takes an owned buffer);
    /// a caller that can give up its buffer uses [`write_owned`](Self::write_owned),
    /// which does not copy.
    ///
    /// # Errors
    ///
    /// [`KernelError::WrongDevice`] for a buffer of another device.
    ///
    /// # Panics
    ///
    /// If `data` and `slice` differ in length.
    pub fn write<E: DeviceElement>(
        &mut self,
        slice: DeviceSliceMut<'_, E>,
        data: &[E],
    ) -> Result<(), KernelError> {
        assert_eq!(slice.len(), data.len(), "write: slice and data lengths");
        self.write_owned(slice, data.to_vec())
    }

    /// [`write`](Self::write) of a buffer the caller gives up: `data` is handed to CubeCL
    /// as it is (`Bytes::from_elems`), without the copy `write` makes. One upload, no
    /// sync, the same counters, checks and errors as `write` (Phase 4S T11).
    ///
    /// # Errors
    ///
    /// [`KernelError::WrongDevice`] for a buffer of another device.
    ///
    /// # Panics
    ///
    /// If `data` and `slice` differ in length.
    pub fn write_owned<E: DeviceElement>(
        &mut self,
        slice: DeviceSliceMut<'_, E>,
        data: Vec<E>,
    ) -> Result<(), KernelError> {
        assert_eq!(slice.len(), data.len(), "write: slice and data lengths");
        self.check_owner(slice.device())?;
        let bytes = size_of_val(data.as_slice());
        if !data.is_empty() {
            let handle = slice.byte_range_handle();
            self.client
                .write(&handle, cubecl::bytes::Bytes::from_elems(data));
        }
        self.count_upload(bytes);
        Ok(())
    }

    /// Copies `slice` into `out`: waits for every queued launch (one sync) and returns
    /// any launch error attached to the buffer (device-path.md F9). The copy of
    /// [`download_view`](Self::download_view)'s values, with its counters and errors.
    ///
    /// # Errors
    ///
    /// [`KernelError::WrongDevice`]; [`KernelError::Device`] with CubeCL's message if a
    /// launch that wrote the buffer, or the copy, failed.
    ///
    /// # Panics
    ///
    /// If `out` and `slice` differ in length.
    pub fn download<E: DeviceElement>(
        &mut self,
        slice: DeviceSlice<'_, E>,
        out: &mut [E],
    ) -> Result<(), KernelError> {
        assert_eq!(slice.len(), out.len(), "download: slice and output lengths");
        let values = self.download_view(slice)?;
        out.copy_from_slice(&values);
        Ok(())
    }

    /// Downloads `slice` and lends CubeCL's host copy of it, without copying it into a
    /// caller's slice (Phase 4S T11, decision 14): waits for every queued launch (one
    /// sync) and returns any launch error attached to the buffer (device-path.md F9,
    /// §12). Counts one download of `slice`'s bytes and one sync, as
    /// [`download`](Self::download) does.
    ///
    /// The values stay in CubeCL's host memory (pinned memory from CUDA's pool, a mapped
    /// staging buffer on wgpu) until the [`HostValues`] is dropped; drop it before the next
    /// download.
    ///
    /// # Errors
    ///
    /// [`KernelError::WrongDevice`]; [`KernelError::Device`] with CubeCL's message if a
    /// launch that wrote the buffer, or the copy, failed.
    pub fn download_view<E: DeviceElement>(
        &mut self,
        slice: DeviceSlice<'_, E>,
    ) -> Result<HostValues<E>, KernelError> {
        self.check_owner(slice.device())?;
        self.counters.downloads += 1;
        self.counters.syncs += 1;
        let handle = slice.byte_range_handle();
        let len = slice.len();
        if len == 0 {
            cubecl::future::block_on(self.client.sync_buffers([&handle])).map_err(device_error)?;
            return Ok(HostValues::new(None));
        }
        let bytes = self.client.read_one(handle).map_err(device_error)?;
        assert_eq!(
            E::from_bytes(&bytes).len(),
            len,
            "download: CubeCL returned another length"
        );
        self.counters.download_bytes += (len * size_of::<E>()) as u64;
        Ok(HostValues::new(Some(bytes)))
    }

    /// Waits for every queued launch and transfer: one sync.
    ///
    /// # Errors
    ///
    /// [`KernelError::Device`] for a device fault. A failed launch is reported by the
    /// download of a buffer it wrote, not here (device-path.md F9).
    pub fn sync(&mut self) -> Result<(), KernelError> {
        self.counters.syncs += 1;
        cubecl::future::block_on(self.client.sync()).map_err(device_error)
    }

    /// Whether a [timing window](Self::open_window) is timed by the device itself,
    /// without waiting for it (device-path.md §8.3, F14): on a GPU backend whose runtime
    /// times on the device (wgpu with timestamp queries, which Metal has; CUDA by
    /// events). False on the CPU runtime: it reports device timing, but its windows
    /// drain the stream at both ends and read the host clock, so each window costs two
    /// waits.
    pub fn times_on_device(&self) -> bool {
        self.backend().is_gpu() && self.client.properties().timing_method == TimingMethod::Device
    }

    /// Opens a timing window at the current position of the stream
    /// (`Client::profile_start`). On a device that [times on the
    /// device](Self::times_on_device) it submits the queued work and returns without
    /// waiting; elsewhere it waits for the stream. Close it with
    /// [`close_window`](Self::close_window) on the same thread.
    ///
    /// # Errors
    ///
    /// [`KernelError::Device`] if the runtime cannot open a window.
    pub fn open_window(&mut self) -> Result<TimingWindow, KernelError> {
        let launched = self.launched;
        self.client
            .profile_start()
            .map(|window| TimingWindow { window, launched })
            .map_err(device_error)
    }

    /// Closes `window` at the current position of the stream (`Client::profile_end`):
    /// the time from its opening to here, of the work queued in between, to
    /// [`resolve`](WindowTime::resolve) once that work has run. A window without device
    /// work (no launch between its ends) measures nothing on every backend: wgpu reports
    /// nothing for it, and the time CUDA's events or the CPU runtime's host clock give it
    /// is dropped. Counts one window, and two syncs if the window was not timed on the
    /// device (the runtime waited for the stream at both ends: the CPU runtime, or a wgpu
    /// stream that found the device's timestamp-query budget spent).
    ///
    /// # Errors
    ///
    /// [`KernelError::Device`] if the runtime cannot close the window.
    pub fn close_window(&mut self, window: TimingWindow) -> Result<WindowTime, KernelError> {
        let duration = match self.client.profile_end(window.window) {
            Ok(duration) => Some(duration),
            Err(ProfileError::NotMeasured { .. }) => None,
            Err(error) => return Err(device_error(error)),
        };
        let empty = self.launched == window.launched;
        self.counters.windows += 1;
        let on_device = match &duration {
            Some(duration) => {
                self.backend().is_gpu() && duration.timing_method() == TimingMethod::Device
            }
            None => self.times_on_device(),
        };
        if !on_device {
            self.counters.syncs += 2;
        }
        Ok(WindowTime {
            duration: duration.filter(|_| !empty),
            on_device,
        })
    }

    /// The CubeCL client, for the launch wrappers.
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// The shape of an elementwise launch over `work ≥ 1` elements on this device.
    pub(crate) fn elementwise_grid(&self, work: usize) -> Grid {
        Grid::elementwise(&self.info, work, self.cpu_units)
    }

    /// Counts one kernel launch.
    pub(crate) fn count_launch(&mut self) {
        self.counters.launches += 1;
        self.launched += 1;
    }

    /// Fails with [`KernelError::WrongDevice`] unless `device` is this device's id.
    pub(crate) fn check_owner(&self, device: u64) -> Result<(), KernelError> {
        if device == self.id {
            Ok(())
        } else {
            Err(KernelError::WrongDevice)
        }
    }

    fn count_upload(&mut self, bytes: usize) {
        self.counters.uploads += 1;
        self.counters.upload_bytes += bytes as u64;
    }

    /// The precision, length and memory checks of a new buffer of `len` elements.
    fn check_element<E: DeviceElement>(&self, len: usize) -> Result<(), KernelError> {
        if let Some(precision) = E::PRECISION {
            self.require(precision)?;
        }
        if len > MAX_ELEMENTS {
            return Err(KernelError::TooLarge {
                what: "buffer",
                len,
            });
        }
        if let Some(available) = self.available_memory() {
            let requested = storage_bytes::<E>(len) as u64;
            if requested > available {
                return Err(KernelError::OutOfMemory {
                    requested,
                    limit: available,
                });
            }
        }
        Ok(())
    }
}

/// The bytes a buffer of `len` elements occupies: at least one element, so that an
/// empty buffer still has a handle to bind.
fn storage_bytes<E: DeviceElement>(len: usize) -> usize {
    len.max(1) * size_of::<E>()
}

/// Whether `client`'s device does arithmetic in `T`.
fn does_arithmetic<T: CubeElement>(client: &Client) -> bool {
    client
        .properties()
        .type_usage(T::cube_type())
        .contains(cubecl::features::TypeUsage::Arithmetic)
}

/// A CubeCL server error as a [`KernelError::Device`].
fn device_error(e: impl fmt::Display) -> KernelError {
    KernelError::Device {
        reason: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Device` moves between threads (it is held by one at a time).
    #[test]
    fn device_is_send() {
        fn send<T: Send>() {}
        send::<Device>();
        send::<DeviceBuffer<f64>>();
        send::<IndexBuffer>();
    }

    fn info(backend: BackendKind) -> DeviceInfo {
        DeviceInfo {
            backend,
            name: "test".into(),
            compiler: "test".into(),
            cubecl_version: CUBECL_VERSION,
            f32: true,
            f64: false,
            plane_size: (1, 1),
            max_shared_memory: 0,
            max_units_per_cube: if backend.is_gpu() { 1024 } else { 16 },
            max_cube_count: (65_535, 65_535, 65_535),
            max_memory: None,
        }
    }

    /// Launch shapes cover the work exactly once: GPU blocks of one element, CPU
    /// blocks of `chunk` per unit, one unit for small launches.
    #[test]
    fn elementwise_grids_cover_the_work() {
        for backend in BackendKind::ALL {
            let info = info(backend);
            for work in [
                1,
                7,
                255,
                256,
                257,
                CPU_MIN_CHUNK - 1,
                CPU_MIN_CHUNK,
                100_003,
                1 << 27,
            ] {
                let g = Grid::elementwise(&info, work, CPU_MAX_UNITS);
                assert!(g.units >= 1 && g.cubes >= 1 && g.chunk >= 1);
                assert!(g.threads() as usize * g.chunk as usize >= work.min(g.threads() as usize));
                if backend.is_gpu() {
                    assert_eq!((g.units, g.chunk), (256, 1));
                    assert!(g.cubes <= 65_535);
                } else {
                    assert_eq!(g.cubes, 1);
                    assert_eq!(
                        g.units,
                        if work < CPU_MIN_CHUNK {
                            1
                        } else {
                            CPU_MAX_UNITS
                        }
                    );
                    assert!(g.units as usize * g.chunk as usize >= work);
                }
            }
        }
    }

    /// A cap on the CPU units bounds the units of every CPU launch and leaves GPU
    /// launches as they are; the work is still covered.
    #[test]
    fn capped_grids_use_at_most_the_cap() {
        for backend in BackendKind::ALL {
            let info = info(backend);
            for cap in [1, 2, 4, 7, 16, 64] {
                for work in [1, CPU_MIN_CHUNK, 100_003] {
                    let g = Grid::elementwise(&info, work, cap);
                    let full = Grid::elementwise(&info, work, CPU_MAX_UNITS);
                    if backend.is_gpu() {
                        assert_eq!(g, full);
                    } else {
                        assert!(g.units <= cap.max(1) && g.units <= CPU_MAX_UNITS);
                        assert!(g.units as usize * g.chunk as usize >= work);
                    }
                }
            }
        }
    }

    /// A launch error surfaces at a view (Phase 4S T11), on every backend compiled in.
    #[cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
    mod launch_errors {
        use super::*;

        /// Kernels whose launch fails, to see a launch error surface at a download: a plane
        /// sum, which the CPU runtime does not lower (device-path.md F19), and an increment
        /// launched on a GPU with a cube of more units than the device allows.
        #[cube(launch_unchecked)]
        fn plane_sum_kernel(x: &mut [f32]) {
            let i = UNIT_POS as usize;
            x[i] = plane_sum(x[i]);
        }

        #[cube(launch_unchecked)]
        fn increment_kernel(x: &mut [f32]) {
            let i = ABSOLUTE_POS;
            if i < x.len() {
                x[i] += 1.0;
            }
        }

        /// Queues a launch into `buffer` that fails on `device`'s backend.
        fn queue_failing_launch(device: &mut Device, buffer: &DeviceBuffer<f32>) {
            let (handle, len) = buffer.as_slice().binding();
            let client = device.client();
            if device.backend().is_gpu() {
                let units = device.info().max_units_per_cube.saturating_mul(2).max(2048);
                // SAFETY: each unit writes x[ABSOLUTE_POS] only below `len`, the buffer's
                // length; the cube is larger than the device allows, so the launch fails.
                unsafe {
                    increment_kernel::launch_unchecked(
                        client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(units),
                        BufferArg::from_raw_parts(handle, len),
                    );
                }
            } else {
                // SAFETY: one cube of `len` units, unit u reads and writes x[u] < len.
                unsafe {
                    plane_sum_kernel::launch_unchecked(
                        client,
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(len as u32),
                        BufferArg::from_raw_parts(handle, len),
                    );
                }
            }
        }

        /// A launch error surfaces at `download_view` as at `download` (device-path.md §12,
        /// F9), as `KernelError::Device`, with the same counters (one download, one sync, no
        /// bytes).
        fn launch_errors_surface_at_the_view(kind: BackendKind) {
            let mut device = Device::open(kind).unwrap();
            println!("{}", device.info());
            for view in [false, true] {
                let buffer = device.upload(&[1.0f32; 4]).unwrap();
                queue_failing_launch(&mut device, &buffer);
                device.reset_counters();
                let error = if view {
                    device.download_view(buffer.as_slice()).unwrap_err()
                } else {
                    let mut out = [0.0f32; 4];
                    device.download(buffer.as_slice(), &mut out).unwrap_err()
                };
                assert!(matches!(error, KernelError::Device { .. }), "{error}");
                let c = device.counters();
                assert_eq!((c.downloads, c.syncs, c.download_bytes), (1, 1, 0));
                println!(
                    "  {}: {}",
                    if view { "download_view" } else { "download" },
                    root_cause(&error.to_string())
                );
            }
        }

        /// The innermost cause of CubeCL's error chain ("Caused by:" lines), for the report.
        fn root_cause(message: &str) -> &str {
            let lines: Vec<&str> = message.lines().map(str::trim).collect();
            lines
                .iter()
                .rposition(|l| *l == "Caused by:")
                .and_then(|i| lines.get(i + 1))
                .or(lines.first())
                .copied()
                .unwrap_or_default()
        }

        #[cfg(feature = "cpu")]
        #[test]
        fn launch_errors_surface_at_the_view_cpu() {
            launch_errors_surface_at_the_view(BackendKind::Cpu);
            println!("backends run: cpu; not run: metal, cuda (in their own ignored tests)");
        }

        #[cfg(feature = "metal")]
        #[test]
        #[ignore = "Metal: run by hand, outside the sandbox"]
        fn launch_errors_surface_at_the_view_metal() {
            launch_errors_surface_at_the_view(BackendKind::Metal);
            println!("backends run: metal; not run: cpu, cuda (in their own tests)");
        }

        #[cfg(feature = "cuda")]
        #[test]
        #[ignore = "CUDA: run by hand on locust"]
        fn launch_errors_surface_at_the_view_cuda() {
            launch_errors_surface_at_the_view(BackendKind::Cuda);
            println!("backends run: cuda; not run: cpu, metal (in their own tests)");
        }
    }

    #[test]
    fn report_line() {
        let line = info(BackendKind::Metal).to_string();
        assert_eq!(
            line,
            format!("metal (test), test, CubeCL {CUBECL_VERSION}, f32")
        );
    }
}
