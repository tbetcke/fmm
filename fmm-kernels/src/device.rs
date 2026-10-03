//! Backends, devices, the capability query, transfers and counters
//! (device-path.md §3.1, §5.1).

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use cubecl::client::Client;
use cubecl::features::Plane;
use cubecl::prelude::*;

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
    /// CUDA, feature `cuda`: type-checked, never run here.
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
    /// [`Device::write`]), empty ones included.
    pub uploads: u64,
    /// Bytes copied from the device to the host.
    pub download_bytes: u64,
    /// Device-to-host copies ([`Device::download`]), empty ones included.
    pub downloads: u64,
    /// Kernel launches. A call with nothing to do launches nothing.
    pub launches: u64,
    /// Waits for the device: one per [`Device::download`] and per [`Device::sync`].
    pub syncs: u64,
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

    /// The shape of an elementwise launch over `work ≥ 1` elements.
    ///
    /// - GPU: 256 units per cube, one element per unit and stride (coalesced), as many
    ///   cubes as cover `work`, capped at the device's limit (the units then stride).
    /// - CPU runtime: each unit is a task on CubeCL's worker pool and loops over the
    ///   cubes (device-path.md F18), so one cube, and each unit one contiguous block:
    ///   one unit below [`CPU_MIN_CHUNK`] elements, else up to `max_units_per_cube`
    ///   (at most [`CPU_MAX_UNITS`]). Only two cube sizes occur, so few kernel variants
    ///   are compiled.
    pub fn elementwise(info: &DeviceInfo, work: usize) -> Self {
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
            let max_units = info.max_units_per_cube.clamp(1, CPU_MAX_UNITS);
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
/// The most units of a CPU-runtime elementwise launch (T2, T3: 12–16 units per cube).
pub(crate) const CPU_MAX_UNITS: u32 = 16;
/// Below this many elements a CPU-runtime elementwise launch uses one unit.
pub(crate) const CPU_MIN_CHUNK: usize = 1 << 14;

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
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("info", &self.info)
            .field("id", &self.id)
            .field("counters", &self.counters)
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
    /// launches already queued.
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
        self.check_owner(slice.device())?;
        if !data.is_empty() {
            let handle = slice.byte_range_handle();
            self.client
                .write(&handle, cubecl::bytes::Bytes::from_elems(data.to_vec()));
        }
        self.count_upload(size_of_val(data));
        Ok(())
    }

    /// Copies `slice` into `out`: waits for every queued launch (one sync) and returns
    /// any launch error attached to the buffer (device-path.md F9).
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
        self.check_owner(slice.device())?;
        self.counters.downloads += 1;
        self.counters.syncs += 1;
        let handle = slice.byte_range_handle();
        if out.is_empty() {
            return cubecl::future::block_on(self.client.sync_buffers([&handle]))
                .map_err(device_error);
        }
        let bytes = self.client.read_one(handle).map_err(device_error)?;
        out.copy_from_slice(E::from_bytes(&bytes));
        self.counters.download_bytes += size_of_val(out) as u64;
        Ok(())
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

    /// The CubeCL client, for the launch wrappers.
    pub(crate) fn client(&self) -> &Client {
        &self.client
    }

    /// Counts one kernel launch.
    pub(crate) fn count_launch(&mut self) {
        self.counters.launches += 1;
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
        if let Some(limit) = self.info.max_memory {
            let requested = storage_bytes::<E>(len) as u64;
            let usage = self.client.memory_usage();
            let available = limit.saturating_sub(usage.bytes_in_use + usage.bytes_padding);
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
                let g = Grid::elementwise(&info, work);
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

    #[test]
    fn report_line() {
        let line = info(BackendKind::Metal).to_string();
        assert_eq!(
            line,
            format!("metal (test), test, CubeCL {CUBECL_VERSION}, f32")
        );
    }
}
