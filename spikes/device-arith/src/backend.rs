//! The backends the spike runs on, and their clients.

use cubecl::prelude::*;

/// A CubeCL backend compiled into the spike.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// The CubeCL CPU runtime (LLVM JIT).
    Cpu,
    /// wgpu with the MSL compiler (the `metal` feature of `nd-fmm-kernels`).
    Metal,
    /// CUDA through CubeCL's default LLVM NVPTX path (run on locust's H100, Phase 4S T3).
    Cuda,
}

impl Backend {
    /// The name on the command line and in the tables.
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// The backend of a command-line name.
    pub fn parse(name: &str) -> Option<Self> {
        [Self::Cpu, Self::Metal, Self::Cuda]
            .into_iter()
            .find(|b| b.name() == name)
    }

    /// Whether the backend is compiled in.
    pub fn compiled(self) -> bool {
        match self {
            Self::Cpu => cfg!(feature = "cpu"),
            Self::Metal => cfg!(feature = "metal"),
            Self::Cuda => cfg!(feature = "cuda"),
        }
    }

    /// Every backend compiled in.
    pub fn runnable() -> Vec<Self> {
        [Self::Metal, Self::Cuda, Self::Cpu]
            .into_iter()
            .filter(|b| b.compiled())
            .collect()
    }

    /// The compiler path behind the runtime: `client.name()` says only `cuda` on CUDA,
    /// whichever path compiled the kernels (device-path.md F25).
    pub fn compiler(self) -> &'static str {
        match self {
            Self::Cpu => "LLVM JIT",
            Self::Metal => "MSL",
            Self::Cuda if cfg!(feature = "cuda-cpp") => "NVRTC (cuda-cpp)",
            Self::Cuda => "LLVM NVPTX",
        }
    }

    /// Whether the backend is a GPU (timings are taken only there, and on the CPU runtime
    /// for the CPU-shaped P2P).
    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// A client of the backend's default device.
    ///
    /// # Panics
    ///
    /// If the backend is not compiled in or the device cannot be created (for Metal,
    /// inside the macOS sandbox; for CUDA, without a GPU or driver).
    pub fn client(self) -> Client {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu => cubecl::Device::cpu().expect("CPU runtime").client(),
            #[cfg(feature = "metal")]
            Self::Metal => cubecl::Device::metal_msl(cubecl::device::WgpuDeviceKind::DefaultDevice)
                .expect("Metal device (wgpu, MSL)")
                .client(),
            #[cfg(feature = "cuda")]
            Self::Cuda => cubecl::Device::cuda(0).expect("CUDA device").client(),
            #[allow(unreachable_patterns)]
            other => panic!("backend {} is not compiled in", other.name()),
        }
    }
}

/// Whether the client's device runs f64.
pub fn supports_f64(client: &Client) -> bool {
    client.properties().supports_type(f64::elem_type_native())
}

/// One line describing the client's device for the report.
pub fn describe(client: &Client, backend: Backend) -> String {
    let props = client.properties();
    let hw = &props.hardware;
    format!(
        "{}: device `{}`, runtime `{}`, compiler {}, f64 supported = {}, plane size {}..{}, \
         max shared memory {} B, max units per cube {}",
        backend.name(),
        props.identity.name,
        client.name(),
        backend.compiler(),
        supports_f64(client),
        hw.plane_size_min,
        hw.plane_size_max,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
    )
}

/// Reads a buffer of `T` back from the device.
pub fn read<T: crate::real::Real>(client: &Client, handle: cubecl::server::Handle) -> Vec<T> {
    let bytes = client.read_one(handle).expect("read back a buffer");
    T::from_bytes(&bytes).to_vec()
}

/// Reads a buffer of u32 back from the device.
pub fn read_u32(client: &Client, handle: cubecl::server::Handle) -> Vec<u32> {
    let bytes = client.read_one(handle).expect("read back a buffer");
    u32::from_bytes(&bytes).to_vec()
}

/// Waits for every queued launch.
pub fn sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("device sync");
}

/// Files the PTX that CUDA dumped since the last call under `label`.
///
/// With `CUBECL_CUDA_DUMP_PTX=<dir>`, cubecl-cuda writes the PTX of every kernel it loads
/// into `<dir>`, named after the last 180 characters of the kernel id. Those omit the
/// precision and the comptime arguments, so variants overwrite each other. Called after
/// the launches of one variant, this moves the new files into `<dir>/<label>/`. Without
/// the variable, or on another backend, it does nothing.
pub fn keep_ptx(label: &str) {
    let Some(dir) = std::env::var_os("CUBECL_CUDA_DUMP_PTX") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    let label: String = label
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    let target = dir.join(label);
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "ptx") {
            std::fs::create_dir_all(&target).expect("create the PTX label directory");
            std::fs::rename(&path, target.join(entry.file_name())).expect("move a PTX dump");
        }
    }
}
