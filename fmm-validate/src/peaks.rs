//! Per-device peaks of the kernel harness examples (Phase 4S T7): the floating-point peak
//! in each precision, the memory bandwidth and the P2P peak model of
//! docs/design/device-path.md §13.4, keyed by the device name that
//! `nd_fmm_kernels::DeviceInfo::name` reports. A device the table does not name has an
//! unknown peak, and the examples print no fraction of it.
//!
//! | Device (`DeviceInfo::name`) | f32 GFLOP/s | f64 GFLOP/s | GB/s | Source |
//! | --- | ---: | ---: | ---: | --- |
//! | `Apple M3 Max` | 14,300 | none | 400 | derived in spikes/cubecl-gemm/SPIKE_REPORT.md (40 cores × 128 lanes × 2 × about 1.4 GHz), not measured |
//! | `NVIDIA GH200 480GB` | 67,000 | 34,000 | 4,000 | datasheet (the GH200's H100, without tensor cores: CubeCL has no FP64 MMA, and the f32 path takes no TF32), at the maximum SM clock of 1,980 MHz |
//!
//! The P2P peak model ([`Peaks::p2p_model`]) counts lane-operations per pair as
//! device-path.md §13.4 does: 10 for the potential (3 subtractions, a multiply and 2 fmas
//! for r², the inverse square root, the select of r² = 0, the product q ρ, the add into
//! φ̂) and 15 with the gradient (5 more), at one lane-operation per lane and clock. It is a
//! model: on CUDA's LLVM path the inverse square root is two IEEE sequences,
//! `sqrt.rn` and `rcp.rn` (device-path.md F33, F36), which take more than one issue slot.
//!
//! ```
//! use nd_fmm_validate::peaks::{Peaks, Precision};
//!
//! let m3 = Peaks::of("Apple M3 Max").unwrap();
//! assert_eq!(m3.gflops(Precision::F32), Some(14_300.0));
//! assert_eq!(m3.gflops(Precision::F64), None);
//! assert_eq!(m3.p2p_model(Precision::F32, false), Some(720e9));
//! assert!(Peaks::of("some other GPU").is_none());
//! ```

/// The precision of a peak. Kept apart from `nd_fmm_kernels::Precision` so that the table
/// builds without the `gpu` feature.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// f32.
    F32,
    /// f64.
    F64,
}

/// The peaks of one device (module documentation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peaks {
    /// The device name, as `DeviceInfo::name` reports it.
    pub name: &'static str,
    /// The f32 peak, GFLOP/s (an fma counts two flops).
    pub f32_gflops: f64,
    /// The f64 peak, GFLOP/s; `None` where the device does no f64 arithmetic.
    pub f64_gflops: Option<f64>,
    /// The memory bandwidth, GB/s.
    pub bandwidth_gbs: f64,
    /// Where the f32 figure comes from, as the examples print it after the number.
    pub f32_source: &'static str,
    /// Where the f64 figure comes from, likewise.
    pub f64_source: &'static str,
    /// Where the bandwidth comes from, likewise.
    pub bandwidth_source: &'static str,
    /// The P2P model ([`p2p_model`](Self::p2p_model)) in f32 and f64, pairs per second of
    /// the potential and of the potential with the gradient; `None` where a precision has
    /// no model.
    p2p: [Option<[f64; 2]>; 2],
    /// The P2P model's own wording, for reports.
    pub p2p_source: &'static str,
}

/// The lane-operations per pair of the P2P model: the potential, then with the gradient.
pub const P2P_OPERATIONS: [f64; 2] = [10.0, 15.0];

/// The P2P model of `lane_operations` per second: over 10 and 15 operations per pair.
const fn p2p_pairs(lane_operations: f64) -> [f64; 2] {
    [
        lane_operations / P2P_OPERATIONS[0],
        lane_operations / P2P_OPERATIONS[1],
    ]
}

/// The H100 of the GH200: 132 SMs, 128 f32 and 64 f64 lanes each, at 1.98 GHz.
const H100_SMS: f64 = 132.0;

/// The maximum SM clock of locust's GH200, `nvidia-smi -q -d CLOCK` (Max Clocks, SM).
const H100_CLOCK_HZ: f64 = 1.98e9;

/// Every device the table names.
const TABLE: [Peaks; 2] = [
    Peaks {
        name: "Apple M3 Max",
        f32_gflops: 14_300.0,
        f64_gflops: None,
        bandwidth_gbs: 400.0,
        f32_source: "M3 Max GPU f32, derived in the spike report, not measured",
        f64_source: "none: Metal has no f64",
        bandwidth_source: "M3 Max, Apple's figure",
        // 40 cores × 128 lanes × 1.4 GHz = 7.2 × 10¹² lane-operations per second over 10
        // and 15 operations, as device-path.md §13.4 rounds them.
        p2p: [Some([720e9, 480e9]), None],
        p2p_source: "the C4.2 peak model of device-path.md §13.4",
    },
    Peaks {
        name: "NVIDIA GH200 480GB",
        f32_gflops: 67_000.0,
        f64_gflops: Some(34_000.0),
        bandwidth_gbs: 4_000.0,
        f32_source: "H100 of the GH200, f32 without tensor cores, datasheet, at the \
                     1,980 MHz maximum SM clock (clocks not locked)",
        f64_source: "H100 of the GH200, f64 without tensor cores, datasheet, at the \
                     1,980 MHz maximum SM clock (clocks not locked)",
        bandwidth_source: "GH200 HBM3 (96 GB), datasheet",
        p2p: [
            Some(p2p_pairs(H100_SMS * 128.0 * H100_CLOCK_HZ)),
            Some(p2p_pairs(H100_SMS * 64.0 * H100_CLOCK_HZ)),
        ],
        p2p_source: "model: 132 SMs × 128 (f32) or 64 (f64) lanes × 1.98 GHz over 10 (φ) or \
                     15 (φ, ∇φ) operations per pair (Phase 4S T7)",
    },
];

impl Peaks {
    /// The peaks of the device named `name`, or `None` (an unknown peak).
    pub fn of(name: &str) -> Option<Self> {
        TABLE.iter().find(|p| p.name == name).copied()
    }

    /// The peaks of a device as `nd_fmm_kernels` reports it: by its name, on a GPU backend
    /// only. The CPU runtime reports the host's name ("Apple M3 Max" on the M3 Max), and
    /// has no peak here.
    #[cfg(feature = "gpu")]
    pub fn of_info(info: &nd_fmm_kernels::DeviceInfo) -> Option<Self> {
        if info.backend.is_gpu() {
            Self::of(&info.name)
        } else {
            None
        }
    }

    /// The peak in `precision`, GFLOP/s, if the device does arithmetic in it.
    pub fn gflops(&self, precision: Precision) -> Option<f64> {
        match precision {
            Precision::F32 => Some(self.f32_gflops),
            Precision::F64 => self.f64_gflops,
        }
    }

    /// Where the peak in `precision` comes from.
    pub fn source(&self, precision: Precision) -> &'static str {
        match precision {
            Precision::F32 => self.f32_source,
            Precision::F64 => self.f64_source,
        }
    }

    /// The P2P peak model, pairs per second, of the potential or (with `gradients`) the
    /// potential and the gradient (module documentation).
    pub fn p2p_model(&self, precision: Precision, gradients: bool) -> Option<f64> {
        let pairs = self.p2p[match precision {
            Precision::F32 => 0,
            Precision::F64 => 1,
        }]?;
        Some(pairs[usize::from(gradients)])
    }
}

/// `100 · value / peak` with two decimals, or "–" for an unknown peak.
pub fn percent(value: f64, peak: Option<f64>) -> String {
    peak.map_or("–".to_owned(), |peak| {
        format!("{:.2}", 100.0 * value / peak)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_m3_max_keeps_its_phase_4_figures() {
        let m3 = Peaks::of("Apple M3 Max").unwrap();
        assert_eq!(m3.f32_gflops, 14_300.0);
        assert_eq!(format!("{}", m3.f32_gflops), "14300");
        assert_eq!(m3.p2p_model(Precision::F32, false), Some(720e9));
        assert_eq!(m3.p2p_model(Precision::F32, true), Some(480e9));
        assert_eq!(m3.p2p_model(Precision::F64, false), None);
    }

    #[test]
    fn the_h100_model() {
        let h = Peaks::of("NVIDIA GH200 480GB").unwrap();
        let f32_phi = h.p2p_model(Precision::F32, false).unwrap();
        let f64_grad = h.p2p_model(Precision::F64, true).unwrap();
        assert!((f32_phi / 1e9 - 3_345.4).abs() < 0.1, "{f32_phi}");
        assert!((f64_grad / 1e9 - 1_115.1).abs() < 0.1, "{f64_grad}");
        assert_eq!(percent(33_500.0, h.gflops(Precision::F32)), "50.00");
        assert_eq!(percent(1.0, None), "–");
    }
}
