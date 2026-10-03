//! The device P2P (Phase 4 T6, C4.2): the Laplace near field of one level, potential
//! and optionally gradient, f32 and f64, in three layouts of one formulation
//! (docs/design/device-path.md §6.2; CONVENTIONS §3.13, "Device kernels").
//!
//! # What it computes
//!
//! One call covers one level's rows of the plan's near view (`nd_fmm_plan::lists`):
//! row r is the target leaf `first_leaf + r`, and its entries are the source leaves of
//! its near list, the leaf itself included. Each target point x of the leaf adds, into
//! the target output, starting from the value already there,
//!
//! φ̂ += Σ q / |u_t − ŷ| and ĝ −= Σ q (u_t − ŷ) / |u_t − ŷ|³,
//!
//! over the sources of the row in **row order** and, within a leaf, in **point order**,
//! the order of `LaplaceOperator`'s P2P with `nd_fmm_simd::P2pKernel` (README
//! requirement 4). Here u_t is the target's leaf-scaled position and ŷ the source's in
//! the frame of the target leaf (CONVENTIONS §3.13, "Operators in scaled coordinates"):
//! u_s itself for the leaf itself (s = t), and for s ≠ t the mapped
//! ŷ = fl(ĉ(s|t) + r̂(s|t) u_s), with the exact frames of §3.13 formed on the device from
//! the leaves' integer coordinates ([`LeafCoordinates`]), as `geometry::relative_frame`
//! of `nd-fmm-exec` forms them, bit for bit.
//!
//! **Formulation** (signed off with T3, spikes/device-arith/REPORT.md,
//! "Recommendation"; every layout, backend and precision):
//!
//! - ŷₖ = `fma(r̂, u_s,k, ĉₖ)`, an explicit fma, never inline in the subtraction (r̂ u_s
//!   is exact, so this is the host's fl(ĉ + r̂ u_s) bit for bit); u_s itself for s = t;
//! - dₖ = u_t,k − ŷₖ; r² = `fma(d₂, d₂, fma(d₁, d₁, d₀ · d₀))`;
//! - ρ = `inverse_sqrt(r²)`, then ρ = `select(r² == 0, 0, ρ)`: a pair with r² = 0
//!   contributes nothing (on leaf-scaled data exactly the coincident pairs, §3.13), and
//!   no mask multiplies a possibly infinite value; no Newton step;
//! - φ̂ = `fma(q, ρ, φ̂)`; with gradients w = (q ρ)(ρ ρ) and ĝₖ = `fma(−w, dₖ, ĝₖ)`.
//!
//! Each target's accumulators are loaded from the output, updated term by term in the
//! order above and stored once per target block (device-path.md §6.1): splitting a row
//! into calls at any leaf boundary gives the same bits, a target's bits do not depend on
//! the position of its leaf in the level or on the other targets, and repeated launches
//! give the same bits. No atomics: every output value has one owning unit per launch.
//!
//! # Layouts ([`P2pLayout`])
//!
//! | Layout | Default on | Parallel unit | Sources |
//! | --- | --- | --- | --- |
//! | [`Cube`](P2pLayout::Cube) | Metal, CUDA | one cube of U units per target leaf; unit u owns targets u, u + U, … | staged in tiles of U mapped sources and charges in shared memory, `sync_cube` around each tile |
//! | [`Plane`](P2pLayout::Plane) | none (a candidate, T12) | one plane per target leaf, several leaves per cube; lane l owns targets l, l + P, … | each plane stages its own tile of P sources, `sync_plane` around it |
//! | [`Cpu`](P2pLayout::Cpu) | the CPU runtime | one unit per core, each a contiguous range of target leaves; targets in `Vector<T, N>` lanes, K vectors per block | broadcast one by one, no shared memory, no barrier |
//!
//! All three run the formulation above in the order above, so every test runs on every
//! layout. Tile sizes and vector widths are comptime, so each (layout, precision,
//! gradients) is one compiled kernel. Leaves larger than one tile, the last partial tile
//! or target block, and leaves without points run through the same code with masked
//! units; the trip counts of every loop with a barrier are the same for every unit of
//! the cube (or plane), so no barrier diverges.
//!
//! The CPU layout ([`cpu_vectors`]): vectors of N targets, by default the host's vector
//! width ([`CPU_VECTOR_BITS`]: 128-bit NEON on aarch64, 4 × f32 or 2 × f64; 256-bit AVX2
//! on x86_64, 8 × f32 or 4 × f64), and K vector blocks per unit, so that K N = 8 targets
//! share each broadcast source (K = 2 in f32 and 4 in f64 on NEON, as T3's prototype and
//! the NEON kernel of `nd-fmm-simd`). Its units per cube are capped by
//! [`Device::limit_units`] (`threads(n)` of `nd-fmm-exec`, device-path.md §11). On the
//! CPU runtime `inverse_sqrt` is fl(1 / fl(√r²)), correctly rounded `sqrt` and division.
//!
//! The GPU layouts also run on the CPU runtime (correctness only: its planes are one
//! unit, and a kernel with shared memory gets a worker per unit, device-path.md F18).
//!
//! # Launch
//!
//! [`p2p`] is the safe launch wrapper of one level's call; it checks every length,
//! owner and index bound on the host first, and allocates nothing. [`near_frames`]
//! writes the frames the kernels form, for tests against `geometry::relative_frame`
//! (device-path.md §6.1).

use std::fmt;

use cubecl::prelude::*;

use crate::buffer::{DeviceFloat, DeviceSlice, DeviceSliceMut};
use crate::device::{BackendKind, Device, DeviceInfo, Precision};
use crate::error::KernelError;
use crate::view::{IndexView, LeafCoordinates, PointOffsets};

/// The units of the [`Cube`](P2pLayout::Cube) layout by default on the GPU backends (and
/// so its source tile): 64 (device-path.md §6.2; candidates 32, 64, 128).
pub const GPU_CUBE_UNITS: u32 = 64;

/// The vector width of the [`Cpu`](P2pLayout::Cpu) layout by default, in bits: the host's,
/// 256 (AVX2) on x86_64 and 128 (NEON) elsewhere. Only the M3 Max (NEON) timed it.
pub const CPU_VECTOR_BITS: u32 = if cfg!(target_arch = "x86_64") {
    256
} else {
    128
};

/// The layout of a P2P launch (module documentation, "Layouts").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum P2pLayout {
    /// One cube of `units` units per target leaf; the source tile holds `units`
    /// sources.
    Cube {
        /// Units per cube, at least 1.
        units: u32,
    },
    /// One plane per target leaf, `planes` planes per cube; the source tile of each
    /// plane holds one source per lane. Needs a device with one plane size.
    Plane {
        /// Planes per cube, at least 1.
        planes: u32,
    },
    /// The CPU layout: targets in vectors of `vector_bits` bits ([`cpu_vectors`]), one
    /// unit per core.
    Cpu {
        /// The vector width in bits: 64, 128, 256 or 512; by default
        /// [`CPU_VECTOR_BITS`].
        vector_bits: u32,
    },
}

impl P2pLayout {
    /// The default layout of a device: [`Cpu`](Self::Cpu) with the host's vector width
    /// ([`CPU_VECTOR_BITS`]) on the CPU runtime, otherwise [`Cube`](Self::Cube) with
    /// [`GPU_CUBE_UNITS`] units (fewer if the device allows fewer per cube). Chosen at
    /// build from the backend (device-path.md §6.2, §6.5).
    pub fn default_for(info: &DeviceInfo) -> Self {
        match info.backend {
            BackendKind::Cpu => Self::Cpu {
                vector_bits: CPU_VECTOR_BITS,
            },
            BackendKind::Metal | BackendKind::Cuda => Self::Cube {
                units: GPU_CUBE_UNITS.min(info.max_units_per_cube.max(1)),
            },
        }
    }

    /// The target block and source tile of the layout on `info` in `precision`: U for
    /// [`Cube`](Self::Cube), the plane size for [`Plane`](Self::Plane), K N for
    /// [`Cpu`](Self::Cpu). Leaves of n points run ⌈n / tile⌉ blocks.
    pub fn tile(self, info: &DeviceInfo, precision: Precision) -> usize {
        match self {
            Self::Cube { units } => units as usize,
            Self::Plane { .. } => info.plane_size.0 as usize,
            Self::Cpu { vector_bits } => {
                let (lanes, blocks) = cpu_vectors(vector_bits, precision);
                lanes * blocks
            }
        }
    }

    /// Checks that the device can run the layout in `precision`.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedLayout`] for zero units or planes, more units per cube
    /// than the device allows, a tile larger than its shared memory, a plane layout on a
    /// device whose plane size varies, or a vector width other than 64, 128, 256 or 512
    /// bits.
    pub fn check(self, info: &DeviceInfo, precision: Precision) -> Result<(), KernelError> {
        let refuse = |reason: String| {
            Err(KernelError::UnsupportedLayout {
                layout: self.to_string(),
                reason,
            })
        };
        let value = match precision {
            Precision::F32 => 4,
            Precision::F64 => 8,
        };
        let (units, tile) = match self {
            Self::Cube { units } => (units, units),
            Self::Plane { planes } => {
                let (min, max) = info.plane_size;
                if min != max || min == 0 {
                    return refuse(format!("the plane size varies ({min}–{max})"));
                }
                (planes.saturating_mul(min), planes.saturating_mul(min))
            }
            Self::Cpu { vector_bits } => {
                return if [64, 128, 256, 512].contains(&vector_bits) {
                    Ok(())
                } else {
                    refuse(format!("vectors of {vector_bits} bits"))
                };
            }
        };
        if units == 0 {
            return refuse("no units".into());
        }
        if units > info.max_units_per_cube {
            return refuse(format!(
                "{units} units per cube, the device allows {}",
                info.max_units_per_cube
            ));
        }
        let shared = 4 * tile as usize * value;
        if shared > info.max_shared_memory {
            return refuse(format!(
                "{shared} bytes of shared memory, the device has {}",
                info.max_shared_memory
            ));
        }
        Ok(())
    }
}

impl fmt::Display for P2pLayout {
    /// `cube (64 units)`, `plane (2 planes per cube)` or `cpu (vector lanes)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube { units } => write!(f, "cube ({units} units)"),
            Self::Plane { planes } => write!(f, "plane ({planes} planes per cube)"),
            Self::Cpu { vector_bits } => write!(f, "cpu ({vector_bits}-bit vectors)"),
        }
    }
}

/// The vector width N and the blocks K of the [`Cpu`](P2pLayout::Cpu) layout with
/// vectors of `vector_bits` bits in `precision`: N = `vector_bits` / (32 or 64), at most
/// 8, and K with K N = 8 targets per block, as T3's prototype and the NEON kernel of
/// `nd-fmm-simd` (128 bits: N = 4, K = 2 in f32; N = 2, K = 4 in f64).
pub fn cpu_vectors(vector_bits: u32, precision: Precision) -> (usize, usize) {
    let element = match precision {
        Precision::F32 => 32,
        Precision::F64 => 64,
    };
    let lanes = (vector_bits as usize / element).clamp(1, 8);
    (lanes, 8 / lanes)
}

/// The inputs of one level's P2P call ([`p2p`]): the near view of the level, the leaves'
/// coordinates and the leaf stores (CONVENTIONS §3.13).
#[derive(Clone, Copy, Debug)]
pub struct P2pInputs<'a, T: DeviceFloat> {
    /// Row r: the source leaves of target leaf `first_leaf + r`, in the order they are
    /// added (the plan's near view of the level, ascending, the leaf itself included).
    pub near: &'a IndexView,
    /// The leaf of row 0; rows are consecutive leaves.
    pub first_leaf: usize,
    /// The level and index of every leaf: the frames between leaves.
    pub leaves: &'a LeafCoordinates,
    /// The point offsets of the source store, by leaf.
    pub source_offsets: &'a PointOffsets,
    /// The source store: per leaf of n points, n leaf-scaled triples, then n charges.
    pub sources: DeviceSlice<'a, T>,
    /// The point offsets of the target input and output, by leaf.
    pub target_offsets: &'a PointOffsets,
    /// The target input: per leaf of n points, n leaf-scaled triples.
    pub target_input: DeviceSlice<'a, T>,
}

/// 2⁻¹⁶, exact in f32 and f64.
const TWO_TO_MINUS_16: f32 = 1.0 / 65_536.0;

/// 2ᵉ for −16 ≤ e ≤ 16, exactly: an integer power of two converted, times 2⁻¹⁶ below 1.
#[cube]
fn pow2<F: Float>(e: i32) -> F {
    if e >= 0i32 {
        F::cast_from(1u32 << u32::cast_from(e))
    } else {
        F::cast_from(1u32 << u32::cast_from(e + 16i32)) * F::new(TWO_TO_MINUS_16)
    }
}

/// The larger of the levels of leaves s and t: the reference level L of §3.13.
#[cube]
fn reference_level(leaves: &[u32], s: usize, t: usize) -> u32 {
    let (ls, lt) = (leaves[4 * s], leaves[4 * t]);
    let mut big = ls;
    if lt > ls {
        big = lt;
    }
    big
}

/// Component k of the frame centre ĉ(s|t) = (C_L(s) − C_L(t)) 2^(l_t − L) (CONVENTIONS
/// §3.13, "Relative frames"): the integer difference, exact in `i32` (|N| ≤ 131,070),
/// converted exactly and scaled by an exact power of two. No step rounds.
#[cube]
fn frame_centre<F: Float>(leaves: &[u32], s: usize, t: usize, k: usize) -> F {
    let big = reference_level(leaves, s, t);
    let (ls, lt) = (leaves[4 * s], leaves[4 * t]);
    let cs = (2u32 * leaves[4 * s + 1 + k] + 1u32) << (big - ls);
    let ct = (2u32 * leaves[4 * t + 1 + k] + 1u32) << (big - lt);
    let difference = i32::cast_from(cs) - i32::cast_from(ct);
    F::cast_from(difference) * pow2::<F>(i32::cast_from(lt) - i32::cast_from(big))
}

/// The frame ratio r̂(s|t) = 2^(l_t − l_s), exact.
#[cube]
fn frame_ratio<F: Float>(leaves: &[u32], s: usize, t: usize) -> F {
    pow2::<F>(i32::cast_from(leaves[4 * t]) - i32::cast_from(leaves[4 * s]))
}

/// The GPU layouts: one group of `group` units per target leaf, `groups` groups per cube,
/// each group staging tiles of `group` mapped sources in its region of shared memory.
/// With `plane_sync` a group is a plane (`sync_plane`), otherwise the cube (`sync_cube`,
/// `groups` = 1).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn p2p_tiled_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    leaves: &[u32],
    source_offsets: &[u32],
    target_offsets: &[u32],
    sources: &[F],
    target_input: &[F],
    target_output: &mut [F],
    source_base: u32,
    input_base: u32,
    output_base: u32,
    first_leaf: u32,
    nrows: u32,
    #[comptime] group: usize,
    #[comptime] groups: usize,
    #[comptime] plane_sync: bool,
    #[comptime] gradients: bool,
) {
    let mut tile_x = Shared::<[F]>::new_slice(group * groups);
    let mut tile_y = Shared::<[F]>::new_slice(group * groups);
    let mut tile_z = Shared::<[F]>::new_slice(group * groups);
    let mut tile_q = Shared::<[F]>::new_slice(group * groups);
    let mut member = 0usize;
    let mut lane = UNIT_POS as usize;
    if plane_sync {
        member = PLANE_POS as usize;
        lane = UNIT_POS_PLANE as usize;
    }
    let cube = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    let row = cube * groups + member;
    let tile0 = member * group;
    // The row condition is the same for every unit of a group, so every barrier below is
    // reached by all of them or by none.
    if row < nrows as usize {
        let t_leaf = first_leaf as usize + row;
        let t_first = target_offsets[t_leaf] as usize;
        let n_t = target_offsets[t_leaf + 1] as usize - t_first;
        let mut per_point = 1usize;
        if gradients {
            per_point = 4usize;
        }
        let phi0 = output_base as usize + per_point * t_first;
        let grad0 = phi0 + n_t;
        let in0 = input_base as usize + 3 * t_first;
        let e_first = row_offsets[row] as usize;
        let e_last = row_offsets[row + 1] as usize;
        let mut block = 0usize;
        while block < n_t {
            let t = block + lane;
            let active = t < n_t;
            let mut x0 = F::new(0.0f32);
            let mut x1 = F::new(0.0f32);
            let mut x2 = F::new(0.0f32);
            let mut phi = F::new(0.0f32);
            let mut g0 = F::new(0.0f32);
            let mut g1 = F::new(0.0f32);
            let mut g2 = F::new(0.0f32);
            if active {
                x0 = target_input[in0 + 3 * t];
                x1 = target_input[in0 + 3 * t + 1];
                x2 = target_input[in0 + 3 * t + 2];
                phi = target_output[phi0 + t];
                if gradients {
                    g0 = target_output[grad0 + 3 * t];
                    g1 = target_output[grad0 + 3 * t + 1];
                    g2 = target_output[grad0 + 3 * t + 2];
                }
            }
            for e in e_first..e_last {
                let j = entries[e] as usize;
                let s_first = source_offsets[j] as usize;
                let n_s = source_offsets[j + 1] as usize - s_first;
                let base = source_base as usize + 4 * s_first;
                let mapped = j != t_leaf;
                let c0 = frame_centre::<F>(leaves, j, t_leaf, 0usize);
                let c1 = frame_centre::<F>(leaves, j, t_leaf, 1usize);
                let c2 = frame_centre::<F>(leaves, j, t_leaf, 2usize);
                let ratio = frame_ratio::<F>(leaves, j, t_leaf);
                let mut k0 = 0usize;
                while k0 < n_s {
                    let k = k0 + lane;
                    if k < n_s {
                        let mut y0 = sources[base + 3 * k];
                        let mut y1 = sources[base + 3 * k + 1];
                        let mut y2 = sources[base + 3 * k + 2];
                        if mapped {
                            y0 = fma(ratio, y0, c0);
                            y1 = fma(ratio, y1, c1);
                            y2 = fma(ratio, y2, c2);
                        }
                        tile_x[tile0 + lane] = y0;
                        tile_y[tile0 + lane] = y1;
                        tile_z[tile0 + lane] = y2;
                        tile_q[tile0 + lane] = sources[base + 3 * n_s + k];
                    }
                    if plane_sync {
                        sync_plane();
                    } else {
                        sync_cube();
                    }
                    let mut count = n_s - k0;
                    if count > group {
                        count = group;
                    }
                    if active {
                        for i in 0..count {
                            let d0 = x0 - tile_x[tile0 + i];
                            let d1 = x1 - tile_y[tile0 + i];
                            let d2 = x2 - tile_z[tile0 + i];
                            let q = tile_q[tile0 + i];
                            let r2 = fma(d2, d2, fma(d1, d1, d0 * d0));
                            let rho =
                                select(r2 == F::new(0.0f32), F::new(0.0f32), r2.inverse_sqrt());
                            phi = fma(q, rho, phi);
                            if gradients {
                                let w = (q * rho) * (rho * rho);
                                g0 = fma(-w, d0, g0);
                                g1 = fma(-w, d1, g1);
                                g2 = fma(-w, d2, g2);
                            }
                        }
                    }
                    if plane_sync {
                        sync_plane();
                    } else {
                        sync_cube();
                    }
                    k0 += group;
                }
            }
            if active {
                target_output[phi0 + t] = phi;
                if gradients {
                    target_output[grad0 + 3 * t] = g0;
                    target_output[grad0 + 3 * t + 1] = g1;
                    target_output[grad0 + 3 * t + 2] = g2;
                }
            }
            block += group;
        }
    }
}

/// The CPU layout: one cube, each unit a contiguous range of the rows; per row blocks of
/// `blocks` vectors of N targets, every source of the row broadcast to them in turn.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn p2p_vector_kernel<F: Float, N: Size>(
    row_offsets: &[u32],
    entries: &[u32],
    leaves: &[u32],
    source_offsets: &[u32],
    target_offsets: &[u32],
    sources: &[F],
    target_input: &[F],
    target_output: &mut [F],
    source_base: u32,
    input_base: u32,
    output_base: u32,
    first_leaf: u32,
    nrows: u32,
    #[comptime] blocks: usize,
    #[comptime] gradients: bool,
) {
    let width = N::value();
    let units = CUBE_DIM as usize;
    let unit = UNIT_POS as usize;
    let rows = nrows as usize;
    let share = rows / units;
    let extra = rows % units;
    let mut first = unit * share + extra;
    let mut count = share;
    if unit < extra {
        first = unit * share + unit;
        count = share + 1;
    }
    let zero = Vector::<F, N>::new(F::new(0.0f32));
    let mut tx = Array::<Vector<F, N>>::new(blocks);
    let mut ty = Array::<Vector<F, N>>::new(blocks);
    let mut tz = Array::<Vector<F, N>>::new(blocks);
    let mut ap = Array::<Vector<F, N>>::new(blocks);
    let mut ax = Array::<Vector<F, N>>::new(blocks);
    let mut ay = Array::<Vector<F, N>>::new(blocks);
    let mut az = Array::<Vector<F, N>>::new(blocks);
    for row in first..first + count {
        let t_leaf = first_leaf as usize + row;
        let t_first = target_offsets[t_leaf] as usize;
        let n_t = target_offsets[t_leaf + 1] as usize - t_first;
        let mut per_point = 1usize;
        if gradients {
            per_point = 4usize;
        }
        let phi0 = output_base as usize + per_point * t_first;
        let grad0 = phi0 + n_t;
        let in0 = input_base as usize + 3 * t_first;
        let e_first = row_offsets[row] as usize;
        let e_last = row_offsets[row + 1] as usize;
        let mut b0 = 0usize;
        while b0 < n_t {
            #[unroll]
            for b in 0..blocks {
                let mut vx = zero;
                let mut vy = zero;
                let mut vz = zero;
                let mut vp = zero;
                let mut vgx = zero;
                let mut vgy = zero;
                let mut vgz = zero;
                #[unroll]
                for l in 0..width {
                    let t = b0 + b * width + l;
                    if t < n_t {
                        vx.insert(l, target_input[in0 + 3 * t]);
                        vy.insert(l, target_input[in0 + 3 * t + 1]);
                        vz.insert(l, target_input[in0 + 3 * t + 2]);
                        vp.insert(l, target_output[phi0 + t]);
                        if gradients {
                            vgx.insert(l, target_output[grad0 + 3 * t]);
                            vgy.insert(l, target_output[grad0 + 3 * t + 1]);
                            vgz.insert(l, target_output[grad0 + 3 * t + 2]);
                        }
                    }
                }
                tx[b] = vx;
                ty[b] = vy;
                tz[b] = vz;
                ap[b] = vp;
                ax[b] = vgx;
                ay[b] = vgy;
                az[b] = vgz;
            }
            for e in e_first..e_last {
                let j = entries[e] as usize;
                let s_first = source_offsets[j] as usize;
                let n_s = source_offsets[j + 1] as usize - s_first;
                let base = source_base as usize + 4 * s_first;
                let mapped = j != t_leaf;
                let c0 = frame_centre::<F>(leaves, j, t_leaf, 0usize);
                let c1 = frame_centre::<F>(leaves, j, t_leaf, 1usize);
                let c2 = frame_centre::<F>(leaves, j, t_leaf, 2usize);
                let ratio = frame_ratio::<F>(leaves, j, t_leaf);
                for k in 0..n_s {
                    let mut y0 = sources[base + 3 * k];
                    let mut y1 = sources[base + 3 * k + 1];
                    let mut y2 = sources[base + 3 * k + 2];
                    if mapped {
                        y0 = fma(ratio, y0, c0);
                        y1 = fma(ratio, y1, c1);
                        y2 = fma(ratio, y2, c2);
                    }
                    let sx = Vector::<F, N>::new(y0);
                    let sy = Vector::<F, N>::new(y1);
                    let sz = Vector::<F, N>::new(y2);
                    let q = Vector::<F, N>::new(sources[base + 3 * n_s + k]);
                    #[unroll]
                    for b in 0..blocks {
                        let dx = tx[b] - sx;
                        let dy = ty[b] - sy;
                        let dz = tz[b] - sz;
                        let r2 = fma(dz, dz, fma(dy, dy, dx * dx));
                        let rho = select_many(r2.equal(&zero), zero, r2.inverse_sqrt());
                        ap[b] = fma(q, rho, ap[b]);
                        if gradients {
                            let w = (q * rho) * (rho * rho);
                            ax[b] = fma(-w, dx, ax[b]);
                            ay[b] = fma(-w, dy, ay[b]);
                            az[b] = fma(-w, dz, az[b]);
                        }
                    }
                }
            }
            #[unroll]
            for b in 0..blocks {
                let vp = ap[b];
                let vgx = ax[b];
                let vgy = ay[b];
                let vgz = az[b];
                #[unroll]
                for l in 0..width {
                    let t = b0 + b * width + l;
                    if t < n_t {
                        target_output[phi0 + t] = vp.extract(l);
                        if gradients {
                            target_output[grad0 + 3 * t] = vgx.extract(l);
                            target_output[grad0 + 3 * t + 1] = vgy.extract(l);
                            target_output[grad0 + 3 * t + 2] = vgz.extract(l);
                        }
                    }
                }
            }
            b0 += blocks * width;
        }
    }
}

/// The frames of every entry of the rows: (ĉ₀, ĉ₁, ĉ₂, r̂) per entry, (0, 0, 0, 1) for the
/// leaf itself; rows in blocks of `chunk` per unit and stride.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn frames_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    leaves: &[u32],
    frames: &mut [F],
    frames_base: u32,
    first_leaf: u32,
    nrows: u32,
    chunk: u32,
    threads: u32,
) {
    let (rows, chunk) = (nrows as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < rows {
        let mut end = start + chunk;
        if end > rows {
            end = rows;
        }
        for row in start..end {
            let t = first_leaf as usize + row;
            for e in row_offsets[row] as usize..row_offsets[row + 1] as usize {
                let j = entries[e] as usize;
                let at = frames_base as usize + 4 * e;
                if j == t {
                    frames[at] = F::new(0.0f32);
                    frames[at + 1] = F::new(0.0f32);
                    frames[at + 2] = F::new(0.0f32);
                    frames[at + 3] = F::new(1.0f32);
                } else {
                    frames[at] = frame_centre::<F>(leaves, j, t, 0usize);
                    frames[at + 1] = frame_centre::<F>(leaves, j, t, 1usize);
                    frames[at + 2] = frame_centre::<F>(leaves, j, t, 2usize);
                    frames[at + 3] = frame_ratio::<F>(leaves, j, t);
                }
            }
        }
        start += stride;
    }
}

/// The checks every launch shares: owners, and the shape of the near view against the
/// leaves and the offsets. Returns the number of rows.
fn check_rows(
    device: &Device,
    near: &IndexView,
    first_leaf: usize,
    leaves: &LeafCoordinates,
    target_leaves: usize,
) -> Result<usize, KernelError> {
    for owner in [
        near.row_offsets().device(),
        near.entries().device(),
        leaves.buffer().as_slice().device(),
    ] {
        device.check_owner(owner)?;
    }
    let nrows = near.nrows();
    assert!(
        first_leaf + nrows <= target_leaves && first_leaf + nrows <= leaves.len(),
        "p2p: rows {first_leaf}..{} for {target_leaves} target leaves and {} leaf \
         coordinates",
        first_leaf + nrows,
        leaves.len()
    );
    assert!(
        near.is_empty() || near.entries().bound() <= leaves.len() as u64,
        "p2p: near entries up to {} for {} leaf coordinates",
        near.entries().bound().saturating_sub(1),
        leaves.len()
    );
    assert!(
        u32::try_from(first_leaf + nrows).is_ok(),
        "p2p: {} rows exceed the 32-bit index range",
        first_leaf + nrows
    );
    Ok(nrows)
}

/// The cube grid of `cubes ≥ 1` cubes: one dimension up to the device's limit, two above
/// it (device-path.md §6.2, "Grid").
fn cube_grid(info: &DeviceInfo, cubes: usize) -> (u32, u32) {
    let max_x = info.max_cube_count.0.max(1) as usize;
    if cubes <= max_x {
        (cubes as u32, 1)
    } else {
        (max_x as u32, cubes.div_ceil(max_x) as u32)
    }
}

/// One level's P2P: adds the near field of the rows of `inputs.near` into
/// `target_output` (module documentation), in `layout`, with gradients if `gradients`.
/// `target_output` is the target output store: per leaf of n points, n values, or with
/// gradients the n potentials and then n triples (CONVENTIONS §3.13).
///
/// A call without rows or entries launches nothing. One launch otherwise, which
/// allocates nothing; on the CPU layout its units per cube are at most
/// [`Device::units_cap`].
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the device cannot run `layout` in `T`
/// ([`P2pLayout::check`]).
///
/// # Panics
///
/// If the rows exceed the target leaves or the leaf coordinates, an entry addresses no
/// leaf of the source offsets or the coordinates, or a store does not hold its values:
/// 4 per source point, 3 per target point, and 1 (4 with gradients) per target point in
/// the output.
pub fn p2p<T: DeviceFloat>(
    device: &mut Device,
    layout: P2pLayout,
    gradients: bool,
    inputs: &P2pInputs<'_, T>,
    target_output: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    let P2pInputs {
        near,
        first_leaf,
        leaves,
        source_offsets,
        sources,
        target_offsets,
        target_input,
    } = *inputs;
    for owner in [
        source_offsets.buffer().as_slice().device(),
        target_offsets.buffer().as_slice().device(),
        sources.device(),
        target_input.device(),
        target_output.device(),
    ] {
        device.check_owner(owner)?;
    }
    let nrows = check_rows(device, near, first_leaf, leaves, target_offsets.nleaves())?;
    layout.check(device.info(), T::FLOAT)?;
    assert!(
        near.is_empty() || near.entries().bound() <= source_offsets.nleaves() as u64,
        "p2p: near entries up to {} for {} source leaves",
        near.entries().bound().saturating_sub(1),
        source_offsets.nleaves()
    );
    let per_point = if gradients { 4 } else { 1 };
    assert_eq!(
        sources.len(),
        4 * source_offsets.total(),
        "p2p: the source store holds 4 values per point"
    );
    assert_eq!(
        target_input.len(),
        3 * target_offsets.total(),
        "p2p: the target input holds 3 values per point"
    );
    assert_eq!(
        target_output.len(),
        per_point * target_offsets.total(),
        "p2p: the target output holds {per_point} values per point"
    );
    if nrows == 0 || near.is_empty() {
        return Ok(());
    }
    let info = device.info().clone();
    let (ro, ro_len) = near.row_offsets().binding();
    let (en, en_len) = near.entries().binding();
    let (lv, lv_len) = leaves.buffer().as_slice().binding();
    let (so, so_len) = source_offsets.buffer().as_slice().binding();
    let (to, to_len) = target_offsets.buffer().as_slice().binding();
    let (sh, sh_len) = sources.binding();
    let (ih, ih_len) = target_input.binding();
    let (oh, oh_len) = target_output.binding();
    let bases = (
        sources.offset() as u32,
        target_input.offset() as u32,
        target_output.offset() as u32,
    );
    let client = device.client().clone();
    match layout {
        P2pLayout::Cube { .. } | P2pLayout::Plane { .. } => {
            let (group, groups, plane_sync) = match layout {
                P2pLayout::Cube { units } => (units as usize, 1usize, false),
                P2pLayout::Plane { planes } => (info.plane_size.0 as usize, planes as usize, true),
                P2pLayout::Cpu { .. } => unreachable!("the CPU layout is launched below"),
            };
            let (cubes_x, cubes_y) = cube_grid(&info, nrows.div_ceil(groups));
            // SAFETY: every handle is a whole buffer with its element count, as
            // `from_raw_parts` requires. The kernel reads row offsets r and r + 1 of rows
            // r < nrows (the view has nrows + 1, validated CSR) and their entries, each
            // below the bounds checked above: the leaf coordinates at 4 j + c, the source
            // offsets at j and j + 1, and the target offsets at first_leaf + r + 1 ≤
            // nleaves. Offsets are validated CSR (`PointOffsets`), so each leaf's chunk
            // lies within the store, whose length was asserted against the totals:
            // sources at base + 4 P_j + i < base + 4 total, target input and output
            // likewise. Shared memory holds group · groups values per array, and a unit
            // touches its group's region only. Rows at or past nrows (the last cube of a
            // plane layout or a 2-D grid) do nothing.
            unsafe {
                p2p_tiled_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_1d((group * groups) as u32),
                    BufferArg::from_raw_parts(ro, ro_len),
                    BufferArg::from_raw_parts(en, en_len),
                    BufferArg::from_raw_parts(lv, lv_len),
                    BufferArg::from_raw_parts(so, so_len),
                    BufferArg::from_raw_parts(to, to_len),
                    BufferArg::from_raw_parts(sh, sh_len),
                    BufferArg::from_raw_parts(ih, ih_len),
                    BufferArg::from_raw_parts(oh, oh_len),
                    bases.0,
                    bases.1,
                    bases.2,
                    first_leaf as u32,
                    nrows as u32,
                    group,
                    groups,
                    plane_sync,
                    gradients,
                );
            }
        }
        P2pLayout::Cpu { vector_bits } => {
            let (lanes, blocks) = cpu_vectors(vector_bits, T::FLOAT);
            let units = (device.units_cap() as usize)
                .min(info.max_units_per_cube.max(1) as usize)
                .min(nrows)
                .max(1);
            // SAFETY: as for the tiled kernel: every handle is a whole buffer with its
            // element count; rows, entries, leaves and offsets are within the bounds
            // checked above, and every chunk within its store. Each unit covers its own
            // contiguous range of rows, so no two units write one output value; lanes
            // past a leaf's points are neither loaded nor stored.
            unsafe {
                p2p_vector_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(units as u32),
                    lanes,
                    BufferArg::from_raw_parts(ro, ro_len),
                    BufferArg::from_raw_parts(en, en_len),
                    BufferArg::from_raw_parts(lv, lv_len),
                    BufferArg::from_raw_parts(so, so_len),
                    BufferArg::from_raw_parts(to, to_len),
                    BufferArg::from_raw_parts(sh, sh_len),
                    BufferArg::from_raw_parts(ih, ih_len),
                    BufferArg::from_raw_parts(oh, oh_len),
                    bases.0,
                    bases.1,
                    bases.2,
                    first_leaf as u32,
                    nrows as u32,
                    blocks,
                    gradients,
                );
            }
        }
    }
    device.count_launch();
    Ok(())
}

/// Writes the frame of every entry of `near` into `frames`, four values per entry in
/// entry order: (ĉ₀, ĉ₁, ĉ₂, r̂) of the source leaf seen from the row's leaf
/// `first_leaf + r`, as the P2P kernels form them, and (0, 0, 0, 1) where the source is
/// the row's leaf. For tests against `geometry::relative_frame` of `nd-fmm-exec`
/// (device-path.md §6.1). One elementwise launch over the rows; none without an entry.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// If the rows exceed the leaf coordinates, an entry addresses no leaf, or `frames` does
/// not hold four values per entry.
pub fn near_frames<T: DeviceFloat>(
    device: &mut Device,
    near: &IndexView,
    first_leaf: usize,
    leaves: &LeafCoordinates,
    frames: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    device.check_owner(frames.device())?;
    let nrows = check_rows(device, near, first_leaf, leaves, leaves.len())?;
    assert_eq!(
        frames.len(),
        4 * near.len(),
        "near_frames: four values per entry"
    );
    if nrows == 0 || near.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(nrows);
    let (ro, ro_len) = near.row_offsets().binding();
    let (en, en_len) = near.entries().binding();
    let (lv, lv_len) = leaves.buffer().as_slice().binding();
    let (fh, fh_len) = frames.binding();
    // SAFETY: every handle is a whole buffer with its element count. The kernel reads the
    // row offsets and entries of rows below nrows, the leaf coordinates of entries and of
    // rows' leaves (both within the bounds checked above), and writes four values per
    // entry at frames.offset() + 4 e, within `frames` (its length was asserted).
    unsafe {
        frames_kernel::launch_unchecked::<T>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(ro, ro_len),
            BufferArg::from_raw_parts(en, en_len),
            BufferArg::from_raw_parts(lv, lv_len),
            BufferArg::from_raw_parts(fh, fh_len),
            frames.offset() as u32,
            first_leaf as u32,
            nrows as u32,
            grid.chunk,
            grid.threads(),
        );
    }
    device.count_launch();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CUBECL_VERSION;

    fn info(backend: BackendKind, plane: (u32, u32)) -> DeviceInfo {
        DeviceInfo {
            backend,
            name: "test".into(),
            compiler: "test".into(),
            cubecl_version: CUBECL_VERSION,
            f32: true,
            f64: true,
            plane_size: plane,
            max_shared_memory: 32 * 1024,
            max_units_per_cube: if backend.is_gpu() { 1024 } else { 16 },
            max_cube_count: (65_535, 65_535, 65_535),
            max_memory: None,
        }
    }

    #[test]
    fn defaults_follow_the_backend() {
        assert_eq!(
            P2pLayout::default_for(&info(BackendKind::Cpu, (1, 1))),
            P2pLayout::Cpu {
                vector_bits: CPU_VECTOR_BITS
            }
        );
        for backend in [BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(
                P2pLayout::default_for(&info(backend, (32, 32))),
                P2pLayout::Cube { units: 64 }
            );
        }
    }

    #[test]
    fn layouts_are_checked_against_the_device() {
        let metal = info(BackendKind::Metal, (32, 32));
        for layout in [
            P2pLayout::Cube { units: 64 },
            P2pLayout::Plane { planes: 4 },
            P2pLayout::Cpu { vector_bits: 128 },
        ] {
            assert!(layout.check(&metal, Precision::F32).is_ok(), "{layout}");
        }
        assert!(
            P2pLayout::Cube { units: 0 }
                .check(&metal, Precision::F32)
                .is_err()
        );
        assert!(
            P2pLayout::Cube { units: 2048 }
                .check(&metal, Precision::F32)
                .is_err()
        );
        // 1,024 units of 4 f64 values: 32 KB, the limit; 2 more planes do not fit.
        assert!(
            P2pLayout::Plane { planes: 32 }
                .check(&metal, Precision::F64)
                .is_ok()
        );
        assert!(
            P2pLayout::Plane { planes: 33 }
                .check(&metal, Precision::F64)
                .is_err()
        );
        let varying = info(BackendKind::Cuda, (16, 32));
        assert!(
            P2pLayout::Plane { planes: 2 }
                .check(&varying, Precision::F32)
                .is_err()
        );
        let cpu = info(BackendKind::Cpu, (1, 1));
        assert!(
            P2pLayout::Cube { units: 64 }
                .check(&cpu, Precision::F32)
                .is_err()
        );
        assert_eq!(P2pLayout::Plane { planes: 8 }.tile(&cpu, Precision::F32), 1);
        assert_eq!(
            P2pLayout::Cpu { vector_bits: 128 }.tile(&cpu, Precision::F64),
            8
        );
        assert!(
            P2pLayout::Cpu { vector_bits: 96 }
                .check(&cpu, Precision::F32)
                .is_err()
        );
    }

    #[test]
    fn cpu_vectors_hold_eight_targets_per_block() {
        for bits in [64, 128, 256, 512] {
            for precision in [Precision::F32, Precision::F64] {
                let (lanes, blocks) = cpu_vectors(bits, precision);
                assert_eq!(lanes * blocks, 8, "{bits} bits, {precision}");
            }
        }
        assert_eq!(cpu_vectors(128, Precision::F32), (4, 2));
        assert_eq!(cpu_vectors(128, Precision::F64), (2, 4));
        assert_eq!(cpu_vectors(256, Precision::F32), (8, 1));
        assert_eq!(cpu_vectors(256, Precision::F64), (4, 2));
    }

    #[test]
    fn grids_split_into_two_dimensions_past_the_limit() {
        let mut device = info(BackendKind::Metal, (32, 32));
        device.max_cube_count = (100, 100, 100);
        assert_eq!(cube_grid(&device, 100), (100, 1));
        assert_eq!(cube_grid(&device, 101), (100, 2));
        assert_eq!(cube_grid(&device, 10_000), (100, 100));
    }
}
