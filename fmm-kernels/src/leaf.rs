//! The device leaf expansion operators (Phase 4 T7, C4.3): P2M, L2P, P2L and M2P, with
//! the solid harmonics they evaluate by the Cartesian recursion of CONVENTIONS §3.5, f32
//! and f64, p comptime up to [`MAX_DEGREE`] (docs/design/device-path.md §6.3).
//!
//! # What each operator computes
//!
//! Every call covers one level of the plan, in the leaf-scaled data of CONVENTIONS §3.13
//! and the scaled coefficients of §3.7 in the real storage of §3.6, (p + 1)² values per
//! box. Each is `nd_fmm_ref::leaf`'s operator at the frame of §3.13, "Operators in
//! scaled coordinates", and adds into its output:
//!
//! | Operator | Rows of the view | Entries, in the order added | Frame applied | `nd_fmm_ref::leaf` |
//! | --- | --- | --- | --- | --- |
//! | [`p2m`] | box t of the level | its own leaf (if it is a local leaf), then its points in point order | ((0, 0, 0), 1) on u_s | `p2m` |
//! | [`p2l`] | box t of the level | its X list (leaves on the level above), ascending, each leaf's points in point order | (ĉ(t\|s), r̂(t\|s)) on u_s | `p2l` |
//! | [`l2p`] | local leaf `first_leaf + r` | its own box on the level | ((0, 0, 0), 1) on u_t | `l2p` |
//! | [`m2p`] | local leaf `first_leaf + r` | its W list (boxes on the level below), ascending | (ĉ(s\|t), r̂(s\|t)) on u_t | `m2p` |
//!
//! The frames are exact, formed in the kernel from the integer box and leaf indices
//! ([`BoxCoordinates`], [`LeafCoordinates`]) as the P2P kernels form theirs (CONVENTIONS
//! §3.13, "Relative frames"); never a floating-point shift. The scaled point
//! (u − ĉ) / r̂ is formed as (u − ĉ) · 2^k with the exact reciprocal 2^k of r̂, which is
//! the host's division bit for bit; so are L2P's and M2P's factors 1/r̂ and 1/r̂², applied
//! the same way. L2P runs before M2P on a level because the evaluator calls them in that
//! order (fmm-plan-redesign §7.5): two launches on one stream.
//!
//! # Arithmetic
//!
//! The harmonics follow `nd_fmm_math::harmonics` operation for operation: the recursion
//! coefficients are formed in the kernel as `harmonics` forms them, their integers comptime
//! constants and their divisions in T (for `regular`, `1 / int(2m)`,
//! `int(2n − 1) · z / int((n + m)(n − m))` and `−r² / int((n + m)(n − m))`; for
//! `irregular`, one `inv_r2 = 1 / r²` per point, then `int(2m − 1) · inv_r2`,
//! `int(2m + 1) · z · inv_r2`, `int(2n − 1) · z · inv_r2` and
//! `−int((n − 1 − m)(n − 1 + m)) · inv_r2`), never precomputed on the host. The gradients
//! are the ladders of `regular_grad` and `irregular_grad`; M2P's gradient takes Iₚ₊₁ from
//! the recursion run to degree p + 1, which is `irregular_grad`'s on-the-fly degree p + 1,
//! operation for operation. The contractions are `nd_fmm_ref::leaf`'s, per degree
//! Cₙ⁰Xₙ⁰ + 2 Σₘ≥₁ (Re Cₙᵐ Re Xₙᵐ − Im Cₙᵐ Im Xₙᵐ), with the gradient values formed on the
//! fly in that order instead of stored. The loops over n and m are unrolled (p is
//! comptime), so the harmonics live in a local array of (p + 1)² values ((p + 2)² for
//! M2P's gradient) indexed by constants.
//!
//! Like every device kernel, these follow CONVENTIONS §3.13, "Device kernels": any lone
//! a · b ± c may be fused into an fma by cubecl-opt (every backend), and Metal's division
//! and square root are not correctly rounded. So the operators agree with `nd-fmm-math`
//! and `nd_fmm_ref::leaf` to rounding, not bit for bit (spikes/device-arith/REPORT.md,
//! "Recommendation", rule 6: harmonics and the leaf operators are tested with
//! tolerances).
//!
//! # Order and determinism
//!
//! Every output value has one owning unit per launch, which loads it, adds its
//! contributions one by one in the order of the table above and stores it (device-path.md
//! §6.1): a target point for L2P and M2P, a coefficient slot of a box for P2M and P2L. No
//! atomics, and no reduction tree: P2M and P2L use the coefficient-owner scheme of
//! device-path.md §6.3, which adds the points of a box in point order, as `leaf::p2m` and
//! `leaf::p2l` do. Splitting a row into several calls at any entry gives the same bits,
//! and repeated launches give the same bits.
//!
//! # Layouts ([`LeafLayout`])
//!
//! | Layout | Default on | P2M, P2L | L2P, M2P |
//! | --- | --- | --- | --- |
//! | [`Cube`](LeafLayout::Cube) | Metal, CUDA | one cube of U units per box; unit c owns the coefficient slots c, c + U, …; the points in tiles of P: each of the first P units computes the harmonics of one point of the tile into shared memory (P (p + 1)² values and P charges), `sync_cube`, then each owner adds q_j conj(X(u_j)) for the tile's points in point order | one cube of U units per target leaf, unit u owning the targets u, u + U, …; each entry's coefficients staged once in shared memory ((p + 1)² values), `sync_cube` around each |
//! | [`Cpu`](LeafLayout::Cpu) | the CPU runtime | one cube, one unit per core (at most [`Device::units_cap`]), each a contiguous range of boxes, every coefficient of a box in a local array | the same, each a contiguous range of target leaves, each entry's coefficients copied to a local array once per target leaf |
//!
//! Both layouts add every output's contributions in the same order, so every test runs on
//! both, and the cube layout also runs on the CPU runtime (correctness only: a kernel
//! with shared memory gets a worker per unit there, device-path.md F18). P, U and p are
//! comptime, so each (layout, operator, p, precision, gradients) is one compiled kernel.
//! A row without entries, or a leaf without points, adds nothing; a call without entries
//! launches nothing.
//!
//! # Launch
//!
//! [`p2m`], [`p2l`], [`l2p`] and [`m2p`] are the safe launch wrappers of one level's call:
//! each checks every length, owner and index bound on the host first, launches once and
//! allocates nothing. [`x_frames`] and [`w_frames`] write the frames the kernels form,
//! and [`harmonics`] the harmonics at points, for tests.

use std::fmt;

use cubecl::prelude::*;

use crate::buffer::{DeviceFloat, DeviceSlice, DeviceSliceMut};
use crate::device::{BackendKind, Device, DeviceInfo, Precision};
use crate::error::KernelError;
use crate::frame;
use crate::p2p::cube_grid;
use crate::view::{BoxCoordinates, IndexView, LeafCoordinates, PointOffsets};

/// The largest degree p the leaf kernels take: that of the FMM (CONVENTIONS §3.9, M2L up
/// to p = 20).
pub const MAX_DEGREE: usize = 20;

/// The units per cube of the [`Cube`](LeafLayout::Cube) layout by default on the GPU
/// backends: 64 (target points per block of L2P and M2P, coefficient owners of P2M and
/// P2L).
pub const GPU_LEAF_UNITS: u32 = 64;

/// The point tile of P2M and P2L in the [`Cube`](LeafLayout::Cube) layout by default: 32
/// points, fewer where the shared memory does not hold them (device-path.md §6.3: 10 KB
/// at p = 8 in f32).
pub const GPU_LEAF_TILE: u32 = 32;

/// The number of coefficients of degree ≤ p, (p + 1)².
pub const fn coefficients(p: usize) -> usize {
    (p + 1) * (p + 1)
}

/// The slot of (n, m) in real storage (CONVENTIONS §3.6): n² + n + m.
const fn slot(n: usize, m: i64) -> usize {
    ((n * n + n) as i64 + m) as usize
}

/// Whether slot c holds an imaginary part (m < 0).
const fn imaginary(c: usize) -> bool {
    let mut n = 0;
    while (n + 1) * (n + 1) <= c {
        n += 1;
    }
    c < n * n + n
}

/// The layout of a leaf-operator launch (module documentation, "Layouts").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LeafLayout {
    /// One cube of `units` units per box (P2M, P2L) or target leaf (L2P, M2P); P2M and
    /// P2L stage the harmonics of `tile` points at a time in shared memory.
    Cube {
        /// Units per cube, at least 1.
        units: u32,
        /// Points per tile of P2M and P2L, 1 ≤ `tile` ≤ `units`.
        tile: u32,
    },
    /// One cube of one unit per core, each a contiguous range of rows, no shared memory.
    Cpu,
}

impl LeafLayout {
    /// The default layout of a device at degree p in `precision`: [`Cpu`](Self::Cpu) on
    /// the CPU runtime, otherwise [`Cube`](Self::Cube) with [`GPU_LEAF_UNITS`] units (fewer
    /// if the device allows fewer per cube) and a tile of [`GPU_LEAF_TILE`] points, fewer
    /// if the shared memory holds fewer. Chosen at build from the backend
    /// (device-path.md §6.1, §6.3).
    pub fn default_for(info: &DeviceInfo, p: usize, precision: Precision) -> Self {
        match info.backend {
            BackendKind::Cpu => Self::Cpu,
            BackendKind::Metal | BackendKind::Cuda => {
                let units = GPU_LEAF_UNITS.min(info.max_units_per_cube.max(1));
                let per_point = (coefficients(p) + 1) * value_bytes(precision);
                let fit = (info.max_shared_memory / per_point).max(1);
                let tile = (GPU_LEAF_TILE as usize).min(units as usize).min(fit);
                Self::Cube {
                    units,
                    tile: tile as u32,
                }
            }
        }
    }

    /// Checks that the device can run the layout at degree p in `precision`.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedLayout`] for zero units or a tile of zero or more points
    /// than units, more units per cube than the device allows, or more shared memory
    /// than it has: the larger of P ((p + 1)² + 1) values (P2M, P2L) and (p + 1)² values
    /// (L2P, M2P).
    pub fn check(
        self,
        info: &DeviceInfo,
        p: usize,
        precision: Precision,
    ) -> Result<(), KernelError> {
        let refuse = |reason: String| {
            Err(KernelError::UnsupportedLayout {
                layout: self.to_string(),
                reason,
            })
        };
        let Self::Cube { units, tile } = self else {
            return Ok(());
        };
        if units == 0 || tile == 0 || tile > units {
            return refuse(format!("{units} units with a tile of {tile} points"));
        }
        if units > info.max_units_per_cube {
            return refuse(format!(
                "{units} units per cube, the device allows {}",
                info.max_units_per_cube
            ));
        }
        let n = coefficients(p);
        let shared = (tile as usize * (n + 1)).max(n) * value_bytes(precision);
        if shared > info.max_shared_memory {
            return refuse(format!(
                "{shared} bytes of shared memory at p = {p}, the device has {}",
                info.max_shared_memory
            ));
        }
        Ok(())
    }
}

impl fmt::Display for LeafLayout {
    /// `cube (64 units, tile 32)` or `cpu (one unit per core)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube { units, tile } => write!(f, "cube ({units} units, tile {tile})"),
            Self::Cpu => f.write_str("cpu (one unit per core)"),
        }
    }
}

/// Bytes per value in `precision`.
fn value_bytes(precision: Precision) -> usize {
    match precision {
        Precision::F32 => 4,
        Precision::F64 => 8,
    }
}

/// The solid harmonics of [`harmonics`]: regular Rₙᵐ or irregular Iₙᵐ (CONVENTIONS
/// §3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Basis {
    /// Rₙᵐ, `nd_fmm_math::harmonics::regular` (and `regular_grad`).
    Regular,
    /// Iₙᵐ, `nd_fmm_math::harmonics::irregular` (and `irregular_grad`).
    Irregular,
}

/// The inputs of one level's P2M or P2L call ([`p2m`], [`p2l`]): boxes of a level and
/// the source leaves of each.
#[derive(Clone, Copy, Debug)]
pub struct SourceInputs<'a, T: DeviceFloat> {
    /// Row t, for box t of `level`: the leaves whose sources it takes, in the order they
    /// are added (P2M: the box's own leaf, if it is a local leaf; P2L: its X list, leaves
    /// on the level above, ascending).
    pub view: &'a IndexView,
    /// The level of the boxes.
    pub level: usize,
    /// The index of every box of every level (P2L's frames).
    pub boxes: &'a BoxCoordinates,
    /// The level and index of every leaf (P2L's frames).
    pub leaves: &'a LeafCoordinates,
    /// The point offsets of the source store, by leaf.
    pub source_offsets: &'a PointOffsets,
    /// The source store: per leaf of n points, n leaf-scaled triples, then n charges.
    pub sources: DeviceSlice<'a, T>,
}

/// The inputs of one level's L2P or M2P call ([`l2p`], [`m2p`]): target leaves of a level
/// and the boxes whose expansions each evaluates.
#[derive(Clone, Copy, Debug)]
pub struct TargetInputs<'a, T: DeviceFloat> {
    /// Row r, for leaf `first_leaf + r`: the boxes whose expansions it evaluates, in the
    /// order they are added (L2P: the leaf's own box on `level`; M2P: its W list, boxes on
    /// `level + 1`, ascending).
    pub view: &'a IndexView,
    /// The level of the target leaves.
    pub level: usize,
    /// The leaf of row 0; rows are consecutive leaves.
    pub first_leaf: usize,
    /// The index of every box of every level (M2P's frames).
    pub boxes: &'a BoxCoordinates,
    /// The level and index of every leaf (M2P's frames).
    pub leaves: &'a LeafCoordinates,
    /// The point offsets of the target input and output, by leaf.
    pub target_offsets: &'a PointOffsets,
    /// The target input: per leaf of n points, n leaf-scaled triples.
    pub target_input: DeviceSlice<'a, T>,
}

// --- Harmonics -----------------------------------------------------------------------

/// Rₙᵐ(x, y, z), n ≤ `p`, into `values` in real storage: `nd_fmm_math::harmonics::regular`
/// operation for operation (module documentation, "Arithmetic").
#[cube]
fn regular<F: Float>(x: F, y: F, z: F, values: &mut Array<F>, #[comptime] p: usize) {
    let r2 = x * x + y * y + z * z;
    let mut re = F::new(1.0f32);
    let mut im = F::new(0.0f32);
    #[unroll]
    for m in 0..p + 1 {
        if comptime!(m > 0) {
            let s = F::new(1.0f32) / F::new(comptime!((2 * m) as f32));
            let next_re = (x * re - y * im) * s;
            let next_im = (x * im + y * re) * s;
            re = next_re;
            im = next_im;
        }
        values[comptime!(slot(m, m as i64))] = re;
        if comptime!(m > 0) {
            values[comptime!(slot(m, -(m as i64)))] = im;
        }
        if comptime!(m < p) {
            values[comptime!(slot(m + 1, m as i64))] = z * re;
            if comptime!(m > 0) {
                values[comptime!(slot(m + 1, -(m as i64)))] = z * im;
            }
            #[unroll]
            for n in m + 2..p + 1 {
                let denominator = F::new(comptime!(((n + m) * (n - m)) as f32));
                let c1 = F::new(comptime!((2 * n - 1) as f32)) * z / denominator;
                let c2 = -r2 / denominator;
                three_term::<F>(values, c1, c2, n, m);
            }
        }
    }
}

/// Iₙᵐ(x, y, z), n ≤ `p`, into `values` in real storage:
/// `nd_fmm_math::harmonics::irregular` operation for operation, with one
/// `inv_r2 = 1 / r²` per point (module documentation, "Arithmetic").
#[cube]
fn irregular<F: Float>(x: F, y: F, z: F, values: &mut Array<F>, #[comptime] p: usize) {
    let r2 = x * x + y * y + z * z;
    let inv_r2 = F::new(1.0f32) / r2;
    let mut re = F::new(1.0f32) / r2.sqrt();
    let mut im = F::new(0.0f32);
    #[unroll]
    for m in 0..p + 1 {
        if comptime!(m > 0) {
            let s = F::new(comptime!((2 * m - 1) as f32)) * inv_r2;
            let next_re = (x * re - y * im) * s;
            let next_im = (x * im + y * re) * s;
            re = next_re;
            im = next_im;
        }
        values[comptime!(slot(m, m as i64))] = re;
        if comptime!(m > 0) {
            values[comptime!(slot(m, -(m as i64)))] = im;
        }
        if comptime!(m < p) {
            let c = F::new(comptime!((2 * m + 1) as f32)) * z * inv_r2;
            values[comptime!(slot(m + 1, m as i64))] = c * re;
            if comptime!(m > 0) {
                values[comptime!(slot(m + 1, -(m as i64)))] = c * im;
            }
            #[unroll]
            for n in m + 2..p + 1 {
                let c1 = F::new(comptime!((2 * n - 1) as f32)) * z * inv_r2;
                let c2 = F::new(comptime!(-(((n - 1 - m) * (n - 1 + m)) as f32))) * inv_r2;
                three_term::<F>(values, c1, c2, n, m);
            }
        }
    }
}

/// Xₙᵐ = c₁ Xₙ₋₁ᵐ + c₂ Xₙ₋₂ᵐ, real and imaginary parts alike (`three_term` of
/// `nd_fmm_math::harmonics`).
#[cube]
fn three_term<F: Float>(
    values: &mut Array<F>,
    c1: F,
    c2: F,
    #[comptime] n: usize,
    #[comptime] m: usize,
) {
    let one = comptime!(slot(n - 1, m as i64));
    let two = comptime!(slot(n - 2, m as i64));
    values[comptime!(slot(n, m as i64))] = c1 * values[one] + c2 * values[two];
    if comptime!(m > 0) {
        let one = comptime!(slot(n - 1, -(m as i64)));
        let two = comptime!(slot(n - 2, -(m as i64)));
        values[comptime!(slot(n, -(m as i64)))] = c1 * values[one] + c2 * values[two];
    }
}

/// The harmonics of `basis` (comptime `irregular`) at (x, y, z), degrees ≤ `degree`.
#[cube]
fn solid<F: Float>(
    x: F,
    y: F,
    z: F,
    values: &mut Array<F>,
    #[comptime] degree: usize,
    #[comptime] irregular_basis: bool,
) {
    if irregular_basis {
        irregular::<F>(x, y, z, values, degree);
    } else {
        regular::<F>(x, y, z, values, degree);
    }
}

/// The complex value of X_ns^m from real storage for any integer m, zero for |m| > ns
/// and for ns < 0 (`value_at` of `nd_fmm_math::harmonics`, with the conjugate symmetry of
/// CONVENTIONS §3.3).
#[cube]
fn value_at<F: Float>(values: &Array<F>, #[comptime] ns: i64, #[comptime] m: i64) -> (F, F) {
    let k = comptime!(m.abs());
    let mut re = F::new(0.0f32);
    let mut im = F::new(0.0f32);
    if comptime!(ns >= 0 && k <= ns) {
        re = values[comptime!(slot(ns as usize, k))];
        if comptime!(k > 0) {
            im = values[comptime!(slot(ns as usize, -k))];
        }
        if comptime!(m < 0 && k % 2 == 0) {
            im = -im;
        }
        if comptime!(m < 0 && k % 2 == 1) {
            re = -re;
        }
    }
    (re, im)
}

/// The gradient of Xₙᵐ, m ≥ 0, from the ladder of `regular_grad` (degree n − 1) or
/// `irregular_grad` (degree n + 1, σ = −1): (∂x re, ∂x im, ∂y re, ∂y im, ∂z re, ∂z im),
/// as `store_ladder` of `nd_fmm_math::harmonics` forms them.
#[cube]
fn ladder<F: Float>(
    values: &Array<F>,
    #[comptime] n: usize,
    #[comptime] m: usize,
    #[comptime] irregular_basis: bool,
) -> (F, F, F, F, F, F) {
    let ns = comptime!(if irregular_basis {
        n as i64 + 1
    } else {
        n as i64 - 1
    });
    let (ar, ai) = value_at::<F>(values, ns, comptime!(m as i64 - 1));
    let (br, bi) = value_at::<F>(values, ns, comptime!(m as i64 + 1));
    let (cr, ci) = value_at::<F>(values, ns, comptime!(m as i64));
    let half = F::new(0.5f32);
    let mut zr = cr;
    let mut zi = ci;
    if irregular_basis {
        zr = -cr;
        zi = -ci;
    }
    (
        (ar - br) * half,
        (ai - bi) * half,
        -(ai + bi) * half,
        (ar + br) * half,
        zr,
        zi,
    )
}

/// Σₙ≤ₚ Σₘ Cₙᵐ Xₙᵐ from real storage by the doubling rule of CONVENTIONS §3.6, with the
/// coefficients at `coefficients[base..]`: `contract` of `nd_fmm_ref::leaf`.
#[cube]
fn contract<F: Float, L: List<F>>(
    coefficients: &L,
    base: usize,
    values: &Array<F>,
    #[comptime] p: usize,
) -> F {
    let two = F::new(2.0f32);
    let mut sum = F::new(0.0f32);
    #[unroll]
    for n in 0..p + 1 {
        let zero = comptime!(slot(n, 0));
        let mut orders = F::new(0.0f32);
        #[unroll]
        for m in 1..n + 1 {
            let re = comptime!(slot(n, m as i64));
            let im = comptime!(slot(n, -(m as i64)));
            orders = orders + coefficients[base + re] * values[re]
                - coefficients[base + im] * values[im];
        }
        sum = sum + coefficients[base + zero] * values[zero] + two * orders;
    }
    sum
}

/// The three Cartesian components of Σₙ≤ₚ Σₘ Cₙᵐ (∇Xₙᵐ), each contracted as
/// [`contract`] contracts the values, the gradient values formed by [`ladder`] in that
/// order.
#[cube]
fn contract_gradient<F: Float, L: List<F>>(
    coefficients: &L,
    base: usize,
    values: &Array<F>,
    #[comptime] p: usize,
    #[comptime] irregular_basis: bool,
) -> (F, F, F) {
    let two = F::new(2.0f32);
    let mut sx = F::new(0.0f32);
    let mut sy = F::new(0.0f32);
    let mut sz = F::new(0.0f32);
    #[unroll]
    for n in 0..p + 1 {
        let zero = comptime!(slot(n, 0));
        let mut ox = F::new(0.0f32);
        let mut oy = F::new(0.0f32);
        let mut oz = F::new(0.0f32);
        #[unroll]
        for m in 1..n + 1 {
            let re = comptime!(slot(n, m as i64));
            let im = comptime!(slot(n, -(m as i64)));
            let (xr, xi, yr, yi, zr, zi) = ladder::<F>(values, n, m, irregular_basis);
            let cr = coefficients[base + re];
            let ci = coefficients[base + im];
            ox = ox + cr * xr - ci * xi;
            oy = oy + cr * yr - ci * yi;
            oz = oz + cr * zr - ci * zi;
        }
        let (x0, _xi, y0, _yi, z0, _zi) = ladder::<F>(values, n, 0usize, irregular_basis);
        let c0 = coefficients[base + zero];
        sx = sx + c0 * x0 + two * ox;
        sy = sy + c0 * y0 + two * oy;
        sz = sz + c0 * z0 + two * oz;
    }
    (sx, sy, sz)
}

// --- Kernels ---------------------------------------------------------------------------

/// P2M and P2L, the cube layout: one cube per box, coefficient owners, point tiles staged
/// in shared memory (module documentation).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn expand_cube_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    boxes: &[u32],
    leaves: &[u32],
    source_offsets: &[u32],
    sources: &[F],
    coefficients: &mut [F],
    box_base: u32,
    level: u32,
    source_base: u32,
    coefficient_base: u32,
    nrows: u32,
    #[comptime] p: usize,
    #[comptime] units: usize,
    #[comptime] tile: usize,
    #[comptime] irregular_basis: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let owned = comptime!(nc.div_ceil(units));
    let mut staged = Shared::<[F]>::new_slice(comptime!(tile * nc));
    let mut charges = Shared::<[F]>::new_slice(tile);
    let mut values = Array::<F>::new(nc);
    let mut acc = Array::<F>::new(owned);
    let unit = UNIT_POS as usize;
    let row = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    // The row and its entry range are the same for every unit of the cube, so every
    // barrier below is reached by all of them or by none.
    if row < nrows as usize {
        let e_first = row_offsets[row] as usize;
        let e_last = row_offsets[row + 1] as usize;
        if e_first < e_last {
            let out0 = coefficient_base as usize + row * nc;
            #[unroll]
            for k in 0..owned {
                let c = unit + k * units;
                if c < nc {
                    acc[k] = coefficients[out0 + c];
                }
            }
            let tb = 3 * (box_base as usize + row);
            for e in e_first..e_last {
                let j = entries[e] as usize;
                let s_first = source_offsets[j] as usize;
                let n_s = source_offsets[j + 1] as usize - s_first;
                let base = source_base as usize + 4 * s_first;
                // P2L: the frame (ĉ(t|s), r̂(t|s)) of the box seen from the leaf, applied
                // as (u − ĉ) · 2^(l_t − l_s).
                let ls = leaves[4 * j];
                let mut c0 = F::new(0.0f32);
                let mut c1 = F::new(0.0f32);
                let mut c2 = F::new(0.0f32);
                let mut scale = F::new(1.0f32);
                if irregular_basis {
                    c0 = frame::centre::<F>(level, boxes[tb], ls, leaves[4 * j + 1]);
                    c1 = frame::centre::<F>(level, boxes[tb + 1], ls, leaves[4 * j + 2]);
                    c2 = frame::centre::<F>(level, boxes[tb + 2], ls, leaves[4 * j + 3]);
                    scale = frame::ratio::<F>(ls, level);
                }
                let mut k0 = 0usize;
                while k0 < n_s {
                    let i = k0 + unit;
                    if unit < tile && i < n_s {
                        let mut v0 = sources[base + 3 * i];
                        let mut v1 = sources[base + 3 * i + 1];
                        let mut v2 = sources[base + 3 * i + 2];
                        if irregular_basis {
                            v0 = (v0 - c0) * scale;
                            v1 = (v1 - c1) * scale;
                            v2 = (v2 - c2) * scale;
                        }
                        solid::<F>(v0, v1, v2, &mut values, p, irregular_basis);
                        // conj(X): the imaginary parts negated, so that every owner adds
                        // q times the staged value.
                        let at = unit * nc;
                        #[unroll]
                        for c in 0..nc {
                            if comptime!(imaginary(c)) {
                                staged[at + c] = -values[c];
                            } else {
                                staged[at + c] = values[c];
                            }
                        }
                        charges[unit] = sources[base + 3 * n_s + i];
                    }
                    sync_cube();
                    let mut count = n_s - k0;
                    if count > tile {
                        count = tile;
                    }
                    for i in 0..count {
                        let q = charges[i];
                        #[unroll]
                        for k in 0..owned {
                            let c = unit + k * units;
                            if c < nc {
                                acc[k] += q * staged[i * nc + c];
                            }
                        }
                    }
                    sync_cube();
                    k0 += tile;
                }
            }
            #[unroll]
            for k in 0..owned {
                let c = unit + k * units;
                if c < nc {
                    coefficients[out0 + c] = acc[k];
                }
            }
        }
    }
}

/// P2M and P2L, the CPU layout: one cube, each unit a contiguous range of boxes, every
/// coefficient of a box in a local array.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn expand_cpu_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    boxes: &[u32],
    leaves: &[u32],
    source_offsets: &[u32],
    sources: &[F],
    coefficients: &mut [F],
    box_base: u32,
    level: u32,
    source_base: u32,
    coefficient_base: u32,
    nrows: u32,
    #[comptime] p: usize,
    #[comptime] irregular_basis: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let mut values = Array::<F>::new(nc);
    let mut acc = Array::<F>::new(nc);
    let (first, count) = unit_rows(nrows as usize);
    for row in first..first + count {
        let e_first = row_offsets[row] as usize;
        let e_last = row_offsets[row + 1] as usize;
        if e_first < e_last {
            let out0 = coefficient_base as usize + row * nc;
            #[unroll]
            for c in 0..nc {
                acc[c] = coefficients[out0 + c];
            }
            let tb = 3 * (box_base as usize + row);
            for e in e_first..e_last {
                let j = entries[e] as usize;
                let s_first = source_offsets[j] as usize;
                let n_s = source_offsets[j + 1] as usize - s_first;
                let base = source_base as usize + 4 * s_first;
                let ls = leaves[4 * j];
                let mut c0 = F::new(0.0f32);
                let mut c1 = F::new(0.0f32);
                let mut c2 = F::new(0.0f32);
                let mut scale = F::new(1.0f32);
                if irregular_basis {
                    c0 = frame::centre::<F>(level, boxes[tb], ls, leaves[4 * j + 1]);
                    c1 = frame::centre::<F>(level, boxes[tb + 1], ls, leaves[4 * j + 2]);
                    c2 = frame::centre::<F>(level, boxes[tb + 2], ls, leaves[4 * j + 3]);
                    scale = frame::ratio::<F>(ls, level);
                }
                for i in 0..n_s {
                    let mut v0 = sources[base + 3 * i];
                    let mut v1 = sources[base + 3 * i + 1];
                    let mut v2 = sources[base + 3 * i + 2];
                    if irregular_basis {
                        v0 = (v0 - c0) * scale;
                        v1 = (v1 - c1) * scale;
                        v2 = (v2 - c2) * scale;
                    }
                    solid::<F>(v0, v1, v2, &mut values, p, irregular_basis);
                    let q = sources[base + 3 * n_s + i];
                    // `add_conjugate` of `nd_fmm_ref::leaf`.
                    #[unroll]
                    for c in 0..nc {
                        if comptime!(imaginary(c)) {
                            acc[c] -= q * values[c];
                        } else {
                            acc[c] += q * values[c];
                        }
                    }
                }
            }
            #[unroll]
            for c in 0..nc {
                coefficients[out0 + c] = acc[c];
            }
        }
    }
}

/// The contiguous range of `rows` of this unit, (first, count): the rows split as evenly
/// as possible over the units of the one cube, the first `rows % units` units one more.
#[cube]
fn unit_rows(rows: usize) -> (usize, usize) {
    let units = CUBE_DIM as usize;
    let unit = UNIT_POS as usize;
    let share = rows / units;
    let extra = rows % units;
    let mut first = unit * share + extra;
    let mut count = share;
    if unit < extra {
        first = unit * share + unit;
        count = share + 1;
    }
    (first, count)
}

/// Adds one expansion's potential (and gradient) to a target's accumulators, from the
/// harmonics `values` at its scaled point: the contractions, scaled by 1/r̂ and 1/r̂²
/// (`scale`, exact) for M2P, as `leaf::m2p` divides by r̂ and r̂².
#[cube]
fn accumulate<F: Float, L: List<F>>(
    coefficients: &L,
    values: &Array<F>,
    scale: F,
    accumulators: (F, F, F, F),
    #[comptime] p: usize,
    #[comptime] irregular_basis: bool,
    #[comptime] gradients: bool,
) -> (F, F, F, F) {
    let (mut phi, mut g0, mut g1, mut g2) = accumulators;
    let sum = contract::<F, L>(coefficients, 0usize, values, p);
    if irregular_basis {
        phi += sum * scale;
    } else {
        phi += sum;
    }
    if gradients {
        let (sx, sy, sz) =
            contract_gradient::<F, L>(coefficients, 0usize, values, p, irregular_basis);
        if irregular_basis {
            let scale2 = scale * scale;
            g0 += sx * scale2;
            g1 += sy * scale2;
            g2 += sz * scale2;
        } else {
            g0 += sx;
            g1 += sy;
            g2 += sz;
        }
    }
    (phi, g0, g1, g2)
}

/// The frame of entry box s (on `entry_level`) seen from target leaf t, for M2P:
/// (ĉ(s|t), 1/r̂(s|t) = 2^(l_s − l_t)); for L2P the unit frame.
#[cube]
fn target_frame<F: Float>(
    boxes: &[u32],
    leaves: &[u32],
    box_base: usize,
    entry_level: u32,
    s: usize,
    t_leaf: usize,
    #[comptime] irregular_basis: bool,
) -> (F, F, F, F) {
    let mut c0 = F::new(0.0f32);
    let mut c1 = F::new(0.0f32);
    let mut c2 = F::new(0.0f32);
    let mut scale = F::new(1.0f32);
    if irregular_basis {
        let sb = 3 * (box_base + s);
        let lt = leaves[4 * t_leaf];
        c0 = frame::centre::<F>(entry_level, boxes[sb], lt, leaves[4 * t_leaf + 1]);
        c1 = frame::centre::<F>(entry_level, boxes[sb + 1], lt, leaves[4 * t_leaf + 2]);
        c2 = frame::centre::<F>(entry_level, boxes[sb + 2], lt, leaves[4 * t_leaf + 3]);
        scale = frame::ratio::<F>(lt, entry_level);
    }
    (c0, c1, c2, scale)
}

/// L2P and M2P, the cube layout: one cube per target leaf, one unit per target point,
/// each entry's coefficients staged in shared memory (module documentation).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn evaluate_cube_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    boxes: &[u32],
    leaves: &[u32],
    target_offsets: &[u32],
    target_input: &[F],
    coefficients: &[F],
    target_output: &mut [F],
    box_base: u32,
    entry_level: u32,
    coefficient_base: u32,
    input_base: u32,
    output_base: u32,
    first_leaf: u32,
    nrows: u32,
    #[comptime] p: usize,
    #[comptime] units: usize,
    #[comptime] irregular_basis: bool,
    #[comptime] gradients: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let degree = comptime!(if irregular_basis && gradients {
        p + 1
    } else {
        p
    });
    let mut staged = Shared::<[F]>::new_slice(nc);
    let mut values = Array::<F>::new(comptime!((degree + 1) * (degree + 1)));
    let unit = UNIT_POS as usize;
    let row = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    // The row, its entries and its targets are the same for every unit of the cube, so
    // every barrier below is reached by all of them or by none.
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
            let t = block + unit;
            let active = t < n_t;
            let mut x0 = F::new(0.0f32);
            let mut x1 = F::new(0.0f32);
            let mut x2 = F::new(0.0f32);
            let mut acc = (
                F::new(0.0f32),
                F::new(0.0f32),
                F::new(0.0f32),
                F::new(0.0f32),
            );
            if active {
                x0 = target_input[in0 + 3 * t];
                x1 = target_input[in0 + 3 * t + 1];
                x2 = target_input[in0 + 3 * t + 2];
                acc.0 = target_output[phi0 + t];
                if gradients {
                    acc.1 = target_output[grad0 + 3 * t];
                    acc.2 = target_output[grad0 + 3 * t + 1];
                    acc.3 = target_output[grad0 + 3 * t + 2];
                }
            }
            for e in e_first..e_last {
                let s = entries[e] as usize;
                let cbase = coefficient_base as usize + s * nc;
                let mut c = unit;
                while c < nc {
                    staged[c] = coefficients[cbase + c];
                    c += units;
                }
                let (c0, c1, c2, scale) = target_frame::<F>(
                    boxes,
                    leaves,
                    box_base as usize,
                    entry_level,
                    s,
                    t_leaf,
                    irregular_basis,
                );
                sync_cube();
                if active {
                    let mut v0 = x0;
                    let mut v1 = x1;
                    let mut v2 = x2;
                    if irregular_basis {
                        v0 = (x0 - c0) * scale;
                        v1 = (x1 - c1) * scale;
                        v2 = (x2 - c2) * scale;
                    }
                    solid::<F>(v0, v1, v2, &mut values, degree, irregular_basis);
                    acc = accumulate::<F, Shared<[F]>>(
                        &staged,
                        &values,
                        scale,
                        acc,
                        p,
                        irregular_basis,
                        gradients,
                    );
                }
                sync_cube();
            }
            if active {
                target_output[phi0 + t] = acc.0;
                if gradients {
                    target_output[grad0 + 3 * t] = acc.1;
                    target_output[grad0 + 3 * t + 1] = acc.2;
                    target_output[grad0 + 3 * t + 2] = acc.3;
                }
            }
            block += units;
        }
    }
}

/// L2P and M2P, the CPU layout: one cube, each unit a contiguous range of target leaves;
/// per entry the coefficients copied to a local array, then every target of the leaf
/// (its accumulators loaded and stored around each entry, which keeps every target's
/// order).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn evaluate_cpu_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    boxes: &[u32],
    leaves: &[u32],
    target_offsets: &[u32],
    target_input: &[F],
    coefficients: &[F],
    target_output: &mut [F],
    box_base: u32,
    entry_level: u32,
    coefficient_base: u32,
    input_base: u32,
    output_base: u32,
    first_leaf: u32,
    nrows: u32,
    #[comptime] p: usize,
    #[comptime] irregular_basis: bool,
    #[comptime] gradients: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let degree = comptime!(if irregular_basis && gradients {
        p + 1
    } else {
        p
    });
    let mut staged = Array::<F>::new(nc);
    let mut values = Array::<F>::new(comptime!((degree + 1) * (degree + 1)));
    let (first, count) = unit_rows(nrows as usize);
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
        for e in row_offsets[row] as usize..row_offsets[row + 1] as usize {
            let s = entries[e] as usize;
            let cbase = coefficient_base as usize + s * nc;
            #[unroll]
            for c in 0..nc {
                staged[c] = coefficients[cbase + c];
            }
            let (c0, c1, c2, scale) = target_frame::<F>(
                boxes,
                leaves,
                box_base as usize,
                entry_level,
                s,
                t_leaf,
                irregular_basis,
            );
            for t in 0..n_t {
                let x0 = target_input[in0 + 3 * t];
                let x1 = target_input[in0 + 3 * t + 1];
                let x2 = target_input[in0 + 3 * t + 2];
                let mut acc = (
                    target_output[phi0 + t],
                    F::new(0.0f32),
                    F::new(0.0f32),
                    F::new(0.0f32),
                );
                if gradients {
                    acc.1 = target_output[grad0 + 3 * t];
                    acc.2 = target_output[grad0 + 3 * t + 1];
                    acc.3 = target_output[grad0 + 3 * t + 2];
                }
                let mut v0 = x0;
                let mut v1 = x1;
                let mut v2 = x2;
                if irregular_basis {
                    v0 = (x0 - c0) * scale;
                    v1 = (x1 - c1) * scale;
                    v2 = (x2 - c2) * scale;
                }
                solid::<F>(v0, v1, v2, &mut values, degree, irregular_basis);
                acc = accumulate::<F, Array<F>>(
                    &staged,
                    &values,
                    scale,
                    acc,
                    p,
                    irregular_basis,
                    gradients,
                );
                target_output[phi0 + t] = acc.0;
                if gradients {
                    target_output[grad0 + 3 * t] = acc.1;
                    target_output[grad0 + 3 * t + 1] = acc.2;
                    target_output[grad0 + 3 * t + 2] = acc.3;
                }
            }
        }
    }
}

/// The harmonics (and gradients) of `count` points, for tests: per point (p + 1)² values,
/// then with gradients ∂x, ∂y and ∂z, each (p + 1)² values.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn harmonics_kernel<F: Float>(
    points: &[F],
    out: &mut [F],
    points_base: u32,
    out_base: u32,
    count: u32,
    #[comptime] p: usize,
    #[comptime] irregular_basis: bool,
    #[comptime] gradients: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let degree = comptime!(if irregular_basis && gradients {
        p + 1
    } else {
        p
    });
    let stride = comptime!(if gradients { 4 * nc } else { nc });
    let mut values = Array::<F>::new(comptime!((degree + 1) * (degree + 1)));
    let threads = CUBE_COUNT * CUBE_DIM as usize;
    let mut i = ABSOLUTE_POS;
    while i < count as usize {
        let at = points_base as usize + 3 * i;
        solid::<F>(
            points[at],
            points[at + 1],
            points[at + 2],
            &mut values,
            degree,
            irregular_basis,
        );
        let o = out_base as usize + i * stride;
        #[unroll]
        for c in 0..nc {
            out[o + c] = values[c];
        }
        if gradients {
            #[unroll]
            for n in 0..p + 1 {
                #[unroll]
                for m in 0..n + 1 {
                    let (xr, xi, yr, yi, zr, zi) = ladder::<F>(&values, n, m, irregular_basis);
                    let re = comptime!(slot(n, m as i64));
                    out[o + nc + re] = xr;
                    out[o + 2 * nc + re] = yr;
                    out[o + 3 * nc + re] = zr;
                    if comptime!(m > 0) {
                        let im = comptime!(slot(n, -(m as i64)));
                        out[o + nc + im] = xi;
                        out[o + 2 * nc + im] = yi;
                        out[o + 3 * nc + im] = zi;
                    }
                }
            }
        }
        i += threads;
    }
}

/// The frames of every entry of an X view (`w` false: box t of `level` seen from leaf s,
/// (ĉ(t|s), r̂(t|s))) or a W view (`w` true: box s of `level + 1` seen from leaf
/// `first_leaf + r`, (ĉ(s|t), r̂(s|t))); four values per entry, rows in blocks of `chunk`
/// per unit and stride.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn far_frames_kernel<F: Float>(
    row_offsets: &[u32],
    entries: &[u32],
    boxes: &[u32],
    leaves: &[u32],
    frames: &mut [F],
    frames_base: u32,
    box_base: u32,
    box_level: u32,
    first_leaf: u32,
    nrows: u32,
    chunk: u32,
    threads: u32,
    #[comptime] w: bool,
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
            for e in row_offsets[row] as usize..row_offsets[row + 1] as usize {
                let j = entries[e] as usize;
                let at = frames_base as usize + 4 * e;
                // (box, leaf): X, box `row` and leaf j; W, box j and leaf first_leaf + row.
                let mut b = row;
                let mut leaf = j;
                if w {
                    b = j;
                    leaf = first_leaf as usize + row;
                }
                // Both are the frame of the box seen from the leaf.
                let bb = 3 * (box_base as usize + b);
                let ll = leaves[4 * leaf];
                for k in 0..3usize {
                    frames[at + k] =
                        frame::centre::<F>(box_level, boxes[bb + k], ll, leaves[4 * leaf + 1 + k]);
                }
                frames[at + 3] = frame::ratio::<F>(box_level, ll);
            }
        }
        start += stride;
    }
}

// --- Launch wrappers -------------------------------------------------------------------

/// The checks of a P2M or P2L call; returns the number of rows.
fn check_sources<T: DeviceFloat>(
    device: &Device,
    what: &str,
    p: usize,
    inputs: &SourceInputs<'_, T>,
    out: &DeviceSliceMut<'_, T>,
) -> Result<usize, KernelError> {
    let SourceInputs {
        view,
        level,
        boxes,
        leaves,
        source_offsets,
        sources,
    } = *inputs;
    for owner in [
        view.row_offsets().device(),
        view.entries().device(),
        boxes.buffer().as_slice().device(),
        leaves.buffer().as_slice().device(),
        source_offsets.buffer().as_slice().device(),
        sources.device(),
        out.device(),
    ] {
        device.check_owner(owner)?;
    }
    assert!(p <= MAX_DEGREE, "{what}: p = {p} exceeds {MAX_DEGREE}");
    assert!(
        level < boxes.nlevels(),
        "{what}: level {level} of {} levels",
        boxes.nlevels()
    );
    let nrows = view.nrows();
    assert_eq!(
        nrows,
        boxes.len(level),
        "{what}: one row per box of level {level}"
    );
    assert!(
        view.is_empty()
            || (view.entries().bound() <= source_offsets.nleaves() as u64
                && view.entries().bound() <= leaves.len() as u64),
        "{what}: entries up to {} for {} source leaves and {} leaf coordinates",
        view.entries().bound().saturating_sub(1),
        source_offsets.nleaves(),
        leaves.len()
    );
    assert_eq!(
        sources.len(),
        4 * source_offsets.total(),
        "{what}: the source store holds 4 values per point"
    );
    assert_eq!(
        out.len(),
        coefficients(p) * nrows,
        "{what}: the coefficients of the level, (p + 1)^2 per box"
    );
    Ok(nrows)
}

/// The checks of an L2P or M2P call, whose entries are boxes on `entry_level`; returns
/// the number of rows.
fn check_targets<T: DeviceFloat>(
    device: &Device,
    what: &str,
    (p, entry_level): (usize, usize),
    inputs: &TargetInputs<'_, T>,
    coefficients_in: &DeviceSlice<'_, T>,
    (output, gradients): (&DeviceSliceMut<'_, T>, bool),
) -> Result<usize, KernelError> {
    let TargetInputs {
        view,
        first_leaf,
        boxes,
        leaves,
        target_offsets,
        target_input,
        ..
    } = *inputs;
    for owner in [
        view.row_offsets().device(),
        view.entries().device(),
        boxes.buffer().as_slice().device(),
        leaves.buffer().as_slice().device(),
        target_offsets.buffer().as_slice().device(),
        target_input.device(),
        coefficients_in.device(),
        output.device(),
    ] {
        device.check_owner(owner)?;
    }
    assert!(p <= MAX_DEGREE, "{what}: p = {p} exceeds {MAX_DEGREE}");
    let nrows = view.nrows();
    assert!(
        first_leaf + nrows <= target_offsets.nleaves() && first_leaf + nrows <= leaves.len(),
        "{what}: rows {first_leaf}..{} for {} target leaves and {} leaf coordinates",
        first_leaf + nrows,
        target_offsets.nleaves(),
        leaves.len()
    );
    assert!(
        u32::try_from(first_leaf + nrows).is_ok(),
        "{what}: {} rows exceed the 32-bit index range",
        first_leaf + nrows
    );
    assert_eq!(
        target_input.len(),
        3 * target_offsets.total(),
        "{what}: the target input holds 3 values per point"
    );
    let per_point = if gradients { 4 } else { 1 };
    assert_eq!(
        output.len(),
        per_point * target_offsets.total(),
        "{what}: the target output holds {per_point} values per point"
    );
    if !view.is_empty() {
        assert!(
            entry_level < boxes.nlevels(),
            "{what}: entries on level {entry_level} of {} levels",
            boxes.nlevels()
        );
        let nboxes = boxes.len(entry_level);
        assert!(
            view.entries().bound() <= nboxes as u64,
            "{what}: entries up to {} for {nboxes} boxes on level {entry_level}",
            view.entries().bound().saturating_sub(1)
        );
        assert_eq!(
            coefficients_in.len(),
            coefficients(p) * nboxes,
            "{what}: the coefficients of level {entry_level}, (p + 1)^2 per box"
        );
    }
    Ok(nrows)
}

/// The units of a CPU-layout launch over `nrows` rows: the device's cap, at most one
/// unit per row.
fn cpu_units(device: &Device, nrows: usize) -> u32 {
    (device.units_cap() as usize)
        .min(device.info().max_units_per_cube.max(1) as usize)
        .min(nrows)
        .max(1) as u32
}

/// P2M or P2L (`irregular_basis`) of one level.
fn expand<T: DeviceFloat>(
    device: &mut Device,
    (layout, p, irregular_basis): (LeafLayout, usize, bool),
    inputs: &SourceInputs<'_, T>,
    out: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    let what = if irregular_basis { "p2l" } else { "p2m" };
    let nrows = check_sources(device, what, p, inputs, &out)?;
    layout.check(device.info(), p, T::FLOAT)?;
    let view = inputs.view;
    if nrows == 0 || view.is_empty() {
        return Ok(());
    }
    let info = device.info().clone();
    let (ro, ro_len) = view.row_offsets().binding();
    let (en, en_len) = view.entries().binding();
    let (bx, bx_len) = inputs.boxes.buffer().as_slice().binding();
    let (lv, lv_len) = inputs.leaves.buffer().as_slice().binding();
    let (so, so_len) = inputs.source_offsets.buffer().as_slice().binding();
    let (sh, sh_len) = inputs.sources.binding();
    let (oh, oh_len) = out.binding();
    let scalars = (
        inputs.boxes.offset(inputs.level) as u32,
        inputs.level as u32,
        inputs.sources.offset() as u32,
        out.offset() as u32,
        nrows as u32,
    );
    let client = device.client().clone();
    match layout {
        LeafLayout::Cube { units, tile } => {
            let (cubes_x, cubes_y) = cube_grid(&info, nrows);
            // SAFETY: every handle is a whole buffer with its element count, as
            // `from_raw_parts` requires. The kernel reads row offsets r and r + 1 of rows
            // r < nrows (the view has nrows + 1, validated CSR) and their entries, each a
            // leaf below the bounds checked above: the leaf coordinates at 4 j + c and the
            // source offsets at j and j + 1; the box coordinates of row r of the level at
            // 3 (offset + r) + k, within the level's boxes (nrows = its box count). Offsets
            // are validated CSR (`PointOffsets`), so each leaf's chunk lies within the
            // source store, whose length was asserted against the total; the coefficients
            // of row r lie at out.offset() + r (p + 1)² + c < out.offset() + out.len().
            // Shared memory holds tile (p + 1)² and tile values, written by units below
            // tile at their own region; each unit writes only the slots it owns. Rows at
            // or past nrows (a 2-D grid) do nothing.
            unsafe {
                expand_cube_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(ro, ro_len),
                    BufferArg::from_raw_parts(en, en_len),
                    BufferArg::from_raw_parts(bx, bx_len),
                    BufferArg::from_raw_parts(lv, lv_len),
                    BufferArg::from_raw_parts(so, so_len),
                    BufferArg::from_raw_parts(sh, sh_len),
                    BufferArg::from_raw_parts(oh, oh_len),
                    scalars.0,
                    scalars.1,
                    scalars.2,
                    scalars.3,
                    scalars.4,
                    p,
                    units as usize,
                    tile as usize,
                    irregular_basis,
                );
            }
        }
        LeafLayout::Cpu => {
            let units = cpu_units(device, nrows);
            // SAFETY: as for the cube layout: every handle is a whole buffer with its
            // element count; rows, entries, leaves, boxes and offsets are within the bounds
            // checked above, and every chunk within its store. Each unit covers its own
            // contiguous range of rows, so no two units write one coefficient.
            unsafe {
                expand_cpu_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(ro, ro_len),
                    BufferArg::from_raw_parts(en, en_len),
                    BufferArg::from_raw_parts(bx, bx_len),
                    BufferArg::from_raw_parts(lv, lv_len),
                    BufferArg::from_raw_parts(so, so_len),
                    BufferArg::from_raw_parts(sh, sh_len),
                    BufferArg::from_raw_parts(oh, oh_len),
                    scalars.0,
                    scalars.1,
                    scalars.2,
                    scalars.3,
                    scalars.4,
                    p,
                    irregular_basis,
                );
            }
        }
    }
    device.count_launch();
    Ok(())
}

/// L2P or M2P (`irregular_basis`) of one level, the expansions `coefficients_in` of the
/// boxes on `entry_level`.
fn evaluate<T: DeviceFloat>(
    device: &mut Device,
    (layout, p, irregular_basis, gradients): (LeafLayout, usize, bool, bool),
    inputs: &TargetInputs<'_, T>,
    coefficients_in: DeviceSlice<'_, T>,
    target_output: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    let (what, entry_level) = if irregular_basis {
        ("m2p", inputs.level + 1)
    } else {
        ("l2p", inputs.level)
    };
    let nrows = check_targets(
        device,
        what,
        (p, entry_level),
        inputs,
        &coefficients_in,
        (&target_output, gradients),
    )?;
    layout.check(device.info(), p, T::FLOAT)?;
    let view = inputs.view;
    if nrows == 0 || view.is_empty() {
        return Ok(());
    }
    let info = device.info().clone();
    let (ro, ro_len) = view.row_offsets().binding();
    let (en, en_len) = view.entries().binding();
    let (bx, bx_len) = inputs.boxes.buffer().as_slice().binding();
    let (lv, lv_len) = inputs.leaves.buffer().as_slice().binding();
    let (to, to_len) = inputs.target_offsets.buffer().as_slice().binding();
    let (ih, ih_len) = inputs.target_input.binding();
    let (ch, ch_len) = coefficients_in.binding();
    let (oh, oh_len) = target_output.binding();
    let scalars = (
        inputs.boxes.offset(entry_level) as u32,
        entry_level as u32,
        coefficients_in.offset() as u32,
        inputs.target_input.offset() as u32,
        target_output.offset() as u32,
        inputs.first_leaf as u32,
        nrows as u32,
    );
    let client = device.client().clone();
    match layout {
        LeafLayout::Cube { units, .. } => {
            let (cubes_x, cubes_y) = cube_grid(&info, nrows);
            // SAFETY: every handle is a whole buffer with its element count. The kernel
            // reads row offsets r and r + 1 of rows r < nrows and their entries, each a box
            // on the entry level below the bound checked above: its coefficients at
            // coefficients.offset() + s (p + 1)² + c, within the slice of the level's
            // boxes, and its coordinates at 3 (offset + s) + k. The target offsets at
            // first_leaf + r + 1 ≤ nleaves and the leaf coordinates at 4 (first_leaf + r)
            // + k lie within the bounds checked above; offsets are validated CSR, so each
            // leaf's chunk lies within the target input and output, whose lengths were
            // asserted against the totals. Shared memory holds (p + 1)² values, each
            // written by one unit between barriers; each unit writes only its own targets'
            // outputs. Rows at or past nrows (a 2-D grid) do nothing.
            unsafe {
                evaluate_cube_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(ro, ro_len),
                    BufferArg::from_raw_parts(en, en_len),
                    BufferArg::from_raw_parts(bx, bx_len),
                    BufferArg::from_raw_parts(lv, lv_len),
                    BufferArg::from_raw_parts(to, to_len),
                    BufferArg::from_raw_parts(ih, ih_len),
                    BufferArg::from_raw_parts(ch, ch_len),
                    BufferArg::from_raw_parts(oh, oh_len),
                    scalars.0,
                    scalars.1,
                    scalars.2,
                    scalars.3,
                    scalars.4,
                    scalars.5,
                    scalars.6,
                    p,
                    units as usize,
                    irregular_basis,
                    gradients,
                );
            }
        }
        LeafLayout::Cpu => {
            let units = cpu_units(device, nrows);
            // SAFETY: as for the cube layout: every handle is a whole buffer with its
            // element count; rows, entries, boxes, leaves and offsets are within the bounds
            // checked above, and every chunk within its store. Each unit covers its own
            // contiguous range of target leaves, so no two units write one output value.
            unsafe {
                evaluate_cpu_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(ro, ro_len),
                    BufferArg::from_raw_parts(en, en_len),
                    BufferArg::from_raw_parts(bx, bx_len),
                    BufferArg::from_raw_parts(lv, lv_len),
                    BufferArg::from_raw_parts(to, to_len),
                    BufferArg::from_raw_parts(ih, ih_len),
                    BufferArg::from_raw_parts(ch, ch_len),
                    BufferArg::from_raw_parts(oh, oh_len),
                    scalars.0,
                    scalars.1,
                    scalars.2,
                    scalars.3,
                    scalars.4,
                    scalars.5,
                    scalars.6,
                    p,
                    irregular_basis,
                    gradients,
                );
            }
        }
    }
    device.count_launch();
    Ok(())
}

/// P2M of one level: adds the multipole of every box's own leaf, from its sources at the
/// unit frame, into `multipoles`, the multipoles of the level ((p + 1)² per box, box t at
/// t (p + 1)²), in `layout` (module documentation; `nd_fmm_ref::leaf::p2m`).
///
/// A call without entries launches nothing. One launch otherwise, which allocates
/// nothing.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the device cannot run `layout` at p in `T`
/// ([`LeafLayout::check`]).
///
/// # Panics
///
/// If p exceeds [`MAX_DEGREE`], the level does not exist or has another number of boxes
/// than the view rows, an entry addresses no leaf of the source offsets or the
/// coordinates, or a store does not hold its values: 4 per source point, (p + 1)² per box.
pub fn p2m<T: DeviceFloat>(
    device: &mut Device,
    layout: LeafLayout,
    p: usize,
    inputs: &SourceInputs<'_, T>,
    multipoles: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    expand(device, (layout, p, false), inputs, multipoles)
}

/// P2L of one level: adds the local expansion of the sources of every box's X list, at
/// the frames (ĉ(t|s), r̂(t|s)), into `locals`, the locals of the level ((p + 1)² per box),
/// in `layout` (module documentation; `nd_fmm_ref::leaf::p2l`).
///
/// # Errors
///
/// As [`p2m`].
///
/// # Panics
///
/// As [`p2m`].
pub fn p2l<T: DeviceFloat>(
    device: &mut Device,
    layout: LeafLayout,
    p: usize,
    inputs: &SourceInputs<'_, T>,
    locals: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    expand(device, (layout, p, true), inputs, locals)
}

/// L2P of one level: adds the potential (and with `gradients` the gradient) of each
/// target leaf's own box's local expansion at its targets, at the unit frame, into
/// `target_output` (per leaf of n points, n values, or with gradients the n potentials and
/// then n triples, CONVENTIONS §3.13). `locals` holds the locals of `inputs.level`
/// ((p + 1)² per box). Module documentation; `nd_fmm_ref::leaf::l2p`.
///
/// A call without entries launches nothing. One launch otherwise, which allocates
/// nothing.
///
/// # Errors
///
/// As [`p2m`].
///
/// # Panics
///
/// If p exceeds [`MAX_DEGREE`], the rows exceed the target leaves or the leaf
/// coordinates, an entry addresses no box of the level, or a store does not hold its
/// values: 3 per target point in the input, 1 (4 with gradients) per target point in the
/// output, (p + 1)² per box of the level.
pub fn l2p<T: DeviceFloat>(
    device: &mut Device,
    layout: LeafLayout,
    p: usize,
    gradients: bool,
    inputs: &TargetInputs<'_, T>,
    locals: DeviceSlice<'_, T>,
    target_output: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    evaluate(
        device,
        (layout, p, false, gradients),
        inputs,
        locals,
        target_output,
    )
}

/// M2P of one level: adds the potential (and with `gradients` the gradient) of the
/// multipoles of each target leaf's W list at its targets, at the frames (ĉ(s|t), r̂(s|t)),
/// into `target_output`, in W-list order. `multipoles` holds the multipoles of
/// `inputs.level + 1` ((p + 1)² per box); on the deepest level, whose rows are empty, it
/// may be empty. Module documentation; `nd_fmm_ref::leaf::m2p`.
///
/// # Errors
///
/// As [`p2m`].
///
/// # Panics
///
/// As [`l2p`], the entries being boxes of `inputs.level + 1`.
pub fn m2p<T: DeviceFloat>(
    device: &mut Device,
    layout: LeafLayout,
    p: usize,
    gradients: bool,
    inputs: &TargetInputs<'_, T>,
    multipoles: DeviceSlice<'_, T>,
    target_output: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    evaluate(
        device,
        (layout, p, true, gradients),
        inputs,
        multipoles,
        target_output,
    )
}

/// The frames of a far view on the device, for tests against `geometry::relative_frame`
/// of `nd-fmm-exec`: four values per entry in entry order.
fn far_frames<T: DeviceFloat>(
    device: &mut Device,
    view: &IndexView,
    (box_level, first_leaf, w): (usize, usize, bool),
    boxes: &BoxCoordinates,
    leaves: &LeafCoordinates,
    frames: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    for owner in [
        view.row_offsets().device(),
        view.entries().device(),
        boxes.buffer().as_slice().device(),
        leaves.buffer().as_slice().device(),
        frames.device(),
    ] {
        device.check_owner(owner)?;
    }
    assert_eq!(frames.len(), 4 * view.len(), "four values per entry");
    let nrows = view.nrows();
    if nrows == 0 || view.is_empty() {
        return Ok(());
    }
    assert!(
        box_level < boxes.nlevels(),
        "boxes on level {box_level} of {}",
        boxes.nlevels()
    );
    if w {
        assert!(
            first_leaf + nrows <= leaves.len()
                && view.entries().bound() <= boxes.len(box_level) as u64,
            "w_frames: rows or entries out of range"
        );
    } else {
        assert!(
            nrows <= boxes.len(box_level) && view.entries().bound() <= leaves.len() as u64,
            "x_frames: rows or entries out of range"
        );
    }
    let grid = device.elementwise_grid(nrows);
    let (ro, ro_len) = view.row_offsets().binding();
    let (en, en_len) = view.entries().binding();
    let (bx, bx_len) = boxes.buffer().as_slice().binding();
    let (lv, lv_len) = leaves.buffer().as_slice().binding();
    let (fh, fh_len) = frames.binding();
    // SAFETY: every handle is a whole buffer with its element count. The kernel reads the
    // row offsets and entries of rows below nrows, the coordinates of boxes and leaves
    // within the bounds asserted above, and writes four values per entry at
    // frames.offset() + 4 e, within `frames` (its length was asserted).
    unsafe {
        far_frames_kernel::launch_unchecked::<T>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(ro, ro_len),
            BufferArg::from_raw_parts(en, en_len),
            BufferArg::from_raw_parts(bx, bx_len),
            BufferArg::from_raw_parts(lv, lv_len),
            BufferArg::from_raw_parts(fh, fh_len),
            frames.offset() as u32,
            boxes.offset(box_level) as u32,
            box_level as u32,
            first_leaf as u32,
            nrows as u32,
            grid.chunk,
            grid.threads(),
            w,
        );
    }
    device.count_launch();
    Ok(())
}

/// Writes the frame of every entry of a P2L call's X view into `frames`, four values per
/// entry in entry order: (ĉ(t|s), r̂(t|s)) of box t of the level (row t) seen from leaf s
/// (the entry), as [`p2l`] forms it. For tests against `geometry::relative_frame` of
/// `nd-fmm-exec` (device-path.md §6.1). One elementwise launch; none without an entry.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// If the rows exceed the level's boxes, an entry addresses no leaf, or `frames` does not
/// hold four values per entry.
pub fn x_frames<T: DeviceFloat>(
    device: &mut Device,
    inputs: &SourceInputs<'_, T>,
    frames: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    far_frames(
        device,
        inputs.view,
        (inputs.level, 0, false),
        inputs.boxes,
        inputs.leaves,
        frames,
    )
}

/// Writes the frame of every entry of an M2P call's W view into `frames`, four values per
/// entry in entry order: (ĉ(s|t), r̂(s|t)) of box s on the level below (the entry) seen
/// from leaf `first_leaf + r` (row r), as [`m2p`] forms it. For tests against
/// `geometry::relative_frame`.
///
/// # Errors
///
/// As [`x_frames`].
///
/// # Panics
///
/// If the rows exceed the leaf coordinates, an entry addresses no box of the level below,
/// or `frames` does not hold four values per entry.
pub fn w_frames<T: DeviceFloat>(
    device: &mut Device,
    inputs: &TargetInputs<'_, T>,
    frames: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    far_frames(
        device,
        inputs.view,
        (inputs.level + 1, inputs.first_leaf, true),
        inputs.boxes,
        inputs.leaves,
        frames,
    )
}

/// The solid harmonics of `basis` of degree ≤ p at `points` (triples), as the leaf
/// kernels evaluate them, for tests against `nd_fmm_math::harmonics`: per point
/// (p + 1)² values in real storage, then with `gradients` the components ∂x, ∂y and ∂z,
/// (p + 1)² values each (the layout of `regular_grad` and `irregular_grad`). One
/// launch, none for no point.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// If p exceeds [`MAX_DEGREE`], `points` is not a whole number of triples, or `out` does
/// not hold (p + 1)² (4 (p + 1)² with gradients) values per point.
pub fn harmonics<T: DeviceFloat>(
    device: &mut Device,
    p: usize,
    basis: Basis,
    gradients: bool,
    points: DeviceSlice<'_, T>,
    out: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    device.check_owner(points.device())?;
    device.check_owner(out.device())?;
    assert!(p <= MAX_DEGREE, "harmonics: p = {p} exceeds {MAX_DEGREE}");
    assert_eq!(points.len() % 3, 0, "harmonics: points are triples");
    let count = points.len() / 3;
    let per_point = coefficients(p) * if gradients { 4 } else { 1 };
    assert_eq!(out.len(), per_point * count, "harmonics: the output length");
    if count == 0 {
        return Ok(());
    }
    let info = device.info();
    let (units, cubes) = if info.backend.is_gpu() {
        let units = 64.min(info.max_units_per_cube.max(1));
        let cubes = count
            .div_ceil(units as usize)
            .min(info.max_cube_count.0.max(1) as usize);
        (units, cubes as u32)
    } else {
        (cpu_units(device, count), 1)
    };
    let (ph, ph_len) = points.binding();
    let (oh, oh_len) = out.binding();
    // SAFETY: both handles are whole buffers with their element counts. Unit i reads the
    // triple at points.offset() + 3 i and writes per_point values at out.offset() +
    // per_point i, for i < count only; both lengths were asserted.
    unsafe {
        harmonics_kernel::launch_unchecked::<T>(
            device.client(),
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(ph, ph_len),
            BufferArg::from_raw_parts(oh, oh_len),
            points.offset() as u32,
            out.offset() as u32,
            count as u32,
            p,
            basis == Basis::Irregular,
            gradients,
        );
    }
    device.count_launch();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CUBECL_VERSION;

    fn info(backend: BackendKind) -> DeviceInfo {
        DeviceInfo {
            backend,
            name: "test".into(),
            compiler: "test".into(),
            cubecl_version: CUBECL_VERSION,
            f32: true,
            f64: true,
            plane_size: (32, 32),
            max_shared_memory: 32 * 1024,
            max_units_per_cube: if backend.is_gpu() { 1024 } else { 16 },
            max_cube_count: (65_535, 65_535, 65_535),
            max_memory: None,
        }
    }

    #[test]
    fn slots_follow_the_real_storage() {
        assert_eq!(slot(0, 0), 0);
        assert_eq!(slot(1, -1), 1);
        assert_eq!(slot(1, 1), 3);
        assert_eq!(slot(20, 20), 440);
        let imaginary_slots: Vec<usize> = (0..9).filter(|&c| imaginary(c)).collect();
        assert_eq!(imaginary_slots, vec![1, 4, 5]);
        assert_eq!(coefficients(20), 441);
    }

    #[test]
    fn defaults_follow_the_backend_and_the_shared_memory() {
        let cpu = info(BackendKind::Cpu);
        assert_eq!(
            LeafLayout::default_for(&cpu, 8, Precision::F64),
            LeafLayout::Cpu
        );
        let metal = info(BackendKind::Metal);
        assert_eq!(
            LeafLayout::default_for(&metal, 8, Precision::F32),
            LeafLayout::Cube {
                units: 64,
                tile: 32
            }
        );
        // p = 20 in f32: 442 values of 4 bytes per point, 18 points in 32 KB.
        assert_eq!(
            LeafLayout::default_for(&metal, 20, Precision::F32),
            LeafLayout::Cube {
                units: 64,
                tile: 18
            }
        );
        for p in [0, 3, 8, 20] {
            for precision in [Precision::F32, Precision::F64] {
                let layout = LeafLayout::default_for(&metal, p, precision);
                assert!(
                    layout.check(&metal, p, precision).is_ok(),
                    "{layout} at {p}"
                );
            }
        }
    }

    #[test]
    fn layouts_are_checked_against_the_device() {
        let metal = info(BackendKind::Metal);
        let check =
            |units, tile, p| LeafLayout::Cube { units, tile }.check(&metal, p, Precision::F32);
        assert!(check(64, 32, 8).is_ok());
        assert!(check(0, 1, 3).is_err());
        assert!(check(4, 0, 3).is_err());
        assert!(check(4, 8, 3).is_err());
        assert!(check(2048, 32, 3).is_err());
        // 32 points of 82 f32 values fit 32 KB; 128 points do not.
        assert!(check(128, 128, 8).is_err());
        assert!(LeafLayout::Cpu.check(&metal, 20, Precision::F64).is_ok());
        assert_eq!(
            LeafLayout::Cube {
                units: 64,
                tile: 32
            }
            .to_string(),
            "cube (64 units, tile 32)"
        );
    }
}
