//! Rotation M2L on the device (Phase 4 T10, C4.6): point and shoot from the M2L family of
//! the rotation tables of Phase 2 (C2.3), one launch per level (docs/design/device-path.md
//! §6.6).
//!
//! # What a level call computes
//!
//! A V view ([`GroupedView`]) holds the pairs (target t, source s, offset d) of one level,
//! in rows by target with the offsets ascending (CONVENTIONS §3.12, the table index of
//! `m2l_offsets`). [`m2l`] adds, for every target t, the M2L of the multipole of each
//! entry (s, d) of its row into its local, in row order (README requirement 4). Each
//! translation is `nd_fmm_tables::rotation::ShiftTables::apply` of the M2L family, step
//! for step (CONVENTIONS §3.11, "Rotation of coefficients", "Coaxial translations"):
//!
//! | Alignment of offset d | Steps, each over the (p + 1)² slots of the scaled coefficients |
//! | --- | --- |
//! | `Rotated { polar, azimuth }` | 1. the z-rotation T(R_z(−φ)): slot 0 of each degree copied, each pair (+m, −m) rotated by Cₘ, Sₘ; 2. the forward y-blocks T_M(R_y(−θ)), per degree n a (2n + 1)² block; 3. the coaxial step of the distance, per order i the (p + 1 − i)² matrix from input degree n to output degree j; 4. the backward y-blocks T_L(R_y(θ)); 5. the z-rotation back T(R_z(φ)), added into the local |
//! | `Up`, `Down` (d on the z axis) | the coaxial step alone, added into the local term by term, with the parity (−1)ⁿ⁺ʲ for `Down` |
//!
//! The tables are those of `RotationTables::tables(Operator::M2l)` in their own storage
//! ([`RotationArrays`]; nothing is converted but the per-offset geometry, which becomes four
//! `u32`), uploaded once ([`RotationTables`]): with B = (p + 1)(2p + 1)(2p + 3)/3 and
//! C = (p + 1)(p + 2)(2p + 3)/6, 2 B per polar angle off the axis (47), 2p per azimuth
//! (32), C per distance (15) and 4 `u32` per offset (316):
//! 94 B + 64 p + 15 C values and 5,056 bytes ([`RotationTables::bytes`]):
//!
//! | p | 3 | 8 | 12 | 16 | 18 | 20 |
//! | --- | ---: | ---: | ---: | ---: | ---: | ---: |
//! | f32, MB | 0.04 | 0.39 | 1.16 | 2.58 | 3.59 | 4.85 |
//! | f64, MB | 0.07 | 0.77 | 2.31 | 5.15 | 7.18 | 9.69 |
//!
//! against 316 (p + 1)⁴ values for the dense tables (8.3 MB at p = 8 in f32, 492 MB at
//! p = 20 in f64).
//!
//! # Arithmetic and order
//!
//! Every output value of a step is one owner's: unit k of the cube (or the unit of the CPU
//! layout) computes slot k of the step's output from the step's input, as one accumulator
//! from zero updated by an explicit `fma` per term in the order of `ShiftTables::apply`
//! (columns ascending in a y-block row, input degrees ascending in the coaxial step); the
//! z-rotation forms `fma(C, u, −(S v))` for slot +m and `fma(S, u, C v)` for slot −m (S
//! negated for the way back, which is exact). The last step adds into the owner's
//! accumulator, which started from the local as L2L left it, and the accumulator is stored
//! once per row (device-path.md §6.1). On the axis the coaxial terms are added into the
//! accumulator directly, `acc = fma(±a, x, acc)`, as the host adds them into the local.
//!
//! The explicit `fma`s pin the rounding on every backend (spikes/device-arith/REPORT.md,
//! "Recommendation", rule 4): the kernel equals a host loop of `mul_add` in this order bit
//! for bit (tested). It does not equal `RotationTables::m2l` bit for bit: the host rounds
//! every product before its addition, and cubecl-opt's `InstCombinePass` fuses every lone
//! a · b ± c on every backend with no switch (CONVENTIONS §3.13, "Device kernels"), so no
//! formulation of a multiply–add chain keeps the host's roundings (T3's rule 6:
//! rotation is tested with tolerances; device-path.md §9.2). The two agree to rounding.
//!
//! No atomics: a row's locals are written by its own cube (or unit) only. Rows meet their
//! entries in row order, so the result does not depend on the layout or the launch shape,
//! and repeated calls give the same bits.
//!
//! # Layouts ([`RotationLayout`])
//!
//! | Layout | Default on | Mapping |
//! | --- | --- | --- |
//! | [`Cube`](RotationLayout::Cube) | Metal, CUDA | one cube of U units per row with an entry; unit u owns slots u, u + U, … (by default U is (p + 1)² rounded up to the plane size, 96 at p = 8 and 448 at p = 20, so one slot each); two working vectors of (p + 1)² values in shared memory, each step followed by `sync_cube` |
//! | [`Cpu`](RotationLayout::Cpu) | the CPU runtime | one cube, one unit per core (at most [`Device::units_cap`]), each a contiguous range of the rows with an entry; the working vectors and the row's accumulators in local arrays; no shared memory, no barrier |
//!
//! Units without a slot idle in every step but reach every barrier. Both layouts form
//! every value by the same operations, so they give the same bits; the cube layout also
//! runs on the CPU runtime (correctness only: a kernel with shared memory gets a worker per
//! unit there, device-path.md F18).
//!
//! # Launch
//!
//! [`RotationTables::upload`] uploads the tables once (five uploads). [`RotationPlan::new`]
//! validates a level's V view on the host and uploads its rows with an entry (one upload).
//! [`m2l`] runs a level call: one launch, nothing for a view without pairs, no allocation.

use std::fmt;

use cubecl::prelude::*;

use crate::buffer::{DeviceBuffer, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexBuffer};
use crate::device::{BackendKind, Device, DeviceInfo, Precision};
use crate::error::KernelError;
use crate::leaf::{MAX_DEGREE, coefficients, unit_rows};
use crate::p2p::cube_grid;
use crate::view::{GroupedArrays, GroupedView, row_to_batch};

/// The alignment code of an offset along +z (`Alignment::Up`): the coaxial step alone.
const UP: u32 = 0;

/// The alignment code of an offset along −z (`Alignment::Down`): the coaxial step alone,
/// with the parity (−1)ⁿ⁺ʲ.
const DOWN: u32 = 1;

/// The alignment code of every other offset (`Alignment::Rotated`).
const ROTATED: u32 = 2;

/// The values of the y-rotation blocks of one polar angle, Σₙ (2n + 1)² =
/// (p + 1)(2p + 1)(2p + 3)/3 (`nd_fmm_math::rotation::blocks_len`).
pub const fn blocks_len(p: usize) -> usize {
    (p + 1) * (2 * p + 1) * (2 * p + 3) / 3
}

/// The coaxial factors of one distance, Σᵢ (p + 1 − i)² = (p + 1)(p + 2)(2p + 3)/6.
pub const fn coaxial_len(p: usize) -> usize {
    (p + 1) * (p + 2) * (2 * p + 3) / 6
}

/// How the shift of one offset is brought onto the +z axis: `nd_fmm_tables::rotation::
/// Alignment`, with its indices as `u32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Alignment {
    /// The shift lies along +z: the coaxial step alone.
    Up,
    /// The shift lies along −z: the coaxial step alone, with the parity (−1)ⁿ⁺ʲ.
    Down,
    /// Rotation by the polar angle `polar` and the azimuth `azimuth`.
    Rotated {
        /// The index of the polar angle's blocks.
        polar: u32,
        /// The index of the azimuth's factors.
        azimuth: u32,
    },
}

/// The geometry of one offset: `nd_fmm_tables::rotation::Shift`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Shift {
    /// The rotation, if any.
    pub alignment: Alignment,
    /// The index of the distance's coaxial factors.
    pub distance: u32,
}

/// The host arrays of the M2L family of the rotation tables, in the storage of
/// `nd_fmm_tables::rotation::ShiftTables` (module documentation): `RotationTables::
/// tables(Operator::M2l)` with each accessor's slices concatenated in index order.
#[derive(Clone, Copy, Debug)]
pub struct RotationArrays<'a, T> {
    /// The degree p.
    pub p: usize,
    /// The forward y-rotation blocks T_M(R_y(−θ)) of each polar angle off the axis,
    /// [`blocks_len`]`(p)` values each (`forward_blocks`).
    pub forward: &'a [T],
    /// The backward y-rotation blocks T_L(R_y(θ)), in the layout of `forward`
    /// (`backward_blocks`).
    pub backward: &'a [T],
    /// C₁, S₁, …, Cₚ, Sₚ of each azimuth, 2p values each (`azimuth_factors`).
    pub azimuth: &'a [T],
    /// The coaxial factors of each distance, [`coaxial_len`]`(p)` values each
    /// (`coaxial_factors`).
    pub coaxial: &'a [T],
    /// The geometry of each offset, in table order (`shift`).
    pub shifts: &'a [Shift],
}

/// The M2L family of the rotation tables on the device, uploaded once (module
/// documentation): four buffers of values in the storage of `ShiftTables`, and the
/// geometry of each offset as four `u32` (alignment, polar angle, azimuth, distance).
#[derive(Debug)]
pub struct RotationTables<T: DeviceFloat> {
    p: usize,
    forward: DeviceBuffer<T>,
    backward: DeviceBuffer<T>,
    azimuth: DeviceBuffer<T>,
    coaxial: DeviceBuffer<T>,
    shifts: IndexBuffer,
    count: usize,
}

impl<T: DeviceFloat> RotationTables<T> {
    /// Validates the arrays and uploads them (five uploads).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If p exceeds [`MAX_DEGREE`], an array is not a whole number of its pieces, the
    /// forward and backward blocks differ in count, or an offset's polar angle, azimuth or
    /// distance is not below the count of its array.
    pub fn upload(
        device: &mut Device,
        arrays: &RotationArrays<'_, T>,
    ) -> Result<Self, KernelError> {
        let p = arrays.p;
        assert!(
            p <= MAX_DEGREE,
            "RotationTables: p = {p} exceeds {MAX_DEGREE}"
        );
        let whole = |what: &str, len: usize, piece: usize| -> usize {
            assert!(
                len.is_multiple_of(piece),
                "RotationTables: {len} {what} values are not whole pieces of {piece}"
            );
            len / piece
        };
        let polars = whole("forward", arrays.forward.len(), blocks_len(p));
        assert_eq!(
            whole("backward", arrays.backward.len(), blocks_len(p)),
            polars,
            "RotationTables: forward and backward blocks differ in count"
        );
        let distances = whole("coaxial", arrays.coaxial.len(), coaxial_len(p));
        // At p = 0 there is no factor, and no step reads one.
        let azimuths = if p == 0 {
            assert!(
                arrays.azimuth.is_empty(),
                "RotationTables: azimuth factors at p = 0"
            );
            usize::MAX
        } else {
            whole("azimuth", arrays.azimuth.len(), 2 * p)
        };
        let mut shifts = Vec::with_capacity(4 * arrays.shifts.len());
        for (d, shift) in arrays.shifts.iter().enumerate() {
            assert!(
                (shift.distance as usize) < distances,
                "RotationTables: offset {d} has distance {} of {distances}",
                shift.distance
            );
            shifts.extend(match shift.alignment {
                Alignment::Up => [UP, 0, 0, shift.distance],
                Alignment::Down => [DOWN, 0, 0, shift.distance],
                Alignment::Rotated { polar, azimuth } => {
                    assert!(
                        (polar as usize) < polars && (azimuth as usize) < azimuths,
                        "RotationTables: offset {d} has polar angle {polar} of {polars} and \
                         azimuth {azimuth} of {azimuths}"
                    );
                    [ROTATED, polar, azimuth, shift.distance]
                }
            });
        }
        Ok(Self {
            p,
            forward: device.upload(arrays.forward)?,
            backward: device.upload(arrays.backward)?,
            azimuth: device.upload(arrays.azimuth)?,
            coaxial: device.upload(arrays.coaxial)?,
            shifts: device.upload_indices(&shifts)?,
            count: arrays.shifts.len(),
        })
    }

    /// The bytes [`upload`](Self::upload) allocates for `arrays`.
    pub fn bytes(arrays: &RotationArrays<'_, T>) -> u64 {
        Device::buffer_bytes::<T>(arrays.forward.len())
            + Device::buffer_bytes::<T>(arrays.backward.len())
            + Device::buffer_bytes::<T>(arrays.azimuth.len())
            + Device::buffer_bytes::<T>(arrays.coaxial.len())
            + Device::buffer_bytes::<u32>(4 * arrays.shifts.len())
    }

    /// The degree p.
    pub fn p(&self) -> usize {
        self.p
    }

    /// The number of offsets.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The number of values stored (the geometry not counted), as
    /// `ShiftTables::storage_len`.
    pub fn storage_len(&self) -> usize {
        self.forward.len() + self.backward.len() + self.azimuth.len() + self.coaxial.len()
    }
}

/// The layout of a rotation M2L launch (module documentation, "Layouts").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RotationLayout {
    /// One cube of `units` units per row with an entry, unit u the owner of slots u,
    /// u + `units`, …; the working vectors in shared memory.
    Cube {
        /// Units per cube, at least 1.
        units: u32,
    },
    /// One cube of one unit per core, each a contiguous range of rows, no shared memory.
    Cpu,
}

impl RotationLayout {
    /// The default layout of a device at degree p: [`Cpu`](Self::Cpu) on the CPU runtime,
    /// otherwise [`Cube`](Self::Cube) with (p + 1)² units rounded up to the plane size
    /// (device-path.md §6.6).
    pub fn default_for(info: &DeviceInfo, p: usize) -> Self {
        match info.backend {
            BackendKind::Cpu => Self::Cpu,
            BackendKind::Metal | BackendKind::Cuda => {
                let plane = info.plane_size.0.max(1) as usize;
                Self::Cube {
                    units: coefficients(p).next_multiple_of(plane) as u32,
                }
            }
        }
    }

    /// Checks that the device can run the layout at degree p in `precision`.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedLayout`] for a cube of no unit, more units than the device
    /// allows per cube, or more shared memory than it has (two vectors of (p + 1)²
    /// values).
    pub fn check(
        self,
        info: &DeviceInfo,
        p: usize,
        precision: Precision,
    ) -> Result<(), KernelError> {
        let Self::Cube { units } = self else {
            return Ok(());
        };
        let refuse = |reason: String| {
            Err(KernelError::UnsupportedLayout {
                layout: self.to_string(),
                reason,
            })
        };
        let n = coefficients(p);
        if units == 0 {
            return refuse("no unit".into());
        }
        if units > info.max_units_per_cube {
            return refuse(format!(
                "{units} units per cube, the device allows {}",
                info.max_units_per_cube
            ));
        }
        let bytes = match precision {
            Precision::F32 => 4,
            Precision::F64 => 8,
        };
        let shared = 2 * n * bytes;
        if shared > info.max_shared_memory {
            return refuse(format!(
                "{shared} bytes of shared memory at p = {p}, the device has {}",
                info.max_shared_memory
            ));
        }
        Ok(())
    }
}

impl fmt::Display for RotationLayout {
    /// `cube (96 units)` or `cpu (one unit per core)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube { units } => write!(f, "cube ({units} units)"),
            Self::Cpu => f.write_str("cpu (one unit per core)"),
        }
    }
}

/// The schedule of one level's rotation M2L, built once: the rows of the V view with an
/// entry, one cube (or one unit's share) each.
#[derive(Debug)]
pub struct RotationPlan {
    rows: IndexBuffer,
    nrows: usize,
    len: usize,
    sources_bound: usize,
}

impl RotationPlan {
    /// Validates the V view with the host `arrays` (`sources_bound` the columns of the
    /// input, as [`GroupedView::upload`]) and uploads the rows with an entry (one upload).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If the arrays are not a valid grouped view ([`row_to_batch`]).
    pub fn new<G: Copy + Into<u32>>(
        device: &mut Device,
        arrays: &GroupedArrays<'_, G>,
        sources_bound: usize,
    ) -> Result<Self, KernelError> {
        row_to_batch(arrays, sources_bound);
        let rows = Self::rows(arrays.row_offsets);
        Ok(Self {
            rows: device.upload_indices(&rows)?,
            nrows: arrays.row_offsets.len() - 1,
            len: arrays.sources.len(),
            sources_bound,
        })
    }

    /// The rows with an entry of the CSR offsets `row_offsets`.
    fn rows(row_offsets: &[u32]) -> Vec<u32> {
        row_offsets
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[1] > w[0])
            .map(|(t, _)| t as u32)
            .collect()
    }

    /// The bytes [`new`](Self::new) uploads for a view with these row offsets.
    pub fn bytes(row_offsets: &[u32]) -> u64 {
        Device::buffer_bytes::<u32>(Self::rows(row_offsets).len())
    }

    /// The rows with an entry: the cubes of the cube layout.
    pub fn active_rows(&self) -> usize {
        self.rows.len()
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the view has no pair: a level call launches nothing.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// One level call of rotation M2L (module documentation): adds, for every target t of
/// `view`, the M2L of `multipoles[:, s]` across offset d of each entry (s, d) of its row
/// into `locals[:, t]`, in row order, with the tables `tables`, in `layout`. One launch;
/// nothing for a view without pairs. Allocates nothing.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the device cannot run `layout` at the tables'
/// degree in `T` ([`RotationLayout::check`]).
///
/// # Panics
///
/// If `plan` was not built for `view` (rows, pairs or source bound differ), the view has
/// more offsets than the tables, or `multipoles` and `locals` do not hold (p + 1)² values
/// per source and per row.
pub fn m2l<T: DeviceFloat>(
    device: &mut Device,
    layout: RotationLayout,
    plan: &RotationPlan,
    view: &GroupedView,
    tables: &RotationTables<T>,
    multipoles: DeviceSlice<'_, T>,
    locals: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    for owner in [
        plan.rows.as_slice().device(),
        view.row_offsets().device(),
        view.sources().device(),
        view.groups().device(),
        tables.forward.as_slice().device(),
        tables.backward.as_slice().device(),
        tables.azimuth.as_slice().device(),
        tables.coaxial.as_slice().device(),
        tables.shifts.as_slice().device(),
        multipoles.device(),
        locals.device(),
    ] {
        device.check_owner(owner)?;
    }
    let p = tables.p;
    let n = coefficients(p);
    assert!(
        view.nrows() == plan.nrows
            && view.len() == plan.len
            && view.sources_bound() == plan.sources_bound,
        "rotation::m2l: the plan was built for another view"
    );
    assert!(
        view.ngroups() <= tables.count,
        "rotation::m2l: {} offsets for tables of {}",
        view.ngroups(),
        tables.count
    );
    assert_eq!(
        multipoles.len(),
        plan.sources_bound * n,
        "rotation::m2l: the multipoles hold {} values, not {} columns of {n}",
        multipoles.len(),
        plan.sources_bound
    );
    assert_eq!(
        locals.len(),
        plan.nrows * n,
        "rotation::m2l: the locals hold {} values, not {} columns of {n}",
        locals.len(),
        plan.nrows
    );
    layout.check(device.info(), p, T::FLOAT)?;
    if plan.is_empty() {
        return Ok(());
    }
    let info = device.info().clone();
    let active = plan.rows.len();
    let (rh, rl) = plan.rows.as_slice().binding();
    let (oh, ol) = view.row_offsets().binding();
    let (sh, sl) = view.sources().binding();
    let (gh, gl) = view.groups().binding();
    let (dh, dl) = tables.shifts.as_slice().binding();
    let (fh, fl) = tables.forward.as_slice().binding();
    let (bh, bl) = tables.backward.as_slice().binding();
    let (ah, al) = tables.azimuth.as_slice().binding();
    let (ch, cl) = tables.coaxial.as_slice().binding();
    let (ih, il) = multipoles.binding();
    let (lh, ll) = locals.binding();
    let scalars = (
        active as u32,
        multipoles.offset() as u32,
        locals.offset() as u32,
    );
    let client = device.client().clone();
    match layout {
        RotationLayout::Cube { units } => {
            let (cubes_x, cubes_y) = cube_grid(&info, active);
            // SAFETY: every handle is a whole buffer with its element count. The kernel
            // reads the rows list at r < active (its length), each a row t < nrows of the
            // view (built from its row offsets), the row offsets t and t + 1 (a validated
            // CSR of nrows + 1 offsets) and the sources and groups of the row's entries.
            // Each group d < ngroups ≤ count (asserted) addresses the four geometry values
            // 4 d + 0..4 < 4 count; their indices were checked below the counts of the
            // forward and backward blocks, azimuth factors and coaxial factors at upload,
            // and the kernel reads within one piece of each (p is the tables' degree; at
            // p = 0 no azimuth factor is read). Sources are below sources_bound, so the
            // multipoles are read at their offset plus s (p + 1)² + k < multipoles.len()
            // (asserted), and each row's locals at their offset plus t (p + 1)² + k
            // < locals.len() (asserted). Unit u owns the slots k = u, u + units, … below
            // (p + 1)² of its row, the only unit that writes them; shared memory holds two
            // vectors of (p + 1)² values, each slot written by its owner between barriers,
            // and every unit of a cube reaches every barrier (the row and its entries are
            // the cube's). Cubes past `active` (a 2-D grid) do nothing.
            unsafe {
                rotation_cube_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(rh, rl),
                    BufferArg::from_raw_parts(oh, ol),
                    BufferArg::from_raw_parts(sh, sl),
                    BufferArg::from_raw_parts(gh, gl),
                    BufferArg::from_raw_parts(dh, dl),
                    BufferArg::from_raw_parts(fh, fl),
                    BufferArg::from_raw_parts(bh, bl),
                    BufferArg::from_raw_parts(ah, al),
                    BufferArg::from_raw_parts(ch, cl),
                    BufferArg::from_raw_parts(ih, il),
                    BufferArg::from_raw_parts(lh, ll),
                    scalars.0,
                    scalars.1,
                    scalars.2,
                    p,
                    units as usize,
                );
            }
        }
        RotationLayout::Cpu => {
            let units = (device.units_cap() as usize)
                .min(info.max_units_per_cube.max(1) as usize)
                .min(active)
                .max(1) as u32;
            // SAFETY: as for the cube layout: every buffer is read and written within the
            // bounds checked above. Each unit covers its own contiguous range of the rows
            // list, whose rows are distinct, so no two units write one local; the working
            // vectors are the unit's local arrays.
            unsafe {
                rotation_cpu_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(rh, rl),
                    BufferArg::from_raw_parts(oh, ol),
                    BufferArg::from_raw_parts(sh, sl),
                    BufferArg::from_raw_parts(gh, gl),
                    BufferArg::from_raw_parts(dh, dl),
                    BufferArg::from_raw_parts(fh, fl),
                    BufferArg::from_raw_parts(bh, bl),
                    BufferArg::from_raw_parts(ah, al),
                    BufferArg::from_raw_parts(ch, cl),
                    BufferArg::from_raw_parts(ih, il),
                    BufferArg::from_raw_parts(lh, ll),
                    scalars.0,
                    scalars.1,
                    scalars.2,
                    p,
                );
            }
        }
    }
    device.count_launch();
    Ok(())
}

// --- One output slot of each step ------------------------------------------------------

/// The degree n of slot k (n² ≤ k < (n + 1)²).
#[cube]
fn degree_of(k: usize) -> usize {
    let mut n = 0usize;
    while (n + 1) * (n + 1) <= k {
        n += 1;
    }
    n
}

/// Slot (n, m) of the z-rotation T(R_z(−φ)) of the vector at `x[x0..]`, or with
/// `inverse` of T(R_z(φ)), the factors C₁, S₁, … at `factors[f0..]`: x at slot 0;
/// fma(C, u, −(S v)) at +m and fma(S, u, C v) at −m, with u and v the values at +m and
/// −m and S negated for the inverse (`ShiftTables::rotate_z`).
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a step's arguments are its buffers and the slot"
)]
fn rotate_z<F: Float, L: List<F> + ?Sized>(
    x: &L,
    x0: usize,
    factors: &[F],
    f0: usize,
    n: usize,
    m: i32,
    inverse: bool,
) -> F {
    let centre = x0 + n * n + n;
    let mut value = x[centre];
    if m != 0 {
        let mut order = m;
        if m < 0 {
            order = -m;
        }
        let mm = order as usize;
        let c = factors[f0 + 2 * (mm - 1)];
        let mut s = factors[f0 + 2 * (mm - 1) + 1];
        if inverse {
            s = -s;
        }
        let u = x[centre + mm];
        let v = x[centre - mm];
        if m > 0 {
            value = fma(c, u, -(s * v));
        } else {
            value = fma(s, u, c * v);
        }
    }
    value
}

/// Slot (n, m) of the y-blocks at `blocks[b0..]` (row-major per degree, in the layout of
/// `nd_fmm_math::rotation::blocks`) applied to the vector at `x[x0..]`: row m + n of block
/// n against the degree's 2n + 1 values, one accumulator from zero, an fma per column in
/// increasing order (`ShiftTables`' `apply_blocks`).
#[cube]
fn y_blocks<F: Float, L: List<F> + ?Sized>(
    x: &L,
    x0: usize,
    blocks: &[F],
    b0: usize,
    n: usize,
    m: i32,
) -> F {
    let width = 2 * n + 1;
    // Σ_{n' < n} (2n' + 1)² = (4n³ − n)/3.
    let row = b0 + (4 * n * n * n - n) / 3 + (m + n as i32) as usize * width;
    let first = x0 + n * n;
    let mut acc = F::new(0.0f32);
    for c in 0..width {
        acc = fma(blocks[row + c], x[first + c], acc);
    }
    acc
}

/// Slot (j, m) of the coaxial step of M2L with the factors at `factors[c0..]` applied to
/// the vector at `x[x0..]`, added to `init`: for order i = |m|, row j − i of the order-i
/// matrix against input degrees n = i to p in increasing order, `acc = fma(±a, x, acc)`,
/// with the parity (−1)ⁿ⁺ʲ if `down` (`ShiftTables::coaxial_step`).
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a step's arguments are its buffers and the slot"
)]
fn coaxial_step<F: Float, L: List<F> + ?Sized>(
    x: &L,
    x0: usize,
    factors: &[F],
    c0: usize,
    j: usize,
    m: i32,
    down: bool,
    init: F,
    #[comptime] p: usize,
) -> F {
    let mut order = m;
    if m < 0 {
        order = -m;
    }
    let i = order as usize;
    let width = p + 1 - i;
    // Σ_{i' < i} (p + 1 − i')² = S(p + 1) − S(p + 1 − i), S(a) = a(a + 1)(2a + 1)/6.
    let top = comptime!((p + 1) * (p + 2) * (2 * p + 3) / 6);
    let start = top - width * (width + 1) * (2 * width + 1) / 6;
    let row = c0 + start + (j - i) * width;
    let mut acc = init;
    for n in i..p + 1 {
        let mut a = factors[row + n - i];
        if down && (n + j) % 2 == 1 {
            a = -a;
        }
        let centre = x0 + n * n + n;
        let mut at = centre + i;
        if m < 0 {
            at = centre - i;
        }
        acc = fma(a, x[at], acc);
    }
    acc
}

// --- Kernels ---------------------------------------------------------------------------

/// Rotation M2L, the cube layout: one cube per row of the rows list, unit u the owner of
/// slots u, u + `units`, … (module documentation).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn rotation_cube_kernel<F: Float>(
    rows: &[u32],
    row_offsets: &[u32],
    sources: &[u32],
    groups: &[u32],
    shifts: &[u32],
    forward: &[F],
    backward: &[F],
    azimuth: &[F],
    coaxial: &[F],
    input: &[F],
    locals: &mut [F],
    active: u32,
    input_offset: u32,
    locals_offset: u32,
    #[comptime] p: usize,
    #[comptime] units: usize,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let per_unit = comptime!(((p + 1) * (p + 1)).div_ceil(units));
    let blocks = comptime!((p + 1) * (2 * p + 1) * (2 * p + 3) / 3);
    let coaxial_values = comptime!((p + 1) * (p + 2) * (2 * p + 3) / 6);
    let mut first = Shared::<[F]>::new_slice(nc);
    let mut second = Shared::<[F]>::new_slice(nc);
    let cube = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    // The row and its entries are the same for every unit of the cube, so every barrier
    // below is reached by all of them or by none.
    if cube < active as usize {
        let t = rows[cube] as usize;
        let unit = UNIT_POS as usize;
        let out = locals_offset as usize + t * nc;
        // The unit's slots: their degrees, orders and accumulators.
        let mut degrees = Array::<u32>::new(per_unit);
        let mut orders = Array::<i32>::new(per_unit);
        let mut acc = Array::<F>::new(per_unit);
        #[unroll]
        for q in 0..per_unit {
            let k = unit + q * units;
            let mut n = 0usize;
            let mut value = F::new(0.0f32);
            if k < nc {
                n = degree_of(k);
                value = locals[out + k];
            }
            degrees[q] = n as u32;
            orders[q] = k as i32 - (n * n + n) as i32;
            acc[q] = value;
        }
        for e in row_offsets[t] as usize..row_offsets[t + 1] as usize {
            let x0 = input_offset as usize + sources[e] as usize * nc;
            let d = 4 * groups[e] as usize;
            let alignment = shifts[d];
            let b0 = shifts[d + 1] as usize * blocks;
            let f0 = shifts[d + 2] as usize * comptime!(2 * p);
            let c0 = shifts[d + 3] as usize * coaxial_values;
            if alignment == ROTATED {
                #[unroll]
                for q in 0..per_unit {
                    let k = unit + q * units;
                    if k < nc {
                        let (n, m) = (degrees[q] as usize, orders[q]);
                        first[k] = rotate_z::<F, [F]>(input, x0, azimuth, f0, n, m, false);
                    }
                }
                sync_cube();
                #[unroll]
                for q in 0..per_unit {
                    let k = unit + q * units;
                    if k < nc {
                        let (n, m) = (degrees[q] as usize, orders[q]);
                        second[k] = y_blocks::<F, Shared<[F]>>(&first, 0, forward, b0, n, m);
                    }
                }
                sync_cube();
                #[unroll]
                for q in 0..per_unit {
                    let k = unit + q * units;
                    if k < nc {
                        let (n, m) = (degrees[q] as usize, orders[q]);
                        first[k] = coaxial_step::<F, Shared<[F]>>(
                            &second,
                            0,
                            coaxial,
                            c0,
                            n,
                            m,
                            false,
                            F::new(0.0f32),
                            p,
                        );
                    }
                }
                sync_cube();
                #[unroll]
                for q in 0..per_unit {
                    let k = unit + q * units;
                    if k < nc {
                        let (n, m) = (degrees[q] as usize, orders[q]);
                        second[k] = y_blocks::<F, Shared<[F]>>(&first, 0, backward, b0, n, m);
                    }
                }
                sync_cube();
                #[unroll]
                for q in 0..per_unit {
                    let k = unit + q * units;
                    if k < nc {
                        let (n, m) = (degrees[q] as usize, orders[q]);
                        acc[q] += rotate_z::<F, Shared<[F]>>(&second, 0, azimuth, f0, n, m, true);
                    }
                }
            } else {
                #[unroll]
                for q in 0..per_unit {
                    let k = unit + q * units;
                    if k < nc {
                        let (n, m) = (degrees[q] as usize, orders[q]);
                        acc[q] = coaxial_step::<F, [F]>(
                            input,
                            x0,
                            coaxial,
                            c0,
                            n,
                            m,
                            alignment == DOWN,
                            acc[q],
                            p,
                        );
                    }
                }
            }
        }
        #[unroll]
        for q in 0..per_unit {
            let k = unit + q * units;
            if k < nc {
                locals[out + k] = acc[q];
            }
        }
    }
}

/// Rotation M2L, the CPU layout: one cube, each unit a contiguous range of the rows list;
/// per row the accumulators and per entry the two working vectors in local arrays, every
/// slot of a step in turn (module documentation).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn rotation_cpu_kernel<F: Float>(
    rows: &[u32],
    row_offsets: &[u32],
    sources: &[u32],
    groups: &[u32],
    shifts: &[u32],
    forward: &[F],
    backward: &[F],
    azimuth: &[F],
    coaxial: &[F],
    input: &[F],
    locals: &mut [F],
    active: u32,
    input_offset: u32,
    locals_offset: u32,
    #[comptime] p: usize,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let blocks = comptime!((p + 1) * (2 * p + 1) * (2 * p + 3) / 3);
    let coaxial_values = comptime!((p + 1) * (p + 2) * (2 * p + 3) / 6);
    let mut acc = Array::<F>::new(nc);
    let mut first = Array::<F>::new(nc);
    let mut second = Array::<F>::new(nc);
    let (first_row, count) = unit_rows(active as usize);
    for r in first_row..first_row + count {
        let t = rows[r] as usize;
        let out = locals_offset as usize + t * nc;
        for k in 0..nc {
            acc[k] = locals[out + k];
        }
        for e in row_offsets[t] as usize..row_offsets[t + 1] as usize {
            let x0 = input_offset as usize + sources[e] as usize * nc;
            let d = 4 * groups[e] as usize;
            let alignment = shifts[d];
            let b0 = shifts[d + 1] as usize * blocks;
            let f0 = shifts[d + 2] as usize * comptime!(2 * p);
            let c0 = shifts[d + 3] as usize * coaxial_values;
            if alignment == ROTATED {
                for n in 0..p + 1 {
                    for q in 0..2 * n + 1 {
                        let m = q as i32 - n as i32;
                        first[n * n + q] = rotate_z::<F, [F]>(input, x0, azimuth, f0, n, m, false);
                    }
                }
                for n in 0..p + 1 {
                    for q in 0..2 * n + 1 {
                        let m = q as i32 - n as i32;
                        second[n * n + q] = y_blocks::<F, Array<F>>(&first, 0, forward, b0, n, m);
                    }
                }
                for n in 0..p + 1 {
                    for q in 0..2 * n + 1 {
                        let m = q as i32 - n as i32;
                        first[n * n + q] = coaxial_step::<F, Array<F>>(
                            &second,
                            0,
                            coaxial,
                            c0,
                            n,
                            m,
                            false,
                            F::new(0.0f32),
                            p,
                        );
                    }
                }
                for n in 0..p + 1 {
                    for q in 0..2 * n + 1 {
                        let m = q as i32 - n as i32;
                        second[n * n + q] = y_blocks::<F, Array<F>>(&first, 0, backward, b0, n, m);
                    }
                }
                for n in 0..p + 1 {
                    for q in 0..2 * n + 1 {
                        let m = q as i32 - n as i32;
                        acc[n * n + q] +=
                            rotate_z::<F, Array<F>>(&second, 0, azimuth, f0, n, m, true);
                    }
                }
            } else {
                for n in 0..p + 1 {
                    for q in 0..2 * n + 1 {
                        let m = q as i32 - n as i32;
                        acc[n * n + q] = coaxial_step::<F, [F]>(
                            input,
                            x0,
                            coaxial,
                            c0,
                            n,
                            m,
                            alignment == DOWN,
                            acc[n * n + q],
                            p,
                        );
                    }
                }
            }
        }
        for k in 0..nc {
            locals[out + k] = acc[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(backend: BackendKind) -> DeviceInfo {
        DeviceInfo {
            backend,
            name: "test".into(),
            compiler: "test".into(),
            cubecl_version: crate::CUBECL_VERSION,
            f32: true,
            f64: !backend.is_gpu(),
            plane_size: (32, 32),
            max_shared_memory: 32_768,
            max_units_per_cube: 1024,
            max_cube_count: (65_535, 65_535, 65_535),
            max_memory: None,
        }
    }

    #[test]
    fn default_layouts_cover_every_slot() {
        assert_eq!(
            RotationLayout::default_for(&info(BackendKind::Cpu), 8),
            RotationLayout::Cpu
        );
        for (p, units) in [
            (0, 32),
            (2, 32),
            (4, 32),
            (5, 64),
            (8, 96),
            (16, 320),
            (20, 448),
        ] {
            let layout = RotationLayout::default_for(&info(BackendKind::Metal), p);
            assert_eq!(layout, RotationLayout::Cube { units }, "p = {p}");
            layout
                .check(&info(BackendKind::Metal), p, Precision::F32)
                .unwrap();
        }
        // No unit, too many for the device, too much shared memory; fewer units than slots
        // are fine.
        let metal = info(BackendKind::Metal);
        assert!(
            RotationLayout::Cube { units: 0 }
                .check(&metal, 8, Precision::F32)
                .is_err()
        );
        RotationLayout::Cube { units: 4 }
            .check(&metal, 8, Precision::F32)
            .unwrap();
        assert!(
            RotationLayout::Cube { units: 448 }
                .check(
                    &DeviceInfo {
                        max_shared_memory: 4096,
                        ..metal.clone()
                    },
                    20,
                    Precision::F64
                )
                .is_err()
        );
        assert!(
            RotationLayout::Cube { units: 2048 }
                .check(&metal, 8, Precision::F32)
                .is_err()
        );
    }

    #[test]
    fn piece_lengths_match_the_tables_crate() {
        // B and C of nd_fmm_tables::rotation at p = 0, 1, 8 and 20.
        for (p, b, c) in [(0, 1, 1), (1, 10, 5), (8, 969, 285), (20, 12_341, 3_311)] {
            assert_eq!(blocks_len(p), b, "B at p = {p}");
            assert_eq!(coaxial_len(p), c, "C at p = {p}");
            let sum: usize = (0..=p).map(|n| (2 * n + 1) * (2 * n + 1)).sum();
            assert_eq!(blocks_len(p), sum);
            let sum: usize = (0..=p).map(|i| (p + 1 - i) * (p + 1 - i)).sum();
            assert_eq!(coaxial_len(p), sum);
        }
    }

    #[test]
    fn plans_keep_the_rows_with_an_entry() {
        assert_eq!(RotationPlan::rows(&[0, 2, 2, 5, 5]), vec![0, 2]);
        assert!(RotationPlan::rows(&[0, 0, 0]).is_empty());
        assert_eq!(RotationPlan::bytes(&[0, 2, 2, 5, 5]), 8);
    }
}
