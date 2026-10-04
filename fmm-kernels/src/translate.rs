//! The grouped translations with dense tables (Phase 4 T8, C4.4: M2M and L2L; T9, C4.5:
//! dense M2L): one level's M2M, L2L or M2L as gathers, a grouped GEMM and a reduction or
//! scatter-add (docs/design/device-path.md §6.4, §6.5).
//!
//! # What a level call computes
//!
//! A grouped view ([`GroupedView`]) holds the pairs (target t, source s, group g) of one
//! level, in rows by target (groups ascending) and in batches by group. Group g has the
//! table A_g, an n × n matrix with n = (p + 1)², stored column-major as `MatrixSet` stores
//! it (CONVENTIONS §3.12, "Matrix layout": matrix g at g n²). [`grouped`] adds, for every
//! target, A_g x_s of each entry of its row into its output column, in row order:
//!
//! | Use | Rows (targets) | Sources | Groups | [`Accumulate`] |
//! | --- | --- | --- | --- | --- |
//! | M2M (either pass) | parents on l | children on l + 1 | octants | [`Rows`](Accumulate::Rows) |
//! | L2L | children on l | the parent on l − 1 | octants | [`Scatter`](Accumulate::Scatter): one entry per row |
//! | M2L (T9) | boxes on l | V-list sources on l | the 316 offsets, in index order | [`Rows`](Accumulate::Rows) |
//!
//! The tables of a view ([`Tables`]) are uploaded once: the 8 octant matrices of M2M or
//! L2L, or the 316 offset matrices of `M2lTables` (or of `M2lClasses::expand`), n² values
//! each. A target meets each group at most once (a parent has one child per octant, a box
//! one V-list source per offset), and a row's groups ascend, which for M2L is the offset
//! order of every V row (requirement 4).
//!
//! # Structure (device-path.md §6.4, structure (B))
//!
//! The batch entries of a view are cut into **chunks**, contiguous ranges of entries in
//! batch order whose gathered inputs X and products Y fit in the scratch budget
//! ([`DEFAULT_SCRATCH_BYTES`], 128 MB by default). Per chunk, three launches:
//!
//! 1. **gather**: X[:, c] = input[:, s] for the chunk's entries in batch order
//!    ([`movement::gather_columns`](crate::movement::gather_columns));
//! 2. **grouped GEMM**: Y[:, c] = A_g X[:, c] for every column, into the temporary
//!    (β = 0), in one launch over a tile schedule built at build: each tile a group and a
//!    range of its columns ([`TileSchedule`]), so no cube idles and each uses its group's
//!    table;
//! 3. **accumulate**: [`Accumulate::Rows`] runs a reduction that owns each target value:
//!    it loads it, adds Y[:, pos(e)] for each entry e of the target's row that lies in
//!    the chunk, in row order, and stores it once, with pos(e) from the view's
//!    row-to-batch map; [`Accumulate::Scatter`] adds each column into its target
//!    ([`movement::scatter_add_columns`](crate::movement::scatter_add_columns)), for views whose targets are distinct over the
//!    level (L2L: a child has one parent).
//!
//! No atomics anywhere: the GEMM writes distinct columns of Y, and only the reduction,
//! which owns each target, or the scatter-add, whose targets are distinct, writes the
//! output. Chunks run in ascending batch order and a row's entries ascend in group, so
//! every target meets its entries in row order whichever chunks they fall in; with the
//! hand-written GEMM the result does not depend on the budget (bit for bit). Batches
//! without a column have no tile and no padded column, and a view without pairs launches
//! nothing.
//!
//! **Structure (A)**, one gather, GEMM and scatter-add per group in group order
//! ([`per_group`], [`PerGroupPlan`]), adds the same products in the same order into each
//! target: it is the test reference of (B), bit for bit with the same GEMM, at three
//! launches per group with a column (up to 948 per M2L level, against three per chunk).
//!
//! # The hand-written GEMM ([`GemmLayout`])
//!
//! Y = A X with A n × n and X n × k, column-major as the level buffers and the tables are;
//! read row-major, that is the plain row-major product X_rm Aᵀ_rm (design §6.4). n is a
//! comptime parameter, every k works (0, 1 and k not a multiple of the tile included).
//! **Summation order**: each output Y_ij is one accumulator, starting from zero, updated
//! as `acc = fma(A_ik, X_kj, acc)` for k = 0, 1, …, n − 1 in turn, and stored once. The
//! explicit `fma` pins the rounding on every backend (spikes/device-arith/REPORT.md,
//! "Recommendation", rule 4), so the product equals a host loop
//! `acc = x_k.mul_add(a_ik, acc)` in the same order, bit for bit (rule 6, measured by T3
//! on Metal and the CPU runtime). It differs from `MatrixSet::apply`, which adds each
//! unfused product into the output directly, by rounding only (≤ n u_T relative to the
//! terms; device-path.md §9.2). Accumulating in the GEMM from a gathered copy of the
//! output (β = 1) is rejected by the design (§6.4).
//!
//! Two layouts of one inner block, chosen per backend (design §6.5; spikes/cubecl-gemm,
//! `tiled-reg-cols` and `tiled-reg-rows` with rows and columns exchanged):
//!
//! | Layout | Default on | Mapping |
//! | --- | --- | --- |
//! | [`Cube`](GemmLayout::Cube) | Metal, CUDA | one cube per tile, `rows` × `columns` units; unit (u, v) owns the rows u, u + `rows`, … of `per_unit` consecutive columns: adjacent units read adjacent entries of a table column and write adjacent values of Y (coalesced), and share the X values they read |
//! | [`Cpu`](GemmLayout::Cpu) | the CPU runtime | one cube, one unit per core (at most [`Device::units_cap`]), each a contiguous range of tiles of `per_unit` columns, rows in blocks of `block`; no shared memory, no barrier |
//!
//! # The library GEMM
//!
//! Under [`GemmPolicy::Auto`], a view in f32 at p ≥ 8 (n ≥ 81) on a GPU backend tries
//! `cubek-matmul` with the strategy named explicitly,
//! `Strategy::MultiLevel(SimpleCyclicCmma)` (never the per-call `Strategy::Auto`,
//! requirement 6; design §6.5). The library needs one shape per launch, so each chunk is
//! a run of contiguous groups padded to the run's widest batch, k columns each: X and Y
//! are [G_c, k, n] row-major (strides [k n, n, 1]) and the tables [G_c, n, n] from the
//! chunk's first group, one batched launch per chunk (numpy-style batches, device-path.md
//! §6.4 and F22):
//! - **chunks**: runs of groups from a group with a column to the last one with a column
//!   whose padded width G_c k fits the budget's columns, in ascending group order; a group
//!   wider than that alone, in pieces of the budget's columns. The padding columns gather
//!   a valid source and their products are never read;
//! - **tables**: the library reads the second copy of [`Tables`], with matrix g at g s and
//!   s = n² rounded up to [`LIBRARY_TABLE_ALIGNMENT`] bytes ([`library_stride`]), so that
//!   each chunk binds its first table at an aligned offset (wgpu binds storage buffers
//!   only at aligned offsets; device-path.md F6). The hand-written kernel reads the
//!   compact copy;
//! - **a probe launch** of every chunk shape at build must succeed (`Unavailable` and every
//!   other setup error send the view to the hand-written kernel);
//! - **the input-precision guard**: each probe's resolved `MatmulElems` keep T for the
//!   stage and register types of both inputs and of the accumulator. A strategy that
//!   rounds f32 inputs to TF32, F16 or BF16 (as the accelerated routines do on a backend
//!   that registers TF32, CUDA) is never used (docs/phase4/README.md, "M2L strategies").
//!   The guard reads the resolved element types, never the backend's name; where a probe
//!   fails, the guard cannot be established and the hand-written kernel runs.
//!
//! Otherwise the hand-written kernel runs, for the whole view. The choice is fixed per
//! view at build and reported ([`GroupedPlan::gemm`], [`GroupedPlan::library_rejection`]).
//! The library sums in its own tiles' order, fixed for a shape, device and driver
//! (device-path.md §9.1).
//!
//! # Orientation (Phase 4 T12)
//!
//! [`PlanSettings::orientation`] chooses the layout of X and Y in the scratch
//! ([`Orientation`]). [`BoxMajor`](Orientation::BoxMajor), the default, keeps each
//! column's n coefficients contiguous, the layout of device-path.md §6.4: the GEMM sees
//! M = columns and N = n. [`CoefficientMajor`](Orientation::CoefficientMajor) cuts each
//! chunk into blocks of w columns, the whole chunk for the hand-written kernel and one
//! group's padded batch (k_c columns) for the library, and stores coefficient k of column
//! c of block b at (b n + k) w + c: the GEMMs see the spike's orientation, M = n and
//! N = columns (spikes/cubecl-gemm; T9 measured the box-major orientation alone at
//! 71–84% of it for the library). The gather writes X in that layout
//! ([`movement::gather_coefficients`](crate::movement::gather_coefficients)), the
//! hand-written kernel reads and writes it with the same mapping of units, accumulators
//! and k order, so **its products are bit for bit those of the box-major layout**, the
//! library reads the tables as column-major A instead of row-major Aᵀ (its own order,
//! like the box-major library), and the reduction reads Y in the same layout; L2L then
//! runs the reduction, which adds its one product per child as the scatter-add does.
//!
//! # Launch
//!
//! [`Tables::upload`] uploads a view's tables once (with the library copy where the policy
//! may take the library). [`GroupedPlan::new`] validates the host arrays and builds the
//! chunks, the tile schedule and (for the library) the padded gather indices, uploading
//! each once. [`TranslationScratch`] holds X and Y, allocated once for every view.
//! [`grouped`] runs a level call and allocates nothing; [`per_group`] runs structure (A);
//! [`gemm`] runs one GEMM over a [`TileSchedule`] and [`library`] one library GEMM, for
//! tests and reports.

use std::fmt;
use std::ops::Range;

use cubecl::prelude::*;
use cubecl::zspace::{Shape, Strides};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::launch::launch_ref;
use cubek_matmul::multi_level;
use cubek_matmul::strategy::Strategy;
use cubek_std::InputBinding;

use crate::buffer::{DeviceBuffer, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexBuffer};
use crate::device::{BackendKind, Device, DeviceInfo, Precision};
use crate::error::KernelError;
use crate::leaf::unit_rows;
use crate::movement::{gather_coefficients, gather_columns, scatter_add_columns};
use crate::p2p::cube_grid;
use crate::view::{GroupedArrays, GroupedView, row_to_batch};

/// The scratch budget of the gathered inputs and products of one chunk together, X and Y,
/// by default: 128 MB (device-path.md §6.4, a model that T9 measures).
pub const DEFAULT_SCRATCH_BYTES: u64 = 128 << 20;

/// The units along the rows of the [`Cube`](GemmLayout::Cube) layout by default: at most
/// one plane of 32, fewer for small n (the next power of two above n).
pub const GPU_GEMM_ROWS: u32 = 32;

/// The units per cube of the [`Cube`](GemmLayout::Cube) layout by default: 64, the rows
/// times the columns.
pub const GPU_GEMM_UNITS: u32 = 64;

/// The columns per unit of the default layouts.
pub const GEMM_COLUMNS_PER_UNIT: u32 = 4;

/// The rows per block of the [`Cpu`](GemmLayout::Cpu) layout by default (at most n).
pub const CPU_GEMM_BLOCK: u32 = 8;

/// The alignment in bytes of each matrix of the library copy of [`Tables`]: 256, the
/// largest storage-buffer offset alignment wgpu asks of a backend
/// (`min_storage_buffer_offset_alignment`; 32 on Metal) and its default memory alignment.
pub const LIBRARY_TABLE_ALIGNMENT: usize = 256;

/// The values from one matrix of the library copy of [`Tables`] to the next: n² rounded
/// up to [`LIBRARY_TABLE_ALIGNMENT`] bytes in `precision`.
pub fn library_stride(n: usize, precision: Precision) -> usize {
    (n * n).next_multiple_of(LIBRARY_TABLE_ALIGNMENT / value_bytes(precision))
}

/// The order of the tables, (p + 1)² for p up to the largest degree of the FMM.
fn check_order(what: &str, n: usize) {
    let max = crate::leaf::coefficients(crate::leaf::MAX_DEGREE);
    assert!(
        (1..=max).contains(&n),
        "{what}: tables of order {n}; expected 1 to {max}"
    );
}

/// The bytes of one value of `precision`.
fn value_bytes(precision: Precision) -> usize {
    match precision {
        Precision::F32 => 4,
        Precision::F64 => 8,
    }
}

/// The layout of the hand-written GEMM (module documentation, "The hand-written GEMM").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmLayout {
    /// One cube per tile of `columns · per_unit` columns, `rows × columns` units: unit
    /// (u, v) owns rows u, u + `rows`, … of the tile's columns v · `per_unit` onwards.
    Cube {
        /// Units along the rows, at least 1.
        rows: u32,
        /// Units along the columns, at least 1.
        columns: u32,
        /// Columns per unit, at least 1.
        per_unit: u32,
    },
    /// One cube of one unit per core, each a contiguous range of tiles of `per_unit`
    /// columns, the rows in blocks of `block`.
    Cpu {
        /// Rows per block, at least 1.
        block: u32,
        /// Columns per tile, at least 1.
        per_unit: u32,
    },
}

impl GemmLayout {
    /// The default layout of a device for tables of order n: [`Cpu`](Self::Cpu) on the
    /// CPU runtime with blocks of [`CPU_GEMM_BLOCK`] rows (at most n) and
    /// [`GEMM_COLUMNS_PER_UNIT`] columns; otherwise [`Cube`](Self::Cube) with up to
    /// [`GPU_GEMM_ROWS`] units along the rows (the next power of two at or above n, if
    /// smaller), [`GPU_GEMM_UNITS`] units in all and [`GEMM_COLUMNS_PER_UNIT`] columns per
    /// unit (design §6.5).
    pub fn default_for(info: &DeviceInfo, n: usize) -> Self {
        match info.backend {
            BackendKind::Cpu => Self::Cpu {
                block: CPU_GEMM_BLOCK.min(n.max(1) as u32),
                per_unit: GEMM_COLUMNS_PER_UNIT,
            },
            BackendKind::Metal | BackendKind::Cuda => {
                let units = GPU_GEMM_UNITS.min(info.max_units_per_cube.max(1));
                let rows = GPU_GEMM_ROWS
                    .min(n.max(1).next_power_of_two() as u32)
                    .min(units);
                Self::Cube {
                    rows,
                    columns: (units / rows).max(1),
                    per_unit: GEMM_COLUMNS_PER_UNIT,
                }
            }
        }
    }

    /// The columns of one tile: `columns · per_unit` for the cube layout, `per_unit` for
    /// the CPU layout.
    pub fn tile(self) -> usize {
        match self {
            Self::Cube {
                columns, per_unit, ..
            } => columns as usize * per_unit as usize,
            Self::Cpu { per_unit, .. } => per_unit as usize,
        }
    }

    /// Checks that the device can run the layout.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedLayout`] for a zero parameter, or a cube of more units
    /// than the device allows per cube.
    pub fn check(self, info: &DeviceInfo) -> Result<(), KernelError> {
        let refuse = |reason: String| {
            Err(KernelError::UnsupportedLayout {
                layout: self.to_string(),
                reason,
            })
        };
        match self {
            Self::Cube {
                rows,
                columns,
                per_unit,
            } => {
                if rows == 0 || columns == 0 || per_unit == 0 {
                    return refuse("a zero parameter".into());
                }
                let units = u64::from(rows) * u64::from(columns);
                if units > u64::from(info.max_units_per_cube) {
                    return refuse(format!(
                        "{units} units per cube, the device allows {}",
                        info.max_units_per_cube
                    ));
                }
                Ok(())
            }
            Self::Cpu { block, per_unit } => {
                if block == 0 || per_unit == 0 {
                    refuse("a zero parameter".into())
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl fmt::Display for GemmLayout {
    /// `cube(32 x 2 units, 4 columns per unit)` or `cpu(blocks of 8 rows, 4 columns)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Cube {
                rows,
                columns,
                per_unit,
            } => write!(
                f,
                "cube({rows} x {columns} units, {per_unit} columns per unit)"
            ),
            Self::Cpu { block, per_unit } => {
                write!(f, "cpu(blocks of {block} rows, {per_unit} columns)")
            }
        }
    }
}

/// The layout of a chunk's gathered inputs X and products Y (module documentation,
/// "Orientation"; Phase 4 T12).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Orientation {
    /// Each column's n coefficients contiguous, as in a level buffer: read row-major, X is
    /// k × n and the GEMM has M = columns, N = n (device-path.md §6.4). The default.
    #[default]
    BoxMajor,
    /// Each coefficient's values across the columns of a block contiguous: X is n × w per
    /// block of w columns (the whole chunk for the hand-written kernel, one group's padded
    /// batch for the library), the spike's orientation with M = n, N = columns.
    CoefficientMajor,
}

impl fmt::Display for Orientation {
    /// `box-major` or `coefficient-major`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::BoxMajor => "box-major",
            Self::CoefficientMajor => "coefficient-major",
        })
    }
}

/// Which GEMMs a [`GroupedPlan`] may use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GemmPolicy {
    /// The rule of design §6.5: the library in f32 at p ≥ 8 on a GPU backend where it
    /// passes the checks of the module documentation ("The library GEMM"), the
    /// hand-written kernel otherwise.
    #[default]
    Auto,
    /// Always the hand-written kernel.
    HandWritten,
}

/// The GEMM a [`GroupedPlan`] runs, fixed at build.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gemm {
    /// The hand-written kernel in this layout.
    HandWritten(GemmLayout),
    /// `cubek-matmul` with `Strategy::MultiLevel(SimpleCyclicCmma)`, one batched launch of
    /// the padded batches.
    Library,
}

impl fmt::Display for Gemm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HandWritten(layout) => write!(f, "hand-written {layout}"),
            Self::Library => f.write_str("library (matmul_simple_cyclic_cmma)"),
        }
    }
}

/// How a level call adds its products into the output (module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Accumulate {
    /// A reduction per target in row order: M2M (up to eight children per parent) and
    /// M2L.
    Rows,
    /// A scatter-add of each product into its target, for views whose targets are
    /// distinct over the level (L2L). With the library's padded layout the reduction
    /// runs instead, which adds the single product the same way.
    Scatter,
}

/// The tiles of a grouped GEMM, three `u32` per tile (group, first column, columns), in
/// group order and, within a group, column order; built and uploaded once.
#[derive(Debug)]
pub struct TileSchedule {
    tiles: IndexBuffer,
    /// Tile ranges, one per chunk.
    segments: Vec<Range<usize>>,
    /// The tile width the schedule was built for.
    width: usize,
    /// One more than the largest group of a tile.
    groups: usize,
    /// The columns the tiles of each segment address.
    columns: Vec<usize>,
}

/// The host side of a tile schedule: tiles of up to `width` columns over `segments`, each
/// a list of (group, first column, columns) runs.
fn tiles_of(width: usize, runs: &[Vec<(usize, usize, usize)>]) -> (Vec<u32>, Vec<Range<usize>>) {
    let mut tiles = Vec::new();
    let mut segments = Vec::with_capacity(runs.len());
    for segment in runs {
        let start = tiles.len() / 3;
        for &(g, first, count) in segment {
            let mut c = 0;
            while c < count {
                let len = width.min(count - c);
                tiles.extend([g as u32, (first + c) as u32, len as u32]);
                c += len;
            }
        }
        segments.push(start..tiles.len() / 3);
    }
    (tiles, segments)
}

impl TileSchedule {
    /// The schedule of one GEMM over the batches of `batch_offsets` (`ngroups + 1` CSR
    /// offsets over the columns): group g's columns are
    /// `batch_offsets[g]..batch_offsets[g + 1]`, in tiles of [`GemmLayout::tile`] columns;
    /// empty groups have no tile. One upload.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If `batch_offsets` are not CSR offsets.
    pub fn new(
        device: &mut Device,
        layout: GemmLayout,
        batch_offsets: &[u32],
    ) -> Result<Self, KernelError> {
        assert!(
            !batch_offsets.is_empty()
                && batch_offsets[0] == 0
                && batch_offsets.windows(2).all(|w| w[0] <= w[1]),
            "TileSchedule: the batch offsets are not CSR offsets"
        );
        let runs: Vec<(usize, usize, usize)> = batch_offsets
            .windows(2)
            .enumerate()
            .map(|(g, w)| (g, w[0] as usize, (w[1] - w[0]) as usize))
            .collect();
        let columns = *batch_offsets.last().unwrap() as usize;
        Self::upload(device, layout.tile(), &[runs], vec![columns])
    }

    fn upload(
        device: &mut Device,
        width: usize,
        runs: &[Vec<(usize, usize, usize)>],
        columns: Vec<usize>,
    ) -> Result<Self, KernelError> {
        let (tiles, segments) = tiles_of(width, runs);
        let groups = runs
            .iter()
            .flatten()
            .filter(|r| r.2 > 0)
            .map(|r| r.0 + 1)
            .max()
            .unwrap_or(0);
        Ok(Self {
            tiles: device.upload_indices(&tiles)?,
            segments,
            width,
            groups,
            columns,
        })
    }

    /// The number of tiles.
    pub fn len(&self) -> usize {
        self.tiles.len() / 3
    }

    /// True if there is no tile.
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    /// The tile width the schedule was built for.
    pub fn width(&self) -> usize {
        self.width
    }
}

/// The tables of grouped translations on the device, uploaded once (module documentation):
/// `count` matrices of order n, matrix g at g n², column-major as `MatrixSet` stores them
/// (CONVENTIONS §3.12, "Matrix layout"), which the hand-written kernel reads; and, where the
/// library GEMM may run, a second copy with matrix g at g s, s = [`library_stride`] (zero
/// padding between the matrices), which the library reads.
#[derive(Debug)]
pub struct Tables<T: DeviceFloat> {
    compact: DeviceBuffer<T>,
    library: Option<DeviceBuffer<T>>,
    n: usize,
    count: usize,
}

impl<T: DeviceFloat> Tables<T> {
    /// Uploads the matrices of order n in `matrices` (matrix g at g n²), and with `library`
    /// also the library copy: one upload, or two.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If n is not an order 1 to 441, or `matrices` is not a whole number of matrices.
    pub fn upload(
        device: &mut Device,
        matrices: &[T],
        n: usize,
        library: bool,
    ) -> Result<Self, KernelError> {
        check_order("Tables", n);
        assert!(
            matrices.len().is_multiple_of(n * n),
            "Tables: {} values are not whole matrices of order {n}",
            matrices.len()
        );
        let count = matrices.len() / (n * n);
        let compact = device.upload(matrices)?;
        let library = if library {
            let stride = library_stride(n, T::FLOAT);
            let mut padded = vec![T::from_int(0); count * stride];
            for (g, matrix) in matrices.chunks_exact(n * n).enumerate() {
                padded[g * stride..g * stride + n * n].copy_from_slice(matrix);
            }
            Some(device.upload(&padded)?)
        } else {
            None
        };
        Ok(Self {
            compact,
            library,
            n,
            count,
        })
    }

    /// The bytes [`upload`](Self::upload) allocates for `count` matrices of order n.
    pub fn bytes(n: usize, count: usize, library: bool) -> u64 {
        Device::buffer_bytes::<T>(count * n * n)
            + if library {
                Device::buffer_bytes::<T>(count * library_stride(n, T::FLOAT))
            } else {
                0
            }
    }

    /// The compact copy: matrix g at g n².
    pub fn compact(&self) -> DeviceSlice<'_, T> {
        self.compact.as_slice()
    }

    /// True if the library copy was uploaded.
    pub fn has_library_copy(&self) -> bool {
        self.library.is_some()
    }

    /// Frees the library copy, for tables whose plans all take the hand-written kernel
    /// (Phase 4 T12: the copy uploaded for tuning, when no level chose the library); the
    /// bytes it held, 0 if there was none. A library plan over these tables then panics
    /// in [`grouped`], as for tables uploaded without the copy.
    pub fn release_library_copy(&mut self) -> u64 {
        self.library
            .take()
            .map_or(0, |copy| Device::buffer_bytes::<T>(copy.len()))
    }

    /// The order n of the matrices.
    pub fn n(&self) -> usize {
        self.n
    }

    /// The number of matrices.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The library copy of the `groups` matrices from `first`, and its stride.
    fn library_run(&self, first: usize, groups: usize) -> (DeviceSlice<'_, T>, usize) {
        let stride = library_stride(self.n, T::FLOAT);
        let copy = self
            .library
            .as_ref()
            .expect("the tables have a library copy");
        (
            copy.slice(first * stride..(first + groups) * stride),
            stride,
        )
    }
}

/// The gathered inputs X and the products Y of a chunk, allocated once and shared by
/// every view of an FMM.
#[derive(Debug)]
pub struct TranslationScratch<T: DeviceFloat> {
    x: DeviceBuffer<T>,
    y: DeviceBuffer<T>,
}

impl<T: DeviceFloat> TranslationScratch<T> {
    /// Allocates X and Y of `values` values each (zeroed).
    ///
    /// # Errors
    ///
    /// As [`Device::alloc`].
    pub fn new(device: &mut Device, values: usize) -> Result<Self, KernelError> {
        Ok(Self {
            x: device.alloc(values)?,
            y: device.alloc(values)?,
        })
    }

    /// The bytes [`new`](Self::new) allocates for `values` values each.
    pub fn bytes(values: usize) -> u64 {
        2 * Device::buffer_bytes::<T>(values)
    }

    /// The values of X (and of Y).
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// True if X and Y hold no value.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// The products Y of the last chunk, for tests.
    pub fn products(&self) -> DeviceSlice<'_, T> {
        self.y.as_slice()
    }
}

/// One chunk of a [`GroupedPlan`].
#[derive(Clone, Debug)]
struct Chunk {
    /// Its batch entries.
    entries: Range<usize>,
    /// Its columns of X and Y.
    width: usize,
    /// The padded layout of the library: the columns per group, the widest batch of the
    /// chunk's groups (0: compact, column c of the chunk is entry `entries.start + c`).
    padded: usize,
    /// The library's first group (0 for a compact chunk).
    first_group: usize,
    /// The library's batches: groups `first_group..first_group + groups`.
    groups: usize,
    /// The library's padded gather indices of the chunk, in the plan's padded sources.
    sources: Range<usize>,
}

impl Chunk {
    /// The columns of one block of the coefficient-major layout (module documentation,
    /// "Orientation"): the columns per group of a padded chunk, else the whole chunk.
    fn block(&self) -> usize {
        if self.padded > 0 {
            self.padded
        } else {
            self.width
        }
    }
}

/// The facts of a chunk layout, computed on the host (module documentation): the chunks,
/// the runs of tiles per chunk and the most columns of a chunk.
#[derive(Clone, Debug)]
struct Layout {
    chunks: Vec<Chunk>,
    runs: Vec<Vec<(usize, usize, usize)>>,
    columns: usize,
}

/// The compact chunks of `batch_offsets` with at most `max_columns` columns each.
fn compact_layout(batch_offsets: &[u32], max_columns: usize) -> Layout {
    let total = *batch_offsets.last().unwrap() as usize;
    let mut chunks = Vec::new();
    let mut runs = Vec::new();
    let mut start = 0;
    while start < total {
        let end = total.min(start + max_columns);
        let mut segment = Vec::new();
        for (g, w) in batch_offsets.windows(2).enumerate() {
            let (b0, b1) = (w[0] as usize, w[1] as usize);
            let (lo, hi) = (b0.max(start), b1.min(end));
            if lo < hi {
                segment.push((g, lo - start, hi - lo));
            }
        }
        chunks.push(Chunk {
            entries: start..end,
            width: end - start,
            padded: 0,
            first_group: 0,
            groups: 0,
            sources: 0..0,
        });
        runs.push(segment);
        start = end;
    }
    let columns = chunks.iter().map(|c| c.width).max().unwrap_or(0);
    Layout {
        chunks,
        runs,
        columns,
    }
}

/// The library's padded chunks of `batch_offsets` with at most `max_columns` columns each
/// (module documentation, "The library GEMM"): runs of contiguous groups, from a group with
/// a column to the last one with a column, padded to the run's widest batch; a group wider
/// than `max_columns` alone, in pieces. No tiles.
fn library_layout(batch_offsets: &[u32], max_columns: usize) -> Layout {
    let ngroups = batch_offsets.len() - 1;
    let k = |g: usize| (batch_offsets[g + 1] - batch_offsets[g]) as usize;
    let at = |g: usize| batch_offsets[g] as usize;
    let mut chunks = Vec::new();
    let mut sources = 0;
    let mut g = 0;
    while g < ngroups {
        if k(g) == 0 {
            g += 1;
            continue;
        }
        if k(g) > max_columns {
            let mut c = 0;
            while c < k(g) {
                let len = max_columns.min(k(g) - c);
                chunks.push(Chunk {
                    entries: at(g) + c..at(g) + c + len,
                    width: len,
                    padded: len,
                    first_group: g,
                    groups: 1,
                    sources: sources..sources + len,
                });
                sources += len;
                c += len;
            }
            g += 1;
            continue;
        }
        let (first, mut last, mut widest) = (g, g, k(g));
        for h in g + 1..ngroups {
            let wider = widest.max(k(h));
            if (h + 1 - first) * wider > max_columns {
                break;
            }
            if k(h) > 0 {
                (last, widest) = (h, wider);
            }
        }
        let groups = last + 1 - first;
        let width = groups * widest;
        chunks.push(Chunk {
            entries: at(first)..at(last + 1),
            width,
            padded: widest,
            first_group: first,
            groups,
            sources: sources..sources + width,
        });
        sources += width;
        g = last + 1;
    }
    let columns = chunks.iter().map(|c| c.width).max().unwrap_or(0);
    Layout {
        runs: vec![Vec::new(); chunks.len()],
        chunks,
        columns,
    }
}

/// What a [`GroupedPlan`] for a view of these batches needs on the device at most,
/// before it is built: the bytes of its index buffers, and columns of X and Y. Exact for
/// the hand-written kernel; with a library candidate the larger of the two layouts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanSize {
    /// The bytes of the index buffers the plan uploads.
    pub bytes: u64,
    /// The columns of X and Y its chunks need.
    pub columns: usize,
}

/// The settings of a [`GroupedPlan`]: tables of order n, the hand-written layout, the
/// policy and the scratch budget in bytes (X and Y together).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanSettings {
    /// The order n = (p + 1)² of the tables.
    pub n: usize,
    /// The layout of the hand-written kernel.
    pub layout: GemmLayout,
    /// Which GEMMs the plan may use.
    pub policy: GemmPolicy,
    /// The bytes X and Y of a chunk may take together, [`DEFAULT_SCRATCH_BYTES`] by
    /// default.
    pub budget: u64,
    /// The layout of X and Y ([`Orientation::BoxMajor`] by default).
    pub orientation: Orientation,
}

impl PlanSettings {
    /// The most columns of a chunk in `precision`: half the budget over n values, at
    /// least 1.
    fn max_columns(&self, precision: Precision) -> usize {
        let per_column = 2 * self.n as u64 * value_bytes(precision) as u64;
        usize::try_from(self.budget / per_column)
            .unwrap_or(usize::MAX)
            .max(1)
    }

    /// Whether the policy lets a view in `precision` on `backend` try the library: f32,
    /// p ≥ 8, a GPU backend (module documentation). Its tables then need the library copy
    /// ([`Tables::upload`] with `library`).
    pub fn library_candidate(&self, backend: BackendKind, precision: Precision) -> bool {
        self.policy == GemmPolicy::Auto
            && precision == Precision::F32
            && self.n >= 81
            && backend.is_gpu()
    }

    /// The library's padded layout over `batch_offsets`, if the policy lets the view try
    /// the library and it has a pair.
    fn library_layout(
        &self,
        backend: BackendKind,
        precision: Precision,
        batch_offsets: &[u32],
    ) -> Option<Layout> {
        (self.library_candidate(backend, precision) && *batch_offsets.last().unwrap() > 0)
            .then(|| library_layout(batch_offsets, self.max_columns(precision)))
    }

    /// The size of a plan over `batch_offsets` for `backend` and `precision`, at most
    /// (no device needed; for the memory check before anything is allocated).
    pub fn size(
        &self,
        backend: BackendKind,
        precision: Precision,
        batch_offsets: &[u32],
    ) -> PlanSize {
        let compact = compact_layout(batch_offsets, self.max_columns(precision));
        let (tiles, _) = tiles_of(self.layout.tile(), &compact.runs);
        let library = self.library_layout(backend, precision, batch_offsets);
        let padded = library
            .as_ref()
            .map_or(0, |l| l.chunks.last().map_or(0, |c| c.sources.end));
        PlanSize {
            bytes: Device::buffer_bytes::<u32>(tiles.len())
                + if library.is_some() {
                    Device::buffer_bytes::<u32>(padded)
                } else {
                    0
                },
            columns: compact
                .columns
                .max(library.map_or(0, |l: Layout| l.columns)),
        }
    }
}

/// The schedule of one grouped view, built once (module documentation, "Launch"): its
/// chunks, the tile schedule of the hand-written kernel, or the padded gather of the
/// library.
#[derive(Debug)]
pub struct GroupedPlan {
    n: usize,
    nrows: usize,
    ngroups: usize,
    len: usize,
    sources_bound: usize,
    precision: Precision,
    gemm: Gemm,
    orientation: Orientation,
    chunks: Vec<Chunk>,
    tiles: TileSchedule,
    /// The library's gather indices: per padded column its source.
    padded_sources: Option<IndexBuffer>,
    columns: usize,
    library_rejection: Option<String>,
}

impl GroupedPlan {
    /// Builds the plan of the grouped view with the host `arrays` (the arrays its
    /// [`GroupedView`] was uploaded from; `sources_bound` the columns of the input) for
    /// tables of order `settings.n` in `T`, and uploads its index arrays (one or two
    /// uploads). With a library candidate (module documentation) it probes the library
    /// with one launch of each distinct chunk shape, into `scratch` and the chunk's tables,
    /// and keeps the library only if every launch succeeds and passes the input-precision
    /// guard; [`library_rejection`](Self::library_rejection) says why not otherwise.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`]; [`KernelError::WrongDevice`] for buffers of another device.
    ///
    /// # Panics
    ///
    /// If the arrays are not a valid grouped view (as [`row_to_batch`]), n is not an order
    /// 1 to 441 or not the tables' order, `tables` holds fewer than one matrix per group,
    /// or a library candidate's `scratch` is shorter than [`PlanSettings::size`] states.
    pub fn new<G: Copy + Into<u32>, T: DeviceFloat>(
        device: &mut Device,
        arrays: &GroupedArrays<'_, G>,
        sources_bound: usize,
        settings: &PlanSettings,
        tables: &Tables<T>,
        scratch: &mut TranslationScratch<T>,
    ) -> Result<Self, KernelError> {
        let n = settings.n;
        check_order("GroupedPlan", n);
        assert_eq!(tables.n, n, "GroupedPlan: tables of order {}", tables.n);
        // Validates the rows and batches (the map itself lives in the view).
        row_to_batch(arrays, sources_bound);
        let batch_offsets = arrays.batch_offsets;
        let ngroups = batch_offsets.len() - 1;
        assert!(
            tables.count >= ngroups,
            "GroupedPlan: {} tables for {ngroups} groups",
            tables.count
        );
        let mut library_rejection = None;
        let library = match settings.library_layout(device.backend(), T::FLOAT, batch_offsets) {
            Some(layout) => {
                check_owners(
                    device,
                    &[tables.compact().device(), scratch.x.as_slice().device()],
                )?;
                assert!(
                    scratch.len() >= layout.columns * n,
                    "GroupedPlan: a scratch of {} values for {} columns of {n}",
                    scratch.len(),
                    layout.columns
                );
                library_rejection = probe_library(
                    device,
                    &layout.chunks,
                    (n, settings.orientation),
                    tables,
                    scratch,
                );
                library_rejection.is_none().then_some(layout)
            }
            None => None,
        };
        let (gemm, layout, padded_sources) = match library {
            Some(layout) => {
                // Each group's columns of the chunk, then copies of a valid source up to the
                // chunk's widest batch.
                let mut sources = Vec::with_capacity(layout.chunks.last().unwrap().sources.end);
                for chunk in &layout.chunks {
                    let filler = arrays.batch_sources[chunk.entries.start];
                    for g in chunk.first_group..chunk.first_group + chunk.groups {
                        let lo = (batch_offsets[g] as usize).max(chunk.entries.start);
                        let hi = (batch_offsets[g + 1] as usize).min(chunk.entries.end);
                        sources.extend_from_slice(&arrays.batch_sources[lo..hi]);
                        sources.extend(std::iter::repeat_n(filler, chunk.padded - (hi - lo)));
                    }
                    debug_assert_eq!(sources.len(), chunk.sources.end);
                }
                let padded = device.upload_indices(&sources)?;
                (Gemm::Library, layout, Some(padded))
            }
            None => (
                Gemm::HandWritten(settings.layout),
                compact_layout(batch_offsets, settings.max_columns(T::FLOAT)),
                None,
            ),
        };
        let widths = layout.chunks.iter().map(|c| c.width).collect();
        let tiles = TileSchedule::upload(device, settings.layout.tile(), &layout.runs, widths)?;
        Ok(Self {
            n,
            nrows: arrays.row_offsets.len() - 1,
            ngroups,
            len: arrays.batch_sources.len(),
            sources_bound,
            precision: T::FLOAT,
            gemm,
            orientation: settings.orientation,
            chunks: layout.chunks,
            tiles,
            padded_sources,
            columns: layout.columns,
            library_rejection,
        })
    }

    /// The GEMM the plan runs.
    pub fn gemm(&self) -> Gemm {
        self.gemm
    }

    /// The layout of X and Y in the scratch.
    pub fn orientation(&self) -> Orientation {
        self.orientation
    }

    /// Why a library candidate runs the hand-written kernel: the probe's error, or the
    /// guard's finding; `None` if the library runs or was not a candidate.
    pub fn library_rejection(&self) -> Option<&str> {
        self.library_rejection.as_deref()
    }

    /// The number of chunks with an entry: each is three launches.
    pub fn nchunks(&self) -> usize {
        self.chunks.len()
    }

    /// The most columns of X and Y a chunk uses: the scratch needs `columns() · n`
    /// values each.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the view has no pair: a level call launches nothing.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of tiles of the hand-written kernel.
    pub fn ntiles(&self) -> usize {
        self.tiles.len()
    }

    /// The columns the GEMMs of a level call compute, over every chunk: the pairs with the
    /// hand-written kernel; with the library also its padding columns, whose products are
    /// never read (the useful share is [`len`](Self::len) over this).
    pub fn gemm_columns(&self) -> usize {
        self.chunks.iter().map(|c| c.width).sum()
    }
}

/// Probes the library once per distinct (groups, columns per group) shape of `chunks`, at
/// the first chunk of each shape, into the scratch; `None` if every probe launched and
/// passed the input-precision guard, else why not.
fn probe_library<T: DeviceFloat>(
    device: &mut Device,
    chunks: &[Chunk],
    (n, orientation): (usize, Orientation),
    tables: &Tables<T>,
    scratch: &mut TranslationScratch<T>,
) -> Option<String> {
    if !tables.has_library_copy() {
        return Some("the tables have no library copy".into());
    }
    let mut probed: Vec<(usize, usize)> = Vec::new();
    for chunk in chunks {
        let shape = (chunk.groups, chunk.padded);
        if probed.contains(&shape) {
            continue;
        }
        probed.push(shape);
        let (rhs, stride) = tables.library_run(chunk.first_group, chunk.groups);
        let values = chunk.width * n;
        if let Err(reason) = library_gemm(
            device,
            (chunk.groups, chunk.padded, n),
            orientation,
            (rhs, stride),
            scratch.x.slice(..values),
            scratch.y.slice_mut(..values),
        ) {
            return Some(format!(
                "[{}, {}, {n}] (groups, columns per group, order): {reason}",
                chunk.groups, chunk.padded
            ));
        }
    }
    None
}

/// The input and output of a level call: in different buffers (M2L: multipoles to
/// locals), or at disjoint ranges of one buffer (M2M and L2L: one level of the multipoles
/// or locals to another).
#[derive(Debug)]
pub enum Operands<'a, T: DeviceFloat> {
    /// The input and output in different buffers.
    Separate {
        /// The input columns, those the sources address.
        input: DeviceSlice<'a, T>,
        /// The output columns, one per row.
        output: DeviceSliceMut<'a, T>,
    },
    /// The input and output at two disjoint ranges of one buffer.
    Shared {
        /// The buffer.
        buffer: &'a mut DeviceBuffer<T>,
        /// The range of the input columns.
        input: Range<usize>,
        /// The range of the output columns.
        output: Range<usize>,
    },
}

impl<T: DeviceFloat> Operands<'_, T> {
    /// The lengths of the input and output.
    fn lens(&self) -> (usize, usize) {
        match self {
            Self::Separate { input, output } => (input.len(), output.len()),
            Self::Shared { input, output, .. } => (input.len(), output.len()),
        }
    }

    /// The input, read-only.
    fn input(&self) -> DeviceSlice<'_, T> {
        match self {
            Self::Separate { input, .. } => *input,
            Self::Shared { buffer, input, .. } => buffer.slice(input.clone()),
        }
    }

    /// The output, writable.
    fn output(&mut self) -> DeviceSliceMut<'_, T> {
        match self {
            Self::Separate { output, .. } => output.reborrow(),
            Self::Shared { buffer, output, .. } => buffer.slice_mut(output.clone()),
        }
    }
}

/// One level call of a grouped translation (module documentation): adds, for every
/// target t of `view`, A_g input[:, s] of each entry (s, g) of its row into output[:, t],
/// in row order, through the chunks of `plan`, the tables `tables` (matrix g for group g)
/// and `scratch`. Three launches per chunk; nothing for a view without pairs. Allocates
/// nothing.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the device cannot run the plan's hand-written
/// layout; [`KernelError::Device`] if the library launch fails (it passed its probe at
/// build).
///
/// # Panics
///
/// If `plan` was not built for `view` (rows, groups, pairs or source bound differ) or for
/// `T`, the tables are not of order n or hold fewer than one matrix per group (or, for a
/// library plan, have no library copy), the input or output does not hold n values per
/// column of the view, `scratch` is shorter than the plan needs, the two ranges of
/// [`Operands::Shared`] overlap, or (debug builds) an [`Accumulate::Scatter`] chunk repeats
/// a target.
pub fn grouped<T: DeviceFloat>(
    device: &mut Device,
    plan: &GroupedPlan,
    view: &GroupedView,
    accumulate: Accumulate,
    tables: &Tables<T>,
    operands: Operands<'_, T>,
    scratch: &mut TranslationScratch<T>,
) -> Result<(), KernelError> {
    level_call(
        device,
        (plan, view, accumulate),
        tables,
        operands,
        scratch,
        &Stage::ALL,
    )
}

/// One launch of each chunk of a level call (module documentation, "Structure").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// 1. The gather of the chunk's inputs into X.
    Gather,
    /// 2. The grouped GEMM into Y.
    Gemm,
    /// 3. The reduction or scatter-add of Y into the output.
    Accumulate,
}

impl Stage {
    /// The three stages in the order of a level call.
    pub const ALL: [Self; 3] = [Self::Gather, Self::Gemm, Self::Accumulate];
}

/// For profiling: the launch of `stage` for every chunk of a level call of [`grouped`]
/// (same arguments, same checks), one launch per chunk; the other stages do not run, so
/// the output means nothing unless the three run in a level call's order. Allocates
/// nothing.
///
/// # Errors
///
/// As [`grouped`].
///
/// # Panics
///
/// As [`grouped`].
#[allow(clippy::too_many_arguments)]
pub fn grouped_stage<T: DeviceFloat>(
    device: &mut Device,
    plan: &GroupedPlan,
    view: &GroupedView,
    accumulate: Accumulate,
    tables: &Tables<T>,
    operands: Operands<'_, T>,
    scratch: &mut TranslationScratch<T>,
    stage: Stage,
) -> Result<(), KernelError> {
    level_call(
        device,
        (plan, view, accumulate),
        tables,
        operands,
        scratch,
        &[stage],
    )
}

/// [`grouped`] with only the launches of `stages`.
fn level_call<T: DeviceFloat>(
    device: &mut Device,
    (plan, view, accumulate): (&GroupedPlan, &GroupedView, Accumulate),
    tables: &Tables<T>,
    mut operands: Operands<'_, T>,
    scratch: &mut TranslationScratch<T>,
    stages: &[Stage],
) -> Result<(), KernelError> {
    let n = plan.n;
    assert_eq!(
        T::FLOAT,
        plan.precision,
        "grouped: a plan of another precision"
    );
    assert!(
        view.nrows() == plan.nrows
            && view.ngroups() == plan.ngroups
            && view.len() == plan.len
            && view.sources_bound() == plan.sources_bound,
        "grouped: the plan was built for another view"
    );
    let (input_len, output_len) = operands.lens();
    assert_eq!(
        input_len,
        plan.sources_bound * n,
        "grouped: the input holds {input_len} values, not {} columns of {n}",
        plan.sources_bound
    );
    assert_eq!(
        output_len,
        plan.nrows * n,
        "grouped: the output holds {output_len} values, not {} columns of {n}",
        plan.nrows
    );
    if let Operands::Shared { input, output, .. } = &operands {
        assert!(
            input.end <= output.start || output.end <= input.start,
            "grouped: the input {input:?} and output {output:?} overlap"
        );
    }
    assert!(
        tables.n == n && tables.count >= plan.ngroups,
        "grouped: {} tables of order {} for {} groups of order {n}",
        tables.count,
        tables.n,
        plan.ngroups
    );
    assert!(
        plan.gemm != Gemm::Library || tables.has_library_copy(),
        "grouped: a library plan needs the library copy of the tables"
    );
    assert!(
        scratch.len() >= plan.columns * n,
        "grouped: a scratch of {} values for {} columns of {n}",
        scratch.len(),
        plan.columns
    );
    check_owners(
        device,
        &[
            tables.compact().device(),
            operands.input().device(),
            scratch.x.as_slice().device(),
        ],
    )?;
    if let Gemm::HandWritten(layout) = plan.gemm {
        layout.check(device.info())?;
    }
    if plan.is_empty() {
        return Ok(());
    }
    let coefficient_major = plan.orientation == Orientation::CoefficientMajor;
    for (index, chunk) in plan.chunks.iter().enumerate() {
        let values = chunk.width * n;
        // 1. Gather.
        if stages.contains(&Stage::Gather) {
            let sources = match &plan.padded_sources {
                Some(padded) => padded.slice(chunk.sources.clone()),
                None => view.batch_sources().slice(chunk.entries.clone()),
            };
            if coefficient_major {
                gather_coefficients(
                    device,
                    n,
                    operands.input(),
                    sources,
                    chunk.block(),
                    scratch.x.slice_mut(..values),
                )?;
            } else {
                gather_columns(
                    device,
                    n,
                    operands.input(),
                    sources,
                    scratch.x.slice_mut(..values),
                )?;
            }
        }
        // 2. The grouped GEMM.
        match plan.gemm {
            _ if !stages.contains(&Stage::Gemm) => {}
            Gemm::HandWritten(layout) => gemm_segment(
                device,
                (layout, plan.orientation),
                n,
                tables.compact(),
                &plan.tiles,
                index,
                scratch.x.slice(..values),
                scratch.y.slice_mut(..values),
            )?,
            Gemm::Library => library_gemm(
                device,
                (chunk.groups, chunk.padded, n),
                plan.orientation,
                tables.library_run(chunk.first_group, chunk.groups),
                scratch.x.slice(..values),
                scratch.y.slice_mut(..values),
            )
            .map_err(|reason| KernelError::Device { reason })?,
        }
        // 3. Accumulate.
        if !stages.contains(&Stage::Accumulate) {
            continue;
        }
        match (accumulate, chunk.padded, plan.orientation) {
            (Accumulate::Scatter, 0, Orientation::BoxMajor) => scatter_add_columns(
                device,
                n,
                scratch.y.slice(..values),
                view.batch_targets().slice(chunk.entries.clone()),
                operands.output(),
            )?,
            _ => reduce_rows(
                device,
                (n, plan.orientation),
                view,
                chunk,
                scratch.y.slice(..values),
                operands.output(),
            )?,
        }
    }
    Ok(())
}

/// The schedule of structure (A) for one grouped view, built once ([`per_group`]): per
/// group its batch range and, for the hand-written kernel, its tiles (one segment per
/// group, columns from 0).
#[derive(Debug)]
pub struct PerGroupPlan {
    n: usize,
    nrows: usize,
    len: usize,
    sources_bound: usize,
    precision: Precision,
    gemm: Gemm,
    batch_offsets: Vec<u32>,
    tiles: TileSchedule,
    widest: usize,
}

impl PerGroupPlan {
    /// Builds structure (A) of the grouped view with the host `arrays` (`sources_bound` the
    /// columns of the input) for tables of order n in `T` with `gemm`, and uploads its tile
    /// schedule (one upload). For [`Gemm::Library`] it probes each group's shape once, into
    /// `scratch`.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`]; [`KernelError::UnsupportedLayout`] if the device cannot run
    /// the hand-written layout, or for the library if the tables have no library copy or
    /// a probe fails (the library's reason, or the input-precision guard's).
    ///
    /// # Panics
    ///
    /// As [`GroupedPlan::new`]; also if `scratch` holds fewer than n values per column of
    /// the widest group.
    pub fn new<G: Copy + Into<u32>, T: DeviceFloat>(
        device: &mut Device,
        arrays: &GroupedArrays<'_, G>,
        sources_bound: usize,
        n: usize,
        gemm: Gemm,
        tables: &Tables<T>,
        scratch: &mut TranslationScratch<T>,
    ) -> Result<Self, KernelError> {
        check_order("PerGroupPlan", n);
        assert_eq!(tables.n, n, "PerGroupPlan: tables of order {}", tables.n);
        row_to_batch(arrays, sources_bound);
        let batch_offsets = arrays.batch_offsets;
        let ngroups = batch_offsets.len() - 1;
        assert!(
            tables.count >= ngroups,
            "PerGroupPlan: {} tables for {ngroups} groups",
            tables.count
        );
        let widths: Vec<usize> = batch_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .collect();
        let widest = widths.iter().copied().max().unwrap_or(0);
        assert!(
            scratch.len() >= widest * n,
            "PerGroupPlan: a scratch of {} values for {widest} columns of {n}",
            scratch.len()
        );
        let refuse = |reason: String| KernelError::UnsupportedLayout {
            layout: gemm.to_string(),
            reason,
        };
        let (width, runs) = match gemm {
            Gemm::HandWritten(layout) => {
                layout.check(device.info())?;
                let runs: Vec<Vec<(usize, usize, usize)>> = widths
                    .iter()
                    .enumerate()
                    .map(|(g, &k)| vec![(g, 0, k)])
                    .collect();
                (layout.tile(), runs)
            }
            Gemm::Library => {
                // One chunk per group with a column, from column 0, for the probe.
                let chunks: Vec<Chunk> = widths
                    .iter()
                    .enumerate()
                    .filter(|&(_, &k)| k > 0)
                    .map(|(g, &k)| Chunk {
                        entries: 0..k,
                        width: k,
                        padded: k,
                        first_group: g,
                        groups: 1,
                        sources: 0..k,
                    })
                    .collect();
                if let Some(reason) =
                    probe_library(device, &chunks, (n, Orientation::BoxMajor), tables, scratch)
                {
                    return Err(refuse(reason));
                }
                (1, vec![Vec::new(); ngroups])
            }
        };
        let tiles = TileSchedule::upload(device, width, &runs, widths)?;
        Ok(Self {
            n,
            nrows: arrays.row_offsets.len() - 1,
            len: arrays.batch_sources.len(),
            sources_bound,
            precision: T::FLOAT,
            gemm,
            batch_offsets: batch_offsets.to_vec(),
            tiles,
            widest,
        })
    }

    /// The GEMM it runs.
    pub fn gemm(&self) -> Gemm {
        self.gemm
    }

    /// The columns of the widest group: the scratch needs `columns() · n` values each.
    pub fn columns(&self) -> usize {
        self.widest
    }

    /// The groups with a column: each is three launches.
    pub fn nonempty_groups(&self) -> usize {
        self.batch_offsets
            .windows(2)
            .filter(|w| w[1] > w[0])
            .count()
    }
}

/// Structure (A) of device-path.md §6.4 (module documentation), the test reference of
/// [`grouped`]: for each group g in group order with a column, one gather of its batch's
/// inputs, one GEMM with table g (the plan's) and one scatter-add of the products into
/// the batch's targets, which are distinct within a batch. Each target therefore meets its
/// entries in row order and adds each product once, as in [`grouped`]. Three launches per
/// group with a column. Allocates nothing.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::Device`] if a library launch fails (it passed its probe at build).
///
/// # Panics
///
/// As [`grouped`] (a batch that repeats a target panics in debug builds).
pub fn per_group<T: DeviceFloat>(
    device: &mut Device,
    plan: &PerGroupPlan,
    view: &GroupedView,
    tables: &Tables<T>,
    mut operands: Operands<'_, T>,
    scratch: &mut TranslationScratch<T>,
) -> Result<(), KernelError> {
    let n = plan.n;
    assert_eq!(
        T::FLOAT,
        plan.precision,
        "per_group: a plan of another precision"
    );
    assert!(
        view.nrows() == plan.nrows
            && view.ngroups() + 1 == plan.batch_offsets.len()
            && view.len() == plan.len
            && view.sources_bound() == plan.sources_bound,
        "per_group: the plan was built for another view"
    );
    let (input_len, output_len) = operands.lens();
    assert!(
        input_len == plan.sources_bound * n && output_len == plan.nrows * n,
        "per_group: the input or output does not hold n = {n} values per column"
    );
    assert!(
        tables.n == n && tables.count + 1 >= plan.batch_offsets.len(),
        "per_group: the tables do not fit the plan"
    );
    assert!(
        scratch.len() >= plan.widest * n,
        "per_group: a scratch of {} values for {} columns of {n}",
        scratch.len(),
        plan.widest
    );
    check_owners(
        device,
        &[
            tables.compact().device(),
            operands.input().device(),
            scratch.x.as_slice().device(),
        ],
    )?;
    for (g, w) in plan.batch_offsets.windows(2).enumerate() {
        let batch = w[0] as usize..w[1] as usize;
        let values = batch.len() * n;
        if values == 0 {
            continue;
        }
        gather_columns(
            device,
            n,
            operands.input(),
            view.batch_sources().slice(batch.clone()),
            scratch.x.slice_mut(..values),
        )?;
        match plan.gemm {
            Gemm::HandWritten(layout) => gemm_segment(
                device,
                (layout, Orientation::BoxMajor),
                n,
                tables.compact(),
                &plan.tiles,
                g,
                scratch.x.slice(..values),
                scratch.y.slice_mut(..values),
            )?,
            Gemm::Library => library_gemm(
                device,
                (1, batch.len(), n),
                Orientation::BoxMajor,
                tables.library_run(g, 1),
                scratch.x.slice(..values),
                scratch.y.slice_mut(..values),
            )
            .map_err(|reason| KernelError::Device { reason })?,
        }
        scatter_add_columns(
            device,
            n,
            scratch.y.slice(..values),
            view.batch_targets().slice(batch),
            operands.output(),
        )?;
    }
    Ok(())
}

/// One GEMM over every tile of `schedule`: y[:, c] = A_g x[:, c] for every column c of
/// group g's tiles, tables `tables` (matrix g at g n², column-major), in `layout`
/// (module documentation, "The hand-written GEMM"). Columns of no tile are not written.
/// One launch; nothing for an empty schedule. Allocates nothing.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the device cannot run `layout`.
///
/// # Panics
///
/// If n is not an order 1 to 441, the schedule was built for another tile width, the
/// tables hold fewer matrices than the schedule's groups, or x or y holds fewer than n
/// values per column of the schedule.
pub fn gemm<T: DeviceFloat>(
    device: &mut Device,
    layout: GemmLayout,
    n: usize,
    tables: DeviceSlice<'_, T>,
    schedule: &TileSchedule,
    x: DeviceSlice<'_, T>,
    y: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    check_order("gemm", n);
    assert_eq!(
        schedule.segments.len(),
        1,
        "gemm: the schedule of one GEMM has one segment"
    );
    layout.check(device.info())?;
    gemm_segment(
        device,
        (layout, Orientation::BoxMajor),
        n,
        tables,
        schedule,
        0,
        x,
        y,
    )
}

/// One library GEMM (module documentation, "The library GEMM") of `groups` batches of `k`
/// columns each: `y[g] = A_g x[g]` for g < `groups`, x and y the row-major `[G, k, n]`
/// arrays (column j of batch g at (g k + j) n) and the tables matrix g at g n², every
/// buffer from its start. One launch, through `cubek-matmul` with
/// `Strategy::MultiLevel(SimpleCyclicCmma)`, with the input-precision guard; for tests and
/// reports (a [`GroupedPlan`] decides at build whether a level call uses it).
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the library rejects the shape or the resolved
/// element types fail the guard, with the reason.
///
/// # Panics
///
/// If n is not an order 1 to 441, `groups` or `k` is 0, a slice does not start at the
/// beginning of its buffer, or a buffer holds fewer values than the shape.
pub fn library<T: DeviceFloat>(
    device: &mut Device,
    (groups, k, n): (usize, usize, usize),
    tables: DeviceSlice<'_, T>,
    x: DeviceSlice<'_, T>,
    y: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    check_order("library", n);
    check_owners(device, &[tables.device(), x.device(), y.device()])?;
    assert!(groups > 0 && k > 0, "library: an empty shape");
    assert!(
        tables.offset() == 0 && x.offset() == 0 && y.offset() == 0,
        "library: the operands must start their buffers"
    );
    assert!(
        tables.len() >= groups * n * n && x.len() >= groups * k * n && y.len() >= groups * k * n,
        "library: the buffers are shorter than the shape [{groups}, {k}, {n}]"
    );
    library_gemm(
        device,
        (groups, k, n),
        Orientation::BoxMajor,
        (tables, n * n),
        x,
        y,
    )
    .map_err(|reason| KernelError::UnsupportedLayout {
        layout: Gemm::Library.to_string(),
        reason,
    })
}

/// The tiles of segment `segment` of `schedule` (module documentation): one launch.
#[allow(clippy::too_many_arguments)]
fn gemm_segment<T: DeviceFloat>(
    device: &mut Device,
    (layout, orientation): (GemmLayout, Orientation),
    n: usize,
    tables: DeviceSlice<'_, T>,
    schedule: &TileSchedule,
    segment: usize,
    x: DeviceSlice<'_, T>,
    y: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    check_owners(
        device,
        &[
            tables.device(),
            schedule.tiles.as_slice().device(),
            x.device(),
            y.device(),
        ],
    )?;
    assert_eq!(
        schedule.width,
        layout.tile(),
        "gemm: a schedule of tiles of {} columns for a layout of {}",
        schedule.width,
        layout.tile()
    );
    let columns = schedule.columns[segment];
    assert!(
        tables.len() >= schedule.groups * n * n,
        "gemm: {} table values for {} groups of order {n}",
        tables.len(),
        schedule.groups
    );
    assert!(
        x.len() >= columns * n && y.len() >= columns * n,
        "gemm: x and y hold {} and {} values, the schedule addresses {columns} columns of {n}",
        x.len(),
        y.len()
    );
    let tiles = schedule.segments[segment].clone();
    if tiles.is_empty() {
        return Ok(());
    }
    let info = device.info().clone();
    let cap = device.units_cap();
    let ((ah, al), (th, tl), (xh, xl), (yh, yl)) = (
        tables.binding(),
        schedule.tiles.as_slice().binding(),
        x.binding(),
        y.binding(),
    );
    let scalars = (
        tables.offset() as u32,
        tiles.start as u32,
        tiles.len() as u32,
        x.offset() as u32,
        y.offset() as u32,
    );
    // Coefficient-major: one block of the segment's columns (module documentation,
    // "Orientation"); box-major ignores the stride.
    let coefficient_major = orientation == Orientation::CoefficientMajor;
    let stride = columns as u32;
    let client = device.client().clone();
    match layout {
        GemmLayout::Cube {
            rows,
            columns,
            per_unit,
        } => {
            let (cubes_x, cubes_y) = cube_grid(&info, tiles.len());
            let tm = n.div_ceil(rows as usize);
            // SAFETY: every handle is a whole buffer with its element count, as
            // `from_raw_parts` requires. The kernel reads tiles 3 (t0 + t) + 0..3 for
            // t < ntiles, within the schedule (segment ranges were built with it). Each
            // tile's group g < schedule.groups and its columns first..first + count lie
            // within the segment's `columns` (built so), so the tables are read at
            // offset + g n² + k n + i < offset + tables.len() (asserted above, i, k < n)
            // and x and y at their offsets plus c n + k (box-major) or k columns + c
            // (coefficient-major) with c < columns and k < n, both below columns · n
            // (asserted within x.len() and y.len()). Rows at or past n and columns past
            // the tile's count are masked; cubes past ntiles (a 2-D grid) do nothing.
            // Distinct units write distinct (row, column) values of y.
            unsafe {
                gemm_cube_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(cubes_x, cubes_y, 1),
                    CubeDim::new_2d(rows, columns),
                    BufferArg::from_raw_parts(ah, al),
                    scalars.0,
                    BufferArg::from_raw_parts(th, tl),
                    scalars.1,
                    scalars.2,
                    BufferArg::from_raw_parts(xh, xl),
                    scalars.3,
                    BufferArg::from_raw_parts(yh, yl),
                    scalars.4,
                    stride,
                    n,
                    rows as usize,
                    tm,
                    per_unit as usize,
                    k_step(tm),
                    coefficient_major,
                );
            }
        }
        GemmLayout::Cpu { block, per_unit } => {
            let units = (cap as usize)
                .min(info.max_units_per_cube.max(1) as usize)
                .min(tiles.len())
                .max(1) as u32;
            // SAFETY: as for the cube layout: the tiles, tables, x and y are read and
            // written within the bounds asserted above, in either orientation. Each unit
            // covers its own contiguous range of tiles, and tiles cover distinct columns,
            // so no two units write one value.
            unsafe {
                gemm_cpu_kernel::launch_unchecked::<T>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(units),
                    BufferArg::from_raw_parts(ah, al),
                    scalars.0,
                    BufferArg::from_raw_parts(th, tl),
                    scalars.1,
                    scalars.2,
                    BufferArg::from_raw_parts(xh, xl),
                    scalars.3,
                    BufferArg::from_raw_parts(yh, yl),
                    scalars.4,
                    stride,
                    n,
                    block as usize,
                    per_unit as usize,
                    k_step(block as usize),
                    coefficient_major,
                );
            }
        }
    }
    device.count_launch();
    Ok(())
}

/// The reduction of one chunk (module documentation): for every row t of the view and
/// every coefficient i, `acc = output[t n + i]`, then for each entry e of row t in row
/// order whose batch position k lies in the chunk, `acc = acc + y[col(e) n + i]`, and the
/// value is stored if an entry was added. col(e) is k − start for a compact chunk, and
/// (g − g₀) k_c + k − max(start, b_g) for a padded one (group g of the entry, g₀ the
/// chunk's first group, k_c its columns per group, b_g the group's first batch position).
/// Coefficient-major ("Orientation"), y is read at (b n + i) w + c for column col(e) =
/// b w + c with blocks of w columns ([`Chunk::block`]). One launch.
fn reduce_rows<T: DeviceFloat>(
    device: &mut Device,
    (n, orientation): (usize, Orientation),
    view: &GroupedView,
    chunk: &Chunk,
    y: DeviceSlice<'_, T>,
    output: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    check_owners(
        device,
        &[view.row_offsets().device(), y.device(), output.device()],
    )?;
    let work = view.nrows() * n;
    if work == 0 {
        return Ok(());
    }
    let grid = device.elementwise_grid(work);
    let (ro, rl) = view.row_offsets().binding();
    let (mh, ml) = view.row_to_batch().binding();
    let (gh, gl) = view.groups().binding();
    let (bh, bl) = view.batch_offsets().binding();
    let (yh, yl) = y.binding();
    let (oh, ol) = output.binding();
    // SAFETY: every handle is a whole buffer with its element count. The kernel reads row
    // offsets t and t + 1 for t < nrows (the view's validated CSR of nrows + 1 offsets),
    // the row-to-batch map and the groups at the row's entries (parallel arrays of the
    // view's pairs), and for a padded chunk the batch offset of the entry's group
    // (groups < ngroups, validated at upload). An entry is added only if its batch
    // position lies in the chunk, whose columns (compact: position − start; padded:
    // (g − g0) k_c + position − max(start, batch offset), with g in the chunk's groups
    // and at most k_c entries of g in the chunk, so below the chunk's width) lie within y
    // (y.len() = width n): box-major at column n + i, coefficient-major at
    // (b n + i) w + c with column = b w + c, b < width / w (w divides the width: the
    // chunk, or its groups' padded batches), both below width n. The output is read and
    // written at its offset plus t n + i < output.len() = nrows n (asserted by
    // `grouped`); each unit owns its items.
    unsafe {
        reduce_kernel::launch_unchecked::<T>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(ro, rl),
            BufferArg::from_raw_parts(mh, ml),
            BufferArg::from_raw_parts(gh, gl),
            BufferArg::from_raw_parts(bh, bl),
            BufferArg::from_raw_parts(yh, yl),
            y.offset() as u32,
            BufferArg::from_raw_parts(oh, ol),
            output.offset() as u32,
            chunk.entries.start as u32,
            chunk.entries.end as u32,
            chunk.padded as u32,
            chunk.first_group as u32,
            chunk.block() as u32,
            work as u32,
            grid.chunk,
            grid.threads(),
            n,
            orientation == Orientation::CoefficientMajor,
        );
    }
    device.count_launch();
    Ok(())
}

/// The library GEMM of `shape` = (groups G, columns per group k, order n) with the
/// tables `(tables, stride)`, matrix g at g · stride of the slice (module documentation,
/// "The library GEMM"), with the input-precision guard. Box-major, Y[g] = X[g] Bᵀ[g] for
/// the row-major [G, k, n] arrays x and y, the tables read as [G, n, n] row-major (B = Aᵀ);
/// coefficient-major ("Orientation"), Y[g] = A[g] X[g] for the row-major [G, n, k] arrays
/// x and y, the tables read as [G, n, n] column-major (A itself). Each operand is bound
/// from its slice's first value: the callers pass slices at offset 0 or, for the tables'
/// library copy, at an aligned offset. One launch; the error is the library's, or the
/// guard's finding.
fn library_gemm<T: DeviceFloat>(
    device: &mut Device,
    (groups, k, n): (usize, usize, usize),
    orientation: Orientation,
    (tables, stride): (DeviceSlice<'_, T>, usize),
    x: DeviceSlice<'_, T>,
    y: DeviceSliceMut<'_, T>,
) -> Result<(), String> {
    debug_assert!(stride >= n * n && tables.len() >= (groups - 1) * stride + n * n);
    debug_assert!(x.len() >= groups * k * n && y.len() >= groups * k * n);
    let elem = T::elem_type_native();
    let strategy: Strategy = multi_level::Strategy::SimpleCyclicCmma(Default::default()).into();
    let binding = |handle, strides: [usize; 3], shape: [usize; 2]| {
        // SAFETY: each handle is restricted to its slice's bytes. Box-major, x and y are
        // [G, k, n] with strides [k n, n, 1] and the tables [G, n, n] with [stride, n, 1];
        // coefficient-major, x and y are [G, n, k] with [n k, k, 1] and the tables
        // [G, n, n] with [stride, 1, n]. Either way x and y are addressed below G k n ≤
        // their lengths and the tables below (G − 1) stride + n² ≤ its length (asserted
        // by the callers), so the library reads and writes within them.
        unsafe {
            TensorBinding::from_raw_parts(
                handle,
                Strides::from(strides),
                Shape::from([groups, shape[0], shape[1]]),
            )
        }
    };
    let (lhs, rhs, out) = match orientation {
        Orientation::BoxMajor => (
            binding(x.byte_range_handle(), [k * n, n, 1], [k, n]),
            binding(tables.byte_range_handle(), [stride, n, 1], [n, n]),
            binding(y.byte_range_handle(), [k * n, n, 1], [k, n]),
        ),
        Orientation::CoefficientMajor => (
            binding(tables.byte_range_handle(), [stride, 1, n], [n, n]),
            binding(x.byte_range_handle(), [n * k, k, 1], [n, k]),
            binding(y.byte_range_handle(), [n * k, k, 1], [n, k]),
        ),
    };
    let mut dtypes = MatmulElems::from_single_dtype(elem);
    let result = launch_ref(
        &strategy,
        device.client(),
        InputBinding::Normal(lhs, elem),
        InputBinding::Normal(rhs, elem),
        out,
        &mut dtypes,
    );
    device.count_launch();
    result.map_err(|e| format!("{e:?}"))?;
    keeps_precision(&dtypes, elem)
}

/// The input-precision guard (module documentation): every stage and register type of
/// the inputs and the accumulator is `elem`.
fn keeps_precision(elems: &MatmulElems, elem: ElemType) -> Result<(), String> {
    let inner = [
        ("lhs stage", elems.lhs_stage),
        ("rhs stage", elems.rhs_stage),
        ("accumulator stage", elems.acc_stage),
        ("lhs register", elems.lhs_register),
        ("rhs register", elems.rhs_register),
        ("accumulator register", elems.acc_register),
    ];
    match inner.iter().find(|(_, t)| *t != elem) {
        None => Ok(()),
        Some((what, t)) => Err(format!(
            "input-precision guard: the {what} type is {t:?}, not {elem:?}"
        )),
    }
}

fn check_owners(device: &Device, owners: &[u64]) -> Result<(), KernelError> {
    owners.iter().try_for_each(|&o| device.check_owner(o))
}

/// One block of the product (module documentation, "Summation order"): rows `row + r ·
/// row_stride` for r < tm and the `cols ≤ tn` columns `col..col + cols` of y = A x, A at
/// `a0` (column-major, order n), x and y with coefficient k of column c at `x0 + c n + k`
/// and `y0 + c n + k`, or with `cm` (coefficient-major, "Orientation") at
/// `x0 + k stride + c` and `y0 + k stride + c`. Each value one accumulator from zero,
/// `acc = fma(A_ik, x_kc, acc)` for k ascending, stored once, in either orientation. Rows
/// at or past n are masked.
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn product_block<F: Float>(
    tables: &[F],
    a0: usize,
    x: &[F],
    x0: usize,
    y: &mut [F],
    y0: usize,
    row: usize,
    row_stride: usize,
    col: usize,
    cols: usize,
    stride: usize,
    #[comptime] n: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] ks: usize,
    #[comptime] cm: bool,
) {
    let mut acc = Array::<F>::new(comptime!(tm * tn));
    #[unroll]
    for q in 0..comptime!(tm * tn) {
        acc[q] = F::new(0.0f32);
    }
    // The k loop in steps of ks, the rest unrolled after: each step loads its x and A
    // values first, then adds their products in k order (the order does not change).
    let full = comptime!(n / ks);
    for s in 0..full {
        product_steps::<F>(
            tables,
            a0,
            x,
            x0,
            &mut acc,
            s * ks,
            row,
            row_stride,
            col,
            cols,
            stride,
            n,
            tm,
            tn,
            ks,
            cm,
        );
    }
    if comptime!(!n.is_multiple_of(ks)) {
        product_steps::<F>(
            tables,
            a0,
            x,
            x0,
            &mut acc,
            full * ks,
            row,
            row_stride,
            col,
            cols,
            stride,
            n,
            tm,
            tn,
            comptime!(n % ks),
            cm,
        );
    }
    #[unroll]
    for r in 0..tm {
        let i = row + r * row_stride;
        if i < n {
            #[unroll]
            for c in 0..tn {
                if c < cols {
                    if comptime!(cm) {
                        y[y0 + i * stride + col + c] = acc[r * tn + c];
                    } else {
                        y[y0 + (col + c) * n + i] = acc[r * tn + c];
                    }
                }
            }
        }
    }
}

/// The k steps per iteration of [`product_block`]'s loop for a unit of at most
/// [`STEP_ROWS`] rows: their loads are issued before their products, so that the unit does
/// not wait on one load per k (the kernel is latency bound when a level has few tiles).
/// With more rows a unit holds enough independent products, and the extra registers cost
/// more than they save (measured on the M3 Max: +10% at p = 3 and 8, −22% at p = 16).
const K_STEP: usize = 4;

/// The most rows per unit that take [`K_STEP`] steps per iteration; above, one.
const STEP_ROWS: usize = 4;

/// The k steps per iteration of a unit of `tm` rows.
fn k_step(tm: usize) -> usize {
    if tm <= STEP_ROWS { K_STEP } else { 1 }
}

/// `steps` consecutive k of [`product_block`] from `k0`: loads the x and A values of every
/// step, then `acc = fma(A_ik, x_kc, acc)` for k = k0, k0 + 1, … in turn.
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn product_steps<F: Float>(
    tables: &[F],
    a0: usize,
    x: &[F],
    x0: usize,
    acc: &mut Array<F>,
    k0: usize,
    row: usize,
    row_stride: usize,
    col: usize,
    cols: usize,
    stride: usize,
    #[comptime] n: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] steps: usize,
    #[comptime] cm: bool,
) {
    let mut xv = Array::<F>::new(comptime!(steps * tn));
    let mut av = Array::<F>::new(comptime!(steps * tm));
    #[unroll]
    for u in 0..steps {
        let k = k0 + u;
        #[unroll]
        for c in 0..tn {
            let mut v = F::new(0.0f32);
            if c < cols {
                if comptime!(cm) {
                    v = x[x0 + k * stride + col + c];
                } else {
                    v = x[x0 + (col + c) * n + k];
                }
            }
            xv[u * tn + c] = v;
        }
        #[unroll]
        for r in 0..tm {
            let i = row + r * row_stride;
            let mut a = F::new(0.0f32);
            if i < n {
                a = tables[a0 + k * n + i];
            }
            av[u * tm + r] = a;
        }
    }
    #[unroll]
    for u in 0..steps {
        #[unroll]
        for r in 0..tm {
            #[unroll]
            for c in 0..tn {
                acc[r * tn + c] = fma(av[u * tm + r], xv[u * tn + c], acc[r * tn + c]);
            }
        }
    }
}

/// The GEMM, cube layout: one cube per tile; unit (u, v) the rows u, u + `rows`, … (tm of
/// them) of the tile's columns v tn onwards.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn gemm_cube_kernel<F: Float>(
    tables: &[F],
    table_offset: u32,
    tiles: &[u32],
    tile_offset: u32,
    ntiles: u32,
    x: &[F],
    x_offset: u32,
    y: &mut [F],
    y_offset: u32,
    stride: u32,
    #[comptime] n: usize,
    #[comptime] rows: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] ks: usize,
    #[comptime] cm: bool,
) {
    let tile = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
    if tile < ntiles as usize {
        let at = 3 * (tile_offset as usize + tile);
        let g = tiles[at] as usize;
        let first = tiles[at + 1] as usize;
        let count = tiles[at + 2] as usize;
        let c0 = UNIT_POS_Y as usize * tn;
        if c0 < count {
            let mut cols = count - c0;
            if cols > tn {
                cols = tn;
            }
            product_block::<F>(
                tables,
                table_offset as usize + g * comptime!(n * n),
                x,
                x_offset as usize,
                y,
                y_offset as usize,
                UNIT_POS_X as usize,
                rows,
                first + c0,
                cols,
                stride as usize,
                n,
                tm,
                tn,
                ks,
                cm,
            );
        }
    }
}

/// The GEMM, CPU layout: one cube, each unit a contiguous range of tiles, each tile in
/// blocks of `block` rows.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn gemm_cpu_kernel<F: Float>(
    tables: &[F],
    table_offset: u32,
    tiles: &[u32],
    tile_offset: u32,
    ntiles: u32,
    x: &[F],
    x_offset: u32,
    y: &mut [F],
    y_offset: u32,
    stride: u32,
    #[comptime] n: usize,
    #[comptime] block: usize,
    #[comptime] tn: usize,
    #[comptime] ks: usize,
    #[comptime] cm: bool,
) {
    let (first_tile, count_tiles) = unit_rows(ntiles as usize);
    for tile in first_tile..first_tile + count_tiles {
        let at = 3 * (tile_offset as usize + tile);
        let g = tiles[at] as usize;
        let first = tiles[at + 1] as usize;
        let count = tiles[at + 2] as usize;
        let mut row = 0usize;
        while row < n {
            product_block::<F>(
                tables,
                table_offset as usize + g * comptime!(n * n),
                x,
                x_offset as usize,
                y,
                y_offset as usize,
                row,
                1usize,
                first,
                count,
                stride as usize,
                n,
                block,
                tn,
                ks,
                cm,
            );
            row += block;
        }
    }
}

/// The reduction (`reduce_rows`): items e < work = nrows n, in blocks of `chunk` per unit
/// and stride.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn reduce_kernel<F: Float>(
    row_offsets: &[u32],
    row_to_batch: &[u32],
    groups: &[u32],
    batch_offsets: &[u32],
    y: &[F],
    y_offset: u32,
    output: &mut [F],
    output_offset: u32,
    k_first: u32,
    k_end: u32,
    padded: u32,
    first_group: u32,
    block: u32,
    work: u32,
    chunk: u32,
    threads: u32,
    #[comptime] n: usize,
    #[comptime] cm: bool,
) {
    let (k_first, k_end, padded) = (k_first as usize, k_end as usize, padded as usize);
    let block = block as usize;
    let (work, chunk) = (work as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < work {
        let mut end = start + chunk;
        if end > work {
            end = work;
        }
        for item in start..end {
            let t = item / n;
            let i = item % n;
            let out = output_offset as usize + item;
            let mut acc = output[out];
            let mut added = false;
            for e in row_offsets[t] as usize..row_offsets[t + 1] as usize {
                let k = row_to_batch[e] as usize;
                if k >= k_first && k < k_end {
                    let mut column = k - k_first;
                    if padded > 0 {
                        let g = groups[e] as usize;
                        let mut first = batch_offsets[g] as usize;
                        if first < k_first {
                            first = k_first;
                        }
                        column = (g - first_group as usize) * padded + k - first;
                    }
                    if comptime!(cm) {
                        let b = column / block;
                        acc += y[y_offset as usize + (b * n + i) * block + column - b * block];
                    } else {
                        acc += y[y_offset as usize + column * n + i];
                    }
                    added = true;
                }
            }
            if added {
                output[out] = acc;
            }
        }
        start += stride;
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
    fn default_layouts_fit_the_order() {
        for n in [1, 4, 16, 81, 169, 441] {
            let cpu = GemmLayout::default_for(&info(BackendKind::Cpu), n);
            assert_eq!(
                cpu,
                GemmLayout::Cpu {
                    block: 8.min(n as u32),
                    per_unit: 4
                }
            );
            let gpu = GemmLayout::default_for(&info(BackendKind::Metal), n);
            let GemmLayout::Cube {
                rows,
                columns,
                per_unit,
            } = gpu
            else {
                panic!("{gpu}")
            };
            assert!(rows as usize >= n.min(32) && rows * columns == 64 && per_unit == 4);
            gpu.check(&info(BackendKind::Metal)).unwrap();
        }
    }

    #[test]
    fn tiles_cover_each_group_in_order() {
        // Groups of 0, 5, 1 and 9 columns at offsets 0, 0, 5, 6; tiles of 4.
        let runs = vec![vec![(0, 0, 0), (1, 0, 5), (2, 5, 1), (3, 6, 9)]];
        let (tiles, segments) = tiles_of(4, &runs);
        assert_eq!(segments, vec![0..6]);
        assert_eq!(
            tiles,
            vec![
                1, 0, 4, 1, 4, 1, 2, 5, 1, 3, 6, 4, 3, 10, 4, 3, 14, 1 //
            ]
        );
    }

    #[test]
    fn compact_chunks_cut_batches_at_the_budget() {
        // 3 groups of 4, 0 and 7 columns; chunks of at most 5 columns.
        let layout = compact_layout(&[0, 4, 4, 11], 5);
        let entries: Vec<_> = layout.chunks.iter().map(|c| c.entries.clone()).collect();
        assert_eq!(entries, vec![0..5, 5..10, 10..11]);
        assert_eq!(layout.runs[0], vec![(0, 0, 4), (2, 4, 1)]);
        assert_eq!(layout.runs[1], vec![(2, 0, 5)]);
        assert_eq!(layout.runs[2], vec![(2, 0, 1)]);
        assert_eq!(layout.columns, 5);
        assert!(compact_layout(&[0, 0, 0], 5).chunks.is_empty());
    }

    #[test]
    fn library_candidates_follow_the_rule() {
        let settings = |n, policy| PlanSettings {
            n,
            layout: GemmLayout::Cpu {
                block: 8,
                per_unit: 4,
            },
            policy,
            budget: DEFAULT_SCRATCH_BYTES,
            orientation: Orientation::BoxMajor,
        };
        let offsets = [0, 3, 3, 10];
        let s = settings(81, GemmPolicy::Auto);
        assert!(s.library_candidate(BackendKind::Metal, Precision::F32));
        assert!(!s.library_candidate(BackendKind::Metal, Precision::F64));
        assert!(!s.library_candidate(BackendKind::Cpu, Precision::F32));
        assert!(
            !settings(64, GemmPolicy::Auto).library_candidate(BackendKind::Metal, Precision::F32)
        );
        assert!(
            !settings(81, GemmPolicy::HandWritten)
                .library_candidate(BackendKind::Metal, Precision::F32)
        );
        // One chunk of the three groups padded to 7 columns each.
        let layout = s
            .library_layout(BackendKind::Metal, Precision::F32, &offsets)
            .unwrap();
        assert_eq!(layout.chunks.len(), 1);
        assert_eq!(layout.columns, 21);
        assert!(
            s.library_layout(BackendKind::Metal, Precision::F32, &[0, 0, 0])
                .is_none()
        );
        let size = s.size(BackendKind::Metal, Precision::F32, &offsets);
        assert_eq!(size.columns, 21);
        assert_eq!(size.bytes, 4 * (3 * 3 + 21));
    }

    /// The library's chunks: contiguous runs of groups from a group with a column to the
    /// last with one, padded to the run's widest batch within the budget; a group wider
    /// than the budget alone, in pieces.
    #[test]
    fn library_chunks_pad_runs_of_groups() {
        // Groups of 0, 4, 2, 0, 3, 0, 9 and 0 columns.
        let offsets = [0, 0, 4, 6, 6, 9, 9, 18, 18];
        // (entries, first group, groups, columns per group, padded sources) per chunk.
        type Summary = (Range<usize>, usize, usize, usize, Range<usize>);
        let summary = |max: usize| -> Vec<Summary> {
            library_layout(&offsets, max)
                .chunks
                .into_iter()
                .map(|c| (c.entries, c.first_group, c.groups, c.padded, c.sources))
                .collect()
        };
        // Everything fits: groups 1 to 6, six groups of 9 columns.
        assert_eq!(summary(100), vec![(0..18, 1, 6, 9, 0..54)]);
        // At most 16 columns: groups 1 to 4 (4 groups of 4); group 6 alone (9).
        assert_eq!(
            summary(16),
            vec![(0..9, 1, 4, 4, 0..16), (9..18, 6, 1, 9, 16..25)]
        );
        // At most 4 columns: group 1 alone; group 2 alone (groups 2 to 4 would be three
        // groups of 3); group 4 alone; group 6, wider than 4, in pieces of 4, 4 and 1.
        assert_eq!(
            summary(4),
            vec![
                (0..4, 1, 1, 4, 0..4),
                (4..6, 2, 1, 2, 4..6),
                (6..9, 4, 1, 3, 6..9),
                (9..13, 6, 1, 4, 9..13),
                (13..17, 6, 1, 4, 13..17),
                (17..18, 6, 1, 1, 17..18),
            ]
        );
        let layout = library_layout(&offsets, 4);
        assert_eq!(layout.columns, 4);
        assert!(layout.runs.iter().all(Vec::is_empty));
        assert!(library_layout(&[0, 0, 0], 4).chunks.is_empty());
    }

    #[test]
    fn library_strides_align_every_matrix() {
        assert_eq!(library_stride(81, Precision::F32), 6592);
        assert_eq!(library_stride(81, Precision::F64), 6592);
        assert_eq!(library_stride(16, Precision::F32), 256);
        assert_eq!(library_stride(1, Precision::F64), 32);
        for n in [1, 4, 81, 169, 441] {
            for precision in [Precision::F32, Precision::F64] {
                let bytes = library_stride(n, precision) * value_bytes(precision);
                assert_eq!(bytes % LIBRARY_TABLE_ALIGNMENT, 0);
                assert!(library_stride(n, precision) >= n * n);
            }
        }
    }

    /// The guard rejects a strategy whose resolved inputs or accumulator lose precision:
    /// TF32 or F16 stage or register types for f32 data.
    #[test]
    fn the_input_precision_guard_rejects_lower_precisions() {
        let f32_elem = f32::elem_type_native();
        let all = MatmulElems::from_single_dtype(f32_elem);
        assert_eq!(keeps_precision(&all, f32_elem), Ok(()));
        let tf32 = ElemType::Float(cubecl::ir::FloatKind::TF32);
        let f16 = ElemType::Float(cubecl::ir::FloatKind::F16);
        for lowered in [
            MatmulElems {
                lhs_stage: tf32,
                rhs_stage: tf32,
                ..all.clone()
            },
            MatmulElems {
                lhs_register: tf32,
                ..all.clone()
            },
            MatmulElems {
                rhs_register: f16,
                ..all.clone()
            },
            MatmulElems {
                acc_register: f16,
                ..all.clone()
            },
        ] {
            assert!(keeps_precision(&lowered, f32_elem).is_err(), "{lowered:?}");
        }
    }
}
