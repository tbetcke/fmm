//! The grouped translations with dense tables (Phase 4 T8, C4.4; T9 adds dense M2L): one
//! level's M2M, L2L or M2L as gathers, a grouped GEMM and a reduction or scatter-add
//! (docs/design/device-path.md §6.4, §6.5).
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
//! | M2L (T9) | boxes on l | V-list sources on l | offsets | [`Rows`](Accumulate::Rows) |
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
//! every target meets its entries in row order whichever chunks they fall in; the
//! result does not depend on the budget (bit for bit).
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
//! requirement 6; design §6.5). The library needs one shape per launch, so the batches are
//! padded to the largest, k_max columns each: X and Y are [G, k_max, n] row-major
//! (strides [k_max n, n, 1]) and the tables [G, n, n] (strides [n², n, 1]), one batched
//! launch for the whole view (numpy-style batches, device-path.md F22). It is used only
//! where, at build ([`GroupedPlan::new`]), decided by the shape alone:
//! - the padded X and Y fit in the scratch budget, so the view is one chunk (the tables
//!   are bound whole, from group 0; wgpu binds no unaligned offset);
//! - a probe launch of that shape succeeds (`Unavailable` and every other setup error
//!   send the view to the hand-written kernel);
//! - **the input-precision guard**: the probe's resolved `MatmulElems` keep T for the
//!   stage and register types of both inputs and of the accumulator. A strategy that
//!   rounds f32 inputs to TF32, F16 or BF16 (as the accelerated routines do on a backend
//!   that registers TF32, CUDA) is never used (docs/phase4/README.md, "M2L strategies").
//!
//! Otherwise the hand-written kernel runs. The choice is fixed per view at build and
//! reported ([`GroupedPlan::gemm`], [`GroupedPlan::library_rejection`]). The library sums
//! in its own tiles' order, fixed for a shape, device and driver (device-path.md §9.1); the
//! padding columns gather a valid source and their products are never read.
//!
//! # Launch
//!
//! [`GroupedPlan::new`] validates the host arrays and builds the chunks, the tile schedule
//! and (for the library) the padded gather indices, uploading each once.
//! [`TranslationScratch`] holds X and Y, allocated once for every view. [`grouped`] runs a
//! level call and allocates nothing; [`gemm`] runs one GEMM over a [`TileSchedule`], for
//! tests and the per-group reference (structure (A) of the design).

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
use crate::movement::{gather_columns, scatter_add_columns};
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
    /// The padded layout of the library: the columns per group, k_max (0: compact,
    /// column c of the chunk is entry `entries.start + c`).
    padded: usize,
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

/// The columns of a view's widest group, k_max.
fn widest(batch_offsets: &[u32]) -> usize {
    batch_offsets
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0)
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
    /// p ≥ 8, a GPU backend (module documentation).
    fn library_candidate(&self, backend: BackendKind, precision: Precision) -> bool {
        self.policy == GemmPolicy::Auto
            && precision == Precision::F32
            && self.n >= 81
            && backend.is_gpu()
    }

    /// The padded width of a library chunk over `batch_offsets`, if the policy allows the
    /// library and it fits the budget as one chunk.
    fn library_width(
        &self,
        backend: BackendKind,
        precision: Precision,
        batch_offsets: &[u32],
    ) -> Option<usize> {
        let width = (batch_offsets.len() - 1) * widest(batch_offsets);
        (self.library_candidate(backend, precision)
            && width > 0
            && width <= self.max_columns(precision))
        .then_some(width)
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
        let library = self.library_width(backend, precision, batch_offsets);
        PlanSize {
            bytes: Device::buffer_bytes::<u32>(tiles.len())
                + library.map_or(0, Device::buffer_bytes::<u32>),
            columns: compact.columns.max(library.unwrap_or(0)),
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
    /// with one launch of the view's padded shape, into `scratch` and the first groups
    /// of `tables`, and keeps the library only if the launch succeeds and passes the
    /// input-precision guard; [`library_rejection`](Self::library_rejection) says why not
    /// otherwise.
    ///
    /// # Errors
    ///
    /// As [`Device::upload`]; [`KernelError::WrongDevice`] for buffers of another device.
    ///
    /// # Panics
    ///
    /// If the arrays are not a valid grouped view (as [`row_to_batch`]), n is not an order
    /// 1 to 441, or a library candidate's `scratch` is shorter than
    /// [`PlanSettings::size`] states or `tables` holds fewer than one matrix per group.
    pub fn new<G: Copy + Into<u32>, T: DeviceFloat>(
        device: &mut Device,
        arrays: &GroupedArrays<'_, G>,
        sources_bound: usize,
        settings: &PlanSettings,
        tables: DeviceSlice<'_, T>,
        scratch: &mut TranslationScratch<T>,
    ) -> Result<Self, KernelError> {
        let n = settings.n;
        check_order("GroupedPlan", n);
        // Validates the rows and batches (the map itself lives in the view).
        row_to_batch(arrays, sources_bound);
        let batch_offsets = arrays.batch_offsets;
        let ngroups = batch_offsets.len() - 1;
        let compact = compact_layout(batch_offsets, settings.max_columns(T::FLOAT));
        let mut library_rejection = None;
        let library = match settings.library_width(device.backend(), T::FLOAT, batch_offsets) {
            Some(width) => {
                check_owners(device, &[tables.device(), scratch.x.as_slice().device()])?;
                assert!(
                    tables.len() >= ngroups * n * n,
                    "GroupedPlan: {} table values for {ngroups} groups of order {n}",
                    tables.len()
                );
                assert!(
                    scratch.len() >= width * n,
                    "GroupedPlan: a scratch of {} values for {width} columns of {n}",
                    scratch.len()
                );
                if tables.offset() != 0 {
                    library_rejection = Some("the tables do not start their buffer".into());
                    None
                } else {
                    let k_max = width / ngroups;
                    let probe = library_gemm(
                        device,
                        (ngroups, k_max, n),
                        tables,
                        scratch.x.as_slice(),
                        scratch.y.as_slice_mut(),
                    );
                    match probe {
                        Ok(()) => Some(k_max),
                        Err(reason) => {
                            library_rejection = Some(reason);
                            None
                        }
                    }
                }
            }
            None => None,
        };
        let (gemm, chunks, runs, columns, padded_sources) = match library {
            Some(k_max) => {
                let width = ngroups * k_max;
                // Each group's columns, then copies of a valid source up to k_max.
                let filler = arrays.batch_sources[0];
                let mut sources = vec![filler; width];
                for (g, w) in batch_offsets.windows(2).enumerate() {
                    let batch = &arrays.batch_sources[w[0] as usize..w[1] as usize];
                    sources[g * k_max..g * k_max + batch.len()].copy_from_slice(batch);
                }
                let chunk = Chunk {
                    entries: 0..arrays.batch_sources.len(),
                    width,
                    padded: k_max,
                };
                (
                    Gemm::Library,
                    vec![chunk],
                    vec![Vec::new()],
                    width,
                    Some(device.upload_indices(&sources)?),
                )
            }
            None => (
                Gemm::HandWritten(settings.layout),
                compact.chunks,
                compact.runs,
                compact.columns,
                None,
            ),
        };
        let widths = chunks.iter().map(|c| c.width).collect();
        let tiles = TileSchedule::upload(device, settings.layout.tile(), &runs, widths)?;
        Ok(Self {
            n,
            nrows: arrays.row_offsets.len() - 1,
            ngroups,
            len: arrays.batch_sources.len(),
            sources_bound,
            precision: T::FLOAT,
            gemm,
            chunks,
            tiles,
            padded_sources,
            columns,
            library_rejection,
        })
    }

    /// The GEMM the plan runs.
    pub fn gemm(&self) -> Gemm {
        self.gemm
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
/// in row order, through the chunks of `plan`, the tables `tables` (matrix g at g n²,
/// column-major) and `scratch`. Three launches per chunk; nothing for a view without
/// pairs. Allocates nothing.
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
/// `T`, the tables hold fewer than one matrix per group, the input or output does not
/// hold n values per column of the view, `scratch` is shorter than the plan needs, the
/// two ranges of [`Operands::Shared`] overlap, or (debug builds) an
/// [`Accumulate::Scatter`] chunk repeats a target.
pub fn grouped<T: DeviceFloat>(
    device: &mut Device,
    plan: &GroupedPlan,
    view: &GroupedView,
    accumulate: Accumulate,
    tables: DeviceSlice<'_, T>,
    mut operands: Operands<'_, T>,
    scratch: &mut TranslationScratch<T>,
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
        tables.len() >= plan.ngroups * n * n,
        "grouped: {} table values for {} groups of order {n}",
        tables.len(),
        plan.ngroups
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
            tables.device(),
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
    for (index, chunk) in plan.chunks.iter().enumerate() {
        let values = chunk.width * n;
        // 1. Gather.
        let sources = match &plan.padded_sources {
            Some(padded) => padded.as_slice(),
            None => view.batch_sources().slice(chunk.entries.clone()),
        };
        gather_columns(
            device,
            n,
            operands.input(),
            sources,
            scratch.x.slice_mut(..values),
        )?;
        // 2. The grouped GEMM.
        match plan.gemm {
            Gemm::HandWritten(layout) => gemm_segment(
                device,
                layout,
                n,
                tables,
                &plan.tiles,
                index,
                scratch.x.slice(..values),
                scratch.y.slice_mut(..values),
            )?,
            Gemm::Library => library_gemm(
                device,
                (plan.ngroups, chunk.padded, n),
                tables,
                scratch.x.as_slice(),
                scratch.y.as_slice_mut(),
            )
            .map_err(|reason| KernelError::Device { reason })?,
        }
        // 3. Accumulate.
        match (accumulate, chunk.padded) {
            (Accumulate::Scatter, 0) => scatter_add_columns(
                device,
                n,
                scratch.y.slice(..values),
                view.batch_targets().slice(chunk.entries.clone()),
                operands.output(),
            )?,
            _ => reduce_rows(
                device,
                n,
                view,
                chunk,
                scratch.y.slice(..values),
                operands.output(),
            )?,
        }
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
    gemm_segment(device, layout, n, tables, schedule, 0, x, y)
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
    library_gemm(device, (groups, k, n), tables, x, y).map_err(|reason| {
        KernelError::UnsupportedLayout {
            layout: Gemm::Library.to_string(),
            reason,
        }
    })
}

/// The tiles of segment `segment` of `schedule` (module documentation): one launch.
#[allow(clippy::too_many_arguments)]
fn gemm_segment<T: DeviceFloat>(
    device: &mut Device,
    layout: GemmLayout,
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
            // and x and y at their offsets plus c n + k with c < columns (asserted
            // within x.len() and y.len()). Rows at or past n and columns past the tile's
            // count are masked; cubes past ntiles (a 2-D grid) do nothing. Distinct
            // units write distinct (row, column) values of y.
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
                    n,
                    rows as usize,
                    tm,
                    per_unit as usize,
                    k_step(tm),
                );
            }
        }
        GemmLayout::Cpu { block, per_unit } => {
            let units = (cap as usize)
                .min(info.max_units_per_cube.max(1) as usize)
                .min(tiles.len())
                .max(1) as u32;
            // SAFETY: as for the cube layout: the tiles, tables, x and y are read and
            // written within the bounds asserted above. Each unit covers its own
            // contiguous range of tiles, and tiles cover distinct columns, so no two
            // units write one value.
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
                    n,
                    block as usize,
                    per_unit as usize,
                    k_step(block as usize),
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
/// value is stored if an entry was added. One launch.
fn reduce_rows<T: DeviceFloat>(
    device: &mut Device,
    n: usize,
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
    // (g − g0) k_max + position − batch offset, below the chunk's width) lie within y
    // (y.len() = width n). The output is read and written at its offset plus t n + i
    // < output.len() = nrows n (asserted by `grouped`); each unit owns its items.
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
            work as u32,
            grid.chunk,
            grid.threads(),
            n,
        );
    }
    device.count_launch();
    Ok(())
}

/// The library GEMM of `shape` = (groups G, columns per group k, order n): Y[g] = X[g]
/// Bᵀ[g] for the row-major [G, k, n] arrays x and y and the tables read as [G, n, n]
/// row-major (module documentation, "The library GEMM"), with the input-precision guard.
/// One launch; the error is the library's, or the guard's finding.
fn library_gemm<T: DeviceFloat>(
    device: &mut Device,
    (groups, k, n): (usize, usize, usize),
    tables: DeviceSlice<'_, T>,
    x: DeviceSlice<'_, T>,
    y: DeviceSliceMut<'_, T>,
) -> Result<(), String> {
    debug_assert!(tables.len() >= groups * n * n && x.len() >= groups * k * n);
    debug_assert!(y.len() >= groups * k * n && tables.offset() == 0);
    debug_assert!(x.offset() == 0 && y.offset() == 0);
    let elem = T::elem_type_native();
    let strategy: Strategy = multi_level::Strategy::SimpleCyclicCmma(Default::default()).into();
    let binding = |handle, strides: [usize; 3]| {
        // SAFETY: the strides and shape [G, rows, n] describe at most the elements the
        // buffer holds (asserted by the callers: tables ≥ G n², x and y ≥ G k n), from
        // offset 0, so the library reads and writes within the buffer.
        unsafe {
            TensorBinding::from_raw_parts(
                handle,
                Strides::from(strides),
                Shape::from([groups, strides[0] / n, n]),
            )
        }
    };
    let lhs = binding(x.binding().0, [k * n, n, 1]);
    let rhs = binding(tables.binding().0, [n * n, n, 1]);
    let out = binding(y.binding().0, [k * n, n, 1]);
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
/// `a0` (column-major, order n), x and y with column c at `x0 + c n` and `y0 + c n`. Each
/// value one accumulator from zero, `acc = fma(A_ik, x_kc, acc)` for k ascending, stored
/// once. Rows at or past n are masked.
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
    #[comptime] n: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] ks: usize,
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
            n,
            tm,
            tn,
            ks,
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
            n,
            tm,
            tn,
            comptime!(n % ks),
        );
    }
    #[unroll]
    for r in 0..tm {
        let i = row + r * row_stride;
        if i < n {
            #[unroll]
            for c in 0..tn {
                if c < cols {
                    y[y0 + (col + c) * n + i] = acc[r * tn + c];
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
    #[comptime] n: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] steps: usize,
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
                v = x[x0 + (col + c) * n + k];
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
    #[comptime] n: usize,
    #[comptime] rows: usize,
    #[comptime] tm: usize,
    #[comptime] tn: usize,
    #[comptime] ks: usize,
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
                n,
                tm,
                tn,
                ks,
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
    #[comptime] n: usize,
    #[comptime] block: usize,
    #[comptime] tn: usize,
    #[comptime] ks: usize,
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
                n,
                block,
                tn,
                ks,
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
    work: u32,
    chunk: u32,
    threads: u32,
    #[comptime] n: usize,
) {
    let (k_first, k_end, padded) = (k_first as usize, k_end as usize, padded as usize);
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
                        column = g * padded + k - batch_offsets[g] as usize;
                    }
                    acc += y[y_offset as usize + column * n + i];
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
        };
        let offsets = [0, 3, 3, 10];
        let s = settings(81, GemmPolicy::Auto);
        assert_eq!(
            s.library_width(BackendKind::Metal, Precision::F32, &offsets),
            Some(21)
        );
        assert_eq!(
            s.library_width(BackendKind::Metal, Precision::F64, &offsets),
            None
        );
        assert_eq!(
            s.library_width(BackendKind::Cpu, Precision::F32, &offsets),
            None
        );
        assert_eq!(
            settings(64, GemmPolicy::Auto).library_width(
                BackendKind::Metal,
                Precision::F32,
                &offsets
            ),
            None
        );
        assert_eq!(
            settings(81, GemmPolicy::HandWritten).library_width(
                BackendKind::Metal,
                Precision::F32,
                &offsets
            ),
            None
        );
        let size = s.size(BackendKind::Metal, Precision::F32, &offsets);
        assert_eq!(size.columns, 21);
        assert_eq!(size.bytes, 4 * (3 * 3 + 21));
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
