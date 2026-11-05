//! Plane-wave M2L on the device (Helmholtz CH7.5a; Laplace's merge and shift of Phase 6
//! T13, CONVENTIONS §3.14): the steps of `nd_fmm_tables::plane_wave` as kernels, one
//! launch per step and level call.
//!
//! # What a level call computes
//!
//! A level call is described by a [`PlaneWavePlan`], built on the host from plain arrays
//! ([`PlaneWaveLists`]; `nd-fmm-exec` derives them from a level's V view and the merge's
//! geometry, `nd_fmm_tables::plane_wave::geometry`). Its four steps, each a launch over
//! its own items, read and write the buffers of a [`PlaneWaveScratch`]:
//!
//! | Step | Items | Each item computes | Host function it restates |
//! | --- | --- | --- | --- |
//! | [`outgoing`] | (source i, frame f) | the multipole of column `sources[i]` in frame f, its weighted modes, and its up and down waves into wave blocks 6 i + 2 f and 6 i + 2 f + 1 | `rotate_multipole`, `outgoing_modes`, `waves_from_modes` |
//! | [`merge`] | accumulator a | Σ over its entries in order of shift × wave, from zero, into accumulator block a | `shift_add` (`WaveKernel::merge`) |
//! | [`gather`] | incoming block g | the sum of its accumulators in order, from zero, times its child shift | `add`, `shift` (`WaveKernel::gather`) |
//! | [`local`] | target t | per frame f = 0, 1, 2 with an incoming block: the sums of its up and down blocks, the local in the frame, and the local rotated back, added to column t of the locals | `sums_from_waves`, `local_from_sums`, `add_local_from_frame` |
//!
//! A block holds 2H values, the H real parts and then the H imaginary parts. Direction d
//! (`Direction::index`: up, down, north, south, east, west) is frame d / 2, up for even d;
//! wave block 6 i + d holds source i's waves for direction d. A local item names, per
//! direction, its incoming block or [`NO_BLOCK`], which reads as zero waves (the host's
//! zeroed half); a frame without either is skipped, as the host skips it.
//! [`m2l`] runs the four steps in order.
//!
//! # Arithmetic and order
//!
//! Every output value has one owner, and every chain runs in the order of the host function
//! it restates. The loops that `nd_fmm_tables::plane_wave` writes with `mul_add`
//! (`waves_from_modes`, `shift_add`, `shift`, `sums_from_waves`) are the same fused
//! multiply–adds here, written as explicit `fma`, so these steps give the host functions'
//! bits on the same inputs. The steps that the host rounds apart (`rotate_multipole`,
//! `outgoing_modes`, `local_from_sums`, `add_local_from_frame`: a product, then a sum)
//! are written as one `fma` per term here: cubecl-opt's `InstCombinePass` fuses every lone
//! a · b ± c on every backend (CONVENTIONS §3.13, "Device kernels"), so no formulation keeps
//! the host's roundings. The kernels equal a host replica of their order with `mul_add`
//! bit for bit (tested), and the host functions to rounding.
//!
//! No atomics: an accumulator, an incoming block and a local are written by one cube (or
//! unit) only; the locals of one launch are distinct columns (checked when the plan is
//! built). The
//! result does not depend on the layout, the launch shape or a repetition.
//!
//! # Layouts ([`PlaneWaveLayout`])
//!
//! | Layout | Default on | Mapping |
//! | --- | --- | --- |
//! | [`Cube`](PlaneWaveLayout::Cube) | Metal, CUDA | one cube of U units per item, the units striding over its outputs; [`outgoing`] and [`local`] stage the frame's coefficients and the modes or sums in shared memory, with `sync_cube` between steps |
//! | [`Cpu`](PlaneWaveLayout::Cpu) | the CPU runtime | one cube, one unit per core (at most [`Device::units_cap`]), each a contiguous range of the items, working vectors in local arrays |
//!
//! # Launch
//!
//! [`PlaneWaveTables::upload`] uploads the tables once; [`PlaneWavePlan::new`] validates a
//! level call's lists and uploads them; [`PlaneWaveScratch::new`] allocates the waves, the
//! accumulators and the incoming blocks. A step launches nothing without items and
//! allocates nothing.

use std::fmt;

use cubecl::prelude::*;

use crate::buffer::{DeviceBuffer, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexBuffer};
use crate::device::{BackendKind, Device, DeviceInfo, Precision};
use crate::error::KernelError;
use crate::leaf::{MAX_DEGREE, coefficients, unit_rows};
use crate::p2p::cube_grid;

/// A direction without an incoming block in a local item: zero waves.
pub const NO_BLOCK: u32 = u32::MAX;

/// The directions of merge and shift, two per frame.
pub const DIRECTIONS: usize = 6;

/// The children of a parent, each with a shift per direction.
pub const CHILDREN: usize = 8;

/// The default units per cube of the [`Cube`](PlaneWaveLayout::Cube) layout on a GPU.
pub const GPU_UNITS: u32 = 128;

/// The values of the frame rotation blocks of one frame, Σₙ (2n + 1)² =
/// (p + 1)(2p + 1)(2p + 3)/3.
pub const fn blocks_len(p: usize) -> usize {
    (p + 1) * (2 * p + 1) * (2 * p + 3) / 3
}

/// One λ node of the waves: `nd_fmm_tables::plane_wave::LambdaBlock`, field for field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LambdaNode {
    /// Its first value among the H values of a half block.
    pub start: usize,
    /// Its values, Nₖ'/2.
    pub len: usize,
    /// Its Fourier modes Fₖ': orders 0 ≤ m < Fₖ'.
    pub modes: usize,
    /// Its first mode among all the modes.
    pub mode_start: usize,
    /// Its first value in the cos and sin tables: Fₖ' rows of `len` values.
    pub trig: usize,
}

/// The host arrays of the plane-wave tables of one degree, in the storage of
/// `nd_fmm_tables::plane_wave::PlaneWaveTables` (its accessor of the same name for each).
#[derive(Clone, Copy, Debug)]
pub struct PlaneWaveArrays<'a, T> {
    /// The degree p.
    pub p: usize,
    /// The λ nodes in order (`blocks`), their values contiguous from 0 to H and their
    /// modes from 0 to F.
    pub nodes: &'a [LambdaNode],
    /// (wₖ/Nₖ') λₖⁿ 2^{−(n+1)} at n s + k (`outgoing_powers`).
    pub outgoing_powers: &'a [T],
    /// 2 λₖⁿ 2^{−n} at k (p + 1) + n (`incoming_powers`).
    pub incoming_powers: &'a [T],
    /// cos mαⱼ per node, row m of node k at `trig + m len` (`cos`).
    pub cos: &'a [T],
    /// sin mαⱼ, as `cos` (`sin`).
    pub sin: &'a [T],
    /// The multipole rotations into frames 1 and 2, [`blocks_len`]`(p)` values each,
    /// transposed per degree (`multipole_rotation`).
    pub multipole_rotations: [&'a [T]; 2],
    /// The local rotations back from frames 1 and 2, in that layout (`local_rotation`).
    pub local_rotations: [&'a [T]; 2],
    /// The distinct source shifts, 2H values each (`source_shifts`).
    pub source_shifts: &'a [T],
    /// The child shifts, 2H values each, that of direction d and child o at 8 d + o
    /// (`child_shifts`).
    pub child_shifts: &'a [T],
}

/// The plane-wave tables of one degree on the device, uploaded once: the arrays of
/// [`PlaneWaveArrays`], the λ nodes as five `u32` each, and per value of a half block and
/// per mode the index of its node.
#[derive(Debug)]
pub struct PlaneWaveTables<T: DeviceFloat> {
    p: usize,
    h: usize,
    modes: usize,
    lambdas: usize,
    source_shift_count: usize,
    nodes: IndexBuffer,
    value_nodes: IndexBuffer,
    mode_nodes: IndexBuffer,
    outgoing_powers: DeviceBuffer<T>,
    incoming_powers: DeviceBuffer<T>,
    cos: DeviceBuffer<T>,
    sin: DeviceBuffer<T>,
    multipole_rotations: DeviceBuffer<T>,
    local_rotations: DeviceBuffer<T>,
    source_shifts: DeviceBuffer<T>,
    child_shifts: DeviceBuffer<T>,
}

impl<T: DeviceFloat> PlaneWaveTables<T> {
    /// Validates the arrays and uploads them (eleven uploads).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If p exceeds [`MAX_DEGREE`], the nodes are not contiguous or a node has more than
    /// p + 1 modes, or an array does not have its length.
    pub fn upload(
        device: &mut Device,
        arrays: &PlaneWaveArrays<'_, T>,
    ) -> Result<Self, KernelError> {
        let p = arrays.p;
        assert!(
            p <= MAX_DEGREE,
            "PlaneWaveTables: p = {p} exceeds {MAX_DEGREE}"
        );
        let s = arrays.nodes.len();
        let (mut h, mut modes, mut trig) = (0, 0, 0);
        let mut nodes = Vec::with_capacity(5 * s);
        let mut value_nodes = Vec::new();
        let mut mode_nodes = Vec::new();
        for (k, node) in arrays.nodes.iter().enumerate() {
            assert!(
                node.start == h && node.mode_start == modes && node.trig == trig,
                "PlaneWaveTables: node {k} is not contiguous with the nodes before it"
            );
            assert!(
                node.modes <= p + 1,
                "PlaneWaveTables: node {k} has {} modes at p = {p}",
                node.modes
            );
            nodes.extend(
                [node.start, node.len, node.modes, node.mode_start, node.trig].map(|x| x as u32),
            );
            value_nodes.extend(std::iter::repeat_n(k as u32, node.len));
            mode_nodes.extend(std::iter::repeat_n(k as u32, node.modes));
            h += node.len;
            modes += node.modes;
            trig += node.modes * node.len;
        }
        let w = 2 * h;
        let check = |what: &str, len: usize, want: usize| {
            assert_eq!(
                len, want,
                "PlaneWaveTables: {len} values of {what}, not {want}"
            );
        };
        check("outgoing powers", arrays.outgoing_powers.len(), (p + 1) * s);
        check("incoming powers", arrays.incoming_powers.len(), s * (p + 1));
        check("cos", arrays.cos.len(), trig);
        check("sin", arrays.sin.len(), trig);
        for f in 0..2 {
            check(
                "a multipole rotation",
                arrays.multipole_rotations[f].len(),
                blocks_len(p),
            );
            check(
                "a local rotation",
                arrays.local_rotations[f].len(),
                blocks_len(p),
            );
        }
        check(
            "child shifts",
            arrays.child_shifts.len(),
            DIRECTIONS * CHILDREN * w,
        );
        assert!(
            w > 0 && arrays.source_shifts.len().is_multiple_of(w),
            "PlaneWaveTables: {} source shift values are not whole shifts of {w}",
            arrays.source_shifts.len()
        );
        let pair = |a: [&[T]; 2]| [a[0], a[1]].concat();
        Ok(Self {
            p,
            h,
            modes,
            lambdas: s,
            source_shift_count: arrays.source_shifts.len() / w,
            nodes: device.upload_indices(&nodes)?,
            value_nodes: device.upload_indices(&value_nodes)?,
            mode_nodes: device.upload_indices(&mode_nodes)?,
            outgoing_powers: device.upload(arrays.outgoing_powers)?,
            incoming_powers: device.upload(arrays.incoming_powers)?,
            cos: device.upload(arrays.cos)?,
            sin: device.upload(arrays.sin)?,
            multipole_rotations: device.upload(&pair(arrays.multipole_rotations))?,
            local_rotations: device.upload(&pair(arrays.local_rotations))?,
            source_shifts: device.upload(arrays.source_shifts)?,
            child_shifts: device.upload(arrays.child_shifts)?,
        })
    }

    /// The degree p.
    pub fn p(&self) -> usize {
        self.p
    }

    /// H, the complex values of a block.
    pub fn h(&self) -> usize {
        self.h
    }

    /// The values of a block, 2H.
    pub fn block_len(&self) -> usize {
        2 * self.h
    }

    /// The Fourier modes of all the λ nodes, F.
    pub fn modes(&self) -> usize {
        self.modes
    }

    /// The number of distinct source shifts.
    pub fn source_shift_count(&self) -> usize {
        self.source_shift_count
    }

    /// The bytes [`upload`](Self::upload) allocates for `arrays`, before any upload: the
    /// arrays as given, five `u32` per λ node and one per value of a half block and per
    /// mode.
    pub fn bytes_of(arrays: &PlaneWaveArrays<'_, T>) -> u64 {
        let (h, modes) = arrays
            .nodes
            .iter()
            .fold((0, 0), |(h, f), node| (h + node.len, f + node.modes));
        [
            arrays.outgoing_powers.len(),
            arrays.incoming_powers.len(),
            arrays.cos.len(),
            arrays.sin.len(),
            arrays.multipole_rotations[0].len() + arrays.multipole_rotations[1].len(),
            arrays.local_rotations[0].len() + arrays.local_rotations[1].len(),
            arrays.source_shifts.len(),
            arrays.child_shifts.len(),
        ]
        .into_iter()
        .map(Device::buffer_bytes::<T>)
        .sum::<u64>()
            + [5 * arrays.nodes.len(), h, modes]
                .into_iter()
                .map(Device::buffer_bytes::<u32>)
                .sum::<u64>()
    }

    /// The bytes on the device.
    pub fn bytes(&self) -> u64 {
        [
            &self.outgoing_powers,
            &self.incoming_powers,
            &self.cos,
            &self.sin,
            &self.multipole_rotations,
            &self.local_rotations,
            &self.source_shifts,
            &self.child_shifts,
        ]
        .iter()
        .map(|b| Device::buffer_bytes::<T>(b.len()))
        .sum::<u64>()
            + [&self.nodes, &self.value_nodes, &self.mode_nodes]
                .iter()
                .map(|b| Device::buffer_bytes::<u32>(b.len()))
                .sum::<u64>()
    }
}

/// The layout of a plane-wave launch (module documentation, "Layouts").
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaneWaveLayout {
    /// One cube of `units` units per item, striding over its outputs; shared memory for
    /// the outgoing and local steps.
    Cube {
        /// Units per cube, at least 1.
        units: u32,
    },
    /// One cube of one unit per core, each a contiguous range of the items.
    Cpu,
}

impl PlaneWaveLayout {
    /// The default layout of a device: [`Cpu`](Self::Cpu) on the CPU runtime, otherwise
    /// [`Cube`](Self::Cube) with [`GPU_UNITS`] units.
    pub fn default_for(info: &DeviceInfo) -> Self {
        match info.backend {
            BackendKind::Cpu => Self::Cpu,
            BackendKind::Metal | BackendKind::Cuda => Self::Cube { units: GPU_UNITS },
        }
    }

    /// Checks that the device can run the layout with tables of degree p and F `modes` in
    /// `precision`.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedLayout`] for a cube of no unit, more units than the device
    /// allows per cube, or more shared memory than it has ((p + 1)² + 4F values).
    pub fn check(
        self,
        info: &DeviceInfo,
        p: usize,
        modes: usize,
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
        let shared = (coefficients(p) + 4 * modes) * bytes;
        if shared > info.max_shared_memory {
            return refuse(format!(
                "{shared} bytes of shared memory at p = {p} with {modes} modes, the device \
                 has {}",
                info.max_shared_memory
            ));
        }
        Ok(())
    }
}

impl fmt::Display for PlaneWaveLayout {
    /// `cube (128 units)` or `cpu (one unit per core)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cube { units } => write!(f, "cube ({units} units)"),
            Self::Cpu => f.write_str("cpu (one unit per core)"),
        }
    }
}

/// The lists of one level call on the host (module documentation): plain `u32` arrays.
#[derive(Clone, Copy, Debug)]
pub struct PlaneWaveLists<'a> {
    /// The multipole column of each source i, whose waves are wave blocks 6 i to 6 i + 5.
    pub sources: &'a [u32],
    /// The CSR offsets of the accumulators' entries: accumulator a has the entries
    /// `merge_offsets[a] .. merge_offsets[a + 1]` (A + 1 offsets, from 0).
    pub merge_offsets: &'a [u32],
    /// Two values per entry: its wave block and its source shift's index.
    pub merge_entries: &'a [u32],
    /// The child shift of each incoming block g, 8 d + o.
    pub gather_shifts: &'a [u32],
    /// The CSR offsets of the incoming blocks' accumulators (G + 1 offsets, from 0).
    pub gather_offsets: &'a [u32],
    /// The accumulators of each incoming block, in order.
    pub gather_entries: &'a [u32],
    /// The local column of each local item, distinct.
    pub locals: &'a [u32],
    /// Six values per local item: its incoming block per direction, or [`NO_BLOCK`].
    pub incoming: &'a [u32],
}

impl PlaneWaveLists<'_> {
    /// The bytes [`PlaneWavePlan::new`] uploads for the lists: [`PlaneWavePlan::bytes`]
    /// before the upload.
    pub fn bytes(&self) -> u64 {
        [
            self.sources,
            self.merge_offsets,
            self.merge_entries,
            self.gather_shifts,
            self.gather_offsets,
            self.gather_entries,
            self.locals,
            self.incoming,
        ]
        .iter()
        .map(|b| Device::buffer_bytes::<u32>(b.len()))
        .sum()
    }
}

/// The lists of one level call on the device, validated: built once
/// ([`new`](Self::new)).
#[derive(Debug)]
pub struct PlaneWavePlan {
    sources: IndexBuffer,
    merge_offsets: IndexBuffer,
    merge_entries: IndexBuffer,
    gather_shifts: IndexBuffer,
    gather_offsets: IndexBuffer,
    gather_entries: IndexBuffer,
    locals: IndexBuffer,
    incoming: IndexBuffer,
    nsources: usize,
    naccumulators: usize,
    nincoming: usize,
    nlocals: usize,
    sources_bound: usize,
    locals_bound: usize,
    /// The source shifts and block length of the tables the plan was validated against.
    source_shifts: usize,
    block_len: usize,
}

/// Panics unless `offsets` are CSR offsets from 0 over `entries` entries.
fn check_csr(what: &str, offsets: &[u32], entries: usize) {
    assert!(
        offsets.first() == Some(&0),
        "PlaneWavePlan: the {what} offsets do not start at 0"
    );
    assert!(
        offsets.windows(2).all(|w| w[0] <= w[1]),
        "PlaneWavePlan: the {what} offsets decrease"
    );
    assert_eq!(
        *offsets.last().unwrap() as usize,
        entries,
        "PlaneWavePlan: the {what} offsets end at {}, not at the {entries} entries",
        offsets.last().unwrap()
    );
}

impl PlaneWavePlan {
    /// Validates `lists` against `tables` (the multipoles holding `sources_bound` columns,
    /// the locals `locals_bound`) and uploads them (eight uploads).
    ///
    /// # Errors
    ///
    /// As [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If an offset list is not CSR from 0 over its entries, an index lies outside what it
    /// indexes (a source column, a wave block, a shift, an accumulator, an incoming block
    /// or a local column), the local columns repeat, the incoming blocks are not six per
    /// local item, or a buffer of the call would exceed the 32-bit index range.
    pub fn new<T: DeviceFloat>(
        device: &mut Device,
        lists: &PlaneWaveLists<'_>,
        tables: &PlaneWaveTables<T>,
        sources_bound: usize,
        locals_bound: usize,
    ) -> Result<Self, KernelError> {
        let nsources = lists.sources.len();
        assert!(
            lists.sources.iter().all(|&s| (s as usize) < sources_bound),
            "PlaneWavePlan: a source column lies outside the {sources_bound} multipoles"
        );
        let naccumulators = lists.merge_offsets.len().saturating_sub(1);
        assert!(
            lists.merge_entries.len().is_multiple_of(2),
            "PlaneWavePlan: two values per merge entry"
        );
        check_csr("merge", lists.merge_offsets, lists.merge_entries.len() / 2);
        for entry in lists.merge_entries.as_chunks::<2>().0 {
            assert!(
                (entry[0] as usize) < DIRECTIONS * nsources,
                "PlaneWavePlan: wave block {} of {} sources",
                entry[0],
                nsources
            );
            assert!(
                (entry[1] as usize) < tables.source_shift_count,
                "PlaneWavePlan: source shift {} of {}",
                entry[1],
                tables.source_shift_count
            );
        }
        let nincoming = lists.gather_shifts.len();
        assert_eq!(
            lists.gather_offsets.len(),
            nincoming + 1,
            "PlaneWavePlan: G + 1 gather offsets"
        );
        check_csr("gather", lists.gather_offsets, lists.gather_entries.len());
        assert!(
            lists
                .gather_shifts
                .iter()
                .all(|&c| (c as usize) < DIRECTIONS * CHILDREN),
            "PlaneWavePlan: a child shift beyond the 48"
        );
        assert!(
            lists
                .gather_entries
                .iter()
                .all(|&a| (a as usize) < naccumulators),
            "PlaneWavePlan: an accumulator beyond the {naccumulators}"
        );
        let nlocals = lists.locals.len();
        assert_eq!(
            lists.incoming.len(),
            DIRECTIONS * nlocals,
            "PlaneWavePlan: six incoming blocks per local item"
        );
        assert!(
            lists
                .incoming
                .iter()
                .all(|&g| g == NO_BLOCK || (g as usize) < nincoming),
            "PlaneWavePlan: an incoming block beyond the {nincoming}"
        );
        assert!(
            lists.locals.iter().all(|&t| (t as usize) < locals_bound),
            "PlaneWavePlan: a local column lies outside the {locals_bound} locals"
        );
        let w = tables.block_len();
        for (what, values) in [
            ("multipoles", sources_bound * coefficients(tables.p)),
            ("locals", locals_bound * coefficients(tables.p)),
            ("waves", DIRECTIONS * nsources * w),
            ("accumulators", naccumulators * w),
            ("incoming blocks", nincoming * w),
            ("source shifts", tables.source_shift_count * w),
        ] {
            assert!(
                u32::try_from(values).is_ok(),
                "PlaneWavePlan: {values} values of {what} exceed the 32-bit index range"
            );
        }
        let mut columns = lists.locals.to_vec();
        columns.sort_unstable();
        assert!(
            columns.windows(2).all(|w| w[0] < w[1]),
            "PlaneWavePlan: a local column repeats"
        );
        Ok(Self {
            sources: device.upload_indices(lists.sources)?,
            merge_offsets: device.upload_indices(lists.merge_offsets)?,
            merge_entries: device.upload_indices(lists.merge_entries)?,
            gather_shifts: device.upload_indices(lists.gather_shifts)?,
            gather_offsets: device.upload_indices(lists.gather_offsets)?,
            gather_entries: device.upload_indices(lists.gather_entries)?,
            locals: device.upload_indices(lists.locals)?,
            incoming: device.upload_indices(lists.incoming)?,
            nsources,
            naccumulators,
            nincoming,
            nlocals,
            sources_bound,
            locals_bound,
            source_shifts: tables.source_shift_count,
            block_len: tables.block_len(),
        })
    }

    /// The sources, each six wave blocks.
    pub fn sources(&self) -> usize {
        self.nsources
    }

    /// The accumulators, A.
    pub fn accumulators(&self) -> usize {
        self.naccumulators
    }

    /// The incoming blocks, G.
    pub fn incoming(&self) -> usize {
        self.nincoming
    }

    /// The local items.
    pub fn locals(&self) -> usize {
        self.nlocals
    }

    /// The bytes on the device.
    pub fn bytes(&self) -> u64 {
        [
            &self.sources,
            &self.merge_offsets,
            &self.merge_entries,
            &self.gather_shifts,
            &self.gather_offsets,
            &self.gather_entries,
            &self.locals,
            &self.incoming,
        ]
        .iter()
        .map(|b| Device::buffer_bytes::<u32>(b.len()))
        .sum()
    }
}

/// The working buffers of a level call: the waves (6 blocks per source), the accumulators
/// and the incoming blocks, 2H values per block.
#[derive(Debug)]
pub struct PlaneWaveScratch<T: DeviceFloat> {
    waves: DeviceBuffer<T>,
    accumulators: DeviceBuffer<T>,
    incoming: DeviceBuffer<T>,
}

impl<T: DeviceFloat> PlaneWaveScratch<T> {
    /// The buffers of `plan` with `tables` (three allocations).
    ///
    /// # Errors
    ///
    /// As [`Device::alloc`].
    pub fn new(
        device: &mut Device,
        plan: &PlaneWavePlan,
        tables: &PlaneWaveTables<T>,
    ) -> Result<Self, KernelError> {
        let w = tables.block_len();
        Ok(Self {
            waves: device.alloc(DIRECTIONS * plan.nsources * w)?,
            accumulators: device.alloc(plan.naccumulators * w)?,
            incoming: device.alloc(plan.nincoming * w)?,
        })
    }

    /// The buffers of the widest of `plans` with `tables`, which serve every one of them
    /// (three allocations): the most sources, accumulators and incoming blocks of any.
    ///
    /// # Errors
    ///
    /// As [`Device::alloc`].
    pub fn covering<'a>(
        device: &mut Device,
        plans: impl IntoIterator<Item = &'a PlaneWavePlan>,
        tables: &PlaneWaveTables<T>,
    ) -> Result<Self, KernelError> {
        let [sources, accumulators, incoming] = Self::widest(plans);
        let w = tables.block_len();
        Ok(Self {
            waves: device.alloc(DIRECTIONS * sources * w)?,
            accumulators: device.alloc(accumulators * w)?,
            incoming: device.alloc(incoming * w)?,
        })
    }

    /// The most sources, accumulators and incoming blocks of any of `plans`.
    fn widest<'a>(plans: impl IntoIterator<Item = &'a PlaneWavePlan>) -> [usize; 3] {
        plans.into_iter().fold([0; 3], |[s, a, g], plan| {
            [
                s.max(plan.nsources),
                a.max(plan.naccumulators),
                g.max(plan.nincoming),
            ]
        })
    }

    /// The bytes [`new`](Self::new) allocates for `plan` with blocks of `block_len` values.
    pub fn bytes(plan: &PlaneWavePlan, block_len: usize) -> u64 {
        Self::bytes_of(
            [plan.nsources, plan.naccumulators, plan.nincoming],
            block_len,
        )
    }

    /// The bytes of the buffers of a plan of `[sources, accumulators, incoming blocks]` with
    /// blocks of `block_len` values, which [`new`](Self::new) and
    /// [`covering`](Self::covering) allocate.
    pub fn bytes_of([sources, accumulators, incoming]: [usize; 3], block_len: usize) -> u64 {
        Device::buffer_bytes::<T>(DIRECTIONS * sources * block_len)
            + Device::buffer_bytes::<T>(accumulators * block_len)
            + Device::buffer_bytes::<T>(incoming * block_len)
    }

    /// The waves, wave block b at b 2H.
    pub fn waves(&self) -> DeviceSlice<'_, T> {
        self.waves.as_slice()
    }

    /// The accumulators, accumulator a at a 2H.
    pub fn accumulators(&self) -> DeviceSlice<'_, T> {
        self.accumulators.as_slice()
    }

    /// The incoming blocks, block g at g 2H.
    pub fn incoming(&self) -> DeviceSlice<'_, T> {
        self.incoming.as_slice()
    }

    /// Panics unless `plan` was validated against tables like `tables` (their source
    /// shifts and block length) and the buffers hold at least `plan`'s with them.
    fn check(&self, plan: &PlaneWavePlan, tables: &PlaneWaveTables<T>) {
        let w = tables.block_len();
        assert!(
            plan.source_shifts == tables.source_shift_count && plan.block_len == w,
            "plane_wave: the plan was validated against other tables"
        );
        assert!(
            self.waves.len() >= DIRECTIONS * plan.nsources * w
                && self.accumulators.len() >= plan.naccumulators * w
                && self.incoming.len() >= plan.nincoming * w,
            "plane_wave: the scratch is too small for the plan with these tables"
        );
    }
}

/// The units of the CPU layout for `items` items.
fn cpu_units(device: &Device, items: usize) -> u32 {
    (device.units_cap() as usize)
        .min(device.info().max_units_per_cube.max(1) as usize)
        .min(items)
        .max(1) as u32
}

/// The outgoing step (module documentation): the waves of every source in every frame
/// into the scratch's wave blocks. One launch; nothing without sources.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device;
/// [`KernelError::UnsupportedLayout`] if the device cannot run `layout`.
///
/// # Panics
///
/// If `multipoles` does not hold the plan's source bound of (p + 1)² values, the plan was
/// validated against other tables, or the scratch is too small for the plan.
pub fn outgoing<T: DeviceFloat>(
    device: &mut Device,
    layout: PlaneWaveLayout,
    plan: &PlaneWavePlan,
    tables: &PlaneWaveTables<T>,
    multipoles: DeviceSlice<'_, T>,
    scratch: &mut PlaneWaveScratch<T>,
) -> Result<(), KernelError> {
    for owner in [
        plan.sources.as_slice().device(),
        tables.nodes.as_slice().device(),
        tables.value_nodes.as_slice().device(),
        tables.mode_nodes.as_slice().device(),
        tables.outgoing_powers.as_slice().device(),
        tables.cos.as_slice().device(),
        tables.sin.as_slice().device(),
        tables.multipole_rotations.as_slice().device(),
        multipoles.device(),
        scratch.waves.as_slice().device(),
    ] {
        device.check_owner(owner)?;
    }
    scratch.check(plan, tables);
    let (p, n) = (tables.p, coefficients(tables.p));
    assert_eq!(
        multipoles.len(),
        plan.sources_bound * n,
        "plane_wave::outgoing: the multipoles hold {} values, not {} columns of {n}",
        multipoles.len(),
        plan.sources_bound
    );
    assert!(
        u32::try_from(multipoles.offset() + multipoles.len()).is_ok(),
        "plane_wave::outgoing: the multipoles exceed the 32-bit index range"
    );
    layout.check(device.info(), p, tables.modes, T::FLOAT)?;
    let items = 3 * plan.nsources;
    if items == 0 {
        return Ok(());
    }
    let (sh, sl) = plan.sources.as_slice().binding();
    let (nh, nl) = tables.nodes.as_slice().binding();
    let (vh, vl) = tables.value_nodes.as_slice().binding();
    let (mh, ml) = tables.mode_nodes.as_slice().binding();
    let (ph, pl) = tables.outgoing_powers.as_slice().binding();
    let (ch, cl) = tables.cos.as_slice().binding();
    let (zh, zl) = tables.sin.as_slice().binding();
    let (rh, rl) = tables.multipole_rotations.as_slice().binding();
    let (ih, il) = multipoles.binding();
    let (wh, wl) = scratch.waves.as_slice().binding();
    let scalars = (
        items as u32,
        multipoles.offset() as u32,
        tables.h as u32,
        tables.lambdas as u32,
    );
    let client = device.client().clone();
    let (cubes, units, cube) = match layout {
        PlaneWaveLayout::Cube { units } => {
            let (x, y) = cube_grid(device.info(), items);
            (CubeCount::Static(x, y, 1), units, true)
        }
        PlaneWaveLayout::Cpu => (CubeCount::Static(1, 1, 1), cpu_units(device, items), false),
    };
    // SAFETY: every handle is a whole buffer with its element count. Item c < items reads
    // source c / 3 < nsources, whose column is below sources_bound (checked at the plan's
    // build), so the multipole is read at its offset plus s (p + 1)² + k < multipoles.len()
    // (asserted). The nodes hold five values per λ node, the value and mode maps one node
    // index < s per value of a half block (H) and per mode (F), the powers (p + 1) s, cos
    // and sin each node's rows (checked at upload: the nodes are contiguous), the rotations
    // two frames of blocks_len(p). The waves hold at least 6 nsources blocks of 2H (checked
    // against the plan), and item c writes blocks 6 (c / 3) + 2 (c % 3) and the next, its own. The
    // cube layout's shared memory holds (p + 1)² + 4F values, each written by its owner
    // between barriers that every unit of the cube reaches (the item is the cube's); the
    // CPU layout's units cover disjoint ranges of the items with local arrays.
    unsafe {
        outgoing_kernel::launch_unchecked::<T>(
            &client,
            cubes,
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(sh, sl),
            BufferArg::from_raw_parts(nh, nl),
            BufferArg::from_raw_parts(vh, vl),
            BufferArg::from_raw_parts(mh, ml),
            BufferArg::from_raw_parts(ph, pl),
            BufferArg::from_raw_parts(ch, cl),
            BufferArg::from_raw_parts(zh, zl),
            BufferArg::from_raw_parts(rh, rl),
            BufferArg::from_raw_parts(ih, il),
            BufferArg::from_raw_parts(wh, wl),
            scalars.0,
            scalars.1,
            scalars.2,
            scalars.3,
            p,
            tables.modes,
            units as usize,
            cube,
        );
    }
    device.count_launch();
    Ok(())
}

/// The merge step (module documentation): every accumulator from the scratch's waves into
/// its accumulator block. One launch; nothing without accumulators.
///
/// # Errors
///
/// As [`outgoing`].
///
/// # Panics
///
/// If the plan was validated against other tables, or the scratch is too small for the
/// plan.
pub fn merge<T: DeviceFloat>(
    device: &mut Device,
    layout: PlaneWaveLayout,
    plan: &PlaneWavePlan,
    tables: &PlaneWaveTables<T>,
    scratch: &mut PlaneWaveScratch<T>,
) -> Result<(), KernelError> {
    for owner in [
        plan.merge_offsets.as_slice().device(),
        plan.merge_entries.as_slice().device(),
        tables.source_shifts.as_slice().device(),
        scratch.waves.as_slice().device(),
        scratch.accumulators.as_slice().device(),
    ] {
        device.check_owner(owner)?;
    }
    scratch.check(plan, tables);
    layout.check(device.info(), tables.p, tables.modes, T::FLOAT)?;
    let items = plan.naccumulators;
    if items == 0 {
        return Ok(());
    }
    let (oh, ol) = plan.merge_offsets.as_slice().binding();
    let (eh, el) = plan.merge_entries.as_slice().binding();
    let (th, tl) = tables.source_shifts.as_slice().binding();
    let (wh, wl) = scratch.waves.as_slice().binding();
    let (ah, al) = scratch.accumulators.as_slice().binding();
    let client = device.client().clone();
    let (cubes, units, cube) = match layout {
        PlaneWaveLayout::Cube { units } => {
            let (x, y) = cube_grid(device.info(), items);
            (CubeCount::Static(x, y, 1), units, true)
        }
        PlaneWaveLayout::Cpu => (CubeCount::Static(1, 1, 1), cpu_units(device, items), false),
    };
    // SAFETY: every handle is a whole buffer with its element count. Accumulator a < A
    // reads the offsets a and a + 1 (A + 1 offsets, validated CSR over the entries) and its
    // entries' pairs, each a wave block < 6 nsources and a source shift < the shift count
    // (checked at the plan's build against tables with the shift count and block length of
    // these, asserted by the scratch check), read at block · 2H + j < their buffers' lengths (the
    // waves checked against the plan, the shifts whole blocks of 2H at upload). It writes
    // its own block a of the accumulators (at least A blocks, checked). The CPU layout's units cover
    // disjoint ranges of the accumulators.
    unsafe {
        merge_kernel::launch_unchecked::<T>(
            &client,
            cubes,
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(oh, ol),
            BufferArg::from_raw_parts(eh, el),
            BufferArg::from_raw_parts(th, tl),
            BufferArg::from_raw_parts(wh, wl),
            BufferArg::from_raw_parts(ah, al),
            items as u32,
            tables.h as u32,
            units as usize,
            cube,
        );
    }
    device.count_launch();
    Ok(())
}

/// The gather step (module documentation): every incoming block from the scratch's
/// accumulators. One launch; nothing without incoming blocks.
///
/// # Errors
///
/// As [`outgoing`].
///
/// # Panics
///
/// If the plan was validated against other tables, or the scratch is too small for the
/// plan.
pub fn gather<T: DeviceFloat>(
    device: &mut Device,
    layout: PlaneWaveLayout,
    plan: &PlaneWavePlan,
    tables: &PlaneWaveTables<T>,
    scratch: &mut PlaneWaveScratch<T>,
) -> Result<(), KernelError> {
    for owner in [
        plan.gather_shifts.as_slice().device(),
        plan.gather_offsets.as_slice().device(),
        plan.gather_entries.as_slice().device(),
        tables.child_shifts.as_slice().device(),
        scratch.accumulators.as_slice().device(),
        scratch.incoming.as_slice().device(),
    ] {
        device.check_owner(owner)?;
    }
    scratch.check(plan, tables);
    layout.check(device.info(), tables.p, tables.modes, T::FLOAT)?;
    let items = plan.nincoming;
    if items == 0 {
        return Ok(());
    }
    let (sh, sl) = plan.gather_shifts.as_slice().binding();
    let (oh, ol) = plan.gather_offsets.as_slice().binding();
    let (eh, el) = plan.gather_entries.as_slice().binding();
    let (th, tl) = tables.child_shifts.as_slice().binding();
    let (ah, al) = scratch.accumulators.as_slice().binding();
    let (gh, gl) = scratch.incoming.as_slice().binding();
    let client = device.client().clone();
    let (cubes, units, cube) = match layout {
        PlaneWaveLayout::Cube { units } => {
            let (x, y) = cube_grid(device.info(), items);
            (CubeCount::Static(x, y, 1), units, true)
        }
        PlaneWaveLayout::Cpu => (CubeCount::Static(1, 1, 1), cpu_units(device, items), false),
    };
    // SAFETY: every handle is a whole buffer with its element count. Incoming block g < G
    // reads its child shift < 48 (checked; the child shifts hold 48 blocks of 2H, checked
    // at upload), the offsets g and g + 1 (G + 1, validated CSR) and its accumulators,
    // each < A (checked), whose blocks lie in the accumulators (at least A blocks, checked
    // against the plan). It writes its own block g of the incoming blocks (at least G
    // blocks, checked). The
    // CPU layout's units cover disjoint ranges.
    unsafe {
        gather_kernel::launch_unchecked::<T>(
            &client,
            cubes,
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(sh, sl),
            BufferArg::from_raw_parts(oh, ol),
            BufferArg::from_raw_parts(eh, el),
            BufferArg::from_raw_parts(th, tl),
            BufferArg::from_raw_parts(ah, al),
            BufferArg::from_raw_parts(gh, gl),
            items as u32,
            tables.h as u32,
            units as usize,
            cube,
        );
    }
    device.count_launch();
    Ok(())
}

/// The local step (module documentation): every local item's frames added to its column
/// of `locals`. One launch; nothing without local items.
///
/// # Errors
///
/// As [`outgoing`].
///
/// # Panics
///
/// If `locals` does not hold the plan's local bound of (p + 1)² values, the plan was
/// validated against other tables, or the scratch is too small for the plan.
pub fn local<T: DeviceFloat>(
    device: &mut Device,
    layout: PlaneWaveLayout,
    plan: &PlaneWavePlan,
    tables: &PlaneWaveTables<T>,
    scratch: &PlaneWaveScratch<T>,
    locals: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    for owner in [
        plan.locals.as_slice().device(),
        plan.incoming.as_slice().device(),
        tables.nodes.as_slice().device(),
        tables.mode_nodes.as_slice().device(),
        tables.incoming_powers.as_slice().device(),
        tables.cos.as_slice().device(),
        tables.sin.as_slice().device(),
        tables.local_rotations.as_slice().device(),
        scratch.incoming.as_slice().device(),
        locals.device(),
    ] {
        device.check_owner(owner)?;
    }
    scratch.check(plan, tables);
    let (p, n) = (tables.p, coefficients(tables.p));
    assert_eq!(
        locals.len(),
        plan.locals_bound * n,
        "plane_wave::local: the locals hold {} values, not {} columns of {n}",
        locals.len(),
        plan.locals_bound
    );
    assert!(
        u32::try_from(locals.offset() + locals.len()).is_ok(),
        "plane_wave::local: the locals exceed the 32-bit index range"
    );
    layout.check(device.info(), p, tables.modes, T::FLOAT)?;
    let items = plan.nlocals;
    if items == 0 {
        return Ok(());
    }
    let (th, tl) = plan.locals.as_slice().binding();
    let (gh, gl) = plan.incoming.as_slice().binding();
    let (nh, nl) = tables.nodes.as_slice().binding();
    let (mh, ml) = tables.mode_nodes.as_slice().binding();
    let (ph, pl) = tables.incoming_powers.as_slice().binding();
    let (ch, cl) = tables.cos.as_slice().binding();
    let (zh, zl) = tables.sin.as_slice().binding();
    let (rh, rl) = tables.local_rotations.as_slice().binding();
    let (wh, wl) = scratch.incoming.as_slice().binding();
    let (lh, ll) = locals.binding();
    let scalars = (
        items as u32,
        locals.offset() as u32,
        tables.h as u32,
        tables.lambdas as u32,
    );
    let client = device.client().clone();
    let (cubes, units, cube) = match layout {
        PlaneWaveLayout::Cube { units } => {
            let (x, y) = cube_grid(device.info(), items);
            (CubeCount::Static(x, y, 1), units, true)
        }
        PlaneWaveLayout::Cpu => (CubeCount::Static(1, 1, 1), cpu_units(device, items), false),
    };
    // SAFETY: every handle is a whole buffer with its element count. Item l < nlocals reads
    // its column < locals_bound (checked at the plan's build), so its local lies at the
    // offset plus t (p + 1)² + k < locals.len() (asserted), and its six incoming blocks,
    // each NO_BLOCK (not read) or < G (checked), read at block · 2H + j within the
    // incoming blocks (at least G blocks, checked against the plan). The tables are read as in the
    // outgoing step (the incoming powers s (p + 1), the local rotations two frames of
    // blocks_len(p)). The columns of one launch are distinct (checked at the plan's
    // build), so each local has one owner: in the cube layout unit u owns slots u, u + U, …
    // of its item's local, and shared memory holds (p + 1)² + 4F values written by their
    // owners between barriers every unit reaches (the item and its frames are the cube's);
    // the CPU layout's units cover disjoint ranges of the items.
    unsafe {
        local_kernel::launch_unchecked::<T>(
            &client,
            cubes,
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(th, tl),
            BufferArg::from_raw_parts(gh, gl),
            BufferArg::from_raw_parts(nh, nl),
            BufferArg::from_raw_parts(mh, ml),
            BufferArg::from_raw_parts(ph, pl),
            BufferArg::from_raw_parts(ch, cl),
            BufferArg::from_raw_parts(zh, zl),
            BufferArg::from_raw_parts(rh, rl),
            BufferArg::from_raw_parts(wh, wl),
            BufferArg::from_raw_parts(lh, ll),
            scalars.0,
            scalars.1,
            scalars.2,
            scalars.3,
            p,
            tables.modes,
            units as usize,
            cube,
        );
    }
    device.count_launch();
    Ok(())
}

/// One level call of plane-wave M2L (module documentation): [`outgoing`], [`merge`],
/// [`gather`] and [`local`] in order, into `locals`. At most four launches; none for a
/// step without items. Allocates nothing.
///
/// # Errors
///
/// As [`outgoing`].
///
/// # Panics
///
/// As the four steps.
pub fn m2l<T: DeviceFloat>(
    device: &mut Device,
    layout: PlaneWaveLayout,
    plan: &PlaneWavePlan,
    tables: &PlaneWaveTables<T>,
    multipoles: DeviceSlice<'_, T>,
    locals: DeviceSliceMut<'_, T>,
    scratch: &mut PlaneWaveScratch<T>,
) -> Result<(), KernelError> {
    outgoing(device, layout, plan, tables, multipoles, scratch)?;
    merge(device, layout, plan, tables, scratch)?;
    gather(device, layout, plan, tables, scratch)?;
    local(device, layout, plan, tables, scratch, locals)
}

// --- One value of each step ------------------------------------------------------------

/// The degree n of slot k (n² ≤ k < (n + 1)²).
#[cube]
fn degree_of(k: usize) -> usize {
    let mut n = 0usize;
    while (n + 1) * (n + 1) <= k {
        n += 1;
    }
    n
}

/// The cube's item, or the first of this unit's range with its count: in the cube layout
/// one item per cube (none past `items`), in the CPU layout the unit's contiguous range.
#[cube]
fn item_range(items: usize, #[comptime] cube: bool) -> (usize, usize) {
    let mut first = 0usize;
    let mut count = 0usize;
    if comptime!(cube) {
        let c = CUBE_POS_Y as usize * CUBE_COUNT_X as usize + CUBE_POS_X as usize;
        if c < items {
            first = c;
            count = 1;
        }
    } else {
        let (f, n) = unit_rows(items);
        first = f;
        count = n;
    }
    (first, count)
}

/// The first output of this unit and the stride: unit u and U in the cube layout, 0 and 1
/// in the CPU layout (one unit per item).
#[cube]
fn unit_stride(#[comptime] units: usize, #[comptime] cube: bool) -> (usize, usize) {
    let mut first = 0usize;
    let mut stride = 1usize;
    if comptime!(cube) {
        first = UNIT_POS as usize;
        stride = units;
    }
    (first, stride)
}

/// A barrier in the cube layout; nothing in the CPU layout.
#[cube]
fn barrier(#[comptime] cube: bool) {
    if comptime!(cube) {
        sync_cube();
    }
}

/// Slot k of the multipole at `x[x0..]` in `frame` (`rotate_multipole`): frame 0 copies;
/// frames 1 and 2 form slot (n, i) from zero as an fma per input slot j of the degree in
/// ascending order, with the transposed block entry (i, j).
#[cube]
fn rotated_slot<F: Float>(
    x: &[F],
    x0: usize,
    rotations: &[F],
    frame: usize,
    k: usize,
    #[comptime] p: usize,
) -> F {
    let mut value = x[x0 + k];
    if frame > 0 {
        let blocks = comptime!((p + 1) * (2 * p + 1) * (2 * p + 3) / 3);
        let n = degree_of(k);
        let i = k - n * n;
        let width = 2 * n + 1;
        // Σ_{n' < n} (2n' + 1)² = (4n³ − n)/3.
        let start = (frame - 1) * blocks + (4 * n * n * n - n) / 3;
        let mut acc = F::new(0.0f32);
        for j in 0..width {
            acc = fma(rotations[start + j * width + i], x[x0 + n * n + j], acc);
        }
        value = acc;
    }
    value
}

/// One sum of A of `outgoing_modes` for node k and order m: Σ over the degrees n from
/// `first` to p in steps of 2, ascending, from zero, an fma per term of the power and the
/// coefficient (n, m), or with `negative` (n, −m).
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a step's arguments are its buffers and the mode"
)]
fn mode_sum<F: Float, L: List<F> + ?Sized>(
    rotated: &L,
    powers: &[F],
    lambdas: usize,
    k: usize,
    m: usize,
    first: usize,
    negative: bool,
    #[comptime] p: usize,
) -> F {
    let mut acc = F::new(0.0f32);
    let mut n = first;
    while n <= p {
        let mut at = n * n + n + m;
        if negative {
            at = n * n + n - m;
        }
        acc = fma(powers[n * lambdas + k], rotated[at], acc);
        n += 2;
    }
    acc
}

/// The weighted coefficients (a, b) of order m from A's real and imaginary parts
/// (`outgoing_modes`): (A_r, 0) at m = 0; otherwise with σ = ±2 the sign of (−i)ᵐ,
/// (σ A_r, −(σ A_i)) for even m and (−(σ A_r), σ A_i) for odd m.
#[cube]
fn weigh<F: Float>(m: usize, ar: F, ai: F) -> (F, F) {
    let mut a = ar;
    let mut b = F::new(0.0f32);
    if m > 0 {
        let mut sign = F::new(2.0f32);
        if m % 4 >= 2 {
            sign = F::new(-2.0f32);
        }
        if m % 2 == 1 {
            a = -(sign * ar);
            b = sign * ai;
        } else {
            a = sign * ar;
            b = -(sign * ai);
        }
    }
    (a, b)
}

/// The modes of the multipole `rotated` (in its frame) into `modes`, four per mode
/// (a↑, b↑, a↓, b↓) (`outgoing_modes`): this unit's modes q = first, first + stride, ….
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a step's arguments are its buffers and the unit's share"
)]
fn write_modes<F: Float, L: List<F> + ?Sized, M: List<F> + ?Sized>(
    rotated: &L,
    modes: &mut M,
    nodes: &[u32],
    mode_nodes: &[u32],
    powers: &[F],
    lambdas: usize,
    first: usize,
    stride: usize,
    #[comptime] p: usize,
    #[comptime] count: usize,
) {
    let mut q = first;
    while q < count {
        let k = mode_nodes[q] as usize;
        let m = q - nodes[5 * k + 3] as usize;
        let er = mode_sum::<F, L>(rotated, powers, lambdas, k, m, m, false, p);
        let or = mode_sum::<F, L>(rotated, powers, lambdas, k, m, m + 1, false, p);
        let mut ei = F::new(0.0f32);
        let mut oi = F::new(0.0f32);
        if m > 0 {
            ei = mode_sum::<F, L>(rotated, powers, lambdas, k, m, m, true, p);
            oi = mode_sum::<F, L>(rotated, powers, lambdas, k, m, m + 1, true, p);
        }
        let (ua, ub) = weigh::<F>(m, er + or, ei + oi);
        let (da, db) = weigh::<F>(m, er - or, ei - oi);
        modes[4 * q] = ua;
        modes[4 * q + 1] = ub;
        modes[4 * q + 2] = da;
        modes[4 * q + 3] = db;
        q += stride;
    }
}

/// Value v of a source's up and down waves from its modes (`waves_from_modes`): v = (dir 2
/// + half) H + j, from zero, for the modes m of the half's parity ascending, fma(a, cos)
/// and fma(b, sin) with the direction's (a, b).
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a step's arguments are its buffers and the value"
)]
fn wave_value<F: Float, L: List<F> + ?Sized>(
    modes: &L,
    nodes: &[u32],
    value_nodes: &[u32],
    cos: &[F],
    sin: &[F],
    h: usize,
    v: usize,
) -> F {
    let dir = v / (2 * h);
    let half = (v % (2 * h)) / h;
    let j = v % h;
    let k = value_nodes[j] as usize;
    let start = nodes[5 * k] as usize;
    let len = nodes[5 * k + 1] as usize;
    let count = nodes[5 * k + 2] as usize;
    let mode_start = nodes[5 * k + 3] as usize;
    let trig = nodes[5 * k + 4] as usize;
    let jj = j - start;
    let mut acc = F::new(0.0f32);
    let mut m = half;
    while m < count {
        let q = 4 * (mode_start + m) + 2 * dir;
        let row = trig + m * len + jj;
        acc = fma(modes[q], cos[row], acc);
        acc = fma(modes[q + 1], sin[row], acc);
        m += 2;
    }
    acc
}

/// Sum v of the incoming sums of one frame (`sums_from_waves`): v = 4 q + 2 dir + c for
/// mode q, the up (dir 0) or down (dir 1) block, cos (c = 0) or sin (c = 1): over the
/// node's values j ascending, from zero, fma(x, trig) with x the half of m's parity. Zero
/// for a block [`NO_BLOCK`].
#[cube]
#[allow(
    clippy::too_many_arguments,
    reason = "a step's arguments are its buffers and the sum"
)]
fn sum_value<F: Float>(
    incoming: &[F],
    blocks: (u32, u32),
    nodes: &[u32],
    mode_nodes: &[u32],
    cos: &[F],
    sin: &[F],
    h: usize,
    v: usize,
) -> F {
    let q = v / 4;
    let dir = (v % 4) / 2;
    let use_sin = v % 2 == 1;
    let mut block = blocks.0;
    if dir == 1 {
        block = blocks.1;
    }
    let mut acc = F::new(0.0f32);
    if block != NO_BLOCK {
        let k = mode_nodes[q] as usize;
        let start = nodes[5 * k] as usize;
        let len = nodes[5 * k + 1] as usize;
        let m = q - nodes[5 * k + 3] as usize;
        let trig = nodes[5 * k + 4] as usize + m * len;
        let x0 = block as usize * 2 * h + (m % 2) * h + start;
        for jj in 0..len {
            let mut c = cos[trig + jj];
            if use_sin {
                c = sin[trig + jj];
            }
            acc = fma(incoming[x0 + jj], c, acc);
        }
    }
    acc
}

/// (a, b) times iᵐ: (a, b), (−b, a), (−a, −b), (b, −a) for m mod 4 = 0, 1, 2, 3.
#[cube]
fn times_i_pow<F: Float>(m: usize, a: F, b: F) -> (F, F) {
    let r = m % 4;
    let mut x = a;
    let mut y = b;
    if r == 1 {
        x = -b;
        y = a;
    } else if r == 2 {
        x = -a;
        y = -b;
    } else if r == 3 {
        x = b;
        y = -a;
    }
    (x, y)
}

/// Slot k = (n, ±m) of the local in the frame from the sums (`local_from_sums`): from zero,
/// for each λ node with more than m modes in order, an fma of 2 λₖⁿ 2^{−n} and the node's
/// value: G(m)/2 of the up and down sums, combined as down + up (n − m even) or down − up
/// (odd), times iᵐ, its real part at +m and its imaginary part at −m.
#[cube]
fn local_slot<F: Float, L: List<F> + ?Sized>(
    sums: &L,
    nodes: &[u32],
    powers: &[F],
    lambdas: usize,
    k: usize,
    #[comptime] p: usize,
) -> F {
    let n = degree_of(k);
    let order = k as i32 - (n * n + n) as i32;
    let mut mi = order;
    if order < 0 {
        mi = -order;
    }
    let m = mi as usize;
    let mut acc = F::new(0.0f32);
    for kk in 0..lambdas {
        if m < nodes[5 * kk + 2] as usize {
            let q = 4 * (nodes[5 * kk + 3] as usize + m);
            let (s0, s1, s2, s3) = (sums[q], sums[q + 1], sums[q + 2], sums[q + 3]);
            // G(m)/2: (C, −S) for even m, (S, C) for odd m.
            let mut up = (s0, -s1);
            let mut down = (s2, -s3);
            if m % 2 == 1 {
                up = (s1, s0);
                down = (s3, s2);
            }
            let mut pair = (up.0 + down.0, up.1 + down.1);
            if (n - m) % 2 == 1 {
                pair = (down.0 - up.0, down.1 - up.1);
            }
            let (re, im) = times_i_pow::<F>(m, pair.0, pair.1);
            let mut value = re;
            if order < 0 {
                value = im;
            }
            acc = fma(powers[kk * comptime!(p + 1) + n], value, acc);
        }
    }
    acc
}

/// Slot k = (n, i) of the frame's local `frame_local` rotated back from `frame` and added
/// to `acc` (`add_local_from_frame`): frame 0 adds it; frames 1 and 2 an fma per input
/// slot j of the degree ascending, with the transposed block entry (i, j).
#[cube]
fn add_back<F: Float, L: List<F> + ?Sized>(
    frame_local: &L,
    rotations: &[F],
    frame: usize,
    k: usize,
    acc: F,
    #[comptime] p: usize,
) -> F {
    let mut value = acc;
    if frame == 0 {
        value += frame_local[k];
    } else {
        let blocks = comptime!((p + 1) * (2 * p + 1) * (2 * p + 3) / 3);
        let n = degree_of(k);
        let i = k - n * n;
        let width = 2 * n + 1;
        let start = (frame - 1) * blocks + (4 * n * n * n - n) / 3;
        for j in 0..width {
            value = fma(
                rotations[start + j * width + i],
                frame_local[n * n + j],
                value,
            );
        }
    }
    value
}

// --- Kernels ---------------------------------------------------------------------------

/// The outgoing step: per item (source, frame) the rotated multipole, its modes and its
/// waves (module documentation).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn outgoing_kernel<F: Float>(
    sources: &[u32],
    nodes: &[u32],
    value_nodes: &[u32],
    mode_nodes: &[u32],
    powers: &[F],
    cos: &[F],
    sin: &[F],
    rotations: &[F],
    input: &[F],
    waves: &mut [F],
    items: u32,
    input_offset: u32,
    h: u32,
    lambdas: u32,
    #[comptime] p: usize,
    #[comptime] modes: usize,
    #[comptime] units: usize,
    #[comptime] cube: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let fm = comptime!(4 * modes);
    let h = h as usize;
    let (first_item, count) = item_range(items as usize, cube);
    let (first, stride) = unit_stride(units, cube);
    if comptime!(cube) {
        let mut rotated = Shared::<[F]>::new_slice(nc);
        let mut weighted = Shared::<[F]>::new_slice(fm);
        // The item is the cube's, so every unit reaches every barrier or none does.
        if count > 0 {
            let c = first_item;
            let (i, frame) = (c / 3, c % 3);
            let x0 = input_offset as usize + sources[i] as usize * nc;
            let mut k = first;
            while k < nc {
                rotated[k] = rotated_slot::<F>(input, x0, rotations, frame, k, p);
                k += stride;
            }
            barrier(cube);
            write_modes::<F, Shared<[F]>, Shared<[F]>>(
                &rotated,
                &mut weighted,
                nodes,
                mode_nodes,
                powers,
                lambdas as usize,
                first,
                stride,
                p,
                modes,
            );
            barrier(cube);
            let out = (6 * i + 2 * frame) * 2 * h;
            let mut v = first;
            while v < 4 * h {
                waves[out + v] =
                    wave_value::<F, Shared<[F]>>(&weighted, nodes, value_nodes, cos, sin, h, v);
                v += stride;
            }
        }
    } else {
        let mut rotated = Array::<F>::new(nc);
        let mut weighted = Array::<F>::new(fm);
        for c in first_item..first_item + count {
            let (i, frame) = (c / 3, c % 3);
            let x0 = input_offset as usize + sources[i] as usize * nc;
            for k in 0..nc {
                rotated[k] = rotated_slot::<F>(input, x0, rotations, frame, k, p);
            }
            write_modes::<F, Array<F>, Array<F>>(
                &rotated,
                &mut weighted,
                nodes,
                mode_nodes,
                powers,
                lambdas as usize,
                0,
                1,
                p,
                modes,
            );
            let out = (6 * i + 2 * frame) * 2 * h;
            for v in 0..4 * h {
                waves[out + v] =
                    wave_value::<F, Array<F>>(&weighted, nodes, value_nodes, cos, sin, h, v);
            }
        }
    }
}

/// The merge step: per accumulator, each value from zero over its entries in order,
/// re ← fma(x_r, t_r), fma(−x_i, t_i), im ← fma(x_r, t_i), fma(x_i, t_r) (`shift_add`).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    reason = "a kernel's arguments are its buffers and scalars"
)]
fn merge_kernel<F: Float>(
    offsets: &[u32],
    entries: &[u32],
    shifts: &[F],
    waves: &[F],
    accumulators: &mut [F],
    items: u32,
    h: u32,
    #[comptime] units: usize,
    #[comptime] cube: bool,
) {
    let h = h as usize;
    let (first_item, count) = item_range(items as usize, cube);
    let (first, stride) = unit_stride(units, cube);
    for a in first_item..first_item + count {
        let (begin, end) = (offsets[a] as usize, offsets[a + 1] as usize);
        let out = a * 2 * h;
        let mut j = first;
        while j < h {
            let mut re = F::new(0.0f32);
            let mut im = F::new(0.0f32);
            for e in begin..end {
                let x = entries[2 * e] as usize * 2 * h + j;
                let t = entries[2 * e + 1] as usize * 2 * h + j;
                let (xr, xi) = (waves[x], waves[x + h]);
                let (tr, ti) = (shifts[t], shifts[t + h]);
                re = fma(xr, tr, re);
                re = fma(-xi, ti, re);
                im = fma(xr, ti, im);
                im = fma(xi, tr, im);
            }
            accumulators[out + j] = re;
            accumulators[out + h + j] = im;
            j += stride;
        }
    }
}

/// The gather step: per incoming block, each value the sum of its accumulators in order
/// from zero, then re = fma(−x_i, t_i, x_r t_r) and im = fma(x_r, t_i, x_i t_r) with its
/// child shift (`add`, `shift`).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn gather_kernel<F: Float>(
    child: &[u32],
    offsets: &[u32],
    entries: &[u32],
    shifts: &[F],
    accumulators: &[F],
    incoming: &mut [F],
    items: u32,
    h: u32,
    #[comptime] units: usize,
    #[comptime] cube: bool,
) {
    let h = h as usize;
    let (first_item, count) = item_range(items as usize, cube);
    let (first, stride) = unit_stride(units, cube);
    for g in first_item..first_item + count {
        let (begin, end) = (offsets[g] as usize, offsets[g + 1] as usize);
        let t0 = child[g] as usize * 2 * h;
        let out = g * 2 * h;
        let mut j = first;
        while j < h {
            let mut xr = F::new(0.0f32);
            let mut xi = F::new(0.0f32);
            for e in begin..end {
                let a = entries[e] as usize * 2 * h + j;
                xr += accumulators[a];
                xi += accumulators[a + h];
            }
            let (tr, ti) = (shifts[t0 + j], shifts[t0 + h + j]);
            incoming[out + j] = fma(-xi, ti, xr * tr);
            incoming[out + h + j] = fma(xr, ti, xi * tr);
            j += stride;
        }
    }
}

/// The local step: per item its local loaded, each frame with an incoming block added
/// (sums, the local in the frame, rotated back), and stored (module documentation).
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    reason = "a kernel's arguments are its buffers and scalars, and #[cube] code has no iterators"
)]
fn local_kernel<F: Float>(
    columns: &[u32],
    blocks: &[u32],
    nodes: &[u32],
    mode_nodes: &[u32],
    powers: &[F],
    cos: &[F],
    sin: &[F],
    rotations: &[F],
    incoming: &[F],
    locals: &mut [F],
    items: u32,
    locals_offset: u32,
    h: u32,
    lambdas: u32,
    #[comptime] p: usize,
    #[comptime] modes: usize,
    #[comptime] units: usize,
    #[comptime] cube: bool,
) {
    let nc = comptime!((p + 1) * (p + 1));
    let fm = comptime!(4 * modes);
    let h = h as usize;
    let lambdas = lambdas as usize;
    let (first_item, count) = item_range(items as usize, cube);
    if comptime!(cube) {
        let per_unit = comptime!(((p + 1) * (p + 1)).div_ceil(units));
        let mut sums = Shared::<[F]>::new_slice(fm);
        let mut frame_local = Shared::<[F]>::new_slice(nc);
        // The item and its frames are the cube's, so every unit reaches every barrier or
        // none does.
        if count > 0 {
            let l = first_item;
            let unit = UNIT_POS as usize;
            let out = locals_offset as usize + columns[l] as usize * nc;
            let mut acc = Array::<F>::new(per_unit);
            #[unroll]
            for r in 0..per_unit {
                let k = unit + r * units;
                let mut value = F::new(0.0f32);
                if k < nc {
                    value = locals[out + k];
                }
                acc[r] = value;
            }
            for frame in 0..3usize {
                let up = blocks[6 * l + 2 * frame];
                let down = blocks[6 * l + 2 * frame + 1];
                if up != NO_BLOCK || down != NO_BLOCK {
                    let mut v = unit;
                    while v < fm {
                        sums[v] =
                            sum_value::<F>(incoming, (up, down), nodes, mode_nodes, cos, sin, h, v);
                        v += units;
                    }
                    sync_cube();
                    let mut k = unit;
                    while k < nc {
                        frame_local[k] =
                            local_slot::<F, Shared<[F]>>(&sums, nodes, powers, lambdas, k, p);
                        k += units;
                    }
                    sync_cube();
                    #[unroll]
                    for r in 0..per_unit {
                        let k = unit + r * units;
                        if k < nc {
                            acc[r] = add_back::<F, Shared<[F]>>(
                                &frame_local,
                                rotations,
                                frame,
                                k,
                                acc[r],
                                p,
                            );
                        }
                    }
                    sync_cube();
                }
            }
            #[unroll]
            for r in 0..per_unit {
                let k = unit + r * units;
                if k < nc {
                    locals[out + k] = acc[r];
                }
            }
        }
    } else {
        let mut acc = Array::<F>::new(nc);
        let mut sums = Array::<F>::new(fm);
        let mut frame_local = Array::<F>::new(nc);
        for l in first_item..first_item + count {
            let out = locals_offset as usize + columns[l] as usize * nc;
            for k in 0..nc {
                acc[k] = locals[out + k];
            }
            for frame in 0..3usize {
                let up = blocks[6 * l + 2 * frame];
                let down = blocks[6 * l + 2 * frame + 1];
                if up != NO_BLOCK || down != NO_BLOCK {
                    for v in 0..fm {
                        sums[v] =
                            sum_value::<F>(incoming, (up, down), nodes, mode_nodes, cos, sin, h, v);
                    }
                    for k in 0..nc {
                        frame_local[k] =
                            local_slot::<F, Array<F>>(&sums, nodes, powers, lambdas, k, p);
                    }
                    for k in 0..nc {
                        acc[k] =
                            add_back::<F, Array<F>>(&frame_local, rotations, frame, k, acc[k], p);
                    }
                }
            }
            for k in 0..nc {
                locals[out + k] = acc[k];
            }
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
    fn layouts_fit_or_are_refused() {
        assert_eq!(
            PlaneWaveLayout::default_for(&info(BackendKind::Cpu)),
            PlaneWaveLayout::Cpu
        );
        let metal = info(BackendKind::Metal);
        let default = PlaneWaveLayout::default_for(&metal);
        assert_eq!(default, PlaneWaveLayout::Cube { units: GPU_UNITS });
        // p = 18 with 608 modes: (361 + 2432) values, 11,172 bytes in f32 and 22,344 in
        // f64, against 16 KB of shared memory.
        let small = DeviceInfo {
            max_shared_memory: 16_384,
            ..metal.clone()
        };
        default.check(&small, 18, 608, Precision::F32).unwrap();
        assert!(default.check(&small, 18, 608, Precision::F64).is_err());
        assert!(
            PlaneWaveLayout::Cube { units: 0 }
                .check(&metal, 8, 100, Precision::F32)
                .is_err()
        );
        assert!(
            PlaneWaveLayout::Cube { units: 2048 }
                .check(&metal, 8, 100, Precision::F32)
                .is_err()
        );
        PlaneWaveLayout::Cpu
            .check(&metal, 20, 10_000, Precision::F64)
            .unwrap();
    }

    #[test]
    fn blocks_len_is_the_sum_over_degrees() {
        for p in [0, 1, 8, 20] {
            let sum: usize = (0..=p).map(|n| (2 * n + 1) * (2 * n + 1)).sum();
            assert_eq!(blocks_len(p), sum);
        }
    }

    #[test]
    fn csr_offsets_are_accepted() {
        check_csr("merge", &[0, 1, 1, 3], 3);
        check_csr("merge", &[0], 0);
    }

    #[test]
    #[should_panic(expected = "do not start at 0")]
    fn csr_offsets_start_at_zero() {
        check_csr("merge", &[1, 2], 2);
    }

    #[test]
    #[should_panic(expected = "decrease")]
    fn csr_offsets_do_not_decrease() {
        check_csr("merge", &[0, 2, 1], 1);
    }

    #[test]
    #[should_panic(expected = "not at the 3 entries")]
    fn csr_offsets_end_at_the_entries() {
        check_csr("merge", &[0, 2], 3);
    }
}
