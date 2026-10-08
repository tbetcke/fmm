//! Data movement kernels: zeroing, gathering and scattering columns, and scattering
//! values by index (fmm-plan-redesign §10, the GEMM check of §6.4; device-path.md §3.1),
//! and (Phase 4S T9) the output pass of `nd-fmm-exec`, [`gather_output`], which gathers
//! the leaf-ordered target output into the caller's order and scales it, in f64 on a
//! device with f64 arithmetic.
//!
//! Each function is a safe launch wrapper over one `#[cube]` kernel, generic over the
//! element type. It checks its arguments on the host (lengths, the owning device, the
//! index bound of [`IndexSlice::bound`], distinct indices in debug builds), so the
//! kernel launches without bounds checks, and it allocates nothing. A call with nothing
//! to do launches nothing. Every operation is a copy, except the single add per
//! element of [`scatter_add_columns`] and the f64 division and rounding of
//! [`gather_output`]; the results equal plain host loops bit for bit
//! on every backend (spikes/device-arith/REPORT.md, "Recommendation", rule 6), where the
//! scatter-add's operands and sum are normal (Metal flushes subnormal sums,
//! CONVENTIONS §3.13, "Device kernels").
//!
//! Matrices are column-major with columns of n values: column c of a slice `x` is
//! `x[c n..(c + 1) n]` (the `LevelBuffers` layout, CONVENTIONS §3.6).

use cubecl::prelude::*;

use crate::buffer::{
    DeviceBuffer, DeviceElement, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexBuffer, IndexSlice,
};
use crate::device::{Device, Precision};
use crate::error::KernelError;

/// x[offset..offset + len] = 0, in blocks of `chunk` elements per unit and stride.
#[cube(launch_unchecked)]
fn zero_kernel<E: Numeric>(x: &mut [E], offset: u32, len: u32, chunk: u32, threads: u32) {
    let (offset, len, chunk) = (offset as usize, len as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < len {
        let mut end = start + chunk;
        if end > len {
            end = len;
        }
        for i in start..end {
            x[offset + i] = E::from_int(0);
        }
        start += stride;
    }
}

/// y[y0 + e] = x[x0 + idx[i0 + e / n] n + e % n] for e < work: gathered columns.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gather_kernel<E: Numeric>(
    x: &[E],
    x_offset: u32,
    indices: &[u32],
    index_offset: u32,
    y: &mut [E],
    y_offset: u32,
    work: u32,
    chunk: u32,
    threads: u32,
    #[comptime] n: usize,
) {
    let (x_offset, index_offset, y_offset) =
        (x_offset as usize, index_offset as usize, y_offset as usize);
    let (work, chunk) = (work as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < work {
        let mut end = start + chunk;
        if end > work {
            end = work;
        }
        for e in start..end {
            let column = indices[index_offset + e / n] as usize;
            y[y_offset + e] = x[x_offset + column * n + e % n];
        }
        start += stride;
    }
}

/// y[y0 + e] = x[x0 + idx[i0 + b w + c] n + k] for e = (b n + k) w + c < work: gathered
/// columns, coefficient-major in blocks of w columns.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gather_coefficients_kernel<E: Numeric>(
    x: &[E],
    x_offset: u32,
    indices: &[u32],
    index_offset: u32,
    y: &mut [E],
    y_offset: u32,
    block: u32,
    work: u32,
    chunk: u32,
    threads: u32,
    #[comptime] n: usize,
) {
    let (x_offset, index_offset, y_offset) =
        (x_offset as usize, index_offset as usize, y_offset as usize);
    let (block, work, chunk) = (block as usize, work as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < work {
        let mut end = start + chunk;
        if end > work {
            end = work;
        }
        for e in start..end {
            let row = e / block;
            let b = row / n;
            let k = row - b * n;
            let column = indices[index_offset + b * block + e - row * block] as usize;
            y[y_offset + e] = x[x_offset + column * n + k];
        }
        start += stride;
    }
}

/// x[x0 + idx[i0 + e / n] n + e % n] += y[y0 + e] for e < work: one add per element.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scatter_add_kernel<F: Float>(
    y: &[F],
    y_offset: u32,
    indices: &[u32],
    index_offset: u32,
    x: &mut [F],
    x_offset: u32,
    work: u32,
    chunk: u32,
    threads: u32,
    #[comptime] n: usize,
) {
    let (x_offset, index_offset, y_offset) =
        (x_offset as usize, index_offset as usize, y_offset as usize);
    let (work, chunk) = (work as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < work {
        let mut end = start + chunk;
        if end > work {
            end = work;
        }
        for e in start..end {
            let column = indices[index_offset + e / n] as usize;
            let k = x_offset + column * n + e % n;
            x[k] += y[y_offset + e];
        }
        start += stride;
    }
}

/// x[x0 + idx[i0 + e]] = y[y0 + e] for e < work.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn scatter_values_kernel<E: Numeric>(
    y: &[E],
    y_offset: u32,
    indices: &[u32],
    index_offset: u32,
    x: &mut [E],
    x_offset: u32,
    work: u32,
    chunk: u32,
    threads: u32,
) {
    let (x_offset, index_offset, y_offset) =
        (x_offset as usize, index_offset as usize, y_offset as usize);
    let (work, chunk) = (work as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < work {
        let mut end = start + chunk;
        if end > work {
            end = work;
        }
        for e in start..end {
            x[x_offset + indices[index_offset + e] as usize] = y[y_offset + e];
        }
        start += stride;
    }
}

/// The caller-ordered output of [`gather_output`]: for every target i < `work`, with
/// j = leaves[i], P = offsets[j] and k = points[i] − P, out[i] = F(f64(store[o P + k]) /
/// scales[2 j]), and with gradients (o = 4, n = offsets[j + 1] − P) out[work + 3 i + c] =
/// F(f64(store[4 P + n + 3 k + c]) / scales[2 j + 1]) for c < 3. One f64 division and
/// one rounding to F per value.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn gather_output_kernel<F: Float>(
    store: &[F],
    store_offset: u32,
    offsets: &[u32],
    points: &[u32],
    leaves: &[u32],
    scales: &[f64],
    out: &mut [F],
    out_offset: u32,
    work: u32,
    chunk: u32,
    threads: u32,
    #[comptime] gradients: bool,
) {
    let (store_offset, out_offset) = (store_offset as usize, out_offset as usize);
    let (work, chunk) = (work as usize, chunk as usize);
    let stride = threads as usize * chunk;
    let mut start = ABSOLUTE_POS * chunk;
    while start < work {
        let mut end = start + chunk;
        if end > work {
            end = work;
        }
        for i in start..end {
            let j = leaves[i] as usize;
            let first = offsets[j] as usize;
            let k = points[i] as usize - first;
            if gradients {
                let count = offsets[j + 1] as usize - first;
                let base = store_offset + 4 * first;
                out[out_offset + i] = F::cast_from(f64::cast_from(store[base + k]) / scales[2 * j]);
                let g = base + count + 3 * k;
                let scale = scales[2 * j + 1];
                let o = out_offset + work + 3 * i;
                out[o] = F::cast_from(f64::cast_from(store[g]) / scale);
                out[o + 1] = F::cast_from(f64::cast_from(store[g + 1]) / scale);
                out[o + 2] = F::cast_from(f64::cast_from(store[g + 2]) / scale);
            } else {
                out[out_offset + i] =
                    F::cast_from(f64::cast_from(store[store_offset + first + k]) / scales[2 * j]);
            }
        }
        start += stride;
    }
}

/// Where every target's output lies in a leaf-ordered target output, and the scales
/// of its leaf: the input of [`gather_output`] (Phase 4S T9), uploaded once per FMM.
///
/// For every target in the caller's order, its point in leaf order (u32) and its leaf
/// (u32); for every leaf, its point offsets (u32, as [`PointOffsets`](crate::view::PointOffsets)
/// holds them) and its two scales in f64. Validated at upload, so that
/// [`gather_output`] reads every target's values without a bounds check: the offsets
/// start at 0 and never decrease, every leaf index is below the number of leaves, and
/// every target's point lies within its leaf.
#[derive(Debug)]
pub struct OutputOrder {
    offsets: IndexBuffer,
    points: IndexBuffer,
    leaves: IndexBuffer,
    scales: DeviceBuffer<f64>,
    total: usize,
}

impl OutputOrder {
    /// Validates the arrays on the host and uploads them (four uploads): `offsets`, one
    /// more than the leaves, the point offsets of the leaf-ordered store (CSR, from 0);
    /// `points[i]` and `leaves[i]`, target i's point in leaf order and its leaf; and
    /// `scales`, two per leaf (`[2 j]` divides φ, `[2 j + 1]` ∇φ). The targets need not
    /// be every point, nor distinct.
    ///
    /// # Errors
    ///
    /// [`KernelError::UnsupportedPrecision`] on a device without f64 arithmetic (Metal),
    /// before anything is uploaded; otherwise as [`Device::upload`].
    ///
    /// # Panics
    ///
    /// If `offsets` is empty, does not start at 0 or decreases; if `points` and `leaves`
    /// differ in length or `scales` does not hold two values per leaf; if a leaf index is
    /// not below the number of leaves, or a point does not lie within its leaf.
    pub fn upload(
        device: &mut Device,
        offsets: &[u32],
        points: &[u32],
        leaves: &[u32],
        scales: &[f64],
    ) -> Result<Self, KernelError> {
        device.require(Precision::F64)?;
        assert!(
            offsets.first() == Some(&0) && offsets.windows(2).all(|w| w[0] <= w[1]),
            "OutputOrder: the offsets must start at 0 and never decrease"
        );
        let nleaves = offsets.len() - 1;
        assert_eq!(
            points.len(),
            leaves.len(),
            "OutputOrder: one leaf per target"
        );
        assert_eq!(
            scales.len(),
            2 * nleaves,
            "OutputOrder: two scales per leaf"
        );
        for (i, (&s, &j)) in points.iter().zip(leaves).enumerate() {
            let j = j as usize;
            assert!(
                j < nleaves && offsets[j] <= s && s < offsets[j + 1],
                "OutputOrder: target {i} at point {s} does not lie in its leaf {j}"
            );
        }
        Ok(Self {
            offsets: device.upload_indices(offsets)?,
            points: device.upload_indices(points)?,
            leaves: device.upload_indices(leaves)?,
            scales: device.upload(scales)?,
            total: offsets[nleaves] as usize,
        })
    }

    /// The number of targets.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// True if there is no target.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// The number of leaves.
    pub fn nleaves(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The points of every leaf together: the last offset.
    pub fn total(&self) -> usize {
        self.total
    }

    /// The bytes of the four buffers on the device, for a memory check before uploading:
    /// `targets` targets and `nleaves` leaves.
    pub fn bytes(targets: usize, nleaves: usize) -> u64 {
        Device::buffer_bytes::<u32>(nleaves + 1)
            + 2 * Device::buffer_bytes::<u32>(targets)
            + Device::buffer_bytes::<f64>(2 * nleaves)
    }

    fn device(&self) -> u64 {
        self.scales.as_slice().device()
    }
}

/// The output pass on the device (Phase 4S T9): φ and ∇φ of every target of `order` in
/// its order, from the leaf-ordered target output `store` (CONVENTIONS §3.13, "Target
/// input and output": per leaf, the φ̂ of its points, then with `gradients` their ĝ, three
/// values each), each value divided by its leaf's scale in f64 and rounded once to T:
/// `out[i] = T(f64(φ̂) / scales[2 j])`, and with `gradients` `out[n + 3 i + c] =
/// T(f64(ĝ_c) / scales[2 j + 1])` for the n targets. Bit for bit the host loop
/// `T::from_f64(x.to_f64() / scale)` where the device divides f64 correctly rounded and
/// rounds to nearest (the CPU runtime and CUDA, device-path.md F27, F33), and subnormal
/// results are not flushed.
///
/// # Errors
///
/// [`KernelError::UnsupportedPrecision`] on a device without f64 arithmetic (an
/// [`OutputOrder`] cannot be uploaded there either); [`KernelError::WrongDevice`] if a
/// buffer belongs to another device.
///
/// # Panics
///
/// If `store` does not hold o values per point of `order` (o = 4 with `gradients`, else
/// 1), or `out` does not hold o values per target.
pub fn gather_output<T: DeviceFloat>(
    device: &mut Device,
    order: &OutputOrder,
    store: DeviceSlice<'_, T>,
    gradients: bool,
    out: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    device.require(Precision::F64)?;
    check_owners(device, &[order.device(), store.device(), out.device()])?;
    let o = if gradients { 4 } else { 1 };
    assert_eq!(
        store.len(),
        o * order.total(),
        "gather_output: a store of {} points of {o} values",
        order.total()
    );
    assert_eq!(
        out.len(),
        o * order.len(),
        "gather_output: an output of {} targets of {o} values",
        order.len()
    );
    if order.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(order.len());
    let (sh, sl) = store.binding();
    let (oh, ol) = order.offsets.as_slice().binding();
    let (ph, pl) = order.points.as_slice().binding();
    let (lh, ll) = order.leaves.as_slice().binding();
    let (ch, cl) = order.scales.as_slice().binding();
    let (yh, yl) = out.binding();
    // SAFETY: each handle is a whole buffer with its element count, as `from_raw_parts`
    // requires. For i < order.len() the kernel reads points[i] and leaves[i] (within
    // their buffers); `OutputOrder::upload` checked j = leaves[i] < nleaves and
    // offsets[j] ≤ points[i] < offsets[j + 1], so offsets[j], offsets[j + 1] and
    // scales[2 j + 1] lie within their buffers, and the store index o P + k (and with
    // gradients 4 P + n + 3 k + 2 < 4 offsets[j + 1]) is below o total = store.len(),
    // offset by store.offset() within the store's buffer. The writes out.offset() + i and
    // out.offset() + n + 3 i + 2 are below out.offset() + o n = the end of `out`.
    unsafe {
        gather_output_kernel::launch_unchecked::<T>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(sh, sl),
            store.offset() as u32,
            BufferArg::from_raw_parts(oh, ol),
            BufferArg::from_raw_parts(ph, pl),
            BufferArg::from_raw_parts(lh, ll),
            BufferArg::from_raw_parts(ch, cl),
            BufferArg::from_raw_parts(yh, yl),
            out.offset() as u32,
            order.len() as u32,
            grid.chunk,
            grid.threads(),
            gradients,
        );
    }
    device.count_launch();
    Ok(())
}

/// Sets every element of `slice` to zero: +0.0 for floats (the bit pattern 0).
///
/// # Errors
///
/// [`KernelError::WrongDevice`] for a buffer of another device.
pub fn zero<E: DeviceElement>(
    device: &mut Device,
    slice: DeviceSliceMut<'_, E>,
) -> Result<(), KernelError> {
    device.check_owner(slice.device())?;
    if slice.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(slice.len());
    let (handle, len) = slice.binding();
    // SAFETY: `handle` is the whole buffer of `len` elements (at least one is allocated
    // for an empty buffer), as `from_raw_parts` requires. The kernel writes elements
    // offset + i for i < slice.len(), and the slice lies within the buffer
    // (`DeviceBuffer::slice_mut` checked it), so no access is out of bounds.
    unsafe {
        zero_kernel::launch_unchecked::<E>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(handle, len),
            slice.offset() as u32,
            slice.len() as u32,
            grid.chunk,
            grid.threads(),
        );
    }
    device.count_launch();
    Ok(())
}

/// Gathers columns: `y[:, j] = x[:, indices[j]]` for columns of `n` values (`n` is a
/// comptime parameter of the kernel). Indices may repeat.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// If `n` is 0, `x.len()` is not a multiple of `n`, `y.len()` is not
/// `n · indices.len()`, or a non-empty launch has `indices.bound()` above the number
/// of columns of `x`.
pub fn gather_columns<E: DeviceElement>(
    device: &mut Device,
    n: usize,
    x: DeviceSlice<'_, E>,
    indices: IndexSlice<'_>,
    y: DeviceSliceMut<'_, E>,
) -> Result<(), KernelError> {
    check_owners(device, &[x.device(), indices.device(), y.device()])?;
    check_columns("gather_columns", n, x.len(), indices, y.len());
    if y.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(y.len());
    let ((xh, xl), (ih, il), (yh, yl)) = (x.binding(), indices.binding(), y.binding());
    // SAFETY: each handle is a whole buffer with its element count, as `from_raw_parts`
    // requires. The kernel reads indices[offset + e / n] for e < y.len() = n ·
    // indices.len() (within the index slice), reads x at x.offset() + c n + r with
    // c < indices.bound() ≤ x.len() / n and r < n (within x), and writes
    // y[y.offset() + e] (within y); `check_columns` asserted these bounds.
    unsafe {
        gather_kernel::launch_unchecked::<E>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(xh, xl),
            x.offset() as u32,
            BufferArg::from_raw_parts(ih, il),
            indices.offset() as u32,
            BufferArg::from_raw_parts(yh, yl),
            y.offset() as u32,
            y.len() as u32,
            grid.chunk,
            grid.threads(),
            n,
        );
    }
    device.count_launch();
    Ok(())
}

/// Gathers columns coefficient-major: for columns of `n` values and blocks of `block`
/// columns, `y[(b n + k) block + c] = x[k, indices[b block + c]]` (coefficient k of
/// column `indices[j]` of `x`, j = b block + c): within each block of `block` gathered
/// columns, each coefficient's values contiguous (the coefficient-major layout of
/// `translate`, "Orientation", Phase 4 T12). With one block it is the transpose of
/// [`gather_columns`]'s result. A copy: bit for bit a host loop.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// As [`gather_columns`]; and if `block` is 0 or does not divide `indices.len()`.
pub fn gather_coefficients<E: DeviceElement>(
    device: &mut Device,
    n: usize,
    x: DeviceSlice<'_, E>,
    indices: IndexSlice<'_>,
    block: usize,
    y: DeviceSliceMut<'_, E>,
) -> Result<(), KernelError> {
    check_owners(device, &[x.device(), indices.device(), y.device()])?;
    check_columns("gather_coefficients", n, x.len(), indices, y.len());
    assert!(
        block > 0 && indices.len().is_multiple_of(block),
        "gather_coefficients: blocks of {block} columns for {} columns",
        indices.len()
    );
    if y.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(y.len());
    let ((xh, xl), (ih, il), (yh, yl)) = (x.binding(), indices.binding(), y.binding());
    // SAFETY: each handle is a whole buffer with its element count, as `from_raw_parts`
    // requires. For e < y.len() = n · indices.len() the kernel reads the index
    // b block + c < indices.len() (b < indices.len() / block, c < block; within the index
    // slice), reads x at x.offset() + column n + k with column < indices.bound() ≤
    // x.len() / n and k < n (within x), and writes y[y.offset() + e] (within y);
    // `check_columns` and the block assertion established these bounds.
    unsafe {
        gather_coefficients_kernel::launch_unchecked::<E>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(xh, xl),
            x.offset() as u32,
            BufferArg::from_raw_parts(ih, il),
            indices.offset() as u32,
            BufferArg::from_raw_parts(yh, yl),
            y.offset() as u32,
            block as u32,
            y.len() as u32,
            grid.chunk,
            grid.threads(),
            n,
        );
    }
    device.count_launch();
    Ok(())
}

/// Adds columns into a scatter: `x[:, indices[j]] += y[:, j]` for columns of `n`
/// values, one add per element in the order `x + y`.
///
/// **Precondition:** the indices of one launch are distinct, so that no two units
/// write one element (no atomics, README requirement 5). Debug builds check it on the
/// host and panic.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// As [`gather_columns`], with `y` the input and `x` the output; and in debug builds,
/// if an index repeats.
pub fn scatter_add_columns<T: DeviceFloat>(
    device: &mut Device,
    n: usize,
    y: DeviceSlice<'_, T>,
    indices: IndexSlice<'_>,
    x: DeviceSliceMut<'_, T>,
) -> Result<(), KernelError> {
    check_owners(device, &[y.device(), indices.device(), x.device()])?;
    check_columns("scatter_add_columns", n, x.len(), indices, y.len());
    #[cfg(debug_assertions)]
    assert_distinct("scatter_add_columns", indices);
    if y.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(y.len());
    let ((yh, yl), (ih, il), (xh, xl)) = (y.binding(), indices.binding(), x.binding());
    // SAFETY: as in `gather_columns`, with the roles of x and y exchanged: y is read at
    // y.offset() + e for e < y.len(), x read and written at x.offset() + c n + r with
    // c < indices.bound() ≤ x.len() / n and r < n. Distinct indices (the documented
    // precondition) make the writes of different units disjoint.
    unsafe {
        scatter_add_kernel::launch_unchecked::<T>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(yh, yl),
            y.offset() as u32,
            BufferArg::from_raw_parts(ih, il),
            indices.offset() as u32,
            BufferArg::from_raw_parts(xh, xl),
            x.offset() as u32,
            y.len() as u32,
            grid.chunk,
            grid.threads(),
            n,
        );
    }
    device.count_launch();
    Ok(())
}

/// Scatters values: `x[indices[j]] = y[j]` (assignment), e.g. the charges into their
/// slots of the source store (device-path.md §3.1).
///
/// **Precondition:** the indices of one launch are distinct, so that the result does
/// not depend on which unit writes last. Debug builds check it on the host and panic.
///
/// # Errors
///
/// [`KernelError::WrongDevice`] if a buffer belongs to another device.
///
/// # Panics
///
/// If `y.len()` is not `indices.len()`, a non-empty launch has `indices.bound()` above
/// `x.len()`, or (debug builds) an index repeats.
pub fn scatter_values<E: DeviceElement>(
    device: &mut Device,
    y: DeviceSlice<'_, E>,
    indices: IndexSlice<'_>,
    x: DeviceSliceMut<'_, E>,
) -> Result<(), KernelError> {
    check_owners(device, &[y.device(), indices.device(), x.device()])?;
    check_columns("scatter_values", 1, x.len(), indices, y.len());
    #[cfg(debug_assertions)]
    assert_distinct("scatter_values", indices);
    if y.is_empty() {
        return Ok(());
    }
    let grid = device.elementwise_grid(y.len());
    let ((yh, yl), (ih, il), (xh, xl)) = (y.binding(), indices.binding(), x.binding());
    // SAFETY: each handle is a whole buffer with its element count. The kernel reads
    // y and the indices at their offsets plus e < y.len() = indices.len(), and writes
    // x at x.offset() + c with c < indices.bound() ≤ x.len(); `check_columns`
    // asserted these bounds.
    unsafe {
        scatter_values_kernel::launch_unchecked::<E>(
            device.client(),
            CubeCount::Static(grid.cubes, 1, 1),
            CubeDim::new_1d(grid.units),
            BufferArg::from_raw_parts(yh, yl),
            y.offset() as u32,
            BufferArg::from_raw_parts(ih, il),
            indices.offset() as u32,
            BufferArg::from_raw_parts(xh, xl),
            x.offset() as u32,
            y.len() as u32,
            grid.chunk,
            grid.threads(),
        );
    }
    device.count_launch();
    Ok(())
}

fn check_owners(device: &Device, owners: &[u64]) -> Result<(), KernelError> {
    owners.iter().try_for_each(|&o| device.check_owner(o))
}

/// The shape checks of a column gather or scatter between a matrix of `matrix_len`
/// values and a packed matrix of `packed_len` values.
fn check_columns(
    what: &str,
    n: usize,
    matrix_len: usize,
    indices: IndexSlice<'_>,
    packed_len: usize,
) {
    assert!(n >= 1, "{what}: column size n = 0");
    assert_eq!(
        matrix_len % n,
        0,
        "{what}: a matrix of {matrix_len} values is not made of columns of {n}"
    );
    assert_eq!(
        packed_len,
        n * indices.len(),
        "{what}: {} indices of columns of {n} values need {} values, not {packed_len}",
        indices.len(),
        n * indices.len()
    );
    let columns = (matrix_len / n) as u64;
    assert!(
        indices.is_empty() || indices.bound() <= columns,
        "{what}: indices up to {} address a matrix of {columns} columns",
        indices.bound().saturating_sub(1)
    );
}

/// The first value that occurs twice in `indices`, if any.
#[cfg(any(debug_assertions, test))]
pub(crate) fn first_repeat(indices: &[u32]) -> Option<u32> {
    let mut sorted = indices.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).find(|w| w[0] == w[1]).map(|w| w[0])
}

#[cfg(debug_assertions)]
fn assert_distinct(what: &str, indices: IndexSlice<'_>) {
    if let Some(i) = first_repeat(indices.host()) {
        panic!("{what}: index {i} repeats within one launch; the indices must be distinct");
    }
}

#[cfg(test)]
mod tests {
    use super::first_repeat;

    #[test]
    fn first_repeat_finds_repeats_only() {
        assert_eq!(first_repeat(&[]), None);
        assert_eq!(first_repeat(&[5]), None);
        assert_eq!(first_repeat(&[3, 1, 2, 0]), None);
        assert_eq!(first_repeat(&[3, 1, 3, 0]), Some(3));
        assert_eq!(first_repeat(&[7, 7]), Some(7));
    }
}
