//! Data movement kernels: zeroing, gathering and scattering columns, and scattering
//! values by index (fmm-plan-redesign §10, the GEMM check of §6.4; device-path.md §3.1).
//!
//! Each function is a safe launch wrapper over one `#[cube]` kernel, generic over the
//! element type. It checks its arguments on the host (lengths, the owning device, the
//! index bound of [`IndexSlice::bound`], distinct indices in debug builds), so the
//! kernel launches without bounds checks, and it allocates nothing. A call with nothing
//! to do launches nothing. Every operation is a copy, except the single add per
//! element of [`scatter_add_columns`]; the results equal plain host loops bit for bit
//! on every backend (spikes/device-arith/REPORT.md, "Recommendation", rule 6), where the
//! scatter-add's operands and sum are normal (Metal flushes subnormal sums,
//! CONVENTIONS §3.13, "Device kernels").
//!
//! Matrices are column-major with columns of n values: column c of a slice `x` is
//! `x[c n..(c + 1) n]` (the `LevelBuffers` layout, CONVENTIONS §3.6).

use cubecl::prelude::*;

use crate::buffer::{DeviceElement, DeviceFloat, DeviceSlice, DeviceSliceMut, IndexSlice};
use crate::device::Device;
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
