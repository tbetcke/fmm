//! Device buffers and views of element ranges of them (device-path.md §3.1).

use std::marker::PhantomData;
use std::ops::{Bound, Range, RangeBounds};

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::device::Precision;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for u32 {}
}

/// An element type a [`DeviceBuffer`] holds: `f32`, `f64` or `u32` (sealed).
///
/// Index arrays are `u32` only on the device. The `u16` offset indices and `u8` octants
/// of `GroupedCsr` are widened at upload
/// ([`Device::upload_indices_widened`](crate::Device::upload_indices_widened)), so there
/// are no `u16` or `u8` buffers.
pub trait DeviceElement:
    CubeElement + Numeric + Copy + PartialEq + Send + Sync + 'static + sealed::Sealed
{
    /// The float precision the type needs on the device; `None` for `u32`.
    const PRECISION: Option<Precision>;
}

impl DeviceElement for f32 {
    const PRECISION: Option<Precision> = Some(Precision::F32);
}

impl DeviceElement for f64 {
    const PRECISION: Option<Precision> = Some(Precision::F64);
}

impl DeviceElement for u32 {
    const PRECISION: Option<Precision> = None;
}

/// A float element type of the kernels: `f32` or `f64` (sealed).
pub trait DeviceFloat: DeviceElement + Float {
    /// The precision of the type.
    const FLOAT: Precision;
}

impl DeviceFloat for f32 {
    const FLOAT: Precision = Precision::F32;
}

impl DeviceFloat for f64 {
    const FLOAT: Precision = Precision::F64;
}

/// `len` elements of `E` on one [`Device`](crate::Device); freed on drop.
///
/// Created by [`Device::alloc`](crate::Device::alloc) and
/// [`Device::upload`](crate::Device::upload). Kernels take a buffer whole, with the
/// element offset of their range as a scalar argument, because wgpu binding offsets
/// must be aligned and level offsets are not (device-path.md §4.1).
pub struct DeviceBuffer<E: DeviceElement> {
    handle: Handle,
    len: usize,
    device: u64,
    element: PhantomData<E>,
}

impl<E: DeviceElement> std::fmt::Debug for DeviceBuffer<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("element", &E::type_name())
            .field("len", &self.len)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl<E: DeviceElement> DeviceBuffer<E> {
    pub(crate) fn new(handle: Handle, len: usize, device: u64) -> Self {
        Self {
            handle,
            len,
            device,
            element: PhantomData,
        }
    }

    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer holds no element.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The whole buffer, read-only.
    pub fn as_slice(&self) -> DeviceSlice<'_, E> {
        self.slice(..)
    }

    /// The whole buffer, writable.
    pub fn as_slice_mut(&mut self) -> DeviceSliceMut<'_, E> {
        self.slice_mut(..)
    }

    /// The elements in `range`, read-only.
    ///
    /// # Panics
    ///
    /// If `range` is not within `0..len`.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> DeviceSlice<'_, E> {
        let range = to_range(range, self.len);
        DeviceSlice {
            buffer: self,
            start: range.start,
            len: range.len(),
        }
    }

    /// The elements in `range`, writable.
    ///
    /// # Panics
    ///
    /// If `range` is not within `0..len`.
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> DeviceSliceMut<'_, E> {
        let range = to_range(range, self.len);
        DeviceSliceMut {
            buffer: self,
            start: range.start,
            len: range.len(),
        }
    }
}

/// A range of elements of a [`DeviceBuffer`], read-only.
#[derive(Debug)]
pub struct DeviceSlice<'a, E: DeviceElement> {
    buffer: &'a DeviceBuffer<E>,
    start: usize,
    len: usize,
}

impl<E: DeviceElement> Clone for DeviceSlice<'_, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<E: DeviceElement> Copy for DeviceSlice<'_, E> {}

impl<E: DeviceElement> DeviceSlice<'_, E> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The offset of the first element in its buffer.
    pub fn offset(&self) -> usize {
        self.start
    }

    pub(crate) fn device(&self) -> u64 {
        self.buffer.device
    }

    /// The handle of the whole buffer and its length in elements, for a launch.
    pub(crate) fn binding(&self) -> (Handle, usize) {
        (self.buffer.handle.clone(), self.buffer.len.max(1))
    }

    /// A handle restricted to the slice's bytes, for transfers.
    pub(crate) fn byte_range_handle(&self) -> Handle {
        byte_range(&self.buffer.handle, self.start, self.len, size_of::<E>())
    }
}

/// A range of elements of a [`DeviceBuffer`], writable.
#[derive(Debug)]
pub struct DeviceSliceMut<'a, E: DeviceElement> {
    buffer: &'a mut DeviceBuffer<E>,
    start: usize,
    len: usize,
}

impl<E: DeviceElement> DeviceSliceMut<'_, E> {
    /// The number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The offset of the first element in its buffer.
    pub fn offset(&self) -> usize {
        self.start
    }

    /// A shorter-lived writable view of the same range.
    pub fn reborrow(&mut self) -> DeviceSliceMut<'_, E> {
        DeviceSliceMut {
            buffer: self.buffer,
            start: self.start,
            len: self.len,
        }
    }

    pub(crate) fn device(&self) -> u64 {
        self.buffer.device
    }

    /// The handle of the whole buffer and its length in elements, for a launch.
    pub(crate) fn binding(&self) -> (Handle, usize) {
        (self.buffer.handle.clone(), self.buffer.len.max(1))
    }

    /// A handle restricted to the slice's bytes, for transfers.
    pub(crate) fn byte_range_handle(&self) -> Handle {
        byte_range(&self.buffer.handle, self.start, self.len, size_of::<E>())
    }
}

/// An index array on the device, with the host-side facts the launch wrappers check
/// before launching without bounds checks (device-path.md §6.1, "Launch hygiene").
///
/// Created by [`Device::upload_indices`](crate::Device::upload_indices). Its
/// [`bound`](Self::bound) is one more than the largest entry of the whole array, so a
/// launch on any range of it checks the same bound: keep index arrays that address
/// different matrices in different buffers. Debug builds keep a host copy, against
/// which the scatters check that the indices of one launch are distinct.
pub struct IndexBuffer {
    buffer: DeviceBuffer<u32>,
    bound: u64,
    #[cfg(debug_assertions)]
    host: Vec<u32>,
}

impl std::fmt::Debug for IndexBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IndexBuffer")
            .field("len", &self.buffer.len)
            .field("bound", &self.bound)
            .finish_non_exhaustive()
    }
}

impl IndexBuffer {
    pub(crate) fn new(buffer: DeviceBuffer<u32>, indices: &[u32]) -> Self {
        Self {
            buffer,
            bound: indices.iter().max().map_or(0, |&m| u64::from(m) + 1),
            #[cfg(debug_assertions)]
            host: indices.to_vec(),
        }
    }

    /// The number of indices.
    pub fn len(&self) -> usize {
        self.buffer.len
    }

    /// True if there is no index.
    pub fn is_empty(&self) -> bool {
        self.buffer.len == 0
    }

    /// One more than the largest index, or 0 for an empty array.
    pub fn bound(&self) -> u64 {
        self.bound
    }

    /// The device buffer, e.g. to download it.
    pub fn buffer(&self) -> &DeviceBuffer<u32> {
        &self.buffer
    }

    /// All indices.
    pub fn as_slice(&self) -> IndexSlice<'_> {
        self.slice(..)
    }

    /// The indices in `range`.
    ///
    /// # Panics
    ///
    /// If `range` is not within `0..len`.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> IndexSlice<'_> {
        let range = to_range(range, self.buffer.len);
        IndexSlice {
            indices: self,
            start: range.start,
            len: range.len(),
        }
    }
}

/// A range of an [`IndexBuffer`]: the indices of one launch.
#[derive(Clone, Copy, Debug)]
pub struct IndexSlice<'a> {
    indices: &'a IndexBuffer,
    start: usize,
    len: usize,
}

impl<'a> IndexSlice<'a> {
    /// The indices in `range` of this range (T8: a chunk's part of a batch array).
    ///
    /// # Panics
    ///
    /// If `range` is not within `0..len`.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> IndexSlice<'a> {
        let range = to_range(range, self.len);
        IndexSlice {
            indices: self.indices,
            start: self.start + range.start,
            len: range.len(),
        }
    }
}

impl IndexSlice<'_> {
    /// The number of indices.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the range is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The offset of the first index in its buffer.
    pub fn offset(&self) -> usize {
        self.start
    }

    /// The bound of the whole [`IndexBuffer`].
    pub fn bound(&self) -> u64 {
        self.indices.bound
    }

    pub(crate) fn device(&self) -> u64 {
        self.indices.buffer.device
    }

    /// The handle of the whole buffer and its length in elements, for a launch.
    pub(crate) fn binding(&self) -> (Handle, usize) {
        (
            self.indices.buffer.handle.clone(),
            self.indices.buffer.len.max(1),
        )
    }

    /// The host copy of the range (debug builds only).
    #[cfg(debug_assertions)]
    pub(crate) fn host(&self) -> &[u32] {
        &self.indices.host[self.start..self.start + self.len]
    }
}

/// `range` resolved against a length.
fn to_range(range: impl RangeBounds<usize>, len: usize) -> Range<usize> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s + 1,
        Bound::Unbounded => 0,
    };
    let end = match range.end_bound() {
        Bound::Included(&e) => e + 1,
        Bound::Excluded(&e) => e,
        Bound::Unbounded => len,
    };
    assert!(
        start <= end && end <= len,
        "range {start}..{end} out of bounds for a buffer of {len} elements"
    );
    start..end
}

/// `handle` restricted to its elements `start..start + len` of `size` bytes each.
fn byte_range(handle: &Handle, start: usize, len: usize, size: usize) -> Handle {
    let begin = (start * size) as u64;
    let end = ((start + len) * size) as u64;
    // Pool allocations carry offsets already; both offsets add to them.
    let trailing = handle.size_in_used() - end;
    handle.clone().offset_start(begin).offset_end(trailing)
}
