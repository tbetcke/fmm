//! The host drivers of the level calls, generic over the kernel (Helmholtz CH2.0;
//! docs/design/helmholtz-fmm-proposal.md §6.2): what a level call does besides the
//! arithmetic of one box or leaf.
//!
//! A driver walks the plan's target-centric view of a level call, hands each target to
//! one thread with one scratch set ([`Execution`]), and calls the kernel's body for each
//! entry of the target's row, in the row's order. Under the batched M2L it runs blocks of
//! [`M2L_BLOCK_TARGETS`] targets through the kernel's column kernel instead ([`m2l`]).
//! The kernel ([`HostKernel`], a type parameter fixed at compile time) supplies everything
//! else: the operators of one box or leaf, its scratch, its tables and the charge vectors.
//! The order in which a target's entries are added is the driver's, so it is the same for
//! every kernel; the order inside one entry's operator is the kernel's. The operator module
//! documentation, "Batched execution and order", "The batched M2L", "Threads" and
//! "Scratch", describes it for Laplace.
//!
//! [`LaplaceOperator`](crate::operator::LaplaceOperator) is the first kernel. It ignores
//! the level, reads its one p at run time (as before CH2.0), and its outputs are bit for
//! bit those of the code before the drivers moved here. The Helmholtz-family operators
//! read their per-level degree and tables from the level.

use core::fmt;
use core::ops::Range;
use std::sync::{Arc, Mutex, PoisonError, TryLockError};

use nd_fmm_plan::index::BoxIndex;
use nd_fmm_plan::lists::{Children, Csr, Parents, VList};
use nd_fmm_plan::operator::{L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p};
use nd_fmm_plan::store::{LeafSlice, LeafSliceMut, LevelSlice, LevelSliceMut};
use nd_fmm_ref::Frame;
use nd_fmm_simd::{MatrixKernel, SimdScalar};
use nd_fmm_tables::MatrixSet;
use nd_fmm_tables::geometry::m2l_offset_index;
use nd_octree::{MortonKey, morton};
use rayon::ThreadPool;
use rayon::prelude::*;

use crate::geometry::relative_frame;
use crate::operator::M2L_BLOCK_TARGETS;

/// The boxes per piece below which the pool runs a piece of the parents serially.
pub(crate) const PARENT_GRAIN: usize = 64;

/// What a kernel gives the drivers: the operators of one box or leaf, its scratch, and
/// the batched M2L's tables and column kernel. Resolved at compile time; the drivers add
/// no dispatch (proposal §2, constraint 1).
///
/// Every operator method accumulates into its output. `level` is the level call's own
/// level (the `level` of its batch, `nd_fmm_plan::operator`): the output's level for P2M,
/// M2M, M2L, P2L and L2L (the children of M2M one below it, the parents of L2L one above),
/// and the leaves' level for L2P, M2P and P2P (the W boxes of M2P one below it). The
/// source leaves of P2L and P2P may lie on other levels; the drivers find them by their
/// keys, and a kernel that needs their level takes it from the frame's radius.
///
/// A box block's layout is the kernel's own: the target-by-target drivers take each block's
/// size from the plan's stores (the operator's `multipole_size(level)` and
/// `local_size(level)`, which may differ) and pass whole blocks to the kernel. So complex
/// expansions in two planes, multipoles and locals of different sizes, and real shifts
/// applied to both planes all live inside the kernel's bodies (Helmholtz CH3.5,
/// docs/helmholtz/tasks/CH3.5.md). Only the batched M2L ([`m2l_block`]) reads a layout:
/// Laplace's, with [`vectors`](Self::vectors) expansions of the table's n values per box,
/// one after the other, multipole and local alike, and square n × n tables. A kernel with
/// any other layout returns `None` from [`batched_m2l`](Self::batched_m2l).
pub(crate) trait HostKernel<T: SimdScalar>: Sync {
    /// One thread's scratch set: everything a target body writes besides its output.
    type Scratch: Send;

    /// The charge vectors m: the columns per box of the batched M2L.
    fn vectors(&self) -> usize;

    /// P2M of the source chunk `sources` of a leaf into its `multipole`.
    fn p2m(&self, level: usize, sources: &[T], multipole: &mut [T], scratch: &mut Self::Scratch);

    /// M2M of the `child` with child index `o` into its `parent`.
    fn m2m(
        &self,
        level: usize,
        o: usize,
        child: &[T],
        parent: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// M2L of a V-list pair with offset index `d` into a target of child index `child`.
    fn m2l(
        &self,
        level: usize,
        d: usize,
        child: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// L2L of the `parent` into its `child` with child index `o`.
    fn l2l(
        &self,
        level: usize,
        o: usize,
        parent: &[T],
        child: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// P2L of the source chunk `sources` of a leaf at `frame` = (ĉ(t|s), r̂(t|s)).
    fn p2l(
        &self,
        level: usize,
        frame: &Frame<T>,
        sources: &[T],
        local: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// L2P of a leaf's `local` at its target points.
    fn l2p(
        &self,
        level: usize,
        local: &[T],
        target_input: &[T],
        target_output: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// M2P of a W box's `multipole` at `frame` = (ĉ(s|t), r̂(s|t)).
    fn m2p(
        &self,
        level: usize,
        frame: &Frame<T>,
        multipole: &[T],
        target_input: &[T],
        target_output: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// P2P from the leaf's own chunk if `frame` is `None`, otherwise from another leaf's
    /// sources at `frame` = (ĉ(s|t), r̂(s|t)).
    fn p2p(
        &self,
        level: usize,
        frame: Option<&Frame<T>>,
        sources: &[T],
        target_input: &[T],
        target_output: &mut [T],
        scratch: &mut Self::Scratch,
    );

    /// The batched M2L's 316 tables of `level` and its column kernel, or `None` to run
    /// M2L target by target.
    fn batched_m2l(&self, level: usize) -> Option<(&MatrixSet<T>, &MatrixKernel<T>)>;

    /// The batched M2L's column lists and packing buffer in `scratch`; never called for a
    /// kernel whose [`batched_m2l`](Self::batched_m2l) is `None` on every level, which may
    /// keep an empty [`Columns`] here.
    fn columns(scratch: &mut Self::Scratch) -> &mut Columns<T>;
}

/// The scratch of one block of the batched M2L: the packing buffer of the column kernel
/// (`MatrixKernel::pack_len` of the table's n; 4 (p + 1)² values for Laplace) and the column offsets of one offset's
/// pairs in the block, at most m [`M2L_BLOCK_TARGETS`] each (a target occurs at most once
/// per offset, with m columns). Sized at construction; a block only clears and refills
/// them.
#[derive(Clone, Debug)]
pub(crate) struct Columns<T> {
    pub(crate) pack: Vec<T>,
    pub(crate) sources: Vec<usize>,
    pub(crate) targets: Vec<usize>,
}

/// Where the level calls run, and the scratch sets they use (operator module
/// documentation, "Threads" and "Scratch").
#[derive(Debug)]
pub(crate) struct Execution<S> {
    /// One set, or one per thread of `pool`, indexed by `rayon::current_thread_index()`.
    /// Set 0 also serves the serial path and the per-pair methods.
    pub(crate) scratch: Vec<Mutex<S>>,
    /// The pool of the level calls, if any.
    pub(crate) pool: Option<Arc<ThreadPool>>,
    /// Runs the level calls serially even with a pool.
    pub(crate) serial: bool,
}

/// A copy shares the pool and gets its own scratch sets.
impl<S: Clone> Clone for Execution<S> {
    fn clone(&self) -> Self {
        Self {
            scratch: self
                .scratch
                .iter()
                .map(|set| Mutex::new(set.lock().unwrap_or_else(PoisonError::into_inner).clone()))
                .collect(),
            pool: self.pool.clone(),
            serial: self.serial,
        }
    }
}

impl<S: Send> Execution<S> {
    /// Scratch set 0, without locking.
    pub(crate) fn first(&mut self) -> &mut S {
        self.scratch[0]
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Runs `body(t, chunk, scratch)` for every chunk t of `output`: in index order on
    /// the calling thread, or in the pool with `par_chunks_mut`.
    pub(crate) fn boxes<T: Send, F>(&mut self, mut output: LevelSliceMut<'_, T>, body: F)
    where
        F: Fn(usize, &mut [T], &mut S) + Sync,
    {
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let (sets, size) = (&self.scratch, output.size());
                pool.install(|| {
                    output
                        .as_mut_slice()
                        .par_chunks_mut(size)
                        .enumerate()
                        .for_each(|(t, chunk)| with_scratch(sets, |s| body(t, chunk, s)));
                });
            }
            _ => {
                let scratch = self.first();
                for (t, chunk) in output.chunks_mut().enumerate() {
                    body(t, chunk, scratch);
                }
            }
        }
    }

    /// Runs `body(first, block, scratch)` for every block of `targets` consecutive chunks
    /// of `output` (the last may be shorter), `first` the index of its first chunk: in
    /// order on the calling thread, or in the pool with `par_chunks_mut`.
    pub(crate) fn blocks<T: Send, F>(
        &mut self,
        mut output: LevelSliceMut<'_, T>,
        targets: usize,
        body: F,
    ) where
        F: Fn(usize, &mut [T], &mut S) + Sync,
    {
        let size = output.size() * targets;
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let sets = &self.scratch;
                pool.install(|| {
                    output
                        .as_mut_slice()
                        .par_chunks_mut(size)
                        .enumerate()
                        .for_each(|(i, block)| with_scratch(sets, |s| body(i * targets, block, s)));
                });
            }
            _ => {
                let scratch = self.first();
                for (i, block) in output.as_mut_slice().chunks_mut(size).enumerate() {
                    body(i * targets, block, scratch);
                }
            }
        }
    }

    /// Runs `body(i, chunk, scratch)` for every chunk i of `size` values of `buffer`: in
    /// index order on the calling thread, or in the pool with `par_chunks_mut`.
    pub(crate) fn chunks<T: Send, F>(&mut self, buffer: &mut [T], size: usize, body: F)
    where
        F: Fn(usize, &mut [T], &mut S) + Sync,
    {
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let sets = &self.scratch;
                pool.install(|| {
                    buffer
                        .par_chunks_mut(size)
                        .enumerate()
                        .for_each(|(i, chunk)| with_scratch(sets, |s| body(i, chunk, s)));
                });
            }
            _ => {
                let scratch = self.first();
                for (i, chunk) in buffer.chunks_mut(size).enumerate() {
                    body(i, chunk, scratch);
                }
            }
        }
    }

    /// Runs `body(boxes, chunks, scratch)` over pieces of whole parents of the level
    /// `slice` (`size` values per box), the boxes `boxes` and their chunks, `starts(t)`
    /// telling whether box t begins a parent:
    /// the whole level on the calling thread, or in the pool, halved at parent boundaries
    /// by [`split_parents`] down to [`PARENT_GRAIN`] boxes.
    pub(crate) fn parents<T: Send, F, B>(
        &mut self,
        slice: &mut [T],
        size: usize,
        starts: B,
        body: F,
    ) where
        F: Fn(Range<usize>, &mut [T], &mut S) + Sync,
        B: Fn(usize) -> bool + Sync,
    {
        let boxes = slice.len() / size;
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let sets = &self.scratch;
                pool.install(|| split_parents(slice, 0..boxes, size, &starts, sets, &body));
            }
            _ => body(0..boxes, slice, self.first()),
        }
    }

    /// Runs `body(r, chunk, scratch)` for every leaf r of `output`: in index order on the
    /// calling thread, or in the pool, split by [`split_leaves`].
    pub(crate) fn leaves<T: Send, F>(&mut self, mut output: LeafSliceMut<'_, T>, body: F)
    where
        F: Fn(usize, &mut [T], &mut S) + Sync,
    {
        match (&self.pool, self.serial) {
            (Some(pool), false) => {
                let sets = &self.scratch;
                pool.install(|| split_leaves(output, 0, sets, &body));
            }
            _ => {
                let scratch = self.first();
                for (r, chunk) in output.chunks_mut().enumerate() {
                    body(r, chunk, scratch);
                }
            }
        }
    }
}

/// Runs `body(first + r, chunk r, scratch)` for every leaf r of `output`, halving the
/// leaves recursively under `rayon::join`: one disjoint mutable slice per leaf, by the
/// CSR offsets, without allocation.
fn split_leaves<T: Send, S: Send, F>(
    mut output: LeafSliceMut<'_, T>,
    first: usize,
    sets: &[Mutex<S>],
    body: &F,
) where
    F: Fn(usize, &mut [T], &mut S) + Sync,
{
    match output.nleaves() {
        0 => {}
        1 => with_scratch(sets, |s| body(first, output.chunk_mut(0), s)),
        n => {
            let mid = n / 2;
            let (left, right) = output.split_at_mut(mid);
            rayon::join(
                || split_leaves(left, first, sets, body),
                || split_leaves(right, first + mid, sets, body),
            );
        }
    }
}

/// Runs `body(boxes, chunks, scratch)` on `boxes` (their chunks `chunks`, `size` values
/// each), halving it at a box where `starts` holds under `rayon::join` while it holds more
/// than [`PARENT_GRAIN`] boxes; a piece that no such box divides runs whole.
fn split_parents<T: Send, S: Send, F, B>(
    chunks: &mut [T],
    boxes: Range<usize>,
    size: usize,
    starts: &B,
    sets: &[Mutex<S>],
    body: &F,
) where
    F: Fn(Range<usize>, &mut [T], &mut S) + Sync,
    B: Fn(usize) -> bool + Sync,
{
    let mut mid = boxes.start + boxes.len() / 2;
    while mid < boxes.end && !starts(mid) {
        mid += 1;
    }
    if boxes.len() <= PARENT_GRAIN || mid == boxes.end {
        with_scratch(sets, |s| body(boxes, chunks, s));
        return;
    }
    let (left, right) = chunks.split_at_mut((mid - boxes.start) * size);
    rayon::join(
        || split_parents(left, boxes.start..mid, size, starts, sets, body),
        || split_parents(right, mid..boxes.end, size, starts, sets, body),
    );
}

/// Runs `f` with the scratch set of the current pool thread.
///
/// # Panics
///
/// Outside a pool thread, or if the set is locked: only this thread locks it, and only
/// around a target body that never yields to rayon, so that would be a defect.
#[inline]
fn with_scratch<S>(sets: &[Mutex<S>], f: impl FnOnce(&mut S)) {
    let i = rayon::current_thread_index().expect("a level call runs on the pool's threads");
    let mut set = match sets[i].try_lock() {
        Ok(set) => set,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => panic!("the scratch set of pool thread {i} is in use"),
    };
    f(&mut set);
}

/// The shared inputs of one box call (P2M, M2M, M2L, P2L, L2L): its level, the index,
/// its view and its input.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BoxCall<'a, V, I> {
    pub(crate) level: usize,
    pub(crate) index: &'a BoxIndex,
    pub(crate) view: &'a V,
    pub(crate) input: I,
}

/// The shared inputs of one leaf call (L2P, M2P, P2P): its level, the index, the leaf
/// index of row 0, its view, its input and the target input of its rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LeafCall<'a, T, I> {
    pub(crate) level: usize,
    pub(crate) index: &'a BoxIndex,
    pub(crate) first_leaf: usize,
    pub(crate) view: &'a Csr,
    pub(crate) input: I,
    pub(crate) target_input: LeafSlice<'a, T>,
}

/// P2M, target by target ([`p2m_target`]).
pub(crate) fn p2m<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: P2m<'_, T>,
) {
    let P2m {
        level,
        index,
        leaves,
        sources,
        multipoles,
    } = batch;
    let call = BoxCall {
        level,
        index,
        view: leaves,
        input: sources,
    };
    execution.boxes(multipoles, |t, multipole, scratch| {
        p2m_target(kernel, &call, t, multipole, scratch);
    });
}

/// M2M, target by target ([`m2m_target`]).
pub(crate) fn m2m<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: M2m<'_, T>,
) {
    let M2m {
        level,
        index,
        children,
        child_multipoles,
        multipoles,
        ..
    } = batch;
    let call = BoxCall {
        level,
        index,
        view: children,
        input: child_multipoles,
    };
    execution.boxes(multipoles, |t, multipole, scratch| {
        m2m_target(kernel, &call, t, multipole, scratch);
    });
}

/// M2L: by blocks of [`M2L_BLOCK_TARGETS`] targets ([`m2l_block`]) where the kernel has
/// a batched M2L on `level`, target by target ([`m2l_target`]) otherwise.
pub(crate) fn m2l<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: M2l<'_, T>,
) {
    let M2l {
        level,
        index,
        pairs,
        multipoles,
        locals,
    } = batch;
    let call = BoxCall {
        level,
        index,
        view: pairs,
        input: multipoles,
    };
    match kernel.batched_m2l(level) {
        Some(batched) => {
            if pairs.is_empty() {
                return;
            }
            execution.blocks(locals, M2L_BLOCK_TARGETS, |first, block, scratch| {
                m2l_block(kernel, &call, batched, first, block, scratch);
            });
        }
        None => execution.boxes(locals, |t, local, scratch| {
            m2l_target(kernel, &call, t, local, scratch);
        }),
    }
}

/// P2L, target by target ([`p2l_target`]).
pub(crate) fn p2l<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: P2l<'_, T>,
) {
    let P2l {
        level,
        index,
        x,
        sources,
        locals,
    } = batch;
    let call = BoxCall {
        level,
        index,
        view: x,
        input: sources,
    };
    execution.boxes(locals, |t, local, scratch| {
        p2l_target(kernel, &call, t, local, scratch);
    });
}

/// L2L, target by target ([`l2l_target`]).
pub(crate) fn l2l<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: L2l<'_, T>,
) {
    let L2l {
        level,
        index,
        parents,
        parent_locals,
        locals,
    } = batch;
    let call = BoxCall {
        level,
        index,
        view: parents,
        input: parent_locals,
    };
    execution.boxes(locals, |t, local, scratch| {
        l2l_target(kernel, &call, t, local, scratch);
    });
}

/// L2P, leaf by leaf ([`l2p_target`]).
pub(crate) fn l2p<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: L2p<'_, T>,
) {
    let L2p {
        level,
        index,
        leaves,
        boxes,
        locals,
        target_input,
        target_output,
    } = batch;
    let call = LeafCall {
        level,
        index,
        first_leaf: leaves.start,
        view: boxes,
        input: locals,
        target_input,
    };
    execution.leaves(target_output, |r, output, scratch| {
        l2p_target(kernel, &call, r, output, scratch);
    });
}

/// M2P, leaf by leaf ([`m2p_target`]).
pub(crate) fn m2p<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: M2p<'_, T>,
) {
    let M2p {
        level,
        index,
        leaves,
        w,
        multipoles,
        target_input,
        target_output,
    } = batch;
    let call = LeafCall {
        level,
        index,
        first_leaf: leaves.start,
        view: w,
        input: multipoles,
        target_input,
    };
    execution.leaves(target_output, |r, output, scratch| {
        m2p_target(kernel, &call, r, output, scratch);
    });
}

/// P2P, leaf by leaf ([`p2p_target`]).
pub(crate) fn p2p<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    execution: &mut Execution<K::Scratch>,
    batch: P2p<'_, T>,
) {
    let P2p {
        level,
        index,
        leaves,
        near,
        sources,
        target_input,
        target_output,
    } = batch;
    let call = LeafCall {
        level,
        index,
        first_leaf: leaves.start,
        view: near,
        input: sources,
        target_input,
    };
    execution.leaves(target_output, |r, output, scratch| {
        p2p_target(kernel, &call, r, output, scratch);
    });
}

/// The P2M body of box t: the source chunk of its leaf, if it has one.
#[inline]
pub(crate) fn p2m_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &BoxCall<'_, Csr, LeafSlice<'_, T>>,
    t: usize,
    multipole: &mut [T],
    scratch: &mut K::Scratch,
) {
    for &j in call.view.row(t) {
        debug_assert_eq!(
            call.index.leaf_key(j as usize),
            call.index.key(call.level, t)
        );
        kernel.p2m(call.level, call.input.chunk(j as usize), multipole, scratch);
    }
}

/// The M2M body of parent t: its children, by octant.
#[inline]
pub(crate) fn m2m_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &BoxCall<'_, Children, LevelSlice<'_, T>>,
    t: usize,
    multipole: &mut [T],
    scratch: &mut K::Scratch,
) {
    let (children, octants) = call.view.row(t);
    for (&c, &o) in children.iter().zip(octants) {
        debug_check_octant(
            call.index.key(call.level + 1, c as usize),
            call.index.key(call.level, t),
            o as usize,
        );
        kernel.m2m(
            call.level,
            o as usize,
            call.input.chunk(c as usize),
            multipole,
            scratch,
        );
    }
}

/// The M2L body of target t: its V-list sources, by offset index.
#[inline]
pub(crate) fn m2l_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
    t: usize,
    local: &mut [T],
    scratch: &mut K::Scratch,
) {
    let (sources, offsets) = call.view.row(t);
    let child = match sources.is_empty() {
        true => 0,
        false => morton::child_index(call.index.key(call.level, t)),
    };
    for (&s, &d) in sources.iter().zip(offsets) {
        debug_assert_eq!(
            offset_index(
                call.index.key(call.level, s as usize),
                call.index.key(call.level, t)
            ),
            d as usize,
            "the offset index of a V-list entry disagrees with its keys"
        );
        kernel.m2l(
            call.level,
            d as usize,
            child,
            call.input.chunk(s as usize),
            local,
            scratch,
        );
    }
}

/// The batched M2L of the block of targets `first ..` whose locals are `locals`
/// (operator module documentation, "The batched M2L"): for every offset index d in order,
/// the pairs of `batch(d)` whose target lies in the block, found by two binary searches,
/// in one call of the column kernel with the table of d, the sources' multipoles and
/// the targets' locals read in place.
#[inline]
pub(crate) fn m2l_block<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &BoxCall<'_, VList, LevelSlice<'_, T>>,
    (tables, matrix): (&MatrixSet<T>, &MatrixKernel<T>),
    first: usize,
    locals: &mut [T],
    scratch: &mut K::Scratch,
) {
    let (n, m) = (tables.n(), kernel.vectors());
    let end = first + locals.len() / (m * n);
    let multipoles = call.input.as_slice();
    let columns = K::columns(scratch);
    for d in 0..call.view.ngroups() {
        let (targets, sources) = call.view.batch(d);
        let lo = targets.partition_point(|&t| (t as usize) < first);
        let hi = lo + targets[lo..].partition_point(|&t| (t as usize) < end);
        if lo == hi {
            continue;
        }
        columns.sources.clear();
        columns.targets.clear();
        for (&t, &s) in targets[lo..hi].iter().zip(&sources[lo..hi]) {
            debug_assert_eq!(
                offset_index(
                    call.index.key(call.level, s as usize),
                    call.index.key(call.level, t as usize)
                ),
                d,
                "the offset index of a V-list batch disagrees with its keys"
            );
            // The m columns of the pair, vector after vector.
            for v in 0..m {
                columns.sources.push((s as usize * m + v) * n);
                columns.targets.push(((t as usize - first) * m + v) * n);
            }
        }
        matrix.accumulate_columns(
            n,
            tables.matrix(d),
            multipoles,
            &columns.sources,
            locals,
            &columns.targets,
            &mut columns.pack,
        );
    }
}

/// The P2L body of box t: its X-list leaves, by leaf index.
#[inline]
pub(crate) fn p2l_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &BoxCall<'_, Csr, LeafSlice<'_, T>>,
    t: usize,
    local: &mut [T],
    scratch: &mut K::Scratch,
) {
    let target = call.index.key(call.level, t);
    for &j in call.view.row(t) {
        let frame = relative_frame(target, call.index.leaf_key(j as usize));
        kernel.p2l(
            call.level,
            &frame,
            call.input.chunk(j as usize),
            local,
            scratch,
        );
    }
}

/// The L2L body of box t: its parent, if it has a row.
#[inline]
pub(crate) fn l2l_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &BoxCall<'_, Parents, LevelSlice<'_, T>>,
    t: usize,
    local: &mut [T],
    scratch: &mut K::Scratch,
) {
    let (parents, octants) = call.view.row(t);
    for (&parent, &o) in parents.iter().zip(octants) {
        debug_check_octant(
            call.index.key(call.level, t),
            call.index.key(call.level - 1, parent as usize),
            o as usize,
        );
        kernel.l2l(
            call.level,
            o as usize,
            call.input.chunk(parent as usize),
            local,
            scratch,
        );
    }
}

/// The L2P body of row r: the local of the leaf's box.
#[inline]
pub(crate) fn l2p_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &LeafCall<'_, T, LevelSlice<'_, T>>,
    r: usize,
    output: &mut [T],
    scratch: &mut K::Scratch,
) {
    for &i in call.view.row(r) {
        let local = call.input.chunk(i as usize);
        kernel.l2p(
            call.level,
            local,
            call.target_input.chunk(r),
            output,
            scratch,
        );
    }
}

/// The M2P body of row r: its W-list boxes, by box index.
#[inline]
pub(crate) fn m2p_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &LeafCall<'_, T, LevelSlice<'_, T>>,
    r: usize,
    output: &mut [T],
    scratch: &mut K::Scratch,
) {
    let target = call.index.leaf_key(call.first_leaf + r);
    for &s in call.view.row(r) {
        let frame = relative_frame(call.index.key(call.level + 1, s as usize), target);
        let multipole = call.input.chunk(s as usize);
        kernel.m2p(
            call.level,
            &frame,
            multipole,
            call.target_input.chunk(r),
            output,
            scratch,
        );
    }
}

/// The P2P body of row r: its near list, the leaf itself included, by leaf index.
#[inline]
pub(crate) fn p2p_target<T: SimdScalar, K: HostKernel<T>>(
    kernel: &K,
    call: &LeafCall<'_, T, LeafSlice<'_, T>>,
    r: usize,
    output: &mut [T],
    scratch: &mut K::Scratch,
) {
    let leaf = call.first_leaf + r;
    let target = call.index.leaf_key(leaf);
    for &j in call.view.row(r) {
        let j = j as usize;
        let frame = (j != leaf).then(|| relative_frame(call.index.leaf_key(j), target));
        kernel.p2p(
            call.level,
            frame.as_ref(),
            call.input.chunk(j),
            call.target_input.chunk(r),
            output,
            scratch,
        );
    }
}

/// A key with its level and index, for panic messages.
struct Key(MortonKey);

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (level, index) = morton::decode(self.0);
        write!(f, "{} (level {level}, index {index:?})", self.0)
    }
}

/// The position of index(target) − index(source) in `V_LIST_DIRECTIONS`.
///
/// # Panics
///
/// With both keys if the levels differ or the offset is not a V-list offset.
pub(crate) fn offset_index(source: MortonKey, target: MortonKey) -> usize {
    let (source_level, source_index) = morton::decode(source);
    let (target_level, target_index) = morton::decode(target);
    assert_eq!(
        source_level,
        target_level,
        "M2L between boxes on different levels: source {}, target {}",
        Key(source),
        Key(target)
    );
    let d = [0, 1, 2].map(|k| target_index[k] as i64 - source_index[k] as i64);
    m2l_offset_index(d).unwrap_or_else(|| {
        panic!(
            "M2L from source {} to target {}: the offset {d:?} is not a V-list offset",
            Key(source),
            Key(target)
        )
    })
}

/// Debug builds: checks that `parent` is the parent of `child` and `o` its child index.
#[inline]
pub(crate) fn debug_check_octant(child: MortonKey, parent: MortonKey, o: usize) {
    debug_assert!(
        morton::parent(child) == Some(parent),
        "{} is not the parent of {}",
        Key(parent),
        Key(child)
    );
    debug_assert_eq!(morton::child_index(child), o, "octant of {}", Key(child));
}
