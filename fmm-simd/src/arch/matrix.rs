//! The column-accumulation body of [`MatrixKernel`](crate::MatrixKernel), written once
//! over the vector layer [`Simd`] (docs/design/optimisation.md §2.2, §2.3, §2.11).
//!
//! For each column j it adds A · B[:, b_j] into C[:, c_j], A an n × n column-major
//! matrix. The columns run in groups of [`COLUMNS`]; a group's B columns are packed
//! k-major into the caller's buffer, and the rows of its C columns run in blocks of R
//! vectors, then of one vector while whole vectors remain, then one row at a time on the
//! scalar layer. A block loads its R × [`COLUMNS`] values of C into registers, visits k =
//! 0 … n − 1 in order, and stores them back:
//!
//! ```text
//! for k = 0 … n − 1:
//!     a = A[r0 .. r0 + R W, k]                 R loads
//!     for each column j of the group:
//!         b = splat(B[k, b_j])
//!         separate: c_j = c_j + a · b          (multiply, then add: two roundings)
//!         fused:    c_j = fma(a, b, c_j)       (one rounding)
//! ```
//!
//! Every element of C thus receives the chain c ← c + B[k, b_j] · A[r, k] (or its fused
//! form) for k ascending, starting from its value in C, computed from its own row of A
//! and column of B only: the result does not depend on the ISA, R, the group, the row
//! block or which other columns share the call, bit for bit. The separate form is the
//! chain of `nd_fmm_tables::MatrixSet::apply`. Rust never contracts a multiplication and
//! an addition into an fma on its own, and IEEE 754 multiplication, addition and fma are
//! correctly rounded on every ISA, so the vector paths equal the scalar layer exactly.

use super::Simd;
use super::scalar::Scalar;
use crate::SimdScalar;

/// The columns of a group: B columns packed together, C columns held in registers.
pub(crate) const COLUMNS: usize = 4;

/// Row vectors per block on the scalar layer (rows, since a vector is one value there).
pub(crate) const SCALAR_ROWS: usize = 4;

/// C[:, c_j] += A · B[:, b_j] for every j, on the layer `s` with blocks of `R` row
/// vectors, the separate or (`FUSED`) the fused form (module documentation).
///
/// The caller has checked the shapes: `a` holds n² values; every `b_offsets[j] + n` is at
/// most `b.len()`; `c_offsets` has the length of `b_offsets`, and `c_offsets[j] + n` is at
/// most `c_offsets[j + 1]` and the last at most `c.len()`; `pack` holds at least
/// [`COLUMNS`] n values.
///
/// # Panics
///
/// If the shapes are not as above (an out-of-bounds slice, or overlapping C columns).
#[expect(
    clippy::too_many_arguments,
    reason = "the shape of the public `MatrixKernel::accumulate_columns`, plus the layer"
)]
#[inline(always)]
pub(crate) fn accumulate_body<T: SimdScalar, S: Simd<T>, const R: usize, const FUSED: bool>(
    s: S,
    n: usize,
    a: &[T],
    b: &[T],
    b_offsets: &[usize],
    c: &mut [T],
    c_offsets: &[usize],
    pack: &mut [T],
) {
    let pack = &mut pack[..COLUMNS * n];
    let (b_groups, b_rest) = b_offsets.as_chunks::<COLUMNS>();
    let (c_groups, c_rest) = c_offsets.as_chunks::<COLUMNS>();
    for (bs, cs) in b_groups.iter().zip(c_groups) {
        group::<T, S, R, COLUMNS, FUSED>(s, n, a, b, bs, c, cs, pack);
    }
    for (&bo, &co) in b_rest.iter().zip(c_rest) {
        group::<T, S, R, 1, FUSED>(s, n, a, b, &[bo], c, &[co], &mut pack[..n]);
    }
}

/// One group of `NB` columns: packs its B columns k-major into `pack`, then runs every
/// row block of its C columns.
#[expect(
    clippy::too_many_arguments,
    reason = "the body's arguments, split by group"
)]
#[inline(always)]
fn group<T: SimdScalar, S: Simd<T>, const R: usize, const NB: usize, const FUSED: bool>(
    s: S,
    n: usize,
    a: &[T],
    b: &[T],
    b_offsets: &[usize; NB],
    c: &mut [T],
    c_offsets: &[usize; NB],
    pack: &mut [T],
) {
    let columns: [&[T]; NB] = core::array::from_fn(|j| &b[b_offsets[j]..b_offsets[j] + n]);
    for (k, packed) in pack.as_chunks_mut::<NB>().0.iter_mut().enumerate() {
        for (value, column) in packed.iter_mut().zip(&columns) {
            *value = column[k];
        }
    }
    let mut c: [&mut [T]; NB] = c
        .get_disjoint_mut(core::array::from_fn(|j| c_offsets[j]..c_offsets[j] + n))
        .expect("the columns of C are disjoint and in bounds");
    let (w, rows) = (S::W, R * S::W);
    let mut r0 = 0;
    while r0 + rows <= n {
        block::<T, S, R, NB, FUSED>(s, n, r0, a, pack, &mut c);
        r0 += rows;
    }
    while r0 + w <= n {
        block::<T, S, 1, NB, FUSED>(s, n, r0, a, pack, &mut c);
        r0 += w;
    }
    while r0 < n {
        block::<T, Scalar, 1, NB, FUSED>(Scalar, n, r0, a, pack, &mut c);
        r0 += 1;
    }
}

/// Rows `r0 .. r0 + R W` of the group's C columns: loaded once, every k in order, stored
/// once.
#[inline(always)]
fn block<T: SimdScalar, S: Simd<T>, const R: usize, const NB: usize, const FUSED: bool>(
    s: S,
    n: usize,
    r0: usize,
    a: &[T],
    pack: &[T],
    c: &mut [&mut [T]; NB],
) {
    let w = S::W;
    let mut acc: [[S::V; NB]; R] =
        core::array::from_fn(|i| core::array::from_fn(|j| s.load(&c[j][r0 + i * w..])));
    for (column, packed) in a.chunks_exact(n).zip(pack.as_chunks::<NB>().0) {
        let rows = &column[r0..r0 + R * w];
        let av: [S::V; R] = core::array::from_fn(|i| s.load(&rows[i * w..]));
        for (j, &value) in packed.iter().enumerate() {
            let bv = s.splat(value);
            for (acc_i, &a_i) in acc.iter_mut().zip(&av) {
                acc_i[j] = if FUSED {
                    s.fma(a_i, bv, acc_i[j])
                } else {
                    s.add(acc_i[j], s.mul(a_i, bv))
                };
            }
        }
    }
    for (i, acc_i) in acc.iter().enumerate() {
        for (j, &value) in acc_i.iter().enumerate() {
            s.store(value, &mut c[j][r0 + i * w..]);
        }
    }
}
