//! The bodies of the plane-wave kernels of [`WaveKernel`](crate::WaveKernel), written once
//! over the vector layer [`Simd`] (Phase 6 T13, C6.5; CONVENTIONS §3.14).
//!
//! Every output value receives a fixed chain of IEEE 754 operations, each fused
//! multiply–add rounded once, and depends on its own inputs only: the vector paths give
//! the scalar layer's bits on every ISA, whatever W, the register block or the tails.
//! The chains are those of `nd_fmm_tables::plane_wave` (`shift_add`, `add`, `shift`,
//! `waves_from_modes` and `sums_from_waves`), which define them:
//!
//! - [`merge_body`]: per value, for each source i in order, re ← fma(x_r, t_r, re),
//!   re ← fnma(x_i, t_i, re), im ← fma(x_r, t_i, im), im ← fma(x_i, t_r, im), starting
//!   from the accumulator's value; vectors over the values, R of them for the real and
//!   the imaginary parts held in registers across the sources.
//! - [`gather_body`]: per value, x ← 0, x ← x + aᵢ for each accumulator i in order, then
//!   re ← fnma(x_i, t_i, x_r t_r) (x_r t_r rounded, then x_i t_i subtracted fused),
//!   im ← fma(x_r, t_i, x_i t_r).
//! - [`waves_body`]: per λ node and value j, the four outputs from zero, for each mode m
//!   ascending fma(a, cos mαⱼ, x) then fma(b, sin mαⱼ, x) into the real (even m) or the
//!   imaginary (odd m) parts; vectors over j, R of each output in registers across the
//!   modes, the rows of [`WaveTables`] padded to the lanes so that every load is whole.
//! - [`sums_body`]: per λ node, mode and output, from zero, fma(xⱼ, cⱼ, s) for j
//!   ascending; vectors over a node's row of [`WaveTables`] (the pairs (cos, sin) of the
//!   even modes, then of the odd modes, each padded to the lanes), both directions at
//!   once, the four values xⱼ (each direction's real and imaginary part) broadcast.

use super::scalar::Scalar;
use super::{MAX_LANES, Simd};
use crate::SimdScalar;
use crate::waves::WaveBlock;

/// Vectors per register block of [`merge_body`] (for each of the two parts) and of
/// [`sums_body`] (for each direction), and of [`waves_body`] where 32 registers hold it.
pub(crate) const BLOCK: usize = 4;

/// The largest row of a node in [`sums_body`]: twice its modes (at most 63,
/// `WaveKernel::sums`), each parity's pairs rounded up to the lanes.
const MAX_ROW: usize = 2 * 64 + 2 * MAX_LANES;

/// acc += Σᵢ xᵢ ⊙ tᵢ, complex, over waves of 2H values (the real parts, then the imaginary
/// parts), xᵢ the 2H values of `waves` at `wave_offsets[i]` and tᵢ those of `shifts` at
/// `shift_offsets[i]` (module documentation). Shapes checked by the caller.
#[inline(always)]
pub(crate) fn merge_body<T: SimdScalar, S: Simd<T>>(
    s: S,
    acc: &mut [T],
    waves: &[T],
    wave_offsets: &[usize],
    shifts: &[T],
    shift_offsets: &[usize],
) {
    let h = acc.len() / 2;
    let (re, im) = acc.split_at_mut(h);
    let mut j = 0;
    while j + BLOCK * S::W <= h {
        merge_block::<T, S, BLOCK>(s, j, h, re, im, waves, wave_offsets, shifts, shift_offsets);
        j += BLOCK * S::W;
    }
    while j + S::W <= h {
        merge_block::<T, S, 1>(s, j, h, re, im, waves, wave_offsets, shifts, shift_offsets);
        j += S::W;
    }
    while j < h {
        merge_block::<T, Scalar, 1>(
            Scalar,
            j,
            h,
            re,
            im,
            waves,
            wave_offsets,
            shifts,
            shift_offsets,
        );
        j += 1;
    }
}

/// Values j … j + R W of [`merge_body`]: loaded once, every source in order, stored once.
#[expect(
    clippy::too_many_arguments,
    reason = "the body's arguments, split by block"
)]
#[inline(always)]
fn merge_block<T: SimdScalar, S: Simd<T>, const R: usize>(
    s: S,
    j: usize,
    h: usize,
    re: &mut [T],
    im: &mut [T],
    waves: &[T],
    wave_offsets: &[usize],
    shifts: &[T],
    shift_offsets: &[usize],
) {
    let (w, span) = (S::W, R * S::W);
    let (re, im) = (&mut re[j..j + span], &mut im[j..j + span]);
    let mut ar: [S::V; R] = core::array::from_fn(|r| s.load(&re[r * w..]));
    let mut ai: [S::V; R] = core::array::from_fn(|r| s.load(&im[r * w..]));
    for (&wo, &so) in wave_offsets.iter().zip(shift_offsets) {
        // Each source's parts at the block's values, sliced once.
        let (x, t) = (&waves[wo..wo + 2 * h], &shifts[so..so + 2 * h]);
        let (xr, xi) = (&x[j..j + span], &x[h + j..h + j + span]);
        let (tr, ti) = (&t[j..j + span], &t[h + j..h + j + span]);
        let parts = xr
            .chunks_exact(w)
            .zip(xi.chunks_exact(w))
            .zip(tr.chunks_exact(w).zip(ti.chunks_exact(w)));
        for (r, ((xr, xi), (tr, ti))) in parts.enumerate() {
            let (xr, xi, tr, ti) = (s.load(xr), s.load(xi), s.load(tr), s.load(ti));
            ar[r] = s.fma(xr, tr, ar[r]);
            ar[r] = s.fnma(xi, ti, ar[r]);
            ai[r] = s.fma(xr, ti, ai[r]);
            ai[r] = s.fma(xi, tr, ai[r]);
        }
    }
    for r in 0..R {
        s.store(ar[r], &mut re[r * w..]);
        s.store(ai[r], &mut im[r * w..]);
    }
}

/// v ← (Σᵢ aᵢ) ⊙ t, complex, over waves of 2H values, aᵢ the 2H values of `accs` at
/// `acc_offsets[i]` (module documentation). Shapes checked by the caller.
#[inline(always)]
pub(crate) fn gather_body<T: SimdScalar, S: Simd<T>>(
    s: S,
    v: &mut [T],
    accs: &[T],
    acc_offsets: &[usize],
    t: &[T],
) {
    let h = v.len() / 2;
    let (vr, vi) = v.split_at_mut(h);
    let mut j = 0;
    while j + S::W <= h {
        gather_value(s, j, h, vr, vi, accs, acc_offsets, t);
        j += S::W;
    }
    while j < h {
        gather_value(Scalar, j, h, vr, vi, accs, acc_offsets, t);
        j += 1;
    }
}

/// Values j … j + W of [`gather_body`].
#[expect(
    clippy::too_many_arguments,
    reason = "the body's arguments, split by vector"
)]
#[inline(always)]
fn gather_value<T: SimdScalar, S: Simd<T>>(
    s: S,
    j: usize,
    h: usize,
    vr: &mut [T],
    vi: &mut [T],
    accs: &[T],
    acc_offsets: &[usize],
    t: &[T],
) {
    let (mut xr, mut xi) = (s.splat(T::zero()), s.splat(T::zero()));
    let w = S::W;
    for &o in acc_offsets {
        let a = &accs[o..o + 2 * h];
        xr = s.add(xr, s.load(&a[j..j + w]));
        xi = s.add(xi, s.load(&a[h + j..h + j + w]));
    }
    let (tr, ti) = (s.load(&t[j..j + w]), s.load(&t[h + j..h + j + w]));
    // re = x_r t_r − x_i t_i, the product x_r t_r rounded, im = fma(x_r, t_i, x_i t_r).
    let re = s.fnma(xi, ti, s.mul(xr, tr));
    let im = s.fma(xr, ti, s.mul(xi, tr));
    s.store(re, &mut vr[j..j + w]);
    s.store(im, &mut vi[j..j + w]);
}

/// The up and down waves (2H values each) from the weighted modes `modes` (four per mode:
/// a↑, b↑, a↓, b↓), with the kernel's tables `tables` (module documentation), R vectors of
/// each of the four outputs in registers (4R accumulators, 4 coefficients, 2 rows: R = 2
/// on AVX2's 16 registers, 4 on 32). Writes every value of the nodes' blocks in `up` and
/// `down`. Shapes checked by the caller; the tables' lanes are `S::W`.
#[inline(always)]
pub(crate) fn waves_body<T: SimdScalar, S: Simd<T>, const R: usize>(
    s: S,
    tables: &WaveTables<T>,
    modes: &[T],
    up: &mut [T],
    down: &mut [T],
) {
    let h = up.len() / 2;
    for node in &tables.nodes {
        let chunks = node.stride / S::W;
        let mut c = 0;
        while c + R <= chunks {
            waves_group::<T, S, R>(s, tables, node, c * S::W, h, modes, up, down);
            c += R;
        }
        while c < chunks {
            waves_group::<T, S, 1>(s, tables, node, c * S::W, h, modes, up, down);
            c += 1;
        }
    }
}

/// Values j … j + R W of the node `node` in [`waves_body`]: the four outputs of R vectors
/// in registers, every mode in order with its four coefficients splatted once, the rows
/// read whole (padded to the lanes), the values beyond the node's length not stored.
#[expect(
    clippy::too_many_arguments,
    reason = "the body's arguments, split by register block"
)]
#[inline(always)]
fn waves_group<T: SimdScalar, S: Simd<T>, const R: usize>(
    s: S,
    tables: &WaveTables<T>,
    node: &Node,
    j: usize,
    h: usize,
    modes: &[T],
    up: &mut [T],
    down: &mut [T],
) {
    let w = S::W;
    let zero = s.splat(T::zero());
    let (mut ur, mut ui, mut dr, mut di) = ([zero; R], [zero; R], [zero; R], [zero; R]);
    // The node's coefficients and rows, sliced once, so that the loop checks no bound.
    let span = node.modes * node.stride;
    let coefficients = &modes[4 * node.mode_start..4 * (node.mode_start + node.modes)];
    let rows = tables.cos[node.trig..node.trig + span]
        .chunks_exact(node.stride)
        .zip(tables.sin[node.trig..node.trig + span].chunks_exact(node.stride));
    for (m, (coefficients, (c, sn))) in coefficients.as_chunks::<4>().0.iter().zip(rows).enumerate()
    {
        let (c, sn) = (&c[j..j + R * w], &sn[j..j + R * w]);
        // Each parity names its accumulators (a choice between them at run time would
        // keep them in memory).
        if m % 2 == 0 {
            waves_mode(s, coefficients, c, sn, &mut ur, &mut dr);
        } else {
            waves_mode(s, coefficients, c, sn, &mut ui, &mut di);
        }
    }
    for r in 0..R {
        let at = j + r * w;
        if at >= node.len {
            break;
        }
        let n = (node.len - at).min(w);
        let base = node.start + at;
        store_n(s, ur[r], &mut up[base..base + n]);
        store_n(s, ui[r], &mut up[h + base..h + base + n]);
        store_n(s, dr[r], &mut down[base..base + n]);
        store_n(s, di[r], &mut down[h + base..h + base + n]);
    }
}

/// One mode of [`waves_group`] into its up and down accumulators `u` and `d`, with its
/// coefficients (a↑, b↑, a↓, b↓) and its rows `c` and `sn` at the group's values (R W
/// each): per vector fma(a↑, cos, u), fma(b↑, sin, u), fma(a↓, cos, d), fma(b↓, sin, d).
#[inline(always)]
fn waves_mode<T: SimdScalar, S: Simd<T>, const R: usize>(
    s: S,
    coefficients: &[T; 4],
    c: &[T],
    sn: &[T],
    u: &mut [S::V; R],
    d: &mut [S::V; R],
) {
    let [au, bu, ad, bd] = coefficients.map(|x| s.splat(x));
    for (r, (c, sn)) in c.chunks_exact(S::W).zip(sn.chunks_exact(S::W)).enumerate() {
        let (c, sn) = (s.load(c), s.load(sn));
        u[r] = s.fma(au, c, u[r]);
        u[r] = s.fma(bu, sn, u[r]);
        d[r] = s.fma(ad, c, d[r]);
        d[r] = s.fma(bd, sn, d[r]);
    }
}

/// The first `out.len()` (at most W) lanes of `v` into `out`: a whole store, or through a
/// padded copy for fewer.
#[inline(always)]
fn store_n<T: SimdScalar, S: Simd<T>>(s: S, v: S::V, out: &mut [T]) {
    if out.len() == S::W {
        s.store(v, out);
    } else {
        let mut padded = [T::zero(); MAX_LANES];
        s.store(v, &mut padded);
        out.copy_from_slice(&padded[..out.len()]);
    }
}

/// The incoming sums of the up and down waves (2H values each), four per mode, from the
/// kernel's tables `tables` (module documentation). Writes every value of `sums` the nodes
/// name. Shapes checked by the caller; the tables' lanes are `S::W`.
#[inline(always)]
pub(crate) fn sums_body<T: SimdScalar, S: Simd<T>>(
    s: S,
    tables: &WaveTables<T>,
    up: &[T],
    down: &[T],
    sums: &mut [T],
) {
    let h = up.len() / 2;
    // The row of one node, both directions: the even modes' pairs, then the odd modes'.
    let mut out = [[T::zero(); MAX_ROW]; 2];
    for node in &tables.nodes {
        let row = node.even_pad + node.odd_pad;
        let chunks = row / S::W;
        let waves = [up, down]
            .map(|wave| [0, h].map(|half| &wave[half + node.start..half + node.start + node.len]));
        let table = &tables.pairs[node.pairs..node.pairs + node.len * row];
        // The even modes' chunks, then the odd modes', each parity a pass over the rows.
        let even = node.even_pad / S::W;
        let mut c = 0;
        while c + BLOCK <= even {
            sums_block::<T, S, BLOCK, false>(s, c, &waves, table, row, &mut out);
            c += BLOCK;
        }
        while c < even {
            sums_block::<T, S, 1, false>(s, c, &waves, table, row, &mut out);
            c += 1;
        }
        while c + BLOCK <= chunks {
            sums_block::<T, S, BLOCK, true>(s, c, &waves, table, row, &mut out);
            c += BLOCK;
        }
        while c < chunks {
            sums_block::<T, S, 1, true>(s, c, &waves, table, row, &mut out);
            c += 1;
        }
        // Value 2i + c of a parity's segment is mode parity + 2i, cos (c = 0) or sin.
        let even_modes = node.modes.div_ceil(2);
        for (dir, out) in out.iter().enumerate() {
            for (parity, offset, count) in [
                (0, 0, even_modes),
                (1, node.even_pad, node.modes - even_modes),
            ] {
                let pairs = out[offset..offset + 2 * count].as_chunks::<2>().0;
                for (i, pair) in pairs.iter().enumerate() {
                    let k = 4 * (node.mode_start + parity + 2 * i) + 2 * dir;
                    sums[k] = pair[0];
                    sums[k + 1] = pair[1];
                }
            }
        }
    }
}

/// Chunks c … c + R of one node's row in [`sums_body`], all of the even modes' pairs or
/// (`ODD`) all of the odd modes', for both directions: from zero, every row j in order,
/// each chunk times the value of its direction (the real parts for the even modes, the
/// imaginary parts for the odd), broadcast.
#[inline(always)]
fn sums_block<T: SimdScalar, S: Simd<T>, const R: usize, const ODD: bool>(
    s: S,
    c: usize,
    waves: &[[&[T]; 2]; 2],
    table: &[T],
    row: usize,
    out: &mut [[T; MAX_ROW]; 2],
) {
    let w = S::W;
    let zero = s.splat(T::zero());
    let (mut up, mut down) = ([zero; R], [zero; R]);
    let part = usize::from(ODD);
    let (xu, xd) = (waves[0][part], waves[1][part]);
    for ((rowj, &xu), &xd) in table.chunks_exact(row).zip(xu).zip(xd) {
        let (xu, xd) = (s.splat(xu), s.splat(xd));
        for (r, value) in rowj[c * w..(c + R) * w].chunks_exact(w).enumerate() {
            let value = s.load(value);
            up[r] = s.fma(xu, value, up[r]);
            down[r] = s.fma(xd, value, down[r]);
        }
    }
    for r in 0..R {
        let at = (c + r) * w;
        s.store(up[r], &mut out[0][at..at + w]);
        s.store(down[r], &mut out[1][at..at + w]);
    }
}

/// One λ node in the kernel's layout ([`WaveTables`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Node {
    /// Its first value among the H values of a half, and its values.
    pub(crate) start: usize,
    pub(crate) len: usize,
    /// Its modes, and its first mode among all the nodes'.
    pub(crate) modes: usize,
    pub(crate) mode_start: usize,
    /// `len` rounded up to the lanes: the row length of `cos` and `sin`.
    pub(crate) stride: usize,
    /// Its rows in `cos` and `sin`: row m at `trig + m stride`.
    pub(crate) trig: usize,
    /// Its rows in `pairs`: row j at `pairs + j (even_pad + odd_pad)`, the even modes'
    /// pairs (2 ⌈F/2⌉ values) padded to the lanes, then the odd modes' likewise.
    pub(crate) pairs: usize,
    pub(crate) even_pad: usize,
    pub(crate) odd_pad: usize,
}

/// The tables of [`WaveKernel`](crate::WaveKernel)'s Fourier passes in the layout of one
/// lane count W (crate-private fields; built by `WaveKernel::tables`): per λ node the cos
/// and sin rows padded to W with zeros, and the incoming pairs by value, each parity's
/// segment padded to W with zeros, so that every vector load is whole.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveTables<T> {
    pub(crate) nodes: Vec<Node>,
    pub(crate) cos: Vec<T>,
    pub(crate) sin: Vec<T>,
    pub(crate) pairs: Vec<T>,
    /// The lanes W of the layout.
    pub(crate) lanes: usize,
    /// H, and the modes of all the nodes.
    pub(crate) h: usize,
    pub(crate) modes: usize,
}

impl<T: SimdScalar> WaveTables<T> {
    /// The layout of `blocks` with the rows of `cos` and `sin` (row m of node k at
    /// `trig + m len`) for `lanes` lanes.
    pub(crate) fn new(blocks: &[WaveBlock], cos: &[T], sin: &[T], lanes: usize) -> Self {
        let pad = |n: usize| n.div_ceil(lanes) * lanes;
        let (mut nodes, mut tc, mut ts, mut pairs) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut h, mut modes) = (0, 0);
        for b in blocks {
            let (stride, even) = (pad(b.len), b.modes.div_ceil(2));
            let (even_pad, odd_pad) = (pad(2 * even), pad(2 * (b.modes - even)));
            let node = Node {
                start: b.start,
                len: b.len,
                modes: b.modes,
                mode_start: b.mode_start,
                stride,
                trig: tc.len(),
                pairs: pairs.len(),
                even_pad,
                odd_pad,
            };
            for m in 0..b.modes {
                let row = b.trig + m * b.len;
                tc.extend_from_slice(&cos[row..row + b.len]);
                ts.extend_from_slice(&sin[row..row + b.len]);
                tc.resize(tc.len() + stride - b.len, T::zero());
                ts.resize(ts.len() + stride - b.len, T::zero());
            }
            for j in 0..b.len {
                for (parity, padded) in [(0, even_pad), (1, odd_pad)] {
                    let at = pairs.len();
                    for m in (parity..b.modes).step_by(2) {
                        pairs.push(cos[b.trig + m * b.len + j]);
                        pairs.push(sin[b.trig + m * b.len + j]);
                    }
                    pairs.resize(at + padded, T::zero());
                }
            }
            h = h.max(b.start + b.len);
            modes = modes.max(b.mode_start + b.modes);
            nodes.push(node);
        }
        Self {
            nodes,
            cos: tc,
            sin: ts,
            pairs,
            lanes,
            h,
            modes,
        }
    }
}
