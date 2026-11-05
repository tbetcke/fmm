//! Plane-wave M2L (Helmholtz CH7.5a; Laplace's merge and shift, CONVENTIONS §3.14): level
//! calls of `nd_fmm_kernels::plane_wave` over V lists of a uniform region of boxes, with
//! the tables of `nd_fmm_tables::PlaneWaveTables`, on both layouts the backend runs, in f32
//! and, where the device supports it, f64.
//!
//! The level: a region of side 6 (3 × 3 × 3 parents, 216 boxes) or 8 (64 parents, 512
//! boxes), each box's V list the boxes of its parent's colleagues that are not adjacent to
//! it. The *full* level has every box; the *adaptive* level drops every seventh box (no
//! source, no target) and computes only four of every five targets (the others have no V
//! row), so accumulators lose slots and parents lose children. The lists are built as
//! `nd-fmm-exec`'s host level call builds its order (`src/operator_plane_wave.rs`): per
//! parent the slots from its computed children's rows, per direction the accumulators of
//! `MergeGeometry` whose mask meets the computed children, each over its slots ascending,
//! and per computed child its accumulators in order.
//!
//! - **The host replica**, bit for bit: the four steps in the kernels' order (module
//!   documentation of `plane_wave`), the multipole rotation, the modes, the local from the
//!   sums and the rotation back with `mul_add` per term; the waves, the accumulators, the
//!   incoming blocks and the locals each checked. The accumulators and incoming blocks
//!   are also `nd_fmm_tables::plane_wave::{shift_add, add, shift}` of the device's own
//!   waves and accumulators, and the sums `sums_from_waves`, so those steps give the host
//!   functions' bits. Layouts equal each other, repeated calls are bit-identical, a call
//!   is four launches.
//! - **The host functions** (the tables' `rotate_multipole`, `outgoing_modes`,
//!   `local_from_sums`, `add_local_from_frame`, which round products and sums apart; the
//!   chain of `nd-fmm-exec`'s host `PlaneWave`): the waves within `WAVE_BOUND` and the
//!   locals within `LOCAL_BOUND` u_T of the host's, relative to each block's or column's
//!   largest value.
//! - **The pair path**: each target's local against Σ `PlaneWaveTables::m2l` over its V
//!   pairs along the merge's direction (`pair_direction`), relative to the column's
//!   largest value, within twice the host level call's distance from it plus `PAIR_FLOOR`
//!   u_T: merge and shift equals the pair path to rounding, not bit for bit.
//! - **The direct M2L** (f64): one parent's 216 slots into its eight children on levels
//!   2, 9 and 16 against `nd_fmm_ref::direct::m2l`, as the potential at four points per
//!   child within the quadrature's bound E Σ|q| (`nd_fmm_tables::plane_wave`'s test
//!   measure: E the quadrature's measured error, or the rounding floor 5e-10).
//!
//! u_T = 2⁻²⁴ (f32), 2⁻⁵³ (f64). Multipoles are uniform in [−1, 1] (the replica checks)
//! or those of four random charges per box (the direct check).

use nd_fmm_kernels::plane_wave::{
    LambdaNode, NO_BLOCK, PlaneWaveArrays, PlaneWaveLayout, PlaneWaveLists, PlaneWavePlan,
    PlaneWaveScratch, PlaneWaveTables, m2l,
};
use nd_fmm_kernels::{BackendKind, Device, Precision};
use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::{Frame, Workspace, direct};
use nd_fmm_tables::PlaneWaveTables as HostTables;
use nd_fmm_tables::geometry::m2l_offset_index;
use nd_fmm_tables::plane_wave::geometry::{CHILDREN, SLOTS, child_bits, slot_coordinates, slot_of};
use nd_fmm_tables::plane_wave::{
    Direction, add, shift, shift_add, sums_from_waves, waves_from_modes,
};

use super::{Real, assert_same, t, uniform, unit, w};
use crate::common::{Rng, tests_on};

/// The waves against the host functions', in u_T of the block's largest value.
const WAVE_BOUND: f64 = 128.0;

/// The locals against the host functions' chain, in u_T of the column's largest value.
const LOCAL_BOUND: f64 = 1024.0;

/// The locals against the pair path beyond twice the host level call's own distance from
/// it, in u_T of the column's largest value (merge and shift differs from the pair path by
/// rounding that grows with p: 15–90 u_T at p ≤ 12, 2000 u_T at p = 18 in f64).
const PAIR_FLOOR: f64 = 64.0;

/// The rounding floor of the quadrature's bound (`nd_fmm_tables::plane_wave`'s tests).
const ROUNDING_FLOOR: f64 = 5e-10;

/// The host tables of degree p in T and their arrays for the device.
struct Tables<T: Real> {
    host: HostTables<T>,
    nodes: Vec<LambdaNode>,
}

impl<T: Real> Tables<T> {
    fn new(p: usize) -> Self {
        let host = HostTables::<T>::build(p);
        let nodes = host
            .blocks()
            .iter()
            .map(|b| LambdaNode {
                start: b.start,
                len: b.len,
                modes: b.modes,
                mode_start: b.mode_start,
                trig: b.trig,
            })
            .collect();
        Self { host, nodes }
    }

    fn arrays(&self) -> PlaneWaveArrays<'_, T> {
        let h = &self.host;
        PlaneWaveArrays {
            p: h.p(),
            nodes: &self.nodes,
            outgoing_powers: h.outgoing_powers(),
            incoming_powers: h.incoming_powers(),
            cos: h.cos(),
            sin: h.sin(),
            multipole_rotations: [h.multipole_rotation(1), h.multipole_rotation(2)],
            local_rotations: [h.local_rotation(1), h.local_rotation(2)],
            source_shifts: h.source_shifts(),
            child_shifts: h.child_shifts(),
        }
    }

    fn p(&self) -> usize {
        self.host.p()
    }

    fn nc(&self) -> usize {
        (self.p() + 1) * (self.p() + 1)
    }

    fn wlen(&self) -> usize {
        self.host.wave_len()
    }
}

/// A region of `side`³ boxes (`side` even), box (x, y, z) at (x side + y) side + z.
struct Region {
    side: i64,
    present: Vec<bool>,
    computed: Vec<bool>,
}

impl Region {
    fn new(side: i64, adaptive: bool) -> Self {
        let n = (side * side * side) as usize;
        Self {
            side,
            present: (0..n).map(|b| !adaptive || b % 7 != 3).collect(),
            computed: (0..n).map(|b| !adaptive || b % 5 != 1).collect(),
        }
    }

    fn boxes(&self) -> usize {
        self.present.len()
    }

    fn coordinates(&self, b: usize) -> [i64; 3] {
        let s = self.side as usize;
        [b / (s * s), (b / s) % s, b % s].map(|x| x as i64)
    }

    fn index(&self, c: [i64; 3]) -> Option<usize> {
        c.iter()
            .all(|&x| (0..self.side).contains(&x))
            .then(|| ((c[0] * self.side + c[1]) * self.side + c[2]) as usize)
    }

    /// The V row of target `t` (present and computed): (source, offset t − s), sources
    /// ascending.
    fn row(&self, t: usize) -> Vec<(usize, [i64; 3])> {
        if !self.present[t] || !self.computed[t] {
            return Vec::new();
        }
        let ct = self.coordinates(t);
        (0..self.boxes())
            .filter(|&s| self.present[s])
            .filter_map(|s| {
                let cs = self.coordinates(s);
                let d: [i64; 3] = core::array::from_fn(|i| ct[i] - cs[i]);
                let colleague =
                    (0..3).all(|i| (ct[i].div_euclid(2) - cs[i].div_euclid(2)).abs() <= 1);
                let separated = d.iter().any(|x| x.abs() >= 2);
                (colleague && separated).then_some((s, d))
            })
            .collect()
    }
}

/// The lists of one level call, owned.
#[derive(Default)]
struct Lists {
    sources: Vec<u32>,
    merge_offsets: Vec<u32>,
    merge_entries: Vec<u32>,
    gather_shifts: Vec<u32>,
    gather_offsets: Vec<u32>,
    gather_entries: Vec<u32>,
    locals: Vec<u32>,
    incoming: Vec<u32>,
}

impl Lists {
    /// The lists of `region` with `tables` (module documentation): every present box a
    /// source, at its own position in the multipoles; locals at the box's column.
    fn new<T: Real>(region: &Region, tables: &HostTables<T>) -> Self {
        let wlen = tables.wave_len();
        let mut lists = Self {
            merge_offsets: vec![0],
            gather_offsets: vec![0],
            ..Self::default()
        };
        let mut position = vec![u32::MAX; region.boxes()];
        for (b, at) in position.iter_mut().enumerate() {
            if region.present[b] {
                *at = lists.sources.len() as u32;
                lists.sources.push(b as u32);
            }
        }
        let geometry = tables.geometry();
        let parents = region.side / 2;
        for parent in 0..parents * parents * parents {
            let pc = [
                parent / (parents * parents),
                (parent / parents) % parents,
                parent % parents,
            ];
            let children: [Option<usize>; CHILDREN] = core::array::from_fn(|o| {
                let b = child_bits(o);
                region.index(core::array::from_fn(|i| 2 * pc[i] + b[i]))
            });
            let mut computed = 0u8;
            let mut slots = [u32::MAX; SLOTS];
            for (o, child) in children.iter().enumerate() {
                let Some(t) = *child else { continue };
                let row = region.row(t);
                if row.is_empty() {
                    continue;
                }
                computed |= 1 << o;
                for (s, d) in row {
                    let slot = slot_of(o, d).expect("a V pair lies in the parent's slots");
                    slots[slot] = position[s];
                }
            }
            if computed == 0 {
                continue;
            }
            let mut incoming = [[NO_BLOCK; 6]; CHILDREN];
            for (d, direction) in Direction::ALL.into_iter().enumerate() {
                let accumulators = geometry.accumulators(direction);
                let mut index = vec![None; accumulators.len()];
                for (a, accumulator) in accumulators.iter().enumerate() {
                    if accumulator.mask & computed == 0 {
                        continue;
                    }
                    let start = lists.merge_entries.len();
                    for &slot in &accumulator.slots {
                        let i = slots[slot as usize];
                        if i == u32::MAX {
                            continue;
                        }
                        let shift = tables
                            .source_shift_offset(direction, slot as usize)
                            .expect("a slot of an accumulator has a shift")
                            / wlen;
                        lists.merge_entries.extend([6 * i + d as u32, shift as u32]);
                    }
                    if lists.merge_entries.len() > start {
                        index[a] = Some(lists.merge_offsets.len() as u32 - 1);
                        lists
                            .merge_offsets
                            .push(lists.merge_entries.len() as u32 / 2);
                    }
                }
                for (o, block) in incoming.iter_mut().enumerate() {
                    if computed & (1 << o) == 0 {
                        continue;
                    }
                    let start = lists.gather_entries.len();
                    for (a, accumulator) in accumulators.iter().enumerate() {
                        if let (Some(i), true) = (index[a], accumulator.mask & (1 << o) != 0) {
                            lists.gather_entries.push(i);
                        }
                    }
                    if lists.gather_entries.len() > start {
                        block[d] = lists.gather_shifts.len() as u32;
                        lists.gather_shifts.push((8 * d + o) as u32);
                        lists.gather_offsets.push(lists.gather_entries.len() as u32);
                    }
                }
            }
            for (o, child) in children.iter().enumerate() {
                if computed & (1 << o) != 0 && incoming[o].iter().any(|&g| g != NO_BLOCK) {
                    lists.locals.push(child.unwrap() as u32);
                    lists.incoming.extend(incoming[o]);
                }
            }
        }
        lists
    }

    fn view(&self) -> PlaneWaveLists<'_> {
        PlaneWaveLists {
            sources: &self.sources,
            merge_offsets: &self.merge_offsets,
            merge_entries: &self.merge_entries,
            gather_shifts: &self.gather_shifts,
            gather_offsets: &self.gather_offsets,
            gather_entries: &self.gather_entries,
            locals: &self.locals,
            incoming: &self.incoming,
        }
    }
}

/// The results of a level call: its waves, accumulators, incoming blocks and locals.
struct Outcome<T> {
    waves: Vec<T>,
    accumulators: Vec<T>,
    incoming: Vec<T>,
    locals: Vec<T>,
}

/// The four steps on the host: the kernels' order (`replica`) or the tables' functions.
fn host_call<T: Real>(
    tables: &Tables<T>,
    lists: &Lists,
    multipoles: &[T],
    locals: &[T],
    replica: bool,
) -> Outcome<T> {
    let h = &tables.host;
    let (nc, wlen) = (tables.nc(), tables.wlen());
    let zero = t::<T>(0.0);
    // The waves: per source and frame, up and down.
    let mut waves = vec![zero; 6 * lists.sources.len() * wlen];
    let (mut rotated, mut modes) = (vec![zero; nc], vec![zero; 4 * h.mode_count()]);
    let mut work = vec![zero; h.work_len()];
    for (i, &s) in lists.sources.iter().enumerate() {
        let x = &multipoles[s as usize * nc..(s as usize + 1) * nc];
        for frame in 0..3 {
            if replica {
                replica_rotate(h, frame, x, &mut rotated);
                replica_modes(h, &rotated, &mut modes);
            } else {
                h.rotate_multipole(frame, x, &mut rotated);
                h.outgoing_modes(&rotated, &mut modes, &mut work);
            }
            let at = (6 * i + 2 * frame) * wlen;
            let (up, down) = waves[at..at + 2 * wlen].split_at_mut(wlen);
            waves_from_modes(h, &modes, up, down);
        }
    }
    let accumulators = merge_on_host(lists, &waves, h.source_shifts(), wlen);
    let incoming = gather_on_host(lists, &accumulators, h.child_shifts(), wlen);
    let locals = local_on_host(tables, lists, &incoming, locals, replica);
    Outcome {
        waves,
        accumulators,
        incoming,
        locals,
    }
}

/// The merge from `waves` with `shift_add`.
fn merge_on_host<T: Real>(lists: &Lists, waves: &[T], shifts: &[T], wlen: usize) -> Vec<T> {
    let count = lists.merge_offsets.len() - 1;
    let mut out = vec![t::<T>(0.0); count * wlen];
    for a in 0..count {
        let acc = &mut out[a * wlen..(a + 1) * wlen];
        for e in lists.merge_offsets[a] as usize..lists.merge_offsets[a + 1] as usize {
            let (b, s) = (
                lists.merge_entries[2 * e] as usize,
                lists.merge_entries[2 * e + 1] as usize,
            );
            shift_add(
                acc,
                &waves[b * wlen..(b + 1) * wlen],
                &shifts[s * wlen..(s + 1) * wlen],
            );
        }
    }
    out
}

/// The gather from `accumulators` with `add` and `shift`.
fn gather_on_host<T: Real>(lists: &Lists, accumulators: &[T], shifts: &[T], wlen: usize) -> Vec<T> {
    let count = lists.gather_shifts.len();
    let mut out = vec![t::<T>(0.0); count * wlen];
    for g in 0..count {
        let v = &mut out[g * wlen..(g + 1) * wlen];
        for e in lists.gather_offsets[g] as usize..lists.gather_offsets[g + 1] as usize {
            let a = lists.gather_entries[e] as usize;
            add(v, &accumulators[a * wlen..(a + 1) * wlen]);
        }
        let c = lists.gather_shifts[g] as usize;
        shift(v, &shifts[c * wlen..(c + 1) * wlen]);
    }
    out
}

/// The local step from `incoming` into a copy of `locals`.
fn local_on_host<T: Real>(
    tables: &Tables<T>,
    lists: &Lists,
    incoming: &[T],
    locals: &[T],
    replica: bool,
) -> Vec<T> {
    let h = &tables.host;
    let (nc, wlen) = (tables.nc(), tables.wlen());
    let zero = t::<T>(0.0);
    let mut out = locals.to_vec();
    let (mut sums, mut frame_local) = (vec![zero; 4 * h.mode_count()], vec![zero; nc]);
    let zeros = vec![zero; wlen];
    for (l, &column) in lists.locals.iter().enumerate() {
        let local = &mut out[column as usize * nc..(column as usize + 1) * nc];
        for frame in 0..3 {
            let blocks = [
                lists.incoming[6 * l + 2 * frame],
                lists.incoming[6 * l + 2 * frame + 1],
            ];
            if blocks == [NO_BLOCK; 2] {
                continue;
            }
            let wave = |g: u32| -> &[T] {
                if g == NO_BLOCK {
                    &zeros
                } else {
                    &incoming[g as usize * wlen..(g as usize + 1) * wlen]
                }
            };
            sums_from_waves(h, wave(blocks[0]), wave(blocks[1]), &mut sums);
            if replica {
                replica_local(h, &sums, &mut frame_local);
                replica_add_back(h, frame, &frame_local, local);
            } else {
                h.local_from_sums(&sums, &mut frame_local);
                h.add_local_from_frame(frame, &frame_local, local);
            }
        }
    }
    out
}

/// `rotate_multipole` in the kernel's order: an fma per term from zero.
fn replica_rotate<T: Real>(h: &HostTables<T>, frame: usize, x: &[T], out: &mut [T]) {
    if frame == 0 {
        out.copy_from_slice(x);
        return;
    }
    let b = h.multipole_rotation(frame);
    let mut start = 0;
    for n in 0..=h.p() {
        let width = 2 * n + 1;
        for i in 0..width {
            out[n * n + i] = (0..width).fold(t::<T>(0.0), |acc, j| {
                b[start + j * width + i].mul_add(x[n * n + j], acc)
            });
        }
        start += width * width;
    }
}

/// `outgoing_modes` in the kernel's order: an fma per term from zero.
fn replica_modes<T: Real>(h: &HostTables<T>, x: &[T], modes: &mut [T]) {
    let (p, s) = (h.p(), h.blocks().len());
    let powers = h.outgoing_powers();
    let two = t::<T>(2.0);
    for (k, block) in h.blocks().iter().enumerate() {
        for m in 0..block.modes {
            let sum = |first: usize, negative: bool| {
                (first..=p).step_by(2).fold(t::<T>(0.0), |acc, n| {
                    let at = if negative {
                        n * n + n - m
                    } else {
                        n * n + n + m
                    };
                    powers[n * s + k].mul_add(x[at], acc)
                })
            };
            let (er, or) = (sum(m, false), sum(m + 1, false));
            let (ei, oi) = if m > 0 {
                (sum(m, true), sum(m + 1, true))
            } else {
                (t::<T>(0.0), t::<T>(0.0))
            };
            let weigh = |ar: T, ai: T| -> (T, T) {
                if m == 0 {
                    return (ar, t::<T>(0.0));
                }
                let sign = if m % 4 < 2 { two } else { -two };
                if m % 2 == 0 {
                    (sign * ar, -(sign * ai))
                } else {
                    (-(sign * ar), sign * ai)
                }
            };
            let q = 4 * (block.mode_start + m);
            let (ua, ub) = weigh(er + or, ei + oi);
            let (da, db) = weigh(er - or, ei - oi);
            modes[q..q + 4].copy_from_slice(&[ua, ub, da, db]);
        }
    }
}

/// `local_from_sums` in the kernel's order: per slot from zero, an fma per λ node.
fn replica_local<T: Real>(h: &HostTables<T>, sums: &[T], local: &mut [T]) {
    let p = h.p();
    let powers = h.incoming_powers();
    for n in 0..=p {
        for order in -(n as i64)..=n as i64 {
            let m = order.unsigned_abs() as usize;
            let mut acc = t::<T>(0.0);
            for (k, block) in h.blocks().iter().enumerate() {
                if m >= block.modes {
                    continue;
                }
                let q = 4 * (block.mode_start + m);
                let s = &sums[q..q + 4];
                let (up, down) = if m.is_multiple_of(2) {
                    ((s[0], -s[1]), (s[2], -s[3]))
                } else {
                    ((s[1], s[0]), (s[3], s[2]))
                };
                let (a, b) = if (n - m).is_multiple_of(2) {
                    (up.0 + down.0, up.1 + down.1)
                } else {
                    (down.0 - up.0, down.1 - up.1)
                };
                let (re, im) = match m % 4 {
                    0 => (a, b),
                    1 => (-b, a),
                    2 => (-a, -b),
                    _ => (b, -a),
                };
                let value = if order < 0 { im } else { re };
                acc = powers[k * (p + 1) + n].mul_add(value, acc);
            }
            local[((n * n + n) as i64 + order) as usize] = acc;
        }
    }
}

/// `add_local_from_frame` in the kernel's order: frame 0 adds; the others an fma per term.
fn replica_add_back<T: Real>(h: &HostTables<T>, frame: usize, local: &[T], out: &mut [T]) {
    if frame == 0 {
        for (o, &l) in out.iter_mut().zip(local) {
            *o += l;
        }
        return;
    }
    let b = h.local_rotation(frame);
    let mut start = 0;
    for n in 0..=h.p() {
        let width = 2 * n + 1;
        for i in 0..width {
            let o = &mut out[n * n + i];
            for j in 0..width {
                *o = b[start + j * width + i].mul_add(local[n * n + j], *o);
            }
        }
        start += width * width;
    }
}

/// The layouts a backend runs here: on the CPU runtime its CPU layout and a cube of 8
/// units (correctness only); on a GPU its default cube, a cube of 32 and the CPU layout.
fn layouts(device: &Device) -> Vec<PlaneWaveLayout> {
    let default = PlaneWaveLayout::default_for(device.info());
    match device.backend() {
        BackendKind::Cpu => vec![
            default,
            PlaneWaveLayout::Cube {
                units: 8.min(device.info().max_units_per_cube.max(1)),
            },
        ],
        _ => vec![
            default,
            PlaneWaveLayout::Cube { units: 32 },
            PlaneWaveLayout::Cpu,
        ],
    }
}

/// `repeats` level calls on the device in `layout` from the same locals, checked
/// bit-identical and four launches each; returns the last call's buffers.
fn device_call<T: Real>(
    device: &mut Device,
    layout: PlaneWaveLayout,
    tables: &Tables<T>,
    lists: &Lists,
    multipoles: &[T],
    locals: &[T],
    repeats: usize,
) -> Outcome<T> {
    let nc = tables.nc();
    let uploaded = PlaneWaveTables::upload(device, &tables.arrays()).unwrap();
    let plan = PlaneWavePlan::new(
        device,
        &lists.view(),
        &uploaded,
        multipoles.len() / nc,
        locals.len() / nc,
    )
    .unwrap();
    let mut scratch = PlaneWaveScratch::new(device, &plan, &uploaded).unwrap();
    let input = device.upload(multipoles).unwrap();
    let mut output = device.upload(locals).unwrap();
    let mut results: Vec<Outcome<T>> = Vec::new();
    for _ in 0..repeats {
        device.write(output.as_slice_mut(), locals).unwrap();
        let before = device.counters().launches;
        m2l(
            device,
            layout,
            &plan,
            &uploaded,
            input.as_slice(),
            output.as_slice_mut(),
            &mut scratch,
        )
        .unwrap();
        assert_eq!(
            device.counters().launches - before,
            4,
            "four launches per call"
        );
        let get = |device: &mut Device, slice: nd_fmm_kernels::DeviceSlice<'_, T>| {
            let mut out = vec![t::<T>(0.0); slice.len()];
            device.download(slice, &mut out).unwrap();
            out
        };
        results.push(Outcome {
            waves: get(device, scratch.waves()),
            accumulators: get(device, scratch.accumulators()),
            incoming: get(device, scratch.incoming()),
            locals: get(device, output.as_slice()),
        });
    }
    let last = results.pop().unwrap();
    for earlier in &results {
        assert_same("a repeated call's waves", &earlier.waves, &last.waves);
        assert_same("a repeated call's locals", &earlier.locals, &last.locals);
    }
    last
}

/// The largest |got − want| over blocks of `len` values, each relative to the block's
/// largest |want|, in u_T.
fn blockwise<T: Real>(got: &[T], want: &[T], len: usize) -> f64 {
    got.chunks(len)
        .zip(want.chunks(len))
        .map(|(g, h)| {
            let scale = h.iter().map(|&x| w(x).abs()).fold(0.0, f64::max);
            let e = g
                .iter()
                .zip(h)
                .map(|(&a, &b)| (w(a) - w(b)).abs())
                .fold(0.0, f64::max);
            if scale > 0.0 {
                e / (scale * unit::<T>())
            } else {
                e
            }
        })
        .fold(0.0, f64::max)
}

/// Σ `PlaneWaveTables::m2l` over every target's V pairs along the merge's direction, into
/// a copy of `locals`.
fn pair_path<T: Real>(
    tables: &Tables<T>,
    region: &Region,
    multipoles: &[T],
    locals: &[T],
) -> Vec<T> {
    let nc = tables.nc();
    let mut out = locals.to_vec();
    let mut scratch = tables.host.scratch();
    for target in 0..region.boxes() {
        let c = region.coordinates(target);
        let child = (4 * (c[0] & 1) + 2 * (c[1] & 1) + (c[2] & 1)) as usize;
        for (s, d) in region.row(target) {
            let offset = m2l_offset_index(d).expect("a V offset");
            tables.host.m2l(
                offset,
                HostTables::<T>::pair_direction(offset, child),
                &multipoles[s * nc..(s + 1) * nc],
                &mut out[target * nc..(target + 1) * nc],
                &mut scratch,
            );
        }
    }
    out
}

/// Every check of the module documentation at degree p on the full or adaptive region
/// of side `side`; prints one line per layout.
fn check_region<T: Real>(device: &mut Device, p: usize, side: i64, adaptive: bool) {
    let tables = Tables::<T>::new(p);
    let region = Region::new(side, adaptive);
    let lists = Lists::new(&region, &tables.host);
    let nc = tables.nc();
    let mut rng = Rng::new(0xc75a + p as u64 + 100 * side as u64 + u64::from(adaptive));
    let multipoles = uniform::<T>(&mut rng, region.boxes() * nc);
    let locals = uniform::<T>(&mut rng, region.boxes() * nc);
    let want = host_call(&tables, &lists, &multipoles, &locals, true);
    let host = host_call(&tables, &lists, &multipoles, &locals, false);
    let pair = pair_path(&tables, &region, &multipoles, &locals);
    let name = format!(
        "{} p = {p}, {} region of side {side} ({} sources, {} accumulators, {} incoming, {} locals)",
        T::FLOAT,
        if adaptive { "adaptive" } else { "full" },
        lists.sources.len(),
        lists.merge_offsets.len() - 1,
        lists.gather_shifts.len(),
        lists.locals.len()
    );
    let mut first: Option<Vec<T>> = None;
    for layout in layouts(device) {
        let what = format!("{name}, {layout}");
        let got = device_call(device, layout, &tables, &lists, &multipoles, &locals, 2);
        assert_same(&format!("{what}: the waves"), &got.waves, &want.waves);
        let wlen = tables.wlen();
        assert_same(
            &format!("{what}: the accumulators, shift_add of the device's waves"),
            &got.accumulators,
            &merge_on_host(&lists, &got.waves, tables.host.source_shifts(), wlen),
        );
        assert_same(
            &format!("{what}: the incoming blocks, add and shift of the device's accumulators"),
            &got.incoming,
            &gather_on_host(&lists, &got.accumulators, tables.host.child_shifts(), wlen),
        );
        assert_same(
            &format!("{what}: the locals, the replica of the device's incoming blocks"),
            &got.locals,
            &local_on_host(&tables, &lists, &got.incoming, &locals, true),
        );
        assert_same(&format!("{what}: the locals"), &got.locals, &want.locals);
        match &first {
            Some(first) => assert_same(
                &format!("{what}: against the first layout"),
                &got.locals,
                first,
            ),
            None => first = Some(got.locals.clone()),
        }
        let waves = blockwise(&got.waves, &host.waves, wlen);
        let local = blockwise(&got.locals, &host.locals, nc);
        let against_pair = blockwise(&got.locals, &pair, nc);
        let host_pair = blockwise(&host.locals, &pair, nc);
        println!(
            "  {what}: bit for bit the host replica (every step), repeated calls bit-identical; \
             waves ≤ {waves:.1} u_T and locals ≤ {local:.1} u_T from the host functions', \
             ≤ {against_pair:.1} u_T from the pair path (the host's {host_pair:.1} u_T)"
        );
        assert!(
            waves <= WAVE_BOUND,
            "{what}: waves {waves:.1} u_T from the host functions'"
        );
        assert!(
            local <= LOCAL_BOUND,
            "{what}: locals {local:.1} u_T from the host functions'"
        );
        assert!(
            against_pair <= 2.0 * host_pair + PAIR_FLOOR,
            "{what}: locals {against_pair:.1} u_T from the pair path"
        );
    }
}

/// The replica, the host functions and the pair path: the full and the adaptive region
/// of side 6, and the adaptive region of side 8 at p = 3.
fn plane_wave_level_calls(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        for p in [3, 8] {
            check_region::<T>(device, p, 6, false);
            check_region::<T>(device, p, 6, true);
        }
        check_region::<T>(device, 3, 8, true);
    }
    body::<f32>(device);
    if device.supports(Precision::F64) {
        body::<f64>(device);
    } else {
        println!("  f64: not supported by this device, not run");
    }
}

// --- The direct M2L (f64) ----------------------------------------------------------------

/// A xorshift generator, as `nd_fmm_tables::plane_wave`'s tests.
struct Spread(u64);

impl Spread {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Box `index` on `level` of the dyadic unit domain: centre and half-width.
fn frame(level: u32, index: [i64; 3]) -> Frame<f64> {
    let width = 0.5f64.powi(level as i32);
    Frame::new(index.map(|i| (i as f64 + 0.5) * width), width / 2.0)
}

/// Four charges in the box `f` and their scaled multipole (CONVENTIONS §3.7).
fn charges(p: usize, f: &Frame<f64>, rng: &mut Spread) -> (Vec<([f64; 3], f64)>, Vec<f64>) {
    let layout = Layout::new(p);
    let (mut multipole, mut r) = (vec![0.0; layout.len()], vec![0.0; layout.len()]);
    let mut out = Vec::new();
    for _ in 0..4 {
        let u: [f64; 3] = core::array::from_fn(|_| 2.0 * rng.next() - 1.0);
        let q = rng.next();
        harmonics::regular(p, u, &mut r);
        for n in 0..=p {
            for m in -(n as isize)..=n as isize {
                let i = layout.idx(n, m);
                multipole[i] += q * if m < 0 { -r[i] } else { r[i] };
            }
        }
        out.push((core::array::from_fn(|i| f.centre[i] + f.radius * u[i]), q));
    }
    (out, multipole)
}

/// r φ of the scaled local `local` of the box `f` at x (§3.6, §3.7).
fn potential(p: usize, f: &Frame<f64>, local: &[f64], x: [f64; 3]) -> f64 {
    let layout = Layout::new(p);
    let mut r = vec![0.0; layout.len()];
    harmonics::regular(
        p,
        core::array::from_fn(|i| (x[i] - f.centre[i]) / f.radius),
        &mut r,
    );
    let mut sum = 0.0;
    for n in 0..=p {
        sum += local[layout.idx(n, 0)] * r[layout.idx(n, 0)];
        for m in 1..=n as isize {
            let (a, b) = (layout.idx(n, m), layout.idx(n, -m));
            sum += 2.0 * (local[a] * r[a] - local[b] * r[b]);
        }
    }
    sum
}

/// Levels 2, 9 and 16 with a parent each (`nd_fmm_tables::plane_wave`'s tests).
const LEVELS: [(u32, [i64; 3]); 3] = [
    (2, [0, 1, 0]),
    (9, [127, 128, 127]),
    (16, [16383, 16384, 16383]),
];

/// One parent's 216 slots into its eight children on the device against `direct::m2l`;
/// returns the largest error over the bound E Σ|q|.
fn parent_against_direct(
    device: &mut Device,
    tables: &Tables<f64>,
    level: u32,
    parent: [i64; 3],
) -> f64 {
    let p = tables.p();
    let nc = tables.nc();
    // The region of side 6 whose centre parent (1, 1, 1) is `parent`: box a of the region is
    // slot a, at 2 parent − 2 + a on the level; only the centre parent's children compute.
    let mut region = Region::new(6, false);
    for b in 0..region.boxes() {
        let c = region.coordinates(b);
        region.computed[b] = c.iter().all(|x| x / 2 == 1);
    }
    let lists = Lists::new(&region, &tables.host);
    let mut rng = Spread(0x7e57 ^ u64::from(level) ^ (p as u64) << 8);
    let frames: Vec<Frame<f64>> = (0..region.boxes())
        .map(|b| {
            let a = region.coordinates(b);
            frame(level, core::array::from_fn(|i| 2 * parent[i] - 2 + a[i]))
        })
        .collect();
    let mut multipoles = vec![0.0; region.boxes() * nc];
    let mut sources = Vec::new();
    for (b, f) in frames.iter().enumerate() {
        let (q, m) = charges(p, f, &mut rng);
        multipoles[b * nc..(b + 1) * nc].copy_from_slice(&m);
        sources.push(q);
    }
    let zero = vec![0.0; region.boxes() * nc];
    let layout = PlaneWaveLayout::default_for(device.info());
    let got = device_call(device, layout, tables, &lists, &multipoles, &zero, 1).locals;
    let e = tables
        .host
        .quadrature()
        .measured_error()
        .max(ROUNDING_FLOOR);
    let mut ws = Workspace::new(p);
    let mut ratio: f64 = 0.0;
    for o in 0..CHILDREN {
        let b = child_bits(o);
        let target = region.index(core::array::from_fn(|i| 2 + b[i])).unwrap();
        let mut reference = vec![0.0; nc];
        let mut charge_sum = 0.0;
        for slot in 0..SLOTS {
            if tables.host.geometry().direction(slot, o).is_none() {
                continue;
            }
            let s = region.index(slot_coordinates(slot)).unwrap();
            direct::m2l(
                p,
                &frames[s],
                &frames[target],
                &mut ws,
                &multipoles[s * nc..(s + 1) * nc],
                &mut reference,
            );
            charge_sum += sources[s].iter().map(|&(_, q)| q.abs()).sum::<f64>();
        }
        let local = &got[target * nc..(target + 1) * nc];
        let f = &frames[target];
        for _ in 0..4 {
            let x = core::array::from_fn(|i| f.centre[i] + f.radius * (2.0 * rng.next() - 1.0));
            let error = (potential(p, f, local, x) - potential(p, f, &reference, x)).abs();
            ratio = ratio.max(error / (e * charge_sum));
        }
    }
    ratio
}

/// The direct check at the degrees `degrees` in f64.
fn against_direct(device: &mut Device, degrees: &[usize]) {
    if !device.supports(Precision::F64) {
        println!("  f64: not supported by this device, not run");
        return;
    }
    for &p in degrees {
        let tables = Tables::<f64>::new(p);
        for (level, parent) in LEVELS {
            let ratio = parent_against_direct(device, &tables, level, parent);
            assert!(
                ratio <= 1.0,
                "p = {p}, level {level}: {ratio:.3} of the quadrature's bound"
            );
            println!(
                "  f64 p = {p:2}, level {level:2}: {ratio:.3} of the quadrature's bound E Σ|q| \
                 against direct::m2l"
            );
        }
    }
}

/// The direct check at p = 3 and 8.
fn plane_wave_against_direct(device: &mut Device) {
    against_direct(device, &[3, 8]);
}

/// Refusals and empty calls: a cube of no unit; a plan without local items launches
/// three times.
fn plane_wave_refused(device: &mut Device) {
    let tables = Tables::<f32>::new(3);
    let mut region = Region::new(6, false);
    region.computed.fill(false);
    let lists = Lists::new(&region, &tables.host);
    assert!(lists.locals.is_empty() && lists.merge_entries.is_empty());
    let nc = tables.nc();
    let uploaded = PlaneWaveTables::upload(device, &tables.arrays()).unwrap();
    let plan = PlaneWavePlan::new(
        device,
        &lists.view(),
        &uploaded,
        region.boxes(),
        region.boxes(),
    )
    .unwrap();
    let mut scratch = PlaneWaveScratch::new(device, &plan, &uploaded).unwrap();
    let input = device
        .upload(&vec![t::<f32>(1.0); region.boxes() * nc])
        .unwrap();
    let mut output = device
        .upload(&vec![t::<f32>(0.0); region.boxes() * nc])
        .unwrap();
    let before = device.counters().launches;
    let layout = PlaneWaveLayout::default_for(device.info());
    m2l(
        device,
        layout,
        &plan,
        &uploaded,
        input.as_slice(),
        output.as_slice_mut(),
        &mut scratch,
    )
    .unwrap();
    assert_eq!(
        device.counters().launches - before,
        1,
        "the waves only: no accumulator, block or local"
    );
    let refused = m2l(
        device,
        PlaneWaveLayout::Cube { units: 0 },
        &plan,
        &uploaded,
        input.as_slice(),
        output.as_slice_mut(),
        &mut scratch,
    );
    assert!(refused.is_err(), "a cube of no unit");
    println!(
        "  f32 plane-wave M2L: no target, the waves alone; {}",
        refused.unwrap_err()
    );
}

tests_on!(cpu: plane_wave_level_calls, plane_wave_against_direct, plane_wave_refused);

/// Every degree p ≤ 20 against the direct M2L, and the level calls at p = 12, 18 and 20
/// in f64, on the CPU runtime (release).
#[cfg(feature = "cpu")]
#[test]
#[ignore = "every degree on the CPU runtime: run with -- --ignored"]
fn plane_wave_every_degree_on_the_cpu_runtime() {
    crate::common::run(BackendKind::Cpu, "plane_wave_every_degree", |device| {
        against_direct(device, &(0..=20).collect::<Vec<_>>());
        for p in [12, 18, 20] {
            check_region::<f64>(device, p, 6, true);
        }
    });
}

mod gpu {
    use super::*;

    tests_on!(metal: plane_wave_level_calls, plane_wave_against_direct, plane_wave_refused);
    tests_on!(cuda: plane_wave_level_calls, plane_wave_against_direct, plane_wave_refused);
}
