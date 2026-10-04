//! Rotation M2L (T10, C4.6): level calls of `nd_fmm_kernels::rotation::m2l` over V-list-
//! shaped views of the 316 offsets with the M2L family of `nd_fmm_tables::RotationTables`,
//! on both layouts the backend runs, in f32 and, where the device supports it, f64.
//!
//! The views: rows are boxes with a random subset of the 316 offsets in index order (one
//! source per offset, sources distinct within a row); every seventh row has no entry. A
//! second view holds the special offsets explicitly: the four on the z axis
//! (`Alignment::Up` at (0, 0, 2) and (0, 0, 3), `Down` at (0, 0, −2) and (0, 0, −3)) and
//! every offset with the smallest and the largest stored polar angle, one row per offset
//! and a row with all of them. Multipoles and locals are separate buffers, as the device
//! operator passes them; rows with entries start from random locals, rows without from
//! zero.
//!
//! - **The host replica**, bit for bit: each target's row in offset order, each pair the
//!   kernel's operations in the kernel's order (`ShiftTables::apply` step for step, every
//!   multiply–add an explicit `mul_add`; module documentation of `rotation`), on every
//!   layout. The layouts equal each other, repeated calls are bit-identical, rows without
//!   entries stay zero, and one call is one launch.
//! - **`RotationTables::m2l`** (the host operator, which rounds every product): within
//!   2 (5p + 10) u_T τ, with τ each output's terms, the same chain run on the absolute
//!   values of the inputs, tables and starting locals (each step's rounding is within u_T
//!   of its terms, and the chain has at most 5p + 10 of them). The share of values equal
//!   bit for bit is printed: the kernel's fused multiply–adds differ from the host's in
//!   the last bits (cubecl-opt's contraction; CONVENTIONS §3.13, "Device kernels").
//!
//! u_T = 2⁻²⁴ (f32), 2⁻⁵³ (f64). Values are uniform in [−1, 1], so a flushing backend adds
//! them as the host does. The accuracy against `nd_fmm_ref::direct` per degree, on levels
//! 2, 9 and 16, is `nd-fmm-exec`'s operator test (`tests/operator/device_rotation.rs`).

use nd_fmm_kernels::rotation::{
    Alignment, RotationArrays, RotationLayout, RotationPlan, RotationTables, Shift, m2l,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, Precision};
use nd_fmm_tables::RotationScratch;
use nd_fmm_tables::geometry::m2l_offsets;
use nd_fmm_tables::rotation::{self as host, Operator, ShiftTables};

use super::{Real, assert_same, download, t, uniform, unit, w};
use crate::common::{Rng, tests_on};

/// The V-list offsets of `nd_fmm_tables::geometry::m2l_offsets`.
const OFFSETS: usize = 316;

/// The M2L family of the rotation tables at degree p in T, and its arrays for the device.
struct HostTables<T: Real> {
    tables: ShiftTables<T>,
    forward: Vec<T>,
    backward: Vec<T>,
    azimuth: Vec<T>,
    coaxial: Vec<T>,
    shifts: Vec<Shift>,
}

impl<T: Real> HostTables<T> {
    fn new(p: usize) -> Self {
        let tables = nd_fmm_tables::RotationTables::<f64>::build(p)
            .tables(Operator::M2l)
            .cast::<T>();
        let concat = |count: usize, piece: &dyn Fn(usize) -> Vec<T>| -> Vec<T> {
            (0..count).flat_map(piece).collect()
        };
        let forward = concat(tables.polar_count(), &|i| tables.forward_blocks(i).to_vec());
        let backward = concat(tables.polar_count(), &|i| {
            tables.backward_blocks(i).to_vec()
        });
        let azimuth = concat(tables.azimuth_count(), &|i| {
            tables.azimuth_factors(i).to_vec()
        });
        let coaxial = concat(tables.distance_count(), &|i| {
            tables.coaxial_factors(i).to_vec()
        });
        let shifts = (0..tables.count())
            .map(|d| {
                let shift = tables.shift(d);
                Shift {
                    alignment: match shift.alignment {
                        host::Alignment::Up => Alignment::Up,
                        host::Alignment::Down => Alignment::Down,
                        host::Alignment::Rotated { polar, azimuth } => Alignment::Rotated {
                            polar: polar as u32,
                            azimuth: azimuth as u32,
                        },
                    },
                    distance: shift.distance as u32,
                }
            })
            .collect();
        Self {
            tables,
            forward,
            backward,
            azimuth,
            coaxial,
            shifts,
        }
    }

    fn arrays(&self) -> RotationArrays<'_, T> {
        RotationArrays {
            p: self.tables.p(),
            forward: &self.forward,
            backward: &self.backward,
            azimuth: &self.azimuth,
            coaxial: &self.coaxial,
            shifts: &self.shifts,
        }
    }

    fn p(&self) -> usize {
        self.tables.p()
    }

    /// The kernel's translation of offset d on the host (module documentation), added to
    /// `local`; with `magnitude` the same chain on absolute values, every factor and input
    /// taken by its magnitude (the terms τ).
    fn replica(&self, d: usize, x: &[T], local: &mut [T], magnitude: bool) {
        let p = self.p();
        let nc = (p + 1) * (p + 1);
        let v = |a: T| if magnitude { t::<T>(w(a).abs()) } else { a };
        let fma = |a: T, b: T, c: T| v(a).mul_add(v(b), c);
        let slot = |n: usize, m: i64| (n * n + n) as i64 + m;
        let slots = || (0..=p).flat_map(|n| (-(n as i64)..=n as i64).map(move |m| (n, m)));
        let shift = self.tables.shift(d);
        let coaxial = self.tables.coaxial_factors(shift.distance);
        // Slot (j, m) of the coaxial step on `input`, from `init`.
        let coaxial_slot = |input: &[T], j: usize, m: i64, down: bool, init: T| -> T {
            let i = m.unsigned_abs() as usize;
            let width = p + 1 - i;
            let start: usize = (0..i).map(|i| (p + 1 - i) * (p + 1 - i)).sum();
            let row = start + (j - i) * width;
            let mut acc = init;
            for n in i..=p {
                let mut a = coaxial[row + n - i];
                if down && (n + j) % 2 == 1 {
                    a = -a;
                }
                acc = fma(a, input[slot(n, m) as usize], acc);
            }
            acc
        };
        match shift.alignment {
            host::Alignment::Up | host::Alignment::Down => {
                let down = shift.alignment == host::Alignment::Down;
                for (j, m) in slots() {
                    let k = slot(j, m) as usize;
                    local[k] = coaxial_slot(x, j, m, down, local[k]);
                }
            }
            host::Alignment::Rotated { polar, azimuth } => {
                let factors = self.tables.azimuth_factors(azimuth);
                let rotate_z = |input: &[T], n: usize, m: i64, inverse: bool| -> T {
                    let centre = slot(n, 0) as usize;
                    if m == 0 {
                        return v(input[centre]);
                    }
                    let mm = m.unsigned_abs() as usize;
                    let c = factors[2 * (mm - 1)];
                    let s = factors[2 * (mm - 1) + 1];
                    let s = if inverse { -s } else { s };
                    let (u, w) = (input[centre + mm], input[centre - mm]);
                    if m > 0 {
                        let sv = v(s) * v(w);
                        fma(c, u, if magnitude { sv } else { -sv })
                    } else {
                        fma(s, u, v(c) * v(w))
                    }
                };
                let y_blocks = |blocks: &[T], input: &[T], n: usize, m: i64| -> T {
                    let width = 2 * n + 1;
                    let row = (4 * n * n * n - n) / 3 + (m + n as i64) as usize * width;
                    (0..width).fold(t::<T>(0.0), |acc, c| {
                        fma(blocks[row + c], input[n * n + c], acc)
                    })
                };
                let (forward, backward) = (
                    self.tables.forward_blocks(polar),
                    self.tables.backward_blocks(polar),
                );
                let mut first = vec![t::<T>(0.0); nc];
                let mut second = vec![t::<T>(0.0); nc];
                for (n, m) in slots() {
                    first[slot(n, m) as usize] = rotate_z(x, n, m, false);
                }
                for (n, m) in slots() {
                    second[slot(n, m) as usize] = y_blocks(forward, &first, n, m);
                }
                for (n, m) in slots() {
                    first[slot(n, m) as usize] = coaxial_slot(&second, n, m, false, t::<T>(0.0));
                }
                for (n, m) in slots() {
                    second[slot(n, m) as usize] = y_blocks(backward, &first, n, m);
                }
                for (n, m) in slots() {
                    let k = slot(n, m) as usize;
                    local[k] += rotate_z(&second, n, m, true);
                }
            }
        }
    }
}

/// A V-list-shaped view: the six arrays of `GroupedArrays` with `u16` offset indices.
struct VView {
    row_offsets: Vec<u32>,
    sources: Vec<u32>,
    groups: Vec<u16>,
    batch_offsets: Vec<u32>,
    batch_targets: Vec<u32>,
    batch_sources: Vec<u32>,
    nsources: usize,
}

impl VView {
    /// A view of `rows`, each a list of offsets ascending, sources among `nsources`
    /// (at least 316) distinct within a row: a random rotation of the sources by offset.
    fn new(rng: &mut Rng, rows: &[Vec<u16>], nsources: usize) -> Self {
        let rows: Vec<Vec<(u16, u32)>> = rows
            .iter()
            .map(|row| {
                let shift = rng.below(nsources);
                row.iter()
                    .map(|&d| (d, ((shift + 7 * d as usize) % nsources) as u32))
                    .collect()
            })
            .collect();
        let mut view = Self {
            row_offsets: vec![0],
            sources: Vec::new(),
            groups: Vec::new(),
            batch_offsets: vec![0],
            batch_targets: Vec::new(),
            batch_sources: Vec::new(),
            nsources,
        };
        for row in &rows {
            for &(d, s) in row {
                view.groups.push(d);
                view.sources.push(s);
            }
            view.row_offsets.push(view.sources.len() as u32);
        }
        for d in 0..OFFSETS as u16 {
            for (target, row) in rows.iter().enumerate() {
                if let Some(&(_, s)) = row.iter().find(|e| e.0 == d) {
                    view.batch_targets.push(target as u32);
                    view.batch_sources.push(s);
                }
            }
            view.batch_offsets.push(view.batch_targets.len() as u32);
        }
        view
    }

    /// `nrows` rows (every seventh empty), each offset with probability 1 in `sparsity`.
    fn random(rng: &mut Rng, nrows: usize, sparsity: usize) -> Self {
        let rows: Vec<Vec<u16>> = (0..nrows)
            .map(|r| {
                if r % 7 == 6 {
                    Vec::new()
                } else {
                    (0..OFFSETS as u16)
                        .filter(|_| rng.below(sparsity) == 0)
                        .collect()
                }
            })
            .collect();
        Self::new(rng, &rows, OFFSETS + nrows)
    }

    fn arrays(&self) -> GroupedArrays<'_, u16> {
        GroupedArrays {
            row_offsets: &self.row_offsets,
            sources: &self.sources,
            groups: &self.groups,
            batch_offsets: &self.batch_offsets,
            batch_targets: &self.batch_targets,
            batch_sources: &self.batch_sources,
        }
    }

    fn nrows(&self) -> usize {
        self.row_offsets.len() - 1
    }

    fn row(&self, r: usize) -> std::ops::Range<usize> {
        self.row_offsets[r] as usize..self.row_offsets[r + 1] as usize
    }
}

/// The special offsets (module documentation), in index order: the four on the z axis and
/// those with the smallest and the largest stored polar angle.
fn special_offsets<T: Real>(tables: &ShiftTables<T>) -> Vec<u16> {
    let polars: Vec<f64> = (0..tables.polar_count())
        .map(|i| tables.polar_angle(i))
        .collect();
    let smallest = (0..polars.len())
        .min_by(|&a, &b| polars[a].total_cmp(&polars[b]))
        .unwrap();
    let largest = (0..polars.len())
        .max_by(|&a, &b| polars[a].total_cmp(&polars[b]))
        .unwrap();
    let offsets = m2l_offsets();
    (0..OFFSETS)
        .filter(|&d| match tables.shift(d).alignment {
            host::Alignment::Up | host::Alignment::Down => true,
            host::Alignment::Rotated { polar, .. } => polar == smallest || polar == largest,
        })
        .inspect(|&d| {
            let o = offsets[d];
            let on_axis = o[0] == 0 && o[1] == 0;
            assert_eq!(
                on_axis,
                matches!(
                    tables.shift(d).alignment,
                    host::Alignment::Up | host::Alignment::Down
                ),
                "offset {d} {o:?}"
            );
        })
        .map(|d| d as u16)
        .collect()
}

/// The layouts a backend runs in these tests: on the CPU runtime its CPU layout and a cube
/// of at most (p + 1)² units, as many as the device allows (correctness only; several
/// slots per unit where it allows fewer); on a GPU its default cube, a cube of 64 more
/// units, a cube of 8 units (several slots per unit for p ≥ 3) and the CPU layout.
fn layouts(device: &Device, p: usize) -> Vec<RotationLayout> {
    let default = RotationLayout::default_for(device.info(), p);
    let nc = ((p + 1) * (p + 1)) as u32;
    match device.backend() {
        BackendKind::Cpu => vec![
            default,
            RotationLayout::Cube {
                units: nc.min(device.info().max_units_per_cube.max(1)),
            },
        ],
        _ => {
            let RotationLayout::Cube { units } = default else {
                panic!("a GPU's default layout is a cube: {default}");
            };
            vec![
                default,
                RotationLayout::Cube { units: units + 64 },
                RotationLayout::Cube { units: 8 },
                RotationLayout::Cpu,
            ]
        }
    }
}

/// The data of one level call: the tables, the view, the multipoles and the locals.
struct Call<'a, T: Real> {
    tables: &'a HostTables<T>,
    view: VView,
    multipoles: Vec<T>,
    locals: Vec<T>,
}

impl<'a, T: Real> Call<'a, T> {
    fn new(rng: &mut Rng, tables: &'a HostTables<T>, view: VView) -> Self {
        let p = tables.p();
        let nc = (p + 1) * (p + 1);
        let multipoles = uniform::<T>(rng, view.nsources * nc);
        let mut locals = uniform::<T>(rng, view.nrows() * nc);
        for r in 0..view.nrows() {
            if view.row(r).is_empty() {
                locals[r * nc..(r + 1) * nc].fill(t::<T>(0.0));
            }
        }
        Self {
            tables,
            view,
            multipoles,
            locals,
        }
    }

    fn nc(&self) -> usize {
        let p = self.tables.p();
        (p + 1) * (p + 1)
    }

    /// The host replica of the call, or with `magnitude` its terms τ.
    fn replica(&self, magnitude: bool) -> Vec<T> {
        let nc = self.nc();
        let mut out: Vec<T> = self
            .locals
            .iter()
            .map(|&v| if magnitude { t::<T>(w(v).abs()) } else { v })
            .collect();
        for r in 0..self.view.nrows() {
            for e in self.view.row(r) {
                let (d, s) = (self.view.groups[e] as usize, self.view.sources[e] as usize);
                self.tables.replica(
                    d,
                    &self.multipoles[s * nc..(s + 1) * nc],
                    &mut out[r * nc..(r + 1) * nc],
                    magnitude,
                );
            }
        }
        out
    }

    /// The call with `RotationTables::m2l` (`ShiftTables::apply`) in row order.
    fn host(&self) -> Vec<T> {
        let nc = self.nc();
        let mut out = self.locals.clone();
        let mut scratch = RotationScratch::new(self.tables.p());
        for r in 0..self.view.nrows() {
            for e in self.view.row(r) {
                let (d, s) = (self.view.groups[e] as usize, self.view.sources[e] as usize);
                self.tables.tables.apply(
                    d,
                    &self.multipoles[s * nc..(s + 1) * nc],
                    &mut out[r * nc..(r + 1) * nc],
                    &mut scratch,
                );
            }
        }
        out
    }

    /// `repeats` level calls on the device in `layout` from the same locals, checked
    /// bit-identical and one launch each; returns the locals.
    fn device(&self, device: &mut Device, layout: RotationLayout, repeats: usize) -> Vec<T> {
        let tables = RotationTables::upload(device, &self.tables.arrays()).unwrap();
        let view = GroupedView::upload(device, &self.view.arrays(), self.view.nsources).unwrap();
        let plan = RotationPlan::new(device, &self.view.arrays(), self.view.nsources).unwrap();
        let multipoles = device.upload(&self.multipoles).unwrap();
        let mut locals = device.upload(&self.locals).unwrap();
        let mut results: Vec<Vec<T>> = Vec::new();
        for _ in 0..repeats {
            device.write(locals.as_slice_mut(), &self.locals).unwrap();
            let before = device.counters().launches;
            m2l(
                device,
                layout,
                &plan,
                &view,
                &tables,
                multipoles.as_slice(),
                locals.as_slice_mut(),
            )
            .unwrap();
            assert_eq!(
                device.counters().launches - before,
                u64::from(!self.view.sources.is_empty()),
                "one launch per call with a pair"
            );
            assert_same(
                "the multipoles",
                &download(device, &multipoles),
                &self.multipoles,
            );
            results.push(download(device, &locals));
        }
        for repeat in &results[1..] {
            assert_same("a repeated level call", repeat, &results[0]);
        }
        results.swap_remove(0)
    }

    /// Asserts `got` within 2 (5p + 10) u_T τ of the host operator; returns the worst error
    /// in u_T τ and the share of values equal bit for bit.
    fn within_host(&self, what: &str, got: &[T]) -> (f64, f64) {
        let host = self.host();
        let tau = self.replica(true);
        let p = self.tables.p();
        let factor = 2.0 * (5 * p + 10) as f64 * unit::<T>();
        let mut worst = 0.0f64;
        for (i, (&g, &h)) in got.iter().zip(&host).enumerate() {
            let e = (w(g) - w(h)).abs();
            let tau = w(tau[i]);
            assert!(
                e <= factor * tau,
                "{what}, value {i}: {e:e} from the host operator > {:e}",
                factor * tau
            );
            if tau > 0.0 {
                worst = worst.max(e / (unit::<T>() * tau));
            }
        }
        let same = got
            .iter()
            .zip(&host)
            .filter(|(g, h)| g.bits() == h.bits())
            .count();
        (worst, same as f64 / got.len().max(1) as f64)
    }

    /// Asserts that every row without entries is zero.
    fn empty_rows_zero(&self, what: &str, got: &[T]) {
        let nc = self.nc();
        for r in 0..self.view.nrows() {
            if self.view.row(r).is_empty() {
                assert!(
                    got[r * nc..(r + 1) * nc]
                        .iter()
                        .all(|&v| w(v).to_bits() == 0),
                    "{what}: row {r} without entries is not zero"
                );
            }
        }
    }

    /// Every check of the module documentation on every layout; prints one line per
    /// layout.
    fn check(&self, device: &mut Device, name: &str) {
        let p = self.tables.p();
        let want = self.replica(false);
        let pairs = self.view.sources.len();
        let mut first: Option<Vec<T>> = None;
        for layout in layouts(device, p) {
            let what = format!(
                "{} rotation M2L p = {p}, {name}, {pairs} pairs, {layout}",
                T::FLOAT
            );
            let got = self.device(device, layout, 2);
            assert_same(&format!("{what}: the host replica"), &got, &want);
            self.empty_rows_zero(&what, &got);
            match &first {
                Some(first) => {
                    assert_same(&format!("{what}: against the first layout"), &got, first)
                }
                None => first = Some(got.clone()),
            }
            let (worst, same) = self.within_host(&what, &got);
            println!(
                "  {what}: bit for bit the host replica, repeated calls bit-identical, empty \
                 rows zero; ≤ {worst:.2} u_T τ from RotationTables::m2l, {:.1}% of the \
                 values bit for bit",
                100.0 * same
            );
        }
    }
}

/// Random V views at degree p.
fn check_random<T: Real>(device: &mut Device, p: usize) {
    let tables = HostTables::<T>::new(p);
    let mut rng = Rng::new(0x7_a100 + p as u64);
    let view = VView::random(&mut rng, 12, 6);
    Call::new(&mut rng, &tables, view).check(device, "random rows");
}

/// The special offsets at degree p: one row each, a row with all of them, an empty row.
fn check_special<T: Real>(device: &mut Device, p: usize) {
    let tables = HostTables::<T>::new(p);
    let special = special_offsets(&tables.tables);
    assert_eq!(
        special
            .iter()
            .filter(|&&d| {
                matches!(
                    tables.tables.shift(d as usize).alignment,
                    host::Alignment::Up | host::Alignment::Down
                )
            })
            .count(),
        4,
        "four offsets on the z axis"
    );
    let mut rows: Vec<Vec<u16>> = special.iter().map(|&d| vec![d]).collect();
    rows.push(special.clone());
    rows.push(Vec::new());
    let mut rng = Rng::new(0x7_a200 + p as u64);
    let view = VView::new(&mut rng, &rows, OFFSETS + rows.len());
    let name = format!("{} special offsets", special.len());
    Call::new(&mut rng, &tables, view).check(device, &name);
}

fn random<T: Real>(device: &mut Device) {
    for p in [0usize, 3, 8] {
        check_random::<T>(device, p);
    }
}

fn special<T: Real>(device: &mut Device) {
    for p in [1usize, 3, 8] {
        check_special::<T>(device, p);
    }
}

fn rotation_level_calls_equal_the_host_replica(device: &mut Device) {
    each_precision!(device, random);
}

fn rotation_special_offsets(device: &mut Device) {
    each_precision!(device, special);
}

/// A view without pairs launches nothing, and a layout the device cannot run is refused.
fn rotation_empty_and_refused(device: &mut Device) {
    let tables = HostTables::<f32>::new(3);
    let mut rng = Rng::new(0x7_a300);
    let view = VView::new(&mut rng, &[Vec::new(), Vec::new()], OFFSETS);
    let call = Call::new(&mut rng, &tables, view);
    for layout in layouts(device, 3) {
        assert_same(
            "an empty view",
            &call.device(device, layout, 1),
            &call.locals,
        );
    }
    let gpu_view = GroupedView::upload(device, &call.view.arrays(), OFFSETS).unwrap();
    let plan = RotationPlan::new(device, &call.view.arrays(), OFFSETS).unwrap();
    let uploaded = RotationTables::upload(device, &tables.arrays()).unwrap();
    let multipoles = device.upload(&call.multipoles).unwrap();
    let mut locals = device.upload(&call.locals).unwrap();
    let refused = m2l(
        device,
        RotationLayout::Cube { units: 0 },
        &plan,
        &gpu_view,
        &uploaded,
        multipoles.as_slice(),
        locals.as_slice_mut(),
    );
    assert!(refused.is_err(), "a cube of no unit");
    println!(
        "  f32 rotation M2L: an empty view launches nothing; {}",
        refused.unwrap_err()
    );
}

tests_on!(cpu: rotation_level_calls_equal_the_host_replica, rotation_special_offsets,
    rotation_empty_and_refused);

mod gpu {
    use super::*;

    tests_on!(metal: rotation_level_calls_equal_the_host_replica, rotation_special_offsets,
        rotation_empty_and_refused);
}
