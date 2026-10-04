//! Dense M2L as a grouped translation (T9, C4.5): level calls over V-list-shaped views of
//! the 316 offsets, on every hand-written layout the backend runs, in f32 and, where the
//! device supports it, f64; on Metal also the library GEMM (CMMA) in f32 at p = 8.
//!
//! The views: rows are boxes with a random subset of the 316 offsets in index order (one
//! source per offset, sources distinct within a row); every seventh row has no entry, as a
//! ghost or a box without a V list. The input (multipoles) and output (locals) are separate
//! buffers (`Operands::Separate`), as the device operator passes them; rows with entries
//! start from random locals, rows without from zero.
//!
//! - **Batches**: a level call equals the host emulation of each target's V row in
//!   offset order (load, add each entry's fma-loop product in row order, store) bit for
//!   bit with the hand-written GEMM, with one chunk and with chunks that cut batches; so
//!   each target gets every entry of its row once. Within the GEMM tolerance of T8 of
//!   applying the rows with `MatrixSet::apply` in row order ((n + 9) u_T relative to the
//!   terms). Rows without entries stay zero, bit for bit.
//! - **Grouped against per offset**: a level call in structure (B) equals structure (A)
//!   (`per_group`: one gather, GEMM and scatter-add per offset in index order), bit for bit
//!   with the same GEMM (device-path.md §6.4): every hand-written layout and, on Metal, the
//!   library.
//! - **Library** (Metal, f32, p = 8): the plan's GEMM and chunks are reported; the library
//!   and the hand-written GEMM agree within the GEMM tolerance on the same batches
//!   (whether bit for bit is printed); the library's level call is within the GEMM
//!   tolerance of `MatrixSet::apply` in row order, also in several chunks; repeated level
//!   calls are bit-identical with both GEMMs (the library strategy is fixed); structure
//!   (A) with the library equals (B) bit for bit where the library takes every offset's
//!   shape. On the CPU runtime the plan must take the hand-written kernel.

use nd_fmm_kernels::translate::{
    Accumulate, DEFAULT_SCRATCH_BYTES, Gemm, GemmLayout, GemmPolicy, GroupedPlan, Operands,
    PerGroupPlan, PlanSettings, Tables, TranslationScratch, grouped, per_group,
};
use nd_fmm_kernels::view::{GroupedArrays, GroupedView};
use nd_fmm_kernels::{BackendKind, Device, Precision};
use nd_fmm_tables::MatrixSet;

use super::{Real, assert_same, download, host_product, layouts, t, terms, uniform, unit, w};
use crate::common::{Rng, tests_on};

/// The V-list offsets of `nd_fmm_tables::geometry::m2l_offsets`.
const OFFSETS: usize = 316;

/// A V-list-shaped view (module documentation): the six arrays of `GroupedArrays` with
/// `u16` offset indices.
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
    /// `nrows` boxes, each (but every seventh) with each offset with probability 1 in
    /// `sparsity`, sources among `nsources` (at least 316), distinct within a row.
    fn random(rng: &mut Rng, nrows: usize, nsources: usize, sparsity: usize) -> Self {
        let mut rows: Vec<Vec<(u16, u32)>> = Vec::with_capacity(nrows);
        for r in 0..nrows {
            let mut row = Vec::new();
            if r % 7 != 6 {
                // Distinct sources: a random rotation of the sources by offset.
                let shift = rng.below(nsources);
                for d in 0..OFFSETS {
                    if rng.below(sparsity) == 0 {
                        row.push((d as u16, ((shift + 7 * d) % nsources) as u32));
                    }
                }
            }
            rows.push(row);
        }
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

    fn widest(&self) -> usize {
        self.batch_offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0)
    }
}

/// The data of an M2L level call: 316 tables, the view, the multipoles and the locals.
struct M2lCall<T: Real> {
    set: MatrixSet<T>,
    view: VView,
    multipoles: Vec<T>,
    locals: Vec<T>,
}

impl<T: Real> M2lCall<T> {
    fn new(seed: u64, p: usize, nrows: usize, sparsity: usize) -> Self {
        let n = (p + 1) * (p + 1);
        let mut rng = Rng::new(seed);
        let mut set = MatrixSet::<T>::zeros(n, OFFSETS);
        for d in 0..OFFSETS {
            set.matrix_mut(d)
                .copy_from_slice(&uniform::<T>(&mut rng, n * n));
        }
        let nsources = OFFSETS + nrows;
        let view = VView::random(&mut rng, nrows, nsources, sparsity);
        let multipoles = uniform::<T>(&mut rng, nsources * n);
        let mut locals = uniform::<T>(&mut rng, nrows * n);
        for r in 0..nrows {
            if view.row(r).is_empty() {
                locals[r * n..(r + 1) * n].fill(t::<T>(0.0));
            }
        }
        Self {
            set,
            view,
            multipoles,
            locals,
        }
    }

    fn n(&self) -> usize {
        self.set.n()
    }

    /// The host emulation of a level call (module documentation).
    fn host_rows(&self) -> Vec<T> {
        let n = self.n();
        let mut out = self.locals.clone();
        for r in 0..self.view.nrows() {
            for e in self.view.row(r) {
                let (d, s) = (self.view.groups[e] as usize, self.view.sources[e] as usize);
                let y = host_product(self.set.matrix(d), &self.multipoles[s * n..(s + 1) * n], n);
                for i in 0..n {
                    out[r * n + i] += y[i];
                }
            }
        }
        out
    }

    /// The rows applied with `MatrixSet::apply` in row order, and the terms of each output.
    fn apply_rows(&self) -> (Vec<T>, Vec<f64>) {
        let n = self.n();
        let mut out = self.locals.clone();
        let mut tau: Vec<f64> = self.locals.iter().map(|&v| w(v).abs()).collect();
        for r in 0..self.view.nrows() {
            for e in self.view.row(r) {
                let (d, s) = (self.view.groups[e] as usize, self.view.sources[e] as usize);
                let x = &self.multipoles[s * n..(s + 1) * n];
                self.set.apply(d, x, &mut out[r * n..(r + 1) * n]);
                for (i, v) in terms(self.set.matrix(d), x, n).into_iter().enumerate() {
                    tau[r * n + i] += v;
                }
            }
        }
        (out, tau)
    }

    /// Runs structure (B) with `settings` (`repeats` level calls from the same locals,
    /// checked bit-identical) and returns the locals, the plan's GEMM and its chunks.
    fn grouped(
        &self,
        device: &mut Device,
        settings: &PlanSettings,
        repeats: usize,
    ) -> (Vec<T>, Gemm, usize) {
        let n = self.n();
        let library = settings.library_candidate(device.backend(), T::FLOAT);
        let tables = Tables::upload(device, self.set.as_slice(), n, library).unwrap();
        let view = GroupedView::upload(device, &self.view.arrays(), self.view.nsources).unwrap();
        let size = settings.size(device.backend(), T::FLOAT, &self.view.batch_offsets);
        let mut scratch = TranslationScratch::<T>::new(device, size.columns * n).unwrap();
        let plan = GroupedPlan::new(
            device,
            &self.view.arrays(),
            self.view.nsources,
            settings,
            &tables,
            &mut scratch,
        )
        .unwrap();
        assert!(plan.columns() <= size.columns);
        if let Some(reason) = plan.library_rejection() {
            println!("    library not used, the hand-written kernel runs: {reason}");
        }
        let multipoles = device.upload(&self.multipoles).unwrap();
        let mut locals = device.upload(&self.locals).unwrap();
        let mut results: Vec<Vec<T>> = Vec::new();
        for _ in 0..repeats {
            device.write(locals.as_slice_mut(), &self.locals).unwrap();
            let before = device.counters().launches;
            grouped(
                device,
                &plan,
                &view,
                Accumulate::Rows,
                &tables,
                Operands::Separate {
                    input: multipoles.as_slice(),
                    output: locals.as_slice_mut(),
                },
                &mut scratch,
            )
            .unwrap();
            assert_eq!(
                device.counters().launches - before,
                3 * plan.nchunks() as u64,
                "three launches per chunk"
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
        (results.swap_remove(0), plan.gemm(), plan.nchunks())
    }

    /// Runs structure (A) with `gemm` and returns the locals, or why the plan could not be
    /// built (the library rejects some narrow per-offset shapes).
    fn try_per_group(&self, device: &mut Device, gemm: Gemm) -> Result<Vec<T>, String> {
        let n = self.n();
        let tables = Tables::upload(device, self.set.as_slice(), n, gemm == Gemm::Library).unwrap();
        let view = GroupedView::upload(device, &self.view.arrays(), self.view.nsources).unwrap();
        let mut scratch =
            TranslationScratch::<T>::new(device, self.view.widest().max(1) * n).unwrap();
        let plan = PerGroupPlan::new(
            device,
            &self.view.arrays(),
            self.view.nsources,
            n,
            gemm,
            &tables,
            &mut scratch,
        )
        .map_err(|e| e.to_string())?;
        let multipoles = device.upload(&self.multipoles).unwrap();
        let mut locals = device.upload(&self.locals).unwrap();
        let before = device.counters().launches;
        per_group(
            device,
            &plan,
            &view,
            &tables,
            Operands::Separate {
                input: multipoles.as_slice(),
                output: locals.as_slice_mut(),
            },
            &mut scratch,
        )
        .unwrap();
        assert_eq!(
            device.counters().launches - before,
            3 * plan.nonempty_groups() as u64,
            "three launches per offset with a pair"
        );
        Ok(download(device, &locals))
    }

    /// Runs structure (A) with `gemm` and returns the locals.
    fn per_group(&self, device: &mut Device, gemm: Gemm) -> Vec<T> {
        self.try_per_group(device, gemm).unwrap()
    }

    /// Asserts `got` within (n + 9) u_T τ of `apply` (the GEMM tolerance of T8) and
    /// returns the worst error in u_T τ.
    fn within_apply(&self, what: &str, got: &[T], apply: &[T], tau: &[f64]) -> f64 {
        let n = self.n();
        let mut worst = 0.0f64;
        for (i, (&a, &tau)) in apply.iter().zip(tau).enumerate() {
            let e = (w(got[i]) - w(a)).abs();
            let bound = (n + 9) as f64 * unit::<T>() * tau;
            assert!(e <= bound, "{what}, value {i}: {e:e} > {bound:e}");
            if tau > 0.0 {
                worst = worst.max(e / (unit::<T>() * tau));
            }
        }
        worst
    }

    /// Asserts that every row without entries is zero.
    fn empty_rows_zero(&self, what: &str, got: &[T]) {
        let n = self.n();
        for r in 0..self.view.nrows() {
            if self.view.row(r).is_empty() {
                assert!(
                    got[r * n..(r + 1) * n].iter().all(|&v| w(v).to_bits() == 0),
                    "{what}: row {r} without entries is not zero"
                );
            }
        }
    }
}

/// The batch and chunk checks of the module documentation at degree p, every layout.
fn check_batches<T: Real>(device: &mut Device, p: usize) {
    let n = (p + 1) * (p + 1);
    let call = M2lCall::<T>::new(0x7_9100 + p as u64, p, 24, 4);
    let want = call.host_rows();
    let (apply, tau) = call.apply_rows();
    let pairs = call.view.sources.len();
    for layout in layouts(device, n) {
        let what = format!("{} M2L p = {p}, {pairs} pairs, {layout}", T::FLOAT);
        let mut chunks = Vec::new();
        // One chunk; chunks of 97 columns, which cut batches.
        for columns in [usize::MAX, 97] {
            let budget = if columns == usize::MAX {
                DEFAULT_SCRATCH_BYTES
            } else {
                (2 * n * columns * size_of::<T>()) as u64
            };
            let settings = PlanSettings {
                n,
                layout,
                policy: GemmPolicy::HandWritten,
                budget,
            };
            let repeats = if columns == usize::MAX { 2 } else { 1 };
            let (got, gemm, nchunks) = call.grouped(device, &settings, repeats);
            assert_eq!(gemm, Gemm::HandWritten(layout));
            chunks.push(nchunks);
            assert_same(
                &format!("{what}, {nchunks} chunks: the host rows"),
                &got,
                &want,
            );
            call.empty_rows_zero(&what, &got);
        }
        assert_eq!(chunks, vec![1, pairs.div_ceil(97)]);
        let worst = call.within_apply(&what, &want, &apply, &tau);
        println!(
            "  {what}: bit for bit the host rows in 1 and {} chunks, each entry once, \
             repeated calls bit-identical, empty rows zero; ≤ {worst:.2} u_T τ from apply",
            chunks[1]
        );
    }
}

/// Structure (B) against (A) at degree p, every hand-written layout, bit for bit.
fn check_per_offset<T: Real>(device: &mut Device, p: usize) {
    let n = (p + 1) * (p + 1);
    let call = M2lCall::<T>::new(0x7_9200 + p as u64, p, 24, 3);
    for layout in layouts(device, n) {
        let settings = PlanSettings {
            n,
            layout,
            policy: GemmPolicy::HandWritten,
            budget: DEFAULT_SCRATCH_BYTES,
        };
        let (b, _, _) = call.grouped(device, &settings, 1);
        let a = call.per_group(device, Gemm::HandWritten(layout));
        assert_same(
            &format!(
                "{} M2L p = {p}, {layout}: grouped against per offset",
                T::FLOAT
            ),
            &b,
            &a,
        );
        println!(
            "  {} M2L p = {p}, {layout}: structure (B) equals (A) (gather, GEMM and \
             scatter-add per offset in index order) bit for bit",
            T::FLOAT
        );
    }
}

fn batches<T: Real>(device: &mut Device) {
    for p in [0usize, 3, 8] {
        check_batches::<T>(device, p);
    }
}

fn per_offset<T: Real>(device: &mut Device) {
    for p in [0usize, 3, 8] {
        check_per_offset::<T>(device, p);
    }
}

fn m2l_level_calls_equal_the_host_rows(device: &mut Device) {
    each_precision!(device, batches);
}

fn m2l_grouped_equals_per_offset(device: &mut Device) {
    each_precision!(device, per_offset);
}

/// The library at p = 8 in f32 (module documentation). On the CPU runtime the plan under
/// `GemmPolicy::Auto` must take the hand-written kernel.
fn m2l_library(device: &mut Device) {
    let p = 8;
    let n = (p + 1) * (p + 1);
    let layout = GemmLayout::default_for(device.info(), n);
    let (nrows, sparsity) = if device.backend().is_gpu() {
        (480, 2)
    } else {
        (24, 4)
    };
    let call = M2lCall::<f32>::new(0x7_9300, p, nrows, sparsity);
    let pairs = call.view.sources.len();
    let (apply, tau) = call.apply_rows();
    let auto = |budget| PlanSettings {
        n,
        layout,
        policy: GemmPolicy::Auto,
        budget,
    };
    let (library, gemm, chunks) = call.grouped(device, &auto(DEFAULT_SCRATCH_BYTES), 3);
    if device.backend() == BackendKind::Cpu {
        assert_eq!(
            gemm,
            Gemm::HandWritten(layout),
            "the CPU runtime has no CMMA"
        );
        println!("  f32 M2L p = 8, {pairs} pairs: the plan takes {gemm} on the CPU runtime");
        return;
    }
    let worst = call.within_apply("library M2L p = 8", &library, &apply, &tau);
    println!(
        "  f32 M2L p = 8, {pairs} pairs: GEMM {gemm} in {chunks} chunk(s); ≤ {worst:.2} u_T τ \
         from apply; three level calls bit-identical"
    );
    if gemm != Gemm::Library {
        println!("  the library did not take the shape: the hand-written kernel ran instead");
        return;
    }
    // The hand-written GEMM on the same batches, repeated.
    let hand_settings = PlanSettings {
        policy: GemmPolicy::HandWritten,
        ..auto(DEFAULT_SCRATCH_BYTES)
    };
    let (hand, _, _) = call.grouped(device, &hand_settings, 3);
    let (mut d2, mut r2, mut worst_pair) = (0.0f64, 0.0f64, 0.0f64);
    for (i, (&a, &b)) in library.iter().zip(&hand).enumerate() {
        let e = (w(a) - w(b)).abs();
        let bound = 2.0 * (n + 9) as f64 * unit::<f32>() * tau[i];
        assert!(
            e <= bound,
            "library against hand-written, value {i}: {e:e} > {bound:e}"
        );
        if tau[i] > 0.0 {
            worst_pair = worst_pair.max(e / (unit::<f32>() * tau[i]));
        }
        d2 += (w(a) - w(b)).powi(2);
        r2 += w(b).powi(2);
    }
    let same = library
        .iter()
        .zip(&hand)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "  f32 M2L p = 8: library against hand-written ≤ {worst_pair:.2} u_T τ, relative L2 \
         {:.2e} ({}); the hand-written calls bit-identical",
        (d2 / r2).sqrt(),
        if same {
            "bit for bit"
        } else {
            "not bit for bit"
        }
    );
    // The library in several chunks (at most 1,000 columns each).
    let small = (2 * n * 1000 * size_of::<f32>()) as u64;
    let (several, gemm_small, chunks_small) = call.grouped(device, &auto(small), 2);
    assert_eq!(gemm_small, Gemm::Library);
    let worst_small = call.within_apply("library M2L p = 8, chunks", &several, &apply, &tau);
    let same = several
        .iter()
        .zip(&library)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    println!(
        "  f32 M2L p = 8: the library in {chunks_small} chunks ≤ {worst_small:.2} u_T τ from \
         apply; {} the one-chunk call",
        if same {
            "bit for bit"
        } else {
            "not bit for bit (other padded shapes)"
        }
    );
    // Structure (A) with the library, one launch per offset: bit for bit the grouped call
    // where the library takes every offset's shape.
    match call.try_per_group(device, Gemm::Library) {
        Ok(a) => {
            assert_same(
                "library M2L p = 8: grouped against per offset",
                &library,
                &a,
            );
            println!("  f32 M2L p = 8: structure (B) equals (A) with the library, bit for bit");
        }
        Err(reason) => {
            println!("  f32 M2L p = 8: (A) with the library not built ({reason}); not compared")
        }
    }
}

tests_on!(cpu: m2l_level_calls_equal_the_host_rows, m2l_grouped_equals_per_offset, m2l_library);

mod gpu {
    use super::*;

    tests_on!(metal: m2l_level_calls_equal_the_host_rows, m2l_grouped_equals_per_offset,
        m2l_library);
}
