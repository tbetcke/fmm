//! The level M2L of `Batched` (Phase 6 T2) and `Rotation`, as `nd-fmm-exec` runs them on
//! one rank at one thread: `LaplaceKernels::m2l_block` (blocks of `M2L_BLOCK_TARGETS` = 64
//! consecutive targets, for each the offsets in index order, one column-kernel call per
//! offset) and `m2l_target` (target by target, each source's `RotationTables::m2l`).

use nd_fmm_simd::{Isa, MatrixKernel, MulAdd};
use nd_fmm_tables::{M2lTables, RotationScratch, RotationTables};

use crate::geometry::{Level, Pair};

/// Targets per block of the batched M2L (`nd_fmm_exec::operator::M2L_BLOCK_TARGETS`).
pub const BLOCK: usize = 64;

/// The V pairs grouped by offset index, each group by target: (targets, sources).
pub struct Batches {
    groups: Vec<(Vec<usize>, Vec<usize>)>,
}

impl Batches {
    /// Groups `pairs` (by target, then source) by offset index.
    pub fn new(pairs: &[Pair]) -> Self {
        let mut groups = vec![(Vec::new(), Vec::new()); 316];
        for pair in pairs {
            groups[pair.offset].0.push(pair.target);
            groups[pair.offset].1.push(pair.source);
        }
        Self { groups }
    }
}

/// `Batched`'s level M2L: adds the M2L of `multipoles` into `locals`.
pub struct BatchedM2l {
    tables: M2lTables<f64>,
    kernel: MatrixKernel<f64>,
    pack: Vec<f64>,
    sources: Vec<usize>,
    targets: Vec<usize>,
}

impl BatchedM2l {
    /// The tables of degree `p` and the column kernel of `isa` (the machine's,
    /// `Isa::detect`, if `None`) in `mode`.
    ///
    /// # Panics
    ///
    /// If the machine lacks `isa`.
    pub fn new(p: usize, mode: MulAdd, isa: Option<Isa>) -> Self {
        let n = (p + 1) * (p + 1);
        let kernel = match isa {
            Some(isa) => MatrixKernel::new(isa, mode).expect("the ISA is available"),
            None => MatrixKernel::detect(mode),
        };
        Self {
            tables: M2lTables::build(p),
            kernel,
            pack: vec![0.0; MatrixKernel::<f64>::pack_len(n)],
            sources: Vec::with_capacity(BLOCK),
            targets: Vec::with_capacity(BLOCK),
        }
    }

    /// The kernel's ISA.
    pub fn kernel(&self) -> MatrixKernel<f64> {
        self.kernel
    }

    /// The level call.
    pub fn apply(
        &mut self,
        level: &Level,
        batches: &Batches,
        multipoles: &[f64],
        locals: &mut [f64],
    ) {
        let n = self.tables.matrices().n();
        for (block, block_locals) in locals.chunks_mut(BLOCK * n).enumerate() {
            let first = block * BLOCK;
            let end = (first + BLOCK).min(level.boxes());
            for (d, (targets, sources)) in batches.groups.iter().enumerate() {
                let lo = targets.partition_point(|&t| t < first);
                let hi = lo + targets[lo..].partition_point(|&t| t < end);
                if lo == hi {
                    continue;
                }
                self.sources.clear();
                self.targets.clear();
                for (&t, &s) in targets[lo..hi].iter().zip(&sources[lo..hi]) {
                    self.sources.push(s * n);
                    self.targets.push((t - first) * n);
                }
                self.kernel.accumulate_columns(
                    n,
                    self.tables.matrices().matrix(d),
                    multipoles,
                    &self.sources,
                    block_locals,
                    &self.targets,
                    &mut self.pack,
                );
            }
        }
    }
}

/// `Rotation`'s level M2L.
pub struct RotationM2l {
    tables: RotationTables<f64>,
    scratch: RotationScratch<f64>,
}

impl RotationM2l {
    /// The rotation tables of degree `p`.
    pub fn new(p: usize) -> Self {
        Self {
            tables: RotationTables::build(p),
            scratch: RotationScratch::new(p),
        }
    }

    /// The level call: target by target, each target's sources in row order.
    pub fn apply(&mut self, pairs: &[Pair], multipoles: &[f64], locals: &mut [f64]) {
        let n = self.tables.p() + 1;
        let n = n * n;
        for pair in pairs {
            let (s, t) = (pair.source, pair.target);
            self.tables.m2l(
                pair.offset,
                &multipoles[s * n..(s + 1) * n],
                &mut locals[t * n..(t + 1) * n],
                &mut self.scratch,
            );
        }
    }
}
