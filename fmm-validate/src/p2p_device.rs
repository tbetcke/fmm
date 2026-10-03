//! The device P2P kernel of `nd-fmm-kernels` (Phase 4 T6, C4.2) on the workloads of
//! [`p2p_kernels`](crate::p2p_kernels), for the device rows of the `p2p_kernels`
//! example (feature `gpu`).
//!
//! The kernel takes one level's call: a near view, leaf coordinates, point offsets and
//! leaf-scaled stores ([`nd_fmm_kernels::p2p`]). [`DeviceProblem`] builds that call from
//! the host workloads:
//!
//! - **W1** ([`DeviceProblem::w1`]): `rows` target leaves, row r the set r mod 64 of the
//!   pool, so that a launch covers a level-sized number of leaves (at least 4,096 on a
//!   GPU, device-path.md §13.4) while the data stay those of the pool. A W1 set is a
//!   target leaf of half-width 1 at the origin and its 26 neighbours of the same level,
//!   centred at 2 (a, b, c): in leaf-scaled coordinates the targets and the centre leaf's
//!   sources are the points themselves, a neighbour's are u_s = y − 2 (a, b, c), exact,
//!   and the kernel maps them back with ĉ = 2 (a, b, c), r̂ = 1, giving y bit for bit.
//!   - *per-pair*: each row holds the 27 leaves of its set in the set's order, the centre
//!     leaf as the row's own leaf (no mapping), as `LaplaceOperator` passes a near row;
//!   - *gathered*: each row holds one source leaf, the set's 27 n_t sources at the
//!     row's own key (identity frame, mapped by fma(1, y, 0) = y).
//! - **W2** ([`DeviceProblem::w2`]): the N targets in rows of `chunk` targets, each row
//!   one source leaf holding every source, at the same key (identity frame).
//!
//! Every pair of the host set is a pair of the device problem with the same values, so
//! the oracle of the set ([`Oracle`]) measures the device output; pairs per launch count
//! coincident pairs, as on the host. Timings queue launches between syncs
//! ([`DeviceProblem::seconds_per_launch`]), compilation excluded.
//!
//! [`seconds_per_evaluation`] times `nd_fmm_simd` on the same leaves on one thread or on
//! several (scoped threads, each its own contiguous share of the leaves), for the
//! comparison.

use std::time::Instant;

use nd_fmm_kernels::p2p::{P2pInputs, P2pLayout, p2p};
use nd_fmm_kernels::view::{IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{Device, DeviceBuffer, DeviceFloat, KernelError};
use nd_fmm_math::RealScalar;
use nd_fmm_simd::SimdScalar;

use crate::bench::median_time_per_call;
use crate::p2p_kernels::{Accuracy, Form, Kernel, Oracle, Outputs, Set, accuracy};

/// The index of the centre leaf among the 27 leaves of a W1 set.
const CENTRE: usize = 13;

/// The key of every target leaf of W1: level 2, index (1, 1, 1), so that its 26
/// neighbours have valid indices.
const W1_KEY: (u32, [u32; 3]) = (2, [1, 1, 1]);

/// The offset (a, b, c) ∈ {−1, 0, 1}³ of leaf j of a W1 set (the order of
/// `p2p_kernels::w1`).
fn w1_offset(j: usize) -> [i64; 3] {
    [j / 9, (j / 3) % 3, j % 3].map(|o| o as i64 - 1)
}

/// One leaf of a device problem on the host.
#[derive(Clone, Debug)]
struct HostLeaf<T> {
    key: (u32, [u32; 3]),
    sources: Vec<[T; 3]>,
    charges: Vec<T>,
    targets: Vec<[T; 3]>,
}

/// A P2P call on the device: the near view, the leaves and the stores, with the output
/// zeroed at upload.
pub struct DeviceProblem<T: DeviceFloat> {
    near: IndexView,
    leaves: LeafCoordinates,
    source_offsets: PointOffsets,
    target_offsets: PointOffsets,
    sources: DeviceBuffer<T>,
    target_input: DeviceBuffer<T>,
    output: DeviceBuffer<T>,
    /// The targets of each row, in row order.
    targets: Vec<usize>,
    gradients: bool,
    pairs: usize,
}

impl<T: DeviceFloat + RealScalar> DeviceProblem<T> {
    /// Uploads `leaves`, of which the first `rows.len()` are the rows' targets.
    fn upload(
        device: &mut Device,
        leaves: &[HostLeaf<T>],
        rows: &[Vec<u32>],
        gradients: bool,
    ) -> Result<Self, KernelError> {
        let scan = |count: &dyn Fn(&HostLeaf<T>) -> usize| -> Vec<u32> {
            std::iter::once(0)
                .chain(leaves.iter().scan(0u32, |total, leaf| {
                    *total += count(leaf) as u32;
                    Some(*total)
                }))
                .collect()
        };
        let (so, to) = (scan(&|l| l.sources.len()), scan(&|l| l.targets.len()));
        let mut sources = Vec::with_capacity(4 * so[so.len() - 1] as usize);
        let mut input = Vec::with_capacity(3 * to[to.len() - 1] as usize);
        for leaf in leaves {
            sources.extend(leaf.sources.iter().flatten());
            sources.extend(&leaf.charges);
            input.extend(leaf.targets.iter().flatten());
        }
        let row_offsets: Vec<u32> = std::iter::once(0)
            .chain(rows.iter().scan(0u32, |total, row| {
                *total += row.len() as u32;
                Some(*total)
            }))
            .collect();
        let entries: Vec<u32> = rows.iter().flatten().copied().collect();
        let keys: Vec<(u32, [u32; 3])> = leaves.iter().map(|l| l.key).collect();
        let per_point = if gradients { 4 } else { 1 };
        let targets: Vec<usize> = leaves[..rows.len()]
            .iter()
            .map(|l| l.targets.len())
            .collect();
        let pairs = rows
            .iter()
            .zip(&targets)
            .map(|(row, &n_t)| {
                n_t * row
                    .iter()
                    .map(|&j| leaves[j as usize].sources.len())
                    .sum::<usize>()
            })
            .sum();
        Ok(Self {
            near: IndexView::upload(device, &row_offsets, &entries, leaves.len())?,
            leaves: LeafCoordinates::upload(device, &keys)?,
            source_offsets: PointOffsets::upload(device, &so)?,
            target_offsets: PointOffsets::upload(device, &to)?,
            sources: device.upload(&sources)?,
            target_input: device.upload(&input)?,
            output: device.alloc(per_point * input.len() / 3)?,
            targets,
            gradients,
            pairs,
        })
    }

    /// W1 of `pool` in `form`: `rows` target leaves, row r the set r mod `pool.len()`
    /// (module documentation).
    ///
    /// # Errors
    ///
    /// As `Device::upload`.
    ///
    /// # Panics
    ///
    /// If a neighbour's source minus its centre is not exact (it is for the W1 inputs).
    pub fn w1(
        device: &mut Device,
        pool: &[Set<T>],
        form: Form,
        rows: usize,
        gradients: bool,
    ) -> Result<Self, KernelError> {
        let mut leaves: Vec<HostLeaf<T>> = (0..rows)
            .map(|r| {
                let set = &pool[r % pool.len()];
                let (sources, charges) = match form {
                    Form::PerPair => (
                        set.leaves[CENTRE].sources.clone(),
                        set.leaves[CENTRE].charges.clone(),
                    ),
                    Form::Gathered => (Vec::new(), Vec::new()),
                };
                HostLeaf {
                    key: W1_KEY,
                    sources,
                    charges,
                    targets: set.targets.clone(),
                }
            })
            .collect();
        // The source leaves of each set: per-pair, the 26 neighbours in leaf-scaled
        // coordinates; gathered, one leaf of every source.
        let mut set_leaves: Vec<Vec<u32>> = Vec::with_capacity(pool.len());
        for set in pool {
            let mut indices = Vec::new();
            match form {
                Form::PerPair => {
                    for (j, leaf) in set.leaves.iter().enumerate() {
                        if j == CENTRE {
                            indices.push(u32::MAX);
                            continue;
                        }
                        let offset = w1_offset(j);
                        let centre = offset.map(|o| <T as RealScalar>::from_f64(2.0 * o as f64));
                        let sources = leaf
                            .sources
                            .iter()
                            .map(|y| {
                                std::array::from_fn(|k| {
                                    let u = y[k] - centre[k];
                                    assert_eq!(
                                        RealScalar::to_f64(centre[k] + u),
                                        RealScalar::to_f64(y[k]),
                                        "u_s = y − ĉ must be exact"
                                    );
                                    u
                                })
                            })
                            .collect();
                        indices.push(leaves.len() as u32);
                        leaves.push(HostLeaf {
                            key: (2, offset.map(|o| (1 + o) as u32)),
                            sources,
                            charges: leaf.charges.clone(),
                            targets: Vec::new(),
                        });
                    }
                }
                Form::Gathered => {
                    indices.push(leaves.len() as u32);
                    leaves.push(HostLeaf {
                        key: W1_KEY,
                        sources: set.sources.clone(),
                        charges: set.charges.clone(),
                        targets: Vec::new(),
                    });
                }
            }
            set_leaves.push(indices);
        }
        let near: Vec<Vec<u32>> = (0..rows)
            .map(|r| {
                set_leaves[r % pool.len()]
                    .iter()
                    .map(|&j| if j == u32::MAX { r as u32 } else { j })
                    .collect()
            })
            .collect();
        Self::upload(device, &leaves, &near, gradients)
    }

    /// W2 of `set` (one W2 set): its targets in rows of `chunk`, every row the one leaf of
    /// all sources (module documentation).
    ///
    /// # Errors
    ///
    /// As `Device::upload`.
    pub fn w2(
        device: &mut Device,
        set: &Set<T>,
        chunk: usize,
        gradients: bool,
    ) -> Result<Self, KernelError> {
        let key = (0, [0; 3]);
        let mut leaves: Vec<HostLeaf<T>> = set
            .targets
            .chunks(chunk.max(1))
            .map(|targets| HostLeaf {
                key,
                sources: Vec::new(),
                charges: Vec::new(),
                targets: targets.to_vec(),
            })
            .collect();
        let rows = leaves.len();
        leaves.push(HostLeaf {
            key,
            sources: set.sources.clone(),
            charges: set.charges.clone(),
            targets: Vec::new(),
        });
        let near = vec![vec![rows as u32]; rows];
        Self::upload(device, &leaves, &near, gradients)
    }

    /// The pairs of one launch, n_s n_t per row, coincident ones included.
    pub fn pairs(&self) -> usize {
        self.pairs
    }

    /// The number of rows (target leaves) of a launch.
    pub fn rows(&self) -> usize {
        self.targets.len()
    }

    /// One launch in `layout`, adding into the output.
    ///
    /// # Errors
    ///
    /// As `nd_fmm_kernels::p2p::p2p`.
    pub fn launch(&mut self, device: &mut Device, layout: P2pLayout) -> Result<(), KernelError> {
        let inputs = P2pInputs {
            near: &self.near,
            first_leaf: 0,
            leaves: &self.leaves,
            source_offsets: &self.source_offsets,
            sources: self.sources.as_slice(),
            target_offsets: &self.target_offsets,
            target_input: self.target_input.as_slice(),
        };
        p2p(
            device,
            layout,
            self.gradients,
            &inputs,
            self.output.as_slice_mut(),
        )
    }

    /// The outputs of the first targets: zeroes the output, launches once and returns φ
    /// and ∇φ of rows `rows`, in target order.
    ///
    /// # Errors
    ///
    /// As `Device::download`.
    pub fn outputs(
        &mut self,
        device: &mut Device,
        layout: P2pLayout,
        rows: std::ops::Range<usize>,
    ) -> Result<Outputs<T>, KernelError> {
        nd_fmm_kernels::movement::zero(device, self.output.as_slice_mut())?;
        self.launch(device, layout)?;
        let mut all = vec![<T as RealScalar>::from_f64(0.0); self.output.len()];
        device.download(self.output.as_slice(), &mut all)?;
        let per_point = if self.gradients { 4 } else { 1 };
        let mut out = Outputs::new(0);
        let mut start = 0;
        for (r, &n) in self.targets.iter().enumerate() {
            if rows.contains(&r) {
                let chunk = &all[per_point * start..per_point * (start + n)];
                out.potential.extend(&chunk[..n]);
                if self.gradients {
                    out.gradient.extend(chunk[n..].as_chunks::<3>().0);
                } else {
                    out.gradient.extend(std::iter::repeat_n(
                        [<T as RealScalar>::from_f64(0.0); 3],
                        n,
                    ));
                }
            }
            start += n;
        }
        Ok(out)
    }

    /// The accuracy of `layout` against `oracles[i]`, the oracle of rows `rows(i)`
    /// (truncated to the oracle's targets).
    ///
    /// # Errors
    ///
    /// As [`outputs`](Self::outputs).
    pub fn accuracy(
        &mut self,
        device: &mut Device,
        layout: P2pLayout,
        oracles: &[(Oracle, std::ops::Range<usize>)],
    ) -> Result<Accuracy, KernelError> {
        let mut worst = Accuracy::default();
        for (oracle, rows) in oracles {
            let out = self.outputs(device, layout, rows.clone())?;
            let n = oracle.potential.len();
            let gradient = self.gradients.then_some(&out.gradient[..n]);
            worst = worst.worst(accuracy(oracle, &out.potential[..n], gradient));
        }
        Ok(worst)
    }

    /// Seconds per launch in `layout`: a warm-up launch and a sync first (compilation
    /// excluded), then the median over batches of launches queued between syncs
    /// ([`median_time_per_call`]).
    ///
    /// # Errors
    ///
    /// As `nd_fmm_kernels::p2p::p2p` and `Device::sync`.
    pub fn seconds_per_launch(
        &mut self,
        device: &mut Device,
        layout: P2pLayout,
    ) -> Result<f64, KernelError> {
        self.launch(device, layout)?;
        device.sync()?;
        let mut failure = None;
        let seconds = median_time_per_call(|calls| {
            let start = Instant::now();
            for _ in 0..calls {
                if let Err(error) = self.launch(device, layout) {
                    failure.get_or_insert(error);
                }
            }
            if let Err(error) = device.sync() {
                failure.get_or_insert(error);
            }
            start.elapsed()
        });
        match failure {
            Some(error) => Err(error),
            None => Ok(seconds),
        }
    }
}

/// Seconds per evaluation of `leaves` W1 leaves (the pool cycled, leaf i the set
/// i mod `pool.len()`) by `kernel` in `form`, on `threads` threads: one thread runs on
/// the caller; more are scoped threads, each a contiguous share of the leaves, spawned
/// once per batch (the median over batches, [`median_time_per_call`]).
pub fn seconds_per_evaluation<T: SimdScalar>(
    kernel: &Kernel<T>,
    pool: &[Set<T>],
    form: Form,
    gradients: bool,
    leaves: usize,
    threads: usize,
) -> f64 {
    let mut outputs: Vec<Outputs<T>> = (0..leaves)
        .map(|i| Outputs::new(pool[i % pool.len()].targets.len()))
        .collect();
    let share = leaves.div_ceil(threads.max(1)).max(1);
    median_time_per_call(|calls| {
        let start = Instant::now();
        if threads <= 1 {
            for _ in 0..calls {
                for (i, out) in outputs.iter_mut().enumerate() {
                    kernel.evaluate(&pool[i % pool.len()], form, gradients, out);
                }
            }
        } else {
            std::thread::scope(|scope| {
                for (part, chunk) in outputs.chunks_mut(share).enumerate() {
                    scope.spawn(move || {
                        for _ in 0..calls {
                            for (k, out) in chunk.iter_mut().enumerate() {
                                let i = part * share + k;
                                kernel.evaluate(&pool[i % pool.len()], form, gradients, out);
                            }
                        }
                    });
                }
            });
        }
        start.elapsed()
    })
}

/// Seconds per evaluation of the W2 set `set` (one call over all pairs) by `kernel`, on
/// `threads` threads: more than one splits the targets into contiguous shares, one
/// scoped thread each, spawned once per batch ([`median_time_per_call`]).
pub fn seconds_per_all_pairs<T: SimdScalar>(
    kernel: &Kernel<T>,
    set: &Set<T>,
    gradients: bool,
    threads: usize,
) -> f64 {
    let mut out = Outputs::new(set.targets.len());
    let share = set.targets.len().div_ceil(threads.max(1)).max(1);
    median_time_per_call(|calls| {
        let start = Instant::now();
        let (potential, gradient) = (&mut out.potential, &mut out.gradient);
        std::thread::scope(|scope| {
            let parts = set
                .targets
                .chunks(share)
                .zip(potential.chunks_mut(share))
                .zip(gradient.chunks_mut(share));
            for ((targets, phi), grad) in parts {
                let mut work = move || {
                    for _ in 0..calls {
                        kernel.call(
                            &set.sources,
                            &set.charges,
                            targets,
                            phi,
                            gradients.then_some(&mut grad[..]),
                        );
                    }
                };
                if threads <= 1 {
                    work();
                } else {
                    scope.spawn(work);
                }
            }
        });
        start.elapsed()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w1_offsets_follow_the_set_order() {
        assert_eq!(w1_offset(0), [-1, -1, -1]);
        assert_eq!(w1_offset(CENTRE), [0, 0, 0]);
        assert_eq!(w1_offset(26), [1, 1, 1]);
        assert_eq!(w1_offset(5), [-1, 0, 1]);
    }
}
