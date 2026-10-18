//! Phase 6 T8d: the coincident-pair rule of the production dipole P2P
//! (`nd_fmm_kernels::p2p::p2p` with `SourceKind::Dipoles`, the backend's default layout) on
//! the adversarial and random pairs of the §3.13 check (`pairs`), each with a seeded moment
//! d̂ uniform in [−1, 1)³ (CONVENTIONS §3.13, "Device kernels").
//!
//! Each pair is one row of one launch: the target leaf t with u_t, and the source leaf s
//! with u_s and d̂ (the leaf itself for s = t), so the kernel forms ŷ = fma(r̂, u_s, ĉ) and
//! r̂ d̂ from the leaves' integer coordinates. Checked per pair, against the host's inputs
//! (ŷ = fl(ĉ + r̂ u_s), d_t = r̂ d̂) and `nd_fmm_ref::p2p::p2p_dipoles` in T:
//! - a coincident pair (u_t == ŷ, which the reference skips) adds exactly zero to φ̂ and ĝ;
//! - every other pair in the f32 dipole range (r² ≥ 2⁻⁷⁹ for |d_t,ₖ| ≤ 2; every pair in
//!   f64) is finite and within the device dipole contract, 32 u_T (φ̂, relative to
//!   |d_t| / r²) and 176 u_T (each component of ĝ, relative to |d_t| / r³).

use nd_fmm_kernels::p2p::{P2pInputs, P2pLayout, p2p};
use nd_fmm_kernels::view::{IndexView, LeafCoordinates, PointOffsets};
use nd_fmm_kernels::{Device, DeviceFloat, SourceKind};
use nd_fmm_validate::SplitMix64;

use crate::pairs::Pair;
use crate::real::Real;

/// The outcome of one pair set.
#[derive(Clone, Copy, Debug, Default)]
pub struct Outcome {
    /// Pairs.
    pub pairs: usize,
    /// Coincident pairs.
    pub coincident: usize,
    /// Coincident pairs with a nonzero term (the rule fails).
    pub coincident_nonzero: usize,
    /// Non-coincident pairs in range with a non-finite term.
    pub nonfinite: usize,
    /// Non-coincident f32 pairs below the dipole range, not checked.
    pub below_range: usize,
    /// The largest potential error, u_T relative to |d_t| / r².
    pub potential: f64,
    /// The largest gradient error, u_T relative to |d_t| / r³.
    pub gradient: f64,
}

impl Outcome {
    /// Whether the rule and the contract hold.
    pub fn holds(&self) -> bool {
        self.coincident_nonzero == 0
            && self.nonfinite == 0
            && self.potential <= 32.0
            && self.gradient <= 176.0
    }
}

/// Runs `pairs` through the production dipole kernel on `device` in `layout` and checks
/// them (module documentation).
pub fn run<T: Real + DeviceFloat>(
    device: &mut Device,
    layout: P2pLayout,
    pairs: &[Pair<T>],
    seed: u64,
) -> Outcome {
    let mut rng = SplitMix64::new(seed);
    let moments: Vec<[T; 3]> = pairs
        .iter()
        .map(|_| [0; 3].map(|_| T::narrow(2.0 * rng.uniform() - 1.0)))
        .collect();
    let rows = pairs.len();
    // The target leaves (one per row) first, then the source leaves of the other pairs.
    let mut keys: Vec<(u32, [u32; 3])> = pairs
        .iter()
        .map(|p| (p.keys.1.level, p.keys.1.index))
        .collect();
    let mut entries = Vec::with_capacity(rows);
    let mut source_counts = vec![0u32; rows];
    let mut targets = Vec::with_capacity(3 * rows);
    for (r, p) in pairs.iter().enumerate() {
        targets.extend(p.ut);
        if p.self_pair {
            source_counts[r] = 1;
            entries.push(r as u32);
        } else {
            entries.push(keys.len() as u32);
            keys.push((p.keys.0.level, p.keys.0.index));
            source_counts.push(1);
        }
    }
    // The source store: every pair's one source, in its leaf, u_s and then d̂.
    let mut store = vec![T::narrow(0.0); 6 * rows];
    let mut offsets = vec![0u32];
    for &n in &source_counts {
        offsets.push(offsets.last().unwrap() + n);
    }
    let mut extra = rows;
    for (r, p) in pairs.iter().enumerate() {
        let leaf = if p.self_pair {
            r
        } else {
            extra += 1;
            extra - 1
        };
        let at = 6 * offsets[leaf] as usize;
        store[at..at + 3].copy_from_slice(&p.us);
        store[at + 3..at + 6].copy_from_slice(&moments[r]);
    }
    let mut target_offsets: Vec<u32> = (0..=rows as u32).collect();
    target_offsets.resize(keys.len() + 1, rows as u32);
    let row_offsets: Vec<u32> = (0..=rows as u32).collect();
    let near = IndexView::upload(device, &row_offsets, &entries, keys.len()).unwrap();
    let leaves = LeafCoordinates::upload(device, &keys).unwrap();
    let source_offsets = PointOffsets::upload(device, &offsets).unwrap();
    let target_offsets = PointOffsets::upload(device, &target_offsets).unwrap();
    let sources = device.upload(&store).unwrap();
    let target_input = device.upload(&targets).unwrap();
    let mut output = device.alloc::<T>(4 * rows).unwrap();
    let inputs = P2pInputs {
        near: &near,
        first_leaf: 0,
        leaves: &leaves,
        source_offsets: &source_offsets,
        sources: sources.as_slice(),
        target_offsets: &target_offsets,
        target_input: target_input.as_slice(),
        vectors: 1,
        source_kind: SourceKind::Dipoles,
    };
    p2p(device, layout, true, &inputs, output.as_slice_mut()).unwrap();
    let mut out = vec![T::narrow(0.0); output.len()];
    device.download(output.as_slice(), &mut out).unwrap();
    // The outputs: per target leaf of one target, φ̂ and then ĝ.
    let floor = if T::BITS == 24 { 2f64.powi(-79) } else { 0.0 };
    let mut o = Outcome {
        pairs: rows,
        ..Outcome::default()
    };
    for (r, p) in pairs.iter().enumerate() {
        let (phi, grad) = (out[4 * r], [out[4 * r + 1], out[4 * r + 2], out[4 * r + 3]]);
        if p.coincident() {
            o.coincident += 1;
            if phi.widen() != 0.0 || grad.iter().any(|g| g.widen() != 0.0) {
                o.coincident_nonzero += 1;
            }
            continue;
        }
        let y = p.y();
        let d: [T; 3] = if p.self_pair {
            moments[r]
        } else {
            moments[r].map(|c| p.r * c)
        };
        let r2: f64 = (0..3)
            .map(|k| (p.ut[k].widen() - y[k].widen()).powi(2))
            .sum();
        if r2 < floor {
            o.below_range += 1;
            continue;
        }
        if !phi.widen().is_finite() || grad.iter().any(|g| !g.widen().is_finite()) {
            o.nonfinite += 1;
            continue;
        }
        let (mut rphi, mut rg) = ([T::narrow(0.0)], [[T::narrow(0.0); 3]]);
        nd_fmm_ref::p2p::p2p_dipoles(&[y], &[d], &[p.ut], &mut rphi, Some(&mut rg));
        let dn = d.iter().map(|c| c.widen().powi(2)).sum::<f64>().sqrt();
        if dn == 0.0 {
            continue;
        }
        o.potential = o
            .potential
            .max((phi.widen() - rphi[0].widen()).abs() / (dn / r2) / T::U);
        for k in 0..3 {
            o.gradient = o
                .gradient
                .max((grad[k].widen() - rg[0][k].widen()).abs() / (dn / (r2 * r2.sqrt())) / T::U);
        }
    }
    o
}
