//! The 34 reflection classes of the 316 V-list offsets (proposal §5.4).
//!
//! The reflections x → −x, y → −y, z → −z and the swap x ↔ y map the signature grid onto
//! itself ([`crate::grid`]), so the transfer function of an offset o is that of its
//! class representative read through a permutation of the samples:
//! T_o(ŝ) = Σ cₗ Pₗ(ŝ · ô) = Σ cₗ Pₗ(Sŝ · Sô) = T_rep(Sŝ) for the symmetry S with
//! S o = rep. The representative is (min(|a|, |b|), max(|a|, |b|), |c|) of o = (a, b, c),
//! and S flips the negative components, then swaps x and y if |a| > |b|.

use nd_fmm_spike_truncation::geometry::{dist2, offsets};

/// One class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Class {
    /// The representative (0 ≤ x ≤ y, z ≥ 0).
    pub rep: [i32; 3],
    /// |t|² in box widths.
    pub d2: i32,
    /// Number of offsets in the class.
    pub size: usize,
}

/// The symmetry that maps an offset onto its representative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Symmetry {
    /// Flip x, y, z first.
    pub flip: [bool; 3],
    /// Then swap x and y.
    pub swap: bool,
}

impl Symmetry {
    /// S x.
    pub fn apply(&self, x: [i32; 3]) -> [i32; 3] {
        let mut y = x;
        for d in 0..3 {
            if self.flip[d] {
                y[d] = -y[d];
            }
        }
        if self.swap {
            y.swap(0, 1);
        }
        y
    }

    /// The sample (i′, j′) of the direction S ŝ for sample (i, j) of a grid with `rings`
    /// rings and `nphi` azimuths (a multiple of 4).
    pub fn sample(&self, rings: usize, nphi: usize, i: usize, j: usize) -> (usize, usize) {
        let (mut i, mut j) = (i, j);
        // x → −x: φ → π − φ; y → −y: φ → −φ; z → −z: θ → π − θ; x ↔ y: φ → π/2 − φ.
        if self.flip[0] {
            j = (nphi / 2 + nphi - j) % nphi;
        }
        if self.flip[1] {
            j = (nphi - j) % nphi;
        }
        if self.flip[2] {
            i = rings - 1 - i;
        }
        if self.swap {
            j = (nphi / 4 + nphi - j) % nphi;
        }
        (i, j)
    }
}

/// The representative of o and the symmetry that maps o onto it.
pub fn class_of(o: [i32; 3]) -> ([i32; 3], Symmetry) {
    let flip = [o[0] < 0, o[1] < 0, o[2] < 0];
    let swap = o[0].abs() > o[1].abs();
    let s = Symmetry { flip, swap };
    (s.apply(o), s)
}

/// The 34 classes, ordered by |t|² and then by the representative.
pub fn classes() -> Vec<Class> {
    let mut out: Vec<Class> = Vec::new();
    for o in offsets() {
        let (rep, _) = class_of(o);
        match out.iter_mut().find(|c| c.rep == rep) {
            Some(c) => c.size += 1,
            None => out.push(Class {
                rep,
                d2: dist2(rep),
                size: 1,
            }),
        }
    }
    out.sort_by_key(|c| (c.d2, c.rep));
    out
}
