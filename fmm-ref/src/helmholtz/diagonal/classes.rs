//! The 34 reflection classes of the 316 M2L offsets (proposal §5.4; S6's `classes.rs`).

/// One reflection class of the M2L offsets: the offsets that the reflections x → −x,
/// y → −y, z → −z and the swap x ↔ y map onto its representative.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Class {
    /// The representative, 0 ≤ x ≤ y and z ≥ 0, in box widths.
    pub representative: [i32; 3],
    /// |t|² in box widths squared.
    pub squared_distance: i32,
    /// The number of offsets in the class.
    pub size: usize,
}

/// The symmetry S that maps an offset o onto its class representative: it flips the
/// negative components of o, then swaps x and y if |o_x| > |o_y|.
///
/// The transfer function of o is the representative's read through S:
/// T_o(ŝ) = Σ cₗ Pₗ(ŝ · ô) = Σ cₗ Pₗ(Sŝ · Sô) = T_rep(Sŝ), and S maps the samples of a
/// [`Grid`](super::Grid) onto samples ([`Symmetry::sample`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Symmetry {
    /// Flip x, y and z first.
    pub flip: [bool; 3],
    /// Then swap x and y.
    pub swap: bool,
}

impl Symmetry {
    /// S x.
    pub fn apply<V: Copy + core::ops::Neg<Output = V>>(&self, x: [V; 3]) -> [V; 3] {
        let mut y = x;
        for (v, &f) in y.iter_mut().zip(&self.flip) {
            if f {
                *v = -*v;
            }
        }
        if self.swap {
            y.swap(0, 1);
        }
        y
    }

    /// S⁻¹ x: the swap, then the flips.
    pub fn apply_inverse<V: Copy + core::ops::Neg<Output = V>>(&self, x: [V; 3]) -> [V; 3] {
        let mut y = x;
        if self.swap {
            y.swap(0, 1);
        }
        for (v, &f) in y.iter_mut().zip(&self.flip) {
            if f {
                *v = -*v;
            }
        }
        y
    }

    /// The sample (i′, j′) of the direction S ŝ for the sample (i, j) of a grid with `rings`
    /// rings and `azimuths` azimuths (a multiple of 4): x → −x maps j to (N_φ/2 − j) mod N_φ,
    /// y → −y maps j to −j mod N_φ, z → −z maps ring i to `rings` − 1 − i, x ↔ y maps j to
    /// (N_φ/4 − j) mod N_φ, applied in that order.
    ///
    /// # Panics
    ///
    /// If `azimuths` is not a multiple of 4.
    pub fn sample(&self, rings: usize, azimuths: usize, i: usize, j: usize) -> (usize, usize) {
        assert!(
            azimuths.is_multiple_of(4),
            "the azimuth count must be a multiple of 4"
        );
        let (mut i, mut j) = (i, j);
        if self.flip[0] {
            j = (azimuths / 2 + azimuths - j) % azimuths;
        }
        if self.flip[1] {
            j = (azimuths - j) % azimuths;
        }
        if self.flip[2] {
            i = rings - 1 - i;
        }
        if self.swap {
            j = (azimuths / 4 + azimuths - j) % azimuths;
        }
        (i, j)
    }
}

/// The 316 offsets of the M2L (the V list of one level, in box widths): every o with
/// max |oᵢ| ∈ {2, 3}, in lexicographic order.
pub fn offsets() -> Vec<[i32; 3]> {
    let mut out = Vec::with_capacity(316);
    for i in -3i32..=3 {
        for j in -3i32..=3 {
            for k in -3i32..=3 {
                if i.abs().max(j.abs()).max(k.abs()) >= 2 {
                    out.push([i, j, k]);
                }
            }
        }
    }
    out
}

/// The representative of the offset o, (min(|o_x|, |o_y|), max(|o_x|, |o_y|), |o_z|), and
/// the symmetry that maps o onto it.
///
/// ```
/// use nd_fmm_ref::helmholtz::diagonal::class_of;
///
/// let (rep, s) = class_of([-3, 1, -2]);
/// assert_eq!(rep, [1, 3, 2]);
/// assert_eq!(s.apply([-3, 1, -2]), rep);
/// assert_eq!(s.apply_inverse(rep), [-3, 1, -2]);
/// ```
pub fn class_of(o: [i32; 3]) -> ([i32; 3], Symmetry) {
    let s = Symmetry {
        flip: [o[0] < 0, o[1] < 0, o[2] < 0],
        swap: o[0].abs() > o[1].abs(),
    };
    (s.apply(o), s)
}

/// The 34 classes of the 316 offsets, ordered by |t|² and then by the representative.
///
/// ```
/// use nd_fmm_ref::helmholtz::diagonal::classes;
///
/// let classes = classes();
/// assert_eq!(classes.len(), 34);
/// assert_eq!(classes.iter().map(|c| c.size).sum::<usize>(), 316);
/// assert_eq!(classes[0].representative, [0, 0, 2]);
/// ```
pub fn classes() -> Vec<Class> {
    let mut out: Vec<Class> = Vec::new();
    for o in offsets() {
        let (rep, _) = class_of(o);
        match out.iter_mut().find(|c| c.representative == rep) {
            Some(c) => c.size += 1,
            None => out.push(Class {
                representative: rep,
                squared_distance: rep.iter().map(|x| x * x).sum(),
                size: 1,
            }),
        }
    }
    out.sort_by_key(|c| (c.squared_distance, c.representative));
    out
}
