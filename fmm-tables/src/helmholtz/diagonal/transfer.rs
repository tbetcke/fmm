//! A level's transfer functions of the diagonal form, one per offset class, and the pair step
//! (proposal §5.4, "Diagonal translation" and "Reflection classes"; CONVENTIONS §3.15
//! "Signatures"; Helmholtz CH4.2a).
//!
//! # What is stored
//!
//! [`DiagonalM2l`] holds, for one level in its own units (`HelmholtzTables`' units: the
//! wavenumber k̂_l, boxes of half-width 1, the shift of a V offset d being t = 2d), the
//! transfer functions of the offset classes a caller names, each once, at the samples of the
//! grid of degree p (CONVENTIONS §3.15, "The grid"). Sample (i, j) of class c holds
//!
//! ```text
//! W_c(ŝᵢⱼ) = 4π ik̂ e^{2β − Im(k̂)|t_c|} · T̂_c(ŝᵢⱼ)
//! ```
//!
//! with T̂_c the stored transfer function of the class representative (by either build of
//! "Two builds" below, in f64) and the scalar factor of the diagonal step, formed in f64 as
//! [`HelmholtzTables`](crate::helmholtz::HelmholtzTables) forms the RCR M2L's and multiplied in
//! f64; the product is rounded to `T` (an f32 table is the f64 table rounded). The factor
//! depends on |t| alone, which is the class's, so it is folded in as the RCR tables fold it,
//! and the pair is one complex multiply-add per sample. The series of T̂ is never summed in the
//! working precision (proposal §3.6; the interpolated build sums S6's interpolation of
//! double-double values in f64, the variant S6 measured and CH0.6 signed off). This storage is the crate's: CONVENTIONS §3.15 leaves how T̂ is
//! computed and stored to CH4.2 ("Not fixed here"). Classes that run RCR store nothing
//! (proposal §5.4).
//!
//! The classes are `nd_fmm_ref::helmholtz::diagonal::classes()`'s 34, by their index there
//! (ordered by |t|², then by the representative).
//!
//! # Two builds
//!
//! T̂_c comes from `nd-fmm-ref` in one of two ways ([`TransferBuild`]; proposal §5.4, "per
//! sample in O(p³) per class at build time, or interpolated in ŝ · t̂ from an
//! extended-precision table in O(p²) per class"):
//!
//! - **direct** (CH4.2a): each sample is ref's `Transfer`, the series summed in double-double
//!   at the sample's ŝ · t̂ and rounded once;
//! - **interpolated** (CH4.2c1, S6's `Interp`): T̂ depends on ŝ through ψ = arccos(ŝ · t̂)
//!   alone and is a cosine polynomial of degree L = 2p in ψ, so ref's `AngleTable` holds it at
//!   M = [`interpolation_steps`]`(p)` = 3L + 1 steps of [0, π] (each value double-double, rounded
//!   once), and each sample reads it by the Gaussian-regularised sinc series of Qian (2003),
//!   [`INTERPOLATION_TERMS`] terms on each side of its position, which ref computes in
//!   double-double; the series is summed in f64. Its rounding is S6's 0.3–6× that of the direct
//!   build (median 1.2), and the bound E holds with C_T = 0.23 in place of 0.15
//!   ([`TransferBuild::bound_constants`]; signed off at CH0.6, question 3). The table costs
//!   O(M p) double-double operations per class, a sample O(1) (72 f64 terms with an `exp`
//!   each), against O(p) double-double operations per sample directly.
//!
//! Both fold the same factor in the same way. The interpolated tables call libm's `atan2`,
//! `sin` and `exp`, so they are a function of the machine's libm as well as of p, k̂ and the
//! class: the same on every thread and rank of a run, not bit for bit between libms.
//!
//! # The build in parts
//!
//! [`DiagonalM2l::build_by`] cuts the build into parts of one ring of one class (Helmholtz
//! CH4.2c1; one part per class before), (2p + 1) parts per class, and the interpolated build
//! first into one part per class for its angle tables: on a node's threads a level's build
//! then costs about its classes' work over the threads, not one class's (CH4.2b2's report §5,
//! 2.3 s at p = 131 on Kathleen at 1 × 40). Each part is a function of the build's inputs and
//! its number, and the parts are put together in (class, ring) order, so every map gives the
//! serial build's bits; ref's `Transfer::ring_samples` is its `Transfer::new` ring for ring bit
//! for bit, so the direct build is CH4.2a's tables bit for bit.
//!
//! # Reading a class through an offset's symmetry
//!
//! The symmetry S of an offset o maps it onto its class representative, and
//! T_o(ŝ) = T_rep(S ŝ); S maps the grid's samples onto samples (CONVENTIONS §3.15,
//! "Classes"): z → −z maps ring i to 2p − i, and x → −x, y → −y and x ↔ y map the azimuth j
//! to N_φ/2 − j, −j and N_φ/4 − j (mod N_φ), applied in that order. Each azimuth map is a
//! reflection j ↦ a − j, so their composition is j ↦ (start + j) or (start − j) mod N_φ. An
//! offset's [`OffsetMap`] holds its class, the ring flip and that affine map, derived from
//! `Symmetry::sample` and checked against it on every sample of the grid of degree 3 and 8
//! (tests). The pair reads each target ring's samples from one ring of the class's in at most
//! two contiguous runs, forward or reversed: no table per offset, no index per sample.
//!
//! # The pair and its order
//!
//! [`DiagonalM2l::pair`] adds W_c(S ŝ) F(ŝ) to a target's accumulator at every sample:
//! `g = g + w · f`, the complex product (`num-complex`'s, each real product and sum rounded
//! apart, never fused) then the sum. A sample's chain is its accumulator and the pairs of its
//! target's row in row order, so a target's bits depend on its row alone, as
//! [`m2l_level`](super::m2l_level) requires (proposal §2, constraint 2). The level's RCR pairs
//! of the same target are added after F → R, in row order (proposal §5.4, step 3): the
//! caller's (nd-fmm-exec, CH4.2b), tested here at the table level.
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_tables::geometry::m2l_offset_index;
//! use nd_fmm_tables::helmholtz::diagonal::DiagonalM2l;
//!
//! // Level units: k̂ = 6, classes 10 to 33 (the farther ones) stored.
//! let p = 6;
//! let classes: Vec<usize> = (10..34).collect();
//! let m2l = DiagonalM2l::<f64>::build(p, Complex::new(6.0, 0.0), &classes);
//! let d = m2l_offset_index([3, -3, 2]).unwrap();
//! assert!(m2l.has_offset(d));
//! let signature = vec![Complex::new(1.0, 0.0); m2l.samples()];
//! let mut accumulator = vec![Complex::new(0.0, 0.0); m2l.samples()];
//! m2l.pair(d, &signature, &mut accumulator);
//! let mut w = vec![Complex::new(0.0, 0.0); m2l.samples()];
//! m2l.offset_samples(d, &mut w);
//! assert_eq!(accumulator, w);
//! ```

use nd_fmm_math::{Complex, RealScalar};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_ref::helmholtz::diagonal::{
    AngleTable, BoundConstants, Grid, Transfer, azimuth_count, bound, class_of, classes,
};

use crate::geometry::{M2L_OFFSET_COUNT, m2l_offsets};
use crate::parts::{Parts, Serial};

/// The number of reflection classes of the 316 V offsets
/// (`nd_fmm_ref::helmholtz::diagonal::classes`).
pub const CLASS_COUNT: usize = 34;

/// How a V offset reads its class's transfer function (module documentation, "Reading a
/// class through an offset's symmetry"): the target's sample (i, j) reads the class's sample
/// (i′, j′) with i′ = 2p − i if `flip_rings` (else i) and j′ = (`start` − j) mod N_φ if
/// `reverse` (else (`start` + j) mod N_φ).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OffsetMap {
    /// The class, an index into `nd_fmm_ref::helmholtz::diagonal::classes()`.
    pub class: usize,
    /// Whether ring i reads ring 2p − i (the symmetry flips z).
    pub flip_rings: bool,
    /// The class's azimuth read at j = 0.
    pub start: usize,
    /// Whether the azimuth map reverses the direction (an odd number of azimuth reflections).
    pub reverse: bool,
}

/// One level's transfer functions of the diagonal form, for the offset classes a caller names,
/// with the diagonal step's factor folded in (module documentation). Immutable once built.
#[derive(Clone, Debug)]
pub struct DiagonalM2l<T> {
    p: usize,
    k_hat: Complex<f64>,
    rings: usize,
    azimuths: usize,
    /// Per class its position among the stored ones, or `None`.
    stored: [Option<usize>; CLASS_COUNT],
    /// The stored classes' samples, class after class in class order.
    values: Vec<Complex<T>>,
    /// Per V offset (by table index, `m2l_offsets`) its map.
    maps: Vec<OffsetMap>,
}

/// The maps of the 316 offsets at the grid of degree p, by table index.
fn offset_maps(p: usize) -> Vec<OffsetMap> {
    let rings = 2 * p + 1;
    let azimuths = azimuth_count(p);
    let classes = classes();
    m2l_offsets()
        .iter()
        .map(|d| {
            let o = d.map(|x| x as i32);
            let (rep, symmetry) = class_of(o);
            let class = classes
                .iter()
                .position(|c| c.representative == rep)
                .expect("every V offset lies in a class");
            let (i0, start) = symmetry.sample(rings, azimuths, 0, 0);
            let (_, next) = symmetry.sample(rings, azimuths, 0, 1);
            let reverse = next != (start + 1) % azimuths;
            debug_assert!(!reverse || next == (start + azimuths - 1) % azimuths);
            OffsetMap {
                class,
                flip_rings: i0 != 0,
                start,
                reverse,
            }
        })
        .collect()
}

/// The level's constants in its units: `Level::new(k̂, 1)`.
fn level(k_hat: Complex<f64>) -> Level<f64> {
    Level::new(k_hat, 1.0)
}

/// |t| = 2 |rep| of a class in the level's units, formed as `HelmholtzTables` forms the RCR
/// M2L's distance.
fn distance(class: usize) -> f64 {
    let squared = classes()[class].squared_distance;
    2.0 * f64::from(squared).sqrt()
}

/// 4π ik e^{2β − Im(k)|t|} in f64, as `HelmholtzTables` forms the RCR M2L's factor.
fn factor(level: &Level<f64>, t: f64) -> Complex<f64> {
    let k = level.k();
    let beta = level.damping();
    let c = 4.0 * core::f64::consts::PI;
    Complex::new(-(c * k.im), c * k.re) * (beta + beta - k.im * t).exp()
}

/// The frames of class c's representative in the level's units: from the origin to 2 rep.
fn class_frames(class: usize) -> (Frame<f64>, Frame<f64>) {
    let rep = classes()[class].representative;
    (
        Frame::new([0.0; 3], 1.0),
        Frame::new(rep.map(|x| 2.0 * f64::from(x)), 1.0),
    )
}

/// How [`DiagonalM2l`] builds its transfer functions T̂ (proposal §5.4; module documentation,
/// "Two builds"). Both compute in double-double what they take from `nd-fmm-ref` and round it
/// once; neither sums T̂ in the working precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferBuild {
    /// Each sample from ref's `Transfer`: O(p) double-double operations per sample, O(p³) per
    /// class (Helmholtz CH4.2a). The rule's constants are S6's for T rounded per sample.
    #[default]
    Direct,
    /// Each sample interpolated in ψ = arccos(ŝ · t̂) from ref's `AngleTable` (Helmholtz
    /// CH4.2c1; S6's `Interp`): O(p²) double-double operations per class for the table, and per
    /// sample [`INTERPOLATION_TERMS`] · 2 terms in f64 at a position from double-double. The
    /// rule's constants are S6's for T interpolated (C_T = 0.23).
    Interpolated,
}

impl TransferBuild {
    /// The constants of the bound E for tables of this build (S6, signed off at CH0.6): in f64
    /// `BoundConstants::F64` (direct) or `F64_INTERPOLATED` (interpolated); for a 4-byte table
    /// (`single`) `F32` either way, since an f32 table is the f64 table rounded and the
    /// interpolation's error in f64 lies about 2²⁹ times below f32's rounding.
    pub fn bound_constants(self, single: bool) -> BoundConstants {
        match (single, self) {
            (true, _) => BoundConstants::F32,
            (false, Self::Direct) => BoundConstants::F64,
            (false, Self::Interpolated) => BoundConstants::F64_INTERPOLATED,
        }
    }
}

/// The interpolated build's oversampling σ: the angle grid has M = σ L + 1 steps of [0, π] for
/// the bandwidth L = 2p (S6's 3).
pub const INTERPOLATION_OVERSAMPLING: usize = 3;

/// The interpolated build's terms on each side of a sample (S6's 36).
pub const INTERPOLATION_TERMS: usize = 36;

/// The steps M of the interpolated build's angle grid at degree p: σ · 2p + 1.
pub const fn interpolation_steps(p: usize) -> usize {
    INTERPOLATION_OVERSAMPLING * 2 * p + 1
}

/// T̂ read from an angle table by S6's Gaussian-regularised sinc series (Qian 2003): 2h terms of
/// the table's even, 2π-periodic extension, the window e^{−d²/(2r²)} with
/// r² = (h − 2)/(π − δ), δ = Lπ/M the band in radians per step, in f64.
struct Interpolator<'a> {
    values: &'a [Complex<f64>],
    steps: usize,
    inv_two_r2: f64,
}

impl<'a> Interpolator<'a> {
    fn new(table: &'a AngleTable) -> Self {
        let steps = table.steps();
        let band = (2 * table.degree()) as f64 * core::f64::consts::PI / steps as f64;
        let r2 = (INTERPOLATION_TERMS as f64 - 2.0).max(1.0) / (core::f64::consts::PI - band);
        Self {
            values: table.values(),
            steps,
            inv_two_r2: 0.5 / r2,
        }
    }

    /// Entry k of the even, 2π-periodic extension of the table.
    fn at(&self, k: isize) -> Complex<f64> {
        let period = 2 * self.steps as isize;
        let mut k = k.rem_euclid(period);
        if k > self.steps as isize {
            k = period - k;
        }
        self.values[k as usize]
    }

    /// T̂ at the position n + f on the grid (0 ≤ f < 1).
    fn eval(&self, n: isize, f: f64) -> Complex<f64> {
        if f == 0.0 {
            return self.at(n);
        }
        // sin(π(f − d)) = (−1)^d sin(πf); for f near 1 as sin(π(1 − f)), 1 − f exact, since
        // πf rounds next to π and sin would lose its relative accuracy there.
        let g = if f > 0.5 { 1.0 - f } else { f };
        let s = (core::f64::consts::PI * g).sin() / core::f64::consts::PI;
        let half = INTERPOLATION_TERMS as isize;
        let mut acc = Complex::new(0.0, 0.0);
        for d in 1 - half..=half {
            let dist = f - d as f64;
            let sign = if d % 2 == 0 { 1.0 } else { -1.0 };
            let w = sign * s / dist * (-dist * dist * self.inv_two_r2).exp();
            acc += self.at(n + d) * w;
        }
        acc
    }
}

/// T̂ of ring `ring` read from class `table` (its positions on `grid`).
fn interpolated_ring(grid: &Grid<f64>, table: &AngleTable, ring: usize) -> Vec<Complex<f64>> {
    let interpolator = Interpolator::new(table);
    (0..grid.azimuths())
        .map(|j| {
            let (n, f) = table.position(grid, ring, j);
            interpolator.eval(n, f)
        })
        .collect()
}

impl<T: RealScalar> DiagonalM2l<T> {
    /// The transfer functions of degree p for the wavenumber `k_hat` in the level's units, of
    /// the classes `stored` (indices into `nd_fmm_ref::helmholtz::diagonal::classes()`,
    /// ascending), built directly ([`TransferBuild::Direct`]) on the calling thread
    /// ([`build_by`](Self::build_by)).
    ///
    /// # Panics
    ///
    /// As [`build_by`](Self::build_by).
    pub fn build(p: usize, k_hat: Complex<f64>, stored: &[usize]) -> Self {
        Self::build_by(p, k_hat, stored, TransferBuild::Direct, &Serial)
    }

    /// As [`build`](Self::build), on `parts` ([`build_by`](Self::build_by) with
    /// [`TransferBuild::Direct`]).
    ///
    /// # Panics
    ///
    /// As [`build_by`](Self::build_by).
    pub fn build_on(p: usize, k_hat: Complex<f64>, stored: &[usize], parts: &impl Parts) -> Self {
        Self::build_by(p, k_hat, stored, TransferBuild::Direct, parts)
    }

    /// The transfer functions of the classes `stored` by `build`, on `parts` (`crate::parts`;
    /// module documentation, "Two builds" and "The build in parts"): under
    /// [`TransferBuild::Interpolated`] first one part per stored class (its angle table), then
    /// under both one part per (class, ring), each a function of p, k̂, the class and the ring
    /// alone, put together in (class, ring) order, so the tables are the serial build's bit for
    /// bit on any map.
    ///
    /// # Panics
    ///
    /// If `stored` is not ascending without repeats or names a class beyond
    /// [`CLASS_COUNT`], or k̂ = 0.
    pub fn build_by(
        p: usize,
        k_hat: Complex<f64>,
        stored: &[usize],
        build: TransferBuild,
        parts: &impl Parts,
    ) -> Self {
        assert!(k_hat.norm() > 0.0, "the diagonal form needs k ≠ 0");
        assert!(
            stored.windows(2).all(|w| w[0] < w[1]),
            "the classes must ascend without repeats: {stored:?}"
        );
        assert!(
            stored.iter().all(|&c| c < CLASS_COUNT),
            "a class beyond {CLASS_COUNT}: {stored:?}"
        );
        let level = level(k_hat);
        let grid = Grid::<f64>::new(p);
        let rings = grid.rings();
        let tables: Vec<AngleTable> = match build {
            TransferBuild::Direct => Vec::new(),
            TransferBuild::Interpolated => parts.map(stored.len(), &|i| {
                let (from, to) = class_frames(stored[i]);
                AngleTable::new(p, &level, &from, &to, interpolation_steps(p))
            }),
        };
        let ring_parts = parts.map(stored.len() * rings, &|part| {
            let (i, ring) = (part / rings, part % rings);
            let class = stored[i];
            let samples = match build {
                TransferBuild::Direct => {
                    let (from, to) = class_frames(class);
                    Transfer::ring_samples(&grid, &level, &from, &to, ring)
                }
                TransferBuild::Interpolated => interpolated_ring(&grid, &tables[i], ring),
            };
            let f = factor(&level, distance(class));
            samples
                .into_iter()
                .map(|s| {
                    let w = s * f;
                    Complex::new(T::from_f64(w.re), T::from_f64(w.im))
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(
            ring_parts.len(),
            stored.len() * rings,
            "the map gives one result per part"
        );
        let mut positions = [None; CLASS_COUNT];
        for (i, &c) in stored.iter().enumerate() {
            positions[c] = Some(i);
        }
        Self {
            p,
            k_hat,
            rings,
            azimuths: grid.azimuths(),
            stored: positions,
            values: ring_parts.concat(),
            maps: offset_maps(p),
        }
    }

    /// The bytes of the tables of degree p with `classes` classes stored, without building
    /// them ([`bytes`](Self::bytes)).
    pub fn bytes_for(p: usize, classes: usize) -> usize {
        classes * (2 * p + 1) * azimuth_count(p) * size_of::<Complex<T>>()
            + M2L_OFFSET_COUNT * size_of::<OffsetMap>()
    }

    /// The bytes the tables hold: the stored classes' samples and the offsets' maps.
    pub fn bytes(&self) -> usize {
        self.values.len() * size_of::<Complex<T>>() + self.maps.len() * size_of::<OffsetMap>()
    }

    /// The degree p.
    pub fn degree(&self) -> usize {
        self.p
    }

    /// The wavenumber k̂ in the level's units.
    pub fn k_hat(&self) -> Complex<f64> {
        self.k_hat
    }

    /// The samples of a signature, (2p + 1) N_φ.
    pub fn samples(&self) -> usize {
        self.rings * self.azimuths
    }

    /// Whether class c's transfer function is stored.
    pub fn is_stored(&self, class: usize) -> bool {
        self.stored.get(class).is_some_and(Option::is_some)
    }

    /// The stored classes, ascending.
    pub fn stored_classes(&self) -> Vec<usize> {
        (0..CLASS_COUNT).filter(|&c| self.is_stored(c)).collect()
    }

    /// The map of the V offset of table index `offset` (`m2l_offsets`).
    ///
    /// # Panics
    ///
    /// If `offset` is not below 316.
    pub fn offset_map(&self, offset: usize) -> OffsetMap {
        self.maps[offset]
    }

    /// Whether the V offset of table index `offset` has its class stored.
    ///
    /// # Panics
    ///
    /// If `offset` is not below 316.
    pub fn has_offset(&self, offset: usize) -> bool {
        self.is_stored(self.maps[offset].class)
    }

    /// Class c's stored samples W_c, at the grid's indices i N_φ + j, or `None`.
    pub fn class_samples(&self, class: usize) -> Option<&[Complex<T>]> {
        let n = self.samples();
        self.stored
            .get(class)
            .copied()
            .flatten()
            .map(|i| &self.values[i * n..(i + 1) * n])
    }

    fn stored_class(&self, offset: usize) -> (OffsetMap, &[Complex<T>]) {
        let map = self.maps[offset];
        let w = self.class_samples(map.class).unwrap_or_else(|| {
            panic!(
                "offset {offset} lies in class {}, which is not stored",
                map.class
            )
        });
        (map, w)
    }

    fn check_length(&self, what: &str, len: usize) {
        assert_eq!(
            len,
            self.samples(),
            "the {what} has {len} samples, the grid of degree {} {}",
            self.p,
            self.samples()
        );
    }

    /// Writes the transfer function W_o of the V offset of table index `offset` (its class's
    /// read through its symmetry) to `out`, at the grid's indices: exact copies.
    ///
    /// # Panics
    ///
    /// If the offset's class is not stored or `out` is not [`samples`](Self::samples) long.
    pub fn offset_samples(&self, offset: usize, out: &mut [Complex<T>]) {
        self.check_length("output", out.len());
        let (map, w) = self.stored_class(offset);
        let n = self.azimuths;
        for (i, out) in out.chunks_exact_mut(n).enumerate() {
            let ring = self.ring_of(map, i);
            let w = &w[ring * n..(ring + 1) * n];
            for (j, o) in out.iter_mut().enumerate() {
                *o = w[Self::azimuth_of(map, n, j)];
            }
        }
    }

    fn ring_of(&self, map: OffsetMap, i: usize) -> usize {
        if map.flip_rings {
            self.rings - 1 - i
        } else {
            i
        }
    }

    fn azimuth_of(map: OffsetMap, n: usize, j: usize) -> usize {
        if map.reverse {
            (map.start + n - j) % n
        } else {
            (map.start + j) % n
        }
    }

    /// The pair step of the V offset of table index `offset`: `accumulator` += W_o · `signature`
    /// at every sample, W_o the offset's transfer function (its class's read through its
    /// symmetry; module documentation, "The pair and its order"). Allocates nothing.
    ///
    /// # Panics
    ///
    /// If the offset's class is not stored, or `signature` or `accumulator` is not
    /// [`samples`](Self::samples) long.
    pub fn pair(&self, offset: usize, signature: &[Complex<T>], accumulator: &mut [Complex<T>]) {
        self.check_length("signature", signature.len());
        self.check_length("accumulator", accumulator.len());
        let (map, w) = self.stored_class(offset);
        let n = self.azimuths;
        let s = map.start;
        for (i, (g, f)) in accumulator
            .chunks_exact_mut(n)
            .zip(signature.chunks_exact(n))
            .enumerate()
        {
            let ring = self.ring_of(map, i);
            let w = &w[ring * n..(ring + 1) * n];
            if map.reverse {
                // j ↦ s − j for j ≤ s, then n + s − j.
                add(&mut g[..=s], &f[..=s], w[..=s].iter().rev());
                add(&mut g[s + 1..], &f[s + 1..], w[s + 1..].iter().rev());
            } else {
                // j ↦ s + j for j < n − s, then j − (n − s).
                add(&mut g[..n - s], &f[..n - s], &w[s..]);
                add(&mut g[n - s..], &f[n - s..], &w[..s]);
            }
        }
    }
}

/// g += w · f along a run of samples, w from `w` in its order.
fn add<'a, T: RealScalar + 'a>(
    g: &mut [Complex<T>],
    f: &[Complex<T>],
    w: impl IntoIterator<Item = &'a Complex<T>>,
) {
    for ((g, &f), &w) in g.iter_mut().zip(f).zip(w) {
        *g = *g + w * f;
    }
}

/// The bound E(l, c) of proposal §3.6 for each of the 34 classes on a level of degree p and
/// wavenumber k̂ in its units (|t| = 2|rep|, circumradius √3), by
/// `nd_fmm_ref::helmholtz::diagonal::bound` with `constants`: the input of the rule that takes
/// a class to the diagonal form where E ≤ θ ε (CH4.2b, CH4.3). A function of k̂, p and the
/// class alone (proposal §2, constraint 9).
///
/// # Panics
///
/// If k̂ = 0.
pub fn class_bounds(p: usize, k_hat: Complex<f64>, constants: BoundConstants) -> Vec<f64> {
    let a = level(k_hat).circumradius();
    (0..CLASS_COUNT)
        .map(|c| bound(k_hat, distance(c), a, p, constants))
        .collect()
}
