//! The RCR tables of the Helmholtz family on the levels of an octree, and the table-driven
//! M2M, L2L and M2L (CONVENTIONS §3.15, "Rotations" and "Coaxial translations";
//! docs/design/helmholtz-fmm-proposal.md §5.2, §5.3 Rotation; Helmholtz CH3.1a).
//!
//! [`nd_fmm_ref::helmholtz::rcr`] translates an expansion in three steps: it rotates the
//! input so that the shift t = c′ − c lies on +z (a 2 × 2 rotation per order by −φ, then
//! the transposed y-block of the polar angle θ), applies the coaxial table per order from
//! the input's degrees to the output's, and rotates back. It builds the y-blocks and takes
//! the coaxial table on every call. On an octree level the shifts are fixed, so
//! [`HelmholtzTables`] precomputes per level what the three steps need: the coaxial
//! tables with the stored form's factors folded in, and the y-blocks.
//!
//! # Levels and units
//!
//! Level l works in the units of its own boxes (half-width 1, centres two apart), where
//! the wavenumber is k̂_l = k r_l, r_l the level's half-width. The tables are built from
//! k̂₀ = k r₀ of the root and k̂_l = k̂₀ / 2^l, exact in floating point, and level l's
//! constants are those of `nd_fmm_ref::helmholtz::Level::new(k̂_l, 1)`. In these units:
//!
//! | Operator | Level | From | To | Shift t | Degrees |
//! | --- | --- | --- | --- | --- | --- |
//! | M2M | the parent's, l | child o of l + 1 | its parent | −s_o (child units) | p_{l+1} → p_l |
//! | L2L | the child's, l + 1 | the parent on l | child o | s_o (child units) | p_l → p_{l+1} |
//! | M2L | l ≥ 2 | source | target, offset d | 2d | p_l → p_l |
//!
//! with s_o the octant direction ([`octant_direction`]) and d the V-list offset
//! index(target) − index(source) ([`m2l_offsets`]). The M2M and L2L tables of a level pair
//! are built in the child's units, as `rcr` builds them from the child's level:
//! `rcr::m2m_table(p_{l+1}, p_l, child)` and `rcr::l2l_table(p_l, p_{l+1}, child)` with
//! child = `Level::new(k̂_{l+1}, 1)` and parent = `Level::new(k̂_{l+1}, 2)`.
//!
//! # The stored form, folded in
//!
//! Each table holds per order m the full coefficient of the coaxial step from degree n to
//! n′, as `rcr`'s module documentation ("Stored form") writes it, formed in f64 exactly as
//! `rcr` forms it and rounded to T:
//!
//! ```text
//! M2M  Ê_{n′n} · (s_c/s_P)ⁿ s_P^{n−n′}       (Ê of rcr::m2m_table)
//! L2L  Ê_{n′n} · (s_c/s_P)^{n′} s_P^{n′−n}    (Ê of rcr::l2l_table)
//! M2L  Ê_{n′n} · 4π ik e^{2β − Im(k)|t|}     (Ê of rcr::m2l_table)
//! ```
//!
//! For real k (Im k = 0) the M2M and L2L coefficients are real (§3.1): the tables keep
//! their real parts ([`HelmholtzTables::real_shifts`]) and apply them to each plane of an
//! expansion, the real shifts of proposal §5.2. Everything else is complex.
//!
//! # Shifts and their keys
//!
//! `rcr` forms the polar angles and the distance from the shift in floating point. Shifts
//! of one geometric class (the same |d|², or the same polar angle) can round to different
//! bits: |t| of (2, 2, 1) and (1, 2, 2) is formed by two `hypot`s in different orders. The
//! tables are therefore keyed by the bits `rcr` would form: one M2L coaxial table per
//! distinct |t| of the 316 shifts (at least the 15 values of |d|²), one y-block set per
//! distinct (cos θ, sin θ), and the azimuth of each shift per offset. So every application
//! uses exactly the angles, coaxial table and factor of `rcr` at the same shift, and in f64
//! gives `rcr`'s bits.
//!
//! # Blocks and the budget
//!
//! The y-blocks of degree p take [`compact_blocks_len`]`(p)` values, about (2/3) p³. The two
//! polar angles of the octants (cos θ = ±1/√3) serve M2M and L2L of a level pair at degree
//! max(p_l, p_{l+1}) and are always stored. M2L's blocks, one set per distinct polar angle
//! at degree p_l, are stored level by level from the finest while their total stays within
//! the byte budget ([`HelmholtzTables::build`]); on the other levels they are formed per
//! call into the scratch, by the same function from the same angles, so the bits do not
//! depend on the budget. Both are built in T by
//! [`y_blocks`] as `rcr` builds them, not in f64: the
//! blocks formed per call cannot be built in f64 without a second scratch, and the stored
//! ones must match them.
//!
//! # Application
//!
//! [`HelmholtzTables::m2m`], [`l2l`](HelmholtzTables::l2l) and
//! [`m2l`](HelmholtzTables::m2l) translate m expansions per box, one after the other, each
//! of one or two planes of (p + 1)² values (real part, then imaginary part) in the layout
//! of CONVENTIONS §3.6, and add (+=) into the output. The steps restate `rcr`'s private
//! `translate`, in its order of operations: per plane the 2 × 2 rotations by −φ and the
//! transposed y-block; per output slot the coaxial sum over the input degrees, a complex
//! product per term (an input of one plane as Im = 0) or, for the real shifts, a real
//! product per plane; per plane the y-block and the rotations by φ, then the addition.
//! This crate otherwise never restates an operator formula; like [`rotation`] and
//! [`plane_wave`], this module does, and its tests pin it to `rcr` bit for bit.
//!
//! [`rotation`]: crate::rotation
//! [`plane_wave`]: crate::plane_wave
//!
//! # Example
//!
//! ```
//! use nd_fmm_math::Complex;
//! use nd_fmm_ref::Frame;
//! use nd_fmm_ref::helmholtz::{Level, Workspace, rcr};
//! use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
//!
//! // Three levels for k̂₀ = k r₀ = 4 + 0.5i, degrees 0, 0 and 8.
//! let k0 = Complex::new(4.0, 0.5);
//! let tables = HelmholtzTables::<f64>::build(k0, &[0, 0, 8], DEFAULT_BLOCK_BUDGET);
//! let mut scratch = tables.scratch();
//!
//! // M2L on level 2 at the offset index 0, d = (−3, −3, −3), against rcr.
//! let p = 8;
//! let len = 2 * (p + 1) * (p + 1);
//! let multipole: Vec<f64> = (0..len).map(|i| 1.0 / (1.0 + i as f64)).collect();
//! let mut local = vec![0.0; len];
//! tables.m2l(2, 0, &multipole, 2, &mut local, 1, &mut scratch);
//!
//! let level = Level::new(tables.k_hat(2), 1.0);
//! let (from, to) = (Frame::new([0.0; 3], 1.0), Frame::new([-6.0, -6.0, -6.0], 1.0));
//! let table = rcr::m2l_table(p, p, &level, &from, &to);
//! let mut expected = vec![0.0; len];
//! rcr::m2l(p, p, &level, &from, &to, &table, &mut Workspace::new(p), &multipole, &mut expected);
//! assert_eq!(local, expected);
//! ```

use nd_fmm_math::orthonormal::rotation::{
    Scratch as BlockScratch, compact_blocks_len, compact_offset, y_blocks,
};
use nd_fmm_math::{Complex, Layout, RealScalar};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, rcr};

use crate::geometry::{M2L_OFFSET_COUNT, OCTANT_COUNT, m2l_offsets, octant_direction};

/// The default byte budget of the stored M2L y-blocks, 512 MiB.
pub const DEFAULT_BLOCK_BUDGET: usize = 512 << 20;

/// The RCR tables of an octree's levels for one wavenumber and the degrees p_l (the
/// [module documentation](self)).
#[derive(Clone, Debug)]
pub struct HelmholtzTables<T: RealScalar> {
    real_shifts: bool,
    levels: Vec<LevelTables<T>>,
    /// The azimuth and the key of the polar angle of each M2M shift −s_o; L2L's shift s_o
    /// has the opposite polar angle and azimuth.
    octants: [(Azimuth<T>, usize); OCTANT_COUNT],
    /// Per V-list offset: its azimuth, polar-angle key and distance key.
    offsets: Vec<OffsetKeys<T>>,
    /// The distinct (cos θ, sin θ) of the 316 shifts, by key.
    m2l_angles: Vec<(T, T)>,
    /// The largest degree of any level.
    p_max: usize,
}

/// The tables of one level.
#[derive(Clone, Debug)]
struct LevelTables<T: RealScalar> {
    /// k̂_l = k r_l in f64.
    k_hat: Complex<f64>,
    p: usize,
    /// M2M from level l + 1 into this level, and L2L from this level into level l + 1;
    /// `None` on the finest level.
    up: Option<Shift<T>>,
    down: Option<Shift<T>>,
    /// The y-blocks of the octant angles at max(p_l, p_{l+1}), by key; empty on the finest
    /// level.
    pair_blocks: [Vec<T>; 2],
    /// M2L per distance key; empty on levels 0 and 1.
    m2l: Vec<Shift<T>>,
    /// The M2L y-blocks per polar-angle key at p_l, if stored.
    m2l_blocks: Option<Vec<Vec<T>>>,
}

/// A coaxial step's coefficients from degrees ≤ p_in to ≤ p_out: per order m the block of
/// rows n′ = m..=p_out by columns n = m..=p_in, row-major; complex unless real.
#[derive(Clone, Debug)]
struct Shift<T> {
    p_in: usize,
    p_out: usize,
    /// The start of each order's block.
    starts: Vec<usize>,
    re: Vec<T>,
    /// Empty for a real step.
    im: Vec<T>,
}

/// cos φ and sin φ of a shift (φ = 0 on the axis), as `rcr` forms them.
#[derive(Clone, Copy, Debug)]
struct Azimuth<T> {
    cos: T,
    sin: T,
}

#[derive(Clone, Copy, Debug)]
struct OffsetKeys<T> {
    azimuth: Azimuth<T>,
    angle: usize,
    distance: usize,
}

/// The caller-owned scratch of the applications: one expansion of two planes at the
/// largest degree twice, and the y-blocks of one polar angle with their work space.
pub struct HelmholtzScratch<T: RealScalar> {
    rotated: Vec<T>,
    work: Vec<T>,
    blocks: Vec<T>,
    block_scratch: BlockScratch<T>,
}

impl<T: RealScalar> HelmholtzTables<T> {
    /// Builds the tables of the levels 0..`degrees.len()` for the root's k̂₀ = k r₀ (r₀ the
    /// root's half-width, in f64) and the degrees p_l, storing M2L's y-blocks from the
    /// finest level up while their total stays within `block_budget` bytes. M2L tables are
    /// built on levels ≥ 2 only: no V list lies above level 2.
    ///
    /// Building is serial and deterministic. Its cost is that of the coaxial tables, O(p³)
    /// per table (a projection where Re(k̂) a > 40, CONVENTIONS §3.15), and of the blocks.
    ///
    /// # Panics
    ///
    /// Unless k̂₀ is finite and nonzero with Re ≥ 0, Im ≥ 0 and Re > 0 (Yukawa's real path,
    /// Re k = 0, is not served here), and `degrees` is not empty; and as the coaxial
    /// builders of `nd_fmm_math::coaxial` (for instance where the (S|R) start column leaves
    /// the f64 range).
    pub fn build(k_root: Complex<f64>, degrees: &[usize], block_budget: usize) -> Self {
        assert!(
            k_root.re.is_finite() && k_root.im.is_finite(),
            "k̂₀ must be finite"
        );
        assert!(
            k_root.re > 0.0 && k_root.im >= 0.0,
            "k̂₀ must have Re > 0 and Im ≥ 0 (Yukawa's real path is not served)"
        );
        assert!(!degrees.is_empty(), "no levels");
        let real_shifts = k_root.im == 0.0;
        let nlevels = degrees.len();
        let k_hat = |l: usize| k_root * (0.5f64).powi(l as i32);

        // The shifts' keys, from the floating-point shifts `rcr` forms.
        let octants = core::array::from_fn(|o| {
            let t = octant_direction(o).map(|s| -T::from_f64(s as f64));
            let angles = Angles::of(t);
            let key = usize::from(t[2] < T::zero());
            (angles.azimuth, key)
        });
        // The two polar angles of the octant shifts, key 0 for z > 0 and 1 for z < 0.
        let octant_angles = [1.0, -1.0].map(|z| {
            let angles = Angles::of([T::one(), T::one(), T::from_f64(z)]);
            (angles.cos_theta, angles.sin_theta)
        });
        let mut m2l_angles: Vec<(T, T)> = Vec::new();
        let mut distances: Vec<(f64, [f64; 3])> = Vec::new();
        let offsets: Vec<OffsetKeys<T>> = m2l_offsets()
            .iter()
            .map(|d| {
                let t = d.map(|x| T::from_f64(2.0 * x as f64));
                let angles = Angles::of(t);
                let polar = (angles.cos_theta, angles.sin_theta);
                let angle = find_or_push(&mut m2l_angles, polar, |a, b| {
                    a.0.to_f64().to_bits() == b.0.to_f64().to_bits()
                        && a.1.to_f64().to_bits() == b.1.to_f64().to_bits()
                });
                let t64 = d.map(|x| 2.0 * x as f64);
                let distance = find_or_push(&mut distances, (norm(t64), t64), |a, b| {
                    a.0.to_bits() == b.0.to_bits()
                });
                OffsetKeys {
                    azimuth: angles.azimuth,
                    angle,
                    distance,
                }
            })
            .collect();

        let mut levels: Vec<LevelTables<T>> = (0..nlevels)
            .map(|l| {
                let p = degrees[l];
                let (up, down, pair_blocks) = if l + 1 < nlevels {
                    let (p_parent, p_child) = (p, degrees[l + 1]);
                    let child = Level::new(k_hat(l + 1), 1.0);
                    let parent = Level::new(k_hat(l + 1), 2.0);
                    let (sc, sp) = (child.scale(), parent.scale());
                    let m2m = rcr::m2m_table(p_child, p_parent, &child);
                    let l2l = rcr::l2l_table(p_parent, p_child, &child);
                    let up = Shift::new(p_child, p_parent, real_shifts, |m, n_out, n_in| {
                        m2m.get(m, n_out, n_in) * power_ratio(sc, n_in, sp, n_out)
                    });
                    let down = Shift::new(p_parent, p_child, real_shifts, |m, n_out, n_in| {
                        l2l.get(m, n_out, n_in) * power_ratio(sc, n_out, sp, n_in)
                    });
                    let degree = p_parent.max(p_child);
                    let mut scratch = BlockScratch::new(degree);
                    let blocks = octant_angles.map(|(c, s)| {
                        let mut out = vec![T::zero(); compact_blocks_len(degree)];
                        y_blocks(degree, c, s, &mut out, &mut scratch);
                        out
                    });
                    (Some(up), Some(down), blocks)
                } else {
                    (None, None, [Vec::new(), Vec::new()])
                };
                let m2l = if l >= 2 {
                    let level = Level::new(k_hat(l), 1.0);
                    let from = Frame::new([0.0; 3], 1.0);
                    distances
                        .iter()
                        .map(|&(t, centre)| {
                            let to = Frame::new(centre, 1.0);
                            let table = rcr::m2l_table(p, p, &level, &from, &to);
                            let k = level.k();
                            let beta = level.damping();
                            let factor = four_pi_ik(k) * (beta + beta - k.im * t).exp();
                            Shift::new(p, p, false, |m, n_out, n_in| {
                                table.get(m, n_out, n_in) * factor
                            })
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                LevelTables {
                    k_hat: k_hat(l),
                    p,
                    up,
                    down,
                    pair_blocks,
                    m2l,
                    m2l_blocks: None,
                }
            })
            .collect();

        // M2L's blocks, finest level first, within the budget.
        let mut left = block_budget;
        for level in levels.iter_mut().skip(2).rev() {
            let bytes = m2l_angles.len() * compact_blocks_len(level.p) * size_of::<T>();
            if bytes > left {
                continue;
            }
            left -= bytes;
            let mut scratch = BlockScratch::new(level.p);
            level.m2l_blocks = Some(
                m2l_angles
                    .iter()
                    .map(|&(c, s)| {
                        let mut out = vec![T::zero(); compact_blocks_len(level.p)];
                        y_blocks(level.p, c, s, &mut out, &mut scratch);
                        out
                    })
                    .collect(),
            );
        }

        Self {
            real_shifts,
            p_max: degrees.iter().copied().max().unwrap_or(0),
            levels,
            octants,
            offsets,
            m2l_angles,
        }
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// The degree p_l of `level`.
    pub fn degree(&self, level: usize) -> usize {
        self.levels[level].p
    }

    /// k̂_l = k r_l of `level`, in f64: the wavenumber in the level's units.
    pub fn k_hat(&self, level: usize) -> Complex<f64> {
        self.levels[level].k_hat
    }

    /// Whether the M2M and L2L coefficients are real (real k), applied to each plane.
    pub fn real_shifts(&self) -> bool {
        self.real_shifts
    }

    /// The number of distinct M2L distances |t| of the 316 shifts, one coaxial table each
    /// per level (module documentation, "Shifts and their keys").
    pub fn m2l_distances(&self) -> usize {
        self.levels.iter().map(|l| l.m2l.len()).max().unwrap_or(0)
    }

    /// The number of distinct polar angles of the 316 shifts, one y-block set each.
    pub fn m2l_polar_angles(&self) -> usize {
        self.m2l_angles.len()
    }

    /// Whether `level`'s M2L y-blocks are stored (else formed per call).
    pub fn m2l_blocks_stored(&self, level: usize) -> bool {
        self.levels[level].m2l_blocks.is_some()
    }

    /// The bytes of every table and stored block.
    pub fn bytes(&self) -> usize {
        let values: usize = self
            .levels
            .iter()
            .map(|l| {
                let shifts = l.up.iter().chain(&l.down).chain(&l.m2l);
                shifts.map(|s| s.re.len() + s.im.len()).sum::<usize>()
                    + l.pair_blocks.iter().map(Vec::len).sum::<usize>()
                    + l.m2l_blocks.iter().flatten().map(Vec::len).sum::<usize>()
            })
            .sum();
        values * size_of::<T>()
    }

    /// A scratch set for the applications on every level.
    pub fn scratch(&self) -> HelmholtzScratch<T> {
        let len = Layout::new(self.p_max).len();
        HelmholtzScratch {
            rotated: vec![T::zero(); 2 * len],
            work: vec![T::zero(); 2 * len],
            blocks: vec![T::zero(); compact_blocks_len(self.p_max)],
            block_scratch: BlockScratch::new(self.p_max),
        }
    }

    /// M2M: adds the `vectors` multipoles of child `o` (on `level` + 1) to those of its
    /// parent on `level`, each of `planes` planes (2, or 1 for the real multipoles of real
    /// k and real charges).
    ///
    /// # Panics
    ///
    /// If `level` has no child level, o ≥ 8, `planes` is not 1 or 2 (1 only with real
    /// shifts), or a slice is not `vectors` expansions of `planes` planes at its degree.
    #[allow(clippy::too_many_arguments)]
    pub fn m2m(
        &self,
        level: usize,
        o: usize,
        child: &[T],
        parent: &mut [T],
        planes: usize,
        vectors: usize,
        scratch: &mut HelmholtzScratch<T>,
    ) {
        let tables = &self.levels[level];
        let shift = tables.up.as_ref().expect("M2M into the finest level");
        let (azimuth, key) = self.octants[o];
        self.translate(
            shift,
            Blocks::Stored(&tables.pair_blocks[key]),
            azimuth,
            (child, planes),
            (parent, planes),
            vectors,
            scratch,
        );
    }

    /// L2L: adds the `vectors` locals of the parent on `level` − 1 to those of its child
    /// `o` on `level`, each of `planes` planes (2, or 1 with real shifts only).
    ///
    /// # Panics
    ///
    /// If `level` is 0, o ≥ 8, or as [`m2m`](Self::m2m).
    #[allow(clippy::too_many_arguments)]
    pub fn l2l(
        &self,
        level: usize,
        o: usize,
        parent: &[T],
        child: &mut [T],
        planes: usize,
        vectors: usize,
        scratch: &mut HelmholtzScratch<T>,
    ) {
        assert!(level > 0, "L2L into level 0");
        let tables = &self.levels[level - 1];
        let shift = tables.down.as_ref().expect("a parent level has an L2L");
        // The shift s_o is −(M2M's): the opposite polar angle and azimuth.
        let (azimuth, key) = self.octants[o];
        let azimuth = Azimuth {
            cos: -azimuth.cos,
            sin: -azimuth.sin,
        };
        self.translate(
            shift,
            Blocks::Stored(&tables.pair_blocks[1 - key]),
            azimuth,
            (parent, planes),
            (child, planes),
            vectors,
            scratch,
        );
    }

    /// M2L on `level` ≥ 2 at the V-list offset index `d`: adds the `vectors` multipoles of
    /// the source, each of `multipole_planes` planes (2, or 1 for real k and real charges),
    /// translated to the target's complex locals (two planes each).
    ///
    /// # Panics
    ///
    /// If `level` < 2, d ≥ 316, `multipole_planes` is not 1 or 2, or a slice is not
    /// `vectors` expansions at the level's degree.
    #[allow(clippy::too_many_arguments)]
    pub fn m2l(
        &self,
        level: usize,
        d: usize,
        multipole: &[T],
        multipole_planes: usize,
        local: &mut [T],
        vectors: usize,
        scratch: &mut HelmholtzScratch<T>,
    ) {
        assert!(level >= 2, "no M2L on level {level}");
        assert!(d < M2L_OFFSET_COUNT, "offset index {d}");
        let tables = &self.levels[level];
        let keys = self.offsets[d];
        let blocks = match &tables.m2l_blocks {
            Some(blocks) => Blocks::Stored(&blocks[keys.angle]),
            None => Blocks::PerCall(self.m2l_angles[keys.angle]),
        };
        self.translate(
            &tables.m2l[keys.distance],
            blocks,
            keys.azimuth,
            (multipole, multipole_planes),
            (local, 2),
            vectors,
            scratch,
        );
    }

    /// The three steps of the module documentation for `vectors` expansions.
    #[allow(clippy::too_many_arguments)]
    fn translate(
        &self,
        shift: &Shift<T>,
        blocks: Blocks<'_, T>,
        azimuth: Azimuth<T>,
        (input, in_planes): (&[T], usize),
        (output, out_planes): (&mut [T], usize),
        vectors: usize,
        scratch: &mut HelmholtzScratch<T>,
    ) {
        let real = shift.im.is_empty();
        assert!(
            matches!(in_planes, 1 | 2)
                && (if real {
                    out_planes == in_planes
                } else {
                    out_planes == 2
                }),
            "{in_planes} input and {out_planes} output planes for a {} step",
            if real { "real" } else { "complex" }
        );
        let (p_in, p_out) = (shift.p_in, shift.p_out);
        let (len_in, len_out) = (Layout::new(p_in).len(), Layout::new(p_out).len());
        let (size_in, size_out) = (in_planes * len_in, out_planes * len_out);
        assert_eq!(input.len(), vectors * size_in, "input length");
        assert_eq!(output.len(), vectors * size_out, "output length");
        let HelmholtzScratch {
            rotated,
            work,
            blocks: block_buffer,
            block_scratch,
        } = scratch;
        let blocks: &[T] = match blocks {
            Blocks::Stored(blocks) => blocks,
            Blocks::PerCall((c, s)) => {
                let p = p_in.max(p_out);
                let out = &mut block_buffer[..compact_blocks_len(p)];
                y_blocks(p, c, s, out, block_scratch);
                out
            }
        };
        for (input, output) in input
            .chunks_exact(size_in)
            .zip(output.chunks_exact_mut(size_out))
        {
            // 1. Rotate each plane: z by −φ, then the transposed y-block.
            for plane in 0..in_planes {
                let x = &mut work[..len_in];
                x.copy_from_slice(&input[plane * len_in..(plane + 1) * len_in]);
                rotate_z(p_in, azimuth, -T::one(), x);
                apply_y(
                    p_in,
                    blocks,
                    true,
                    x,
                    &mut rotated[plane * len_in..(plane + 1) * len_in],
                );
            }

            // 2. The coaxial step, per order, into `work`.
            let shifted = &mut work[..out_planes * len_out];
            shifted.fill(T::zero());
            for n_out in 0..=p_out {
                for m in 0..=n_out.min(p_in) {
                    let row = shift.row(m, n_out);
                    for slot in slots(m) {
                        let i_out = index(n_out, slot);
                        let terms = (m..=p_in).map(|n_in| index(n_in, slot));
                        if real {
                            for plane in 0..in_planes {
                                let x = &rotated[plane * len_in..(plane + 1) * len_in];
                                let mut sum = T::zero();
                                for (&e, i_in) in shift.re[row.clone()].iter().zip(terms.clone()) {
                                    sum = sum + e * x[i_in];
                                }
                                shifted[plane * len_out + i_out] = sum;
                            }
                        } else {
                            let mut sum = Complex::new(T::zero(), T::zero());
                            let entries = shift.re[row.clone()].iter().zip(&shift.im[row.clone()]);
                            for ((&re, &im), i_in) in entries.zip(terms) {
                                let c = if in_planes == 2 {
                                    Complex::new(rotated[i_in], rotated[len_in + i_in])
                                } else {
                                    Complex::new(rotated[i_in], T::zero())
                                };
                                sum = sum + Complex::new(re, im) * c;
                            }
                            shifted[i_out] = sum.re;
                            shifted[len_out + i_out] = sum.im;
                        }
                    }
                }
            }

            // 3. Rotate back: the y-block, then z by φ, and accumulate.
            for plane in 0..out_planes {
                let y = &mut rotated[..len_out];
                apply_y(
                    p_out,
                    blocks,
                    false,
                    &shifted[plane * len_out..(plane + 1) * len_out],
                    y,
                );
                rotate_z(p_out, azimuth, T::one(), y);
                for (o, &v) in output[plane * len_out..(plane + 1) * len_out]
                    .iter_mut()
                    .zip(y.iter())
                {
                    *o = *o + v;
                }
            }
        }
    }
}

impl<T: RealScalar> Shift<T> {
    /// The step from p_in to p_out with coefficients `entry(m, n′, n)` (f64), rounded to
    /// T; real steps keep the real parts.
    fn new(
        p_in: usize,
        p_out: usize,
        real: bool,
        entry: impl Fn(usize, usize, usize) -> Complex<f64>,
    ) -> Self {
        let orders = p_in.min(p_out) + 1;
        let mut starts = Vec::with_capacity(orders);
        let (mut re, mut im) = (Vec::new(), Vec::new());
        for m in 0..orders {
            starts.push(re.len());
            for n_out in m..=p_out {
                for n_in in m..=p_in {
                    let e = entry(m, n_out, n_in);
                    re.push(T::from_f64(e.re));
                    if !real {
                        im.push(T::from_f64(e.im));
                    }
                }
            }
        }
        Self {
            p_in,
            p_out,
            starts,
            re,
            im,
        }
    }

    /// The entries of order m and output degree n′ ≥ m, over n = m..=p_in.
    #[inline]
    fn row(&self, m: usize, n_out: usize) -> core::ops::Range<usize> {
        let width = self.p_in + 1 - m;
        let start = self.starts[m] + (n_out - m) * width;
        start..start + width
    }
}

/// The y-blocks of an application.
enum Blocks<'a, T> {
    Stored(&'a [T]),
    /// cos θ and sin θ, formed per call into the scratch.
    PerCall((T, T)),
}

/// The polar angles of a shift as `rcr` forms them: cos θ, sin θ ≥ 0, and the azimuth.
struct Angles<T> {
    cos_theta: T,
    sin_theta: T,
    azimuth: Azimuth<T>,
}

impl<T: RealScalar> Angles<T> {
    fn of([x, y, z]: [T; 3]) -> Self {
        let rho = x.hypot(y);
        let r = rho.hypot(z);
        let (cos, sin) = if rho > T::zero() {
            (x / rho, y / rho)
        } else {
            (T::one(), T::zero())
        };
        Self {
            cos_theta: z / r,
            sin_theta: rho / r,
            azimuth: Azimuth { cos, sin },
        }
    }
}

/// The index of `value` in `values` by `same`, appended if new.
fn find_or_push<V: Copy>(values: &mut Vec<V>, value: V, same: impl Fn(&V, &V) -> bool) -> usize {
    match values.iter().position(|v| same(v, &value)) {
        Some(i) => i,
        None => {
            values.push(value);
            values.len() - 1
        }
    }
}

/// Applies Oⁿ(R_z(σφ)), n ≤ p, in place: for m > 0, with c = cos mσφ and s = sin mσφ,
/// slot +m ← c x₊ − s x₋ and slot −m ← s x₊ + c x₋ (as `rcr`).
fn rotate_z<T: RealScalar>(p: usize, azimuth: Azimuth<T>, sigma: T, x: &mut [T]) {
    let step = (azimuth.cos, sigma * azimuth.sin);
    let mut phase = (T::one(), T::zero());
    for m in 1..=p {
        phase = (
            phase.0 * step.0 - phase.1 * step.1,
            phase.1 * step.0 + phase.0 * step.1,
        );
        let (c, s) = phase;
        for n in m..=p {
            let (plus, minus) = (index(n, m as isize), index(n, -(m as isize)));
            let (a, b) = (x[plus], x[minus]);
            x[plus] = c * a - s * b;
            x[minus] = s * a + c * b;
        }
    }
}

/// `out` = Oⁿ(R_y(θ)) x per degree n ≤ p, or its transpose with `transpose`, from the
/// compact blocks (cos half over the slots m ≥ 0, sin half over m < 0; as `rcr`).
fn apply_y<T: RealScalar>(p: usize, blocks: &[T], transpose: bool, x: &[T], out: &mut [T]) {
    for n in 0..=p {
        let block = &blocks[compact_offset(n)..compact_offset(n + 1)];
        let (cos, sin) = block.split_at((n + 1) * (n + 1));
        for a in 0..=n {
            let mut sum = T::zero();
            for b in 0..=n {
                let e = if transpose {
                    cos[b * (n + 1) + a]
                } else {
                    cos[a * (n + 1) + b]
                };
                sum = sum + e * x[index(n, b as isize)];
            }
            out[index(n, a as isize)] = sum;
        }
        for a in 1..=n {
            let mut sum = T::zero();
            for b in 1..=n {
                let e = if transpose {
                    sin[(b - 1) * n + (a - 1)]
                } else {
                    sin[(a - 1) * n + (b - 1)]
                };
                sum = sum + e * x[index(n, -(b as isize))];
            }
            out[index(n, -(a as isize))] = sum;
        }
    }
}

/// The slots +m and −m of order m (one slot for m = 0).
fn slots(m: usize) -> impl Iterator<Item = isize> {
    let m = m as isize;
    core::iter::once(m).chain((m > 0).then_some(-m))
}

/// idx(n, m) of §3.6.
#[inline]
fn index(n: usize, m: isize) -> usize {
    (n * n + n).wrapping_add_signed(m)
}

/// aⁿ / b^{n′}, formed as (a/b)ⁿ b^{n − n′} (as `rcr`).
fn power_ratio(a: f64, n: usize, b: f64, n_prime: usize) -> f64 {
    (a / b).powi(n as i32) * b.powi(n as i32 - n_prime as i32)
}

/// 4π ik = 4π (−Im k + i Re k) (as `rcr`).
fn four_pi_ik(k: Complex<f64>) -> Complex<f64> {
    let c = 4.0 * core::f64::consts::PI;
    Complex::new(-(c * k.im), c * k.re)
}

fn norm([x, y, z]: [f64; 3]) -> f64 {
    x.hypot(y).hypot(z)
}
