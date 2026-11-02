//! The symmetry-reduced M2L tables of the Helmholtz family (Helmholtz CH3.2b;
//! docs/design/helmholtz-fmm-proposal.md §5.3, Classes): per level the matrices of the 16
//! class representatives of CONVENTIONS §3.12 in the layout of [`DenseM2l`], and the
//! cube-group transforms of the orthonormal coefficients of §3.15.
//!
//! # The transforms
//!
//! A Helmholtz coefficient of degree n is that of **Y**_n times a radial function of |ρ|
//! (§3.15, "Radial functions and scaling"), so for sources y ↦ c + P(y − c), P in O_h,
//! multipoles and locals alike become Oⁿ(P) times the old ones, with
//! **Y**_n(P x̂) = Oⁿ(P) **Y**_n(x̂). For a rotation Oⁿ is the block of §3.15 ("Rotations");
//! for an improper P, parity **Y**_n(−x̂) = (−1)ⁿ **Y**_n(x̂) gives Oⁿ(P) = (−1)ⁿ Oⁿ(−P),
//! the rule of §3.12 ("Coefficients under improper P") for Dⁿ, and Oⁿ(P) = N Dⁿ(P) N⁻¹ for
//! every P. Oⁿ is orthogonal and a homomorphism, so Oⁿ(Pᵀ) = Oⁿ(P)ᵀ. The factors of the
//! stored form (s_l^n, e^{±β_l}, 4π ik and the M2L's e^{2β − Im(k)|t|}) are per degree or
//! depend on |t| only, so they commute with Oⁿ, and §3.12's operator identity reads
//!
//! M2L(P d₀) = O(P) M2L(d₀) O(Pᵀ),
//!
//! with d₀ the representative of the offset's class and P its element (§3.12, "Symmetry
//! classes"; `crate::symmetry::class_of`).
//!
//! [`CubeTransforms`] builds Oⁿ(P), n ≤ p, for the 48 elements. A rotation Q is factored as
//! R_z(α) R_y(β) R_z(γ) (the z-y-z form of §3.8) with β ∈ {0, π/2, π} and α, γ quarter
//! turns, found by matching integer matrices. Oⁿ(R_z(α)) of a quarter turn is a signed
//! permutation (cos mα and sin mα formed in integers), Oⁿ(R_y(π)) = (−1)ⁿ F (the π − β rule
//! of §3.15 at β = 0) and Oⁿ(R_y(π/2)) is `nd_fmm_math::orthonormal::rotation::y_blocks` at
//! cos β = 0 and sin β = 1 exactly, the generator of §3.15. The product only picks entries
//! and signs, so it adds no rounding to the y-block's. The 16 elements that map the z-axis
//! to ±itself (β ∈ {0, π}) are signed permutations of the slots, each pair (±m) of a degree
//! to itself (§3.12, "The z-axis elements"), stored as one source slot and sign per slot and
//! applied exactly; the 32 others are dense per-degree blocks, row-major over the slots
//! m = −n to n as `nd_fmm_math::rotation::blocks`. Block n does not depend on p, so the
//! transforms of the largest degree serve every level.
//!
//! # Application
//!
//! [`ClassM2l::m2l`] adds, per vector, the one-vector operation: each multipole plane is
//! taken through O(Pᵀ) into scratch, the class matrix is applied there by
//! [`DenseM2l`]'s chain (one plane or two) into a zeroed complex local in scratch, and
//! O(P) of each of its two planes is added to the target's local. A transform's slot i of
//! degree n is Σⱼ Oⁿᵢⱼ xⱼ over the slots j of degree n in increasing order, each product and
//! sum rounded apart, then added to the output; a permutation's is ± x_σ(i), added. The
//! cost per pair is Dense's 2n² table values plus two transforms per plane, dense
//! (Σₙ (2n + 1)² multiply–adds, O(p³)) for the offsets whose element moves the z-axis.
//!
//! The class matrices are [`DenseM2l`]'s of the representatives bit for bit (the same build
//! from `HelmholtzTables<f64>`), so Classes equals Dense to rounding; at a representative
//! (element I) into a zero local it is Dense bit for bit.
//!
//! # Levels and the budget
//!
//! [`ClassM2l::select`] picks the levels whose 16 matrices fit a byte budget from the finest
//! level up, as [`DenseM2l::select`], charging the transforms' bytes as the largest chosen
//! degree grows: a function of the degrees and the budget only (proposal §2, constraint 9).
//!
//! [`DenseM2l`]: super::dense::DenseM2l
//! [`DenseM2l::select`]: super::dense::DenseM2l::select

use nd_fmm_math::orthonormal::rotation::{Scratch, compact_blocks_len, compact_offset, y_blocks};
use nd_fmm_math::rotation::{block_range, blocks_len};
use nd_fmm_math::{Complex, Layout, RealScalar};

use super::dense;
use super::{DEFAULT_BLOCK_BUDGET, HelmholtzTables};
use crate::geometry::{M2L_OFFSET_COUNT, m2l_offset_index};
use crate::symmetry::{
    GROUP_ORDER, M2L_CLASS_COUNT, SignedPermutation, class_of, class_representatives,
};

/// The number of elements of O_h that map the z-axis to ±itself (§3.12, "Symmetry
/// classes"): their transforms are signed permutations.
const Z_AXIS_ELEMENTS: usize = 16;

/// The quarter turns as integer (cos φ, sin φ), φ = 0, π/2, π, 3π/2.
const QUARTER_TURNS: [(i64, i64); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// How one element's transform is stored.
#[derive(Clone, Copy, Debug)]
enum Stored {
    /// Permutation t of `CubeTransforms::sources` and `signs`.
    Permutation(usize),
    /// Dense blocks t of `CubeTransforms::blocks`.
    Dense(usize),
}

/// The transforms Oⁿ(P), n ≤ p, of the orthonormal coefficients of CONVENTIONS §3.15 under
/// the 48 elements of O_h (the [module documentation](self), "The transforms").
#[derive(Clone, Debug)]
pub struct CubeTransforms<T: RealScalar> {
    p: usize,
    /// Per element g, where its transform is.
    stored: Vec<Stored>,
    /// Per permutation, the source slot of each of the (p + 1)² slots.
    sources: Vec<Vec<u32>>,
    /// Per permutation, the sign (±1) of each slot.
    signs: Vec<Vec<T>>,
    /// Per dense element, `blocks_len(p)` values.
    blocks: Vec<Vec<T>>,
}

impl<T: RealScalar> CubeTransforms<T> {
    /// Builds the transforms of degree ≤ `p` of all 48 elements, in f64 and rounded to T
    /// (exact for the permutations). Allocates.
    pub fn new(p: usize) -> Self {
        let mut compact = vec![0.0f64; compact_blocks_len(p)];
        y_blocks(p, 0.0, 1.0, &mut compact, &mut Scratch::new(p));
        let len = Layout::new(p).len();
        let mut full = vec![0.0f64; blocks_len(p)];
        let mut stored = Vec::with_capacity(GROUP_ORDER);
        let (mut sources, mut signs, mut blocks) = (Vec::new(), Vec::new(), Vec::new());
        for element in SignedPermutation::all() {
            let improper = !element.is_proper();
            let rotation = if improper { negated(element) } else { element };
            let (alpha, beta, gamma) = factor(rotation);
            for n in 0..=p {
                fill_block(n, alpha, beta, gamma, &compact, &mut full[block_range(n)]);
                if improper && n % 2 == 1 {
                    for v in &mut full[block_range(n)] {
                        *v = -*v;
                    }
                }
            }
            if fixes_z_axis(element) {
                let (mut source, mut sign) = (vec![0u32; len], vec![T::zero(); len]);
                for n in 0..=p {
                    let width = 2 * n + 1;
                    for (i, row) in full[block_range(n)].chunks_exact(width).enumerate() {
                        let nonzero: Vec<usize> = (0..width).filter(|&j| row[j] != 0.0).collect();
                        assert!(
                            nonzero.len() == 1 && row[nonzero[0]].abs() == 1.0,
                            "a z-axis element's block is a signed permutation"
                        );
                        let j = nonzero[0];
                        source[n * n + i] = (n * n + j) as u32;
                        sign[n * n + i] = T::from_f64(row[j]);
                    }
                }
                stored.push(Stored::Permutation(sources.len()));
                sources.push(source);
                signs.push(sign);
            } else {
                stored.push(Stored::Dense(blocks.len()));
                blocks.push(full.iter().map(|&v| T::from_f64(v)).collect());
            }
        }
        Self {
            p,
            stored,
            sources,
            signs,
            blocks,
        }
    }

    /// The bytes of the transforms of degree ≤ p: 16 permutations of (p + 1)² slots (a
    /// `u32` source and a sign in T each) and 32 dense sets of `blocks_len(p)` values.
    pub fn bytes_for(p: usize) -> usize {
        let len = Layout::new(p).len();
        Z_AXIS_ELEMENTS * len * (size_of::<u32>() + size_of::<T>())
            + (GROUP_ORDER - Z_AXIS_ELEMENTS) * blocks_len(p) * size_of::<T>()
    }

    /// The degree p.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Whether the transform of `element` is a signed permutation (the 16 elements that map
    /// the z-axis to ±itself) rather than dense blocks.
    pub fn is_permutation(&self, element: SignedPermutation) -> bool {
        matches!(self.stored[element.index()], Stored::Permutation(_))
    }

    /// The block Oⁿ(P) of `element` = P and degree n ≤ p, (2n + 1)² values row-major over
    /// the slots m = −n to n (expanded from the permutation where it is one). Allocates.
    ///
    /// # Panics
    ///
    /// If n > p.
    pub fn block(&self, element: SignedPermutation, n: usize) -> Vec<T> {
        assert!(n <= self.p, "degree {n} above p = {}", self.p);
        match self.stored[element.index()] {
            Stored::Dense(t) => self.blocks[t][block_range(n)].to_vec(),
            Stored::Permutation(t) => {
                let width = 2 * n + 1;
                let mut block = vec![T::zero(); width * width];
                for i in 0..width {
                    let j = self.sources[t][n * n + i] as usize - n * n;
                    block[i * width + j] = self.signs[t][n * n + i];
                }
                block
            }
        }
    }

    /// Adds O(P) x to `out` for `element` = P, degree by degree (module documentation,
    /// "Application"), on one plane of (q + 1)² values, q ≤ p, in the storage of
    /// CONVENTIONS §3.6. Allocates nothing.
    ///
    /// # Panics
    ///
    /// If `x` and `out` differ in length or are not (q + 1)² values for some q ≤ p.
    #[inline]
    pub fn apply(&self, element: SignedPermutation, x: &[T], out: &mut [T]) {
        assert_eq!(x.len(), out.len(), "`x` and `out` lengths");
        let q = x.len().isqrt();
        assert!(
            q * q == x.len() && q >= 1 && q - 1 <= self.p,
            "a plane of (q + 1)^2 values with q ≤ p = {}",
            self.p
        );
        match self.stored[element.index()] {
            Stored::Permutation(t) => {
                let (sources, signs) = (&self.sources[t], &self.signs[t]);
                for ((o, &j), &s) in out.iter_mut().zip(sources).zip(signs) {
                    *o = *o + s * x[j as usize];
                }
            }
            Stored::Dense(t) => {
                let blocks = &self.blocks[t];
                for n in 0..q {
                    let (lo, width) = (n * n, 2 * n + 1);
                    let x = &x[lo..lo + width];
                    let rows = blocks[block_range(n)].chunks_exact(width);
                    for (o, row) in out[lo..lo + width].iter_mut().zip(rows) {
                        let sum = row
                            .iter()
                            .zip(x)
                            .fold(T::zero(), |acc, (&a, &v)| acc + a * v);
                        *o = *o + sum;
                    }
                }
            }
        }
    }
}

/// −P, proper when P is improper.
fn negated(element: SignedPermutation) -> SignedPermutation {
    // −I is element 7, and −P = (−I) P.
    SignedPermutation::from_index(7).compose(&element)
}

/// Whether P maps the z-axis to ±itself, π(2) = 2.
fn fixes_z_axis(element: SignedPermutation) -> bool {
    element.permutation()[2] == 2
}

/// P as an integer matrix, row-major: entry (a, π(a)) is s_a.
fn integer_matrix(element: SignedPermutation) -> [[i64; 3]; 3] {
    let (pi, s) = (element.permutation(), element.signs());
    core::array::from_fn(|a| core::array::from_fn(|b| if b == pi[a] { s[a] } else { 0 }))
}

/// R_z(α) R_y(β) R_z(γ) in integers, for (cos, sin) pairs of quarter turns (§3.8's z-y-z
/// form, as `nd_fmm_math::rotation::euler_zyz`).
fn euler(a: (i64, i64), b: (i64, i64), g: (i64, i64)) -> [[i64; 3]; 3] {
    let ((ca, sa), (cb, sb), (cg, sg)) = (a, b, g);
    [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ]
}

/// The factors (α, β, γ) of a rotation of the cube as (cos, sin) pairs, β ∈ {0, π/2, π}
/// and γ = 0 where β is 0 or π.
fn factor(rotation: SignedPermutation) -> ((i64, i64), (i64, i64), (i64, i64)) {
    let target = integer_matrix(rotation);
    for beta in [(1, 0), (0, 1), (-1, 0)] {
        let gammas: &[(i64, i64)] = if beta.1 == 0 {
            &QUARTER_TURNS[..1]
        } else {
            &QUARTER_TURNS
        };
        for &gamma in gammas {
            for alpha in QUARTER_TURNS {
                if euler(alpha, beta, gamma) == target {
                    return (alpha, beta, gamma);
                }
            }
        }
    }
    unreachable!("every rotation of the cube has quarter-turn Euler angles")
}

/// (cos mφ, sin mφ) in integers for a quarter turn (cos φ, sin φ).
fn turn(phi: (i64, i64), m: usize) -> (i64, i64) {
    let (mut c, mut s) = (1, 0);
    for _ in 0..m % 4 {
        (c, s) = (c * phi.0 - s * phi.1, s * phi.0 + c * phi.1);
    }
    (c, s)
}

/// The single entry of row slot i (order i − n) of Oⁿ(R_z(φ)) for a quarter turn: its
/// column slot and sign. Row +m has cos mφ at +m and −sin mφ at −m, row −m sin mφ at +m and
/// cos mφ at −m (`nd_fmm_math::orthonormal::rotation::blocks`).
fn z_row(n: usize, phi: (i64, i64), i: usize) -> (usize, i64) {
    let m = i as i64 - n as i64;
    let (c, s) = turn(phi, m.unsigned_abs() as usize);
    let column = |order: i64| (order + n as i64) as usize;
    match m {
        0 => (i, 1),
        _ if m > 0 => {
            if c != 0 {
                (i, c)
            } else {
                (column(-m), -s)
            }
        }
        _ => {
            if c != 0 {
                (i, c)
            } else {
                (column(-m), s)
            }
        }
    }
}

/// The single entry of column slot j of Oⁿ(R_z(φ)): its row slot and sign (the rows of
/// [`z_row`] read by column).
fn z_column(n: usize, phi: (i64, i64), j: usize) -> (usize, i64) {
    let m = j as i64 - n as i64;
    let (c, s) = turn(phi, m.unsigned_abs() as usize);
    let row = |order: i64| (order + n as i64) as usize;
    match m {
        0 => (j, 1),
        _ if m > 0 => {
            if c != 0 {
                (j, c)
            } else {
                (row(-m), s)
            }
        }
        _ => {
            if c != 0 {
                (j, c)
            } else {
                (row(-m), -s)
            }
        }
    }
}

/// Writes Oⁿ(R_z(α) R_y(β) R_z(γ)) into `block`, (2n + 1)² values row-major: entry (i, j)
/// is the y-block's entry (σ(i), τ(j)) with the signs of the two z-rotations.
fn fill_block(
    n: usize,
    alpha: (i64, i64),
    beta: (i64, i64),
    gamma: (i64, i64),
    compact: &[f64],
    block: &mut [f64],
) {
    let width = 2 * n + 1;
    let compact = &compact[compact_offset(n)..compact_offset(n + 1)];
    let (cos, sin) = compact.split_at((n + 1) * (n + 1));
    // The y-block entry of slots (a, b), orders a − n and b − n.
    let y = |a: usize, b: usize| -> f64 {
        let (ma, mb) = (a as i64 - n as i64, b as i64 - n as i64);
        match beta {
            (1, 0) => f64::from(u8::from(a == b)),
            (-1, 0) => {
                // (−1)ⁿ F: +1 on m ≥ 0, −1 on m < 0.
                let parity = if n % 2 == 0 { 1.0 } else { -1.0 };
                let f = if ma >= 0 { 1.0 } else { -1.0 };
                if a == b { parity * f } else { 0.0 }
            }
            _ => {
                if ma >= 0 && mb >= 0 {
                    cos[ma as usize * (n + 1) + mb as usize]
                } else if ma < 0 && mb < 0 {
                    sin[(-ma - 1) as usize * n + (-mb - 1) as usize]
                } else {
                    0.0
                }
            }
        }
    };
    for i in 0..width {
        let (a, sa) = z_row(n, alpha, i);
        for j in 0..width {
            let (b, sb) = z_column(n, gamma, j);
            block[i * width + j] = (sa * sb) as f64 * y(a, b);
        }
    }
}

/// The symmetry-reduced M2L tables of the chosen levels of an octree (the
/// [module documentation](self)).
#[derive(Clone, Debug)]
pub struct ClassM2l<T: RealScalar> {
    /// Per level its degree and, if it has Classes, its 16 matrices.
    levels: Vec<(usize, Option<Vec<T>>)>,
    /// The transforms at the largest degree of the levels with Classes.
    transforms: Option<CubeTransforms<T>>,
    /// Per offset index its class, element P and Pᵀ.
    offsets: Vec<(usize, SignedPermutation, SignedPermutation)>,
}

impl<T: RealScalar> ClassM2l<T> {
    /// The bytes of one level's 16 matrices at degree p: 16 · 2 (p + 1)⁴ values of T.
    pub fn level_bytes(p: usize) -> usize {
        let n = Layout::new(p).len();
        M2L_CLASS_COUNT * 2 * n * n * size_of::<T>()
    }

    /// The levels whose matrices fit in `budget` bytes with the transforms of the largest
    /// chosen degree, from the finest level up; never levels 0 and 1 (module documentation,
    /// "Levels and the budget").
    pub fn select(degrees: &[usize], budget: usize) -> Vec<bool> {
        let mut chosen = vec![false; degrees.len()];
        let mut left = budget;
        let mut largest: Option<usize> = None;
        let transforms = |p: Option<usize>| p.map_or(0, CubeTransforms::<T>::bytes_for);
        for (l, &p) in degrees.iter().enumerate().skip(2).rev() {
            let grown = largest.map_or(p, |q| q.max(p));
            let bytes = Self::level_bytes(p) + transforms(Some(grown)) - transforms(largest);
            if bytes <= left {
                left -= bytes;
                chosen[l] = true;
                largest = Some(grown);
            }
        }
        chosen
    }

    /// Builds the matrices of the levels with `levels[l]` set and the transforms of their
    /// largest degree, for the root's k̂₀ = k r₀ and the degrees p_l (module documentation);
    /// with no level set, builds nothing.
    ///
    /// # Panics
    ///
    /// If `levels` and `degrees` differ in length, level 0 or 1 is chosen, or as
    /// [`HelmholtzTables::build`] panics.
    pub fn build(k_root: Complex<f64>, degrees: &[usize], levels: &[bool]) -> Self {
        assert_eq!(
            levels.len(),
            degrees.len(),
            "one choice per level: {} choices, {} degrees",
            levels.len(),
            degrees.len()
        );
        assert!(
            levels.iter().take(2).all(|&on| !on),
            "no class M2L on levels 0 and 1: they have no V list"
        );
        let offsets = (0..M2L_OFFSET_COUNT)
            .map(|d| {
                let (class, element) = class_of(d);
                (class, element, element.transpose())
            })
            .collect();
        let largest = degrees
            .iter()
            .zip(levels)
            .filter(|&(_, &on)| on)
            .map(|(&p, _)| p)
            .max();
        let Some(largest) = largest else {
            return Self {
                levels: degrees.iter().map(|&p| (p, None)).collect(),
                transforms: None,
                offsets,
            };
        };
        let tables = HelmholtzTables::<f64>::build(k_root, degrees, DEFAULT_BLOCK_BUDGET);
        let representatives = class_representatives()
            .map(|d| m2l_offset_index(d).expect("a representative is a V-list offset"));
        let levels = degrees
            .iter()
            .zip(levels)
            .enumerate()
            .map(|(l, (&p, &on))| {
                let matrices = on.then(|| dense::matrices(&tables, l, representatives.into_iter()));
                (p, matrices)
            })
            .collect();
        Self {
            levels,
            transforms: Some(CubeTransforms::new(largest)),
            offsets,
        }
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// The degree p_l of `level`.
    pub fn degree(&self, level: usize) -> usize {
        self.levels[level].0
    }

    /// Whether `level` has class tables.
    pub fn is_class(&self, level: usize) -> bool {
        self.levels[level].1.is_some()
    }

    /// The transforms, if any level has class tables.
    pub fn transforms(&self) -> Option<&CubeTransforms<T>> {
        self.transforms.as_ref()
    }

    /// The class and the element P of offset index `d` (`crate::symmetry::class_of`).
    ///
    /// # Panics
    ///
    /// If d ≥ 316.
    pub fn class_of(&self, d: usize) -> (usize, SignedPermutation) {
        let (class, element, _) = self.offsets[d];
        (class, element)
    }

    /// The bytes of every level's matrices and of the transforms.
    pub fn bytes(&self) -> usize {
        let matrices: usize = self
            .levels
            .iter()
            .filter_map(|(_, m)| m.as_ref())
            .map(|m| m.len() * size_of::<T>())
            .sum();
        matrices
            + self
                .transforms
                .as_ref()
                .map_or(0, |t| CubeTransforms::<T>::bytes_for(t.p()))
    }

    /// The 2n × n matrix of `class` on `level`, column-major, in [`DenseM2l`](super::dense::DenseM2l)'s layout:
    /// bit for bit `DenseM2l::matrix` of the class's representative.
    ///
    /// # Panics
    ///
    /// If `level` has no class tables or `class` ≥ 16.
    pub fn matrix(&self, level: usize, class: usize) -> &[T] {
        assert!(class < M2L_CLASS_COUNT, "class {class}");
        let (p, matrices) = &self.levels[level];
        let matrices = matrices
            .as_ref()
            .unwrap_or_else(|| panic!("no class M2L on level {level}"));
        let size = 2 * Layout::new(*p).len().pow(2);
        &matrices[class * size..(class + 1) * size]
    }

    /// Caller-owned scratch for [`m2l`](Self::m2l) on every level.
    pub fn scratch(&self) -> ClassScratch<T> {
        ClassScratch::new(self.transforms.as_ref().map_or(0, CubeTransforms::p))
    }

    /// M2L on `level` at the offset index `d`: adds the translations of `vectors`
    /// multipoles of `planes` planes (2, or 1 for the real multipoles of real k and real
    /// charges) to the target's `vectors` complex locals (module documentation,
    /// "Application"). Allocates nothing.
    ///
    /// # Panics
    ///
    /// If `level` has no class tables, d ≥ 316, `planes` is not 1 or 2, a slice is not
    /// `vectors` expansions at the level's degree, or `scratch` is for a smaller degree.
    #[allow(clippy::too_many_arguments)]
    pub fn m2l(
        &self,
        level: usize,
        d: usize,
        multipole: &[T],
        planes: usize,
        local: &mut [T],
        vectors: usize,
        scratch: &mut ClassScratch<T>,
    ) {
        let (class, element, transpose) = self.offsets[d];
        let a = self.matrix(level, class);
        let transforms = self.transforms.as_ref().expect("a level with classes");
        assert!(matches!(planes, 1 | 2), "{planes} multipole planes");
        let n = Layout::new(self.levels[level].0).len();
        assert_eq!(multipole.len(), vectors * planes * n, "multipole length");
        assert_eq!(local.len(), vectors * 2 * n, "local length");
        assert!(
            scratch.rotated.len() >= 2 * n,
            "scratch for a degree below the level's"
        );
        let rotated = &mut scratch.rotated[..planes * n];
        let translated = &mut scratch.translated[..2 * n];
        for (m, l) in multipole
            .chunks_exact(planes * n)
            .zip(local.chunks_exact_mut(2 * n))
        {
            rotated.fill(T::zero());
            for (x, out) in m.chunks_exact(n).zip(rotated.chunks_exact_mut(n)) {
                transforms.apply(transpose, x, out);
            }
            translated.fill(T::zero());
            dense::apply(a, n, planes, rotated, translated);
            for (x, out) in translated.chunks_exact(n).zip(l.chunks_exact_mut(n)) {
                transforms.apply(element, x, out);
            }
        }
    }
}

/// Caller-owned scratch of [`ClassM2l::m2l`]: a rotated multipole and a translated local,
/// two planes each, of degree p.
#[derive(Clone, Debug)]
pub struct ClassScratch<T: RealScalar> {
    rotated: Vec<T>,
    translated: Vec<T>,
}

impl<T: RealScalar> ClassScratch<T> {
    /// Scratch for degrees up to `p`, 4 (p + 1)² values.
    pub fn new(p: usize) -> Self {
        let n = Layout::new(p).len();
        Self {
            rotated: vec![T::zero(); 2 * n],
            translated: vec![T::zero(); 2 * n],
        }
    }
}
