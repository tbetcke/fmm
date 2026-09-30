//! The cube symmetry group O_h, its action on V-list offsets and on expansion
//! coefficients, and the 16 symmetry classes of the M2L offsets (CONVENTIONS §3.12).
//!
//! - [`SignedPermutation`] is one of the 48 elements of O_h, (P v)_a = s_a v_π(a),
//!   numbered in the enumeration order of §3.12 ("The cube symmetry group").
//! - [`class_of`] assigns each of the 316 V-list offsets its class and group element
//!   under the rule of §3.12 ("Symmetry classes"): the class of the representative
//!   with 0 ≤ d_x ≤ d_y ≤ d_z, and the first element P with P · representative = d.
//! - [`CoefficientTransform`] holds the per-degree blocks of T_M(P) = K Dⁿ(P) K or
//!   T_L(P) = K S Dⁿ(P) S⁻¹ K (§3.12, "Coefficients under improper P"), the map of
//!   multipole or local coefficients under sources y ↦ c + P(y − c).
//!
//! With these, M2L(P d) = T_L(P) M2L(d) T_M(Pᵀ) for every P in O_h (§3.12, "Operator
//! identities"), so the 16 class matrices and the transforms carry all 316 M2L tables
//! ([`M2lClasses`](crate::m2l::M2lClasses)). The same transforms map the M2M and L2L
//! tables of octant 0 to those of every octant; the octant tables stay dense.
//!
//! ```
//! use nd_fmm_tables::geometry::{m2l_offset_index, m2l_offsets};
//! use nd_fmm_tables::symmetry::{SignedPermutation, class_of, class_representatives};
//!
//! let all = SignedPermutation::all();
//! assert_eq!(all[7].apply([1, 2, 3]), [-1, -2, -3]); // g = 7 is −I
//!
//! let index = m2l_offset_index([-2, 3, 0]).unwrap();
//! let (class, element) = class_of(index);
//! assert_eq!(class_representatives()[class], [0, 2, 3]);
//! assert_eq!(element.apply([0, 2, 3]), m2l_offsets()[index]);
//! ```

use nd_fmm_math::rotation::{block_range, blocks, blocks_len, to_irregular};
use nd_fmm_math::{Layout, RealScalar};

use crate::geometry::{M2L_OFFSET_COUNT, m2l_offsets};

/// Number of elements of the cube group O_h.
pub const GROUP_ORDER: usize = 48;

/// Number of symmetry classes of the 316 V-list offsets under O_h (CONVENTIONS §3.12,
/// "Symmetry classes").
pub const M2L_CLASS_COUNT: usize = 16;

/// The six permutations of (0, 1, 2) in lexicographic order; permutation k is that of
/// the elements 8k to 8k + 7 (CONVENTIONS §3.12, "The cube symmetry group").
const PERMUTATIONS: [[u8; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// One element P of the cube symmetry group O_h: a permutation π of the axes
/// (0, 1, 2) = (x, y, z) and signs s ∈ {±1}³, acting as (P v)_a = s_a v_π(a)
/// (CONVENTIONS §3.12, "The cube symmetry group").
///
/// As a matrix, P has the entry s_a in row a and column π(a), and zeros elsewhere.
/// det P = sgn(π) s_x s_y s_z; 24 elements are proper (the rotations of the cube) and
/// 24 improper, among them −I. Element g, 0 ≤ g < 48, of the enumeration order is
///
/// g = 8k + 4 \[s_x < 0\] + 2 \[s_y < 0\] + \[s_z < 0\],
///
/// with k the position of (π(0), π(1), π(2)) in the lexicographic list of the six
/// permutations; so g = 0 is I, g = 7 is −I, and the elements g < 8 are diagonal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SignedPermutation {
    /// π(a) for a = 0, 1, 2.
    permutation: [u8; 3],
    /// Whether s_a = −1, for a = 0, 1, 2.
    negative: [bool; 3],
}

impl SignedPermutation {
    /// The identity I, element 0.
    pub const IDENTITY: Self = Self::from_index(0);

    /// Returns the 48 elements of O_h in the enumeration order of CONVENTIONS §3.12:
    /// element g is at position g.
    pub const fn all() -> [Self; GROUP_ORDER] {
        let mut all = [Self::IDENTITY; GROUP_ORDER];
        let mut g = 0;
        while g < GROUP_ORDER {
            all[g] = Self::from_index(g);
            g += 1;
        }
        all
    }

    /// Returns element `g` of the enumeration order (type documentation).
    ///
    /// # Panics
    ///
    /// If `g >= 48`.
    pub const fn from_index(g: usize) -> Self {
        assert!(g < GROUP_ORDER, "group element index must be below 48");
        Self {
            permutation: PERMUTATIONS[g / 8],
            negative: [g & 4 != 0, g & 2 != 0, g & 1 != 0],
        }
    }

    /// Returns the position g of this element in the enumeration order, the inverse of
    /// [`SignedPermutation::from_index`].
    pub const fn index(&self) -> usize {
        let mut k = 0;
        while !same_permutation(PERMUTATIONS[k], self.permutation) {
            k += 1;
        }
        let [x, y, z] = self.negative;
        8 * k + 4 * (x as usize) + 2 * (y as usize) + (z as usize)
    }

    /// Returns the permutation π as (π(0), π(1), π(2)).
    pub const fn permutation(&self) -> [usize; 3] {
        let [a, b, c] = self.permutation;
        [a as usize, b as usize, c as usize]
    }

    /// Returns the signs (s_x, s_y, s_z).
    pub const fn signs(&self) -> [i64; 3] {
        let [x, y, z] = self.negative;
        [sign(x), sign(y), sign(z)]
    }

    /// Returns P d for an integer vector d: component a is s_a d_π(a).
    pub const fn apply(&self, d: [i64; 3]) -> [i64; 3] {
        let s = self.signs();
        let [a, b, c] = self.permutation();
        [s[0] * d[a], s[1] * d[b], s[2] * d[c]]
    }

    /// Returns P as a row-major 3 × 3 matrix, `m[a][b]` = P_ab, with entries 0 and ±1;
    /// the form expected by `nd_fmm_math::rotation::blocks`.
    pub fn matrix<T: RealScalar>(&self) -> [[T; 3]; 3] {
        let s = self.signs();
        let pi = self.permutation();
        core::array::from_fn(|a| {
            core::array::from_fn(|b| {
                if b == pi[a] {
                    T::from_f64(s[a] as f64)
                } else {
                    T::zero()
                }
            })
        })
    }

    /// Returns det P = sgn(π) s_x s_y s_z, +1 or −1.
    pub const fn det(&self) -> i64 {
        let [a, b, c] = self.permutation;
        // A permutation of three elements is even if and only if it is cyclic.
        let even = (a + 1) % 3 == b && (b + 1) % 3 == c;
        let s = self.signs();
        (if even { 1 } else { -1 }) * s[0] * s[1] * s[2]
    }

    /// Whether P is proper, det P = +1: a rotation of the cube.
    pub const fn is_proper(&self) -> bool {
        self.det() == 1
    }

    /// Returns Pᵀ = P⁻¹: the permutation π⁻¹, with the sign s_a moved to the axis π(a).
    pub const fn transpose(&self) -> Self {
        let mut permutation = [0; 3];
        let mut negative = [false; 3];
        let mut a = 0;
        while a < 3 {
            let b = self.permutation[a] as usize;
            permutation[b] = a as u8;
            negative[b] = self.negative[a];
            a += 1;
        }
        Self {
            permutation,
            negative,
        }
    }

    /// Returns the product P Q, with `self` = P and `other` = Q, so that
    /// (P Q) d = P (Q d): (P Q v)_a = s_a t_π(a) v_σ(π(a)) for Q = (σ, t).
    pub const fn compose(&self, other: &Self) -> Self {
        let mut permutation = [0; 3];
        let mut negative = [false; 3];
        let mut a = 0;
        while a < 3 {
            let b = self.permutation[a] as usize;
            permutation[a] = other.permutation[b];
            negative[a] = self.negative[a] != other.negative[b];
            a += 1;
        }
        Self {
            permutation,
            negative,
        }
    }

    /// Returns −P, which is proper when P is improper.
    const fn negate(&self) -> Self {
        let [x, y, z] = self.negative;
        Self {
            permutation: self.permutation,
            negative: [!x, !y, !z],
        }
    }
}

/// Whether two permutations are equal (a `const` comparison).
const fn same_permutation(a: [u8; 3], b: [u8; 3]) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2]
}

/// −1 if `negative`, +1 otherwise.
const fn sign(negative: bool) -> i64 {
    if negative { -1 } else { 1 }
}

/// Returns the representatives of the 16 symmetry classes in class order: the V-list
/// offsets with 0 ≤ d_x ≤ d_y ≤ d_z, in lexicographic order (CONVENTIONS §3.12,
/// "Symmetry classes").
pub const fn class_representatives() -> [[i64; 3]; M2L_CLASS_COUNT] {
    [
        [0, 0, 2],
        [0, 0, 3],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 2],
        [0, 2, 3],
        [0, 3, 3],
        [1, 1, 2],
        [1, 1, 3],
        [1, 2, 2],
        [1, 2, 3],
        [1, 3, 3],
        [2, 2, 2],
        [2, 2, 3],
        [2, 3, 3],
        [3, 3, 3],
    ]
}

/// Returns the class and the group element of the V-list offset with table index
/// `index` (CONVENTIONS §3.12, "Symmetry classes").
///
/// The class is that of the representative, the sorted absolute values of d. The
/// element is the first P in the enumeration order of [`SignedPermutation::all`] with
/// P · representative = d; it is unique up to the stabiliser of the representative, and
/// the rule makes the class form reproducible. Then
/// M2L(d) = T_L(P) M2L(representative) T_M(Pᵀ).
///
/// # Panics
///
/// If `index >= 316`.
pub fn class_of(index: usize) -> (usize, SignedPermutation) {
    assert!(
        index < M2L_OFFSET_COUNT,
        "M2L offset index {index} out of range for {M2L_OFFSET_COUNT} offsets"
    );
    let d = m2l_offsets()[index];
    let mut representative = d.map(i64::abs);
    representative.sort_unstable();
    let class = class_representatives()
        .iter()
        .position(|&r| r == representative)
        .expect("the sorted absolute values of a V-list offset are a representative");
    let element = SignedPermutation::all()
        .into_iter()
        .find(|e| e.apply(representative) == d)
        .expect("every offset is the image of its representative");
    (class, element)
}

/// Which coefficients a [`CoefficientTransform`] acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Expansion {
    /// Multipole coefficients: T_M(P) = K Dⁿ(P) K.
    Multipole,
    /// Local coefficients: T_L(P) = K S Dⁿ(P) S⁻¹ K.
    Local,
}

/// The transform of multipole coefficients, T_M(P) = K Dⁿ(P) K, or of local
/// coefficients, T_L(P) = K S Dⁿ(P) S⁻¹ K, of degree n ≤ p, for an element P of O_h
/// (CONVENTIONS §3.12, "Coefficients under improper P").
///
/// For sources y ↦ c + P(y − c), the coefficients of degree n of an expansion about c
/// become T(P) times the old ones, degree by degree. T_M and T_L are homomorphisms,
/// T(P)⁻¹ = T(Pᵀ) and T(−I) = diag((−1)ⁿ). Here Dⁿ(P) is the rotation block of §3.8
/// for proper P and (−1)ⁿ Dⁿ(−P) for improper P, K = diag(+1 for m ≥ 0, −1 for m < 0)
/// is conjugation in real storage (§3.11) and S = diag((n − |m|)! (n + |m|)!) (§3.8).
///
/// Storage: the blocks of degrees 0 to p, contiguous and row-major in the layout of
/// `nd_fmm_math::rotation::blocks`: degree n occupies `block_range(n)`, with rows and
/// columns in slot order m = −n to n. That is `blocks_len(p)` =
/// (p + 1)(2p + 1)(2p + 3)/3 reals, and [`CoefficientTransform::apply`] costs as many
/// multiply–adds, O(p³). The blocks are built in f64 and rounded entry by entry to `T`.
///
/// For the 16 elements with P e_z = ±e_z, the blocks are signed permutations of the
/// slots with entries 0 and ±1, and T_L(P) = T_M(P) (§3.12, "The z-axis elements").
#[derive(Clone, Debug, PartialEq)]
pub struct CoefficientTransform<T: RealScalar> {
    pub(crate) p: usize,
    pub(crate) blocks: Vec<T>,
}

impl<T: RealScalar> CoefficientTransform<T> {
    /// Builds T_M(P) (for [`Expansion::Multipole`]) or T_L(P) (for
    /// [`Expansion::Local`]) of degree ≤ `p` for `element` = P.
    ///
    /// The blocks are `nd_fmm_math::rotation::blocks` of P if P is proper, and of −P
    /// times (−1)ⁿ if P is improper; for T_L they are converted with `to_irregular` to
    /// S Dⁿ S⁻¹. Both are then multiplied by K on both sides, which negates entry
    /// (i, j) when exactly one of the orders i and j is negative. All in f64, then
    /// rounded to `T`. Allocates.
    pub fn new(p: usize, element: SignedPermutation, expansion: Expansion) -> Self {
        let (rotation, improper) = if element.is_proper() {
            (element, false)
        } else {
            (element.negate(), true)
        };
        let mut d = vec![0.0; blocks_len(p)];
        blocks(p, &rotation.matrix::<f64>(), &mut d);
        if improper {
            for n in (1..=p).step_by(2) {
                for v in &mut d[block_range(n)] {
                    *v = -*v;
                }
            }
        }
        if expansion == Expansion::Local {
            to_irregular(p, &mut d);
        }
        for n in 1..=p {
            let width = 2 * n + 1;
            for (at, v) in d[block_range(n)].iter_mut().enumerate() {
                // Rows and columns run over m = −n..n, so slot t has order t − n.
                let (i, j) = (at / width, at % width);
                if (i < n) != (j < n) {
                    *v = -*v;
                }
            }
        }
        Self {
            p,
            blocks: d.into_iter().map(T::from_f64).collect(),
        }
    }

    /// Builds T_M(P) = K Dⁿ(P) K of degree ≤ `p` for `element` = P
    /// ([`CoefficientTransform::new`] with [`Expansion::Multipole`]).
    pub fn multipole(p: usize, element: SignedPermutation) -> Self {
        Self::new(p, element, Expansion::Multipole)
    }

    /// Builds T_L(P) = K S Dⁿ(P) S⁻¹ K of degree ≤ `p` for `element` = P
    /// ([`CoefficientTransform::new`] with [`Expansion::Local`]).
    pub fn local(p: usize, element: SignedPermutation) -> Self {
        Self::new(p, element, Expansion::Local)
    }

    /// Returns the degree p.
    #[inline]
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the blocks of degrees 0 to p, `blocks_len(p)` reals in the layout of
    /// `nd_fmm_math::rotation::blocks` (type documentation).
    #[inline]
    pub fn blocks(&self) -> &[T] {
        &self.blocks
    }

    /// Returns the block of degree n, (2n + 1)² reals, row-major with rows and columns
    /// in slot order m = −n to n.
    ///
    /// # Panics
    ///
    /// If n > p.
    #[inline]
    pub fn block(&self, n: usize) -> &[T] {
        assert!(n <= self.p, "degree {n} above p = {}", self.p);
        &self.blocks[block_range(n)]
    }

    /// Adds T x to `out`, degree by degree: slot i of degree n gets
    /// Σⱼ Tⁿᵢⱼ xⱼ over the slots j of degree n, summed in increasing j and then added
    /// to `out`. Both have length (p + 1)² in the storage of CONVENTIONS §3.6.
    /// Accumulates and allocates nothing; costs `blocks_len(p)` multiply–adds.
    ///
    /// # Panics
    ///
    /// If `x` or `out` does not have length (p + 1)².
    pub fn apply(&self, x: &[T], out: &mut [T]) {
        let layout = Layout::new(self.p);
        assert_eq!(
            x.len(),
            layout.len(),
            "`x` must have length (p + 1)^2 = {}",
            layout.len()
        );
        assert_eq!(
            out.len(),
            layout.len(),
            "`out` must have length (p + 1)^2 = {}",
            layout.len()
        );
        for (n, range) in layout.degrees() {
            let block = &self.blocks[block_range(n)];
            let x = &x[range.clone()];
            for (o, row) in out[range].iter_mut().zip(block.chunks_exact(2 * n + 1)) {
                let sum = row
                    .iter()
                    .zip(x)
                    .fold(T::zero(), |acc, (&a, &v)| acc + a * v);
                *o = *o + sum;
            }
        }
    }

    /// Returns the same transform in precision `U`, every entry rounded to nearest.
    pub fn cast<U: RealScalar>(&self) -> CoefficientTransform<U> {
        CoefficientTransform {
            p: self.p,
            blocks: self
                .blocks
                .iter()
                .map(|&v| U::from_f64(v.to_f64()))
                .collect(),
        }
    }
}
