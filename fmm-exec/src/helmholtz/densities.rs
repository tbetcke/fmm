//! Complex densities of the Helmholtz family (Helmholtz CH3.0c; CONVENTIONS §3.15,
//! "Storage" and "Inputs and outputs"; docs/design/helmholtz-fmm-proposal.md §3.3, §6.3):
//! the path of an FMM, its leaf chunks and expansion sizes, the charges in and the split
//! potentials and gradients out, and the two leaf operators that read and write a leaf's
//! own chunks, P2M and L2P.
//!
//! # The path
//!
//! [`Densities`] is fixed for an FMM: the kind of k ([`WavenumberKind`]), the kind of the
//! charges ([`ChargeKind`], §3.15's real-charge flag) and the number m ≥ 1 of charge
//! vectors. Every size and layout below depends on these three only (proposal §2,
//! constraint 9).
//!
//! - **Columns.** A real charge vector is one real column, a complex one two adjacent real
//!   columns, Re q then Im q: c = 1 or 2 columns per vector, c m in all
//!   ([`Densities::columns`]). The output has two columns per vector, Re then Im, except
//!   on Yukawa's real path (Yukawa with real charges), which has one
//!   ([`Densities::output_columns`]).
//! - **Expansions** (§3.15, "Storage"): m per box, one vector after the other, each of
//!   `planes` (p + 1)² values in the layout of CONVENTIONS §3.6. A complex charge vector
//!   forms **one** complex expansion, its real plane then its imaginary plane, never two.
//!   Real charges form a real multipole for Yukawa and for real k, and a real local for
//!   Yukawa ([`Densities::multipole_planes`], [`Densities::local_planes`]). On the Yukawa
//!   path a complex vector is two real vectors: two real expansions, of Re q and of Im q,
//!   which is the same storage. The sizes are §3.15's table ([`Densities::multipole_size`],
//!   [`Densities::local_size`]).
//!
//! # The leaf chunks
//!
//! CONVENTIONS §3.15 ("Inputs and outputs") leaves the in-memory leaf chunks to CH3.0, not
//! versioned. They are §3.13's chunks of several charge vectors, with the columns as the
//! vectors:
//!
//! ```text
//! source chunk     u_0 … u_(n−1)  then per point j its c m strengths:
//!                  q_j^0 … q_j^(m−1)                            (real charges, c = 1)
//!                  Re q_j^0  Im q_j^0  …  Re q_j^(m−1)  Im q_j^(m−1)   (complex, c = 2)
//! target output    the output columns' φ̂ one after the other, then their ĝ triples:
//!                  Re φ̂^0  Im φ̂^0  …  Re ĝ^0  Im ĝ^0  …            (two columns per vector)
//!                  φ̂^0 … φ̂^(m−1)  ĝ^0 … ĝ^(m−1)                  (Yukawa's real path)
//! target input     u_0 … u_(n−1), as §3.13
//! ```
//!
//! Point j's strength of vector v, part `part` (0 Re, 1 Im), is at 3n + j c m + c v + part
//! ([`Densities::charge_index`]); output column o's φ̂ of point i at o n + i, and its ĝ at
//! o_m n + 3 (o n + i) with o_m the output columns ([`Densities::potential_index`],
//! [`Densities::gradient_index`]); column o = 2v + part, or v on Yukawa's real path
//! ([`Densities::output_column`]). So:
//!
//! - every chunk is a §3.13 chunk of c m source and o_m output vectors, value for value,
//!   and §3.13's charge load and output pass carry over with m replaced by the columns;
//! - §3.15's output order, per vector Re φ then Im φ, and Re ∇φ triples then Im ∇φ
//!   triples, is the order of the columns;
//! - Yukawa with complex charges has, value for value, the chunks, expansions and output
//!   of Yukawa with the 2m real vectors Re q⁰, Im q⁰, …, Re q^(m−1), Im q^(m−1).
//!
//! # Scaling
//!
//! As for Laplace (§3.13, "Leaf-scaled coordinates" and "Output"), a leaf's points are in
//! the units of the leaf, u = (x − c)/r, and its operators work in its own frame (centre 0,
//! radius 1) at the scaled wavenumber k̂ = k r, formed in f64 and rounded to T once:
//! e^{ik|x−y|}/|x − y| = (1/r) e^{ik̂|u−v|}/|u − v|, so a chunk holds φ̂ = r φ̃ and
//! ĝ = r² ∇φ̃ (φ̃ = 4πφ, §3.15 "Where the factors go"), and the output pass divides by
//! Laplace's scales 4π r and 4π r². The stored form's power scale and damping depend on
//! k a only (`nd_fmm_ref::helmholtz::Level`), so they are the level's in either unit, to
//! rounding: s and β formed from k̂ are not bit for bit those formed from k and r, so the
//! translations of a level (CH3.1) form them from the same k̂.
//!
//! # Operators
//!
//! [`Densities::p2m`] and [`Densities::l2p`] are the operators of a leaf's own frame:
//! each source or target is one call of the reference operator of
//! `nd_fmm_ref::helmholtz::leaf` (the complex path, or the `_yukawa` forms on the Yukawa
//! path, column by column for complex vectors), in point order, accumulated (+=), so each
//! vector gets exactly the one-vector operator. Their scratch is a [`LeafScratch`], sized
//! at construction. The operators that need a second box's frame (P2L, M2P, P2P) belong to
//! the Helmholtz operator (CH3.1).

use core::f64::consts::PI;

use nd_fmm_math::{Complex, Layout, RealScalar};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf};
use thiserror::Error;

use super::{Wavenumber, WavenumberKind};
use crate::operator::SimdScalar;

/// Whether an FMM's charges are real or complex: the real-charge flag of CONVENTIONS
/// §3.15 ("Inputs and outputs").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ChargeKind {
    /// Real charges: one real column per vector.
    Real,
    /// Complex charges: two real columns per vector, Re q then Im q.
    Complex,
}

/// The path of a Helmholtz FMM: the kind of k, the kind of the charges and the number of
/// charge vectors, and the sizes and chunk layouts they fix ([module documentation](self)).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Densities {
    kind: WavenumberKind,
    charges: ChargeKind,
    vectors: usize,
}

/// Why charges do not fit a [`Densities`].
#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum DensityError {
    /// Not one entry per charge vector.
    #[error("expected {expected} charge vectors, got {got}")]
    VectorCount {
        /// The vectors of the path.
        expected: usize,
        /// The vectors given.
        got: usize,
    },
    /// A vector of the other charge kind.
    #[error("charge vector {vector} is not of the path's charge kind")]
    ChargeKind {
        /// The vector.
        vector: usize,
    },
    /// A plane without one entry per source.
    #[error("charge vector {vector} has {got} entries in a plane, expected {expected}")]
    Length {
        /// The vector.
        vector: usize,
        /// The number of sources.
        expected: usize,
        /// The plane's length.
        got: usize,
    },
}

/// One charge vector at the sources, as CONVENTIONS §3.15 ("Inputs and outputs") gives it:
/// Re q alone, or the two planes (Re q, Im q).
#[derive(Clone, Copy, Debug)]
pub enum Charges<'a, T> {
    /// Real charges.
    Real(&'a [T]),
    /// Complex charges as two planes of equal length.
    Complex {
        /// Re q.
        re: &'a [T],
        /// Im q.
        im: &'a [T],
    },
}

impl<T> Charges<'_, T> {
    /// The kind of the charges.
    pub fn kind(&self) -> ChargeKind {
        match self {
            Self::Real(_) => ChargeKind::Real,
            Self::Complex { .. } => ChargeKind::Complex,
        }
    }
}

impl Densities {
    /// The path of k's `kind`, `charges` and m = `vectors` charge vectors.
    ///
    /// # Panics
    ///
    /// If `vectors` is 0.
    pub fn new(kind: WavenumberKind, charges: ChargeKind, vectors: usize) -> Self {
        assert!(vectors > 0, "an FMM needs at least one charge vector");
        Self {
            kind,
            charges,
            vectors,
        }
    }

    /// The kind of k.
    pub fn kind(&self) -> WavenumberKind {
        self.kind
    }

    /// The kind of the charges.
    pub fn charges(&self) -> ChargeKind {
        self.charges
    }

    /// The number m of charge vectors.
    pub fn vectors(&self) -> usize {
        self.vectors
    }

    /// Whether this is Yukawa's real path: Yukawa with real charges, real throughout.
    fn yukawa_real(&self) -> bool {
        self.kind == WavenumberKind::Yukawa && self.charges == ChargeKind::Real
    }

    /// Real columns per charge vector: 1 for real charges, 2 for complex ones.
    fn charge_planes(&self) -> usize {
        match self.charges {
            ChargeKind::Real => 1,
            ChargeKind::Complex => 2,
        }
    }

    /// Output columns per charge vector: 1 on Yukawa's real path, 2 otherwise.
    fn output_planes(&self) -> usize {
        if self.yukawa_real() { 1 } else { 2 }
    }

    /// The real charge columns, c m: one per real vector, two (Re q, Im q) per complex one.
    pub fn columns(&self) -> usize {
        self.charge_planes() * self.vectors
    }

    /// The real output columns: m on Yukawa's real path, else 2m (Re, Im per vector).
    pub fn output_columns(&self) -> usize {
        self.output_planes() * self.vectors
    }

    /// The (p + 1)² planes of one vector's multipole: 1 for real charges with Yukawa or
    /// real k, else 2 (one complex expansion, or Yukawa's two real ones).
    pub fn multipole_planes(&self) -> usize {
        match (self.kind, self.charges) {
            (WavenumberKind::Yukawa | WavenumberKind::Real, ChargeKind::Real) => 1,
            _ => 2,
        }
    }

    /// The (p + 1)² planes of one vector's local: 1 on Yukawa's real path, else 2.
    pub fn local_planes(&self) -> usize {
        self.output_planes()
    }

    /// The multipole values of a box at degree p, all vectors: §3.15's table,
    /// `multipole_planes` (p + 1)² m.
    pub fn multipole_size(&self, p: usize) -> usize {
        self.multipole_planes() * Layout::new(p).len() * self.vectors
    }

    /// The local values of a box at degree p, all vectors: §3.15's table,
    /// `local_planes` (p + 1)² m.
    pub fn local_size(&self, p: usize) -> usize {
        self.local_planes() * Layout::new(p).len() * self.vectors
    }

    /// The values per source point of a source chunk: 3 + c m.
    pub fn source_point_size(&self) -> usize {
        3 + self.columns()
    }

    /// The values per target point of a target output chunk: the output columns, times 4
    /// with gradients.
    pub fn target_output_point_size(&self, gradients: bool) -> usize {
        let per_column = if gradients { 4 } else { 1 };
        per_column * self.output_columns()
    }

    /// The index in a source chunk of n points of point j's strength of vector v, part
    /// `part` (0 Re q, 1 Im q): 3n + j c m + c v + part.
    ///
    /// # Panics
    ///
    /// If v ≥ m, or `part` is not 0 for real charges or not below 2.
    pub fn charge_index(&self, n: usize, j: usize, v: usize, part: usize) -> usize {
        let c = self.charge_planes();
        assert!(v < self.vectors && part < c, "vector {v}, part {part}");
        3 * n + j * self.columns() + c * v + part
    }

    /// The output column of vector v, part `part` (0 Re, 1 Im): 2v + part, or v on
    /// Yukawa's real path.
    ///
    /// # Panics
    ///
    /// If v ≥ m, or `part` is not 0 on Yukawa's real path or not below 2.
    pub fn output_column(&self, v: usize, part: usize) -> usize {
        let c = self.output_planes();
        assert!(v < self.vectors && part < c, "vector {v}, part {part}");
        c * v + part
    }

    /// The index in a target output chunk of n points of output column o = `column`'s φ̂
    /// at point i: o n + i.
    ///
    /// # Panics
    ///
    /// If `column` is not below [`output_columns`](Self::output_columns) or i ≥ n.
    pub fn potential_index(&self, n: usize, column: usize, i: usize) -> usize {
        assert!(
            column < self.output_columns() && i < n,
            "column {column}, point {i}"
        );
        column * n + i
    }

    /// The index in a target output chunk of n points with gradients of output column
    /// o = `column`'s ĝ at point i, component `axis`: o_m n + 3 (o n + i) + axis, o_m the
    /// [`output_columns`](Self::output_columns).
    ///
    /// # Panics
    ///
    /// If `column` is not below the output columns, i ≥ n or `axis` > 2.
    pub fn gradient_index(&self, n: usize, column: usize, i: usize, axis: usize) -> usize {
        assert!(
            column < self.output_columns() && i < n && axis < 3,
            "column {column}, point {i}, axis {axis}"
        );
        self.output_columns() * n + 3 * (column * n + i) + axis
    }

    /// Checks `vectors` against the path for n sources: m vectors, each of the path's
    /// charge kind, every plane of length n.
    ///
    /// # Errors
    ///
    /// The first failure, in that order: [`DensityError::VectorCount`],
    /// [`DensityError::ChargeKind`], [`DensityError::Length`].
    pub fn check_charges<T>(
        &self,
        vectors: &[Charges<'_, T>],
        n: usize,
    ) -> Result<(), DensityError> {
        if vectors.len() != self.vectors {
            return Err(DensityError::VectorCount {
                expected: self.vectors,
                got: vectors.len(),
            });
        }
        for (vector, charges) in vectors.iter().enumerate() {
            if charges.kind() != self.charges {
                return Err(DensityError::ChargeKind { vector });
            }
            let lengths = match charges {
                Charges::Real(q) => [q.len(), n],
                Charges::Complex { re, im } => [re.len(), im.len()],
            };
            if let Some(&got) = lengths.iter().find(|&&len| len != n) {
                return Err(DensityError::Length {
                    vector,
                    expected: n,
                    got,
                });
            }
        }
        Ok(())
    }

    /// Writes the strengths of a leaf's points into `strengths`, the part of its source
    /// chunk after the coordinate triples: the leaf's point j is source `points[j]` of
    /// `vectors`, and its c m strengths are copied in the order of
    /// [`charge_index`](Self::charge_index).
    ///
    /// # Panics
    ///
    /// If `strengths.len() != c m · points.len()`, or `vectors` does not have m entries of
    /// the path's kind each covering every index of `points`
    /// ([`check_charges`](Self::check_charges)).
    pub fn write_leaf_charges<T: Copy>(
        &self,
        vectors: &[Charges<'_, T>],
        points: &[usize],
        strengths: &mut [T],
    ) {
        assert_eq!(vectors.len(), self.vectors, "charge vectors");
        assert_eq!(strengths.len(), self.columns() * points.len(), "strengths");
        for (point, &source) in strengths.chunks_exact_mut(self.columns()).zip(points) {
            for (v, charges) in vectors.iter().enumerate() {
                match (charges, self.charges) {
                    (Charges::Real(q), ChargeKind::Real) => point[v] = q[source],
                    (Charges::Complex { re, im }, ChargeKind::Complex) => {
                        point[2 * v] = re[source];
                        point[2 * v + 1] = im[source];
                    }
                    _ => panic!("charge vector {v} is not of the path's charge kind"),
                }
            }
        }
    }

    /// P2M in a leaf's own frame: adds the m multipoles of degree p of the leaf's n
    /// sources to `multipoles` ([`multipole_size`](Self::multipole_size)`(p)` values),
    /// for the wavenumber `k` and the leaf's half-width `radius` (module documentation,
    /// "Scaling" and "Operators"). `chunk` is the leaf's source chunk, at least
    /// n [`source_point_size`](Self::source_point_size) values.
    ///
    /// Per vector: a complex vector forms one complex expansion from both planes
    /// (`nd_fmm_ref::helmholtz::leaf::p2m`), real charges for complex k a complex one with
    /// Im q = 0; real charges for real k the real plane of that operator, summed over the
    /// sources in scratch and then added (its imaginary plane is exactly zero); Yukawa
    /// `p2m_yukawa`, per column for a complex vector.
    ///
    /// # Panics
    ///
    /// If `k`'s kind is not the path's, `radius` is not finite and positive, k̂ = k r
    /// rounded to T is zero or not finite (`nd_fmm_ref::helmholtz::Level::new`), `scratch`
    /// was built for a smaller degree than p, `chunk` is too short or `multipoles` has
    /// another length.
    #[allow(clippy::too_many_arguments)]
    pub fn p2m<T: SimdScalar>(
        &self,
        k: &Wavenumber,
        radius: f64,
        p: usize,
        chunk: &[T],
        n: usize,
        scratch: &mut LeafScratch<T>,
        multipoles: &mut [T],
    ) {
        let (level, frame) = self.leaf_level(k, radius);
        let len = Layout::new(p).len();
        assert!(scratch.p >= p, "scratch for degree {}, not {p}", scratch.p);
        assert_eq!(multipoles.len(), self.multipole_size(p), "multipoles");
        let (coordinates, strengths) = chunk.split_at(3 * n);
        let strengths = &strengths[..self.columns() * n];
        let points = || coordinates.as_chunks::<3>().0.iter().copied();
        let ws = &mut scratch.workspace;
        let size = self.multipole_planes() * len;
        for (v, multipole) in multipoles.chunks_exact_mut(size).enumerate() {
            let strength = |j: usize, part: usize| strengths[self.charge_index(0, j, v, part)];
            match (self.kind, self.charges) {
                (WavenumberKind::Yukawa, ChargeKind::Real) => {
                    for (j, u) in points().enumerate() {
                        let q = [strength(j, 0)];
                        leaf::p2m_yukawa(p, &level, &frame, &[u], &q, ws, multipole);
                    }
                }
                (WavenumberKind::Yukawa, ChargeKind::Complex) => {
                    let (re, im) = multipole.split_at_mut(len);
                    for (j, u) in points().enumerate() {
                        let (q_re, q_im) = ([strength(j, 0)], [strength(j, 1)]);
                        leaf::p2m_yukawa(p, &level, &frame, &[u], &q_re, ws, re);
                        leaf::p2m_yukawa(p, &level, &frame, &[u], &q_im, ws, im);
                    }
                }
                (WavenumberKind::Real, ChargeKind::Real) => {
                    let sum = &mut scratch.complex[..2 * len];
                    sum.fill(T::zero());
                    for (j, u) in points().enumerate() {
                        let q = [Complex::new(strength(j, 0), T::zero())];
                        leaf::p2m(p, &level, &frame, &[u], &q, ws, sum);
                    }
                    for (m, s) in multipole.iter_mut().zip(&sum[..len]) {
                        *m = *m + *s;
                    }
                }
                (_, charges) => {
                    for (j, u) in points().enumerate() {
                        let q = match charges {
                            ChargeKind::Real => Complex::new(strength(j, 0), T::zero()),
                            ChargeKind::Complex => Complex::new(strength(j, 0), strength(j, 1)),
                        };
                        leaf::p2m(p, &level, &frame, &[u], &[q], ws, multipole);
                    }
                }
            }
        }
    }

    /// L2P in a leaf's own frame: adds the potentials of the m locals of degree p
    /// (`locals`, [`local_size`](Self::local_size)`(p)` values), and with `gradients` their
    /// gradients, at the leaf's targets `targets` (its target input chunk, leaf-scaled)
    /// into its target output chunk `output`, for the wavenumber `k` and the leaf's
    /// half-width `radius` (module documentation, "Scaling" and "Operators").
    ///
    /// Per vector: `nd_fmm_ref::helmholtz::leaf::l2p` for a complex local, its real and
    /// imaginary parts into the vector's two output columns; Yukawa's real path
    /// `l2p_yukawa`, and for a complex vector on the Yukawa path `l2p_yukawa` per column.
    ///
    /// # Panics
    ///
    /// If `k`'s kind is not the path's, `radius` is not finite and positive, k̂ = k r
    /// rounded to T is zero or not finite (`nd_fmm_ref::helmholtz::Level::new`), `scratch`
    /// was built for a smaller degree than p, `locals` has another length, `targets` is not a
    /// whole number of triples or `output` does not have
    /// [`target_output_point_size`](Self::target_output_point_size) values per target.
    #[allow(clippy::too_many_arguments)]
    pub fn l2p<T: SimdScalar>(
        &self,
        k: &Wavenumber,
        radius: f64,
        p: usize,
        locals: &[T],
        targets: &[T],
        scratch: &mut LeafScratch<T>,
        output: &mut [T],
        gradients: bool,
    ) {
        let (level, frame) = self.leaf_level(k, radius);
        let len = Layout::new(p).len();
        assert!(scratch.p >= p, "scratch for degree {}, not {p}", scratch.p);
        assert_eq!(locals.len(), self.local_size(p), "locals");
        assert_eq!(targets.len() % 3, 0, "target triples");
        let n = targets.len() / 3;
        assert_eq!(
            output.len(),
            n * self.target_output_point_size(gradients),
            "target output"
        );
        let points = || targets.as_chunks::<3>().0.iter().copied();
        let ws = &mut scratch.workspace;
        // Adds a column's potential and gradient at point i.
        let mut add = |column: usize, i: usize, phi: T, g: [T; 3]| {
            let at = self.potential_index(n, column, i);
            output[at] = output[at] + phi;
            if gradients {
                for (axis, &ga) in g.iter().enumerate() {
                    let at = self.gradient_index(n, column, i, axis);
                    output[at] = output[at] + ga;
                }
            }
        };
        let size = self.local_planes() * len;
        for (v, local) in locals.chunks_exact(size).enumerate() {
            if self.kind == WavenumberKind::Yukawa {
                // One real local per column: Re q's (and Im q's for a complex vector).
                for (part, plane) in local.chunks_exact(len).enumerate() {
                    let column = self.output_column(v, part);
                    for (i, u) in points().enumerate() {
                        let (mut phi, mut g) = ([T::zero()], [[T::zero(); 3]]);
                        let grad = gradients.then_some(&mut g[..]);
                        leaf::l2p_yukawa(p, &level, &frame, plane, &[u], ws, &mut phi, grad);
                        add(column, i, phi[0], g[0]);
                    }
                }
            } else {
                let zero = Complex::new(T::zero(), T::zero());
                let (re, im) = (self.output_column(v, 0), self.output_column(v, 1));
                for (i, u) in points().enumerate() {
                    let (mut phi, mut g) = ([zero], [[zero; 3]]);
                    let grad = gradients.then_some(&mut g[..]);
                    leaf::l2p(p, &level, &frame, local, &[u], ws, &mut phi, grad);
                    add(re, i, phi[0].re, g[0].map(|c| c.re));
                    add(im, i, phi[0].im, g[0].map(|c| c.im));
                }
            }
        }
    }

    /// The level and frame of a leaf of half-width `radius` in its own units: k̂ = k r in
    /// f64, rounded to T; centre 0, radius 1.
    fn leaf_level<T: SimdScalar>(&self, k: &Wavenumber, radius: f64) -> (Level<T>, Frame<T>) {
        assert_eq!(
            k.kind(),
            self.kind,
            "the wavenumber's kind is not the path's"
        );
        assert!(
            radius.is_finite() && radius > 0.0,
            "the leaf's half-width must be finite and positive, got {radius}"
        );
        let k_hat = Complex::new(T::from_f64(k.re() * radius), T::from_f64(k.im() * radius));
        (
            Level::new(k_hat, T::one()),
            Frame::new([T::zero(); 3], T::one()),
        )
    }

    /// The output pass of one leaf (module documentation, "Scaling"): its target output
    /// chunk `chunk` of n = `targets.len()` points, with or without gradients, written
    /// into `output` at the caller's target indices `targets`, each value
    /// `T::from_f64(x.to_f64() / scale)` with the scales 4π r (φ) and 4π r² (∇φ) of the
    /// leaf's half-width r = `radius`, as Laplace's output pass (CONVENTIONS §3.13,
    /// "Output"). Output column o of point i goes to column o of `output` at
    /// `targets[i]`; overwritten, not accumulated.
    ///
    /// # Panics
    ///
    /// If `output` is not of this path (vectors and columns) or has no gradients where
    /// `gradients` asks for them, `chunk` has another length or a target index is out of
    /// range.
    pub fn write_leaf_output<T: SimdScalar>(
        &self,
        chunk: &[T],
        radius: f64,
        gradients: bool,
        targets: &[usize],
        output: &mut Output<T>,
    ) {
        assert_eq!(
            (output.vectors, output.planes),
            (self.vectors, self.output_planes()),
            "the output is not of this path"
        );
        let n = targets.len();
        assert_eq!(
            chunk.len(),
            n * self.target_output_point_size(gradients),
            "target output"
        );
        let size = output.ntargets();
        let (phi_scale, g_scale) = (4.0 * PI * radius, 4.0 * PI * radius * radius);
        let scaled = |x: T, scale: f64| T::from_f64(RealScalar::to_f64(x) / scale);
        for column in 0..self.output_columns() {
            for (i, &t) in targets.iter().enumerate() {
                assert!(t < size, "target {t} of {size}");
                output.potential[column * size + t] =
                    scaled(chunk[self.potential_index(n, column, i)], phi_scale);
            }
        }
        if gradients {
            let gradient = output
                .gradient
                .as_mut()
                .expect("the output has no gradients");
            for column in 0..self.output_columns() {
                for (i, &t) in targets.iter().enumerate() {
                    gradient[column * size + t] = core::array::from_fn(|axis| {
                        scaled(chunk[self.gradient_index(n, column, i, axis)], g_scale)
                    });
                }
            }
        }
    }
}

/// The scratch of [`Densities::p2m`] and [`Densities::l2p`] up to a degree, allocated once.
pub struct LeafScratch<T: SimdScalar> {
    p: usize,
    workspace: Workspace<T>,
    /// One complex expansion, for the real multipole of real k.
    complex: Vec<T>,
}

impl<T: SimdScalar> LeafScratch<T> {
    /// The scratch of operators of degree at most `p`.
    pub fn new(p: usize) -> Self {
        Self {
            p,
            workspace: Workspace::new(p),
            complex: vec![T::zero(); 2 * Layout::new(p).len()],
        }
    }
}

/// Part of an output: the real plane, and the imaginary one where the path has it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Split<'a, U> {
    /// The real parts, one per target.
    pub re: &'a [U],
    /// The imaginary parts, one per target; `None` on Yukawa's real path.
    pub im: Option<&'a [U]>,
}

/// The potentials φ and gradients ∇φ of a Helmholtz FMM at its N targets (CONVENTIONS
/// §3.15, "Kernel and factors" and "Inputs and outputs"; 1/(4π) included), in split
/// planes: per vector v the real parts over the targets, then the imaginary parts (none on
/// Yukawa's real path); with gradients the Re ∇φ triples, then the Im ∇φ triples. So
/// output column o = 2v + part (v on Yukawa's real path) is `potential[o N .. (o + 1) N]`
/// and likewise `gradient`.
#[derive(Clone, Debug, PartialEq)]
pub struct Output<T> {
    vectors: usize,
    planes: usize,
    /// φ by output column, N values each.
    pub potential: Vec<T>,
    /// ∇φ by output column, N triples each, if the FMM has gradients.
    pub gradient: Option<Vec<[T; 3]>>,
}

impl<T: SimdScalar> Output<T> {
    /// The zero output of `densities` at `ntargets` targets, with gradients if asked.
    pub fn new(densities: &Densities, ntargets: usize, gradients: bool) -> Self {
        let len = densities.output_columns() * ntargets;
        Self {
            vectors: densities.vectors(),
            planes: densities.output_planes(),
            potential: vec![T::zero(); len],
            gradient: gradients.then(|| vec![[T::zero(); 3]; len]),
        }
    }
}

impl<T> Output<T> {
    /// The number of charge vectors m.
    pub fn vectors(&self) -> usize {
        self.vectors
    }

    /// Whether the output has imaginary planes: all but Yukawa's real path.
    pub fn is_complex(&self) -> bool {
        self.planes == 2
    }

    /// The number of targets N.
    pub fn ntargets(&self) -> usize {
        self.potential.len() / (self.planes * self.vectors)
    }

    /// The potentials of vector `v`.
    ///
    /// # Panics
    ///
    /// If `v` is not below [`vectors`](Self::vectors).
    pub fn potential_of(&self, v: usize) -> Split<'_, T> {
        self.split(&self.potential, v)
    }

    /// The gradients of vector `v`, if the output has gradients.
    ///
    /// # Panics
    ///
    /// As [`potential_of`](Self::potential_of).
    pub fn gradient_of(&self, v: usize) -> Option<Split<'_, [T; 3]>> {
        self.gradient.as_deref().map(|g| self.split(g, v))
    }

    /// Vector v's planes of `values`.
    fn split<'a, U>(&self, values: &'a [U], v: usize) -> Split<'a, U> {
        assert!(v < self.vectors, "vector {v} of {}", self.vectors);
        let n = self.ntargets();
        let column = |o: usize| &values[o * n..(o + 1) * n];
        Split {
            re: column(self.planes * v),
            im: (self.planes == 2).then(|| column(2 * v + 1)),
        }
    }
}
