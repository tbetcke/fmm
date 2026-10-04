//! The translation tables of the Laplace operator: an M2L strategy, the table families
//! it needs, and their application by child index and offset index (CONVENTIONS §3.11,
//! §3.12).
//!
//! [`Tables`] holds the tables of one resolved [`M2lStrategy`] at one degree p, and
//! nothing else:
//!
//! | Strategy | M2M, L2L | M2L | Cost of one M2L | Storage in f64 at p = 8 / 16 |
//! | --- | --- | --- | --- | --- |
//! | [`Dense`](M2lStrategy::Dense) | [`M2mTables`], [`L2lTables`] | [`M2lTables`] | (p + 1)⁴ | 17.4 MB / 222 MB |
//! | [`Classes`](M2lStrategy::Classes) | [`M2mTables`], [`L2lTables`] | [`M2lClasses`] | (p + 1)⁴ + O(p³) | 2.4 MB / 26.4 MB |
//! | [`Rotation`](M2lStrategy::Rotation) | [`RotationTables`] | [`RotationTables`] | ≈ (10/3) p³ | 0.8 MB / 5.6 MB |
//!
//! (The dense octant tables are 8 (p + 1)⁴ reals per family; the rest is from the
//! documentation of the families in `nd-fmm-tables`.)
//!
//! Every table is looked up by integer key, never by a shift: M2M and L2L by the child
//! index o = `morton::child_index(child)`, M2L by the position of the offset
//! d = index(target) − index(source) in `V_LIST_DIRECTIONS`, which is
//! `m2l_offset_index(d)` (§3.12). Each application accumulates into its output and
//! allocates nothing; the scratch a family needs is a [`TableScratch`], made once by
//! [`Tables::scratch`].
//!
//! Tables are built in f64 and rounded to `T` entry by entry, by the families' own
//! `build` (§3.12, "Matrix layout"). [`Tables::load_or_build`] goes through a
//! caller-supplied [`TableCache`] instead; no directory is used otherwise.
//!
//! ```
//! use nd_fmm_exec::tables::{M2lStrategy, Tables};
//!
//! let tables = Tables::<f64>::build(2, M2lStrategy::Auto);
//! assert_eq!(tables.strategy(), M2lStrategy::Dense);
//!
//! // The M2L of offset index 0, d = (−3, −3, −3), of a unit monopole.
//! let mut scratch = tables.scratch();
//! let mut multipole = vec![0.0; 9];
//! multipole[0] = 1.0;
//! let mut local = vec![0.0; 9];
//! tables.m2l(0, &multipole, &mut local, &mut scratch);
//! // The local monopole is 1/|b| with b = 2d, |b| = 6√3 (CONVENTIONS §3.11).
//! assert!((local[0] - 1.0 / (6.0 * 3f64.sqrt())).abs() < 1e-15);
//! ```

use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::{Stored, TableKind};
#[cfg(feature = "gpu")]
use nd_fmm_tables::rotation::{Operator, ShiftTables};
use nd_fmm_tables::{
    CacheOutcome, L2lTables, M2lClasses, M2lScratch, M2lTables, M2mTables, RotationScratch,
    RotationTables, TableCache,
};

/// How M2L, and with it M2M and L2L, is applied (docs/phase3/README.md, "Strategy").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum M2lStrategy {
    /// Dense octant tables for M2M and L2L and the dense M2L tables of the 316 offsets:
    /// one (p + 1)² × (p + 1)² product per translation.
    Dense,
    /// Dense octant tables, and the 16 class matrices of M2L with the coefficient
    /// transforms of the cube group (CONVENTIONS §3.12, "Symmetry classes"). It saves
    /// memory, not time: an M2L costs the dense product plus two O(p³) transforms.
    Classes,
    /// Rotation, coaxial translation and rotation back, from [`RotationTables`], for
    /// all three translations: O(p³) per translation.
    Rotation,
    /// [`Dense`](Self::Dense) for p ≤ [`AUTO_DENSE_MAX_P`](Self::AUTO_DENSE_MAX_P),
    /// [`Rotation`](Self::Rotation) above; see [`resolve`](Self::resolve).
    #[default]
    Auto,
}

impl M2lStrategy {
    /// The largest degree at which [`Auto`](Self::Auto) resolves to
    /// [`Dense`](Self::Dense).
    pub const AUTO_DENSE_MAX_P: usize = 8;

    /// Returns the strategy used at degree `p`: `self`, or for [`Auto`](Self::Auto)
    /// [`Dense`](Self::Dense) when p ≤ 8 and [`Rotation`](Self::Rotation) when p ≥ 9.
    /// Never returns `Auto`.
    ///
    /// *Source of the rule.* The Phase 2 recommendation for the per-pair CPU M2L
    /// (docs/design/laplace-fmm-plan.md §7, Phase 2 T8): the dense table is fastest
    /// for p ≤ 8 and table-driven rotation for the larger measured degrees, within
    /// about 20 % of each other in between; rotation also needs far less memory per
    /// rank (5.6 MB against 211 MB for the dense M2L tables at p = 16) and wins for M2M and L2L from p = 8. The
    /// timings were single-threaded, one product or one rotation per pair; the batched
    /// GEMM path of Phase 4 changes the comparison and will have its own rule.
    pub fn resolve(self, p: usize) -> Self {
        match self {
            Self::Auto if p <= Self::AUTO_DENSE_MAX_P => Self::Dense,
            Self::Auto => Self::Rotation,
            resolved => resolved,
        }
    }
}

/// The table families of one resolved strategy.
#[derive(Clone, Debug, PartialEq)]
enum Families<T: RealScalar> {
    Dense {
        m2m: M2mTables<T>,
        l2l: L2lTables<T>,
        m2l: M2lTables<T>,
    },
    Classes {
        m2m: M2mTables<T>,
        l2l: L2lTables<T>,
        m2l: M2lClasses<T>,
    },
    /// Boxed: the struct of three shift-table families is much larger than the others.
    Rotation(Box<RotationTables<T>>),
}

/// The translation tables of one strategy at one degree p; see the
/// [module documentation](self).
#[derive(Clone, Debug, PartialEq)]
pub struct Tables<T: RealScalar> {
    p: usize,
    families: Families<T>,
}

/// The scratch of a [`Tables`] application: none for [`M2lStrategy::Dense`], an
/// [`M2lScratch`] for [`M2lStrategy::Classes`] and a [`RotationScratch`] for
/// [`M2lStrategy::Rotation`], each 2 (p + 1)² reals. Made by [`Tables::scratch`].
#[derive(Clone, Debug)]
pub struct TableScratch<T: RealScalar> {
    kind: ScratchKind<T>,
}

#[derive(Clone, Debug)]
enum ScratchKind<T: RealScalar> {
    Dense,
    Classes(M2lScratch<T>),
    Rotation(RotationScratch<T>),
}

impl<T: RealScalar> Tables<T> {
    /// Builds the tables that `strategy`, resolved at `p` ([`M2lStrategy::resolve`]),
    /// needs, and no others. Serial and deterministic; allocates the tables.
    ///
    /// Building costs O(p⁶) for the dense families (8 (p + 1)² `direct` calls per
    /// octant family, 316 (p + 1)² for M2L, 16 (p + 1)² for the classes) and O(p⁴) for
    /// the rotation tables; at p = 16 the dense M2L tables take seconds, the others
    /// well under one.
    pub fn build(p: usize, strategy: M2lStrategy) -> Self {
        let families = match strategy.resolve(p) {
            M2lStrategy::Dense => Families::Dense {
                m2m: M2mTables::build(p),
                l2l: L2lTables::build(p),
                m2l: M2lTables::build(p),
            },
            M2lStrategy::Classes => Families::Classes {
                m2m: M2mTables::build(p),
                l2l: L2lTables::build(p),
                m2l: M2lClasses::build(p),
            },
            M2lStrategy::Rotation => Families::Rotation(Box::new(RotationTables::build(p))),
            M2lStrategy::Auto => unreachable!("resolve never returns Auto"),
        };
        Self { p, families }
    }

    /// Returns the degree p of every table.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the resolved strategy: [`M2lStrategy::Dense`],
    /// [`M2lStrategy::Classes`] or [`M2lStrategy::Rotation`], never `Auto`.
    pub fn strategy(&self) -> M2lStrategy {
        match self.families {
            Families::Dense { .. } => M2lStrategy::Dense,
            Families::Classes { .. } => M2lStrategy::Classes,
            Families::Rotation(_) => M2lStrategy::Rotation,
        }
    }

    /// Returns the table families held, in the order [`load_or_build`](Self::load_or_build)
    /// reports them.
    pub fn kinds(&self) -> &'static [TableKind] {
        match self.families {
            Families::Dense { .. } => &[TableKind::M2m, TableKind::L2l, TableKind::M2l],
            Families::Classes { .. } => &[TableKind::M2m, TableKind::L2l, TableKind::M2lClasses],
            Families::Rotation(_) => &[TableKind::Rotation],
        }
    }

    /// Allocates the scratch that [`m2m`](Self::m2m), [`l2l`](Self::l2l) and
    /// [`m2l`](Self::m2l) need with these tables.
    pub fn scratch(&self) -> TableScratch<T> {
        let kind = match self.families {
            Families::Dense { .. } => ScratchKind::Dense,
            Families::Classes { .. } => ScratchKind::Classes(M2lScratch::new(self.p)),
            Families::Rotation(_) => ScratchKind::Rotation(RotationScratch::new(self.p)),
        };
        TableScratch { kind }
    }

    /// Adds the M2M of the multipole `input` of the child with child index `o` to the
    /// multipole `output` of its parent (CONVENTIONS §3.11, §3.12): the matrix of `o`
    /// ((p + 1)⁴ multiply–adds), or the rotation entry of `o` (≈ 3p³).
    ///
    /// # Panics
    ///
    /// If `o >= 8`, a slice does not have length (p + 1)², or `scratch` was not made by
    /// these tables' [`scratch`](Self::scratch).
    #[inline]
    pub fn m2m(&self, o: usize, input: &[T], output: &mut [T], scratch: &mut TableScratch<T>) {
        match &self.families {
            Families::Dense { m2m, .. } | Families::Classes { m2m, .. } => {
                m2m.apply(o, input, output)
            }
            Families::Rotation(rotation) => {
                rotation.m2m(o, input, output, scratch.rotation());
            }
        }
    }

    /// Adds the L2L of the local `input` of a parent to the local `output` of its child
    /// with child index `o` (CONVENTIONS §3.11, §3.12): the matrix of `o` ((p + 1)⁴
    /// multiply–adds), or the rotation entry of `o` (≈ 3p³).
    ///
    /// # Panics
    ///
    /// As [`m2m`](Self::m2m).
    #[inline]
    pub fn l2l(&self, o: usize, input: &[T], output: &mut [T], scratch: &mut TableScratch<T>) {
        match &self.families {
            Families::Dense { l2l, .. } | Families::Classes { l2l, .. } => {
                l2l.apply(o, input, output)
            }
            Families::Rotation(rotation) => {
                rotation.l2l(o, input, output, scratch.rotation());
            }
        }
    }

    /// Adds the M2L of the source multipole `multipole` to the target local `local`, a
    /// V-list pair whose offset has the position `offset_index` in `V_LIST_DIRECTIONS`
    /// (CONVENTIONS §3.11, §3.12): the dense matrix ((p + 1)⁴ multiply–adds), the class
    /// matrix between two coefficient transforms ((p + 1)⁴ + O(p³)), or rotation
    /// (≈ (10/3) p³).
    ///
    /// # Panics
    ///
    /// If `offset_index >= 316`, a slice does not have length (p + 1)², or `scratch` was
    /// not made by these tables' [`scratch`](Self::scratch).
    #[inline]
    pub fn m2l(
        &self,
        offset_index: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut TableScratch<T>,
    ) {
        match &self.families {
            Families::Dense { m2l, .. } => m2l.apply(offset_index, multipole, local),
            Families::Classes { m2l, .. } => {
                m2l.apply(offset_index, multipole, local, scratch.classes());
            }
            Families::Rotation(rotation) => {
                rotation.m2l(offset_index, multipole, local, scratch.rotation());
            }
        }
    }
}

/// The families the device path uploads (device-path.md §3.3, §6.8): crate-private, so
/// the public surface of [`Tables`] does not change.
#[cfg(feature = "gpu")]
impl<T: RealScalar> Tables<T> {
    /// The dense octant tables of M2M and L2L, under `Dense` and `Classes`; `None` under
    /// `Rotation`, whose device path builds or loads them itself.
    pub(crate) fn octant_families(&self) -> Option<(&M2mTables<T>, &L2lTables<T>)> {
        match &self.families {
            Families::Dense { m2m, l2l, .. } | Families::Classes { m2m, l2l, .. } => {
                Some((m2m, l2l))
            }
            Families::Rotation(_) => None,
        }
    }

    /// The dense M2L tables, under `Dense`.
    pub(crate) fn dense_m2l(&self) -> Option<&M2lTables<T>> {
        match &self.families {
            Families::Dense { m2l, .. } => Some(m2l),
            _ => None,
        }
    }

    /// The class form of M2L, under `Classes`.
    pub(crate) fn classes_m2l(&self) -> Option<&M2lClasses<T>> {
        match &self.families {
            Families::Classes { m2l, .. } => Some(m2l),
            _ => None,
        }
    }

    /// The M2L family of the rotation tables, under `Rotation` (Phase 4 T10).
    pub(crate) fn rotation_m2l(&self) -> Option<&ShiftTables<T>> {
        match &self.families {
            Families::Rotation(rotation) => Some(rotation.tables(Operator::M2l)),
            _ => None,
        }
    }
}

impl<T: Stored> Tables<T> {
    /// Loads the tables that `strategy`, resolved at `p`, needs from `cache`, or builds
    /// and stores those that are missing or rejected ([`TableCache::load_or_build`]),
    /// and reports what happened to every family, in the order of
    /// [`kinds`](Self::kinds).
    ///
    /// Never fails because of the cache. Every rank of a run may call it on the same
    /// directory at once: stores are atomic (Phase 2 T7).
    pub fn load_or_build(
        p: usize,
        strategy: M2lStrategy,
        cache: &TableCache,
    ) -> (Self, Vec<(TableKind, CacheOutcome)>) {
        let (families, outcomes) = match strategy.resolve(p) {
            M2lStrategy::Dense => {
                let (m2m, m2m_outcome) = cache.load_or_build(p);
                let (l2l, l2l_outcome) = cache.load_or_build(p);
                let (m2l, m2l_outcome) = cache.load_or_build(p);
                (
                    Families::Dense { m2m, l2l, m2l },
                    vec![
                        (TableKind::M2m, m2m_outcome),
                        (TableKind::L2l, l2l_outcome),
                        (TableKind::M2l, m2l_outcome),
                    ],
                )
            }
            M2lStrategy::Classes => {
                let (m2m, m2m_outcome) = cache.load_or_build(p);
                let (l2l, l2l_outcome) = cache.load_or_build(p);
                let (m2l, m2l_outcome) = cache.load_or_build(p);
                (
                    Families::Classes { m2m, l2l, m2l },
                    vec![
                        (TableKind::M2m, m2m_outcome),
                        (TableKind::L2l, l2l_outcome),
                        (TableKind::M2lClasses, m2l_outcome),
                    ],
                )
            }
            M2lStrategy::Rotation => {
                let (rotation, outcome) = cache.load_or_build(p);
                (
                    Families::Rotation(Box::new(rotation)),
                    vec![(TableKind::Rotation, outcome)],
                )
            }
            M2lStrategy::Auto => unreachable!("resolve never returns Auto"),
        };
        (Self { p, families }, outcomes)
    }
}

impl<T: RealScalar> TableScratch<T> {
    /// The scratch of [`M2lClasses::apply`].
    fn classes(&mut self) -> &mut M2lScratch<T> {
        match &mut self.kind {
            ScratchKind::Classes(scratch) => scratch,
            _ => panic!("the table scratch does not belong to Classes tables"),
        }
    }

    /// The scratch of the [`RotationTables`] operators.
    fn rotation(&mut self) -> &mut RotationScratch<T> {
        match &mut self.kind {
            ScratchKind::Rotation(scratch) => scratch,
            _ => panic!("the table scratch does not belong to Rotation tables"),
        }
    }
}
