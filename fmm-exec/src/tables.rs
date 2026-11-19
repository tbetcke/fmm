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
//! | [`Batched`](M2lStrategy::Batched) | [`M2mTables`], [`L2lTables`] | [`M2lTables`] | (p + 1)⁴, per level as products over columns | 17.4 MB / 222 MB |
//! | [`PlaneWave`](M2lStrategy::PlaneWave) | [`RotationTables`] | [`PlaneWaveTables`] | per box, not per pair: about 31 shifts of H complex values and 12 Fourier passes | 0.8 MB + 0.7 MB / 5.6 MB + 2.1 MB |
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
//! // `Auto` at f64 p = 2: Batched, with Dense's tables (Phase 6 T11).
//! let tables = Tables::<f64>::build(2, M2lStrategy::Auto);
//! assert_eq!(tables.strategy(), M2lStrategy::Batched);
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

use std::fmt;

use nd_fmm_math::RealScalar;
use nd_fmm_tables::cache::{Stored, TableKind};
#[cfg(feature = "gpu")]
use nd_fmm_tables::rotation::{Operator, ShiftTables};
use nd_fmm_tables::{
    CacheOutcome, L2lTables, M2lClasses, M2lScratch, M2lTables, M2mTables, MatrixSet,
    PlaneWaveScratch, PlaneWaveTables, RotationScratch, RotationTables, TableCache,
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
    /// The tables of [`Dense`](Self::Dense), with M2L applied per level on the host as
    /// products of each offset's table with the columns of its pairs (Phase 6 T2, C6.6;
    /// docs/design/optimisation.md §2): each table is read once per block of targets
    /// instead of once per target, and every local receives the chain of `Dense` (offsets
    /// in index order, k ascending, multiply and add rounded apart), so the output is
    /// `Dense`'s bit for bit; with the opt-in fused mode (`FmmBuilder::m2l_mul_add`) it
    /// is within n u of it. M2M and L2L are `Dense`'s. The device runs it as `Dense`,
    /// whose dense M2L is already a product per level (docs/design/device-path.md §6.4).
    /// [`Auto`](Self::Auto)'s choice on the host at f64 p ≤ 6 and f32 p ≤ 7 (Phase 6 T11).
    Batched,
    /// M2L by plane waves with merge and shift (Phase 6 T13, C6.5; CONVENTIONS §3.14,
    /// `nd_fmm_tables::plane_wave`): each level call converts the multipoles it reads into
    /// waves in six directions, merges them per parent, shifts them to the children and
    /// converts them into locals, so its work is per box, not per V pair. An approximate
    /// operator with its own error budget: the quadrature of each degree keeps the FMM's
    /// errors against the direct sum within about 1.1× those of the exact M2L
    /// (`nd_fmm_tables::plane_wave::nodes_for_degree`); not `Dense`'s bits, but
    /// deterministic, the same at every thread count and overlap setting, and on P ranks
    /// the one-rank output bit for bit. M2M and L2L are `Rotation`'s. A device runs it by
    /// merge and shift on the device (`FmmBuilder::device_plane_wave`, default
    /// `DevicePlaneWave::Kernels` since Helmholtz CH7.5c), or as `Dense` with dense tables
    /// the device path builds under `DevicePlaneWave::Dense`. [`Auto`](Self::Auto)'s choice on
    /// the host at f64 p ≥ 7 and f32 p ≥ 8 (Phase 6 T11).
    PlaneWave,
    /// On the host [`Batched`](Self::Batched) below a degree that depends on the
    /// precision and [`PlaneWave`](Self::PlaneWave) from it (Phase 6 T11); see
    /// [`resolve_on_node`](Self::resolve_on_node).
    #[default]
    Auto,
}

impl M2lStrategy {
    /// The lowest degree at which [`Auto`](Self::Auto) resolves to
    /// [`PlaneWave`](Self::PlaneWave) on the host in f64 (Phase 6 T11): below it
    /// [`Batched`](Self::Batched).
    pub const AUTO_PLANE_WAVE_MIN_P_F64: usize = 7;

    /// The lowest degree at which [`Auto`](Self::Auto) resolves to
    /// [`PlaneWave`](Self::PlaneWave) on the host in f32 (Phase 6 T11): below it
    /// [`Batched`](Self::Batched).
    pub const AUTO_PLANE_WAVE_MIN_P_F32: usize = 8;

    /// The lowest degree at which [`Auto`](Self::Auto) resolves to
    /// [`PlaneWave`](Self::PlaneWave) for values of `value_bytes` bytes: 4 is f32, every
    /// other size takes f64's.
    pub const fn auto_plane_wave_min_p(value_bytes: usize) -> usize {
        if value_bytes == 4 {
            Self::AUTO_PLANE_WAVE_MIN_P_F32
        } else {
            Self::AUTO_PLANE_WAVE_MIN_P_F64
        }
    }

    /// Returns the strategy used in the node context `node`: `self`, or for
    /// [`Auto`](Self::Auto) [`PlaneWave`](Self::PlaneWave) from degree
    /// [`auto_plane_wave_min_p`](Self::auto_plane_wave_min_p) (f64 p ≥ 7, f32 p ≥ 8) and
    /// [`Batched`](Self::Batched) below it. Never returns `Auto`.
    /// [`choose`](Self::choose) also gives the reason.
    ///
    /// *Source of the rule* (Phase 6 T11, decision 7 of docs/phase6/README.md, signed off
    /// on 2026-10-20; fmm-validate/results/phase6.md). Measured on one node of each
    /// machine (the M3 Max, locust and Kathleen with AVX-512), the cube and the Plummer
    /// sphere at N = 10⁶ and 10⁷, f64 p = 3, 6, 7 (Kathleen), 8, 12, 18 and f32 p = 3, 6,
    /// 7 (Kathleen), 8, at one rank per core, Phase 5N's split and one rank per node:
    /// Batched (fused from p = 6, `FmmBuilder::m2l_mul_add`) was the fastest strategy or
    /// within 1.05 of it at f64 p ≤ 6 and f32 p ≤ 7 on Kathleen (within 1.15 on the NEON
    /// machines but at f64 p = 6 on the M3 Max, below), PlaneWave the fastest at f64 p ≥ 7
    /// and f32 p = 8 (within 1.05) everywhere; Rotation, Dense and Classes were never the
    /// fastest. Both hold their tables once per rank, and at one rank per core Batched's
    /// dense tables (6.4 MB at f64 p = 6) cost less than Dense's per-target reads did, so
    /// neither the ranks per node nor a node budget enters. The exception left: at f64
    /// p = 6 PlaneWave was 4–28 % faster on the M3 Max (mixed on locust, slower on
    /// Kathleen), kept on Batched for one machine-independent threshold. The rule it
    /// replaced (Phase 5N T4: Dense or Rotation by the dense tables' bytes per rank and per
    /// node) is recorded in docs/design/node-m2l.md §11.
    ///
    /// The batched GEMM path of the device has its own rule: with a device backend and
    /// M2L on the device, `FmmBuilder::build` resolves `Auto` by the device path's
    /// choice (Phase 4 T12, the `tune` module, feature `gpu`), not by this.
    pub fn resolve_on_node(self, node: &NodeContext) -> Self {
        self.choose(node).strategy()
    }

    /// Returns the strategy used in the node context `node` with its reason
    /// ([`StrategyChoice`]): [`StrategyChoice::Named`] for a strategy other than
    /// [`Auto`](Self::Auto), and for `Auto` [`StrategyChoice::Node`] with the strategy
    /// of [`resolve_on_node`](Self::resolve_on_node).
    pub fn choose(self, node: &NodeContext) -> StrategyChoice {
        if self != Self::Auto {
            return StrategyChoice::Named(self);
        }
        let strategy = if node.p >= Self::auto_plane_wave_min_p(node.value_bytes) {
            Self::PlaneWave
        } else {
            Self::Batched
        };
        StrategyChoice::Node {
            strategy,
            node: *node,
        }
    }
}

/// The inputs of [`M2lStrategy::Auto`]'s rule on the host ([`M2lStrategy::resolve_on_node`]):
/// the degree and the precision, the same on every rank of a run.
///
/// `FmmBuilder::build` takes both from its settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeContext {
    /// The degree p.
    pub p: usize,
    /// The bytes of one table value: `size_of::<T>()`, 8 for f64 and 4 for f32.
    pub value_bytes: usize,
}

impl NodeContext {
    /// The context of degree `p` and values of `value_bytes` bytes.
    pub fn new(p: usize, value_bytes: usize) -> Self {
        Self { p, value_bytes }
    }

    /// The context of degree `p` in precision `T`: the rule of [`Tables::build`] and
    /// [`Tables::load_or_build`].
    pub fn one_rank<T>(p: usize) -> Self {
        Self::new(p, std::mem::size_of::<T>())
    }

    /// D(p, T): the bytes of the dense tables of one rank, 332 (p + 1)⁴ values (316 M2L
    /// and 2 × 8 octant matrices of (p + 1)² × (p + 1)²); 17.4 MB in f64 at p = 8.
    /// Saturates instead of overflowing.
    pub fn dense_rank_bytes(&self) -> usize {
        let n = (self.p + 1).saturating_pow(4);
        332usize.saturating_mul(n).saturating_mul(self.value_bytes)
    }
}

/// The strategy of a build and why it was chosen: [`M2lStrategy::choose`], and
/// `FmmBuilder::build` for the device path; reported in `ThreadingReport::m2l`.
///
/// Its `Display` is the reason, for reports: for example `Auto → PlaneWave: f64 p = 8 ≥ 7`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StrategyChoice {
    /// A strategy the builder named, never overridden.
    Named(M2lStrategy),
    /// [`M2lStrategy::Auto`] resolved by the rule of [`M2lStrategy::resolve_on_node`] in
    /// the context `node`.
    Node {
        /// The resolved strategy: [`M2lStrategy::Batched`] or [`M2lStrategy::PlaneWave`].
        strategy: M2lStrategy,
        /// The inputs of the rule.
        node: NodeContext,
    },
    /// [`M2lStrategy::Auto`] resolved by the device path, with a device backend and M2L
    /// on the device (Phase 4 T12, the `tune` module, feature `gpu`).
    Device(M2lStrategy),
}

impl StrategyChoice {
    /// The resolved strategy, never [`M2lStrategy::Auto`] (unless a strategy so named).
    pub fn strategy(&self) -> M2lStrategy {
        match *self {
            Self::Named(strategy) | Self::Device(strategy) => strategy,
            Self::Node { strategy, .. } => strategy,
        }
    }
}

impl fmt::Display for StrategyChoice {
    /// `Rotation (named)`; `Auto → Batched: f64 p = 6 < 7` or `Auto → PlaneWave: f32 p = 8
    /// ≥ 8`; `Auto → Dense (the device path)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Named(strategy) => write!(f, "{strategy:?} (named)"),
            Self::Device(strategy) => write!(f, "Auto → {strategy:?} (the device path)"),
            Self::Node { strategy, node } => {
                let precision = if node.value_bytes == 4 { "f32" } else { "f64" };
                let threshold = M2lStrategy::auto_plane_wave_min_p(node.value_bytes);
                let relation = if node.p >= threshold { "≥" } else { "<" };
                write!(
                    f,
                    "Auto → {strategy:?}: {precision} p = {} {relation} {threshold}",
                    node.p
                )
            }
        }
    }
}

/// The table families of one resolved strategy.
#[derive(Clone, Debug, PartialEq)]
enum Families<T: RealScalar> {
    /// `Dense`, or with `batched` `Batched`: the same tables.
    Dense {
        m2m: M2mTables<T>,
        l2l: L2lTables<T>,
        m2l: M2lTables<T>,
        batched: bool,
    },
    Classes {
        m2m: M2mTables<T>,
        l2l: L2lTables<T>,
        m2l: M2lClasses<T>,
    },
    /// Boxed: the struct of three shift-table families is much larger than the others.
    Rotation(Box<RotationTables<T>>),
    /// `PlaneWave`: Rotation's tables for M2M and L2L, the plane waves for M2L.
    PlaneWave {
        rotation: Box<RotationTables<T>>,
        waves: Box<PlaneWaveTables<T>>,
    },
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
/// [`M2lStrategy::Rotation`], each 2 (p + 1)² reals, and for [`M2lStrategy::PlaneWave`]
/// a [`RotationScratch`] and a [`PlaneWaveScratch`] (the waves of one pair). Made by
/// [`Tables::scratch`].
#[derive(Clone, Debug)]
pub struct TableScratch<T: RealScalar> {
    kind: ScratchKind<T>,
}

#[derive(Clone, Debug)]
enum ScratchKind<T: RealScalar> {
    Dense,
    Classes(M2lScratch<T>),
    Rotation(RotationScratch<T>),
    PlaneWave(RotationScratch<T>, Box<PlaneWaveScratch<T>>),
}

impl<T: RealScalar> Tables<T> {
    /// Builds the tables that `strategy` needs, and no others: a named strategy's, or for
    /// [`M2lStrategy::Auto`] those of the host rule in precision `T`
    /// ([`M2lStrategy::resolve_on_node`] with [`NodeContext::one_rank`]), as `FmmBuilder`
    /// resolves it on the host. Serial and deterministic; allocates the
    /// tables.
    ///
    /// Building costs O(p⁶) for the dense families (8 (p + 1)² `direct` calls per
    /// octant family, 316 (p + 1)² for M2L, 16 (p + 1)² for the classes) and O(p⁴) for
    /// the rotation tables; at p = 16 the dense M2L tables take seconds, the others
    /// well under one.
    ///
    /// # Panics
    ///
    /// Under [`M2lStrategy::PlaneWave`] if `p` exceeds the plane-wave quadratures'
    /// [`nd_fmm_tables::plane_wave::MAX_DEGREE`] (20); `FmmBuilder` refuses such a p
    /// first, for every strategy (`SettingsError::DegreeTooLarge`).
    pub fn build(p: usize, strategy: M2lStrategy) -> Self {
        let families = match strategy.resolve_on_node(&NodeContext::one_rank::<T>(p)) {
            strategy @ (M2lStrategy::Dense | M2lStrategy::Batched) => Families::Dense {
                m2m: M2mTables::build(p),
                l2l: L2lTables::build(p),
                m2l: M2lTables::build(p),
                batched: strategy == M2lStrategy::Batched,
            },
            M2lStrategy::Classes => Families::Classes {
                m2m: M2mTables::build(p),
                l2l: L2lTables::build(p),
                m2l: M2lClasses::build(p),
            },
            M2lStrategy::Rotation => Families::Rotation(Box::new(RotationTables::build(p))),
            M2lStrategy::PlaneWave => Families::PlaneWave {
                rotation: Box::new(RotationTables::build(p)),
                waves: Box::new(PlaneWaveTables::build(p)),
            },
            M2lStrategy::Auto => unreachable!("resolve never returns Auto"),
        };
        Self { p, families }
    }

    /// Returns the degree p of every table.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Returns the resolved strategy: [`M2lStrategy::Dense`], [`M2lStrategy::Classes`],
    /// [`M2lStrategy::Rotation`], [`M2lStrategy::Batched`] or [`M2lStrategy::PlaneWave`],
    /// never `Auto`.
    pub fn strategy(&self) -> M2lStrategy {
        match self.families {
            Families::Dense { batched: false, .. } => M2lStrategy::Dense,
            Families::Dense { batched: true, .. } => M2lStrategy::Batched,
            Families::Classes { .. } => M2lStrategy::Classes,
            Families::Rotation(_) => M2lStrategy::Rotation,
            Families::PlaneWave { .. } => M2lStrategy::PlaneWave,
        }
    }

    /// Returns the cached table families held, in the order
    /// [`load_or_build`](Self::load_or_build) reports them (the plane-wave tables of
    /// `PlaneWave` are not cached: they build in milliseconds).
    pub fn kinds(&self) -> &'static [TableKind] {
        match self.families {
            Families::Dense { .. } => &[TableKind::M2m, TableKind::L2l, TableKind::M2l],
            Families::Classes { .. } => &[TableKind::M2m, TableKind::L2l, TableKind::M2lClasses],
            Families::Rotation(_) | Families::PlaneWave { .. } => &[TableKind::Rotation],
        }
    }

    /// Allocates the scratch that [`m2m`](Self::m2m), [`l2l`](Self::l2l) and
    /// [`m2l`](Self::m2l) need with these tables.
    pub fn scratch(&self) -> TableScratch<T> {
        let kind = match self.families {
            Families::Dense { .. } => ScratchKind::Dense,
            Families::Classes { .. } => ScratchKind::Classes(M2lScratch::new(self.p)),
            Families::Rotation(_) => ScratchKind::Rotation(RotationScratch::new(self.p)),
            Families::PlaneWave { ref waves, .. } => {
                ScratchKind::PlaneWave(RotationScratch::new(self.p), Box::new(waves.scratch()))
            }
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
            Families::Rotation(rotation) | Families::PlaneWave { rotation, .. } => {
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
            Families::Rotation(rotation) | Families::PlaneWave { rotation, .. } => {
                rotation.l2l(o, input, output, scratch.rotation());
            }
        }
    }

    /// Adds the M2L of the source multipole `multipole` to the target local `local`, a
    /// V-list pair whose offset has the position `offset_index` in `V_LIST_DIRECTIONS`
    /// (CONVENTIONS §3.11, §3.12): the dense matrix ((p + 1)⁴ multiply–adds; under
    /// `Batched` too, one pair alone), the class matrix between two coefficient transforms
    /// ((p + 1)⁴ + O(p³)), rotation (≈ (10/3) p³), or under `PlaneWave` the pair's plane
    /// waves along the first axis of z, y, x on which the offset is 2 or 3 boxes
    /// (`PlaneWaveTables::m2l_of_offset`; [`m2l_at`](Self::m2l_at) takes the direction of
    /// merge and shift instead).
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
            Families::PlaneWave { waves, .. } => {
                waves.m2l_of_offset(offset_index, multipole, local, scratch.plane_wave());
            }
        }
    }

    /// [`m2l`](Self::m2l) of a pair whose target has the child index `child` (CONVENTIONS
    /// §3.12): the same for every strategy but `PlaneWave`, which translates the pair
    /// along the direction merge and shift gives it (`PlaneWaveTables::pair_direction`),
    /// so that the pair by pair path equals the level call to rounding.
    ///
    /// # Panics
    ///
    /// As [`m2l`](Self::m2l), and if `child >= 8`.
    #[inline]
    pub fn m2l_at(
        &self,
        offset_index: usize,
        child: usize,
        multipole: &[T],
        local: &mut [T],
        scratch: &mut TableScratch<T>,
    ) {
        match &self.families {
            Families::PlaneWave { waves, .. } => {
                let direction = PlaneWaveTables::<T>::pair_direction(offset_index, child);
                waves.m2l(
                    offset_index,
                    direction,
                    multipole,
                    local,
                    scratch.plane_wave(),
                );
            }
            _ => self.m2l(offset_index, multipole, local, scratch),
        }
    }
}

impl<T: RealScalar> Tables<T> {
    /// The 316 dense M2L matrices under `Batched`, for the per-level products of the
    /// operator (Phase 6 T2); `None` under every other strategy.
    pub(crate) fn batched_m2l(&self) -> Option<&MatrixSet<T>> {
        match &self.families {
            Families::Dense {
                m2l, batched: true, ..
            } => Some(m2l.matrices()),
            _ => None,
        }
    }

    /// The plane-wave tables under `PlaneWave`, for the level calls of the operator
    /// (Phase 6 T13); `None` under every other strategy.
    pub fn plane_waves(&self) -> Option<&PlaneWaveTables<T>> {
        match &self.families {
            Families::PlaneWave { waves, .. } => Some(waves),
            _ => None,
        }
    }
}

/// The families the device path uploads (device-path.md §3.3, §6.8): crate-private, so
/// the public surface of [`Tables`] does not change.
#[cfg(feature = "gpu")]
impl<T: RealScalar> Tables<T> {
    /// The dense octant tables of M2M and L2L, under `Dense` and `Classes`; `None` under
    /// `Rotation` and `PlaneWave`, whose device path builds or loads them itself.
    pub(crate) fn octant_families(&self) -> Option<(&M2mTables<T>, &L2lTables<T>)> {
        match &self.families {
            Families::Dense { m2m, l2l, .. } | Families::Classes { m2m, l2l, .. } => {
                Some((m2m, l2l))
            }
            Families::Rotation(_) | Families::PlaneWave { .. } => None,
        }
    }

    /// The dense M2L tables, under `Dense` and `Batched`.
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
    /// Loads the tables that `strategy` needs (resolved as in [`build`](Self::build))
    /// from `cache`, or builds
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
        let (families, outcomes) = match strategy.resolve_on_node(&NodeContext::one_rank::<T>(p)) {
            strategy @ (M2lStrategy::Dense | M2lStrategy::Batched) => {
                let (m2m, m2m_outcome) = cache.load_or_build(p);
                let (l2l, l2l_outcome) = cache.load_or_build(p);
                let (m2l, m2l_outcome) = cache.load_or_build(p);
                (
                    Families::Dense {
                        m2m,
                        l2l,
                        m2l,
                        batched: strategy == M2lStrategy::Batched,
                    },
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
            M2lStrategy::PlaneWave => {
                let (rotation, outcome) = cache.load_or_build(p);
                (
                    Families::PlaneWave {
                        rotation: Box::new(rotation),
                        waves: Box::new(PlaneWaveTables::build(p)),
                    },
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
            ScratchKind::Rotation(scratch) | ScratchKind::PlaneWave(scratch, _) => scratch,
            _ => panic!("the table scratch does not belong to Rotation tables"),
        }
    }

    /// The scratch of [`PlaneWaveTables::m2l`].
    fn plane_wave(&mut self) -> &mut PlaneWaveScratch<T> {
        match &mut self.kind {
            ScratchKind::PlaneWave(_, scratch) => scratch,
            _ => panic!("the table scratch does not belong to PlaneWave tables"),
        }
    }
}
