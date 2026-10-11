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
//! // `Auto` on one rank: Dense, since the dense tables at p = 2 are small (0.2 MB).
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

use std::fmt;

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
    /// [`Dense`](Self::Dense) while the dense tables of a rank, and of all the ranks of a
    /// node together, are small enough, [`Rotation`](Self::Rotation) otherwise; see
    /// [`resolve_on_node`](Self::resolve_on_node).
    #[default]
    Auto,
}

impl M2lStrategy {
    /// The largest degree at which [`Auto`](Self::Auto) resolved to
    /// [`Dense`](Self::Dense) before Phase 5N.
    #[deprecated(
        note = "Auto no longer resolves by p alone (Phase 5N T4, docs/design/node-m2l.md \
                §4): see M2lStrategy::resolve_on_node, AUTO_DENSE_RANK_BYTES and \
                AUTO_DENSE_NODE_BYTES"
    )]
    pub const AUTO_DENSE_MAX_P: usize = 8;

    /// The dense tables of one rank, D(p, T) bytes ([`NodeContext::dense_rank_bytes`]),
    /// above which [`Auto`](Self::Auto) resolves to [`Rotation`](Self::Rotation) whatever
    /// the node: f64 p ≤ 6 (6.4 MB) and f32 p ≤ 8 (8.7 MB) lie below it, f64 p = 7
    /// (10.9 MB) and f32 p = 9 (13.3 MB) above. Dense wins on one rank at the first and
    /// loses at f64 p = 8 on every machine measured (docs/design/node-m2l.md §2.1, §3).
    /// Between them the measurements of Phase 5N T4 split (one rank, N = 10⁶): at f64
    /// p = 7 Dense is 6–10 % faster on the M3 Max and locust and 10–15 % slower on
    /// Kathleen, at f32 p = 9 (beyond f32's useful degree) 14–31 % faster on all three;
    /// the cap stays, as signed off (docs/design/node-m2l.md, "Outcome").
    pub const AUTO_DENSE_RANK_BYTES: usize = 9_000_000;

    /// The default node budget of [`Auto`](Self::Auto): the private dense tables of all
    /// the ranks of a node, ranks per node × D(p, T), above which it resolves to
    /// [`Rotation`](Self::Rotation). 64 MB is the signed-off machine-independent default
    /// (docs/design/node-m2l.md §4): no single budget suits every machine and precision,
    /// so a machine may set its own (`FmmBuilder::dense_node_budget`). The harness uses
    /// 96 MB on locust and Kathleen, whose larger shared caches keep Dense faster at f32
    /// p = 8 to 8–10 ranks per node, and 56 MB on the M3 Max, where at f64 p = 6 Dense
    /// was 1.11× (8 ranks per node) and 1.83× (10) Rotation's time (Phase 5N T4).
    pub const AUTO_DENSE_NODE_BYTES: usize = 64_000_000;

    /// Returns the strategy used at degree `p` in f64 on one rank per node with the
    /// default budget: [`resolve_on_node`](Self::resolve_on_node) with
    /// [`NodeContext::one_rank::<f64>(p)`](NodeContext::one_rank).
    #[deprecated(
        note = "the rule depends on the precision and the ranks per node (Phase 5N T4): \
                use M2lStrategy::resolve_on_node"
    )]
    pub fn resolve(self, p: usize) -> Self {
        self.resolve_on_node(&NodeContext::one_rank::<f64>(p))
    }

    /// Returns the strategy used in the node context `node`: `self`, or for
    /// [`Auto`](Self::Auto) [`Dense`](Self::Dense) iff the dense tables of one rank,
    /// D(p, T) = 332 (p + 1)⁴ · size_of(T) bytes, are at most
    /// [`AUTO_DENSE_RANK_BYTES`](Self::AUTO_DENSE_RANK_BYTES) and those of the node,
    /// ranks per node × D(p, T), at most the node budget; [`Rotation`](Self::Rotation)
    /// otherwise. Never returns `Auto`. [`choose`](Self::choose) also gives the reason.
    ///
    /// *Source of the rule* (Phase 5N T4, signed off as rule B of docs/design/
    /// node-m2l.md §4 on 2026-10-11, decision 5 of docs/phase5n/README.md). Every rank
    /// holds its own dense tables (316 M2L and 16 octant matrices), and a dense M2L reads
    /// a different matrix for every V pair, so the tables are streamed from the nearest
    /// level that holds the copies of every rank of a node. While those fit the node's
    /// shared cache Dense keeps its one-rank cost; beyond, every pair streams from memory
    /// at the share of the socket's bandwidth that is left, and the cost grows with the
    /// ranks (at f64 p = 8 from 1.15 to 4.93 µs per pair from 1 to 12 ranks on the M3
    /// Max, 1.18 to 9.63 from 1 to 72 on locust, 2.92 to 8.83 from 1 to 40 on Kathleen),
    /// while Rotation's tables (0.83 MB at p = 8) stay in each core's L2. Measured over
    /// 194 cells (N = 10⁶, the cube and the Plummer sphere, f64 p = 3, 6, 8, 10, 12 and
    /// f32 p = 8, one rank to 12, 72 and 40 per node and the splits with threads), the
    /// rule was scored within 1.10 of the best strategy except the f32 p = 8 band of 8–10
    /// ranks per node on locust and Kathleen. Measured with it in Phase 5N T4 (the same
    /// grid, the budgets of the harness): at p = 8 within 1.03 of the best at every count
    /// of that grid on the three machines (f64: Rotation, the best in every cell), and in
    /// the band where the budget decides (f32 p = 8 at 12–16 and f64 p = 6 at 8–12 ranks
    /// per node) within 1.16. Threads share their rank's copy and so
    /// do not enter; no machine property enters, so a job resolves alike on every
    /// machine. The rule of Phases 2–5 (Dense for p ≤ 8, from Phase 2 T8's one-thread,
    /// one-rank, one-pair timings, docs/design/laplace-fmm-plan.md §7) put f64 p = 8 on
    /// Dense, 1.07–1.27× Rotation's time on one rank and up to 6.3× at 72 ranks per node.
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
        let rank = node.dense_rank_bytes();
        let strategy =
            if rank <= Self::AUTO_DENSE_RANK_BYTES && node.dense_node_bytes() <= node.node_budget {
                Self::Dense
            } else {
                Self::Rotation
            };
        StrategyChoice::Node {
            strategy,
            node: *node,
        }
    }
}

/// The inputs of [`M2lStrategy::Auto`]'s rule on the host ([`M2lStrategy::resolve_on_node`];
/// Phase 5N T4, docs/design/node-m2l.md §4): every one the same on every rank of a run.
///
/// `FmmBuilder::build` takes p and the precision from its settings, the budget from
/// `FmmBuilder::dense_node_budget`, and the ranks per node from one `split_shared` of
/// the communicator, the largest over the nodes, agreed in the all-reduce of its step 7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeContext {
    /// The degree p.
    pub p: usize,
    /// The bytes of one table value: `size_of::<T>()`, 8 for f64 and 4 for f32.
    pub value_bytes: usize,
    /// The ranks of a node, each with its own copy of the tables: the largest number over
    /// the nodes of the run. Threads do not count: they share their rank's copy.
    pub ranks_per_node: usize,
    /// The bytes the private dense tables of a node may take
    /// ([`M2lStrategy::AUTO_DENSE_NODE_BYTES`] by default).
    pub node_budget: usize,
}

impl NodeContext {
    /// The context of degree `p`, values of `value_bytes` bytes, `ranks_per_node` ranks
    /// per node and a node budget of `node_budget` bytes.
    pub fn new(p: usize, value_bytes: usize, ranks_per_node: usize, node_budget: usize) -> Self {
        Self {
            p,
            value_bytes,
            ranks_per_node,
            node_budget,
        }
    }

    /// The context of degree `p` in precision `T` on one rank per node with the default
    /// budget: the rule of [`Tables::build`] and [`Tables::load_or_build`].
    pub fn one_rank<T>(p: usize) -> Self {
        Self::new(
            p,
            std::mem::size_of::<T>(),
            1,
            M2lStrategy::AUTO_DENSE_NODE_BYTES,
        )
    }

    /// D(p, T): the bytes of the dense tables of one rank, 332 (p + 1)⁴ values (316 M2L
    /// and 2 × 8 octant matrices of (p + 1)² × (p + 1)²); 17.4 MB in f64 at p = 8.
    /// Saturates instead of overflowing.
    pub fn dense_rank_bytes(&self) -> usize {
        let n = (self.p + 1).saturating_pow(4);
        332usize.saturating_mul(n).saturating_mul(self.value_bytes)
    }

    /// The private dense tables of a node: ranks per node × D(p, T). Saturates.
    pub fn dense_node_bytes(&self) -> usize {
        self.ranks_per_node.saturating_mul(self.dense_rank_bytes())
    }
}

/// The strategy of a build and why it was chosen: [`M2lStrategy::choose`], and
/// `FmmBuilder::build` for the device path; reported in `ThreadingReport::m2l`.
///
/// Its `Display` is the reason, for reports: for example `Auto → Rotation: 12 ranks per
/// node × 8.7 MB > 64 MB`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StrategyChoice {
    /// A strategy the builder named, never overridden.
    Named(M2lStrategy),
    /// [`M2lStrategy::Auto`] resolved by the rule of [`M2lStrategy::resolve_on_node`] in
    /// the context `node`.
    Node {
        /// The resolved strategy: [`M2lStrategy::Dense`] or [`M2lStrategy::Rotation`].
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

/// `bytes` in MB (10⁶ bytes), one decimal, without a trailing `.0`.
fn megabytes(bytes: usize) -> String {
    let text = format!("{:.1}", bytes as f64 / 1e6);
    match text.strip_suffix(".0") {
        Some(whole) => whole.to_owned(),
        None => text,
    }
}

impl fmt::Display for StrategyChoice {
    /// `Rotation (named)`; `Auto → Dense: 4 ranks per node × 6.4 MB ≤ 64 MB`, `Auto →
    /// Rotation: 17.4 MB per rank > 9 MB` or `Auto → Rotation: 12 ranks per node × 8.7 MB
    /// > 64 MB`; `Auto → Dense (the device path)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Named(strategy) => write!(f, "{strategy:?} (named)"),
            Self::Device(strategy) => write!(f, "Auto → {strategy:?} (the device path)"),
            Self::Node { strategy, node } => {
                let rank = node.dense_rank_bytes();
                write!(f, "Auto → {strategy:?}: ")?;
                if rank > M2lStrategy::AUTO_DENSE_RANK_BYTES {
                    return write!(
                        f,
                        "{} MB per rank > {} MB",
                        megabytes(rank),
                        megabytes(M2lStrategy::AUTO_DENSE_RANK_BYTES)
                    );
                }
                let ranks = node.ranks_per_node;
                write!(
                    f,
                    "{ranks} rank{} per node × {} MB {} {} MB",
                    if ranks == 1 { "" } else { "s" },
                    megabytes(rank),
                    if strategy == M2lStrategy::Dense {
                        "≤"
                    } else {
                        ">"
                    },
                    megabytes(node.node_budget)
                )
            }
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
    /// Builds the tables that `strategy` needs, and no others: a named strategy's, or for
    /// [`M2lStrategy::Auto`] those of the rule on one rank per node in precision `T`
    /// ([`M2lStrategy::resolve_on_node`] with [`NodeContext::one_rank`]). `FmmBuilder`
    /// resolves `Auto` with the node first. Serial and deterministic; allocates the
    /// tables.
    ///
    /// Building costs O(p⁶) for the dense families (8 (p + 1)² `direct` calls per
    /// octant family, 316 (p + 1)² for M2L, 16 (p + 1)² for the classes) and O(p⁴) for
    /// the rotation tables; at p = 16 the dense M2L tables take seconds, the others
    /// well under one.
    pub fn build(p: usize, strategy: M2lStrategy) -> Self {
        let families = match strategy.resolve_on_node(&NodeContext::one_rank::<T>(p)) {
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
