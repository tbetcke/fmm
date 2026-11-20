//! The Helmholtz family's operator of the plan (Helmholtz CH3.1a;
//! docs/design/helmholtz-fmm-proposal.md §5.1–§5.3, §5.7, §6.2, §6.3; CONVENTIONS §3.15):
//! [`HelmholtzOperator`], for real and complex k with real and complex charges, RCR on
//! every level.
//!
//! # Levels and units
//!
//! Each level l works in the units of its boxes (half-width r_l = 1), with the wavenumber
//! k̂_l = k r_l formed in f64 and rounded to T once, as the leaf operators of
//! [`Densities`] form it; the expansions of level l carry the stored form of
//! `nd_fmm_ref::helmholtz::Level::new(k̂_l, 1)` and have degree p_l (the
//! [`degree`](mod@super::degree) module). The chunks and box blocks are those of [`Densities`]
//! at p_l: m expansions per box, one after the other, each of
//! [`Densities::multipole_planes`] or [`Densities::local_planes`] planes.
//!
//! # Operators
//!
//! | Kind | Body |
//! | --- | --- |
//! | P2M, L2P | [`Densities::p2m`], [`Densities::l2p`] in the leaf's own frame |
//! | M2M, L2L, M2L | `nd_fmm_tables::helmholtz::HelmholtzTables` (RCR with the level's tables; real shifts on each plane for real k) |
//! | M2L under [`HelmholtzM2l::Dense`] | `nd_fmm_tables::helmholtz::dense::DenseM2l` on the levels whose tables fit the budget, RCR on the others |
//! | M2L under [`HelmholtzM2l::Classes`] | `nd_fmm_tables::helmholtz::classes::ClassM2l` on the levels whose tables fit the budget, RCR on the others |
//! | M2L under [`HelmholtzM2l::Batched`] | `nd_fmm_tables::helmholtz::batched::BatchedM2l` by `nd_fmm_simd::MatrixKernel`, block by block, on the levels whose tables fit the budget, RCR on the others |
//! | M2L under [`HelmholtzM2l::PlaneWave`] | `nd_fmm_tables::helmholtz::plane_wave::merge::PlaneWaveM2l`, merge and shift on the levels with a quadrature, RCR on the others |
//! | P2L | `nd_fmm_ref::helmholtz::leaf::p2l` in the source leaf's units, at (ĉ(t\|s), r̂(t\|s)) |
//! | M2P | `nd_fmm_ref::helmholtz::leaf::m2p` in the target leaf's units, at (ĉ(s\|t), r̂(s\|t)) |
//! | P2P | `nd_fmm_simd::P2pKernel::evaluate_helmholtz` on the ISA of the [`P2pChoice`], or `nd_fmm_ref::helmholtz::p2p::p2p` for [`P2pChoice::Reference`], in the target leaf's units, the sources mapped as ĉ(s\|t) + r̂(s\|t) u |
//!
//! A box of level l holds r_l φ̃ in its own units, as a leaf's chunk does (CONVENTIONS
//! §3.13 and §3.15, "Level units"; `nd_fmm_tables::helmholtz`, "Levels and units"). P2L
//! forms the local in the source leaf's units, so it scales the charges by r̂(t|s) =
//! r_t / r_s (exact, a power of two); M2P needs nothing, as a multipole's factor 4π ik̂ carries the change of units.
//! P2L and M2P pass the reference a `Level` in the points' units: the wavenumber of the
//! leaf whose points they read, and the frame's radius as the other box's half-width, so
//! its power scale and damping are the other box's level's (the half-widths differ by
//! powers of two, which scale exactly). A W box lies on a finer level than the leaf, an X
//! leaf on a coarser one than the box; the operator finds their levels from the frame's
//! radius. M2P widens a real multipole (real k, real charges) to two planes in scratch.
//!
//! P2P, M2P and L2P accumulate into the target output chunk: M2P, and P2P with the
//! reference, load each target's two output columns of a vector as complex values, let the
//! reference operator add its terms to them, and store them back; L2P is
//! [`Densities::l2p`]'s. Every operator gives each charge vector exactly the one-vector
//! operator, and adds (+=); none applies 1/(4π) or the leaf radius (the output pass does,
//! CONVENTIONS §3.13).
//!
//! # P2P kernel
//!
//! P2P runs the kernel of a [`P2pChoice`] (Helmholtz CH3.4c), set by
//! [`with_p2p`](HelmholtzOperator::with_p2p) and by default [`P2pChoice::Auto`], as
//! Laplace's operator does: `nd_fmm_simd::P2pKernel::evaluate_helmholtz` (Helmholtz CH3.4b)
//! on the default ISA of the machine (`Isa::detect`) or on a named one, and
//! [`P2pChoice::Reference`] keeps `nd_fmm_ref::helmholtz::p2p::p2p`, the trusted slower
//! path. The kernel takes one vector of all the leaf's targets per call: the charges of
//! vector v, Re q and Im q of a source together (the chunk's own strengths for one vector,
//! gathered into scratch for several), and the output chunk's planes as they lie, since the
//! potentials of output columns 2v and 2v + 1 are 2 n_t adjacent values and their gradients
//! 2 n_t adjacent triples, the kernel's layout. Each ISA is within the kernel's accuracy
//! contract of the reference (docs/design/simd-p2p.md §4.12), and `Isa::Scalar` is the
//! reference bit for bit. For one machine, ISA and build the output is deterministic, the
//! same at every thread count and on P ranks (constraint 2);
//! [`p2p_kernel`](HelmholtzOperator::p2p_kernel) reports the kernel that runs.
//!
//! # M2L
//!
//! M2L runs by the level's RCR tables ([`HelmholtzM2l::Rotation`], the operator's default
//! until [`with_m2l`](HelmholtzOperator::with_m2l); `HelmholtzFmmBuilder`'s is
//! [`HelmholtzM2l::Auto`]), or under
//! [`HelmholtzM2l::Dense`] (Helmholtz CH3.2a; proposal §5.3) by one complex matrix per
//! offset on each level whose tables fit a byte budget, chosen from the finest level up
//! (`DenseM2l::select`, a function of the degrees and the budget only, so every rank
//! chooses alike), and by RCR on the others: [`with_m2l`](HelmholtzOperator::with_m2l).
//! The dense tables are Rotation's coefficients applied to unit multipoles in f64 and
//! rounded to T, so Dense agrees with Rotation to rounding, not bit for bit. Dense combines
//! the four real products of a complex M2L pair by pair, in the target's row order as every
//! body, so its output is the same at every thread count and on P ranks the one-rank
//! output; it adds no per-target epilogue and no driver change.
//!
//! Under [`HelmholtzM2l::Classes`] (Helmholtz CH3.2b; proposal §5.3) a level whose tables
//! fit the same budget (`ClassM2l::select`, from the finest level up) holds the 16 class
//! matrices of CONVENTIONS §3.12 in Dense's layout, bit for bit Dense's matrices of the
//! representatives, and the cube-group transforms of the orthonormal coefficients: each pair
//! takes its multipole through O(Pᵀ), the class matrix by Dense's chain and O(P) into the
//! local, signed permutations for the elements that fix the z-axis and dense per-degree
//! blocks for the others. So Classes agrees with Dense, and with Rotation, to rounding, and
//! is deterministic and rank-invariant as Dense is.
//!
//! Under [`HelmholtzM2l::Batched`] (Helmholtz CH3.2c; proposal §5.3) a level whose tables
//! fit the same budget (Dense's `select`) holds Dense's matrices in split planes, A_r and
//! A_i each n × n (`BatchedM2l`), and its M2L level call runs as Laplace's batched M2L
//! does, with its own block (not `driver::m2l_block`, whose layout is Laplace's; the user,
//! 2026-11-02): blocks of `M2L_BLOCK_TARGETS` consecutive targets, and in each the offset
//! indices d in order, the pairs of the grouped view's `batch(d)` whose target lies in the
//! block, m columns a pair, through `nd_fmm_simd::MatrixKernel::accumulate_columns` (on the
//! ISA of the P2P kernel, the scalar path for [`P2pChoice::Reference`]), each table read
//! once per block. Per offset, on each column, in the default [`MulAdd::Separate`] mode
//! (`MatrixSet::apply`'s steps, product and sum rounded apart):
//!
//! - real multipoles (real k, real charges): L_r += A_r M, then L_i += A_i M, Dense's chain
//!   bit for bit;
//! - complex multipoles: L_r += A_r M_r; L_r −= A_i M_i (L_r negated, A_i M_i added and L_r
//!   negated back, exact); L_i += A_i M_r; L_i += A_r M_i. Each column sums the real plane's
//!   products before the imaginary plane's where Dense alternates them per k, so Batched is
//!   Dense to rounding here (CH3.2a's sign-off 2), with no per-target epilogue.
//!
//! In the opt-in [`MulAdd::Fused`] mode ([`with_m2l_mul_add`](HelmholtzOperator::with_m2l_mul_add))
//! each step is one fma: Dense to rounding also with real multipoles (per element within
//! about n u of the separate form, relative to the sum of its terms' magnitudes), and about
//! 1.8 times the separate form's rate on the M6, where the kernel is bound by its arithmetic
//! (fmm-validate/results/helmholtz-ch3-2c.md).
//!
//! An element depends on its own column only, and a target meets the offsets in the order
//! of its row, so the level call is the pair body (the same calls with one column per
//! vector) bit for bit: the same at every thread count, block split and ISA, and on P ranks
//! the one-rank output. Levels without tables run target by target by RCR.
//!
//! [`HelmholtzM2l::Auto`] (Helmholtz CH3.2d) is the rule per level: Batched in the fused
//! mode on every vector ISA ([`auto_m2l_mul_add`]; unless
//! [`with_m2l_mul_add`](HelmholtzOperator::with_m2l_mul_add) names a mode) on the levels of degree p_l at most a crossover that fit the budget, from
//! the finest level up (`BatchedM2l::select_within`), RCR on the others. The crossover is
//! [`auto_batched_max_degree`] of the column kernel's ISA and the multipole planes, or the
//! degree named by [`with_m2l_max_degree`](HelmholtzOperator::with_m2l_max_degree); so the
//! choice is a function of the degrees, the budget and the machine class only (proposal
//! §2, constraint 9), and each level runs Batched's or Rotation's level call as those
//! settings do, deterministic and rank-invariant alike. `HelmholtzFmmBuilder` agrees the
//! crossover over the ranks.
//!
//! Since Helmholtz CH4.0 Auto also reads each level's V list over the whole tree, a
//! [`LevelShape`] of its pairs and boxes ([`with_level_shapes`](HelmholtzOperator::with_level_shapes),
//! which `HelmholtzFmmBuilder` counts and sums over the ranks): Batched reads each offset's
//! table once per block and offset met, so on a level whose blocks meet few pairs per offset
//! (a sparse level: a surface's boxes have short V rows; or a coarse one: level 2 is a single
//! block) it reads n² values for few columns and loses to Rotation. A level then runs
//! Batched only where the estimate of its columns per table read,
//! [`LevelShape::batched_columns`], is at least [`auto_batched_min_columns`] of the ISA, the
//! planes and its degree (and p_l at most the crossover, as before). The shapes are integers
//! summed over the ranks, so this term is a function of the global tree (constraint 9) and
//! every rank chooses alike. Batched's level call also narrows its blocks on a level with
//! fewer blocks than threads ([`batched_block_targets`]; or a width named by
//! [`with_m2l_block_targets`](HelmholtzOperator::with_m2l_block_targets)), so that every thread
//! has a block: a split of the execution only, since the call is the pair chain bit for bit
//! at every width. The term cannot read the thread count (Batched's bits are not Rotation's).
//!
//! By default (Helmholtz CH6.2h, decision D8, signed off 2026-11-08;
//! [`with_auto_plane_wave`](HelmholtzOperator::with_auto_plane_wave) turns it off) Auto takes
//! the exponential M2L first: on each level of degree p_l at least
//! [`auto_plane_wave_min_degree`] of the ISA and the planes (or the degree named by
//! [`with_m2l_plane_wave_min_degree`](HelmholtzOperator::with_m2l_plane_wave_min_degree)) to
//! which the precision rule below gives a quadrature, at the level's own degree for
//! ε ≥ 10⁻⁶ or no tolerance and at the raised degree for 10⁻⁸ ≤ ε < 10⁻⁶ (Helmholtz
//! CH6.1g; from 10⁻⁷ since CH6.1h's sign-off, 2026-11-11, D8 amended), with Im κ_l below
//! [`AUTO_PLANE_WAVE_MAX_IM_KAPPA`], in f64 only; then Batched on the other levels as before
//! (the levels with waves take no batched tables and none of the budget), RCR on the rest.
//! A function of k, the levels' widths and degrees, the tolerance and the machine class
//! (constraint 9); each level runs PlaneWave's, Batched's or Rotation's level call
//! unchanged, so the output is deterministic and rank-invariant as theirs.
//! `HelmholtzFmmBuilder` agrees the smallest degree over the ranks.
//!
//! Under [`HelmholtzM2l::PlaneWave`] (Helmholtz CH6.1d; opt-in; CONVENTIONS §3.14, "Merge
//! and shift", with the conversions and shifts of §3.15, "Exponential M2L") a level runs the
//! exponential M2L where its [`PlaneWavePrecision`] gives it a quadrature: by the rule
//! (`nd_fmm_tables::helmholtz::plane_wave::merge::select`, measured in the FMM in Helmholtz
//! CH6.1e: iprec 2 at the FMM's tolerance ε ≥ 10⁻⁶;
//! [`with_plane_wave_tolerance`](HelmholtzOperator::with_plane_wave_tolerance), or without
//! one by the level's degree; a function of k, the level's width, its degree and ε) or
//! named per level, and only where κ_l = k w_l lies in the quadrature's tables; RCR on the
//! others. Below ε = 10⁻⁶ the rule raises the degree of each level with a quadrature
//! (Helmholtz CH6.1g, CH6.1f's resolving degree; `merge::plane_wave_degree`): iprec 3 at
//! p_w = max(p_l, 28) for 10⁻⁷ ≤ ε < 10⁻⁶ and at max(p_l, 32) for 10⁻⁸ ≤ ε < 10⁻⁷, none
//! below 10⁻⁸. The operator then builds its RCR tables at those degrees (again at the
//! degrees given when a later setting takes the waves away), so the level's stores, its M2L
//! and its leaves' operators run at p_w ([`degree`](HelmholtzOperator::degree);
//! [`base_degree`](HelmholtzOperator::base_degree) the degree given), and by
//! [`PlaneWaveRaise`] ([`with_plane_wave_raise`](HelmholtzOperator::with_plane_wave_raise);
//! Helmholtz CH6.1h) its translations too ([`PlaneWaveRaise::Level`]) or not:
//! under [`PlaneWaveRaise::Leaves`] and [`PlaneWaveRaise::M2l`] the M2M from the level to
//! its parent reads, and the L2L into it fills, the first p_l degrees of its expansions
//! ([`translation_degree`](HelmholtzOperator::translation_degree);
//! `HelmholtzTables::build_with_m2l_degrees`), while the M2M from its children fills, and the
//! L2L to them reads, all p_w; under [`PlaneWaveRaise::M2l`] its leaves' operators (P2M, L2P,
//! P2L into and M2P from its boxes) also work on the first p_l degrees, through a scratch
//! expansion at p_l. The default, [`PlaneWaveRaise::Rule`] (signed off 2026-11-11), runs `M2l`
//! at 10⁻⁷ ≤ ε < 10⁻⁶ and `Leaves` at 10⁻⁸ ≤ ε < 10⁻⁷. Named precisions and named degrees
//! raise nothing. Under the rule the FMM's errors stay within about 1.1×
//! Rotation's at the degrees given on CH3.1c's cases (`fmm-validate/results/helmholtz-ch6-1e.md`)
//! and, below 10⁻⁶, on CH6.1e's cases at 10⁻⁷ and 10⁻⁸
//! (`fmm-validate/results/helmholtz-ch6-1g.md`). The quadratures' α counts bound the
//! region's corner, lateral offsets to 4√2 box widths, unless
//! [`with_plane_wave_reach`](HelmholtzOperator::with_plane_wave_reach) names a smaller reach
//! (Helmholtz CH6.2e, for the measurements). Its level call restates Laplace's
//! (`crate::operator`, "The plane-wave M2L") without its vector kernels, by chunks of
//! parents (Helmholtz CH6.2a): the level is cut, in index order, into chunks of consecutive
//! whole parents whose computed children's distinct V sources fit the waves buffer (each
//! parent joins the chunk while they fit; one parent's sources, at most its 216 slots, always
//! do). For each chunk, each charge vector and each frame f = 0, 1, 2,
//!
//! 1. **outgoing**: the up and down waves of frame f of the chunk's sources, own and ghost
//!    boxes (the multipole rotated into the frame by Oⁿ(Q), its modes and the waves, 2W
//!    reals each, every value complex), the pool splitting the sources;
//! 2. **merge and shift**, parent by parent (consecutive boxes with one parent key), the
//!    pool splitting the chunk at parent boundaries: the parent's computed children (with
//!    a V row) and their sources by slot; for the up and then the down direction of f,
//!    every accumulator whose (geometric) mask holds a computed child sums its slots
//!    present in ascending order, each source's waves times its source shift, from zero;
//!    every computed child adds its accumulators in ascending mask order and multiplies the
//!    sum by its child shift; then each computed child with waves converts its up and then
//!    its down waves into a local in the frame and adds it, rotated back by Oⁿ(Q)ᵀ.
//!    Or by child ([`PlaneWaveSplit`]; Helmholtz CH6.2f), one task per box of the chunk:
//!    each computed child finds its parent's run and their sources by slot, sums the
//!    accumulators whose mask holds it, adds them and converts, the steps above for this
//!    child alone (its siblings' shared accumulators computed again by each). The shared
//!    drivers' split by parents never cuts a piece below `PARENT_GRAIN` = 64 boxes, so a
//!    level of 64 boxes ran on one thread; [`PlaneWaveSplit::Auto`] takes the split by child
//!    on a chunk whose split by parents makes at most half as many pieces holding a computed
//!    box as the pool has threads.
//!
//! Every local receives frame 0, then 1, then 2, each a fixed chain of its parent's
//! accumulators, which depend on the geometry of the slots alone, and a box's waves are a
//! function of its multipole alone, so the output is the same at every thread count and
//! every bound of the buffer and, as Laplace's argument goes, on P ranks the one-rank output
//! bit for bit, under either split (a child's chain is the same whichever task runs it). A
//! source read by two chunks has its waves computed in each. The pair body
//! ([`PairOperator`]) translates each pair by `PlaneWaveLevel::m2l` along the direction
//! merge and shift gives it (`PlaneWaveTables::pair_direction` of the offset and the
//! target's child index), which the level call equals to rounding (its two shift factors
//! are the pair's to rounding). It approximates M2L within its quadrature's error, so it is
//! not Rotation's output. The waves buffer holds one frame of a chunk's sources, 2W reals
//! per box, at most a byte bound
//! ([`with_plane_wave_buffer`](HelmholtzOperator::with_plane_wave_buffer); none: a whole
//! level; at least [`MIN_BUFFER_BOXES`] boxes of each level), with a position per box of the
//! largest level and the chunk's source list, sized from the plan by `HelmholtzFmmBuilder`
//! ([`with_plane_wave_boxes`](HelmholtzOperator::with_plane_wave_boxes)), which bounds it
//! by its node budget over the ranks per node (proposal §2, constraint 7); each scratch set
//! holds a level's buffers and the pair scratch of every level with waves;
//! [`plane_wave_bytes`](HelmholtzOperator::plane_wave_bytes) reports them with the tables,
//! and [`plane_wave_counts`](HelmholtzOperator::plane_wave_counts) the chunks and the waves
//! computed.
//! f64 only (§3.15, "Precision and range"): f32's range is untested (the factors reach
//! about 10³⁴ at iprec 3).
//!
//! The diagonal form (Helmholtz CH4.2b1; proposal §3.6, §5.4; opt-in until CH4.3,
//! [`with_diagonal`](HelmholtzOperator::with_diagonal), default [`HelmholtzDiagonal::Off`])
//! runs on the levels whose M2L is RCR's under the setting above (no waves, no Dense, Classes
//! or Batched tables), so it composes with every [`HelmholtzM2l`]: each offset class c of
//! such a level with E(l, c) ≤ θ ε ([`diagonal_rule_classes`]: the bound of
//! `nd_fmm_tables::helmholtz::diagonal::class_bounds`, θ = 1/2, S6's constants of f64 for the
//! transfer functions' build, or of f32 at ε ≥ 10⁻⁴ only, decision D3) takes `DiagonalM2l`'s
//! transfer function; the others stay on RCR. The transfer functions are interpolated in ŝ · t̂
//! from a double-double table ([`TransferBuild::Interpolated`], C_T = 0.23, the default
//! [`DEFAULT_DIAGONAL_BUILD`] since its sign-off) or built per sample in double-double
//! ([`TransferBuild::Direct`], C_T = 0.15, so a class may pass at a tighter tolerance; the
//! trusted path; Helmholtz CH4.2c2,
//! [`with_diagonal_build`](HelmholtzOperator::with_diagonal_build)).
//! [`HelmholtzDiagonal::Rule`] reads the FMM's tolerance
//! ([`with_plane_wave_tolerance`](HelmholtzOperator::with_plane_wave_tolerance)),
//! [`HelmholtzDiagonal::Tolerance`] a named one. A level takes it only where its scratch fits
//! the budget ([`with_diagonal_budget`](HelmholtzOperator::with_diagonal_budget)) at the largest
//! count of its V sources over the ranks
//! ([`with_diagonal_sources`](HelmholtzOperator::with_diagonal_sources)), else the whole level
//! runs RCR, the same on every rank. Its level call forms the signature of every distinct
//! source of a diagonal pair and vector once (S → F, blocks over the pool), then, in chunks of
//! consecutive targets over the pool, zeroes each target's accumulator per vector, adds its
//! diagonal pairs in row order, F → R into its local, then its RCR pairs in row order: each
//! local's chain is its own row's, so the output is the same at every thread count and chunk
//! width ([`with_diagonal_chunk`](HelmholtzOperator::with_diagonal_chunk)) and on P ranks the
//! one-rank output (proposal §5.4, step 3). [`m2l_on`](HelmholtzOperator::m2l_on) reports such
//! a level as [`HelmholtzM2l::Rotation`], [`diagonal_classes`](HelmholtzOperator::diagonal_classes)
//! its classes, [`diagonal_counts`](HelmholtzOperator::diagonal_counts) its pairs. The pair body
//! ([`PairOperator`]) takes a diagonal offset by S → F, the pair and F → R of the pair alone,
//! the level call to rounding. Its tables are not cached; their build time is
//! [`TableTimings::diagonal`].
//!
//! The time rule (Helmholtz CH4.3b1, [`HelmholtzDiagonal::Auto`], opt-in until CH4.3b2) takes
//! the bound's classes of a level only where a cost model of the level calls puts the diagonal
//! level below the M2L the level would run otherwise ([`DiagonalCosts`], per machine class,
//! [`diagonal_costs`]; named by [`with_diagonal_costs`](HelmholtzOperator::with_diagonal_costs)):
//! S → F and F → R per box with a V row, the diagonal pair per pair of a diagonal class, RCR's
//! pair per pair of the others, the exponential M2L per box with a V row, each a (p + 1)² +
//! b (p + 1)³ thread-ns. It reads the level's V lists ([`DiagonalShape`]: the boxes with a V
//! row and the pairs per class,
//! [`with_diagonal_shapes`](HelmholtzOperator::with_diagonal_shapes), which the builder counts
//! in its plan and sums over the ranks; the uniform level's without them): on an RCR level
//! against RCR ([`diagonal_beats_rcr`]); under [`HelmholtzM2l::Auto`] on a level with waves
//! against the waves at their degree ([`diagonal_beats_plane_waves`]: the waves cost per box,
//! the form per pair, so the order turns on the pairs per box). A level taken from the waves
//! runs at its own degree, not the waves' raised one, takes no batched tables, and runs the
//! cheaper of the form and RCR by the same model (where RCR is the cheaper, it also beats the
//! waves: at 10⁻⁸ the waves' raised degree made a sparse level 31× RCR's on Kathleen). The builder's V rule reads `wave_levels` before the plan, a
//! superset of the operator's levels with waves; its grouping per (parent, source) converts
//! exact pairs under every M2L. Each choice is a function of agreed counts, (p, k̂, ε, T) and
//! agreed constants, so the same on every rank and at every rank count; the scratch budget's
//! whole-level fallback comes first. It takes no level below k a = [`DIAGONAL_AUTO_MIN_KA`]
//! (2.8, S6's calibrated range).
//!
//! Every M2L level call, under every setting, is timed on the calling thread's clock and
//! summed per level ([`m2l_times`](HelmholtzOperator::m2l_times); Helmholtz CH6.2b): two
//! clock reads per call, no value changed; so is every P2P level call
//! ([`p2p_times`](HelmholtzOperator::p2p_times); Helmholtz CH4.0d2), and every level call of
//! the other kinds ([`kind_times`](HelmholtzOperator::kind_times), in the order of
//! [`TIMED_KINDS`]; Helmholtz CH4.0d3). The exponential M2L's level call also times its
//! two passes per chunk, vector and frame, the waves and merge and shift
//! ([`plane_wave_times`](HelmholtzOperator::plane_wave_times); Helmholtz CH6.2d): four clock
//! reads more, no value changed; and merge and shift's thread-seconds by part, on each
//! worker's own clock (Helmholtz CH6.2f): its tasks (busy), the accumulators' shifts, the
//! children's sums and shifts, and the conversions, a few clock reads per parent or child
//! and frame, drained from the scratch sets after the level call. The diagonal level call
//! times its two passes, S → F and the targets, and the workers' thread-seconds of S → F, the
//! diagonal pairs, F → R and the RCR pairs, the same way
//! ([`diagonal_times`](HelmholtzOperator::diagonal_times); Helmholtz CH4.2b2), with each
//! level's tables' build ([`diagonal_build_time`](HelmholtzOperator::diagonal_build_time)).
//!
//! # The tables' build
//!
//! Every setting that changes the tables chooses them again (`select_tables`), each level's
//! M2L first, then the RCR tables: built again only where their degrees change, and with
//! the M2L y-blocks stored for the levels whose M2L is RCR's alone (Helmholtz CH4.0h; a
//! level with waves, Batched, Dense or Classes never reads them). The builds run their parts
//! (`nd_fmm_tables::parts`) on the operator's pool, each part a function of its inputs alone,
//! so the tables, and the output, are the serial build's bit for bit at every thread count;
//! the scratch sets, one per thread, are made on the pool too. An operator from
//! [`HelmholtzOperator::unbuilt`] builds nothing until
//! [`build_tables`](HelmholtzOperator::build_tables), so that `HelmholtzFmmBuilder` names
//! every setting and the pool first and builds the tables once (before CH4.0h the builder's
//! operator built the RCR tables at the degrees given, and again where the exponential
//! M2L's rule raised a level, and Batched built a second set of every level's tables in
//! f64). [`HelmholtzOperator::table_timings`] reports the build by part.
//!
//! # Execution
//!
//! The level calls run on the shared drivers of `nd-fmm-exec` (Helmholtz CH2.0, CH3.5),
//! target by target in the plan's order, serially or on the operator's pool
//! ([`HelmholtzOperator::with_pool`]), each thread with its own scratch; the output is the
//! same at every thread count. The shared batched M2L does not apply (`driver::m2l_block`
//! assumes Laplace's layout); under [`HelmholtzM2l::Batched`] the M2L level call of a level
//! with tables runs the operator's own block instead, and under [`HelmholtzM2l::PlaneWave`]
//! a level with waves merge and shift ("M2L"). [`PairOperator`] runs the same
//! bodies pair by pair.
//!
//! A Rotation (RCR) level splits by pairs where its boxes are few (Helmholtz CH4.0i;
//! [`RcrSplit`], [`with_rcr_split`](HelmholtzOperator::with_rcr_split)): target by target
//! one task holds a box's whole V row, so a level of fewer boxes than threads leaves threads
//! idle (level 2 of W1 at kD = 500, 64 boxes at p = 131). By pairs, the level's V pairs in
//! the plan's order run in chunks that fit a buffer
//! ([`with_rcr_buffer`](HelmholtzOperator::with_rcr_buffer)): every pair of a chunk writes
//! its contribution, the terms `HelmholtzTables::m2l` adds to the local, into a slot of its
//! own on the pool, the chunk's pairs by polar angle and distance so that consecutive pairs
//! share their y-blocks (which the scratch keeps where they are formed per call) and coaxial
//! table; then every target adds its pairs' slots in its row's order. Each value of a local
//! receives the same terms, computed by the same operations and added in the same order, so
//! the output is target by target's bit for bit under every split, thread count and buffer,
//! and [`RcrSplit::Auto`] may read the pool's threads: by pairs on a level of degree at least
//! [`RCR_PAIR_MIN_DEGREE`], and of degree at least [`RCR_PAIR_FEW_DEGREE`] where its targets
//! with a V row are fewer than [`RCR_PAIR_SHARE`] times them ([`rcr_by_pairs`]).
//!
//! Yukawa (Re k = 0) is not served: its real path is CH2.1's.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mpi::traits::Equivalence;
use nd_fmm_math::{Complex, Layout};
use nd_fmm_plan::lists::NOFFSETS;
use nd_fmm_plan::operator::{
    FmmOperator, FmmSizes, L2l, L2p, M2l, M2m, M2p, P2l, P2m, P2p, PairOperator,
};
use nd_fmm_ref::Frame;
use nd_fmm_ref::helmholtz::{Level, Workspace, leaf, p2p};
use nd_fmm_simd::{
    HelmholtzCharges, Isa, IsaUnavailable, MatrixKernel, MulAdd, P2pKernel, SimdScalar,
};
use nd_fmm_tables::MatrixSet;
use nd_fmm_tables::helmholtz::batched::BatchedM2l;
use nd_fmm_tables::helmholtz::classes::{ClassM2l, ClassScratch};
use nd_fmm_tables::helmholtz::dense::DenseM2l;
use nd_fmm_tables::helmholtz::plane_wave::Precision;
use nd_fmm_tables::helmholtz::plane_wave::merge::{
    PlaneWaveM2l, RESOLVING_STEP_TOLERANCE, plane_wave_degree, select,
};
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{RHO_MAX, assert_reach, domain_of};
use nd_fmm_tables::helmholtz::{
    DEFAULT_BLOCK_BUDGET, HelmholtzScratch, HelmholtzTables, TableChoice,
};
use nd_fmm_tables::parts::Parts;
use nd_octree::{MortonKey, morton};
use rayon::ThreadPool;
use rayon::prelude::*;
use thiserror::Error;

use super::batched::Batched;
use super::{Densities, LeafScratch, Wavenumber, WavenumberKind};
use crate::driver::{self, Columns, Execution, HostKernel, debug_check_octant, offset_index};
use crate::geometry::{Domain, radius, relative_frame};
use crate::operator::{M2L_BLOCK_TARGETS, P2pChoice};

#[path = "operator_diagonal.rs"]
mod diagonal;
#[path = "operator_plane_wave.rs"]
mod plane_wave;
#[path = "operator_rcr.rs"]
mod rcr;

pub use diagonal::{
    DEFAULT_DIAGONAL_BUILD, DIAGONAL_AUTO_MIN_KA, DIAGONAL_COST_CONSTANTS, DIAGONAL_COSTS_NEON,
    DIAGONAL_COSTS_X86, DIAGONAL_F32_MIN_TOLERANCE, DiagonalCosts, DiagonalCounts, DiagonalShape,
    DiagonalTimes, HelmholtzDiagonal, WaveLevelChoice, diagonal_auto_in_range,
    diagonal_beats_plane_waves, diagonal_beats_rcr, diagonal_chunk_targets, diagonal_costs,
    diagonal_rule_classes, offset_class, wave_level_choice,
};
use diagonal::{DiagonalBuffer, DiagonalLevel, DiagonalScratch, DiagonalSettings};
pub use nd_fmm_tables::helmholtz::diagonal::TransferBuild;
pub use plane_wave::{MIN_BUFFER_BOXES, PlaneWaveCounts, PlaneWaveSplit, PlaneWaveTimes};
use plane_wave::{PlaneBuffer, PlaneScratch};
use rcr::RcrBuffer;
pub use rcr::{
    DEFAULT_RCR_BUFFER, RCR_PAIR_FEW_DEGREE, RCR_PAIR_MIN_DEGREE, RCR_PAIR_SHARE, RcrCounts,
    RcrSplit, rcr_by_pairs,
};

/// Why [`HelmholtzOperator::new`] refused its inputs.
#[derive(Clone, Copy, Debug, Error, PartialEq)]
pub enum OperatorError {
    /// k is Yukawa's (Re k = 0), whose real path is Helmholtz CH2.1's.
    #[error("Yukawa (Re k = 0) is not served by this operator")]
    Yukawa,
    /// The densities are of another kind of k.
    #[error("the densities are for {densities:?} k, the wavenumber is {k:?}")]
    KindMismatch {
        /// The kind of the densities.
        densities: WavenumberKind,
        /// The kind of the wavenumber.
        k: WavenumberKind,
    },
    /// No degrees, so no levels.
    #[error("no levels")]
    NoLevels,
}

/// The M2L of the Helmholtz operator's levels (module documentation, "M2L"; proposal §5.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HelmholtzM2l {
    /// RCR on every level (the operator's default before
    /// [`with_m2l`](HelmholtzOperator::with_m2l)): rotation, coaxial translation, rotation
    /// back.
    Rotation,
    /// One complex matrix per offset on the levels whose tables fit the dense budget, from
    /// the finest level up; RCR on the others.
    Dense,
    /// The 16 class matrices per level and the cube-group transforms, on the levels whose
    /// tables fit the dense budget, from the finest level up; RCR on the others.
    Classes,
    /// Dense's matrices in split planes, applied block by block by the column kernel, on the
    /// levels whose tables fit the dense budget, from the finest level up; RCR on the others.
    Batched,
    /// The exponential M2L by merge and shift (Helmholtz CH6.1d; opt-in) on the levels with
    /// a quadrature of the [`PlaneWavePrecision`] setting
    /// ([`with_plane_wave_precision`](HelmholtzOperator::with_plane_wave_precision)), RCR on
    /// the others. f64 only (CONVENTIONS §3.15, "Precision and range"): in f32 the factors,
    /// up to about 10³⁴ at iprec 3, are untested and the setting is not refused.
    PlaneWave,
    /// The rule per level (Helmholtz CH3.2d): Batched, in [`auto_m2l_mul_add`]'s mode (fused
    /// on every vector ISA) unless one is named,
    /// on the levels of degree at most the crossover of the machine class
    /// ([`auto_batched_max_degree`]) whose tables fit the dense budget, from the finest level
    /// up; RCR on the others. The default of `HelmholtzFmmBuilder` (and of this type) since
    /// Helmholtz CH3.2d. Since Helmholtz CH6.2h (D8; off with
    /// [`with_auto_plane_wave`](HelmholtzOperator::with_auto_plane_wave)) the exponential M2L
    /// first, on the levels of degree at least
    /// [`auto_plane_wave_min_degree`] with a quadrature of the rule and Im κ_l below
    /// [`AUTO_PLANE_WAVE_MAX_IM_KAPPA`], in f64 only.
    #[default]
    Auto,
}

/// The precision of each level's quadrature under [`HelmholtzM2l::PlaneWave`] (Helmholtz
/// CH6.1d; module documentation, "M2L").
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PlaneWavePrecision {
    /// `nd_fmm_tables::helmholtz::plane_wave::merge::select`: on each level with a V list
    /// whose κ_l = k w_l lies in the quadrature's tables, the precision
    /// `precision_for_tolerance` gives the FMM's tolerance
    /// ([`HelmholtzOperator::with_plane_wave_tolerance`]), or without one the precision
    /// `precision_for_degree` gives the level's degree (measured in the FMM, Helmholtz
    /// CH6.1e), none elsewhere. Below a tolerance of 10⁻⁶ each level with a precision runs
    /// at the raised degree `plane_wave_degree` gives (Helmholtz CH6.1g; module
    /// documentation, "M2L"). A function of k, the level's width, its degree and the
    /// tolerance only (proposal §2, constraint 9).
    #[default]
    Rule,
    /// The precision of each level by index, `None` for RCR, for the measurements and the
    /// tests, at the levels' degrees as given (never raised). Entries for levels 0 and 1 (no V list) and beyond the tree are ignored, and
    /// levels beyond the list take none; a level whose κ_l lies outside the tables runs RCR
    /// whatever it names.
    PerLevel(Vec<Option<Precision>>),
}

/// Which operators of a level with waves run at the resolving degree p_w where
/// [`PlaneWavePrecision::Rule`] raises it below ε = 10⁻⁶ (Helmholtz CH6.1g, CH6.1h; module
/// documentation, "M2L"). In every form the level's stores (and so the ghost exchange) and
/// its M2L are at p_w, and so are the M2M from its children (it fills all p_w degrees from
/// the children's) and the L2L to them (it reads all p_w). The outputs of the forms differ;
/// each is deterministic and rank-invariant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaneWaveRaise {
    /// The rule (the default; Helmholtz CH6.1h, signed off 2026-11-11): [`Self::M2l`] at
    /// 10⁻⁷ ≤ ε < 10⁻⁶ and [`Self::Leaves`] at 10⁻⁸ ≤ ε < 10⁻⁷ (`M2l` erred 1.53× Rotation's
    /// at 10⁻⁸ on the clusters at kD = 10; fmm-validate/results/helmholtz-ch6-1h.md), a
    /// function of ε only ([`resolved`](Self::resolved)).
    #[default]
    Rule,
    /// Every operator of the level at p_w (Helmholtz CH6.1g): also the M2M to its parent
    /// and the L2L from it, and its leaves' operators (P2M, L2P, P2L into and M2P from its
    /// boxes).
    Level,
    /// The leaves' operators at p_w, the M2M to the parent from the first p_l degrees and
    /// the L2L from the parent into them (Helmholtz CH6.1h).
    Leaves,
    /// The M2L alone at p_w (Helmholtz CH6.1h): the leaves' operators, the M2M to the parent
    /// and the L2L from it on the first p_l degrees. A leaf's multipole then has no degree
    /// above p_l, and its L2P reads none.
    M2l,
}

impl PlaneWaveRaise {
    /// The form that runs at the FMM's tolerance: [`Self::Rule`]'s choice, [`Self::M2l`]
    /// for ε ≥ 10⁻⁷ (`RESOLVING_STEP_TOLERANCE`; also without a tolerance, where nothing is
    /// raised) and [`Self::Leaves`] below; a named form as named.
    pub fn resolved(self, tolerance: Option<f64>) -> Self {
        match self {
            Self::Rule => match tolerance {
                Some(epsilon) if epsilon < RESOLVING_STEP_TOLERANCE => Self::Leaves,
                _ => Self::M2l,
            },
            named => named,
        }
    }
}

/// The kinds of [`HelmholtzOperator::kind_times`], in its order (Helmholtz CH4.0d3).
pub const TIMED_KINDS: [&str; 6] = ["P2M", "M2M", "L2L", "L2P", "P2L", "M2P"];

/// The crossover degree of [`HelmholtzM2l::Auto`] (Helmholtz CH3.2d; the operator module
/// documentation, "M2L"): the largest p_l at which the rule runs the fused Batched M2L, for
/// the column kernel's ISA (the machine class) and the multipole's planes (1 for real k with
/// real charges, 2 otherwise). A function of these two alone, so every rank of one machine
/// class chooses alike (proposal §2, constraint 9). Set from the end-to-end crossover sweeps
/// of fmm-validate/results/helmholtz-ch3-2d.md: NEON on the M6 and locust, AVX-512 and AVX2
/// on Kathleen; the scalar kernel from its cost per pair on the M6 and Kathleen.
pub fn auto_batched_max_degree(isa: Isa, planes: usize) -> usize {
    let one_plane = planes == 1;
    match (isa, one_plane) {
        (Isa::Avx512, _) => 15,
        (Isa::Avx2, false) => 11,
        (Isa::Scalar, false) => 6,
        (Isa::Scalar, true) => 8,
        // NEON (and any later ISA until measured); AVX2 with one plane.
        (_, false) => 10,
        (_, true) => 12,
    }
}

/// The smallest degree at which [`HelmholtzM2l::Auto`] with the exponential M2L
/// ([`HelmholtzOperator::with_auto_plane_wave`]; Helmholtz CH6.2h, D8) takes it, for the
/// column kernel's ISA and the multipole's planes: one above Batched's crossover
/// ([`auto_batched_max_degree`]), so the exponential M2L takes the levels Rotation ran,
/// except AVX-512 with two planes, 12, where it beats Batched on most levels of degree 13
/// to 15 (fmm-validate/results/helmholtz-ch6-2b.md, §5; helmholtz-ch6-2h.md). A function
/// of these two alone, so every rank of one machine class chooses alike (proposal §2,
/// constraint 9).
pub fn auto_plane_wave_min_degree(isa: Isa, planes: usize) -> usize {
    match (isa, planes == 1) {
        (Isa::Avx512, false) => 12,
        _ => auto_batched_max_degree(isa, planes) + 1,
    }
}

/// The bound on Im κ_l, κ_l = k w_l the level's wavenumber in box widths, below which
/// [`HelmholtzM2l::Auto`] with the exponential M2L takes it (Helmholtz CH6.2h): 5π, the
/// domains whose default counts nd-fmm-exec's `tests/helmholtz/plane_wave_reach.rs` guards.
/// From 5π the field across a V offset is below about 2·10⁻¹⁴ of the near field and both
/// counts' chains err 10⁻⁹–10⁻⁵ absolute (fmm-validate/results/helmholtz-ch6-2g.md, §3);
/// such levels run Batched or RCR.
pub const AUTO_PLANE_WAVE_MAX_IM_KAPPA: f64 = 5.0 * std::f64::consts::PI;

/// The settings that choose the levels whose M2L is the exponential one ([`wave_levels`]):
/// those of [`HelmholtzOperator`] and `HelmholtzFmmBuilder` alike, the builder's agreed over
/// the ranks (Helmholtz CHA.11).
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaveSettings<'a> {
    /// The M2L setting ([`HelmholtzOperator::with_m2l`]).
    pub(crate) m2l: HelmholtzM2l,
    /// The precisions under [`HelmholtzM2l::PlaneWave`]
    /// ([`HelmholtzOperator::with_plane_wave_precision`]).
    pub(crate) precision: &'a PlaneWavePrecision,
    /// Whether [`HelmholtzM2l::Auto`] may take the exponential M2L
    /// ([`HelmholtzOperator::with_auto_plane_wave`]).
    pub(crate) auto_plane_wave: bool,
    /// The smallest degree at which [`HelmholtzM2l::Auto`] takes it
    /// ([`HelmholtzOperator::plane_wave_min_degree`]).
    pub(crate) min_degree: usize,
    /// The FMM's tolerance, if it has one
    /// ([`HelmholtzOperator::with_plane_wave_tolerance`]).
    pub(crate) tolerance: Option<f64>,
}

/// The levels whose M2L is the exponential one, each with its quadrature's precision (`None`
/// on the others), or `None` where the setting runs no exponential M2L on any level (Helmholtz
/// CHA.11): under [`HelmholtzM2l::PlaneWave`] the rule's (`merge::select` of k̂_0 = k r_0, the
/// degrees `base` the levels were given and the tolerance) or the named precisions on the
/// levels from 2 whose κ_l = 2 k̂_0 / 2^l lies in the tables (`domain_of`; the tables build
/// none elsewhere); under [`HelmholtzM2l::Auto`] with waves, in f64 only, the rule's on the
/// levels of degree at least the smallest with Im κ_l < [`AUTO_PLANE_WAVE_MAX_IM_KAPPA`]
/// (Helmholtz CH6.2h, D8). A function of the settings and the degrees alone:
/// `HelmholtzFmmBuilder` reads these levels before the plan for the V rule (Helmholtz
/// CH4.0d2), and the operator builds their tables, less, under [`HelmholtzDiagonal::Auto`],
/// those the diagonal form's time rule takes after the plan (Helmholtz CH4.3b1): the
/// operator's levels with waves are these or fewer, never more. The V rule's grouping per
/// (parent, source) converts exact pairs under every M2L, so a level the form takes keeps
/// an exact plan.
pub(crate) fn wave_levels<T>(
    settings: &WaveSettings<'_>,
    k_root: Complex<f64>,
    base: &[usize],
) -> Option<Vec<Option<Precision>>> {
    // `HelmholtzTables`' k̂_l, doubled, as `PlaneWaveM2l` forms it.
    let kappa = |l: usize| k_root * (0.5f64).powi(l as i32) * 2.0;
    match settings.m2l {
        HelmholtzM2l::PlaneWave => Some(match settings.precision {
            PlaneWavePrecision::Rule => select(k_root, base, settings.tolerance),
            PlaneWavePrecision::PerLevel(named) => (0..base.len())
                .map(|l| {
                    named
                        .get(l)
                        .copied()
                        .flatten()
                        .filter(|_| l >= 2 && domain_of(kappa(l)).is_some())
                })
                .collect(),
        }),
        m2l if auto_takes_plane_waves::<T>(m2l, settings.auto_plane_wave) => Some(
            select(k_root, base, settings.tolerance)
                .into_iter()
                .enumerate()
                .map(|(l, precision)| {
                    precision.filter(|_| {
                        base[l] >= settings.min_degree && kappa(l).im < AUTO_PLANE_WAVE_MAX_IM_KAPPA
                    })
                })
                .collect(),
        ),
        _ => None,
    }
}

/// Whether the setting is [`HelmholtzM2l::Auto`] with the exponential M2L, which runs in f64
/// only (CONVENTIONS §3.15, "Precision and range").
fn auto_takes_plane_waves<T>(m2l: HelmholtzM2l, auto_plane_wave: bool) -> bool {
    m2l == HelmholtzM2l::Auto
        && auto_plane_wave
        && std::mem::size_of::<T>() == std::mem::size_of::<f64>()
}

/// The column kernel's mode of [`HelmholtzM2l::Auto`] on `isa` unless one is named
/// (Helmholtz CH3.2d): [`MulAdd::Fused`], where each step is one hardware fma, but
/// [`MulAdd::Separate`] for the scalar kernel in a build whose target has no fma
/// instruction (x86_64 without the `fma` target feature), where `mul_add` is a library
/// call: about 10× Rotation's cost per pair on Kathleen at p = 6
/// (fmm-validate/results/helmholtz-ch3-2d.md, §3). A function of the ISA and the build only,
/// so every rank of one machine class and build chooses alike.
pub fn auto_m2l_mul_add(isa: Isa) -> MulAdd {
    let hardware_fma = cfg!(any(target_arch = "aarch64", target_feature = "fma"));
    match isa {
        Isa::Scalar if !hardware_fma => MulAdd::Separate,
        _ => MulAdd::Fused,
    }
}

/// The V list of one level over the whole tree (Helmholtz CH4.0): its pairs and its boxes,
/// each counted once over the ranks, by the rank that computes the box (`Plan::computes`,
/// the replicated top included). Integers summed over the ranks, so the same on every rank
/// and at every rank count; `HelmholtzFmmBuilder` counts them from its plan
/// ([`with_level_shapes`](HelmholtzOperator::with_level_shapes)).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LevelShape {
    /// The V pairs whose target is a box of the level.
    pub pairs: u64,
    /// The boxes of the level.
    pub targets: u64,
}

impl LevelShape {
    /// m̂, the estimate of the columns per table read of Batched's level call on this level
    /// at one rank (the operator module documentation, "M2L"): the pairs over the (block,
    /// offset) visits of ⌈targets / 64⌉ blocks of [`M2L_BLOCK_TARGETS`] consecutive boxes,
    /// each block of b̄ boxes (the mean) meeting 316 (1 − (1 − L̄/316)^b̄) distinct offsets of
    /// the 316, L̄ = pairs / targets the mean V row. 0 without pairs. A function of the two
    /// counts alone, so every rank computes the same value.
    pub fn batched_columns(&self) -> f64 {
        if self.pairs == 0 || self.targets == 0 {
            return 0.0;
        }
        let (pairs, targets) = (self.pairs as f64, self.targets as f64);
        let blocks = self.targets.div_ceil(M2L_BLOCK_TARGETS as u64) as f64;
        let per_block = targets / blocks;
        let offsets = NOFFSETS as f64;
        let row = (pairs / targets).min(offsets);
        let visited = offsets * (1.0 - (1.0 - row / offsets).powf(per_block));
        pairs / (blocks * visited)
    }
}

/// m*, the fewest columns per table read at which [`HelmholtzM2l::Auto`] runs Batched on a
/// level of degree p (Helmholtz CH4.0), for the column kernel's ISA and the multipole's
/// planes (fmm-validate/results/helmholtz-ch4-0.md, §1–§2): the column count c at which the
/// fused Batched M2L per pair reaches the class's margin over Rotation's speed in
/// `helmholtz_m2l`, from a fit of Batched's cost as a/c + b per degree, the margin the
/// level call needs over the pair (1.45 on AVX-512 and AVX2 with two planes, 1.2 with one,
/// 1.15 on NEON, from the FMM's per-level times at one thread); never falling as p grows;
/// 64 (a full block's columns, which no level reaches) from p = 16 and where Batched never
/// reaches the margin. NEON's is the larger of the M6's and locust's, but at p = 11 with one
/// plane the M6's (locust's pairs there are not monotone in c, and on both machines the
/// level of 24 columns per read runs Batched faster); the scalar kernel (the reference
/// path) takes NEON's, unmeasured. Auto runs Batched on a level only where
/// [`LevelShape::batched_columns`] is at least this and p_l at most its crossover
/// ([`auto_batched_max_degree`] or the degree named), a condition of its own. A function of
/// the machine class and p alone (proposal §2, constraint 9).
pub fn auto_batched_min_columns(isa: Isa, planes: usize, p: usize) -> f64 {
    const AVX512_ONE: [f64; 16] = [
        0.4, 0.4, 0.4, 0.4, 0.6, 0.8, 0.9, 1.1, 1.6, 2.5, 3.1, 3.8, 4.6, 5.4, 9.1, 11.8,
    ];
    const AVX512_TWO: [f64; 16] = [
        0.7, 0.7, 0.7, 0.7, 0.9, 1.2, 1.3, 1.6, 2.4, 3.9, 4.6, 5.7, 7.0, 10.4, 15.9, 28.2,
    ];
    const AVX2_ONE: [f64; 16] = [
        0.3, 0.3, 0.3, 0.3, 0.5, 0.7, 0.9, 1.2, 1.9, 3.1, 3.9, 10.6, 14.5, 16.3, 32.3, 41.3,
    ];
    const AVX2_TWO: [f64; 16] = [
        0.6, 0.6, 0.6, 0.6, 0.8, 1.0, 1.4, 2.1, 3.1, 5.6, 7.5, 37.5, 64.0, 64.0, 64.0, 64.0,
    ];
    const NEON_ONE: [f64; 16] = [
        0.3, 0.3, 0.3, 0.3, 0.6, 0.8, 1.3, 2.1, 3.2, 8.0, 13.7, 13.7, 64.0, 64.0, 64.0, 64.0,
    ];
    const NEON_TWO: [f64; 16] = [
        0.4, 0.4, 0.4, 0.4, 0.8, 1.2, 2.1, 3.9, 7.5, 23.7, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0,
    ];
    let table = match (isa, planes == 1) {
        (Isa::Avx512, true) => &AVX512_ONE,
        (Isa::Avx512, false) => &AVX512_TWO,
        (Isa::Avx2, true) => &AVX2_ONE,
        (Isa::Avx2, false) => &AVX2_TWO,
        // NEON, the scalar kernel (unmeasured) and any later ISA until measured.
        (_, true) => &NEON_ONE,
        (_, false) => &NEON_TWO,
    };
    table.get(p).copied().unwrap_or(M2L_BLOCK_TARGETS as f64)
}

/// The narrowest block of Batched's level call under the split of
/// [`batched_block_targets`] (Helmholtz CH4.0).
pub const M2L_MIN_BLOCK_TARGETS: usize = 8;

/// The width of Batched's blocks on a level of `rows` boxes run on `threads` threads
/// (Helmholtz CH4.0): [`M2L_BLOCK_TARGETS`] on one thread; on more, ⌈rows / threads⌉ within
/// [`M2L_MIN_BLOCK_TARGETS`] to [`M2L_BLOCK_TARGETS`], so that a level of fewer than
/// `threads` full blocks still gives each thread a block. An execution split: the level call
/// is the pair chain bit for bit at every width (the operator module documentation, "M2L";
/// proposal §2, constraint 9 as amended at H6.G).
pub fn batched_block_targets(rows: usize, threads: usize) -> usize {
    if threads <= 1 {
        return M2L_BLOCK_TARGETS;
    }
    rows.div_ceil(threads)
        .clamp(M2L_MIN_BLOCK_TARGETS, M2L_BLOCK_TARGETS)
}

/// The default byte budget of the dense or class M2L tables of one rank, 1 GiB.
pub const DEFAULT_DENSE_BUDGET: usize = 1 << 30;

/// The wall time of building the operator's tables, by part, summed over its construction
/// and every later choice of its M2L ([`HelmholtzOperator::table_timings`]; Helmholtz
/// CH4.0h). This rank's, on the calling thread's clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TableTimings {
    /// The RCR tables (`HelmholtzTables`): the translations' and M2L's coaxial tables and
    /// the y-blocks, every time they are built.
    pub rcr: Duration,
    /// The matrices of Dense, Classes and Batched (their own f64 RCR tables included).
    pub matrices: Duration,
    /// The exponential M2L's tables (`PlaneWaveM2l`).
    pub plane_waves: Duration,
    /// The diagonal form's transfer functions and conversions (`DiagonalM2l`, `Conversions`;
    /// Helmholtz CH4.2b1).
    pub diagonal: Duration,
    /// The scratch sets of every thread, sized for the tables.
    pub scratch: Duration,
}

impl TableTimings {
    /// The sum of the parts.
    pub fn total(&self) -> Duration {
        self.rcr + self.matrices + self.plane_waves + self.diagonal + self.scratch
    }
}

/// The Helmholtz-family operator (module documentation).
pub struct HelmholtzOperator<T: SimdScalar> {
    kernel: Kernel<T>,
    execution: Execution<Scratch<T>>,
    /// Under [`HelmholtzM2l::PlaneWave`], the level call's waves of one frame of a chunk's
    /// sources, up and down, and their positions.
    plane_buffer: PlaneBuffer<T>,
    /// The Rotation M2L's buffer of contributions by pairs (Helmholtz CH4.0i).
    rcr_buffer: RcrBuffer<T>,
    /// The diagonal M2L's source signatures and slots (Helmholtz CH4.2b1).
    diagonal_buffer: DiagonalBuffer<T>,
    /// The M2L level calls' wall time per level since the tables were last chosen
    /// ([`m2l_times`](HelmholtzOperator::m2l_times)).
    m2l_times: Vec<Duration>,
    /// The P2P level calls' wall time per level since the tables were last chosen
    /// ([`p2p_times`](HelmholtzOperator::p2p_times)).
    p2p_times: Vec<Duration>,
    /// The other kinds' level calls' wall time per level since the tables were last chosen
    /// ([`kind_times`](HelmholtzOperator::kind_times)).
    kind_times: Vec<[Duration; 6]>,
    /// The exponential M2L's passes per level since the tables were last chosen
    /// ([`plane_wave_times`](HelmholtzOperator::plane_wave_times)).
    plane_wave_times: Vec<PlaneWaveTimes>,
    /// The tables' build time by part since construction
    /// ([`table_timings`](HelmholtzOperator::table_timings)).
    table_timings: TableTimings,
    /// Whether the tables wait for [`build_tables`](HelmholtzOperator::build_tables)
    /// ([`unbuilt`](HelmholtzOperator::unbuilt)).
    unbuilt: bool,
}

/// The parts of a table build on the operator's pool, or on the calling thread without one
/// (Helmholtz CH4.0h; `nd_fmm_tables::parts`): rayon's indexed collect keeps the parts'
/// order, so the tables are the serial build's bit for bit.
struct PoolParts<'a>(Option<&'a ThreadPool>);

impl Parts for PoolParts<'_> {
    fn map<R: Send>(&self, n: usize, f: &(dyn Fn(usize) -> R + Sync)) -> Vec<R> {
        match self.0 {
            Some(pool) => pool.install(|| (0..n).into_par_iter().map(f).collect()),
            None => (0..n).map(f).collect(),
        }
    }
}

/// What the bodies read: the path, the levels and the tables.
struct Kernel<T: SimdScalar> {
    k: Wavenumber,
    densities: Densities,
    gradients: bool,
    max_leaf_points: usize,
    /// r_l of each level, in the units of the points.
    radii: Vec<f64>,
    /// The degrees the operator was given; the tables' may be raised (Helmholtz CH6.1g).
    base_degrees: Vec<usize>,
    /// The byte budget of the stored M2L y-blocks, for the tables built again.
    block_budget: usize,
    tables: HelmholtzTables<T>,
    /// The M2L setting.
    m2l: HelmholtzM2l,
    /// The byte budget of the M2L tables ([`HelmholtzOperator::with_m2l`]).
    dense_budget: usize,
    /// The crossover degree named for [`HelmholtzM2l::Auto`], or `None` for the rule's.
    max_degree: Option<usize>,
    /// Whether [`HelmholtzM2l::Auto`] takes the exponential M2L (Helmholtz CH6.2h).
    auto_plane_wave: bool,
    /// The smallest degree at which Auto takes the exponential M2L if named, else `None`
    /// for the rule's ([`auto_plane_wave_min_degree`]).
    plane_wave_min_degree: Option<usize>,
    /// The levels' V lists over the whole tree, which Auto's term for Batched reads
    /// (Helmholtz CH4.0); `None`: no term.
    shapes: Option<Vec<LevelShape>>,
    /// m* by degree if named (an agreement over ranks), else `None` for
    /// [`auto_batched_min_columns`]'s.
    min_columns: Option<Vec<f64>>,
    /// The width of Batched's blocks if named, else `None` for [`batched_block_targets`].
    block_targets: Option<usize>,
    /// The batched M2L's multiply–add mode if named, else `None` (the setting's default).
    mul_add: Option<MulAdd>,
    /// The dense M2L tables of the levels that have them, under [`HelmholtzM2l::Dense`].
    dense: Option<DenseM2l<T>>,
    /// The class M2L tables of the levels that have them, under [`HelmholtzM2l::Classes`].
    classes: Option<ClassM2l<T>>,
    /// The batched M2L tables of the levels that have them, under [`HelmholtzM2l::Batched`].
    batched: Option<BatchedM2l<T>>,
    /// The quadratures' precisions under [`HelmholtzM2l::PlaneWave`].
    plane_wave_precision: PlaneWavePrecision,
    /// The FMM's tolerance, which [`PlaneWavePrecision::Rule`] reads, if it has one.
    plane_wave_tolerance: Option<f64>,
    /// The lateral reach of the quadratures' α counts (`RHO_MAX`, the region's corner,
    /// unless named; Helmholtz CH6.2e).
    plane_wave_reach: f64,
    /// Which operators of a raised level run at p_w (Helmholtz CH6.1h).
    plane_wave_raise: PlaneWaveRaise,
    /// The exponential M2L's tables of the levels that have them, under
    /// [`HelmholtzM2l::PlaneWave`].
    plane_waves: Option<PlaneWaveM2l<T>>,
    /// The diagonal M2L's settings (Helmholtz CH4.2b1).
    diagonal_settings: DiagonalSettings,
    /// The diagonal M2L's tables per level, where a level takes it.
    diagonal: Vec<Option<DiagonalLevel<T>>>,
    /// The levels the time rule took from the waves (Helmholtz CH4.3b2a), per level.
    from_waves: Vec<bool>,
    /// The batched M2L's column kernel, on the P2P kernel's ISA (scalar for the reference).
    matrix: MatrixKernel<T>,
    /// The P2P kernel of `nd-fmm-simd`, or `None` for the reference.
    p2p: Option<P2pKernel<T>>,
}

/// One thread's scratch.
struct Scratch<T: SimdScalar> {
    leaf: LeafScratch<T>,
    workspace: Workspace<T>,
    tables: HelmholtzScratch<T>,
    /// The class M2L's rotated multipole and translated local.
    classes: ClassScratch<T>,
    /// The leaf operators' expansions at a level's translation degree, under
    /// [`PlaneWaveRaise::M2l`] (Helmholtz CH6.1h): m locals at the largest degree.
    first: Vec<T>,
    /// Another leaf's sources in the target's units (P2P).
    mapped: Vec<[T; 3]>,
    /// One vector's charges.
    charges: Vec<Complex<T>>,
    /// One vector's charges for the P2P kernel, Re q and Im q per source.
    strengths: Vec<T>,
    /// One multipole of two planes (M2P of a real multipole).
    widened: Vec<T>,
    /// The batched M2L's packing buffer and column lists (sized for the batched levels).
    columns: Columns<T>,
    /// The exponential M2L's buffers, under [`HelmholtzM2l::PlaneWave`].
    plane: Option<PlaneScratch<T>>,
    /// The diagonal M2L's buffers, where a level takes it (Helmholtz CH4.2b1).
    diagonal: Option<DiagonalScratch<T>>,
}

impl<T: SimdScalar> HelmholtzOperator<T> {
    /// The operator of an octree on `domain` with `degrees.len()` levels of degrees p_l,
    /// for `k` and the path `densities` (its kind must be k's), with gradients in the
    /// target output if `gradients`, for leaves with at most `max_leaf_points` source
    /// points (every leaf P2P reads, local and ghost: on several ranks a global maximum).
    /// Builds the tables of every level with the default block budget
    /// ([`DEFAULT_BLOCK_BUDGET`]) and one scratch set; runs serially until
    /// [`with_pool`](Self::with_pool), with the P2P kernel of [`P2pChoice::Auto`] until
    /// [`with_p2p`](Self::with_p2p).
    ///
    /// # Errors
    ///
    /// [`OperatorError`]: Yukawa, a kind mismatch, no degrees.
    pub fn new(
        k: Wavenumber,
        densities: Densities,
        domain: &Domain,
        degrees: &[usize],
        gradients: bool,
        max_leaf_points: usize,
    ) -> Result<Self, OperatorError> {
        Self::with_block_budget(
            k,
            densities,
            domain,
            degrees,
            gradients,
            max_leaf_points,
            DEFAULT_BLOCK_BUDGET,
        )
    }

    /// As [`new`](Self::new), with `block_budget` bytes for the stored M2L y-blocks
    /// (`HelmholtzTables::build`); the output does not depend on it.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_block_budget(
        k: Wavenumber,
        densities: Densities,
        domain: &Domain,
        degrees: &[usize],
        gradients: bool,
        max_leaf_points: usize,
        block_budget: usize,
    ) -> Result<Self, OperatorError> {
        let mut operator = Self::unbuilt(
            k,
            densities,
            domain,
            degrees,
            gradients,
            max_leaf_points,
            block_budget,
        )?;
        operator.unbuilt = false;
        operator.select_tables();
        Ok(operator)
    }

    /// As [`with_block_budget`](Self::with_block_budget), but without the tables
    /// (Helmholtz CH4.0h): the setters record their settings and build nothing, and
    /// [`build_tables`](Self::build_tables) then builds the tables of the settings once, on
    /// the pool's threads if [`with_pool`](Self::with_pool) gave one, the same tables as
    /// the setters applied to [`with_block_budget`](Self::with_block_budget)'s operator in
    /// any order. Until then the operator reports the degrees given and no M2L tables, and a
    /// level call panics. `HelmholtzFmmBuilder` builds its operator so.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn unbuilt(
        k: Wavenumber,
        densities: Densities,
        domain: &Domain,
        degrees: &[usize],
        gradients: bool,
        max_leaf_points: usize,
        block_budget: usize,
    ) -> Result<Self, OperatorError> {
        if k.kind() == WavenumberKind::Yukawa {
            return Err(OperatorError::Yukawa);
        }
        if densities.kind() != k.kind() {
            return Err(OperatorError::KindMismatch {
                densities: densities.kind(),
                k: k.kind(),
            });
        }
        if degrees.is_empty() {
            return Err(OperatorError::NoLevels);
        }
        let radii: Vec<f64> = (0..degrees.len()).map(|l| radius(l, domain)).collect();
        let k_root = Complex::new(k.re() * radii[0], k.im() * radii[0]);
        let mut table_timings = TableTimings::default();
        // No table yet: the levels and their degrees alone.
        let none = vec![false; degrees.len()];
        let choice = TableChoice {
            translations: false,
            m2l_levels: Some(&none),
            ..TableChoice::new(block_budget)
        };
        let tables = HelmholtzTables::build_by(k_root, degrees, &choice, &PoolParts(None));
        let kernel = Kernel {
            k,
            densities,
            gradients,
            max_leaf_points,
            radii,
            base_degrees: degrees.to_vec(),
            block_budget,
            tables,
            m2l: HelmholtzM2l::Rotation,
            dense_budget: DEFAULT_DENSE_BUDGET,
            max_degree: None,
            auto_plane_wave: true,
            plane_wave_min_degree: None,
            shapes: None,
            min_columns: None,
            block_targets: None,
            mul_add: None,
            dense: None,
            classes: None,
            batched: None,
            plane_wave_precision: PlaneWavePrecision::Rule,
            plane_wave_tolerance: None,
            plane_wave_reach: RHO_MAX,
            plane_wave_raise: PlaneWaveRaise::Rule,
            plane_waves: None,
            diagonal_settings: DiagonalSettings::default(),
            diagonal: Vec::new(),
            from_waves: Vec::new(),
            matrix: MatrixKernel::detect(MulAdd::Separate),
            p2p: Some(P2pKernel::detect()),
        };
        let clock = Instant::now();
        let execution = Execution {
            scratch: vec![Mutex::new(Scratch::new(&kernel))],
            pool: None,
            serial: false,
        };
        table_timings.scratch += clock.elapsed();
        let m2l_times = vec![Duration::ZERO; kernel.tables.nlevels()];
        let p2p_times = vec![Duration::ZERO; kernel.tables.nlevels()];
        let kind_times = vec![[Duration::ZERO; 6]; kernel.tables.nlevels()];
        let plane_wave_times = vec![PlaneWaveTimes::default(); kernel.tables.nlevels()];
        Ok(Self {
            kernel,
            execution,
            plane_buffer: PlaneBuffer::default(),
            rcr_buffer: RcrBuffer::default(),
            diagonal_buffer: DiagonalBuffer::default(),
            m2l_times,
            p2p_times,
            kind_times,
            plane_wave_times,
            table_timings,
            unbuilt: true,
        })
    }

    /// Builds the tables of the operator's settings ([`unbuilt`](Self::unbuilt)), once, on
    /// the pool's threads if it has one; nothing more on an operator with tables.
    pub fn build_tables(mut self) -> Self {
        if self.unbuilt {
            self.unbuilt = false;
            self.select_tables();
        }
        self
    }

    /// Runs every level call in `pool`, each target on one of its threads, with one
    /// scratch set per thread. The output is bit for bit the serial operator's.
    pub fn with_pool(mut self, pool: Arc<ThreadPool>) -> Self {
        if self.unbuilt {
            // `build_tables` makes the scratch sets, one per thread, for the tables.
            self.execution.pool = Some(pool);
            return self;
        }
        let n = pool.current_num_threads().max(1);
        let clock = Instant::now();
        self.execution.scratch = (0..n)
            .map(|_| Mutex::new(Scratch::new(&self.kernel)))
            .collect();
        self.table_timings.scratch += clock.elapsed();
        self.execution.pool = Some(pool);
        self.size_rcr_buffer();
        self
    }

    /// Runs M2L by `choice` (module documentation, "M2L"): under [`HelmholtzM2l::Dense`]
    /// builds the dense tables of the levels that fit in `dense_budget` bytes from the
    /// finest level up (`DenseM2l::select`; [`DEFAULT_DENSE_BUDGET`] is a default), under
    /// [`HelmholtzM2l::Classes`] the class tables within the same budget
    /// (`ClassM2l::select`, the transforms included), under [`HelmholtzM2l::Batched`]
    /// Dense's tables in split planes (`BatchedM2l::select`, Dense's), RCR on the others;
    /// under [`HelmholtzM2l::Auto`] Dense's tables in split planes on the levels of degree at
    /// most the crossover ([`max_degree`](Self::max_degree)) that fit the budget
    /// (`BatchedM2l::select_within`), RCR on the others, with the column kernel in
    /// [`auto_m2l_mul_add`]'s mode unless [`with_m2l_mul_add`](Self::with_m2l_mul_add) named
    /// one (before or after; the other settings' default is [`MulAdd::Separate`]);
    /// under [`HelmholtzM2l::Rotation`] ignores the budget. The tables are built in f64,
    /// O(p⁵) per offset and level (16 offsets a level for Classes, 316 for Dense), on the
    /// pool's threads if [`with_pool`](Self::with_pool) came first (module documentation,
    /// "The tables' build").
    pub fn with_m2l(mut self, choice: HelmholtzM2l, dense_budget: usize) -> Self {
        self.kernel.m2l = choice;
        self.kernel.dense_budget = dense_budget;
        let mul_add = self.kernel.mul_add_for(self.kernel.matrix.isa());
        self.kernel.matrix = MatrixKernel::new(self.kernel.matrix.isa(), mul_add)
            .expect("the kernel's own ISA is available");
        self.select_tables();
        self
    }

    /// Names the crossover degree of [`HelmholtzM2l::Auto`] (`None`: the rule's,
    /// [`auto_batched_max_degree`] of the column kernel's ISA and the multipole planes), for
    /// the measurements, the tests and an agreement over ranks; the other settings ignore
    /// it. Under Auto the levels are chosen again if the degree changes.
    pub fn with_m2l_max_degree(mut self, max_degree: Option<usize>) -> Self {
        let before = self.kernel.auto_max_degree();
        self.kernel.max_degree = max_degree;
        if self.kernel.m2l == HelmholtzM2l::Auto && self.kernel.auto_max_degree() != before {
            self.select_tables();
        }
        self
    }

    /// The crossover degree under [`HelmholtzM2l::Auto`]: the named one, else the rule's for
    /// the column kernel's ISA and the multipole planes; `None` under the other settings.
    pub fn max_degree(&self) -> Option<usize> {
        (self.kernel.m2l == HelmholtzM2l::Auto).then(|| self.kernel.auto_max_degree())
    }

    /// Gives [`HelmholtzM2l::Auto`] the levels' V lists over the whole tree, one
    /// [`LevelShape`] per level (Helmholtz CH4.0; module documentation, "M2L"): a level then
    /// runs Batched only where its [`LevelShape::batched_columns`] is at least
    /// [`auto_batched_min_columns`] of the ISA, the planes and its degree. `None` (the
    /// operator's default): no such term, CH3.2d's rule. Under Auto the levels are chosen
    /// again; the other settings ignore it.
    ///
    /// # Panics
    ///
    /// If the shapes are not one per level.
    pub fn with_level_shapes(mut self, shapes: Option<Vec<LevelShape>>) -> Self {
        if let Some(shapes) = &shapes {
            assert_eq!(
                shapes.len(),
                self.kernel.base_degrees.len(),
                "one level shape per level"
            );
        }
        if self.kernel.shapes != shapes {
            self.kernel.shapes = shapes;
            if self.kernel.m2l == HelmholtzM2l::Auto {
                self.select_tables();
            }
        }
        self
    }

    /// The byte budget of the M2L tables ([`with_m2l`](Self::with_m2l)).
    pub fn dense_budget(&self) -> usize {
        self.kernel.dense_budget
    }

    /// The levels' V lists Auto's term reads ([`with_level_shapes`](Self::with_level_shapes)),
    /// if given.
    pub fn level_shapes(&self) -> Option<&[LevelShape]> {
        self.kernel.shapes.as_deref()
    }

    /// Names m*, Auto's fewest columns per table read for Batched, by degree: entry p for a
    /// level of degree p, [`M2L_BLOCK_TARGETS`] beyond the table (`None`, the default: the
    /// rule's, [`auto_batched_min_columns`] of the column kernel's ISA and the multipole
    /// planes), for an agreement over ranks (`HelmholtzFmmBuilder::build` names the largest
    /// of every rank's, so ranks of different ISAs choose alike) and the tests. Under Auto
    /// with level shapes the levels are chosen again; the other settings ignore it.
    pub fn with_m2l_min_columns(mut self, table: Option<Vec<f64>>) -> Self {
        if self.kernel.min_columns != table {
            self.kernel.min_columns = table;
            if self.kernel.m2l == HelmholtzM2l::Auto && self.kernel.shapes.is_some() {
                self.select_tables();
            }
        }
        self
    }

    /// m* of degree p that Auto's term reads: the named table's, else the rule's.
    pub fn m2l_min_columns(&self, p: usize) -> f64 {
        self.kernel.auto_min_columns(p)
    }

    /// Names the width of Batched's blocks in its level call (`None`, the default:
    /// [`batched_block_targets`] of the level's boxes and the pool's threads), for the
    /// measurements and the tests; at least 1, and above [`M2L_BLOCK_TARGETS`] run at that.
    /// The output does not depend on it (module documentation, "M2L").
    ///
    /// # Panics
    ///
    /// If the width is 0.
    pub fn with_m2l_block_targets(mut self, targets: Option<usize>) -> Self {
        assert_ne!(
            targets,
            Some(0),
            "a block of Batched holds at least one target"
        );
        self.kernel.block_targets = targets;
        self
    }

    /// The width of Batched's blocks if named
    /// ([`with_m2l_block_targets`](Self::with_m2l_block_targets)).
    pub fn m2l_block_targets(&self) -> Option<usize> {
        self.kernel.block_targets
    }

    /// Lets [`HelmholtzM2l::Auto`] take the exponential M2L (Helmholtz CH6.2h, D8; module
    /// documentation, "M2L"; on by default since D8's sign-off, 2026-11-08): on each level
    /// of degree at least
    /// [`plane_wave_min_degree`](Self::plane_wave_min_degree) to which the precision rule
    /// (`nd_fmm_tables::helmholtz::plane_wave::merge::select`, with the tolerance of
    /// [`with_plane_wave_tolerance`](Self::with_plane_wave_tolerance)) gives a quadrature and
    /// whose Im κ_l is below [`AUTO_PLANE_WAVE_MAX_IM_KAPPA`], in f64 only, the level at
    /// the rule's raised degree below 10⁻⁶ (Helmholtz CH6.1g, CH6.1h); Batched and RCR on the other levels as before. Before or after [`with_m2l`](Self::with_m2l); under
    /// Auto the levels are chosen again; the other settings ignore it.
    pub fn with_auto_plane_wave(mut self, on: bool) -> Self {
        if self.kernel.auto_plane_wave != on {
            self.kernel.auto_plane_wave = on;
            if self.kernel.m2l == HelmholtzM2l::Auto {
                self.select_tables();
            }
        }
        self
    }

    /// Names the smallest degree at which [`HelmholtzM2l::Auto`] takes the exponential M2L
    /// (`None`: the rule's, [`auto_plane_wave_min_degree`] of the column kernel's ISA and the
    /// multipole planes), for the measurements, the tests and an agreement over ranks; read
    /// only with [`with_auto_plane_wave`](Self::with_auto_plane_wave). Under Auto with the
    /// exponential M2L the levels are chosen again if the degree changes.
    pub fn with_m2l_plane_wave_min_degree(mut self, min_degree: Option<usize>) -> Self {
        let before = self.kernel.auto_plane_wave_min_degree();
        self.kernel.plane_wave_min_degree = min_degree;
        if self.kernel.auto_takes_plane_waves()
            && self.kernel.auto_plane_wave_min_degree() != before
        {
            self.select_tables();
        }
        self
    }

    /// The smallest degree at which [`HelmholtzM2l::Auto`] takes the exponential M2L: the
    /// named one, else the rule's; `None` unless Auto with
    /// [`with_auto_plane_wave`](Self::with_auto_plane_wave).
    pub fn plane_wave_min_degree(&self) -> Option<usize> {
        self.kernel
            .auto_takes_plane_waves()
            .then(|| self.kernel.auto_plane_wave_min_degree())
    }

    /// Takes each level's quadrature under [`HelmholtzM2l::PlaneWave`] by `precision`
    /// ([`PlaneWavePrecision::Rule`] by default), before or after
    /// [`with_m2l`](Self::with_m2l); the other settings ignore it. Under PlaneWave the levels'
    /// tables are built again.
    pub fn with_plane_wave_precision(mut self, precision: PlaneWavePrecision) -> Self {
        self.kernel.plane_wave_precision = precision;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave {
            self.select_tables();
        }
        self
    }

    /// Gives [`PlaneWavePrecision::Rule`] the FMM's tolerance ε (`Degrees::Tolerance`'s;
    /// `None`, the default, for named degrees, when the rule reads each level's degree),
    /// before or after [`with_m2l`](Self::with_m2l); [`HelmholtzDiagonal::Rule`] reads it too
    /// (Helmholtz CH4.2b1). Under PlaneWave, Auto with waves and the diagonal rule the levels'
    /// tables are built again.
    pub fn with_plane_wave_tolerance(mut self, tolerance: Option<f64>) -> Self {
        self.kernel.plane_wave_tolerance = tolerance;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave
            || self.kernel.auto_takes_plane_waves()
            || self.kernel.diagonal_settings.choice == HelmholtzDiagonal::Rule
        {
            self.select_tables();
        }
        self
    }

    /// Takes the α counts of every level's quadrature under [`HelmholtzM2l::PlaneWave`] for
    /// the lateral offsets ρ ≤ `reach` (box widths; `Quadrature::with_reach`) in place of the
    /// region's corner 4√2 (`RHO_MAX`, the default), before or after
    /// [`with_m2l`](Self::with_m2l). The counts then bound the kernel on ρ ≤ `reach` only;
    /// below the corner they are fewer where Im κ < π/2 (on some strongly damped levels
    /// more; fmm-validate/results/helmholtz-ch6-2e.md §1). For the measurements (Helmholtz
    /// CH6.2e). Under PlaneWave the levels' tables are built again.
    ///
    /// # Panics
    ///
    /// If `reach` is not finite or not above the source's reach √2/2.
    pub fn with_plane_wave_reach(mut self, reach: f64) -> Self {
        assert_reach(reach);
        self.kernel.plane_wave_reach = reach;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave || self.kernel.auto_takes_plane_waves() {
            self.select_tables();
        }
        self
    }

    /// Runs the translations of each raised level by `raise` (default
    /// [`PlaneWaveRaise::Rule`]; Helmholtz CH6.1h), before or after
    /// [`with_m2l`](Self::with_m2l). Where the rule raises a level (PlaneWave's rule below
    /// ε = 10⁻⁶, Auto at 10⁻⁸ ≤ ε < 10⁻⁷) the tables are built again.
    pub fn with_plane_wave_raise(mut self, raise: PlaneWaveRaise) -> Self {
        self.kernel.plane_wave_raise = raise;
        if self.kernel.m2l == HelmholtzM2l::PlaneWave || self.kernel.auto_takes_plane_waves() {
            self.select_tables();
        }
        self
    }

    /// Which operators of a raised level run at p_w
    /// ([`with_plane_wave_raise`](Self::with_plane_wave_raise)).
    pub fn plane_wave_raise(&self) -> PlaneWaveRaise {
        self.kernel.plane_wave_raise
    }

    /// The lateral reach of the quadratures' α counts
    /// ([`with_plane_wave_reach`](Self::with_plane_wave_reach)).
    pub fn plane_wave_reach(&self) -> f64 {
        self.kernel.plane_wave_reach
    }

    /// The waves per direction of `level`, Σₖ Nₖ of its quadrature, under
    /// [`HelmholtzM2l::PlaneWave`] (or [`HelmholtzM2l::Auto`] with waves); 0 where the level
    /// runs another M2L.
    pub fn plane_wave_waves(&self, level: usize) -> usize {
        self.kernel
            .plane_wave_level(level)
            .map_or(0, |l| l.tables().wave_len() / 2)
    }

    /// The precision of `level`'s quadrature under [`HelmholtzM2l::PlaneWave`] (or
    /// [`HelmholtzM2l::Auto`] with waves), `None` where the level runs another M2L.
    pub fn plane_wave_precision(&self, level: usize) -> Option<Precision> {
        self.kernel.plane_wave_level(level).map(|l| l.precision())
    }

    /// Sizes the buffers of the exponential M2L's level call for levels of `boxes[l]` boxes
    /// (each level's own and ghost boxes), so that no level call grows them; nothing unless
    /// a level has waves ([`HelmholtzM2l::PlaneWave`], or [`HelmholtzM2l::Auto`] with
    /// waves). `HelmholtzFmmBuilder` sizes them from the plan, after the bound
    /// ([`with_plane_wave_buffer`](Self::with_plane_wave_buffer)): sized before it, the
    /// buffer holds every box of the largest level until the bound is set.
    pub fn with_plane_wave_boxes(mut self, boxes: &[usize]) -> Self {
        self.plane_buffer.level_boxes = boxes.to_vec();
        self.plane_buffer.size(self.kernel.plane_waves.as_ref());
        self
    }

    /// Bounds the waves buffer of the exponential M2L's level call at `bytes` (`None`: every
    /// box of a level, the default). A level call then runs by chunks of whole parents whose
    /// sources' waves fit the bound, at least [`MIN_BUFFER_BOXES`] boxes of each level
    /// ("M2L" in the module documentation); the output is the same at every bound.
    /// `HelmholtzFmmBuilder` sets it from its node budget
    /// ([`plane_wave_budget`](super::HelmholtzFmmBuilder::plane_wave_budget)).
    pub fn with_plane_wave_buffer(mut self, bytes: Option<usize>) -> Self {
        self.plane_buffer.bound = bytes;
        self.plane_buffer.size(self.kernel.plane_waves.as_ref());
        self
    }

    /// Splits the exponential M2L's merge and shift over the pool by `split` (default
    /// [`PlaneWaveSplit::Auto`]; Helmholtz CH6.2f). The output is the same bit for bit under
    /// every setting.
    pub fn with_plane_wave_split(mut self, split: PlaneWaveSplit) -> Self {
        self.plane_buffer.split = split;
        self
    }

    /// How merge and shift splits over the pool
    /// ([`with_plane_wave_split`](Self::with_plane_wave_split)).
    pub fn plane_wave_split(&self) -> PlaneWaveSplit {
        self.plane_buffer.split
    }

    /// The waves buffer's byte bound ([`with_plane_wave_buffer`](Self::with_plane_wave_buffer)).
    pub fn plane_wave_buffer(&self) -> Option<usize> {
        self.plane_buffer.bound
    }

    /// The bytes of the waves buffer as sized (part of
    /// [`plane_wave_bytes`](Self::plane_wave_bytes)).
    pub fn plane_wave_buffer_bytes(&self) -> usize {
        self.plane_buffer.wave_bytes()
    }

    /// The most bytes the waves buffer held since the operator was built: at least
    /// [`plane_wave_buffer_bytes`](Self::plane_wave_buffer_bytes), more if it was sized for
    /// the plan's boxes before its bound was set (Helmholtz CH4.0c).
    pub fn plane_wave_buffer_peak_bytes(&self) -> usize {
        self.plane_buffer.peak
    }

    /// The counts of the exponential M2L's level calls since its tables were last chosen:
    /// level calls, chunks and the waves computed ([`PlaneWaveCounts`]).
    pub fn plane_wave_counts(&self) -> PlaneWaveCounts {
        self.plane_buffer.counts
    }

    /// The wall time of the M2L level calls of each level since the tables were last chosen
    /// (Helmholtz CH6.2b): one entry per level, the time of every [`FmmOperator::m2l`] call
    /// on it summed, whichever M2L ran there ([`m2l_on`](Self::m2l_on)); zero on the levels
    /// without a call. This rank's, on the calling thread's clock; the pair body is not timed.
    pub fn m2l_times(&self) -> &[Duration] {
        &self.m2l_times
    }

    /// The wall time of the P2P level calls of each level since the tables were last chosen
    /// (Helmholtz CH4.0d2, for the V pairs' near rule's cost): as
    /// [`m2l_times`](Self::m2l_times), one entry per level of the target leaves.
    pub fn p2p_times(&self) -> &[Duration] {
        &self.p2p_times
    }

    /// The wall time of the level calls of the other kinds of each level since the tables
    /// were last chosen (Helmholtz CH4.0d3, for the leaf size's cost): as
    /// [`m2l_times`](Self::m2l_times), one entry per level (the level of each batch), the
    /// kinds in the order of [`TIMED_KINDS`].
    pub fn kind_times(&self) -> &[[Duration; 6]] {
        &self.kind_times
    }

    /// The wall time of the exponential M2L's two passes on each level since the tables were
    /// last chosen (Helmholtz CH6.2d): one entry per level, zero where no level call ran merge
    /// and shift; within the level's [`m2l_times`](Self::m2l_times) entry. This rank's, on
    /// the calling thread's clock.
    pub fn plane_wave_times(&self) -> &[PlaneWaveTimes] {
        &self.plane_wave_times
    }

    /// The bytes the exponential M2L holds: its tables, the level call's buffers (the waves,
    /// a position per box and the chunk's sources) and every scratch set's buffers (0 unless
    /// a level has waves, under [`HelmholtzM2l::PlaneWave`] or [`HelmholtzM2l::Auto`]).
    pub fn plane_wave_bytes(&self) -> usize {
        let tables = self
            .kernel
            .plane_waves
            .as_ref()
            .map_or(0, PlaneWaveM2l::bytes);
        let scratch: usize = self
            .execution
            .scratch
            .iter()
            .filter_map(|s| s.lock().ok()?.plane.as_ref().map(PlaneScratch::bytes))
            .sum();
        tables + self.plane_buffer.bytes() + scratch
    }

    /// Builds the M2L tables of the setting, the budget and (under Auto) the crossover, and
    /// the scratch sized for them; under [`HelmholtzM2l::PlaneWave`]'s rule with a tolerance
    /// below 10⁻⁶, and under [`HelmholtzM2l::Auto`] at 10⁻⁸ ≤ ε < 10⁻⁷, the RCR tables at
    /// the raised degrees (Helmholtz CH6.1g), or again at the degrees given where an earlier
    /// setting raised them.
    fn select_tables(&mut self) {
        if self.unbuilt {
            return;
        }
        let parts = PoolParts(self.execution.pool.as_deref());
        let kernel = &mut self.kernel;
        let timings = &mut self.table_timings;
        let choice = kernel.m2l;
        let dense_budget = kernel.dense_budget;
        let base = kernel.base_degrees.clone();
        let k_root = kernel.tables.k_hat(0);
        let tolerance = kernel.plane_wave_tolerance;
        // The levels with waves, which the builder's V rule reads too (Helmholtz CHA.11). Under
        // Auto they are the rule's quadratures from the smallest degree on, below the bound on
        // Im κ_l (since CH6.1h's sign-off, 2026-11-11, at every tolerance with a quadrature),
        // less those the diagonal form's time rule takes (Helmholtz CH4.3b1).
        let costs = kernel.diagonal_costs();
        let mut precisions = wave_levels::<T>(
            &WaveSettings {
                m2l: choice,
                precision: &kernel.plane_wave_precision,
                auto_plane_wave: kernel.auto_plane_wave,
                min_degree: kernel.auto_plane_wave_min_degree(),
                tolerance,
            },
            k_root,
            &base,
        );
        // Under the time rule (Helmholtz CH4.3b1, CH4.3b2a) a level with waves in its range
        // runs the cheapest of the waves, the form and RCR at its resolving degree p_w, which
        // it keeps whatever M2L runs (matched accuracy): the form's classes are the bound's at
        // p_w, and its scratch budget's whole-level fallback comes first (constraint 7).
        let mut from_waves = vec![false; base.len()];
        if let (HelmholtzDiagonal::Auto, HelmholtzM2l::Auto, Some(epsilon), Some(levels)) = (
            kernel.diagonal_settings.choice,
            choice,
            tolerance,
            precisions.as_mut(),
        ) {
            for (l, precision) in levels.iter_mut().enumerate() {
                if precision.is_none() {
                    continue;
                }
                let k_hat = kernel.tables.k_hat(l);
                if !diagonal_auto_in_range(k_hat) {
                    continue;
                }
                let p_w = plane_wave_degree(base[l], epsilon);
                let classes =
                    if kernel
                        .diagonal_settings
                        .fits::<T>(l, p_w, kernel.densities.vectors())
                    {
                        diagonal_rule_classes::<T>(
                            p_w,
                            k_hat,
                            epsilon,
                            kernel.diagonal_settings.build,
                        )
                    } else {
                        Vec::new()
                    };
                let shape = kernel.diagonal_settings.shape(l);
                if wave_level_choice(&costs, p_w, &shape, &classes) != WaveLevelChoice::PlaneWave {
                    *precision = None;
                    from_waves[l] = true;
                }
            }
        }
        // The rule's resolving degree on the levels with waves (Helmholtz CH6.1g), under
        // PlaneWave's rule and under Auto, and on those the time rule took from them: the
        // level's stores, its M2L and its leaves at p_w = max(p_l, d), and its translations to
        // and from its parent too unless `PlaneWaveRaise::M2l` keeps them at p_l (Helmholtz
        // CH6.1h).
        let by_rule = choice == HelmholtzM2l::Auto
            || (choice == HelmholtzM2l::PlaneWave
                && kernel.plane_wave_precision == PlaneWavePrecision::Rule);
        let degrees: Vec<usize> = match (tolerance, &precisions) {
            (Some(epsilon), Some(precisions)) if by_rule => base
                .iter()
                .zip(precisions)
                .zip(&from_waves)
                .map(|((&p, precision), &taken)| {
                    if precision.is_some() || taken {
                        plane_wave_degree(p, epsilon)
                    } else {
                        p
                    }
                })
                .collect(),
            _ => base.clone(),
        };
        let translation = match kernel.plane_wave_raise.resolved(tolerance) {
            PlaneWaveRaise::Rule | PlaneWaveRaise::Level => &degrees,
            PlaneWaveRaise::Leaves | PlaneWaveRaise::M2l => &base,
        };
        // Each level's M2L first (Helmholtz CH4.0h), so that the RCR tables store their
        // M2L blocks for the levels whose M2L reads them alone.
        let clock = Instant::now();
        kernel.plane_waves = precisions.as_ref().map(|precisions| {
            PlaneWaveM2l::build_with_reach_on(
                k_root,
                &degrees,
                precisions,
                kernel.plane_wave_reach,
                &parts,
            )
        });
        // No level with waves: nothing to hold.
        kernel.plane_waves = kernel
            .plane_waves
            .take()
            .filter(|m2l| (0..m2l.nlevels()).any(|l| m2l.level(l).is_some()));
        timings.plane_waves += clock.elapsed();
        let clock = Instant::now();
        kernel.dense = (choice == HelmholtzM2l::Dense).then(|| {
            let levels = DenseM2l::<T>::select(&degrees, dense_budget);
            DenseM2l::build_on(k_root, &degrees, &levels, &parts)
        });
        kernel.classes = (choice == HelmholtzM2l::Classes).then(|| {
            let levels = ClassM2l::<T>::select(&degrees, dense_budget);
            ClassM2l::build_on(k_root, &degrees, &levels, &parts)
        });
        let max_degree = match choice {
            HelmholtzM2l::Batched => Some(usize::MAX),
            HelmholtzM2l::Auto => Some(kernel.auto_max_degree()),
            _ => None,
        };
        kernel.batched = max_degree.map(|max_degree| {
            let levels = if choice == HelmholtzM2l::Auto {
                // `select_within`'s choice with the levels that have waves, and those whose
                // tables would be read for too few columns (Helmholtz CH4.0), left out: they
                // take no batched tables and spend none of the budget.
                let mut chosen = vec![false; degrees.len()];
                let mut left = dense_budget;
                for (l, &p) in degrees.iter().enumerate().skip(2).rev() {
                    // A level the diagonal form took from the waves stays the form's.
                    let waves =
                        precisions.as_ref().is_some_and(|w| w[l].is_some()) || from_waves[l];
                    let sparse = kernel.shapes.as_ref().is_some_and(|shapes| {
                        shapes[l].batched_columns() < kernel.auto_min_columns(p)
                    });
                    if waves || sparse || p > max_degree {
                        continue;
                    }
                    let bytes = BatchedM2l::<T>::level_bytes(p);
                    if bytes <= left {
                        left -= bytes;
                        chosen[l] = true;
                    }
                }
                chosen
            } else {
                BatchedM2l::<T>::select_within(&degrees, dense_budget, max_degree)
            };
            BatchedM2l::build_on(k_root, &degrees, &levels, &parts)
        });
        timings.matrices += clock.elapsed();
        kernel.from_waves = from_waves;
        // The levels whose M2L is RCR's: only they read its M2L blocks.
        let rcr: Vec<bool> = (0..degrees.len())
            .map(|l| {
                l >= 2
                    && kernel.plane_wave_level(l).is_none()
                    && !kernel.dense.as_ref().is_some_and(|d| d.is_dense(l))
                    && !kernel.classes.as_ref().is_some_and(|c| c.is_class(l))
                    && kernel.batched(l).is_none()
            })
            .collect();
        let clock = Instant::now();
        let tables = &mut kernel.tables;
        if !tables.has_translations()
            || (0..degrees.len()).any(|l| {
                tables.m2l_degree(l) != degrees[l]
                    || tables.degree(l) != translation[l]
                    || (l >= 2 && !tables.has_m2l(l))
            })
        {
            let choice = TableChoice {
                m2l_degrees: Some(&degrees),
                block_levels: Some(&rcr),
                ..TableChoice::new(kernel.block_budget)
            };
            *tables = HelmholtzTables::build_by(k_root, translation, &choice, &parts);
        } else {
            tables.set_m2l_blocks(&rcr, kernel.block_budget, &parts);
        }
        timings.rcr += clock.elapsed();
        // The diagonal form on the RCR levels where the rule takes some class (Helmholtz
        // CH4.2b1), after the RCR tables, whose wavenumbers it reads.
        let clock = Instant::now();
        kernel.diagonal = diagonal::select::<T>(
            &kernel.diagonal_settings,
            tolerance,
            &rcr,
            &costs,
            &degrees,
            |l| kernel.tables.k_hat(l),
            kernel.densities.vectors(),
            &parts,
        );
        timings.diagonal += clock.elapsed();
        self.diagonal_buffer.size(
            &kernel.diagonal,
            &kernel.diagonal_settings,
            kernel.densities.vectors(),
        );
        self.plane_buffer.size(kernel.plane_waves.as_ref());
        self.plane_buffer.counts = PlaneWaveCounts::default();
        self.rcr_buffer.counts = RcrCounts::default();
        self.m2l_times.fill(Duration::ZERO);
        self.p2p_times.fill(Duration::ZERO);
        self.kind_times.fill([Duration::ZERO; 6]);
        self.plane_wave_times.fill(PlaneWaveTimes::default());
        // The class scratch, the batched columns and the exponential M2L's buffers are sized
        // by their tables' largest level.
        // One per thread of the pool, made on its threads (Helmholtz CH4.0h; no value depends
        // on which), the old sets dropped first: at p = 135 a set holds about 27 MB.
        let clock = Instant::now();
        let kernel = &*kernel;
        let n = self
            .execution
            .pool
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads().max(1));
        self.execution.scratch.clear();
        self.execution.scratch = parts.map(n, &|_| Mutex::new(Scratch::new(kernel)));
        timings.scratch += clock.elapsed();
        self.size_rcr_buffer();
    }

    /// Sizes the Rotation M2L's buffer for the plan's levels that run by pairs on the
    /// pool's threads ([`with_rcr_levels`](Self::with_rcr_levels)); nothing while the
    /// tables are unbuilt.
    fn size_rcr_buffer(&mut self) {
        if self.unbuilt {
            return;
        }
        let threads = match (&self.execution.pool, self.execution.serial) {
            (Some(_), false) => self.execution.scratch.len(),
            _ => 1,
        };
        let m = self.kernel.densities.vectors();
        let rotation: Vec<bool> = (0..self.kernel.tables.nlevels())
            .map(|l| {
                self.m2l_on(l) == HelmholtzM2l::Rotation && self.kernel.diagonal_level(l).is_none()
            })
            .collect();
        let tables = &self.kernel.tables;
        self.rcr_buffer.size(
            threads,
            |l| rotation.get(l).copied().unwrap_or(false),
            |l| tables.m2l_degree(l),
            |l| tables.m2l_contribution_len(l, m),
        );
    }

    /// Splits the Rotation (RCR) M2L's level calls over the pool by `split` (default
    /// [`RcrSplit::Auto`]; Helmholtz CH4.0i; module documentation, "Execution"). The output
    /// is the same bit for bit under every setting.
    pub fn with_rcr_split(mut self, split: RcrSplit) -> Self {
        self.rcr_buffer.split = split;
        self.size_rcr_buffer();
        self
    }

    /// How the Rotation M2L's level calls split over the pool
    /// ([`with_rcr_split`](Self::with_rcr_split)).
    pub fn rcr_split(&self) -> RcrSplit {
        self.rcr_buffer.split
    }

    /// Bounds the buffer of the Rotation M2L's contributions by pairs at `bytes` (default
    /// [`DEFAULT_RCR_BUFFER`]; `None`: every pair of a level): a level call by pairs then
    /// runs by chunks of pairs whose contributions fit, at least one pair per thread. The
    /// output is the same at every bound. `HelmholtzFmmBuilder` sets it from its node budget
    /// ([`rcr_budget`](super::HelmholtzFmmBuilder::rcr_budget)) and sizes it from the plan
    /// ([`with_rcr_levels`](Self::with_rcr_levels)); without the plan's levels the first level
    /// call by pairs of a level grows it, up to the bound.
    pub fn with_rcr_buffer(mut self, bytes: Option<usize>) -> Self {
        self.rcr_buffer.bound = bytes;
        self.size_rcr_buffer();
        self
    }

    /// The byte bound of the Rotation M2L's buffer ([`with_rcr_buffer`](Self::with_rcr_buffer)).
    pub fn rcr_buffer(&self) -> Option<usize> {
        self.rcr_buffer.bound
    }

    /// Sizes the Rotation M2L's buffer for levels of `levels[l]` = (V pairs, targets with a
    /// V row) on this rank, so that no level call by pairs grows it. `HelmholtzFmmBuilder`
    /// counts them in the plan.
    pub fn with_rcr_levels(mut self, levels: &[(usize, usize)]) -> Self {
        self.rcr_buffer.levels = levels.to_vec();
        self.size_rcr_buffer();
        self
    }

    /// The bytes of the Rotation M2L's buffer as sized.
    pub fn rcr_buffer_bytes(&self) -> usize {
        self.rcr_buffer.bytes()
    }

    /// The most bytes the Rotation M2L's buffer held since the operator was built.
    pub fn rcr_buffer_peak_bytes(&self) -> usize {
        self.rcr_buffer.peak
    }

    /// The counts of the Rotation M2L's level calls by pairs since its M2L was last chosen
    /// ([`RcrCounts`]).
    pub fn rcr_counts(&self) -> &RcrCounts {
        &self.rcr_buffer.counts
    }

    /// Runs the M2L of the levels whose M2L is RCR's by the diagonal form where its rule takes
    /// an offset class (Helmholtz CH4.2b1; module documentation, "M2L"; default
    /// [`HelmholtzDiagonal::Off`]): each class with E(l, c) ≤ θ ε
    /// ([`diagonal_rule_classes`]) on the diagonal form, RCR on the others. The tables are
    /// chosen again.
    ///
    /// # Panics
    ///
    /// If [`HelmholtzDiagonal::Tolerance`] names ε ≤ 0 or NaN.
    pub fn with_diagonal(mut self, choice: HelmholtzDiagonal) -> Self {
        if let HelmholtzDiagonal::Tolerance(epsilon) = choice {
            assert!(
                epsilon > 0.0,
                "the diagonal form's tolerance {epsilon} is not positive"
            );
        }
        self.kernel.diagonal_settings.choice = choice;
        self.select_tables();
        self
    }

    /// The diagonal M2L's setting ([`with_diagonal`](Self::with_diagonal)).
    pub fn diagonal(&self) -> HelmholtzDiagonal {
        self.kernel.diagonal_settings.choice
    }

    /// Builds the diagonal form's transfer functions by `build` (Helmholtz CH4.2c2; default
    /// [`DEFAULT_DIAGONAL_BUILD`], interpolated): the rule then reads that build's constants
    /// ([`diagonal_rule_classes`]; C_T = 0.15 direct, 0.23 interpolated, in f64), so the
    /// interpolated build may take fewer classes at a tolerance. The tables are chosen again.
    pub fn with_diagonal_build(mut self, build: TransferBuild) -> Self {
        self.kernel.diagonal_settings.build = build;
        self.select_tables();
        self
    }

    /// The diagonal form's build of the transfer functions
    /// ([`with_diagonal_build`](Self::with_diagonal_build)).
    pub fn diagonal_build(&self) -> TransferBuild {
        self.kernel.diagonal_settings.build
    }

    /// Bounds the scratch of a diagonal level call on this rank at `bytes` (`None`, the
    /// operator's default: no bound): a level takes the diagonal form only where
    /// `nd_fmm_tables::helmholtz::diagonal::scratch_shape` of its degree, its sources'
    /// signatures (the counts of [`with_diagonal_sources`](Self::with_diagonal_sources) times
    /// the vectors) and a chunk of accumulators fits, else RCR (the whole-level fallback;
    /// without the counts no level falls back). `HelmholtzFmmBuilder` passes
    /// `DEFAULT_SIGNATURE_NODE_BUDGET` over the ranks per node. The tables are chosen again.
    pub fn with_diagonal_budget(mut self, bytes: Option<usize>) -> Self {
        self.kernel.diagonal_settings.budget = bytes;
        self.select_tables();
        self
    }

    /// The diagonal level call's scratch bound
    /// ([`with_diagonal_budget`](Self::with_diagonal_budget)).
    pub fn diagonal_budget(&self) -> Option<usize> {
        self.kernel.diagonal_settings.budget
    }

    /// Gives the diagonal form each level's distinct V sources (own and ghost boxes), the
    /// largest count over the ranks, which the budget's fallback reads and the buffer is sized
    /// for, so that every rank takes the same levels (proposal §2, constraints 2 and 9; a level
    /// of count 0 has no V pairs on any rank and takes no tables). `HelmholtzFmmBuilder`
    /// counts them in its plan and agrees them (all-reduce max). The tables are chosen again.
    pub fn with_diagonal_sources(mut self, sources: Option<Vec<usize>>) -> Self {
        self.kernel.diagonal_settings.sources = sources;
        self.select_tables();
        self
    }

    /// Names the width of the diagonal level call's chunks of targets (`None`, the default:
    /// [`diagonal_chunk_targets`] of the level's boxes and the pool's threads), for the tests;
    /// no bit depends on it. The scratch is sized again.
    ///
    /// # Panics
    ///
    /// If the width is 0.
    pub fn with_diagonal_chunk(mut self, chunk: Option<usize>) -> Self {
        assert_ne!(chunk, Some(0), "a chunk holds at least one target");
        self.kernel.diagonal_settings.chunk = chunk;
        self.select_tables();
        self
    }

    /// Names the time rule's costs ([`HelmholtzDiagonal::Auto`]; `None`, the default: those
    /// of the column kernel's ISA, [`diagonal_costs`]). `HelmholtzFmmBuilder` agrees them over
    /// the ranks (all-reduce max), so that ranks of different ISAs choose the same levels
    /// (constraints 2 and 9; Helmholtz CH4.3b1). The tables are chosen again.
    pub fn with_diagonal_costs(mut self, costs: Option<DiagonalCosts>) -> Self {
        self.kernel.diagonal_settings.costs = costs;
        self.select_tables();
        self
    }

    /// The time rule's costs as the operator reads them
    /// ([`with_diagonal_costs`](Self::with_diagonal_costs)).
    pub fn diagonal_costs(&self) -> DiagonalCosts {
        self.kernel.diagonal_costs()
    }

    /// Gives the time rule each level's V lists ([`DiagonalShape`]: the boxes with a V row
    /// and the pairs per class, summed over the ranks; `None`, the default: the uniform
    /// level's, [`DiagonalShape::uniform`]). `HelmholtzFmmBuilder` counts them in its plan and
    /// agrees them (all-reduce sum; Helmholtz CH4.3b1). The tables are chosen again.
    pub fn with_diagonal_shapes(mut self, shapes: Option<Vec<DiagonalShape>>) -> Self {
        self.kernel.diagonal_settings.shapes = shapes;
        self.select_tables();
        self
    }

    /// The levels' V lists the time rule reads
    /// ([`with_diagonal_shapes`](Self::with_diagonal_shapes)), if given.
    pub fn diagonal_shapes(&self) -> Option<&[DiagonalShape]> {
        self.kernel.diagonal_settings.shapes.as_deref()
    }

    /// Whether the diagonal form's time rule ([`HelmholtzDiagonal::Auto`]) took `level` from
    /// the exponential M2L that [`HelmholtzM2l::Auto`] would run there (Helmholtz CH4.3b2a): the
    /// level then runs the form or RCR ([`diagonal_classes`](Self::diagonal_classes) says which)
    /// at its resolving degree.
    pub fn taken_from_waves(&self, level: usize) -> bool {
        self.kernel.from_waves.get(level).copied().unwrap_or(false)
    }

    /// The offset classes (indices into `nd_fmm_ref::helmholtz::diagonal::classes()`) that
    /// take the diagonal form on `level`, or `None` where the level has none.
    pub fn diagonal_classes(&self, level: usize) -> Option<Vec<usize>> {
        self.kernel
            .diagonal_level(level)
            .map(|d| d.m2l.stored_classes())
    }

    /// The bytes of the diagonal form: the transfer functions and conversions of its levels,
    /// the operator's buffer of source signatures as sized, and every thread's scratch.
    pub fn diagonal_bytes(&self) -> usize {
        let tables: usize = self
            .kernel
            .diagonal
            .iter()
            .flatten()
            .map(|d| d.m2l.bytes() + d.conversions.bytes())
            .sum();
        let scratch: usize = self
            .execution
            .scratch
            .iter()
            .map(|s| {
                s.lock()
                    .expect("no level call holds a scratch set")
                    .diagonal
                    .as_ref()
                    .map_or(0, DiagonalScratch::bytes)
            })
            .sum();
        tables + self.diagonal_buffer.bytes() + scratch
    }

    /// The most bytes the diagonal form's buffer of source signatures held since the operator
    /// was built.
    pub fn diagonal_buffer_peak_bytes(&self) -> usize {
        self.diagonal_buffer.peak
    }

    /// The counts of the diagonal level calls since the operator's M2L was last chosen
    /// ([`DiagonalCounts`]).
    pub fn diagonal_counts(&self) -> &DiagonalCounts {
        &self.diagonal_buffer.counts
    }

    /// The diagonal level calls' times per level since the operator's M2L was last chosen
    /// ([`DiagonalTimes`]; Helmholtz CH4.2b2): one entry per level, zero where no diagonal
    /// level call ran; the passes within the level's [`m2l_times`](Self::m2l_times) entry.
    /// This rank's.
    pub fn diagonal_times(&self) -> &[DiagonalTimes] {
        &self.diagonal_buffer.times
    }

    /// The wall time of building `level`'s transfer functions (one part per (class, ring) on the
    /// pool) and its conversions when the tables were last chosen (Helmholtz CH4.2b2), or
    /// `None` where the level has no diagonal form. This rank's.
    pub fn diagonal_build_time(&self, level: usize) -> Option<(Duration, Duration)> {
        self.kernel.diagonal_level(level).map(|d| d.build)
    }

    /// The real multiply-adds of one conversion (S → F or F → R) of a box on `level`, both
    /// steps (`nd_fmm_tables::helmholtz::diagonal::Conversions::multiply_adds`; Helmholtz
    /// CH4.2b2), or `None` where the level has no diagonal form.
    pub fn diagonal_multiply_adds(&self, level: usize) -> Option<u64> {
        self.kernel.diagonal_level(level).map(|d| {
            let (legendre, phi) = d.conversions.multiply_adds();
            legendre + phi
        })
    }

    /// The wall time of building the tables by part since the operator was constructed
    /// (Helmholtz CH4.0h): every build of the RCR tables, of the matrices and of the
    /// exponential M2L's tables, and the scratch sets, summed; this rank's.
    pub fn table_timings(&self) -> TableTimings {
        self.table_timings
    }

    /// The M2L setting ([`with_m2l`](Self::with_m2l)).
    pub fn m2l_choice(&self) -> HelmholtzM2l {
        self.kernel.m2l
    }

    /// The M2L that runs on `level`: [`HelmholtzM2l::Dense`], [`HelmholtzM2l::Classes`] or
    /// [`HelmholtzM2l::Batched`] where the level has dense, class or batched tables (under
    /// [`HelmholtzM2l::Auto`] the levels the rule gives Batched), [`HelmholtzM2l::PlaneWave`]
    /// where it has a quadrature (under [`HelmholtzM2l::Auto`] with
    /// [`with_auto_plane_wave`](Self::with_auto_plane_wave) the levels its rule gives
    /// waves), [`HelmholtzM2l::Rotation`] everywhere else (levels 0 and 1 have no M2L).
    pub fn m2l_on(&self, level: usize) -> HelmholtzM2l {
        let kernel = &self.kernel;
        if kernel.plane_wave_level(level).is_some() {
            return HelmholtzM2l::PlaneWave;
        }
        match (&kernel.dense, &kernel.classes, &kernel.batched) {
            (Some(dense), _, _) if dense.is_dense(level) => HelmholtzM2l::Dense,
            (_, Some(classes), _) if classes.is_class(level) => HelmholtzM2l::Classes,
            (_, _, Some(batched)) if batched.is_batched(level) => HelmholtzM2l::Batched,
            _ => HelmholtzM2l::Rotation,
        }
    }

    /// The bytes of the dense M2L tables (0 unless [`HelmholtzM2l::Dense`]).
    pub fn dense_bytes(&self) -> usize {
        self.kernel.dense.as_ref().map_or(0, DenseM2l::bytes)
    }

    /// The bytes of the class M2L tables, the transforms included (0 unless
    /// [`HelmholtzM2l::Classes`]).
    pub fn class_bytes(&self) -> usize {
        self.kernel.classes.as_ref().map_or(0, ClassM2l::bytes)
    }

    /// The bytes of the batched M2L tables (0 unless [`HelmholtzM2l::Batched`] or
    /// [`HelmholtzM2l::Auto`]).
    pub fn batched_bytes(&self) -> usize {
        self.kernel.batched.as_ref().map_or(0, BatchedM2l::bytes)
    }

    /// Runs P2P with the kernel of `choice` (module documentation, "P2P kernel"), and the
    /// batched M2L's column kernel on its ISA (the scalar path for
    /// [`P2pChoice::Reference`]). Under [`HelmholtzM2l::Auto`] without a named crossover
    /// the levels are chosen again if the ISA's crossover differs.
    ///
    /// # Errors
    ///
    /// [`IsaUnavailable`] for `P2pChoice::Isa(isa)` if this machine cannot run `isa`.
    pub fn with_p2p(mut self, choice: P2pChoice) -> Result<Self, IsaUnavailable> {
        let before = self.kernel.auto_max_degree();
        let before_plane_wave = self.kernel.auto_plane_wave_min_degree();
        self.kernel.p2p = match choice {
            P2pChoice::Auto => Some(P2pKernel::detect()),
            P2pChoice::Reference => None,
            P2pChoice::Isa(isa) => Some(P2pKernel::new(isa)?),
        };
        let isa = self.kernel.p2p.as_ref().map_or(Isa::Scalar, P2pKernel::isa);
        self.kernel.matrix = MatrixKernel::new(isa, self.kernel.mul_add_for(isa))?;
        if self.kernel.m2l == HelmholtzM2l::Auto
            && (self.kernel.auto_max_degree() != before
                || (self.kernel.auto_plane_wave
                    && self.kernel.auto_plane_wave_min_degree() != before_plane_wave))
        {
            self.select_tables();
        }
        Ok(self)
    }

    /// Runs the batched M2L's column kernel in the mode `mul_add` (module documentation,
    /// "M2L"; [`MulAdd::Separate`] by default): [`MulAdd::Fused`] forms each step as one
    /// fma, within n u of the separate form per element, so the output is Dense's to
    /// rounding also with real multipoles; deterministic, the same at every thread count,
    /// ISA and rank count. Only [`HelmholtzM2l::Batched`] and [`HelmholtzM2l::Auto`] read
    /// it. A named mode holds for every later [`with_m2l`](Self::with_m2l) and
    /// [`with_p2p`](Self::with_p2p); unnamed, Auto's is [`auto_m2l_mul_add`] of the ISA and
    /// the others' [`MulAdd::Separate`].
    pub fn with_m2l_mul_add(mut self, mul_add: MulAdd) -> Self {
        self.kernel.mul_add = Some(mul_add);
        self.kernel.matrix = MatrixKernel::new(self.kernel.matrix.isa(), mul_add)
            .expect("the kernel's own ISA is available");
        self
    }

    /// The batched M2L's multiply–add mode ([`with_m2l_mul_add`](Self::with_m2l_mul_add)).
    pub fn m2l_mul_add(&self) -> MulAdd {
        self.kernel.matrix.mul_add()
    }

    /// The P2P kernel as it runs: [`P2pChoice::Reference`] or [`P2pChoice::Isa`] with the
    /// ISA of the kernel, never [`P2pChoice::Auto`].
    pub fn p2p_kernel(&self) -> P2pChoice {
        match &self.kernel.p2p {
            None => P2pChoice::Reference,
            Some(kernel) => P2pChoice::Isa(kernel.isa()),
        }
    }

    /// The degree p_l at which `level` runs, of its stores and its M2L (and of its leaves'
    /// operators but under [`PlaneWaveRaise::M2l`], [`translation_degree`](Self::translation_degree)):
    /// the degree given, or the raised degree of a level with waves under
    /// [`HelmholtzM2l::PlaneWave`]'s rule below ε = 10⁻⁶ and under [`HelmholtzM2l::Auto`] at
    /// 10⁻⁸ ≤ ε < 10⁻⁷ (Helmholtz CH6.1g; module documentation, "M2L").
    pub fn degree(&self, level: usize) -> usize {
        self.kernel.tables.m2l_degree(level)
    }

    /// The degree of `level`'s translations to and from its parent level (the M2M out of
    /// it reads, the L2L into it fills, its expansions' first degrees): [`degree`](Self::degree)
    /// unless [`PlaneWaveRaise::Leaves`] or [`PlaneWaveRaise::M2l`] keeps a raised level's at
    /// the degree given (Helmholtz CH6.1h); under [`PlaneWaveRaise::M2l`] also its leaves'
    /// operators'.
    pub fn translation_degree(&self, level: usize) -> usize {
        self.kernel.tables.degree(level)
    }

    /// The degree of `level` the operator was given ([`new`](Self::new)), before any raise.
    pub fn base_degree(&self, level: usize) -> usize {
        self.kernel.base_degrees[level]
    }

    /// The number of levels.
    pub fn nlevels(&self) -> usize {
        self.kernel.tables.nlevels()
    }

    /// The path.
    pub fn densities(&self) -> &Densities {
        &self.kernel.densities
    }

    /// The wavenumber.
    pub fn wavenumber(&self) -> Wavenumber {
        self.kernel.k
    }

    /// Whether the target output holds gradients.
    pub fn gradients(&self) -> bool {
        self.kernel.gradients
    }

    /// The RCR tables of the levels.
    pub fn tables(&self) -> &HelmholtzTables<T> {
        &self.kernel.tables
    }
}

impl<T: SimdScalar> Scratch<T> {
    fn new(kernel: &Kernel<T>) -> Self {
        let p_max = (0..kernel.tables.nlevels())
            .map(|l| kernel.tables.m2l_degree(l))
            .max()
            .unwrap_or(0);
        // The batched M2L's largest n, and m columns per pair of a block.
        let n_batched = kernel.batched.as_ref().map_or(0, |b| {
            (0..b.nlevels())
                .filter(|&l| b.is_batched(l))
                .map(|l| Layout::new(b.degree(l)).len())
                .max()
                .unwrap_or(0)
        });
        let pairs = match n_batched {
            0 => 0,
            _ => kernel.densities.vectors() * M2L_BLOCK_TARGETS,
        };
        Self {
            leaf: LeafScratch::new(p_max),
            workspace: Workspace::new(p_max),
            tables: kernel.tables.scratch(),
            classes: kernel
                .classes
                .as_ref()
                .map_or_else(|| ClassScratch::new(0), ClassM2l::scratch),
            first: vec![T::zero(); kernel.densities.local_size(p_max)],
            mapped: vec![[T::zero(); 3]; kernel.max_leaf_points],
            charges: vec![Complex::new(T::zero(), T::zero()); kernel.max_leaf_points],
            strengths: vec![T::zero(); 2 * kernel.max_leaf_points],
            widened: vec![T::zero(); 2 * Layout::new(p_max).len()],
            columns: Columns {
                pack: vec![T::zero(); MatrixKernel::<T>::pack_len(n_batched)],
                sources: Vec::with_capacity(pairs),
                targets: Vec::with_capacity(pairs),
            },
            plane: kernel.plane_waves.as_ref().map(PlaneScratch::new),
            diagonal: DiagonalScratch::new(
                &kernel.diagonal,
                &kernel.diagonal_settings,
                kernel.densities.vectors(),
            ),
        }
    }
}

impl<T: SimdScalar> Kernel<T> {
    /// The column kernel's mode on `isa`: the named one, else the setting's default (under
    /// [`HelmholtzM2l::Auto`] [`auto_m2l_mul_add`], else [`MulAdd::Separate`]).
    fn mul_add_for(&self, isa: Isa) -> MulAdd {
        self.mul_add.unwrap_or(match self.m2l {
            HelmholtzM2l::Auto => auto_m2l_mul_add(isa),
            _ => MulAdd::Separate,
        })
    }

    /// The crossover of [`HelmholtzM2l::Auto`]: the named one, else the rule's.
    fn auto_max_degree(&self) -> usize {
        self.max_degree.unwrap_or_else(|| {
            auto_batched_max_degree(self.matrix.isa(), self.densities.multipole_planes())
        })
    }

    /// m* of degree p under [`HelmholtzM2l::Auto`]'s term: the named table's (64 beyond it),
    /// else [`auto_batched_min_columns`] of the column kernel's ISA and the planes.
    fn auto_min_columns(&self, p: usize) -> f64 {
        match &self.min_columns {
            Some(table) => table.get(p).copied().unwrap_or(M2L_BLOCK_TARGETS as f64),
            None => {
                auto_batched_min_columns(self.matrix.isa(), self.densities.multipole_planes(), p)
            }
        }
    }

    /// The diagonal form's time rule's costs: named
    /// ([`with_diagonal_costs`](HelmholtzOperator::with_diagonal_costs)), else those of the
    /// column kernel's ISA (Helmholtz CH4.3b1).
    fn diagonal_costs(&self) -> DiagonalCosts {
        self.diagonal_settings
            .costs
            .unwrap_or_else(|| diagonal_costs(self.matrix.isa()))
    }

    /// The smallest degree at which [`HelmholtzM2l::Auto`] takes the exponential M2L: the
    /// named one, else the rule's.
    fn auto_plane_wave_min_degree(&self) -> usize {
        self.plane_wave_min_degree.unwrap_or_else(|| {
            auto_plane_wave_min_degree(self.matrix.isa(), self.densities.multipole_planes())
        })
    }

    /// Whether the setting is [`HelmholtzM2l::Auto`] with the exponential M2L, which runs in
    /// f64 only (CONVENTIONS §3.15, "Precision and range").
    fn auto_takes_plane_waves(&self) -> bool {
        auto_takes_plane_waves::<T>(self.m2l, self.auto_plane_wave)
    }

    /// The degree of `level`'s expansions and M2L.
    fn p(&self, level: usize) -> usize {
        self.tables.m2l_degree(level)
    }

    /// The degree of `level`'s leaf operators (P2M, L2P, P2L into and M2P from its boxes):
    /// its expansions' unless [`PlaneWaveRaise::M2l`] keeps a raised level's at its
    /// translation degree (Helmholtz CH6.1h).
    fn leaf_degree(&self, level: usize) -> usize {
        match self.plane_wave_raise.resolved(self.plane_wave_tolerance) {
            PlaneWaveRaise::M2l => self.tables.degree(level),
            PlaneWaveRaise::Rule | PlaneWaveRaise::Level | PlaneWaveRaise::Leaves => {
                self.tables.m2l_degree(level)
            }
        }
    }

    /// The batched M2L of `level`, if it has tables.
    fn batched(&self, level: usize) -> Option<Batched<'_, T>> {
        let tables = self.batched.as_ref().filter(|b| b.is_batched(level))?;
        Some(Batched {
            tables,
            matrix: &self.matrix,
            planes: self.densities.multipole_planes(),
            vectors: self.densities.vectors(),
        })
    }

    /// k̂ of `level` rounded to T.
    fn k_hat(&self, level: usize) -> Complex<T> {
        let k = self.tables.k_hat(level);
        Complex::new(T::from_f64(k.re), T::from_f64(k.im))
    }

    /// log₂ of a frame's radius, a power of two: the difference of two boxes' levels.
    fn levels_apart(frame: &Frame<T>) -> isize {
        frame.radius.to_f64().log2().round() as isize
    }

    /// The level `level` + `j`.
    fn shifted(level: usize, j: isize) -> usize {
        level
            .checked_add_signed(j)
            .expect("a frame's box lies on a level ≥ 0")
    }

    /// The points of a source chunk.
    fn source_points<'a>(&self, sources: &'a [T]) -> &'a [[T; 3]] {
        let n = sources.len() / self.densities.source_point_size();
        &sources.as_chunks::<3>().0[..n]
    }

    /// Vector v's charges of the n sources of `sources` into `out`, as complex values.
    fn charges_of(&self, sources: &[T], n: usize, v: usize, out: &mut [Complex<T>]) {
        assert!(
            n <= out.len(),
            "a leaf has {n} source points, more than max_leaf_points = {}",
            out.len()
        );
        let complex = self.densities.columns() == 2 * self.densities.vectors();
        for (j, q) in out[..n].iter_mut().enumerate() {
            let re = sources[self.densities.charge_index(n, j, v, 0)];
            let im = if complex {
                sources[self.densities.charge_index(n, j, v, 1)]
            } else {
                T::zero()
            };
            *q = Complex::new(re, im);
        }
    }

    fn p2m(&self, level: usize, sources: &[T], multipole: &mut [T], s: &mut Scratch<T>) {
        let n = self.source_points(sources).len();
        if n == 0 {
            return;
        }
        let (p, q) = (self.leaf_degree(level), self.p(level));
        if p == q {
            self.densities.p2m(
                &self.k,
                self.radii[level],
                p,
                sources,
                n,
                &mut s.leaf,
                multipole,
            );
        } else {
            // At p into zero, then added into the first degrees (PlaneWaveRaise::M2l).
            let first = &mut s.first[..self.densities.multipole_size(p)];
            first.fill(T::zero());
            self.densities.p2m(
                &self.k,
                self.radii[level],
                p,
                sources,
                n,
                &mut s.leaf,
                first,
            );
            add_to_first_degrees(first, p, multipole, q);
        }
    }

    fn l2p(&self, level: usize, local: &[T], targets: &[T], output: &mut [T], s: &mut Scratch<T>) {
        if targets.is_empty() {
            return;
        }
        let (p, q) = (self.leaf_degree(level), self.p(level));
        let local = if p == q {
            local
        } else {
            // The first degrees at p (PlaneWaveRaise::M2l).
            let first = &mut s.first[..self.densities.local_size(p)];
            copy_first_degrees(local, q, first, p);
            &*first
        };
        self.densities.l2p(
            &self.k,
            self.radii[level],
            p,
            local,
            targets,
            &mut s.leaf,
            output,
            self.gradients,
        );
    }

    fn m2m(&self, level: usize, o: usize, child: &[T], parent: &mut [T], s: &mut Scratch<T>) {
        let planes = self.densities.multipole_planes();
        let m = self.densities.vectors();
        self.tables
            .m2m(level, o, child, parent, planes, m, &mut s.tables);
    }

    fn l2l(&self, level: usize, o: usize, parent: &[T], child: &mut [T], s: &mut Scratch<T>) {
        let planes = self.densities.local_planes();
        let m = self.densities.vectors();
        self.tables
            .l2l(level, o, parent, child, planes, m, &mut s.tables);
    }

    fn m2l(
        &self,
        (level, d, child): (usize, usize, usize),
        multipole: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let planes = self.densities.multipole_planes();
        let m = self.densities.vectors();
        if let Some(merge) = self.plane_wave_level(level) {
            return self.plane_wave_pair(merge, (level, d, child), multipole, local, s);
        }
        if self
            .diagonal_level(level)
            .is_some_and(|diagonal| diagonal.m2l.has_offset(d))
        {
            let scratch = s
                .diagonal
                .as_mut()
                .expect("the scratch has the diagonal form's buffers");
            return self.diagonal_pair(level, d, multipole, local, scratch);
        }
        match (&self.dense, &self.classes, self.batched(level)) {
            (Some(dense), _, _) if dense.is_dense(level) => {
                dense.m2l(level, d, multipole, planes, local, m);
            }
            (_, Some(classes), _) if classes.is_class(level) => {
                classes.m2l(level, d, multipole, planes, local, m, &mut s.classes);
            }
            (_, _, Some(batched)) => batched.pair(level, d, multipole, local, &mut s.columns),
            _ => self
                .tables
                .m2l(level, d, multipole, planes, local, m, &mut s.tables),
        }
    }

    /// P2L into a box of `level` from a leaf at `frame` = (ĉ(t|s), r̂(t|s)), in the leaf's
    /// units.
    fn p2l(
        &self,
        level: usize,
        frame: &Frame<T>,
        sources: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let points = self.source_points(sources);
        let n = points.len();
        if n == 0 {
            return;
        }
        let (p, q) = (self.leaf_degree(level), self.p(level));
        // r̂(t|s) = r_t / r_s = 2^(l_s − l_t).
        let source_level = Self::shifted(level, Self::levels_apart(frame));
        let units = Level::new(self.k_hat(source_level), frame.radius);
        let size = self.densities.local_size(q) / self.densities.vectors();
        for (v, local) in local.chunks_exact_mut(size).enumerate() {
            self.charges_of(sources, n, v, &mut s.charges);
            // In the leaf's units the reference forms r_s φ̃; the box holds r_t φ̃:
            // the charges times r̂(t|s), exactly (a power of two).
            for q in &mut s.charges[..n] {
                *q = *q * frame.radius;
            }
            if p == q {
                leaf::p2l(
                    p,
                    &units,
                    frame,
                    points,
                    &s.charges[..n],
                    &mut s.workspace,
                    local,
                );
            } else {
                // At p into zero, then added into the first degrees (PlaneWaveRaise::M2l).
                let first = &mut s.first[..2 * Layout::new(p).len()];
                first.fill(T::zero());
                leaf::p2l(
                    p,
                    &units,
                    frame,
                    points,
                    &s.charges[..n],
                    &mut s.workspace,
                    first,
                );
                add_to_first_degrees(first, p, local, q);
            }
        }
    }

    /// M2P at the targets of a leaf of `level` from a box at `frame` = (ĉ(s|t), r̂(s|t)), in
    /// the leaf's units.
    fn m2p(
        &self,
        level: usize,
        frame: &Frame<T>,
        multipole: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let targets = targets.as_chunks::<3>().0;
        if targets.is_empty() {
            return;
        }
        // r̂(s|t) = r_s / r_t = 2^(l_t − l_s).
        let source_level = Self::shifted(level, -Self::levels_apart(frame));
        let (p, q) = (self.leaf_degree(source_level), self.p(source_level));
        let units = Level::new(self.k_hat(level), frame.radius);
        let planes = self.densities.multipole_planes();
        let (len, stride) = (Layout::new(p).len(), Layout::new(q).len());
        let Scratch {
            workspace,
            widened,
            first,
            ..
        } = s;
        for (v, multipole) in multipole.chunks_exact(planes * stride).enumerate() {
            // The first degrees at p (PlaneWaveRaise::M2l).
            let multipole = if p == q {
                multipole
            } else {
                let first = &mut first[..planes * len];
                copy_first_degrees(multipole, q, first, p);
                &*first
            };
            let multipole = if planes == 2 {
                multipole
            } else {
                let w = &mut widened[..2 * len];
                w[..len].copy_from_slice(multipole);
                w[len..].fill(T::zero());
                &*w
            };
            self.each_target(v, targets.len(), output, |i, phi, grad| {
                leaf::m2p(
                    p,
                    &units,
                    frame,
                    multipole,
                    &targets[i..=i],
                    workspace,
                    phi,
                    grad,
                );
            });
        }
    }

    /// P2P at the targets of a leaf of `level` from its own sources (`frame` `None`) or
    /// another leaf's at `frame` = (ĉ(s|t), r̂(s|t)).
    fn p2p(
        &self,
        level: usize,
        frame: Option<&Frame<T>>,
        sources: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        let points = self.source_points(sources);
        let targets = targets.as_chunks::<3>().0;
        let n = points.len();
        if n == 0 || targets.is_empty() {
            return;
        }
        let k = self.k_hat(level);
        let Scratch {
            mapped,
            charges,
            strengths,
            ..
        } = s;
        let points = match frame {
            None => points,
            Some(frame) => {
                assert!(
                    n <= mapped.len(),
                    "a leaf has {n} source points, more than max_leaf_points = {}",
                    mapped.len()
                );
                let mapped = &mut mapped[..n];
                for (y, u) in mapped.iter_mut().zip(points) {
                    // r̂ u is exact (r̂ is a power of two); the sum rounds once (§3.13).
                    *y = core::array::from_fn(|i| frame.centre[i] + frame.radius * u[i]);
                }
                &*mapped
            }
        };
        let Some(kernel) = &self.p2p else {
            for v in 0..self.densities.vectors() {
                self.charges_of(sources, n, v, charges);
                let charges = &charges[..n];
                self.each_target(v, targets.len(), output, |i, phi, grad| {
                    p2p::p2p(k, points, charges, &targets[i..=i], phi, grad);
                });
            }
            return;
        };
        let d = &self.densities;
        let (m, nt) = (d.vectors(), targets.len());
        // The real columns of a vector's charges: 1 (real) or 2 (complex).
        let c = d.columns() / m;
        // The potentials of every output column, then their gradients (CONVENTIONS §3.13).
        let (potentials, gradients) = output.split_at_mut(d.output_columns() * nt);
        for v in 0..m {
            let charges = if m == 1 {
                &sources[3 * n..3 * n + c * n]
            } else {
                assert!(
                    c * n <= strengths.len(),
                    "a leaf has {n} source points, more than max_leaf_points = {}",
                    strengths.len() / 2
                );
                let w = &mut strengths[..c * n];
                for (j, q) in w.chunks_exact_mut(c).enumerate() {
                    q.copy_from_slice(&sources[d.charge_index(n, j, v, 0)..][..c]);
                }
                &*w
            };
            let charges = if c == 1 {
                HelmholtzCharges::Real(charges)
            } else {
                HelmholtzCharges::Complex(charges)
            };
            let re = d.output_column(v, 0);
            let potential = &mut potentials[re * nt..(re + 2) * nt];
            let gradient = self
                .gradients
                .then(|| &mut gradients.as_chunks_mut::<3>().0[re * nt..(re + 2) * nt]);
            kernel.evaluate_helmholtz(k, points, charges, targets, potential, gradient);
        }
    }

    /// For each of the `nt` targets of vector v: loads its two output columns as complex
    /// values, lets `add` accumulate into them (a potential, and a gradient with
    /// gradients), and stores them back.
    fn each_target(
        &self,
        v: usize,
        nt: usize,
        output: &mut [T],
        mut add: impl FnMut(usize, &mut [Complex<T>], Option<&mut [[Complex<T>; 3]]>),
    ) {
        let d = &self.densities;
        let (re, im) = (d.output_column(v, 0), d.output_column(v, 1));
        for i in 0..nt {
            let (pr, pi) = (d.potential_index(nt, re, i), d.potential_index(nt, im, i));
            let mut phi = [Complex::new(output[pr], output[pi])];
            if self.gradients {
                let mut g = [core::array::from_fn(|axis| {
                    Complex::new(
                        output[d.gradient_index(nt, re, i, axis)],
                        output[d.gradient_index(nt, im, i, axis)],
                    )
                })];
                add(i, &mut phi, Some(&mut g));
                for (axis, c) in g[0].iter().enumerate() {
                    output[d.gradient_index(nt, re, i, axis)] = c.re;
                    output[d.gradient_index(nt, im, i, axis)] = c.im;
                }
            } else {
                add(i, &mut phi, None);
            }
            output[pr] = phi[0].re;
            output[pi] = phi[0].im;
        }
    }
}

/// The bodies above on the shared drivers.
impl<T: SimdScalar> HostKernel<T> for Kernel<T> {
    type Scratch = Scratch<T>;

    #[inline]
    fn vectors(&self) -> usize {
        self.densities.vectors()
    }

    #[inline]
    fn p2m(&self, level: usize, sources: &[T], multipole: &mut [T], s: &mut Scratch<T>) {
        Kernel::p2m(self, level, sources, multipole, s);
    }

    #[inline]
    fn m2m(&self, level: usize, o: usize, child: &[T], parent: &mut [T], s: &mut Scratch<T>) {
        Kernel::m2m(self, level, o, child, parent, s);
    }

    #[inline]
    fn m2l(
        &self,
        level: usize,
        d: usize,
        child: usize,
        multipole: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::m2l(self, (level, d, child), multipole, local, s);
    }

    #[inline]
    fn l2l(&self, level: usize, o: usize, parent: &[T], child: &mut [T], s: &mut Scratch<T>) {
        Kernel::l2l(self, level, o, parent, child, s);
    }

    #[inline]
    fn p2l(
        &self,
        level: usize,
        frame: &Frame<T>,
        sources: &[T],
        local: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::p2l(self, level, frame, sources, local, s);
    }

    #[inline]
    fn l2p(&self, level: usize, local: &[T], targets: &[T], output: &mut [T], s: &mut Scratch<T>) {
        Kernel::l2p(self, level, local, targets, output, s);
    }

    #[inline]
    fn m2p(
        &self,
        level: usize,
        frame: &Frame<T>,
        multipole: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::m2p(self, level, frame, multipole, targets, output, s);
    }

    #[inline]
    fn p2p(
        &self,
        level: usize,
        frame: Option<&Frame<T>>,
        sources: &[T],
        targets: &[T],
        output: &mut [T],
        s: &mut Scratch<T>,
    ) {
        Kernel::p2p(self, level, frame, sources, targets, output, s);
    }

    #[inline]
    fn batched_m2l(&self, _level: usize) -> Option<(&MatrixSet<T>, &MatrixKernel<T>)> {
        None
    }

    #[inline]
    fn columns(scratch: &mut Scratch<T>) -> &mut Columns<T> {
        &mut scratch.columns
    }
}

impl<T: SimdScalar + Equivalence + Default> FmmSizes for HelmholtzOperator<T> {
    type Value = T;

    /// [`Densities::multipole_size`] at p_l: m (p_l + 1)² or 2 m (p_l + 1)².
    fn multipole_size(&self, level: usize) -> usize {
        self.kernel.densities.multipole_size(self.degree(level))
    }

    /// [`Densities::local_size`] at p_l: 2 m (p_l + 1)².
    fn local_size(&self, level: usize) -> usize {
        self.kernel.densities.local_size(self.degree(level))
    }

    /// [`Densities::source_point_size`]: 3 + c m.
    fn source_point_size(&self) -> usize {
        self.kernel.densities.source_point_size()
    }

    /// 3, the leaf-scaled position.
    fn target_input_point_size(&self) -> usize {
        3
    }

    /// [`Densities::target_output_point_size`]: 2 m, or 8 m with gradients.
    fn target_output_point_size(&self) -> usize {
        self.kernel
            .densities
            .target_output_point_size(self.kernel.gradients)
    }
}

/// Adds the expansions of degree p in `from`, one plane after the other, to the first degrees
/// of as many planes of degree q ≥ p in `to` (`PlaneWaveRaise::M2l`; Helmholtz CH6.1h).
fn add_to_first_degrees<T: SimdScalar>(from: &[T], p: usize, to: &mut [T], q: usize) {
    let (len, stride) = (Layout::new(p).len(), Layout::new(q).len());
    for (from, to) in from.chunks_exact(len).zip(to.chunks_exact_mut(stride)) {
        for (t, &f) in to.iter_mut().zip(from) {
            *t = *t + f;
        }
    }
}

/// Copies the first degrees ≤ p of the planes of degree q in `from` into planes of degree p
/// in `to` (`PlaneWaveRaise::M2l`; Helmholtz CH6.1h).
fn copy_first_degrees<T: SimdScalar>(from: &[T], q: usize, to: &mut [T], p: usize) {
    let (len, stride) = (Layout::new(p).len(), Layout::new(q).len());
    for (from, to) in from.chunks_exact(stride).zip(to.chunks_exact_mut(len)) {
        to.copy_from_slice(&from[..len]);
    }
}

/// Target by target through the shared drivers, but for the M2L of a level with batched
/// tables (module documentation, "Execution" and "M2L").
impl<T: SimdScalar + Equivalence + Default> FmmOperator for HelmholtzOperator<T> {
    fn p2m(&mut self, batch: P2m<'_, T>) {
        let (level, start) = (batch.level, Instant::now());
        driver::p2m(&self.kernel, &mut self.execution, batch);
        self.kind_times[level][0] += start.elapsed();
    }

    fn m2m(&mut self, batch: M2m<'_, T>) {
        let (level, start) = (batch.level, Instant::now());
        driver::m2m(&self.kernel, &mut self.execution, batch);
        self.kind_times[level][1] += start.elapsed();
    }

    /// Under [`HelmholtzM2l::Batched`] block by block on the levels with tables, under
    /// [`HelmholtzM2l::PlaneWave`] by merge and shift on the levels with waves (module
    /// documentation, "M2L"), target by target elsewhere.
    ///
    /// Each call's wall time is added to its level's [`m2l_times`](Self::m2l_times).
    fn m2l(&mut self, batch: M2l<'_, T>) {
        let level = batch.level;
        let start = Instant::now();
        if self.kernel.plane_wave_level(level).is_some() {
            self.plane_wave_m2l(batch);
        } else if self.kernel.diagonal_level(level).is_some() {
            self.diagonal_m2l(batch);
        } else {
            match self.kernel.batched(level) {
                Some(batched) => {
                    // The block width: named, or by the level's boxes and the threads that
                    // run the call (Helmholtz CH4.0), which changes no bit.
                    let threads = match (&self.execution.pool, self.execution.serial) {
                        (Some(pool), false) => pool.current_num_threads(),
                        _ => 1,
                    };
                    let width = self
                        .kernel
                        .block_targets
                        .unwrap_or_else(|| batched_block_targets(batch.locals.len(), threads));
                    batched.m2l(&mut self.execution, batch, width, |s| &mut s.columns)
                }
                None if self.m2l_on(level) == HelmholtzM2l::Rotation => {
                    self.rotation_m2l(batch);
                }
                None => driver::m2l(&self.kernel, &mut self.execution, batch),
            }
        }
        self.m2l_times[level] += start.elapsed();
    }

    fn p2l(&mut self, batch: P2l<'_, T>) {
        let (level, start) = (batch.level, Instant::now());
        driver::p2l(&self.kernel, &mut self.execution, batch);
        self.kind_times[level][4] += start.elapsed();
    }

    fn l2l(&mut self, batch: L2l<'_, T>) {
        let (level, start) = (batch.level, Instant::now());
        driver::l2l(&self.kernel, &mut self.execution, batch);
        self.kind_times[level][2] += start.elapsed();
    }

    fn l2p(&mut self, batch: L2p<'_, T>) {
        let (level, start) = (batch.level, Instant::now());
        driver::l2p(&self.kernel, &mut self.execution, batch);
        self.kind_times[level][3] += start.elapsed();
    }

    fn m2p(&mut self, batch: M2p<'_, T>) {
        let (level, start) = (batch.level, Instant::now());
        driver::m2p(&self.kernel, &mut self.execution, batch);
        self.kind_times[level][5] += start.elapsed();
    }

    fn p2p(&mut self, batch: P2p<'_, T>) {
        let level = batch.level;
        let start = Instant::now();
        driver::p2p(&self.kernel, &mut self.execution, batch);
        self.p2p_times[level] += start.elapsed();
    }
}

/// The same bodies pair by pair, for `nd_fmm_plan::operator::PerPair` and the tests: the
/// level from the keys, the octant or offset index as given (debug builds check it
/// against the keys), the frames from the keys, the self pair of P2P by
/// `source == target`. Serial, on the first scratch set.
impl<T: SimdScalar + Equivalence + Default> PairOperator for HelmholtzOperator<T> {
    fn p2m(&mut self, leaf: MortonKey, sources: &[T], multipole: &mut [T]) {
        let level = morton::level(leaf) as usize;
        self.kernel
            .p2m(level, sources, multipole, self.execution.first());
    }

    fn m2m(
        &mut self,
        child: MortonKey,
        parent: MortonKey,
        octant: usize,
        child_multipole: &[T],
        parent_multipole: &mut [T],
    ) {
        debug_check_octant(child, parent, octant);
        let level = morton::level(parent) as usize;
        self.kernel.m2m(
            level,
            octant,
            child_multipole,
            parent_multipole,
            self.execution.first(),
        );
    }

    fn m2l(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        offset: usize,
        source_multipole: &[T],
        target_local: &mut [T],
    ) {
        debug_assert_eq!(
            offset_index(source, target),
            offset,
            "the offset index of a V-list pair disagrees with its keys"
        );
        let level = morton::level(target) as usize;
        let child = morton::child_index(target);
        self.kernel.m2l(
            (level, offset, child),
            source_multipole,
            target_local,
            self.execution.first(),
        );
    }

    fn p2l(&mut self, source: MortonKey, target: MortonKey, sources: &[T], local: &mut [T]) {
        let frame = relative_frame(target, source);
        let level = morton::level(target) as usize;
        self.kernel
            .p2l(level, &frame, sources, local, self.execution.first());
    }

    fn l2l(
        &mut self,
        parent: MortonKey,
        child: MortonKey,
        octant: usize,
        parent_local: &[T],
        child_local: &mut [T],
    ) {
        debug_check_octant(child, parent, octant);
        let level = morton::level(child) as usize;
        self.kernel.l2l(
            level,
            octant,
            parent_local,
            child_local,
            self.execution.first(),
        );
    }

    fn l2p(&mut self, leaf: MortonKey, local: &[T], target_input: &[T], output: &mut [T]) {
        let level = morton::level(leaf) as usize;
        self.kernel
            .l2p(level, local, target_input, output, self.execution.first());
    }

    fn m2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        let frame = relative_frame(source, target);
        let level = morton::level(target) as usize;
        self.kernel.m2p(
            level,
            &frame,
            source_multipole,
            target_input,
            target_output,
            self.execution.first(),
        );
    }

    fn p2p(
        &mut self,
        source: MortonKey,
        target: MortonKey,
        sources: &[T],
        target_input: &[T],
        target_output: &mut [T],
    ) {
        let frame = (source != target).then(|| relative_frame(source, target));
        let level = morton::level(target) as usize;
        self.kernel.p2p(
            level,
            frame.as_ref(),
            sources,
            target_input,
            target_output,
            self.execution.first(),
        );
    }
}

#[cfg(test)]
mod tests {
    use nd_fmm_math::Complex;
    use nd_fmm_tables::helmholtz::plane_wave::Precision;
    use nd_octree::PhysicalBox;

    use super::{
        DEFAULT_BLOCK_BUDGET, DiagonalCosts, HelmholtzDiagonal, HelmholtzM2l, HelmholtzOperator,
        PlaneWavePrecision, WaveSettings, wave_levels,
    };
    use crate::geometry::{Domain, radius};
    use crate::helmholtz::{ChargeKind, Densities, Wavenumber};
    use crate::operator::SimdScalar;

    /// Small degrees (the tables dominate the test's time), each with waves under the rule.
    const DEGREES: [usize; 6] = [0, 0, 3, 4, 6, 8];

    /// Every level's quadrature in the operator built with `settings` at k, and in
    /// `wave_levels` of the same settings (Helmholtz CHA.11): the builder's V rule reads the
    /// second, so the two must be one choice.
    fn check<T: SimdScalar>(k: Wavenumber, settings: &WaveSettings<'_>) {
        let domain = Domain::new(&PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).unwrap();
        let densities = Densities::new(k.kind(), ChargeKind::Complex, 1);
        // The settings first and the tables once, as `HelmholtzFmmBuilder::build` does.
        let unbuilt = || {
            HelmholtzOperator::<T>::unbuilt(
                k,
                densities,
                &domain,
                &DEGREES,
                false,
                8,
                DEFAULT_BLOCK_BUDGET,
            )
            .unwrap()
            .with_plane_wave_tolerance(settings.tolerance)
            .with_m2l_plane_wave_min_degree(Some(settings.min_degree))
            .with_auto_plane_wave(settings.auto_plane_wave)
            .with_plane_wave_precision(settings.precision.clone())
            .with_m2l(settings.m2l, 0)
        };
        let op = unbuilt().build_tables();
        // The diagonal form's time rule at free costs takes levels from the waves after the
        // plan (Helmholtz CH4.3b1): the operator's levels with waves then lie within the
        // function's.
        let free = unbuilt()
            .with_diagonal(HelmholtzDiagonal::Auto)
            .with_diagonal_costs(Some(FREE_DIAGONAL))
            .build_tables();
        let r0 = radius(0, &domain);
        let k_root = Complex::new(k.re() * r0, k.im() * r0);
        let levels = wave_levels::<T>(settings, k_root, &DEGREES);
        let operator: Vec<Option<Precision>> = (0..DEGREES.len())
            .map(|l| op.plane_wave_precision(l))
            .collect();
        let chosen: Vec<Option<Precision>> = match &levels {
            Some(levels) => levels.clone(),
            None => vec![None; DEGREES.len()],
        };
        let case = format!("k {k:?}, {settings:?}, {}", std::any::type_name::<T>());
        assert_eq!(chosen, operator, "{case}");
        for (l, chosen) in chosen.iter().enumerate() {
            let waves = free.plane_wave_precision(l);
            assert!(waves.is_none() || waves == *chosen, "{case}, level {l}");
        }
        // `None` exactly where the setting runs no exponential M2L.
        let runs = settings.m2l == HelmholtzM2l::PlaneWave
            || (settings.m2l == HelmholtzM2l::Auto
                && settings.auto_plane_wave
                && std::mem::size_of::<T>() == std::mem::size_of::<f64>());
        assert_eq!(levels.is_some(), runs, "{case}");
    }

    /// The settings of one k and tolerance: PlaneWave's rule and named precisions, Auto with
    /// waves on at two smallest degrees and off, and another M2L.
    fn settings<'a>(
        tolerance: Option<f64>,
        precisions: [&'a PlaneWavePrecision; 3],
    ) -> Vec<WaveSettings<'a>> {
        let [rule, ..] = precisions;
        let plane_wave = precisions.map(|precision| WaveSettings {
            m2l: HelmholtzM2l::PlaneWave,
            precision,
            auto_plane_wave: false,
            min_degree: 0,
            tolerance,
        });
        let auto =
            [(true, 0), (true, 6), (false, 0)].map(|(auto_plane_wave, min_degree)| WaveSettings {
                m2l: HelmholtzM2l::Auto,
                precision: rule,
                auto_plane_wave,
                min_degree,
                tolerance,
            });
        let rotation = WaveSettings {
            m2l: HelmholtzM2l::Rotation,
            precision: precisions[1],
            auto_plane_wave: true,
            min_degree: 0,
            tolerance,
        };
        plane_wave
            .into_iter()
            .chain(auto)
            .chain([rotation])
            .collect()
    }

    /// Costs under which the diagonal form beats every other M2L wherever a class passes.
    const FREE_DIAGONAL: DiagonalCosts = DiagonalCosts {
        s2f: [0.0, 0.0],
        f2r: [0.0, 0.0],
        pair: [0.0, 0.0],
        rotation: [1.0, 1.0],
        plane_wave: [1e12, 1e12],
    };

    /// The levels with waves under every setting that chooses them (Helmholtz CHA.11), at no
    /// tolerance and four, for real k and, under Auto, damped k (Im κ₂ = 25 ≥ 5π), f64 and
    /// (where the gate reads it) f32; and where κ₂ lies outside the tables, by its real part
    /// and by its imaginary part, so that named precisions take no waves there (the builder
    /// assumed they did before CHA.11). The operator calls `wave_levels` itself, so the test
    /// guards the levels its tables build against the function; Auto's and the rule's choice
    /// are checked against their definitions in `tests/helmholtz/plane_wave.rs`.
    #[test]
    fn wave_levels_are_the_operators() {
        use Precision::{Iprec1, Iprec2, Iprec3};
        let rule = PlaneWavePrecision::Rule;
        let named = PlaneWavePrecision::PerLevel(vec![Some(Iprec2); 64]);
        let mixed = PlaneWavePrecision::PerLevel(vec![
            Some(Iprec3),
            None,
            Some(Iprec1),
            None,
            Some(Iprec2),
        ]);
        for k in [
            Wavenumber::new(20.0, 0.0).unwrap(),
            Wavenumber::new(20.0, 100.0).unwrap(),
        ] {
            for tolerance in [None, Some(1e-3), Some(1e-6), Some(1e-8), Some(1e-9)] {
                for settings in settings(tolerance, [&rule, &named, &mixed]) {
                    // The damping bound is Auto's alone.
                    if k.im() > 0.0 && settings.m2l == HelmholtzM2l::PlaneWave {
                        continue;
                    }
                    check::<f64>(k, &settings);
                    if settings.m2l != HelmholtzM2l::PlaneWave {
                        check::<f32>(k, &settings);
                    }
                }
            }
        }
        // κ_l = 400 · 2⁻ˡ: level 2 outside the tables (Re κ₂ = 100 ≥ 16π), levels 3–5 inside;
        // κ_l = (20 + 200i) 2⁻ˡ: level 2 outside them by its damping (Im κ₂ = 50 ≥ 12π).
        for k in [
            Wavenumber::new(400.0, 0.0).unwrap(),
            Wavenumber::new(20.0, 200.0).unwrap(),
        ] {
            for tolerance in [None, Some(1e-6)] {
                for settings in &settings(tolerance, [&rule, &named, &mixed])[..4] {
                    check::<f64>(k, settings);
                }
            }
        }
    }
}
