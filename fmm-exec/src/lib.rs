//! # `nd-fmm-exec`
//!
//! The Laplace kernel of the nd-project FMM on top of `nd-fmm-plan`, host path. The
//! plan supplies the tree, the interaction lists, the data stores and the ghost
//! exchange, and stays kernel-agnostic; this crate supplies everything that knows about
//! 1/|x − y|: box geometry from Morton keys, the leaf-scaled point data, the Laplace
//! operator on the plan's level-batched interface (from the tables of `nd-fmm-tables`,
//! the leaf operators of `nd-fmm-ref` and the SIMD P2P kernel of `nd-fmm-simd`), and the
//! user-facing FMM object. It is generic over f32 and f64 ([`RealScalar`] and
//! `nd_fmm_simd::SimdScalar`). With the `gpu` feature it also has a device path
//! (Phase 4): the same FMM with its operators on a device of `nd-fmm-kernels`.
//!
//! ## Conventions
//!
//! All basis functions, phases, storage and scaling are defined in
//! [`docs/CONVENTIONS.md`][conventions], the single source of truth; code cites it as
//! `CONVENTIONS §3.x`. This crate relies on:
//!
//! - Kernel (§3.1): every operator expands 1/|x − y|. The factor 1/(4π) is applied
//!   here, once, when output is produced.
//! - Scaling (§3.7): coefficients are stored as M̃ = M / rⁿ and L̃ = L · rⁿ⁺¹ with r
//!   the box half-width, so the translation tables are the same on every level.
//! - Box geometry and operator tables (§3.12): the cubic domain with lower corner a and
//!   side w, the box centres and half-widths of each level, the child index
//!   o = 4x + 2y + z and the order of the 316 V-list offsets.
//! - Leaf data and relative geometry (§3.13): integer centres, the exact relative
//!   frames between boxes, leaf-scaled point coordinates, the layout of source and
//!   target chunks, and the output scaling φ = φ̂ / (4π r_t), ∇φ = ĝ / (4π r_t²).
//!
//! ## Rules
//!
//! - **Tables by integer key.** M2M and L2L tables are looked up by the child index
//!   `morton::child_index`, M2L tables by the index of the V-list offset, never by a
//!   floating-point shift (§3.12). The order of `nd-fmm-tables` agrees with that of
//!   `nd-octree` and of `nd_fmm_plan::interaction_manager::V_LIST_DIRECTIONS`; the
//!   tests of this crate check it.
//! - **Geometry from integer keys.** Every operator takes its geometry from
//!   [`geometry`]: relative frames from [`geometry::relative_frame`], computed from the
//!   integer key indices alone and exact in f32 and f64. No library code forms a shift
//!   as the difference of two floating-point centres. The domain enters only when
//!   points are loaded ([`geometry::leaf_coordinates`]) and when output is produced.
//! - **1/(4π) once.** Every operator accumulates (+=) into leaf-scaled output. The
//!   factor 1/(4π) and the powers of the target leaf radius are applied once, by the
//!   FMM object when it hands output to the caller (§3.1, §3.13), never inside an
//!   operator or a table.
//! - **MPI discipline.** Every rank reaches every collective in the same order, also a
//!   rank with empty input, and an error that depends on one rank's input is agreed by
//!   all ranks before the next collective. Operator and geometry code never calls MPI;
//!   at most one test per test executable initialises it.
//! - **Threads.** Opt-in ([`FmmBuilder::threads`](fmm::FmmBuilder::threads), default 1):
//!   rayon over the targets of each level call, in a pool the FMM owns, bit-identical to
//!   the serial path for every thread count. Worker threads never call MPI or BLAS. To
//!   use threads, initialise MPI with
//!   `mpi::initialize_with_threading(mpi::Threading::Funneled)`; see [`threading`].
//!
//! ## Contents
//!
//! - [`geometry`]: the validated cubic [`Domain`](geometry::Domain), box centres and
//!   radii (§3.12), integer centres, exact relative frames and leaf-scaled coordinates
//!   (§3.13).
//! - [`tables`]: the M2L strategy ([`M2lStrategy`](tables::M2lStrategy)) and the
//!   translation tables it needs ([`Tables`](tables::Tables)), built or loaded from a
//!   cache, applied by child index and offset index (§3.11, §3.12).
//! - [`operator`]: [`LaplaceOperator`](operator::LaplaceOperator), the Laplace kernel on
//!   the level-batched interface of `nd-fmm-plan` and on its per-pair interface, with
//!   per-pair kernels for every operator (§3.11–§3.13; C3.1), and the choice of its P2P
//!   kernel, [`P2pChoice`](operator::P2pChoice) (C3S.5).
//! - [`fmm`]: the user-facing [`FmmBuilder`](fmm::FmmBuilder) and [`Fmm`](fmm::Fmm),
//!   which load the caller's points into an octree and evaluate potentials and
//!   gradients in the caller's order, with 1/(4π) applied once (§3.1, §3.13; C3.2),
//!   optionally on several threads (C3.5).
//! - [`threading`]: the rules for rayon threads, MPI and BLAS, how to launch with one
//!   BLAS thread, and the [`ThreadingReport`](threading::ThreadingReport) of an FMM.
//! - `device` (feature `gpu`): the device path, `DeviceOperator`, its report and its
//!   transfer accounting (C4.1).
//!
//! ## Device path (feature `gpu`)
//!
//! | Feature | Enables |
//! | --- | --- |
//! | `gpu` | `nd-fmm-kernels`, the `device` module and the device backends' settings |
//! | `cpu` | `gpu` and the CubeCL CPU runtime ([`Backend::Cpu`](fmm::Backend::Cpu)): f32 and f64, the correctness backend |
//! | `metal` | `gpu` and Metal ([`Backend::Metal`](fmm::Backend::Metal)): f32 only, run outside the macOS sandbox |
//! | `cuda` | `gpu` and CUDA ([`Backend::Cuda`](fmm::Backend::Cuda)): type-checked only |
//!
//! None is on by default, and [`Backend::Host`](fmm::Backend::Host) stays the default
//! backend, so the host path builds and runs exactly as without the features. The
//! `device` module documents:
//!
//! - **residency**: views, geometry and tables uploaded once per `Fmm`, the points
//!   once per build, the charges once and the output once per evaluation; nothing is
//!   allocated on the device during an evaluation;
//! - **host fallback**: every operator kind can run on the host with explicit
//!   transfers; with every kind there the output equals the host path's bit for bit.
//!   From Phase 4 T6 P2P runs on the device by default (`fmm::DeviceP2pLayout`);
//! - **determinism**: every launch and transfer is issued from the calling thread, in
//!   order on one stream; two evaluations are bit-identical;
//! - **errors**: device settings are refused at build with a
//!   [`SettingsError`](fmm::SettingsError) agreed by step 1's all-reduce; a device runs on
//!   one rank until C5.1; device failures are [`FmmError::Device`](fmm::FmmError::Device);
//! - **threads**: with the CPU runtime no rayon pool, `threads(n)` caps its units per
//!   cube; with Metal or CUDA the pool serves the host-fallback kinds only.
//!
//! `nd-fmm-exec` stays free of `unsafe` and reaches CubeCL only through
//! `nd-fmm-kernels`.
//!
//! ## Example
//!
//! A complete evaluation on one rank: potentials and gradients of random charges at the
//! sources themselves (a point does not act on itself).
//!
//! ```no_run
//! use mpi::traits::Communicator;
//! use nd_fmm_exec::fmm::FmmBuilder;
//!
//! let universe = mpi::initialize().expect("MPI initialises once");
//! let comm = universe.world();
//!
//! let points: Vec<[f64; 3]> = (0..10_000)
//!     .map(|i| {
//!         let t = i as f64;
//!         [(0.37 * t).sin(), (0.71 * t).cos(), (0.13 * t).sin()]
//!     })
//!     .collect();
//! let charges: Vec<f64> = (0..points.len()).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
//!
//! // Degree 8, with gradients; every other setting at its default.
//! let mut fmm = FmmBuilder::<f64>::new(8)
//!     .gradients(true)
//!     .build(&points, &points, &comm)
//!     .expect("on one rank every point is owned");
//! let output = fmm.evaluate(&charges).expect("one charge per source");
//!
//! // φ(xᵢ) = Σⱼ qⱼ / (4π |xᵢ − yⱼ|), in the order of `points`.
//! assert_eq!(output.potential.len(), points.len());
//! let gradient = output.gradient.expect("built with gradients");
//! println!(
//!     "rank {}: φ(x₀) = {:e}, ∇φ(x₀) = {:?}, {} leaves on {} levels, M2L {:?}",
//!     comm.rank(),
//!     output.potential[0],
//!     gradient[0],
//!     fmm.nleaves(),
//!     fmm.nlevels(),
//!     fmm.strategy()
//! );
//! ```
//!
//! [conventions]: https://github.com/tbetcke/fmm/blob/main/docs/CONVENTIONS.md
//! [`RealScalar`]: nd_fmm_math::RealScalar

#[cfg(feature = "gpu")]
pub mod device;
pub mod fmm;
pub mod geometry;
pub mod operator;
pub mod tables;
pub mod threading;
