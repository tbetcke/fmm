//! # `nd-octree`
//!
//! `nd-octree` builds adaptive, 2:1-balanced three-dimensional octrees from
//! Morton keys. It is a topology library: an [`Octree`] stores keys, ownership,
//! key classifications, and neighbour relationships, **not** the input point
//! arrays or application data attached to them. It can be used with one MPI rank
//! or distributed across a communicator.
//!
//! ## Prerequisites and quick start
//!
//! MPI is a required dependency, including for a one-rank program. Install an
//! MPI implementation and its development headers before building this crate.
//! Construction is distributed, so initialize MPI and keep both the MPI universe
//! and communicator alive for at least as long as the tree. The example below is
//! `no_run` because MPI programs should be launched by an MPI launcher rather
//! than by rustdoc:
//!
//! ```no_run
//! use mpi::traits::Communicator;
//! use nd_octree::{
//!     constants::DEEPEST_LEVEL,
//!     morton,
//!     octree::{compute_global_bounding_box, OctreeOptions},
//!     points_to_morton, Octree,
//! };
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//! use rlst::rlst_dynamic_array;
//!
//! let universe = mpi::initialize().expect("MPI must be initialized once");
//! let world = universe.world();
//!
//! // Each rank contributes 1,000 standard-normal points in a `[3, n]` array;
//! // each column is one `[x, y, z]` point. The rank seeds its stream deterministically.
//! let mut rng = ChaCha8Rng::seed_from_u64(world.rank() as u64);
//! let mut points = rlst_dynamic_array!(f64, [3, 1_000]);
//! points.fill_from_standard_normal(&mut rng);
//!
//! // The box and tree construction are collective; mapping local points is not.
//! let bounding_box = compute_global_bounding_box(&points, &world);
//! let fine_keys = points_to_morton(&points, DEEPEST_LEVEL as usize, &bounding_box);
//! let query = fine_keys[0];
//! // Refine up to level 12, targeting at most 4 finest-level keys per leaf.
//! // At the level limit, a leaf may still contain more than 4 keys.
//! let options = OctreeOptions::new().with_max_level(12).with_max_fine_keys(4);
//! let tree = Octree::new(&fine_keys, options, &world);
//!
//! // All ranks must participate in this collective call, even with `&[]`.
//! let found = tree.lookup_leaves(&[query]).expect("MPI counts fit");
//! let location = found[0].expect("a valid finest-level key has a leaf");
//! assert!(morton::is_ancestor(location.leaf, query));
//! assert!(location.owner_rank < world.size() as usize);
//! ```
//!
//! For a complete MPI invocation, build one of the executable examples and run
//! it with `mpirun`; see the repository's `examples/` directory and
//! [Testing and performance](#testing-and-performance).
//!
//! ## From points to a tree
//!
//! Point arrays accepted by [`points_to_morton`] and
//! [`octree::compute_global_bounding_box`] have shape **`[3, n]`**: each column
//! is one `[x, y, z]` point. A [`PhysicalBox`] uses
//! `[xmin, ymin, zmin, xmax, ymax, zmax]`. Points mapped to keys must be inside
//! that box: a point on or beyond a face is clamped into the nearest cell rather
//! than reported, so keep points inside the box if you need a faithful mapping.
//! [`octree::compute_global_bounding_box`] panics for a degenerate domain, which
//! happens when no rank contributes a point or when every point coincides.
//!
//! A typical application computes a collective box, maps its local points at
//! [`constants::DEEPEST_LEVEL`], and passes those keys to [`Octree::new`]. The
//! mapping itself is local; the global-box computation and construction are
//! collective. For example, the geometric and key primitives can also be used
//! without constructing a tree:
//!
//! ```
//! use nd_octree::{morton, PhysicalBox};
//!
//! let domain = PhysicalBox::new([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
//! let key = morton::from_physical_point([0.5, 1.0, 1.5], &domain, 4);
//! assert_eq!(morton::decode(key), (4, [4, 8, 12]));
//! assert_eq!(morton::physical_box(key, &domain).coordinates(),
//!            [0.5, 1.0, 1.5, 0.625, 1.125, 1.625]);
//! ```
//!
//! [`Octree::new`] takes the local keys, one [`OctreeOptions`] value carrying
//! every other setting, and the communicator.
//! [`OctreeOptions::with_max_level`] bounds the leaf level (at most 16) on any
//! number of ranks, while [`OctreeOptions::with_max_fine_keys`] sets the target
//! maximum number of finest keys in a leaf.
//! Construction linearizes, partitions, refines, and balances the distributed
//! topology. It may redistribute keys internally, but the resulting `Octree`
//! retains topology only; maintain application data and any corresponding
//! redistribution outside the crate. Keys must be valid and at level 16; a rank
//! may contribute no keys at all. Ownership is partitioned over a coarse tree
//! whose depth is bounded by `max_level`, and every rank needs at least one of
//! its blocks, so construction panics with a descriptive message when very few
//! distinct keys, a small `max_level`, or many ranks leave fewer blocks than
//! ranks. See [`Octree::new`].
//!
//! ## Morton keys
//!
//! [`MortonKey`] is `u64`. Levels 0 through 16 are supported. The low 15 bits
//! encode the level and the next 48 bits encode the interleaved position; the
//! high bit is an invalid marker. The root is the valid key `0`, obtained with
//! [`morton::root`], so it must not be used as an invalid sentinel. Use
//! [`morton::invalid_key`] and [`morton::is_valid`] instead. Prefer constructors
//! such as [`morton::from_index_and_level`] instead of manual bit operations.
//!
//! ```
//! use nd_octree::morton;
//!
//! let root = morton::root();
//! assert!(morton::is_valid(root));
//! assert_eq!(morton::level(root), 0);
//! let child = morton::from_index_and_level([1, 0, 1], 1);
//! assert_eq!(morton::parent(child), Some(root));
//! assert!(!morton::is_valid(morton::invalid_key()));
//! ```
//!
//! ## Topology, ownership, and neighbours
//!
//! [`Octree::leaf_keys`] returns locally owned leaves. [`Octree::all_keys`]
//! classifies every stored key as [`octree::KeyType::LocalLeaf`],
//! [`octree::KeyType::LocalInterior`], [`octree::KeyType::Global`],
//! [`octree::KeyType::GhostLeaf`], or [`octree::KeyType::GhostInterior`]. Global
//! keys are shared ancestors of the distributed coarse tree. Ghost variants
//! carry the originating, zero-based owner rank; use
//! [`octree::KeyType::ghost_rank`] when that rank is needed. Ghosts provide
//! interface topology but are not local ownership.
//!
//! The same [`OctreeOptions`] value selects the optional layers. With
//! [`OctreeOptions::with_ghost_children`] enabled, the tree also stores the
//! children of the interior boxes that border this rank: for every non-ghost key
//! and every same-level neighbour cell of it that the tree holds as an interior
//! box, all eight children of that neighbour are keys of [`Octree::all_keys`]
//! too. That is one step of closure, not a fixed point, and it is what a fast
//! multipole method needs to resolve V- and W-list entries locally.
//!
//! [`Octree::neighbour_map`] has entries only for non-ghost keys. Its values are
//! distinct adjacent keys, which may include ghosts. Interior-key neighbours are
//! at the same level. Each leaf entry contains, for every neighbouring cell,
//! either the same-level neighbour (which may be an interior key) or that
//! neighbour's parent, one level coarser; a coarser neighbour is listed once
//! even when it covers several neighbouring cells.
//!
//! ## Operations and communication
//!
//! | Operation | Scope | Notes |
//! | --- | --- | --- |
//! | [`points_to_morton`] | local | Maps `[3, n]` points using an explicit box. |
//! | [`octree::compute_global_bounding_box`] | collective | All ranks contribute a padded cubic box. |
//! | [`Octree::new`] | collective | All ranks in the communicator construct one distributed topology. Optional layers selected by [`OctreeOptions`] enlarge the exchanged payload but add no round. |
//! | [`Octree::owner_rank`] | local | Valid finest-level key to its owner, without communication. |
//! | [`Octree::local_leaf`] | local | Finds a containing leaf only when this rank owns the key. |
//! | [`Octree::lookup_leaves`] | collective | Routes batch queries to owners and preserves input order. |
//! | [`Octree::global_max_level`] | collective | Reduces the local maximum level. |
//! | [`vtk::write_vtu`] | local | Writes the given leaves as one serial `.vtu` document. |
//! | [`vtk::write_pvtu`] | collective | Writes one `.vtu` piece per rank and a `.pvtu` manifest. |
//!
//! `owner_rank` and `local_leaf` reject invalid keys first, then keys that are
//! not at the finest level with [`LookupError`]. `local_leaf` returns `Ok(None)`
//! for a valid key owned by another rank, even if a ghost is present. In contrast,
//! `lookup_leaves` returns `Result<Vec<Result<LeafLocation, LookupError>>,
//! LookupBatchError>`: individual malformed queries remain in the inner result,
//! while the outer error reports an MPI count/displacement overflow that prevents
//! the batch exchange. Every rank must enter each `lookup_leaves` call in the
//! same order, including ranks with an empty slice or only invalid keys.
//!
//! To look up physical points, compose the local mapping and collective lookup:
//! map points with [`points_to_morton`] at level 16 using the same box used for
//! construction, then pass the resulting keys to `lookup_leaves` collectively.
//! The crate intentionally does not retain point arrays or provide a separate
//! physical-point lookup API.
//!
//! ## Visualization
//!
//! [`vtk`] exports leaves as VTK unstructured-grid geometry for viewers such as
//! ParaView, with no dependencies beyond the crate itself. [`vtk::write_vtu`]
//! writes one serial piece to any `std::io::Write` sink;
//! [`vtk::write_pvtu`] is collective and writes a complete parallel data set,
//! one piece per rank plus a manifest. Each leaf becomes a hexahedral cell
//! carrying its refinement level, owner rank, and Morton key.
//!
//! ## Testing and performance
//!
//! Debug construction performs expensive distributed invariant checks; use a
//! release build for measurements, not to hide correctness failures. The crate
//! is limited to three dimensions, levels 0--16, MPI communicators, and a
//! topology-only representation. Unit tests cover serial Morton operations;
//! `examples/test_mpi_*.rs` are executable MPI integration checks. From the
//! repository root, the main checks are `cargo fmt -- --check`,
//! `cargo clippy -- -D warnings`, `RUST_MIN_STACK=8388608 cargo test`, and
//! `cargo doc --no-deps`. Build examples and run them under `mpirun` to exercise
//! distributed paths.
#![cfg_attr(feature = "strict", deny(warnings), deny(unused_crate_dependencies))]
#![warn(missing_docs)]

pub mod constants;
pub mod geometry;
pub mod morton;
pub mod octree;
pub mod tools;
pub mod types;
pub mod vtk;

pub use crate::geometry::PhysicalBox;
pub use crate::octree::{LeafLocation, LookupBatchError, LookupError, Octree, OctreeOptions};
pub use morton::MortonKey;
pub use octree::points_to_morton;
