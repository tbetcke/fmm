# nd-octree

`nd-octree` is a Rust library for adaptive, 2:1-balanced three-dimensional
Morton-key octrees. It supports one-rank and distributed MPI construction and
stores topology, ownership, and neighbour relationships rather than application
point data.

MPI is required, including for a one-rank application. See the [crate-level
guide](https://docs.rs/nd-octree/latest/nd_octree/) for prerequisites, point
layout and bounding-box rules, distributed construction, lookup semantics, and
testing instructions.

## Morton-key basics

```rust
use nd_octree::morton;

let key = morton::from_index_and_level([3, 2, 1], 4);
assert_eq!(morton::decode(key), (4, [3, 2, 1]));
assert!(morton::is_valid(key));
assert_eq!(morton::root(), 0); // root is valid, not a sentinel
```

For MPI construction and collective leaf lookup, see the runnable
`examples/test_mpi_leaf_lookup.rs` program and launch it with `mpirun` after
building examples.

## VTK visualization

```sh
cargo build --example test_mpi_vtk
mpirun -n 3 target/debug/examples/test_mpi_vtk target/vtk-example
```

Open `target/vtk-example/adaptive & 'tree'.pvtu` in ParaView, apply the reader,
select **Surface With Edges**, and color by `refinement_level` or `owner_rank`.
The example creates a deterministic nonuniform adaptive tree and also checks
empty output and collective failure handling. The optional argument is an output
directory shared by all ranks. Default output stays in ignored `target/`.

The writer is the public `nd_octree::vtk` module: `write_vtu` accepts a
`Write` sink, owned leaf keys, an explicit `PhysicalBox`, and owner rank;
`write_pvtu` accepts a communicator, existing shared directory, filename stem,
owned leaf keys (`tree.leaf_keys()`), and the same explicit physical domain.
Every rank must call the latter with the same directory/stem, including empty
ranks. Only owned leaves are exported, not ghosts or interior nodes.

Output is dependency-free ASCII XML. Each leaf is a VTK hexahedron with eight
independent vertices obtained from `morton::physical_box` and
`PhysicalBox::corners`. Cell arrays are
`refinement_level` (UInt32), `owner_rank` (Int32), and `morton_key` (UInt64,
written as exact decimal integers, never floating point). Root and empty pieces
are supported. No compression, vertex deduplication, renderer, time series, or
neighbour visualization is provided; ASCII output favors inspectability over
size and throughput.

Pieces use relative, XML-escaped filenames `<stem>.rank<N>.vtu`. A stem must be
one filename component. All ranks coordinate validation and I/O results; rank
zero publishes the `.pvtu` via a temporary file and rename only after every
piece has flushed successfully. A previous manifest is removed before rewriting
pieces. Errors are returned on every rank (local errors retain their details).
Failures may leave partial pieces or a temporary file for manual cleanup. Do not
export concurrently with the same stem or read an export while overwriting it.
This protocol handles returned I/O errors, not crashed processes or filesystem
failures that prevent collective MPI progress. The caller owns directory setup
and must coordinate its errors before entering the exporter.

Serial writer tests run with `cargo test`; distributed checks run explicitly with
`mpirun -n 1` and `mpirun -n 3` on `target/debug/examples/test_mpi_vtk`.
