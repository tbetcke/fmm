//! ASCII VTK export of octree leaves.
//!
//! This module turns Morton keys into VTK unstructured-grid geometry so that a
//! tree can be inspected in a viewer such as ParaView. It writes the XML
//! formats directly and pulls in no additional dependencies.
//!
//! Each leaf becomes one `VTK_HEXAHEDRON` cell with its own eight vertices, in
//! corner order as returned by [`PhysicalBox::corners`]. Cells carry three data
//! arrays: `refinement_level`, `owner_rank`, and `morton_key`.
//!
//! [`write_vtu`] writes one serial piece to any [`Write`] sink.
//! [`write_pvtu`] is the collective entry point: every rank writes its own
//! piece and rank zero publishes the `.pvtu` manifest that references them.
//!
//! ```
//! use nd_octree::{PhysicalBox, morton, vtk};
//!
//! let domain = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
//! let mut xml = Vec::new();
//! vtk::write_vtu(&mut xml, &[morton::root()], &domain, 0).unwrap();
//! assert!(String::from_utf8(xml).unwrap().contains("NumberOfCells=\"1\""));
//! ```

use mpi::{collective::SystemOperation, traits::CommunicatorCollectives};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::Path,
};

use crate::{MortonKey, constants::DEEPEST_LEVEL, geometry::PhysicalBox, morton};

const FIELDS: [(&str, &str); 3] = [
    ("UInt32", "refinement_level"),
    ("Int32", "owner_rank"),
    ("UInt64", "morton_key"),
];

/// Write leaves as a serial VTK unstructured grid (`.vtu`).
///
/// Only the keys passed in are written; in a distributed setting a rank
/// normally passes its own [`Octree::leaf_keys`](crate::Octree::leaf_keys) and
/// its rank number, so that ghosts are not duplicated across pieces. Vertices
/// are emitted per cell, in `VTK_HEXAHEDRON` corner order, so cells are not
/// merged at shared corners.
///
/// # Parameters
///
/// - `out`: Sink for the XML document.
/// - `leaves`: Keys to write, one hexahedral cell each. May be empty.
/// - `domain`: Physical box the keys are interpreted in, usually the bounding
///   box the tree was built with.
/// - `owner`: Value written to the `owner_rank` cell array.
///
/// # Returns
///
/// `Ok(())` once the document has been written, or an error from `out`.
/// [`io::ErrorKind::InvalidInput`] is returned without writing anything if a
/// key is invalid or deeper than [`DEEPEST_LEVEL`], or if the domain is not a
/// finite, non-degenerate box.
///
/// # Examples
///
/// ```
/// use nd_octree::{PhysicalBox, morton, vtk};
///
/// let domain = PhysicalBox::new([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
/// let leaves = [morton::from_index_and_level([0, 0, 0], 1)];
/// let mut xml = Vec::new();
/// vtk::write_vtu(&mut xml, &leaves, &domain, 3).unwrap();
/// let xml = String::from_utf8(xml).unwrap();
/// assert!(xml.contains("NumberOfPoints=\"8\" NumberOfCells=\"1\""));
///
/// // An invalid key is rejected.
/// assert!(vtk::write_vtu(Vec::new(), &[morton::invalid_key()], &domain, 0).is_err());
/// ```
pub fn write_vtu(
    mut out: impl Write,
    leaves: &[MortonKey],
    domain: &PhysicalBox,
    owner: i32,
) -> io::Result<()> {
    let c = domain.coordinates();
    if leaves
        .iter()
        .any(|&k| !morton::is_valid(k) || morton::level(k) > DEEPEST_LEVEL as usize)
        || !c.iter().all(|v| v.is_finite())
        || (0..3).any(|i| c[i] >= c[i + 3] || !(c[i + 3] - c[i]).is_finite())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid key or domain",
        ));
    }
    writeln!(
        out,
        "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n<UnstructuredGrid><Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
        leaves.len() * 8,
        leaves.len()
    )?;
    writeln!(
        out,
        "<Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">"
    )?;
    for &key in leaves {
        for p in morton::physical_box(key, domain).corners() {
            writeln!(out, "{} {} {}", p[0], p[1], p[2])?;
        }
    }
    writeln!(out, "</DataArray></Points><Cells>")?;
    for (ty, name) in [
        ("Int64", "connectivity"),
        ("Int64", "offsets"),
        ("UInt8", "types"),
    ] {
        writeln!(
            out,
            "<DataArray type=\"{ty}\" Name=\"{name}\" format=\"ascii\">"
        )?;
        for i in 0..leaves.len() {
            match name {
                "connectivity" => {
                    for j in 0..8 {
                        write!(out, "{} ", i * 8 + j)?;
                    }
                }
                "offsets" => write!(out, "{} ", (i + 1) * 8)?,
                _ => write!(out, "12 ")?,
            }
        }
        writeln!(out, "\n</DataArray>")?;
    }
    writeln!(out, "</Cells><CellData>")?;
    for (ty, name) in FIELDS {
        writeln!(
            out,
            "<DataArray type=\"{ty}\" Name=\"{name}\" format=\"ascii\">"
        )?;
        for &key in leaves {
            match name {
                "refinement_level" => write!(out, "{} ", morton::level(key))?,
                "owner_rank" => write!(out, "{owner} ")?,
                _ => write!(out, "{key} ")?,
            }
        }
        writeln!(out, "\n</DataArray>")?;
    }
    writeln!(out, "</CellData></Piece></UnstructuredGrid></VTKFile>")
}

/// Escape the XML attribute syntax characters in a piece filename.
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Write the parallel manifest referencing one piece per rank.
fn manifest(mut out: impl Write, stem: &str, size: i32) -> io::Result<()> {
    writeln!(
        out,
        "<?xml version=\"1.0\"?>\n<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\"><PUnstructuredGrid GhostLevel=\"0\"><PPoints><PDataArray type=\"Float64\" NumberOfComponents=\"3\"/></PPoints><PCellData>"
    )?;
    for (ty, name) in FIELDS {
        writeln!(out, "<PDataArray type=\"{ty}\" Name=\"{name}\"/>")?;
    }
    writeln!(out, "</PCellData>")?;
    for rank in 0..size {
        writeln!(out, "<Piece Source=\"{}.rank{rank}.vtu\"/>", escape(stem))?;
    }
    writeln!(out, "</PUnstructuredGrid></VTKFile>")
}

/// Turn a local result into the same result on every rank.
///
/// All ranks must reach this call in the same order; the reduction is what
/// keeps a failing rank from being abandoned mid-export.
fn agree<C: CommunicatorCollectives>(comm: &C, result: io::Result<()>) -> io::Result<()> {
    let mut failed = 0;
    comm.all_reduce_into(
        &i32::from(result.is_err()),
        &mut failed,
        SystemOperation::max(),
    );
    if failed == 0 {
        Ok(())
    } else {
        result.and(Err(io::Error::other("VTK export failed on another rank")))
    }
}

/// Write a distributed tree as a parallel VTK unstructured grid (`.pvtu`).
///
/// Each rank writes `<stem>.rank<n>.vtu` with its own leaves, and rank zero
/// publishes `<stem>.pvtu` referencing every piece. The manifest is written to
/// a temporary file and renamed, and any stale manifest is removed before the
/// first piece is touched, so a failed export never leaves a manifest that
/// points at a mix of old and new pieces. Pieces themselves may survive a
/// failure.
///
/// `directory` must already exist and be visible under the same path on every
/// rank; `directory` and `stem` must be identical on all ranks. Do not run
/// concurrent exports with the same stem.
///
/// # Collective operation
///
/// This is a collective call. Every rank of `comm` must enter it, in the same
/// order relative to other collectives, including ranks with no leaves. Any
/// error — invalid input, a failure on a single rank, or an I/O failure during
/// publication — is returned on all ranks.
///
/// # Parameters
///
/// - `comm`: Communicator whose ranks jointly write the data set.
/// - `directory`: Existing, shared output directory.
/// - `stem`: Base filename, a single path component: not empty, not `.` or
///   `..`, and free of path separators and control characters.
/// - `leaves`: Keys owned by this rank, usually
///   [`Octree::leaf_keys`](crate::Octree::leaf_keys). May be empty.
/// - `domain`: Physical box the keys are interpreted in, identical on all ranks.
///
/// # Returns
///
/// `Ok(())` on every rank once all pieces and the manifest are written, and an
/// error on every rank otherwise.
///
/// # Examples
///
/// ```no_run
/// use mpi::traits::Communicator;
/// use nd_octree::{Octree, PhysicalBox, morton, octree::OctreeOptions, vtk};
/// use std::path::Path;
///
/// let universe = mpi::initialize().expect("MPI must be initialized once");
/// let world = universe.world();
///
/// let keys: Vec<_> = (0..64)
///     .map(|i| morton::from_index_and_level([i, 2 * i, 3 * i], 16))
///     .collect();
/// let options = OctreeOptions::new().with_max_level(12).with_max_fine_keys(4);
/// let tree = Octree::new(&keys, options, &world);
/// let domain = PhysicalBox::new([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
///
/// // Collective: every rank calls this, even one that owns no leaves.
/// vtk::write_pvtu(&world, Path::new("."), "tree", tree.leaf_keys(), &domain)
///     .expect("VTK export succeeds on all ranks");
/// ```
pub fn write_pvtu<C: CommunicatorCollectives>(
    comm: &C,
    directory: &Path,
    stem: &str,
    leaves: &[MortonKey],
    domain: &PhysicalBox,
) -> io::Result<()> {
    let valid = !stem.is_empty()
        && stem != "."
        && stem != ".."
        && !stem.contains(['/', '\\'])
        && !stem
            .chars()
            .any(|c| c.is_control() || matches!(c, '\u{fffe}' | '\u{ffff}'));
    agree(
        comm,
        if valid {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "stem must be a filename component",
            ))
        },
    )?;
    let path = directory.join(format!("{stem}.pvtu"));
    // Remove stale publication before touching any piece from an earlier export.
    agree(
        comm,
        if comm.rank() == 0 {
            match fs::remove_file(&path) {
                Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
                r => r,
            }
        } else {
            Ok(())
        },
    )?;
    let piece = (|| {
        let mut out = BufWriter::new(File::create(
            directory.join(format!("{stem}.rank{}.vtu", comm.rank())),
        )?);
        write_vtu(&mut out, leaves, domain, comm.rank())?;
        out.flush()
    })();
    agree(comm, piece)?;
    let publication = if comm.rank() == 0 {
        (|| {
            let temp = directory.join(format!("{stem}.pvtu.tmp"));
            let mut out = BufWriter::new(File::create(&temp)?);
            manifest(&mut out, stem, comm.size())?;
            out.flush()?;
            drop(out);
            fs::rename(temp, path)
        })()
    } else {
        Ok(())
    };
    agree(comm, publication)
}

#[cfg(test)]
mod test {
    use super::{FIELDS, escape, manifest};

    #[test]
    fn test_manifest_lists_one_piece_per_rank() {
        let mut bytes = Vec::new();
        manifest(&mut bytes, "tree", 3).unwrap();
        let xml = String::from_utf8(bytes).unwrap();

        assert!(xml.contains("type=\"PUnstructuredGrid\""));
        assert_eq!(xml.matches("<Piece Source=").count(), 3);
        for rank in 0..3 {
            assert!(xml.contains(&format!("<Piece Source=\"tree.rank{rank}.vtu\"/>")));
        }
        // The manifest must announce exactly the cell arrays the pieces carry.
        for (ty, name) in FIELDS {
            assert!(xml.contains(&format!("<PDataArray type=\"{ty}\" Name=\"{name}\"/>")));
        }
        assert_eq!(xml.matches("<PDataArray").count(), 1 + FIELDS.len());

        // A communicator of size zero cannot occur, but must not produce a piece.
        let mut bytes = Vec::new();
        manifest(&mut bytes, "tree", 0).unwrap();
        assert!(!String::from_utf8(bytes).unwrap().contains("<Piece"));
    }

    #[test]
    fn test_stem_is_escaped_in_the_manifest() {
        // `write_pvtu` accepts these characters in a stem, so the manifest has to
        // escape them rather than emit broken XML.
        assert_eq!(escape("a&b<c>\"d\'e"), "a&amp;b&lt;c&gt;&quot;d&apos;e");
        // Ampersands introduced by the escaping itself are not escaped again.
        assert_eq!(escape("&amp;"), "&amp;amp;");

        let mut bytes = Vec::new();
        manifest(&mut bytes, "a & 'b'", 1).unwrap();
        let xml = String::from_utf8(bytes).unwrap();
        assert!(xml.contains("<Piece Source=\"a &amp; &apos;b&apos;.rank0.vtu\"/>"));
    }
}
