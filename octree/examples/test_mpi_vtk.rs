//! Export a deterministic nonuniform tree and exercise collective I/O failures.
use mpi::traits::{Communicator, CommunicatorCollectives};
use nd_octree::{Octree, OctreeOptions, PhysicalBox, morton, vtk};
use rlst::distributed_tools::array_tools::gather_to_all;
use std::{fs, io, path::PathBuf};

// Agree before any rank advances, including rank-zero setup and readback errors.
fn on_root(
    comm: &impl CommunicatorCollectives,
    action: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    let result = if comm.rank() == 0 { action() } else { Ok(()) };
    let mut failed = 0;
    comm.all_reduce_into(
        &i32::from(result.is_err()),
        &mut failed,
        mpi::collective::SystemOperation::max(),
    );
    if failed == 0 {
        Ok(())
    } else {
        result.and(Err(io::Error::other("VTK example failed on another rank")))
    }
}

fn check(valid: bool) -> io::Result<()> {
    if valid {
        Ok(())
    } else {
        Err(io::Error::other("VTK readback mismatch"))
    }
}
fn main() -> io::Result<()> {
    let universe = mpi::initialize().unwrap();
    let comm = universe.world();
    let directory = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "target/vtk-example".into()),
    );
    on_root(&comm, || fs::create_dir_all(&directory))?;
    let mut keys = Vec::new();
    for i in 0..96 {
        keys.push(morton::from_index_and_level(
            [1000 + i % 4, 2000 + (i / 4) % 4, 3000 + (i / 16) % 4],
            16,
        ));
    }
    for i in 0..16 {
        let x = comm.rank() as usize * 4096 + i * 197;
        keys.push(morton::from_index_and_level(
            [x % 65536, x * 3 % 65536, x * 7 % 65536],
            16,
        ));
    }
    let options = OctreeOptions::new()
        .with_max_level(12)
        .with_max_fine_keys(4);
    let tree = Octree::new(&keys, options, &comm);
    let domain = PhysicalBox::new([-2., 3., 4., 6., 7., 16.]);
    let all = gather_to_all(tree.leaf_keys(), &comm);
    assert!(
        all.iter()
            .any(|&k| morton::level(k) != morton::level(all[0]))
    );
    let stem = "adaptive & 'tree'";
    vtk::write_pvtu(&comm, &directory, stem, tree.leaf_keys(), &domain).unwrap();
    let counts = gather_to_all(&[tree.leaf_keys().len()], &comm);
    on_root(&comm, || {
        let manifest = fs::read_to_string(directory.join(format!("{stem}.pvtu")))?;
        check(manifest.matches("<Piece Source=").count() == comm.size() as usize)?;
        check(manifest.contains("adaptive &amp; &apos;tree&apos;"))?;
        for rank in 0..comm.size() {
            let xml = fs::read_to_string(directory.join(format!("{stem}.rank{rank}.vtu")))?;
            check(xml.contains(&format!("NumberOfCells=\"{}\"", counts[rank as usize])))?;
        }
        check(counts.iter().sum::<usize>() == all.len())
    })?;
    vtk::write_pvtu(&comm, &directory, "empty", &[], &domain).unwrap();
    let root = if comm.rank() == 0 {
        vec![morton::root()]
    } else {
        vec![]
    };
    vtk::write_pvtu(&comm, &directory, "root", &root, &domain).unwrap();
    // A directory at exactly one piece path forces an error on just that rank.
    let blocked = directory.join("failure.rank0.vtu");
    on_root(&comm, || {
        fs::create_dir_all(&blocked)?;
        fs::write(directory.join("failure.pvtu"), "stale manifest")
    })?;
    assert!(vtk::write_pvtu(&comm, &directory, "failure", tree.leaf_keys(), &domain).is_err());
    on_root(&comm, || {
        check(!directory.join("failure.pvtu").try_exists()?)
    })?;
    // Publication failure must also be returned on all ranks.
    on_root(&comm, || {
        fs::create_dir_all(directory.join("publication.pvtu.tmp"))
    })?;
    assert!(vtk::write_pvtu(&comm, &directory, "publication", &[], &domain).is_err());
    on_root(&comm, || {
        check(!directory.join("publication.pvtu").try_exists()?)
    })?;
    assert!(vtk::write_pvtu(&comm, &directory, "../bad", &[], &domain).is_err());
    for stem in ["bad\u{fffe}", "bad\u{ffff}"] {
        assert!(vtk::write_pvtu(&comm, &directory, stem, &[], &domain).is_err());
    }
    // Exercise the same collective directory setup used for the CLI argument.
    let file_directory = directory.join("not-a-directory");
    on_root(&comm, || fs::write(&file_directory, "file"))?;
    assert!(on_root(&comm, || fs::create_dir_all(&file_directory)).is_err());
    if comm.rank() == 0 {
        println!(
            "VTK geometry exported to {}; collective checks passed",
            directory.display()
        );
    }
    Ok(())
}
