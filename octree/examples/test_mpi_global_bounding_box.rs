//! Test the computation of a global bounding box across MPI ranks.

use mpi::traits::Communicator;
use nd_octree::{constants::DEEPEST_LEVEL, octree::compute_global_bounding_box};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rlst::{SliceArray, distributed_tools::array_tools::gather_to_rank, rlst_dynamic_array};

pub fn main() {
    // Initialise MPI
    let universe = mpi::initialize().unwrap();

    // Get the world communicator
    let comm = universe.world();

    // Initialise a seeded Rng.
    let mut rng = ChaCha8Rng::seed_from_u64(2);

    // Create `npoints` per rank.
    let npoints = 10;

    // Size of the communicator
    let size = comm.size() as usize;

    // Generate random points.

    let mut points = rlst_dynamic_array!(f64, [3, npoints]);
    points.fill_from_standard_normal(&mut rng);

    // Compute the distributed bounding box.

    let bounding_box = compute_global_bounding_box(&points, &comm);

    // Copy all points to root and compare local bounding box there.

    if let Some(points_root) = gather_to_rank(points.data().unwrap(), 0, &comm) {
        // Compute the bounding box on root.

        let points_root = SliceArray::from_shape(&points_root, [3, npoints * size]);
        let single_comm = mpi::topology::SimpleCommunicator::self_comm();
        let expected = compute_global_bounding_box(&points_root, &single_comm);
        assert_eq!(expected.coordinates(), bounding_box.coordinates());

        // The box is cubic, centred on the global point cloud, and pads the
        // largest extent by one cell on the deepest level. Checking this against
        // the gathered points catches a padding or centring error that the
        // comparison above would reproduce on both sides. Each bound is built
        // from its own axis midpoint, so these identities hold only up to
        // rounding, not exactly.
        let close = |left: f64, right: f64| {
            (left - right).abs() <= 8.0 * f64::EPSILON * f64::max(left.abs(), right.abs())
        };
        let coords = bounding_box.coordinates();
        let extents = [
            coords[3] - coords[0],
            coords[4] - coords[1],
            coords[5] - coords[2],
        ];
        assert!(close(extents[0], extents[1]));
        assert!(close(extents[1], extents[2]));

        let mut diameter: f64 = 0.0;
        for axis in 0..3 {
            let values = points_root
                .col_iter()
                .map(|point| point.get_value([axis]).unwrap())
                .collect::<Vec<_>>();
            let lower = values.iter().copied().reduce(f64::min).unwrap();
            let upper = values.into_iter().reduce(f64::max).unwrap();
            // Every point lies strictly inside, so no point is clamped into a
            // boundary cell by `points_to_morton`.
            assert!(coords[axis] < lower && upper < coords[3 + axis]);
            // The box is centred on the midpoint along each axis separately.
            assert!(close(coords[axis] + coords[3 + axis], lower + upper));
            diameter = f64::max(diameter, upper - lower);
        }
        let padding = 1.0 / (1u64 << DEEPEST_LEVEL) as f64;
        assert!(close(extents[0], diameter * (1.0 + padding)));
    }

    // A rank without any points must still enter the collective and must not
    // influence the result.
    let empty = rlst_dynamic_array!(f64, [3, 0]);
    let root_only = if comm.rank() == 0 {
        compute_global_bounding_box(&points, &comm)
    } else {
        compute_global_bounding_box(&empty, &comm)
    };
    let single_comm = mpi::topology::SimpleCommunicator::self_comm();
    if comm.rank() == 0 {
        assert_eq!(
            root_only.coordinates(),
            compute_global_bounding_box(&points, &single_comm).coordinates()
        );
    }
}
