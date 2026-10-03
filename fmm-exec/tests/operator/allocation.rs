//! No allocation: a run of every operator on the largest leaf leaves the operator's
//! buffers as they were built. The `Workspace` and the table scratch are fixed-length
//! buffers that the operators only borrow slices of; the P2P mapping buffer is the one
//! that a careless implementation would grow, and its capacity is checked before and
//! after, with every P2P kernel (the kernels of `nd-fmm-simd` have no scratch).

use nd_octree::morton;

use crate::common::{
    Kind, STRATEGIES, SplitMix64, dyadic_domain, grid_points, len, operator, p2p_choices,
    random_coefficients, source_chunk, target_chunk,
};

#[test]
fn a_run_of_every_operator_on_the_largest_leaf_does_not_reallocate() {
    // Error measure: exact equality of the capacities.
    let domain = dyadic_domain();
    let mut rng = SplitMix64::new(0x7851);
    let max = 300;
    let choices = p2p_choices("allocation");
    for (strategy, &choice) in STRATEGIES
        .iter()
        .flat_map(|&s| choices.iter().map(move |c| (s, c)))
    {
        let p = 6;
        let mut op = operator(strategy, p, max).with_p2p(choice).unwrap();
        let capacity = op.scratch_capacity();
        assert_eq!(capacity, max);

        let target = morton::from_index_and_level([5, 6, 7], 4);
        let parent = morton::parent(target).unwrap();
        let near = morton::from_index_and_level([6, 6, 7], 4);
        let far = morton::from_index_and_level([5, 6, 4], 4);
        let x_source = morton::from_index_and_level([2, 3, 2], 3);
        let w_source = morton::from_index_and_level([13, 12, 14], 5);
        let (_, u_s) = grid_points(near, &domain, max, &mut rng);
        let (_, u_t) = grid_points(target, &domain, max, &mut rng);
        let sources = source_chunk(&u_s, &rng.charges(max));
        let own = source_chunk(&u_t, &rng.charges(max));
        let targets = target_chunk(&u_t);
        let mut output = vec![0.0; 4 * max];
        let multipole = random_coefficients(Kind::Multipole, p, &mut rng);
        let mut expansion = vec![0.0; len(p)];

        op.p2m_leaf(&own, &mut expansion);
        op.m2m_pair(target, parent, &multipole, &mut expansion);
        op.m2l_pair(far, target, &multipole, &mut expansion);
        op.p2l_pair(x_source, target, &sources, &mut expansion);
        op.l2l_pair(parent, target, &multipole, &mut expansion);
        op.l2p_leaf(&multipole, &targets, &mut output);
        op.m2p_pair(w_source, target, &multipole, &targets, &mut output);
        op.p2p_pair(target, target, &own, &targets, &mut output);
        op.p2p_pair(near, target, &sources, &targets, &mut output);

        assert!(output.iter().chain(&expansion).all(|v| v.is_finite()));
        assert_eq!(op.scratch_capacity(), capacity, "{strategy:?}, {choice}");
    }
    eprintln!("every operator on a leaf of 300 points: scratch capacity unchanged");
}
