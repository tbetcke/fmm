//! The operators of an FMM.

use mpi::traits::Equivalence;
use nd_octree::MortonKey;

/// The translation operators of an FMM.
///
/// An operator acts on the data of a single pair of boxes, identified by their
/// Morton keys, so that an implementation can derive the geometry and the
/// level of the boxes. Every operator **accumulates** into its output, that is
/// it adds its contribution to the values already there.
///
/// All data is stored as flat slices of [`Value`](Self::Value): multipoles and
/// local expansions with a size that may depend on the level, and source and
/// target data with a fixed size per leaf.
pub trait FmmOperator {
    /// The scalar type of all FMM data.
    type Value: Equivalence + Copy + Default;

    /// The number of values of a multipole expansion on `level`.
    fn multipole_size(&self, level: usize) -> usize;

    /// The number of values of a local expansion on `level`.
    fn local_size(&self, level: usize) -> usize;

    /// The number of source values of a leaf.
    fn source_size(&self) -> usize;

    /// The number of target values of a leaf.
    fn target_size(&self) -> usize;

    /// Add the multipole expansion of the sources of `leaf`.
    fn p2m(&self, leaf: MortonKey, sources: &[Self::Value], multipole: &mut [Self::Value]);

    /// Translate the multipole of `child` into the multipole of `parent`.
    fn m2m(
        &self,
        child: MortonKey,
        parent: MortonKey,
        child_multipole: &[Self::Value],
        parent_multipole: &mut [Self::Value],
    );

    /// Translate the multipole of `source` into the local expansion of `target`
    /// (V-list).
    fn m2l(
        &self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[Self::Value],
        target_local: &mut [Self::Value],
    );

    /// Add the sources of the leaf `source` to the local expansion of `target`
    /// (X-list).
    fn p2l(
        &self,
        source: MortonKey,
        target: MortonKey,
        sources: &[Self::Value],
        target_local: &mut [Self::Value],
    );

    /// Translate the local expansion of `parent` into the local expansion of
    /// `child`.
    fn l2l(
        &self,
        parent: MortonKey,
        child: MortonKey,
        parent_local: &[Self::Value],
        child_local: &mut [Self::Value],
    );

    /// Evaluate the local expansion of `leaf` at its targets.
    fn l2p(&self, leaf: MortonKey, local: &[Self::Value], targets: &mut [Self::Value]);

    /// Evaluate the multipole of `source` at the targets of the leaf `target`
    /// (W-list).
    fn m2p(
        &self,
        source: MortonKey,
        target: MortonKey,
        source_multipole: &[Self::Value],
        targets: &mut [Self::Value],
    );

    /// Evaluate the sources of the leaf `source` directly at the targets of the
    /// leaf `target` (U-list and self-interaction).
    fn p2p(
        &self,
        source: MortonKey,
        target: MortonKey,
        sources: &[Self::Value],
        targets: &mut [Self::Value],
    );
}
