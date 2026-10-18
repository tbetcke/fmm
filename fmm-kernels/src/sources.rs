//! The strengths a source store holds (Phase 6 T8d, C6.4; CONVENTIONS §3.7, "Dipole
//! sources", and §3.13, "Source chunks"), shared by the P2P kernels and the leaf operators
//! P2M and P2L.

use std::fmt;

/// The strengths of every point of a source store, fixed for an FMM (`nd-fmm-exec`'s
/// `SourceType`). A leaf of n points holds, after its n leaf-scaled coordinate triples,
/// the m n charges point-major ([`Charges`](Self::Charges), m charge vectors), the n moments
/// d̂ as triples point-major ([`Dipoles`](Self::Dipoles)), or the n charges and then the n
/// moments ([`ChargesAndDipoles`](Self::ChargesAndDipoles)); dipoles carry one set of
/// strengths (m = 1). CONVENTIONS §3.13, "Source chunks".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// Point charges: 3 + m values per point.
    #[default]
    Charges,
    /// Point dipoles: 6 values per point, the moment in units of the source leaf.
    Dipoles,
    /// A charge and a dipole at every point: 7 values per point.
    ChargesAndDipoles,
}

impl SourceKind {
    /// Whether the points carry charges.
    pub fn has_charges(self) -> bool {
        self != Self::Dipoles
    }

    /// Whether the points carry dipole moments.
    pub fn has_dipoles(self) -> bool {
        self != Self::Charges
    }

    /// The values per point of a source store with m = `vectors` charge vectors: 3 + m,
    /// 6 or 7.
    pub fn point_size(self, vectors: usize) -> usize {
        3 + if self.has_charges() { vectors } else { 0 } + if self.has_dipoles() { 3 } else { 0 }
    }
}

impl fmt::Display for SourceKind {
    /// `charges`, `dipoles` or `charges+dipoles`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Charges => "charges",
            Self::Dipoles => "dipoles",
            Self::ChargesAndDipoles => "charges+dipoles",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_sizes_follow_the_source_chunks() {
        assert_eq!(SourceKind::Charges.point_size(1), 4);
        assert_eq!(SourceKind::Charges.point_size(3), 6);
        assert_eq!(SourceKind::Dipoles.point_size(1), 6);
        assert_eq!(SourceKind::ChargesAndDipoles.point_size(1), 7);
        assert!(SourceKind::ChargesAndDipoles.has_charges());
        assert!(!SourceKind::Dipoles.has_charges());
        assert!(!SourceKind::Charges.has_dipoles());
        assert_eq!(SourceKind::default(), SourceKind::Charges);
        assert_eq!(SourceKind::ChargesAndDipoles.to_string(), "charges+dipoles");
    }
}
