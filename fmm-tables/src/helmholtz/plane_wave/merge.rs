//! Merge and shift of the Helmholtz exponential M2L (Helmholtz CH6.1d; CONVENTIONS §3.14,
//! "Merge and shift", with the conversions and shifts of §3.15, "Exponential M2L"): per
//! level the [`PlaneWaveLevel`] of a quadrature and the shift factors of the merge, and the
//! choice of a level's quadrature from its degree.
//!
//! # The shifts
//!
//! A V pair from slot a (one of the 216 children of a parent's colleagues) to child b of
//! the parent has the offset b + 2 − a box widths, translated along its direction
//! ([`MergeGeometry`], §3.14). Merge and shift factors its shift as slot → parent, then
//! parent → child (shifts compose: T(d₁) T(d₂) = T(d₁ + d₂)):
//!
//! - the **source shift** of a slot, d = (5 − 2a)/2 per axis, rotated into the direction's
//!   frame; the slots of the six directions share one factor table per distinct (rotated
//!   d, up or down), as Laplace's tables do ([`source_shift`](MergeLevel::source_shift));
//! - the **child shift** of a direction and child, d = (2b − 1)/2 per axis, rotated
//!   ([`child_shift`](MergeLevel::child_shift)).
//!
//! Each table is [`PlaneWaveLevel::shift_factors`]' 2W reals, formed in f64 and rounded to
//! T. The product of a slot's and a child's shift is the pair's to rounding
//! (`tests/helmholtz/plane_wave_merge.rs`).
//!
//! # The quadrature of a level
//!
//! Measured in the FMM (Helmholtz CH6.1e, `fmm-validate/results/helmholtz-ch6-1e.md`): on
//! CH3.1c's cases at the calibrated degrees, each level with a quadrature forced to each
//! precision alone, the FMM's relative ℓ2 errors of φ and ∇φ against the direct sum over
//! Rotation's. At the level's own degree the precision a level needs follows the FMM's
//! tolerance ε, not its degree or κ_l: iprec 2 stays within 1.05× Rotation's on every level
//! and case for 10⁻⁶ ≤ ε ≤ 10⁻³ ([`PLANE_WAVE_MIN_TOLERANCE`]) and misses it from ε = 10⁻⁷
//! (up to 5.4× there, 77× at 10⁻⁸); iprec 1 misses it from ε = 10⁻³ (up to 1.15× on one
//! level alone, 3.6× at 10⁻⁴); iprec 3 holds on most levels at 10⁻⁷ and 10⁻⁸ but not on
//! the deep levels of the clustered trees (κ_l ≤ 0.08: 1.12× at 10⁻⁷, 1.76× at 10⁻⁸ on one
//! level alone), where its degree-p waves err on the truncated multipole's tail (CH6.1c's
//! degree study).
//!
//! Below 10⁻⁶ the level's degree is raised (Helmholtz CH6.1f, the resolving degree,
//! `fmm-validate/results/helmholtz-ch6-1f.md` §3; signed off 2026-11-09): on the pair chain
//! FMM3D's tables fail only on the plateau's degrees (p = 20 at 10⁻⁷, 24 at 10⁻⁸), whose
//! multipoles' tails their waves do not resolve, and hold from a raised degree. iprec 2's
//! own error, though (its stated error reaches 10⁻⁷), lies above Rotation's on the FMM's
//! trees at 10⁻⁷ whatever the degree (Helmholtz CH6.1g: 1.25–6.6× on the tests' trees at
//! degrees 24 and 32, where the pair chain's worst-case charges hid it), so the band below
//! 10⁻⁶ takes iprec 3 (signed off 2026-11-11: "iprec 3 at max(p,28)"), at a degree that
//! resolves its tail on the pair chain (p + 8: 0.02–0.93× RCR's error at 10⁻⁷, 0.01–0.17× at
//! 10⁻⁸). A level with a quadrature runs at p_w = max(p_l, d) ([`plane_wave_degree`], d the
//! [`resolving_degree`]):
//!
//! | ε | precision | d |
//! | --- | --- | ---: |
//! | ε ≥ [`PLANE_WAVE_MIN_TOLERANCE`] (10⁻⁶) | iprec 2 | – (p_l) |
//! | [`RESOLVING_STEP_TOLERANCE`] (10⁻⁷) ≤ ε < 10⁻⁶ | iprec 3 | [`RESOLVING_DEGREE`] (28) |
//! | [`RESOLVING_MIN_TOLERANCE`] (10⁻⁸) ≤ ε < 10⁻⁷ | iprec 3 | [`RESOLVING_DEGREE_TIGHT`] (32) |
//! | ε < 10⁻⁸ | none (Rotation) | – |
//!
//! [`precision_for_tolerance`] gives the precision, [`resolving_degree`] d; the FMM raises
//! the level's degree itself (`nd_fmm_exec::helmholtz::HelmholtzOperator`), so every
//! operator of the level runs at p_w. In the FMM (Helmholtz CH6.1g,
//! `fmm-validate/results/helmholtz-ch6-1g.md`) the rule keeps CH6.1e's cases at 10⁻⁷ and
//! 10⁻⁸ within 1.1× Rotation's error at the level's own degree.
//!
//! An FMM with named degrees has no tolerance. [`precision_for_degree`] then takes the
//! tolerance a degree p meets on the low-frequency plateau of the calibrated degree rule
//! (`nd_fmm_exec::helmholtz::DegreeRule::CALIBRATED`, ⌈0.30 L² + 0.25 L + 2.60⌉ with
//! L = log₁₀(1/ε)): iprec 2 to [`MAX_PLANE_WAVE_DEGREE`], the plateau at
//! [`PLANE_WAVE_MIN_TOLERANCE`], none above, and never a raised degree. A high-frequency
//! level has a larger degree at the same ε, so it may get none where the tolerance would
//! give it iprec 2.
//!
//! [`select`] applies either on every level with a V list whose κ_l = k w_l lies in the
//! tables. Both are functions of k, the level's width, its degree and ε only (proposal §2,
//! constraint 9), so every rank chooses alike.

use nd_fmm_math::{Complex, RealScalar};

use super::pair::PlaneWaveLevel;
use super::quadrature::{Precision, Quadrature, RHO_MAX, assert_reach, domain_of};
use crate::plane_wave::geometry::{self, CHILDREN, Direction, MergeGeometry, SLOTS};

/// The smallest tolerance at which a level with a quadrature runs at its own degree
/// (Helmholtz CH6.1e; the [module documentation](self)); below it [`resolving_degree`]
/// raises the degree. D8's bound on `HelmholtzM2l::Auto`'s exponential M2L.
pub const PLANE_WAVE_MIN_TOLERANCE: f64 = 1e-6;

/// The tolerance below which the [`resolving_degree`] is [`RESOLVING_DEGREE_TIGHT`]
/// (Helmholtz CH6.1g).
pub const RESOLVING_STEP_TOLERANCE: f64 = 1e-7;

/// The smallest tolerance at which [`precision_for_tolerance`] gives a precision (Helmholtz
/// CH6.1f); Rotation below.
pub const RESOLVING_MIN_TOLERANCE: f64 = 1e-8;

/// The resolving degree for [`RESOLVING_STEP_TOLERANCE`] ≤ ε < [`PLANE_WAVE_MIN_TOLERANCE`]
/// (Helmholtz CH6.1g; signed off 2026-11-11).
pub const RESOLVING_DEGREE: usize = 28;

/// The resolving degree for [`RESOLVING_MIN_TOLERANCE`] ≤ ε < [`RESOLVING_STEP_TOLERANCE`]
/// (Helmholtz CH6.1f; signed off 2026-11-09).
pub const RESOLVING_DEGREE_TIGHT: usize = 32;

/// The largest degree [`precision_for_degree`] gives a precision: the calibrated plateau
/// at [`PLANE_WAVE_MIN_TOLERANCE`].
pub const MAX_PLANE_WAVE_DEGREE: usize = 15;

/// The precision the rule takes at ε ≥ [`PLANE_WAVE_MIN_TOLERANCE`], iprec 2 (the [module
/// documentation](self)); iprec 3 below.
pub const RULE_PRECISION: Precision = Precision::Iprec2;

/// The precision of the levels of an FMM at tolerance ε (the [module
/// documentation](self)): [`RULE_PRECISION`] for ε ≥ [`PLANE_WAVE_MIN_TOLERANCE`], iprec 3
/// for [`RESOLVING_MIN_TOLERANCE`] ≤ ε < [`PLANE_WAVE_MIN_TOLERANCE`], `None` below
/// (Rotation there) and for a ε that is not positive and finite. Below
/// [`PLANE_WAVE_MIN_TOLERANCE`] the precision holds at the [`resolving_degree`] only.
pub fn precision_for_tolerance(epsilon: f64) -> Option<Precision> {
    if !epsilon.is_finite() || epsilon < RESOLVING_MIN_TOLERANCE {
        None
    } else if epsilon < PLANE_WAVE_MIN_TOLERANCE {
        Some(Precision::Iprec3)
    } else {
        Some(RULE_PRECISION)
    }
}

/// The resolving degree d of the levels with a quadrature at tolerance ε (Helmholtz CH6.1f
/// and CH6.1g; the [module documentation](self)): 0 (the level's own degree) for
/// ε ≥ [`PLANE_WAVE_MIN_TOLERANCE`], [`RESOLVING_DEGREE`] for
/// [`RESOLVING_STEP_TOLERANCE`] ≤ ε < [`PLANE_WAVE_MIN_TOLERANCE`],
/// [`RESOLVING_DEGREE_TIGHT`] for [`RESOLVING_MIN_TOLERANCE`] ≤ ε <
/// [`RESOLVING_STEP_TOLERANCE`], and 0 where [`precision_for_tolerance`] gives none.
pub fn resolving_degree(epsilon: f64) -> usize {
    match precision_for_tolerance(epsilon) {
        None => 0,
        Some(_) if epsilon >= PLANE_WAVE_MIN_TOLERANCE => 0,
        Some(_) if epsilon >= RESOLVING_STEP_TOLERANCE => RESOLVING_DEGREE,
        Some(_) => RESOLVING_DEGREE_TIGHT,
    }
}

/// The degree p_w = max(p, d) at which a level of degree `p` with a quadrature runs at the
/// FMM's tolerance ε, d the [`resolving_degree`] (Helmholtz CH6.1g): p itself for
/// ε ≥ [`PLANE_WAVE_MIN_TOLERANCE`], never below p.
pub fn plane_wave_degree(p: usize, epsilon: f64) -> usize {
    p.max(resolving_degree(epsilon))
}

/// The precision of a level of named degree `p`, without a tolerance (the [module
/// documentation](self)): [`RULE_PRECISION`] to [`MAX_PLANE_WAVE_DEGREE`], `None` above
/// (Rotation there).
pub fn precision_for_degree(p: usize) -> Option<Precision> {
    (p <= MAX_PLANE_WAVE_DEGREE).then_some(RULE_PRECISION)
}

/// The precision of each level of a tree with k̂₀ = k r₀ at the root and degrees p_l, at
/// the FMM's tolerance if it has one: [`precision_for_tolerance`] of `tolerance` (which
/// below [`PLANE_WAVE_MIN_TOLERANCE`] holds at [`plane_wave_degree`] only), or
/// without one [`precision_for_degree`] of p_l, on the levels l ≥ 2 whose
/// κ_l = 2 k̂₀ 2⁻ˡ lies in the tables ([`domain_of`]), `None` on levels 0 and 1 (no V list)
/// and on the others. `None` everywhere unless Re k̂₀ > 0 and Im k̂₀ ≥ 0 are finite.
pub fn select(
    k_root: Complex<f64>,
    degrees: &[usize],
    tolerance: Option<f64>,
) -> Vec<Option<Precision>> {
    let valid =
        k_root.re > 0.0 && k_root.im >= 0.0 && k_root.re.is_finite() && k_root.im.is_finite();
    degrees
        .iter()
        .enumerate()
        .map(|(l, &p)| {
            let kappa = level_k_hat(k_root, l) * 2.0;
            (valid && l >= 2 && domain_of(kappa).is_some())
                .then(|| match tolerance {
                    Some(epsilon) => precision_for_tolerance(epsilon),
                    None => precision_for_degree(p),
                })
                .flatten()
        })
        .collect()
}

/// k̂_l = k̂₀ 2⁻ˡ, as `HelmholtzTables` forms it.
fn level_k_hat(k_root: Complex<f64>, level: usize) -> Complex<f64> {
    k_root * (0.5f64).powi(level as i32)
}

/// One level's exponential M2L by merge and shift: its precision, its [`PlaneWaveLevel`]
/// and the merge's shift factors (the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct MergeLevel<T: RealScalar> {
    precision: Precision,
    tables: PlaneWaveLevel<T>,
    /// The distinct source shifts, 2W reals each.
    source_shifts: Vec<T>,
    /// Per direction and slot, the index of its source shift, or `u16::MAX`.
    source_index: Vec<[u16; SLOTS]>,
    /// Per direction and child, its child shift, 2W reals each.
    child_shifts: Vec<T>,
}

impl<T: RealScalar> MergeLevel<T> {
    /// The level with k̂ = k r_l at degree p with the quadrature of κ = 2k̂ at `precision`,
    /// or `None` where κ lies outside the tables. Builds in f64 and rounds to T; allocates.
    ///
    /// # Panics
    ///
    /// As [`PlaneWaveLevel::new`] where κ lies in the tables but Re k̂ = 0 (a κ with a
    /// negative or non-finite part has no domain, so `None`).
    pub fn new(
        k_hat: Complex<f64>,
        p: usize,
        precision: Precision,
        geometry: &MergeGeometry,
    ) -> Option<Self> {
        Self::with_reach(k_hat, p, precision, geometry, RHO_MAX)
    }

    /// The level of [`new`](Self::new) with the α counts of
    /// [`Quadrature::with_reach`] for the lateral reach `reach` (Helmholtz CH6.2e);
    /// `with_reach(…, RHO_MAX)` is `new` bit for bit.
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and as [`Quadrature::with_reach`] for a reach that is not
    /// finite or not above the source's reach.
    pub fn with_reach(
        k_hat: Complex<f64>,
        p: usize,
        precision: Precision,
        geometry: &MergeGeometry,
        reach: f64,
    ) -> Option<Self> {
        let quadrature = Quadrature::with_reach(k_hat * 2.0, precision, reach)?;
        let tables = PlaneWaveLevel::new(k_hat, p, &quadrature);
        let w = tables.wave_len();
        let mut factors = vec![T::zero(); w];
        let mut keys: Vec<([i64; 3], bool)> = Vec::new();
        let mut source_shifts = Vec::new();
        let mut source_index = vec![[u16::MAX; SLOTS]; 6];
        for direction in Direction::ALL {
            let up = direction.is_up();
            for accumulator in geometry.accumulators(direction) {
                for &slot in &accumulator.slots {
                    let a = geometry::slot_coordinates(slot as usize);
                    // In half box widths, from the slot's centre to the parent's.
                    let key = (direction.rotate(a.map(|x| 5 - 2 * x)), up);
                    let index = keys.iter().position(|&k| k == key).unwrap_or_else(|| {
                        keys.push(key);
                        tables.shift_factors(key.0.map(|x| x as f64 / 2.0), up, &mut factors);
                        source_shifts.extend_from_slice(&factors);
                        keys.len() - 1
                    });
                    source_index[direction.index()][slot as usize] =
                        u16::try_from(index).expect("fewer than 2¹⁶ shifts");
                }
            }
        }
        let mut child_shifts = Vec::with_capacity(6 * CHILDREN * w);
        for direction in Direction::ALL {
            for child in 0..CHILDREN {
                let b = geometry::child_bits(child);
                let d = direction.rotate(b.map(|x| (2 * x - 1) as f64 / 2.0));
                tables.shift_factors(d, direction.is_up(), &mut factors);
                child_shifts.extend_from_slice(&factors);
            }
        }
        Some(Self {
            precision,
            tables,
            source_shifts,
            source_index,
            child_shifts,
        })
    }

    /// The precision of its quadrature.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The level's conversions and pair translation.
    pub fn tables(&self) -> &PlaneWaveLevel<T> {
        &self.tables
    }

    /// The source shift of `slot` in `direction` (2W reals), or `None` if no pair of the
    /// slot has that direction.
    pub fn source_shift(&self, direction: Direction, slot: usize) -> Option<&[T]> {
        let index = self.source_index[direction.index()][slot];
        (index != u16::MAX).then(|| {
            let w = self.tables.wave_len();
            &self.source_shifts[index as usize * w..(index as usize + 1) * w]
        })
    }

    /// The distinct source shifts.
    pub fn source_shift_count(&self) -> usize {
        self.source_shifts.len() / self.tables.wave_len()
    }

    /// The child shift of `child` in `direction` (2W reals).
    pub fn child_shift(&self, direction: Direction, child: usize) -> &[T] {
        let w = self.tables.wave_len();
        let at = (direction.index() * CHILDREN + child) * w;
        &self.child_shifts[at..at + w]
    }

    /// The bytes of its tables and shifts.
    pub fn bytes(&self) -> usize {
        self.tables.bytes()
            + (self.source_shifts.len() + self.child_shifts.len()) * size_of::<T>()
            + self.source_index.len() * size_of::<[u16; SLOTS]>()
    }
}

/// The exponential M2L of a tree's levels: a [`MergeLevel`] on each level with a precision,
/// and the merge's geometry.
#[derive(Clone, Debug)]
pub struct PlaneWaveM2l<T: RealScalar> {
    levels: Vec<Option<MergeLevel<T>>>,
    geometry: MergeGeometry,
}

impl<T: RealScalar> PlaneWaveM2l<T> {
    /// The levels of a tree with k̂₀ = k r₀ and degrees p_l, level l with the quadrature of
    /// `precisions[l]` where it is `Some` and κ_l lies in the tables, none elsewhere
    /// ([`select`] gives the rule's precisions). Builds in f64, serially; allocates.
    ///
    /// # Panics
    ///
    /// If the slices differ in length, a precision is named on level 0 or 1, or a level
    /// with a precision and a domain has Re k̂ = 0.
    pub fn build(
        k_root: Complex<f64>,
        degrees: &[usize],
        precisions: &[Option<Precision>],
    ) -> Self {
        Self::build_with_reach(k_root, degrees, precisions, RHO_MAX)
    }

    /// The levels of [`build`](Self::build), every level's α counts for the lateral reach
    /// `reach` ([`Quadrature::with_reach`]; Helmholtz CH6.2e); `build_with_reach(…,
    /// RHO_MAX)` is `build` bit for bit.
    ///
    /// # Panics
    ///
    /// As [`build`](Self::build), and for a reach that is not finite or not above the
    /// source's reach.
    pub fn build_with_reach(
        k_root: Complex<f64>,
        degrees: &[usize],
        precisions: &[Option<Precision>],
        reach: f64,
    ) -> Self {
        assert_reach(reach);
        assert_eq!(
            precisions.len(),
            degrees.len(),
            "one precision per level: {} precisions, {} degrees",
            precisions.len(),
            degrees.len()
        );
        assert!(
            precisions.iter().take(2).all(Option::is_none),
            "no exponential M2L on levels 0 and 1: they have no V list"
        );
        let geometry = MergeGeometry::new();
        let levels = degrees
            .iter()
            .zip(precisions)
            .enumerate()
            .map(|(l, (&p, &precision))| {
                precision.and_then(|precision| {
                    MergeLevel::with_reach(level_k_hat(k_root, l), p, precision, &geometry, reach)
                })
            })
            .collect();
        Self { levels, geometry }
    }

    /// The level's merge tables, or `None` where it has no quadrature.
    pub fn level(&self, level: usize) -> Option<&MergeLevel<T>> {
        self.levels.get(level).and_then(Option::as_ref)
    }

    /// The level's precision, or `None` where it has no quadrature.
    pub fn precision(&self, level: usize) -> Option<Precision> {
        self.level(level).map(MergeLevel::precision)
    }

    /// The merge's accumulators and directions, the same on every level.
    pub fn geometry(&self) -> &MergeGeometry {
        &self.geometry
    }

    /// The levels of the tree.
    pub fn nlevels(&self) -> usize {
        self.levels.len()
    }

    /// The bytes of every level's tables and shifts.
    pub fn bytes(&self) -> usize {
        self.levels.iter().flatten().map(MergeLevel::bytes).sum()
    }
}

/// `v` ⊙= `factors` value by value, complex, on split planes of equal length: per value
/// the real part re_v re_f − im_v im_f and the imaginary part re_v im_f + im_v re_f, each
/// product and sum rounded apart.
///
/// # Panics
///
/// If the slices differ in length or have an odd length.
pub fn shift<T: RealScalar>(v: &mut [T], factors: &[T]) {
    assert!(
        v.len() == factors.len() && v.len().is_multiple_of(2),
        "wave slice lengths"
    );
    let w = v.len() / 2;
    let (re, im) = v.split_at_mut(w);
    let (f_re, f_im) = factors.split_at(w);
    for i in 0..w {
        let (a, b) = (re[i], im[i]);
        re[i] = a * f_re[i] - b * f_im[i];
        im[i] = a * f_im[i] + b * f_re[i];
    }
}

/// `v` += `a` value by value.
///
/// # Panics
///
/// If the slices differ in length.
pub fn add<T: RealScalar>(v: &mut [T], a: &[T]) {
    assert_eq!(v.len(), a.len(), "wave slice lengths");
    for (x, &y) in v.iter_mut().zip(a) {
        *x = *x + y;
    }
}
