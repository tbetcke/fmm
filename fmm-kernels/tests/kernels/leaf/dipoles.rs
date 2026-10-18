//! The device P2M and P2L with dipole sources (Phase 6 T8d, C6.4) on every layout the
//! backend runs, in f32 and, where the device supports it, f64, at p ∈ {0, 3, 8} (p = 20 in
//! f64 in the ignored sweep), for `SourceKind::Dipoles` and `ChargesAndDipoles`.
//!
//! Error measure (the coefficient measure of the charge tests): per degree in the
//! weighting of CONVENTIONS §3.8, against `nd_fmm_ref::leaf::{p2m_dipoles, p2l_dipoles}`
//! (after `p2m` or `p2l` with charges, per leaf) in f64 at the relative frames of
//! CONVENTIONS §3.13 on the inputs as rounded to T, relative to the term magnitudes
//! τᵢ = Σⱼ |qⱼ Xᵢ(vⱼ)| + Σⱼ Σₖ |d̂ⱼ,ₖ ∂ₖXᵢ(vⱼ)| (d̂ in the frame's units); within 1e-13
//! (f64) or 1e-5 (f32). Bits: repeated launches, a row split per entry, and the order (a
//! launch with charges and dipoles equals, entry by entry, the charges' launch followed by
//! the moments').

use nd_fmm_kernels::leaf::{LeafLayout, SourceInputs, coefficients, p2l, p2m};
use nd_fmm_kernels::{Device, Precision, SourceKind};
use nd_fmm_ref::{Frame, Workspace, leaf as reference};

use super::{
    Basis, DEGREES, LEVELS, Real, SourceCase, SourcesOnDevice, Worst, charge, degree_error,
    download, host_harmonics, layouts, p2l_case, p2m_case, t, tolerance, w,
};
use crate::common::{Rng, tests_on};

/// The two source kinds with dipoles.
const KINDS: [SourceKind; 2] = [SourceKind::Dipoles, SourceKind::ChargesAndDipoles];

/// A source case with a moment uniform in [−1, 1)³ at every point.
struct DipoleCase<T> {
    case: SourceCase<T>,
    moments: Vec<Vec<[T; 3]>>,
}

impl<T: Real> DipoleCase<T> {
    fn new(rng: &mut Rng, case: SourceCase<T>) -> Self {
        let moments = case
            .sources
            .iter()
            .map(|s| {
                s.iter()
                    .map(|_| [0; 3].map(|_| t::<T>(charge(rng))))
                    .collect()
            })
            .collect();
        Self { case, moments }
    }

    /// Uploads the case with `rows` as its view, `out` as the coefficients and the source
    /// store of `kind`.
    fn upload(
        &self,
        device: &mut Device,
        rows: &[Vec<u32>],
        out: &[T],
        kind: SourceKind,
    ) -> SourcesOnDevice<T> {
        let mut on = self.case.upload(device, rows, out);
        let mut store = Vec::new();
        for ((u, q), d) in self
            .case
            .sources
            .iter()
            .zip(&self.case.charges)
            .zip(&self.moments)
        {
            store.extend(u.iter().flatten());
            if kind.has_charges() {
                store.extend(q);
            }
            if kind.has_dipoles() {
                store.extend(d.iter().flatten());
            }
        }
        on.sources = device.upload(&store).unwrap();
        on
    }

    /// Runs P2M or P2L (`irregular`) of `kind` from zero coefficients.
    fn run(
        &self,
        device: &mut Device,
        (layout, p, irregular): (LeafLayout, usize, bool),
        kind: SourceKind,
    ) -> Vec<T> {
        let zeros = vec![t::<T>(0.0); coefficients(p) * self.case.boxes.len()];
        let mut on = self.upload(device, &self.case.rows, &zeros, kind);
        launch(device, &mut on, (layout, p, irregular), kind);
        download(device, &on.out)
    }

    /// The reference coefficients of box t in f64 and in T, and their terms: per entry
    /// leaf its charges (with charges), then its moments.
    fn reference(
        &self,
        (p, irregular, kind): (usize, bool, SourceKind),
        t_box: usize,
    ) -> [Vec<f64>; 3] {
        let n = coefficients(p);
        let mut want = vec![0.0f64; n];
        let mut in_t = vec![t::<T>(0.0); n];
        let mut terms = vec![0.0f64; n];
        let mut ws64 = Workspace::<f64>::new(p);
        let mut ws = Workspace::<T>::new(p);
        let basis = if irregular {
            Basis::Irregular
        } else {
            Basis::Regular
        };
        for &j in &self.case.rows[t_box] {
            let j = j as usize;
            let frame = self.case.frame_of(t_box, j, irregular);
            let frame_t = Frame::new(frame.centre.map(t::<T>), t::<T>(frame.radius));
            let sources = &self.case.sources[j];
            let charges = &self.case.charges[j];
            let moments = &self.moments[j];
            let u64s: Vec<[f64; 3]> = sources.iter().map(|u| u.map(w::<T>)).collect();
            if kind.has_charges() {
                let q64: Vec<f64> = charges.iter().map(|&q| w(q)).collect();
                if irregular {
                    reference::p2l(p, &frame, &u64s, &q64, &mut ws64, &mut want);
                    reference::p2l(p, &frame_t, sources, charges, &mut ws, &mut in_t);
                } else {
                    reference::p2m(p, &frame, &u64s, &q64, &mut ws64, &mut want);
                    reference::p2m(p, &frame_t, sources, charges, &mut ws, &mut in_t);
                }
                for (y, q) in u64s.iter().zip(&q64) {
                    let values = &host_harmonics(basis, p, frame.scaled(*y), false)[0];
                    for (o, v) in terms.iter_mut().zip(values) {
                        *o += (q * v).abs();
                    }
                }
            }
            let d64: Vec<[f64; 3]> = moments.iter().map(|d| d.map(w::<T>)).collect();
            if irregular {
                reference::p2l_dipoles(p, &frame, &u64s, &d64, &mut ws64, &mut want);
                reference::p2l_dipoles(p, &frame_t, sources, moments, &mut ws, &mut in_t);
            } else {
                reference::p2m_dipoles(p, &frame, &u64s, &d64, &mut ws64, &mut want);
                reference::p2m_dipoles(p, &frame_t, sources, moments, &mut ws, &mut in_t);
            }
            for (y, d) in u64s.iter().zip(&d64) {
                let h = host_harmonics(basis, p, frame.scaled(*y), true);
                for (i, o) in terms.iter_mut().enumerate() {
                    *o += (0..3)
                        .map(|k| (d[k] / frame.radius * h[1 + k][i]).abs())
                        .sum::<f64>();
                }
            }
        }
        [want, in_t.into_iter().map(w::<T>).collect(), terms]
    }

    /// Checks every box of `got` against the reference; adds the errors to `worst`.
    fn check(
        &self,
        what: &str,
        (p, irregular, kind): (usize, bool, SourceKind),
        got: &[T],
        worst: &mut Worst,
    ) {
        let n = coefficients(p);
        for t_box in 0..self.case.boxes.len() {
            let [want, in_t, terms] = self.reference((p, irregular, kind), t_box);
            let got: Vec<f64> = got[t_box * n..(t_box + 1) * n]
                .iter()
                .map(|&v| w(v))
                .collect();
            assert!(
                got.iter().all(|v| v.is_finite()),
                "{what}: box {t_box}: not finite"
            );
            let multipole = !irregular;
            let device = degree_error(multipole, p, &[&got], &[&want], &[&terms]);
            let host = degree_error(multipole, p, &[&in_t], &[&want], &[&terms]);
            assert!(
                device.0 <= tolerance::<T>(),
                "{what}: p = {p}, box {t_box}: error {:e} at degree {} exceeds {:e} (reference \
                 in T {:e})",
                device.0,
                device.1,
                tolerance::<T>(),
                host.0
            );
            worst.add(device, host);
        }
    }
}

/// Launches P2M or P2L of `on`, whose store holds `kind`.
fn launch<T: Real>(
    device: &mut Device,
    on: &mut SourcesOnDevice<T>,
    (layout, p, irregular): (LeafLayout, usize, bool),
    kind: SourceKind,
) {
    let inputs = SourceInputs {
        view: &on.view,
        level: on.level,
        boxes: &on.boxes,
        leaves: &on.leaves,
        source_offsets: &on.source_offsets,
        sources: on.sources.as_slice(),
        vectors: 1,
        source_kind: kind,
    };
    let out = on.out.as_slice_mut();
    if irregular {
        p2l(device, layout, p, &inputs, out).unwrap();
    } else {
        p2m(device, layout, p, &inputs, out).unwrap();
    }
}

/// P2M (its own leaf) and P2L (the X list and same-level leaves two boxes away) with
/// dipoles on levels 2, 9 and 16 against `nd_fmm_ref::leaf`.
fn dipole_expansions(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        expansions_at::<T>(device, &DEGREES);
    }
    each_precision!(device, body);
}

/// [`dipole_expansions`] at `degrees`.
pub(super) fn expansions_at<T: Real>(device: &mut Device, degrees: &[usize]) {
    let mut rng = Rng::new(0xd1b0_7002);
    for layout in layouts(device) {
        for &level in &LEVELS {
            for irregular in [false, true] {
                let name = if irregular { "P2L" } else { "P2M" };
                for kind in KINDS {
                    let mut worst = Worst::default();
                    for &p in degrees {
                        let case = if irregular {
                            p2l_case::<T>(&mut rng, level, 12)
                        } else {
                            p2m_case::<T>(&mut rng, level)
                        };
                        let case = DipoleCase::new(&mut rng, case);
                        let got = case.run(device, (layout, p, irregular), kind);
                        case.check(
                            &format!("{name}, {kind}, {layout}, level {level}"),
                            (p, irregular, kind),
                            &got,
                            &mut worst,
                        );
                    }
                    worst.print_coefficients(&format!(
                        "{name}, {kind}, {layout}, {}, level {level}",
                        T::FLOAT
                    ));
                }
            }
        }
    }
}

/// Bits, for both kinds and P2M and P2L at p = 8: repeated launches, a row split into one
/// call per entry, and the order (charges and dipoles in one launch equal, entry by entry,
/// a launch of the charges followed by a launch of the moments).
fn dipole_invariance_and_order(device: &mut Device) {
    fn body<T: Real>(device: &mut Device) {
        let mut rng = Rng::new(0xd1b0_7007);
        let p = 8;
        let same = |a: &[T], b: &[T]| a.iter().zip(b).all(|(x, y)| x.bits() == y.bits());
        for layout in layouts(device) {
            for irregular in [false, true] {
                let case = if irregular {
                    p2l_case::<T>(&mut rng, 9, 12)
                } else {
                    p2m_case::<T>(&mut rng, 9)
                };
                let case = DipoleCase::new(&mut rng, case);
                let longest = case.case.rows.iter().map(Vec::len).max().unwrap();
                let per_entry = |k: usize| -> Vec<Vec<u32>> {
                    case.case
                        .rows
                        .iter()
                        .map(|row| row.get(k).map(|&j| vec![j]).unwrap_or_default())
                        .collect()
                };
                for kind in KINDS {
                    let whole = case.run(device, (layout, p, irregular), kind);
                    assert!(
                        same(&case.run(device, (layout, p, irregular), kind), &whole),
                        "{layout}, {kind}: repeated"
                    );
                    let zeros = vec![t::<T>(0.0); whole.len()];
                    let mut on = case.upload(device, &case.case.rows, &zeros, kind);
                    for k in 0..longest {
                        let mut part = case.upload(device, &per_entry(k), &[], kind);
                        std::mem::swap(&mut part.out, &mut on.out);
                        launch(device, &mut part, (layout, p, irregular), kind);
                        std::mem::swap(&mut part.out, &mut on.out);
                    }
                    assert!(
                        same(&download(device, &on.out), &whole),
                        "{layout}, {kind}: split per entry"
                    );
                }
                let both = case.run(
                    device,
                    (layout, p, irregular),
                    SourceKind::ChargesAndDipoles,
                );
                let zeros = vec![t::<T>(0.0); both.len()];
                let mut on = case.upload(device, &case.case.rows, &zeros, SourceKind::Charges);
                for k in 0..longest {
                    for kind in [SourceKind::Charges, SourceKind::Dipoles] {
                        let mut part = case.upload(device, &per_entry(k), &[], kind);
                        std::mem::swap(&mut part.out, &mut on.out);
                        launch(device, &mut part, (layout, p, irregular), kind);
                        std::mem::swap(&mut part.out, &mut on.out);
                    }
                }
                assert!(
                    same(&download(device, &on.out), &both),
                    "{layout}, irregular {irregular}: charges and dipoles against the charges \
                     and then the moments of each entry"
                );
            }
            println!(
                "  {layout}, {}: repeated launches, rows split per entry and the order of \
                 charges and moments bit for bit (P2M, P2L)",
                T::FLOAT
            );
        }
    }
    each_precision!(device, body);
}

tests_on!(cpu: dipole_expansions, dipole_invariance_and_order);
tests_on!(metal: dipole_expansions, dipole_invariance_and_order);
tests_on!(cuda: dipole_expansions, dipole_invariance_and_order);
