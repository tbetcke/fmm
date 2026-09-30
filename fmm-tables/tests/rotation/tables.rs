//! The tables at the canonical frames: agreement with `nd_fmm_ref::rotation` for every
//! entry, the on-axis offsets, the distinct-angle and distance counts, the rotation
//! rules on point sources, the round trip of the rotations, accumulation, storage,
//! determinism and argument checks.

use nd_fmm_math::Layout;
use nd_fmm_math::rotation::{blocks_len, euler_zyz};
use nd_fmm_ref::{Frame, Workspace, leaf, rotation};
use nd_fmm_tables::RotationTables;
use nd_fmm_tables::geometry::{OCTANT_COUNT, m2l_frames, m2l_offset_index};
use nd_fmm_tables::rotation::{
    Alignment, M2L_AZIMUTH_COUNT, M2L_DISTANCE_COUNT, M2L_POLAR_ANGLE_COUNT, Operator,
    RotationScratch, ShiftTables,
};
use nd_fmm_tables::symmetry::Expansion;

use crate::common::{
    DEBUG_DEGREES, Kind, OPERATORS, P_DEBUG, REFERENCE_TOL, SplitMix64, Worst, degree_norms, dense,
    error, input_kind, len, random_coefficients, reference, run, tables, translate,
};

/// The signature of `ShiftTables::rotate` and `ShiftTables::rotate_back`.
type Rotate = fn(&ShiftTables<f64>, usize, &[f64], &mut [f64], &mut RotationScratch<f64>);

/// The on-axis V-list offsets.
const AXIS_OFFSETS: [[i64; 3]; 4] = [[0, 0, -3], [0, 0, -2], [0, 0, 2], [0, 0, 3]];

#[test]
fn operators_equal_reference_rotation_at_the_canonical_frames() {
    // Error measure: output coefficients per degree in the orthonormal weighting (Nₘ
    // for M2M, Nₘ/Sₘ for L2L and M2L), relative to the terms |Aᵢₖ xₖ| of the dense
    // table (`main`). All 316 offsets and all 8 octants, three random inputs each,
    // p ≤ 8.
    let mut rng = SplitMix64::new(0x0706_0001);
    let worst = [
        Worst::new("M2L tables vs rotation, canonical frames, p ≤ 8 (terms)"),
        Worst::new("M2M tables vs rotation, canonical frames, p ≤ 8 (terms)"),
        Worst::new("L2L tables vs rotation, canonical frames, p ≤ 8 (terms)"),
    ];
    for p in DEBUG_DEGREES {
        let mut ws = Workspace::new(p);
        for (op, worst) in OPERATORS.into_iter().zip(&worst) {
            let family = tables(p).tables(op);
            let set = dense(p).of(op);
            for t in 0..op.count() {
                let (from, to) = op.frames(t);
                for _ in 0..3 {
                    let x = random_coefficients(input_kind(op), p, &mut rng);
                    let got = run(family, t, &x);
                    let want = translate(reference(op), p, (&from, &to), &mut ws, &x);
                    let e = worst.update(error(op, set, t, &x, &got, &want));
                    assert!(e <= REFERENCE_TOL, "{op:?}, p = {p}, entry {t}: {e:.3e}");
                }
            }
        }
    }
}

#[test]
fn on_axis_offsets_use_no_rotation_and_equal_the_coaxial_form_bit_for_bit() {
    // Error measure: bit-for-bit equality with `nd_fmm_ref::rotation::m2l`, which on
    // the z axis applies the coaxial form of §3.11 alone, accumulating into the same
    // nonzero output. Also: every other offset and every octant is rotated.
    let mut rng = SplitMix64::new(0x0706_0002);
    for p in 0..=P_DEBUG {
        let family = tables(p).tables(Operator::M2l);
        let mut ws = Workspace::new(p);
        let mut scratch = RotationScratch::new(p);
        for d in AXIS_OFFSETS {
            let index = m2l_offset_index(d).unwrap();
            let shift = family.shift(index);
            let want = if d[2] > 0 {
                Alignment::Up
            } else {
                Alignment::Down
            };
            assert_eq!(shift.alignment, want, "{d:?}");
            assert_eq!(family.distance(shift.distance), 2.0 * d[2].abs() as f64);

            let (source, target) = m2l_frames(index);
            let x = random_coefficients(Kind::Multipole, p, &mut rng);
            let start = random_coefficients(Kind::Local, p, &mut rng);
            let mut got = start.clone();
            family.apply(index, &x, &mut got, &mut scratch);
            let mut coaxial = start.clone();
            family.translate_coaxial(index, &x, &mut coaxial);
            let mut want = start.clone();
            rotation::m2l(p, &source, &target, &mut ws, &x, &mut want);
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "{d:?}, p = {p}");
            assert_eq!(bits(&coaxial), bits(&want), "{d:?}, p = {p}");

            // No rotation: `rotate` and `rotate_back` add their input unchanged.
            for rotate in [ShiftTables::rotate as Rotate, ShiftTables::rotate_back] {
                let mut out = vec![0.0; len(p)];
                rotate(family, index, &x, &mut out, &mut scratch);
                assert_eq!(bits(&out), bits(&x));
            }
        }
        for t in 0..Operator::M2l.count() {
            let d = Operator::M2l.direction(t);
            let on_axis = d[0] == 0 && d[1] == 0;
            let rotated = matches!(family.shift(t).alignment, Alignment::Rotated { .. });
            assert_eq!(rotated, !on_axis, "{d:?}");
        }
        for op in [Operator::M2m, Operator::L2l] {
            let family = tables(p).tables(op);
            for o in 0..OCTANT_COUNT {
                assert!(matches!(
                    family.shift(o).alignment,
                    Alignment::Rotated { .. }
                ));
            }
        }
    }
}

/// Polar angle and azimuth of the canonical shift of entry `t` of `op`, as
/// `nd_fmm_ref::rotation` computes them (CONVENTIONS §3.2).
fn own_angles(op: Operator, t: usize) -> (f64, f64) {
    let (from, to) = op.frames(t);
    let [x, y, z]: [f64; 3] = core::array::from_fn(|i| to.centre[i] - from.centre[i]);
    ((x * x + y * y).sqrt().atan2(z), y.atan2(x))
}

/// The number of values in `values` that differ by more than `gap` from every earlier
/// one.
fn distinct(values: impl IntoIterator<Item = f64>, gap: f64) -> usize {
    let mut seen: Vec<f64> = Vec::new();
    for v in values {
        if seen.iter().all(|s| (s - v).abs() > gap) {
            seen.push(v);
        }
    }
    seen.len()
}

#[test]
fn distinct_angle_and_distance_counts_are_as_documented() {
    // Error measure: exact counts. Independently of the tables, the angles of the
    // canonical shifts are grouped with a gap of 1e-12, far below the smallest distance
    // between distinct angles and far above rounding. Every stored angle is then within
    // 4 ε of each of its entries' own angles, and distinct stored angles are apart.
    let op = Operator::M2l;
    let offsets = 0..op.count();
    let polar = distinct(offsets.clone().map(|t| own_angles(op, t).0), 1e-12);
    let off_axis = offsets.clone().filter(|&t| {
        let d = op.direction(t);
        d[0] != 0 || d[1] != 0
    });
    let azimuth = distinct(off_axis.map(|t| own_angles(op, t).1), 1e-12);
    let distance = distinct(
        offsets.map(|t| {
            let (source, target) = m2l_frames(t);
            let [x, y, z]: [f64; 3] = core::array::from_fn(|i| target.centre[i] - source.centre[i]);
            (x * x + y * y + z * z).sqrt()
        }),
        1e-12,
    );
    assert_eq!(
        (polar, azimuth, distance),
        (M2L_POLAR_ANGLE_COUNT, M2L_AZIMUTH_COUNT, M2L_DISTANCE_COUNT)
    );
    assert_eq!((polar, azimuth, distance), (49, 32, 15));

    let rotation = tables(3);
    for (op, counts) in [
        (Operator::M2l, (47, 32, 15)),
        (Operator::M2m, (2, 4, 1)),
        (Operator::L2l, (2, 4, 1)),
    ] {
        let family = rotation.tables(op);
        assert_eq!(
            (
                family.polar_count(),
                family.azimuth_count(),
                family.distance_count()
            ),
            counts,
            "{op:?}"
        );
        for t in 0..op.count() {
            if let Alignment::Rotated { polar, azimuth } = family.shift(t).alignment {
                let (theta, phi) = own_angles(op, t);
                let eps = 4.0 * f64::EPSILON;
                assert!((family.polar_angle(polar) - theta).abs() <= eps * theta.abs());
                assert!((family.azimuth_angle(azimuth) - phi).abs() <= eps * phi.abs());
            }
        }
        let polar: Vec<f64> = (0..family.polar_count())
            .map(|k| family.polar_angle(k))
            .collect();
        assert_eq!(distinct(polar, 1e-3), family.polar_count(), "{op:?}");
        let azimuth: Vec<f64> = (0..family.azimuth_count())
            .map(|k| family.azimuth_angle(k))
            .collect();
        assert_eq!(distinct(azimuth, 1e-3), family.azimuth_count(), "{op:?}");
    }
}

/// Rotates the points `y` about the origin by `q`.
fn rotate_points(q: &[[f64; 3]; 3], y: &[[f64; 3]]) -> Vec<[f64; 3]> {
    y.iter()
        .map(|y| q.map(|row| row[0] * y[0] + row[1] * y[1] + row[2] * y[2]))
        .collect()
}

/// The P2M (multipole rule) or P2L (local rule) coefficients of `charges` at
/// `sources` in the frame (0, 1).
fn p2x(expansion: Expansion, p: usize, sources: &[[f64; 3]], charges: &[f64]) -> Vec<f64> {
    let frame = Frame::new([0.0; 3], 1.0);
    let mut ws = Workspace::new(p);
    let mut out = vec![0.0; len(p)];
    match expansion {
        Expansion::Multipole => leaf::p2m(p, &frame, sources, charges, &mut ws, &mut out),
        Expansion::Local => leaf::p2l(p, &frame, sources, charges, &mut ws, &mut out),
    }
    out
}

/// Worst per-degree weighted error of `got` against `want`, relative to the weighted
/// degree norms of `want`.
fn relative_degree_error(kind: Kind, p: usize, got: &[f64], want: &[f64]) -> f64 {
    let diff: Vec<f64> = got.iter().zip(want).map(|(a, b)| a - b).collect();
    let norms = degree_norms(kind, p, want);
    degree_norms(kind, p, &diff)
        .iter()
        .zip(&norms)
        .map(|(e, n)| e / n)
        .fold(0.0, f64::max)
}

#[test]
fn rotations_follow_the_rule_of_their_kind_on_point_sources() {
    // CONVENTIONS §3.11, "Rotation of coefficients": for sources y ↦ Q y about the
    // centre, the coefficients of degree n become T(Q) times the old ones, with the
    // multipole rule for P2M and the local rule for P2L. `rotate` applies T_in(Q) and
    // `rotate_back` T_out(Qᵀ), with Q = R_y(−θ) R_z(−φ) from each entry's own shift.
    // Sources at |u| ≤ √3/2 for P2M and |u| ≥ 2√3 for P2L (§3.9).
    //
    // Error measure: per degree in the orthonormal weighting of the rule, relative to
    // the weighted degree norms of the coefficients of the rotated sources, bound 1e-13
    // (the harmonics of the two source sets carry their own rounding).
    let p = P_DEBUG;
    let mut rng = SplitMix64::new(0x0706_0003);
    let worst = Worst::new("T(Q) on point-source coefficients, p = 8 (degree norms)");
    let mut scratch = RotationScratch::new(p);
    for op in OPERATORS {
        let family = tables(p).tables(op);
        for t in 0..op.count() {
            let (theta, phi) = own_angles(op, t);
            let q = euler_zyz(0.0, -theta, -phi);
            let q_transposed: [[f64; 3]; 3] = core::array::from_fn(|i| q.map(|row| row[i]));
            for (expansion, rotate, q) in [
                (op.input(), ShiftTables::rotate as Rotate, q),
                (op.output(), ShiftTables::rotate_back, q_transposed),
            ] {
                if !matches!(family.shift(t).alignment, Alignment::Rotated { .. }) {
                    continue;
                }
                let sources: Vec<[f64; 3]> = (0..4)
                    .map(|_| {
                        let u = rng.in_cube();
                        match expansion {
                            Expansion::Multipole => u.map(|v| 0.5 * v),
                            Expansion::Local => u.map(|v| v.signum() * (2.0 + 2.0 * v.abs())),
                        }
                    })
                    .collect();
                let charges: Vec<f64> = (0..4).map(|_| rng.charge()).collect();
                let before = p2x(expansion, p, &sources, &charges);
                let want = p2x(expansion, p, &rotate_points(&q, &sources), &charges);
                let mut got = vec![0.0; len(p)];
                rotate(family, t, &before, &mut got, &mut scratch);
                let kind = crate::common::kind(expansion);
                let e = worst.update(relative_degree_error(kind, p, &got, &want));
                assert!(e <= 1e-13, "{op:?}, entry {t}, {expansion:?}: {e:.3e}");
            }
        }
    }
}

#[test]
fn rotate_back_inverts_rotate_for_m2m_and_l2l() {
    // For M2M and L2L input and output follow the same rule, so T(Qᵀ) T(Q) = I.
    //
    // Error measure: per degree in the orthonormal weighting of the rule, relative to
    // the weighted degree norms of the input, bound 1e-14.
    let mut rng = SplitMix64::new(0x0706_0004);
    let worst = Worst::new("T(Qᵀ) T(Q) = I, M2M and L2L, p ≤ 8 (degree norms)");
    for p in DEBUG_DEGREES {
        let mut scratch = RotationScratch::new(p);
        for op in [Operator::M2m, Operator::L2l] {
            let family = tables(p).tables(op);
            for o in 0..OCTANT_COUNT {
                let x = random_coefficients(input_kind(op), p, &mut rng);
                let mut rotated = vec![0.0; len(p)];
                family.rotate(o, &x, &mut rotated, &mut scratch);
                let mut back = vec![0.0; len(p)];
                family.rotate_back(o, &rotated, &mut back, &mut scratch);
                let e = worst.update(relative_degree_error(input_kind(op), p, &back, &x));
                assert!(e <= 1e-14, "{op:?}, o = {o}, p = {p}: {e:.3e}");
            }
        }
    }
}

#[test]
fn operators_accumulate_and_match_the_ref_signatures() {
    // Error measure: the accumulated result minus the incoming output against the
    // result from zero, per degree relative to the terms of the dense table and the
    // incoming values, bound 1e-15. `RotationTables::m2l`, `m2m` and `l2l` are the
    // family's `apply`, bit for bit.
    let p = 5;
    let mut rng = SplitMix64::new(0x0706_0005);
    let rotation = tables(p);
    let mut scratch = RotationScratch::new(p);
    for op in OPERATORS {
        let family = rotation.tables(op);
        for t in [0, op.count() / 2, op.count() - 1] {
            let x = random_coefficients(input_kind(op), p, &mut rng);
            let start = random_coefficients(crate::common::output_kind(op), p, &mut rng);
            let from_zero = run(family, t, &x);
            let mut got = start.clone();
            family.apply(t, &x, &mut got, &mut scratch);
            let mut via_tables = start.clone();
            match op {
                Operator::M2l => rotation.m2l(t, &x, &mut via_tables, &mut scratch),
                Operator::M2m => rotation.m2m(t, &x, &mut via_tables, &mut scratch),
                Operator::L2l => rotation.l2l(t, &x, &mut via_tables, &mut scratch),
            }
            assert_eq!(got, via_tables, "{op:?}");
            let added: Vec<f64> = got.iter().zip(&start).map(|(a, b)| a - b).collect();
            let mut scale = crate::common::terms(dense(p).of(op), t, &x);
            for (s, b) in scale.iter_mut().zip(&start) {
                *s += b.abs();
            }
            let e = crate::common::degree_error(
                crate::common::output_kind(op),
                p,
                &added,
                &from_zero,
                &scale,
            );
            assert!(e <= 1e-15, "{op:?}, entry {t}: {e:.3e}");
        }
    }
}

#[test]
fn storage_follows_the_documented_formula() {
    // Error measure: exact counts. With B = (p + 1)(2p + 1)(2p + 3)/3 and
    // C = (p + 1)(p + 2)(2p + 3)/6: 94 B + 64 p + 15 C for M2L, 4 B + 8 p + C for M2M
    // and L2L, 102 B + 80 p + 17 C in all.
    for p in 0..=P_DEBUG {
        let b = blocks_len(p);
        assert_eq!(b, (p + 1) * (2 * p + 1) * (2 * p + 3) / 3);
        let c = (p + 1) * (p + 2) * (2 * p + 3) / 6;
        let rotation = tables(p);
        let family = |op| rotation.tables(op);
        assert_eq!(
            family(Operator::M2l).storage_len(),
            94 * b + 64 * p + 15 * c
        );
        for op in [Operator::M2m, Operator::L2l] {
            assert_eq!(family(op).storage_len(), 4 * b + 8 * p + c);
        }
        assert_eq!(rotation.storage_len(), 102 * b + 80 * p + 17 * c);
        for op in OPERATORS {
            let family = family(op);
            for k in 0..family.polar_count() {
                assert_eq!(family.forward_blocks(k).len(), b);
                assert_eq!(family.backward_blocks(k).len(), b);
            }
            for k in 0..family.azimuth_count() {
                assert_eq!(family.azimuth_factors(k).len(), 2 * p);
            }
            for k in 0..family.distance_count() {
                assert_eq!(family.coaxial_factors(k).len(), c);
            }
        }
    }
    let bytes = |p: usize| 8 * RotationTables::<f64>::build(p).storage_len();
    for p in [8, 16, 20] {
        eprintln!(
            "rotation tables p = {p}: {} bytes ({:.2} MB) in f64",
            bytes(p),
            bytes(p) as f64 / 1e6
        );
    }
}

/// Every stored real of `tables`, as bits, and every shift.
fn contents(tables: &RotationTables<f64>) -> (Vec<u64>, Vec<String>) {
    let mut bits = Vec::new();
    let mut shifts = Vec::new();
    for op in OPERATORS {
        let family = tables.tables(op);
        let mut push = |v: &[f64]| bits.extend(v.iter().map(|x| x.to_bits()));
        for k in 0..family.polar_count() {
            push(&[family.polar_angle(k)]);
            push(family.forward_blocks(k));
            push(family.backward_blocks(k));
        }
        for k in 0..family.azimuth_count() {
            push(&[family.azimuth_angle(k)]);
            push(family.azimuth_factors(k));
        }
        for k in 0..family.distance_count() {
            push(&[family.distance(k)]);
            push(family.coaxial_factors(k));
        }
        shifts.extend((0..family.count()).map(|t| format!("{:?}", family.shift(t))));
    }
    (bits, shifts)
}

#[test]
fn building_twice_gives_bit_identical_tables() {
    // Error measure: bit-for-bit equality of every stored real and every shift.
    for p in [0, 3, P_DEBUG] {
        let (a, b) = (
            RotationTables::<f64>::build(p),
            RotationTables::<f64>::build(p),
        );
        assert_eq!(contents(&a), contents(&b), "p = {p}");
        assert_eq!(contents(&a), contents(tables(p)), "p = {p}");
        assert!(contents(&a).0.len() >= a.storage_len());
    }
}

#[test]
fn coaxial_factors_serve_both_orders_of_a_pair() {
    // CONVENTIONS §3.11: in real storage slots +i and −i are transformed by the same
    // real factors. The stored factors come from slot +i; the reference on the axis
    // applied to slot (n, −i) gives the same column at slot (j, −i), and nothing
    // elsewhere.
    //
    // Error measure: bit-for-bit equality.
    let p = 6;
    let layout = Layout::new(p);
    let mut ws = Workspace::new(p);
    for op in OPERATORS {
        let family = tables(p).tables(op);
        let (from, to) = op.frames(0);
        let axis_from = Frame::new([0.0; 3], from.radius);
        for k in 0..family.distance_count() {
            let axis_to = Frame::new([0.0, 0.0, family.distance(k)], to.radius);
            let factors = family.coaxial_factors(k);
            let mut start = 0;
            for i in 0..=p {
                let width = p + 1 - i;
                for n in i..=p {
                    for m in [i as isize, -(i as isize)] {
                        let mut unit = vec![0.0; len(p)];
                        unit[layout.idx(n, m)] = 1.0;
                        let column =
                            translate(reference(op), p, (&axis_from, &axis_to), &mut ws, &unit);
                        for (slot, &value) in column.iter().enumerate() {
                            let (j, order) = layout.nm(slot);
                            let want = if order == m && j >= i {
                                factors[start + (j - i) * width + (n - i)]
                            } else {
                                0.0
                            };
                            assert_eq!(value.to_bits(), want.to_bits(), "{op:?} ({n}, {m})");
                        }
                    }
                }
                start += width * width;
            }
        }
    }
}

#[test]
#[should_panic(expected = "`scratch` must be of degree p = 3")]
fn wrong_scratch_degree_panics() {
    let mut scratch = RotationScratch::new(2);
    let mut out = vec![0.0; 16];
    tables(3).m2l(0, &[0.0; 16], &mut out, &mut scratch);
}

#[test]
#[should_panic(expected = "`input` must have length (p + 1)^2 = 16")]
fn wrong_input_length_panics() {
    let mut scratch = RotationScratch::new(3);
    let mut out = vec![0.0; 16];
    tables(3).m2m(0, &[0.0; 9], &mut out, &mut scratch);
}

#[test]
#[should_panic(expected = "index out of bounds")]
fn octant_index_8_panics() {
    let mut scratch = RotationScratch::new(3);
    let mut out = vec![0.0; 16];
    tables(3).l2l(8, &[0.0; 16], &mut out, &mut scratch);
}
