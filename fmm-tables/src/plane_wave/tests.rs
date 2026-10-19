//! Tests of the plane-wave tables (Phase 6 T13, "Tests that define done": the tables and
//! the level M2L against `nd_fmm_ref::direct::m2l` on levels 2, 9 and 16).
//!
//! Error measure (Phase 6 T10's, `spikes/plane-wave/SPIKE_REPORT.md` §5): the potential
//! r φ = Σ L̃ R(v) of each target's local at four random points of the box, against the
//! same of `nd_fmm_ref::direct::m2l`'s local, |Δ| over the bound E Σ_V |q|, E the
//! quadrature's measured error for 1/|u| in box widths ([`Quadrature::measured_error`]):
//! a unit charge contributes r/|x − y| = ½ (1/|u|) to r φ, so the quadrature alone allows
//! ½ E per unit charge, and the bound doubles it for the dropped Fourier modes and the
//! local's truncation at p. Where E lies below the waves' rounding floor (5e-10, from
//! s ≈ 27 on) the floor replaces it. The boxes lie in the dyadic unit domain, whose
//! centres are exact on every level (crate documentation, "Error measures").

use nd_fmm_math::{Layout, harmonics};
use nd_fmm_ref::{Frame, Workspace, direct};

use super::geometry::{CHILDREN, SLOTS, child_bits, slot_coordinates};
use super::*;

/// A xorshift generator; the tests need spread, not statistics.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn quadrature_tables_are_consistent() {
    for s in quadrature::NODES {
        let q = Quadrature::with_nodes(s);
        assert_eq!(q.nodes(), s);
        assert_eq!(q.weights.len(), s);
        assert_eq!(q.alpha_nodes.len(), s);
        assert_eq!(q.fourier.len(), s);
        assert!(q.lambdas.windows(2).all(|w| w[0] < w[1]));
        assert!(q.max_error > 0.0 && q.max_error < 0.2);
    }
    // FMM3D's sums for the sizes its lfmm3d chooses (checked by hand against
    // lwtsexp_sep1.f in Phase 6 T10): s = 12, 20, 29, 37.
    let sums: Vec<usize> = [12, 20, 29, 37]
        .iter()
        .map(|&s| Quadrature::with_nodes(s).fmm3d_terms())
        .collect();
    assert_eq!(sums, [217, 705, 1627, 2770]);
    for p in 0..=MAX_DEGREE {
        assert!(quadrature::NODES.contains(&nodes_for_degree(p)));
    }
}

/// The λ sum meets the stated error (many α nodes); FMM3D's α counts exceed it 6–23×
/// (Phase 6 T10).
#[test]
fn stated_and_measured_errors() {
    for s in [8, 12] {
        let q = Quadrature::with_nodes(s);
        let mut worst: f64 = 0.0;
        for &z in &[1.0, 1.3, 2.0, 3.1, 4.0] {
            for &(x, y) in &[(0.0, 0.0), (1.0, -2.0), (4.0, 4.0), (-3.5, 2.5)] {
                let mut sum = 0.0;
                for k in 0..s {
                    let (lambda, n) = (q.lambdas[k], 400);
                    let mut inner = 0.0;
                    for j in 0..n {
                        let alpha = core::f64::consts::TAU * j as f64 / n as f64;
                        inner += (lambda * (x * alpha.cos() + y * alpha.sin())).cos();
                    }
                    sum += q.weights[k] / n as f64 * (-lambda * z).exp() * inner;
                }
                let exact = 1.0 / (x * x + y * y + z * z).sqrt();
                worst = worst.max((sum - exact).abs());
            }
        }
        assert!(worst <= 1.05 * q.max_error, "s = {s}: {worst:.3e}");
        let measured = q.measured_error();
        assert!(measured > 5.0 * q.max_error && measured < 25.0 * q.max_error);
    }
}

#[test]
fn tables_have_their_sizes() {
    for p in [0, 3, 8, 18] {
        let tables = PlaneWaveTables::<f64>::build(p);
        let q = tables.quadrature();
        assert_eq!(tables.h(), q.half_terms());
        assert_eq!(tables.blocks().len(), q.nodes());
        let modes: usize = (0..q.nodes()).map(|k| q.modes(k, p)).sum();
        assert_eq!(tables.mode_count(), modes);
        // 6 × 6 lateral offsets times 4 signed ones along (±1.5, ±2.5 box widths) at
        // most; 104 occur.
        assert_eq!(tables.source_shift_count(), 104);
        for direction in Direction::ALL {
            for accumulator in tables.geometry().accumulators(direction) {
                for &slot in &accumulator.slots {
                    assert!(tables.source_shift(direction, slot as usize).is_some());
                }
            }
        }
        // f32 tables are the f64 tables rounded.
        let single = PlaneWaveTables::<f32>::build(p);
        assert_eq!(single.cos()[3], tables.cos()[3] as f32);
    }
}

/// A source shift to the parent's centre followed by the child shift is the shift of the
/// pair, to rounding: the factorisation merge and shift rests on.
#[test]
fn source_and_child_shifts_compose_to_the_pair_shift() {
    let p = 6;
    let tables = PlaneWaveTables::<f64>::build(p);
    let mut worst: f64 = 0.0;
    for slot in 0..SLOTS {
        for child in 0..CHILDREN {
            let Some(direction) = tables.geometry().direction(slot, child) else {
                continue;
            };
            let (a, b) = (slot_coordinates(slot), child_bits(child));
            let d: [i64; 3] = core::array::from_fn(|i| b[i] + 2 - a[i]);
            let pair = tables.factors_f64(direction.rotate(d).map(|x| x as f64), direction);
            let mut composed = tables.source_shift(direction, slot).unwrap().to_vec();
            shift(&mut composed, tables.child_shift(direction, child));
            for (x, y) in composed.iter().zip(&pair) {
                worst = worst.max((x - y).abs());
            }
        }
    }
    assert!(worst <= 1e-15, "{worst:e}");
}

/// The rounding floor of the waves, in units of Σ|q|: the large terms λⁿ M̂ₙ at the largest
/// λ round to about 2–4e-10 Σ|q| from s = 27 on (Phase 6 T10, `spikes/plane-wave/
/// SPIKE_REPORT.md` §5; the study of [`level_every_degree`] at p ≥ 17).
const ROUNDING_FLOOR: f64 = 5e-10;

/// The bound per unit of Σ|q| of the module documentation, E, or the rounding floor where
/// the quadrature's measured error lies below it.
fn bound(tables: &PlaneWaveTables<f64>) -> f64 {
    tables.quadrature().measured_error().max(ROUNDING_FLOOR)
}

/// The four charges and multipole of one box (scaled, §3.7) and four target points.
struct BoxData {
    charges: Vec<([f64; 3], f64)>,
    points: Vec<[f64; 3]>,
    multipole: Vec<f64>,
}

/// Box `index` on `level` of the dyadic unit domain: centre and half-width.
fn frame(level: u32, index: [i64; 3]) -> Frame<f64> {
    let width = 0.5f64.powi(level as i32);
    Frame::new(index.map(|i| (i as f64 + 0.5) * width), width / 2.0)
}

fn box_data(p: usize, f: &Frame<f64>, rng: &mut Rng) -> BoxData {
    let layout = Layout::new(p);
    let mut multipole = vec![0.0; layout.len()];
    let mut r = vec![0.0; layout.len()];
    let mut charges = Vec::new();
    for _ in 0..4 {
        let u: [f64; 3] = core::array::from_fn(|_| 2.0 * rng.next() - 1.0);
        let q = rng.next();
        // M̃ = q conj Rₙᵐ(u) (§3.7); conj negates the slots m < 0 in real storage.
        harmonics::regular(p, u, &mut r);
        for n in 0..=p {
            for m in -(n as isize)..=n as isize {
                let i = layout.idx(n, m);
                multipole[i] += q * if m < 0 { -r[i] } else { r[i] };
            }
        }
        charges.push((core::array::from_fn(|i| f.centre[i] + f.radius * u[i]), q));
    }
    let points = (0..4)
        .map(|_| core::array::from_fn(|i| f.centre[i] + f.radius * (2.0 * rng.next() - 1.0)))
        .collect();
    BoxData {
        charges,
        points,
        multipole,
    }
}

/// r φ of the scaled local `local` of the box `f` at x (§3.6, §3.7).
fn potential(p: usize, f: &Frame<f64>, local: &[f64], x: [f64; 3]) -> f64 {
    let layout = Layout::new(p);
    let mut r = vec![0.0; layout.len()];
    harmonics::regular(
        p,
        core::array::from_fn(|i| (x[i] - f.centre[i]) / f.radius),
        &mut r,
    );
    let mut sum = 0.0;
    for n in 0..=p {
        sum += local[layout.idx(n, 0)] * r[layout.idx(n, 0)];
        for m in 1..=n as isize {
            let (a, b) = (layout.idx(n, m), layout.idx(n, -m));
            sum += 2.0 * (local[a] * r[a] - local[b] * r[b]);
        }
    }
    sum
}

/// The errors of one parent's level M2L.
#[derive(Debug)]
struct Outcome {
    /// max over points of |φ_pw − φ_direct| / (E Σ_V |q|).
    ratio: f64,
    /// max over points of |φ_pw − φ_direct| / Σ_V |q|.
    error: f64,
    /// max over points of |φ_direct − φ_charges| / Σ_V |q|: the M2L truncation.
    truncation: f64,
}

/// The M2L of every V pair into the eight children of the parent at `parent` on
/// `level − 1`, from the 216 slots: by plane waves, each pair translated directly along
/// its merge direction (the waves of each source once per direction, one shift per pair,
/// the local of each child once per frame), and by `nd_fmm_ref::direct::m2l`.
fn parent_m2l(level: u32, parent: [i64; 3], tables: &PlaneWaveTables<f64>, seed: u64) -> Outcome {
    let p = tables.p();
    let n = (p + 1) * (p + 1);
    let (w, h) = (tables.wave_len(), tables.h());
    let mut rng = Rng(seed);
    let slot_frames: Vec<Frame<f64>> = (0..SLOTS)
        .map(|slot| {
            let a = slot_coordinates(slot);
            frame(level, core::array::from_fn(|i| 2 * parent[i] - 2 + a[i]))
        })
        .collect();
    let sources: Vec<BoxData> = slot_frames
        .iter()
        .map(|f| box_data(p, f, &mut rng))
        .collect();
    // The waves of every slot in every direction: 6 × 2H.
    let (mut rotated, mut modes) = (vec![0.0; n], vec![0.0; 4 * tables.mode_count()]);
    let mut work = vec![0.0; tables.work_len()];
    let waves: Vec<Vec<f64>> = sources
        .iter()
        .map(|source| {
            let mut out = vec![0.0; 6 * w];
            for frame in 0..3 {
                tables.rotate_multipole(frame, &source.multipole, &mut rotated);
                tables.outgoing_modes(&rotated, &mut modes, &mut work);
                let (up, down) = out[2 * frame * w..(2 * frame + 2) * w].split_at_mut(w);
                waves_from_modes(tables, &modes, up, down);
            }
            out
        })
        .collect();
    let e = bound(tables);
    let mut ws = Workspace::new(p);
    let (mut ratio, mut error, mut truncation): (f64, f64, f64) = (0.0, 0.0, 0.0);
    for child in 0..CHILDREN {
        let b = child_bits(child);
        let target = frame(level, core::array::from_fn(|i| 2 * parent[i] + b[i]));
        let mut incoming = vec![0.0; 6 * w];
        let mut reference = vec![0.0; n];
        let mut charges = Vec::new();
        for slot in 0..SLOTS {
            let Some(direction) = tables.geometry().direction(slot, child) else {
                continue;
            };
            let a = slot_coordinates(slot);
            let d: [i64; 3] = core::array::from_fn(|i| b[i] + 2 - a[i]);
            let factors = tables.factors_f64(direction.rotate(d).map(|x| x as f64), direction);
            let dir = direction.index();
            shift_add(
                &mut incoming[dir * w..(dir + 1) * w],
                &waves[slot][dir * w..(dir + 1) * w],
                &factors,
            );
            direct::m2l(
                p,
                &slot_frames[slot],
                &target,
                &mut ws,
                &sources[slot].multipole,
                &mut reference,
            );
            charges.extend_from_slice(&sources[slot].charges);
        }
        let mut local = vec![0.0; n];
        let (mut sums, mut frame_local) = (vec![0.0; 4 * tables.mode_count()], vec![0.0; n]);
        for frame in 0..3 {
            let (up, down) = incoming[2 * frame * w..(2 * frame + 2) * w].split_at(w);
            sums_from_waves(tables, up, down, &mut sums);
            tables.local_from_sums(&sums, &mut frame_local);
            tables.add_local_from_frame(frame, &frame_local, &mut local);
        }
        debug_assert_eq!(incoming.len(), 12 * h);
        let charge_sum: f64 = charges.iter().map(|&(_, q)| q.abs()).sum();
        for &x in &box_data(p, &target, &mut rng).points {
            let (pw, exact) = (
                potential(p, &target, &local, x),
                potential(p, &target, &reference, x),
            );
            let direct_sum: f64 = charges
                .iter()
                .map(|&(y, q)| {
                    let r: f64 = (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt();
                    q * target.radius / r
                })
                .sum();
            ratio = ratio.max((pw - exact).abs() / (e * charge_sum));
            error = error.max((pw - exact).abs() / charge_sum);
            truncation = truncation.max((exact - direct_sum).abs() / charge_sum);
        }
    }
    Outcome {
        ratio,
        error,
        truncation,
    }
}

/// The tables' pair translation against `direct::m2l` on levels 2, 9 and 16, within
/// the quadrature's bound (module documentation), at the degrees of the debug run.
#[test]
fn pair_m2l_within_the_quadrature_error() {
    for p in [0, 3, 8] {
        let tables = PlaneWaveTables::<f64>::build(p);
        for (level, parent) in LEVELS {
            let out = parent_m2l(level, parent, &tables, 0x5eed + p as u64);
            assert!(out.ratio <= 1.0, "p = {p}, level {level}: {out:?}");
        }
    }
}

/// Levels 2, 9 and 16 with a parent each: on level 2 a parent of level 1, whose slots
/// partly lie outside the unit domain (their boxes are still well defined, so every slot
/// takes part); on levels 9 and 16 an interior parent.
const LEVELS: [(u32, [i64; 3]); 3] = [
    (2, [0, 1, 0]),
    (9, [127, 128, 127]),
    (16, [16383, 16384, 16383]),
];

/// Every degree p ≤ 18 on levels 2, 9 and 16 with the quadrature of
/// [`nodes_for_degree`]: within the quadrature's bound, and the plane waves' error at most
/// 0.1× the M2L truncation error where the quadratures reach it. Release:
/// `cargo test -p nd-fmm-tables --release -- --ignored --nocapture plane_wave`.
#[test]
#[ignore = "release: every degree, three levels"]
fn pair_m2l_every_degree() {
    for p in 0..=MAX_DEGREE {
        let tables = PlaneWaveTables::<f64>::build(p);
        for (level, parent) in LEVELS {
            let out = parent_m2l(level, parent, &tables, 0xfeed + p as u64);
            println!(
                "p = {p:2}, s = {:2}, level {level:2}: error {:.2e} Σ|q| ({:.3} E), \
                 M2L truncation {:.2e} Σ|q|",
                tables.quadrature().nodes(),
                out.error,
                out.ratio,
                out.truncation
            );
            assert!(out.ratio <= 1.0, "p = {p}, level {level}: {out:?}");
        }
    }
}

/// A uniform level in the dyadic unit domain with four random charges per box, its V
/// pairs, and the exact M2L of every pair (`direct::m2l`) at degree `p`.
struct LevelProblem {
    level: u32,
    p: usize,
    frames: Vec<Frame<f64>>,
    boxes: Vec<BoxData>,
    /// Per target: (source, offset d = target − source).
    pairs: Vec<Vec<(usize, [i64; 3])>>,
    reference: Vec<Vec<f64>>,
}

impl LevelProblem {
    fn new(level: u32, p: usize, seed: u64) -> Self {
        let side = 1i64 << level;
        let coordinates: Vec<[i64; 3]> = (0..side.pow(3))
            .map(|b| [b / (side * side), (b / side) % side, b % side])
            .collect();
        let frames: Vec<Frame<f64>> = coordinates.iter().map(|&c| frame(level, c)).collect();
        let mut rng = Rng(seed);
        let boxes: Vec<BoxData> = frames.iter().map(|f| box_data(p, f, &mut rng)).collect();
        let index = |c: [i64; 3]| (c[0] * side * side + c[1] * side + c[2]) as usize;
        let n = (p + 1) * (p + 1);
        let mut ws = Workspace::new(p);
        let mut pairs = Vec::new();
        let mut reference = Vec::new();
        for (t, &ct) in coordinates.iter().enumerate() {
            let parent = ct.map(|x| x.div_euclid(2));
            let mut row = Vec::new();
            let mut local = vec![0.0; n];
            for slot in 0..SLOTS {
                let a = slot_coordinates(slot);
                let cs: [i64; 3] = core::array::from_fn(|i| 2 * parent[i] - 2 + a[i]);
                let d: [i64; 3] = core::array::from_fn(|i| ct[i] - cs[i]);
                if cs.iter().any(|&x| !(0..side).contains(&x)) || d.iter().all(|x| x.abs() <= 1) {
                    continue;
                }
                let s = index(cs);
                direct::m2l(
                    p,
                    &frames[s],
                    &frames[t],
                    &mut ws,
                    &boxes[s].multipole,
                    &mut local,
                );
                row.push((s, d));
            }
            pairs.push(row);
            reference.push(local);
        }
        Self {
            level,
            p,
            frames,
            boxes,
            pairs,
            reference,
        }
    }

    /// The plane waves' error and the M2L truncation, each the largest over every target
    /// point relative to the target's Σ_V |q|, and the largest ratio to the bound E Σ_V |q|:
    /// each pair translated directly along its merge direction.
    fn errors(&self, tables: &PlaneWaveTables<f64>) -> Outcome {
        let p = self.p;
        assert_eq!(tables.p(), p);
        let (n, w) = ((p + 1) * (p + 1), tables.wave_len());
        let (mut rotated, mut modes) = (vec![0.0; n], vec![0.0; 4 * tables.mode_count()]);
        let mut work = vec![0.0; tables.work_len()];
        let waves: Vec<Vec<f64>> = self
            .boxes
            .iter()
            .map(|source| {
                let mut out = vec![0.0; 6 * w];
                for frame in 0..3 {
                    tables.rotate_multipole(frame, &source.multipole, &mut rotated);
                    tables.outgoing_modes(&rotated, &mut modes, &mut work);
                    let (up, down) = out[2 * frame * w..(2 * frame + 2) * w].split_at_mut(w);
                    waves_from_modes(tables, &modes, up, down);
                }
                out
            })
            .collect();
        // The pair shifts by (direction, offset index), computed once.
        let mut factors: Vec<Option<Vec<f64>>> = vec![None; 6 * M2L_OFFSET_COUNT];
        let e = bound(tables);
        let (mut ratio, mut error, mut truncation): (f64, f64, f64) = (0.0, 0.0, 0.0);
        let mut rng = Rng(0x7e57 ^ self.level as u64);
        for (t, row) in self.pairs.iter().enumerate() {
            let target = &self.frames[t];
            let side = 1i64 << self.level;
            let ct = [
                t as i64 / (side * side),
                (t as i64 / side) % side,
                t as i64 % side,
            ];
            let child = (4 * (ct[0] & 1) + 2 * (ct[1] & 1) + (ct[2] & 1)) as usize;
            let mut incoming = vec![0.0; 6 * w];
            for &(s, d) in row {
                let slot = geometry::slot_of(child, d).unwrap();
                let direction = tables.geometry().direction(slot, child).unwrap();
                let dir = direction.index();
                let entry = &mut factors[dir * M2L_OFFSET_COUNT + m2l_offset_index(d).unwrap()];
                let shift_factors = entry.get_or_insert_with(|| {
                    tables.factors_f64(direction.rotate(d).map(|x| x as f64), direction)
                });
                shift_add(
                    &mut incoming[dir * w..(dir + 1) * w],
                    &waves[s][dir * w..(dir + 1) * w],
                    shift_factors,
                );
            }
            let mut local = vec![0.0; n];
            let (mut sums, mut frame_local) = (vec![0.0; 4 * tables.mode_count()], vec![0.0; n]);
            for frame in 0..3 {
                let (up, down) = incoming[2 * frame * w..(2 * frame + 2) * w].split_at(w);
                sums_from_waves(tables, up, down, &mut sums);
                tables.local_from_sums(&sums, &mut frame_local);
                tables.add_local_from_frame(frame, &frame_local, &mut local);
            }
            let charges: Vec<([f64; 3], f64)> = row
                .iter()
                .flat_map(|&(s, _)| self.boxes[s].charges.iter().copied())
                .collect();
            let charge_sum: f64 = charges.iter().map(|&(_, q)| q.abs()).sum();
            for _ in 0..4 {
                let x: [f64; 3] = core::array::from_fn(|i| {
                    target.centre[i] + target.radius * (2.0 * rng.next() - 1.0)
                });
                let (pw, exact) = (
                    potential(p, target, &local, x),
                    potential(p, target, &self.reference[t], x),
                );
                let direct_sum: f64 = charges
                    .iter()
                    .map(|&(y, q)| {
                        let r: f64 = (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt();
                        q * target.radius / r
                    })
                    .sum();
                ratio = ratio.max((pw - exact).abs() / (e * charge_sum));
                error = error.max((pw - exact).abs() / charge_sum);
                truncation = truncation.max((exact - direct_sum).abs() / charge_sum);
            }
        }
        Outcome {
            ratio,
            error,
            truncation,
        }
    }
}

/// The choice of [`nodes_for_degree`] (the spike's "matched" rule, Phase 6 T10): for every
/// degree, the fewest λ nodes from s = 2 whose error over the whole of level 3 (512 boxes,
/// 53,352 V pairs) is at most 0.1× the M2L truncation error there, printed as the table,
/// each quadrature checked against its bound. Release:
/// `cargo test -p nd-fmm-tables --release -- --ignored --nocapture level_every_degree`.
#[test]
#[ignore = "release: every degree and quadrature on level 3"]
fn level_every_degree() {
    let mut chosen = Vec::new();
    for p in 0..=MAX_DEGREE + 2 {
        let problem = LevelProblem::new(3, p, 0xabc + p as u64);
        let mut matched = None;
        for s in quadrature::NODES {
            let tables = PlaneWaveTables::<f64>::with_quadrature(p, Quadrature::with_nodes(s));
            let out = problem.errors(&tables);
            println!(
                "p = {p:2}, s = {s:2}: error {:.2e} Σ|q| ({:.3} E), truncation {:.2e} Σ|q|",
                out.error, out.ratio, out.truncation
            );
            if out.error <= 0.1 * out.truncation {
                matched = Some(s);
                break;
            }
        }
        println!("p = {p}: matched s = {matched:?}");
        chosen.push(matched);
    }
    println!("{chosen:?}");
}
