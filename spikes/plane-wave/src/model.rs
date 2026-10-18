//! The cost model of the stop rule (docs/design/optimisation.md §8.2): flops and memory
//! traffic of plane-wave M2L per box, at the throughputs that T2 measured for `Batched`
//! and `Rotation` inside the FMM (fmm-validate/results/phase6-m2l.md §5, §6), on the
//! uniform cube at N = 10⁶ (levels 2–5).
//!
//! # The work per box (plane-wave M2L, f64; real flops)
//!
//! With s λ nodes, Nₖ' α nodes per λ (FMM3D's Nₖ made even), half of them stored,
//! H = Σₖ Nₖ'/2 complex values per direction, and Fₖ' = min(Fₖ, p + 1) Fourier modes:
//!
//! | step | count per box | flops each |
//! | --- | --- | --- |
//! | rotate the multipole to the y and x frames | 2 | 2 B(p), B(p) = Σₙ (2n + 1)² |
//! | multipole → Fourier modes, up and down together | 3 frames | 4 Σₖ Tₖ, Tₖ = Σ_{m < Fₖ'} (p − m + 1) |
//! | Fourier → plane waves | 6 directions | 4 Σₖ (Nₖ'/2)(Fₖ' − 1) |
//! | merge: a source's waves times a shift into an accumulator | shifts / boxes | 8 H |
//! | an accumulator added into a child | adds / boxes | 2 H |
//! | a child's sum times its shift | child shifts / boxes | 6 H |
//! | plane waves → Fourier modes | 6 directions | 4 Σₖ (Nₖ'/2) Fₖ' |
//! | Fourier modes → local, up and down together | 3 frames | 4 Σₖ Tₖ |
//! | rotate the y and x locals back | 2 | 2 B(p) |
//!
//! The counts of shifts, accumulators and child shifts come from the level's
//! enumeration ([`Counts`]); a V pair costs `Batched` 2 (p + 1)⁴ flops and `Rotation`
//! 2 (p + 1)(10p² + 32p + 9)/3 (`RotationTables::m2l`).
//!
//! # Rates
//!
//! The measured time per V pair of `Batched` and the derived one of `Rotation` give
//! their achieved rates, R_B = 2 (p + 1)⁴ / t_B and R_R = flops_R / t_R. Three cases:
//! *optimistic* every plane-wave flop at R_B (the fastest loop the host runs in the
//! FMM), *central* the rotations and Fourier conversions at R_R (short loops like
//! Rotation's) and the shifts at R_B (streaming complex products), *pessimistic* every
//! flop at R_R. `Rotation`'s time per pair: from its evaluation time against `Batched`'s,
//! t_R = t_B + (total_R − total_B) / pairs (every other stage is the same code), checked
//! against the per-pair times §6 lists at p = 8 and 12.
//!
//! # The quadrature
//!
//! Before the spike: the smallest FMM3D table whose measured error
//! (`Quadrature::measured_error`, the worst pair of the region) is at most β times our
//! relative L2 error of φ at p on the cube (β = 0.1 central: errors within about 1.1×
//! today's, the budget of a compressed operator in optimisation.md §3.2; β = 1
//! optimistic: errors up to about 2×). After it: the matched quadrature of the spike's
//! accuracy sweep ([`MATCHED`]), the smallest s whose level error is at most 0.1× the
//! M2L truncation error.
//!
//! # The GPU (one H100)
//!
//! [`gpu_rows`]: the plane waves' flops and traffic on the stage at the device dense
//! M2L's measured efficiency (fmm-validate/results/phase6-device.md §4.1, the best
//! variant's share of the peak on the largest level) and at 70 % of the HBM bandwidth,
//! against the dense M2L's measured stage time. A model only: no plane-wave kernel exists.

use crate::geometry::{Counts, Level};
use crate::quadrature::Quadrature;

/// The machines of the workloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Machine {
    /// Apple M3 Max, one performance core per rank.
    M3Max,
    /// locust, GH200, Neoverse-V2.
    Locust,
    /// Kathleen, Xeon Gold 6248, the AVX2 kernel.
    Kathleen,
}

impl Machine {
    /// The name in the report.
    pub fn name(self) -> &'static str {
        match self {
            Machine::M3Max => "M3 Max",
            Machine::Locust => "locust",
            Machine::Kathleen => "Kathleen",
        }
    }
}

/// One measured configuration of T2 (the cube, N = 10⁶).
#[derive(Clone, Copy, Debug)]
pub struct Measured {
    /// The machine.
    pub machine: Machine,
    /// `true` for f64.
    pub f64: bool,
    /// The degree.
    pub p: usize,
    /// Ranks × threads, as T2 names it.
    pub split: &'static str,
    /// `Batched`'s M2L time per V pair, µs (per rank: one rank per core).
    pub batched: f64,
    /// The fused mode's, µs.
    pub fused: f64,
    /// `Rotation`'s, µs (measured, or derived from the totals).
    pub rotation: f64,
    /// Whether `rotation` was measured per pair (§6) rather than derived (§5).
    pub rotation_measured: bool,
}

/// V pairs of the cube's tree at N = 10⁶ (uniform, levels 2–5), from [`Counts`].
pub fn cube_pairs() -> usize {
    (2..=5).map(|l| Level { level: l }.pairs().len()).sum()
}

/// The evaluation totals (ms) of `Rotation` and `Batched` and `Batched`'s time per pair,
/// fused per pair (µs), one rank at one thread (phase6-m2l.md §5, the cube).
#[rustfmt::skip]
const ONE_CORE: [(Machine, bool, usize, f64, f64, f64, f64); 24] = [
    // machine, f64, p, total Rotation, total Batched, Batched µs/pair, fused µs/pair
    (Machine::M3Max, true, 3, 1_762.0, 983.0, 0.029, 0.021),
    (Machine::M3Max, true, 6, 3_825.0, 2_157.0, 0.190, 0.116),
    (Machine::M3Max, true, 8, 6_324.0, 4_309.0, 0.505, 0.297),
    (Machine::M3Max, true, 12, 15_288.0, 15_240.0, 2.186, 1.310),
    (Machine::M3Max, true, 18, 43_705.0, 80_974.0, 12.633, 8.482),
    (Machine::M3Max, false, 3, 1_311.0, 571.0, 0.019, 0.015),
    (Machine::M3Max, false, 6, 3_373.0, 1_261.0, 0.101, 0.065),
    (Machine::M3Max, false, 8, 5_872.0, 2_432.0, 0.258, 0.158),
    (Machine::Locust, true, 3, 3_132.0, 2_410.0, 0.047, 0.037),
    (Machine::Locust, true, 6, 5_835.0, 3_964.0, 0.263, 0.175),
    (Machine::Locust, true, 8, 9_339.0, 6_689.0, 0.668, 0.438),
    (Machine::Locust, true, 12, 22_163.0, 21_992.0, 3.045, 2.050),
    (Machine::Locust, true, 18, 61_051.0, 155_935.0, 24.736, 20.685),
    (Machine::Locust, false, 3, 1_897.0, 1_011.0, 0.032, 0.027),
    (Machine::Locust, false, 6, 4_416.0, 1_922.0, 0.144, 0.103),
    (Machine::Locust, false, 8, 7_547.0, 3_410.0, 0.348, 0.232),
    (Machine::Kathleen, true, 3, 4_015.0, 2_243.0, 0.045, 0.039),
    (Machine::Kathleen, true, 6, 9_552.0, 3_912.0, 0.235, 0.171),
    (Machine::Kathleen, true, 8, 16_736.0, 7_062.0, 0.659, 0.479),
    (Machine::Kathleen, true, 12, 44_193.0, 30_818.0, 4.238, 3.049),
    (Machine::Kathleen, true, 18, 130_451.0, 182_545.0, 28.247, 20.763),
    (Machine::Kathleen, false, 3, 3_040.0, 1_161.0, 0.032, 0.029),
    (Machine::Kathleen, false, 6, 8_050.0, 2_278.0, 0.135, 0.101),
    (Machine::Kathleen, false, 8, 14_395.0, 4_026.0, 0.335, 0.242),
];

/// Per pair at the node's ranks, one rank per core (phase6-m2l.md §6, the cube, f64):
/// Rotation, Batched, fused (µs).
#[rustfmt::skip]
const NODE: [(Machine, usize, &str, f64, f64, f64); 6] = [
    (Machine::M3Max, 8, "12 × 1", 1.019, 0.567, 0.374),
    (Machine::M3Max, 12, "12 × 1", 2.716, 2.625, 1.807),
    (Machine::Locust, 8, "72 × 1", 1.139, 0.871, 0.734),
    (Machine::Locust, 12, "72 × 1", 4.013, 5.032, 4.092),
    (Machine::Kathleen, 8, "40 × 1", 3.086, 1.116, 0.869),
    (Machine::Kathleen, 12, "40 × 1", 9.435, 6.207, 4.579),
];

/// Rotation per pair measured at one rank, one thread (§6), for the check of the
/// derived values: (machine, p, µs).
pub const ROTATION_ONE_CORE_MEASURED: [(Machine, usize, f64); 6] = [
    (Machine::M3Max, 8, 0.845),
    (Machine::M3Max, 12, 2.226),
    (Machine::Locust, 8, 1.102),
    (Machine::Locust, 12, 3.106),
    (Machine::Kathleen, 8, 2.258),
    (Machine::Kathleen, 12, 6.500),
];

/// Every configuration of the model.
pub fn measured() -> Vec<Measured> {
    let pairs = cube_pairs() as f64;
    let mut out: Vec<Measured> = ONE_CORE
        .iter()
        .map(|&(machine, f64, p, rot, bat, tb, tf)| Measured {
            machine,
            f64,
            p,
            split: "1 × 1",
            batched: tb,
            fused: tf,
            rotation: tb + (rot - bat) * 1e3 / pairs,
            rotation_measured: false,
        })
        .collect();
    out.extend(
        NODE.iter()
            .map(|&(machine, p, split, tr, tb, tf)| Measured {
                machine,
                f64: true,
                p,
                split,
                batched: tb,
                fused: tf,
                rotation: tr,
                rotation_measured: true,
            }),
    );
    out
}

/// Our relative L2 error of φ on the cube at p (f64; f32 is within 5 % at p ≤ 8):
/// phase6-fmm3d.md (N = 10⁶, p ≤ 12) and laplace-fmm-plan §4 (p = 18, the C3.2 tree).
pub fn fmm_error(p: usize) -> f64 {
    match p {
        3 => 2.9e-3,
        6 => 1.3e-4,
        8 => 1.9e-5,
        12 => 6.9e-7,
        18 => 1.0e-8,
        _ => panic!("no measured error at p = {p}"),
    }
}

/// The flops of the steps per box, by step (see the module documentation).
#[derive(Clone, Copy, Debug, Default)]
pub struct Flops {
    /// The four rotations.
    pub rotations: f64,
    /// Multipole → Fourier and Fourier → local.
    pub fourier: f64,
    /// Fourier ↔ plane waves.
    pub physical: f64,
    /// The merge, the additions and the child shifts.
    pub shifts: f64,
}

impl Flops {
    /// The total.
    pub fn total(&self) -> f64 {
        self.rotations + self.fourier + self.physical + self.shifts
    }
}

/// Σₙ (2n + 1)², the multiply-adds of one rotation of degree p.
pub fn rotation_terms(p: usize) -> usize {
    (p + 1) * (2 * p + 1) * (2 * p + 3) / 3
}

/// The flops of one V pair under `Rotation` (`RotationTables::m2l`).
pub fn rotation_pair_flops(p: usize) -> f64 {
    2.0 * ((p + 1) * (10 * p * p + 32 * p + 9)) as f64 / 3.0
}

/// The flops of one V pair under `Batched`.
pub fn batched_pair_flops(p: usize) -> f64 {
    2.0 * ((p + 1) as f64).powi(4)
}

/// The plane-wave flops of a level with counts `c`, degree p and quadrature `q`, per box
/// of the level.
pub fn flops_per_box(c: &Counts, p: usize, q: &Quadrature) -> Flops {
    let s = q.nodes();
    let even = q.even_alpha_nodes();
    let h = q.half_terms() as f64;
    let t: usize = (0..s)
        .map(|k| (0..q.modes(k, p)).map(|m| p - m + 1).sum::<usize>())
        .sum();
    let to_physical: usize = (0..s).map(|k| even[k] / 2 * (q.modes(k, p) - 1)).sum();
    let to_fourier: usize = (0..s).map(|k| even[k] / 2 * q.modes(k, p)).sum();
    let boxes = c.boxes as f64;
    Flops {
        rotations: 4.0 * 2.0 * rotation_terms(p) as f64,
        fourier: 2.0 * 3.0 * 4.0 * t as f64,
        physical: 6.0 * 4.0 * (to_physical + to_fourier) as f64,
        shifts: (8.0 * h * c.shifts as f64
            + 2.0 * h * c.child_adds as f64
            + 6.0 * h * c.child_shifts as f64)
            / boxes,
    }
}

/// The DRAM traffic per box of the plane waves (bytes, f64), at best (each source's
/// waves written once and read once) and at worst (read once per shift), against the
/// tables `Batched` streams per box (each of the 316 tables read once per block of 64
/// targets).
pub fn traffic_per_box(c: &Counts, p: usize, q: &Quadrature) -> (f64, f64, f64) {
    let wave = 16.0 * q.half_terms() as f64;
    let coefficients = 8.0 * ((p + 1) * (p + 1)) as f64;
    let expansions = 3.0 * coefficients; // multipole read, local read and written
    let best = 2.0 * 6.0 * wave + expansions;
    let worst = 6.0 * wave + wave * c.shifts as f64 / c.boxes as f64 + expansions;
    let tables = 316.0 * 8.0 * ((p + 1) as f64).powi(4) / 64.0;
    (best, worst, tables)
}

/// One row of the model: the M2L stage of the cube's tree at N = 10⁶.
#[derive(Clone, Debug)]
pub struct Row {
    /// The configuration.
    pub measured: Measured,
    /// β of the quadrature choice.
    pub beta: f64,
    /// The λ nodes of the chosen quadrature.
    pub s: usize,
    /// H, complex values per direction.
    pub h: usize,
    /// The bar: the faster of `Batched` and `Rotation`, ms, and its name.
    pub bar: (f64, &'static str),
    /// `Batched` fused, ms.
    pub fused: f64,
    /// Plane waves, optimistic, central and pessimistic, ms.
    pub planewave: [f64; 3],
    /// Flops per box on the deepest level.
    pub flops: Flops,
}

impl Row {
    /// Bar over plane waves in the central case.
    pub fn central_gain(&self) -> f64 {
        self.bar.0 / self.planewave[1]
    }
}

/// The matched quadrature per p: the smallest s whose plane-wave level error is at most
/// 0.1× the M2L truncation error (tests/accuracy.rs, `level_3_every_p`).
pub const MATCHED: [(usize, usize); 5] = [(3, 10), (6, 15), (8, 18), (12, 21), (18, 32)];

/// The matched quadrature at p.
pub fn matched(p: usize) -> Quadrature {
    let s = MATCHED
        .iter()
        .find(|&&(q, _)| q == p)
        .expect("a matched p")
        .1;
    Quadrature::with_nodes(s)
}

/// The model's rows for β.
pub fn rows(beta: f64) -> Vec<Row> {
    rows_with(beta, |p| Quadrature::for_error(beta * fmm_error(p)))
}

/// The model's rows with the matched quadrature.
pub fn rows_matched() -> Vec<Row> {
    rows_with(f64::NAN, matched)
}

/// The model's rows with the quadrature `select(p)`; `beta` is recorded only.
pub fn rows_with(beta: f64, select: impl Fn(usize) -> Quadrature) -> Vec<Row> {
    let levels: Vec<Counts> = (2..=5).map(|l| Counts::of(&Level { level: l })).collect();
    let pairs: usize = levels.iter().map(|c| c.pairs).sum();
    measured()
        .into_iter()
        .map(|m| {
            let q = select(m.p);
            let r_b = batched_pair_flops(m.p) / m.batched; // flops per µs
            let r_r = rotation_pair_flops(m.p) / m.rotation;
            let mut pw = [0.0; 3];
            for c in &levels {
                let f = flops_per_box(c, m.p, &q);
                let boxes = c.boxes as f64;
                let conv = f.rotations + f.fourier + f.physical;
                pw[0] += boxes * f.total() / r_b;
                pw[1] += boxes * (conv / r_r + f.shifts / r_b);
                pw[2] += boxes * f.total() / r_r;
            }
            let (batched, rotation) = (pairs as f64 * m.batched, pairs as f64 * m.rotation);
            let bar = if batched <= rotation {
                (batched * 1e-3, "Batched")
            } else {
                (rotation * 1e-3, "Rotation")
            };
            Row {
                measured: m,
                beta,
                s: q.nodes(),
                h: q.half_terms(),
                bar,
                fused: pairs as f64 * m.fused * 1e-3,
                planewave: pw.map(|t| t * 1e-3),
                flops: flops_per_box(levels.last().unwrap(), m.p, &q),
            }
        })
        .collect()
}

/// One GPU row: the dense M2L stage measured on the H100 and the plane waves modelled.
#[derive(Clone, Debug)]
pub struct GpuRow {
    /// `true` for f64.
    pub f64: bool,
    /// The degree.
    pub p: usize,
    /// The dense M2L stage, ms (the best variant of T7).
    pub dense: f64,
    /// The plane waves' flops over the stage.
    pub flops: f64,
    /// Their traffic over the stage, best and worst, bytes.
    pub bytes: (f64, f64),
    /// Their time, compute-bound and memory-bound (worst traffic), ms.
    pub planewave: (f64, f64),
}

/// The device dense M2L stage on the cube at N = 10⁶ (phase6-device.md §4.1, the best
/// variant, summed over the level calls, ms) and its efficiency on the largest level.
#[rustfmt::skip]
const DEVICE: [(bool, usize, f64, f64); 8] = [
    (false, 3, 0.97, 0.064), (false, 6, 3.23, 0.147), (false, 8, 5.80, 0.218),
    (true, 3, 1.24, 0.094), (true, 6, 4.95, 0.184), (true, 8, 10.27, 0.238),
    (true, 12, 40.07, 0.259), (true, 18, 139.31, 0.337),
];

/// The GPU rows with the matched quadrature: peak 67 (f32) and 34 (f64) TFLOP/s, HBM
/// 4.0 TB/s (GH200's H100, 96 GB HBM3), 70 % of it reached.
pub fn gpu_rows() -> Vec<GpuRow> {
    let levels: Vec<Counts> = (2..=5).map(|l| Counts::of(&Level { level: l })).collect();
    DEVICE
        .iter()
        .map(|&(f64, p, dense, efficiency)| {
            let q = matched(p);
            let width = if f64 { 1.0 } else { 0.5 };
            let (mut flops, mut best, mut worst) = (0.0, 0.0, 0.0);
            for c in &levels {
                let b = c.boxes as f64;
                flops += b * flops_per_box(c, p, &q).total();
                let (lo, hi, _) = traffic_per_box(c, p, &q);
                best += b * lo * width;
                worst += b * hi * width;
            }
            let peak = if f64 { 34e12 } else { 67e12 };
            GpuRow {
                f64,
                p,
                dense,
                flops,
                bytes: (best, worst),
                planewave: (
                    flops / (efficiency * peak) * 1e3,
                    worst / (0.7 * 4.0e12) * 1e3,
                ),
            }
        })
        .collect()
}
