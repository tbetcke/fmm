//! S → F and F → R as products per box against the RCR M2L pair on one core (Helmholtz
//! CH4.1a; review 2026-11-19, A5). Prints Markdown on stdout.
//!
//! For each `p@k̂:boxes:pairs` of `--cases` (default: W1's high-frequency levels at kD = 500,
//! levels 3 and 2 at the degrees of 10⁻³, 10⁻⁶ and 10⁻⁸, and at kD = 1000, levels 4, 3 and 2,
//! with the boxes and V pairs of kD = 500's tree, `fmm-validate/results/helmholtz-ch4-0e.md`
//! §3.1):
//!
//! - **the conversions**: `nd_fmm_tables::helmholtz::diagonal::Conversions::{s2f, f2r}` over
//!   a batch of `min(boxes, --batch)` boxes (default 64), the time per box, and the rate of the
//!   real multiply-adds they compute (`Conversions::multiply_adds`, two flops each); the
//!   signatures and locals are checked against the reference
//!   (`nd_fmm_ref::helmholtz::diagonal::{s2f, f2r}`) on the first box before any timing;
//! - **the RCR pair**: `HelmholtzTables::m2l_contribution` in class order into a slot per
//!   pair, then `add_m2l_contribution` in index order, over the 316 offsets of level 2 of
//!   three, the M2L blocks stored within the default budget (`DEFAULT_BLOCK_BUDGET`), as
//!   `nd-fmm-exec`'s RCR level call runs a level by pairs on one thread (`helmholtz_rcr_pair`'s
//!   "pairs"), with two multipole planes (complex charges, §9.1's default) and with one;
//! - **the diagonal pair, a stand-in**: CH4.2's step G_t += T̂_c[π(s)] F_s over every sample,
//!   with π the class permutation that reverses the rings and reflects the azimuths (a gather
//!   of the transfer function), 34 transfer functions, `min(boxes, 16)` source signatures and
//!   8 target accumulators cycled over 316 pairs;
//! - **the level model**: the level on one core with RCR, pairs × the RCR pair (two planes),
//!   against the diagonal form, boxes × (S → F + F → R) + pairs × the diagonal pair (every
//!   pair diagonal: S6's calibrated rule sends no class of real k to RCR at k a ≥ 45 for
//!   ε ≥ 10⁻⁸, `spikes/diagonal/results/summary.md` §3 "real, calibrated"), and the diagonal
//!   form with the φ step free (the ceiling of any FFT's gain, the φ step's share of the
//!   conversions taken from its share of the multiply-adds).
//!
//! Each figure is the median of `--reps` passes (default 3) after one untimed pass. Run in
//! release mode with one BLAS thread, on one core of a quiet machine:
//!
//! ```text
//! OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
//!     cargo run --release -p nd-fmm-validate --example helmholtz_conversions -- --cases 72@31.25:512:12216
//! ```
//!
//! Timings are wall times on this machine, reported and never asserted.

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_math::{Complex, Layout};
use nd_fmm_ref::helmholtz::Level;
use nd_fmm_ref::helmholtz::diagonal::{self, Grid, SignatureWorkspace};
use nd_fmm_tables::geometry::M2L_OFFSET_COUNT;
use nd_fmm_tables::helmholtz::diagonal::{ConversionScratch, Conversions};
use nd_fmm_tables::helmholtz::{DEFAULT_BLOCK_BUDGET, HelmholtzTables, add_m2l_contribution};
use nd_fmm_validate::SplitMix64;
use nd_fmm_validate::bench::{cores, cpu_model, target};

/// The level of the RCR tables (levels 0 and 1 have no V list).
const LEVEL: usize = 2;

/// The reflection classes of a level (proposal §5.4).
const CLASSES: usize = 34;

/// One case: the degree, k̂ = k r_l of the level, and the level's boxes and V pairs.
#[derive(Clone, Copy)]
struct Case {
    p: usize,
    k_hat: f64,
    boxes: usize,
    pairs: usize,
}

struct Arguments {
    cases: Vec<Case>,
    batch: usize,
    reps: usize,
}

fn default_cases() -> Vec<Case> {
    let case = |p, k_hat, boxes, pairs| Case {
        p,
        k_hat,
        boxes,
        pairs,
    };
    let mut cases = Vec::new();
    // kD = 500: level 3 (k a 54.1) and level 2 (k a 108.3) at 10⁻³, 10⁻⁶, 10⁻⁸.
    for p in [66, 72, 76] {
        cases.push(case(p, 31.25, 512, 12_216));
    }
    for p in [123, 131, 135] {
        cases.push(case(p, 62.5, 64, 2_504));
    }
    // kD = 1000 at 10⁻⁶: levels 4, 3, 2 (k a 54.1, 108.3, 216.5), kD = 500's tree.
    cases.push(case(72, 31.25, 3_840, 54_000));
    cases.push(case(131, 62.5, 512, 12_216));
    cases.push(case(245, 125.0, 64, 2_504));
    cases
}

fn arguments() -> Arguments {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || -> ! {
        eprintln!(
            "usage: helmholtz_conversions [--cases p@k̂:boxes:pairs,…] [--batch n] [--reps n]; \
             got {args:?}"
        );
        std::process::exit(2);
    };
    let mut parsed = Arguments {
        cases: default_cases(),
        batch: 64,
        reps: 3,
    };
    let positive = |value: &str| value.parse::<usize>().ok().filter(|&n| n > 0);
    for pair in args.chunks(2) {
        let [flag, value] = pair else { usage() };
        let ok = match flag.as_str() {
            "--cases" => value
                .split(',')
                .map(|c| {
                    let (p, rest) = c.split_once('@')?;
                    let mut parts = rest.split(':');
                    let k_hat = parts.next()?.parse::<f64>().ok().filter(|&k| k > 0.0)?;
                    let boxes = positive(parts.next()?)?;
                    let pairs = positive(parts.next()?)?;
                    Some(Case {
                        p: p.parse().ok()?,
                        k_hat,
                        boxes,
                        pairs,
                    })
                })
                .collect::<Option<Vec<_>>>()
                .map(|v| parsed.cases = v)
                .is_some(),
            "--batch" => positive(value).map(|n| parsed.batch = n).is_some(),
            "--reps" => positive(value).map(|n| parsed.reps = n).is_some(),
            _ => false,
        };
        if !ok {
            usage();
        }
    }
    parsed
}

/// The median over `reps` of `pass()`'s seconds, after one untimed pass.
fn median(reps: usize, mut pass: impl FnMut()) -> f64 {
    pass();
    let mut times: Vec<f64> = (0..reps)
        .map(|_| {
            let start = Instant::now();
            pass();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    times[reps / 2]
}

/// Seconds per pair of the FMM's RCR pair path on `planes` multipole planes.
fn rcr_pair(tables: &HelmholtzTables<f64>, p: usize, planes: usize, reps: usize) -> f64 {
    let len = Layout::new(p).len();
    let mut rng = SplitMix64::new(p as u64 * 31 + planes as u64);
    let multipoles: Vec<Vec<f64>> = (0..M2L_OFFSET_COUNT)
        .map(|_| (0..planes * len).map(|_| rng.uniform() - 0.5).collect())
        .collect();
    let mut locals = vec![vec![0.0f64; 2 * len]; M2L_OFFSET_COUNT];
    let mut class_order: Vec<usize> = (0..M2L_OFFSET_COUNT).collect();
    class_order.sort_by_key(|&d| tables.m2l_classes(d));
    let size = tables.m2l_contribution_len(LEVEL, 1);
    let mut slots = vec![0.0f64; M2L_OFFSET_COUNT * size];
    let mut scratch = tables.scratch();
    let seconds = median(reps, || {
        for &d in &class_order {
            tables.m2l_contribution(
                LEVEL,
                d,
                &multipoles[d],
                planes,
                &mut slots[d * size..(d + 1) * size],
                1,
                &mut scratch,
            );
        }
        for (d, local) in locals.iter_mut().enumerate() {
            add_m2l_contribution(&slots[d * size..(d + 1) * size], local);
        }
    });
    black_box(&locals);
    seconds / M2L_OFFSET_COUNT as f64
}

/// Seconds per pair of the stand-in diagonal step (module documentation).
fn diagonal_pair(conversions: &Conversions<f64>, sources: usize, reps: usize) -> f64 {
    let (p, azimuths, samples) = (
        conversions.degree(),
        conversions.azimuths(),
        conversions.samples(),
    );
    let rings = 2 * p + 1;
    let permutation: Vec<u32> = (0..samples)
        .map(|s| {
            let (i, j) = (s / azimuths, s % azimuths);
            ((rings - 1 - i) * azimuths + (azimuths - j) % azimuths) as u32
        })
        .collect();
    let mut rng = SplitMix64::new(p as u64);
    let mut signature = || -> Vec<Complex<f64>> {
        (0..samples)
            .map(|_| Complex::new(rng.uniform() - 0.5, rng.uniform() - 0.5))
            .collect()
    };
    let transfers: Vec<Vec<Complex<f64>>> = (0..CLASSES).map(|_| signature()).collect();
    let signatures: Vec<Vec<Complex<f64>>> = (0..sources).map(|_| signature()).collect();
    let mut targets = vec![vec![Complex::new(0.0, 0.0); samples]; 8];
    let seconds = median(reps, || {
        for pair in 0..M2L_OFFSET_COUNT {
            let (t, f) = (&transfers[pair % CLASSES], &signatures[pair % sources]);
            let g = &mut targets[pair % 8];
            for ((g, &f), &at) in g.iter_mut().zip(f).zip(&permutation) {
                *g += t[at as usize] * f;
            }
        }
    });
    black_box(&targets);
    seconds / M2L_OFFSET_COUNT as f64
}

/// The largest difference of a box's results from the reference's, over the reference's
/// largest value: S → F and F → R.
fn check(conversions: &Conversions<f64>, k_hat: f64, multipole: &[f64]) -> (f64, f64) {
    let p = conversions.degree();
    let level = Level::new(Complex::new(k_hat, 0.0), 1.0);
    assert_eq!(level.scale(), 1.0, "a high-frequency level has scale 1");
    let grid = Grid::<f64>::new(p);
    let mut ws = SignatureWorkspace::new(p);
    let mut scratch = ConversionScratch::new(conversions);
    let samples = conversions.samples();
    let mut ours = vec![Complex::new(0.0, 0.0); samples];
    conversions.s2f(multipole, &mut ours, &mut scratch);
    let mut reference = vec![Complex::new(0.0, 0.0); samples];
    diagonal::s2f(&grid, &level, multipole, &mut ws, &mut reference);
    let relative = |error: f64, scale: f64| error / scale;
    let largest = |it: &mut dyn Iterator<Item = f64>| it.fold(0.0f64, f64::max);
    let s2f = relative(
        largest(&mut ours.iter().zip(&reference).map(|(a, b)| (a - b).norm())),
        largest(&mut reference.iter().map(|z| z.norm())),
    );
    let mut local = vec![0.0; conversions.expansion_len()];
    conversions.f2r(&reference, &mut local, &mut scratch);
    let mut expected = vec![0.0; conversions.expansion_len()];
    diagonal::f2r(&grid, &level, &reference, &mut ws, &mut expected);
    let f2r = relative(
        largest(&mut local.iter().zip(&expected).map(|(a, b)| (a - b).abs())),
        largest(&mut expected.iter().map(|v| v.abs())),
    );
    (s2f, f2r)
}

fn main() {
    let arguments = arguments();
    println!("# Helmholtz S → F and F → R as products against the RCR pair (CH4.1a)");
    println!();
    println!(
        "Machine: {}, {}; target {}; one thread; median of {} passes; conversions over a batch \
         of up to {} boxes.",
        cpu_model(),
        cores(),
        target(),
        arguments.reps,
        arguments.batch
    );
    println!();
    println!(
        "| p | k̂ | N_φ | S → F ms | F → R ms | GF/s | rel. err. S → F, F → R | RCR pair ms (2 planes) \
         | RCR pair ms (1 plane) | diagonal pair ms | S → F + F → R ÷ RCR pair | φ share \
         | boxes | pairs | RCR level s | diagonal level s | diagonal ÷ RCR | φ free s | φ free ÷ RCR |"
    );
    println!(
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: \
         | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for &Case {
        p,
        k_hat,
        boxes,
        pairs,
    } in &arguments.cases
    {
        let conversions = Conversions::<f64>::new(p, 1.0);
        let mut scratch = ConversionScratch::new(&conversions);
        let (len, samples) = (conversions.expansion_len(), conversions.samples());
        let batch = boxes.min(arguments.batch);
        let mut rng = SplitMix64::new(p as u64 * 7 + 1);
        let multipoles: Vec<f64> = (0..batch * len).map(|_| rng.uniform() - 0.5).collect();
        let (s2f_error, f2r_error) = check(&conversions, k_hat, &multipoles[..len]);
        let mut signatures = vec![Complex::new(0.0, 0.0); batch * samples];
        let mut locals = vec![0.0; batch * len];
        let s2f = median(arguments.reps, || {
            conversions.s2f(&multipoles, &mut signatures, &mut scratch)
        }) / batch as f64;
        let f2r = median(arguments.reps, || {
            conversions.f2r(&signatures, &mut locals, &mut scratch)
        }) / batch as f64;
        black_box(&locals);
        let (legendre, phi) = conversions.multiply_adds();
        let rate = 2.0 * (legendre + phi) as f64 * 2.0 / (s2f + f2r) / 1e9;
        let phi_share = phi as f64 / (legendre + phi) as f64;

        let k_root = Complex::new(4.0 * k_hat, 0.0);
        let tables = HelmholtzTables::<f64>::build(k_root, &[0, 0, p], DEFAULT_BLOCK_BUDGET);
        let rcr2 = rcr_pair(&tables, p, 2, arguments.reps);
        let rcr1 = rcr_pair(&tables, p, 1, arguments.reps);
        drop(tables);
        let diag = diagonal_pair(&conversions, boxes.min(16), arguments.reps);

        let rcr_level = pairs as f64 * rcr2;
        let diagonal_level = boxes as f64 * (s2f + f2r) + pairs as f64 * diag;
        let phi_free = boxes as f64 * (s2f + f2r) * (1.0 - phi_share) + pairs as f64 * diag;
        println!(
            "| {p} | {k_hat} | {} | {:.3} | {:.3} | {rate:.2} | {s2f_error:.1e}, {f2r_error:.1e} \
             | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {boxes} | {pairs} | {rcr_level:.3} \
             | {diagonal_level:.3} | {:.3} | {phi_free:.3} | {:.3} |",
            conversions.azimuths(),
            1e3 * s2f,
            1e3 * f2r,
            1e3 * rcr2,
            1e3 * rcr1,
            1e3 * diag,
            (s2f + f2r) / rcr2,
            phi_share,
            diagonal_level / rcr_level,
            phi_free / rcr_level,
        );
    }
}
