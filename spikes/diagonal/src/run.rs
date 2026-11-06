//! One unit of the sweep: every class, placement and variant at one (kind, k a, p).
//!
//! Level 0 of S7's chains: boxes of width 1, the source box S at the origin, the target
//! box at a class representative t. Per placement (S7's three), S's 128 sources and the
//! target box's 64 targets are S7's C1 clouds (the same seeds), and the multipole of S is
//! S7's P2M in double-double, unscaled. Per class:
//!
//! - the reference local: S7's RCR in double-double (rotation, (S|R), rotation back) at
//!   degree p, the truncated M2L;
//! - the direct sum in double-double;
//! - the diagonal chain in each [`Variant`]: the multipole and T rounded to the working
//!   precision, S → F, T F and F → R in it, the local widened back to double-double;
//! - every local evaluated by S7's L2P in double-double, so that the errors are the M2L's.
//!
//! The errors are S7's relative ℓ2 norms over the targets, of the potential and the
//! gradient (proposal §9.1): `trunc` RCR against the direct sum (the truncation),
//! `round` the diagonal against RCR (its rounding), `total` the diagonal against the
//! direct sum.

use std::any::Any;
use std::time::Instant;

use nd_fmm_spike_truncation::calibrate::parallel_map;
use nd_fmm_spike_truncation::chain::{self, Field, Kind, Setup, Work, e2p, p2e, rcr};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::geometry::{self, Placement, Rng, dist2};
use nd_fmm_spike_truncation::harmonics::{Harmonics, Polar, idx, len};
use nd_fmm_spike_truncation::radial;
use nd_fmm_spike_truncation::real::{Dd, Real};
use nd_fmm_spike_truncation::rotation::{Direction, Rotation};

use crate::classes::{Class, classes};
use crate::grid::{Grid, round};
use crate::signature::{f2r_ring, finish, s2f_ring};
use crate::transfer::{Interp, Transfer, TransferF64, angle, log10_term};

/// The working precision of a variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// f64, T rounded at every sample.
    F64,
    /// f32, T rounded at every sample.
    F32,
    /// f64, T interpolated in ŝ · t̂ ([`Interp`]).
    F64Interp,
    /// f64, T built in f64 ([`TransferF64`]): the terms rounded, the sum in f64.
    F64Built,
}

/// The bandwidth L of T.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    /// L = P − 1 = p, the signatures' bandwidth.
    Signature,
    /// L = ⌈3(P − 1)/2⌉.
    Middle,
    /// L = 2P − 2 = 2p, the proposal's: the diagonal form is then the truncated M2L.
    Full,
}

impl Band {
    /// The three, ascending.
    pub const ALL: [Band; 3] = [Band::Signature, Band::Middle, Band::Full];

    /// L at degree p.
    pub fn l(self, p: usize) -> usize {
        match self {
            Band::Signature => p,
            Band::Middle => (3 * p).div_ceil(2),
            Band::Full => 2 * p,
        }
    }

    /// A short name.
    pub fn name(self) -> &'static str {
        match self {
            Band::Signature => "P-1",
            Band::Middle => "3(P-1)/2",
            Band::Full => "2P-2",
        }
    }
}

/// A precision and a bandwidth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    /// The working precision.
    pub precision: Precision,
    /// The bandwidth of T.
    pub band: Band,
}

impl Variant {
    /// A short name.
    pub fn name(self) -> String {
        let p = match self.precision {
            Precision::F64 => "f64",
            Precision::F32 => "f32",
            Precision::F64Interp => "f64i",
            Precision::F64Built => "f64b",
        };
        format!("{p}:{}", self.band.name())
    }
}

/// The variants of a unit: f64 and f32 at the three bandwidths; at 2P − 2 also f64 with T
/// interpolated and f64 with T built in f64.
pub const VARIANTS: [Variant; 8] = [
    Variant {
        precision: Precision::F64,
        band: Band::Signature,
    },
    Variant {
        precision: Precision::F64,
        band: Band::Middle,
    },
    Variant {
        precision: Precision::F64,
        band: Band::Full,
    },
    Variant {
        precision: Precision::F32,
        band: Band::Signature,
    },
    Variant {
        precision: Precision::F32,
        band: Band::Middle,
    },
    Variant {
        precision: Precision::F32,
        band: Band::Full,
    },
    Variant {
        precision: Precision::F64Interp,
        band: Band::Full,
    },
    Variant {
        precision: Precision::F64Built,
        band: Band::Full,
    },
];

/// Points, seeds and limits.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Sources per box (S7's 128).
    pub n_src: usize,
    /// Targets per box (S7's 64).
    pub n_tgt: usize,
    /// The seed of the placements and charges (S7's 2026).
    pub seed: u64,
    /// Worker threads.
    pub threads: usize,
    /// Oversampling σ of the interpolation table.
    pub sigma: f64,
    /// Terms on each side of an interpolated sample.
    pub half: usize,
    /// A bandwidth whose log₁₀ |k||t| max (2j + 1)|hⱼ| exceeds this is not run (T would
    /// leave the double-double exponent range); its errors are recorded as infinite.
    pub max_log10_term: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_src: 128,
            n_tgt: 64,
            seed: 2026,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
            sigma: 3.0,
            half: 36,
            max_log10_term: 280.0,
        }
    }
}

/// S7's seed of a cloud (`calibrate::seed`, copied): `what` 1 for S's sources, 4 for the
/// target box at `offset`.
fn seed(cfg: &Config, placement: Placement, what: u64, offset: [i32; 3]) -> Rng {
    let p = placement as u64;
    let o = ((offset[0] + 8) * 256 + (offset[1] + 8) * 16 + (offset[2] + 8)) as u64;
    Rng::new(cfg.seed ^ (p << 40) ^ (what << 32) ^ o.wrapping_mul(0x2545_F491_4F6C_DD1D))
}

/// The points of a C1 cloud of S7 (and its charges) in the box at `c`.
pub fn cloud(
    cfg: &Config,
    placement: Placement,
    what: u64,
    offset: [i32; 3],
    n: usize,
) -> (Vec<[Dd; 3]>, Vec<Complex<Dd>>) {
    let mut rng = seed(cfg, placement, what, offset);
    let c = offset.map(|x| x as f64);
    let raw = geometry::points(placement, c, 1.0, 1.0, n, &mut rng);
    let q = geometry::charges::<Dd>(n, &mut rng);
    (geometry::to_real(&raw), q)
}

/// A complex double-double rounded to W.
#[inline]
pub fn narrow<W: Real>(z: Complex<Dd>) -> Complex<W> {
    Complex::new(round(z.re), round(z.im))
}

/// A complex W widened to double-double, exactly.
#[inline]
pub fn widen<W: Real>(z: Complex<W>) -> Complex<Dd> {
    if let Some(d) = (&z as &dyn Any).downcast_ref::<Complex<Dd>>() {
        return *d;
    }
    Complex::new(Dd::from_f64(z.re.to_f64()), Dd::from_f64(z.im.to_f64()))
}

/// The diagonal M2L of one multipole (degree p, double-double) in working precision W
/// with T at `tr`'s snapshot `si` rounded from double-double, every step in W: the local,
/// widened.
pub fn diagonal<W: Real>(
    grid: &Grid,
    tr: &Transfer,
    si: usize,
    m: &[Complex<Dd>],
) -> Vec<Complex<Dd>> {
    diagonal_with(grid, m, |i, j| {
        let mut tv = vec![Complex::ZERO; tr.snaps.len()];
        tr.eval(tr.mu(grid.direction(i, j)), &mut tv);
        narrow::<W>(tv[si])
    })
}

/// The diagonal M2L in f64 with T built in f64 ([`TransferF64`]).
pub fn diagonal_f64_built(grid: &Grid, tr: &TransferF64, m: &[Complex<Dd>]) -> Vec<Complex<Dd>> {
    diagonal_with(grid, m, |i, j| {
        tr.eval(grid.direction(i, j).map(round::<f64>))
    })
}

/// The diagonal M2L in W with T(sample i, j) from `t`.
pub fn diagonal_with<W: Real>(
    grid: &Grid,
    m: &[Complex<Dd>],
    t: impl Fn(usize, usize) -> Complex<W>,
) -> Vec<Complex<Dd>> {
    let p = grid.p;
    let nphi = grid.nphi;
    let tab = grid.tables::<W>();
    let mw: Vec<Complex<W>> = m.iter().map(|&z| narrow(z)).collect();
    let mut harm = Harmonics::<Dd>::new(p);
    let mut theta = Vec::new();
    let mut ab = vec![Complex::<W>::ZERO; 2 * (p + 1)];
    let mut f = vec![Complex::<W>::ZERO; nphi];
    let mut acc = vec![Complex::<W>::ZERO; len(p)];
    for i in 0..grid.rings() {
        grid.theta::<W>(i, &mut harm, &mut theta);
        s2f_ring(p, &theta, &mw, &tab, &mut ab, &mut f);
        for (j, fj) in f.iter_mut().enumerate() {
            *fj = t(i, j) * *fj;
        }
        f2r_ring(p, &theta, tab.w[i], &f, &tab, &mut ab, &mut acc);
    }
    finish(p, &mut acc);
    acc.into_iter().map(widen).collect()
}

/// The results of one class.
#[derive(Clone, Debug)]
pub struct ClassResult {
    /// Index into [`classes`].
    pub class: usize,
    /// The class.
    pub info: Class,
    /// log₁₀ |k||t| maxⱼ≤L (2j + 1)|hⱼ(k|t|)| per [`Band::ALL`].
    pub log10_term: [f64; 3],
    /// log₁₀ of max |T_L| over the samples per band (NaN where not run).
    pub log10_max_t: [f64; 3],
    /// Per placement: RCR against the direct sum (potential, gradient).
    pub trunc: Vec<(f64, f64)>,
    /// Per placement and [`VARIANTS`]: the diagonal against RCR.
    pub round: Vec<[(f64, f64); 8]>,
    /// Per placement and [`VARIANTS`]: the diagonal against the direct sum.
    pub total: Vec<[(f64, f64); 8]>,
    /// Wall time of the class, seconds.
    pub seconds: f64,
}

/// The results of a unit.
#[derive(Clone, Debug)]
pub struct UnitResult {
    /// The kind of k.
    pub kind: Kind,
    /// κ a at the M2L level.
    pub ka: f64,
    /// The degree.
    pub p: usize,
    /// Per class run.
    pub classes: Vec<ClassResult>,
    /// Wall time, seconds.
    pub seconds: f64,
}

/// Per placement: the sources, the double-double multipole and the signatures in f64 and
/// f32 (ring after ring, N_φ samples each).
struct Shared {
    sources: Vec<(Vec<[Dd; 3]>, Vec<Complex<Dd>>)>,
    multipoles: Vec<Vec<Complex<Dd>>>,
    sig64: Vec<Vec<Complex<f64>>>,
    sig32: Vec<Vec<Complex<f32>>>,
}

/// Runs the classes `class_ids` (indices into [`classes`]) at (kind, k a, p).
pub fn run_unit(kind: Kind, ka: f64, p: usize, class_ids: &[usize], cfg: &Config) -> UnitResult {
    assert!(p >= 2, "the three bandwidths need p ≥ 2");
    let start = Instant::now();
    let mut setup = Setup::<Dd>::new(kind, ka);
    // Unscaled expansions: k a ≥ 2.8 here, where S7's scale is near 1 and the degrees
    // stay far inside the exponent range.
    setup.s = [Dd::ONE; 3];
    let grid = Grid::new(p);
    let nphi = grid.nphi;
    let placements = Placement::ALL;
    let sources: Vec<_> = placements
        .iter()
        .map(|&pl| cloud(cfg, pl, 1, [0; 3], cfg.n_src))
        .collect();
    let multipoles: Vec<Vec<Complex<Dd>>> = sources
        .iter()
        .map(|(pts, q)| p2e(true, setup.k, Dd::ONE, [Dd::ZERO; 3], pts, q, p))
        .collect();
    let tab64 = grid.tables::<f64>();
    let tab32 = grid.tables::<f32>();
    let m64: Vec<Vec<Complex<f64>>> = multipoles
        .iter()
        .map(|m| m.iter().map(|&z| narrow(z)).collect())
        .collect();
    let m32: Vec<Vec<Complex<f32>>> = multipoles
        .iter()
        .map(|m| m.iter().map(|&z| narrow(z)).collect())
        .collect();
    let rings: Vec<usize> = (0..grid.rings()).collect();
    type RingSig = (Vec<Vec<Complex<f64>>>, Vec<Vec<Complex<f32>>>);
    let per_ring: Vec<RingSig> = parallel_map(&rings, cfg.threads, |&i| {
        let mut harm = Harmonics::<Dd>::new(p);
        grid.eval_ring(i, &mut harm);
        let (mut th64, mut th32) = (Vec::new(), Vec::new());
        grid.theta_of::<f64>(&harm, &mut th64);
        grid.theta_of::<f32>(&harm, &mut th32);
        let mut ab64 = vec![Complex::ZERO; 2 * (p + 1)];
        let mut ab32 = vec![Complex::ZERO; 2 * (p + 1)];
        let mut out = (Vec::new(), Vec::new());
        for pl in 0..placements.len() {
            let mut f64s = vec![Complex::ZERO; nphi];
            s2f_ring(p, &th64, &m64[pl], &tab64, &mut ab64, &mut f64s);
            let mut f32s = vec![Complex::ZERO; nphi];
            s2f_ring(p, &th32, &m32[pl], &tab32, &mut ab32, &mut f32s);
            out.0.push(f64s);
            out.1.push(f32s);
        }
        out
    });
    let mut sig64 = vec![Vec::with_capacity(grid.samples()); placements.len()];
    let mut sig32 = vec![Vec::with_capacity(grid.samples()); placements.len()];
    for (r64, r32) in per_ring {
        for pl in 0..placements.len() {
            sig64[pl].extend_from_slice(&r64[pl]);
            sig32[pl].extend_from_slice(&r32[pl]);
        }
    }
    let shared = Shared {
        sources,
        multipoles,
        sig64,
        sig32,
    };
    let all = classes();
    let results = parallel_map(class_ids, cfg.threads, |&ci| {
        run_class(&setup, &grid, &shared, ci, all[ci], cfg)
    });
    UnitResult {
        kind,
        ka,
        p,
        classes: results,
        seconds: start.elapsed().as_secs_f64(),
    }
}

/// The accumulators of one placement: per band f64 and f32, f64 interpolated and f64
/// built.
struct Acc {
    a64: Vec<Vec<Complex<f64>>>,
    a32: Vec<Vec<Complex<f32>>>,
    ai: Vec<Complex<f64>>,
    ab: Vec<Complex<f64>>,
}

fn run_class(
    setup: &Setup<Dd>,
    grid: &Grid,
    shared: &Shared,
    ci: usize,
    info: Class,
    cfg: &Config,
) -> ClassResult {
    let start = Instant::now();
    let p = grid.p;
    let nphi = grid.nphi;
    let k = setup.k;
    let rep = info.rep;
    let c_t: [Dd; 3] = rep.map(|x| Dd::from_f64(x as f64));
    let np = shared.sources.len();
    // The reference: S7's RCR and the direct sum, in double-double.
    let rot = Rotation::new(Direction::new(c_t), p);
    let table = setup.m2l_table(dist2(rep), p);
    let mut work = Work::new(p);
    let mut local = vec![Complex::ZERO; len(p)];
    let mut targets = Vec::with_capacity(np);
    let mut exact = Vec::with_capacity(np);
    let mut reference: Vec<Vec<Field<Dd>>> = Vec::with_capacity(np);
    let mut trunc = Vec::with_capacity(np);
    for (pl, &placement) in Placement::ALL.iter().enumerate() {
        let (tgt, _) = cloud(cfg, placement, 4, rep, cfg.n_tgt);
        let (src, q) = &shared.sources[pl];
        let ex = chain::direct(k, src, q, &tgt);
        rcr(
            &rot,
            &table,
            p,
            p,
            None,
            &shared.multipoles[pl],
            &mut local,
            &mut work,
        );
        let mut fields = Vec::new();
        e2p(
            true,
            k,
            Dd::ONE,
            c_t,
            &local,
            p,
            &tgt,
            &mut work,
            &mut fields,
        );
        trunc.push(chain::errors(&fields, &ex));
        targets.push(tgt);
        exact.push(ex);
        reference.push(fields);
    }
    // The bound's term per band, and the bands that stay in range.
    let kt = k.to_f64() * (dist2(rep) as f64).sqrt();
    let log10_term = Band::ALL.map(|b| log10_term(kt, b.l(p)));
    let ok: Vec<bool> = log10_term
        .iter()
        .map(|&v| v <= cfg.max_log10_term)
        .collect();
    let snaps: Vec<usize> = Band::ALL
        .iter()
        .zip(&ok)
        .filter(|(_, ok)| **ok)
        .map(|(b, _)| b.l(p))
        .collect();
    let nb = snaps.len();
    let inf = (f64::INFINITY, f64::INFINITY);
    let mut round_err = vec![[inf; 8]; np];
    let mut total_err = vec![[inf; 8]; np];
    let mut log10_max_t = [f64::NAN; 3];
    if nb > 0 {
        let tr = Transfer::new(k, rep, &snaps);
        let full_ok = ok[2];
        let interp = full_ok.then(|| Interp::new(&tr, nb - 1, cfg.sigma, cfg.half));
        let built = full_ok.then(|| TransferF64::new(&tr, nb - 1));
        let tab64 = grid.tables::<f64>();
        let tab32 = grid.tables::<f32>();
        let mut acc: Vec<Acc> = (0..np)
            .map(|_| Acc {
                a64: vec![vec![Complex::ZERO; len(p)]; nb],
                a32: vec![vec![Complex::ZERO; len(p)]; nb],
                ai: vec![Complex::ZERO; len(p)],
                ab: vec![Complex::ZERO; len(p)],
            })
            .collect();
        let mut harm = Harmonics::<Dd>::new(p);
        let (mut th64, mut th32) = (Vec::new(), Vec::new());
        let mut tv = vec![Complex::ZERO; nb];
        let mut t64 = vec![vec![Complex::<f64>::ZERO; nphi]; nb];
        let mut t32 = vec![vec![Complex::<f32>::ZERO; nphi]; nb];
        let mut ti = vec![Complex::<f64>::ZERO; nphi];
        let mut tb = vec![Complex::<f64>::ZERO; nphi];
        let mut g64 = vec![Complex::<f64>::ZERO; nphi];
        let mut g32 = vec![Complex::<f32>::ZERO; nphi];
        let mut cs64 = vec![Complex::<f64>::ZERO; 2 * (p + 1)];
        let mut cs32 = vec![Complex::<f32>::ZERO; 2 * (p + 1)];
        let mut max_t = vec![0.0f64; nb];
        for i in 0..grid.rings() {
            grid.eval_ring(i, &mut harm);
            grid.theta_of::<f64>(&harm, &mut th64);
            grid.theta_of::<f32>(&harm, &mut th32);
            for j in 0..nphi {
                let mu = tr.mu(grid.direction(i, j));
                tr.eval(mu, &mut tv);
                for b in 0..nb {
                    t64[b][j] = narrow(tv[b]);
                    t32[b][j] = narrow(tv[b]);
                    max_t[b] = max_t[b].max(t64[b][j].abs());
                }
                if let Some(interp) = &interp {
                    ti[j] = interp.eval(angle(grid.direction(i, j), tr.that));
                }
                if let Some(built) = &built {
                    tb[j] = built.eval(grid.direction(i, j).map(round::<f64>));
                }
            }
            let range = i * nphi..(i + 1) * nphi;
            for (pl, a) in acc.iter_mut().enumerate() {
                let f64s = &shared.sig64[pl][range.clone()];
                let f32s = &shared.sig32[pl][range.clone()];
                for b in 0..nb {
                    for j in 0..nphi {
                        g64[j] = t64[b][j] * f64s[j];
                        g32[j] = t32[b][j] * f32s[j];
                    }
                    f2r_ring(p, &th64, tab64.w[i], &g64, &tab64, &mut cs64, &mut a.a64[b]);
                    f2r_ring(p, &th32, tab32.w[i], &g32, &tab32, &mut cs32, &mut a.a32[b]);
                }
                if interp.is_some() {
                    for j in 0..nphi {
                        g64[j] = ti[j] * f64s[j];
                    }
                    f2r_ring(p, &th64, tab64.w[i], &g64, &tab64, &mut cs64, &mut a.ai);
                    for j in 0..nphi {
                        g64[j] = tb[j] * f64s[j];
                    }
                    f2r_ring(p, &th64, tab64.w[i], &g64, &tab64, &mut cs64, &mut a.ab);
                }
            }
        }
        // Band index b of the run snapshots → index into Band::ALL.
        let band_of: Vec<usize> = (0..3).filter(|&b| ok[b]).collect();
        for (b, &bi) in band_of.iter().enumerate() {
            log10_max_t[bi] = max_t[b].log10();
        }
        let mut locals: Vec<Vec<(usize, Vec<Complex<Dd>>)>> = vec![Vec::new(); np];
        for (pl, a) in acc.iter_mut().enumerate() {
            let mut put = |v: usize, acc: &mut [Complex<_>]| {
                finish(p, acc);
                locals[pl].push((v, acc.iter().map(|&z| widen(z)).collect()));
            };
            for (b, &bi) in band_of.iter().enumerate() {
                put(bi, &mut a.a64[b]);
            }
            if interp.is_some() {
                put(6, &mut a.ai);
                put(7, &mut a.ab);
            }
            for (b, &bi) in band_of.iter().enumerate() {
                finish(p, &mut a.a32[b]);
                let l: Vec<Complex<Dd>> = a.a32[b].iter().map(|&z| widen(z)).collect();
                locals[pl].push((3 + bi, l));
            }
        }
        for (pl, ls) in locals.iter().enumerate() {
            let views: Vec<&[Complex<Dd>]> = ls.iter().map(|(_, l)| &l[..]).collect();
            let fields = l2p_many(k, c_t, &views, p, &targets[pl]);
            for ((v, _), f) in ls.iter().zip(&fields) {
                round_err[pl][*v] = chain::errors(f, &reference[pl]);
                total_err[pl][*v] = chain::errors(f, &exact[pl]);
            }
        }
    }
    ClassResult {
        class: ci,
        info,
        log10_term,
        log10_max_t,
        trunc,
        round: round_err,
        total: total_err,
        seconds: start.elapsed().as_secs_f64(),
    }
}

/// S7's L2P (`chain::e2p`, regular, unscaled) of several locals of degree p about c at the
/// targets, the basis evaluated once per target: per local the potential and gradient.
pub fn l2p_many(
    k: Complex<Dd>,
    c: [Dd; 3],
    locals: &[&[Complex<Dd>]],
    p: usize,
    targets: &[[Dd; 3]],
) -> Vec<Vec<Field<Dd>>> {
    let mut harm = Harmonics::<Dd>::new(p);
    let (mut rv, mut rd, mut rw) = (
        vec![Complex::ZERO; p + 1],
        vec![Complex::ZERO; p + 1],
        vec![Complex::ZERO; p + 2],
    );
    let ik = Complex::I * k;
    let mut out = vec![Vec::with_capacity(targets.len()); locals.len()];
    for x in targets {
        let polar = Polar::new([x[0] - c[0], x[1] - c[1], x[2] - c[2]]);
        radial::with_derivative(true, k * polar.r, Dd::ONE, &mut rv, &mut rd, &mut rw);
        harm.eval(&polar, true);
        let [rh, th, ph] = polar.frame();
        let inv_r = Dd::ONE / polar.r;
        for (e, o) in locals.iter().zip(out.iter_mut()) {
            let (mut phi, mut ar, mut at, mut ap) =
                (Complex::ZERO, Complex::ZERO, Complex::ZERO, Complex::ZERO);
            for n in 0..=p {
                let (mut sy, mut st, mut sp) = (Complex::ZERO, Complex::ZERO, Complex::ZERO);
                for m in -(n as isize)..=n as isize {
                    let i = idx(n, m);
                    sy += e[i] * harm.y[i];
                    st += e[i] * harm.dtheta[i];
                    sp += e[i] * harm.dphi[i];
                }
                phi += sy * rv[n];
                ar += sy * rd[n];
                at += st * rv[n];
                ap += sp * rv[n];
            }
            // ∇(f(kr) Y) = k f′ Y r̂ + (f/r)(∂_θ Y θ̂ + (1/sin θ) ∂_φ Y φ̂).
            let ar = ar * k;
            let (at, ap) = (at * inv_r, ap * inv_r);
            let grad = std::array::from_fn(|d| (ar * rh[d] + at * th[d] + ap * ph[d]) * ik);
            o.push((phi * ik, grad));
        }
    }
    out
}
