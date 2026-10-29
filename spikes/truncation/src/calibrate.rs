//! The chains over offsets and placements, and the search for the smallest p.
//!
//! Level 0 is the M2L level (boxes of width 1, the source box S at the origin, the target
//! box T at an offset), level 1 the children and level 2 the grandchildren:
//!
//! - **C1**: P2M in S at p₀ → M2L to T → L2P in T;
//! - **C2**: P2M in S's children at p₁ → M2M to S at p₀ → M2L to T → L2L to T's children
//!   → L2P;
//! - **C3m**: P2M in S's grandchildren at p₂ → M2M → M2M → the multipole of S evaluated in
//!   T (M2P);
//! - **C3l**: the local of S from the sources in T (P2L at p₀) → L2L → L2L → L2P in S's
//!   grandchildren.
//!
//! Each against the direct sum, as relative ℓ2 errors of potential and gradient over the
//! targets (proposal §9.1).

use crate::chain::{self, Field, Setup, Work, e2p, p2e, rcr};
use crate::coaxial::Table;
use crate::complex::Complex;
use crate::geometry::{self, Placement, Rng, child_index, children, dist2};
use crate::harmonics::len;
use crate::real::Real;
use crate::rotation::{Direction, Rotation};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The degree p = ⌈P⌉ − 1 of proposal §3.5 (Gumerov and Duraiswami 2009, eq. 18, δ = 2;
/// S8's `degree`).
pub fn formula_p(ka: f64, eps: f64) -> usize {
    let p_hi = ka + (1.5 * (1.0 / eps).ln()).powf(2.0 / 3.0) * ka.powf(1.0 / 3.0);
    let p_lo = 1.0 - (eps * 0.5f64.powf(1.5)).ln() / 2.0f64.ln();
    ((p_lo.powi(4) + p_hi.powi(4)).powf(0.25)).ceil() as usize - 1
}

/// Which chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Chain {
    /// P2M → M2L → L2P on one level.
    C1,
    /// P2M → M2M → M2L → L2L → L2P over two levels.
    C2,
    /// P2M → M2M → M2M → M2P over three levels.
    C3m,
    /// P2L → L2L → L2L → L2P over three levels.
    C3l,
}

impl Chain {
    /// A short name.
    pub fn name(self) -> &'static str {
        match self {
            Chain::C1 => "C1",
            Chain::C2 => "C2",
            Chain::C3m => "C3m",
            Chain::C3l => "C3l",
        }
    }

    /// The level of the leaves of the sources and of the targets.
    fn leaf_levels(self) -> (i32, i32) {
        match self {
            Chain::C1 => (0, 0),
            Chain::C2 => (1, 1),
            Chain::C3m => (2, 0),
            Chain::C3l => (0, 2),
        }
    }
}

/// Points and seeds.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Sources per box.
    pub n_src: usize,
    /// Targets per box.
    pub n_tgt: usize,
    /// The seed of the placements and charges.
    pub seed: u64,
    /// Worker threads.
    pub threads: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            n_src: 128,
            n_tgt: 64,
            seed: 2026,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }
}

/// The error of one (offset, placement, degrees).
#[derive(Clone, Copy, Debug)]
pub struct Record {
    /// Index into the offsets given.
    pub offset: usize,
    /// The placement.
    pub placement: Placement,
    /// Index into the degrees given.
    pub degrees: usize,
    /// Relative ℓ2 error of the potential.
    pub pot: f64,
    /// Relative ℓ2 error of the gradient.
    pub grad: f64,
}

fn seed(cfg: &Config, placement: Placement, what: u64, offset: [i32; 3]) -> Rng {
    let p = placement as u64;
    let o = ((offset[0] + 8) * 256 + (offset[1] + 8) * 16 + (offset[2] + 8)) as u64;
    Rng::new(cfg.seed ^ (p << 40) ^ (what << 32) ^ o.wrapping_mul(0x2545_F491_4F6C_DD1D))
}

/// Box centre of an offset (width 1).
fn centre(o: [i32; 3]) -> [f64; 3] {
    [o[0] as f64, o[1] as f64, o[2] as f64]
}

fn real3<T: Real>(x: [f64; 3]) -> [T; 3] {
    [T::from_f64(x[0]), T::from_f64(x[1]), T::from_f64(x[2])]
}

/// The eight unit-cube directions (±1, ±1, ±1), sign bit d set meaning +, as rotations.
fn child_rotations<T: Real>(p: usize) -> Vec<Rotation<T>> {
    children([0.0; 3], 1.0)
        .iter()
        .map(|c| Rotation::new(Direction::new(real3::<T>(*c)), p))
        .collect()
}

/// The sources and charges of a box.
struct Cloud<T> {
    points: Vec<[T; 3]>,
    raw: Vec<[f64; 3]>,
    q: Vec<Complex<T>>,
}

fn cloud<T: Real>(
    placement: Placement,
    c: [f64; 3],
    leaf_level: i32,
    n: usize,
    rng: &mut Rng,
) -> Cloud<T> {
    let raw = geometry::points(placement, c, 1.0, 0.5f64.powi(leaf_level), n, rng);
    let q = geometry::charges::<T>(n, rng);
    Cloud {
        points: geometry::to_real(&raw),
        raw,
        q,
    }
}

/// The points and charges of one leaf.
type Leaf<T> = (Vec<[T; 3]>, Vec<Complex<T>>);

/// The multipole of S (at the origin) to degree p₀ from its leaves at `leaf_level`, by
/// P2M at the leaf degree and M2M up (C1: P2M directly).
fn upward<T: Real>(
    setup: &Setup<T>,
    src: &Cloud<T>,
    leaf_level: usize,
    deg: [usize; 3],
    shift: &[Option<Table<T>>; 3],
    rots: &[Rotation<T>],
    work: &mut Work<T>,
) -> Vec<Complex<T>> {
    let k = setup.k;
    if leaf_level == 0 {
        return p2e(
            true,
            k,
            setup.s[0],
            [T::ZERO; 3],
            &src.points,
            &src.q,
            deg[0],
        );
    }
    // Group the sources by leaf: (child, grandchild).
    let kids = children([0.0; 3], 1.0);
    let mut groups: HashMap<(usize, usize), Leaf<T>> = HashMap::new();
    for ((x, y), q) in src.raw.iter().zip(&src.points).zip(&src.q) {
        let i = child_index([0.0; 3], *x);
        let j = if leaf_level == 2 {
            child_index(kids[i], *x)
        } else {
            0
        };
        let e = groups.entry((i, j)).or_default();
        e.0.push(*y);
        e.1.push(*q);
    }
    let mut out = vec![Complex::ZERO; len(deg[0])];
    let mut tmp = vec![Complex::ZERO; len(deg[0].max(deg[1]))];
    for i in 0..8 {
        // The multipole of child i at degree p₁.
        let mut child = vec![Complex::ZERO; len(deg[1])];
        if leaf_level == 1 {
            if let Some((pts, q)) = groups.get(&(i, 0)) {
                child = p2e(true, k, setup.s[1], real3(kids[i]), pts, q, deg[1]);
            } else {
                continue;
            }
        } else {
            let grand = children(kids[i], 0.5);
            let scale = setup.shift_scale(2, true, deg[1].max(deg[2]));
            let mut any = false;
            for j in 0..8 {
                if let Some((pts, q)) = groups.get(&(i, j)) {
                    any = true;
                    let m = p2e(true, k, setup.s[2], real3(grand[j]), pts, q, deg[2]);
                    // t = c_child − c_grandchild: direction 7 − j.
                    let table = shift[2].as_ref().unwrap();
                    rcr(
                        &rots[7 - j],
                        table,
                        deg[2],
                        deg[1],
                        Some(&scale),
                        &m,
                        &mut tmp,
                        work,
                    );
                    for (o, v) in child.iter_mut().zip(&tmp) {
                        *o += *v;
                    }
                }
            }
            if !any {
                continue;
            }
        }
        let scale = setup.shift_scale(1, true, deg[0].max(deg[1]));
        let table = shift[1].as_ref().unwrap();
        rcr(
            &rots[7 - i],
            table,
            deg[1],
            deg[0],
            Some(&scale),
            &child,
            &mut tmp,
            work,
        );
        for (o, v) in out.iter_mut().zip(&tmp) {
            *o += *v;
        }
    }
    out
}

/// The fields at the targets from a local of degree p₀ about `c` at level 0, by L2L to
/// the leaves at `leaf_level` and L2P there.
#[allow(clippy::too_many_arguments)]
fn downward<T: Real>(
    setup: &Setup<T>,
    c: [f64; 3],
    local: &[Complex<T>],
    tgt: &Cloud<T>,
    leaf_level: usize,
    deg: [usize; 3],
    shift: &[Option<Table<T>>; 3],
    rots: &[Rotation<T>],
    work: &mut Work<T>,
) -> Vec<Field<T>> {
    let k = setup.k;
    let mut out = Vec::new();
    if leaf_level == 0 {
        e2p(
            true,
            k,
            setup.s[0],
            real3(c),
            local,
            deg[0],
            &tgt.points,
            work,
            &mut out,
        );
        return out;
    }
    // Fields in the order of the targets.
    let mut fields = vec![(Complex::ZERO, [Complex::ZERO; 3]); tgt.points.len()];
    let kids = children(c, 1.0);
    let mut lc = vec![Complex::ZERO; len(deg[0].max(deg[1]))];
    let mut lg = vec![Complex::ZERO; len(deg[1].max(deg[2]))];
    let mut part = Vec::new();
    for i in 0..8 {
        let idx: Vec<usize> = (0..tgt.raw.len())
            .filter(|&t| child_index(c, tgt.raw[t]) == i)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let scale = setup.shift_scale(1, false, deg[0].max(deg[1]));
        rcr(
            &rots[i],
            shift[1].as_ref().unwrap(),
            deg[0],
            deg[1],
            Some(&scale),
            local,
            &mut lc,
            work,
        );
        if leaf_level == 1 {
            let pts: Vec<[T; 3]> = idx.iter().map(|&t| tgt.points[t]).collect();
            e2p(
                true,
                k,
                setup.s[1],
                real3(kids[i]),
                &lc,
                deg[1],
                &pts,
                work,
                &mut part,
            );
            for (t, f) in idx.iter().zip(&part) {
                fields[*t] = *f;
            }
            continue;
        }
        let grand = children(kids[i], 0.5);
        let scale = setup.shift_scale(2, false, deg[1].max(deg[2]));
        for j in 0..8 {
            let jdx: Vec<usize> = idx
                .iter()
                .copied()
                .filter(|&t| child_index(kids[i], tgt.raw[t]) == j)
                .collect();
            if jdx.is_empty() {
                continue;
            }
            rcr(
                &rots[j],
                shift[2].as_ref().unwrap(),
                deg[1],
                deg[2],
                Some(&scale),
                &lc,
                &mut lg,
                work,
            );
            let pts: Vec<[T; 3]> = jdx.iter().map(|&t| tgt.points[t]).collect();
            e2p(
                true,
                k,
                setup.s[2],
                real3(grand[j]),
                &lg,
                deg[2],
                &pts,
                work,
                &mut part,
            );
            for (t, f) in jdx.iter().zip(&part) {
                fields[*t] = *f;
            }
        }
    }
    out.extend(fields);
    out
}

/// Runs `chain` for every offset, placement and degree assignment [p₀, p₁, p₂] (unused
/// levels ignored), in parallel over the offsets.
pub fn run<T: Real>(
    setup: &Setup<T>,
    chain: Chain,
    offsets: &[[i32; 3]],
    placements: &[Placement],
    degrees: &[[usize; 3]],
    cfg: &Config,
) -> Vec<Record> {
    let top: [usize; 3] = std::array::from_fn(|l| degrees.iter().map(|d| d[l]).max().unwrap());
    let (src_leaf, tgt_leaf) = chain.leaf_levels();
    let levels = src_leaf.max(tgt_leaf) as usize;
    let deg_of = |d: [usize; 3]| -> [usize; 3] {
        std::array::from_fn(|l| if l <= levels { d[l] } else { 0 })
    };
    // Tables: M2L per distance at p₀ (C1, C2), shifts per level at their top degree.
    let mut d2s: Vec<i32> = offsets.iter().map(|&o| dist2(o)).collect();
    d2s.sort_unstable();
    d2s.dedup();
    let needs_m2l = matches!(chain, Chain::C1 | Chain::C2);
    let m2l: HashMap<i32, Table<T>> = if needs_m2l {
        parallel_map(&d2s, cfg.threads, |&d2| setup.m2l_table(d2, top[0]))
            .into_iter()
            .zip(&d2s)
            .map(|(t, &d)| (d, t))
            .collect()
    } else {
        HashMap::new()
    };
    let shift: [Option<Table<T>>; 3] = std::array::from_fn(|l| {
        (l >= 1 && l <= levels).then(|| setup.shift_table(l, top[l - 1].max(top[l])))
    });
    let rot_top = if levels >= 1 {
        top[..=levels].iter().copied().max().unwrap()
    } else {
        0
    };
    let rots: Vec<Rotation<T>> = if levels >= 1 {
        child_rotations(rot_top)
    } else {
        Vec::new()
    };
    // Per placement: S's sources (C1, C2, C3m) or targets (C3l), and S's multipoles per
    // degree assignment.
    struct Shared<T> {
        cloud: Cloud<T>,
        multipoles: Vec<Vec<Complex<T>>>,
    }
    let shared: Vec<Shared<T>> = placements
        .iter()
        .map(|&pl| {
            if chain == Chain::C3l {
                let mut rng = seed(cfg, pl, 2, [0; 3]);
                Shared {
                    cloud: cloud(pl, [0.0; 3], tgt_leaf, cfg.n_tgt, &mut rng),
                    multipoles: Vec::new(),
                }
            } else {
                let mut rng = seed(cfg, pl, 1, [0; 3]);
                let src = cloud::<T>(pl, [0.0; 3], src_leaf, cfg.n_src, &mut rng);
                let mut work = Work::new(rot_top.max(top[0]));
                let multipoles = if chain == Chain::C1 {
                    // P2M coefficients do not depend on p: one expansion at the top degree.
                    vec![upward(setup, &src, 0, top, &shift, &rots, &mut work)]
                } else {
                    let ds: Vec<[usize; 3]> = degrees.iter().map(|&d| deg_of(d)).collect();
                    parallel_map(&ds, cfg.threads, |&d| {
                        let mut work = Work::new(rot_top.max(top[0]));
                        upward(setup, &src, src_leaf as usize, d, &shift, &rots, &mut work)
                    })
                };
                Shared {
                    cloud: src,
                    multipoles,
                }
            }
        })
        .collect();
    let results = Mutex::new(Vec::new());
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..cfg.threads.min(offsets.len()).max(1) {
            scope.spawn(|| {
                let mut work = Work::new(top[0]);
                let mut local = vec![Complex::ZERO; len(top[0])];
                let mut fields = Vec::new();
                let mut mine = Vec::new();
                loop {
                    let oi = next.fetch_add(1, Ordering::Relaxed);
                    if oi >= offsets.len() {
                        break;
                    }
                    let o = offsets[oi];
                    let c_t = centre(o);
                    let rot =
                        needs_m2l.then(|| Rotation::new(Direction::new(real3::<T>(c_t)), top[0]));
                    for (pi, &pl) in placements.iter().enumerate() {
                        let sh = &shared[pi];
                        // The other box's points and the reference.
                        let (src_pts, src_q, tgt) = if chain == Chain::C3l {
                            let mut rng = seed(cfg, pl, 3, o);
                            let src = cloud::<T>(pl, c_t, src_leaf, cfg.n_src, &mut rng);
                            (src.points, src.q, None)
                        } else {
                            let mut rng = seed(cfg, pl, 4, o);
                            let tgt = cloud::<T>(pl, c_t, tgt_leaf, cfg.n_tgt, &mut rng);
                            (sh.cloud.points.clone(), sh.cloud.q.clone(), Some(tgt))
                        };
                        let tgt_ref = tgt.as_ref().unwrap_or(&sh.cloud);
                        let exact = chain::direct(setup.k, &src_pts, &src_q, &tgt_ref.points);
                        let p2l = (chain == Chain::C3l).then(|| {
                            p2e(
                                false,
                                setup.k,
                                setup.s[0],
                                [T::ZERO; 3],
                                &src_pts,
                                &src_q,
                                top[0],
                            )
                        });
                        for (di, &d) in degrees.iter().enumerate() {
                            let d = deg_of(d);
                            let p0 = d[0];
                            let got = match chain {
                                Chain::C1 | Chain::C2 => {
                                    let m = if chain == Chain::C1 {
                                        &sh.multipoles[0][..len(p0)]
                                    } else {
                                        &sh.multipoles[di][..]
                                    };
                                    rcr(
                                        rot.as_ref().unwrap(),
                                        &m2l[&dist2(o)],
                                        p0,
                                        p0,
                                        None,
                                        m,
                                        &mut local,
                                        &mut work,
                                    );
                                    downward(
                                        setup,
                                        c_t,
                                        &local[..len(p0)],
                                        tgt_ref,
                                        tgt_leaf as usize,
                                        d,
                                        &shift,
                                        &rots,
                                        &mut work,
                                    )
                                }
                                Chain::C3m => {
                                    e2p(
                                        false,
                                        setup.k,
                                        setup.s[0],
                                        [T::ZERO; 3],
                                        &sh.multipoles[di],
                                        p0,
                                        &tgt_ref.points,
                                        &mut work,
                                        &mut fields,
                                    );
                                    fields.clone()
                                }
                                Chain::C3l => {
                                    let l = &p2l.as_ref().unwrap()[..len(p0)];
                                    downward(
                                        setup,
                                        [0.0; 3],
                                        l,
                                        tgt_ref,
                                        tgt_leaf as usize,
                                        d,
                                        &shift,
                                        &rots,
                                        &mut work,
                                    )
                                }
                            };
                            let (pot, grad) = chain::errors(&got, &exact);
                            mine.push(Record {
                                offset: oi,
                                placement: pl,
                                degrees: di,
                                pot,
                                grad,
                            });
                        }
                    }
                }
                results.lock().unwrap().extend(mine);
            });
        }
    });
    let mut out = results.into_inner().unwrap();
    out.sort_by_key(|r| (r.degrees, r.offset, r.placement as usize));
    out
}

/// f over the items on `threads` threads, in order.
pub fn parallel_map<I: Sync, O: Send>(
    items: &[I],
    threads: usize,
    f: impl Fn(&I) -> O + Sync,
) -> Vec<O> {
    let next = AtomicUsize::new(0);
    let out: Mutex<Vec<(usize, O)>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads.min(items.len()).max(1) {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= items.len() {
                        break;
                    }
                    let v = f(&items[i]);
                    out.lock().unwrap().push((i, v));
                }
            });
        }
    });
    let mut out = out.into_inner().unwrap();
    out.sort_by_key(|(i, _)| *i);
    out.into_iter().map(|(_, v)| v).collect()
}

/// The worst (potential, gradient) error per degree assignment over the records.
pub fn worst(records: &[Record], n_degrees: usize) -> Vec<(f64, f64)> {
    let mut out = vec![(0.0f64, 0.0f64); n_degrees];
    for r in records {
        let w = &mut out[r.degrees];
        w.0 = w.0.max(r.pot);
        w.1 = w.1.max(r.grad);
    }
    out
}

/// The smallest p of a scanned curve (p ascending, error) from which every error is at
/// most `threshold`; None if the last one is above.
pub fn smallest_passing(curve: &[(usize, f64)], threshold: f64) -> Option<usize> {
    let mut best = None;
    for &(p, e) in curve.iter().rev() {
        if e <= threshold {
            best = Some(p);
        } else {
            break;
        }
    }
    best
}
