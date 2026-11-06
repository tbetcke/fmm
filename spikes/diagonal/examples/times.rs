//! Prototype times of the diagonal M2L and of RCR (S6 output 7), f64, one thread.
//!
//! ```text
//! cargo run -p nd-fmm-spike-diagonal --release --example times -- [--p 8,16,...] [--reps n]
//! ```
//!
//! Per degree p (k a set so that p is §3.5's formula degree at 10⁻⁶, the class the
//! nearest, (0, 2, 0)): the diagonal pair (N_l complex multiply-adds of T and the source's
//! signature into the target's accumulator), S → F and F → R per box, and an RCR pair with
//! S7's f64 rotation, (S|R) table and rotation back, the tables built once outside the
//! timing; also the build of the class's tables: T in double-double over the samples, its
//! interpolation table and interpolated samples, the RCR rotation and (S|R) table. Each the
//! median of `reps` timings. These are the spike's prototypes, not tuned kernels: the
//! ratios, not the absolute times, are the result.

use std::hint::black_box;
use std::time::Instant;

use nd_fmm_spike_diagonal::grid::Grid;
use nd_fmm_spike_diagonal::rules::formula;
use nd_fmm_spike_diagonal::run::narrow;
use nd_fmm_spike_diagonal::signature::{f2r_ring, finish, s2f_ring};
use nd_fmm_spike_diagonal::transfer::{Interp, Transfer, angle};
use nd_fmm_spike_truncation::chain::{Kind, Setup, Work, rcr};
use nd_fmm_spike_truncation::complex::Complex;
use nd_fmm_spike_truncation::geometry::Rng;
use nd_fmm_spike_truncation::harmonics::{Harmonics, len};
use nd_fmm_spike_truncation::real::Dd;
use nd_fmm_spike_truncation::rotation::{Direction, Rotation};

/// The median of `reps` timings of f, seconds.
fn median(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

/// The smallest k a on a 1/64 grid whose formula degree at 10⁻⁶ is p.
fn ka_for(p: usize) -> f64 {
    let mut ka = 1.0;
    while formula(ka, 1e-6) < p {
        ka += 1.0 / 64.0;
    }
    ka
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .map(|i| args[i + 1].clone())
    };
    let ps: Vec<usize> = get("--p")
        .unwrap_or_else(|| "24,32,48,64,96,128,192,256".into())
        .split(',')
        .map(|x| x.parse().unwrap())
        .collect();
    let reps: usize = get("--reps").map_or(5, |r| r.parse().unwrap());
    println!(
        "| p | k a | N_l | diagonal pair (µs) | S → F per box (µs) | F → R per box (µs) | \
         RCR pair (µs) | RCR / diagonal | T build, dd (ms) | T interpolated (ms) | \
         RCR tables (ms) |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for p in ps {
        let ka = ka_for(p);
        let rep = [0, 2, 0];
        let grid = Grid::new(p);
        let tab = grid.tables::<f64>();
        let nl = grid.samples();
        let mut rng = Rng::new(p as u64);
        let mut cplx = |n: usize| -> Vec<Complex<f64>> {
            (0..n)
                .map(|_| Complex::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)))
                .collect()
        };
        let m = cplx(len(p));
        let t = cplx(nl);
        let f = cplx(nl);
        let mut g = vec![Complex::ZERO; nl];
        let diag = median(reps, || {
            for ((gj, &tj), &fj) in g.iter_mut().zip(&t).zip(&f) {
                *gj += tj * fj;
            }
            black_box(&g);
        });
        // The rings' Θ in f64, kept as a host table would be.
        let mut harm = Harmonics::<Dd>::new(p);
        let thetas: Vec<Vec<f64>> = (0..grid.rings())
            .map(|i| {
                let mut th = Vec::new();
                grid.theta::<f64>(i, &mut harm, &mut th);
                th
            })
            .collect();
        let mut ab = vec![Complex::ZERO; 2 * (p + 1)];
        let mut sig = vec![Complex::ZERO; nl];
        let s2f = median(reps, || {
            for (i, th) in thetas.iter().enumerate() {
                let n = grid.nphi;
                s2f_ring(p, th, &m, &tab, &mut ab, &mut sig[i * n..(i + 1) * n]);
            }
            black_box(&sig);
        });
        let mut acc = vec![Complex::ZERO; len(p)];
        let f2r = median(reps, || {
            acc.fill(Complex::ZERO);
            for (i, th) in thetas.iter().enumerate() {
                let n = grid.nphi;
                f2r_ring(
                    p,
                    th,
                    tab.w[i],
                    &sig[i * n..(i + 1) * n],
                    &tab,
                    &mut ab,
                    &mut acc,
                );
            }
            finish(p, &mut acc);
            black_box(&acc);
        });
        // RCR in f64 with S7's operators.
        let mut setup = Setup::<f64>::new(Kind::Real, ka);
        setup.s = [1.0; 3];
        let c_t = rep.map(|x| x as f64);
        let (rot, table, rcr_build) = {
            let s = Instant::now();
            let rot = Rotation::new(Direction::new(c_t), p);
            let table = setup.m2l_table(4, p);
            (rot, table, s.elapsed().as_secs_f64())
        };
        let mut work = Work::new(p);
        let mut out = vec![Complex::ZERO; len(p)];
        let rcr_t = median(reps, || {
            rcr(&rot, &table, p, p, None, &m, &mut out, &mut work);
            black_box(&out);
        });
        // The class's T: in double-double at every sample, or by interpolation.
        let k = Setup::<Dd>::new(Kind::Real, ka).k;
        let tr = Transfer::new(k, rep, &[2 * p]);
        let mut tv = [Complex::ZERO];
        let start = Instant::now();
        let mut tdd = Vec::with_capacity(nl);
        for i in 0..grid.rings() {
            for j in 0..grid.nphi {
                tr.eval(tr.mu(grid.direction(i, j)), &mut tv);
                tdd.push(narrow::<f64>(tv[0]));
            }
        }
        let t_dd = start.elapsed().as_secs_f64();
        black_box(&tdd);
        let start = Instant::now();
        let it = Interp::new(&tr, 0, 3.0, 36);
        let mut ti = Vec::with_capacity(nl);
        for i in 0..grid.rings() {
            for j in 0..grid.nphi {
                ti.push(it.eval(angle(grid.direction(i, j), tr.that)));
            }
        }
        let t_i = start.elapsed().as_secs_f64();
        black_box(&ti);
        println!(
            "| {p} | {ka:.2} | {nl} | {:.1} | {:.0} | {:.0} | {:.1} | {:.2} | {:.0} | {:.0} | {:.0} |",
            diag * 1e6,
            s2f * 1e6,
            f2r * 1e6,
            rcr_t * 1e6,
            rcr_t / diag,
            t_dd * 1e3,
            t_i * 1e3,
            rcr_build * 1e3
        );
    }
}
