//! The quadratures of the Helmholtz exponential M2L (Helmholtz CH6.1a,
//! `nd_fmm_tables::helmholtz::plane_wave::quadrature`): FMM3D's tables as its files hold
//! them, the lookup, the node maps, |J_m| of complex argument, and the counts on the region.
//!
//! Error measure: FMM3D's, the largest |Q − G| over the sample of the region, Q the
//! quadrature and G = e^{iκ|u|}/|u| in box widths (the kernel the λ quadrature integrates,
//! exactly; S4 §1), against the error the file states for the domain. The acceptance
//! (docs/helmholtz/tasks/CH6.1a.md): the kernel with the counts within [`KERNEL_BOUND`] of
//! it and the point-to-point chain within [`CHAIN_BOUND`], on the grid of the region; the λ
//! sum with α resolved within [`LAMBDA_BOUND`] (S4 §3.1's worst over all tables).

use core::f64::consts::{PI, TAU};

use nd_fmm_math::Complex;
use nd_fmm_tables::helmholtz::plane_wave::quadrature::{
    DOMAINS, IM_BREAKS, NX, NY, NodeMap, RE_BREAKS, RHO_MAX, bessel_j_norms, domain_of, domains,
    rectangle,
};
use nd_fmm_tables::helmholtz::plane_wave::{Precision, Quadrature};

type C = Complex<f64>;

/// The kernel with the counts, against the stated error.
const KERNEL_BOUND: f64 = 2.0;

/// The point-to-point chain with the counts, against the stated error.
const CHAIN_BOUND: f64 = 3.0;

/// The λ sum with α resolved, against the stated error (S4 §3.1: 1.5 at iprec 1–3).
const LAMBDA_BOUND: f64 = 1.5;

/// The domains of S4's fixture: 1 (κ ≈ 0, type 2), 11 (Re κ in [3π, 4π], type 2), 139
/// (Im κ in [1, π/2], type 2), 172 (κ near 10 + 3i, type 1), 236 (κ near 0.5 + 10i,
/// type 1).
const FIXTURE_DOMAINS: [usize; 5] = [1, 11, 139, 172, 236];

/// The kernel e^{iκ|u|}/|u| in box widths.
fn kernel(kappa: C, u: [f64; 3]) -> C {
    let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    (C::i() * kappa * r).exp() / r
}

/// The nine points of a rectangle: its corners (the upper edges pulled in by 10⁻⁹
/// relative), the edge midpoints and the centre.
fn nine_points(domain: usize) -> Vec<C> {
    let (lo, hi) = rectangle(domain);
    let hi = C::new(hi.re * (1.0 - 1e-9), hi.im * (1.0 - 1e-9));
    let re = [lo.re, 0.5 * (lo.re + hi.re), hi.re];
    let im = [lo.im, 0.5 * (lo.im + hi.im), hi.im];
    im.iter()
        .flat_map(|&y| re.iter().map(move |&x| C::new(x, y)))
        .collect()
}

/// The centre of a rectangle.
fn centre(domain: usize) -> C {
    let (lo, hi) = rectangle(domain);
    (lo + hi) * 0.5
}

/// The grid of the region: z ∈ {1, 1.25, …, 4}, x, y ∈ {−4, −3.5, …, 4}.
struct Grid {
    xy: Vec<f64>,
    z: Vec<f64>,
}

impl Grid {
    fn region() -> Self {
        Self {
            xy: (0..17).map(|i| -4.0 + 0.5 * i as f64).collect(),
            z: (0..13).map(|i| 1.0 + 0.25 * i as f64).collect(),
        }
    }
}

/// The largest |Q − G| on the grid, Q the quadrature with the α counts `counts`.
fn kernel_error(q: &Quadrature, counts: &[usize]) -> f64 {
    kernel_errors(q, counts, RHO_MAX)[1]
}

/// The largest |Q − G| on the grid points with lateral offset ρ ≤ `reach`, and on the whole
/// grid, Q the quadrature with the α counts `counts`; the sums separated in x, y and z per
/// (k, j).
fn kernel_errors(q: &Quadrature, counts: &[usize], reach: f64) -> [f64; 2] {
    let grid = Grid::region();
    let (nx, nz) = (grid.xy.len(), grid.z.len());
    let mut sum = vec![C::new(0.0, 0.0); nz * nx * nx];
    let (mut ex, mut ey) = (vec![C::new(0.0, 0.0); nx], vec![C::new(0.0, 0.0); nx]);
    for (k, &n) in counts.iter().enumerate() {
        let c = q.weights()[k] / n as f64;
        let ez: Vec<C> = grid.z.iter().map(|&z| (-q.lambda()[k] * z).exp()).collect();
        let it = C::i() * q.t()[k];
        for j in 0..n {
            let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
            for (i, &x) in grid.xy.iter().enumerate() {
                ex[i] = (it * (x * ca)).exp();
                ey[i] = (it * (x * sa)).exp();
            }
            for (iz, &z) in ez.iter().enumerate() {
                let cz = c * z;
                for (iy, &y) in ey.iter().enumerate() {
                    let czy = cz * y;
                    let row = &mut sum[(iz * nx + iy) * nx..][..nx];
                    for (s, &e) in row.iter_mut().zip(&ex) {
                        *s += czy * e;
                    }
                }
            }
        }
    }
    let mut worst = [0.0f64; 2];
    let mut i = 0;
    for &z in &grid.z {
        for &y in &grid.xy {
            for &x in &grid.xy {
                let e = (sum[i] - kernel(q.kappa(), [x, y, z])).norm();
                if x * x + y * y <= reach * reach {
                    worst[0] = worst[0].max(e);
                }
                worst[1] = worst[1].max(e);
                i += 1;
            }
        }
    }
    worst
}

/// The λ sum alone: α counts that resolve J₀(tₖ ρ) to round-off on the region.
fn lambda_error(q: &Quadrature) -> f64 {
    let converged: Vec<usize> = q
        .t()
        .iter()
        .map(|t| (t.norm() * RHO_MAX).ceil() as usize + 60)
        .collect();
    kernel_error(q, &converged)
}

/// The point-to-point chain against the kernel: sources at the centre and the corners of
/// the box [−½, ½]³ and `interior` seeded points in it, their waves at λₖ truncated to the
/// orders |m| ≤ Fₖ (coefficients from a DFT exact to round-off) and sampled at the Nₖ α
/// nodes, evaluated at the targets x, y ∈ {−3.5, −3, …, 3.5}, z ∈ {1.5, 2, …, 3.5} (the
/// corners, edge and face midpoints and centres of the up direction's target boxes):
/// the largest |Q − G|.
fn chain_error(q: &Quadrature, interior: usize) -> f64 {
    chain_errors(q, interior, RHO_MAX)[1]
}

/// The chain of [`chain_error`]: the largest |Q − G| over the pairs whose lateral offset
/// is ρ ≤ `reach`, and over all pairs.
fn chain_errors(q: &Quadrature, interior: usize, reach: f64) -> [f64; 2] {
    let mut sources = vec![[0.0; 3]];
    for b in 0..8 {
        let h = |bit: usize| if b & bit != 0 { 0.5 } else { -0.5 };
        sources.push([h(4), h(2), h(1)]);
    }
    let mut state = 0x5eed_c404_u64;
    let mut unit = || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..interior {
        sources.push([unit() - 0.5, unit() - 0.5, unit() - 0.5]);
    }
    let target_xy: Vec<f64> = (0..15).map(|i| -3.5 + 0.5 * i as f64).collect();
    let target_z: Vec<f64> = (0..5).map(|i| 1.5 + 0.5 * i as f64).collect();
    let (nx, nz) = (target_xy.len(), target_z.len());
    let mut worst = [0.0f64; 2];
    for s in &sources {
        let mut sum = vec![C::new(0.0, 0.0); nz * nx * nx];
        let (mut ex, mut ey) = (vec![C::new(0.0, 0.0); nx], vec![C::new(0.0, 0.0); nx]);
        for k in 0..q.nodes() {
            let n = q.alpha_counts()[k];
            let f = q.fourier_counts()[k] as i64;
            let c = q.weights()[k] / n as f64;
            let (lambda, t) = (q.lambda()[k], q.t()[k]);
            let wave = |alpha: f64| {
                let (sa, ca) = alpha.sin_cos();
                (lambda * s[2] - C::i() * t * (s[0] * ca + s[1] * sa)).exp()
            };
            let m = 2 * (f as usize + (t.norm() * 0.75).ceil() as usize + 32);
            let samples: Vec<C> = (0..m).map(|l| wave(TAU * l as f64 / m as f64)).collect();
            let coeffs: Vec<C> = (-f..=f)
                .map(|order| {
                    samples
                        .iter()
                        .enumerate()
                        .map(|(l, v)| {
                            v * C::from_polar(1.0, -TAU * (order * l as i64) as f64 / m as f64)
                        })
                        .sum::<C>()
                        / m as f64
                })
                .collect();
            let values: Vec<C> = (0..n)
                .map(|j| {
                    let a = TAU * j as f64 / n as f64;
                    (-f..=f)
                        .zip(&coeffs)
                        .map(|(order, c)| c * C::from_polar(1.0, order as f64 * a))
                        .sum()
                })
                .collect();
            let ez: Vec<C> = target_z.iter().map(|&z| (-lambda * z).exp()).collect();
            let it = C::i() * t;
            for (j, v) in values.iter().enumerate() {
                let (sa, ca) = (TAU * j as f64 / n as f64).sin_cos();
                for (i, &x) in target_xy.iter().enumerate() {
                    ex[i] = (it * (x * ca)).exp();
                    ey[i] = (it * (x * sa)).exp();
                }
                let cv = c * v;
                for (iz, &z) in ez.iter().enumerate() {
                    let cz = cv * z;
                    for (iy, &y) in ey.iter().enumerate() {
                        let czy = cz * y;
                        let row = &mut sum[(iz * nx + iy) * nx..][..nx];
                        for (q, &e) in row.iter_mut().zip(&ex) {
                            *q += czy * e;
                        }
                    }
                }
            }
        }
        let mut i = 0;
        for &z in &target_z {
            for &y in &target_xy {
                for &x in &target_xy {
                    let u = [x - s[0], y - s[1], z - s[2]];
                    let e = (sum[i] - kernel(q.kappa(), u)).norm();
                    if u[0] * u[0] + u[1] * u[1] <= reach * reach {
                        worst[0] = worst[0].max(e);
                    }
                    worst[1] = worst[1].max(e);
                    i += 1;
                }
            }
        }
    }
    worst
}

/// The kernel (and optionally the chain) of the quadrature at κ against the bounds.
fn check_counts(kappa: C, precision: Precision, chain: Option<usize>) {
    let q = Quadrature::new(kappa, precision).expect("κ in the tables");
    let e = q.stated_error();
    let kernel = kernel_error(&q, q.alpha_counts());
    assert!(
        kernel <= KERNEL_BOUND * e,
        "kernel at κ = {kappa}, iprec {}, domain {}: {kernel:.3e} > {KERNEL_BOUND} × {e:.3e}",
        precision.iprec(),
        q.domain()
    );
    if let Some(interior) = chain {
        let chain = chain_error(&q, interior);
        assert!(
            chain <= CHAIN_BOUND * e,
            "chain at κ = {kappa}, iprec {}, domain {}: {chain:.3e} > {CHAIN_BOUND} × {e:.3e}",
            precision.iprec(),
            q.domain()
        );
    }
}

#[test]
#[allow(
    clippy::excessive_precision,
    reason = "FMM3D's literals keep their 16 digits, as in its files"
)]
fn the_data_are_fmm3ds() {
    // Nodes per precision, as `grep -c 'xs('` counts them in FMM3D's files.
    let totals = [7434, 10654, 14486];
    for (precision, total) in Precision::ALL.into_iter().zip(totals) {
        let table = domains(precision);
        assert_eq!(table.len(), DOMAINS);
        assert_eq!(table.iter().map(|d| d.nodes.len()).sum::<usize>(), total);
        for (i, d) in table.iter().enumerate() {
            assert_eq!(d.index, i + 1);
            let (lo, hi) = rectangle(d.index);
            for (stated, exact) in [(d.lower, lo), (d.upper, hi)] {
                for (a, b) in [(stated.re, exact.re), (stated.im, exact.im)] {
                    assert!(
                        (a - b).abs() <= 1e-15 * b.abs(),
                        "domain {}: {a} vs {b}",
                        d.index
                    );
                }
            }
            let expected = if lo.im >= PI / 2.0 {
                NodeMap::RealT
            } else {
                NodeMap::Contour
            };
            assert_eq!(d.map, expected, "domain {}", d.index);
            assert_eq!(d.nodes.len(), d.weights.len());
            assert!(d.stated_error > 0.0 && d.stated_error < 1.0);
        }
    }
    // The first node and weight of the fixture's domains (S4's fixture, cut from
    // FMM3D's files): (iprec, domain, n, x₁, w₁).
    let first = [
        (2, 1, 21, 0.4622470085766964e-03, 0.1179127434939096e-02),
        (2, 11, 35, 0.1022864511700021e-04, 0.2626694912652480e-04),
        (2, 139, 20, 0.4504786663677753e-03, 0.1153263684274191e-02),
        (2, 172, 37, 0.5374660865070208e+00, 0.6888772330835508e+00),
        (2, 236, 14, 0.7036883080554905e+00, 0.9029380520049202e+00),
        (3, 1, 33, 0.2394280611347332e-03, 0.6141065100712753e-03),
        (3, 11, 44, 0.8900512402451899e-05, 0.2285607284989730e-04),
        (3, 139, 32, 0.2383246765643577e-03, 0.6117931675174485e-03),
        (3, 172, 52, 0.4814929601627414e+00, 0.6172822780395152e+00),
        (3, 236, 26, 0.4871630396135040e+00, 0.6536424087869508e+00),
    ];
    for (iprec, domain, n, x, w) in first {
        let d = &domains(Precision::ALL[iprec - 1])[domain - 1];
        assert_eq!(d.nodes.len(), n, "iprec {iprec}, domain {domain}");
        assert_eq!(d.nodes[0], x);
        assert_eq!(d.weights[0], w);
    }
    // The last node and weight of the same domains at every precision, from FMM3D's files
    // (`xs(n)`, `ws(n)`): (iprec, domain, n, xₙ, wₙ).
    let last = [
        (1, 1, 12, 0.8811933181071725e-01, 0.1207341200009158e-01),
        (1, 11, 26, 0.1054128253993683e+00, 0.9657468295394336e-02),
        (1, 139, 10, 0.7048956932878120e-01, 0.1189057898878958e-01),
        (1, 172, 22, 0.1496047036956045e+02, 0.1018366096767002e+01),
        (1, 236, 1, 0.3647057884873068e+01, 0.5389135443514403e+01),
        (2, 1, 21, 0.1586626589243695e+00, 0.1259332951418082e-01),
        (2, 11, 35, 0.1714661223874895e+00, 0.1051494208149056e-01),
        (2, 139, 20, 0.1514318804220985e+00, 0.1273146853251371e-01),
        (2, 172, 37, 0.2260996983132893e+02, 0.1265002467422740e+01),
        (2, 236, 14, 0.1332347618506859e+02, 0.1214099359631963e+01),
        (3, 1, 33, 0.2488977877058401e+00, 0.3727621470193036e-01),
        (3, 11, 44, 0.2362804496735852e+00, 0.1065007186065562e-01),
        (3, 139, 32, 0.2432755759628999e+00, 0.3639171774754243e-01),
        (3, 172, 52, 0.3088777266736086e+02, 0.1197570724553097e+01),
        (3, 236, 26, 0.2463562542321771e+02, 0.3852526260915091e+01),
    ];
    for (iprec, domain, n, x, w) in last {
        let d = &domains(Precision::ALL[iprec - 1])[domain - 1];
        assert_eq!(d.nodes.len(), n, "iprec {iprec}, domain {domain}");
        assert_eq!(d.nodes[n - 1], x);
        assert_eq!(d.weights[n - 1], w);
    }
}

#[test]
fn the_lookup() {
    for domain in 1..=DOMAINS {
        for kappa in nine_points(domain) {
            assert_eq!(domain_of(kappa), Some(domain), "κ = {kappa}");
        }
    }
    // Breakpoints belong to the rectangle above.
    assert_eq!(domain_of(C::new(RE_BREAKS[1], 0.0)), Some(2));
    assert_eq!(domain_of(C::new(0.0, IM_BREAKS[1])), Some(1 + NX));
    // Outside: Re κ ≥ 16π, Im κ ≥ 12π, negative or non-finite parts.
    for kappa in [
        C::new(16.0 * PI, 0.0),
        C::new(0.0, 12.0 * PI),
        C::new(100.0, 1.0),
        C::new(-1e-3, 0.0),
        C::new(1.0, -1e-3),
        C::new(f64::NAN, 0.0),
        C::new(f64::INFINITY, 0.0),
    ] {
        assert_eq!(domain_of(kappa), None, "κ = {kappa}");
        assert!(Quadrature::new(kappa, Precision::Iprec2).is_none());
    }
    assert_eq!(domain_of(C::new(16.0 * PI * (1.0 - 1e-15), 0.0)), Some(NX));
    assert_eq!(
        domain_of(C::new(0.0, 12.0 * PI * (1.0 - 1e-15))),
        Some(1 + (NY - 1) * NX)
    );
    // hwts3e's precision by tolerance.
    let cases = [
        (1e-1, Some(Precision::Iprec1)),
        (5e-3, Some(Precision::Iprec1)),
        (1e-3, Some(Precision::Iprec1)),
        (5e-4, Some(Precision::Iprec1)),
        (4.99e-4, Some(Precision::Iprec2)),
        (1e-6, Some(Precision::Iprec2)),
        (5e-7, Some(Precision::Iprec2)),
        (4.99e-7, Some(Precision::Iprec3)),
        (1e-8, Some(Precision::Iprec3)),
        (5e-10, Some(Precision::Iprec3)),
        (4.99e-10, None),
        (0.0, None),
        (-1e-6, None),
        (f64::NAN, None),
    ];
    for (eps, expected) in cases {
        assert_eq!(Precision::for_tolerance(eps), expected, "ε = {eps}");
    }
}

#[test]
fn the_node_maps() {
    for precision in Precision::ALL {
        for domain in FIXTURE_DOMAINS {
            let kappa = centre(domain);
            let q = Quadrature::new(kappa, precision).unwrap();
            let d = &domains(precision)[domain - 1];
            assert_eq!(q.domain(), domain);
            assert_eq!(q.nodes(), d.nodes.len());
            for k in 0..q.nodes() {
                let (l, t) = (q.lambda()[k], q.t()[k]);
                // t² = λ² + κ² by construction; type 1 has t = x real.
                assert!(
                    (t * t - (l * l + kappa * kappa)).norm() <= 1e-12 * (l * l).norm().max(1.0)
                );
                // The node maps restated (the code's formula, so a check of its
                // transcription only; the λ sum below checks the maps against the
                // kernel): type 1 has t = x real; type 2 runs from λ(u = 0) = −iκ along
                // u − iκ + i Re(κ) u/(1 + u).
                match d.map {
                    NodeMap::RealT => {
                        let x = d.nodes[k];
                        assert!((t - C::new(x, 0.0)).norm() <= 1e-13 * x.max(1.0));
                    }
                    NodeMap::Contour => {
                        let u = 100.0 * d.nodes[k];
                        let expected = u - C::i() * kappa + C::i() * (kappa.re * u / (1.0 + u));
                        assert!((l - expected).norm() <= 1e-15 * expected.norm().max(1.0));
                    }
                }
            }
            // The λ sum with α resolved against the kernel (S4 §3.1).
            let e = lambda_error(&q);
            assert!(
                e <= LAMBDA_BOUND * q.stated_error(),
                "λ sum, iprec {}, domain {domain}: {e:.3e} > {LAMBDA_BOUND} × {:.3e}",
                precision.iprec(),
                q.stated_error()
            );
        }
    }
}

/// |J_m(z)| against the trapezoidal integral J_m(z) = (1/M) Σⱼ e^{i(z sin αⱼ − m αⱼ)},
/// exact to round-off for M large, relative to e^{|Im z|} (the size of the integrand).
#[test]
fn bessel_of_complex_argument() {
    let zs = [
        C::new(0.3, 0.0),
        C::new(7.5, 0.0),
        C::new(120.0, 0.0),
        C::new(3.0, 2.0),
        C::new(30.0, -5.0),
        C::new(50.0, 20.0),
        C::new(0.0, 15.0),
        C::new(-40.0, 3.0),
        C::new(380.0, -40.0),
    ];
    for z in zs {
        let top = 2 * z.norm().ceil() as usize + 80;
        let j = bessel_j_norms(z, top);
        let m_nodes = 2 * (z.norm().ceil() as usize + top) + 400;
        let size = z.im.abs().exp();
        for m in (0..=top).step_by(7).chain([top]) {
            let exact: C = (0..m_nodes)
                .map(|l| {
                    let a = TAU * l as f64 / m_nodes as f64;
                    (C::i() * (z * a.sin() - m as f64 * a)).exp()
                })
                .sum::<C>()
                / m_nodes as f64;
            assert!(
                (j[m] - exact.norm()).abs() <= 1e-13 * size,
                "|J_{m}({z})| = {:e} vs {:e}",
                j[m],
                exact.norm()
            );
        }
    }
    assert_eq!(
        bessel_j_norms(C::new(0.0, 0.0), 3),
        vec![1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(bessel_j_norms(C::new(1e-200, 0.0), 2), vec![1.0, 0.0, 0.0]);
}

/// The tails the counts threshold on, relative: for m ≥ 2|z| against the power series
/// J_m(z) = Σⱼ (−1)ʲ (z/2)^{m+2j} / (j! (m+j)!) (its first term in logarithms, against
/// overflow of m!), within its own rounding: 10⁻¹⁴ of Σ |terms| (the series cancels while
/// its terms grow, by up to e^{|z|²/(4m)} ≤ e^{|z|/8} at m = 2|z|) plus 10⁻¹³ of the value.
/// The arguments stay at |z| ≤ 60, where that bound is informative; the trapezoid of
/// [`bessel_of_complex_argument`] covers |z| to 380.
#[test]
fn bessel_tails_relative() {
    let zs = [
        C::new(7.5, 0.0),
        C::new(3.0, 2.0),
        C::new(30.0, -5.0),
        C::new(50.0, 20.0),
        C::new(0.0, 15.0),
        C::new(60.0, 0.5),
    ];
    let ln_factorial = |n: usize| (1..=n).map(|i| (i as f64).ln()).sum::<f64>();
    for z in zs {
        let first = (2.0 * z.norm()).ceil() as usize;
        let top = first + 60;
        let j = bessel_j_norms(z, top);
        let half = z * 0.5;
        for (m, &value) in j.iter().enumerate().skip(first) {
            // The terms relative to the first: (−(z/2)²)ʲ m! / (j! (m+j)!).
            let (mut term, mut sum, mut size) = (C::new(1.0, 0.0), C::new(1.0, 0.0), 1.0f64);
            for i in 1..400 {
                term *= -(half * half) / (i as f64 * (m + i) as f64);
                sum += term;
                size += term.norm();
                if term.norm() < 1e-18 * size {
                    break;
                }
            }
            let first_term = (m as f64 * half.norm().ln() - ln_factorial(m)).exp();
            let exact = first_term * sum.norm();
            if exact < 1e-300 {
                continue;
            }
            assert!(
                (value - exact).abs() <= 1e-14 * first_term * size + 1e-13 * exact,
                "|J_{m}({z})| = {value:e} vs {exact:e}"
            );
        }
    }
}

/// The acceptance in debug builds: the kernel at the centres of the fixture's domains at
/// every precision, and the chain at their centres at iprec 2 (each check costs about a
/// second in debug; the ignored [`the_counts_on_every_domain`] runs the rest).
#[test]
fn the_counts_on_sampled_domains() {
    for precision in Precision::ALL {
        for domain in FIXTURE_DOMAINS {
            let chain = (precision == Precision::Iprec2).then_some(4);
            check_counts(centre(domain), precision, chain);
        }
    }
}

/// The reference levels of S4 §5 (root side 1, κ_l = kD / 2^l, levels 2 to 8, real,
/// Yukawa and damped kD) at the precisions of ε = 10⁻³, 10⁻⁶, 10⁻⁸, where they fall in the
/// tables: the kernel and the chain.
#[test]
#[ignore = "release: the reference levels"]
fn the_counts_on_the_reference_levels() {
    let cases = [
        C::new(10.0, 0.0),
        C::new(100.0, 0.0),
        C::new(500.0, 0.0),
        C::new(0.0, 10.0),
        C::new(0.0, 100.0),
        C::new(100.0, 5.0),
        C::new(500.0, 25.0),
    ];
    for kd in cases {
        for level in 2..=8 {
            let kappa = kd / f64::powi(2.0, level);
            if domain_of(kappa).is_none() {
                continue;
            }
            for eps in [1e-3, 1e-6, 1e-8] {
                check_counts(kappa, Precision::for_tolerance(eps).unwrap(), Some(4));
            }
        }
    }
}

/// The acceptance in full: the kernel at the nine points of every domain and precision,
/// and the chain at every domain's centre (the chain costs about 10⁸ operations per
/// quadrature, so nine points of every domain would take hours on one thread; S4 §3
/// measured its chain at the centres too). About 6 minutes on the M6 (380 s with the
/// reference levels, beside other work).
#[test]
#[ignore = "release: 437 domains × 3 precisions, about 6 minutes"]
fn the_counts_on_every_domain() {
    let mut worst = [(0.0f64, 0.0f64); 3];
    for precision in Precision::ALL {
        for domain in 1..=DOMAINS {
            let w = &mut worst[precision.iprec() - 1];
            for kappa in nine_points(domain) {
                let q = Quadrature::new(kappa, precision).unwrap();
                let kernel = kernel_error(&q, q.alpha_counts()) / q.stated_error();
                w.0 = w.0.max(kernel);
                assert!(
                    kernel <= KERNEL_BOUND,
                    "κ = {kappa}, iprec {}, domain {domain}: kernel {kernel:.2} × the stated \
                     error",
                    precision.iprec()
                );
            }
            let q = Quadrature::new(centre(domain), precision).unwrap();
            let chain = chain_error(&q, 4) / q.stated_error();
            w.1 = w.1.max(chain);
            assert!(
                chain <= CHAIN_BOUND,
                "centre of domain {domain}, iprec {}: chain {chain:.2} × the stated error",
                precision.iprec()
            );
        }
    }
    for (p, (kernel, chain)) in worst.iter().enumerate() {
        println!(
            "iprec {}: kernel {kernel:.3} and chain {chain:.3} × the stated error",
            p + 1
        );
    }
}

/// FMM3D's reach (S4 §3.2), the smaller one CH6.2e measures beside the corner's.
const FMM3D_REACH: f64 = 4.0;

/// The counts for a smaller reach (Helmholtz CH6.2e): the corner's reach is `new` bit for
/// bit, the waves do not grow as the reach shrinks and the Fourier counts stay, and at
/// [`FMM3D_REACH`] the kernel on the grid points with ρ ≤ 4 is within [`KERNEL_BOUND`] of
/// the stated error, at the centres of the fixture's domains at every precision.
#[test]
fn the_counts_for_a_smaller_reach() {
    for precision in Precision::ALL {
        for domain in FIXTURE_DOMAINS {
            let kappa = centre(domain);
            let corner = Quadrature::new(kappa, precision).unwrap();
            assert_eq!(
                Quadrature::with_reach(kappa, precision, RHO_MAX).as_ref(),
                Some(&corner)
            );
            assert_eq!(corner.reach(), RHO_MAX);
            let mut previous = corner.clone();
            for reach in [5.0, FMM3D_REACH, 3.0] {
                let q = Quadrature::with_reach(kappa, precision, reach).unwrap();
                assert_eq!(q.reach(), reach);
                assert_eq!(q.fourier_counts(), corner.fourier_counts());
                assert_eq!(q.lambda(), corner.lambda());
                assert_eq!(q.weights(), corner.weights());
                // The maximum over ρ is sampled at 64 equal steps of [0, reach], so one
                // node's count may exceed a larger reach's by a sample's worth (one).
                for (n, m) in q.alpha_counts().iter().zip(previous.alpha_counts()) {
                    assert!(*n <= m + 1, "domain {domain}, reach {reach}: {n} > {m} + 1");
                }
                assert!(
                    q.wave_count() <= previous.wave_count(),
                    "domain {domain}, reach {reach}: {} > {} waves",
                    q.wave_count(),
                    previous.wave_count()
                );
                previous = q;
            }
            let q = Quadrature::with_reach(kappa, precision, FMM3D_REACH).unwrap();
            let e = q.stated_error();
            let [within, _] = kernel_errors(&q, q.alpha_counts(), FMM3D_REACH);
            assert!(
                within <= KERNEL_BOUND * e,
                "kernel at ρ ≤ 4, domain {domain}, iprec {}: {within:.3e} > {KERNEL_BOUND} × \
                 {e:.3e}",
                precision.iprec()
            );
        }
    }
}

#[test]
#[should_panic(expected = "the reach must be finite and above")]
fn a_reach_within_the_source_box_panics() {
    let _ = Quadrature::with_reach(C::new(1.0, 0.0), Precision::Iprec2, 0.5);
}

/// The counts at [`FMM3D_REACH`] on every domain (Helmholtz CH6.2e): the kernel at ρ ≤ 4
/// within [`KERNEL_BOUND`] at the nine points of every domain and precision; printed per
/// precision, the waves Σₖ Nₖ over the corner's (smallest, largest, and over the domains with
/// Im κ < π/2, where FMM3D's sit at 1/1.51 to 1/1.31 of the corner's, CH6.1a), the kernel's
/// error on the whole region (the corner's) over the stated error, and the point-to-point
/// chain at every domain's centre within ρ ≤ 4 and on the whole region. About 7 minutes on
/// the M6.
#[test]
#[ignore = "release: 437 domains × 3 precisions, about 7 minutes"]
fn the_counts_for_a_smaller_reach_on_every_domain() {
    let lower = |domain: usize| rectangle(domain).0;
    for precision in Precision::ALL {
        let mut ratio = [f64::INFINITY, 0.0];
        let mut ratio_type2 = [f64::INFINITY, 0.0];
        let mut kernel = [0.0f64; 2];
        let mut chain = [0.0f64; 2];
        let mut worst_corner = (0.0f64, 0usize);
        let mut largest = 0;
        for domain in 1..=DOMAINS {
            for kappa in nine_points(domain) {
                let q = Quadrature::with_reach(kappa, precision, FMM3D_REACH).unwrap();
                let e = q.stated_error();
                let [within, all] = kernel_errors(&q, q.alpha_counts(), FMM3D_REACH);
                assert!(
                    within <= KERNEL_BOUND * e,
                    "κ = {kappa}, iprec {}, domain {domain}: kernel at ρ ≤ 4 {:.2} × the \
                     stated error",
                    precision.iprec(),
                    within / e
                );
                kernel[0] = kernel[0].max(within / e);
                kernel[1] = kernel[1].max(all / e);
                if all / e > worst_corner.0 {
                    worst_corner = (all / e, domain);
                }
            }
            let kappa = centre(domain);
            let corner = Quadrature::new(kappa, precision).unwrap();
            let q = Quadrature::with_reach(kappa, precision, FMM3D_REACH).unwrap();
            let r = q.wave_count() as f64 / corner.wave_count() as f64;
            if r > ratio[1] {
                largest = domain;
            }
            ratio = [ratio[0].min(r), ratio[1].max(r)];
            if lower(domain).im < PI / 2.0 {
                ratio_type2 = [ratio_type2[0].min(r), ratio_type2[1].max(r)];
            }
            let [within, all] = chain_errors(&q, 4, FMM3D_REACH);
            chain[0] = chain[0].max(within / q.stated_error());
            chain[1] = chain[1].max(all / q.stated_error());
        }
        println!(
            "iprec {}: waves at ρ_max = 4 over the corner's {:.3}–{:.3} (the largest on \
             domain {largest}; Im κ < π/2: {:.3}–{:.3}); kernel at ρ ≤ 4 {:.3}, on the region {:.3} × the stated error \
             (worst domain {}); chain at ρ ≤ 4 {:.3}, on the region {:.3}",
            precision.iprec(),
            ratio[0],
            ratio[1],
            ratio_type2[0],
            ratio_type2[1],
            kernel[0],
            kernel[1],
            worst_corner.1,
            chain[0],
            chain[1],
        );
    }
}

#[test]
fn deterministic() {
    for precision in Precision::ALL {
        let kappa = C::new(12.3, 0.7);
        assert_eq!(
            Quadrature::new(kappa, precision),
            Quadrature::new(kappa, precision)
        );
    }
}
