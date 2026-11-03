//! The quadratures of the Helmholtz exponential M2L per level (Helmholtz CH6.1a; the
//! [module documentation](super) has the representation): FMM3D's tables, their lookup by
//! κ = k w and a tolerance, the node maps, and the α and Fourier counts.
//!
//! # FMM3D's tables
//!
//! FMM3D tabulates these quadratures; it does not generate them
//! (`spikes/pw-quadrature/SPIKE_REPORT.md` §2). Its `hwts3e` takes κ, finds its **domain**,
//! one of 23 × 19 rectangles of κ (Re κ on [`RE_BREAKS`] up to 16π, Im κ on [`IM_BREAKS`]
//! up to 12π; [`domain_of`]), and reads one quadrature per domain and **precision**
//! (`iprec`, from the tolerance; [`Precision`]), valid uniformly over the rectangle: s real
//! nodes xₖ and weights wₖ, a quadrature type and the error the file states, the largest
//! |Q − G| over the region in box widths. This module holds iprec 1, 2 and 3 (D1's
//! tolerances 10⁻³ to 10⁻⁸; S4 §6 and its sign-off 1), NIST's, extracted from FMM3D v2.1.0
//! (`src/Helmholtz/hwts_3d_iprec{1,2,3}.txt`) by
//! `tools/fixtures/helmholtz_pw_quadrature.sh` into `hwts_iprec{1,2,3}.txt` beside this
//! file, which keep the files' notice.
//!
//! The nodes are mapped to λ by the type at κ ([`Quadrature::new`], as `hwts3e`):
//!
//! - type 1 (the domains with Im κ ≥ π/2), real t: λₖ = √(xₖ² − κ²), weight wₖ xₖ / λₖ;
//! - type 2 (Im κ < π/2), a contour in λ from −iκ: with u = 100 xₖ and a = Re κ,
//!   λₖ = u − iκ + i a u/(1 + u), weight 100 wₖ (1 + i a/(1 + u)²);
//!
//! and tₖ = √(λₖ² + κ²) (principal roots throughout).
//!
//! # The counts
//!
//! FMM3D's own α counts (`hnumphys.f`) miss the stated error at the region's corner,
//! ρ = 4√2, by up to 119× (S4 §3.2), so they are not used: the counts here are S4's bound
//! (`spikes/pw-quadrature/src/generator.rs`, `tail_count`) for complex tₖ. The trapezoidal
//! rule of Nₖ points in α errs for J₀(tₖ ρ) by 2 Σ_{l≥1} i^{lN} J_{lN}(tₖ ρ) cos(lNφ), so with
//! the node's size cₖ = 2 |wₖ| e^{−Re λₖ} (at z = 1, the region's nearest plane) and the
//! share e/(4s) of the stated error e per node:
//!
//! - Fₖ is the fewest orders with cₖ max_{m > Fₖ} max_{ρ ≤ r_s} |J_m(tₖ ρ)| ≤ e/(4s), r_s = √2/2
//!   the lateral reach of a point from its box's centre, so the source's waves keep the
//!   orders |m| ≤ Fₖ;
//! - Nₖ is the fewest with cₖ max_{m ≥ Nₖ} max_{ρ ≤ ρ_max} |J_m(tₖ ρ)| ≤ e/(4s), ρ_max = 4√2,
//!   and at least the count for ρ_max − r_s plus Fₖ (the targets' reach from the source
//!   box's centre), so that the truncated orders do not alias.
//!
//! The maxima over ρ are taken at 64 equal steps of ρ, and over all orders at or above the
//! count, since below the turning point m ≈ |tₖ| ρ one |J_m| can vanish at a zero while the
//! next order does not (S4 §4.1). |J_m(z)| of complex z comes from Miller's backward
//! recurrence normalised by e^{iz} = J₀(z) + 2 Σₙ iⁿ Jₙ(z), taken at the conjugate with
//! Im z ≤ 0, where |e^{iz}| is as large as the largest term, so the sum does not cancel
//! (|J_m(z̄)| = |J_m(z)|). Everything is a function of κ and the precision only (proposal
//! §2, constraint 9), built in f64.
//!
//! `tests/helmholtz/plane_wave_quadrature.rs` checks the result on the region against the
//! kernel e^{iκ|u|}/|u| (the trusted path: the quadrature's own integrand, exactly), with the
//! counts within 2× the stated error and the point-to-point chain within 3× it; the
//! wave counts against FMM3D's are in docs/helmholtz/tasks/CH6.1a.md.

use std::sync::LazyLock;

use nd_fmm_math::Complex;

/// The complex scalar of the quadratures.
type C = Complex<f64>;

/// The breakpoints of Re κ (`hwts3dgetd`'s `cx`, 24 values, 23 intervals): 0, 0.02, 0.04,
/// 0.1, 0.2, 0.4, 1, π/2, then π to 16π in steps of π.
pub const RE_BREAKS: [f64; 24] = {
    let pi = core::f64::consts::PI;
    [
        0.0,
        0.02,
        0.04,
        0.10,
        0.20,
        0.40,
        1.00,
        pi / 2.0,
        pi,
        2.0 * pi,
        3.0 * pi,
        4.0 * pi,
        5.0 * pi,
        6.0 * pi,
        7.0 * pi,
        8.0 * pi,
        9.0 * pi,
        10.0 * pi,
        11.0 * pi,
        12.0 * pi,
        13.0 * pi,
        14.0 * pi,
        15.0 * pi,
        16.0 * pi,
    ]
};

/// The breakpoints of Im κ (`hwts3dgetd`'s `cy`, the first 20 of [`RE_BREAKS`], 19
/// intervals, to 12π).
pub const IM_BREAKS: [f64; 20] = {
    let mut b = [0.0; 20];
    let mut i = 0;
    while i < 20 {
        b[i] = RE_BREAKS[i];
        i += 1;
    }
    b
};

/// Domains along Re κ.
pub const NX: usize = 23;

/// Domains along Im κ.
pub const NY: usize = 19;

/// Domains per precision, [`NX`] × [`NY`].
pub const DOMAINS: usize = NX * NY;

/// The lateral reach of a point from its box's centre, r_s = √2/2 box widths.
pub const SOURCE_REACH: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// The region's largest lateral offset, ρ_max = 4√2 box widths.
pub const RHO_MAX: f64 = 4.0 * core::f64::consts::SQRT_2;

/// One of FMM3D's precisions held here (`iprec` 1, 2, 3; `hwts3e`'s comments: 3, 6 and 9
/// digits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    /// iprec 1: stated errors up to 10⁻⁴.
    Iprec1,
    /// iprec 2: stated errors up to 10⁻⁷.
    Iprec2,
    /// iprec 3: stated errors up to 10⁻¹⁰.
    Iprec3,
}

impl Precision {
    /// The precisions held, coarsest first.
    pub const ALL: [Precision; 3] = [Precision::Iprec1, Precision::Iprec2, Precision::Iprec3];

    /// `hwts3e`'s precision for a tolerance ε: iprec 1 for ε < 5·10⁻³, 2 below 5·10⁻⁴, 3
    /// below 5·10⁻⁷. Above (FMM3D's iprec 0) the coarsest held, iprec 1; below 5·10⁻¹⁰
    /// (iprec 4, not held) or for a ε that is not positive and finite, `None`.
    pub fn for_tolerance(eps: f64) -> Option<Self> {
        if !(eps > 0.0 && eps.is_finite()) {
            None
        } else if eps >= 0.5e-3 {
            Some(Precision::Iprec1)
        } else if eps >= 0.5e-6 {
            Some(Precision::Iprec2)
        } else if eps >= 0.5e-9 {
            Some(Precision::Iprec3)
        } else {
            None
        }
    }

    /// FMM3D's `iprec`, 1 to 3.
    pub fn iprec(self) -> usize {
        match self {
            Precision::Iprec1 => 1,
            Precision::Iprec2 => 2,
            Precision::Iprec3 => 3,
        }
    }

    /// The position in [`Precision::ALL`].
    fn index(self) -> usize {
        self.iprec() - 1
    }
}

/// `hwts3dgetd`: the domain (1-based, `ix + (iy − 1)` [`NX`], FMM3D's numbering) of κ, or
/// `None` outside the tables (Re κ ≥ 16π or Im κ ≥ 12π) and for a negative or non-finite
/// part (`hwts3dgetd` maps a negative part to the first row or column; here it is refused).
/// A κ on a breakpoint belongs to the rectangle above it.
pub fn domain_of(kappa: Complex<f64>) -> Option<usize> {
    let valid = |v: f64| v >= 0.0 && v.is_finite();
    if !(valid(kappa.re) && valid(kappa.im)) {
        return None;
    }
    let ix = RE_BREAKS.partition_point(|&b| b <= kappa.re);
    let iy = IM_BREAKS.partition_point(|&b| b <= kappa.im);
    (ix <= NX && iy <= NY).then_some(ix + (iy - 1) * NX)
}

/// The rectangle of a domain (1-based): its lower and upper corners in κ.
///
/// # Panics
///
/// If `domain` is not in 1..=[`DOMAINS`].
pub fn rectangle(domain: usize) -> (Complex<f64>, Complex<f64>) {
    assert!(
        (1..=DOMAINS).contains(&domain),
        "domain {domain} outside 1..={DOMAINS}"
    );
    let ix = (domain - 1) % NX;
    let iy = (domain - 1) / NX;
    (
        C::new(RE_BREAKS[ix], IM_BREAKS[iy]),
        C::new(RE_BREAKS[ix + 1], IM_BREAKS[iy + 1]),
    )
}

/// The node map of a domain's quadrature (`hwts3e`'s `iquadtype`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeMap {
    /// Type 1: real t, λ = √(x² − κ²).
    RealT,
    /// Type 2: the contour λ = u − iκ + i Re(κ) u/(1 + u), u = 100 x.
    Contour,
}

/// One tabulated quadrature of FMM3D: a domain at one precision, as its file states it.
#[derive(Clone, Debug)]
pub struct Domain {
    /// The domain, 1-based.
    pub index: usize,
    /// The node map.
    pub map: NodeMap,
    /// The rectangle's lower corner as the file states it.
    pub lower: Complex<f64>,
    /// The rectangle's upper corner as the file states it.
    pub upper: Complex<f64>,
    /// The error the file states (`err`): the largest |Q − G| over the region, box widths.
    pub stated_error: f64,
    /// The real nodes xₖ.
    pub nodes: Vec<f64>,
    /// The real weights wₖ.
    pub weights: Vec<f64>,
}

impl Domain {
    /// λₖ and the weight of node k at κ, by the node map (the module documentation).
    fn map_node(&self, k: usize, kappa: C) -> (C, C) {
        let (x, w) = (self.nodes[k], self.weights[k]);
        match self.map {
            NodeMap::RealT => {
                let lambda = (C::new(x * x, 0.0) - kappa * kappa).sqrt();
                (lambda, w * x / lambda)
            }
            NodeMap::Contour => {
                let i = C::i();
                let u = 100.0 * x;
                let a = kappa.re.abs();
                let lambda = u - i * kappa + i * (a * u / (1.0 + u));
                let weight = (C::new(1.0, 0.0) + i * (a / ((1.0 + u) * (1.0 + u)))) * (100.0 * w);
                (lambda, weight)
            }
        }
    }
}

/// FMM3D's quadratures at a precision: all [`DOMAINS`] domains, by index − 1.
pub fn domains(precision: Precision) -> &'static [Domain] {
    &TABLES[precision.index()]
}

/// The three precisions' tables, parsed on first use.
static TABLES: LazyLock<[Vec<Domain>; 3]> = LazyLock::new(|| {
    [
        parse(include_str!("hwts_iprec1.txt")),
        parse(include_str!("hwts_iprec2.txt")),
        parse(include_str!("hwts_iprec3.txt")),
    ]
});

/// Parses one of `hwts_iprec{1,2,3}.txt` (their header gives the format).
///
/// # Panics
///
/// On a malformed file: they are generated and checked in, so this is a defect.
fn parse(text: &str) -> Vec<Domain> {
    let number = |s: &str| -> f64 { s.parse().unwrap_or_else(|e| panic!("{s:?}: {e}")) };
    let mut lines = text.lines().filter(|l| !l.starts_with('#'));
    let mut out: Vec<Domain> = Vec::with_capacity(DOMAINS);
    while let Some(head) = lines.next() {
        let f: Vec<&str> = head.split_whitespace().collect();
        assert!(f.len() == 9 && f[0] == "domain", "bad header {head:?}");
        let index: usize = f[1].parse().expect("a domain index");
        assert_eq!(index, out.len() + 1, "domains out of order");
        let map = match f[2] {
            "1" => NodeMap::RealT,
            "2" => NodeMap::Contour,
            t => panic!("quadrature type {t} in domain {index}"),
        };
        let n: usize = f[8].parse().expect("a node count");
        let (mut nodes, mut weights) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let line = lines.next().expect("a node line");
            let (x, w) = line.split_once(' ').expect("a node and a weight");
            nodes.push(number(x));
            weights.push(number(w));
        }
        out.push(Domain {
            index,
            map,
            lower: C::new(number(f[3]), number(f[4])),
            upper: C::new(number(f[5]), number(f[6])),
            stated_error: number(f[7]),
            nodes,
            weights,
        });
    }
    assert_eq!(out.len(), DOMAINS, "domain count");
    out
}

/// The quadrature of one level: FMM3D's nodes of the domain of κ = k w at a precision,
/// mapped to λ at κ, with the counts of the module documentation.
#[derive(Clone, Debug, PartialEq)]
pub struct Quadrature {
    kappa: C,
    precision: Precision,
    domain: usize,
    stated_error: f64,
    lambda: Vec<C>,
    t: Vec<C>,
    weight: Vec<C>,
    alpha: Vec<usize>,
    fourier: Vec<usize>,
}

impl Quadrature {
    /// The quadrature for κ = k w (the level's wavenumber in box widths) at a precision,
    /// or `None` where κ lies outside the tables ([`domain_of`]).
    pub fn new(kappa: Complex<f64>, precision: Precision) -> Option<Self> {
        let domain = domain_of(kappa)?;
        let table = &domains(precision)[domain - 1];
        let s = table.nodes.len();
        let (mut lambda, mut weight, mut t) = (
            Vec::with_capacity(s),
            Vec::with_capacity(s),
            Vec::with_capacity(s),
        );
        for k in 0..s {
            let (l, w) = table.map_node(k, kappa);
            lambda.push(l);
            weight.push(w);
            t.push((l * l + kappa * kappa).sqrt());
        }
        let share = table.stated_error / (4.0 * s as f64);
        let (mut alpha, mut fourier) = (Vec::with_capacity(s), Vec::with_capacity(s));
        for k in 0..s {
            let scale = 2.0 * weight[k].norm() * (-lambda[k].re).exp();
            let f = tail_count(t[k], SOURCE_REACH, scale, share) - 1;
            let n = tail_count(t[k], RHO_MAX, scale, share)
                .max(tail_count(t[k], RHO_MAX - SOURCE_REACH, scale, share) + f);
            alpha.push(n);
            fourier.push(f);
        }
        Some(Self {
            kappa,
            precision,
            domain,
            stated_error: table.stated_error,
            lambda,
            t,
            weight,
            alpha,
            fourier,
        })
    }

    /// κ = k w.
    pub fn kappa(&self) -> Complex<f64> {
        self.kappa
    }

    /// The precision.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The domain of κ, 1-based.
    pub fn domain(&self) -> usize {
        self.domain
    }

    /// The error FMM3D's file states for the domain's λ quadrature.
    pub fn stated_error(&self) -> f64 {
        self.stated_error
    }

    /// The number of nodes, s.
    pub fn nodes(&self) -> usize {
        self.lambda.len()
    }

    /// The decay rates λₖ.
    pub fn lambda(&self) -> &[Complex<f64>] {
        &self.lambda
    }

    /// The transverse wavenumbers tₖ = √(λₖ² + κ²).
    pub fn t(&self) -> &[Complex<f64>] {
        &self.t
    }

    /// The weights wₖ (of dλ).
    pub fn weights(&self) -> &[Complex<f64>] {
        &self.weight
    }

    /// The α counts Nₖ.
    pub fn alpha_counts(&self) -> &[usize] {
        &self.alpha
    }

    /// The Fourier counts Fₖ: a source's orders |m| ≤ Fₖ are kept at λₖ.
    pub fn fourier_counts(&self) -> &[usize] {
        &self.fourier
    }

    /// The complex wave values per direction, Σₖ Nₖ (FMM3D's `nexptotp`).
    pub fn wave_count(&self) -> usize {
        self.alpha.iter().sum()
    }

    /// The Fourier coefficients per direction, Σₖ (2Fₖ + 1) (FMM3D's `nexptot`).
    pub fn mode_count(&self) -> usize {
        self.fourier.iter().map(|&f| 2 * f + 1).sum()
    }
}

/// The fewest n ≥ 1 with `scale` max_{m ≥ n} max_{ρ ≤ reach} |J_m(t ρ)| ≤ `share`, the
/// maximum over ρ at 64 equal steps (the module documentation, "The counts").
fn tail_count(t: C, reach: f64, scale: f64, share: f64) -> usize {
    let x = t.norm() * reach;
    let top = (x.ceil() as usize) * 2 + 80;
    let mut worst = vec![0.0f64; top + 1];
    for i in 1..=64 {
        let j = bessel_j_norms(t * (reach * i as f64 / 64.0), top);
        for (w, v) in worst.iter_mut().zip(&j) {
            *w = w.max(*v);
        }
    }
    for m in (0..top).rev() {
        worst[m] = worst[m].max(worst[m + 1]);
    }
    (1..top).find(|&n| scale * worst[n] <= share).unwrap_or(top)
}

/// |J₀(z)|, …, |J_top(z)| for complex z: Miller's backward recurrence
/// J_{n−1} = (2n/z) Jₙ − J_{n+1} from an order well above max(top, |z|), normalised by
/// e^{iz} = J₀(z) + 2 Σ_{n≥1} iⁿ Jₙ(z) at z or its conjugate, whichever has Im ≤ 0 (the
/// module documentation, "The counts"), rescaled against overflow.
///
/// Public for the tests. Its range is the counts': |z| up to a few hundred and |Im z| up to
/// about 100 (the nodes give |tₖ| ρ_max ≤ 320 and |Im tₖ| ρ_max ≤ 6). Beyond, the
/// unnormalised sum or e^{iz} itself can overflow (|Im z| above about 700 for the latter),
/// and the result is then infinite or NaN. Below |z| = 10⁻¹⁵⁰ it returns J₀ = 1 and zeros,
/// within (|z|/2)^m/m! of the values.
pub fn bessel_j_norms(z: Complex<f64>, top: usize) -> Vec<f64> {
    let mut out = vec![0.0; top + 1];
    let r = z.norm();
    if r < 1e-150 {
        out[0] = 1.0;
        return out;
    }
    let z = if z.im > 0.0 { z.conj() } else { z };
    let start = top.max(r.ceil() as usize) + 20 + (40.0 * r.max(1.0)).sqrt().ceil() as usize;
    let mut values = vec![C::new(0.0, 0.0); top + 1];
    let (mut above, mut cur) = (C::new(0.0, 0.0), C::new(1e-300, 0.0));
    // Σ iⁿ Jₙ over n ≥ 1, unnormalised: iⁿ cycles 1, i, −1, −i from n = 0.
    let powers = [
        C::new(1.0, 0.0),
        C::new(0.0, 1.0),
        C::new(-1.0, 0.0),
        C::new(0.0, -1.0),
    ];
    let mut sum = C::new(0.0, 0.0);
    let inv = z.inv();
    for n in (1..=start).rev() {
        // cur = Jₙ, above = J_{n+1} (unnormalised).
        let below = inv * (2.0 * n as f64) * cur - above;
        if n <= top {
            values[n] = cur;
        }
        sum += powers[n % 4] * cur;
        above = cur;
        cur = below;
        if cur.norm() > 1e250 {
            let s = 1e-250;
            above *= s;
            cur *= s;
            sum *= s;
            for v in values.iter_mut() {
                *v *= s;
            }
        }
    }
    values[0] = cur;
    let total = cur + sum * 2.0;
    // e^{iz} / total by its modulus and phase: the unnormalised total may be as small as
    // 10⁻¹⁶⁵, whose |·|² (which a complex quotient forms) underflows.
    let size = total.norm();
    let norm = (C::i() * z).exp() * (total / size).conj() / size;
    for (o, v) in out.iter_mut().zip(&values) {
        *o = (v * norm).norm();
    }
    out
}
